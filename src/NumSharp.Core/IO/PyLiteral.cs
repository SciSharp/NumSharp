using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace NumSharp.IO
{
    /// <summary>
    ///     A Python tuple. Distinct from <see cref="List{T}"/> because the .npy header validator has to
    ///     tell <c>(3,)</c> from <c>[3]</c> — NumPy rejects a header whose <c>shape</c> is not a tuple.
    /// </summary>
    internal sealed class PyTuple : IReadOnlyList<object>
    {
        private readonly List<object> _items;

        public PyTuple(List<object> items) => _items = items;

        public object this[int index] => _items[index];
        public int Count => _items.Count;
        public IEnumerator<object> GetEnumerator() => _items.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => _items.GetEnumerator();
    }

    /// <summary>
    ///     Parses the Python literal subset that appears in a .npy header, standing in for NumPy's
    ///     <c>ast.literal_eval</c> (<c>numpy/lib/_format_impl.py</c>).
    /// </summary>
    /// <remarks>
    ///     Only the literal grammar is accepted — dict, list, tuple, str, int, float, complex-free
    ///     numerics, <c>True</c>/<c>False</c>/<c>None</c>. There is no name lookup, no operators and no
    ///     calls, so (like <c>literal_eval</c>) parsing a hostile header cannot execute anything. The
    ///     size guard that protects against pathological input lives in the caller
    ///     (<see cref="NpyFormat.ReadArrayHeader"/>), matching NumPy's <c>max_header_size</c> check.
    ///
    ///     Python values map to CLR as: str→<see cref="string"/>, int→<see cref="long"/>,
    ///     float→<see cref="double"/>, bool→<see cref="bool"/>, None→<c>null</c>, tuple→<see cref="PyTuple"/>,
    ///     list→<see cref="List{Object}"/>, dict→<see cref="Dictionary{String,Object}"/>.
    /// </remarks>
    internal static class PyLiteral
    {
        /// <summary>Thrown when the input is not a well-formed Python literal (≙ Python's SyntaxError).</summary>
        internal sealed class SyntaxException : Exception
        {
            public SyntaxException(string message) : base(message) { }
        }

        /// <summary>
        ///     Parse a complete Python literal expression. Trailing whitespace is ignored; any other
        ///     trailing content is an error.
        /// </summary>
        /// <param name="s">The literal source.</param>
        /// <param name="allowLongSuffix">
        ///     Accept the Python 2 <c>L</c> integer suffix (<c>3L</c>). NumPy only retries with its
        ///     <c>_filter_header</c> for format versions ≤ (2, 0), so v3.0 headers reject it.
        /// </param>
        public static object Parse(string s, bool allowLongSuffix = false)
        {
            var p = new Parser(s, allowLongSuffix);
            object v = p.ParseValue();
            p.SkipWhitespace();
            if (!p.AtEnd)
                throw new SyntaxException($"unexpected trailing content at position {p.Position}");
            return v;
        }

        /// <summary>
        ///     Render a parsed value the way Python's <c>repr()</c> would. NumPy interpolates
        ///     <c>repr()</c> output into its header error messages, so error text matches verbatim.
        /// </summary>
        public static string Repr(object v)
        {
            var sb = new StringBuilder();
            Repr(v, sb);
            return sb.ToString();
        }

        private static void Repr(object v, StringBuilder sb)
        {
            switch (v)
            {
                case null:
                    sb.Append("None");
                    return;
                case bool b:
                    sb.Append(b ? "True" : "False");
                    return;
                case string s:
                    ReprString(s, sb);
                    return;
                case long l:
                    sb.Append(l.ToString(CultureInfo.InvariantCulture));
                    return;
                case double d:
                    ReprDouble(d, sb);
                    return;
                case PyTuple t:
                    sb.Append('(');
                    for (int i = 0; i < t.Count; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        Repr(t[i], sb);
                    }
                    if (t.Count == 1) sb.Append(','); // Python's 1-tuple repr: (3,)
                    sb.Append(')');
                    return;
                case List<object> list:
                    sb.Append('[');
                    for (int i = 0; i < list.Count; i++)
                    {
                        if (i > 0) sb.Append(", ");
                        Repr(list[i], sb);
                    }
                    sb.Append(']');
                    return;
                case Dictionary<string, object> dict:
                    sb.Append('{');
                    bool first = true;
                    foreach (var kv in dict)
                    {
                        if (!first) sb.Append(", ");
                        first = false;
                        ReprString(kv.Key, sb);
                        sb.Append(": ");
                        Repr(kv.Value, sb);
                    }
                    sb.Append('}');
                    return;
                default:
                    sb.Append(v);
                    return;
            }
        }

        // Python prefers single quotes, switching to double only when the value contains a single
        // quote but no double quote.
        private static void ReprString(string s, StringBuilder sb)
        {
            char quote = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
            sb.Append(quote);
            foreach (char c in s)
            {
                switch (c)
                {
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c == quote) sb.Append('\\');
                        sb.Append(c);
                        break;
                }
            }
            sb.Append(quote);
        }

        private static void ReprDouble(double d, StringBuilder sb)
        {
            if (double.IsNaN(d)) { sb.Append("nan"); return; }
            if (double.IsPositiveInfinity(d)) { sb.Append("inf"); return; }
            if (double.IsNegativeInfinity(d)) { sb.Append("-inf"); return; }

            // "R" is shortest-round-trip, which is what Python's float repr produces. Python always
            // shows a decimal point or exponent so the value reads as a float; .NET does not.
            string s = d.ToString("R", CultureInfo.InvariantCulture);
            if (s.IndexOf('.') < 0 && s.IndexOf('e') < 0 && s.IndexOf('E') < 0 &&
                s.IndexOf("Inf", StringComparison.Ordinal) < 0 && s.IndexOf("NaN", StringComparison.Ordinal) < 0)
                s += ".0";
            sb.Append(s);
        }

        private struct Parser
        {
            private readonly string _s;
            private readonly bool _allowLongSuffix;
            private int _i;

            public Parser(string s, bool allowLongSuffix)
            {
                _s = s;
                _allowLongSuffix = allowLongSuffix;
                _i = 0;
            }

            public bool AtEnd => _i >= _s.Length;
            public int Position => _i;

            public void SkipWhitespace()
            {
                while (_i < _s.Length && (_s[_i] == ' ' || _s[_i] == '\t' || _s[_i] == '\n' || _s[_i] == '\r'))
                    _i++;
            }

            private char Peek()
            {
                if (_i >= _s.Length)
                    throw new SyntaxException("unexpected EOF while parsing");
                return _s[_i];
            }

            private void Expect(char c)
            {
                SkipWhitespace();
                if (_i >= _s.Length || _s[_i] != c)
                    throw new SyntaxException($"expected '{c}' at position {_i}");
                _i++;
            }

            public object ParseValue()
            {
                SkipWhitespace();
                char c = Peek();
                switch (c)
                {
                    case '{': return ParseDict();
                    case '[': return ParseList();
                    case '(': return ParseTuple();
                    case '\'':
                    case '"': return ParseString();
                }
                if (c == '-' || c == '+' || char.IsDigit(c) || c == '.')
                    return ParseNumber();
                if (char.IsLetter(c) || c == '_')
                    return ParseKeyword();
                throw new SyntaxException($"invalid syntax at position {_i}: '{c}'");
            }

            private object ParseKeyword()
            {
                int start = _i;
                while (_i < _s.Length && (char.IsLetterOrDigit(_s[_i]) || _s[_i] == '_'))
                    _i++;
                string word = _s.Substring(start, _i - start);
                switch (word)
                {
                    case "True": return true;
                    case "False": return false;
                    case "None": return null;
                    // Deliberately NOT nan/inf: they are Names, not literals, so ast.literal_eval
                    // rejects them too. Accepting them here would make this parser more permissive
                    // than the thing it stands in for.
                    default: throw new SyntaxException($"unknown name '{word}' at position {start}");
                }
            }

            private object ParseDict()
            {
                Expect('{');
                var dict = new Dictionary<string, object>(StringComparer.Ordinal);
                SkipWhitespace();
                if (Peek() == '}') { _i++; return dict; }

                while (true)
                {
                    SkipWhitespace();
                    if (Peek() != '\'' && Peek() != '"')
                        throw new SyntaxException($"dict keys must be strings, at position {_i}");
                    string key = ParseString();
                    Expect(':');
                    dict[key] = ParseValue();

                    SkipWhitespace();
                    char c = Peek();
                    if (c == ',')
                    {
                        _i++;
                        SkipWhitespace();
                        if (Peek() == '}') { _i++; return dict; } // trailing comma — NumPy always writes one
                        continue;
                    }
                    if (c == '}') { _i++; return dict; }
                    throw new SyntaxException($"expected ',' or '}}' at position {_i}");
                }
            }

            private object ParseList()
            {
                Expect('[');
                return ParseSequence(']', out _); // brackets always build a list, comma or not
            }

            private object ParseTuple()
            {
                Expect('(');
                List<object> items = ParseSequence(')', out bool sawComma);

                // It is the COMMA that makes a tuple in Python, not the parentheses: `(1)` is just the
                // int 1 in grouping parens, while `(1,)` is a 1-tuple and `()` is the empty tuple. The
                // .npy header validator leans on the difference — NumPy rejects `'shape': (1)` with
                // "shape is not valid: 1" because it never saw a tuple.
                if (items.Count == 1 && !sawComma)
                    return items[0];

                return new PyTuple(items);
            }

            private List<object> ParseSequence(char close, out bool sawComma)
            {
                var items = new List<object>();
                sawComma = false;

                SkipWhitespace();
                if (Peek() == close) { _i++; return items; }

                while (true)
                {
                    items.Add(ParseValue());
                    SkipWhitespace();
                    char c = Peek();
                    if (c == ',')
                    {
                        _i++;
                        sawComma = true;
                        SkipWhitespace();
                        if (Peek() == close) { _i++; return items; } // trailing comma: (3,) / [1, 2,]
                        continue;
                    }
                    if (c == close) { _i++; return items; }
                    throw new SyntaxException($"expected ',' or '{close}' at position {_i}");
                }
            }

            private string ParseString()
            {
                char quote = Peek();
                _i++;
                var sb = new StringBuilder();
                while (true)
                {
                    if (_i >= _s.Length)
                        throw new SyntaxException("unterminated string literal");
                    char c = _s[_i++];
                    if (c == quote)
                        return sb.ToString();
                    if (c != '\\')
                    {
                        sb.Append(c);
                        continue;
                    }
                    if (_i >= _s.Length)
                        throw new SyntaxException("unterminated escape sequence");
                    char e = _s[_i++];
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case '0': sb.Append('\0'); break;
                        case '\\': sb.Append('\\'); break;
                        case '\'': sb.Append('\''); break;
                        case '"': sb.Append('"'); break;
                        case 'x': sb.Append(ParseHexEscape(2)); break;
                        case 'u': sb.Append(ParseHexEscape(4)); break;
                        case 'U': sb.Append(char.ConvertFromUtf32(ParseHexCode(8))); break;
                        default: sb.Append('\\').Append(e); break;
                    }
                }
            }

            private char ParseHexEscape(int digits) => (char)ParseHexCode(digits);

            private int ParseHexCode(int digits)
            {
                if (_i + digits > _s.Length)
                    throw new SyntaxException("truncated hex escape");
                int v = 0;
                for (int k = 0; k < digits; k++)
                {
                    int d = HexVal(_s[_i + k]);
                    if (d < 0) throw new SyntaxException("invalid hex escape");
                    v = (v << 4) | d;
                }
                _i += digits;
                return v;
            }

            private static int HexVal(char c)
            {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            }

            private object ParseNumber()
            {
                int start = _i;
                if (_i < _s.Length && (_s[_i] == '-' || _s[_i] == '+'))
                    _i++;

                bool isFloat = false;
                while (_i < _s.Length)
                {
                    char c = _s[_i];
                    if (char.IsDigit(c) || c == '_') { _i++; continue; }
                    if (c == '.') { isFloat = true; _i++; continue; }
                    if (c == 'e' || c == 'E')
                    {
                        // An exponent only continues the number if a digit/sign follows, otherwise
                        // this 'e' begins a following name.
                        int j = _i + 1;
                        if (j < _s.Length && (_s[j] == '+' || _s[j] == '-')) j++;
                        if (j < _s.Length && char.IsDigit(_s[j])) { isFloat = true; _i = j + 1; continue; }
                    }
                    break;
                }

                string text = _s.Substring(start, _i - start).Replace("_", "");
                if (text.Length == 0 || text == "-" || text == "+")
                    throw new SyntaxException($"invalid number at position {start}");

                // Python 2 wrote longs as `3L`. NumPy strips the suffix via _filter_header, but only
                // when retrying a v1.0/v2.0 header; v3.0 post-dates Python 2 and rejects it.
                if (!isFloat && _i < _s.Length && (_s[_i] == 'L' || _s[_i] == 'l'))
                {
                    if (!_allowLongSuffix)
                        throw new SyntaxException($"invalid decimal literal at position {_i}");
                    _i++;
                }

                if (!isFloat && long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l))
                    return l;
                if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
                    return d;
                throw new SyntaxException($"invalid number '{text}' at position {start}");
            }
        }
    }
}
