using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NumSharp.Tests.TextIO
{
    /// <summary>
    ///     Reading the text I/O oracle (<c>test/oracle/gen_textio_oracle.py</c> →
    ///     <c>TextIO/corpus/textio_&lt;section&gt;.jsonl</c>): the corpus's value encodings, NumPy's outcomes, and their
    ///     comparison with what NumSharp returned or raised.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Encodings.</b> A string travels as a JSON string, or as <c>{"u16": hex}</c> (UTF-16LE code units) when it holds
    ///     a lone surrogate, which System.Text.Json refuses to decode. Bytes travel as <c>{"utf8": str}</c> when they are
    ///     strict UTF-8 (re-encoding the decoded string gives the same bytes, because Python's strict decoder rejects every
    ///     sequence UTF-8 cannot produce), else as <c>{"hex": ...}</c>. A large output travels as its SHA-256 (<c>sha</c>
    ///     over bytes, <c>sha16</c> over a text's UTF-16LE code units) and length.
    ///     </para>
    ///     <para>
    ///     <b>Errors.</b> NumPy's exception is compared by class and verbatim text. A same-named NumSharp exception always
    ///     corresponds; otherwise <see cref="ErrorTypes"/> lists the house .NET class these five functions answer a
    ///     NumPy class with (pinned to what they raise: a regression to a looser class fails). .NET's
    ///     <c>" (Parameter 'x')"</c> suffix is not part of the text.
    ///     </para>
    /// </remarks>
    internal static class TextIOOracleCorpus
    {
        /// <summary>The corpus file of one section, beside the test assembly (the project copies the corpus there).</summary>
        /// <param name="section">The section name (<c>fromtext</c>, <c>loadtxt</c>, …).</param>
        /// <returns>The absolute path.</returns>
        internal static string PathOf(string section) =>
            Path.Combine(AppContext.BaseDirectory, "TextIO", "corpus", $"textio_{section}.jsonl");

        /// <summary>Decodes a corpus string: a JSON string, <c>{"u16": hex}</c>, or null.</summary>
        /// <param name="e">The JSON value.</param>
        /// <returns>The string, or null for a JSON null.</returns>
        /// <exception cref="InvalidOperationException">Any other JSON value.</exception>
        internal static string Str(JsonElement e)
        {
            switch (e.ValueKind)
            {
                case JsonValueKind.Null:
                    return null;
                case JsonValueKind.String:
                    return e.GetString();
                case JsonValueKind.Object when e.TryGetProperty("u16", out var hex):
                    return FromU16(hex.GetString());
                default:
                    throw new InvalidOperationException($"not a corpus string: {e.GetRawText()}");
            }
        }

        /// <summary>Decodes corpus bytes: <c>{"utf8": str}</c> or <c>{"hex": ...}</c>.</summary>
        /// <param name="e">The JSON object.</param>
        /// <returns>The bytes.</returns>
        /// <exception cref="InvalidOperationException">Any other JSON value.</exception>
        internal static byte[] Bytes(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("utf8", out var u))
                    return Encoding.UTF8.GetBytes(u.GetString());
                if (e.TryGetProperty("hex", out var h))
                    return Convert.FromHexString(h.GetString());
            }
            throw new InvalidOperationException($"not corpus bytes: {Shorten(e.GetRawText())}");
        }

        /// <summary>A string from its UTF-16LE hex (lone surrogates kept).</summary>
        /// <param name="hex">The hex text.</param>
        /// <returns>The string.</returns>
        internal static string FromU16(string hex) => new string(MemoryMarshal.Cast<byte, char>(Convert.FromHexString(hex)));

        /// <summary>A string's UTF-16LE hex, lowercase (the converter-log keys and StringDType cells use it).</summary>
        /// <param name="s">The string.</param>
        /// <returns>The hex text.</returns>
        internal static string ToU16(string s) => Convert.ToHexString(MemoryMarshal.AsBytes(s.AsSpan())).ToLowerInvariant();

        /// <summary>The lowercase hex SHA-256 of a byte span.</summary>
        /// <param name="data">The bytes.</param>
        /// <returns>The digest.</returns>
        internal static string Sha256(ReadOnlySpan<byte> data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();

        /// <summary>
        ///     The NumPy dtype of a corpus dtype name: a NumPy dtype string, or a StringDType variant (<c>T</c>,
        ///     <c>T_nanNA</c>, <c>T_strNA</c>, <c>T_nocoerce</c>).
        /// </summary>
        /// <param name="name">The corpus name.</param>
        /// <returns>The descriptor.</returns>
        internal static DType MakeDtype(string name) => name switch
        {
            "T" => np.dtypes.StringDType(),
            "T_nanNA" => np.dtypes.StringDType(double.NaN),
            "T_strNA" => np.dtypes.StringDType("NA"),
            "T_nocoerce" => np.dtypes.StringDType(coerce: false),
            _ => np.dtype(name),
        };

        /// <summary>A JSON parameter as the C# object a ported NumPy call passes: lists become <c>object[]</c>.</summary>
        /// <param name="e">The JSON value.</param>
        /// <returns>The object.</returns>
        /// <exception cref="InvalidOperationException">A JSON object (no parameter is one).</exception>
        internal static object Param(JsonElement e) => e.ValueKind switch
        {
            JsonValueKind.Null => null,
            JsonValueKind.String => e.GetString(),
            // The object cast is load-bearing: `cond ? long : double` is typed double, which would hand a Python int (a
            // comments list's 1) to NumSharp as the float 1.0.
            JsonValueKind.Number => e.TryGetInt64(out long l) ? (object)l : e.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Array => e.EnumerateArray().Select(Param).ToArray(),
            _ => throw new InvalidOperationException("parameter kind " + e.ValueKind),
        };

        /// <summary>
        ///     The .NET exception classes NumSharp answers a NumPy exception class with in these five functions, besides a
        ///     same-named class (NumSharp's <c>ValueError</c>, <c>TypeError</c>, <c>UnicodeEncodeError</c>, <c>LookupError</c>,
        ///     …, which carry every other error the corpus records).
        /// </summary>
        /// <remarks>
        ///     Pinned to the pairs the corpus produces (measured 2026-10-15 by replaying it with this table empty: 95
        ///     <c>OverflowError</c> cases, nothing else), so an error that regresses to a looser .NET class fails instead of
        ///     passing on a broad alias. Add a pair only for a class NumSharp deliberately keeps.
        /// </remarks>
        private static readonly Dictionary<string, string[]> ErrorTypes = new()
        {
            // NumSharp's house type for Python's OverflowError (integer fields out of a dtype's range, a float() past the
            // range of float16's double, ...); NumSharp has no class of that name.
            ["OverflowError"] = new[] { "OverflowException" },
        };

        /// <summary>Compares NumSharp's exception with NumPy's recorded one.</summary>
        /// <param name="want">NumPy's exception (<c>{"type", "msg"}</c>).</param>
        /// <param name="got">NumSharp's exception.</param>
        /// <returns>Null when they correspond, else the divergence.</returns>
        internal static string CompareError(JsonElement want, Exception got)
        {
            string wantType = want.GetProperty("type").GetString();
            string wantText = Str(want.GetProperty("msg")).Trim();
            string gotType = got.GetType().Name;
            string gotText = Normalize(got.Message);
            bool typeOk = gotType == wantType || (ErrorTypes.TryGetValue(wantType, out var allowed) && allowed.Contains(gotType));
            if (typeOk && gotText == wantText)
                return null;
            return $"error mismatch\n      numpy:    {wantType}: {Shorten(wantText)}\n      numsharp: {gotType}: {Shorten(gotText)}";
        }

        /// <summary>The verdict for a call NumPy answered with a result while NumSharp raised.</summary>
        /// <param name="got">NumSharp's exception.</param>
        /// <returns>The divergence text (with the inner exception, which usually names the cause).</returns>
        internal static string UnexpectedError(Exception got) =>
            $"numpy returned a result, numsharp raised {got.GetType().Name}: {Shorten(Normalize(got.Message))}" +
            (got.InnerException != null ? $" [inner {got.InnerException.GetType().Name}: {Shorten(got.InnerException.Message)}]" : "");

        /// <summary>Strips .NET's <c>" (Parameter 'x')"</c> suffix and the surrounding whitespace from a message.</summary>
        /// <param name="m">The message.</param>
        /// <returns>The text NumPy's message is compared with.</returns>
        internal static string Normalize(string m)
        {
            if (string.IsNullOrEmpty(m))
                return "";
            int p = m.IndexOf(" (Parameter '", StringComparison.Ordinal);
            if (p >= 0)
                m = m.Substring(0, p);
            return m.Trim();
        }

        /// <summary>Shortens a text for a failure report.</summary>
        /// <param name="s">The text.</param>
        /// <returns>At most 600 characters.</returns>
        internal static string Shorten(string s) => s == null ? "null" : s.Length <= 600 ? s : s.Substring(0, 600) + "…";

        /// <summary>
        ///     Compares a numeric or fixed-width result with NumPy's: the dtype name, the shape and the C-order bytes of the
        ///     written prefix (<c>nread</c> elements, when the case records one: np.fromstring's uninitialized tail is not
        ///     part of NumPy's answer).
        /// </summary>
        /// <param name="want">NumPy's result (<c>dtype</c>, <c>shape</c>, <c>hex</c> or <c>sha</c>, optional <c>nread</c>).</param>
        /// <param name="got">NumSharp's result.</param>
        /// <returns>Null when they match, else the divergence.</returns>
        internal static string CompareArray(JsonElement want, NDArray got)
        {
            string wantDtype = want.GetProperty("dtype").GetString();
            long[] wantShape = want.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            string gotDtype = got.dtype.ToString();
            long[] gotShape = got.shape.Select(x => (long)x).ToArray();
            byte[] all;
            try
            {
                all = got.tobytes();
            }
            catch (Exception e)
            {
                return $"tobytes() threw {e.GetType().Name}: {e.Message}";
            }
            ReadOnlySpan<byte> payload = all;
            if (want.TryGetProperty("nread", out var nr))
                payload = payload.Slice(0, Math.Min(payload.Length, nr.GetInt32() * got.dtype.itemsize));
            string wantBytes, gotBytes;
            if (want.TryGetProperty("sha", out var sha))
            {
                wantBytes = sha.GetString();
                gotBytes = Sha256(payload);
            }
            else
            {
                wantBytes = want.GetProperty("hex").GetString();
                gotBytes = Convert.ToHexString(payload).ToLowerInvariant();
            }
            if (gotDtype == wantDtype && gotShape.SequenceEqual(wantShape) && gotBytes == wantBytes)
                return null;
            return $"value mismatch\n      numpy:    {wantDtype}{Shape(wantShape)} {Shorten(wantBytes)}\n" +
                   $"      numsharp: {gotDtype}{Shape(gotShape)} {Shorten(gotBytes)}";
        }

        /// <summary>
        ///     Compares a StringDType result with NumPy's elements, each as a canonical key: <c>None</c>, <c>nan</c> (a NaN NA
        ///     object), <c>s:</c> plus the text's UTF-16LE hex (exact, lone surrogates included) or <c>o:</c> plus another
        ///     object's repr.
        /// </summary>
        /// <param name="want">NumPy's result (<c>dtype</c>, <c>shape</c>, <c>cells</c>: see <see cref="CellKey"/>).</param>
        /// <param name="got">NumSharp's result.</param>
        /// <returns>Null when they match, else the divergence (the elements shown as JSON text).</returns>
        internal static string CompareCells(JsonElement want, NDArray got)
        {
            string wantDtype = want.GetProperty("dtype").GetString();
            long[] wantShape = want.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
            var wantCells = want.GetProperty("cells").EnumerateArray().Select(CellKey).ToList();
            string gotDtype = got.dtype.ToString();
            long[] gotShape = got.shape.Select(x => (long)x).ToArray();
            List<string> gotCells;
            try
            {
                gotCells = new List<string>();
                using var flat = got.ravel();
                for (long i = 0; i < flat.size; i++)
                {
                    object x = flat.item(i);
                    gotCells.Add(x switch
                    {
                        null => "None",
                        double d when double.IsNaN(d) => "nan",
                        string s => "s:" + ToU16(s),
                        _ => "o:" + x,
                    });
                }
            }
            catch (Exception e)
            {
                gotCells = new List<string> { $"<{e.GetType().Name}: {e.Message}>" };
            }
            if (gotDtype == wantDtype && gotShape.SequenceEqual(wantShape) && gotCells.SequenceEqual(wantCells))
                return null;
            return $"value mismatch\n      numpy:    {wantDtype}{Shape(wantShape)} {Shorten(ShowCells(wantCells))}\n" +
                   $"      numsharp: {gotDtype}{Shape(gotShape)} {Shorten(ShowCells(gotCells))}";
        }

        /// <summary>
        ///     The canonical key of one recorded StringDType element: <c>null</c> → <c>None</c>, <c>{"nan": true}</c> →
        ///     <c>nan</c>, <c>{"o": repr}</c> → <c>o:repr</c>, a string (or <c>{"u16": hex}</c>) → <c>s:</c> plus its UTF-16LE hex.
        /// </summary>
        /// <param name="e">The recorded element.</param>
        /// <returns>The key.</returns>
        internal static string CellKey(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Null)
                return "None";
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("nan", out _))
                    return "nan";
                if (e.TryGetProperty("o", out var repr))
                    return "o:" + repr.GetString();
            }
            return "s:" + ToU16(Str(e));
        }

        /// <summary>Canonical element keys as readable text: each <c>s:</c> key's string JSON-quoted.</summary>
        /// <param name="keys">The keys.</param>
        /// <returns>The comma-separated elements.</returns>
        private static string ShowCells(IEnumerable<string> keys) =>
            string.Join(", ", keys.Select(k => k.StartsWith("s:", StringComparison.Ordinal)
                ? JsonSerializer.Serialize(FromU16(k.Substring(2)))
                : k));

        /// <summary>Compares written bytes with NumPy's (<c>utf8</c> / <c>hex</c> inline, or <c>sha</c> and <c>len</c>).</summary>
        /// <param name="want">NumPy's output.</param>
        /// <param name="got">NumSharp's bytes.</param>
        /// <returns>Null when they match, else the divergence.</returns>
        internal static string CompareBytes(JsonElement want, byte[] got)
        {
            if (want.TryGetProperty("sha", out var sha))
            {
                if (Sha256(got) == sha.GetString())
                    return null;
                return $"bytes differ (sha): numpy {want.GetProperty("len").GetInt64()} bytes, numsharp {got.Length}";
            }
            byte[] expected = Bytes(want);
            if (expected.AsSpan().SequenceEqual(got))
                return null;
            return $"bytes differ\n      numpy:    {Shorten(Convert.ToHexString(expected).ToLowerInvariant())}\n" +
                   $"      numsharp: {Shorten(Convert.ToHexString(got).ToLowerInvariant())}";
        }

        /// <summary>Compares a written text with NumPy's (<c>text</c> inline, or <c>sha16</c> and <c>len</c>).</summary>
        /// <param name="want">NumPy's output.</param>
        /// <param name="got">NumSharp's text.</param>
        /// <returns>Null when they match, else the divergence.</returns>
        internal static string CompareText(JsonElement want, string got)
        {
            if (want.TryGetProperty("sha16", out var sha))
            {
                if (Sha256(MemoryMarshal.AsBytes(got.AsSpan())) == sha.GetString())
                    return null;
                return $"text differs (sha): numpy {want.GetProperty("len").GetInt64()} chars, numsharp {got.Length}";
            }
            string expected = Str(want.GetProperty("text"));
            if (expected == got)
                return null;
            return $"text differs\n      numpy:    {Shorten(JsonSerializer.Serialize(expected))}\n" +
                   $"      numsharp: {Shorten(JsonSerializer.Serialize(got))}";
        }

        /// <summary>
        ///     The C# twin of a recorded Python converter: it answers each field from the case's call log (the field's
        ///     text → the result NumPy's converter returned, or the exception it raised).
        /// </summary>
        /// <param name="name">The converter's menu name.</param>
        /// <param name="log">The case's call log (<c>conv_calls</c>).</param>
        /// <returns>The converter.</returns>
        /// <exception cref="InvalidOperationException">(From the converter) a field NumPy's converter never saw: NumSharp
        ///     called it with a text NumPy did not, which the case then reports.</exception>
        internal static Func<string, object> ConverterTwin(string name, JsonElement log)
        {
            return s =>
            {
                if (!log.TryGetProperty(name, out var calls) || !calls.TryGetProperty(ToU16(s), out var v))
                    throw new InvalidOperationException($"UNLOGGED converter call {name}({JsonSerializer.Serialize(s)})");
                switch (v.GetProperty("t").GetString())
                {
                    case "n": return null;
                    case "bool": return v.GetProperty("v").GetBoolean();
                    case "i":
                    {
                        var big = BigInteger.Parse(v.GetProperty("v").GetString(), System.Globalization.CultureInfo.InvariantCulture);
                        return big >= long.MinValue && big <= long.MaxValue ? (object)(long)big : big;
                    }
                    case "f": return BitConverter.UInt64BitsToDouble(v.GetProperty("v").GetUInt64());
                    case "c":
                        return new Complex(BitConverter.UInt64BitsToDouble(v.GetProperty("re").GetUInt64()),
                                           BitConverter.UInt64BitsToDouble(v.GetProperty("im").GetUInt64()));
                    case "h": return BitConverter.UInt16BitsToHalf((ushort)v.GetProperty("v").GetInt32());
                    case "b": return Convert.FromHexString(v.GetProperty("v").GetString());
                    case "s": return FromU16(v.GetProperty("v").GetString());
                    case "x":
                    {
                        string type = v.GetProperty("type").GetString(), msg = Str(v.GetProperty("msg"));
                        // The converter's own exception, as the C# class a ported converter would raise.
                        throw type switch
                        {
                            "ZeroDivisionError" => new DivideByZeroException(msg),
                            "ValueError" => new ValueError(msg),
                            "OverflowError" => new OverflowException(msg),
                            "TypeError" => new TypeError(msg),
                            _ => new InvalidOperationException(type + ": " + msg),
                        };
                    }
                }
                throw new InvalidOperationException("unknown converter result kind " + v.GetRawText());
            };
        }

        /// <summary>A shape as Python prints it, without spaces.</summary>
        /// <param name="s">The dimensions.</param>
        /// <returns>For example <c>(2,3)</c>.</returns>
        private static string Shape(long[] s) => "(" + string.Join(",", s) + ")";
    }
}
