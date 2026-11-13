using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        // ---------------------------------------------------------------------------------------------------------
        // Comma strings: NumPy's _convert_from_commastring (descriptor.c) over numpy._core._internal._commastring
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The <see cref="Exception.Data"/> key under which <see cref="NumPyOnlyDType"/> records NumPy's item size of the
        ///     descriptor it refuses (a boxed <see cref="int"/>). Only the comma-string conversion reads it.
        /// </summary>
        private const string NumPyItemSizeKey = "NumSharp.NumPyItemSize";

        /// <summary>The refusal's parenthetical for a comma string NumPy reads as a sub-array dtype (<c>'3f8'</c>, <c>'(2,3)i4'</c>).</summary>
        private const string SubArrayKind = "a sub-array dtype";

        /// <summary>The refusal's parenthetical for a comma string NumPy reads as a structured dtype (<c>'i4,f8'</c>, <c>'f8,'</c>).</summary>
        private const string StructuredKind = "a structured dtype";

        /// <summary>
        ///     NumPy's <c>NPY_MAXDIMS</c> (64 in NumPy 2.x): the most dimensions <c>PyArray_IntpConverter</c> accepts, and so the
        ///     most a sub-array shape in a comma string may have. NumSharp's own arrays have no such limit; this one is part of
        ///     NumPy's error behaviour (<c>'(1,...,1)f8'</c> with 65 ones is "invalid shape in fixed-type tuple.").
        /// </summary>
        private const int NPY_MAXDIMS = 64;

        /// <summary>
        ///     The refusal of a string that IS a valid NumPy dtype of a class NumSharp has no storage for: a
        ///     <see cref="NotSupportedException"/> <c>NumPy dtype 'X' is not supported by NumSharp</c> that also records the
        ///     item size of the descriptor NumPy would build, under <see cref="NumPyItemSizeKey"/> in <see cref="Exception.Data"/>.
        /// </summary>
        /// <param name="type">The string to quote (the caller's spelling: the whole string, or the endian-stripped code).</param>
        /// <param name="numpyItemSize">NumPy's <c>elsize</c> of the refused descriptor (complex64 8, object the pointer size,
        ///     the unsized void 0, <c>'V8'</c> 8, the host's long double width, a sub-array's total bytes).</param>
        /// <param name="kind">A parenthetical naming the descriptor family (<see cref="SubArrayKind"/>,
        ///     <see cref="StructuredKind"/>), or null for a class NumSharp lacks.</param>
        /// <returns>The exception to throw (its type is exactly <see cref="NotSupportedException"/>).</returns>
        /// <remarks>
        ///     The item size is what lets a comma string DEFER the refusal. NumPy converts every field of a comma string before
        ///     it builds the structured dtype, and checks a sub-array's byte size against its base's item size, so
        ///     <c>'F,xyz'</c> is NumPy's TypeError for <c>'xyz'</c>, <c>'(300000000,)F'</c> its ValueError "dtype size in bytes
        ///     must fit into a C int." and <c>'()V'</c> its ValueError "invalid itemsize in generic type tuple". NumSharp raises
        ///     those too and raises <see cref="NotSupportedException"/> only where NumPy returns a dtype:
        ///     <see cref="ConvertCommaItemString"/> catches the tagged refusal of a field and carries its item size on.
        /// </remarks>
        private static NotSupportedException NumPyOnlyDType(string type, int numpyItemSize, string kind = null)
        {
            var refusal = new NotSupportedException(kind == null
                ? $"NumPy dtype '{type}' is not supported by NumSharp"
                : $"NumPy dtype '{type}' ({kind}) is not supported by NumSharp");
            refusal.Data[NumPyItemSizeKey] = numpyItemSize;
            return refusal;
        }

        /// <summary>
        ///     One descriptor of the comma-string conversion: a NumSharp descriptor, or the facts NumPy's conversion reads of a
        ///     descriptor NumPy builds and NumSharp has no class for (a sub-array, a structured dtype, complex64 / object /
        ///     void). Holding the facts instead of raising lets the conversion go on to the later fields, whose errors NumPy
        ///     raises first.
        /// </summary>
        private readonly struct CommaDescr
        {
            /// <summary>A descriptor NumSharp has.</summary>
            /// <param name="descr">The descriptor (not null).</param>
            internal CommaDescr(DType descr)
            {
                Descr = descr;
                ItemSize = descr.itemsize;
                HasFields = false;
                Kind = null;
            }

            /// <summary>A descriptor NumPy builds and NumSharp has no class for.</summary>
            /// <param name="itemSize">NumPy's <c>elsize</c> of it.</param>
            /// <param name="hasFields">Whether it is structured (NumPy's <c>PyDataType_HASFIELDS</c>).</param>
            /// <param name="kind">The refusal's parenthetical (<see cref="SubArrayKind"/>, <see cref="StructuredKind"/>), or
            ///     null for a class NumSharp lacks (complex64, object, void, the 16-byte long double).</param>
            internal CommaDescr(int itemSize, bool hasFields, string kind)
            {
                Descr = null;
                ItemSize = itemSize;
                HasFields = hasFields;
                Kind = kind;
            }

            /// <summary>The NumSharp descriptor, or null when NumSharp has no class for NumPy's.</summary>
            internal readonly DType Descr;

            /// <summary>NumPy's <c>elsize</c>: bytes per element (a <c>U</c> descriptor's are 4 per character).</summary>
            internal readonly int ItemSize;

            /// <summary>NumPy's <c>PyDataType_HASFIELDS</c>: a structured descriptor (only a NumPy-only one can be).</summary>
            internal readonly bool HasFields;

            /// <summary>The refusal's parenthetical, or null.</summary>
            internal readonly string Kind;

            /// <summary>
            ///     NumPy's <c>PyDataType_ISUNSIZED</c> (<c>elsize == 0 &amp;&amp; !HASFIELDS</c>): the unsized <c>S</c> /
            ///     <c>U</c> / <c>V</c> classes, and a sub-array of zero elements (<c>'0f8'</c>, NumPy's own quirk), whose repeat
            ///     count is read as an ITEM SIZE instead of a shape.
            /// </summary>
            internal bool IsUnsized => ItemSize == 0 && !HasFields;
        }

        /// <summary>
        ///     The <c>ast.literal_eval</c> value of a comma string's repeat group: a Python int (<c>'3'</c>, <c>'(3)'</c>) or a
        ///     tuple of ints (<c>'(2,3)'</c>, <c>'2,'</c>, <c>'()'</c>). The group's regex admits nothing else.
        /// </summary>
        private readonly struct CommaRepeats
        {
            /// <summary>Wraps a literal value.</summary>
            /// <param name="values">The int (one value) or the tuple's ints, at Python's arbitrary precision.</param>
            /// <param name="isTuple">True for a tuple (even of one element), false for an int.</param>
            internal CommaRepeats(BigInteger[] values, bool isTuple)
            {
                Values = values;
                IsTuple = isTuple;
            }

            /// <summary>The int (one value), or the tuple's items.</summary>
            internal readonly BigInteger[] Values;

            /// <summary>True for a tuple, false for an int.</summary>
            internal readonly bool IsTuple;
        }

        /// <summary>
        ///     One item of <c>_commastring</c>'s result: a dtype string (Python's <c>str</c> item) or a
        ///     <c>(dtype, repeats)</c> pair (Python's 2-tuple item).
        /// </summary>
        private readonly struct CommaItem
        {
            /// <summary>Builds an item.</summary>
            /// <param name="dtype">The field's dtype text with its merged byte order in front (NumPy's <c>order + dtype</c>).</param>
            /// <param name="repeats">The repeat group's literal, or null when the group was empty.</param>
            internal CommaItem(string dtype, CommaRepeats? repeats)
            {
                Dtype = dtype;
                Repeats = repeats;
            }

            /// <summary>The dtype text (possibly empty).</summary>
            internal readonly string Dtype;

            /// <summary>The repeats, or null for a plain dtype string.</summary>
            internal readonly CommaRepeats? Repeats;
        }

        /// <summary>
        ///     NumPy's <c>_convert_from_commastring</c>: split the string with <c>_commastring</c>, then convert the result - a
        ///     <c>(dtype, repeats)</c> pair through <c>_convert_from_tuple</c>, a list of fields through
        ///     <c>_convert_from_list</c>.
        /// </summary>
        /// <param name="dtype">A string <see cref="CheckForCommaString"/> accepted (<c>'3f8'</c>, <c>'()U5'</c>, <c>'i4,f8'</c>).</param>
        /// <returns>
        ///     The descriptor: a NumSharp one where NumPy's result is a plain dtype (<c>'()f8'</c> is the float64 singleton,
        ///     <c>'2U'</c> a fresh <c>&lt;U2</c>), else the facts of NumPy's sub-array / structured / NumSharp-less result.
        /// </returns>
        /// <exception cref="ValueError">NumPy's texts: <c>format number N of "…" is not recognized</c>, <c>inconsistent
        ///     byte-order specification X and Y</c>, <c>invalid itemsize in generic type tuple</c>, <c>invalid shape in
        ///     fixed-type tuple.</c> (and its three <c>: …</c> variants), <c>Expected at least one field name</c>.</exception>
        /// <exception cref="SyntaxError">A repeat group that is not a Python literal (CPython 3.12's texts, see
        ///     <see cref="LiteralEvalRepeats"/>).</exception>
        /// <exception cref="TypeError">A field's dtype NumPy does not understand (<c>'3xyz'</c>, <c>'i4,xyz'</c>, the empty
        ///     field of <c>'3'</c>), with <see cref="ConvertFromStr"/>'s texts.</exception>
        /// <exception cref="RuntimeError"><c>_commastring</c> returned a bare string; NumPy keeps the branch, but no string
        ///     that passes <see cref="CheckForCommaString"/> reaches it.</exception>
        private static CommaDescr ConvertFromCommaString(string dtype)
        {
            List<CommaItem> parsed = CommaString(dtype, out bool isList);
            // _commastring returns `result if islist else result[0]`: a list, or its one item - a 2-tuple when the item has
            // repeats, a str otherwise.
            if (!isList && parsed[0].Repeats != null)
                return ConvertCommaTuple(parsed[0]);
            if (isList && parsed.Count >= 1)
                return ConvertCommaList(parsed);
            throw new RuntimeError("_commastring should return a tuple with len == 2, or a list with len >= 1");
        }

        /// <summary>
        ///     NumPy's <c>numpy._core._internal._commastring</c>: split a comma string into fields with <c>format_re</c>
        ///     (<c>order1, repeats, order2, dtype</c>), separated by <c>sep_re</c> (<c>\s*,\s*</c>) and optionally padded at the
        ///     end (<c>space_re</c>, <c>\s+$</c>); merge each field's two byte orders and read its repeats with
        ///     <c>ast.literal_eval</c>.
        /// </summary>
        /// <param name="astr">The whole comma string (not empty).</param>
        /// <param name="isList">Set to NumPy's <c>islist</c>: true once a separator was read, so the caller converts a list.</param>
        /// <returns>The fields in order (at least one).</returns>
        /// <exception cref="ValueError"><c>format number N of "astr" is not recognized</c> (the text after a field is neither
        ///     a separator nor trailing whitespace), or <c>inconsistent byte-order specification X and Y</c> (two different
        ///     byte orders around the repeats, <c>'='</c> read as the host's).</exception>
        /// <exception cref="SyntaxError">A repeat group that is not a Python literal.</exception>
        /// <remarks>
        ///     Every field is split and literal-evaluated BEFORE any is converted, as in NumPy: <c>'xyz,f8,,'</c> is the third
        ///     field's SyntaxError, not the first field's TypeError. Each regex group is greedy and may be empty, so
        ///     <c>format_re</c> always matches (NumPy's <c>except (TypeError, AttributeError)</c> for a failed match cannot
        ///     fire) and the scan below is the greedy match itself. <c>\s</c> is Python's Unicode whitespace
        ///     (<see cref="PyUnicode.IsSpace"/>: the no-break space and U+001C..U+001F count, U+200B does not); the other
        ///     classes are ASCII. NumPy also warns (DeprecationWarning) for a parenthesized single number (<c>'(2)f8,i4'</c>);
        ///     NumSharp has no warning channel, and the value is the same.
        /// </remarks>
        private static List<CommaItem> CommaString(string astr, out bool isList)
        {
            int startindex = 0;
            var result = new List<CommaItem>();
            isList = false;
            while (startindex < astr.Length)
            {
                // format_re = (?P<order1>[<>|=]?)(?P<repeats> *[(]?[ ,0-9]*[)]? *)(?P<order2>[<>|=]?)
                //             (?P<dtype>[A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?), matched at startindex.
                int pos = startindex;
                char order1 = pos < astr.Length && IsByteOrderChar(astr[pos]) ? astr[pos++] : '\0';
                int repeatsStart = pos;
                while (pos < astr.Length && astr[pos] == ' ')
                    pos++;
                if (pos < astr.Length && astr[pos] == '(')
                    pos++;
                while (pos < astr.Length && (astr[pos] == ' ' || astr[pos] == ',' || char.IsAsciiDigit(astr[pos])))
                    pos++;
                if (pos < astr.Length && astr[pos] == ')')
                    pos++;
                while (pos < astr.Length && astr[pos] == ' ')
                    pos++;
                string repeats = astr.Substring(repeatsStart, pos - repeatsStart);
                char order2 = pos < astr.Length && IsByteOrderChar(astr[pos]) ? astr[pos++] : '\0';
                int dtypeStart = pos;
                while (pos < astr.Length && (char.IsAsciiLetterOrDigit(astr[pos]) || astr[pos] == '.' || astr[pos] == '?'))
                    pos++;
                // The optional bracket group needs at least one class character and its ']'; otherwise it matches nothing
                // and the '[' is left for the separator test (which then fails: 'M8[s' is "not recognized").
                if (pos < astr.Length && astr[pos] == '[')
                {
                    int close = pos + 1;
                    while (close < astr.Length && (char.IsAsciiLetterOrDigit(astr[close]) || astr[close] == ',' || astr[close] == '.'))
                        close++;
                    if (close > pos + 1 && close < astr.Length && astr[close] == ']')
                        pos = close + 1;
                }
                string dtype = astr.Substring(dtypeStart, pos - dtypeStart);
                startindex = pos;

                // Separator or ending padding
                if (startindex < astr.Length)
                {
                    if (IsPySpaceToEnd(astr, startindex))
                    {
                        startindex = astr.Length;
                    }
                    else
                    {
                        int p = startindex;
                        while (p < astr.Length && PyUnicode.IsSpace(astr[p]))
                            p++;
                        if (p >= astr.Length || astr[p] != ',')
                            throw new ValueError(string.Create(CultureInfo.InvariantCulture, $"format number {result.Count + 1} of \"{astr}\" is not recognized"));
                        p++;
                        while (p < astr.Length && PyUnicode.IsSpace(astr[p]))
                            p++;
                        startindex = p;
                        isList = true;
                    }
                }

                // The two byte orders must agree; _convorder reads '=' as the host's order before comparing, so the message
                // shows '<' for an '=' on a little-endian host ('>2=i4': "inconsistent byte-order specification > and <").
                char order;
                if (order2 == '\0')
                {
                    order = order1;
                }
                else if (order1 == '\0')
                {
                    order = order2;
                }
                else
                {
                    char first = order1 == '=' ? DType.NPY_NATBYTE : order1;
                    char second = order2 == '=' ? DType.NPY_NATBYTE : order2;
                    if (first != second)
                        throw new ValueError($"inconsistent byte-order specification {first} and {second}");
                    order = first;
                }

                // A native, explicit-native or not-applicable order is dropped: the field's text is the bare dtype then.
                if (order == '|' || order == '=' || order == DType.NPY_NATBYTE)
                    order = '\0';
                string itemDtype = order == '\0' ? dtype : order + dtype;
                result.Add(repeats.Length == 0
                    ? new CommaItem(itemDtype, null)
                    : new CommaItem(itemDtype, LiteralEvalRepeats(repeats)));
            }

            return result;
        }

        /// <summary>
        ///     Python's <c>space_re.match(astr, pos)</c> (<c>\s+$</c>): the text from <paramref name="start"/> is non-empty and
        ///     all Python whitespace. (<c>$</c> also matches before a final <c>'\n'</c>, which is whitespace itself, so the test
        ///     is "whitespace to the end".)
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="start">The position to test from (below the length).</param>
        /// <returns>True when only whitespace remains.</returns>
        private static bool IsPySpaceToEnd(string text, int start)
        {
            for (int i = start; i < text.Length; i++)
                if (!PyUnicode.IsSpace(text[i]))
                    return false;
            return true;
        }

        /// <summary>The tokens CPython's tokenizer makes of a repeat group (spaces separate, and are no token).</summary>
        private enum RepeatsToken
        {
            /// <summary><c>'('</c>, which only the group's first character can be.</summary>
            Open,

            /// <summary><c>')'</c>, which only the group's last character (before trailing spaces) can be.</summary>
            Close,

            /// <summary><c>','</c>.</summary>
            Comma,

            /// <summary>A decimal integer literal: a run of digits.</summary>
            Number,
        }

        /// <summary>
        ///     <c>ast.literal_eval(repeats)</c> on CPython 3.12 for the language of NumPy's repeat group
        ///     (<c> *[(]?[ ,0-9]*[)]? *</c>), with CPython's SyntaxError texts.
        /// </summary>
        /// <param name="source">The non-empty repeat group: spaces, an optional leading <c>'('</c>, digits, spaces and commas,
        ///     an optional <c>')'</c>, spaces.</param>
        /// <returns>The int or the tuple of ints.</returns>
        /// <exception cref="SyntaxError">
        ///     CPython's texts, each with <c> (&lt;unknown&gt;, line 1)</c> appended as <c>str(e)</c> has it: <c>invalid
        ///     syntax</c> (line 0 for an all-space group), <c>'(' was never closed</c>, <c>unmatched ')'</c>, <c>leading zeros
        ///     in decimal integer literals are not permitted; use an 0o prefix for octal integers</c>, <c>invalid syntax.
        ///     Perhaps you forgot a comma?</c>.
        /// </exception>
        /// <remarks>
        ///     A decision model of CPython's two parser passes, verified against <c>ast.literal_eval</c> for every string of
        ///     the language up to 7 characters over <c> (),0123</c> (483,719 strings) and up to 9 over <c> (),01</c> (621,354),
        ///     on CPython 3.12.12 (Windows) and 3.12.3 (Linux):
        ///     <list type="number">
        ///       <item><description><c>literal_eval</c> strips leading spaces; nothing left is <c>invalid syntax</c> on line 0
        ///       (the parser reads an empty source).</description></item>
        ///       <item><description>The TOKENIZER's errors win, the first in source order: a number with a leading zero and
        ///       another digit (<c>01</c>; <c>00</c> is 0), a <c>')'</c> with no open <c>'('</c>. The parser's error pass ends
        ///       by tokenizing the rest of the source, and a tokenizer error found there replaces its own.</description></item>
        ///       <item><description>Otherwise the parse: <c>NUM (',' NUM)* [',']</c>, or that (or nothing) inside the
        ///       parentheses. A number right after a number inside the parentheses is the error pass's <c>invalid_expression</c>
        ///       ("Perhaps you forgot a comma?", only at a parenthesis level above 0) - unless that second number ends an
        ///       unclosed source, where peeking past it reads the end of input inside <c>'('</c> first and raises
        ///       <c>'(' was never closed</c>. A valid prefix ended by the end of input inside <c>'('</c> is <c>'(' was never
        ///       closed</c>. Anything else is <c>invalid syntax</c>.</description></item>
        ///     </list>
        ///     A parenthesized single number is the int (<c>'(2)'</c> is 2), a trailing comma makes a tuple (<c>'2,'</c> is
        ///     <c>(2,)</c>), <c>'()'</c> is the empty tuple. Ints keep Python's arbitrary precision.
        /// </remarks>
        private static CommaRepeats LiteralEvalRepeats(string source)
        {
            // literal_eval: node_or_string.lstrip(" \t") (the group itself holds spaces only).
            int start = 0;
            while (start < source.Length && (source[start] == ' ' || source[start] == '\t'))
                start++;
            if (start == source.Length)
                throw new SyntaxError("invalid syntax (<unknown>, line 0)");

            // Tokenize the whole source first: a tokenizer error anywhere is the answer, the first in source order.
            var tokens = new List<(RepeatsToken What, int Start, int Length)>();
            int level = 0;
            for (int i = start; i < source.Length;)
            {
                char c = source[i];
                switch (c)
                {
                    case ' ':
                        i++;
                        continue;
                    case '(':
                        level++;
                        tokens.Add((RepeatsToken.Open, i, 1));
                        i++;
                        continue;
                    case ')':
                        if (level == 0)
                            throw RepeatsSyntax("unmatched ')'");
                        level--;
                        tokens.Add((RepeatsToken.Close, i, 1));
                        i++;
                        continue;
                    case ',':
                        tokens.Add((RepeatsToken.Comma, i, 1));
                        i++;
                        continue;
                }

                // The repeat group admits only digits besides the four characters above; anything else would loop forever.
                Debug.Assert(char.IsAsciiDigit(c), "the repeat group holds spaces, parentheses, commas and digits only");
                if (!char.IsAsciiDigit(c))
                    throw RepeatsSyntax("invalid syntax");
                int end = i;
                bool nonzero = false;
                while (end < source.Length && char.IsAsciiDigit(source[end]))
                {
                    nonzero |= source[end] != '0';
                    end++;
                }
                // CPython's tokenizer: a decimal literal may start with '0' only when every digit is '0'.
                if (c == '0' && nonzero)
                    throw RepeatsSyntax("leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers");
                tokens.Add((RepeatsToken.Number, i, end - i));
                i = end;
            }

            // The parse. '(' can only come first and ')' only last (the group's regex), so one optional pair surrounds the body.
            bool paren = tokens.Count > 0 && tokens[0].What == RepeatsToken.Open;
            int k = paren ? 1 : 0;
            var values = new List<BigInteger>();
            bool expectNumber = true;
            bool trailingComma = false;
            for (; k < tokens.Count; k++)
            {
                var token = tokens[k];
                if (token.What == RepeatsToken.Close)
                    break;
                if (token.What == RepeatsToken.Number)
                {
                    if (!expectNumber)
                    {
                        if (!paren)
                            throw RepeatsSyntax("invalid syntax");
                        // Inside '(' the error pass parses the second number as invalid_expression's right operand and
                        // peeks at the token after it: the end of input there raises "never closed" before the comma hint.
                        throw RepeatsSyntax(k == tokens.Count - 1 ? "'(' was never closed" : "invalid syntax. Perhaps you forgot a comma?");
                    }
                    values.Add(BigInteger.Parse(source.AsSpan(token.Start, token.Length), NumberStyles.None, CultureInfo.InvariantCulture));
                    expectNumber = false;
                    trailingComma = false;
                }
                else
                {
                    if (expectNumber)
                        throw RepeatsSyntax("invalid syntax");
                    expectNumber = true;
                    trailingComma = true;
                }
            }

            bool closed = k < tokens.Count && tokens[k].What == RepeatsToken.Close;
            if (paren && !closed)
                throw RepeatsSyntax("'(' was never closed");
            if (paren && values.Count == 0)
                return new CommaRepeats(Array.Empty<BigInteger>(), isTuple: true);
            if (values.Count == 1 && !trailingComma)
                return new CommaRepeats(values.ToArray(), isTuple: false);
            return new CommaRepeats(values.ToArray(), isTuple: true);
        }

        /// <summary>A <see cref="SyntaxError"/> as <c>str()</c> of CPython's shows it for <c>ast.parse</c> source with no file name.</summary>
        /// <param name="message">CPython's message.</param>
        /// <returns>The error to throw.</returns>
        private static SyntaxError RepeatsSyntax(string message) => new SyntaxError(message + " (<unknown>, line 1)");

        /// <summary>
        ///     NumPy's <c>_convert_from_any</c> of one dtype string inside a comma string: a comma string again (the item of
        ///     <c>'2&gt;3f8'</c> is <c>'&gt;3f8'</c>) converts through <see cref="ConvertFromCommaString"/>; any other string through
        ///     <see cref="ConvertFromStr"/>, whose refusal of a NumPy dtype NumSharp lacks is kept as that dtype's facts.
        /// </summary>
        /// <param name="dtype">The item's dtype text (possibly empty).</param>
        /// <returns>The converted descriptor.</returns>
        /// <exception cref="TypeError">A string NumPy does not understand (the empty one included), with NumPy's text.</exception>
        /// <exception cref="ValueError">A bad datetime divisor, or a nested comma string's ValueError.</exception>
        /// <exception cref="SyntaxError">A nested comma string's repeat group is not a Python literal.</exception>
        private static CommaDescr ConvertCommaItemString(string dtype)
        {
            // _convert_from_str's own order: the empty string fails before the comma test (which is false for it anyway).
            if (dtype.Length != 0 && CheckForCommaString(dtype))
                return ConvertFromCommaString(dtype);
            try
            {
                return new CommaDescr(ConvertFromStr(dtype));
            }
            catch (NotSupportedException refusal) when (refusal.Data[NumPyItemSizeKey] is int numpyItemSize)
            {
                // A valid NumPy dtype NumSharp has no class for ('F', 'O8', 'V16', 'complex64'): keep NumPy's item size so
                // the tuple and list conversions check it as NumPy does, and refuse once the whole string has converted.
                return new CommaDescr(numpyItemSize, hasFields: false, kind: null);
            }
        }

        /// <summary>
        ///     NumPy's <c>_convert_from_tuple</c> for a <c>(dtype, repeats)</c> item: an unsized type reads the repeats as its
        ///     ITEM SIZE (<c>'2U'</c> is <c>&lt;U2</c>, <c>'0S'</c> a fresh <c>S0</c>); <c>(type, ())</c> is the type itself
        ///     (<c>'()f8'</c> is the float64 singleton); any other repeats are a sub-array SHAPE (<c>'3f8'</c>,
        ///     <c>'(2,3)i4'</c>), validated as NumPy validates it.
        /// </summary>
        /// <param name="item">An item with repeats.</param>
        /// <returns>The descriptor: NumSharp's for an unsized string class or <c>()</c>, else a sub-array's facts.</returns>
        /// <exception cref="ValueError">
        ///     NumPy's texts: <c>invalid itemsize in generic type tuple</c> (an unsized type whose repeats are a tuple or do
        ///     not fit a C int, 4 bytes per character for <c>U</c>: <c>'()U'</c>, <c>'(1,)S'</c>, <c>'536870912U'</c>),
        ///     <c>invalid shape in fixed-type tuple.</c> (a dimension past <c>npy_intp</c>, more than 64 dimensions),
        ///     <c>invalid shape in fixed-type tuple: dimension does not fit into a C int.</c>, <c>invalid shape in fixed-type
        ///     tuple: dtype size in bytes must fit into a C int.</c>
        /// </exception>
        /// <exception cref="TypeError">The item's dtype is not understood, or a <c>StringDType</c> sub-array
        ///     (<c>'2T'</c>, <c>'(2,)T'</c>): NumPy 2.5's <c>numpy.dtypes.StringDType is not currently supported within
        ///     subarray dtypes.</c></exception>
        /// <exception cref="SyntaxError">A nested comma string's repeat group is not a Python literal.</exception>
        private static CommaDescr ConvertCommaTuple(CommaItem item)
        {
            CommaDescr type = ConvertCommaItemString(item.Dtype);
            CommaRepeats val = item.Repeats.Value;

            // _try_convert_from_inherit_tuple(type, val) answers NotImplemented for every repeats value: a tuple of integers
            // is "a subarray or flexible type", and a Python int is no dtype (_convert_from_any fails, the error is cleared).

            if (type.IsUnsized)
            {
                // interpret next item as a typesize: PyArray_PyIntAsInt fails for a tuple (no __index__) and for an int past
                // a C int; NumPy then replaces its error.
                int itemsize = !val.IsTuple && val.Values[0] >= int.MinValue && val.Values[0] <= int.MaxValue ? (int)val.Values[0] : -1;
                if (type.Descr?.type_num == NPY_TYPES.NPY_UNICODE)
                    itemsize = itemsize > int.MaxValue / 4 ? -1 : itemsize * 4;
                if (itemsize < 0)
                    throw new ValueError("invalid itemsize in generic type tuple");
                // PyArray_DESCR_REPLACE + elsize: a fresh copy of the unsized descriptor with the new size, byte order and
                // metadata kept ('2>U' is '>U2'). NumSharp's unsized classes are the two string classes.
                if (type.Descr != null)
                    return new CommaDescr(new DType(type.Descr.Meta, type.Descr.byteorder, null, itemsize, type.Descr.Meta.TypeChar, null, type.Descr.metadata));
                return new CommaDescr(itemsize, type.HasFields, type.Kind);
            }

            // (A metadata dict as the second item would merge here; the repeats are never a dict.)

            // interpret next item as shape: PyArray_IntpConverter (an int is a 1-d shape), NumPy's text replacing its errors.
            if (!IntpConverter(val, out long[] shape) || shape.Length > NPY_MAXDIMS)
                throw new ValueError("invalid shape in fixed-type tuple.");
            // if (type, ()) was given it is equivalent to type...
            if (shape.Length == 0 && val.IsTuple)
                return type;

            // NumPy 2.5's arraydescr_new_from_subarray: a subarray dtype is never attached to an array, so a base with
            // per-instance state (a finalize slot: StringDType) could never be finalized — refused by the DType class's
            // tp_name, right after the shape converted and before the dimensions are validated ('134217728T' included).
            // NumPy 2.4 built the subarray.
            if (type.Descr?.Meta is VStringDTypeMeta)
                throw new TypeError("numpy.dtypes.StringDType is not currently supported within subarray dtypes.");

            /* validate and set shape */
            foreach (long dim in shape)
            {
                if (dim < 0)
                    throw new ValueError("invalid shape in fixed-type tuple: dimension smaller then zero.");
                if (dim > int.MaxValue)
                    throw new ValueError("invalid shape in fixed-type tuple: dimension does not fit into a C int.");
            }
            long items = OverflowMultiplyList(shape);
            bool overflowed = items < 0 || items > int.MaxValue || (long)type.ItemSize * items > int.MaxValue;
            if (overflowed)
                throw new ValueError("invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.");
            return new CommaDescr((int)(type.ItemSize * items), hasFields: false, SubArrayKind);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_IntpConverter</c> over a repeats value: an int is a one-dimensional shape, a tuple its items;
        ///     a value past <c>npy_intp</c> (the process's pointer width) or more than <see cref="NPY_MAXDIMS"/> items fails.
        /// </summary>
        /// <param name="value">The repeats.</param>
        /// <param name="shape">The dimensions, or null on failure.</param>
        /// <returns>False where NumPy's converter fails (the caller raises NumPy's replacement text).</returns>
        private static bool IntpConverter(CommaRepeats value, out long[] shape)
        {
            shape = null;
            if (value.IsTuple && value.Values.Length > NPY_MAXDIMS)
                return false;
            var dims = new long[value.Values.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                // dimension_from_scalar: PyArray_PyIntAsIntp raises OverflowError past npy_intp.
                BigInteger v = value.Values[i];
                if (v < (long)nint.MinValue || v > (long)nint.MaxValue)
                    return false;
                dims[i] = (long)v;
            }
            shape = dims;
            return true;
        }

        /// <summary>
        ///     NumPy's <c>PyArray_OverflowMultiplyList</c>: the product of the dimensions in <c>npy_intp</c>, read left to right,
        ///     0 as soon as a dimension is 0, -1 as soon as the running product overflows - so on a 64-bit host
        ///     <c>(2147483647, 2147483647, 2147483647, 0)</c> overflows before its 0 while <c>(2147483647, 2, 0)</c> is 0.
        /// </summary>
        /// <param name="dims">The dimensions (each already checked to be non-negative and within a C int).</param>
        /// <returns>The product, or -1 on overflow.</returns>
        private static long OverflowMultiplyList(long[] dims)
        {
            BigInteger prod = BigInteger.One;
            foreach (long dim in dims)
            {
                if (dim == 0)
                    return 0;
                prod *= dim;
                if (prod > (long)nint.MaxValue)
                    return -1;
            }
            return (long)prod;
        }

        /// <summary>
        ///     NumPy's <c>_convert_from_list</c> for <c>_commastring</c>'s field list: drop a trailing empty field (<c>'i4,&lt;'</c>
        ///     ends in one), then convert every field in order - so the first failing field raises - and build the structured
        ///     dtype, which NumSharp has no class for.
        /// </summary>
        /// <param name="items">The fields (at least one).</param>
        /// <returns>The structured dtype's facts (its item size is the sum of the fields').</returns>
        /// <exception cref="ValueError"><c>Expected at least one field name</c> (the only field was empty: <c>'\t,'</c>), or a
        ///     field's ValueError.</exception>
        /// <exception cref="TypeError">A field's dtype is not understood (an empty field before another: <c>'\t,i4'</c>).</exception>
        /// <exception cref="SyntaxError">A nested comma string's repeat group is not a Python literal.</exception>
        private static CommaDescr ConvertCommaList(List<CommaItem> items)
        {
            int n = items.Count;
            // Ignore any empty string at end which _internal._commastring can produce
            CommaItem last = items[n - 1];
            if (last.Repeats == null && last.Dtype.Length == 0)
                n--;
            if (n == 0)
                throw new ValueError("Expected at least one field name");

            // NumPy sums the fields' sizes in a C int without an overflow check; the sum is only reported here (a structured
            // descriptor is never the base of another conversion), so it wraps the same way.
            int totalsize = 0;
            for (int i = 0; i < n; i++)
            {
                CommaDescr conv = items[i].Repeats == null ? ConvertCommaItemString(items[i].Dtype) : ConvertCommaTuple(items[i]);
                totalsize = unchecked(totalsize + conv.ItemSize);
            }
            return new CommaDescr(totalsize, hasFields: true, StructuredKind);
        }
    }
}
