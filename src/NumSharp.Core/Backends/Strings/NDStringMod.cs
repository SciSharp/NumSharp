using System;
using System.Collections.Generic;

// =============================================================================
// NDStringMod.cs — the compiled format of np.strings.mod (strings phase 2, P2.6 milestone 3): CPython 3.12's parse of a %
// format, over the code points of a str element or the bytes of a bytes element, %(key) specs included
// =============================================================================
//
// CPYTHON'S PARSE (Objects/unicodeobject.c unicode_format_arg_parse, Objects/bytesobject.c _PyBytes_FormatEx)
//   Both formatters parse AS they format, so which error a malformed format raises depends on the argument: the parse of a
//   spec runs up to its conversion character, taking the arguments a '*' width or precision asks for on the way, and the
//   conversion character is checked only after its own argument was taken ('%s %y' % 5 is "not enough arguments for format
//   string", '%y %s' % 5 is "unsupported format character 'y' (0x79) at index 1"). The steps of one spec, after its '%':
//     1. '%' again: a literal percent sign, no argument.
//     2. '(' : a mapping key up to the balancing ')' (nested parentheses counted). Without a mapping argument the parse
//        raises TypeError "format requires a mapping" before reading the key; a string that ends inside the key is
//        ValueError "incomplete format key"; otherwise the key is looked up at once (the lookup's own error wins over
//        anything later in the spec) and the result becomes the argument the rest of the spec takes.
//     3. Flags '-', '+', ' ', '#', '0', any number, any order.
//     4. The width: '*' (the next argument, which must be a Python int: "* wants int"; negative left-justifies) or digits,
//        each one checked against PY_SSIZE_T_MAX before it is accumulated (ValueError "width too big").
//     5. The precision: '.' then '*' (a Python int; negative means 0) or digits checked against INT_MAX ("precision too
//        big"), or nothing (precision 0).
//     6. ONE length modifier 'h' / 'l' / 'L', skipped ('%ld' is '%d'; '%lld' converts with 'l', unsupported).
//     7. The end of the string here is ValueError "incomplete format" (after the '*' arguments were taken).
//     8. The conversion character (a code point for str, a byte for bytes), checked when the spec runs.
//   The bytes formatter has the same steps over bytes; its key is a bytes object and its error index counts bytes.
//
// HOW IT IS USED HERE
//   ModFormat.Compile records the steps of every spec once per distinct format element (np.strings.mod's format is most
//   often ONE string broadcast over the values, so the parse runs once per call); a spec that ends in a parse error is
//   the last step, carrying its failure and the '*' flags read before it, and the runner (NDCharObjectPath.Mod.cs) raises
//   each error at CPython's point. A text step records a range of the element's units, which the runner copies from the
//   element it formats (the compiled steps hold no text: one compiled format serves every equal element). FastSpec marks a
//   format of one positional argument, which the runner formats without CPython's argument state
//   (NDCharObjectPath.ModFast.cs).
// =============================================================================

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     One <c>%</c> format compiled for <c>np.strings.mod</c>: CPython's parse of each conversion spec (see the file
    ///     header), recorded over the units of a format element without looking at an argument.
    /// </summary>
    /// <remarks>
    ///     An instance holds unit RANGES, not text: the runner reads the text and the keys from the element it formats, so a
    ///     format compiled once serves every element with the same units. Immutable after construction; the walk that
    ///     compiles it is single-threaded, but nothing prevents sharing.
    /// </remarks>
    internal sealed class ModFormat
    {
        /// <summary>What one step does.</summary>
        internal enum OpKind : byte
        {
            /// <summary>Copies a run of the format's own units.</summary>
            Text,

            /// <summary>Writes the <c>%</c> of a <c>%%</c> (no argument).</summary>
            Percent,

            /// <summary>Converts one argument (a conversion spec), or raises the parse error the spec ends with.</summary>
            Spec,
        }

        /// <summary>The parse error a spec ends with: CPython raises it when the spec's parse reaches that point.</summary>
        internal enum ParseFailure : byte
        {
            /// <summary>The spec parsed completely.</summary>
            None,

            /// <summary>The string ended inside a <c>%(key)</c>: <c>ValueError: incomplete format key</c> (raised after the
            ///     mapping check).</summary>
            IncompleteKey,

            /// <summary>The width's digits exceed <c>PY_SSIZE_T_MAX</c>: <c>ValueError: width too big</c>.</summary>
            WidthTooBig,

            /// <summary>The precision's digits exceed <c>INT_MAX</c>: <c>ValueError: precision too big</c>.</summary>
            PrecisionTooBig,

            /// <summary>The string ended before the conversion character: <c>ValueError: incomplete format</c>.</summary>
            Incomplete,
        }

        /// <summary>One step of a compiled format.</summary>
        internal readonly struct Op
        {
            /// <summary>What the step does.</summary>
            internal readonly OpKind Kind;

            /// <summary>A text step's first unit, or a key spec's first key unit.</summary>
            internal readonly int Start;

            /// <summary>A text step's number of units, or a key spec's number of key units.</summary>
            internal readonly int Length;

            /// <summary>The spec starts with a <c>%(key)</c> (<see cref="Start"/> / <see cref="Length"/> give the key).</summary>
            internal readonly bool HasKey;

            /// <summary>The <c>-</c> flag (left-justify).</summary>
            internal readonly bool Left;

            /// <summary>The <c>+</c> flag.</summary>
            internal readonly bool Plus;

            /// <summary>The <c> </c> flag.</summary>
            internal readonly bool Space;

            /// <summary>The <c>0</c> flag.</summary>
            internal readonly bool Zero;

            /// <summary>The <c>#</c> flag.</summary>
            internal readonly bool Alt;

            /// <summary>The width is a <c>*</c>: the next argument (taken before a later parse error).</summary>
            internal readonly bool WidthStar;

            /// <summary>The precision is a <c>*</c>: the next argument after the width (taken before a later parse error).</summary>
            internal readonly bool PrecisionStar;

            /// <summary>The written width, or -1 for none (also when it is a <c>*</c>).</summary>
            internal readonly long Width;

            /// <summary>The written precision (0 for a bare <c>.</c>), or -1 for none (also when it is a <c>*</c>).</summary>
            internal readonly int Precision;

            /// <summary>The conversion unit (a code point of a str format, a byte value 0–255 of a bytes format), or -1 when
            ///     the spec ends in a parse error first.</summary>
            internal readonly int Conversion;

            /// <summary>The conversion unit's index in the format's units (code points for str, bytes for bytes): the index
            ///     the unsupported-character error reports.</summary>
            internal readonly int ConversionIndex;

            /// <summary>The parse error the spec ends with.</summary>
            internal readonly ParseFailure Failure;

            /// <summary>Builds a step; see the fields.</summary>
            /// <param name="kind">What the step does.</param>
            /// <param name="start">A text step's first unit, or a key's first unit.</param>
            /// <param name="length">A text step's unit count, or a key's unit count.</param>
            /// <param name="hasKey">Whether the spec has a key.</param>
            /// <param name="left">The <c>-</c> flag.</param>
            /// <param name="plus">The <c>+</c> flag.</param>
            /// <param name="space">The <c> </c> flag.</param>
            /// <param name="zero">The <c>0</c> flag.</param>
            /// <param name="alt">The <c>#</c> flag.</param>
            /// <param name="widthStar">A <c>*</c> width.</param>
            /// <param name="width">The written width (-1 for none).</param>
            /// <param name="precisionStar">A <c>*</c> precision.</param>
            /// <param name="precision">The written precision (-1 for none).</param>
            /// <param name="conversion">The conversion unit (-1 for none).</param>
            /// <param name="conversionIndex">Its unit index.</param>
            /// <param name="failure">The parse error the spec ends with.</param>
            internal Op(OpKind kind, int start, int length, bool hasKey, bool left, bool plus, bool space, bool zero,
                bool alt, bool widthStar, long width, bool precisionStar, int precision, int conversion,
                int conversionIndex, ParseFailure failure)
            {
                Kind = kind;
                Start = start;
                Length = length;
                HasKey = hasKey;
                Left = left;
                Plus = plus;
                Space = space;
                Zero = zero;
                Alt = alt;
                WidthStar = widthStar;
                Width = width;
                PrecisionStar = precisionStar;
                Precision = precision;
                Conversion = conversion;
                ConversionIndex = conversionIndex;
                Failure = failure;
            }
        }

        /// <summary>The compiled steps, in order; a step with a parse failure is the last.</summary>
        internal readonly Op[] Ops;

        /// <summary>The number of conversion specs (steps of kind <see cref="OpKind.Spec"/>).</summary>
        internal readonly int SpecCount;

        /// <summary>The number of units the format was compiled from.</summary>
        internal readonly int UnitCount;

        /// <summary>
        ///     The index in <see cref="Ops"/> of the format's only spec when the format takes exactly ONE argument by
        ///     position — one complete spec, no <c>%(key)</c>, no <c>*</c> width or precision — else -1. Such a format
        ///     consumes a non-tuple value whole whatever its type, so no argument count or mapping rule can raise, and the
        ///     runner formats it without CPython's argument machinery (the fast element path).
        /// </summary>
        internal readonly int FastSpec = -1;

        /// <summary>Wraps compiled steps.</summary>
        /// <param name="ops">The steps.</param>
        /// <param name="unitCount">The format's unit count.</param>
        private ModFormat(Op[] ops, int unitCount)
        {
            Ops = ops;
            UnitCount = unitCount;
            int last = -1;
            for (int i = 0; i < ops.Length; i++)
            {
                if (ops[i].Kind != OpKind.Spec)
                    continue;
                SpecCount++;
                last = i;
            }
            if (SpecCount == 1)
            {
                ref readonly var spec = ref ops[last];
                if (spec.Failure == ParseFailure.None && !spec.HasKey && !spec.WidthStar && !spec.PrecisionStar)
                    FastSpec = last;
            }
        }

        /// <summary>Compiles the code points of a str format (a <c>U</c> element without its trailing NULs, a
        ///     <c>StringDType</c> text decoded, a Python str).</summary>
        /// <param name="units">The code points (any value; a unit past U+10FFFF is an ordinary unit of the parse).</param>
        /// <param name="count">Their number.</param>
        /// <returns>The compiled format.</returns>
        /// <exception cref="OverflowException">More than <see cref="int.MaxValue"/> units.</exception>
        internal static unsafe ModFormat Compile(uint* units, long count) => Parse(units, checked((int)count));

        /// <summary>Compiles the bytes of a bytes format (an <c>S</c> element without its trailing NULs).</summary>
        /// <param name="bytes">The bytes.</param>
        /// <param name="count">Their number.</param>
        /// <returns>The compiled format.</returns>
        /// <exception cref="OverflowException">More than <see cref="int.MaxValue"/> bytes.</exception>
        internal static unsafe ModFormat Compile(byte* bytes, long count) => Parse(bytes, checked((int)count));

        /// <summary>
        ///     The parse (see the file header) over either unit width: literal runs, <c>%%</c>, and each spec's key,
        ///     flags, width, precision, length modifier and conversion unit, stopping after the first spec that ends in a
        ///     parse error (CPython never reads past it).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="uint"/> (str) or <see cref="byte"/> (bytes).</typeparam>
        /// <param name="s">The units.</param>
        /// <param name="n">Their number.</param>
        /// <returns>The compiled format.</returns>
        private static unsafe ModFormat Parse<TUnit>(TUnit* s, int n) where TUnit : unmanaged
        {
            var ops = new List<Op>();
            int i = 0;
            while (i < n)
            {
                if (U(s, i) != '%')
                {
                    // A literal run up to the next '%'.
                    int start = i;
                    while (i < n && U(s, i) != '%')
                        i++;
                    ops.Add(new Op(OpKind.Text, start, i - start, false, false, false, false, false, false, false, -1,
                        false, -1, -1, -1, ParseFailure.None));
                    continue;
                }

                // The unit after the '%'. A second '%' is a literal percent sign; the end of the string reads CPython's NUL
                // terminator, which is not one and fails the spec as "incomplete format" below.
                i++;
                if (i < n && U(s, i) == '%')
                {
                    ops.Add(new Op(OpKind.Percent, 0, 0, false, false, false, false, false, false, false, -1, false, -1,
                        -1, -1, ParseFailure.None));
                    i++;
                    continue;
                }

                if (!ParseSpec(s, n, ref i, out var spec))
                {
                    ops.Add(spec);
                    break;
                }
                ops.Add(spec);
            }
            return new ModFormat(ops.ToArray(), n);
        }

        /// <summary>One unit's value: a byte 0–255, or a code point.</summary>
        /// <typeparam name="TUnit"><see cref="uint"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The units.</param>
        /// <param name="i">The index.</param>
        /// <returns>The value.</returns>
        private static unsafe uint U<TUnit>(TUnit* s, int i) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? ((byte*)s)[i] : ((uint*)s)[i];

        /// <summary>Whether a unit is an ASCII decimal digit (CPython compares the value with <c>'0'</c> and <c>'9'</c>;
        ///     the bytes formatter's <c>isdigit</c> in the C locale is the same set).</summary>
        /// <param name="c">The unit's value.</param>
        /// <returns>True for <c>0</c>–<c>9</c>.</returns>
        private static bool IsDigit(uint c) => c - '0' <= 9;

        /// <summary>
        ///     Parses one conversion spec after its <c>%</c> (steps 2–8 of the file header).
        /// </summary>
        /// <typeparam name="TUnit"><see cref="uint"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The units.</param>
        /// <param name="n">Their number.</param>
        /// <param name="i">In: the index just after the <c>%</c>. Out: the index after the conversion unit (or where the
        ///     parse failed).</param>
        /// <param name="spec">The parsed spec, or the spec carrying its parse failure.</param>
        /// <returns>False when the spec ends in a parse error (compiling stops after it).</returns>
        private static unsafe bool ParseSpec<TUnit>(TUnit* s, int n, ref int i, out Op spec) where TUnit : unmanaged
        {
            // "%(key)": the key runs to the ')' that balances the '('. Whether there is a mapping to look it up in is the
            // runner's question (it raises "format requires a mapping" before this failure, as CPython does).
            bool hasKey = false;
            int keyStart = 0, keyLength = 0;
            if (i < n && U(s, i) == '(')
            {
                hasKey = true;
                i++;
                keyStart = i;
                int depth = 1;
                while (depth > 0 && i < n)
                {
                    uint k = U(s, i);
                    if (k == ')')
                        depth--;
                    else if (k == '(')
                        depth++;
                    i++;
                }
                if (depth > 0)
                {
                    spec = Failed(true, keyStart, i - keyStart, false, false, false, false, false, false, -1, false, -1,
                        ParseFailure.IncompleteKey);
                    return false;
                }
                keyLength = i - keyStart - 1;
            }

            // Flags, in any order and any number ('0' is always a flag, so a width of digits never starts with 0).
            bool left = false, plus = false, space = false, zero = false, alt = false;
            for (; i < n; i++)
            {
                uint c = U(s, i);
                if (c == '-') left = true;
                else if (c == '+') plus = true;
                else if (c == ' ') space = true;
                else if (c == '#') alt = true;
                else if (c == '0') zero = true;
                else break;
            }

            // The width: '*' (an argument, taken at run time) or digits, each checked against PY_SSIZE_T_MAX before it is
            // accumulated — CPython's overflow test.
            bool widthStar = false;
            long width = -1;
            if (i < n && U(s, i) == '*')
            {
                widthStar = true;
                i++;
            }
            else if (i < n && IsDigit(U(s, i)))
            {
                width = U(s, i) - '0';
                i++;
                while (i < n && IsDigit(U(s, i)))
                {
                    long d = U(s, i) - '0';
                    if (width > (long.MaxValue - d) / 10)
                    {
                        spec = Failed(hasKey, keyStart, keyLength, left, plus, space, zero, alt, false, -1, false, -1,
                            ParseFailure.WidthTooBig);
                        return false;
                    }
                    width = width * 10 + d;
                    i++;
                }
            }

            // The precision: '.' then '*' or digits (none is precision 0), the digits checked against INT_MAX.
            bool precisionStar = false;
            int precision = -1;
            if (i < n && U(s, i) == '.')
            {
                precision = 0;
                i++;
                if (i < n && U(s, i) == '*')
                {
                    precisionStar = true;
                    i++;
                }
                else if (i < n && IsDigit(U(s, i)))
                {
                    precision = (int)(U(s, i) - '0');
                    i++;
                    while (i < n && IsDigit(U(s, i)))
                    {
                        int d = (int)(U(s, i) - '0');
                        if (precision > (int.MaxValue - d) / 10)
                        {
                            spec = Failed(hasKey, keyStart, keyLength, left, plus, space, zero, alt, widthStar, width,
                                false, -1, ParseFailure.PrecisionTooBig);
                            return false;
                        }
                        precision = precision * 10 + d;
                        i++;
                    }
                }
            }

            // ONE length modifier is skipped ("%ld" is "%d"); a second one is the conversion unit ("%lld" is 'l').
            if (i < n && U(s, i) is 'h' or 'l' or 'L')
                i++;

            if (i >= n)
            {
                spec = Failed(hasKey, keyStart, keyLength, left, plus, space, zero, alt, widthStar, width, precisionStar,
                    precision, ParseFailure.Incomplete);
                return false;
            }

            // The conversion unit. Whether it is one the formatter supports is decided when the spec runs, after its
            // argument is taken.
            spec = new Op(OpKind.Spec, keyStart, keyLength, hasKey, left, plus, space, zero, alt, widthStar, width,
                precisionStar, precision, (int)U(s, i), i, ParseFailure.None);
            i++;
            return true;
        }

        /// <summary>A spec step that ends in a parse error.</summary>
        /// <param name="hasKey">Whether a key was read (its lookup runs before the error).</param>
        /// <param name="keyStart">The key's first unit.</param>
        /// <param name="keyLength">The key's unit count.</param>
        /// <param name="left">The <c>-</c> flag.</param>
        /// <param name="plus">The <c>+</c> flag.</param>
        /// <param name="space">The <c> </c> flag.</param>
        /// <param name="zero">The <c>0</c> flag.</param>
        /// <param name="alt">The <c>#</c> flag.</param>
        /// <param name="widthStar">Whether a <c>*</c> width was read before the error (its argument is taken first).</param>
        /// <param name="width">The written width.</param>
        /// <param name="precisionStar">Whether a <c>*</c> precision was read before the error.</param>
        /// <param name="precision">The written precision.</param>
        /// <param name="failure">The error.</param>
        /// <returns>The step.</returns>
        private static Op Failed(bool hasKey, int keyStart, int keyLength, bool left, bool plus, bool space, bool zero,
            bool alt, bool widthStar, long width, bool precisionStar, int precision, ParseFailure failure)
            => new Op(OpKind.Spec, keyStart, keyLength, hasKey, left, plus, space, zero, alt, widthStar, width,
                precisionStar, precision, -1, -1, failure);
    }
}
