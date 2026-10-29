using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Translates slice expressions to concatenation along an axis — the machinery shared by
        ///     <see cref="r_"/> and <see cref="c_"/>.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.lib.index_tricks.AxisConcatenator</c>. See <see cref="RClass"/>
        ///     for the full usage documentation.
        ///     <para>
        ///     <b>How Python's <c>a[1:5:2]</c> is spelled in C#.</b> C# has no slice literal, so a slice is
        ///     written as a string: <c>np.r_["1:5:2"]</c>. Strings do double duty here, exactly as in NumPy,
        ///     and are told apart by a colon — <b>a string containing <c>':'</c> is a slice expression, a
        ///     string without one is a NumPy special directive</b> (<c>"r"</c>, <c>"c"</c>, <c>"-1"</c>,
        ///     <c>"0,2"</c>, <c>"1,2,0"</c>). The two grammars are disjoint in NumPy too — a directive never
        ///     contains a colon because Python's own syntax supplies the slices — so every NumPy expression
        ///     transcribes verbatim. A slice string may hold several comma-separated slices
        ///     (<c>np.r_["1:3, 5:8"]</c> ≙ <c>np.r_[1:3, 5:8]</c>), and a <see cref="Slice"/> or
        ///     <see cref="Slice"/>[] object works too (so <c>np.s_[…]</c> composes).
        ///     </para>
        ///     <para>
        ///     <b>Weak scalars (NEP50).</b> NumPy distinguishes a Python literal (<c>5</c>, weak — adopts the
        ///     other operand's dtype) from a NumPy scalar (<c>np.int64(5)</c>, strong). C# has no such split,
        ///     so NumSharp maps it by type: <c>bool</c> / the eight integer primitives / <c>float</c> /
        ///     <c>double</c> / <see cref="Complex"/> are <b>weak</b> (the C# literal is the Python literal),
        ///     while <c>char</c>, <see cref="Half"/> and <c>decimal</c> — which have no Python literal — plus
        ///     every <see cref="NDArray"/> (including 0-d) are <b>strong</b>. To force a strong scalar, wrap
        ///     it: <c>np.r_[int8Array, NDArray.Scalar(1L)]</c> yields int64 where <c>np.r_[int8Array, 1L]</c>
        ///     yields int8. As in NumPy, a weak integer that does not fit the resolved dtype raises
        ///     <see cref="OverflowException"/> rather than wrapping.
        ///     </para>
        ///     <para>
        ///     <b>Divergence — no <c>bmat</c> branch.</b> NumPy routes a single bare string key into
        ///     <c>matrixlib.bmat</c>, which resolves the words in it against the CALLER'S Python frame
        ///     (<c>sys._getframe().f_back</c>) as variable names. C# has no equivalent, and the branch raises
        ///     <c>NameError</c> for every string literal anyway (<c>np.r_['1 2; 3 4']</c> → "name '1' is not
        ///     defined"), so NumSharp gives a lone string the same reading as any other entry.
        ///     </para>
        /// </remarks>
        public abstract class AxisConcatenator
        {
            /// <summary>Axis to concatenate along (NumPy's <c>axis</c>).</summary>
            protected readonly int _axis;

            /// <summary>Whether the result is coerced to a 2-D "matrix" (NumPy's <c>matrix</c>).</summary>
            protected readonly bool _matrix;

            /// <summary>Minimum dimensionality forced onto each entry (NumPy's <c>ndmin</c>).</summary>
            protected readonly int _ndmin;

            /// <summary>Where upgraded entries put their original last axis (NumPy's <c>trans1d</c>).</summary>
            protected readonly int _trans1d;

            internal AxisConcatenator(int axis = 0, bool matrix = false, int ndmin = 1, int trans1d = -1)
            {
                _axis = axis;
                _matrix = matrix;
                _ndmin = ndmin;
                _trans1d = trans1d;
            }

            /// <summary>
            ///     Expands the index expression and concatenates the pieces.
            /// </summary>
            /// <param name="key">
            ///     Slice-expression strings, special-directive strings, <see cref="Slice"/> objects,
            ///     <see cref="NDArray"/>s, C# scalars, arrays and collections — in any mix.
            /// </param>
            public NDArray this[params object[] key]
                // A numpy.matrix entry: the joined result comes back as a matrix, as np.concatenate returns it in
                // NumPy. The function name is spelled out for the caller audit (an indexer's caller name is "Item",
                // its frame is get_Item).
                => NDMatrixBoundary.SeesNested(key, "get_Item")
                ? NDMatrixBoundary.Wrapped((self: this, key), static s => s.self[s.key])
                : NDArray.AxisConcatenate(key, _axis, _matrix, _ndmin, _trans1d);

            /// <summary>NumPy's <c>__len__</c>, which is 0 for these objects.</summary>
            public int Count => 0;

            #region slice expressions

            /// <summary>
            ///     A parsed <c>start:stop:step</c> expression, each field the Python number NumPy computes with. Unlike
            ///     <see cref="Slice"/> (which indexes and is integer-only) this carries real bounds and NumPy's imaginary
            ///     step, where <c>start:stop:Nj</c> means "N points, stop inclusive" (<c>linspace</c> in <c>np.r_</c>,
            ///     <c>arange(0, N) * step + start</c> in <c>np.mgrid</c> / <c>np.ogrid</c>).
            /// </summary>
            /// <remarks>
            ///     A field written as digits (or omitted: start 0, step 1) is a Python INT, held exactly in
            ///     <see cref="IntStart"/> / <see cref="IntStop"/> / <see cref="IntStep"/>; any other field (a point, an exponent,
            ///     <c>Infinity</c>, <c>NaN</c>) is a Python float. <see cref="Start"/> / <see cref="Stop"/> / <see cref="Step"/>
            ///     hold every field as a double: an int field correctly rounded (CPython's <c>float(int)</c>, but ±inf past the
            ///     double range where CPython raises, so a computation that can meet such a field converts the int value). The
            ///     int values matter past 2**53, where a double no longer holds every integer: NumPy's lengths and int64 values
            ///     come from Python's exact int arithmetic (<c>np.r_[2**53 + 1:2**53 + 3]</c> has two elements; a double
            ///     reading of the bounds made four).
            /// </remarks>
            internal readonly struct SliceSpec
            {
                /// <summary>The start as a double (0 when omitted).</summary>
                public readonly double Start;

                /// <summary>The stop as a double (0 when omitted: see <see cref="HasStop"/>).</summary>
                public readonly double Stop;

                /// <summary>The step as a double (1 when omitted); for an imaginary step, its imaginary part (a point count).</summary>
                public readonly double Step;

                /// <summary>The start's exact value when <see cref="StartIsInt"/> (0 when omitted).</summary>
                public readonly BigInteger IntStart;

                /// <summary>The stop's exact value when <see cref="StopIsInt"/> and <see cref="HasStop"/>.</summary>
                public readonly BigInteger IntStop;

                /// <summary>The step's exact value when <see cref="StepIsInt"/> (1 when omitted).</summary>
                public readonly BigInteger IntStep;

                /// <summary>The start is a Python int (written as digits, or omitted).</summary>
                public readonly bool StartIsInt;

                /// <summary>
                ///     The stop is a Python int, or omitted: Python's None never meets arithmetic as a number (arange moves the
                ///     start into the stop slot, nd_grid raises), so it does not make the expression a float one.
                /// </summary>
                public readonly bool StopIsInt;

                /// <summary>The step is a Python int (written as digits, or omitted); false for an imaginary step.</summary>
                public readonly bool StepIsInt;

                /// <summary>False for <c>a:</c> and <c>:</c>, where NumPy's arange re-reads start as stop.</summary>
                public readonly bool HasStop;

                /// <summary>The step was written <c>Nj</c>: <see cref="Step"/> is a point COUNT.</summary>
                public readonly bool ImaginaryStep;

                /// <summary>
                ///     Every field is a Python int and the step is real: NumPy's arange discovers int64 for it, and its
                ///     <c>(stop - start) / step</c> is Python's int division (<c>division by zero</c> for a zero step).
                /// </summary>
                public readonly bool Integral;

                /// <summary>Builds a slice from its fields, each a double and, for a Python int, its exact value.</summary>
                /// <param name="start">The start as a double.</param>
                /// <param name="intStart">The start's exact value when it is a Python int, else null (a float).</param>
                /// <param name="stop">The stop as a double (any value when <paramref name="hasStop"/> is false).</param>
                /// <param name="intStop">The stop's exact value when it is a Python int, else null.</param>
                /// <param name="hasStop">Whether the stop was written.</param>
                /// <param name="step">The step as a double (an imaginary step's imaginary part).</param>
                /// <param name="intStep">The step's exact value when it is a Python int, else null (a float or imaginary step).</param>
                /// <param name="imaginaryStep">Whether the step was written <c>Nj</c>.</param>
                public SliceSpec(double start, BigInteger? intStart, double stop, BigInteger? intStop, bool hasStop,
                    double step, BigInteger? intStep, bool imaginaryStep)
                {
                    Start = start;
                    Stop = stop;
                    Step = step;
                    IntStart = intStart ?? BigInteger.Zero;
                    IntStop = intStop ?? BigInteger.Zero;
                    IntStep = intStep ?? BigInteger.One;
                    StartIsInt = intStart.HasValue;
                    StopIsInt = !hasStop || intStop.HasValue;
                    StepIsInt = intStep.HasValue && !imaginaryStep;
                    HasStop = hasStop;
                    ImaginaryStep = imaginaryStep;
                    Integral = StartIsInt && StopIsInt && StepIsInt;
                }

                /// <summary>Adapts a NumSharp <see cref="Slice"/> (Python ints, never an imaginary step).</summary>
                /// <param name="slice">The slice.</param>
                /// <returns>The slice's fields: start 0 when absent, the stop only when present.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="slice"/> is null.</exception>
                /// <exception cref="ArgumentException"><paramref name="slice"/> is <c>...</c> or <c>newaxis</c>.</exception>
                public static SliceSpec FromSlice(Slice slice)
                {
                    if (slice is null)
                        throw new ArgumentNullException(nameof(slice));
                    if (slice.IsEllipsis || slice.IsNewAxis)
                        throw new ArgumentException(
                            "'...' and 'newaxis' are index-only and cannot appear in a concatenation expression.",
                            nameof(slice));

                    long start = slice.Start ?? 0;
                    long stop = slice.Stop ?? 0;
                    return new SliceSpec(start, start, stop, slice.Stop.HasValue ? (BigInteger?)stop : null, slice.Stop.HasValue,
                        slice.Step, slice.Step, false);
                }

                /// <summary>
                ///     NumPy's slice branch of <c>np.r_</c> / <c>np.c_</c> (and <c>np.mgrid</c> / <c>np.ogrid</c>'s single real
                ///     slice): <c>arange(start, stop, step)</c> over the Python numbers, or <c>linspace(start, stop, N)</c> for
                ///     an imaginary step.
                /// </summary>
                /// <returns>A fresh 1-D array: int64 when every field is a Python int, else float64.</returns>
                /// <exception cref="ValueError">An imaginary step without a stop; arange's NaN length or length past
                ///     npy_intp; the allocation guard's <c>array is too big</c>.</exception>
                /// <exception cref="DivideByZeroException">A zero step: Python's <c>division by zero</c> for ints,
                ///     <c>float division by zero</c> once a float is involved.</exception>
                public NDArray Materialize()
                {
                    if (ImaginaryStep)
                    {
                        if (!HasStop)
                            throw new ValueError(
                                "an imaginary step requires a stop value: 'start:stop:Nj'.");
                        // NumPy: size = int(abs(step)); linspace(start, stop, num=size).
                        return linspace(Start, Stop, (long)Math.Abs(Step), true, NPTypeCode.Double);
                    }

                    // Python ints within int64 take arange's int route (PyArray_ArangeObj over Python ints): the exact length,
                    // exact int64 values past 2**53, and Python's int division, so a zero step is "division by zero".
                    if (Integral && FitsInt64(IntStart) && FitsInt64(IntStop) && FitsInt64(IntStep))
                    {
                        // NumPy's arange(start, None, step) is arange(0, start, step): a missing stop promotes start into
                        // the stop slot. np.r_[5::2] is therefore [0, 2, 4].
                        return HasStop
                            ? arange((long)IntStart, (long)IntStop, (long)IntStep)
                            : arange(0L, (long)IntStart, (long)IntStep);
                    }

                    // A float field makes arange's float route (a zero step is "float division by zero"). A Python int past
                    // int64 is read as its double here too, NumSharp's limit: NumPy discovers uint64 or object for it.
                    var code = Integral ? NPTypeCode.Int64 : NPTypeCode.Double;
                    return HasStop
                        ? arange(Start, Stop, Step, code)
                        : arange(0d, Start, Step, code);
                }

                /// <summary>Whether a Python int fits an int64.</summary>
                /// <param name="value">The int.</param>
                /// <returns>True within <c>[-2**63, 2**63 - 1]</c>.</returns>
                internal static bool FitsInt64(BigInteger value) => value >= long.MinValue && value <= long.MaxValue;
            }

            /// <summary>Parses one <c>start:stop[:step]</c> token.</summary>
            /// <param name="token">The token: two or three colon-separated fields, any of which may be empty.</param>
            /// <returns>The slice, each field the Python number it spells.</returns>
            /// <exception cref="ArgumentException">Fewer than two or more than three fields, or a field that is no number
            ///     (<c>Invalid slice notation: '...'</c>).</exception>
            internal static SliceSpec ParseSliceToken(string token)
            {
                var fields = token.Split(':');
                if (fields.Length < 2 || fields.Length > 3)
                    throw new ArgumentException($"Invalid slice notation: '{token.Trim()}'", nameof(token));

                double start = 0d;
                BigInteger? intStart = BigInteger.Zero;
                if (!string.IsNullOrWhiteSpace(fields[0]))
                    start = ParseNumber(fields[0], token, out intStart, out _, allowImaginary: false);

                bool hasStop = !string.IsNullOrWhiteSpace(fields[1]);
                double stop = 0d;
                BigInteger? intStop = null;
                if (hasStop)
                    stop = ParseNumber(fields[1], token, out intStop, out _, allowImaginary: false);

                double step = 1d;
                BigInteger? intStep = BigInteger.One;
                bool imaginary = false;
                if (fields.Length == 3 && !string.IsNullOrWhiteSpace(fields[2]))
                    step = ParseNumber(fields[2], token, out intStep, out imaginary, allowImaginary: true);

                return new SliceSpec(start, intStart, stop, intStop, hasStop, step, intStep, imaginary);
            }

            /// <summary>
            ///     Reads one numeric field as the Python literal it spells. Python's LITERAL rule decides the type, not the
            ///     value: digits with an optional sign are an int (<c>0</c>, <c>-5</c>, kept exactly at any size), anything
            ///     else that parses (<c>0.0</c>, <c>1e3</c>, <c>Infinity</c>, <c>NaN</c>) is a float — which is what makes
            ///     <c>np.r_["0:5"]</c> int64 while <c>np.r_["0.0:5"]</c> is float64.
            /// </summary>
            /// <param name="field">The field's text.</param>
            /// <param name="token">The whole token, for the error text.</param>
            /// <param name="asInt">The field's exact value when it is a Python int, else null.</param>
            /// <param name="imaginary">Whether a trailing <c>j</c> made the field imaginary (only when
            ///     <paramref name="allowImaginary"/>).</param>
            /// <param name="allowImaginary">Whether the field may end in <c>j</c> (the step).</param>
            /// <returns>The field as a double (correctly rounded; ±inf past the double range).</returns>
            /// <exception cref="ArgumentException">The field is no number (<c>Invalid slice notation: '...'</c>).</exception>
            private static double ParseNumber(string field, string token, out BigInteger? asInt,
                out bool imaginary, bool allowImaginary)
            {
                string f = field.Trim();
                imaginary = false;

                if (allowImaginary && f.Length > 0 && (f[f.Length - 1] == 'j' || f[f.Length - 1] == 'J'))
                {
                    imaginary = true;
                    f = f.Substring(0, f.Length - 1);
                    if (f.Length == 0 || f == "+")
                        f = "1";
                    else if (f == "-")
                        f = "-1";
                }

                // Digits with an optional sign are a Python int, kept exactly (a double rounds the integers past 2**53); its
                // double is float(int), correctly rounded. Within int64 the scan is the whole parse: a C# long -> double
                // conversion rounds to nearest, ties to even, as float() does - and "-0" is the int 0, so +0.0.
                switch (ScanIntLiteral(f, out long small))
                {
                    case IntLiteral.Int64:
                        asInt = small;
                        return small;
                    case IntLiteral.Big:
                        asInt = BigInteger.Parse(f, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
                        // Correctly rounded too; ±inf past the double range (where CPython's float(int) raises: the
                        // computations that can meet such a field convert the int value instead).
                        return double.Parse(f, NumberStyles.Float, CultureInfo.InvariantCulture);
                }

                if (!double.TryParse(f, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                    throw new ArgumentException($"Invalid slice notation: '{token.Trim()}'", nameof(token));
                asInt = null;
                return value;
            }

            /// <summary>What <see cref="ScanIntLiteral"/> found in a field.</summary>
            private enum IntLiteral
            {
                /// <summary>Not a Python int literal (a float, or no number at all).</summary>
                None,

                /// <summary>An int literal whose value fits int64.</summary>
                Int64,

                /// <summary>An int literal past int64.</summary>
                Big,
            }

            /// <summary>
            ///     Scans a trimmed field for a Python int literal as NumSharp spells one — an optional sign, then ASCII digits
            ///     only — and reads its value when it fits int64, in the same pass. Every float field (<c>0.5</c>,
            ///     <c>1e3</c>, <c>Infinity</c>) is told apart without a parse attempt.
            /// </summary>
            /// <param name="f">The trimmed field.</param>
            /// <param name="value">The value when the result is <see cref="IntLiteral.Int64"/>, else 0.</param>
            /// <returns>The kind of field.</returns>
            private static IntLiteral ScanIntLiteral(string f, out long value)
            {
                value = 0;
                int i = 0;
                bool negative = false;
                if (f.Length > 0 && (f[0] == '+' || f[0] == '-'))
                {
                    negative = f[0] == '-';
                    i = 1;
                }

                if (i == f.Length)
                    return IntLiteral.None;

                // The magnitude accumulates in a ulong until it would overflow; the digits after that only need checking.
                ulong magnitude = 0;
                bool big = false;
                for (; i < f.Length; i++)
                {
                    uint digit = (uint)(f[i] - '0');
                    if (digit > 9)
                        return IntLiteral.None;
                    if (big)
                        continue;
                    if (magnitude > (ulong.MaxValue - digit) / 10)
                        big = true;
                    else
                        magnitude = magnitude * 10 + digit;
                }

                if (big)
                    return IntLiteral.Big;
                if (negative)
                {
                    // -2**63 is an int64 too: its magnitude's unchecked negation is long.MinValue.
                    if (magnitude > 9223372036854775808UL)
                        return IntLiteral.Big;
                    value = unchecked(-(long)magnitude);
                    return IntLiteral.Int64;
                }

                if (magnitude > long.MaxValue)
                    return IntLiteral.Big;
                value = (long)magnitude;
                return IntLiteral.Int64;
            }

            #endregion
        }

        /// <summary>
        ///     Translates slice expressions to concatenation along the FIRST axis. Two use cases:
        ///     comma-separated arrays are stacked along axis 0, and slice notation or scalars build a 1-D
        ///     array.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.r_</c>. Slice notation <c>"start:stop:step"</c> is
        ///     <c>np.arange(start, stop, step)</c>; an imaginary step (<c>"start:stop:Nj"</c>) is
        ///     <c>np.linspace(start, stop, N)</c> with the stop INCLUSIVE. After expansion everything is
        ///     concatenated.
        ///     <para>
        ///     An optional leading directive string changes the output. <c>"r"</c> / <c>"c"</c> coerce to a
        ///     2-D matrix (a 1-D result becomes 1×N for <c>"r"</c> and N×1 for <c>"c"</c>; a 2-D result is
        ///     unchanged). An integer string selects the axis to stack along. Two comma-separated integers
        ///     also set the minimum dimensionality each entry is forced to; a third says which axis the
        ///     upgraded entries' original axes should start at (default <c>-1</c>, i.e. the 1s are prepended).
        ///     </para>
        ///     See <see cref="AxisConcatenator"/> for how slices are spelled in C# and how weak scalars are
        ///     mapped. https://numpy.org/doc/stable/reference/generated/numpy.r_.html
        /// </remarks>
        /// <example>
        /// <code>
        /// np.r_[np.array(new[] {1, 2, 3}), 0, 0, np.array(new[] {4, 5, 6})];  // [1 2 3 0 0 4 5 6]
        /// np.r_["-1:1:6j", new[] {0, 0, 0}, 5, 6];   // [-1 -0.6 -0.2 0.2 0.6 1 0 0 0 5 6]
        /// np.r_["0,2", new[] {1, 2, 3}, new[] {4, 5, 6}];  // [[1 2 3] [4 5 6]]
        /// np.r_["-1", a, a];                          // concatenate along the last axis
        /// </code>
        /// </example>
        public sealed class RClass : AxisConcatenator
        {
            internal RClass() : base(0)
            {
            }
        }

        /// <summary>
        ///     Builds arrays by concatenating along the first axis — see <see cref="RClass"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.r_.html</remarks>
        public static RClass r_ { get; } = new RClass();
    }
}
