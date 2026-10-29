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
            ///     A parsed <c>start:stop:step</c> expression. Unlike <see cref="Slice"/> (which indexes and
            ///     is integer-only) this carries real bounds and NumPy's imaginary step, where
            ///     <c>start:stop:Nj</c> means "N points, stop inclusive" — i.e. <c>linspace</c>.
            /// </summary>
            internal readonly struct SliceSpec
            {
                public readonly double Start;
                public readonly double Stop;
                public readonly double Step;

                /// <summary>False for <c>a:</c> and <c>:</c>, where NumPy's arange re-reads start as stop.</summary>
                public readonly bool HasStop;

                /// <summary>The step was written <c>Nj</c>: <see cref="Step"/> is a point COUNT.</summary>
                public readonly bool ImaginaryStep;

                /// <summary>Every field was written as an integer literal ⇒ int64, else float64.</summary>
                public readonly bool Integral;

                public SliceSpec(double start, double stop, bool hasStop, double step,
                    bool imaginaryStep, bool integral)
                {
                    Start = start;
                    Stop = stop;
                    HasStop = hasStop;
                    Step = step;
                    ImaginaryStep = imaginaryStep;
                    Integral = integral;
                }

                /// <summary>Adapts a NumSharp <see cref="Slice"/> (always integral, never imaginary).</summary>
                public static SliceSpec FromSlice(Slice slice)
                {
                    if (slice is null)
                        throw new ArgumentNullException(nameof(slice));
                    if (slice.IsEllipsis || slice.IsNewAxis)
                        throw new ArgumentException(
                            "'...' and 'newaxis' are index-only and cannot appear in a concatenation expression.",
                            nameof(slice));

                    return new SliceSpec(slice.Start ?? 0, slice.Stop ?? 0, slice.Stop.HasValue,
                        slice.Step, false, true);
                }

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

                    // A zero step is left to arange, exactly as NumPy leaves it to its own
                    // (where it surfaces as Python's ZeroDivisionError from (stop-start)/step).
                    var code = Integral ? NPTypeCode.Int64 : NPTypeCode.Double;

                    // NumPy's arange(start, None, step) is arange(0, start, step): a missing stop
                    // promotes start into the stop slot. np.r_[5::2] is therefore [0, 2, 4].
                    return HasStop
                        ? arange(Start, Stop, Step, code)
                        : arange(0d, Start, Step, code);
                }
            }

            /// <summary>Parses one <c>start:stop[:step]</c> token.</summary>
            internal static SliceSpec ParseSliceToken(string token)
            {
                var fields = token.Split(':');
                if (fields.Length < 2 || fields.Length > 3)
                    throw new ArgumentException($"Invalid slice notation: '{token.Trim()}'", nameof(token));

                bool integral = true;

                double start = 0d;
                if (!string.IsNullOrWhiteSpace(fields[0]))
                {
                    start = ParseNumber(fields[0], token, out bool startIntegral, out _, allowImaginary: false);
                    integral &= startIntegral;
                }

                bool hasStop = !string.IsNullOrWhiteSpace(fields[1]);
                double stop = 0d;
                if (hasStop)
                {
                    stop = ParseNumber(fields[1], token, out bool stopIntegral, out _, allowImaginary: false);
                    integral &= stopIntegral;
                }

                double step = 1d;
                bool imaginary = false;
                if (fields.Length == 3 && !string.IsNullOrWhiteSpace(fields[2]))
                {
                    step = ParseNumber(fields[2], token, out bool stepIntegral, out imaginary, allowImaginary: true);
                    integral &= stepIntegral;
                }

                return new SliceSpec(start, stop, hasStop, step, imaginary, integral && !imaginary);
            }

            /// <summary>
            ///     Reads one numeric field. <paramref name="integral"/> follows Python's LITERAL rule, not
            ///     the value's: <c>0</c> is integral, <c>0.0</c> and <c>1e3</c> are not — which is what makes
            ///     <c>np.r_["0:5"]</c> int64 while <c>np.r_["0.0:5"]</c> is float64.
            /// </summary>
            private static double ParseNumber(string field, string token, out bool integral,
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

                integral = f.IndexOf('.') < 0 && f.IndexOf('e') < 0 && f.IndexOf('E') < 0;

                if (!double.TryParse(f, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                    throw new ArgumentException($"Invalid slice notation: '{token.Trim()}'", nameof(token));

                return value;
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
