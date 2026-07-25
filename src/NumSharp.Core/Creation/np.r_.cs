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
            public NDArray this[params object[] key] => Build(key);

            /// <summary>NumPy's <c>__len__</c>, which is 0 for these objects.</summary>
            public int Count => 0;

            private NDArray Build(object[] key)
            {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));

                // Copy the attributes — the first entry may override them.
                int axis = _axis, ndmin = _ndmin, trans1d = _trans1d;
                bool matrix = _matrix, col = false;

                var objs = new List<Operand>(key.Length);

                for (int k = 0; k < key.Length; k++)
                {
                    object item = key[k];
                    switch (item)
                    {
                        case null:
                            throw new ArgumentNullException($"key[{k}]",
                                "index expression entries must not be null.");

                        case string s:
                            if (s.IndexOf(':') < 0)
                            {
                                // No colon ⇒ a NumPy special directive, which must lead.
                                if (k != 0)
                                    throw new ValueError("special directives must be the first entry.");
                                ApplyDirective(s, ref axis, ref ndmin, ref trans1d, ref matrix, ref col);
                                break;
                            }

                            foreach (var token in s.Split(','))
                            {
                                if (string.IsNullOrWhiteSpace(token))
                                    continue;
                                objs.Add(FromSlice(ParseSliceToken(token), ndmin, trans1d));
                            }

                            break;

                        case Slice slice:
                            objs.Add(FromSlice(SliceSpec.FromSlice(slice), ndmin, trans1d));
                            break;

                        case Slice[] slices:
                            foreach (var one in slices)
                                objs.Add(FromSlice(SliceSpec.FromSlice(one), ndmin, trans1d));
                            break;

                        // Weak (NEP50 "python literal") scalars.
                        case bool:
                            objs.Add(Operand.Scalar(item, WeakKind.Bool));
                            break;
                        case sbyte or byte or short or ushort or int or uint or long or ulong:
                            objs.Add(Operand.Scalar(item, WeakKind.Int));
                            break;
                        case float or double:
                            objs.Add(Operand.Scalar(item, WeakKind.Float));
                            break;
                        case Complex:
                            objs.Add(Operand.Scalar(item, WeakKind.Complex));
                            break;

                        // Strong scalars — NumSharp dtypes with no Python literal form.
                        case char or Half or decimal:
                            objs.Add(Operand.StrongScalar(item));
                            break;

                        default:
                            objs.Add(FromArrayLike(item, ndmin, trans1d));
                            break;
                    }
                }

                NDArray[] arrays;
                if (objs.Count != 0)
                {
                    NPTypeCode final = ResolveDtype(objs);
                    arrays = new NDArray[objs.Count];
                    for (int i = 0; i < objs.Count; i++)
                        arrays[i] = objs[i].Materialize(final, ndmin);
                }
                else
                {
                    // Directives only — drops through to concatenate's own error, as in NumPy.
                    arrays = Array.Empty<NDArray>();
                }

                var res = concatenate(arrays, axis);

                if (matrix)
                {
                    int oldndim = res.ndim;
                    res = asmatrix(res);
                    if (oldndim == 1 && col)
                        res = res.T;
                }

                return res;
            }

            #region directives

            private static void ApplyDirective(string item, ref int axis, ref int ndmin,
                ref int trans1d, ref bool matrix, ref bool col)
            {
                if (item == "r" || item == "c")
                {
                    matrix = true;
                    col = item == "c";
                    return;
                }

                if (item.IndexOf(',') >= 0)
                {
                    // "axis,ndmin" or "axis,ndmin,trans1d". NumPy reads only vec[:2] and vec[2],
                    // so a fourth field is silently ignored — including a non-numeric one.
                    var vec = item.Split(',');
                    if (!TryParseDirectiveInt(vec[0], out int a) || !TryParseDirectiveInt(vec[1], out int n))
                        throw new ValueError($"unknown special directive '{item}'");

                    int t = trans1d;
                    if (vec.Length == 3 && !TryParseDirectiveInt(vec[2], out t))
                        throw new ValueError($"unknown special directive '{item}'");

                    axis = a;
                    ndmin = n;
                    trans1d = t;
                    return;
                }

                if (TryParseDirectiveInt(item, out int only))
                {
                    axis = only;
                    return;
                }

                throw new ValueError("unknown special directive");
            }

            private static bool TryParseDirectiveInt(string s, out int value)
                => int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);

            #endregion

            #region entry conversion

            /// <summary>
            ///     NumPy's slice branch: <c>arange</c>/<c>linspace</c>, then <c>ndmin</c> padding and the
            ///     <c>swapaxes(-1, trans1d)</c> that differs from the array branch's permutation.
            /// </summary>
            private static Operand FromSlice(in SliceSpec spec, int ndmin, int trans1d)
            {
                NDArray newobj = spec.Materialize();
                if (ndmin > 1)
                {
                    newobj = AtLeastNdView(newobj, ndmin);
                    if (trans1d != -1)
                        newobj = swapaxes(newobj, -1, trans1d);
                }

                return Operand.Array(newobj);
            }

            /// <summary>
            ///     NumPy's array branch: <c>array(item, ndmin=ndmin)</c> followed, when the entry was
            ///     upgraded, by the <c>defaxes</c> permutation that places the original axes at
            ///     <c>trans1d</c>.
            /// </summary>
            private static Operand FromArrayLike(object item, int ndmin, int trans1d)
            {
                NDArray arr = item as NDArray ?? asanyarray(item);
                int itemNdim = arr.ndim;

                NDArray newobj = AtLeastNdView(arr, ndmin);
                if (trans1d != -1 && itemNdim < ndmin)
                {
                    int k2 = ndmin - itemNdim;
                    int k1 = trans1d;
                    if (k1 < 0)
                        k1 += k2 + 1;
                    newobj = transpose(newobj, BuildTransposeAxes(ndmin, k1, k2));
                }

                return Operand.Array(newobj);
            }

            /// <summary>
            ///     <c>defaxes[:k1] + defaxes[k2:] + defaxes[k1:k2]</c> over <c>defaxes = range(ndmin)</c>,
            ///     with Python's forgiving list-slice clamping (an out-of-range <c>trans1d</c> yields a
            ///     malformed permutation and lands on transpose's own "axes don't match array").
            /// </summary>
            private static int[] BuildTransposeAxes(int ndmin, int k1, int k2)
            {
                var axes = new List<int>(ndmin);
                AppendRangeSlice(axes, ndmin, 0, k1);
                AppendRangeSlice(axes, ndmin, k2, ndmin);
                AppendRangeSlice(axes, ndmin, k1, k2);
                return axes.ToArray();
            }

            /// <summary>Appends <c>list(range(n))[start:stop]</c> with Python's index clamping.</summary>
            private static void AppendRangeSlice(List<int> into, int n, int start, int stop)
            {
                if (start < 0)
                    start = Math.Max(0, start + n);
                else if (start > n)
                    start = n;

                if (stop < 0)
                    stop = Math.Max(0, stop + n);
                else if (stop > n)
                    stop = n;

                for (int i = start; i < stop; i++)
                    into.Add(i);
            }

            #endregion

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

            #region dtype resolution

            /// <summary>
            ///     NEP50 kinds an entry can contribute. Ordered by dominance so the winner is the maximum.
            /// </summary>
            private enum WeakKind : byte
            {
                Strong = 0,
                Bool = 1,
                Int = 2,
                Float = 3,
                Complex = 4
            }

            /// <summary>One entry of the index expression, kept unmaterialized until the dtype is known.</summary>
            private readonly struct Operand
            {
                private readonly NDArray _array;
                private readonly object _scalar;
                private readonly NPTypeCode _code;
                private readonly WeakKind _weak;

                private Operand(NDArray array, object scalar, NPTypeCode code, WeakKind weak)
                {
                    _array = array;
                    _scalar = scalar;
                    _code = code;
                    _weak = weak;
                }

                public static Operand Array(NDArray a) => new Operand(a, null, a.typecode, WeakKind.Strong);

                public static Operand Scalar(object value, WeakKind weak)
                    => new Operand(null, value, NPTypeCode.Empty, weak);

                public static Operand StrongScalar(object value)
                    => new Operand(null, value, value.GetType().GetTypeCode(), WeakKind.Strong);

                public bool IsStrong => _weak == WeakKind.Strong;
                public NPTypeCode Code => _code;
                public WeakKind Weak => _weak;

                public NDArray Materialize(NPTypeCode final, int ndmin)
                {
                    NDArray arr;
                    if (_array is not null)
                    {
                        arr = _array.typecode == final ? _array : _array.astype(final);
                    }
                    else
                    {
                        if (_weak == WeakKind.Int)
                            CheckWeakIntFits(_scalar, final);

                        arr = asanyarray(_scalar);
                        if (arr.typecode != final)
                            arr = arr.astype(final);
                    }

                    return AtLeastNdView(arr, ndmin);
                }
            }

            /// <summary>
            ///     NumPy's <c>result_type(*result_type_objs)</c>: arrays and NumSharp-only scalars contribute
            ///     their dtype, C# literals contribute a weak kind that adopts it.
            /// </summary>
            private static NPTypeCode ResolveDtype(List<Operand> objs)
            {
                bool anyStrong = false;
                NPTypeCode strong = NPTypeCode.Empty;
                WeakKind weak = WeakKind.Strong;

                foreach (var op in objs)
                {
                    if (op.IsStrong)
                    {
                        strong = anyStrong ? NDExprTypeRules.PromoteStrong(strong, op.Code) : op.Code;
                        anyStrong = true;
                    }
                    else if (op.Weak > weak)
                    {
                        weak = op.Weak;
                    }
                }

                if (!anyStrong)
                {
                    // All literals — NEP50 defaults.
                    return weak switch
                    {
                        WeakKind.Bool => NPTypeCode.Boolean,
                        WeakKind.Int => NPTypeCode.Int64,
                        WeakKind.Float => NPTypeCode.Double,
                        _ => NPTypeCode.Complex
                    };
                }

                return weak switch
                {
                    // No literal, or a bool literal (which adopts anything).
                    WeakKind.Strong or WeakKind.Bool => strong,
                    // An int literal adopts anything but bool, which it lifts to the default int.
                    WeakKind.Int => strong == NPTypeCode.Boolean ? NPTypeCode.Int64 : strong,
                    // A float literal keeps a float-kind dtype's width, else forces the default float.
                    WeakKind.Float => NDExprTypeRules.IsFloatKind(strong)
                                      || strong == NPTypeCode.Decimal
                                      || strong == NPTypeCode.Complex
                        ? strong
                        : NPTypeCode.Double,
                    // NumSharp has a single complex width, so complex64 collapses onto Complex.
                    _ => NPTypeCode.Complex
                };
            }

            /// <summary>
            ///     NumPy raises <c>OverflowError</c> — never wraps — when a weak integer literal does not fit
            ///     the dtype it adopted: <c>np.r_[uint8Array, -1]</c> is an error, not 255.
            /// </summary>
            private static void CheckWeakIntFits(object value, NPTypeCode final)
            {
                if (value is ulong big && big > long.MaxValue)
                {
                    if (final != NPTypeCode.UInt64 && NDExprTypeRules.IsIntegerKind(final))
                        throw new OverflowException(
                            $"Python integer {big} out of bounds for {final.AsNumpyDtypeName()}");
                    return;
                }

                NDExprTypeRules.CheckIntLiteralFits(
                    Convert.ToInt64(value, CultureInfo.InvariantCulture), final);
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
