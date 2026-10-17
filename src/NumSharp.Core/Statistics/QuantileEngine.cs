using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     13 NumPy quantile estimation methods (Hyndman &amp; Fan 1996 + NumPy-only variations).
    ///     Mirrors numpy.lib._function_base_impl._QuantileMethods.
    /// </summary>
    public enum QuantileMethod
    {
        Linear,                       // R-7, NumPy default — (n-1)*q
        Lower,                        // floor((n-1)*q), discrete
        Higher,                       // ceil((n-1)*q), discrete
        Nearest,                      // round((n-1)*q), discrete
        Midpoint,                     // 0.5*(floor+ceil), gamma=0 at integer indices
        InvertedCdf,                  // H&F R-1, discrete
        AveragedInvertedCdf,          // H&F R-2
        ClosestObservation,           // H&F R-3, discrete
        InterpolatedInvertedCdf,      // H&F R-4, α=0, β=1
        Hazen,                        // H&F R-5, α=β=0.5
        Weibull,                      // H&F R-6, α=β=0
        MedianUnbiased,               // H&F R-8, α=β=1/3
        NormalUnbiased,               // H&F R-9, α=β=3/8
    }

    /// <summary>
    ///     Orchestrator for <c>np.median</c>, <c>np.percentile</c>, <c>np.quantile</c>.
    ///     Responsibilities split:
    ///       <list type="bullet">
    ///         <item>This file: input validation, axis normalization, dtype-promotion rule,
    ///               output-shape computation, keepdims/out plumbing, scratch rental.</item>
    ///         <item><see cref="DirectILKernelGenerator.Quantile"/>: per-dtype IL-emitted kernel
    ///               that owns the outer row-loop and dispatches into a JIT-specialized
    ///               generic row processor. No per-dtype switch executes per call after
    ///               the first cache miss.</item>
    ///       </list>
    /// </summary>
    internal static class QuantileEngine
    {
        internal static bool IsDiscrete(QuantileMethod m) =>
            m == QuantileMethod.Lower ||
            m == QuantileMethod.Higher ||
            m == QuantileMethod.Nearest ||
            m == QuantileMethod.InvertedCdf ||
            m == QuantileMethod.ClosestObservation;

        /// <summary>
        ///     NumPy's <c>_QuantileMethods[method]</c> lookup: the method named by one of NumPy's thirteen names.
        /// </summary>
        /// <param name="method">The method name (case-sensitive, NumPy's spelling).</param>
        /// <returns>The method.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="method"/> is null.</exception>
        /// <exception cref="ValueError">An unknown name: NumPy's text, <c>'x' is not a valid method. Use one of:
        ///     dict_keys([...])</c>.</exception>
        internal static QuantileMethod ParseMethod(string method)
        {
            if (method == null) throw new ArgumentNullException(nameof(method));
            if (TryParseMethod(method, out var parsed))
                return parsed;
            // _quantile's `raise ValueError(f"{method!r} is not a valid method. Use one of: "
            // f"{_QuantileMethods.keys()}")`, verbatim: the name's Python repr, then the dict's key view in its
            // insertion order (ValueError is an ArgumentException, so existing catches still see it).
            throw new ValueError(
                $"{Backends.Printing.PyRepr.Str(method)} is not a valid method. Use one of: dict_keys(['inverted_cdf', " +
                "'averaged_inverted_cdf', 'closest_observation', 'interpolated_inverted_cdf', 'hazen', 'weibull', " +
                "'linear', 'median_unbiased', 'normal_unbiased', 'lower', 'higher', 'midpoint', 'nearest'])");
        }

        /// <summary>
        ///     <see cref="ParseMethod"/> without the error: whether <paramref name="method"/> is one of NumPy's thirteen
        ///     names. <see cref="Compute"/> uses it to raise the unknown-name error where NumPy's <c>_quantile</c> does.
        /// </summary>
        /// <param name="method">The method name (case-sensitive; null is not a name).</param>
        /// <param name="parsed">The method, or <see cref="QuantileMethod.Linear"/> when the name is unknown.</param>
        /// <returns>True for a known name.</returns>
        internal static bool TryParseMethod(string method, out QuantileMethod parsed)
        {
            switch (method)
            {
                case "linear":                    parsed = QuantileMethod.Linear; return true;
                case "lower":                     parsed = QuantileMethod.Lower; return true;
                case "higher":                    parsed = QuantileMethod.Higher; return true;
                case "nearest":                   parsed = QuantileMethod.Nearest; return true;
                case "midpoint":                  parsed = QuantileMethod.Midpoint; return true;
                case "inverted_cdf":              parsed = QuantileMethod.InvertedCdf; return true;
                case "averaged_inverted_cdf":     parsed = QuantileMethod.AveragedInvertedCdf; return true;
                case "closest_observation":      parsed = QuantileMethod.ClosestObservation; return true;
                case "interpolated_inverted_cdf": parsed = QuantileMethod.InterpolatedInvertedCdf; return true;
                case "hazen":                     parsed = QuantileMethod.Hazen; return true;
                case "weibull":                   parsed = QuantileMethod.Weibull; return true;
                case "median_unbiased":           parsed = QuantileMethod.MedianUnbiased; return true;
                case "normal_unbiased":           parsed = QuantileMethod.NormalUnbiased; return true;
                default:                          parsed = QuantileMethod.Linear; return false;
            }
        }

        /// <summary>
        ///     The numeric quantile engine behind <c>np.median</c> / <c>np.percentile</c> / <c>np.quantile</c> and their NaN
        ///     forms: stages each reduced line, partitions it at the indexes the method needs and selects or interpolates.
        /// </summary>
        /// <param name="a">The numeric data.</param>
        /// <param name="q">The quantiles in [0, 1], in C order (values of q's dtype, see <paramref name="qType"/>).</param>
        /// <param name="axisArr">The reduced axes, or null for the flattened data.</param>
        /// <param name="out">An optional output array, written and returned itself: NumPy's two routes (see the
        ///     remarks).</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (a copy is staged anyway).</param>
        /// <param name="method">The estimation method (ignored when <paramref name="methodName"/> is given).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="qIsScalar">Whether q is a scalar: the result then has no leading q axis.</param>
        /// <param name="emptyReturnsNaN">np.median's rule: an empty reduced line yields NaN instead of raising.</param>
        /// <param name="ignoreNaN">The np.nan* forms: NaN elements are skipped.</param>
        /// <param name="allowBooleanContinuous">np.median's rule: bool data interpolates (through its mean) instead of
        ///     raising NumPy's boolean-subtract TypeError.</param>
        /// <param name="medianMean">
        ///     When true, the even-size slice is reduced with NumPy's <c>np.median</c> rule —
        ///     the <b>mean</b> of the two central order statistics, <c>(a+b)/2</c> — instead of
        ///     the quantile <c>_lerp</c> at q=0.5. The two disagree by up to 1 ULP on ~10 % of
        ///     inputs (NumPy computes <c>np.median</c> via <c>mean(part[middle])</c>, a different
        ///     code path from <c>np.quantile(a, 0.5)</c>), so <c>np.median</c>/<c>np.nanmedian</c>
        ///     MUST set this to stay bit-exact. Only meaningful with <paramref name="method"/> =
        ///     <see cref="QuantileMethod.Linear"/> and q = 0.5 (the sole caller shape); ignored by
        ///     the complex path, whose <c>(1-γ)·a + γ·b</c> form already equals the mean at γ=0.5.
        /// </param>
        /// <param name="weakQ">NumPy 2.4's <c>weak_q = type(q) in (int, float)</c>: true when q was a Python int or float
        ///     (a C# number), whose interpolation weight is a WEAK float that keeps float16 / float32 data's dtype; false for
        ///     a q ARRAY, 0-d included, whose weight promotes the result to <c>result_type(data, q)</c>
        ///     (<c>np.quantile(f32, np.array(0.5))</c> is float64). Null (the default) reads <paramref name="qIsScalar"/>,
        ///     the rule of the callers that pass a C# number for a scalar q.</param>
        /// <param name="qShape">The dims of a q ARRAY of rank 1 or more (its values are <paramref name="q"/>, in C order):
        ///     the result's leading axes. NumPy keeps a 2-d q's two axes (<c>np.quantile(a, q.reshape(2, 2))</c> is
        ///     <c>(2, 2)</c>) and refuses rank 3 and above (<c>_quantile_ureduce_func</c>: <c>q must be a scalar or 1d</c>),
        ///     after the axis checks. Null: one leading axis of <c>q.Length</c> unless <paramref name="qIsScalar"/>.</param>
        /// <param name="methodName">The method as the caller named it, checked where NumPy's <c>_quantile</c> reads
        ///     <c>_QuantileMethods[method]</c>: after the axis and rank checks, before the boolean subtract, and never by the
        ///     nan forms when every slice is all NaN (their answer is then NaN in the data's dtype, whatever the name). Null:
        ///     <paramref name="method"/> was parsed by the caller (np.median's fixed 'linear').</param>
        /// <param name="integerQ">Whether q is an integer (or bool) array: 'linear''s virtual index <c>(n - 1) * q</c> is
        ///     then an integer, which NumPy's <c>_quantile</c> takes instead of interpolating (its "supports_integers"
        ///     path) — the 'lower' selection, in the data's dtype. For an integer q ARRAY the virtual index of weibull,
        ///     averaged_inverted_cdf and interpolated_inverted_cdf is an integer too, so their lerp runs with an int64
        ///     weight: an int64 result over integer data (not uint64).</param>
        /// <param name="qType">q's dtype for a q ARRAY: <see cref="NPTypeCode.Half"/> or <see cref="NPTypeCode.Single"/>
        ///     makes NumPy compute the virtual index, the neighbours, the weight and <c>1 - t</c> in float16 / float32, and
        ///     the lerp in <c>result_type(data, q)</c>; an integer dtype (Int64 for a bool q) computes them in that integer
        ///     dtype, wrapping (see <see cref="QuantileIndexes"/>). Every other value (the default, a weak q) computes in
        ///     float64.</param>
        /// <returns>The quantiles; <paramref name="out"/> itself when one is given; a read-only 0-d result for a scalar q
        ///     over every axis (NumPy's scalar).</returns>
        /// <exception cref="ValueError">A q of rank above 2 (<paramref name="qShape"/>), except for the nan forms over data
        ///     that is NaN everywhere, where every slice returns <c>np.full(q.shape, nan)</c> before NumPy reaches the
        ///     check.</exception>
        /// <exception cref="ValueError">The partition refuses an index (NumPy's <c>kth(=k) out of bounds (n)</c>): a float16
        ///     q over a line of about 2051 elements or more (or a float32 q past 2^24) rounds an index to n, or makes it
        ///     NaN or infinite (INT64_MIN as intp), and an unsigned integer q's 0 makes inverted_cdf / closest_observation
        ///     take its dtype's maximum (<c>kth(=255) out of bounds (8)</c>) — for the nan forms per slice, at the first
        ///     slice in C order.</exception>
        /// <exception cref="OverflowException">An integer q array: <c>n</c> or <c>n - 1</c> outside its dtype (<c>Python
        ///     integer 200 out of bounds for int8</c>), or an unsigned q under weibull / averaged_inverted_cdf /
        ///     interpolated_inverted_cdf (<c>Python integer -1 out of bounds for uint8</c>) — before every other check of
        ///     the computation.</exception>
        /// <exception cref="TypeError">'linear' with a uint64 q array: <c>Partition index must be integer</c> (its kth list is
        ///     uint64, or float64 with the -1 NumPy appends over float data).</exception>
        /// <remarks>
        ///     <para><b>The nan forms' all-NaN slices.</b> NumPy's <c>_nanquantile_1d</c> returns
        ///     <c>np.full(q.shape, np.nan, dtype=arr1d.dtype)</c> for a slice holding nothing but NaN — the DATA's dtype even
        ///     when a strong q array makes every other slice float64 — and <c>np.apply_along_axis</c> allocates its result
        ///     from the FIRST slice's answer. So float16 / float32 data with a q array gives a result of the data's dtype
        ///     exactly when the first slice (the only one for a flat reduction) is all NaN, the other slices' float64
        ///     answers rounded into it (probed 2.4.2: <c>np.nanquantile([[nan, nan], [1, 2]] as float32, np.array(0.5),
        ///     axis=1)</c> is float32 <c>[nan, 1.5]</c>; with the rows swapped it is float64).</para>
        ///     <para><b>out=.</b> <c>_ureduce</c> first indexes out with <c>(Ellipsis,) + index_out</c> under keepdims. A
        ///     selection method then runs <c>take(arr, indexes, axis=0, out=out)</c>: out must have the result's shape, cast
        ///     SAFELY to the data's dtype and be writeable, and receives the values through an unsafe cast; a slice holding
        ///     NaN is then copied in with <c>np.copyto</c> (same_kind from the data). An interpolating method runs
        ///     <c>_lerp(..., out=out)</c>: <c>add</c> casts its output into out (same_kind) and broadcasts it (out may be
        ///     larger), and <c>subtract(..., dtype=type(out.dtype))</c> computes the t &gt;= 0.5 branch in out's dtype. The
        ///     nan forms assign their result (<c>out[...] = result</c>: broadcast, unsafe). np.median / np.nanmedian keep
        ///     the result-then-copy rule (NumPy's median writes through <c>np.mean(..., out=out)</c>, not ported), and so
        ///     do the NumSharp-only decimal and complex data and a decimal out.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="q"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">An axis repeats.</exception>
        /// <exception cref="IndexError">keepdims with an out of too few dimensions (NumPy's <c>too many indices for
        ///     array</c>).</exception>
        /// <exception cref="TypeError">Bool data with an interpolating method (NumPy's boolean-subtract text, after the
        ///     partition's validation and the neighbour lookup), unless <paramref name="allowBooleanContinuous"/>.</exception>
        /// <exception cref="IndexOutOfRangeException">An array with no element whose lookup fails (NumPy's IndexError
        ///     texts): float data with an empty reduced line (<c>index -1 is out of bounds for axis 0 with size 0</c>); a
        ///     selection method over integer data with an empty line and a non-empty result (<c>cannot do a non-empty take
        ///     from an empty axes.</c>) or an index out of range (<c>index k is out of bounds for axis 0 with size n</c>);
        ///     an interpolating method whose neighbour is out of range — unless
        ///     <paramref name="emptyReturnsNaN"/>.</exception>
        /// <exception cref="ValueError">A selection method's out of another shape (<c>output array does not match result of
        ///     ndarray.take</c>) or read-only (<c>WRITEBACKIFCOPY base is read-only</c>); an interpolating method's out that
        ///     is read-only (<c>output array is read-only</c>) or does not broadcast (<c>operands could not be broadcast
        ///     together with shapes ...</c>, <c>non-broadcastable output operand with shape ... doesn't match the broadcast
        ///     shape ...</c>); the nan forms' assignment (<c>could not broadcast input array from shape ... into shape
        ///     ...</c>, <c>assignment destination is read-only</c>).</exception>
        /// <exception cref="TypeError">A selection method's out whose dtype does not cast safely to the data's
        ///     (<c>Cannot cast array data from dtype('float64') to dtype('float32') according to the rule 'safe'</c>; "scalar"
        ///     for a 0-d out), or a slice holding NaN copied into an out the data does not cast to under same_kind (NumPy's
        ///     <c>np.copyto</c> text).</exception>
        /// <exception cref="ArgumentException">An interpolating method's out that the lerp's dtype does not cast to under
        ///     same_kind: NumPy's UFuncTypeError <c>Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')
        ///     with casting rule 'same_kind'</c> (the house type of the ufunc out= routes). The legacy routes: an
        ///     <paramref name="out"/> of another shape than the result.</exception>
        [NDScoped]
        public static NDArray Compute(
            NDArray a, double[] q, int[] axisArr, NDArray @out,
            bool overwrite_input, QuantileMethod method, bool keepdims, bool qIsScalar,
            bool emptyReturnsNaN = false, bool ignoreNaN = false, bool allowBooleanContinuous = false,
            bool medianMean = false, bool? weakQ = null, long[] qShape = null, string methodName = null,
            bool integerQ = false, NPTypeCode qType = NPTypeCode.Double)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            if (q is null) throw new ArgumentNullException(nameof(q));

            // Boundary scope: the staging views/copies, the raw result behind the keepdims view,
            // and every helper temp are reclaimed at exit; the yielded result (or the caller's
            // untracked @out) is the only survivor.

            // np.nan* short-circuit a totally-empty input (some dimension is 0) to
            // np.nanmean(a, axis, keepdims) BEFORE any method/q dispatch. NumPy does this
            // in _nanquantile_unchecked / nanmedian, so the q dimension is dropped and the
            // dtype follows nanmean (float widths preserved, everything else -> float64).
            // This runs before the boolean-continuous guard so that, like NumPy, an empty
            // boolean nanquantile returns nan instead of raising.
            if (ignoreNaN && a.size == 0)
                return BuildEmptyNanResult(a, axisArr, keepdims, @out);

            // _ureduce normalizes the axis BEFORE the reduction function runs, so an out-of-range or repeated axis is
            // reported before the boolean subtract below (np.quantile(bool, 0.5, axis=5) is NumPy's AxisError).
            int[] reducedAxes = axisArr != null ? NormalizeAxes(axisArr, a.ndim) : null;

            // The out NumPy's reduction writes into: under keepdims, _ureduce's `out[(Ellipsis,) + index_out]` (each reduced
            // axis taken at 0), indexed right after the axes and before anything else is checked. np.median's out
            // (np.mean's rules), the NumSharp-only decimal / complex data and a decimal out keep the legacy
            // result-then-copy rule instead (see the remarks).
            bool legacyOut = medianMean || a.typecode == NPTypeCode.Decimal || a.typecode == NPTypeCode.Complex
                             || (@out is not null && @out.typecode == NPTypeCode.Decimal);
            NDArray target = @out is not null && keepdims && !legacyOut ? KeepdimsOutTarget(@out, a.ndim, reducedAxes) : @out;

            // `_quantile_ureduce_func`'s first statement refuses a q of rank above 2, after the axis checks (and, for the
            // nan forms, after the empty-input return above). The nan forms reach it per slice and only for a slice that
            // still holds a value once its NaNs are removed, so over data that is NaN everywhere every slice answers
            // `np.full(q.shape, nan)` in the data's dtype instead (probed 2.4.2: (2, 2, 2) NaNs, float32 for float32 data).
            if (qShape is { Length: > 2 })
            {
                if (!(ignoreNaN && IsFloatTc(a.typecode) && AllNaN(a)))
                    throw new ValueError("q must be a scalar or 1d");
                return AllNaNSlicesResult(a, axisArr, keepdims, qShape, @out);
            }

            // The method name, where NumPy's `_quantile` looks it up: after the axis and rank checks, before the boolean
            // subtract. The nan forms reach `_quantile` only through a slice that still holds a value once its NaNs are
            // removed, so over data that is NaN everywhere an unknown name raises nothing and every slice answers NaN.
            if (methodName != null && !TryParseMethod(methodName, out method))
            {
                if (ignoreNaN && IsFloatTc(a.typecode) && AllNaN(a))
                    return AllNaNSlicesResult(a, axisArr, keepdims,
                        qIsScalar ? Array.Empty<long>() : qShape ?? new long[] { q.Length }, @out);
                ParseMethod(methodName);   // throws NumPy's ValueError (or ArgumentNullException for a null name)
            }
            // An integer (or bool) q: 'linear''s virtual index is an integer, which `_quantile` takes ("supports_integers")
            // instead of interpolating: the 'lower' selection.
            bool linearTake = integerQ && method == QuantileMethod.Linear;
            if (linearTake)
                method = QuantileMethod.Lower;

            bool weak = weakQ ?? qIsScalar;
            // q's precision: a float16 / float32 q ARRAY computes its indexes and weights in its own dtype, an integer q
            // array in its integer dtype (wrapping; NumPy's NEP 50 arithmetic with the Python numbers of the method); a
            // Python q became a float64 / int64 array, exact in double.
            NPTypeCode precision = !weak && qType is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.SByte or NPTypeCode.Byte
                or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64
                or NPTypeCode.UInt64
                ? qType
                : NPTypeCode.Double;
            // An integer q ARRAY makes weibull / averaged_inverted_cdf / interpolated_inverted_cdf's virtual index an
            // integer: their lerp then runs with an integer weight of q's dtype (a weak Python int's `float(gamma)` does
            // not).
            bool integerWeight = integerQ && !weak && method is QuantileMethod.AveragedInvertedCdf or QuantileMethod.Weibull
                or QuantileMethod.InterpolatedInvertedCdf;
            bool discrete = IsDiscrete(method);
            // `np.issubdtype(arr.dtype, np.inexact)`: a float array may hold NaN (the take path appends -1 to its kth list,
            // `_get_indexes` maps a NaN virtual index to the last element, and an empty line fails at `arr[-1, ...]`).
            bool inexact = IsFloatTc(a.typecode);
            // NumPy raises TypeError for boolean input with a continuous (interpolating)
            // method: the underlying lerp performs `b - a` and boolean subtraction is
            // unsupported. Discrete methods only index (never subtract), so they are fine;
            // np.median / np.nanmedian coerce through mean() and pass
            // allowBooleanContinuous: true to keep computing a float64 result. NumPy reaches `b - a` after the partition and
            // the neighbour lookup, so the check runs after their errors below.
            bool booleanSubtract = !discrete && !allowBooleanContinuous && a.typecode == NPTypeCode.Boolean;

            NPTypeCode outTypeCode = ResolveOutputDtype(a.typecode, method, weak, precision, integerWeight);

            // NaN-ignoring (np.nan*) is only meaningful for dtypes that can hold a NaN.
            // Integer / bool / char rows never contain NaN, so the regular kernel (with
            // its precomputed partition indices) is both correct and faster there.
            bool nanAware = ignoreNaN && inexact;

            // ── Stage the data so the reduction axis is innermost (stride-1 rows). ──
            // The IL kernel below walks rows by adding (n * sizeof(T)) per iteration.
            NDArray staged;
            int n;
            long outerSize;

            // Only a null axis reduces every axis. An EMPTY tuple reduces none: NumPy's _ureduce keeps every axis and
            // reshapes to `a.shape[:nkeep] + (-1,)`, a new trailing axis of length 1 (the product of no axes), so each
            // element is its own slice — the multi-axis branch below does exactly that for zero axes.
            if (axisArr == null)
            {
                if (a.ndim == 1 && a.Shape.IsContiguous)        staged = a;
                else if (a.Shape.IsContiguous)                  staged = a.reshape((long)a.size);
                else                                            staged = a.flatten();
                n = (int)staged.size;
                outerSize = 1;
            }
            else
            {
                int[] normalized = reducedAxes;

                if (normalized.Length == 1)
                {
                    int axis = normalized[0];
                    int nd = a.ndim;
                    if (nd == 1 || axis == nd - 1)
                        staged = a.Shape.IsContiguous ? a : a.copy();
                    else
                        staged = np.moveaxis(a, axis, nd - 1).copy();
                    n = (int)staged.Shape.dimensions[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.Shape.dimensions[i];
                }
                else
                {
                    // Multi-axis: merge the reduction axes into one at the back.
                    int[] sortedAxis = (int[])normalized.Clone();
                    Array.Sort(sortedAxis);
                    int[] keep = Enumerable.Range(0, a.ndim).Where(i => Array.BinarySearch(sortedAxis, i) < 0).ToArray();
                    int[] dest = Enumerable.Range(0, keep.Length).ToArray();
                    NDArray moved = np.moveaxis(a, keep, dest);
                    long reduced = 1;
                    for (int i = 0; i < normalized.Length; i++) reduced *= a.Shape.dimensions[normalized[i]];
                    long[] newShape = new long[keep.Length + 1];
                    for (int i = 0; i < keep.Length; i++) newShape[i] = a.Shape.dimensions[keep[i]];
                    newShape[keep.Length] = reduced;
                    staged = moved.reshape(Shape.Adopt(newShape)).copy();
                    n = (int)staged.Shape.dimensions[staged.ndim - 1];
                    outerSize = 1;
                    for (int i = 0; i < staged.ndim - 1; i++) outerSize *= staged.Shape.dimensions[i];
                }
            }

            // ── Output shape: q's shape (if non-scalar) prepended to the reduced shape. ──
            // A 2-d q keeps both axes: the kernel writes q index j's row block at j * outerSize, which in a C-contiguous
            // (q0, q1, outer...) buffer is exactly element (j / q1, j % q1) — the same bytes as a (q0 * q1, outer...) one.
            int qLen = q.Length;
            long[] qDims = qIsScalar ? Array.Empty<long>() : qShape ?? new long[] { qLen };
            int outerDims = staged.ndim - 1;
            long[] outShape;
            if (axisArr == null)
            {
                outShape = (long[])qDims.Clone();
            }
            else
            {
                outShape = new long[qDims.Length + outerDims];
                for (int i = 0; i < qDims.Length; i++) outShape[i] = qDims[i];
                for (int i = 0; i < outerDims; i++) outShape[qDims.Length + i] = staged.Shape.dimensions[i];
            }
            int qNdim = qDims.Length;
            Shape resultShape = outShape.Length == 0 ? new Shape() : Shape.Adopt(outShape);

            // ── np.median's empty reduction axis: NaN (its mean of nothing). ──
            if (n == 0 && emptyReturnsNaN)
            {
                NDArray nanResult = new NDArray(outTypeCode, resultShape);
                FillWithNaN(nanResult, outTypeCode);
                return FinalizeResult(nanResult, a, axisArr, keepdims, qNdim, @out);
            }

            // ── The index plan: NumPy's virtual indexes, neighbours and weights in q's precision, for a line of n. ──
            // The nan forms plan each slice again in the kernel (their slices lose their NaNs), but an empty array and the
            // plain path read this one.
            var raw = new long[2 * qLen];
            var weight = new double[2 * qLen];
            unsafe
            {
                fixed (double* qPtr = q)
                fixed (long* rawPtr = raw)
                fixed (double* weightPtr = weight)
                    DirectILKernelGenerator.QuantilePlan(precision, method, n, qPtr, qLen, inexact, rawPtr, weightPtr);
            }

            // 'linear' with a uint64 q takes `(n - 1) * q`, a uint64 array, and partitions at it (with -1 appended over
            // float data): neither uint64 nor its promotion with -1 (float64) casts safely to intp, so
            // partition_prep_kth_array refuses it — whatever the array's size, before the take or `arr[-1, ...]`; the nan
            // forms reach it at their first slice that holds a value.
            if (linearTake && precision == NPTypeCode.UInt64 && !(nanAware && AllNaN(a)))
                throw new TypeError("Partition index must be integer");

            // How NumPy writes the result into out (see the remarks): 0 none, 1 take, 2 lerp, 3 the nan forms' assignment,
            // 4 the legacy copy.
            int outRoute = @out is null ? 0 : legacyOut ? 4 : ignoreNaN ? 3 : discrete ? 1 : 2;

            // ── An array with no element (an empty line, or no line at all). ──
            // NumPy's partition validates nothing then (partition_prep_kth_array skips an array of size 0); the take or the
            // neighbour lookup reports what it reads, in _quantile's order.
            if (n == 0 || outerSize == 0)
            {
                if (n == 0 && inexact)
                    // `slices_having_nans = np.isnan(arr[-1, ...])`, before the take and the neighbour lookup alike.
                    throw new IndexOutOfRangeException("index -1 is out of bounds for axis 0 with size 0");
                long resultSize = 1;
                foreach (long d in outShape) resultSize *= d;
                if (discrete)
                {
                    // take(arr, virtual_indexes, axis=0, out=out): out's checks, then the empty axis, then every index.
                    if (outRoute == 1)
                        TakeOutChecks(target, outShape, a);
                    if (n == 0 && resultSize != 0)
                        throw new IndexOutOfRangeException("cannot do a non-empty take from an empty axes.");
                    CheckLookups(raw, qLen, 0, n);
                }
                else
                {
                    // `previous = arr[previous_indexes]`, then `next = arr[next_indexes]`, then _lerp.
                    CheckLookups(raw, qLen, 0, n);
                    CheckLookups(raw, qLen, 1, n);
                    if (booleanSubtract)
                        throw BooleanSubtractError();
                    if (outRoute == 2)
                        LerpOutChecks(target, outTypeCode, outShape);
                }
                NDArray emptyResult = new NDArray(outTypeCode, resultShape);
                return FinishResult(emptyResult, a, axisArr, keepdims, qNdim, @out, target, outRoute, staged: null, outerDims);
            }

            // ── partition_prep_kth_array (the plain path's one partition; the nan forms validate each slice). ──
            if (!nanAware)
            {
                unsafe
                {
                    fixed (long* rawPtr = raw)
                    {
                        if (discrete)
                            QuantileIndexes.CheckTakeKth(rawPtr, qLen, 2, n);
                        else
                            QuantileIndexes.CheckInterpolationKth(rawPtr, 2 * qLen, n);
                    }
                }
            }
            if (booleanSubtract)
                throw BooleanSubtractError();

            // out's checks before anything is written: the take's (shape, safe cast, writeable), or the lerp's add (writeable,
            // same_kind cast, broadcast). A float / complex out other than the lerp's dtype receives the lerp itself: its
            // t >= 0.5 branch is computed in out's dtype (NumPy's `subtract(..., dtype=type(out.dtype))`).
            NPTypeCode bufferType = outTypeCode;
            if (outRoute == 1)
                TakeOutChecks(target, outShape, a);
            else if (outRoute == 2)
            {
                LerpOutChecks(target, outTypeCode, outShape);
                if (LerpsIntoOut(outTypeCode, target.typecode, integerWeight))
                    bufferType = target.typecode;
            }

            NDArray resultRaw = new NDArray(bufferType, resultShape);

            // The plain path's partition indexes: every plan index made partition-ready (NumPy's -1 is the last element),
            // kept per quantile for the cells and sorted + deduplicated for the partition. Deduplicating shrinks the
            // number of QuickSelect passes when q values share indices (e.g. q=[25,50,75] often share neighbouring
            // picks on small n).
            var cellIndex = new int[2 * qLen];
            for (int i = 0; i < cellIndex.Length; i++)
            {
                long k = raw[i];
                cellIndex[i] = (int)(k < 0 ? k + n : k);
            }
            int[] kSortedManaged = nanAware || n <= 1 ? Array.Empty<int>() : SortedDistinct(cellIndex);

            int srcSize = TypeSize(a.typecode);
            byte[] scratchBytes = ArrayPool<byte>.Shared.Rent(checked(srcSize * n));
            double[] qCopy = q;   // q is already a flat array we own; no need to copy
            // Per-row plan scratch for the NaN path (2 entries per q, reused across rows so the per-row plan allocates
            // nothing).
            int rowScratch = Math.Max(1, 2 * qCopy.Length);
            int[] rowKManaged = nanAware ? ArrayPool<int>.Shared.Rent(rowScratch) : null;
            long[] rowRawManaged = nanAware ? ArrayPool<long>.Shared.Rent(rowScratch) : null;
            double[] rowWeightManaged = nanAware ? ArrayPool<double>.Shared.Rent(rowScratch) : null;

            try
            {
                unsafe
                {
                    fixed (byte* scratchPtr = scratchBytes)
                    fixed (int* kPtr = kSortedManaged)
                    fixed (int* cellIndexPtr = cellIndex)
                    fixed (double* cellWeightPtr = weight)
                    fixed (double* qPtr = qCopy)
                    fixed (int* rowKPtr = rowKManaged)
                    fixed (long* rowRawPtr = rowRawManaged)
                    fixed (double* rowWeightPtr = rowWeightManaged)
                    {
                        long dstOuterStride = qIsScalar ? 0L : outerSize;
                        // Logical element 0 of the staged operand. Most staging paths copy/flatten
                        // (offset 0), but the 1-D-contiguous and last-axis-contiguous fast paths
                        // keep `staged = a` with its original Shape.offset — so the base MUST fold in
                        // `offset * itemsize`, exactly as np.average/np.cov do. Omitting it read from
                        // the buffer start and ignored the slice (e.g. a `simple_slice_offset` view).
                        byte* stagedBase = (byte*)staged.Address + staged.Shape.offset * srcSize;
                        // Complex has no IL quantile kernel — values have no natural
                        // total order and the IL pipeline is float/int-only.  Route
                        // through a managed lexicographic-sort + interpolate path that
                        // matches NumPy's complex quantile semantics (lexicographic
                        // by real first, imag as tie-break; q-interpolation operates
                        // on Complex × double which is well-defined).
                        if (a.typecode == NPTypeCode.Complex)
                        {
                            ComputeComplexQuantile(
                                srcBase: (System.Numerics.Complex*)stagedBase,
                                outer: outerSize, n: n,
                                method: method,
                                q: qPtr, nQs: qCopy.Length,
                                dstBase: (System.Numerics.Complex*)resultRaw.Address,
                                dstOuterStride: dstOuterStride);
                        }
                        else
                        {
                            DirectILKernelGenerator.Quantile(
                                srcType: a.typecode,
                                outType: bufferType,
                                lerpType: outTypeCode,
                                qType: precision,
                                method: method,
                                srcBase: stagedBase,
                                scratchBase: scratchPtr,
                                outer: outerSize,
                                n: n,
                                kSorted: kPtr,
                                nKs: kSortedManaged.Length,
                                cellIndex: cellIndexPtr,
                                cellWeight: cellWeightPtr,
                                q: qPtr,
                                nQs: qCopy.Length,
                                dstBase: resultRaw.Address,
                                dstOuterStride: dstOuterStride,
                                ignoreNaN: nanAware,
                                rowKScratch: rowKPtr,
                                rowRaw: rowRawPtr,
                                rowWeight: rowWeightPtr,
                                medianMean: medianMean);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(scratchBytes);
                if (rowKManaged != null) ArrayPool<int>.Shared.Return(rowKManaged);
                if (rowRawManaged != null) ArrayPool<long>.Shared.Return(rowRawManaged);
                if (rowWeightManaged != null) ArrayPool<double>.Shared.Return(rowWeightManaged);
            }

            // The nan forms' all-NaN FIRST slice (see the remarks): np.apply_along_axis allocates its result from the
            // first slice's answer, `np.full(q.shape, nan, dtype=arr1d.dtype)` when that slice holds only NaN, so a strong
            // q over float16 / float32 data (the only case where the computed dtype differs from the data's) gives the
            // data's dtype, every other slice's float64 answer rounded into it. Staged row 0 is NumPy's first slice: the
            // reduced axes are innermost and the kept ones in C order, as np.ndindex walks them.
            if (nanAware && outTypeCode != a.typecode && FirstRowAllNaN(staged, outerSize, n))
                resultRaw = resultRaw.astype(a.typecode);

            // The nan forms along an axis return np.apply_along_axis's layout (an out receives the values instead).
            if (ignoreNaN && !legacyOut && axisArr != null && outerDims > 0 && outRoute == 0)
                resultRaw = AlongAxisLayout(resultRaw, qNdim);

            return FinishResult(resultRaw, a, axisArr, keepdims, qNdim, @out, target, outRoute, staged, outerDims);
        }

        /// <summary>
        ///     Whether every element of a float array is NaN, the nan forms' all-NaN slice test. The first element settles
        ///     the common case without a pass; only an array that starts with a NaN pays the vectorized <c>isnan</c> +
        ///     <c>all</c> reduction (its temporary released here).
        /// </summary>
        /// <param name="x">A non-empty float16 / float32 / float64 array (any layout).</param>
        /// <returns>True when no element is a number.</returns>
        private static bool AllNaN(NDArray x)
        {
            if (!double.IsNaN(Utilities.Converts.ToDouble(x.GetAtIndex(0))))
                return false;
            using var nan = np.isnan(x);
            return np.all(nan);
        }

        /// <summary>
        ///     Whether the first reduced line of the staged data (row 0 of its <c>(outer, n)</c> view: NumPy's first slice)
        ///     holds nothing but NaN.
        /// </summary>
        /// <param name="staged">The staged data: the reduced axis innermost and contiguous.</param>
        /// <param name="outerSize">The number of lines.</param>
        /// <param name="n">The length of each line (at least 1).</param>
        /// <returns>True when line 0 is all NaN.</returns>
        private static bool FirstRowAllNaN(NDArray staged, long outerSize, int n)
        {
            if (outerSize == 1)
                return AllNaN(staged);
            using var lines = staged.reshape(Shape.Adopt(new[] { outerSize, (long)n }));
            using var first = lines[0];
            return AllNaN(first);
        }

        /// <summary>
        ///     The nan forms' answer for a q of rank above 2 over data that is NaN everywhere: every slice is
        ///     <c>np.full(q.shape, np.nan, dtype=arr1d.dtype)</c> (NumPy never reaches the rank check), stacked as
        ///     <c>np.apply_along_axis</c> stacks them: q's axes first, then the kept axes.
        /// </summary>
        /// <param name="a">The data (float, all NaN, non-empty).</param>
        /// <param name="axisArr">The reduced axes (already validated), or null for the flattened data.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="qShape">q's dims (rank 3 or more).</param>
        /// <param name="out">An optional output array of the result's shape.</param>
        /// <returns>The NaN result in the data's dtype; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        private static NDArray AllNaNSlicesResult(NDArray a, int[] axisArr, bool keepdims, long[] qShape, NDArray @out)
        {
            var dims = new List<long>(qShape);
            if (axisArr != null)
            {
                var reduced = new HashSet<int>(NormalizeAxes(axisArr, a.ndim));
                for (int i = 0; i < a.ndim; i++)
                    if (!reduced.Contains(i))
                        dims.Add(a.Shape.dimensions[i]);
            }
            var result = new NDArray(a.typecode, Shape.Adopt(dims.ToArray()));
            FillWithNaN(result, a.typecode);
            return FinalizeResult(result, a, axisArr, keepdims, qShape.Length, @out);
        }

        /// <summary>keepdims / out= plumbing shared by the normal and empty-axis paths.</summary>
        /// <param name="resultRaw">The computed result: q's axes, then the kept axes.</param>
        /// <param name="a">The data.</param>
        /// <param name="axisArr">The reduced axes, or null.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="qNdim">How many leading axes are q's (0 for a scalar q).</param>
        /// <param name="out">An optional output array.</param>
        /// <returns>The result, or <paramref name="out"/> itself.</returns>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        private static NDArray FinalizeResult(
            NDArray resultRaw, NDArray a, int[] axisArr, bool keepdims, int qNdim, NDArray @out)
        {
            NDArray result = ApplyKeepdims(resultRaw, a, axisArr, keepdims, qNdim);

            if (@out is not null)
            {
                if (!result.Shape.dimensions.SequenceEqual(@out.Shape.dimensions))
                    throw new ArgumentException(
                        $"Wrong shape of argument 'out', shape={string.Join(",", result.Shape.dimensions)} is required; " +
                        $"got shape={string.Join(",", @out.Shape.dimensions)}.");
                np.copyto(@out, result);
                return @out;
            }
            // A fresh 0-d result (scalar q, axis=None — the np.median/percentile/quantile flat
            // forms) is a numpy SCALAR at the boundary: read-only (PyArray_Return semantics).
            return result.MarkReductionScalar();
        }

        /// <summary>
        ///     Fills <paramref name="result"/> with the dtype's NaN — used for the empty-axis
        ///     median path (NumPy returns nan for the median of an empty slice). Decimal has
        ///     no NaN and no NumPy analogue here, so it falls back to 0.
        /// </summary>
        private static void FillWithNaN(NDArray result, NPTypeCode tc)
        {
            if (result.size == 0) return;
            object nan = tc switch
            {
                NPTypeCode.Double  => double.NaN,
                NPTypeCode.Single  => float.NaN,
                NPTypeCode.Half    => Half.NaN,
                NPTypeCode.Complex => new System.Numerics.Complex(double.NaN, double.NaN),
                NPTypeCode.Decimal => (object)0m,
                _                  => double.NaN,
            };
            result.Storage.InternalArray.Fill(nan);
        }

        /// <summary>
        ///     Result for np.nan* on a totally-empty input: the reduced shape (axes removed,
        ///     or set to 1 under keepdims), filled with NaN, dtype following nanmean (float
        ///     widths preserved, everything else → float64). The q dimension is dropped to
        ///     match NumPy's <c>return np.nanmean(a, axis, out, keepdims)</c> short-circuit.
        /// </summary>
        private static NDArray BuildEmptyNanResult(NDArray a, int[] axisArr, bool keepdims, NDArray @out)
        {
            NPTypeCode tc =
                IsFloatTc(a.typecode)            ? a.typecode :
                a.typecode == NPTypeCode.Complex ? NPTypeCode.Complex :
                a.typecode == NPTypeCode.Decimal ? NPTypeCode.Decimal :
                                                   NPTypeCode.Double;

            int nd = a.ndim;
            // A null axis reduces every axis; an empty tuple reduces none (NumPy's normalize_axis_tuple of () is ()).
            bool reduceAll = axisArr == null;
            HashSet<int> axes = reduceAll ? null : new HashSet<int>(NormalizeAxes(axisArr, nd));

            long[] shapeArr;
            if (keepdims)
            {
                shapeArr = new long[nd];
                for (int i = 0; i < nd; i++)
                    shapeArr[i] = (reduceAll || axes.Contains(i)) ? 1L : a.Shape.dimensions[i];
            }
            else
            {
                var dims = new List<long>();
                if (!reduceAll)
                    for (int i = 0; i < nd; i++)
                        if (!axes.Contains(i)) dims.Add(a.Shape.dimensions[i]);
                shapeArr = dims.ToArray();   // length 0 → 0-D scalar
            }

            NDArray result = shapeArr.Length == 0
                ? new NDArray(tc, new Shape())
                : new NDArray(tc, Shape.Adopt(shapeArr));
            FillWithNaN(result, tc);

            if (@out is not null)
            {
                if (!result.Shape.dimensions.SequenceEqual(@out.Shape.dimensions))
                    throw new ArgumentException(
                        $"Wrong shape of argument 'out', shape={string.Join(",", result.Shape.dimensions)} is required; " +
                        $"got shape={string.Join(",", @out.Shape.dimensions)}.");
                np.copyto(@out, result);
                return @out;
            }
            // A fresh 0-d result (scalar q, axis=None — the np.median/percentile/quantile flat
            // forms) is a numpy SCALAR at the boundary: read-only (PyArray_Return semantics).
            return result.MarkReductionScalar();
        }

        // ── helpers ─────────────────────────────────────────────────────────────────

        /// <summary>True for the three dtypes that can carry a NaN (drives the np.nan* row path).</summary>
        internal static bool IsFloatTc(NPTypeCode tc) =>
            tc == NPTypeCode.Double || tc == NPTypeCode.Single || tc == NPTypeCode.Half;

        /// <summary>
        ///     Output-dtype rule, matching NumPy 2.x exactly (verified against 2.4.2):
        ///     <list type="bullet">
        ///       <item>Discrete methods index via <c>take</c> → preserve the input dtype
        ///             (bool→bool, int→int, float32→float32).</item>
        ///       <item>Continuous methods interpolate via <c>_lerp</c>, whose <c>diff_b_a * t</c> and <c>a + ...</c> promote
        ///             the data's dtype with t's:
        ///         <list type="bullet">
        ///           <item>an integer q ARRAY with an integer virtual index (weibull, averaged_inverted_cdf,
        ///                 interpolated_inverted_cdf: <paramref name="integerWeight"/>) → <c>result_type(data, q)</c>: int32
        ///                 for int32 data with an int8 q, int16 for uint8 data with an int16 q, int64 for a bool q (its
        ///                 arithmetic is int64), float64 for uint64 data, float16 for float16 data with an int8 q.</item>
        ///           <item>a float16 / float32 q ARRAY (<paramref name="qType"/>) → <c>result_type(data, q)</c>: float16 for
        ///                 bool / int8 / uint8 data with a float16 q, float32 for int16 / uint16 / char (or any of those with
        ///                 a float32 q) and float32 data, the data's own dtype for float16 / float32 data, float64 for
        ///                 int32 and wider.</item>
        ///           <item>integer / bool / char → float64.</item>
        ///           <item>float64 → float64.</item>
        ///           <item>float16 / float32 → float64 when <paramref name="weakQ"/> is
        ///                 false (a float64 q array, 0-d included, makes gamma a strong float64 that
        ///                 upcasts), but the input width is preserved for NumPy 2.4's
        ///                 <c>weak_q</c> (a Python int or float q: NEP50 weak promotion). This is
        ///                 also why np.median(float32)→float32.</item>
        ///         </list>
        ///       </item>
        ///     </list>
        ///     Decimal and Complex are NumSharp-only and always preserved.
        /// </summary>
        /// <param name="inTc">The data's dtype.</param>
        /// <param name="method">The estimation method.</param>
        /// <param name="weakQ">Whether q is a weak Python number (NumPy's <c>weak_q</c>).</param>
        /// <param name="qType">q's precision: Half or Single for a float16 / float32 q array, an integer q's dtype (Int64 for a
        ///     bool q), Double otherwise; ignored under a weak q.</param>
        /// <param name="integerWeight">Whether the lerp's weight is an integer array of <paramref name="qType"/> (see the
        ///     summary).</param>
        /// <returns>The result's dtype (the lerp's dtype for an interpolating method).</returns>
        /// <exception cref="NotSupportedException">A dtype the engine has no rule for.</exception>
        internal static NPTypeCode ResolveOutputDtype(NPTypeCode inTc, QuantileMethod method, bool weakQ,
            NPTypeCode qType = NPTypeCode.Double, bool integerWeight = false)
        {
            if (IsDiscrete(method))
                return inTc;   // take() preserves the input dtype for every dtype

            // The int64 weight and a float16 / float32 q array: NumPy's own promotion of the data's dtype with t's (the
            // NumSharp-only decimal and complex keep their dtype, as the rules below do).
            if (inTc != NPTypeCode.Decimal && inTc != NPTypeCode.Complex)
            {
                if (integerWeight)
                    return np.promote_types(inTc, qType).GetTypeCode();
                if (!weakQ && qType is NPTypeCode.Half or NPTypeCode.Single)
                    return np.promote_types(inTc, qType).GetTypeCode();
            }

            // Continuous (interpolating) methods.
            switch (inTc)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    return NPTypeCode.Double;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                    return weakQ ? inTc : NPTypeCode.Double;
                case NPTypeCode.Double:
                    return NPTypeCode.Double;
                case NPTypeCode.Decimal:
                    return NPTypeCode.Decimal;
                case NPTypeCode.Complex:
                    return NPTypeCode.Complex;
                default:
                    throw new NotSupportedException(inTc.ToString());
            }
        }

        /// <summary>
        ///     Managed quantile path for <see cref="System.Numerics.Complex"/> inputs.
        ///     Sorts each outer row lexicographically (Real first, Imaginary as
        ///     tie-break — matches NumPy's complex ordering) then computes each
        ///     <paramref name="q"/> via the requested <paramref name="method"/>.
        ///     Continuous methods interpolate between two adjacent sorted values
        ///     via <c>(1-γ)·a + γ·b</c>, which is well-defined for Complex.
        ///
        ///     Used because Complex has no total order and the IL quantile kernel
        ///     is integer/float only. Throughput is dominated by Array.Sort —
        ///     adequate for typical quantile workloads (np.median, np.percentile)
        ///     which are not in the hot path for most callers.
        /// </summary>
        private static unsafe void ComputeComplexQuantile(
            System.Numerics.Complex* srcBase, long outer, int n,
            QuantileMethod method, double* q, int nQs,
            System.Numerics.Complex* dstBase, long dstOuterStride)
        {
            var scratch = new System.Numerics.Complex[n];
            for (long row = 0; row < outer; row++)
            {
                System.Numerics.Complex* srcRow = srcBase + row * n;
                for (int i = 0; i < n; i++) scratch[i] = srcRow[i];
                Array.Sort(scratch, ComplexLexicographicComparer.Instance);

                for (int qi = 0; qi < nQs; qi++)
                {
                    double qv = q[qi];
                    System.Numerics.Complex result = ComplexQuantileAtMethod(scratch, n, qv, method);
                    dstBase[qi * dstOuterStride + row] = result;
                }
            }
        }

        private static System.Numerics.Complex ComplexQuantileAtMethod(
            System.Numerics.Complex[] sorted, int n, double q, QuantileMethod method)
        {
            // Continuous (Linear / R-7) is the default; other methods adjust α, β.
            // For Complex we support every method by computing the fractional index
            // via the same rule used in the IL kernel and interpolating with the
            // Complex+double mixed operators.
            (double alpha, double beta, bool discrete) = MethodParameters(method);

            double virtualIndex;
            if (discrete)
            {
                virtualIndex = DiscreteIndex(n, q, method);
                int idx = (int)Math.Round(virtualIndex);
                if (idx < 0) idx = 0;
                if (idx >= n) idx = n - 1;
                return sorted[idx];
            }

            virtualIndex = ContinuousVirtualIndex(n, q, alpha, beta);
            int lo = (int)Math.Floor(virtualIndex);
            int hi = (int)Math.Ceiling(virtualIndex);
            if (lo < 0) lo = 0;
            if (hi >= n) hi = n - 1;
            double gamma = virtualIndex - lo;
            if (lo == hi) return sorted[lo];
            // Linear interpolation: (1-γ)·a + γ·b — Complex × double promotes both ops.
            return (1.0 - gamma) * sorted[lo] + gamma * sorted[hi];
        }

        private static (double alpha, double beta, bool discrete) MethodParameters(QuantileMethod m)
        {
            switch (m)
            {
                case QuantileMethod.Linear:                  return (1.0, 1.0, false); // R-7
                case QuantileMethod.Lower:                   return (0, 0, true);
                case QuantileMethod.Higher:                  return (0, 0, true);
                case QuantileMethod.Nearest:                 return (0, 0, true);
                case QuantileMethod.Midpoint:                return (0.5, 0.5, false);
                case QuantileMethod.InvertedCdf:             return (0, 0, true);
                case QuantileMethod.AveragedInvertedCdf:     return (0, 1, false);
                case QuantileMethod.ClosestObservation:      return (0, 0, true);
                case QuantileMethod.InterpolatedInvertedCdf: return (0, 1, false);
                case QuantileMethod.Hazen:                   return (0.5, 0.5, false);
                case QuantileMethod.Weibull:                 return (0, 0, false);
                case QuantileMethod.MedianUnbiased:          return (1.0/3, 1.0/3, false);
                case QuantileMethod.NormalUnbiased:          return (3.0/8, 3.0/8, false);
                default: return (1.0, 1.0, false);
            }
        }

        private static double ContinuousVirtualIndex(int n, double q, double alpha, double beta)
        {
            // NumPy R-α,β rule: virtual_index = q*(n - α - β + 1) + α - 1
            return q * (n - alpha - beta + 1) + alpha - 1;
        }

        private static double DiscreteIndex(int n, double q, QuantileMethod method)
        {
            switch (method)
            {
                case QuantileMethod.Lower:   return Math.Floor(q * (n - 1));
                case QuantileMethod.Higher:  return Math.Ceiling(q * (n - 1));
                case QuantileMethod.Nearest: return Math.Round(q * (n - 1), MidpointRounding.ToEven);
                case QuantileMethod.InvertedCdf:        return Math.Max(0, Math.Ceiling(q * n) - 1);
                case QuantileMethod.ClosestObservation: return Math.Max(0, Math.Round(q * n - 0.5, MidpointRounding.ToEven) - 1);
                default: return q * (n - 1);
            }
        }

        private sealed class ComplexLexicographicComparer : System.Collections.Generic.IComparer<System.Numerics.Complex>
        {
            public static readonly ComplexLexicographicComparer Instance = new();
            public int Compare(System.Numerics.Complex x, System.Numerics.Complex y)
            {
                int r = x.Real.CompareTo(y.Real);
                if (r != 0) return r;
                return x.Imaginary.CompareTo(y.Imaginary);
            }
        }

        /// <summary>
        ///     <c>_ureduce</c>'s <c>axis = _nx.normalize_axis_tuple(axis, nd)</c> (median / percentile / quantile and their
        ///     nan twins): every axis range-checked first, then NumPy's plain "repeated axis".
        /// </summary>
        /// <param name="axes">The axes as given.</param>
        /// <param name="ndim">The input's rank.</param>
        /// <returns>The normalized axes in input order (empty for an empty tuple).</returns>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">An axis appears twice ("repeated axis" — NumPy passes no argname here).</exception>
        internal static int[] NormalizeAxes(int[] axes, int ndim)
        {
            if (axes.Length == 0) return Array.Empty<int>();
            return DefaultEngine.normalize_axis_tuple(axes, ndim);
        }

        /// <summary>
        ///     <c>_ureduce</c>'s keepdims: the result reshaped so every reduced axis of the data is back with length one,
        ///     after q's leading axes.
        /// </summary>
        /// <param name="result">The result: q's axes, then the kept axes.</param>
        /// <param name="source">The data.</param>
        /// <param name="axisArr">The reduced axes, or null for all of them.</param>
        /// <param name="keepdims">Whether to keep the reduced axes (false returns <paramref name="result"/>).</param>
        /// <param name="qNdim">How many leading axes of <paramref name="result"/> are q's.</param>
        /// <returns>The reshaped view, or <paramref name="result"/> itself.</returns>
        internal static NDArray ApplyKeepdims(NDArray result, NDArray source, int[] axisArr, bool keepdims, int qNdim)
        {
            if (!keepdims) return result;

            int qDim = qNdim;
            int srcNd = source.ndim;

            long[] outShape;
            // keepdims sets the REDUCED axes to 1: every axis for a null axis, none for an empty tuple (the branch below
            // keeps each of the result's dimensions then).
            if (axisArr == null)
            {
                outShape = new long[qDim + srcNd];
                for (int i = 0; i < qDim; i++) outShape[i] = result.Shape.dimensions[i];
                for (int i = 0; i < srcNd; i++) outShape[qDim + i] = 1;
            }
            else
            {
                int[] norm = NormalizeAxes(axisArr, srcNd);
                var axisSet = new HashSet<int>(norm);
                outShape = new long[qDim + srcNd];
                for (int i = 0; i < qDim; i++) outShape[i] = result.Shape.dimensions[i];

                int srcIdx = qDim;
                for (int i = 0; i < srcNd; i++)
                {
                    if (axisSet.Contains(i)) outShape[qDim + i] = 1;
                    else outShape[qDim + i] = result.Shape.dimensions[srcIdx++];
                }
            }
            return result.reshape(Shape.Adopt(outShape));
        }

        /// <summary>
        ///     <c>_ureduce</c>'s keepdims out: <c>out[(Ellipsis,) + index_out]</c>, every reduced axis taken at 0 (every axis
        ///     for a flattened reduction) — the view the reduction writes into.
        /// </summary>
        /// <param name="out">The caller's out.</param>
        /// <param name="nd">The data's rank.</param>
        /// <param name="reduced">The normalized reduced axes, or null for every axis.</param>
        /// <returns>The view of <paramref name="out"/>.</returns>
        /// <exception cref="IndexError">An out of fewer dimensions than the data (NumPy's <c>too many indices for array:
        ///     array is 1-dimensional, but 2 were indexed</c>), or an empty axis where a reduced axis is taken at 0.</exception>
        internal static NDArray KeepdimsOutTarget(NDArray @out, int nd, int[] reduced)
        {
            var index = new Slice[1 + nd];
            index[0] = Slice.Ellipsis;
            for (int d = 0; d < nd; d++)
                index[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.Index(0) : Slice.All;
            return @out[index];
        }

        /// <summary>
        ///     <c>PyArray_TakeFrom</c>'s checks of out, in its order: the result's exact shape, a SAFE cast from out's dtype to
        ///     the data's (<c>PyArray_FromArray(out, data dtype)</c>), and writeability (its WRITEBACKIFCOPY base).
        /// </summary>
        /// <param name="target">The out the take writes (keepdims' view of the caller's).</param>
        /// <param name="resultShape">The result's shape.</param>
        /// <param name="data">The data (its dtype).</param>
        /// <exception cref="ValueError">Another shape (<c>output array does not match result of ndarray.take</c>), or a
        ///     read-only out (<c>WRITEBACKIFCOPY base is read-only</c>).</exception>
        /// <exception cref="TypeError">A dtype that does not cast safely to the data's: NumPy's <c>Cannot cast array data
        ///     from dtype('float64') to dtype('float32') according to the rule 'safe'</c> ("scalar" for a 0-d out).</exception>
        private static void TakeOutChecks(NDArray target, long[] resultShape, NDArray data)
        {
            if (!target.Shape.dimensions.AsSpan().SequenceEqual(resultShape))
                throw new ValueError("output array does not match result of ndarray.take");
            if (!np.can_cast(target.dtype, data.dtype, "safe"))
                throw new TypeError($"Cannot cast {(target.ndim == 0 ? "scalar" : "array data")} from {target.dtype.ToString(true)} " +
                                    $"to {data.dtype.ToString(true)} according to the rule 'safe'");
            if (!target.Shape.IsWriteable)
                throw new ValueError("WRITEBACKIFCOPY base is read-only");
        }

        /// <summary>
        ///     The checks of <c>_lerp</c>'s <c>add(a, diff_b_a * t, out=out)</c> before it writes, in the ufunc's order: out's
        ///     writeability, the same_kind cast of the lerp's dtype into out's, then the broadcast of the two inputs (the
        ///     result's shape each) with out — which may be larger than the result, but never stretched.
        /// </summary>
        /// <param name="target">The out the lerp writes.</param>
        /// <param name="lerpType">The lerp's dtype (the add's output).</param>
        /// <param name="resultShape">The result's shape.</param>
        /// <exception cref="ValueError">A read-only out (<c>output array is read-only</c>); shapes that do not broadcast
        ///     (<c>operands could not be broadcast together with shapes (2,) (2,) (3,) </c>) or would stretch out
        ///     (<c>non-broadcastable output operand with shape (2,1) doesn't match the broadcast shape (2,2)</c>).</exception>
        /// <exception cref="ArgumentException">A dtype the lerp's does not cast to under same_kind: NumPy's UFuncTypeError
        ///     <c>Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>
        ///     (the house type of the ufunc out= routes).</exception>
        private static void LerpOutChecks(NDArray target, NPTypeCode lerpType, long[] resultShape)
        {
            NumSharpException.ThrowIfNotWriteable(target.Shape, "output array");
            DType lerp = DType.From(lerpType);
            if (lerpType != target.typecode && !np.can_cast(lerp, target.dtype, "same_kind"))
                throw new ArgumentException($"Cannot cast ufunc 'add' output from {lerp.ToString(true)} to " +
                                            $"{target.dtype.ToString(true)} with casting rule 'same_kind'");

            // NpyIter's broadcast of (a, diff_b_a * t, out): right-aligned, a dimension of 1 stretches; out may not stretch.
            long[] outDims = target.Shape.dimensions;
            int nd = Math.Max(resultShape.Length, outDims.Length);
            var broadcast = new long[nd];
            bool compatible = true;
            for (int i = 0; i < nd; i++)
            {
                long r = i < nd - resultShape.Length ? 1 : resultShape[i - (nd - resultShape.Length)];
                long o = i < nd - outDims.Length ? 1 : outDims[i - (nd - outDims.Length)];
                if (r != o && r != 1 && o != 1)
                {
                    compatible = false;
                    break;
                }
                broadcast[i] = r == 1 ? o : r;
            }
            string result = PyTuple(resultShape), outShape = PyTuple(outDims);
            if (!compatible)
                throw new ValueError($"operands could not be broadcast together with shapes {result} {result} {outShape} ");
            if (!broadcast.AsSpan().SequenceEqual(outDims))
                throw new ValueError($"non-broadcastable output operand with shape {outShape} doesn't match the broadcast " +
                                     $"shape {PyTuple(broadcast)}");
        }

        /// <summary>
        ///     Whether out receives the lerp itself (the kernel writes out's dtype): a float or complex out other than the
        ///     lerp's dtype, whose t &gt;= 0.5 branch NumPy computes in out's dtype. An integer lerp (an integer q's weight
        ///     over integer data) takes one element, so its value cast afterwards is the same and it keeps its own buffer.
        /// </summary>
        /// <param name="lerpType">The lerp's dtype.</param>
        /// <param name="outType">out's dtype.</param>
        /// <param name="integerWeight">Whether the lerp runs with an integer weight (its float lerp, over float data, still
        ///     computes the out's branch in out's dtype).</param>
        /// <returns>True when the kernel must write out's dtype.</returns>
        private static bool LerpsIntoOut(NPTypeCode lerpType, NPTypeCode outType, bool integerWeight)
            => lerpType != outType
               && outType is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex
               && lerpType is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double;

        /// <summary>
        ///     The result written where NumPy writes it: returned (keepdims applied; a 0-d result marked as NumPy's scalar),
        ///     or into out by its route — the take's unsafe cast and its NaN-slice <c>np.copyto</c> check, the lerp's
        ///     (broadcast) write, the nan forms' <c>out[...] = result</c>, or the legacy copy.
        /// </summary>
        /// <param name="resultRaw">The computed result: q's axes, then the kept axes.</param>
        /// <param name="a">The data.</param>
        /// <param name="axisArr">The reduced axes, or null.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="qNdim">How many leading axes of the result are q's.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="target">The view of out the reduction writes (keepdims), or out itself.</param>
        /// <param name="outRoute">0 none, 1 take, 2 lerp, 3 the nan forms, 4 legacy.</param>
        /// <param name="staged">The staged data (the NaN-slice check of the take route), or null for an empty array.</param>
        /// <param name="keptDims">How many kept axes the data has (0: NumPy's <c>arr[-1, ...]</c> is a scalar).</param>
        /// <returns>The result, or <paramref name="out"/> itself.</returns>
        /// <exception cref="ArgumentException">The legacy route: out of another shape.</exception>
        /// <exception cref="ValueError">The nan forms' assignment does not broadcast, or out is read-only.</exception>
        /// <exception cref="TypeError">The take route: a slice holding NaN is copied into an out the data's dtype does not
        ///     cast to under same_kind (NumPy's <c>np.copyto(result, arr[-1, ...], where=slices_having_nans)</c>: <c>Cannot
        ///     cast scalar from dtype('float64') to dtype('int64') according to the rule 'same_kind'</c>, "array data" with
        ///     kept axes) — after the take wrote out, as NumPy's does.</exception>
        private static NDArray FinishResult(NDArray resultRaw, NDArray a, int[] axisArr, bool keepdims, int qNdim, NDArray @out,
            NDArray target, int outRoute, NDArray staged, int keptDims)
        {
            switch (outRoute)
            {
                case 0:
                    return FinalizeResult(resultRaw, a, axisArr, keepdims, qNdim, null);
                case 1:
                    // take(..., out=out): the values through an unsafe cast; then `np.copyto(result, arr[-1, ...],
                    // where=slices_having_nans)` (same_kind from the data's dtype) wherever a slice holds NaN.
                    np.copyto(target, resultRaw, "unsafe");
                    if (staged is not null && IsFloatTc(a.typecode) && !np.can_cast(a.dtype, target.dtype, "same_kind")
                        && HasNaN(staged))
                        throw new TypeError($"Cannot cast {(keptDims == 0 ? "scalar" : "array data")} from " +
                                            $"{a.dtype.ToString(true)} to {target.dtype.ToString(true)} according to the rule 'same_kind'");
                    return @out;
                case 2:
                    // _lerp's add / subtract wrote out (broadcast; the buffer already has out's dtype when the lerp computed
                    // in it). The NaN slices' np.copyto reaches every float / complex out under same_kind.
                    np.copyto(target, resultRaw, "unsafe");
                    return @out;
                case 3:
                    // the nan forms: `out[...] = result` — an array broadcast through the unsafe cast, but a 0-d result
                    // (a 0-d q over a flat line) is a NumPy SCALAR, stored by out's setitem.
                    if (resultRaw.ndim == 0)
                        StoreNumPyScalar(target, resultRaw);
                    else
                        np.copyto(target, resultRaw, "unsafe");
                    return @out;
                default:
                    return FinalizeResult(resultRaw, a, axisArr, keepdims, qNdim, @out);
            }
        }

        /// <summary>
        ///     The layout of the nan forms' result along an axis: <c>_nanquantile_ureduce_func</c> calls
        ///     <c>np.apply_along_axis</c>, which fills a C-contiguous buffer of the kept axes followed by q's axes and returns
        ///     it transposed (q's axes where the reduced axis was), and then moves q's axes first — a view that does not own
        ///     its data and whose q axes are the innermost in memory (F-ordered for one kept axis and a 1-d q). A scalar q
        ///     leaves the buffer's C order, still as a view.
        /// </summary>
        /// <param name="result">The computed result: q's axes, then the kept axes (C-contiguous).</param>
        /// <param name="qNdim">How many leading axes are q's.</param>
        /// <returns>The same values in NumPy's layout.</returns>
        private static NDArray AlongAxisLayout(NDArray result, int qNdim)
        {
            if (qNdim == 0)
                return result.view();
            int nd = result.ndim;
            var front = new int[qNdim];
            var back = new int[qNdim];
            for (int i = 0; i < qNdim; i++)
            {
                front[i] = i;
                back[i] = nd - qNdim + i;
            }
            // apply_along_axis's buffer: the kept axes, then q's, in C order; nanquantile's moveaxis then views it with q's
            // axes first.
            NDArray buffer = np.moveaxis(result, front, back).copy();
            return np.moveaxis(buffer, back, front);
        }

        /// <summary>
        ///     NumPy's <c>out[...] = result</c> when the result is a NumPy SCALAR (the nan forms' answer for a 0-d q over a flat
        ///     line): <c>PyArray_Pack</c> of the scalar into out's dtype (<see cref="ReductionInitial.PackNumPyScalar"/>). Only a
        ///     SIGNED integer out treats a NumPy scalar of another type as its own (<c>signed_integers_is_known_scalar_types</c>)
        ///     and stores it with its setitem: Python's <c>int()</c> (NaN and infinity raise), the C long conversion, then
        ///     <c>@TYPE@_safe_pyint_setitem</c>'s range check (an int8 / int16 / int32 out refuses a value outside its range).
        ///     Every other out (unsigned, bool, float, complex) knows only Python's own scalar types, so it CASTS the NumPy
        ///     scalar (<c>npy_cast_raw_scalar_item</c>): an unsigned out wraps a negative value and takes the cast's value for
        ///     NaN and infinity (0 into uint8, 2^63 into uint64), a bool out takes its truth. A text out or a text answer
        ///     (the weighted branch picks elements of S / U / StringDType data) keeps the string store.
        /// </summary>
        /// <param name="target">The out (or its keepdims view): one element.</param>
        /// <param name="scalar">The 0-d result.</param>
        /// <exception cref="ValueError">A read-only out (<c>assignment destination is read-only</c>, checked first, as
        ///     NumPy's subscript assignment does), or a NaN into a signed integer out (<c>cannot convert float NaN to
        ///     integer</c>).</exception>
        /// <exception cref="OverflowException">Into a signed integer out: an infinity (<c>cannot convert float infinity to
        ///     integer</c>), a value outside C long (<c>Python int too large to convert to C long</c>; C long is 64 bits here:
        ///     NumPy on Windows has a 32-bit long, and the LP64 (Linux / macOS) answer is the one pinned), or a value outside
        ///     the out's range (<c>Python integer 300 out of bounds for int8</c>).</exception>
        internal static unsafe void StoreNumPyScalar(NDArray target, NDArray scalar)
        {
            NumSharpException.ThrowIfNotWriteable(target.Shape);
            NPTypeCode tt = target.typecode;
            if (!tt.IsNumerical() || !scalar.typecode.IsNumerical())
            {
                // A text out or a text answer (S / U / StringDType: the weighted branch picks elements of text data) goes
                // through the string store, whose casts and refusals are the text paths' own.
                np.copyto(target, scalar, "unsafe");
                return;
            }

            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackNumPyScalar(scalar.Storage.GetAtIndex(0), tt,
                new Span<byte>(slot, ReductionInitial.SlotBytes));
            using var packed = new NDArray(tt, Shape.Scalar, false);
            Buffer.MemoryCopy(slot, (byte*)packed.Address, packed.dtypesize, packed.dtypesize);
            // The packed element already has out's dtype, so the store is exact (and reaches a keepdims view's element).
            np.copyto(target, packed, "unsafe");
        }

        /// <summary>Whether a float array holds a NaN anywhere (any of its lines holds one).</summary>
        /// <param name="x">The array.</param>
        /// <returns>True when an element is NaN.</returns>
        private static bool HasNaN(NDArray x)
        {
            using var nan = np.isnan(x);
            return np.any(nan);
        }

        /// <summary>
        ///     The lookups of an array with no element (the take's index check, or <c>arr[previous_indexes]</c> /
        ///     <c>arr[next_indexes]</c>): each index of the plan in q's order must lie in <c>[-n, n)</c>.
        /// </summary>
        /// <param name="raw">The plan's raw indexes (2 per quantile).</param>
        /// <param name="qLen">The number of quantiles.</param>
        /// <param name="slot">0 for the previous (or taken) index, 1 for the next one.</param>
        /// <param name="n">The axis' length.</param>
        /// <exception cref="IndexOutOfRangeException">NumPy's IndexError <c>index k is out of bounds for axis 0 with size
        ///     n</c> for the first index out of range (k as computed).</exception>
        private static void CheckLookups(long[] raw, int qLen, int slot, long n)
        {
            for (int j = 0; j < qLen; j++)
            {
                long k = raw[2 * j + slot];
                if (k < -n || k >= n)
                    throw new IndexOutOfRangeException($"index {k} is out of bounds for axis 0 with size {n}");
            }
        }

        /// <summary>The ascending distinct values of a list (the partition's kth list).</summary>
        /// <param name="values">The values (not modified).</param>
        /// <returns>A new sorted array without repeats.</returns>
        private static int[] SortedDistinct(int[] values)
        {
            var sorted = (int[])values.Clone();
            Array.Sort(sorted);
            int u = 0;
            for (int i = 0; i < sorted.Length; i++)
                if (u == 0 || sorted[u - 1] != sorted[i])
                    sorted[u++] = sorted[i];
            Array.Resize(ref sorted, u);
            return sorted;
        }

        /// <summary>NumPy's TypeError for <c>_lerp</c>'s <c>b - a</c> over bool data.</summary>
        /// <returns>The exception to throw.</returns>
        private static TypeError BooleanSubtractError()
            => new TypeError("numpy boolean subtract, the `-` operator, is not supported, use the "
                             + "bitwise_xor, the `^` operator, or the logical_xor function instead.");

        /// <summary>A shape as NumPy prints it in error texts: <c>(2,3)</c>, <c>(3,)</c>, <c>()</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The text.</returns>
        private static string PyTuple(long[] dims)
            => dims.Length == 0 ? "()" : dims.Length == 1 ? $"({dims[0]},)" : "(" + string.Join(",", dims) + ")";

        private static int TypeSize(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => 1,
            NPTypeCode.Byte    => 1,
            NPTypeCode.SByte   => 1,
            NPTypeCode.Int16   => 2,
            NPTypeCode.UInt16  => 2,
            NPTypeCode.Half    => 2,
            NPTypeCode.Char    => 2,
            NPTypeCode.Int32   => 4,
            NPTypeCode.UInt32  => 4,
            NPTypeCode.Single  => 4,
            NPTypeCode.Int64   => 8,
            NPTypeCode.UInt64  => 8,
            NPTypeCode.Double  => 8,
            NPTypeCode.Decimal => 16,
            NPTypeCode.Complex => 16,
            _                  => throw new NotSupportedException(tc.ToString()),
        };
    }
}
