// The functions of numpy/lib/_nanfunctions_impl.py: nansum, nanprod, nanmin and nanmax so far; the plan's
// phase PE moves the rest of the family here (nanquantile, nanmedian, nancumsum, nanmean, nanvar, ...).

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        // NaN-aware reductions
        public abstract NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanProd(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false);
        public abstract NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.nancumsum</c>: the running sum with every NaN counted as zero (<c>_replace_nan(a, 0)</c>, then
        ///     <c>np.cumsum</c>). A leading run of NaNs therefore gives zeros, never NaN.
        /// </summary>
        /// <param name="a">
        ///     The array, a plain one (the facade decides the <c>numpy.matrix</c> class first). Text is not inexact, so it
        ///     is <c>np.cumsum</c>'s text: StringDType concatenates, fixed-width text is refused or parsed under a numeric
        ///     <paramref name="dtype"/>.
        /// </param>
        /// <param name="axis">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop and result dtype, or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape that receives it and is returned, or null.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric loop cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     NumPy writes nancumsum in Python (<c>numpy/lib/_nanfunctions_impl.py</c>); DefaultEngine's override runs
        ///     those statements over <see cref="ReduceCumAdd"/>. An override replaces <c>np.nancumsum</c> for every array
        ///     whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE).
        /// </remarks>
        public abstract NDArray NanCumSum(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.nancumprod</c>: the running product with every NaN counted as one (<c>_replace_nan(a, 1)</c>,
        ///     then <c>np.cumprod</c>).
        /// </summary>
        /// <param name="a">
        ///     The array, a plain one (the facade decides the <c>numpy.matrix</c> class first). Text is not inexact: a
        ///     numeric <paramref name="dtype"/> parses it, and a parsed <c>'nan'</c> is multiplied, not replaced by one.
        /// </param>
        /// <param name="axis">The accumulated axis (negative counts from the end), or null for the flattened array.</param>
        /// <param name="dtype">The loop and result dtype, or null for NEP 50's accumulator of the input.</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape that receives it and is returned, or null.</param>
        /// <returns><paramref name="out"/> itself when given, else a fresh array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A text operand or request NumPy's loop resolution refuses.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or text a numeric dtype cannot parse.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     DefaultEngine's override runs NumPy's statements over <see cref="ReduceCumMul"/>. An override replaces
        ///     <c>np.nancumprod</c> for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE).
        /// </remarks>
        public abstract NDArray NanCumProd(NDArray a, int? axis = null, DType dtype = null, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.nanmean</c>: the mean of the non-NaN elements, the NaN-free sum divided by their count. A
        ///     slice without one gives NaN (the division's 0 / 0). An input that cannot hold NaN (integer, bool, char,
        ///     decimal, text) is <c>np.mean</c>'s, keywords included.
        /// </summary>
        /// <param name="a">The array, a plain one (the facade decides the <c>numpy.matrix</c> class first).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">
        ///     The loop of the sum (NumPy's <c>dtype=</c>), or null for the input's dtype. Must be inexact for an inexact
        ///     input.
        /// </param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <returns>
        ///     The mean in the sum's dtype; 0-d (NumPy's scalar) for a flat reduction without
        ///     <paramref name="keepdims"/>.
        /// </returns>
        /// <exception cref="TypeError">
        ///     An inexact input with a non-inexact <paramref name="dtype"/>: "If a is inexact, then dtype must be inexact"
        ///     (raised before the axis is checked, as in NumPy).
        /// </exception>
        /// <exception cref="AxisError">
        ///     <paramref name="axis"/> outside the operand, except 0 and -1 on a 0-d inexact operand, which reduce flat
        ///     (ufunc.reduce's rule: nanmean's first axis consumer is np.sum).
        /// </exception>
        /// <remarks>
        ///     NumPy writes nanmean in Python (<c>numpy/lib/_nanfunctions_impl.py</c>); DefaultEngine's override runs those
        ///     statements: the sum over NumPy's K-order copy in its own schedule (a float16 or float32 input sums in its own
        ///     dtype), then the division by the int64 count in the float64 loop, or Smith's complex loop for complex. An
        ///     override replaces <c>np.nanmean</c> for every array whose engine it is. New in 0.71.0 (the np-layer dtype
        ///     plan, phase PE).
        /// </remarks>
        public abstract NDArray NanMean(NDArray a, int? axis = null, DType dtype = null, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.nanvar</c>: the variance of the non-NaN elements. It differs from <c>np.var</c> in three
        ///     observable ways: the deviations are rounded into the input's dtype before they are squared, a complex square
        ///     is the real part of <c>x * conj(x)</c>, and a slice without degrees of freedom is NaN (np.var gives +inf).
        /// </summary>
        /// <param name="a">The array, a plain one (the facade decides the <c>numpy.matrix</c> class first).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">
        ///     The loop of both sums (NumPy's <c>dtype=</c>), or null for the input's dtype. Must be inexact for an inexact
        ///     input.
        /// </param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is the non-NaN count minus <paramref name="ddof"/>.</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <returns>
        ///     The variance: the loop dtype, float64 for a complex input without <paramref name="dtype"/>; 0-d (NumPy's
        ///     scalar) for a flat reduction without <paramref name="keepdims"/>.
        /// </returns>
        /// <exception cref="TypeError">
        ///     An inexact input with a non-inexact <paramref name="dtype"/>: "If a is inexact, then dtype must be inexact".
        /// </exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand.</exception>
        /// <remarks>
        ///     An input that cannot hold NaN (integer, bool, char, decimal, text) is <c>np.var</c>'s, keywords included.
        ///     DefaultEngine's override runs NumPy's <c>_nanfunctions_impl.nanvar</c> statements. An override replaces
        ///     <c>np.nanvar</c> for every array whose engine it is; <see cref="NanStd"/> is a separate member, so an engine
        ///     that changes one should change both. New in 0.71.0 (the np-layer dtype plan, phase PE).
        /// </remarks>
        public abstract NDArray NanVar(NDArray a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.nanstd</c>: the square root of <see cref="NanVar"/>, in the variance's dtype.
        /// </summary>
        /// <param name="a">The array, a plain one (the facade decides the <c>numpy.matrix</c> class first).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for a flat reduction.</param>
        /// <param name="dtype">
        ///     The loop of both sums (NumPy's <c>dtype=</c>), or null for the input's dtype. Must be inexact for an inexact
        ///     input.
        /// </param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is the non-NaN count minus <paramref name="ddof"/>.</param>
        /// <param name="keepdims">Keep the reduced axis as a length-1 dimension.</param>
        /// <returns>
        ///     The standard deviation: the loop dtype, float64 for a complex input without <paramref name="dtype"/>; 0-d for
        ///     a flat reduction without <paramref name="keepdims"/>.
        /// </returns>
        /// <exception cref="TypeError">
        ///     An inexact input with a non-inexact <paramref name="dtype"/>: "If a is inexact, then dtype must be inexact".
        /// </exception>
        /// <exception cref="AxisError"><paramref name="axis"/> outside the operand.</exception>
        /// <remarks>
        ///     An input that cannot hold NaN is <c>np.std</c>'s, keywords included. New in 0.71.0 (the np-layer dtype plan,
        ///     phase PE).
        /// </remarks>
        public abstract NDArray NanStd(NDArray a, int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false);
    }
}
