// The functions of numpy/lib/_nanfunctions_impl.py: nansum, nanprod, nanmin, nanmax, nancumsum, nancumprod, nanmean,
// nanvar, nanstd, nanquantile, nanpercentile and nanmedian (the np-layer dtype plan's phase PE moved the last eight here).

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
        /// <summary>
        ///     NumPy's <c>np.nanmin</c>, the whole call: the minimum ignoring NaNs, which on an ndarray IS
        ///     <c>np.fmin.reduce</c> (NumPy's "All-NaN slice" RuntimeWarning is not modeled), with NumPy's <c>out=</c>,
        ///     <c>initial=</c> and <c>where=</c>.
        /// </summary>
        /// <param name="a">The array (integer, bool, char and decimal arrays hold no NaN: the plain <c>np.min</c>).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result (the loop is
        ///     <c>promote_types(out, a)</c>), or null for a fresh result.</param>
        /// <param name="initial">NumPy's <c>initial=</c>: the value every output starts from, packed into the loop dtype as
        ///     NumPy's <c>PyArray_Pack</c> does, or null for none.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask, only the selected elements reduce; it needs
        ///     <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d one for a full
        ///     reduction without keepdims. An all-NaN slice gives its first NaN, bits intact.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation fmin which has no identity</c>, NumPy's ValueError); a non-bool mask.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask without
        ///     <paramref name="initial"/>, an <paramref name="initial"/> the loop cannot hold.</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="TypeError">Text, or a complex <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks>Widened with <paramref name="out"/>, <paramref name="initial"/> and <paramref name="where"/> in 0.71.0
        ///     (the np-layer dtype plan, phase PE, E5).</remarks>
        public abstract NDArray NanMin(NDArray a, int? axis = null, bool keepdims = false, NDArray @out = null,
            object initial = null, NDArray @where = null);

        /// <summary>
        ///     NumPy's <c>np.nanmax</c>, the whole call: the maximum ignoring NaNs, which on an ndarray IS
        ///     <c>np.fmax.reduce</c> (NumPy's "All-NaN slice" RuntimeWarning is not modeled), with NumPy's <c>out=</c>,
        ///     <c>initial=</c> and <c>where=</c>.
        /// </summary>
        /// <param name="a">The array (integer, bool, char and decimal arrays hold no NaN: the plain <c>np.max</c>).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="out">NumPy's <c>out=</c>: the array that receives the result (the loop is
        ///     <c>promote_types(out, a)</c>), or null for a fresh result.</param>
        /// <param name="initial">NumPy's <c>initial=</c>: the value every output starts from, packed into the loop dtype as
        ///     NumPy's <c>PyArray_Pack</c> does, or null for none.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask, only the selected elements reduce; it needs
        ///     <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise a fresh array, a read-only 0-d one for a full
        ///     reduction without keepdims. An all-NaN slice gives its first NaN, bits intact.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="ArgumentException">An empty reduction without <paramref name="initial"/> (<c>zero-size array to
        ///     reduction operation fmax which has no identity</c>, NumPy's ValueError); a non-bool mask.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>, a mask without
        ///     <paramref name="initial"/>, an <paramref name="initial"/> the loop cannot hold.</exception>
        /// <exception cref="OverflowException">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <exception cref="TypeError">Text, or a complex <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="IncorrectShapeException">The mask or <paramref name="out"/> does not broadcast.</exception>
        /// <remarks>Widened with <paramref name="out"/>, <paramref name="initial"/> and <paramref name="where"/> in 0.71.0
        ///     (the np-layer dtype plan, phase PE, E5).</remarks>
        public abstract NDArray NanMax(NDArray a, int? axis = null, bool keepdims = false, NDArray @out = null,
            object initial = null, NDArray @where = null);

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

        /// <summary>
        ///     NumPy's <c>np.nanquantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *,
        ///     weights=None)</c>: <see cref="Quantile"/> of the values that are not NaN; a slice holding nothing but NaN
        ///     gives NaN.
        /// </summary>
        /// <param name="a">The data, a plain array (a <c>StringDType</c> array computes the quantile of what its NaN-like
        ///     NAs leave; fixed-width text meets <c>isnan</c>'s refusal).</param>
        /// <param name="q">The quantiles in [0, 1], NumPy's array_like (the forms of <see cref="Quantile"/>'s q).</param>
        /// <param name="axis">The reduced axes, or null for the flattened data; an EMPTY array reduces nothing.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">One of NumPy's thirteen estimation methods, matched exactly.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (only with <c>inverted_cdf</c>), or null. With weights, empty data
        ///     answers <c>np.nanmean</c>; other data runs the weighted branch over each line's non-NaN values.</param>
        /// <returns>The quantiles of the non-NaN values (NaN for an all-NaN slice; for q of two or more dimensions over
        ///     merged axes, q's axes are moved as NumPy's <c>_nanquantile_ureduce_func</c> moves them);
        ///     <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex; a text q; fixed-width text data (<c>isnan</c> has no
        ///     text loop, or <c>np.nanmean</c>'s add.reduce refuses empty text); bool data with an interpolating method;
        ///     weights of another shape without an axis, or text weights.</exception>
        /// <exception cref="ValueError">A quantile out of range; an empty q of rank other than 1; a repeated axis; a q of rank
        ///     above 2 (except over data that is NaN everywhere); the weights' checks.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a null q, or <paramref name="method"/> is
        ///     null.</exception>
        /// <exception cref="ArgumentException">An unknown method name (except over data that is NaN everywhere), or an out
        ///     of another shape.</exception>
        /// <exception cref="IndexError">Data that is not empty, under keepdims, with an out NumPy cannot index
        ///     (<c>_ureduce</c>'s prologue), before the text refusals.</exception>
        /// <exception cref="InvalidCastException">A q array of a complex dtype.</exception>
        /// <exception cref="OverflowException">An integer q array whose virtual indexes leave its dtype.</exception>
        /// <remarks>
        ///     NumPy writes nanquantile in Python; DefaultEngine's override runs its statements in NumPy's order.
        ///     <c>np.nanquantile</c> calls it after its class rule; an override replaces it for every array whose engine it
        ///     is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray NanQuantile(NDArray a, object q, int[] axis = null, NDArray @out = null,
            bool overwrite_input = false, string method = "linear", bool keepdims = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method='linear',
        ///     keepdims=False, *, weights=None)</c>: <see cref="NanQuantile"/> with q given in percent,
        ///     <c>q = np.true_divide(q, 100)</c> first.
        /// </summary>
        /// <param name="a">The data, a plain array.</param>
        /// <param name="q">The percentiles in [0, 100], NumPy's array_like (the forms of <see cref="Quantile"/>'s q).</param>
        /// <param name="axis">The reduced axes, or null for the flattened data; an EMPTY array reduces nothing.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">One of NumPy's thirteen estimation methods, matched exactly.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (only with <c>inverted_cdf</c>), or null.</param>
        /// <returns>The percentiles of the non-NaN values (NaN for an all-NaN slice); <paramref name="out"/> itself when one
        ///     is given.</returns>
        /// <exception cref="ValueError">A percentile outside [0, 100] (on the quotient), and the other ValueErrors of
        ///     <see cref="NanQuantile"/>.</exception>
        /// <exception cref="TypeError">The TypeErrors of <see cref="NanQuantile"/>.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a null q, or <paramref name="method"/> is
        ///     null.</exception>
        /// <exception cref="ArgumentException">An unknown method name, or an out of another shape.</exception>
        /// <exception cref="IndexError">Data that is not empty, under keepdims, with an out NumPy cannot index.</exception>
        /// <exception cref="InvalidCastException">A q array of a complex dtype.</exception>
        /// <remarks>
        ///     NumPy writes nanpercentile in Python; <c>np.nanpercentile</c> calls this after its class rule. An override
        ///     replaces it for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the
        ///     internal class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray NanPercentile(NDArray a, object q, int[] axis = null, NDArray @out = null,
            bool overwrite_input = false, string method = "linear", bool keepdims = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False)</c>: the median of the
        ///     values that are not NaN (the mean of the two central values of an even count); a slice that is entirely NaN,
        ///     or empty, gives NaN.
        /// </summary>
        /// <param name="a">The data, a plain array (bool computes as its 0 / 1 values; text is refused as NumPy refuses it,
        ///     a <c>StringDType</c> array drops its NaN-like NAs and takes the median of the rest).</param>
        /// <param name="axis">The reduced axes, or null for the flattened data; an EMPTY array reduces nothing. A one-axis
        ///     call passes a one-element array.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The NaN-ignoring median; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats; a <c>StringDType</c> array's errors (a None-like NA in a sorted
        ///     line, an empty input's mean, the store into out).</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data that NumPy's NaN test (or, for empty data, its <c>nanmean</c>)
        ///     refuses.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> row result NumPy stores in an OBJECT
        ///     array.</exception>
        /// <exception cref="IndexError">Data that is not empty, under keepdims, with an out NumPy cannot index.</exception>
        /// <remarks>
        ///     NumPy writes nanmedian in Python; <c>np.nanmedian</c> calls this after its class rule. An override replaces it
        ///     for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal
        ///     class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray NanMedian(NDArray a, int[] axis = null, NDArray @out = null, bool overwrite_input = false,
            bool keepdims = false);
    }
}
