using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis, ignoring NaNs.
        ///     <paramref name="q"/> must be in the range [0, 1]. A slice that is entirely NaN
        ///     (or empty) yields NaN, matching NumPy's "All-NaN slice" behaviour.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The quantile, in [0, 1].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The NaN-ignoring quantile; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data: NumPy's NaN test (<c>isnan</c> has no text loop), or for an
        ///     empty input <c>np.nanmean</c>'s add.reduce refusal.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // NumPy's np.nanquantile body (q's checks, the weighted branch, the text routes, the NaN-ignoring quantile) is
            // TensorEngine.NanQuantile's: the facade holds only the class rule above.
            return a.TensorEngine.NanQuantile(a, q, axis.HasValue ? new[] { axis.Value } : null, @out,
                overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     Compute the q-th quantiles of the data along the specified axis, ignoring NaNs.
        ///     Each value in <paramref name="q"/> must be in [0, 1]. The result's first axis is q.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The quantiles, each in [0, 1].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The NaN-ignoring quantiles, q's axis first; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="q"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data: NumPy's NaN test (<c>isnan</c> has no text loop), or for an
        ///     empty input <c>np.nanmean</c>'s add.reduce refusal.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // NumPy's np.nanquantile body is TensorEngine.NanQuantile's (a null q is reported there, after the complex
            // check).
            return a.TensorEngine.NanQuantile(a, q, axis.HasValue ? new[] { axis.Value } : null, @out,
                overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the reduced axes are
        ///     merged into one before the partition, a slice that holds only NaNs (or nothing) gives NaN, and an EMPTY
        ///     axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The quantile, in [0, 1].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The NaN-ignoring quantile over the reduced axes; <paramref name="out"/> itself when one is
        ///     given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]"),
        ///     or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data: NumPy's NaN test (<c>isnan</c> has no text loop), or for an empty input
        ///     <c>np.nanmean</c>'s add.reduce refusal, after the axis checks (see
        ///     <see cref="Backends.Strings.NDStringUfuncs.QuantileAnswer"/>).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (<c>np.nanquantile(m, 0.5, axis=(0, 1))</c> returns the input's values).</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.nanquantile body is TensorEngine.NanQuantile's.
            return a.TensorEngine.NanQuantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantiles over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the result's first
        ///     axis is q, a slice that holds only NaNs (or nothing) gives NaN, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The quantiles, each in [0, 1].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>One NaN-ignoring quantile per element of <paramref name="q"/> along the first axis, over the reduced
        ///     axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="q"/>, <paramref name="a"/> or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="ValueError">A quantile is outside [0, 1] ("Quantiles must be in the range [0, 1]"), or an
        ///     axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data: NumPy's NaN test (<c>isnan</c> has no text loop), or for an empty input
        ///     <c>np.nanmean</c>'s add.reduce refusal, after the axis checks (see
        ///     <see cref="Backends.Strings.NDStringUfuncs.QuantileAnswer"/>).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code.</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.nanquantile body is TensorEngine.NanQuantile's (a null q is reported there, after the complex
            // check).
            return a.TensorEngine.NanQuantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     NDArray-q overload — NumPy's <c>np.nanquantile(a, q_array)</c>: q of any real dtype, of rank 0 to 2 (a 2-d q
        ///     keeps both axes in front of the result; rank 3 and above is refused after the axis checks).
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">An array of quantiles in [0, 1]. An integer or bool q makes NumPy's 'linear' virtual index
        ///     <c>(n - 1) * q</c> an integer, which it TAKES instead of interpolating (the data's dtype:
        ///     <c>np.nanquantile([1, 2, 3], np.array(1))</c> is the int64 3), and an array q is a STRONG float64 weight
        ///     (<c>np.nanquantile(f32, np.array(0.5))</c> is float64).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantiles of the non-NaN values: one per element of q (a scalar result for a 0-d q), with the axis reduced; for empty numeric data and an empty text q, NumPy's <c>np.nanmean(a, axis)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/> (NumPy's [0, 1] check: Python's <c>'&lt;=' not supported
        ///     between instances of 'float' and 'numpy.str_'</c> for a 1-d q of fewer than 10 elements, <c>q.min()</c>'s
        ///     <c>minimum</c> refusal otherwise). Text data: NumPy's NaN test (<c>isnan</c> has no text loop), or the
        ///     add.reduce refusal of <c>np.nanmean</c> for empty text data.</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>), an empty q
        ///     of rank other than 1 (its <c>q.min()</c>), an unknown method (checked only when some slice holds a value), a q
        ///     of rank above 2 (<c>q must be a scalar or 1d</c>, unless every slice is all NaN), or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // NumPy's np.nanquantile body is TensorEngine.NanQuantile's: a null or text q, the range check at any rank,
            // an integer q's index taken instead of interpolated, a weak C# literal, q's rank, all in NumPy's order.
            return a.TensorEngine.NanQuantile(a, q, axis.HasValue ? new[] { axis.Value } : null, @out,
                overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     NDArray-q overload over SEVERAL axes at once — NumPy's <c>np.nanquantile(a, q_array, axis=(i, j))</c>: the
        ///     reduced axes are merged into one, the result's leading axes are q's, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">An array of quantiles in [0, 1], of rank 0 to 2 (see the one-axis overload for the integer take
        ///     and the strong weight).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantiles of the non-NaN values over the reduced axes; <paramref name="out"/> itself when one is
        ///     given; for empty numeric data, NumPy's <c>np.nanmean(a, axis)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex ("a must be an array of real numbers", NumPy's first
        ///     statement); a text <paramref name="q"/> (its [0, 1] check refuses); text data (NumPy's NaN test: <c>isnan</c>
        ///     has no text loop), after the axis checks.</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1], an empty q of rank other than 1, an unknown method, a q of
        ///     rank above 2 (unless every slice is all NaN), or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain, as for every
        ///     tuple-axis overload.</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.nanquantile body is TensorEngine.NanQuantile's (see the one-axis overload).
            return a.TensorEngine.NanQuantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile ignoring NaNs for a Python INT q (a C# integer literal): NumPy's <c>q = np.asanyarray(q)</c>
        ///     makes it an int64 array, whose 'linear' virtual index <c>(n - 1) * q</c> is an integer that NumPy TAKES
        ///     instead of interpolating — the data's own dtype and element (<c>np.nanquantile([1, 2, 3], 1)</c> is the int64
        ///     3, <c>np.nanquantile([1., inf, nan], 1)</c> is inf where interpolating at 1.0 makes <c>inf - inf</c>).
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">The quantile: 0 or 1 (anything else is outside [0, 1]).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantile of the non-NaN values with the axis reduced (a scalar result over every axis).</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex, or text data (NumPy's NaN test).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] (<c>Quantiles must be in the range [0,
        ///     1]</c>), or an unknown method.</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>
        ///     C# binds an integer literal here rather than to the <see cref="double"/> overload (a conversion to
        ///     <see cref="long"/> is the better one), which is what makes <c>np.nanquantile(a, 1)</c> NumPy's call. The
        ///     Python int is WEAK for the result dtype (NumPy's <c>weak_q</c>): an interpolating method keeps float16 /
        ///     float32 data's dtype.
        /// </remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, long q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return nanquantile(a, qArray, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile ignoring NaNs over SEVERAL axes for a Python INT q (see the <see cref="long"/> overload with
        ///     one axis for NumPy's rule): the integer index is taken, whatever the data.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">The quantile: 0 or 1.</param>
        /// <param name="axis">The axes to reduce; null reduces the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantile of the non-NaN values over the reduced axes.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex, or text data (NumPy's NaN test).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1], an unknown method, or an axis
        ///     repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/> (its NaN-like NAs removed, the rest
        ///     taking <c>np.quantile</c>'s rules; an all-NaN slice gives the NA object): a line of two or more elements
        ///     holding a None-like NA (<c>Cannot compare null that is not a nan-like value</c>), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>could not convert string to float: 'b'</c>).</exception>
        /// <exception cref="NotSupportedException">A NEP 55 <c>StringDType</c> reduction along an axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError">With <paramref name="weights"/> of the reduced axes only, over data with a
        ///     kept axis longer than 1: NumPy 2.4.2's own error (<c>index 1 is out of bounds for axis 0 with size 1</c>),
        ///     reproduced.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>), q's rank and the method.</exception>
        public static NDArray nanquantile(NDArray a, long q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return nanquantile(a, qArray, axis, @out, overwrite_input, method, keepdims, weights);
        }
    }
}
