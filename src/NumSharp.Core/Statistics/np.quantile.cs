using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis.
        ///     <paramref name="q"/> must be in the range [0, 1].
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
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
        /// <returns>The quantile; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data with an interpolating method (NumPy subtracts the text), an
        ///     <paramref name="out"/> that does not cast safely to the data, or keepdims over a 0-d result.</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take
        ///     from an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.quantile.html</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: with weights= NumPy's result is empty_like of the chararray, the plain call
            // kept as a chararray (its checks run inside it, so an invalid argument raises the same error); without
            // weights the flag goes down, where NumPy's text route meets the STRING partition (DefaultEngine.QuantileFunction).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body (q's checks, the weighted branch, the text routes, the numeric quantile) is
            // TensorEngine.Quantile's: the facade holds only the class rules above.
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     Compute the q-th quantiles of the data along the specified axis.
        ///     Each value in <paramref name="q"/> must be in [0, 1]. Result's first axis is q.
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
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
        /// <returns>The quantiles, q's axis first; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, <paramref name="q"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data with an interpolating method (NumPy subtracts the text), an
        ///     <paramref name="out"/> that does not cast safely to the data, or keepdims over a 0-d result.</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take
        ///     from an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body is TensorEngine.Quantile's (a null q is reported there, after the complex check).
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into
        ///     one before the partition, so the result keeps only the other axes (0-d when every axis is reduced), and an
        ///     EMPTY axis list reduces nothing (each element is its own quantile).
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
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
        /// <returns>The quantile over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]"),
        ///     or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="IndexOutOfRangeException">The reduced axes hold no element (NumPy's IndexError text).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (the median / percentile / quantile family's gap class).</remarks>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body is TensorEngine.Quantile's.
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantiles over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the result's first axis is q, the
        ///     reduced axes are merged into one before the partition, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
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
        /// <returns>One quantile per element of <paramref name="q"/> along the first axis, over the reduced axes;
        ///     <paramref name="out"/> itself when one is given.</returns>
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
        /// <exception cref="IndexOutOfRangeException">The reduced axes hold no element (NumPy's IndexError text).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (with a list of quantiles NumPy even raises a matrix-product error).</remarks>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body is TensorEngine.Quantile's (a null q is reported there, after the complex check).
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     NDArray-q overload — accepts a 0-D or 1-D NDArray of quantile values (and, over text data, NumPy's 2-D q).
        ///     A numeric quantile of higher rank is rejected (NumPy raises "q must be a scalar or 1d" from rank 3).
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of quantiles in [0, 1] (up to 2-d over text data). An integer or bool q makes
        ///     NumPy's 'linear' virtual index <c>(n - 1) * q</c> an integer, which it TAKES instead of interpolating: the
        ///     'lower' answer, in the data's own dtype.</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantiles: one per element of q (a scalar result for a 0-d q), with the axis reduced.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/> (NumPy's [0, 1] check: Python's <c>'&lt;=' not supported
        ///     between instances of 'float' and 'numpy.str_'</c> for a 1-d q of fewer than 10 elements, <c>q.min()</c>'s
        ///     <c>minimum</c> refusal otherwise; an empty one reaches the string repeat's refusal).</exception>
        /// <exception cref="ArgumentException">A numeric quantile's q of rank above 1, or a quantile outside [0, 1] (NumPy's
        ///     ValueError texts).</exception>
        /// <exception cref="ValueError">Over text data: a q of rank above 2 (<c>q must be a scalar or 1d</c>, after the axis),
        ///     or an empty q of two dimensions (its <c>q.min()</c> range check).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body is TensorEngine.Quantile's: a null or text q, the range check at any rank, an
            // integer q's index taken instead of interpolated, a weak C# literal, q's rank, all in NumPy's order.
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     NDArray-q overload over SEVERAL axes at once — NumPy's <c>np.quantile(a, q_array, axis=(i, j))</c>: the
        ///     reduced axes are merged into one before the partition, the result's leading axes are q's, and an EMPTY axis
        ///     list reduces nothing.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of quantiles in [0, 1] (up to 2-d over text data). An integer or bool q makes
        ///     NumPy's 'linear' virtual index <c>(n - 1) * q</c> an integer, which it TAKES instead of interpolating.</param>
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
        /// <returns>The quantiles over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex ("a must be an array of real numbers", NumPy's first
        ///     statement); a text <paramref name="q"/> (its [0, 1] check refuses); over text data an interpolating method
        ///     (NumPy subtracts the text), an <paramref name="out"/> that does not cast safely to the data, or keepdims over
        ///     a 0-d result.</exception>
        /// <exception cref="ArgumentException">A numeric quantile's q of rank above 1, a quantile outside [0, 1], an
        ///     unknown method, or an <paramref name="out"/> of another shape (numeric data; NumPy's ValueError
        ///     texts).</exception>
        /// <exception cref="ValueError">An axis repeats; over text data a q of rank above 2, an empty q of two
        ///     dimensions, or an <paramref name="out"/> of another shape.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take, or the interpolating methods'
        ///     neighbour lookup).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain, as for every
        ///     tuple-axis overload (see the <see cref="double"/>-q one).</remarks>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.quantile body is TensorEngine.Quantile's (see the one-axis overload).
            return chararray
                ? DefaultEngine.Of(a).QuantileFunction("quantile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : a.TensorEngine.Quantile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile for a Python INT q (a C# integer literal), NumPy 2.4's rule for it: <c>q =
        ///     np.asanyarray(q)</c> makes it an int64 array, whose 'linear' virtual index <c>(n - 1) * q</c> is an integer
        ///     that NumPy TAKES instead of interpolating — the data's own dtype and element, over float data too
        ///     (<c>np.quantile([1., inf], 1)</c> is inf), and over text the string itself.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">The quantile: 0 or 1 (anything else is outside [0, 1]).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names). Every selection method
        ///     takes the same element for q = 0 or 1; the other interpolating methods still interpolate.</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantile with the axis reduced (a scalar result over every axis).</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers".</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]").</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <remarks>
        ///     C# binds an integer literal here rather than to the <see cref="double"/> overload (a conversion to
        ///     <see cref="long"/> is the better one), which is what makes <c>np.quantile(a, 0)</c> NumPy's call: over int
        ///     data it returns an int, over text it computes (NumPy's take path) where a float q would subtract text.
        ///     NumPy before 2.4 converted a Python int q to the data's dtype over float data; 2.4 no longer does.
        /// </remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, long q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return quantile(a, qArray, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th quantile over SEVERAL axes for a Python INT q (see the <see cref="long"/> overload with one axis for
        ///     NumPy's rule): the integer index is taken, whatever the data.
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
        /// <returns>The quantile over the reduced axes.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex.</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1], or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before q's rank, the method and the data's refusals.</exception>
        public static NDArray quantile(NDArray a, long q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return quantile(a, qArray, axis, @out, overwrite_input, method, keepdims, weights);
        }
    }
}
