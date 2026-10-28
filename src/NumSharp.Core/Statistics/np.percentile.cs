using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th percentile of the data along the specified axis.
        ///     <paramref name="q"/> must be in [0, 100]. Equivalent to <c>np.quantile(a, q/100)</c>.
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentile, in [0, 100].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The percentile; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>).</exception>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.percentile.html</remarks>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        public static NDArray percentile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: with weights= NumPy's result is empty_like of the chararray, the plain call
            // kept as a chararray (its checks run inside it, so an invalid argument raises the same error); without
            // weights the flag goes down, where NumPy's text route meets the STRING partition (DefaultEngine.QuantileFunction).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body (q divided by 100 and checked, the weighted branch, the text routes, the numeric
            // quantile) is TensorEngine.Percentile's: the facade holds only the class rules above.
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     Compute the q-th percentiles of the data along the specified axis.
        ///     Each value in <paramref name="q"/> must be in [0, 100]. The result's first axis is q.
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentiles, each in [0, 100].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The percentiles, q's axis first; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>).</exception>
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
        public static NDArray percentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body is TensorEngine.Percentile's (a null q is reported there, after the complex check).
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th percentile over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into
        ///     one before the partition, so the result keeps only the other axes (0-d when every axis is reduced), and an
        ///     EMPTY axis list reduces nothing (each element is its own percentile).
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentile, in [0, 100].</param>
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
        /// <returns>The percentile over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 100] ("Percentiles must be in the range
        ///     [0, 100]"), or an axis repeats.</exception>
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
        public static NDArray percentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body is TensorEngine.Percentile's.
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th percentiles over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the result's first axis is q,
        ///     the reduced axes are merged into one before the partition, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's take
        ///     path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentiles, each in [0, 100].</param>
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
        /// <returns>One percentile per element of <paramref name="q"/> along the first axis, over the reduced axes;
        ///     <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="q"/>, <paramref name="a"/> or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="ValueError">A percentile is outside [0, 100] ("Percentiles must be in the range [0, 100]"),
        ///     or an axis repeats.</exception>
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
        ///     own Python code (with a list of percentiles NumPy even raises a matrix-product error).</remarks>
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
        public static NDArray percentile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body is TensorEngine.Percentile's (a null q is reported there, after the complex check).
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th percentiles (q in [0, 100]) of the data, q given as a 0-d or 1-d array — NumPy's
        ///     <c>np.percentile(a, q_array)</c>.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of percentiles in [0, 100] (up to 2-d over text data).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The percentiles: one per element of q (a scalar result for a 0-d q), with the axis reduced.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/>: NumPy divides it by 100 first, and <c>divide</c> has no
        ///     text loop (<c>ufunc 'divide' not supported for the input types, …</c>).</exception>
        /// <exception cref="ArgumentException">A numeric percentile's q of rank above 1 (NumPy's ValueError text).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100]; over text data a q of rank above 2 (<c>q must be a
        ///     scalar or 1d</c>, after the axis) or an empty q of two dimensions (its <c>q.min()</c> range
        ///     check).</exception>
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
        public static NDArray percentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body is TensorEngine.Percentile's: a null or text q, np.true_divide(q, 100) and the
            // range check at any rank (a percentile never takes the integer index), a weak C# literal, q's rank, all in
            // NumPy's order.
            int[] axes = axis.HasValue ? new[] { axis.Value } : null;
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axes, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axes, @out, overwrite_input, method, keepdims, weights);
        }

        /// <summary>
        ///     The q-th percentiles over SEVERAL axes at once, q given as an array — NumPy's
        ///     <c>np.percentile(a, q_array, axis=(i, j))</c>: the reduced axes are merged into one before the partition, the
        ///     result's leading axes are q's, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of percentiles in [0, 100] (up to 2-d over text data).</param>
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
        /// <returns>The percentiles over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex ("a must be an array of real numbers", NumPy's first
        ///     statement); a text <paramref name="q"/> (NumPy divides it by 100 first, and <c>divide</c> has no text loop);
        ///     over text data an interpolating method, an <paramref name="out"/> that does not cast safely to the data, or
        ///     keepdims over a 0-d result.</exception>
        /// <exception cref="ArgumentException">A numeric percentile's q of rank above 1, an unknown method, or an
        ///     <paramref name="out"/> of another shape (numeric data; NumPy's ValueError texts).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100]; an axis repeats; over text data a q of rank above 2,
        ///     an empty q of two dimensions, or an <paramref name="out"/> of another shape.</exception>
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
        ///     tuple-axis overload.</remarks>
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
        public static NDArray percentile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // a numpy.char.chararray operand: the weights= rule here, the flag below (see the double-q overload).
            if (chararray && weights is not null)
                return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                    static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
            // NumPy's np.percentile body is TensorEngine.Percentile's (see the one-axis overload).
            return chararray
                ? DefaultEngine.QuantileFunction("percentile", a, new QuantileArgument(q), axis, @out,
                    overwrite_input, method, keepdims, weights, chararray: true)
                : QuantileFamilyEngine(a).Percentile(a, q, axis, @out, overwrite_input, method, keepdims, weights);
        }
    }
}
