using System;
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
        public static NDArray percentile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            // `q = np.true_divide(q, 100, out=...)` (float64 for a Python number), then `_quantile_is_valid` on the
            // QUOTIENT: -5e-324 divides to -0.0 and passes.
            double fraction = PercentileFraction(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText;
            // a StringDType array reads the two neighbours before _lerp refuses them).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                using var qArray = NDArray.Scalar(fraction);
                return QuantileWeighted.Compute(a, qArray, axisArr, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText("percentile", a, new[] { fraction }, System.Array.Empty<long>(), false, axisArr, @out, method, keepdims, chararray);
            return QuantileEngine.Compute(a, new[] { fraction }, axisArr, @out, overwrite_input,
                default, keepdims, qIsScalar: true, methodName: method);
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
        public static NDArray percentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                qFracs[i] = PercentileFraction(q[i]);
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText;
            // a StringDType array reads the two neighbours before _lerp refuses them).
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                using var qArray = np.array(qFracs);
                return QuantileWeighted.Compute(a, qArray, axisArr, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText("percentile", a, qFracs, new long[] { qFracs.Length }, false, axisArr, @out, method, keepdims, chararray);
            return QuantileEngine.Compute(a, qFracs, axisArr, @out, overwrite_input,
                default, keepdims, qIsScalar: false, methodName: method);
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
        public static NDArray percentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            // `q = np.true_divide(q, 100, out=...)` (float64 for a Python number), then `_quantile_is_valid` on the
            // QUOTIENT: -5e-324 divides to -0.0 and passes.
            double fraction = PercentileFraction(q);
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText;
            // a StringDType array reads the two neighbours before _lerp refuses them).
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                using var qArray = NDArray.Scalar(fraction);
                return QuantileWeighted.Compute(a, qArray, axis, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText("percentile", a, new[] { fraction }, System.Array.Empty<long>(), false, axis, @out, method, keepdims, chararray);
            return QuantileEngine.Compute(a, new[] { fraction }, axis, @out, overwrite_input,
                default, keepdims, qIsScalar: true, methodName: method);
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
        public static NDArray percentile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                qFracs[i] = PercentileFraction(q[i]);
            }
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText;
            // a StringDType array reads the two neighbours before _lerp refuses them).
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                using var qArray = np.array(qFracs);
                return QuantileWeighted.Compute(a, qArray, axis, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText("percentile", a, qFracs, new long[] { qFracs.Length }, false, axis, @out, method, keepdims, chararray);
            return QuantileEngine.Compute(a, qFracs, axis, @out, overwrite_input,
                default, keepdims, qIsScalar: false, methodName: method);
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
        public static NDArray percentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims, weights), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check), before the
            // data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsVString(q))
                throw Backends.Strings.NDVStringStatistics.QuantileVStringQ("percentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("percentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            // Text data: NumPy accepts a q of any rank up to 2 (_quantile_ureduce_func refuses rank 3 and above, after the
            // range and axis checks), so the text route runs before the numeric path's rank check.
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                // `q = np.true_divide(q, 100)` in q's own dtype (a float32 q stays float32), after the range check.
                PercentileFractions(q);
                using var hundred = NDArray.PythonLiteral(100L);
                using var qArray = np.true_divide(q, hundred);
                return QuantileWeighted.Compute(a, qArray, axis.HasValue ? new[] { axis.Value } : null, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                // np.true_divide(q, 100) is float64 whatever q held: percentile never takes the integer index.
                return QuantileOfText("percentile", a, PercentileFractions(q), (long[])q.Shape.dimensions.Clone(), false,
                    axis.HasValue ? new[] { axis.Value } : null, @out, method, keepdims, chararray, QuantileIndexes.PrecisionOf(q));
            bool qIsScalar = q.ndim == 0;
            // `np.true_divide(q, 100)` and `_quantile_is_valid` at any rank; q's rank is `_quantile_ureduce_func`'s check,
            // after the axes, inside the engine (a 2-d q keeps both axes).
            double[] qArr = PercentileFractions(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // weak_q = type(q) in (int, float): only a C# number (its 0-d array marked WasPythonLiteral) keeps float16 /
            // float32 data's dtype under interpolation; an array q, 0-d included, is a strong float64.
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                default, keepdims, qIsScalar, weakQ: q.WasPythonLiteral, methodName: method,
                qShape: qIsScalar ? null : (long[])q.Shape.dimensions.Clone(), qType: QuantileIndexes.PrecisionOf(q));
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
        public static NDArray percentile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            bool chararray = NDCharArrayBoundary.Sees(a);   // the user's call decides the chararray rule (before any PlainFor region hides it)
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100), before the data is looked at.
            if (Backends.Strings.NDStringUfuncs.IsVString(q))
                throw Backends.Strings.NDVStringStatistics.QuantileVStringQ("percentile", a, q, method, axis);
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("percentile", a, q, method, axis);
            double[] fractions = PercentileFractions(q);
            // Text data: q's rank is NumPy's check after the axes (rank 3 and above refused), so it runs in the text route;
            // np.true_divide(q, 100) is float64 whatever q held, so percentile never takes the integer index.
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
            // branch for every dtype (QuantileWeighted); a chararray operand keeps its class (NumPy's result is
            // empty_like of the chararray).
            if (weights is not null)
            {
                if (chararray)
                    return NDCharArrayBoundary.Kept((a, q, axis, @out, overwrite_input, method, keepdims, weights),
                        static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims, s.weights), @out);
                // `q = np.true_divide(q, 100)` in q's own dtype (a float32 q stays float32), after the range check.
                PercentileFractions(q);
                using var hundred = NDArray.PythonLiteral(100L);
                using var qArray = np.true_divide(q, hundred);
                return QuantileWeighted.Compute(a, qArray, axis, @out, keepdims, method, weights);
            }
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText("percentile", a, fractions, (long[])q.Shape.dimensions.Clone(), false, axis, @out,
                    method, keepdims, chararray, QuantileIndexes.PrecisionOf(q));
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64. q's rank
            // is checked after the axes, inside the engine (a 2-d q keeps both axes).
            return QuantileEngine.Compute(a, fractions, axis, @out, overwrite_input,
                default, keepdims, qIsScalar: q.ndim == 0, weakQ: q.WasPythonLiteral, methodName: method,
                qShape: q.ndim == 0 ? null : (long[])q.Shape.dimensions.Clone(), qType: QuantileIndexes.PrecisionOf(q));
        }

        /// <summary>
        ///     NumPy's <c>q = np.true_divide(q, 100, out=...)</c> followed by <c>_quantile_is_valid(q)</c>, over a q array: the
        ///     quotients in C order, computed in q's own dtype (a float16 / float32 q divides in float16 / float32 and its
        ///     quotient is what NumPy's index arithmetic reads; every other q divides in float64), each checked to lie in
        ///     [0, 1], and an empty q of rank other than 1 refused by the <c>q.min()</c> that NumPy's check takes for it.
        /// </summary>
        /// <param name="q">The percentiles (any numeric dtype).</param>
        /// <returns>The quantiles <c>q / 100</c> in C order (values of q's precision, see
        ///     <see cref="QuantileIndexes.PrecisionOf"/>).</returns>
        /// <exception cref="ValueError">A quotient outside [0, 1] (or NaN): <c>Percentiles must be in the range [0,
        ///     100]</c> — a float16 <c>100.0625</c> divides to <c>1.0009765625</c> and is refused, while a negative value that
        ///     divides to <c>-0.0</c> passes; an empty q of rank other than 1: <c>zero-size array to reduction operation
        ///     minimum which has no identity</c>.</exception>
        private static double[] PercentileFractions(NDArray q)
        {
            var fractions = new double[q.size];
            if (fractions.Length == 0 && q.ndim != 1)
                throw new ValueError("zero-size array to reduction operation minimum which has no identity");
            NPTypeCode precision = QuantileIndexes.PrecisionOf(q);
            for (long i = 0; i < fractions.Length; i++)
            {
                // One true_divide in q's dtype: the double quotient rounded once to float16 / float32 is NumPy's (a float16
                // or float32 division computed in double and rounded is correctly rounded).
                double fraction = QuantileIndexes.Round(precision, QuantileValueAt(q, i) / 100.0);
                ValidatePercentileFraction(fraction);
                fractions[i] = fraction;
            }
            return fractions;
        }

        /// <summary>
        ///     NumPy's <c>q = np.true_divide(q, 100, out=...)</c> and <c>_quantile_is_valid(q)</c> for one float64 percentile (a
        ///     Python number, or an element of a list): the quotient, checked to lie in [0, 1].
        /// </summary>
        /// <param name="q">The percentile.</param>
        /// <returns><c>q / 100</c>.</returns>
        /// <exception cref="ValueError">The quotient is outside [0, 1] (or NaN): "Percentiles must be in the range [0,
        ///     100]". The check reads the quotient, so a percentile of <c>-5e-324</c> (whose quotient is <c>-0.0</c>)
        ///     passes, as in NumPy.</exception>
        private static double PercentileFraction(double q)
        {
            double fraction = q / 100.0;
            ValidatePercentileFraction(fraction);
            return fraction;
        }

        /// <summary>NumPy's <c>_quantile_is_valid</c> of one quotient <c>q / 100</c>: inside <c>[0, 1]</c> (NaN is not).</summary>
        /// <param name="fraction">The quotient.</param>
        /// <exception cref="ValueError"><paramref name="fraction"/> is outside [0, 1] ("Percentiles must be in the range [0,
        ///     100]").</exception>
        private static void ValidatePercentileFraction(double fraction)
        {
            if (!(fraction >= 0.0 && fraction <= 1.0))
                throw new ValueError("Percentiles must be in the range [0, 100]");
        }
    }
}
