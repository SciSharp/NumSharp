using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th percentile of the data along the specified axis, ignoring NaNs.
        ///     <paramref name="q"/> must be in [0, 100]. Equivalent to <c>np.nanquantile(a, q/100)</c>.
        ///     A slice that is entirely NaN (or empty) yields NaN.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The percentile, in [0, 100].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The NaN-ignoring percentile; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>).</exception>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html</remarks>
        public static NDArray nanpercentile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidatePercentile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType: np.isnan finds its NaN-like NAs, and NumPy computes the percentile of the rest.
                return Backends.Strings.NDVStringStatistics.NanQuantile("nanpercentile", a, new[] { q / 100.0 }, System.Array.Empty<long>(), false, axisArr,
                    @out, method, keepdims);
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axisArr, true);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     Compute the q-th percentiles of the data along the specified axis, ignoring NaNs.
        ///     Each value in <paramref name="q"/> must be in [0, 100]. The result's first axis is q.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The percentiles, each in [0, 100].</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The NaN-ignoring percentiles, q's axis first; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>).</exception>
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
        public static NDArray nanpercentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType: np.isnan finds its NaN-like NAs, and NumPy computes the percentile of the rest.
                return Backends.Strings.NDVStringStatistics.NanQuantile("nanpercentile", a, qFracs, new long[] { qFracs.Length }, false, axisArr, @out, method,
                    keepdims);
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axisArr, false);
            return QuantileEngine.Compute(a, qFracs, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     The q-th percentile over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the reduced axes
        ///     are merged into one before the partition, a slice that holds only NaNs (or nothing) gives NaN, and an EMPTY
        ///     axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The percentile, in [0, 100].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The NaN-ignoring percentile over the reduced axes; <paramref name="out"/> itself when one is
        ///     given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 100] ("Percentiles must be in the range
        ///     [0, 100]"), or an axis repeats.</exception>
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
        ///     own Python code (<c>np.nanpercentile(m, 50, axis=(0, 1))</c> returns the input's values).</remarks>
        public static NDArray nanpercentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidatePercentile(q);
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType: np.isnan finds its NaN-like NAs, and NumPy computes the percentile of the rest.
                return Backends.Strings.NDVStringStatistics.NanQuantile("nanpercentile", a, new[] { q / 100.0 }, System.Array.Empty<long>(), false, axis,
                    @out, method, keepdims);
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axis, true);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     The q-th percentiles over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the result's
        ///     first axis is q, a slice that holds only NaNs (or nothing) gives NaN, and an EMPTY axis list reduces
        ///     nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text is refused as NumPy refuses it, a NEP 55
        ///     <c>StringDType</c> array computes: see the exceptions).</param>
        /// <param name="q">The percentiles, each in [0, 100].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>One NaN-ignoring percentile per element of <paramref name="q"/> along the first axis, over the reduced
        ///     axes; <paramref name="out"/> itself when one is given.</returns>
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
        public static NDArray nanpercentile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType: np.isnan finds its NaN-like NAs, and NumPy computes the percentile of the rest.
                return Backends.Strings.NDVStringStatistics.NanQuantile("nanpercentile", a, qFracs, new long[] { qFracs.Length }, false, axis, @out, method,
                    keepdims);
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axis, false);
            return QuantileEngine.Compute(a, qFracs, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     The q-th percentiles (q in [0, 100]) of the data ignoring NaNs, q given as a 0-d or 1-d array — NumPy's
        ///     <c>np.nanpercentile(a, q_array)</c>.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of percentiles in [0, 100].</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The percentiles of the non-NaN values: one per element of q (a scalar result for a 0-d q), with the axis reduced.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/>: NumPy divides it by 100 first, and <c>divide</c> has no
        ///     text loop. Text data: NumPy's NaN test (<c>isnan</c> has no text loop), or the add.reduce refusal of
        ///     <c>np.nanmean</c> for empty text data.</exception>
        /// <exception cref="ArgumentException">A q of rank above 1, or a percentile outside [0, 100] (NumPy's ValueError
        ///     texts).</exception>
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
        public static NDArray nanpercentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check), before the
            // data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsVString(q))
                throw Backends.Strings.NDVStringStatistics.QuantileVStringQ("nanpercentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("nanpercentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType data: q divided by 100 and checked (any rank), then the data; q's rank is checked per line.
                return Backends.Strings.NDVStringStatistics.NanQuantile("nanpercentile", a, PercentileFractions(q), (long[])q.Shape.dimensions.Clone(), false,
                    axis.HasValue ? new[] { axis.Value } : null, @out, method, keepdims);
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr;
            if (q.ndim == 0)
            {
                double v = QuantileValueAt(q, 0);
                ValidatePercentile(v);
                qArr = new[] { v / 100.0 };
            }
            else
            {
                qArr = new double[q.size];
                for (long i = 0; i < q.size; i++)
                {
                    double v = QuantileValueAt(q, i);
                    ValidatePercentile(v);
                    qArr[i] = v / 100.0;
                }
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axisArr, qIsScalar);
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true);
        }
    }
}
