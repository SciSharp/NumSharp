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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html</remarks>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray nanquantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidateQuantile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanquantile", a, method, axisArr, true);
            return QuantileEngine.Compute(a, new[] { q }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray nanquantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanquantile", a, method, axisArr, false);
            return QuantileEngine.Compute(a, q, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray nanquantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidateQuantile(q);
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanquantile", a, method, axis, true);
            return QuantileEngine.Compute(a, new[] { q }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray nanquantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanquantile", a, method, axis, false);
            return QuantileEngine.Compute(a, q, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     NDArray-q overload — accepts a 0-D or 1-D NDArray of quantile values.
        ///     Higher-rank q is rejected (NumPy raises "q must be a scalar or 1d").
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of quantiles in [0, 1].</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The quantiles of the non-NaN values: one per element of q (a scalar result for a 0-d q), with the axis reduced; for empty numeric data and an empty text q, NumPy's <c>np.nanmean(a, axis)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/> (NumPy's [0, 1] check: Python's <c>'&lt;=' not supported
        ///     between instances of 'float' and 'numpy.str_'</c> for a 1-d q of fewer than 10 elements, <c>q.min()</c>'s
        ///     <c>minimum</c> refusal otherwise). Text data: NumPy's NaN test (<c>isnan</c> has no text loop), or the
        ///     add.reduce refusal of <c>np.nanmean</c> for empty text data.</exception>
        /// <exception cref="ArgumentException">A q of rank above 1, or a quantile outside [0, 1] (NumPy's ValueError
        ///     texts).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray nanquantile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanquantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its [0, 1] check on q, before the data or q's rank is looked at; an empty q
            // passes it, and with empty numeric data NumPy then returns np.nanmean(a, axis) (the helper's null).
            if (Backends.Strings.NDStringUfuncs.IsString(q))
            {
                var textQ = Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("nanquantile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
                if (textQ is not null)
                    throw textQ;
                return np.nanmean(a, axis, keepdims: keepdims);
            }
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr = QArrayFromNDArray(q);
            for (int i = 0; i < qArr.Length; i++) ValidateQuantile(qArr[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanquantile", a, method, axisArr, qIsScalar);
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true);
        }
    }
}
