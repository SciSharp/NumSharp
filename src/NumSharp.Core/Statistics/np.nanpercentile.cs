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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html</remarks>
        public static NDArray nanpercentile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            ValidatePercentile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axisArr, true);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
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
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axisArr, false);
            return QuantileEngine.Compute(a, qFracs, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidatePercentile(q);
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer("nanpercentile", a, method, axis, true);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
            // text itself (NDStringUfuncs.QuantileAnswer).
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
        public static NDArray nanpercentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => nanpercentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check), before the
            // data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("nanpercentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr;
            if (q.ndim == 0)
            {
                double v = Convert.ToDouble(q.GetAtIndex(0));
                ValidatePercentile(v);
                qArr = new[] { v / 100.0 };
            }
            else
            {
                qArr = new double[q.size];
                for (long i = 0; i < q.size; i++)
                {
                    double v = Convert.ToDouble(q.GetAtIndex(i));
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
