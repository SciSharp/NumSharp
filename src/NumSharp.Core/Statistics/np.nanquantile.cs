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
        public static NDArray nanquantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidateQuantile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, new[] { q }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanquantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, q, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanquantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidateQuantile(q);
            return QuantileEngine.Compute(a, new[] { q }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanquantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            return QuantileEngine.Compute(a, q, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        /// <summary>
        ///     NDArray-q overload — accepts a 0-D or 1-D NDArray of quantile values.
        ///     Higher-rank q is rejected (NumPy raises "q must be a scalar or 1d").
        /// </summary>
        public static NDArray nanquantile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr = QArrayFromNDArray(q);
            for (int i = 0; i < qArr.Length; i++) ValidateQuantile(qArr[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true);
        }
    }
}
