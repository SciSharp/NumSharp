using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis.
        ///     <paramref name="q"/> must be in the range [0, 1].
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.quantile.html</remarks>
        public static NDArray quantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidateQuantile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, new[] { q }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        /// <summary>
        ///     Compute the q-th quantiles of the data along the specified axis.
        ///     Each value in <paramref name="q"/> must be in [0, 1]. Result's first axis is q.
        /// </summary>
        public static NDArray quantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, q, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     Compute the q-th quantile, reducing along multiple axes.
        /// </summary>
        public static NDArray quantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidateQuantile(q);
            return QuantileEngine.Compute(a, new[] { q }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        public static NDArray quantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            return QuantileEngine.Compute(a, q, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     NDArray-q overload — accepts a 0-D or 1-D NDArray of quantile values.
        ///     Higher-rank q is rejected (NumPy raises "q must be a scalar or 1d").
        /// </summary>
        public static NDArray quantile(NDArray a, NDArray q,
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
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar);
        }

        private static void ValidateQuantile(double q)
        {
            if (!(q >= 0.0 && q <= 1.0))
                throw new ArgumentException("Quantiles must be in the range [0, 1]");
        }

        private static double[] QArrayFromNDArray(NDArray q)
        {
            if (q.ndim == 0)
                return new[] { Convert.ToDouble(q.GetAtIndex(0)) };
            double[] outQ = new double[q.size];
            for (long i = 0; i < q.size; i++) outQ[i] = Convert.ToDouble(q.GetAtIndex(i));
            return outQ;
        }
    }
}
