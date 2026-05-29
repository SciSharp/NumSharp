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
            ValidatePercentile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, qFracs, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            ValidatePercentile(q);
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
            return QuantileEngine.Compute(a, qFracs, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false,
                emptyReturnsNaN: true, ignoreNaN: true);
        }

        public static NDArray nanpercentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            if (q is null) throw new ArgumentNullException(nameof(q));
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
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true);
        }
    }
}
