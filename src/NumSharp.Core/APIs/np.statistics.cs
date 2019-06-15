using NumSharp.Backends;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static T amax<T>(NDArray nd)
            => nd.amax<T>();

        public static NDArray amin(NDArray nd, int? axis = null)
            => nd.amin(axis);

        public static NDArray amax(NDArray nd, int axis)
            => nd.amax(axis);

        public static NDArray max(NDArray nd, int axis)
            => nd.max(axis);

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.
        ///     Returns the average of the array elements.
        ///     The average is taken over the flattened array by default, otherwise over the specified axis.
        ///     float64 intermediate and return values are used for integer inputs.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="axis">Axis or axes along which the means are computed. The default is to compute the mean of the flattened array.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.1/reference/generated/numpy.mean.html</remarks>
        public static NDArray mean(NDArray nd, int axis = -1)
            => BackendFactory.GetEngine().Mean(nd, axis);
    }
}
