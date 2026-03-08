using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Returns an element-wise indication of the sign of a number. <br></br>
        ///     The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sign of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sign.html</remarks>
        public static NDArray sign(in NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.Sign(x, dtype);

        /// <summary>
        ///     Returns an element-wise indication of the sign of a number. <br></br>
        ///     The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
        /// </summary>
        /// <param name="x">Angle, in radians (2 \pi rad equals 360 degrees).</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The sign of x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sign.html</remarks>
        public static NDArray sign(in NDArray x, Type dtype) 
            => x.TensorEngine.Sign(x, dtype);
    }
}
