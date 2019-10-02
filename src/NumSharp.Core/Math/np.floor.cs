using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the floor of the input, element-wise. <br></br>
        ///     The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The floor of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.floor.html</remarks>
        public static NDArray floor(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Floor(x, outType);

        /// <summary>
        ///     Return the floor of the input, element-wise. <br></br>
        ///     The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The floor of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.floor.html</remarks>
        public static NDArray floor(in NDArray x, Type outType) 
            => x.TensorEngine.Floor(x, outType);
    }
}
