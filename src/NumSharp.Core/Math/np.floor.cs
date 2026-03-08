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
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The floor of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.floor.html</remarks>
        public static NDArray floor(in NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.Floor(x, dtype);

        /// <summary>
        ///     Return the floor of the input, element-wise. <br></br>
        ///     The floor of the scalar x is the largest integer i, such that i <= x. It is often denoted as \lfloor x \rfloor.
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The floor of each element in x. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.floor.html</remarks>
        public static NDArray floor(in NDArray x, Type dtype) 
            => x.TensorEngine.Floor(x, dtype);
    }
}
