using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the ceiling of the input, element-wise.<br></br>
        ///     The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as \lceil x \rceil.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The ceiling of each element in x, with float dtype. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ceil.html</remarks>
        public static NDArray ceil(NDArray x, NPTypeCode? dtype = null)
            => x.TensorEngine.Ceil(x, dtype);

        /// <summary>
        ///     Return the ceiling of the input, element-wise.<br></br>
        ///     The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as \lceil x \rceil.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="dtype">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The ceiling of each element in x, with float dtype. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ceil.html</remarks>
        public static NDArray ceil(NDArray x, Type dtype) 
            => x.TensorEngine.Ceil(x, dtype);
    }
}
