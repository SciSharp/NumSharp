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
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The ceiling of each element in x, with float dtype. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ceil.html</remarks>
        public static NDArray ceil(in NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Ceil(x, outType);

        /// <summary>
        ///     Return the ceiling of the input, element-wise.<br></br>
        ///     The ceil of the scalar x is the smallest integer i, such that i >= x. It is often denoted as \lceil x \rceil.
        /// </summary>
        /// <param name="x">Input data.</param>
        /// <param name="outType">The dtype the returned ndarray should be of, only non integer values are supported.</param>
        /// <returns>The ceiling of each element in x, with float dtype. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ceil.html</remarks>
        public static NDArray ceil(in NDArray x, Type outType) 
            => x.TensorEngine.Ceil(x, outType);
    }
}
