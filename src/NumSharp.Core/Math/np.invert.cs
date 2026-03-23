using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise.
        /// Computes the bit-wise NOT of the underlying binary representation of
        /// the integers in the input arrays.For signed integer inputs, the
        /// two's complement is returned. In a two's-complement system negative
        /// numbers are represented by the two's complement of the absolute value.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        public static NDArray invert(NDArray x, NPTypeCode? outType = null)
            => x.TensorEngine.Invert(x, outType);

        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        public static NDArray invert(NDArray x, Type outType)
            => x.TensorEngine.Invert(x, outType);

        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise. Alias for invert.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        public static NDArray bitwise_not(NDArray x, NPTypeCode? outType = null)
            => invert(x, outType);

        /// <summary>
        /// Compute bit-wise inversion, or bit-wise NOT, element-wise. Alias for invert.
        /// </summary>
        /// <param name="x">Only integer and boolean types are handled.</param>
        /// <param name="outType">The dtype the returned ndarray should be of.</param>
        /// <returns>Result. This is a scalar if x is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.invert.html</remarks>
        public static NDArray bitwise_not(NDArray x, Type outType)
            => invert(x, outType);
    }
}
