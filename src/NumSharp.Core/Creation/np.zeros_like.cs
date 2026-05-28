using System;

namespace NumSharp
{
    public static partial class np
    {
        ///  <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <returns>Array of zeros with the same shape and type as `nd`.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html</remarks>
        public static NDArray zeros_like(NDArray a, Type dtype = null) => zeros_like(a, dtype, 'K');

        ///  <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        ///  </summary>
        /// <param name="a">The shape and data-type of a define these same attributes of the returned array.</param>
        ///  <param name="dtype">Overrides the data type of the result.</param>
        ///  <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves source layout).</param>
        ///  <returns>Array of zeros with the same shape and type as `nd`.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html</remarks>
        public static NDArray zeros_like(NDArray a, Type dtype, char order)
        {
            char physical = OrderResolver.Resolve(order, a.Shape);
            var resolvedShape = new Shape((long[])a.shape.Clone(), physical);
            return np.zeros(resolvedShape, dtype ?? a.dtype);
        }
    }
}
