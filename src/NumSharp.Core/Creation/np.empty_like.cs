using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        public static NDArray empty_like(NDArray prototype, Type dtype = null, Shape shape = default, string device = null)
            => empty_like(prototype, dtype, shape, 'K', device);

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="dtype">Overrides the data type of the result.</param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves prototype layout).</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        public static NDArray empty_like(NDArray prototype, Type dtype, Shape shape, char order, string device = null)
        {
            ValidateDevice(device);
            char physical = OrderResolver.Resolve(order, prototype.Shape);
            var dims = shape.IsEmpty ? (long[])prototype.shape.Clone() : (long[])shape;
            var resolvedShape = new Shape(dims, physical);
            return new NDArray(dtype ?? prototype.dtype, resolvedShape, false);
        }

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="typeCode">Overrides the data type of the result.</param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        public static NDArray empty_like(NDArray prototype, NPTypeCode typeCode, Shape shape = default)
            => empty_like(prototype, typeCode, shape, 'K');

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.
        /// </summary>
        /// <param name="prototype">The shape and data-type of prototype define these same attributes of the returned array.</param>
        /// <param name="typeCode">Overrides the data type of the result.</param>
        /// <param name="shape">Overrides the shape of the result.</param>
        /// <param name="order">Memory layout: 'C', 'F', 'A' or 'K' (default, preserves prototype layout).</param>
        /// <returns>Array of uninitialized (arbitrary) data with the same shape and type as prototype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html</remarks>
        public static NDArray empty_like(NDArray prototype, NPTypeCode typeCode, Shape shape, char order)
        {
            char physical = OrderResolver.Resolve(order, prototype.Shape);
            var dims = shape.IsEmpty ? (long[])prototype.shape.Clone() : (long[])shape;
            var resolvedShape = new Shape(dims, physical);
            return empty(resolvedShape, typeCode);
        }
    }
}
