using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shapes">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(int shape)
        {
            return empty(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(int[] shape)
        {
            return empty(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(long[] shape)
        {
            return empty(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty<T>(int[] shape)
        {
            return empty(new Shape(shape), typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty<T>(long[] shape)
        {
            return empty(new Shape(shape), typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="dtype">
        ///     Desired output dtype — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Default (null) is numpy.float64.
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape, DType dtype, string device = null)
        {
            ValidateDevice(device);
            return new NDArray(dtype ?? DType.Double, shape, false);
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="dtype">Desired output data-type for the array, e.g, numpy.int8. Default is numpy.float64.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape)
        {
            return new NDArray(NPTypeCode.Double, shape, false);
        }

        /// <summary>
        ///     Return a new array of given shape and type with a specified memory layout.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="order">Memory layout: 'C' (row-major), 'F' (column-major), 'A' (any), 'K' (keep).
        /// With no source array, 'A' and 'K' default to 'C'.</param>
        /// <param name="dtype">Desired output data-type. Default is numpy.float64.</param>
        /// <returns>Array of uninitialized data with the requested memory layout.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape, char order, DType dtype = null)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = new Shape(shape.dimensions, physical);
            return new NDArray(dtype ?? DType.Double, orderedShape, false);
        }
    }
}
