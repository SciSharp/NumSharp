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
        /// <param name="dtype">Desired output data-type for the array, e.g, numpy.int8. Default is numpy.float64.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape, Type dtype)
        {
            return empty(shape, (dtype ?? typeof(double)).GetTypeCode());
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="typeCode">Desired output data-type for the array, e.g, numpy.int8. Default is numpy.float64.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            return new NDArray(typeCode, shape, false);
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
        public static NDArray empty(Shape shape, char order, Type dtype = null)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = new Shape(shape.dimensions, physical);
            return new NDArray((dtype ?? typeof(double)).GetTypeCode(), orderedShape, false);
        }
    }
}
