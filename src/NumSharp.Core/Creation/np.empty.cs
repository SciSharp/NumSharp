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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(params int[] shapes)
        {
            return empty(shapes, null);
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shapes">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty<T>(params int[] shapes)
        {
            return empty(shapes, typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.
        /// </summary>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="dtype">Desired output data-type for the array, e.g, numpy.int8. Default is numpy.float64.</param>
        /// <returns>Array of uninitialized (arbitrary) data of the given shape, dtype, and order. Object arrays will be initialized to None.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html</remarks>
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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html</remarks>
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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.empty.html</remarks>
        public static NDArray empty(Shape shape)
        {
            return new NDArray(NPTypeCode.Double, shape, false);
        }
    }
}
