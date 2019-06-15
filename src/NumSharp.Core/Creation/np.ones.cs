using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(params int[] shapes)
        {
            return ones(typeof(double), shapes);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Type dtype = null, params int[] shapes)
        {
            return ones(new Shape(shapes), dtype: dtype);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <typeparam name="T">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</typeparam>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones<T>(params int[] shapes)
        {
            return ones(new Shape(shapes), typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, Type dtype = null)
        {
            dtype = dtype ?? typeof(double);
            var nd = new NDArray(dtype, 0).ones(dtype, shape);
            return nd;
        }
    }
}
