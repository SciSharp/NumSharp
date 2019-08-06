using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shapes">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(params int[] shapes)
        {
            return zeros(shapes, null);
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shapes">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, type <typeparamref name="T"/>.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros<T>(params int[] shapes) where T : unmanaged
        {
            return zeros(shapes, typeof(T));
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape, Type dtype)
        {
            return zeros(shape, (dtype ?? typeof(double)).GetTypeCode());
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <param name="typeCode">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            return new NDArray(typeCode, shape, true); //already allocates inside.
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape)
        {
            return new NDArray(NPTypeCode.Double, shape, true); //already allocates inside.
        }
    }
}
