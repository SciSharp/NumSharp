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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(int shape)
        {
            return zeros(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(int[] shape)
        {
            return zeros(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(long[] shape)
        {
            return zeros(new Shape(shape), (Type)null);
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, type <typeparamref name="T"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros<T>(int[] shape) where T : unmanaged
        {
            return zeros(new Shape(shape), typeof(T));
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <returns>Array of zeros with the given shape, type <typeparamref name="T"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros<T>(long[] shape) where T : unmanaged
        {
            return zeros(new Shape(shape), typeof(T));
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape, Type dtype, string device = null)
        {
            ValidateDevice(device);
            return zeros(shape, (dtype ?? typeof(double)).GetTypeCode());
        }

        /// <summary>
        ///     Return a new double array of given shape, filled with zeros.
        /// </summary>
        /// <param name="shape">Shape of the new array,</param>
        /// <param name="typeCode">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of zeros with the given shape, dtype.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape)
        {
            return new NDArray(NPTypeCode.Double, shape, true); //already allocates inside.
        }

        /// <summary>
        ///     Return a new array of zeros with a specified memory layout — the port of NumPy's
        ///     <c>np.zeros(shape, dtype, order='C')</c> order parameter (mirrors <see cref="empty(Shape, char, Type)"/>).
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="order">Memory layout: 'C' (row-major), 'F' (column-major), 'A'/'K' (default to 'C' with no source).</param>
        /// <param name="dtype">Desired data-type. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of zeros in the requested layout (the fill is order-independent, so only the flags differ).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.zeros.html</remarks>
        public static NDArray zeros(Shape shape, char order, Type dtype = null)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = new Shape(shape.dimensions, physical);
            return new NDArray((dtype ?? typeof(double)).GetTypeCode(), orderedShape, true);
        }
    }
}
