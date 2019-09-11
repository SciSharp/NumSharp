using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shapes">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(ValueType fill_value, params int[] shapes)
        {
            return full(fill_value, shapes, null);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shapes">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full<T>(ValueType fill_value, params int[] shapes) where T : unmanaged
        {
            return full(fill_value, shapes, typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="dtype">The desired data-type for the array The default, null, means np.array(fill_value).dtype.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(ValueType fill_value, Shape shape, Type dtype)
        {
            return full(fill_value, shape, (fill_value.GetType()).GetTypeCode());
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(ValueType fill_value, Shape shape)
        {
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(fill_value.GetType(), shape.size, fill_value), shape));
        }


        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="dtype">The desired data-type for the array The default, null, means np.array(fill_value).dtype.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, ValueType fill_value, Type dtype)
        {
            return full(fill_value, shape, dtype);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="typeCode">The desired data-type for the array The default, null, means np.array(fill_value).dtype.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, ValueType fill_value, NPTypeCode typeCode)
        {
            return full(fill_value, shape, typeCode);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, ValueType fill_value)
        {
            return full(fill_value, shape);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="fill_value">Fill value.</param>
        /// <param name="shape">Shape of the empty array, e.g., (2, 3) or 2.</param>
        /// <param name="typeCode">The desired data-type for the array The default, null, means np.array(fill_value).dtype.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.full.html</remarks>
        public static NDArray full(ValueType fill_value, Shape shape, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, Converts.ChangeType(fill_value, (TypeCode)typeCode)), shape));
        }
    }
}
