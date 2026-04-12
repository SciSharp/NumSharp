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
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(int[] shape, object fill_value)
            => full(new Shape(shape), fill_value);

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(long[] shape, object fill_value)
            => full(new Shape(shape), fill_value);

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full<T>(int[] shape, object fill_value) where T : unmanaged
            => full(new Shape(shape), fill_value, typeof(T));

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full<T>(long[] shape, object fill_value) where T : unmanaged
            => full(new Shape(shape), fill_value, typeof(T));

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <param name="dtype">The desired data-type for the array. Default infers from fill_value.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, object fill_value, Type dtype = null)
        {
            // When dtype is explicitly provided, use it
            if (dtype != null)
                return full(shape, fill_value, dtype.GetTypeCode());
            // When dtype is null, infer from fill_value
            return full(shape, fill_value, fill_value.GetType().GetTypeCode());
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <param name="typeCode">The desired data-type for the array.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, object fill_value, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode.AsType(), shape.size, Converts.ChangeType(fill_value, (TypeCode)typeCode)), shape));
        }
    }
}
