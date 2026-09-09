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
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Default (null) infers from fill_value.
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        public static NDArray full(Shape shape, object fill_value, DType dtype = null, string device = null)
        {
            ValidateDevice(device);
            // When dtype is explicitly provided, use it; when null, infer from fill_value.
            var typeCode = dtype != null ? dtype.GetTypeCode() : fill_value.GetType().GetTypeCode();
            return FullCore(shape, fill_value, typeCode);
        }

        /// <summary>
        ///     The storage-lane core behind <see cref="full(Shape,object,DType,string)"/>: a new array of given shape and
        ///     lane, filled with fill_value.
        /// </summary>
        /// <param name="shape">Shape of the array, e.g., (2, 3) or 2.</param>
        /// <param name="fill_value">Fill value (scalar).</param>
        /// <param name="typeCode">The desired data-type for the array.</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.full.html</remarks>
        private static NDArray FullCore(Shape shape, object fill_value, NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            // This path allocates from shape.size directly instead of going through
            // UnmanagedStorage.Allocate, so it needs the dimension guard explicitly: for
            // (2^62, 2^62) the element count ITSELF wraps to 0, and a zero-element request is
            // one the allocator happily satisfies. Checking dimensions rather than the wrapped
            // product is the only way to see it.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, Converts.ChangeType(fill_value, typeCode)), shape));
        }
    }
}
