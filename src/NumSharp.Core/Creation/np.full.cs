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
        /// <param name="fill_value">Fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array) that
        ///     broadcasts into the result, as NumPy's <c>copyto(a, fill_value, casting='unsafe')</c> broadcasts it.</param>
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Default (null) infers from fill_value (an array fill value gives its own dtype).
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <returns>Array of fill_value with the given shape, dtype, and order.</returns>
        /// <exception cref="NotSupportedException"><paramref name="fill_value"/> is null (Python's <c>None</c>) and
        ///     <paramref name="dtype"/> is null: NumPy builds an object array there, a dtype NumSharp does not have.</exception>
        /// <exception cref="TypeError"><paramref name="fill_value"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype, for a non-empty shape: Python's <c>int(None)</c>, raised by NumPy's
        ///     <c>copyto(a, None, casting='unsafe')</c> (an empty result converts nothing and returns).</exception>
        /// <exception cref="ValueError"><paramref name="fill_value"/> is null and <paramref name="dtype"/> is a
        ///     <c>StringDType(coerce=False)</c>; or an array fill value that does not broadcast into the result (<c>could not
        ///     broadcast input array from shape (2,) into shape (3,)</c>), or whose text does not parse into a numeric
        ///     <paramref name="dtype"/>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.full.html
        ///     <para>A null <paramref name="fill_value"/> is Python's <c>None</c>, filled the way NumPy's
        ///     <c>a = empty(shape, dtype); copyto(a, None, casting='unsafe')</c> fills it: NaN (the positive <c>NPY_NAN</c>) for
        ///     a float dtype, <c>nan+nanj</c> for complex, <c>False</c> for bool, <c>'None'</c> (truncated) for <c>S</c> /
        ///     <c>U</c>, a <c>StringDType</c>'s NA object or <c>'None'</c>. Before NumSharp stored None, this returned ZEROS.</para>
        /// </remarks>
        public static NDArray full(Shape shape, object fill_value, DType dtype = null, string device = null)
        {
            DeviceArguments.ValidateDevice(device);
            return NDArray.Full(shape, fill_value, dtype);
        }
    }
}
