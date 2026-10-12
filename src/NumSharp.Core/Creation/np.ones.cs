using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(int shape)
        {
            return ones(new Shape(shape), typeof(double));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(int[] shape)
        {
            return ones(new Shape(shape), typeof(double));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(long[] shape)
        {
            return ones(new Shape(shape), typeof(double));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(int[] shape, DType dtype)
        {
            return ones(new Shape(shape), dtype: dtype);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <typeparam name="T">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</typeparam>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones<T>(int[] shape) where T : unmanaged
        {
            return ones(new Shape(shape), typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">
        ///     The desired dtype for the array — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> (<see cref="uint8"/>) all convert
        ///     implicitly. Default (null) is <see cref="float64"/> / <see cref="double"/>.
        /// </param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ones.html. A string dtype is NumPy's
        ///     <c>a = empty(shape, dtype); copyto(a, 1, casting='unsafe')</c>: every element is the text <c>'1'</c> (an
        ///     unsized <c>'U'</c> / <c>'S'</c> request is ONE character, NumPy's allocation rule).
        /// </remarks>
        public static NDArray ones(Shape shape, DType dtype, string device = null)
        {
            ValidateDevice(device);
            if (Backends.Strings.NDStringStorage.IsString(dtype) || dtype?.Meta is VStringDTypeMeta)
                return OnesOfText(shape, dtype);
            return OnesCore(shape, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }

        /// <summary>
        ///     <c>np.ones</c> for a string dtype — NumPy's <c>empty</c> + <c>copyto(a, 1, casting='unsafe')</c>: the number one
        ///     written as its <c>str()</c>, <c>'1'</c>, into every element (<see cref="Backends.Strings.NDStringArrays.Full"/>
        ///     with the Python int 1).
        /// </summary>
        /// <param name="shape">The dimensions (a fresh dense layout is kept, a view's re-laid out in C order).</param>
        /// <param name="dtype">The string dtype (sized or not).</param>
        /// <returns>The new string array.</returns>
        private static NDArray OnesOfText(Shape shape, DType dtype)
            => dtype.Meta is VStringDTypeMeta
                ? Backends.Strings.NDVStringMove.Full(shape, 1L, dtype) // the int 1 cast: '1' (coerce does not apply to a cast)
                : Backends.Strings.NDStringArrays.Full(shape, 1L, dtype);


        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape)
        {
            return OnesCore(shape, NPTypeCode.Double);
        }

        /// <summary>
        ///     The storage-lane core behind every public <c>ones(..., dtype)</c> overload (which all take the single
        ///     <see cref="DType"/> descriptor parameter, like NumPy's <c>dtype=</c>) and <c>ones_like</c>.
        /// </summary>
        /// <param name="shape">Shape of the new array. A dense layout (C, F or a KEEPORDER permutation at offset 0) is
        ///     kept, which is how <c>ones_like(a, order='K')</c> gets <c>a</c>'s memory order; any view layout is re-laid out
        ///     in C order (<see cref="Shape.ForFreshAllocation"/>).</param>
        /// <param name="typeCode">The desired data-type for the array.</param>
        /// <returns>A fresh, owned, writeable array of <paramref name="shape"/>'s dimensions with every element one.</returns>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows (<see cref="AllocationGuard"/>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        private static NDArray OnesCore(Shape shape, NPTypeCode typeCode)
        {
            object one = null;
            switch (typeCode)
            {
                case NPTypeCode.Complex:
                    one = new Complex(1d, 0d);
                    break;
                case NPTypeCode.Half:
                    one = (Half)1;
                    break;
                case NPTypeCode.SByte:
                    one = (sbyte)1;
                    break;
                case NPTypeCode.String:
                    one = "1";
                    break;
                case NPTypeCode.Char:
                    // Char is NumSharp's uint16-like numeric dtype. Numeric one is U+0001;
                    // the printable character '1' is 0x0031 and violates the uint16 oracle.
                    one = (char)1;
                    break;
                default:
                    one = Converts.ChangeType((byte)1, typeCode);
                    break;
            }

            // Allocates from shape.size directly (not via UnmanagedStorage.Allocate), so the
            // dimension guard has to be explicit here — see np.full.
            AllocationGuard.CheckDimensions(shape.dimensions, typeCode);

            // The storage constructor installs the shape as given, so the fresh-layout rule of the allocating funnel has
            // to be applied here: a view's Shape (np.ones(x.Shape) for x = a[::4]) would otherwise address past the
            // `size`-element buffer, and a read-only source's Shape would make the new array read-only. It must be the
            // storage constructor, as in np.full: the public wrap constructor new NDArray(IArraySlice, Shape) reads
            // dimensions only (NumPy's ndarray(shape, buffer=...)), which would turn the kept F / KEEPORDER layout of
            // ones_like(a, order='K') into C order.
            return new NDArray(new UnmanagedStorage(ArraySlice.Allocate(typeCode, shape.size, one), shape.ForFreshAllocation()));
        }

        /// <summary>
        ///     Return a new array of ones with a specified memory layout — the port of NumPy's
        ///     <c>np.ones(shape, dtype, order='C')</c> order parameter (mirrors <see cref="empty(Shape, char, DType)"/>).
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="order">Memory layout: 'C' (row-major), 'F' (column-major), 'A'/'K' (default to 'C' with no source).</param>
        /// <param name="dtype">Desired dtype (a <see cref="Type"/>, <see cref="NPTypeCode"/>, dtype string or <see cref="DType"/> — all convert implicitly). Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of ones in the requested layout (the fill is order-independent, so only the flags differ).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, char order, DType dtype = null)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = Shape.Adopt(shape.dimensions, physical);
            if (Backends.Strings.NDStringStorage.IsString(dtype) || dtype?.Meta is VStringDTypeMeta)
                return OnesOfText(orderedShape, dtype);
            return OnesCore(orderedShape, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }
    }
}
