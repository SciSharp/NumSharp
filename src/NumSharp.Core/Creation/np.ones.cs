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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, DType dtype, string device = null)
        {
            ValidateDevice(device);
            return OnesCore(shape, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }


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
        ///     <see cref="DType"/> descriptor parameter, like NumPy's <c>dtype=</c>).
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="typeCode">The desired data-type for the array.</param>
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

            return new NDArray(ArraySlice.Allocate(typeCode, shape.size, one), shape);
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
            var orderedShape = new Shape(shape.dimensions, physical);
            return OnesCore(orderedShape, dtype?.GetTypeCode() ?? NPTypeCode.Double);
        }
    }
}
