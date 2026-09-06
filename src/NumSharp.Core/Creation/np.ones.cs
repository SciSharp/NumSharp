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
        public static NDArray ones(int[] shape, Type dtype)
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
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted (Array-API parity).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, Type dtype, string device = null)
        {
            ValidateDevice(device);
            return ones(shape, (dtype ?? typeof(double)).GetTypeCode());
        }


        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape)
        {
            return ones(shape, NPTypeCode.Double);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="typeCode">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, NPTypeCode typeCode)
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
        ///     <c>np.ones(shape, dtype, order='C')</c> order parameter (mirrors <see cref="empty(Shape, char, Type)"/>).
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="order">Memory layout: 'C' (row-major), 'F' (column-major), 'A'/'K' (default to 'C' with no source).</param>
        /// <param name="dtype">Desired data-type. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <returns>Array of ones in the requested layout (the fill is order-independent, so only the flags differ).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, char order, Type dtype = null)
        {
            char physical = OrderResolver.Resolve(order);
            var orderedShape = new Shape(shape.dimensions, physical);
            return ones(orderedShape, (dtype ?? typeof(double)).GetTypeCode());
        }
    }
}
