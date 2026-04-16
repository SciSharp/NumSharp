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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, Type dtype)
        {
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
                    one = '1';
                    break;
                default:
                    one = Converts.ChangeType((byte)1, typeCode);
                    break;
            }

            return new NDArray(ArraySlice.Allocate(typeCode, shape.size, one), shape);
        }
    }
}
