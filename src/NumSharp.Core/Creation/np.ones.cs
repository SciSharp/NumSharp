using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
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
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(params int[] shapes)
        {
            return ones(typeof(double), shapes);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Type dtype = null, params int[] shapes)
        {
            return ones(new Shape(shapes), dtype: dtype);
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shapes">Shape of the new array.</param>
        /// <typeparam name="T">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</typeparam>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones<T>(params int[] shapes)
        {
            return ones(new Shape(shapes), typeof(T));
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">Shape of the new array.</param>
        /// <param name="dtype">The desired data-type for the array, e.g., <see cref="uint8"/>. Default is <see cref="float64"/> / <see cref="double"/>.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html</remarks>
        public static NDArray ones(Shape shape, Type dtype = null)
        {
            //TODO! allocate with fill..
            dtype = dtype ?? typeof(double);
            var nd = new NDArray(dtype, shape); //already allocates inside.
            object one = null;
            switch (nd.GetTypeCode)
            {
                case NPTypeCode.Complex:
                    one = new Complex(1d, 0d);
                    break;
                case NPTypeCode.NDArray:
                    one = NDArray.Scalar(1, np.int32);
                    break;                
                case NPTypeCode.String:
                    one = "1";
                    break;
                default:
                    one = Convert.ChangeType((byte)1, dtype);
                    break;
            }

            switch (nd.GetTypeCode)
            {

#if _REGEN
                %foreach supported_currently_supported, supported_currently_supported_lowercase%
                case NPTypeCode.#1:
                {
                    ((ArraySlice<#2>) nd.Array).AsSpan.Fill((#2)one);
                    break;
                }
                %
                default:
                    throw new NotImplementedException();
#else
                case NPTypeCode.Byte:
                {
                    ((ArraySlice<byte>) nd.Array).AsSpan.Fill((byte)one);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    ((ArraySlice<short>) nd.Array).AsSpan.Fill((short)one);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    ((ArraySlice<ushort>) nd.Array).AsSpan.Fill((ushort)one);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    ((ArraySlice<int>) nd.Array).AsSpan.Fill((int)one);
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    ((ArraySlice<uint>) nd.Array).AsSpan.Fill((uint)one);
                    break;
                }
                case NPTypeCode.Int64:
                {
                    ((ArraySlice<long>) nd.Array).AsSpan.Fill((long)one);
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    ((ArraySlice<ulong>) nd.Array).AsSpan.Fill((ulong)one);
                    break;
                }
                case NPTypeCode.Char:
                {
                    ((ArraySlice<char>) nd.Array).AsSpan.Fill((char)one);
                    break;
                }
                case NPTypeCode.Double:
                {
                    ((ArraySlice<double>) nd.Array).AsSpan.Fill((double)one);
                    break;
                }
                case NPTypeCode.Single:
                {
                    ((ArraySlice<float>) nd.Array).AsSpan.Fill((float)one);
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    ((ArraySlice<decimal>) nd.Array).AsSpan.Fill((decimal)one);
                    break;
                }
                default:
                    throw new NotImplementedException();
#endif
            }

            return nd;
        }
    }
}
