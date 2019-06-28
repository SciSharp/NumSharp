using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Utilities;

#if _REGEN_GLOBAL
    %supportedTypes = ["NDArray","Complex","Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Char","Double","Single","Decimal","String"]
    %supportTypesLower = ["NDArray","Complex","bool","byte","short","ushort","int","uint","long","ulong","char","double","float","decimal","string"]

    %supportedTypes_Primitives = ["Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Char","Double","Single","Decimal","String"]
    %supportTypesLower_Primitives = ["bool","byte","short","ushort","int","uint","long","ulong","char","double","float","decimal","string"]
#endif

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
                %foreach except(supportedTypes, "String")%
                case NPTypeCode.#1:
                {
                    new Span<#1>((#1[]) nd.Array).Fill((#1)one);
                    break;
                }
                %
                default:
                    throw new NotImplementedException();
#else
                case NPTypeCode.NDArray:
                {
                    new Span<NDArray>((NDArray[]) nd.Array).Fill((NDArray)one);
                    break;
                }
                case NPTypeCode.Complex:
                {
                    new Span<Complex>((Complex[]) nd.Array).Fill((Complex)one);
                    break;
                }
                case NPTypeCode.Boolean:
                {
                    new Span<Boolean>((Boolean[]) nd.Array).Fill((Boolean)one);
                    break;
                }
                case NPTypeCode.Byte:
                {
                    new Span<Byte>((Byte[]) nd.Array).Fill((Byte)one);
                    break;
                }
                case NPTypeCode.Int16:
                {
                    new Span<Int16>((Int16[]) nd.Array).Fill((Int16)one);
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    new Span<UInt16>((UInt16[]) nd.Array).Fill((UInt16)one);
                    break;
                }
                case NPTypeCode.Int32:
                {
                    new Span<Int32>((Int32[]) nd.Array).Fill((Int32)one);
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    new Span<UInt32>((UInt32[]) nd.Array).Fill((UInt32)one);
                    break;
                }
                case NPTypeCode.Int64:
                {
                    new Span<Int64>((Int64[]) nd.Array).Fill((Int64)one);
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    new Span<UInt64>((UInt64[]) nd.Array).Fill((UInt64)one);
                    break;
                }
                case NPTypeCode.Char:
                {
                    new Span<Char>((Char[]) nd.Array).Fill((Char)one);
                    break;
                }
                case NPTypeCode.Double:
                {
                    new Span<Double>((Double[]) nd.Array).Fill((Double)one);
                    break;
                }
                case NPTypeCode.Single:
                {
                    new Span<Single>((Single[]) nd.Array).Fill((Single)one);
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    new Span<Decimal>((Decimal[]) nd.Array).Fill((Decimal)one);
                    break;
                }
                case NPTypeCode.String:
                {
                    new Span<String>((String[]) nd.Array).Fill((String)one);
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
