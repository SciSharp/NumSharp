using System;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    public static class Arrays
    {
        /// <summary>
        ///     Creates an array of 1D of type <paramref name="type"/>.
        /// </summary>
        /// <typeparam name="T">The type of the array</typeparam>
        /// <param name="type">The type to create this array.</param>
        /// <param name="length">The length of the array</param>
        /// <remarks>Do not use this if you are trying to create jagged or multidimensional array.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array Create(Type type, int length)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();

            return Array.CreateInstance(type, length);
        }

        /// <summary>
        ///     Creates an array of specific <paramref name="length"/> of type <paramref name="type"/>.
        /// </summary>
        /// <param name="type">The type to create this array.</param>
        /// <param name="length">The length of the array</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array Create(Type type, int[] length)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();

            return Array.CreateInstance(type, length);
        }

        /// <summary>
        ///     Creates an array 1D of type <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The type of the array</typeparam>
        /// <param name="length">The length of the array</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T[] Create<T>(int length)
        {
            return new T[length];
        }

        /// <summary>
        ///     Creates an array of 1D of type <paramref name="typeCode"/>.
        /// </summary>
        /// <param name="typeCode">The type to create this array.</param>
        /// <param name="length">The length of the array</param>
        /// <remarks>Do not use this if you are trying to create jagged or multidimensional array.</remarks>
        public static Array Create(NPTypeCode typeCode, int length)
        {
            switch (typeCode)
            {
#if _REGEN
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    return new #2[length];
                }
                %
                default:
                    throw new NotImplementedException();
#else

                case NPTypeCode.NDArray:
                {
                    return new NDArray[length];
                }

                case NPTypeCode.Complex:
                {
                    return new Complex[length];
                }

                case NPTypeCode.Boolean:
                {
                    return new bool[length];
                }

                case NPTypeCode.Byte:
                {
                    return new byte[length];
                }

                case NPTypeCode.Int16:
                {
                    return new short[length];
                }

                case NPTypeCode.UInt16:
                {
                    return new ushort[length];
                }

                case NPTypeCode.Int32:
                {
                    return new int[length];
                }

                case NPTypeCode.UInt32:
                {
                    return new uint[length];
                }

                case NPTypeCode.Int64:
                {
                    return new long[length];
                }

                case NPTypeCode.UInt64:
                {
                    return new ulong[length];
                }

                case NPTypeCode.Char:
                {
                    return new char[length];
                }

                case NPTypeCode.Double:
                {
                    return new double[length];
                }

                case NPTypeCode.Single:
                {
                    return new float[length];
                }

                case NPTypeCode.Decimal:
                {
                    return new decimal[length];
                }

                case NPTypeCode.String:
                {
                    return new string[length];
                }

                default:
                    throw new NotImplementedException();
#endif
            }
        }

        /// <summary>
        ///     Creates an array of 1D of type <paramref name="typeCode"/> with length of 1 and a single <paramref name="value"/> inside.
        /// </summary>
        /// <param name="typeCode">The type to create this array.</param>
        /// <param name="value">The value to insert</param>
        /// <remarks>Do not use this if you are trying to create jagged or multidimensional array.</remarks>
        public static Array Wrap(NPTypeCode typeCode, object value)
        {
            switch (typeCode)
            {
#if _REGEN
                %foreach supported_dtypes,supported_dtypes_lowercase%
                case NPTypeCode.#1:
                {
                    return new #2[1] {(#1)value};
                }
                %
                default:
                    throw new NotImplementedException();
#else

                case NPTypeCode.NDArray:
                {
                    return new NDArray[1] {(NDArray)value};
                }

                case NPTypeCode.Complex:
                {
                    return new Complex[1] {(Complex)value};
                }

                case NPTypeCode.Boolean:
                {
                    return new bool[1] {(Boolean)value};
                }

                case NPTypeCode.Byte:
                {
                    return new byte[1] {(Byte)value};
                }

                case NPTypeCode.Int16:
                {
                    return new short[1] {(Int16)value};
                }

                case NPTypeCode.UInt16:
                {
                    return new ushort[1] {(UInt16)value};
                }

                case NPTypeCode.Int32:
                {
                    return new int[1] {(Int32)value};
                }

                case NPTypeCode.UInt32:
                {
                    return new uint[1] {(UInt32)value};
                }

                case NPTypeCode.Int64:
                {
                    return new long[1] {(Int64)value};
                }

                case NPTypeCode.UInt64:
                {
                    return new ulong[1] {(UInt64)value};
                }

                case NPTypeCode.Char:
                {
                    return new char[1] {(Char)value};
                }

                case NPTypeCode.Double:
                {
                    return new double[1] {(Double)value};
                }

                case NPTypeCode.Single:
                {
                    return new float[1] {(Single)value};
                }

                case NPTypeCode.Decimal:
                {
                    return new decimal[1] {(Decimal)value};
                }

                case NPTypeCode.String:
                {
                    return new string[1] {(String)value};
                }

                default:
                    throw new NotImplementedException();
#endif
            }
        }

        /// <summary>
        ///     Extracts shape and type from given <paramref name="array"/>.
        /// </summary>
        /// <param name="array">The array to extract D<see cref="Type"/> and <see cref="Shape"/> from.</param>
        public static (Shape Shape, Type DType) ExtractStructure(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            //get lengths incase it is multi-dimensional
            if (array.Rank > 1)
            {
                int[] dim = new int[array.Rank];
                for (int idx = 0; idx < dim.Length; idx++)
                    dim[idx] = array.GetLength(idx);
                var shape = new Shape(dim);
                Type elementType = array.GetType();
                // ReSharper disable once PossibleNullReferenceException
                while (elementType.IsArray)
                    elementType = elementType.GetElementType();

                return (shape, elementType);
            }

            // single dimension.
            return (new Shape(array.Length), array.GetType().GetElementType());
        }

        /// <summary>
        ///     Extracts shape and type from given <paramref name="array"/>.
        /// </summary>
        /// <param name="array">The array to extract D<see cref="Type"/> and <see cref="Shape"/> from.</param>
        public static (Shape Shape, Type DType) ExtractStructure<T>(T[] array)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }

            //this is single dimensional array.
            var shape = new Shape(array.Length);
            Type elementType = array.GetType().GetElementType();

            return (shape, elementType);
        }
    }
}
