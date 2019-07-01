using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    //todo!  write manually Complex[] ToComplex(String[] sourceArray)
    //todo!  that knows to parse input like pythons complex syntax: 10+5j   readmore: https://stackoverflow.com/a/28873083/1481186

    /// <summary>
    ///     Presents all possible combinations of array conversion of types supported by numpy.
    /// </summary>
    /// <remarks>Implementation is based on Array.ConvertAll from corefx source at https://github.com/dotnet/corefx/blob/b2097cbdcb26f7f317252334ddcce101a20b7f3d/src/Common/src/CoreLib/System/Array.cs#L586</remarks>
    public static class ArrayConvert
    {
        #region Cloning

        /// <summary>
        ///     Creates a clone of given <see cref="sourceArray"/> using <see cref="Span{T}.CopyTo"/>.
        /// </summary>
        /// <param name="sourceArray">The array to clone</param>
        /// <remarks>If possible, for performance reasons use generic version of this method.</remarks>
        public static Array Clone(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            //handle element type
            var elementType = sourceArray.GetType().GetElementType();
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            Array output;
            //handle array length
            var dims = sourceArray.Rank;
            if (dims > 1)
            {
                int[] dimensions = new int[dims];
                for (int idx = 0; idx < dims; idx++)
                    dimensions[idx] = sourceArray.GetLength(idx);
                output = Arrays.Create(elementType, dimensions);
            }
            else
            {
                output = Arrays.Create(elementType, sourceArray.Length);
            }

            Array.Copy(sourceArray, 0, output, 0, sourceArray.Length);

            return output;
        }

        /// <summary>
        ///     Creates a clone of given <see cref="sourceArray"/> using <see cref="Span{T}.CopyTo"/>.
        /// </summary>
        /// <param name="sourceArray">The array to clone</param>
        public static T[] Clone<T>(T[] sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var output = new T[sourceArray.Length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Creates a clone of given <see cref="sourceArray"/> using <see cref="Array.CopyTo(System.Array,int)"/>.
        /// </summary>
        /// <param name="sourceArray">The array to clone</param>
        public static T[,] Clone<T>(T[,] sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var output = new T[sourceArray.GetLength(0), sourceArray.GetLength(1)];
            Array.Copy(sourceArray, 0, output, 0, sourceArray.Length);

            return output;
        }

        /// <summary>
        ///     Creates a clone of given <see cref="sourceArray"/> using <see cref="Array.CopyTo(System.Array,int)"/>.
        /// </summary>
        /// <param name="sourceArray">The array to clone</param>
        public static T[,,] Clone<T>(T[,,] sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var output = new T[sourceArray.GetLength(0), sourceArray.GetLength(1), sourceArray.GetLength(2)];
            Array.Copy(sourceArray, 0, output, 0, sourceArray.Length);

            return output;
        }

        /// <summary>
        ///     Creates a clone of given <see cref="sourceArray"/> using <see cref="Array.CopyTo(System.Array,int)"/>.
        /// </summary>
        /// <param name="sourceArray">The array to clone</param>
        public static T[,,,] Clone<T>(T[,,,] sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var output = new T[sourceArray.GetLength(0), sourceArray.GetLength(1), sourceArray.GetLength(2), sourceArray.GetLength(4)];
            Array.Copy(sourceArray, 0, output, 0, sourceArray.Length);

            return output;
        }

        #endregion

        #region NonGeneric To Type

        /// <summary>
        ///     Converts <see cref="sourceArray"/> to an array of type <see cref="returnType"/>.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <param name="returnType">The type to convert the data to</param>
        /// <returns></returns>
        /// <remarks>If <see cref="sourceArray"/>'s element type equals to <see cref="returnType"/> then a copy is returned</remarks>
        public static Array To(Array sourceArray, Type returnType)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            if (returnType == null)
            {
                throw new ArgumentNullException(nameof(returnType));
            }

            if (returnType.IsArray)
            {
                returnType = returnType.GetElementType();
            }

            switch (returnType.GetTypeCode())
            {
#if _REGEN
                %foreach supported_dtypes %
                case NPTypeCode.#1:
                {
                    return To#1(arr);
                }
                %
#else

                case NPTypeCode.Boolean:
                {
                    return ToBoolean(sourceArray);
                }

                case NPTypeCode.Byte:
                {
                    return ToByte(sourceArray);
                }

                case NPTypeCode.Int16:
                {
                    return ToInt16(sourceArray);
                }

                case NPTypeCode.UInt16:
                {
                    return ToUInt16(sourceArray);
                }

                case NPTypeCode.Int32:
                {
                    return ToInt32(sourceArray);
                }

                case NPTypeCode.UInt32:
                {
                    return ToUInt32(sourceArray);
                }

                case NPTypeCode.Int64:
                {
                    return ToInt64(sourceArray);
                }

                case NPTypeCode.UInt64:
                {
                    return ToUInt64(sourceArray);
                }

                case NPTypeCode.Char:
                {
                    return ToChar(sourceArray);
                }

                case NPTypeCode.Double:
                {
                    return ToDouble(sourceArray);
                }

                case NPTypeCode.Single:
                {
                    return ToSingle(sourceArray);
                }

                case NPTypeCode.Decimal:
                {
                    return ToDecimal(sourceArray);
                }

                case NPTypeCode.String:
                {
                    return ToString(sourceArray);
                }

                case NPTypeCode.Complex:
                {
                    return ToComplex(sourceArray);
                }
#endif
                default:
                    throw new NotSupportedException($"Unable to convert {sourceArray.GetType().GetElementType()?.Name} to {returnType?.Name}.");
            }
        }

        /// <summary>
        ///     Converts <see cref="sourceArray"/> to an array of type <see cref="typeCode"/>.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <param name="typeCode">The type to convert the data to</param>
        /// <returns></returns>
        /// <remarks>If <see cref="sourceArray"/>'s element type equals to <see cref="typeCode"/> then a copy is returned</remarks>
        public static Array To(Array sourceArray, NPTypeCode typeCode)
        {
            switch (typeCode)
            {
#if _REGEN
                %foreach supported_dtypes %
                case NPTypeCode.#1:
                {
                    return To#1(arr);
                }
                %
#else

                case NPTypeCode.Boolean:
                {
                    return ToBoolean(sourceArray);
                }

                case NPTypeCode.Byte:
                {
                    return ToByte(sourceArray);
                }

                case NPTypeCode.Int16:
                {
                    return ToInt16(sourceArray);
                }

                case NPTypeCode.UInt16:
                {
                    return ToUInt16(sourceArray);
                }

                case NPTypeCode.Int32:
                {
                    return ToInt32(sourceArray);
                }

                case NPTypeCode.UInt32:
                {
                    return ToUInt32(sourceArray);
                }

                case NPTypeCode.Int64:
                {
                    return ToInt64(sourceArray);
                }

                case NPTypeCode.UInt64:
                {
                    return ToUInt64(sourceArray);
                }

                case NPTypeCode.Char:
                {
                    return ToChar(sourceArray);
                }

                case NPTypeCode.Double:
                {
                    return ToDouble(sourceArray);
                }

                case NPTypeCode.Single:
                {
                    return ToSingle(sourceArray);
                }

                case NPTypeCode.Decimal:
                {
                    return ToDecimal(sourceArray);
                }

                case NPTypeCode.String:
                {
                    return ToString(sourceArray);
                }

                case NPTypeCode.Complex:
                {
                    return ToComplex(sourceArray);
                }
#endif
                default:
                    throw new NotSupportedException($"Unable to convert {sourceArray.GetType().GetElementType()?.Name} to NPTypeCode.{typeCode}.");
            }
        }

        /// <summary>
        ///     Converts <see cref="sourceArray"/> to an array of type <see cref="returnType"/>.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <param name="returnType">The type to convert the data to</param>
        /// <returns></returns>
        /// <remarks>If <see cref="sourceArray"/>'s element type equals to <see cref="returnType"/> then a copy is returned</remarks>
        public static T[] To<T>(Array sourceArray)
        {
            return (T[])To(sourceArray, typeof(T)); // no need for direct cast.
        }

        #endregion

        #region From NonGeneric

#if _REGEN
        %foreach supported_dtypes%

        public static #1[] To#1(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return To#1(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return To#1(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return To#1(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return To#1(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return To#1(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return To#1(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return To#1(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return To#1(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return To#1(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return To#1(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return To#1(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return To#1(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return To#1(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
        %

#else
        public static Boolean[] ToBoolean(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToBoolean(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToBoolean(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToBoolean(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToBoolean(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToBoolean(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToBoolean(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToBoolean(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToBoolean(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToBoolean(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToBoolean(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToBoolean(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToBoolean(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToBoolean(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Byte[] ToByte(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToByte(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToByte(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToByte(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToByte(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToByte(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToByte(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToByte(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToByte(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToByte(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToByte(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToByte(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToByte(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToByte(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Int16[] ToInt16(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToInt16(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToInt16(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToInt16(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToInt16(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToInt16(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToInt16(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToInt16(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToInt16(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToInt16(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToInt16(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToInt16(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToInt16(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToInt16(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static UInt16[] ToUInt16(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToUInt16(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToUInt16(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToUInt16(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToUInt16(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToUInt16(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToUInt16(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToUInt16(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToUInt16(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToUInt16(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToUInt16(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToUInt16(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToUInt16(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToUInt16(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Int32[] ToInt32(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToInt32(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToInt32(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToInt32(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToInt32(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToInt32(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToInt32(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToInt32(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToInt32(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToInt32(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToInt32(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToInt32(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToInt32(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToInt32(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static UInt32[] ToUInt32(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToUInt32(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToUInt32(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToUInt32(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToUInt32(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToUInt32(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToUInt32(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToUInt32(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToUInt32(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToUInt32(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToUInt32(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToUInt32(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToUInt32(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToUInt32(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Int64[] ToInt64(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToInt64(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToInt64(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToInt64(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToInt64(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToInt64(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToInt64(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToInt64(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToInt64(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToInt64(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToInt64(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToInt64(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToInt64(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToInt64(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static UInt64[] ToUInt64(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToUInt64(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToUInt64(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToUInt64(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToUInt64(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToUInt64(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToUInt64(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToUInt64(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToUInt64(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToUInt64(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToUInt64(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToUInt64(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToUInt64(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToUInt64(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Char[] ToChar(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToChar(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToChar(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToChar(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToChar(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToChar(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToChar(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToChar(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToChar(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToChar(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToChar(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToChar(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToChar(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToChar(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Double[] ToDouble(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToDouble(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToDouble(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToDouble(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToDouble(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToDouble(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToDouble(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToDouble(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToDouble(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToDouble(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToDouble(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToDouble(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToDouble(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToDouble(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Single[] ToSingle(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToSingle(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToSingle(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToSingle(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToSingle(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToSingle(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToSingle(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToSingle(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToSingle(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToSingle(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToSingle(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToSingle(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToSingle(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToSingle(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Decimal[] ToDecimal(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToDecimal(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToDecimal(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToDecimal(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToDecimal(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToDecimal(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToDecimal(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToDecimal(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToDecimal(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToDecimal(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToDecimal(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToDecimal(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToDecimal(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToDecimal(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static String[] ToString(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToString(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToString(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToString(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToString(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToString(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToString(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToString(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToString(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToString(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToString(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToString(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToString(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToString(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        public static Complex[] ToComplex(Array sourceArray)
        {
            if (sourceArray == null)
            {
                throw new ArgumentNullException(nameof(sourceArray));
            }

            var fromTypeCode = sourceArray.GetType().GetElementType().GetTypeCode();
            switch (fromTypeCode)
            {
                case NPTypeCode.Boolean:
                    return ToComplex(sourceArray as Boolean[]);
                case NPTypeCode.Byte:
                    return ToComplex(sourceArray as Byte[]);
                case NPTypeCode.Int16:
                    return ToComplex(sourceArray as Int16[]);
                case NPTypeCode.UInt16:
                    return ToComplex(sourceArray as UInt16[]);
                case NPTypeCode.Int32:
                    return ToComplex(sourceArray as Int32[]);
                case NPTypeCode.UInt32:
                    return ToComplex(sourceArray as UInt32[]);
                case NPTypeCode.Int64:
                    return ToComplex(sourceArray as Int64[]);
                case NPTypeCode.UInt64:
                    return ToComplex(sourceArray as UInt64[]);
                case NPTypeCode.Char:
                    return ToComplex(sourceArray as Char[]);
                case NPTypeCode.Double:
                    return ToComplex(sourceArray as Double[]);
                case NPTypeCode.Single:
                    return ToComplex(sourceArray as Single[]);
                case NPTypeCode.Decimal:
                    return ToComplex(sourceArray as Decimal[]);
                case NPTypeCode.String:
                    return ToComplex(sourceArray as String[]);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
#endif

        #endregion

        #region Generic

        #region To Same Type

#if _REGEN
        %foreach supported_dtypes%
        
        /// <summary>
        ///     Converts <see cref="#1"/> array to a <see cref="#1"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type #1</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static #1[] To#1(#1[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));
            
            var length = sourceArray.Length;
            var output = new #1[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }
        %
#else


        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Complex"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        /// <remarks>Based on benchmark ArrayCopying</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Complex[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            sourceArray.AsSpan().CopyTo(output);

            return output;
        }
#endif

        #endregion

#if _REGEN
        %foreach forevery(supported_primitives, supported_primitives, true)%
        
        /// <summary>
        ///     Converts <see cref="#1"/> array to a <see cref="#2"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type #2</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static #2[] To#2(#1[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));
            
            var length = sourceArray.Length;
            var output = new #2[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.To#2(sourceArray[i]);
            }

            return output;
        }
        %
#else

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="String"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String[] ToString(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new String[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToString(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Boolean"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean[] ToBoolean(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Boolean[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToBoolean(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Byte"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte[] ToByte(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Byte[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToByte(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Int16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16[] ToInt16(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="UInt16"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16[] ToUInt16(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt16[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt16(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Int32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32[] ToInt32(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="UInt32"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32[] ToUInt32(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt32[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt32(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Int64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64[] ToInt64(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Int64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="UInt64"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64[] ToUInt64(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new UInt64[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToUInt64(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Char"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char[] ToChar(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Char[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToChar(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Double"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double[] ToDouble(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDouble(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Single"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single[] ToSingle(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Single[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToSingle(sourceArray[i]);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Decimal"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal[] ToDecimal(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Decimal[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = Convert.ToDecimal(sourceArray[i]);
            }

            return output;
        }
#endif

        #endregion

        #region Complex

#if _REGEN
        %foreach supported_primitives%
        
        /// <summary>
        ///     Converts <see cref="#1"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(#1[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));
            
            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }
        %
#else


        /// <summary>
        ///     Converts <see cref="Boolean"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Boolean[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Byte"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Byte[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int16"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Int16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt16"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(UInt16[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int32"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Int32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt32"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(UInt32[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Int64"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Int64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="UInt64"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(UInt64[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Char"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Char[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Double"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Double[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Single"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Single[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="Decimal"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(Decimal[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }

        /// <summary>
        ///     Converts <see cref="String"/> array to a <see cref="Complex"/> array.
        /// </summary>
        /// <param name="sourceArray">The array to convert</param>
        /// <returns>Converted array of type Complex</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex[] ToComplex(String[] sourceArray)
        {
            if (sourceArray == null)
                throw new ArgumentNullException(nameof(sourceArray));

            var length = sourceArray.Length;
            var output = new Complex[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = new Complex(Convert.ToDouble(sourceArray[i]), 0d);
            }

            return output;
        }
#endif

        #endregion
    }
}
