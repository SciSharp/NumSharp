using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
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
        ///     Creates a clone of given <see cref="sourceArray"/>.
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
        ///     Creates a clone of given <see cref="sourceArray"/>.
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
#if _REGEN1
                %foreach supported_dtypes %
                case NPTypeCode.#1: return To#1(sourceArray);
                %
#else
                case NPTypeCode.Boolean: return ToBoolean(sourceArray);
                case NPTypeCode.Byte: return ToByte(sourceArray);
                case NPTypeCode.Int32: return ToInt32(sourceArray);
                case NPTypeCode.Int64: return ToInt64(sourceArray);
                case NPTypeCode.Single: return ToSingle(sourceArray);
                case NPTypeCode.Double: return ToDouble(sourceArray);
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
#if _REGEN1
                %foreach supported_dtypes %
                case NPTypeCode.#1: return To#1(sourceArray);
                %
#else

                case NPTypeCode.Boolean: return ToBoolean(sourceArray);
                case NPTypeCode.Byte: return ToByte(sourceArray);
                case NPTypeCode.Int32: return ToInt32(sourceArray);
                case NPTypeCode.Int64: return ToInt64(sourceArray);
                case NPTypeCode.Single: return ToSingle(sourceArray);
                case NPTypeCode.Double: return ToDouble(sourceArray);
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

#if _REGEN1
        %foreach all_dtypes%

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
                    return To#1((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return To#1((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return To#1((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return To#1((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return To#1((Single[]) sourceArray);
                case NPTypeCode.String:
                    return To#1((String[]) sourceArray);
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
                    return ToBoolean((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToBoolean((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToBoolean((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToBoolean((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToBoolean((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToBoolean((String[]) sourceArray);
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
                    return ToByte((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToByte((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToByte((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToByte((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToByte((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToByte((String[]) sourceArray);
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
                    return ToInt32((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToInt32((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToInt32((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToInt32((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToInt32((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToInt32((String[]) sourceArray);
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
                    return ToInt64((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToInt64((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToInt64((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToInt64((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToInt64((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToInt64((String[]) sourceArray);
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
                    return ToSingle((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToSingle((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToSingle((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToSingle((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToSingle((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToSingle((String[]) sourceArray);
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
                    return ToDouble((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToDouble((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToDouble((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToDouble((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToDouble((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToDouble((String[]) sourceArray);
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
                    return ToString((Boolean[]) sourceArray);
                case NPTypeCode.Byte:
                    return ToString((Byte[]) sourceArray);
                case NPTypeCode.Int32:
                    return ToString((Int32[]) sourceArray);
                case NPTypeCode.Int64:
                    return ToString((Int64[]) sourceArray);
                case NPTypeCode.Single:
                    return ToString((Single[]) sourceArray);
                case NPTypeCode.String:
                    return ToString((String[]) sourceArray);
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
#endif

        #endregion

        #region Generic

        #region To Same Type

#if _REGEN1
        %foreach all_dtypes%
        
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
#endif

        #endregion

#if _REGEN1
        #region Compute
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
            Parallel.For(0, length, i=> output[i] = Converts.To#2(sourceArray[i]));
            return output;
        }
        %
        #endregion
#else


        #region Compute
        
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToString(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToBoolean(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToByte(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt32(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToInt64(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToSingle(sourceArray[i]));
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
            Parallel.For(0, length, i=> output[i] = Converts.ToDouble(sourceArray[i]));
            return output;
        }
        #endregion
#endif

        #endregion

        #region Complex

#if _REGEN1
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
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

            Parallel.For(0, length, i => new Complex(Converts.ToDouble(sourceArray[i]), 0d));
            return output;
        }
#endif

        #endregion
    }
}
