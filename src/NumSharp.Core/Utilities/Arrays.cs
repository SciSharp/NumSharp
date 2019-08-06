using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    public static class Arrays
    {
        /// <summary>
        ///     Inserts item into a specific index.
        /// </summary>
        /// <param name="source">The array to insert the value to.</param>
        /// <param name="index">The index to insert to.</param>
        /// <param name="value"></param>
        public static void Insert<T>(ref T[] source, int index, T value)
        {
            if (index < 0 || index > source.Length)
                throw new ArgumentOutOfRangeException(nameof(index));

            Array.Resize(ref source, source.Length + 1);
            Array.Copy(source, index, source, index + 1, source.Length - index - 1);
            source[index] = value;
        }

        /// <summary>
        ///     Inserts item into a specific index.
        /// </summary>
        /// <param name="source">The array to insert the value to.</param>
        /// <param name="index">The index to insert to.</param>
        /// <param name="value"></param>
        public static T[] Insert<T>(T[] source, int index, T value) where T : unmanaged
        {
            if (index < 0 || index > source.Length)
                throw new ArgumentOutOfRangeException(nameof(index));

            unsafe
            {
                var ret = new T[source.Length + 1];
                fixed (T* src = source)
                {
                    fixed (T* dst = ret)
                    {
                        new Span<T>(src, index).CopyTo(new Span<T>(dst, index));
                        *(dst+index) = value;
                        var left = source.Length - index;
                        new Span<T>(src + index, left).CopyTo(new Span<T>(dst + index + 1, left));
                    }
                }

                return ret;
            }
        }

        /// <summary>
        ///     Inserts item into a specific index.
        /// </summary>
        /// <param name="source">The array to insert copy and insert value to.</param>
        /// <param name="index">The index to insert to.</param>
        /// <returns>a copy of <see cref="source"/> with the appended value.</returns>
        public static T[] AppendAt<T>(T[] source, int index, T value)
        {
            var ret = (T[])source.Clone();
            Insert(ref source, index, value);
            return ret;
        }

        /// <summary>
        ///     Removes a specific index from given array.
        /// </summary>
        /// <param name="source">The array to remove <paramref name="index"/> from.</param>
        /// <param name="index">The index to remove.</param>
        /// <returns></returns>
        public static T[] RemoveAt<T>(this T[] source, int index)
        {
            T[] dest = new T[source.Length - 1];
            if (index > 0)
                Array.Copy(source, 0, dest, 0, index);

            if (index < source.Length - 1)
                Array.Copy(source, index + 1, dest, index, source.Length - index - 1);

            return dest;
        }

        /// <summary>
        ///     Flattens any type of <see cref="Array"/>.
        /// </summary>
        /// <remarks>Supports both jagged array and multi-dim arrays.</remarks>
        public static Array Flatten(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            IEnumerable _flat(IEnumerable @this)
            {
                foreach (var item in @this)
                {
                    if (item is IEnumerable enumerable)
                    {
                        foreach (var subitem in _flat(enumerable))
                        {
                            yield return subitem;
                        }
                    }
                    else yield return item;
                }
            }

            return Arrays.Create(array.ResolveElementType(), _flat(array));
        }

        /// <summary>
        ///     Performs fast concatenation of multiple arrays
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arrays"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)512)]
        public static T[] Concat<T>(params T[][] arrays)
        {
            int sum = 0;
            foreach (var array in arrays)
            {
                sum += array.Length;
            }

            T[] ret = new T[sum];
            int offset = 0;
            for (int i = 0; i < arrays.Length; i++)
            {
                var arr = arrays[i];
                var arrlen = arr.Length;
                Array.Copy(arr, 0, ret, offset, arrlen);
                offset += arrlen;
            }

            return ret;
        }

        /// <summary>
        ///     Performs fast concatenation of multiple arrays
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="arrays"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static T[] Concat<T>(T[] left, T[] right)
        {
            T[] ret = new T[left.Length + right.Length];
            Array.Copy(left, 0, ret, 0, left.Length);
            Array.Copy(right, 0, ret, left.Length, right.Length);

            return ret;
        }

        /// <summary>
        ///     Resolves <see cref="Array"/> element type recusivly.
        /// </summary>
        /// <param name="arr"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static Type ResolveElementType(this Array arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            var t = arr.GetType().GetElementType();
            // ReSharper disable once PossibleNullReferenceException
            while (t.IsArray)
                t = t.GetElementType();

            return t;
        }

        /// <summary>
        ///     Resolves <see cref="Array"/>'s rank, supports both jagged array and multidim array.
        /// </summary>
        /// <returns>The number of ranks <paramref name="arr"/> has</returns>
        [MethodImpl((MethodImplOptions)768)]
        public static int ResolveRank(this Array arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            var nestedArraysRenk = 1;

            var t = arr.GetType().GetElementType();
            // ReSharper disable once PossibleNullReferenceException
            while (t.IsArray)
            {
                t = t.GetElementType();
                nestedArraysRenk++;
            }

            return Math.Max(arr.Rank, nestedArraysRenk);
        }

        /// <summary>
        ///     Resolves the shape of this given array.
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Supports multi-dim and jagged arrays.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static (Shape Shape, Type Type) ResolveShapeAndType(this Array array)
        {
            //get lengths incase it is multi-dimensional
            if (array.Rank > 1)
            {
                //is multidim
                int[] dim = new int[array.Rank];
                for (int idx = 0; idx < dim.Length; idx++)
                    dim[idx] = array.GetLength(idx);
                return (new Shape(dim), array.GetType().GetElementType());
            }
            else
            {
                if (array.GetType().GetElementType()?.IsArray == true)
                {
                    //is jagged
                    Array curr = array;
                    var dimList = new List<int>(16);
                    do
                    {
                        var child = (Array)curr.GetValue(0);
                        dimList.Add(child.Length);
                        curr = child;
                    } while (curr.GetType().GetElementType()?.IsArray == true);

                    return (new Shape(dimList.ToArray()), curr.GetType().GetElementType());
                }

                //is 1d
                return (new Shape(array.Length), array.GetType().GetElementType());
            }
        }

        /// <summary>
        ///     Resolves the shape of this given array.
        /// </summary>
        /// <param name="array"></param>
        /// <remarks>Supports multi-dim and jagged arrays.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public static Shape ResolveShape(this Array array)
        {
            Shape shape;
            //get lengths incase it is multi-dimensional
            if (array.Rank > 1)
            {
                int[] dim = new int[array.Rank];
                for (int idx = 0; idx < dim.Length; idx++)
                    dim[idx] = array.GetLength(idx);
                shape = new Shape(dim);
            }
            else
            {
                if (array.GetType().GetElementType()?.IsArray == true)
                {
                    //is jagged
                    Array curr = array;
                    var dimList = new List<int>(16);
                    do
                    {
                        var child = (Array)curr.GetValue(0);
                        dimList.Add(child.Length);
                        curr = child;
                    } while (curr.GetType().GetElementType()?.IsArray == true);

                    return new Shape(dimList.ToArray());
                }

                shape = new Shape(array.Length);
            }

            return shape;
        }

        /// <summary>
        ///     Creates an array of 1D of type <paramref name="type"/>.
        /// </summary>
        /// <typeparam name="T">The type of the array</typeparam>
        /// <param name="type">The type to create this array.</param>
        /// <param name="length">The length of the array</param>
        /// <remarks>Do not use this if you are trying to create jagged or multidimensional array.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array Create(Type type, IEnumerable enumerable)
        {
            // ReSharper disable once PossibleNullReferenceException
            while (type.IsArray)
                type = type.GetElementType();
            var l = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(type));
            foreach (var v in enumerable)
                l.Add(v);

            return (Array)l.GetType().GetMethod("ToArray", BindingFlags.Public | BindingFlags.Instance).Invoke(l, null);
        }

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
                %foreach all_dtypes,all_dtypes_lowercase%
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
                %foreach all_dtypes,all_dtypes_lowercase%
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
