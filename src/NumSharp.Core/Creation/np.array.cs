using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Wraps given <paramref name="nd"/> in an alias. If <paramref name="copy"/> is true then returns a clone.
        /// </summary>
        /// <param name="nd"></param>
        /// <param name="copy">If <paramref name="copy"/> is true then returns a clone.</param>
        [MethodImpl((MethodImplOptions)768)]
        public static NDArray array(NDArray nd, bool copy = false) => copy ? new NDArray(nd.Storage.Clone()) : new NDArray(nd.Storage);

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from an array with an unknown size or dtype.
        /// </summary>
        /// <param name="ndmin">Specifies the minimum number of dimensions that the resulting array should have. Ones will be pre-pended to the shape as needed to meet this requirement.</param>
        /// <param name="copy">Always copies if the array is larger than 1-d.</param>
        /// <param name="order">Not used.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        [MethodImpl((MethodImplOptions)512)]
        [SuppressMessage("ReSharper", "InvalidXmlDocComment")]
        public static NDArray array(Array array, Type dtype = null, int ndmin = 1, bool copy = true, char order = 'C')
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            var arrType = array.ResolveElementType();

            //handle dim expansion and extract shape
            Shape shape;
            var dims = array.ResolveRank();
            var missing = dims - ndmin;

            if (missing < 0)
            {
                shape = Arrays.Concat(Enumerable.Repeat(1, Math.Abs(missing)).ToArray(), Shape.ExtractShape(array));
            }
            else
            {
                shape = Shape.ExtractShape(array);
            }

            //flatten
            if (shape.NDim > 1)
            {
                array = Arrays.Flatten(array);
                copy = false;
            }

            if (dtype != null && dtype != arrType)
            {
                array = ArrayConvert.To(array, dtype);
                copy = false;
            }

            return new NDArray(copy ? (Array)array.Clone() : array, shape, order);
        }

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        public static NDArray array<T>(params T[] data) where T : unmanaged => new NDArray(ArraySlice.FromArray(data, true), Shape.Vector(data.Length));

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The enumeration of data to create <see cref="NDArray"/> from.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        public static NDArray array<T>(IEnumerable<T> data) where T : unmanaged
        {
            var slice = ArraySlice.FromArray(data.ToArray(), false);
            return new NDArray(slice, Shape.Vector(slice.Count));
        }

        /// <summary>
        ///     Creates a Vector <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The enumeration of data to create <see cref="NDArray"/> from.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>
        ///     https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>
        ///     Always performs a copy.<br></br>
        ///     <paramref name="size"/> can be used to limit the amount of items to read form <paramref name="data"/>. Reading stops on either size or <paramref name="data"/> ends.
        /// </remarks>
        public static NDArray array<T>(IEnumerable<T> data, int size) where T : unmanaged
        {
            var slice = new ArraySlice<T>(new UnmanagedMemoryBlock<T>(size));
            unsafe
            {
                using (var enumerator = data.GetEnumerator())
                {
                    Func<bool> next = enumerator.MoveNext;

                    var addr = slice.Address;
                    for (int i = 0; i < size && next(); i++)
                    {
                        addr[i] = enumerator.Current;
                    }
                }
            }

            return new NDArray(slice, Shape.Vector(slice.Count));
        }

        /// <summary>
        ///     Create a vector <see cref="NDArray"/> of dtype <see cref="char"/>.
        /// </summary>
        /// <param name="chars"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NDArray array(string chars)
        {
            if (chars == null)
                throw new ArgumentNullException(nameof(chars));
            if (chars.Length == 0)
                return new NDArray(NPTypeCode.Char, 0);

            unsafe
            {
                var ret = new ArraySlice<char>(new UnmanagedMemoryBlock<char>(chars.Length));
                fixed (char* strChars = chars)
                {
                    var src = strChars;
                    var dst = ret.Address;
                    var len = sizeof(char) * chars.Length;
                    Buffer.MemoryCopy(src, dst, len, len);
                }

                return new NDArray(ret);
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Shape is taken from the first item of each array/nested array.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl((MethodImplOptions)512)]
        public static NDArray array<T>(T[][] data) where T : unmanaged
        {
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2));

                var strides = @out.strides;
                int stride1 = strides[0];
                Debug.Assert(strides[1] == 1);

                T* addr = (T*)@out.Address;
                Parallel.For(0, len1, i1 =>
                {
                    T* addr1 = addr + i1 * stride1;
                    T[] src1 = data[i1];
                    Parallel.For(0, len2, i2 =>
                    {
                        *(addr1 + i2) = src1[i2];
                    });
                });

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Shape is taken from the first item of each array/nested array.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl((MethodImplOptions)512)]
        public static NDArray array<T>(T[][][] data) where T : unmanaged
        {
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3));

                var strides = @out.strides;
                int stride1 = strides[0];
                int stride2 = strides[1];
                Debug.Assert(strides[2] == 1);

                T* addr = (T*)@out.Address;
                Parallel.For(0, len1, i1 =>
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][] src1 = data[i1];
                    Parallel.For(0, len2, i2 =>
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[] src2 = src1[i2];
                        Parallel.For(0, len3, i3 =>
                        {
                            *(addr2 + i3) = src2[i3];
                        });
                    });
                });

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Shape is taken from the first item of each array/nested array.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl((MethodImplOptions)512)]
        public static NDArray array<T>(T[][][][] data) where T : unmanaged
        {
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;
                int len4 = data[0][0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3, len4));

                var strides = @out.strides;
                int stride1 = strides[0];
                int stride2 = strides[1];
                int stride3 = strides[2];
                Debug.Assert(strides[3] == 1);

                T* addr = (T*)@out.Address;
                Parallel.For(0, len1, i1 =>
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][][] src1 = data[i1];
                    Parallel.For(0, len2, i2 =>
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[][] src2 = src1[i2];
                        Parallel.For(0, len3, i3 =>
                        {
                            T* addr3 = addr2 + i3 * stride3;
                            T[] src3 = src2[i3];
                            Parallel.For(0, len4, i4 =>
                            {
                                *(addr3 + i4) = src3[i4];
                            });
                        });
                    });
                });

                return @out;
            }
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from. Shape is taken from the first item of each array/nested array.</param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html <br></br>Always performs a copy.</remarks>
        [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        [MethodImpl((MethodImplOptions)512)]
        public static NDArray array<T>(T[][][][][] data) where T : unmanaged
        {
            unsafe
            {
                int len1 = data.Length;
                int len2 = data[0].Length;
                int len3 = data[0][0].Length;
                int len4 = data[0][0][0].Length;
                int len5 = data[0][0][0][0].Length;

                NDArray @out = new NDArray(InfoOf<T>.NPTypeCode, new Shape(len1, len2, len3, len4, len5));

                var strides = @out.strides;
                int stride1 = strides[0];
                int stride2 = strides[1];
                int stride3 = strides[2];
                int stride4 = strides[3];
                Debug.Assert(strides[4] == 1);

                T* addr = (T*)@out.Address;
                Parallel.For(0, len1, i1 =>
                {
                    T* addr1 = addr + i1 * stride1;
                    T[][][][] src1 = data[i1];
                    Parallel.For(0, len2, i2 =>
                    {
                        T* addr2 = addr1 + i2 * stride2;
                        T[][][] src2 = src1[i2];
                        Parallel.For(0, len3, i3 =>
                        {
                            T* addr3 = addr2 + i3 * stride3;
                            T[][] src3 = src2[i3];
                            Parallel.For(0, len4, i4 =>
                            {
                                T* addr4 = addr3 + i4 * stride4;
                                T[] src4 = src3[i4];
                                Parallel.For(0, len5, i5 =>
                                {
                                    *(addr4 + i5) = src4[i5];
                                });
                            });
                        });
                    });
                });

                return @out;
            }
        }

#if _REGEN1
        %l = "data.GetLength("
        %r = ")"
        %foreach range(1,16)%

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[#(repeat(",", i))] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(#(repeat("^l+str(n.ToString())+r", i+1, ", "))));
        }
        %
#else

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13), data.GetLength(14)));
        }

        /// <summary>
        ///     Creates an <see cref="NDArray"/> from given <paramref name="data"/>.
        /// </summary>
        /// <typeparam name="T">The type of given array, must be compliant to numpy's supported dtypes.</typeparam>
        /// <param name="data">The array to create <see cref="NDArray"/> from.</param>
        /// <param name="copy">
        ///     If true then the array will be copied to a newly allocated memory.<br></br>
        ///     If false then the array will be pinned by calling <see cref="GCHandle.Alloc(object)"/>.
        /// </param>
        /// <returns>An <see cref="NDArray"/> with the data and shape of the given array.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html</remarks>
        public static NDArray array<T>(T[,,,,,,,,,,,,,,,] data, bool copy = true) where T : unmanaged
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));

            return new NDArray(ArraySlice.FromArray(data, copy), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2), data.GetLength(3), data.GetLength(4), data.GetLength(5), data.GetLength(6), data.GetLength(7), data.GetLength(8), data.GetLength(9), data.GetLength(10), data.GetLength(11), data.GetLength(12), data.GetLength(13), data.GetLength(14), data.GetLength(15)));
        }
#endif
    }
}
