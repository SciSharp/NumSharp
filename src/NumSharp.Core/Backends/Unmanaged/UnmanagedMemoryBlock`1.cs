using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Unmanaged.Memory;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public unsafe struct UnmanagedMemoryBlock<T> : IUnmanagedMemoryBlock, IMemoryBlock<T>, IEnumerable<T>, IEquatable<UnmanagedMemoryBlock<T>>, ICloneable where T : unmanaged
    {
        private readonly Disposer _disposer;
        public readonly long Count;
        public readonly T* Address;
        public readonly long BytesCount;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <remarks>Does claim ownership since allocation is publicly.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        public UnmanagedMemoryBlock(long count)
        {
            var bytes = BytesCount = count * InfoOf<T>.Size;
            var ptr = Marshal.AllocHGlobal(new IntPtr(bytes));
            _disposer = new Disposer(ptr);
            Address = (T*)ptr;
            Count = count;
        }

        /// <summary>
        ///     Construct as a wrapper around pointer and given length without claiming ownership.
        /// </summary>
        /// <param name="ptr"></param>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <remarks>Does claim ownership.</remarks>
        [MethodImpl((MethodImplOptions)512)]
        public UnmanagedMemoryBlock(T* ptr, long count)
        {
            _disposer = Disposer.Null;
            Address = ptr;
            Count = count;
            BytesCount = count * InfoOf<T>.Size;
        }

        /// <summary>
        ///     Construct with externally allocated memory and a custom <paramref name="dispose"/> function.
        /// </summary>
        /// <param name="start"></param>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <param name="dispose"></param>
        /// <remarks>Does claim ownership.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedMemoryBlock(T* start, long count, Action dispose)
        {
            Count = count;
            BytesCount = InfoOf<T>.Size * count;
            _disposer = new Disposer(dispose);
            Address = start;
        }

        /// <summary>
        ///     Construct with externally allocated memory settings this memory block as owner.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <remarks>Does claim ownership.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedMemoryBlock(GCHandle handle, long count)
        {
            Count = count;
            BytesCount = InfoOf<T>.Size * count;
            Address = (T*)handle.AddrOfPinnedObject();
            _disposer = new Disposer(handle);
        }

        /// <summary>
        ///     Construct with externally allocated memory and a custom <paramref name="dispose"/> function.
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <param name="dispose"></param>
        /// <remarks>Does claim ownership.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedMemoryBlock(GCHandle handle, long count, Action dispose)
        {
            Count = count;
            BytesCount = InfoOf<T>.Size * count;
            Address = (T*)handle.AddrOfPinnedObject();
            _disposer = new Disposer(dispose);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        /// <param name="fill"></param>
        [MethodImpl((MethodImplOptions)768)]
        public UnmanagedMemoryBlock(long count, T fill) : this(count)
        {
            Fill(fill, 0, count);
        }

        #region Static

        #region FromArray

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }


        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,,,] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromArray(T[,,,,,,,,,,,,,,,] arr, bool copy)
        {
            if (!copy)
                return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length);

            var ret = new UnmanagedMemoryBlock<T>(arr.Length);
            fixed (T* arrptr = arr)
            {
                new UnmanagedMemoryBlock<T>(arrptr, arr.Length).CopyTo(ret);
            }

            return ret;
        }

        #endregion

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromBuffer(byte[] arr)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(arr, GCHandleType.Pinned), arr.Length / InfoOf<T>.Size);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromBuffer(byte[] arr, bool copy)
        {
            return new UnmanagedMemoryBlock<T>(GCHandle.Alloc(copy ? arr.Clone() : arr, GCHandleType.Pinned), arr.Length / InfoOf<T>.Size);
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> FromPool(StackedMemoryPool manager)
        {
            Debug.Assert(manager.SingleSize / InfoOf<T>.Size > 0);
            var buffer = manager.Take();
            return new UnmanagedMemoryBlock<T>((T*)buffer, 1, () => manager.Return(buffer));
        }

        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> Copy(UnmanagedMemoryBlock<T> source)
        {
            var itemCount = source.Count;
            var len = itemCount * InfoOf<T>.Size;
            var ret = new UnmanagedMemoryBlock<T>(itemCount);
            source.CopyTo(ret);
            //source.AsSpan().CopyTo(ret.AsSpan()); //TODO! Benchmark at netcore 3.0, it should be faster than buffer.memorycopy.
            return ret;
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="address">The address of the first <typeparamref name="T"/></param>
        /// <param name="count">How many <typeparamref name="T"/> to copy, not how many bytes.</param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> Copy(void* address, int count)
        {
            var len = count * InfoOf<T>.Size;
            var ret = new UnmanagedMemoryBlock<T>(count);
            new UnmanagedMemoryBlock<T>((T*)address, count).CopyTo(ret);
            //source.AsSpan().CopyTo(ret.AsSpan()); //TODO! Benchmark at netcore 3.0, it should be faster than buffer.memorycopy.
            return ret;
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="address"></param>
        /// <param name="count">How many <typeparamref name="T"/> to copy, not how many bytes.</param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> Copy(IntPtr address, int count)
        {
            return Copy((void*)address, count);
        }

        /// <summary>
        ///     
        /// </summary>
        /// <param name="address">The address of the first <typeparamref name="T"/></param>
        /// <param name="count">How many <typeparamref name="T"/> to copy, not how many bytes.</param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public static UnmanagedMemoryBlock<T> Copy(T* address, int count)
        {
            return Copy((void*)address, count);
        }

        #endregion

        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(Address + index);
            [MethodImpl((MethodImplOptions)768)] set => *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void Reallocate(long length, bool copyOldValues = false)
        {
            if (copyOldValues)
            {
                var @new = new UnmanagedMemoryBlock<T>(length);
                var bytes = Math.Min(Count, length) * InfoOf<T>.Size;
                Buffer.MemoryCopy(Address, @new.Address, bytes, bytes);
                Free();
                this = @new;
            }
            else
            {
                //we do not have to allocate first in we do not copy values.
                Free();
                this = new UnmanagedMemoryBlock<T>(length);
            }
        }

        [MethodImpl((MethodImplOptions)512)]
        public void Reallocate(long length, T fill, bool copyOldValues = false)
        {
            if (copyOldValues)
            {
                var @new = new UnmanagedMemoryBlock<T>(length);
                var bytes = Math.Min(Count, length) * InfoOf<T>.Size;
                Buffer.MemoryCopy(Address, @new.Address, bytes, bytes);

                if (length > Count)
                {
                    @new.Fill(fill, Count, length - Count);
                }

                Free();
                this = @new;
            }
            else
            {
                //we do not have to allocate first in we do not copy values.
                Free();
                this = new UnmanagedMemoryBlock<T>(length, fill);
            }
        }

        /// <summary>
        ///     Fills the contents of this span with the given value.
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value)
        {
            if (Count == 0)
                return;

            if (Unsafe.SizeOf<T>() == 1 && Count < uint.MaxValue)
            {
                T tmp = value; // Avoid taking address of the "value" argument. It would regress performance of the loop below.

                Unsafe.InitBlockUnaligned(Address, Unsafe.As<T, byte>(ref tmp), (uint)Count);
            }
            else
            {
                // Do all math as nuint to avoid unnecessary 64->32->64 bit integer truncations
                UInt64 length = (uint)Count;
                if (length == 0)
                    return;

                T* addr = Address;

                // TODO: Create block fill for value types of power of two sizes e.g. 2,4,8,16

                ulong i = 0;
                for (; i < (length & ~7UL); i += 8)
                {
                    *(addr + (i)) = value;
                    *(addr + (i + 1)) = value;
                    *(addr + (i + 2)) = value;
                    *(addr + (i + 3)) = value;
                    *(addr + (i + 4)) = value;
                    *(addr + (i + 5)) = value;
                    *(addr + (i + 6)) = value;
                    *(addr + (i + 7)) = value;
                }

                if (i < (length & ~3UL))
                {
                    *(addr + (i)) = value;
                    *(addr + (i + 1)) = value;
                    *(addr + (i + 2)) = value;
                    *(addr + (i + 3)) = value;
                    i += 4;
                }

                for (; i < length; i++)
                {
                    *(addr + i) = value;
                }
            }
        }

        /// <summary>
        ///     Fills the contents of this span with the given value.
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value, long offset, long count)
        {
            if (Count == 0 || count == 0)
                return;

            if (Unsafe.SizeOf<T>() == 1 && Count < uint.MaxValue)
            {
                if (Count - offset <= 0)
                    return;

                T tmp = value; // Avoid taking address of the "value" argument. It would regress performance of the loop below.

                Unsafe.InitBlockUnaligned(Address + offset, Unsafe.As<T, byte>(ref tmp), (uint)count);
            }
            else
            {
                // Do all math as nuint to avoid unnecessary 64->32->64 bit integer truncations
                if (Count - offset <= 0)
                    return;

                ulong length = (ulong)count;
                T* addr = Address + offset;

                // TODO: Create block fill for value types of power of two sizes e.g. 2,4,8,16

                ulong i = 0;
                for (; i < (length & ~7UL); i += 8)
                {
                    *(addr + (i)) = value;
                    *(addr + (i + 1)) = value;
                    *(addr + (i + 2)) = value;
                    *(addr + (i + 3)) = value;
                    *(addr + (i + 4)) = value;
                    *(addr + (i + 5)) = value;
                    *(addr + (i + 6)) = value;
                    *(addr + (i + 7)) = value;
                }

                if (i < (length & ~3UL))
                {
                    *(addr + (i)) = value;
                    *(addr + (i + 1)) = value;
                    *(addr + (i + 2)) = value;
                    *(addr + (i + 3)) = value;
                    i += 4;
                }

                for (; i < length; i++)
                {
                    *(addr + i) = value;
                }
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetIndex(int index)
        {
            return *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public ref T GetRefTo(int index)
        {
            return ref *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, ref T value)
        {
            *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, T value)
        {
            *(Address + index) = value;
        }


        [MethodImpl((MethodImplOptions)768)]
        public T GetIndex(long index)
        {
            return *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public ref T GetRefTo(long index)
        {
            return ref *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(long index, ref T value)
        {
            *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(long index, T value)
        {
            *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void Free()
        {
            _disposer.Dispose();
        }

        [MethodImpl((MethodImplOptions)768)]
        public IEnumerator<T> GetEnumerator()
        {
            for (var i = 0; i < Count; i++) yield return GetIndex(i);
        }

        [MethodImpl((MethodImplOptions)768)]
        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        [MethodImpl((MethodImplOptions)768)]
        public bool Contains(T item)
        {
            long len = Count;
            for (var i = 0; i < len; i++)
            {
                if ((*(Address + i)).Equals(item)) return true;
            }

            return false;
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(T[] array, int arrayIndex)
        {
            long len = Count;
            for (var i = 0; i < len; i++)
            {
                array[i + arrayIndex] = *(Address + i);
            }
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(UnmanagedMemoryBlock<T> memoryBlock, long arrayIndex)
        {
            //TODO! at netcore 3, AsSpan.CopyTo might be faster.
            Buffer.MemoryCopy(Address + arrayIndex, memoryBlock.Address, InfoOf<T>.Size * memoryBlock.Count, InfoOf<T>.Size * (Count - arrayIndex));
        }

        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(T* array, long arrayIndex, long lengthToCopy)
        {
            //TODO! at netcore 3, AsSpan.CopyTo might be faster.
            var len = InfoOf<T>.Size * lengthToCopy;
            Buffer.MemoryCopy(Address + arrayIndex, array, len, len);
        }

        /// <summary>Copies the elements of the <see cref="T:System.Collections.ICollection" /> to an <see cref="T:System.Array" />, starting at a particular <see cref="T:System.Array" /> index.</summary>
        /// <param name="array">The one-dimensional <see cref="T:System.Array" /> that is the destination of the elements copied from <see cref="T:System.Collections.ICollection" />. The <see cref="T:System.Array" /> must have zero-based indexing. </param>
        /// <param name="arrayIndex">The zero-based index in <paramref name="array" /> at which copying begins. </param>
        /// <exception cref="T:System.ArgumentNullException">
        /// <paramref name="array" /> is <see langword="null" />. </exception>
        /// <exception cref="T:System.ArgumentOutOfRangeException">
        /// <paramref name="arrayIndex" /> is less than zero. </exception>
        /// <exception cref="T:System.ArgumentException">
        /// <paramref name="array" /> is multidimensional.-or- The number of elements in the source <see cref="T:System.Collections.ICollection" /> is greater than the available space from <paramref name="arrayIndex" /> to the end of the destination <paramref name="array" />.-or-The type of the source <see cref="T:System.Collections.ICollection" /> cannot be cast automatically to the type of the destination <paramref name="array" />.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public void CopyTo(Array array, int arrayIndex)
        {
            if (!(array is T[] arr))
                throw new InvalidCastException("Unable to CopyTo a type that is not " + typeof(T).Name + "[]");

            CopyTo(arr, arrayIndex);
        }

        /// <summary>
        ///     Performs a copy to this memory block.
        /// </summary>
        /// <returns></returns>
        public UnmanagedMemoryBlock<T> Clone()
        {
            var ret = new UnmanagedMemoryBlock<T>(Count);
            this.CopyTo(ret);
            return ret;
        }

        #region Explicit Implementations

        /// <summary>
        ///     The size of a single item stored in <see cref="IMemoryBlock.Address"/>.
        /// </summary>
        int IMemoryBlock.ItemLength => InfoOf<T>.Size;

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe T* IMemoryBlock<T>.Address => Address;

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe void* IMemoryBlock.Address => Address;

        /// <summary>
        ///     How many items are stored in <see cref="IMemoryBlock.Address"/>?
        /// </summary>
        /// <remarks></remarks>
        long IMemoryBlock.Count => Count;

        /// <summary>
        ///     The items with length of <see cref="IMemoryBlock.TypeCode"/> are present in <see cref="IMemoryBlock.Address"/>.
        /// </summary>
        /// <remarks>Calculated by <see cref="IMemoryBlock.Count"/>*<see cref="IMemoryBlock.ItemLength"/></remarks>
        long IMemoryBlock.BytesLength => BytesCount;

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of the type stored inside this memory block.
        /// </summary>
        NPTypeCode IMemoryBlock.TypeCode => InfoOf<T>.NPTypeCode;

        [EditorBrowsable(EditorBrowsableState.Never)]
        public ref T GetPinnableReference()
        {
            return ref *Address;
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        object ICloneable.Clone()
        {
            return Clone();
        }

        #endregion

        #region Equality

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>
        /// <see langword="true" /> if the current object is equal to the <paramref name="other" /> parameter; otherwise, <see langword="false" />.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public bool Equals(UnmanagedMemoryBlock<T> other)
        {
            return Count == other.Count && Address == other.Address;
        }

        /// <summary>Indicates whether this instance and a specified object are equal.</summary>
        /// <param name="obj">The object to compare with the current instance. </param>
        /// <returns>
        /// <see langword="true" /> if <paramref name="obj" /> and this instance are the same type and represent the same value; otherwise, <see langword="false" />. </returns>
        [MethodImpl((MethodImplOptions)768)]
        public override bool Equals(object obj)
        {
            return obj is UnmanagedMemoryBlock<T> other && Equals(other);
        }

        /// <summary>Returns the hash code for this instance.</summary>
        /// <returns>A 32-bit signed integer that is the hash code for this instance.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public override int GetHashCode()
        {
            unchecked
            {
                return (unchecked((int)Count) * 397) ^ unchecked((int)(long)Address);
            }
        }

        /// <summary>Returns a value that indicates whether the values of two <see cref="T:NumSharp.Backends.Unmanaged.UnmanagedArray`1" /> objects are equal.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if the <paramref name="left" /> and <paramref name="right" /> parameters have the same value; otherwise, false.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public static bool operator ==(UnmanagedMemoryBlock<T> left, UnmanagedMemoryBlock<T> right)
        {
            return left.Equals(right);
        }

        /// <summary>Returns a value that indicates whether two <see cref="T:NumSharp.Backends.Unmanaged.UnmanagedArray`1" /> objects have different values.</summary>
        /// <param name="left">The first value to compare.</param>
        /// <param name="right">The second value to compare.</param>
        /// <returns>true if <paramref name="left" /> and <paramref name="right" /> are not equal; otherwise, false.</returns>
        [MethodImpl((MethodImplOptions)768)]
        public static bool operator !=(UnmanagedMemoryBlock<T> left, UnmanagedMemoryBlock<T> right)
        {
            return !left.Equals(right);
        }

        #endregion

        private class Disposer : IDisposable
        {
            public static readonly Disposer Null = new Disposer();

            private enum AllocationType
            {
                AllocHGlobal,
                GCHandle,
                External,
                Wrap
            }

            private bool Disposed;
            private readonly AllocationType _type;

            private readonly IntPtr Address;
            private readonly GCHandle _gcHandle;
            private readonly Action _dispose;


            /// <summary>
            ///     Construct a AllocationType.AllocHGlobal
            /// </summary>
            /// <param name="address"></param>
            public Disposer(IntPtr address)
            {
                Address = address;
                _type = AllocationType.AllocHGlobal;
            }

            /// <summary>
            ///     Construct a AllocationType.GCHandle
            /// </summary>
            /// <param name="gcHandle"></param>
            public Disposer(GCHandle gcHandle)
            {
                _gcHandle = gcHandle;
                _type = AllocationType.GCHandle;
            }

            /// <summary>
            ///     Construct a AllocationType.External
            /// </summary>
            /// <param name="dispose"></param>
            public Disposer(Action dispose)
            {
                _dispose = dispose;
                _type = AllocationType.External;
            }

            /// <summary>
            ///     Construct a AllocationType.Wrap
            /// </summary>
            private Disposer()
            {
                _type = AllocationType.Wrap;
            }

            [MethodImpl((MethodImplOptions)768), SuppressMessage("ReSharper", "PossibleInvalidOperationException")]
            private void ReleaseUnmanagedResources()
            {
                if (Disposed)
                    return;

                Disposed = true;

                switch (_type)
                {
                    case AllocationType.AllocHGlobal:
                        Marshal.FreeHGlobal(Address);
                        return;
                    case AllocationType.Wrap:
                        return;
                    case AllocationType.External:
                        _dispose();
                        return;
                    case AllocationType.GCHandle:
                        _gcHandle.Free();
                        return;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            /// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
            public void Dispose()
            {
                ReleaseUnmanagedResources();
                GC.SuppressFinalize(this);
            }

            /// <summary>Allows an object to try to free resources and perform other cleanup operations before it is reclaimed by garbage collection.</summary>
            ~Disposer()
            {
                ReleaseUnmanagedResources();
            }
        }
    }
}
