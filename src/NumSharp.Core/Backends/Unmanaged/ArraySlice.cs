using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public static class ArraySlice
    {
        public static ArraySlice<T> FromArray<T>(T[] array, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedArray<T>.FromArray(array, copy));
        }

        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromBuffer<T>(byte[] arr, bool copy = false) where T : unmanaged
        {
            return new ArraySlice<T>(UnmanagedArray<T>.FromBuffer(arr, copy));
        }

        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromPool<T>(int count, InternalBufferManager pool) where T : unmanaged
        {
            //TODO! Upgrade InternalBufferManager to use pre-pinned arrays.
            return new ArraySlice<T>(UnmanagedArray<T>.FromPool(count, pool));
        }
    }

    /// <summary>
    ///     <see cref="ArraySlice{T}"/> is similar to <see cref="Span{T}"/> but it can be moved around without having to follow `ref struct` rules.
    /// </summary>
    /// <typeparam name="T">The type of the internal unmanaged memory</typeparam>
    public readonly unsafe struct ArraySlice<T> : IArraySlice where T : unmanaged
    {
        private static readonly InternalBufferManager.PooledBufferManager _buffer = ScalarMemoryPool.Instance;

        /// <summary>
        ///     The memory block this <see cref="ArraySlice{T}"/> is stored in.
        /// </summary>
        /// <remarks>If <see cref="IsSlice"/> is false then this slice represents the entire MemoryBlock.</remarks>
        public readonly UnmanagedArray<T> MemoryBlock;

        public readonly T* Address;
        public readonly int Count;

        /// <summary>
        ///     Is this <see cref="ArraySlice{T}"/> a smaller part/slice of an unmanaged allocation?
        /// </summary>
        public readonly bool IsSlice;


        /// A Span representing this slice.
        public Span<T> AsSpan
        {
            [MethodImpl((MethodImplOptions)768)] get => new Span<T>(Address, Count);
        }

        public ArraySlice(UnmanagedArray<T> memoryBlock)
        {
            MemoryBlock = memoryBlock;
            IsSlice = false;
            Address = (T*)MemoryBlock.Address;
            Count = MemoryBlock.Count;
        }

        public ArraySlice(UnmanagedArray<T> memoryBlock, Span<T> slice)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = slice.Length;
            Address = (T*)Unsafe.AsPointer(ref slice.GetPinnableReference());
        }

        /// <returns></returns>
        public bool IsEmpty
        {
            [MethodImpl((MethodImplOptions)768)] get => Count == 0;
        }

        /// <param name="index"></param>
        /// <returns></returns>
        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(Address + index);
            [MethodImpl((MethodImplOptions)768)] set => *(Address + index) = value;
        }

        public int Length
        {
            [MethodImpl((MethodImplOptions)768)] get => Count;
        }

        public T* Start
        {
            [MethodImpl((MethodImplOptions)768)] get => Address;
        }

        /// <param name="value"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value)
        {
            AsSpan.Fill(value);
        }

        /// <param name="start"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start)
        {
            return new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start));
        }

        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start, int length)
        {
            return new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start, length));
        }

        /// <param name="destination"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination)
        {
            AsSpan.CopyTo(destination);
        }

        [EditorBrowsable(EditorBrowsableState.Never)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference()
        {
            return ref Unsafe.AsRef<T>(Address);
        }

        /// <summary>
        ///     Wrap a <see cref="T"/> inside <see cref="UnmanagedByteStorage{T}"/>.
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static ArraySlice<T> Scalar(T val) //TODO! use it, why is it not used.
        {
            return new ArraySlice<T>(UnmanagedArray<T>.FromPool(1, _buffer));
        }

        [MethodImpl((MethodImplOptions)768)]
        public T[] ToArray()
        {
            return AsSpan.ToArray();
        }

        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Clone()
        {
            return new ArraySlice<T>(UnmanagedArray<T>.Copy(Address, Count));
        }

        #region Explicit Interfaces

        object ICloneable.Clone()
        {
            return Clone();
        }

        Type IArraySlice.ArrayType => typeof(T);
        void* IArraySlice.Address => Address;
        int IArraySlice.Count => Count;

        IUnmanagedArray IArraySlice.MemoryBlock
        {
            [MethodImpl((MethodImplOptions)768)]
            get
            {
                return MemoryBlock;
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        T1 IArraySlice.GetIndex<T1>(int index)
        {
            return *(T1*)(void*)Address;
        }

        [MethodImpl((MethodImplOptions)768)]
        void IArraySlice.SetIndex<T1>(int index, T1 value)
        {
            *(T1*)(void*)Address = value;
        }

        #endregion

        /// <summary>
        ///     Performs dispose on the internal unmanaged memory block.<br></br>
        /// </summary>
        /// <remarks>
        ///     Dangerous because this array might be a slice and there might be other slices who are a slice from current internal unmanaged array.<br></br>
        ///     So releasing the unmanaged memory might cause memory corruption elsewhere.
        /// </remarks>
        public void DangerousFree()
        {
            // ReSharper disable once ImpureMethodCallOnReadonlyValueField
            MemoryBlock.Free();
        }

        #region Casts

        public static ArraySlice<T> FromArray(T[] array, bool copy = false)
        {
            return new ArraySlice<T>(UnmanagedArray<T>.FromArray(array, copy));
        }


        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromBuffer(byte[] arr, bool copy = false)
        {
            return new ArraySlice<T>(UnmanagedArray<T>.FromBuffer(arr, copy));
        }

        /// <param name="count">The length in objects of <typeparamref name="T"/> and not in bytes.</param>
        [MethodImpl((MethodImplOptions)768)]
        public static ArraySlice<T> FromPool(int count, InternalBufferManager pool)
        {
            //TODO! Upgrade InternalBufferManager to use pre-pinned arrays.
            return new ArraySlice<T>(UnmanagedArray<T>.FromPool(count, pool));
        }

#if _REGEN
        %foreach supported_currently_supported,supported_currently_supported_lowercase%
        public static ArraySlice<#2> FromArray(#2[] #2s, bool copy = false)
        {
            return new ArraySlice<#2>(UnmanagedArray<#2>.FromArray(#2s, copy));
        }

        %
#else
        public static ArraySlice<byte> FromArray(byte[] bytes, bool copy = false)
        {
            return new ArraySlice<byte>(UnmanagedArray<byte>.FromArray(bytes, copy));
        }

        public static ArraySlice<short> FromArray(short[] shorts, bool copy = false)
        {
            return new ArraySlice<short>(UnmanagedArray<short>.FromArray(shorts, copy));
        }

        public static ArraySlice<ushort> FromArray(ushort[] ushorts, bool copy = false)
        {
            return new ArraySlice<ushort>(UnmanagedArray<ushort>.FromArray(ushorts, copy));
        }

        public static ArraySlice<int> FromArray(int[] ints, bool copy = false)
        {
            return new ArraySlice<int>(UnmanagedArray<int>.FromArray(ints, copy));
        }

        public static ArraySlice<uint> FromArray(uint[] uints, bool copy = false)
        {
            return new ArraySlice<uint>(UnmanagedArray<uint>.FromArray(uints, copy));
        }

        public static ArraySlice<long> FromArray(long[] longs, bool copy = false)
        {
            return new ArraySlice<long>(UnmanagedArray<long>.FromArray(longs, copy));
        }

        public static ArraySlice<ulong> FromArray(ulong[] ulongs, bool copy = false)
        {
            return new ArraySlice<ulong>(UnmanagedArray<ulong>.FromArray(ulongs, copy));
        }

        public static ArraySlice<char> FromArray(char[] chars, bool copy = false)
        {
            return new ArraySlice<char>(UnmanagedArray<char>.FromArray(chars, copy));
        }

        public static ArraySlice<double> FromArray(double[] doubles, bool copy = false)
        {
            return new ArraySlice<double>(UnmanagedArray<double>.FromArray(doubles, copy));
        }

        public static ArraySlice<float> FromArray(float[] floats, bool copy = false)
        {
            return new ArraySlice<float>(UnmanagedArray<float>.FromArray(floats, copy));
        }

        public static ArraySlice<decimal> FromArray(decimal[] decimals, bool copy = false)
        {
            return new ArraySlice<decimal>(UnmanagedArray<decimal>.FromArray(decimals, copy));
        }
#endif

        #endregion
    }
}
