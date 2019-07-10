using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using NumSharp.Memory.Pooling;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     <see cref="ArraySlice{T}"/> is similar to <see cref="Span{T}"/> but it can be moved around without having to follow `ref struct` rules.
    /// </summary>
    /// <typeparam name="T">The type of the internal unmanaged memory</typeparam>
    public unsafe struct ArraySlice<T> : ICloneable where T : unmanaged
    {
        private static readonly InternalBufferManager.PooledBufferManager _buffer = ScalarMemoryPool.Instance;
        private readonly UnmanagedArray<T> _memoryBlock;
        private readonly T* address;
        private readonly int length;

        /// <summary>
        ///     Is this <see cref="ArraySlice{T}"/> a smaller part/slice of an unmanaged allocation?
        /// </summary>
        public readonly bool IsSlice;

        /// A Span representing this slice.
        private Span<T> _getspan
        {
            [MethodImpl((MethodImplOptions)768)] get => new Span<T>(address, length);
        }

        public ArraySlice(UnmanagedArray<T> memoryBlock)
        {
            _memoryBlock = memoryBlock;
            IsSlice = false;
            address = (T*)_memoryBlock.Address;
            length = _memoryBlock.Count;
        }

        public ArraySlice(UnmanagedArray<T> memoryBlock, Span<T> slice)
        {
            _memoryBlock = memoryBlock;
            IsSlice = true;
            length = slice.Length;
            address = (T*)Unsafe.AsPointer(ref slice.GetPinnableReference());
        }

        /// <returns></returns>
        public bool IsEmpty
        {
            [MethodImpl((MethodImplOptions)768)] get => length == 0;
        }

        /// <param name="index"></param>
        /// <returns></returns>
        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(address + index);
            [MethodImpl((MethodImplOptions)768)] set => *(address + index) = value;
        }

        public int Length
        {
            [MethodImpl((MethodImplOptions)768)] get => length;
        }

        public T* Start
        {
            [MethodImpl((MethodImplOptions)768)] get => address;
        }

        /// <param name="value"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value)
        {
            _getspan.Fill(value);
        }

        /// <param name="start"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start)
        {
            return new ArraySlice<T>(_memoryBlock, _getspan.Slice(start));
        }

        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start, int length)
        {
            return new ArraySlice<T>(_memoryBlock, _getspan.Slice(start, length));
        }

        /// <param name="destination"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination)
        {
            _getspan.CopyTo(destination);
        }

        [EditorBrowsable(EditorBrowsableState.Never)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference()
        {
            return ref Unsafe.AsRef<T>(address);
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
            return _getspan.ToArray();
        }

        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Clone()
        {
            return new ArraySlice<T>(UnmanagedArray<T>.Copy(address, length));
        }

        object ICloneable.Clone()
        {
            return Clone();
        }

        /// <summary>
        ///     Performs dispose on the internal unmanaged array.<br></br>
        /// </summary>
        /// <remarks>
        ///     Dangerous because this array might be a slice and there might be other slices who are a slice from current internal unmanaged array.<br></br>
        ///     So releasing the unmanaged memory might cause memory corruption elsewhere.
        /// </remarks>
        public void DangerousFree()
        {
            // ReSharper disable once ImpureMethodCallOnReadonlyValueField
            _memoryBlock.Free();
        }
    }
}
