using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Memory.Pooling;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     <see cref="ArraySlice{T}"/> is similar to <see cref="Span{T}"/> but it can be moved around without having to follow `ref struct` rules.
    /// </summary>
    /// <typeparam name="T">The type that the <see cref="MemoryBlock"/> implements.</typeparam>
    public readonly unsafe struct ArraySlice<T> : IArraySlice, IMemoryBlock<T>, IEnumerable<T> where T : unmanaged
    {
        public static NPTypeCode TypeCode { get; } = InfoOf<T>.NPTypeCode;

        /// <summary>
        ///     The memory block this <see cref="ArraySlice{T}"/> is stored in.
        /// </summary>
        /// <remarks>If <see cref="IsSlice"/> is false then this slice represents the entire MemoryBlock.</remarks>
        public readonly UnmanagedMemoryBlock<T> MemoryBlock;

        public readonly T* Address;
        public readonly void* VoidAddress;

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

        public int ItemLength
        {
            [MethodImpl((MethodImplOptions)768)] get => InfoOf<T>.Size;
        }

        #region Construction

        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock)
        {
            MemoryBlock = memoryBlock;
            IsSlice = false;
            VoidAddress = Address = MemoryBlock.Address;
            Count = MemoryBlock.Count;
        }

        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, Span<T> slice)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = slice.Length;
            VoidAddress = Address = (T*)Unsafe.AsPointer(ref slice.GetPinnableReference());
        }

        /// <summary>
        ///     Creates a sliced <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="memoryBlock"></param>
        /// <param name="offset">The offset in <typeparamref name="T"/> and not bytes - relative to the <paramref name="memoryBlock"/>.</param>
        /// <param name="count">The number of <typeparamref name="T"/> this slice should contain - relative to the <paramref name="memoryBlock"/></param>
        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, int offset, int count)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = count;
            VoidAddress = Address = (memoryBlock.Address + offset);
            //TODO! we should check that is does not exceed bounds.
        }

        #endregion

        #region Accessing

        /// A Span representing this slice.
        /// <remarks>Does not perform copy.</remarks>
        Span<T1> IArraySlice.AsSpan<T1>()
        {
            return new Span<T1>(VoidAddress, Count);
        }

        /// <param name="index"></param>
        /// <returns></returns>
        object IArraySlice.this[int index]
        {
            get => *(Address + index);
            set => *(Address + index) = (T)value;
        }

        /// <param name="index"></param>
        /// <returns></returns>
        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)] get => *(Address + index);
            [MethodImpl((MethodImplOptions)768)] set => *(Address + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetIndex(int index)
        {
            return *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, object value)
        {
            *(Address + index) = (T)value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, T value)
        {
            *(Address + index) = value;
        }

        #region Implicit Interface

        [MethodImpl((MethodImplOptions)768)]
        TRet IArraySlice.GetIndex<TRet>(int index)
        {
            Debug.Assert(InfoOf<TRet>.Size == InfoOf<T>.Size);
            return *((TRet*)VoidAddress + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        void IArraySlice.SetIndex<TVal>(int index, TVal value)
        {
            Debug.Assert(InfoOf<TVal>.Size == InfoOf<T>.Size);
            *((TVal*)VoidAddress + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        object IArraySlice.GetIndex(int index)
        {
            return *(Address + index);
        }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        T* IMemoryBlock<T>.Address => Address;

        #endregion

        #endregion

        [MethodImpl((MethodImplOptions)768)]
        public bool Contains(T item)
        {
            int len = Count;
            for (var i = 0; i < len; i++) //TODO! Parallel.For?
            {
                if ((*(Address + i)).Equals(item)) return true;
            }

            return false;
        }

        /// <param name="value"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value) => AsSpan.Fill(value);

        /// <summary>
        ///     Fills all indexes with <paramref name="value"/>.
        /// </summary>
        /// <param name="value"></param>
        void IArraySlice.Fill(object value) => Fill((T)value);

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice IArraySlice.Slice(int start) => Slice(start);

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <param name="count">The number of items to slice (not bytes)</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice IArraySlice.Slice(int start, int count) => Slice(start, count);

        /// <param name="destination"></param>
        void IArraySlice.CopyTo<T1>(Span<T1> destination) => new Span<T1>(Address, Count).CopyTo(destination);

        /// <summary>
        ///     Gets pinnable reference of the first item in the memory block storage.
        /// </summary>
        ref T1 IArraySlice.GetPinnableReference<T1>() => ref Unsafe.AsRef<T1>(VoidAddress);

        /// <param name="start"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start) => new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start));

        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start, int length) => new ArraySlice<T>(MemoryBlock, AsSpan.Slice(start, length));


        /// <param name="destination"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination) => AsSpan.CopyTo(destination);

        /// <summary>
        ///     Copies the entire array to address.
        /// </summary>
        /// <param name="dst">The address to copy to</param>
        /// <remarks>The destiniton has to be atleast the size of this array, otherwise memory corruption is likely to occur.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(IntPtr dst) => AsSpan.CopyTo(new Span<T>(dst.ToPointer(), Count * InfoOf<T>.Size));

        /// <param name="destination"></param>
        /// <param name="sourceOffset">offset of source via count (not bytes)</param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination, int sourceOffset)
        {
            new Span<T>(Address + sourceOffset, Count - sourceOffset).CopyTo(destination);
            AsSpan.CopyTo(destination);
        }

        /// <param name="destination"></param>
        /// <param name="sourceOffset">offset of source via count (not bytes)</param>
        /// <param name="sourceLength">How many items to copy</param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination, int sourceOffset, int sourceLength)
        {
            new Span<T>(Address + sourceOffset, sourceLength).CopyTo(destination);
            AsSpan.CopyTo(destination);
        }

        [EditorBrowsable(EditorBrowsableState.Never)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference() => ref Unsafe.AsRef<T>(Address);

        ArraySlice<T1> IArraySlice.Clone<T1>() => new ArraySlice<T1>(UnmanagedMemoryBlock<T1>.Copy(Address, Count));

        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Clone() => new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));

        [MethodImpl((MethodImplOptions)768)]
        IArraySlice IArraySlice.Clone() => new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));

        #region Explicit Interfaces

        object ICloneable.Clone() => new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe void* IMemoryBlock.Address
        {
            [MethodImpl((MethodImplOptions)768)] get => VoidAddress;
        }

        /// <summary>
        ///     How many items are stored in <see cref="IMemoryBlock.Address"/>?
        /// </summary>
        /// <remarks></remarks>
        int IMemoryBlock.Count
        {
            [MethodImpl((MethodImplOptions)768)] get => Count;
        }

        /// <summary>
        ///     The items with length of <see cref="IMemoryBlock.TypeCode"/> are present in <see cref="IMemoryBlock.Address"/>.
        /// </summary>
        /// <remarks>Calculated by <see cref="IMemoryBlock.Count"/>*<see cref="IMemoryBlock.ItemLength"/></remarks>
        int IMemoryBlock.BytesLength
        {
            [MethodImpl((MethodImplOptions)768)] get => Count * InfoOf<T>.Size;
        }

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of the type stored inside this memory block.
        /// </summary>
        NPTypeCode IMemoryBlock.TypeCode
        {
            [MethodImpl((MethodImplOptions)768)] get => TypeCode;
        }

        IMemoryBlock IArraySlice.MemoryBlock
        {
            [MethodImpl((MethodImplOptions)768)] get => MemoryBlock;
        }

        #endregion

        /// <summary>
        ///     Performs dispose on the internal unmanaged memory block.<br></br>
        /// </summary>
        /// <remarks>
        ///     Dangerous because this <see cref="ArraySlice"/> might be a <see cref="IsSlice"/> therefore there might be other slices that point to current <see cref="MemoryBlock"/>.<br></br>
        ///     So releasing the <see cref="MemoryBlock"/> might cause memory corruption elsewhere.<br></br>
        ///     It is best to leave MemoryBlock to GC.
        /// </remarks>
        public void DangerousFree()
        {
            // ReSharper disable once ImpureMethodCallOnReadonlyValueField
            MemoryBlock.Free();
        }

        /// <summary>
        ///     Copies this <see cref="IArraySlice"/> contents into a new array.
        /// </summary>
        /// <returns></returns>
        Array IArraySlice.ToArray()
        {
            return ToArray();
        }

        [MethodImpl((MethodImplOptions)768)]
        public T[] ToArray()
        {
            return AsSpan.ToArray();
        }

        #region Allocation

        /// <summary>
        ///     Allocate an array filled filled with <paramref name="fill"/>.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <param name="fill">The item to fill the newly allocated memory with.</param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate(int count, T fill)
            => new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count, fill));

        /// <summary>
        ///     Allocate an array filled with default value of <typeparamref name="T"/>.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <param name="fillDefault">Should the newly allocated memory be filled with the default of <typeparamref name="T"/></param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate(int count, bool fillDefault)
            => !fillDefault ? Allocate(count) : new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count, default(T)));

        /// <summary>
        ///     Allocate an array filled with noisy memory.
        /// </summary>
        /// <param name="count">How many items this array will have (aka Count).</param>
        /// <returns>A newly allocated array.</returns>
        public static ArraySlice<T> Allocate(int count)
            => new ArraySlice<T>(new UnmanagedMemoryBlock<T>(count));

        #endregion

        #region Enumerators

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            return _enumerate().GetEnumerator();
        }

        private IEnumerable<T> _enumerate()
        {
            var len = this.Count;
            for (int i = 0; i < len; i++)
            {
                yield return this[i];
            }
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator"></see> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        #endregion
    }
}
