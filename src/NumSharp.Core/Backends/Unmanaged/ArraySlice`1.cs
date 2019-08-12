using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Utilities;
#pragma warning disable SA1121 // explicitly using type aliases instead of built-in types
#if BIT64
using nuint = System.UInt64;
#else
using nuint = System.UInt32;
#endif

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
            Count = (int)MemoryBlock.Count; //TODO! when long index, remove cast int
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
        /// <param name="start">The offset in <typeparamref name="T"/> and not bytes - relative to the <paramref name="memoryBlock"/>.</param>
        /// <param name="count">The number of <typeparamref name="T"/> this slice should contain - relative to the <paramref name="memoryBlock"/></param>
        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, T* address, int count)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = count;
            VoidAddress = Address = (T*)address;
            //TODO! we should check that is does not exceed bounds.
        }

        /// <summary>
        ///     Creates a sliced <see cref="ArraySlice{T}"/>.
        /// </summary>
        /// <param name="memoryBlock"></param>
        /// <param name="start">The offset in <typeparamref name="T"/> and not bytes - relative to the <paramref name="memoryBlock"/>.</param>
        /// <param name="count">The number of <typeparamref name="T"/> this slice should contain - relative to the <paramref name="memoryBlock"/></param>
        public ArraySlice(UnmanagedMemoryBlock<T> memoryBlock, T* address, long count)
        {
            MemoryBlock = memoryBlock;
            IsSlice = true;
            Count = (int)count; //TODO! When index long, this should not cast.
            VoidAddress = Address = address;
#if DEBUG
            if (address + count > memoryBlock.Address + memoryBlock.Count)
                throw new ArgumentOutOfRangeException(nameof(address));

            if (address + count < memoryBlock.Address || address < memoryBlock.Address)
                throw new ArgumentOutOfRangeException(nameof(address));
#endif
        }

        #endregion

        #region Accessing

        /// <param name="index"></param>
        /// <returns></returns>
        object IArraySlice.this[int index]
        {
            get
            {
                Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
                return *(Address + index);
            }
            set
            {
                Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
                *(Address + index) = (T)value;
            }
        }

        /// <param name="index"></param>
        /// <returns></returns>
        public T this[int index]
        {
            [MethodImpl((MethodImplOptions)768)]
            get
            {
                Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
                return *(Address + index);
            }
            [MethodImpl((MethodImplOptions)768)]
            set
            {
                Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
                *(Address + index) = value;
            }
        }

        [MethodImpl((MethodImplOptions)768)]
        public T GetIndex(int index)
        {
            return *(Address + index);
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, object value)
        {
            Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
            *(Address + index) = (T)value;
        }

        [MethodImpl((MethodImplOptions)768)]
        public void SetIndex(int index, T value)
        {
            Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
            *(Address + index) = value;
        }

        #endregion

        [MethodImpl((MethodImplOptions)768)]
        public bool Contains(T item)
        {
            bool equals = false;
            var addr = Address;
            Parallel.For(0L, Count, (i, state) =>
            {
                if ((*(addr + i)).Equals(item))
                {
                    equals = true;
                    state.Stop();
                }
            });

            return equals;
        }

        /// <param name="value"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void Fill(T value)
        {
            if (Unsafe.SizeOf<T>() == 1)
            {
                uint length = (uint)Count;
                if (length == 0)
                    return;

                T tmp = value; // Avoid taking address of the "value" argument. It would regress performance of the loop below.
                Unsafe.InitBlockUnaligned(Address, Unsafe.As<T, byte>(ref tmp), length);
            }
            else
            {
                // Do all math as nuint to avoid unnecessary 64->32->64 bit integer truncations
                nuint length = (uint)Count;
                if (length == 0)
                    return;

                T* addr = Address;

                // TODO: Create block fill for value types of power of two sizes e.g. 2,4,8,16

                nuint i = 0;
                for (; i < (length & ~(nuint)7); i += 8)
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

                if (i < (length & ~(nuint)3))
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

        /// <param name="start"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start)
        {
            if ((uint)start > (uint)Count)
                throw new ArgumentOutOfRangeException(nameof(start));

            return new ArraySlice<T>(MemoryBlock, Address + start, Count - start);
        }

        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Slice(int start, int length)
        {
#if BIT64
            // Since start and length are both 32-bit, their sum can be computed across a 64-bit domain
            // without loss of fidelity. The cast to uint before the cast to ulong ensures that the
            // extension from 32- to 64-bit is zero-extending rather than sign-extending. The end result
            // of this is that if either input is negative or if the input sum overflows past Int32.MaxValue,
            // that information is captured correctly in the comparison against the backing _length field.
            // We don't use this same mechanism in a 32-bit process due to the overhead of 64-bit arithmetic.
            if ((ulong)(uint)start + (ulong)(uint)length > (ulong)(uint)Count)
                throw new ArgumentOutOfRangeException(nameof(length));
#else
            if ((uint)start > (uint)Count || (uint)length > (uint)(Count - start))
                throw new ArgumentOutOfRangeException(nameof(length));
#endif
            return new ArraySlice<T>(MemoryBlock, Address + start, length);
        }


        public bool TryCopyTo(Span<T> destination)
        {
            if ((uint)Count > (uint)destination.Length)
                return false;

            Buffer.MemoryCopy(Unsafe.AsPointer(ref destination.GetPinnableReference()), Address, destination.Length * ItemLength, Count * ItemLength);
            return true;
        }

        /// <param name="destination"></param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination)
        {
            if ((uint)Count <= (uint)destination.Length)
            {
                Buffer.MemoryCopy(Unsafe.AsPointer(ref destination.GetPinnableReference()), Address, destination.Length * ItemLength, Count * ItemLength);
            }
            else
            {
                throw new ArgumentException("Destinition was too short.");
            }
        }

        /// <summary>
        ///     Copies the entire array to address.
        /// </summary>
        /// <param name="dst">The address to copy to</param>
        /// <remarks>The destiniton has to be atleast the size of this array, otherwise memory corruption is likely to occur.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(IntPtr dst)
        {
            // Using "if (!TryCopyTo(...))" results in two branches: one for the length
            // check, and one for the result of TryCopyTo. Since these checks are equivalent,
            // we can optimize by performing the check once ourselves then calling Memmove directly.

            Buffer.MemoryCopy((void*)dst, Address, Count, Count);
        }

        /// <summary>
        ///     Copies the entire array to address.
        /// </summary>
        /// <param name="dst">The address to copy to</param>
        /// <remarks>The destiniton has to be atleast the size of this array, otherwise memory corruption is likely to occur.</remarks>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(IntPtr dst, int sourceOffset, int sourceCount)
        {
            // Using "if (!TryCopyTo(...))" results in two branches: one for the length
            // check, and one for the result of TryCopyTo. Since these checks are equivalent,
            // we can optimize by performing the check once ourselves then calling Memmove directly.
            var len = Count * ItemLength;
            Buffer.MemoryCopy((void*)dst, Address, len, len);
        }

        /// <param name="destination"></param>
        /// <param name="sourceOffset">offset of source via count (not bytes)</param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination, int sourceOffset)
        {
            CopyTo(destination, sourceOffset, Count - sourceOffset);
        }

        /// <param name="destination"></param>
        /// <param name="sourceOffset">offset of source via count (not bytes)</param>
        /// <param name="sourceLength">How many items to copy</param>
        [MethodImpl((MethodImplOptions)768)]
        public void CopyTo(Span<T> destination, int sourceOffset, int sourceLength)
        {
            CopyTo(destination, sourceOffset, sourceLength);
        }

        [EditorBrowsable(EditorBrowsableState.Never)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference() => ref Unsafe.AsRef<T>(Address);


        [MethodImpl((MethodImplOptions)768)]
        public ArraySlice<T> Clone() => new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));


        #region Explicit Interfaces

        /// A Span representing this slice.
        /// <remarks>Does not perform copy.</remarks>
        Span<T1> IArraySlice.AsSpan<T1>()
        {
            return new Span<T1>(VoidAddress, Count);
        }

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
            Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
            *((TVal*)VoidAddress + index) = value;
        }

        [MethodImpl((MethodImplOptions)768)]
        object IArraySlice.GetIndex(int index)
        {
            Debug.Assert(index < Count, "index < Count, Memory corruption expected.");
            return *(Address + index);
        }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        T* IMemoryBlock<T>.Address => Address;

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
        void IArraySlice.CopyTo<T1>(Span<T1> destination)
        {
            this.CopyTo(Unsafe.AsPointer(ref destination.GetPinnableReference()));
        }

        /// <summary>
        ///     Gets pinnable reference of the first item in the memory block storage.
        /// </summary>
        ref T1 IArraySlice.GetPinnableReference<T1>() => ref Unsafe.AsRef<T1>(VoidAddress);

        [MethodImpl((MethodImplOptions)768)]
        IArraySlice IArraySlice.Clone() => new ArraySlice<T>(UnmanagedMemoryBlock<T>.Copy(Address, Count));

        ArraySlice<T1> IArraySlice.Clone<T1>() => new ArraySlice<T1>(UnmanagedMemoryBlock<T1>.Copy(Address, Count));

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
        long IMemoryBlock.Count
        {
            [MethodImpl((MethodImplOptions)768)] get => Count;
        }

        /// <summary>
        ///     The items with length of <see cref="IMemoryBlock.TypeCode"/> are present in <see cref="IMemoryBlock.Address"/>.
        /// </summary>
        /// <remarks>Calculated by <see cref="IMemoryBlock.Count"/>*<see cref="IMemoryBlock.ItemLength"/></remarks>
        long IMemoryBlock.BytesLength
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

        /// <summary>
        ///     Copies this <see cref="IArraySlice"/> contents into a new array.
        /// </summary>
        /// <returns></returns>
        Array IArraySlice.ToArray() => ToArray();

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
        /// Copies the contents of this span into a new array.  This heap
        /// allocates, so should generally be avoided, however it is sometimes
        /// necessary to bridge the gap with APIs written in terms of arrays.
        /// </summary>
        [MethodImpl((MethodImplOptions)768)]
        public T[] ToArray()
        {
            if (Count == 0)
                return Array.Empty<T>();

            var destination = new T[Count];
            var len = Count * ItemLength;
            fixed (T* dst = destination)
                Buffer.MemoryCopy(Address, dst, len, len);
            return destination;
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
