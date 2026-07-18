using System;
using System.Collections;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    public interface IArraySlice : IMemoryBlock, ICloneable, IEnumerable
    {
        IMemoryBlock MemoryBlock { get; }

        T GetIndex<T>(long index) where T : unmanaged;

        object GetIndex(long index);

        void SetIndex<T>(long index, T value) where T : unmanaged;

        void SetIndex(long index, object value);

        new IArraySlice Clone();

        /// <summary>
        /// Returns an UnmanagedSpan representing this slice's memory.
        /// </summary>
        /// <remarks>Does not perform copy. Supports long indexing for arrays &gt; 2B elements.</remarks>
        unsafe UnmanagedSpan<T> AsSpan<T>() where T : unmanaged;

        /// <param name="index"></param>
        /// <returns></returns>
        unsafe object this[long index] {  get;  set; }

        /// <summary>
        ///     Fills all indexes with <paramref name="value"/>.
        /// </summary>
        /// <param name="value"></param>
        void Fill(object value);

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice Slice(long start);

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <param name="count">The number of items to slice (not bytes)</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice Slice(long start, long count);

        /// <param name="destination"></param>
        void CopyTo<T>(Span<T> destination);

        /// <summary>
        /// Copies this slice's contents to an UnmanagedSpan destination.
        /// Supports long indexing for arrays &gt; 2B elements.
        /// </summary>
        /// <param name="destination"></param>
        void CopyTo<T>(UnmanagedSpan<T> destination) where T : unmanaged;

        /// <summary>
        ///     Gets pinnable reference of the first item in the memory block storage.
        /// </summary>
        ref T GetPinnableReference<T>() where T : unmanaged;

        ArraySlice<T> Clone<T>() where T : unmanaged;

        /// <summary>
        ///     Performs dispose on the internal unmanaged memory block.<br></br>
        /// </summary>
        /// <remarks>
        ///     Dangerous because this <see cref="ArraySlice"/> might be a <see cref="IsSlice"/> therefore there might be other slices that point to current <see cref="MemoryBlock"/>.<br></br>
        ///     So releasing the <see cref="MemoryBlock"/> might cause memory corruption elsewhere.<br></br>
        ///     It is best to leave MemoryBlock to GC.
        /// </remarks>
        void DangerousFree();

        /// <summary>
        ///     Copies this <see cref="IArraySlice"/> contents into a new array.
        /// </summary>
        /// <returns></returns>
        Array ToArray();

        /// <summary>
        ///     Atomically increment the reference count on the underlying
        ///     <see cref="MemoryBlock"/>. Returns <c>false</c> if the block
        ///     has already been released and must not be referenced further.
        /// </summary>
        bool TryAddRef();

        /// <summary>
        ///     Atomically decrement the reference count. Frees the underlying
        ///     buffer immediately on the final release.
        /// </summary>
        void Release();

        /// <summary>
        ///     Diagnostic: <c>true</c> once the underlying buffer has been freed.
        /// </summary>
        bool IsReleased { get; }

        /// <summary>
        ///     <c>true</c> when the underlying <see cref="MemoryBlock"/> is held by
        ///     at most one logical reference (this owner), i.e. no other
        ///     <see cref="NDArray"/> / view shares its buffer. Non-owning wraps
        ///     (external / pinned memory) report <c>true</c> since their refcount is
        ///     immortal and meaningless. Used by <c>ndarray.resize</c>'s refcheck to
        ///     mirror NumPy's "references or is referenced by another array" guard.
        /// </summary>
        bool IsUniquelyReferenced { get; }
    }
}
