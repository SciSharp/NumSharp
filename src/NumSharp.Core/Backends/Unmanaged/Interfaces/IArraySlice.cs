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

        /// <summary>
        ///     Reads a strided run of elements as their C# values — the run form of <see cref="GetIndex(long)"/>: element
        ///     <c>start + i * stride</c> into <c>destination[i]</c> for every <c>i</c> of <paramref name="destination"/>.
        ///     <c>ndarray.tolist()</c> reads each innermost row of its nested lists with one call.
        /// </summary>
        /// <param name="start">The first element, as an index into this slice (what <see cref="GetIndex(long)"/> takes).</param>
        /// <param name="stride">The step between consecutive elements, in elements: 0 reads one element repeatedly (a
        ///     broadcast axis), a negative step walks backwards (a reversed view).</param>
        /// <param name="destination">Receives the values; its length is the number of elements read.</param>
        /// <exception cref="SystemError">A <c>U</c> element holds a unit above <c>U+10FFFF</c> (CPython's <c>invalid maximum
        ///     character passed to PyUnicode_New</c>). Raised at the first such element: the slots before it are filled, the
        ///     rest are not.</exception>
        /// <exception cref="OutOfMemoryException">A <c>StringDType</c> element whose cell is corrupt (NumPy's <c>Failed to
        ///     load string in StringDType getitem</c>) or whose text is too long for a .NET string; raised as above.</exception>
        /// <remarks>
        ///     Every implementation gives exactly what <see cref="GetIndex(long)"/> gives for each element — the same boxed
        ///     type, the same text, a fresh object per element, the same exception at the same element (a slice outside the
        ///     library raises what its own <see cref="GetIndex(long)"/> raises) — and reads nothing outside the run's
        ///     elements. The caller
        ///     guarantees that every element of the run is inside the slice; nothing here checks it (an index outside reads
        ///     other memory, as <see cref="GetIndex(long)"/> would). This default calls <see cref="GetIndex(long)"/> per
        ///     element; the library's slices read the run in one pass: a numeric slice boxes straight from its memory, a
        ///     fixed-width string slice decodes each element without the per-call dispatch, and a <c>StringDType</c> slice
        ///     takes its allocator's lock once for the run instead of once per element (NumPy's <c>getitem</c> locks per
        ///     element; the lock guards the same reads).
        /// </remarks>
        void GetItems(long start, long stride, Span<object> destination)
        {
            for (int i = 0; i < destination.Length; i++)
                destination[i] = GetIndex(start + i * stride);
        }

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
        ///     Atomically decrement the reference count WITHOUT freeing the buffer
        ///     when the count reaches zero — the finalizer-path counterpart of
        ///     <see cref="Release"/>. A finalizer proves only that one wrapper became
        ///     unreachable, not that no other reachable alias (a bare storage/slice)
        ///     still reads the buffer; the block's own finalizer frees it once the
        ///     whole alias graph is unreachable.
        /// </summary>
        void Abandon();

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
