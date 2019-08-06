using System;
using System.Collections;

namespace NumSharp.Backends.Unmanaged
{
    public interface IArraySlice : IMemoryBlock, ICloneable, IEnumerable
    {
        IMemoryBlock MemoryBlock { get; }

        T GetIndex<T>(int index) where T : unmanaged;

        object GetIndex(int index);

        void SetIndex<T>(int index, T value) where T : unmanaged;

        void SetIndex(int index, object value);

        new IArraySlice Clone();

        /// A Span representing this slice.
        /// <remarks>Does not perform copy.</remarks>
        Span<T> AsSpan<T>();

        /// <param name="index"></param>
        /// <returns></returns>
        unsafe object this[int index] {  get;  set; }

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
        IArraySlice Slice(int start);

        /// <summary>
        ///     Perform a slicing on this <see cref="IMemoryBlock"/> without copying data.
        /// </summary>
        /// <param name="start">The index to start from</param>
        /// <param name="count">The number of items to slice (not bytes)</param>
        /// <remarks>Creates a slice without copying.</remarks>
        IArraySlice Slice(int start, int count);

        /// <param name="destination"></param>
        void CopyTo<T>(Span<T> destination);

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
    }
}
