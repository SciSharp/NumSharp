namespace NumSharp.Backends.Unmanaged
{
    public interface IMemoryBlock
    {
        /// <summary>
        ///     The size of a single item stored in <see cref="Address"/>.
        /// </summary>
        /// <remarks>Equivalent to <see cref="NPTypeCode.SizeOf"/> extension.</remarks>
        int ItemLength { get; }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe void* Address { get; }

        /// <summary>
        ///     How many items are stored in <see cref="Address"/>.
        /// </summary>
        /// <remarks>Not to confuse with <see cref="BytesLength"/></remarks>
        long Count { get; }

        /// <summary>
        ///     How many bytes are stored in this memory block.
        /// </summary>
        /// <remarks>Calculated by <see cref="Count"/>*<see cref="ItemLength"/></remarks>
        long BytesLength { get; }

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of the type stored inside this memory block.
        /// </summary>
        NPTypeCode TypeCode { get; }
    }

    public interface IMemoryBlock<T> : IMemoryBlock where T : unmanaged
    {
        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        new unsafe T* Address { get; }
    }
}
