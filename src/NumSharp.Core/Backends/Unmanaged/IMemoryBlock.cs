using System;

namespace NumSharp.Backends.Unmanaged
{
    public interface IMemoryBlock
    {
        /// <summary>
        ///     The size of a single item stored in <see cref="Address"/>.
        /// </summary>
        int ItemLength { get; }

        /// <summary>
        ///     The start address of this memory block.
        /// </summary>
        unsafe void* Address { get; }

        /// <summary>
        ///     How many items are stored in <see cref="Address"/>?
        /// </summary>
        /// <remarks></remarks>
        int Count { get; }

        /// <summary>
        ///     The items with length of <see cref="TypeCode"/> are present in <see cref="Address"/>.
        /// </summary>
        /// <remarks>Calculated by <see cref="Count"/>*<see cref="ItemLength"/></remarks>
        int BytesLength { get; }

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of the type stored inside this memory block.
        /// </summary>
        NPTypeCode TypeCode { get; }

    }
}
