using System;
using System.Runtime.CompilerServices;
using NumSharp.Generic;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// A growable buffer for collecting long indices, backed by NDArray storage.
    /// Replaces LongIndexBuffer - uses NumSharp's existing unmanaged memory infrastructure.
    /// </summary>
    /// <remarks>
    /// Benefits over LongIndexBuffer:
    /// - Uses NDArray's proven unmanaged storage
    /// - No separate Dispose needed (NDArray handles cleanup)
    /// - ToResult() returns the storage directly (no copy for final result)
    /// - Integrates with NumSharp's memory management
    ///
    /// Growth strategy:
    /// - Doubles capacity below 1 billion elements
    /// - 33% growth above 1 billion (matches Hashset/LongList pattern)
    /// </remarks>
    public unsafe struct IndexCollector
    {
        private const long LargeGrowthThreshold = 1_000_000_000L;

        private NDArray<long> _storage;
        private long _count;
        private long _capacity;

        /// <summary>
        /// Creates a new collector with the specified initial capacity.
        /// </summary>
        /// <param name="initialCapacity">Initial capacity (number of elements).</param>
        public IndexCollector(long initialCapacity)
        {
            if (initialCapacity <= 0)
                initialCapacity = 16;

            _capacity = initialCapacity;
            _storage = new NDArray<long>(new Shape(initialCapacity));
            _count = 0;
        }

        /// <summary>
        /// Number of elements collected.
        /// </summary>
        public long Count => _count;

        /// <summary>
        /// Pointer to the underlying data.
        /// </summary>
        public long* Data => (long*)_storage.Address;

        /// <summary>
        /// Adds a value, growing the storage if necessary.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(long value)
        {
            if (_count >= _capacity)
                Grow();
            ((long*)_storage.Address)[_count++] = value;
        }

        /// <summary>
        /// Gets the value at the specified index.
        /// </summary>
        public long this[long index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ((long*)_storage.Address)[index];
        }

        /// <summary>
        /// Grows the storage capacity.
        /// Uses 2x growth below 1B elements, 33% growth above.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Grow()
        {
            long newCapacity = _capacity < LargeGrowthThreshold
                ? _capacity * 2
                : _capacity + (_capacity / 3);

            var newStorage = new NDArray<long>(new Shape(newCapacity));

            // Copy existing data
            Buffer.MemoryCopy(
                _storage.Address,
                newStorage.Address,
                newCapacity * sizeof(long),
                _count * sizeof(long));

            _storage = newStorage;
            _capacity = newCapacity;
        }

        /// <summary>
        /// Returns the collected indices as an NDArray, trimmed to actual count.
        /// </summary>
        /// <remarks>
        /// If count equals capacity, returns storage directly (no copy).
        /// Otherwise, creates a properly-sized copy.
        /// </remarks>
        public NDArray<long> ToResult()
        {
            if (_count == 0)
                return new NDArray<long>(0);

            if (_count == _capacity)
            {
                // Perfect fit - return storage directly
                return _storage;
            }

            // Need to trim - create properly sized result
            var result = new NDArray<long>(new Shape(_count));
            Buffer.MemoryCopy(
                _storage.Address,
                result.Address,
                _count * sizeof(long),
                _count * sizeof(long));
            return result;
        }
    }
}
