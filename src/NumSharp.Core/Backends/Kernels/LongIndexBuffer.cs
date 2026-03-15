using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// A growable unmanaged buffer for collecting long indices.
    /// Replaces List&lt;long&gt; for long-indexing support (>2B elements).
    /// </summary>
    /// <remarks>
    /// Unlike List&lt;T&gt; which is limited to int.MaxValue capacity,
    /// this buffer uses unmanaged memory and supports long capacity/count.
    /// </remarks>
    public unsafe struct LongIndexBuffer : IDisposable
    {
        private long* _data;
        private long _count;
        private long _capacity;

        /// <summary>
        /// Creates a new buffer with the specified initial capacity.
        /// </summary>
        /// <param name="initialCapacity">Initial capacity (number of elements).</param>
        public LongIndexBuffer(long initialCapacity)
        {
            if (initialCapacity <= 0)
                initialCapacity = 16;

            _capacity = initialCapacity;
            _data = (long*)NativeMemory.Alloc((nuint)(_capacity * sizeof(long)));
            _count = 0;
        }

        /// <summary>
        /// Number of elements in the buffer.
        /// </summary>
        public long Count => _count;

        /// <summary>
        /// Pointer to the buffer data.
        /// </summary>
        public long* Data => _data;

        /// <summary>
        /// Adds a value to the buffer, growing if necessary.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(long value)
        {
            if (_count >= _capacity)
                Grow();
            _data[_count++] = value;
        }

        /// <summary>
        /// Gets the value at the specified index.
        /// </summary>
        public long this[long index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data[index];
        }

        /// <summary>
        /// Doubles the buffer capacity.
        /// </summary>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private void Grow()
        {
            long newCapacity = _capacity * 2;
            var newData = (long*)NativeMemory.Alloc((nuint)(newCapacity * sizeof(long)));
            Buffer.MemoryCopy(_data, newData, newCapacity * sizeof(long), _count * sizeof(long));
            NativeMemory.Free(_data);
            _data = newData;
            _capacity = newCapacity;
        }

        /// <summary>
        /// Copies the buffer contents to a new NDArray&lt;long&gt;.
        /// </summary>
        public NumSharp.Generic.NDArray<long> ToNDArray()
        {
            var result = new NumSharp.Generic.NDArray<long>(new Shape(_count));
            if (_count > 0)
            {
                Buffer.MemoryCopy(_data, (void*)result.Address, _count * sizeof(long), _count * sizeof(long));
            }
            return result;
        }

        /// <summary>
        /// Frees the unmanaged memory.
        /// </summary>
        public void Dispose()
        {
            if (_data != null)
            {
                NativeMemory.Free(_data);
                _data = null;
            }
        }
    }
}
