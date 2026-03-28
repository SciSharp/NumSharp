using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// A span-like structure over unmanaged memory that supports long indexing.
    /// Unlike <see cref="Span{T}"/>, this can represent more than int.MaxValue elements.
    /// </summary>
    /// <typeparam name="T">The unmanaged element type.</typeparam>
    public readonly unsafe struct UnmanagedSpan<T> : IEnumerable<T> where T : unmanaged
    {
        private readonly T* _pointer;
        private readonly long _length;

        /// <summary>
        /// Creates an UnmanagedSpan from a pointer and length.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedSpan(T* pointer, long length)
        {
            _pointer = pointer;
            _length = length;
        }

        /// <summary>
        /// Creates an UnmanagedSpan from a void pointer and length.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedSpan(void* pointer, long length)
        {
            _pointer = (T*)pointer;
            _length = length;
        }

        /// <summary>
        /// The number of elements in the span.
        /// </summary>
        public long Length => _length;

        /// <summary>
        /// Returns true if Length is 0.
        /// </summary>
        public bool IsEmpty => _length == 0;

        /// <summary>
        /// Gets a pointer to the first element.
        /// </summary>
        public T* Pointer => _pointer;

        /// <summary>
        /// Gets or sets the element at the specified index.
        /// </summary>
        public ref T this[long index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if DEBUG
                if ((ulong)index >= (ulong)_length)
                    throw new IndexOutOfRangeException($"Index {index} is out of range for span of length {_length}");
#endif
                return ref _pointer[index];
            }
        }

        /// <summary>
        /// Gets a reference to the first element.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref T GetPinnableReference()
        {
            if (_length == 0)
                throw new InvalidOperationException("Cannot get pinnable reference of empty span");
            return ref _pointer[0];
        }

        /// <summary>
        /// Forms a slice out of the current span starting at the specified index.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedSpan<T> Slice(long start)
        {
            if ((ulong)start > (ulong)_length)
                throw new ArgumentOutOfRangeException(nameof(start));
            return new UnmanagedSpan<T>(_pointer + start, _length - start);
        }

        /// <summary>
        /// Forms a slice out of the current span starting at the specified index for the specified length.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UnmanagedSpan<T> Slice(long start, long length)
        {
            if ((ulong)start > (ulong)_length || (ulong)length > (ulong)(_length - start))
                throw new ArgumentOutOfRangeException();
            return new UnmanagedSpan<T>(_pointer + start, length);
        }

        /// <summary>
        /// Copies the contents of this span to a destination span.
        /// </summary>
        public void CopyTo(UnmanagedSpan<T> destination)
        {
            if (_length > destination._length)
                throw new ArgumentException("Destination span is too short.");

            Buffer.MemoryCopy(_pointer, destination._pointer, destination._length * sizeof(T), _length * sizeof(T));
        }

        /// <summary>
        /// Copies the contents of this span to a Span (only if length fits in int).
        /// </summary>
        public void CopyTo(Span<T> destination)
        {
            if (_length > int.MaxValue)
                throw new InvalidOperationException($"Cannot copy {_length} elements to Span<T> (exceeds int.MaxValue)");
            if (_length > destination.Length)
                throw new ArgumentException("Destination span is too short.");

            new Span<T>(_pointer, (int)_length).CopyTo(destination);
        }

        /// <summary>
        /// Fills the span with the specified value.
        /// </summary>
        public void Fill(T value)
        {
            for (long i = 0; i < _length; i++)
                _pointer[i] = value;
        }

        /// <summary>
        /// Clears the span (fills with default value).
        /// </summary>
        public void Clear()
        {
            var size = _length * sizeof(T);
            // Use Buffer.MemoryCopy with zeroed source or iterate
            for (long i = 0; i < _length; i++)
                _pointer[i] = default;
        }

        /// <summary>
        /// Creates a Span from this UnmanagedSpan if the length fits in an int.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown if length exceeds int.MaxValue.</exception>
        public Span<T> AsSpan()
        {
            if (_length > int.MaxValue)
                throw new InvalidOperationException($"Cannot create Span<T> for {_length} elements (exceeds int.MaxValue)");
            return new Span<T>(_pointer, (int)_length);
        }

        /// <summary>
        /// Tries to create a Span from this UnmanagedSpan.
        /// </summary>
        /// <returns>True if successful, false if length exceeds int.MaxValue.</returns>
        public bool TryAsSpan(out Span<T> span)
        {
            if (_length > int.MaxValue)
            {
                span = default;
                return false;
            }
            span = new Span<T>(_pointer, (int)_length);
            return true;
        }

        /// <summary>
        /// Copies the contents to a new array.
        /// </summary>
        public T[] ToArray()
        {
            if (_length > Array.MaxLength)
                throw new InvalidOperationException($"Cannot create array of {_length} elements (exceeds Array.MaxLength)");

            var array = new T[_length];
            fixed (T* dest = array)
            {
                Buffer.MemoryCopy(_pointer, dest, _length * sizeof(T), _length * sizeof(T));
            }
            return array;
        }

        /// <summary>
        /// Returns an enumerator for this span.
        /// </summary>
        public Enumerator GetEnumerator() => new Enumerator(this);

        IEnumerator<T> IEnumerable<T>.GetEnumerator() => new EnumeratorObject(this);
        IEnumerator IEnumerable.GetEnumerator() => new EnumeratorObject(this);

        /// <summary>
        /// Enumerator for UnmanagedSpan.
        /// </summary>
        public ref struct Enumerator
        {
            private readonly T* _pointer;
            private readonly long _length;
            private long _index;

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal Enumerator(UnmanagedSpan<T> span)
            {
                _pointer = span._pointer;
                _length = span._length;
                _index = -1;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool MoveNext()
            {
                long next = _index + 1;
                if (next < _length)
                {
                    _index = next;
                    return true;
                }
                return false;
            }

            public ref T Current
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => ref _pointer[_index];
            }
        }

        /// <summary>
        /// Boxed enumerator for IEnumerable interface.
        /// </summary>
        private sealed class EnumeratorObject : IEnumerator<T>
        {
            private readonly T* _pointer;
            private readonly long _length;
            private long _index;

            internal EnumeratorObject(UnmanagedSpan<T> span)
            {
                _pointer = span._pointer;
                _length = span._length;
                _index = -1;
            }

            public bool MoveNext()
            {
                long next = _index + 1;
                if (next < _length)
                {
                    _index = next;
                    return true;
                }
                return false;
            }

            public T Current => _pointer[_index];
            object IEnumerator.Current => Current;

            public void Reset() => _index = -1;
            public void Dispose() { }
        }

        /// <summary>
        /// Creates an empty UnmanagedSpan.
        /// </summary>
        public static UnmanagedSpan<T> Empty => default;
    }
}
