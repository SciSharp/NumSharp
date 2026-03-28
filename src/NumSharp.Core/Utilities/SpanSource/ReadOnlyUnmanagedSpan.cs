using System;
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;
using System.Runtime.Versioning;

using EditorBrowsableAttribute = System.ComponentModel.EditorBrowsableAttribute;
using EditorBrowsableState = System.ComponentModel.EditorBrowsableState;

#pragma warning disable 0809 // Obsolete member 'UnmanagedSpan<T>.Equals(object)' overrides non-obsolete member 'object.Equals(object)'

namespace NumSharp.Utilities
{
    /// <summary>
    /// ReadOnlyUnmanagedSpan represents a contiguous region of arbitrary memory. Unlike arrays, it can point to either managed
    /// or native memory, or to memory allocated on the stack. It is type-safe and memory-safe.
    /// </summary>
    [DebuggerTypeProxy(typeof(UnmanagedSpanDebugView<>))]
    [DebuggerDisplay("{ToString(),raw}")]
    public readonly ref struct ReadOnlyUnmanagedSpan<T> where T : unmanaged
    {
        /// <summary>A byref or a native ptr.</summary>
        internal readonly ref T _reference;
        /// <summary>The number of elements this ReadOnlyUnmanagedSpan contains.</summary>
        private readonly long _length;

        /// <summary>
        /// Creates a new read-only span over the entirety of the target array.
        /// </summary>
        /// <param name="array">The target array.</param>
        /// <remarks>Returns default when <paramref name="array"/> is null.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlyUnmanagedSpan(T[]? array)
        {
            if (array == null)
            {
                this = default;
                return; // returns default
            }

            _reference = ref MemoryMarshal.GetArrayDataReference(array);
            _length = array.Length;
        }

        /// <summary>
        /// Creates a new read-only span over the portion of the target array beginning
        /// at 'start' index and ending at 'end' index (exclusive).
        /// </summary>
        /// <param name="array">The target array.</param>
        /// <param name="start">The zero-based index at which to begin the read-only span.</param>
        /// <param name="length">The number of items in the read-only span.</param>
        /// <remarks>Returns default when <paramref name="array"/> is null.</remarks>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the specified <paramref name="start"/> or end index is not in the range (&lt;0 or &gt;Length).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlyUnmanagedSpan(T[]? array, int start, int length)
        {
            if (array == null)
            {
                if (start != 0 || length != 0)
                    ThrowHelper.ThrowArgumentOutOfRangeException();
                this = default;
                return; // returns default
            }
#if TARGET_64BIT
            // See comment in UnmanagedSpan<T>.Slice for how this works.
            if ((ulong)(uint)start + (ulong)(uint)length > (ulong)(uint)array.Length)
                ThrowHelper.ThrowArgumentOutOfRangeException();
#else
            if ((uint)start > (uint)array.Length || (uint)length > (uint)(array.Length - start))
                ThrowHelper.ThrowArgumentOutOfRangeException();
#endif

            _reference = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(array), (nint)(uint)start /* force zero-extension */);
            _length = length;
        }

        /// <summary>
        /// Creates a new read-only span over the target unmanaged buffer.  Clearly this
        /// is quite dangerous, because we are creating arbitrarily typed T's
        /// out of a void*-typed block of memory.  And the length is not checked.
        /// But if this creation is correct, then all subsequent uses are correct.
        /// </summary>
        /// <param name="pointer">An unmanaged pointer to memory.</param>
        /// <param name="length">The number of <typeparamref name="T"/> elements the memory contains.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when <typeparamref name="T"/> is reference type or contains pointers and hence cannot be stored in unmanaged memory.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the specified <paramref name="length"/> is negative.
        /// </exception>
        [CLSCompliant(false)]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe ReadOnlyUnmanagedSpan(void* pointer, long length)
        {
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
                ThrowHelper.ThrowArgument_TypeContainsReferences(typeof(T));
            if (length < 0)
                ThrowHelper.ThrowArgumentOutOfRangeException();

            _reference = ref *(T*)pointer;
            _length = length;
        }

        /// <summary>Creates a new <see cref="ReadOnlyUnmanagedSpan{T}"/> of length 1 around the specified reference.</summary>
        /// <param name="reference">A reference to data.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlyUnmanagedSpan(ref readonly T reference)
        {
            _reference = ref Unsafe.AsRef(in reference);
            _length = 1;
        }

        // Constructor for internal use only. It is not safe to expose publicly, and is instead exposed via the unsafe MemoryMarshal.CreateReadOnlyUnmanagedSpan.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal ReadOnlyUnmanagedSpan(ref T reference, long length)
        {
            Debug.Assert(length >= 0);

            _reference = ref reference;
            _length = length;
        }

        /// <summary>
        /// Returns the specified element of the read-only span.
        /// </summary>
        /// <param name="index">The zero-based index.</param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException">
        /// Thrown when index less than 0 or index greater than or equal to Length
        /// </exception>
        public ref readonly T this[long index]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if ((ulong)index >= (ulong)_length)
                    ThrowHelper.ThrowIndexOutOfRangeException();
                return ref Unsafe.Add(ref _reference, (nint)index);
            }
        }

        /// <summary>
        /// The number of items in the read-only span.
        /// </summary>
        public long Length
        {
            get => _length;
        }

        /// <summary>
        /// Gets a value indicating whether this <see cref="ReadOnlyUnmanagedSpan{T}"/> is empty.
        /// </summary>
        /// <value><see langword="true"/> if this span is empty; otherwise, <see langword="false"/>.</value>
        public bool IsEmpty
        {
            get => _length == 0;
        }

        /// <summary>
        /// Returns false if left and right point at the same memory and have the same length.  Note that
        /// this does *not* check to see if the *contents* are equal.
        /// </summary>
        public static bool operator !=(ReadOnlyUnmanagedSpan<T> left, ReadOnlyUnmanagedSpan<T> right) => !(left == right);

        /// <summary>
        /// This method is not supported as spans cannot be boxed. To compare two spans, use operator==.
        /// </summary>
        /// <exception cref="NotSupportedException">
        /// Always thrown by this method.
        /// </exception>
        [Obsolete("Equals() on ReadOnlyUnmanagedSpan will always throw an exception. Use the equality operator instead.")]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override bool Equals(object? obj) =>
            throw new NotSupportedException(SR.NotSupported_CannotCallEqualsOnSpan);

        /// <summary>
        /// This method is not supported as spans cannot be boxed.
        /// </summary>
        /// <exception cref="NotSupportedException">
        /// Always thrown by this method.
        /// </exception>
        [Obsolete("GetHashCode() on ReadOnlyUnmanagedSpan will always throw an exception.")]
        [EditorBrowsable(EditorBrowsableState.Never)]
        public override int GetHashCode() =>
            throw new NotSupportedException(SR.NotSupported_CannotCallGetHashCodeOnSpan);

        /// <summary>
        /// Defines an implicit conversion of an array to a <see cref="ReadOnlyUnmanagedSpan{T}"/>
        /// </summary>
        public static implicit operator ReadOnlyUnmanagedSpan<T>(T[]? array) => new ReadOnlyUnmanagedSpan<T>(array);

        /// <summary>
        /// Defines an implicit conversion of a <see cref="ArraySegment{T}"/> to a <see cref="ReadOnlyUnmanagedSpan{T}"/>
        /// </summary>
        public static implicit operator ReadOnlyUnmanagedSpan<T>(ArraySegment<T> segment)
            => new ReadOnlyUnmanagedSpan<T>(segment.Array, segment.Offset, segment.Count);

        /// <summary>
        /// Returns a 0-length read-only span whose base is the null pointer.
        /// </summary>
        public static ReadOnlyUnmanagedSpan<T> Empty => default;

        // Note: CastUp<TDerived> method removed - not applicable for unmanaged types

        /// <summary>Gets an enumerator for this span.</summary>
        public Enumerator GetEnumerator() => new Enumerator(this);

        /// <summary>Enumerates the elements of a <see cref="ReadOnlyUnmanagedSpan{T}"/>.</summary>
        public ref struct Enumerator : IEnumerator<T>
        {
            /// <summary>The span being enumerated.</summary>
            private readonly ReadOnlyUnmanagedSpan<T> _span;
            /// <summary>The next index to yield.</summary>
            private long _index;

            /// <summary>Initialize the enumerator.</summary>
            /// <param name="span">The span to enumerate.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal Enumerator(ReadOnlyUnmanagedSpan<T> span)
            {
                _span = span;
                _index = -1;
            }

            /// <summary>Advances the enumerator to the next element of the span.</summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool MoveNext()
            {
                long index = _index + 1;
                if (index < _span.Length)
                {
                    _index = index;
                    return true;
                }

                return false;
            }

            /// <summary>Gets the element at the current position of the enumerator.</summary>
            public ref readonly T Current
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => ref _span[_index];
            }

            /// <inheritdoc />
            T IEnumerator<T>.Current => Current;

            /// <inheritdoc />
            object IEnumerator.Current => Current!;

            /// <inheritdoc />
            void IEnumerator.Reset() => _index = -1;

            /// <inheritdoc />
            void IDisposable.Dispose() { }
        }

        /// <summary>
        /// Returns a reference to the 0th element of the UnmanagedSpan. If the UnmanagedSpan is empty, returns null reference.
        /// It can be used for pinning and is required to support the use of span within a fixed statement.
        /// </summary>
        [EditorBrowsable(EditorBrowsableState.Never)]
        public ref readonly T GetPinnableReference()
        {
            // Ensure that the native code has just one forward branch that is predicted-not-taken.
            ref T ret = ref Unsafe.NullRef<T>();
            if (_length != 0) ret = ref _reference;
            return ref ret;
        }

        /// <summary>
        /// Copies the contents of this read-only span into destination span. If the source
        /// and destinations overlap, this method behaves as if the original values in
        /// a temporary location before the destination is overwritten.
        /// </summary>
        /// <param name="destination">The span to copy items into.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when the destination UnmanagedSpan is shorter than the source UnmanagedSpan.
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void CopyTo(UnmanagedSpan<T> destination)
        {
            if ((ulong)_length <= (ulong)destination.Length)
            {
                UnmanagedBuffer.Memmove(ref destination._reference, ref Unsafe.AsRef(in _reference), checked((nuint)_length));
            }
            else
            {
                ThrowHelper.ThrowArgumentException_DestinationTooShort();
            }
        }

        /// <summary>
        /// Copies the contents of this read-only span into destination span. If the source
        /// and destinations overlap, this method behaves as if the original values in
        /// a temporary location before the destination is overwritten.
        /// </summary>
        /// <returns>If the destination span is shorter than the source span, this method
        /// return false and no data is written to the destination.</returns>
        /// <param name="destination">The span to copy items into.</param>
        public bool TryCopyTo(UnmanagedSpan<T> destination)
        {
            bool retVal = false;
            if ((ulong)_length <= (ulong)destination.Length)
            {
                UnmanagedBuffer.Memmove(ref destination._reference, ref Unsafe.AsRef(in _reference), checked((nuint)_length));
                retVal = true;
            }
            return retVal;
        }

        /// <summary>
        /// Returns true if left and right point at the same memory and have the same length.  Note that
        /// this does *not* check to see if the *contents* are equal.
        /// </summary>
        public static bool operator ==(ReadOnlyUnmanagedSpan<T> left, ReadOnlyUnmanagedSpan<T> right) =>
            left._length == right._length &&
            Unsafe.AreSame(ref left._reference, ref right._reference);

        /// <summary>
        /// For <see cref="ReadOnlyUnmanagedSpan{Char}"/>, returns a new instance of string that represents the characters pointed to by the span.
        /// Otherwise, returns a <see cref="string"/> with the name of the type and the number of elements.
        /// </summary>
        public override unsafe string ToString()
        {
            if (typeof(T) == typeof(char))
            {
                // For char spans, create string. Need to handle long length.
                if (_length > int.MaxValue)
                    return $"NumSharp.Utilities.ReadOnlyUnmanagedSpan<Char>[{_length}]";
                return new string((char*)Unsafe.AsPointer(ref Unsafe.AsRef(in _reference)), 0, (int)_length);
            }
            return $"NumSharp.Utilities.ReadOnlyUnmanagedSpan<{typeof(T).Name}>[{_length}]";
        }

        /// <summary>
        /// Forms a slice out of the given read-only span, beginning at 'start'.
        /// </summary>
        /// <param name="start">The zero-based index at which to begin this slice.</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the specified <paramref name="start"/> index is not in range (&lt;0 or &gt;Length).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlyUnmanagedSpan<T> Slice(long start)
        {
            if ((ulong)start > (ulong)_length)
                ThrowHelper.ThrowArgumentOutOfRangeException();

            return new ReadOnlyUnmanagedSpan<T>(ref Unsafe.Add(ref _reference, (nint)start), _length - start);
        }

        /// <summary>
        /// Forms a slice out of the given read-only span, beginning at 'start', of given length
        /// </summary>
        /// <param name="start">The zero-based index at which to begin this slice.</param>
        /// <param name="length">The desired length for the slice (exclusive).</param>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when the specified <paramref name="start"/> or end index is not in range (&lt;0 or &gt;Length).
        /// </exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlyUnmanagedSpan<T> Slice(long start, long length)
        {
            // For 64-bit lengths, we need to check that start + length doesn't overflow
            // and that the result is within bounds
            if ((ulong)start > (ulong)_length || (ulong)length > (ulong)(_length - start))
                ThrowHelper.ThrowArgumentOutOfRangeException();

            return new ReadOnlyUnmanagedSpan<T>(ref Unsafe.Add(ref _reference, (nint)start), length);
        }

        /// <summary>
        /// Copies the contents of this read-only span into a new array.  This heap
        /// allocates, so should generally be avoided, however it is sometimes
        /// necessary to bridge the gap with APIs written in terms of arrays.
        /// </summary>
        public T[] ToArray()
        {
            if (IsEmpty)
            {
                return [];
            }

            var destination = new T[Length];
            CopyTo(destination);
            return destination;
        }
    }
}
