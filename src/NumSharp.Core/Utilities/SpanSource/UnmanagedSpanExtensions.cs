// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// Adapted for NumSharp UnmanagedSpan<T> with long indexing support.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Extension methods for UnmanagedSpan{T} and ReadOnlyUnmanagedSpan{T}.
    /// These provide SIMD-accelerated operations where possible.
    /// </summary>
    public static class UnmanagedSpanExtensions
    {
        // ==================================================================================
        // Contains
        // ==================================================================================

        /// <summary>
        /// Searches for the specified value and returns true if found.
        /// Uses SIMD acceleration for numeric types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Contains<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return Contains((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified value and returns true if found.
        /// Uses SIMD acceleration for numeric types.
        /// </summary>
        public static unsafe bool Contains<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            if (span.Length == 0)
                return false;

            if (span.Length > int.MaxValue)
            {
                return ContainsLong(span, value);
            }

            // Use Unsafe.AsRef to convert readonly ref to mutable ref (safe since we're only reading)
            ref T searchSpace = ref Unsafe.AsRef(in span.GetPinnableReference());
            int length = (int)span.Length;

            // Try SIMD path for numeric types via byte reinterpretation
            if (TryContainsValueType(ref searchSpace, value, length, out bool result))
                return result;

            return UnmanagedSpanHelpers.Contains(ref searchSpace, value, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryContainsValueType<T>(ref T searchSpace, T value, int length, out bool result) where T : unmanaged
        {
            result = false;

            if (typeof(T) == typeof(byte))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, byte>(ref searchSpace), Unsafe.As<T, byte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(sbyte))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, sbyte>(ref searchSpace), Unsafe.As<T, sbyte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(short))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, short>(ref searchSpace), Unsafe.As<T, short>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(ushort))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, short>(ref searchSpace), Unsafe.As<T, short>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(int))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, int>(ref searchSpace), Unsafe.As<T, int>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(uint))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, int>(ref searchSpace), Unsafe.As<T, int>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(long))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, long>(ref searchSpace), Unsafe.As<T, long>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(ulong))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, long>(ref searchSpace), Unsafe.As<T, long>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(float))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, float>(ref searchSpace), Unsafe.As<T, float>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(double))
            {
                result = UnmanagedSpanHelpers.ContainsValueType(ref Unsafe.As<T, double>(ref searchSpace), Unsafe.As<T, double>(ref value), length);
                return true;
            }

            return false; // Not a supported numeric type
        }

        private static bool ContainsLong<T>(ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            long remaining = span.Length;
            long offset = 0;
            while (remaining > 0)
            {
                int chunkSize = (int)Math.Min(remaining, int.MaxValue);
                if (Contains(span.Slice(offset, chunkSize), value))
                    return true;
                offset += chunkSize;
                remaining -= chunkSize;
            }
            return false;
        }

        // ==================================================================================
        // IndexOf
        // ==================================================================================

        /// <summary>
        /// Searches for the specified value and returns the index of its first occurrence.
        /// Returns -1 if not found. Uses SIMD acceleration for numeric types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOf<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return IndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified value and returns the index of its first occurrence.
        /// Returns -1 if not found. Uses SIMD acceleration for numeric types.
        /// </summary>
        public static unsafe long IndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            if (span.Length == 0)
                return -1;

            if (span.Length > int.MaxValue)
            {
                return IndexOfLong(span, value);
            }

            ref T searchSpace = ref Unsafe.AsRef(in span.GetPinnableReference());
            int length = (int)span.Length;

            if (TryIndexOfValueType(ref searchSpace, value, length, out int result))
                return result;

            return UnmanagedSpanHelpers.IndexOf(ref searchSpace, value, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryIndexOfValueType<T>(ref T searchSpace, T value, int length, out int result) where T : unmanaged
        {
            result = -1;

            if (typeof(T) == typeof(byte))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, byte>(ref searchSpace), Unsafe.As<T, byte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(sbyte))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, sbyte>(ref searchSpace), Unsafe.As<T, sbyte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(short))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, short>(ref searchSpace), Unsafe.As<T, short>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(ushort))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, short>(ref searchSpace), Unsafe.As<T, short>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(int))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, int>(ref searchSpace), Unsafe.As<T, int>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(uint))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, int>(ref searchSpace), Unsafe.As<T, int>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(long))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, long>(ref searchSpace), Unsafe.As<T, long>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(ulong))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, long>(ref searchSpace), Unsafe.As<T, long>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(float))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, float>(ref searchSpace), Unsafe.As<T, float>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(double))
            {
                result = UnmanagedSpanHelpers.IndexOfValueType(ref Unsafe.As<T, double>(ref searchSpace), Unsafe.As<T, double>(ref value), length);
                return true;
            }

            return false;
        }

        private static long IndexOfLong<T>(ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            long remaining = span.Length;
            long offset = 0;
            while (remaining > 0)
            {
                int chunkSize = (int)Math.Min(remaining, int.MaxValue);
                long idx = IndexOf(span.Slice(offset, chunkSize), value);
                if (idx >= 0)
                    return offset + idx;
                offset += chunkSize;
                remaining -= chunkSize;
            }
            return -1;
        }

        // ==================================================================================
        // LastIndexOf
        // ==================================================================================

        /// <summary>
        /// Searches for the specified value and returns the index of its last occurrence.
        /// Returns -1 if not found. Uses SIMD acceleration for numeric types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOf<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified value and returns the index of its last occurrence.
        /// Returns -1 if not found. Uses SIMD acceleration for numeric types.
        /// </summary>
        public static unsafe long LastIndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            if (span.Length == 0)
                return -1;

            if (span.Length > int.MaxValue)
            {
                return LastIndexOfLong(span, value);
            }

            ref T searchSpace = ref Unsafe.AsRef(in span.GetPinnableReference());
            int length = (int)span.Length;

            if (TryLastIndexOfValueType(ref searchSpace, value, length, out int result))
                return result;

            return UnmanagedSpanHelpers.LastIndexOf(ref searchSpace, value, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TryLastIndexOfValueType<T>(ref T searchSpace, T value, int length, out int result) where T : unmanaged
        {
            result = -1;

            if (typeof(T) == typeof(byte))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, byte>(ref searchSpace), Unsafe.As<T, byte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(sbyte))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, sbyte>(ref searchSpace), Unsafe.As<T, sbyte>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(short))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, short>(ref searchSpace), Unsafe.As<T, short>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(int))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, int>(ref searchSpace), Unsafe.As<T, int>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(long))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, long>(ref searchSpace), Unsafe.As<T, long>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(float))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, float>(ref searchSpace), Unsafe.As<T, float>(ref value), length);
                return true;
            }
            if (typeof(T) == typeof(double))
            {
                result = UnmanagedSpanHelpers.LastIndexOfValueType(ref Unsafe.As<T, double>(ref searchSpace), Unsafe.As<T, double>(ref value), length);
                return true;
            }

            return false;
        }

        private static long LastIndexOfLong<T>(ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            long remaining = span.Length;
            while (remaining > 0)
            {
                int chunkSize = (int)Math.Min(remaining, int.MaxValue);
                long offset = remaining - chunkSize;
                long idx = LastIndexOf(span.Slice(offset, chunkSize), value);
                if (idx >= 0)
                    return offset + idx;
                remaining = offset;
            }
            return -1;
        }

        // ==================================================================================
        // SequenceEqual
        // ==================================================================================

        /// <summary>
        /// Determines whether two spans are equal by comparing elements using IEquatable{T}.Equals.
        /// Uses SIMD acceleration for numeric types.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SequenceEqual<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            return SequenceEqual((ReadOnlyUnmanagedSpan<T>)span, other);
        }

        /// <summary>
        /// Determines whether two spans are equal by comparing elements using IEquatable{T}.Equals.
        /// Uses SIMD acceleration for numeric types.
        /// </summary>
        public static unsafe bool SequenceEqual<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            if (span.Length != other.Length)
                return false;

            if (span.Length == 0)
                return true;

            if (span.Length > int.MaxValue)
            {
                return SequenceEqualLong(span, other);
            }

            ref T first = ref Unsafe.AsRef(in span.GetPinnableReference());
            ref T second = ref Unsafe.AsRef(in other.GetPinnableReference());
            int length = (int)span.Length;

            if (TrySequenceEqualValueType(ref first, ref second, length, out bool result))
                return result;

            return UnmanagedSpanHelpers.SequenceEqual(ref first, ref second, length);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe bool TrySequenceEqualValueType<T>(ref T first, ref T second, int length, out bool result) where T : unmanaged
        {
            result = false;

            if (typeof(T) == typeof(byte))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, byte>(ref first), ref Unsafe.As<T, byte>(ref second), length);
                return true;
            }
            if (typeof(T) == typeof(short))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, short>(ref first), ref Unsafe.As<T, short>(ref second), length);
                return true;
            }
            if (typeof(T) == typeof(int))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, int>(ref first), ref Unsafe.As<T, int>(ref second), length);
                return true;
            }
            if (typeof(T) == typeof(long))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, long>(ref first), ref Unsafe.As<T, long>(ref second), length);
                return true;
            }
            if (typeof(T) == typeof(float))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, float>(ref first), ref Unsafe.As<T, float>(ref second), length);
                return true;
            }
            if (typeof(T) == typeof(double))
            {
                result = UnmanagedSpanHelpers.SequenceEqualValueType(ref Unsafe.As<T, double>(ref first), ref Unsafe.As<T, double>(ref second), length);
                return true;
            }

            return false;
        }

        private static bool SequenceEqualLong<T>(ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            long remaining = span.Length;
            long offset = 0;
            while (remaining > 0)
            {
                int chunkSize = (int)Math.Min(remaining, int.MaxValue);
                if (!SequenceEqual(span.Slice(offset, chunkSize), other.Slice(offset, chunkSize)))
                    return false;
                offset += chunkSize;
                remaining -= chunkSize;
            }
            return true;
        }

        // ==================================================================================
        // Reverse
        // ==================================================================================

        /// <summary>
        /// Reverses the elements in the span.
        /// Uses SIMD acceleration for power-of-2 element sizes.
        /// </summary>
        public static unsafe void Reverse<T>(this UnmanagedSpan<T> span) where T : unmanaged
        {
            if (span.Length <= 1)
                return;

            if (span.Length > (long)nuint.MaxValue)
            {
                ReverseLong(span);
                return;
            }

            ref T reference = ref span.GetPinnableReference();
            UnmanagedSpanHelpers.Reverse(ref reference, (nuint)span.Length);
        }

        private static void ReverseLong<T>(UnmanagedSpan<T> span) where T : unmanaged
        {
            long left = 0;
            long right = span.Length - 1;
            while (left < right)
            {
                T temp = span[left];
                span[left] = span[right];
                span[right] = temp;
                left++;
                right--;
            }
        }

        // ==================================================================================
        // Fill
        // ==================================================================================

        /// <summary>
        /// Fills the span with the specified value.
        /// Uses SIMD acceleration for power-of-2 element sizes up to 8 bytes.
        /// </summary>
        public static unsafe void Fill<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged
        {
            if (span.Length == 0)
                return;

            if (span.Length > (long)nuint.MaxValue)
            {
                FillLong(span, value);
                return;
            }

            ref T reference = ref span.GetPinnableReference();
            UnmanagedSpanHelpers.Fill(ref reference, (nuint)span.Length, value);
        }

        private static void FillLong<T>(UnmanagedSpan<T> span, T value) where T : unmanaged
        {
            long remaining = span.Length;
            long offset = 0;
            while (remaining > 0)
            {
                long chunkSize = Math.Min(remaining, (long)nuint.MaxValue);
                var chunk = span.Slice(offset, chunkSize);
                ref T reference = ref chunk.GetPinnableReference();
                UnmanagedSpanHelpers.Fill(ref reference, (nuint)chunkSize, value);
                offset += chunkSize;
                remaining -= chunkSize;
            }
        }

        // ==================================================================================
        // BinarySearch
        // ==================================================================================

        /// <summary>
        /// Searches a sorted span for the specified value using binary search.
        /// Returns the index if found, or a negative number if not found.
        /// The negative number is the bitwise complement of the index where the value should be inserted.
        /// </summary>
        public static long BinarySearch<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IComparable<T>
        {
            return BinarySearch((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches a sorted span for the specified value using binary search.
        /// Returns the index if found, or a negative number if not found.
        /// The negative number is the bitwise complement of the index where the value should be inserted.
        /// </summary>
        public static long BinarySearch<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IComparable<T>
        {
            long lo = 0;
            long hi = span.Length - 1;

            while (lo <= hi)
            {
                long mid = lo + ((hi - lo) >> 1);
                int cmp = span[mid].CompareTo(value);

                if (cmp == 0)
                    return mid;
                if (cmp < 0)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }

            return ~lo;
        }

        // ==================================================================================
        // StartsWith / EndsWith
        // ==================================================================================

        /// <summary>
        /// Determines whether the span starts with the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool StartsWith<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            return StartsWith((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Determines whether the span starts with the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool StartsWith<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length > span.Length)
                return false;

            return SequenceEqual(span.Slice(0, value.Length), value);
        }

        /// <summary>
        /// Determines whether the span ends with the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool EndsWith<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            return EndsWith((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Determines whether the span ends with the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool EndsWith<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length > span.Length)
                return false;

            return SequenceEqual(span.Slice(span.Length - value.Length), value);
        }

        // ==================================================================================
        // CopyTo
        // ==================================================================================

        /// <summary>
        /// Copies the contents of this span into a destination span.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void CopyTo<T>(this ReadOnlyUnmanagedSpan<T> source, UnmanagedSpan<T> destination) where T : unmanaged
        {
            if (source.Length > destination.Length)
                ThrowHelper.ThrowArgumentException_DestinationTooShort();

            if (source.Length == 0)
                return;

            ref T srcRef = ref Unsafe.AsRef(in source.GetPinnableReference());
            ref T dstRef = ref destination.GetPinnableReference();
            UnmanagedBuffer.Memmove(ref dstRef, ref srcRef, (ulong)source.Length);
        }

        /// <summary>
        /// Attempts to copy the contents of this span into a destination span.
        /// Returns false if the destination is too short.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe bool TryCopyTo<T>(this ReadOnlyUnmanagedSpan<T> source, UnmanagedSpan<T> destination) where T : unmanaged
        {
            if (source.Length > destination.Length)
                return false;

            if (source.Length == 0)
                return true;

            ref T srcRef = ref Unsafe.AsRef(in source.GetPinnableReference());
            ref T dstRef = ref destination.GetPinnableReference();
            UnmanagedBuffer.Memmove(ref dstRef, ref srcRef, (ulong)source.Length);
            return true;
        }
    }
}
