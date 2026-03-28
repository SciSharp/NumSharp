// NumSharp UnmanagedSpan Extensions
// Provides Span<T>-like extension methods for UnmanagedSpan<T> with long indexing support.

using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Extension methods for <see cref="UnmanagedSpan{T}"/> and <see cref="ReadOnlyUnmanagedSpan{T}"/>.
    /// </summary>
    public static class UnmanagedSpanExtensions
    {
        #region IndexOf

        /// <summary>
        /// Searches for the specified value and returns the index of its first occurrence.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOf<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return IndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified value and returns the index of its first occurrence.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        public static long IndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            for (long i = 0; i < span.Length; i++)
            {
                if (value.Equals(span[i]))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the specified sequence and returns the index of its first occurrence.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOf<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            return IndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified sequence and returns the index of its first occurrence.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        public static long IndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length == 0)
                return 0;
            if (value.Length > span.Length)
                return -1;

            long limit = span.Length - value.Length;
            for (long i = 0; i <= limit; i++)
            {
                if (span.Slice(i, value.Length).SequenceEqual(value))
                    return i;
            }
            return -1;
        }

        #endregion

        #region LastIndexOf

        /// <summary>
        /// Searches for the specified value and returns the index of its last occurrence.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOf<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified value and returns the index of its last occurrence.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        public static long LastIndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            for (long i = span.Length - 1; i >= 0; i--)
            {
                if (value.Equals(span[i]))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the specified sequence and returns the index of its last occurrence.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOf<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOf((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches for the specified sequence and returns the index of its last occurrence.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        public static long LastIndexOf<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length == 0)
                return span.Length;
            if (value.Length > span.Length)
                return -1;

            for (long i = span.Length - value.Length; i >= 0; i--)
            {
                if (span.Slice(i, value.Length).SequenceEqual(value))
                    return i;
            }
            return -1;
        }

        #endregion

        #region IndexOfAny

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOfAny<T>(this UnmanagedSpan<T> span, T value0, T value1) where T : unmanaged, IEquatable<T>
        {
            return IndexOfAny((ReadOnlyUnmanagedSpan<T>)span, value0, value1);
        }

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        public static long IndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : unmanaged, IEquatable<T>
        {
            for (long i = 0; i < span.Length; i++)
            {
                T current = span[i];
                if (value0.Equals(current) || value1.Equals(current))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOfAny<T>(this UnmanagedSpan<T> span, T value0, T value1, T value2) where T : unmanaged, IEquatable<T>
        {
            return IndexOfAny((ReadOnlyUnmanagedSpan<T>)span, value0, value1, value2);
        }

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        public static long IndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : unmanaged, IEquatable<T>
        {
            for (long i = 0; i < span.Length; i++)
            {
                T current = span[i];
                if (value0.Equals(current) || value1.Equals(current) || value2.Equals(current))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long IndexOfAny<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> values) where T : unmanaged, IEquatable<T>
        {
            return IndexOfAny((ReadOnlyUnmanagedSpan<T>)span, values);
        }

        /// <summary>
        /// Searches for the first index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the first occurrence, or -1 if not found.</returns>
        public static long IndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> values) where T : unmanaged, IEquatable<T>
        {
            for (long i = 0; i < span.Length; i++)
            {
                if (values.Contains(span[i]))
                    return i;
            }
            return -1;
        }

        #endregion

        #region LastIndexOfAny

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOfAny<T>(this UnmanagedSpan<T> span, T value0, T value1) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOfAny((ReadOnlyUnmanagedSpan<T>)span, value0, value1);
        }

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        public static long LastIndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, T value0, T value1) where T : unmanaged, IEquatable<T>
        {
            for (long i = span.Length - 1; i >= 0; i--)
            {
                T current = span[i];
                if (value0.Equals(current) || value1.Equals(current))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOfAny<T>(this UnmanagedSpan<T> span, T value0, T value1, T value2) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOfAny((ReadOnlyUnmanagedSpan<T>)span, value0, value1, value2);
        }

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        public static long LastIndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, T value0, T value1, T value2) where T : unmanaged, IEquatable<T>
        {
            for (long i = span.Length - 1; i >= 0; i--)
            {
                T current = span[i];
                if (value0.Equals(current) || value1.Equals(current) || value2.Equals(current))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long LastIndexOfAny<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> values) where T : unmanaged, IEquatable<T>
        {
            return LastIndexOfAny((ReadOnlyUnmanagedSpan<T>)span, values);
        }

        /// <summary>
        /// Searches for the last index of any of the specified values.
        /// </summary>
        /// <returns>The zero-based index of the last occurrence, or -1 if not found.</returns>
        public static long LastIndexOfAny<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> values) where T : unmanaged, IEquatable<T>
        {
            for (long i = span.Length - 1; i >= 0; i--)
            {
                if (values.Contains(span[i]))
                    return i;
            }
            return -1;
        }

        #endregion

        #region Contains

        /// <summary>
        /// Determines whether the span contains the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Contains<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return IndexOf(span, value) >= 0;
        }

        /// <summary>
        /// Determines whether the span contains the specified value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Contains<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return IndexOf(span, value) >= 0;
        }

        #endregion

        #region SequenceEqual

        /// <summary>
        /// Determines whether two sequences are equal by comparing elements using IEquatable{T}.Equals(T).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SequenceEqual<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            return SequenceEqual((ReadOnlyUnmanagedSpan<T>)span, other);
        }

        /// <summary>
        /// Determines whether two sequences are equal by comparing elements using IEquatable{T}.Equals(T).
        /// </summary>
        public static bool SequenceEqual<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            if (span.Length != other.Length)
                return false;

            for (long i = 0; i < span.Length; i++)
            {
                if (!span[i].Equals(other[i]))
                    return false;
            }
            return true;
        }

        /// <summary>
        /// Determines whether two sequences are equal by comparing elements using a custom comparer.
        /// </summary>
        public static bool SequenceEqual<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other, IEqualityComparer<T>? comparer) where T : unmanaged
        {
            return SequenceEqual((ReadOnlyUnmanagedSpan<T>)span, other, comparer);
        }

        /// <summary>
        /// Determines whether two sequences are equal by comparing elements using a custom comparer.
        /// </summary>
        public static bool SequenceEqual<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other, IEqualityComparer<T>? comparer) where T : unmanaged
        {
            if (span.Length != other.Length)
                return false;

            comparer ??= EqualityComparer<T>.Default;

            for (long i = 0; i < span.Length; i++)
            {
                if (!comparer.Equals(span[i], other[i]))
                    return false;
            }
            return true;
        }

        #endregion

        #region BinarySearch

        /// <summary>
        /// Searches a sorted span for a value using binary search.
        /// </summary>
        /// <returns>
        /// The zero-based index of the item if found; otherwise, a negative number that is the
        /// bitwise complement of the index of the next element that is larger than item.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long BinarySearch<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IComparable<T>
        {
            return BinarySearch((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Searches a sorted span for a value using binary search.
        /// </summary>
        /// <returns>
        /// The zero-based index of the item if found; otherwise, a negative number that is the
        /// bitwise complement of the index of the next element that is larger than item.
        /// </returns>
        public static long BinarySearch<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IComparable<T>
        {
            return BinarySearch(span, value, Comparer<T>.Default);
        }

        /// <summary>
        /// Searches a sorted span for a value using binary search with a custom comparer.
        /// </summary>
        /// <returns>
        /// The zero-based index of the item if found; otherwise, a negative number that is the
        /// bitwise complement of the index of the next element that is larger than item.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long BinarySearch<T>(this UnmanagedSpan<T> span, T value, IComparer<T>? comparer) where T : unmanaged
        {
            return BinarySearch((ReadOnlyUnmanagedSpan<T>)span, value, comparer);
        }

        /// <summary>
        /// Searches a sorted span for a value using binary search with a custom comparer.
        /// </summary>
        /// <returns>
        /// The zero-based index of the item if found; otherwise, a negative number that is the
        /// bitwise complement of the index of the next element that is larger than item.
        /// </returns>
        public static long BinarySearch<T>(this ReadOnlyUnmanagedSpan<T> span, T value, IComparer<T>? comparer) where T : unmanaged
        {
            comparer ??= Comparer<T>.Default;

            long lo = 0;
            long hi = span.Length - 1;

            while (lo <= hi)
            {
                long mid = lo + ((hi - lo) >> 1);
                int cmp = comparer.Compare(span[mid], value);

                if (cmp == 0)
                    return mid;
                if (cmp < 0)
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }

            return ~lo;
        }

        #endregion

        #region Sort

        /// <summary>
        /// Sorts the elements in the span using the default comparer.
        /// </summary>
        public static void Sort<T>(this UnmanagedSpan<T> span) where T : unmanaged, IComparable<T>
        {
            Sort(span, Comparer<T>.Default);
        }

        /// <summary>
        /// Sorts the elements in the span using a custom comparer.
        /// </summary>
        public static void Sort<T>(this UnmanagedSpan<T> span, IComparer<T>? comparer) where T : unmanaged
        {
            comparer ??= Comparer<T>.Default;
            IntroSort(span, 0, span.Length - 1, 2 * Log2(span.Length), comparer);
        }

        /// <summary>
        /// Sorts the elements in the span using a comparison delegate.
        /// </summary>
        public static void Sort<T>(this UnmanagedSpan<T> span, Comparison<T> comparison) where T : unmanaged
        {
            Sort(span, Comparer<T>.Create(comparison));
        }

        /// <summary>
        /// Sorts a pair of spans (keys and values) based on the keys.
        /// </summary>
        public static void Sort<TKey, TValue>(this UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items)
            where TKey : unmanaged, IComparable<TKey>
            where TValue : unmanaged
        {
            Sort(keys, items, Comparer<TKey>.Default);
        }

        /// <summary>
        /// Sorts a pair of spans (keys and values) based on the keys using a custom comparer.
        /// </summary>
        public static void Sort<TKey, TValue>(this UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, IComparer<TKey>? comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            if (keys.Length != items.Length)
                throw new ArgumentException("Keys and items spans must have the same length.");

            comparer ??= Comparer<TKey>.Default;
            IntroSortWithItems(keys, items, 0, keys.Length - 1, 2 * Log2(keys.Length), comparer);
        }

        private static long Log2(long value)
        {
            long result = 0;
            while (value > 1)
            {
                value >>= 1;
                result++;
            }
            return result;
        }

        private static void IntroSort<T>(UnmanagedSpan<T> span, long lo, long hi, long depthLimit, IComparer<T> comparer) where T : unmanaged
        {
            while (hi > lo)
            {
                long partitionSize = hi - lo + 1;

                if (partitionSize <= 16)
                {
                    InsertionSort(span, lo, hi, comparer);
                    return;
                }

                if (depthLimit == 0)
                {
                    HeapSort(span, lo, hi, comparer);
                    return;
                }

                depthLimit--;
                long p = Partition(span, lo, hi, comparer);
                IntroSort(span, p + 1, hi, depthLimit, comparer);
                hi = p - 1;
            }
        }

        private static void IntroSortWithItems<TKey, TValue>(UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, long lo, long hi, long depthLimit, IComparer<TKey> comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            while (hi > lo)
            {
                long partitionSize = hi - lo + 1;

                if (partitionSize <= 16)
                {
                    InsertionSortWithItems(keys, items, lo, hi, comparer);
                    return;
                }

                if (depthLimit == 0)
                {
                    HeapSortWithItems(keys, items, lo, hi, comparer);
                    return;
                }

                depthLimit--;
                long p = PartitionWithItems(keys, items, lo, hi, comparer);
                IntroSortWithItems(keys, items, p + 1, hi, depthLimit, comparer);
                hi = p - 1;
            }
        }

        private static void InsertionSort<T>(UnmanagedSpan<T> span, long lo, long hi, IComparer<T> comparer) where T : unmanaged
        {
            for (long i = lo + 1; i <= hi; i++)
            {
                T key = span[i];
                long j = i - 1;
                while (j >= lo && comparer.Compare(span[j], key) > 0)
                {
                    span[j + 1] = span[j];
                    j--;
                }
                span[j + 1] = key;
            }
        }

        private static void InsertionSortWithItems<TKey, TValue>(UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, long lo, long hi, IComparer<TKey> comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            for (long i = lo + 1; i <= hi; i++)
            {
                TKey key = keys[i];
                TValue item = items[i];
                long j = i - 1;
                while (j >= lo && comparer.Compare(keys[j], key) > 0)
                {
                    keys[j + 1] = keys[j];
                    items[j + 1] = items[j];
                    j--;
                }
                keys[j + 1] = key;
                items[j + 1] = item;
            }
        }

        private static long Partition<T>(UnmanagedSpan<T> span, long lo, long hi, IComparer<T> comparer) where T : unmanaged
        {
            long mid = lo + (hi - lo) / 2;

            // Median of three
            if (comparer.Compare(span[lo], span[mid]) > 0) Swap(ref span[lo], ref span[mid]);
            if (comparer.Compare(span[lo], span[hi]) > 0) Swap(ref span[lo], ref span[hi]);
            if (comparer.Compare(span[mid], span[hi]) > 0) Swap(ref span[mid], ref span[hi]);

            T pivot = span[mid];
            Swap(ref span[mid], ref span[hi - 1]);

            long i = lo;
            long j = hi - 1;

            while (true)
            {
                while (comparer.Compare(span[++i], pivot) < 0) { }
                while (comparer.Compare(pivot, span[--j]) < 0) { }
                if (i >= j) break;
                Swap(ref span[i], ref span[j]);
            }

            Swap(ref span[i], ref span[hi - 1]);
            return i;
        }

        private static long PartitionWithItems<TKey, TValue>(UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, long lo, long hi, IComparer<TKey> comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            long mid = lo + (hi - lo) / 2;

            // Median of three
            if (comparer.Compare(keys[lo], keys[mid]) > 0) { Swap(ref keys[lo], ref keys[mid]); Swap(ref items[lo], ref items[mid]); }
            if (comparer.Compare(keys[lo], keys[hi]) > 0) { Swap(ref keys[lo], ref keys[hi]); Swap(ref items[lo], ref items[hi]); }
            if (comparer.Compare(keys[mid], keys[hi]) > 0) { Swap(ref keys[mid], ref keys[hi]); Swap(ref items[mid], ref items[hi]); }

            TKey pivot = keys[mid];
            Swap(ref keys[mid], ref keys[hi - 1]);
            Swap(ref items[mid], ref items[hi - 1]);

            long i = lo;
            long j = hi - 1;

            while (true)
            {
                while (comparer.Compare(keys[++i], pivot) < 0) { }
                while (comparer.Compare(pivot, keys[--j]) < 0) { }
                if (i >= j) break;
                Swap(ref keys[i], ref keys[j]);
                Swap(ref items[i], ref items[j]);
            }

            Swap(ref keys[i], ref keys[hi - 1]);
            Swap(ref items[i], ref items[hi - 1]);
            return i;
        }

        private static void HeapSort<T>(UnmanagedSpan<T> span, long lo, long hi, IComparer<T> comparer) where T : unmanaged
        {
            long n = hi - lo + 1;
            for (long i = n / 2; i >= 1; i--)
            {
                Heapify(span, i, n, lo, comparer);
            }
            for (long i = n; i > 1; i--)
            {
                Swap(ref span[lo], ref span[lo + i - 1]);
                Heapify(span, 1, i - 1, lo, comparer);
            }
        }

        private static void HeapSortWithItems<TKey, TValue>(UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, long lo, long hi, IComparer<TKey> comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            long n = hi - lo + 1;
            for (long i = n / 2; i >= 1; i--)
            {
                HeapifyWithItems(keys, items, i, n, lo, comparer);
            }
            for (long i = n; i > 1; i--)
            {
                Swap(ref keys[lo], ref keys[lo + i - 1]);
                Swap(ref items[lo], ref items[lo + i - 1]);
                HeapifyWithItems(keys, items, 1, i - 1, lo, comparer);
            }
        }

        private static void Heapify<T>(UnmanagedSpan<T> span, long i, long n, long lo, IComparer<T> comparer) where T : unmanaged
        {
            T val = span[lo + i - 1];
            while (i <= n / 2)
            {
                long child = 2 * i;
                if (child < n && comparer.Compare(span[lo + child - 1], span[lo + child]) < 0)
                    child++;
                if (comparer.Compare(val, span[lo + child - 1]) >= 0)
                    break;
                span[lo + i - 1] = span[lo + child - 1];
                i = child;
            }
            span[lo + i - 1] = val;
        }

        private static void HeapifyWithItems<TKey, TValue>(UnmanagedSpan<TKey> keys, UnmanagedSpan<TValue> items, long i, long n, long lo, IComparer<TKey> comparer)
            where TKey : unmanaged
            where TValue : unmanaged
        {
            TKey key = keys[lo + i - 1];
            TValue item = items[lo + i - 1];
            while (i <= n / 2)
            {
                long child = 2 * i;
                if (child < n && comparer.Compare(keys[lo + child - 1], keys[lo + child]) < 0)
                    child++;
                if (comparer.Compare(key, keys[lo + child - 1]) >= 0)
                    break;
                keys[lo + i - 1] = keys[lo + child - 1];
                items[lo + i - 1] = items[lo + child - 1];
                i = child;
            }
            keys[lo + i - 1] = key;
            items[lo + i - 1] = item;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Swap<T>(ref T a, ref T b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        #endregion

        #region Reverse

        /// <summary>
        /// Reverses the elements in the span in-place.
        /// </summary>
        public static void Reverse<T>(this UnmanagedSpan<T> span) where T : unmanaged
        {
            if (span.Length <= 1)
                return;

            long i = 0;
            long j = span.Length - 1;
            while (i < j)
            {
                T temp = span[i];
                span[i] = span[j];
                span[j] = temp;
                i++;
                j--;
            }
        }

        #endregion

        #region StartsWith / EndsWith

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
        public static bool StartsWith<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length > span.Length)
                return false;
            return span.Slice(0, value.Length).SequenceEqual(value);
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
        public static bool EndsWith<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> value) where T : unmanaged, IEquatable<T>
        {
            if (value.Length > span.Length)
                return false;
            return span.Slice(span.Length - value.Length).SequenceEqual(value);
        }

        #endregion

        #region Count

        /// <summary>
        /// Counts the number of occurrences of the specified value in the span.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Count<T>(this UnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            return Count((ReadOnlyUnmanagedSpan<T>)span, value);
        }

        /// <summary>
        /// Counts the number of occurrences of the specified value in the span.
        /// </summary>
        public static long Count<T>(this ReadOnlyUnmanagedSpan<T> span, T value) where T : unmanaged, IEquatable<T>
        {
            long count = 0;
            for (long i = 0; i < span.Length; i++)
            {
                if (value.Equals(span[i]))
                    count++;
            }
            return count;
        }

        #endregion

        #region CommonPrefixLength

        /// <summary>
        /// Determines the length of any common prefix shared between the span and the other span.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long CommonPrefixLength<T>(this UnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            return CommonPrefixLength((ReadOnlyUnmanagedSpan<T>)span, other);
        }

        /// <summary>
        /// Determines the length of any common prefix shared between the span and the other span.
        /// </summary>
        public static long CommonPrefixLength<T>(this ReadOnlyUnmanagedSpan<T> span, ReadOnlyUnmanagedSpan<T> other) where T : unmanaged, IEquatable<T>
        {
            long minLength = Math.Min(span.Length, other.Length);
            for (long i = 0; i < minLength; i++)
            {
                if (!span[i].Equals(other[i]))
                    return i;
            }
            return minLength;
        }

        #endregion

        #region Replace

        /// <summary>
        /// Replaces all occurrences of oldValue with newValue.
        /// </summary>
        public static void Replace<T>(this UnmanagedSpan<T> span, T oldValue, T newValue) where T : unmanaged, IEquatable<T>
        {
            for (long i = 0; i < span.Length; i++)
            {
                if (oldValue.Equals(span[i]))
                    span[i] = newValue;
            }
        }

        #endregion
    }
}
