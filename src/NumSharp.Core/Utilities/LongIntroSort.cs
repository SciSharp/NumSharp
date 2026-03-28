using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// IntroSort implementation supporting long indexing for arrays exceeding int.MaxValue elements.
    /// Ported from .NET runtime's ArraySortHelper with int indices replaced by long.
    ///
    /// Algorithm: Introspective Sort (hybrid of QuickSort, HeapSort, InsertionSort)
    /// - QuickSort for large unsorted segments
    /// - HeapSort when recursion depth exceeds O(log n) to avoid O(n²) worst case
    /// - InsertionSort for small partitions (≤16 elements) for cache efficiency
    /// </summary>
    internal static class LongIntroSort
    {
        private const long IntrosortSizeThreshold = 16;

        /// <summary>
        /// Sorts elements in-place using IntroSort algorithm.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged and comparable)</typeparam>
        /// <param name="ptr">Pointer to first element</param>
        /// <param name="length">Number of elements to sort</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Sort<T>(T* ptr, long length) where T : unmanaged, IComparable<T>
        {
            if (length > 1)
            {
                IntroSort(ptr, 0, length - 1, 2 * (Log2((ulong)length) + 1));
            }
        }

        /// <summary>
        /// Sorts elements in-place using IntroSort algorithm with custom comparer.
        /// </summary>
        /// <typeparam name="T">Element type (must be unmanaged)</typeparam>
        /// <param name="ptr">Pointer to first element</param>
        /// <param name="length">Number of elements to sort</param>
        /// <param name="comparer">Comparison delegate</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void Sort<T>(T* ptr, long length, Comparison<T> comparer) where T : unmanaged
        {
            if (length > 1)
            {
                IntroSort(ptr, 0, length - 1, 2 * (Log2((ulong)length) + 1), comparer);
            }
        }

        /// <summary>
        /// Log base 2 for ulong values.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long Log2(ulong value)
        {
            return BitOperations.Log2(value);
        }

        #region Generic IComparable<T> version

        private static unsafe void IntroSort<T>(T* ptr, long lo, long hi, long depthLimit) where T : unmanaged, IComparable<T>
        {
            while (hi > lo)
            {
                long partitionSize = hi - lo + 1;

                if (partitionSize <= IntrosortSizeThreshold)
                {
                    if (partitionSize == 2)
                    {
                        SwapIfGreater(ptr, lo, hi);
                    }
                    else if (partitionSize == 3)
                    {
                        SwapIfGreater(ptr, lo, hi - 1);
                        SwapIfGreater(ptr, lo, hi);
                        SwapIfGreater(ptr, hi - 1, hi);
                    }
                    else
                    {
                        InsertionSort(ptr, lo, hi);
                    }
                    return;
                }

                if (depthLimit == 0)
                {
                    HeapSort(ptr, lo, hi);
                    return;
                }

                depthLimit--;
                long p = PickPivotAndPartition(ptr, lo, hi);
                IntroSort(ptr, p + 1, hi, depthLimit);
                hi = p - 1;
            }
        }

        private static unsafe long PickPivotAndPartition<T>(T* ptr, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long middle = lo + ((hi - lo) >> 1);

            // Median-of-three pivot selection
            SwapIfGreater(ptr, lo, middle);
            SwapIfGreater(ptr, lo, hi);
            SwapIfGreater(ptr, middle, hi);

            T pivot = ptr[middle];
            Swap(ptr, middle, hi - 1);

            long left = lo;
            long right = hi - 1;

            while (left < right)
            {
                while (ptr[++left].CompareTo(pivot) < 0) ;
                while (pivot.CompareTo(ptr[--right]) < 0) ;

                if (left >= right)
                    break;

                Swap(ptr, left, right);
            }

            if (left != hi - 1)
            {
                Swap(ptr, left, hi - 1);
            }

            return left;
        }

        private static unsafe void InsertionSort<T>(T* ptr, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
                T t = ptr[i + 1];

                while (j >= lo && t.CompareTo(ptr[j]) < 0)
                {
                    ptr[j + 1] = ptr[j];
                    j--;
                }

                ptr[j + 1] = t;
            }
        }

        private static unsafe void HeapSort<T>(T* ptr, long lo, long hi) where T : unmanaged, IComparable<T>
        {
            long n = hi - lo + 1;

            for (long i = n >> 1; i >= 1; i--)
            {
                DownHeap(ptr, i, n, lo);
            }

            for (long i = n; i > 1; i--)
            {
                Swap(ptr, lo, lo + i - 1);
                DownHeap(ptr, 1, i - 1, lo);
            }
        }

        private static unsafe void DownHeap<T>(T* ptr, long i, long n, long lo) where T : unmanaged, IComparable<T>
        {
            T d = ptr[lo + i - 1];

            while (i <= n >> 1)
            {
                long child = 2 * i;

                if (child < n && ptr[lo + child - 1].CompareTo(ptr[lo + child]) < 0)
                {
                    child++;
                }

                if (d.CompareTo(ptr[lo + child - 1]) >= 0)
                    break;

                ptr[lo + i - 1] = ptr[lo + child - 1];
                i = child;
            }

            ptr[lo + i - 1] = d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SwapIfGreater<T>(T* ptr, long i, long j) where T : unmanaged, IComparable<T>
        {
            if (ptr[i].CompareTo(ptr[j]) > 0)
            {
                T temp = ptr[i];
                ptr[i] = ptr[j];
                ptr[j] = temp;
            }
        }

        #endregion

        #region Custom Comparison<T> version

        private static unsafe void IntroSort<T>(T* ptr, long lo, long hi, long depthLimit, Comparison<T> comparer) where T : unmanaged
        {
            while (hi > lo)
            {
                long partitionSize = hi - lo + 1;

                if (partitionSize <= IntrosortSizeThreshold)
                {
                    if (partitionSize == 2)
                    {
                        SwapIfGreater(ptr, lo, hi, comparer);
                    }
                    else if (partitionSize == 3)
                    {
                        SwapIfGreater(ptr, lo, hi - 1, comparer);
                        SwapIfGreater(ptr, lo, hi, comparer);
                        SwapIfGreater(ptr, hi - 1, hi, comparer);
                    }
                    else
                    {
                        InsertionSort(ptr, lo, hi, comparer);
                    }
                    return;
                }

                if (depthLimit == 0)
                {
                    HeapSort(ptr, lo, hi, comparer);
                    return;
                }

                depthLimit--;
                long p = PickPivotAndPartition(ptr, lo, hi, comparer);
                IntroSort(ptr, p + 1, hi, depthLimit, comparer);
                hi = p - 1;
            }
        }

        private static unsafe long PickPivotAndPartition<T>(T* ptr, long lo, long hi, Comparison<T> comparer) where T : unmanaged
        {
            long middle = lo + ((hi - lo) >> 1);

            // Median-of-three pivot selection
            SwapIfGreater(ptr, lo, middle, comparer);
            SwapIfGreater(ptr, lo, hi, comparer);
            SwapIfGreater(ptr, middle, hi, comparer);

            T pivot = ptr[middle];
            Swap(ptr, middle, hi - 1);

            long left = lo;
            long right = hi - 1;

            while (left < right)
            {
                while (comparer(ptr[++left], pivot) < 0) ;
                while (comparer(pivot, ptr[--right]) < 0) ;

                if (left >= right)
                    break;

                Swap(ptr, left, right);
            }

            if (left != hi - 1)
            {
                Swap(ptr, left, hi - 1);
            }

            return left;
        }

        private static unsafe void InsertionSort<T>(T* ptr, long lo, long hi, Comparison<T> comparer) where T : unmanaged
        {
            for (long i = lo; i < hi; i++)
            {
                long j = i;
                T t = ptr[i + 1];

                while (j >= lo && comparer(t, ptr[j]) < 0)
                {
                    ptr[j + 1] = ptr[j];
                    j--;
                }

                ptr[j + 1] = t;
            }
        }

        private static unsafe void HeapSort<T>(T* ptr, long lo, long hi, Comparison<T> comparer) where T : unmanaged
        {
            long n = hi - lo + 1;

            for (long i = n >> 1; i >= 1; i--)
            {
                DownHeap(ptr, i, n, lo, comparer);
            }

            for (long i = n; i > 1; i--)
            {
                Swap(ptr, lo, lo + i - 1);
                DownHeap(ptr, 1, i - 1, lo, comparer);
            }
        }

        private static unsafe void DownHeap<T>(T* ptr, long i, long n, long lo, Comparison<T> comparer) where T : unmanaged
        {
            T d = ptr[lo + i - 1];

            while (i <= n >> 1)
            {
                long child = 2 * i;

                if (child < n && comparer(ptr[lo + child - 1], ptr[lo + child]) < 0)
                {
                    child++;
                }

                if (comparer(d, ptr[lo + child - 1]) >= 0)
                    break;

                ptr[lo + i - 1] = ptr[lo + child - 1];
                i = child;
            }

            ptr[lo + i - 1] = d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void SwapIfGreater<T>(T* ptr, long i, long j, Comparison<T> comparer) where T : unmanaged
        {
            if (comparer(ptr[i], ptr[j]) > 0)
            {
                T temp = ptr[i];
                ptr[i] = ptr[j];
                ptr[j] = temp;
            }
        }

        #endregion

        #region Shared helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Swap<T>(T* ptr, long i, long j) where T : unmanaged
        {
            T temp = ptr[i];
            ptr[i] = ptr[j];
            ptr[j] = temp;
        }

        #endregion
    }
}
