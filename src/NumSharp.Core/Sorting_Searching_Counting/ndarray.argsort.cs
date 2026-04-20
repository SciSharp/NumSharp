using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the indices that would sort an array.
        ///
        /// Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.
        /// It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
        /// Supports arrays with >2B elements using long indexing.
        /// </summary>
        public NDArray argsort<T>(int axis = -1) where T : unmanaged
        {
            if (ndim < axis + 1) {
                throw new IndexOutOfRangeException($"Axis = {axis} is out bounds for dimension = {ndim}");
            }

            // argsort's internal GetAtIndex<T> / SortLong<T> paths assume a C-contiguous
            // logical layout. For non-C-contig inputs (F-contig, sliced, transposed),
            // materialize a C-contig copy up front — matches NumPy's behavior of
            // returning a C-contig index array regardless of input layout.
            if (!Shape.IsContiguous)
                return this.copy('C').argsort<T>(axis);

            // Axis -1 means that sort with respect to last axis
            if (axis == -1) {
                axis = ndim-1;
            }

            // Example: If shape is 3x2x3 and we are sorting w.r.t axis = 2 required size is 3x2
            var requiredSize = shape.Take(axis).Concat(shape.Skip(axis + 1)).ToArray();

            if (requiredSize.Length == 0)
            {
                // NumPy argsort always returns int64 (long) indices
                // Use NumPy-compatible comparison that puts NaN at the end
                // Use LongRange for indices > int.MaxValue
                var sorted = LongRange(size)
                    .Select(i => new {Data = GetAtIndex<T>(i), Index = i})
                    .OrderBy(item => item.Data, NumPyComparer<T>.Instance)
                    .Select(item => item.Index)
                    .ToArray();
                return np.array(sorted);
            }

            // Sorted arguments array - NumPy argsort always returns int64 (long) indices
            var resultArray = new NDArray(typeof(long), shape);

            var accessingIndices = AccessorCreatorLong(requiredSize, Enumerable.Empty<long>(), 0);

            // Append the previous indices the sorting accessors
            // Use LongRange for axis sizes > int.MaxValue
            var append = LongRange(shape[axis]);
            var argSort = accessingIndices.Aggregate(Enumerable.Empty<SortedDataLong>(), (allSortedData, seq) =>
            {
                var sortMe = append.Select(value => AppendorLong(value, axis, seq));
                var sortedIndex = SortLong<T>(sortMe);
                return allSortedData.Concat(sortMe.Zip(sortedIndex, (a, b) => new SortedDataLong(a.ToArray(), b)));
            });

            foreach (var arg in argSort)
            {
                resultArray[arg.DataAccessor] = arg.Index;
            }

            return resultArray;
        }

        /// <summary>
        /// Generates a sequence of long values from 0 to count-1.
        /// Replaces Enumerable.Range for long indexing support.
        /// </summary>
        private static IEnumerable<long> LongRange(long count)
        {
            for (long i = 0; i < count; i++)
                yield return i;
        }

        /// <summary>
        /// Appends the given value to the sequences (long version)
        /// </summary>
        private static IEnumerable<long> AppendorLong(long value, int axis, IEnumerable<long> sequences)
        {
            return sequences.Take(axis).Concat(value.Yield()).Concat(sequences.Skip(axis));
        }

        /// <summary>
        /// Creates the indices with which we need to access to the array (long version)
        /// If shape is 3x2x3 and we are sorting w.r.t axis = 2
        /// Return value is [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
        /// </summary>
        private static IEnumerable<IEnumerable<long>> AccessorCreatorLong(long[] originalIndices, IEnumerable<long> previousStep, int currentStep)
        {
            if (originalIndices.Length == currentStep + 1)
            {
                var iterateUntil = originalIndices[currentStep];
                var result = LongRange(iterateUntil).Select(idx => previousStep.Concat(idx.Yield()));
                return result;
            }

            var finalResult = Enumerable.Empty<IEnumerable<long>>();
            return LongRange(originalIndices[currentStep]).Aggregate(finalResult, (current, val) =>
                current.Concat(AccessorCreatorLong(originalIndices, previousStep.Concat(val.Yield()), currentStep + 1)));
        }

        /// <summary>
        /// Sorts the given data (long version).
        /// NumPy sort order: -Inf &lt; normal values &lt; +Inf &lt; NaN
        /// </summary>
        private IEnumerable<long> SortLong<T>(IEnumerable<IEnumerable<long>> accessIndex) where T : unmanaged
        {
            // Extract the scalar value from the NDArray for proper comparison.
            // this[indices] returns an NDArray even for scalar results, and NDArray
            // doesn't implement IComparable, so we must extract the underlying value.
            // Use long index for result
            long idx = 0;
            var sort = accessIndex.Select(x => new {Data = this[x.ToArray()].GetAtIndex<T>(0), Index = idx++});

            // Use NumPy-compatible comparison that puts NaN at the end
            return sort.OrderBy(a => a.Data, NumPyComparer<T>.Instance).Select(a => a.Index);
        }

        /// <summary>
        /// NumPy-compatible comparer for floating-point types.
        /// Ordering: -Inf &lt; normal values &lt; +Inf &lt; NaN
        /// </summary>
        private sealed class NumPyComparer<T> : IComparer<T> where T : unmanaged
        {
            public static readonly NumPyComparer<T> Instance = new NumPyComparer<T>();

            public int Compare(T x, T y)
            {
                // Handle double
                if (typeof(T) == typeof(double))
                {
                    double dx = (double)(object)x;
                    double dy = (double)(object)y;
                    return CompareDouble(dx, dy);
                }

                // Handle float
                if (typeof(T) == typeof(float))
                {
                    float fx = (float)(object)x;
                    float fy = (float)(object)y;
                    return CompareFloat(fx, fy);
                }

                // For non-floating types, use default comparison
                return Comparer<T>.Default.Compare(x, y);
            }

            private static int CompareDouble(double x, double y)
            {
                // NaN sorts to end (greater than everything including +Inf)
                bool xNaN = double.IsNaN(x);
                bool yNaN = double.IsNaN(y);

                if (xNaN && yNaN) return 0;
                if (xNaN) return 1;  // x > y (NaN at end)
                if (yNaN) return -1; // x < y (y is NaN, at end)

                // Standard comparison for non-NaN values
                return x.CompareTo(y);
            }

            private static int CompareFloat(float x, float y)
            {
                // NaN sorts to end (greater than everything including +Inf)
                bool xNaN = float.IsNaN(x);
                bool yNaN = float.IsNaN(y);

                if (xNaN && yNaN) return 0;
                if (xNaN) return 1;  // x > y (NaN at end)
                if (yNaN) return -1; // x < y (y is NaN, at end)

                // Standard comparison for non-NaN values
                return x.CompareTo(y);
            }
        }

        /// <summary>
        /// Data class representing a single sorted element with long indices.
        /// </summary>
        private class SortedDataLong
        {
            /// <summary>
            /// Indexes to access this sorted data. Example: If Array being sorted is shape of 3x2x3
            /// DataAccessor is of the form AxBxC
            /// </summary>
            public long[] DataAccessor { get; }

            /// <summary>
            /// Index of Sorted Element.
            /// </summary>
            public long Index { get; }

            /// <summary>
            /// Data Class Which Represents a Single Sorted Data
            /// </summary>
            public SortedDataLong(long[] dataAccessor, long index)
            {
                DataAccessor = dataAccessor;
                Index = index;
            }
        }
    }
}
