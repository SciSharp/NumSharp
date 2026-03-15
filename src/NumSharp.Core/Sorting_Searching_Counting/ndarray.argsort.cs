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
        /// Perform an indirect sort along the given axis using the algorithm specified by the kind keyword.It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
        /// </summary>
        public NDArray argsort<T>(int axis = -1) where T : unmanaged
        {
            if (ndim < axis + 1) {
                throw new IndexOutOfRangeException($"Axis = {axis} is out bounds for dimension = {ndim}");
            }

            // Axis -1 means that sort with respect to last axis
            if (axis == -1) {
                axis = ndim-1;
            }

            // Example: If shape is 3x2x3 and we are soritng w.r.t axis = 2 required size is 3x2
            var requiredSize = shape.Take(axis).Concat(shape.Skip(axis + 1)).Select(x => (int)x).ToArray();

            if (requiredSize.Length == 0)
            {
                // NumPy argsort always returns int64 (long) indices
                // Use NumPy-compatible comparison that puts NaN at the end
                // Enumerable.Range is limited to int.MaxValue, throw clear error for larger arrays
                if (size > int.MaxValue)
                    throw new NotSupportedException($"argsort does not support arrays with more than {int.MaxValue} elements. Array size: {size}");

                var sorted = Enumerable.Range(0, (int)size)
                    .Select(i => new {Data = GetAtIndex<T>(i), Index = (long)i})
                    .OrderBy(item => item.Data, NumPyComparer<T>.Instance)
                    .Select(item => item.Index)
                    .ToArray();
                return np.array(sorted);
            }

            // Sorted arguments array - NumPy argsort always returns int64 (long) indices
            var resultArray = new NDArray(typeof(long), shape);

            var accessingIndices = AccessorCreator(requiredSize, Enumerable.Empty<int>(), 0);

            // Append the previous indices the sorting accessors
            // Enumerable.Range is limited to int.MaxValue, throw clear error for larger axis sizes
            if (shape[axis] > int.MaxValue)
                throw new NotSupportedException($"argsort does not support axis with more than {int.MaxValue} elements. Axis {axis} size: {shape[axis]}");
            var append = Enumerable.Range(0, (int)shape[axis]);
            var argSort = accessingIndices.Aggregate(Enumerable.Empty<SortedData>(), (allSortedData, seq) =>
            {
                var sortMe = append.Select(value => Appendor(value, axis, seq));
                var sortedIndex = Sort<T>(sortMe);
                return allSortedData.Concat(sortMe.Zip(sortedIndex, (a, b) => new SortedData(a.ToArray(), b)));
            });

            foreach (var arg in argSort)
            {
                resultArray[arg.DataAccessor] = arg.Index;
            }

            return resultArray;
        }

        /// <summary>
        /// Appends the given value to the sequences
        /// </summary>
        /// <param name="value"></param>
        /// <param name="axis"></param>
        /// <param name="sequences"></param>
        /// <returns></returns>
        private static IEnumerable<int> Appendor(int value, int axis, IEnumerable<int> sequences)
        {
            return sequences.Take(axis).Concat((value).Yield()).Concat(sequences.Skip(axis));
        }

        /// <summary>
        /// Creates the indices with which we need to access to the array
        /// If shape is 3x2x3 and we are soritng w.r.t axis = 2
        /// Return value is[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
        /// </summary>
        /// <param name="originalIndices"></param>
        /// <param name="previousStep"></param>
        /// <param name="currentStep"></param>
        /// <returns></returns>
        private static IEnumerable<IEnumerable<int>> AccessorCreator(int[] originalIndices, IEnumerable<int> previousStep, int currentStep)
        {
            if (originalIndices.Length == currentStep + 1)
            {
                var iterateUntil = originalIndices[currentStep];
                var result = Enumerable.Range(0, iterateUntil).Select(_ => previousStep.Concat((_).Yield()));
                return result;
            }

            var finalResult = Enumerable.Empty<IEnumerable<int>>();
            return Enumerable.Range(0, originalIndices[currentStep]).Aggregate(finalResult, (current, val) => current.Concat(AccessorCreator(originalIndices, previousStep.Concat((val).Yield()), currentStep + 1)));
        }

        /// <summary>
        /// Sorts the given data. This method should implement quick sort etc...
        /// NumPy sort order: -Inf &lt; normal values &lt; +Inf &lt; NaN
        /// </summary>
        /// <typeparam name="T">Type of parameters</typeparam>
        /// <param name="accessIndex">Indexes to access the data</param>
        /// <returns>Sorted Data</returns>
        private IEnumerable<int> Sort<T>(IEnumerable<IEnumerable<int>> accessIndex) where T : unmanaged
        {
            // Extract the scalar value from the NDArray for proper comparison.
            // this[indices] returns an NDArray even for scalar results, and NDArray
            // doesn't implement IComparable, so we must extract the underlying value.
            var sort = accessIndex.Select((x, index) => new {Data = this[x.ToArray()].GetAtIndex<T>(0), Index = index});

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

        private class SortedData
        {
            /// <summary>
            /// Indexes to access this sorted data. Example: If Array being sorted is shape of 3x2x3
            /// DataAccessor is of the form AxBxC
            /// </summary>
            public int[] DataAccessor { get; }

            /// <summary>
            /// Index of Sorted Element.
            /// </summary>
            public int Index { get; }

            /// <summary>
            /// Data Class Which Represents a Single Sorted Data
            /// </summary>
            public SortedData(int[] dataAccessor, int index)
            {
                DataAccessor = dataAccessor;
                Index = index;
            }
        }
    }
}
