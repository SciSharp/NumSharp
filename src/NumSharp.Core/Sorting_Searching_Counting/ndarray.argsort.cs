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
        public NDArray argsort<T>(int axis = -1)
        {
            if (ndim < axis + 1) {
                throw new IndexOutOfRangeException($"Axis = {axis} is out bounds for dimension = {ndim}");
            }

            // Axis -1 means that sort with respect to last axis
            if (axis == -1) {
                axis = ndim-1;
            }

            // Example: If shape is 3x2x3 and we are soritng w.r.t axis = 2 required size is 3x2
            var requiredSize = shape.Take(axis).Concat(shape.Skip(axis + 1)).ToArray();

            if (requiredSize.Length == 0)
            {
                var data = Array;
                var sorted = Enumerable.Range(0, size)
                    .Select(_ => new {Data = data[_], Index = _})
                    .OrderBy(_ => _.Data)
                    .Select(_ => _.Index)
                    .ToArray();
                return np.array(sorted);
            }

            // Sorted arguments array
            var resultArray = new NDArray(typeof(T), shape);

            var accessingIndices = AccessorCreator(requiredSize, Enumerable.Empty<int>(), 0);

            // Append the previous indices the sorting accessors
            var append = Enumerable.Range(0, shape[axis]);
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
        /// </summary>
        /// <typeparam name="T">Type of parameters</typeparam>
        /// <param name="accessIndex">Indexes to access the data</param>
        /// <returns>Sorted Data</returns>
        private IEnumerable<int> Sort<T>(IEnumerable<IEnumerable<int>> accessIndex)
        {
            var sort = accessIndex.Select((x, index) => new {Data = this[x.ToArray()], Index = index});
            return sort.OrderBy(a => a.Data).Select(a => a.Index);
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
