using System;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static long searchsorted(NDArray a, int v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);
            return BinarySearch(a, (double)v, side == "left", sorter);
        }

        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static long searchsorted(NDArray a, double v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);
            return BinarySearch(a, v, side == "left", sorter);
        }

        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array <paramref name="a"/> such that, if the corresponding elements
        /// in <paramref name="v"/> were inserted before the indices, the order of <paramref name="a"/> would be preserved.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Values to insert into <paramref name="a"/>. May be a scalar or any shape.</param>
        /// <param name="side">If "left" (default), the index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Array of insertion points with the same shape as <paramref name="v"/>, or a scalar if <paramref name="v"/> is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static NDArray searchsorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);

            bool leftSide = side == "left";

            // Scalar v -> scalar output.
            if (v.Shape.IsScalar)
            {
                double target = Converts.ToDouble(v.Storage.GetAtIndex(0));
                long idx = BinarySearch(a, target, leftSide, sorter);
                return NDArray.Scalar(idx);
            }

            // Build a fresh contiguous output shape with v's dimensions (drop v's offset/strides).
            Shape outShape = new Shape(v.shape);

            // Empty v -> empty output preserving v's shape.
            if (v.size == 0)
                return new NDArray(NPTypeCode.Int64, outShape, false);

            // Multi-dim v -> result preserves v's shape.
            NDArray output = new NDArray(NPTypeCode.Int64, outShape, false);
            unsafe
            {
                long* outPtr = (long*)output.Address;
                for (long i = 0; i < v.size; i++)
                {
                    double target = Converts.ToDouble(v.Storage.GetAtIndex(i));
                    outPtr[i] = BinarySearch(a, target, leftSide, sorter);
                }
            }

            return output;
        }

        private static void ValidateSearchSorted(NDArray a, string side, NDArray sorter)
        {
            if (side != "left" && side != "right")
                throw new ArgumentException($"search side must be 'left' or 'right' (got '{side}')", nameof(side));

            // NumPy: "object too deep for desired array" for ndim > 1.
            if (a.ndim > 1)
                throw new ArgumentException("object too deep for desired array", nameof(a));

            if (sorter is not null)
            {
                if (sorter.ndim != 1)
                    throw new ArgumentException("sorter must be 1-D array", nameof(sorter));
                if (sorter.size != a.size)
                    throw new ArgumentException("sorter.size must equal a.size", nameof(sorter));
            }
        }

        /// <summary>
        /// Binary search for the insertion position of <paramref name="target"/> in 1-D array <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">Sorted 1-D array (or unsorted with <paramref name="sorter"/> giving the sort order).</param>
        /// <param name="target">Target value to find position for.</param>
        /// <param name="leftSide">If true, returns the leftmost insertion index (NumPy side='left' / bisect_left).
        /// If false, returns the rightmost insertion index (NumPy side='right' / bisect_right).</param>
        /// <param name="sorter">Optional sort-order indices. When non-null, <paramref name="arr"/> is read via <c>arr[sorter[mid]]</c>.</param>
        /// <returns>Index where target should be inserted.</returns>
        /// <remarks>https://en.wikipedia.org/wiki/Binary_search_algorithm</remarks>
        private static long BinarySearch(NDArray arr, double target, bool leftSide, NDArray sorter)
        {
            long L = 0;
            long R = arr.size;
            while (L < R)
            {
                long m = (L + R) / 2;
                long readIdx = sorter is null ? m : Convert.ToInt64(sorter.Storage.GetAtIndex(m));
                // Converts.ToDouble handles all 15 dtypes including Half/Complex (System.Convert throws on those).
                double val = Converts.ToDouble(arr.Storage.GetAtIndex(readIdx));

                // bisect_left:  move right while val <  target  (returns first i with a[i] >= target)
                // bisect_right: move right while val <= target  (returns first i with a[i] >  target)
                bool moveRight = leftSide ? (val < target) : (val <= target);
                if (moveRight)
                    L = m + 1;
                else
                    R = m;
            }

            return L;
        }
    }
}
