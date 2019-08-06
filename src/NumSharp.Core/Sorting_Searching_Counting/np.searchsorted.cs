namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array a such that, if the corresponding elements in v were inserted before the indices, the order of a would be preserved.
        /// </summary>
        /// <param name="a">Input array. Must be sorted in ascending order.</param>
        /// <param name="v">Values to insert into a.</param>
        /// <returns>Array of insertion points with the same shape as v.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html</remarks>
        public static NDArray searchsorted(NDArray a, NDArray v)
        {
            // TODO currently no support for multidimensional a
            NDArray output = new int[v.shape[0]];
            for (int i = 0; i < v.size; i++)
            {
                double target = (double) v[i];
                int idx = binarySearchRightmost(a, target);
                output[i] = (NDArray) idx;
            }

            return output;
        }

        /// <summary>
        /// Find the (right-most) position of a target value within a sorted array.
        /// </summary>
        /// <param name="arr">Sorted array.</param>
        /// <param name="target">Target value to find position of.</param>
        /// <returns>Would-be index of target value within the array.</returns>
        /// <remarks>https://en.wikipedia.org/wiki/Binary_search_algorithm</remarks>
        private static int binarySearchRightmost(NDArray arr, double target)
        {
            // TODO should probably not work on NDArray? Does it make sense to do binary search on multidimensional arrays?
            int L = 0;
            int R = arr.size;
            int m;
            double val;
            while (L < R)
            {
                m = (L + R) / 2;
                val = (double) arr[m];
                if (val < target)
                {
                    L = m + 1;
                }
                else
                {
                    R = m;
                }
            }

            return L;
        }
    }
}
