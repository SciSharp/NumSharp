namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input array. Must be sorted in ascending order.</param>
        /// <param name="v">Value to insert into a.</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static int searchsorted(NDArray a, int v)
        {
            return binarySearchRightmost(a, v);
        }

        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input array. Must be sorted in ascending order.</param>
        /// <param name="v">Value to insert into a.</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static int searchsorted(NDArray a, double v)
        {
            return binarySearchRightmost(a, v);
        }

        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array a such that, if the corresponding elements in v were inserted before the indices, the order of a would be preserved.
        /// </summary>
        /// <param name="a">Input array. Must be sorted in ascending order.</param>
        /// <param name="v">Values to insert into a.</param>
        /// <returns>Array of insertion points with the same shape as v.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static NDArray searchsorted(NDArray a, NDArray v)
        {
            // TODO currently no support for multidimensional a

            // Handle scalar input - return scalar output
            if (v.Shape.IsScalar || v.size == 0)
            {
                if (v.size == 0)
                    return new NDArray(typeof(int), Shape.Vector(0), false);

                double target = v.GetDouble();
                int idx = binarySearchRightmost(a, target);
                return NDArray.Scalar(idx);
            }

            // Handle 1D array input
            NDArray output = new int[v.size];
            for (int i = 0; i < v.size; i++)
            {
                double target = v.GetDouble(i);  // Use GetDouble for proper value extraction
                int idx = binarySearchRightmost(a, target);
                output.SetInt32(idx, i);
            }

            return output;
        }

        /// <summary>
        /// Find the left-most position where target should be inserted to maintain order.
        /// This is equivalent to NumPy's searchsorted with side='left' (default).
        /// </summary>
        /// <param name="arr">Sorted array (1D).</param>
        /// <param name="target">Target value to find position for.</param>
        /// <returns>Index where target should be inserted.</returns>
        /// <remarks>https://en.wikipedia.org/wiki/Binary_search_algorithm</remarks>
        private static int binarySearchRightmost(NDArray arr, double target)
        {
            int L = 0;
            int R = arr.size;
            while (L < R)
            {
                int m = (L + R) / 2;
                double val = arr.GetDouble(m);  // Use GetDouble for proper value extraction
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
