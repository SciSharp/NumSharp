using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray searchsorted(NDArray a, NDArray v)
        {
            // TODO currently no support for multidimensional a
            NDArray output = new int[v.shape[0]];
            for (int i = 0; i < v.size; i++)
            {
                double target = v[i];
                int idx = binarySearchRightmost(a, target);
                output[i] = idx;
            }
            return output;
        }

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
                val = arr[m];
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
