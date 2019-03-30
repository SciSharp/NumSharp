using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Sparse
{
    public class sparse
    {
        public csr_matrix diags(NDArray[] diagonals, int[] offsets = null, Shape shape = null, string format = null, Type dtype = null)
        {
            int m = shape;
            var data_arr = np.zeros(m);

            for (int j = 0; j < diagonals.Length; j++)
            {
                var diagonal = diagonals[j];
            }

            return null;
        }
    }
}
