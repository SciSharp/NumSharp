using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.Sparse
{
    public class sparse
    {
        public csr_matrix diags(NDArray[] diagonals, int[] offsets = null, Shape shape = null, string format = null, Type dtype = null)
        {
            var (m, n) = shape.BiShape;
            var data_arr = np.zeros(m);

            for (int j = 0; j < diagonals.Length; j++)
            {
                var diagonal = diagonals[j];
            }

            return null;
        }
    }
}
