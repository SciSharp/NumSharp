using NumSharp.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core.Sparse
{
    public partial class _cs_matrix
    {
        /// <summary>
        /// Sum the matrix over the given axis.  If the axis is None, sum over both rows and columns, returning a scalar.
        /// </summary>
        /// <param name="axis"></param>
        /// <returns></returns>
        public matrix sum(int? axis = null)
        {
            return spmatrix.sum(this, axis: 0, dtype: dtype);
        }
    }
}
