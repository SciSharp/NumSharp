using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Returns a view of the array with axes transposed.
        /// For a 1-D array this has no effect, as a transposed vector is simply the same vector.To convert a 1-D array into a 2D column vector, an 
        /// additional dimension must be added. np.atleast2d(a).T achieves this, as does a[:, np.newaxis]. 
        /// For a 2-D array, this is a standard matrix transpose. For an n-D array, if axes are given, their 
        /// order indicates how the axes are permuted (see Examples). If axes are not provided 
        /// and a.shape = (i[0], i[1], ...i[n - 2], i[n - 1]), then a.transpose().shape = (i[n - 1], i[n - 2], ...i[1], i[0]).
        /// </summary>
        /// <param name="x">Input array</param>
        /// <param name="axes">None or no argument: reverses the order of the axes. 
        /// Tuple of ints: i in the j-th place in the tuple means a’s i-th axis becomes a.transpose()’s j-th axis.</param>
        /// <returns>View of a, with axes suitably permuted</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.transpose.html?highlight=transpose#numpy.ndarray.transpose</remarks>
        public override NDArray Transpose(NDArray x, int[] axes = null)
        {
            if (x == null)
                throw new ArgumentNullException(nameof(x));

            NDArray nd;

            if (x.ndim == 1)
            {
                // Returns a copy of the original by definition
                nd = new NDArray(x.Array, x.shape);
            }
            else if (x.ndim == 2)
            {
                if (axes != null)
                {
                    throw new NotImplementedException("Axis specification for Transpose is not yet supported");
                }

                nd = new NDArray(x.typecode, x.shape.Reverse().ToArray());
                NDArray src = x;
                if (src.Shape.IsSliced)
                {
                    // Work with a contiguous array. This incurs the cost of
                    // a copy, but that may not be worse than the computational
                    // complexity of the slice iterator for this particular job.
                    src = x.copy();
                }
                unsafe
                {
                    // Transpose can be done without typed operations since there is no math
                    Unmanaged.IArraySlice data = nd.GetData();
                    Unmanaged.IArraySlice srcData = src.GetData();
                    int row = 0;
                    int col = 0;
                    int rowMax = nd.shape[0];
                    int rowStride = nd.shape[1];
                    for (int i = 0; i < data.Count; i++)
                    {
                        data[row * rowStride + col] = srcData[i];
                        row++;
                        if (row == rowMax)
                        {
                            row = 0;
                            col++;
                        }
                    }
                }
            }
            else
            {
                throw new NotImplementedException();
            }

            return nd;
        }
    }
}
