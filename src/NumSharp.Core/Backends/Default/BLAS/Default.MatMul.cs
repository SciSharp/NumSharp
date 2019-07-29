using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html</remarks>
        public override NDArray Matmul(in NDArray lhs, in NDArray rhs)
        {
            var leftshape = lhs.Shape;
            var rightshape = rhs.Shape;
            if (leftshape.IsScalar || rightshape.IsScalar)
                throw new InvalidOperationException("Matmul can't handle scalar multiplication, use `*` or `np.dot(..)` instead");

            var ndimLeft = leftshape.NDim;
            var ndimright = rightshape.NDim;

            //If both arguments are 2-D they are multiplied like conventional matrices.
            if (ndimLeft == 2 && ndimright == 2)
            {
                return MultiplyMatrix(lhs, rhs);
            }

            //todo If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

            //If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
            if ((ndimLeft == 2 && ndimright == 1))
            {
                rhs.Shape = ExpandEndDim(rightshape);
                NDArray ret = null;
                try
                {
                    ret = MultiplyMatrix(lhs, rhs);
                    return ret;
                }
                finally
                {
                    rhs.Shape = rightshape;
                    // ReSharper disable once PossibleNullReferenceException
                    ret.Shape = rightshape;
                }
            }

            //If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
            if (ndimLeft == 1 && ndimright == 2)
            {
                throw new NotSupportedException("Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?)");
            }


            return null;
        }
    }
}
