using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {

        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html</remarks>
        public override NDArray Dot(in NDArray left, in NDArray right)
        {
            //Dot product of two arrays.Specifically,
            //If both a and b are 1 - D arrays, it is inner product of vectors(without complex conjugation).
            //If both a and b are 2 - D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            //V If either a or b is 0 - D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a *b is preferred.
            //If a is an N - D array and b is a 1 - D array, it is a sum product over the last axis of a and b.
            //If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the last axis of a and the second-to - last axis of b:
            //  dot(a, b)[i, j, k, m] = sum(a[i, j,:] * b[k,:, m])
            var leftshape = left.Shape;
            var rightshape = right.Shape;
            var isLeftScalar = leftshape.IsScalar;
            var isRightScalar = rightshape.IsScalar;

            if (isLeftScalar && isRightScalar)
            {
                return Multiply(left, right);
            }

            //If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
            if (isLeftScalar || isRightScalar)
            {
                return Multiply(left, right);
            }

            //If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            if (leftshape.NDim == 2 && rightshape.NDim == 2)
            {
                return MultiplyMatrix(left, right);
            }

            //If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
            if (leftshape.NDim == 1 && rightshape.NDim == 1)
            {
                Debug.Assert(leftshape[0] == rightshape[0]);
                return ReduceAdd(left * right, null, false);
            }

            //If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            if (leftshape.NDim >= 2 && rightshape.NDim == 1)
            {
                //TODO! this doesn't seem right, read desc
                //var right_broadcasted = new NDArray(right.Storage.Alias(np.broadcast_to(rightshape, leftshape)));
                return np.sum(left * right, axis: 1);
            }
            //todo!  ValueError: shapes (4,) and (2,4) not aligned: 4 (dim 0) != 2 (dim 0)

            if (leftshape.NDim == 1)
            {
                throw new NotSupportedException("lhs cannot be 1-D, use `np.multiply` or `*` for this case.");
            }

            //left cant be 0 or 1 by this point
            //If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:
            //dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
            if (rightshape.NDim >= 2)
            {
                return DotNDMD(left, right);
            }

            throw new NotSupportedException();
        }

        private static int[] ExpandStartDim(Shape shape)
        {
            var ret = new int[shape.NDim + 1];
            ret[0] = 1;
            Array.Copy(shape.dimensions, 0, ret, 1, shape.NDim);
            return ret;
        }

        private static Shape ExpandEndDim(Shape shape)
        {
            var ret = new int[shape.NDim + 1];
            ret[ret.Length - 1] = 1;
            Array.Copy(shape.dimensions, 0, ret, 0, shape.NDim);
            return new Shape(ret);
        }
    }
}
