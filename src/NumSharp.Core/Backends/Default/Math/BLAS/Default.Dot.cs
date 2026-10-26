using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {

        /// <summary>
        ///     <c>np.dot</c>'s whole call (<see cref="TensorEngine.Dot(NDArray, NDArray, NDArray)"/>): the text operands'
        ///     answers with <paramref name="out"/> in NumPy's order, the product (<see cref="DotProduct"/>), then
        ///     <c>new_array_for_sum</c>'s strict out= check and the copy into <paramref name="out"/>.
        /// </summary>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        /// <param name="out">The out array, or null.</param>
        /// <returns>The product, or <paramref name="out"/> after the copy.</returns>
        /// <exception cref="ValueError">An out array the strict check refuses (the two texts of
        ///     <see cref="TensorEngine.Dot(NDArray, NDArray, NDArray)"/>).</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.dot.html</remarks>
        public override NDArray Dot(NDArray left, NDArray right, NDArray @out = null)
        {
            // Text: NumPy's alignment, out= and item-size answers, in its order (the product would report the item size
            // before looking at out=).
            Strings.NDStringUfuncs.ThrowIfProduct(left, right, @out);
            var result = DotProduct(left, right);
            if (@out is null)
                return result;

            // Port of NumPy's new_array_for_sum (numpy/_core/src/multiarray/common.c): the ndim,
            // dtype and C-array checks fire together as one message; only then is each dimension
            // compared. ISCARRAY = C-contiguous AND writeable (ALIGNED is always true here).
            if (@out.ndim != result.ndim
                || @out.typecode != result.typecode
                || !@out.Shape.IsContiguous
                || !@out.Shape.IsWriteable)
                throw new ValueError(
                    "output array is not acceptable (must have the right datatype, " +
                    "number of dimensions, and be a C-Array)");

            for (int d = 0; d < result.ndim; d++)
                if (result.Shape.dimensions[d] != @out.Shape.dimensions[d])
                    throw new ValueError("output array has wrong dimensions");

            // dtype and shape already validated equal, and the product is freshly computed (it cannot overlap out), so
            // this is np.copyto(out, result, "unsafe")'s unmasked copy: NDIter.Copy, with no call up into np.*.
            NDIter.Copy(@out, result);
            return @out;
        }

        /// <summary>
        ///     The product of <c>np.dot</c> without out=: the BLAS backend when one takes the call, else the shape routes
        ///     (a multiply with a 0-d operand, the 2-D matrix product, the 1-D inner product, the N-D x 1-D contraction,
        ///     the 1-D x 2-D row product, the N-D x M-D sum product).
        /// </summary>
        /// <param name="left">The first operand.</param>
        /// <param name="right">The second operand.</param>
        /// <returns>A fresh product.</returns>
        /// <exception cref="IncorrectShapeException">Misaligned operands.</exception>
        /// <exception cref="ArgumentException">Misaligned N-D x 1-D or 1-D x 2-D operands (NumPy's text).</exception>
        private NDArray DotProduct(NDArray left, NDArray right)
        {
            // A string operand: a 0-d one makes dot a multiply (its string loops answer), any other fails NumPy's
            // dot-function lookup ("data type must provide an itemsize"). Before the BLAS hook, which never takes text.
            Strings.NDStringUfuncs.ThrowIfProduct(left, right);
            //Dot product of two arrays.Specifically,
            //If both a and b are 1 - D arrays, it is inner product of vectors(without complex conjugation).
            //If both a and b are 2 - D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.
            //V If either a or b is 0 - D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a *b is preferred.
            //If a is an N - D array and b is a 1 - D array, it is a sum product over the last axis of a and b.
            //If a is an N - D array and b is an M - D array(where M >= 2), it is a sum product over the last axis of a and the second-to - last axis of b:
            //  dot(a, b)[i, j, k, m] = sum(a[i, j,:] * b[k,:, m])
            // An external BLAS, when one is installed (see TensorEngine.Blas). np.dot is NOT
            // matmul's dispatcher — the two disagree on e.g. a strided matrix times a vector — so
            // a backend gets its own entry point here, ahead of every shape branch below.
            // Read the property ONCE: it is settable at any time from any thread, and testing it
            // then calling it would be a null-dereference every time a concurrent OpenBlasEngine.Disable()
            // landed between the two reads (measured: ~2% of products under a hostile flipper).
            var blas = Blas;
            NDArray blasResult;
            if (blas != null && blas.TryDot(left, right, out blasResult))
                return blasResult;

            var leftshape = left.Shape;
            var rightshape = right.Shape;
            var isLeftScalar = leftshape.IsScalar;
            var isRightScalar = rightshape.IsScalar;

            if (isLeftScalar && isRightScalar)
            {
                return Multiply(left, right);
            }

            //If either a or b is 0-D (scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a * b is preferred.
            // NumPy's own float/complex route differs in the sign of a zero: cblas_matrixproduct treats a size-1
            // operand as a scalar and, for a result longer than one element, runs ?axpy into a ZERO-FILLED output,
            // so np.dot(-0.0, [1, 1]) is [+0, +0] where this multiply gives [-0, -0]; a (1,1)·(1,1) pair is the
            // plain product there but a zero-started sum in MultiplyMatrix below. Open gap, pinned by
            // OpenBugsReductionTests.Dot_SizeOneOperand_FollowsNumPysScalarBranch_OnTheManagedPath.
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
            // Fused single-pass kernel (no temp product array) — see Default.Dot.Fused.cs.
            if (leftshape.NDim == 1 && rightshape.NDim == 1)
            {
                return DotInner1D(left, right);
            }

            //If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
            // This is exactly matmul's contraction (a's last axis with the promoted column vector), so
            // route through Matmul — it preserves the input dtype (NumPy: dot(int32, int32) -> int32)
            // and contracts the correct (last) axis, where np.sum used the wider accumulator (int64)
            // and a hard-coded axis=1 (wrong for N-D).
            if (leftshape.NDim >= 2 && rightshape.NDim == 1)
            {
                // np.dot reports a mismatch in its own words (cblasfuncs.c / multiarraymodule.c
                // dot_alignment_error). Matmul's check words the same condition as a gufunc core-dimension
                // mismatch, so test here, before the delegation.
                long contracted = leftshape.dimensions[leftshape.NDim - 1];
                if (contracted != rightshape.dimensions[0])
                    throw new ArgumentException(
                        $"shapes {leftshape.ToPythonTuple()} and {rightshape.ToPythonTuple()} not aligned: " +
                        $"{contracted} (dim {leftshape.NDim - 1}) != {rightshape.dimensions[0]} (dim 0)");
                return Matmul(left, right);
            }
            // If a is 1-D and b is 2-D, treat a as row vector and do matrix multiply, then squeeze
            // (n,) dot (n, m) -> (m,)
            if (leftshape.NDim == 1 && rightshape.NDim == 2)
            {
                if (leftshape[0] != rightshape[0])
                    throw new ArgumentException($"shapes ({leftshape[0]},) and ({rightshape[0]},{rightshape[1]}) not aligned: {leftshape[0]} (dim 0) != {rightshape[0]} (dim 0)");

                // Reshape 1D to row vector (1, n), multiply, squeeze back to 1D
                var leftReshaped = left.reshape(1, leftshape[0]);
                var result = MultiplyMatrix(leftReshaped, right);
                leftReshaped.Dispose(); // row-vector alias of the input — the input keeps its own ref

                // NumPy's dot result OWNS its data. The (1, m) product is a fresh C-contiguous buffer that only
                // `result` wraps, so drop the unit axis by relabelling its shape in place: no second array
                // object, and the vector stays the owner. Any other product (a view an engine shortcut handed
                // back) keeps the reshape-view route.
                var productShape = result.Shape;
                if (result.Storage.OwnsData && productShape.IsContiguous && productShape.offset == 0)
                {
                    result.Storage.Reshape(Shape.Vector(rightshape[1]));
                    return result;
                }

                var squeezed = result.reshape(rightshape[1]);
                result.Dispose();       // the returned view holds its own counted ref on the product buffer
                return squeezed;
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
    }
}
