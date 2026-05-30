using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;
using NumSharp.Utilities.Maths;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Matrix product matching NumPy's <c>matmul</c> gufunc with signature
        /// <c>(n?,k),(k,m?)-&gt;(n?,m?)</c> — the <c>?</c> dims are the optional ones inserted for a
        /// 1-D operand:
        /// <list type="bullet">
        ///   <item>both 1-D → inner product (0-D scalar);</item>
        ///   <item>1-D × 2-D → the 1-D side is prepended a 1, then that dim is removed;</item>
        ///   <item>2-D × 1-D → the 1-D side is appended a 1, then that dim is removed;</item>
        ///   <item>≥3-D → batched: the leading "stack" dims broadcast and a 2-D matmul runs on the
        ///         trailing <c>[n,k]·[k,m]</c> of each batch element.</item>
        /// </list>
        /// 0-D (scalar) operands are rejected (NumPy raises — use <c>*</c> or <c>np.dot</c>).
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.matmul.html</remarks>
        public override NDArray Matmul(NDArray lhs, NDArray rhs)
        {
            if (lhs.ndim == 0 || rhs.ndim == 0)
                throw new InvalidOperationException(
                    "matmul: Input operand does not have enough dimensions (a 0-D / scalar operand " +
                    "is not allowed by the gufunc signature (n?,k),(k,m?)->(n?,m?); use `*` or `np.dot`).");

            // NumPy 1-D promotion: a 1-D lhs is treated as a row (prepend 1: [k] -> [1,k]); a 1-D rhs
            // as a column (append 1: [k] -> [k,1]). The inserted dim is squeezed back out at the end.
            bool lhsWas1D = lhs.ndim == 1;
            bool rhsWas1D = rhs.ndim == 1;
            if (lhsWas1D) lhs = np.expand_dims(lhs, 0);
            if (rhsWas1D) rhs = np.expand_dims(rhs, 1);

            long K = lhs.shape[lhs.ndim - 1];
            long Kr = rhs.shape[rhs.ndim - 2];
            if (K != Kr)
                throw new IncorrectShapeException(
                    $"matmul: Input operand core dimension mismatch (n?,k),(k,m?)->(n?,m?): " +
                    $"{K} (lhs last axis) != {Kr} (rhs second-to-last axis).");

            NDArray result = (lhs.ndim == 2 && rhs.ndim == 2)
                ? MultiplyMatrix(lhs, rhs)
                : BatchedMatmul(lhs, rhs);

            // Remove the dimensions that were inserted for 1-D operands. NumPy order: drop the
            // prepended lhs dim (-2) first, then the appended rhs dim (-1) — recomputing ndim between
            // the two squeezes keeps the axis indices valid (both inserted dims have size 1).
            if (lhsWas1D) result = np.squeeze(result, result.ndim - 2);
            if (rhsWas1D) result = np.squeeze(result, result.ndim - 1);

            return result;
        }

        /// <summary>
        /// Stacked / batched matmul: both operands are ≥2-D (after any 1-D promotion). The trailing
        /// two axes are the matrix dims <c>[n,k]·[k,m]</c>; the leading "stack" axes broadcast against
        /// each other (NumPy treats matmul as a gufunc over those stack dims). Broadcasts ONLY the
        /// stack dims — the matrix dims are left intact — then runs the 2-D kernel per batch element.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        protected static NDArray BatchedMatmul(NDArray lhs, NDArray rhs)
        {
            long n = lhs.shape[lhs.ndim - 2];
            long k = lhs.shape[lhs.ndim - 1];
            long m = rhs.shape[rhs.ndim - 1];

            // Stack (batch) dims = every axis except the trailing matrix pair.
            long[] batchA = TakeDims(lhs.Shape, lhs.ndim - 2);
            long[] batchB = TakeDims(rhs.Shape, rhs.ndim - 2);
            long[] batch = BroadcastStackDims(batchA, batchB);

            // Broadcast each operand's stack dims up to the common batch, keeping its matrix dims.
            var lhsFull = new Shape(batch.Concat(new[] { n, k }).ToArray());
            var rhsFull = new Shape(batch.Concat(new[] { k, m }).ToArray());
            var lhsB = np.broadcast_to(lhs, lhsFull);
            var rhsB = np.broadcast_to(rhs, rhsFull);

            var resultType = np._FindCommonArrayType(lhs.GetTypeCode, rhs.GetTypeCode);
            var resultShape = new Shape(batch.Concat(new[] { n, m }).ToArray());
            var ret = new NDArray(resultType, resultShape);

            // Iterate the batch coordinates; each integer-index slice is a 2-D [n,k]·[k,m]->[n,m].
            var iterShape = new Shape(batch.Length == 0 ? new long[] { 1 } : batch);
            var len = iterShape.size;
            var incr = new ValueCoordinatesIncrementor(ref iterShape);
            var index = incr.Index;
            for (long i = 0; i < len; i++, incr.Next())
                MultiplyMatrix(lhsB[index], rhsB[index], ret[index]);

            return ret;
        }

        /// <summary>Copy the first <paramref name="count"/> dimensions of <paramref name="shape"/> as long[].</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long[] TakeDims(Shape shape, int count)
        {
            var dims = new long[count];
            for (int i = 0; i < count; i++)
                dims[i] = shape.dimensions[i];
            return dims;
        }

        /// <summary>
        /// Broadcast two batch (stack) dimension lists right-aligned, NumPy rules: equal, or one is 1.
        /// </summary>
        private static long[] BroadcastStackDims(long[] a, long[] b)
        {
            int nd = Math.Max(a.Length, b.Length);
            var outDims = new long[nd];
            for (int i = 0; i < nd; i++)
            {
                long da = i < nd - a.Length ? 1 : a[i - (nd - a.Length)];
                long db = i < nd - b.Length ? 1 : b[i - (nd - b.Length)];
                if (da == db || da == 1 || db == 1)
                    outDims[i] = Math.Max(da, db);
                else
                    throw new IncorrectShapeException(
                        $"matmul: stacked (batch) dimensions are not broadcastable: {da} vs {db}.");
            }
            return outDims;
        }
    }
}
