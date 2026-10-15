using NumSharp.Backends.Kernels;
using NumSharp.Generic;

// =============================================================================
// DefaultEngine.CompareOp.Rows.cs — comparisons whose iteration has two levels,
// run without an iterator
// =============================================================================
//
// NumPy runs a ufunc without its iterator when the operands are "trivially
// iterable" (try_trivial_single_output_loop). The contiguous equal-shape and
// array-against-one-value forms take that bypass in
// TryTrivialContiguousComparisonOp. Everything else used to build an NDIter:
// a matrix against a row or a column, an outer product, a stepped or reversed
// view. The iterator costs about 0.45 us to construct and 15 ns for every
// inner-loop chunk it hands out, which for a (25, 40) operand is most of the
// call (1.8 us, where NumPy's whole call takes 1.7).
//
// Almost all of those iterations have the same simple structure: R runs of N
// elements, where each operand's runs are a fixed byte stride apart. That is
// what this file detects from the operands' strides, and ComparisonRun.RunRows
// executes: one vector-kernel call per run, no iterator.
//
// An iteration with more levels (a 3-D operand whose rows of rows are stepped),
// an F-ordered result, narrow rows (other than a matrix against one row) and
// every out= / where= call keep the iterator route.
//
// RunRows has two forms of its own for whole iterations: an operand whose rows
// are adjacent elements (an F-ordered matrix beside a C-ordered one) is copied
// a tile of several rows at a time, and a matrix against one short row compares
// several rows per kernel call. Both are described in ComparisonRun.cs.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>Highest result rank <see cref="TryRowsComparisonOp"/> analyses (its work arrays are on the stack).</summary>
        private const int RowsComparisonMaxRank = 32;

        /// <summary>
        ///     Runs <c>lhs op rhs</c> as R runs of N elements when the broadcast iteration of the two operands, in C
        ///     order, has that structure, and returns the fresh C-ordered Boolean result. Returns null, having
        ///     allocated nothing, when it does not (the caller's iterator route then serves the call).
        /// </summary>
        /// <param name="lhs">Left operand.</param>
        /// <param name="rhs">Right operand.</param>
        /// <param name="op">The comparison.</param>
        /// <param name="lhsType">Stored dtype of <paramref name="lhs"/>.</param>
        /// <param name="rhsType">Stored dtype of <paramref name="rhs"/>.</param>
        /// <returns>The result, or null to decline.</returns>
        /// <remarks>
        ///     <para><b>The structure.</b> Axes of extent 1 are dropped (they address one element whatever their
        ///     stride). On the remaining axes each operand has a stride per axis: its own, or 0 where it is stretched.
        ///     Two neighbouring axes are one run for an operand when the outer stride equals the inner stride times the
        ///     inner extent, which is the coalescing rule of NumPy's iterator. The iteration has two levels when the
        ///     axes split at some position <c>s</c> into a leading group and a trailing group that are each one run for
        ///     BOTH operands; the smallest such <c>s</c> gives the longest runs. <c>s == 0</c> is a single run (a
        ///     stepped 1-D array, <c>a[:, ::2]</c>, a fully broadcast operand). A 2-D pair always splits at 1.</para>
        ///     <para><b>Declined</b> (null): operand shapes that do not broadcast (the iterator route raises NumPy's
        ///     error), an empty result, a result NumPy lays out in F order (every array operand F-contiguous, see
        ///     <see cref="ShouldProduceFContigOutput(NDArray, NDArray, Shape)"/>), more levels than two, and many rows
        ///     of fewer than <see cref="ComparisonRun.DirectRowMinElements"/> elements whose inputs are contiguous or
        ///     one value each (the iterator's 2-D block kernel loops such rows inside one call), unless they are a
        ///     matrix against one row (<see cref="ComparisonRun.RepeatsRow"/>).</para>
        ///     <para>The result is fresh, so no operand overlaps it and the order of the runs cannot be observed.</para>
        /// </remarks>
        private unsafe NDArray<bool>? TryRowsComparisonOp(NDArray lhs, NDArray rhs, ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            var run = ComparisonRun.Get(op, lhsType, rhsType);
            if (run is null)
                return null;

            Shape ls = lhs.Shape, rs = rhs.Shape;
            int lRank = ls.NDim, rRank = rs.NDim;
            int rank = lRank > rRank ? lRank : rRank;
            if (rank == 0 || rank > RowsComparisonMaxRank)
                return null;

            long[] lDims = ls.dimensions, rDims = rs.dimensions, lStrides = ls.strides, rStrides = rs.strides;

            // Result dims (all axes), and per axis of extent > 1: the extent and each operand's element stride.
            long* dims = stackalloc long[rank];
            long* extent = stackalloc long[rank];
            long* lStep = stackalloc long[rank];
            long* rStep = stackalloc long[rank];
            int axes = 0;
            long size = 1;
            for (int d = 0; d < rank; d++)
            {
                // Operands are aligned at their LAST axis; a missing leading axis has extent 1.
                int li = d - (rank - lRank), ri = d - (rank - rRank);
                long lExtent = li >= 0 ? lDims[li] : 1;
                long rExtent = ri >= 0 ? rDims[ri] : 1;
                long dim;
                if (lExtent == rExtent)
                    dim = lExtent;
                else if (lExtent == 1)
                    dim = rExtent;
                else if (rExtent == 1)
                    dim = lExtent;
                else
                    return null;        // not broadcastable: the iterator route words NumPy's error
                dims[d] = dim;
                size *= dim;
                if (dim > 1)
                {
                    extent[axes] = dim;
                    // An operand stretched along this axis repeats one element: stride 0.
                    lStep[axes] = lExtent == dim ? lStrides[li] : 0;
                    rStep[axes] = rExtent == dim ? rStrides[ri] : 0;
                    axes++;
                }
            }
            // An empty result and a one-element result have their own rules upstream.
            if (size == 0 || axes == 0)
                return null;

            // NumPy lays the result out in F order when no array operand is C-contiguous-only and at least one is
            // F-contiguous-only; the iterator route allocates and fills that layout.
            if (rank > 1)
            {
                bool lArray = ls.size > 1, rArray = rs.size > 1;
                bool lOnlyF = lArray && ls.IsFContiguous && !ls.IsContiguous;
                bool rOnlyF = rArray && rs.IsFContiguous && !rs.IsContiguous;
                bool lOnlyC = lArray && ls.IsContiguous && !ls.IsFContiguous;
                bool rOnlyC = rArray && rs.IsContiguous && !rs.IsFContiguous;
                if ((lOnlyF || rOnlyF) && !lOnlyC && !rOnlyC)
                    return null;
            }

            // The smallest split whose two groups are each one run for both operands.
            int split = 0;
            for (; split < axes; split++)
            {
                if (IsOneRun(lStep, extent, 0, split) && IsOneRun(lStep, extent, split, axes)
                    && IsOneRun(rStep, extent, 0, split) && IsOneRun(rStep, extent, split, axes))
                    break;
            }
            if (split == axes)
                return null;            // three or more levels

            long rows = 1, count = 1;
            for (int a = 0; a < split; a++)
                rows *= extent[a];
            for (int a = split; a < axes; a++)
                count *= extent[a];

            int lItem = lhs.dtypesize, rItem = rhs.dtypesize;
            long lInner = lStep[axes - 1] * lItem, rInner = rStep[axes - 1] * rItem;
            long lOuter = split > 0 ? lStep[split - 1] * lItem : 0;
            long rOuter = split > 0 ? rStep[split - 1] * rItem : 0;

            // Many NARROW rows whose inputs are contiguous or one value each: the iterator's 2-D block kernel loops
            // them inside one call (about 1.5 ns a row plus 1 ns an element), which a kernel call per row (8 ns)
            // does not beat under DirectRowMinElements elements. Every other layout is faster here: its rows cost
            // the same per element on both routes, and this one builds no iterator. So is a matrix against ONE
            // row, however narrow: RunRows compares many of its rows per kernel call.
            if (rows > 1 && count < ComparisonRun.DirectRowMinElements
                && (lInner == lItem || lInner == 0) && (rInner == rItem || rInner == 0)
                && !run.RepeatsRow(lInner, lOuter, rInner, rOuter, count, rows, count))
                return null;

            // An operand that already has the result's dims lends its Shape when that Shape is a clean C layout
            // (no dims copy, no stride and flag computation); otherwise the dims are copied off the stack.
            Shape resultShape;
            if (lRank == rank && ls.size == size)
                resultShape = CanonicalResultShape(ls, false);
            else if (rRank == rank && rs.size == size)
                resultShape = CanonicalResultShape(rs, false);
            else
            {
                var resultDims = new long[rank];
                for (int d = 0; d < rank; d++)
                    resultDims[d] = dims[d];
                resultShape = Shape.Adopt(resultDims);
            }

            // Every one of the size slots is written by RunRows, so the buffer is taken as it comes.
            var result = new NDArray<bool>(resultShape, false);
            run.RunRows(
                (byte*)lhs.Address + ls.offset * lItem, lInner, lOuter,
                (byte*)rhs.Address + rs.offset * rItem, rInner, rOuter,
                (byte*)result.Address, 1, count,
                rows, count);
            return result;
        }

        /// <summary>
        ///     Whether the axes <c>[from, to)</c> of one operand are a single run: each axis' stride equals the stride
        ///     of the axis inside it times that axis' extent (an empty or one-axis group always is).
        /// </summary>
        /// <param name="step">The operand's element stride per axis (0 where it is stretched).</param>
        /// <param name="extent">The result's extent per axis.</param>
        /// <param name="from">First axis of the group.</param>
        /// <param name="to">One past the last axis of the group.</param>
        /// <returns>True when the group coalesces into one stride, that of its last axis.</returns>
        private static unsafe bool IsOneRun(long* step, long* extent, int from, int to)
        {
            for (int a = from; a < to - 1; a++)
            {
                if (step[a] != step[a + 1] * extent[a + 1])
                    return false;
            }
            return true;
        }
    }
}
