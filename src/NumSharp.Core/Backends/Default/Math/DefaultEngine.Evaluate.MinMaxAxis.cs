using System;
using System.Numerics;
using NumSharp.Backends.Iteration;

// =============================================================================
// DefaultEngine.Evaluate.MinMaxAxis.cs — NumPy-EXACT axis Min / Max for np.evaluate
// (plan docs/plans/ndexpr-evaluate.md, "Perf review 2026-09-23", axis min/max lever)
// =============================================================================
//
// np.max(child, axis) in NumPy reduces the array the child's ufunc chain materializes, with the per-element schedule
// Default.Reduction.MinMax.Exact.cs ports (contiguous ROW simd_reduce_c / strided-row 8-accumulator unroll / SLAB
// sequential fold). The fold path np.evaluate used before (seeded identity + 4-accumulator kernel) matched the VALUE
// but not those bits — which zero sign survives a ±0 tie, whether a NaN comes back canonical or with its payload —
// and missed 204 of 600 probed axis cases. Routes, each reducing exactly the buffer NumPy reduces:
//
//   * a bare leaf: the array itself, through the engine's exact axis core (every non-broadcast layout);
//   * a computed child whose operands all share C or F order: STREAMED — the child kernel evaluates rows or slabs
//     block by block into L1 scratch (NumPy's ufunc output would be C / F, so its reduce walks exactly these rows /
//     slabs), no temp;
//   * any other float child: materialized in the layout NumPy's own ufunc would allocate (NpyIter's K-order axis
//     permutation — MaterializeChildNumPyLayout), then reduced by the exact core over THAT layout.
//
// Integer children that do not stream keep the fold: an integer min/max is order-free, so it is already exact.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// The NumPy-exact AXIS <c>Min</c> / <c>Max</c> for np.evaluate: picks the route that reduces the same buffer,
        /// in the same per-element schedule, NumPy's <c>np.max(child, axis)</c> would (see the file header).
        /// </summary>
        /// <remarks>
        /// Declines (null, nothing allocated — the seeded axis fold runs as before, value-exact) for any other reduce
        /// kind, when <see cref="NDExpr.DisableExactMinMax"/> is set, for an empty axis or result (the fold's empty
        /// handling and NumPy's errors stay in charge), for a dtype outside <see cref="NumPyMinMaxReduce.Supports"/>,
        /// for a child with no iterator operand, for a broadcast bare leaf, and for a non-streamable INTEGER child
        /// (order-free, so the fold is exact and materializing would only cost). The result is a fresh array holding the
        /// child dtype (== the reduction's accumulator and result dtype for Min / Max); it may be F-contiguous (an
        /// F-walked stream, or an F input's engine result) — callers treat it as a dense block.
        /// </remarks>
        /// <param name="program">The axis reduction program.</param>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="kind">The reduce kind (only <see cref="NDExprReduceKind.Min"/> / <see cref="NDExprReduceKind.Max"/> engage).</param>
        /// <param name="axisSize">The reduced axis' extent.</param>
        /// <param name="reducedShape">The result shape (input shape with <paramref name="axis"/> removed; scalar for 1-D).</param>
        /// <returns>The fresh reduced array (the caller owns it), or null.</returns>
        private unsafe NDArray TryExactAxisMinMaxEval(NDExprProgram program, NDArray[] inputs, int axis, NDExprReduceKind kind,
            long axisSize, Shape reducedShape)
        {
            if ((kind != NDExprReduceKind.Min && kind != NDExprReduceKind.Max) || NDExpr.DisableExactMinMax)
                return null;
            if (axisSize == 0 || reducedShape.size == 0)
                return null;

            var child = program.ChildElementwiseProgram;
            var t = child.ResultType;
            if (!NumPyMinMaxReduce.Supports(t))
                return null;

            bool isMax = kind == NDExprReduceKind.Max;
            var ops = child.IteratorOperands(inputs);
            if (ops.Length == 0)
                return null;

            // NumPy reduces the array itself: the engine's exact core honors every non-broadcast layout through its
            // strides (null for a broadcast leaf → the fold).
            if (child.Bound is InputNode && ops.Length == 1)
                return TryExactAxisMinMax(ops[0], axis, isMax);

            if (CanStreamChild(child, ops, allowF: true))
            {
                var streamed = StreamExactAxisMinMax(child, inputs, ops, axis, isMax, t, reducedShape);
                NDExpr.StreamingReductions++;
                NDExpr.ExactMinMaxRuns++;
                return streamed;
            }

            if (!NumPyMinMaxReduce.IsFloat(t))
                return null;

            // A computed float child NumPy would materialize: materialize it in NumPy's own layout and reduce THAT —
            // the layout decides the schedule (which axis is innermost, whether it is contiguous). Both the logical view
            // and its dense buffer are released as soon as the reduction has read them.
            var view = MaterializeChildNumPyLayout(child, inputs, ops, out var buffer);
            try
            {
                return TryExactAxisMinMax(view, axis, isMax);
            }
            finally
            {
                if (!ReferenceEquals(view, buffer))
                    view.Dispose();
                buffer.Dispose();
            }
        }

        /// <summary>
        /// The streamed form of the exact axis reduction over a computed child whose operands all share C or F order:
        /// binds the child for streaming and reduces it in its memory walk — the C order of its dims, or for an all-F
        /// operand set the C order of its REVERSED dims — so that <c>outer · K · inner</c> is exactly the layout of the
        /// C / F buffer NumPy's ufunc would materialize, and each output is folded in NumPy's schedule for that buffer.
        /// </summary>
        /// <remarks>
        /// The result is fresh and dense in the SAME walk: C-contiguous, or F-contiguous for an F walk of a
        /// rank &gt;= 2 result, so output <c>(o, i)</c> sits at memory position <c>o · inner + i</c> either way. It is
        /// disposed if a kernel throws, so a failing reduction never strands its buffer.
        /// </remarks>
        /// <param name="child">The child program (validated by <see cref="CanStreamChild"/> with F allowed).</param>
        /// <param name="inputs">Every input of the call, in input order (for the parameter block).</param>
        /// <param name="ops">The child's iterator operands.</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="t">The child dtype (satisfies <see cref="NumPyMinMaxReduce.Supports"/>).</param>
        /// <param name="reducedShape">The result shape.</param>
        /// <returns>The fresh reduced array.</returns>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        private static unsafe NDArray StreamExactAxisMinMax(NDExprProgram child, NDArray[] inputs, NDArray[] ops, int axis,
            bool isMax, NPTypeCode t, Shape reducedShape)
        {
            var stream = new NDExprChildStream { Kernel = child.Kernel };
            int nop = ops.Length;
            byte** bases = stackalloc byte*[nop];
            long* elemBytes = stackalloc long[nop];
            void** ptrs = stackalloc void*[nop + 1];
            long* strides = stackalloc long[nop + 1];
            BindChildStream(ref stream, ops, t, bases, elemBytes, ptrs, strides);

            if (child.ParamCount > 0)
            {
                // Elementwise aux layout (slot 0 onward); stackalloc lives until the method returns.
                byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                child.PackParams(inputs, paramBlock);
                stream.Aux = paramBlock;
            }

            byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
            stream.Scratch = scratch;
            stream.Block = EvaluateStreamScratchBytes / t.SizeOf();

            // CanStreamChild accepted all-C or all-F; an operand that is not C-contiguous means the shared order is F
            // (an operand contiguous in BOTH orders fits either, so it never decides).
            bool fOrder = false;
            for (int j = 0; j < ops.Length; j++)
            {
                if (!ops[j].Shape.IsContiguous)
                {
                    fOrder = true;
                    break;
                }
            }

            var dims = ops[0].Shape.dimensions;
            int nd = dims.Length;
            long K = dims[axis];
            long outer = 1, inner = 1;
            if (!fOrder)
            {
                for (int d = 0; d < axis; d++) outer *= dims[d];
                for (int d = axis + 1; d < nd; d++) inner *= dims[d];
            }
            else
            {
                // Memory is the C order of the reversed dims: the axes AFTER `axis` are the slow (outer) ones.
                for (int d = axis + 1; d < nd; d++) outer *= dims[d];
                for (int d = 0; d < axis; d++) inner *= dims[d];
            }

            var result = fOrder && reducedShape.NDim > 1
                ? new NDArray(t, new Shape((long[])reducedShape.dimensions.Clone(), 'F'), false)
                : new NDArray(t, reducedShape, false);
            try
            {
                byte* dst = (byte*)result.Address;
                switch (t)
                {
                    case NPTypeCode.Double: if (isMax) StreamAxisMinMax<double, NumPyMinMaxReduce.MaxLane<double>>(ref stream, outer, K, inner, (double*)dst, scratch); else StreamAxisMinMax<double, NumPyMinMaxReduce.MinLane<double>>(ref stream, outer, K, inner, (double*)dst, scratch); break;
                    case NPTypeCode.Single: if (isMax) StreamAxisMinMax<float, NumPyMinMaxReduce.MaxLane<float>>(ref stream, outer, K, inner, (float*)dst, scratch); else StreamAxisMinMax<float, NumPyMinMaxReduce.MinLane<float>>(ref stream, outer, K, inner, (float*)dst, scratch); break;
                    case NPTypeCode.SByte: if (isMax) StreamAxisMinMax<sbyte, NumPyMinMaxReduce.MaxLane<sbyte>>(ref stream, outer, K, inner, (sbyte*)dst, scratch); else StreamAxisMinMax<sbyte, NumPyMinMaxReduce.MinLane<sbyte>>(ref stream, outer, K, inner, (sbyte*)dst, scratch); break;
                    case NPTypeCode.Byte: if (isMax) StreamAxisMinMax<byte, NumPyMinMaxReduce.MaxLane<byte>>(ref stream, outer, K, inner, dst, scratch); else StreamAxisMinMax<byte, NumPyMinMaxReduce.MinLane<byte>>(ref stream, outer, K, inner, dst, scratch); break;
                    case NPTypeCode.Int16: if (isMax) StreamAxisMinMax<short, NumPyMinMaxReduce.MaxLane<short>>(ref stream, outer, K, inner, (short*)dst, scratch); else StreamAxisMinMax<short, NumPyMinMaxReduce.MinLane<short>>(ref stream, outer, K, inner, (short*)dst, scratch); break;
                    case NPTypeCode.UInt16: if (isMax) StreamAxisMinMax<ushort, NumPyMinMaxReduce.MaxLane<ushort>>(ref stream, outer, K, inner, (ushort*)dst, scratch); else StreamAxisMinMax<ushort, NumPyMinMaxReduce.MinLane<ushort>>(ref stream, outer, K, inner, (ushort*)dst, scratch); break;
                    case NPTypeCode.Int32: if (isMax) StreamAxisMinMax<int, NumPyMinMaxReduce.MaxLane<int>>(ref stream, outer, K, inner, (int*)dst, scratch); else StreamAxisMinMax<int, NumPyMinMaxReduce.MinLane<int>>(ref stream, outer, K, inner, (int*)dst, scratch); break;
                    case NPTypeCode.UInt32: if (isMax) StreamAxisMinMax<uint, NumPyMinMaxReduce.MaxLane<uint>>(ref stream, outer, K, inner, (uint*)dst, scratch); else StreamAxisMinMax<uint, NumPyMinMaxReduce.MinLane<uint>>(ref stream, outer, K, inner, (uint*)dst, scratch); break;
                    case NPTypeCode.Int64: if (isMax) StreamAxisMinMax<long, NumPyMinMaxReduce.MaxLane<long>>(ref stream, outer, K, inner, (long*)dst, scratch); else StreamAxisMinMax<long, NumPyMinMaxReduce.MinLane<long>>(ref stream, outer, K, inner, (long*)dst, scratch); break;
                    case NPTypeCode.UInt64: if (isMax) StreamAxisMinMax<ulong, NumPyMinMaxReduce.MaxLane<ulong>>(ref stream, outer, K, inner, (ulong*)dst, scratch); else StreamAxisMinMax<ulong, NumPyMinMaxReduce.MinLane<ulong>>(ref stream, outer, K, inner, (ulong*)dst, scratch); break;
                    default: throw new NotSupportedException($"NumPy axis min/max schedule: dtype {t} is not served.");
                }

                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        /// The typed body of <see cref="StreamExactAxisMinMax"/> over the dense walk <c>outer · K · inner</c>:
        /// <list type="bullet">
        ///   <item><c>K == 1</c>: every output IS its one element — NumPy copies it (the first-visit copy, no loop), so
        ///         the child is produced straight into the result.</item>
        ///   <item>ROWS (<c>inner == 1</c>, the reduced axis is the buffer's innermost and contiguous): each output is
        ///         <see cref="NumPyMinMaxReduce.ReduceRow{T,TLane}"/> of its row — many short rows per produced block, or
        ///         a row longer than a block streamed through <see cref="StreamMinMax{T,TLane}"/> (one unbroken
        ///         <c>simd_reduce_c</c> schedule, as NumPy never splits a row).</item>
        ///   <item>SLABS (<c>inner &gt; 1</c>): slab 0 is produced straight into the output block (NumPy's first-visit
        ///         copy), then every further slab folds in with <see cref="NumPyMinMaxReduce.CombineRun{T,TLane}"/> in
        ///         increasing axis order — the elementwise sequential fold. Narrow slab sets are produced several slabs
        ///         per kernel call; a wide one block by block with the output block kept hot across the whole axis.</item>
        /// </list>
        /// </summary>
        /// <typeparam name="T">The child element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="stream">The bound stream.</param>
        /// <param name="outer">Product of the walk's axes outside the reduced one.</param>
        /// <param name="K">The reduced axis' extent (&gt;= 1).</param>
        /// <param name="inner">Product of the walk's axes inside the reduced one.</param>
        /// <param name="dst">The dense result (<c>outer · inner</c> elements, same walk).</param>
        /// <param name="scratch">The stream's scratch block.</param>
        private static unsafe void StreamAxisMinMax<T, TLane>(ref NDExprChildStream stream, long outer, long K, long inner,
            T* dst, byte* scratch)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            long block = stream.Block;
            if (K == 1)
            {
                // Child index (o·1 + 0)·inner + i == result index o·inner + i: produce straight into the result.
                ProduceBlocked(ref stream, 0, outer * inner, (byte*)dst, sizeof(T));
                return;
            }

            if (inner == 1)
            {
                if (K > block)
                {
                    for (long o = 0; o < outer; o++)
                        dst[o] = StreamMinMax<T, TLane>(ref stream, o * K, K, scratch);
                    return;
                }

                long rowsPer = block / K;
                for (long o0 = 0; o0 < outer; o0 += rowsPer)
                {
                    long rows = Math.Min(rowsPer, outer - o0);
                    stream.Produce(o0 * K, rows * K, scratch);
                    T* x = (T*)scratch;
                    for (long r = 0; r < rows; r++, x += K)
                        dst[o0 + r] = NumPyMinMaxReduce.ReduceRow<T, TLane>(x, K);
                }

                return;
            }

            if (inner <= block)
            {
                long slabsPer = block / inner;
                for (long o = 0; o < outer; o++)
                {
                    T* d = dst + o * inner;
                    stream.Produce(o * K * inner, inner, (byte*)d);   // slab 0: NumPy's first-visit copy
                    for (long j0 = 1; j0 < K; j0 += slabsPer)
                    {
                        long jc = Math.Min(slabsPer, K - j0);
                        stream.Produce((o * K + j0) * inner, jc * inner, scratch);
                        T* x = (T*)scratch;
                        for (long j = 0; j < jc; j++, x += inner)
                            NumPyMinMaxReduce.CombineRun<T, TLane>(d, 1, x, 1, inner);
                    }
                }

                return;
            }

            for (long o = 0; o < outer; o++)
            {
                for (long ib = 0; ib < inner; ib += block)
                {
                    long m = Math.Min(block, inner - ib);
                    T* d = dst + o * inner + ib;
                    stream.Produce(o * K * inner + ib, m, (byte*)d);
                    for (long k = 1; k < K; k++)
                    {
                        stream.Produce((o * K + k) * inner + ib, m, scratch);
                        NumPyMinMaxReduce.CombineRun<T, TLane>(d, 1, (T*)scratch, 1, m);
                    }
                }
            }
        }

        /// <summary>
        /// Materialize a computed child in the layout NumPy's own ufunc would allocate for it: a dense, positive-stride
        /// array whose memory axis order is NpyIter's K-order permutation over the child's operands
        /// (<see cref="NumPyIteratorAxisOrder"/> — <c>npyiter_new_temp_array</c> lays the output out in the iterator's
        /// axis order and never negates its strides, even on an axis the iterator flipped). When that is the layout
        /// <see cref="EvaluateCore"/> would pick on its own (C, or F for an all-strictly-F operand set) the plain fresh
        /// evaluation is returned; otherwise a C buffer over the permuted dims is allocated and the child is evaluated
        /// into its transposed view.
        /// </summary>
        /// <remarks>
        /// Why the layout matters: a reduction over the materialized child follows the child's STRIDES — for an axis
        /// min/max, whether the reduced axis is innermost and contiguous picks the schedule; for a flat one, the memory
        /// order IS the reduction order — so a child materialized in any other layout reduces in a different order and,
        /// for floats, can differ in a ±0 tie or a NaN payload. The caller owns both returned arrays: dispose
        /// <paramref name="buffer"/>, and the returned view as well when it is a different object.
        /// </remarks>
        /// <param name="child">The child elementwise program.</param>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="ops">The child's iterator operands (at least one).</param>
        /// <param name="buffer">Receives the dense buffer in memory order (the returned array itself, or its base).</param>
        /// <returns>The materialized child in logical axis order (a view over <paramref name="buffer"/> when permuted).</returns>
        private NDArray MaterializeChildNumPyLayout(NDExprProgram child, NDArray[] inputs, NDArray[] ops, out NDArray buffer)
        {
            Shape iter = ResolveInputShape(ops);
            int nd = iter.NDim;
            int[] mem = NumPyIteratorAxisOrder(ops, iter);
            bool heuristicF = AreAllInputsStrictFContig(ops, iter);

            bool isC = true, isF = true;
            for (int i = 0; i < nd; i++)
            {
                isC &= mem[i] == i;
                isF &= mem[i] == nd - 1 - i;
            }

            if ((isC && !heuristicF) || (isF && heuristicF))
            {
                buffer = EvaluateCore(child, inputs, null);
                return buffer;
            }

            var memDims = new long[nd];
            var axes = new int[nd];
            for (int i = 0; i < nd; i++)
            {
                memDims[i] = iter.dimensions[mem[i]];
                axes[mem[i]] = i;   // logical axis mem[i] is memory position i
            }

            buffer = new NDArray(child.ResultType, new Shape(memDims), false);
            NDArray view = null;
            try
            {
                view = np.transpose(buffer, axes);
                EvaluateCore(child, inputs, view);   // writes through the view into the buffer
                return view;
            }
            catch
            {
                // Nothing escapes on failure: release the view (if made) and the buffer before rethrowing.
                view?.Dispose();
                buffer.Dispose();
                buffer = null;
                throw;
            }
        }

        /// <summary>
        /// NpyIter's K-order axis permutation for an operand set — a port of <c>npyiter_find_best_axis_ordering</c>
        /// (numpy/_core/src/multiarray/nditer_constr.c): a stable insertion sort from reversed C order by |stride|, where
        /// each operand votes only when BOTH compared axes have a non-zero stride for it (a broadcast or extent-1 axis
        /// has stride 0 there), the first operand to decide sets the direction and, on a conflict between operands,
        /// C order wins. The allocated ufunc output does not vote (its strides do not exist yet).
        /// </summary>
        /// <remarks>
        /// Strides are compared in ELEMENTS: an operand's two strides are scaled by the same item size, so the per-operand
        /// comparison NumPy makes in bytes has the same outcome. Negative strides compare by magnitude (<c>intp_abs</c>);
        /// the sign only decides whether NpyIter flips the axis, which never changes the allocated output's layout.
        /// </remarks>
        /// <param name="ops">The operands (each broadcastable to <paramref name="iter"/>).</param>
        /// <param name="iter">The iteration (broadcast) shape.</param>
        /// <returns>The logical axes in memory order, OUTERMOST first (<c>[0, 1, …]</c> is C order).</returns>
        internal static int[] NumPyIteratorAxisOrder(NDArray[] ops, Shape iter)
        {
            int nd = iter.NDim;
            int nop = ops.Length;
            // st[d * nop + iop]: operand iop's stride on AXISDATA index d, which is logical axis nd-1-d (NpyIter keeps
            // its axis data in reversed C order); broadcast and extent-1 operand axes carry 0, as npyiter_fill_axisdata
            // sets them.
            var st = new long[Math.Max(nd * nop, 1)];
            for (int iop = 0; iop < nop; iop++)
            {
                var s = ops[iop].Shape;
                int off = nd - s.NDim;
                for (int d = 0; d < nd; d++)
                {
                    int oa = nd - 1 - d - off;
                    st[d * nop + iop] = oa < 0 || s.dimensions[oa] == 1 ? 0 : s.strides[oa];
                }
            }

            var perm = new int[nd];
            for (int d = 0; d < nd; d++)
                perm[d] = d;

            for (int i0 = 1; i0 < nd; i0++)
            {
                int ipos = i0;
                int j0 = perm[i0];
                for (int i1 = i0 - 1; i1 >= 0; i1--)
                {
                    bool ambig = true, shouldSwap = false;
                    int j1 = perm[i1];
                    for (int iop = 0; iop < nop; iop++)
                    {
                        long s0 = st[j0 * nop + iop], s1 = st[j1 * nop + iop];
                        if (s0 != 0 && s1 != 0)
                        {
                            if (Math.Abs(s1) <= Math.Abs(s0))
                                shouldSwap = false;   // set even when already decided: C order wins a conflict
                            else if (ambig)
                                shouldSwap = true;
                            ambig = false;
                        }
                    }

                    if (!ambig)
                    {
                        if (shouldSwap)
                            ipos = i1;
                        else
                            break;
                    }
                }

                if (ipos != i0)
                {
                    for (int i1 = i0; i1 > ipos; i1--)
                        perm[i1] = perm[i1 - 1];
                    perm[ipos] = j0;
                }
            }

            // perm[0] is the innermost axisdata index; memory order outermost-first reads it backwards, and axisdata index
            // d is logical axis nd-1-d.
            var mem = new int[nd];
            for (int i = 0; i < nd; i++)
                mem[i] = nd - 1 - perm[nd - 1 - i];
            return mem;
        }
    }
}
