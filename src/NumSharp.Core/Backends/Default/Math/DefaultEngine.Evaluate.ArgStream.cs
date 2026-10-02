using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

// =============================================================================
// DefaultEngine.Evaluate.ArgStream.cs — streamed np.evaluate(NDExpr.ArgMax / ArgMin(...))
// (plan docs/plans/ndexpr-evaluate.md, perf review lever "ArgMax/ArgMin")
// =============================================================================
//
// The fused ArgMax / ArgMin used to MATERIALIZE the whole child (an N-element array written to memory) and hand
// it to the engine argmax: two extra memory passes over the child, plus a temp allocation — on top of an engine
// axis argmax that was itself a per-output scalar loop. The engine half of this lever
// (Default.Reduction.ArgAxis.Fast.cs) made the engine fold fast; this half removes the temp:
//
//   * a BARE LEAF child (`ArgMax((NDExpr)a)`) is the operand itself, so the engine argmax runs on it directly —
//     every layout, no copy;
//   * a COMPUTED child over all-C-contiguous operands is produced block by block into L1 scratch through its own
//     fused kernel (the M3 NDExprChildStream) and folded as it arrives:
//       - flat, or the reduced axis innermost (ROWS): each row folds with the flat SIMD argmax kernel of its
//         dtype; a row longer than the scratch folds chunk by chunk, the chunk winners combined in increasing
//         order with the SAME strict "replace only when better" rule, stopping once the answer can no longer
//         change (a NaN, an integer extreme, a Boolean deciding byte);
//       - an outer axis (SLABS): the engine's lane fold (SeedSlabChunk / FoldSlabRows) fed with the produced rows.
//
// Exact by construction: an argmax is an index chosen by NumPy's first-occurrence rule, and every path applies that
// rule to the child's elements in logical C order — the order NumPy's argmax visits the C-contiguous array its own
// PyArray_ArgMax builds. Declines (null, nothing allocated) keep the materialize route, which stays correct: F /
// strided / broadcast / mixed-order operands (the result is C-contiguous and NumPy's index order is logical C, so a
// memory-order walk of anything but C would visit elements out of order), Half / Decimal / Complex children, empty
// children, and an out-of-range axis (the materialize route raises NumPy's error).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Computes <c>np.evaluate(NDExpr.ArgMax / ArgMin(child[, axis[, keepdims]]))</c> without materializing the child,
        /// or returns <see langword="null"/> — having allocated nothing — so the caller materializes as before.
        /// </summary>
        /// <param name="program">The reduction program (its <c>Reduce</c> is an ArgMax / ArgMin).</param>
        /// <param name="inputs">Every input of the call, in input order (iterator operands and hoisted parameters).</param>
        /// <param name="childNdim">The child's rank when a result is returned (the caller's flat-keepdims and out checks use it); 0 otherwise.</param>
        /// <returns>
        /// The same array the materialize route's engine call would return: for a flat reduction a fresh 0-d int64
        /// (<see cref="NDArray.Scalar{T}(T)"/>, keepdims left to the caller's <c>KeepdimsFlat</c>); for an axis reduction a
        /// fresh C-contiguous int64 array of the reduced shape, keepdims already applied and a 0-d result marked as a
        /// reduction scalar (exactly what <c>np.argmax(child, axis, keepdims)</c> hands back). Null when declined.
        /// </returns>
        /// <exception cref="Exception">
        /// Whatever the child kernel raises mid-stream (a <c>Call</c> node's delegate, …) propagates unchanged after the
        /// partially written result is disposed; every decline is decided before the first element is produced, so an
        /// exception never means "fall back".
        /// </exception>
        private unsafe NDArray TryStreamArgReduce(NDExprProgram program, NDArray[] inputs, out int childNdim)
        {
            childNdim = 0;
            var reduce = program.Reduce;
            bool isMax = reduce.Kind == NDExprReduceKind.ArgMax;
            var child = program.ChildElementwiseProgram;
            // Test hook: DisableStreamingReduce forces the materialize route, so a test can compare the two.
            if (child is null || NDExpr.DisableStreamingReduce || !IsArgStreamType(child.ResultType))
                return null;
            var ops = child.IteratorOperands(inputs);
            if (ops.Length == 0)
                return null;   // every input a hoisted 0-d parameter: the materialize route's 0-d child handles it

            // A bare LEAF child is the operand itself: the engine argmax (whose fast fold handles every layout) runs on
            // it directly — the materialize route would first copy it element for element into a fresh C array and
            // reduce THAT, the same values in the same logical order. The dtype check keeps any cast leaf off this path.
            if (child.Bound is InputNode && ops.Length == 1 && ops[0].typecode == child.ResultType)
            {
                var leaf = ops[0];
                int lnd = leaf.ndim;
                if (leaf.size == 0)
                    return null;   // the materialize route raises NumPy's empty-sequence error
                if (reduce.Axis is int la)
                {
                    int lax = la < 0 ? la + lnd : la;
                    if (lax < 0 || lax >= lnd)
                        return null;   // the materialize route raises NumPy's AxisError
                    childNdim = lnd;
                    NDExpr.StreamingReductions++;
                    return isMax ? np.argmax(leaf, lax, reduce.Keepdims) : np.argmin(leaf, lax, reduce.Keepdims);
                }

                childNdim = lnd;
                NDExpr.StreamingReductions++;
                return NDArray.Scalar(isMax ? np.argmax(leaf) : np.argmin(leaf));
            }

            // A computed child streams over all-C-contiguous operands only: there, flat memory index k IS logical C
            // index k — the order NumPy's argmax visits — and the reduced axis splits memory as [outer, axis, inner].
            if (!CanStreamChild(child, ops, allowF: false))
                return null;

            var dims = ops[0].Shape.dimensions;
            int nd = dims.Length;
            long n = ops[0].size;
            if (n == 0)
                return null;   // the materialize route raises NumPy's empty-sequence error

            long outer, rowLen, inner;
            Shape resultShape;
            int axis = -1;
            if (reduce.Axis is int a)
            {
                axis = a < 0 ? a + nd : a;
                if (axis < 0 || axis >= nd)
                    return null;   // the materialize route raises NumPy's AxisError
                outer = 1;
                inner = 1;
                for (int d = 0; d < axis; d++) outer *= dims[d];
                for (int d = axis + 1; d < nd; d++) inner *= dims[d];
                rowLen = dims[axis];
                var reducedDims = new long[nd - 1];
                for (int d = 0, rd = 0; d < nd; d++)
                    if (d != axis)
                        reducedDims[rd++] = dims[d];
                resultShape = reducedDims.Length > 0 ? new Shape(reducedDims) : Shape.NewScalar();
            }
            else
            {
                // Flat: the whole C-contiguous child is ONE row in logical order.
                outer = 1;
                rowLen = n;
                inner = 1;
                resultShape = Shape.NewScalar();
            }

            var stream = new NDExprChildStream { Kernel = child.Kernel };
            int nop = ops.Length;
            byte** bases = stackalloc byte*[nop];
            long* elemBytes = stackalloc long[nop];
            void** ptrs = stackalloc void*[nop + 1];
            long* strides = stackalloc long[nop + 1];
            BindChildStream(ref stream, ops, child.ResultType, bases, elemBytes, ptrs, strides);
            if (child.ParamCount > 0)
            {
                // The child is an ELEMENTWISE program: its hoisted parameters live in the elementwise aux layout (slot 0
                // onward). stackalloc memory lives until the METHOD returns, so this block-scoped buffer stays valid for
                // every Produce below.
                byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                child.PackParams(inputs, paramBlock);
                stream.Aux = paramBlock;
            }

            // Two L1 blocks: the produce target, and (slabs only) the lanes' running best values — the fold compares a
            // produced row against the best, so they cannot share one buffer.
            byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
            byte* best = stackalloc byte[AxisArgScratchBytes];
            stream.Scratch = scratch;
            stream.Block = EvaluateStreamScratchBytes / child.ResultType.SizeOf();

            childNdim = nd;
            NDExpr.StreamingReductions++;
            if (reduce.Axis is null)
            {
                // Flat: one row, one index — the materialize route wraps np.argmax's scalar the same way.
                long index;
                StreamArgDispatch(child.ResultType, isMax, ref stream, 1, rowLen, 1, &index, scratch, best);
                return NDArray.Scalar(index);
            }

            // Every output is written by the fold (rows write each output, slabs seed each lane), so the result is
            // allocated uninitialized; disposed on an exception so a failing child kernel does not strand its buffer.
            var result = new NDArray(NPTypeCode.Int64, resultShape, false);
            try
            {
                StreamArgDispatch(child.ResultType, isMax, ref stream, outer, rowLen, inner, (long*)result.Address, scratch, best);
            }
            catch
            {
                result.Dispose();
                throw;
            }

            if (reduce.Keepdims)
            {
                // np.argmax(child, axis, keepdims=True): the reduced axis kept as size 1.
                var kd = (long[])dims.Clone();
                kd[axis] = 1;
                result.Storage.Reshape(new Shape(kd));
            }

            // The engine marks a 0-d axis result (a 1-D child reduced over its only axis) as a reduction scalar.
            return result.MarkReductionScalar();
        }

        /// <summary>Whether the argmax stream serves <paramref name="t"/>: the dtypes the fast fold's rules cover.</summary>
        /// <param name="t">The child's result dtype.</param>
        /// <returns>True for Boolean, the eight integer widths, Char, Single and Double.</returns>
        private static bool IsArgStreamType(NPTypeCode t) => t switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Char or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                or NPTypeCode.Single or NPTypeCode.Double => true,
            _ => false,
        };

        /// <summary>Picks the <c>(T, rule)</c> instantiation of <see cref="StreamArg{T,TRule}"/> for the child dtype.</summary>
        /// <param name="childType">The child's result dtype (one <see cref="IsArgStreamType"/> accepts).</param>
        /// <param name="isMax">Argmax (<see langword="true"/>) or argmin.</param>
        /// <param name="stream">The bound child stream.</param>
        /// <param name="outer">Rows (inner == 1) or output slabs (inner &gt; 1).</param>
        /// <param name="rowLen">Elements along the reduced axis.</param>
        /// <param name="inner">Outputs per slab (1 = the reduced axis is innermost).</param>
        /// <param name="dst">The C-contiguous int64 result (outer × inner elements).</param>
        /// <param name="scratch">The produce target (<see cref="EvaluateStreamScratchBytes"/> bytes).</param>
        /// <param name="best">The slab fold's best-value block (<see cref="AxisArgScratchBytes"/> bytes).</param>
        /// <exception cref="Exception">Whatever the child kernel raises while producing a block.</exception>
        private static unsafe void StreamArgDispatch(NPTypeCode childType, bool isMax, ref NDExprChildStream stream,
            long outer, long rowLen, long inner, long* dst, byte* scratch, byte* best)
        {
            switch (childType)
            {
                case NPTypeCode.Boolean:
                    if (isMax)
                        StreamArg<byte, ArgMaxBoolRule>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    else
                        StreamArg<byte, ArgMinBoolRule>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    break;
                case NPTypeCode.Byte: StreamArgInt<byte>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.SByte: StreamArgInt<sbyte>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.Int16: StreamArgInt<short>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                // Char orders exactly like its UInt16 code unit.
                case NPTypeCode.UInt16:
                case NPTypeCode.Char: StreamArgInt<ushort>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.Int32: StreamArgInt<int>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.UInt32: StreamArgInt<uint>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.Int64: StreamArgInt<long>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.UInt64: StreamArgInt<ulong>(isMax, ref stream, outer, rowLen, inner, dst, scratch, best); break;
                case NPTypeCode.Single:
                    if (isMax)
                        StreamArg<float, ArgMaxFloatRule<float>>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    else
                        StreamArg<float, ArgMinFloatRule<float>>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    break;
                case NPTypeCode.Double:
                    if (isMax)
                        StreamArg<double, ArgMaxFloatRule<double>>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    else
                        StreamArg<double, ArgMinFloatRule<double>>(ref stream, outer, rowLen, inner, dst, scratch, best);
                    break;
            }
        }

        /// <summary>The integer-rule branch of <see cref="StreamArgDispatch"/>.</summary>
        /// <typeparam name="T">The integer lane type (Char arrives as <see cref="ushort"/>).</typeparam>
        /// <param name="isMax">Argmax (<see langword="true"/>) or argmin.</param>
        /// <param name="stream">The bound child stream.</param>
        /// <param name="outer">Rows or output slabs.</param>
        /// <param name="rowLen">Elements along the reduced axis.</param>
        /// <param name="inner">Outputs per slab.</param>
        /// <param name="dst">The int64 result.</param>
        /// <param name="scratch">The produce target.</param>
        /// <param name="best">The slab fold's best-value block.</param>
        /// <exception cref="Exception">Whatever the child kernel raises while producing a block.</exception>
        private static unsafe void StreamArgInt<T>(bool isMax, ref NDExprChildStream stream, long outer, long rowLen,
            long inner, long* dst, byte* scratch, byte* best)
            where T : unmanaged, IBinaryInteger<T>, IMinMaxValue<T>
        {
            if (isMax)
                StreamArg<T, ArgMaxIntRule<T>>(ref stream, outer, rowLen, inner, dst, scratch, best);
            else
                StreamArg<T, ArgMinIntRule<T>>(ref stream, outer, rowLen, inner, dst, scratch, best);
        }

        /// <summary>Routes a streamed argmax / argmin to its rows form (the reduced axis innermost) or its slabs form.</summary>
        /// <typeparam name="T">The lane type as produced.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="stream">The bound child stream.</param>
        /// <param name="outer">Rows (inner == 1) or output slabs.</param>
        /// <param name="rowLen">Elements along the reduced axis.</param>
        /// <param name="inner">Outputs per slab (1 = rows).</param>
        /// <param name="dst">The int64 result.</param>
        /// <param name="scratch">The produce target.</param>
        /// <param name="best">The slab fold's best-value block.</param>
        /// <exception cref="Exception">Whatever the child kernel raises while producing a block.</exception>
        private static unsafe void StreamArg<T, TRule>(ref NDExprChildStream stream, long outer, long rowLen, long inner,
            long* dst, byte* scratch, byte* best)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            if (inner == 1)
                StreamArgRows<T, TRule>(ref stream, outer, rowLen, dst, (T*)scratch);
            else
                StreamArgSlabs<T, TRule>(ref stream, outer, rowLen, inner, dst, (T*)scratch, (T*)best);
        }

        /// <summary>
        /// The reduced axis is contiguous in the child (or the reduction is flat — one row of every element): output
        /// <c>o</c> is the argmax / argmin of row <c>o</c>. Short rows are produced many per scratch block and each folded
        /// by the dtype's flat SIMD kernel; a row longer than the block is produced chunk by chunk.
        /// </summary>
        /// <typeparam name="T">The lane type as produced.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="stream">The bound child stream.</param>
        /// <param name="outer">Number of rows (= outputs).</param>
        /// <param name="rowLen">Row length (&gt; 0).</param>
        /// <param name="dst">The result — every output written, none read.</param>
        /// <param name="scratch">The produce target (<see cref="NDExprChildStream.Block"/> elements).</param>
        /// <remarks>
        /// A long row's chunk winners combine in increasing chunk order with the rule's strict <c>Better</c>: each chunk's
        /// winner is the first occurrence of that chunk's extreme (or its first NaN / deciding byte), so replacing only on a
        /// strict improvement keeps the earliest chunk on a tie — the global first occurrence. For Boolean the winner of a
        /// chunk with no deciding byte is index 0 with a non-deciding value, which <c>Better</c> never takes, so the
        /// "0 when none" answer survives. The row stops producing once its best is decided.
        /// </remarks>
        /// <exception cref="Exception">Whatever the child kernel raises while producing a block.</exception>
        private static unsafe void StreamArgRows<T, TRule>(ref NDExprChildStream stream, long outer, long rowLen,
            long* dst, T* scratch)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            long block = stream.Block;
            if (rowLen <= block)
            {
                long rowsPerBlock = block / rowLen;
                for (long o = 0; o < outer; o += rowsPerBlock)
                {
                    long r = Math.Min(rowsPerBlock, outer - o);
                    stream.Produce(o * rowLen, r * rowLen, (byte*)scratch);
                    T* row = scratch;
                    for (long j = 0; j < r; j++, row += rowLen)
                        dst[o + j] = rowLen < AxisArgRowSimdMin
                            ? ArgFoldStrided<T, TRule>(row, rowLen, 1)
                            : TRule.RowSimd(row, rowLen);
                }

                return;
            }

            for (long o = 0; o < outer; o++)
            {
                T best = default;
                long bestIndex = 0;
                for (long c = 0; c < rowLen; c += block)
                {
                    long m = Math.Min(block, rowLen - c);
                    stream.Produce(o * rowLen + c, m, (byte*)scratch);
                    long ci = m < AxisArgRowSimdMin ? ArgFoldStrided<T, TRule>(scratch, m, 1) : TRule.RowSimd(scratch, m);
                    T cv = scratch[ci];
                    // The first chunk seeds the row; later chunks replace only on a strict improvement (see remarks).
                    if (c == 0 || TRule.Better(cv, best))
                    {
                        best = cv;
                        bestIndex = c + ci;
                    }

                    if (TRule.Decided(best))
                        break;
                }

                dst[o] = bestIndex;
            }
        }

        /// <summary>
        /// An outer axis is reduced: output slab <c>o</c> (<paramref name="inner"/> outputs) folds the
        /// <paramref name="rowLen"/> input rows <c>(o·rowLen + k)·inner</c> lane by lane with the engine's slab fold. Short
        /// rows are produced many per scratch block; rows wider than one lane chunk are folded one chunk of lanes at a
        /// time, a row chunk per produce.
        /// </summary>
        /// <typeparam name="T">The lane type as produced.</typeparam>
        /// <typeparam name="TRule">The selection rule.</typeparam>
        /// <param name="stream">The bound child stream.</param>
        /// <param name="outer">Output slabs.</param>
        /// <param name="rowLen">Rows per slab (the reduced axis length, &gt; 0).</param>
        /// <param name="inner">Outputs per slab (&gt; 1).</param>
        /// <param name="dst">The C-contiguous result — every output seeded, none read before.</param>
        /// <param name="scratch">The produce target (<see cref="NDExprChildStream.Block"/> elements).</param>
        /// <param name="best">The lanes' running best values (<see cref="AxisArgScratchBytes"/> bytes).</param>
        /// <remarks>
        /// Rows arrive in increasing axis order within each lane chunk, which is what <see cref="FoldSlabRows"/>'s strict
        /// rule needs. A Boolean chunk stops producing once every lane is decided.
        /// </remarks>
        /// <exception cref="Exception">Whatever the child kernel raises while producing a block.</exception>
        private static unsafe void StreamArgSlabs<T, TRule>(ref NDExprChildStream stream, long outer, long rowLen,
            long inner, long* dst, T* scratch, T* best)
            where T : unmanaged where TRule : struct, IArgRule<T>
        {
            long block = stream.Block;
            // Lanes one chunk folds at a time — the engine's cap (best values + result slots stay in L1). Never more
            // than the produce block: both derive from 8 KB, and the chunk also caps at AxisArgMaxChunkLanes.
            long chunk = Math.Min(AxisArgScratchBytes / sizeof(T), AxisArgMaxChunkLanes);
            for (long o = 0; o < outer; o++)
            {
                long slab = o * rowLen * inner;   // flat index of element (o, 0, 0)
                long* dslab = dst + o * inner;
                if (inner <= chunk)
                {
                    // Every lane in one chunk: produce whole rows, many per block.
                    int lanes = (int)inner;
                    long rowsPerBlock = block / inner;
                    bool live = true;
                    for (long k = 0; k < rowLen && live; k += rowsPerBlock)
                    {
                        long kc = Math.Min(rowsPerBlock, rowLen - k);
                        stream.Produce(slab + k * inner, kc * inner, (byte*)scratch);
                        if (k == 0)
                        {
                            live = SeedSlabChunk<T, TRule>(scratch, 1, lanes, dslab, 1, best);
                            if (live && kc > 1)
                                live = FoldSlabRows<T, TRule>(scratch + inner, inner, 1, kc - 1, 1, lanes, dslab, 1, best);
                        }
                        else
                            live = FoldSlabRows<T, TRule>(scratch, inner, k, kc, 1, lanes, dslab, 1, best);
                    }
                }
                else
                {
                    // Wide rows: one chunk of lanes at a time, one row chunk per produce (≥ chunk elements, so the
                    // per-produce cost is amortized).
                    for (long l0 = 0; l0 < inner; l0 += chunk)
                    {
                        int lanes = (int)Math.Min(chunk, inner - l0);
                        bool live = true;
                        for (long k = 0; k < rowLen && live; k++)
                        {
                            stream.Produce(slab + k * inner + l0, lanes, (byte*)scratch);
                            live = k == 0
                                ? SeedSlabChunk<T, TRule>(scratch, 1, lanes, dslab + l0, 1, best)
                                : FoldSlabRows<T, TRule>(scratch, 0, k, 1, 1, lanes, dslab + l0, 1, best);
                        }
                    }
                }
            }
        }
    }
}
