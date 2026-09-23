using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// DefaultEngine.Evaluate.Stream.cs — streaming (no-temp) exact reductions for np.evaluate
// (plan docs/plans/ndexpr-evaluate.md, Phase 2 M3)
// =============================================================================
//
// M1/M2 made the fused float Sum / Mean / Prod reductions NumPy-exact by MATERIALIZING the whole
// reduced child (an N-element array written to memory) and then reducing that buffer with NumPy's own
// schedule — pairwise_sum for add.reduce, a sequential fold for multiply.reduce. Exact, but it turned
// the fused reduction into three memory passes (write the child, read it back, plus the input reads):
// `np.evaluate(Sum(a*b))` measured barely faster than the unfused `np.sum(a*b)` (1.06x at 4M).
//
// The schedule NumPy runs depends only on the element COUNT, never on where the values live, so the
// temp is unnecessary. pairwise_sum(n) splits n at `n2 = n/2 - (n/2)%8` down to leaves; this file
// drives that SAME recursion from the host and, at each sub-range small enough to be cache-resident
// (EvaluateStreamScratchBytes), evaluates the fused child kernel into a stack scratch block and folds
// the block with the SAME cached pairwise fold every np.add.reduce kernel calls. Because the fold of a
// sub-range is by definition the value the full-array fold computes for that sub-tree, the result is
// bit-identical to materialize-then-reduce — by construction, not by tuning. The sequential product
// streams the same way (the fold order is just increasing index). The child values never leave L1:
// one read of each input, no N-element write, no N-element read back, no temp allocation.
//
// Scope (everything else keeps the materialize path, which stays correct):
//   * the reduction's iterator operands are CONTIGUOUS in one shared memory order with identical dims
//     (all C-contiguous, or — flat only — all F-contiguous), none broadcast. Then the flat memory index
//     of every operand names the same logical element, and that memory order IS the order the
//     materialized child would be reduced in (EvaluateCore allocates it C, or F exactly when every
//     operand is strictly F-contiguous — AreAllInputsStrictFContig). Strided / broadcast / mixed-order
//     inputs fall back.
//   * flat Sum/Mean over float32 / float64 / complex128 and Prod over float32 / float64 (the M1 divert
//     set), and the AXIS forms of the same set over all-C-contiguous inputs (the M2 divert set): the
//     reduced axis innermost reduces each row with the whole-row pairwise tree (NumPy's PINNED loop),
//     an outer axis accumulates slab after slab in increasing axis order (NumPy's SLAB loop) — both
//     verified against NumPy 2.4.2 on long rows (20K / 100K) and many short rows (50K x 3).
//
// NOTHING here may use an x86-only intrinsic without an IsSupported gate: the elementwise
// accumulation below is System.Numerics.Vector<T> (portable), and the folds are the existing kernels.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// The stack scratch budget, in bytes, a streaming reduction evaluates the fused child into per
        /// leaf (1024 float64 / 2048 float32 / 512 complex128 elements). Small enough to stay L1-resident
        /// beside the input streams — the whole point is that the child values never reach memory — and far
        /// above the pairwise leaf threshold (128 elements, 64 complexes), which the recursion requires.
        /// Measured on the fusion probe: 8 KB beat 4 KB at 4M and 16/32 KB at 100K–1M (the scratch then
        /// starts evicting the input streams).
        /// </summary>
        internal const int EvaluateStreamScratchBytes = 8192;

        /// <summary>
        /// Upper bound on the reduction's iterator operands for which the streaming path stack-allocates its
        /// per-operand pointer tables. A tree with more distinct array leaves than this is rare and simply
        /// takes the materialize path (no correctness consequence).
        /// </summary>
        private const int EvaluateStreamMaxOperands = 64;

        /// <summary>
        /// A contiguous fused child bound for streaming: the child's elementwise kernel plus everything
        /// needed to evaluate any flat sub-range of it into a caller buffer, and the typed pairwise folds the
        /// sum path reduces each block with. Every pointer field references memory stack-allocated by the
        /// method that owns the stream, so the struct is a <c>ref struct</c> and never outlives that frame.
        /// </summary>
        private unsafe ref struct NDExprChildStream
        {
            /// <summary>The child sub-tree's fused elementwise kernel over <c>[operands…, output]</c>.</summary>
            public NDInnerLoopFunc Kernel;

            /// <summary>Number of iterator operands (the kernel's output operand is slot <c>Nop</c>).</summary>
            public int Nop;

            /// <summary>Per operand, the address of its flat element 0 (base address + view offset).</summary>
            public byte** Bases;

            /// <summary>Per operand, its element size in bytes — the contiguous stride of its memory walk.</summary>
            public long* ElemBytes;

            /// <summary>The <c>Nop + 1</c> data-pointer table handed to the kernel, rewritten per produce.</summary>
            public void** Ptrs;

            /// <summary>The <c>Nop + 1</c> byte-stride table handed to the kernel (each operand's element size, then the output's).</summary>
            public long* Strides;

            /// <summary>The hoisted 0-d parameters in the elementwise aux layout (slot 0 onward), or null when there are none.</summary>
            public void* Aux;

            /// <summary>The <see cref="EvaluateStreamScratchBytes"/> stack block each leaf is evaluated into.</summary>
            public byte* Scratch;

            /// <summary>Capacity of <see cref="Scratch"/> in child elements — the largest range produced in one call.</summary>
            public long Block;

            /// <summary>The float64 pairwise fold (null unless the child is float64 and a sum is streamed).</summary>
            public PairwiseFoldDouble FoldDouble;

            /// <summary>The float32 pairwise fold (null unless the child is float32 and a sum is streamed).</summary>
            public PairwiseFoldSingle FoldSingle;

            /// <summary>The complex128 pairwise fold (null unless the child is complex128 and a sum is streamed).</summary>
            public PairwiseFoldComplex FoldComplex;

            /// <summary>
            /// Evaluate the child's elements with flat memory indices <c>[start, start + count)</c> into
            /// <paramref name="dst"/> (contiguous, the child dtype). Every operand shares one contiguous memory
            /// order, so element <c>start</c> of each operand is <c>Bases[j] + start·ElemBytes[j]</c> and the
            /// kernel's contiguous SIMD loop runs the whole range in one call.
            /// </summary>
            /// <param name="start">Flat memory index of the first element to produce.</param>
            /// <param name="count">Number of elements (≥ 1, ≤ the capacity of <paramref name="dst"/>).</param>
            /// <param name="dst">Destination for the <paramref name="count"/> child values.</param>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public void Produce(long start, long count, byte* dst)
            {
                for (int j = 0; j < Nop; j++)
                    Ptrs[j] = Bases[j] + start * ElemBytes[j];
                Ptrs[Nop] = dst;
                Kernel(Ptrs, Strides, count, Aux);
            }
        }

        /// <summary>
        /// Can the reduction child over <paramref name="ops"/> be streamed (evaluated range by range with raw
        /// pointer arithmetic)? Requires at least one operand (a tree with no streamed operand is the all-0-d
        /// case, which the fold path handles), a kernel, a bounded operand count, identical dimensions for
        /// every operand, no broadcast view, and ONE shared contiguous memory order — all C-contiguous, or
        /// (when <paramref name="allowF"/>) all F-contiguous. That shared order is what makes "flat memory
        /// index k" the same logical element in every operand AND the order the materialized child would be
        /// reduced in.
        /// </summary>
        /// <param name="child">The reduction's child elementwise program.</param>
        /// <param name="ops">The child's iterator operands (parameters excluded).</param>
        /// <param name="allowF">Whether an all-F-contiguous operand set qualifies (the flat reduce walks memory order; the axis reduce needs C order).</param>
        /// <returns>True when the streaming path applies.</returns>
        private static bool CanStreamChild(NDExprProgram child, NDArray[] ops, bool allowF)
        {
            // Test hook: force the materialize path so a test can compare the two byte for byte.
            if (NDExpr.DisableStreamingReduce)
                return false;
            if (child?.Kernel is null || ops.Length == 0 || ops.Length > EvaluateStreamMaxOperands)
                return false;

            var s0 = ops[0].Shape;
            bool allC = true, allF = allowF;
            for (int j = 0; j < ops.Length; j++)
            {
                var s = ops[j].Shape;
                // A stride-0 axis of extent > 1 re-reads memory, so a linear walk would be wrong even if a
                // contiguity flag happened to be set; exclude it explicitly (the trivial-loop routes do too).
                if (s.IsBroadcasted || !SameDims(s, s0))
                    return false;
                allC &= s.IsContiguous;
                allF &= s.IsFContiguous;
            }

            return allC || allF;
        }

        /// <summary>
        /// Fill the pointer tables of <paramref name="stream"/> for <paramref name="ops"/> and the child dtype
        /// <paramref name="childType"/>. The tables themselves are stack memory owned by the caller.
        /// </summary>
        /// <param name="stream">The stream being bound.</param>
        /// <param name="ops">The child's iterator operands (already validated by <see cref="CanStreamChild"/>).</param>
        /// <param name="childType">The child's result dtype — the scratch/output element type.</param>
        /// <param name="bases">Caller stack table of <c>ops.Length</c> base pointers.</param>
        /// <param name="elemBytes">Caller stack table of <c>ops.Length</c> element sizes.</param>
        /// <param name="ptrs">Caller stack table of <c>ops.Length + 1</c> kernel data pointers.</param>
        /// <param name="strides">Caller stack table of <c>ops.Length + 1</c> kernel byte strides.</param>
        private static unsafe void BindChildStream(ref NDExprChildStream stream, NDArray[] ops, NPTypeCode childType,
            byte** bases, long* elemBytes, void** ptrs, long* strides)
        {
            for (int j = 0; j < ops.Length; j++)
            {
                int size = ops[j].typecode.SizeOf();
                elemBytes[j] = size;
                strides[j] = size;
                // Logical element 0: a contiguous slice re-seats Address (offset 0), a strided parent keeps its
                // base with a non-zero offset — Address + offset·itemsize is right for both.
                bases[j] = (byte*)ops[j].Address + (long)ops[j].Shape.offset * size;
            }

            strides[ops.Length] = childType.SizeOf();
            stream.Nop = ops.Length;
            stream.Bases = bases;
            stream.ElemBytes = elemBytes;
            stream.Ptrs = ptrs;
            stream.Strides = strides;
        }

        /// <summary>
        /// Bind the typed pairwise fold <paramref name="childType"/> needs for a streamed SUM. Returns false
        /// when the host has no SIMD pairwise fold for it (dynamic codegen disabled, no SIMD) — the caller then
        /// keeps the materialize path, which reaches the same missing kernel and reports it its own way.
        /// </summary>
        /// <param name="stream">The stream receiving the fold.</param>
        /// <param name="childType">Double, Single or Complex.</param>
        /// <returns>True when the fold is bound.</returns>
        private static bool BindSumFold(ref NDExprChildStream stream, NPTypeCode childType)
        {
            switch (childType)
            {
                case NPTypeCode.Double:
                    stream.FoldDouble = ILKernelGenerator.TryGetPairwiseFoldDouble();
                    return stream.FoldDouble is not null;
                case NPTypeCode.Single:
                    stream.FoldSingle = ILKernelGenerator.TryGetPairwiseFoldSingle();
                    return stream.FoldSingle is not null;
                case NPTypeCode.Complex:
                    stream.FoldComplex = ILKernelGenerator.TryGetPairwiseFoldComplex();
                    return stream.FoldComplex is not null;
                default:
                    return false;
            }
        }

        /// <summary>
        /// Streaming form of the flat (axis=None) exact reduction the M1 divert runs: writes into
        /// <paramref name="accSlot"/> exactly what <see cref="PairwiseSumInto"/> / <see cref="SequentialProductInto"/>
        /// would write over the MATERIALIZED child — <c>*slot = *slot + pairwise_sum(child)</c> for a sum (the
        /// PINNED reduce contract, the slot pre-zeroed), <c>*slot = 1·x0·x1·…</c> for a product — without ever
        /// allocating the child. Returns false (writing nothing) when the child is not streamable, so the caller
        /// materializes as before.
        /// </summary>
        /// <param name="child">The reduction's child elementwise program (its result dtype is the reduction dtype).</param>
        /// <param name="inputs">The call's inputs (iterator operands and hoisted parameters, in input order).</param>
        /// <param name="n">Total element count of the child (&gt; 0).</param>
        /// <param name="sequentialProd">True for np.multiply.reduce (float32/float64), false for np.add.reduce.</param>
        /// <param name="accSlot">The pre-zeroed 16-byte accumulator slot the caller reads back.</param>
        /// <returns>True when the reduction was computed here; false when the caller must materialize.</returns>
        private static unsafe bool TryStreamFlatReduce(NDExprProgram child, NDArray[] inputs, long n, bool sequentialProd, byte* accSlot)
        {
            var childType = child.ResultType;
            if (sequentialProd ? childType != NPTypeCode.Double && childType != NPTypeCode.Single
                               : childType != NPTypeCode.Double && childType != NPTypeCode.Single && childType != NPTypeCode.Complex)
                return false;

            var ops = child.IteratorOperands(inputs);
            if (!CanStreamChild(child, ops, allowF: true))
                return false;

            var stream = new NDExprChildStream { Kernel = child.Kernel };
            if (!sequentialProd && !BindSumFold(ref stream, childType))
                return false;

            int nop = ops.Length;
            byte** bases = stackalloc byte*[nop];
            long* elemBytes = stackalloc long[nop];
            void** ptrs = stackalloc void*[nop + 1];
            long* strides = stackalloc long[nop + 1];
            BindChildStream(ref stream, ops, childType, bases, elemBytes, ptrs, strides);

            // Hoisted 0-d parameters, packed once in the ELEMENTWISE aux layout the child kernel reads (slot 0
            // onward — the child program is an elementwise program, not the reduce kernel with its acc slot).
            // stackalloc memory lives until the METHOD returns, so the block-scoped declaration is safe.
            if (child.ParamCount > 0)
            {
                byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                child.PackParams(inputs, paramBlock);
                stream.Aux = paramBlock;
            }

            byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
            stream.Scratch = scratch;
            stream.Block = EvaluateStreamScratchBytes / childType.SizeOf();
            NDExpr.StreamingReductions++;

            switch (childType)
            {
                case NPTypeCode.Double:
                    if (sequentialProd)
                        *(double*)accSlot = StreamProductDouble(ref stream, 0, n, 1d);
                    else
                        // The PINNED reduce writes *out + fold; the slot is the runtime zero, so the -0.0 of an
                        // all-negative-zero child becomes +0.0 exactly as it does over the materialized buffer.
                        *(double*)accSlot = *(double*)accSlot + StreamPairwiseDouble(ref stream, 0, n);
                    return true;

                case NPTypeCode.Single:
                    if (sequentialProd)
                        *(float*)accSlot = StreamProductSingle(ref stream, 0, n, 1f);
                    else
                        *(float*)accSlot = *(float*)accSlot + StreamPairwiseSingle(ref stream, 0, n);
                    return true;

                default: // Complex (sum only — the product was rejected above)
                    Vector128.Store(
                        Vector128.Add(Vector128.Load((double*)accSlot), StreamPairwiseComplex(ref stream, 0, n)),
                        (double*)accSlot);
                    return true;
            }
        }

        /// <summary>
        /// Streaming form of the M2 axis divert over an all-C-contiguous child: returns a FRESH C-contiguous
        /// array of <paramref name="reducedShape"/> holding, element for element, what
        /// <see cref="ExactAxisSum"/> / <see cref="SequentialAxisProd"/> return for the materialized child — or
        /// null (having allocated nothing) when the child is not streamable, so the caller materializes.
        /// <para>
        /// An ALL-F-contiguous operand set streams too (plan lever 4 follow-up): an F-contiguous array is the
        /// C-contiguous array of its reversed dims, so reducing axis <c>k</c> walks memory as the C reduction of
        /// axis <c>nd-1-k</c> over the reversed dims — <c>outer = Π dims[k+1:]</c>, <c>inner = Π dims[:k]</c> — and
        /// that memory-order walk is exactly the schedule NumPy's reduce iterator (K order) runs over the
        /// F-contiguous child its own <c>np.sum(a*b, axis)</c> materializes (whole-run pairwise when the reduced
        /// axis is the contiguous one, slab accumulation otherwise). The result is then F-contiguous
        /// (<paramref name="reducedShape"/>'s dims in F order — NumPy's K-order output), which every caller handles
        /// as a dense block. No C-materializing path computed this: the materialize route copies to C (C order)
        /// and the fold path folds, so this is the only NumPy-exact route for F inputs.
        /// </para>
        /// <list type="bullet">
        ///   <item>1-D child (reduce-all): the flat semantics — <c>0 + pairwise(child)</c> / sequential product.</item>
        ///   <item>size-1 reduced axis of a multi-D child (sum): the engine's trivial-axis reduction COPIES the
        ///         values (no <c>0 + x</c>), so the child is produced straight into the result.</item>
        ///   <item>reduced axis innermost (<c>inner == 1</c>): each output is <c>0 + pairwise(row)</c> over its
        ///         whole contiguous row (NumPy's PINNED loop), many short rows per scratch block.</item>
        ///   <item>outer reduced axis (<c>inner &gt; 1</c>): each output starts at the identity and accumulates
        ///         slab after slab in increasing axis order (NumPy's SLAB loop), per-lane exact.</item>
        /// </list>
        /// </summary>
        /// <param name="child">The reduction's child elementwise program.</param>
        /// <param name="inputs">The call's inputs (iterator operands and hoisted parameters, in input order).</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="sequentialProd">True for np.multiply.reduce (float32/float64), false for np.add.reduce.</param>
        /// <param name="reducedShape">The output shape (input shape with <paramref name="axis"/> removed; scalar for a 1-D child).</param>
        /// <returns>The fresh reduced array (the caller owns it), or null when the caller must materialize.</returns>
        private unsafe NDArray TryStreamAxisReduce(NDExprProgram child, NDArray[] inputs, int axis, bool sequentialProd, Shape reducedShape)
        {
            var childType = child.ResultType;
            if (sequentialProd ? childType != NPTypeCode.Double && childType != NPTypeCode.Single
                               : childType != NPTypeCode.Double && childType != NPTypeCode.Single && childType != NPTypeCode.Complex)
                return null;

            // One shared memory order: all C (the M2 divert's C-contiguous child layout) or all F (walked as the C
            // order of the reversed dims — see the summary). A mixed set fails the shared-order test.
            var ops = child.IteratorOperands(inputs);
            if (!CanStreamChild(child, ops, allowF: true))
                return null;
            bool fOrder = false;
            for (int j = 0; j < ops.Length; j++)
            {
                // CanStreamChild accepted allC || allF; any operand that is not C-contiguous means the shared
                // order is F (an operand contiguous in BOTH orders fits either, so it never decides).
                if (!ops[j].Shape.IsContiguous)
                {
                    fOrder = true;
                    break;
                }
            }

            var stream = new NDExprChildStream { Kernel = child.Kernel };
            if (!sequentialProd && !BindSumFold(ref stream, childType))
                return null;

            int nop = ops.Length;
            byte** bases = stackalloc byte*[nop];
            long* elemBytes = stackalloc long[nop];
            void** ptrs = stackalloc void*[nop + 1];
            long* strides = stackalloc long[nop + 1];
            BindChildStream(ref stream, ops, childType, bases, elemBytes, ptrs, strides);

            if (child.ParamCount > 0)
            {
                // stackalloc memory lives until the METHOD returns, so the block-scoped declaration is safe.
                byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                child.PackParams(inputs, paramBlock);
                stream.Aux = paramBlock;
            }

            byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
            stream.Scratch = scratch;
            stream.Block = EvaluateStreamScratchBytes / childType.SizeOf();

            var dims = ops[0].Shape.dimensions;
            int nd = dims.Length;
            long axisSize = dims[axis];
            long outer = 1, inner = 1;
            if (!fOrder)
            {
                for (int d = 0; d < axis; d++) outer *= dims[d];
                for (int d = axis + 1; d < nd; d++) inner *= dims[d];
            }
            else
            {
                // Memory is the C order of the reversed dims: the axes AFTER `axis` are the slow (outer) ones and
                // the axes BEFORE it the fast (inner) ones.
                for (int d = axis + 1; d < nd; d++) outer *= dims[d];
                for (int d = 0; d < axis; d++) inner *= dims[d];
            }

            // A fresh result (offset 0): C-contiguous, the layout ExactAxisSum/SequentialAxisProd return for a
            // C-contiguous child, or — for an F walk — F-contiguous, so its memory order (the C order of the
            // reversed reduced dims) is exactly the order the streams below write it in. Disposed on an exception
            // so a failing kernel does not strand its buffer.
            var result = fOrder && reducedShape.NDim > 1
                ? new NDArray(childType, new Shape((long[])reducedShape.dimensions.Clone(), 'F'), false)
                : new NDArray(childType, reducedShape, false);
            NDExpr.StreamingReductions++;
            try
            {
                byte* dst = (byte*)result.Address;

                if (nd == 1)
                {
                    // Reduce-all over the only axis: ExactAxisSum's 1-D branch is the flat pairwise
                    // (0 + pairwise), SequentialAxisProd's is one sequential product — the flat streams.
                    StreamFlatInto(ref stream, childType, axisSize, sequentialProd, dst);
                    return result;
                }

                if (!sequentialProd && axisSize == 1)
                {
                    // The engine reduces a size-1 axis by COPYING it (ReduceAdd → HandleTrivialAxisReduction), so
                    // the materialized path returns x itself, not 0 + x; child index (o·1 + 0)·inner + i equals
                    // result index o·inner + i, so the child is produced straight into the result.
                    ProduceBlocked(ref stream, 0, outer * inner, dst, childType.SizeOf());
                    return result;
                }

                if (inner == 1)
                    StreamPinnedRows(ref stream, childType, outer, axisSize, sequentialProd, dst);
                else
                    StreamSlabs(ref stream, childType, outer, axisSize, inner, sequentialProd, dst);
                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Plan lever 4 — a weighted average's two sums, streamed: the numerator <c>Σ(v·w)</c> over
        /// <see cref="NDExprProgram.AvgNumeratorProgram"/> and the denominator <c>Σ(w)</c> over
        /// <see cref="NDExprProgram.AvgDenominatorProgram"/>, each through the flat / axis Sum streams above — no
        /// materialized child, no product array, no weights copy (the materialize route allocated and wrote three
        /// n-element arrays and read them back: 17.6 ms at 4M against NumPy's 11.7).
        /// </summary>
        /// <remarks>
        /// Bit-identical to the materialize route's <c>ExactSumArray(v·w)</c> / <c>ExactSumArray(w)</c> wherever that
        /// route sums in the operands' memory order — C-contiguous operands, flat and axis: the streams run the
        /// same pairwise recursion / PINNED / SLAB schedule over the same element values (the product of the
        /// result-dtype casts is the same IEEE value however it is produced). For ALL-F operands the flat stream
        /// sums in F memory order, which is NumPy's order for the F-contiguous product (<c>np.multiply</c> keeps K
        /// order, <c>.sum()</c> walks memory) — the materialize route's C copy summed those in C order and was NOT
        /// NumPy-exact there. The axis streams are C-only, so an F axis average still materializes. Declines (false,
        /// outputs null) for anything the Sum streams decline — broadcast, strided, mixed order, more than
        /// <see cref="EvaluateStreamMaxOperands"/> operands, a 0-d-only operand list — and for an EMPTY input, whose
        /// zero total weight the materialize route reports with NumPy's error.
        /// </remarks>
        /// <param name="program">The weighted-average program.</param>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="avg">The bound average node (its axis / keepdims).</param>
        /// <param name="rt">The result dtype (Single / Double / Complex — Half and Decimal were rejected upstream).</param>
        /// <param name="num">Receives the numerator sums (0-d for a flat average, the reduced shape for an axis one), or null.</param>
        /// <param name="den">Receives the denominator sums, same shape as <paramref name="num"/>, or null.</param>
        /// <param name="nax">Receives the normalized reduction axis, or null for a flat average.</param>
        /// <param name="childNdim">Receives the operands' rank — the flat keepdims result is <c>(1,)*childNdim</c>.</param>
        /// <returns>True when both sums were streamed; false leaves every output unset for the materialize route.</returns>
        /// <exception cref="AxisError">The average's axis is out of range for the operands' rank (the same error the materialize route raises).</exception>
        private unsafe bool TryStreamWeightedAverage(NDExprProgram program, NDArray[] inputs, WeightedAverageNode avg, NPTypeCode rt,
            out NDArray num, out NDArray den, out int? nax, out int childNdim)
        {
            num = null;
            den = null;
            nax = null;
            childNdim = 0;

            var numProgram = program.AvgNumeratorProgram;
            var denProgram = program.AvgDenominatorProgram;

            // Both children carry this program's input signature, so they stream the same operand list; its shape
            // is the average's (the streams accept only identical, unbroadcast dims).
            var ops = numProgram.IteratorOperands(inputs);
            if (ops.Length == 0)
                return false;
            long n = ops[0].size;
            if (n == 0)
                return false;
            int nd = ops[0].ndim;

            if (avg.Axis is null)
            {
                byte* numSlot = stackalloc byte[NDExprParamPlan.SlotBytes];
                byte* denSlot = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)numSlot = 0; *(ulong*)(numSlot + 8) = 0;
                *(ulong*)denSlot = 0; *(ulong*)(denSlot + 8) = 0;
                if (!TryStreamFlatReduce(numProgram, inputs, n, sequentialProd: false, numSlot)
                    || !TryStreamFlatReduce(denProgram, inputs, n, sequentialProd: false, denSlot))
                    return false;

                // The same 0-d result ExactSumArray builds from its accumulator slot.
                num = ScalarFromSlot(numSlot, rt);
                den = ScalarFromSlot(denSlot, rt);
                childNdim = nd;
                return true;
            }

            int ax = NormalizeAxis(avg.Axis.Value, nd);
            var reducedDims = new long[nd - 1];
            for (int d = 0, rd = 0; d < nd; d++) if (d != ax) reducedDims[rd++] = ops[0].Shape.dimensions[d];
            Shape reducedShape = reducedDims.Length > 0 ? new Shape(reducedDims) : Shape.NewScalar();

            var numArr = TryStreamAxisReduce(numProgram, inputs, ax, sequentialProd: false, reducedShape);
            if (numArr is null)
                return false;
            var denArr = TryStreamAxisReduce(denProgram, inputs, ax, sequentialProd: false, reducedShape);
            if (denArr is null)
            {
                // Unreachable in practice (both children share the operand list and dtype, so the gates agree);
                // release the numerator and let the materialize route recompute both.
                numArr.Dispose();
                return false;
            }

            num = numArr;
            den = denArr;
            nax = ax;
            childNdim = nd;
            return true;
        }

        /// <summary>
        /// Plan lever 3 (bool folds) — a FLAT <c>Any</c> / <c>All</c> / <c>Sum</c>-of-a-bool-child (<c>CountNonzero</c>)
        /// reduction streamed: the child's own SIMD elementwise kernel evaluates each ≤8 KB block into L1 scratch
        /// (one byte per element) and the block is folded with the BCL's vectorized span primitives —
        /// <c>IndexOfAnyExcept(0)</c> for Any, <c>IndexOf(0)</c> for All, <c>Count(0)</c> for the count. The scalar
        /// 4-accumulator fold kernel these kinds rode ran ~0.5 ns/element (any(a&gt;b) 0.25× NumPy at 100K, 4× slower
        /// than NumSharp's own unfused <c>np.any(a &gt; b)</c>). When the child is the factories' <c>x != 0</c> over a
        /// bool <c>x</c> (<see cref="NDExprProgram.NonzeroBoolOperandProgram"/>), <c>x</c> itself is streamed: the
        /// same bools, but through <c>x</c>'s SIMD kernel instead of the scalar int64-typed comparison.
        /// </summary>
        /// <remarks>
        /// Bit-exact BY CONSTRUCTION, not by schedule: logical OR / AND and an integer count are associative and
        /// commutative, so any blocking and any traversal order give the identical answer — which is also why any
        /// shared contiguous order (all C or all F) streams, and why Any / All may STOP at the first deciding block
        /// (a later element cannot change the answer; only a side-effecting <c>Call</c> node could observe the skipped
        /// evaluations, and element evaluation order is not contractual). The caller has already seeded
        /// <paramref name="slot"/> with the identity (Any 0, All 1, Sum 0); this folds the whole child into it.
        /// Declines (false, slot untouched) for any other kind, a non-bool child, a Sum whose accumulator is not
        /// int64, or operands the streams cannot walk — the fold kernel then runs as before.
        /// </remarks>
        /// <param name="program">The flat reduction program.</param>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="n">The child's element count (&gt; 0).</param>
        /// <param name="kind">The reduction kind.</param>
        /// <param name="accType">The accumulator dtype the slot holds.</param>
        /// <param name="slot">The identity-seeded accumulator slot the result is folded into.</param>
        /// <returns>True when the reduction was computed here.</returns>
        private static unsafe bool TryStreamBoolFold(NDExprProgram program, NDArray[] inputs, long n,
            NDExprReduceKind kind, NPTypeCode accType, byte* slot)
        {
            bool any = kind == NDExprReduceKind.Any;
            bool all = kind == NDExprReduceKind.All;
            bool count = kind == NDExprReduceKind.Sum && accType == NPTypeCode.Int64;
            if (!(any || all || count))
                return false;

            // The factories spell the child `x != 0`; for a bool x that IS x, and x's own kernel keeps its SIMD body
            // where the int64-typed `bool != 0` comparison runs scalar — so stream x directly when the pattern holds
            // (the same bools, element for element), else the child exactly as written.
            var child = program.NonzeroBoolOperandProgram ?? program.ChildElementwiseProgram;
            if (child is null || child.ResultType != NPTypeCode.Boolean)
                return false;
            var ops = child.IteratorOperands(inputs);
            if (!CanStreamChild(child, ops, allowF: true))
                return false;

            NDExpr.StreamingReductions++;
            long nonzero = 0;

            if (child.Bound is InputNode && ops.Length == 1)
            {
                // A bare bool LEAF (`any(mask)`, `count_nonzero(mask)`): the child's elements ARE the operand's bytes,
                // and CanStreamChild proved them one dense C- or F-contiguous block (no broadcast), so fold that block
                // in place. Running the leaf's identity kernel would only copy it into scratch first — measured
                // 0.65-0.83x NumPy's count_nonzero(bool) with the copy. Order is free (OR / AND / a count).
                byte* p = (byte*)ops[0].Address + (long)ops[0].Shape.offset; // itemsize 1: offset is in bytes
                for (long start = 0; start < n; start += BoolScanChunkBytes)
                {
                    int m = (int)Math.Min(BoolScanChunkBytes, n - start);
                    if (FoldBoolBlock(new ReadOnlySpan<byte>(p + start, m), any, all, ref nonzero, slot))
                        return true;
                }
            }
            else
            {
                var stream = new NDExprChildStream { Kernel = child.Kernel };
                int nop = ops.Length;
                byte** bases = stackalloc byte*[nop];
                long* elemBytes = stackalloc long[nop];
                void** ptrs = stackalloc void*[nop + 1];
                long* strides = stackalloc long[nop + 1];
                BindChildStream(ref stream, ops, NPTypeCode.Boolean, bases, elemBytes, ptrs, strides);

                if (child.ParamCount > 0)
                {
                    // The child is an ELEMENTWISE program: its parameters live in the elementwise aux layout
                    // (slot 0 onward), not after a reduce accumulator. stackalloc lives until the method returns.
                    byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                    child.PackParams(inputs, paramBlock);
                    stream.Aux = paramBlock;
                }

                byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
                stream.Scratch = scratch;
                stream.Block = EvaluateStreamScratchBytes; // one byte per bool element

                for (long start = 0; start < n; start += stream.Block)
                {
                    int m = (int)Math.Min(stream.Block, n - start);
                    stream.Produce(start, m, scratch);
                    if (FoldBoolBlock(new ReadOnlySpan<byte>(scratch, m), any, all, ref nonzero, slot))
                        return true;
                }
            }

            // Any: no true element — the seeded False stands; All: no false element — the seeded True stands;
            // the count joins the seeded 0 (int64 addition, the Sum fold's own arithmetic).
            if (count)
                *(long*)slot += nonzero;
            return true;
        }

        /// <summary>
        /// Largest block the in-place bool-leaf scan hands one span primitive: spans are int-length, so an operand past
        /// 2 GiB is folded in 1 GiB pieces. The pieces cost nothing measurable — the scan is bandwidth-bound — and an
        /// Any / All still stops inside the first piece that decides it.
        /// </summary>
        private const int BoolScanChunkBytes = 1 << 30;

        /// <summary>
        /// Fold one block of bools (one byte each, any nonzero byte is True — NumPy's truthiness, so a bool array
        /// holding a raw byte 2 counts as True here exactly as in the fold kernel's <c>x != 0</c>) into a streamed
        /// <c>Any</c> / <c>All</c> / count reduction.
        /// </summary>
        /// <param name="block">The bools to fold.</param>
        /// <param name="any">Folding logical OR: the first True decides the reduction.</param>
        /// <param name="all">Folding logical AND: the first False decides the reduction.</param>
        /// <param name="nonzero">The running count of True elements; advanced only when neither
        /// <paramref name="any"/> nor <paramref name="all"/> is set.</param>
        /// <param name="slot">The accumulator slot; written ONLY when this block decides the reduction.</param>
        /// <returns>True when <paramref name="block"/> decided the whole reduction (the caller must stop scanning:
        /// <paramref name="slot"/> already holds the answer); false to continue with the next block.</returns>
        private static unsafe bool FoldBoolBlock(ReadOnlySpan<byte> block, bool any, bool all, ref long nonzero, byte* slot)
        {
            if (any)
            {
                // logical_or: the first true element decides the whole reduction.
                if (block.IndexOfAnyExcept((byte)0) < 0)
                    return false;
                *slot = 1;
                return true;
            }

            if (all)
            {
                // logical_and: the first false element decides the whole reduction.
                if (block.IndexOf((byte)0) < 0)
                    return false;
                *slot = 0;
                return true;
            }

            nonzero += block.Length - block.Count((byte)0);
            return false;
        }

        /// <summary>
        /// A fresh 0-d array of <paramref name="t"/> holding the accumulator slot's value — the exact construction
        /// <c>ExactSumArray</c>'s flat branch performs, so a streamed flat sum is indistinguishable from it.
        /// </summary>
        /// <param name="slot">The 16-byte accumulator slot (value at offset 0, dtype <paramref name="t"/>).</param>
        /// <param name="t">The accumulator / result dtype.</param>
        /// <returns>The 0-d result.</returns>
        private static unsafe NDArray ScalarFromSlot(byte* slot, NPTypeCode t)
        {
            var r = new NDArray(t, Shape.NewScalar(), false);
            NDIterCasting.ConvertValue(slot, (byte*)r.Address + (long)r.Shape.offset * t.SizeOf(), t, t);
            return r;
        }

        /// <summary>
        /// Produce <paramref name="count"/> child elements starting at flat index <paramref name="start"/> straight
        /// into <paramref name="dst"/>, one kernel call per <see cref="NDExprChildStream.Block"/> (the kernel handles
        /// any count, but bounding it keeps each call's operand streams prefetch-friendly and uniform with the rest).
        /// </summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="start">First flat element index.</param>
        /// <param name="count">Number of elements to produce.</param>
        /// <param name="dst">Destination (contiguous, child dtype).</param>
        /// <param name="elemSize">Child element size in bytes.</param>
        private static unsafe void ProduceBlocked(ref NDExprChildStream stream, long start, long count, byte* dst, int elemSize)
        {
            for (long k = 0; k < count; k += stream.Block)
            {
                long m = Math.Min(stream.Block, count - k);
                stream.Produce(start + k, m, dst + k * elemSize);
            }
        }

        /// <summary>
        /// Write the flat reduction of the whole bound child (<paramref name="n"/> elements) into the result slot
        /// <paramref name="dst"/>: <c>0 + pairwise(child)</c> for a sum (the PINNED reduce contract over a
        /// zero-seeded slot), the sequential product for <paramref name="sequentialProd"/>.
        /// </summary>
        /// <param name="stream">The bound stream (sum fold bound when summing).</param>
        /// <param name="childType">Double, Single or Complex (sum only).</param>
        /// <param name="n">Element count (&gt; 0).</param>
        /// <param name="sequentialProd">True for the product.</param>
        /// <param name="dst">The destination element (child dtype).</param>
        private static unsafe void StreamFlatInto(ref NDExprChildStream stream, NPTypeCode childType, long n, bool sequentialProd, byte* dst)
        {
            switch (childType)
            {
                case NPTypeCode.Double:
                    if (sequentialProd)
                    {
                        *(double*)dst = StreamProductDouble(ref stream, 0, n, 1d);
                    }
                    else
                    {
                        // Seed the slot with a RUNTIME zero and add through memory, mirroring the PINNED kernel's
                        // `*out = *out + fold` (a constant 0.0 + x is also left alone by the JIT, but the memory
                        // form makes the -0.0 → +0.0 contract independent of that).
                        *(double*)dst = 0d;
                        *(double*)dst = *(double*)dst + StreamPairwiseDouble(ref stream, 0, n);
                    }

                    return;

                case NPTypeCode.Single:
                    if (sequentialProd)
                    {
                        *(float*)dst = StreamProductSingle(ref stream, 0, n, 1f);
                    }
                    else
                    {
                        *(float*)dst = 0f;
                        *(float*)dst = *(float*)dst + StreamPairwiseSingle(ref stream, 0, n);
                    }

                    return;

                default: // Complex sum
                    Vector128.Store(Vector128<double>.Zero, (double*)dst);
                    Vector128.Store(Vector128.Add(Vector128.Load((double*)dst), StreamPairwiseComplex(ref stream, 0, n)),
                        (double*)dst);
                    return;
            }
        }

        /// <summary>
        /// float64 pairwise sum of child elements <c>[start, start + count)</c>, bit-identical to
        /// <c>pairwise_sum</c> over the same range of the materialized child: above the scratch capacity it splits
        /// exactly as the fold does (<c>left = count/2 − (count/2)%8</c>, left subtree first), and at or below it
        /// evaluates the range into scratch and hands it to the fold, which continues the SAME recursion.
        /// </summary>
        /// <param name="stream">The bound stream (FoldDouble set).</param>
        /// <param name="start">First flat element index of the range.</param>
        /// <param name="count">Range length (&gt; 0).</param>
        /// <returns>The pairwise sum of the range.</returns>
        private static unsafe double StreamPairwiseDouble(ref NDExprChildStream stream, long start, long count)
        {
            if (count <= stream.Block)
            {
                stream.Produce(start, count, stream.Scratch);
                return stream.FoldDouble(stream.Scratch, count, 1);
            }

            long left = count / 2;
            left -= left % 8;
            // Left subtree fully before the right one: each leaf's fold consumes the scratch before the next
            // leaf overwrites it, and IEEE addition of the two subtree sums matches the fold's own `add`.
            double l = StreamPairwiseDouble(ref stream, start, left);
            return l + StreamPairwiseDouble(ref stream, start + left, count - left);
        }

        /// <summary>
        /// float32 twin of <see cref="StreamPairwiseDouble"/> — accumulates in float32 exactly like NumPy's
        /// <c>FLOAT_pairwise_sum</c> (C# <c>float + float</c> is a single-precision IEEE add).
        /// </summary>
        /// <param name="stream">The bound stream (FoldSingle set).</param>
        /// <param name="start">First flat element index of the range.</param>
        /// <param name="count">Range length (&gt; 0).</param>
        /// <returns>The float32 pairwise sum of the range.</returns>
        private static unsafe float StreamPairwiseSingle(ref NDExprChildStream stream, long start, long count)
        {
            if (count <= stream.Block)
            {
                stream.Produce(start, count, stream.Scratch);
                return stream.FoldSingle(stream.Scratch, count, 1);
            }

            long left = count / 2;
            left -= left % 8;
            float l = StreamPairwiseSingle(ref stream, start, left);
            return l + StreamPairwiseSingle(ref stream, start + left, count - left);
        }

        /// <summary>
        /// complex128 pairwise sum of child complexes <c>[start, start + count)</c> as <c>[re, im]</c>. The complex
        /// fold halves the interleaved DOUBLE count (NumPy's <c>CDOUBLE_pairwise_sum</c> works on <c>2n</c>
        /// doubles), so its split is <c>left = (count − count%8) / 2</c> complexes — mirrored here, NOT the real
        /// folds' rule; subtree sums combine component-wise (one <c>Vector128</c> add, as the fold's own).
        /// </summary>
        /// <param name="stream">The bound stream (FoldComplex set).</param>
        /// <param name="start">First flat complex index of the range.</param>
        /// <param name="count">Range length in complexes (&gt; 0).</param>
        /// <returns><c>[reSum, imSum]</c> of the range.</returns>
        private static unsafe Vector128<double> StreamPairwiseComplex(ref NDExprChildStream stream, long start, long count)
        {
            if (count <= stream.Block)
            {
                stream.Produce(start, count, stream.Scratch);
                return stream.FoldComplex(stream.Scratch, count, 16);
            }

            long left = (count - count % 8) / 2;
            var l = StreamPairwiseComplex(ref stream, start, left);
            return Vector128.Add(l, StreamPairwiseComplex(ref stream, start + left, count - left));
        }

        /// <summary>
        /// Sequential float64 product of child elements <c>[start, start + count)</c> continuing from
        /// <paramref name="acc"/> — np.multiply.reduce's increasing-index <c>acc *= x</c> chain, block by block.
        /// </summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="start">First flat element index.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="acc">The running product entering the range (1 at the start of a reduction).</param>
        /// <returns>The running product after the range.</returns>
        private static unsafe double StreamProductDouble(ref NDExprChildStream stream, long start, long count, double acc)
        {
            double* x = (double*)stream.Scratch;
            for (long k = 0; k < count; k += stream.Block)
            {
                long m = Math.Min(stream.Block, count - k);
                stream.Produce(start + k, m, stream.Scratch);
                for (long i = 0; i < m; i++)
                    acc *= x[i];
            }

            return acc;
        }

        /// <summary>float32 twin of <see cref="StreamProductDouble"/> (float32 multiply chain).</summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="start">First flat element index.</param>
        /// <param name="count">Number of elements.</param>
        /// <param name="acc">The running product entering the range.</param>
        /// <returns>The running product after the range.</returns>
        private static unsafe float StreamProductSingle(ref NDExprChildStream stream, long start, long count, float acc)
        {
            float* x = (float*)stream.Scratch;
            for (long k = 0; k < count; k += stream.Block)
            {
                long m = Math.Min(stream.Block, count - k);
                stream.Produce(start + k, m, stream.Scratch);
                for (long i = 0; i < m; i++)
                    acc *= x[i];
            }

            return acc;
        }

        /// <summary>
        /// PINNED axis reduction (reduced axis innermost, <c>inner == 1</c>): <c>dst[o]</c> reduces the contiguous row
        /// <c>[o·axisSize, (o+1)·axisSize)</c>. A sum writes <c>0 + pairwise(row)</c> — the engine seeds the output
        /// with +0 and the PINNED kernel adds the whole-row fold; a product writes the row's sequential product.
        /// Rows that fit the scratch are produced many at a time (one kernel call per block, not per row); longer
        /// rows stream through the pairwise recursion one row at a time.
        /// </summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="childType">Double, Single or Complex (sum only for Complex).</param>
        /// <param name="outer">Number of rows (outputs).</param>
        /// <param name="axisSize">Row length (≥ 2 — a size-1 sum axis is the copy branch).</param>
        /// <param name="sequentialProd">True for the product.</param>
        /// <param name="dst">The C-contiguous result buffer (<paramref name="outer"/> elements).</param>
        private static unsafe void StreamPinnedRows(ref NDExprChildStream stream, NPTypeCode childType,
            long outer, long axisSize, bool sequentialProd, byte* dst)
        {
            if (axisSize > stream.Block)
            {
                // Long rows: each row is its own streamed flat reduction.
                int es = childType.SizeOf();
                for (long o = 0; o < outer; o++)
                    StreamRowInto(ref stream, childType, o * axisSize, axisSize, sequentialProd, dst + o * es);
                return;
            }

            long rowsPer = stream.Block / axisSize;
            for (long o0 = 0; o0 < outer; o0 += rowsPer)
            {
                long rows = Math.Min(rowsPer, outer - o0);
                stream.Produce(o0 * axisSize, rows * axisSize, stream.Scratch);
                switch (childType)
                {
                    case NPTypeCode.Double:
                    {
                        double* x = (double*)stream.Scratch;
                        double* d = (double*)dst + o0;
                        for (long r = 0; r < rows; r++, x += axisSize)
                        {
                            if (sequentialProd)
                            {
                                double acc = 1d;
                                for (long k = 0; k < axisSize; k++) acc *= x[k];
                                d[r] = acc;
                            }
                            else
                            {
                                d[r] = 0d;
                                d[r] = d[r] + RowFoldDouble(ref stream, x, axisSize);
                            }
                        }

                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        float* x = (float*)stream.Scratch;
                        float* d = (float*)dst + o0;
                        for (long r = 0; r < rows; r++, x += axisSize)
                        {
                            if (sequentialProd)
                            {
                                float acc = 1f;
                                for (long k = 0; k < axisSize; k++) acc *= x[k];
                                d[r] = acc;
                            }
                            else
                            {
                                d[r] = 0f;
                                d[r] = d[r] + RowFoldSingle(ref stream, x, axisSize);
                            }
                        }

                        break;
                    }
                    default: // Complex sum
                    {
                        double* x = (double*)stream.Scratch;
                        double* d = (double*)dst + 2 * o0;
                        for (long r = 0; r < rows; r++, x += 2 * axisSize, d += 2)
                        {
                            Vector128.Store(Vector128<double>.Zero, d);
                            Vector128.Store(Vector128.Add(Vector128.Load(d), RowFoldComplex(ref stream, x, axisSize)), d);
                        }

                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Reduce ONE long row (<paramref name="count"/> &gt; the scratch capacity) into <paramref name="dst"/>:
        /// <c>0 + pairwise(row)</c> through the streamed recursion, or the streamed sequential product.
        /// </summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="childType">Double, Single or Complex (sum only for Complex).</param>
        /// <param name="start">Flat index of the row's first element.</param>
        /// <param name="count">Row length.</param>
        /// <param name="sequentialProd">True for the product.</param>
        /// <param name="dst">The output element.</param>
        private static unsafe void StreamRowInto(ref NDExprChildStream stream, NPTypeCode childType,
            long start, long count, bool sequentialProd, byte* dst)
        {
            switch (childType)
            {
                case NPTypeCode.Double:
                    if (sequentialProd)
                    {
                        *(double*)dst = StreamProductDouble(ref stream, start, count, 1d);
                    }
                    else
                    {
                        *(double*)dst = 0d;
                        *(double*)dst = *(double*)dst + StreamPairwiseDouble(ref stream, start, count);
                    }

                    return;
                case NPTypeCode.Single:
                    if (sequentialProd)
                    {
                        *(float*)dst = StreamProductSingle(ref stream, start, count, 1f);
                    }
                    else
                    {
                        *(float*)dst = 0f;
                        *(float*)dst = *(float*)dst + StreamPairwiseSingle(ref stream, start, count);
                    }

                    return;
                default:
                    Vector128.Store(Vector128<double>.Zero, (double*)dst);
                    Vector128.Store(Vector128.Add(Vector128.Load((double*)dst), StreamPairwiseComplex(ref stream, start, count)),
                        (double*)dst);
                    return;
            }
        }

        /// <summary>
        /// The float64 pairwise fold of one short, already-produced row. Rows below the fold's 8-element BASE
        /// threshold are summed inline with the fold's own BASE loop (<c>res = -0.0; res += x[k]</c>) — identical
        /// arithmetic, minus a delegate call per row, which dominates a many-short-rows reduction.
        /// </summary>
        /// <param name="stream">The bound stream (FoldDouble set).</param>
        /// <param name="x">The row's first element (in scratch).</param>
        /// <param name="m">Row length.</param>
        /// <returns><c>pairwise_sum(row)</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double RowFoldDouble(ref NDExprChildStream stream, double* x, long m)
        {
            if (m < 8)
            {
                double res = -0.0;
                for (long k = 0; k < m; k++) res += x[k];
                return res;
            }

            return stream.FoldDouble(x, m, 1);
        }

        /// <summary>float32 twin of <see cref="RowFoldDouble"/> (BASE seed <c>-0.0f</c>, float32 adds).</summary>
        /// <param name="stream">The bound stream (FoldSingle set).</param>
        /// <param name="x">The row's first element (in scratch).</param>
        /// <param name="m">Row length.</param>
        /// <returns><c>pairwise_sum(row)</c> in float32.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float RowFoldSingle(ref NDExprChildStream stream, float* x, long m)
        {
            if (m < 8)
            {
                float res = -0.0f;
                for (long k = 0; k < m; k++) res += x[k];
                return res;
            }

            return stream.FoldSingle(x, m, 1);
        }

        /// <summary>
        /// complex128 twin of <see cref="RowFoldDouble"/>: rows below the complex fold's 4-complex BASE threshold
        /// are summed inline component-wise from a <c>(-0.0, -0.0)</c> seed, exactly the fold's BASE loop.
        /// </summary>
        /// <param name="stream">The bound stream (FoldComplex set).</param>
        /// <param name="x">The row's first complex, as interleaved <c>(re, im)</c> doubles.</param>
        /// <param name="m">Row length in complexes.</param>
        /// <returns><c>[reSum, imSum]</c> of the row.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<double> RowFoldComplex(ref NDExprChildStream stream, double* x, long m)
        {
            if (m < 4)
            {
                double re = -0.0, im = -0.0;
                for (long k = 0; k < m; k++)
                {
                    re += x[2 * k];
                    im += x[2 * k + 1];
                }

                return Vector128.Create(re, im);
            }

            return stream.FoldComplex(x, m, 16);
        }

        /// <summary>
        /// SLAB axis reduction (outer reduced axis, <c>inner &gt; 1</c>): every output element starts at the
        /// identity (+0 for a sum — what the engine seeds — or 1 for a product) and folds the child's slabs in
        /// increasing axis index: <c>out[o, i] = ((id ∘ x[o,0,i]) ∘ x[o,1,i]) ∘ …</c>, NumPy's streaming SLAB loop.
        /// Each output element's operation sequence is independent of the others, so the loops are ordered for
        /// locality: a narrow slab set is produced many slabs per kernel call, a wide one block by block with the
        /// output block kept hot across the whole axis.
        /// </summary>
        /// <param name="stream">The bound stream.</param>
        /// <param name="childType">Double, Single or Complex (sum only for Complex).</param>
        /// <param name="outer">Product of the dimensions before the reduced axis.</param>
        /// <param name="axisSize">Reduced axis length.</param>
        /// <param name="inner">Product of the dimensions after the reduced axis (&gt; 1).</param>
        /// <param name="sequentialProd">True for the product.</param>
        /// <param name="dst">The C-contiguous result buffer (<c>outer·inner</c> elements).</param>
        private static unsafe void StreamSlabs(ref NDExprChildStream stream, NPTypeCode childType,
            long outer, long axisSize, long inner, bool sequentialProd, byte* dst)
        {
            // Complex slabs accumulate as interleaved doubles: a component-wise add is exactly the complex add.
            int lanesPerElem = childType == NPTypeCode.Complex ? 2 : 1;
            int es = childType.SizeOf();

            if (inner <= stream.Block)
            {
                long slabsPer = stream.Block / inner;
                for (long o = 0; o < outer; o++)
                {
                    byte* d = dst + o * inner * es;
                    FillIdentity(d, inner * lanesPerElem, childType, sequentialProd);
                    for (long j0 = 0; j0 < axisSize; j0 += slabsPer)
                    {
                        long jc = Math.Min(slabsPer, axisSize - j0);
                        stream.Produce((o * axisSize + j0) * inner, jc * inner, stream.Scratch);
                        for (long t = 0; t < jc; t++)
                            Accumulate(d, stream.Scratch + t * inner * es, inner * lanesPerElem, childType, sequentialProd);
                    }
                }

                return;
            }

            for (long o = 0; o < outer; o++)
            {
                for (long ib = 0; ib < inner; ib += stream.Block)
                {
                    long m = Math.Min(stream.Block, inner - ib);
                    byte* d = dst + (o * inner + ib) * es;
                    FillIdentity(d, m * lanesPerElem, childType, sequentialProd);
                    for (long j = 0; j < axisSize; j++)
                    {
                        stream.Produce((o * axisSize + j) * inner + ib, m, stream.Scratch);
                        Accumulate(d, stream.Scratch, m * lanesPerElem, childType, sequentialProd);
                    }
                }
            }
        }

        /// <summary>
        /// Seed <paramref name="count"/> accumulator lanes with the reduction identity: +0 for a sum (the value
        /// <c>SeedReduceIdentity</c> writes, so an all-negative-zero column still yields +0), 1 for a product.
        /// </summary>
        /// <param name="d">First lane.</param>
        /// <param name="count">Number of lanes (complex elements count twice).</param>
        /// <param name="childType">Double, Single or Complex (Complex lanes are doubles).</param>
        /// <param name="product">True to seed 1, false to seed +0.</param>
        private static unsafe void FillIdentity(byte* d, long count, NPTypeCode childType, bool product)
        {
            if (childType == NPTypeCode.Single)
                new Span<float>(d, checked((int)count)).Fill(product ? 1f : 0f);
            else
                new Span<double>(d, checked((int)count)).Fill(product ? 1d : 0d);
        }

        /// <summary>
        /// <c>d[k] = d[k] + x[k]</c> (or <c>*</c>) for <paramref name="count"/> lanes. Lanes are independent, so the
        /// portable <see cref="Vector{T}"/> body is per-lane identical to the scalar statement (no reassociation).
        /// </summary>
        /// <param name="d">Accumulator lanes (read-modify-write).</param>
        /// <param name="x">The slab's values.</param>
        /// <param name="count">Number of lanes.</param>
        /// <param name="childType">Double, Single or Complex (Complex lanes are doubles).</param>
        /// <param name="product">True to multiply, false to add.</param>
        private static unsafe void Accumulate(byte* d, byte* x, long count, NPTypeCode childType, bool product)
        {
            if (childType == NPTypeCode.Single)
                AccumulateLanes((float*)d, (float*)x, count, product);
            else
                AccumulateLanes((double*)d, (double*)x, count, product);
        }

        /// <summary>
        /// Typed lane loop behind <see cref="Accumulate"/>: <see cref="Vector{T}"/> strides then a scalar tail. Each
        /// lane performs exactly one IEEE add (or multiply) per slab — the same operation, in the same order, as
        /// the engine's SLAB kernel.
        /// </summary>
        /// <typeparam name="T">float or double.</typeparam>
        /// <param name="d">Accumulator lanes.</param>
        /// <param name="x">The slab's values.</param>
        /// <param name="count">Number of lanes.</param>
        /// <param name="product">True to multiply, false to add.</param>
        private static unsafe void AccumulateLanes<T>(T* d, T* x, long count, bool product) where T : unmanaged, INumber<T>
        {
            long i = 0;
            if (Vector.IsHardwareAccelerated && count >= Vector<T>.Count)
            {
                int w = Vector<T>.Count;
                // Unaligned reads/writes: the scratch block and the output are not vector-aligned in general.
                if (product)
                {
                    for (; i <= count - w; i += w)
                        Unsafe.WriteUnaligned(d + i,
                            Unsafe.ReadUnaligned<Vector<T>>(d + i) * Unsafe.ReadUnaligned<Vector<T>>(x + i));
                }
                else
                {
                    for (; i <= count - w; i += w)
                        Unsafe.WriteUnaligned(d + i,
                            Unsafe.ReadUnaligned<Vector<T>>(d + i) + Unsafe.ReadUnaligned<Vector<T>>(x + i));
                }
            }

            if (product)
            {
                for (; i < count; i++) d[i] = d[i] * x[i];
            }
            else
            {
                for (; i < count; i++) d[i] = d[i] + x[i];
            }
        }
    }
}

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        /// <summary>
        /// Test / diagnostics hook: when set on the current thread, np.evaluate SKIPS the streaming (no-temp)
        /// exact reductions (DefaultEngine.Evaluate.Stream.cs) and materializes the reduced child as the M1/M2
        /// paths always did. It exists so a test can reduce the same tree BOTH ways and prove the results are
        /// byte-identical — the whole safety claim of the streaming path. Thread-static so a parallel test never
        /// perturbs another's run.
        /// </summary>
        [System.ThreadStatic] internal static bool DisableStreamingReduce;

        /// <summary>
        /// Test / diagnostics hook: incremented (on the current thread) each time a flat or axis reduction is
        /// actually computed by the streaming path. A test resets it, evaluates, and asserts it moved — proving
        /// the fast path ENGAGED rather than silently falling back to materialization (which would still give the
        /// right answer and hide a dead optimization). Thread-static for the same reason as
        /// <see cref="DisableStreamingReduce"/>.
        /// </summary>
        [System.ThreadStatic] internal static int StreamingReductions;
    }
}
