using System;
using System.Collections.Concurrent;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// np.evaluate — fused expression evaluation (roadmap Wave 6.1).
    ///
    /// Compiles an <see cref="NDExpr"/> tree to ONE NDIter pass: every
    /// elementwise node runs inside a single inner-loop kernel, so a chained
    /// expression like (a-b)/(a+b) reads each operand once and writes the
    /// result once — no intermediate arrays, no extra memory traffic. The POC
    /// measured 2.8–5.4× over NumPy's unfused ufunc chains for exactly these
    /// shapes.
    ///
    /// Semantics:
    ///   • dtypes follow NumPy 2.x result_type per NODE (NDExpr.Typing.cs):
    ///     the fused result is bit-compatible with the unfused NumPy sequence,
    ///     including int32 wraparound before promotion.
    ///   • operands broadcast together exactly like a ufunc call; repeated
    ///     NDArray references deduplicate to one iterator operand.
    ///   • out= joins the broadcast (never stretched), needs a same_kind cast
    ///     from the resolved dtype, and may alias an input — COPY_IF_OVERLAP
    ///     gives ufunc-grade overlap safety.
    ///   • a root ReduceNode (NDExpr.Sum/Prod/Min/Max/Mean) runs a one-pass
    ///     accumulating kernel over the inputs only: sum(a*b) never
    ///     materializes a*b.
    ///
    /// Per call, everything that depends only on the tree and the operand dtypes
    /// — binding, the typing pass, the vector plan, the kernel — comes from a
    /// cached <see cref="NDExprProgram"/>: the root's own slot for a hoisted tree,
    /// or the global structural cache (<see cref="NDExprProgramCache"/>) for a tree
    /// rebuilt per call, keyed by the tree's structural hash and verified node by
    /// node. The host does only what depends on the SHAPES: the iteration shape
    /// (an identical-dims fast path skips the broadcast machinery), the result
    /// allocation and the iterator.
    ///
    /// Known divergences from NumPy (documented, by design):
    ///   • float32/float16 sum/prod/mean accumulate in float64 (NumPy uses
    ///     pairwise accumulation at the input dtype) — fused results can
    ///     differ in the last ulps, usually MORE accurate.
    /// </summary>
    public partial class DefaultEngine
    {
        // Per-arity flag arrays for the evaluate iterator configs — allocated
        // once per input count (Wave 2.2 discipline: no per-call new[] for
        // call-invariant data).
        private static readonly ConcurrentDictionary<int, NDIterPerOpFlags[]> s_evalElementwiseFlags = new();
        private static readonly ConcurrentDictionary<int, NDIterPerOpFlags[]> s_evalReduceFlags = new();

        private static NDIterPerOpFlags[] EvalElementwiseFlags(int nIn)
            => s_evalElementwiseFlags.GetOrAdd(nIn, static n =>
            {
                var flags = new NDIterPerOpFlags[n + 1];
                for (int i = 0; i < n; i++)
                    flags[i] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                flags[n] = NDIterPerOpFlags.WRITEONLY
                           | NDIterPerOpFlags.NO_BROADCAST
                           | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                return flags;
            });

        private static NDIterPerOpFlags[] EvalReduceFlags(int nIn)
            => s_evalReduceFlags.GetOrAdd(nIn, static n =>
            {
                var flags = new NDIterPerOpFlags[n];
                for (int i = 0; i < n; i++)
                    flags[i] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                return flags;
            });

        /// <summary>
        /// Evaluate a tree whose array leaves are embedded NDArrays
        /// (<see cref="NDExpr.Arr"/> / implicit conversion).
        /// </summary>
        [NDScoped] // engine boundary: any leaf wrappers BindArrays mints are reclaimed; the fused result (or @out) is yielded
        public override unsafe NDArray Evaluate(NDExpr expr, NDArray @out = null)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));

            // The root's slot, else the global structural cache, else a build — and the distinct arrays the
            // tree references in binding order (the operand list the program was compiled against).
            var program = expr.GetProgram(null, out var operands);
            return EvaluateCore(program, operands, @out);
        }

        /// <summary>
        /// Evaluate a tree built over positional <see cref="NDExpr.Input"/>
        /// leaves against an explicit operand list.
        /// </summary>
        public override unsafe NDArray Evaluate(NDExpr expr, NDArray[] operands, NDArray @out = null)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));
            if (operands is null) throw new ArgumentNullException(nameof(operands));
            if (operands.Length == 0)
                throw new ArgumentException("operands must not be empty.", nameof(operands));
            foreach (var op in operands)
                if (op is null)
                    throw new ArgumentNullException(nameof(operands), "no operand may be null.");

            var program = expr.GetProgram(operands);
            return EvaluateCore(program, operands, @out);
        }

        /// <summary>
        /// The <see cref="CompiledExpression"/> entry: a program compiled ahead of time, evaluated
        /// against <paramref name="operands"/> (the program's own embedded arrays, or a positional list
        /// the handle has already matched to the compiled dtype signature).
        /// </summary>
        internal override NDArray Evaluate(NDExprProgram program, NDArray[] operands, NDArray @out)
            => EvaluateCore(program, operands, @out);

        /// <summary>True when every operand has exactly the same dimensions (no broadcasting to resolve).</summary>
        private static bool AllSameDims(NDArray[] ops)
        {
            var d0 = ops[0].Shape.dimensions;
            for (int i = 1; i < ops.Length; i++)
            {
                var d = ops[i].Shape.dimensions;
                if (d.Length != d0.Length)
                    return false;
                for (int k = 0; k < d.Length; k++)
                    if (d[k] != d0[k])
                        return false;
            }

            return true;
        }

        /// <summary>
        /// The clean (C-strided, offset 0) shape every input operand broadcasts to. Identical dims — the
        /// common case, and the only one for a single operand — is one clone of the first operand's
        /// dims; anything else runs <see cref="BroadcastInputDims"/>.
        /// </summary>
        private static Shape ResolveInputShape(NDArray[] ops)
            => AllSameDims(ops) ? ops[0].Shape.Clean() : new Shape(BroadcastInputDims(ops));

        /// <summary>
        /// The broadcast of every input operand's dimensions, computed NumPy's way (aligned from the
        /// right; a 1 stretches, any other extent must agree) into ONE fresh <c>long[]</c>. The pairwise
        /// <see cref="Broadcast(Shape, Shape)"/> fold this replaces built two Shapes per operand (dims
        /// and strides each) and the host then cloned the result twice more through
        /// <see cref="Shape.Clean"/> — ~450 ns and ~1 KB per call, the whole gap between <c>a*b+c</c>
        /// (identical dims, 404 ns at n = 8) and <c>a*b+k</c> with a 0-d <c>k</c> (823 ns), i.e. the
        /// parameter form a sweep uses so ONE kernel serves every constant. A mismatch raises the
        /// ufunc-layer text with EVERY operand's shape listed, as NumPy does (<c>operands could not be
        /// broadcast together with shapes (2,3) (2,3) (4,) </c> — trailing space included); the fold
        /// could only name its running shape and the offending operand.
        /// </summary>
        private static long[] BroadcastInputDims(NDArray[] ops)
        {
            int nd = 0;
            for (int i = 0; i < ops.Length; i++)
            {
                int n = ops[i].Shape.NDim;
                if (n > nd)
                    nd = n;
            }

            var dims = new long[nd];
            for (int d = 0; d < nd; d++)
                dims[d] = 1;

            for (int i = 0; i < ops.Length; i++)
            {
                var od = ops[i].Shape.dimensions;
                int off = nd - od.Length;
                for (int k = 0; k < od.Length; k++)
                {
                    long v = od[k];
                    if (v == 1)
                        continue;
                    ref long cur = ref dims[off + k];
                    if (cur == 1)
                        cur = v;
                    else if (cur != v)
                        throw EvaluateBroadcastError(ops);
                }
            }

            return dims;
        }

        private static IncorrectShapeException EvaluateBroadcastError(NDArray[] ops)
        {
            var sb = new StringBuilder("operands could not be broadcast together with shapes ");
            foreach (var op in ops)
                sb.Append(op.Shape.ToPythonTuple()).Append(' ');
            return new IncorrectShapeException(sb.ToString());
        }

        private unsafe NDArray EvaluateCore(NDExprProgram program, NDArray[] inputs, NDArray @out)
        {
            if (program.Reduce is not null)
                return EvaluateReduce(program, inputs, @out);

            // The iterator streams the non-parameter inputs; a 0-d parameter reaches the kernel through
            // the aux block instead (NDExpr.Params.cs) and never joins the broadcast — it has no dims.
            var ops = program.IteratorOperands(inputs);
            var kernel = program.Kernel;
            var resolvedType = program.ResultType;

            if (@out is not null)
                ValidateOutCast(resolvedType, @out.typecode, "evaluate");

            // Iteration shape: the inputs' broadcast (one clone for identical dims, one fresh dims
            // array otherwise — ResolveInputShape), then out joins per the ufunc rules (never
            // stretched; its verbatim errors live in ResolveUfuncIterationShape).
            Shape inputShape = ResolveInputShape(ops);
            Shape iterShape = @out is null
                ? inputShape
                : ResolveUfuncIterationShape(inputShape, ops, @out, null).Clean();

            // NumPy-aligned layout preservation (mirrors TryExecuteBinaryOpViaNDIter):
            // when every input operand is strictly F-contiguous, allocate the result
            // column-major and iterate F-order so the iterator coalesces to ONE
            // contiguous inner loop. Forcing C-order here made np.evaluate stride across
            // rows on F/transposed operands — ~16x slower than the unfused chain (the
            // fused F/T cliff). Only the fresh-alloc case re-orders; a provided out keeps
            // its own layout.
            bool allStrictFContig = AreAllInputsStrictFContig(ops, iterShape);
            Shape targetShape = (@out is null && allStrictFContig)
                ? new Shape((long[])iterShape.dimensions.Clone(), 'F')
                : iterShape;

            var target = @out ?? new NDArray(resolvedType, targetShape, false);
            if (target.size == 0)
                return target;

            // F-order iteration only when the result buffer is actually F-contig (fresh
            // F-alloc above, or a provided F-contig out); else the output writes would
            // themselves stride. C-order everywhere else (unchanged default).
            var order = (allStrictFContig && target.Shape.IsFContiguous && !target.Shape.IsContiguous)
                ? NPY_ORDER.NPY_FORTRANORDER
                : NPY_ORDER.NPY_CORDER;

            bool outNeedsCast = target.typecode != resolvedType;
            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode[] opDtypes = null;
            if (outNeedsCast)
            {
                // The kernel writes the resolved dtype into the out operand's
                // buffer; the windowed flush casts (same_kind was validated, so
                // the iterator runs UNSAFE exactly like NumPy's ufunc layer).
                globalFlags |= NDIterGlobalFlags.BUFFERED
                             | NDIterGlobalFlags.GROWINNER
                             | NDIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                opDtypes = new NPTypeCode[ops.Length + 1];
                Array.Copy(program.InputTypes, opDtypes, ops.Length);
                opDtypes[ops.Length] = resolvedType;
            }

            var operands = new NDArray[ops.Length + 1];
            Array.Copy(ops, operands, ops.Length);
            operands[ops.Length] = target;

            using var iter = NDIterRef.MultiNew(
                operands.Length, operands,
                globalFlags, order, casting,
                EvalElementwiseFlags(ops.Length),
                opDtypes);

            // Parameters: their single elements packed once, here, into the aux block the kernel's
            // prologue loads (16 bytes per parameter) — read BEFORE the pass, so one aliasing `out`
            // keeps its original value.
            byte* aux = null;
            if (program.ParamCount > 0)
            {
                // stackalloc memory lives until the METHOD returns, so the block-scoped declaration is safe.
                byte* buf = stackalloc byte[NDExprParamPlan.SlotBytes * program.ParamCount];
                program.PackParams(inputs, buf);
                aux = buf;
            }

            iter.ForEach(kernel, aux);
            return target;
        }

        /// <summary>
        /// N-input analogue of <see cref="AreAllOperandsStrictFContig"/> for the fused
        /// elementwise path: true when the result is multi-dimensional (NDim &gt; 1,
        /// size &gt; 1) and every NON-scalar input operand is strictly F-contiguous
        /// (<c>IsFContiguous &amp;&amp; !IsContiguous</c>), with at least one such operand.
        /// Scalars / size-1 operands don't force the decision. When true, np.evaluate
        /// allocates the result column-major and iterates F-order so the iterator
        /// coalesces to one contiguous inner loop instead of striding across rows (the
        /// F/transpose cliff). Broadcast (stride-0), strided and C-contig inputs all
        /// return false → the C-order default.
        /// </summary>
        internal static bool AreAllInputsStrictFContig(NDArray[] ops, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;

            bool anyPureF = false;
            for (int i = 0; i < ops.Length; i++)
            {
                var s = ops[i].Shape;
                if (s.IsScalar || s.size <= 1)
                    continue;                                   // scalars don't force a preference
                if (!(s.IsFContiguous && !s.IsContiguous))
                    return false;                               // a non-scalar non-pure-F input → C default
                anyPureF = true;
            }

            return anyPureF;
        }

        // =====================================================================
        // Fused reductions
        // =====================================================================

        private static string ReduceUfuncName(NDExprReduceKind kind) => kind switch
        {
            NDExprReduceKind.Sum => "add",
            NDExprReduceKind.Prod => "multiply",
            NDExprReduceKind.Min => "minimum",
            NDExprReduceKind.Max => "maximum",
            NDExprReduceKind.Any => "logical_or",
            NDExprReduceKind.All => "logical_and",
            NDExprReduceKind.Ptp => "ptp",
            NDExprReduceKind.NanMin => "nanmin",
            NDExprReduceKind.NanMax => "nanmax",
            _ => "mean",
        };

        private unsafe NDArray EvaluateReduce(NDExprProgram program, NDArray[] inputs, NDArray @out)
        {
            var reduce = program.Reduce;

            // Plan P2 M4-tail — the order-independent range / NaN-aware min-max kinds are host-delegated:
            // materialize the child and reduce THAT buffer through the already-NumPy-exact engine
            // reduction, which is bit-identical to NumPy regardless of layout because min/max (and their
            // difference) do not depend on summation order — so they need neither the fold kernel below
            // nor the pairwise divert the summation kinds do. Handles BOTH flat and axis, so it must run
            // before the axis dispatch.
            if (reduce.Kind is NDExprReduceKind.Ptp or NDExprReduceKind.NanMin or NDExprReduceKind.NanMax)
                return EvaluateDelegatingReduce(program, inputs, @out);

            if (reduce.Axis is int ax)
                return EvaluateAxisReduce(program, inputs, @out, ax);

            var ops = program.IteratorOperands(inputs);
            var accType = program.ReduceAccType;
            var resultType = program.ResultType;

            if (@out is not null)
            {
                ValidateOutCast(resultType, @out.typecode, "evaluate");
                if (@out.ndim != 0)
                    throw new ArgumentException(
                        $"output parameter for reduction operation {ReduceUfuncName(reduce.Kind)} " +
                        $"has the wrong number of dimensions: Found {@out.ndim} but expected 0");
            }

            long n;
            if (AllSameDims(ops))
            {
                n = ops[0].size;
            }
            else
            {
                n = 1;
                foreach (var d in BroadcastInputDims(ops))
                    n *= d;
            }

            // -------------------------------------------------------------------------------------
            // Plan P2.1 — flat Sum/Mean/Prod at NumPy-EXACT precision.
            //
            // The 4-accumulator fold that FlatReduceKernel emits below is an order-of-summation drift
            // from NumPy's pairwise add.reduce (and its lane-interleaved product is wrong for a
            // non-associative float multiply) — MisalignedRegistry E1. For the dtypes NumPy reduces
            // through a well-defined pairwise (sum) or sequential (prod) schedule we reproduce it
            // BIT-FOR-BIT: materialize the child sub-tree through the already-bit-exact elementwise
            // engine — a fresh, contiguous array in NumPy's own K-order layout — then reduce that
            // buffer IN MEMORY ORDER with the same pairwise kernel np.sum runs, or a sequential
            // product for np.prod. Because the materialized layout matches the intermediate NumPy
            // itself would build, the memory-order reduction equals np.<reduce>(materialized child).
            //
            // Deliberately NOT diverted (still folded below, still E1/#12-excused — each a later
            // milestone): Half (no float16 pairwise kernel yet); complex Prod (a complex-multiply
            // chain, npy_cmul FMA-contracted on NumPy's win-amd64 build, not on .NET's — gap #12);
            // Min/Max (the clamp fold is value-exact; the complex-NaN-order case E5 is separate).
            if (n > 0)
            {
                var childProgram = program.ChildElementwiseProgram;
                var exprType = childProgram.ResultType;            // the child expression's dtype
                bool pairwiseSum = (reduce.Kind == NDExprReduceKind.Sum || reduce.Kind == NDExprReduceKind.Mean)
                    && (exprType == NPTypeCode.Single || exprType == NPTypeCode.Double || exprType == NPTypeCode.Complex);
                bool sequentialProd = reduce.Kind == NDExprReduceKind.Prod
                    && (exprType == NPTypeCode.Single || exprType == NPTypeCode.Double);

                if (pairwiseSum || sequentialProd)
                {
                    // Materialize the child once (fresh + contiguous); `using` releases it after the
                    // reduce — it never escapes.
                    using var materialized = EvaluateCore(childProgram, inputs, null);

                    // The reduction runs at the materialized (== result) dtype; a 16-byte slot holds
                    // any scalar including Complex.
                    byte* accSlot = stackalloc byte[NDExprParamPlan.SlotBytes];
                    *(ulong*)accSlot = 0;
                    *(ulong*)(accSlot + 8) = 0;
                    byte* mSrc = (byte*)materialized.Address
                                 + (long)materialized.Shape.offset * exprType.SizeOf();

                    if (sequentialProd)
                        SequentialProductInto(mSrc, n, exprType, accSlot);   // np.multiply.reduce order
                    else
                        PairwiseSumInto(mSrc, n, exprType, accSlot);         // np.add.reduce (pairwise)

                    if (reduce.Kind == NDExprReduceKind.Mean)
                        DivideAccByCount(accSlot, exprType, n);              // np._mean: divide at result dtype

                    var reduceResult = @out ?? new NDArray(resultType, Shape.NewScalar(), false);
                    byte* rDst = (byte*)reduceResult.Address
                                 + (long)reduceResult.Shape.offset * reduceResult.typecode.SizeOf();
                    NDIterCasting.ConvertValue(accSlot, rDst, exprType, reduceResult.typecode);
                    return reduceResult;
                }
            }

            // Slot 0 (16 bytes) is the accumulator — wide enough for decimal / Complex; the hoisted
            // parameters follow it (NDExprParamPlan.ReduceParamOffset), packed once per call.
            byte* slot = stackalloc byte[NDExprParamPlan.SlotBytes * (1 + program.ParamCount)];
            *(ulong*)slot = 0;
            *(ulong*)(slot + 8) = 0;
            if (program.ParamCount > 0)
                program.PackParams(inputs, slot + NDExprParamPlan.ReduceParamOffset);

            if (n == 0)
            {
                switch (reduce.Kind)
                {
                    case NDExprReduceKind.Min:
                    case NDExprReduceKind.Max:
                        throw new ArgumentException(
                            $"zero-size array to reduction operation {ReduceUfuncName(reduce.Kind)} which has no identity");
                    case NDExprReduceKind.Prod:
                        WriteOne(slot, accType);
                        break;
                    case NDExprReduceKind.Mean:
                        WriteMeanOfEmpty(slot, accType); // NaN, like np.mean([]) (NumPy warns)
                        break;
                    case NDExprReduceKind.All:
                        *slot = 1; // np.all([]) == True (vacuous truth); Any([]) keeps the zeroed slot (False)
                        break;
                        // Sum / Any: identity 0 already in the slot.
                }
            }
            else
            {
                switch (reduce.Kind)
                {
                    case NDExprReduceKind.Prod:
                        WriteOne(slot, accType);
                        break;
                    case NDExprReduceKind.Min:
                    case NDExprReduceKind.Max:
                        WriteMinMaxIdentity(slot, accType, isMin: reduce.Kind == NDExprReduceKind.Min);
                        break;
                    case NDExprReduceKind.All:
                        *slot = 1; // logical_and identity True; the kernel folds AND into it (Any: 0 already)
                        break;
                        // Sum / Mean / Any: identity 0 already in the slot.
                }

                var kernel = program.FlatReduceKernel;
                using var iter = NDIterRef.MultiNew(
                    ops.Length, ops,
                    NDIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    EvalReduceFlags(ops.Length),
                    null);

                iter.ForEach(kernel, slot);

                if (reduce.Kind == NDExprReduceKind.Mean)
                {
                    switch (accType)
                    {
                        case NPTypeCode.Double:
                            *(double*)slot /= n;
                            break;
                        case NPTypeCode.Decimal:
                            *(decimal*)slot /= n;
                            break;
                        case NPTypeCode.Complex:
                            *(System.Numerics.Complex*)slot = ComplexDivideByCountLikeNumPy(*(System.Numerics.Complex*)slot, n);
                            break;
                        default:
                            throw new NotSupportedException($"mean accumulator {accType} — typing bug.");
                    }
                }
            }

            var result = @out ?? new NDArray(resultType, Shape.NewScalar(), false);
            byte* dst = (byte*)result.Address
                        + (long)result.Shape.offset * result.typecode.SizeOf();
            NDIterCasting.ConvertValue(slot, dst, accType, result.typecode);
            return result;
        }

        /// <summary>
        /// Host-delegated reductions (plan P2 M4-tail): <c>Ptp</c> / <c>NanMin</c> / <c>NanMax</c>.
        /// Materializes the reduction's child once (the M1/M2 route — a fresh, contiguous array in
        /// NumPy's own K-order layout) and reduces THAT buffer through the corresponding public engine
        /// reduction (<see cref="np.ptp(NDArray,int?,NDArray,bool)"/> /
        /// <see cref="np.nanmin(NDArray,int?,bool)"/> / <see cref="np.nanmax(NDArray,int?,bool)"/>),
        /// which are themselves NumPy-exact. These three are ORDER-INDEPENDENT — a minimum, a maximum,
        /// or their difference is the same value whatever order the elements are visited in — so reducing
        /// the materialized buffer is bit-identical to <c>np.&lt;kind&gt;(child)</c> for ANY input layout,
        /// the property the summation kinds (Sum/Mean/Prod, and the still-open NanMean/Std/Var) lack and
        /// why they need the pairwise divert instead. The child dtype is preserved
        /// (<see cref="ReduceNode.ResolveReduceResultType"/> returns <c>child</c>), so the engine
        /// reduction's result dtype already equals <paramref name="program"/>.ResultType and no post-cast
        /// is needed except into a caller <paramref name="out"/> of a different (same_kind-castable) dtype.
        /// </summary>
        /// <param name="program">The compiled reduction program; its <c>Reduce</c> carries the kind, axis and keepdims.</param>
        /// <param name="inputs">The call's operand arrays, driving the child's materialization.</param>
        /// <param name="out">Optional destination; must be 0-d for a flat reduction and the reduced shape for an axis one. Null allocates a fresh result.</param>
        /// <returns>The reduced array (a fresh 0-d scalar / reduced-shape array, or <paramref name="out"/> itself when supplied).</returns>
        /// <exception cref="ArgumentException">A flat reduction was given a non-0-d <paramref name="out"/>; or the engine reduction rejects a zero-size input (no identity), an out shape mismatch, or an out dtype not reachable by a same_kind cast.</exception>
        private unsafe NDArray EvaluateDelegatingReduce(NDExprProgram program, NDArray[] inputs, NDArray @out)
        {
            var reduce = program.Reduce;
            var resultType = program.ResultType;

            // Validate the out cast up front (same order as the fold paths: the cast rule is checked
            // before any compute), so an illegal out dtype fails with evaluate's message, not the
            // engine reduction's.
            if (@out is not null)
                ValidateOutCast(resultType, @out.typecode, "evaluate");

            // Materialize the child once (fresh + contiguous). `using` releases it after the delegated
            // reduction has read it; every delegated reduction allocates a FRESH result (np.ptp is a
            // subtract, np.nanmin/nanmax are reductions), so `computed` never aliases `materialized`.
            using var materialized = EvaluateCore(program.ChildElementwiseProgram, inputs, null);

            NDArray computed = reduce.Kind switch
            {
                NDExprReduceKind.Ptp    => np.ptp(materialized, reduce.Axis, keepdims: reduce.Keepdims),
                NDExprReduceKind.NanMin => np.nanmin(materialized, reduce.Axis, reduce.Keepdims),
                NDExprReduceKind.NanMax => np.nanmax(materialized, reduce.Axis, reduce.Keepdims),
                _ => throw new NotSupportedException(
                    $"EvaluateDelegatingReduce reached with non-delegating kind {reduce.Kind} — dispatch bug."),
            };

            if (@out is null)
                return computed;

            // A flat reduction produces a 0-d scalar; NumPy rejects a non-0-d out here with this exact
            // shape of message (the fold path raises the identical text — see EvaluateReduce).
            if (reduce.Axis is null && @out.ndim != 0)
                throw new ArgumentException(
                    $"output parameter for reduction operation {ReduceUfuncName(reduce.Kind)} " +
                    $"has the wrong number of dimensions: Found {@out.ndim} but expected 0");

            // copyto applies the (already-validated) same_kind cast and shape-checks the axis case,
            // matching the axis fold path's `np.copyto(@out, reduced)` tail.
            np.copyto(@out, computed);
            return @out;
        }

        // Axis-aware fused reduction: one pass over the inputs, accumulating into a per-output
        // operand under a REDUCE iterator. evaluate(Sum(a*b, axis:k)) never materializes a*b.
        private unsafe NDArray EvaluateAxisReduce(NDExprProgram program, NDArray[] inputs, NDArray @out, int axis)
        {
            var reduce = program.Reduce;
            var ops = program.IteratorOperands(inputs);

            // Broadcast all inputs to one shape (same rule as the elementwise path).
            Shape inputShape = ResolveInputShape(ops);
            int ndim = inputShape.NDim;
            axis = NormalizeAxis(axis, ndim);

            var accType = program.ReduceAccType;
            var resultType = program.ResultType;

            if (@out is not null)
                ValidateOutCast(resultType, @out.typecode, "evaluate");

            // Output (reduced) shape = input shape with `axis` removed; reduce-all on 1-D → scalar.
            long axisSize = inputShape[axis];
            var reducedDims = new long[ndim - 1];
            for (int d = 0, rd = 0; d < ndim; d++) if (d != axis) reducedDims[rd++] = inputShape[d];
            Shape reducedShape = reducedDims.Length > 0 ? new Shape(reducedDims) : Shape.NewScalar();

            // -------------------------------------------------------------------------------------
            // Plan P2 M2 — axis Sum/Mean/Prod at NumPy-EXACT precision (mirrors the flat M1 divert).
            //
            // program.AxisReduceKernel below is the 4-accumulator fold: an order-of-summation drift
            // from NumPy's add.reduce (pairwise on a contiguous reduced axis, sequential otherwise)
            // and a lane-interleaved product that reorders the non-associative float multiply —
            // MisalignedRegistry E1. For the dtypes NumPy reduces through a well-defined schedule we
            // reproduce it BIT-FOR-BIT: materialize the child through the already-bit-exact
            // elementwise engine (a fresh, C-contiguous K-order array — the same intermediate NumPy's
            // own unfused np.<reduce>(a*b, axis) would build) and reduce THAT buffer with the exact
            // schedule np.sum · np.prod use. Sum/Mean ride the engine's own axis add.reduce (the
            // IL-emitted pairwise kernel — bit-for-bit np.add.reduce on every layout, PINNED for a
            // contiguous reduced axis, SLAB-sequential for an outer one); Prod runs a sequential
            // memory-order fold (np.multiply.reduce is never pairwise).
            //
            // Gated to a C-contiguous materialized child (!all-strict-F): a fresh child is C-contig
            // unless every input is strictly F-contiguous, in which case NumPy's own reduce would
            // pairwise the (F-contiguous) axis-0 and yield an F-ordered result — an F-layout corner
            // left on the folded path (still E1-excused, and value-exact over the corpus's benign
            // pools). Deliberately NOT diverted (each a later milestone): Half at any axis (no float16
            // pairwise kernel yet); complex Prod (a complex-multiply chain, npy_cmul FMA-contracted on
            // NumPy's win-amd64 build, not on .NET's — the documented multiply gap #12, which an order
            // fix cannot close). Empty reductions (axisSize == 0 / reducedShape.size == 0) stay on the
            // folded path — its seeded 0/1 identities are order-independent and already value-exact.
            var childProgram = program.ChildElementwiseProgram;
            var childType = childProgram.ResultType;
            bool diverts = axisSize > 0 && reducedShape.size > 0
                           && !AreAllInputsStrictFContig(ops, inputShape)
                           && (((reduce.Kind == NDExprReduceKind.Sum || reduce.Kind == NDExprReduceKind.Mean)
                                && (childType == NPTypeCode.Single || childType == NPTypeCode.Double || childType == NPTypeCode.Complex))
                               || (reduce.Kind == NDExprReduceKind.Prod
                                   && (childType == NPTypeCode.Single || childType == NPTypeCode.Double)));

            // The dtype of the reduced accumulator (outAcc): the child dtype on the divert path
            // (NumPy accumulates a float sum at the INPUT precision — f32 stays f32, never the f64
            // accType the fold path widens to), else the folded path's accType.
            NPTypeCode redType = diverts ? childType : accType;
            NDArray outAcc;

            if (diverts)
            {
                // Materialize the child once (fresh + C-contiguous); `using` releases it after the
                // reduce — outAcc is a fresh array, never a view into it, so this is safe.
                using var materialized = EvaluateCore(childProgram, inputs, null);
                outAcc = reduce.Kind == NDExprReduceKind.Prod
                    ? SequentialAxisProd(materialized, axis, childType, reducedShape)   // np.multiply.reduce order
                    : ExactAxisSum(materialized, axis, childType, reducedShape);        // np.add.reduce (pairwise / SLAB)
            }
            else
            {
                outAcc = new NDArray(accType, reducedShape, false);
                if (outAcc.size != 0)
                {
                    // Seed the accumulator with the reduction identity (Mean accumulates a Sum).
                    var seedOp = reduce.Kind switch
                    {
                        NDExprReduceKind.Prod => ReductionOp.Prod,
                        NDExprReduceKind.Min => ReductionOp.Min,
                        NDExprReduceKind.Max => ReductionOp.Max,
                        // logical_or / logical_and: SeedReduceIdentity writes False / True per output slot
                        // (GetIdentity(Any) == false, GetIdentity(All) == true); the kernel folds OR / AND.
                        NDExprReduceKind.Any => ReductionOp.Any,
                        NDExprReduceKind.All => ReductionOp.All,
                        _ => ReductionOp.Sum, // Sum / Mean
                    };
                    ILKernelGenerator.SeedReduceIdentity(outAcc, seedOp);

                    if (axisSize != 0)
                    {
                        var kernel = program.AxisReduceKernel;

                        // op_axes: identity for every input; output maps reduce axis → -1 (stride 0).
                        var opAxes = new int[ops.Length + 1][];
                        for (int i = 0; i < ops.Length; i++)
                        {
                            var a = new int[ndim];
                            for (int d = 0; d < ndim; d++) a[d] = d;
                            opAxes[i] = a;
                        }
                        var outAxes = new int[ndim];
                        for (int d = 0, oc = 0; d < ndim; d++) outAxes[d] = (d == axis) ? -1 : oc++;
                        opAxes[ops.Length] = outAxes;

                        var operands = new NDArray[ops.Length + 1];
                        for (int i = 0; i < ops.Length; i++)
                        {
                            bool same = ops[i].ndim == ndim;
                            if (same)
                                for (int d = 0; d < ndim; d++)
                                    if (ops[i].shape[d] != inputShape[d]) { same = false; break; }
                            operands[i] = same ? ops[i] : np.broadcast_to(ops[i], inputShape);
                        }
                        operands[ops.Length] = outAcc;

                        var opFlags = new NDIterPerOpFlags[ops.Length + 1];
                        for (int i = 0; i < ops.Length; i++) opFlags[i] = NDIterPerOpFlags.READONLY;
                        opFlags[ops.Length] = NDIterPerOpFlags.READWRITE;

                        using var iter = NDIterRef.AdvancedNew(
                            operands.Length, operands,
                            NDIterGlobalFlags.REDUCE_OK | NDIterGlobalFlags.EXTERNAL_LOOP,
                            NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                            opFlags, null, ndim, opAxes);

                        // Same aux layout as the flat kernel (slot 0 reserved, parameters after it).
                        byte* aux = null;
                        if (program.ParamCount > 0)
                        {
                            // stackalloc memory lives until the METHOD returns, so the block-scoped declaration is safe.
                            byte* buf = stackalloc byte[NDExprParamPlan.SlotBytes * (1 + program.ParamCount)];
                            program.PackParams(inputs, buf + NDExprParamPlan.ReduceParamOffset);
                            aux = buf;
                        }

                        iter.ForEach(kernel, aux);
                    }
                }
            }

            // Mean: divide by the reduced-axis count at the accumulator dtype (redType), matching
            // NumPy's _methods._mean — f32 divides in f32, complex through the Smith-form
            // reciprocal-multiply (component-wise division keeps the other part finite and diverges
            // on a NaN slice). Shared by both paths.
            if (reduce.Kind == NDExprReduceKind.Mean && outAcc.size != 0)
            {
                if (redType == NPTypeCode.Complex)
                {
                    // np.mean divides the complex sum by the count through the COMPLEX true_divide
                    // loop, not component-wise — see ComplexDivideByCountLikeNumPy.
                    for (long i = 0; i < outAcc.size; i++)
                        outAcc.SetAtIndex(ComplexDivideByCountLikeNumPy((System.Numerics.Complex)outAcc.GetAtIndex(i), axisSize), i);
                }
                else
                {
                    ILKernelGenerator.MeanDivideByCount(outAcc, axisSize);
                }
            }

            // Cast accumulator dtype → result dtype (no-op when equal — the divert path already
            // reduces at the result precision; the fold path's f16/f32 mean accumulates in double
            // and narrows here).
            NDArray reduced = redType == resultType ? outAcc : Cast(outAcc, resultType, copy: true);

            if (reduce.Keepdims)
            {
                var kd = new long[ndim];
                for (int d = 0, rd = 0; d < ndim; d++) kd[d] = (d == axis) ? 1 : reduced.shape[rd++];
                reduced = reduced.reshape(kd);
            }

            if (@out is not null) { np.copyto(@out, reduced); return @out; }
            return reduced;
        }

        /// <summary>
        /// np.mean's final <c>true_divide(sum, count)</c> runs NumPy's COMPLEX division loop (Smith's
        /// method with the real divisor: <c>rat = 0, scl = 1/n</c>), so <c>out_r = (re + im*0)*scl</c>
        /// and <c>out_i = (im - re*0)*scl</c> — a NaN or inf in EITHER part poisons BOTH
        /// (<c>nan*0 = nan</c>). System.Numerics' component-wise <c>z / n</c> keeps the other part
        /// finite and diverged from np.mean on every NaN-carrying complex slice (probed 2.4.2:
        /// <c>np.mean([nan+1j, 2+3j]) == nan+nanj</c>).
        /// </summary>
        private static System.Numerics.Complex ComplexDivideByCountLikeNumPy(System.Numerics.Complex z, long n)
        {
            double scl = 1.0 / n;
            double re = z.Real, im = z.Imaginary;
            return new System.Numerics.Complex((re + im * 0.0) * scl, (im - re * 0.0) * scl);
        }

        /// <summary>
        /// NumPy-exact pairwise sum of <paramref name="n"/> CONTIGUOUS <paramref name="tc"/> elements at
        /// <paramref name="src"/> into the 16-byte accumulator at <paramref name="dst"/> (which the caller
        /// pre-zeroes). Reuses the SAME kernel np.add.reduce runs
        /// (<see cref="ILKernelGenerator.TryEmitPairwiseSumKernel"/>) through its PINNED route — a
        /// stride-0 output makes the kernel fold the whole input into the single slot
        /// (<c>*dst += fold(src, n, 1)</c>) — so the result is bit-identical to NumPy's <c>pairwise_sum</c>.
        /// Only float32 / float64 / complex128 have a pairwise kernel; the flat-reduce divert gates on
        /// exactly those, so a missing kernel here is a wiring bug, not a runtime input case.
        /// </summary>
        /// <param name="src">Base of the contiguous source buffer (logical element 0).</param>
        /// <param name="n">Element count (must be &gt; 0).</param>
        /// <param name="tc">The element dtype (Single, Double or Complex).</param>
        /// <param name="dst">The pre-zeroed 16-byte accumulator slot the sum is written into.</param>
        /// <exception cref="NotSupportedException">No pairwise sum kernel exists for <paramref name="tc"/>.</exception>
        private static unsafe void PairwiseSumInto(byte* src, long n, NPTypeCode tc, byte* dst)
        {
            var kernel = ILKernelGenerator.TryEmitPairwiseSumKernel(tc)
                ?? throw new NotSupportedException($"No pairwise sum kernel for {tc}.");
            // The pairwise kernel is an NDInnerLoopFunc over operands [in, out]: a stride-0 OUTPUT
            // selects its PINNED route, which reduces the whole `n`-element contiguous input into the
            // single (pre-zeroed) accumulator slot.
            void** dataptrs = stackalloc void*[2];
            long* strides = stackalloc long[2];
            dataptrs[0] = src;
            dataptrs[1] = dst;
            strides[0] = tc.SizeOf();   // contiguous input
            strides[1] = 0;             // stride 0 ⇒ reduce into dst
            kernel(dataptrs, strides, n, null);
        }

        /// <summary>
        /// NumPy-exact sequential product of <paramref name="n"/> contiguous elements into
        /// <paramref name="dst"/> — a plain scalar <c>acc *= x</c> chain, matching np.multiply.reduce
        /// exactly (a lane-interleaved fold would REORDER the non-associative float multiply and diverge).
        /// float32 / float64 only; complex Prod stays on the folded path (its npy_cmul FMA gap is
        /// documented as #12, not an order artifact this can fix).
        /// </summary>
        /// <param name="src">Base of the contiguous source buffer.</param>
        /// <param name="n">Element count (must be &gt; 0).</param>
        /// <param name="tc">The element dtype (Single or Double).</param>
        /// <param name="dst">The accumulator slot the product is written into.</param>
        /// <exception cref="NotSupportedException"><paramref name="tc"/> is not Single or Double.</exception>
        private static unsafe void SequentialProductInto(byte* src, long n, NPTypeCode tc, byte* dst)
        {
            switch (tc)
            {
                case NPTypeCode.Single:
                {
                    float* p = (float*)src;
                    float acc = 1f;
                    for (long i = 0; i < n; i++) acc *= p[i];
                    *(float*)dst = acc;
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* p = (double*)src;
                    double acc = 1d;
                    for (long i = 0; i < n; i++) acc *= p[i];
                    *(double*)dst = acc;
                    break;
                }
                default:
                    throw new NotSupportedException($"Sequential product for {tc} is not diverted here.");
            }
        }

        /// <summary>
        /// Divide the mean accumulator in place by the element count at the RESULT dtype, matching
        /// NumPy's <c>_methods._mean</c>: float32 divides in float32 (the count cast to float32) and
        /// float64 in float64, and complex divides through the same Smith-form reciprocal-multiply np
        /// uses (<see cref="ComplexDivideByCountLikeNumPy"/>, which poisons BOTH components on a NaN part).
        /// </summary>
        /// <param name="acc">The accumulator slot holding the sum on entry, the mean on return.</param>
        /// <param name="tc">The accumulator dtype (Single, Double or Complex).</param>
        /// <param name="n">The element count the mean divides by.</param>
        /// <exception cref="NotSupportedException"><paramref name="tc"/> is not a diverted mean dtype.</exception>
        private static unsafe void DivideAccByCount(byte* acc, NPTypeCode tc, long n)
        {
            switch (tc)
            {
                case NPTypeCode.Single:
                    *(float*)acc /= (float)n;    // np.float32 / n → float32 (n cast to float32, as NumPy does)
                    break;
                case NPTypeCode.Double:
                    *(double*)acc /= (double)n;
                    break;
                case NPTypeCode.Complex:
                    *(System.Numerics.Complex*)acc =
                        ComplexDivideByCountLikeNumPy(*(System.Numerics.Complex*)acc, n);
                    break;
                default:
                    throw new NotSupportedException($"Mean divide for {tc} is not diverted here.");
            }
        }

        /// <summary>
        /// NumPy-exact axis <c>add.reduce</c> of a FRESH, C-contiguous <paramref name="child"/> along
        /// <paramref name="axis"/> (plan P2 M2), returning a fresh reduced array of dtype
        /// <paramref name="tc"/> and shape <paramref name="reducedShape"/>. A 1-D child (reduce-to-
        /// scalar) folds through the flat pairwise helper — which also covers Complex, dodging the
        /// engine's complex 1-D axis-reduce throw; a multi-D child delegates to the engine's own axis
        /// <see cref="DefaultEngine.ReduceAdd"/>, whose IL-emitted pairwise kernel is bit-for-bit
        /// identical to NumPy's <c>pairwise_sum</c> on a contiguous input (pairwise when the reduced
        /// axis is contiguous, sequential streaming when it is outer) and allocates a C-order result
        /// (C input → C output), so the bytes equal <c>np.add.reduce(materialized child, axis)</c>.
        /// Only Single / Double / Complex reach here (the M2 divert gate); <paramref name="tc"/> is
        /// passed as the loop dtype so the reduction accumulates at the child's precision (a float sum
        /// stays float, never the widened f64 accType the fold path uses).
        /// </summary>
        /// <param name="child">The freshly materialized, C-contiguous child expression result.</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="tc">The child (and reduced) dtype — Single, Double or Complex.</param>
        /// <param name="reducedShape">The output shape (input shape with <paramref name="axis"/> removed; scalar for a 1-D child).</param>
        /// <returns>A fresh reduced array; a writeable 0-d scalar for a 1-D child, else the engine's axis-reduce result.</returns>
        private unsafe NDArray ExactAxisSum(NDArray child, int axis, NPTypeCode tc, Shape reducedShape)
        {
            if (child.ndim == 1)
            {
                // Reduce-all over the only axis → a scalar; the flat pairwise fold IS np.add.reduce and
                // handles Complex (unlike the engine's 1-D complex axis path, which throws).
                byte* accSlot = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)accSlot = 0;
                *(ulong*)(accSlot + 8) = 0;
                byte* src = (byte*)child.Address + (long)child.Shape.offset * tc.SizeOf();
                PairwiseSumInto(src, child.size, tc, accSlot);
                var scalar = new NDArray(tc, reducedShape, false);
                NDIterCasting.ConvertValue(accSlot,
                    (byte*)scalar.Address + (long)scalar.Shape.offset * tc.SizeOf(), tc, tc);
                return scalar;
            }

            // Multi-D: the engine's axis add.reduce over the C-contiguous child rides the same pairwise
            // kernel np.add.reduce runs — bit-exact on every layout — and returns a C-order result the
            // size of reducedShape. The explicit dtype keeps the accumulation at the child precision.
            return ReduceAdd(child, axis, keepdims: false, dtype: DType.From(tc), @out: null);
        }

        /// <summary>
        /// NumPy-exact axis <c>multiply.reduce</c> of a FRESH, C-contiguous <paramref name="child"/>
        /// along <paramref name="axis"/> (plan P2 M2). NumPy's product reduction is never pairwise —
        /// it accumulates in increasing axis-coordinate order — so this is a plain sequential fold in
        /// that order, which the child's C-contiguous layout makes a straight memory walk:
        /// <list type="bullet">
        ///   <item>innermost axis (<c>inner == 1</c>): each output is the sequential product of one
        ///         contiguous <c>axisSize</c>-stripe.</item>
        ///   <item>outer axis (<c>inner &gt; 1</c>): SLAB — seed the output run to 1, then stream
        ///         <c>out[o·inner+i] *= child[(o·axisSize+j)·inner+i]</c> over increasing j, exactly
        ///         NumPy's outer-axis reduce loop.</item>
        /// </list>
        /// Both are bit-identical to <c>np.multiply.reduce(child, axis)</c>. Only Single / Double reach
        /// here — complex Prod stays folded (npy_cmul FMA gap #12, not an order artifact this can fix).
        /// </summary>
        /// <param name="child">The freshly materialized, C-contiguous child expression result.</param>
        /// <param name="axis">The already-normalized reduction axis.</param>
        /// <param name="tc">The child (and reduced) dtype — Single or Double.</param>
        /// <param name="reducedShape">The output shape (input shape with <paramref name="axis"/> removed).</param>
        /// <returns>A fresh C-contiguous reduced array holding the sequential products.</returns>
        /// <exception cref="NotSupportedException"><paramref name="tc"/> is not Single or Double.</exception>
        private static unsafe NDArray SequentialAxisProd(NDArray child, int axis, NPTypeCode tc, Shape reducedShape)
        {
            var dims = child.Shape.dimensions;
            int nd = dims.Length;
            long axisSize = dims[axis];
            long inner = 1; for (int d = axis + 1; d < nd; d++) inner *= dims[d];
            long outer = 1; for (int d = 0; d < axis; d++) outer *= dims[d];

            var result = new NDArray(tc, reducedShape, false); // C-contiguous, outer*inner elements
            byte* srcB = (byte*)child.Address + (long)child.Shape.offset * tc.SizeOf();
            byte* dstB = (byte*)result.Address + (long)result.Shape.offset * tc.SizeOf();

            switch (tc)
            {
                case NPTypeCode.Single:
                {
                    float* src = (float*)srcB;
                    float* dst = (float*)dstB;
                    if (inner == 1)
                    {
                        for (long o = 0; o < outer; o++)
                        {
                            float acc = 1f;
                            long baseIdx = o * axisSize;
                            for (long j = 0; j < axisSize; j++) acc *= src[baseIdx + j];
                            dst[o] = acc;
                        }
                    }
                    else
                    {
                        long outCount = outer * inner;
                        for (long k = 0; k < outCount; k++) dst[k] = 1f;
                        for (long o = 0; o < outer; o++)
                            for (long j = 0; j < axisSize; j++)
                            {
                                long b = (o * axisSize + j) * inner;
                                long ob = o * inner;
                                for (long i = 0; i < inner; i++) dst[ob + i] *= src[b + i];
                            }
                    }
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* src = (double*)srcB;
                    double* dst = (double*)dstB;
                    if (inner == 1)
                    {
                        for (long o = 0; o < outer; o++)
                        {
                            double acc = 1d;
                            long baseIdx = o * axisSize;
                            for (long j = 0; j < axisSize; j++) acc *= src[baseIdx + j];
                            dst[o] = acc;
                        }
                    }
                    else
                    {
                        long outCount = outer * inner;
                        for (long k = 0; k < outCount; k++) dst[k] = 1d;
                        for (long o = 0; o < outer; o++)
                            for (long j = 0; j < axisSize; j++)
                            {
                                long b = (o * axisSize + j) * inner;
                                long ob = o * inner;
                                for (long i = 0; i < inner; i++) dst[ob + i] *= src[b + i];
                            }
                    }
                    break;
                }
                default:
                    throw new NotSupportedException($"Sequential axis product for {tc} is not diverted here.");
            }

            return result;
        }

        private static unsafe void WriteOne(byte* slot, NPTypeCode accType)
        {
            switch (accType)
            {
                case NPTypeCode.Int64: *(long*)slot = 1; break;
                case NPTypeCode.UInt64: *(ulong*)slot = 1; break;
                case NPTypeCode.Double: *(double*)slot = 1.0; break;
                case NPTypeCode.Decimal: *(decimal*)slot = 1m; break;
                case NPTypeCode.Complex: *(System.Numerics.Complex*)slot = System.Numerics.Complex.One; break;
                default:
                    throw new NotSupportedException($"prod accumulator {accType} — typing bug.");
            }
        }

        private static unsafe void WriteMeanOfEmpty(byte* slot, NPTypeCode accType)
        {
            switch (accType)
            {
                case NPTypeCode.Double: *(double*)slot = double.NaN; break;
                case NPTypeCode.Decimal: *(decimal*)slot = 0m; break; // decimal has no NaN
                case NPTypeCode.Complex: *(System.Numerics.Complex*)slot = new System.Numerics.Complex(double.NaN, double.NaN); break;
                default:
                    throw new NotSupportedException($"mean accumulator {accType} — typing bug.");
            }
        }

        private static unsafe void WriteMinMaxIdentity(byte* slot, NPTypeCode accType, bool isMin)
        {
            switch (accType)
            {
                case NPTypeCode.Boolean: *slot = isMin ? (byte)1 : (byte)0; break;
                case NPTypeCode.Byte: *slot = isMin ? byte.MaxValue : byte.MinValue; break;
                case NPTypeCode.SByte: *(sbyte*)slot = isMin ? sbyte.MaxValue : sbyte.MinValue; break;
                case NPTypeCode.Int16: *(short*)slot = isMin ? short.MaxValue : short.MinValue; break;
                case NPTypeCode.UInt16: *(ushort*)slot = isMin ? ushort.MaxValue : ushort.MinValue; break;
                case NPTypeCode.Char: *(char*)slot = isMin ? char.MaxValue : char.MinValue; break;
                case NPTypeCode.Int32: *(int*)slot = isMin ? int.MaxValue : int.MinValue; break;
                case NPTypeCode.UInt32: *(uint*)slot = isMin ? uint.MaxValue : uint.MinValue; break;
                case NPTypeCode.Int64: *(long*)slot = isMin ? long.MaxValue : long.MinValue; break;
                case NPTypeCode.UInt64: *(ulong*)slot = isMin ? ulong.MaxValue : ulong.MinValue; break;
                case NPTypeCode.Half: *(Half*)slot = isMin ? Half.PositiveInfinity : Half.NegativeInfinity; break;
                case NPTypeCode.Single: *(float*)slot = isMin ? float.PositiveInfinity : float.NegativeInfinity; break;
                case NPTypeCode.Double: *(double*)slot = isMin ? double.PositiveInfinity : double.NegativeInfinity; break;
                case NPTypeCode.Decimal: *(decimal*)slot = isMin ? decimal.MaxValue : decimal.MinValue; break;
                case NPTypeCode.Complex:
                    // (±inf, ±inf) is the identity under the lexicographic (real, imag) order the
                    // complex clamp folds with — mirrors ILKernelGenerator.SeedReduceIdentity.
                    *(System.Numerics.Complex*)slot = isMin
                        ? new System.Numerics.Complex(double.PositiveInfinity, double.PositiveInfinity)
                        : new System.Numerics.Complex(double.NegativeInfinity, double.NegativeInfinity);
                    break;
                default:
                    throw new NotSupportedException($"min/max accumulator {accType} — typing bug.");
            }
        }
    }
}
