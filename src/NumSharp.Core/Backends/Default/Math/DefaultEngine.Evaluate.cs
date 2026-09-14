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

        // Per-arity flag arrays for the MASKED (where=) evaluate config — the same input flags as
        // EvalElementwiseFlags, but the output is WRITEMASKED (so ForEach's masked driver only writes
        // its mask-true runs, leaving masked-off slots untouched) and a trailing ARRAYMASK mask operand
        // is appended (NumPy's ufunc where= layout: the mask rides as op[nop], outputs WRITEMASKED —
        // ufunc_object.c:2190-2226). The array is nIn+2 long: [inputs…, masked output, mask].
        private static readonly ConcurrentDictionary<int, NDIterPerOpFlags[]> s_evalElementwiseMaskedFlags = new();

        private static NDIterPerOpFlags[] EvalElementwiseMaskedFlags(int nIn)
            => s_evalElementwiseMaskedFlags.GetOrAdd(nIn, static n =>
            {
                var flags = new NDIterPerOpFlags[n + 2];
                for (int i = 0; i < n; i++)
                    flags[i] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                flags[n] = NDIterPerOpFlags.WRITEONLY
                           | NDIterPerOpFlags.WRITEMASKED
                           | NDIterPerOpFlags.NO_BROADCAST
                           | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                flags[n + 1] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK;
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
        public override unsafe NDArray Evaluate(NDExpr expr, NDArray @out = null, in NDEvaluateOptions options = default)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));

            // The root's slot, else the global structural cache, else a build — and the distinct arrays the
            // tree references in binding order (the operand list the program was compiled against).
            var program = expr.GetProgram(null, out var operands);
            return EvaluateCore(program, operands, @out, options);
        }

        /// <summary>
        /// Evaluate a tree built over positional <see cref="NDExpr.Input"/>
        /// leaves against an explicit operand list.
        /// </summary>
        public override unsafe NDArray Evaluate(NDExpr expr, NDArray[] operands, NDArray @out = null, in NDEvaluateOptions options = default)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));
            if (operands is null) throw new ArgumentNullException(nameof(operands));
            if (operands.Length == 0)
                throw new ArgumentException("operands must not be empty.", nameof(operands));
            foreach (var op in operands)
                if (op is null)
                    throw new ArgumentNullException(nameof(operands), "no operand may be null.");

            var program = expr.GetProgram(operands);
            return EvaluateCore(program, operands, @out, options);
        }

        /// <summary>
        /// The <see cref="CompiledExpression"/> entry: a program compiled ahead of time, evaluated
        /// against <paramref name="operands"/> (the program's own embedded arrays, or a positional list
        /// the handle has already matched to the compiled dtype signature).
        /// </summary>
        internal override NDArray Evaluate(NDExprProgram program, NDArray[] operands, NDArray @out, in NDEvaluateOptions options = default)
            => EvaluateCore(program, operands, @out, options);

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

        private unsafe NDArray EvaluateCore(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options = default)
        {
            // Plan P4.5 — dtype= and order= shape a FRESH elementwise result, so they are unsupported on a
            // reduction tree (a reduction fixes its own result dtype — its accumulator — and layout, and
            // NumPy's reductions carry no order= parameter). casting= governs out= VALIDATION and threads
            // through to the reduce paths unchanged. Reject the two result-shaping keywords up front, before
            // any work; a bare/explicit 'K' order is the no-op default and does not trip this.
            bool isReduction = program.Average is not null || program.Reduce is not null;
            if (isReduction && (options.HasDtype || options.HasExplicitOrder))
                throw new NotSupportedException(
                    "np.evaluate: dtype= and order= are not supported on a reduction tree (a reduction " +
                    "fixes its result dtype and layout). Cast the reduction result instead: " +
                    "np.evaluate(expr).astype(dtype).");

            // where= is the ufunc masked-WRITE convention (write the result only where the mask is True),
            // which is meaningful only for a per-element result. A masked REDUCTION is a different operation
            // (skip masked-off elements from the accumulation, which needs the right per-kind identity), and
            // np.evaluate's reduction kinds are host-computed without it — so reject where= on a reduction
            // rather than silently ignore it. The remedy is to mask the inputs before reducing.
            if (isReduction && options.HasWhere)
                throw new NotSupportedException(
                    "np.evaluate: where= is not supported on a reduction tree (a masked reduction is a " +
                    "different operation). Mask the inputs before reducing instead, e.g. " +
                    "np.evaluate(NDExpr.Sum(NDExpr.Where(mask, expr, NDExpr.Const(0)))).");

            // A weighted average reduces over TWO sub-trees (values, weights); it is host-computed like
            // the M4c summation kinds, so it precedes the elementwise + single-child-reduce dispatch.
            if (program.Average is not null)
                return EvaluateWeightedAverage(program, inputs, @out, options);

            if (program.Reduce is not null)
                return EvaluateReduce(program, inputs, @out, options);

            // The iterator streams the non-parameter inputs; a 0-d parameter reaches the kernel through
            // the aux block instead (NDExpr.Params.cs) and never joins the broadcast — it has no dims.
            var ops = program.IteratorOperands(inputs);
            var kernel = program.Kernel;
            var resolvedType = program.ResultType;

            // dtype= is an implicit root cast (plan P4.5): the tree still COMPUTES at its natural NEP50
            // result type, and the result is cast to the requested dtype on the way to a FRESH buffer — the
            // exact out= buffered-cast machinery below. dtype= and a caller out= are mutually exclusive (the
            // API rejects both), so this only ever widens a fresh allocation's dtype.
            var targetType = options.Dtype ?? resolvedType;

            // where= (plan P4.5): the boolean write mask. NumPy requires it to be exactly bool (its
            // _wheremask_converter casts with the 'safe' rule, which only bool→bool passes) — validate it
            // with the same check the ufunc where= path uses, so the verbatim TypeError text matches. When
            // present it rides the iterator as a trailing ARRAYMASK operand below and ForEach's masked driver
            // runs the fused kernel per mask-true run, leaving masked-off destination slots untouched.
            NDArray where = options.Where;
            if (where is not null)
                ValidateWhereMask(where);

            if (@out is not null)
                ValidateOutCast(resolvedType, @out.typecode, "evaluate", options.Casting);

            // Iteration shape: the inputs' broadcast (one clone for identical dims, one fresh dims
            // array otherwise — ResolveInputShape), then out and the where mask join per the ufunc rules
            // (both broadcast in, but a provided out is never stretched — its verbatim errors live in
            // ResolveUfuncIterationShape). A where= without out= still joins the mask so the fresh result's
            // shape includes it (NumPy: where broadcasts with everything).
            Shape inputShape = ResolveInputShape(ops);
            Shape iterShape = (@out is null && where is null)
                ? inputShape
                : ResolveUfuncIterationShape(inputShape, ops, @out, where).Clean();

            // NumPy-aligned layout preservation (mirrors TryExecuteBinaryOpViaNDIter):
            // by default ('K') the result is column-major only when every input operand is
            // strictly F-contiguous, so the iterator coalesces to ONE contiguous inner loop.
            // Forcing C-order here made np.evaluate stride across rows on F/transposed
            // operands — ~16x slower than the unfused chain (the fused F/T cliff). order=
            // (plan P4.5) overrides that heuristic for the FRESH result: 'C' forces C, 'F'
            // forces F even from C inputs, 'A'/'K' keep the heuristic. A provided out keeps
            // its own layout (order= applies to a fresh alloc only).
            bool allStrictFContig = AreAllInputsStrictFContig(ops, iterShape);
            bool wantFOrder = @out is null && ResolveEvalOrder(options.Order, allStrictFContig);
            Shape targetShape = wantFOrder
                ? new Shape((long[])iterShape.dimensions.Clone(), 'F')
                : iterShape;

            var target = @out ?? new NDArray(targetType, targetShape, false);
            if (target.size == 0)
                return target;

            // Iterate in the RESULT buffer's own contiguity order so the output writes never
            // stride — F when the target is strictly F-contig (a fresh 'F'/heuristic alloc or a
            // provided F-contig out), C otherwise. Elementwise results are order-independent, so
            // this changes only traversal (perf), never values.
            var order = (target.Shape.IsFContiguous && !target.Shape.IsContiguous)
                ? NPY_ORDER.NPY_FORTRANORDER
                : NPY_ORDER.NPY_CORDER;

            // where= appends ONE more operand (the mask) and flips the output to WRITEMASKED, so the
            // operand/flag/dtype arrays are all one slot longer and the output is followed by the mask.
            bool masked = where is not null;
            int nData = ops.Length;              // input operands (the mask/out are extras)
            int nOps = nData + (masked ? 2 : 1); // inputs + output [+ mask]

            bool outNeedsCast = target.typecode != resolvedType;
            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode[] opDtypes = null;
            if (outNeedsCast)
            {
                // The kernel writes the resolved dtype into the out operand's
                // buffer; the windowed flush casts (same_kind was validated, so
                // the iterator runs UNSAFE exactly like NumPy's ufunc layer, and
                // under WRITEMASKED the same flush also honours the mask).
                globalFlags |= NDIterGlobalFlags.BUFFERED
                             | NDIterGlobalFlags.GROWINNER
                             | NDIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                opDtypes = new NPTypeCode[nOps];
                Array.Copy(program.InputTypes, opDtypes, nData);
                opDtypes[nData] = resolvedType;
                if (masked)
                    opDtypes[nData + 1] = NPTypeCode.Empty; // the mask is never cast (bool → nonzero test)
            }

            var operands = new NDArray[nOps];
            Array.Copy(ops, operands, nData);
            operands[nData] = target;
            if (masked)
                operands[nData + 1] = where;

            using var iter = NDIterRef.MultiNew(
                nOps, operands,
                globalFlags, order, casting,
                masked ? EvalElementwiseMaskedFlags(nData) : EvalElementwiseFlags(nData),
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

        /// <summary>
        /// Resolve np.evaluate's <c>order=</c> (plan P4.5) to whether the FRESH elementwise result is
        /// allocated F-contiguous: <c>'C'</c> → never (row-major), <c>'F'</c> → always (column-major, even
        /// from C-contiguous inputs), <c>'A'</c>/<c>'K'</c> → today's heuristic (<paramref
        /// name="allStrictFContig"/> — F only when every input is strictly F-contiguous). The order char is
        /// already validated at the API boundary, so the default arm is unreachable for a legal call and is
        /// only a defensive C fallback.
        /// </summary>
        /// <param name="order">The (API-validated) order char: 'C'/'F'/'A'/'K', case-insensitive.</param>
        /// <param name="allStrictFContig">Whether the input operands make the <c>'K'</c> heuristic prefer F.</param>
        /// <returns>True to allocate the fresh result column-major (F), false for row-major (C).</returns>
        private static bool ResolveEvalOrder(char order, bool allStrictFContig)
        {
            switch (order)
            {
                case 'C':
                case 'c':
                    return false;
                case 'F':
                case 'f':
                    return true;
                case 'A':
                case 'a':
                case 'K':
                case 'k':
                    return allStrictFContig;
                default:
                    return allStrictFContig; // API-validated; defensive
            }
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
            NDExprReduceKind.ArgMax => "argmax",
            NDExprReduceKind.ArgMin => "argmin",
            NDExprReduceKind.NanMean => "nanmean",
            NDExprReduceKind.Var => "var",
            NDExprReduceKind.Std => "std",
            _ => "mean",
        };

        // ---- flat keepdims (plan P2 M5) ----------------------------------------------------------
        // NumPy's reductions honor keepdims with axis=None too: np.sum(a, axis=None, keepdims=True) is
        // shape (1,)*a.ndim, one element kept broadcast-friendly. Every reduce host path (fold /
        // delegating / stat / average) produces the flat result as a 0-d scalar, so keepdims is a pure
        // SHAPE post-step — reshape that one element to (1,…,1). These three helpers keep it uniform
        // across the four paths. Tuple axis is NOT offered (every NumSharp reduction is single-axis).

        /// <summary>
        /// The result shape for a FLAT (axis=None) reduction: the 0-d scalar shape without keepdims, or
        /// (1,)*<paramref name="childNdim"/> with it (NumPy's <c>keepdims=True</c> shape). A 0-d child
        /// (<paramref name="childNdim"/> 0) stays 0-d either way, matching <c>np.sum(scalar, keepdims=True)</c>.
        /// </summary>
        /// <param name="childNdim">The reduced expression's rank.</param>
        /// <param name="keepdims">Whether the reduced axes are kept as size 1.</param>
        /// <returns>A 0-d scalar shape, or the all-ones (1,…,1) shape of rank <paramref name="childNdim"/>.</returns>
        private static Shape FlatReduceShape(int childNdim, bool keepdims)
        {
            if (!keepdims || childNdim == 0)
                return Shape.NewScalar();
            var dims = new long[childNdim];
            for (int i = 0; i < childNdim; i++) dims[i] = 1;
            return new Shape(dims);
        }

        /// <summary>
        /// Normalize a FLAT reduction's single-element result to its NumPy shape: (1,)*<paramref
        /// name="childNdim"/> when <paramref name="keepdims"/> is set, else a 0-d scalar. The result
        /// holds exactly one element, so this is a free reshape — the VALUE is identical, only the
        /// wrapper rank differs; a 0-d child stays 0-d. Used by the delegating / stat / average paths,
        /// which produce their flat result as a materialized <see cref="NDArray"/> (the fold path
        /// allocates the shape directly via <see cref="FlatReduceShape"/>).
        /// <para>
        /// The <c>keepdims=false</c> branch reshapes to 0-d rather than passing the input through — a
        /// FLAT reduce is a scalar, and a DELEGATED engine reduction can hand back a spurious rank:
        /// <c>np.nanmin</c>/<c>np.nanmax</c> on a size-1 1-D input return shape <c>(1,)</c>, not <c>()</c>
        /// (a pre-existing engine quirk the fold/ptp/argmax paths do not have). Normalizing here keeps
        /// evaluate's flat contract 0-d regardless of the delegate's shape.
        /// </para>
        /// </summary>
        /// <param name="reduced">The single-element result of a flat reduction (0-d or a spurious (1,)).</param>
        /// <param name="childNdim">The reduced expression's rank.</param>
        /// <param name="keepdims">Whether to keep every axis as size 1.</param>
        /// <returns>A 0-d scalar (keepdims off), or a (1,…,1) view (keepdims on).</returns>
        private static NDArray KeepdimsFlat(NDArray reduced, int childNdim, bool keepdims)
        {
            if (keepdims && childNdim > 0)
            {
                var dims = new long[childNdim];
                for (int i = 0; i < childNdim; i++) dims[i] = 1;
                return reduced.reshape(dims);
            }
            // A flat reduce is a 0-d scalar; normalize away any spurious rank from a delegated reduction
            // (Shape.NewScalar() is the canonical 0-d shape the fold path allocates via FlatReduceShape).
            return reduced.ndim == 0 ? reduced : reduced.reshape(Shape.NewScalar());
        }

        /// <summary>
        /// Validate a caller <c>out=</c> for a FLAT reduction: it must be 0-d without keepdims, or exactly
        /// (1,)*<paramref name="childNdim"/> (one element, <paramref name="childNdim"/> axes) with it.
        /// Raises NumPy's "wrong number of dimensions" message — the same text every reduce path used for
        /// the 0-d-only check before M5.
        /// </summary>
        /// <param name="out">The caller-supplied output array.</param>
        /// <param name="childNdim">The reduced expression's rank (the keepdims out rank).</param>
        /// <param name="keepdims">Whether the reduction keeps every axis as size 1.</param>
        /// <param name="ufuncName">The reduce ufunc name, for the message.</param>
        /// <exception cref="ArgumentException">The out rank (or, under keepdims, its element count) does not match the flat result.</exception>
        private static void ValidateFlatReduceOut(NDArray @out, int childNdim, bool keepdims, string ufuncName)
        {
            int expected = keepdims ? childNdim : 0;
            if (@out.ndim != expected || (keepdims && @out.size != 1))
                throw new ArgumentException(
                    $"output parameter for reduction operation {ufuncName} " +
                    $"has the wrong number of dimensions: Found {@out.ndim} but expected {expected}");
        }

        private unsafe NDArray EvaluateReduce(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options = default)
        {
            var reduce = program.Reduce;

            // Plan P2 M4-tail + M4c — the range / NaN-aware min-max kinds AND the int64 index kinds are
            // host-delegated: materialize the child and reduce THAT buffer through the already-NumPy-exact
            // engine reduction. The min/max/ptp kinds are bit-identical regardless of layout because their
            // VALUE does not depend on summation order; argmax/argmin's index does depend on the C-order
            // tie/NaN rule, but the materialized child is the fresh C-contiguous buffer NumPy's own argmax
            // builds, so it matches too. Either way they need neither the fold kernel below nor the
            // pairwise divert the summation kinds do. Handles BOTH flat and axis, so it must run before
            // the axis dispatch.
            if (reduce.Kind is NDExprReduceKind.Ptp or NDExprReduceKind.NanMin or NDExprReduceKind.NanMax
                or NDExprReduceKind.ArgMax or NDExprReduceKind.ArgMin)
                return EvaluateDelegatingReduce(program, inputs, @out, options);

            // Plan P2 M4c (summation kinds) — NanMean/Var/Std. These are NOT order-independent (they
            // sum), so they can't ride EvaluateDelegatingReduce's engine np.nanmean/np.var (whose flat
            // sum is a drifting multi-accumulator fold). Instead the host materializes the child once and
            // reproduces NumPy's nanmean / _var op for op over the SAME pairwise sum the M1/M2 diverts
            // use — bit-exact, no fold kernel, handling BOTH flat and axis (so it precedes the axis
            // dispatch, like the delegating kinds).
            if (reduce.Kind is NDExprReduceKind.NanMean or NDExprReduceKind.Var or NDExprReduceKind.Std)
                return EvaluateStatReduce(program, inputs, @out, options);

            if (reduce.Axis is int ax)
                return EvaluateAxisReduce(program, inputs, @out, ax, options);

            var ops = program.IteratorOperands(inputs);
            var accType = program.ReduceAccType;
            var resultType = program.ResultType;

            // The element count AND the reduced expression's rank in one pass: a flat keepdims reduce
            // returns shape (1,)*childNdim (plan P2 M5), so the rank must be known before the out check
            // and the result allocation, not only for the reduction loop.
            long n;
            int childNdim;
            if (AllSameDims(ops))
            {
                n = ops[0].size;
                childNdim = ops[0].ndim;
            }
            else
            {
                var bdims = BroadcastInputDims(ops);
                n = 1;
                foreach (var d in bdims)
                    n *= d;
                childNdim = bdims.Length;
            }

            if (@out is not null)
            {
                ValidateOutCast(resultType, @out.typecode, "evaluate", options.Casting);
                ValidateFlatReduceOut(@out, childNdim, reduce.Keepdims, ReduceUfuncName(reduce.Kind));
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

                    // Flat keepdims (plan P2 M5) allocates (1,)*childNdim — still ONE element at offset
                    // 0, so the direct single-element write below is unchanged; only the wrapper rank
                    // differs. A caller out= (already ValidateFlatReduceOut'd) is used as given.
                    var reduceResult = @out ?? new NDArray(resultType, FlatReduceShape(childNdim, reduce.Keepdims), false);
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

            // Flat keepdims (plan P2 M5): (1,)*childNdim is one element at offset 0, so the direct write
            // is unchanged — only the result's wrapper rank differs from the 0-d scalar form.
            var result = @out ?? new NDArray(resultType, FlatReduceShape(childNdim, reduce.Keepdims), false);
            byte* dst = (byte*)result.Address
                        + (long)result.Shape.offset * result.typecode.SizeOf();
            NDIterCasting.ConvertValue(slot, dst, accType, result.typecode);
            return result;
        }

        /// <summary>
        /// Host-delegated reductions (plan P2 M4-tail + M4c): the value kinds <c>Ptp</c> / <c>NanMin</c> /
        /// <c>NanMax</c> and the int64 index kinds <c>ArgMax</c> / <c>ArgMin</c>. Materializes the
        /// reduction's child once (the M1/M2 route — a fresh, contiguous array in NumPy's own K-order
        /// layout) and reduces THAT buffer through the corresponding public engine reduction
        /// (<see cref="np.ptp(NDArray,int?,NDArray,bool)"/> / <see cref="np.nanmin(NDArray,int?,bool)"/> /
        /// <see cref="np.nanmax(NDArray,int?,bool)"/> / <see cref="np.argmax(NDArray,int,bool)"/> /
        /// <see cref="np.argmin(NDArray,int,bool)"/>), which are themselves NumPy-exact.
        /// <para>
        /// Ptp/NanMin/NanMax are ORDER-INDEPENDENT — a minimum, a maximum, or their difference is the same
        /// value whatever order the elements are visited in — so reducing the materialized buffer is
        /// bit-identical to <c>np.&lt;kind&gt;(child)</c> for ANY input layout. ArgMax/ArgMin are the
        /// subtler M4c case: their index DOES depend on the C-order tie/NaN rule (first maximum / first
        /// NaN wins), but the materialized child is exactly the fresh C-contiguous buffer NumPy's own
        /// <c>argmax</c> would build and reduce, so their index matches too. Both properties are what the
        /// summation kinds (Sum/Mean/Prod, and the still-open NanMean/Std/Var) lack — why those need the
        /// pairwise divert instead.
        /// </para>
        /// The value kinds preserve the child dtype and the index kinds always yield int64, so
        /// (<see cref="ReduceNode.ResolveReduceResultType"/>) already equals the engine reduction's own
        /// result dtype — no post-cast is needed except into a caller <paramref name="out"/> of a
        /// different (same_kind-castable) dtype. A flat <c>argmax</c>/<c>argmin</c> returns a scalar
        /// <c>long</c>, wrapped into a 0-d int64 array here.
        /// </summary>
        /// <param name="program">The compiled reduction program; its <c>Reduce</c> carries the kind, axis and keepdims.</param>
        /// <param name="inputs">The call's operand arrays, driving the child's materialization.</param>
        /// <param name="out">Optional destination; must be 0-d for a flat reduction and the reduced shape for an axis one. Null allocates a fresh result.</param>
        /// <returns>The reduced array (a fresh 0-d scalar / reduced-shape array, or <paramref name="out"/> itself when supplied).</returns>
        /// <exception cref="ArgumentException">A flat reduction was given a non-0-d <paramref name="out"/>; or the engine reduction rejects a zero-size input (no identity), an out shape mismatch, or an out dtype not reachable by a same_kind cast.</exception>
        private unsafe NDArray EvaluateDelegatingReduce(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options = default)
        {
            var reduce = program.Reduce;
            var resultType = program.ResultType;

            // Validate the out cast up front (same order as the fold paths: the cast rule is checked
            // before any compute), so an illegal out dtype fails with evaluate's message, not the
            // engine reduction's.
            if (@out is not null)
                ValidateOutCast(resultType, @out.typecode, "evaluate", options.Casting);

            // Materialize the child once (fresh + contiguous). `using` releases it after the delegated
            // reduction has read it; every delegated reduction allocates a FRESH result (np.ptp is a
            // subtract, np.nanmin/nanmax are reductions), so `computed` never aliases `materialized`.
            using var materialized = EvaluateCore(program.ChildElementwiseProgram, inputs, null);

            // A FLAT reduce (axis == null) calls the engine reductions WITHOUT keepdims — they return a
            // 0-d scalar, which KeepdimsFlat below reshapes to (1,)*childNdim when requested (plan P2 M5);
            // the AXIS path passes keepdims straight through, since the engine reduction already keeps
            // the reduced axis as size 1. (np.argmax/argmin's flat form is a scalar `long` regardless,
            // wrapped into a 0-d int64 array.)
            bool axisKd = reduce.Axis is int && reduce.Keepdims;
            NDArray computed = reduce.Kind switch
            {
                NDExprReduceKind.Ptp    => np.ptp(materialized, reduce.Axis, keepdims: axisKd),
                NDExprReduceKind.NanMin => np.nanmin(materialized, reduce.Axis, axisKd),
                NDExprReduceKind.NanMax => np.nanmax(materialized, reduce.Axis, axisKd),
                NDExprReduceKind.ArgMax => reduce.Axis is int axMax
                    ? np.argmax(materialized, axMax, reduce.Keepdims)
                    : NDArray.Scalar(np.argmax(materialized)),
                NDExprReduceKind.ArgMin => reduce.Axis is int axMin
                    ? np.argmin(materialized, axMin, reduce.Keepdims)
                    : NDArray.Scalar(np.argmin(materialized)),
                _ => throw new NotSupportedException(
                    $"EvaluateDelegatingReduce reached with non-delegating kind {reduce.Kind} — dispatch bug."),
            };

            // Flat keepdims (plan P2 M5): keep every axis as size 1 → (1,)*childNdim over the
            // materialized child's rank. A 0-d child stays 0-d; keepdims=false is a no-op.
            if (reduce.Axis is null)
                computed = KeepdimsFlat(computed, materialized.ndim, reduce.Keepdims);

            if (@out is null)
                return computed;

            // A flat reduction's out must be 0-d (or (1,)*childNdim under keepdims); NumPy rejects a
            // wrong-rank out with this message (the fold path raises the identical text — see
            // EvaluateReduce). The axis case's out shape is checked by np.copyto below.
            if (reduce.Axis is null)
                ValidateFlatReduceOut(@out, materialized.ndim, reduce.Keepdims, ReduceUfuncName(reduce.Kind));

            // copyto applies the (already-validated) same_kind cast and shape-checks the axis case,
            // matching the axis fold path's `np.copyto(@out, reduced)` tail.
            np.copyto(@out, computed);
            return @out;
        }

        // =====================================================================================
        // Plan P2 M4c (summation kinds) — NanMean / Var / Std, host-computed over the materialized
        // child with the SAME NumPy-exact pairwise sum the M1/M2 diverts use.
        //
        // These CANNOT delegate to the engine's np.nanmean/np.var (their flat sum is a drifting
        // multi-accumulator SIMD fold, not NumPy's pairwise), and they CANNOT be tree rewrites
        // (nanmean's divisor is a per-slab non-NaN count; var/std are two-pass with a ddof divisor).
        // So the host materializes the child once and reproduces NumPy's own algorithm op for op:
        //   NanMean = pairwise_sum(NaN→0) / count_of_non_NaN
        //   Var     = pairwise_sum((x − mean)²) / max(N − ddof, 0),  mean = pairwise_sum(x) / N
        //   Std     = sqrt(Var)
        // A FLAT reduction folds the child's buffer in MEMORY order (bit-exact for a C- or
        // F-contiguous child alike, since np.add.reduce over a contiguous buffer iterates memory
        // order); the AXIS path rides the engine's own bit-exact axis add.reduce over the
        // C-contiguous child (the strict-F multi-D axis corner is the same one M2 leaves excused).
        // Float16 and Decimal are rejected — a bit-exact float16 var needs a float16 pairwise kernel
        // (the "Half not diverted" gap M1/M2 share) and Decimal has neither a NumPy analog nor a
        // pairwise kernel here. Bit-exact vs NumPy 2.4.2 for the other 13 dtypes; NO excuse.
        // =====================================================================================
        private unsafe NDArray EvaluateStatReduce(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options = default)
        {
            var reduce = program.Reduce;
            var resultType = program.ResultType;

            // Validate the out cast up front (same order as every reduce path).
            if (@out is not null)
                ValidateOutCast(resultType, @out.typecode, "evaluate", options.Casting);

            using var materialized = EvaluateCore(program.ChildElementwiseProgram, inputs, null);
            NPTypeCode ct = materialized.typecode;

            // Half needs a float16 pairwise sum kernel that does not exist (NumPy accumulates a float16
            // variance IN float16 — a widen-to-float32 computation is NOT bit-exact, ~30% of inputs
            // differ); Decimal has no NumPy analog and no pairwise kernel. Reject both with a directed
            // message rather than silently returning a divergent value.
            if (ct == NPTypeCode.Half || ct == NPTypeCode.Decimal)
                throw new NotSupportedException(
                    $"np.evaluate {ReduceUfuncName(reduce.Kind)} does not support the {ct} dtype yet " +
                    $"(a bit-exact float16/decimal reduction needs a pairwise sum kernel that dtype lacks; " +
                    $"use np.{ReduceUfuncName(reduce.Kind)} directly). Plan P2 M-Half/M-Decimal.");

            NDArray computed = reduce.Kind == NDExprReduceKind.NanMean
                ? ExactNanMean(materialized, reduce.Axis, reduce.Keepdims, resultType)
                : ExactVarStd(materialized, reduce.Axis, reduce.Keepdims, reduce.Ddof,
                              sqrtResult: reduce.Kind == NDExprReduceKind.Std, resultType);

            // ExactNanMean/ExactVarStd apply keepdims on the AXIS path (via FinishReduced); a FLAT reduce
            // returns a 0-d scalar, so keep every axis as size 1 here → (1,)*childNdim (plan P2 M5).
            if (reduce.Axis is null)
                computed = KeepdimsFlat(computed, materialized.ndim, reduce.Keepdims);

            if (@out is null)
                return computed;

            // A flat reduction's out must be 0-d (or (1,)*childNdim under keepdims); NumPy rejects a
            // wrong-rank out with this message (same shape as the fold / delegating paths). The axis
            // out shape is checked by np.copyto below.
            if (reduce.Axis is null)
                ValidateFlatReduceOut(@out, materialized.ndim, reduce.Keepdims, ReduceUfuncName(reduce.Kind));

            np.copyto(@out, computed);
            return @out;
        }

        // =====================================================================================
        // Plan P2 M4c-average — weighted np.average, host-computed over the TWO materialized children.
        //
        // np.average(v, w) = Σ(v·w) / Σ(w), with BOTH sums and the product forced to ONE result dtype
        // (NumPy's `np.multiply(v, w, dtype=rt).sum(dtype=rt) / w.sum(dtype=rt)`), so an integer average
        // is a float64 computation with no integer wrap. The host materializes the two children once
        // (the M1/M2 route — fresh, contiguous arrays at their natural dtypes), casts each to the result
        // dtype, forms the product and reduces the product and the weights with the SAME NumPy-exact
        // pairwise sum (`ExactSumArray`) the M1/M2 diverts use — NOT the drifting multi-accumulator
        // engine sum the library `np.average` itself uses, which is exactly why the fused Average is
        // BIT-EXACT with NumPy where the library `np.average` is only allclose at large N. Then divides.
        //
        // Zero total weight (an empty input included) raises DivideByZeroException with NumPy's verbatim
        // "Weights sum to zero, can't be normalized" — checked BEFORE the divide, exactly as
        // np.average does; a NaN weight makes the denominator NaN (NaN != 0), so it is not a zero and
        // the result is NaN, matching NumPy. Half / Decimal result dtypes are rejected (no pairwise sum
        // kernel — the gap M1/M2/M4c-summation share). Bit-exact vs NumPy 2.4.2 for Single/Double/
        // Complex; NO MisalignedRegistry excuse.
        // =====================================================================================
        private unsafe NDArray EvaluateWeightedAverage(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options = default)
        {
            var avg = program.Average;
            NPTypeCode rt = program.ResultType;   // Single / Double / Complex (Half / Decimal rejected below)

            // Validate the out cast up front (same order as every reduce path: the cast rule is checked
            // before any compute), so an illegal out dtype fails with evaluate's message.
            if (@out is not null)
                ValidateOutCast(rt, @out.typecode, "evaluate", options.Casting);

            // A bit-exact float16 / decimal average needs a pairwise sum kernel that dtype lacks (the
            // "Half not diverted" gap M1/M2/M4c-summation share). Reject with a directed message rather
            // than silently returning a divergent value — use np.average directly for those.
            if (rt == NPTypeCode.Half || rt == NPTypeCode.Decimal)
                throw new NotSupportedException(
                    $"np.evaluate average does not support the {rt} result dtype yet " +
                    "(a bit-exact float16/decimal reduction needs a pairwise sum kernel that dtype lacks; " +
                    "use np.average directly). Plan P2 M-Half/M-Decimal.");

            // Materialize both children once (fresh, contiguous, at their natural dtypes). `using`
            // releases them after the product / weight-sum have read them.
            using var vMat = EvaluateCore(program.AvgValuesProgram, inputs, null);
            using var wMat = EvaluateCore(program.AvgWeightsProgram, inputs, null);

            // Cast each child to the result dtype as a FRESH, C-contiguous array (so disposal never
            // aliases vMat / wMat, and the pairwise sum reads a straight memory walk = NumPy's C-order).
            NDArray vRt = ToResultContig(vMat, rt);
            NDArray wRt = ToResultContig(wMat, rt);

            // Product at the result dtype (the two children broadcast together, like any fused binary);
            // guard C-contiguity for ExactSumArray (a C-input multiply is C, but stay defensive).
            NDArray prod0 = vRt * wRt;
            NDArray prod = prod0.Shape.IsContiguous ? prod0 : prod0.copy();
            if (!ReferenceEquals(prod, prod0)) prod0.Dispose();
            vRt.Dispose();

            // Denominator weights at the result dtype, broadcast to the PRODUCT's shape so the total
            // weight counts each element as many times as the numerator does (a same-shape average — the
            // np.average contract — needs no broadcast; wForDen is then wRt itself).
            NDArray wForDen = SameDims(wRt.Shape, prod.Shape)
                ? wRt
                : np.broadcast_to(wRt, prod.Shape).copy();

            int? nax = avg.Axis is int rawAxis ? NormalizeAxis(rawAxis, prod.ndim) : (int?)null;
            int avgChildNdim = prod.ndim;                      // the (broadcast) rank — the flat keepdims result is (1,)*avgChildNdim

            NDArray num = ExactSumArray(prod, nax, rt);        // Σ(v·w) at rt (pairwise / axis add.reduce)
            NDArray den = ExactSumArray(wForDen, nax, rt);     // Σ(w)   at rt

            prod.Dispose();
            if (!ReferenceEquals(wForDen, wRt)) wForDen.Dispose();
            wRt.Dispose();

            // NumPy raises BEFORE dividing when any (per-slab) total weight is exactly zero.
            if (AverageDenominatorHasZero(den, rt))
            {
                num.Dispose();
                den.Dispose();
                throw new DivideByZeroException("Weights sum to zero, can't be normalized");
            }

            NDArray computed = num / den;                      // rt/rt → rt (complex divide is bit-exact)
            num.Dispose();
            den.Dispose();

            if (avg.Keepdims && nax is int kdAxis)
            {
                NDArray kd = np.expand_dims(computed, kdAxis);
                computed = kd;
            }
            else if (avg.Axis is null)
            {
                // Flat keepdims (plan P2 M5): keep every axis as size 1 → (1,)*avgChildNdim over the
                // (broadcast) product's rank. keepdims=false is a no-op; a 0-d child stays 0-d.
                computed = KeepdimsFlat(computed, avgChildNdim, avg.Keepdims);
            }

            if (@out is null)
                return computed;

            // A flat average's out must be 0-d (or (1,)*avgChildNdim under keepdims); NumPy rejects a
            // wrong-rank out here (same message shape as the fold / delegating / stat paths). The axis
            // out shape is checked by np.copyto below.
            if (avg.Axis is null)
                ValidateFlatReduceOut(@out, avgChildNdim, avg.Keepdims, "average");

            np.copyto(@out, computed);
            return @out;
        }

        /// <summary>Cast a materialized child to <paramref name="rt"/> as a FRESH, C-contiguous array (a copy even when already that dtype, so disposal never aliases the source).</summary>
        /// <param name="m">The materialized child (any dtype, any contiguous layout).</param>
        /// <param name="rt">The result dtype (Single / Double / Complex here).</param>
        /// <returns>A fresh C-contiguous array of dtype <paramref name="rt"/>.</returns>
        private static NDArray ToResultContig(NDArray m, NPTypeCode rt)
            => m.typecode == rt ? m.copy() : m.astype(DType.From(rt));

        /// <summary>Do two shapes have identical dimensions (so no broadcast is needed for the denominator sum)?</summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>True when the dimension arrays match element for element.</returns>
        private static bool SameDims(Shape a, Shape b)
        {
            var da = a.dimensions; var db = b.dimensions;
            if (da.Length != db.Length) return false;
            for (int i = 0; i < da.Length; i++)
                if (da[i] != db[i]) return false;
            return true;
        }

        /// <summary>
        /// Does any element of the (fresh, C-contiguous) denominator <paramref name="den"/> equal exactly
        /// zero — NumPy's <c>scl == 0.0</c> zero-weight check (per slab for the axis form)? A complex
        /// denominator is zero iff BOTH components are zero; a NaN denominator is NOT zero (so a NaN weight
        /// yields NaN, not a raise), matching NumPy.
        /// </summary>
        /// <param name="den">The denominator (total weight) array, dtype <paramref name="rt"/>.</param>
        /// <param name="rt">The denominator dtype (Single / Double / Complex).</param>
        /// <returns>True when any total weight is exactly zero.</returns>
        /// <exception cref="NotSupportedException"><paramref name="rt"/> is not a served average dtype (a typing bug).</exception>
        private static unsafe bool AverageDenominatorHasZero(NDArray den, NPTypeCode rt)
        {
            long n = den.size;
            byte* p = (byte*)den.Address + (long)den.Shape.offset * rt.SizeOf();
            switch (rt)
            {
                case NPTypeCode.Single: { float* f = (float*)p; for (long i = 0; i < n; i++) if (f[i] == 0f) return true; return false; }
                case NPTypeCode.Double: { double* f = (double*)p; for (long i = 0; i < n; i++) if (f[i] == 0d) return true; return false; }
                case NPTypeCode.Complex: { var z = (System.Numerics.Complex*)p; for (long i = 0; i < n; i++) if (z[i].Real == 0d && z[i].Imaginary == 0d) return true; return false; }
                default: throw new NotSupportedException($"average denominator dtype {rt} — typing bug.");
            }
        }

        /// <summary>
        /// <c>np.nanmean</c> over the freshly materialized child. Result dtype <paramref name="resultT"/>
        /// (int/bool/char→float64, float32/float64 preserved, complex128 preserved). A float/complex
        /// child skips NaN per element (an all-NaN slice → NaN); an integer/bool/char child carries no
        /// NaN and is exactly the plain mean.
        /// </summary>
        private unsafe NDArray ExactNanMean(NDArray m, int? axis, bool keepdims, NPTypeCode resultT)
        {
            NPTypeCode ct = m.typecode;
            bool floaty = ct == NPTypeCode.Single || ct == NPTypeCode.Double || ct == NPTypeCode.Complex;
            // Compute dtype: complex stays complex, float32 stays float32, everything else → float64.
            NPTypeCode accT = ct == NPTypeCode.Single ? NPTypeCode.Single
                            : ct == NPTypeCode.Complex ? NPTypeCode.Complex
                            : NPTypeCode.Double;

            if (axis is null)
            {
                long n = m.size;
                // A memory-order accT source buffer (m's buffer directly for a float/complex child;
                // a widened float64 buffer for an integer/bool/char child).
                using var srcHolder = StatFlatSource(m, accT, out byte* src);

                if (!floaty)
                {
                    // No NaN — nanmean == mean: pairwise sum / n at accT (float64).
                    return FlatMeanScalar(src, n, accT, n, resultT);
                }

                // Build the NaN-cleaned buffer and count the finite elements (memory order).
                var cleaned = new NDArray(accT, new Shape(n), false);
                byte* cp = (byte*)cleaned.Address;
                long count = BuildCleanedFlat(src, cp, n, accT);

                byte* slot = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)slot = 0; *(ulong*)(slot + 8) = 0;
                PairwiseSumInto(cp, n, accT, slot);
                cleaned.Dispose();

                DivideSlotByCount(slot, accT, count);              // total / count (Smith form for complex)
                return ScalarFromSlot(slot, accT, resultT);
            }

            // ---- axis ----
            int nd = m.ndim;
            int ax = NormalizeAxis(axis.Value, nd);
            long axisSize = m.Shape.dimensions[ax];
            NDArray w = EnsureContiguous(m, accT);                 // C-contiguous accT child

            NDArray reduced;
            if (!floaty)
            {
                reduced = ExactSumArray(w, ax, accT);              // float64 axis sum (no NaN)
                DivideByConst(reduced, accT, axisSize);            // / axisSize
            }
            else
            {
                // Per-slab: nansum(cleaned) / count(present). `cleaned` is NaN→0, `present` is 1.0/0.0.
                var cleaned = new NDArray(accT, w.Shape.Clone(), false);
                var present = new NDArray(NPTypeCode.Double, w.Shape.Clone(), false);
                BuildCleanedAndPresent(
                    (byte*)w.Address + (long)w.Shape.offset * accT.SizeOf(),
                    (byte*)cleaned.Address, (double*)present.Address, w.size, accT);

                NDArray totals = ExactSumArray(cleaned, ax, accT);
                NDArray counts = ExactSumArray(present, ax, NPTypeCode.Double); // exact integer counts as float64
                cleaned.Dispose(); present.Dispose();

                DivideByCounts(totals, accT, counts);              // per-element total/count
                counts.Dispose();
                reduced = totals;
            }

            return FinishReduced(reduced, ax, keepdims, accT, resultT);
        }

        /// <summary>
        /// <c>np.var</c> (or <c>np.std</c> when <paramref name="sqrtResult"/>) over the materialized
        /// child: NumPy's two-pass <c>_var</c> — <c>mean = Σx / N</c>, then <c>Σ(x − mean)² / max(N −
        /// ddof, 0)</c> — reproduced over the exact pairwise sum. A complex128 child yields a REAL
        /// float64 result (<c>|x − mean|²</c>). NaN PROPAGATES (every element participates), and an
        /// empty / degenerate divisor yields NaN / ±inf, matching NumPy.
        /// </summary>
        private unsafe NDArray ExactVarStd(NDArray m, int? axis, bool keepdims, int ddof, bool sqrtResult, NPTypeCode resultT)
        {
            NPTypeCode ct = m.typecode;
            bool complexInput = ct == NPTypeCode.Complex;
            // Mean/accumulate dtype and deviation (|x−mean|²) dtype.
            NPTypeCode accT = ct == NPTypeCode.Single ? NPTypeCode.Single
                            : complexInput ? NPTypeCode.Complex
                            : NPTypeCode.Double;
            NPTypeCode devT = complexInput ? NPTypeCode.Double : accT; // complex variance is real

            if (axis is null)
            {
                long n = m.size;
                using var srcHolder = StatFlatSource(m, accT, out byte* src);

                // mean = Σx / n (Smith divide for complex).
                byte* meanSlot = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)meanSlot = 0; *(ulong*)(meanSlot + 8) = 0;
                PairwiseSumInto(src, n, accT, meanSlot);
                DivideSlotByCount(meanSlot, accT, n);

                // d2[i] = (x−mean)² (real) or |x−mean|² (complex), in memory order → a devT buffer.
                var d2flat = new NDArray(devT, new Shape(n), false);
                BuildSquaredDeviationsFlat(src, (byte*)d2flat.Address, meanSlot, n, accT, devT);

                byte* ssum = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)ssum = 0; *(ulong*)(ssum + 8) = 0;
                PairwiseSumInto((byte*)d2flat.Address, n, devT, ssum);
                d2flat.Dispose();

                DivideSlotByCount(ssum, devT, System.Math.Max(n - ddof, 0));
                if (sqrtResult) SqrtSlot(ssum, devT);
                return ScalarFromSlot(ssum, devT, resultT);
            }

            // ---- axis ----
            int nd = m.ndim;
            int ax = NormalizeAxis(axis.Value, nd);
            long axisSize = m.Shape.dimensions[ax];
            NDArray w = EnsureContiguous(m, accT);                 // C-contiguous accT child

            // mean along axis, kept as size-1 for the broadcast subtract.
            NDArray meanReduced = ExactSumArray(w, ax, accT);
            DivideByConst(meanReduced, accT, axisSize);
            NDArray meanKD = np.expand_dims(meanReduced, ax);
            NDArray dOrig = w - meanKD;                            // broadcast subtract (elementwise-exact)
            NDArray d = dOrig.Shape.IsContiguous ? dOrig : dOrig.copy(); // the sq-dev loop reads memory order
            if (!ReferenceEquals(d, dOrig)) dOrig.Dispose();
            meanReduced.Dispose();                                 // meanKD is a view of it; consumed by the subtract

            // d2[i] = (x−mean)² / |x−mean|², memory order → devT buffer of d's shape.
            var d2 = new NDArray(devT, d.Shape.Clone(), false);
            BuildSquaredDeviationsContig(
                (byte*)d.Address + (long)d.Shape.offset * accT.SizeOf(),
                (byte*)d2.Address, d.size, accT, devT);
            d.Dispose();

            NDArray sumsq = ExactSumArray(d2, ax, devT);
            d2.Dispose();
            DivideByConst(sumsq, devT, System.Math.Max(axisSize - ddof, 0));
            if (sqrtResult) SqrtInPlace(sumsq, devT);

            return FinishReduced(sumsq, ax, keepdims, devT, resultT);
        }

        // ---- shared stat helpers -----------------------------------------------------------------

        /// <summary>
        /// Exact NumPy <c>add.reduce</c> of a fresh C-contiguous array (dtype <paramref name="accT"/> ∈
        /// {Single, Double, Complex}) — flat (<paramref name="axis"/> null → 0-d) or along an axis
        /// (→ reduced), reusing the pairwise (flat) / axis (M2) primitives.
        /// </summary>
        private unsafe NDArray ExactSumArray(NDArray a, int? axis, NPTypeCode accT)
        {
            if (axis is null)
            {
                byte* slot = stackalloc byte[NDExprParamPlan.SlotBytes];
                *(ulong*)slot = 0; *(ulong*)(slot + 8) = 0;
                byte* src = (byte*)a.Address + (long)a.Shape.offset * accT.SizeOf();
                PairwiseSumInto(src, a.size, accT, slot);
                var r = new NDArray(accT, Shape.NewScalar(), false);
                NDIterCasting.ConvertValue(slot, (byte*)r.Address + (long)r.Shape.offset * accT.SizeOf(), accT, accT);
                return r;
            }

            int nd = a.ndim;
            int ax = NormalizeAxis(axis.Value, nd);
            var reducedDims = new long[nd - 1];
            for (int d = 0, rd = 0; d < nd; d++) if (d != ax) reducedDims[rd++] = a.Shape.dimensions[d];
            Shape reducedShape = reducedDims.Length > 0 ? new Shape(reducedDims) : Shape.NewScalar();
            return ExactAxisSum(a, ax, accT, reducedShape);
        }

        /// <summary>
        /// A memory-order source buffer of dtype <paramref name="accT"/> over the FLAT child <paramref
        /// name="m"/>: for a float/complex child (already accT) it points straight at the child's buffer
        /// with no allocation (returns null holder); for an integer/bool/char child it allocates a
        /// float64 buffer and widens the child's memory-order elements into it (via per-element
        /// <see cref="NDIterCasting.ConvertValue"/>, so every source dtype is handled uniformly).
        /// </summary>
        private static unsafe NDArray StatFlatSource(NDArray m, NPTypeCode accT, out byte* src)
        {
            NPTypeCode ct = m.typecode;
            if (ct == accT)
            {
                src = (byte*)m.Address + (long)m.Shape.offset * accT.SizeOf();
                return null; // no buffer to dispose
            }

            long n = m.size;
            var buf = new NDArray(accT, new Shape(n), false);
            byte* srcB = (byte*)m.Address + (long)m.Shape.offset * ct.SizeOf();
            byte* dstB = (byte*)buf.Address;
            int ss = ct.SizeOf(), ds = accT.SizeOf();
            for (long i = 0; i < n; i++)
                NDIterCasting.ConvertValue(srcB + i * ss, dstB + i * ds, ct, accT);
            src = dstB;
            return buf;
        }

        /// <summary>Ensure a C-contiguous child of dtype <paramref name="accT"/> for the axis path.</summary>
        private static NDArray EnsureContiguous(NDArray m, NPTypeCode accT)
            => m.typecode == accT
                ? (m.Shape.IsContiguous ? m : m.copy())
                : m.astype(DType.From(accT)); // astype returns a fresh C-contiguous array

        /// <summary>Flat mean of a real/complex source into a fresh 0-d scalar (accT) cast to <paramref name="resultT"/>.</summary>
        private static unsafe NDArray FlatMeanScalar(byte* src, long n, NPTypeCode accT, long count, NPTypeCode resultT)
        {
            byte* slot = stackalloc byte[NDExprParamPlan.SlotBytes];
            *(ulong*)slot = 0; *(ulong*)(slot + 8) = 0;
            PairwiseSumInto(src, n, accT, slot);
            DivideSlotByCount(slot, accT, count);
            return ScalarFromSlot(slot, accT, resultT);
        }

        /// <summary>Wrap a 16-byte accumulator slot of dtype <paramref name="fromT"/> into a fresh 0-d array cast to <paramref name="toT"/>.</summary>
        private static unsafe NDArray ScalarFromSlot(byte* slot, NPTypeCode fromT, NPTypeCode toT)
        {
            var r = new NDArray(toT, Shape.NewScalar(), false);
            NDIterCasting.ConvertValue(slot, (byte*)r.Address + (long)r.Shape.offset * toT.SizeOf(), fromT, toT);
            return r;
        }

        /// <summary>Divide a 16-byte accumulator slot (Single/Double/Complex) in place by an integer count.</summary>
        private static unsafe void DivideSlotByCount(byte* slot, NPTypeCode tc, long count)
        {
            switch (tc)
            {
                case NPTypeCode.Single: *(float*)slot /= (float)count; break;
                case NPTypeCode.Double: *(double*)slot /= (double)count; break;
                case NPTypeCode.Complex:
                    *(System.Numerics.Complex*)slot = ComplexDivideByCountLikeNumPy(*(System.Numerics.Complex*)slot, count);
                    break;
                default: throw new NotSupportedException($"stat divide for {tc} — typing bug.");
            }
        }

        /// <summary>Square root of a 16-byte accumulator slot (Single/Double) in place.</summary>
        private static unsafe void SqrtSlot(byte* slot, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Single: *(float*)slot = MathF.Sqrt(*(float*)slot); break;
                case NPTypeCode.Double: *(double*)slot = System.Math.Sqrt(*(double*)slot); break;
                default: throw new NotSupportedException($"stat sqrt for {tc} — typing bug.");
            }
        }

        /// <summary>Divide every element of a fresh reduced array (Single/Double/Complex) by a constant integer count.</summary>
        private static unsafe void DivideByConst(NDArray arr, NPTypeCode tc, long count)
        {
            long nEl = arr.size;
            byte* p = (byte*)arr.Address + (long)arr.Shape.offset * tc.SizeOf();
            switch (tc)
            {
                case NPTypeCode.Single: { float* f = (float*)p; float d = (float)count; for (long i = 0; i < nEl; i++) f[i] /= d; break; }
                case NPTypeCode.Double: { double* f = (double*)p; double d = (double)count; for (long i = 0; i < nEl; i++) f[i] /= d; break; }
                case NPTypeCode.Complex: { var f = (System.Numerics.Complex*)p; for (long i = 0; i < nEl; i++) f[i] = ComplexDivideByCountLikeNumPy(f[i], count); break; }
                default: throw new NotSupportedException($"stat divide-by-const for {tc} — typing bug.");
            }
        }

        /// <summary>Divide totals[i] (Single/Double/Complex) by the per-element float64 count[i] (nanmean's per-slab divisor).</summary>
        private static unsafe void DivideByCounts(NDArray totals, NPTypeCode tc, NDArray counts)
        {
            long nEl = totals.size;
            byte* p = (byte*)totals.Address + (long)totals.Shape.offset * tc.SizeOf();
            double* c = (double*)((byte*)counts.Address + (long)counts.Shape.offset * sizeof(double));
            switch (tc)
            {
                case NPTypeCode.Single: { float* f = (float*)p; for (long i = 0; i < nEl; i++) f[i] /= (float)c[i]; break; }
                case NPTypeCode.Double: { double* f = (double*)p; for (long i = 0; i < nEl; i++) f[i] /= c[i]; break; }
                case NPTypeCode.Complex: { var f = (System.Numerics.Complex*)p; for (long i = 0; i < nEl; i++) f[i] = ComplexDivideByCountLikeNumPy(f[i], (long)c[i]); break; }
                default: throw new NotSupportedException($"stat divide-by-counts for {tc} — typing bug.");
            }
        }

        /// <summary>Elementwise square root of a fresh reduced array (Single/Double) in place.</summary>
        private static unsafe void SqrtInPlace(NDArray arr, NPTypeCode tc)
        {
            long nEl = arr.size;
            byte* p = (byte*)arr.Address + (long)arr.Shape.offset * tc.SizeOf();
            switch (tc)
            {
                case NPTypeCode.Single: { float* f = (float*)p; for (long i = 0; i < nEl; i++) f[i] = MathF.Sqrt(f[i]); break; }
                case NPTypeCode.Double: { double* f = (double*)p; for (long i = 0; i < nEl; i++) f[i] = System.Math.Sqrt(f[i]); break; }
                default: throw new NotSupportedException($"stat sqrt for {tc} — typing bug.");
            }
        }

        /// <summary>Apply keepdims (re-insert the size-1 axis) and cast a fresh reduced array to <paramref name="resultT"/>.</summary>
        private static NDArray FinishReduced(NDArray reduced, int axis, bool keepdims, NPTypeCode fromT, NPTypeCode resultT)
        {
            NDArray r = keepdims ? np.expand_dims(reduced, axis) : reduced;
            return fromT == resultT ? r : r.astype(DType.From(resultT));
        }

        /// <summary>
        /// Build the flat NaN-cleaned buffer (NaN→0) from a memory-order source and return the count of
        /// finite elements. A complex element is NaN if EITHER component is NaN (np.isnan semantics).
        /// </summary>
        private static unsafe long BuildCleanedFlat(byte* src, byte* cleaned, long n, NPTypeCode accT)
        {
            long count = 0;
            switch (accT)
            {
                case NPTypeCode.Single:
                {
                    float* s = (float*)src; float* d = (float*)cleaned;
                    for (long i = 0; i < n; i++) { float v = s[i]; bool nan = float.IsNaN(v); d[i] = nan ? 0f : v; if (!nan) count++; }
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* s = (double*)src; double* d = (double*)cleaned;
                    for (long i = 0; i < n; i++) { double v = s[i]; bool nan = double.IsNaN(v); d[i] = nan ? 0d : v; if (!nan) count++; }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var s = (System.Numerics.Complex*)src; var d = (System.Numerics.Complex*)cleaned;
                    for (long i = 0; i < n; i++) { var v = s[i]; bool nan = double.IsNaN(v.Real) || double.IsNaN(v.Imaginary); d[i] = nan ? System.Numerics.Complex.Zero : v; if (!nan) count++; }
                    break;
                }
                default: throw new NotSupportedException($"nanmean clean for {accT} — typing bug.");
            }
            return count;
        }

        /// <summary>Build the C-contiguous NaN-cleaned buffer (NaN→0) and the parallel float64 present buffer (1.0 finite / 0.0 NaN) for the axis nanmean.</summary>
        private static unsafe void BuildCleanedAndPresent(byte* src, byte* cleaned, double* present, long n, NPTypeCode accT)
        {
            switch (accT)
            {
                case NPTypeCode.Single:
                {
                    float* s = (float*)src; float* d = (float*)cleaned;
                    for (long i = 0; i < n; i++) { float v = s[i]; bool nan = float.IsNaN(v); d[i] = nan ? 0f : v; present[i] = nan ? 0d : 1d; }
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* s = (double*)src; double* d = (double*)cleaned;
                    for (long i = 0; i < n; i++) { double v = s[i]; bool nan = double.IsNaN(v); d[i] = nan ? 0d : v; present[i] = nan ? 0d : 1d; }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var s = (System.Numerics.Complex*)src; var d = (System.Numerics.Complex*)cleaned;
                    for (long i = 0; i < n; i++) { var v = s[i]; bool nan = double.IsNaN(v.Real) || double.IsNaN(v.Imaginary); d[i] = nan ? System.Numerics.Complex.Zero : v; present[i] = nan ? 0d : 1d; }
                    break;
                }
                default: throw new NotSupportedException($"nanmean clean for {accT} — typing bug.");
            }
        }

        /// <summary>Flat squared-deviation buffer from a memory-order source and a scalar mean slot (real: (x−m)²; complex: |x−m|² into a float64 buffer).</summary>
        private static unsafe void BuildSquaredDeviationsFlat(byte* src, byte* d2, byte* meanSlot, long n, NPTypeCode accT, NPTypeCode devT)
        {
            switch (accT)
            {
                case NPTypeCode.Single:
                {
                    float* s = (float*)src; float* o = (float*)d2; float m = *(float*)meanSlot;
                    for (long i = 0; i < n; i++) { float t = s[i] - m; o[i] = t * t; }
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* s = (double*)src; double* o = (double*)d2; double m = *(double*)meanSlot;
                    for (long i = 0; i < n; i++) { double t = s[i] - m; o[i] = t * t; }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var s = (System.Numerics.Complex*)src; double* o = (double*)d2; var m = *(System.Numerics.Complex*)meanSlot;
                    // |x−m|² as re²+im² (two separate multiplies + add — NOT an FMA; verified to match
                    // np.var(complex) bit-for-bit over 20,000 adversarial inputs, where an FMA does not).
                    for (long i = 0; i < n; i++) { double tr = s[i].Real - m.Real; double ti = s[i].Imaginary - m.Imaginary; o[i] = tr * tr + ti * ti; }
                    break;
                }
                default: throw new NotSupportedException($"var deviations for {accT} — typing bug.");
            }
        }

        /// <summary>Squared-deviation buffer from an already-differenced C-contiguous buffer (real: x²; complex: |x|² into float64).</summary>
        private static unsafe void BuildSquaredDeviationsContig(byte* d, byte* d2, long n, NPTypeCode accT, NPTypeCode devT)
        {
            switch (accT)
            {
                case NPTypeCode.Single:
                {
                    float* s = (float*)d; float* o = (float*)d2;
                    for (long i = 0; i < n; i++) { float t = s[i]; o[i] = t * t; }
                    break;
                }
                case NPTypeCode.Double:
                {
                    double* s = (double*)d; double* o = (double*)d2;
                    for (long i = 0; i < n; i++) { double t = s[i]; o[i] = t * t; }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var s = (System.Numerics.Complex*)d; double* o = (double*)d2;
                    for (long i = 0; i < n; i++) { double tr = s[i].Real; double ti = s[i].Imaginary; o[i] = tr * tr + ti * ti; }
                    break;
                }
                default: throw new NotSupportedException($"var deviations for {accT} — typing bug.");
            }
        }

        // Axis-aware fused reduction: one pass over the inputs, accumulating into a per-output
        // operand under a REDUCE iterator. evaluate(Sum(a*b, axis:k)) never materializes a*b.
        private unsafe NDArray EvaluateAxisReduce(NDExprProgram program, NDArray[] inputs, NDArray @out, int axis, in NDEvaluateOptions options = default)
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
                ValidateOutCast(resultType, @out.typecode, "evaluate", options.Casting);

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
