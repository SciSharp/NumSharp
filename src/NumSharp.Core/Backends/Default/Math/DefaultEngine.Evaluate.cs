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
            _ => "mean",
        };

        private unsafe NDArray EvaluateReduce(NDExprProgram program, NDArray[] inputs, NDArray @out)
        {
            var reduce = program.Reduce;
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
                        // Sum: identity 0 already in the slot.
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
                        // Sum / Mean: identity 0 already in the slot.
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

            var outAcc = new NDArray(accType, reducedShape, false);
            if (outAcc.size != 0)
            {
                // Seed the accumulator with the reduction identity (Mean accumulates a Sum).
                var seedOp = reduce.Kind switch
                {
                    NDExprReduceKind.Prod => ReductionOp.Prod,
                    NDExprReduceKind.Min => ReductionOp.Min,
                    NDExprReduceKind.Max => ReductionOp.Max,
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

                if (reduce.Kind == NDExprReduceKind.Mean)
                {
                    if (accType == NPTypeCode.Complex)
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
            }

            // Cast accumulator dtype → result dtype (no-op when equal — e.g. f16/f32 mean
            // accumulates in double then narrows here).
            NDArray reduced = accType == resultType ? outAcc : Cast(outAcc, resultType, copy: true);

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
