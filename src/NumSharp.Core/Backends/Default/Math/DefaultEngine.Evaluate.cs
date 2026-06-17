using System;
using System.Collections.Concurrent;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// np.evaluate — fused expression evaluation (roadmap Wave 6.1).
    ///
    /// Compiles an <see cref="NpyExpr"/> tree to ONE NpyIter pass: every
    /// elementwise node runs inside a single inner-loop kernel, so a chained
    /// expression like (a-b)/(a+b) reads each operand once and writes the
    /// result once — no intermediate arrays, no extra memory traffic. The POC
    /// measured 2.8–5.4× over NumPy's unfused ufunc chains for exactly these
    /// shapes.
    ///
    /// Semantics:
    ///   • dtypes follow NumPy 2.x result_type per NODE (NpyExpr.Typing.cs):
    ///     the fused result is bit-compatible with the unfused NumPy sequence,
    ///     including int32 wraparound before promotion.
    ///   • operands broadcast together exactly like a ufunc call; repeated
    ///     NDArray references deduplicate to one iterator operand.
    ///   • out= joins the broadcast (never stretched), needs a same_kind cast
    ///     from the resolved dtype, and may alias an input — COPY_IF_OVERLAP
    ///     gives ufunc-grade overlap safety.
    ///   • a root ReduceNode (NpyExpr.Sum/Prod/Min/Max/Mean) runs a one-pass
    ///     accumulating kernel over the inputs only: sum(a*b) never
    ///     materializes a*b.
    ///
    /// Known divergences from NumPy (documented, by design):
    ///   • float32/float16 sum/prod/mean accumulate in float64 (NumPy uses
    ///     pairwise accumulation at the input dtype) — fused results can
    ///     differ in the last ulps, usually MORE accurate.
    ///   • power with a negative integer exponent ARRAY computes a wrapped
    ///     value where NumPy raises per element (literal exponents are
    ///     checked at compile time with NumPy's exact error).
    /// </summary>
    public partial class DefaultEngine
    {
        // Per-arity flag arrays for the evaluate iterator configs — allocated
        // once per input count (Wave 2.2 discipline: no per-call new[] for
        // call-invariant data).
        private static readonly ConcurrentDictionary<int, NpyIterPerOpFlags[]> s_evalElementwiseFlags = new();
        private static readonly ConcurrentDictionary<int, NpyIterPerOpFlags[]> s_evalReduceFlags = new();

        private static NpyIterPerOpFlags[] EvalElementwiseFlags(int nIn)
            => s_evalElementwiseFlags.GetOrAdd(nIn, static n =>
            {
                var flags = new NpyIterPerOpFlags[n + 1];
                for (int i = 0; i < n; i++)
                    flags[i] = NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                flags[n] = NpyIterPerOpFlags.WRITEONLY
                           | NpyIterPerOpFlags.NO_BROADCAST
                           | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                return flags;
            });

        private static NpyIterPerOpFlags[] EvalReduceFlags(int nIn)
            => s_evalReduceFlags.GetOrAdd(nIn, static n =>
            {
                var flags = new NpyIterPerOpFlags[n];
                for (int i = 0; i < n; i++)
                    flags[i] = NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
                return flags;
            });

        /// <summary>
        /// Evaluate a tree whose array leaves are embedded NDArrays
        /// (<see cref="NpyExpr.Arr"/> / implicit conversion).
        /// </summary>
        public override unsafe NDArray Evaluate(NpyExpr expr, NDArray @out = null)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));

            var bind = new NpyExprBindContext();
            var bound = expr.BindArrays(bind);
            if (bind.Operands.Count == 0)
                throw new ArgumentException(
                    "expression references no arrays — embed NDArrays in the tree " +
                    "(NpyExpr.Arr / implicit conversion) or use the (expr, operands) overload with NpyExpr.Input(i) leaves.",
                    nameof(expr));

            return EvaluateCore(bound, bind.Operands.ToArray(), @out);
        }

        /// <summary>
        /// Evaluate a tree built over positional <see cref="NpyExpr.Input"/>
        /// leaves against an explicit operand list.
        /// </summary>
        public override unsafe NDArray Evaluate(NpyExpr expr, NDArray[] operands, NDArray @out = null)
        {
            if (expr is null) throw new ArgumentNullException(nameof(expr));
            if (operands is null) throw new ArgumentNullException(nameof(operands));
            if (operands.Length == 0)
                throw new ArgumentException("operands must not be empty.", nameof(operands));
            foreach (var op in operands)
                if (op is null)
                    throw new ArgumentNullException(nameof(operands), "no operand may be null.");

            var bind = new NpyExprBindContext();
            var bound = expr.BindArrays(bind);
            if (bind.Operands.Count != 0)
                throw new ArgumentException(
                    "expression mixes embedded array leaves with a positional operand list — use one binding style.",
                    nameof(expr));

            return EvaluateCore(bound, operands, @out);
        }

        private unsafe NDArray EvaluateCore(NpyExpr bound, NDArray[] ops, NDArray @out)
        {
            if (bound is ReduceNode reduce)
            {
                if (reduce.Child.ContainsReduce)
                    throw new NotSupportedException(
                        "nested reductions are not supported — a reduction must be the root of the expression.");
                return EvaluateReduce(reduce, ops, @out);
            }

            if (bound.ContainsReduce)
                throw new NotSupportedException(
                    "reduction nodes must be the root of the expression — " +
                    "elementwise use of a reduced value needs two np.evaluate calls.");

            var inputTypes = new NPTypeCode[ops.Length];
            for (int i = 0; i < ops.Length; i++)
                inputTypes[i] = ops[i].typecode;

            var kernel = bound.CompileNumPy(inputTypes, out var resolvedType);

            if (@out is not null)
                ValidateOutCast(resolvedType, @out.typecode, "evaluate");

            // Broadcast all inputs together (pairwise fold raises the standard
            // broadcast error), then let out join per the ufunc rules.
            Shape inputShape = ops[0].Shape.Clean();
            for (int i = 1; i < ops.Length; i++)
                (inputShape, _) = Broadcast(inputShape, ops[i].Shape);

            var iterShape = ResolveUfuncIterationShape(inputShape.Clean(), ops, @out, null);

            var target = @out ?? new NDArray(resolvedType, iterShape.Clean(), false);
            if (target.size == 0)
                return target;

            bool outNeedsCast = target.typecode != resolvedType;
            var globalFlags = NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode[] opDtypes = null;
            if (outNeedsCast)
            {
                // The kernel writes the resolved dtype into the out operand's
                // buffer; the windowed flush casts (same_kind was validated, so
                // the iterator runs UNSAFE exactly like NumPy's ufunc layer).
                globalFlags |= NpyIterGlobalFlags.BUFFERED
                             | NpyIterGlobalFlags.GROWINNER
                             | NpyIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                opDtypes = new NPTypeCode[ops.Length + 1];
                Array.Copy(inputTypes, opDtypes, ops.Length);
                opDtypes[ops.Length] = resolvedType;
            }

            var operands = new NDArray[ops.Length + 1];
            Array.Copy(ops, operands, ops.Length);
            operands[ops.Length] = target;

            using var iter = NpyIterRef.MultiNew(
                operands.Length, operands,
                globalFlags, NPY_ORDER.NPY_CORDER, casting,
                EvalElementwiseFlags(ops.Length),
                opDtypes);

            iter.ForEach(kernel);
            return target;
        }

        // =====================================================================
        // Fused reductions
        // =====================================================================

        private static string ReduceUfuncName(NpyExprReduceKind kind) => kind switch
        {
            NpyExprReduceKind.Sum => "add",
            NpyExprReduceKind.Prod => "multiply",
            NpyExprReduceKind.Min => "minimum",
            NpyExprReduceKind.Max => "maximum",
            _ => "mean",
        };

        private unsafe NDArray EvaluateReduce(ReduceNode reduce, NDArray[] ops, NDArray @out)
        {
            if (reduce.Axis is int ax)
                return EvaluateAxisReduce(reduce, ops, @out, ax);

            var inputTypes = new NPTypeCode[ops.Length];
            for (int i = 0; i < ops.Length; i++)
                inputTypes[i] = ops[i].typecode;

            var kernel = reduce.CompileReduceKernel(inputTypes, out var accType, out var resultType);

            if (@out is not null)
            {
                ValidateOutCast(resultType, @out.typecode, "evaluate");
                if (@out.ndim != 0)
                    throw new ArgumentException(
                        $"output parameter for reduction operation {ReduceUfuncName(reduce.Kind)} " +
                        $"has the wrong number of dimensions: Found {@out.ndim} but expected 0");
            }

            Shape inputShape = ops[0].Shape.Clean();
            for (int i = 1; i < ops.Length; i++)
                (inputShape, _) = Broadcast(inputShape, ops[i].Shape);

            long n = inputShape.size;

            // 16 bytes covers the widest accumulators (decimal / Complex).
            byte* slot = stackalloc byte[16];
            *(ulong*)slot = 0;
            *(ulong*)(slot + 8) = 0;

            if (n == 0)
            {
                switch (reduce.Kind)
                {
                    case NpyExprReduceKind.Min:
                    case NpyExprReduceKind.Max:
                        throw new ArgumentException(
                            $"zero-size array to reduction operation {ReduceUfuncName(reduce.Kind)} which has no identity");
                    case NpyExprReduceKind.Prod:
                        WriteOne(slot, accType);
                        break;
                    case NpyExprReduceKind.Mean:
                        WriteMeanOfEmpty(slot, accType); // NaN, like np.mean([]) (NumPy warns)
                        break;
                        // Sum: identity 0 already in the slot.
                }
            }
            else
            {
                switch (reduce.Kind)
                {
                    case NpyExprReduceKind.Prod:
                        WriteOne(slot, accType);
                        break;
                    case NpyExprReduceKind.Min:
                    case NpyExprReduceKind.Max:
                        WriteMinMaxIdentity(slot, accType, isMin: reduce.Kind == NpyExprReduceKind.Min);
                        break;
                        // Sum / Mean: identity 0 already in the slot.
                }

                using var iter = NpyIterRef.MultiNew(
                    ops.Length, ops,
                    NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    EvalReduceFlags(ops.Length),
                    null);

                iter.ForEach(kernel, slot);

                if (reduce.Kind == NpyExprReduceKind.Mean)
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
                            *(System.Numerics.Complex*)slot /= n;
                            break;
                        default:
                            throw new NotSupportedException($"mean accumulator {accType} — typing bug.");
                    }
                }
            }

            var result = @out ?? new NDArray(resultType, Shape.NewScalar(), false);
            byte* dst = (byte*)result.Address
                        + (long)result.Shape.offset * result.typecode.SizeOf();
            NpyIterCasting.ConvertValue(slot, dst, accType, result.typecode);
            return result;
        }

        // Axis-aware fused reduction: one pass over the inputs, accumulating into a per-output
        // operand under a REDUCE iterator. evaluate(Sum(a*b, axis:k)) never materializes a*b.
        private unsafe NDArray EvaluateAxisReduce(ReduceNode reduce, NDArray[] ops, NDArray @out, int axis)
        {
            var inputTypes = new NPTypeCode[ops.Length];
            for (int i = 0; i < ops.Length; i++)
                inputTypes[i] = ops[i].typecode;

            // Broadcast all inputs to one shape (same rule as the elementwise path).
            Shape inputShape = ops[0].Shape.Clean();
            for (int i = 1; i < ops.Length; i++)
                (inputShape, _) = Broadcast(inputShape, ops[i].Shape);
            int ndim = inputShape.NDim;
            axis = NormalizeAxis(axis, ndim);

            var kernel = reduce.CompileAxisReduceKernel(inputTypes, out var accType, out var resultType);

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
                    NpyExprReduceKind.Prod => ReductionOp.Prod,
                    NpyExprReduceKind.Min => ReductionOp.Min,
                    NpyExprReduceKind.Max => ReductionOp.Max,
                    _ => ReductionOp.Sum, // Sum / Mean
                };
                ILKernelGenerator.SeedReduceIdentity(outAcc, seedOp);

                if (axisSize != 0)
                {
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

                    var opFlags = new NpyIterPerOpFlags[ops.Length + 1];
                    for (int i = 0; i < ops.Length; i++) opFlags[i] = NpyIterPerOpFlags.READONLY;
                    opFlags[ops.Length] = NpyIterPerOpFlags.READWRITE;

                    using var iter = NpyIterRef.AdvancedNew(
                        operands.Length, operands,
                        NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.EXTERNAL_LOOP,
                        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
                        opFlags, null, ndim, opAxes);
                    iter.ForEach(kernel);
                }

                if (reduce.Kind == NpyExprReduceKind.Mean)
                    ILKernelGenerator.MeanDivideByCount(outAcc, axisSize);
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
                default:
                    throw new NotSupportedException($"min/max accumulator {accType} — typing bug.");
            }
        }
    }
}
