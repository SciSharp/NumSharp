using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NpyExpr.Evaluate.cs — np.evaluate surface of the expression DSL (Wave 6.1)
// =============================================================================
//
// Adds the pieces that turn the Tier-3C compiler into a user-facing fused
// evaluator:
//
//   • ArrayNode — an NDArray embedded directly as a leaf, so trees read
//     naturally: (NpyExpr)a * b + 2. np.evaluate REBINDS array leaves to
//     positional InputNodes, deduplicating repeated references — the same
//     NDArray instance appearing twice becomes ONE iterator operand
//     ((a-b)/(a+b) iterates 3 streams, not 5).
//   • implicit conversions NDArray→NpyExpr and numeric→Const, so a single
//     cast at the head of an expression lights up the whole operator set.
//   • ReduceNode — root-only fused reductions (sum/prod/min/max/mean of an
//     arbitrary elementwise tree) compiled to a one-pass accumulating
//     kernel: sum(a*b) reads a and b once and never materializes a*b.
//
// Binding is a pure rewrite: nodes are immutable, so BindArrays returns the
// same instance when no array leaf lives below a node, or a rebuilt node
// otherwise. The bound tree is what typing + emission consume.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>Reduction kinds supported by <see cref="ReduceNode"/>.</summary>
    public enum NpyExprReduceKind : byte
    {
        Sum,
        Prod,
        Min,
        Max,
        Mean,
    }

    /// <summary>
    /// Operand collection for binding array leaves: deduplicates by reference
    /// so a repeated NDArray maps to one iterator operand.
    /// </summary>
    internal sealed class NpyExprBindContext
    {
        public readonly List<NDArray> Operands = new();

        public int IndexOf(NDArray array)
        {
            for (int i = 0; i < Operands.Count; i++)
            {
                if (ReferenceEquals(Operands[i], array))
                    return i;
            }

            Operands.Add(array);
            return Operands.Count - 1;
        }
    }

    public abstract partial class NpyExpr
    {
        // ===================================================================
        // Array leaves + literal sugar
        // ===================================================================

        /// <summary>
        /// Embed an NDArray directly as an expression leaf. np.evaluate binds
        /// every distinct array (by reference) to one iterator operand.
        /// </summary>
        public static NpyExpr Arr(NDArray array) => new ArrayNode(array);

        public static implicit operator NpyExpr(NDArray array) => new ArrayNode(array);
        public static implicit operator NpyExpr(double value) => Const(value);
        public static implicit operator NpyExpr(float value) => Const(value);
        public static implicit operator NpyExpr(int value) => Const(value);
        public static implicit operator NpyExpr(long value) => Const(value);

        // Mixed NpyExpr/NDArray operators. Exact-match overloads are required:
        // through implicit conversions alone, `expr * ndarray` is ambiguous
        // between NpyExpr.op_*(NpyExpr, NpyExpr) and NDArray's own
        // object-accepting operator overloads.
        public static NpyExpr operator +(NpyExpr a, NDArray b) => Add(a, Arr(b));
        public static NpyExpr operator +(NDArray a, NpyExpr b) => Add(Arr(a), b);
        public static NpyExpr operator -(NpyExpr a, NDArray b) => Subtract(a, Arr(b));
        public static NpyExpr operator -(NDArray a, NpyExpr b) => Subtract(Arr(a), b);
        public static NpyExpr operator *(NpyExpr a, NDArray b) => Multiply(a, Arr(b));
        public static NpyExpr operator *(NDArray a, NpyExpr b) => Multiply(Arr(a), b);
        public static NpyExpr operator /(NpyExpr a, NDArray b) => Divide(a, Arr(b));
        public static NpyExpr operator /(NDArray a, NpyExpr b) => Divide(Arr(a), b);
        public static NpyExpr operator %(NpyExpr a, NDArray b) => Mod(a, Arr(b));
        public static NpyExpr operator %(NDArray a, NpyExpr b) => Mod(Arr(a), b);
        public static NpyExpr operator &(NpyExpr a, NDArray b) => BitwiseAnd(a, Arr(b));
        public static NpyExpr operator &(NDArray a, NpyExpr b) => BitwiseAnd(Arr(a), b);
        public static NpyExpr operator |(NpyExpr a, NDArray b) => BitwiseOr(a, Arr(b));
        public static NpyExpr operator |(NDArray a, NpyExpr b) => BitwiseOr(Arr(a), b);
        public static NpyExpr operator ^(NpyExpr a, NDArray b) => BitwiseXor(a, Arr(b));
        public static NpyExpr operator ^(NDArray a, NpyExpr b) => BitwiseXor(Arr(a), b);

        // Scalar operators. Also required as exact matches: without them a
        // literal binds to the (NpyExpr, NDArray) overload through NDArray's
        // implicit numeric conversions (NDArray is the better conversion
        // target because NDArray→NpyExpr exists), silently turning a WEAK
        // NEP50 literal into a strong scalar array — f4+2.5 would promote to
        // f8 instead of staying f4.
        public static NpyExpr operator +(NpyExpr a, double b) => Add(a, Const(b));
        public static NpyExpr operator +(double a, NpyExpr b) => Add(Const(a), b);
        public static NpyExpr operator +(NpyExpr a, long b) => Add(a, Const(b));
        public static NpyExpr operator +(long a, NpyExpr b) => Add(Const(a), b);
        public static NpyExpr operator +(NpyExpr a, int b) => Add(a, Const(b));
        public static NpyExpr operator +(int a, NpyExpr b) => Add(Const(a), b);
        public static NpyExpr operator -(NpyExpr a, double b) => Subtract(a, Const(b));
        public static NpyExpr operator -(double a, NpyExpr b) => Subtract(Const(a), b);
        public static NpyExpr operator -(NpyExpr a, long b) => Subtract(a, Const(b));
        public static NpyExpr operator -(long a, NpyExpr b) => Subtract(Const(a), b);
        public static NpyExpr operator -(NpyExpr a, int b) => Subtract(a, Const(b));
        public static NpyExpr operator -(int a, NpyExpr b) => Subtract(Const(a), b);
        public static NpyExpr operator *(NpyExpr a, double b) => Multiply(a, Const(b));
        public static NpyExpr operator *(double a, NpyExpr b) => Multiply(Const(a), b);
        public static NpyExpr operator *(NpyExpr a, long b) => Multiply(a, Const(b));
        public static NpyExpr operator *(long a, NpyExpr b) => Multiply(Const(a), b);
        public static NpyExpr operator *(NpyExpr a, int b) => Multiply(a, Const(b));
        public static NpyExpr operator *(int a, NpyExpr b) => Multiply(Const(a), b);
        public static NpyExpr operator /(NpyExpr a, double b) => Divide(a, Const(b));
        public static NpyExpr operator /(double a, NpyExpr b) => Divide(Const(a), b);
        public static NpyExpr operator /(NpyExpr a, long b) => Divide(a, Const(b));
        public static NpyExpr operator /(long a, NpyExpr b) => Divide(Const(a), b);
        public static NpyExpr operator /(NpyExpr a, int b) => Divide(a, Const(b));
        public static NpyExpr operator /(int a, NpyExpr b) => Divide(Const(a), b);
        public static NpyExpr operator %(NpyExpr a, double b) => Mod(a, Const(b));
        public static NpyExpr operator %(double a, NpyExpr b) => Mod(Const(a), b);
        public static NpyExpr operator %(NpyExpr a, long b) => Mod(a, Const(b));
        public static NpyExpr operator %(long a, NpyExpr b) => Mod(Const(a), b);
        public static NpyExpr operator %(NpyExpr a, int b) => Mod(a, Const(b));
        public static NpyExpr operator %(int a, NpyExpr b) => Mod(Const(a), b);
        public static NpyExpr operator &(NpyExpr a, long b) => BitwiseAnd(a, Const(b));
        public static NpyExpr operator &(long a, NpyExpr b) => BitwiseAnd(Const(a), b);
        public static NpyExpr operator &(NpyExpr a, int b) => BitwiseAnd(a, Const(b));
        public static NpyExpr operator &(int a, NpyExpr b) => BitwiseAnd(Const(a), b);
        public static NpyExpr operator |(NpyExpr a, long b) => BitwiseOr(a, Const(b));
        public static NpyExpr operator |(long a, NpyExpr b) => BitwiseOr(Const(a), b);
        public static NpyExpr operator |(NpyExpr a, int b) => BitwiseOr(a, Const(b));
        public static NpyExpr operator |(int a, NpyExpr b) => BitwiseOr(Const(a), b);
        public static NpyExpr operator ^(NpyExpr a, long b) => BitwiseXor(a, Const(b));
        public static NpyExpr operator ^(long a, NpyExpr b) => BitwiseXor(Const(a), b);
        public static NpyExpr operator ^(NpyExpr a, int b) => BitwiseXor(a, Const(b));
        public static NpyExpr operator ^(int a, NpyExpr b) => BitwiseXor(Const(a), b);

        // ===================================================================
        // Reduction factories (root-only — see ReduceNode)
        // ===================================================================

        /// <summary>One-pass fused sum of the expression (NumPy dtype rules: int→int64, uint→uint64, floats preserved).</summary>
        public static NpyExpr Sum(NpyExpr x) => new ReduceNode(NpyExprReduceKind.Sum, x);

        /// <summary>One-pass fused product of the expression.</summary>
        public static NpyExpr Prod(NpyExpr x) => new ReduceNode(NpyExprReduceKind.Prod, x);

        /// <summary>One-pass fused minimum of the expression (NaN-propagating, like np.min).</summary>
        public static NpyExpr Min(NpyExpr x) => new ReduceNode(NpyExprReduceKind.Min, x);

        /// <summary>One-pass fused maximum of the expression (NaN-propagating, like np.max).</summary>
        public static NpyExpr Max(NpyExpr x) => new ReduceNode(NpyExprReduceKind.Max, x);

        /// <summary>One-pass fused arithmetic mean of the expression (ints→float64, floats preserved).</summary>
        public static NpyExpr Mean(NpyExpr x) => new ReduceNode(NpyExprReduceKind.Mean, x);

        // ===================================================================
        // Binding
        // ===================================================================

        /// <summary>
        /// Rewrite array leaves into positional inputs, collecting the distinct
        /// arrays into <paramref name="ctx"/>. Returns the same instance when
        /// the subtree contains no array leaf.
        /// </summary>
        internal abstract NpyExpr BindArrays(NpyExprBindContext ctx);

        /// <summary>True if any node in the subtree is a <see cref="ReduceNode"/>.</summary>
        internal abstract bool ContainsReduce { get; }
    }

    public sealed partial class InputNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx) => this;
        internal override bool ContainsReduce => false;
    }

    public sealed partial class ConstNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx) => this;
        internal override bool ContainsReduce => false;
    }

    public sealed partial class BinaryNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new BinaryNode(_op, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
    }

    public sealed partial class UnaryNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new UnaryNode(_op, c);
        }

        internal override bool ContainsReduce => _child.ContainsReduce;
    }

    public sealed partial class ComparisonNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new ComparisonNode(_op, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
    }

    public sealed partial class MinMaxNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new MinMaxNode(_isMin, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
    }

    public sealed partial class WhereNode
    {
        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var c = _cond.BindArrays(ctx);
            var a = _a.BindArrays(ctx);
            var b = _b.BindArrays(ctx);
            return ReferenceEquals(c, _cond) && ReferenceEquals(a, _a) && ReferenceEquals(b, _b)
                ? this
                : new WhereNode(c, a, b);
        }

        internal override bool ContainsReduce => _cond.ContainsReduce || _a.ContainsReduce || _b.ContainsReduce;
    }

    public sealed partial class CallNode
    {
        /// <summary>Clone with new args — reuses the registered slot/method, no re-registration.</summary>
        private CallNode(CallNode source, NpyExpr[] args)
        {
            _kind = source._kind;
            _method = source._method;
            _delegateType = source._delegateType;
            _slotId = source._slotId;
            _args = args;
            _paramCodes = source._paramCodes;
            _returnCode = source._returnCode;
            _signatureId = source._signatureId;
        }

        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            NpyExpr[]? rebound = null;
            for (int i = 0; i < _args.Length; i++)
            {
                var b = _args[i].BindArrays(ctx);
                if (!ReferenceEquals(b, _args[i]) && rebound is null)
                {
                    rebound = new NpyExpr[_args.Length];
                    Array.Copy(_args, rebound, i);
                }

                if (rebound is not null)
                    rebound[i] = b;
            }

            return rebound is null ? this : new CallNode(this, rebound);
        }

        internal override bool ContainsReduce
        {
            get
            {
                foreach (var a in _args)
                    if (a.ContainsReduce)
                        return true;
                return false;
            }
        }
    }

    // =========================================================================
    // Node: ArrayNode — an NDArray leaf, replaced by Input(i) during binding.
    // Never reaches typing or emission: np.evaluate always binds first.
    // =========================================================================

    public sealed partial class ArrayNode : NpyExpr
    {
        private readonly NDArray _array;

        public ArrayNode(NDArray array)
            => _array = array ?? throw new ArgumentNullException(nameof(array));

        internal NDArray Array => _array;

        public override bool SupportsSimd => true;

        public override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before compilation — evaluate the tree via np.evaluate, " +
                "or rewrite array leaves to NpyExpr.Input(i) and pass the arrays to the iterator.");

        public override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before compilation — evaluate the tree via np.evaluate.");

        public override void AppendSignature(StringBuilder sb)
            => sb.Append("Arr[unbound]");

        internal override NpyExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NpyExpr, NPTypeCode> nodeTypes)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before typing — evaluate the tree via np.evaluate.");

        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
            => new InputNode(ctx.IndexOf(_array));

        internal override bool ContainsReduce => false;
    }

    // =========================================================================
    // Node: ReduceNode — root-only fused reduction over an elementwise tree.
    //
    // np.evaluate drives it as: iterate the INPUT operands only (no output
    // operand), run a raw accumulating inner loop that evaluates the child
    // tree per element and folds into a host-owned accumulator slot (aux).
    //
    // dtype rules (NumPy 2.4.2, probed):
    //   sum/prod: bool/int→int64, uint→uint64, floats preserved
    //   min/max:  input dtype preserved
    //   mean:     bool/int→float64, floats preserved
    //
    // Accumulation detail: float sums/products/means accumulate in float64
    // and cast back once at the end — tighter than NumPy's pairwise f32 loop,
    // so f32 results can differ from np.sum in the last ulps (documented).
    // Min/max accumulate at the exact result dtype (comparisons are exact)
    // and propagate NaN like np.min/np.max.
    // =========================================================================

    public sealed partial class ReduceNode : NpyExpr
    {
        private readonly NpyExprReduceKind _kind;
        private readonly NpyExpr _child;

        public ReduceNode(NpyExprReduceKind kind, NpyExpr child)
        {
            _kind = kind;
            _child = child ?? throw new ArgumentNullException(nameof(child));
        }

        internal NpyExprReduceKind Kind => _kind;
        internal NpyExpr Child => _child;

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
            => throw new InvalidOperationException(
                $"Reduction nodes are driven by np.evaluate as the tree root — " +
                $"{_kind} cannot be emitted as an elementwise value.");

        public override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
            => throw new InvalidOperationException(
                "Reduction nodes have no vector path — they are driven by np.evaluate.");

        public override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Reduce").Append(_kind).Append('(');
            _child.AppendSignature(sb);
            sb.Append(')');
        }

        internal override NpyExpr BindArrays(NpyExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new ReduceNode(_kind, c);
        }

        internal override bool ContainsReduce => true;

        internal override NpyExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NpyExpr, NPTypeCode> nodeTypes)
        {
            var ct = _child.InferType(inputTypes, nodeTypes);
            var childType = ResolveChild(_child, ct, ct.IsWeak ? ct.DefaultCode : ct.Code, nodeTypes);

            var result = ResolveReduceResultType(_kind, childType);
            nodeTypes[this] = result;
            return NpyExprTypeInfo.Strong(result);
        }

        internal static NPTypeCode ResolveReduceResultType(NpyExprReduceKind kind, NPTypeCode child)
        {
            switch (kind)
            {
                case NpyExprReduceKind.Sum:
                case NpyExprReduceKind.Prod:
                    return child switch
                    {
                        NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Int16 or
                        NPTypeCode.Int32 or NPTypeCode.Int64 => NPTypeCode.Int64,
                        NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.Char or
                        NPTypeCode.UInt32 or NPTypeCode.UInt64 => NPTypeCode.UInt64,
                        _ => child, // floats / Decimal / Complex preserved
                    };

                case NpyExprReduceKind.Min:
                case NpyExprReduceKind.Max:
                    if (child == NPTypeCode.Complex)
                        throw new NotSupportedException(
                            "min/max reduction over complex expressions is not supported by np.evaluate.");
                    return child;

                case NpyExprReduceKind.Mean:
                    return child switch
                    {
                        NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or
                        NPTypeCode.Decimal or NPTypeCode.Complex => child,
                        _ => NPTypeCode.Double,
                    };

                default:
                    throw new NotSupportedException($"Unknown reduce kind {kind}.");
            }
        }

        /// <summary>
        /// Accumulator dtype: result dtype, except f16/f32 sums/products/means
        /// widen to f64 (cast back at the end — see class doc).
        /// </summary>
        internal static NPTypeCode ResolveAccType(NpyExprReduceKind kind, NPTypeCode result)
        {
            if (kind == NpyExprReduceKind.Min || kind == NpyExprReduceKind.Max)
                return result;
            return result == NPTypeCode.Half || result == NPTypeCode.Single
                ? NPTypeCode.Double
                : result;
        }

        /// <summary>
        /// Compile the one-pass accumulating inner loop. The kernel evaluates
        /// the child tree per element (4-way unrolled with 4 accumulators for
        /// ILP) and folds into <c>*(Tacc*)aux</c>; the host initializes aux
        /// with the reduction identity and reads it back after iteration.
        /// </summary>
        internal NpyInnerLoopFunc CompileReduceKernel(
            NPTypeCode[] inputTypes,
            out NPTypeCode accType, out NPTypeCode resultType,
            string? cacheKey = null)
        {
            var resolved = ResolveNumPyTypes(inputTypes, out var nodeTypes);
            resultType = resolved;
            var acc = ResolveAccType(_kind, resolved);
            accType = acc;
            var exprType = nodeTypes[_child];
            int nIn = inputTypes.Length;
            var kind = _kind;
            var child = _child;

            string key = (cacheKey ?? DeriveCacheKey(inputTypes, resolved)) + "|npreduce";

            return DirectILKernelGenerator.CompileRawInnerLoop(il =>
            {
                // ---- locals -------------------------------------------------
                var ptrLocals = new LocalBuilder[nIn];
                var strideLocals = new LocalBuilder[nIn];
                var inputLocals = new LocalBuilder[nIn];
                for (int j = 0; j < nIn; j++)
                {
                    ptrLocals[j] = il.DeclareLocal(typeof(byte*));
                    strideLocals[j] = il.DeclareLocal(typeof(long));
                    inputLocals[j] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(inputTypes[j]));
                }

                var accClr = DirectILKernelGenerator.GetClrType(acc);
                var accLocals = new LocalBuilder[4];
                for (int l = 0; l < 4; l++)
                    accLocals[l] = il.DeclareLocal(accClr);

                var locI = il.DeclareLocal(typeof(long));
                var locN4 = il.DeclareLocal(typeof(long));

                var ctx = new NpyExprCompileContext(inputTypes, exprType, inputLocals, vectorMode: false, nodeTypes);

                // ---- prologue: unpack dataptrs / strides --------------------
                for (int j = 0; j < nIn; j++)
                {
                    il.Emit(OpCodes.Ldarg_0);
                    if (j > 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, j * sizeof(long));
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }

                    il.Emit(OpCodes.Ldind_I);
                    il.Emit(OpCodes.Stloc, ptrLocals[j]);

                    il.Emit(OpCodes.Ldarg_1);
                    if (j > 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, j * sizeof(long));
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }

                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stloc, strideLocals[j]);
                }

                // acc0 carries in from aux (running value across chunks);
                // acc1..acc3 start at the per-chunk identity: 0 for sum,
                // 1 for prod, and the CURRENT carry value for min/max
                // (idempotent under min/max, so no double counting).
                il.Emit(OpCodes.Ldarg_3);
                DirectILKernelGenerator.EmitLoadIndirect(il, acc);
                il.Emit(OpCodes.Stloc, accLocals[0]);
                for (int l = 1; l < 4; l++)
                {
                    switch (kind)
                    {
                        case NpyExprReduceKind.Sum:
                        case NpyExprReduceKind.Mean:
                            WhereNode.EmitPushZeroPublic(il, acc);
                            break;
                        case NpyExprReduceKind.Prod:
                            il.Emit(OpCodes.Ldc_I4_1);
                            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, acc);
                            break;
                        default: // Min / Max
                            il.Emit(OpCodes.Ldloc, accLocals[0]);
                            break;
                    }

                    il.Emit(OpCodes.Stloc, accLocals[l]);
                }

                // n4 = count & ~3
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, ~3L);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locN4);

                // i = 0
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locI);

                void EmitLane(int lane)
                {
                    // load inputs for this lane into the shared input locals
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        if (lane > 0)
                        {
                            il.Emit(OpCodes.Ldloc, strideLocals[j]);
                            il.Emit(OpCodes.Ldc_I4, lane);
                            il.Emit(OpCodes.Conv_I8);
                            il.Emit(OpCodes.Mul);
                            il.Emit(OpCodes.Conv_I);
                            il.Emit(OpCodes.Add);
                        }

                        DirectILKernelGenerator.EmitLoadIndirect(il, inputTypes[j]);
                        il.Emit(OpCodes.Stloc, inputLocals[j]);
                    }

                    // accLane = fold(accLane, (Tacc)expr)
                    il.Emit(OpCodes.Ldloc, accLocals[lane]);
                    child.EmitScalar(il, ctx);
                    DirectILKernelGenerator.EmitConvertTo(il, exprType, acc);
                    EmitFold(il, kind, acc);
                    il.Emit(OpCodes.Stloc, accLocals[lane]);
                }

                void EmitAdvance(int elements)
                {
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        il.Emit(OpCodes.Ldloc, strideLocals[j]);
                        if (elements != 1)
                        {
                            il.Emit(OpCodes.Ldc_I4, elements);
                            il.Emit(OpCodes.Conv_I8);
                            il.Emit(OpCodes.Mul);
                        }

                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                        il.Emit(OpCodes.Stloc, ptrLocals[j]);
                    }
                }

                // ---- unrolled loop ------------------------------------------
                var lblLoop4 = il.DefineLabel();
                var lblLoop4End = il.DefineLabel();
                il.MarkLabel(lblLoop4);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locN4);
                il.Emit(OpCodes.Bge, lblLoop4End);

                for (int lane = 0; lane < 4; lane++)
                    EmitLane(lane);
                EmitAdvance(4);

                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 4L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblLoop4);
                il.MarkLabel(lblLoop4End);

                // ---- scalar tail --------------------------------------------
                var lblTail = il.DefineLabel();
                var lblTailEnd = il.DefineLabel();
                il.MarkLabel(lblTail);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Bge, lblTailEnd);

                EmitLane(0);
                EmitAdvance(1);

                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblTail);
                il.MarkLabel(lblTailEnd);

                // ---- write back: *aux = fold(acc0, acc1, acc2, acc3) --------
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, accLocals[0]);
                for (int l = 1; l < 4; l++)
                {
                    il.Emit(OpCodes.Ldloc, accLocals[l]);
                    EmitFold(il, kind, acc);
                }

                DirectILKernelGenerator.EmitStoreIndirect(il, acc);
                il.Emit(OpCodes.Ret);
            }, key);
        }

        /// <summary>
        /// Fold [acc, value] → [acc'] at the accumulator dtype. Sum/Prod reuse
        /// the binary scalar emitters (full 15-dtype coverage); min/max use
        /// Math.Min/Max (NaN-propagating), with And/Or for bool and a
        /// double-roundtrip for Half (no Math.Min(Half) overload — the
        /// roundtrip is exact and keeps NaN propagation).
        /// </summary>
        private static void EmitFold(ILGenerator il, NpyExprReduceKind kind, NPTypeCode acc)
        {
            switch (kind)
            {
                case NpyExprReduceKind.Sum:
                case NpyExprReduceKind.Mean:
                    if (acc == NPTypeCode.Boolean)
                    {
                        il.Emit(OpCodes.Or);
                        return;
                    }

                    DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, acc);
                    return;

                case NpyExprReduceKind.Prod:
                    if (acc == NPTypeCode.Boolean)
                    {
                        il.Emit(OpCodes.And);
                        return;
                    }

                    DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, acc);
                    return;
            }

            bool isMin = kind == NpyExprReduceKind.Min;

            if (acc == NPTypeCode.Boolean)
            {
                il.Emit(isMin ? OpCodes.And : OpCodes.Or);
                return;
            }

            if (acc == NPTypeCode.Half)
            {
                // [accH, valH] → Math.Min/Max in double, back to Half (exact
                // roundtrip; keeps NaN propagation).
                var locVal = il.DeclareLocal(typeof(Half));
                il.Emit(OpCodes.Stloc, locVal);
                DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Half, NPTypeCode.Double);
                il.Emit(OpCodes.Ldloc, locVal);
                DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Half, NPTypeCode.Double);
                il.EmitCall(OpCodes.Call,
                    ScalarMethodCache.Get(typeof(Math), isMin ? "Min" : "Max", typeof(double), typeof(double)), null);
                DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, NPTypeCode.Half);
                return;
            }

            var clr = DirectILKernelGenerator.GetClrType(acc);
            System.Reflection.MethodInfo? method = null;
            try
            {
                method = ScalarMethodCache.Get(typeof(Math), isMin ? "Min" : "Max", clr, clr);
            }
            catch (MissingMethodException)
            {
            }

            if (method != null)
            {
                il.EmitCall(OpCodes.Call, method, null);
                return;
            }

            // Branchy fallback (Char): [acc, val] → select.
            var locV = il.DeclareLocal(clr);
            var locA = il.DeclareLocal(clr);
            il.Emit(OpCodes.Stloc, locV);
            il.Emit(OpCodes.Stloc, locA);
            var lblElse = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Ldloc, locV);
            DirectILKernelGenerator.EmitComparisonOperation(
                il, isMin ? ComparisonOp.LessEqual : ComparisonOp.GreaterEqual, acc);
            il.Emit(OpCodes.Brfalse, lblElse);
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Br, lblEnd);
            il.MarkLabel(lblElse);
            il.Emit(OpCodes.Ldloc, locV);
            il.MarkLabel(lblEnd);
        }
    }
}
