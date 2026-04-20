using System;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NpyExpr.cs — Expression DSL (Tier C of the custom-op API)
// =============================================================================
//
// A small algebraic AST over NpyIter operands. Compiles to an
// NpyInnerLoopFunc by emitting (scalarBody, vectorBody) pairs that
// ILKernelGenerator.CompileInnerLoop wraps in the standard 4× unroll shell.
//
// TYPE DISCIPLINE
// ---------------
// All intermediate computation happens in the output dtype. Input loads
// auto-promote to output dtype; constants are pushed as output dtype. This
// mirrors NumPy's casting-by-output behavior for simple ufunc composition
// and keeps the AST trivial to type-check.
//
// For fine-grained type control, use ExecuteElementWise directly (Tier B).
//
// SIMD
// ----
// The vector path is enabled iff every input type equals the output type
// AND every node's op supports SIMD. Otherwise the compiled kernel carries
// a scalar-only body; the factory's strided fallback handles all cases.
//
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Abstract expression node. Subclasses describe computations over
    /// NpyIter operands; Compile() produces an NpyInnerLoopFunc.
    /// </summary>
    public abstract class NpyExpr
    {
        // ----- Contract (internal API used by the compiler) -----

        /// <summary>
        /// Emit scalar code. On exit, the evaluation stack must have exactly
        /// one value of dtype <c>ctx.OutputType</c>.
        /// </summary>
        internal abstract void EmitScalar(ILGenerator il, NpyExprCompileContext ctx);

        /// <summary>
        /// Emit vector code. On exit, the evaluation stack must have exactly
        /// one <c>Vector{W}&lt;T&gt;</c> of element type <c>ctx.OutputType</c>.
        /// Called only when <see cref="SupportsSimd"/> is true and all input
        /// types equal the output type.
        /// </summary>
        internal abstract void EmitVector(ILGenerator il, NpyExprCompileContext ctx);

        /// <summary>
        /// True if this node and its entire sub-tree have a SIMD emit path.
        /// </summary>
        internal abstract bool SupportsSimd { get; }

        /// <summary>
        /// Stable structural signature. Used to derive a cache key when the
        /// user doesn't supply one.
        /// </summary>
        internal abstract void AppendSignature(StringBuilder sb);

        // ----- Compilation -----

        /// <summary>
        /// Compile the tree to an <see cref="NpyInnerLoopFunc"/>.
        /// </summary>
        internal NpyInnerLoopFunc Compile(
            NPTypeCode[] inputTypes, NPTypeCode outputType, string? cacheKey)
        {
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));

            string key = cacheKey ?? DeriveCacheKey(inputTypes, outputType);
            int nIn = inputTypes.Length;

            bool wantSimd = SupportsSimd && AllEqual(inputTypes, outputType);

            Action<ILGenerator> scalarBody = il =>
            {
                // Shell delivers N inputs on stack: stack[bottom]=in0, stack[top]=inN-1.
                // Stash each into a local (reverse order since we pop top first).
                var scalarLocals = new LocalBuilder[nIn];
                for (int i = nIn - 1; i >= 0; i--)
                {
                    scalarLocals[i] = il.DeclareLocal(ILKernelGenerator.GetClrType(inputTypes[i]));
                    il.Emit(OpCodes.Stloc, scalarLocals[i]);
                }
                var ctx = new NpyExprCompileContext(inputTypes, outputType, scalarLocals, vectorMode: false);
                EmitScalar(il, ctx);
                // Stack now: [result : outputType]  — factory stores it.
            };

            Action<ILGenerator>? vectorBody = null;
            if (wantSimd)
            {
                vectorBody = il =>
                {
                    var vectorLocals = new LocalBuilder[nIn];
                    var vecType = ILKernelGenerator.GetVectorType(ILKernelGenerator.GetClrType(inputTypes[0]));
                    for (int i = nIn - 1; i >= 0; i--)
                    {
                        vectorLocals[i] = il.DeclareLocal(vecType);
                        il.Emit(OpCodes.Stloc, vectorLocals[i]);
                    }
                    var ctx = new NpyExprCompileContext(inputTypes, outputType, vectorLocals, vectorMode: true);
                    EmitVector(il, ctx);
                };
            }

            var operandTypes = new NPTypeCode[nIn + 1];
            Array.Copy(inputTypes, operandTypes, nIn);
            operandTypes[nIn] = outputType;

            return ILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, key);
        }

        private string DeriveCacheKey(NPTypeCode[] inputTypes, NPTypeCode outputType)
        {
            var sb = new StringBuilder("NpyExpr:");
            AppendSignature(sb);
            sb.Append(":in=");
            for (int i = 0; i < inputTypes.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(inputTypes[i]);
            }
            sb.Append(":out=").Append(outputType);
            return sb.ToString();
        }

        private static bool AllEqual(NPTypeCode[] inputs, NPTypeCode output)
        {
            foreach (var t in inputs) if (t != output) return false;
            return true;
        }

        // ===================================================================
        // Leaf factories
        // ===================================================================

        /// <summary>Reference the i-th operand of the iterator (0-based input index).</summary>
        public static NpyExpr Input(int index) => new InputNode(index);

        /// <summary>Push a constant of the given .NET type. Value is converted to the output dtype when evaluated.</summary>
        public static NpyExpr Const(double value) => new ConstNode(value);
        public static NpyExpr Const(float value) => new ConstNode(value);
        public static NpyExpr Const(long value) => new ConstNode(value);
        public static NpyExpr Const(int value) => new ConstNode(value);

        // ===================================================================
        // Binary factories
        // ===================================================================

        // Arithmetic
        public static NpyExpr Add(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Add, a, b);
        public static NpyExpr Subtract(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Subtract, a, b);
        public static NpyExpr Multiply(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Multiply, a, b);
        public static NpyExpr Divide(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Divide, a, b);
        public static NpyExpr Mod(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Mod, a, b);
        public static NpyExpr Power(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.Power, a, b);
        public static NpyExpr FloorDivide(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.FloorDivide, a, b);
        public static NpyExpr ATan2(NpyExpr y, NpyExpr x) => new BinaryNode(BinaryOp.ATan2, y, x);

        // Bitwise
        public static NpyExpr BitwiseAnd(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.BitwiseAnd, a, b);
        public static NpyExpr BitwiseOr(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.BitwiseOr, a, b);
        public static NpyExpr BitwiseXor(NpyExpr a, NpyExpr b) => new BinaryNode(BinaryOp.BitwiseXor, a, b);

        // Scalar-branchy combinators compiled to IL
        public static NpyExpr Min(NpyExpr a, NpyExpr b) => new MinMaxNode(isMin: true, a, b);
        public static NpyExpr Max(NpyExpr a, NpyExpr b) => new MinMaxNode(isMin: false, a, b);
        public static NpyExpr Clamp(NpyExpr x, NpyExpr lo, NpyExpr hi) => Min(Max(x, lo), hi);
        public static NpyExpr Where(NpyExpr cond, NpyExpr a, NpyExpr b) => new WhereNode(cond, a, b);

        // ===================================================================
        // Unary factories
        // ===================================================================

        // Core arithmetic
        public static NpyExpr Sqrt(NpyExpr x) => new UnaryNode(UnaryOp.Sqrt, x);
        public static NpyExpr Abs(NpyExpr x) => new UnaryNode(UnaryOp.Abs, x);
        public static NpyExpr Negate(NpyExpr x) => new UnaryNode(UnaryOp.Negate, x);
        public static NpyExpr Square(NpyExpr x) => new UnaryNode(UnaryOp.Square, x);
        public static NpyExpr Reciprocal(NpyExpr x) => new UnaryNode(UnaryOp.Reciprocal, x);
        public static NpyExpr Sign(NpyExpr x) => new UnaryNode(UnaryOp.Sign, x);
        public static NpyExpr Cbrt(NpyExpr x) => new UnaryNode(UnaryOp.Cbrt, x);

        // Exp / Log family
        public static NpyExpr Exp(NpyExpr x) => new UnaryNode(UnaryOp.Exp, x);
        public static NpyExpr Exp2(NpyExpr x) => new UnaryNode(UnaryOp.Exp2, x);
        public static NpyExpr Expm1(NpyExpr x) => new UnaryNode(UnaryOp.Expm1, x);
        public static NpyExpr Log(NpyExpr x) => new UnaryNode(UnaryOp.Log, x);
        public static NpyExpr Log2(NpyExpr x) => new UnaryNode(UnaryOp.Log2, x);
        public static NpyExpr Log10(NpyExpr x) => new UnaryNode(UnaryOp.Log10, x);
        public static NpyExpr Log1p(NpyExpr x) => new UnaryNode(UnaryOp.Log1p, x);

        // Trigonometric
        public static NpyExpr Sin(NpyExpr x) => new UnaryNode(UnaryOp.Sin, x);
        public static NpyExpr Cos(NpyExpr x) => new UnaryNode(UnaryOp.Cos, x);
        public static NpyExpr Tan(NpyExpr x) => new UnaryNode(UnaryOp.Tan, x);
        public static NpyExpr Sinh(NpyExpr x) => new UnaryNode(UnaryOp.Sinh, x);
        public static NpyExpr Cosh(NpyExpr x) => new UnaryNode(UnaryOp.Cosh, x);
        public static NpyExpr Tanh(NpyExpr x) => new UnaryNode(UnaryOp.Tanh, x);
        public static NpyExpr ASin(NpyExpr x) => new UnaryNode(UnaryOp.ASin, x);
        public static NpyExpr ACos(NpyExpr x) => new UnaryNode(UnaryOp.ACos, x);
        public static NpyExpr ATan(NpyExpr x) => new UnaryNode(UnaryOp.ATan, x);
        public static NpyExpr Deg2Rad(NpyExpr x) => new UnaryNode(UnaryOp.Deg2Rad, x);
        public static NpyExpr Rad2Deg(NpyExpr x) => new UnaryNode(UnaryOp.Rad2Deg, x);

        // Rounding
        public static NpyExpr Floor(NpyExpr x) => new UnaryNode(UnaryOp.Floor, x);
        public static NpyExpr Ceil(NpyExpr x) => new UnaryNode(UnaryOp.Ceil, x);
        public static NpyExpr Round(NpyExpr x) => new UnaryNode(UnaryOp.Round, x);
        public static NpyExpr Truncate(NpyExpr x) => new UnaryNode(UnaryOp.Truncate, x);

        // Bitwise / logical
        public static NpyExpr BitwiseNot(NpyExpr x) => new UnaryNode(UnaryOp.BitwiseNot, x);
        public static NpyExpr LogicalNot(NpyExpr x) => new UnaryNode(UnaryOp.LogicalNot, x);

        // Predicates (returns numeric 0/1 at output dtype — NumPy-compatible)
        public static NpyExpr IsNaN(NpyExpr x) => new UnaryNode(UnaryOp.IsNan, x);
        public static NpyExpr IsFinite(NpyExpr x) => new UnaryNode(UnaryOp.IsFinite, x);
        public static NpyExpr IsInf(NpyExpr x) => new UnaryNode(UnaryOp.IsInf, x);

        // ===================================================================
        // Comparison factories (produce 0/1 at output dtype)
        // ===================================================================

        public static NpyExpr Equal(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.Equal, a, b);
        public static NpyExpr NotEqual(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.NotEqual, a, b);
        public static NpyExpr Less(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.Less, a, b);
        public static NpyExpr LessEqual(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.LessEqual, a, b);
        public static NpyExpr Greater(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.Greater, a, b);
        public static NpyExpr GreaterEqual(NpyExpr a, NpyExpr b) => new ComparisonNode(ComparisonOp.GreaterEqual, a, b);

        // ===================================================================
        // Call — invoke an arbitrary .NET delegate or MethodInfo per element.
        // ===================================================================
        //
        // Three entry points:
        //   (a) Typed Func<...> overloads — allow passing method groups
        //       (e.g. `Math.Sqrt`, `Math.Pow`) without an explicit cast.
        //       C# overload resolution picks these when the compiler can infer
        //       the delegate signature from the method group.
        //
        //   (b) `Call(Delegate func, params NpyExpr[] args)` — catch-all for
        //       any pre-constructed delegate. Method groups will NOT bind to
        //       this directly (the C# compiler needs a specific delegate
        //       target type). Cast or use a typed Func<...> overload.
        //
        //   (c) `Call(MethodInfo, ...)` and `Call(MethodInfo, object target, ...)`
        //       — bypass the delegate layer entirely. Static and instance methods
        //       respectively. Useful when reflecting over types at runtime.
        //
        // Implementation notes:
        //   * Static methods with no target are emitted as a direct `call`
        //     opcode to the underlying `MethodInfo` — no indirection.
        //   * Instance methods or delegates with captured state are stored in a
        //     process-wide slot dictionary (`DelegateSlots`). The emitted IL
        //     loads the delegate via an integer ID and invokes it through
        //     `Delegate.Invoke` (callvirt).
        //   * SIMD is always disabled for trees containing a CallNode.
        //   * Argument values are auto-converted from `ctx.OutputType` to each
        //     parameter's dtype; the return value is converted back to
        //     `ctx.OutputType` before leaving the node.

        /// <summary>Invoke a static method (no target).</summary>
        public static NpyExpr Call(System.Reflection.MethodInfo method, params NpyExpr[] args)
            => new CallNode(method, target: null, args);

        /// <summary>Invoke an instance method on a target object.</summary>
        public static NpyExpr Call(System.Reflection.MethodInfo method, object target, params NpyExpr[] args)
            => new CallNode(method, target, args);

        /// <summary>Invoke any delegate. Method-group arguments need a typed Func overload; use a cast or the typed overloads below.</summary>
        public static NpyExpr Call(Delegate func, params NpyExpr[] args)
            => new CallNode(func, args);

        // Typed Func<...> overloads — enable `NpyExpr.Call(Math.Sqrt, x)` without cast.
        public static NpyExpr Call<TR>(Func<TR> func)
            => new CallNode(func, Array.Empty<NpyExpr>());
        public static NpyExpr Call<T1, TR>(Func<T1, TR> func, NpyExpr a1)
            => new CallNode(func, new[] { a1 });
        public static NpyExpr Call<T1, T2, TR>(Func<T1, T2, TR> func, NpyExpr a1, NpyExpr a2)
            => new CallNode(func, new[] { a1, a2 });
        public static NpyExpr Call<T1, T2, T3, TR>(Func<T1, T2, T3, TR> func, NpyExpr a1, NpyExpr a2, NpyExpr a3)
            => new CallNode(func, new[] { a1, a2, a3 });
        public static NpyExpr Call<T1, T2, T3, T4, TR>(Func<T1, T2, T3, T4, TR> func, NpyExpr a1, NpyExpr a2, NpyExpr a3, NpyExpr a4)
            => new CallNode(func, new[] { a1, a2, a3, a4 });

        // ===================================================================
        // Operator overloads (syntactic sugar)
        // ===================================================================

        public static NpyExpr operator +(NpyExpr a, NpyExpr b) => Add(a, b);
        public static NpyExpr operator -(NpyExpr a, NpyExpr b) => Subtract(a, b);
        public static NpyExpr operator *(NpyExpr a, NpyExpr b) => Multiply(a, b);
        public static NpyExpr operator /(NpyExpr a, NpyExpr b) => Divide(a, b);
        public static NpyExpr operator %(NpyExpr a, NpyExpr b) => Mod(a, b);
        public static NpyExpr operator &(NpyExpr a, NpyExpr b) => BitwiseAnd(a, b);
        public static NpyExpr operator |(NpyExpr a, NpyExpr b) => BitwiseOr(a, b);
        public static NpyExpr operator ^(NpyExpr a, NpyExpr b) => BitwiseXor(a, b);
        public static NpyExpr operator -(NpyExpr a) => Negate(a);
        public static NpyExpr operator ~(NpyExpr a) => BitwiseNot(a);
        public static NpyExpr operator !(NpyExpr a) => LogicalNot(a);
    }

    // =========================================================================
    // Compile-time context shared with each node
    // =========================================================================

    internal sealed class NpyExprCompileContext
    {
        public NPTypeCode[] InputTypes { get; }
        public NPTypeCode OutputType { get; }
        public LocalBuilder[] InputLocals { get; }
        public bool VectorMode { get; }

        public NpyExprCompileContext(
            NPTypeCode[] inputTypes, NPTypeCode outputType,
            LocalBuilder[] inputLocals, bool vectorMode)
        {
            InputTypes = inputTypes;
            OutputType = outputType;
            InputLocals = inputLocals;
            VectorMode = vectorMode;
        }
    }

    // =========================================================================
    // Node: Input(i) — reference operand i
    // =========================================================================

    internal sealed class InputNode : NpyExpr
    {
        private readonly int _index;
        public InputNode(int index)
        {
            if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
            _index = index;
        }

        internal override bool SupportsSimd => true;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            if (_index >= ctx.InputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({_index}) out of range; compile provided {ctx.InputTypes.Length} inputs.");

            il.Emit(OpCodes.Ldloc, ctx.InputLocals[_index]);
            // Auto-convert if input type differs from output type.
            var inType = ctx.InputTypes[_index];
            if (inType != ctx.OutputType)
                ILKernelGenerator.EmitConvertTo(il, inType, ctx.OutputType);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            if (_index >= ctx.InputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({_index}) out of range; compile provided {ctx.InputTypes.Length} inputs.");

            // Vector mode is only used when all input types == output type
            // (enforced by Compile), so no conversion is needed here.
            il.Emit(OpCodes.Ldloc, ctx.InputLocals[_index]);
        }

        internal override void AppendSignature(StringBuilder sb)
            => sb.Append("In[").Append(_index).Append(']');
    }

    // =========================================================================
    // Node: Constant
    // =========================================================================

    internal sealed class ConstNode : NpyExpr
    {
        // Store as double — widest scalar; convert down to outputType on emit.
        // Also preserve an exact-int path for integer-typed outputs.
        private readonly double _valueFp;
        private readonly long _valueInt;
        private readonly bool _isIntegerLiteral;

        public ConstNode(double v) { _valueFp = v; _valueInt = 0; _isIntegerLiteral = false; }
        public ConstNode(float v) { _valueFp = v; _valueInt = 0; _isIntegerLiteral = false; }
        public ConstNode(long v) { _valueInt = v; _valueFp = v; _isIntegerLiteral = true; }
        public ConstNode(int v) { _valueInt = v; _valueFp = v; _isIntegerLiteral = true; }

        internal override bool SupportsSimd => true;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            EmitLoadTyped(il, ctx.OutputType);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            EmitLoadTyped(il, ctx.OutputType);
            ILKernelGenerator.EmitVectorCreate(il, ctx.OutputType);
        }

        private void EmitLoadTyped(ILGenerator il, NPTypeCode target)
        {
            switch (target)
            {
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, (float)_valueFp);
                    return;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, _valueFp);
                    return;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, _isIntegerLiteral ? _valueInt : (long)_valueFp);
                    return;
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Char:
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_I4, _isIntegerLiteral ? (int)_valueInt : (int)_valueFp);
                    return;
                default:
                    throw new NotSupportedException(
                        $"ConstNode cannot emit for output dtype {target}.");
            }
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Const[");
            if (_isIntegerLiteral) sb.Append(_valueInt); else sb.Append(_valueFp);
            sb.Append(']');
        }
    }

    // =========================================================================
    // Node: Binary op
    // =========================================================================

    internal sealed class BinaryNode : NpyExpr
    {
        private readonly BinaryOp _op;
        private readonly NpyExpr _left;
        private readonly NpyExpr _right;

        public BinaryNode(BinaryOp op, NpyExpr left, NpyExpr right)
        {
            _op = op;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        internal override bool SupportsSimd
            => _left.SupportsSimd && _right.SupportsSimd && IsSimdOp(_op);

        // Must match ILKernelGenerator.EmitVectorOperation's supported set.
        // Mod, Power, FloorDivide, ATan2 are scalar-only.
        private static bool IsSimdOp(BinaryOp op)
            => op == BinaryOp.Add || op == BinaryOp.Subtract ||
               op == BinaryOp.Multiply || op == BinaryOp.Divide ||
               op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr ||
               op == BinaryOp.BitwiseXor;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            _left.EmitScalar(il, ctx);
            _right.EmitScalar(il, ctx);
            ILKernelGenerator.EmitScalarOperation(il, _op, ctx.OutputType);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            _left.EmitVector(il, ctx);
            _right.EmitVector(il, ctx);
            ILKernelGenerator.EmitVectorOperation(il, _op, ctx.OutputType);
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append(_op).Append('(');
            _left.AppendSignature(sb);
            sb.Append(',');
            _right.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Unary op
    // =========================================================================

    internal sealed class UnaryNode : NpyExpr
    {
        private readonly UnaryOp _op;
        private readonly NpyExpr _child;

        public UnaryNode(UnaryOp op, NpyExpr child)
        {
            _op = op;
            _child = child ?? throw new ArgumentNullException(nameof(child));
        }

        internal override bool SupportsSimd
            => _child.SupportsSimd && IsSimdUnary(_op);

        // Must match ILKernelGenerator.EmitUnaryVectorOperation's supported set.
        // (See ILKernelGenerator.Unary.Vector.cs). Ops not listed here stay scalar-only.
        // Round and Truncate are intentionally excluded: Vector256.Round/Truncate only
        // exist in .NET 9+ but NumSharp's library targets net8 as well, and the emit
        // path fails there with "Could not find Round/Truncate for Vector256`1".
        private static bool IsSimdUnary(UnaryOp op)
            => op == UnaryOp.Negate || op == UnaryOp.Abs || op == UnaryOp.Sqrt ||
               op == UnaryOp.Floor || op == UnaryOp.Ceil ||
               op == UnaryOp.Square || op == UnaryOp.Reciprocal ||
               op == UnaryOp.Deg2Rad || op == UnaryOp.Rad2Deg || op == UnaryOp.BitwiseNot;

        // Predicates leave a bool (I4 0/1) on the stack — not outputType. The wrapper
        // below converts to outputType so the factory's Stind matches.
        private static bool IsPredicateResult(UnaryOp op)
            => op == UnaryOp.IsNan || op == UnaryOp.IsFinite || op == UnaryOp.IsInf;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            // LogicalNot needs a special path. ILKernelGenerator's emit uses Ldc_I4_0+Ceq
            // which is only correct when the input value fits in I4 (Int32 and narrower).
            // For Int64/Single/Double/Decimal the types mismatch on the stack. Rewrite
            // as (x == 0) using the comparison emit, which handles all types correctly.
            if (_op == UnaryOp.LogicalNot)
            {
                _child.EmitScalar(il, ctx);
                // push zero of outputType, compare Equal
                WhereNode.EmitPushZeroPublic(il, ctx.OutputType);
                ILKernelGenerator.EmitComparisonOperation(il, ComparisonOp.Equal, ctx.OutputType);
                ILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, ctx.OutputType);
                return;
            }

            _child.EmitScalar(il, ctx);
            ILKernelGenerator.EmitUnaryScalarOperation(il, _op, ctx.OutputType);
            if (IsPredicateResult(_op))
                ILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, ctx.OutputType);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            _child.EmitVector(il, ctx);
            ILKernelGenerator.EmitUnaryVectorOperation(il, _op, ctx.OutputType);
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append(_op).Append('(');
            _child.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Comparison op (produces numeric 0/1 at output dtype)
    //
    // Comparisons in NumPy return bool arrays, but NpyExpr's single-output-dtype
    // model collapses that to "0 or 1 at output dtype", which composes cleanly
    // with arithmetic (e.g. (x > 0) * x for ReLU). The I4 0/1 produced by
    // EmitComparisonOperation is converted to the output dtype after emission.
    //
    // Scalar-only — SIMD would require writing bool output and rerouting through
    // the Comparison kernel pipeline, which is beyond this tier.
    // =========================================================================

    internal sealed class ComparisonNode : NpyExpr
    {
        private readonly ComparisonOp _op;
        private readonly NpyExpr _left;
        private readonly NpyExpr _right;

        public ComparisonNode(ComparisonOp op, NpyExpr left, NpyExpr right)
        {
            _op = op;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        internal override bool SupportsSimd => false;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            _left.EmitScalar(il, ctx);
            _right.EmitScalar(il, ctx);
            // Both operands are already at ctx.OutputType (InputNode auto-converts).
            ILKernelGenerator.EmitComparisonOperation(il, _op, ctx.OutputType);
            // EmitComparisonOperation leaves an I4 (0 or 1) on the stack.
            // Convert to ctx.OutputType so the final Stind opcode matches.
            ILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, ctx.OutputType);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            throw new InvalidOperationException("ComparisonNode has no vector path.");
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Cmp").Append(_op).Append('(');
            _left.AppendSignature(sb);
            sb.Append(',');
            _right.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Min/Max — scalar-only branchy select
    //
    // Min(a, b) = a < b ? a : b
    // Max(a, b) = a > b ? a : b
    // NaN handling: matches NumPy's minimum/maximum — if either operand is NaN,
    // result is NaN (because the C# compare opcodes on NaN return 0).
    //
    // Branch-free equivalent via Math.Min/Math.Max would handle NaN differently
    // (returns the non-NaN operand) — NumPy's np.minimum/np.maximum return NaN,
    // so the branchy lowering matches NumPy exactly. For NumPy's np.fmin/np.fmax
    // (NaN-skipping) users can compose with IsNaN + Where.
    // =========================================================================

    internal sealed class MinMaxNode : NpyExpr
    {
        private readonly bool _isMin;
        private readonly NpyExpr _left;
        private readonly NpyExpr _right;

        public MinMaxNode(bool isMin, NpyExpr left, NpyExpr right)
        {
            _isMin = isMin;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        internal override bool SupportsSimd => false;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            // Prefer Math.Min/Max — they propagate NaN per IEEE 754, matching NumPy's
            // np.minimum/np.maximum. Fall back to a branchy select for dtypes without
            // a Math.Min/Max overload (Char, Boolean).
            EmitBranchy(il, ctx);
        }

        private void EmitBranchy(ILGenerator il, NpyExprCompileContext ctx)
        {
            var clrType = ILKernelGenerator.GetClrType(ctx.OutputType);
            var locL = il.DeclareLocal(clrType);
            var locR = il.DeclareLocal(clrType);

            _left.EmitScalar(il, ctx);
            il.Emit(OpCodes.Stloc, locL);
            _right.EmitScalar(il, ctx);
            il.Emit(OpCodes.Stloc, locR);

            // Prefer Math.Min/Max if available (NaN-propagating for floats).
            string methodName = _isMin ? "Min" : "Max";
            var method = typeof(Math).GetMethod(
                methodName,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static,
                null,
                new[] { clrType, clrType },
                null);
            if (method != null)
            {
                il.Emit(OpCodes.Ldloc, locL);
                il.Emit(OpCodes.Ldloc, locR);
                il.EmitCall(OpCodes.Call, method, null);
                return;
            }

            // Fallback: branchy select via comparison (for Char / Boolean).
            var lblElse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Ldloc, locR);
            ILKernelGenerator.EmitComparisonOperation(
                il,
                _isMin ? ComparisonOp.LessEqual : ComparisonOp.GreaterEqual,
                ctx.OutputType);
            il.Emit(OpCodes.Brfalse, lblElse);
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Br, lblEnd);
            il.MarkLabel(lblElse);
            il.Emit(OpCodes.Ldloc, locR);
            il.MarkLabel(lblEnd);
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            throw new InvalidOperationException("MinMaxNode has no vector path.");
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append(_isMin ? "Min(" : "Max(");
            _left.AppendSignature(sb);
            sb.Append(',');
            _right.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Where(cond, a, b) — scalar-only ternary
    //
    // cond is evaluated at the output dtype. Non-zero means "true".
    // Equivalent to np.where(cond, a, b), with cond coerced to bool.
    // =========================================================================

    internal sealed class WhereNode : NpyExpr
    {
        private readonly NpyExpr _cond;
        private readonly NpyExpr _a;
        private readonly NpyExpr _b;

        public WhereNode(NpyExpr cond, NpyExpr a, NpyExpr b)
        {
            _cond = cond ?? throw new ArgumentNullException(nameof(cond));
            _a = a ?? throw new ArgumentNullException(nameof(a));
            _b = b ?? throw new ArgumentNullException(nameof(b));
        }

        internal override bool SupportsSimd => false;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            var lblElse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // Evaluate cond in outputType, then compare to zero so we have a
            // verifiable I4 0/1 on the stack before brfalse.
            _cond.EmitScalar(il, ctx);
            EmitPushZero(il, ctx.OutputType);
            ILKernelGenerator.EmitComparisonOperation(il, ComparisonOp.NotEqual, ctx.OutputType);

            il.Emit(OpCodes.Brfalse, lblElse);

            _a.EmitScalar(il, ctx);
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblElse);
            _b.EmitScalar(il, ctx);

            il.MarkLabel(lblEnd);
        }

        private static void EmitPushZero(ILGenerator il, NPTypeCode type)
            => EmitPushZeroPublic(il, type);

        internal static void EmitPushZeroPublic(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 0f);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 0d);
                    break;
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    break;
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Ldc_I4_0);
                    break;
                case NPTypeCode.Decimal:
                    var fld = typeof(decimal).GetField(nameof(decimal.Zero));
                    il.Emit(OpCodes.Ldsfld, fld!);
                    break;
                default:
                    throw new NotSupportedException($"Zero-push unsupported for {type}");
            }
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            throw new InvalidOperationException("WhereNode has no vector path.");
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Where(");
            _cond.AppendSignature(sb);
            sb.Append(',');
            _a.AppendSignature(sb);
            sb.Append(',');
            _b.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Call — invoke an arbitrary .NET method (delegate or MethodInfo).
    //
    // THREE PATHS
    // -----------
    // 1. Static method, no captures  → emit `call <methodinfo>` directly.
    //    Zero indirection. Used when `Target == null && Method.IsStatic` for a
    //    Delegate, or when the user passes a MethodInfo without an instance.
    //
    // 2. Instance method with a target object → stash the target in the slot
    //    dictionary, emit a lookup for the target, then `callvirt <methodinfo>`.
    //
    // 3. Delegate with captured state (closure / instance method wrapper) →
    //    stash the whole delegate, emit a lookup, then `callvirt Invoke`.
    //
    // TYPE DISCIPLINE
    // ---------------
    // Per-argument auto-conversion from `ctx.OutputType` to the method's param
    // dtype; return value converted from the method's return dtype to
    // `ctx.OutputType`. Same model as InputNode's auto-convert — keeps the DSL
    // uniform.
    //
    // Unsupported param/return types (anything not in the 12-type set) are
    // rejected at node construction time.
    //
    // SIMD
    // ----
    // Always false. A managed call from inside a vector loop kills SIMD.
    // =========================================================================

    internal sealed class CallNode : NpyExpr
    {
        private enum Kind
        {
            StaticMethod,  // direct `call <methodinfo>`
            BoundTarget,   // load target from slots, then `callvirt <methodinfo>`
            Delegate,      // load delegate from slots, then `callvirt Invoke`
        }

        private readonly Kind _kind;
        private readonly System.Reflection.MethodInfo _method;
        private readonly Type _delegateType; // only for Kind.Delegate
        private readonly int _slotId;        // only for Kind.BoundTarget / Kind.Delegate
        private readonly NpyExpr[] _args;
        private readonly NPTypeCode[] _paramCodes;
        private readonly NPTypeCode _returnCode;
        private readonly string _signatureId;

        public CallNode(Delegate func, NpyExpr[] args)
        {
            if (func is null) throw new ArgumentNullException(nameof(func));
            if (args is null) throw new ArgumentNullException(nameof(args));
            foreach (var a in args)
                if (a is null) throw new ArgumentNullException(nameof(args), "No arg may be null.");

            _args = args;
            _delegateType = func.GetType();

            var mi = func.Method;
            var parameters = mi.GetParameters();
            if (parameters.Length != args.Length)
                throw new ArgumentException(
                    $"Delegate {mi.Name} expects {parameters.Length} arg(s), got {args.Length}.",
                    nameof(args));

            _paramCodes = MapParamCodes(parameters);
            _returnCode = MapReturnCode(mi.ReturnType, mi);

            if (func.Target is null && mi.IsStatic)
            {
                // Fast path: compile to a direct static call.
                _kind = Kind.StaticMethod;
                _method = mi;
                _slotId = -1;
            }
            else
            {
                // Slow path: stash whole delegate and call Invoke through slots.
                _kind = Kind.Delegate;
                _method = _delegateType.GetMethod("Invoke")
                    ?? throw new InvalidOperationException("Delegate has no Invoke method.");
                _slotId = DelegateSlots.RegisterDelegate(func);
            }

            _signatureId = BuildMethodSignatureId(mi);
        }

        public CallNode(System.Reflection.MethodInfo method, object? target, NpyExpr[] args)
        {
            if (method is null) throw new ArgumentNullException(nameof(method));
            if (args is null) throw new ArgumentNullException(nameof(args));
            foreach (var a in args)
                if (a is null) throw new ArgumentNullException(nameof(args), "No arg may be null.");

            _args = args;
            _delegateType = null!;

            var parameters = method.GetParameters();
            if (parameters.Length != args.Length)
                throw new ArgumentException(
                    $"Method {method.Name} expects {parameters.Length} arg(s), got {args.Length}.",
                    nameof(args));

            _paramCodes = MapParamCodes(parameters);
            _returnCode = MapReturnCode(method.ReturnType, method);

            if (target is null)
            {
                if (!method.IsStatic)
                    throw new ArgumentException(
                        $"Method {method.Name} is an instance method; pass a target object.",
                        nameof(target));
                _kind = Kind.StaticMethod;
                _method = method;
                _slotId = -1;
            }
            else
            {
                if (method.IsStatic)
                    throw new ArgumentException(
                        $"Method {method.Name} is static; do not pass a target object.",
                        nameof(target));
                if (!method.DeclaringType!.IsInstanceOfType(target))
                    throw new ArgumentException(
                        $"Target is {target.GetType().FullName}, method declares {method.DeclaringType.FullName}.",
                        nameof(target));
                _kind = Kind.BoundTarget;
                _method = method;
                _slotId = DelegateSlots.RegisterTarget(target);
            }

            _signatureId = BuildMethodSignatureId(method);
        }

        private static NPTypeCode[] MapParamCodes(System.Reflection.ParameterInfo[] parameters)
        {
            var codes = new NPTypeCode[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                var pt = parameters[i].ParameterType;
                var tc = pt.GetTypeCode();
                if (!IsSupported(tc))
                    throw new ArgumentException(
                        $"Parameter {i} type {pt.Name} is not one of the 12 supported NPTypeCode dtypes.",
                        nameof(parameters));
                codes[i] = tc;
            }
            return codes;
        }

        private static NPTypeCode MapReturnCode(Type returnType, System.Reflection.MethodInfo mi)
        {
            if (returnType == typeof(void))
                throw new ArgumentException(
                    $"Method {mi.Name} returns void; NpyExpr.Call requires a value-returning method.");
            var tc = returnType.GetTypeCode();
            if (!IsSupported(tc))
                throw new ArgumentException(
                    $"Return type {returnType.Name} of {mi.Name} is not one of the 12 supported NPTypeCode dtypes.");
            return tc;
        }

        private static bool IsSupported(NPTypeCode code)
            => code switch
            {
                NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or
                NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or
                NPTypeCode.Char or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => true,
                _ => false,
            };

        private static string BuildMethodSignatureId(System.Reflection.MethodInfo mi)
        {
            var sb = new StringBuilder();
            sb.Append(mi.DeclaringType?.FullName ?? "_");
            sb.Append('.').Append(mi.Name);
            sb.Append('#').Append(mi.MetadataToken);
            // Module handle disambiguates when the same metadata token collides
            // across dynamic assemblies (can happen with DynamicMethod).
            sb.Append('@').Append(mi.Module.ModuleVersionId);
            return sb.ToString();
        }

        internal override bool SupportsSimd => false;

        internal override void EmitScalar(ILGenerator il, NpyExprCompileContext ctx)
        {
            switch (_kind)
            {
                case Kind.StaticMethod:
                    EmitArgs(il, ctx);
                    il.EmitCall(OpCodes.Call, _method, null);
                    break;

                case Kind.BoundTarget:
                    // Load target: DelegateSlots.LookupTarget(slotId)  → object
                    il.Emit(OpCodes.Ldc_I4, _slotId);
                    il.EmitCall(OpCodes.Call, DelegateSlots.LookupTargetMethod, null);
                    // Cast to the method's declaring type
                    var declaring = _method.DeclaringType!;
                    if (declaring.IsValueType)
                    {
                        // Unbox to a managed reference; call uses managed ref for value-type 'this'
                        il.Emit(OpCodes.Unbox, declaring);
                    }
                    else
                    {
                        il.Emit(OpCodes.Castclass, declaring);
                    }
                    EmitArgs(il, ctx);
                    il.EmitCall(OpCodes.Callvirt, _method, null);
                    break;

                case Kind.Delegate:
                    // Load delegate: DelegateSlots.LookupDelegate(slotId) → Delegate
                    il.Emit(OpCodes.Ldc_I4, _slotId);
                    il.EmitCall(OpCodes.Call, DelegateSlots.LookupDelegateMethod, null);
                    il.Emit(OpCodes.Castclass, _delegateType);
                    EmitArgs(il, ctx);
                    il.EmitCall(OpCodes.Callvirt, _method, null);
                    break;
            }

            if (_returnCode != ctx.OutputType)
                ILKernelGenerator.EmitConvertTo(il, _returnCode, ctx.OutputType);
        }

        private void EmitArgs(ILGenerator il, NpyExprCompileContext ctx)
        {
            for (int i = 0; i < _args.Length; i++)
            {
                _args[i].EmitScalar(il, ctx);
                // Every arg leaves ctx.OutputType on the stack — convert if the
                // method's parameter dtype is different.
                if (_paramCodes[i] != ctx.OutputType)
                    ILKernelGenerator.EmitConvertTo(il, ctx.OutputType, _paramCodes[i]);
            }
        }

        internal override void EmitVector(ILGenerator il, NpyExprCompileContext ctx)
        {
            throw new InvalidOperationException("CallNode has no vector path.");
        }

        internal override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Call[").Append(_signatureId);
            if (_kind == Kind.BoundTarget)
                sb.Append(",target#").Append(_slotId);
            sb.Append("](");
            for (int i = 0; i < _args.Length; i++)
            {
                if (i > 0) sb.Append(',');
                _args[i].AppendSignature(sb);
            }
            sb.Append(')');
        }
    }

    // =========================================================================
    // DelegateSlots — process-wide registry of captured delegates and bound
    // instance targets, keyed by a monotonically-increasing int.
    //
    // The IL emitter stores an integer ID in the kernel's bytecode and looks
    // up the managed object at runtime. Strong references — entries live for
    // the process lifetime. Users should register delegates once at startup
    // (static field or DI singleton), not inside a hot loop.
    //
    // Thread-safe: ConcurrentDictionary + Interlocked.Increment.
    // =========================================================================

    internal static class DelegateSlots
    {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, Delegate> _delegates = new();
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<int, object> _targets = new();
        private static int _nextId;

        public static readonly System.Reflection.MethodInfo LookupDelegateMethod =
            typeof(DelegateSlots).GetMethod(nameof(LookupDelegate),
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;

        public static readonly System.Reflection.MethodInfo LookupTargetMethod =
            typeof(DelegateSlots).GetMethod(nameof(LookupTarget),
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;

        public static int RegisterDelegate(Delegate d)
        {
            int id = System.Threading.Interlocked.Increment(ref _nextId);
            _delegates[id] = d;
            return id;
        }

        public static int RegisterTarget(object t)
        {
            int id = System.Threading.Interlocked.Increment(ref _nextId);
            _targets[id] = t;
            return id;
        }

        // Called from emitted IL.
        public static Delegate LookupDelegate(int id) => _delegates[id];
        public static object LookupTarget(int id) => _targets[id];

        // Test hook.
        internal static int RegisteredCount => _delegates.Count + _targets.Count;

        internal static void Clear()
        {
            _delegates.Clear();
            _targets.Clear();
        }
    }
}
