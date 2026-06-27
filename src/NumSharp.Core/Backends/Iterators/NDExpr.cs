using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.cs — Expression DSL (Tier 3C of the custom-op API)
// =============================================================================
//
// A small algebraic AST over NDIter operands. Compiles to an
// NDInnerLoopFunc by emitting (scalarBody, vectorBody) pairs that
// DirectILKernelGenerator.CompileInnerLoop wraps in the standard 4× unroll shell.
//
// TYPE DISCIPLINE — TWO MODES
// ---------------------------
// Legacy mode (Compile): all intermediate computation happens in the output
// dtype. Input loads auto-promote to output dtype; constants are pushed as
// output dtype. Simple, but does NOT match NumPy for mixed-dtype trees
// (NumPy computes each ufunc node at its own result_type — int32*int32
// wraps in int32 even when the final result is float64).
//
// NumPy mode (CompileNumPy, used by np.evaluate): a typing pass resolves
// every node to its NumPy 2.x result_type (NEP50, incl. weak python-scalar
// semantics for constants); emission computes each node at its resolved
// dtype and converts at node edges. See NDExpr.Typing.cs.
//
// For fine-grained type control, use ExecuteElementWise directly (Tier 3B).
//
// SIMD
// ----
// The vector path is enabled iff every input type equals the output type
// (in NumPy mode: the whole tree resolves homogeneous) AND every node's op
// supports SIMD. Otherwise the compiled kernel carries a scalar-only body;
// the factory's strided fallback handles all cases.
//
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Abstract expression node. Subclasses describe computations over
    /// NDIter operands; Compile() produces an NDInnerLoopFunc.
    /// </summary>
    public abstract partial class NDExpr
    {
        // ----- Contract (internal API used by the compiler) -----

        /// <summary>
        /// Emit scalar code. On exit, the evaluation stack must have exactly
        /// one value of dtype <c>ctx.OutputType</c>.
        /// </summary>
        public abstract void EmitScalar(ILGenerator il, NDExprCompileContext ctx);

        /// <summary>
        /// Emit vector code. On exit, the evaluation stack must have exactly
        /// one <c>Vector{W}&lt;T&gt;</c> of element type <c>ctx.OutputType</c>.
        /// Called only when <see cref="SupportsSimd"/> is true and all input
        /// types equal the output type.
        /// </summary>
        public abstract void EmitVector(ILGenerator il, NDExprCompileContext ctx);

        /// <summary>
        /// True if this node and its entire sub-tree have a SIMD emit path. Structural and
        /// type-independent — prefer <see cref="SupportsSimdAt"/> for the compile decision, since
        /// some ops only vectorize at certain element types/runtimes.
        /// </summary>
        public abstract bool SupportsSimd { get; }

        /// <summary>
        /// Whether this node and its sub-tree have a SIMD emit path WHEN every operand and the
        /// output share element type <paramref name="t"/> (the compiler only vectorizes a tree when
        /// all its types are equal — see <see cref="Compile"/>). Defaults to the structural
        /// <see cref="SupportsSimd"/>; nodes whose SIMD-ability is type- or runtime-dependent
        /// override this. The key case: rounding ops (Floor/Ceil/Round/Truncate) bind a per-type
        /// <c>Vector{N}</c> BCL method that exists only for float/double on a capable runtime, so
        /// vectorizing them at an integer type (or on a runtime without the method) would hit the
        /// emitter's "Could not find ..." throw — this gate routes those to scalar instead.
        /// </summary>
        public virtual bool SupportsSimdAt(NPTypeCode t) => SupportsSimd;

        /// <summary>
        /// Stable structural signature. Used to derive a cache key when the
        /// user doesn't supply one.
        /// </summary>
        public abstract void AppendSignature(StringBuilder sb);

        // ----- Compilation -----

        /// <summary>
        /// Compile the tree to an <see cref="NDInnerLoopFunc"/>.
        /// </summary>
        public NDInnerLoopFunc Compile(
            NPTypeCode[] inputTypes, NPTypeCode outputType, string? cacheKey)
        {
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));

            string key = cacheKey ?? DeriveCacheKey(inputTypes, outputType);
            int nIn = inputTypes.Length;

            // Type-aware: a tree only vectorizes when every operand and the output share a type
            // (AllEqual), so SupportsSimdAt is evaluated at that single shared type. This catches
            // ops that vectorize only at certain element types/runtimes (e.g. rounding ops, which
            // have no Vector{N} method for integers or on pre-.NET-9 runtimes).
            bool wantSimd = AllEqual(inputTypes, outputType) && SupportsSimdAt(outputType);

            Action<ILGenerator> scalarBody = il =>
            {
                // Shell delivers N inputs on stack: stack[bottom]=in0, stack[top]=inN-1.
                // Stash each into a local (reverse order since we pop top first).
                var scalarLocals = new LocalBuilder[nIn];
                for (int i = nIn - 1; i >= 0; i--)
                {
                    scalarLocals[i] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(inputTypes[i]));
                    il.Emit(OpCodes.Stloc, scalarLocals[i]);
                }
                var ctx = new NDExprCompileContext(inputTypes, outputType, scalarLocals, vectorMode: false);
                EmitScalar(il, ctx);
                // Stack now: [result : outputType]  — factory stores it.
            };

            Action<ILGenerator>? vectorBody = null;
            if (wantSimd)
            {
                vectorBody = il =>
                {
                    var vectorLocals = new LocalBuilder[nIn];
                    var vecType = DirectILKernelGenerator.GetVectorType(DirectILKernelGenerator.GetClrType(inputTypes[0]));
                    for (int i = nIn - 1; i >= 0; i--)
                    {
                        vectorLocals[i] = il.DeclareLocal(vecType);
                        il.Emit(OpCodes.Stloc, vectorLocals[i]);
                    }
                    var ctx = new NDExprCompileContext(inputTypes, outputType, vectorLocals, vectorMode: true);
                    EmitVector(il, ctx);
                };
            }

            var operandTypes = new NPTypeCode[nIn + 1];
            Array.Copy(inputTypes, operandTypes, nIn);
            operandTypes[nIn] = outputType;

            return DirectILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, key);
        }

        private protected string DeriveCacheKey(NPTypeCode[] inputTypes, NPTypeCode outputType)
        {
            var sb = new StringBuilder("NDExpr:");
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
        public static NDExpr Input(int index) => new InputNode(index);

        /// <summary>Push a constant of the given .NET type. Value is converted to the output dtype when evaluated.</summary>
        public static NDExpr Const(double value) => new ConstNode(value);
        public static NDExpr Const(float value) => new ConstNode(value);
        public static NDExpr Const(long value) => new ConstNode(value);
        public static NDExpr Const(int value) => new ConstNode(value);

        // ===================================================================
        // Binary factories
        // ===================================================================

        // Arithmetic
        public static NDExpr Add(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Add, a, b);
        public static NDExpr Subtract(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Subtract, a, b);
        public static NDExpr Multiply(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Multiply, a, b);
        public static NDExpr Divide(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Divide, a, b);
        public static NDExpr Mod(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Mod, a, b);
        public static NDExpr Power(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.Power, a, b);
        public static NDExpr FloorDivide(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.FloorDivide, a, b);
        public static NDExpr ATan2(NDExpr y, NDExpr x) => new BinaryNode(BinaryOp.ATan2, y, x);

        // Bitwise
        public static NDExpr BitwiseAnd(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.BitwiseAnd, a, b);
        public static NDExpr BitwiseOr(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.BitwiseOr, a, b);
        public static NDExpr BitwiseXor(NDExpr a, NDExpr b) => new BinaryNode(BinaryOp.BitwiseXor, a, b);

        // Scalar-branchy combinators compiled to IL
        public static NDExpr Min(NDExpr a, NDExpr b) => new MinMaxNode(isMin: true, a, b);
        public static NDExpr Max(NDExpr a, NDExpr b) => new MinMaxNode(isMin: false, a, b);
        public static NDExpr Clamp(NDExpr x, NDExpr lo, NDExpr hi) => Min(Max(x, lo), hi);
        public static NDExpr Where(NDExpr cond, NDExpr a, NDExpr b) => new WhereNode(cond, a, b);

        // ===================================================================
        // Unary factories
        // ===================================================================

        // Core arithmetic
        public static NDExpr Sqrt(NDExpr x) => new UnaryNode(UnaryOp.Sqrt, x);
        public static NDExpr Abs(NDExpr x) => new UnaryNode(UnaryOp.Abs, x);
        public static NDExpr Negate(NDExpr x) => new UnaryNode(UnaryOp.Negate, x);
        public static NDExpr Square(NDExpr x) => new UnaryNode(UnaryOp.Square, x);
        public static NDExpr Reciprocal(NDExpr x) => new UnaryNode(UnaryOp.Reciprocal, x);
        public static NDExpr Sign(NDExpr x) => new UnaryNode(UnaryOp.Sign, x);
        public static NDExpr Cbrt(NDExpr x) => new UnaryNode(UnaryOp.Cbrt, x);

        // Exp / Log family
        public static NDExpr Exp(NDExpr x) => new UnaryNode(UnaryOp.Exp, x);
        public static NDExpr Exp2(NDExpr x) => new UnaryNode(UnaryOp.Exp2, x);
        public static NDExpr Expm1(NDExpr x) => new UnaryNode(UnaryOp.Expm1, x);
        public static NDExpr Log(NDExpr x) => new UnaryNode(UnaryOp.Log, x);
        public static NDExpr Log2(NDExpr x) => new UnaryNode(UnaryOp.Log2, x);
        public static NDExpr Log10(NDExpr x) => new UnaryNode(UnaryOp.Log10, x);
        public static NDExpr Log1p(NDExpr x) => new UnaryNode(UnaryOp.Log1p, x);

        // Trigonometric
        public static NDExpr Sin(NDExpr x) => new UnaryNode(UnaryOp.Sin, x);
        public static NDExpr Cos(NDExpr x) => new UnaryNode(UnaryOp.Cos, x);
        public static NDExpr Tan(NDExpr x) => new UnaryNode(UnaryOp.Tan, x);
        public static NDExpr Sinh(NDExpr x) => new UnaryNode(UnaryOp.Sinh, x);
        public static NDExpr Cosh(NDExpr x) => new UnaryNode(UnaryOp.Cosh, x);
        public static NDExpr Tanh(NDExpr x) => new UnaryNode(UnaryOp.Tanh, x);
        public static NDExpr ASin(NDExpr x) => new UnaryNode(UnaryOp.ASin, x);
        public static NDExpr ACos(NDExpr x) => new UnaryNode(UnaryOp.ACos, x);
        public static NDExpr ATan(NDExpr x) => new UnaryNode(UnaryOp.ATan, x);
        public static NDExpr Deg2Rad(NDExpr x) => new UnaryNode(UnaryOp.Deg2Rad, x);
        public static NDExpr Rad2Deg(NDExpr x) => new UnaryNode(UnaryOp.Rad2Deg, x);

        // Rounding
        public static NDExpr Floor(NDExpr x) => new UnaryNode(UnaryOp.Floor, x);
        public static NDExpr Ceil(NDExpr x) => new UnaryNode(UnaryOp.Ceil, x);
        public static NDExpr Round(NDExpr x) => new UnaryNode(UnaryOp.Round, x);
        public static NDExpr Truncate(NDExpr x) => new UnaryNode(UnaryOp.Truncate, x);

        // Bitwise / logical
        public static NDExpr BitwiseNot(NDExpr x) => new UnaryNode(UnaryOp.BitwiseNot, x);
        public static NDExpr LogicalNot(NDExpr x) => new UnaryNode(UnaryOp.LogicalNot, x);

        // Predicates (returns numeric 0/1 at output dtype — NumPy-compatible)
        public static NDExpr IsNaN(NDExpr x) => new UnaryNode(UnaryOp.IsNan, x);
        public static NDExpr IsFinite(NDExpr x) => new UnaryNode(UnaryOp.IsFinite, x);
        public static NDExpr IsInf(NDExpr x) => new UnaryNode(UnaryOp.IsInf, x);

        // ===================================================================
        // Comparison factories (produce 0/1 at output dtype)
        // ===================================================================

        public static NDExpr Equal(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.Equal, a, b);
        public static NDExpr NotEqual(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.NotEqual, a, b);
        public static NDExpr Less(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.Less, a, b);
        public static NDExpr LessEqual(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.LessEqual, a, b);
        public static NDExpr Greater(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.Greater, a, b);
        public static NDExpr GreaterEqual(NDExpr a, NDExpr b) => new ComparisonNode(ComparisonOp.GreaterEqual, a, b);

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
        //   (b) `Call(Delegate func, params NDExpr[] args)` — catch-all for
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
        public static NDExpr Call(System.Reflection.MethodInfo method, params NDExpr[] args)
            => new CallNode(method, target: null, args);

        /// <summary>Invoke an instance method on a target object.</summary>
        public static NDExpr Call(System.Reflection.MethodInfo method, object target, params NDExpr[] args)
            => new CallNode(method, target, args);

        /// <summary>Invoke any delegate. Method-group arguments need a typed Func overload; use a cast or the typed overloads below.</summary>
        public static NDExpr Call(Delegate func, params NDExpr[] args)
            => new CallNode(func, args);

        // Typed Func<...> overloads — enable `NDExpr.Call(Math.Sqrt, x)` without cast.
        public static NDExpr Call<TR>(Func<TR> func)
            => new CallNode(func, Array.Empty<NDExpr>());
        public static NDExpr Call<T1, TR>(Func<T1, TR> func, NDExpr a1)
            => new CallNode(func, new[] { a1 });
        public static NDExpr Call<T1, T2, TR>(Func<T1, T2, TR> func, NDExpr a1, NDExpr a2)
            => new CallNode(func, new[] { a1, a2 });
        public static NDExpr Call<T1, T2, T3, TR>(Func<T1, T2, T3, TR> func, NDExpr a1, NDExpr a2, NDExpr a3)
            => new CallNode(func, new[] { a1, a2, a3 });
        public static NDExpr Call<T1, T2, T3, T4, TR>(Func<T1, T2, T3, T4, TR> func, NDExpr a1, NDExpr a2, NDExpr a3, NDExpr a4)
            => new CallNode(func, new[] { a1, a2, a3, a4 });

        // ===================================================================
        // Operator overloads (syntactic sugar)
        // ===================================================================

        public static NDExpr operator +(NDExpr a, NDExpr b) => Add(a, b);
        public static NDExpr operator -(NDExpr a, NDExpr b) => Subtract(a, b);
        public static NDExpr operator *(NDExpr a, NDExpr b) => Multiply(a, b);
        public static NDExpr operator /(NDExpr a, NDExpr b) => Divide(a, b);
        public static NDExpr operator %(NDExpr a, NDExpr b) => Mod(a, b);
        public static NDExpr operator &(NDExpr a, NDExpr b) => BitwiseAnd(a, b);
        public static NDExpr operator |(NDExpr a, NDExpr b) => BitwiseOr(a, b);
        public static NDExpr operator ^(NDExpr a, NDExpr b) => BitwiseXor(a, b);
        public static NDExpr operator -(NDExpr a) => Negate(a);
        public static NDExpr operator ~(NDExpr a) => BitwiseNot(a);
        public static NDExpr operator !(NDExpr a) => LogicalNot(a);
    }

    // =========================================================================
    // Compile-time context shared with each node
    // =========================================================================

    public sealed class NDExprCompileContext
    {
        public NPTypeCode[] InputTypes { get; }
        public NPTypeCode OutputType { get; }
        public LocalBuilder[] InputLocals { get; }
        public bool VectorMode { get; }

        /// <summary>
        /// Per-node resolved dtypes (NumPy result_type mode, produced by the
        /// typing pass for <see cref="NDExpr.CompileNumPy"/>). Null in legacy
        /// mode, where every node computes at <see cref="OutputType"/>.
        /// Keyed by node reference.
        /// </summary>
        public IReadOnlyDictionary<NDExpr, NPTypeCode>? NodeTypes { get; }

        public NDExprCompileContext(
            NPTypeCode[] inputTypes, NPTypeCode outputType,
            LocalBuilder[] inputLocals, bool vectorMode)
            : this(inputTypes, outputType, inputLocals, vectorMode, null)
        {
        }

        public NDExprCompileContext(
            NPTypeCode[] inputTypes, NPTypeCode outputType,
            LocalBuilder[] inputLocals, bool vectorMode,
            IReadOnlyDictionary<NDExpr, NPTypeCode>? nodeTypes)
        {
            InputTypes = inputTypes;
            OutputType = outputType;
            InputLocals = inputLocals;
            VectorMode = vectorMode;
            NodeTypes = nodeTypes;
        }

        /// <summary>
        /// The dtype this node computes at: its typing-pass entry in NumPy
        /// mode, or the uniform <see cref="OutputType"/> in legacy mode.
        /// </summary>
        internal NPTypeCode TypeOf(NDExpr node)
            => NodeTypes is null
                ? OutputType
                : NodeTypes.TryGetValue(node, out var t)
                    ? t
                    : throw new InvalidOperationException(
                        $"NDExpr typing pass produced no dtype for node {node.GetType().Name} — typing/emission mismatch.");
    }

    // =========================================================================
    // Node: Input(i) — reference operand i
    // =========================================================================

    public sealed partial class InputNode : NDExpr
    {
        private readonly int _index;
        public InputNode(int index)
        {
            if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
            _index = index;
        }

        internal int Index => _index;

        public override bool SupportsSimd => true;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            if (_index >= ctx.InputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({_index}) out of range; compile provided {ctx.InputTypes.Length} inputs.");

            il.Emit(OpCodes.Ldloc, ctx.InputLocals[_index]);
            // Leave this node's resolved dtype on the stack. Legacy mode
            // resolves every node to OutputType (auto-promote at load);
            // NumPy mode resolves to the operand's native dtype, so parents
            // convert at their edges instead.
            var inType = ctx.InputTypes[_index];
            DirectILKernelGenerator.EmitConvertTo(il, inType, ctx.TypeOf(this));
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            if (_index >= ctx.InputTypes.Length)
                throw new InvalidOperationException(
                    $"Input({_index}) out of range; compile provided {ctx.InputTypes.Length} inputs.");

            // Vector mode is only used when all input types == output type
            // (enforced by Compile), so no conversion is needed here.
            il.Emit(OpCodes.Ldloc, ctx.InputLocals[_index]);
        }

        public override void AppendSignature(StringBuilder sb)
            => sb.Append("In[").Append(_index).Append(']');
    }

    // =========================================================================
    // Node: Constant
    // =========================================================================

    public sealed partial class ConstNode : NDExpr
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

        internal bool IsIntegerLiteral => _isIntegerLiteral;
        internal long IntegerValue => _valueInt;
        internal double FloatValue => _valueFp;

        public override bool SupportsSimd => true;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            EmitLoadTyped(il, ctx.TypeOf(this));
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            EmitLoadTyped(il, ctx.TypeOf(this));
            DirectILKernelGenerator.EmitVectorCreate(il, ctx.TypeOf(this));
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
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Char:
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_I4, _isIntegerLiteral ? (int)_valueInt : (int)_valueFp);
                    return;
                case NPTypeCode.Half:
                    // No Ldc for Half — load double then convert (Half consts
                    // arise from NEP50 weak adoption: f2_array + 2.5 stays f2).
                    il.Emit(OpCodes.Ldc_R8, _isIntegerLiteral ? _valueInt : _valueFp);
                    DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, NPTypeCode.Half);
                    return;
                case NPTypeCode.Decimal:
                    if (_isIntegerLiteral)
                    {
                        il.Emit(OpCodes.Ldc_I8, _valueInt);
                        DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int64, NPTypeCode.Decimal);
                    }
                    else
                    {
                        il.Emit(OpCodes.Ldc_R8, _valueFp);
                        DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, NPTypeCode.Decimal);
                    }
                    return;
                case NPTypeCode.Complex:
                    il.Emit(OpCodes.Ldc_R8, _isIntegerLiteral ? _valueInt : _valueFp);
                    DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Double, NPTypeCode.Complex);
                    return;
                default:
                    throw new NotSupportedException(
                        $"ConstNode cannot emit for output dtype {target}.");
            }
        }

        public override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Const[");
            if (_isIntegerLiteral) sb.Append(_valueInt); else sb.Append(_valueFp);
            sb.Append(']');
        }
    }

    // =========================================================================
    // Node: Binary op
    // =========================================================================

    public sealed partial class BinaryNode : NDExpr
    {
        private readonly BinaryOp _op;
        private readonly NDExpr _left;
        private readonly NDExpr _right;

        public BinaryNode(BinaryOp op, NDExpr left, NDExpr right)
        {
            _op = op;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        public override bool SupportsSimd
            => _left.SupportsSimd && _right.SupportsSimd && IsSimdOp(_op);

        public override bool SupportsSimdAt(NPTypeCode t)
            => _left.SupportsSimdAt(t) && _right.SupportsSimdAt(t) && IsSimdOp(_op);

        // Must match DirectILKernelGenerator.EmitVectorOperation's supported set.
        // Mod, Power, FloorDivide, ATan2 are scalar-only.
        private static bool IsSimdOp(BinaryOp op)
            => op == BinaryOp.Add || op == BinaryOp.Subtract ||
               op == BinaryOp.Multiply || op == BinaryOp.Divide ||
               op == BinaryOp.BitwiseAnd || op == BinaryOp.BitwiseOr ||
               op == BinaryOp.BitwiseXor;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            var my = ctx.TypeOf(this);
            _left.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_left), my);
            _right.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_right), my);

            // NumPy's bool loops: add(bool,bool) is logical OR, multiply is
            // logical AND. The generic Add opcode would store byte 2 for
            // True+True, which is not a valid bool.
            if (my == NPTypeCode.Boolean && _op == BinaryOp.Add)
            {
                il.Emit(OpCodes.Or);
                return;
            }

            if (my == NPTypeCode.Boolean && _op == BinaryOp.Multiply)
            {
                il.Emit(OpCodes.And);
                return;
            }

            DirectILKernelGenerator.EmitScalarOperation(il, _op, my);
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            _left.EmitVector(il, ctx);
            _right.EmitVector(il, ctx);
            DirectILKernelGenerator.EmitVectorOperation(il, _op, ctx.OutputType);
        }

        public override void AppendSignature(StringBuilder sb)
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

    public sealed partial class UnaryNode : NDExpr
    {
        private readonly UnaryOp _op;
        private readonly NDExpr _child;

        public UnaryNode(UnaryOp op, NDExpr child)
        {
            _op = op;
            _child = child ?? throw new ArgumentNullException(nameof(child));
        }

        public override bool SupportsSimd
            => _child.SupportsSimd && IsSimdUnary(_op);

        public override bool SupportsSimdAt(NPTypeCode t)
            => _child.SupportsSimdAt(t) && IsSimdUnaryAt(_op, t);

        // Type/runtime-aware refinement of IsSimdUnary. The rounding family binds a per-type
        // Vector{N} BCL method that exists only for float/double on a capable runtime (Floor/Ceil
        // .NET 7+, Round/Truncate .NET 9+), so its SIMD-ability depends on the element type AND the
        // running runtime — defer to the shared capability gate. This is what lets the fused path
        // vectorize Round/Truncate on .NET 9+ while (a) staying scalar on net8.0 and (b) NOT
        // crashing on integer rounding, where the typing keeps the dtype and there is no
        // Vector{N}.Floor(int) (the gate returns false -> scalar identity emit).
        private static bool IsSimdUnaryAt(UnaryOp op, NPTypeCode t)
            => IsRoundingOp(op)
                ? DirectILKernelGenerator.RoundingVectorSimdAvailable(op, t)
                : IsSimdUnary(op);

        // Structural SIMD set used by the type-independent SupportsSimd. The rounding family
        // (Floor/Ceil/Round/Truncate) is gated per type+runtime by IsSimdUnaryAt instead, so it is
        // omitted here — SupportsSimdAt is the gate the compiler actually consults.
        private static bool IsSimdUnary(UnaryOp op)
            => op == UnaryOp.Negate || op == UnaryOp.Abs || op == UnaryOp.Sqrt ||
               op == UnaryOp.Square || op == UnaryOp.Reciprocal ||
               op == UnaryOp.Deg2Rad || op == UnaryOp.Rad2Deg || op == UnaryOp.BitwiseNot;

        // Predicates leave a bool (I4 0/1) on the stack — not outputType. The wrapper
        // below converts to outputType so the factory's Stind matches.
        private static bool IsPredicateResult(UnaryOp op)
            => op == UnaryOp.IsNan || op == UnaryOp.IsFinite || op == UnaryOp.IsInf;

        // NumPy preserves integer dtypes through floor/ceil/round/trunc — the op
        // is an identity there (and Math.Floor has no integer overloads to call).
        private static bool IsRoundingOp(UnaryOp op)
            => op == UnaryOp.Floor || op == UnaryOp.Ceil ||
               op == UnaryOp.Round || op == UnaryOp.Truncate;

        private static bool IsIntegerKind(NPTypeCode t)
            => t == NPTypeCode.Boolean || t == NPTypeCode.Byte || t == NPTypeCode.SByte ||
               t == NPTypeCode.Int16 || t == NPTypeCode.UInt16 || t == NPTypeCode.Char ||
               t == NPTypeCode.Int32 || t == NPTypeCode.UInt32 ||
               t == NPTypeCode.Int64 || t == NPTypeCode.UInt64;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            var my = ctx.TypeOf(this);
            var childType = ctx.TypeOf(_child);

            // LogicalNot needs a special path. DirectILKernelGenerator's emit uses Ldc_I4_0+Ceq
            // which is only correct when the input value fits in I4 (Int32 and narrower).
            // For Int64/Single/Double/Decimal the types mismatch on the stack. Rewrite
            // as (x == 0) using the comparison emit, which handles all types correctly.
            // The test runs at the CHILD's dtype (NumPy: logical_not(f8) inspects f8).
            if (_op == UnaryOp.LogicalNot)
            {
                _child.EmitScalar(il, ctx);
                WhereNode.EmitPushZeroPublic(il, childType);
                DirectILKernelGenerator.EmitComparisonOperation(il, ComparisonOp.Equal, childType);
                DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, my);
                return;
            }

            // Predicates also run at the child's dtype (NumPy: isnan(int) is
            // all-False without promoting the input).
            if (IsPredicateResult(_op))
            {
                _child.EmitScalar(il, ctx);
                DirectILKernelGenerator.EmitUnaryScalarOperation(il, _op, childType);
                DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, my);
                return;
            }

            _child.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, childType, my);

            // Identity cases at integer/bool dtypes (NumPy preserves the value).
            if (IsRoundingOp(_op) && IsIntegerKind(my))
                return;
            if (_op == UnaryOp.Abs && my == NPTypeCode.Boolean)
                return;

            // NumPy invert on bool is logical not; the raw Not opcode on the
            // I4 0/1 would produce -1/-2.
            if (_op == UnaryOp.BitwiseNot && my == NPTypeCode.Boolean)
            {
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Xor);
                return;
            }

            DirectILKernelGenerator.EmitUnaryScalarOperation(il, _op, my);
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            _child.EmitVector(il, ctx);
            DirectILKernelGenerator.EmitUnaryVectorOperation(il, _op, ctx.OutputType);
        }

        public override void AppendSignature(StringBuilder sb)
        {
            sb.Append(_op).Append('(');
            _child.AppendSignature(sb);
            sb.Append(')');
        }
    }

    // =========================================================================
    // Node: Comparison op (produces numeric 0/1 at output dtype)
    //
    // Comparisons in NumPy return bool arrays, but NDExpr's single-output-dtype
    // model collapses that to "0 or 1 at output dtype", which composes cleanly
    // with arithmetic (e.g. (x > 0) * x for ReLU). The I4 0/1 produced by
    // EmitComparisonOperation is converted to the output dtype after emission.
    //
    // Scalar-only — SIMD would require writing bool output and rerouting through
    // the Comparison kernel pipeline, which is beyond this tier.
    // =========================================================================

    public sealed partial class ComparisonNode : NDExpr
    {
        private readonly ComparisonOp _op;
        private readonly NDExpr _left;
        private readonly NDExpr _right;

        public ComparisonNode(ComparisonOp op, NDExpr left, NDExpr right)
        {
            _op = op;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            var my = ctx.TypeOf(this); // Boolean in NumPy mode; OutputType legacy
            var lT = ctx.TypeOf(_left);
            var rT = ctx.TypeOf(_right);
            // NumPy compares at the operands' common dtype (result_type of the
            // two children), then yields bool. Legacy mode compares at
            // OutputType, where both children already sit.
            var cmpType = ctx.NodeTypes is null
                ? ctx.OutputType
                : NDExprTypeRules.PromoteStrong(lT, rT);

            _left.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, lT, cmpType);
            _right.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, rT, cmpType);
            DirectILKernelGenerator.EmitComparisonOperation(il, _op, cmpType);
            // EmitComparisonOperation leaves an I4 (0 or 1) on the stack.
            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, my);
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            throw new InvalidOperationException("ComparisonNode has no vector path.");
        }

        public override void AppendSignature(StringBuilder sb)
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

    public sealed partial class MinMaxNode : NDExpr
    {
        private readonly bool _isMin;
        private readonly NDExpr _left;
        private readonly NDExpr _right;

        public MinMaxNode(bool isMin, NDExpr left, NDExpr right)
        {
            _isMin = isMin;
            _left = left ?? throw new ArgumentNullException(nameof(left));
            _right = right ?? throw new ArgumentNullException(nameof(right));
        }

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            // Prefer Math.Min/Max — they propagate NaN per IEEE 754, matching NumPy's
            // np.minimum/np.maximum. Fall back to a branchy select for dtypes without
            // a Math.Min/Max overload (Char, Boolean, Half, Complex).
            EmitBranchy(il, ctx);
        }

        private void EmitBranchy(ILGenerator il, NDExprCompileContext ctx)
        {
            var my = ctx.TypeOf(this);
            var clrType = DirectILKernelGenerator.GetClrType(my);
            var locL = il.DeclareLocal(clrType);
            var locR = il.DeclareLocal(clrType);

            _left.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_left), my);
            il.Emit(OpCodes.Stloc, locL);
            _right.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_right), my);
            il.Emit(OpCodes.Stloc, locR);

            // Prefer Math.Min/Max if available (NaN-propagating for floats).
            // ScalarMethodCache.Get throws on missing; fall back to the manual ldloc/branch
            // path below for types without a Math overload (e.g. Char).
            string methodName = _isMin ? "Min" : "Max";
            System.Reflection.MethodInfo method = null;
            try { method = ScalarMethodCache.Get(typeof(Math), methodName, clrType, clrType); }
            catch (MissingMethodException) { /* fall through */ }
            if (method != null)
            {
                il.Emit(OpCodes.Ldloc, locL);
                il.Emit(OpCodes.Ldloc, locR);
                il.EmitCall(OpCodes.Call, method, null);
                return;
            }

            // Fallback: branchy select via comparison (for Char / Boolean / Half).
            var lblElse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Ldloc, locR);
            DirectILKernelGenerator.EmitComparisonOperation(
                il,
                _isMin ? ComparisonOp.LessEqual : ComparisonOp.GreaterEqual,
                my);
            il.Emit(OpCodes.Brfalse, lblElse);
            il.Emit(OpCodes.Ldloc, locL);
            il.Emit(OpCodes.Br, lblEnd);
            il.MarkLabel(lblElse);
            il.Emit(OpCodes.Ldloc, locR);
            il.MarkLabel(lblEnd);
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            throw new InvalidOperationException("MinMaxNode has no vector path.");
        }

        public override void AppendSignature(StringBuilder sb)
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

    public sealed partial class WhereNode : NDExpr
    {
        private readonly NDExpr _cond;
        private readonly NDExpr _a;
        private readonly NDExpr _b;

        public WhereNode(NDExpr cond, NDExpr a, NDExpr b)
        {
            _cond = cond ?? throw new ArgumentNullException(nameof(cond));
            _a = a ?? throw new ArgumentNullException(nameof(a));
            _b = b ?? throw new ArgumentNullException(nameof(b));
        }

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
        {
            var my = ctx.TypeOf(this);
            var condType = ctx.TypeOf(_cond);
            var lblElse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // Evaluate cond at its own dtype (NumPy nonzero-tests the
            // condition without promoting it), then compare to zero so we
            // have a verifiable I4 0/1 on the stack before brfalse.
            _cond.EmitScalar(il, ctx);
            EmitPushZero(il, condType);
            DirectILKernelGenerator.EmitComparisonOperation(il, ComparisonOp.NotEqual, condType);

            il.Emit(OpCodes.Brfalse, lblElse);

            _a.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_a), my);
            il.Emit(OpCodes.Br, lblEnd);

            il.MarkLabel(lblElse);
            _b.EmitScalar(il, ctx);
            DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_b), my);

            il.MarkLabel(lblEnd);
        }

        private static void EmitPushZero(ILGenerator il, NPTypeCode type)
            => EmitPushZeroPublic(il, type);

        public static void EmitPushZeroPublic(ILGenerator il, NPTypeCode type)
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
                case NPTypeCode.Complex:
                    // System.Numerics.Complex.Zero is a static readonly field (0 + 0i). Needed when
                    // a mixed where(cond, complex, real) promotes the real operand to complex.
                    var cZero = typeof(System.Numerics.Complex).GetField(nameof(System.Numerics.Complex.Zero));
                    il.Emit(OpCodes.Ldsfld, cZero!);
                    break;
                case NPTypeCode.Half:
                    // Half has no Zero constant; push float 0 and convert via the explicit operator.
                    il.Emit(OpCodes.Ldc_R4, 0f);
                    il.Emit(OpCodes.Call, typeof(Half).GetMethod("op_Explicit", new[] { typeof(float) })
                        ?? throw new MissingMethodException(typeof(Half).FullName, "op_Explicit(float)"));
                    break;
                default:
                    throw new NotSupportedException($"Zero-push unsupported for {type}");
            }
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            throw new InvalidOperationException("WhereNode has no vector path.");
        }

        public override void AppendSignature(StringBuilder sb)
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

    public sealed partial class CallNode : NDExpr
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
        private readonly NDExpr[] _args;
        private readonly NPTypeCode[] _paramCodes;
        private readonly NPTypeCode _returnCode;
        private readonly string _signatureId;

        public CallNode(Delegate func, NDExpr[] args)
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

        public CallNode(System.Reflection.MethodInfo method, object? target, NDExpr[] args)
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
                    $"Method {mi.Name} returns void; NDExpr.Call requires a value-returning method.");
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

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
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

            DirectILKernelGenerator.EmitConvertTo(il, _returnCode, ctx.TypeOf(this));
        }

        private void EmitArgs(ILGenerator il, NDExprCompileContext ctx)
        {
            for (int i = 0; i < _args.Length; i++)
            {
                _args[i].EmitScalar(il, ctx);
                // Each arg leaves its own resolved dtype on the stack
                // (== ctx.OutputType in legacy mode) — convert to the
                // method's parameter dtype if different.
                DirectILKernelGenerator.EmitConvertTo(il, ctx.TypeOf(_args[i]), _paramCodes[i]);
            }
        }

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
        {
            throw new InvalidOperationException("CallNode has no vector path.");
        }

        public override void AppendSignature(StringBuilder sb)
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

    public static class DelegateSlots
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
        public static int RegisteredCount => _delegates.Count + _targets.Count;

        public static void Clear()
        {
            _delegates.Clear();
            _targets.Clear();
        }
    }
}
