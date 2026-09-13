using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Program.cs — the compiled program behind np.evaluate (ndexpr-evaluate.md Phase 3)
// =============================================================================
//
// np.evaluate used to redo, on EVERY call, everything that depends only on the tree and the
// operand dtypes: rebind the array leaves (a cloned tree), re-run the NEP50 typing pass (a
// reference-keyed Dictionary), rebuild the kernel's string cache key (a StringBuilder), and
// re-plan the vector emission — ~1.0 µs and ~3 KB of garbage per call at n = 8, three times the
// unfused chain's fixed cost, so every small-array call lost to the chain it replaces.
//
// A program is that dtype-dependent work done ONCE per root NDExpr instance and cached on it:
// the bound tree, the operand dtype signature, the resolved result dtype and the compiled
// kernel (elementwise), or the reduction node with its lazily compiled flat / axis kernels.
// The embedded-array form owns its operand list (the arrays the tree references — the same
// instances every call, so their dtypes cannot change); the positional form re-validates the
// dtype signature per call and recompiles on a change. Programs are immutable snapshots, so
// the cache is a benign race: two threads racing on a fresh tree compile the same kernel
// (which the kernel cache dedups) and one of them wins the slot.
//
// CompiledExpression (NDExpr.Compile) is the explicit handle over the same machinery — numexpr's
// NumExpr(expr) object: compile ahead of the hot loop, evaluate many times, and for the
// positional form pin ONE dtype signature (a call with other dtypes is an error, not a silent
// recompile).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    [NDBorrowed] // the embedded operands are the caller's arrays, referenced by the tree itself
    internal sealed class NDExprProgram
    {
        /// <summary>The tree with every array leaf rewritten to a positional <see cref="InputNode"/>.</summary>
        public readonly NDExpr Bound;

        /// <summary>The distinct arrays the tree references (embedded form), or null for the positional form.</summary>
        public readonly NDArray[] EmbeddedOperands;

        public readonly NPTypeCode[] InputTypes;
        public readonly bool ForcedScalar;

        /// <summary>The fused elementwise kernel; null when the root is a reduction.</summary>
        public readonly NDInnerLoopFunc Kernel;

        /// <summary>The tree's NumPy result dtype (the elementwise output, or the reduction's result).</summary>
        public readonly NPTypeCode ResultType;

        /// <summary>The root reduction, or null for an elementwise tree.</summary>
        public readonly ReduceNode Reduce;

        /// <summary>The reduction's accumulator dtype (see <see cref="ReduceNode.ResolveAccType"/>).</summary>
        public readonly NPTypeCode ReduceAccType;

        private NDInnerLoopFunc _flatReduce;
        private NDInnerLoopFunc _axisReduce;

        private NDExprProgram(NDExpr bound, NDArray[] embedded, NPTypeCode[] inputTypes, bool forcedScalar,
            NDInnerLoopFunc kernel, NPTypeCode resultType, ReduceNode reduce, NPTypeCode reduceAcc)
        {
            Bound = bound;
            EmbeddedOperands = embedded;
            InputTypes = inputTypes;
            ForcedScalar = forcedScalar;
            Kernel = kernel;
            ResultType = resultType;
            Reduce = reduce;
            ReduceAccType = reduceAcc;
        }

        /// <summary>The flat (axis=None) accumulating kernel, compiled on first use.</summary>
        public NDInnerLoopFunc FlatReduceKernel
            => _flatReduce ??= Reduce.CompileReduceKernel(InputTypes, out _, out _);

        /// <summary>The axis-aware accumulating kernel (axis-independent), compiled on first use.</summary>
        public NDInnerLoopFunc AxisReduceKernel
            => _axisReduce ??= Reduce.CompileAxisReduceKernel(InputTypes, out _, out _);

        /// <summary>Positional form: does this program serve these operands' dtypes?</summary>
        public bool Matches(NDArray[] operands)
        {
            if (operands.Length != InputTypes.Length)
                return false;
            for (int i = 0; i < operands.Length; i++)
                if (operands[i].typecode != InputTypes[i])
                    return false;
            return true;
        }

        /// <summary>
        /// Bind, type and compile <paramref name="root"/> against explicit operands (null for the
        /// embedded-array form). Validation errors (no arrays, mixed binding styles, nested / non-root
        /// reductions, NumPy typing errors) surface here and leave no program cached.
        /// </summary>
        public static NDExprProgram Build(NDExpr root, NDArray[] positional)
        {
            NPTypeCode[] types = null;
            if (positional is not null)
            {
                types = new NPTypeCode[positional.Length];
                for (int i = 0; i < positional.Length; i++)
                    types[i] = positional[i].typecode;
            }

            return Build(root, types);
        }

        /// <summary>
        /// Bind, type and compile <paramref name="root"/> for a positional dtype signature (null for the
        /// embedded-array form, whose signature is read off the embedded arrays).
        /// </summary>
        public static NDExprProgram Build(NDExpr root, NPTypeCode[] positionalTypes)
        {
            var bind = new NDExprBindContext();
            var bound = root.BindArrays(bind);

            NDArray[] embedded = null;
            NPTypeCode[] inputTypes;
            if (positionalTypes is null)
            {
                if (bind.Operands.Count == 0)
                    throw new ArgumentException(
                        "expression references no arrays — embed NDArrays in the tree " +
                        "(NDExpr.Arr / implicit conversion) or use the (expr, operands) overload with NDExpr.Input(i) leaves.",
                        "expr");
                embedded = bind.Operands.ToArray();
                inputTypes = new NPTypeCode[embedded.Length];
                for (int i = 0; i < embedded.Length; i++)
                    inputTypes[i] = embedded[i].typecode;
            }
            else
            {
                if (bind.Operands.Count != 0)
                    throw new ArgumentException(
                        "expression mixes embedded array leaves with a positional operand list — use one binding style.",
                        "expr");
                inputTypes = positionalTypes;
            }

            bool forcedScalar = NDExpr.ForceScalar;

            if (bound is ReduceNode reduce)
            {
                if (reduce.Child.ContainsReduce)
                    throw new NotSupportedException(
                        "nested reductions are not supported — a reduction must be the root of the expression.");

                var resolved = reduce.ResolveNumPyTypes(inputTypes, out _);
                var acc = ReduceNode.ResolveAccType(reduce.Kind, resolved);
                return new NDExprProgram(bound, embedded, inputTypes, forcedScalar, null, resolved, reduce, acc);
            }

            if (bound.ContainsReduce)
                throw new NotSupportedException(
                    "reduction nodes must be the root of the expression — " +
                    "elementwise use of a reduced value needs two np.evaluate calls.");

            var kernel = bound.CompileNumPy(inputTypes, out var resultType);
            return new NDExprProgram(bound, embedded, inputTypes, forcedScalar, kernel, resultType, null, NPTypeCode.Empty);
        }

        /// <summary>NumPy-style rendering of a dtype signature: <c>(float64, int32)</c>.</summary>
        public static string SignatureText(IReadOnlyList<NPTypeCode> types)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < types.Count; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(types[i].AsNumpyDtypeName());
            }

            return sb.Append(')').ToString();
        }
    }

    public abstract partial class NDExpr
    {
        private NDExprProgram _program;

        /// <summary>
        /// The compiled program for this root: the cached one when it still applies (same binding
        /// form, same operand dtypes, same scalar/vector mode), else a fresh build that replaces it.
        /// </summary>
        internal NDExprProgram GetProgram(NDArray[] positional)
        {
            var p = _program;
            if (p is not null && p.ForcedScalar == ForceScalar)
            {
                bool hit = positional is null
                    ? p.EmbeddedOperands is not null
                    : p.EmbeddedOperands is null && p.Matches(positional);
                if (hit)
                    return p;
            }

            p = NDExprProgram.Build(this, positional);
            _program = p;
            return p;
        }

        /// <summary>
        ///     Compile this tree ahead of its first evaluation (NumSharp extension; numexpr's
        ///     <c>NumExpr(expr)</c> object). The tree must reference its arrays directly
        ///     (<see cref="Arr"/> / implicit conversion): binding, the NumPy typing pass and the
        ///     kernel JIT (~1 ms for a new tree shape) run here, and every
        ///     <see cref="CompiledExpression.Evaluate(NDArray)"/> afterwards does only the per-call
        ///     work — the iteration shape, the result allocation and the iterator (~0.4 µs at n = 8).
        ///     <c>np.evaluate(expr)</c> reaches the same cached program implicitly on its second call
        ///     with the same tree instance; the handle makes the compile step explicit and hoistable.
        ///     A positional tree (<see cref="Input"/> leaves) compiles with
        ///     <see cref="Compile(NPTypeCode[])"/>, which pins its operand dtypes.
        /// </summary>
        /// <exception cref="InvalidOperationException">The tree has no array leaves.</exception>
        public CompiledExpression Compile()
        {
            if (FirstArray() is null)
                throw new InvalidOperationException(
                    "expression references no arrays — a positional tree (NDExpr.Input leaves) is compiled " +
                    "with Compile(params NPTypeCode[] inputTypes), which pins the operand dtypes.");

            return new CompiledExpression(this, GetProgram(null));
        }

        /// <summary>
        ///     Compile a positional tree (<see cref="Input"/> leaves) for one operand dtype signature —
        ///     <c>inputTypes[i]</c> is the dtype <c>Input(i)</c> will be evaluated against. The handle's
        ///     <see cref="CompiledExpression.Evaluate(NDArray[], NDArray)"/> requires exactly that signature
        ///     (a different one is a <see cref="TypeError"/>, never a silent recompile), so a hot loop
        ///     pays no per-call typing. Not the Tier-3C <see cref="Compile(NPTypeCode[], NPTypeCode, string)"/>,
        ///     which emits a raw inner-loop kernel computed entirely at ONE output dtype; this one types
        ///     every node with NumPy's result_type, exactly as <c>np.evaluate</c> does.
        /// </summary>
        public CompiledExpression Compile(params NPTypeCode[] inputTypes)
        {
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));
            if (inputTypes.Length == 0)
                throw new ArgumentException("inputTypes must name at least one operand dtype.", nameof(inputTypes));
            for (int i = 0; i < inputTypes.Length; i++)
                if (inputTypes[i] == NPTypeCode.Empty)
                    throw new ArgumentException($"inputTypes[{i}] is not a storage dtype.", nameof(inputTypes));

            var types = (NPTypeCode[])inputTypes.Clone();
            var program = NDExprProgram.Build(this, types);
            _program = program;
            return new CompiledExpression(this, program);
        }
    }

    /// <summary>
    ///     An <see cref="NDExpr"/> tree compiled ahead of evaluation — the handle
    ///     <see cref="NDExpr.Compile()"/> / <see cref="NDExpr.Compile(NPTypeCode[])"/> return. Holds the
    ///     bound tree, the resolved dtypes and the fused kernel; <see cref="Evaluate(NDArray)"/> runs the
    ///     per-call work only. Immutable and safe to share across threads (each evaluation builds its own
    ///     iterator). The arrays an embedded-form handle references are the caller's: it neither owns nor
    ///     disposes them, and it reads their CURRENT contents on every evaluation.
    /// </summary>
    [NDBorrowed] // the operands are the caller's arrays, referenced by the tree itself
    public sealed class CompiledExpression
    {
        private readonly NDExpr _root;
        private readonly NDExprProgram _program;

        internal CompiledExpression(NDExpr root, NDExprProgram program)
        {
            _root = root;
            _program = program;
        }

        /// <summary>The tree this handle was compiled from.</summary>
        public NDExpr Expression => _root;

        /// <summary>True for a positional tree (<see cref="NDExpr.Input"/> leaves, operands supplied per call).</summary>
        public bool IsPositional => _program.EmbeddedOperands is null;

        /// <summary>True when the root is a reduction (Sum / Prod / Min / Max / Mean).</summary>
        public bool IsReduction => _program.Reduce is not null;

        /// <summary>Number of distinct operands the kernel reads (embedded arrays are deduplicated by reference).</summary>
        public int OperandCount => _program.InputTypes.Length;

        /// <summary>The operand dtype signature this handle is compiled for, in operand order.</summary>
        public NPTypeCode[] InputTypes => (NPTypeCode[])_program.InputTypes.Clone();

        /// <summary>The NumPy result dtype of an evaluation (the elementwise output, or the reduction's result).</summary>
        public NPTypeCode ResultType => _program.ResultType;

        /// <summary>
        ///     Evaluate an embedded-array handle over the arrays the tree references (their current
        ///     contents). Same contract as <c>np.evaluate(expr, out)</c>: <paramref name="out"/> joins
        ///     the broadcast but is never stretched, takes a same_kind cast from
        ///     <see cref="ResultType"/>, and may alias an input.
        /// </summary>
        public NDArray Evaluate(NDArray @out = null)
        {
            if (IsPositional)
                throw new InvalidOperationException(
                    $"this expression was compiled for {OperandCount} positional operand(s) — pass them: Evaluate(operands, out).");

            var ops = _program.EmbeddedOperands;
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            return ops[0].TensorEngine.Evaluate(_program, ops, @out);
        }

        /// <summary>
        ///     Evaluate a positional handle against <paramref name="operands"/> (<c>Input(i)</c> reads
        ///     <c>operands[i]</c>), which must carry exactly the dtype signature the handle was compiled
        ///     for (<see cref="InputTypes"/>); otherwise a <see cref="TypeError"/>. Shapes broadcast per
        ///     call like a ufunc's.
        /// </summary>
        public NDArray Evaluate(NDArray[] operands, NDArray @out = null)
        {
            if (operands is null) throw new ArgumentNullException(nameof(operands));
            if (!IsPositional)
                throw new InvalidOperationException(
                    "this expression was compiled over embedded arrays — evaluate it with Evaluate(out).");
            foreach (var op in operands)
                if (op is null)
                    throw new ArgumentNullException(nameof(operands), "no operand may be null.");
            if (!_program.Matches(operands))
            {
                var got = new NPTypeCode[operands.Length];
                for (int i = 0; i < operands.Length; i++)
                    got[i] = operands[i].typecode;
                throw new TypeError(
                    $"compiled expression expects operand dtypes {NDExprProgram.SignatureText(_program.InputTypes)} " +
                    $"but was called with {NDExprProgram.SignatureText(got)} — compile a handle per signature " +
                    "(NDExpr.Compile(params NPTypeCode[])) or use np.evaluate, which recompiles per signature.");
            }

            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            return operands[0].TensorEngine.Evaluate(_program, operands, @out);
        }
    }
}
