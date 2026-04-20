using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NpyIter.Execution.Custom.cs — Tier 3A / 3B / 3C entry points for user-defined
// inner-loop kernels. All three routes funnel into the same
// NpyIterRef.ForEach(NpyInnerLoopFunc, aux) driver; only kernel creation
// differs.
//
//   Tier 3A (ExecuteRawIL)        — caller emits the entire IL body
//   Tier 3B (ExecuteElementWise)  — caller emits per-element scalar + vector
//                                  bodies; the factory wraps them in the
//                                  4×-unrolled SIMD + scalar-strided shell
//   Tier 3C (ExecuteExpression)   — caller composes an NpyExpr tree which is
//                                  compiled to a Tier-3B kernel
//
// All entry points validate that the iterator's NOp matches the operand type
// array length so common mistakes fail fast.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    internal unsafe ref partial struct NpyIterRef
    {
        // =====================================================================
        // Tier 3A — Raw IL escape hatch
        // =====================================================================

        /// <summary>
        /// Compile and run a user-authored inner-loop kernel. The delegate
        /// signature is <see cref="NpyInnerLoopFunc"/>; the body must emit
        /// its own <c>ret</c>. Cached by <paramref name="cacheKey"/>, so the
        /// IL generator is invoked exactly once per key.
        /// </summary>
        /// <remarks>
        /// The caller is responsible for cacheKey uniqueness: two different
        /// IL bodies compiled under the same key will silently alias.
        /// </remarks>
        public void ExecuteRawIL(Action<ILGenerator> emitBody, string cacheKey, void* auxdata = null)
        {
            if (emitBody is null) throw new ArgumentNullException(nameof(emitBody));
            var kernel = ILKernelGenerator.CompileRawInnerLoop(emitBody, cacheKey);
            ForEach(kernel, auxdata);
        }

        // =====================================================================
        // Tier 3B — Templated inner loop
        // =====================================================================

        /// <summary>
        /// Compile and run an element-wise kernel using user-supplied scalar
        /// and optional vector emit bodies. The factory wraps the bodies in
        /// a 4×-unrolled SIMD loop (when the operand types allow) plus a
        /// scalar-strided fallback for non-contiguous inner axes.
        /// </summary>
        /// <param name="operandTypes">
        /// [input0, input1, ..., output] — one entry per iterator operand.
        /// Length must equal <see cref="NOp"/>.
        /// </param>
        /// <param name="scalarBody">
        /// Per-element IL body. On entry, stack holds the N input values
        /// (operand 0 deepest, operand N-1 on top). On exit, stack must hold
        /// exactly one value of the output dtype.
        /// </param>
        /// <param name="vectorBody">
        /// Per-vector IL body (optional). When supplied AND all operand
        /// dtypes are identical AND SIMD-capable, emitted as the fast path.
        /// Stack contract mirrors <paramref name="scalarBody"/> but with
        /// <c>Vector{W}&lt;T&gt;</c> in place of scalar values.
        /// </param>
        /// <param name="cacheKey">Unique identifier for this kernel.</param>
        public void ExecuteElementWise(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            if (operandTypes is null) throw new ArgumentNullException(nameof(operandTypes));
            if (operandTypes.Length != _state->NOp)
                throw new ArgumentException(
                    $"operandTypes length ({operandTypes.Length}) must match iterator NOp ({_state->NOp}).",
                    nameof(operandTypes));

            var kernel = ILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, cacheKey);
            ForEach(kernel);
        }

        /// <summary>Convenience: 1-input + 1-output (unary).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ExecuteElementWiseUnary(
            NPTypeCode inType, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
            => ExecuteElementWise(new[] { inType, outType }, scalarBody, vectorBody, cacheKey);

        /// <summary>Convenience: 2-input + 1-output (binary).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ExecuteElementWiseBinary(
            NPTypeCode lhs, NPTypeCode rhs, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
            => ExecuteElementWise(new[] { lhs, rhs, outType }, scalarBody, vectorBody, cacheKey);

        /// <summary>Convenience: 3-input + 1-output (ternary, FMA-shaped).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ExecuteElementWiseTernary(
            NPTypeCode a, NPTypeCode b, NPTypeCode c, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
            => ExecuteElementWise(new[] { a, b, c, outType }, scalarBody, vectorBody, cacheKey);

        // =====================================================================
        // Tier 3C — Expression DSL
        // =====================================================================

        /// <summary>
        /// Compile and run an expression tree over the iterator's operands.
        /// The tree's leaves reference inputs by position (NpyExpr.Input(i))
        /// and constants; interior nodes combine them via primitive ops. The
        /// compiler produces the same style of kernel as
        /// <see cref="ExecuteElementWise(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/>.
        /// </summary>
        /// <param name="expression">Root of the expression tree.</param>
        /// <param name="inputTypes">
        /// Dtypes of the first N operands (all inputs). Length must equal
        /// <see cref="NOp"/> - 1.
        /// </param>
        /// <param name="outputType">Dtype of the last operand (the output).</param>
        /// <param name="cacheKey">
        /// Optional cache key; if null, a key is derived from the tree's
        /// structural signature.
        /// </param>
        public void ExecuteExpression(
            NpyExpr expression,
            NPTypeCode[] inputTypes,
            NPTypeCode outputType,
            string? cacheKey = null)
        {
            if (expression is null) throw new ArgumentNullException(nameof(expression));
            if (inputTypes is null) throw new ArgumentNullException(nameof(inputTypes));
            if (inputTypes.Length + 1 != _state->NOp)
                throw new ArgumentException(
                    $"inputTypes length ({inputTypes.Length}) + 1 must equal iterator NOp ({_state->NOp}).",
                    nameof(inputTypes));

            var kernel = expression.Compile(inputTypes, outputType, cacheKey);
            ForEach(kernel);
        }
    }
}
