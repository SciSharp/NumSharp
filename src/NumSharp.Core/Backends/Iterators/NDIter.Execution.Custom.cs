using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDIter.Execution.Custom.cs — Tier 3A / 3B / 3C entry points for user-defined
// inner-loop kernels. All three routes funnel into the same
// NDIterRef.ForEach(NDInnerLoopFunc, aux) driver; only kernel creation
// differs.
//
//   Tier 3A (ExecuteRawIL)        — caller emits the entire IL body
//   Tier 3B (ExecuteElementWise)  — caller emits per-element scalar + vector
//                                  bodies; the factory wraps them in the
//                                  4×-unrolled SIMD + scalar-strided shell
//   Tier 3C (ExecuteExpression)   — caller composes an NDExpr tree which is
//                                  compiled to a Tier-3B kernel
//
// All entry points validate that the iterator's NOp matches the operand type
// array length so common mistakes fail fast.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public unsafe ref partial struct NDIterRef
    {
        // =====================================================================
        // Tier 3A — Raw IL escape hatch
        // =====================================================================

        /// <summary>
        /// Compile and run a user-authored inner-loop kernel. The delegate
        /// signature is <see cref="NDInnerLoopFunc"/>; the body must emit
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
            var kernel = DirectILKernelGenerator.CompileRawInnerLoop(emitBody, cacheKey);
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

            // A trailing ARRAYMASK operand (NumPy ufunc where= convention:
            // op[nop] = wheremask) is driven by ForEach's masked inner loop,
            // not by the kernel — the kernel compiles over the data operands
            // only, so operandTypes excludes the mask slot.
            int kernelNOp = _state->NOp;
            if (_state->MaskOp == kernelNOp - 1 && operandTypes.Length == kernelNOp - 1)
                kernelNOp--;

            if (operandTypes.Length != kernelNOp)
                throw new ArgumentException(
                    $"operandTypes length ({operandTypes.Length}) must match iterator NOp ({_state->NOp}).",
                    nameof(operandTypes));

            var kernel = DirectILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, cacheKey);
            ForEach(kernel);
        }

        /// <summary>
        /// Packed-key form of <see cref="ExecuteElementWise(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/>
        /// for the production ufunc routes: a cache hit runs the kernel without building a
        /// string or touching <paramref name="operandTypes"/> beyond its length; a miss
        /// compiles under the equivalent string key (see <see cref="InnerLoopKernelKey"/>).
        /// </summary>
        public void ExecuteElementWise(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            if (operandTypes is null) throw new ArgumentNullException(nameof(operandTypes));

            ValidateElementWiseOperandCount(operandTypes.Length);

            if (vectorBody != null && Is2DElementwiseShape() &&
                TryExecute2DElementwise(operandTypes, scalarBody, vectorBody, key))
                return;

            if (!DirectILKernelGenerator.TryGetInnerLoop(key, out var kernel))
                kernel = DirectILKernelGenerator.CompileInnerLoop(operandTypes, scalarBody, vectorBody, key);
            ForEach(kernel);
        }

        /// <summary>
        /// The operand-count contract of <see cref="ExecuteElementWise(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/>
        /// (a trailing ARRAYMASK operand is driven by ForEach, not the kernel), shared by the
        /// packed-key forms so a hit never allocates the dtype array just to count it.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ValidateElementWiseOperandCount(int operandTypeCount)
        {
            int kernelNOp = _state->NOp;
            if (_state->MaskOp == kernelNOp - 1 && operandTypeCount == kernelNOp - 1)
                kernelNOp--;

            if (operandTypeCount != kernelNOp)
                throw new ArgumentException(
                    $"operandTypes length ({operandTypeCount}) must match iterator NOp ({_state->NOp}).",
                    "operandTypes");
        }

        // =====================================================================
        // 2-D coalesced-block fast path (narrow strided rows)
        // =====================================================================

        /// <summary>
        /// Type-independent gate for the <see cref="ND2DElementwiseKernel"/> route: the iteration
        /// is a CONTIGUOUS inner axis under one or more outer axes that flatten to a single strided
        /// run (a <c>m[:, :w]</c> column slice, a broadcast-row operand, a trailing-narrow N-D view
        /// <c>x[..., :w]</c>, …). The ordinary per-chunk route drives these one row at a time under
        /// EXTERNAL_LOOP, paying the odometer advance AND the kernel's own SIMD-viability prologue
        /// PER ROW; the block kernel loops the (flattened) outer axis itself, prologue once —
        /// ~1.7-2.1x NumPy on narrow rows vs the ~0.8x the per-row route gets
        /// (docs/NDITER_PERF_DISCOVERY.md §7 angle 2).
        ///
        /// The outer axes must be MUTUALLY contiguous for every operand
        /// (<c>stride[d] == stride[d+1] * shape[d+1]</c>) so they collapse to one
        /// <c>(outerCount, outerStride)</c> — true for any trailing-narrow slice, since such a view
        /// leaves the outer axes exactly as contiguous as the source. NumSharp's NDIter does not
        /// coalesce these axes itself, so a 3-D <c>x[:, :, :w]</c> stays NDim=3 here and this flatten
        /// is what lets it reach the block kernel. Cheap and allocation-free: the common
        /// contiguous/1-D case fails the <c>NDim &gt;= 2</c> / inner-contiguous test immediately, so
        /// the callers only build an operand-type array when this returns true.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool Is2DElementwiseShape()
        {
            uint f = _state->ItFlags;
            if ((f & (uint)NDIterFlags.BUFFER) != 0) return false;        // cast/promote → windowed path
            if ((f & (uint)NDIterFlags.EXLOOP) == 0) return false;        // need the coalesced inner axis
            if ((f & (uint)NDIterFlags.ONEITERATION) != 0) return false;  // single chunk is already optimal
            int ndim = _state->NDim;
            if (ndim < 2) return false;                                   // 1-D is the single-chunk contig path
            if (_state->MaskOp >= 0) return false;                        // where= → ForEach masked driver

            int inner = ndim - 1;
            long* shape = _state->Shape;
            int nop = _state->NOp;
            for (int op = 0; op < nop; op++)
            {
                // The inner (last) axis must be element-contiguous — that is what lets the block
                // kernel SIMD each row in place. A strided/broadcast inner axis keeps the per-chunk
                // route (which honors arbitrary inner strides).
                if (_state->GetStride(inner, op) != 1) return false;
                // Outer axes must be mutually contiguous so they flatten to one (count, stride).
                for (int d = 0; d < inner - 1; d++)
                    if (_state->GetStride(d, op) != _state->GetStride(d + 1, op) * shape[d + 1])
                        return false;
            }
            return true;
        }

        /// <summary>
        /// Compile (or fetch) and run the 2-D block kernel for a shape
        /// <see cref="Is2DElementwiseShape"/> has already approved. Returns false — having
        /// touched nothing — when the operands are not all the same SIMD-capable dtype (mixed
        /// dtypes, Decimal/Half/Complex, comparison→bool), so the caller falls back to the
        /// per-chunk <see cref="ForEach"/>. The <paramref name="scalarBody"/>/<paramref name="vectorBody"/>
        /// are the SAME emit delegates the per-chunk kernel uses, so the two routes are
        /// byte-identical (element-wise ops carry no cross-element state, so looping the outer
        /// axis inside the kernel cannot change any result).
        /// </summary>
        private bool TryExecute2DElementwise(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            if (vectorBody is null) return false;
            if (!DirectILKernelGenerator.Enabled) return false;

            int nop = _state->NOp;
            if (operandTypes.Length != nop) return false;                 // defensive (mask already excluded)
            if (!DirectILKernelGenerator.CanSimdAllOperands(operandTypes)) return false;

            int ndim = _state->NDim;
            int inner = ndim - 1;
            long* shape = _state->Shape;

            long innerCount = shape[inner];
            long outerCount = 1;
            for (int d = 0; d < inner; d++)
                outerCount *= shape[d];

            // The mutually-contiguous outer axes collapse to a single stride: the innermost outer
            // axis's (axis inner-1) stride reproduces the whole odometer's C-order walk. Data
            // pointers traverse SOURCE-array memory, so scale by SrcElementSizes exactly as
            // NDIter.ExternalLoopNext does (identical to ElementSizes on this unbuffered path). A
            // broadcast operand's stride is 0.
            long* outerStrides = stackalloc long[nop];
            for (int op = 0; op < nop; op++)
                outerStrides[op] = _state->GetStride(inner - 1, op) * _state->SrcElementSizes[op];

            if (!DirectILKernelGenerator.TryGet2DKernel(key, out var kernel))
                kernel = DirectILKernelGenerator.Compile2DElementwiseKernel(operandTypes, scalarBody, vectorBody, key);

            kernel(GetDataPtrArray(), innerCount, outerStrides, outerCount);
            return true;
        }

        /// <summary>Convenience: 1-input + 1-output (unary).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void ExecuteElementWiseUnary(
            NPTypeCode inType, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
            => ExecuteElementWise(new[] { inType, outType }, scalarBody, vectorBody, cacheKey);

        /// <summary>Packed-key unary form: a cache hit allocates nothing.</summary>
        public void ExecuteElementWiseUnary(
            NPTypeCode inType, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            ValidateElementWiseOperandCount(2);
            if (vectorBody != null && Is2DElementwiseShape() &&
                TryExecute2DElementwise(new[] { inType, outType }, scalarBody, vectorBody, key))
                return;
            if (!DirectILKernelGenerator.TryGetInnerLoop(key, out var kernel))
                kernel = DirectILKernelGenerator.CompileInnerLoop(new[] { inType, outType }, scalarBody, vectorBody, key);
            ForEach(kernel);
        }

        /// <summary>Packed-key binary form: a cache hit allocates nothing.</summary>
        public void ExecuteElementWiseBinary(
            NPTypeCode lhs, NPTypeCode rhs, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            ValidateElementWiseOperandCount(3);
            if (vectorBody != null && Is2DElementwiseShape() &&
                TryExecute2DElementwise(new[] { lhs, rhs, outType }, scalarBody, vectorBody, key))
                return;
            if (!DirectILKernelGenerator.TryGetInnerLoop(key, out var kernel))
                kernel = DirectILKernelGenerator.CompileInnerLoop(new[] { lhs, rhs, outType }, scalarBody, vectorBody, key);
            ForEach(kernel);
        }

        /// <summary>Convenience: 2-input + 1-output (binary).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void ExecuteElementWiseBinary(
            NPTypeCode lhs, NPTypeCode rhs, NPTypeCode outType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
            => ExecuteElementWise(new[] { lhs, rhs, outType }, scalarBody, vectorBody, cacheKey);

        /// <summary>Convenience: 3-input + 1-output (ternary, FMA-shaped).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        /// The tree's leaves reference inputs by position (NDExpr.Input(i))
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
            NDExpr expression,
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

            // EXTERNAL_LOOP guard (the measured ~40× foot-gun): without EXLOOP
            // the driver advances one element at a time and the per-chunk
            // kernel runs with count==1 — silently correct, catastrophically
            // slow. A single-chunk iteration (ONEITERATION) is exempt: the
            // kernel gets the whole range in one call either way.
            bool exloop = (_state->ItFlags & (uint)NDIterFlags.EXLOOP) != 0;
            bool oneiter = (_state->ItFlags & (uint)NDIterFlags.ONEITERATION) != 0;
            if (!exloop && !oneiter && _state->IterSize > 1)
                throw new InvalidOperationException(
                    "ExecuteExpression requires an iterator constructed with NDIterGlobalFlags.EXTERNAL_LOOP — " +
                    "without it the compiled kernel is invoked once per element (~40× slower). " +
                    "Add EXTERNAL_LOOP to the construction flags (np.evaluate configures this automatically).");

            var kernel = expression.Compile(inputTypes, outputType, cacheKey);
            ForEach(kernel);
        }
    }
}
