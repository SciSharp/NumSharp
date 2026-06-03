using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Binary operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a binary operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Operation to perform</param>
        /// <returns>Result array with promoted type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray ExecuteBinaryOp(NDArray lhs, NDArray rhs, BinaryOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Determine result type using NumPy type promotion rules
            var resultType = np._FindCommonType(lhs, rhs);

            // NumPy bool arithmetic: the bool dtype has no integer add/multiply ufunc loop — `+`
            // is logical OR and `*` is logical AND (so True + True == True, raw byte 1, not 2).
            // resultType is Boolean only when both operands are bool; remap the op so every
            // downstream kernel path (SIMD/scalar, same-type/mixed) emits the bitwise op and writes
            // a normalized 0/1 byte. (`-` has no bool loop and already throws like NumPy.)
            if (resultType == NPTypeCode.Boolean)
            {
                if (op == BinaryOp.Add) op = BinaryOp.BitwiseOr;
                else if (op == BinaryOp.Multiply) op = BinaryOp.BitwiseAnd;
            }

            // NumPy: true division (/) always returns float64 for integer types
            // This matches Python 3 / NumPy 2.x semantics where / is "true division"
            // Group 3 = float (Single, Double), Group 4 = Decimal
            if (op == BinaryOp.Divide && resultType.GetGroup() < 3)
            {
                resultType = NPTypeCode.Double;
            }

            // NumPy Power promotion (NEP50). Most cases are already handled by
            // _FindCommonType (which applies weak/strict scalar rules correctly):
            //   - i32_arr ** i32_arr  → int32
            //   - i32_arr ** i64_arr  → int64
            //   - f32_arr ** f64_arr  → float64
            //   - f32_arr ** i32_arr  → float64 (NEP50 strict)
            //   - f32_arr ** Python int (0-D weak) → float32 (NEP50 weak)
            //   - i32_arr ** f32_scalar (cross-array)  → float64 (handled below)
            //
            // The one rule _FindCommonType doesn't cover is `int_scalar ** float_arr`:
            // a 0-D int scalar is treated as "weak", which preserves the float's dtype,
            // but NumPy's int-base + float-exp rule promotes unconditionally to float64.
            // (Group 0=Byte/Char, 1=signed int, 2=unsigned int, 3=float, 4=decimal)
            //
            // Known limitation: explicit 0-D integer arrays (`np.array(2, int32)`)
            // are indistinguishable from C# `int 2` after `np.asanyarray`. NumPy would
            // strict-promote `f32_arr ** np.int32(2)` to float64; NumSharp preserves
            // float32 for both `f32_arr ** 2` (correct) and `f32_arr ** np.array(2, int32)`
            // (misaligned with NumPy but rare in idiomatic C# code).
            if (op == BinaryOp.Power)
            {
                var lhsGroup = lhsType.GetGroup();
                var rhsGroup = rhsType.GetGroup();

                if (lhsGroup <= 2 && rhsGroup == 3)
                {
                    resultType = NPTypeCode.Double;
                }
            }

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteScalarScalar(lhs, rhs, op, resultType);
            }

            // -------- O(1) trivial-loop bypass -----------------------------
            // NumSharp analogue of NumPy's check_for_trivial_loop +
            // try_trivial_single_output_loop (ufunc_object.c), which handle a
            // single strided inner loop "without using the (heavy) iterator."
            // When both operands share ONE contiguous layout (both C, or both F),
            // have identical shape (no broadcast) and the same dtype as the result
            // (no cast), a single linear walk over all three buffers visits the
            // same logical element — so we route straight to the existing DirectIL
            // SimdFull whole-array kernel and skip the NpyIter MultiNew/Initialize
            // construction (measured ~600-2000 ns/call: 22-24% of a small
            // contiguous op, <=3% once n>=64K). Returns null for anything that is
            // not trivially contiguous (broadcast/mixed-C-F/strided/cast/scalar-
            // broadcast/unsupported emit) → falls through to the NpyIter route
            // below with behaviour unchanged.
            {
                var trivial = TryTrivialContiguousBinaryOp(lhs, rhs, op, lhsType, rhsType, resultType);
                if (trivial is not null) return trivial;
            }

            // -------- NpyIter Tier 3B fast path (all binary ops) -----------
            // Routes through the NpyIter inner-loop kernel factory, which
            // collapses coalesce + SIMD dispatch (contig, SimdScalarLeft,
            // SimdScalarRight, scalar-strided) into a single emitted kernel
            // driven by NpyIter's multi-operand iterator.
            //
            // Same-dtype: full SIMD path (CanSimdAllOperands passes inside
            // the factory). Measured 2.3-4.7× wins across 12 variations,
            // at parity with NumPy 2.x.
            //
            // Mixed-dtype: scalar body emits per-operand EmitConvertTo
            // before EmitScalarOperation, mirroring the direct path's
            // EmitConvertTo + EmitScalarOperation sequence. Vector body
            // is null (factory drops to scalar-strided). Equivalent perf
            // for the mixed-dtype cases the direct path used to handle.
            {
                var routed = TryExecuteBinaryOpViaNpyIter(lhs, rhs, op, lhsType, rhsType, resultType);
                if (routed is not null) return routed;
            }

            // Broadcast shapes
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // NumPy-aligned layout preservation: when EVERY non-scalar operand is strictly
            // F-contig, allocate the result in F-order up front and skip the post-kernel
            // copy. Pre-L3-a this branch ran with C-allocated result + `result.copy('F')`
            // at the end. Allocating F here saves the copy AND lets the L3-a coalesce
            // collapse to 1-D SimdFull (≈15× speedup for the 1K×1K F-contig case).
            //
            // The stricter "all-F" rule is required for kernel correctness: kernels still
            // walk the result buffer with linear `i*elemSize` indexing (C-order coords).
            // If result is F-contig but any input operand is neither C nor F (e.g. negative
            // strides, partial broadcast), the kernel writes positions that don't match
            // the input's logical coords. The legacy `result.copy('F')` path stays for
            // the looser "any F, no strict C" case via the post-kernel branch below.
            bool allStrictFContig = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            Shape resultShape = allStrictFContig
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            // Allocate result
            var result = new NDArray(resultType, resultShape, false);

            // Empty broadcast result: no elements to compute. The kernels below assume
            // >= 1 element and walk stride-0 broadcast dims as if non-empty, corrupting
            // memory when a sibling dim is 0 (e.g. (3,1,1) op (1,0,2) -> (3,0,2)).
            // (The NpyIter fast path above returns early for the same reason.)
            if (result.size == 0)
                return result;

            // L3-a: pre-coalesce adjacent dims with compatible strides for BOTH operands
            // (and the result). This collapses F-contig N-D to 1-D contig, so the path
            // classifier promotes from `General` (≈13× slower) to `SimdFull`. Broadcast
            // and arbitrary strided cases survive unchanged because their cross-axis
            // stride relationships don't satisfy the merge condition.
            int origNdim = resultShape.NDim;
            long* coalShape = stackalloc long[origNdim > 0 ? origNdim : 1];
            long* coalLhsStr = stackalloc long[origNdim > 0 ? origNdim : 1];
            long* coalRhsStr = stackalloc long[origNdim > 0 ? origNdim : 1];
            long* coalResStr = stackalloc long[origNdim > 0 ? origNdim : 1];
            for (int d = 0; d < origNdim; d++)
            {
                coalShape[d] = resultShape.dimensions[d];
                coalLhsStr[d] = leftShape.strides[d];
                coalRhsStr[d] = rightShape.strides[d];
                coalResStr[d] = resultShape.strides[d];
            }
            int coalNdim = CoalesceTernaryDimensions(coalShape, coalLhsStr, coalRhsStr, coalResStr, origNdim);

            // Classify execution path using coalesced strides
            ExecutionPath path = ClassifyPath(coalLhsStr, coalRhsStr, coalShape, coalNdim, resultType);

            // Get kernel key
            var key = new MixedTypeKernelKey(lhsType, rhsType, resultType, op, path);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.GetMixedTypeKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel using coalesced shape/strides
                ExecuteKernelCoalesced(kernel, lhs, rhs, result, leftShape, rightShape,
                    coalShape, coalLhsStr, coalRhsStr, coalNdim);
            }
            else
            {
                // Fallback to legacy implementation
                FallbackBinaryOp(lhs, rhs, result, op, leftShape, rightShape);
            }

            // NumPy F-output preservation for the LOOSER case (at least one strict-F operand
            // but not all): result is currently C-contig (correct kernel output). Copy to F
            // to match NumPy. The strict-all-F case skipped this branch by allocating F up
            // front and the equality below short-circuits.
            if (!allStrictFContig && ShouldProduceFContigOutput(lhs, rhs, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        ///     Try to execute a binary op via NpyIter Tier 3B for any dtype
        ///     combination (same or mixed). Returns the result array on
        ///     success, or null if the route is not applicable (broadcast
        ///     result exceeds int.MaxValue, NpyIter not built for long-
        ///     shape arithmetic; unsupported op/dtype emit).
        ///
        ///     Allocates the output as F-contig when both inputs are strictly
        ///     F (matches the pre-existing direct-path rule from L3-b) and
        ///     picks the NpyIter order accordingly: NPY_FORTRANORDER for
        ///     strict-F-both, NPY_CORDER everywhere else. NPY_CORDER also
        ///     handles the reversed-stride case correctly because NpyIter
        ///     normalizes negative inner strides during init.
        ///
        ///     Same-dtype path (<paramref name="lhsType"/> == <paramref name="rhsType"/>
        ///     == <paramref name="resultType"/>): scalar body is
        ///     <see cref="DirectILKernelGenerator.EmitScalarOperation"/>; vector
        ///     body is supplied when the dtype and op both support SIMD.
        ///
        ///     Mixed-dtype path: scalar body emits a load-shuffle that
        ///     converts each input from its source dtype to the result
        ///     dtype before invoking EmitScalarOperation — mirrors the
        ///     direct path's EmitConvertTo + EmitScalarOperation sequence
        ///     in EmitGeneralLoop / EmitChunkLoop. Vector body is null
        ///     because Tier 3B's <c>CanSimdAllOperands</c> rejects mixed
        ///     dtypes; factory drops straight to the scalar-strided loop.
        ///
        ///     After the kernel runs, applies the "looser-F" post-copy step
        ///     that the direct path uses: if the result is C-contig but the
        ///     NumPy-aligned rule says it should be F (at least one strict-F
        ///     input, no strict-C input), return <c>result.copy('F')</c>.
        /// </summary>
        private unsafe NDArray? TryExecuteBinaryOpViaNpyIter(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // Broadcast → clean shape so we know what the result looks like.
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // NpyIter's internal shape arithmetic is int-bounded; route only
            // when the broadcast result fits. Pre-existing test
            // LongIndexingBroadcastTest exercises the > int.MaxValue path via
            // the direct allocator (which is also int-limited but doesn't
            // throw on the shape calc itself). Falling through to the direct
            // path keeps the prior behaviour for those edge cases.
            if (cleanShape.size < 0) return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue) return null;

            // Mirror the direct path: F-allocate output when every non-scalar
            // operand is strict-F. Otherwise default to C and let the
            // post-kernel "looser-F" copy step rectify when needed.
            bool allStrictFContig = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            Shape resultShape = allStrictFContig
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            var result = new NDArray(resultType, resultShape, false);

            // Empty broadcast result (a stride-0 broadcast dim alongside a zero-size
            // dim, e.g. (3,1,1) op (1,0,2) -> (3,0,2)): there is nothing to compute.
            // Returning here is REQUIRED — the NpyIter element-wise path corrupts the
            // heap when driven over a 0-element broadcast (the direct kernel path below
            // is guarded the same way). Matches NumPy: empty op -> empty result.
            if (result.size == 0)
                return result;

            // Order selection — see method-summary comment.
            var order = allStrictFContig
                ? NPY_ORDER.NPY_FORTRANORDER
                : NPY_ORDER.NPY_CORDER;

            // SIMD viability: requires equal dtypes (CanSimdAllOperands in
            // the Tier 3B factory enforces this anyway, but we short-circuit
            // here to keep the vector body null when known to be unusable).
            // Op gate: Decimal/Half/Complex go scalar-only (CanUseSimd
            // returns false for them); Mod/Power/FloorDivide/ATan2 go
            // scalar-only via CanUseSimdForOp.
            bool sameDtype = lhsType == rhsType && lhsType == resultType;
            bool simdViable = sameDtype
                              && DirectILKernelGenerator.CanUseSimd(resultType)
                              && DirectILKernelGenerator.CanUseSimdForOp(op);

            // Build per-element scalar emit body. For same-dtype we just call
            // EmitScalarOperation directly. For mixed-dtype we wrap it with
            // a per-operand convert pass (the direct path's
            // EmitGeneralLoop / EmitChunkLoop does the same).
            Action<ILGenerator> scalarBody;
            if (sameDtype)
            {
                scalarBody = il => DirectILKernelGenerator.EmitScalarOperation(il, op, resultType);
            }
            else
            {
                NPTypeCode capLhs = lhsType, capRhs = rhsType, capRes = resultType;
                BinaryOp capOp = op;
                scalarBody = il => EmitMixedScalarBody(il, capLhs, capRhs, capRes, capOp);
            }

            Action<ILGenerator>? vectorBody = simdViable
                ? il => DirectILKernelGenerator.EmitVectorOperation(il, op, resultType)
                : null;

            // Cache key MUST encode all three dtypes; mixed-dtype kernels
            // are distinct from same-dtype ones for the same op.
            string cacheKey = $"npy_binop_{op}_{lhsType}_{rhsType}_{resultType}";

            try
            {
                using var iter = NpyIterRef.MultiNew(
                    3, new[] { lhs, rhs, result },
                    NpyIterGlobalFlags.EXTERNAL_LOOP,
                    order,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    new[]
                    {
                        NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.READONLY,
                        NpyIterPerOpFlags.WRITEONLY,
                    });

                iter.ExecuteElementWiseBinary(lhsType, rhsType, resultType, scalarBody, vectorBody, cacheKey);
            }
            catch (NotSupportedException)
            {
                // EmitScalarOperation / EmitVectorOperation / EmitConvertTo
                // can throw for combos they don't cover. Surface as null so
                // the caller falls back to the direct path.
                return null;
            }

            // Looser-F preservation: matches the post-kernel branch in the
            // direct path. Triggers when the result is currently C-contig but
            // the NumPy rule says it should be F because at least one input
            // is strict-F and no input is strict-C.
            if (!allStrictFContig && ShouldProduceFContigOutput(lhs, rhs, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        ///     O(1)-gated trivial-loop bypass — the NumSharp analogue of NumPy's
        ///     <c>check_for_trivial_loop</c> + <c>try_trivial_single_output_loop</c>
        ///     (ufunc_object.c), which handle a single strided inner loop "without
        ///     using the (heavy) iterator."
        ///
        ///     Fires only when the op needs neither broadcasting nor casting and
        ///     both operands share ONE contiguous layout, so element k of each
        ///     operand's buffer (from its own offset) is the same logical element:
        ///       • dtypes identical (lhs == rhs == result) — no cast;
        ///       • shapes identical (<see cref="Shape.Equals(Shape)"/>, which
        ///         compares size + dimensions and ignores strides) — no broadcast;
        ///       • neither operand broadcasted (stride-0 dim with extent > 1);
        ///       • both C-contiguous (→ C result) or both F-contiguous (→ F result).
        ///     1-D arrays are both C and F; the C branch is tested first (C result
        ///     == F result there), so the F branch implies ndim > 1 strictly-F,
        ///     matching <see cref="AreAllOperandsStrictFContig"/>'s F-alloc rule.
        ///     Contiguous slices (offset != 0) qualify because
        ///     <see cref="ExecuteKernel"/> applies each operand's offset.
        ///
        ///     Routes to the SAME <see cref="ExecutionPath.SimdFull"/> DirectIL
        ///     kernel the post-NpyIter fallback uses, so results are identical for
        ///     every dtype/op (the generator emits a SIMD or scalar loop per dtype).
        ///     Unsupported emits (e.g. bool subtract) throw inside the generator;
        ///     we catch and return null so the existing path raises/handles the
        ///     case exactly as before. Any non-trivial case returns null →
        ///     caller proceeds to the NpyIter route.
        /// </summary>
        private unsafe NDArray? TryTrivialContiguousBinaryOp(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // No cast: all three dtypes identical. Promotion cases (/ -> f64,
            // int-base ** float -> f64, mixed dtypes) have resultType != input and
            // are excluded here, deferring to the NpyIter route that does the cast.
            if (lhsType != rhsType || lhsType != resultType)
                return null;

            var ls = lhs.Shape;
            var rs = rhs.Shape;

            // Scalar-broadcast: exactly one operand is scalar/size-1 and the other
            // is a contiguous array (size > 1). NumPy's trivial loop broadcasts 0-D
            // operands this way. The scalar is read once, the array walked linearly,
            // and the result takes the array operand's shape+layout. (Both size-1, or
            // both size > 1, fall through to the equal-shape branch below.)
            bool lhsScalarLike = ls.IsScalar || ls.size == 1;
            bool rhsScalarLike = rs.IsScalar || rs.size == 1;
            if (lhsScalarLike ^ rhsScalarLike)
            {
                return TryScalarBroadcastBinaryOp(
                    lhs, rhs, op, resultType,
                    array: rhsScalarLike ? lhs : rhs,
                    scalarIsRhs: rhsScalarLike);
            }

            // Identical logical shape (no broadcast). Shape.Equals compares size +
            // dimensions and ignores strides/offset — exactly the "same shape,
            // either layout" test we want.
            if (!ls.Equals(rs))
                return null;

            // A stride-0 dim with extent > 1 breaks the linear-walk assumption even
            // if the contiguity flags happen to look set; exclude explicitly.
            if (ls.IsBroadcasted || rs.IsBroadcasted)
                return null;

            // One shared contiguous layout (C checked first; see summary).
            bool bothC = ls.IsContiguous && rs.IsContiguous;
            bool bothF = !bothC && ls.IsFContiguous && rs.IsFContiguous;
            if (!bothC && !bothF)
                return null;

            // SimdFull kernel: ignores strides, walks result.size linearly. Emit may
            // be unsupported for some op/dtype (e.g. bool '-') — fall through to the
            // existing path so it raises (or handles) the case identically.
            var key = new MixedTypeKernelKey(lhsType, rhsType, resultType, op, ExecutionPath.SimdFull);
            MixedTypeKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetMixedTypeKernel(key);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            if (kernel == null)
                return null;

            var dims = (long[])ls.dimensions.Clone();
            Shape resultShape = bothF ? new Shape(dims, 'F') : new Shape(dims);
            var result = new NDArray(resultType, resultShape, false);

            // Empty result: nothing to compute (the kernel assumes >= 1 element).
            if (result.size == 0)
                return result;

            ExecuteKernel(kernel, lhs, rhs, result, ls, rs);
            return result;
        }

        /// <summary>
        ///     Scalar-broadcast arm of the trivial-loop bypass: <c>array op scalar</c>
        ///     or <c>scalar op array</c> where the array operand is contiguous. Routes
        ///     to the existing <see cref="ExecutionPath.SimdScalarRight"/> /
        ///     <see cref="ExecutionPath.SimdScalarLeft"/> DirectIL kernel (scalar read
        ///     once, array walked linearly), skipping NpyIter construction. The result
        ///     takes the array operand's shape and layout (C, or strictly-F) so the
        ///     linear write aligns with the linear array read. Returns null (→ NpyIter)
        ///     when the array operand is non-contiguous or the emit is unsupported.
        ///
        ///     Callers guarantee identical dtypes (same-dtype gate in
        ///     <see cref="TryTrivialContiguousBinaryOp"/>), so the kernel key uses
        ///     <paramref name="resultType"/> for all three operand slots.
        /// </summary>
        private unsafe NDArray? TryScalarBroadcastBinaryOp(
            NDArray lhs, NDArray rhs, BinaryOp op, NPTypeCode resultType,
            NDArray array, bool scalarIsRhs)
        {
            var arrShape = array.Shape;
            if (arrShape.IsBroadcasted)
                return null;

            bool isC = arrShape.IsContiguous;
            bool isF = !isC && arrShape.IsFContiguous;
            if (!isC && !isF)
                return null;   // strided/transposed array operand → NpyIter

            var path = scalarIsRhs ? ExecutionPath.SimdScalarRight : ExecutionPath.SimdScalarLeft;
            var key = new MixedTypeKernelKey(resultType, resultType, resultType, op, path);
            MixedTypeKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetMixedTypeKernel(key);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            if (kernel == null)
                return null;

            var dims = (long[])arrShape.dimensions.Clone();
            Shape resultShape = isF ? new Shape(dims, 'F') : new Shape(dims);
            var result = new NDArray(resultType, resultShape, false);
            if (result.size == 0)
                return result;

            // lhs/rhs keep their original operand positions; the SimdScalarRight/Left
            // kernel reads the scalar side once and walks the array side linearly.
            ExecuteKernel(kernel, lhs, rhs, result, lhs.Shape, rhs.Shape);
            return result;
        }

        /// <summary>
        ///     Mixed-dtype scalar-body emitter. On entry the stack carries
        ///     <c>[lhs (lhsType), rhs (rhsType)]</c>. On exit it carries one
        ///     value of <paramref name="resultType"/>. Handles all three
        ///     conversion combinations (lhs-only, rhs-only, both) via a
        ///     temp local for the rhs value so we can reach the lhs at the
        ///     bottom of the stack.
        ///
        ///     Same-dtype callers do NOT go through this path — they call
        ///     <see cref="DirectILKernelGenerator.EmitScalarOperation"/> directly,
        ///     skipping the local allocation and reload.
        /// </summary>
        private static void EmitMixedScalarBody(
            ILGenerator il,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType,
            BinaryOp op)
        {
            // Stack: [lhs (lhsType), rhs (rhsType)]
            //
            // Stash rhs into a local so we can convert the bottom-of-stack lhs
            // first, then reload rhs and convert it. Doing it this order keeps
            // the final stack as [lhs (resultType), rhs (resultType)] which
            // is what EmitScalarOperation expects.
            var locRhs = il.DeclareLocal(DirectILKernelGenerator.GetClrType(rhsType));
            il.Emit(OpCodes.Stloc, locRhs);
            if (lhsType != resultType)
                DirectILKernelGenerator.EmitConvertTo(il, lhsType, resultType);
            il.Emit(OpCodes.Ldloc, locRhs);
            if (rhsType != resultType)
                DirectILKernelGenerator.EmitConvertTo(il, rhsType, resultType);

            DirectILKernelGenerator.EmitScalarOperation(il, op, resultType);
        }

        /// <summary>
        /// NumPy-aligned rule: the output is F-contiguous when every non-scalar operand
        /// is strictly F-contiguous (IsFContiguous && !IsContiguous).
        /// Scalars (and 1-element shapes, both C and F) do not change the decision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ShouldProduceFContigOutput(NDArray a, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;
            var s = a.Shape;
            // Scalars and size-1 shapes don't force a preference.
            if (s.IsScalar || s.size <= 1)
                return false;
            return s.IsFContiguous && !s.IsContiguous;
        }

        /// <summary>
        ///     Stricter L3-a rule: every non-scalar operand must be strictly F-contiguous
        ///     (not just one of them). Required for the F-allocated-result optimization
        ///     because the kernel walks the result buffer linearly. If any operand is
        ///     neither C nor F (negative strides, partial broadcast, custom view), its
        ///     linear walk doesn't align with the F-output's linear walk → wrong values.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool AreAllOperandsStrictFContig(NDArray lhs, NDArray rhs, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;

            bool lhsScalar = lhs.Shape.IsScalar || lhs.Shape.size <= 1;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.Shape.size <= 1;

            bool lhsPureF = !lhsScalar && lhs.Shape.IsFContiguous && !lhs.Shape.IsContiguous;
            bool rhsPureF = !rhsScalar && rhs.Shape.IsFContiguous && !rhs.Shape.IsContiguous;

            // Strict-all-F requires every non-scalar operand to be pure-F.
            // The "all scalars" case never reaches here (excluded upstream).
            if (!lhsScalar && !lhsPureF) return false;
            if (!rhsScalar && !rhsPureF) return false;

            // At least one non-scalar op must be pure-F (otherwise both are scalars,
            // which the upstream IsScalar+IsScalar path already short-circuits).
            return lhsPureF || rhsPureF;
        }

        /// <summary>
        /// Binary variant — require that every non-scalar operand is strictly F-contiguous
        /// and at least one of them is (otherwise the scalar+scalar case is excluded upstream).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ShouldProduceFContigOutput(NDArray lhs, NDArray rhs, Shape resultShape)
        {
            if (resultShape.NDim <= 1 || resultShape.size <= 1)
                return false;

            bool lhsScalar = lhs.Shape.IsScalar || lhs.Shape.size <= 1;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.Shape.size <= 1;

            bool lhsPureF = !lhsScalar && lhs.Shape.IsFContiguous && !lhs.Shape.IsContiguous;
            bool rhsPureF = !rhsScalar && rhs.Shape.IsFContiguous && !rhs.Shape.IsContiguous;
            bool lhsPureC = !lhsScalar && lhs.Shape.IsContiguous && !lhs.Shape.IsFContiguous;
            bool rhsPureC = !rhsScalar && rhs.Shape.IsContiguous && !rhs.Shape.IsFContiguous;

            // If any non-scalar operand is strictly C-contig, fall through to the C default.
            if (lhsPureC || rhsPureC)
                return false;

            // At least one non-scalar operand must be strictly F-contig to trigger F output.
            return lhsPureF || rhsPureF;
        }

        /// <summary>
        /// Execute scalar × scalar operation using IL-generated delegate.
        /// </summary>
        private NDArray ExecuteScalarScalar(NDArray lhs, NDArray rhs, BinaryOp op, NPTypeCode resultType)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new BinaryScalarKernelKey(lhsType, rhsType, resultType, op);
            var func = DirectILKernelGenerator.GetBinaryScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarLhs(func, lhs.GetBoolean(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Byte => InvokeBinaryScalarLhs(func, lhs.GetByte(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.SByte => InvokeBinaryScalarLhs(func, lhs.GetSByte(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarLhs(func, lhs.GetInt16(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarLhs(func, lhs.GetUInt16(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarLhs(func, lhs.GetInt32(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarLhs(func, lhs.GetUInt32(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarLhs(func, lhs.GetInt64(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarLhs(func, lhs.GetUInt64(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Char => InvokeBinaryScalarLhs(func, lhs.GetChar(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Half => InvokeBinaryScalarLhs(func, lhs.GetHalf(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Single => InvokeBinaryScalarLhs(func, lhs.GetSingle(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Double => InvokeBinaryScalarLhs(func, lhs.GetDouble(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarLhs(func, lhs.GetDecimal(Array.Empty<long>()), rhs, rhsType, resultType),
                NPTypeCode.Complex => InvokeBinaryScalarLhs(func, lhs.GetComplex(Array.Empty<long>()), rhs, rhsType, resultType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue binary scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeBinaryScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, NDArray rhs, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetBoolean(Array.Empty<long>()), resultType),
                NPTypeCode.Byte => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetByte(Array.Empty<long>()), resultType),
                NPTypeCode.SByte => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetSByte(Array.Empty<long>()), resultType),
                NPTypeCode.Int16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt16(Array.Empty<long>()), resultType),
                NPTypeCode.UInt16 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt16(Array.Empty<long>()), resultType),
                NPTypeCode.Int32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt32(Array.Empty<long>()), resultType),
                NPTypeCode.UInt32 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt32(Array.Empty<long>()), resultType),
                NPTypeCode.Int64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetInt64(Array.Empty<long>()), resultType),
                NPTypeCode.UInt64 => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetUInt64(Array.Empty<long>()), resultType),
                NPTypeCode.Char => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetChar(Array.Empty<long>()), resultType),
                NPTypeCode.Half => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetHalf(Array.Empty<long>()), resultType),
                NPTypeCode.Single => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetSingle(Array.Empty<long>()), resultType),
                NPTypeCode.Double => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDouble(Array.Empty<long>()), resultType),
                NPTypeCode.Decimal => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetDecimal(Array.Empty<long>()), resultType),
                NPTypeCode.Complex => InvokeBinaryScalarRhs(func, lhsVal, rhs.GetComplex(Array.Empty<long>()), resultType),
                _ => throw new NotSupportedException($"RHS type {rhsType} not supported")
            };
        }

        /// <summary>
        /// Complete binary scalar dispatch with typed LHS and RHS values.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static NDArray InvokeBinaryScalarRhs<TLhs, TRhs>(
            Delegate func, TLhs lhsVal, TRhs rhsVal, NPTypeCode resultType)
        {
            // Dispatch based on result type
            return resultType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TLhs, TRhs, bool>)func)(lhsVal, rhsVal)),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TLhs, TRhs, byte>)func)(lhsVal, rhsVal)),
                NPTypeCode.SByte => NDArray.Scalar(((Func<TLhs, TRhs, sbyte>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TLhs, TRhs, short>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TLhs, TRhs, ushort>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TLhs, TRhs, int>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TLhs, TRhs, uint>)func)(lhsVal, rhsVal)),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TLhs, TRhs, long>)func)(lhsVal, rhsVal)),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TLhs, TRhs, ulong>)func)(lhsVal, rhsVal)),
                NPTypeCode.Char => NDArray.Scalar(((Func<TLhs, TRhs, char>)func)(lhsVal, rhsVal)),
                NPTypeCode.Half => NDArray.Scalar(((Func<TLhs, TRhs, Half>)func)(lhsVal, rhsVal)),
                NPTypeCode.Single => NDArray.Scalar(((Func<TLhs, TRhs, float>)func)(lhsVal, rhsVal)),
                NPTypeCode.Double => NDArray.Scalar(((Func<TLhs, TRhs, double>)func)(lhsVal, rhsVal)),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TLhs, TRhs, decimal>)func)(lhsVal, rhsVal)),
                NPTypeCode.Complex => NDArray.Scalar(((Func<TLhs, TRhs, System.Numerics.Complex>)func)(lhsVal, rhsVal)),
                _ => throw new NotSupportedException($"Result type {resultType} not supported")
            };
        }

        /// <summary>
        /// Classify execution path based on strides.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ExecutionPath ClassifyPath(
            long* lhsStrides, long* rhsStrides, long* shape, int ndim, NPTypeCode resultType)
        {
            if (ndim == 0)
                return ExecutionPath.SimdFull;

            bool lhsContiguous = StrideDetector.IsContiguous(lhsStrides, shape, ndim);
            bool rhsContiguous = StrideDetector.IsContiguous(rhsStrides, shape, ndim);

            if (lhsContiguous && rhsContiguous)
                return ExecutionPath.SimdFull;

            // SimdScalarRight/Left require the non-scalar operand to be contiguous
            // because their loops use simple i * elemSize indexing
            bool rhsScalar = StrideDetector.IsScalar(rhsStrides, ndim);
            if (rhsScalar && lhsContiguous)
                return ExecutionPath.SimdScalarRight;

            bool lhsScalar = StrideDetector.IsScalar(lhsStrides, ndim);
            if (lhsScalar && rhsContiguous)
                return ExecutionPath.SimdScalarLeft;

            // L3-a/L3-b: SimdChunk now handles ANY constant-stride inner dim
            // (contig=1, broadcast=0, strided>1, negative-stride). The emitted IL
            // hoists outer coord calc out of the inner loop, so even arbitrary
            // strided cases beat the General path's per-element mod/div by ~4-5×.
            // General is left as a safety fallback but is no longer reachable for
            // ndim >= 1 — kept for documentation / future use cases.
            if (ndim >= 1)
                return ExecutionPath.SimdChunk;

            return ExecutionPath.General;
        }

        /// <summary>
        /// Execute the IL-generated kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteKernel(
            MixedTypeKernel kernel,
            NDArray lhs, NDArray rhs, NDArray result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            // The Shape.offset represents the element offset into the underlying storage
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (long* lhsStrides = lhsShape.strides)
            fixed (long* rhsStrides = rhsShape.strides)
            fixed (long* shape = result.shape)
            {
                kernel(
                    (void*)lhsAddr,
                    (void*)rhsAddr,
                    (void*)result.Address,
                    lhsStrides,
                    rhsStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }

        /// <summary>
        ///     Execute kernel using pre-coalesced shape and strides (L3-a). The kernel
        ///     iterates <c>result.size</c> elements writing to the original (uncoalesced)
        ///     result buffer; the coalesced shape/strides only steer the read pattern
        ///     and path classification. For <see cref="ExecutionPath.SimdFull"/> the
        ///     kernel ignores strides anyway, so coalescing F-contig N-D to 1-D contig
        ///     is a free win.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExecuteKernelCoalesced(
            MixedTypeKernel kernel,
            NDArray lhs, NDArray rhs, NDArray result,
            Shape lhsShape, Shape rhsShape,
            long* coalShape, long* coalLhsStr, long* coalRhsStr, int coalNdim)
        {
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            kernel(
                (void*)lhsAddr,
                (void*)rhsAddr,
                (void*)result.Address,
                coalLhsStr,
                coalRhsStr,
                coalShape,
                coalNdim,
                result.size
            );
        }

        /// <summary>
        ///     Merge adjacent dims of a 2-operand iteration into a smaller ndim
        ///     whenever both operands share a compatible cross-axis stride relation.
        ///     Mutates the input buffers in place; returns the new ndim.
        ///
        ///     Two adjacent dims (d, d+1) merge when, for BOTH operands:
        ///       • either dim has size 1 (trivial), or
        ///       • C-order: stride[d] == stride[d+1] * shape[d+1], or
        ///       • F-order: stride[d+1] == stride[d] * shape[d], or
        ///       • both strides are 0 (joint broadcast).
        ///     Mixed cases (one operand C-order, the other F-order; one broadcast,
        ///     the other not) leave the dims separate — the path classifier picks
        ///     <see cref="ExecutionPath.SimdChunk"/> or <see cref="ExecutionPath.General"/>
        ///     instead.
        /// </summary>
        /// <remarks>
        ///     The merged stride is the smaller non-zero one (the contiguous neighbour);
        ///     if both are 0, the result stays 0. Shape merges as <c>shape[d]*shape[d+1]</c>.
        /// </remarks>
        /// <summary>
        ///     Merge-direction classifier for one operand on an adjacent pair of dims.
        ///     <list type="bullet">
        ///       <item><see cref="MergeMode.Trivial"/> — either dim is size 1, or both
        ///       strides are 0 (joint broadcast). Can merge in any direction.</item>
        ///       <item><see cref="MergeMode.COrder"/> — stride[d] == stride[d+1]*shape[d+1]
        ///       (descending stride order = row-major).</item>
        ///       <item><see cref="MergeMode.FOrder"/> — stride[d+1] == stride[d]*shape[d]
        ///       (ascending stride order = column-major).</item>
        ///       <item><see cref="MergeMode.None"/> — neither relation holds; the dims
        ///       can't be coalesced for this operand.</item>
        ///     </list>
        ///     The 3-operand merge requires either all-Trivial or one consistent order
        ///     (Trivial mixes with anything). Mixing C and F silently transposes — that
        ///     was the bug in <c>arr * arr.T</c> and reversed-transposed adds.
        /// </summary>
        private enum MergeMode { None, COrder, FOrder, Trivial }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static MergeMode ClassifyMergePair(long sw, long sr, long strideW, long strideR)
        {
            if (sw == 1 || sr == 1) return MergeMode.Trivial;
            if (strideW == 0 && strideR == 0) return MergeMode.Trivial;
            // C-order: outer stride spans the inner dim's full extent.
            if (strideW == strideR * sr) return MergeMode.COrder;
            // F-order: inner stride spans the outer dim's full extent.
            if (strideR == strideW * sw) return MergeMode.FOrder;
            return MergeMode.None;
        }

        /// <summary>
        ///     Coalesce adjacent dims when ALL THREE operands (lhs, rhs, result) agree
        ///     on the merge direction. Trivial classifications mix with either order;
        ///     mixing COrder and FOrder among operands rejects the merge to avoid the
        ///     "silent transpose" bug (the kernel walks linearly across heterogeneously
        ///     ordered operands — element positions diverge).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe int CoalesceTernaryDimensions(
            long* shape, long* lhsStrides, long* rhsStrides, long* resStrides, int ndim)
        {
            if (ndim <= 1) return ndim;

            int writeAxis = 0;
            int newNdim = 1;

            for (int readAxis = 1; readAxis < ndim; readAxis++)
            {
                long shapeW = shape[writeAxis];
                long shapeR = shape[readAxis];

                var lhsMode = ClassifyMergePair(shapeW, shapeR, lhsStrides[writeAxis], lhsStrides[readAxis]);
                var rhsMode = ClassifyMergePair(shapeW, shapeR, rhsStrides[writeAxis], rhsStrides[readAxis]);
                var resMode = ClassifyMergePair(shapeW, shapeR, resStrides[writeAxis], resStrides[readAxis]);

                bool canMerge = lhsMode != MergeMode.None
                             && rhsMode != MergeMode.None
                             && resMode != MergeMode.None
                             && AgreeOnOrder(lhsMode, rhsMode, resMode);

                if (canMerge)
                {
                    shape[writeAxis] = shapeW * shapeR;
                    lhsStrides[writeAxis] = MergeStride(lhsStrides[writeAxis], lhsStrides[readAxis]);
                    rhsStrides[writeAxis] = MergeStride(rhsStrides[writeAxis], rhsStrides[readAxis]);
                    resStrides[writeAxis] = MergeStride(resStrides[writeAxis], resStrides[readAxis]);
                }
                else
                {
                    writeAxis++;
                    if (writeAxis != readAxis)
                    {
                        shape[writeAxis] = shapeR;
                        lhsStrides[writeAxis] = lhsStrides[readAxis];
                        rhsStrides[writeAxis] = rhsStrides[readAxis];
                        resStrides[writeAxis] = resStrides[readAxis];
                    }
                    newNdim++;
                }
            }

            return newNdim;
        }

        /// <summary>
        ///     True when the three merge modes are mutually consistent. Trivial pairs
        ///     with anything; COrder and FOrder are mutually exclusive once present.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool AgreeOnOrder(MergeMode a, MergeMode b, MergeMode c)
        {
            bool hasC = a == MergeMode.COrder || b == MergeMode.COrder || c == MergeMode.COrder;
            bool hasF = a == MergeMode.FOrder || b == MergeMode.FOrder || c == MergeMode.FOrder;
            return !(hasC && hasF);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long MergeStride(long strideW, long strideR)
        {
            if (strideW == 0 && strideR == 0) return 0;
            if (strideW == 0) return strideR;
            if (strideR == 0) return strideW;
            return strideW < strideR ? strideW : strideR;
        }

        /// <summary>
        /// Fallback to legacy implementation when IL kernel is not available.
        /// </summary>
        private void FallbackBinaryOp(
            NDArray lhs, NDArray rhs, NDArray result,
            BinaryOp op, Shape lhsShape, Shape rhsShape)
        {
            // For now, throw - all kernels should be generatable
            // In future, this could call the legacy generated code
            throw new NotSupportedException(
                $"IL kernel not available for {lhs.GetTypeCode} {op} {rhs.GetTypeCode} -> {result.GetTypeCode}. " +
                "Please report this as a bug.");
        }
    }
}
