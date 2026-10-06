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
        // Call-invariant per-operand flag arrays for the NDIter routes
        // (Wave 2.2): identical on every call, so allocating them per call
        // was pure small-N overhead. The operand arrays stay per-call --
        // the iterator stores the reference (_operands) and the overlap
        // machinery can construct nested iterators (np.copyto) on the same
        // thread, so thread-static reuse would alias live iterators.
        private static readonly NDIterPerOpFlags[] s_binaryIterFlags =
        {
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
            NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
        };

        // Element-count cutoff below which a SAME-DTYPE single-broadcast op skips
        // the NDIter route in favour of the lighter direct SimdChunk kernel (see
        // the gate in TryExecuteBinaryOpViaNDIter). NDIter's multi-operand
        // construction costs a fixed ~0.5 µs that dominates a tiny broadcast:
        // measured 1.8-2.3× faster on 1K row/col/3-D float64 broadcasts. The
        // direct SimdChunk kernel matches or beats NDIter up to ~16K elements;
        // above that NDIter's coalescing + vectorization pulls ahead, so the
        // cutoff sits safely below the crossover (with margin for host-regime
        // measurement noise). Env override NS_BROADCAST_DIRECT_MAX for tuning.
        internal static readonly long DirectBroadcastMaxElements =
            long.TryParse(Environment.GetEnvironmentVariable("NS_BROADCAST_DIRECT_MAX"), out var t) && t >= 0
                ? t : 8192;

        /// <summary>
        /// Execute a binary operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Operation to perform</param>
        /// <param name="@out">Optional provided output (NumPy ufunc out=): the
        /// result is written into it (dtype must be same_kind-castable from the
        /// loop dtype, shape joins the broadcast without stretching) and the
        /// same instance is returned.</param>
        /// <param name="where">Optional bool write mask (NumPy ufunc where=):
        /// only mask-true elements are computed/written; false slots keep the
        /// prior out contents (uninitialized for a fresh result).</param>
        /// <param name="dtype">Optional explicit loop dtype (NumPy ufunc dtype=):
        /// overrides NEP50 promotion — the loop COMPUTES in this dtype (probed
        /// 2.4.2: power(10,11,dtype=f64) = 1e11 exactly, no int wrap; the
        /// f32-rounding of sqrt/power dtype=f32 lands even in a wider out).
        /// Each input must be same_kind-castable to it, and a provided out is
        /// validated against it rather than the promoted dtype.</param>
        /// <returns>Result array with promoted type (or <paramref name="@out"/>)</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray ExecuteBinaryOp(NDArray lhs, NDArray rhs, BinaryOp op,
            NDArray @out = null, NDArray where = null, NPTypeCode? dtype = null)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Determine result type using NumPy type promotion rules
            var resultType = np._FindCommonType(lhs, rhs);

            // NumPy: true division (/) always returns float64 for integer types
            // This matches Python 3 / NumPy 2.x semantics where / is "true division"
            // Group 3 = float (Single, Double), Group 4 = Decimal
            if (op == BinaryOp.Divide && resultType.GetGroup() < 3)
            {
                resultType = NPTypeCode.Double;
            }

            // NumPy Power promotion (NEP50) is exactly result_type(base, exp) — which
            // _FindCommonType already computes, including the int-base/float-exp cases:
            //   - i32_arr ** i32_arr  → int32         - f32_arr ** f64_arr → float64
            //   - i32_arr ** f32_arr  → float64       - i16_arr ** f32_arr → float32
            //   - u16_arr ** f32_arr  → float32 (uint16 fits in float32, probed 2.4.2)
            //   - f32_arr ** i32_arr  → float64 (NEP50 strict)
            //   - 2 (weak int) ** f32_arr → float32   - 2 ** f64_arr → float64
            // A prior override forced `lhsGroup<=2 && rhsGroup==3 → float64` for EVERY
            // int-base ** float-exp, which wrongly upcast {bool,int8,int16,uint8,uint16,
            // char} ** float32 to float64 (NumPy keeps float32) — a NumPy-1.x rule NEP50
            // removed. Removed: result_type (above) is the single source of truth.

            // ufunc dtype= (NumPy loop-signature override): the loop runs IN the
            // requested dtype — inputs are cast (same_kind-validated, NumPy's
            // UFuncTypeError text) and computation happens at that precision.
            // Replaces the promoted dtype BEFORE the out/where branch so a
            // provided out is validated against the dtype-overridden loop
            // (probed: power(dtype=f32, out=i32) reports float32 → int32).
            if (dtype.HasValue && dtype.Value != resultType)
            {
                ValidateBinaryInputCasts(lhsType, rhsType, dtype.Value, UfuncName(op));
                resultType = dtype.Value;
            }

            // Several NumPy ufuncs have NO bool loop — their smallest integer loop starts at int8,
            // so a bool×bool input promotes to int8 (probed 2.4.2):
            //   left_shift/right_shift (bb->b absent),  floor_divide (True//True -> 1:int8),
            //   remainder/mod (True%True -> 0:int8),    power (True**True -> 1:int8),
            //   fmod (True fmod True -> 0:int8).
            // Bump BEFORE the bool->logical remap below so these never take the bitwise-OR/AND path.
            if (resultType == NPTypeCode.Boolean &&
                (op == BinaryOp.LeftShift || op == BinaryOp.RightShift ||
                 op == BinaryOp.FloorDivide || op == BinaryOp.Mod || op == BinaryOp.Fmod || op == BinaryOp.Power))
                resultType = NPTypeCode.SByte;

            // NumPy bool arithmetic: the bool dtype has no integer add/multiply ufunc loop — `+`
            // is logical OR and `*` is logical AND (so True + True == True, raw byte 1, not 2).
            // The remap keys off the FINAL loop dtype (i.e. AFTER the dtype= override):
            // add(bool, bool, dtype=i32) runs the i32 add loop and returns 2 (probed 2.4.2),
            // while add(bool, bool) and add(bool, bool, dtype=bool) stay logical OR. The op
            // remap makes every downstream kernel path (SIMD/scalar, same-type/mixed) emit
            // the bitwise op and write a normalized 0/1 byte. (`-` has no bool loop and
            // already throws like NumPy.)
            if (resultType == NPTypeCode.Boolean)
            {
                if (op == BinaryOp.Add) op = BinaryOp.BitwiseOr;
                else if (op == BinaryOp.Multiply) op = BinaryOp.BitwiseAnd;
                // bool has no NaN domain: maximum/fmax == logical OR, minimum/fmin == logical AND
                // (max(F,T)=T=OR, min(F,T)=F=AND). Remap so the bitwise kernels handle them.
                else if (op == BinaryOp.Maximum || op == BinaryOp.FMax) op = BinaryOp.BitwiseOr;
                else if (op == BinaryOp.Minimum || op == BinaryOp.FMin) op = BinaryOp.BitwiseAnd;
            }

            // ufunc out=/where= path (Wave 2.1): the loop dtype above is final
            // (NumPy resolves the loop from the INPUTS; out only constrains the
            // final cast), so branch after promotion and before any allocation.
            // NumPy likewise disables the trivial loop when a wheremask or
            // provided out needs the full iterator (ufunc_object.c:2213).
            if (@out is not null || where is not null)
            {
                // NumPy keeps the trivial loop for a provided out when every operand is trivially iterable
                // (try_trivial_single_output_loop); only a mask, a cast or a layout / overlap the loop cannot take
                // needs the iterator.
                if (where is null && TryTrivialBinaryOpInto(lhs, rhs, op, lhsType, rhsType, resultType, @out))
                    return @out;
                return ExecuteBinaryUfuncInto(lhs, rhs, op, lhsType, rhsType, resultType, @out, where);
            }

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteScalarScalar(lhs, rhs, op, resultType);
            }

            // NEP50 weak-scalar fast-path enablement. A Python-scalar literal arrives wrapped at
            // its OWN dtype (`a * 2` -> int32 0-d; `a * 2.0` -> float64 0-d), so `f32[] * 2`
            // reaches here as f32×int32. resultType already reflects NumPy's weak promotion (here
            // f32), but the same-dtype gate inside TryTrivialContiguousBinaryOp / the SIMD-viability
            // gate inside TryExecuteBinaryOpViaNDIter key off the RAW operand dtypes and bail
            // (f32 != int32), routing to the heavy per-element-convert mixed path — measured ~2.25×
            // slower than the SimdScalarRight scalar-broadcast loop it should take (100K f32:
            // 0.080 ms vs 0.035 ms). When the ARRAY operand already IS resultType and only the
            // 0-d/size-1 scalar differs, materialize that scalar at resultType (one element —
            // exactly what NumPy does, casting the scalar to the loop dtype), so both operands share
            // resultType and the same-dtype SIMD paths below light up. When the array ALSO differs
            // (e.g. int32[] * 2.5 -> float64) we leave it: the array genuinely needs a cast, which is
            // the mixed path's job. (scalar×scalar already returned above; value is identical either
            // way — the mixed path converts the same scalar to resultType per element.)
            // NEP50 scalar-cast temp. Cast(..., copy:true) mints a FRESH 0-d/size-1 array and we
            // REASSIGN the operand parameter to it, so the temp is unreachable once the op returns
            // — an undisposed intermediate (a pooled buffer + a finalizable graph) reclaimable only
            // by a future GC + finalizer pass. It is captured here and disposed at every exit below.
            // Deliberately NOT an [NDScoped] boundary: ExecuteBinaryOp is the library's hottest path
            // (every +,-,*,/,%,… on arrays), so a method scope would impose Open/Track/Dispose on ALL
            // binary ops merely to reclaim a temp that ONLY this narrow array-op-scalar-needing-cast
            // sub-path ever mints — the targeted dispose costs nothing on the common (same-dtype /
            // array-array) path. Result buffers are always fresh (a kernel writes a new output from
            // the operands), never an alias of this scalar temp, so disposing it at return is safe.
            // Gate: UndisposedIntermediateTests.BinaryScalarCastTemp_IsDisposed.
            NDArray scalarCastTemp = null;
            if (lhsType != rhsType)
            {
                bool lhsScalarLike = lhs.Shape.IsScalar || lhs.Shape.size == 1;
                bool rhsScalarLike = rhs.Shape.IsScalar || rhs.Shape.size == 1;
                if (rhsScalarLike && !lhsScalarLike && lhsType == resultType)
                {
                    scalarCastTemp = rhs = Cast(rhs, resultType, copy: true);
                    rhsType = resultType;
                }
                else if (lhsScalarLike && !rhsScalarLike && rhsType == resultType)
                {
                    scalarCastTemp = lhs = Cast(lhs, resultType, copy: true);
                    lhsType = resultType;
                }
            }

            // -------- O(1) trivial-loop bypass -----------------------------
            // NumSharp analogue of NumPy's check_for_trivial_loop +
            // try_trivial_single_output_loop (ufunc_object.c), which handle a
            // single strided inner loop "without using the (heavy) iterator."
            // When both operands share ONE contiguous layout (both C, or both F),
            // have identical shape (no broadcast) and the same dtype as the result
            // (no cast), a single linear walk over all three buffers visits the
            // same logical element — so we route straight to the existing DirectIL
            // SimdFull whole-array kernel and skip the NDIter MultiNew/Initialize
            // construction (measured ~600-2000 ns/call: 22-24% of a small
            // contiguous op, <=3% once n>=64K). Returns null for anything that is
            // not trivially contiguous (broadcast/mixed-C-F/strided/cast/scalar-
            // broadcast/unsupported emit) → falls through to the NDIter route
            // below with behaviour unchanged.
            {
                var trivial = TryTrivialContiguousBinaryOp(lhs, rhs, op, lhsType, rhsType, resultType);
                if (trivial is not null) { scalarCastTemp?.Dispose(); return trivial; }
            }

            // -------- NDIter Tier 3B fast path (all binary ops) -----------
            // Routes through the NDIter inner-loop kernel factory, which
            // collapses coalesce + SIMD dispatch (contig, SimdScalarLeft,
            // SimdScalarRight, scalar-strided) into a single emitted kernel
            // driven by NDIter's multi-operand iterator.
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
                var routed = TryExecuteBinaryOpViaNDIter(lhs, rhs, op, lhsType, rhsType, resultType);
                if (routed is not null) { scalarCastTemp?.Dispose(); return routed; }
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
            //
            // A strictly F operand can still BROADCAST: a (2,1,2) F array against a (2,3,2) F
            // array is "strict F" by its own flags, yet its stride-0 middle axis blocks the
            // F-order merge below, so the coalesced iteration keeps several dims and the linear
            // C-order write over an F result scrambles the values (every SIMD dtype under the
            // small-broadcast cutoff, and float16 add/subtract/multiply/divide at any size,
            // which always take this route). So an F result is allocated here only when every
            // non-scalar operand has the result's own dims; a broadcast all-F set computes in C
            // order and the post-kernel branch relabels it F, NumPy's layout for that set.
            //
            // Such a broadcast all-F set is instead walked with its axes REVERSED: the kernel's C-coordinate order over
            // the reversed dims is the F memory order of the original ones, so the linear write fills an F-allocated
            // result correctly, every operand is read along its own contiguous (or stride-0) axis, and no C result is
            // copied to F afterwards. Elementwise, so the visiting order changes no value; NumPy's iterator walks such a
            // set in F order too (K order). This is the float16 add / subtract / multiply / divide route for every size
            // (plan api-parameter-parity X4: a (30,40,20) F * (30,1,20) F product read 1,200-element strides along the
            // C-order inner axis, then copied the result to F).
            bool allStrictF = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            bool walkReversed = allStrictF && HasNonScalarBroadcastOperand(lhs, rhs, cleanShape);
            Shape resultShape = allStrictF
                ? Shape.Adopt((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            // Allocate result
            var result = new NDArray(resultType, resultShape, false);

            // Empty broadcast result: no elements to compute. The kernels below assume
            // >= 1 element and walk stride-0 broadcast dims as if non-empty, corrupting
            // memory when a sibling dim is 0 (e.g. (3,1,1) op (1,0,2) -> (3,0,2)).
            // (The NDIter fast path above returns early for the same reason.)
            if (result.size == 0)
            {
                scalarCastTemp?.Dispose();
                return result;
            }

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
                // A reversed walk reads axis (ndim - 1 - d) as the d-th: the F result's strides then descend, so the
                // coalescing below sees a C-ordered result and merges the axes the F operands run along.
                int s = walkReversed ? origNdim - 1 - d : d;
                coalShape[d] = resultShape.dimensions[s];
                coalLhsStr[d] = leftShape.strides[s];
                coalRhsStr[d] = rightShape.strides[s];
                coalResStr[d] = resultShape.strides[s];
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
            // front (a broadcast all-F set too, through the reversed walk).
            if (!allStrictF && ShouldProduceFContigOutput(lhs, rhs, result.Shape))
            {
                scalarCastTemp?.Dispose();
                // copy('F') mints a fresh F-contig array; the C-contig `result` the kernel wrote
                // is now dead — dispose it rather than drop it to the finalizer (mechanism-3 leak).
                var fResult = result.copy('F');
                result.Dispose();
                return fResult;
            }

            scalarCastTemp?.Dispose();
            return result;
        }

        /// <summary>
        ///     Try to execute a binary op via NDIter Tier 3B for any dtype
        ///     combination (same or mixed). Returns the result array on
        ///     success, or null if the route is not applicable (broadcast
        ///     result exceeds int.MaxValue, NDIter not built for long-
        ///     shape arithmetic; unsupported op/dtype emit).
        ///
        ///     Allocates the output as F-contig when both inputs are strictly
        ///     F (matches the pre-existing direct-path rule from L3-b) and
        ///     picks the NDIter order accordingly: NPY_FORTRANORDER for
        ///     strict-F-both, NPY_CORDER everywhere else. NPY_CORDER also
        ///     handles the reversed-stride case correctly because NDIter
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
        private unsafe NDArray? TryExecuteBinaryOpViaNDIter(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // Broadcast → clean shape so we know what the result looks like.
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // (Half,Half)→Half add/sub/mul/div DECLINE Tier 3B: the direct route serves
            // them with the bit-exact SIMD widen-compute-narrow kernels
            // (Binary.Arith.Half.cs — float32 compute + Giesen RTNE narrow, NumPy's
            // exact HALF loop incl. the NaN-payload operand-order pin), while this
            // route's scalar inner-loop body bridges Half through DOUBLE (its hardware
            // (double)Half widen QUIETS sNaN before the op and double-rounds
            // differently on exponent-gap sums). Contiguous same-shape pairs already
            // took the trivial bypass; this redirect sends the scalar-broadcast
            // layouts to SimdScalarLeft/Right and leaves strided on the same
            // EmitScalarOperation numerics as before (via SimdChunk).
            // Power joins them (plan api-parameter-parity X13): HALF_power is powf on the halves widened bit for bit, and
            // the same kernels run it with a vector widen / narrow around a per-lane powf.
            if (lhsType == NPTypeCode.Half && rhsType == NPTypeCode.Half && resultType == NPTypeCode.Half
                && (op == BinaryOp.Add || op == BinaryOp.Subtract || op == BinaryOp.Multiply || op == BinaryOp.Divide
                    || op == BinaryOp.Power))
                return null;

            // NDIter's internal shape arithmetic is int-bounded; route only
            // when the broadcast result fits. Pre-existing test
            // LongIndexingBroadcastTest exercises the > int.MaxValue path via
            // the direct allocator (which is also int-limited but doesn't
            // throw on the shape calc itself). Falling through to the direct
            // path keeps the prior behaviour for those edge cases.
            if (cleanShape.size < 0) return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue) return null;

            // Small SAME-DTYPE single-broadcast fast path: for a tiny result the
            // NDIter multi-operand construction (~0.5 µs) dominates, and the direct
            // SimdChunk kernel (reached by falling through to the direct route below)
            // is measurably faster there — 1.8-2.3× on 1K row/col/3-D float64
            // broadcasts, bringing them from ~0.55× NumPy to faster-than-NumPy.
            // Returning null hands the op to the direct path in ExecuteBinaryOp.
            //
            // The route is gated to exactly the cases the direct SimdChunk kernel is
            // proven BIT-IDENTICAL to NDIter on (verified by the oracle FuzzMatrix
            // AND the layout-parity integration gates):
            //   • identical dtypes — the direct path's mixed-dtype convert loop
            //     diverges from NDIter on some NEP50 promotions;
            //   • a SIMD-capable op+dtype — Half/Decimal/Complex (and the scalar-only
            //     ops Mod/Power/FloorDivide/ATan2) take a scalar path whose specials
            //     diverge (e.g. float16  256 − inf);
            //   • EXACTLY ONE operand broadcast (XOR) — a genuine single-operand
            //     broadcast (row/col/N-D, one operand carries the full result shape).
            //     A kron-style DOUBLE broadcast (both stretched on complementary axes)
            //     leaves outputs unwritten in SimdChunk; a same-shape non-broadcast op
            //     is either contiguous (already handled by the trivial bypass) or a
            //     strided view SimdChunk mishandles — both must stay on NDIter;
            //   • both operands C- or F-contiguous — SimdChunk mishandles a strided /
            //     trailing-size-1 / negative-stride operand that NDIter absorbs
            //     (the (N,1)-strided-view and reshape-view integration gates), so
            //     only clean-layout inputs take this route;
            //   • NOT every non-scalar operand strictly F — NumPy lays such a set out F,
            //     and the direct route can only write it in C and copy it to F (its F
            //     allocation is unsafe once an operand broadcasts, see ExecuteBinaryOp),
            //     which measured 4-6× slower than this route's F allocation (16³ float64:
            //     7.1 µs vs 1.7 µs; NumPy 2.7 µs).
            // Above DirectBroadcastMaxElements NDIter's coalescing/vectorization wins,
            // so the op stays here.
            if (lhsType == rhsType && lhsType == resultType
                && cleanShape.size < DirectBroadcastMaxElements
                && (leftShape.IsBroadcasted ^ rightShape.IsBroadcasted)
                && IsContiguousCorF(lhs.Shape) && IsContiguousCorF(rhs.Shape)
                && !AreAllOperandsStrictFContig(lhs, rhs, cleanShape)
                && DirectILKernelGenerator.CanUseSimdBinary(op, resultType))
                return null;

            // Mirror the direct path: F-allocate output when every non-scalar
            // operand is strict-F. Otherwise default to C and let the
            // post-kernel "looser-F" copy step rectify when needed.
            bool allStrictFContig = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            Shape resultShape = allStrictFContig
                ? Shape.Adopt((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            var result = new NDArray(resultType, resultShape, false);

            // Empty broadcast result (a stride-0 broadcast dim alongside a zero-size
            // dim, e.g. (3,1,1) op (1,0,2) -> (3,0,2)): there is nothing to compute.
            // Returning here is REQUIRED — the NDIter element-wise path corrupts the
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
                              && DirectILKernelGenerator.CanUseSimdBinary(op, resultType);

            // NOTE (Wave 4, measured): the buffered-cast route (NumPy's ufunc
            // config — cast inputs to the computation dtype in 8192-element
            // windows, then run the same-dtype SIMD body) was implemented and
            // A/B-measured here, and LOST to this fused per-element-convert
            // path for every SIMD-able binary op class on i9-13900K/Release:
            //   add contig 2M:  buffered 2.20 ms vs fused 1.49 ms
            //   add strided 1M: buffered 3.18 ms vs fused 2.98 ms
            //   div contig 2M:  buffered 1.72 ms vs fused 1.61 ms
            // The extra buffer round-trip (~16 B/element of L2 traffic + window
            // machinery) outweighs the SIMD gain on cheap ops. NumPy buffers
            // because its AOT C loops cannot fuse casts; our runtime IL CAN —
            // the fused path also beats NumPy itself (i32+f64 4M: 5.8-6.9 ms vs
            // NumPy 7.3-7.6 ms). Promoting UNARY math ops (sqrt/exp/...) DO use
            // the buffered-cast route (see DefaultEngine.UnaryOp) where the
            // SIMD body wins 1.36x+. Revisit only with new A/B evidence.
            //
            // Build per-element scalar emit body. For same-dtype we just call
            // EmitScalarOperation directly. For mixed-dtype we wrap it with
            // a per-operand convert pass (the direct path's
            // EmitGeneralLoop / EmitChunkLoop does the same).
            // NumPy's elementwise NaN operand order (float add/multiply keep the SECOND operand's NaN when both are
            // NaN): see DirectILKernelGenerator.NaNOperandOrder.cs.
            Action<ILGenerator> scalarBody;
            if (sameDtype)
            {
                scalarBody = il => DirectILKernelGenerator.EmitScalarOperation(il, op, resultType, NaNOperandOrder.NumPyUfunc);
            }
            else
            {
                NPTypeCode capLhs = lhsType, capRhs = rhsType, capRes = resultType;
                BinaryOp capOp = op;
                scalarBody = il => EmitMixedScalarBody(il, capLhs, capRhs, capRes, capOp);
            }

            Action<ILGenerator>? vectorBody = simdViable
                ? il => DirectILKernelGenerator.EmitVectorOperation(il, op, resultType, NaNOperandOrder.NumPyUfunc)
                : null;

            // Cache key MUST encode all three dtypes; mixed-dtype kernels
            // are distinct from same-dtype ones for the same op. Packed key
            // (no per-call string): npy_binop_{op}_{lhsType}_{rhsType}_{resultType}.
            var cacheKey = InnerLoopKernelKey.Binary(op, lhsType, rhsType, resultType);

            try
            {
                // COPY_IF_OVERLAP + OVERLAP_ASSUME_ELEMENTWISE mirrors NumPy's
                // ufunc iterator flags (ufunc_object.c:1070): overlapping
                // write/read operands force a write-back temporary; exact
                // aliasing stays copy-free because this loop is elementwise.
                // With a freshly allocated result this is a cheap extent check.
                using var iter = NDIterRef.MultiNew(
                    3, new[] { lhs, rhs, result },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP,
                    order,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    s_binaryIterFlags);

                // How NumPy would split this iteration into inner-loop calls — read only by the cold path of a
                // float add/multiply whose broadcast operand is NaN (the struct outlives the call below).
                BroadcastNaNRule nanRule = NumPyBroadcastNaNRule(op, lhsType, rhsType, resultType, resultType,
                    lhs.size, rhs.size, masked: false, iter.NDim, iter.HasNegPerm, iter.GetInnerStrideArray());
                iter.ExecuteElementWiseBinary(lhsType, rhsType, resultType, scalarBody, vectorBody, cacheKey, &nanRule);
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
            {
                var fResult = result.copy('F');
                result.Dispose();   // C-contig kernel output now dead (mechanism-3 leak)
                return fResult;
            }

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
        ///     kernel the post-NDIter fallback uses, so results are identical for
        ///     every dtype/op (the generator emits a SIMD or scalar loop per dtype).
        ///     Unsupported emits (e.g. bool subtract) throw inside the generator;
        ///     we catch and return null so the existing path raises/handles the
        ///     case exactly as before. Any non-trivial case returns null →
        ///     caller proceeds to the NDIter route.
        /// </summary>
        private unsafe NDArray? TryTrivialContiguousBinaryOp(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType)
        {
            // No cast: all three dtypes identical. Promotion cases (/ -> f64,
            // int-base ** float -> f64, mixed dtypes) have resultType != input and
            // are excluded here, deferring to the NDIter route that does the cast.
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
                // The result takes the array's shape only when the one-element operand adds no dimension: a (1,1)
                // operand against a (7,) array broadcasts to (1,7) in NumPy, which this shortcut would return as (7,)
                // (plan api-parameter-parity X19). Such a call takes the broadcasting routes below.
                if (rhsScalarLike ? rs.NDim > ls.NDim : ls.NDim > rs.NDim)
                    return null;
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

            // Reuse a canonical input shape (offset 0, owns its buffer, right order)
            // instead of cloning dims + reconstructing strides/flags. bothF implies
            // strictly column-major ls (1-D would have satisfied bothC first).
            Shape resultShape = CanonicalResultShape(ls, bothF);
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
        ///     once, array walked linearly), skipping NDIter construction. The result
        ///     takes the array operand's shape and layout (C, or strictly-F) so the
        ///     linear write aligns with the linear array read. Returns null (→ NDIter)
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
                return null;   // strided/transposed array operand → NDIter

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

            Shape resultShape = CanonicalResultShape(arrShape, isF);
            var result = new NDArray(resultType, resultShape, false);
            if (result.size == 0)
                return result;

            // lhs/rhs keep their original operand positions; the SimdScalarRight/Left
            // kernel reads the scalar side once and walks the array side linearly.
            ExecuteKernel(kernel, lhs, rhs, result, lhs.Shape, rhs.Shape);
            return result;
        }

        /// <summary>
        ///     The trivial loop for a binary ufunc WITH a provided output: NumPy's ufunc runs
        ///     <c>try_trivial_single_output_loop</c> for <c>out=</c> too whenever every operand is trivially iterable,
        ///     and so does this — the same whole-array DirectIL kernel the fresh-result trivial loop
        ///     (<see cref="TryTrivialContiguousBinaryOp"/>) calls, writing straight into <paramref name="out"/>, with no
        ///     iterator. The iterator route stays for everything else (a cast, a broadcast array, a strided or mixed
        ///     layout, an overlap to resolve).
        /// </summary>
        /// <param name="lhs">The first input.</param>
        /// <param name="rhs">The second input.</param>
        /// <param name="op">The binary op, after the bool remaps of <see cref="ExecuteBinaryOp"/>.</param>
        /// <param name="lhsType">The first input's dtype.</param>
        /// <param name="rhsType">The second input's dtype.</param>
        /// <param name="resultType">The loop dtype.</param>
        /// <param name="out">The provided output; written in place when the call is trivial.</param>
        /// <returns>True when the result was written into <paramref name="out"/>; false (nothing written, nothing
        /// raised) when the call is not trivially iterable or the kernel is not available, so the caller takes the
        /// iterator route — which also raises every error (a read-only out, a bad cast, a shape mismatch).</returns>
        /// <remarks>
        ///     <para>
        ///     Trivial means: one dtype for both inputs and the output (no cast); a writeable, non-broadcast output,
        ///     C- or F-contiguous; every array input of the output's exact dimensions, contiguous in the output's
        ///     order; at most one size-1 input, of rank no higher than the output's (it broadcasts, as a stride-0
        ///     scalar, without growing the shape — NumPy's 0-d operands); and no memory overlap that NumPy's
        ///     <c>COPY_IF_OVERLAP</c> would resolve: an array input is either disjoint from the output or the output
        ///     itself (same first element and, with the same dimensions and order, the same strides — the in-place
        ///     <c>np.add(x, 1, out=x)</c>), and a size-1 input is disjoint (it is read once, before the loop overwrites
        ///     anything). The overlaps NumPy's <c>PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK</c> also accepts (an input
        ///     reading ahead of the output) go to the iterator, whose copy gives the same values.
        ///     </para>
        ///     <para>
        ///     Bit-identical to the iterator route: both run the same scalar/vector op bodies, and a float add /
        ///     multiply with a NaN broadcast operand applies NumPy's AVX2 scalar-broadcast region rule over the one
        ///     inner-loop call NumPy's trivial loop makes (<see cref="NumPyBroadcastNaNRule"/> computes the same split
        ///     for the iterator). What it saves is the iterator's construction — and, for float16, the per-element
        ///     scalar body: the iterator route has no float16 vector body (there is no <c>Vector256&lt;Half&gt;</c>
        ///     arithmetic), while the DirectIL kernels serve float16 add / subtract / multiply / divide with the SIMD
        ///     widen-compute-narrow kernels (<c>DirectILKernelGenerator.TryGetHalfArithKernel</c>): 3.9 → ~0.5 ns per
        ///     element for <c>np.add(0-d, x, out=o)</c> over 1M float16 elements.
        ///     </para>
        /// </remarks>
        private unsafe bool TryTrivialBinaryOpInto(
            NDArray lhs, NDArray rhs, BinaryOp op,
            NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType, NDArray @out)
        {
            if (DisableTrivialBinaryOut)
                return false;

            if (lhsType != rhsType || lhsType != resultType || @out.GetTypeCode != resultType)
                return false;

            // The shifts have their own kernels (NumPy's overflow rule); the DirectIL binary kernels are not theirs.
            if (op == BinaryOp.LeftShift || op == BinaryOp.RightShift)
                return false;

            var os = @out.Shape;
            // A read-only or broadcast output, or an empty one, is the iterator route's to raise / return.
            if (!os.IsWriteable || os.IsBroadcasted || os.size == 0)
                return false;
            bool outC = os.IsContiguous;
            bool outF = !outC && os.IsFContiguous;
            if (!outC && !outF)
                return false;

            var ls = lhs.Shape;
            var rs = rhs.Shape;
            bool lhsScalar = ls.size == 1 && ls.NDim <= os.NDim;
            bool rhsScalar = rs.size == 1 && rs.NDim <= os.NDim;
            bool lhsArray = !lhsScalar && IsTrivialArrayOperand(ls, os, outC);
            bool rhsArray = !rhsScalar && IsTrivialArrayOperand(rs, os, outC);

            ExecutionPath path;
            if (lhsArray && rhsArray)
                path = ExecutionPath.SimdFull;
            else if (lhsScalar && rhsArray)
                path = ExecutionPath.SimdScalarLeft;
            else if (lhsArray && rhsScalar)
                path = ExecutionPath.SimdScalarRight;
            else if (resultType == NPTypeCode.Half && IsHalfArithOp(op))
                return TryHalfArithBroadcastInto(lhs, rhs, op, @out, outC);
            else
                return false;   // two size-1 inputs filling a larger out, a broadcast array, a strided input: iterator

            // Overlap: an array input must be disjoint from out or be out itself; a size-1 input must be disjoint.
            if (!TrivialOverlapOk(lhs, @out, lhsArray) || !TrivialOverlapOk(rhs, @out, rhsArray))
                return false;

            MixedTypeKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetMixedTypeKernel(
                    new MixedTypeKernelKey(resultType, resultType, resultType, op, path));
            }
            catch (NotSupportedException)
            {
                return false;   // no DirectIL kernel for this (op, dtype): the iterator route handles / raises it
            }
            if (kernel == null)
                return false;

            int itemSize = resultType.SizeOf();
            byte* lhsAddr = (byte*)lhs.Address + ls.offset * itemSize;
            byte* rhsAddr = (byte*)rhs.Address + rs.offset * itemSize;
            byte* outAddr = (byte*)@out.Address + os.offset * itemSize;
            fixed (long* lhsStrides = ls.strides)
            fixed (long* rhsStrides = rs.strides)
            fixed (long* dims = os.dimensions)
            {
                kernel(lhsAddr, rhsAddr, outAddr, lhsStrides, rhsStrides, dims, os.NDim, os.size);
            }
            TrivialBinaryOutRuns++;
            return true;
        }

        /// <summary>Whether <paramref name="op"/> is one of the float16 ufuncs the SIMD widen-compute-narrow kernels
        /// serve (<c>DirectILKernelGenerator.TryGetHalfArithKernel</c>): add, subtract, multiply, divide and power (powf per
        /// lane between the vector widen and narrow, plan api-parameter-parity X13).</summary>
        /// <param name="op">The binary op.</param>
        /// <returns>True for add, subtract, multiply, divide and power.</returns>
        private static bool IsHalfArithOp(BinaryOp op)
            => op == BinaryOp.Add || op == BinaryOp.Subtract || op == BinaryOp.Multiply || op == BinaryOp.Divide
               || op == BinaryOp.Power;

        /// <summary>
        ///     float16 add / subtract / multiply / divide into a C- or F-contiguous <paramref name="out"/> whose inputs
        ///     broadcast to it or are strided: the direct route's strided widen-compute-narrow kernel
        ///     (<c>HalfArithStrided</c>) writes straight into <paramref name="out"/>, walked in its memory order. NumPy
        ///     runs its iterator there, whose HALF loop is scalar; so did NumSharp, with a scalar float16 body of ~5 ns an
        ///     element (plan api-parameter-parity X15: numpy.ma's anom subtracts a column mean into an <c>out=</c>
        ///     buffer).
        /// </summary>
        /// <param name="lhs">The first float16 input.</param>
        /// <param name="rhs">The second float16 input.</param>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="out">The float16 output: writeable, not broadcast, non-empty, C- or F-contiguous (checked by the
        ///     caller).</param>
        /// <param name="outC">True when <paramref name="out"/> is C-contiguous, false when it is strictly F-contiguous
        ///     (then the axes are walked reversed, so the linear write follows its memory order).</param>
        /// <returns>True when the result was written; false (nothing written) when an input does not broadcast to
        ///     <paramref name="out"/>'s dimensions or may overlap it — the iterator route raises NumPy's broadcast error
        ///     or resolves the overlap with its copy.</returns>
        /// <remarks>
        ///     Bit-identical to the iterator route: NumPy's float16 arithmetic is elementwise with a per-element NaN
        ///     priority (no call-schedule rule), and the SIMD kernels are its HALF loop. An input may be <paramref name="out"/>
        ///     itself (the same first element, dimensions and strides: each element is read before it is written at the
        ///     same position); any other overlap goes to the iterator, as in <see cref="TrivialOverlapOk"/>.
        /// </remarks>
        private unsafe bool TryHalfArithBroadcastInto(NDArray lhs, NDArray rhs, BinaryOp op, NDArray @out, bool outC)
        {
            var os = @out.Shape;
            int nd = os.NDim;
            long* lStr = stackalloc long[nd > 0 ? nd : 1];
            long* rStr = stackalloc long[nd > 0 ? nd : 1];
            if (!HalfBroadcastStridesTo(lhs.Shape, os, lStr) || !HalfBroadcastStridesTo(rhs.Shape, os, rStr))
                return false;

            if (!HalfInputMayRun(lhs, @out) || !HalfInputMayRun(rhs, @out))
                return false;

            // Walk out's memory order: C as given, F with the axes reversed (its strides then descend).
            long* shape = stackalloc long[nd > 0 ? nd : 1];
            long* ls = stackalloc long[nd > 0 ? nd : 1];
            long* rs = stackalloc long[nd > 0 ? nd : 1];
            long* res = stackalloc long[nd > 0 ? nd : 1];
            long[] dims = os.dimensions, outStrides = os.strides;
            for (int d = 0; d < nd; d++)
            {
                int s = outC ? d : nd - 1 - d;
                shape[d] = dims[s];
                ls[d] = lStr[s];
                rs[d] = rStr[s];
                res[d] = outStrides[s];
            }
            int cnd = CoalesceTernaryDimensions(shape, ls, rs, res, nd);
            ExecutionPath path = ClassifyPath(ls, rs, shape, cnd, NPTypeCode.Half);

            MixedTypeKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetMixedTypeKernel(
                    new MixedTypeKernelKey(NPTypeCode.Half, NPTypeCode.Half, NPTypeCode.Half, op, path));
            }
            catch (NotSupportedException)
            {
                return false;
            }
            if (kernel == null)
                return false;

            byte* lhsAddr = (byte*)lhs.Address + lhs.Shape.offset * 2;
            byte* rhsAddr = (byte*)rhs.Address + rhs.Shape.offset * 2;
            byte* outAddr = (byte*)@out.Address + os.offset * 2;
            kernel(lhsAddr, rhsAddr, outAddr, ls, rs, shape, cnd, os.size);
            TrivialBinaryOutRuns++;
            return true;
        }

        /// <summary>
        ///     The ELEMENT strides an input walks <paramref name="target"/>'s dimensions with (right-aligned broadcasting:
        ///     an extent-1 or missing axis reads stride 0), or false when it does not broadcast to them without stretching
        ///     the target.
        /// </summary>
        /// <param name="input">The input's shape.</param>
        /// <param name="target">The output's shape.</param>
        /// <param name="strides">Receives one stride per output axis.</param>
        /// <returns>True when the input broadcasts to the output's dimensions.</returns>
        private static unsafe bool HalfBroadcastStridesTo(in Shape input, in Shape target, long* strides)
        {
            int nd = target.NDim, ind = input.NDim;
            if (ind > nd)
                return false;
            long[] idims = input.dimensions, istr = input.strides, tdims = target.dimensions;
            for (int d = 0; d < nd; d++)
            {
                int k = d - (nd - ind);
                if (k < 0 || idims[k] == 1)
                    strides[d] = 0;
                else if (idims[k] == tdims[d])
                    strides[d] = istr[k];
                else
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     Whether an input may be read alongside the output in one forward pass: it shares no memory with
        ///     <paramref name="out"/> (NumPy's bounded solver, max_work 1), or it IS <paramref name="out"/> — the same first
        ///     element, dimensions and strides.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="out">The output.</param>
        /// <returns>True when no overlap needs the iterator's copy.</returns>
        private static unsafe bool HalfInputMayRun(NDArray input, NDArray @out)
        {
            if ((byte*)input.Address + input.Shape.offset * 2 == (byte*)@out.Address + @out.Shape.offset * 2
                && SameDims(input.Shape, @out.Shape))
            {
                long[] a = input.Shape.strides, b = @out.Shape.strides;
                bool same = true;
                for (int d = 0; d < a.Length && same; d++)
                    same = a[d] == b[d];
                if (same)
                    return true;
            }
            return NDMemOverlap.SolveMayShareMemory(input, @out, maxWork: 1) == MemOverlap.No;
        }

        /// <summary>
        ///     Test hook: when true on the calling thread, a binary ufunc with <c>out=</c> never takes
        ///     <see cref="TryTrivialBinaryOpInto"/> and always runs the iterator route — so a test can compute the same
        ///     call through both routes and compare their bytes. Thread-static: parallel tests do not see each other's
        ///     setting. Production code never sets it.
        /// </summary>
        [ThreadStatic] internal static bool DisableTrivialBinaryOut;

        /// <summary>
        ///     Test hook: the number of binary <c>out=</c> calls <see cref="TryTrivialBinaryOpInto"/> served on the
        ///     calling thread (it only ever increases; a test reads it before and after a call to learn which route ran).
        /// </summary>
        [ThreadStatic] internal static int TrivialBinaryOutRuns;

        /// <summary>
        ///     Whether an input can be walked element for element against a contiguous output: the output's exact
        ///     dimensions, no broadcast axis, and contiguous in the output's order (C, or F).
        /// </summary>
        /// <param name="s">The input's shape.</param>
        /// <param name="outShape">The output's shape (C- or F-contiguous).</param>
        /// <param name="outC">True when the output is C-contiguous, false when it is strictly F-contiguous.</param>
        /// <returns>True when buffer element k of the input is the logical element of the output's buffer element k.</returns>
        private static bool IsTrivialArrayOperand(in Shape s, in Shape outShape, bool outC)
            => !s.IsBroadcasted && SameDims(s, outShape) && (outC ? s.IsContiguous : s.IsFContiguous);

        /// <summary>
        ///     The overlap half of NumPy's trivial-loop test, restricted to the two cases a single forward kernel pass
        ///     is safe for: the input and the output share no memory, or (an array input only) the input IS the
        ///     output — the same first element, which with the same dimensions and contiguity means the same strides.
        /// </summary>
        /// <param name="input">The input operand.</param>
        /// <param name="out">The output operand.</param>
        /// <param name="isArray">True for an array input walked alongside the output, false for a size-1 input read
        /// once before the loop.</param>
        /// <returns>True when the kernel may run on the operands as they are.</returns>
        private static unsafe bool TrivialOverlapOk(NDArray input, NDArray @out, bool isArray)
        {
            // Compare first ELEMENTS, not (Address, offset) pairs: a contiguous slice re-seats Storage.Address with
            // offset 0 while another view of the same element may keep the base address and a non-zero offset.
            if (isArray
                && (byte*)input.Address + input.Shape.offset * input.dtypesize
                   == (byte*)@out.Address + @out.Shape.offset * @out.dtypesize)
                return true;
            // Bounds first, then NumPy's bounded solver (max_work 1, as PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK asks).
            return NDMemOverlap.SolveMayShareMemory(input, @out, maxWork: 1) == MemOverlap.No;
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

            // NumPy's cast loop hands the converted values to the same elementwise loop: the second operand's NaN wins.
            DirectILKernelGenerator.EmitScalarOperation(il, op, resultType, NaNOperandOrder.NumPyUfunc);
        }

        /// <summary>
        ///     Decides how NumPy 2.4.2 would split a binary ufunc iteration into inner-loop calls, for the one thing that
        ///     depends on it: which NaN a float add/multiply returns when its broadcast (stride-0) operand is NaN
        ///     (<see cref="DirectILKernelGenerator.EmitBroadcastNaNRegion"/>). Every other result is unaffected.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     A rule is returned only when NumPy's calls provably coincide with NumSharp's kernel calls: exactly one input of
        ///     size 1 (stride 0 everywhere), and NumSharp's iterator walking ONE 1-D run (all axes coalesced) in memory order
        ///     without flipping an axis — the run NumPy also makes, as its trivial loop (<c>try_trivial_single_output_loop</c>),
        ///     an unbuffered 1-D iterator, or one mask-true run of either (<c>generic_masked_strided_loop</c>).
        ///     </para>
        ///     <para>
        ///     Inside such a call NumPy takes its SIMD broadcast path only when the array input and the output are
        ///     element-contiguous — or are the contiguous copy/buffer NumPy makes of a cast operand. A cast ARRAY or OUTPUT is
        ///     buffered in <c>np.getbufsize()</c>-element chunks (<c>npyiter_find_buffering_setup</c>; a cast 1-D input of at
        ///     most that size is copied whole by <c>check_for_trivial_loop</c>, one chunk either way); a cast 0-d input is
        ///     copied whole, so it never chunks. A reversed array (negative stride) is refused: NumPy's iterator may flip it
        ///     where NumSharp's does not (NumSharp allocates the output before iterating). Buffered chunks of a MASKED call are
        ///     not modeled either. Refused or unmodeled cases get the default rule — the second operand's NaN — which is also
        ///     what NumPy does whenever it buffers a broadcast operand of a multi-axis iteration.
        ///     </para>
        /// </remarks>
        /// <param name="op">The binary op (only float add/multiply get a non-default rule).</param>
        /// <param name="lhsType">The first input's stored dtype.</param>
        /// <param name="rhsType">The second input's stored dtype.</param>
        /// <param name="resultType">The loop dtype.</param>
        /// <param name="outType">The output array's dtype (differs from the loop dtype for a cast out=).</param>
        /// <param name="lhsSize">The first input's element count (1 = broadcast with stride 0).</param>
        /// <param name="rhsSize">The second input's element count.</param>
        /// <param name="masked">True for a <c>where=</c> iteration (calls are mask-true runs).</param>
        /// <param name="iterNDim">The iterator's dimension count after coalescing.</param>
        /// <param name="negPerm">Whether the iterator flipped a negative-stride axis.</param>
        /// <param name="innerElementStrides">The iterator's per-operand inner strides in ELEMENTS (input 0, input 1, output, …).</param>
        /// <returns>The rule to pass as the kernel's <c>auxdata</c>; <c>default</c> keeps the second-operand rule.</returns>
        internal static unsafe BroadcastNaNRule NumPyBroadcastNaNRule(
            BinaryOp op, NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode resultType, NPTypeCode outType,
            long lhsSize, long rhsSize, bool masked, int iterNDim, bool negPerm, long* innerElementStrides)
        {
            if (!DirectILKernelGenerator.HasNaNOperandOrder(op, resultType))
                return default;

            bool lhsScalar = lhsSize == 1, rhsScalar = rhsSize == 1;
            if (lhsScalar == rhsScalar)
                return default;                                   // no broadcast operand, or two (NumPy's scalar loop)
            if (iterNDim != 1 || negPerm || innerElementStrides is null)
                return default;                                   // not one run in memory order: NumPy's calls differ

            int arrayOp = lhsScalar ? 1 : 0;
            long arrayStride = innerElementStrides[arrayOp];
            long outStride = innerElementStrides[2];
            if (arrayStride <= 0)
                return default;

            bool arrayCast = (lhsScalar ? rhsType : lhsType) != resultType;
            bool outCast = outType != resultType;

            // NumPy's SIMD broadcast path needs a contiguous array input and output; a cast operand is a contiguous copy.
            if (!arrayCast && arrayStride != 1)
                return default;
            if (!outCast && outStride != 1)
                return default;

            // A cast 0-d input alone is copied whole by check_for_trivial_loop, so it keeps the single call.
            if (!arrayCast && !outCast)
                return BroadcastNaNRule.PerCall();                // trivial loop / unbuffered iterator: one call (per mask run)
            if (masked)
                return default;                                   // buffered chunks crossed with mask runs: not modeled
            return BroadcastNaNRule.PerCall(np.getbufsize());
        }

        /// <summary>
        ///     True when <paramref name="s"/> presents a clean C- or F-contiguous
        ///     layout (or is scalar/size-1, trivially contiguous). Gate for the
        ///     small-broadcast direct-path route: the direct SimdChunk kernel is only
        ///     verified bit-identical to NDIter when both operands are contiguous —
        ///     a strided / trailing-size-1 / negative-stride operand (which NDIter
        ///     absorbs but SimdChunk mishandles) must stay on the NDIter route.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool IsContiguousCorF(Shape s)
            => s.size <= 1 || s.IsContiguous || s.IsFContiguous;

        /// <summary>
        /// NumPy-aligned rule: the output is F-contiguous when every non-scalar operand
        /// is strictly F-contiguous (IsFContiguous && !IsContiguous).
        /// Scalars (and 1-element shapes, both C and F) do not change the decision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        ///     True when a non-scalar operand of a binary op broadcasts to the result shape, i.e.
        ///     its own dimensions differ from <paramref name="resultShape"/>'s (a scalar or any
        ///     one-element operand does not count: it broadcasts as a scalar the kernels read once).
        ///     The direct kernels (<see cref="ExecuteKernelCoalesced"/>) write the result linearly,
        ///     in C-coordinate order over the coalesced dims, so an F-allocated result is correct
        ///     only when every dim coalesces into one run. Strictly F operands of the result's own
        ///     dims always do; a broadcast operand's stride-0 axis blocks the merge, so the direct
        ///     path must not allocate F for such a set (see <see cref="ExecuteBinaryOp"/>).
        /// </summary>
        /// <param name="lhs">The left operand, before broadcasting.</param>
        /// <param name="rhs">The right operand, before broadcasting.</param>
        /// <param name="resultShape">The broadcast result shape.</param>
        /// <returns>True when a non-scalar operand's dimensions differ from the result's.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static bool HasNonScalarBroadcastOperand(NDArray lhs, NDArray rhs, Shape resultShape)
        {
            bool lhsScalar = lhs.Shape.IsScalar || lhs.Shape.size <= 1;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.Shape.size <= 1;
            return (!lhsScalar && !SameDims(lhs.Shape, resultShape))
                || (!rhsScalar && !SameDims(rhs.Shape, resultShape));
        }

        /// <summary>
        /// Binary variant — require that every non-scalar operand is strictly F-contiguous
        /// and at least one of them is (otherwise the scalar+scalar case is excluded upstream).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        ///     Build the result <see cref="Shape"/> for a trivial-loop bypass without the
        ///     redundant dims-clone + strides-alloc + flag/size/stride walks that
        ///     <c>new Shape(dims[, 'F'])</c> performs.
        ///
        ///     The bypass result must be a clean, offset-0, owns-its-buffer layout whose
        ///     dimensions match <paramref name="src"/> in the requested order. When
        ///     <paramref name="src"/> is ALREADY canonical — <c>offset == 0</c>, the backing
        ///     buffer is exactly <c>size</c> elements (<c>bufferSize == size</c>, i.e. not a
        ///     window into a larger parent) and it is contiguous in the target order — it is
        ///     byte-for-byte what the constructor would produce. Because <see cref="Shape"/>
        ///     is an immutable readonly struct whose <c>dimensions</c>/<c>strides</c> arrays
        ///     are never mutated after construction, the result can share it verbatim, and we
        ///     skip: the <c>dimensions.Clone()</c>, the <c>ComputeContiguousStrides</c>
        ///     allocation, and the four array walks (strides + size/hash +
        ///     <c>ComputeFlagsStatic</c>'s C/F/broadcast passes).
        ///
        ///     A sliced source (<c>offset != 0</c>) or a view into a larger buffer
        ///     (<c>bufferSize &gt; size</c>) must NOT be reused — the freshly-allocated result
        ///     starts at 0 and owns exactly <c>size</c> elements, so inheriting the source's
        ///     offset/bufferSize would mis-describe it. Those fall back to building a fresh
        ///     canonical Shape exactly as before.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static Shape CanonicalResultShape(Shape src, bool wantF)
        {
            if (src.offset == 0 && src.bufferSize == src.size)
            {
                // 1-D arrays are both C- and F-contiguous; the C check (taken first for
                // wantF==false) returns them, so the F branch only fires for strictly
                // column-major (ndim > 1) sources — matching new Shape(dims, 'F').
                if (!wantF)
                {
                    if (src.IsContiguous) return src;
                }
                else if (src.IsFContiguous && !src.IsContiguous)
                {
                    return src;
                }
            }

            var dims = (long[])src.dimensions.Clone();
            return wantF ? Shape.Adopt(dims, 'F') : Shape.Adopt(dims);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
            fixed (long* shape = result.Shape.dimensions)
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
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
                    // Use the pre-merge shapeW/shapeR locals (shape[writeAxis] is overwritten
                    // below) so each operand's merged stride is resolved against the real axis
                    // sizes — a size-1 axis must defer to its neighbour's stride.
                    lhsStrides[writeAxis] = MergeStride(shapeW, lhsStrides[writeAxis], shapeR, lhsStrides[readAxis]);
                    rhsStrides[writeAxis] = MergeStride(shapeW, rhsStrides[writeAxis], shapeR, rhsStrides[readAxis]);
                    resStrides[writeAxis] = MergeStride(shapeW, resStrides[writeAxis], shapeR, resStrides[readAxis]);
                    shape[writeAxis] = shapeW * shapeR;
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool AgreeOnOrder(MergeMode a, MergeMode b, MergeMode c)
        {
            bool hasC = a == MergeMode.COrder || b == MergeMode.COrder || c == MergeMode.COrder;
            bool hasF = a == MergeMode.FOrder || b == MergeMode.FOrder || c == MergeMode.FOrder;
            return !(hasC && hasF);
        }

        /// <summary>
        ///     The stride of an axis formed by merging an outer dim (<paramref name="shapeW"/>,
        ///     <paramref name="strideW"/>) with the adjacent inner dim (<paramref name="shapeR"/>,
        ///     <paramref name="strideR"/>). A valid merge (validated by
        ///     <see cref="ClassifyMergePair"/>) is C-order (<c>strideW == strideR*shapeR</c>) or
        ///     F-order (<c>strideR == strideW*shapeW</c>); the merged axis walks the pair in that
        ///     order, so its stride is the INNER/fastest one — the stride of SMALLER MAGNITUDE.
        ///
        ///     The prior <c>strideW &lt; strideR</c> (smaller VALUE) was correct only for
        ///     non-negative strides: for a reversed view whose merged pair is e.g. (-24, -8),
        ///     smaller-value keeps -24 (the OUTER stride) where the merged axis must step by -8,
        ///     so the kernel then read the wrong elements (silent garbage on float16 — the sole
        ///     dtype routed to this direct path; every other dtype coalesces inside NDIter). A
        ///     size-1 axis contributes no offset (its coord is always 0), so the merged axis is
        ///     driven entirely by the OTHER axis's stride regardless of magnitude.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static long MergeStride(long shapeW, long strideW, long shapeR, long strideR)
        {
            if (shapeW == 1) return strideR;   // outer phantom → inner drives
            if (shapeR == 1) return strideW;   // inner phantom → outer drives
            if (strideW == 0 && strideR == 0) return 0;
            if (strideW == 0) return strideR;
            if (strideR == 0) return strideW;
            return Math.Abs(strideW) <= Math.Abs(strideR) ? strideW : strideR;
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
