using System;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Comparison operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute a comparison operation using IL-generated kernels.
        /// Handles type promotion, broadcasting, and kernel dispatch.
        /// Result is always NDArray&lt;bool&gt;.
        /// </summary>
        /// <param name="lhs">Left operand</param>
        /// <param name="rhs">Right operand</param>
        /// <param name="op">Comparison operation to perform</param>
        /// <returns>Result array with bool type</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe NDArray<bool> ExecuteComparisonOp(NDArray lhs, NDArray rhs, ComparisonOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;

            // Handle scalar × scalar case
            if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
            {
                return ExecuteComparisonScalarScalar(lhs, rhs, op);
            }

            // -------- O(1) trivial-loop bypass -----------------------------
            // Same rationale as the binary route (NumPy check_for_trivial_loop):
            // equal-shape contiguous (both C or both F) and array-vs-scalar cases
            // route straight to the DirectIL SimdFull / SimdScalar comparison
            // kernel, skipping NDIter construction. Unlike the binary bypass this
            // allows mixed dtypes — comparison promotes per element inside the
            // kernel with no cast temp, and the result is always bool. Returns null
            // for broadcast/mixed-C-F/strided/unsupported → NDIter route below.
            {
                var trivial = TryTrivialContiguousComparisonOp(lhs, rhs, op, lhsType, rhsType);
                if (trivial is not null) return trivial;
            }

            // -------- NDIter Tier 3B fast path (all comparison ops) -----------
            // Mirrors the binary-op routing pattern. The iterator coalesces
            // dimensions, normalizes negative strides, and resolves stride=0
            // broadcast operands into per-outer-iter pointer advance — turning
            // the (M,N) op (M,1) broadcast from a per-element gather into a
            // contig inner-loop kernel of size N, repeated M times. Closes
            // 14-30× regressions on strided/broadcast comparison variants vs
            // the direct path. Contig same-dtype is at parity with the direct
            // path because the same SIMD body lives inside the inner kernel.
            {
                var routed = TryExecuteComparisonOpViaNDIter(lhs, rhs, op, lhsType, rhsType);
                if (routed is not null) return routed;
            }

            // Broadcast shapes
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var resultShape = leftShape.Clean();

            // Allocate result (always bool)
            var result = new NDArray<bool>(resultShape, true);

            // Classify execution path using strides
            ExecutionPath path;
            fixed (long* lhsStrides = leftShape.strides)
            fixed (long* rhsStrides = rightShape.strides)
            fixed (long* shape = resultShape.dimensions)
            {
                path = ClassifyPath(lhsStrides, rhsStrides, shape, resultShape.NDim, NPTypeCode.Boolean);
            }

            // Get kernel key
            var key = new ComparisonKernelKey(lhsType, rhsType, op, path);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.GetComparisonKernel(key);

            if (kernel != null)
            {
                // Execute IL kernel
                ExecuteComparisonKernel(kernel, lhs, rhs, result, leftShape, rightShape);
            }
            else
            {
                // Fallback - should not happen
                throw new NotSupportedException(
                    $"IL kernel not available for comparison {lhsType} {op} {rhsType}. " +
                    "Please report this as a bug.");
            }

            // NumPy-aligned layout preservation: comparisons preserve F-contig.
            // copy('F') returns an NDArray; wrap it back as NDArray<bool> via MakeGeneric.
            if (ShouldProduceFContigOutput(lhs, rhs, result.Shape))
                return result.copy('F').MakeGeneric<bool>();

            return result;
        }

        /// <summary>
        ///     Comparison arm of the trivial-loop bypass (see the binary
        ///     <c>TryTrivialContiguousBinaryOp</c>). Handles two trivially
        ///     iterable shapes — equal-shape contiguous (both C or both F → one
        ///     linear SimdFull walk) and array-vs-scalar (scalar read once, array
        ///     walked linearly via SimdScalarLeft/Right) — and routes to the
        ///     existing DirectIL comparison kernel, skipping NDIter construction.
        ///     Dtypes may differ (the kernel promotes per element; result is always
        ///     bool). The bool result takes the array operand's layout (C, or
        ///     strictly-F) so the linear write matches the linear read, matching the
        ///     post-kernel <see cref="ShouldProduceFContigOutput(NDArray, NDArray, Shape)"/>
        ///     branch. Returns null (→ NDIter) for broadcast, mixed C/F, strided,
        ///     or unsupported emit.
        /// </summary>
        private unsafe NDArray<bool>? TryTrivialContiguousComparisonOp(
            NDArray lhs, NDArray rhs, ComparisonOp op, NPTypeCode lhsType, NPTypeCode rhsType)
        {
            var ls = lhs.Shape;
            var rs = rhs.Shape;

            bool lhsScalarLike = ls.IsScalar || ls.size == 1;
            bool rhsScalarLike = rs.IsScalar || rs.size == 1;

            ExecutionPath path;
            Shape arrShape;   // operand whose shape+layout the result follows
            if (lhsScalarLike ^ rhsScalarLike)
            {
                // Scalar-broadcast: the array operand drives the result.
                var array = rhsScalarLike ? lhs : rhs;
                arrShape = array.Shape;
                if (arrShape.IsBroadcasted)
                    return null;
                if (!arrShape.IsContiguous && !arrShape.IsFContiguous)
                    return null;   // strided/transposed array operand → NDIter
                path = rhsScalarLike ? ExecutionPath.SimdScalarRight : ExecutionPath.SimdScalarLeft;
            }
            else
            {
                // Equal-shape (both arrays, or both size-1). Same dims, one shared
                // contiguous layout.
                if (!ls.Equals(rs))
                    return null;
                if (ls.IsBroadcasted || rs.IsBroadcasted)
                    return null;
                bool bothC = ls.IsContiguous && rs.IsContiguous;
                bool bothF = !bothC && ls.IsFContiguous && rs.IsFContiguous;
                if (!bothC && !bothF)
                    return null;
                path = ExecutionPath.SimdFull;
                arrShape = bothF ? rs : ls;
            }

            var key = new ComparisonKernelKey(lhsType, rhsType, op, path);
            ComparisonKernel kernel;
            try
            {
                kernel = DirectILKernelGenerator.GetComparisonKernel(key);
            }
            catch (NotSupportedException)
            {
                return null;
            }
            if (kernel == null)
                return null;

            // Strictly-F (ndim > 1 column-major) → F result; C or 1-D → C result.
            bool isF = arrShape.IsFContiguous && !arrShape.IsContiguous;
            Shape resultShape = CanonicalResultShape(arrShape, isF);
            // The SimdFull / SimdScalar comparison kernels write every output byte
            // (4×-unrolled SIMD + remainder + scalar tail span [0, size)), so the
            // zero-fill is dead work for the bypass — allocate without it.
            var result = new NDArray<bool>(resultShape, false);
            if (result.size == 0)
                return result;

            ExecuteComparisonKernel(kernel, lhs, rhs, result, lhs.Shape, rhs.Shape);
            return result;
        }

        /// <summary>
        /// Execute scalar × scalar comparison using IL-generated delegate.
        /// </summary>
        private NDArray<bool> ExecuteComparisonScalarScalar(NDArray lhs, NDArray rhs, ComparisonOp op)
        {
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            var key = new ComparisonScalarKernelKey(lhsType, rhsType, op);
            var func = DirectILKernelGenerator.GetComparisonScalarDelegate(key);

            // Dispatch based on lhs type first
            return lhsType switch
            {
                NPTypeCode.Boolean => InvokeComparisonScalarLhs(func, lhs.GetBoolean(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.SByte => InvokeComparisonScalarLhs(func, lhs.GetSByte(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Byte => InvokeComparisonScalarLhs(func, lhs.GetByte(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int16 => InvokeComparisonScalarLhs(func, lhs.GetInt16(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt16 => InvokeComparisonScalarLhs(func, lhs.GetUInt16(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int32 => InvokeComparisonScalarLhs(func, lhs.GetInt32(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt32 => InvokeComparisonScalarLhs(func, lhs.GetUInt32(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Int64 => InvokeComparisonScalarLhs(func, lhs.GetInt64(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.UInt64 => InvokeComparisonScalarLhs(func, lhs.GetUInt64(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Char => InvokeComparisonScalarLhs(func, lhs.GetChar(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Half => InvokeComparisonScalarLhs(func, lhs.GetHalf(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Single => InvokeComparisonScalarLhs(func, lhs.GetSingle(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Double => InvokeComparisonScalarLhs(func, lhs.GetDouble(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Decimal => InvokeComparisonScalarLhs(func, lhs.GetDecimal(Array.Empty<long>()), rhs, rhsType),
                NPTypeCode.Complex => InvokeComparisonScalarLhs(func, lhs.GetComplex(Array.Empty<long>()), rhs, rhsType),
                _ => throw new NotSupportedException($"LHS type {lhsType} not supported")
            };
        }

        /// <summary>
        /// Continue comparison scalar dispatch with typed LHS value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static NDArray<bool> InvokeComparisonScalarLhs<TLhs>(
            Delegate func, TLhs lhsVal, NDArray rhs, NPTypeCode rhsType)
        {
            // Dispatch based on rhs type
            return rhsType switch
            {
                NPTypeCode.Boolean => NDArray.Scalar(((Func<TLhs, bool, bool>)func)(lhsVal, rhs.GetBoolean(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.SByte => NDArray.Scalar(((Func<TLhs, sbyte, bool>)func)(lhsVal, rhs.GetSByte(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Byte => NDArray.Scalar(((Func<TLhs, byte, bool>)func)(lhsVal, rhs.GetByte(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Int16 => NDArray.Scalar(((Func<TLhs, short, bool>)func)(lhsVal, rhs.GetInt16(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.UInt16 => NDArray.Scalar(((Func<TLhs, ushort, bool>)func)(lhsVal, rhs.GetUInt16(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Int32 => NDArray.Scalar(((Func<TLhs, int, bool>)func)(lhsVal, rhs.GetInt32(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.UInt32 => NDArray.Scalar(((Func<TLhs, uint, bool>)func)(lhsVal, rhs.GetUInt32(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Int64 => NDArray.Scalar(((Func<TLhs, long, bool>)func)(lhsVal, rhs.GetInt64(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.UInt64 => NDArray.Scalar(((Func<TLhs, ulong, bool>)func)(lhsVal, rhs.GetUInt64(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Char => NDArray.Scalar(((Func<TLhs, char, bool>)func)(lhsVal, rhs.GetChar(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Half => NDArray.Scalar(((Func<TLhs, Half, bool>)func)(lhsVal, rhs.GetHalf(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Single => NDArray.Scalar(((Func<TLhs, float, bool>)func)(lhsVal, rhs.GetSingle(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Double => NDArray.Scalar(((Func<TLhs, double, bool>)func)(lhsVal, rhs.GetDouble(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Decimal => NDArray.Scalar(((Func<TLhs, decimal, bool>)func)(lhsVal, rhs.GetDecimal(Array.Empty<long>()))).MakeGeneric<bool>(),
                NPTypeCode.Complex => NDArray.Scalar(((Func<TLhs, System.Numerics.Complex, bool>)func)(lhsVal, rhs.GetComplex(Array.Empty<long>()))).MakeGeneric<bool>(),
                _ => throw new NotSupportedException($"RHS type {rhsType} not supported")
            };
        }

        /// <summary>
        /// Execute the IL-generated comparison kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ExecuteComparisonKernel(
            ComparisonKernel kernel,
            NDArray lhs, NDArray rhs, NDArray<bool> result,
            Shape lhsShape, Shape rhsShape)
        {
            // Get element sizes for offset calculation
            int lhsElemSize = lhs.dtypesize;
            int rhsElemSize = rhs.dtypesize;

            // Calculate base addresses accounting for shape offsets (for sliced views)
            byte* lhsAddr = (byte*)lhs.Address + lhsShape.offset * lhsElemSize;
            byte* rhsAddr = (byte*)rhs.Address + rhsShape.offset * rhsElemSize;

            fixed (long* lhsStrides = lhsShape.strides)
            fixed (long* rhsStrides = rhsShape.strides)
            fixed (long* shape = result.shape)
            {
                kernel(
                    (void*)lhsAddr,
                    (void*)rhsAddr,
                    (bool*)result.Address,
                    lhsStrides,
                    rhsStrides,
                    shape,
                    result.ndim,
                    result.size
                );
            }
        }

        #region Public API - Comparison Operations (TensorEngine overrides)

        // ---- ONE NumPy-shaped override per comparison ufunc ----------------
        // Merged bare + out=/where= forms (no duplicate overloads). The bare
        // path (out == null && where == null) keeps the existing trivial/SIMD
        // ladder via ExecuteComparisonOp and returns an NDArray<bool> instance
        // (TensorEngine contract — the C# operators cast it back for free).
        // The out=/where= path routes straight to the Into-path: NumPy's
        // masked execution never takes the trivial loop (ufunc_object.c:2213),
        // and a provided out needs reference identity + write-masking, so the
        // scalar×scalar / trivial-bypass arms of the ladder are skipped by
        // design (0-d EXLOOP works post-Wave-2.1). No F-layout post-step.
        // typeCode is validate-only: comparisons have bool-output loops ONLY —
        // NumPy raises the no-loop TypeError for any non-bool dtype= (probed
        // 2.4.2: equal(a, b, dtype=f64/i32) raises; dtype=bool is a no-op).

        public override NDArray Compare(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "equal");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.Equal);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.Equal, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        public override NDArray NotEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "not_equal");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.NotEqual);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.NotEqual, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        public override NDArray Less(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "less");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.Less);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.Less, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        public override NDArray LessEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "less_equal");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.LessEqual);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.LessEqual, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        public override NDArray Greater(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "greater");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.Greater);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.Greater, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        public override NDArray GreaterEqual(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            ValidateBoolLoopDtype(typeCode, "greater_equal");
            if (@out is null && where is null)
                return ExecuteComparisonOp(lhs, rhs, ComparisonOp.GreaterEqual);
            return ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp.GreaterEqual, lhs.GetTypeCode, rhs.GetTypeCode, @out, where);
        }

        #endregion

        /// <summary>
        ///     Tier 3B NDIter routing for comparison ops. Output is always bool.
        ///
        ///     Returns the result NDArray&lt;bool&gt; on success, or null when the
        ///     iterator can't handle the shapes (e.g. > int.MaxValue dimension) so
        ///     the caller falls back to the direct path.
        ///
        ///     ─── Why this beats the direct path on broadcast/strided ───
        ///     The direct ComparisonKernel walks the output via coordinate math
        ///     (one ndim-loop per output element). For (M,N) op (M,1) it does M*N
        ///     full coord recomputes and gather loads. NDIter coalesces the
        ///     M*N output into the canonical iteration order, sets the broadcast
        ///     operand's stride to 0, and dispatches a contig inner kernel of
        ///     length N for each of M outer steps — the inner kernel becomes a
        ///     tight scalar/SIMD loop without coord math.
        ///
        ///     ─── Scalar vs vector body ───
        ///     For same-dtype with a SIMD-capable comparison type, the vector
        ///     body would produce the mask + PDEP-store fast path that the
        ///     direct kernel already does. The Tier 3B 4×-unrolled wrapper
        ///     however requires same-type-across-all-operands (CanSimdAllOperands
        ///     in the InnerLoop factory) which doesn't hold here because the
        ///     output is bool. We pass null for the vector body; the iterator
        ///     drops to a per-element scalar loop driven by the emitted body.
        ///     This trades some contig perf for huge wins on strided/broadcast.
        /// </summary>
        private unsafe NDArray<bool>? TryExecuteComparisonOpViaNDIter(
            NDArray lhs, NDArray rhs, ComparisonOp op,
            NPTypeCode lhsType, NPTypeCode rhsType)
        {
            // ─── Routing decision: NDIter wins on broadcast/strided, the
            // direct SIMD kernel wins on plain contig same-shape.
            //
            // The direct ComparisonKernel has a vector body that emits
            // Vector256.Compare + ExtractMostSignificantBits + PDEP-packed
            // store — the fastest contig path we have. The Tier 3B 4×-unrolled
            // wrapper can't host that because bool-output breaks its same-type
            // invariant, so routing contig through here costs us 3× on the
            // simple shape.
            //
            // Skip NDIter routing for the "easy" contig cases that the direct
            // kernel handles best. Anything with broadcast (different shapes)
            // or non-contig storage benefits from the iterator's coalesce +
            // stride normalization + per-outer-iter pointer advance.
            bool sameShape = lhs.Shape.NDim == rhs.Shape.NDim
                             && lhs.Shape.size == rhs.Shape.size;
            if (sameShape && lhs.Shape.IsContiguous && rhs.Shape.IsContiguous)
                return null;

            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // NDIter shape arithmetic is int-bounded.
            if (cleanShape.size < 0) return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue) return null;

            // Mirror binary-op F-preservation: F-allocate when every non-scalar
            // operand is strict-F; else default to C and let the post-kernel
            // looser-F copy step rectify.
            bool allStrictFContig = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            Shape resultShape = allStrictFContig
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            var result = new NDArray<bool>(resultShape, true);

            var order = allStrictFContig
                ? NPY_ORDER.NPY_FORTRANORDER
                : NPY_ORDER.NPY_CORDER;

            // Resolve common comparison type once (same rule as the kernel key).
            var comparisonType = lhsType == rhsType
                ? lhsType
                : np._FindCommonScalarType(lhsType, rhsType);

            // Per-element body. Stack on entry: [lhs (lhsType), rhs (rhsType)].
            // Stash rhs into a local so we can convert lhs (bottom of stack)
            // first, then reload rhs and convert it, matching the mixed-binary
            // pattern. Then EmitComparisonOperation pops both and pushes bool.
            // NOTE (Wave 4, measured): the buffered-cast route was tried here
            // and reverted — comparisons are add-class cheap ops where the
            // fused per-element convert beats the buffer round-trip (see the
            // A/B table in DefaultEngine.BinaryOp). Promoting unary math keeps
            // the buffered route (DefaultEngine.UnaryOp).
            NPTypeCode capLhs = lhsType, capRhs = rhsType, capCmp = comparisonType;
            ComparisonOp capOp = op;
            Action<ILGenerator> scalarBody = il =>
            {
                if (capLhs == capRhs && capLhs == capCmp)
                {
                    // Same-dtype fast path — no convert pass.
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
                else
                {
                    var locRhs = il.DeclareLocal(DirectILKernelGenerator.GetClrType(capRhs));
                    il.Emit(OpCodes.Stloc, locRhs);
                    if (capLhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capLhs, capCmp);
                    il.Emit(OpCodes.Ldloc, locRhs);
                    if (capRhs != capCmp)
                        DirectILKernelGenerator.EmitConvertTo(il, capRhs, capCmp);
                    DirectILKernelGenerator.EmitComparisonOperation(il, capOp, capCmp);
                }
            };

            // Vector body intentionally null: the Tier 3B 4×-unrolled wrapper
            // requires same-dtype-across-all-operands and bool output breaks
            // that invariant. Inner-loop factory falls to scalar-strided body.
            string cacheKey = $"npy_cmp_{op}_{lhsType}_{rhsType}";

            try
            {
                // COPY_IF_OVERLAP + OVERLAP_ASSUME_ELEMENTWISE per NumPy's ufunc
                // iterator flags (ufunc_object.c:1070); cheap extent check here
                // since the bool result is freshly allocated.
                using var iter = NDIterRef.MultiNew(
                    3, new[] { lhs, rhs, result },
                    NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP,
                    order,
                    NPY_CASTING.NPY_SAFE_CASTING,
                    new[]
                    {
                        NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                        NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                        NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
                    });

                iter.ExecuteElementWiseBinary(lhsType, rhsType, NPTypeCode.Boolean, scalarBody, null, cacheKey);
            }
            catch (NotSupportedException)
            {
                return null;
            }

            if (!allStrictFContig && ShouldProduceFContigOutput(lhs, rhs, result.Shape))
                return result.copy('F').MakeGeneric<bool>();

            return result;
        }
    }
}
