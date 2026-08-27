using System;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bit shift operations: left_shift and right_shift.
    ///
    /// NumPy alignment (probed against NumPy 2.4.2): both are integer ufuncs whose loops are
    /// all same-type (<c>bb-&gt;b</c> .. <c>QQ-&gt;Q</c>). Mixed operands therefore promote to
    /// <c>result_type(x1, x2)</c> and the shift runs at that width; bool inputs (no bool loop)
    /// promote to int8. The op is wired into the shared binary pipeline
    /// (<see cref="DefaultEngine.ExecuteBinaryOp"/>) so promotion, broadcasting, strided/sliced
    /// views and scalar×scalar all flow through NDIter + the IL scalar kernel — the per-element
    /// shift IL lives in <see cref="DirectILKernelGenerator.EmitShiftFromStack"/>. The common
    /// <c>array &lt;&lt; scalar</c> case takes a dedicated 4×-unrolled SIMD kernel
    /// (<see cref="DirectILKernelGenerator.GetShiftScalarKernel{T}"/>).
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-count cutoff below which a <c>bool_array &lt;&lt; scalar</c> shift is served by
        /// widening the bool to the promoted dtype (existing SIMD <c>astype</c>) and running the
        /// vectorized uniform-count shift kernel, instead of the single-pass fused-SCALAR NDIter
        /// route. The fused-scalar route has no per-lane SIMD body (the bool value operand is not
        /// the same dtype as the wide count/result, so <c>CanSimdAllOperands</c> drops the vector
        /// body), so its scalar inner loop loses to astype+SIMD for small/medium arrays — measured
        /// 100K <c>bool&lt;&lt;2</c> ≈ 65 µs (fused-scalar) vs ≈ 44 µs (astype+SIMD), where NumPy's
        /// own bool shift is ≈ 42.5 µs (so the scalar route is ~0.65× while astype+SIMD reaches
        /// ~0.97×). The cutoff keeps the widened int64 buffer L2-resident (256K × 8 B = 2 MB), so
        /// the two-pass traffic stays in cache; above it the second pass spills to RAM and the
        /// single-pass fused-scalar route wins (10M <c>bool&lt;&lt;2</c> ≈ 13.9 ms vs ≈ 27 ms for
        /// two passes over the 80 MB int64 buffer). Env override NS_BOOL_SHIFT_SIMD_MAX for tuning.
        /// </summary>
        internal static readonly long BoolSimdScalarShiftMaxElements =
            long.TryParse(Environment.GetEnvironmentVariable("NS_BOOL_SHIFT_SIMD_MAX"), out var t) && t >= 0
                ? t : 262144;

        /// <summary>
        /// Bitwise left shift (x1 &lt;&lt; x2).
        /// </summary>
        public override NDArray LeftShift(NDArray lhs, NDArray rhs)
        {
            ValidateShiftType(lhs, "left_shift");
            ValidateShiftType(rhs, "left_shift");
            return ExecuteShift(lhs, rhs, isLeftShift: true);
        }

        /// <summary>
        /// Bitwise right shift (x1 &gt;&gt; x2).
        /// Arithmetic shift for signed types (sign bit extended); logical shift for unsigned.
        /// </summary>
        public override NDArray RightShift(NDArray lhs, NDArray rhs)
        {
            ValidateShiftType(lhs, "right_shift");
            ValidateShiftType(rhs, "right_shift");
            return ExecuteShift(lhs, rhs, isLeftShift: false);
        }

        /// <summary>
        /// Validate that the array dtype has a shift loop. NumPy's left_shift/right_shift loops
        /// cover bool and the integer dtypes; bool promotes to int8 (handled in
        /// <see cref="ExecuteBinaryOp"/>). Char rides along as a NumSharp integer extension.
        /// Float/complex/decimal raise NumPy's verbatim no-loop TypeError.
        /// </summary>
        private static void ValidateShiftType(NDArray arr, string opName)
        {
            var typeCode = arr.GetTypeCode;
            if (typeCode.IsInteger() || typeCode == NPTypeCode.Boolean || typeCode == NPTypeCode.Char)
                return;

            throw new TypeError($"ufunc '{opName}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''");
        }

        /// <summary>
        /// Resolve a shift through the shared binary pipeline. The hot <c>array &lt;&lt; scalar</c>
        /// case is intercepted by the SIMD kernel; everything else (mixed dtype, strided,
        /// broadcast, scalar×scalar) flows through <see cref="ExecuteBinaryOp"/>, which handles
        /// NEP50 promotion and drives the per-element shift IL via NDIter.
        /// </summary>
        private unsafe NDArray ExecuteShift(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            // Algebraic fast path: `bool_array >> scalar` with a nonzero count is ALL ZEROS —
            // the loop dtype's value set is {0, 1} (NumPy casts bool normalized, probed 2.4.2),
            // so 1 >> s == 0 for s in [1, bits-1] and every overflow count (s < 0, s >= bits)
            // fills with the sign bit of a non-negative value, also 0. One zeroed allocation
            // replaces a full cast-shift-write pass (10M: ~0.001 ms vs NumPy's 14.9 ms).
            var boolZero = TryBoolScalarRightShiftZeros(lhs, rhs, isLeftShift);
            if (boolZero is not null)
                return boolZero;

            // Fast path: contiguous `array << scalar` — the dedicated uniform-count SIMD kernel
            // (covers every width incl. 8/16-bit via Vector{N}.ShiftLeft).
            var fast = TrySimdScalarShift(lhs, rhs, isLeftShift);
            if (fast is not null)
                return fast;

            // Everything else (array << array, strided, broadcast, transposed, mixed dtype) goes
            // through the NDIter Tier-3B kernel: a per-vector variable shift drives the factory's
            // 4×-unrolled contiguous, scalar-broadcast, and AVX2-gather strided SIMD paths, with a
            // scalar inner loop where no per-lane SIMD shift exists (8/16-bit, int64 arith-right
            // without AVX512).
            var viaIter = ExecuteShiftViaNDIter(lhs, rhs, isLeftShift);
            if (viaIter is not null)
                return viaIter;

            // Backstop for scalar×scalar and shapes beyond int range: the unified binary pipeline
            // (with the EmitShiftFromStack scalar kernel) handles them correctly.
            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            return ExecuteBinaryOp(lhs, rhs, op);
        }

        /// <summary>
        /// Drive the shift through the NDIter Tier-3B inner-loop factory. Operands are cast to
        /// the promoted loop dtype so the iterator sees one dtype (same-dtype views are kept
        /// strided so the factory's hardware-gather path can SIMD them without materializing).
        /// The vector body (<see cref="DirectILKernelGenerator.EmitShiftVectorBody"/>) is supplied
        /// when the dtype/direction has a per-lane variable shift; otherwise the factory uses the
        /// overflow-correct scalar body. Returns null for scalar×scalar and over-int-range shapes
        /// so the caller can fall back to the unified pipeline.
        /// </summary>
        private unsafe NDArray? ExecuteShiftViaNDIter(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            // scalar × scalar → let ExecuteBinaryOp's dedicated scalar path handle it.
            bool lhsScalar = lhs.Shape.IsScalar || lhs.size <= 1;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.size <= 1;
            if (lhsScalar && rhsScalar)
                return null;

            var resultType = ShiftResultType(lhs, rhs);

            // Cast inputs to the loop dtype (NumPy casts to the loop signature). Same-dtype
            // operands keep their view (possibly strided) — the gather SIMD path reads them in
            // place; a differing dtype materializes to a contiguous copy at the loop dtype.
            // EXCEPT a bool value operand: it always promotes (int64/int8), and materializing the
            // widened copy costs a full extra pass over the wide buffer. Keep it as bool and fuse
            // the normalize-convert into the kernel's per-element load (EmitMixedScalarBody), the
            // same one-pass structure the unified mixed-dtype binary path uses.
            bool fusedBoolValue = lhs.GetTypeCode == NPTypeCode.Boolean;
            var value = fusedBoolValue || lhs.GetTypeCode == resultType ? lhs : lhs.astype(resultType);
            var count = rhs.GetTypeCode == resultType ? rhs : rhs.astype(resultType);
            var valueLoopType = fusedBoolValue ? NPTypeCode.Boolean : resultType;

            var (valueShape, countShape) = Broadcast(value.Shape, count.Shape);
            var cleanShape = valueShape.Clean();
            if (cleanShape.size < 0)
                return null;
            for (int i = 0; i < cleanShape.NDim; i++)
                if (cleanShape.dimensions[i] > int.MaxValue)
                    return null;

            // Mirror the unified path's NumPy-aligned F-order preservation.
            bool allStrictFContig = AreAllOperandsStrictFContig(value, count, cleanShape);
            Shape resultShape = allStrictFContig
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;

            var result = new NDArray(resultType, resultShape, false);
            if (result.size == 0)
                return result;

            var order = allStrictFContig ? NPY_ORDER.NPY_FORTRANORDER : NPY_ORDER.NPY_CORDER;

            var capType = resultType;
            bool capLeft = isLeftShift;
            // Fused bool value: the scalar body converts the raw bool byte to the loop dtype
            // (EmitConvertTo's `!= 0` normalize — NumPy's bool cast) before the shift; the vector
            // body needs same-dtype operands (CanSimdAllOperands) so it is dropped. Non-bool keeps
            // the pre-cast same-dtype operands and the per-lane variable-shift SIMD body.
            Action<ILGenerator> scalarBody = fusedBoolValue
                ? il => EmitMixedScalarBody(il, NPTypeCode.Boolean, capType, capType,
                                            capLeft ? BinaryOp.LeftShift : BinaryOp.RightShift)
                : il => DirectILKernelGenerator.EmitShiftFromStack(il, capType, capLeft);
            Action<ILGenerator>? vectorBody = !fusedBoolValue &&
                                              DirectILKernelGenerator.ShiftVariableSupported(resultType, isLeftShift)
                ? il => DirectILKernelGenerator.EmitShiftVectorBody(il, capType, capLeft)
                : null;
            string cacheKey = $"npy_shift_{(isLeftShift ? "L" : "R")}_{valueLoopType}_{resultType}";

            using (var iter = NDIterRef.MultiNew(
                3, new[] { value, count, result },
                NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP,
                order, NPY_CASTING.NPY_SAFE_CASTING, s_binaryIterFlags))
            {
                iter.ExecuteElementWiseBinary(valueLoopType, resultType, resultType, scalarBody, vectorBody, cacheKey);
            }

            if (!allStrictFContig && ShouldProduceFContigOutput(value, count, result.Shape))
                return result.copy('F');

            return result;
        }

        /// <summary>
        /// NumPy shift promotion: the same-type loop selected for <c>result_type(lhs, rhs)</c>,
        /// with bool bumped to int8 (no bool shift loop). Mirrors the promotion
        /// <see cref="ExecuteBinaryOp"/> applies, so the SIMD fast path and the general path agree.
        /// </summary>
        private static NPTypeCode ShiftResultType(NDArray lhs, NDArray rhs)
        {
            var rt = np._FindCommonType(lhs, rhs);
            return rt == NPTypeCode.Boolean ? NPTypeCode.SByte : rt;
        }

        /// <summary>
        /// <c>bool_array &gt;&gt; scalar</c> with any count except 0 yields all zeros (see
        /// <see cref="ExecuteShift"/>): return a zero-filled result of the promoted dtype without
        /// running a kernel. The shape/order mirror the NDIter route exactly — broadcast with the
        /// count operand, F-allocated when the value operand is strictly F. A count of 0 keeps the
        /// fused cast-copy path (result == astype), and left shifts never take this route.
        /// </summary>
        private NDArray? TryBoolScalarRightShiftZeros(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            if (isLeftShift || lhs.GetTypeCode != NPTypeCode.Boolean)
                return null;
            bool rhsScalar = rhs.Shape.IsScalar || rhs.size == 1;
            bool lhsArray = !(lhs.Shape.IsScalar || lhs.size <= 1);
            if (!rhsScalar || !lhsArray)
                return null;

            var resultType = ShiftResultType(lhs, rhs);
            int bitWidth = resultType.SizeOf() * 8;
            int s = ReadSaturatedShiftCount(rhs, bitWidth);
            if (s == 0)
                return null;

            var (valueShape, _) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = valueShape.Clean();
            bool allStrictFContig = AreAllOperandsStrictFContig(lhs, rhs, cleanShape);
            Shape resultShape = allStrictFContig
                ? new Shape((long[])cleanShape.dimensions.Clone(), 'F')
                : cleanShape;
            return new NDArray(resultType, resultShape, fillZeros: true);
        }

        /// <summary>
        /// SIMD fast path for <c>contiguous array &lt;&lt; scalar</c>. The shift amount is uniform,
        /// so the overflow check is resolved once and the 4×-unrolled <c>Vector{N}.Shift*</c>
        /// kernel runs over the whole buffer. Returns null (→ <see cref="ExecuteBinaryOp"/>) when
        /// the shape is not array-vs-scalar, the value operand is non-contiguous, or the promoted
        /// dtype has no vector shift (Char).
        /// </summary>
        private unsafe NDArray TrySimdScalarShift(NDArray lhs, NDArray rhs, bool isLeftShift)
        {
            // Only array (value) << scalar (count). scalar×scalar and scalar<<array fall through.
            bool rhsScalar = rhs.Shape.IsScalar || rhs.size == 1;
            bool lhsArray = !(lhs.Shape.IsScalar || lhs.size <= 1);
            if (!rhsScalar || !lhsArray)
                return null;

            // A bool value operand always promotes (int64 against a weak int count, int8 against
            // bool). Taking this path materializes the whole promoted array via astype and then
            // re-reads it (two passes over the widened buffer). For a LARGE array that costs twice
            // the wide-buffer write wall (bool<<2 at 10M ≈ 27 ms astype+SIMD vs ≈ 13.9 ms for the
            // single-pass fused-scalar NDIter route below, which fuses the bool->loop-dtype convert
            // into the shift's per-element load), so large bool shifts stay on that route. But for a
            // SMALL/MEDIUM array the fused-scalar route's per-element SCALAR inner loop (no per-lane
            // SIMD body — see BoolSimdScalarShiftMaxElements) is far slower than astype + the
            // vectorized uniform-count shift kernel, so route those here (astype widens below, then
            // the SIMD scalar-shift kernel runs). Cutoff keeps the int64 temp L2-resident.
            if (lhs.GetTypeCode == NPTypeCode.Boolean && lhs.size > BoolSimdScalarShiftMaxElements)
                return null;

            var resultType = ShiftResultType(lhs, rhs);
            if (!DirectILKernelGenerator.IsShiftSimdSupported(resultType))
                return null;

            // The kernel walks the value buffer linearly, so the value operand (widened to the
            // result dtype) must be contiguous. A same-dtype strided view defers to NDIter.
            NDArray value;
            if (lhs.GetTypeCode != resultType)
                value = lhs.astype(resultType);          // contiguous C-order copy at the loop dtype
            else if (lhs.Shape.IsContiguous)
                value = lhs;
            else
                return null;

            if (!value.Shape.IsContiguous)
                return null;

            int bitWidth = resultType.SizeOf() * 8;
            int shiftArg = ReadSaturatedShiftCount(rhs, bitWidth);

            // A widened value (astype produced a fresh copy at the loop dtype) is a transient the
            // kernel only reads: return its buffer to the pool the instant the kernel finishes,
            // rather than leaving it for a lagging GC/finalizer pass. The bool path always widens,
            // so this is one vs two discarded buffers per call (it is what keeps the small-N
            // astype+SIMD route at parity with NumPy). The result is a separate fresh allocation,
            // never an alias of value, so disposing value here is safe.
            bool valueIsTemp = !ReferenceEquals(value, lhs);

            var result = new NDArray(resultType, new Shape((long[])value.shape.Clone()), false);
            if (result.size == 0)
            {
                if (valueIsTemp) value.Dispose();
                return result;
            }

            NpFunc.Invoke(resultType, SimdScalarShiftDispatch<int>, value, result, shiftArg, isLeftShift);
            if (valueIsTemp) value.Dispose();
            return result;
        }

        /// <summary>
        /// Read the single shift count from a scalar/size-1 operand and saturate it into
        /// <c>[0, bitWidth]</c>: any count that is negative or &gt;= <paramref name="bitWidth"/>
        /// maps to <paramref name="bitWidth"/> so the kernel's once-per-call overflow branch
        /// fires (left/unsigned-right → 0, signed-right → sign fill), matching NumPy. Reading at
        /// the operand's own dtype preserves the magnitude decision regardless of promotion.
        /// </summary>
        private static unsafe int ReadSaturatedShiftCount(NDArray rhs, int bitWidth)
        {
            byte* p = (byte*)rhs.Address + (long)rhs.Shape.offset * rhs.dtypesize;
            long s;
            switch (rhs.GetTypeCode)
            {
                case NPTypeCode.Boolean: s = (*p != 0) ? 1 : 0; break;
                case NPTypeCode.Byte:    s = *p; break;
                case NPTypeCode.SByte:   s = *(sbyte*)p; break;
                case NPTypeCode.Int16:   s = *(short*)p; break;
                case NPTypeCode.UInt16:  s = *(ushort*)p; break;
                case NPTypeCode.Char:    s = *(char*)p; break;
                case NPTypeCode.Int32:   s = *(int*)p; break;
                case NPTypeCode.UInt32:  s = *(uint*)p; break;
                case NPTypeCode.Int64:   s = *(long*)p; break;
                case NPTypeCode.UInt64:
                {
                    ulong u = *(ulong*)p;
                    return u >= (ulong)bitWidth ? bitWidth : (int)u;
                }
                default: return bitWidth;
            }
            return (s < 0 || s >= bitWidth) ? bitWidth : (int)s;
        }

        /// <summary>
        /// Typecode-dispatched (via <see cref="NpFunc"/>) invocation of the SIMD scalar-shift
        /// kernel. The value operand is contiguous; its base address honours
        /// <see cref="Shape.offset"/> so a contiguous slice is handled without a copy.
        /// </summary>
        private static unsafe void SimdScalarShiftDispatch<T>(NDArray value, NDArray output, int shiftArg, bool isLeftShift) where T : unmanaged
        {
            var kernel = DirectILKernelGenerator.GetShiftScalarKernel<T>(isLeftShift);
            if (kernel == null)
                throw new NotSupportedException($"Shift SIMD kernel unavailable for {typeof(T).Name}.");

            byte* inBase = (byte*)value.Address + (long)value.Shape.offset * value.dtypesize;
            kernel((T*)inBase, (T*)output.Address, shiftArg, output.size);
        }
    }
}
