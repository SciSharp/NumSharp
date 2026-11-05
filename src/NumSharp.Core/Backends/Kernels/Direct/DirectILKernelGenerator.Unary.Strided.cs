using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.Unary.Strided.cs - Fused strided-SIMD unary kernel
// =============================================================================
//
// RESPONSIBILITY:
//   The fastest route for a unary op over a NON-contiguous 1-D source. Instead of
//   the gather-to-scratch-then-contiguous-SIMD-kernel two-step (DefaultEngine's
//   TryBufferedStridedUnaryOp), this fuses both into ONE emitted loop:
//
//        strided gather  ->  Vector{W}.Create(lanes)  ->  unary vector op  ->  contiguous store
//
//   per inner-loop vector, single pass, no scratch tile, no per-chunk delegate
//   dispatch. The op body reuses EmitUnaryVectorOperation (Sqrt/Negate/Abs/Square/
//   Floor/Ceil/Round/Truncate/Reciprocal/Deg2Rad/Rad2Deg) — zero per-op
//   duplication, one emit covers every SIMD unary op.
//
// SIGNATURE (StridedUnaryKernel):
//   void(void* src, long srcByteStride, void* dst, long count)
//     src           strided source base (already offset-adjusted by the caller)
//     srcByteStride source stride in BYTES (may be negative for reversed views)
//     dst           contiguous destination base
//     count         element count
//
// WIDTH-ADAPTIVE: the loop emits Vector{128|256|512} via VectorBits + the
//   VectorMethodCache.CreateElements / EmitVectorStore helpers — one source path
//   covers all widths.
//
// SCOPE: same-width SIMD only (InputType == OutputType). The caller
//   (DefaultEngine.TryStridedSimdUnaryOp) gates to float/double — the measured win
//   (expensive vector ops like Sqrt) and where Vector.Create over a handful of lanes
//   is cheap. Promoting/integer/predicate cases keep their existing routes.
//
// RELATED FILES:
//   - DirectILKernelGenerator.Unary.cs        - contiguous unary kernels (sibling)
//   - DirectILKernelGenerator.Unary.Vector.cs - EmitUnaryVectorOperation (reused)
//   - VectorMethodCache.cs                     - CreateElements (lane-count Create)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Fused strided-source unary kernel: builds each SIMD vector directly from a
    /// strided 1-D source via lane-count scalar gathers, applies the unary op, and
    /// stores contiguously — single pass, no scratch buffer, no per-tile dispatch.
    /// </summary>
    /// <param name="src">Strided source base pointer (already offset-adjusted).</param>
    /// <param name="srcByteStride">Source stride in BYTES (may be negative for reversed views).</param>
    /// <param name="dst">Contiguous destination base pointer.</param>
    /// <param name="count">Number of elements.</param>
    public unsafe delegate void StridedUnaryKernel(void* src, long srcByteStride, void* dst, long count);

    public static partial class DirectILKernelGenerator
    {
        #region Fused Strided-SIMD Unary Kernel

        /// <summary>
        /// Cache for fused strided-SIMD unary kernels, keyed by the same
        /// <see cref="UnaryKernelKey"/> the caller uses for the contiguous SIMD kernel
        /// (its <c>IsContiguous</c> field is irrelevant here — the source is always strided).
        /// </summary>
        internal static readonly ConcurrentDictionary<UnaryKernelKey, StridedUnaryKernel> _stridedUnaryCache = new();

        /// <summary>
        /// Get or generate a fused strided-SIMD unary kernel for the given key.
        /// Gating (same-width SIMD-capable op, supported dtype) is the caller's responsibility.
        /// </summary>
        public static StridedUnaryKernel GetStridedUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _stridedUnaryCache.GetOrAdd(key, GenerateStridedUnaryKernel);
        }

        private static StridedUnaryKernel GenerateStridedUnaryKernel(UnaryKernelKey key)
        {
            // StridedUnaryKernel signature:
            // void(void* src, long srcByteStride, void* dst, long count)
            var dm = new DynamicMethod(
                name: $"StridedUnary_{key}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(long), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();
            EmitStridedUnaryBody(il, key, GetTypeSize(key.InputType), GetTypeSize(key.OutputType));
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<StridedUnaryKernel>();
        }

        /// <summary>
        /// Emit the three-stage fused loop: a 2x-unrolled SIMD body, a 1-vector remainder,
        /// and a scalar tail. The SIMD stages assemble Vector{W}&lt;T&gt; from <c>vstep</c>
        /// strided scalar loads (<see cref="VectorMethodCache.CreateElements"/>), apply the op
        /// (<see cref="EmitUnaryVectorOperation"/>), and store contiguously; the tail walks one
        /// strided element at a time (<see cref="EmitUnaryScalarOperation"/>).
        /// </summary>
        private static void EmitStridedUnaryBody(ILGenerator il, UnaryKernelKey key, int inSize, int outSize)
        {
            int vstep = GetVectorCount(key.InputType);   // lanes per vector
            const int unroll = 2;
            long unrollStep = (long)vstep * unroll;

            var locSrc = il.DeclareLocal(typeof(void*));     // running source byte pointer
            var locStride = il.DeclareLocal(typeof(long));   // source byte stride
            var locI = il.DeclareLocal(typeof(long));        // elements done / output index
            var locCount = il.DeclareLocal(typeof(long));    // total count
            var locUnrollEnd = il.DeclareLocal(typeof(long)); // count - unrollStep
            var locVectorEnd = il.DeclareLocal(typeof(long)); // count - vstep

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // locStride = srcByteStride; locCount = count; locSrc = src; locI = 0
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, locStride);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, locCount);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // unrollEnd = count - unrollStep
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = count - vstep
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // === 2x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            // if (i > unrollEnd) goto UnrollEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int n = 0; n < unroll; n++)
                EmitStridedVectorOp(il, key, outSize, vstep, locSrc, locStride, locI, (long)n * vstep);

            // src += unrollStep * stride;  i += unrollStep
            EmitAdvanceSrc(il, locSrc, locStride, unrollStep);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === 1-VECTOR REMAINDER ===
            il.MarkLabel(lblRem);
            // if (i > vectorEnd) goto RemEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            EmitStridedVectorOp(il, key, outSize, vstep, locSrc, locStride, locI, 0);

            // src += vstep * stride;  i += vstep
            EmitAdvanceSrc(il, locSrc, locStride, vstep);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL ===
            il.MarkLabel(lblTail);
            // if (i >= count) goto TailEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Bge, lblTailEnd);

            // dst[i] = op(*(TIn*)src)
            il.Emit(OpCodes.Ldarg_2); // dst
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)outSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldloc, locSrc);
            EmitLoadIndirect(il, key.InputType);
            if (key.InputType != key.OutputType)
                EmitConvertTo(il, key.InputType, key.OutputType);
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            EmitStoreIndirect(il, key.OutputType);

            // src += stride;  i++
            il.Emit(OpCodes.Ldloc, locSrc);
            il.Emit(OpCodes.Ldloc, locStride);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// Emit one fused vector: gather <c>vstep</c> strided scalars
        /// (<c>*(TIn*)(src + (elemBase+k)*stride)</c> for k=0..vstep-1) into a
        /// <c>Vector{W}&lt;TIn&gt;</c> via <see cref="VectorMethodCache.CreateElements"/>, apply the
        /// unary op, and store the result contiguously at <c>dst + (i + elemBase)*outSize</c>.
        /// <paramref name="elemBase"/> is the compile-time lane offset of this vector within the
        /// unrolled group (0 for the first, vstep for the second, …).
        /// </summary>
        private static void EmitStridedVectorOp(
            ILGenerator il, UnaryKernelKey key, int outSize, int vstep,
            LocalBuilder locSrc, LocalBuilder locStride, LocalBuilder locI, long elemBase)
        {
            var inClr = GetSimdLaneType(key.InputType); // Boolean gathers into byte lanes

            // Build Vector<T> from vstep strided scalar loads. Lane k (Create arg k) <- lane 0
            // is the lowest element, so element (i+elemBase+k) of the logical source maps to
            // output position (i+elemBase+k) — a faithful sequential gather.
            for (int k = 0; k < vstep; k++)
            {
                long laneIdx = elemBase + k;
                il.Emit(OpCodes.Ldloc, locSrc);
                if (laneIdx != 0)
                {
                    il.Emit(OpCodes.Ldc_I8, laneIdx);
                    il.Emit(OpCodes.Ldloc, locStride);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                EmitLoadIndirect(il, key.InputType);
            }
            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateElements(VectorBits, inClr), null);

            // Apply the unary vector op (Sqrt / Negate / Abs / Square / Floor / ...).
            EmitUnaryVectorOperation(il, key.Op, key.InputType);

            // Store contiguously at dst + (i + elemBase) * outSize.
            il.Emit(OpCodes.Ldarg_2); // dst
            il.Emit(OpCodes.Ldloc, locI);
            if (elemBase != 0)
            {
                il.Emit(OpCodes.Ldc_I8, elemBase);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)outSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitVectorStore(il, key.OutputType);
        }

        /// <summary>
        /// Emit <c>src += elems * stride</c> (byte pointer advance). <paramref name="elems"/> is a
        /// compile-time constant (the per-iteration element count); the stride is the runtime
        /// byte stride local.
        /// </summary>
        private static void EmitAdvanceSrc(ILGenerator il, LocalBuilder locSrc, LocalBuilder locStride, long elems)
        {
            il.Emit(OpCodes.Ldloc, locSrc);
            il.Emit(OpCodes.Ldc_I8, elems);
            il.Emit(OpCodes.Ldloc, locStride);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);
        }

        #endregion

        #region Reversed-Run Unary Kernel

        /// <summary>
        /// Cache for the reversed-run unary kernels (a 1-D source at byte stride −elemSize), keyed like
        /// <see cref="_stridedUnaryCache"/>. Separate kernels, so the strided kernel's IL stays as it was.
        /// </summary>
        internal static readonly ConcurrentDictionary<UnaryKernelKey, StridedUnaryKernel> _reversedUnaryCache = new();

        /// <summary>
        /// Get or generate the reversed-run unary kernel of <paramref name="key"/>: the
        /// <see cref="StridedUnaryKernel"/> signature for a REVERSED source (<c>a[::-1]</c>, byte stride −elemSize),
        /// which loads each vector with one contiguous load and a lane reversal (InnerLoop.Reversed.cs) where the
        /// strided kernel assembles it from lane-count scalar loads.
        /// </summary>
        /// <param name="key">A same-width key (<c>InputType == OutputType</c>) whose op has a vector body, on a dtype
        /// <see cref="CanReverseLanes"/> approves; the caller checks both.</param>
        /// <returns>The kernel. Its <c>srcByteStride</c> argument is ignored: the source MUST be reversed.</returns>
        /// <exception cref="InvalidOperationException">IL generation is disabled.</exception>
        /// <exception cref="NotSupportedException">The op has no vector or scalar emission for the dtype.</exception>
        public static StridedUnaryKernel GetReversedUnaryKernel(UnaryKernelKey key)
        {
            if (!Enabled)
                throw new InvalidOperationException("IL generation is disabled");

            return _reversedUnaryCache.GetOrAdd(key, GenerateReversedUnaryKernel);
        }

        /// <summary>Generate the kernel <see cref="GetReversedUnaryKernel"/> serves.</summary>
        /// <param name="key">The same-width key.</param>
        /// <returns>The kernel.</returns>
        private static StridedUnaryKernel GenerateReversedUnaryKernel(UnaryKernelKey key)
        {
            var dm = new DynamicMethod(
                name: $"ReversedUnary_{key}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(long), typeof(void*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();
            EmitReversedUnaryBody(il, key);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<StridedUnaryKernel>();
        }

        /// <summary>
        /// Emit the reversed-run unary loop: a 4x-unrolled vector loop, a 1-vector remainder and a scalar tail. Vector
        /// group <c>o</c> of the iteration at logical element <c>i</c> loads the V source elements that end at element
        /// <c>i + o</c> (lowest address <c>src + (j − o − (V−1))·e</c>, with <c>j = −i</c>), reverses the lanes, applies the
        /// op's vector body and stores at <c>dst + (i + o)·e</c>; the tail reads <c>src + j·e</c>.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="key">The same-width key.</param>
        /// <remarks>
        /// The elements that take the scalar body are the last <c>count mod V</c>, as in the strided kernel (its 2x
        /// unroll and 1-vector remainder leave the same tail) and in the buffered route's contiguous kernel (its tiles are
        /// whole vectors), so the vector / scalar split of every element — and so every result byte — is unchanged.
        /// </remarks>
        private static void EmitReversedUnaryBody(ILGenerator il, UnaryKernelKey key)
        {
            NPTypeCode dtype = key.InputType;
            int elemSize = GetTypeSize(dtype);
            long vectorCount = GetVectorCount(dtype);
            long unrollStep = vectorCount * 4;

            var locControl = EmitReverseControl(il, dtype);
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));
            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));          // always −i
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // src = arg0 (logical element 0, the highest address); dst = arg2; bounds; i = j = 0
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, locDst);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            for (int u = 0; u < 4; u++)
                EmitReversedUnaryGroup(il, key, locSrc, locDst, locI, locJ, locControl, u * vectorCount);
            EmitStepIndices(il, locI, locJ, unrollStep);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === 1-VECTOR REMAINDER ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);
            EmitReversedUnaryGroup(il, key, locSrc, locDst, locI, locJ, locControl, 0);
            EmitStepIndices(il, locI, locJ, vectorCount);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL: dst[i] = op(src[-i]) ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Bge, lblTailEnd);
            EmitRunElementAddr(il, locDst, locI, 0, elemSize);
            EmitRunElementAddr(il, locSrc, locJ, 0, elemSize);
            EmitLoadIndirect(il, dtype);
            EmitUnaryScalarOperation(il, key.Op, key.OutputType);
            EmitStoreIndirect(il, key.OutputType);
            EmitStepIndices(il, locI, locJ, 1);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// One vector group of <see cref="EmitReversedUnaryBody"/>: load the V source elements ending at logical element
        /// <c>i + offset</c>, reverse the lanes, apply the op's vector body, store at <c>dst + (i + offset)·e</c>.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="key">The same-width key.</param>
        /// <param name="locSrc">The source pointer at logical element 0.</param>
        /// <param name="locDst">The destination pointer.</param>
        /// <param name="locI">i.</param>
        /// <param name="locJ">j = −i.</param>
        /// <param name="locControl">The lane-reversal control (null for 8-byte lanes).</param>
        /// <param name="offset">The group's element offset from i.</param>
        private static void EmitReversedUnaryGroup(
            ILGenerator il, UnaryKernelKey key, LocalBuilder locSrc, LocalBuilder locDst, LocalBuilder locI, LocalBuilder locJ,
            LocalBuilder? locControl, long offset)
        {
            int elemSize = GetTypeSize(key.InputType);
            long lastLane = GetVectorCount(key.InputType) - 1;
            EmitRunElementAddr(il, locSrc, locJ, -(offset + lastLane), elemSize);
            EmitVectorLoad(il, key.InputType);
            EmitReverseLanes(il, key.InputType, locControl);
            EmitUnaryVectorOperation(il, key.Op, key.InputType);
            EmitRunElementAddr(il, locDst, locI, offset, elemSize);
            EmitVectorStore(il, key.OutputType);
        }

        #endregion
    }
}
