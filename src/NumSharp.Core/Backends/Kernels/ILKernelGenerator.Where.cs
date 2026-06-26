using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// ILKernelGenerator.Where.cs — per-chunk multi-operand np.where kernel
// =============================================================================
//
// CONTRACT (NpyInnerLoopFunc — the per-chunk model)
// -------------------------------------------------
//   void(void** dataptrs, long* strides, long count, void* aux)
//
//   operand 0 = cond   (bool, 1 byte)   — already coerced to Boolean by np.where
//   operand 1 = x      (T)
//   operand 2 = y      (T)
//   operand 3 = result (T)
//
//   strides[op] are BYTE strides for the inner loop (NumPy convention). The
//   driving NpyIterRef.ForEach advances dataptrs between chunks; this kernel
//   only walks ONE chunk of `count` elements.
//
// WHY A DEDICATED KERNEL (vs NpyExpr.Where)
// -----------------------------------------
// The old non-contiguous path compiled np.where through NpyExpr.Where, which:
//   (a) is scalar-only (WhereNode.SupportsSimd == false), and
//   (b) loads `cond` AS the output dtype and compares it to zero per element —
//       NpyExpr's "all inputs load at output dtype" rule forces a bool→T cast
//       (e.g. bool→double) on every element before a float compare-to-zero.
// This kernel instead reads cond as a raw bool byte (one Ldind_U1 + brfalse) and
// adds a SIMD ConditionalSelect fast path — faster even before SIMD fires.
//
// PATH SELECTION (decided at runtime, per chunk)
// ----------------------------------------------
//   SIMD ConditionalSelect : cond stride == 1 AND x/y/result stride == elemSize
//                            (the inner loop is contiguous for all operands).
//                            Reuses the proven bool-mask expansion from
//                            DirectILKernelGenerator.EmitInlineMaskCreation.
//   Scalar strided         : everything else (broadcast cond/x/y, transposed,
//                            step>1 slices) and all non-SIMD dtypes
//                            (Boolean/Char/Half/Decimal/Complex). Walks each
//                            operand by its own byte stride.
//
// The SIMD fast path fires for the common "row mask over a matrix" broadcast
// (cond shape (1,M) / (M,) broadcasting over rows) and any inner-contiguous
// view; column-broadcast cond ((N,1)) and genuinely strided layouts use the
// scalar path — which is still materially faster than the old NpyExpr scalar
// loop because it skips the per-element cond cast.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        // outType -> compiled per-chunk where kernel. Keyed by output dtype only:
        // cond is always Boolean and x/y/result always share the output dtype by
        // the time np.where reaches the iterator (operands are pre-cast).
        internal static readonly ConcurrentDictionary<NPTypeCode, NpyInnerLoopFunc> _whereInnerCache = new();

        /// <summary>
        /// Get (or generate and cache) the per-chunk np.where inner-loop kernel for
        /// the given output dtype. Drive it via <c>NpyIterRef.ForEach</c> over a
        /// 4-operand iterator ordered [cond, x, y, result].
        /// </summary>
        internal static NpyInnerLoopFunc GetWhereInnerLoop(NPTypeCode outType)
            => _whereInnerCache.GetOrAdd(outType, GenerateWhereInnerLoop);

        private static NpyInnerLoopFunc GenerateWhereInnerLoop(NPTypeCode outType)
        {
            int elemSize = DirectILKernelGenerator.GetTypeSize(outType);

            // SIMD eligibility — mirrors DirectILKernelGenerator.GenerateWhereKernelIL so
            // the contiguous-inner fast path is bit-identical to the whole-array kernel.
            //  * 1-byte dtypes only touch portable Vector128/256 APIs (any SIMD host).
            //  * 2/4/8-byte dtypes need Avx2 (V256) / Sse41 (V128) to sign-extend the
            //    bool-mask lanes up to the data element width.
            bool canSimdDtype = elemSize <= 8 && DirectILKernelGenerator.CanUseSimd(outType);
            bool needsX86 = elemSize > 1;
            int vb = DirectILKernelGenerator.VectorBits;
            bool useV256 = vb >= 256 && (!needsX86 || Avx2.IsSupported);
            bool useV128 = !useV256 && vb >= 128 && (!needsX86 || Sse41.IsSupported);
            bool emitSimd = canSimdDtype && (useV256 || useV128);
            int simdBits = useV256 ? 256 : 128;
            long vectorCount = emitSimd ? (simdBits / 8) / elemSize : 0;

            var dm = new DynamicMethod(
                name: $"NpyWhereInner_{outType}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            // ---- Snapshot operand pointers + inner-loop byte strides into locals. ----
            var pc = il.DeclareLocal(typeof(byte*));  // cond
            var px = il.DeclareLocal(typeof(byte*));  // x
            var py = il.DeclareLocal(typeof(byte*));  // y
            var pr = il.DeclareLocal(typeof(byte*));  // result
            var sc = il.DeclareLocal(typeof(long));
            var sx = il.DeclareLocal(typeof(long));
            var sy = il.DeclareLocal(typeof(long));
            var sr = il.DeclareLocal(typeof(long));
            var locI = il.DeclareLocal(typeof(long));

            EmitLoadPtr(il, 0, pc); EmitLoadPtr(il, 1, px); EmitLoadPtr(il, 2, py); EmitLoadPtr(il, 3, pr);
            EmitLoadStride(il, 0, sc); EmitLoadStride(il, 1, sx); EmitLoadStride(il, 2, sy); EmitLoadStride(il, 3, sr);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            var lblScalar = il.DefineLabel();

            if (emitSimd)
            {
                // Contiguous-inner check. cond is 1 byte so its natural stride is 1;
                // x/y/result are elemSize bytes. Any mismatch → scalar strided.
                il.Emit(OpCodes.Ldloc, sc); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Bne_Un, lblScalar);
                il.Emit(OpCodes.Ldloc, sx); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Bne_Un, lblScalar);
                il.Emit(OpCodes.Ldloc, sy); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Bne_Un, lblScalar);
                il.Emit(OpCodes.Ldloc, sr); il.Emit(OpCodes.Ldc_I8, (long)elemSize); il.Emit(OpCodes.Bne_Un, lblScalar);

                EmitWhereSimdLoop(il, simdBits, outType, elemSize, vectorCount, pc, px, py, pr, locI);
                // Falls through with locI at the first not-yet-vectorized element; the
                // scalar loop below finishes the tail [locI, count).
            }

            il.MarkLabel(lblScalar);
            EmitWhereScalarStrided(il, outType, pc, px, py, pr, sc, sx, sy, sr, locI);

            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<NpyInnerLoopFunc>();
        }

        // ----- prologue helpers -------------------------------------------------

        // dst = (byte*)dataptrs[op]
        private static void EmitLoadPtr(ILGenerator il, int op, LocalBuilder dst)
        {
            il.Emit(OpCodes.Ldarg_0);
            if (op > 0)
            {
                il.Emit(OpCodes.Ldc_I4, op * IntPtr.Size);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, dst);
        }

        // dst = strides[op]  (bytes)
        private static void EmitLoadStride(ILGenerator il, int op, LocalBuilder dst)
        {
            il.Emit(OpCodes.Ldarg_1);
            if (op > 0)
            {
                il.Emit(OpCodes.Ldc_I4, op * sizeof(long));
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, dst);
        }

        // ----- SIMD ConditionalSelect path (4× unroll + 1-vector remainder) -----

        private static void EmitWhereSimdLoop(
            ILGenerator il, int simdBits, NPTypeCode outType, int elemSize, long vectorCount,
            LocalBuilder pc, LocalBuilder px, LocalBuilder py, LocalBuilder pr, LocalBuilder locI)
        {
            long unrollStep = vectorCount * 4;

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVecEnd = il.DeclareLocal(typeof(long));
            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vecEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // 4× unrolled SIMD loop
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            for (long u = 0; u < 4; u++)
                EmitWhereSimdBody(il, simdBits, outType, elemSize, pc, px, py, pr, locI, u * vectorCount);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // 1-vector remainder loop
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);
            EmitWhereSimdBody(il, simdBits, outType, elemSize, pc, px, py, pr, locI, 0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);
        }

        // One SIMD lane-group: result[i+off..] = select(maskFromCond, x, y).
        // Stack discipline mirrors DirectILKernelGenerator.EmitWhereV256BodyWithOffset:
        // the bool-mask (Vector{N}<unsignedT>) is passed straight into the bitwise
        // ConditionalSelect<T> — same-width vector, accepted by the JIT for dynamic methods.
        private static void EmitWhereSimdBody(
            ILGenerator il, int simdBits, NPTypeCode outType, int elemSize,
            LocalBuilder pc, LocalBuilder px, LocalBuilder py, LocalBuilder pr,
            LocalBuilder locI, long offset)
        {
            var clr = DirectILKernelGenerator.GetClrType(outType);
            var loadM = VectorMethodCache.Load(simdBits, clr);
            var storeM = VectorMethodCache.Store(simdBits, clr);
            var selM = VectorMethodCache.ConditionalSelect(simdBits, clr);

            // mask = expand(cond + (i + offset))   — cond is 1 byte/element.
            il.Emit(OpCodes.Ldloc, pc);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset != 0) { il.Emit(OpCodes.Ldc_I8, offset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            DirectILKernelGenerator.EmitInlineMaskCreation(il, simdBits, elemSize);

            // vX = Load(x + (i + offset) * elemSize)
            EmitVecAddr(il, px, locI, offset, elemSize);
            il.Emit(OpCodes.Call, loadM);

            // vY = Load(y + (i + offset) * elemSize)
            EmitVecAddr(il, py, locI, offset, elemSize);
            il.Emit(OpCodes.Call, loadM);

            // ConditionalSelect(mask, vX, vY)  -> picks x where cond true.
            il.Emit(OpCodes.Call, selM);

            // Store(result_vec, result + (i + offset) * elemSize)  — Store(source, dest*).
            EmitVecAddr(il, pr, locI, offset, elemSize);
            il.Emit(OpCodes.Call, storeM);
        }

        // basePtr + (i + offset) * elemSize
        private static void EmitVecAddr(ILGenerator il, LocalBuilder basePtr, LocalBuilder locI, long offset, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset != 0) { il.Emit(OpCodes.Ldc_I8, offset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        // ----- scalar strided fallback (also finishes the SIMD tail) ------------

        private static void EmitWhereScalarStrided(
            ILGenerator il, NPTypeCode outType,
            LocalBuilder pc, LocalBuilder px, LocalBuilder py, LocalBuilder pr,
            LocalBuilder sc, LocalBuilder sx, LocalBuilder sy, LocalBuilder sr,
            LocalBuilder locI)
        {
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblTakeY = il.DefineLabel();
            var lblStore = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);   // count
            il.Emit(OpCodes.Bge, lblEnd);

            // result address: pr + i*sr  (kept on stack for the Store)
            EmitStridedAddr(il, pr, locI, sr);

            // c = *(byte*)(pc + i*sc)
            EmitStridedAddr(il, pc, locI, sc);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblTakeY);

            // true → load x[i]
            EmitStridedAddr(il, px, locI, sx);
            DirectILKernelGenerator.EmitLoadIndirect(il, outType);
            il.Emit(OpCodes.Br, lblStore);

            // false → load y[i]
            il.MarkLabel(lblTakeY);
            EmitStridedAddr(il, py, locI, sy);
            DirectILKernelGenerator.EmitLoadIndirect(il, outType);

            il.MarkLabel(lblStore);
            DirectILKernelGenerator.EmitStoreIndirect(il, outType);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        // basePtr + i * strideBytes
        private static void EmitStridedAddr(ILGenerator il, LocalBuilder basePtr, LocalBuilder locI, LocalBuilder strideBytes)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, strideBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }
    }
}
