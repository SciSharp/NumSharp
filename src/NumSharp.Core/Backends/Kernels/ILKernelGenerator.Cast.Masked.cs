using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;

namespace NumSharp.Backends.Kernels
{
    // =============================================================================
    // ILKernelGenerator.Cast.Masked.cs
    //   OWNERSHIP: where-masked cross-dtype copy kernels.
    //
    //   PROBLEM:
    //     np.copyto(dst, src, where=mask) was 7-13× slower than NumPy because the
    //     existing CopyWithMask is a fully scalar coordinate-iterated loop that:
    //       - computes O(ndim) muladds per element to derive 3 offsets
    //       - dispatches ConvertValue via switch + double round-trip per masked element
    //
    //   STRATEGY:
    //     One IL-emitted DynamicMethod per (srcType, dstType) pair. Signature:
    //         void(void* src, void* dst, void* mask,
    //              long* srcStrides, long* dstStrides, long* maskStrides,
    //              long* shape, int ndim)
    //     Mask is always Boolean (1 byte/element). Mask strides are in elements
    //     (== bytes for bool).
    //
    //     Inner loop branch (per outer coord position):
    //       - If srcStride[innerAxis] == 1 AND dstStride[innerAxis] == 1
    //         AND maskStride[innerAxis] == 1 AND strategy supports SIMD masked path:
    //           SIMD ConditionalSelect over inner row.
    //       - Else: scalar inner loop with incremental offsets + per-element
    //         mask gate + inline conversion (no double round-trip).
    //
    //     Outer dims walked via stackalloc coord array; offsets advance
    //     incrementally (no mod/div).
    //
    //   SIMD MASKED PATH:
    //     For each store, expand `vstep` mask bytes to V<dst_element_width>
    //     via WidenLower chain + comparison-to-zero, then:
    //       vNew      = <convert V<src> -> V<dst> per strategy>
    //       vExisting = V.Load(dst + offset)
    //       result    = V.ConditionalSelect(maskExpanded, vNew, vExisting)
    //       result.Store(dst + offset)
    //
    //   FALL-THROUGH:
    //     - Same-type and 1:1-lane strategies (MemoryCopy, Int32ToSingle,
    //       SingleToInt32) take the SIMD masked path.
    //     - Widening / narrowing / multi-lane-ratio strategies take the scalar
    //       inner-loop branch (still faster than the old C# coord-iter because
    //       offsets are incremental and conversion is specialized IL, not the
    //       double round-trip switch).
    //
    //   CALLER: np.copyto(...) → CopyWithMask in Manipulation/np.copyto.cs.
    // =============================================================================
    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// where-masked cross-dtype copy kernel delegate.
        /// </summary>
        public unsafe delegate void MaskedCastKernel(
            void* src, void* dst, void* mask,
            long* srcStrides, long* dstStrides, long* maskStrides,
            long* shape, int ndim);

        private static readonly ConcurrentDictionary<CastKernelKey, MaskedCastKernel> _maskedCastCache = new();
        private static readonly ConcurrentDictionary<CastKernelKey, byte> _maskedCastUnsupported = new();

        /// <summary>Number of cached masked-cast kernels (diagnostics).</summary>
        public static int MaskedCastCachedCount => _maskedCastCache.Count;

        /// <summary>
        /// Get or generate a masked-cast kernel for the given (src, dst) pair.
        /// Returns <c>null</c> for unsupported pairs (Boolean/Char/Half/Complex/Decimal involved).
        /// </summary>
        public static MaskedCastKernel TryGetMaskedCastKernel(NPTypeCode srcType, NPTypeCode dstType)
        {
            if (!Enabled) return null;

            var key = new CastKernelKey(srcType, dstType);
            if (_maskedCastCache.TryGetValue(key, out var existing)) return existing;
            if (_maskedCastUnsupported.ContainsKey(key)) return null;

            try
            {
                var kernel = GenerateMaskedCastKernel(key);
                if (kernel == null)
                {
                    _maskedCastUnsupported.TryAdd(key, 0);
                    return null;
                }
                return _maskedCastCache.GetOrAdd(key, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGetMaskedCastKernel({srcType}, {dstType}): {ex.GetType().Name}: {ex.Message}");
                _maskedCastUnsupported.TryAdd(key, 0);
                return null;
            }
        }

        private static MaskedCastKernel GenerateMaskedCastKernel(CastKernelKey key)
        {
            var (strategy, simdBits, vstep) = ResolveStrategy(key.Src, key.Dst);
            if (strategy == CastStrategy.None)
                return null;

            var dm = new DynamicMethod(
                name: $"MaskedCast_{key}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(void*),  // src        (arg 0)
                    typeof(void*),  // dst        (arg 1)
                    typeof(void*),  // mask       (arg 2)
                    typeof(long*),  // srcStrides (arg 3)
                    typeof(long*),  // dstStrides (arg 4)
                    typeof(long*),  // maskStrides(arg 5)
                    typeof(long*),  // shape      (arg 6)
                    typeof(int),    // ndim       (arg 7)
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            EmitMaskedCastBody(dm.GetILGenerator(), key, strategy, simdBits, vstep);
            return dm.CreateDelegate<MaskedCastKernel>();
        }

        private static void EmitMaskedCastBody(ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            const int maskSize = 1; // bool

            // Locals
            var locInnerN          = il.DeclareLocal(typeof(long));
            var locInnerSrcStride  = il.DeclareLocal(typeof(long));
            var locInnerDstStride  = il.DeclareLocal(typeof(long));
            var locInnerMaskStride = il.DeclareLocal(typeof(long));
            var locOuterSrcOffset  = il.DeclareLocal(typeof(long));
            var locOuterDstOffset  = il.DeclareLocal(typeof(long));
            var locOuterMaskOffset = il.DeclareLocal(typeof(long));
            var locCoords          = il.DeclareLocal(typeof(long*));
            var locOuterNdim       = il.DeclareLocal(typeof(int));

            // Labels
            var lblScalar0DStart   = il.DefineLabel();
            var lblOuterLoopHead   = il.DefineLabel();
            var lblOuterLoopBody   = il.DefineLabel();
            var lblInnerContigPath = il.DefineLabel();
            var lblInnerScalarPath = il.DefineLabel();
            var lblAfterInner      = il.DefineLabel();
            var lblRet             = il.DefineLabel();

            // ---- ndim == 0: single conditional element ----
            il.Emit(OpCodes.Ldarg_S, (byte)7);   // ndim
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bne_Un, lblScalar0DStart);

            // if (*(bool*)mask) dst = (TDst) src
            il.Emit(OpCodes.Ldarg_2);   // mask
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblRet);

            il.Emit(OpCodes.Ldarg_1);   // dst
            il.Emit(OpCodes.Ldarg_0);   // src
            EmitLoadIndirect(il, key.Src);
            EmitConvertTo(il, key.Src, key.Dst);
            EmitStoreIndirect(il, key.Dst);
            il.Emit(OpCodes.Br, lblRet);

            il.MarkLabel(lblScalar0DStart);

            // ---- IL any-true prescan ----
            // Walks the mask buffer once before doing any masked work. If no true byte is found
            // we branch straight to lblRet, matching NumPy's all-false short-circuit. The scan
            // is shaped to cover only the UNIQUE bytes of the mask buffer:
            //   - broadcast dim (stride==0): skipped (those bytes repeat)
            //   - contig dim (stride == expected): included
            //   - other strided pattern: prescan disabled (jump to lblPrescanSkip)
            // The contig sub-range starting at maskBase is scanned with V256/V128 EqualsAll(Zero).
            // For mixed/all-true masks the first non-zero byte aborts the prescan immediately
            // (~ns overhead). For all-false the scan runs to completion (~0.3 ms for 10M bytes).
            EmitMaskAnyTruePrescan(il, lblRet);

            // ---- innerN, innerSrcStride, innerDstStride, innerMaskStride ----
            // (innerIdx = ndim-1)
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Conv_I);

            // shape[ndim-1]
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerN);

            // srcStrides[ndim-1]
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerSrcStride);

            // dstStrides[ndim-1]
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerDstStride);

            // maskStrides[ndim-1]
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerMaskStride);

            // outerNdim = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOuterNdim);

            // Coords array via localloc
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            var lblSizeReady = il.DefineLabel();
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bgt, lblSizeReady);
            il.Emit(OpCodes.Pop);
            il.Emit(OpCodes.Ldc_I4_1);
            il.MarkLabel(lblSizeReady);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);

            // Zero coords[]
            {
                var locK = il.DeclareLocal(typeof(int));
                var zeroHead = il.DefineLabel();
                var zeroBody = il.DefineLabel();

                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Stloc, locK);
                il.MarkLabel(zeroHead);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldloc, locOuterNdim);
                il.Emit(OpCodes.Blt, zeroBody);
                il.Emit(OpCodes.Br, lblOuterLoopHead);

                il.MarkLabel(zeroBody);
                il.Emit(OpCodes.Ldloc, locCoords);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_8);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stind_I8);

                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, zeroHead);
            }

            // Initialize outer offsets
            il.MarkLabel(lblOuterLoopHead);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterSrcOffset);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterDstOffset);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterMaskOffset);

            il.MarkLabel(lblOuterLoopBody);

            // Branch: all 3 inner strides unit?
            il.Emit(OpCodes.Ldloc, locInnerSrcStride);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblInnerScalarPath);
            il.Emit(OpCodes.Ldloc, locInnerDstStride);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblInnerScalarPath);
            il.Emit(OpCodes.Ldloc, locInnerMaskStride);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, lblInnerScalarPath);

            // ---- Inner contig path: try SIMD if strategy supports masked SIMD; else scalar inner with mask gate ----
            il.MarkLabel(lblInnerContigPath);

            Action pushSrcInnerBase = () =>
            {
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
                il.Emit(OpCodes.Ldc_I8, (long)srcSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            };
            Action pushDstInnerBase = () =>
            {
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locOuterDstOffset);
                il.Emit(OpCodes.Ldc_I8, (long)dstSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            };
            Action pushMaskInnerBase = () =>
            {
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locOuterMaskOffset);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            };

            EmitMaskedInnerLoop(il, key, strategy, simdBits, vstep,
                pushSrcInnerBase, pushDstInnerBase, pushMaskInnerBase,
                pushInnerCount: () => il.Emit(OpCodes.Ldloc, locInnerN));

            il.Emit(OpCodes.Br, lblAfterInner);

            // ---- Inner scalar path: per-element with branch on mask ----
            il.MarkLabel(lblInnerScalarPath);
            EmitScalarMaskedInner(il, key, srcSize, dstSize, maskSize,
                locInnerN, locInnerSrcStride, locInnerDstStride, locInnerMaskStride,
                locOuterSrcOffset, locOuterDstOffset, locOuterMaskOffset);

            il.MarkLabel(lblAfterInner);

            // ---- Advance outer coords ----
            il.Emit(OpCodes.Ldloc, locOuterNdim);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Ble, lblRet);

            EmitMaskedOuterAdvance(il, locCoords, locOuterNdim,
                locOuterSrcOffset, locOuterDstOffset, locOuterMaskOffset,
                lblOuterLoopBody, lblRet);

            il.MarkLabel(lblRet);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        /// Inner scalar masked loop: walks innerN elements with mask gate + incremental offsets.
        /// </summary>
        private static void EmitScalarMaskedInner(
            ILGenerator il, CastKernelKey key,
            int srcSize, int dstSize, int maskSize,
            LocalBuilder locInnerN,
            LocalBuilder locInnerSrcStride, LocalBuilder locInnerDstStride, LocalBuilder locInnerMaskStride,
            LocalBuilder locOuterSrcOffset, LocalBuilder locOuterDstOffset, LocalBuilder locOuterMaskOffset)
        {
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd  = il.DefineLabel();
            var lblSkip       = il.DefineLabel();

            var locSi          = il.DeclareLocal(typeof(long));
            var locInnerSrcOff = il.DeclareLocal(typeof(long));
            var locInnerDstOff = il.DeclareLocal(typeof(long));
            var locInnerMaskOff= il.DeclareLocal(typeof(long));

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locSi);
            il.Emit(OpCodes.Ldloc, locOuterSrcOffset);
            il.Emit(OpCodes.Stloc, locInnerSrcOff);
            il.Emit(OpCodes.Ldloc, locOuterDstOffset);
            il.Emit(OpCodes.Stloc, locInnerDstOff);
            il.Emit(OpCodes.Ldloc, locOuterMaskOffset);
            il.Emit(OpCodes.Stloc, locInnerMaskOff);

            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locSi);
            il.Emit(OpCodes.Ldloc, locInnerN);
            il.Emit(OpCodes.Bge, lblScalarEnd);

            // Load mask byte at (mask + innerMaskOff)
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locInnerMaskOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblSkip);

            // dst[innerDstOff] = (TDst) src[innerSrcOff]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locInnerDstOff);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locInnerSrcOff);
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.Src);
            EmitConvertTo(il, key.Src, key.Dst);
            EmitStoreIndirect(il, key.Dst);

            il.MarkLabel(lblSkip);

            // Advance offsets and si
            il.Emit(OpCodes.Ldloc, locInnerSrcOff);
            il.Emit(OpCodes.Ldloc, locInnerSrcStride);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locInnerSrcOff);

            il.Emit(OpCodes.Ldloc, locInnerDstOff);
            il.Emit(OpCodes.Ldloc, locInnerDstStride);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locInnerDstOff);

            il.Emit(OpCodes.Ldloc, locInnerMaskOff);
            il.Emit(OpCodes.Ldloc, locInnerMaskStride);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locInnerMaskOff);

            il.Emit(OpCodes.Ldloc, locSi);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSi);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblScalarEnd);
        }

        /// <summary>
        /// Inner contig SIMD masked loop. For strategies that emit a single store per iter
        /// (Int32→Single, Single→Int32, MemoryCopy), use SIMD ConditionalSelect to mask the
        /// conversion. For all other strategies (widen/narrow/etc.), fall back to a scalar
        /// inner loop with mask gate and inline conversion.
        /// </summary>
        private static void EmitMaskedInnerLoop(
            ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep,
            Action pushSrcBase, Action pushDstBase, Action pushMaskBase,
            Action pushInnerCount)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);
            const int maskSize = 1;

            bool canUseSimdMasked =
                simdBits > 0 && vstep > 0 &&
                (strategy == CastStrategy.MemoryCopy ||
                 strategy == CastStrategy.Int32ToSingle ||
                 strategy == CastStrategy.SingleToInt32);

            if (!canUseSimdMasked)
            {
                // Inner-contig but no SIMD masked variant: walk scalar with mask gate.
                EmitContigScalarMaskedInner(il, key, srcSize, dstSize, maskSize,
                    pushSrcBase, pushDstBase, pushMaskBase, pushInnerCount);
                return;
            }

            // ---- SIMD masked path ----
            var locI = il.DeclareLocal(typeof(long));
            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd  = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();
            var lblScalarEnd  = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // SIMD loop: while (i + vstep <= count)
            il.MarkLabel(lblSimdHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Add);
            pushInnerCount();
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            EmitSimdMaskedIteration(il, key, strategy, simdBits, vstep, locI,
                pushSrcBase, pushDstBase, pushMaskBase);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vstep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            // Scalar tail
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            pushInnerCount();
            il.Emit(OpCodes.Bge, lblScalarEnd);

            // Load mask[i]
            pushMaskBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblScalarSkip);

            // dst[i] = (TDst) src[i]
            pushDstBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            pushSrcBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.Src);
            EmitConvertTo(il, key.Src, key.Dst);
            EmitStoreIndirect(il, key.Dst);

            il.MarkLabel(lblScalarSkip);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblScalarEnd);
        }

        /// <summary>
        /// Same shape as the inner-contig path but no SIMD — used for widen/narrow strategies
        /// where SIMD masking would require complex per-store mask expansion.
        /// </summary>
        private static void EmitContigScalarMaskedInner(
            ILGenerator il, CastKernelKey key,
            int srcSize, int dstSize, int maskSize,
            Action pushSrcBase, Action pushDstBase, Action pushMaskBase,
            Action pushInnerCount)
        {
            var locI = il.DeclareLocal(typeof(long));
            var lblHead = il.DefineLabel();
            var lblSkip = il.DefineLabel();
            var lblEnd  = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            pushInnerCount();
            il.Emit(OpCodes.Bge, lblEnd);

            // Load mask[i]
            pushMaskBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblSkip);

            // dst[i] = (TDst) src[i]
            pushDstBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)dstSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            pushSrcBase();
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)srcSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, key.Src);
            EmitConvertTo(il, key.Src, key.Dst);
            EmitStoreIndirect(il, key.Dst);

            il.MarkLabel(lblSkip);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        /// <summary>
        /// Emit one SIMD masked iteration. Supported strategies:
        ///   MemoryCopy / Int32ToSingle / SingleToInt32 (all are 1:1 lane count, single store per iter).
        /// </summary>
        private static void EmitSimdMaskedIteration(
            ILGenerator il, CastKernelKey key, CastStrategy strategy, int simdBits, int vstep, LocalBuilder locI,
            Action pushSrcBase, Action pushDstBase, Action pushMaskBase)
        {
            int srcSize = GetTypeSize(key.Src);
            int dstSize = GetTypeSize(key.Dst);

            // Resolve dst element CLR type for V loads.
            Type dstElem = GetClrType(key.Dst);

            // Compute V<dst_new> via the strategy's normal IL.
            switch (strategy)
            {
                case CastStrategy.MemoryCopy:
                    // Same-type: V<dst>.Load(srcPtr + i*size)
                    EmitLoadVector(il, dstElem, simdBits, () =>
                    {
                        pushSrcBase();
                        EmitOffsetExpr(il, locI, 0, srcSize);
                    });
                    break;

                case CastStrategy.Int32ToSingle:
                    EmitLoadVector(il, typeof(int), simdBits, () =>
                    {
                        pushSrcBase();
                        EmitOffsetExpr(il, locI, 0, srcSize);
                    });
                    il.EmitCall(OpCodes.Call, GetConvertToSingleFromInt32Method(simdBits), null);
                    break;

                case CastStrategy.SingleToInt32:
                    EmitLoadVector(il, typeof(float), simdBits, () =>
                    {
                        pushSrcBase();
                        EmitOffsetExpr(il, locI, 0, srcSize);
                    });
                    il.EmitCall(OpCodes.Call, GetConvertToInt32FromSingleMethod(simdBits), null);
                    break;

                default:
                    throw new InvalidOperationException($"SIMD masked emission not implemented for {strategy}");
            }

            // Stack: V<dst_new>
            var locVnew = il.DeclareLocal(VType(simdBits, dstElem));
            il.Emit(OpCodes.Stloc, locVnew);

            // Load existing V<dst> from dst pointer
            EmitLoadVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, dstSize);
            });
            var locVold = il.DeclareLocal(VType(simdBits, dstElem));
            il.Emit(OpCodes.Stloc, locVold);

            // Expand vstep mask bytes to V<dst> (each lane: -1 if mask byte != 0, else 0)
            EmitMaskExpandToVDst(il, dstElem, simdBits, vstep, locI, pushMaskBase);
            // Stack: V<int-of-dstSize-width>.

            // ConditionalSelect(maskVec.AsT(), vNew, vOld) → V<dst>
            // We compute maskVec as V<int_T_of_dst_size>; reinterpret as V<dst> via As.
            // Stack so far: ..., maskVec (as int width = dstSize). Reinterpret to V<dst>.
            EmitAsToDstElement(il, dstSize, dstElem, simdBits);

            il.Emit(OpCodes.Ldloc, locVnew);
            il.Emit(OpCodes.Ldloc, locVold);
            il.EmitCall(OpCodes.Call, GetConditionalSelectMethod(simdBits, dstElem), null);

            // Store at dst + i*dstSize
            EmitStoreVector(il, dstElem, simdBits, () =>
            {
                pushDstBase();
                EmitOffsetExpr(il, locI, 0, dstSize);
            });
        }

        /// <summary>
        /// Push V&lt;intT_of_dstSize&gt; mask (all-bits-set per lane where mask byte != 0).
        /// </summary>
        private static void EmitMaskExpandToVDst(ILGenerator il, Type dstElem, int simdBits, int vstep, LocalBuilder locI, Action pushMaskBase)
        {
            int dstSize = System.Runtime.InteropServices.Marshal.SizeOf(dstElem);
            // Use signed int element of dstSize width for the widen chain.
            Type intT_dst = dstSize switch
            {
                1 => typeof(sbyte),
                2 => typeof(short),
                4 => typeof(int),
                8 => typeof(long),
                _ => throw new NotSupportedException($"Unsupported dst size {dstSize} for mask expansion")
            };

            // We need to load `vstep` mask bytes into V128/V256/V512 of bytes.
            // V128<byte>.Count = 16, so for vstep <= 16, use V128 load.
            // For vstep > 16, use V256<byte> load (up to 32 bytes).
            int loadVbits = (vstep <= 16) ? 128 : (vstep <= 32 ? 256 : 512);

            // Load V<byte> from (mask + i)
            EmitLoadVector(il, typeof(byte), loadVbits, () =>
            {
                pushMaskBase();
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            });

            // We have V<byte>. Now we need to widen it to V<intT_dst> for `vstep` lanes.
            // Widening pattern: byte → ushort → uint → ulong (signed-as via WidenLower variants).
            // But since our values are 0 or 1 (bool), zero-extension is fine.
            //
            // The result must be V_simdBits<intT_dst> with `vstep` lanes filled with -1 or 0.
            //
            // For dstSize == 4 (int): widen byte → ushort → uint (V<uint>).
            //                          Reinterpret as V<int>. Compare with Zero → -1 if 0, 0 otherwise.
            //                          We want OPPOSITE: -1 if !=0. So XOR with all-ones (Negate).
            //                          Simpler: use 0 - V<int> = -V (0→0, 1→-1).

            // For SIMD width matching: emitted strategy uses simdBits. So we need to widen the
            // bytes to fill V_simdBits<intT_dst>.
            //
            // Widening from V128<byte> (16 bytes) to V128<intT_dst>:
            //   intT_dst = sbyte: no widen, just reinterpret.
            //   intT_dst = short: WidenLower (16 → 8 elements), reinterpret as short.
            //   intT_dst = int:   WidenLower → ushort, WidenLower → uint, reinterpret as int.
            //   intT_dst = long:  WidenLower → ushort, WidenLower → uint, WidenLower → ulong, reinterpret.
            //
            // But our SIMD width may be 256, not 128. After widening V128<byte> to V128<int>, we'd have
            // 4 ints. But we want 8 ints (vstep=8 for V256<int>). So we need to use V256<byte>.WidenLower
            // OR widen to V128<int> and combine two into V256<int>.
            //
            // Easiest path: load mask as V<simdBits/8 bytes>, widen through.
            // For V256 with vstep=8, mask is 8 bytes; we load V64<byte> (8 bytes) — but V64 doesn't exist.
            // V128 is the smallest; we load 16 bytes and use lower 8.
            //
            // Approach: V128.WidenLower chains naturally produce V128<wider>. For V256 output, we'd
            // call Vector256.Create(loVector, hiVector) after producing two V128 halves.

            // This implementation handles the common case: simdBits=256, vstep=8, dstSize=4 (Int32→Single etc.)
            // dstSize=8, vstep=4 (rare for SIMD; we never select this strategy currently)
            // dstSize=1, vstep=32 (V256<byte> MemoryCopy)

            if (simdBits == 128 && dstSize == 1)
            {
                // V128<byte> mask is already correct width (16 lanes). Subtract from zero to flip.
                EmitVectorZeroSubMask(il, 128, typeof(sbyte));
            }
            else if (simdBits == 128 && dstSize == 2)
            {
                // Widen V128<byte> → V128<ushort> → reinterpret as short → zero-subtract
                il.EmitCall(OpCodes.Call, GetWidenLowerMethod(128, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, GetAsMethod(128, typeof(ushort), typeof(short)), null);
                EmitVectorZeroSubMask(il, 128, typeof(short));
            }
            else if (simdBits == 128 && dstSize == 4)
            {
                // Widen V128<byte> → V128<ushort> → V128<uint> → V128<int> → zero-sub
                il.EmitCall(OpCodes.Call, GetWidenLowerMethod(128, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, GetWidenLowerMethod(128, typeof(ushort)), null);
                il.EmitCall(OpCodes.Call, GetAsMethod(128, typeof(uint), typeof(int)), null);
                EmitVectorZeroSubMask(il, 128, typeof(int));
            }
            else if (simdBits == 256 && dstSize == 4)
            {
                // We have V128<byte> with 16 bytes. We want V256<int> (8 lanes from first 8 bytes).
                //   Step 1: V128.WidenLower(V128<byte>) → V128<ushort> (first 8 bytes as ushorts).
                //   Step 2: WidenLower (V128<ushort>) → V128<uint>     (first 4 of 8 as uints).
                //   Step 3: WidenUpper (V128<ushort>) → V128<uint>     (next 4 as uints).
                //   Step 4: Vector256.Create(lo, hi) → V256<uint>.
                //   Step 5: Reinterpret as V256<int>. Then zero-sub.

                var locStep1 = il.DeclareLocal(VType(128, typeof(ushort)));
                il.EmitCall(OpCodes.Call, GetWidenLowerMethod(128, typeof(byte)), null);
                il.Emit(OpCodes.Stloc, locStep1);

                il.Emit(OpCodes.Ldloc, locStep1);
                il.EmitCall(OpCodes.Call, GetWidenLowerMethod(128, typeof(ushort)), null);
                var locLo = il.DeclareLocal(VType(128, typeof(uint)));
                il.Emit(OpCodes.Stloc, locLo);

                il.Emit(OpCodes.Ldloc, locStep1);
                il.EmitCall(OpCodes.Call, GetWidenUpperMethod(128, typeof(ushort)), null);
                var locHi = il.DeclareLocal(VType(128, typeof(uint)));
                il.Emit(OpCodes.Stloc, locHi);

                il.Emit(OpCodes.Ldloc, locLo);
                il.Emit(OpCodes.Ldloc, locHi);
                il.EmitCall(OpCodes.Call, GetCreateFromTwoMethod(256, typeof(uint)), null);

                il.EmitCall(OpCodes.Call, GetAsMethod(256, typeof(uint), typeof(int)), null);
                EmitVectorZeroSubMask(il, 256, typeof(int));
            }
            else
            {
                throw new NotSupportedException($"Mask expansion not implemented for simdBits={simdBits}, dstSize={dstSize}");
            }
        }

        /// <summary>
        /// Stack has V&lt;intT&gt; on top with values 0 or 1. Emit: Vector&lt;intT&gt;.Zero - v
        /// to convert (0 → 0, 1 → -1).
        /// </summary>
        private static void EmitVectorZeroSubMask(ILGenerator il, int simdBits, Type intT)
        {
            // Save v
            var locV = il.DeclareLocal(VType(simdBits, intT));
            il.Emit(OpCodes.Stloc, locV);

            // Get Zero static property
            var zeroProp = VType(simdBits, intT).GetProperty("Zero", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException($"Vector{simdBits}<{intT.Name}>.Zero not found");
            il.EmitCall(OpCodes.Call, zeroProp.GetGetMethod(), null);
            il.Emit(OpCodes.Ldloc, locV);
            // Stack: Zero, v. Subtract.
            var subOp = VType(simdBits, intT).GetMethod("op_Subtraction",
                BindingFlags.Public | BindingFlags.Static, null,
                new[] { VType(simdBits, intT), VType(simdBits, intT) }, null)
                ?? throw new InvalidOperationException($"Vector{simdBits}<{intT.Name}>.op_Subtraction not found");
            il.EmitCall(OpCodes.Call, subOp, null);
        }

        /// <summary>
        /// Stack has V&lt;intT_of_dstSize&gt;. Reinterpret as V&lt;dstElem&gt; via As&lt;...&gt;.
        /// </summary>
        private static void EmitAsToDstElement(ILGenerator il, int dstSize, Type dstElem, int simdBits)
        {
            Type intT_dst = dstSize switch
            {
                1 => typeof(sbyte),
                2 => typeof(short),
                4 => typeof(int),
                8 => typeof(long),
                _ => throw new NotSupportedException()
            };
            if (intT_dst == dstElem) return;
            il.EmitCall(OpCodes.Call, GetAsMethod(simdBits, intT_dst, dstElem), null);
        }

        private static MethodInfo GetConditionalSelectMethod(int simdBits, Type elementType)
        {
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "ConditionalSelect" && m.IsGenericMethod &&
                            m.GetParameters().Length == 3)
                .MakeGenericMethod(elementType);
        }

        /// <summary>
        /// Vector{simdBits}.Create(V128_lo, V128_hi) - combine two V128 into one V256/V512.
        /// </summary>
        private static MethodInfo GetCreateFromTwoMethod(int simdBits, Type elementType)
        {
            var halfV = VType(simdBits / 2, elementType);
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "Create" && !m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetParameters()[0].ParameterType == halfV);
        }

        /// <summary>
        /// Emit outer-coord advance for the masked kernel (3 offsets to update).
        /// </summary>
        private static void EmitMaskedOuterAdvance(
            ILGenerator il,
            LocalBuilder locCoords, LocalBuilder locOuterNdim,
            LocalBuilder locOuterSrcOffset, LocalBuilder locOuterDstOffset, LocalBuilder locOuterMaskOffset,
            System.Reflection.Emit.Label lblOuterLoopBody, System.Reflection.Emit.Label lblRet)
        {
            var locAxis = il.DeclareLocal(typeof(int));
            var advHead = il.DefineLabel();
            var advBody = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locOuterNdim);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAxis);

            il.MarkLabel(advHead);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bge, advBody);
            il.Emit(OpCodes.Br, lblRet);

            il.MarkLabel(advBody);

            // coords[axis]++
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);

            // Add stride[axis] to each offset
            EmitAddStrideToOffset(il, locOuterSrcOffset, locAxis, (byte)3);    // srcStrides
            EmitAddStrideToOffset(il, locOuterDstOffset, locAxis, (byte)4);    // dstStrides
            EmitAddStrideToOffset(il, locOuterMaskOffset, locAxis, (byte)5);   // maskStrides

            // if (coords[axis] < shape[axis]) goto lblOuterLoopBody
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldarg_S, (byte)6);   // shape
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Blt, lblOuterLoopBody);

            // coords[axis] = 0; offsets -= strides[axis] * shape[axis]
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);

            EmitSubStrideTimesShapeFromOffset(il, locOuterSrcOffset, locAxis, (byte)3);
            EmitSubStrideTimesShapeFromOffset(il, locOuterDstOffset, locAxis, (byte)4);
            EmitSubStrideTimesShapeFromOffset(il, locOuterMaskOffset, locAxis, (byte)5);

            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAxis);
            il.Emit(OpCodes.Br, advHead);
        }

        private static void EmitAddStrideToOffset(ILGenerator il, LocalBuilder locOffset, LocalBuilder locAxis, byte stridesArg)
        {
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Ldarg_S, stridesArg);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOffset);
        }

        private static void EmitSubStrideTimesShapeFromOffset(ILGenerator il, LocalBuilder locOffset, LocalBuilder locAxis, byte stridesArg)
        {
            il.Emit(OpCodes.Ldloc, locOffset);
            il.Emit(OpCodes.Ldarg_S, stridesArg);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldarg_S, (byte)6);   // shape
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locOffset);
        }

        // =================================================================
        // IL prescan: any-true scan of the mask buffer.
        // Branches to <paramref name="lblRet"/> when no true byte is found.
        // Falls through (continues IL execution) otherwise.
        // Uses arg 2 (mask), arg 5 (maskStrides), arg 6 (shape), arg 7 (ndim).
        // =================================================================

        /// <summary>
        /// Emit IL that scans the mask buffer for any non-zero byte. If none are found,
        /// branches to <paramref name="lblRet"/>. Else falls through.
        ///
        /// Phase 1: Compute the contiguous unique-byte length of the mask. Walks dims from
        /// inner to outer; broadcast dims (stride==0) are skipped, contig dims multiply
        /// unique_size by shape[d]. A stride that doesn't match the expected contig stride
        /// disables the scan (falls through without branching).
        ///
        /// Phase 2: SIMD scan of <c>unique_size</c> bytes starting at <c>maskBase</c>.
        /// V256<byte> when available, else V128<byte>, with a scalar tail. The first
        /// non-zero byte aborts the scan (falls through with prescan disabled).
        /// Reaching the end of unique_size without finding a non-zero byte branches to
        /// <paramref name="lblRet"/>.
        /// </summary>
        private static void EmitMaskAnyTruePrescan(ILGenerator il, System.Reflection.Emit.Label lblRet)
        {
            int scanBits = VectorBits >= 256 ? 256 : (VectorBits >= 128 ? 128 : 0);

            var locUniqueSize     = il.DeclareLocal(typeof(long));
            var locExpectedStride = il.DeclareLocal(typeof(long));
            var locAxis           = il.DeclareLocal(typeof(int));
            var locScanI          = il.DeclareLocal(typeof(long));

            var lblWalkHead       = il.DefineLabel();
            var lblWalkBody       = il.DefineLabel();
            var lblWalkAdvance    = il.DefineLabel();
            var lblScanBegin      = il.DefineLabel();
            var lblSimdHead       = il.DefineLabel();
            var lblSimdEnd        = il.DefineLabel();
            var lblScalarHead     = il.DefineLabel();
            var lblSkipPrescan    = il.DefineLabel();

            // ---- Phase 1: compute unique_size = product of shape[d] over non-broadcast contig dims ----
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Stloc, locUniqueSize);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Stloc, locExpectedStride);

            // axis = ndim - 1
            il.Emit(OpCodes.Ldarg_S, (byte)7);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAxis);

            il.MarkLabel(lblWalkHead);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Bge, lblWalkBody);
            il.Emit(OpCodes.Br, lblScanBegin);   // axis < 0 -> done walking

            il.MarkLabel(lblWalkBody);

            // s = maskStrides[axis]
            il.Emit(OpCodes.Ldarg_S, (byte)5);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            var locS = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Stloc, locS);

            // if (s == 0) skip — broadcast dim
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Beq, lblWalkAdvance);

            // if (s != expected_stride) -> non-contig, skip prescan
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Ldloc, locExpectedStride);
            il.Emit(OpCodes.Bne_Un, lblSkipPrescan);

            // unique_size *= shape[axis]
            il.Emit(OpCodes.Ldloc, locUniqueSize);
            il.Emit(OpCodes.Ldarg_S, (byte)6);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locUniqueSize);

            // expected_stride = unique_size
            il.Emit(OpCodes.Ldloc, locUniqueSize);
            il.Emit(OpCodes.Stloc, locExpectedStride);

            il.MarkLabel(lblWalkAdvance);
            il.Emit(OpCodes.Ldloc, locAxis);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locAxis);
            il.Emit(OpCodes.Br, lblWalkHead);

            // ---- Phase 2: scan unique_size bytes starting at mask base ----
            il.MarkLabel(lblScanBegin);

            // If unique_size <= 0, defensive skip (nothing to scan)
            il.Emit(OpCodes.Ldloc, locUniqueSize);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblSkipPrescan);

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locScanI);

            // SIMD scan loop (only emitted when SIMD is available)
            if (scanBits > 0)
            {
                int chunkBytes = scanBits / 8;

                il.MarkLabel(lblSimdHead);
                il.Emit(OpCodes.Ldloc, locScanI);
                il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locUniqueSize);
                il.Emit(OpCodes.Bgt, lblSimdEnd);

                // v = V<scanBits>.Load((byte*)(mask + scanI))
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldloc, locScanI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.EmitCall(OpCodes.Call, GetVectorLoadMethod(scanBits, typeof(byte)), null);

                // V<scanBits><byte>.Zero
                var zeroProp = VType(scanBits, typeof(byte)).GetProperty("Zero",
                    BindingFlags.Public | BindingFlags.Static)
                    ?? throw new InvalidOperationException($"Vector{scanBits}<byte>.Zero not found");
                il.EmitCall(OpCodes.Call, zeroProp.GetGetMethod(), null);

                // EqualsAll(v, Zero) — returns bool. If true, all bytes were 0.
                il.EmitCall(OpCodes.Call, GetEqualsAllMethod(scanBits, typeof(byte)), null);
                il.Emit(OpCodes.Brfalse, lblSkipPrescan);   // any non-zero -> abort scan

                // scanI += chunkBytes
                il.Emit(OpCodes.Ldloc, locScanI);
                il.Emit(OpCodes.Ldc_I8, (long)chunkBytes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locScanI);
                il.Emit(OpCodes.Br, lblSimdHead);

                il.MarkLabel(lblSimdEnd);
            }

            // Scalar tail
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locScanI);
            il.Emit(OpCodes.Ldloc, locUniqueSize);
            il.Emit(OpCodes.Bge, lblRet);   // reached end without finding true -> all-false, return

            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locScanI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brtrue, lblSkipPrescan);   // found true -> abort scan

            il.Emit(OpCodes.Ldloc, locScanI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locScanI);
            il.Emit(OpCodes.Br, lblScalarHead);

            // Fall-through label: prescan failed or was disabled; continue with masked work.
            il.MarkLabel(lblSkipPrescan);
        }

        /// <summary>
        /// Vector{simdBits}.EqualsAll&lt;T&gt;(V&lt;T&gt;, V&lt;T&gt;) -> bool.
        /// Returns the IL-emittable MethodInfo.
        /// </summary>
        private static MethodInfo GetEqualsAllMethod(int simdBits, Type elementType)
        {
            return ContainerType(simdBits)
                .GetMethods(BindingFlags.Public | BindingFlags.Static)
                .First(m => m.Name == "EqualsAll" && m.IsGenericMethod &&
                            m.GetParameters().Length == 2 &&
                            m.GetGenericArguments().Length == 1 &&
                            m.ReturnType == typeof(bool))
                .MakeGenericMethod(elementType);
        }
    }
}
