using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// ILKernelGenerator.Filter.cs — fused IL kernel for np.extract / np.compress
// =============================================================================
//
// RESPONSIBILITY:
//   Mask-driven gather. Equivalent to take(src, flatnonzero(mask)) but in a
//   single IL pass — no intermediate indices NDArray, no separate dispatch
//   through ravel + flatnonzero + take. Saves the per-call allocation +
//   construction overhead which dominates the small-slab regime.
//
//   Used by:
//     np.extract   (1-D path: outerSize=1, innerSize=elemBytes)
//     np.compress  (axis path: outerSize=prod(shape[:axis]),
//                              innerSize=prod(shape[axis+1:]) * elemBytes)
//
//   Two-pass algorithm (matches np.nonzero's pre-size-then-fill pattern):
//     Pass 1: popcount mask via the existing ArgwhereCountKernel<bool>
//             → N = number of true positions
//     Pass 2: allocate result then run this kernel which walks the mask with
//             SIMD bit-scan and copies one slab per outer per True.
//
// KERNEL FAMILY (DynamicMethod-emitted; cached by innerSize hint):
//
//   * FilterAxisKernel(byte* src, byte* mask, long maskSize,
//                       long outerSize, long srcOuterStride,
//                       long dstOuterStride, long innerSize, byte* dst)
//                       → long count
//
//   We emit ONE kernel per innerSize class:
//       innerSize ∈ {1, 2, 4, 8, 16}  → typed Ldind/Stind for the per-slab
//                                       copy (single instruction per True).
//       innerSize = anything else    → cpblk(innerSize) bulk copy.
//
//   The "Typed" variant ignores the innerSize argument at runtime and bakes
//   the size into the emitted IL via Ldind_*/Stind_*. The "Bulk" variant uses
//   cpblk with innerSize loaded as the count. This avoids the per-True call
//   overhead cpblk(small-N) pays — the 1-D extract case (innerSize == 8) is
//   the most common and gains the most.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted fused mask-driven gather. Reads <paramref name="src"/> at
    /// each True position in <paramref name="mask"/> and emits one slab per
    /// outer block into <paramref name="dst"/>. The runtime
    /// <paramref name="innerSize"/> is honoured by the "bulk" variant; the
    /// typed variants (1/2/4/8/16-byte) ignore it (the size is baked into IL).
    /// </summary>
    public unsafe delegate long FilterAxisKernel(
        byte* src, byte* mask, long maskSize,
        long outerSize, long srcOuterStride, long dstOuterStride,
        long innerSize, byte* dst);

    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Cache key for the filter kernel — innerSize bucket. Keys 1/2/4/8/16
        /// use typed Ldind/Stind; 0 is the catch-all bulk-cpblk kernel.
        /// </summary>
        private static readonly ConcurrentDictionary<int, FilterAxisKernel> _filterAxis = new();

        /// <summary>
        /// IL-emitted kernel cached by <paramref name="innerSize"/>. Pass the
        /// actual innerSize you'll use at call time; the function buckets
        /// {1,2,4,8,16} into typed-copy variants and anything else into the
        /// bulk-cpblk variant. Returns <c>null</c> only when
        /// <see cref="Enabled"/> is false.
        /// </summary>
        public static FilterAxisKernel GetFilterAxisKernel(long innerSize)
        {
            if (!Enabled)
                return null;

            int key = innerSize switch
            {
                1 => 1,
                2 => 2,
                4 => 4,
                8 => 8,
                16 => 16,
                _ => 0,    // bulk
            };

            return _filterAxis.GetOrAdd(key, static k =>
            {
                try { return GenerateFilterAxisKernelIL(k); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[ILKernel] GetFilterAxisKernel({k}): {ex.GetType().Name}: {ex.Message}");
                    return null;
                }
            });
        }

        private static FilterAxisKernel GenerateFilterAxisKernelIL(int innerSizeHint)
        {
            var dm = new DynamicMethod(
                name: $"IL_FilterAxis_{innerSizeHint}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(byte*),  // 1 mask
                    typeof(long),   // 2 maskSize
                    typeof(long),   // 3 outerSize
                    typeof(long),   // 4 srcOuterStride
                    typeof(long),   // 5 dstOuterStride
                    typeof(long),   // 6 innerSize (ignored for typed variants)
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            // SIMD scan on bool buffer: lane element = byte, lane size = 1 byte.
            // Cap at 256 because ExtractMostSignificantBits returns uint (32 bits).
            int simdBits = VectorBits >= 256 ? 256 : (VectorBits >= 128 ? 128 : 0);
            bool useSimd = simdBits >= 128;
            int laneCount = simdBits / 8;                              // bytes per chunk
            uint chunkMask = laneCount >= 32 ? uint.MaxValue : ((1u << laneCount) - 1);

            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));
            var locSrcAt = il.DeclareLocal(typeof(byte*));
            var locDstAt = il.DeclareLocal(typeof(byte*));
            var locK = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locNz = il.DeclareLocal(typeof(uint));
            var locVecEnd = il.DeclareLocal(typeof(long));
            var locPos = il.DeclareLocal(typeof(int));

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblBitHead = il.DefineLabel();
            var lblBitEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();

            // i = 0; j = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locJ);

            if (useSimd)
            {
                // vecEnd = maskSize - laneCount
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, (long)laneCount);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVecEnd);

                // ---- SIMD bit-scan loop ----
                il.MarkLabel(lblSimdHead);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVecEnd);
                il.Emit(OpCodes.Bgt, lblSimdEnd);

                // vec = Vector{N}<byte>.Load(mask + i)
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Load(simdBits, typeof(byte)), null);

                // zero = Vector{N}<byte>.Zero
                il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(simdBits, typeof(byte)), null);

                // cmp = Equals(vec, zero) — true lanes where mask==0
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(simdBits, typeof(byte)), null);

                // bits = ExtractMostSignificantBits(cmp)
                il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(simdBits, typeof(byte)), null);

                // nz = ~bits & chunkMask
                il.Emit(OpCodes.Not);
                il.Emit(OpCodes.Ldc_I4, unchecked((int)chunkMask));
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locNz);

                // ---- inner bit-scan ----
                il.MarkLabel(lblBitHead);
                il.Emit(OpCodes.Ldloc, locNz);
                il.Emit(OpCodes.Ldc_I4_0);
                il.Emit(OpCodes.Beq, lblBitEnd);

                // pos = TrailingZeroCount(nz)
                il.Emit(OpCodes.Ldloc, locNz);
                il.EmitCall(OpCodes.Call, BitOpsTrailingZeroCount32, null);
                il.Emit(OpCodes.Stloc, locPos);

                // idx = i + pos
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locPos);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locIdx);

                // Emit the per-True outer-loop gather (typed or bulk).
                EmitOuterGather(il, innerSizeHint, locIdx, locJ, locSrcAt, locDstAt, locK);

                // j++
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locJ);

                // nz &= nz - 1  (clear lowest set bit)
                il.Emit(OpCodes.Ldloc, locNz);
                il.Emit(OpCodes.Ldloc, locNz);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locNz);

                il.Emit(OpCodes.Br, lblBitHead);
                il.MarkLabel(lblBitEnd);

                // i += laneCount
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)laneCount);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblSimdHead);

                il.MarkLabel(lblSimdEnd);
            }

            // ---- Scalar tail (or whole loop when no SIMD) ----
            il.MarkLabel(lblScalarHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblScalarEnd);

            // if (mask[i] == 0) goto skip
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblScalarSkip);

            // idx = i
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stloc, locIdx);

            EmitOuterGather(il, innerSizeHint, locIdx, locJ, locSrcAt, locDstAt, locK);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblScalarSkip);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblScalarHead);

            il.MarkLabel(lblScalarEnd);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ret);

            return (FilterAxisKernel)dm.CreateDelegate(typeof(FilterAxisKernel));
        }

        /// <summary>
        /// Emits the per-True inner loop, picking a typed move for
        /// innerSizeHint ∈ {1,2,4,8,16} or cpblk for hint=0 (bulk).
        /// </summary>
        private static void EmitOuterGather(
            ILGenerator il, int innerSizeHint,
            LocalBuilder locIdx, LocalBuilder locJ,
            LocalBuilder locSrcAt, LocalBuilder locDstAt, LocalBuilder locK)
        {
            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();

            // For typed variants we know innerSize at emit time → use that constant
            // for the pointer-advance arithmetic. For bulk we load innerSize at
            // runtime from arg6.
            void EmitInnerSize()
            {
                if (innerSizeHint == 0)
                {
                    il.Emit(OpCodes.Ldarg, 6);   // innerSize (long)
                }
                else
                {
                    il.Emit(OpCodes.Ldc_I8, (long)innerSizeHint);
                }
            }

            // sp = src(arg0) + idx * innerSize
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locIdx);
            EmitInnerSize();
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrcAt);

            // dp = dst(arg7) + j * innerSize
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locJ);
            EmitInnerSize();
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstAt);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldarg_3);                    // outerSize
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // Copy one slab. Typed variants emit Ldind/Stind pair; bulk emits cpblk.
            EmitSlabCopy(il, innerSizeHint, locDstAt, locSrcAt);

            // sp += srcOuterStride(arg4)
            il.Emit(OpCodes.Ldloc, locSrcAt);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrcAt);

            // dp += dstOuterStride(arg5)
            il.Emit(OpCodes.Ldloc, locDstAt);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstAt);

            // k++
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);
        }

        /// <summary>
        /// Single-slab copy. For innerSizeHint 1/2/4/8 emits one Ldind+Stind;
        /// for 16 emits two Ldind_I8+Stind_I8; for 0 (bulk) emits cpblk loading
        /// innerSize from arg6.
        /// </summary>
        private static void EmitSlabCopy(
            ILGenerator il, int innerSizeHint, LocalBuilder locDstAt, LocalBuilder locSrcAt)
        {
            switch (innerSizeHint)
            {
                case 1:
                    // *dp = *sp  (1 byte)
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldind_U1);
                    il.Emit(OpCodes.Stind_I1);
                    return;
                case 2:
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldind_U2);
                    il.Emit(OpCodes.Stind_I2);
                    return;
                case 4:
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldind_I4);
                    il.Emit(OpCodes.Stind_I4);
                    return;
                case 8:
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    return;
                case 16:
                    // Two 8-byte moves — covers decimal/Complex.
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldc_I4_8);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldc_I4_8);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    return;
                case 0:
                default:
                    // Bulk: cpblk(dp, sp, (uint)innerSize)
                    il.Emit(OpCodes.Ldloc, locDstAt);
                    il.Emit(OpCodes.Ldloc, locSrcAt);
                    il.Emit(OpCodes.Ldarg, 6);
                    il.Emit(OpCodes.Conv_U4);
                    il.Emit(OpCodes.Cpblk);
                    return;
            }
        }
    }
}
