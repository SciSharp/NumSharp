using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.MaskGather.cs — the boolean-mask gather of slabs of one byte width
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A boolean mask over a C-contiguous array selects whole slabs: one element for a full mask, one row of the trailing
// axes for a mask over the leading ones. The numeric types copy them with CopyMaskedElementsHelper<T>, a typed element
// per selected position. A slab of another width — a fixed-width string element (<U8 is 32 bytes), a row of them — had
// no such loop: the string route gathered through an index array (flatnonzero, then a fancy gather of unit-view rows),
// two passes and three temporaries for one selection. This kernel is the one pass: the mask is scanned a vector at a
// time (one compare against zero and a movemask per 32 or 16 mask bytes), the set bits are walked with
// TrailingZeroCount, and each selected slab is copied with a constant-size cpblk (FixedCopyKindFor), which the JIT
// unrolls into vector moves.
//
// CONTRACT
// --------
//   MaskGatherKernel(byte* src, byte* mask, long count, long slabBytes, byte* dst)
//       For i in [0, count): when mask[i] != 0, copies the slab at src + i * slabBytes to the next slab of dst.
//       Returns the number of slabs written. The caller sizes dst from the count of non-zero mask bytes
//       (CountTrueSimdHelper) and keeps src, mask and dst alive. Any non-zero byte selects (a bool array viewed from
//       uint8 data may hold any value), as NumPy's nonzero reads it.
//
// One kernel per copy kind: a width of 1 to MaxFixedCopyBytes bytes is baked in (the slabBytes argument is then unused);
// kind 0 copies and addresses with the runtime slabBytes. The vector width is the startup VectorBits (32 mask bytes a
// step at 256 bits and above, 16 at 128, a scalar loop without SIMD).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Copies, in order, every slab of <paramref name="src"/> whose mask byte is non-zero into consecutive slabs of
    ///     <paramref name="dst"/>; returns the number of slabs written (see the file header for the contract).
    /// </summary>
    /// <param name="src">The first slab of the source (C-contiguous slabs).</param>
    /// <param name="mask">The mask, one byte per slab.</param>
    /// <param name="count">The number of mask bytes (= source slabs).</param>
    /// <param name="slabBytes">The slab size in bytes; read only by a kind-0 kernel.</param>
    /// <param name="dst">The destination, room for one slab per non-zero mask byte.</param>
    public unsafe delegate long MaskGatherKernel(byte* src, byte* mask, long count, long slabBytes, byte* dst);

    public static partial class DirectILKernelGenerator
    {
        /// <summary>The mask gather kernels by copy kind.</summary>
        private static readonly ConcurrentDictionary<int, MaskGatherKernel> _maskGatherKernels = new();

        /// <summary>
        ///     The mask gather kernel for one slab width: <paramref name="copyKind"/> is the
        ///     <see cref="FixedCopyKindFor"/> of the slab (its byte count, baked in), or 0 for a slab wider than
        ///     <see cref="MaxFixedCopyBytes"/> (copied with the runtime count).
        /// </summary>
        /// <param name="copyKind">The copy kind.</param>
        /// <returns>The kernel, or <c>null</c> when runtime IL generation is unavailable (<see cref="Enabled"/> false, or
        ///     the emission failed: the caller keeps its general route).</returns>
        public static MaskGatherKernel GetMaskGatherKernel(int copyKind)
        {
            if (!Enabled)
                return null;
            if (_maskGatherKernels.TryGetValue(copyKind, out var cached))
                return cached;

            try
            {
                return _maskGatherKernels.GetOrAdd(copyKind, GenerateMaskGatherKernelIL(copyKind));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetMaskGatherKernel({copyKind}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits the mask gather. Pseudocode (W = the slab width, a constant for a positive kind; L = 32 or 16 mask
        ///     bytes per vector step):
        ///     <code>
        ///     long MaskGather(byte* src, byte* mask, long count, long slab, byte* dst) {
        ///         byte* dp = dst; long n = 0, i = 0;
        ///         for (; i &lt;= count - L; i += L) {
        ///             uint bits = ~movemask(load(mask + i) == 0) &amp; laneMask;   // non-zero bytes
        ///             while (bits != 0) {
        ///                 int pos = tzcnt(bits);
        ///                 copy W bytes: dp &lt;- src + (i + pos) * W;  dp += W;  n++;
        ///                 bits &amp;= bits - 1;
        ///             }
        ///         }
        ///         for (; i &lt; count; i++) if (mask[i] != 0) { copy W bytes: dp &lt;- src + i * W; dp += W; n++; }
        ///         return n;
        ///     }
        ///     </code>
        /// </summary>
        /// <param name="copyKind">The copy kind (see <see cref="GetMaskGatherKernel"/>).</param>
        /// <returns>The kernel delegate.</returns>
        private static MaskGatherKernel GenerateMaskGatherKernelIL(int copyKind)
        {
            var dm = new DynamicMethod(
                name: $"IL_MaskGather_c{copyKind}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(byte*),  // 1 mask
                    typeof(long),   // 2 count
                    typeof(long),   // 3 slabBytes
                    typeof(byte*),  // 4 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var locI = il.DeclareLocal(typeof(long));
            var locN = il.DeclareLocal(typeof(long));
            var locDp = il.DeclareLocal(typeof(byte*));
            var locSp = il.DeclareLocal(typeof(byte*));

            // dp = dst; n = 0; i = 0
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Stloc, locDp);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locN);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // Copies the slab at element index (on the stack, int64) to dp, then advances dp and n.
            void EmitCopySlabAtIndexOnStack()
            {
                // sp = src + index * W
                if (copyKind > 0)
                    EmitScaleByWidth(il, copyKind);
                else
                {
                    il.Emit(OpCodes.Ldarg_3);
                    il.Emit(OpCodes.Mul);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldarg_0);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locSp);

                EmitElementCopy(il, copyKind, locDp, locSp, () => il.Emit(OpCodes.Ldarg_3));

                // dp += W; n++
                if (copyKind > 0)
                    EmitAdvancePtr(il, locDp, copyKind);
                else
                {
                    il.Emit(OpCodes.Ldloc, locDp);
                    il.Emit(OpCodes.Ldarg_3);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, locDp);
                }
                il.Emit(OpCodes.Ldloc, locN);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locN);
            }

            if (VectorBits >= 128)
            {
                // The vector stage: L mask bytes a step. ExtractMostSignificantBits of (bytes == 0) marks the ZERO bytes, so
                // its complement, cut to the L lanes, marks the selected slabs.
                int simdBits = VectorBits >= 256 ? 256 : 128;
                int lanes = simdBits / 8;
                uint laneMask = lanes >= 32 ? uint.MaxValue : (1u << lanes) - 1;

                var locVecEnd = il.DeclareLocal(typeof(long));
                var locBits = il.DeclareLocal(typeof(uint));
                var lblVecHead = il.DefineLabel();
                var lblVecEnd = il.DefineLabel();
                var lblBitHead = il.DefineLabel();
                var lblBitEnd = il.DefineLabel();

                // vecEnd = count - L (negative when count < L: the vector stage is skipped)
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locVecEnd);

                il.MarkLabel(lblVecHead);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locVecEnd);
                il.Emit(OpCodes.Bgt, lblVecEnd);

                // bits = ~ExtractMostSignificantBits(Equals(Load(mask + i), Zero)) & laneMask
                il.Emit(OpCodes.Ldarg_1);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Load(simdBits, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(simdBits, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(simdBits, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(simdBits, typeof(byte)), null);
                il.Emit(OpCodes.Not);
                il.Emit(OpCodes.Ldc_I4, unchecked((int)laneMask));
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locBits);

                il.MarkLabel(lblBitHead);
                il.Emit(OpCodes.Ldloc, locBits);
                il.Emit(OpCodes.Brfalse, lblBitEnd);

                // index = i + TrailingZeroCount(bits)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locBits);
                il.EmitCall(OpCodes.Call, BitOpsTrailingZeroCount32, null);
                il.Emit(OpCodes.Conv_I8);
                il.Emit(OpCodes.Add);
                EmitCopySlabAtIndexOnStack();

                // bits &= bits - 1
                il.Emit(OpCodes.Ldloc, locBits);
                il.Emit(OpCodes.Ldloc, locBits);
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locBits);
                il.Emit(OpCodes.Br, lblBitHead);

                il.MarkLabel(lblBitEnd);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblVecHead);

                il.MarkLabel(lblVecEnd);
            }

            // The scalar tail (the whole mask without SIMD): one mask byte at a time.
            var lblTailHead = il.DefineLabel();
            var lblTailNext = il.DefineLabel();
            var lblDone = il.DefineLabel();

            il.MarkLabel(lblTailHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblDone);

            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblTailNext);

            il.Emit(OpCodes.Ldloc, locI);
            EmitCopySlabAtIndexOnStack();

            il.MarkLabel(lblTailNext);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTailHead);

            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldloc, locN);
            il.Emit(OpCodes.Ret);

            return (MaskGatherKernel)dm.CreateDelegate(typeof(MaskGatherKernel));
        }
    }
}
