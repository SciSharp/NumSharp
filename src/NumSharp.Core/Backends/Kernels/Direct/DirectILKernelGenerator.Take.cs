using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.Take.cs — IL kernel for np.take
// =============================================================================
//
// RESPONSIBILITY:
//   np.take gathers slices from a source array using an integer-index array.
//   For axis=None: take from the flattened source (1-element-per-index gather).
//   For axis=k:    take slabs of `innerSize` bytes along the k-th axis; output
//                   shape is src.shape[:k] + indices.shape + src.shape[k+1:].
//
//   Both cases share the same kernel — axis=None is just (outerSize=1,
//   maxItem=src.size, innerSize=elemBytes). The dtype-agnostic byte-level copy
//   inside the loop uses the IL `cpblk` opcode, which the JIT lowers to
//   architecture-optimal memcpy (rep movsb / vector copy depending on size).
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * TakeKernel
//       (byte* src,            // contig source buffer
//        long* indices,        // contig int64 indices
//        long indicesCount,    // m: index count
//        long outerSize,       // n: product of src.shape[:axis] (1 for axis=None)
//        long maxItem,         // src.shape[axis] (or src.size for axis=None)
//        long innerSize,       // bytes per gathered slab (= elemBytes * inner_dims)
//        int mode,             // 0=raise, 1=wrap, 2=clip
//        byte* dst)            // contig dest buffer (caller-allocated)
//        -> long: count of fully-completed (outer × index) pairs; less than
//                 outerSize * indicesCount only on RAISE OOB, where the
//                 returned value is the offending pair index (caller reads
//                 indices[returned % indicesCount] for the diagnostic).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted gather kernel for <c>np.take</c>. The source is treated as a
    /// 3-D layout (outerSize, maxItem, innerSize-bytes). For each (outer, j) pair
    /// the kernel reads <c>indices[j]</c>, applies <paramref name="mode"/>, and
    /// copies <c>innerSize</c> bytes from the source slab to the destination
    /// position.
    /// </summary>
    /// <returns>
    /// <c>outerSize * indicesCount</c> on success. On RAISE OOB the returned
    /// value is the row-major index of the first failing (outer, j) pair.
    /// </returns>
    public unsafe delegate long TakeKernel(
        byte* src, long* indices, long indicesCount, long outerSize,
        long maxItem, long innerSize, int mode, byte* dst);

    public static partial class DirectILKernelGenerator
    {
        private static TakeKernel _takeKernel;

        /// <summary>
        /// IL-emitted take kernel (singleton — same kernel handles any ndim,
        /// any elemBytes, any innerSize, both axis=None and axis=k). Returns
        /// <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static TakeKernel GetTakeKernel()
        {
            if (!Enabled)
                return null;

            var cached = _takeKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateTakeKernelIL();
                Interlocked.CompareExchange(ref _takeKernel, k, null);
                return _takeKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTakeKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the take kernel. Pseudocode:
        /// <code>
        /// long Take(byte* src, long* indices, long m, long n,
        ///           long maxItem, long innerSize, int mode, byte* dst) {
        ///     for (long outer = 0; outer &lt; n; outer++) {
        ///         for (long j = 0; j &lt; m; j++) {
        ///             long idx = indices[j];
        ///             switch (mode) {
        ///                 case 0: if (idx &lt; 0 || idx &gt;= maxItem) return outer*m+j; break;
        ///                 case 1: idx = wrap(idx, maxItem); break;
        ///                 case 2: if (idx&lt;0) idx=0; else if (idx&gt;=maxItem) idx=maxItem-1; break;
        ///             }
        ///             byte* srcSlab = src + (outer * maxItem + idx) * innerSize;
        ///             byte* dstSlab = dst + (outer * m + j) * innerSize;
        ///             cpblk(dstSlab, srcSlab, innerSize);
        ///         }
        ///     }
        ///     return n * m;
        /// }
        /// </code>
        /// </summary>
        private static TakeKernel GenerateTakeKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_Take",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(long*),  // 1 indices
                    typeof(long),   // 2 indicesCount
                    typeof(long),   // 3 outerSize
                    typeof(long),   // 4 maxItem
                    typeof(long),   // 5 innerSize
                    typeof(int),    // 6 mode
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locOuter = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locPair = il.DeclareLocal(typeof(long));   // outer*m + j (also serves as the failure return)
            var locSrcSlab = il.DeclareLocal(typeof(byte*));
            var locDstSlab = il.DeclareLocal(typeof(byte*));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblJHead = il.DefineLabel();
            var lblJEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblIdxResolved = il.DefineLabel();

            // outer = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuter);

            // ----- Outer loop -----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldarg_3);                // outerSize
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // ----- Index loop -----
            il.MarkLabel(lblJHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);                // indicesCount
            il.Emit(OpCodes.Bge, lblJEnd);

            // pair = outer * indicesCount + j (used both for fail return and for dst offset)
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locPair);

            // idx = indices[j]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locIdx);

            // ----- Mode dispatch -----
            EmitTakeModeDispatch(il, locIdx, lblFail, lblIdxResolved);

            il.MarkLabel(lblIdxResolved);

            // srcSlab = src + (outer * maxItem + idx) * innerSize
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldarg, 4);               // maxItem
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldarg, 5);               // innerSize
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrcSlab);

            // dstSlab = dst + pair * innerSize
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locPair);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstSlab);

            // cpblk(dstSlab, srcSlab, innerSize) — Cpblk byte count is uint32; for
            // innerSize > 2^32 we'd need a chunked loop, but per-slab sizes that
            // large don't arise in practice (would require > 4 GB per element which
            // exceeds NDArray capacity).
            il.Emit(OpCodes.Ldloc, locDstSlab);
            il.Emit(OpCodes.Ldloc, locSrcSlab);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Conv_U4);
            il.Emit(OpCodes.Cpblk);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblJHead);

            il.MarkLabel(lblJEnd);

            // outer++
            il.Emit(OpCodes.Ldloc, locOuter);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuter);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);

            // Success: return outerSize * indicesCount
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ret);

            // Fail: return pair (the row-major index of the failing (outer, j))
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locPair);
            il.Emit(OpCodes.Ret);

            return (TakeKernel)dm.CreateDelegate(typeof(TakeKernel));
        }

        /// <summary>
        /// Emits mode-handling for <c>idx</c> against <c>arg.maxItem</c>. After
        /// the block, the value in <c>locIdx</c> is in <c>[0, maxItem)</c>; on
        /// RAISE OOB control jumps to <paramref name="lblFail"/>.
        /// </summary>
        private static void EmitTakeModeDispatch(
            ILGenerator il, LocalBuilder locIdx, Label lblFail, Label lblResolved)
        {
            var lblWrap = il.DefineLabel();
            var lblClip = il.DefineLabel();

            // mode = arg 6
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Beq, lblWrap);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Ldc_I4_2);
            il.Emit(OpCodes.Beq, lblClip);

            // ----- RAISE -----
            // if (idx < 0 || idx >= maxItem) goto fail
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblFail);
            il.Emit(OpCodes.Br, lblResolved);

            // ----- WRAP — NumPy's staged form -----
            //   if (idx < 0) { idx += m; if (idx < 0) { idx %= m; if (idx != 0) idx += m; } }
            //   else if (idx >= m) { idx -= m; if (idx >= m) idx %= m; }
            il.MarkLabel(lblWrap);
            {
                var lblWrapNeg = il.DefineLabel();
                var lblWrapGe = il.DefineLabel();
                var lblWrapNegInnerEnd = il.DefineLabel();
                var lblWrapDone = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Blt, lblWrapNeg);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Bge, lblWrapGe);
                il.Emit(OpCodes.Br, lblWrapDone);

                il.MarkLabel(lblWrapNeg);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblWrapDone);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblWrapNegInnerEnd);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locIdx);
                il.MarkLabel(lblWrapNegInnerEnd);
                il.Emit(OpCodes.Br, lblWrapDone);

                il.MarkLabel(lblWrapGe);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Blt, lblWrapDone);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locIdx);

                il.MarkLabel(lblWrapDone);
                il.Emit(OpCodes.Br, lblResolved);
            }

            // ----- CLIP -----
            il.MarkLabel(lblClip);
            {
                var lblClipDone = il.DefineLabel();
                var lblClipGe = il.DefineLabel();

                // if (idx < 0) idx = 0
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblClipGe);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Br, lblClipDone);

                // else if (idx >= maxItem) idx = maxItem - 1
                il.MarkLabel(lblClipGe);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Blt, lblClipDone);
                il.Emit(OpCodes.Ldarg, 4);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locIdx);

                il.MarkLabel(lblClipDone);
                il.Emit(OpCodes.Br, lblResolved);
            }
        }
    }
}
