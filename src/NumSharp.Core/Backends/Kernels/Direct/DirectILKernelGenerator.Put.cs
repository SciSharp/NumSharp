using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// DirectILKernelGenerator.Put.cs — IL kernel for np.put
// =============================================================================
//
// RESPONSIBILITY:
//   np.put scatters values into a target array at flat-indexed positions, with
//   mode-handled out-of-bounds resolution and cyclic broadcasting of `values`
//   when shorter than `indices`. The kernel is dtype-agnostic via byte-level
//   <c>cpblk</c> with a runtime <c>elemBytes</c> argument.
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * PutKernel
//       (byte* dst,            // target flat buffer (in-place)
//        long* indices,        // contig int64
//        long indicesCount,
//        byte* values,         // contig source buffer
//        long valuesCount,     // > 0; caller short-circuits when 0
//        long maxItem,         // dst.size (for mode check)
//        long elemBytes,
//        int mode)             // 0=raise, 1=wrap, 2=clip
//        -> long: indicesCount on success, or i on RAISE OOB.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted scatter kernel for <c>np.put</c>. Writes
    /// <c>dst[apply_mode(indices[i])] = values[i % valuesCount]</c> for each
    /// <c>i</c> in <c>[0, indicesCount)</c>.
    /// </summary>
    /// <returns>
    /// <paramref name="indicesCount"/> on success. On RAISE OOB the row index
    /// of the first failing entry; the caller reads <c>indices[returned]</c>
    /// for the diagnostic.
    /// </returns>
    public unsafe delegate long PutKernel(
        byte* dst, long* indices, long indicesCount,
        byte* values, long valuesCount,
        long maxItem, long elemBytes, int mode);

    public static partial class DirectILKernelGenerator
    {
        private static PutKernel _putKernel;

        /// <summary>
        /// IL-emitted put kernel (singleton — same kernel handles any dtype
        /// via the <c>elemBytes</c> runtime argument and any mode).
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static PutKernel GetPutKernel()
        {
            if (!Enabled)
                return null;

            var cached = _putKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GeneratePutKernelIL();
                Interlocked.CompareExchange(ref _putKernel, k, null);
                return _putKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPutKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the put kernel. Pseudocode:
        /// <code>
        /// long Put(byte* dst, long* indices, long ni,
        ///          byte* values, long nv, long maxItem,
        ///          long elemBytes, int mode) {
        ///     for (long i = 0; i &lt; ni; i++) {
        ///         long idx = indices[i];
        ///         switch (mode) {
        ///             case 0: if (idx &lt; 0 || idx &gt;= maxItem) return i; break;
        ///             case 1: idx = wrap(idx, maxItem); break;
        ///             case 2: if (idx&lt;0) idx=0; else if (idx&gt;=maxItem) idx=maxItem-1; break;
        ///         }
        ///         byte* srcPtr = values + (i % nv) * elemBytes;
        ///         byte* dstPtr = dst + idx * elemBytes;
        ///         cpblk(dstPtr, srcPtr, elemBytes);
        ///     }
        ///     return ni;
        /// }
        /// </code>
        /// </summary>
        private static PutKernel GeneratePutKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_Put",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 dst
                    typeof(long*),  // 1 indices
                    typeof(long),   // 2 indicesCount
                    typeof(byte*),  // 3 values
                    typeof(long),   // 4 valuesCount
                    typeof(long),   // 5 maxItem
                    typeof(long),   // 6 elemBytes
                    typeof(int),    // 7 mode
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locSrcPtr = il.DeclareLocal(typeof(byte*));
            var locDstPtr = il.DeclareLocal(typeof(byte*));

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblIdxResolved = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblEnd);

            // idx = indices[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locIdx);

            // Mode dispatch — reuse the Take helper. The Take kernel's mode helper
            // uses arg 4 for maxItem; here our maxItem is arg 5. We can't reuse the
            // helper directly without shifting arg slots, so we re-emit the dispatch
            // inline against arg 5.
            EmitPutModeDispatch(il, locIdx, lblFail, lblIdxResolved);

            il.MarkLabel(lblIdxResolved);

            // srcPtr = values + (i % valuesCount) * elemBytes
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);          // valuesCount
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Ldarg, 6);          // elemBytes
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrcPtr);

            // dstPtr = dst + idx * elemBytes
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstPtr);

            // cpblk(dstPtr, srcPtr, elemBytes)
            il.Emit(OpCodes.Ldloc, locDstPtr);
            il.Emit(OpCodes.Ldloc, locSrcPtr);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Conv_U4);
            il.Emit(OpCodes.Cpblk);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ret);

            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ret);

            return (PutKernel)dm.CreateDelegate(typeof(PutKernel));
        }

        /// <summary>
        /// Mode dispatch for <c>np.put</c> — same semantics as the Take helper
        /// but uses arg 5 for <c>maxItem</c> instead of arg 4.
        /// </summary>
        private static void EmitPutModeDispatch(
            ILGenerator il, LocalBuilder locIdx, Label lblFail, Label lblResolved)
        {
            var lblWrap = il.DefineLabel();
            var lblClip = il.DefineLabel();

            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Beq, lblWrap);
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldc_I4_2);
            il.Emit(OpCodes.Beq, lblClip);

            // RAISE
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Bge, lblFail);
            il.Emit(OpCodes.Br, lblResolved);

            // WRAP
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
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Bge, lblWrapGe);
                il.Emit(OpCodes.Br, lblWrapDone);

                il.MarkLabel(lblWrapNeg);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblWrapDone);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblWrapNegInnerEnd);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locIdx);
                il.MarkLabel(lblWrapNegInnerEnd);
                il.Emit(OpCodes.Br, lblWrapDone);

                il.MarkLabel(lblWrapGe);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Blt, lblWrapDone);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locIdx);

                il.MarkLabel(lblWrapDone);
                il.Emit(OpCodes.Br, lblResolved);
            }

            // CLIP
            il.MarkLabel(lblClip);
            {
                var lblClipDone = il.DefineLabel();
                var lblClipGe = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblClipGe);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locIdx);
                il.Emit(OpCodes.Br, lblClipDone);

                il.MarkLabel(lblClipGe);
                il.Emit(OpCodes.Ldloc, locIdx);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Blt, lblClipDone);
                il.Emit(OpCodes.Ldarg, 5);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locIdx);

                il.MarkLabel(lblClipDone);
                il.Emit(OpCodes.Br, lblResolved);
            }
        }
    }
}
