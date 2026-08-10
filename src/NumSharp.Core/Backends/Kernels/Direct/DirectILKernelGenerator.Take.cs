using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;

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
        private static readonly ConcurrentDictionary<int, TakeKernel> _takeKernels = new();

        // Software-prefetch tuning for the random-gather inner loop. NumPy's take is a scalar memcpy
        // with no prefetch, so at scale (source larger than LLC) it stalls on gather latency; issuing
        // a prefetch for the slab DIST indices ahead raises memory-level parallelism and is a pure win
        // there, neutral when the source is already cache-resident (prefetch of a live line is free).
        // Emitted only when SSE is available (x86); a no-op address never faults, so unnormalized
        // future indices are safe to prefetch.
        private const long PrefetchDistance = 32;
        private static readonly MethodInfo _ssePrefetch0 = typeof(Sse).GetMethod(nameof(Sse.Prefetch0));

        /// <summary>
        /// IL-emitted take kernel, generated once per (copy-width, prefetch) pair. <paramref name="copyKind"/>
        /// is the <see cref="CopyKindFor"/> of the per-index slab size (1/2/4/8/16 for a typed MOV copy,
        /// 0 for a runtime-sized <c>cpblk</c>); <paramref name="prefetch"/> emits the software-prefetch
        /// inner body (a large-source win, a small-source pessimization — the caller gates it on the
        /// gathered region exceeding cache). One kernel handles any ndim, both axis=None and axis=k.
        /// Returns <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static TakeKernel GetTakeKernel(int copyKind, bool prefetch)
        {
            if (!Enabled)
                return null;

            int key = (copyKind << 1) | (prefetch ? 1 : 0);
            if (_takeKernels.TryGetValue(key, out var cached))
                return cached;

            try
            {
                var k = GenerateTakeKernelIL(copyKind, prefetch);
                return _takeKernels.GetOrAdd(key, k);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTakeKernel({copyKind},{prefetch}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        // ---- shared element-copy specialization (used by Take / Put / Place) ----

        /// <summary>
        ///     The typed-copy width for a slab of <paramref name="bytes"/> bytes: 1/2/4/8/16 when the
        ///     slab is exactly one primitive width (a single element for every NumSharp dtype, since all
        ///     itemsizes are in {1,2,4,8,16}), else 0 meaning "use a runtime-sized cpblk". Selecting a
        ///     typed width lets the copy compile to one or two MOVs instead of a per-element memcpy —
        ///     the cpblk-of-a-tiny-runtime-size pathology that made the gather/scatter kernels ~2x slow.
        /// </summary>
        internal static int CopyKindFor(long bytes) => bytes switch
        {
            1 => 1,
            2 => 2,
            4 => 4,
            8 => 8,
            16 => 16,
            _ => 0,
        };

        /// <summary>
        ///     Emit the per-element copy for a gather/scatter inner body. For a fixed
        ///     <paramref name="copyKind"/> width this is a typed load+store (one MOV, two for 16 bytes);
        ///     for copyKind 0 it is <c>cpblk</c> with a runtime byte count pushed by
        ///     <paramref name="pushByteCount"/>. Consumes nothing else; leaves the stack empty.
        /// </summary>
        internal static void EmitElementCopy(ILGenerator il, int copyKind, LocalBuilder dst, LocalBuilder src, Action pushByteCount)
        {
            switch (copyKind)
            {
                case 1:
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldind_U1); il.Emit(OpCodes.Stind_I1); break;
                case 2:
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldind_U2); il.Emit(OpCodes.Stind_I2); break;
                case 4:
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldind_U4); il.Emit(OpCodes.Stind_I4); break;
                case 8:
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stind_I8); break;
                case 16:
                    // Two 8-byte MOVs cover Complex/Decimal without needing SSE.
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stind_I8);
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldc_I4_8); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, src); il.Emit(OpCodes.Ldc_I4_8); il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldind_I8); il.Emit(OpCodes.Stind_I8); break;
                default:
                    il.Emit(OpCodes.Ldloc, dst); il.Emit(OpCodes.Ldloc, src); pushByteCount(); il.Emit(OpCodes.Conv_U4); il.Emit(OpCodes.Cpblk); break;
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
        private static TakeKernel GenerateTakeKernelIL(int copyKind, bool prefetch)
        {
            var dm = new DynamicMethod(
                name: $"IL_Take_c{copyKind}_{(prefetch ? "pf" : "np")}",
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

            bool usePrefetch = prefetch && Sse.IsSupported && _ssePrefetch0 != null;

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

            // Prefetch the slab that index (j + DIST) will gather, so its cache line is in flight
            // before we reach it. Guarded by (j + DIST < indicesCount) so the index-array read never
            // runs past its allocation; the prefetch target itself uses the RAW future index (no mode
            // normalization) which is fine — an off / negative address just wastes the prefetch.
            if (usePrefetch)
            {
                var lblSkipPf = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, PrefetchDistance);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_2);            // indicesCount
                il.Emit(OpCodes.Bge, lblSkipPf);

                il.Emit(OpCodes.Ldarg_0);            // src
                il.Emit(OpCodes.Ldloc, locOuter);
                il.Emit(OpCodes.Ldarg, 4);           // maxItem
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Ldarg_1);            // indices
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, PrefetchDistance);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldc_I8, 8L);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I8);           // indices[j + DIST]
                il.Emit(OpCodes.Add);                // outer*maxItem + idxFuture
                il.Emit(OpCodes.Ldarg, 5);           // innerSize
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);                // src + (outer*maxItem + idxFuture) * innerSize
                il.Emit(OpCodes.Call, _ssePrefetch0);
                il.MarkLabel(lblSkipPf);
            }

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

            // Element copy: a typed MOV (or two, for 16-byte Complex/Decimal) when the slab is one
            // primitive width — the common gather case — else cpblk with the runtime innerSize (arg 5).
            // For the typed widths innerSize == copyKind, so the slab addressing above stays correct.
            // (Cpblk byte count is uint32; per-slab sizes > 2^32 can't arise — they'd need > 4 GB per
            // element, beyond NDArray capacity.)
            EmitElementCopy(il, copyKind, locDstSlab, locSrcSlab, () => il.Emit(OpCodes.Ldarg, 5));

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
            // NumPy's check_and_adjust_index (mapping.c): a negative index is normalized ONCE
            // (idx += maxItem), then bounds-checked — so take([-1]) is the last element, while an
            // index still out of range after the shift (idx < 0 || idx >= maxItem) fails. The
            // original (pre-shift) value survives in indices[] for the caller's diagnostic.
            //   if (idx < 0) idx += maxItem;
            //   if (idx < 0 || idx >= maxItem) goto fail;
            var lblRaiseBounds = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblRaiseBounds);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, 4);               // maxItem
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdx);
            il.MarkLabel(lblRaiseBounds);
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
