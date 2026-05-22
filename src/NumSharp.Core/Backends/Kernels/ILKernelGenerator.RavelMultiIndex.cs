using System;
using System.Reflection.Emit;
using System.Threading;

// =============================================================================
// ILKernelGenerator.RavelMultiIndex.cs — IL kernel for np.ravel_multi_index
// =============================================================================
//
// RESPONSIBILITY:
//   np.ravel_multi_index is the inverse of np.unravel_index: given a tuple of
//   per-axis coordinate arrays, return the flat index array. Per element the
//   work is `ndim` mul-add + mode handling (raise / wrap / clip). The kernel
//   is dtype-agnostic (caller casts each coord array to int64) — a single
//   DynamicMethod handles any ndim, both C / F order (encoded via ravel
//   strides at the call site), and per-axis mode selection.
//
// LAYOUT
//   Per-element nest (outer=row, inner=axis) outperforms axis-major
//   (outer=axis, inner=row) at ndim=2 by avoiding the read-modify-write
//   traffic on the accumulator; both layouts converge at ndim>=3 because
//   the per-axis snapshot benefits eventually offset the extra out[] reads.
//   We ship the per-element layout — the simpler one with the better small-
//   ndim profile.
//
// KERNEL (DynamicMethod-emitted, singleton):
//
//   * RavelMultiIndexKernel
//       (long** coords,        // ndim contig int64 buffers (caller casts)
//        long count,
//        long* dims,
//        long* ravelStrides,   // C/F selection baked here at the call site
//        int* modes,           // per-axis: 0=raise, 1=wrap, 2=clip
//        long ndim,
//        long* outIndices)     // contig int64 output buffer
//        -> long: count on success, else the row index of the first OOB
//                 coord under mode=raise.
//
//   On the failure path the caller throws "invalid entry in coordinates array"
//   — NumPy doesn't report axis or value, so neither do we.
//
// WRAP SEMANTICS
//   Matches NumPy's staged fast-path: single +/- m brings j into range for
//   the common one-step-out case; falls back to a modulo + sign-correction
//   chain only when j is multiple-m's outside the range.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted multi-coord→flat-index folder for <c>np.ravel_multi_index</c>.
    /// Caller pre-casts each coord array to contig int64 and computes
    /// <paramref name="ravelStrides"/> (C or F order baked into the strides);
    /// the kernel reads <paramref name="coords"/>[d][i] linearly, applies the
    /// per-axis <paramref name="modes"/> clipping/wrapping, and writes the
    /// summed flat index into <paramref name="outIndices"/>.
    /// </summary>
    /// <returns>
    /// <paramref name="count"/> on success. On failure (mode=raise + OOB coord),
    /// the row index of the first offending row.
    /// </returns>
    public unsafe delegate long RavelMultiIndexKernel(
        long** coords, long count, long* dims, long* ravelStrides,
        int* modes, long ndim, long* outIndices);

    public static partial class ILKernelGenerator
    {
        private static RavelMultiIndexKernel _ravelMultiIndexKernel;

        /// <summary>
        /// IL-emitted multi→flat folder (singleton — same kernel handles any
        /// ndim, both orders, and arbitrary per-axis mode tuples). Returns
        /// <c>null</c> only when <see cref="Enabled"/> is false.
        /// </summary>
        public static RavelMultiIndexKernel GetRavelMultiIndexKernel()
        {
            if (!Enabled)
                return null;

            var cached = _ravelMultiIndexKernel;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateRavelMultiIndexKernelIL();
                Interlocked.CompareExchange(ref _ravelMultiIndexKernel, k, null);
                return _ravelMultiIndexKernel;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetRavelMultiIndexKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the ravel-multi-index kernel. Pseudocode:
        /// <code>
        /// long Ravel(long** coords, long count, long* dims, long* ravelStrides,
        ///            int* modes, long ndim, long* outIndices) {
        ///     for (long i = 0; i &lt; count; i++) {
        ///         long raveled = 0;
        ///         for (long d = 0; d &lt; ndim; d++) {
        ///             long j = coords[d][i];
        ///             long m = dims[d];
        ///             switch (modes[d]) {
        ///                 case 0 /*RAISE*/: if (j &lt; 0 || j &gt;= m) return i; break;
        ///                 case 1 /*WRAP*/:  j = wrap(j, m); break;
        ///                 case 2 /*CLIP*/:  if (j &lt; 0) j = 0; else if (j &gt;= m) j = m-1; break;
        ///             }
        ///             raveled += j * ravelStrides[d];
        ///         }
        ///         outIndices[i] = raveled;
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        private static RavelMultiIndexKernel GenerateRavelMultiIndexKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_RavelMultiIndex",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(long**), // 0 coords
                    typeof(long),   // 1 count
                    typeof(long*),  // 2 dims
                    typeof(long*),  // 3 ravelStrides
                    typeof(int*),   // 4 modes
                    typeof(long),   // 5 ndim
                    typeof(long*),  // 6 outIndices
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locI = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locRaveled = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));
            var locM = il.DeclareLocal(typeof(long));
            var locMode = il.DeclareLocal(typeof(int));
            var locColPtr = il.DeclareLocal(typeof(long*));
            var locStride = il.DeclareLocal(typeof(long));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblDLoopHead = il.DefineLabel();
            var lblDLoopEnd = il.DefineLabel();
            var lblDStep = il.DefineLabel();

            var lblModeWrap = il.DefineLabel();
            var lblModeClip = il.DefineLabel();
            var lblModeRaise = il.DefineLabel();
            var lblWrapDone = il.DefineLabel();
            var lblClipDone = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // ----- Outer loop -----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // raveled = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locRaveled);

            // d = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locD);

            // ----- Per-axis fold -----
            il.MarkLabel(lblDLoopHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 5);     // ndim
            il.Emit(OpCodes.Bge, lblDLoopEnd);

            // colPtr = coords[d]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locColPtr);

            // j = colPtr[i]
            il.Emit(OpCodes.Ldloc, locColPtr);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locJ);

            // m = dims[d]
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locM);

            // mode = modes[d]
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 4L);   // sizeof(int)
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I4);
            il.Emit(OpCodes.Stloc, locMode);

            // Dispatch on mode:
            //   1 -> wrap, 2 -> clip, anything else (incl. 0) -> raise.
            il.Emit(OpCodes.Ldloc, locMode);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Beq, lblModeWrap);
            il.Emit(OpCodes.Ldloc, locMode);
            il.Emit(OpCodes.Ldc_I4_2);
            il.Emit(OpCodes.Beq, lblModeClip);

            // ----- RAISE: validate, no mutation -----
            il.MarkLabel(lblModeRaise);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblFail);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldloc, locM);
            il.Emit(OpCodes.Bge, lblFail);
            il.Emit(OpCodes.Br, lblDStep);

            // ----- WRAP: NumPy's staged wrap (see compiled_base.c:ravel_multi_index_loop) -----
            //   if j < 0:
            //       j += m
            //       if j < 0:
            //           j %= m              // C# % can give negative for neg dividend
            //           if j != 0: j += m
            //   else if j >= m:
            //       j -= m
            //       if j >= m:
            //           j %= m
            //   else: nothing
            il.MarkLabel(lblModeWrap);
            {
                var lblWrapNeg = il.DefineLabel();
                var lblWrapGe = il.DefineLabel();
                var lblWrapNegInner = il.DefineLabel();
                var lblWrapNegInnerEnd = il.DefineLabel();
                var lblWrapGeInner = il.DefineLabel();

                // if (j < 0) goto wrapNeg
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Blt, lblWrapNeg);

                // else if (j >= m) goto wrapGe
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Bge, lblWrapGe);

                // else: no change
                il.Emit(OpCodes.Br, lblWrapDone);

                // wrapNeg: j += m
                il.MarkLabel(lblWrapNeg);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locJ);
                // if (j < 0) need full %
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblWrapDone);
                il.MarkLabel(lblWrapNegInner);
                // j %= m  (still negative or zero after %, since j < 0 here)
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locJ);
                // if (j != 0) j += m
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblWrapNegInnerEnd);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locJ);
                il.MarkLabel(lblWrapNegInnerEnd);
                il.Emit(OpCodes.Br, lblWrapDone);

                // wrapGe: j -= m
                il.MarkLabel(lblWrapGe);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locJ);
                // if (j >= m) j %= m
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Blt, lblWrapDone);
                il.MarkLabel(lblWrapGeInner);
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Rem);
                il.Emit(OpCodes.Stloc, locJ);

                il.MarkLabel(lblWrapDone);
                il.Emit(OpCodes.Br, lblDStep);
            }

            // ----- CLIP: saturate -----
            il.MarkLabel(lblModeClip);
            {
                var lblClipNeg = il.DefineLabel();
                var lblClipGe = il.DefineLabel();
                // if (j < 0) j = 0
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bge, lblClipNeg);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locJ);
                il.Emit(OpCodes.Br, lblClipDone);
                il.MarkLabel(lblClipNeg);
                // else if (j >= m) j = m - 1
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Blt, lblClipDone);
                il.Emit(OpCodes.Ldloc, locM);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Sub);
                il.Emit(OpCodes.Stloc, locJ);
                il.MarkLabel(lblClipDone);
                il.Emit(OpCodes.Br, lblDStep);
            }

            // ----- D-step: raveled += j * ravelStrides[d] -----
            il.MarkLabel(lblDStep);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locStride);

            il.Emit(OpCodes.Ldloc, locRaveled);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldloc, locStride);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locRaveled);

            // d++
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblDLoopHead);

            il.MarkLabel(lblDLoopEnd);

            // outIndices[i] = raveled
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locRaveled);
            il.Emit(OpCodes.Stind_I8);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ----- Fail: return i -----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ret);

            // ----- Success: return count -----
            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ret);

            return (RavelMultiIndexKernel)dm.CreateDelegate(typeof(RavelMultiIndexKernel));
        }
    }
}
