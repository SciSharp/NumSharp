using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.TakeAlongAxis.cs — IL gather kernel for np.take_along_axis
// =============================================================================
//
// RESPONSIBILITY:
//   np.take_along_axis(arr, indices, axis) is a per-element gather: for every
//   position in the (broadcast) result it reads ONE index out of `indices` and
//   uses it to look up ONE element of `arr` along `axis`. NumPy implements it as
//   advanced indexing (arr[_make_along_axis_idx(...)]); we implement the same
//   semantics as a single whole-array strided odometer (the DirectILKernelGenerator
//   contract — the kernel walks dimensions/strides itself).
//
//   Unlike np.take (one index vector shared by every 1-D slice), the index here
//   varies per output element, so there is no bulk-slab copy: the loop moves one
//   element (elemBytes) at a time. The kernel is therefore keyed by `elemBytes`
//   and emits a typed load/store for the exact element width (1/2/4/8/16), which
//   is meaningfully faster than a variable-size `cpblk` in this per-element loop.
//
//   NO SIMD GATHER — measured, not assumed. A hardware gather (VPGATHERQD/QQ) was
//   benchmarked against this scalar loop for the hot 4/8-byte, axis=last, contiguous
//   case and came in at only ~1.16x on this host (AVX2; AVX-512 gather unavailable) —
//   BEFORE the SIMD index bounds-validation and the scalar OOB-fallback a raise-mode
//   gather still needs. That does not justify a per-width gather kernel, its
//   correctness surface, or its CPU-dependence (gather is at/below parity on several
//   microarchitectures); NumPy's own advanced-index gather is likewise a scalar loop.
//   The wins that DID pay off are structural: the outer-odometer/inner-loop split
//   (per-slice carry, below) and the branchless-free but branch-light resolve.
//
//   The result buffer is freshly allocated C-contiguous, so the destination is
//   written linearly (dst + flat*elemBytes). Both `indices` and `arr` are read
//   through per-result-dimension strides, so ANY layout (C/F/strided/reversed/
//   sliced/broadcast) works with no materialisation — a broadcast source or index
//   dimension is just a stride-0 entry, and `arr` keeps its own (possibly negative)
//   axis stride for the gather. Advanced indexing wraps a single negative index and
//   raises on anything still out of range, which is exactly the RAISE mode here.
//
// KERNEL (DynamicMethod-emitted, cached per elemBytes):
//
//   long TakeAlongAxis(
//       byte* arrBase,          // arr.Address + arr.offset*elem
//       long* arrStrides,       // per result-dim arr stride in BYTES (0 at axis & at broadcast dims)
//       long  axisStrideBytes,  // arr stride along `axis` in BYTES (used with the resolved index)
//       long  axisLen,          // M = arr.shape[axis]
//       long* idxBase,          // int64 index buffer (idx array offset already applied)
//       long* idxStrides,       // per result-dim idx stride in ELEMENTS (0 at broadcast dims)
//       byte* dstBase,          // contig result buffer (caller-allocated)
//       long* shape,            // result dims
//       long  ndim,             // result ndim (>= 1)
//       long  totalSize,        // result element count (> 0)
//       long* outBadIdx)        // set to the offending ORIGINAL index on RAISE OOB
//       -> long: totalSize on success, else the flat position of the first OOB
//                element (with *outBadIdx holding its pre-wrap index value).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted per-element gather kernel for <c>np.take_along_axis</c>. Walks the
    /// C-contiguous result with an incremental strided odometer, reading one int64
    /// index and gathering one <c>elemBytes</c>-wide element of the source per step.
    /// </summary>
    /// <returns>
    /// <c>totalSize</c> on success. On RAISE out-of-bounds the returned value is the
    /// flat position of the first failing element and <c>*outBadIdx</c> holds its
    /// original (pre-wrap) index value for the caller's diagnostic.
    /// </returns>
    public unsafe delegate long TakeAlongAxisKernel(
        byte* arrBase, long* arrStrides, long axisStrideBytes, long axisLen,
        long* idxBase, long* idxStrides, byte* dstBase,
        long* shape, long ndim, long totalSize, long* outBadIdx);

    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<int, TakeAlongAxisKernel> _takeAlongAxisKernels
            = new ConcurrentDictionary<int, TakeAlongAxisKernel>();

        /// <summary>
        /// IL-emitted take_along_axis gather kernel, cached per element width
        /// (<paramref name="elemBytes"/> ∈ {1,2,4,8,16}). Returns <c>null</c> only
        /// when <see cref="Enabled"/> is false.
        /// </summary>
        public static TakeAlongAxisKernel GetTakeAlongAxisKernel(int elemBytes)
        {
            if (!Enabled)
                return null;

            if (_takeAlongAxisKernels.TryGetValue(elemBytes, out var cached))
                return cached;

            try
            {
                return _takeAlongAxisKernels.GetOrAdd(elemBytes, GenerateTakeAlongAxisKernelIL);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTakeAlongAxisKernel({elemBytes}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the gather kernel for a fixed element width. The traversal is an OUTER odometer
        /// over dimensions <c>[0, ndim-1)</c> wrapped around a tight INNER loop over the innermost
        /// dimension, so the multi-level carry runs once per 1-D slice rather than once per element
        /// (the innermost dimension is usually the take axis, i.e. the argsort/argmax pattern).
        /// Pseudocode:
        /// <code>
        /// long TakeAlongAxis(...) {
        ///     long* coord = stackalloc long[ndim];      // outer dims, zeroed
        ///     long lastN = shape[ndim-1], aLast = arrStrides[ndim-1], iLast = idxStrides[ndim-1];
        ///     long flat = 0, arrOuter = 0, idxOuter = 0;
        ///     while (flat &lt; totalSize) {
        ///         long arrOff = arrOuter, idxOff = idxOuter;
        ///         for (long t = 0; t &lt; lastN; t++) {   // innermost dimension
        ///             long idx = idxBase[idxOff];        // original (for diagnostics)
        ///             long r = idx;
        ///             if (r &lt; 0) r += axisLen;         // advanced-index single wrap
        ///             if (r &lt; 0 || r &gt;= axisLen) { *outBadIdx = idx; return flat; }
        ///             *(T*)(dstBase + flat*elemBytes) = *(T*)(arrBase + arrOff + r*axisStrideBytes);
        ///             flat++; arrOff += aLast; idxOff += iLast;
        ///         }
        ///         for (long d = ndim - 2; d &gt;= 0; d--) {   // outer odometer
        ///             coord[d]++; arrOuter += arrStrides[d]; idxOuter += idxStrides[d];
        ///             if (coord[d] &lt; shape[d]) break;
        ///             coord[d] = 0;
        ///             arrOuter -= arrStrides[d] * shape[d];
        ///             idxOuter -= idxStrides[d] * shape[d];
        ///         }
        ///     }
        ///     return totalSize;
        /// }
        /// </code>
        /// </summary>
        private static TakeAlongAxisKernel GenerateTakeAlongAxisKernelIL(int elemBytes)
        {
            var dm = new DynamicMethod(
                name: $"IL_TakeAlongAxis_{elemBytes}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 arrBase
                    typeof(long*),  // 1 arrStrides
                    typeof(long),   // 2 axisStrideBytes
                    typeof(long),   // 3 axisLen
                    typeof(long*),  // 4 idxBase
                    typeof(long*),  // 5 idxStrides
                    typeof(byte*),  // 6 dstBase
                    typeof(long*),  // 7 shape
                    typeof(long),   // 8 ndim
                    typeof(long),   // 9 totalSize
                    typeof(long*),  // 10 outBadIdx
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoord = il.DeclareLocal(typeof(long*));
            var locFlat = il.DeclareLocal(typeof(long));
            var locArrOuter = il.DeclareLocal(typeof(long));
            var locIdxOuter = il.DeclareLocal(typeof(long));
            var locArrOff = il.DeclareLocal(typeof(long));
            var locIdxOff = il.DeclareLocal(typeof(long));
            var locT = il.DeclareLocal(typeof(long));
            var locLastN = il.DeclareLocal(typeof(long));
            var locALast = il.DeclareLocal(typeof(long));
            var locILast = il.DeclareLocal(typeof(long));
            var locIdxVal = il.DeclareLocal(typeof(long));
            var locResolved = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));

            var lblZeroHead = il.DefineLabel();
            var lblZeroEnd = il.DefineLabel();
            var lblOuterHead = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();
            var lblBounds = il.DefineLabel();
            var lblCarryHead = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblDone = il.DefineLabel();

            // coord = stackalloc long[ndim]
            il.Emit(OpCodes.Ldarg, 8);            // ndim
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoord);

            // for (d=0; d<ndim; d++) coord[d]=0;
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locD);
            il.MarkLabel(lblZeroHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 8);
            il.Emit(OpCodes.Bge, lblZeroEnd);
            EmitElemAddr(il, locCoord, locD);     // &coord[d]
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblZeroHead);
            il.MarkLabel(lblZeroEnd);

            // lastN = shape[ndim-1]; aLast = arrStrides[ndim-1]; iLast = idxStrides[ndim-1];
            il.Emit(OpCodes.Ldarg, 8); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locD);
            EmitElemLoad(il, 7, locD); il.Emit(OpCodes.Stloc, locLastN);
            EmitElemLoad(il, 1, locD); il.Emit(OpCodes.Stloc, locALast);
            EmitElemLoad(il, 5, locD); il.Emit(OpCodes.Stloc, locILast);

            // flat = arrOuter = idxOuter = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locArrOuter);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locIdxOuter);

            // ---- outer loop ----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldarg, 9);            // totalSize
            il.Emit(OpCodes.Bge, lblDone);
            // arrOff = arrOuter; idxOff = idxOuter; t = 0
            il.Emit(OpCodes.Ldloc, locArrOuter); il.Emit(OpCodes.Stloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locIdxOuter); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locT);

            // ---- inner loop over the innermost dimension ----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locT);
            il.Emit(OpCodes.Ldloc, locLastN);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // idxVal = idxBase[idxOff]
            il.Emit(OpCodes.Ldarg, 4);            // idxBase
            il.Emit(OpCodes.Ldloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locIdxVal);

            // Advanced-index resolve: a single conditional wrap, then ONE unsigned bounds compare.
            //   if (resolved < 0) resolved += axisLen;             // wrap once (advanced-index rule)
            //   if ((ulong)resolved >= (ulong)axisLen) goto Fail;  // catches BOTH still-<0 and >=axisLen
            // The wrap stays a BRANCH (not a branchless shift/and/add) deliberately: take_along_axis
            // indices are overwhelmingly non-negative (argsort/argmax output), so this branch predicts
            // not-taken and costs ~nothing, whereas an unconditional sign-shift+and+add would spend ALU
            // on every element for a wrap that almost never fires. The unsigned compare then folds the
            // two remaining bounds checks into one (a negative result reads as a huge unsigned).
            il.Emit(OpCodes.Ldloc, locIdxVal);
            il.Emit(OpCodes.Stloc, locResolved);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblBounds);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 3);            // axisLen
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locResolved);
            il.MarkLabel(lblBounds);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 3);
            il.Emit(OpCodes.Bge_Un, lblFail);     // (ulong)resolved >= (ulong)axisLen

            // src = arrBase + arrOff + resolved * axisStrideBytes
            il.Emit(OpCodes.Ldarg, 0);            // arrBase
            il.Emit(OpCodes.Ldloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 2);            // axisStrideBytes
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // dst = dstBase + flat * elemBytes
            il.Emit(OpCodes.Ldarg, 6);            // dstBase
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, (long)elemBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            // *(T*)dst = *(T*)src  (typed copy of elemBytes)
            EmitElemCopy(il, locDst, locSrc, elemBytes);

            // flat++; arrOff += aLast; idxOff += iLast; t++
            il.Emit(OpCodes.Ldloc, locFlat); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldloc, locArrOff); il.Emit(OpCodes.Ldloc, locALast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locIdxOff); il.Emit(OpCodes.Ldloc, locILast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldloc, locT); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locT);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ---- outer odometer advance: for (d=ndim-2; d>=0; d--) ----
            il.Emit(OpCodes.Ldarg, 8);            // ndim
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            // if (d < 0) goto OuterHead;  (ndim == 1 -> no outer dims)
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblOuterHead);

            il.MarkLabel(lblCarryHead);
            // coord[d]++
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);
            // arrOuter += arrStrides[d]
            il.Emit(OpCodes.Ldloc, locArrOuter);
            EmitElemLoad(il, 1, locD);            // arrStrides[d]
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locArrOuter);
            // idxOuter += idxStrides[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 5, locD);            // idxStrides[d]
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // if (coord[d] < shape[d]) goto OuterHead;
            EmitElemLoad(il, -1, locD, locCoord); // coord[d]
            EmitElemLoad(il, 7, locD);            // shape[d]
            il.Emit(OpCodes.Blt, lblOuterHead);
            // carry: coord[d] = 0
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            // arrOuter -= arrStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locArrOuter);
            EmitElemLoad(il, 1, locD);
            EmitElemLoad(il, 7, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locArrOuter);
            // idxOuter -= idxStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 5, locD);
            EmitElemLoad(il, 7, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            // if (d >= 0) goto CarryHead; else fall through -> OuterHead exits via flat>=totalSize
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblCarryHead);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ---- fail: *outBadIdx = idxVal; return flat ----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldarg, 10);           // outBadIdx
            il.Emit(OpCodes.Ldloc, locIdxVal);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ret);

            // ---- done: return totalSize ----
            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg, 9);
            il.Emit(OpCodes.Ret);

            return (TakeAlongAxisKernel)dm.CreateDelegate(typeof(TakeAlongAxisKernel));
        }

        /// <summary>Push <c>&amp;base[d]</c> for a <c>long*</c> local (base + d*8).</summary>
        private static void EmitElemAddr(ILGenerator il, LocalBuilder locBase, LocalBuilder locD)
        {
            il.Emit(OpCodes.Ldloc, locBase);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary>
        /// Push <c>base[d]</c> (a long). When <paramref name="argIndex"/> ≥ 0 the base
        /// pointer is that <c>long*</c> argument; when -1 it is the <paramref name="locBase"/>
        /// local (used for <c>coord[d]</c>).
        /// </summary>
        private static void EmitElemLoad(ILGenerator il, int argIndex, LocalBuilder locD, LocalBuilder locBase = null)
        {
            if (argIndex >= 0)
                il.Emit(OpCodes.Ldarg, argIndex);
            else
                il.Emit(OpCodes.Ldloc, locBase);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
        }

        /// <summary>Emit a typed element copy <c>*dst = *src</c> for the given byte width.</summary>
        private static void EmitElemCopy(ILGenerator il, LocalBuilder locDst, LocalBuilder locSrc, int elemBytes)
        {
            switch (elemBytes)
            {
                case 1:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldind_U1);
                    il.Emit(OpCodes.Stind_I1);
                    break;
                case 2:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldind_U2);
                    il.Emit(OpCodes.Stind_I2);
                    break;
                case 4:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldind_U4);
                    il.Emit(OpCodes.Stind_I4);
                    break;
                case 8:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    break;
                case 16:
                    // low 8 bytes
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    // high 8 bytes
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldc_I4_8);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldc_I4_8);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stind_I8);
                    break;
                default:
                    // Fallback for any other width (not reached by the 15 NumSharp dtypes).
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldc_I4, elemBytes);
                    il.Emit(OpCodes.Cpblk);
                    break;
            }
        }
    }
}
