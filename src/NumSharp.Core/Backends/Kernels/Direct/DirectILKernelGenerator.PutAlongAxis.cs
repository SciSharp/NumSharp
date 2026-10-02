using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.PutAlongAxis.cs — IL scatter kernels for np.put_along_axis
// =============================================================================
//
// RESPONSIBILITY:
//   np.put_along_axis(arr, indices, values, axis) is the SETTER twin of
//   take_along_axis: for every position in the (broadcast) iteration space it
//   reads ONE index out of `indices` and writes ONE value (read from the
//   broadcast `values`) into `arr` along `axis`. NumPy implements it as advanced
//   assignment (arr[_make_along_axis_idx(...)] = values); we implement the same
//   semantics as a whole-array strided odometer (the DirectILKernelGenerator
//   contract — the kernel walks dimensions/strides itself), the exact mirror of
//   the TakeAlongAxis gather.
//
//   ATOMICITY (two kernels, not one). NumPy's advanced assignment validates
//   EVERY index against the axis bound BEFORE it writes anything
//   (PyArray_MapIterCheckIndices), so a single out-of-range index leaves `arr`
//   completely untouched (probed 2.4.2: a valid write that PRECEDES the bad
//   index in C-order does NOT land). We reproduce that with TWO passes over the
//   iteration space:
//     1. GetPutAlongAxisValidateKernel() — walks the index odometer, resolves the
//        advanced-index wrap and bounds-checks each index. Dtype-agnostic (indices
//        are always int64 here), so ONE cached instance serves every element width.
//        Returns totalSize on success, else the flat position of the first bad
//        index (with *outBadIdx holding its pre-wrap value) — the caller raises.
//     2. GetPutAlongAxisScatterKernel(elemBytes) — runs ONLY after validation
//        succeeds, so its hot loop resolves the wrap but carries NO bounds branch
//        at all (the indices are already known in range), which is what makes the
//        scatter fast. Keyed by elemBytes ∈ {1,2,4,8,16}; emits a typed store per
//        width via the shared EmitElemCopy (see TakeAlongAxis.cs).
//
//   Splitting validation into its own dtype-agnostic kernel — rather than a single
//   two-pass method per element width — both guarantees the atomicity contract and
//   keeps the scatter's inner loop branch-free on the bound.
//
//   NO SIMD SCATTER — same reasoning as TakeAlongAxis's gather (measured ~1.16x for
//   a hardware gather on this host before the bounds/fallback overhead; NumPy's own
//   advanced-index scatter is a scalar loop). The wins are structural: the
//   outer-odometer / inner-loop split (per-slice carry) and the branch-light resolve.
//
//   ANY layout works with no materialisation. `arr` is the DESTINATION and is
//   written through its own per-dimension strides (a broadcast/size-1 source dim
//   is a stride-0 entry, so several iteration positions collapse onto one arr
//   element — last write wins, matching NumPy); `arr` keeps its own (possibly
//   negative) axis stride. `indices` and `values` are READ through their own
//   per-dimension strides (a broadcast dim is a stride-0 entry), so C / F /
//   strided / reversed / sliced / broadcast inputs all work directly.
//
// KERNELS (DynamicMethod-emitted):
//
//   long PutAlongAxisValidate(               // dtype-agnostic, cached once
//       long* idxBase,          // int64 index buffer (idx offset already applied)
//       long* idxStrides,       // per iter-dim idx stride in ELEMENTS (0 at broadcast dims)
//       long  axisLen,          // M = arr.shape[axis]
//       long* shape,            // iteration dims (the broadcast result shape)
//       long  ndim,             // iteration ndim (>= 1)
//       long  totalSize,        // iteration element count (> 0)
//       long* outBadIdx)        // set to the offending ORIGINAL index on OOB
//       -> long: totalSize if every index is in range, else the flat position of
//                the first out-of-bounds element (*outBadIdx = its pre-wrap value).
//
//   void PutAlongAxisScatter(                // keyed per elemBytes
//       byte* arrBase,          // arr.Address + arr.offset*elem  (DESTINATION)
//       long* arrStrides,       // per iter-dim arr stride in BYTES (0 at axis & at broadcast dims)
//       long  axisStrideBytes,  // arr stride along `axis` in BYTES (used with the resolved index)
//       long  axisLen,          // M = arr.shape[axis]  (for the negative-index wrap)
//       long* idxBase,          // int64 index buffer (idx offset already applied)
//       long* idxStrides,       // per iter-dim idx stride in ELEMENTS (0 at broadcast dims)
//       byte* valBase,          // values buffer (already arr's dtype; offset applied)
//       long* valStrides,       // per iter-dim val stride in BYTES (0 at broadcast/leading dims)
//       long* shape,            // iteration dims
//       long  ndim,             // iteration ndim (>= 1)
//       long  totalSize);       // iteration element count (> 0)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted index-validation pass for <c>np.put_along_axis</c>. Walks the iteration
    /// odometer, applies the advanced-index negative wrap, and bounds-checks each int64
    /// index against the axis length — WITHOUT writing anything, so a caller can guarantee
    /// NumPy's all-or-nothing assignment (a single out-of-range index leaves the destination
    /// untouched). Dtype-agnostic (indices are always int64), hence a single cached instance.
    /// </summary>
    /// <returns>
    /// <c>totalSize</c> when every index is in range; otherwise the flat position of the first
    /// out-of-bounds element, with <c>*outBadIdx</c> set to that index's original (pre-wrap)
    /// value for the caller's diagnostic.
    /// </returns>
    public unsafe delegate long PutAlongAxisValidateKernel(
        long* idxBase, long* idxStrides, long axisLen,
        long* shape, long ndim, long totalSize, long* outBadIdx);

    /// <summary>
    /// IL-emitted per-element scatter kernel for <c>np.put_along_axis</c>. Walks the iteration
    /// odometer, reads one value through the (broadcast) <c>values</c> strides and writes it into
    /// <c>arr</c> at the resolved axis position. Runs ONLY after
    /// <see cref="PutAlongAxisValidateKernel"/> has passed, so it applies the negative-index wrap
    /// but performs NO bounds check in the hot loop.
    /// </summary>
    public unsafe delegate void PutAlongAxisScatterKernel(
        byte* arrBase, long* arrStrides, long axisStrideBytes, long axisLen,
        long* idxBase, long* idxStrides, byte* valBase, long* valStrides,
        long* shape, long ndim, long totalSize);

    public static partial class DirectILKernelGenerator
    {
        // The validation kernel is dtype-agnostic (indices are int64 regardless of arr's dtype),
        // so ONE instance serves every call. Keyed by a constant 0 in a tiny concurrent map purely
        // to reuse the thread-safe GetOrAdd lazy-init idiom used throughout this file.
        private static readonly ConcurrentDictionary<int, PutAlongAxisValidateKernel> _putAlongAxisValidateKernel
            = new ConcurrentDictionary<int, PutAlongAxisValidateKernel>();

        private static readonly ConcurrentDictionary<int, PutAlongAxisScatterKernel> _putAlongAxisScatterKernels
            = new ConcurrentDictionary<int, PutAlongAxisScatterKernel>();

        /// <summary>
        /// The IL index-validation kernel for <c>np.put_along_axis</c> (dtype-agnostic, generated
        /// once). Returns <c>null</c> only when <see cref="Enabled"/> is false or IL generation fails.
        /// </summary>
        public static PutAlongAxisValidateKernel GetPutAlongAxisValidateKernel()
        {
            if (!Enabled)
                return null;

            if (_putAlongAxisValidateKernel.TryGetValue(0, out var cached))
                return cached;

            try
            {
                return _putAlongAxisValidateKernel.GetOrAdd(0, _ => GeneratePutAlongAxisValidateIL());
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPutAlongAxisValidateKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// The IL scatter kernel for <c>np.put_along_axis</c>, cached per element width
        /// (<paramref name="elemBytes"/> ∈ {1,2,4,8,16}). Returns <c>null</c> only when
        /// <see cref="Enabled"/> is false or IL generation fails.
        /// </summary>
        public static PutAlongAxisScatterKernel GetPutAlongAxisScatterKernel(int elemBytes)
        {
            if (!Enabled)
                return null;

            if (_putAlongAxisScatterKernels.TryGetValue(elemBytes, out var cached))
                return cached;

            try
            {
                return _putAlongAxisScatterKernels.GetOrAdd(elemBytes, GeneratePutAlongAxisScatterIL);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPutAlongAxisScatterKernel({elemBytes}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the dtype-agnostic index-validation pass. Structurally the TakeAlongAxis odometer
        /// with only the index read + resolve + bounds check (no source gather, no destination store):
        /// an OUTER odometer over dimensions <c>[0, ndim-1)</c> wrapped around a tight INNER loop over
        /// the innermost dimension. Pseudocode:
        /// <code>
        /// long Validate(...) {
        ///     long* coord = stackalloc long[ndim];      // outer dims, zeroed
        ///     long lastN = shape[ndim-1], iLast = idxStrides[ndim-1];
        ///     long flat = 0, idxOuter = 0;
        ///     while (flat &lt; totalSize) {
        ///         long idxOff = idxOuter;
        ///         for (long t = 0; t &lt; lastN; t++) {
        ///             long idx = idxBase[idxOff];
        ///             long r = idx;
        ///             if (r &lt; 0) r += axisLen;                     // advanced-index single wrap
        ///             if ((ulong)r &gt;= (ulong)axisLen) { *outBadIdx = idx; return flat; }
        ///             flat++; idxOff += iLast;
        ///         }
        ///         for (long d = ndim - 2; d &gt;= 0; d--) {           // outer odometer
        ///             coord[d]++; idxOuter += idxStrides[d];
        ///             if (coord[d] &lt; shape[d]) break;
        ///             coord[d] = 0; idxOuter -= idxStrides[d] * shape[d];
        ///         }
        ///     }
        ///     return totalSize;
        /// }
        /// </code>
        /// </summary>
        private static PutAlongAxisValidateKernel GeneratePutAlongAxisValidateIL()
        {
            var dm = new DynamicMethod(
                name: "IL_PutAlongAxisValidate",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(long*),  // 0 idxBase
                    typeof(long*),  // 1 idxStrides
                    typeof(long),   // 2 axisLen
                    typeof(long*),  // 3 shape
                    typeof(long),   // 4 ndim
                    typeof(long),   // 5 totalSize
                    typeof(long*),  // 6 outBadIdx
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoord = il.DeclareLocal(typeof(long*));
            var locFlat = il.DeclareLocal(typeof(long));
            var locIdxOuter = il.DeclareLocal(typeof(long));
            var locIdxOff = il.DeclareLocal(typeof(long));
            var locT = il.DeclareLocal(typeof(long));
            var locLastN = il.DeclareLocal(typeof(long));
            var locILast = il.DeclareLocal(typeof(long));
            var locIdxVal = il.DeclareLocal(typeof(long));
            var locResolved = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));

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
            il.Emit(OpCodes.Ldarg, 4);            // ndim
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
            il.Emit(OpCodes.Ldarg, 4);
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

            // lastN = shape[ndim-1]; iLast = idxStrides[ndim-1];
            il.Emit(OpCodes.Ldarg, 4); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locD);
            EmitElemLoad(il, 3, locD); il.Emit(OpCodes.Stloc, locLastN);
            EmitElemLoad(il, 1, locD); il.Emit(OpCodes.Stloc, locILast);

            // flat = idxOuter = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locIdxOuter);

            // ---- outer loop ----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldarg, 5);            // totalSize
            il.Emit(OpCodes.Bge, lblDone);
            // idxOff = idxOuter; t = 0
            il.Emit(OpCodes.Ldloc, locIdxOuter); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locT);

            // ---- inner loop over the innermost dimension ----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locT);
            il.Emit(OpCodes.Ldloc, locLastN);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // idxVal = idxBase[idxOff]
            il.Emit(OpCodes.Ldarg, 0);            // idxBase
            il.Emit(OpCodes.Ldloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locIdxVal);

            // Advanced-index resolve: single conditional wrap, then ONE unsigned bounds compare.
            //   if (resolved < 0) resolved += axisLen;
            //   if ((ulong)resolved >= (ulong)axisLen) goto Fail;   // catches BOTH still-<0 and >=axisLen
            il.Emit(OpCodes.Ldloc, locIdxVal);
            il.Emit(OpCodes.Stloc, locResolved);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblBounds);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 2);            // axisLen
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locResolved);
            il.MarkLabel(lblBounds);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 2);
            il.Emit(OpCodes.Bge_Un, lblFail);     // (ulong)resolved >= (ulong)axisLen

            // flat++; idxOff += iLast; t++
            il.Emit(OpCodes.Ldloc, locFlat); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldloc, locIdxOff); il.Emit(OpCodes.Ldloc, locILast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldloc, locT); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locT);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ---- outer odometer advance: for (d=ndim-2; d>=0; d--) ----
            il.Emit(OpCodes.Ldarg, 4);            // ndim
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblOuterHead);   // ndim == 1 -> no outer dims

            il.MarkLabel(lblCarryHead);
            // coord[d]++
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);
            // idxOuter += idxStrides[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 1, locD);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // if (coord[d] < shape[d]) goto OuterHead;
            EmitElemLoad(il, -1, locD, locCoord); // coord[d]
            EmitElemLoad(il, 3, locD);            // shape[d]
            il.Emit(OpCodes.Blt, lblOuterHead);
            // carry: coord[d] = 0
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            // idxOuter -= idxStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 1, locD);
            EmitElemLoad(il, 3, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            // if (d >= 0) goto CarryHead; else exit via flat>=totalSize
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblCarryHead);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ---- fail: *outBadIdx = idxVal; return flat ----
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldarg, 6);            // outBadIdx
            il.Emit(OpCodes.Ldloc, locIdxVal);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ret);

            // ---- done: return totalSize ----
            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ret);

            return (PutAlongAxisValidateKernel)dm.CreateDelegate(typeof(PutAlongAxisValidateKernel));
        }

        /// <summary>
        /// Emits the scatter kernel for a fixed element width. Same OUTER-odometer / INNER-loop shape
        /// as the TakeAlongAxis gather but mirrored: it READS one value through the (broadcast)
        /// <c>values</c> strides and WRITES it into <c>arr</c> at the resolved axis position. Runs only
        /// after validation, so it applies the negative-index wrap but carries NO bounds check.
        /// Pseudocode:
        /// <code>
        /// void Scatter(...) {
        ///     long* coord = stackalloc long[ndim];      // outer dims, zeroed
        ///     long lastN = shape[ndim-1], aLast = arrStrides[ndim-1],
        ///          iLast = idxStrides[ndim-1], vLast = valStrides[ndim-1];
        ///     long flat = 0, arrOuter = 0, idxOuter = 0, valOuter = 0;
        ///     while (flat &lt; totalSize) {
        ///         long arrOff = arrOuter, idxOff = idxOuter, valOff = valOuter;
        ///         for (long t = 0; t &lt; lastN; t++) {
        ///             long r = idxBase[idxOff];
        ///             if (r &lt; 0) r += axisLen;                      // resolve (already validated)
        ///             *(T*)(arrBase + arrOff + r*axisStrideBytes) = *(T*)(valBase + valOff);
        ///             flat++; arrOff += aLast; idxOff += iLast; valOff += vLast;
        ///         }
        ///         for (long d = ndim - 2; d &gt;= 0; d--) {            // outer odometer
        ///             coord[d]++; arrOuter += arrStrides[d]; idxOuter += idxStrides[d]; valOuter += valStrides[d];
        ///             if (coord[d] &lt; shape[d]) break;
        ///             coord[d] = 0;
        ///             arrOuter -= arrStrides[d]*shape[d]; idxOuter -= idxStrides[d]*shape[d]; valOuter -= valStrides[d]*shape[d];
        ///         }
        ///     }
        /// }
        /// </code>
        /// </summary>
        private static PutAlongAxisScatterKernel GeneratePutAlongAxisScatterIL(int elemBytes)
        {
            var dm = new DynamicMethod(
                name: $"IL_PutAlongAxisScatter_{elemBytes}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 arrBase
                    typeof(long*),  // 1 arrStrides
                    typeof(long),   // 2 axisStrideBytes
                    typeof(long),   // 3 axisLen
                    typeof(long*),  // 4 idxBase
                    typeof(long*),  // 5 idxStrides
                    typeof(byte*),  // 6 valBase
                    typeof(long*),  // 7 valStrides
                    typeof(long*),  // 8 shape
                    typeof(long),   // 9 ndim
                    typeof(long),   // 10 totalSize
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoord = il.DeclareLocal(typeof(long*));
            var locFlat = il.DeclareLocal(typeof(long));
            var locArrOuter = il.DeclareLocal(typeof(long));
            var locIdxOuter = il.DeclareLocal(typeof(long));
            var locValOuter = il.DeclareLocal(typeof(long));
            var locArrOff = il.DeclareLocal(typeof(long));
            var locIdxOff = il.DeclareLocal(typeof(long));
            var locValOff = il.DeclareLocal(typeof(long));
            var locT = il.DeclareLocal(typeof(long));
            var locLastN = il.DeclareLocal(typeof(long));
            var locALast = il.DeclareLocal(typeof(long));
            var locILast = il.DeclareLocal(typeof(long));
            var locVLast = il.DeclareLocal(typeof(long));
            var locResolved = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));

            var lblZeroHead = il.DefineLabel();
            var lblZeroEnd = il.DefineLabel();
            var lblOuterHead = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();
            var lblResolved = il.DefineLabel();
            var lblCarryHead = il.DefineLabel();
            var lblDone = il.DefineLabel();

            // coord = stackalloc long[ndim]
            il.Emit(OpCodes.Ldarg, 9);            // ndim
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
            il.Emit(OpCodes.Ldarg, 9);
            il.Emit(OpCodes.Bge, lblZeroEnd);
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblZeroHead);
            il.MarkLabel(lblZeroEnd);

            // lastN = shape[ndim-1]; aLast = arrStrides[ndim-1]; iLast = idxStrides[ndim-1]; vLast = valStrides[ndim-1];
            il.Emit(OpCodes.Ldarg, 9); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locD);
            EmitElemLoad(il, 8, locD); il.Emit(OpCodes.Stloc, locLastN);
            EmitElemLoad(il, 1, locD); il.Emit(OpCodes.Stloc, locALast);
            EmitElemLoad(il, 5, locD); il.Emit(OpCodes.Stloc, locILast);
            EmitElemLoad(il, 7, locD); il.Emit(OpCodes.Stloc, locVLast);

            // flat = arrOuter = idxOuter = valOuter = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locArrOuter);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locIdxOuter);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locValOuter);

            // ---- outer loop ----
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldarg, 10);           // totalSize
            il.Emit(OpCodes.Bge, lblDone);
            // arrOff = arrOuter; idxOff = idxOuter; valOff = valOuter; t = 0
            il.Emit(OpCodes.Ldloc, locArrOuter); il.Emit(OpCodes.Stloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locIdxOuter); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldloc, locValOuter); il.Emit(OpCodes.Stloc, locValOff);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locT);

            // ---- inner loop over the innermost dimension ----
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locT);
            il.Emit(OpCodes.Ldloc, locLastN);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // resolved = idxBase[idxOff]; if (resolved < 0) resolved += axisLen;  (already validated in range)
            il.Emit(OpCodes.Ldarg, 4);            // idxBase
            il.Emit(OpCodes.Ldloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locResolved);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblResolved);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 3);            // axisLen
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locResolved);
            il.MarkLabel(lblResolved);

            // dst = arrBase + arrOff + resolved * axisStrideBytes
            il.Emit(OpCodes.Ldarg, 0);            // arrBase
            il.Emit(OpCodes.Ldloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locResolved);
            il.Emit(OpCodes.Ldarg, 2);            // axisStrideBytes
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            // src = valBase + valOff
            il.Emit(OpCodes.Ldarg, 6);            // valBase
            il.Emit(OpCodes.Ldloc, locValOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // *(T*)dst = *(T*)src  (typed copy of elemBytes)
            EmitElemCopy(il, locDst, locSrc, elemBytes);

            // flat++; arrOff += aLast; idxOff += iLast; valOff += vLast; t++
            il.Emit(OpCodes.Ldloc, locFlat); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Ldloc, locArrOff); il.Emit(OpCodes.Ldloc, locALast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locArrOff);
            il.Emit(OpCodes.Ldloc, locIdxOff); il.Emit(OpCodes.Ldloc, locILast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldloc, locValOff); il.Emit(OpCodes.Ldloc, locVLast); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locValOff);
            il.Emit(OpCodes.Ldloc, locT); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locT);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // ---- outer odometer advance: for (d=ndim-2; d>=0; d--) ----
            il.Emit(OpCodes.Ldarg, 9);            // ndim
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblOuterHead);   // ndim == 1 -> no outer dims

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
            EmitElemLoad(il, 1, locD);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locArrOuter);
            // idxOuter += idxStrides[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 5, locD);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // valOuter += valStrides[d]
            il.Emit(OpCodes.Ldloc, locValOuter);
            EmitElemLoad(il, 7, locD);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locValOuter);
            // if (coord[d] < shape[d]) goto OuterHead;
            EmitElemLoad(il, -1, locD, locCoord); // coord[d]
            EmitElemLoad(il, 8, locD);            // shape[d]
            il.Emit(OpCodes.Blt, lblOuterHead);
            // carry: coord[d] = 0
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            // arrOuter -= arrStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locArrOuter);
            EmitElemLoad(il, 1, locD);
            EmitElemLoad(il, 8, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locArrOuter);
            // idxOuter -= idxStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locIdxOuter);
            EmitElemLoad(il, 5, locD);
            EmitElemLoad(il, 8, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locIdxOuter);
            // valOuter -= valStrides[d] * shape[d]
            il.Emit(OpCodes.Ldloc, locValOuter);
            EmitElemLoad(il, 7, locD);
            EmitElemLoad(il, 8, locD);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locValOuter);
            // d--
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            // if (d >= 0) goto CarryHead; else exit via flat>=totalSize
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblCarryHead);
            il.Emit(OpCodes.Br, lblOuterHead);

            // ---- done ----
            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ret);

            return (PutAlongAxisScatterKernel)dm.CreateDelegate(typeof(PutAlongAxisScatterKernel));
        }
    }
}
