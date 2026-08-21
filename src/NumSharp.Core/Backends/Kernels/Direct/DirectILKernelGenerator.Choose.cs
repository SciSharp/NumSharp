using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.Choose.cs — IL selection kernels for np.choose
// =============================================================================
//
// RESPONSIBILITY:
//   np.choose(a, choices) picks, for every position I of the (broadcast) result,
//   ONE element out of the n `choices` arrays: result[I] = choices[a[I]][I]. NumPy
//   implements it as a multi-iterator that broadcasts all n choices AND the index
//   array `a` to a common shape and memcpy's one element per position
//   (item_selection.c: PyArray_Choose). We implement the same semantics as a single
//   whole-array kernel (the DirectILKernelGenerator contract — the kernel walks the
//   dimensions/strides itself), keyed by element width so it is dtype-agnostic.
//
//   The number of choices `n` is dynamic, so both kernels receive the choice base
//   pointers as a `byte**` and select choiceBases[mi] at run time (mi = the resolved
//   index). There is deliberately NO per-dtype specialisation — the byte-width key
//   (1/2/4/8/16) plus EmitElemCopy (shared with TakeAlongAxis) covers all 15 dtypes.
//
//   TWO PATHS, the SimdFull/General split every DirectILKernelGenerator op makes:
//     * FLAT     — every operand (all choices + index) is C-contiguous at the result
//                  shape (no broadcast, no stride tricks). Element `flat` lives at
//                  `base + flat*elem` in every choice and at `idx[flat]` in the index,
//                  so the loop is a tight flat gather with no coordinate bookkeeping.
//     * STRIDED  — anything else (broadcast/scalar/strided/transposed/negative-stride/
//                  sliced). An odometer walks the C-contiguous result; the index offset
//                  and the SELECTED choice's element offset are each a dot product of
//                  the current coordinate with that operand's per-dimension byte strides
//                  (a broadcast dim is a 0 stride, a negative-stride view a signed one),
//                  so any layout is read in place with no materialisation. Only the
//                  selected choice's offset is computed per element (not all n), so the
//                  per-element cost is O(ndim), independent of n.
//
//   NO SIMD GATHER — the selection index varies per element and a hardware gather was
//   measured at only ~1.16x for the contiguous 4/8-byte case in TakeAlongAxis (same
//   host, AVX2), before the bounds-validation a raise-mode gather still needs; NumPy's
//   own choose loop is likewise scalar. The result buffer is freshly allocated
//   C-contiguous, so the destination is always written linearly.
//
//   MODE is baked into the kernel (cache key = width × mode), so there is no per-element
//   mode branch:
//     * clip  — index < 0 -> 0, index >= n -> n-1 (never fails).
//     * wrap  — index mod n with sign correction (never fails).
//     * raise — index must be in [0, n-1]; the kernel returns the failing flat position
//               (< totalSize) so the caller raises ValueError("invalid entry in choice
//               array"). Because the result is a fresh temp, a raise mid-loop never
//               leaves a partially written `out` (NumPy's copy_existing_out guarantee).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Flat np.choose kernel: every operand is C-contiguous at the result shape, so element
    ///     <c>flat</c> is at <c>choiceBases[mi] + flat*elem</c> and <c>((long*)idxBase)[flat]</c>.
    /// </summary>
    /// <returns><c>totalSize</c> on success, or the flat position of the first out-of-bounds index
    /// (raise mode only; clip/wrap kernels always return <c>totalSize</c>).</returns>
    public unsafe delegate long ChooseFlatKernel(
        byte** choiceBases, byte* idxBase, byte* dstBase, long totalSize, long nChoices);

    /// <summary>
    ///     Strided np.choose kernel: an odometer over the C-contiguous result; the index and the
    ///     selected choice are read through their own per-dimension BYTE strides (0 = broadcast).
    /// </summary>
    /// <returns><c>totalSize</c> on success, or the flat position of the first out-of-bounds index
    /// (raise mode only).</returns>
    public unsafe delegate long ChooseStridedKernel(
        byte** choiceBases, long* choiceStrides, byte* idxBase, long* idxStrides,
        byte* dstBase, long* shape, long ndim, long totalSize, long nChoices);

    public static partial class DirectILKernelGenerator
    {
        // Mode constants (kernel-local; the np-layer maps the "raise"/"wrap"/"clip" string here).
        internal const int ChooseModeRaise = 0;
        internal const int ChooseModeWrap = 1;
        internal const int ChooseModeClip = 2;

        // Cache key = elemBytes * 16 + mode (elemBytes ∈ {1,2,4,8,16}, mode ∈ {0,1,2}) — unique.
        private static readonly ConcurrentDictionary<int, ChooseFlatKernel> _chooseFlatKernels
            = new ConcurrentDictionary<int, ChooseFlatKernel>();
        private static readonly ConcurrentDictionary<int, ChooseStridedKernel> _chooseStridedKernels
            = new ConcurrentDictionary<int, ChooseStridedKernel>();

        /// <summary>
        ///     IL-emitted flat choose kernel, cached per (element width, mode). Returns <c>null</c>
        ///     only when <see cref="Enabled"/> is false.
        /// </summary>
        public static ChooseFlatKernel GetChooseFlatKernel(int elemBytes, int mode)
        {
            if (!Enabled)
                return null;

            int key = elemBytes * 16 + mode;
            if (_chooseFlatKernels.TryGetValue(key, out var cached))
                return cached;

            try
            {
                return _chooseFlatKernels.GetOrAdd(key, _ => GenerateChooseFlatKernelIL(elemBytes, mode));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetChooseFlatKernel({elemBytes},{mode}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     IL-emitted strided choose kernel, cached per (element width, mode). Returns <c>null</c>
        ///     only when <see cref="Enabled"/> is false.
        /// </summary>
        public static ChooseStridedKernel GetChooseStridedKernel(int elemBytes, int mode)
        {
            if (!Enabled)
                return null;

            int key = elemBytes * 16 + mode;
            if (_chooseStridedKernels.TryGetValue(key, out var cached))
                return cached;

            try
            {
                return _chooseStridedKernels.GetOrAdd(key, _ => GenerateChooseStridedKernelIL(elemBytes, mode));
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetChooseStridedKernel({elemBytes},{mode}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Emits the flat gather kernel for a fixed element width and mode. Pseudocode:
        /// <code>
        /// long ChooseFlat(byte** choiceBases, byte* idxBase, byte* dstBase, long totalSize, long n) {
        ///     long* idx = (long*)idxBase;
        ///     for (long flat = 0; flat &lt; totalSize; flat++) {
        ///         long v = idx[flat];
        ///         long mi = resolve(v, n);                 // raise: if oob return flat;
        ///         byte* cb = choiceBases[mi];
        ///         *(T*)(dstBase + flat*elem) = *(T*)(cb + flat*elem);
        ///     }
        ///     return totalSize;
        /// }
        /// </code>
        /// </summary>
        private static ChooseFlatKernel GenerateChooseFlatKernelIL(int elemBytes, int mode)
        {
            var dm = new DynamicMethod(
                name: $"IL_ChooseFlat_{elemBytes}_{mode}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte**), // 0 choiceBases
                    typeof(byte*),  // 1 idxBase (int64)
                    typeof(byte*),  // 2 dstBase
                    typeof(long),   // 3 totalSize
                    typeof(long),   // 4 nChoices
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locFlat = il.DeclareLocal(typeof(long));
            var locV = il.DeclareLocal(typeof(long));
            var locMi = il.DeclareLocal(typeof(long));
            var locN = il.DeclareLocal(typeof(long));
            var locCb = il.DeclareLocal(typeof(byte*));
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));

            var lblHead = il.DefineLabel();
            var lblDone = il.DefineLabel();
            var lblFail = il.DefineLabel();

            // n = nChoices
            il.Emit(OpCodes.Ldarg, 4); il.Emit(OpCodes.Stloc, locN);
            // flat = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locFlat);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldarg, 3);            // totalSize
            il.Emit(OpCodes.Bge, lblDone);

            // v = ((long*)idxBase)[flat]  ==  *(long*)(idxBase + flat*8)
            il.Emit(OpCodes.Ldarg, 1);            // idxBase
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locV);

            // mi = resolve(v)
            EmitResolveMode(il, mode, locV, locN, locMi, lblFail);

            // cb = choiceBases[mi] = *(byte**)(choiceBases + mi*8)
            il.Emit(OpCodes.Ldarg, 0);            // choiceBases
            il.Emit(OpCodes.Ldloc, locMi);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locCb);

            // src = cb + flat*elem
            il.Emit(OpCodes.Ldloc, locCb);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, (long)elemBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // dst = dstBase + flat*elem
            il.Emit(OpCodes.Ldarg, 2);            // dstBase
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, (long)elemBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            EmitElemCopy(il, locDst, locSrc, elemBytes);

            // flat++
            il.Emit(OpCodes.Ldloc, locFlat); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locFlat);
            il.Emit(OpCodes.Br, lblHead);

            // fail (raise only): return flat
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ret);

            // done: return totalSize
            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg, 3);
            il.Emit(OpCodes.Ret);

            return (ChooseFlatKernel)dm.CreateDelegate(typeof(ChooseFlatKernel));
        }

        /// <summary>
        /// Emits the strided gather kernel for a fixed element width and mode. An odometer walks the
        /// C-contiguous result; each operand is read through its own per-dimension BYTE strides.
        /// Pseudocode:
        /// <code>
        /// long ChooseStrided(byte** choiceBases, long* choiceStrides, byte* idxBase, long* idxStrides,
        ///                    byte* dstBase, long* shape, long ndim, long totalSize, long n) {
        ///     long* coord = stackalloc long[ndim];          // zeroed
        ///     for (long flat = 0; flat &lt; totalSize; flat++) {
        ///         long idxOff = 0;                           // Σ coord[d]*idxStrides[d]
        ///         for (long d = 0; d &lt; ndim; d++) idxOff += coord[d]*idxStrides[d];
        ///         long v = *(long*)(idxBase + idxOff);
        ///         long mi = resolve(v, n);                   // raise: if oob return flat;
        ///         long baseK = mi*ndim, srcOff = 0;          // Σ coord[d]*choiceStrides[mi*ndim+d]
        ///         for (long d = 0; d &lt; ndim; d++) srcOff += coord[d]*choiceStrides[baseK+d];
        ///         *(T*)(dstBase + flat*elem) = *(T*)(choiceBases[mi] + srcOff);
        ///         for (long d = ndim-1; d &gt;= 0; d--) {      // odometer
        ///             coord[d]++; if (coord[d] &lt; shape[d]) break; coord[d] = 0;
        ///         }
        ///     }
        ///     return totalSize;
        /// }
        /// </code>
        /// </summary>
        private static ChooseStridedKernel GenerateChooseStridedKernelIL(int elemBytes, int mode)
        {
            var dm = new DynamicMethod(
                name: $"IL_ChooseStrided_{elemBytes}_{mode}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte**), // 0 choiceBases
                    typeof(long*),  // 1 choiceStrides (n*ndim, BYTES)
                    typeof(byte*),  // 2 idxBase (int64 logical start)
                    typeof(long*),  // 3 idxStrides (ndim, BYTES)
                    typeof(byte*),  // 4 dstBase
                    typeof(long*),  // 5 shape (ndim)
                    typeof(long),   // 6 ndim
                    typeof(long),   // 7 totalSize
                    typeof(long),   // 8 nChoices
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoord = il.DeclareLocal(typeof(long*));
            var locFlat = il.DeclareLocal(typeof(long));
            var locIdxOff = il.DeclareLocal(typeof(long));
            var locV = il.DeclareLocal(typeof(long));
            var locMi = il.DeclareLocal(typeof(long));
            var locN = il.DeclareLocal(typeof(long));
            var locBaseK = il.DeclareLocal(typeof(long));
            var locSrcOff = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locCb = il.DeclareLocal(typeof(byte*));
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));

            var lblZeroHead = il.DefineLabel();
            var lblZeroEnd = il.DefineLabel();
            var lblHead = il.DefineLabel();
            var lblDone = il.DefineLabel();
            var lblFail = il.DefineLabel();
            var lblIdxDotHead = il.DefineLabel();
            var lblIdxDotEnd = il.DefineLabel();
            var lblSrcDotHead = il.DefineLabel();
            var lblSrcDotEnd = il.DefineLabel();
            var lblCarryHead = il.DefineLabel();
            var lblCarryEnd = il.DefineLabel();

            // n = nChoices
            il.Emit(OpCodes.Ldarg, 8); il.Emit(OpCodes.Stloc, locN);

            // coord = stackalloc long[ndim]
            il.Emit(OpCodes.Ldarg, 6);            // ndim
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoord);

            // for (d=0; d<ndim; d++) coord[d]=0;
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locD);
            il.MarkLabel(lblZeroHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Bge, lblZeroEnd);
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldloc, locD); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblZeroHead);
            il.MarkLabel(lblZeroEnd);

            // flat = 0
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locFlat);

            // ---- main loop ----
            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldarg, 7);            // totalSize
            il.Emit(OpCodes.Bge, lblDone);

            // idxOff = Σ coord[d]*idxStrides[d]
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locD);
            il.MarkLabel(lblIdxDotHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Bge, lblIdxDotEnd);
            il.Emit(OpCodes.Ldloc, locIdxOff);
            EmitElemLoad(il, -1, locD, locCoord);  // coord[d]
            EmitElemLoad(il, 3, locD);             // idxStrides[d]
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdxOff);
            il.Emit(OpCodes.Ldloc, locD); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblIdxDotHead);
            il.MarkLabel(lblIdxDotEnd);

            // v = *(long*)(idxBase + idxOff)
            il.Emit(OpCodes.Ldarg, 2);            // idxBase
            il.Emit(OpCodes.Ldloc, locIdxOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locV);

            // mi = resolve(v)
            EmitResolveMode(il, mode, locV, locN, locMi, lblFail);

            // baseK = mi*ndim
            il.Emit(OpCodes.Ldloc, locMi);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locBaseK);

            // srcOff = Σ coord[d]*choiceStrides[baseK+d]
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locSrcOff);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locD);
            il.MarkLabel(lblSrcDotHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Bge, lblSrcDotEnd);
            il.Emit(OpCodes.Ldloc, locSrcOff);
            EmitElemLoad(il, -1, locD, locCoord);  // coord[d]
            // choiceStrides[baseK + d]
            il.Emit(OpCodes.Ldarg, 1);             // choiceStrides
            il.Emit(OpCodes.Ldloc, locBaseK);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrcOff);
            il.Emit(OpCodes.Ldloc, locD); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblSrcDotHead);
            il.MarkLabel(lblSrcDotEnd);

            // cb = choiceBases[mi]
            il.Emit(OpCodes.Ldarg, 0);
            il.Emit(OpCodes.Ldloc, locMi);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I);
            il.Emit(OpCodes.Stloc, locCb);

            // src = cb + srcOff
            il.Emit(OpCodes.Ldloc, locCb);
            il.Emit(OpCodes.Ldloc, locSrcOff);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // dst = dstBase + flat*elem
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ldc_I8, (long)elemBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            EmitElemCopy(il, locDst, locSrc, elemBytes);

            // flat++
            il.Emit(OpCodes.Ldloc, locFlat); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locFlat);

            // ---- odometer: for (d = ndim-1; d >= 0; d--) { coord[d]++; if (coord[d] < shape[d]) break; coord[d]=0; } ----
            il.Emit(OpCodes.Ldarg, 6); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locD);
            il.MarkLabel(lblCarryHead);
            // if (d < 0) goto carryEnd;
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblCarryEnd);
            // coord[d]++
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);
            // if (coord[d] < shape[d]) goto carryEnd;
            EmitElemLoad(il, -1, locD, locCoord);  // coord[d]
            EmitElemLoad(il, 5, locD);             // shape[d]
            il.Emit(OpCodes.Blt, lblCarryEnd);
            // coord[d] = 0
            EmitElemAddr(il, locCoord, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stind_I8);
            // d--
            il.Emit(OpCodes.Ldloc, locD); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Sub); il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblCarryHead);
            il.MarkLabel(lblCarryEnd);

            il.Emit(OpCodes.Br, lblHead);

            // fail (raise only): return flat
            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locFlat);
            il.Emit(OpCodes.Ret);

            // done: return totalSize
            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ret);

            return (ChooseStridedKernel)dm.CreateDelegate(typeof(ChooseStridedKernel));
        }

        /// <summary>
        ///     Emit the index-resolution for the given clip mode: pushes the resolved choice index
        ///     into <paramref name="locMi"/> from the raw index value <paramref name="locV"/> and the
        ///     choice count <paramref name="locN"/>. RAISE branches to <paramref name="lblFail"/> on an
        ///     out-of-range index; WRAP/CLIP never fail.
        /// </summary>
        private static void EmitResolveMode(ILGenerator il, int mode, LocalBuilder locV, LocalBuilder locN, LocalBuilder locMi, Label lblFail)
        {
            switch (mode)
            {
                case ChooseModeRaise:
                {
                    // if ((ulong)v >= (ulong)n) goto fail;   // one unsigned compare catches v<0 and v>=n
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Bge_Un, lblFail);
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Stloc, locMi);
                    break;
                }
                case ChooseModeWrap:
                {
                    // mi = v % n;  if (mi < 0) mi += n;
                    var lblNoAdd = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Rem);          // signed remainder (sign follows dividend)
                    il.Emit(OpCodes.Stloc, locMi);
                    il.Emit(OpCodes.Ldloc, locMi);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Bge, lblNoAdd);
                    il.Emit(OpCodes.Ldloc, locMi);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, locMi);
                    il.MarkLabel(lblNoAdd);
                    break;
                }
                case ChooseModeClip:
                {
                    // mi = v < 0 ? 0 : (v >= n ? n-1 : v);
                    var lblNotNeg = il.DefineLabel();
                    var lblInRange = il.DefineLabel();
                    var lblEnd = il.DefineLabel();
                    // if (v >= 0) goto notNeg; else { mi = 0; goto end; }
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Bge, lblNotNeg);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Stloc, locMi);
                    il.Emit(OpCodes.Br, lblEnd);
                    // notNeg: if (v < n) goto inRange; else { mi = n-1; goto end; }
                    il.MarkLabel(lblNotNeg);
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Blt, lblInRange);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Ldc_I8, 1L);
                    il.Emit(OpCodes.Sub);
                    il.Emit(OpCodes.Stloc, locMi);
                    il.Emit(OpCodes.Br, lblEnd);
                    // inRange: mi = v
                    il.MarkLabel(lblInRange);
                    il.Emit(OpCodes.Ldloc, locV);
                    il.Emit(OpCodes.Stloc, locMi);
                    il.MarkLabel(lblEnd);
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(mode));
            }
        }
    }
}
