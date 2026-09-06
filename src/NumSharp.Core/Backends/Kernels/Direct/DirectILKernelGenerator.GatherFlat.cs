using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.GatherFlat.cs — the lean 1-D gather / scatter kernels
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// The general take/put kernels (DirectILKernelGenerator.Take.cs / .Put.cs) serve
// any (outer, axis, inner-slab) factorisation with a runtime slab size and a
// per-element mode dispatch. The hottest gather/scatter shape in the library is
// none of that: a 1-D C-contiguous source, one flat integer index array, one
// primitive-width element per index — `a[idx]`, `a[idx] = v`, `np.take(a, idx)`,
// `np.put(a, idx, v)`. NumPy serves exactly that shape with its `mapiter_trivial`
// loop (lowlevel_strided_loops.c.src): load index, check-and-adjust, one typed
// MOV — ~0.57 ns/element for a cache-resident float64 source at 100K.
//
// The general kernel pays, per element, the pair index (outer*m + j), three
// multiplies by RUNTIME slab sizes (which the JIT cannot turn into shifts), and
// two mode compares, and measured ~0.75 ns/element on the same shape. These flat
// kernels drop all of it: the element width is a compile-time constant (so the
// address is one shift), the destination / index / values pointers are running
// cursors, the mode is RAISE only (the only mode fancy indexing has; np.take/put
// with wrap/clip keep the general kernel), and negative normalisation plus the
// bounds test are one add and one UNSIGNED compare — NumPy's check_and_adjust_index.
//
// CONTRACTS
// ---------
//   TakeFlatKernel(byte* src, void* indices, long count, long maxItem, byte* dst)
//       dst[j] = src[adjust(indices[j])] for j in [0, count); returns count, or
//       the first failing j (index outside [-maxItem, maxItem)) — nothing past it
//       is written, and the caller reads indices[j] for the diagnostic.
//   PutFlatKernel(byte* dst, void* indices, long count, byte* values, long valuesCount, long maxItem)
//       dst[adjust(indices[j])] = values[j mod valuesCount]; same return contract
//       (a caller that must not write partially validates the indices FIRST —
//       FancyIndexScan — exactly as NumPy's mapiter_trivial_set does).
//
// Both are generated per (element width ∈ {1,2,4,8,16}, index width ∈ {4,8}
// bytes, take: prefetch on/off). The index pointer is `void*` because an int32
// index array is read in place — no widening copy — which is the common C# case
// (`np.array(new int[] {…})` is int32) and the reason `np.take(a, idx32)` used to
// cost 27 % more than `np.take(a, idx64)` at 100K.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Lean 1-D gather: <c>dst[j] = src[indices[j]]</c> over <paramref name="count"/> elements of
    /// one compile-time width, RAISE semantics (a negative index is normalised once, then
    /// bounds-checked). <paramref name="indices"/> addresses int32 or int64 values, per the
    /// kernel's generation. Returns <paramref name="count"/>, or the first failing position.
    /// </summary>
    public unsafe delegate long TakeFlatKernel(byte* src, void* indices, long count, long maxItem, byte* dst);

    /// <summary>
    /// Lean 1-D scatter: <c>dst[indices[j]] = values[j mod valuesCount]</c> (a wrapping values
    /// cursor: <c>valuesCount == 1</c> broadcasts a scalar, <c>== count</c> is a straight copy).
    /// RAISE semantics as <see cref="TakeFlatKernel"/>; returns <paramref name="count"/> or the
    /// first failing position — elements before it HAVE been written.
    /// </summary>
    public unsafe delegate long PutFlatKernel(byte* dst, void* indices, long count, byte* values, long valuesCount, long maxItem);

    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<int, TakeFlatKernel> _takeFlatKernels = new();
        private static readonly ConcurrentDictionary<int, PutFlatKernel> _putFlatKernels = new();

        /// <summary>
        /// The flat gather kernel for one element width (<paramref name="copyKind"/> ∈ {1,2,4,8,16};
        /// the <see cref="CopyKindFor"/> of the per-index slab) and index width. <c>null</c> when
        /// runtime IL generation is unavailable or the slab is not a single primitive width
        /// (copyKind 0 — the general <see cref="TakeKernel"/> serves those with <c>cpblk</c>).
        /// </summary>
        public static TakeFlatKernel GetTakeFlatKernel(int copyKind, bool idx32, bool prefetch)
        {
            if (!Enabled || copyKind == 0)
                return null;

            int key = (copyKind << 2) | (idx32 ? 2 : 0) | (prefetch ? 1 : 0);
            if (_takeFlatKernels.TryGetValue(key, out var cached))
                return cached;

            try
            {
                var k = GenerateTakeFlatKernelIL(copyKind, idx32, prefetch);
                return _takeFlatKernels.GetOrAdd(key, k);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTakeFlatKernel({copyKind},{idx32},{prefetch}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// The flat scatter kernel for one element width and index width; <c>null</c> under the same
        /// conditions as <see cref="GetTakeFlatKernel"/>.
        /// </summary>
        public static PutFlatKernel GetPutFlatKernel(int copyKind, bool idx32)
        {
            if (!Enabled || copyKind == 0)
                return null;

            int key = (copyKind << 1) | (idx32 ? 1 : 0);
            if (_putFlatKernels.TryGetValue(key, out var cached))
                return cached;

            try
            {
                var k = GeneratePutFlatKernelIL(copyKind, idx32);
                return _putFlatKernels.GetOrAdd(key, k);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetPutFlatKernel({copyKind},{idx32}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>log2 of a typed copy width (1/2/4/8/16 → 0/1/2/3/4) — the address shift.</summary>
        private static int ShiftForCopyKind(int copyKind) => copyKind switch
        {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            _ => throw new ArgumentOutOfRangeException(nameof(copyKind), copyKind, "flat kernels need a typed width"),
        };

        /// <summary>
        /// Emit <c>idx = *(int32|int64*)ptrLocal</c> (sign-extended to int64) into <paramref name="locIdx"/>.
        /// </summary>
        private static void EmitFlatIndexLoad(ILGenerator il, LocalBuilder locPtr, LocalBuilder locIdx, bool idx32)
        {
            il.Emit(OpCodes.Ldloc, locPtr);
            if (idx32)
            {
                il.Emit(OpCodes.Ldind_I4);
                il.Emit(OpCodes.Conv_I8);
            }
            else
            {
                il.Emit(OpCodes.Ldind_I8);
            }
            il.Emit(OpCodes.Stloc, locIdx);
        }

        /// <summary>
        /// NumPy's <c>check_and_adjust_index</c> on <paramref name="locIdx"/> against the int64 argument
        /// <paramref name="maxItemArg"/>: <c>if (idx &lt; 0) idx += maxItem; if ((ulong)idx &gt;= (ulong)maxItem) goto fail;</c>
        /// — the unsigned compare rejects a still-negative index and an over-range one in one test.
        /// </summary>
        private static void EmitFlatRaiseAdjust(ILGenerator il, LocalBuilder locIdx, int maxItemArg, Label lblFail)
        {
            var lblNonNeg = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Bge, lblNonNeg);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, maxItemArg);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locIdx);
            il.MarkLabel(lblNonNeg);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldarg, maxItemArg);
            il.Emit(OpCodes.Bge_Un, lblFail);
        }

        /// <summary>Emit <c>local += delta</c> on a pointer local.</summary>
        private static void EmitAdvancePtr(ILGenerator il, LocalBuilder loc, int delta)
        {
            il.Emit(OpCodes.Ldloc, loc);
            il.Emit(OpCodes.Ldc_I4, delta);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, loc);
        }

        /// <summary>
        /// Emits the flat gather. Pseudocode (W = copyKind, IW = index width, S = log2 W):
        /// <code>
        /// long TakeFlat(byte* src, void* ind, long count, long maxItem, byte* dst) {
        ///     byte* ip = ind, dp = dst;
        ///     for (long j = 0; j &lt; count; j++, ip += IW, dp += W) {
        ///         [prefetch: if (j + DIST &lt; count) prefetch(src + ((long)ip[DIST] &lt;&lt; S));]
        ///         long idx = *ip; if (idx &lt; 0) idx += maxItem;
        ///         if ((ulong)idx &gt;= (ulong)maxItem) return j;
        ///         *(W*)dp = *(W*)(src + (idx &lt;&lt; S));
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        private static TakeFlatKernel GenerateTakeFlatKernelIL(int copyKind, bool idx32, bool prefetch)
        {
            int shift = ShiftForCopyKind(copyKind);
            int idxWidth = idx32 ? 4 : 8;
            bool usePrefetch = prefetch && Sse.IsSupported && _ssePrefetch0 != null;

            var dm = new DynamicMethod(
                name: $"IL_TakeFlat_c{copyKind}_i{idxWidth}_{(usePrefetch ? "pf" : "np")}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(void*),  // 1 indices
                    typeof(long),   // 2 count
                    typeof(long),   // 3 maxItem
                    typeof(byte*),  // 4 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locJ = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locIp = il.DeclareLocal(typeof(byte*));
            var locDp = il.DeclareLocal(typeof(byte*));
            var locSp = il.DeclareLocal(typeof(byte*));

            var lblHead = il.DefineLabel();
            var lblDone = il.DefineLabel();
            var lblFail = il.DefineLabel();

            // ip = indices; dp = dst; j = 0
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, locIp);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Stloc, locDp);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblDone);

            if (usePrefetch)
            {
                // if (j + DIST < count) Prefetch0(src + ((long)ip[DIST] << S))  — the RAW future index
                // (no adjustment): a wild address only wastes the prefetch, it never faults.
                var lblSkipPf = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, locJ);
                il.Emit(OpCodes.Ldc_I8, PrefetchDistance);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Bge, lblSkipPf);

                il.Emit(OpCodes.Ldarg_0);                            // src
                il.Emit(OpCodes.Ldloc, locIp);
                il.Emit(OpCodes.Ldc_I4, (int)PrefetchDistance * idxWidth);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);                                // ip + DIST*IW
                if (idx32) { il.Emit(OpCodes.Ldind_I4); il.Emit(OpCodes.Conv_I8); }
                else il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I4, shift);
                il.Emit(OpCodes.Shl);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);                                // src + (idxFuture << S)
                il.Emit(OpCodes.Call, _ssePrefetch0);
                il.MarkLabel(lblSkipPf);
            }

            EmitFlatIndexLoad(il, locIp, locIdx, idx32);
            EmitFlatRaiseAdjust(il, locIdx, maxItemArg: 3, lblFail);

            // sp = src + (idx << S)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I4, shift);
            il.Emit(OpCodes.Shl);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSp);

            EmitElementCopy(il, copyKind, locDp, locSp, null);

            EmitAdvancePtr(il, locIp, idxWidth);
            EmitAdvancePtr(il, locDp, copyKind);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ret);

            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ret);

            return (TakeFlatKernel)dm.CreateDelegate(typeof(TakeFlatKernel));
        }

        /// <summary>
        /// Emits the flat scatter. Pseudocode (W = copyKind, IW = index width, S = log2 W):
        /// <code>
        /// long PutFlat(byte* dst, void* ind, long count, byte* values, long nv, long maxItem) {
        ///     byte* ip = ind, vp = values, vend = values + (nv &lt;&lt; S);
        ///     for (long j = 0; j &lt; count; j++, ip += IW) {
        ///         long idx = *ip; if (idx &lt; 0) idx += maxItem;
        ///         if ((ulong)idx &gt;= (ulong)maxItem) return j;
        ///         *(W*)(dst + (idx &lt;&lt; S)) = *(W*)vp;
        ///         vp += W; if (vp == vend) vp = values;      // wrapping values cursor
        ///     }
        ///     return count;
        /// }
        /// </code>
        /// </summary>
        private static PutFlatKernel GeneratePutFlatKernelIL(int copyKind, bool idx32)
        {
            int shift = ShiftForCopyKind(copyKind);
            int idxWidth = idx32 ? 4 : 8;

            var dm = new DynamicMethod(
                name: $"IL_PutFlat_c{copyKind}_i{idxWidth}",
                returnType: typeof(long),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 dst
                    typeof(void*),  // 1 indices
                    typeof(long),   // 2 count
                    typeof(byte*),  // 3 values
                    typeof(long),   // 4 valuesCount
                    typeof(long),   // 5 maxItem
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locJ = il.DeclareLocal(typeof(long));
            var locIdx = il.DeclareLocal(typeof(long));
            var locIp = il.DeclareLocal(typeof(byte*));
            var locVp = il.DeclareLocal(typeof(byte*));
            var locVend = il.DeclareLocal(typeof(byte*));
            var locDp = il.DeclareLocal(typeof(byte*));

            var lblHead = il.DefineLabel();
            var lblDone = il.DefineLabel();
            var lblFail = il.DefineLabel();

            // ip = indices; vp = values; vend = values + (nv << S); j = 0
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, locIp);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, locVp);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I4, shift);
            il.Emit(OpCodes.Shl);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locVend);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblDone);

            EmitFlatIndexLoad(il, locIp, locIdx, idx32);
            EmitFlatRaiseAdjust(il, locIdx, maxItemArg: 5, lblFail);

            // dp = dst + (idx << S)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locIdx);
            il.Emit(OpCodes.Ldc_I4, shift);
            il.Emit(OpCodes.Shl);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDp);

            EmitElementCopy(il, copyKind, locDp, locVp, null);

            // vp += W; if (vp == vend) vp = values;
            var lblNoWrap = il.DefineLabel();
            EmitAdvancePtr(il, locVp, copyKind);
            il.Emit(OpCodes.Ldloc, locVp);
            il.Emit(OpCodes.Ldloc, locVend);
            il.Emit(OpCodes.Bne_Un, lblNoWrap);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Stloc, locVp);
            il.MarkLabel(lblNoWrap);

            EmitAdvancePtr(il, locIp, idxWidth);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblDone);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ret);

            il.MarkLabel(lblFail);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ret);

            return (PutFlatKernel)dm.CreateDelegate(typeof(PutFlatKernel));
        }
    }
}
