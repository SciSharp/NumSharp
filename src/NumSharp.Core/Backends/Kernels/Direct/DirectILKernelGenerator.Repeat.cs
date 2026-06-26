using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Repeat — IL-generated kernels for np.repeat
// =============================================================================
//
// Single entry point for every np.repeat dispatch. Callers compute the geometry
// (n_outer, n, chunkBytes, counts) and hand raw pointers to a cached kernel
// generated for that chunk size. The kernel emits the canonical NumPy 3-loop
// (mirrors `npy_fastrepeat_impl` in numpy/_core/src/multiarray/item_selection.c):
//
//   for (i = 0; i < n_outer; i++)
//     for (j = 0; j < n; j++)
//       cnt = broadcast ? broadcastCount : perJCounts[j];
//       for (k = 0; k < cnt; k++)
//         memcpy(dst, src, chunkBytes); dst += chunkBytes;
//       src += chunkBytes;
//
// Two kernel families per chunk size to lift the broadcast/per-j branch out of
// the inner loop — measurable on chunk=elsize axis-innermost paths where the
// j-loop runs hundreds of thousands of times:
//   RepeatBroadcastKernel: scalar / size-1 repeats. Tight 3-loop, no count load.
//   RepeatPerJKernel:      per-j repeat counts. Loads perJCounts[j] each j.
//
// Inner-copy strategy by chunk size:
//   1, 2, 4, 8:  scalar pre-broadcast → Vector{N}.Create(val) hoisted into a
//                local once per j, then a three-stage k-loop:
//                  • SIMD body — Vector{N}.Store writes `lanes` copies per iter
//                    (lanes = VectorBytes / chunkBytes; N is the startup-baked
//                    VectorBits → V128/V256/V512).
//                  • Scalar tail — single-element store for the remainder.
//                For r ≥ lanes (typical bulk repeats) this dispatches one wide
//                store per `lanes` k-iterations instead of `lanes` scalar stores.
//   16:          Vector128<byte> preload + Vector128.Store in the k-loop.
//                If VectorBits ≥ 256 we additionally pack two copies of the
//                16-byte slab into a Vector256 via Create(v128, v128) and emit a
//                wider SIMD body that writes 2 copies per iter.
//   anything else: cpblk with the size baked in as a constant so the JIT
//                specializes the memcpy (.NET 7+ optimizes constant-size cpblk).
//
// One generated kernel per (chunkBytes, kind) tuple. For typical workloads
// (axis=last on 15 dtypes -> 5 sizes; a few common shape/axis combos -> a
// handful more) the cache stays tiny.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>Broadcast variant — every j uses the same <paramref name="count"/>.</summary>
        public unsafe delegate void RepeatBroadcastKernel(
            byte* src,
            byte* dst,
            long n_outer,
            long n,
            long count);

        /// <summary>Per-j variant — counts[j] varies; must have length <c>n</c>.</summary>
        public unsafe delegate void RepeatPerJKernel(
            byte* src,
            byte* dst,
            long n_outer,
            long n,
            long* counts);

        internal static readonly ConcurrentDictionary<int, RepeatBroadcastKernel> _repeatBroadcastCache = new();
        internal static readonly ConcurrentDictionary<int, RepeatPerJKernel> _repeatPerJCache = new();

        /// <summary>
        /// Returns the cached IL-emitted broadcast-repeat kernel for the given slab size.
        /// First call for a size triggers IL generation, later calls hit a dictionary lookup.
        /// </summary>
        public static RepeatBroadcastKernel GetRepeatBroadcastKernel(int chunkBytes)
        {
            if (chunkBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(chunkBytes), "chunkBytes must be positive");

            return _repeatBroadcastCache.GetOrAdd(chunkBytes, GenerateBroadcastKernel);
        }

        /// <summary>
        /// Returns the cached IL-emitted per-j repeat kernel for the given slab size.
        /// </summary>
        public static RepeatPerJKernel GetRepeatPerJKernel(int chunkBytes)
        {
            if (chunkBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(chunkBytes), "chunkBytes must be positive");

            return _repeatPerJCache.GetOrAdd(chunkBytes, GeneratePerJKernel);
        }

        private static RepeatBroadcastKernel GenerateBroadcastKernel(int chunkBytes)
        {
            var dm = new DynamicMethod(
                name: $"RepeatBroadcast_{chunkBytes}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0: src
                    typeof(byte*),  // 1: dst
                    typeof(long),   // 2: n_outer
                    typeof(long),   // 3: n
                    typeof(long),   // 4: count
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            EmitRepeatBody(
                dm.GetILGenerator(),
                chunkBytes,
                emitCountLoad: (il, locJ) =>
                {
                    il.Emit(OpCodes.Ldarg_S, (byte)4);
                });

            return dm.CreateDelegate<RepeatBroadcastKernel>();
        }

        private static RepeatPerJKernel GeneratePerJKernel(int chunkBytes)
        {
            var dm = new DynamicMethod(
                name: $"RepeatPerJ_{chunkBytes}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0: src
                    typeof(byte*),  // 1: dst
                    typeof(long),   // 2: n_outer
                    typeof(long),   // 3: n
                    typeof(long*),  // 4: counts
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            EmitRepeatBody(
                dm.GetILGenerator(),
                chunkBytes,
                emitCountLoad: (il, locJ) =>
                {
                    // counts[j] = *(long*)(counts + j*8)
                    il.Emit(OpCodes.Ldarg_S, (byte)4);
                    il.Emit(OpCodes.Ldloc, locJ);
                    il.Emit(OpCodes.Ldc_I8, 8L);
                    il.Emit(OpCodes.Mul);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldind_I8);
                });

            return dm.CreateDelegate<RepeatPerJKernel>();
        }

        // Element type used to broadcast a chunk-sized scalar into a wide vector.
        // Returns null for chunks that can't be a single primitive lane.
        private static Type GetBroadcastElemType(int chunkBytes) => chunkBytes switch
        {
            1 => typeof(byte),
            2 => typeof(ushort),
            4 => typeof(uint),
            8 => typeof(ulong),
            _ => null
        };

        // Emits the shared 3-loop body. `emitCountLoad` pushes a `long` onto the
        // stack — the per-j repeat count. Caller controls how that's computed
        // (constant arg for broadcast, indexed load for per-j).
        private static void EmitRepeatBody(
            ILGenerator il,
            int chunkBytes,
            Action<ILGenerator, LocalBuilder> emitCountLoad)
        {
            // Locals
            var locSrc = il.DeclareLocal(typeof(byte*));
            var locDst = il.DeclareLocal(typeof(byte*));
            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locCnt = il.DeclareLocal(typeof(long));

            // Preloaded value local — declared per chunk-size strategy.
            LocalBuilder locVal = chunkBytes switch
            {
                1 => il.DeclareLocal(typeof(byte)),
                2 => il.DeclareLocal(typeof(ushort)),
                4 => il.DeclareLocal(typeof(uint)),
                8 => il.DeclareLocal(typeof(ulong)),
                16 => il.DeclareLocal(typeof(Vector128<byte>)),
                _ => null
            };

            // Wide broadcast vector — covers chunks {1,2,4,8} when SIMD is available,
            // and chunks=16 when VectorBits >= 256 (pack two copies into V256/V512).
            Type wideVecElem = GetBroadcastElemType(chunkBytes);
            bool useWideBroadcast = wideVecElem != null && VectorBits >= 128;
            LocalBuilder locWideVec = useWideBroadcast
                ? il.DeclareLocal(VectorMethodCache.V(VectorBits, wideVecElem))
                : null;

            bool useChunk16Wide = chunkBytes == 16 && VectorBits >= 256;
            LocalBuilder locWide16 = useChunk16Wide
                ? il.DeclareLocal(VectorMethodCache.V(VectorBits, typeof(byte)))
                : null;

            // src/dst mutable copies
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, locSrc);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Stloc, locDst);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            var lblOuter = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblInner = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();
            var lblScalarTail = il.DefineLabel();
            var lblScalarTailEnd = il.DefineLabel();

            // ===== OUTER LOOP =====
            il.MarkLabel(lblOuter);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2); // n_outer
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // ===== INNER LOOP =====
            il.MarkLabel(lblInner);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_3); // n
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // Push count via callback, store into locCnt
            emitCountLoad(il, locJ);
            il.Emit(OpCodes.Stloc, locCnt);

            // Preload value into register for small chunks (saves a load per k iter)
            if (locVal != null)
                EmitPreload(il, locSrc, locVal, chunkBytes);

            // k = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locK);

            // ===== SIMD-broadcast stage =====
            // The Vector.Create + SIMD loop are gated by `cnt >= lanes` so workloads
            // with r < lanes (e.g. r=2 chunk=8 V256 where lanes=4) skip the setup
            // entirely and fall straight to the scalar tail — no regression vs. a
            // pure-scalar kernel.
            if (locWideVec != null)
            {
                int lanes = VectorBytes / chunkBytes;
                var lblSkipSimd = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locCnt);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Blt, lblSkipSimd);

                // locWideVec = Vector{N}.Create(val) — broadcasts the scalar into all lanes.
                il.Emit(OpCodes.Ldloc, locVal);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(VectorBits, wideVecElem), null);
                il.Emit(OpCodes.Stloc, locWideVec);

                EmitSimdBroadcastStage(
                    il, locK, locCnt, locDst, locWideVec,
                    vecElem: wideVecElem,
                    lanesPerIter: lanes,
                    bytesPerIter: VectorBytes);

                il.MarkLabel(lblSkipSimd);
            }
            else if (locWide16 != null)
            {
                int lanes = VectorBytes / 16;
                var lblSkipSimd = il.DefineLabel();

                il.Emit(OpCodes.Ldloc, locCnt);
                il.Emit(OpCodes.Ldc_I8, (long)lanes);
                il.Emit(OpCodes.Blt, lblSkipSimd);

                // locWide16 = Vector{N}.Create(v128, v128) — pack two copies of the 16-byte slab.
                il.Emit(OpCodes.Ldloc, locVal);
                il.Emit(OpCodes.Ldloc, locVal);
                il.EmitCall(OpCodes.Call, VectorMethodCache.CreateFromHalves(VectorBits, typeof(byte)), null);
                il.Emit(OpCodes.Stloc, locWide16);

                EmitSimdBroadcastStage(
                    il, locK, locCnt, locDst, locWide16,
                    vecElem: typeof(byte),
                    lanesPerIter: lanes,
                    bytesPerIter: VectorBytes);

                il.MarkLabel(lblSkipSimd);
            }

            // ===== SCALAR TAIL — single-chunk-per-iter loop for the leftover =====
            il.MarkLabel(lblScalarTail);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locCnt);
            il.Emit(OpCodes.Bge, lblScalarTailEnd);

            EmitChunkCopy(il, locSrc, locDst, locVal, chunkBytes);

            // dst += chunkBytes
            il.Emit(OpCodes.Ldloc, locDst);
            EmitLdcPointerSize(il, chunkBytes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            // k++
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblScalarTail);
            il.MarkLabel(lblScalarTailEnd);

            // src += chunkBytes
            il.Emit(OpCodes.Ldloc, locSrc);
            EmitLdcPointerSize(il, chunkBytes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSrc);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblInner);
            il.MarkLabel(lblInnerEnd);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuter);
            il.MarkLabel(lblOuterEnd);

            il.Emit(OpCodes.Ret);
        }

        // Emits the SIMD-broadcast inner loop:
        //   while (k + lanesPerIter <= cnt) {
        //       Vector{N}.Store(vec, dst);
        //       dst += bytesPerIter;
        //       k   += lanesPerIter;
        //   }
        // The vector is already broadcast in `locVec` and the IL uses a single
        // Vector{N}.Store per iter — falls through to the scalar tail for the
        // remaining < lanesPerIter k-iterations.
        private static void EmitSimdBroadcastStage(
            ILGenerator il,
            LocalBuilder locK,
            LocalBuilder locCnt,
            LocalBuilder locDst,
            LocalBuilder locVec,
            Type vecElem,
            int lanesPerIter,
            int bytesPerIter)
        {
            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);

            // if (k + lanes > cnt) break
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, (long)lanesPerIter);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locCnt);
            il.Emit(OpCodes.Bgt, lblEnd);

            // Vector{N}.Store(locVec, (T*)dst)
            il.Emit(OpCodes.Ldloc, locVec);
            il.Emit(OpCodes.Ldloc, locDst);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Store(VectorBits, vecElem), null);

            // dst += bytesPerIter
            il.Emit(OpCodes.Ldloc, locDst);
            il.Emit(OpCodes.Ldc_I4, bytesPerIter);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDst);

            // k += lanesPerIter
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, (long)lanesPerIter);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        // Emit `Ldc_I4 chunkBytes; Conv_I` — chunkBytes as a native-int constant.
        private static void EmitLdcPointerSize(ILGenerator il, int chunkBytes)
        {
            il.Emit(OpCodes.Ldc_I4, chunkBytes);
            il.Emit(OpCodes.Conv_I);
        }

        // Hoist `*(T*)src` into `locVal` so the k-loop only stores.
        private static void EmitPreload(ILGenerator il, LocalBuilder locSrc, LocalBuilder locVal, int chunkBytes)
        {
            il.Emit(OpCodes.Ldloc, locSrc);
            switch (chunkBytes)
            {
                case 1:
                    il.Emit(OpCodes.Ldind_U1);
                    break;
                case 2:
                    il.Emit(OpCodes.Ldind_U2);
                    break;
                case 4:
                    il.Emit(OpCodes.Ldind_U4);
                    break;
                case 8:
                    il.Emit(OpCodes.Ldind_I8);
                    break;
                case 16:
                    il.EmitCall(OpCodes.Call, Vector128LoadByte, null);
                    break;
                default:
                    throw new InvalidOperationException($"EmitPreload: unsupported chunkBytes {chunkBytes}");
            }
            il.Emit(OpCodes.Stloc, locVal);
        }

        // Inner scalar-tail copy. For small chunks emits a typed store from the
        // hoisted register; for larger chunks emits `cpblk` with the size baked in
        // as a constant (so the JIT can specialize the memcpy size).
        private static void EmitChunkCopy(ILGenerator il, LocalBuilder locSrc, LocalBuilder locDst, LocalBuilder locVal, int chunkBytes)
        {
            switch (chunkBytes)
            {
                case 1:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locVal);
                    il.Emit(OpCodes.Stind_I1);
                    return;
                case 2:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locVal);
                    il.Emit(OpCodes.Stind_I2);
                    return;
                case 4:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locVal);
                    il.Emit(OpCodes.Stind_I4);
                    return;
                case 8:
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locVal);
                    il.Emit(OpCodes.Stind_I8);
                    return;
                case 16:
                    // Vector128.Store<byte>(value, dst)
                    il.Emit(OpCodes.Ldloc, locVal);
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.EmitCall(OpCodes.Call, Vector128StoreByte, null);
                    return;
                default:
                    // cpblk(dst, src, chunkBytes) — constant size so JIT can specialize.
                    il.Emit(OpCodes.Ldloc, locDst);
                    il.Emit(OpCodes.Ldloc, locSrc);
                    il.Emit(OpCodes.Ldc_I4, chunkBytes);
                    il.Emit(OpCodes.Cpblk);
                    return;
            }
        }

        // Reflected method handles for Vector128.Load/Store specialized to <byte> —
        // resolved once at type load so the IL emitter never pays GetMethod cost.
        private static readonly MethodInfo Vector128LoadByte =
            (typeof(Vector128).GetMethod(nameof(Vector128.Load), BindingFlags.Public | BindingFlags.Static)
                ?? throw new MissingMethodException("Vector128.Load<T>(T*) not found"))
            .MakeGenericMethod(typeof(byte));

        private static readonly MethodInfo Vector128StoreByte =
            (typeof(Vector128).GetMethod(nameof(Vector128.Store), BindingFlags.Public | BindingFlags.Static)
                ?? throw new MissingMethodException("Vector128.Store<T>(Vector128<T>, T*) not found"))
            .MakeGenericMethod(typeof(byte));
    }
}
