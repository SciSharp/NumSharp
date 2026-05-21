using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Repeat — IL-generated kernels for np.repeat
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
// Both kernel families specialize the inner copy by chunk size:
//   1, 2, 4, 8: preload the slab into a typed register once per j, then store
//   it `cnt` times — saves a load on every k iteration vs. repeated cpblk.
//   16:         same idea via a Vector128<byte> register.
//   anything else: cpblk with the size baked in as a constant so the JIT
//   specializes the memcpy (.NET 7+ optimizes constant-size cpblk).
//
// One generated kernel per (chunkBytes, kind) tuple. For typical workloads
// (axis=last on 15 dtypes -> 5 sizes; a few common shape/axis combos -> a
// handful more) the cache stays tiny.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
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

        private static readonly ConcurrentDictionary<int, RepeatBroadcastKernel> _repeatBroadcastCache = new();
        private static readonly ConcurrentDictionary<int, RepeatPerJKernel> _repeatPerJCache = new();

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
                owner: typeof(ILKernelGenerator),
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
                owner: typeof(ILKernelGenerator),
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
            var lblRepeat = il.DefineLabel();
            var lblRepeatEnd = il.DefineLabel();

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

            // ===== K (REPEAT) LOOP =====
            il.MarkLabel(lblRepeat);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locCnt);
            il.Emit(OpCodes.Bge, lblRepeatEnd);

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
            il.Emit(OpCodes.Br, lblRepeat);
            il.MarkLabel(lblRepeatEnd);

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

        // Inner-copy body. For small chunks emits a typed store from the hoisted
        // register; for larger chunks emits `cpblk` with the size baked in as a
        // constant (so the JIT can specialize the memcpy size).
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
