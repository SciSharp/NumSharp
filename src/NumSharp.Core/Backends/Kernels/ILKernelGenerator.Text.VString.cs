using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// ILKernelGenerator.Text.VString.cs — the inner loops of the StringDType (T) ufuncs: comparisons and truth
// =============================================================================
//
// RESPONSIBILITY
//   The NDIter inner loops (NDInnerLoopFunc) that read NEP 55 StringDType cells: the six comparison ufuncs (NumPy's
//   string_comparison_strided_loop, stringdtype_ufuncs.cpp) and a cell's truth (string_to_bool / the dtype's nonzero).
//   The strings plan's T kernel shell (D11 / D12): the IL owns the loop, the operand order, the operation and the stride
//   walk as emit-time constants; the per-pair body is a C# helper of NDVStringCells ([MethodImpl(OptimizeAndInline)]),
//   which the JIT inlines into the dynamic method with the operation constant folded. The helper's short-cell path
//   (both texts at most 15 bytes, inside their cells) needs no memory but the two cells; every other pair (arena / heap
//   strings, NA cells, the all-zero cell of a never-written element) calls the out-of-line per-pair path.
//
// CONTEXT
//   A comparison kernel's auxdata is a VStringPairAux*: the arena bases of the two inputs' allocators (read by the caller
//   with both allocators locked; a comparison never allocates, so they stay valid for the call), the NA rule of the call
//   (from the FIRST operand's descriptor, NumPy's descr1) and that descriptor's default string. A truth kernel needs no
//   arena (it reads only byte 15 and the size field); its NA truth is an emit-time constant.
//
// CONTRACT
//   dataptrs / strides (BYTES) per operand, any strides (0 for a broadcast operand, negative for a reversed view). The
//   cells must be read through their own buffer's arena (the caller guarantees the iterator does not buffer or copy a
//   StringDType operand: no cast, no COPY_IF_OVERLAP for a bool output).
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>The StringDType comparison kernels, keyed by (operation, operand order).</summary>
        private static readonly ConcurrentDictionary<VStringCompareKey, NDInnerLoopFunc> s_vstringCompareKernels = new();

        /// <summary>The StringDType truth kernels, keyed by the NA cell's truth (0 or 1).</summary>
        private static readonly ConcurrentDictionary<byte, NDInnerLoopFunc> s_vstringTruthKernels = new();

        /// <summary><see cref="NDVStringCells.Equal"/>, the equality body.</summary>
        private static readonly MethodInfo s_vstringEqual = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.Equal), BindingFlags.NonPublic | BindingFlags.Static)
                                                            ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.Equal));

        /// <summary><see cref="NDVStringCells.Less"/>, the ordering body.</summary>
        private static readonly MethodInfo s_vstringLess = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.Less), BindingFlags.NonPublic | BindingFlags.Static)
                                                           ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.Less));

        /// <summary><see cref="NDVStringCells.PrefetchPair"/>, the prefetching kernels' look-ahead.</summary>
        private static readonly MethodInfo s_vstringPrefetchPair = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.PrefetchPair), BindingFlags.NonPublic | BindingFlags.Static)
                                                                   ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.PrefetchPair));

        /// <summary><see cref="NDVStringCells.Truth"/>, the truth body.</summary>
        private static readonly MethodInfo s_vstringTruth = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.Truth), BindingFlags.NonPublic | BindingFlags.Static)
                                                            ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.Truth));

        /// <summary><see cref="NDVStringCells.TruthRun"/>, the contiguous-run truth (AVX2, four cells a step).</summary>
        private static readonly MethodInfo s_vstringTruthRun = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.TruthRun), BindingFlags.NonPublic | BindingFlags.Static)
                                                               ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.TruthRun));

        /// <summary><see cref="NDVStringCells.NARun"/>, the contiguous-run isnan (AVX2, four cells a step).</summary>
        private static readonly MethodInfo s_vstringNARun = typeof(NDVStringCells).GetMethod(nameof(NDVStringCells.NARun), BindingFlags.NonPublic | BindingFlags.Static)
                                                            ?? throw new MissingMethodException(nameof(NDVStringCells), nameof(NDVStringCells.NARun));

        /// <summary>
        ///     Emits the contiguous-run shortcut of a cells-to-bools kernel: when the cells walk at 16 bytes and the bools at 1
        ///     (a C-contiguous pair, or an iterator chunk of one), the whole chunk goes to <paramref name="run"/>
        ///     (<c>run(cells, flags, count[, extra])</c>) and the kernel returns; any other stride pair falls through to the
        ///     element loop the caller emits next.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="frame">The frame (cells first, bools second).</param>
        /// <param name="run">The run helper.</param>
        /// <param name="extra">A trailing byte argument of <paramref name="run"/> (the NA truth), or -1 for none.</param>
        private static void EmitContiguousCellRun(ILGenerator il, TextFrame frame, MethodInfo run, int extra)
        {
            var strided = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, frame.Strides[0]);
            il.Emit(OpCodes.Ldc_I8, (long)VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING);
            il.Emit(OpCodes.Bne_Un, strided);
            il.Emit(OpCodes.Ldloc, frame.Strides[1]);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Bne_Un, strided);
            il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
            il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
            il.Emit(OpCodes.Ldarg_2);
            if (extra >= 0)
                il.Emit(OpCodes.Ldc_I4, extra);
            il.Emit(OpCodes.Call, run);
            il.Emit(OpCodes.Ret);
            il.MarkLabel(strided);
        }

        /// <summary>The StringDType writing kernels (add / multiply / maximum / minimum and their size passes), by identity.</summary>
        private static readonly ConcurrentDictionary<VStringWriteKey, NDInnerLoopFunc> s_vstringWriteKernels = new();

        /// <summary>The StringDType isnan kernel (one form: the NA bit of a cell whose descriptor's NA is NaN-like).</summary>
        private static NDInnerLoopFunc s_vstringIsNaNKernel;

        /// <summary><see cref="VStringWriteContext.FromAux"/>: the managed context of a writing kernel's call.</summary>
        private static readonly MethodInfo s_vstringContextOf = typeof(VStringWriteContext).GetMethod(nameof(VStringWriteContext.FromAux), BindingFlags.NonPublic | BindingFlags.Static)
                                                                ?? throw new MissingMethodException(nameof(VStringWriteContext), nameof(VStringWriteContext.FromAux));

        /// <summary><see cref="VStringWriteContext.Reserve"/>, the size pass's sum (a size kernel adds its local total once).</summary>
        private static readonly FieldInfo s_vstringReserve = typeof(VStringWriteContext).GetField(nameof(VStringWriteContext.Reserve), BindingFlags.NonPublic | BindingFlags.Instance)
                                                             ?? throw new MissingFieldException(nameof(VStringWriteContext), nameof(VStringWriteContext.Reserve));

        /// <summary>
        ///     What a StringDType writing kernel computes per element; the helper of <see cref="NDVStringArithmetic"/> each
        ///     calls.
        /// </summary>
        internal enum VStringWriteOp : byte
        {
            /// <summary><c>out = A + B</c> (<see cref="NDVStringArithmetic.AddElement"/>).</summary>
            Add,

            /// <summary>The arena bytes <c>A + B</c> takes in a fresh output (<see cref="NDVStringArithmetic.AddSize"/>).</summary>
            AddSize,

            /// <summary><c>out = text * count</c>, an int64 count (<see cref="NDVStringArithmetic.MultiplyElement"/>).</summary>
            Multiply,

            /// <summary><c>out = text * count</c>, a uint64 count.</summary>
            MultiplyUnsigned,

            /// <summary>The arena bytes <c>text * count</c> takes in a fresh output, an int64 count.</summary>
            MultiplySize,

            /// <summary>The arena bytes <c>text * count</c> takes in a fresh output, a uint64 count.</summary>
            MultiplySizeUnsigned,

            /// <summary><c>out = maximum(A, B)</c> (<see cref="NDVStringArithmetic.MaxMinElement"/>).</summary>
            Maximum,

            /// <summary><c>out = minimum(A, B)</c>.</summary>
            Minimum,

            /// <summary>An upper bound of the arena bytes the winner of <c>maximum</c> / <c>minimum</c> takes in a fresh output
            ///     (<see cref="NDVStringArithmetic.MaxMinSize"/>).</summary>
            MaxMinSize,
        }

        /// <summary>The identity of a StringDType writing kernel.</summary>
        /// <param name="Op">The per-element operation.</param>
        /// <param name="TextIndex">The operand slot of the first text operand (multiply: the text; the count is the other
        ///     input slot); the output is slot 2 (a size pass has no output operand).</param>
        private readonly record struct VStringWriteKey(VStringWriteOp Op, int TextIndex);

        /// <summary>
        ///     The StringDType writing kernel for one (operation, operand order), generated on first use and cached: the
        ///     inputs at <c>dataptrs[TextIndex]</c> and <c>dataptrs[1 - TextIndex]</c>, the output at <c>dataptrs[2]</c> (none for
        ///     a size pass), and as the auxdata a pointer to the <see cref="System.Runtime.InteropServices.GCHandle"/> of the
        ///     call's <see cref="VStringWriteContext"/>, resolved ONCE per call by the kernel's prologue.
        /// </summary>
        /// <param name="op">The per-element operation.</param>
        /// <param name="textIndex">0 or 1.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="textIndex"/> is not 0 or 1.</exception>
        internal static NDInnerLoopFunc GetVStringWriteKernel(VStringWriteOp op, int textIndex)
        {
            if (textIndex is not (0 or 1))
                throw new ArgumentOutOfRangeException(nameof(textIndex), textIndex, "A writing kernel reads its inputs from slots 0 and 1.");
            if (!TextKernelsAvailable)
                return null;
            var key = new VStringWriteKey(op, textIndex);
            if (s_vstringWriteKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_vstringWriteKernels.GetOrAdd(key, GenerateVStringWriteKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetVStringWriteKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The StringDType isnan kernel: <c>out[i] = the NA bit of cell[i]</c> (the caller runs it only for a descriptor
        ///     whose NA object is NaN-like; any other descriptor's isnan is all False). Cells at <c>dataptrs[0]</c>, the bool
        ///     output at <c>dataptrs[1]</c>; no auxdata.
        /// </summary>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        internal static NDInnerLoopFunc GetVStringIsNaNKernel()
        {
            if (!TextKernelsAvailable)
                return null;
            if (s_vstringIsNaNKernel is { } cached)
                return cached;
            try
            {
                var dm = NewTextKernelMethod("NDVStringIsNaN");
                var il = dm.GetILGenerator();
                var frame = EmitTextFrame(il, new[] { 0, 1 });
                EmitContiguousCellRun(il, frame, s_vstringNARun, -1);
                EmitTextElementLoop(il, frame, new[] { true, true }, () =>
                {
                    // out = cell[15] >> 7: NPY_STRING_MISSING is the top bit of the flags byte.
                    il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                    il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                    il.Emit(OpCodes.Ldc_I4, 15);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldind_U1);
                    il.Emit(OpCodes.Ldc_I4_7);
                    il.Emit(OpCodes.Shr_Un);
                    il.Emit(OpCodes.Stind_I1);
                });
                il.Emit(OpCodes.Ret);
                var kernel = (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
                s_vstringIsNaNKernel = kernel;
                return kernel;
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetVStringIsNaNKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits a writing kernel: the frame, the prologue that resolves the call's context into a local, and the element
        ///     loop whose body calls the operation's helper (the helper decides per element between its in-cell fast path and
        ///     its allocator path).
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateVStringWriteKernel(VStringWriteKey key)
        {
            var dm = NewTextKernelMethod($"NDVStringWrite_{key.Op}_t{key.TextIndex}");
            var il = dm.GetILGenerator();
            bool sizePass = key.Op is VStringWriteOp.AddSize or VStringWriteOp.MultiplySize or VStringWriteOp.MultiplySizeUnsigned
                or VStringWriteOp.MaxMinSize;
            var frame = sizePass
                ? EmitTextFrame(il, new[] { key.TextIndex, 1 - key.TextIndex })
                : EmitTextFrame(il, new[] { key.TextIndex, 1 - key.TextIndex, 2 });
            var ctx = il.DeclareLocal(typeof(VStringWriteContext));
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Call, s_vstringContextOf);
            il.Emit(OpCodes.Stloc, ctx);

            (string name, int flag) = key.Op switch
            {
                VStringWriteOp.Add => (nameof(NDVStringArithmetic.AddElement), -1),
                VStringWriteOp.AddSize => (nameof(NDVStringArithmetic.AddSize), -1),
                VStringWriteOp.Multiply => (nameof(NDVStringArithmetic.MultiplyElement), 0),
                VStringWriteOp.MultiplyUnsigned => (nameof(NDVStringArithmetic.MultiplyElement), 1),
                VStringWriteOp.MultiplySize => (nameof(NDVStringArithmetic.MultiplySize), 0),
                VStringWriteOp.MultiplySizeUnsigned => (nameof(NDVStringArithmetic.MultiplySize), 1),
                VStringWriteOp.Maximum => (nameof(NDVStringArithmetic.MaxMinElement), 1),
                VStringWriteOp.Minimum => (nameof(NDVStringArithmetic.MaxMinElement), 0),
                _ => (nameof(NDVStringArithmetic.MaxMinSize), -1),
            };
            var helper = typeof(NDVStringArithmetic).GetMethod(name, BindingFlags.NonPublic | BindingFlags.Static)
                         ?? throw new MissingMethodException(nameof(NDVStringArithmetic), name);

            var advance = new bool[frame.Pointers.Length];
            for (int k = 0; k < advance.Length; k++)
                advance[k] = true;
            // A size pass's helper returns the element's arena bytes, summed in a local and added to the context's Reserve
            // once at the end: a += on the context object per element was a store-to-load chain through memory.
            var acc = sizePass ? il.DeclareLocal(typeof(long)) : null;
            if (sizePass)
            {
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, acc);
            }
            EmitTextElementLoop(il, frame, advance, () =>
            {
                if (sizePass)
                    il.Emit(OpCodes.Ldloc, acc);
                for (int k = 0; k < frame.Pointers.Length; k++)
                    il.Emit(OpCodes.Ldloc, frame.Pointers[k]);
                il.Emit(OpCodes.Ldloc, ctx);
                if (flag >= 0)
                    il.Emit(OpCodes.Ldc_I4, flag);
                il.Emit(OpCodes.Call, helper);
                if (sizePass)
                {
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Stloc, acc);
                }
            });
            if (sizePass)
            {
                il.Emit(OpCodes.Ldloc, ctx);
                il.Emit(OpCodes.Ldloc, ctx);
                il.Emit(OpCodes.Ldfld, s_vstringReserve);
                il.Emit(OpCodes.Ldloc, acc);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stfld, s_vstringReserve);
            }
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>The identity of a StringDType comparison kernel.</summary>
        /// <param name="Op">The comparison of A against B (<c>greater</c> / <c>greater_equal</c> are <c>less</c> /
        ///     <c>less_equal</c> of the swapped pair).</param>
        /// <param name="AIndex">The operand slot A is read from (0 or 1; B is the other, the bool output is slot 2).</param>
        /// <param name="Prefetch">True for the ordering form that prefetches the texts
        ///     <see cref="NDVStringCells.TextPrefetchDistance"/> pairs ahead (long texts, enough of them that their first lines
        ///     pass L2: <see cref="NDVStringCells.TakesPrefetchingKernel"/>).</param>
        private readonly record struct VStringCompareKey(TextCompareOp Op, int AIndex, bool Prefetch);

        /// <summary>
        ///     The StringDType comparison kernel for one (operation, operand order), generated on first use and cached:
        ///     <c>out[i] = A[i] op B[i]</c> with A = <c>dataptrs[aIndex]</c>, B = <c>dataptrs[1 - aIndex]</c>, the bool
        ///     output <c>dataptrs[2]</c>, and a <see cref="VStringPairAux"/> as the auxdata.
        /// </summary>
        /// <param name="op">The comparison of A against B.</param>
        /// <param name="aIndex">0, or 1 for the swapped pair of <c>greater</c> / <c>greater_equal</c>.</param>
        /// <param name="prefetch">True for the prefetching form of an ordering (the caller decides with
        ///     <see cref="NDVStringCells.TakesPrefetchingKernel"/>); both forms compute the same results.</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled (the caller takes its managed route).</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="aIndex"/> is not 0 or 1.</exception>
        /// <exception cref="ArgumentException"><paramref name="prefetch"/> for an equality (<c>==</c> / <c>!=</c>), which has
        ///     no prefetching form: it decides texts of different lengths by their cells' sizes and reads no text.</exception>
        internal static NDInnerLoopFunc GetVStringCompareKernel(TextCompareOp op, int aIndex, bool prefetch = false)
        {
            if (aIndex is not (0 or 1))
                throw new ArgumentOutOfRangeException(nameof(aIndex), aIndex, "A comparison reads its operands from slots 0 and 1.");
            if (prefetch && op is not (TextCompareOp.Less or TextCompareOp.LessEqual))
                throw new ArgumentException($"The {op} kernel has no prefetching form (an equality reads no text of a pair it decides by the sizes).", nameof(prefetch));
            if (!TextKernelsAvailable)
                return null;
            var key = new VStringCompareKey(op, aIndex, prefetch);
            if (s_vstringCompareKernels.TryGetValue(key, out var cached))
                return cached;
            try
            {
                return s_vstringCompareKernels.GetOrAdd(key, GenerateVStringCompareKernel(key));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                // Emission failed on this runtime: the managed route answers.
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetVStringCompareKernel({key}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     The StringDType truth kernel: <c>out[i] = truth(cell[i])</c> with the cells at <c>dataptrs[0]</c> and the bool
        ///     output at <c>dataptrs[1]</c>; an NA cell reads as <paramref name="naTruth"/> (the route's rule,
        ///     <see cref="NDVStringCells.NaTruth"/>). No auxdata.
        /// </summary>
        /// <param name="naTruth">The truth of an NA cell (0 or 1).</param>
        /// <returns>The kernel, or null when dynamic code is unavailable or disabled.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="naTruth"/> is not 0 or 1.</exception>
        internal static NDInnerLoopFunc GetVStringTruthKernel(byte naTruth)
        {
            if (naTruth > 1)
                throw new ArgumentOutOfRangeException(nameof(naTruth), naTruth, "An NA cell's truth is 0 or 1.");
            if (!TextKernelsAvailable)
                return null;
            if (s_vstringTruthKernels.TryGetValue(naTruth, out var cached))
                return cached;
            try
            {
                return s_vstringTruthKernels.GetOrAdd(naTruth, GenerateVStringTruthKernel(naTruth));
            }
            catch (Exception ex) when (ex is InvalidProgramException or NotSupportedException or PlatformNotSupportedException)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetVStringTruthKernel({naTruth}): {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        ///     Emits a comparison kernel: the frame (A, B, out), then the element loop whose body is one call of the inlined
        ///     pair helper with the operation's flag as a constant (<c>!=</c> / <c>&lt;=</c>), its byte stored into out. The
        ///     prefetching form first asks for the texts of the pair <see cref="NDVStringCells.TextPrefetchDistance"/> ahead
        ///     (<see cref="NDVStringCells.PrefetchPair"/>), while that pair is still inside the chunk.
        /// </summary>
        /// <param name="key">The kernel identity.</param>
        /// <returns>The compiled kernel.</returns>
        /// <remarks>Why a prefetch: a pair of random texts is ordered by their first eight bytes, so an ordering of long
        ///     texts reads one line of each text, every line a cache miss once the texts' first lines pass L2; the hardware
        ///     prefetchers do not follow those steps (consecutive long texts start lines apart, unevenly: no constant stride
        ///     for the stride prefetcher, no adjacent lines for the streamer). Measured (pinned) on <c>np.less</c> over
        ///     100,000 texts of 300–500 bytes: 27.7 → 17.7 ns a pair, where NumPy's loop takes 34.0. An equality has no
        ///     prefetching form (<see cref="NDVStringCells.TakesPrefetchingKernel"/>).</remarks>
        private static NDInnerLoopFunc GenerateVStringCompareKernel(VStringCompareKey key)
        {
            var dm = NewTextKernelMethod($"NDVStringCompare_{key.Op}_a{key.AIndex}{(key.Prefetch ? "_prefetch" : "")}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { key.AIndex, 1 - key.AIndex, 2 });
            bool ordering = key.Op is TextCompareOp.Less or TextCompareOp.LessEqual;
            var helper = ordering ? s_vstringLess : s_vstringEqual;
            // The helpers' last argument: Equal's `negate` (true for !=), Less's `orEqual` (true for <=).
            int flag = key.Op is TextCompareOp.NotEqual or TextCompareOp.LessEqual ? 1 : 0;
            EmitTextElementLoop(il, frame, new[] { true, true, true }, () =>
            {
                if (key.Prefetch)
                {
                    // if (i + D < count) PrefetchPair(pA + D * sA, pB + D * sB, aux): only cells of this chunk are read.
                    var skip = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, frame.Index);
                    il.Emit(OpCodes.Ldc_I8, (long)NDVStringCells.TextPrefetchDistance);
                    il.Emit(OpCodes.Add);
                    il.Emit(OpCodes.Ldarg_2);
                    il.Emit(OpCodes.Bge, skip);
                    for (int k = 0; k < 2; k++)
                    {
                        il.Emit(OpCodes.Ldloc, frame.Pointers[k]);
                        il.Emit(OpCodes.Ldloc, frame.Strides[k]);
                        il.Emit(OpCodes.Ldc_I8, (long)NDVStringCells.TextPrefetchDistance);
                        il.Emit(OpCodes.Mul);
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }
                    il.Emit(OpCodes.Ldarg_3);
                    il.Emit(OpCodes.Call, s_vstringPrefetchPair);
                    il.MarkLabel(skip);
                }
                il.Emit(OpCodes.Ldloc, frame.Pointers[2]);
                il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldc_I4, flag);
                il.Emit(OpCodes.Call, helper);
                il.Emit(OpCodes.Stind_I1);
            });
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }

        /// <summary>
        ///     Emits a truth kernel: the frame (cells, out), then the element loop whose body is one call of the inlined truth
        ///     helper with the NA truth as a constant.
        /// </summary>
        /// <param name="naTruth">The truth of an NA cell.</param>
        /// <returns>The compiled kernel.</returns>
        private static NDInnerLoopFunc GenerateVStringTruthKernel(byte naTruth)
        {
            var dm = NewTextKernelMethod($"NDVStringTruth_na{naTruth}");
            var il = dm.GetILGenerator();
            var frame = EmitTextFrame(il, new[] { 0, 1 });
            EmitContiguousCellRun(il, frame, s_vstringTruthRun, naTruth);
            EmitTextElementLoop(il, frame, new[] { true, true }, () =>
            {
                il.Emit(OpCodes.Ldloc, frame.Pointers[1]);
                il.Emit(OpCodes.Ldloc, frame.Pointers[0]);
                il.Emit(OpCodes.Ldc_I4, (int)naTruth);
                il.Emit(OpCodes.Call, s_vstringTruth);
                il.Emit(OpCodes.Stind_I1);
            });
            il.Emit(OpCodes.Ret);
            return (NDInnerLoopFunc)dm.CreateDelegate(typeof(NDInnerLoopFunc));
        }
    }
}
