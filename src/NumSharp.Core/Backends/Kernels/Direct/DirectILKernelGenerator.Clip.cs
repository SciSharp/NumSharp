using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;

// =============================================================================
// DirectILKernelGenerator.Clip — IL-generated clip kernels
// =============================================================================
//
// Single entry point for ALL clip operations. The engine just builds an
// (NPTypeCode, ClipMode, ClipBoundsKind) tuple, hands raw pointers to
// `Clip(...)`, and DirectILKernelGenerator picks (or generates and caches) a
// DynamicMethod that runs the entire loop.
//
// Width-adaptive: the SIMD section is emitted against
// `GetVectorContainerType()`, which resolves at startup to
// `Vector512` / `Vector256` / `Vector128` based on hardware capability.
// Loop body is `Vector.Max(src, lo) ; Vector.Min(., hi)`.
//
// Semantics: result = Min(Max(src, lo), hi) — matches NumPy.
//   - BothBounds: both clamps applied (in that order, so min > max gives max).
//   - MinOnly:    only Max(src, lo).
//   - MaxOnly:    only Min(src, hi).
//   - BothBounds + Scalar bounds + float32/float64: NumPy's CONST fast path instead
//     (clip.cpp _npy_clip_const_minmax_, taken whenever both bound strides are 0):
//     x = x < lo ? lo : x ; x = x > hi ? hi : x. It keeps x on a tie, where
//     Min(Max()) takes the bound: clip(-0.0, 0, 1) is -0.0 in NumPy, not +0.0.
//     float16 and complex have no such path in NumPy (half_tag / complex_tag are
//     not floating_point_tag), and the integers cannot tell the two forms apart.
//     On x86 those two compare-selects ARE MAXPD(lo, x) / MINPD(hi, x) (the
//     instruction returns its second operand on a tie or a NaN), so the const body
//     is one instruction per bound (TryGetX86OrderedMaxMin).
//
// Stores (the vector section):
//   - An ALIGNMENT PEEL clips the first elements scalar until dst is aligned to
//     the vector width, so every vector store is aligned (a misaligned output cost
//     ~15-20% at 100K float64, and the buffer pool hands out either alignment).
//   - Past ClipStreamingMinBytes (16 MiB) an out-of-place call writes the vectors
//     with aligned NON-TEMPORAL stores and ends with sfence: input plus result no
//     longer fit the last-level cache, and a streaming store skips each line's
//     read-for-ownership (1.5-2.8x at 2.5M-10M elements). The variant is part of
//     the kernel key; both compute the same bytes.
//
// Bounds kind:
//   - Scalar: `lo`, `hi` point to one element of the dtype. Broadcast into
//             a register vector once, before the SIMD loop.
//   - Array:  `lo`, `hi` point to `size`-element arrays of the dtype. Loaded
//             per vector iteration.
//
// The same kernel handles in-place (src == dst) and fused copy+clip
// (src != dst) — the pointers are independent.
//
// Non-SIMD dtypes (Char, Decimal, Half, Complex) get a pure scalar IL loop.
// Complex / Half delegate the per-element clamp to small static helpers
// that implement NaN-aware semantics; the loop itself is IL.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Public API

        public enum ClipMode : byte
        {
            BothBounds = 0,
            MinOnly    = 1,
            MaxOnly    = 2,
        }

        public enum ClipBoundsKind : byte
        {
            Scalar = 0,
            Array  = 1,
        }

        /// <summary>
        /// Universal clip kernel signature: read from <paramref name="src"/>,
        /// clamp to <paramref name="lo"/> / <paramref name="hi"/>, write to
        /// <paramref name="dst"/>. <paramref name="size"/> is element count.
        /// Bound pointers are interpreted per the mode the kernel was
        /// generated for (scalar = single value, array = `size` values).
        /// Bound pointers for unused sides are ignored by the generated IL.
        /// </summary>
        public unsafe delegate void ClipKernel(void* src, void* dst, long size, void* lo, void* hi);

        internal static readonly ConcurrentDictionary<int, ClipKernel> _clipKernelCache = new();

        /// <summary>
        /// Run a clip operation. Picks (and on first call, IL-generates) the
        /// appropriate DynamicMethod for the (dtype, mode, kind) tuple and
        /// invokes it with the supplied pointers.
        /// </summary>
        public static unsafe void Clip(
            NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind,
            void* src, void* dst, long size, void* lo, void* hi)
        {
            // Bit 24 = the streaming-store variant (Complex = 128 occupies bit 23 of the dtype field, so no collision).
            bool streaming = UseClipStreamingStores(dtype, src, dst, size);
            int key = (streaming ? 1 << 24 : 0) | ((int)dtype << 16) | ((int)mode << 8) | (int)kind;
            var kernel = _clipKernelCache.GetOrAdd(key, _ => Generate(dtype, mode, kind, streaming));
            kernel(src, dst, size, lo, hi);
        }

        /// <summary>
        /// The result size, in bytes, from which an out-of-place clip writes its vectors with NON-TEMPORAL stores: once
        /// input plus result outgrow the last-level cache the result is evicted to memory anyway, and a streaming store
        /// skips the read-for-ownership of every destination line (a third of the memory traffic of a read-one-write-one
        /// kernel). Below it the result may still be cache-resident for its consumer, which plain stores keep.
        /// </summary>
        /// <remarks>
        /// 16 MiB: input + result = 32 MiB, about this host's 36 MB L3 (i9-13900K). Measured in one process, plain vs
        /// streaming, into a reused out= (boosted, unpinned): 100K elements 0.60-0.65x (L2-resident: streaming HURTS),
        /// 1M 1.05-1.28x, float64 2.5M (20 MB result) 2.24x, float32 / int32 5M (20 MB) 1.85-2.79x, 10M 1.55-1.64x. The
        /// 4-10 MB results gain at most ~1.2x for the producer while evicting a result a consumer could still read from
        /// L3, so they keep plain stores.
        /// </remarks>
        internal const long ClipStreamingMinBytes = 16L << 20;

        /// <summary>
        /// Test hook: forces the streaming-store variant on (<c>true</c>) or off (<c>false</c>) whatever the size; <c>null</c>
        /// (the default) applies <see cref="ClipStreamingMinBytes"/>. The two variants compute the same bytes; only the store
        /// instruction differs.
        /// </summary>
        internal static bool? ClipStreamingOverride;

        /// <summary>
        /// Whether a flat clip call writes its vectors with non-temporal stores (see <see cref="ClipStreamingMinBytes"/>).
        /// </summary>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="src">The input's first element.</param>
        /// <param name="dst">The output's first element.</param>
        /// <param name="size">The element count.</param>
        /// <returns>True for a large out-of-place call of a vectorized dtype on a host with the streaming store, whose
        /// output is element-aligned (so the peel can align the vector stores, which a streaming store requires).</returns>
        /// <remarks>An in-place call (<paramref name="src"/> == <paramref name="dst"/>) keeps plain stores: the element's line
        /// was just read into the cache, so there is no read-for-ownership to save.</remarks>
        private static unsafe bool UseClipStreamingStores(NPTypeCode dtype, void* src, void* dst, long size)
        {
            // This runs on EVERY flat clip call, so the cheap tests come first and the store lookup is cached: the
            // first version resolved StoreAlignedNonTemporal by reflection per call (~0.4 us and ~0.5 KB of garbage,
            // as much as a whole 1K-element clip).
            if (src == dst)
                return false;
            int sz = GetTypeSize(dtype);
            bool wanted = ClipStreamingOverride ?? size * sz >= ClipStreamingMinBytes;
            if (!wanted || ((long)dst & (sz - 1)) != 0)
                return false;
            return CanUseSimd(dtype) && SupportsVectorMinMax(dtype) && HasNonTemporalStore(dtype);
        }

        /// <summary>Per dtype (indexed by <c>(int)dtype &amp; 31</c>): 0 = not looked up yet, 1 = no streaming store, 2 = available.</summary>
        private static readonly byte[] s_clipNonTemporalAvailable = new byte[32];

        /// <summary>Whether <see cref="GetNonTemporalStore"/> finds a store for the dtype's lane, looked up once per dtype.</summary>
        /// <param name="dtype">A vectorized clip dtype.</param>
        /// <returns>True when the host has the aligned non-temporal store at the baked vector width.</returns>
        private static bool HasNonTemporalStore(NPTypeCode dtype)
        {
            int slot = (int)dtype & 31;
            byte known = System.Threading.Volatile.Read(ref s_clipNonTemporalAvailable[slot]);
            if (known == 0)
            {
                known = GetNonTemporalStore(GetClrType(dtype)) is null ? (byte)1 : (byte)2;
                System.Threading.Volatile.Write(ref s_clipNonTemporalAvailable[slot], known);
            }
            return known == 2;
        }

        /// <summary>
        /// The aligned non-temporal vector store for <paramref name="laneType"/> at the baked vector width
        /// (<c>Sse/Sse2/Avx/Avx512F.StoreAlignedNonTemporal(T*, Vector{W}&lt;T&gt;)</c>), or null when the host has none.
        /// </summary>
        /// <param name="laneType">The lane type.</param>
        /// <returns>The method, or null.</returns>
        private static MethodInfo GetNonTemporalStore(Type laneType)
        {
            Type isa = VectorBits switch
            {
                128 when laneType == typeof(float) && System.Runtime.Intrinsics.X86.Sse.IsSupported => typeof(System.Runtime.Intrinsics.X86.Sse),
                128 when System.Runtime.Intrinsics.X86.Sse2.IsSupported => typeof(System.Runtime.Intrinsics.X86.Sse2),
                256 when System.Runtime.Intrinsics.X86.Avx.IsSupported => typeof(System.Runtime.Intrinsics.X86.Avx),
                512 when System.Runtime.Intrinsics.X86.Avx512F.IsSupported => typeof(System.Runtime.Intrinsics.X86.Avx512F),
                _ => null,
            };
            return isa?.GetMethod("StoreAlignedNonTemporal", new[] { laneType.MakePointerType(), VectorMethodCache.V(VectorBits, laneType) });
        }

        /// <summary><c>Sse.StoreFence</c> (sfence): orders a streaming kernel's weakly-ordered stores before it returns.</summary>
        private static readonly MethodInfo s_clipStoreFence =
            typeof(System.Runtime.Intrinsics.X86.Sse).GetMethod(nameof(System.Runtime.Intrinsics.X86.Sse.StoreFence), Type.EmptyTypes);

        #endregion

        #region IL Generation

        /// <summary>Emits one flat clip kernel (see the file header).</summary>
        /// <param name="dtype">The loop dtype every pointer carries.</param>
        /// <param name="mode">Which bounds are present.</param>
        /// <param name="kind">Scalar bounds (one value each) or array bounds (one per element).</param>
        /// <param name="streaming">Write the vector section with aligned non-temporal stores and end with a store fence
        /// (only requested by <see cref="UseClipStreamingStores"/>, which guarantees an element-aligned output).</param>
        /// <returns>The compiled kernel.</returns>
        private static ClipKernel Generate(NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind, bool streaming)
        {
            // f16 clip is two chained bit-level pair-selects (max with lo, min with hi) on
            // raw ushort lanes — zero float conversions, every (mode, kind) combo served.
            // See Binary.MinMax.Half.cs (the emitted scalar loop below served Half before
            // via per-element HalfMaxNaN/HalfMinNaN calls).
            if (dtype == NPTypeCode.Half)
                return GetHalfClipKernel(mode, kind);

            var dm = new DynamicMethod(
                name: $"Clip_{dtype}_{mode}_{kind}{(streaming ? "_nt" : "")}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void*), typeof(void*), typeof(long), typeof(void*), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            int sz = GetTypeSize(dtype);

            // Locals shared between SIMD loop and scalar tail
            var locI = il.DeclareLocal(typeof(long));

            // For scalar bounds: hoist the scalar value(s) into locals so both
            // the SIMD broadcast vector and the scalar tail can re-use them.
            var clrType = GetClrType(dtype);
            LocalBuilder locLoVal = (mode != ClipMode.MaxOnly && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(clrType) : null;
            LocalBuilder locHiVal = (mode != ClipMode.MinOnly && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(clrType) : null;

            if (locLoVal != null)
            {
                il.Emit(OpCodes.Ldarg_3);                  // lo
                EmitLoadIndirect(il, dtype);
                il.Emit(OpCodes.Stloc, locLoVal);
            }
            if (locHiVal != null)
            {
                il.Emit(OpCodes.Ldarg_S, (byte)4);         // hi
                EmitLoadIndirect(il, dtype);
                il.Emit(OpCodes.Stloc, locHiVal);
            }

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // SIMD section — only for hardware-supported types (skipped for
            // Char / Decimal / Half / Complex, which have no Vector<T> Min/Max).
            MethodInfo ntStore = null;
            if (CanUseSimd(dtype) && SupportsVectorMinMax(dtype))
            {
                // Output-alignment peel: scalar elements until dst + i*sz is vector-aligned, so every vector store
                // after it is aligned. A misaligned 32-byte store splits a cache line every other vector; measured on
                // 100K float64 (L2-resident) a misaligned output cost ~15-20% (13.5-14.5 vs 11.5 us), and the pool
                // hands out buffers at either alignment. A streaming store REQUIRES the alignment.
                var locPeelEnd = il.DeclareLocal(typeof(long));
                EmitClipPeelEnd(il, sz, locPeelEnd);
                EmitScalarLoop(il, dtype, mode, kind, sz, locI, locLoVal, locHiVal, locPeelEnd);

                if (streaming)
                    ntStore = GetNonTemporalStore(GetClrType(dtype));
                EmitSimdLoop(il, dtype, mode, kind, sz, locI, locLoVal, locHiVal, ntStore);
            }

            // Scalar loop covers the tail after SIMD and the entire range for
            // non-SIMD dtypes.
            EmitScalarLoop(il, dtype, mode, kind, sz, locI, locLoVal, locHiVal);

            // Streaming stores are weakly ordered: fence them before the result is handed back.
            if (ntStore != null)
                il.EmitCall(OpCodes.Call, s_clipStoreFence, null);

            il.Emit(OpCodes.Ret);
            return (ClipKernel)dm.CreateDelegate(typeof(ClipKernel));
        }

        // Vector<T>.Min/Max exist for: byte, sbyte, short, ushort, int, uint,
        // long, ulong (since .NET 8), float, double. They don't exist for
        // Char (unsigned 16-bit but no overload) — we route Char through the
        // scalar IL loop.
        private static bool SupportsVectorMinMax(NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or
            NPTypeCode.Int16 or NPTypeCode.UInt16 or
            NPTypeCode.Int32 or NPTypeCode.UInt32 or
            NPTypeCode.Int64 or NPTypeCode.UInt64 or
            NPTypeCode.Single or NPTypeCode.Double => true,
            _ => false
        };

        // Emits the address `argN + byteOffset` (as native int) onto the stack.
        // argIdx: 0=src, 1=dst, 3=lo, 4=hi (arg2 is the long size).
        // Callers pre-compute `byteOffset = i * sz` once per iteration into a
        // local, then pass that local in — avoids recomputing the same i*sz
        // multiplication for the 2–4 pointers touched each iteration.
        private static void EmitOffsetAddrFromLocal(ILGenerator il, int argIdx, LocalBuilder locByteOff)
        {
            switch (argIdx)
            {
                case 0: il.Emit(OpCodes.Ldarg_0); break;
                case 1: il.Emit(OpCodes.Ldarg_1); break;
                case 3: il.Emit(OpCodes.Ldarg_3); break;
                case 4: il.Emit(OpCodes.Ldarg_S, (byte)4); break;
                default: throw new ArgumentException($"Unexpected argIdx {argIdx}");
            }
            il.Emit(OpCodes.Ldloc, locByteOff);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Conv_I);
        }

        private const int ClipUnroll = 4;

        /// <summary>
        /// The vector section of a flat clip kernel: the ClipUnroll×-unrolled body, then single vectors, starting at
        /// <paramref name="locI"/> (after the alignment peel) and leaving <paramref name="locI"/> at the first element the
        /// scalar tail must still do.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="mode">Which bounds are present.</param>
        /// <param name="kind">Scalar or array bounds.</param>
        /// <param name="sz">The element size in bytes.</param>
        /// <param name="locI">The element index (in: the first vector element; out: the first tail element).</param>
        /// <param name="locLoVal">The scalar lower bound (scalar kind), or null.</param>
        /// <param name="locHiVal">The scalar upper bound (scalar kind), or null.</param>
        /// <param name="ntStore">Null for plain stores, or the aligned non-temporal store (the streaming variant).</param>
        private static void EmitSimdLoop(
            ILGenerator il, NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind, int sz,
            LocalBuilder locI, LocalBuilder locLoVal, LocalBuilder locHiVal, MethodInfo ntStore = null)
        {
            var vectorType = VectorMethodCache.V(VectorBits, GetClrType(dtype));
            int vectorCount = GetVectorCount(dtype);
            bool needLo = mode != ClipMode.MaxOnly;
            bool needHi = mode != ClipMode.MinOnly;
            bool constSelect = UsesConstSelect(dtype, mode, kind);

            // Hoist Vector.Create(scalarValue) outside the loop for scalar bounds.
            LocalBuilder locLoVec = (needLo && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(vectorType) : null;
            LocalBuilder locHiVec = (needHi && kind == ClipBoundsKind.Scalar) ? il.DeclareLocal(vectorType) : null;
            if (locLoVec != null)
            {
                il.Emit(OpCodes.Ldloc, locLoVal);
                EmitVectorCreate(il, dtype);
                il.Emit(OpCodes.Stloc, locLoVec);
            }
            if (locHiVec != null)
            {
                il.Emit(OpCodes.Ldloc, locHiVal);
                EmitVectorCreate(il, dtype);
                il.Emit(OpCodes.Stloc, locHiVec);
            }

            long vcBytes = (long)vectorCount * sz;

            // ── ClipUnroll×-unrolled body ──────────────────────────────────────
            // clip has NO loop-carried dependency, so unrolling cuts the per-vector
            // loop overhead (the i*sz multiply + bounds branch) and feeds the two
            // min/max ALU ports more independent work. Each iteration processes
            // ClipUnroll vectors at byteOff0 + k*vcBytes (one multiply, constant
            // adds). Measured f64 100K clip(out=) 0.68x -> ~0.95x vs NumPy.
            var bo = new LocalBuilder[ClipUnroll];
            for (int k = 0; k < ClipUnroll; k++) bo[k] = il.DeclareLocal(typeof(long));

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)(ClipUnroll * vectorCount));
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            var lblUnroll = il.DefineLabel();
            var lblAfterUnroll = il.DefineLabel();
            il.MarkLabel(lblUnroll);
            // if (i > unrollEnd) break  (note: skipped entirely when size < ClipUnroll*vc)
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblAfterUnroll);

            // bo[0] = i * sz ; bo[k] = bo[0] + k*vcBytes
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            for (int k = 1; k < ClipUnroll; k++)
            {
                il.Emit(OpCodes.Ldloc, bo[0]);
                il.Emit(OpCodes.Ldc_I8, k * vcBytes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, bo[k]);
            }
            for (int k = 0; k < ClipUnroll; k++)
                EmitClipVectorBody(il, dtype, kind, bo[k], needLo, needHi, locLoVec, locHiVec, constSelect, ntStore);

            // i += ClipUnroll * vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)(ClipUnroll * vectorCount));
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblAfterUnroll);

            // ── Single-vector remainder ────────────────────────────────────────
            // vecEnd = size - vectorCount
            var locVecEnd = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);
            // if (i > vecEnd) break
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblEnd);

            // byteOff = i * sz
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, bo[0]);
            EmitClipVectorBody(il, dtype, kind, bo[0], needLo, needHi, locLoVec, locHiVec, constSelect, ntStore);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        /// <summary>
        /// True when NumPy runs its CONST fast path for this kernel: a two-sided clip of float32/float64
        /// with scalar bounds (clip.cpp <c>_npy_clip_const_minmax_</c>, the <c>steps[1] == 0 &amp;&amp; steps[2] == 0</c>
        /// branch, floating_point_tag only). Its compare-select keeps the element on a tie with a bound, where the
        /// general <c>_NPY_MIN(_NPY_MAX(x, lo), hi)</c> path takes the bound, so the two disagree on signed zeros:
        /// clip(-0.0, 0.0, 1.0) is -0.0 on the fast path and +0.0 on the general one.
        /// </summary>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="mode">The clip mode. A one-sided clip is NumPy's minimum / maximum (np.clip routes it there),
        /// whose ties take the second operand, so it never uses the const path.</param>
        /// <param name="kind">The bound kind. Array bounds take NumPy's general path; NumPy also takes the const path
        /// for array bounds that are constant along its inner loop, a corner this kernel does not reproduce.</param>
        /// <returns>True to emit the const fast path's compare-select instead of min/max.</returns>
        private static bool UsesConstSelect(NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind)
            => mode == ClipMode.BothBounds && kind == ClipBoundsKind.Scalar
               && (dtype == NPTypeCode.Single || dtype == NPTypeCode.Double);

        /// <summary>
        /// The x86 MAX/MIN instructions for float32/float64 lanes at the baked vector width
        /// (<c>Sse.Max</c> / <c>Sse2.Max</c>, <c>Avx.Max</c>, <c>Avx512F.Max</c> and their Min twins),
        /// whose operand order IS NumPy's clip const path: <c>MAXPS/MAXPD a, b</c> computes
        /// <c>a &gt; b ? a : b</c> and <c>MINPS/MINPD a, b</c> computes <c>a &lt; b ? a : b</c>, both
        /// returning the SECOND operand on a tie (±0 against ∓0) or when either is NaN. So
        /// <c>Max(lo, x)</c> is <c>x &lt; lo ? lo : x</c> and <c>Min(hi, x)</c> is <c>x &gt; hi ? hi : x</c>,
        /// lane for lane — a signed-zero tie keeps the element, a NaN element passes through.
        /// </summary>
        /// <param name="laneType">The lane type: <c>float</c> or <c>double</c>.</param>
        /// <param name="max">The ordered max (<c>Isa.Max(Vector{W}, Vector{W})</c>), or null.</param>
        /// <param name="min">The ordered min, or null.</param>
        /// <returns>
        /// True on an x86 host whose ISA matches <see cref="VectorBits"/>. False for any other lane type
        /// or host: the portable <c>Vector{W}.Max/Min</c> follow IEEE 754-2019 maximum/minimum (a NaN
        /// operand wins, -0 &lt; +0) and ARM's FMAX/FMIN propagate NaN, so neither may replace the
        /// compare-selects — those callers keep the explicit <c>ConditionalSelect</c> form.
        /// </returns>
        /// <remarks>
        /// RyuJIT keeps the operand order of the floating-point x86 Max/Min intrinsics (it treats them
        /// as commutative only for integer lanes), which the oracle's signed-zero tie cases pin: a
        /// swapped pair would turn <c>clip(-0.0, 0.0, 1.0)</c> from -0.0 into +0.0.
        /// </remarks>
        internal static bool TryGetX86OrderedMaxMin(Type laneType, out MethodInfo max, out MethodInfo min)
        {
            max = null;
            min = null;
            if (laneType != typeof(float) && laneType != typeof(double))
                return false;

            Type isa = VectorBits switch
            {
                128 when laneType == typeof(float) && System.Runtime.Intrinsics.X86.Sse.IsSupported => typeof(System.Runtime.Intrinsics.X86.Sse),
                128 when laneType == typeof(double) && System.Runtime.Intrinsics.X86.Sse2.IsSupported => typeof(System.Runtime.Intrinsics.X86.Sse2),
                256 when System.Runtime.Intrinsics.X86.Avx.IsSupported => typeof(System.Runtime.Intrinsics.X86.Avx),
                512 when System.Runtime.Intrinsics.X86.Avx512F.IsSupported => typeof(System.Runtime.Intrinsics.X86.Avx512F),
                _ => null,
            };
            if (isa == null)
                return false;

            var vecType = VectorMethodCache.V(VectorBits, laneType);
            max = isa.GetMethod("Max", new[] { vecType, vecType });
            min = isa.GetMethod("Min", new[] { vecType, vecType });
            if (max == null || min == null)
            {
                max = null;
                min = null;
                return false;
            }
            return true;
        }

        // Emits ONE vector clip step: load src at byteOff, clamp to lo/hi (NaN-propagating
        // float min/max via EmitVectorMinOrMax), store at dst+byteOff. Stack-neutral. Shared
        // by the unrolled body and the single-vector remainder so both stay bit-identical.
        // constSelect emits NumPy's const fast path instead (see UsesConstSelect): a NaN element
        // fails both compares and passes through, so no NaN fix-up is needed there.
        private static void EmitClipVectorBody(
            ILGenerator il, NPTypeCode dtype, ClipBoundsKind kind, LocalBuilder locByteOff,
            bool needLo, bool needHi, LocalBuilder locLoVec, LocalBuilder locHiVec, bool constSelect,
            MethodInfo ntStore = null)
        {
            EmitOffsetAddrFromLocal(il, 0, locByteOff);
            EmitVectorLoad(il, dtype);                                         // [vec]

            if (constSelect && TryGetX86OrderedMaxMin(GetClrType(dtype), out var orderedMax, out var orderedMin))
            {
                // x = Max(lo, x) ; x = Min(hi, x) with x86's ordered MAXPD/MINPD, which ARE the two
                // compare-selects below (see TryGetX86OrderedMaxMin): one instruction per bound
                // instead of a compare plus a blend. The dispatcher has already replaced a NaN bound
                // with a fill, so no NaN-bound fix-up is needed on this path.
                var locX = il.DeclareLocal(VectorMethodCache.V(VectorBits, GetClrType(dtype)));
                il.Emit(OpCodes.Stloc, locX);                                  // []
                il.Emit(OpCodes.Ldloc, locLoVec);
                il.Emit(OpCodes.Ldloc, locX);
                il.EmitCall(OpCodes.Call, orderedMax, null);                   // [lo > x ? lo : x]
                il.Emit(OpCodes.Stloc, locX);
                il.Emit(OpCodes.Ldloc, locHiVec);
                il.Emit(OpCodes.Ldloc, locX);
                il.EmitCall(OpCodes.Call, orderedMin, null);                   // [hi < x ? hi : x]
                EmitClipStore(il, dtype, locByteOff, ntStore);                 // []
                return;
            }

            if (constSelect)
            {
                // x = ConditionalSelect(x < lo, lo, x) ; x = ConditionalSelect(x > hi, hi, x)
                var clrType = GetClrType(dtype);
                var locX = il.DeclareLocal(VectorMethodCache.V(VectorBits, clrType));
                il.Emit(OpCodes.Stloc, locX);                                  // []
                il.Emit(OpCodes.Ldloc, locX);
                il.Emit(OpCodes.Ldloc, locLoVec);
                il.EmitCall(OpCodes.Call, VectorMethodCache.LessThan(VectorBits, clrType), null);          // [x<lo]
                il.Emit(OpCodes.Ldloc, locLoVec);
                il.Emit(OpCodes.Ldloc, locX);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null); // [x']
                il.Emit(OpCodes.Stloc, locX);                                  // []
                il.Emit(OpCodes.Ldloc, locX);
                il.Emit(OpCodes.Ldloc, locHiVec);
                il.EmitCall(OpCodes.Call, VectorMethodCache.GreaterThan(VectorBits, clrType), null);       // [x>hi]
                il.Emit(OpCodes.Ldloc, locHiVec);
                il.Emit(OpCodes.Ldloc, locX);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clrType), null); // [x'']
                EmitClipStore(il, dtype, locByteOff, ntStore);                 // []
                return;
            }

            if (needLo)
            {
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locLoVec);
                else
                {
                    EmitOffsetAddrFromLocal(il, 3, locByteOff);
                    EmitVectorLoad(il, dtype);
                }
                EmitVectorMinOrMax(il, isMax: true, dtype, propagateNaN: true);    // vec = Max(vec, lo)
            }

            if (needHi)
            {
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locHiVec);
                else
                {
                    EmitOffsetAddrFromLocal(il, 4, locByteOff);
                    EmitVectorLoad(il, dtype);
                }
                EmitVectorMinOrMax(il, isMax: false, dtype, propagateNaN: true);   // vec = Min(vec, hi)
            }

            EmitClipStore(il, dtype, locByteOff, ntStore);                     // []
        }

        /// <summary>
        /// Stores the clipped vector on the stack at <c>dst + byteOff</c>: a plain vector store, or — in the streaming
        /// variant — an aligned non-temporal store (the peel guarantees the alignment).
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="locByteOff">The group's byte offset.</param>
        /// <param name="ntStore">Null for a plain store, or the non-temporal store method.</param>
        private static void EmitClipStore(ILGenerator il, NPTypeCode dtype, LocalBuilder locByteOff, MethodInfo ntStore)
        {
            if (ntStore == null)
            {
                EmitOffsetAddrFromLocal(il, 1, locByteOff);
                EmitVectorStore(il, dtype);
                return;
            }
            // StoreAlignedNonTemporal(T* address, Vector{W}<T> source) wants [ptr, vec]; the stack holds [vec].
            var locVec = il.DeclareLocal(VectorMethodCache.V(VectorBits, GetClrType(dtype)));
            il.Emit(OpCodes.Stloc, locVec);
            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            il.Emit(OpCodes.Ldloc, locVec);
            il.EmitCall(OpCodes.Call, ntStore, null);
        }

        /// <summary>
        /// Emits <c>peelEnd</c>: how many leading elements the scalar loop clips before the vector section so that
        /// <c>dst + peelEnd*sz</c> is aligned to the vector width — <c>(W − dst % W) / sz</c>, 0 when dst is already aligned
        /// or not element-aligned (then no peel can help), capped at the size.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="sz">The element size in bytes.</param>
        /// <param name="locPeelEnd">Receives the peel end (an element index).</param>
        private static void EmitClipPeelEnd(ILGenerator il, int sz, LocalBuilder locPeelEnd)
        {
            long vectorBytes = VectorBits / 8;
            var locMis = il.DeclareLocal(typeof(long));
            var lblDone = il.DefineLabel();

            // mis = (long)dst & (W - 1)
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Ldc_I8, vectorBytes - 1);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, locMis);

            // peelEnd = 0; if (mis == 0 || (mis & (sz - 1)) != 0) done
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locPeelEnd);
            il.Emit(OpCodes.Ldloc, locMis);
            il.Emit(OpCodes.Brfalse, lblDone);
            il.Emit(OpCodes.Ldloc, locMis);
            il.Emit(OpCodes.Ldc_I8, (long)(sz - 1));
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Brtrue, lblDone);

            // peelEnd = (W - mis) / sz, capped at size
            il.Emit(OpCodes.Ldc_I8, vectorBytes);
            il.Emit(OpCodes.Ldloc, locMis);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locPeelEnd);
            il.Emit(OpCodes.Ldloc, locPeelEnd);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ble, lblDone);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Stloc, locPeelEnd);

            il.MarkLabel(lblDone);
        }

        /// <summary>
        /// The scalar clip loop from <paramref name="locI"/> to <paramref name="locEnd"/> (or to the size): the peel before
        /// the vector section, the tail after it, and the whole range for dtypes without a vector form.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="mode">Which bounds are present.</param>
        /// <param name="kind">Scalar or array bounds.</param>
        /// <param name="sz">The element size in bytes.</param>
        /// <param name="locI">The element index (advanced in place).</param>
        /// <param name="locLoVal">The scalar lower bound (scalar kind), or null.</param>
        /// <param name="locHiVal">The scalar upper bound (scalar kind), or null.</param>
        /// <param name="locEnd">The exclusive end, or null for the kernel's size argument.</param>
        private static void EmitScalarLoop(
            ILGenerator il, NPTypeCode dtype, ClipMode mode, ClipBoundsKind kind, int sz,
            LocalBuilder locI, LocalBuilder locLoVal, LocalBuilder locHiVal, LocalBuilder locEnd = null)
        {
            var clrType = GetClrType(dtype);
            bool needLo = mode != ClipMode.MaxOnly;
            bool needHi = mode != ClipMode.MinOnly;
            // NumPy's const fast path for float32/float64 scalar bounds (see UsesConstSelect): the tail must
            // agree with the SIMD body element for element, signed zeros included.
            bool constSelect = UsesConstSelect(dtype, mode, kind);

            // Reusable temp for the in-flight value
            var locVal = il.DeclareLocal(clrType);
            var locByteOff = il.DeclareLocal(typeof(long));

            var lblLoop = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.MarkLabel(lblLoop);
            // if (i >= end) break   (end = locEnd, or the size)
            il.Emit(OpCodes.Ldloc, locI);
            if (locEnd != null)
                il.Emit(OpCodes.Ldloc, locEnd);
            else
                il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblEnd);

            // byteOff = i * sz
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)sz);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locByteOff);

            // val = *(T*)(src + byteOff)
            EmitOffsetAddrFromLocal(il, 0, locByteOff);
            EmitLoadIndirect(il, dtype);
            il.Emit(OpCodes.Stloc, locVal);

            if (needLo)
            {
                il.Emit(OpCodes.Ldloc, locVal);
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locLoVal);
                else
                {
                    EmitOffsetAddrFromLocal(il, 3, locByteOff);
                    EmitLoadIndirect(il, dtype);
                }
                if (constSelect)
                    il.EmitCall(OpCodes.Call, GetHelper(dtype == NPTypeCode.Single ? nameof(FloatClipLowSelect) : nameof(DoubleClipLowSelect)), null);
                else
                    EmitScalarClamp(il, dtype, isMax: true);      // val = Max(val, lo)
                il.Emit(OpCodes.Stloc, locVal);
            }

            if (needHi)
            {
                il.Emit(OpCodes.Ldloc, locVal);
                if (kind == ClipBoundsKind.Scalar)
                    il.Emit(OpCodes.Ldloc, locHiVal);
                else
                {
                    EmitOffsetAddrFromLocal(il, 4, locByteOff);
                    EmitLoadIndirect(il, dtype);
                }
                if (constSelect)
                    il.EmitCall(OpCodes.Call, GetHelper(dtype == NPTypeCode.Single ? nameof(FloatClipHighSelect) : nameof(DoubleClipHighSelect)), null);
                else
                    EmitScalarClamp(il, dtype, isMax: false);     // val = Min(val, hi)
                il.Emit(OpCodes.Stloc, locVal);
            }

            // *(T*)(dst + byteOff) = val
            EmitOffsetAddrFromLocal(il, 1, locByteOff);
            il.Emit(OpCodes.Ldloc, locVal);
            EmitStoreIndirect(il, dtype);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblEnd);
        }

        // Emits scalar Max/Min consuming two T values on the stack, producing
        // one T. Uses Math.Max/Min where available (covers byte..ulong, float,
        // double, decimal), falls back to small static helpers for Half /
        // Complex (NaN/lex semantics) and to a branch-based emit for Char.
        //
        // nanIgnore selects the float/double/half/complex NaN policy:
        //   false (default) — NaN-PROPAGATING (np.clip / np.maximum / np.minimum):
        //                     a NaN operand wins, first-operand-NaN preferred.
        //   true            — NaN-IGNORING (np.fmax / np.fmin): the non-NaN operand
        //                     wins; both-NaN returns the second. Integer / decimal /
        //                     char have no NaN domain, so the policy is irrelevant there.
        private static void EmitScalarClamp(ILGenerator il, NPTypeCode dtype, bool isMax, bool nanIgnore = false)
        {
            var clrType = GetClrType(dtype);

            if (dtype == NPTypeCode.Half)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(HalfMaxNum) : nameof(HalfMinNum))
                    : (isMax ? nameof(HalfMaxNaN) : nameof(HalfMinNaN))), null);
                return;
            }

            if (dtype == NPTypeCode.Complex)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(ComplexMaxNum) : nameof(ComplexMinNum))
                    : (isMax ? nameof(ComplexMaxNaN) : nameof(ComplexMinNaN))), null);
                return;
            }

            // float32/float64: NaN-propagating, signed-zero-tie-to-second (matches the SIMD body
            // and NumPy). Math.Max would resolve the +0/-0 tie to +0 and diverge in the scalar tail.
            if (dtype == NPTypeCode.Single)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(FloatMaxNum) : nameof(FloatMinNum))
                    : (isMax ? nameof(FloatMaxNaN) : nameof(FloatMinNaN))), null);
                return;
            }
            if (dtype == NPTypeCode.Double)
            {
                il.EmitCall(OpCodes.Call, GetHelper(nanIgnore
                    ? (isMax ? nameof(DoubleMaxNum) : nameof(DoubleMinNum))
                    : (isMax ? nameof(DoubleMaxNaN) : nameof(DoubleMinNaN))), null);
                return;
            }

            // Math.Max(T, T) / Math.Min(T, T) for all sized numerics + decimal.
            // ScalarMethodCache.Get throws on missing — fall back to the manual select below
            // for the one type that's not covered (Char has no Math.Max overload).
            MethodInfo mathMethod = null;
            try
            {
                mathMethod = ScalarMethodCache.Get(typeof(Math),
                    isMax ? nameof(Math.Max) : nameof(Math.Min), clrType, clrType);
            }
            catch (MissingMethodException) { /* fall through to manual select */ }

            if (mathMethod != null)
            {
                il.EmitCall(OpCodes.Call, mathMethod, null);
                return;
            }

            // Char has no Math.Max overload — emit a manual select.
            // Stack: [a, b]. Result Max: a if a>=b else b ; Min: a if a<=b else b.
            var locA = il.DeclareLocal(clrType);
            var locB = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Stloc, locB);
            il.Emit(OpCodes.Stloc, locA);
            var lblPickB = il.DefineLabel();
            var lblDone  = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Ldloc, locB);
            // for Max: jump to PickB when a < b; else fall through and push a
            // for Min: jump to PickB when a > b; else fall through and push a
            il.Emit(isMax ? OpCodes.Blt : OpCodes.Bgt, lblPickB);
            il.Emit(OpCodes.Ldloc, locA);
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblPickB);
            il.Emit(OpCodes.Ldloc, locB);
            il.MarkLabel(lblDone);
        }

        #endregion

        #region Per-Element Helpers for Non-SIMD Types (called from generated IL)

        // Half: comparison ops work natively in .NET 7+, but Math.Max(Half,Half)
        // doesn't exist as a single-precision-aware overload. NumPy semantics:
        // NaN propagates — if either operand is NaN, result is NaN.
        //
        // Signed-zero tie-break: NumPy's float16 maximum/minimum return the FIRST
        // operand when the two compare equal (so maximum(+0,-0)=+0, maximum(-0,+0)=-0,
        // and likewise for minimum). `a` here is the first operand (clip's src), `b`
        // the second (the bound), so the `>=` / `<=` tie goes to `a` — matching NumPy.
        // (+0 == -0 numerically, so this only changes the result's sign bit on a zero
        // tie; every non-zero equal pair is bit-identical and unaffected.)
        //
        // These per-element helpers run once per element inside the IL kernel's inner loop,
        // so they carry AggressiveInlining (inline into the kernel where the JIT can) plus
        // AggressiveOptimization (full tier-1 codegen from the first call when they're
        // invoked standalone via the kernel's Call, skipping the tier-0 hot-path penalty).
        // NaN-PROPAGATING: the NaN OPERAND is returned VERBATIM (payload + sign preserved),
        // in1 preferred when both are NaN — NumPy's (ge(in1,in2) || isnan(in1)) ? in1 : in2
        // (probed 2.4.2: maximum([nanA],[1]) = nanA's exact bits, maximum([-nan],[nanA]) = -nan).
        // Returning canonical Half.NaN here — as this once did — silently rewrote the payload
        // on every emitted-IL route (mixed-dtype keys, SimdChunk, np.clip's Half clamp).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMaxNaN(Half a, Half b)
        {
            if (Half.IsNaN(a)) return a;
            if (Half.IsNaN(b)) return b;
            return a >= b ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMinNaN(Half a, Half b)
        {
            if (Half.IsNaN(a)) return a;
            if (Half.IsNaN(b)) return b;
            return a <= b ? a : b;
        }

        // Half NaN-IGNORING max/min (np.fmax / np.fmin): the non-NaN operand wins.
        // Tie convention follows HalfMaxNaN/HalfMinNaN (first operand on an equal tie).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMaxNum(Half a, Half b)
        {
            if (Half.IsNaN(b)) return a;   // b-first: both-NaN -> a (first), per the array ufunc loop
            if (Half.IsNaN(a)) return b;
            return a >= b ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfMinNum(Half a, Half b)
        {
            if (Half.IsNaN(b)) return a;
            if (Half.IsNaN(a)) return b;
            return a <= b ? a : b;
        }

        // float32/float64 max/min matching the NaN-propagating SIMD path (EmitVectorMinOrMax)
        // and NumPy maximum/minimum/clip. NaN in the first operand propagates; otherwise the
        // STRICT comparison returns the SECOND operand on a tie — so the signed-zero tie resolves
        // like hardware MAXPS/MINPD (f32/f64 maximum(+0,-0) = -0), the OPPOSITE of float16 (which
        // ties to the first operand, hence the separate Half helpers above) and of Math.Max (+0).
        // Every scalar float32/float64 maximum/minimum goes through these, via EmitScalarClamp: the flat clip
        // kernel's scalar loop (peel and tail), the iterator route's general clip body
        // (DirectILKernelGenerator.Clip.InnerLoop.cs) and the np.maximum / np.minimum scalar operation
        // (EmitScalarOperation). So every clip route and the two ufuncs agree bit for bit on signed zeros and NaN.

        /// <summary>
        /// float32 <c>np.maximum</c> of one pair: a NaN in either operand propagates (the first one's when both are NaN),
        /// and a tie (±0) returns the SECOND operand, as the hardware MAXPS does.
        /// </summary>
        /// <param name="a">The first operand (the element, or the running value).</param>
        /// <param name="b">The second operand (the bound).</param>
        /// <returns><paramref name="a"/> when it is NaN or strictly greater, else <paramref name="b"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMaxNaN(float a, float b) => float.IsNaN(a) ? a : (a > b ? a : b);

        /// <summary>
        /// float32 <c>np.minimum</c> of one pair: a NaN in either operand propagates (the first one's when both are NaN),
        /// and a tie (±0) returns the SECOND operand, as the hardware MINPS does.
        /// </summary>
        /// <param name="a">The first operand (the element, or the running value).</param>
        /// <param name="b">The second operand (the bound).</param>
        /// <returns><paramref name="a"/> when it is NaN or strictly less, else <paramref name="b"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMinNaN(float a, float b) => float.IsNaN(a) ? a : (a < b ? a : b);

        /// <summary>
        /// float64 <c>np.maximum</c> of one pair: a NaN in either operand propagates (the first one's when both are NaN),
        /// and a tie (±0) returns the SECOND operand, as the hardware MAXPD does.
        /// </summary>
        /// <param name="a">The first operand (the element, or the running value).</param>
        /// <param name="b">The second operand (the bound).</param>
        /// <returns><paramref name="a"/> when it is NaN or strictly greater, else <paramref name="b"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMaxNaN(double a, double b) => double.IsNaN(a) ? a : (a > b ? a : b);

        /// <summary>
        /// float64 <c>np.minimum</c> of one pair: a NaN in either operand propagates (the first one's when both are NaN),
        /// and a tie (±0) returns the SECOND operand, as the hardware MINPD does.
        /// </summary>
        /// <param name="a">The first operand (the element, or the running value).</param>
        /// <param name="b">The second operand (the bound).</param>
        /// <returns><paramref name="a"/> when it is NaN or strictly less, else <paramref name="b"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMinNaN(double a, double b) => double.IsNaN(a) ? a : (a < b ? a : b);

        // float32/float64 clip with two SCALAR bounds — NumPy's const fast path (clip.cpp
        // _npy_clip_const_minmax_, floating_point_tag): x < lo ? lo : x, then x > hi ? hi : x. A tie keeps
        // the ELEMENT (clip(-0.0, 0.0, 1.0) = -0.0, where Min(Max()) gives +0.0), a NaN element fails both
        // compares and passes through with its own bits, and min > max still yields max. Only the flat kernel's
        // scalar loop (peel and tail) calls these, so its scalar elements agree with its SIMD body. NaN BOUNDS never
        // reach them: the flat route fills a call whose scalar bound is NaN itself (FillWithNaNBound) before any
        // kernel runs. The iterator route uses FloatClipConst / DoubleClipConst instead, which carry NumPy's
        // NaN-bound rule (a broadcast bound array can hold NaN in some rows only).

        /// <summary>The lower half of NumPy's float32 const clip: the bound replaces the element only when the element is strictly below it.</summary>
        /// <param name="x">The element.</param>
        /// <param name="lo">The lower bound (never NaN here).</param>
        /// <returns><paramref name="lo"/> when <c>x &lt; lo</c>, else <paramref name="x"/> (a tie or a NaN element keeps the element).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatClipLowSelect(float x, float lo) => x < lo ? lo : x;

        /// <summary>The upper half of NumPy's float32 const clip: the bound replaces the element only when the element is strictly above it.</summary>
        /// <param name="x">The element (after the lower half).</param>
        /// <param name="hi">The upper bound (never NaN here).</param>
        /// <returns><paramref name="hi"/> when <c>x &gt; hi</c>, else <paramref name="x"/> (a tie or a NaN element keeps the element).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatClipHighSelect(float x, float hi) => x > hi ? hi : x;

        /// <summary>The lower half of NumPy's float64 const clip: the bound replaces the element only when the element is strictly below it.</summary>
        /// <param name="x">The element.</param>
        /// <param name="lo">The lower bound (never NaN here).</param>
        /// <returns><paramref name="lo"/> when <c>x &lt; lo</c>, else <paramref name="x"/> (a tie or a NaN element keeps the element).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleClipLowSelect(double x, double lo) => x < lo ? lo : x;

        /// <summary>The upper half of NumPy's float64 const clip: the bound replaces the element only when the element is strictly above it.</summary>
        /// <param name="x">The element (after the lower half).</param>
        /// <param name="hi">The upper bound (never NaN here).</param>
        /// <returns><paramref name="hi"/> when <c>x &gt; hi</c>, else <paramref name="x"/> (a tie or a NaN element keeps the element).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleClipHighSelect(double x, double hi) => x > hi ? hi : x;

        // float32/float64 NaN-IGNORING max/min — np.fmax / np.fmin ARRAY semantics (probed
        // 2.4.2): a NaN operand is skipped and the OTHER operand returned. The NaN checks are
        // ordered B-FIRST so both-NaN returns the FIRST operand (a): fmax(nanA, nanB) -> nanA,
        // matching the ufunc loop (NumPy's SCALAR fmax() returns the second — the array loop
        // does not). For two finite operands the strict comparison returns the SECOND on a tie
        // — identical to maximum/minimum and the hardware MAXPS/MINPS the SIMD body uses, so
        // np.fmax([+0],[-0]) -> -0 like np.maximum. Used by EmitScalarClamp(nanIgnore:true) and
        // the strided fmax/fmin loop.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMaxNum(float a, float b) => float.IsNaN(b) ? a : (float.IsNaN(a) ? b : (a > b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static float FloatMinNum(float a, float b) => float.IsNaN(b) ? a : (float.IsNaN(a) ? b : (a < b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMaxNum(double a, double b) => double.IsNaN(b) ? a : (double.IsNaN(a) ? b : (a > b ? a : b));
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static double DoubleMinNum(double a, double b) => double.IsNaN(b) ? a : (double.IsNaN(a) ? b : (a < b ? a : b));

        // Complex: lex ordering on (real, imag). NaN propagation: if either
        // operand contains a NaN component, that operand wins (first-encountered
        // for the Max-then-Min sequence — matches NumPy clip output bit-for-bit).
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static bool ComplexIsNaN(Complex z) => double.IsNaN(z.Real) || double.IsNaN(z.Imaginary);
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMaxNaN(Complex a, Complex b)
        {
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            if (a.Real > b.Real) return a;
            if (a.Real < b.Real) return b;
            return a.Imaginary > b.Imaginary ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMinNaN(Complex a, Complex b)
        {
            if (ComplexIsNaN(a)) return a;
            if (ComplexIsNaN(b)) return b;
            if (a.Real > b.Real) return b;
            if (a.Real < b.Real) return a;
            return a.Imaginary > b.Imaginary ? b : a;
        }

        // Complex NaN-IGNORING max/min (np.fmax / np.fmin): the non-NaN operand wins; lex
        // ordering on (real, imag) otherwise. Mirrors ComplexMaxNaN/ComplexMinNaN with the
        // NaN branches returning the OTHER operand instead of the NaN one.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMaxNum(Complex a, Complex b)
        {
            if (ComplexIsNaN(b)) return a;   // b-first: both-NaN -> a (first)
            if (ComplexIsNaN(a)) return b;
            if (a.Real > b.Real) return a;
            if (a.Real < b.Real) return b;
            return a.Imaginary > b.Imaginary ? a : b;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Complex ComplexMinNum(Complex a, Complex b)
        {
            if (ComplexIsNaN(b)) return a;
            if (ComplexIsNaN(a)) return b;
            if (a.Real > b.Real) return b;
            if (a.Real < b.Real) return a;
            return a.Imaginary > b.Imaginary ? b : a;
        }

        #endregion
    }
}
