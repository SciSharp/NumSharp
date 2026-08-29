using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Binary.MinMax.Half.cs — float16 elementwise
// maximum / minimum / fmax / fmin (bit-level, conversion-free)
// =============================================================================
//
// The elementwise min/max family is pure COMPARISON, so — like the flat reduce in
// Reduction.MinMax.Half.cs — it runs entirely on the raw f16 bit patterns with ZERO
// float conversions, using the same proven order map
//     key = bits ^ (0x8000 | (bits >>ARITH 15))
// (strictly order-preserving over every non-NaN pattern). NumPy's HALF_maximum /
// HALF_minimum / HALF_fmax / HALF_fmin loops are scalar branchy bit-compares
// (npy_half_ge/le — no x86 SIMD HALF loop exists), measured 512/490 µs per 100K on
// 2.4.2, so this family beats NumPy outright rather than chasing F16C.
//
// NumPy semantics reproduced exactly (all probed live on 2.4.2):
//   maximum: (ge(in1,in2) || isnan(in1)) ? in1 : in2 — NaN PROPAGATES, the NaN
//            OPERAND is returned VERBATIM (payload + sign), in1 preferred when both
//            are NaN; ±0 ties return in1 (ge/le call -0 == +0).
//   fmax:    (ge(in1,in2) || isnan(in2)) ? in1 : in2 — NaN is IGNORED (the non-NaN
//            operand wins), both-NaN returns in1; ±0 ties return in1.
// One per-lane select covers all four ops:
//     takeIn1 = nanPick | (~(nan1|nan2) & (cmpKeys | bothZero))
// where cmpKeys is unsigned key >= / <= (via pmaxuw/pminuw + compare-equal),
// bothZero repairs the one spot where the key map disagrees with ge/le (-0 vs +0),
// and nanPick = nan1 (maximum/minimum) or nan2 (fmax/fmin). isMax / nanIgnore are
// LOOP-INVARIANT (perfectly predicted) — the data-dependent select itself is
// branchless, so the in-loop-branch trap that cost the argmax kernels 2.26× does
// not apply.
//
// Serves the (Half,Half)→Half MixedTypeKernel keys for SimdFull and the two
// scalar-broadcast paths (the scalar operand's key/NaN masks hoist out of the
// loop entirely); SimdChunk/General and mixed-dtype keys keep the emitted IL,
// whose scalar clamp helpers (HalfMaxNaN/HalfMinNaN) were fixed alongside to
// return the NaN operand verbatim instead of canonical NaN.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Returns the static bit-level kernel for a (Half,Half)→Half elementwise min/max
        /// key, or null when the (op, path) pair is not served (falls to emitted IL).
        /// </summary>
        internal static unsafe MixedTypeKernel TryGetHalfMinMaxKernel(BinaryOp op, ExecutionPath path)
        {
            switch (path)
            {
                case ExecutionPath.SimdFull:
                    switch (op)
                    {
                        case BinaryOp.Maximum: return HalfMaximumFull;
                        case BinaryOp.Minimum: return HalfMinimumFull;
                        case BinaryOp.FMax: return HalfFMaxFull;
                        case BinaryOp.FMin: return HalfFMinFull;
                    }
                    break;
                case ExecutionPath.SimdScalarRight:
                    switch (op)
                    {
                        case BinaryOp.Maximum: return HalfMaximumScalarRight;
                        case BinaryOp.Minimum: return HalfMinimumScalarRight;
                        case BinaryOp.FMax: return HalfFMaxScalarRight;
                        case BinaryOp.FMin: return HalfFMinScalarRight;
                    }
                    break;
                case ExecutionPath.SimdScalarLeft:
                    switch (op)
                    {
                        case BinaryOp.Maximum: return HalfMaximumScalarLeft;
                        case BinaryOp.Minimum: return HalfMinimumScalarLeft;
                        case BinaryOp.FMax: return HalfFMaxScalarLeft;
                        case BinaryOp.FMin: return HalfFMinScalarLeft;
                    }
                    break;
            }
            return null;
        }

        // ── MixedTypeKernel-signature wrappers (4 ops × 3 paths) ─────────────────

        internal static unsafe void HalfMaximumFull(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectFull((ushort*)l, (ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: false);
        internal static unsafe void HalfMinimumFull(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectFull((ushort*)l, (ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: false);
        internal static unsafe void HalfFMaxFull(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectFull((ushort*)l, (ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: true);
        internal static unsafe void HalfFMinFull(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectFull((ushort*)l, (ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: true);

        internal static unsafe void HalfMaximumScalarRight(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarRight((ushort*)l, *(ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: false);
        internal static unsafe void HalfMinimumScalarRight(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarRight((ushort*)l, *(ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: false);
        internal static unsafe void HalfFMaxScalarRight(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarRight((ushort*)l, *(ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: true);
        internal static unsafe void HalfFMinScalarRight(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarRight((ushort*)l, *(ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: true);

        internal static unsafe void HalfMaximumScalarLeft(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarLeft(*(ushort*)l, (ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: false);
        internal static unsafe void HalfMinimumScalarLeft(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarLeft(*(ushort*)l, (ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: false);
        internal static unsafe void HalfFMaxScalarLeft(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarLeft(*(ushort*)l, (ushort*)r, (ushort*)o, n, isMax: true, nanIgnore: true);
        internal static unsafe void HalfFMinScalarLeft(void* l, void* r, void* o, long* ls, long* rs, long* sh, int nd, long n)
            => HalfPairSelectScalarLeft(*(ushort*)l, (ushort*)r, (ushort*)o, n, isMax: false, nanIgnore: true);

        // ── cores ────────────────────────────────────────────────────────────────

        /// <summary>Both operands streamed (SimdFull: contiguous, same length).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfPairSelectFull(ushort* pa, ushort* pb, ushort* pr, long n, bool isMax, bool nanIgnore)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                for (; i + 16 <= n; i += 16)
                {
                    var a = Avx.LoadVector256((short*)(pa + i));
                    var b = Avx.LoadVector256((short*)(pb + i));
                    var magA = Avx2.And(a, m7fff);
                    var magB = Avx2.And(b, m7fff);
                    var nanA = Avx2.CompareGreaterThan(magA, vinf);      // mags ≤ 0x7fff: sign-safe
                    var nanB = Avx2.CompareGreaterThan(magB, vinf);
                    var ka = HalfKeyV(a, m8000);
                    var kb = HalfKeyV(b, m8000);
                    var cmp = isMax
                        ? Avx2.CompareEqual(Avx2.Max(ka, kb), ka)        // ka >= kb (unsigned)
                        : Avx2.CompareEqual(Avx2.Min(ka, kb), ka);       // ka <= kb (unsigned)
                    var bothZero = Avx2.CompareEqual(Avx2.Or(magA, magB), Vector256<short>.Zero);
                    var anyNaN = Avx2.Or(nanA, nanB);
                    var pick = nanIgnore ? nanB : nanA;
                    var takeA = Avx2.Or(pick, Avx2.AndNot(anyNaN, Avx2.Or(cmp.AsInt16(), bothZero)));
                    Avx.Store((short*)(pr + i), Avx2.BlendVariable(b.AsByte(), a.AsByte(), takeA.AsByte()).AsInt16());
                }
            }
            for (; i < n; i++)
                pr[i] = HalfPairSelectBits(pa[i], pb[i], isMax, nanIgnore);
        }

        /// <summary>
        /// lhs streamed, rhs one broadcast scalar (in2). The scalar's magnitude / NaN mask /
        /// order key are loop-invariant and hoist entirely.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfPairSelectScalarRight(ushort* pa, ushort b, ushort* pr, long n, bool isMax, bool nanIgnore)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var vb = Vector256.Create(unchecked((short)b));
                var magB = Vector256.Create((short)(b & 0x7FFF));
                var nanB = (b & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var kb = Vector256.Create(HalfKeyScalar(b));
                for (; i + 16 <= n; i += 16)
                {
                    var a = Avx.LoadVector256((short*)(pa + i));
                    var magA = Avx2.And(a, m7fff);
                    var nanA = Avx2.CompareGreaterThan(magA, vinf);
                    var ka = HalfKeyV(a, m8000);
                    var cmp = isMax
                        ? Avx2.CompareEqual(Avx2.Max(ka, kb), ka)
                        : Avx2.CompareEqual(Avx2.Min(ka, kb), ka);
                    var bothZero = Avx2.CompareEqual(Avx2.Or(magA, magB), Vector256<short>.Zero);
                    var anyNaN = Avx2.Or(nanA, nanB);
                    var pick = nanIgnore ? nanB : nanA;
                    var takeA = Avx2.Or(pick, Avx2.AndNot(anyNaN, Avx2.Or(cmp.AsInt16(), bothZero)));
                    Avx.Store((short*)(pr + i), Avx2.BlendVariable(vb.AsByte(), a.AsByte(), takeA.AsByte()).AsInt16());
                }
            }
            for (; i < n; i++)
                pr[i] = HalfPairSelectBits(pa[i], b, isMax, nanIgnore);
        }

        /// <summary>
        /// lhs one broadcast scalar (in1 — operand PRIORITY is not symmetric, so this is not
        /// the right-scalar core with swapped pointers), rhs streamed.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfPairSelectScalarLeft(ushort a, ushort* pb, ushort* pr, long n, bool isMax, bool nanIgnore)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var va = Vector256.Create(unchecked((short)a));
                var magA = Vector256.Create((short)(a & 0x7FFF));
                var nanA = (a & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var ka = Vector256.Create(HalfKeyScalar(a));
                for (; i + 16 <= n; i += 16)
                {
                    var b = Avx.LoadVector256((short*)(pb + i));
                    var magB = Avx2.And(b, m7fff);
                    var nanB = Avx2.CompareGreaterThan(magB, vinf);
                    var kb = HalfKeyV(b, m8000);
                    var cmp = isMax
                        ? Avx2.CompareEqual(Avx2.Max(ka, kb), ka)
                        : Avx2.CompareEqual(Avx2.Min(ka, kb), ka);
                    var bothZero = Avx2.CompareEqual(Avx2.Or(magA, magB), Vector256<short>.Zero);
                    var anyNaN = Avx2.Or(nanA, nanB);
                    var pick = nanIgnore ? nanB : nanA;
                    var takeA = Avx2.Or(pick, Avx2.AndNot(anyNaN, Avx2.Or(cmp.AsInt16(), bothZero)));
                    Avx.Store((short*)(pr + i), Avx2.BlendVariable(b.AsByte(), va.AsByte(), takeA.AsByte()).AsInt16());
                }
            }
            for (; i < n; i++)
                pr[i] = HalfPairSelectBits(a, pb[i], isMax, nanIgnore);
        }

        // ── np.clip (Half) — the same pair-select primitive chained max-then-min ────
        //
        // NumPy's clip.cpp: val = _NPY_MIN(_NPY_MAX(x, lo), hi) where
        // _NPY_MAX(a,b) = isnan(a) || ge(a,b) ? a : b (guarded ge, in1 preference) —
        // exactly HalfPairSelectBits(val, bound, isMax, nanIgnore:false) per stage. So a
        // NaN x passes through VERBATIM, a NaN bound REPLACES the value from its stage on
        // (probed 2.4.2: clip(5, nan, 1) = the lo-bound NaN bits), lo>hi resolves to hi
        // (min applied last), and ±0 stage-ties keep the running value.

        /// <summary>
        /// Returns the static bit-level clip kernel for a Half (mode, kind) pair — always
        /// available (every combination is served). Cached by the caller's clip-kernel cache.
        /// </summary>
        internal static unsafe ClipKernel GetHalfClipKernel(ClipMode mode, ClipBoundsKind kind)
        {
            if (kind == ClipBoundsKind.Scalar)
                return (src, dst, n, lo, hi) => HalfClipScalarBounds(
                    (ushort*)src, (ushort*)dst, n,
                    mode != ClipMode.MaxOnly ? *(ushort*)lo : (ushort)0,
                    mode != ClipMode.MinOnly ? *(ushort*)hi : (ushort)0,
                    mode);
            return (src, dst, n, lo, hi) => HalfClipArrayBounds(
                (ushort*)src, (ushort*)dst, n, (ushort*)lo, (ushort*)hi, mode);
        }

        /// <summary>One clamp stage on a streamed lane vector vs a hoisted scalar bound.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<short> HalfClipStageScalar(
            Vector256<short> v, Vector256<short> vb, Vector256<short> magB, Vector256<short> nanB,
            Vector256<ushort> kb, Vector256<short> m7fff, Vector256<short> vinf, Vector256<short> m8000, bool isMax)
        {
            var magV = Avx2.And(v, m7fff);
            var nanV = Avx2.CompareGreaterThan(magV, vinf);
            var kv = HalfKeyV(v, m8000);
            var cmp = isMax
                ? Avx2.CompareEqual(Avx2.Max(kv, kb), kv)
                : Avx2.CompareEqual(Avx2.Min(kv, kb), kv);
            var bothZero = Avx2.CompareEqual(Avx2.Or(magV, magB), Vector256<short>.Zero);
            var takeV = Avx2.Or(nanV, Avx2.AndNot(Avx2.Or(nanV, nanB), Avx2.Or(cmp.AsInt16(), bothZero)));
            return Avx2.BlendVariable(vb.AsByte(), v.AsByte(), takeV.AsByte()).AsInt16();
        }

        /// <summary>Fused clip with scalar bound(s) — both bound mask sets hoist out of the loop.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfClipScalarBounds(ushort* src, ushort* dst, long n, ushort lo, ushort hi, ClipMode mode)
        {
            bool hasLo = mode != ClipMode.MaxOnly, hasHi = mode != ClipMode.MinOnly;
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var vlo = Vector256.Create(unchecked((short)lo));
                var magLo = Vector256.Create((short)(lo & 0x7FFF));
                var nanLo = (lo & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var klo = Vector256.Create(HalfKeyScalar(lo));
                var vhi = Vector256.Create(unchecked((short)hi));
                var magHi = Vector256.Create((short)(hi & 0x7FFF));
                var nanHi = (hi & 0x7FFF) > 0x7C00 ? Vector256<short>.AllBitsSet : Vector256<short>.Zero;
                var khi = Vector256.Create(HalfKeyScalar(hi));
                for (; i + 16 <= n; i += 16)
                {
                    var v = Avx.LoadVector256((short*)(src + i));
                    if (hasLo) v = HalfClipStageScalar(v, vlo, magLo, nanLo, klo, m7fff, vinf, m8000, isMax: true);
                    if (hasHi) v = HalfClipStageScalar(v, vhi, magHi, nanHi, khi, m7fff, vinf, m8000, isMax: false);
                    Avx.Store((short*)(dst + i), v);
                }
            }
            for (; i < n; i++)
            {
                ushort v = src[i];
                if (hasLo) v = HalfPairSelectBits(v, lo, isMax: true, nanIgnore: false);
                if (hasHi) v = HalfPairSelectBits(v, hi, isMax: false, nanIgnore: false);
                dst[i] = v;
            }
        }

        /// <summary>Fused clip with per-element array bound(s) (each `n` values).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfClipArrayBounds(ushort* src, ushort* dst, long n, ushort* lo, ushort* hi, ClipMode mode)
        {
            bool hasLo = mode != ClipMode.MaxOnly, hasHi = mode != ClipMode.MinOnly;
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                for (; i + 16 <= n; i += 16)
                {
                    var v = Avx.LoadVector256((short*)(src + i));
                    if (hasLo)
                    {
                        var b = Avx.LoadVector256((short*)(lo + i));
                        var magB = Avx2.And(b, m7fff);
                        var nanB = Avx2.CompareGreaterThan(magB, vinf);
                        v = HalfClipStageScalar(v, b, magB, nanB, HalfKeyV(b, m8000), m7fff, vinf, m8000, isMax: true);
                    }
                    if (hasHi)
                    {
                        var b = Avx.LoadVector256((short*)(hi + i));
                        var magB = Avx2.And(b, m7fff);
                        var nanB = Avx2.CompareGreaterThan(magB, vinf);
                        v = HalfClipStageScalar(v, b, magB, nanB, HalfKeyV(b, m8000), m7fff, vinf, m8000, isMax: false);
                    }
                    Avx.Store((short*)(dst + i), v);
                }
            }
            for (; i < n; i++)
            {
                ushort v = src[i];
                if (hasLo) v = HalfPairSelectBits(v, lo[i], isMax: true, nanIgnore: false);
                if (hasHi) v = HalfPairSelectBits(v, hi[i], isMax: false, nanIgnore: false);
                dst[i] = v;
            }
        }

        /// <summary>
        /// Scalar lane select — the exact NumPy fold on bits, <paramref name="a"/> is in1.
        /// Used by the SIMD tails and available to any strided caller.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ushort HalfPairSelectBits(ushort a, ushort b, bool isMax, bool nanIgnore)
        {
            bool nA = (a & 0x7FFF) > 0x7C00, nB = (b & 0x7FFF) > 0x7C00;
            if (nA | nB)
                return nanIgnore ? (nB ? a : b) : (nA ? a : b);
            if (((a | b) & 0x7FFF) == 0)
                return a;                                    // ±0 tie → in1
            ushort ka = HalfKeyScalar(a), kb = HalfKeyScalar(b);
            return (isMax ? ka >= kb : ka <= kb) ? a : b;
        }
    }
}
