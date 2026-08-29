using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Unary.Round.Half.cs — float16 floor / ceil / trunc /
// rint (bit-level, conversion-free)
// =============================================================================
//
// Rounding an f16 to an integer is defined entirely on the bit pattern — every
// result is exactly representable (all f16 ≥ 2^10 are already integers), so
// NumPy's widen→floorf/ceilf/truncf/roundevenf→narrow pipeline
// (loops_half.dispatch.c.src's scalar fallback; the AVX-512-FP16 path needs
// Sapphire Rapids and is absent from the win-amd64 wheel — measured 434-724 µs
// per 100K, rint the slowest f16 op of all) collapses to exponent-based
// mantissa masking on raw ushort lanes. Third conversion-free f16 family after
// sign/negate/abs and the comparison/min-max group.
//
// Per lane (probed exhaustively on 2.4.2):
//   NaN (either sign)      → bits | 0x0200 (the arithmetic quiets sNaN; qNaN identity)
//   |x| ≥ 1024 (mag≥0x6400, inf incl.) → identity
//   |x| < 1  (mag<0x3c00)  → trunc: ±0 · floor: neg&&mag≠0 ? -1 : ±0
//                            ceil: pos&&mag≠0 ? +1 : ±0 (ceil(-0.2) = -0!)
//                            rint: mag > 0x3800 ? ±1 : ±0 (±0.5 ties to ±0)
//   1 ≤ |x| < 1024         → fracBits = 25 - exp, mask = (1<<fracBits)-1:
//                            trunc: bits & ~mask
//                            floor: + (mask+1) when negative with a fraction
//                            ceil:  + (mask+1) when positive with a fraction
//                            rint:  (bits + (mask>>1) + intLSB) & ~mask  (RTNE)
//     — the +step carries from mantissa into exponent exactly like the value
//       +1 (floor(-1.001) = -2 crosses 0xbc01→0xc000), and can never reach the
//       sign bit (results ≤ 0x6400 < 0x8000).
//
// The per-lane fracBits demand VARIABLE shifts, which AVX2 only has at 32-bit
// lanes (VPSLLVD/VPSRLVD) — so the kernel widens 8 ushorts to one i32 vector
// per step. Serves UnaryOp.Floor/Ceil/Truncate/Round for the contiguous
// (Half→Half) unary keys; strided keeps the scalar path.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        internal static unsafe void HalfFloorContiguous(void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
            => HalfRoundCore((ushort*)input, (ushort*)output, totalSize, UnaryOp.Floor);
        internal static unsafe void HalfCeilContiguous(void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
            => HalfRoundCore((ushort*)input, (ushort*)output, totalSize, UnaryOp.Ceil);
        internal static unsafe void HalfTruncContiguous(void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
            => HalfRoundCore((ushort*)input, (ushort*)output, totalSize, UnaryOp.Truncate);
        internal static unsafe void HalfRintContiguous(void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
            => HalfRoundCore((ushort*)input, (ushort*)output, totalSize, UnaryOp.Round);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfRoundCore(ushort* src, ushort* dst, long n, UnaryOp op)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create(0x7FFF);
                var m8000 = Vector256.Create(0x8000);
                var vinf = Vector256.Create(0x7C00);
                var vbig = Vector256.Create(0x6400);
                var vone16 = Vector256.Create(0x3C00);
                var vhalf16 = Vector256.Create(0x3800);
                var v25 = Vector256.Create(25);
                var one = Vector256.Create(1);
                var q = Vector256.Create(0x0200);
                for (; i + 8 <= n; i += 8)
                {
                    var h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i));
                    var mag = Avx2.And(h, m7fff);
                    var sign = Avx2.And(h, m8000);
                    var isNaN = Avx2.CompareGreaterThan(mag, vinf);
                    var isBig = Avx2.CompareGreaterThan(mag, Avx2.Subtract(vbig, one)); // mag >= 0x6400
                    var isSmall = Avx2.CompareGreaterThan(vone16, mag);                 // mag < 0x3c00
                    var isNeg = Avx2.CompareGreaterThan(sign, Vector256<int>.Zero);     // sign bit set
                    var magNZ = Avx2.CompareGreaterThan(mag, Vector256<int>.Zero);

                    // mid range: fracBits = 25 - (mag >> 10) ∈ [1..10] (garbage outside, blended away)
                    var fb = Avx2.Subtract(v25, Avx2.ShiftRightLogical(mag, 10));
                    var mask = Avx2.Subtract(Avx2.ShiftLeftLogicalVariable(one, fb.AsUInt32()), one);
                    var notMask = Avx2.AndNot(mask, Vector256<int>.AllBitsSet).AsInt32();
                    var tr = Avx2.And(h, notMask);
                    var hasFrac = Avx2.CompareGreaterThan(Avx2.And(h, mask), Vector256<int>.Zero);
                    var step = Avx2.Add(mask, one);

                    Vector256<int> mid, small;
                    switch (op)
                    {
                        case UnaryOp.Floor:
                            mid = Avx2.Add(tr, Avx2.And(step, Avx2.And(isNeg, hasFrac)));
                            small = Avx2.Or(sign, Avx2.And(vone16, Avx2.And(isNeg, magNZ)));
                            break;
                        case UnaryOp.Ceil:
                            mid = Avx2.Add(tr, Avx2.And(step, Avx2.AndNot(isNeg, hasFrac)));
                            small = Avx2.Or(sign, Avx2.And(vone16, Avx2.AndNot(isNeg, magNZ)));
                            break;
                        case UnaryOp.Truncate:
                            mid = tr;
                            small = sign;
                            break;
                        default: // UnaryOp.Round — RTNE
                            var intLsb = Avx2.And(Avx2.ShiftRightLogicalVariable(h.AsUInt32(), fb.AsUInt32()).AsInt32(), one);
                            mid = Avx2.And(Avx2.Add(Avx2.Add(h, Avx2.ShiftRightLogical(mask, 1)), intLsb), notMask);
                            small = Avx2.Or(sign, Avx2.And(vone16, Avx2.CompareGreaterThan(mag, vhalf16)));
                            break;
                    }

                    var res = Avx2.BlendVariable(mid, small, isSmall);
                    res = Avx2.BlendVariable(res, h, isBig);
                    res = Avx2.BlendVariable(res, Avx2.Or(h, q), isNaN);
                    Sse2.Store(dst + i, Sse41.PackUnsignedSaturate(res.GetLower(), res.GetUpper()));
                }
            }
            for (; i < n; i++)
                dst[i] = HalfRoundBits(src[i], op);
        }

        /// <summary>Scalar lane — the same bit algorithm (used by the tail; strided callers welcome).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ushort HalfRoundBits(ushort bits, UnaryOp op)
        {
            int mag = bits & 0x7FFF;
            if (mag > 0x7C00) return (ushort)(bits | 0x0200);            // NaN → quieted
            if (mag >= 0x6400) return bits;                              // |x| ≥ 1024 (or inf)
            int sign = bits & 0x8000;
            if (mag < 0x3C00)                                            // |x| < 1
            {
                switch (op)
                {
                    case UnaryOp.Floor: return (ushort)(sign != 0 && mag != 0 ? 0xBC00 : sign);
                    case UnaryOp.Ceil: return (ushort)(sign == 0 && mag != 0 ? 0x3C00 : sign);
                    case UnaryOp.Truncate: return (ushort)sign;
                    default: return (ushort)(mag > 0x3800 ? sign | 0x3C00 : sign);
                }
            }
            int fracBits = 25 - (mag >> 10);
            int mask = (1 << fracBits) - 1;
            int tr = bits & ~mask;
            switch (op)
            {
                case UnaryOp.Floor: return (ushort)(sign != 0 && (bits & mask) != 0 ? tr + mask + 1 : tr);
                case UnaryOp.Ceil: return (ushort)(sign == 0 && (bits & mask) != 0 ? tr + mask + 1 : tr);
                case UnaryOp.Truncate: return (ushort)tr;
                default: return (ushort)((bits + (mask >> 1) + ((bits >> fracBits) & 1)) & ~mask);
            }
        }
    }
}
