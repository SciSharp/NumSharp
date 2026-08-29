using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Reduction.MinMax.Half.cs — float16 flat MIN/MAX reduce
// =============================================================================
//
// np.max / np.min on float16 are pure COMPARISON reductions, and f16 ordering is
// sign-magnitude on the raw bits — so unlike every arithmetic f16 op (capped by this
// runtime's lack of vectorized F16C) the whole reduce runs on ushort lanes with ZERO
// float conversions. NumPy's own HALF_maximum/HALF_minimum are scalar branchy
// bit-compares (npy_half_ge/le via Half::LessEqual in half.hpp; loops_minmax.dispatch
// carries no x86 HALF entry), so this is the second f16 family — after
// sign/negate/abs — that genuinely BEATS NumPy instead of chasing its F16C loops.
//
// Order map (proven exhaustively: strictly order-preserving over all 63,490 non-NaN
// patterns, round-trips all 65,536, NaN keys strictly outside [key(-inf), key(+inf)]
// = [0x03ff, 0xfc00]):
//
//     key = bits ^ (0x8000 | (bits >>ARITH 15))    // pos: bits|0x8000, neg: ~bits
//
// Unsigned max/min over keys == float max/min, with two rare-case fixups that
// reproduce NumPy's fold `(ge(acc, x) || isnan(acc)) ? acc : x` exactly:
//
//   • NaN — +NaN keys sit ABOVE key(+inf) and -NaN keys BELOW key(-inf), so NaN
//     presence is decidable from the (keyMax, keyMin) pair; BOTH accumulators are
//     tracked (one extra pminuw/pmaxuw per vector — cheaper than an and+cmp+or NaN
//     mask, and the loop stays branch-free so min and max share ONE core; the
//     min/max split only happens after the loop, avoiding the in-loop-branch trap
//     that cost the argmax kernels 2.26×). Any NaN → rescan for the FIRST NaN
//     element and return it VERBATIM: NumPy's accumulator sticks at the first NaN,
//     payload AND sign preserved (probed 2.4.2: max([nanB,nanA]) = nanB,
//     max([-nan(0xfe22), nanA]) = 0xfe22).
//
//   • ±0 — npy_half_ge/le treat -0 == +0 (`(a|b) == 0x8000u`), so NumPy keeps the
//     FIRST zero seen, while the key map orders -0 < +0. A ±0 result → rescan for
//     the first zero and return its bits (probed 2.4.2: max([-0,+0]) = -0,
//     min([+0,-0]) = +0, max([-1,-0,+0,-0]) = -0).
//
// Every other extremum is a unique bit pattern per value, so the key inverse is
// exact. Contiguous flat reduce only; strided/broadcast inputs keep the NDIter
// fallback, and the axis reduce keeps its own driver.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Flat max over a contiguous float16 buffer, returned as raw bits.
        /// Bit-identical to NumPy 2.4.2's <c>maximum.reduce</c> (first-NaN verbatim,
        /// first-zero sign). <paramref name="n"/> must be ≥ 1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfMaxBitsContiguous(ushort* p, long n)
            => HalfMinMaxBitsCore(p, n, isMin: false);

        /// <summary>
        /// Flat min over a contiguous float16 buffer, returned as raw bits.
        /// Bit-identical to NumPy 2.4.2's <c>minimum.reduce</c> (first-NaN verbatim,
        /// first-zero sign). <paramref name="n"/> must be ≥ 1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfMinBitsContiguous(ushort* p, long n)
            => HalfMinMaxBitsCore(p, n, isMin: true);

        /// <summary>
        /// Shared min+max key-accumulation core. The SIMD loop is direction-free (it
        /// tracks both extremes — the min accumulator doubles as the negative-NaN
        /// detector), so <paramref name="isMin"/> is only consulted after the loop.
        /// Direct AVX2 intrinsics, not the generic Vector256.* API: portable ushort
        /// Max/GreaterThan did not lower to single instructions here (the sign kernel
        /// measured ~30× slower before switching) — pmaxuw/pminuw are exactly one
        /// instruction each.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfMinMaxBitsCore(ushort* p, long n, bool isMin)
        {
            ushort kMax = 0;                 // smallest possible key — max(0, k) == k
            ushort kMin = 0xFFFF;            // largest possible key — min(0xFFFF, k) == k
            long i = 0;

            if (Avx2.IsSupported && n >= 16)
            {
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var accMax = Vector256<ushort>.Zero;
                var accMin = Vector256<ushort>.AllBitsSet;

                // 4×-unrolled body (64 halves/iter): the four transforms are independent,
                // the two accumulate chains are 1-cycle pmaxuw/pminuw trees.
                for (; i + 64 <= n; i += 64)
                {
                    var k0 = HalfKeyV(Avx.LoadVector256((short*)(p + i)), m8000);
                    var k1 = HalfKeyV(Avx.LoadVector256((short*)(p + i + 16)), m8000);
                    var k2 = HalfKeyV(Avx.LoadVector256((short*)(p + i + 32)), m8000);
                    var k3 = HalfKeyV(Avx.LoadVector256((short*)(p + i + 48)), m8000);
                    accMax = Avx2.Max(accMax, Avx2.Max(Avx2.Max(k0, k1), Avx2.Max(k2, k3)));
                    accMin = Avx2.Min(accMin, Avx2.Min(Avx2.Min(k0, k1), Avx2.Min(k2, k3)));
                }
                for (; i + 16 <= n; i += 16)
                {
                    var k = HalfKeyV(Avx.LoadVector256((short*)(p + i)), m8000);
                    accMax = Avx2.Max(accMax, k);
                    accMin = Avx2.Min(accMin, k);
                }

                // Horizontal: PHMINPOSUW is the ushort horizontal MIN; the max reduces
                // through it on the complement (~max(k) == min(~k)).
                var min128 = Sse41.Min(accMin.GetLower(), accMin.GetUpper());
                kMin = Sse41.MinHorizontal(min128).GetElement(0);
                var max128 = Sse41.Max(accMax.GetLower(), accMax.GetUpper());
                var inv = Sse2.Xor(max128, Vector128<ushort>.AllBitsSet);
                kMax = (ushort)~Sse41.MinHorizontal(inv).GetElement(0);
            }

            for (; i < n; i++)
            {
                ushort k = HalfKeyScalar(p[i]);
                if (k > kMax) kMax = k;
                if (k < kMin) kMin = k;
            }

            // NaN anywhere (either sign) → NumPy returns the FIRST NaN verbatim.
            if (kMax > 0xFC00 || kMin < 0x03FF)
                return HalfFirstNaNBits(p, n);

            ushort bits = HalfUnkeyScalar(isMin ? kMin : kMax);

            // ±0 extremum → NumPy keeps the FIRST zero's sign (ge/le call -0 == +0).
            if ((bits & 0x7FFF) == 0)
                return HalfFirstZeroBits(p, n);

            return bits;
        }

        /// <summary>
        /// Flat NaN-SKIPPING max over a contiguous float16 buffer, returned as raw bits —
        /// NumPy's <c>nanmax</c> == <c>fmax.reduce</c> (probed 2.4.2): NaN elements are
        /// skipped, ±0 ties keep the FIRST zero, and an all-NaN input returns the FIRST
        /// element VERBATIM (the fold's accumulator never leaves it). NaN lanes are blended
        /// to the identity key (0 for max), so any finite element forces
        /// keyMax ≥ key(-inf) = 0x03ff — keyMax below that ⟺ no finite element existed.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfNanMaxBitsContiguous(ushort* p, long n)
            => HalfNanMinMaxBitsCore(p, n, isMin: false);

        /// <summary>NaN-skipping min twin (<c>nanmin</c> == <c>fmin.reduce</c>): NaN lanes
        /// blend to key 0xFFFF; keyMin above key(+inf) = 0xfc00 ⟺ all-NaN.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfNanMinBitsContiguous(ushort* p, long n)
            => HalfNanMinMaxBitsCore(p, n, isMin: true);

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfNanMinMaxBitsCore(ushort* p, long n, bool isMin)
        {
            ushort kMax = 0, kMin = 0xFFFF;
            long i = 0;

            if (Avx2.IsSupported && n >= 16)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                var m8000 = Vector256.Create(unchecked((short)0x8000));
                var accMax = Vector256<ushort>.Zero;
                var accMin = Vector256<ushort>.AllBitsSet;
                for (; i + 16 <= n; i += 16)
                {
                    var b = Avx.LoadVector256((short*)(p + i));
                    var nan = Avx2.CompareGreaterThan(Avx2.And(b, m7fff), vinf).AsUInt16();
                    var k = HalfKeyV(b, m8000);
                    if (isMin)
                        accMin = Avx2.Min(accMin, Avx2.Or(nan, k));      // NaN → 0xFFFF (identity)
                    else
                        accMax = Avx2.Max(accMax, Avx2.AndNot(nan, k));  // NaN → 0 (identity)
                }
                if (isMin)
                {
                    var min128 = Sse41.Min(accMin.GetLower(), accMin.GetUpper());
                    kMin = Sse41.MinHorizontal(min128).GetElement(0);
                }
                else
                {
                    var max128 = Sse41.Max(accMax.GetLower(), accMax.GetUpper());
                    var inv = Sse2.Xor(max128, Vector128<ushort>.AllBitsSet);
                    kMax = (ushort)~Sse41.MinHorizontal(inv).GetElement(0);
                }
            }

            for (; i < n; i++)
            {
                if ((p[i] & 0x7FFF) > 0x7C00) continue;                  // skip NaN
                ushort k = HalfKeyScalar(p[i]);
                if (k > kMax) kMax = k;
                if (k < kMin) kMin = k;
            }

            // No finite element → fold accumulator == the first element (a NaN), verbatim.
            if (isMin ? kMin > 0xFC00 : kMax < 0x03FF)
                return p[0];

            ushort bits = HalfUnkeyScalar(isMin ? kMin : kMax);
            if ((bits & 0x7FFF) == 0)
                return HalfFirstZeroBits(p, n);
            return bits;
        }

        /// <summary>16 f16 bit patterns → 16 order keys (3 single-instruction ops).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<ushort> HalfKeyV(Vector256<short> b, Vector256<short> m8000)
            => Avx2.Xor(b, Avx2.Or(Avx2.ShiftRightArithmetic(b, 15), m8000)).AsUInt16();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort HalfKeyScalar(ushort b)
            => (ushort)(b ^ (0x8000 | ((short)b >> 15)));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ushort HalfUnkeyScalar(ushort k)
            => (k & 0x8000) != 0 ? (ushort)(k ^ 0x8000) : (ushort)~k;

        /// <summary>
        /// First element whose magnitude exceeds +inf (i.e. any-sign NaN), verbatim bits.
        /// SIMD probe + movemask so an early NaN in a large array is found at scan speed;
        /// the caller guarantees one exists.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfFirstNaNBits(ushort* p, long n)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                for (; i + 16 <= n; i += 16)
                {
                    var mag = Avx2.And(Avx.LoadVector256((short*)(p + i)), m7fff);
                    var isNan = Avx2.CompareGreaterThan(mag, vinf); // both ≤ 0x7fff: sign-safe
                    int mask = Avx2.MoveMask(isNan.AsByte());
                    if (mask != 0)
                        return p[i + (BitOperations.TrailingZeroCount((uint)mask) >> 1)];
                }
            }
            for (; i < n; i++)
                if ((p[i] & 0x7FFF) > 0x7C00)
                    return p[i];
            return 0x7E00; // unreachable backstop: canonical qNaN
        }

        /// <summary>First ±0 element's bits (decides the extremum's zero sign).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfFirstZeroBits(ushort* p, long n)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                for (; i + 16 <= n; i += 16)
                {
                    var mag = Avx2.And(Avx.LoadVector256((short*)(p + i)), m7fff);
                    var isZero = Avx2.CompareEqual(mag, Vector256<short>.Zero);
                    int mask = Avx2.MoveMask(isZero.AsByte());
                    if (mask != 0)
                        return p[i + (BitOperations.TrailingZeroCount((uint)mask) >> 1)];
                }
            }
            for (; i < n; i++)
                if ((p[i] & 0x7FFF) == 0)
                    return p[i];
            return 0; // unreachable backstop: +0
        }
    }
}
