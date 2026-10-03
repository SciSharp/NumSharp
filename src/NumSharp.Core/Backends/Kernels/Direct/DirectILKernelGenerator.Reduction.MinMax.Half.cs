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
// exact.
//
// RUNS, ROWS AND SLABS (2026-10-03). The same rule serves every layout, flat and per
// axis (the walks live in Default.Reduction.MinMax.Half.cs). NumPy's reduction folds
// each output's elements in its iterator's VISITING order — the reduced axis in
// increasing logical index (NPY_ITER_DONT_NEGATE_STRIDES), a flat reduction in K order
// — and "first" above means first in that order. Three building blocks:
//
//   • a RUN — n elements first[0], first[s], … — is decided by the SIMD cores below
//     when s is ±1 (a reversed run, s = -1, keeps the direction-free value pass and
//     rescans from the HIGH address down, since its visiting order runs backwards in
//     memory), and by the scalar PRIORITY-KEY fold otherwise;
//
//   • the priority key folds all four rules into one "first maximum" search:
//         pk(b) = isnan(b) ? NaNKey : zeroNormalized(key(b)) ^ inv
//     with inv = 0 for max-like / 0xFFFF for min-like (an inverted key puts the
//     smallest value first) and NaNKey = 0xFFFF for maximum / minimum (a NaN outranks
//     every number, and nothing outranks the first NaN) or 0 for fmax / fmin (a NaN
//     loses to every number; an all-NaN run keeps its first element). Non-NaN keys
//     span [0x03ff, 0xfc00] under either inv, so both NaNKey values sit strictly
//     outside it. "Take x iff pk(x) > pk(acc)" — a tie keeps the earlier element —
//     IS NumPy's fold step for every rule (±0 normalize to one key, so a ±0 tie keeps
//     the earlier zero);
//
//   • a SLAB fold — acc[i] = fold(acc[i], x[i]) elementwise — is the pair-select of
//     DirectILKernelGenerator.Binary.MinMax.Half.cs, whose per-lane select IS the fold
//     step with the accumulator as in1.
//
// The fold step is an associative SELECTION ("the first NaN, else the first extreme
// element"), so a run's winner folded into a running winner equals folding the run's
// elements one by one: the walks may reduce long runs with SIMD and stitch the run
// winners together in visiting order.
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
            => HalfMinMaxBitsCore(p, n, isMin: false, reverse: false);

        /// <summary>
        /// Flat min over a contiguous float16 buffer, returned as raw bits.
        /// Bit-identical to NumPy 2.4.2's <c>minimum.reduce</c> (first-NaN verbatim,
        /// first-zero sign). <paramref name="n"/> must be ≥ 1.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfMinBitsContiguous(ushort* p, long n)
            => HalfMinMaxBitsCore(p, n, isMin: true, reverse: false);

        /// <summary>
        /// Shared min+max key-accumulation core. The SIMD loop is direction-free (it
        /// tracks both extremes — the min accumulator doubles as the negative-NaN
        /// detector), so <paramref name="isMin"/> is only consulted after the loop.
        /// Direct AVX2 intrinsics, not the generic Vector256.* API: portable ushort
        /// Max/GreaterThan did not lower to single instructions here (the sign kernel
        /// measured ~30× slower before switching) — pmaxuw/pminuw are exactly one
        /// instruction each.
        /// </summary>
        /// <remarks>
        /// Only the two rare-path rescans depend on the visiting direction: a forward
        /// run is visited from <paramref name="p"/>[0] up, a reversed one (a stride −1
        /// view) from <paramref name="p"/>[n − 1] down — so the "first" NaN or zero of a
        /// reversed run is the one at the HIGHEST address. The value pass reads the
        /// same memory either way.
        /// </remarks>
        /// <param name="p">The run's LOWEST address (its first element when forward, its last when reversed).</param>
        /// <param name="n">The run length (≥ 1).</param>
        /// <param name="isMin">minimum when true, maximum when false.</param>
        /// <param name="reverse">True when the run is visited from the high address down.</param>
        /// <returns>The winner's raw bits: the first NaN verbatim, else the first extreme element.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfMinMaxBitsCore(ushort* p, long n, bool isMin, bool reverse)
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

            // NaN anywhere (either sign) → NumPy returns the FIRST NaN verbatim (first in visiting order).
            if (kMax > 0xFC00 || kMin < 0x03FF)
                return reverse ? HalfLastNaNBits(p, n) : HalfFirstNaNBits(p, n);

            ushort bits = HalfUnkeyScalar(isMin ? kMin : kMax);

            // ±0 extremum → NumPy keeps the FIRST zero's sign (ge/le call -0 == +0).
            if ((bits & 0x7FFF) == 0)
                return reverse ? HalfLastZeroBits(p, n) : HalfFirstZeroBits(p, n);

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
            => HalfNanMinMaxBitsCore(p, n, isMin: false, reverse: false);

        /// <summary>NaN-skipping min twin (<c>nanmin</c> == <c>fmin.reduce</c>): NaN lanes
        /// blend to key 0xFFFF; keyMin above key(+inf) = 0xfc00 ⟺ all-NaN.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfNanMinBitsContiguous(ushort* p, long n)
            => HalfNanMinMaxBitsCore(p, n, isMin: true, reverse: false);

        /// <summary>
        /// The fmax / fmin core behind <see cref="HalfNanMaxBitsContiguous"/> and
        /// <see cref="HalfNanMinBitsContiguous"/>, with the visiting direction made explicit.
        /// </summary>
        /// <remarks>
        /// As in <see cref="HalfMinMaxBitsCore"/>, only the rare paths look at the direction: an all-NaN run returns
        /// its FIRST VISITED element (the fold's accumulator never leaves it) — <paramref name="p"/>[0] forward,
        /// <paramref name="p"/>[n − 1] reversed — and a ±0 extremum rescans for the first visited zero.
        /// </remarks>
        /// <param name="p">The run's LOWEST address (its first element when forward, its last when reversed).</param>
        /// <param name="n">The run length (≥ 1).</param>
        /// <param name="isMin">fmin when true, fmax when false.</param>
        /// <param name="reverse">True when the run is visited from the high address down.</param>
        /// <returns>The winner's raw bits: the first extreme non-NaN element, or the first element of an all-NaN run.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfNanMinMaxBitsCore(ushort* p, long n, bool isMin, bool reverse)
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

            // No finite element → fold accumulator == the first VISITED element (a NaN), verbatim.
            if (isMin ? kMin > 0xFC00 : kMax < 0x03FF)
                return reverse ? p[n - 1] : p[0];

            ushort bits = HalfUnkeyScalar(isMin ? kMin : kMax);
            if ((bits & 0x7FFF) == 0)
                return reverse ? HalfLastZeroBits(p, n) : HalfFirstZeroBits(p, n);
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

        /// <summary>
        /// The LAST NaN in memory — the first one a reversed run visits — verbatim bits. The mirror of
        /// <see cref="HalfFirstNaNBits"/>: vectors from the high end down, the highest set mask bit picks the lane.
        /// </summary>
        /// <param name="p">The run's lowest address.</param>
        /// <param name="n">The run length; the caller guarantees a NaN in <c>p[0 .. n)</c>.</param>
        /// <returns>The NaN's raw bits (payload and sign kept); the canonical quiet NaN if the guarantee is broken.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfLastNaNBits(ushort* p, long n)
        {
            long i = n;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                var vinf = Vector256.Create((short)0x7C00);
                for (; i >= 16; i -= 16)
                {
                    var mag = Avx2.And(Avx.LoadVector256((short*)(p + i - 16)), m7fff);
                    int mask = Avx2.MoveMask(Avx2.CompareGreaterThan(mag, vinf).AsByte());
                    // Each 16-bit lane sets two mask bits; the highest set bit belongs to the highest NaN lane.
                    if (mask != 0)
                        return p[i - 16 + ((31 - BitOperations.LeadingZeroCount((uint)mask)) >> 1)];
                }
            }
            for (; i > 0; i--)
                if ((p[i - 1] & 0x7FFF) > 0x7C00)
                    return p[i - 1];
            return 0x7E00; // unreachable backstop: canonical qNaN
        }

        /// <summary>
        /// The LAST ±0 in memory — the first zero a reversed run visits — whose sign decides a zero extremum.
        /// </summary>
        /// <param name="p">The run's lowest address.</param>
        /// <param name="n">The run length; the caller guarantees a zero in <c>p[0 .. n)</c>.</param>
        /// <returns>The zero's raw bits (0x0000 or 0x8000).</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe ushort HalfLastZeroBits(ushort* p, long n)
        {
            long i = n;
            if (Avx2.IsSupported)
            {
                var m7fff = Vector256.Create((short)0x7FFF);
                for (; i >= 16; i -= 16)
                {
                    var mag = Avx2.And(Avx.LoadVector256((short*)(p + i - 16)), m7fff);
                    int mask = Avx2.MoveMask(Avx2.CompareEqual(mag, Vector256<short>.Zero).AsByte());
                    if (mask != 0)
                        return p[i - 16 + ((31 - BitOperations.LeadingZeroCount((uint)mask)) >> 1)];
                }
            }
            for (; i > 0; i--)
                if ((p[i - 1] & 0x7FFF) == 0)
                    return p[i - 1];
            return 0; // unreachable backstop: +0
        }

        /// <summary>
        /// The shortest ±1-stride run handed to the SIMD cores. Below it the scalar priority-key fold is as fast: the
        /// cores' value pass has no vector work to amortize and a NaN / zero result costs them a second (rescan) pass.
        /// </summary>
        internal const long HalfSimdRunMin = 32;

        /// <summary>
        /// The winner of a float16 RUN — <paramref name="n"/> elements visited in the order <c>first[0]</c>,
        /// <c>first[stride]</c>, …, <c>first[(n − 1)·stride]</c> — under NumPy's sequential HALF loop. maximum / minimum
        /// (<paramref name="nanIgnore"/> false) keep the FIRST NaN verbatim, else the first extreme element; fmax / fmin
        /// (<paramref name="nanIgnore"/> true) skip NaN, keep the first extreme non-NaN element, and an all-NaN run keeps
        /// its first element. A ±0 tie keeps the earlier zero under every rule.
        /// </summary>
        /// <remarks>
        /// Dispatch: a ±1 stride of at least <see cref="HalfSimdRunMin"/> elements goes to the SIMD cores (a −1 run as a
        /// reversed run over its memory range); a zero stride (a broadcast run: n copies of one element) or a lone element
        /// is its first element under every rule; everything else takes <see cref="HalfStridedWinnerBits"/>.
        /// </remarks>
        /// <param name="first">The run's first VISITED element (for a negative stride, its highest address).</param>
        /// <param name="stride">The element stride between consecutive visited elements (any sign, 0 for a broadcast run).</param>
        /// <param name="n">The run length (≥ 1).</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin (NaN skipped) when true, maximum / minimum (NaN propagates) when false.</param>
        /// <returns>The winner's raw bits, exactly NumPy's for the same visiting order.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfRunWinnerBits(ushort* first, long stride, long n, bool isMax, bool nanIgnore)
        {
            if (n <= 1 || stride == 0)
                return *first;

            if (n >= HalfSimdRunMin)
            {
                if (stride == 1)
                    return nanIgnore
                        ? HalfNanMinMaxBitsCore(first, n, isMin: !isMax, reverse: false)
                        : HalfMinMaxBitsCore(first, n, isMin: !isMax, reverse: false);
                if (stride == -1)
                {
                    // Visited from first down to first - (n - 1): the cores read the range from its low end and take the
                    // direction only for the rare-path rescans.
                    ushort* lo = first - (n - 1);
                    return nanIgnore
                        ? HalfNanMinMaxBitsCore(lo, n, isMin: !isMax, reverse: true)
                        : HalfMinMaxBitsCore(lo, n, isMin: !isMax, reverse: true);
                }
            }

            return HalfStridedWinnerBits(first, stride, n, isMax, nanIgnore);
        }

        /// <summary>
        /// The scalar priority-key fold (see the file header): one pass, any stride, every rule — "take the element when
        /// its key is strictly greater" is NumPy's fold step, so the survivor is NumPy's to the bit. Under maximum /
        /// minimum the first NaN outranks everything that follows, so the scan stops there.
        /// </summary>
        /// <remarks>
        /// The data-dependent branch is taken only when the running winner improves — O(log n) times for unordered data —
        /// so it predicts well; the key itself is branch-free. Kept out of line on purpose: inlined into the axis walk's
        /// row loop (a large method) its loop ran ~1.5× slower (0.58 → 0.86 ns per element, measured on stride-3 and
        /// stride-2 rows); a call per long row costs nothing measurable. Short rows take
        /// <see cref="HalfShortRunWinnerBits"/> instead.
        /// </remarks>
        /// <param name="first">The run's first VISITED element.</param>
        /// <param name="stride">The element stride between consecutive visited elements (any sign).</param>
        /// <param name="n">The run length (≥ 1).</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin when true, maximum / minimum when false.</param>
        /// <returns>The winner's raw bits.</returns>
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ushort HalfStridedWinnerBits(ushort* first, long stride, long n, bool isMax, bool nanIgnore)
        {
            uint inv = isMax ? 0u : 0xFFFFu;
            uint nanKey = nanIgnore ? 0u : 0xFFFFu;
            ushort bestBits = *first;
            uint best = HalfPriorityKey(bestBits, inv, nanKey);
            // Only a NaN under maximum / minimum reaches 0xFFFF (every number keys into [0x03ff, 0xfc00]): it sticks.
            if (best == 0xFFFF)
                return bestBits;

            ushort* p = first;
            for (long i = 1; i < n; i++)
            {
                p += stride;
                ushort b = *p;
                uint k = HalfPriorityKey(b, inv, nanKey);
                if (k > best)
                {
                    best = k;
                    bestBits = b;
                    if (k == 0xFFFF)
                        break;      // the first NaN: nothing later can outrank it
                }
            }

            return bestBits;
        }

        /// <summary>
        /// The longest row <see cref="HalfShortRunWinnerBits"/> serves. On a short row the "running winner improves"
        /// branch of <see cref="HalfStridedWinnerBits"/> is a coin flip (the second of three elements beats the first half
        /// of the time), so a mispredict per row costs more than the fold; past this length the branch settles and the
        /// branch-free form's longer dependency chain would cost more instead.
        /// </summary>
        internal const long HalfShortRunMax = 16;

        /// <summary>
        /// The branch-free twin of <see cref="HalfStridedWinnerBits"/> for SHORT runs (an axis reduction's rows of a few
        /// elements): the same priority-key fold, the winner and its key selected by a mask instead of a branch. Every row
        /// is an independent chain, so consecutive rows overlap in the out-of-order core.
        /// </summary>
        /// <remarks>
        /// No early exit: under maximum / minimum a NaN keys to 0xFFFF and no later key exceeds it, so the winner sticks
        /// without one — the scan only finishes the row.
        /// </remarks>
        /// <param name="first">The run's first VISITED element.</param>
        /// <param name="stride">The element stride between consecutive visited elements (any sign).</param>
        /// <param name="n">The run length (≥ 1).</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin when true, maximum / minimum when false.</param>
        /// <returns>The winner's raw bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe ushort HalfShortRunWinnerBits(ushort* first, long stride, long n, bool isMax, bool nanIgnore)
        {
            uint inv = isMax ? 0u : 0xFFFFu;
            uint nanKey = nanIgnore ? 0u : 0xFFFFu;
            uint bestBits = *first;
            uint best = HalfPriorityKey((ushort)bestBits, inv, nanKey);
            ushort* p = first;
            for (long i = 1; i < n; i++)
            {
                p += stride;
                uint b = *p;
                uint k = HalfPriorityKey((ushort)b, inv, nanKey);
                // All ones exactly when k > best: both keys are <= 0xFFFF, so best - k is negative as an int iff k wins.
                uint take = (uint)((int)(best - k) >> 31);
                bestBits ^= (bestBits ^ b) & take;
                best ^= (best ^ k) & take;
            }

            return (ushort)bestBits;
        }

        /// <summary>
        /// The priority key of one float16 under a fold rule: the order key with ±0 merged into one value (npy_half_ge /
        /// le call them equal), inverted for the min-like rules, and a NaN mapped to <paramref name="nanKey"/> — 0xFFFF
        /// (outranks every number: maximum / minimum) or 0 (loses to every number: fmax / fmin).
        /// </summary>
        /// <param name="b">The element's raw bits.</param>
        /// <param name="inv">0 for max-like rules, 0xFFFF for min-like ones.</param>
        /// <param name="nanKey">The key every NaN gets (0xFFFF or 0).</param>
        /// <returns>The key; a strictly greater key is the later element that wins a fold step.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static uint HalfPriorityKey(ushort b, uint inv, uint nanKey)
        {
            uint mag = b & 0x7FFFu;
            uint key = (ushort)(b ^ (0x8000 | ((short)b >> 15)));
            key = mag == 0 ? 0x8000u : key;      // -0 (key 0x7fff) joins +0 (key 0x8000): no other key lies between them
            key ^= inv;
            return mag > 0x7C00u ? nanKey : key;
        }

        /// <summary>
        /// One SLAB step of an axis reduction: <c>acc[i] = fold(acc[i], x[i·xStride])</c> for <c>i</c> in
        /// <c>[0, n)</c>, the running winners as NumPy's in1 — the per-lane pair-select of
        /// <see cref="HalfPairSelectFull"/> (in place: each lane reads both operands before storing) for a contiguous
        /// slab, the scalar <see cref="HalfPairSelectBits"/> otherwise.
        /// </summary>
        /// <param name="acc">The contiguous running winners (updated in place).</param>
        /// <param name="x">The slab's first element.</param>
        /// <param name="xStride">The slab's element stride (any sign).</param>
        /// <param name="n">The number of lanes.</param>
        /// <param name="isMax">maximum / fmax when true, minimum / fmin when false.</param>
        /// <param name="nanIgnore">fmax / fmin when true, maximum / minimum when false.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void HalfFoldSlab(ushort* acc, ushort* x, long xStride, long n, bool isMax, bool nanIgnore)
        {
            if (xStride == 1)
            {
                HalfPairSelectFull(acc, x, acc, n, isMax, nanIgnore);
                return;
            }

            ushort* p = x;
            for (long i = 0; i < n; i++, p += xStride)
                acc[i] = HalfPairSelectBits(acc[i], *p, isMax, nanIgnore);
        }

        /// <summary>Copies a strided float16 run into a contiguous buffer (a memcpy for a contiguous run).</summary>
        /// <param name="dst">The contiguous destination.</param>
        /// <param name="src">The run's first element.</param>
        /// <param name="srcStride">The run's element stride (any sign).</param>
        /// <param name="n">The number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void HalfGatherRun(ushort* dst, ushort* src, long srcStride, long n)
        {
            if (srcStride == 1)
            {
                Buffer.MemoryCopy(src, dst, n * sizeof(ushort), n * sizeof(ushort));
                return;
            }

            ushort* p = src;
            for (long i = 0; i < n; i++, p += srcStride)
                dst[i] = *p;
        }

        /// <summary>Copies a contiguous float16 buffer into a strided run (a memcpy for a contiguous run).</summary>
        /// <param name="dst">The run's first element.</param>
        /// <param name="dstStride">The run's element stride (any sign).</param>
        /// <param name="src">The contiguous source.</param>
        /// <param name="n">The number of elements.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void HalfScatterRun(ushort* dst, long dstStride, ushort* src, long n)
        {
            if (dstStride == 1)
            {
                Buffer.MemoryCopy(src, dst, n * sizeof(ushort), n * sizeof(ushort));
                return;
            }

            ushort* p = dst;
            for (long i = 0; i < n; i++, p += dstStride)
                *p = src[i];
        }
    }
}
