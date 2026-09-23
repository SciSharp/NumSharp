using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// DefaultEngine.Evaluate.MinMax.cs — NumPy-EXACT flat Min / Max for np.evaluate
// (plan docs/plans/ndexpr-evaluate.md, "Perf review 2026-09-23" item 3)
// =============================================================================
//
// A flat `np.max(child)` / `np.min(child)` in NumPy 2.4.2 (win-amd64 wheel, AVX2 dispatch) reduces a
// CONTIGUOUS buffer with ONE call of loops_minmax.dispatch.c.src::simd_reduce_c_{max,min}_<sfx> over the
// n-1 elements after the first (the reduction machinery copies element 0 into the result and iterates the
// rest; for a computed child that buffer is the fresh K-order array the ufunc materialized):
//
//     acc = splat(x[0])
//     while >= 8 vectors remain:  r01 = N(v0,v1) r23 = N(v2,v3) r45 = N(v4,v5) r67 = N(v6,v7)
//                                 acc = N(acc, N(N(r01,r23), N(r45,r67)))
//     while >= 1 vector remains:  acc = N(acc, v)
//     r = any lane of acc NaN ? canonical +NaN : fold acc's halves with P (lo vs hi, down to lane 0)
//     while elements remain:      r = N(r, x)                                   (the SSE scalar op)
//
// with the per-lane rule N(a, b) = isnan(a) ? a : P(a, b) and P(a, b) = a > b ? a : b (min: a < b) — the
// _mm256_max_pd / _mm_max_sd semantics, where the SECOND operand wins a tie and an unordered compare.
// Probed against NumPy 2.4.2 on 2,056 tie-heavy (±0) and NaN-laced cases (float32 + float64, max + min,
// n in 1..20001 incl. lane-structured zeros across the 8192 buffer boundary): this schedule matched every
// result BIT-FOR-BIT, NaN payloads included; a per-8192-chunk restart missed 48, and a sequential scan
// (either tie rule) missed 152-304. So for floats the ORDER is observable exactly where the answer is a
// ±0 (which zero sign wins) or a NaN (a vector-section NaN comes back as the canonical +NaN, a tail NaN
// keeps its payload) — the fused 4-accumulator scalar fold matched NumPy's VALUE but not those bits, which
// is why the evaluate oracle had to fold signed zeros into +0 for min/max. Integers carry neither: an
// integer min/max is order-free, so any schedule is exact and theirs is purely a speed matter.
//
// The kernels are shaped as Vector256<T> on EVERY host — deliberately: the lane structure IS the contract
// (NumPy's AVX2 build generated the committed corpus), and Vector256's portable software path reproduces
// the same lanes where 256-bit SIMD is absent. The float lane op is NumPy's own three instructions when AVX
// is present (vmaxp* + vcmpordp* + vblendvp*, gated on Avx.IsSupported — RyuJIT never swaps the operands of
// a floating-point max/min intrinsic, which is what keeps the second-operand tie rule intact) and an
// explicit compare + ConditionalSelect otherwise; integer lanes use plain Vector256.Max/Min (order-free).
// The portable compare + select form cost ~7 uops a vector against the AVX form's 3 — 11.1 vs ~6 us on a
// 100K bare-leaf max, so the gate is a measured speed matter, not a semantic one.
//
// Routes (DefaultEngine.TryExactFlatMinMax), each exactly the buffer NumPy reduces:
//   * the child is a bare DENSE array leaf (C- or F-contiguous, or any transpose of one — a single dense
//     block): the schedule runs over that block in place, in memory order — NumPy reduces the array itself,
//     and its K-order iterator coalesces a dense block into one inner loop over memory. A NON-dense leaf takes
//     the engine's flat walker (DefaultEngine.TryExactFlatMinMaxArray, Default.Reduction.MinMax.Exact.cs):
//     NumPy's iterator reproduced — coalesced runs, a strided run's 8-accumulator unroll, and for layouts that
//     do not coalesce, NumPy's own buffering decision (row by row in place, or contiguous calls over buffered
//     chunks). Only a BROADCAST leaf keeps the fold (value-exact).
//   * the child is computed and its operands stream (CanStreamChild, dense permutations included): the child
//     kernel evaluates 8 KB blocks into L1 scratch in memory order (a whole number of 8-vector groups per
//     block, so only the LAST block is partial and the schedule is unbroken), no temp.
//   * a computed FLOAT child whose operands do not stream: materialize it (EvaluateCore — the same
//     K-order buffer the M1 Sum divert reduces) and run the schedule over that. An integer child falls
//     back to the (already exact) fold instead.
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    /// A lane-exact port of NumPy 2.4.2's contiguous flat min / max reduction
    /// (<c>simd_reduce_c_{max,min}_&lt;sfx&gt;</c> with the AVX2 <c>npyv</c> intrinsics and the SSE scalar tail).
    /// Stateless; the caller owns every buffer.
    /// </summary>
    /// <remarks>
    /// The bit contract only bites for floats — which zero sign survives a ±0 tie, and whether a NaN result
    /// is the canonical +NaN (any NaN that reached the vector section) or keeps its payload (a NaN met only in
    /// the scalar tail, or a lone element). Integer lanes have no ties that differ in bits and no NaN, so for
    /// them this is simply a vectorized exact reduction.
    /// </remarks>
    internal static partial class NumPyMinMaxReduce
    {
        /// <summary>
        /// True for the dtypes this reduction serves: float64 / float32 (NumPy-exact bits) and the eight integer
        /// widths (order-free, so exact by construction). Half, Char, Decimal, Complex and Boolean are left to the
        /// fold — they have no <see cref="Vector256{T}"/> lane type or no NumPy SIMD loop to reproduce.
        /// </summary>
        /// <param name="t">The child (== result) dtype.</param>
        /// <returns>Whether <see cref="ReduceContiguous"/> and the streamed form accept <paramref name="t"/>.</returns>
        internal static bool Supports(NPTypeCode t)
            => t is NPTypeCode.Double or NPTypeCode.Single
                or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>
        /// True for float64 / float32 — the dtypes whose result BITS depend on the schedule (±0 ties, NaN
        /// canonicalization), and therefore the only ones worth materializing a non-streamable child for.
        /// </summary>
        /// <param name="t">The child dtype.</param>
        /// <returns>Whether <paramref name="t"/> is a float the schedule is a bit contract for.</returns>
        internal static bool IsFloat(NPTypeCode t) => t is NPTypeCode.Double or NPTypeCode.Single;

        /// <summary>
        /// One lane rule — max or min — in its vector (<c>npyv_maxn</c> / <c>npyv_minn</c>) and scalar forms.
        /// "a beats b" is <c>a &gt; b</c> for max and <c>a &lt; b</c> for min: ordered, so false when either side is
        /// NaN, which is what makes the SECOND operand win both a tie and an unordered compare.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        internal interface ILane<T> where T : unmanaged, INumber<T>
        {
            /// <summary>
            /// <c>npyv_maxn</c> / <c>npyv_minn</c>: per lane, <paramref name="a"/> where it is NaN or beats
            /// <paramref name="b"/>, else <paramref name="b"/> — NaN-propagating from either side, the second operand
            /// winning a tie.
            /// </summary>
            /// <param name="a">First operand (the running accumulator side).</param>
            /// <param name="b">Second operand (the incoming side).</param>
            /// <returns>The per-lane result.</returns>
            static abstract Vector256<T> N(Vector256<T> a, Vector256<T> b);

            /// <summary>Scalar "a beats b".</summary>
            /// <param name="a">First operand.</param>
            /// <param name="b">Second operand.</param>
            /// <returns>True when <paramref name="a"/> strictly beats <paramref name="b"/>.</returns>
            static abstract bool Beats(T a, T b);

            /// <summary>
            /// The PLAIN 128-bit lane op of the horizontal step — <c>_mm_max_p*</c> / <c>_mm_min_p*</c> (and the integer
            /// <c>_mm_max_ep*</c> forms): per lane <paramref name="a"/> where it beats <paramref name="b"/>, else
            /// <paramref name="b"/>, with NO NaN blend. Only ever applied after the caller ruled NaN out, so the second
            /// operand winning a tie is the whole contract.
            /// </summary>
            /// <param name="a">First operand (the lanes being kept).</param>
            /// <param name="b">Second operand (the lanes moved down by the cascade) — wins a tie.</param>
            /// <returns>The per-lane result.</returns>
            static abstract Vector128<T> P128(Vector128<T> a, Vector128<T> b);

            /// <summary>
            /// The PLAIN 256-bit lane op — <c>_mm256_max_p*</c> / <c>_mm256_min_p*</c>: per lane <paramref name="a"/> where it
            /// beats <paramref name="b"/>, else <paramref name="b"/>, with NO NaN blend. Equal to <see cref="N"/> on every
            /// lane whose FIRST operand is not NaN — which is what lets <see cref="FoldGroups{T,TLane}"/> run a group whose
            /// lanes it has proven NaN-free with one instruction per op instead of NumPy's three.
            /// </summary>
            /// <param name="a">First operand.</param>
            /// <param name="b">Second operand — wins a tie and an unordered compare.</param>
            /// <returns>The per-lane result.</returns>
            static abstract Vector256<T> P256(Vector256<T> a, Vector256<T> b);
        }

        /// <summary>The maximum lane rule (<c>_mm256_max_pd</c>: <c>a &gt; b ? a : b</c>, NaN-blended).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        internal readonly struct MaxLane<T> : ILane<T> where T : unmanaged, INumber<T>
        {
            /// <inheritdoc />
            /// <remarks>
            /// Floats with AVX run NumPy's own three instructions — <c>vmaxp*</c>, <c>vcmpordp*</c>, <c>vblendvp*</c>
            /// (<c>blendv(a, max(a, b), ord(a, a))</c>). That is safe to spell with <see cref="Avx.Max(Vector256{double}, Vector256{double})"/>
            /// because RyuJIT treats the FLOATING-POINT max/min intrinsics as non-commutative (it never swaps their
            /// operands; only the integer forms are commutative), and the lane-structured ±0 oracle cases would expose a
            /// swap as flipped zero signs. Without AVX the same rule is a portable compare + select. Integer lanes carry
            /// no NaN and no tie that differs in bits, so plain <see cref="Vector256.Max{T}"/> is exact for them.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> N(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                {
                    if (Avx.IsSupported)
                    {
                        var ad = a.AsDouble();
                        return Avx.BlendVariable(ad, Avx.Max(ad, b.AsDouble()), Avx.CompareOrdered(ad, ad)).As<double, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.GreaterThan(a, b) | ~Vector256.Equals(a, a), a, b);
                }

                if (typeof(T) == typeof(float))
                {
                    if (Avx.IsSupported)
                    {
                        var af = a.AsSingle();
                        return Avx.BlendVariable(af, Avx.Max(af, b.AsSingle()), Avx.CompareOrdered(af, af)).As<float, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.GreaterThan(a, b) | ~Vector256.Equals(a, a), a, b);
                }

                return Vector256.Max(a, b);
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Beats(T a, T b) => a > b;

            /// <inheritdoc />
            /// <remarks>
            /// Floats take <c>maxpd</c> / <c>maxps</c> through <see cref="Sse2"/> / <see cref="Sse"/> (the floating-point
            /// max intrinsics are never operand-swapped by RyuJIT, so the second operand keeps winning a tie); without SSE a
            /// portable compare + select spells the same rule. Integer lanes have no bit-distinct ties, so
            /// <see cref="Vector128.Max{T}"/> is exact.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector128<T> P128(Vector128<T> a, Vector128<T> b)
            {
                if (typeof(T) == typeof(double))
                    return Sse2.IsSupported
                        ? Sse2.Max(a.AsDouble(), b.AsDouble()).As<double, T>()
                        : Vector128.ConditionalSelect(Vector128.GreaterThan(a, b), a, b);
                if (typeof(T) == typeof(float))
                    return Sse.IsSupported
                        ? Sse.Max(a.AsSingle(), b.AsSingle()).As<float, T>()
                        : Vector128.ConditionalSelect(Vector128.GreaterThan(a, b), a, b);
                return Vector128.Max(a, b);
            }

            /// <inheritdoc />
            /// <remarks>
            /// Floats take <c>vmaxpd</c> / <c>vmaxps</c> through <see cref="Avx"/> (never operand-swapped by RyuJIT, so the
            /// second operand keeps winning a tie); without AVX a portable compare + select spells the same rule. Integer
            /// lanes have no bit-distinct ties, so <see cref="Vector256.Max{T}"/> is exact.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> P256(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                    return Avx.IsSupported
                        ? Avx.Max(a.AsDouble(), b.AsDouble()).As<double, T>()
                        : Vector256.ConditionalSelect(Vector256.GreaterThan(a, b), a, b);
                if (typeof(T) == typeof(float))
                    return Avx.IsSupported
                        ? Avx.Max(a.AsSingle(), b.AsSingle()).As<float, T>()
                        : Vector256.ConditionalSelect(Vector256.GreaterThan(a, b), a, b);
                return Vector256.Max(a, b);
            }
        }

        /// <summary>The minimum lane rule (<c>_mm256_min_pd</c>: <c>a &lt; b ? a : b</c>, NaN-blended).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        internal readonly struct MinLane<T> : ILane<T> where T : unmanaged, INumber<T>
        {
            /// <inheritdoc />
            /// <remarks>The mirror of <see cref="MaxLane{T}.N"/> (same safety argument for <c>Avx.Min</c>).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> N(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                {
                    if (Avx.IsSupported)
                    {
                        var ad = a.AsDouble();
                        return Avx.BlendVariable(ad, Avx.Min(ad, b.AsDouble()), Avx.CompareOrdered(ad, ad)).As<double, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.LessThan(a, b) | ~Vector256.Equals(a, a), a, b);
                }

                if (typeof(T) == typeof(float))
                {
                    if (Avx.IsSupported)
                    {
                        var af = a.AsSingle();
                        return Avx.BlendVariable(af, Avx.Min(af, b.AsSingle()), Avx.CompareOrdered(af, af)).As<float, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.LessThan(a, b) | ~Vector256.Equals(a, a), a, b);
                }

                return Vector256.Min(a, b);
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Beats(T a, T b) => a < b;

            /// <inheritdoc />
            /// <remarks>The mirror of <see cref="MaxLane{T}.P128"/> (<c>minpd</c> / <c>minps</c> / <see cref="Vector128.Min{T}"/>).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector128<T> P128(Vector128<T> a, Vector128<T> b)
            {
                if (typeof(T) == typeof(double))
                    return Sse2.IsSupported
                        ? Sse2.Min(a.AsDouble(), b.AsDouble()).As<double, T>()
                        : Vector128.ConditionalSelect(Vector128.LessThan(a, b), a, b);
                if (typeof(T) == typeof(float))
                    return Sse.IsSupported
                        ? Sse.Min(a.AsSingle(), b.AsSingle()).As<float, T>()
                        : Vector128.ConditionalSelect(Vector128.LessThan(a, b), a, b);
                return Vector128.Min(a, b);
            }

            /// <inheritdoc />
            /// <remarks>The mirror of <see cref="MaxLane{T}.P256"/> (<c>vminpd</c> / <c>vminps</c> / <see cref="Vector256.Min{T}"/>).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> P256(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                    return Avx.IsSupported
                        ? Avx.Min(a.AsDouble(), b.AsDouble()).As<double, T>()
                        : Vector256.ConditionalSelect(Vector256.LessThan(a, b), a, b);
                if (typeof(T) == typeof(float))
                    return Avx.IsSupported
                        ? Avx.Min(a.AsSingle(), b.AsSingle()).As<float, T>()
                        : Vector256.ConditionalSelect(Vector256.LessThan(a, b), a, b);
                return Vector256.Min(a, b);
            }
        }

        /// <summary>
        /// <c>npyv_maxn</c> / <c>npyv_minn</c> through the lane rule (see <see cref="ILane{T}.N"/>).
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="a">First operand (the running accumulator side).</param>
        /// <param name="b">Second operand.</param>
        /// <returns>The per-lane result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<T> N<T, TLane>(Vector256<T> a, Vector256<T> b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.N(a, b);

        /// <summary>The SSE scalar op (<c>scalar_max_d</c> on x86): the same rule as the vector lanes.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="a">The running value.</param>
        /// <param name="b">The incoming element.</param>
        /// <returns><paramref name="a"/> when it is NaN or beats <paramref name="b"/>, else <paramref name="b"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static T N<T, TLane>(T a, T b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => T.IsNaN(a) || TLane.Beats(a, b) ? a : b;

        /// <summary>
        /// The plain <c>_mm_max_p*</c> / <c>_mm_min_p*</c> lane (no NaN blend) the horizontal step uses — only ever
        /// applied after the caller ruled NaN out.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="a">First operand (the lower half's lane).</param>
        /// <param name="b">Second operand (the upper half's lane) — wins a tie.</param>
        /// <returns>The winning operand.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T P<T, TLane>(T a, T b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.Beats(a, b) ? a : b;

        /// <summary>
        /// Phase 1 of the schedule: fold <paramref name="groups"/> whole 8-vector groups into
        /// <paramref name="acc"/>, each as <c>acc = N(acc, N(N(N(v0,v1), N(v2,v3)), N(N(v4,v5), N(v6,v7))))</c>.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="acc">The running accumulator vector.</param>
        /// <param name="ip">First element of the first group (unaligned is fine).</param>
        /// <param name="groups">Number of whole groups (8 × <see cref="Vector256{T}.Count"/> elements each) to fold.</param>
        /// <returns>The updated accumulator.</returns>
        internal static unsafe Vector256<T> FoldGroups<T, TLane>(Vector256<T> acc, T* ip, long groups)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int vstep = Vector256<T>.Count;
            long g = 0;

            // NaN-free fast path (float lanes with AVX, NaN-free accumulator): NumPy's lane op N(a, b) is
            // blend(a, max(a, b), ord(a, a)) — it differs from the plain max P(a, b) ONLY on a lane whose first operand
            // is NaN. A group whose eight vectors hold no NaN therefore folds bit-identically with P (every
            // intermediate stays NaN-free too), at one instruction per op instead of three. One unordered compare
            // covers a PAIR of vectors (a lane is unordered when either side is NaN), so the proof costs four compares
            // per group. The first group holding a NaN breaks out BEFORE touching acc, and it and every later group take
            // NumPy's blended rule below from exactly the accumulator NumPy would hold. Integer lanes already fold with
            // the plain op (N is Vector256.Max/Min for them), so they skip this.
            if ((typeof(T) == typeof(double) || typeof(T) == typeof(float)) && Avx.IsSupported && Vector256.EqualsAll(acc, acc))
            {
                for (; g < groups; g++, ip += vstep * 8)
                {
                    if (Sse.IsSupported)
                        Sse.Prefetch0(ip + vstep * 8);
                    var v0 = Vector256.Load(ip);
                    var v1 = Vector256.Load(ip + vstep);
                    var v2 = Vector256.Load(ip + vstep * 2);
                    var v3 = Vector256.Load(ip + vstep * 3);
                    var v4 = Vector256.Load(ip + vstep * 4);
                    var v5 = Vector256.Load(ip + vstep * 5);
                    var v6 = Vector256.Load(ip + vstep * 6);
                    var v7 = Vector256.Load(ip + vstep * 7);
                    var nan = (Unordered(v0, v1) | Unordered(v2, v3)) | (Unordered(v4, v5) | Unordered(v6, v7));
                    if (nan.ExtractMostSignificantBits() != 0)
                        break;
                    var r01 = TLane.P256(v0, v1);
                    var r23 = TLane.P256(v2, v3);
                    var r45 = TLane.P256(v4, v5);
                    var r67 = TLane.P256(v6, v7);
                    acc = TLane.P256(acc, TLane.P256(TLane.P256(r01, r23), TLane.P256(r45, r67)));
                }
            }

            for (; g < groups; g++, ip += vstep * 8)
            {
                // NumPy's simd_reduce_c prefetches the NEXT group (NPY_PREFETCH(ip + wstep, 0, 3) → prefetcht0) on SSE2
                // builds; a hint only — it cannot change a result — so it is mirrored where the ISA has it.
                if (Sse.IsSupported)
                    Sse.Prefetch0(ip + vstep * 8);
                var v0 = Vector256.Load(ip);
                var v1 = Vector256.Load(ip + vstep);
                var v2 = Vector256.Load(ip + vstep * 2);
                var v3 = Vector256.Load(ip + vstep * 3);
                var v4 = Vector256.Load(ip + vstep * 4);
                var v5 = Vector256.Load(ip + vstep * 5);
                var v6 = Vector256.Load(ip + vstep * 6);
                var v7 = Vector256.Load(ip + vstep * 7);
                var r01 = N<T, TLane>(v0, v1);
                var r23 = N<T, TLane>(v2, v3);
                var r45 = N<T, TLane>(v4, v5);
                var r67 = N<T, TLane>(v6, v7);
                acc = N<T, TLane>(acc, N<T, TLane>(N<T, TLane>(r01, r23), N<T, TLane>(r45, r67)));
            }

            return acc;
        }

        /// <summary>
        /// <c>vcmpunordp*</c>: all-ones in every lane where <paramref name="a"/> or <paramref name="b"/> is NaN, zero
        /// elsewhere — the NaN proof of <see cref="FoldGroups{T,TLane}"/>'s fast path. Callers gate on
        /// <see cref="Avx.IsSupported"/> and a float lane type.
        /// </summary>
        /// <typeparam name="T">double or float.</typeparam>
        /// <param name="a">First vector.</param>
        /// <param name="b">Second vector.</param>
        /// <returns>The unordered mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<T> Unordered<T>(Vector256<T> a, Vector256<T> b) where T : unmanaged
            => typeof(T) == typeof(double)
                ? Avx.CompareUnordered(a.AsDouble(), b.AsDouble()).As<double, T>()
                : Avx.CompareUnordered(a.AsSingle(), b.AsSingle()).As<float, T>();

        /// <summary>
        /// Phases 1-4 over the LAST <paramref name="len"/> elements of the reduced stream: whole groups, then
        /// single vectors, then the horizontal <c>npyv_reduce_{max,min}n</c>, then the scalar tail.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="acc">The accumulator carried from any earlier <see cref="FoldGroups{T,TLane}"/> calls
        /// (<c>splat(x[0])</c> when this is the whole stream).</param>
        /// <param name="ip">First remaining element.</param>
        /// <param name="len">Remaining element count (&gt;= 1).</param>
        /// <returns>The reduction result.</returns>
        internal static unsafe T Finish<T, TLane>(Vector256<T> acc, T* ip, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int vstep = Vector256<T>.Count;
            long wstep = vstep * 8L;
            long groups = len / wstep;
            acc = FoldGroups<T, TLane>(acc, ip, groups);
            ip += groups * wstep;
            len -= groups * wstep;

            for (; len >= vstep; len -= vstep, ip += vstep)
                acc = N<T, TLane>(acc, Vector256.Load(ip));

            T r = ReduceLanes<T, TLane>(acc);

            // The scalar tail: a NaN met here keeps its own payload (NumPy's SSE op returns the operand).
            for (; len > 0; --len, ++ip)
                r = N<T, TLane>(r, *ip);
            return r;
        }

        /// <summary>
        /// <c>npyv_reduce_{max,min}n_&lt;sfx&gt;</c>: the canonical positive quiet NaN when ANY lane is NaN (the
        /// lane's own payload is NOT kept — NumPy returns a fixed constant), else the halves folded with the plain
        /// lane op down to lane 0 (lo vs hi, then pairs: exactly the extract/shuffle cascade).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="acc">The accumulator vector.</param>
        /// <returns>The horizontal result.</returns>
        private static unsafe T ReduceLanes<T, TLane>(Vector256<T> acc)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            // x == x fails only for a NaN lane; an integer vector always passes.
            if (!Vector256.EqualsAll(acc, acc))
                return CanonicalNaN<T>();

            if (Sse2.IsSupported)
            {
                // NumPy's own cascade, in registers: extract-and-fold the 128-bit halves, then fold the low half with
                // itself moved down by 8, 4, 2, 1 bytes — lane i against lane i + half, the moved lane as the SECOND
                // operand, exactly _mm_*_p?(v, _mm_shuffle_*(v, …)) — until one lane is left. The byte shifts leave zeros
                // in the vacated upper lanes; those lanes are never read. sizeof(T) is a JIT constant per instantiation,
                // so only the levels a lane width needs survive compilation (and the shift counts stay immediates). The
                // stack spill below cost ~a lane-count-long scalar loop per row, which dominated short-row reductions.
                var v = TLane.P128(acc.GetLower(), acc.GetUpper());
                if (sizeof(T) <= 4)
                    v = TLane.P128(v, Sse2.ShiftRightLogical128BitLane(v.AsByte(), 8).As<byte, T>());
                if (sizeof(T) <= 2)
                    v = TLane.P128(v, Sse2.ShiftRightLogical128BitLane(v.AsByte(), 4).As<byte, T>());
                if (sizeof(T) == 1)
                    v = TLane.P128(v, Sse2.ShiftRightLogical128BitLane(v.AsByte(), 2).As<byte, T>());
                // The last level pairs lane 0 with lane 1, one element over.
                v = TLane.P128(v, Sse2.ShiftRightLogical128BitLane(v.AsByte(), (byte)sizeof(T)).As<byte, T>());
                return v.ToScalar();
            }

            Span<T> lanes = stackalloc T[Vector256<T>.Count];
            acc.CopyTo(lanes);
            // lanes[i] = P(lanes[i], lanes[i + half]) halves the width each round: for 4 double lanes that is
            // (a0 vs a2, a1 vs a3) then (lane0 vs lane1) — _mm_max_pd(lo, hi) then the shuffle-by-1 step; for
            // 8 float lanes the extra middle round is the _MM_SHUFFLE(0,0,3,2) step.
            for (int width = lanes.Length; width > 1; width >>= 1)
            {
                int half = width >> 1;
                for (int i = 0; i < half; i++)
                    lanes[i] = P<T, TLane>(lanes[i], lanes[i + half]);
            }

            return lanes[0];
        }

        /// <summary>
        /// NumPy's canonical quiet NaN for the lane type — <c>0x7FF8000000000000</c> / <c>0x7FC00000</c>, the
        /// POSITIVE NaN (.NET's own <see cref="double.NaN"/> is the negative <c>0xFFF8...</c>, so it cannot be used).
        /// </summary>
        /// <typeparam name="T">double or float.</typeparam>
        /// <returns>The canonical NaN.</returns>
        /// <exception cref="InvalidOperationException">An integer lane type — unreachable, integer vectors hold no NaN.</exception>
        private static T CanonicalNaN<T>() where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(double))
            {
                double d = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
                return Unsafe.As<double, T>(ref d);
            }

            if (typeof(T) == typeof(float))
            {
                float f = BitConverter.Int32BitsToSingle(0x7FC00000);
                return Unsafe.As<float, T>(ref f);
            }

            throw new InvalidOperationException($"{typeof(T).Name} lanes cannot hold NaN — min/max schedule bug.");
        }

        /// <summary>
        /// The whole reduction over a contiguous block: <c>x[0]</c> seeds the accumulator and the schedule folds the
        /// other <paramref name="n"/> - 1 elements. A single element is returned as is (NumPy's loop returns before
        /// touching anything when there is nothing after the copied first element, so even a NaN keeps its payload).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="x">The first element of a dense block of <paramref name="n"/> elements in memory order.</param>
        /// <param name="n">Element count (&gt;= 1).</param>
        /// <returns>The reduction result.</returns>
        internal static unsafe T Reduce<T, TLane>(T* x, long n)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => n == 1 ? x[0] : Finish<T, TLane>(Vector256.Create(x[0]), x + 1, n - 1);

        /// <summary>
        /// Dtype dispatch for <see cref="Reduce{T,TLane}"/>: reduce the dense block at <paramref name="x"/> and write
        /// the result (at dtype <paramref name="t"/>) to offset 0 of <paramref name="slot"/>.
        /// </summary>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="x">First byte of the dense block (logical element 0 in memory order).</param>
        /// <param name="n">Element count (&gt;= 1).</param>
        /// <param name="slot">Receives the result (at least 8 bytes).</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceContiguous(NPTypeCode t, bool isMax, byte* x, long n, byte* slot)
        {
            switch (t)
            {
                case NPTypeCode.Double: *(double*)slot = isMax ? Reduce<double, MaxLane<double>>((double*)x, n) : Reduce<double, MinLane<double>>((double*)x, n); break;
                case NPTypeCode.Single: *(float*)slot = isMax ? Reduce<float, MaxLane<float>>((float*)x, n) : Reduce<float, MinLane<float>>((float*)x, n); break;
                case NPTypeCode.SByte: *(sbyte*)slot = isMax ? Reduce<sbyte, MaxLane<sbyte>>((sbyte*)x, n) : Reduce<sbyte, MinLane<sbyte>>((sbyte*)x, n); break;
                case NPTypeCode.Byte: *slot = isMax ? Reduce<byte, MaxLane<byte>>(x, n) : Reduce<byte, MinLane<byte>>(x, n); break;
                case NPTypeCode.Int16: *(short*)slot = isMax ? Reduce<short, MaxLane<short>>((short*)x, n) : Reduce<short, MinLane<short>>((short*)x, n); break;
                case NPTypeCode.UInt16: *(ushort*)slot = isMax ? Reduce<ushort, MaxLane<ushort>>((ushort*)x, n) : Reduce<ushort, MinLane<ushort>>((ushort*)x, n); break;
                case NPTypeCode.Int32: *(int*)slot = isMax ? Reduce<int, MaxLane<int>>((int*)x, n) : Reduce<int, MinLane<int>>((int*)x, n); break;
                case NPTypeCode.UInt32: *(uint*)slot = isMax ? Reduce<uint, MaxLane<uint>>((uint*)x, n) : Reduce<uint, MinLane<uint>>((uint*)x, n); break;
                case NPTypeCode.Int64: *(long*)slot = isMax ? Reduce<long, MaxLane<long>>((long*)x, n) : Reduce<long, MinLane<long>>((long*)x, n); break;
                case NPTypeCode.UInt64: *(ulong*)slot = isMax ? Reduce<ulong, MaxLane<ulong>>((ulong*)x, n) : Reduce<ulong, MinLane<ulong>>((ulong*)x, n); break;
                default: throw new NotSupportedException($"NumPy min/max schedule: dtype {t} is not served.");
            }
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        /// Flat <c>Min</c> / <c>Max</c> computed with NumPy's exact contiguous reduction schedule
        /// (<see cref="NumPyMinMaxReduce"/>), over the very buffer NumPy would reduce: the bare contiguous leaf in
        /// place, any other non-broadcast bare leaf through NumPy's iterator schedule
        /// (<see cref="TryExactFlatMinMaxArray"/>), a streamable computed child block by block through its kernel (no
        /// temp), or — for a float child whose operands do not stream — the materialized child.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Declines (false, <paramref name="slot"/> untouched — the caller's fold runs as before) when the hook
        /// <see cref="NDExpr.DisableExactMinMax"/> is set, for a dtype outside <see cref="NumPyMinMaxReduce.Supports"/>,
        /// for a BROADCAST bare leaf (a non-dense one is handed to <see cref="TryExactFlatMinMaxArray"/>, NumPy's iterator
        /// schedule, which declines only a broadcast layout), and for a non-streamable INTEGER child (an integer
        /// min/max is order-free, so the fold is already exact and materializing would only cost).
        /// </para>
        /// <para>
        /// Consequence for floats: the ±0 tie and NaN-payload bits now equal NumPy's on every route this takes —
        /// the fold matched only the value. Cost: a non-streamable float child is materialized (an n-element temp),
        /// exactly as NumPy materializes it; the streamed and in-place routes allocate nothing.
        /// </para>
        /// </remarks>
        /// <param name="child">The reduction's child elementwise program (<see cref="NDExprProgram.ChildElementwiseProgram"/>).</param>
        /// <param name="inputs">Every input of the call, in input order.</param>
        /// <param name="n">The child's element count (&gt; 0).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="t">The child (== result) dtype.</param>
        /// <param name="slot">Receives the result at dtype <paramref name="t"/> (at least 8 bytes).</param>
        /// <returns>True when the reduction was computed here.</returns>
        private unsafe bool TryExactFlatMinMax(NDExprProgram child, NDArray[] inputs, long n, bool isMax, NPTypeCode t, byte* slot)
        {
            if (NDExpr.DisableExactMinMax || !NumPyMinMaxReduce.Supports(t))
                return false;

            var ops = child.IteratorOperands(inputs);

            if (child.Bound is InputNode && ops.Length == 1)
            {
                // NumPy reduces the array itself. A dense block (C- or F-contiguous or any transpose of one, not
                // broadcast) coalesces to one inner loop over memory order — the schedule below, in place. Any other
                // non-broadcast layout goes through NumPy's iterator as the engine's np.max does — coalesced runs, a
                // strided run's 8-accumulator unroll, or buffered contiguous chunks — which TryExactFlatMinMaxArray
                // reproduces; a broadcast leaf declines there and keeps the fold (value-exact).
                var s = ops[0].Shape;
                if (!IsSharedDensePermutation(ops))
                    return TryExactFlatMinMaxArray(ops[0], isMax, slot);
                // Logical element 0: a contiguous slice re-seats Address (offset 0), an F column block keeps a
                // non-zero offset — Address + offset·itemsize is right for both.
                byte* x = (byte*)ops[0].Address + (long)s.offset * t.SizeOf();
                NumPyMinMaxReduce.ReduceContiguous(t, isMax, x, n, slot);
                NDExpr.ExactMinMaxRuns++;
                return true;
            }

            if (CanStreamChild(child, ops, allowF: true, allowPermuted: true))
            {
                StreamExactMinMax(child, inputs, ops, n, isMax, t, slot);
                NDExpr.StreamingReductions++;
                NDExpr.ExactMinMaxRuns++;
                return true;
            }

            if (!NumPyMinMaxReduce.IsFloat(t))
                return false;

            // A computed float child NumPy would materialize: materialize it the same way — in the layout NumPy's
            // ufunc allocates (MaterializeChildNumPyLayout: NpyIter's K-order axis permutation, which for C / F operands
            // is exactly what EvaluateCore picks on its own) — and run the schedule over its dense buffer in MEMORY
            // order, the order NumPy's K-order reduce iterator coalesces a dense block into. Both arrays are released as
            // soon as the reduction has read them.
            var view = MaterializeChildNumPyLayout(child, inputs, ops, out var buffer);
            try
            {
                byte* m = (byte*)buffer.Address + (long)buffer.Shape.offset * t.SizeOf();
                NumPyMinMaxReduce.ReduceContiguous(t, isMax, m, n, slot);
            }
            finally
            {
                if (!ReferenceEquals(view, buffer))
                    view.Dispose();
                buffer.Dispose();
            }

            NDExpr.ExactMinMaxRuns++;
            return true;
        }

        /// <summary>
        /// The streamed form of <see cref="NumPyMinMaxReduce.Reduce{T,TLane}"/>: bind <paramref name="child"/> for
        /// streaming, evaluate element 0 (the seed), then the rest in 8 KB blocks, and write the result to
        /// <paramref name="slot"/>. Every block but the last holds a whole number of 8-vector groups (8 KB is a
        /// multiple of 8 × 32 bytes for every lane width), so the groups never straddle a block and the fold order is
        /// the single-buffer schedule's exactly.
        /// </summary>
        /// <param name="child">The child program (validated by <see cref="CanStreamChild"/>).</param>
        /// <param name="inputs">Every input of the call, in input order (for the parameter block).</param>
        /// <param name="ops">The child's iterator operands.</param>
        /// <param name="n">The child's element count (&gt; 0).</param>
        /// <param name="isMax">Max when true, min when false.</param>
        /// <param name="t">The child dtype (satisfies <see cref="NumPyMinMaxReduce.Supports"/>).</param>
        /// <param name="slot">Receives the result at dtype <paramref name="t"/>.</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        private static unsafe void StreamExactMinMax(NDExprProgram child, NDArray[] inputs, NDArray[] ops, long n,
            bool isMax, NPTypeCode t, byte* slot)
        {
            var stream = new NDExprChildStream { Kernel = child.Kernel };
            int nop = ops.Length;
            byte** bases = stackalloc byte*[nop];
            long* elemBytes = stackalloc long[nop];
            void** ptrs = stackalloc void*[nop + 1];
            long* strides = stackalloc long[nop + 1];
            BindChildStream(ref stream, ops, t, bases, elemBytes, ptrs, strides);

            if (child.ParamCount > 0)
            {
                // Elementwise aux layout (slot 0 onward); stackalloc lives until the method returns.
                byte* paramBlock = stackalloc byte[NDExprParamPlan.SlotBytes * child.ParamCount];
                child.PackParams(inputs, paramBlock);
                stream.Aux = paramBlock;
            }

            byte* scratch = stackalloc byte[EvaluateStreamScratchBytes];
            stream.Scratch = scratch;
            stream.Block = EvaluateStreamScratchBytes / t.SizeOf();

            switch (t)
            {
                case NPTypeCode.Double: *(double*)slot = isMax ? StreamMinMax<double, NumPyMinMaxReduce.MaxLane<double>>(ref stream, 0, n, scratch) : StreamMinMax<double, NumPyMinMaxReduce.MinLane<double>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.Single: *(float*)slot = isMax ? StreamMinMax<float, NumPyMinMaxReduce.MaxLane<float>>(ref stream, 0, n, scratch) : StreamMinMax<float, NumPyMinMaxReduce.MinLane<float>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.SByte: *(sbyte*)slot = isMax ? StreamMinMax<sbyte, NumPyMinMaxReduce.MaxLane<sbyte>>(ref stream, 0, n, scratch) : StreamMinMax<sbyte, NumPyMinMaxReduce.MinLane<sbyte>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.Byte: *slot = isMax ? StreamMinMax<byte, NumPyMinMaxReduce.MaxLane<byte>>(ref stream, 0, n, scratch) : StreamMinMax<byte, NumPyMinMaxReduce.MinLane<byte>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.Int16: *(short*)slot = isMax ? StreamMinMax<short, NumPyMinMaxReduce.MaxLane<short>>(ref stream, 0, n, scratch) : StreamMinMax<short, NumPyMinMaxReduce.MinLane<short>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.UInt16: *(ushort*)slot = isMax ? StreamMinMax<ushort, NumPyMinMaxReduce.MaxLane<ushort>>(ref stream, 0, n, scratch) : StreamMinMax<ushort, NumPyMinMaxReduce.MinLane<ushort>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.Int32: *(int*)slot = isMax ? StreamMinMax<int, NumPyMinMaxReduce.MaxLane<int>>(ref stream, 0, n, scratch) : StreamMinMax<int, NumPyMinMaxReduce.MinLane<int>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.UInt32: *(uint*)slot = isMax ? StreamMinMax<uint, NumPyMinMaxReduce.MaxLane<uint>>(ref stream, 0, n, scratch) : StreamMinMax<uint, NumPyMinMaxReduce.MinLane<uint>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.Int64: *(long*)slot = isMax ? StreamMinMax<long, NumPyMinMaxReduce.MaxLane<long>>(ref stream, 0, n, scratch) : StreamMinMax<long, NumPyMinMaxReduce.MinLane<long>>(ref stream, 0, n, scratch); break;
                case NPTypeCode.UInt64: *(ulong*)slot = isMax ? StreamMinMax<ulong, NumPyMinMaxReduce.MaxLane<ulong>>(ref stream, 0, n, scratch) : StreamMinMax<ulong, NumPyMinMaxReduce.MinLane<ulong>>(ref stream, 0, n, scratch); break;
                default: throw new NotSupportedException($"NumPy min/max schedule: dtype {t} is not served.");
            }
        }

        /// <summary>
        /// Drive the schedule over the child elements <c>[start, start + n)</c> of a bound stream — the whole child for a
        /// flat reduction (<paramref name="start"/> 0), one long ROW for an axis reduction: element <paramref name="start"/>
        /// seeds <c>splat(x[0])</c>, full blocks fold whole groups (<see cref="NumPyMinMaxReduce.FoldGroups{T,TLane}"/>),
        /// the final (possibly partial) block finishes the schedule (<see cref="NumPyMinMaxReduce.Finish{T,TLane}"/>).
        /// </summary>
        /// <remarks>
        /// The blocks are counted from the element AFTER the seed, so every block but the last holds a whole number of
        /// 8-vector groups whatever <paramref name="start"/> is — the groups line up with the single-buffer schedule over
        /// the same range, which is what keeps a streamed row bit-identical to NumPy's one <c>simd_reduce_c</c> call.
        /// </remarks>
        /// <typeparam name="T">The child element type.</typeparam>
        /// <typeparam name="TLane">Max or min.</typeparam>
        /// <param name="stream">The bound stream (its <c>Block</c> is the scratch capacity in elements).</param>
        /// <param name="start">Flat child index of the range's first element (the seed).</param>
        /// <param name="n">The range's element count (&gt; 0).</param>
        /// <param name="scratch">The stream's scratch block.</param>
        /// <returns>The reduction result.</returns>
        private static unsafe T StreamMinMax<T, TLane>(ref NDExprChildStream stream, long start, long n, byte* scratch)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            stream.Produce(start, 1, scratch);
            T seed = *(T*)scratch;
            if (n == 1)
                return seed;   // nothing after the copied first element: NumPy returns it untouched

            var acc = Vector256.Create(seed);
            long block = stream.Block;               // a whole number of 8-vector groups (see the summary)
            long groupElems = Vector256<T>.Count * 8L;
            for (long k = 1; ; k += block)
            {
                long m = Math.Min(block, n - k);
                stream.Produce(start + k, m, scratch);
                if (k + m >= n)
                    return NumPyMinMaxReduce.Finish<T, TLane>(acc, (T*)scratch, m);
                acc = NumPyMinMaxReduce.FoldGroups<T, TLane>(acc, (T*)scratch, m / groupElems);
            }
        }
    }
}

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        /// <summary>
        /// Test / diagnostics hook: when set on the current thread, the NumPy-exact min/max schedules are SKIPPED —
        /// np.evaluate's flat <c>Min</c> / <c>Max</c> (<c>DefaultEngine.TryExactFlatMinMax</c>) folds with the
        /// 4-accumulator scalar kernel, the engine's flat <c>np.max</c> / <c>np.min</c> / <c>np.ptp</c>
        /// (<c>DefaultEngine.TryExactFlatMinMaxArray</c>) take the IL kernels, and every AXIS <c>Min</c> / <c>Max</c>
        /// (np.evaluate's and the engine's, <c>DefaultEngine.TryExactAxisMinMax</c>) takes the kernels it used before —
        /// value-identical, but not NumPy's ±0-tie / NaN-payload bits. Exists so a probe can time the old paths and a
        /// test can show the two differ exactly where the schedule is observable. Thread-static so a parallel test
        /// never perturbs another's run.
        /// </summary>
        [System.ThreadStatic] internal static bool DisableExactMinMax;

        /// <summary>
        /// Test / diagnostics hook: incremented (on the current thread) each time a <c>Min</c> / <c>Max</c> — flat or
        /// axis, np.evaluate or engine — is computed by a NumPy-exact schedule (any route: in place, streamed or
        /// materialized), so a test can assert the route ENGAGED instead of silently falling back.
        /// </summary>
        [System.ThreadStatic] internal static int ExactMinMaxRuns;
    }
}
