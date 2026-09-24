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
// The NaN-SUPPRESSING rules (np.nanmax / np.nanmin = fmax / fmin.reduce, the "P rules") compute the vector section
// differently but land on the same bits: their vector op maxp is an associative selection ("the LAST largest non-NaN,
// else the FIRST NaN"), so the group tree and the single-vector loop are one order-free fold per lane. FoldVectorsP runs
// it as a BACKWARD scan with the plain op, data first — vmaxp(data, suffix) skips a NaN and lets the later element win a
// tie, one instruction per vector where the tree needed a NaN proof per group — over the 32-byte-ALIGNED vectors covering
// the section, rotating positions back to NumPy's lanes at the end. The alignment matters because the section starts at
// x[1] (x[0] is the seed): NumPy's own loads split a cache line every other vector, which on an aligned buffer cost the
// tree 1.4x the old value-only kernel's time in L1/L2. The N rules keep the literal tree (FoldGroupsTree), and so does a
// P-rule stretch shorter than FoldVectorsPMinElements (768 elements): the fold costs a few ns per call, which a row
// reduction pays per ROW, so rows below that length ran 0.5-0.97x the tree's speed and keep the tree's exact route.
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
    /// Which NumPy min/max-family ufunc a NumPy-exact reduction reproduces — the choice picks the lane rule
    /// (<see cref="NumPyMinMaxReduce.ILane{T}"/>) and, for floats, decides the result's bits.
    /// </summary>
    /// <remarks>
    /// <see cref="Max"/> / <see cref="Min"/> are <c>np.maximum.reduce</c> / <c>np.minimum.reduce</c> (what <c>np.max</c> /
    /// <c>np.min</c> run — NaN-propagating); <see cref="FMax"/> / <see cref="FMin"/> are <c>np.fmax.reduce</c> /
    /// <c>np.fmin.reduce</c> (what <c>np.nanmax</c> / <c>np.nanmin</c> run on an ndarray — NaN-suppressing, and with a
    /// vector op and a scalar op that DISAGREE on a ±0 tie and on which of two NaNs survives, so their schedule places
    /// the vector/scalar split per element). For integer dtypes each pair coincides (no NaN, no bit-distinct ties), so the
    /// dtype dispatchers serve <see cref="FMax"/> / <see cref="FMin"/> through the <see cref="Max"/> / <see cref="Min"/>
    /// lanes — one fewer generic instantiation per width, identical results.
    /// </remarks>
    internal enum MinMaxOp : byte
    {
        /// <summary><c>np.maximum.reduce</c> — <c>np.max</c> / <c>np.amax</c> (NaN-propagating).</summary>
        Max,

        /// <summary><c>np.minimum.reduce</c> — <c>np.min</c> / <c>np.amin</c> (NaN-propagating).</summary>
        Min,

        /// <summary><c>np.fmax.reduce</c> — <c>np.nanmax</c> (a NaN only survives an all-NaN reduction).</summary>
        FMax,

        /// <summary><c>np.fmin.reduce</c> — <c>np.nanmin</c> (a NaN only survives an all-NaN reduction).</summary>
        FMin,
    }

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
        /// True for the maximum-side ops (<see cref="MinMaxOp.Max"/>, <see cref="MinMaxOp.FMax"/>) — the side an integer
        /// dispatch folds onto, since the NaN-suppressing twin of an integer max IS the max.
        /// </summary>
        /// <param name="op">The reduction op.</param>
        /// <returns>Whether <paramref name="op"/> keeps the larger operand.</returns>
        internal static bool IsMaxLike(MinMaxOp op) => op is MinMaxOp.Max or MinMaxOp.FMax;

        /// <summary>
        /// The NumPy ufunc name of <paramref name="op"/> — what NumPy's "zero-size array to reduction operation
        /// &lt;name&gt; which has no identity" error prints (<c>maximum</c> for <c>np.max</c>, <c>fmax</c> for
        /// <c>np.nanmax</c>, even for an integer array).
        /// </summary>
        /// <param name="op">The reduction op.</param>
        /// <returns><c>maximum</c>, <c>minimum</c>, <c>fmax</c> or <c>fmin</c>.</returns>
        internal static string UfuncName(MinMaxOp op) => op switch
        {
            MinMaxOp.Max => "maximum",
            MinMaxOp.Min => "minimum",
            MinMaxOp.FMax => "fmax",
            _ => "fmin",
        };

        /// <summary>
        /// One lane rule of <c>loops_minmax.dispatch.c.src</c> — the three operations a NumPy min/max-family loop is
        /// built from: the VECTOR op (<c>V_INTRIN</c>), the horizontal reduce of the accumulator
        /// (<c>V_REDUCE_INTRIN</c>) and the SCALAR op (<c>SCALAR_OP</c>). <c>np.max</c> / <c>np.min</c>
        /// (<see cref="MaxLane{T}"/> / <see cref="MinLane{T}"/>) use the NaN-PROPAGATING <c>maxn</c> / <c>minn</c> rule, whose
        /// vector and scalar ops agree; <c>np.fmax</c> / <c>np.fmin</c> — and so <c>np.nanmax</c> / <c>np.nanmin</c>, which ARE
        /// <c>fmax.reduce</c> / <c>fmin.reduce</c> — use the NaN-SUPPRESSING <c>maxp</c> / <c>minp</c> rule
        /// (<see cref="FMaxLane{T}"/> / <see cref="FMinLane{T}"/>), whose vector op and scalar op (the MSVC CRT
        /// <c>fmax</c> / <c>fmin</c>) DISAGREE on a ±0 tie and on which of two NaNs survives. "a beats b" is <c>a &gt; b</c>
        /// for max and <c>a &lt; b</c> for min: ordered, so false when either side is NaN.
        /// </summary>
        /// <remarks>
        /// Consequence of the disagreement: for the P rules, WHICH elements a NumPy loop call hands to the vector op and
        /// which to its scalar tail is part of the answer's bits — a schedule that is exact for <c>np.max</c> (where the
        /// split is unobservable) must also place that split exactly for <c>np.nanmax</c>.
        /// </remarks>
        /// <typeparam name="T">The lane type.</typeparam>
        internal interface ILane<T> where T : unmanaged, INumber<T>
        {
            /// <summary>
            /// The loop's VECTOR op per lane. <c>npyv_maxn</c> / <c>npyv_minn</c> (N rules): <paramref name="a"/> where it
            /// is NaN or beats <paramref name="b"/>, else <paramref name="b"/> — NaN-propagating, the second operand winning
            /// a tie. <c>npyv_maxp</c> / <c>npyv_minp</c> (P rules): <paramref name="a"/> where <paramref name="b"/> is NaN or
            /// <paramref name="a"/> beats it, else <paramref name="b"/> — a NaN is dropped for the other operand, two NaNs
            /// keep the FIRST, the second operand still wins a tie.
            /// </summary>
            /// <param name="a">First operand (the running accumulator side).</param>
            /// <param name="b">Second operand (the incoming side).</param>
            /// <returns>The per-lane result.</returns>
            static abstract Vector256<T> N(Vector256<T> a, Vector256<T> b);

            /// <summary>
            /// True for the NaN-propagating <c>maxn</c> / <c>minn</c> rules (<c>np.max</c> / <c>np.min</c>): the horizontal
            /// reduce returns the canonical +NaN when any lane is NaN, and a contiguous call re-canonicalizes a NaN seed.
            /// False for the NaN-suppressing <c>maxp</c> / <c>minp</c> rules (<c>np.fmax</c> / <c>np.nanmax</c>): an all-NaN
            /// accumulator returns lane 0's OWN bits, otherwise NaN lanes are replaced by <see cref="NaNLaneFill"/>.
            /// A JIT-time constant per lane type, so the untaken branch compiles away.
            /// </summary>
            static abstract bool PropagatesNaN { get; }

            /// <summary>
            /// The value a NaN lane is replaced with before the horizontal cascade of a NaN-SUPPRESSING rule — <c>-inf</c>
            /// for max, <c>+inf</c> for min (<c>npyv_reduce_maxp</c>'s <c>npyv_select(notnan, a, ∓inf)</c>). Never read for
            /// a NaN-propagating rule.
            /// </summary>
            static abstract T NaNLaneFill { get; }

            /// <summary>
            /// The VECTOR op <see cref="N"/> on one lane — used where a lane of NumPy's vector section is computed by
            /// scalar code (a strided run without a vector gather, the non-vector-aligned end of a vector section).
            /// </summary>
            /// <param name="a">First operand (the running side).</param>
            /// <param name="b">Second operand (the incoming side).</param>
            /// <returns>What the vector op would put in this lane.</returns>
            static abstract T V(T a, T b);

            /// <summary>
            /// The loop's SCALAR op (<c>SCALAR_OP</c>): the SSE <c>scalar_max_d</c> override for the N rules (the same
            /// rule as <see cref="V"/>); the MSVC CRT <c>fmax</c> / <c>fmin</c> for the P rules — a NaN side yields the
            /// OTHER operand unchanged (two NaNs: the SECOND, raw), and a ±0 tie is decided by sign (<c>fmax(±0, ∓0) = +0</c>,
            /// <c>fmin</c> gives <c>-0</c>) instead of by position.
            /// </summary>
            /// <param name="a">The running value.</param>
            /// <param name="b">The incoming element.</param>
            /// <returns>The scalar op's result.</returns>
            static abstract T S(T a, T b);

            /// <summary>
            /// The SCALAR op <see cref="S"/> applied lane by lane — for scalar-op passes that are vectorized without changing
            /// their result: NumPy's 8-accumulator strided unroll (each accumulator only ever meets its own lane's
            /// elements) and a slab region NumPy runs entirely through its scalar loop. Equal to <see cref="N"/> for the
            /// N rules.
            /// </summary>
            /// <param name="a">First operand.</param>
            /// <param name="b">Second operand.</param>
            /// <returns>The per-lane scalar-op result.</returns>
            static abstract Vector256<T> SV(Vector256<T> a, Vector256<T> b);

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
            public static bool PropagatesNaN => true;

            /// <inheritdoc />
            /// <remarks>Never read: a NaN-propagating rule canonicalizes a NaN accumulator instead of filling it.</remarks>
            public static T NaNLaneFill => default;

            /// <inheritdoc />
            /// <remarks><c>npyv_maxn</c> on one lane — the same rule as <see cref="S"/>.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T V(T a, T b) => T.IsNaN(a) || a > b ? a : b;

            /// <inheritdoc />
            /// <remarks>
            /// NumPy's SSE <c>scalar_max_d</c> (<c>_mm_max_sd</c> blended back to <paramref name="a"/> when it is NaN): the
            /// vector rule on one element, so a max loop's vector / scalar split is unobservable.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T S(T a, T b) => T.IsNaN(a) || a > b ? a : b;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> SV(Vector256<T> a, Vector256<T> b) => N(a, b);

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
            public static bool PropagatesNaN => true;

            /// <inheritdoc />
            /// <remarks>Never read: a NaN-propagating rule canonicalizes a NaN accumulator instead of filling it.</remarks>
            public static T NaNLaneFill => default;

            /// <inheritdoc />
            /// <remarks><c>npyv_minn</c> on one lane — the same rule as <see cref="S"/>.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T V(T a, T b) => T.IsNaN(a) || a < b ? a : b;

            /// <inheritdoc />
            /// <remarks>NumPy's SSE <c>scalar_min_d</c>: the vector rule on one element (see <see cref="MaxLane{T}.S"/>).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T S(T a, T b) => T.IsNaN(a) || a < b ? a : b;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> SV(Vector256<T> a, Vector256<T> b) => N(a, b);

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
        /// The <c>np.fmax</c> lane rule — what <c>np.nanmax</c> (= <c>fmax.reduce</c>) runs: the vector op is
        /// <c>npyv_maxp</c> (<c>blendv(a, max(a, b), ord(b, b))</c> — a NaN is dropped for the other operand, the second
        /// operand wins a tie), the horizontal reduce is <c>npyv_reduce_maxp</c> and the scalar op is the MSVC CRT
        /// <c>fmax</c>.
        /// </summary>
        /// <remarks>
        /// Both ops are associative SELECTIONS, but different ones: the vector op keeps "the LAST maximal non-NaN element,
        /// else the FIRST NaN"; the CRT op keeps "the maximum with -0 below +0, else the LAST NaN". Which op a NumPy call
        /// applies to an element is therefore visible in a ±0 result's sign and in which NaN payload comes back — the
        /// schedule helpers place every element where NumPy's loop does. Only instantiated for float32 / float64 (the
        /// integer <c>fmax</c> is plain <c>max</c>, served by <see cref="MaxLane{T}"/>); the integer branches below exist
        /// so the generic code compiles and stay exact if reached.
        /// </remarks>
        /// <typeparam name="T">The lane type (float or double).</typeparam>
        internal readonly struct FMaxLane<T> : ILane<T> where T : unmanaged, INumber<T>
        {
            /// <inheritdoc />
            /// <remarks>
            /// With AVX: NumPy's own three instructions, <c>vmaxp*</c> + <c>vcmpordp*</c> on <paramref name="b"/> +
            /// <c>vblendvp*</c> (RyuJIT never swaps the operands of a floating-point max/min intrinsic, so the second
            /// operand keeps winning a tie). Without AVX: the portable compare + select of the same rule.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> N(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                {
                    if (Avx.IsSupported)
                    {
                        var bd = b.AsDouble();
                        return Avx.BlendVariable(a.AsDouble(), Avx.Max(a.AsDouble(), bd), Avx.CompareOrdered(bd, bd)).As<double, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.GreaterThan(a, b) | ~Vector256.Equals(b, b), a, b);
                }

                if (typeof(T) == typeof(float))
                {
                    if (Avx.IsSupported)
                    {
                        var bf = b.AsSingle();
                        return Avx.BlendVariable(a.AsSingle(), Avx.Max(a.AsSingle(), bf), Avx.CompareOrdered(bf, bf)).As<float, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.GreaterThan(a, b) | ~Vector256.Equals(b, b), a, b);
                }

                return Vector256.Max(a, b);
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Beats(T a, T b) => a > b;

            /// <inheritdoc />
            /// <remarks><c>npyv_reduce_maxp</c>'s cascade is the plain <c>_mm_max_p*</c> one — the max rule's own.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector128<T> P128(Vector128<T> a, Vector128<T> b) => MaxLane<T>.P128(a, b);

            /// <inheritdoc />
            /// <remarks>
            /// The plain <c>vmaxp*</c>: equal to <see cref="N"/> whenever the SECOND operand is not NaN (a NaN first operand
            /// yields the second, exactly as <c>maxp</c> does) — which is why the NaN-free group fast path needs no
            /// NaN-free accumulator for this rule.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> P256(Vector256<T> a, Vector256<T> b) => MaxLane<T>.P256(a, b);

            /// <inheritdoc />
            public static bool PropagatesNaN => false;

            /// <inheritdoc />
            public static T NaNLaneFill => NegativeInfinity<T>();

            /// <inheritdoc />
            /// <remarks><c>npyv_maxp</c> on one lane: <paramref name="b"/> NaN → <paramref name="a"/> (so two NaNs keep the FIRST).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T V(T a, T b) => T.IsNaN(b) || a > b ? a : b;

            /// <inheritdoc />
            /// <remarks>
            /// The UCRT <c>fmax</c> NumPy's win-amd64 wheel calls (probed against NumPy 2.4.2): a NaN
            /// <paramref name="a"/> returns <paramref name="b"/> as is — even a second NaN, even a signalling one — a NaN
            /// <paramref name="b"/> returns <paramref name="a"/>, and an ordered tie returns the bitwise AND of the two, which
            /// is <c>+0</c> for a ±0 pair and the value itself otherwise (equal non-zero floats have identical bits).
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T S(T a, T b)
            {
                if (T.IsNaN(a))
                    return b;
                if (T.IsNaN(b))
                    return a;
                if (a > b)
                    return a;
                if (b > a)
                    return b;
                return BitAnd(a, b);
            }

            /// <inheritdoc />
            /// <remarks>
            /// With AVX: <c>vmaxp*</c> already returns <paramref name="b"/> for a NaN <paramref name="a"/> (the CRT rule);
            /// a blend then restores <paramref name="a"/> where only <paramref name="b"/> is NaN, and a second blend puts the
            /// bitwise AND on every ordered-equal lane (the ±0 tie). The portable form spells the same three decisions.
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> SV(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double) && Avx.IsSupported)
                {
                    var ad = a.AsDouble();
                    var bd = b.AsDouble();
                    var m = Avx.Max(ad, bd);
                    // Avx.AndNot(x, y) = ~x & y: lanes where a is ordered and b is NaN keep a.
                    m = Avx.BlendVariable(m, ad, Avx.AndNot(Avx.CompareUnordered(ad, ad), Avx.CompareUnordered(bd, bd)));
                    return Avx.BlendVariable(m, Avx.And(ad, bd), Avx.CompareEqual(ad, bd)).As<double, T>();
                }

                if (typeof(T) == typeof(float) && Avx.IsSupported)
                {
                    var af = a.AsSingle();
                    var bf = b.AsSingle();
                    var m = Avx.Max(af, bf);
                    m = Avx.BlendVariable(m, af, Avx.AndNot(Avx.CompareUnordered(af, af), Avx.CompareUnordered(bf, bf)));
                    return Avx.BlendVariable(m, Avx.And(af, bf), Avx.CompareEqual(af, bf)).As<float, T>();
                }

                // Vector256.AndNot(x, y) = x & ~y: a ordered AND b NaN.
                var pickA = Vector256.GreaterThan(a, b) | Vector256.AndNot(Vector256.Equals(a, a), Vector256.Equals(b, b));
                var r = Vector256.ConditionalSelect(pickA, a, b);
                return Vector256.ConditionalSelect(Vector256.Equals(a, b), a & b, r);
            }
        }

        /// <summary>
        /// The <c>np.fmin</c> lane rule — what <c>np.nanmin</c> (= <c>fmin.reduce</c>) runs: <c>npyv_minp</c>,
        /// <c>npyv_reduce_minp</c> and the MSVC CRT <c>fmin</c>. The mirror of <see cref="FMaxLane{T}"/>; the CRT tie rule
        /// gives <c>-0</c> for a ±0 pair (the bitwise OR).
        /// </summary>
        /// <typeparam name="T">The lane type (float or double).</typeparam>
        internal readonly struct FMinLane<T> : ILane<T> where T : unmanaged, INumber<T>
        {
            /// <inheritdoc />
            /// <remarks>The mirror of <see cref="FMaxLane{T}.N"/> (<c>vminp*</c> + <c>vcmpordp*</c> on b + <c>vblendvp*</c>).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> N(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double))
                {
                    if (Avx.IsSupported)
                    {
                        var bd = b.AsDouble();
                        return Avx.BlendVariable(a.AsDouble(), Avx.Min(a.AsDouble(), bd), Avx.CompareOrdered(bd, bd)).As<double, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.LessThan(a, b) | ~Vector256.Equals(b, b), a, b);
                }

                if (typeof(T) == typeof(float))
                {
                    if (Avx.IsSupported)
                    {
                        var bf = b.AsSingle();
                        return Avx.BlendVariable(a.AsSingle(), Avx.Min(a.AsSingle(), bf), Avx.CompareOrdered(bf, bf)).As<float, T>();
                    }

                    return Vector256.ConditionalSelect(Vector256.LessThan(a, b) | ~Vector256.Equals(b, b), a, b);
                }

                return Vector256.Min(a, b);
            }

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static bool Beats(T a, T b) => a < b;

            /// <inheritdoc />
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector128<T> P128(Vector128<T> a, Vector128<T> b) => MinLane<T>.P128(a, b);

            /// <inheritdoc />
            /// <remarks>The plain <c>vminp*</c> — equal to <see cref="N"/> whenever the second operand is not NaN.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> P256(Vector256<T> a, Vector256<T> b) => MinLane<T>.P256(a, b);

            /// <inheritdoc />
            public static bool PropagatesNaN => false;

            /// <inheritdoc />
            public static T NaNLaneFill => PositiveInfinity<T>();

            /// <inheritdoc />
            /// <remarks><c>npyv_minp</c> on one lane.</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T V(T a, T b) => T.IsNaN(b) || a < b ? a : b;

            /// <inheritdoc />
            /// <remarks>
            /// The UCRT <c>fmin</c>: the NaN rules of <see cref="FMaxLane{T}.S"/>, and an ordered tie returns the bitwise OR
            /// (<c>-0</c> for a ±0 pair).
            /// </remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static T S(T a, T b)
            {
                if (T.IsNaN(a))
                    return b;
                if (T.IsNaN(b))
                    return a;
                if (a < b)
                    return a;
                if (b < a)
                    return b;
                return BitOr(a, b);
            }

            /// <inheritdoc />
            /// <remarks>The mirror of <see cref="FMaxLane{T}.SV"/> (<c>vminp*</c>, the OR on ordered-equal lanes).</remarks>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public static Vector256<T> SV(Vector256<T> a, Vector256<T> b)
            {
                if (typeof(T) == typeof(double) && Avx.IsSupported)
                {
                    var ad = a.AsDouble();
                    var bd = b.AsDouble();
                    var m = Avx.Min(ad, bd);
                    m = Avx.BlendVariable(m, ad, Avx.AndNot(Avx.CompareUnordered(ad, ad), Avx.CompareUnordered(bd, bd)));
                    return Avx.BlendVariable(m, Avx.Or(ad, bd), Avx.CompareEqual(ad, bd)).As<double, T>();
                }

                if (typeof(T) == typeof(float) && Avx.IsSupported)
                {
                    var af = a.AsSingle();
                    var bf = b.AsSingle();
                    var m = Avx.Min(af, bf);
                    m = Avx.BlendVariable(m, af, Avx.AndNot(Avx.CompareUnordered(af, af), Avx.CompareUnordered(bf, bf)));
                    return Avx.BlendVariable(m, Avx.Or(af, bf), Avx.CompareEqual(af, bf)).As<float, T>();
                }

                var pickA = Vector256.LessThan(a, b) | Vector256.AndNot(Vector256.Equals(a, a), Vector256.Equals(b, b));
                var r = Vector256.ConditionalSelect(pickA, a, b);
                return Vector256.ConditionalSelect(Vector256.Equals(a, b), a | b, r);
            }
        }

        /// <summary><c>-inf</c> of a float lane type (<c>default</c> for an integer one, which never asks).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <returns>Negative infinity.</returns>
        private static T NegativeInfinity<T>() where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(double))
            {
                double d = double.NegativeInfinity;
                return Unsafe.As<double, T>(ref d);
            }

            if (typeof(T) == typeof(float))
            {
                float f = float.NegativeInfinity;
                return Unsafe.As<float, T>(ref f);
            }

            return default;
        }

        /// <summary><c>+inf</c> of a float lane type (<c>default</c> for an integer one, which never asks).</summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <returns>Positive infinity.</returns>
        private static T PositiveInfinity<T>() where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(double))
            {
                double d = double.PositiveInfinity;
                return Unsafe.As<double, T>(ref d);
            }

            if (typeof(T) == typeof(float))
            {
                float f = float.PositiveInfinity;
                return Unsafe.As<float, T>(ref f);
            }

            return default;
        }

        /// <summary>
        /// The bitwise AND of two ordered-equal floats — the CRT <c>fmax</c> tie result: <c>+0</c> unless both are
        /// <c>-0</c>, the shared bits otherwise. For a non-float lane type the values are identical and <paramref name="a"/>
        /// is returned.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="a">First value.</param>
        /// <param name="b">Second value (ordered-equal to <paramref name="a"/>).</param>
        /// <returns>The AND of the bit patterns.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T BitAnd<T>(T a, T b) where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(double))
            {
                long bits = BitConverter.DoubleToInt64Bits(Unsafe.As<T, double>(ref a)) & BitConverter.DoubleToInt64Bits(Unsafe.As<T, double>(ref b));
                double d = BitConverter.Int64BitsToDouble(bits);
                return Unsafe.As<double, T>(ref d);
            }

            if (typeof(T) == typeof(float))
            {
                int bits = BitConverter.SingleToInt32Bits(Unsafe.As<T, float>(ref a)) & BitConverter.SingleToInt32Bits(Unsafe.As<T, float>(ref b));
                float f = BitConverter.Int32BitsToSingle(bits);
                return Unsafe.As<float, T>(ref f);
            }

            return a;
        }

        /// <summary>
        /// The bitwise OR of two ordered-equal floats — the CRT <c>fmin</c> tie result: <c>-0</c> unless both are
        /// <c>+0</c>, the shared bits otherwise. For a non-float lane type <paramref name="a"/> is returned.
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <param name="a">First value.</param>
        /// <param name="b">Second value (ordered-equal to <paramref name="a"/>).</param>
        /// <returns>The OR of the bit patterns.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T BitOr<T>(T a, T b) where T : unmanaged, INumber<T>
        {
            if (typeof(T) == typeof(double))
            {
                long bits = BitConverter.DoubleToInt64Bits(Unsafe.As<T, double>(ref a)) | BitConverter.DoubleToInt64Bits(Unsafe.As<T, double>(ref b));
                double d = BitConverter.Int64BitsToDouble(bits);
                return Unsafe.As<double, T>(ref d);
            }

            if (typeof(T) == typeof(float))
            {
                int bits = BitConverter.SingleToInt32Bits(Unsafe.As<T, float>(ref a)) | BitConverter.SingleToInt32Bits(Unsafe.As<T, float>(ref b));
                float f = BitConverter.Int32BitsToSingle(bits);
                return Unsafe.As<float, T>(ref f);
            }

            return a;
        }

        /// <summary>
        /// <c>npyv_maxn</c> / <c>npyv_minn</c> through the lane rule (see <see cref="ILane{T}.N"/>).
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="a">First operand (the running accumulator side).</param>
        /// <param name="b">Second operand.</param>
        /// <returns>The per-lane result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<T> N<T, TLane>(Vector256<T> a, Vector256<T> b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.N(a, b);

        /// <summary>
        /// The loop's SCALAR op (<c>SCALAR_OP</c>, see <see cref="ILane{T}.S"/>): the SSE <c>scalar_max_d</c> for the N rules
        /// — the same rule as their vector lanes — and the MSVC CRT <c>fmax</c> / <c>fmin</c> for the P rules. Every scalar
        /// step of every schedule goes through here: the tail of a contiguous reduce call, the 8-accumulator unroll of a
        /// strided one, and the scalar tail of an elementwise SLAB call.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="a">The running value (NumPy's first operand).</param>
        /// <param name="b">The incoming element.</param>
        /// <returns>The lane rule's scalar result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static T N<T, TLane>(T a, T b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.S(a, b);

        /// <summary>
        /// The plain <c>_mm_max_p*</c> / <c>_mm_min_p*</c> lane (no NaN blend) the horizontal step uses — only ever
        /// applied after the caller ruled NaN out.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="a">First operand (the lower half's lane).</param>
        /// <param name="b">Second operand (the upper half's lane) — wins a tie.</param>
        /// <returns>The winning operand.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T P<T, TLane>(T a, T b)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.Beats(a, b) ? a : b;

        /// <summary>
        /// Phase 1 of the schedule: fold <paramref name="groups"/> whole 8-vector groups into
        /// <paramref name="acc"/>, bit-identical to NumPy's <c>acc = N(acc, N(N(N(v0,v1), N(v2,v3)), N(N(v4,v5), N(v6,v7))))</c>
        /// per group. The N rules (<c>np.max</c> / <c>np.min</c>, and every integer lane) run that tree literally
        /// (<see cref="FoldGroupsTree{T,TLane}"/>); the P rules (<c>np.nanmax</c> / <c>np.nanmin</c>) take the order-free
        /// aligned backward fold (<see cref="FoldVectorsP{T,TLane}"/>), which lands on the same bits without a per-group
        /// NaN proof — once the run is long enough (<see cref="FoldVectorsPMinElements"/>) for that fold's fixed cost to pay.
        /// </summary>
        /// <remarks>
        /// The dispatch is on <see cref="ILane{T}.PropagatesNaN"/>, a JIT-time constant per lane type, so an N-rule or integer
        /// instantiation compiles to exactly the tree fold it ran before the P-rule fold existed — nothing but the
        /// <c>np.nanmax</c> / <c>np.nanmin</c> family changes route, and it changes route only for runs of at least
        /// <see cref="FoldVectorsPMinElements"/> elements; a shorter P-rule run keeps the tree, the route it took before. Both
        /// routes are bit-identical, so the threshold is a pure speed choice.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="acc">The running accumulator vector.</param>
        /// <param name="ip">First element of the first group (unaligned is fine).</param>
        /// <param name="groups">Number of whole groups (8 × <see cref="Vector256{T}.Count"/> elements each) to fold.</param>
        /// <returns>The updated accumulator — NumPy's, lane for lane and bit for bit.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe Vector256<T> FoldGroups<T, TLane>(Vector256<T> acc, T* ip, long groups)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => TLane.PropagatesNaN || groups * 8 * Vector256<T>.Count < FoldVectorsPMinElements
                ? FoldGroupsTree<T, TLane>(acc, ip, groups)
                : FoldVectorsP<T, TLane>(acc, ip, groups * 8);

        /// <summary>
        /// NumPy's group tree, literally: fold <paramref name="groups"/> whole 8-vector groups into <paramref name="acc"/>,
        /// each as <c>acc = N(acc, N(N(N(v0,v1), N(v2,v3)), N(N(v4,v5), N(v6,v7))))</c>, with a NaN-free fast path that
        /// runs a group proven NaN-free with the plain lane op. What <see cref="FoldGroups{T,TLane}"/> runs for the N rules;
        /// for the P rules it is the tree <see cref="FoldVectorsP{T,TLane}"/> must equal bit for bit, and the tests compare
        /// the two directly.
        /// </summary>
        /// <remarks>
        /// The P rules' production path only for runs shorter than <see cref="FoldVectorsPMinElements"/> elements. On a long run
        /// it loses to <see cref="FoldVectorsP{T,TLane}"/>: it reads NumPy's vector section starting one element past an
        /// array's base (element 0 is the seed), so on a 32-byte-aligned buffer every other 32-byte load splits a cache line
        /// — the cause of the in-cache gap to the old value-only kernel that the aligned, order-free fold closes. On a short
        /// run that per-vector penalty is smaller than the aligned fold's fixed cost, so the tree stays.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="acc">The running accumulator vector.</param>
        /// <param name="ip">First element of the first group (unaligned is fine).</param>
        /// <param name="groups">Number of whole groups (8 × <see cref="Vector256{T}.Count"/> elements each) to fold.</param>
        /// <returns>The updated accumulator.</returns>
        internal static unsafe Vector256<T> FoldGroupsTree<T, TLane>(Vector256<T> acc, T* ip, long groups)
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
            // The P rules (fmax / fmin) need no NaN-free accumulator: maxp(a, b) = blend(a, max(a, b), ord(b, b)) equals
            // the plain max(a, b) whenever the SECOND operand is NaN-free — a NaN first operand makes vmaxp* return the
            // second, exactly as maxp does — and every second operand of a NaN-free group is NaN-free. So for them the
            // proof per group alone licenses the plain op, even while the accumulator holds a NaN (an all-NaN prefix).
            // The same fact is why a NaN group does NOT end the fast path for them (see the break below): the group after
            // it is licensed by its own proof again, whatever the NaN group left in the accumulator.
            if ((typeof(T) == typeof(double) || typeof(T) == typeof(float)) && Avx.IsSupported
                && (!TLane.PropagatesNaN || Vector256.EqualsAll(acc, acc)))
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
                    {
                        // N rules (np.max / np.min): this group may make the accumulator NaN, and a NaN FIRST operand is
                        // exactly where the plain op differs from NumPy's — so every group from here on must take the
                        // blended rule. Leave with g / ip still on this group; the loop below folds it.
                        if (TLane.PropagatesNaN)
                            break;

                        // P rules (np.nanmax / np.nanmin): only THIS group needs NumPy's blended maxp / minp. Fold it with
                        // the full rule — the same tree the loop below builds — and stay on the fast path: the next group
                        // is licensed by its own NaN-free proof whatever this one leaves in the accumulator. Without this,
                        // one NaN (a NaN in the very first group included) put the whole rest of the run on the three-
                        // instruction rule — measured 2.2x slower than the old value-exact kernel on NaN-laced data.
                        var b01 = N<T, TLane>(v0, v1);
                        var b23 = N<T, TLane>(v2, v3);
                        var b45 = N<T, TLane>(v4, v5);
                        var b67 = N<T, TLane>(v6, v7);
                        acc = N<T, TLane>(acc, N<T, TLane>(N<T, TLane>(b01, b23), N<T, TLane>(b45, b67)));
                        continue;
                    }
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
        /// Below this many whole vectors <see cref="FoldVectorsP{T,TLane}"/> reads NumPy's lanes directly (unaligned) instead
        /// of through the aligned-position remap: with fewer than one group the fold is one short backward chain whose
        /// latency, not its few split loads, sets the cost, and the remap's two masked edge vectors plus the lane rotation
        /// would only add work.
        /// </summary>
        private const long AlignedFoldMinVectors = 8;

        /// <summary>
        /// The shortest P-rule stretch (in ELEMENTS) that <see cref="Finish{T,TLane}"/> and <see cref="FoldGroups{T,TLane}"/>
        /// hand to <see cref="FoldVectorsP{T,TLane}"/>; a shorter one keeps NumPy's tree (<see cref="FoldGroupsTree{T,TLane}"/>
        /// + the single-vector loop), the exact route every P-rule section took before the order-free fold existed. Both
        /// routes produce the same bits, so this only picks the faster one — and below it nothing changed at all.
        /// </summary>
        /// <remarks>
        /// <para>
        /// The aligned fold saves a split cache-line load on every other vector, but costs a fixed amount per call: its
        /// out-of-line frames (the eight chains need callee-saved vector registers, which the Windows x64 ABI makes them
        /// spill and restore), the chunk setup, the two edge vectors and the lane rotation — about 1.5–5 ns. A flat
        /// reduction pays that once; a ROW reduction pays it once per row, and there the tree wins short rows.
        /// </para>
        /// <para>
        /// Measured (np.nanmax(a, axis=1) over C-contiguous (R, L) blocks of 256K elements, pinned, min of 4 runs per side,
        /// tree vs fold time per row): up to 64 float32 / 128 float64 vectors — 512 elements — the fold ran 0.49–0.97x the
        /// tree's speed; from 96 float32 / 192 float64 vectors — 768 elements — on it ran 1.03–1.22x. The crossover sat at
        /// about the same ELEMENT count for both widths (not the same vector count), so the threshold counts elements. It
        /// sits at the first measured WINNING size rather than the interpolated crossover: every shorter stretch keeps the
        /// old route exactly (a length between the two samples might gain a percent or two from the fold, or lose as much),
        /// and every longer one is past the point where the fold measured faster.
        /// </para>
        /// </remarks>
        internal const long FoldVectorsPMinElements = 768;

        /// <summary>
        /// The P rules' vector section — NumPy's 8-vector group tree followed by its single-vector loop — over
        /// <paramref name="vectors"/> whole vectors from <paramref name="ip"/>, computed as ONE order-free fold: bit-identical
        /// to <see cref="FoldGroupsTree{T,TLane}"/> + the loop, at the old value-only kernel's cost (one load and one plain
        /// <c>vmaxp</c> / <c>vminp</c> per vector, no NaN proof, no data-dependent branch).
        /// </summary>
        /// <remarks>
        /// <para>
        /// WHY IT IS EXACT. NumPy's vector op <c>maxp(a, b) = isnan(b) ? a : (a &gt; b ? a : b)</c> is an associative
        /// SELECTION: folded over an ordered sequence in ANY bracketing it returns "the LAST element holding the largest
        /// non-NaN value (±0 compare equal), or the FIRST element when every one is NaN". NumPy's tree and single-vector loop
        /// only ever combine one lane's elements in increasing position, so accumulator lane <c>j</c> after the vector section
        /// is that selection over <c>[acc_j, ip[j], ip[j + Count], ip[j + 2·Count], …]</c> — whatever the tree's shape.
        /// </para>
        /// <para>
        /// WHY IT IS ONE INSTRUCTION PER VECTOR. Scanned BACKWARD, the plain op <c>vmaxp(data, suffix)</c> computes exactly
        /// that selection over the suffix, starting from the fill <see cref="ILane{T}.NaNLaneFill"/> (<c>-inf</c> for fmax,
        /// <c>+inf</c> for fmin — the identity among non-NaN values): a NaN data lane returns the suffix (skipped), a tie
        /// returns the suffix (the LATER element wins), and the suffix itself is never NaN. Eight contiguous chunks scanned
        /// backward side by side give eight independent chains, combined in order with the same op. A forward scan cannot
        /// do this with one instruction: with the accumulator first a NaN data lane would win, with the data first a tie
        /// would keep the EARLIER element (the old value-only kernel's ±0 discrepancy).
        /// </para>
        /// <para>
        /// WHY IT READS ALIGNED MEMORY. The section starts one element past an array's base (element 0 is NumPy's seed), so
        /// NumPy's lanes sit at a fixed offset <c>d</c> from 32-byte boundaries and every other 32-byte load of theirs splits
        /// a cache line — measured 1.4x the old kernel's time in L1/L2. Because the fold is order-free per lane and each lane
        /// keeps its elements in order, the section is read instead as the 32-byte-ALIGNED vectors covering it: memory
        /// position <c>p</c> of every aligned vector holds NumPy lane <c>(p - d) mod Count</c>. The first aligned vector
        /// contributes only positions <c>p &gt;= d</c> and the last only <c>p &lt; d</c> (the rest is blended to the fill
        /// before any arithmetic, so nothing outside the section — possibly another array's bytes or uninitialized memory —
        /// can reach the result), and one lane rotation maps positions back to NumPy's lanes. An aligned 32-byte read never
        /// crosses a page boundary, so reading those edge vectors cannot fault even at the ends of an allocation.
        /// </para>
        /// <para>
        /// THE ONE LANE THE FILL CANNOT DECIDE. A suffix result equal to the fill means either "the lane's data holds no
        /// non-NaN" or "its largest non-NaN is -inf" (+inf for fmin). Combining it with an ordered <paramref name="acc"/>
        /// lane is right in both cases, but a NaN accumulator lane must survive only in the first — NumPy keeps the FIRST of
        /// two NaNs, i.e. the accumulator's own payload. Only such lanes are rescanned (<see cref="KeepAllNaNLanes{T}"/>,
        /// vectorized, stopping once every one has met a non-NaN), so the rescan costs at most one extra pass and only on
        /// data whose lanes are all-NaN or bottom out at an infinity.
        /// </para>
        /// </remarks>
        /// <typeparam name="T">float or double (the P rules exist only for float lanes).</typeparam>
        /// <typeparam name="TLane">A NaN-suppressing rule (<see cref="FMaxLane{T}"/> / <see cref="FMinLane{T}"/>).</typeparam>
        /// <param name="acc">The accumulator entering the section (<c>splat</c> of the seed, or the fold carried from an earlier
        /// block of the same call).</param>
        /// <param name="ip">First element of the section (NumPy's lane 0), any alignment.</param>
        /// <param name="vectors">Whole vectors to fold (&gt;= 0).</param>
        /// <returns>The accumulator NumPy holds after the section, bit for bit.</returns>
        /// <remarks>
        /// Kept out of line on purpose: the production callers (<see cref="Finish{T,TLane}"/>, <see cref="FoldGroups{T,TLane}"/>)
        /// reach it only for sections of at least <see cref="FoldVectorsPMinElements"/> elements, and inlining it into them
        /// would make every SHORT row pay the vector-register spills this frame needs.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        internal static unsafe Vector256<T> FoldVectorsP<T, TLane>(Vector256<T> acc, T* ip, long vectors)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            if (vectors <= 0)
                return acc;

            int vs = Vector256<T>.Count;
            var fill = Vector256.Create(TLane.NaNLaneFill);
            // d: NumPy's lane 0 position inside its 32-byte block. A pointer that is not even element-aligned (a byte-offset
            // view) cannot be remapped lane for lane, so it — like an already aligned one and a short section — folds in place.
            int d = (int)(((nuint)ip & 31) / (nuint)sizeof(T));
            bool aligned = d != 0 && vectors >= AlignedFoldMinVectors && ((nuint)ip & (nuint)(sizeof(T) - 1)) == 0;
            Vector256<T> s;
            if (!aligned)
            {
                s = SuffixFold<T, TLane>(ip, vectors, fill);
            }
            else
            {
                // The vectors+1 aligned vectors covering the section: #0 contributes positions >= d, #vectors positions < d,
                // everything between is whole. Each edge is blended to the fill BEFORE the op touches it, and taken through
                // P256(v, fill) so a NaN edge lane becomes the fill (skipped) like any NaN the scan meets.
                T* a0 = ip - d;
                var tail = LanesFrom<T>(d);
                var first = TLane.P256(Vector256.ConditionalSelect(tail, Vector256.Load(a0), fill), fill);
                var last = TLane.P256(Vector256.ConditionalSelect(tail, fill, Vector256.Load(a0 + vectors * vs)), fill);
                // In sequence order: first, the whole middle, last — combined with the later operand second.
                s = TLane.P256(TLane.P256(first, SuffixFold<T, TLane>(a0 + vs, vectors - 1, fill)), last);
                s = RotateLanes(s, d);
            }

            // Prefix the section with the incoming accumulator: s is never NaN, so the plain op IS maxp here (a NaN acc lane
            // yields s, a tie keeps s — the later element).
            var r = TLane.P256(acc, s);
            var suspect = ~Vector256.Equals(acc, acc) & Vector256.Equals(s, fill);
            return suspect.ExtractMostSignificantBits() == 0 ? r : KeepAllNaNLanes(r, acc, ip, vectors, suspect);
        }

        /// <summary>
        /// The order-free fold of <paramref name="count"/> whole vectors from <paramref name="p"/>: per lane, the LAST element
        /// holding the largest non-NaN value, or <paramref name="fill"/> when the lane has none — eight contiguous chunks
        /// scanned BACKWARD with the plain op, data first (<c>P256(data, suffix)</c>), then combined in order.
        /// </summary>
        /// <remarks>
        /// The chunks are contiguous and the first <c>count % 8</c> of them hold one vector more, so no chain is longer than
        /// <c>count / 8 + 1</c> (a remainder parked on one chunk would be a serial tail of up to seven dependent ops). Each
        /// longer chunk folds its extra — its LAST — vector first, as a backward scan must. Only the ORDER inside a lane is
        /// load-bearing: chunk results combine as <c>P256(earlier, later)</c>, so a tie keeps the later chunk's element. Fewer
        /// than eight vectors fold as one chain (no chunk would reach one vector per chain anyway). Deliberately NOT inlined
        /// into <see cref="FoldVectorsP{T,TLane}"/>: measured, the identical chain loop ran 7–17% slower on long rows once
        /// inlined into the bigger frame (the JIT loop-placement effect), for a saving of one call per fold that the
        /// <see cref="FoldVectorsPMinElements"/> threshold already makes negligible.
        /// </remarks>
        /// <typeparam name="T">float or double.</typeparam>
        /// <typeparam name="TLane">A NaN-suppressing rule.</typeparam>
        /// <param name="p">First vector (any alignment; aligned reads are faster).</param>
        /// <param name="count">Whole vectors (&gt;= 0).</param>
        /// <param name="fill">The rule's fill (<c>-inf</c> / <c>+inf</c>), the identity of the plain op among non-NaN values.</param>
        /// <returns>The per-lane fold, never NaN.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe Vector256<T> SuffixFold<T, TLane>(T* p, long count, Vector256<T> fill)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int vs = Vector256<T>.Count;
            if (count < 8)
            {
                var t = fill;
                for (long k = count - 1; k >= 0; k--)
                    t = TLane.P256(Vector256.Load(p + k * vs), t);
                return t;
            }

            long L = count / 8;
            long extra = count - 8 * L;
            // Chunk c spans L vectors, plus one when c < extra; b1..b7 start where the previous chunk ends.
            T* b0 = p;
            T* b1 = b0 + (extra > 0 ? L + 1 : L) * vs;
            T* b2 = b1 + (extra > 1 ? L + 1 : L) * vs;
            T* b3 = b2 + (extra > 2 ? L + 1 : L) * vs;
            T* b4 = b3 + (extra > 3 ? L + 1 : L) * vs;
            T* b5 = b4 + (extra > 4 ? L + 1 : L) * vs;
            T* b6 = b5 + (extra > 5 ? L + 1 : L) * vs;
            T* b7 = b6 + (extra > 6 ? L + 1 : L) * vs;
            var s0 = fill; var s1 = fill; var s2 = fill; var s3 = fill;
            var s4 = fill; var s5 = fill; var s6 = fill; var s7 = fill;

            // The longer chunks' last vector (index L of the chunk) comes first in a backward scan. Chunk 7 never has one.
            long lastOff = L * vs;
            if (extra > 0) s0 = TLane.P256(Vector256.Load(b0 + lastOff), s0);
            if (extra > 1) s1 = TLane.P256(Vector256.Load(b1 + lastOff), s1);
            if (extra > 2) s2 = TLane.P256(Vector256.Load(b2 + lastOff), s2);
            if (extra > 3) s3 = TLane.P256(Vector256.Load(b3 + lastOff), s3);
            if (extra > 4) s4 = TLane.P256(Vector256.Load(b4 + lastOff), s4);
            if (extra > 5) s5 = TLane.P256(Vector256.Load(b5 + lastOff), s5);
            if (extra > 6) s6 = TLane.P256(Vector256.Load(b6 + lastOff), s6);

            for (long o = lastOff - vs; o >= 0; o -= vs)
            {
                s0 = TLane.P256(Vector256.Load(b0 + o), s0);
                s1 = TLane.P256(Vector256.Load(b1 + o), s1);
                s2 = TLane.P256(Vector256.Load(b2 + o), s2);
                s3 = TLane.P256(Vector256.Load(b3 + o), s3);
                s4 = TLane.P256(Vector256.Load(b4 + o), s4);
                s5 = TLane.P256(Vector256.Load(b5 + o), s5);
                s6 = TLane.P256(Vector256.Load(b6 + o), s6);
                s7 = TLane.P256(Vector256.Load(b7 + o), s7);
            }

            return TLane.P256(TLane.P256(TLane.P256(s0, s1), TLane.P256(s2, s3)), TLane.P256(TLane.P256(s4, s5), TLane.P256(s6, s7)));
        }

        /// <summary>
        /// All-ones in every lane whose position is at least <paramref name="d"/>, zero below — the part of the FIRST aligned
        /// vector that belongs to a section starting <paramref name="d"/> lanes into its 32-byte block (its complement is
        /// the part of the LAST aligned vector that does).
        /// </summary>
        /// <typeparam name="T">A 4- or 8-byte lane type.</typeparam>
        /// <param name="d">The section's start position inside its block (1 .. Count - 1).</param>
        /// <returns>The lane mask.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<T> LanesFrom<T>(int d) where T : unmanaged
            => sizeof(T) == 8
                ? Vector256.GreaterThanOrEqual(Vector256.Create(0L, 1L, 2L, 3L), Vector256.Create((long)d)).As<long, T>()
                : Vector256.GreaterThanOrEqual(Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7), Vector256.Create(d)).As<int, T>();

        /// <summary>
        /// Lane <c>j</c> of the result is lane <c>(j + d) mod Count</c> of <paramref name="g"/> — maps the positions of an
        /// aligned fold back to NumPy's lanes when NumPy's lane 0 sits at position <paramref name="d"/>. A pure permutation
        /// (no arithmetic), so the bits of every lane survive it, NaN payloads included.
        /// </summary>
        /// <remarks>
        /// One <c>vpermd</c> with AVX2, inlined into <see cref="FoldVectorsP{T,TLane}"/>, for BOTH lane widths: rotating
        /// 8-byte lanes by <paramref name="d"/> is rotating 4-byte lanes by <c>2d</c> (each 8-byte lane's two halves move
        /// together, in order), so the rotation is a variable index vector rather than <c>vpermq</c>'s immediate — no
        /// switch on <paramref name="d"/>, whose target changes from row to row when rows start at varying alignments.
        /// Without AVX2 the rotation goes through <see cref="RotateLanesPortable{T}"/>, kept out of line because its
        /// <c>stackalloc</c> would otherwise stop this method from inlining on the AVX2 hosts that never reach it.
        /// </remarks>
        /// <typeparam name="T">A 4- or 8-byte lane type.</typeparam>
        /// <param name="g">The fold by position.</param>
        /// <param name="d">The rotation (1 .. Count - 1).</param>
        /// <returns>The fold by NumPy lane.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector256<T> RotateLanes<T>(Vector256<T> g, int d) where T : unmanaged
        {
            if (Avx2.IsSupported && (sizeof(T) == 4 || sizeof(T) == 8))
            {
                // 32-bit result lane i <- source lane (i + d·(sizeof(T)/4)) & 7: for 8-byte lanes, 64-bit lane j takes the
                // halves 2j+2d and 2j+2d+1 (mod 8), i.e. source lane (j + d) & 3, low half first.
                int k = d * (sizeof(T) / 4);
                var idx = (Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7) + Vector256.Create(k)) & Vector256.Create(7);
                return Avx2.PermuteVar8x32(g.AsInt32(), idx).As<int, T>();
            }

            return RotateLanesPortable(g, d);
        }

        /// <summary>
        /// <see cref="RotateLanes{T}"/> without AVX2: the vector is stored twice back to back and one vector is read back
        /// starting <paramref name="d"/> lanes in — lane <c>j</c> of the result is lane <c>(j + d) mod Count</c> of
        /// <paramref name="g"/>, bits untouched.
        /// </summary>
        /// <remarks>
        /// Runs once per fold, so the store-forwarding stall of reading back across the two stores is noise. Out of line so
        /// the <c>stackalloc</c> never lands in the caller's frame.
        /// </remarks>
        /// <typeparam name="T">A 4- or 8-byte lane type.</typeparam>
        /// <param name="g">The fold by position.</param>
        /// <param name="d">The rotation (1 .. Count - 1).</param>
        /// <returns>The fold by NumPy lane.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe Vector256<T> RotateLanesPortable<T>(Vector256<T> g, int d) where T : unmanaged
        {
            int vs = Vector256<T>.Count;
            T* twice = stackalloc T[2 * vs];
            g.Store(twice);
            g.Store(twice + vs);
            return Vector256.Load(twice + d);
        }

        /// <summary>
        /// The rare fix-up of <see cref="FoldVectorsP{T,TLane}"/>: in each <paramref name="suspect"/> lane (the accumulator is
        /// NaN and the section's fold only reached the fill) keep the accumulator's own NaN when the section holds no non-NaN
        /// element in that lane — NumPy's <c>maxp</c> keeps the FIRST of two NaNs — and leave <paramref name="r"/> (the
        /// fill, which is then the lane's genuine -inf / +inf) otherwise.
        /// </summary>
        /// <remarks>
        /// Reads the section in NumPy's own lane layout (unaligned) and stops once every suspect lane has met a non-NaN, so a
        /// lane that merely bottoms out at an infinity usually settles within a few vectors; only a truly all-NaN lane scans
        /// the whole section — one ordered compare per vector, about the cost of the fold itself. Kept out of line: it runs
        /// only on degenerate data, and the fold's hot path should not carry its loop.
        /// </remarks>
        /// <typeparam name="T">float or double.</typeparam>
        /// <param name="r">The fold's answer where the lane is not NaN-only.</param>
        /// <param name="acc">The accumulator that entered the section.</param>
        /// <param name="ip">First element of the section.</param>
        /// <param name="vectors">Whole vectors in the section.</param>
        /// <param name="suspect">All-ones in the lanes to decide.</param>
        /// <returns>The corrected accumulator.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe Vector256<T> KeepAllNaNLanes<T>(Vector256<T> r, Vector256<T> acc, T* ip, long vectors, Vector256<T> suspect)
            where T : unmanaged
        {
            int vs = Vector256<T>.Count;
            var seen = Vector256<T>.Zero;   // all-ones once the lane met a non-NaN (x == x fails only for NaN)
            for (long k = 0; k < vectors; k++)
            {
                var v = Vector256.Load(ip + k * vs);
                seen |= Vector256.Equals(v, v);
                if ((k & 7) == 7 && Vector256.AndNot(suspect, seen).ExtractMostSignificantBits() == 0)
                    break;
            }

            return Vector256.ConditionalSelect(Vector256.AndNot(suspect, seen), acc, r);
        }

        /// <summary>
        /// Phases 1-4 over the LAST <paramref name="len"/> elements of the reduced stream: whole groups, then
        /// single vectors, then the horizontal <c>npyv_reduce_{max,min}n</c>, then the scalar tail.
        /// </summary>
        /// <remarks>
        /// <para>
        /// A DISPATCHER, inlined into every caller: the N rules — and a P-rule stretch shorter than
        /// <see cref="FoldVectorsPMinElements"/> — run phases 1 and 2 as NumPy spells them (<see cref="FinishTree{T,TLane}"/>:
        /// <see cref="FoldGroupsTree{T,TLane}"/> + the single-vector loop); a longer P-rule stretch goes to
        /// <see cref="FinishFoldP{T,TLane}"/>, which runs them as ONE order-free fold over every whole vector
        /// (<see cref="FoldVectorsP{T,TLane}"/>): NumPy's tree and loop only ever combine elements of the same lane in
        /// increasing position with an associative selection, so the accumulator — and every phase after it — is
        /// bit-identical, but the fold reads aligned memory and needs no per-group NaN proof. The split between the vector
        /// section and the scalar tail is untouched either way: the last <c>len % Count</c> elements still meet the scalar
        /// op, where the P rules' CRT tie / two-NaN rule differs.
        /// </para>
        /// <para>
        /// WHY THE TWO ROUTES ARE TWO METHODS, picked at the call site. Both were tried inside one body and both cost the
        /// short rows. A branch around phases 1 and 2 kept the pointer, the count and the vector count live across the
        /// fold's call, so the JIT pushed and popped four callee-saved registers and grew the frame from 104 to 184 bytes
        /// on EVERY call — 0.78–0.86x on rows of 10 to 100 elements that never take the fold. An early exit fixed the
        /// frame, but one tier-1 body still served both routes, and PGO compiles it ONCE from whatever the process ran
        /// first: after long flat reductions the tree's loops came out laid out as cold code — 0.69–0.84x on the same
        /// short rows. As separate methods each route gets its own profile, and <see cref="FinishTree{T,TLane}"/> is
        /// the tree body exactly as it was before the fold existed. For the N rules the test is a JIT-time constant, so
        /// their callers compile to the one plain call they made before.
        /// </para>
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="acc">The accumulator carried from any earlier <see cref="FoldGroups{T,TLane}"/> calls
        /// (<c>splat(x[0])</c> when this is the whole stream).</param>
        /// <param name="ip">First remaining element.</param>
        /// <param name="len">Remaining element count (&gt;= 1).</param>
        /// <returns>The reduction result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static unsafe T Finish<T, TLane>(Vector256<T> acc, T* ip, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
            => !TLane.PropagatesNaN && len >= FoldVectorsPMinElements
                ? FinishFoldP<T, TLane>(acc, ip, len)
                : FinishTree<T, TLane>(acc, ip, len);

        /// <summary>
        /// Phases 1-4 through NumPy's own spelling: whole groups (<see cref="FoldGroupsTree{T,TLane}"/>), single vectors,
        /// the horizontal <c>npyv_reduce_{max,min}{n,p}</c>, the scalar tail. The route of every N-rule call and of every
        /// P-rule stretch shorter than <see cref="FoldVectorsPMinElements"/> — the body <see cref="Finish{T,TLane}"/> had
        /// before the order-free fold existed, kept verbatim so it compiles (and is profiled) exactly as it did then.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
        /// <param name="acc">The accumulator carried from any earlier <see cref="FoldGroups{T,TLane}"/> calls
        /// (<c>splat(x[0])</c> when this is the whole stream).</param>
        /// <param name="ip">First remaining element.</param>
        /// <param name="len">Remaining element count (&gt;= 1).</param>
        /// <returns>The reduction result.</returns>
        private static unsafe T FinishTree<T, TLane>(Vector256<T> acc, T* ip, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            // NumPy's group tree, then its single-vector loop, literally.
            int vstep = Vector256<T>.Count;
            long wstep = vstep * 8L;
            long groups = len / wstep;
            acc = FoldGroupsTree<T, TLane>(acc, ip, groups);
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
        /// <see cref="Finish{T,TLane}"/> for a P-rule stretch of at least <see cref="FoldVectorsPMinElements"/> elements:
        /// every whole vector through the order-free aligned fold (<see cref="FoldVectorsP{T,TLane}"/>) in place of NumPy's
        /// group tree + single-vector loop, then the same horizontal reduce and scalar tail — NumPy's bits, lane for lane.
        /// </summary>
        /// <remarks>
        /// Out of line, and never inlined into <see cref="Finish{T,TLane}"/>'s callers: the fold's frame, its register
        /// spills and its PGO profile belong to this method, which only long stretches reach (see
        /// <see cref="Finish{T,TLane}"/>'s remarks for what sharing a body with the tree cost the short rows).
        /// </remarks>
        /// <typeparam name="T">float or double (only the P rules come here).</typeparam>
        /// <typeparam name="TLane">A NaN-suppressing rule (<see cref="FMaxLane{T}"/> / <see cref="FMinLane{T}"/>).</typeparam>
        /// <param name="acc">The accumulator entering the stretch.</param>
        /// <param name="ip">First remaining element.</param>
        /// <param name="len">Remaining element count (&gt;= <see cref="FoldVectorsPMinElements"/>).</param>
        /// <returns>The reduction result.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe T FinishFoldP<T, TLane>(Vector256<T> acc, T* ip, long len)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            int vstep = Vector256<T>.Count;
            long vectors = len / vstep;
            acc = FoldVectorsP<T, TLane>(acc, ip, vectors);
            ip += vectors * vstep;
            len -= vectors * vstep;

            T r = ReduceLanes<T, TLane>(acc);

            // The scalar tail, exactly as in Finish: the CRT fmax / fmin, a NaN met here keeping its own payload.
            for (; len > 0; --len, ++ip)
                r = N<T, TLane>(r, *ip);
            return r;
        }

        /// <summary>
        /// The horizontal reduce of the accumulator (<c>V_REDUCE_INTRIN</c>). N rules — <c>npyv_reduce_{max,min}n</c>: the
        /// canonical positive quiet NaN when ANY lane is NaN (the lane's own payload is NOT kept — NumPy returns a fixed
        /// constant), else the halves folded with the plain lane op down to lane 0 (lo vs hi, then pairs: exactly the
        /// extract/shuffle cascade). P rules — <c>npyv_reduce_{max,min}p</c>: lane 0 exactly as it is (payload kept) when
        /// EVERY lane is NaN, else the NaN lanes replaced by <see cref="ILane{T}.NaNLaneFill"/> (<c>-inf</c> for max,
        /// <c>+inf</c> for min) and the same plain cascade.
        /// </summary>
        /// <remarks>
        /// The P-rule fill can only lose to a real lane (an infinity of the losing sign, or an identical one — the same
        /// bits either way), so the cascade picks among the non-NaN lanes; the all-NaN answer is lane 0's own bits, which
        /// is why an all-NaN <c>np.nanmax</c> returns a payload and not the canonical NaN.
        /// </remarks>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="acc">The accumulator vector.</param>
        /// <returns>The horizontal result.</returns>
        private static unsafe T ReduceLanes<T, TLane>(Vector256<T> acc)
            where T : unmanaged, INumber<T> where TLane : struct, ILane<T>
        {
            // x == x fails only for a NaN lane; an integer vector always passes (so neither branch is ever taken for it).
            if (!Vector256.EqualsAll(acc, acc))
            {
                if (TLane.PropagatesNaN)
                    return CanonicalNaN<T>();

                // npyv_reduce_maxp: no non-NaN lane at all → lane 0 as is (npyv_any(notnan) is false).
                if (!Vector256.EqualsAny(acc, acc))
                    return acc.ToScalar();
                acc = Vector256.ConditionalSelect(Vector256.Equals(acc, acc), acc, Vector256.Create(TLane.NaNLaneFill));
            }

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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
        /// <remarks>
        /// Floats get one lane rule per op; an integer dtype folds <see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/>
        /// onto the max / min lanes (see <see cref="MinMaxOp"/>).
        /// </remarks>
        /// <param name="t">The element dtype (must satisfy <see cref="Supports"/>).</param>
        /// <param name="op">The reduction op.</param>
        /// <param name="x">First byte of the dense block (logical element 0 in memory order).</param>
        /// <param name="n">Element count (&gt;= 1).</param>
        /// <param name="slot">Receives the result (at least 8 bytes).</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        internal static unsafe void ReduceContiguous(NPTypeCode t, MinMaxOp op, byte* x, long n, byte* slot)
        {
            bool isMax = IsMaxLike(op);
            switch (t)
            {
                case NPTypeCode.Double:
                    *(double*)slot = op switch
                    {
                        MinMaxOp.Max => Reduce<double, MaxLane<double>>((double*)x, n),
                        MinMaxOp.Min => Reduce<double, MinLane<double>>((double*)x, n),
                        MinMaxOp.FMax => Reduce<double, FMaxLane<double>>((double*)x, n),
                        _ => Reduce<double, FMinLane<double>>((double*)x, n),
                    };
                    break;
                case NPTypeCode.Single:
                    *(float*)slot = op switch
                    {
                        MinMaxOp.Max => Reduce<float, MaxLane<float>>((float*)x, n),
                        MinMaxOp.Min => Reduce<float, MinLane<float>>((float*)x, n),
                        MinMaxOp.FMax => Reduce<float, FMaxLane<float>>((float*)x, n),
                        _ => Reduce<float, FMinLane<float>>((float*)x, n),
                    };
                    break;
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
        /// <param name="op">The reduction op (<see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/> for np.evaluate's
        /// <c>NanMax</c> / <c>NanMin</c>).</param>
        /// <param name="t">The child (== result) dtype.</param>
        /// <param name="slot">Receives the result at dtype <paramref name="t"/> (at least 8 bytes).</param>
        /// <returns>True when the reduction was computed here.</returns>
        private unsafe bool TryExactFlatMinMax(NDExprProgram child, NDArray[] inputs, long n, MinMaxOp op, NPTypeCode t, byte* slot)
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
                    return TryExactFlatMinMaxArray(ops[0], op, slot);
                // Logical element 0: a contiguous slice re-seats Address (offset 0), an F column block keeps a
                // non-zero offset — Address + offset·itemsize is right for both.
                byte* x = (byte*)ops[0].Address + (long)s.offset * t.SizeOf();
                NumPyMinMaxReduce.ReduceContiguous(t, op, x, n, slot);
                NDExpr.ExactMinMaxRuns++;
                return true;
            }

            if (CanStreamChild(child, ops, allowF: true, allowPermuted: true))
            {
                StreamExactMinMax(child, inputs, ops, n, op, t, slot);
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
                NumPyMinMaxReduce.ReduceContiguous(t, op, m, n, slot);
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
        /// <param name="op">The reduction op.</param>
        /// <param name="t">The child dtype (satisfies <see cref="NumPyMinMaxReduce.Supports"/>).</param>
        /// <param name="slot">Receives the result at dtype <paramref name="t"/>.</param>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is not a supported dtype.</exception>
        private static unsafe void StreamExactMinMax(NDExprProgram child, NDArray[] inputs, NDArray[] ops, long n,
            MinMaxOp op, NPTypeCode t, byte* slot)
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

            // Integer lanes serve FMax / FMin through the max / min lanes (identical results, see MinMaxOp).
            bool isMax = NumPyMinMaxReduce.IsMaxLike(op);
            switch (t)
            {
                case NPTypeCode.Double:
                    *(double*)slot = op switch
                    {
                        MinMaxOp.Max => StreamMinMax<double, NumPyMinMaxReduce.MaxLane<double>>(ref stream, 0, n, scratch),
                        MinMaxOp.Min => StreamMinMax<double, NumPyMinMaxReduce.MinLane<double>>(ref stream, 0, n, scratch),
                        MinMaxOp.FMax => StreamMinMax<double, NumPyMinMaxReduce.FMaxLane<double>>(ref stream, 0, n, scratch),
                        _ => StreamMinMax<double, NumPyMinMaxReduce.FMinLane<double>>(ref stream, 0, n, scratch),
                    };
                    break;
                case NPTypeCode.Single:
                    *(float*)slot = op switch
                    {
                        MinMaxOp.Max => StreamMinMax<float, NumPyMinMaxReduce.MaxLane<float>>(ref stream, 0, n, scratch),
                        MinMaxOp.Min => StreamMinMax<float, NumPyMinMaxReduce.MinLane<float>>(ref stream, 0, n, scratch),
                        MinMaxOp.FMax => StreamMinMax<float, NumPyMinMaxReduce.FMaxLane<float>>(ref stream, 0, n, scratch),
                        _ => StreamMinMax<float, NumPyMinMaxReduce.FMinLane<float>>(ref stream, 0, n, scratch),
                    };
                    break;
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
        /// <typeparam name="TLane">The lane rule (max / min, or the NaN-suppressing fmax / fmin).</typeparam>
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
