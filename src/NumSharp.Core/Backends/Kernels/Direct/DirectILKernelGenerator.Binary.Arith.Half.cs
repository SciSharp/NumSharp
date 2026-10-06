using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Binary.Arith.Half.cs — float16 add / subtract /
// multiply / divide via SIMD widen-compute-narrow (bit-exact with NumPy)
// =============================================================================
//
// NumPy's HALF arithmetic loops are scalar C:
//     *(op1) = npy_float_to_half(npy_half_to_float(in1) OP npy_half_to_float(in2))
// — float32 compute, RTNE narrow (loops.c.src HALF_add..divide; ~280-310 µs per
// 100K on 2.4.2). This kernel runs the SAME pipeline 8 lanes at a time with the
// proven exact primitives: HalfBitsToFloatExact (widen, NaN-payload exact) →
// one vaddps/vsubps/vmulps/vdivps → FloatToHalfBits (Giesen RTNE narrow,
// payload-truncating like npy_float_to_half). The compute MUST be float32 —
// a double-precision bridge single-rounds where NumPy's float32 path double-
// rounds (exponent-gap sums), a real 1-ULP divergence class.
//
// NaN-payload priority is a HOST PIN (same class as the MSVC-pinned cast
// kernels — Fuzz/README.md → "Host-dependent values"): the numpy 2.4.2
// win-amd64 wheel compiled the commutative ops with REVERSED operands
// (addss/mulss dst=in2, src=in1) and subtract/divide natural, and x86's NaN
// rule is src1-first — probed live: add(qNaN_a, sNaN_b) = quieted b,
// sub(qNaN_a, sNaN_b) = a, both-qNaN add → in2, both-qNaN div → in1. That
// priority CANNOT be expressed through Avx.Add/Multiply operand order — RyuJIT
// may re-swap commutative intrinsic operands (observed: it did) — so NaN lanes
// are resolved by an EXPLICIT post-op blend on the narrowed 16-bit lanes:
//   add/multiply:    res = nanB ? quiet(b) : nanA ? quiet(a) : fp-result
//   subtract/divide: res = nanA ? quiet(a) : nanB ? quiet(b) : fp-result
// with quiet(x) = x | 0x0200 (identity on qNaN, x86 "quieted" on sNaN, sign
// kept). Non-NaN lanes take the hardware FP result untouched — 0/0 and
// inf-inf produce the default qNaN 0xfe00, overflow saturates to ±inf through
// the narrow, subnormals widen exactly; all instruction-level identical to the
// wheel by construction.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Returns the static widen-compute-narrow kernel for a (Half,Half)→Half arithmetic
        /// key, or null when the (op, path) pair is not served (falls to emitted IL).
        /// </summary>
        /// <remarks>
        /// Power rides the same kernels (plan api-parameter-parity X13): NumPy's HALF_power is powf on the halves widened
        /// bit for bit, narrowed with npy_float_to_half, with no scalar-exponent fast path and no operand-order NaN rule
        /// (powf decides the NaN), so each contiguous kernel hands it to <see cref="HalfPower"/> at entry — vector widen
        /// and narrow around a block loop of MathF.Pow — and the strided route reaches it through the same kernels.
        /// </remarks>
        internal static unsafe MixedTypeKernel TryGetHalfArithKernel(BinaryOp op, ExecutionPath path)
        {
            if (op != BinaryOp.Add && op != BinaryOp.Subtract && op != BinaryOp.Multiply && op != BinaryOp.Divide
                && op != BinaryOp.Power)
                return null;
            switch (path)
            {
                case ExecutionPath.SimdFull:
                    return (l, r, o, ls, rs, sh, nd, n) =>
                        HalfArithFull((ushort*)l, (ushort*)r, (ushort*)o, n, op);
                case ExecutionPath.SimdScalarRight:
                    return (l, r, o, ls, rs, sh, nd, n) =>
                        HalfArithScalarRight((ushort*)l, *(ushort*)r, (ushort*)o, n, op);
                case ExecutionPath.SimdScalarLeft:
                    return (l, r, o, ls, rs, sh, nd, n) =>
                        HalfArithScalarLeft(*(ushort*)l, (ushort*)r, (ushort*)o, n, op);

                // Strided / broadcast layouts. Half is the ONLY dtype routed to the
                // direct kernel here — every other dtype takes NDIter (see the
                // (Half,Half)→Half decline in DefaultEngine.TryExecuteBinaryOpViaNDIter),
                // which is the exact reason the emitted SimdChunk kernel's per-row chunk
                // model — never exercised by another dtype — mis-addresses BOTH a
                // double-broadcast (kron's a⊗b builds two operands broadcast on
                // complementary axes) and a strided view whose inner |stride| ∉ {0,1}
                // (a reversed step-2 slice), silently producing garbage. Serve them with
                // the per-element coordinate odometer instead (EmitGeneralLoop's proven
                // address model + the bit-exact scalar WCN), correct for every layout.
                case ExecutionPath.SimdChunk:
                case ExecutionPath.General:
                    return (l, r, o, ls, rs, sh, nd, n) =>
                        HalfArithStrided((ushort*)l, (ushort*)r, (ushort*)o, ls, rs, sh, nd, n, op);
            }
            return null;
        }

        /// <summary>8 f16 lanes (as a 128-bit ushort vector) → 8 exact f32 lanes.</summary>
        /// <remarks>Internal so the numpy.polynomial evaluation kernels (ILKernelGenerator.Polynomial.Emitter.cs)
        ///     can keep float16 values as float32 lanes on the f16 grid with the SAME widen primitive, which is
        ///     what makes their vector and scalar float16 paths bit-identical. Requires AVX2.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<float> HalfWiden8V(Vector128<ushort> h)
            => HalfBitsToFloatExact(Avx2.ConvertToVector256Int32(h));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> HalfArithOp(Vector256<float> fa, Vector256<float> fb, BinaryOp op)
        {
            switch (op)
            {
                case BinaryOp.Add: return Avx.Add(fa, fb);
                case BinaryOp.Multiply: return Avx.Multiply(fa, fb);
                case BinaryOp.Subtract: return Avx.Subtract(fa, fb);
                default: return Avx.Divide(fa, fb); // Divide
            }
        }

        /// <summary>
        /// The number of float16 elements <see cref="HalfPower"/> widens, raises and narrows at a time: two 1 KB float32
        /// stack blocks, which stay in L1 next to the operands and the result.
        /// </summary>
        private const int HalfPowerBlock = 256;

        /// <summary>
        /// NumPy's HALF_power over <paramref name="n"/> elements (<c>PyUFunc_ee_e_As_ff_f</c> with <c>powf</c>): each
        /// operand widened exactly (a signalling NaN stays signalling, so powf returns NaN for sNaN ** 0 as NumPy's does),
        /// <c>powf</c> per element — the C runtime's, which MathF.Pow calls, as NumPy's npy_powf does; no vector powf
        /// reproduces its bits — and NumPy's npy_float_to_half rounding. HALF_power has NO scalar-exponent fast path
        /// (unlike FLOAT_power / DOUBLE_power), so <c>x ** 2</c> is powf(x, 2), never x*x.
        /// </summary>
        /// <param name="pa">The bases: contiguous when <paramref name="sa"/> is 1, one broadcast value when it is 0.</param>
        /// <param name="sa">The bases' ELEMENT stride, 1 or 0 (the strided route gathers any other stride first).</param>
        /// <param name="pb">The exponents: contiguous when <paramref name="sb"/> is 1, one broadcast value when it is 0.</param>
        /// <param name="sb">The exponents' ELEMENT stride, 1 or 0.</param>
        /// <param name="pr">The contiguous result; it never overlaps an operand (the direct route allocates it).</param>
        /// <param name="n">The element count (0 or more).</param>
        /// <remarks>
        /// <para>
        /// The work is done a block of <see cref="HalfPowerBlock"/> elements at a time: the vector widen of each operand
        /// block, ONE tight scalar loop of MathF.Pow over the block, the vector narrow. A per-8-lane form (extract the
        /// lanes of a widened vector, call powf on each, rebuild the vector) cost ~1.4 ns an element over the bare powf
        /// (~5 ns on this host) in lane shuffles; the block form keeps the overhead to the widen / narrow, which are what
        /// beat NumPy (its HALF loop converts each half with a scalar npy_half_to_float / npy_float_to_half,
        /// ~3.5 ns an element). powf is the floor on both sides, so the ceiling is NumPy's ~8.6 ns / ~5 ns ≈ 1.7×.
        /// </para>
        /// <para>No operand-order NaN rule applies: powf decides the NaN (and pow(1, NaN) is 1, pow(NaN, 0) is 1).</para>
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfPower(ushort* pa, long sa, ushort* pb, long sb, ushort* pr, long n)
        {
            if (!Avx2.IsSupported)
            {
                for (long i = 0; i < n; i++)
                    pr[i] = HalfArithBits(pa[i * sa], pb[i * sb], BinaryOp.Power);
                return;
            }

            // The widened blocks (never zeroed: each is written before it is read). A broadcast operand is widened once,
            // outside the block loop, and its block stays unused.
            float* fa = stackalloc float[HalfPowerBlock];
            float* fb = stackalloc float[HalfPowerBlock];
            float a0 = sa == 0 ? HalfToFloatScalarExact(*pa) : 0f;
            float b0 = sb == 0 ? HalfToFloatScalarExact(*pb) : 0f;

            for (long i = 0; i < n; i += HalfPowerBlock)
            {
                int m = (int)Math.Min(HalfPowerBlock, n - i);
                if (sa != 0)
                    HalfWidenBlock(pa + i, fa, m);
                if (sb != 0)
                    HalfWidenBlock(pb + i, fb, m);

                // The powers, written over the base block (each element is read before its slot is overwritten). One
                // loop per broadcast pattern so the hot loop reads one array, or two, and nothing else.
                if (sb == 0)
                {
                    for (int k = 0; k < m; k++)
                        fa[k] = MathF.Pow(fa[k], b0);
                }
                else if (sa == 0)
                {
                    for (int k = 0; k < m; k++)
                        fa[k] = MathF.Pow(a0, fb[k]);
                }
                else
                {
                    for (int k = 0; k < m; k++)
                        fa[k] = MathF.Pow(fa[k], fb[k]);
                }

                ushort* r = pr + i;
                int j = 0;
                for (; j + 8 <= m; j += 8)
                    Sse2.Store(r + j, HalfNarrow8V(Vector256.Load(fa + j)));
                for (; j < m; j++)
                    r[j] = SingleToHalfBits(fa[j]);
            }
        }

        /// <summary>Widens <paramref name="m"/> contiguous float16 patterns to exact float32 values (8 at a time).</summary>
        /// <param name="src">The first float16 pattern.</param>
        /// <param name="dst">The float32 block (at least <paramref name="m"/> slots).</param>
        /// <param name="m">How many elements (at most <see cref="HalfPowerBlock"/>).</param>
        /// <remarks>Exact, NaN payloads and signalling bits included (<see cref="HalfWiden8V"/> /
        /// <see cref="HalfToFloatScalarExact"/>), as NumPy's npy_half_to_float. Requires AVX2.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HalfWidenBlock(ushort* src, float* dst, int m)
        {
            int k = 0;
            for (; k + 8 <= m; k += 8)
                HalfWiden8V(Sse2.LoadVector128(src + k)).Store(dst + k);
            for (; k < m; k++)
                dst[k] = HalfToFloatScalarExact(src[k]);
        }

        /// <summary>Narrow 8 f32 result lanes back to 8 f16 patterns (128-bit ushort vector).</summary>
        /// <remarks>Internal for the same reason as <see cref="HalfWiden8V"/>: the polynomial kernels round each
        ///     float32-lane result back onto the f16 grid through this exact narrow. Requires AVX2.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<ushort> HalfNarrow8V(Vector256<float> fr)
        {
            var hb = FloatToHalfBits(fr);
            return Sse41.PackUnsignedSaturate(hb.GetLower(), hb.GetUpper());
        }

        /// <summary>
        /// The explicit NaN-priority fixup on the narrowed lanes — blends quiet(a)/quiet(b)
        /// over NaN lanes in the wheel's probed order (in2 first for add/multiply, in1 first
        /// for subtract/divide). Masks are 16-bit-lane compares on the RAW operand bits.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<ushort> HalfArithNaNFix(
            Vector128<ushort> res, Vector128<ushort> ha, Vector128<ushort> hb,
            Vector128<short> nanA, Vector128<short> nanB, bool in2First)
        {
            var quiet = Vector128.Create((ushort)0x0200);
            var qa = Sse2.Or(ha, quiet);
            var qb = Sse2.Or(hb, quiet);
            if (in2First)
            {
                res = Sse41.BlendVariable(res.AsByte(), qa.AsByte(), nanA.AsByte()).AsUInt16();
                res = Sse41.BlendVariable(res.AsByte(), qb.AsByte(), nanB.AsByte()).AsUInt16();
            }
            else
            {
                res = Sse41.BlendVariable(res.AsByte(), qb.AsByte(), nanB.AsByte()).AsUInt16();
                res = Sse41.BlendVariable(res.AsByte(), qa.AsByte(), nanA.AsByte()).AsUInt16();
            }
            return res;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<short> HalfNaNMask8(Vector128<ushort> h, Vector128<short> m7fff, Vector128<short> vinf)
            => Sse2.CompareGreaterThan(Sse2.And(h.AsInt16(), m7fff), vinf); // mags ≤ 0x7fff: sign-safe

        /// <summary>Both operands streamed (SimdFull), 2×-unrolled (16 halves/iter).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfArithFull(ushort* pa, ushort* pb, ushort* pr, long n, BinaryOp op)
        {
            if (op == BinaryOp.Power)
            {
                HalfPower(pa, 1, pb, 1, pr, n);
                return;
            }
            long i = 0;
            if (Avx2.IsSupported)
            {
                bool in2First = op == BinaryOp.Add || op == BinaryOp.Multiply;
                var m7fff = Vector128.Create((short)0x7FFF);
                var vinf = Vector128.Create((short)0x7C00);
                for (; i + 8 <= n; i += 8)
                {
                    var ha = Sse2.LoadVector128(pa + i);
                    var hb = Sse2.LoadVector128(pb + i);
                    var res = HalfNarrow8V(HalfArithOp(HalfWiden8V(ha), HalfWiden8V(hb), op));
                    res = HalfArithNaNFix(res, ha, hb,
                        HalfNaNMask8(ha, m7fff, vinf), HalfNaNMask8(hb, m7fff, vinf), in2First);
                    Sse2.Store(pr + i, res);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfArithBits(pa[i], pb[i], op);
        }

        /// <summary>lhs streamed, rhs one broadcast scalar (in2, widened once).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfArithScalarRight(ushort* pa, ushort b, ushort* pr, long n, BinaryOp op)
        {
            if (op == BinaryOp.Power)
            {
                HalfPower(pa, 1, &b, 0, pr, n);
                return;
            }
            long i = 0;
            if (Avx2.IsSupported)
            {
                bool in2First = op == BinaryOp.Add || op == BinaryOp.Multiply;
                var m7fff = Vector128.Create((short)0x7FFF);
                var vinf = Vector128.Create((short)0x7C00);
                var fb = Vector256.Create(HalfToFloatScalarExact(b));
                var hb = Vector128.Create(b);
                var nanB = (b & 0x7FFF) > 0x7C00 ? Vector128<short>.AllBitsSet : Vector128<short>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    var ha = Sse2.LoadVector128(pa + i);
                    var res = HalfNarrow8V(HalfArithOp(HalfWiden8V(ha), fb, op));
                    res = HalfArithNaNFix(res, ha, hb, HalfNaNMask8(ha, m7fff, vinf), nanB, in2First);
                    Sse2.Store(pr + i, res);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfArithBits(pa[i], b, op);
        }

        /// <summary>lhs one broadcast scalar (in1), rhs streamed — operand ROLES preserved.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfArithScalarLeft(ushort a, ushort* pb, ushort* pr, long n, BinaryOp op)
        {
            if (op == BinaryOp.Power)
            {
                HalfPower(&a, 0, pb, 1, pr, n);
                return;
            }
            long i = 0;
            if (Avx2.IsSupported)
            {
                bool in2First = op == BinaryOp.Add || op == BinaryOp.Multiply;
                var m7fff = Vector128.Create((short)0x7FFF);
                var vinf = Vector128.Create((short)0x7C00);
                var fa = Vector256.Create(HalfToFloatScalarExact(a));
                var ha = Vector128.Create(a);
                var nanA = (a & 0x7FFF) > 0x7C00 ? Vector128<short>.AllBitsSet : Vector128<short>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    var hb = Sse2.LoadVector128(pb + i);
                    var res = HalfNarrow8V(HalfArithOp(fa, HalfWiden8V(hb), op));
                    res = HalfArithNaNFix(res, ha, hb, nanA, HalfNaNMask8(hb, m7fff, vinf), in2First);
                    Sse2.Store(pr + i, res);
                }
            }
            for (; i < n; i++)
                pr[i] = HalfArithBits(a, pb[i], op);
        }

        /// <summary>
        /// The number of float16 elements a strided run is gathered in at a time (<see cref="HalfArithRun"/>): two
        /// 512-byte stack blocks, small enough to stay in L1 next to the destination run.
        /// </summary>
        private const int HalfArithGatherBlock = 256;

        /// <summary>
        /// Strided / broadcast (Half,Half)→Half arithmetic (SimdChunk / General paths): the result is written
        /// C-contiguous, linearly, over <paramref name="shape"/> (C order, the last axis fastest). An odometer walks the
        /// outer axes with incremental element offsets, and each inner run goes to <see cref="HalfArithRun"/>, which
        /// picks the SIMD widen-compute-narrow kernel for the run's strides (both contiguous, one broadcast, or gathered
        /// into a stack block first) — correct for ANY layout: contiguous, broadcast (stride 0), negative stride, or step
        /// (|stride| &gt; 1), and for a double broadcast (kron's operands stretched on complementary axes).
        /// </summary>
        /// <param name="pa">The left operand's first element (offset-adjusted).</param>
        /// <param name="pb">The right operand's first element (offset-adjusted).</param>
        /// <param name="pr">The result's first element; <paramref name="n"/> elements are written linearly.</param>
        /// <param name="lhsStrides">The left operand's ELEMENT strides per coalesced axis (0 broadcasts).</param>
        /// <param name="rhsStrides">The right operand's ELEMENT strides per coalesced axis (0 broadcasts).</param>
        /// <param name="shape">The coalesced result extents.</param>
        /// <param name="ndim">The number of coalesced axes (0 for a single element).</param>
        /// <param name="n">The number of result elements (the product of <paramref name="shape"/>).</param>
        /// <param name="op">Add, Subtract, Multiply, Divide or Power.</param>
        /// <remarks>
        /// Bit-identical to NumPy because every lane runs NumPy's scalar HALF loop (float32 compute + RTNE narrow + the
        /// operand-order NaN pin) — the SIMD kernels and <see cref="HalfArithBits"/> are the same pipeline — and the
        /// operand ROLES are kept on every route (a broadcast left operand takes <see cref="HalfArithScalarLeft"/>, never a
        /// swapped call). It replaced a per-element coordinate decomposition (a div / mod per axis per element) that cost
        /// ~10 ns an element, 0.22–0.48× NumPy (plan api-parameter-parity X4 / X15).
        /// </remarks>
        [SkipLocalsInit]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfArithStrided(
            ushort* pa, ushort* pb, ushort* pr,
            long* lhsStrides, long* rhsStrides, long* shape, int ndim, long n, BinaryOp op)
        {
            if (n <= 0)
                return;
            if (ndim == 0)
            {
                pr[0] = HalfArithBits(pa[0], pb[0], op);
                return;
            }

            // The gather blocks for strided runs (never zeroed: a block is written before the kernel reads it).
            ushort* stageA = stackalloc ushort[HalfArithGatherBlock];
            ushort* stageB = stackalloc ushort[HalfArithGatherBlock];

            int inner = ndim - 1;
            long len = shape[inner], sa = lhsStrides[inner], sb = rhsStrides[inner];
            if (len <= 0)
                return;

            // The outer odometer: coordinates of the axes before the inner one, and the operands' running element
            // offsets (incremental: a carry rewinds an axis by stride * extent instead of recomputing from coordinates).
            long* coord = stackalloc long[inner + 1];
            for (int d = 0; d <= inner; d++)
                coord[d] = 0;
            long offA = 0, offB = 0;

            for (long done = 0; done < n; done += len)
            {
                HalfArithRun(pa + offA, sa, pb + offB, sb, pr + done, len, op, stageA, stageB);

                for (int d = inner - 1; d >= 0; d--)
                {
                    coord[d]++;
                    offA += lhsStrides[d];
                    offB += rhsStrides[d];
                    if (coord[d] < shape[d])
                        break;
                    coord[d] = 0;
                    offA -= lhsStrides[d] * shape[d];
                    offB -= rhsStrides[d] * shape[d];
                }
            }
        }

        /// <summary>
        /// One inner run of <see cref="HalfArithStrided"/>: <c>r[i] = a[i * sa] OP b[i * sb]</c> for i in [0, n), the
        /// result contiguous. Both operands contiguous → <see cref="HalfArithFull"/>; one broadcast (stride 0) and the
        /// other contiguous → <see cref="HalfArithScalarRight"/> / <see cref="HalfArithScalarLeft"/>; both broadcast →
        /// one value; any other stride → the strided operand(s) gathered into a stack block of
        /// <see cref="HalfArithGatherBlock"/> elements, then the same kernels.
        /// </summary>
        /// <param name="a">The run's first left element.</param>
        /// <param name="sa">The left ELEMENT stride along the run (any sign; 0 broadcasts).</param>
        /// <param name="b">The run's first right element.</param>
        /// <param name="sb">The right ELEMENT stride along the run (any sign; 0 broadcasts).</param>
        /// <param name="r">The run's first result element (contiguous).</param>
        /// <param name="n">The run's length (at least 1).</param>
        /// <param name="op">Add, Subtract, Multiply, Divide or Power.</param>
        /// <param name="stageA">A scratch block of <see cref="HalfArithGatherBlock"/> elements for the left operand.</param>
        /// <param name="stageB">A scratch block of <see cref="HalfArithGatherBlock"/> elements for the right operand.</param>
        /// <remarks>
        /// The result is a fresh array, never an operand (the direct route allocates it), so reading the operands
        /// ahead of the stores — a vector at a time, or a gathered block — cannot observe a write.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HalfArithRun(
            ushort* a, long sa, ushort* b, long sb, ushort* r, long n, BinaryOp op, ushort* stageA, ushort* stageB)
        {
            if (sa == 1 && sb == 1)
            {
                HalfArithFull(a, b, r, n, op);
                return;
            }
            if (sa == 1 && sb == 0)
            {
                HalfArithScalarRight(a, *b, r, n, op);
                return;
            }
            if (sa == 0 && sb == 1)
            {
                HalfArithScalarLeft(*a, b, r, n, op);
                return;
            }
            if (sa == 0 && sb == 0)
            {
                // Both operands broadcast along the run: one value, written n times.
                ushort v = HalfArithBits(*a, *b, op);
                new Span<ushort>(r, (int)Math.Min(n, int.MaxValue)).Fill(v);
                for (long i = int.MaxValue; i < n; i++)
                    r[i] = v;
                return;
            }

            // A strided operand (|stride| > 1, or negative): gather a block of it, then run the contiguous kernel the
            // block's operand pair calls for. A contiguous or broadcast operand is read in place.
            for (long k = 0; k < n; k += HalfArithGatherBlock)
            {
                long m = Math.Min(HalfArithGatherBlock, n - k);
                ushort* ra = a + k * sa, rb = b + k * sb;
                ushort* ga = sa == 0 || sa == 1 ? ra : HalfArithGather(ra, sa, m, stageA);
                ushort* gb = sb == 0 || sb == 1 ? rb : HalfArithGather(rb, sb, m, stageB);
                if (sa == 0)
                    HalfArithScalarLeft(*ga, gb, r + k, m, op);
                else if (sb == 0)
                    HalfArithScalarRight(ga, *gb, r + k, m, op);
                else
                    HalfArithFull(ga, gb, r + k, m, op);
            }
        }

        /// <summary>Copies <paramref name="m"/> strided float16 elements into a contiguous block.</summary>
        /// <param name="src">The first element.</param>
        /// <param name="stride">The ELEMENT stride (any nonzero value).</param>
        /// <param name="m">How many elements (at most <see cref="HalfArithGatherBlock"/>).</param>
        /// <param name="block">The destination block.</param>
        /// <returns><paramref name="block"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe ushort* HalfArithGather(ushort* src, long stride, long m, ushort* block)
        {
            // Running pointer, four elements per pass: the loads are independent, so they overlap.
            long j = 0;
            for (; j + 4 <= m; j += 4)
            {
                block[j] = *src;
                block[j + 1] = src[stride];
                block[j + 2] = src[2 * stride];
                block[j + 3] = src[3 * stride];
                src += 4 * stride;
            }
            for (; j < m; j++)
            {
                block[j] = *src;
                src += stride;
            }
            return block;
        }

        /// <summary>
        /// Scalar lane — the exact wheel pipeline with the NaN priority encoded EXPLICITLY:
        /// Add/Multiply resolve in2 first, Subtract/Divide in1 first; |0x0200 is x86
        /// "quieted" (identity on qNaN, sign kept).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ushort HalfArithBits(ushort a, ushort b, BinaryOp op)
        {
            // Power: NumPy's HALF_power, powf on the exactly widened halves (no NaN priority of its own).
            if (op == BinaryOp.Power)
                return SingleToHalfBits(MathF.Pow(HalfToFloatScalarExact(a), HalfToFloatScalarExact(b)));
            bool nA = (a & 0x7FFF) > 0x7C00, nB = (b & 0x7FFF) > 0x7C00;
            if (nA | nB)
            {
                if (op == BinaryOp.Add || op == BinaryOp.Multiply)
                    return (ushort)((nB ? b : a) | 0x0200);
                return (ushort)((nA ? a : b) | 0x0200);
            }
            float fa = HalfToFloatScalarExact(a), fb = HalfToFloatScalarExact(b);
            float r = op switch
            {
                BinaryOp.Add => fa + fb,
                BinaryOp.Subtract => fa - fb,
                BinaryOp.Multiply => fa * fb,
                _ => fa / fb,
            };
            return SingleToHalfBits(r);
        }

        /// <summary>
        /// <see cref="Half"/>-struct wrapper over <see cref="HalfArithBits"/> for the emitted scalar
        /// path (the np.evaluate fused kernel and any other <c>EmitScalarOperation(op, Half)</c> caller):
        /// reinterprets each operand to its raw f16 bits, runs the float32-compute + RTNE-narrow +
        /// NaN-priority pipeline, and reinterprets the result back. Add/Subtract/Multiply/Divide ONLY —
        /// the caller (<c>EmitHalfOperation</c>) keeps power/atan2/mod/floordivide/fmod on their own paths.
        /// <para>
        /// This is the float32 model NumPy's HALF loops run (<c>astype 'e'->'f'</c>); the older emitted
        /// path widened to <b>double</b>, which double-rounds where NumPy single-rounds (exponent-gap
        /// sums — a real 1-ULP divergence class) and BCL-quiets sNaN, so routing through this restores
        /// bit-for-bit NumPy parity AND makes the emitted scalar Half arithmetic identical to
        /// <see cref="HalfArithVec256"/>, which the fused vector==scalar contract requires.
        /// </para>
        /// </summary>
        /// <param name="a">The left operand.</param>
        /// <param name="b">The right operand.</param>
        /// <param name="op">Add, Subtract, Multiply or Divide (any other op is a caller bug).</param>
        /// <returns>The f16 result, float32-computed and RTNE-narrowed, NaN-priority-exact.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Half HalfArithScalarStruct(Half a, Half b, BinaryOp op)
            => BitConverter.UInt16BitsToHalf(
                HalfArithBits(BitConverter.HalfToUInt16Bits(a), BitConverter.HalfToUInt16Bits(b), op));

        /// <summary>
        /// Vector lane for the np.evaluate fused kernel: 16 f16 patterns (as a <see cref="Vector256{T}"/>
        /// of <see cref="ushort"/>) → 16 f16 result patterns, running the SAME widen-compute-narrow +
        /// explicit NaN-priority fixup as <see cref="HalfArithFull"/> on each 8-lane half. Bit-identical
        /// to <see cref="HalfArithScalarStruct"/> (hence to the scalar body the evaluate.jsonl oracle
        /// holds to NumPy) by construction — both compute in float32 with the same proven Giesen
        /// primitives and encode the same operand-order NaN priority.
        /// <para>
        /// The fused shell carries a Half tree's lane as raw <c>Vector256&lt;ushort&gt;</c> (f16 bits),
        /// so a Half arithmetic node consumes two such vectors and produces one — the round-to-f16 lives
        /// HERE, per node, exactly as NumPy's <c>npy_float_to_half(npy_half_to_float(a) OP …)</c> rounds
        /// after every op. Add/Subtract/Multiply/Divide only.
        /// </para>
        /// </summary>
        /// <param name="ha">The left operand's 16 f16 bit patterns.</param>
        /// <param name="hb">The right operand's 16 f16 bit patterns.</param>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <returns>The 16 f16 result bit patterns.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<ushort> HalfArithVec256(Vector256<ushort> ha, Vector256<ushort> hb, BinaryOp op)
        {
            // in2First mirrors the wheel's probed commutative-vs-natural NaN order (HalfArithFull).
            bool in2First = op == BinaryOp.Add || op == BinaryOp.Multiply;
            var m7fff = Vector128.Create((short)0x7FFF);
            var vinf = Vector128.Create((short)0x7C00);

            // Lower / upper 8 lanes each go through the proven 8-lane pipeline; recombine the two
            // 128-bit halves. Splitting is free (GetLower/GetUpper are register moves) and lets the
            // whole 16-lane op reuse HalfArithFull's exact primitives rather than duplicating them.
            var haLo = ha.GetLower(); var hbLo = hb.GetLower();
            var resLo = HalfNarrow8V(HalfArithOp(HalfWiden8V(haLo), HalfWiden8V(hbLo), op));
            resLo = HalfArithNaNFix(resLo, haLo, hbLo,
                HalfNaNMask8(haLo, m7fff, vinf), HalfNaNMask8(hbLo, m7fff, vinf), in2First);

            var haHi = ha.GetUpper(); var hbHi = hb.GetUpper();
            var resHi = HalfNarrow8V(HalfArithOp(HalfWiden8V(haHi), HalfWiden8V(hbHi), op));
            resHi = HalfArithNaNFix(resHi, haHi, hbHi,
                HalfNaNMask8(haHi, m7fff, vinf), HalfNaNMask8(hbHi, m7fff, vinf), in2First);

            return Vector256.Create(resLo, resHi);
        }
    }
}
