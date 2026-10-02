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
        internal static unsafe MixedTypeKernel TryGetHalfArithKernel(BinaryOp op, ExecutionPath path)
        {
            if (op != BinaryOp.Add && op != BinaryOp.Subtract && op != BinaryOp.Multiply && op != BinaryOp.Divide)
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
        /// Strided / broadcast (Half,Half)→Half arithmetic (SimdChunk / General paths):
        /// a per-element coordinate odometer — EmitGeneralLoop's exact address model —
        /// driving the bit-exact scalar widen-compute-narrow (<see cref="HalfArithBits"/>).
        /// For each output element i (written C-contiguous, linearly), i is decomposed into
        /// coordinates over <paramref name="shape"/> (C-order, last axis fastest) and each
        /// operand is read at Σ coord_d·stride_d from its (offset-adjusted) base — so it is
        /// correct for ANY layout: contiguous, broadcast (stride 0), negative stride, or
        /// step (|stride| &gt; 1). Bit-identical to NumPy because HalfArithBits IS NumPy's
        /// scalar HALF loop (float32 compute + RTNE narrow + the operand-order NaN pin);
        /// only the addressing changes vs the contiguous kernels. Scalar by construction —
        /// the contiguous SimdFull / SimdScalar paths keep the SIMD widen-compute-narrow.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfArithStrided(
            ushort* pa, ushort* pb, ushort* pr,
            long* lhsStrides, long* rhsStrides, long* shape, int ndim, long n, BinaryOp op)
        {
            for (long i = 0; i < n; i++)
            {
                long idx = i, lhsOff = 0, rhsOff = 0;
                for (int d = ndim - 1; d >= 0; d--)
                {
                    long dim = shape[d];
                    long coord = idx % dim;
                    idx /= dim;
                    lhsOff += coord * lhsStrides[d];
                    rhsOff += coord * rhsStrides[d];
                }
                pr[i] = HalfArithBits(pa[lhsOff], pb[rhsOff], op);
            }
        }

        /// <summary>
        /// Scalar lane — the exact wheel pipeline with the NaN priority encoded EXPLICITLY:
        /// Add/Multiply resolve in2 first, Subtract/Divide in1 first; |0x0200 is x86
        /// "quieted" (identity on qNaN, sign kept).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ushort HalfArithBits(ushort a, ushort b, BinaryOp op)
        {
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
