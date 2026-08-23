using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Reduction.Half.cs — float16 SUM reduce (widen-compute-narrow)
// =============================================================================
//
// .NET has NO Vector<Half> and NO F16C — `Vector128<Half>` throws NotSupported for
// every op — so float16 arithmetic can only be vectorized by holding the raw 16-bit
// patterns as ushort and widening to float32 (Giesen bit-fiddle, shipped + proven
// 0-diff vs NumPy's npy_half_to_float over all 65536 values). NumPy's own C loop does
// exactly this: HALF_add reduces by templating the FLOAT pairwise_sum with
// `@trf@ = npy_half_to_float` and narrows per inner-loop call
// (`*iop1 = npy_float_to_half(io1)`), so the float16 accumulator saturates.
//
// We reproduce NumPy's per-orientation behavior but run it in a **float32 shadow
// accumulator** (the reduce output is allocated as Single; ReduceAdd narrows it to
// Half once at the end). Per inner-loop call the accumulator is rounded to float16
// PRECISION *in float32* via RoundToF16 (== widen(narrow(x)), verified bit-exact),
// which reproduces NumPy's per-call narrow WITHOUT the float16↔memory round-trip:
//   • PINNED (reduced axis contiguous-inner): fold the stripe in float32 pairwise,
//     RoundToF16 the running slot. sum(ones(4096,f16))==4096, [2048,1,1]==2050.
//   • SLAB (reduced axis outer): each kept-axis slot accumulates across outer steps,
//     RoundToF16 per step, so it SATURATES like NumPy — sum(ones((4096,3)),axis=0)
//     == [2048,2048,2048], not [4096,...].
// Keeping the accumulator in the float32 shadow (instead of reloading+re-widening the
// half output every step) + a cheap in-place RoundToF16 + a 4×-unrolled SLAB body is
// what makes the large-contiguous-kept SLAB beat NumPy (whose C build has hardware
// F16C) rather than lose to it.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// NumPy's HALF_pairwise_sum: widen each float16 element to float32 on read and accumulate
        /// in FLOAT32 with the pairwise tree (blocksize 128, eight accumulators, split kept a
        /// multiple of 8). Returns the float32 stripe sum. Bit-for-bit identical to
        /// <c>np.add.reduce(float16)</c>. <paramref name="stride"/> is in ELEMENTS (may be negative).
        /// The contiguous leaf maps NumPy's eight accumulators onto ONE <see cref="Vector256{Single}"/>
        /// (lane k == r[k]); the strided leaf keeps the scalar eight-accumulator block.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe float PairwiseFoldHalf(ushort* a, long n, long stride)
        {
            if (n < 8)
            {
                float res = -0.0f; // seed -0 (NumPy: summing only -0 must stay -0)
                for (long i = 0; i < n; i++) res += DirectILKernelGenerator.HalfToFloatScalarExact(a[i * stride]);
                return res;
            }
            if (n <= 128)
            {
                if (stride == 1 && Avx2.IsSupported)
                {
                    var acc = LoadWiden8(a);
                    long i = 8, end = n - (n % 8);
                    for (; i + 16 <= end; i += 16) { acc = Avx.Add(acc, LoadWiden8(a + i)); acc = Avx.Add(acc, LoadWiden8(a + i + 8)); }
                    for (; i < end; i += 8) acc = Avx.Add(acc, LoadWiden8(a + i));
                    float res = ((acc.GetElement(0) + acc.GetElement(1)) + (acc.GetElement(2) + acc.GetElement(3)))
                              + ((acc.GetElement(4) + acc.GetElement(5)) + (acc.GetElement(6) + acc.GetElement(7)));
                    for (; i < n; i++) res += DirectILKernelGenerator.HalfToFloatScalarExact(a[i]);
                    return res;
                }
                float r0 = HW(a, 0, stride), r1 = HW(a, 1, stride), r2 = HW(a, 2, stride), r3 = HW(a, 3, stride),
                      r4 = HW(a, 4, stride), r5 = HW(a, 5, stride), r6 = HW(a, 6, stride), r7 = HW(a, 7, stride);
                long j = 8, lim = n - (n % 8);
                for (; j < lim; j += 8)
                {
                    r0 += HW(a, j, stride); r1 += HW(a, j + 1, stride); r2 += HW(a, j + 2, stride); r3 += HW(a, j + 3, stride);
                    r4 += HW(a, j + 4, stride); r5 += HW(a, j + 5, stride); r6 += HW(a, j + 6, stride); r7 += HW(a, j + 7, stride);
                }
                float sres = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
                for (; j < n; j++) sres += HW(a, j, stride);
                return sres;
            }
            long n2 = n / 2; n2 -= n2 % 8;
            return PairwiseFoldHalf(a, n2, stride) + PairwiseFoldHalf(a + n2 * stride, n - n2, stride);
        }

        [MethodImpl(OptimizeAndInline)]
        private static unsafe Vector256<float> LoadWiden8(ushort* p)
            => DirectILKernelGenerator.HalfBitsToFloatExact(Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p)));

        // Base Giesen widen (no exact-NaN blend): finite-exact and inf-correct; only the NaN PAYLOAD
        // differs, which is irrelevant for SUM (any NaN narrows to 0x7e00). Cheaper than the exact
        // widen — used for the elementwise SLAB accumulate where the payload never survives. Inlined
        // in place (not a cross-class call) so the JIT keeps the whole widen→add→round chain in regs.
        [MethodImpl(OptimizeAndInline)]
        private static unsafe Vector256<float> BaseWiden8(ushort* p)
        {
            var h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(p));
            var em = Avx2.And(Vector256.Create(0x7fff), h);
            var sc = Avx.Multiply(Avx2.ShiftLeftLogical(em, 13).AsSingle(), Vector256.Create((254 - 15) << 23).AsSingle());
            var infn = Avx2.CompareGreaterThan(em, Vector256.Create(0x7bff));
            var sg = Avx2.ShiftLeftLogical(Avx2.AndNot(Vector256.Create(0x7fff), h), 16);
            var si = Avx.Or(sg.AsSingle(), Avx.And(infn.AsSingle(), Vector256.Create(255 << 23).AsSingle()));
            return Avx.Or(sc, si);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float HW(ushort* a, long i, long stride)
            => DirectILKernelGenerator.HalfToFloatScalarExact(a[i * stride]);

        // Round a float32 to float16 PRECISION, result stays float32 == widen(narrow(v)). Reproduces
        // NumPy's per-inner-loop narrow without the 16-bit round-trip (verified bit-exact to
        // HalfToFloatScalarExact(SingleToHalfBits(v)) over 8M structured+random floats incl. specials).
        [MethodImpl(OptimizeAndInline)]
        private static float RoundToF16Scalar(float v)
            => DirectILKernelGenerator.HalfToFloatScalarExact(DirectILKernelGenerator.SingleToHalfBits(v));

        [MethodImpl(OptimizeAndInline)]
        private static unsafe Vector256<float> RoundToF16(Vector256<float> v)
        {
            var vi = v.AsInt32();
            var a = Avx2.And(vi, Vector256.Create(0x7fffffff));
            var sign = Avx2.And(vi, Vector256.Create(unchecked((int)0x80000000)));
            // RTNE round the mantissa to 10 bits (the f16-precision round for the normal range).
            var rounded = Avx2.And(Avx2.Add(Avx2.Add(a, Vector256.Create(0xfff)), Avx2.And(Avx2.ShiftRightLogical(a, 13), Vector256.Create(1))),
                                   Vector256.Create(unchecked((int)0xffffe000)));
            var res = Avx2.Or(sign, rounded);
            // saturate to ±inf where the round crosses the f16 max (65504 → 65536).
            res = Avx2.BlendVariable(res, Avx2.Or(sign, Vector256.Create(0x7f800000)),
                                     Avx2.CompareGreaterThan(rounded, Vector256.Create(0x477fffff)));
            // NaN → canonical (sign|0x7fc00000), matching widen(narrow(nan)).
            res = Avx2.BlendVariable(res, Avx2.Or(sign, Vector256.Create(0x7fc00000)),
                                     Avx2.CompareGreaterThan(a, Vector256.Create(0x7f800000)));
            // subnormal f16 (|v| < 2^-14): rare — fix those lanes exactly via scalar widen(narrow).
            var sub = Avx2.CompareGreaterThan(Vector256.Create(0x38800000), a);
            if (!Avx.TestZ(sub, sub))
            {
                float* t = stackalloc float[8]; res.AsSingle().Store(t);
                float* vv = stackalloc float[8]; v.Store(vv);
                for (int k = 0; k < 8; k++)
                    if ((uint)(BitConverter.SingleToInt32Bits(vv[k]) & 0x7fffffff) < 0x38800000u)
                        t[k] = RoundToF16Scalar(vv[k]);
                return Vector256.Load(t);
            }
            return res.AsSingle();
        }

        /// <summary>
        /// Per-chunk reduce kernel for float16 SUM into a FLOAT32 shadow accumulator
        /// (<c>ReduceKernelKey(Sum, Half, Single)</c>; ReduceAdd narrows the Single result to Half).
        /// PINNED (<c>outStride==0</c>): fold the stripe in float32 pairwise, RoundToF16 the running
        /// slot. SLAB (<c>outStride!=0</c>): each kept-axis slot accumulates across the outer (reduced)
        /// steps with a per-step RoundToF16 — the saturating orientation, matching NumPy's per-call
        /// narrow while staying in float32 (no half↔memory round-trip). SLAB is 4×-unrolled.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        internal static unsafe void HalfSumKernel(void** dataptrs, long* strides, long count, void* auxdata)
        {
            ushort* inp = (ushort*)dataptrs[0];
            float* outp = (float*)dataptrs[1];
            long inS = strides[0];   // BYTES (half input)
            long outS = strides[1];  // BYTES (float32 shadow)

            if (outS == 0)
            {
                // PINNED: out_f32 = RoundToF16( out_f32 + pairwise_f32(stripe) )
                *outp = RoundToF16Scalar(*outp + PairwiseFoldHalf(inp, count, inS / 2));
            }
            else
            {
                long os = outS / 4, isr = inS / 2;
                long c = 0;
                if (os == 1 && isr == 1 && Avx2.IsSupported)
                {
                    // Contiguous kept axis: out_f32[c] = RoundToF16( out_f32[c] + widen(in[c]) ). Each
                    // lane is an INDEPENDENT output slot, so the vector body is bit-for-bit identical to
                    // the scalar per-element round. 4×-unrolled (32 halves/iter) to hide the
                    // widen→add→round latency chain across independent 8-blocks.
                    long end32 = count - (count & 31);
                    for (; c < end32; c += 32)
                    {
                        RoundToF16(Avx.Add(Vector256.Load(outp + c),      BaseWiden8(inp + c))).Store(outp + c);
                        RoundToF16(Avx.Add(Vector256.Load(outp + c + 8),  BaseWiden8(inp + c + 8))).Store(outp + c + 8);
                        RoundToF16(Avx.Add(Vector256.Load(outp + c + 16), BaseWiden8(inp + c + 16))).Store(outp + c + 16);
                        RoundToF16(Avx.Add(Vector256.Load(outp + c + 24), BaseWiden8(inp + c + 24))).Store(outp + c + 24);
                    }
                    long end8 = count - (count & 7);
                    for (; c < end8; c += 8)
                        RoundToF16(Avx.Add(Vector256.Load(outp + c), BaseWiden8(inp + c))).Store(outp + c);
                    for (; c < count; c++)
                        outp[c] = RoundToF16Scalar(outp[c] + DirectILKernelGenerator.HalfToFloatScalarExact(inp[c]));
                }
                else
                {
                    for (; c < count; c++)
                        outp[c * os] = RoundToF16Scalar(outp[c * os] + DirectILKernelGenerator.HalfToFloatScalarExact(inp[c * isr]));
                }
            }
        }
    }
}
