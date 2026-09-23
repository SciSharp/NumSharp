using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// DirectILKernelGenerator.Unary.Spacing.Half.cs — float16 np.spacing
// (bit-level, conversion-free)
// =============================================================================
//
// np.spacing at float16 is NumPy's SEPARATE npy_half_spacing routine
// (halffloat.cpp): a pure raw-16-bit bit-fiddle — ALWAYS non-negative, and
// HALVES the ULP at a negative power-of-2 boundary — unlike the SIGNED float32/
// float64 formula. NumPy runs it one element at a time (a plain UNARY_LOOP; the
// AVX-512-FP16 path is Sapphire-Rapids-only and absent from the win-amd64 wheel),
// which is why a naive scalar NumSharp loop was ~0.75x NumPy at 100K/10M. Since
// the routine is integer-only on the ushort pattern it vectorizes over raw lanes
// — the fourth conversion-free f16 family after sign/negate/abs, the comparison/
// min-max group, and floor/ceil/trunc/rint — and then BEATS NumPy.
//
// Per lane, in npy_half_spacing's exact priority order (probed exhaustively on
// 2.4.2, and cross-checked against NDSpacingMath.Spacing(Half) over all 65,536):
//   h_exp = h & 0x7c00 ; h_sig = h & 0x03ff ; expField = h_exp >> 10
//   h_exp == 0x7c00              → 0x7e00 (NPY_HALF_NAN; inf AND nan, both signs)
//   h == 0x7bff                  → 0x7c00 (+inf; largest finite +half overflows)
//   (h & 0x8000) && h_sig == 0   → negative power-of-2 boundary (smaller ULP):
//        h_exp > 0x2c00          → h_exp - 0x2c00        (result normalized)
//        h_exp > 0x0400          → 1 << (expField - 2)   (subnormal, not smallest)
//        else                    → 0x0001                (smallest subnormal)
//   h_exp > 0x2800               → h_exp - 0x2800        (normal)
//   h_exp > 0x0400               → 1 << (expField - 1)   (subnormal, not smallest)
//   else                        → 0x0001
//
// The two `1 << (expField - k)` cases need a per-lane VARIABLE shift, which AVX2
// has only at 32-bit lanes (VPSLLVD) — so, like the rounding kernel, we widen 8
// ushorts to one i32 vector, compute branchlessly with blends (garbage lanes from
// out-of-range shifts are always masked away), and narrow back. Priority is
// reproduced by applying the else-tree first and BlendVariable-ing the higher
// cases over it last-wins (so -inf, which is BOTH NaN/inf AND a neg boundary,
// correctly ends as 0x7e00). Serves UnaryOp.Spacing for contiguous Half→Half
// keys; strided / int-promoted-to-Half keep the scalar NDSpacingMath path.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Contiguous float16 <c>np.spacing</c> kernel (the <see cref="UnaryKernel"/> delegate returned
        /// for a contiguous Half→Half spacing key). Reinterprets both buffers as raw ushort and runs the
        /// branchless <c>npy_half_spacing</c> over AVX2 lanes; the strides/shape/ndim args are unused (the
        /// buffers are contiguous by the dispatch gate).
        /// </summary>
        internal static unsafe void HalfSpacingContiguous(void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
            => HalfSpacingCore((ushort*)input, (ushort*)output, totalSize);

        /// <summary>
        /// Branchless <c>npy_half_spacing</c> over a contiguous ushort run: an AVX2 body (8 lanes,
        /// widened to int32 for the variable shifts) plus a scalar tail. Lane-identical to
        /// <see cref="NDSpacingMath.Spacing(Half)"/> (verified over all 65,536 f16 patterns).
        /// </summary>
        /// <param name="src">Source f16 bit patterns.</param>
        /// <param name="dst">Destination f16 bit patterns.</param>
        /// <param name="n">Element count.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HalfSpacingCore(ushort* src, ushort* dst, long n)
        {
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7c00 = Vector256.Create(0x7C00);
                var m03ff = Vector256.Create(0x03FF);
                var m8000 = Vector256.Create(0x8000);
                var c7e00 = Vector256.Create(0x7E00);
                var c7bff = Vector256.Create(0x7BFF);
                var c7c00v = Vector256.Create(0x7C00);
                var c2c00 = Vector256.Create(0x2C00);
                var c2800 = Vector256.Create(0x2800);
                var c0400 = Vector256.Create(0x0400);
                var one = Vector256.Create(1);
                var two = Vector256.Create(2);
                for (; i + 8 <= n; i += 8)
                {
                    var h = Avx2.ConvertToVector256Int32(Sse2.LoadVector128(src + i)); // zero-extend 8×u16→i32
                    var hExp = Avx2.And(h, m7c00);
                    var hSig = Avx2.And(h, m03ff);
                    var sign = Avx2.And(h, m8000);
                    var expField = Avx2.ShiftRightLogical(hExp, 10);

                    // Case masks (all-ones = true), in npy_half_spacing's priority order.
                    var isNaNInf = Avx2.CompareEqual(hExp, m7c00);
                    var isMaxPos = Avx2.CompareEqual(h, c7bff);
                    var signSet  = Avx2.CompareGreaterThan(sign, Vector256<int>.Zero);
                    var sigZero  = Avx2.CompareEqual(hSig, Vector256<int>.Zero);
                    var isNegBnd = Avx2.And(signSet, sigZero);
                    var expGt2c00 = Avx2.CompareGreaterThan(hExp, c2c00);
                    var expGt2800 = Avx2.CompareGreaterThan(hExp, c2800);
                    var expGt0400 = Avx2.CompareGreaterThan(hExp, c0400);

                    // Variable-shift subnormal candidates (garbage for out-of-range lanes is masked away).
                    var sh1 = Avx2.ShiftLeftLogicalVariable(one, Avx2.Subtract(expField, one).AsUInt32()).AsInt32();  // 1<<(exp-1)
                    var sh2 = Avx2.ShiftLeftLogicalVariable(one, Avx2.Subtract(expField, two).AsUInt32()).AsInt32();  // 1<<(exp-2)
                    var normalNB = Avx2.Subtract(hExp, c2c00);  // neg-boundary normalized ULP
                    var normalE  = Avx2.Subtract(hExp, c2800);  // normal ULP

                    // Negative-boundary sub-tree: normalNB if exp>0x2c00, else sh2 if exp>0x0400, else 0x0001.
                    var nbRes = Avx2.BlendVariable(one, sh2, expGt0400);
                    nbRes = Avx2.BlendVariable(nbRes, normalNB, expGt2c00);

                    // Else sub-tree: normalE if exp>0x2800, else sh1 if exp>0x0400, else 0x0001.
                    var eRes = Avx2.BlendVariable(one, sh1, expGt0400);
                    eRes = Avx2.BlendVariable(eRes, normalE, expGt2800);

                    // Combine by priority (last blend wins): else < negBnd < maxPos < NaN/inf.
                    var res = eRes;
                    res = Avx2.BlendVariable(res, nbRes, isNegBnd);
                    res = Avx2.BlendVariable(res, c7c00v, isMaxPos);
                    res = Avx2.BlendVariable(res, c7e00, isNaNInf);

                    // Narrow 8×i32 (all in [0,0x7e00], never saturates) → 8×u16 and store.
                    Sse2.Store(dst + i, Sse41.PackUnsignedSaturate(res.GetLower(), res.GetUpper()));
                }
            }
            // Scalar tail (and the whole run on a non-AVX2 host): the canonical bit routine.
            for (; i < n; i++)
                dst[i] = BitConverter.HalfToUInt16Bits(NDSpacingMath.Spacing(BitConverter.UInt16BitsToHalf(src[i])));
        }
    }
}
