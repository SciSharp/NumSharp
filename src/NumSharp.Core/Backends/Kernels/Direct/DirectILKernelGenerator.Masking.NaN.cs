using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Masking.NaN.cs - NaN-aware SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - NanSumSimdHelperFloat/Double - sum ignoring NaN values
//   - NanProdSimdHelperFloat/Double - product ignoring NaN values
//   - NanMinSimdHelperFloat/Double - min ignoring NaN values
//   - NanMaxSimdHelperFloat/Double - max ignoring NaN values
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region NaN-aware SIMD Helpers
        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        /// <param name="src">Pointer to contiguous float data</param>
        /// <param name="size">Number of elements</param>
        /// <returns>Sum of non-NaN elements</returns>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe float NanSumSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return 0f;

            float sum = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<float>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Create mask where NaN becomes 0, valid values stay
                    // NaN comparison: x != x is true for NaN
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN, false for NaN
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<float>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous double array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe double NanSumSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return 0.0;

            double sum = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<double>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<double>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous float array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe float NanProdSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return 1f;

            float prod = 1f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1f);
                var oneVec = Vector256.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with 1: if NaN, use 1; otherwise use original
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                // Horizontal product (no built-in, do manually)
                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1f);
                var oneVec = Vector128.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous double array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe double NanProdSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return 1.0;

            double prod = 1.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1.0);
                var oneVec = Vector256.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1.0);
                var oneVec = Vector128.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        // =====================================================================
        // NaN-aware flat min/max — 8x-unrolled raw MIN/MAX skip-NaN reducers.
        //
        // These are np.nanmin / np.nanmax over a contiguous buffer (NumPy's
        // fmin.reduce / fmax.reduce). The old kernels were a SINGLE-ACCUMULATOR,
        // NON-unrolled loop of THREE vector ops per step — a self-equality NaN
        // mask, a ConditionalSelect to blank NaN to ±inf, and Vector256.Max
        // (which on net9+ carries an IEEE NaN-propagation FIXUP the blanking
        // makes redundant). The carried dependency on one accumulator left it
        // latency-bound (~2.3x NumPy on 10M f32).
        //
        // The rewrite uses the x86 MAXPS/MINPS skip-NaN identity directly: with
        // the DATA vector as SRC1 and the accumulator as SRC2, MAXPS returns
        // SRC2 whenever SRC1 is NaN — so a NaN lane is dropped in ONE instruction
        // (no mask, no select, no fixup), and since the accumulator is seeded
        // ±inf it never becomes NaN. Eight independent accumulators break the
        // dependency chain (both MAX ports saturated), so the loop runs at DRAM
        // bandwidth. All-NaN → NaN is recovered exactly: the raw reduction yields
        // the seed (±inf) for an all-NaN OR a genuinely ±inf-extremal array, so a
        // cold scalar rescan (taken only when the result IS the seed — vanishingly
        // rare) distinguishes them. NaN sign is NumPy's canonical quiet NaN.
        //
        // NaN skip / all-NaN->NaN / ±inf are bit-exact with NumPy; -0/+0 tie sign
        // is order-dependent (as the previous kernel was), never reached by any
        // corpus input. Non-x86 / sub-64-element inputs take the scalar tail,
        // which is itself fmin/fmax-exact.
        // =====================================================================

        /// <summary>SIMD helper for NaN-aware minimum of a contiguous float array (NaN ignored; all-NaN -> NaN).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe float NanMinSimdHelperFloat(float* src, long size)
            => NanMinMaxFloatCore(src, size, isMax: false);

        /// <summary>SIMD helper for NaN-aware maximum of a contiguous float array (NaN ignored; all-NaN -> NaN).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe float NanMaxSimdHelperFloat(float* src, long size)
            => NanMinMaxFloatCore(src, size, isMax: true);

        /// <summary>SIMD helper for NaN-aware minimum of a contiguous double array (NaN ignored; all-NaN -> NaN).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe double NanMinSimdHelperDouble(double* src, long size)
            => NanMinMaxDoubleCore(src, size, isMax: false);

        /// <summary>SIMD helper for NaN-aware maximum of a contiguous double array (NaN ignored; all-NaN -> NaN).</summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe double NanMaxSimdHelperDouble(double* src, long size)
            => NanMinMaxDoubleCore(src, size, isMax: true);

        /// <summary>NumPy's canonical quiet NaN (positive), used for an all-NaN slice.</summary>
        private const int NanBitsF32 = 0x7fc00000;
        private const long NanBitsF64 = 0x7ff8000000000000L;

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe float NanMinMaxFloatCore(float* src, long size, bool isMax)
        {
            if (size == 0)
                return BitConverter.Int32BitsToSingle(NanBitsF32);

            // Seed = the op identity that also survives the raw MAXPS/MINPS skip: max->-inf, min->+inf.
            float seed = isMax ? float.NegativeInfinity : float.PositiveInfinity;
            const long Vl = 8;              // Vector256<float>.Count
            const long Unroll = Vl * 8;     // 64 elements / iteration

            if (Avx.IsSupported && size >= Unroll)
            {
                var s = Vector256.Create(seed);
                var a0 = s; var a1 = s; var a2 = s; var a3 = s;
                var a4 = s; var a5 = s; var a6 = s; var a7 = s;
                long i = 0;
                long end = size - Unroll;
                // DATA vector is SRC1 -> a NaN lane returns the accumulator (SRC2); the seed keeps
                // the accumulator non-NaN, so NaNs are skipped with no mask/select/fixup.
                if (isMax)
                    for (; i <= end; i += Unroll)
                    {
                        a0 = Avx.Max(Avx.LoadVector256(src + i), a0);
                        a1 = Avx.Max(Avx.LoadVector256(src + i + Vl), a1);
                        a2 = Avx.Max(Avx.LoadVector256(src + i + Vl * 2), a2);
                        a3 = Avx.Max(Avx.LoadVector256(src + i + Vl * 3), a3);
                        a4 = Avx.Max(Avx.LoadVector256(src + i + Vl * 4), a4);
                        a5 = Avx.Max(Avx.LoadVector256(src + i + Vl * 5), a5);
                        a6 = Avx.Max(Avx.LoadVector256(src + i + Vl * 6), a6);
                        a7 = Avx.Max(Avx.LoadVector256(src + i + Vl * 7), a7);
                    }
                else
                    for (; i <= end; i += Unroll)
                    {
                        a0 = Avx.Min(Avx.LoadVector256(src + i), a0);
                        a1 = Avx.Min(Avx.LoadVector256(src + i + Vl), a1);
                        a2 = Avx.Min(Avx.LoadVector256(src + i + Vl * 2), a2);
                        a3 = Avx.Min(Avx.LoadVector256(src + i + Vl * 3), a3);
                        a4 = Avx.Min(Avx.LoadVector256(src + i + Vl * 4), a4);
                        a5 = Avx.Min(Avx.LoadVector256(src + i + Vl * 5), a5);
                        a6 = Avx.Min(Avx.LoadVector256(src + i + Vl * 6), a6);
                        a7 = Avx.Min(Avx.LoadVector256(src + i + Vl * 7), a7);
                    }

                Vector256<float> m = isMax
                    ? Avx.Max(Avx.Max(Avx.Max(a0, a1), Avx.Max(a2, a3)), Avx.Max(Avx.Max(a4, a5), Avx.Max(a6, a7)))
                    : Avx.Min(Avx.Min(Avx.Min(a0, a1), Avx.Min(a2, a3)), Avx.Min(Avx.Min(a4, a5), Avx.Min(a6, a7)));

                if (isMax) for (; i + Vl <= size; i += Vl) m = Avx.Max(Avx.LoadVector256(src + i), m);
                else       for (; i + Vl <= size; i += Vl) m = Avx.Min(Avx.LoadVector256(src + i), m);

                // Horizontal reduce (m carries no NaN) + scalar tail (NaN skipped by the ordered compare).
                float r = seed;
                for (int k = 0; k < (int)Vl; k++) { float e = m.GetElement(k); if (isMax ? e > r : e < r) r = e; }
                for (; i < size; i++) { float x = src[i]; if (isMax ? x > r : x < r) r = x; }

                if (isMax ? r != float.NegativeInfinity : r != float.PositiveInfinity)
                    return r;                       // has a finite/±inf value — the common exit
                // r == seed: all-NaN, OR a genuinely ±inf extremal array. Distinguish with a rescan.
                for (long k = 0; k < size; k++) if (!float.IsNaN(src[k])) return r;  // r is the real ±inf
                return BitConverter.Int32BitsToSingle(NanBitsF32);                    // all-NaN
            }

            // Scalar path (small / non-x86): fmin/fmax-exact.
            {
                float r = seed; bool any = false;
                for (long i = 0; i < size; i++)
                {
                    float x = src[i];
                    if (!float.IsNaN(x)) { any = true; if (isMax ? x > r : x < r) r = x; }
                }
                return any ? r : BitConverter.Int32BitsToSingle(NanBitsF32);
            }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
        private static unsafe double NanMinMaxDoubleCore(double* src, long size, bool isMax)
        {
            if (size == 0)
                return BitConverter.Int64BitsToDouble(NanBitsF64);

            double seed = isMax ? double.NegativeInfinity : double.PositiveInfinity;
            const long Vl = 4;              // Vector256<double>.Count
            const long Unroll = Vl * 8;     // 32 elements / iteration

            if (Avx.IsSupported && size >= Unroll)
            {
                var s = Vector256.Create(seed);
                var a0 = s; var a1 = s; var a2 = s; var a3 = s;
                var a4 = s; var a5 = s; var a6 = s; var a7 = s;
                long i = 0;
                long end = size - Unroll;
                if (isMax)
                    for (; i <= end; i += Unroll)
                    {
                        a0 = Avx.Max(Avx.LoadVector256(src + i), a0);
                        a1 = Avx.Max(Avx.LoadVector256(src + i + Vl), a1);
                        a2 = Avx.Max(Avx.LoadVector256(src + i + Vl * 2), a2);
                        a3 = Avx.Max(Avx.LoadVector256(src + i + Vl * 3), a3);
                        a4 = Avx.Max(Avx.LoadVector256(src + i + Vl * 4), a4);
                        a5 = Avx.Max(Avx.LoadVector256(src + i + Vl * 5), a5);
                        a6 = Avx.Max(Avx.LoadVector256(src + i + Vl * 6), a6);
                        a7 = Avx.Max(Avx.LoadVector256(src + i + Vl * 7), a7);
                    }
                else
                    for (; i <= end; i += Unroll)
                    {
                        a0 = Avx.Min(Avx.LoadVector256(src + i), a0);
                        a1 = Avx.Min(Avx.LoadVector256(src + i + Vl), a1);
                        a2 = Avx.Min(Avx.LoadVector256(src + i + Vl * 2), a2);
                        a3 = Avx.Min(Avx.LoadVector256(src + i + Vl * 3), a3);
                        a4 = Avx.Min(Avx.LoadVector256(src + i + Vl * 4), a4);
                        a5 = Avx.Min(Avx.LoadVector256(src + i + Vl * 5), a5);
                        a6 = Avx.Min(Avx.LoadVector256(src + i + Vl * 6), a6);
                        a7 = Avx.Min(Avx.LoadVector256(src + i + Vl * 7), a7);
                    }

                Vector256<double> m = isMax
                    ? Avx.Max(Avx.Max(Avx.Max(a0, a1), Avx.Max(a2, a3)), Avx.Max(Avx.Max(a4, a5), Avx.Max(a6, a7)))
                    : Avx.Min(Avx.Min(Avx.Min(a0, a1), Avx.Min(a2, a3)), Avx.Min(Avx.Min(a4, a5), Avx.Min(a6, a7)));

                if (isMax) for (; i + Vl <= size; i += Vl) m = Avx.Max(Avx.LoadVector256(src + i), m);
                else       for (; i + Vl <= size; i += Vl) m = Avx.Min(Avx.LoadVector256(src + i), m);

                double r = seed;
                for (int k = 0; k < (int)Vl; k++) { double e = m.GetElement(k); if (isMax ? e > r : e < r) r = e; }
                for (; i < size; i++) { double x = src[i]; if (isMax ? x > r : x < r) r = x; }

                if (isMax ? r != double.NegativeInfinity : r != double.PositiveInfinity)
                    return r;
                for (long k = 0; k < size; k++) if (!double.IsNaN(src[k])) return r;
                return BitConverter.Int64BitsToDouble(NanBitsF64);
            }

            {
                double r = seed; bool any = false;
                for (long i = 0; i < size; i++)
                {
                    double x = src[i];
                    if (!double.IsNaN(x)) { any = true; if (isMax ? x > r : x < r) r = x; }
                }
                return any ? r : BitConverter.Int64BitsToDouble(NanBitsF64);
            }
        }



        /// <summary>
        /// IL-generated SIMD helper for NaN-aware variance of a contiguous float array.
        /// Returns NaN if all values are NaN or count <= ddof.
        /// Two-pass algorithm: (1) compute mean with count, (2) compute squared differences.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe float NanVarSimdHelperFloat(float* src, long size, int ddof = 0)
        {
            if (size == 0)
                return float.NaN;

            // Pass 1: Compute sum and count
            float sum = 0f;
            float count = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<float>.Zero;
                var countVec = Vector256<float>.Zero;
                var oneVec = Vector256.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector256.Add(sumVec, cleaned);
                    var countMask = Vector256.BitwiseAnd(oneVec, nanMask.AsSingle());
                    countVec = Vector256.Add(countVec, countMask);
                }

                sum = Vector256.Sum(sumVec);
                count = Vector256.Sum(countVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1f;
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<float>.Zero;
                var countVec = Vector128<float>.Zero;
                var oneVec = Vector128.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector128.Add(sumVec, cleaned);
                    var countMask = Vector128.BitwiseAnd(oneVec, nanMask.AsSingle());
                    countVec = Vector128.Add(countVec, countMask);
                }

                sum = Vector128.Sum(sumVec);
                count = Vector128.Sum(countVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1f;
                    }
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1f;
                    }
                }
            }

            if (count <= ddof)
                return float.NaN;

            float mean = sum / count;

            // Pass 2: Compute sum of squared differences
            float sqDiffSum = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var sqDiffVec = Vector256<float>.Zero;
                var meanVec = Vector256.Create(mean);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var diff = Vector256.Subtract(vec, meanVec);
                    var sqDiff = Vector256.Multiply(diff, diff);
                    var cleaned = Vector256.BitwiseAnd(sqDiff, nanMask.AsSingle());
                    sqDiffVec = Vector256.Add(sqDiffVec, cleaned);
                }

                sqDiffSum = Vector256.Sum(sqDiffVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        float diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var sqDiffVec = Vector128<float>.Zero;
                var meanVec = Vector128.Create(mean);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var diff = Vector128.Subtract(vec, meanVec);
                    var sqDiff = Vector128.Multiply(diff, diff);
                    var cleaned = Vector128.BitwiseAnd(sqDiff, nanMask.AsSingle());
                    sqDiffVec = Vector128.Add(sqDiffVec, cleaned);
                }

                sqDiffSum = Vector128.Sum(sqDiffVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        float diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        float diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }

            return sqDiffSum / (count - ddof);
        }

        /// <summary>
        /// IL-generated SIMD helper for NaN-aware variance of a contiguous double array.
        /// Returns NaN if all values are NaN or count <= ddof.
        /// Two-pass algorithm: (1) compute mean with count, (2) compute squared differences.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe double NanVarSimdHelperDouble(double* src, long size, int ddof = 0)
        {
            if (size == 0)
                return double.NaN;

            // Pass 1: Compute sum and count
            double sum = 0.0;
            double count = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<double>.Zero;
                var countVec = Vector256<double>.Zero;
                var oneVec = Vector256.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector256.Add(sumVec, cleaned);
                    var countMask = Vector256.BitwiseAnd(oneVec, nanMask.AsDouble());
                    countVec = Vector256.Add(countVec, countMask);
                }

                sum = Vector256.Sum(sumVec);
                count = Vector256.Sum(countVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1.0;
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<double>.Zero;
                var countVec = Vector128<double>.Zero;
                var oneVec = Vector128.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector128.Add(sumVec, cleaned);
                    var countMask = Vector128.BitwiseAnd(oneVec, nanMask.AsDouble());
                    countVec = Vector128.Add(countVec, countMask);
                }

                sum = Vector128.Sum(sumVec);
                count = Vector128.Sum(countVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1.0;
                    }
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        sum += src[i];
                        count += 1.0;
                    }
                }
            }

            if (count <= ddof)
                return double.NaN;

            double mean = sum / count;

            // Pass 2: Compute sum of squared differences
            double sqDiffSum = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var sqDiffVec = Vector256<double>.Zero;
                var meanVec = Vector256.Create(mean);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var diff = Vector256.Subtract(vec, meanVec);
                    var sqDiff = Vector256.Multiply(diff, diff);
                    var cleaned = Vector256.BitwiseAnd(sqDiff, nanMask.AsDouble());
                    sqDiffVec = Vector256.Add(sqDiffVec, cleaned);
                }

                sqDiffSum = Vector256.Sum(sqDiffVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        double diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var sqDiffVec = Vector128<double>.Zero;
                var meanVec = Vector128.Create(mean);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var diff = Vector128.Subtract(vec, meanVec);
                    var sqDiff = Vector128.Multiply(diff, diff);
                    var cleaned = Vector128.BitwiseAnd(sqDiff, nanMask.AsDouble());
                    sqDiffVec = Vector128.Add(sqDiffVec, cleaned);
                }

                sqDiffSum = Vector128.Sum(sqDiffVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        double diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        double diff = src[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
            }

            return sqDiffSum / (count - ddof);
        }



        #endregion

        #region NaN-aware Half Helpers

        /// <summary>
        /// Helper for NaN-aware sum of a contiguous Half array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        internal static unsafe Half NanSumHalfHelper(Half* src, long size)
        {
            if (size == 0)
                return Half.Zero;

            double sum = 0.0; // Use double for precision during accumulation
            for (long i = 0; i < size; i++)
            {
                if (!Half.IsNaN(src[i]))
                    sum += (double)src[i];
            }
            return (Half)sum;
        }

        /// <summary>
        /// Helper for NaN-aware product of a contiguous Half array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe Half NanProdHalfHelper(Half* src, long size)
        {
            if (size == 0)
                return (Half)1.0;

            double prod = 1.0; // Use double for precision during accumulation
            for (long i = 0; i < size; i++)
            {
                if (!Half.IsNaN(src[i]))
                    prod *= (double)src[i];
            }
            return (Half)prod;
        }

        /// <summary>
        /// Helper for NaN-aware min of a contiguous Half array.
        /// NaN values are ignored. Returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe Half NanMinHalfHelper(Half* src, long size)
        {
            if (size == 0)
                return Half.NaN;

            Half minVal = Half.PositiveInfinity;
            bool foundNonNaN = false;
            for (long i = 0; i < size; i++)
            {
                if (!Half.IsNaN(src[i]))
                {
                    if (src[i] < minVal) minVal = src[i];
                    foundNonNaN = true;
                }
            }
            return foundNonNaN ? minVal : Half.NaN;
        }

        /// <summary>
        /// Helper for NaN-aware max of a contiguous Half array.
        /// NaN values are ignored. Returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe Half NanMaxHalfHelper(Half* src, long size)
        {
            if (size == 0)
                return Half.NaN;

            Half maxVal = Half.NegativeInfinity;
            bool foundNonNaN = false;
            for (long i = 0; i < size; i++)
            {
                if (!Half.IsNaN(src[i]))
                {
                    if (src[i] > maxVal) maxVal = src[i];
                    foundNonNaN = true;
                }
            }
            return foundNonNaN ? maxVal : Half.NaN;
        }

        #endregion
    }
}
