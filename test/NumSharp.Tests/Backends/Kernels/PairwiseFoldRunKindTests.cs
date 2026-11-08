using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels
{
    /// <summary>
    ///     The emitted pairwise folds (<see cref="ILKernelGenerator.TryGetPairwiseFoldDouble"/>, …Single, …Complex and
    ///     <see cref="ILKernelGenerator.PairwiseFoldHalf"/>) against an independent scalar port of NumPy's
    ///     <c>pairwise_sum</c> written here, bit for bit, for every length 0..300 and some long ones, at the strides with
    ///     their own run kinds (1, -1, 2) and at strides that keep the generic leaf (-2, 3).
    /// </summary>
    /// <remarks>
    ///     The run kinds load each block differently (one load; a reversed load; two overlapping loads blended) and
    ///     compute two sibling leaves in one loop, so a wrong lane, a missed tail, a leaf boundary off by one block or a
    ///     swapped combine operand all change the bits of SOME length — the values span ~120 binades (float64) so no
    ///     sum is exact in a different order. The reference is deliberately the plain recursion: n &lt; 8 a -0.0-seeded
    ///     sequential sum, n ≤ 128 eight scalar accumulators (four per part for complex) and the tree combine, a longer
    ///     run split at n/2 rounded down to a multiple of 8 (complex: (n - n%8)/2).
    /// </remarks>
    [TestClass]
    public unsafe class PairwiseFoldRunKindTests
    {
        private static readonly long[] Strides = { 1, -1, 2, -2, 3 };

        private static IEnumerable<long> Lengths()
        {
            for (long n = 0; n <= 300; n++) yield return n;
            foreach (long n in new long[] { 383, 384, 385, 511, 512, 1000, 1023, 1024, 4097, 10007 })
                yield return n;
        }

        /// <summary>A native buffer of <paramref name="count"/> values spread over many binades.</summary>
        private static double* WideDoubles(long count, int seed, int expSpan)
        {
            var rng = new System.Random(seed);
            var p = (double*)NativeMemory.Alloc((nuint)(count * sizeof(double)));
            for (long i = 0; i < count; i++)
                p[i] = System.Math.ScaleB(rng.NextDouble() - 0.5, rng.Next(-expSpan, expSpan));
            return p;
        }

        /// <summary>The address of logical element 0 of a run of <paramref name="n"/> elements at <paramref name="stride"/>
        /// inside a buffer of <paramref name="extent"/> elements (a negative stride starts at the high end).</summary>
        private static long StartIndex(long n, long stride, long extent)
            => stride >= 0 ? 0 : extent - 1;

        private static long Extent(long n, long stride) => System.Math.Max(1, (n == 0 ? 0 : (n - 1) * System.Math.Abs(stride)) + 1);

        // ---------------- reference: NumPy's pairwise_sum, scalar ----------------

        private static double RefDouble(double* a, long n, long s)
        {
            if (n < 8) { double r = -0.0; for (long i = 0; i < n; i++) r += a[i * s]; return r; }
            if (n <= 128)
            {
                var r = new double[8];
                for (int k = 0; k < 8; k++) r[k] = a[k * s];
                long i = 8;
                for (; i < n - n % 8; i += 8)
                    for (int k = 0; k < 8; k++) r[k] += a[(i + k) * s];
                double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; i++) res += a[i * s];
                return res;
            }
            long n2 = n / 2; n2 -= n2 % 8;
            return RefDouble(a, n2, s) + RefDouble(a + n2 * s, n - n2, s);
        }

        private static float RefSingle(float* a, long n, long s)
        {
            if (n < 8) { float r = -0.0f; for (long i = 0; i < n; i++) r += a[i * s]; return r; }
            if (n <= 128)
            {
                var r = new float[8];
                for (int k = 0; k < 8; k++) r[k] = a[k * s];
                long i = 8;
                for (; i < n - n % 8; i += 8)
                    for (int k = 0; k < 8; k++) r[k] += a[(i + k) * s];
                float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; i++) res += a[i * s];
                return res;
            }
            long n2 = n / 2; n2 -= n2 % 8;
            return RefSingle(a, n2, s) + RefSingle(a + n2 * s, n - n2, s);
        }

        private static float RefHalf(ushort* a, long n, long s)
        {
            static float W(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);
            if (n < 8) { float r = -0.0f; for (long i = 0; i < n; i++) r += W(a[i * s]); return r; }
            if (n <= 128)
            {
                var r = new float[8];
                for (int k = 0; k < 8; k++) r[k] = W(a[k * s]);
                long i = 8;
                for (; i < n - n % 8; i += 8)
                    for (int k = 0; k < 8; k++) r[k] += W(a[(i + k) * s]);
                float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for (; i < n; i++) res += W(a[i * s]);
                return res;
            }
            long n2 = n / 2; n2 -= n2 % 8;
            return RefHalf(a, n2, s) + RefHalf(a + n2 * s, n - n2, s);
        }

        /// <summary>NumPy's CDOUBLE_pairwise_sum over complexes at a stride of <paramref name="s"/> complexes.</summary>
        private static (double Re, double Im) RefComplex(double* a, long nc, long s)
        {
            if (nc < 4)
            {
                double rr = -0.0, ri = -0.0;
                for (long c = 0; c < nc; c++) { rr += a[2 * c * s]; ri += a[2 * c * s + 1]; }
                return (rr, ri);
            }
            if (nc <= 64)
            {
                var r = new double[8];
                for (int k = 0; k < 4; k++) { r[2 * k] = a[2 * k * s]; r[2 * k + 1] = a[2 * k * s + 1]; }
                long c = 4;
                for (; c < nc - nc % 4; c += 4)
                    for (int k = 0; k < 4; k++) { r[2 * k] += a[2 * (c + k) * s]; r[2 * k + 1] += a[2 * (c + k) * s + 1]; }
                double rr = (r[0] + r[2]) + (r[4] + r[6]);
                double ri = (r[1] + r[3]) + (r[5] + r[7]);
                for (; c < nc; c++) { rr += a[2 * c * s]; ri += a[2 * c * s + 1]; }
                return (rr, ri);
            }
            long left = (nc - nc % 8) / 2;
            var lhs = RefComplex(a, left, s);
            var rhs = RefComplex(a + 2 * left * s, nc - left, s);
            return (lhs.Re + rhs.Re, lhs.Im + rhs.Im);
        }

        // ---------------- tests ----------------

        [TestMethod]
        public void Float64Fold_EveryRunKind_IsNumPysPairwiseSum()
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldDouble();
            if (fold is null) Assert.Inconclusive("no emitted float64 pairwise fold on this host");
            var fails = new List<string>();
            foreach (long s in Strides)
                foreach (long n in Lengths())
                {
                    long extent = Extent(n, s);
                    double* buf = WideDoubles(extent, (int)(n * 31 + s + 1000), 60);
                    try
                    {
                        double* a = buf + StartIndex(n, s, extent);
                        double expected = RefDouble(a, n, s), actual = fold(a, n, s);
                        if (BitConverter.DoubleToInt64Bits(expected) != BitConverter.DoubleToInt64Bits(actual))
                            fails.Add($"n={n} stride={s}: {expected:R} vs {actual:R}");
                    }
                    finally { NativeMemory.Free(buf); }
                }
            fails.Should().BeEmpty();
        }

        [TestMethod]
        public void Float32Fold_EveryRunKind_IsNumPysPairwiseSum()
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldSingle();
            if (fold is null) Assert.Inconclusive("no emitted float32 pairwise fold on this host");
            var fails = new List<string>();
            foreach (long s in Strides)
                foreach (long n in Lengths())
                {
                    long extent = Extent(n, s);
                    double* src = WideDoubles(extent, (int)(n * 37 + s + 2000), 30);
                    float* buf = (float*)NativeMemory.Alloc((nuint)(extent * sizeof(float)));
                    try
                    {
                        for (long i = 0; i < extent; i++) buf[i] = (float)src[i];
                        float* a = buf + StartIndex(n, s, extent);
                        float expected = RefSingle(a, n, s), actual = fold(a, n, s);
                        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
                            fails.Add($"n={n} stride={s}: {expected:R} vs {actual:R}");
                    }
                    finally { NativeMemory.Free(buf); NativeMemory.Free(src); }
                }
            fails.Should().BeEmpty();
        }

        [TestMethod]
        public void Float16Fold_EveryRunKind_IsNumPysPairwiseSum()
        {
            var fails = new List<string>();
            foreach (long s in Strides)
                foreach (long n in Lengths())
                {
                    long extent = Extent(n, s);
                    var rng = new System.Random((int)(n * 41 + s + 3000));
                    ushort* buf = (ushort*)NativeMemory.Alloc((nuint)(extent * sizeof(ushort)));
                    try
                    {
                        for (long i = 0; i < extent; i++)
                            buf[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextDouble() - 0.5) * 100));
                        ushort* a = buf + StartIndex(n, s, extent);
                        float expected = RefHalf(a, n, s), actual = ILKernelGenerator.PairwiseFoldHalf(a, n, s);
                        if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
                            fails.Add($"n={n} stride={s}: {expected:R} vs {actual:R}");
                    }
                    finally { NativeMemory.Free(buf); }
                }
            fails.Should().BeEmpty();
        }

        [TestMethod]
        public void Complex128Fold_EveryRunKind_IsNumPysPairwiseSum()
        {
            var fold = ILKernelGenerator.TryGetPairwiseFoldComplex();
            if (fold is null) Assert.Inconclusive("no emitted complex128 pairwise fold on this host");
            var fails = new List<string>();
            foreach (long s in Strides)
                foreach (long n in Lengths())
                {
                    long extent = Extent(n, s);
                    double* buf = WideDoubles(2 * extent, (int)(n * 43 + s + 4000), 60);
                    try
                    {
                        double* a = buf + 2 * StartIndex(n, s, extent);
                        var expected = RefComplex(a, n, s);
                        var actual = fold(a, n, s * 16);
                        if (BitConverter.DoubleToInt64Bits(expected.Re) != BitConverter.DoubleToInt64Bits(actual.GetElement(0))
                            || BitConverter.DoubleToInt64Bits(expected.Im) != BitConverter.DoubleToInt64Bits(actual.GetElement(1)))
                            fails.Add($"n={n} stride={s}: ({expected.Re:R},{expected.Im:R}) vs ({actual.GetElement(0):R},{actual.GetElement(1):R})");
                    }
                    finally { NativeMemory.Free(buf); }
                }
            fails.Should().BeEmpty();
        }
    }
}
