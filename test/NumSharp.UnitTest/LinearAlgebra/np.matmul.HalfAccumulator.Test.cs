using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     <c>float16</c> matrix products accumulate in <b>float32</b>, not in Half.
    ///     <para>
    ///     NumPy's <c>matmul_inner_noblas</c> declares <c>float sum = 0</c>, accumulates every
    ///     product there and narrows once with <c>npy_float_to_half(sum)</c>; <c>HALF_dot</c> does
    ///     the same with <c>float tmp = 0.</c>. Accumulating in Half instead does not merely lose a
    ///     ulp, it <b>saturates</b>: half's spacing above 2048 is 2, so <c>2048 + 1 == 2048</c> and
    ///     <c>ones(4096)·ones(4096)</c> stalls at 2048 where NumPy returns 4096.
    ///     </para>
    ///     <para>
    ///     Half is the only float dtype NumPy does NOT hand to cblas (<c>USEBLAS = 1, 1, 0, 0, 1,
    ///     1, …</c> over FLOAT, DOUBLE, LONGDOUBLE, HALF, CFLOAT, CDOUBLE in <c>matmul.c.src</c>),
    ///     so unlike float32/float64 this loop is NumPy's actual reference implementation and can
    ///     be reproduced exactly — no BLAS package required, on any host.
    ///     </para>
    ///     Every expected value below was read off NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class np_matmul_half_accumulator_test
    {
        #region the saturation the Half accumulator caused

        [TestMethod]
        public void OnesDotOnes_IsExact_PastHalfsIntegerResolution()
        {
            // Half represents every integer up to 2048 exactly and then every SECOND one, so a Half
            // accumulator sticks at 2048 forever. A float32 accumulator counts all the way.
            AssertHalfBits(0x4800, Ones(8), "ones(8)·ones(8) = 8");
            AssertHalfBits(0x5400, Ones(64), "ones(64) = 64");
            AssertHalfBits(0x5c00, Ones(256), "ones(256) = 256");
            AssertHalfBits(0x6400, Ones(1024), "ones(1024) = 1024");
            AssertHalfBits(0x69dc, Ones(3000), "ones(3000) = 3000 — a Half accumulator gives 2048");
            AssertHalfBits(0x6c00, Ones(4096), "ones(4096) = 4096 — a Half accumulator gives 2048");
        }

        [TestMethod]
        public void ValuesBelowHalfsResolution_SurviveTheFloat32Accumulator()
        {
            // 2^-11 is half the gap between 1.0 and the next half, so adding one to a Half
            // accumulator holding 1.0 rounds straight back to 1.0 (ties-to-even). Nine of them in
            // float32 do not: NumPy returns 1.00390625 (0x3c04), not 1.0 (0x3c00).
            var v = HalfVector(1.0, Repeat(Math.Pow(2, -11), 9));
            var ones = HalfVector(Repeat(1.0, 10));

            AssertHalfBits(0x3c04, (Half)np.matmul(v.reshape(1, 10), ones.reshape(10, 1)).GetAtIndex(0), "matmul 2-D");
            AssertHalfBits(0x3c04, (Half)np.matmul(v, ones).GetAtIndex(0), "matmul 1-D");
            AssertHalfBits(0x3c04, (Half)np.dot(v.reshape(1, 10), ones.reshape(10, 1)).GetAtIndex(0), "dot 2-D");
            AssertHalfBits(0x3c04, (Half)np.dot(v, ones).GetAtIndex(0), "dot 1-D");
        }

        [TestMethod]
        public void EveryEntryPoint_UsesTheWiderAccumulator()
        {
            // matmul 2-D / 1-D / batched and dot 2-D / 1-D are four different kernels in NumSharp.
            const int k = 4096;
            var row = Ones1D(k).reshape(1, k);
            var col = Ones1D(k).reshape(k, 1);

            AssertHalfBits(0x6c00, (Half)np.matmul(row, col).GetAtIndex(0), "matmul 2-D");
            AssertHalfBits(0x6c00, (Half)np.matmul(Ones1D(k), Ones1D(k)).GetAtIndex(0), "matmul 1-D");
            AssertHalfBits(0x6c00, (Half)np.dot(row, col).GetAtIndex(0), "dot 2-D");
            AssertHalfBits(0x6c00, (Half)np.dot(Ones1D(k), Ones1D(k)).GetAtIndex(0), "dot 1-D");

            var batched = np.matmul(np.stack(new[] { row, row }), np.stack(new[] { col, col }));
            CollectionAssert.AreEqual(new long[] { 2, 1, 1 }, batched.shape);
            AssertHalfBits(0x6c00, (Half)batched.GetAtIndex(0), "matmul batched [0]");
            AssertHalfBits(0x6c00, (Half)batched.GetAtIndex(1), "matmul batched [1]");
        }

        #endregion

        #region layouts, mixed dtypes, and the specials

        [TestMethod]
        public void TheWiderAccumulator_HoldsForStridedAndTransposedOperands()
        {
            // The F-order operand goes down a different branch of the kernel (bStride1 != 1).
            var a = Ramp(6, 300);
            var b = Ramp2(300, 4);
            var expected = new ushort[] { 0xcb00, 0x4000, 0x4400, 0xc800 };

            AssertFirst4(expected, np.matmul(a, b), "C order");
            AssertFirst4(expected, np.matmul(a.T.copy().T, b), "round-tripped through a transpose");
            AssertFirst4(expected, np.matmul(np.asfortranarray(a), b), "F-order lhs");
        }

        [TestMethod]
        public void MixedOperandsThatStillPromoteToHalf_UseTheFloat32AccumulatorToo()
        {
            // half against bool / int8 / uint8 promotes to float16 and lands in the MIXED kernel,
            // which accumulated in double. NumPy casts both inputs to half then runs the half loop.
            var a = Ramp(6, 300);

            var i8 = np.zeros(new Shape(300, 4), NPTypeCode.SByte);
            for (long i = 0; i < i8.size; i++) i8.SetAtIndex((sbyte)((i % 5) + 1), i);
            var mixedInt = np.matmul(a, i8);
            Assert.AreEqual(NPTypeCode.Half, mixedInt.typecode);
            AssertFirst4(new ushort[] { 0xd340, 0xd2e0, 0xcd80, 0xcb00 }, mixedInt, "half @ int8");

            var bl = np.ones(new Shape(300, 4), NPTypeCode.Boolean);
            var mixedBool = np.matmul(a, bl);
            Assert.AreEqual(NPTypeCode.Half, mixedBool.typecode);
            AssertFirst4(new ushort[] { 0xca00, 0xca00, 0xca00, 0xca00 }, mixedBool, "half @ bool");
        }

        [TestMethod]
        public void NanInfAndAccumulatorOverflow_MatchNumpy()
        {
            var ones3 = HalfVector(Repeat(1.0, 3)).reshape(3, 1);

            // The NaN operand is built from NumPy's BIT PATTERN, not from double.NaN: .NET's NaN
            // literal is NEGATIVE (0xffc00000 as float, 0xfe00 as half) where NumPy's is positive
            // (0x7e00), so using the literal would compare two different inputs. The kernel
            // propagates whichever it is given.
            AssertHalfBits(0x7e00, (Half)np.matmul(HalfBits(0x7e00, 0x3c00, 0x3c00).reshape(1, 3), ones3).GetAtIndex(0), "nan propagates");
            AssertHalfBits(0xfe00, (Half)np.matmul(HalfBits(0xfe00, 0x3c00, 0x3c00).reshape(1, 3), ones3).GetAtIndex(0), "a negative NaN stays negative");
            AssertHalfBits(0x7c00, (Half)np.matmul(HalfVector(double.PositiveInfinity, 1.0, 1.0).reshape(1, 3), ones3).GetAtIndex(0), "+inf");
            AssertHalfBits(0xfc00, (Half)np.matmul(HalfVector(double.NegativeInfinity, 1.0, 1.0).reshape(1, 3), ones3).GetAtIndex(0), "-inf");

            // 200*200*3000 = 1.2e8 — fine in the float32 accumulator, overflows on the narrowing.
            var big = HalfVector(Repeat(200.0, 3000));
            AssertHalfBits(0x7c00, (Half)np.matmul(big, big).GetAtIndex(0), "accumulator result overflows half -> +inf");
        }

        [TestMethod]
        public void CancellingPartialSums_MatchNumpy()
        {
            // Partials climb to ~1750 while the answer is 125 — a Half accumulator loses the tail.
            var alt = np.zeros(new Shape(1000), NPTypeCode.Half);
            for (long i = 0; i < 1000; i++) alt.SetAtIndex((Half)(i % 2 == 0 ? 3.5 : -3.25), i);

            AssertHalfBits(0x57d0, (Half)np.matmul(alt, HalfVector(Repeat(1.0, 1000))).GetAtIndex(0), "= 125");
        }

        [TestMethod]
        public void ZeroK_StillProducesExactZeros()
        {
            // The degenerate shortcut must not be disturbed by the new kernel.
            var r = np.matmul(np.zeros(new Shape(2, 3, 0), NPTypeCode.Half),
                              np.zeros(new Shape(2, 0, 5), NPTypeCode.Half));

            CollectionAssert.AreEqual(new long[] { 2, 3, 5 }, r.shape);
            for (long i = 0; i < r.size; i++)
                Assert.AreEqual(0, BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(i)), $"element {i}");
        }

        [TestMethod]
        public void OtherDtypes_AreUnaffected()
        {
            // The change is Half-only: float32/float64 keep their own accumulators, and integers
            // must still wrap in-dtype.
            var f32 = np.ones(new Shape(4096), NPTypeCode.Single);
            Assert.AreEqual(4096f, (float)np.matmul(f32, f32).GetAtIndex(0));

            var f64 = np.ones(new Shape(4096), NPTypeCode.Double);
            Assert.AreEqual(4096d, (double)np.matmul(f64, f64).GetAtIndex(0));

            var i32 = np.ones(new Shape(4096), NPTypeCode.Int32);
            Assert.AreEqual(4096, (int)np.matmul(i32, i32).GetAtIndex(0));
        }

        #endregion

        #region helpers

        private static double[] Repeat(double v, int n)
        {
            var a = new double[n];
            for (int i = 0; i < n; i++) a[i] = v;
            return a;
        }

        /// <summary>Builds a 1-D Half array from scalars and/or double[] runs.</summary>
        private static NDArray HalfVector(params object[] parts)
        {
            var vals = new System.Collections.Generic.List<double>();
            foreach (var p in parts)
            {
                if (p is double d) vals.Add(d);
                else vals.AddRange((double[])p);
            }
            var a = np.zeros(new Shape(vals.Count), NPTypeCode.Half);
            for (long i = 0; i < vals.Count; i++) a.SetAtIndex((Half)vals[(int)i], i);
            return a;
        }

        /// <summary>A 1-D Half array built from raw half bit patterns.</summary>
        private static NDArray HalfBits(params ushort[] bits)
        {
            var a = np.zeros(new Shape(bits.Length), NPTypeCode.Half);
            for (long i = 0; i < bits.Length; i++) a.SetAtIndex(BitConverter.UInt16BitsToHalf(bits[i]), i);
            return a;
        }

        private static NDArray Ones1D(int n) => np.ones(new Shape(n), NPTypeCode.Half);

        private static Half Ones(int k)
            => (Half)np.matmul(Ones1D(k).reshape(1, k), Ones1D(k).reshape(k, 1)).GetAtIndex(0);

        /// <summary>(arange % 11) - 5 in half — exact, and spans both signs.</summary>
        private static NDArray Ramp(long rows, long cols)
        {
            var a = np.zeros(new Shape(rows, cols), NPTypeCode.Half);
            for (long i = 0; i < a.size; i++) a.SetAtIndex((Half)(double)((i % 11) - 5), i);
            return a;
        }

        /// <summary>(arange % 7) - 3 in half.</summary>
        private static NDArray Ramp2(long rows, long cols)
        {
            var a = np.zeros(new Shape(rows, cols), NPTypeCode.Half);
            for (long i = 0; i < a.size; i++) a.SetAtIndex((Half)(double)((i % 7) - 3), i);
            return a;
        }

        private static void AssertHalfBits(ushort expected, Half actual, string because)
            => Assert.AreEqual(expected, BitConverter.HalfToUInt16Bits(actual),
                $"{because}: expected half bits 0x{expected:x4}, got 0x{BitConverter.HalfToUInt16Bits(actual):x4} ({actual})");

        private static void AssertFirst4(ushort[] expected, NDArray r, string because)
        {
            for (int i = 0; i < expected.Length; i++)
                AssertHalfBits(expected[i], (Half)r.GetAtIndex(i), $"{because}, element {i}");
        }

        #endregion
    }
}
