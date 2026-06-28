using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    ///     Cross-platform numerical-parity reproductions surfaced on macOS/ARM64 (GitHub
    ///     <c>macos-latest</c> = Apple Silicon) while x86_64 passed. Each asserts NumPy 2.4.2
    ///     behaviour and is architecture-agnostic — it must hold on x86 AND ARM.
    ///
    ///     Root causes (see the per-region comments):
    ///       B1  binary maximum/minimum signed-zero — SIMD fallback uses <c>Vector.Max</c> →
    ///           ARM <c>FMAX</c> (IEEE maxNum, +0&gt;-0, order-independent) instead of x86
    ///           <c>MAXPS</c> (2nd-operand-on-tie, = NumPy).                 [ARM-only]
    ///       B2  negate signed-zero — vector negate is <c>Zero - x</c>, and 0.0-(+0.0)=+0.0
    ///           on ARM (x86 JIT folds to a sign-bit xor).                   [ARM-only]
    ///       B3  max/min REDUCTION signed-zero — fold uses Math.Max (+0 on tie) instead of
    ///           NumPy's last-tied-operand-wins.                             [all platforms]
    ///       B4  prod/sum of narrow ints → int64 — exact-int widening kernel is AVX2-gated,
    ///           so non-AVX2 (ARM) falls back to a double accumulator that SATURATES
    ///           (e.g. (long)2^70 → Int64.MaxValue) instead of wrapping.     [ARM-only]
    ///
    ///     NumPy oracle (probed 2.4.2):
    ///       maximum/minimum return the SECOND operand on a ±0 tie;
    ///       negative flips the sign bit;
    ///       prod(int16 [2]*70) = 0 (2^70 mod 2^64), prod(int16 [2]*63) = -2^63 = long.MinValue.
    /// </summary>
    [TestClass]
    public class SignedZeroAndIntWideningParityTests
    {
        // -0.0 and +0.0 differ only in the sign bit; AreEqual on the raw bits catches it.
        private static void AssertBits(double expected, double actual, string msg)
            => Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected),
                               BitConverter.DoubleToInt64Bits(actual), msg);

        private static void AssertBits(float expected, float actual, string msg)
            => Assert.AreEqual(BitConverter.SingleToInt32Bits(expected),
                               BitConverter.SingleToInt32Bits(actual), msg);

        private static bool IsNegZero(double v) => BitConverter.DoubleToInt64Bits(v) == long.MinValue;

        // ---------------------------------------------------------------- B1: binary maximum/minimum ±0

        // Sizes chosen to exercise the scalar path (1,3), the SIMD body (16,32) and a
        // ragged tail past the V256/V512 width (33).
        private static readonly int[] Widths = { 1, 3, 16, 32, 33 };

        [TestMethod]
        public void Maximum_F64_SignedZero_ReturnsSecondOperandOnTie()
        {
            foreach (int n in Widths)
            {
                var pos = new double[n]; var neg = new double[n];
                for (int i = 0; i < n; i++) { pos[i] = 0.0; neg[i] = -0.0; }

                var r1 = np.maximum(np.array(pos), np.array(neg));  // tie -> b = -0
                var r2 = np.maximum(np.array(neg), np.array(pos));  // tie -> b = +0
                for (int i = 0; i < n; i++)
                {
                    AssertBits(-0.0, r1.GetDouble(i), $"maximum(+0,-0) n={n} @{i}");
                    AssertBits(0.0, r2.GetDouble(i), $"maximum(-0,+0) n={n} @{i}");
                }
            }
        }

        [TestMethod]
        public void Minimum_F64_SignedZero_ReturnsSecondOperandOnTie()
        {
            foreach (int n in Widths)
            {
                var pos = new double[n]; var neg = new double[n];
                for (int i = 0; i < n; i++) { pos[i] = 0.0; neg[i] = -0.0; }

                var r1 = np.minimum(np.array(pos), np.array(neg));  // tie -> b = -0
                var r2 = np.minimum(np.array(neg), np.array(pos));  // tie -> b = +0
                for (int i = 0; i < n; i++)
                {
                    AssertBits(-0.0, r1.GetDouble(i), $"minimum(+0,-0) n={n} @{i}");
                    AssertBits(0.0, r2.GetDouble(i), $"minimum(-0,+0) n={n} @{i}");
                }
            }
        }

        [TestMethod]
        public void Maximum_F32_SignedZero_ReturnsSecondOperandOnTie()
        {
            foreach (int n in Widths)
            {
                var pos = new float[n]; var neg = new float[n];
                for (int i = 0; i < n; i++) { pos[i] = 0.0f; neg[i] = -0.0f; }

                var r1 = np.maximum(np.array(pos), np.array(neg));
                var r2 = np.minimum(np.array(neg), np.array(pos));
                for (int i = 0; i < n; i++)
                {
                    AssertBits(-0.0f, r1.GetSingle(i), $"maximum_f(+0,-0) n={n} @{i}");
                    AssertBits(0.0f, r2.GetSingle(i), $"minimum_f(-0,+0) n={n} @{i}");
                }
            }
        }

        [TestMethod]
        public void Maximum_ScalarBroadcastLeft_SignedZero()
        {
            // Mirrors the fuzz failure layout `maximum/pp_scalar_left` (scalar broadcast).
            int n = 16;
            var neg = new double[n]; for (int i = 0; i < n; i++) neg[i] = -0.0;
            var scalarPos = np.array(new[] { 0.0 });        // size-1, broadcasts
            var r = np.maximum(scalarPos, np.array(neg));    // tie -> b = -0
            for (int i = 0; i < n; i++)
                AssertBits(-0.0, r.GetDouble(i), $"maximum(+0 scalar, -0[]) @{i}");
        }

        [TestMethod]
        public void Maximum_Minimum_NaN_Propagates_NoRegression()
        {
            // NumPy maximum/minimum propagate NaN from EITHER operand (unlike fmax/fmin).
            var a = np.array(new[] { double.NaN, 1.0, double.NaN, 5.0 });
            var b = np.array(new[] { 2.0, double.NaN, double.NaN, 3.0 });

            var mx = np.maximum(a, b);
            var mn = np.minimum(a, b);
            Assert.IsTrue(double.IsNaN(mx.GetDouble(0)) && double.IsNaN(mx.GetDouble(1)) && double.IsNaN(mx.GetDouble(2)));
            Assert.AreEqual(5.0, mx.GetDouble(3));
            Assert.IsTrue(double.IsNaN(mn.GetDouble(0)) && double.IsNaN(mn.GetDouble(1)) && double.IsNaN(mn.GetDouble(2)));
            Assert.AreEqual(3.0, mn.GetDouble(3));
        }

        // ---------------------------------------------------------------- B2: negate ±0

        [TestMethod]
        public void Negative_F64_SignedZero_FlipsSignBit()
        {
            foreach (int n in new[] { 1, 2, 4, 8, 17, 32 })
            {
                var src = new double[n];
                for (int i = 0; i < n; i++) src[i] = (i % 2 == 0) ? 0.0 : -0.0;
                var r = np.negative(np.array(src));
                for (int i = 0; i < n; i++)
                    AssertBits(i % 2 == 0 ? -0.0 : 0.0, r.GetDouble(i), $"negative f64 n={n} @{i}");
            }
        }

        [TestMethod]
        public void Negative_F32_SignedZero_FlipsSignBit()
        {
            foreach (int n in new[] { 1, 4, 8, 16, 33 })
            {
                var src = new float[n];
                for (int i = 0; i < n; i++) src[i] = (i % 2 == 0) ? 0.0f : -0.0f;
                var r = np.negative(np.array(src));
                for (int i = 0; i < n; i++)
                    AssertBits(i % 2 == 0 ? -0.0f : 0.0f, r.GetSingle(i), $"negative f32 n={n} @{i}");
            }
        }

        [TestMethod]
        public void Negative_Normal_NoRegression()
        {
            var r = np.negative(np.array(new[] { 1.0, -2.0, 3.0, -4.5 }));
            Assert.AreEqual(-1.0, r.GetDouble(0));
            Assert.AreEqual(2.0, r.GetDouble(1));
            Assert.AreEqual(-3.0, r.GetDouble(2));
            Assert.AreEqual(4.5, r.GetDouble(3));
        }

        // ---------------------------------------------------------------- B3: max/min reduction ±0 (all platforms)
        //
        // KNOWN BUG (all platforms, including x86). NumPy's max/min reduction is a sequential fold
        // whose ±0 tie resolves to the LAST tied operand: np.max([+0,-0])=-0, np.max([-0,+0])=+0.
        // NumSharp's reduction folds with Math.Max/Math.Min, which return +0 / -0 on a ±0 tie
        // regardless of order. Matching NumPy exactly requires preserving original-element order
        // through the (vectorized, horizontally-combined) reduction — perf-prohibitive for a pure
        // signed-zero edge — so this stays deferred and excluded from CI via [OpenBugs] rather than
        // documented as intended. Remove [OpenBugs] when the reduction tie-break is fixed.

        [TestMethod, OpenBugs]
        public void Max_Reduction_SignedZero_LastTiedOperandWins()
        {
            // NumPy: np.max([+0,-0]) = -0, np.max([-0,+0]) = +0 (last tied wins).
            AssertBits(-0.0, np.max(np.array(new[] { 0.0, -0.0 })).GetDouble(0), "max([+0,-0])");
            AssertBits(0.0, np.max(np.array(new[] { -0.0, 0.0 })).GetDouble(0), "max([-0,+0])");
        }

        [TestMethod, OpenBugs]
        public void Min_Reduction_SignedZero_LastTiedOperandWins()
        {
            // NumPy: np.min([+0,-0]) = -0, np.min([-0,+0]) = +0.
            AssertBits(-0.0, np.min(np.array(new[] { 0.0, -0.0 })).GetDouble(0), "min([+0,-0])");
            AssertBits(0.0, np.min(np.array(new[] { -0.0, 0.0 })).GetDouble(0), "min([-0,+0])");
        }

        // ---------------------------------------------------------------- B4: integer reduction exact int64 wrap

        [TestMethod]
        public void Prod_Int16_To_Int64_ExactWrap_AllAxesAndFlat()
        {
            // 70 twos: 2^70 mod 2^64 = 0 (NOT Int64.MaxValue from a double accumulator).
            Assert.AreEqual(0L, np.prod(np.full(new Shape(70, 3), 2, typeof(short)), axis: 0).GetInt64(0), "int16 ax0");
            Assert.AreEqual(0L, np.prod(np.full(new Shape(3, 70), 2, typeof(short)), axis: 1).GetInt64(0), "int16 ax1");
            Assert.AreEqual(0L, np.prod(np.full(new Shape(70), 2, typeof(short))).GetInt64(0), "int16 flat");

            // 63 twos: 2^63 as int64 = long.MinValue.
            Assert.AreEqual(long.MinValue, np.prod(np.full(new Shape(63, 3), 2, typeof(short)), axis: 0).GetInt64(0), "int16 2^63 ax0");
        }

        [TestMethod]
        public void Prod_Int8_Int32_UInt16_To_Wide_ExactWrap()
        {
            Assert.AreEqual(0L, np.prod(np.full(new Shape(70, 3), 2, typeof(sbyte)), axis: 0).GetInt64(0), "int8 ax0");
            Assert.AreEqual(0L, np.prod(np.full(new Shape(70), 2, typeof(int))).GetInt64(0), "int32 flat");
            Assert.AreEqual(0UL, np.prod(np.full(new Shape(70), 2, typeof(ushort))).GetUInt64(0), "uint16 flat");
        }

        [TestMethod]
        public void Sum_Int64_ExactWrap()
        {
            // 8 * 2^62 = 2^65 wraps modulo 2^64 to 0.
            var a = np.full(new Shape(8), 1L << 62, typeof(long));
            Assert.AreEqual(0L, np.sum(a).GetInt64(0), "sum int64 2^65 wrap");
        }

        [TestMethod]
        public void Sum_Int16_To_Int64_Exact_NoFloatPrecisionLoss()
        {
            // 100000 * 30000 = 3e9 > int32 range; exact in int64, would be fine in double too,
            // but pins the int64 accumulator dtype + value.
            var a = np.full(new Shape(100_000), 30000, typeof(short));
            Assert.AreEqual(3_000_000_000L, np.sum(a).GetInt64(0), "sum int16 3e9");
        }
    }
}
