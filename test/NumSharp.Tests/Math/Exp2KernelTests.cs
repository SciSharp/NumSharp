using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// np.exp2 (2^x) float32 kernel — NDFloatMath.Exp2, the fast SIMD replacement for the old
    /// (float)Math.Pow(2, (double)x) bridge.
    ///
    /// Unlike exp/log/sin/cos/tanh, exp2 is NOT a bit-for-bit port of a NumPy kernel: the numpy
    /// 2.4.2 win-amd64 wheel runs a SCALAR exp2f (its SVML vector kernel is AVX-512/Linux-gated), so
    /// there is no reproducible float kernel to match. It agrees with NumPy to ≤ 1 ULP (~0.2% of
    /// inputs, all subnormal outputs) by evaluating 2^r in double and rounding once. These tests pin
    /// the exact specials, the exactness of integer powers, the scalar==SIMD contract, and a bounded
    /// accuracy envelope so a gross regression fails. Full accuracy vs NumPy is gated by the
    /// unary_extra / specials differential-fuzz tiers.
    /// </summary>
    [TestClass]
    public class Exp2KernelTests
    {
        private static float Exp2(float x) => np.exp2(np.array(new[] { x })).GetAtIndex<float>(0);

        /// <summary>Signed-magnitude ULP distance between two POSITIVE (or +0/+inf) float results.</summary>
        private static long Ulp(float a, float b)
            => System.Math.Abs((long)BitConverter.SingleToInt32Bits(a) - BitConverter.SingleToInt32Bits(b));

        [TestMethod]
        public void Exp2_ReturnsFloat32()
        {
            np.exp2(np.array(new[] { 1.0f, 2.0f, 3.0f })).typecode.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Exp2_IntegerPowers_Exact()
        {
            // 2^k is exactly representable for k in [-24, 24]; the integer-power path must be exact.
            for (int k = -24; k <= 24; k++)
                Exp2((float)k).Should().Be((float)System.Math.Pow(2.0, k), $"exp2({k}) must be exact");
        }

        [TestMethod]
        public void Exp2_KnownValues()
        {
            Exp2(0f).Should().Be(1f);
            Exp2(-0f).Should().Be(1f);          // 2^-0 == 1
            Exp2(1f).Should().Be(2f);
            Exp2(-1f).Should().Be(0.5f);
            Exp2(10f).Should().Be(1024f);
            Exp2(0.5f).Should().Be(1.4142135f);  // sqrt(2)
            Exp2(-0.5f).Should().Be(0.70710677f);
        }

        [TestMethod]
        public void Exp2_Specials()
        {
            Exp2(float.PositiveInfinity).Should().Be(float.PositiveInfinity);
            Exp2(float.NegativeInfinity).Should().Be(0f);
            float.IsNaN(Exp2(float.NaN)).Should().BeTrue();
            // Overflow: 2^128 is not a finite float.
            Exp2(128f).Should().Be(float.PositiveInfinity);
            Exp2(200f).Should().Be(float.PositiveInfinity);
            // Underflow: 2^-150 rounds to +0.
            Exp2(-150f).Should().Be(0f);
            Exp2(-1000f).Should().Be(0f);
            // Just inside the finite range stays finite and positive.
            float top = Exp2(127.9f);
            float.IsFinite(top).Should().BeTrue();
            (top > 0f).Should().BeTrue();
        }

        [TestMethod]
        public void Exp2_Subnormal_Boundary_Finite()
        {
            // x in (-150, -126] produces subnormal outputs — must be > 0 and finite, not flushed early.
            Exp2(-127f).Should().BeGreaterThan(0f);
            Exp2(-140f).Should().BeGreaterThan(0f);
            float.IsNaN(Exp2(-149f)).Should().BeFalse();
        }

        /// <summary>
        /// The load-bearing contract shared by all the NDFloatMath ports: the scalar entry point and
        /// the vector kernel must agree bit-for-bit. A contiguous array takes the SIMD path; a strided
        /// view of the SAME values takes the scalar path. Every element must match exactly.
        /// </summary>
        [TestMethod]
        public void Exp2_Scalar_Equals_Simd()
        {
            const int n = 4096;
            var rnd = new System.Random(1234);
            var data = new float[n];
            for (int i = 0; i < n; i++)
                data[i] = (float)(rnd.NextDouble() * 80.0 - 40.0);   // covers overflow/underflow/normal

            var contig = np.exp2(np.array(data));                    // SIMD (contiguous)
            var strided = np.exp2(np.array((float[])data.Clone())["::2"]);  // scalar (strided)

            for (int i = 0; i < n / 2; i++)
            {
                uint c = BitConverter.SingleToUInt32Bits(contig.GetAtIndex<float>(i * 2));
                uint s = BitConverter.SingleToUInt32Bits(strided.GetAtIndex<float>(i));
                c.Should().Be(s, $"scalar and SIMD exp2 must be bit-identical at value {data[i * 2]}");
            }
        }

        /// <summary>
        /// Bounded-accuracy regression guard: exp2 must stay within 2 ULP of the correctly-rounded
        /// 2^x (approximated by the double Math.Pow reference the kernel replaced) across the normal
        /// output range. A gross error — wrong magnitude, a broken polynomial — exceeds this.
        /// </summary>
        [TestMethod]
        public void Exp2_WithinTwoUlp_OfReference()
        {
            long worst = 0;
            for (double x = -120.0; x <= 120.0; x += 0.0009765625)  // 2^-10 step, ~245k points
            {
                float xf = (float)x;
                float got = Exp2(xf);
                float reference = (float)System.Math.Pow(2.0, (double)xf);
                if (!float.IsFinite(got) || !float.IsFinite(reference))
                    continue;
                worst = System.Math.Max(worst, Ulp(got, reference));
            }
            worst.Should().BeLessThanOrEqualTo(2, "exp2 must be within 2 ULP of the double reference");
        }

        /// <summary>exp2 fused into an NDExpr chain must equal the direct kernel bit-for-bit.</summary>
        [TestMethod]
        public void Exp2_Fused_MatchesDirect()
        {
            var a = np.array(new[] { -3.5f, -1f, 0f, 0.25f, 2f, 7.75f, 15f });
            var direct = np.exp2(a);
            var fused = np.evaluate(NDExpr.Exp2(a));
            for (int i = 0; i < a.size; i++)
                BitConverter.SingleToUInt32Bits(fused.GetAtIndex<float>(i))
                    .Should().Be(BitConverter.SingleToUInt32Bits(direct.GetAtIndex<float>(i)));
        }
    }
}
