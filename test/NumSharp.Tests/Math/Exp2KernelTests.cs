using System;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

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
        /// the vector kernel must agree bit-for-bit. A contiguous 4,096-element array runs the vector
        /// kernel for every element (a whole number of vectors at any width, so no scalar tail), and each
        /// element must equal <see cref="NDFloatMath.Exp2(float)"/> called directly — the scalar entry
        /// point. A strided view of the same values must reproduce the contiguous bits as well.
        /// </summary>
        /// <remarks>
        /// The strided view used to be this test's only probe, on the premise that it "takes the scalar
        /// path". It does not: the view is gathered into the same vector kernel, so the test compared the
        /// vector kernel with itself and stayed green under an accuracy slip planted in the scalar entry
        /// point alone AND under one planted in the Vector256 overload alone (mutation-checked). Comparing
        /// against the scalar entry point directly fails on either; the strided comparison is kept for what
        /// it does prove — the result does not depend on the input's layout.
        /// </remarks>
        [TestMethod]
        public void Exp2_Scalar_Equals_Simd()
        {
            const int n = 4096;
            var rnd = new System.Random(1234);
            var data = new float[n];
            for (int i = 0; i < n; i++)
                data[i] = (float)(rnd.NextDouble() * 80.0 - 40.0);   // covers overflow/underflow/normal

            var contig = np.exp2(np.array(data));                    // vector kernel, every element
            var strided = np.exp2(np.array((float[])data.Clone())["::2"]);  // strided view: gathered into the vector kernel too

            for (int i = 0; i < n; i++)
            {
                uint v = BitConverter.SingleToUInt32Bits(contig.GetAtIndex<float>(i));
                uint s = BitConverter.SingleToUInt32Bits(NDFloatMath.Exp2(data[i]));
                v.Should().Be(s, $"the scalar entry point and the vector kernel must be bit-identical at value {data[i]}");
            }

            for (int i = 0; i < n / 2; i++)
            {
                uint c = BitConverter.SingleToUInt32Bits(contig.GetAtIndex<float>(i * 2));
                uint s = BitConverter.SingleToUInt32Bits(strided.GetAtIndex<float>(i));
                c.Should().Be(s, $"a strided view must reproduce the contiguous result's bits at value {data[i * 2]}");
            }
        }

        /// <summary>
        /// Bounded-accuracy regression guard: exp2 must stay within 2 ULP of the correctly-rounded
        /// 2^x (approximated by the double Math.Pow reference the kernel replaced) across the normal
        /// output range, on BOTH kernel paths. A gross error — wrong magnitude, a broken polynomial —
        /// exceeds this.
        /// </summary>
        /// <remarks>
        /// The 2^-10 grid over [-120, 120] (245,761 points, each exact in float) is checked on both
        /// paths without one NDArray per point. The SCALAR entry point <see cref="NDFloatMath.Exp2(float)"/>
        /// is called directly for every point — it is the function each former single-element
        /// <c>np.exp2</c> call reached (a one-element array never fills a vector), so that coverage is
        /// unchanged; the dispatch around it stays covered by the per-element tests above. The whole grid
        /// then goes through <c>np.exp2</c> as ONE contiguous array — the vector kernel, which the
        /// per-point form never reached. A strided view is not a scalar-path probe: it runs the vector
        /// kernel too (mutation-checked — an error planted in the Vector256 overload alone shows up on
        /// a <c>::2</c> view). The per-point form cost ~0.36 s in the suite: two NDArrays per point, all
        /// ~491k of them left undisposed for the finalizer.
        /// </remarks>
        [TestMethod]
        public void Exp2_WithinTwoUlp_OfReference()
        {
            const int Steps = 240 * 1024;           // 2^-10 step across [-120, 120]
            var grid = new float[Steps + 1];
            for (int k = 0; k <= Steps; k++)
            {
                // -120 + k * 2^-10 is exact in double and in float (17 significant bits at most), so this is
                // exactly the point sequence the former accumulating `x += 2^-10` loop produced.
                grid[k] = (float)(-120.0 + k * 0.0009765625);
            }

            using var input = np.array(grid);
            using var kernelResult = np.exp2(input);  // contiguous: the vector kernel, plus its scalar tail
            float[] vector = kernelResult.ToArray<float>();

            long worstScalar = 0, worstVector = 0;
            for (int k = 0; k <= Steps; k++)
            {
                float reference = (float)System.Math.Pow(2.0, (double)grid[k]);
                if (!float.IsFinite(reference))
                    continue;
                // A non-finite result against a finite reference is skipped exactly as the per-point form did:
                // Exp2_Specials pins where overflow and underflow must land.
                float scalar = NDFloatMath.Exp2(grid[k]);
                if (float.IsFinite(scalar))
                    worstScalar = System.Math.Max(worstScalar, Ulp(scalar, reference));
                if (float.IsFinite(vector[k]))
                    worstVector = System.Math.Max(worstVector, Ulp(vector[k], reference));
            }

            worstScalar.Should().BeLessThanOrEqualTo(2, "exp2's scalar entry point must be within 2 ULP of the double reference");
            worstVector.Should().BeLessThanOrEqualTo(2, "exp2's vector kernel must be within 2 ULP of the double reference");
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
