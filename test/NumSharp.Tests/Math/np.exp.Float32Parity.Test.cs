using System;
using System.Linq;
using NumSharp.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// <c>np.exp</c> at float32 is a BIT-EXACT port of the kernel NumPy 2.4.2 actually runs
    /// (<c>simd_exp_FLOAT</c> from <c>loops_exponent_log.dispatch.c.src</c>, the AVX2+FMA3
    /// instantiation), living in <see cref="NDFloatMath.Exp(float)"/> plus its
    /// <c>Vector{128,256,512}</c> overloads.
    ///
    /// <para>Every expectation below is a raw <b>bit pattern probed from NumPy 2.4.2</b>, not a
    /// tolerance — which is the point: the previous implementation (<c>MathF.Exp</c>) was ~correctly
    /// rounded and therefore differed from NumPy on roughly a third of all float32 inputs by 1-2 ULP.
    /// Comparing with <c>BeApproximately</c> would pass on both, so these tests compare
    /// <see cref="BitConverter.SingleToUInt32Bits"/>.</para>
    ///
    /// <para>The exhaustive claim (all 2^32 float32 inputs agree with NumPy) is verified by a chunked
    /// checksum sweep during development and pinned in CI by the <c>exp_f32.jsonl</c> fuzz tier; these
    /// tests pin the specials, the boundaries, the dtype tiers and the layout coverage in a form that
    /// names what each case is for.</para>
    /// </summary>
    [TestClass]
    public class NpExpFloat32ParityTest : TestClass
    {
        private static float F(uint bits) => BitConverter.UInt32BitsToSingle(bits);
        private static uint B(float v) => BitConverter.SingleToUInt32Bits(v);

        /// <summary>(input bits, NumPy 2.4.2's exp output bits) — probed, never computed.</summary>
        private static readonly (uint In, uint Out, string What)[] Probed =
        {
            // NaN in ANY spelling collapses to the CANONICAL quiet NaN: NumPy blanks the NaN lanes
            // to zero before computing and writes NPY_NANF back, so sign and payload are discarded.
            // (Note .NET's own float.NaN is 0xffc00000 — the sign bit is set — so returning
            // float.NaN here would be wrong.)
            (0x7fc00000u, 0x7fc00000u, "quiet NaN"),
            (0x7fc00001u, 0x7fc00000u, "NaN with payload"),
            (0x7f800001u, 0x7fc00000u, "signalling NaN"),
            (0xffc00000u, 0x7fc00000u, "negative NaN"),

            (0x7f800000u, 0x7f800000u, "+inf -> +inf"),
            (0xff800000u, 0x00000000u, "-inf -> +0"),
            (0x00000000u, 0x3f800000u, "+0 -> 1"),
            (0x80000000u, 0x3f800000u, "-0 -> 1"),
            (0x00000001u, 0x3f800000u, "smallest subnormal -> 1"),

            // Saturation boundaries, probed to the ULP. xmax is INCLUSIVE (>= overflows).
            (0x42b17218u, 0x7f800000u, "xmax = 88.72283935546875 -> +inf"),
            (0x42b17217u, 0x7f7fff84u, "one ULP below xmax -> largest finite"),
            (0xc2cff1b5u, 0x00000000u, "xmin = -103.97208404541015625 -> +0"),
            (0xc2cff1b3u, 0x00000001u, "just above xmin -> smallest subnormal"),

            // The subnormal-OUTPUT band — the only route through NumPy's fma_scalef_ps denormal
            // split (exponent clamp at -125 plus a division that generates the subnormal).
            (0xc2aea8f6u, 0x0080d71au, "-87.33 -> 1.1832107e-38"),
            (0xc2b00000u, 0x0041edc4u, "-88 -> subnormal 6.054601e-39"),
            (0xc2ce0000u, 0x00000001u, "-103 -> smallest subnormal"),

            // NumPy documents its own worst case here: 2.52 ULP from the true value. A correctly
            // rounded exp returns 0x12b65239 — so this single case distinguishes "we reproduce
            // NumPy" from "we are more accurate than NumPy".
            (0xc2781e37u, 0x12b65236u, "NumPy's documented max-ULP input"),

            // x * log2(e) is EXACTLY -85.5 here, so the magic-constant rint sits on a tie: a fused
            // multiply-add (what the wheel's compiler emitted) rounds the quadrant to -85, a
            // separate multiply-then-add rounds it to -86, and the two answers differ by 1 ULP.
            (0xc26d0e6cu, 0x14b504f3u, "FMA-contraction tie at x*log2e == -85.5"),

            (0x3f800000u, 0x402df855u, "exp(1) = e"),
            (0xbf800000u, 0x3ebc5ab1u, "exp(-1)"),
            (0x41200000u, 0x46ac14efu, "exp(10)"),
            (0x3f000000u, 0x3fd3094cu, "exp(0.5)"),
        };

        [TestMethod]
        public void Exp_Float32_MatchesNumPyBitForBit()
        {
            var input = np.array(Probed.Select(p => F(p.In)).ToArray());
            var result = np.exp(input);

            result.dtype.Should().Be(typeof(float));
            for (int i = 0; i < Probed.Length; i++)
                B(result.GetSingle(i)).Should().Be(Probed[i].Out,
                    $"np.exp(0x{Probed[i].In:x8}) is {Probed[i].What}");
        }

        [TestMethod]
        public void Exp_Float32_ScalarHelper_MatchesTheVectorKernel()
        {
            // The scalar entry point and the width overload must agree element for element — the
            // kernel picks between them by host capability, so a divergence would make results
            // depend on the machine.
            foreach (var (bits, expected, what) in Probed)
                B(NDFloatMath.Exp(F(bits))).Should().Be(expected, what);
        }

        [TestMethod]
        public void Exp_Float32_VectorOverloads_AgreeWithScalar_AtEveryWidth()
        {
            // Vector512/128 may not be the host's active width (and Avx512F may be absent), but the
            // overloads stay callable — the FMA falls back to MathF.FusedMultiplyAdd, which is the
            // same operation — so all three are exercised here regardless of the machine.
            var xs = Probed.Select(p => F(p.In)).Concat(
                Enumerable.Range(0, 64).Select(i => (float)(i * 3.1 - 100.0))).ToArray();

            for (int i = 0; i + 4 <= xs.Length; i += 4)
            {
                var v = System.Runtime.Intrinsics.Vector128.Create(xs.AsSpan(i, 4));
                var r = NDFloatMath.Exp(v);
                for (int k = 0; k < 4; k++)
                    B(r[k]).Should().Be(B(NDFloatMath.Exp(xs[i + k])), $"Vector128 lane {k} at index {i + k}");
            }
            for (int i = 0; i + 8 <= xs.Length; i += 8)
            {
                var v = System.Runtime.Intrinsics.Vector256.Create(xs.AsSpan(i, 8));
                var r = NDFloatMath.Exp(v);
                for (int k = 0; k < 8; k++)
                    B(r[k]).Should().Be(B(NDFloatMath.Exp(xs[i + k])), $"Vector256 lane {k} at index {i + k}");
            }
            for (int i = 0; i + 16 <= xs.Length; i += 16)
            {
                var v = System.Runtime.Intrinsics.Vector512.Create(xs.AsSpan(i, 16));
                var r = NDFloatMath.Exp(v);
                for (int k = 0; k < 16; k++)
                    B(r[k]).Should().Be(B(NDFloatMath.Exp(xs[i + k])), $"Vector512 lane {k} at index {i + k}");
            }
        }

        [TestMethod]
        public void Exp_Float32_SameBitsThroughEveryLayout()
        {
            // NumPy's own loop is layout-independent (its scalar fallback calls the same SIMD
            // routine element by element precisely so strided and contiguous cannot disagree), and
            // NumSharp routes contiguous / strided-gather / NDIter paths through the same kernel.
            var xs = Probed.Select(p => F(p.In))
                .Concat(Enumerable.Range(0, 96).Select(i => (float)(i * 2.0 - 100.0))).ToArray();
            var flat = np.array(xs);
            var expected = np.exp(flat);

            void Same(string what, NDArray view, Func<int, int> sourceIndex, int count)
            {
                var got = np.exp(view).ravel();
                for (int i = 0; i < count; i++)
                    B(got.GetSingle(i)).Should().Be(B(expected.GetSingle(sourceIndex(i))),
                        $"{what} element {i}");
            }

            Same("strided ::2", flat["::2"], i => i * 2, xs.Length / 2);
            Same("reversed ::-1", flat["::-1"], i => xs.Length - 1 - i, xs.Length);
            Same("offset slice", flat["5:"], i => i + 5, xs.Length - 5);

            int rows = 4, cols = xs.Length / 4;
            var m = np.array(xs.Take(rows * cols).ToArray()).reshape(rows, cols);
            Same("2-D contiguous", m, i => i, rows * cols);
            Same("transposed (F view)", m.T, i => (i % rows) * cols + (i / rows), rows * cols);

            // A broadcast view is read-only; exp must still read it correctly.
            var one = np.array(xs.Take(16).ToArray()).reshape(1, 16);
            Same("broadcast", np.broadcast_to(one, new Shape(3, 16)), i => i % 16, 48);
        }

        [TestMethod]
        public void Exp_ZeroDimensional_KeepsShapeAndBits()
        {
            var s = np.exp(NDArray.Scalar(F(0xc2781e37u)));
            s.ndim.Should().Be(0);
            s.dtype.Should().Be(typeof(float));
            B(s.GetSingle(0)).Should().Be(0x12b65236u);
        }

        [TestMethod]
        public void Exp_Empty_ReturnsEmptyFloat32()
        {
            var e = np.exp(np.array(new float[0]));
            e.size.Should().Be(0);
            e.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Exp_NarrowIntegerInputs_UseTheSameFloat32Loop()
        {
            // NumPy's dtype tier: int16/uint16 promote to the 'f->f' loop — the very kernel ported
            // here — while int32 and wider go to the float64 loop. Bits probed from NumPy 2.4.2.
            var i16 = np.array(new short[] { 0, 1, 2, 3, 5, 11, 87, 88, 89, -1, -5, -87, -88, -103, -104, 32767, -32768 });
            var r = np.exp(i16);
            r.dtype.Should().Be(typeof(float));

            uint[] expected =
            {
                0x3f800000u, 0x402df855u, 0x40ec7325u, 0x41a0af2eu, 0x431469c5u, 0x4769e224u,
                0x7e36d808u, 0x7ef882b7u, 0x7f800000u, 0x3ebc5ab1u, 0x3bdcc9feu, 0x00b33687u,
                0x0041edc4u, 0x00000001u, 0x00000000u, 0x7f800000u, 0x00000000u,
            };
            for (int i = 0; i < expected.Length; i++)
                B(r.GetSingle(i)).Should().Be(expected[i], $"exp(int16 element {i})");
        }

        [TestMethod]
        public void Exp_Float64_IsUnaffectedByTheFloat32Port()
        {
            // The port is scoped to the float32 loop; float64 still runs Math.Exp (NumPy's own
            // float64 path on this platform is the scalar CRT exp, and is a separate question).
            var d = np.exp(np.array(new double[] { 1.0, 0.0, -1.0 }));
            d.dtype.Should().Be(typeof(double));
            d.GetDouble(0).Should().Be(System.Math.Exp(1.0));
            d.GetDouble(1).Should().Be(1.0);
        }

        [TestMethod]
        public void Exp_Float32_OutAndWhere_StillHonourTheUfuncContract()
        {
            var src = np.array(new float[] { 1f, 2f, 3f, 4f });
            var dst = np.zeros(new Shape(4), NPTypeCode.Single);
            var mask = np.array(new[] { true, false, true, false });

            var returned = np.exp(src, dst, mask);
            ReferenceEquals(returned, dst).Should().BeTrue("out= returns the same instance");

            // Probed from NumPy 2.4.2: masked-off slots keep out=' prior contents (here, zeros).
            B(dst.GetSingle(0)).Should().Be(0x402df855u, "exp(1) written through out=");
            B(dst.GetSingle(1)).Should().Be(0x00000000u, "masked off — prior content kept");
            B(dst.GetSingle(2)).Should().Be(0x41a0af2eu, "exp(3) written through out=");
            B(dst.GetSingle(3)).Should().Be(0x00000000u, "masked off — prior content kept");
        }
    }
}
