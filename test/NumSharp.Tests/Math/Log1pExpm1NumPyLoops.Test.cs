using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     float32 / float64 <c>np.log1p</c>, <c>np.expm1</c> and the <c>log1p</c> inside <c>np.logaddexp</c> against
    ///     NumPy 2.4.2 (win-amd64), raw bits. NumPy calls the UCRT's <c>log1p</c> / <c>log1pf</c> / <c>expm1</c> /
    ///     <c>expm1f</c>; NumSharp's <c>NDLog1pExpm1Math</c> models them: <c>log1p</c> exactly (Goldberg's correction over
    ///     the UCRT's <c>log</c>), <c>expm1</c> exactly outside <c>[-ln 2, ln 1.5]</c> (<c>exp(x) - 1</c>) and within 2
    ///     (float64) / 3 (float32) ULP inside it (Sun's <c>s_expm1</c>; the UCRT's in-band polynomial is undocumented).
    /// </summary>
    /// <remarks>
    ///     Parity QA-22 / QA-32 (docs/bugs/parity-qa-2026-11-02.md, 2026-11-05): NumSharp computed <c>Exp(x) - 1</c> /
    ///     <c>Log(1 + x)</c>, which cancel near zero (<c>expm1(1e-300)</c> and <c>log1p(1e-300)</c> were 0, <c>-0.0</c> came back
    ///     <c>+0.0</c>), and logaddexp ran an fdlibm <c>log1p</c> one ULP off the UCRT's, which the final addition amplified
    ///     to 64-16,384 ULP where the result cancels toward zero. The digest tests compare a SHA-256 of 100,000 results with
    ///     NumPy's over inputs both sides build with exact arithmetic, so one changed bit fails.
    /// </remarks>
    [TestClass]
    public class Log1pExpm1NumPyLoopsTests
    {
        /// <summary>The raw bits of every element of a float64 result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One word per element.</returns>
        private static ulong[] Bits64(NDArray r) => np.ascontiguousarray(r).view(np.uint64).ToArray<ulong>();

        /// <summary>The raw bits of every element of a float32 result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>One word per element.</returns>
        private static uint[] Bits32(NDArray r) => np.ascontiguousarray(r).view(np.uint32).ToArray<uint>();

        /// <summary>The upper-case hex SHA-256 of a result's C-order bytes.</summary>
        /// <param name="r">The result.</param>
        /// <returns>The digest.</returns>
        private static string Sha(NDArray r) => Convert.ToHexString(SHA256.HashData(np.ascontiguousarray(r).view(np.uint8).ToArray<byte>()));

        /// <summary>A float64 array holding exactly the given bit patterns.</summary>
        /// <param name="bits">The raw words.</param>
        /// <returns>The array.</returns>
        private static NDArray F64(params ulong[] bits) => np.array(bits).view(np.float64);

        /// <summary>A float32 array holding exactly the given bit patterns.</summary>
        /// <param name="bits">The raw words.</param>
        /// <returns>The array.</returns>
        private static NDArray F32(params uint[] bits) => np.array(bits).view(np.float32);

        /// <summary>The ULP distance of two finite float64 values of one sign (their raw bits order like magnitudes).</summary>
        /// <param name="a">One value's bits.</param>
        /// <param name="b">The other value's bits.</param>
        /// <returns>The distance.</returns>
        private static long Ulp(ulong a, ulong b) => System.Math.Abs((long)a - (long)b);

        /// <summary>The 100,000-element float64 grid <c>(i - 50000) * 1.37e-5</c>, i.e. (-0.685, 0.685) in exact IEEE steps.</summary>
        /// <returns>The grid.</returns>
        private static NDArray Grid() => (np.arange(100000.0) - 50000.0) * 1.37e-5;

        /// <summary>
        ///     log1p near zero keeps every digit and the sign of zero; the domain edges are the UCRT's (-1 → -inf, below -1
        ///     and -inf → the POSITIVE quiet NaN, +inf → +inf); above u = 2 the correction still applies (5.5, 7.25: skipping
        ///     it, as a random-sampler shortcut does, differs from NumPy on ~2% of [1, 10]).
        /// </summary>
        [TestMethod]
        public void Log1p_Float64_IsTheUcrtsBits()
        {
            using var x = np.array(new[] { 1e-8, 1e-300, -0.0, 5e-324, 0.5, -0.5, 1.0, 2.0, 5.5, 7.25, 1e300, -0.999999, -1.0, -2.0,
                double.PositiveInfinity, double.NegativeInfinity });

            using var r = np.log1p(x);

            Bits64(r).Should().Equal(0x3E45798EE0636111UL, 0x01A56E1FC2F8F359UL, 0x8000000000000000UL, 0x0000000000000001UL,
                0x3FD9F323ECBF984CUL, 0xBFE62E42FEFA39EFUL, 0x3FE62E42FEFA39EFUL, 0x3FF193EA7AAD030BUL, 0x3FFDF2E6D6E5FBAAUL,
                0x4000E1B775589BD9UL, 0x4085963447F87FB5UL, 0xC02BA18A998FC064UL, 0xFFF0000000000000UL, 0x7FF8000000000000UL,
                0x7FF0000000000000UL, 0x7FF8000000000000UL);
        }

        /// <summary>float32 log1p is the UCRT's log1pf: the same statements computed in float32 (not a narrowed double).</summary>
        [TestMethod]
        public void Log1p_Float32_IsTheUcrtsBits()
        {
            using var x = np.array(new[] { 1e-8f, 6e-8f, 1e-30f, -0.0f, 0.5f, -0.5f, 1.0f, 5.5f, 7.25f, 1e30f, -0.999f, -1.0f, -2.0f,
                float.PositiveInfinity, float.NegativeInfinity });

            using var r = np.log1p(x);

            Bits32(r).Should().Equal(0x322BCC77u, 0x3380D959u, 0x0DA24260u, 0x80000000u, 0x3ECF991Fu, 0xBF317218u, 0x3F317218u,
                0x3FEF9737u, 0x40070DBCu, 0x428A27B5u, 0xC0DD0C70u, 0xFF800000u, 0x7FC00000u, 0x7F800000u, 0x7FC00000u);
        }

        /// <summary>
        ///     100,000 float64 and float32 log1p results over (-0.685, 0.685) and over 2^-50 … 2^50, digest-equal to NumPy's
        ///     (the model is exact, not merely close), through the contiguous kernel, a strided view and np.evaluate.
        /// </summary>
        [TestMethod]
        public void Log1p_Grids_DigestEqualNumPy()
        {
            using var g = Grid();
            var w = new double[100000];
            for (int k = 0; k < w.Length; k++)
                w[k] = System.Math.ScaleB(1.0 + (k % 997) / 997.0, k / 997 - 50);   // NumPy: ldexp(1 + (i % 997) / 997, i // 997 - 50)
            using var wide = np.array(w);
            using var twice = np.repeat(g, 2);

            Sha(np.log1p(g)).Should().Be("AA58D9D4D47E330DD75F7A5CC243C8EA3689B4ED5ACE80A1935CC680E70CF985");
            Sha(np.log1p(twice["::2"])).Should().Be("AA58D9D4D47E330DD75F7A5CC243C8EA3689B4ED5ACE80A1935CC680E70CF985");
            Sha(np.evaluate(NDExpr.Log1p(NDExpr.Arr(g)))).Should().Be("AA58D9D4D47E330DD75F7A5CC243C8EA3689B4ED5ACE80A1935CC680E70CF985");
            Sha(np.log1p(g.astype(np.float32))).Should().Be("B36B1ACF9EE651B1584CAA8F8D4675C8706D45ECCD7E85894418FFFE6A2F341C");
            Sha(np.log1p(wide)).Should().Be("1C7A6497B64CAFEDD153A08B163FB695362C43DD6BFA5E0CFFF2BC6906796BCC");
            Sha(np.log1p(wide.astype(np.float32))).Should().Be("9D995674C84F34963827E5810B00F2E7A6292AF054BDC13DBE866679384FC869");
        }

        /// <summary>
        ///     expm1 outside <c>[-ln 2, ln 1.5]</c> is the UCRT's <c>exp(x) - 1</c> bit for bit (98,297 grid results digest-equal
        ///     to NumPy's, plus the edges: overflow to inf, -inf and underflow to -1); tiny arguments and -0.0 come back
        ///     unchanged; inside the band the result is within 2 ULP of NumPy's.
        /// </summary>
        [TestMethod]
        public void Expm1_Float64_ExactOutsideTheBand_WithinTwoUlpInside()
        {
            using var g60 = Grid() * 60.0;
            using var outside = g60[np.abs(g60) > 0.7];
            using var x = np.array(new[] { 1e-300, -0.0, 5e-324, 1.0, -1.0, 10.0, -10.0, 700.0, -745.0, double.PositiveInfinity,
                double.NegativeInfinity });
            using var band = np.array(new[] { 1e-8, 0.1, -0.5 });

            outside.size.Should().Be(98297);
            Sha(np.expm1(outside)).Should().Be("B88D8012E1210FAAC2768F30525CF460F6570E7D77B91C4457A0005FEFB32CA4");
            Bits64(np.expm1(x)).Should().Equal(0x01A56E1FC2F8F359UL, 0x8000000000000000UL, 0x0000000000000001UL,
                0x3FFB7E151628AED2UL, 0xBFE43A54E4E98864UL, 0x40D5825DCF950560UL, 0xBFEFFFA0CA192A6EUL, 0x7F0D945DF4F8EC8EUL,
                0xBFF0000000000000UL, 0x7FF0000000000000UL, 0xBFF0000000000000UL);
            var got = Bits64(np.expm1(band));
            ulong[] numpy = { 0x3E45798EE3FDB763UL, 0x3FBAEC7B35A00D3AUL, 0xBFD92E9A0720D3EDUL };
            for (int k = 0; k < numpy.Length; k++)
                Ulp(got[k], numpy[k]).Should().BeLessThanOrEqualTo(2, $"in-band expm1 element {k}");
        }

        /// <summary>
        ///     float32 expm1 outside its band is the UCRT's float32 <c>expf(x) - 1f</c> (1.0f is <c>1.7182817</c>, not the
        ///     correctly rounded <c>1.7182818</c> a double route gives); tiny arguments and -0.0 come back unchanged; inside
        ///     the band within 3 ULP of NumPy's undocumented polynomial (6e-8f was <c>1.1920929e-07</c>, twice the value).
        /// </summary>
        [TestMethod]
        public void Expm1_Float32_ExactOutsideTheBand_WithinThreeUlpInside()
        {
            using var x = np.array(new[] { 1e-30f, -0.0f, 1.0f, -1.0f, 20.0f, -20.0f, 88.0f, float.PositiveInfinity, float.NegativeInfinity });
            using var band = np.array(new[] { 6e-8f, 0.1f, -0.5f });

            Bits32(np.expm1(x)).Should().Equal(0x0DA24260u, 0x80000000u, 0x3FDBF0A8u, 0xBF21D2A7u, 0x4DE75844u, 0xBF800000u,
                0x7EF882B7u, 0x7F800000u, 0xBF800000u);
            var got = Bits32(np.expm1(band));
            uint[] numpy = { 0x3380D959u, 0x3DD763DAu, 0xBEC974D0u };
            for (int k = 0; k < numpy.Length; k++)
                ((long)System.Math.Abs((long)got[k] - numpy[k])).Should().BeLessThanOrEqualTo(3, $"in-band expm1f element {k}");
        }

        /// <summary>
        ///     A NaN passes through log1p and expm1 untouched, as the UCRT returns it: a signalling NaN stays signalling, sign
        ///     and payload kept (the old compositions quieted it), on the array kernel and on a 0-d operand.
        /// </summary>
        [TestMethod]
        public void NaN_PassesThroughUntouched()
        {
            using var x64 = F64(0x7FF4000000000001UL, 0xFFF4000000000ABCUL, 0x7FF8000000000123UL, 0xFFF8000000000000UL);
            using var x32 = F32(0x7FA00001u, 0xFFA00ABCu, 0x7FC00123u, 0xFFC00000u);

            Bits64(np.log1p(x64)).Should().Equal(0x7FF4000000000001UL, 0xFFF4000000000ABCUL, 0x7FF8000000000123UL, 0xFFF8000000000000UL);
            Bits64(np.expm1(x64)).Should().Equal(0x7FF4000000000001UL, 0xFFF4000000000ABCUL, 0x7FF8000000000123UL, 0xFFF8000000000000UL);
            Bits32(np.log1p(x32)).Should().Equal(0x7FA00001u, 0xFFA00ABCu, 0x7FC00123u, 0xFFC00000u);
            Bits32(np.expm1(x32)).Should().Equal(0x7FA00001u, 0xFFA00ABCu, 0x7FC00123u, 0xFFC00000u);
            Bits64(np.log1p(x64["0"])).Should().Equal(0x7FF4000000000001UL);
            Bits32(np.expm1(x32["1"])).Should().Equal(0xFFA00ABCu);
        }

        /// <summary>
        ///     logaddexp is exact where its result cancels toward zero (the QA-32 pairs: 64 ULP at float32, 128 at float64 with
        ///     the fdlibm log1p), and over a 100,000-pair grid; logaddexp2's pinned pair is NumPy's too.
        /// </summary>
        [TestMethod]
        public void LogAddExp_CancellationAndGrid_AreNumPysBits()
        {
            var i = Enumerable.Range(0, 100000).Select(k => (double)k).ToArray();
            using var xa = np.array(i.Select(k => -(k + 1.0) * 5e-5).ToArray());
            using var ya = np.array(i.Select(k => (k - 50000.0) * 3.1e-4).ToArray());

            Bits32(np.logaddexp(F32(0xC044C118u), F32(0xBD4546A8u))).Should().Equal(0xBA518900u);
            Bits32(np.logaddexp(F32(0xC06CDAFFu, 0xC02E195Fu, 0xC03A227Cu), F32(0xBCCCE6A7u, 0xBD8B8471u, 0xBD65D43Bu)))
                .Should().Equal(0x31000000u, 0x32000000u, 0xB1800000u);
            Bits64(np.logaddexp(F64(0xC00D9B5FEBCE88F0UL, 0xC005C32BE27E10B8UL, 0xC007444F72C8F0EAUL, 0xBFCD8F29FE47CCAAUL),
                    F64(0xBF999CD4EBB5093BUL, 0xBFB1708E1554F1C2UL, 0xBFACBA875B1E48ABUL, 0xBFF943165E7E95F3UL)))
                .Should().Equal(0x3DCC9DD520000000UL, 0x3DF802BFB0000000UL, 0xBE198493F4000000UL, 0xBDEEA86E80000000UL);
            Bits64(np.logaddexp(np.array(new[] { -3.0742855 }), np.array(new[] { -0.048163086 }))).Should().Equal(0xBF4A312E0D8F95C0UL);
            Bits64(np.logaddexp2(np.array(new[] { -1.0835713039595367 }), np.array(new[] { -0.9211879047455193 })))
                .Should().Equal(0xBF19357A0EE92000UL);
            Sha(np.logaddexp(xa, ya)).Should().Be("BA878F85548E77B81F060D508767DCF790640A81D088D73AD67E57C98EFE825D");
            Sha(np.logaddexp(xa.astype(np.float32), ya.astype(np.float32))).Should().Be("AB2C0BA4285ECAAE5E41BCA3BC6A498B9F85396573333882494CC01B9425BD07");
        }
    }
}
