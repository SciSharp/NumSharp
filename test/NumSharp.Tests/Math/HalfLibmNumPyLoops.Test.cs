using System;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     float16 <c>arctan2</c> / <c>log1p</c> / <c>expm1</c> and float32 <c>arctan2</c> return NumPy 2.4.2's bits
    ///     (win-amd64), raw bits and NaN payloads included. NumPy runs the float16 ufuncs through their float32 loops: the
    ///     half widened bit for bit, the UCRT's <c>atan2f</c> / <c>log1pf</c> / <c>expm1f</c>, then
    ///     <c>npy_float_to_half</c>; its float32 <c>arctan2</c> is <c>atan2f</c>, which <see cref="MathF.Atan2"/> calls.
    /// </summary>
    /// <remarks>
    ///     Parity QA-24 (docs/bugs/parity-qa-2026-11-02.md), the owner's "Match NumPy": NumSharp rounded a float64 result
    ///     once (13 float16 arctan2, 2 log1p and 2 expm1 results one ULP off; 116 of 2M float32 arctan2 pairs), quieted
    ///     every signalling NaN NumPy keeps and gave log1p's domain error a negative NaN. The exhaustive tests compare a
    ///     SHA-256 of every output with NumPy's (computed by the probe over the same inputs), so a single changed bit over
    ///     the 65,536 float16 inputs or the 1,048,576 float32 pairs fails. The float32 bits and the float16 arctan2 digest
    ///     belong to the UCRT and are asserted on win-amd64 only (<see cref="HostLibm"/>); the float16 values and the log1p /
    ///     expm1 digests held on glibc and Apple's libm as well (CI, 2026-11-07) and run everywhere.
    /// </remarks>
    [TestClass]
    public class HalfLibmNumPyLoopsTests
    {
        /// <summary>A 1-element float16 array holding exactly <paramref name="bits"/>.</summary>
        /// <param name="bits">The raw float16 bits.</param>
        /// <returns>The array.</returns>
        private static NDArray H(ushort bits) => np.array(new[] { BitConverter.UInt16BitsToHalf(bits) });

        /// <summary>The raw bits of element 0 of a float16 result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its bits.</returns>
        private static ushort Bits(NDArray r) => BitConverter.HalfToUInt16Bits(r.GetAtIndex<Half>(0));

        /// <summary>Every float16 bit pattern, 0 through 65535, as a float16 array.</summary>
        /// <returns>The array.</returns>
        private static NDArray AllHalves()
        {
            var u = new ushort[65536];
            for (int i = 0; i < u.Length; i++)
                u[i] = (ushort)i;
            return np.array(u).view(np.float16);
        }

        /// <summary>The upper-case hex SHA-256 of an array's elements in C order.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="r">The result.</param>
        /// <returns>The digest.</returns>
        private static string Sha256<T>(NDArray r) where T : unmanaged
            => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(r.ToArray<T>().AsSpan())));

        /// <summary>
        ///     The finding's values: <c>np.arctan2(float16 0x4E8B, 0x87AC)</c> is <c>0x3E48</c> (atan2f's float32 result is a
        ///     float16 midpoint), <c>np.log1p(float16 0x1E03)</c> is <c>0x1DFE</c>, <c>np.expm1</c> of <c>0xB94E</c> /
        ///     <c>0x2864</c> are <c>0xB7C2</c> / <c>0x2878</c> (UCRT expm1f's float32 results sit on float16 midpoints), and
        ///     float32 <c>np.arctan2(0xF5B024C4, 0x7C96E3DC)</c> is atan2f's <c>0xB8956C2E</c>.
        /// </summary>
        [TestMethod]
        public void FindingValues_AreNumPys()
        {
            using var y = H(0x4E8B);
            using var x = H(0x87AC);
            using var arctan2 = np.arctan2(y, x);
            Bits(arctan2).Should().Be(0x3E48);

            using var p = H(0x1E03);
            using var log1p = np.log1p(p);
            Bits(log1p).Should().Be(0x1DFE);

            using var q = H(0xB94E);
            using var expm1 = np.expm1(q);
            Bits(expm1).Should().Be(0xB7C2);
            using var q2 = H(0x2864);
            using var expm1b = np.expm1(q2);
            Bits(expm1b).Should().Be(0x2878);

            // float32 arctan2 IS atan2f's result, unnarrowed, so its last bit belongs to the C library: glibc and Apple's libm
            // return 0xB8956C2D (CI 2026-11-07), as NumPy built on them does. The float16 values above passed on both.
            if (HostLibm.IsPinnedHost)
            {
                using var y32 = np.array(new[] { BitConverter.UInt32BitsToSingle(0xF5B024C4) });
                using var x32 = np.array(new[] { BitConverter.UInt32BitsToSingle(0x7C96E3DC) });
                using var arctan32 = np.arctan2(y32, x32);
                BitConverter.SingleToUInt32Bits(arctan32.GetAtIndex<float>(0)).Should().Be(0xB8956C2E);
            }
        }

        /// <summary>
        ///     The NaN and edge rules of UCRT's float functions: a signalling NaN passes through <c>log1pf</c> /
        ///     <c>expm1f</c> unquieted (either sign), <c>log1p</c> of <c>x &lt; -1</c> or <c>-inf</c> is the POSITIVE quiet
        ///     NaN <c>0x7E00</c>, <c>log1p(-1)</c> is <c>-inf</c>, a negative zero keeps its sign, <c>expm1(-inf)</c> is
        ///     <c>-1</c>.
        /// </summary>
        [TestMethod]
        public void NaNAndEdges_AreUcrtsResults()
        {
            ushort Log1p(ushort b) { using var a = H(b); using var r = np.log1p(a); return Bits(r); }
            ushort Expm1(ushort b) { using var a = H(b); using var r = np.expm1(a); return Bits(r); }

            Log1p(0x7C01).Should().Be(0x7C01, "a positive signalling NaN stays signalling");
            Log1p(0xFC01).Should().Be(0xFC01, "a negative signalling NaN stays signalling");
            Log1p(0x7E05).Should().Be(0x7E05, "a quiet NaN comes back as it is");
            Log1p(0xBC01).Should().Be(0x7E00, "log1pf of x < -1 is the positive quiet NaN");
            Log1p(0xFC00).Should().Be(0x7E00, "log1pf(-inf) too");
            Log1p(0xBC00).Should().Be(0xFC00, "log1p(-1) is -inf");
            Log1p(0x8000).Should().Be(0x8000, "log1p(-0) is -0");
            Expm1(0x7C01).Should().Be(0x7C01, "expm1f keeps a signalling NaN");
            Expm1(0xFDFF).Should().Be(0xFDFF, "with its payload and sign");
            Expm1(0x8000).Should().Be(0x8000, "expm1(-0) is -0");
            Expm1(0xFC00).Should().Be(0xBC00, "expm1(-inf) is -1");
            Expm1(0x7C00).Should().Be(0x7C00, "expm1(+inf) is +inf");
        }

        /// <summary>
        ///     The 0-d x 0-d arctan2 shortcut and np.evaluate's nodes run the same loops: float16 through atan2f and NumPy's
        ///     narrowing, float32 through atan2f.
        /// </summary>
        [TestMethod]
        public void ScalarAndEvaluateRoutes_AreNumPys()
        {
            using var y0 = NDArray.Scalar(BitConverter.UInt16BitsToHalf(0x4E8B));
            using var x0 = NDArray.Scalar(BitConverter.UInt16BitsToHalf(0x87AC));
            using var a0 = np.arctan2(y0, x0);
            BitConverter.HalfToUInt16Bits(a0.GetAtIndex<Half>(0)).Should().Be(0x3E48, "0-d float16");

            // The float32 0-d result is atan2f's own bits: the win-amd64 UCRT's (see FindingValues_AreNumPys).
            if (HostLibm.IsPinnedHost)
            {
                using var y1 = NDArray.Scalar(BitConverter.UInt32BitsToSingle(0xF5B024C4));
                using var x1 = NDArray.Scalar(BitConverter.UInt32BitsToSingle(0x7C96E3DC));
                using var a1 = np.arctan2(y1, x1);
                BitConverter.SingleToUInt32Bits(a1.GetAtIndex<float>(0)).Should().Be(0xB8956C2E, "0-d float32");
            }

            using var y = H(0x4E8B);
            using var x = H(0x87AC);
            using var e1 = np.evaluate(NDExpr.ATan2((NDExpr)y, (NDExpr)x));
            Bits(e1).Should().Be(0x3E48, "np.evaluate's ATan2 node");
            using var p = H(0x1E03);
            using var e2 = np.evaluate(NDExpr.Log1p((NDExpr)p));
            Bits(e2).Should().Be(0x1DFE, "np.evaluate's Log1p node");
            using var q = H(0xB94E);
            using var e3 = np.evaluate(NDExpr.Expm1((NDExpr)q));
            Bits(e3).Should().Be(0xB7C2, "np.evaluate's Expm1 node");
        }

        /// <summary>
        ///     Exhaustive: <c>log1p</c> and <c>expm1</c> of every float16 pattern, and <c>arctan2</c> of every pattern against
        ///     pattern <c>(i * 40503) &amp; 0xFFFF</c> (a bijection), hash to NumPy's digests — contiguous, and log1p over the
        ///     reversed view (the strided route).
        /// </summary>
        [TestMethod]
        public void Float16_EveryPattern_HashesToNumPys()
        {
            using var h = AllHalves();
            var xb = new ushort[65536];
            for (int i = 0; i < xb.Length; i++)
                xb[i] = (ushort)((i * 40503) & 0xFFFF);
            using var xu = np.array(xb);
            using var x = xu.view(np.float16);

            using var log1p = np.log1p(h);
            Sha256<Half>(log1p).Should().Be("941AA909B1129E7E10A6C39BBB5292A36A388D60E21B78076E7723085B4912EF");
            using var expm1 = np.expm1(h);
            Sha256<Half>(expm1).Should().Be("A0E24C5105BCFC9EF1DFA222B382AAC56A8D03809C6C8533AC780D04CD328FE7");
            // Over 65,536 pairs some atan2f result sits close enough to a float16 midpoint that a C library one ULP apart
            // narrows it the other way: Apple's libm does (macOS CI 2026-11-07), so this digest is the UCRT's. The log1p and
            // expm1 digests above held on glibc and Apple's libm too.
            if (HostLibm.IsPinnedHost)
            {
                using var arctan2 = np.arctan2(h, x);
                Sha256<Half>(arctan2).Should().Be("99940207A2B1FCDB2216EDABF1E8B7011EE6E81711E37B849635EDDF7F05A65A");
            }

            // The strided route: log1p of the reversed view, reversed back, is the same 65,536 results.
            using var rev = h["::-1"];
            using var revResult = np.log1p(rev);
            using var back = revResult["::-1"];
            Sha256<Half>(back).Should().Be("941AA909B1129E7E10A6C39BBB5292A36A388D60E21B78076E7723085B4912EF");
        }

        /// <summary>
        ///     Exhaustive enough for float32: arctan2 over 1,048,576 pairs built from two multiplicative hashes of the index
        ///     (every exponent, both signs, specials included) hashes to NumPy's digest — the CRT's atan2f on every pair.
        /// </summary>
        [TestMethod]
        public void Float32_Arctan2_HashesToNumPys()
        {
            // Every one of the 1,048,576 results is atan2f's own float32 bits, so the digest is the win-amd64 UCRT's; glibc's
            // and Apple's atan2f differ in some last bits (CI 2026-11-07).
            HostLibm.RequirePinnedHost(nameof(Float32_Arctan2_HashesToNumPys), "atan2f's float32 results");
            const int n = 1 << 20;
            var ab = new uint[n];
            var bb = new uint[n];
            for (int i = 0; i < n; i++)
            {
                ab[i] = unchecked((uint)((ulong)i * 0x9E3779B1UL));
                bb[i] = unchecked((uint)((ulong)i * 0x85EBCA77UL + 0x2545F491UL));
            }
            using var au = np.array(ab);
            using var bu = np.array(bb);
            using var a = au.view(np.float32);
            using var b = bu.view(np.float32);

            using var r = np.arctan2(a, b);
            Sha256<float>(r).Should().Be("A1E76B2E5B4237B8714DF12C2B2452E99FDCE4CCDD9DABCE5392EF840516F491");
        }
    }
}
