using System;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Interop
{
    /// <summary>
    ///     Byte-level interop contract between NumSharp and NumPy-shaped buffers (the data exchanged with
    ///     Numpy.NET / .npy files / any C-order numpy buffer). Every expectation below was probed against
    ///     NumPy 2.4.2. These pin the divergences and the bit-exact guarantees found while auditing interop:
    ///     endianness handling, NaN-payload / signaling-NaN / subnormal / -0 preservation, uint64 above
    ///     int64.max, complex part layout, Char↔uint16 code units, and Decimal having no numpy dtype.
    /// </summary>
    [TestClass]
    public class NumpyByteContractTests
    {
        // ---- Endianness --------------------------------------------------------------

        [TestMethod]
        public void Endianness_BigEndian_StringDtype_Byteswaps()
        {
            // big-endian int32 for [1, 256, 65536]
            byte[] be = { 0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0 };
            var nd = np.frombuffer(be, ">i4"); // string dtype carries endianness
            CollectionAssert.AreEqual(new[] { 1, 256, 65536 }, nd.ToArray<int>());
        }

        [TestMethod]
        public void Endianness_BigEndian_Float64_StringDtype_Byteswaps()
        {
            double[] vals = { 1.5, -2.25, 1e10 };
            byte[] be = new byte[vals.Length * 8];
            for (int i = 0; i < vals.Length; i++)
            {
                var le = BitConverter.GetBytes(vals[i]); // x64 is little-endian
                Array.Reverse(le);
                Array.Copy(le, 0, be, i * 8, 8);
            }
            var nd = np.frombuffer(be, ">f8");
            CollectionAssert.AreEqual(vals, nd.ToArray<double>());
        }

        [TestMethod]
        public void Endianness_NPTypeCode_IsLittleEndian_NoSwap()
        {
            // The NPTypeCode overload has no endian information => it reads native little-endian.
            // Feeding it big-endian bytes therefore reinterprets them (documents the raw-API hazard).
            byte[] be = { 0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0 };
            var nd = np.frombuffer(be, NPTypeCode.Int32);
            CollectionAssert.AreEqual(new[] { 16777216, 65536, 256 }, nd.ToArray<int>());
        }

        // ---- NaN payloads / signaling-NaN / subnormals / -0 (bit-exact transport) -----

        [TestMethod]
        public void Half_Specials_BitExact_RoundTrip()
        {
            // float16 bits: nan=0x7e00, inf=0x7c00, -inf=0xfc00, -0=0x8000, min-subnormal=0x0001, max=0x7bff
            byte[] bits = { 0x00,0x7e,  0x00,0x7c,  0x00,0xfc,  0x00,0x80,  0x01,0x00,  0xff,0x7b };
            var h = np.frombuffer(bits, NPTypeCode.Half);
            CollectionAssert.AreEqual(bits, h.ToByteArray(), "half bit pattern (incl. NaN payload/subnormal) must survive");
            var a = h.ToArray<Half>();
            Assert.IsTrue(Half.IsNaN(a[0]));
            Assert.IsTrue(Half.IsPositiveInfinity(a[1]));
            Assert.IsTrue(Half.IsNegativeInfinity(a[2]));
            Assert.IsTrue(Half.IsNegative(a[3]) && a[3] == (Half)0.0, "negative zero");
            Assert.IsTrue(a[4] > (Half)0.0 && a[4] < (Half)6e-5, "smallest subnormal");
        }

        [TestMethod]
        public void Single_SignalingNaN_Payload_NotQuieted()
        {
            // 0x7f800001 is a signaling NaN; the transport must not quiet it.
            byte[] sbits = { 0x01, 0x00, 0x80, 0x7f };
            var s = np.frombuffer(sbits, NPTypeCode.Single);
            Assert.IsTrue(float.IsNaN(s.ToArray<float>()[0]));
            CollectionAssert.AreEqual(sbits, s.ToByteArray(), "signaling-NaN bits preserved");
        }

        [TestMethod]
        public void Double_MinSubnormal_Preserved()
        {
            byte[] bits = { 0x01, 0, 0, 0, 0, 0, 0, 0 }; // 0x...0001 == double.Epsilon == 5e-324
            var d = np.frombuffer(bits, NPTypeCode.Double);
            Assert.AreEqual(double.Epsilon, d.ToArray<double>()[0]);
            CollectionAssert.AreEqual(bits, d.ToByteArray());
        }

        [TestMethod]
        public void NegativeZero_SignBit_Preserved()
        {
            var nd = np.array(new[] { -0.0, 0.0 });
            byte[] b = nd.ToByteArray();
            Assert.IsTrue(double.IsNegative(nd.ToArray<double>()[0]));
            Assert.IsFalse(double.IsNegative(nd.ToArray<double>()[1]));
            Assert.AreEqual(0x80, b[7], "-0.0 sign bit in the high byte (little-endian)");
            Assert.AreEqual(0x00, b[15], "+0.0");
        }

        // ---- Integer extremes --------------------------------------------------------

        [TestMethod]
        public void UInt64_AboveInt64Max_RoundTrips()
        {
            ulong[] u = { 0, 1, 9223372036854775808UL, ulong.MaxValue };
            byte[] bytes = new byte[u.Length * 8];
            Buffer.BlockCopy(u, 0, bytes, 0, bytes.Length);
            var nd = np.frombuffer(bytes, NPTypeCode.UInt64);
            CollectionAssert.AreEqual(u, nd.ToArray<ulong>());
            CollectionAssert.AreEqual(bytes, nd.ToByteArray());
        }

        // ---- Complex layout ----------------------------------------------------------

        [TestMethod]
        public void Complex_NaNInf_Parts_BitExact_RoundTrip()
        {
            var c = np.array(new[]
            {
                new Complex(double.NaN, 1),
                new Complex(double.PositiveInfinity, double.NegativeInfinity),
                new Complex(-0.0, 0.0)
            });
            Assert.AreEqual(16, c.dtypesize, "complex128 == 2 doubles");
            var rt = np.frombuffer(c.ToByteArray(), NPTypeCode.Complex);
            CollectionAssert.AreEqual(c.ToByteArray(), rt.ToByteArray());
            var a = rt.ToArray<Complex>();
            Assert.IsTrue(double.IsNaN(a[0].Real) && a[0].Imaginary == 1.0);
            Assert.IsTrue(double.IsPositiveInfinity(a[1].Real) && double.IsNegativeInfinity(a[1].Imaginary));
            Assert.IsTrue(double.IsNegative(a[2].Real), "real part is -0.0");
        }

        // ---- Char (UTF-16 code unit) -------------------------------------------------

        [TestMethod]
        public void Char_MapsToUInt16_CodeUnits_SurrogatePairsSplit()
        {
            // "A😀Z": the emoji U+1F600 is a surrogate pair in UTF-16 -> two code units.
            char[] chars = "A\U0001F600Z".ToCharArray();
            byte[] cb = MemoryMarshal.AsBytes<char>(chars).ToArray();
            var ch = np.frombuffer(cb, NPTypeCode.Char);
            Assert.AreEqual(2, ch.dtypesize, "char is a 2-byte code unit (maps to numpy uint16)");
            CollectionAssert.AreEqual(
                new[] { 65, 55357, 56832, 90 },
                ch.ToArray<char>().Select(c => (int)c).ToArray());
        }

        // ---- Decimal has no numpy dtype ----------------------------------------------

        [TestMethod]
        public void Decimal_Is16Bytes_AndReinterpretAsFloatIsGarbage()
        {
            var dec = np.array(new[] { 1.5m, -2.5m });
            Assert.AreEqual(NPTypeCode.Decimal, dec.typecode);
            Assert.AreEqual(16, dec.dtypesize, "C# decimal is 16 bytes (flags/hi/mid/lo) — NOT IEEE-754");
            Assert.AreEqual(32, dec.ToByteArray().Length);

            // Reinterpreting the 16-byte decimal as IEEE float64 (numpy has no decimal dtype) yields garbage.
            var asDouble = np.frombuffer(dec.ToByteArray(), NPTypeCode.Double);
            Assert.AreNotEqual(1.5, asDouble.ToArray<double>()[0], 1e-6,
                "decimal bytes are not IEEE — a raw reinterpret is meaningless; values must be converted");

            // The correct cross-language path is value conversion (lossy beyond double precision).
            CollectionAssert.AreEqual(new[] { 1.5, -2.5 }, dec.astype(NPTypeCode.Double).ToArray<double>());
        }

        // ---- bool with non-0/1 underlying bytes --------------------------------------

        [TestMethod]
        public void Bool_NonBinaryBytes_StoragePreserved_LikeNumpy()
        {
            // numpy preserves the raw bytes: np.frombuffer(bytes([0,1,2,3,255]), bool).tobytes() == those bytes.
            var b = np.frombuffer(new byte[] { 0, 1, 2, 3, 255 }, NPTypeCode.Boolean);
            CollectionAssert.AreEqual(new byte[] { 0, 1, 2, 3, 255 }, b.ToByteArray());
        }

        [TestMethod]
        public void Bool_NonBinaryBytes_Truthiness_AnyNonzeroIsTrue()
        {
            // numpy: every nonzero byte is True. (NB: C# bool.Equals is a raw byte compare, so a bool whose
            // byte is 2 is NOT .Equals(true) even though it IS truthy — assert truthiness, not equality.)
            var a = np.frombuffer(new byte[] { 0, 1, 2, 3, 255 }, NPTypeCode.Boolean).ToArray<bool>();
            Assert.IsFalse(a[0]);
            Assert.IsTrue(a[1]);
            Assert.IsTrue(a[2]);
            Assert.IsTrue(a[3]);
            Assert.IsTrue(a[4]);
        }

        [TestMethod]
        public void Bool_NonBinaryBytes_Reductions_NormalizeLikeNumpy()
        {
            // numpy normalizes bool in reductions: sum / count count True values (4), not the raw bytes.
            var b = np.frombuffer(new byte[] { 0, 1, 2, 3, 255 }, NPTypeCode.Boolean);
            Assert.AreEqual(4, np.sum(b).astype(NPTypeCode.Int32).ToArray<int>()[0], "sum counts True values");
            Assert.AreEqual(4, b.ToArray<bool>().Count(x => x), "truthiness count");
        }
    }
}
