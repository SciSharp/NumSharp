using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Value-level fidelity for the inputs the bulk round-trip sweeps deliberately skip — NaN
    ///     (sign + payload + signaling), ±inf, signed zero, subnormals, and the exact dtype extremes.
    ///     Ported from the pythonnet bridge's <c>SpecialValueFidelityTests</c>: the ORT crossing is a
    ///     pure byte copy (a shared pointer for <c>AsOrtValue</c>, a <c>memcpy</c> for <c>ToOrtValue</c> /
    ///     <c>ToNDArray</c>), so the contract is <b>bit preservation</b> — whatever bits the source holds
    ///     must survive intact, ORT never canonicalizes them.
    ///
    ///     <para>Where the pythonnet test asks live numpy to compute over the bytes, the ORT analog is
    ///     stronger: every ladder is also driven through a <b>real ONNX Runtime <c>Identity</c> session</b>,
    ///     so the native runtime itself is proven to preserve every special bit end to end. The bit
    ///     patterns are pinned via <c>np.frombuffer</c> (independent of any float parsing), and CLR-origin
    ///     NaN — whose sign bit differs from numpy's canonical <c>np.nan</c> — is proven to round-trip
    ///     uncanonicalized rather than by asserting a specific encoding.</para>
    /// </summary>
    [TestClass]
    public class SpecialValueFidelityTests : OnnxTestBase
    {
        // The full float ladder, per width, as raw bit patterns:
        //   0:+0  1:-0  2:+inf  3:-inf  4:1.0  5:-1.0  6:min-normal  7:min-subnormal
        //   8:max-finite  9:lowest-finite  10:qNaN  11:qNaN(neg — the .NET NaN)  12:sNaN  13:payload-NaN
        private static readonly ulong[] F8 =
        {
            0x0000000000000000, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
            0x3FF0000000000000, 0xBFF0000000000000, 0x0010000000000000, 0x0000000000000001,
            0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, 0x7FF8000000000000, 0xFFF8000000000000,
            0x7FF0000000000001, 0x7FF5555555555555,
        };

        private static readonly uint[] F4 =
        {
            0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x3F800000, 0xBF800000, 0x00800000,
            0x00000001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FD55555,
        };

        private static readonly ushort[] F2 =
        {
            0x0000, 0x8000, 0x7C00, 0xFC00, 0x3C00, 0xBC00, 0x0400, 0x0001,
            0x7BFF, 0xFBFF, 0x7E00, 0xFE00, 0x7C01, 0x7D55,
        };

        // ==============================  float specials: every crossing + a real session  ======

        [TestMethod]
        public void Float64_Specials_CrossBitExact_ThroughEveryPath_AndTheSession()
        {
            using NDArray nd = np.frombuffer(Bytes(F8), NPTypeCode.Double);
            nd.typecode.Should().Be(NPTypeCode.Double);
            AssertCrossesBitExact(nd, "float64 ladder");
        }

        [TestMethod]
        public void Float32_Specials_CrossBitExact_ThroughEveryPath_AndTheSession()
        {
            using NDArray nd = np.frombuffer(Bytes(F4), NPTypeCode.Single);
            nd.typecode.Should().Be(NPTypeCode.Single);
            AssertCrossesBitExact(nd, "float32 ladder");
        }

        [TestMethod]
        public void Float16_Specials_CrossBitExact_ThroughEveryPath_AndTheSession()
        {
            using NDArray nd = np.frombuffer(Bytes(F2), NPTypeCode.Half);
            nd.typecode.Should().Be(NPTypeCode.Half);
            AssertCrossesBitExact(nd, "float16 ladder");
        }

        [TestMethod]
        public void FloatSpecials_ArePreservedSemantically_AfterASessionRoundTrip()
        {
            // Not just the bits — the values are still the specials on the far side (isinf/isnan/signed-zero).
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray nd = np.frombuffer(Bytes(F8), NPTypeCode.Double);
            using NDArray back = session.Run(nd);

            using NDArray isinf = np.isinf(back);
            isinf.GetBoolean(2).Should().BeTrue("+inf survived");
            isinf.GetBoolean(3).Should().BeTrue("-inf survived");

            using NDArray isnan = np.isnan(back);
            foreach (int i in new[] { 10, 11, 12, 13 })
                isnan.GetBoolean(i).Should().BeTrue($"the NaN at index {i} is still a NaN");

            back.GetDouble(0).Should().Be(0.0);
            (1.0 / back.GetDouble(1)).Should().Be(double.NegativeInfinity, "the -0.0 sign bit made it through");
            back.GetDouble(4).Should().Be(1.0);
            back.GetDouble(8).Should().Be(double.MaxValue);
            back.GetDouble(9).Should().Be(double.MinValue);
        }

        // ==============================  signed zero: distinguishable both ways  ================

        [TestMethod]
        public void SignedZero_SurvivesBothDirections_AndTheSignBitIsObservable()
        {
            using NDArray nd = np.array(new[] { 0.0, -0.0 });
            AssertCrossesBitExact(nd, "signed zero");

            using OrtTensor t = nd.AsOrtValue();
            ReadOnlySpan<double> span = t.Value.GetTensorDataAsSpan<double>();
            (1.0 / span[0]).Should().Be(double.PositiveInfinity, "+0.0 stays +0.0");
            (1.0 / span[1]).Should().Be(double.NegativeInfinity, "-0.0 keeps its sign bit across the crossing");

            // and ORT can write a -0.0 back into the shared buffer, visible to NumSharp
            t.Value.GetTensorMutableDataAsSpan<double>()[0] = -0.0;
            (1.0 / nd.GetDouble(0)).Should().Be(double.NegativeInfinity, "ORT's -0.0 write is signbit-visible in NumSharp");
        }

        // ==============================  CLR-origin NaN: preserved, not canonicalized  ==========

        [TestMethod]
        public void ClrOriginNaN_RoundTripsUncanonicalized_ThroughTheCrossingAndTheSession()
        {
            // A .NET NaN is 0xFFF8… (negative quiet); numpy's canonical np.nan is 0x7FF8… — the crossing must
            // NOT canonicalize it. Prove bit-identity through every path AND that it is still a NaN on the far side.
            using NDArray nd = np.array(new[] { double.NaN, 1.0, double.NaN });
            BitConverter.DoubleToUInt64Bits(nd.GetDouble(0)).Should().Be(0xFFF8000000000000, "the .NET NaN encoding, up front");

            AssertCrossesBitExact(nd, "CLR-origin NaN");

            using InferenceSession session = OpenSession("identity_float64");
            using NDArray back = session.Run(nd);
            BitConverter.DoubleToUInt64Bits(back.GetDouble(0)).Should().Be(0xFFF8000000000000, "the exact CLR NaN bits survived the native Identity op");
            back.GetDouble(1).Should().Be(1.0);
            using NDArray isnan = np.isnan(back);
            isnan.GetBoolean(0).Should().BeTrue();
            isnan.GetBoolean(2).Should().BeTrue();
        }

        // ==============================  integer extremes: round-trip both ways + session  ======

        [TestMethod]
        public void IntegerExtremes_CrossBitExact_EveryWidth_ThroughEveryPath_AndTheSession()
        {
            AssertCrossesBitExact(np.array(new byte[] { 0, 1, byte.MaxValue }), "uint8", dispose: true);
            AssertCrossesBitExact(np.array(new sbyte[] { 0, -1, sbyte.MaxValue, sbyte.MinValue }), "int8", dispose: true);
            AssertCrossesBitExact(np.array(new short[] { 0, -1, short.MaxValue, short.MinValue }), "int16", dispose: true);
            AssertCrossesBitExact(np.array(new ushort[] { 0, 1, ushort.MaxValue }), "uint16", dispose: true);
            AssertCrossesBitExact(np.array(new[] { 0, -1, int.MaxValue, int.MinValue }), "int32", dispose: true);
            AssertCrossesBitExact(np.array(new uint[] { 0, 1, uint.MaxValue }), "uint32", dispose: true);
            AssertCrossesBitExact(np.array(new[] { 0L, -1L, long.MaxValue, long.MinValue }), "int64", dispose: true);
            AssertCrossesBitExact(np.array(new[] { 0UL, 1UL, ulong.MaxValue }), "uint64", dispose: true);
            AssertCrossesBitExact(np.array(new[] { false, true, false }), "bool", dispose: true);
        }

        // ==============================  Char / UTF-16 BMP boundaries  ==========================

        [TestMethod]
        public void Char_BmpBoundaries_CrossAsUInt16_BitExact()
        {
            // numpy has no char dtype; a C# char is a 2-byte UTF-16 code unit == uint16. Every BMP boundary
            // must cross byte-identically and (directionally) come back as UInt16.
            ushort[] boundaries = { 0x0000, 0x0041, 0x007F, 0x0080, 0x07FF, 0x0800, 0xD7FF, 0xE000, 0xFFFF };
            byte[] bytes = Bytes(boundaries);
            using NDArray chars = np.frombuffer(bytes, NPTypeCode.Char);
            chars.typecode.Should().Be(NPTypeCode.Char);

            using (OrtTensor t = chars.AsOrtValue())
            {
                t.ElementType.Should().Be(TensorElementType.UInt16);
                ReadOnlySpan<ushort> span = t.Value.GetTensorDataAsSpan<ushort>();
                for (int i = 0; i < boundaries.Length; i++)
                    span[i].Should().Be(boundaries[i], $"code unit 0x{boundaries[i]:X4} crosses untouched");

                using NDArray back = t.Value.ToNDArray();
                back.typecode.Should().Be(NPTypeCode.UInt16, "an ORT UInt16 tensor carries no 'these are chars' bit");
                BytesOf(back).Should().Equal(bytes);
            }

            // through a real Identity session too (declared uint16)
            using (InferenceSession session = OpenSession(IdentityModel(NPTypeCode.Char)))
            using (NDArray y = session.Run(chars))
                BytesOf(y).Should().Equal(bytes, "the native Identity op preserves the code units");

            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        // ==============================  the shared bit-preservation harness  ===================

        /// <summary>
        ///     Assert an array's exact bytes survive every crossing that is supposed to be a byte copy:
        ///     the zero-copy <c>AsOrtValue</c> (its tensor bytes, then <c>ToNDArray</c> and <c>AsNDArray</c>
        ///     back), the copying <c>ToOrtValue</c> (its tensor bytes, then <c>ToNDArray</c> back), and a
        ///     round-trip through a real ORT <c>Identity</c> session. The leak counters must return to zero.
        /// </summary>
        private void AssertCrossesBitExact(NDArray nd, string label, bool dispose = false)
        {
            try
            {
                byte[] bytes = BytesOf(nd);
                NPTypeCode code = nd.typecode;

                using (OrtTensor t = nd.AsOrtValue())
                {
                    t.Value.GetTensorMutableRawData().ToArray().Should().Equal(bytes, $"{label}: AsOrtValue tensor bytes are the exact source bits");
                    using (NDArray copy = t.Value.ToNDArray())
                        BytesOf(copy).Should().Equal(bytes, $"{label}: AsOrtValue -> ToNDArray is bit-identical");
                    using (NDArray view = t.Value.AsNDArray())
                        BytesOf(view).Should().Equal(bytes, $"{label}: AsOrtValue -> AsNDArray view is bit-identical");
                }

                using (OrtValue v = nd.ToOrtValue())
                {
                    v.GetTensorMutableRawData().ToArray().Should().Equal(bytes, $"{label}: ToOrtValue copied the exact bits");
                    using (NDArray copy = v.ToNDArray())
                        BytesOf(copy).Should().Equal(bytes, $"{label}: ToOrtValue -> ToNDArray is bit-identical");
                }

                using (InferenceSession session = OpenSession(IdentityModel(code)))
                using (NDArray y = session.Run(nd))
                    BytesOf(y).Should().Equal(bytes, $"{label}: a real ORT Identity session preserves every bit");

                NDArrayOnnxInterop.LiveExports.Should().Be(0, $"{label}: no export leaked");
                NDArrayOnnxInterop.LiveImports.Should().Be(0, $"{label}: no import leaked");
            }
            finally
            {
                if (dispose)
                    nd.Dispose();
            }
        }

        private static byte[] Bytes(ushort[] p) { var b = new byte[p.Length * sizeof(ushort)]; Buffer.BlockCopy(p, 0, b, 0, b.Length); return b; }
        private static byte[] Bytes(uint[] p) { var b = new byte[p.Length * sizeof(uint)]; Buffer.BlockCopy(p, 0, b, 0, b.Length); return b; }
        private static byte[] Bytes(ulong[] p) { var b = new byte[p.Length * sizeof(ulong)]; Buffer.BlockCopy(p, 0, b, 0, b.Length); return b; }
    }
}
