using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Bit-exact fidelity of adversarial values through a REAL identity session: NaN (payloads included),
    ///     ±inf, signed zero, subnormals and the integer extremes must come back byte-for-byte — the interop
    ///     shares/copies raw bytes and never reinterprets them, so nothing is canonicalized.
    /// </summary>
    [TestClass]
    public class ValueFidelityTests : OnnxTestBase
    {
        private NDArray RoundTrip(string model, NDArray input)
        {
            using InferenceSession session = OpenSession(model);
            using OrtTensor t = input.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs =
                session.Run(options, new[] { "X" }, new[] { t.Value }, new[] { "Y" });
            return outputs[0].ToNDArray();
        }

        [TestMethod]
        public void Float64_Specials_RoundTripBitExact()
        {
            using NDArray x = np.array(new[]
            {
                double.NaN, double.PositiveInfinity, double.NegativeInfinity,
                -0.0, 0.0, double.Epsilon /* 5e-324 subnormal */, 2.2250738585072014e-308 /* min normal */, -1.5,
            });
            using NDArray y = RoundTrip("identity_float64", x);
            y.typecode.Should().Be(NPTypeCode.Double);
            BytesOf(y).Should().Equal(BytesOf(x), "every NaN/inf/-0/subnormal bit pattern survives identity unchanged");
            BitConverter.DoubleToInt64Bits(y.GetDouble(3)).Should().Be(BitConverter.DoubleToInt64Bits(-0.0), "-0.0 keeps its sign bit");
        }

        [TestMethod]
        public void Float32_Specials_RoundTripBitExact()
        {
            using NDArray x = np.array(new[]
            {
                float.NaN, float.PositiveInfinity, float.NegativeInfinity,
                -0.0f, 0.0f, float.Epsilon /* subnormal */, 1.17549435e-38f /* min normal */, -1.5f,
            });
            using NDArray y = RoundTrip("identity_float32", x);
            y.typecode.Should().Be(NPTypeCode.Single);
            BytesOf(y).Should().Equal(BytesOf(x));
            BitConverter.SingleToInt32Bits(y.GetSingle(3)).Should().Be(BitConverter.SingleToInt32Bits(-0.0f));
        }

        [TestMethod]
        public void Float16_NaNPayloadsAndSubnormals_SurviveARealSessionRun()
        {
            ushort[] patterns = { 0x0000, 0x8000, 0x3C00, 0xBC00, 0x0001, 0x8001, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x7D55, 0xFFFF, 0x03FF, 0x7BFF };
            var bytes = new byte[patterns.Length * 2];
            Buffer.BlockCopy(patterns, 0, bytes, 0, bytes.Length);
            using NDArray x = np.frombuffer(bytes, NPTypeCode.Half);
            using NDArray y = RoundTrip("identity_float16", x);
            y.typecode.Should().Be(NPTypeCode.Half);
            BytesOf(y).Should().Equal(bytes, "float16 NaN payloads / subnormals cross a real ORT run bit-identical");
        }

        [TestMethod]
        public void Int64_And_UInt64_Extremes_RoundTripExact()
        {
            using NDArray i64 = np.array(new[] { long.MinValue, long.MaxValue, -1L, 0L, 9223372036854775807L });
            using NDArray yi = RoundTrip("identity_int64", i64);
            yi.GetInt64(0).Should().Be(long.MinValue);
            yi.GetInt64(1).Should().Be(long.MaxValue);
            BytesOf(yi).Should().Equal(BytesOf(i64));

            using NDArray u64 = np.array(new[] { ulong.MinValue, ulong.MaxValue, 9223372036854775808UL });
            using NDArray yu = RoundTrip("identity_uint64", u64);
            yu.GetUInt64(1).Should().Be(ulong.MaxValue, "2^64-1 is not truncated");
            BytesOf(yu).Should().Equal(BytesOf(u64));
        }

        [TestMethod]
        public void SignedAndUnsignedByteExtremes_RoundTripExact()
        {
            using NDArray i8 = np.array(new sbyte[] { sbyte.MinValue, -1, 0, sbyte.MaxValue });
            using NDArray yi = RoundTrip("identity_int8", i8);
            yi.GetValue<sbyte>(0).Should().Be(sbyte.MinValue);
            yi.GetValue<sbyte>(3).Should().Be(sbyte.MaxValue);

            using NDArray u8 = np.array(new byte[] { 0, 1, 127, 128, 255 });
            using NDArray yu = RoundTrip("identity_uint8", u8);
            BytesOf(yu).Should().Equal(BytesOf(u8));
        }

        [TestMethod]
        public void Bool_RoundTripsExact()
        {
            using NDArray b = np.array(new[] { true, false, true, true, false });
            using NDArray y = RoundTrip("identity_bool", b);
            y.typecode.Should().Be(NPTypeCode.Boolean);
            BytesOf(y).Should().Equal(BytesOf(b));
        }
    }
}
