using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The dtype seam (plan §6): 12 zero-copy dtypes, the two conversions (Char → UInt16, Decimal → Double
    ///     at the array verbs only) and the three refused mutual gaps (Complex, BFloat16, String).
    /// </summary>
    [TestClass]
    public class DtypeMapTests : OnnxTestBase
    {
        [TestMethod]
        public void ZeroCopyDtypes_RoundTrip_BothDirections()
        {
            (NPTypeCode code, TensorElementType type)[] expected =
            {
                (NPTypeCode.Boolean, TensorElementType.Bool),
                (NPTypeCode.Byte, TensorElementType.UInt8),
                (NPTypeCode.SByte, TensorElementType.Int8),
                (NPTypeCode.Int16, TensorElementType.Int16),
                (NPTypeCode.UInt16, TensorElementType.UInt16),
                (NPTypeCode.Int32, TensorElementType.Int32),
                (NPTypeCode.UInt32, TensorElementType.UInt32),
                (NPTypeCode.Int64, TensorElementType.Int64),
                (NPTypeCode.UInt64, TensorElementType.UInt64),
                (NPTypeCode.Half, TensorElementType.Float16),
                (NPTypeCode.Single, TensorElementType.Float),
                (NPTypeCode.Double, TensorElementType.Double),
            };

            foreach (var (code, type) in expected)
            {
                NDArrayOnnxInterop.ToTensorElementType(code).Should().Be(type, $"{code} exports as {type}");
                NDArrayOnnxInterop.FromTensorElementType(type).Should().Be(code, $"{type} imports as {code}");
                NDArrayOnnxInterop.TryToTensorElementType(code, out TensorElementType t).Should().BeTrue();
                t.Should().Be(type);
                NDArrayOnnxInterop.TryFromTensorElementType(type, out NPTypeCode c).Should().BeTrue();
                c.Should().Be(code);
            }
        }

        [TestMethod]
        public void Char_ExportsAsUInt16_ButUInt16_ImportsAsUInt16()
        {
            NDArrayOnnxInterop.ToTensorElementType(NPTypeCode.Char).Should().Be(TensorElementType.UInt16);
            NDArrayOnnxInterop.FromTensorElementType(TensorElementType.UInt16).Should().Be(NPTypeCode.UInt16, "an ORT tensor carries no 'these are characters' bit");
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Char).Should().Be(typeof(ushort));
        }

        [TestMethod]
        public void Half_MapsToOrtFloat16_SameSixteenBits()
        {
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Half).Should().Be(typeof(Float16));
            unsafe
            {
                sizeof(Float16).Should().Be(sizeof(Half));
            }

            // the reinterpret is exact: ORT's Float16 exposes the raw 16 bits as `value`
            var h = (Half)1.5f;
            ushort bits = BitConverter.HalfToUInt16Bits(h);
            new Float16(bits).value.Should().Be(bits);
            ((float)new Float16(bits)).Should().Be(1.5f);
        }

        [TestMethod]
        public void Decimal_And_Complex_AreRefusedByTheLowLevelMap()
        {
            new Action(() => NDArrayOnnxInterop.ToTensorElementType(NPTypeCode.Decimal)).Should().Throw<NotSupportedException>()
                .WithMessage("*Decimal*astype(NPTypeCode.Double)*");
            new Action(() => NDArrayOnnxInterop.ToTensorElementType(NPTypeCode.Complex)).Should().Throw<NotSupportedException>()
                .WithMessage("*complex*np.real*");
            NDArrayOnnxInterop.TryToTensorElementType(NPTypeCode.Decimal, out _).Should().BeFalse();
            NDArrayOnnxInterop.TryToTensorElementType(NPTypeCode.Complex, out _).Should().BeFalse();
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Decimal).Should().BeNull();
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Complex).Should().BeNull();
        }

        [TestMethod]
        public void String_BFloat16_Complex_AreRefusedOnImport()
        {
            new Action(() => NDArrayOnnxInterop.FromTensorElementType(TensorElementType.String)).Should().Throw<NotSupportedException>()
                .WithMessage("*GetStringTensorAsArray*");
            new Action(() => NDArrayOnnxInterop.FromTensorElementType(TensorElementType.BFloat16)).Should().Throw<NotSupportedException>()
                .WithMessage("*bfloat16*");
            new Action(() => NDArrayOnnxInterop.FromTensorElementType(TensorElementType.Complex64)).Should().Throw<NotSupportedException>();
            new Action(() => NDArrayOnnxInterop.FromTensorElementType(TensorElementType.Complex128)).Should().Throw<NotSupportedException>();
            NDArrayOnnxInterop.TryFromTensorElementType(TensorElementType.String, out _).Should().BeFalse();
            NDArrayOnnxInterop.TryFromTensorElementType(TensorElementType.BFloat16, out _).Should().BeFalse();
        }

        [TestMethod]
        public void ComplexArray_IsRefusedByEveryExportVerb()
        {
            using var z = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            new Action(() => z.AsOrtValue()).Should().Throw<NotSupportedException>().WithMessage("*complex*");
            new Action(() => z.ToOrtValue()).Should().Throw<NotSupportedException>().WithMessage("*complex*");
            new Action(() => z.AsDenseTensor<Complex>()).Should().Throw<NotSupportedException>();
            new Action(() => z.ToDenseTensor<Complex>()).Should().Throw<NotSupportedException>();
        }
    }
}
