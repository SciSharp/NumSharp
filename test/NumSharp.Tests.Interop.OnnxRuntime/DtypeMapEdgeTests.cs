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
    ///     Edge coverage of the dtype seam that <see cref="DtypeMapTests"/> leaves: the FULL
    ///     <see cref="NDArrayOnnxInterop.ToTensorElementClrType"/> map, the generic <c>TypeCodeOf&lt;T&gt;</c>
    ///     (including the three throwing T's and the catch-all), <c>EnsureElementType&lt;T&gt;</c>'s three
    ///     outcomes, and a string <see cref="OrtValue"/> refused at the <c>IsString</c> gate of both import verbs.
    /// </summary>
    [TestClass]
    public class DtypeMapEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void ToTensorElementClrType_MapsEveryDtype()
        {
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Boolean).Should().Be(typeof(bool));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Byte).Should().Be(typeof(byte));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.SByte).Should().Be(typeof(sbyte));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Int16).Should().Be(typeof(short));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.UInt16).Should().Be(typeof(ushort));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Int32).Should().Be(typeof(int));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.UInt32).Should().Be(typeof(uint));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Int64).Should().Be(typeof(long));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.UInt64).Should().Be(typeof(ulong));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Char).Should().Be(typeof(ushort), "chars cross as UTF-16 code units");
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Half).Should().Be(typeof(Float16), "System.Half is ORT's Float16");
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Single).Should().Be(typeof(float));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Double).Should().Be(typeof(double));
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Decimal).Should().BeNull("no ONNX element type");
            NDArrayOnnxInterop.ToTensorElementClrType(NPTypeCode.Complex).Should().BeNull("no ONNX element type");
        }

        [TestMethod]
        public void TypeCodeOf_MapsEveryOrtElementType_IncludingTheDirectionalFloat16AndChar()
        {
            NDArrayOnnxInterop.TypeCodeOf<bool>().Should().Be(NPTypeCode.Boolean);
            NDArrayOnnxInterop.TypeCodeOf<byte>().Should().Be(NPTypeCode.Byte);
            NDArrayOnnxInterop.TypeCodeOf<sbyte>().Should().Be(NPTypeCode.SByte);
            NDArrayOnnxInterop.TypeCodeOf<short>().Should().Be(NPTypeCode.Int16);
            NDArrayOnnxInterop.TypeCodeOf<ushort>().Should().Be(NPTypeCode.UInt16, "a raw ushort tensor comes back as UInt16, never Char");
            NDArrayOnnxInterop.TypeCodeOf<int>().Should().Be(NPTypeCode.Int32);
            NDArrayOnnxInterop.TypeCodeOf<uint>().Should().Be(NPTypeCode.UInt32);
            NDArrayOnnxInterop.TypeCodeOf<long>().Should().Be(NPTypeCode.Int64);
            NDArrayOnnxInterop.TypeCodeOf<ulong>().Should().Be(NPTypeCode.UInt64);
            NDArrayOnnxInterop.TypeCodeOf<char>().Should().Be(NPTypeCode.Char, "a DenseTensor<char> is unambiguously chars");
            NDArrayOnnxInterop.TypeCodeOf<Float16>().Should().Be(NPTypeCode.Half);
            NDArrayOnnxInterop.TypeCodeOf<Half>().Should().Be(NPTypeCode.Half, "System.Half and ORT's Float16 both map to Half");
            NDArrayOnnxInterop.TypeCodeOf<float>().Should().Be(NPTypeCode.Single);
            NDArrayOnnxInterop.TypeCodeOf<double>().Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void TypeCodeOf_RefusesBFloat16_Complex_Decimal_AndAnUnknownStruct()
        {
            new Action(() => NDArrayOnnxInterop.TypeCodeOf<BFloat16>()).Should().Throw<NotSupportedException>().WithMessage("*bfloat16*");
            new Action(() => NDArrayOnnxInterop.TypeCodeOf<Complex>()).Should().Throw<NotSupportedException>().WithMessage("*complex*");
            new Action(() => NDArrayOnnxInterop.TypeCodeOf<decimal>()).Should().Throw<NotSupportedException>().WithMessage("*Decimal*");
            // any other unmanaged struct falls to the catch-all message naming what ORT tensors carry
            new Action(() => NDArrayOnnxInterop.TypeCodeOf<DateTime>()).Should().Throw<NotSupportedException>()
                .WithMessage("*has no NumSharp dtype*");
        }

        [TestMethod]
        public void EnsureElementType_Matches_Mismatches_AndTheNullWantCase()
        {
            // the C# type ORT uses for the dtype — no throw
            new Action(() => NDArrayOnnxInterop.EnsureElementType<float>(NPTypeCode.Single, "AsDenseTensor")).Should().NotThrow();
            new Action(() => NDArrayOnnxInterop.EnsureElementType<Float16>(NPTypeCode.Half, "AsDenseTensor")).Should().NotThrow();
            new Action(() => NDArrayOnnxInterop.EnsureElementType<ushort>(NPTypeCode.Char, "AsDenseTensor")).Should().NotThrow("chars cross as ushort");

            // wrong T for the dtype — refused, never a silent reinterpret
            new Action(() => NDArrayOnnxInterop.EnsureElementType<int>(NPTypeCode.Single, "AsDenseTensor"))
                .Should().Throw<ArgumentException>().WithMessage("*DenseTensor<Single>*");
            new Action(() => NDArrayOnnxInterop.EnsureElementType<Half>(NPTypeCode.Half, "AsDenseTensor"))
                .Should().Throw<ArgumentException>().WithMessage("*Float16*", "ORT spells float16 as Microsoft.ML.OnnxRuntime.Float16");

            // a dtype with no ORT element type at all (want == null) reports the export gap, not a T mismatch
            new Action(() => NDArrayOnnxInterop.EnsureElementType<Complex>(NPTypeCode.Complex, "AsDenseTensor"))
                .Should().Throw<NotSupportedException>().WithMessage("*complex*");
        }

        [TestMethod]
        public void StringTensor_IsRefused_AtTheIsStringGate_OfBothImportVerbs()
        {
            using OrtValue strings = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 2 });
            strings.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_TENSOR);
            strings.GetTensorTypeAndShape().IsString.Should().BeTrue();

            new Action(() => strings.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*GetStringTensorAsArray*");
            new Action(() => strings.AsNDArray()).Should().Throw<NotSupportedException>().WithMessage("*GetStringTensorAsArray*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a refused import leases nothing");
        }
    }
}
