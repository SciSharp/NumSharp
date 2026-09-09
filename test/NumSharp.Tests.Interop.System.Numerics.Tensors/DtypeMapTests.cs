using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     The dtype seam. Unlike ONNX Runtime (a fixed <c>TensorElementType</c> enum with real gaps),
    ///     System.Numerics.Tensors containers are unconstrained generics: all 15 NumSharp dtypes cross as their
    ///     own CLR type — nothing refused, nothing converted.
    /// </summary>
    [TestClass]
    public class DtypeMapTests : TensorsTestBase
    {
        [TestMethod]
        public void EveryDtype_MapsToItsOwnClrType_AndRoundTrips()
        {
            (NPTypeCode code, Type clr)[] expected =
            {
                (NPTypeCode.Boolean, typeof(bool)),
                (NPTypeCode.Byte, typeof(byte)),
                (NPTypeCode.SByte, typeof(sbyte)),
                (NPTypeCode.Int16, typeof(short)),
                (NPTypeCode.UInt16, typeof(ushort)),
                (NPTypeCode.Int32, typeof(int)),
                (NPTypeCode.UInt32, typeof(uint)),
                (NPTypeCode.Int64, typeof(long)),
                (NPTypeCode.UInt64, typeof(ulong)),
                (NPTypeCode.Char, typeof(char)),
                (NPTypeCode.Half, typeof(Half)),
                (NPTypeCode.Single, typeof(float)),
                (NPTypeCode.Double, typeof(double)),
                (NPTypeCode.Decimal, typeof(decimal)),
                (NPTypeCode.Complex, typeof(Complex)),
            };

            foreach (var (code, clr) in expected)
                NDArrayTensorsInterop.ToTensorElementClrType(code).Should().Be(clr, $"{code} crosses as {clr.Name}");

            NDArrayTensorsInterop.TypeCodeOf<bool>().Should().Be(NPTypeCode.Boolean);
            NDArrayTensorsInterop.TypeCodeOf<byte>().Should().Be(NPTypeCode.Byte);
            NDArrayTensorsInterop.TypeCodeOf<sbyte>().Should().Be(NPTypeCode.SByte);
            NDArrayTensorsInterop.TypeCodeOf<short>().Should().Be(NPTypeCode.Int16);
            NDArrayTensorsInterop.TypeCodeOf<ushort>().Should().Be(NPTypeCode.UInt16);
            NDArrayTensorsInterop.TypeCodeOf<int>().Should().Be(NPTypeCode.Int32);
            NDArrayTensorsInterop.TypeCodeOf<uint>().Should().Be(NPTypeCode.UInt32);
            NDArrayTensorsInterop.TypeCodeOf<long>().Should().Be(NPTypeCode.Int64);
            NDArrayTensorsInterop.TypeCodeOf<ulong>().Should().Be(NPTypeCode.UInt64);
            NDArrayTensorsInterop.TypeCodeOf<char>().Should().Be(NPTypeCode.Char);
            NDArrayTensorsInterop.TypeCodeOf<Half>().Should().Be(NPTypeCode.Half);
            NDArrayTensorsInterop.TypeCodeOf<float>().Should().Be(NPTypeCode.Single);
            NDArrayTensorsInterop.TypeCodeOf<double>().Should().Be(NPTypeCode.Double);
            NDArrayTensorsInterop.TypeCodeOf<decimal>().Should().Be(NPTypeCode.Decimal);
            NDArrayTensorsInterop.TypeCodeOf<Complex>().Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void TypeCodeOf_IsDirectional_UShortIsUInt16_NeverChar()
        {
            // ushort reads back as UInt16 (a tensor carries no "these are characters" bit); char stays char.
            NDArrayTensorsInterop.TypeCodeOf<ushort>().Should().Be(NPTypeCode.UInt16);
            NDArrayTensorsInterop.TypeCodeOf<char>().Should().Be(NPTypeCode.Char);
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Char).Should().Be(typeof(char));
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.UInt16).Should().Be(typeof(ushort));
        }

        [TestMethod]
        public void TypeCodeOf_UnknownStruct_Throws()
        {
            // Guid is unmanaged (satisfies the constraint) but is not a NumSharp element type.
            new Action(() => NDArrayTensorsInterop.TypeCodeOf<Guid>()).Should().Throw<NotSupportedException>()
                .WithMessage("*not a NumSharp element type*");
        }

        [TestMethod]
        public void WrongElementType_IsRefused_RatherThanReinterpreted()
        {
            using var f = Arange(NPTypeCode.Single, 4);
            // a float array is not an int array — reinterpreting the bytes would be silent corruption
            new Action(() => f.AsTensorSpan<int>()).Should().Throw<ArgumentException>()
                .WithMessage("*Single*crosses as Single*");
            new Action(() => f.ToTensor<int>()).Should().Throw<ArgumentException>();

            using var c = Arange(NPTypeCode.Char, 4);
            new Action(() => c.AsTensorSpan<ushort>()).Should().Throw<ArgumentException>()
                .WithMessage("*Char*request <char>*");
        }

        [TestMethod]
        public void Half_Complex_Decimal_AreNotRefused_TheyCross()
        {
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Half).Should().Be(typeof(Half));
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Complex).Should().Be(typeof(Complex));
            NDArrayTensorsInterop.ToTensorElementClrType(NPTypeCode.Decimal).Should().Be(typeof(decimal));

            using var h = Arange(NPTypeCode.Half, 3);
            using (var handle = h.AsTensorSpan<Half>()) handle.Should().NotBeNull();
            using var z = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            using (var handle = z.AsTensorSpan<Complex>()) handle.Should().NotBeNull();
            using var d = Arange(NPTypeCode.Decimal, 3);
            using (var handle = d.AsTensorSpan<decimal>()) handle.Should().NotBeNull();
        }
    }
}
