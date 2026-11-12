using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>The dtype maps both ways: the ten shared dtypes, NEP 50 for the five torch lacks, exact widening for two NumSharp lacks.</summary>
    [TestClass]
    public class DtypeMapTests : TorchTestBase
    {
        /// <summary>The ten shared dtypes map to their own counterparts and back.</summary>
        [TestMethod]
        public void SharedDtypes_MapBothWays()
        {
            (NPTypeCode code, ScalarType st)[] pairs =
            {
                (NPTypeCode.Boolean, ScalarType.Bool), (NPTypeCode.Byte, ScalarType.Byte), (NPTypeCode.SByte, ScalarType.Int8),
                (NPTypeCode.Int16, ScalarType.Int16), (NPTypeCode.Int32, ScalarType.Int32), (NPTypeCode.Int64, ScalarType.Int64),
                (NPTypeCode.Half, ScalarType.Float16), (NPTypeCode.Single, ScalarType.Float32), (NPTypeCode.Double, ScalarType.Float64),
                (NPTypeCode.Complex, ScalarType.ComplexFloat64),
            };
            foreach (var (code, st) in pairs)
            {
                NDArrayTorchInterop.ToScalarType(code).Should().Be(st, $"{code} crosses as its counterpart");
                NDArrayTorchInterop.ToDType(st).GetTypeCode().Should().Be(code, $"{st} imports as its counterpart");
            }
        }

        /// <summary>The five dtypes torch lacks follow NumSharp's own NEP 50 promotion with int8; decimal goes to float64.</summary>
        [TestMethod]
        public void ConvertedDtypes_FollowNep50()
        {
            NDArrayTorchInterop.ToScalarType(np.uint16).Should().Be(ScalarType.Int32);
            NDArrayTorchInterop.ToScalarType(np.uint32).Should().Be(ScalarType.Int64);
            NDArrayTorchInterop.ToScalarType(np.uint64).Should().Be(ScalarType.Float64, "NEP 50: promote_types(uint64, int8) is float64");
            NDArrayTorchInterop.ToScalarType(NPTypeCode.Char).Should().Be(ScalarType.Int32);
            NDArrayTorchInterop.ToScalarType(NPTypeCode.Decimal).Should().Be(ScalarType.Float64);

            // The rule, not a table: each answer is NumSharp's own promote_types(dtype, int8).
            foreach (NPTypeCode code in new[] { NPTypeCode.UInt16, NPTypeCode.UInt32, NPTypeCode.UInt64, NPTypeCode.Char })
            {
                NPTypeCode promoted = np.promote_types(code, NPTypeCode.SByte).GetTypeCode();
                NDArrayTorchInterop.ToDType(NDArrayTorchInterop.ToScalarType(code)).GetTypeCode().Should().Be(promoted, code.ToString());
            }
        }

        /// <summary>Text has no torch counterpart in any form.</summary>
        [TestMethod]
        public void Text_IsRefused()
        {
            Action act = () => NDArrayTorchInterop.ToScalarType(np.dtype("U5"));
            act.Should().Throw<NotSupportedException>().WithMessage("*no string tensors*");
            Action nul = () => NDArrayTorchInterop.ToScalarType(null);
            nul.Should().Throw<ArgumentNullException>();
        }

        /// <summary>complex64 and bfloat16 import widened; libtorch's unsigned codes 27-29 import as NumSharp's unsigned dtypes.</summary>
        [TestMethod]
        public void TorchOnlyDtypes_ImportWidenedOrAsUnsigned()
        {
            NDArrayTorchInterop.ToDType(ScalarType.ComplexFloat32).GetTypeCode().Should().Be(NPTypeCode.Complex);
            NDArrayTorchInterop.ToDType(ScalarType.BFloat16).GetTypeCode().Should().Be(NPTypeCode.Single);
            NDArrayTorchInterop.ToDType((ScalarType)27).GetTypeCode().Should().Be(NPTypeCode.UInt16);
            NDArrayTorchInterop.ToDType((ScalarType)28).GetTypeCode().Should().Be(NPTypeCode.UInt32);
            NDArrayTorchInterop.ToDType((ScalarType)29).GetTypeCode().Should().Be(NPTypeCode.UInt64);
        }

        /// <summary>The quantized dtypes (c10 12-14) and those NumSharp cannot hold even widened are refused, naming the way out.</summary>
        [TestMethod]
        public void QuantizedAndUnknownDtypes_AreRefused()
        {
            foreach (int q in new[] { 12, 13, 14 })
            {
                Action act = () => NDArrayTorchInterop.ToDType((ScalarType)q);
                act.Should().Throw<NotSupportedException>().WithMessage("*quantized*dequantize()*int_repr()*");
            }
            Action half = () => NDArrayTorchInterop.ToDType((ScalarType)8);
            half.Should().Throw<NotSupportedException>().WithMessage("*ComplexHalf*");
            Action float8 = () => NDArrayTorchInterop.ToDType((ScalarType)24);
            float8.Should().Throw<NotSupportedException>();
        }
    }
}
