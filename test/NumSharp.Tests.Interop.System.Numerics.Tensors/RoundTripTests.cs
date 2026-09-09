using System;
using System.Numerics;
using System.Numerics.Tensors;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>Full NDArray ⇄ Tensor round trips across all 15 dtypes and non-trivial layouts.</summary>
    [TestClass]
    public class RoundTripTests : TensorsTestBase
    {
        [TestMethod]
        public void ToTensor_ToNDArray_AllDtypes_ValuesPreserved()
        {
            RT<bool>(NPTypeCode.Boolean);
            RT<byte>(NPTypeCode.Byte);
            RT<sbyte>(NPTypeCode.SByte);
            RT<short>(NPTypeCode.Int16);
            RT<ushort>(NPTypeCode.UInt16);
            RT<int>(NPTypeCode.Int32);
            RT<uint>(NPTypeCode.UInt32);
            RT<long>(NPTypeCode.Int64);
            RT<ulong>(NPTypeCode.UInt64);
            RT<char>(NPTypeCode.Char);
            RT<Half>(NPTypeCode.Half);
            RT<float>(NPTypeCode.Single);
            RT<double>(NPTypeCode.Double);
            RT<decimal>(NPTypeCode.Decimal);
            RT<Complex>(NPTypeCode.Complex);
        }

        private static void RT<T>(NPTypeCode code) where T : unmanaged
        {
            using var nd = Arange(code, 3, 4);
            Tensor<T> t = nd.ToTensor<T>();
            using var back = t.ToNDArray();
            np.array_equal(nd, back).Should().BeTrue($"{code} survives NDArray -> Tensor -> NDArray");
        }

        [TestMethod]
        public void Transposed_RoundTrip_PreservesLogicalValues()
        {
            using var a = Arange(NPTypeCode.Double, 2, 3);
            using var tv = a.transpose();                          // (3,2), strided
            Tensor<double> t = tv.ToTensor<double>();
            using var back = t.ToNDArray();                        // dense (3,2) in logical order
            using var expected = tv.copy();                        // logical (3,2)
            np.array_equal(back, expected).Should().BeTrue();
        }

        [TestMethod]
        public void AsTensorSpan_Then_ToNDArray_IsAContiguousCopyOfTheView()
        {
            using var a = Arange(NPTypeCode.Int32, 2, 3);
            using var tv = a.transpose();                          // (3,2), strided view
            using var h = tv.AsTensorSpan<int>();                  // zero-copy strided span

            using NDArray copy = h.ReadOnlySpan.ToNDArray();       // copy the span out (logical order)
            using var expected = tv.copy();
            np.array_equal(copy, expected).Should().BeTrue();
        }

        [TestMethod]
        public void Tensor_AsNDArray_Then_Back_SharesTheSameMemory()
        {
            var t = Tensor.Create(new float[] { 1, 2, 3, 4, 5, 6 }, new nint[] { 2, 3 });
            using var view = t.AsNDArray();                        // zero-copy
            using var h = view.AsTensorSpan<float>();              // re-export the view

            h.Span[new nint[] { 0, 0 }] = 111f;                    // write via the re-export
            ((float)t[new nint[] { 0, 0 }]).Should().Be(111f, "everything aliases the tensor's backing store");
            view.GetSingle(0, 0).Should().Be(111f);
        }
    }
}
