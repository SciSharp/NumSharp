using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>System.Numerics.Tensors → NumSharp: <c>ToNDArray</c> (copy) and <c>AsNDArray</c> (zero-copy view).</summary>
    [TestClass]
    public class ImportTests : TensorsTestBase
    {
        [TestMethod]
        public void ToNDArray_DenseTensor_CopiesValuesAndShape()
        {
            var t = Tensor.Create(new float[] { 1, 2, 3, 4, 5, 6 }, new nint[] { 2, 3 });
            using var nd = t.ToNDArray();

            nd.Shape.NDim.Should().Be(2);
            nd.shape[0].Should().Be(2);
            nd.shape[1].Should().Be(3);
            nd.typecode.Should().Be(NPTypeCode.Single);
            nd.GetSingle(0, 0).Should().Be(1f);
            nd.GetSingle(1, 2).Should().Be(6f);
        }

        [TestMethod]
        public void ToNDArray_StridedTensor_ReadInLogicalOrder()
        {
            // transposed 3x3 via strides [1,3]: t[i,j] = arr[i + 3j]
            var t = Tensor.Create(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, new nint[] { 3, 3 }, new nint[] { 1, 3 });
            t.IsDense.Should().BeFalse();
            using var nd = t.ToNDArray();

            nd.GetInt32(0, 1).Should().Be(3);
            nd.GetInt32(2, 0).Should().Be(2);
            nd.GetInt32(2, 2).Should().Be(8);
            nd.Shape.IsContiguous.Should().BeTrue("the copy is a fresh C-contiguous array");
        }

        [TestMethod]
        public void ToNDArray_Copy_IsIndependent()
        {
            var t = Tensor.Create(new double[] { 1, 2, 3 }, new nint[] { 3 });
            using var nd = t.ToNDArray();
            t[new nint[] { 0 }] = 999.0;                     // mutate the tensor after the copy
            nd.GetDouble(0).Should().Be(1.0, "ToNDArray is an owning copy");
        }

        [TestMethod]
        public void AsNDArray_DenseTensor_SharesMemory_WriteThrough()
        {
            var t = Tensor.Create(new double[] { 0, 0, 0, 0 }, new nint[] { 2, 2 });
            using (var nd = t.AsNDArray())
            {
                nd.GetDouble(0, 0).Should().Be(0.0);
                nd.SetDouble(42.0, 0, 1);                    // ndarray -> tensor backing
                ((double)t[new nint[] { 0, 1 }]).Should().Be(42.0);

                t[new nint[] { 1, 0 }] = -3.0;               // tensor -> ndarray
                nd.GetDouble(1, 0).Should().Be(-3.0);
            }
        }

        [TestMethod]
        public void AsNDArray_StridedTensor_SharesMemory_NoDensifyingCopy()
        {
            var t = Tensor.Create(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 }, new nint[] { 3, 3 }, new nint[] { 1, 3 });
            using var nd = t.AsNDArray();

            nd.shape[0].Should().Be(3);
            nd.shape[1].Should().Be(3);
            nd.GetInt32(0, 1).Should().Be(3);
            nd.GetInt32(2, 0).Should().Be(2);

            nd.SetInt32(555, 0, 1);                          // writes through the shared strided memory
            ((int)t[new nint[] { 0, 1 }]).Should().Be(555);
        }

        [TestMethod]
        public void AsNDArray_Empty_IsEmptyArray()
        {
            var t = Tensor.Create(Array.Empty<float>(), new nint[] { 0 });
            using var nd = t.AsNDArray();
            nd.size.Should().Be(0);
        }

        [TestMethod]
        public void ToNDArray_FromReadOnlyTensorSpan_Copies()
        {
            var t = Tensor.Create(new float[] { 10, 20, 30, 40 }, new nint[] { 2, 2 });
            ReadOnlyTensorSpan<float> ro = t.AsReadOnlyTensorSpan();
            using NDArray nd = ro.ToNDArray();

            nd.shape[0].Should().Be(2);
            nd.GetSingle(1, 1).Should().Be(40f);
        }

        [TestMethod]
        public void ToNDArray_FromTensorSpan_Copies()
        {
            var t = Tensor.Create(new int[] { 7, 8, 9 }, new nint[] { 3 });
            TensorSpan<int> s = t.AsTensorSpan();
            using NDArray nd = s.ToNDArray();
            nd.GetInt32(2).Should().Be(9);
        }

        [TestMethod]
        public void ToNDArray_And_AsNDArray_Null_Throw()
        {
            new Action(() => NDArrayTensorsInterop.ToNDArray<float>((Tensor<float>)null)).Should().Throw<ArgumentNullException>();
            new Action(() => NDArrayTensorsInterop.AsNDArray<float>((Tensor<float>)null)).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void ToNDArray_AllDtypes_RoundTripBytes()
        {
            AssertImportBytes<bool>(NPTypeCode.Boolean, new bool[] { true, false, true, true, false, true });
            AssertImportBytes<byte>(NPTypeCode.Byte, new byte[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<sbyte>(NPTypeCode.SByte, new sbyte[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<short>(NPTypeCode.Int16, new short[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<ushort>(NPTypeCode.UInt16, new ushort[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<int>(NPTypeCode.Int32, new int[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<uint>(NPTypeCode.UInt32, new uint[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<long>(NPTypeCode.Int64, new long[] { -1, 2, -3, 4, -5, 6 });
            AssertImportBytes<ulong>(NPTypeCode.UInt64, new ulong[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<char>(NPTypeCode.Char, new char[] { 'a', 'b', 'c', 'd', 'e', 'f' });
            AssertImportBytes<Half>(NPTypeCode.Half, new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5, (Half)6 });
            AssertImportBytes<float>(NPTypeCode.Single, new float[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<double>(NPTypeCode.Double, new double[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<decimal>(NPTypeCode.Decimal, new decimal[] { 1, 2, 3, 4, 5, 6 });
            AssertImportBytes<Complex>(NPTypeCode.Complex, new[] { new Complex(1, 2), new Complex(3, 4), new Complex(5, 6), new Complex(7, 8), new Complex(9, 10), new Complex(11, 12) });
        }

        private static void AssertImportBytes<T>(NPTypeCode code, T[] data) where T : unmanaged
        {
            var t = Tensor.Create(data, new nint[] { 2, 3 });
            using NDArray nd = t.ToNDArray();
            nd.typecode.Should().Be(code);
            nd.shape[0].Should().Be(2);
            nd.shape[1].Should().Be(3);
            BytesOf(nd).Should().Equal(MemoryMarshal.AsBytes<T>(data).ToArray(), $"{code} imports byte-exactly");
        }
    }
}
