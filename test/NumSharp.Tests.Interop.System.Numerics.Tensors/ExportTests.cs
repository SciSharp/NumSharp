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
    /// <summary>NumSharp → System.Numerics.Tensors: <c>AsTensorSpan</c> (zero-copy) and <c>ToTensor</c> (copy).</summary>
    [TestClass]
    public class ExportTests : TensorsTestBase
    {
        [TestMethod]
        public void AsTensorSpan_Contiguous_SharesMemory_BothDirections()
        {
            using var a = Arange(NPTypeCode.Single, 2, 3);   // [[0,1,2],[3,4,5]]
            using var h = a.AsTensorSpan<float>();

            TensorSpan<float> s = h.Span;
            s.Lengths.Length.Should().Be(2);
            ((long)s.Lengths[0]).Should().Be(2);
            ((long)s.Lengths[1]).Should().Be(3);
            s[new nint[] { 1, 2 }].Should().Be(5f, "span reads the NDArray's memory");

            s[new nint[] { 0, 0 }] = 99f;                    // span -> ndarray
            a.GetSingle(0, 0).Should().Be(99f);

            a.SetSingle(-7f, 1, 1);                          // ndarray -> span
            h.Span[new nint[] { 1, 1 }].Should().Be(-7f);
        }

        [TestMethod]
        public void AsTensorSpan_OffsetSlice_SharesExactWindow()
        {
            using var a = Arange(NPTypeCode.Int32, 8);       // 0..7
            using var window = a["2:5"];                     // contiguous view at offset 2 -> [2,3,4]
            using var h = window.AsTensorSpan<int>();

            var ro = h.ReadOnlySpan;
            ((long)ro.FlattenedLength).Should().Be(3);
            ro[new nint[] { 0 }].Should().Be(2);
            ro[new nint[] { 2 }].Should().Be(4);

            h.Span[new nint[] { 0 }] = 100;                  // writes a[2]
            a.GetInt32(2).Should().Be(100);
        }

        [TestMethod]
        public void AsTensorSpan_Scalar_CrossesAsRank1SingleElement()
        {
            // System.Numerics.Tensors' pointer ctor cannot express rank 0 (empty lengths infer rank-1), so a
            // NumSharp 0-d scalar crosses as a single-element vector [1] — documented, and the value is intact.
            using var scalar = np.array(7.0).astype(NPTypeCode.Double, copy: true);
            scalar.Shape.NDim.Should().Be(0, "NumSharp builds a genuine 0-d scalar");
            using var h = scalar.AsTensorSpan<double>();
            h.ReadOnlySpan.Rank.Should().Be(1);
            ((long)h.ReadOnlySpan.FlattenedLength).Should().Be(1);
            h.ReadOnlySpan[new nint[] { 0 }].Should().Be(7.0);
        }

        [TestMethod]
        public void AsTensorSpan_Empty_HasZeroLength()
        {
            using var empty = np.arange(0).astype(NPTypeCode.Single, copy: true);
            using var he = empty.AsTensorSpan<float>();
            ((long)he.ReadOnlySpan.FlattenedLength).Should().Be(0);
        }

        [TestMethod]
        public void AsTensorSpan_Null_Throws()
        {
            new Action(() => NDArrayTensorsInterop.AsTensorSpan<float>(null)).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void AsTensorSpan_AllDtypes_ShareBytesExactly()
        {
            AssertSharesBytes<bool>(NPTypeCode.Boolean);
            AssertSharesBytes<byte>(NPTypeCode.Byte);
            AssertSharesBytes<sbyte>(NPTypeCode.SByte);
            AssertSharesBytes<short>(NPTypeCode.Int16);
            AssertSharesBytes<ushort>(NPTypeCode.UInt16);
            AssertSharesBytes<int>(NPTypeCode.Int32);
            AssertSharesBytes<uint>(NPTypeCode.UInt32);
            AssertSharesBytes<long>(NPTypeCode.Int64);
            AssertSharesBytes<ulong>(NPTypeCode.UInt64);
            AssertSharesBytes<char>(NPTypeCode.Char);
            AssertSharesBytes<Half>(NPTypeCode.Half);
            AssertSharesBytes<float>(NPTypeCode.Single);
            AssertSharesBytes<double>(NPTypeCode.Double);
            AssertSharesBytes<decimal>(NPTypeCode.Decimal);
            AssertSharesBytes<Complex>(NPTypeCode.Complex);
        }

        private static void AssertSharesBytes<T>(NPTypeCode code) where T : unmanaged
        {
            using var nd = Arange(code, 2, 3);
            using var h = nd.AsTensorSpan<T>();
            var flat = new T[6];
            h.ReadOnlySpan.FlattenTo(flat);
            MemoryMarshal.AsBytes<T>(flat).ToArray().Should().Equal(BytesOf(nd), $"{code} shares bytes 1:1");
        }

        [TestMethod]
        public void ToTensor_Copy_IsIndependent()
        {
            using var a = Arange(NPTypeCode.Int64, 4);
            Tensor<long> t = a.ToTensor<long>();
            t.IsDense.Should().BeTrue();

            a.SetInt64(777, 0);                              // mutating the source must not touch the copy
            ((long)t[new nint[] { 0 }]).Should().Be(0);
        }

        [TestMethod]
        public void ToTensor_AnyLayout_ReadInLogicalOrder()
        {
            using var a = Arange(NPTypeCode.Int32, 2, 3);    // [[0,1,2],[3,4,5]]
            using var tView = a.transpose();                 // (3,2) strided
            Tensor<int> t = tView.ToTensor<int>();

            t.IsDense.Should().BeTrue("the copy is dense C-order");
            ((long)t.Lengths[0]).Should().Be(3);
            ((long)t.Lengths[1]).Should().Be(2);
            ((int)t[new nint[] { 0, 1 }]).Should().Be(3, "transposed logical value");
            ((int)t[new nint[] { 2, 1 }]).Should().Be(5);
        }

        [TestMethod]
        public void ToTensor_NegativeStride_CopiesFine()
        {
            using var a = Arange(NPTypeCode.Single, 5);
            using var rev = a["::-1"];                        // [4,3,2,1,0] — AsTensorSpan would refuse this
            Tensor<float> t = rev.ToTensor<float>();
            ((float)t[new nint[] { 0 }]).Should().Be(4f);
            ((float)t[new nint[] { 4 }]).Should().Be(0f);
        }

        [TestMethod]
        public void ToTensor_Null_Throws()
        {
            new Action(() => NDArrayTensorsInterop.ToTensor<float>(null)).Should().Throw<ArgumentNullException>();
        }
    }
}
