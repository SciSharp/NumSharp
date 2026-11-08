using System;
using System.Linq;
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

        // ---- edge shapes: the 0-d scalar and the empty array -------------------------------------------

        /// <summary>
        ///     A 0-d NumSharp scalar copies to a single-element vector tensor [1] — the same shape
        ///     <c>AsTensorSpan</c> gives it, since the BCL has no rank 0 — and keeps its value. Regression: it passed
        ///     EMPTY lengths to <c>Tensor.Create</c>, which builds a rank-1 LENGTH-0 tensor, so the value was
        ///     silently dropped.
        /// </summary>
        [TestMethod]
        public void ToTensor_Scalar_CrossesAsSingleElementVector_KeepingItsValue()
        {
            using NDArray scalar = NDArray.Scalar(7.5);
            scalar.ndim.Should().Be(0);

            Tensor<double> t = scalar.ToTensor<double>();
            t.Rank.Should().Be(1);
            ((long)t.Lengths[0]).Should().Be(1);
            t[new nint[] { 0 }].Should().Be(7.5);

            using NDArray back = t.ToNDArray();
            back.shape.Should().Equal(1L);
            back.GetDouble(0).Should().Be(7.5);
        }

        /// <summary>
        ///     An empty array exports with its shape intact and all-zero strides — a span the BCL can flatten, fill and
        ///     reduce, and that imports back to the same shape. Regression: it exported as
        ///     <c>TensorSpan&lt;T&gt;.Empty</c>, which is rank 0 (a (3,0,4) shape was gone, <c>Lengths</c> was empty)
        ///     and unusable: the BCL's own <c>FlattenTo</c> and <c>Tensor.Sum</c> threw IndexOutOfRangeException on it,
        ///     and so did <c>span.ToNDArray()</c>.
        /// </summary>
        [TestMethod]
        public void AsTensorSpan_Empty_KeepsItsShape_AndTheBclCanUseIt()
        {
            long[][] shapes = { new long[] { 0 }, new long[] { 3, 0 }, new long[] { 0, 3 }, new long[] { 2, 0, 4 } };
            foreach (long[] dims in shapes)
            {
                using NDArray empty = np.zeros(new Shape(dims), np.float32);
                using var h = empty.AsTensorSpan<float>();

                h.Lengths.Select(x => (long)x).Should().Equal(dims);
                h.Strides.Should().OnlyContain(s => s == 0, "the BCL accepts a zero-size span only with all-zero strides");

                ReadOnlyTensorSpan<float> ro = h.ReadOnlySpan;
                ro.Rank.Should().Be(dims.Length);
                ro.FlattenTo(Span<float>.Empty);
                Tensor.Sum(ro).Should().Be(0f);
                h.Span.Fill(1f);                                   // an empty writeable array gives a usable mutable span

                using NDArray back = ro.ToNDArray();
                back.shape.Should().Equal(dims);
            }
        }

        /// <summary>
        ///     The two edge-shape fixes hold for all 15 element types: a 0-d array keeps its value through
        ///     <c>ToTensor</c>, and an empty (2,0,3) array survives <c>AsTensorSpan</c> → <c>span.ToNDArray()</c> with its
        ///     shape and dtype.
        /// </summary>
        [TestMethod]
        public void EdgeShapes_AllDtypes_CrossIntact()
        {
            AssertEdgeShapes<bool>(NPTypeCode.Boolean);
            AssertEdgeShapes<byte>(NPTypeCode.Byte);
            AssertEdgeShapes<sbyte>(NPTypeCode.SByte);
            AssertEdgeShapes<short>(NPTypeCode.Int16);
            AssertEdgeShapes<ushort>(NPTypeCode.UInt16);
            AssertEdgeShapes<int>(NPTypeCode.Int32);
            AssertEdgeShapes<uint>(NPTypeCode.UInt32);
            AssertEdgeShapes<long>(NPTypeCode.Int64);
            AssertEdgeShapes<ulong>(NPTypeCode.UInt64);
            AssertEdgeShapes<char>(NPTypeCode.Char);
            AssertEdgeShapes<Half>(NPTypeCode.Half);
            AssertEdgeShapes<float>(NPTypeCode.Single);
            AssertEdgeShapes<double>(NPTypeCode.Double);
            AssertEdgeShapes<decimal>(NPTypeCode.Decimal);
            AssertEdgeShapes<Complex>(NPTypeCode.Complex);
        }

        /// <summary>
        ///     Checks one dtype: the 0-d scalar's bytes come through <c>ToTensor</c> as a one-element tensor, and an
        ///     empty array's shape and dtype come back through a zero-copy span.
        /// </summary>
        /// <typeparam name="T">The CLR element type of <paramref name="code"/>.</typeparam>
        /// <param name="code">The NumSharp dtype under test.</param>
        private static void AssertEdgeShapes<T>(NPTypeCode code) where T : unmanaged
        {
            using (NDArray seven = np.array(7.0))
            using (NDArray scalar = seven.astype(code, copy: true))
            {
                Tensor<T> t = scalar.ToTensor<T>();
                ((long)t.FlattenedLength).Should().Be(1, $"{code}: a 0-d array keeps its one element");
                var value = new T[1];
                t.FlattenTo(value);
                MemoryMarshal.AsBytes<T>(value).ToArray().Should().Equal(BytesOf(scalar), $"{code}: the 0-d value is kept");
            }

            using (NDArray empty = Arange(code, 2, 0, 3))
            using (var h = empty.AsTensorSpan<T>())
            using (NDArray back = h.ReadOnlySpan.ToNDArray())
            {
                back.shape.Should().Equal(new long[] { 2, 0, 3 }, $"{code}: an empty array keeps its shape");
                back.typecode.Should().Be(code);
            }
        }
    }
}
