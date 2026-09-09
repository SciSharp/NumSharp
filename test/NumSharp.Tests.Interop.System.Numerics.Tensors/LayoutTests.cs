using System;
using System.Numerics.Tensors;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Layout coverage — the capability that sets this bridge apart from the row-major-only ONNX bridge:
    ///     a <see cref="TensorSpan{T}"/> carries explicit strides, so contiguous, offset, transposed, strided
    ///     and broadcast NumSharp views all share zero-copy. Only a negative-stride view is refused.
    /// </summary>
    [TestClass]
    public class LayoutTests : TensorsTestBase
    {
        [TestMethod]
        public void Transposed_SharesMemory_LogicalValuesMatch()
        {
            using var a = Arange(NPTypeCode.Double, 2, 3);    // [[0,1,2],[3,4,5]]
            using var t = a.transpose();                      // (3,2)
            using var h = t.AsTensorSpan<double>();
            var s = h.ReadOnlySpan;

            ((long)s.Lengths[0]).Should().Be(3);
            ((long)s.Lengths[1]).Should().Be(2);
            s.IsDense.Should().BeFalse();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    s[new nint[] { i, j }].Should().Be(a.GetDouble(j, i), $"t[{i},{j}] == a[{j},{i}]");
        }

        [TestMethod]
        public void SteppedSlice_SharesMemory()
        {
            using var a = Arange(NPTypeCode.Double, 10);
            using var s2 = a["::2"];                          // [0,2,4,6,8]
            using var h = s2.AsTensorSpan<double>();
            var s = h.ReadOnlySpan;

            ((long)s.FlattenedLength).Should().Be(5);
            for (int k = 0; k < 5; k++)
                s[new nint[] { k }].Should().Be(2.0 * k);
        }

        [TestMethod]
        public void Broadcast_ReadOnly_SharesMemory_ButSpanThrows()
        {
            using var col = Arange(NPTypeCode.Double, 3).reshape(3, 1);
            using var b = np.broadcast_to(col, new Shape(3, 4));   // stride-0 broadcast, read-only
            using var h = b.AsTensorSpan<double>();

            var ro = h.ReadOnlySpan;
            ro.IsDense.Should().BeFalse();
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ro[new nint[] { i, j }].Should().Be(i, "every column of a broadcast row repeats");

            h.IsWriteable.Should().BeFalse();
            new Action(() => { _ = h.Span; }).Should().Throw<InvalidOperationException>()
                .WithMessage("*broadcast*read-only*");
        }

        [TestMethod]
        public void FortranContiguous_SharesMemory()
        {
            using var a = Arange(NPTypeCode.Int32, 3, 4);
            using var f = np.asfortranarray(a);               // column-major, non-negative strides
            using var h = f.AsTensorSpan<int>();
            var s = h.ReadOnlySpan;

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    s[new nint[] { i, j }].Should().Be(a.GetInt32(i, j));
        }

        [TestMethod]
        public void ThreeD_Transposed_SharesMemory()
        {
            using var a = Arange(NPTypeCode.Int32, 2, 3, 4);
            using var t = a.transpose(new int[] { 2, 1, 0 }); // (4,3,2), strided
            using var h = t.AsTensorSpan<int>();
            var s = h.ReadOnlySpan;

            ((long)s.Lengths[0]).Should().Be(4);
            s[new nint[] { 3, 2, 1 }].Should().Be(a.GetInt32(1, 2, 3), "t[k,j,i] == a[i,j,k]");
            s[new nint[] { 0, 0, 1 }].Should().Be(a.GetInt32(1, 0, 0));
        }

        [TestMethod]
        public void UnitAxis_SharesMemory()
        {
            // a genuine length-1 axis: NumSharp gives it a nonzero stride, which the tensor ctor rejects unless
            // normalized to 0 — this pins that normalization.
            using (var a = Arange(NPTypeCode.Single, 3, 1))
            using (var h = a.AsTensorSpan<float>())
            {
                var s = h.ReadOnlySpan;
                ((long)s.Lengths[0]).Should().Be(3);
                ((long)s.Lengths[1]).Should().Be(1);
                for (int i = 0; i < 3; i++)
                    s[new nint[] { i, 0 }].Should().Be(a.GetSingle(i, 0));
            }

            using (var b = Arange(NPTypeCode.Double, 1, 4, 1))   // multiple unit axes
            using (var h = b.AsTensorSpan<double>())
            {
                var s = h.ReadOnlySpan;
                for (int j = 0; j < 4; j++)
                    s[new nint[] { 0, j, 0 }].Should().Be(b.GetDouble(0, j, 0));
            }
        }

        [TestMethod]
        public void NegativeStride_Refused_WithPointerToTheFix()
        {
            using var a = Arange(NPTypeCode.Single, 5);
            using (var rev = a["::-1"])
                new Action(() => rev.AsTensorSpan<float>()).Should().Throw<InvalidOperationException>()
                    .WithMessage("*negative-stride*ascontiguousarray*");

            using var m = Arange(NPTypeCode.Single, 3, 4);
            using (var revCols = m[":, ::-1"])
                new Action(() => revCols.AsTensorSpan<float>()).Should().Throw<InvalidOperationException>();
        }

        [TestMethod]
        public void NegativeStride_MaterializeThenExport_Works()
        {
            using var a = Arange(NPTypeCode.Single, 5);
            using var rev = a["::-1"];
            using var dense = np.ascontiguousarray(rev);      // materialize
            using var h = dense.AsTensorSpan<float>();
            h.ReadOnlySpan[new nint[] { 0 }].Should().Be(4f);
        }
    }
}
