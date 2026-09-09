using System;
using System.Linq;
using System.Numerics;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    /// nd.Unsafe BCL views (Span/ReadOnlySpan/Memory/ReadOnlyMemory/Bytes/Pointer/TryGet*) and the
    /// safe .NET bridges (AsEnumerable&lt;T&gt;, iterator ToArray/CopyTo). The Unsafe views ALIAS the
    /// unmanaged buffer without rooting it; these tests keep the NDArray in scope throughout.
    /// </summary>
    [TestClass]
    public class NDArrayUnsafeInterop_Tests
    {
        // ---------------------------------------------------------------- Span / ReadOnlySpan

        [TestMethod]
        public void Span_Aliases_And_WritesThrough()
        {
            var a = np.arange(6).astype(np.float64);
            Span<double> s = a.Unsafe.Span<double>();
            s.Length.Should().Be(6);
            s[2] = 99;                                   // write through the alias
            a.GetAtIndex<double>(2).Should().Be(99);     // hit the NDArray
            a.Unsafe.Span<double>()[2].Should().Be(99);  // fresh alias sees it too
        }

        [TestMethod]
        public void Span_OffsetSlice_AliasesLogicalWindow()
        {
            // a[2:5] is C-contiguous with Storage.Address re-seated — the alias must be elements 2,3,4.
            var sl = np.arange(10).astype(np.float64)["2:5"];
            sl.Unsafe.Span<double>().ToArray().Should().Equal(2d, 3d, 4d);
        }

        [TestMethod]
        public void ReadOnlySpan_Reads()
        {
            var a = np.arange(3).astype(np.int32);
            ReadOnlySpan<int> ro = a.Unsafe.ReadOnlySpan<int>();
            ro.ToArray().Should().Equal(0, 1, 2);
        }

        [TestMethod]
        public void Span_DtypeMismatch_Throws()
            => new Action(() => { var _ = np.arange(3).astype(np.float64).Unsafe.Span<int>(); })
                .Should().Throw<ArgumentException>();

        [TestMethod]
        public void Span_NonContiguous_Throws()
        {
            var tr = np.arange(6).astype(np.float64).reshape(2, 3).T;   // F-contiguous, not C
            new Action(() => { var _ = tr.Unsafe.Span<double>(); }).Should().Throw<InvalidOperationException>();
            var strided = np.arange(6).astype(np.float64).reshape(2, 3)[":, ::2"];
            new Action(() => { var _ = strided.Unsafe.Span<double>(); }).Should().Throw<InvalidOperationException>();
        }

        // ---------------------------------------------------------------- Memory / ReadOnlyMemory

        [TestMethod]
        public void Memory_Aliases_And_WritesThrough()
        {
            var a = np.arange(4).astype(np.float64);
            Memory<double> m = a.Unsafe.Memory<double>();
            m.Length.Should().Be(4);
            m.Span[3] = -7;
            a.GetAtIndex<double>(3).Should().Be(-7);
        }

        [TestMethod]
        public void ReadOnlyMemory_Reads()
        {
            var a = np.arange(4).astype(np.int64);
            ReadOnlyMemory<long> rm = a.Unsafe.ReadOnlyMemory<long>();
            rm.Span.ToArray().Should().Equal(0L, 1L, 2L, 3L);
        }

        // ---------------------------------------------------------------- Bytes

        [TestMethod]
        public void Bytes_Is_RawWindow_LengthSizeTimesItemsize()
        {
            var a = np.arange(4).astype(np.int32);       // 4 * 4 = 16 bytes
            a.Unsafe.Bytes().Length.Should().Be(16);
            a.Unsafe.ReadOnlyBytes().Length.Should().Be(16);
            // little-endian int32 { 0,1,2,3 }: byte 4 is the low byte of element 1 == 1
            a.Unsafe.Bytes()[4].Should().Be(1);
        }

        // ---------------------------------------------------------------- Pointer (any layout)

        [TestMethod]
        public unsafe void Pointer_Works_ForAnyLayout()
        {
            var a = np.arange(3).astype(np.float64);
            double* p = a.Unsafe.Pointer<double>();
            p[0].Should().Be(0);
            p[1].Should().Be(1);

            // transposed (non-contiguous): the pointer to logical element 0 is still valid.
            var t = np.arange(6).astype(np.float64).reshape(2, 3).T;
            double* pt = t.Unsafe.Pointer<double>();
            pt[0].Should().Be(0);   // element 0 of the buffer

            // dtype mismatch is refused on the typed pointer too
            new Action(() => { var _ = a.Unsafe.Pointer<int>(); }).Should().Throw<ArgumentException>();
        }

        // ---------------------------------------------------------------- TryGet*

        [TestMethod]
        public void TryGetSpan_TrueForContiguous_FalseOtherwise()
        {
            var a = np.arange(6).astype(np.float64);
            a.Unsafe.TryGetSpan<double>(out var s).Should().BeTrue();
            s.Length.Should().Be(6);

            var tr = a.reshape(2, 3).T;
            tr.Unsafe.TryGetSpan<double>(out var s2).Should().BeFalse();
            s2.Length.Should().Be(0);

            a.Unsafe.TryGetSpan<int>(out _).Should().BeFalse();   // wrong dtype
        }

        [TestMethod]
        public void TryGetMemory_TrueForContiguous_FalseOtherwise()
        {
            var a = np.arange(6).astype(np.float64);
            a.Unsafe.TryGetMemory<double>(out var m).Should().BeTrue();
            m.Length.Should().Be(6);
            a.reshape(2, 3).T.Unsafe.TryGetMemory<double>(out _).Should().BeFalse();
        }

        // ---------------------------------------------------------------- dtype coverage (all 15)

        private static void AssertSpanAliases<T>(NDArray a, T probe) where T : unmanaged
        {
            var s = a.Unsafe.Span<T>();
            s[0] = probe;
            Assert.AreEqual(probe, a.Unsafe.Span<T>()[0]);      // fresh alias sees the write
            Assert.AreEqual(probe, a.GetAtIndex<T>(0));         // NDArray sees it too
            // TryGet path agrees
            Assert.IsTrue(a.Unsafe.TryGetSpan<T>(out var s2));
            Assert.AreEqual(probe, s2[0]);
        }

        [TestMethod]
        public void Span_Covers_All15Dtypes()
        {
            AssertSpanAliases<bool>(np.array(new[] { false, true }), true);
            AssertSpanAliases<byte>(np.array(new byte[] { 1, 2 }), (byte)200);
            AssertSpanAliases<sbyte>(np.array(new sbyte[] { 1, 2 }), (sbyte)-5);
            AssertSpanAliases<short>(np.array(new short[] { 1, 2 }), (short)-30000);
            AssertSpanAliases<ushort>(np.array(new ushort[] { 1, 2 }), (ushort)60000);
            AssertSpanAliases<int>(np.array(new[] { 1, 2 }), int.MinValue);
            AssertSpanAliases<uint>(np.array(new uint[] { 1, 2 }), uint.MaxValue);
            AssertSpanAliases<long>(np.array(new long[] { 1, 2 }), long.MinValue);
            AssertSpanAliases<ulong>(np.array(new ulong[] { 1, 2 }), ulong.MaxValue);
            AssertSpanAliases<char>(np.array(new[] { 'a', 'b' }), 'Z');
            AssertSpanAliases<Half>(np.array(new[] { (Half)1, (Half)2 }), (Half)3.5);
            AssertSpanAliases<float>(np.array(new[] { 1f, 2f }), 3.25f);
            AssertSpanAliases<double>(np.array(new[] { 1d, 2d }), 3.5);
            AssertSpanAliases<decimal>(np.array(new[] { 1m, 2m }), 3.75m);
            AssertSpanAliases<Complex>(np.array(new[] { new Complex(1, 1), new Complex(2, 2) }), new Complex(3, -4));
        }

        // ---------------------------------------------------------------- AsEnumerable<T> (LINQ)

        [TestMethod]
        public void AsEnumerable_LINQ_AnyLayout_COrder()
        {
            var m = np.arange(6).astype(np.float64).reshape(2, 3);
            m.AsEnumerable<double>().Sum().Should().Be(15);
            m.T.AsEnumerable<double>().Should().Equal(0d, 3d, 1d, 4d, 2d, 5d);     // logical C-order
            m.AsEnumerable<double>().Where(x => x > 2).Should().Equal(3d, 4d, 5d);
        }

        [TestMethod]
        public void AsEnumerable_DtypeMismatch_ThrowsEagerly()
            // The dtype check fires at call time, not on first enumeration.
            => new Action(() => np.arange(3).astype(np.float64).AsEnumerable<int>())
                .Should().Throw<ArgumentException>();

        // ---------------------------------------------------------------- iterator -> .NET buffer

        [TestMethod]
        public void RefIter_ToArray_And_CopyTo_MemoryOrder()
        {
            var b = np.arange(4).astype(np.float64).reshape(2, 2).T;    // non-contiguous
            np.nditer<double>(b).ToArray().Should().Equal(0d, 1d, 2d, 3d);  // memory ('K') order

            var dst = new double[4];
            np.nditer<double>(b).CopyTo(dst).Should().Be(4);
            dst.Should().Equal(0d, 1d, 2d, 3d);
        }

        [TestMethod]
        public void FlatIter_ToArray_And_CopyTo_COrder()
        {
            var b = np.arange(4).astype(np.float64).reshape(2, 2).T;    // non-contiguous
            np.flat<double>(b).ToArray().Should().Equal(0d, 2d, 1d, 3d);    // logical C-order

            var dst = new double[4];
            np.flat<double>(b).CopyTo(dst).Should().Be(4);
            dst.Should().Equal(0d, 2d, 1d, 3d);
        }

        [TestMethod]
        public void ChunkIter_ToArray_And_CopyTo()
        {
            var a = np.arange(6).astype(np.float64);
            np.nditer_chunks<double>(a).ToArray().Should().Equal(0d, 1d, 2d, 3d, 4d, 5d);

            var dst = new double[6];
            np.nditer_chunks<double>(a).CopyTo(dst).Should().Be(6);
            dst.Should().Equal(0d, 1d, 2d, 3d, 4d, 5d);
        }

        [TestMethod]
        public void CopyTo_TooShort_Throws()
        {
            var a = np.arange(6).astype(np.float64);
            new Action(() => np.flat<double>(a).CopyTo(new double[3]))
                .Should().Throw<ArgumentException>();
        }
    }
}
