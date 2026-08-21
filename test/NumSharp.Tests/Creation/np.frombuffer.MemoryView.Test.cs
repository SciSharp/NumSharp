using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     <c>np.frombuffer(np.MemoryView)</c> parity gate — the consumer side of <c>ndarray.data</c>.
    ///     Every expectation was probed against NumPy 2.4.2's <c>np.frombuffer(memoryview)</c>: it is
    ///     ZERO-COPY (shares the buffer's memory, writes through), reinterprets the bytes as the given
    ///     dtype, requires a C-contiguous buffer, and applies offset(bytes)/count(elements) with the same
    ///     validation as the byte[] path.
    /// </summary>
    [TestClass]
    public class NpFromBufferMemoryViewTest
    {
        [TestMethod]
        public void RoundTrip_ZeroCopy_WriteThrough()
        {
            var a = np.arange(6).astype(np.int32);
            var fb = np.frombuffer(a.data, np.int32);       // numpy: np.frombuffer(a.data, np.int32)
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, fb.ToArray<int>());
            Assert.IsTrue(fb.Shape.IsWriteable);

            fb[0] = 999;                                     // shares memory -> writes through to a
            Assert.AreEqual(999, Convert.ToInt32(a.GetValue(0)));
        }

        [TestMethod]
        public void DefaultDtype_Is_Float64()
        {
            var f = np.zeros(new Shape(3), np.float64);
            Assert.AreEqual(typeof(double), np.frombuffer(f.data).dtype);
        }

        [TestMethod]
        public void Reinterpret_Int32_To_Float32()
        {
            // 1065353216 == 0x3F800000 == 1.0f  (numpy: np.frombuffer(int32(...).data, np.float32) == [1.])
            var r = np.frombuffer(np.array(new[] { 1065353216 }).data, np.float32);
            CollectionAssert.AreEqual(new[] { 1.0f }, r.ToArray<float>());
        }

        [TestMethod]
        public void Reinterpret_Widths_MatchNumpy()
        {
            var a = np.arange(6).astype(np.int32);          // little-endian bytes
            // numpy: int32->int16 doubles the count with interleaved zeros
            CollectionAssert.AreEqual(new short[] { 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0 },
                np.frombuffer(a.data, np.int16).ToArray<short>());
            // numpy: int32->int64 halves the count (pairs of int32 combine little-endian)
            CollectionAssert.AreEqual(new long[] { 4294967296L, 12884901890L, 21474836484L },
                np.frombuffer(a.data, np.int64).ToArray<long>());
        }

        [TestMethod]
        public void TwoD_CContig_Source_Flattens_To_1D()
        {
            var m = np.arange(6).astype(np.int32).reshape(2, 3);
            var fb = np.frombuffer(m.data, np.int32);       // numpy flattens a 2-D C-contig buffer to 1-D
            Assert.AreEqual(1, fb.ndim);
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3, 4, 5 }, fb.ToArray<int>());
        }

        [TestMethod]
        public void Count_And_Offset_InBytesAndElements()
        {
            var g = np.arange(5).astype(np.int32);          // 20 bytes
            // offset is BYTES (4), count is ELEMENTS (2) -> elements 1,2
            var fb = np.frombuffer(g.data, np.int32, count: 2, offset: 4);
            CollectionAssert.AreEqual(new[] { 1, 2 }, fb.ToArray<int>());
        }

        [TestMethod]
        public void ZeroD_Source_Yields_One_Element()
        {
            var fb = np.frombuffer(np.array(5).astype(np.int32).data, np.int32); // numpy: [5]
            CollectionAssert.AreEqual(new[] { 5 }, fb.ToArray<int>());
        }

        [TestMethod]
        public void NonContiguous_Buffer_Throws()
        {
            var t = np.arange(6).astype(np.int32).reshape(2, 3).T;   // F-contiguous, not C
            var ex = Assert.ThrowsException<InvalidOperationException>(() => np.frombuffer(t.data, np.int32));
            StringAssert.Contains(ex.Message, "not C-contiguous");    // numpy BufferError text
            var s = np.arange(10).astype(np.int32)["::2"];           // strided
            Assert.ThrowsException<InvalidOperationException>(() => np.frombuffer(s.data, np.int32));
        }

        [TestMethod]
        public void Offset_Misaligned_And_Count_TooBig_Throw()
        {
            var a = np.arange(3).astype(np.int32);
            Assert.ThrowsException<ArgumentException>(() => np.frombuffer(a.data, np.int32, offset: 2)); // 10%4!=0
            Assert.ThrowsException<ArgumentException>(() => np.frombuffer(a.data, np.int32, count: 10));
        }

        [TestMethod]
        public void Null_Buffer_Throws()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.frombuffer((np.MemoryView)null, np.int32));
        }

        [TestMethod]
        public void Readonly_Source_Propagates_To_Result()
        {
            var frozen = np.arange(4).astype(np.int32);
            frozen.flags.writeable = false;                  // frozen C-contiguous buffer
            var fb = np.frombuffer(frozen.data, np.int32);   // numpy: read-only buffer -> read-only array
            Assert.IsFalse(fb.Shape.IsWriteable);
        }

        [TestMethod]
        public void View_Survives_Source_Dispose()
        {
            var src = np.arange(4).astype(np.int32);
            var fb = np.frombuffer(src.data, np.int32);      // shares the ARC-managed buffer
            src.Dispose();
            GC.Collect();
            GC.WaitForPendingFinalizers();
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, fb.ToArray<int>()); // buffer kept alive
        }

        [TestMethod]
        public void All_Dtypes_RoundTrip()
        {
            NPTypeCode[] all =
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
            };
            foreach (var t in all)
            {
                var s = np.arange(4).astype(t);
                var fb = np.frombuffer(s.data, t);
                Assert.AreEqual(4L, fb.size, $"size for {t}");
                Assert.AreEqual(t, fb.typecode, $"dtype for {t}");
            }
        }

        [TestMethod]
        public void Type_And_String_Overloads()
        {
            var a = np.arange(4).astype(np.int32);
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, np.frombuffer(a.data, typeof(int)).ToArray<int>());
            CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, np.frombuffer(a.data, "<i4").ToArray<int>()); // little-endian view
        }
    }
}
