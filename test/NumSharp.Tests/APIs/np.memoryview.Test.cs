using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     <c>ndarray.data</c> / <see cref="np.MemoryView"/> parity gate. Every expectation was probed
    ///     against NumPy 2.4.2's <c>memoryview</c> (the object <c>ndarray.data</c> returns). The buffer
    ///     ESSENCE is what is asserted: the pointer at the LOGICAL start, the metadata
    ///     (nbytes/itemsize/ndim/shape/byte-strides/format/readonly/contiguity), write-through element
    ///     access, and <c>tobytes</c> in logical order — all bit-identical to NumPy.
    /// </summary>
    [TestClass]
    public class np_memoryview_tests
    {
        private static long S(object x) => x is NDArray nd ? Convert.ToInt64(nd.GetAtIndex(0)) : Convert.ToInt64(x);
        private static string Hex(byte[] b) => Convert.ToHexString(b).ToLowerInvariant();

        // NumPy's int32/uint32 struct code is platform-dependent (NPY_LONG on Windows -> 'l'/'L',
        // NPY_INT elsewhere -> 'i'/'I'); mirror that so the gate is green on every CI OS.
        private static string I32 => OperatingSystem.IsWindows() ? "l" : "i";
        private static string U32 => OperatingSystem.IsWindows() ? "L" : "I";

        [TestMethod]
        public void Metadata_1D_Int32()
        {
            var mv = np.arange(6).astype(np.int32).data;   // numpy: np.arange(6,dtype=int32).data
            Assert.AreEqual(1, mv.ndim);
            Assert.AreEqual(4, mv.itemsize);
            Assert.AreEqual(24L, mv.nbytes);
            Assert.IsFalse(mv.@readonly);
            CollectionAssert.AreEqual(new long[] { 6 }, mv.shape);
            CollectionAssert.AreEqual(new long[] { 4 }, mv.strides);   // BYTES (numpy memoryview.strides)
            Assert.AreEqual(I32, mv.format);
            Assert.IsTrue(mv.c_contiguous);
            Assert.IsTrue(mv.f_contiguous);                            // 1-D is both, like numpy
            Assert.IsTrue(mv.contiguous);
        }

        [TestMethod]
        public void Strides_Are_Bytes_2D()
        {
            // (2,3) int32: numpy memoryview.strides == (12, 4)
            var mv = np.arange(6).astype(np.int32).reshape(2, 3).data;
            CollectionAssert.AreEqual(new long[] { 12, 4 }, mv.strides);
            Assert.IsTrue(mv.c_contiguous);
            Assert.IsFalse(mv.f_contiguous);
        }

        [TestMethod]
        public void Tobytes_C_F_A_LogicalOrder()
        {
            var a = np.arange(6).astype(np.int32).reshape(2, 3);
            // numpy: a.tobytes('C') = 0,1,2,3,4,5 ; 'F' = 0,3,1,4,2,5 ; 'A' = C (a is C-contig)
            Assert.AreEqual("000000000100000002000000030000000400000005000000", Hex(a.data.tobytes("C")));
            Assert.AreEqual("000000000300000001000000040000000200000005000000", Hex(a.data.tobytes("F")));
            Assert.AreEqual("000000000100000002000000030000000400000005000000", Hex(a.data.tobytes("A")));
        }

        [TestMethod]
        public void Transposed_Contiguity_And_Tobytes()
        {
            var t = np.arange(6).astype(np.int32).reshape(2, 3).T;   // (3,2), F-contiguous
            var mv = t.data;
            Assert.IsFalse(mv.c_contiguous);
            Assert.IsTrue(mv.f_contiguous);
            Assert.IsTrue(mv.contiguous);
            CollectionAssert.AreEqual(new long[] { 4, 12 }, mv.strides);
            // numpy: t.tobytes('C') = 0,3,1,4,2,5 ; t.tobytes('A') = physical (F) = 0,1,2,3,4,5
            Assert.AreEqual("000000000300000001000000040000000200000005000000", Hex(mv.tobytes("C")));
            Assert.AreEqual("000000000100000002000000030000000400000005000000", Hex(mv.tobytes("A")));
        }

        [TestMethod]
        public void Pointer_Is_At_Logical_Start()
        {
            var b = np.arange(10).astype(np.int32);
            long basePtr = (long)b.data.Address;
            // numpy: a[3:7].__array_interface__['data'][0] == base + 12 ; a[::-1] == base + 36 (last elem)
            Assert.AreEqual(12L, (long)b["3:7"].data.Address - basePtr);
            Assert.AreEqual(36L, (long)b["::-1"].data.Address - basePtr);
        }

        [TestMethod]
        public void Reversed_Strides_And_LogicalBytes()
        {
            var rev = np.arange(6).astype(np.int32)["::-1"];   // logical [5,4,3,2,1,0]
            var mv = rev.data;
            CollectionAssert.AreEqual(new long[] { -4 }, mv.strides);   // numpy: (-4,)
            Assert.IsFalse(mv.c_contiguous);
            Assert.AreEqual("050000000400000003000000020000000100000000000000", Hex(mv.tobytes("C")));
        }

        [TestMethod]
        public void Broadcast_Is_Readonly_With_ZeroStride()
        {
            var bc = np.broadcast_to(np.arange(3).astype(np.int32), (2, 3));
            var mv = bc.data;
            Assert.IsTrue(mv.@readonly);                               // numpy: broadcast view is readonly
            CollectionAssert.AreEqual(new long[] { 0, 4 }, mv.strides); // stride-0 leading axis
            Assert.AreEqual(24L, mv.nbytes);                          // logical size
            // numpy: bc.tobytes('C') = 0,1,2,0,1,2
            Assert.AreEqual("000000000100000002000000000000000100000002000000", Hex(mv.tobytes("C")));
        }

        [TestMethod]
        public void WriteThrough_Hits_Base_Including_Transposed()
        {
            var w = np.arange(6).astype(np.int32).reshape(2, 3);
            w.data[1, 2] = 999;                 // numpy: w.data[1,2] = 999 -> w[1,2] == 999
            Assert.AreEqual(999L, S(w.GetValue(1, 2)));

            var t = w.T;                        // transposed view, stride-aware write-through
            t.data[2, 0] = 777;                 // maps to w[0,2]
            Assert.AreEqual(777L, S(w.GetValue(0, 2)));
        }

        [TestMethod]
        public void WriteThrough_Coerces_Value_To_Dtype()
        {
            var f = np.zeros(new Shape(3), np.float64);
            f.data[0] = 7;                      // int assigned to float64 buffer -> 7.0 (numpy casts to format)
            Assert.AreEqual(7.0, Convert.ToDouble(f.GetValue(0)), 0.0);
        }

        [TestMethod]
        public void Readonly_Write_Throws()
        {
            var bc = np.broadcast_to(np.arange(3).astype(np.int32), (2, 3));
            Assert.ThrowsException<InvalidOperationException>(() => bc.data[0, 0] = 1);
        }

        [TestMethod]
        public void Element_Get_Negative_And_Reversed()
        {
            var rev = np.arange(6).astype(np.int32)["::-1"];  // logical [5,4,3,2,1,0]
            Assert.AreEqual(5L, S(rev.data[0]));
            Assert.AreEqual(0L, S(rev.data[5]));
            Assert.AreEqual(0L, S(rev.data[-1]));             // negative index counts from the end
        }

        [TestMethod]
        public void ZeroD_Shape_And_Length()
        {
            var s = np.array(5).astype(np.int32);   // 0-d
            var mv = s.data;
            Assert.AreEqual(0, mv.ndim);
            Assert.AreEqual(0, mv.shape.Length);
            Assert.AreEqual(4L, mv.nbytes);         // one itemsize
            Assert.AreEqual("05000000", Hex(mv.tobytes()));
            Assert.ThrowsException<InvalidOperationException>(() => { var _ = mv.Length; }); // len() of unsized object
        }

        [TestMethod]
        public void Empty_Array_Reports_Zero_Bytes()
        {
            var e = np.zeros(new Shape(0, 3), np.float32);
            var mv = e.data;
            Assert.AreEqual(0L, mv.nbytes);
            Assert.AreEqual(0, mv.tobytes().Length);
        }

        [TestMethod]
        public void Format_Codes_All_Dtypes()
        {
            (NPTypeCode t, string f)[] cases =
            {
                (NPTypeCode.Boolean, "?"), (NPTypeCode.SByte, "b"), (NPTypeCode.Byte, "B"),
                (NPTypeCode.Int16, "h"), (NPTypeCode.UInt16, "H"),
                (NPTypeCode.Int32, I32), (NPTypeCode.UInt32, U32),
                (NPTypeCode.Int64, "q"), (NPTypeCode.UInt64, "Q"),
                (NPTypeCode.Half, "e"), (NPTypeCode.Single, "f"), (NPTypeCode.Double, "d"),
                (NPTypeCode.Complex, "Zd"),
                (NPTypeCode.Char, "u"),      // NumSharp-only (no NumPy dtype)
                (NPTypeCode.Decimal, "16s"), // NumSharp-only (no NumPy dtype)
            };
            foreach (var (t, f) in cases)
                Assert.AreEqual(f, np.zeros(new Shape(2), t).data.format, $"format for {t}");
        }

        [TestMethod]
        public void Hex_Matches_LogicalBytes()
        {
            var b = np.arange(4).astype(np.int32);
            Assert.AreEqual("00000000010000000200000003000000", b.data.hex()); // numpy: b.data.hex()
        }

        [TestMethod]
        public void Errors_BadOrder_And_Arity()
        {
            var b = np.arange(4).astype(np.int32);
            Assert.ThrowsException<ArgumentException>(() => b.data.tobytes("Q"));
            Assert.ThrowsException<ArgumentException>(() => { var _ = b.data[0, 1]; }); // 1-D indexed with 2 indices
        }

        [TestMethod]
        public void Obj_Is_Source_And_Handle_Is_Fresh()
        {
            var b = np.arange(4).astype(np.int32);
            Assert.IsTrue(ReferenceEquals(b, b.data.obj));       // numpy: mv.obj is a
            Assert.IsFalse(ReferenceEquals(b.data, b.data));     // a fresh handle per access
        }

        // ---- Consumer round-trip: np.array / asarray / asanyarray / ascontiguousarray /
        //      asfortranarray of ndarray.data (all probed against NumPy 2.4.2). ----

        [TestMethod]
        public void Asarray_Roundtrip_IsView_PreservesShape()
        {
            // numpy: np.asarray(a.data) shares memory and preserves the 2-D shape/layout.
            var src = np.arange(6).astype(np.int32).reshape(2, 3);
            var v = np.asarray(src.data);
            CollectionAssert.AreEqual(new long[] {2, 3 }, v.shape);
            v.SetValue(123, 1, 2);                              // write-through hits the source
            Assert.AreEqual(123L, S(src.GetValue(1, 2)));
        }

        [TestMethod]
        public void Array_Roundtrip_Copies_PreservesShape()
        {
            // numpy: np.array(a.data) COPIES (independent) and preserves shape.
            var src = np.arange(6).astype(np.int32).reshape(2, 3);
            var c = np.array(src.data);
            CollectionAssert.AreEqual(new long[] {2, 3 }, c.shape);
            c.SetValue(123, 1, 2);
            Assert.AreEqual(123L, S(c.GetValue(1, 2)));
            Assert.AreEqual(5L, S(src.GetValue(1, 2)));         // source untouched (copy)
        }

        [TestMethod]
        public void Asanyarray_Roundtrip_IsView()
        {
            // Previously threw NotSupportedException (bound to the object converter). Now a view.
            var src = np.arange(6).astype(np.int32).reshape(2, 3);
            var v = np.asanyarray(src.data);
            CollectionAssert.AreEqual(new long[] {2, 3 }, v.shape);
            v.SetValue(77, 0, 1);
            Assert.AreEqual(77L, S(src.GetValue(0, 1)));
        }

        [TestMethod]
        public void Ascontiguousarray_Roundtrip_ForcesC()
        {
            // numpy: ascontiguousarray(F.data) -> C-contiguous.
            var f = np.arange(6).astype(np.int32).reshape(2, 3).T;   // (3,2) F-contiguous
            Assert.IsFalse(f.Shape.IsContiguous);
            var r = np.ascontiguousarray(f.data);
            CollectionAssert.AreEqual(new long[] {3, 2 }, r.shape);
            Assert.IsTrue(r.Shape.IsContiguous);
        }

        [TestMethod]
        public void Asfortranarray_Roundtrip_ForcesF()
        {
            // numpy: asfortranarray(C.data) -> F-contiguous.
            var c = np.arange(6).astype(np.int32).reshape(2, 3);     // C-contiguous
            var r = np.asfortranarray(c.data);
            CollectionAssert.AreEqual(new long[] {2, 3 }, r.shape);
            Assert.IsTrue(r.Shape.IsFContiguous);
        }

        [TestMethod]
        public void Asarray_Roundtrip_DtypeConverts()
        {
            var src = np.arange(6).astype(np.int32).reshape(2, 3);
            Assert.AreEqual(typeof(float), np.asarray(src.data, np.float32).dtype);   // Type overload
            Assert.AreEqual(typeof(float), np.asarray(src.data, "float32").dtype);    // string overload
        }

        [TestMethod]
        public void Array_From_Readonly_Broadcast_Is_Writeable_Copy()
        {
            // numpy: np.array(broadcast.data) is a writeable copy of the logical values.
            var bc = np.broadcast_to(np.arange(3).astype(np.int32), (2, 3));
            Assert.IsTrue(bc.data.@readonly);
            var c = np.array(bc.data);
            CollectionAssert.AreEqual(new long[] {2, 3 }, c.shape);
            Assert.IsTrue(c.Shape.IsWriteable);
            Assert.AreEqual(0L, S(c.GetValue(0, 0)));
            Assert.AreEqual(2L, S(c.GetValue(1, 2)));
        }

        [TestMethod]
        public void Roundtrip_NullBuffer_Throws()
        {
            Assert.ThrowsException<ArgumentNullException>(() => np.array((np.MemoryView)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.asarray((np.MemoryView)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.asanyarray((np.MemoryView)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.ascontiguousarray((np.MemoryView)null));
            Assert.ThrowsException<ArgumentNullException>(() => np.asfortranarray((np.MemoryView)null));
        }
    }
}
