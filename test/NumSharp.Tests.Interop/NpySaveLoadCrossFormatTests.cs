using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using NumSharp.IO;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The harder half of the live-numpy <c>.npy</c>/<c>.npz</c> coverage — the crossings that
    ///     CONVERT rather than copy, the zero-copy memory-map path, multi-array streams, the content
    ///     dispatch of <c>np.load</c>, and the numpy-only dtypes NumSharp must REFUSE. Everything real
    ///     numpy can emit that a native-endian NumSharp buffer has to reinterpret or reject is exercised
    ///     against the running library, in the pythonic <c>Python.np</c> style.
    /// </summary>
    [TestClass]
    public class NpySaveLoadCrossFormatTests : InteropTestBase
    {
        // ---- temp-file plumbing (self-contained; same shape as NpySaveLoadInteropTests) ----------

        private static void WithNpy(Action<string> body) => WithTemp(".npy", body);
        private static void WithNpz(Action<string> body) => WithTemp(".npz", body);

        private static void WithTemp(string ext, Action<string> body)
        {
            string path = Path.Combine(Path.GetTempPath(), "ns_xfmt_" + Guid.NewGuid().ToString("N") + ext);
            try { body(path); }
            finally { try { if (File.Exists(path)) File.Delete(path); } catch { /* best effort (mmap handles) */ } }
        }

        /// <summary>numpy writes <paramref name="npExpr"/> to <paramref name="path"/> via np.save. Call under the GIL.</summary>
        private static void NumpySave(string path, string npExpr)
        {
            using var p = path.ToPython();
            using PyObject a = Python.np.eval(npExpr);
            using (PyObject _ = Python.np.with("np.save(p, a)", ("p", p), ("a", a))) { }
        }

        // ==============================  big-endian read -> byte-swap to native  ================

        [TestMethod]
        public void Npy_NumpyBigEndian_NumSharpReads_ByteSwapsToNativeTwin()
        {
            // A native-endian NumSharp buffer can't view big-endian memory — but the .npy READER
            // byte-reverses each element on load, so a '>code' file must equal its little-endian twin.
            string[] codes = { "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8" };
            using (Py.GIL())
                foreach (string code in codes)
                    WithNpy(path =>
                    {
                        NumpySave(path, $"np.array([1, 2, 3, 100], dtype='>{code}')");
                        using NDArray nd = np.load_npy(path);
                        nd.typecode.Should().Be(NDArrayPythonInterop.FromNumpyDtypeStr($"<{code}"), $">{code}");
                        using PyObject twin = Python.np.eval($"np.array([1, 2, 3, 100], dtype='<{code}')");
                        ByteContract.AssertSameBytes(nd, twin, $"big-endian '>{code}' .npy byte-swaps to its native twin");
                    });
        }

        [TestMethod]
        public void Npy_NumpyBigEndianComplex_NumSharpReads_SwapsEachHalf()
        {
            using (Py.GIL())
            {
                // complex128 big-endian: each 8-byte half reversed independently.
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([1+2j, -3-4j, 0+0j], dtype='>c16')");
                    using NDArray nd = np.load_npy(path);
                    nd.typecode.Should().Be(NPTypeCode.Complex);
                    using PyObject twin = Python.np.eval("np.array([1+2j, -3-4j, 0+0j], dtype='<c16')");
                    ByteContract.AssertSameBytes(nd, twin, ">c16 swaps each half to the native complex128 twin");
                });
            }
        }

        // ==============================  complex64 (<c8) read -> widen to complex128  ===========

        [TestMethod]
        public void Npy_NumpyComplex64_NumSharpReads_WidensToComplex128()
        {
            using (Py.GIL())
            {
                // NumSharp has a single complex width; a '<c8' file is widened to Complex on load.
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([1+2j, 3+4j, -5-6j], dtype='<c8')");
                    using NDArray nd = np.load_npy(path);
                    nd.typecode.Should().Be(NPTypeCode.Complex, "complex64 widens to complex128");
                    using PyObject twin = Python.np.eval("np.array([1+2j, 3+4j, -5-6j], dtype='<c16')");
                    ByteContract.AssertSameBytes(nd, twin, "<c8 widens byte-exact to the <c16 twin");
                });

                // big-endian complex64: widen AND swap in one pass.
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([1+2j, 3+4j], dtype='>c8')");
                    using NDArray nd = np.load_npy(path);
                    using PyObject twin = Python.np.eval("np.array([1+2j, 3+4j], dtype='<c16')");
                    ByteContract.AssertSameBytes(nd, twin, ">c8 widens-and-swaps to native complex128");
                });
            }
        }

        // ==============================  mmap_mode over numpy-written files  =====================

        [TestMethod]
        public void Npy_Mmap_ReadOnly_OverNumpyFile_IsByteExactAndNonWriteable()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.arange(12, dtype='<f8').reshape(3,4)");

                    var nd = (NDArray)np.load(path, mmap_mode: "r");   // memory-mapped, read-only
                    try
                    {
                        nd.Shape.IsWriteable.Should().BeFalse("mmap 'r' is a read-only view");
                        using PyObject arr = Python.np.eval("np.arange(12, dtype='<f8').reshape(3,4)");
                        ByteContract.AssertSameBytes(nd, arr, "the memory-mapped array reads numpy's file byte-exact");
                    }
                    finally { nd.Dispose(); }   // release the mapping so the temp file can be deleted
                });
        }

        [TestMethod]
        public void Npy_Mmap_ReadWrite_NumSharpWriteThrough_IsVisibleToNumpy()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.zeros(5, dtype='<f8')");

                    var m = (NDArray)np.load(path, mmap_mode: "r+");   // read-write map
                    try
                    {
                        m.Shape.IsWriteable.Should().BeTrue("mmap 'r+' is writeable");
                        m[2] = 42.5;                                   // write through the map...
                    }
                    finally { m.Dispose(); }                           // ...flushed to disk on dispose

                    using var p = path.ToPython();
                    using PyObject reread = Python.np.with("np.load(p)", ("p", p));
                    Python.np.truthy("a[2] == 42.5 and a[0] == 0.0", ("a", reread))
                          .Should().BeTrue("numpy re-reads the value NumSharp wrote through the r+ map");
                });
        }

        // ==============================  several arrays in one .npy stream  ======================

        [TestMethod]
        public void Npy_MultipleArrays_OneStream_NumSharpWrites_NumpyReadsBoth()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    using NDArray a = np.arange(6).astype(NPTypeCode.Int32);
                    using NDArray b = np.arange(4).astype(NPTypeCode.Double).reshape(2, 2);
                    using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
                    {
                        np.save(fs, a);   // successive saves to one handle append
                        np.save(fs, b);
                    }

                    // numpy reads them back in order from one handle.
                    using var scope = Py.CreateScope();
                    scope.Exec("import numpy as np");
                    using var p = path.ToPython();
                    scope.Set("p", p);
                    scope.Exec("f = open(p, 'rb'); _r0 = np.load(f); _r1 = np.load(f); f.close()");
                    using PyObject r0 = scope.Eval("_r0");
                    using PyObject r1 = scope.Eval("_r1");
                    ByteContract.AssertSameBytes(a, r0, "first array in the shared stream");
                    ByteContract.AssertSameBytes(b, r1, "second array in the shared stream");
                });
        }

        [TestMethod]
        public void Npy_MultipleArrays_OneStream_NumpyWrites_NumSharpReadsBoth()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    using PyObject a = Python.np.eval("np.arange(6, dtype='<i4')");
                    using PyObject b = Python.np.eval("np.arange(4, dtype='<f8').reshape(2,2)");
                    using var scope = Py.CreateScope();
                    scope.Exec("import numpy as np");
                    using var p = path.ToPython();
                    scope.Set("p", p);
                    scope.Set("a", a);
                    scope.Set("b", b);
                    scope.Exec("f = open(p, 'wb'); np.save(f, a); np.save(f, b); f.close()");

                    using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read))
                    {
                        using NDArray a2 = np.load_npy(fs);   // successive reads from one handle
                        using NDArray b2 = np.load_npy(fs);
                        ByteContract.AssertSameBytes(a2, a, "first array read back from the shared stream");
                        ByteContract.AssertSameBytes(b2, b, "second array read back from the shared stream");
                    }
                });
        }

        // ==============================  np.load content dispatch (npy vs npz)  ==================

        [TestMethod]
        public void Load_DispatchesByMagic_NpyToNDArray_NpzToNpzFile()
        {
            using (Py.GIL())
            {
                WithNpy(path =>
                {
                    NumpySave(path, "np.arange(5, dtype='<i8')");
                    object loaded = np.load(path);                     // untyped — dispatch on content, not extension
                    loaded.Should().BeAssignableTo<NDArray>("a .npy magic yields an NDArray");
                    ((NDArray)loaded).Dispose();
                });

                WithNpz(path =>
                {
                    using var p = path.ToPython();
                    using PyObject a = Python.np.eval("np.arange(5, dtype='<i8')");
                    using (PyObject _ = Python.np.with("np.savez(p, a)", ("p", p), ("a", a))) { }
                    object loaded = np.load(path);
                    loaded.Should().BeAssignableTo<NpzFile>("a .npz (ZIP) magic yields an NpzFile");
                    ((NpzFile)loaded).Dispose();
                });
            }
        }

        // ==============================  .npz mixed positional + named  =========================

        [TestMethod]
        public void Npz_MixedPositionalAndNamed_RoundTrips_BothDirections()
        {
            using (Py.GIL())
            {
                // NumSharp writes arr_0 + named 'x' -> numpy reads both.
                WithNpz(path =>
                {
                    using NDArray a = np.arange(3).astype(NPTypeCode.Int32);
                    using NDArray x = np.arange(4).astype(NPTypeCode.Double);
                    np.savez(path, new[] { a }, new Dictionary<string, NDArray> { ["x"] = x });

                    using PyObject r0 = NumpyReadNpzMember(path, "arr_0");
                    using PyObject rx = NumpyReadNpzMember(path, "x");
                    ByteContract.AssertSameBytes(a, r0, "positional arr_0");
                    ByteContract.AssertSameBytes(x, rx, "named x");
                });

                // numpy writes arr_0 + named 'x' -> NumSharp reads both.
                WithNpz(path =>
                {
                    using var p = path.ToPython();
                    using PyObject a = Python.np.eval("np.arange(3, dtype='<i4')");
                    using PyObject x = Python.np.eval("np.arange(4, dtype='<f8')");
                    using (PyObject _ = Python.np.with("np.savez(p, a, x=x)", ("p", p), ("a", a), ("x", x))) { }

                    using NpzFile npz = np.load_npz(path);
                    ByteContract.AssertSameBytes(npz["arr_0"], a, "positional arr_0");
                    ByteContract.AssertSameBytes(npz["x"], x, "named x");
                });
            }
        }

        // ==============================  NpzFile surface over numpy data  =======================

        [TestMethod]
        public void Npz_NpzFileSurface_DotAccess_ContainsKey_Files_OverNumpyData()
        {
            using (Py.GIL())
                WithNpz(path =>
                {
                    using var p = path.ToPython();
                    using PyObject w = Python.np.eval("np.arange(6, dtype='<f4').reshape(2,3)");
                    using PyObject b = Python.np.eval("np.arange(2, dtype='<f4')");
                    using (PyObject _ = Python.np.with("np.savez(p, weights=w, biases=b)", ("p", p), ("w", w), ("b", b))) { }

                    using NpzFile npz = np.load_npz(path);
                    npz.Count.Should().Be(2);
                    npz.ContainsKey("weights").Should().BeTrue();
                    npz.ContainsKey("missing").Should().BeFalse();
                    ((IEnumerable<string>)npz.Files).Should().Contain(new[] { "weights", "biases" });

                    NDArray dotAccess = npz.f.weights;   // NumPy's BagObj dot access
                    ByteContract.AssertSameBytes(dotAccess, w, "npz.f.weights (dot access) reads numpy's member byte-exact");
                });
        }

        // ==============================  byte[] load of a numpy file image  ======================

        [TestMethod]
        public void Load_NumpyFileBytes_ViaByteArrayOverload_IsByteExact()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.arange(12, dtype='<f8').reshape(3,4)");
                    byte[] image = File.ReadAllBytes(path);           // the raw .npy numpy produced

                    using NDArray nd = np.load_npy(image);            // NumSharp decodes the in-memory image
                    using PyObject arr = Python.np.eval("np.arange(12, dtype='<f8').reshape(3,4)");
                    ByteContract.AssertSameBytes(nd, arr, "np.load_npy(byte[]) of numpy's image is byte-exact");
                });
        }

        // ==============================  numpy-only dtypes NumSharp must refuse  =================

        [TestMethod]
        public void Load_NumpyObjectArray_IsRefused_WithPickleGuidance()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([1, 2, 3], dtype=object)");   // np.save pickles it (allow_pickle default True)
                    ((Action)(() => { using var _ = np.load_npy(path); }))
                        .Should().Throw<FormatException>().WithMessage("*pickle*");
                });
        }

        [TestMethod]
        public void Load_NumpyStructuredDtype_IsRefused()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([(1, 2.5), (3, 4.5)], dtype=[('a', '<i4'), ('b', '<f8')])");
                    ((Action)(() => { using var _ = np.load_npy(path); }))
                        .Should().Throw<NotSupportedException>("a structured/record dtype has no NumSharp dtype");
                });
        }

        [TestMethod]
        public void Load_NumpyDatetime64_IsRefused()
        {
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.array(['2020-01-01', '2020-06-15'], dtype='datetime64[D]')");
                    ((Action)(() => { using var _ = np.load_npy(path); }))
                        .Should().Throw<NotSupportedException>("datetime64 has no NumSharp dtype");
                });
        }

        // ==============================  numpy text dtypes: <U1 narrows, wider ones refuse  =====

        [TestMethod]
        public void Npy_NumpyUcs4_U1_NumSharpReads_AsChar()
        {
            // The reverse of the Char->'<U1' write: numpy's one-code-point-per-element '<U1' narrows to
            // NumSharp's 2-byte Char on load (BMP code points).
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.array(['A', 'z', '9', '\\u05d0'], dtype='<U1')");
                    using NDArray nd = np.load_npy(path);
                    nd.typecode.Should().Be(NPTypeCode.Char, "numpy <U1 narrows to NumSharp Char");
                    nd.shape.Should().Equal(4);
                    nd.GetAtIndex<char>(0).Should().Be('A');
                    nd.GetAtIndex<char>(3).Should().Be('א');
                });
        }

        [TestMethod]
        public void Npy_NumpyWideText_IsRefused()
        {
            using (Py.GIL())
            {
                // '<U3' is a whole 3-char string per element — not one Char, no NumSharp dtype.
                WithNpy(path =>
                {
                    NumpySave(path, "np.array(['abc', 'de'], dtype='<U3')");
                    ((Action)(() => { using var _ = np.load_npy(path); }))
                        .Should().Throw<NotSupportedException>("multi-char <U3 text has no NumSharp dtype");
                });

                // '|S2' is a bytes-string dtype — likewise unrepresentable.
                WithNpy(path =>
                {
                    NumpySave(path, "np.array([b'ab', b'cd'], dtype='|S2')");
                    ((Action)(() => { using var _ = np.load_npy(path); }))
                        .Should().Throw<NotSupportedException>("bytes-string |S has no NumSharp dtype");
                });
            }
        }

        // ==============================  mmap copy-on-write ('c')  ===============================

        [TestMethod]
        public void Npy_Mmap_CopyOnWrite_NumSharpWrite_IsNotFlushedToDisk()
        {
            // 'c' maps are writeable locally but the changes never reach the file, so numpy re-reading
            // the file must still see the original data — the complement of the r+ write-through test.
            using (Py.GIL())
                WithNpy(path =>
                {
                    NumpySave(path, "np.zeros(4, dtype='<i4')");

                    var m = (NDArray)np.load(path, mmap_mode: "c");
                    try
                    {
                        m.Shape.IsWriteable.Should().BeTrue("'c' maps are writeable...");
                        m[1] = 99;
                        m.GetInt32(1).Should().Be(99, "...and the private copy sees the write");
                    }
                    finally { m.Dispose(); }

                    using var p = path.ToPython();
                    using PyObject reread = Python.np.with("np.load(p)", ("p", p));
                    Python.np.truthy("(a == 0).all()", ("a", reread))
                          .Should().BeTrue("copy-on-write changes are NOT flushed to disk");
                });
        }

        // ==============================  broadcast view saves materialized  =====================

        [TestMethod]
        public void Npy_NumSharpBroadcastView_NumpyReads_Materialized()
        {
            // A stride-0 broadcast view has no dense storage of its logical shape; np.save must
            // materialize it, exactly as numpy's own np.save materializes a broadcast_to array.
            using (Py.GIL())
                WithNpy(path =>
                {
                    using NDArray bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Int32), new Shape(2, 3));
                    np.save(path, bc);

                    using var p = path.ToPython();
                    using PyObject loaded = Python.np.with("np.load(p)", ("p", p));
                    loaded.shape_dims().Should().Equal(new long[] { 2, 3 });
                    ByteContract.AssertSameBytes(bc, loaded, "a broadcast view saves as its materialized (2,3) data");
                });
        }

        // ---- shared with the numpy-read-npz pattern (close the archive so the file can be deleted) -

        private static PyObject NumpyReadNpzMember(string path, string member)
        {
            using var scope = Py.CreateScope();
            scope.Exec("import numpy as np");
            using var p = path.ToPython();
            using var m = member.ToPython();
            scope.Set("p", p);
            scope.Set("m", m);
            scope.Exec("_d = np.load(p); _r = np.array(_d[m]); _d.close()");
            return scope.Eval("_r");
        }
    }
}
