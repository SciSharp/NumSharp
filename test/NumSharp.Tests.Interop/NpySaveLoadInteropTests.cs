using System;
using System.Collections.Generic;
using System.IO;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.IO;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Live-numpy coverage for NumSharp's <c>.npy</c>/<c>.npz</c> stack — the direction the core
    ///     project can't test. The core IO gate replays a COMMITTED corpus of real numpy output (no
    ///     Python at test time); here a real numpy is in-process, so the round-trip is exercised BOTH
    ///     ways against the running library:
    ///     <list type="bullet">
    ///       <item><b>NumSharp writes → numpy reads</b> — the reverse-interop the core project can only
    ///         check via the manual <c>verify_npy_interop.py</c> gate, now automated: numpy must load
    ///         NumSharp's file to a byte-identical array (and NumSharp's writer is byte-for-byte what
    ///         numpy's own <c>np.save</c> emits — asserted directly against numpy's file).</item>
    ///       <item><b>numpy writes → NumSharp reads</b> — <c>np.load_npy</c>/<c>load_npz</c> over genuine
    ///         numpy output, byte-exact.</item>
    ///     </list>
    ///
    ///     <para>numpy is spoken through the established <c>Python.np.*</c> helpers; <c>np</c> stays
    ///     NumSharp. Every crossing is compared with <see cref="ByteContract"/> (canonical C-order bytes
    ///     on both sides). Files live under the OS temp dir and are always cleaned up.</para>
    /// </summary>
    [TestClass]
    public class NpySaveLoadInteropTests : InteropTestBase
    {
        // Every numpy-mappable NumSharp dtype and its native typestr (Char/Decimal are separate cases below).
        private static readonly (string typestr, NPTypeCode tc)[] NumericDtypes =
        {
            ("|b1", NPTypeCode.Boolean), ("|u1", NPTypeCode.Byte),   ("|i1", NPTypeCode.SByte),
            ("<i2", NPTypeCode.Int16),   ("<u2", NPTypeCode.UInt16), ("<i4", NPTypeCode.Int32),
            ("<u4", NPTypeCode.UInt32),  ("<i8", NPTypeCode.Int64),  ("<u8", NPTypeCode.UInt64),
            ("<f2", NPTypeCode.Half),    ("<f4", NPTypeCode.Single), ("<f8", NPTypeCode.Double),
            ("<c16", NPTypeCode.Complex),
        };

        // ==============================  temp-file plumbing  ====================================

        private static void WithNpy(Action<string> body) => WithTemp(".npy", body);
        private static void WithNpz(Action<string> body) => WithTemp(".npz", body);

        private static void WithTemp(string ext, Action<string> body)
        {
            // A path ending in the real extension, so neither np.save (which appends .npy) nor
            // np.savez (which appends .npz) rewrites it — both sides then agree on the filename.
            string path = Path.Combine(Path.GetTempPath(), "ns_interop_" + Guid.NewGuid().ToString("N") + ext);
            try { body(path); }
            finally { try { if (File.Exists(path)) File.Delete(path); } catch { /* best effort */ } }
        }

        /// <summary>numpy reads one member of an <c>.npz</c> and CLOSES the archive (so the file handle is
        /// released and the temp file can be deleted on Windows). Call under the GIL.</summary>
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

        // ==============================  .npy — every dtype, both directions  ===================

        [TestMethod]
        public void Npy_NumSharpWrites_NumpyReads_EveryDtype_ByteExact()
        {
            using (Py.GIL())
                foreach (var (typestr, tc) in NumericDtypes)
                    WithNpy(path =>
                    {
                        using NDArray nd = np.arange(6).astype(tc);
                        np.save(path, nd);                                              // NumSharp writes the .npy

                        using var p = path.ToPython();
                        using PyObject loaded = Python.np.with("np.load(p)", ("p", p)); // real numpy loads it
                        loaded.dtype_str().Should().Be(typestr, $"numpy reads NumSharp's {tc} .npy as {typestr}");
                        loaded.shape_dims().Should().Equal(new long[] { 6 });
                        ByteContract.AssertSameBytes(nd, loaded, $"NumSharp-written {tc} .npy is byte-exact in numpy");
                    });
        }

        [TestMethod]
        public void Npy_NumpyWrites_NumSharpReads_EveryDtype_ByteExact()
        {
            using (Py.GIL())
                foreach (var (typestr, tc) in NumericDtypes)
                    WithNpy(path =>
                    {
                        using var p = path.ToPython();
                        using PyObject arr = Python.np.eval($"np.arange(6).astype('{typestr}')");
                        using (PyObject _ = Python.np.with("np.save(p, a)", ("p", p), ("a", arr))) { }   // numpy writes

                        using NDArray nd = np.load_npy(path);                          // NumSharp reads it
                        nd.typecode.Should().Be(tc, $"NumSharp reads numpy's {typestr} .npy as {tc}");
                        ByteContract.AssertSameBytes(nd, arr, $"numpy-written {typestr} .npy is byte-exact in NumSharp");
                    });
        }

        // ==============================  .npy — the writer is byte-identical to numpy's  ========

        [TestMethod]
        public void Npy_NumSharpWriterBytes_AreIdenticalTo_NumpySave()
        {
            // Not merely "numpy can load it" — NumSharp's whole .npy image (magic + version + header +
            // data) is byte-for-byte what numpy's own np.save writes for the same array. Compared against
            // the actual file numpy produced.
            (NDArray nd, string npExpr)[] cases =
            {
                (np.arange(10).astype(NPTypeCode.Double),                     "np.arange(10, dtype='<f8')"),
                (np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4),        "np.arange(12, dtype='<i4').reshape(3,4)"),
                (np.arange(6).astype(NPTypeCode.Boolean),                     "np.arange(6).astype('|b1')"),
                (np.arange(4).astype(NPTypeCode.Complex),                     "np.arange(4, dtype='<c16')"),
            };

            using (Py.GIL())
                foreach (var (nd, npExpr) in cases)
                    WithNpy(path =>
                    {
                        using (nd)
                        {
                            using var p = path.ToPython();
                            using PyObject arr = Python.np.eval(npExpr);
                            using (PyObject _ = Python.np.with("np.save(p, a)", ("p", p), ("a", arr))) { }   // numpy's reference file

                            byte[] numpyFile = File.ReadAllBytes(path);
                            byte[] numsharp = np.save(nd);                             // NumSharp's in-memory .npy encode
                            numsharp.Should().Equal(numpyFile, $"NumSharp's .npy is byte-identical to numpy's for {npExpr}");
                        }
                    });
        }

        // ==============================  .npy — every layout survives both ways  ================

        [TestMethod]
        public void Npy_NumSharpWrites_NumpyReads_EveryLayout_ByteExact()
        {
            using (Py.GIL())
            {
                (Func<NDArray> make, long[] shape, string label)[] cases =
                {
                    (() => np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4),                          new[] { 3L, 4 }, "contiguous"),
                    (() => np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4).T,                         new[] { 4L, 3 }, "transposed"),
                    (() => np.asfortranarray(np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4)),        new[] { 3L, 4 }, "F-order"),
                    (() => np.arange(24).astype(NPTypeCode.Int32).reshape(4, 6)["1:3, ::2"],               new[] { 2L, 3 }, "sliced"),
                    (() => np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4)["::-1"],                   new[] { 3L, 4 }, "reversed"),
                    (() => np.arange(24).astype(NPTypeCode.Int32).reshape(2, 3, 4),                        new[] { 2L, 3, 4 }, "3-D"),
                };

                foreach (var (make, shape, label) in cases)
                    WithNpy(path =>
                    {
                        using NDArray nd = make();
                        np.save(path, nd);

                        using var p = path.ToPython();
                        using PyObject loaded = Python.np.with("np.load(p)", ("p", p));
                        loaded.shape_dims().Should().Equal(shape, $"{label} keeps its shape through the .npy round-trip");
                        ByteContract.AssertSameBytes(nd, loaded, $"NumSharp-written {label} loads byte-exact in numpy");
                    });
            }
        }

        [TestMethod]
        public void Npy_NumpyWrites_NumSharpReads_EveryLayout_ByteExact()
        {
            using (Py.GIL())
            {
                (string expr, string label)[] cases =
                {
                    ("np.arange(12, dtype='<f8').reshape(3,4)",                 "contiguous"),
                    ("np.arange(12, dtype='<f8').reshape(3,4).T",              "transposed"),
                    ("np.asfortranarray(np.arange(12, dtype='<f8').reshape(3,4))", "F-order"),
                    ("np.arange(24, dtype='<f8').reshape(4,6)[1:3, ::2]",       "sliced"),
                    ("np.arange(12, dtype='<f8').reshape(3,4)[::-1]",           "reversed"),
                    ("np.arange(24, dtype='<f8').reshape(2,3,4)",               "3-D"),
                };

                foreach (var (expr, label) in cases)
                    WithNpy(path =>
                    {
                        using var p = path.ToPython();
                        using PyObject arr = Python.np.eval(expr);
                        using (PyObject _ = Python.np.with("np.save(p, a)", ("p", p), ("a", arr))) { }

                        using NDArray nd = np.load_npy(path);
                        ByteContract.AssertSameBytes(nd, arr, $"numpy-written {label} loads byte-exact in NumSharp");
                    });
            }
        }

        // ==============================  .npy — fortran_order survives NumSharp round-trip  =====

        [TestMethod]
        public void Npy_FortranOrder_SurvivesTheNumSharpRoundTrip()
        {
            // numpy writes an F-contiguous array; NumSharp loads AND re-saves it; numpy re-loads the
            // NumSharp-written file and it is STILL fortran_order and byte-equal — proving NumSharp
            // preserves the fortran_order header flag through both read and write.
            using (Py.GIL())
                WithNpy(pathA => WithNpy(pathB =>
                {
                    using var pa = pathA.ToPython();
                    using var pb = pathB.ToPython();
                    using PyObject fArr = Python.np.eval("np.asfortranarray(np.arange(12, dtype='<f8').reshape(3,4))");
                    using (PyObject _ = Python.np.with("np.save(p, a)", ("p", pa), ("a", fArr))) { }

                    using (NDArray nd = np.load_npy(pathA))                            // NumSharp reads F-order
                        np.save(pathB, nd);                                            // ...and writes it back

                    using PyObject reloaded = Python.np.with("np.load(p)", ("p", pb));
                    Python.np.truthy("a.flags.f_contiguous", ("a", reloaded))
                          .Should().BeTrue("NumSharp preserved fortran_order through load + save");
                    ByteContract.AssertSameBytes(reloaded, fArr, "F-order round-trip through NumSharp is byte-exact");
                }));
        }

        // ==============================  .npy — format versions 1.0 / 2.0 / 3.0  ================

        [TestMethod]
        public void Npy_AllFormatVersions_NumpyReadsThem()
        {
            using (Py.GIL())
                foreach (var version in new (byte major, byte minor)[] { (1, 0), (2, 0), (3, 0) })
                    WithNpy(path =>
                    {
                        using NDArray nd = np.arange(20).astype(NPTypeCode.Double).reshape(4, 5);
                        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write))
                            np.save_version(fs, nd, new NpyFormat.FormatVersion(version.major, version.minor));

                        using var p = path.ToPython();
                        using PyObject loaded = Python.np.with("np.load(p)", ("p", p));
                        ByteContract.AssertSameBytes(nd, loaded, $"numpy reads NumSharp's v{version.major}.{version.minor} .npy");
                    });
        }

        // ==============================  .npy — scalar, empty, special values  ==================

        [TestMethod]
        public void Npy_ScalarAndEmpty_RoundTripBothWays()
        {
            using (Py.GIL())
            {
                // 0-d scalar: NumSharp writes -> numpy reads a 0-d array.
                WithNpy(path =>
                {
                    using NDArray nd = np.array(7.5);
                    np.save(path, nd);
                    using var p = path.ToPython();
                    using PyObject loaded = Python.np.with("np.load(p)", ("p", p));
                    loaded.shape_dims().Should().BeEmpty("a 0-d array has no dimensions");
                    ByteContract.AssertSameBytes(nd, loaded, "0-d .npy is byte-exact");
                });

                // empty (0,3): numpy writes -> NumSharp reads shape + dtype, no data.
                WithNpy(path =>
                {
                    using var p = path.ToPython();
                    using PyObject arr = Python.np.eval("np.empty((0,3), dtype='<f4')");
                    using (PyObject _ = Python.np.with("np.save(p, a)", ("p", p), ("a", arr))) { }
                    using NDArray nd = np.load_npy(path);
                    nd.shape.Should().Equal(0, 3);
                    nd.typecode.Should().Be(NPTypeCode.Single);
                });
            }
        }

        [TestMethod]
        public void Npy_SpecialValues_RoundTripThroughNumSharp_BitIdentical()
        {
            // numpy writes a NaN/inf/-0/subnormal ladder; NumSharp reads it (bit-preserving) and writes
            // it back; numpy re-loads and it is byte-identical to the original — the .npy path preserves
            // every bit, incl. NaN sign and signed zero, through NumSharp.
            using (Py.GIL())
                WithNpy(pathA => WithNpy(pathB =>
                {
                    using var pa = pathA.ToPython();
                    using var pb = pathB.ToPython();
                    using PyObject original = Python.np.eval(
                        "np.array([np.nan, np.inf, -np.inf, -0.0, np.nextafter(0.0, 1.0), " +
                        "np.finfo(np.float64).max, np.finfo(np.float64).min], dtype='<f8')");
                    using (PyObject _ = Python.np.with("np.save(p, a)", ("p", pa), ("a", original))) { }

                    using (NDArray nd = np.load_npy(pathA))
                        np.save(pathB, nd);

                    using PyObject reloaded = Python.np.with("np.load(p)", ("p", pb));
                    ByteContract.AssertSameBytes(reloaded, original, "float specials survive the .npy round-trip bit-for-bit");
                }));
        }

        // ==============================  .npz — both directions, positional + named  ============

        [TestMethod]
        public void Npz_NumSharpWrites_NumpyReads_Positional_ByteExact()
        {
            using (Py.GIL())
                WithNpz(path =>
                {
                    using NDArray a0 = np.arange(6).astype(NPTypeCode.Int32);
                    using NDArray a1 = np.arange(4).astype(NPTypeCode.Double).reshape(2, 2);
                    np.savez(path, a0, a1);                                             // arr_0, arr_1

                    using PyObject r0 = NumpyReadNpzMember(path, "arr_0");
                    using PyObject r1 = NumpyReadNpzMember(path, "arr_1");
                    ByteContract.AssertSameBytes(a0, r0, "npz arr_0 is byte-exact in numpy");
                    ByteContract.AssertSameBytes(a1, r1, "npz arr_1 is byte-exact in numpy");
                });
        }

        [TestMethod]
        public void Npz_NumSharpWrites_NumpyReads_Named_ByteExact()
        {
            using (Py.GIL())
                WithNpz(path =>
                {
                    using NDArray w = np.arange(9).astype(NPTypeCode.Single).reshape(3, 3);
                    using NDArray b = np.arange(3).astype(NPTypeCode.Single);
                    np.savez(path, new Dictionary<string, NDArray> { ["weights"] = w, ["biases"] = b });

                    using PyObject rw = NumpyReadNpzMember(path, "weights");
                    using PyObject rb = NumpyReadNpzMember(path, "biases");
                    ByteContract.AssertSameBytes(w, rw, "npz 'weights' is byte-exact in numpy");
                    ByteContract.AssertSameBytes(b, rb, "npz 'biases' is byte-exact in numpy");
                });
        }

        [TestMethod]
        public void Npz_NumpyWrites_NumSharpReads_Named_ByteExact()
        {
            using (Py.GIL())
                WithNpz(path =>
                {
                    using var p = path.ToPython();
                    using PyObject w = Python.np.eval("np.arange(9, dtype='<f4').reshape(3,3)");
                    using PyObject b = Python.np.eval("np.arange(3, dtype='<f4')");
                    using (PyObject _ = Python.np.with("np.savez(p, weights=w, biases=b)", ("p", p), ("w", w), ("b", b))) { }

                    using NpzFile npz = np.load_npz(path);
                    npz.Keys.Should().Contain(new[] { "weights", "biases" });
                    ByteContract.AssertSameBytes(npz["weights"], w, "numpy npz 'weights' is byte-exact in NumSharp");
                    ByteContract.AssertSameBytes(npz["biases"], b, "numpy npz 'biases' is byte-exact in NumSharp");
                });
        }

        [TestMethod]
        public void Npz_Compressed_RoundTrips_BothDirections()
        {
            using (Py.GIL())
            {
                // NumSharp compressed -> numpy reads.
                WithNpz(path =>
                {
                    using NDArray a = np.arange(100).astype(NPTypeCode.Double);
                    np.savez_compressed(path, a);
                    using PyObject r = NumpyReadNpzMember(path, "arr_0");
                    ByteContract.AssertSameBytes(a, r, "compressed npz decompresses byte-exact in numpy");
                });

                // numpy compressed -> NumSharp reads.
                WithNpz(path =>
                {
                    using var p = path.ToPython();
                    using PyObject a = Python.np.eval("np.arange(100, dtype='<f8')");
                    using (PyObject _ = Python.np.with("np.savez_compressed(p, a)", ("p", p), ("a", a))) { }
                    using NpzFile npz = np.load_npz(path);
                    ByteContract.AssertSameBytes(npz["arr_0"], a, "numpy compressed npz decompresses byte-exact in NumSharp");
                });
            }
        }

        // ==============================  Char (<U1) and the Decimal refusal  ====================

        [TestMethod]
        public void Npy_CharArray_SavesAsUcs4_NumpyReadsTheCodePoints()
        {
            // NumSharp's Char has no numpy numeric dtype; the .npy port stores it as '<U1' (UCS-4 text),
            // so numpy reads back a 1-char string array whose code points are the original Char values.
            using (Py.GIL())
                WithNpy(path =>
                {
                    using NDArray chars = np.array(new[] { 'A', 'z', '9', 'א' });   // last is a BMP non-ASCII code point
                    np.save(path, chars);

                    using var p = path.ToPython();
                    using PyObject loaded = Python.np.with("np.load(p)", ("p", p));
                    loaded.dtype_str().Should().Be("<U1", "NumSharp stores Char as UCS-4 text");
                    Python.np.truthy("(a == np.array(['A','z','9','\\u05d0'])).all()", ("a", loaded))
                          .Should().BeTrue("the code points round-trip through '<U1'");
                });
        }

        [TestMethod]
        public void Npy_DecimalArray_Save_ThrowsWithGuidance()
        {
            // Decimal has no numpy dtype (16-byte, non-IEEE), so the .npy writer refuses it — the same
            // gate the interop's dtype maps enforce.
            WithNpy(path =>
            {
                using NDArray dec = np.array(new decimal[] { 1.5m, -2.5m, 3.25m }).astype(NPTypeCode.Decimal);
                ((Action)(() => np.save(path, dec)))
                    .Should().Throw<NotSupportedException>("decimal has no numpy dtype");
            });
        }
    }
}
