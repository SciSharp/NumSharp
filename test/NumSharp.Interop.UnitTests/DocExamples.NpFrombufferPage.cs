using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for <c>docs/website-src/docs/interop/np-frombuffer.md</c> — reaching any
    ///     Python library through the buffer protocol.
    ///
    ///     <para>Tests read no markdown and assert no prose (the law of <c>98e6045a</c>): each test
    ///     reproduces one code block or table of the page and asserts the behaviour the page claims.
    ///     Third-party-library gates self-skip via <see cref="InteropTestBase.SkipUnless"/> — real
    ///     proof where the package exists, Inconclusive where it doesn't.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_NpFrombufferPage : InteropTestBase
    {
        // ============================  ## The recipe  ============================================

        /// <summary>
        ///     "Take the bytes, name the dtype, hand both to Python" — the page's opening recipe, plus
        ///     its two annotations: `ToNumpyDtypeStr` returns `"&lt;f8"`, and `np.frombuffer` always
        ///     produces a 1-D array that `reshape` re-dimensions.
        /// </summary>
        [TestMethod]
        public void Recipe_RoundTripsThroughFrombuffer()
        {
            var nd = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);

            string dtype, shape;
            using (Py.GIL())
            {
                dtype = NDArrayPythonInterop.ToNumpyDtypeStr(nd.typecode);
                shape = string.Join(", ", nd.shape);

                using PyObject mv = nd.ToMemoryView();
                Scope.Set("mv", mv);
                Scope.Exec($"a = np.frombuffer(mv, '{dtype}')");
            }

            dtype.Should().Be("<f8", "the recipe's comment names the exact dtype string");
            shape.Should().Be("2, 3");
            PyLong("a.ndim").Should().Be(1, "np.frombuffer always produces a 1-D array");

            PyExec($"a = a.reshape({shape})");
            PyExec("a[0, 0] = -5.0");

            nd.GetDouble(0, 0).Should().Be(-5.0, "the Python write lands in NumSharp — same memory");
            PyStr("a.flags['OWNDATA']").Should().Be("False", "numpy agrees the memory is not its own");
        }

        /// <summary>
        ///     "`ToNumpyDtypeStr` is public and total over the dtypes numpy can express" — every
        ///     mappable dtype yields a string numpy itself parses; only Decimal refuses.
        /// </summary>
        [TestMethod]
        public void DtypeStrings_AreTotalOverNumpyExpressibleDtypes()
        {
            var mappable = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte,
                NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
                NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex,
            };

            foreach (NPTypeCode tc in mappable)
            {
                string s = NDArrayPythonInterop.ToNumpyDtypeStr(tc);
                s.Should().NotBeNullOrEmpty();
                PyStr($"np.dtype('{s}').str").Should().NotBeNullOrEmpty($"numpy must parse '{s}' ({tc})");
            }

            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>("Decimal is the one dtype numpy cannot express");
        }

        // ============================  ## What it costs  =========================================

        /// <summary>
        ///     "It is flat, and the copy route is linear." Absolute timings are never gated (spec
        ///     §8.4); the ordering claim is: a 10⁴× size step must move the copy time by far more
        ///     than the view time, and at the large size the copy must dwarf the view.
        /// </summary>
        [TestMethod]
        public void BufferRoute_IsFlatInN_WhileCopyIsLinear()
        {
            long BestTicks(Action a)
            {
                long best = long.MaxValue;
                for (int i = 0; i < 6; i++)
                {
                    var sw = Stopwatch.StartNew();
                    a();
                    sw.Stop();
                    if (i > 0) best = Math.Min(best, sw.ElapsedTicks);   // first pass is warm-up
                }

                return best;
            }

            var small = np.arange(1_000).astype(NPTypeCode.Double);
            var large = np.arange(10_000_000).astype(NPTypeCode.Double);

            long mvSmall = BestTicks(() => { using (Py.GIL()) { using var p = small.ToMemoryView(); } });
            long mvLarge = BestTicks(() => { using (Py.GIL()) { using var p = large.ToMemoryView(); } });
            long cpSmall = BestTicks(() => { using (Py.GIL()) { using var p = small.ToNumpyCopy(); } });
            long cpLarge = BestTicks(() => { using (Py.GIL()) { using var p = large.ToNumpyCopy(); } });

            double ms(long t) => t * 1000.0 / Stopwatch.Frequency;
            string report = $"view {ms(mvSmall):F4}->{ms(mvLarge):F4} ms, copy {ms(cpSmall):F4}->{ms(cpLarge):F4} ms across 10^4x";

            (mvLarge < mvSmall * 20).Should().BeTrue($"the view route must be flat in n ({report})");
            (cpLarge > cpSmall * 50).Should().BeTrue($"the copy route must scale with n ({report})");
            (cpLarge > mvLarge * 10).Should().BeTrue($"at 10M elements the copy must dwarf the view ({report})");

            large.Dispose();
        }

        // ============================  ## What can be exported  ==================================

        /// <summary>
        ///     "`ToMemoryView` refuses it" — every non-contiguous layout row of the table throws the
        ///     page's verbatim `InvalidOperationException`.
        /// </summary>
        [TestMethod]
        public void MemoryView_RefusesNonContiguousLayouts()
        {
            const string documented =
                "the array is not C-contiguous; a flat memoryview would misrepresent it. " +
                "Materialize first: np.ascontiguousarray(nd) or nd.copy().";

            var b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);
            var nonContiguous = new (string Name, NDArray Source)[]
            {
                ("column slice", b["0:4, 0:6:2"]),
                ("transpose", b.T),
                ("reversed", b["::-1"]),
                ("Fortran order", np.asfortranarray(b)),
                ("broadcast", np.broadcast_to(np.arange(3).astype(NPTypeCode.Double), new Shape(2, 3))),
            };

            foreach (var (name, source) in nonContiguous)
            {
                ((Action)(() => { using (Gil()) { using var p = source.ToMemoryView(); } }))
                    .Should().Throw<InvalidOperationException>($"{name} cannot be a flat byte run")
                    .WithMessage(documented);
            }
        }

        /// <summary>
        ///     The table's ✅ rows: C-contiguous (192 bytes), a contiguous row-slice window (96 bytes,
        ///     exactly the window), a 0-d scalar (8 bytes) and an empty array (0 bytes) all export.
        /// </summary>
        [TestMethod]
        public void MemoryView_AcceptsContiguousWindowsScalarsAndEmpty()
        {
            var b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);

            ExportMemoryViewTo("mv_full", b);
            PyLong("len(mv_full)").Should().Be(192, "4 x 6 x 8 bytes");

            ExportMemoryViewTo("mv_rows", b["1:3"]);
            PyLong("len(mv_rows)").Should().Be(96, "a contiguous window at an offset is still contiguous");
            PyStr("np.frombuffer(mv_rows, '<f8').tolist()")
                .Should().Be("[6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]",
                    "the memoryview covers exactly the [offset, offset + size) window, not the parent");

            ExportMemoryViewTo("mv_scalar", np.array(5.0));
            PyLong("len(mv_scalar)").Should().Be(8, "a 0-d float64 scalar is eight bytes");

            ExportMemoryViewTo("mv_empty", np.zeros(new Shape(0, 3), NPTypeCode.Double));
            PyLong("len(mv_empty)").Should().Be(0, "an empty array exports zero bytes");
        }

        /// <summary>
        ///     The dtype table: fourteen of fifteen dtypes cross, both maps agree with the table, and
        ///     numpy's own name for each dtype string is the one printed.
        /// </summary>
        [TestMethod]
        public void DtypeMaps_CoverEveryDtypeExceptDecimal()
        {
            var table = new (NPTypeCode Tc, string Numpy, string Pep, string NumpyName)[]
            {
                (NPTypeCode.Boolean, "|b1", "?", "bool"),
                (NPTypeCode.Byte, "|u1", "B", "uint8"),
                (NPTypeCode.SByte, "|i1", "b", "int8"),
                (NPTypeCode.Int16, "<i2", "h", "int16"),
                (NPTypeCode.UInt16, "<u2", "H", "uint16"),
                (NPTypeCode.Int32, "<i4", "i", "int32"),
                (NPTypeCode.UInt32, "<u4", "I", "uint32"),
                (NPTypeCode.Int64, "<i8", "q", "int64"),
                (NPTypeCode.UInt64, "<u8", "Q", "uint64"),
                (NPTypeCode.Half, "<f2", "e", "float16"),
                (NPTypeCode.Single, "<f4", "f", "float32"),
                (NPTypeCode.Double, "<f8", "d", "float64"),
                (NPTypeCode.Complex, "<c16", "Zd", "complex128"),
                (NPTypeCode.Char, "<u2", "H", "uint16"),
            };

            foreach (var (tc, numpy, pep, numpyName) in table)
            {
                NDArrayPythonInterop.ToNumpyDtypeStr(tc).Should().Be(numpy, $"the table maps {tc} -> {numpy}");
                NDArrayPythonInterop.ToBufferFormat(tc).Should().Be(pep, $"the table maps {tc} -> '{pep}'");
                PyStr($"np.dtype('{numpy}').name").Should().Be(numpyName, $"numpy calls {numpy} '{numpyName}'");
            }
        }

        /// <summary>
        ///     "both maps refuse rather than invent something" — the two Decimal messages, verbatim,
        ///     and the dtype gate fires from <c>ToMemoryView</c> itself.
        /// </summary>
        [TestMethod]
        public void Decimal_ThrowsWithConversionGuidance()
        {
            ((Action)(() => NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>()
                .WithMessage("decimal has no PEP 3118 format (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double).");

            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>()
                .WithMessage("decimal has no numpy dtype (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double).");

            var dec = np.arange(4).astype(NPTypeCode.Decimal);
            ((Action)(() => { using (Gil()) { using var p = dec.ToMemoryView(); } }))
                .Should().Throw<NotSupportedException>().WithMessage("*no PEP 3118 format*");
        }

        // ============================  ## Reading Python's memory  ===============================

        /// <summary>
        ///     The import census table: 8 of the 11 probes view (one of them read-only), complex64
        ///     and UCS-4 fall to copies, big-endian is refused outright.
        /// </summary>
        [TestMethod]
        public void Import_ViewsEveryExporterVariety()
        {
            PyExec("import ctypes, io");

            // the 8 view rows — each really shares (write-crossing checked for the writable ones)
            var viewRows = new (string Expr, NPTypeCode Tc, bool Writable)[]
            {
                ("np.arange(4, dtype='f4')", NPTypeCode.Single, true),
                ("bytearray(b'abcd')", NPTypeCode.Byte, true),
                ("array.array('i', [1, 2, 3])", NPTypeCode.Int32, true),
                ("(ctypes.c_double * 3)(1.5, 2.5, 3.5)", NPTypeCode.Double, true),
                ("memoryview(bytearray(range(8)))[::2]", NPTypeCode.Byte, true),
                ("io.BytesIO(b'12345678').getbuffer()", NPTypeCode.Byte, true),
                ("np.arange(6).reshape(2, 3).T", NPTypeCode.Int64, true),
                ("np.broadcast_to(np.arange(3), (2, 3))", NPTypeCode.Int64, false),
            };

            foreach (var (expr, tc, writable) in viewRows)
            {
                NDArray v = ViewOf(expr, allowReadonly: true);
                v.typecode.Should().Be(tc, expr);
                v.Shape.IsWriteable.Should().Be(writable, expr);
                v.Dispose();
            }

            // the 2 copy rows — the view path declines, the copy path converts
            foreach (string expr in new[] { "np.array([1+2j], dtype='c8')", "np.array(['a', 'b'], dtype='U1')" })
            {
                ((Action)(() => ViewOf(expr, allowReadonly: true).Dispose()))
                    .Should().Throw<NotSupportedException>($"{expr} has no zero-copy representation");
                NDArray c = ImportOf(expr);
                c.Should().NotBeNull($"{expr} still crosses as a copy");
                c.Dispose();
            }

            // the 1 rejected row — both paths refuse
            ((Action)(() => ViewOf("np.arange(4, dtype='>i4')", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>();
            ((Action)(() => ImportOf("np.arange(4, dtype='>i4')").Dispose()))
                .Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     "The first two widen or narrow on copy": complex64 pairs become <see cref="Complex"/>
        ///     values, UCS-4 code points become <see cref="char"/>s — values preserved, never a view.
        /// </summary>
        [TestMethod]
        public void Import_WidensComplex64AndNarrowsUcs4OnCopy()
        {
            PyExec("c8 = np.array([1+2j, 3+4j], dtype='c8')");
            NDArray widened = ImportOf("c8");
            widened.typecode.Should().Be(NPTypeCode.Complex, "complex64 widens to Complex (complex128)");
            ReadAt<Complex>(widened, 0).Should().Be(new Complex(1, 2), "values preserved through the widening");
            ReadAt<Complex>(widened, 1).Should().Be(new Complex(3, 4));
            widened.Dispose();

            PyExec("u1 = np.array(['a', 'b'], dtype='U1')");
            NDArray narrowed = ImportOf("u1");
            narrowed.typecode.Should().Be(NPTypeCode.Char, "UCS-4 text narrows to Char (UTF-16)");
            ReadAt<char>(narrowed, 0).Should().Be('a');
            ReadAt<char>(narrowed, 1).Should().Be('b');
            narrowed.Dispose();
        }

        /// <summary>"big-endian is refused by both paths, with the fix in the message."</summary>
        [TestMethod]
        public void Import_RefusesBigEndian()
        {
            ((Action)(() => ViewOf("np.arange(4, dtype='>i4')", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>()
                .WithMessage("*big-endian*").WithMessage("*newbyteorder('<')*");

            ((Action)(() => ImportOf("np.arange(4, dtype='>i4')").Dispose()))
                .Should().Throw<NotSupportedException>()
                .WithMessage("*big-endian*").WithMessage("*newbyteorder('<')*");
        }

        /// <summary>
        ///     "By default a read-only source is refused outright" — with the page's verbatim message —
        ///     "Pass `allowReadonly: true` and you get a view whose `Shape.IsWriteable` is `false`."
        /// </summary>
        [TestMethod]
        public void ReadonlySource_RefusedByDefault_ViewableOnOptIn()
        {
            ((Action)(() => ViewOf("bytes(b'abcdefgh')").Dispose()))
                .Should().Throw<InvalidOperationException>()
                .WithMessage("the exporter's buffer is read-only; writing through a NumSharp view would corrupt " +
                             "an immutable Python object. Use ToNDArray (copy), or pass allowReadonly:true to " +
                             "take a NON-WRITEABLE view (guarded writes through it throw).");

            NDArray ro = ViewOf("bytes(b'abcdefgh')", allowReadonly: true);
            ro.Shape.IsWriteable.Should().BeFalse("the opt-in view is non-writeable");
            ro.Dispose();
        }

        /// <summary>"a write raises rather than corrupting a `bytes` object" — the page's snippet.</summary>
        [TestMethod]
        public void ReadonlyView_ThrowsOnGuardedWrite()
        {
            NDArray ro = ViewOf("bytes(b'abcdefgh')", allowReadonly: true);

            bool writable = ro.Shape.IsWriteable;
            writable.Should().BeFalse();
            ((Action)(() => ro[0] = (NDArray)(byte)7))
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");

            ro.Dispose();
        }

        // ============================  ## Lifetime  ==============================================

        /// <summary>
        ///     The `ExportAndAbandon` block: dispose the only C# reference, run the GC, and Python
        ///     still reads and writes valid memory until it deletes its own references.
        /// </summary>
        [TestMethod]
        public void Export_SurvivesEveryManagedReferenceDying()
        {
            int pins = NDArrayPythonInterop.LiveExports;

            ExportAndAbandon();
            Pump();

            NDArrayPythonInterop.LiveExports.Should().Be(pins + 1, "Python's frombuffer view pins the buffer");
            PyStr("b3.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0]", "Python still reads valid memory");
            PyExec("b3[0] = 7.0");
            PyStr("b3.tolist()").Should().Be("[7.0, 1.0, 2.0, 3.0]", "...and writes it");

            PyExec("del b3, mvx");
            WaitFor(() => NDArrayPythonInterop.LiveExports == pins)
                .Should().BeTrue("the pin drains when the last Python-side view dies");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void ExportAndAbandon()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                using PyObject mv = nd.ToMemoryView();
                Scope.Set("mvx", mv);
                Scope.Exec("b3 = np.frombuffer(mvx, '<f8')");
            }

            nd.Dispose();   // the only C# reference, gone
        }

        /// <summary>
        ///     "the lease is released when the last NumSharp view over the memory — *including derived
        ///     slices* — dies. Disposal order is irrelevant; the refcount decides."
        /// </summary>
        [TestMethod]
        public void Import_LeaseOutlivesTheOriginalView()
        {
            PyExec("dl = bytearray(range(8))");
            int baseline = NDArrayPythonInterop.LiveImports;

            NDArray original = ViewOf("dl");
            NDArray derived = original["2:"];

            original.Dispose();
            Pump();
            NDArrayPythonInterop.LiveImports.Should().Be(baseline + 1,
                "a derived slice still holds the lease — disposal order is irrelevant");
            ReadAt<byte>(derived, 0).Should().Be(2, "the derived slice still reads valid Python memory");

            derived.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == baseline)
                .Should().BeTrue("the refcount decides: the LAST view releases the lease");
        }

        /// <summary>"Two counters make the live state observable" — they move by exactly one, each way.</summary>
        [TestMethod]
        public void LiveCounters_TrackBothDirections()
        {
            int pinned = NDArrayPythonInterop.LiveExports;
            int leased = NDArrayPythonInterop.LiveImports;

            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil()) { using var p = nd.ToMemoryView(); Scope.Set("cmv", p); }
            NDArrayPythonInterop.LiveExports.Should().Be(pinned + 1, "one live export");

            PyExec("csrc = np.arange(4, dtype='f8')");
            NDArray v = ViewOf("csrc");
            NDArrayPythonInterop.LiveImports.Should().Be(leased + 1, "one live import lease");

            v.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == leased).Should().BeTrue();
            PyExec("del cmv");
            WaitFor(() => NDArrayPythonInterop.LiveExports == pinned).Should().BeTrue();
        }

        /// <summary>
        ///     "It gets pinned: nothing may reallocate it while your view is alive" — CPython's
        ///     verbatim `BufferError` — and the mirror image: an exported NumSharp array refuses
        ///     `resize` while Python holds a view of it.
        /// </summary>
        [TestMethod]
        public void LiveView_LocksTheSourceAgainstResize()
        {
            PyExec("lba = bytearray(b'abcd')");
            NDArray held = ViewOf("lba");

            using (Gil())
                Scope.Exec("try:\n    lba.append(1)\n    lockmsg = ''\nexcept BufferError as e:\n    lockmsg = str(e)");
            PyStr("lockmsg").Should().Be("Existing exports of data: object cannot be re-sized",
                "CPython enforces the protocol while the lease lives");
            held.Dispose();
            Pump();

            var nd = np.arange(8).astype(NPTypeCode.Double);
            ExportTo("locked_nd", nd);
            ((Action)(() => nd.resize(new Shape(16))))
                .Should().Throw<Exception>().WithMessage("*references or is referenced*",
                    "the mirror image: a live Python view pins the NumSharp buffer against reallocation");
            PyExec("del locked_nd");
        }

        /// <summary>"`Dispose()` your view and the object is free again."</summary>
        [TestMethod]
        public void DisposingTheView_ReleasesTheLock()
        {
            PyExec("rba = bytearray(b'abcd')");
            NDArray held = ViewOf("rba");

            held.Dispose();
            Pump();

            PyExec("rba.append(1)");   // must not raise
            PyLong("len(rba)").Should().Be(5, "the lock lifts with the lease");
        }

        // ============================  ## The GIL  ===============================================

        /// <summary>
        ///     "Every conversion verb acquires the GIL itself, re-entrantly" — a bare call from a
        ///     thread holding nothing, and a nested call under an outer acquisition, both work.
        /// </summary>
        [TestMethod]
        public void Verbs_AcquireTheGilThemselves()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            PyObject bare = nd.ToMemoryView();      // no Py.GIL() anywhere on this thread
            using (Gil())
            {
                Scope.Set("g1", bare);
                bare.Dispose();

                using PyObject nested = nd.ToMemoryView();   // re-entrant under an outer acquisition
                Scope.Set("g2", nested);
            }

            PyExec("np.frombuffer(g1, '<f8')[0:1][0]");   // both memoryviews are live and valid
            PyLong("len(g2)").Should().Be(32);
            PyExec("del g1, g2");
        }

        /// <summary>The hot-loop block: one acquisition outside, `requireGIL: false` conversions inside.</summary>
        [TestMethod]
        public void RequireGilFalse_WorksUnderAnOuterAcquisition()
        {
            var batches = new[]
            {
                np.arange(3).astype(NPTypeCode.Double),
                np.arange(3, 6).astype(NPTypeCode.Double),
            };

            var sizes = new List<long>();
            using (Py.GIL())                                                      // one acquisition...
                foreach (var batch in batches)
                    using (PyObject mv = batch.ToMemoryView(requireGIL: false))   // ...N conversions inside
                    {
                        using PyObject nbytes = mv.GetAttr("nbytes");
                        sizes.Add(nbytes.As<long>());
                    }

            sizes.Should().Equal(new long[] { 24, 24 }, "every conversion in the loop completed");
        }

        // ============================  ## Seven libraries, verbatim  =============================

        /// <summary>
        ///     "`torch.frombuffer(mv, dtype=torch.float32)`: same `data_ptr` as `np.frombuffer`; a
        ///     tensor write lands in the `NDArray`."
        /// </summary>
        [TestMethod]
        public void Torch_SharesTheDataPointer()
        {
            SkipUnless("torch");

            var nd = np.arange(6).astype(NPTypeCode.Single);
            using (Py.GIL())
            {
                Scope.Exec("import torch");
                using PyObject mv = nd.ToMemoryView();
                Scope.Set("tmv", mv);
                Scope.Exec("t = torch.frombuffer(tmv, dtype=torch.float32)");
                Scope.Exec("t[0] = 42.0");
            }

            ReadAt<float>(nd, 0).Should().Be(42f, "the tensor write lands in the NDArray");
            PyBool("torch.from_numpy(np.frombuffer(tmv, '<f4')).data_ptr() == t.data_ptr()")
                .Should().BeTrue("torch.frombuffer and np.frombuffer reach the identical address");
            PyExec("del t");
        }

        /// <summary>"`Image.frombuffer(...)`: `getpixel` reads NumSharp's bytes."</summary>
        [TestMethod]
        public void Pillow_ReadsTheExportedBuffer()
        {
            SkipUnless("PIL");

            var img = np.zeros(new Shape(4, 6, 3), NPTypeCode.Byte);
            img[2, 3] = (NDArray)new byte[] { 10, 20, 30 };

            using (Py.GIL())
            {
                Scope.Exec("from PIL import Image");
                using PyObject mv = img.ToMemoryView();
                Scope.Set("pmv", mv);
                Scope.Exec("im = Image.frombuffer('RGB', (6, 4), pmv, 'raw', 'RGB', 0, 1)");
            }

            PyStr("im.getpixel((3, 2))").Should().Be("(10, 20, 30)", "Pillow reads NumSharp's bytes in place");
        }

        /// <summary>
        ///     "A PIL `Image` exports no PEP 3118 buffer, and its `__array_interface__['data']` is a
        ///     `bytes` object ... Call `np.asarray(im)` first; the resulting array imports as a
        ///     read-only view."
        /// </summary>
        [TestMethod]
        public void Pillow_ImageItselfIsNotImportable()
        {
            SkipUnless("PIL");

            PyExec("from PIL import Image\nimsrc = Image.new('RGB', (6, 4))");
            using (Gil())
            {
                using PyObject im = Scope.Eval("imsrc");
                ((Action)(() => NDArrayPythonInterop.ToNDArrayView(im, allowReadonly: true).Dispose()))
                    .Should().Throw<NotSupportedException>()
                    .WithMessage("*(pointer, readonly) tuple*", "bytes-data names no address a view could share");
            }

            NDArray ro = ViewOf("np.asarray(imsrc)", allowReadonly: true);
            ro.typecode.Should().Be(NPTypeCode.Byte);
            ro.Shape.IsWriteable.Should().BeFalse("np.asarray over a PIL image is a read-only view");
            ro.Dispose();
        }

        /// <summary>"`pa.py_buffer(mv)`: `to_numpy(zero_copy_only=True)` returns our address."</summary>
        [TestMethod]
        public void PyArrow_WrapsTheMemoryViewZeroCopy()
        {
            SkipUnless("pyarrow");

            var nd = np.arange(5).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                Scope.Exec("import pyarrow as pa");
                using PyObject mv = nd.ToMemoryView();
                Scope.Set("amv", mv);
                Scope.Exec("abuf = pa.py_buffer(amv)");
                Scope.Exec("aarr = pa.Array.from_buffers(pa.float64(), 5, [None, abuf])");
            }

            PyBool("aarr.to_numpy(zero_copy_only=True).ctypes.data == abuf.address")
                .Should().BeTrue("Arrow wraps the memoryview without copying");
            PyExec("del aarr, abuf");
        }

        /// <summary>"`pd.DataFrame(nd.ToNumpy(), copy=False)`: `np.shares_memory(...)` is `True`."</summary>
        [TestMethod]
        public void Pandas_DataFrameSharesMemory()
        {
            SkipUnless("pandas");

            var x = np.arange(12).reshape(4, 3).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                Scope.Exec("import pandas as pd");
                using PyObject p = x.ToNumpy();
                Scope.Set("pdx", p);
                Scope.Exec("df = pd.DataFrame(pdx, copy=False)");
            }

            PyBool("np.shares_memory(df.to_numpy(), pdx)").Should().BeTrue("the DataFrame is a façade over NumSharp memory");
            PyExec("del df");
        }

        /// <summary>"`pl.Series('v', nd.ToNumpy())`: `to_numpy(allow_copy=False)` shares."</summary>
        [TestMethod]
        public void Polars_SeriesSharesMemory()
        {
            SkipUnless("polars");

            var y = np.arange(5).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                Scope.Exec("import polars as pl");
                using PyObject p = y.ToNumpy();
                Scope.Set("ply", p);
                Scope.Exec("sr = pl.Series('v', ply)");
            }

            PyBool("np.shares_memory(sr.to_numpy(allow_copy=False), ply)").Should().BeTrue();
            PyExec("del sr");
        }

        /// <summary>"`cv2.circle(...)`: draws **into** the NumSharp buffer" — the page's OpenCV block.</summary>
        [TestMethod]
        public void OpenCv_DrawsIntoNumSharpMemory()
        {
            SkipUnless("cv2");

            var frame = np.zeros(new Shape(480, 640, 3), NPTypeCode.Byte);
            using (Py.GIL())
            {
                Scope.Exec("import cv2");
                using PyObject im = frame.ToNumpy();        // cv2 wants an ndarray, not bytes
                Scope.Set("cvim", im);
                Scope.Exec("cv2.circle(cvim, (320, 240), 40, (255, 0, 0), -1)");
            }

            frame.GetByte(240, 320, 0).Should().Be(255, "cv2 rendered straight into NumSharp memory");
            frame.GetByte(240, 320, 1).Should().Be(0);
            frame.GetByte(240, 320, 2).Should().Be(0);
            PyExec("del cvim");
        }

        /// <summary>
        ///     "Nothing above obliges the consumer to know what a numpy array is" — `struct`,
        ///     `hashlib`, `zlib`, a socket, and `memoryview.cast` writing back.
        /// </summary>
        [TestMethod]
        public void Stdlib_ConsumersReadAndWriteTheMemoryView()
        {
            var pcm = np.arange(8).astype(NPTypeCode.Int16);

            using (Py.GIL())
            {
                Scope.Exec("import struct, hashlib, zlib, socket");
                using PyObject mv = pcm.ToMemoryView();
                Scope.Set("smv", mv);

                Scope.Exec("snap = bytes(smv)");                          // pre-write snapshot
                Scope.Exec("first4 = struct.unpack_from('<4h', smv)");
                Scope.Exec("digest = hashlib.sha256(smv).hexdigest()");
                Scope.Exec("packed = zlib.compress(smv)");
                Scope.Exec("s1, s2 = socket.socketpair()\n" +
                           "s1.sendall(smv)\n" +
                           "received = b''\n" +
                           "while len(received) < len(snap):\n" +
                           "    received += s2.recv(64)\n" +
                           "s1.close(); s2.close()");
                Scope.Exec("smv.cast('h')[0] = 77");                      // ...and writes back
            }

            PyStr("first4").Should().Be("(0, 1, 2, 3)", "struct reads the raw bytes directly");
            PyLong("len(digest)").Should().Be(64, "sha256 hashed the buffer in place");
            PyBool("len(packed) > 0").Should().BeTrue("zlib compressed the buffer in place");
            PyBool("received == snap").Should().BeTrue("a socket sends the memoryview's bytes directly");
            ReadAt<short>(pcm, 0).Should().Be(77, "memoryview.cast('h') writes typed elements back through");
        }
    }
}
