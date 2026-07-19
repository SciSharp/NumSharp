using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for every code example, table row and measured number on
    ///     <c>docs/website-src/docs/interop/zero-copy-model.md</c> — the page that explains how a
    ///     conversion decides <b>view</b> or <b>copy</b>.
    ///
    ///     <para>The page's headline is a measurement ("N view, M copy across K exporter varieties"),
    ///     so the census that produces those numbers lives here as
    ///     <see cref="Coverage_Census_MeasuresTheDocumentedTotals"/> and is re-run on every test pass:
    ///     the documentation quotes what this test measures, not a remembered figure.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_ZeroCopyModelPage : InteropTestBase
    {
        // ============================  ## The two semantics  =====================================

        /// <summary>
        ///     The view/copy comparison table, row by row: shared memory, write visibility, the
        ///     Py_buffer lock (view) vs none (copy), and total coverage for the copy path.
        /// </summary>
        [TestMethod]
        public void TwoSemantics_TableRows_HoldForBothPaths()
        {
            PyExec("ts = bytearray(b'abcd')");

            // --- the VIEW column
            NDArray view = ViewOf("ts");
            WriteAt(view, (byte)65, 0);
            PyLong("ts[0]").Should().Be(65, "view: later writes on either side are visible to the other");
            PyExec("ts[1] = 66");
            ReadAt<byte>(view, 1).Should().Be(66);

            PyExec("try:\n    ts.append(1)\n    ts_locked = True\nexcept BufferError:\n    ts_locked = False");
            PyBool("ts_locked").Should().BeFalse("view: holds a Py_buffer lock (blocks resize)");
            view.Dispose();
            Pump();

            // --- the COPY column
            NDArray copy = ImportOf("ts");
            WriteAt(copy, (byte)90, 0);
            PyLong("ts[0]").Should().Be(65, "copy: later writes are invisible to the other side");
            PyExec("ts.append(1)");   // no lock at all
            PyLong("len(ts)").Should().Be(5, "copy: no effect on the Python object");
            copy.Dispose();

            // "Works for: representable dtypes/layouts | EVERYTHING"
            PyExec("c8 = np.array([1+2j], dtype='c8')");
            ((Action)(() => ViewOf("c8", allowReadonly: true).Dispose()))
                .Should().Throw<Exception>("a view only works for representable dtypes/layouts");
            NDArray anything = ImportOf("c8");
            anything.Should().NotBeNull("the copy path works for everything");
            anything.Dispose();
        }

        // ============================  ## `Auto` — view-first, copy only when impossible  ========

        /// <summary>
        ///     The page's first runnable sample, verbatim: register the codec, decode a contiguous
        ///     numpy array, and watch a NumSharp write reach Python.
        /// </summary>
        [TestMethod]
        public void Auto_TheRegisterCodecSample_RunsVerbatim()
        {
            NDArrayPythonInterop.RegisterCodec();          // Auto both ways; once per engine session

            NDArray nd;
            using (Py.GIL())
            {
                Scope.Exec("src = np.arange(6, dtype='f8')");
                using PyObject p = Scope.Get("src");

                nd = p.As<NDArray>();                      // Auto -> zero-copy VIEW
                nd[0] = (NDArray)42.0;                     // ...so this reaches Python
            }

            PyFloat("float(src[0])").Should().BeApproximately(42.0, 1e-12, "scope: src[0] == 42.0");
            nd.Dispose();
        }

        /// <summary>
        ///     "<c>TryDecodeView</c> returns null when no view is constructible — that <c>??</c> IS the
        ///     fallback": the same source decodes as a view under Auto and View, and the unviewable one
        ///     splits the two modes apart.
        /// </summary>
        [TestMethod]
        public void Auto_TheDoubleAttempt_IsWhatSeparatesAutoFromView()
        {
            var auto = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Auto });
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });

            // Viewable source: both modes succeed, both share.
            PyExec("dd = np.arange(4, dtype='f8')");
            foreach (var codec in new[] { auto, viewOnly })
            {
                NDArray nd;
                using (Gil()) { using PyObject p = Scope.Eval("dd"); codec.TryDecode(p, out nd).Should().BeTrue(); }
                WriteAt(nd, -1.0, 0);
                PyFloat("float(dd[0])").Should().BeApproximately(-1.0, 1e-12);
                PyExec("dd[0] = 0.0");
                nd.Dispose();
            }

            // Unviewable source: Auto's ?? falls through to the copy, View declines.
            PyExec("du = np.array([1+2j, 3+4j], dtype='c8')");
            using (Gil())
            {
                using PyObject p = Scope.Eval("du");
                auto.TryDecode(p, out NDArray a).Should().BeTrue("Auto's ?? falls back to a copy");
                a.typecode.Should().Be(NPTypeCode.Complex);
                a.Dispose();

                viewOnly.TryDecode(p, out NDArray v).Should().BeFalse("View declines instead of copying");
                v.Should().BeNull();
            }
        }

        /// <summary>The "three modes" registration snippets and the mode table's three rows.</summary>
        [TestMethod]
        public void Auto_TheThreeModeSnippets_ProduceTheDocumentedOptions()
        {
            // default — share when possible, copy when not
            NumpyCodecOptions.Default.DecodeMode.Should().Be(NumpyCodecMode.Auto);

            // never share: a detached snapshot that never locks the Python object
            var copyOpts = new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy };
            copyOpts.DecodeMode.Should().Be(NumpyCodecMode.Copy);

            // always share, or fail loudly — for code that depends on shared memory
            var viewOpts = new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View };
            viewOpts.DecodeMode.Should().Be(NumpyCodecMode.View);

            // Table: Copy mode copies even when a view IS possible; View mode never silently copies.
            PyExec("tm = bytearray(b'abcd')");
            NDArray snapshot;
            using (Gil())
            {
                using PyObject p = Scope.Eval("tm");
                new NumpyCodec(copyOpts).TryDecode(p, out snapshot).Should().BeTrue();
            }

            WriteAt(snapshot, (byte)90, 0);
            PyLong("tm[0]").Should().Be(97, "Copy mode copies even when a view is possible");
            PyExec("tm.append(1)");   // and never locks the object
            snapshot.Dispose();
        }

        /// <summary>
        ///     "On encode a view and a copy have identical dtype coverage ... so Auto always yields a
        ///     view, and only Copy forces a detached array. Decimal ... falls through to pythonnet's
        ///     CLR-object wrapping rather than failing the conversion."
        /// </summary>
        [TestMethod]
        public void Auto_OnEncode_AlwaysYieldsAView_AndDecimalFallsThrough()
        {
            var auto = new NumpyCodec(new NumpyCodecOptions { EncodeMode = NumpyCodecMode.Auto });
            var nd = np.arange(4).astype(NPTypeCode.Double);

            using (Gil()) { using PyObject p = auto.TryEncode(nd); Scope.Set("ea", p); }
            PyExec("ea[0] = 7.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(7.0, 1e-12, "Auto encode always yields a view");

            using (Gil())
                auto.TryEncode(np.arange(3).astype(NPTypeCode.Decimal))
                    .Should().BeNull("Decimal falls through to CLR wrapping rather than failing");
        }

        // ============================  ## The three routes that produce a view  ==================

        /// <summary>Route 1 — C-contiguous exporters (any object).</summary>
        [TestMethod]
        public void Route1_ContiguousExporters_View()
        {
            PyExec("ba = bytearray(b'abcd')");
            NDArray v = ViewOf("ba");            // Byte[4], writable, shares bytes with `ba`

            v.typecode.Should().Be(NPTypeCode.Byte);
            v.size.Should().Be(4);
            v.Shape.IsWriteable.Should().BeTrue();
            WriteAt(v, (byte)122, 3);
            PyLong("ba[3]").Should().Be(122, "shares bytes with `ba`");
            v.Dispose();

            // "a contiguous numpy array, bytes, bytearray, array.array, a memoryview, a ctypes array,
            //  a BytesIO buffer" — the doc's own list, each proven to take route 1.
            PyExec("import io, ctypes\n" +
                   "r1_np = np.arange(4, dtype='f8')\n" +
                   "r1_by = b'abcd'\n" +
                   "r1_ba = bytearray(b'abcd')\n" +
                   "r1_aa = array.array('i', [1,2,3,4])\n" +
                   "r1_mv = memoryview(bytearray(b'abcd'))\n" +
                   "r1_ct = (ctypes.c_int * 4)()\n" +
                   "r1_io = io.BytesIO(b'abcd').getbuffer()");

            foreach (string name in new[] { "r1_np", "r1_by", "r1_ba", "r1_aa", "r1_mv", "r1_ct", "r1_io" })
            {
                NDArray one = ViewOf(name, allowReadonly: true);
                one.Should().NotBeNull($"{name} is a C-contiguous exporter and must take route 1");
                one.Dispose();
            }
        }

        /// <summary>Route 2 — non-contiguous numpy arrays, via <c>__array_interface__</c>.</summary>
        [TestMethod]
        public void Route2_NonContiguousNumpy_ViewsViaArrayInterface()
        {
            PyExec("base = np.arange(20, dtype='i8')");

            var even = ViewOf("base[::2]");    // stride 2, shape (10,)
            var rev = ViewOf("base[::-1]");    // NEGATIVE stride
            var tr = ViewOf("np.arange(6).reshape(2,3).T");   // F-order strided

            even.shape.Should().BeEquivalentTo(new[] { 10 });
            even.Shape.Strides[0].Should().Be(2, "numpy's byte strides become NumSharp element strides");

            rev.shape.Should().BeEquivalentTo(new[] { 20 });
            rev.Shape.Strides[0].Should().Be(-1, "a reversed source keeps its NEGATIVE stride");
            ReadAt<long>(rev, 0).Should().Be(19, "the normalized window still addresses element 0 correctly");

            // The page's window-normalization block, checked on its own arithmetic:
            //   extent = (20 - 1) * -1 = -19  -> minOffset = -19, maxOffset = 0
            //   spanElements = 0 - (-19) + 1 = 20
            //   shape        = new Shape(dims, elemStrides, offset: -minOffset, ...)
            rev.Shape.Offset.Should().Be(19,
                "offset: -minOffset — the base pointer was shifted down to the lowest touched element");
            ReadAt<long>(rev, 19).Should().Be(0, "and the last logical element is the buffer's first");

            tr.shape.Should().BeEquivalentTo(new[] { 3, 2 });

            // All three are real views: a write crosses back.
            WriteAt(even, -5L, 1);
            PyLong("int(base[2])").Should().Be(-5);
            WriteAt(rev, -6L, 0);
            PyLong("int(base[19])").Should().Be(-6, "negative strides address the same buffer");

            even.Dispose(); rev.Dispose(); tr.Dispose();
        }

        /// <summary>Route 3 — non-contiguous non-numpy exporters, run verbatim.</summary>
        [TestMethod]
        public void Route3_NonContiguousNonNumpy_ViewsViaBufferStrides()
        {
            PyExec("ba = bytearray(range(16))");

            var even = ViewOf("memoryview(ba)[::2]");    // [0,2,4,...] — shared
            var odd = ViewOf("memoryview(ba)[1::2]");    // offset included in the pointer
            var rev = ViewOf("memoryview(ba)[::-1]");    // negative stride

            ReadAt<byte>(even, 0).Should().Be(0);
            ReadAt<byte>(odd, 0).Should().Be(1, "the offset is folded into the pointer");
            ReadAt<byte>(rev, 0).Should().Be(15, "reversed");

            even[3] = (NDArray)(byte)99;                 // logical 3 -> byte 6
            PyLong("ba[6]").Should().Be(99, "python: ba[6] == 99");

            even.Dispose(); odd.Dispose(); rev.Dispose();
        }

        // ============================  ## What cannot be viewed  =================================

        /// <summary>complex64 — "the values must be widened, and widening is a copy".</summary>
        [TestMethod]
        public void Unviewable_Complex64_CopiesAndWidens()
        {
            PyExec("c = np.array([1+2j, 3+4j], dtype='c8')");
            NDArray nd = ImportOf("c");        // Complex (complex128), values preserved, independent

            nd.typecode.Should().Be(NPTypeCode.Complex);
            ReadAt<Complex>(nd, 0).Should().Be(new Complex(1, 2), "values preserved");
            ReadAt<Complex>(nd, 1).Should().Be(new Complex(3, 4));

            WriteAt(nd, new Complex(9, 9), 0);
            PyFloat("float(c[0].real)").Should().BeApproximately(1.0, 1e-12, "independent");
            nd.Dispose();

            // "numpy's complex64 is two 4-byte floats (8 bytes); NumSharp's Complex is two 8-byte
            //  doubles (16 bytes). There is no reinterpretation."
            PyLong("np.dtype('c8').itemsize").Should().Be(8);
            System.Runtime.InteropServices.Marshal.SizeOf<Complex>().Should().Be(16);
        }

        /// <summary>
        ///     Big-endian multi-byte is rejected; "single-byte big-endian dtypes (>i1, |u1, |b1) DO
        ///     view — byte order is meaningless at one byte wide".
        /// </summary>
        [TestMethod]
        public void Unviewable_BigEndianMultiByte_Rejected_ButSingleByteViews()
        {
            PyExec("be4 = np.arange(4, dtype='>i4')");
            ((Action)(() => ViewOf("be4", allowReadonly: true).Dispose()))
                .Should().Throw<Exception>("viewing >i4 on a little-endian machine would byte-swap every value");
            ((Action)(() => ImportOf("be4").Dispose()))
                .Should().Throw<Exception>("the copy path refuses it too rather than silently misreading");

            // The documented Python-side fix.
            PyExec("le4 = be4.astype(be4.dtype.newbyteorder('<'))");
            NDArray ok = ViewOf("le4");
            ReadAt<int>(ok, 3).Should().Be(3);
            ok.Dispose();

            // Single-byte big-endian dtypes still view.
            foreach (string dtype in new[] { ">i1", "|u1", "|b1" })
            {
                PyExec($"sb = np.zeros(4, dtype='{dtype}')");
                NDArray one = ViewOf("sb");
                one.Should().NotBeNull($"{dtype} is one byte wide — byte order is meaningless");
                one.Dispose();
            }
        }

        /// <summary>
        ///     Sub-item strides. The page's Python block is executed to prove the <c>[0, 65536]</c>
        ///     and the overlapping-bytes explanation, then the C# guard's exact message is asserted.
        /// </summary>
        [TestMethod]
        public void Unviewable_SubItemStride_ProducesTheDocumentedOverlap_AndIsGuarded()
        {
            // a = np.arange(4, dtype='i4')                                    # itemsize 4, normal stride 4
            // w = np.lib.stride_tricks.as_strided(a, shape=(2,), strides=(2,))  # stride 2 BYTES
            PyExec("a = np.arange(4, dtype='i4')\n" +
                   "w = np.lib.stride_tricks.as_strided(a, shape=(2,), strides=(2,))");

            PyStr("w.tolist()").Should().Be("[0, 65536]", "the doc prints w -> [0, 65536]");
            PyLong("a.itemsize").Should().Be(4, "itemsize 4, normal stride 4");
            PyLong("a.strides[0]").Should().Be(4);
            PyLong("w.strides[0]").Should().Be(2, "stride 2 BYTES");

            //   elem[0] = bytes[0:4] = 00 00 00 00 = 0
            //   elem[1] = bytes[2:6] = 00 00 01 00 = 65536   <- overlaps elem[0]
            PyStr("a.tobytes()[0:4].hex()").Should().Be("00000000");
            PyStr("a.tobytes()[2:6].hex()").Should().Be("00000100");

            // "2 / 4 = 0.5 elements is not expressible, so the guard is one line" — verbatim message.
            ((Action)(() => ViewOf("w", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>()
                .WithMessage("stride 2 bytes is not a multiple of itemsize 4; " +
                             "NumSharp strides are element-based. Use ToNDArray (copy).");

            // "In practice this is unreachable from ordinary code — normal slicing always yields
            //  element-multiple strides (a[::2] -> 8, a[1::3] -> 12, a[::-1] -> -4)."
            PyExec("o = np.arange(12, dtype='i4')");
            PyLong("o[::2].strides[0]").Should().Be(8);
            PyLong("o[1::3].strides[0]").Should().Be(12);
            PyLong("o[::-1].strides[0]").Should().Be(-4);

            // ...and Auto therefore copies it, rather than failing.
            NDArray copied = ImportOf("w");
            copied.size.Should().Be(2);
            ReadAt<int>(copied, 1).Should().Be(65536, "the copy linearises the overlapping window");
            copied.Dispose();
        }

        // ============================  ## Read-only sources still view  ==========================

        /// <summary>
        ///     The read-only snippet, plus the "useful consequence" the page draws from it:
        ///     <c>IsWriteable == false</c> is a reliable signal that you got the view path.
        /// </summary>
        [TestMethod]
        public void ReadOnlySources_StillView_AsNonWriteable()
        {
            PyExec("rob = b'abcd'");
            NDArray ro = ViewOf("rob", allowReadonly: true);

            ro.Shape.IsWriteable.Should().BeFalse();                 // false
            ((Action)(() => ro[0] = (NDArray)(byte)5))               // throws: assignment destination is read-only
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");
            ro.Dispose();

            // "By default (allowReadonly: false) the verb refuses such sources outright"
            ((Action)(() => ViewOf("rob").Dispose())).Should().Throw<InvalidOperationException>();

            // "since a copy always owns WRITABLE memory, IsWriteable == false is a reliable signal
            //  that you got the view path"
            NDArray copied = ImportOf("rob");
            copied.Shape.IsWriteable.Should().BeTrue("a copy always owns writable memory");
            copied.Dispose();

            // "The codec passes allowReadonly: true, because a non-writeable view is still a view"
            NDArray decoded;
            using (Gil())
            {
                using PyObject p = Scope.Eval("rob");
                new NumpyCodec(NumpyCodecOptions.Default).TryDecode(p, out decoded).Should().BeTrue();
            }

            decoded.Shape.IsWriteable.Should().BeFalse("the codec takes the non-writeable view, not a copy");
            decoded.Dispose();
        }

        // ============================  ## The trade-off: a live view locks the source  ===========

        /// <summary>The lock snippet, run line for line.</summary>
        [TestMethod]
        public void TradeOff_ALiveViewLocksTheSource()
        {
            PyExec("ba = bytearray(b'abcd')");
            var v = ViewOf("ba");

            // PyExec("ba.append(1)") -> BufferError: Existing exports of data: object cannot be re-sized
            using (Gil())
                Scope.Exec("try:\n    ba.append(1)\n    lockmsg = ''\nexcept BufferError as e:\n    lockmsg = str(e)");
            PyStr("lockmsg").Should().Be("Existing exports of data: object cannot be re-sized");

            v.Dispose();               // release the lease...
            Pump();
            PyExec("ba.append(1)");    // ...now fine
            PyLong("len(ba)").Should().Be(5);

            // "numpy is guarded the same way — arr.resize(refcheck=True) refuses while the lease exists."
            PyExec("nba = np.arange(8, dtype='f8')");
            NDArray nv = ViewOf("nba");
            using (Gil())
                Scope.Exec("try:\n    nba.resize((16,), refcheck=True)\n    nmsg = ''\nexcept ValueError as e:\n    nmsg = str(e)");
            PyStr("nmsg").Should().Contain("cannot resize an array that references or is referenced");
            nv.Dispose();
        }

        /// <summary>
        ///     "The lease is released when the LAST NumSharp view over the memory — including derived
        ///     slices like <c>nd["2:"]</c> — is disposed or garbage-collected."
        /// </summary>
        [TestMethod]
        public void TradeOff_TheLeaseOutlivesTheOriginal_UntilDerivedSlicesGo()
        {
            PyExec("dl = bytearray(range(8))");
            int baseline = NDArrayPythonInterop.LiveImports;

            NDArray original = ViewOf("dl");
            NDArray derived = original["2:"];          // a derived slice shares the lease

            original.Dispose();                        // disposing the ORIGINAL is not enough
            Pump();
            NDArrayPythonInterop.LiveImports.Should().Be(baseline + 1,
                "the refcount decides, not disposal order — a derived slice still holds the lease");
            ReadAt<byte>(derived, 0).Should().Be(2, "and the derived slice is still valid");

            derived.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == baseline).Should().BeTrue(
                "the lease is released when the LAST view over the memory goes");
        }

        // ============================  ## Measured coverage  =====================================

        /// <summary>
        ///     The census behind the page's headline number. Every exporter variety the coverage table
        ///     names is put through the <b>view</b> path; whatever declines falls to the <b>copy</b>
        ///     path; whatever both refuse is <b>rejected</b>. The totals asserted here are the totals
        ///     the documentation quotes — change one and this fails.
        /// </summary>
        [TestMethod]
        public void Coverage_Census_MeasuresTheDocumentedTotals()
        {
            PyExec("import io, ctypes");

            var varieties = new List<(string Category, string Name, string Expr)>();
            void Add(string category, string name, string expr) => varieties.Add((category, name, expr));

            // --- bytes / bytearray / memoryview / BytesIO.getbuffer()
            Add("builtin", "bytes", "b'abcd'");
            Add("builtin", "bytearray", "bytearray(b'abcd')");
            Add("builtin", "memoryview", "memoryview(bytearray(b'abcd'))");
            Add("builtin", "BytesIO.getbuffer", "io.BytesIO(b'abcd').getbuffer()");

            // --- array.array, all 12 typecodes
            foreach (char tc in "bBhHiIlLqQfd")
                Add("array.array", $"array.array('{tc}')", $"array.array('{tc}', [1,2,3,4])");

            // --- ctypes arrays
            foreach (string ct in new[] { "c_int", "c_double", "c_ubyte", "c_int16", "c_uint32", "c_int64" })
                Add("ctypes", $"ctypes.{ct}", $"(ctypes.{ct} * 4)()");

            // --- numpy dtypes
            foreach (string dt in new[] { "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8" })
                Add("numpy dtype", $"numpy '{dt}'", $"np.zeros(4, dtype='{dt}')");
            Add("numpy dtype", "numpy 'c16'", "np.zeros(4, dtype='c16')");
            Add("numpy dtype", "numpy '>i1' (single byte)", "np.zeros(4, dtype='>i1')");

            // --- numpy layouts
            Add("numpy layout", "contiguous", "np.arange(12, dtype='f8')");
            Add("numpy layout", "strided", "np.arange(12, dtype='f8')[::2]");
            Add("numpy layout", "reversed", "np.arange(12, dtype='f8')[::-1]");
            Add("numpy layout", "transposed", "np.arange(12, dtype='f8').reshape(3,4).T");
            Add("numpy layout", "F-order", "np.asfortranarray(np.arange(12, dtype='f8').reshape(3,4))");
            Add("numpy layout", "broadcast", "np.broadcast_to(np.arange(3, dtype='f8'), (2,3))");
            Add("numpy layout", "read-only", "np.arange(4, dtype='f8').view()");   // made read-only below
            Add("numpy layout", "0-d", "np.float64(2.5).reshape(())");

            // --- memoryview casts and strided/reversed forms
            Add("memoryview form", "cast('B')", "memoryview(np.arange(4, dtype='f8')).cast('B')");
            Add("memoryview form", "strided [::2]", "memoryview(bytearray(range(16)))[::2]");
            Add("memoryview form", "offset [1::2]", "memoryview(bytearray(range(16)))[1::2]");
            Add("memoryview form", "reversed [::-1]", "memoryview(bytearray(range(16)))[::-1]");

            // --- the genuinely unviewable ones
            Add("unviewable", "numpy complex64", "np.array([1+2j, 3+4j], dtype='c8')");
            Add("unviewable", "sub-item stride (as_strided)",
                "np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), shape=(2,), strides=(2,))");
            Add("unviewable", "big-endian multi-byte", "np.arange(4, dtype='>i4')");

            PyExec("ro_src = np.arange(4, dtype='f8')\nro_src.flags.writeable = False");
            varieties[varieties.FindIndex(v => v.Name == "read-only")] = ("numpy layout", "read-only", "ro_src");

            var results = new List<(string Category, string Name, string Outcome)>();
            foreach (var (category, name, expr) in varieties)
            {
                string outcome = Classify(expr);
                results.Add((category, name, outcome));
            }

            string report = string.Join("\n", results.Select(r => $"  {r.Outcome,-8} {r.Category,-16} {r.Name}"));
            Console.WriteLine($"exporter census ({results.Count} varieties):\n{report}");

            int views = results.Count(r => r.Outcome == "view");
            int copies = results.Count(r => r.Outcome == "copy");
            int rejected = results.Count(r => r.Outcome == "rejected");

            // ---- the numbers the documentation quotes -------------------------------------------
            results.Count.Should().Be(50, "the docs describe a census of 50 exporter varieties");
            views.Should().Be(47, "the docs claim 47 of the 50 varieties share memory");
            copies.Should().Be(2, "the docs claim exactly 2 varieties fall back to a copy");
            rejected.Should().Be(1, "big-endian multi-byte is refused by both paths — byte-swap first");

            // ---- and the table's per-category claims ---------------------------------------------
            foreach (string viewingCategory in new[] { "builtin", "array.array", "ctypes", "numpy dtype", "numpy layout", "memoryview form" })
                results.Where(r => r.Category == viewingCategory)
                       .Should().OnlyContain(r => r.Outcome == "view",
                           $"the coverage table marks every '{viewingCategory}' row as a view");

            results.Single(r => r.Name == "numpy complex64").Outcome.Should().Be("copy", "copy (widened)");
            results.Single(r => r.Name == "sub-item stride (as_strided)").Outcome.Should().Be("copy", "copy (linearised)");
            results.Single(r => r.Name == "big-endian multi-byte").Outcome.Should().Be("rejected", "byte-swap first");
        }

        /// <summary>Runs one exporter through the documented decision order and reports which path took it.</summary>
        private string Classify(string expr)
        {
            NDArray nd = null;
            try
            {
                nd = ViewOf(expr, allowReadonly: true);
                return "view";
            }
            catch
            {
                try
                {
                    nd = ImportOf(expr);
                    return "copy";
                }
                catch
                {
                    return "rejected";
                }
            }
            finally
            {
                nd?.Dispose();
            }
        }

        // ============================  ## Controlling the GIL  ===================================

        /// <summary>The hot-loop opt-out block and the process-wide switch, run verbatim.</summary>
        [TestMethod]
        public void Gil_TheOptOutBlock_RunsVerbatim()
        {
            var batches = new[] { np.arange(4).astype(NPTypeCode.Double), np.arange(4, 8).astype(NPTypeCode.Double) };
            var seen = new List<double>();
            Action<PyObject> consumer = p => { using var s = p.InvokeMethod("sum"); seen.Add(s.As<double>()); };

            using (Py.GIL())                                     // ONE acquisition...
                foreach (var batch in batches)
                    using (PyObject p = batch.ToNumpy(requireGIL: false))   // ...N conversions inside
                        consumer.Invoke(p);

            seen.Should().Equal(new[] { 6.0, 22.0 });

            // NDArrayPythonInterop.RequireGIL = false;          // or process-wide, when EVERY call site holds it
            bool saved = NDArrayPythonInterop.RequireGIL;
            try
            {
                NDArrayPythonInterop.RequireGIL = false;
                NDArrayPythonInterop.RequireGIL.Should().BeFalse();
                using (Py.GIL())
                    using (PyObject p = batches[0].ToNumpy())     // null -> follows the global
                        p.Should().NotBeNull();
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = saved;
            }

            NDArrayPythonInterop.RequireGIL.Should().BeTrue("the documented default is true");
        }

        // ============================  ## Choosing, in one table  ================================

        /// <summary>
        ///     The final "You want / Use" table. Each row names an API; this asserts each one exists
        ///     and does what the row promises (the shutdown row is proven by
        ///     <see cref="ShutdownLeakTests"/>, which is the only place a real engine death is observable).
        /// </summary>
        [TestMethod]
        public void Choosing_EveryRowOfTheFinalTable_NamesAWorkingApi()
        {
            // "Best of both, no thought required -> Auto (default)"
            NumpyCodecOptions.Default.DecodeMode.Should().Be(NumpyCodecMode.Auto);

            // "Python to fill a buffer you own -> View (a silent copy would be a bug)"
            PyExec("fill = np.zeros(4, dtype='f8')");
            NDArray shared;
            using (Gil())
            {
                using PyObject p = Scope.Eval("fill");
                new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View })
                    .TryDecode(p, out shared).Should().BeTrue();
            }

            PyExec("fill[:] = 3.0");
            ReadAt<double>(shared, 0).Should().BeApproximately(3.0, 1e-12, "Python filled the buffer we hold");
            shared.Dispose();

            // "A detached snapshot; never lock or touch the Python object -> Copy / ToNDArray"
            NDArray detached = ImportOf("fill");
            PyExec("fill[:] = 9.0");
            ReadAt<double>(detached, 0).Should().BeApproximately(3.0, 1e-12, "a snapshot never sees later writes");
            detached.Dispose();

            // "A view, deterministically released -> AsNDArray + Dispose()"
            int baseline = NDArrayPythonInterop.LiveImports;
            NDArray det;
            using (Gil()) { using PyObject p = Scope.Eval("fill"); det = p.AsNDArray(); }
            NDArrayPythonInterop.LiveImports.Should().Be(baseline + 1);
            det.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == baseline).Should().BeTrue("Dispose() makes it deterministic");
        }
    }
}
