using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for <c>docs/website-src/docs/interop/pythonnet-numpy.md</c> — the
    ///     package reference page: verbs, costs, layouts, lifetime, codec, GIL, dtypes, versions.
    ///
    ///     <para>Tests read no markdown and assert no prose (the law of <c>98e6045a</c>): each test
    ///     reproduces one code block, table or quoted error of the page and asserts the behaviour
    ///     the page claims. Tables that duplicate runtime knowledge (the version table, the dtype
    ///     table, the census totals) are encoded as test data and compared against the runtime
    ///     source of truth, so the two copies cannot disagree.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_PythonnetNumpyPage : InteropTestBase
    {
        // ============================  ## From zero to a shared array  ===========================

        /// <summary>
        ///     "`BeginAllowThreads` is what lets any thread convert afterwards — including threads
        ///     that have never touched Python."
        /// </summary>
        [TestMethod]
        public void Bootstrap_AFreshThreadConverts()
        {
            PythonEngine.IsInitialized.Should().BeTrue("the documented bootstrap ran in this process");

            var nd = np.arange(3).astype(NPTypeCode.Double);
            Exception failure = null;
            var t = new System.Threading.Thread(() =>
            {
                try
                {
                    using (Py.GIL()) { using var p = nd.ToNumpy(); }
                }
                catch (Exception e) { failure = e; }
            });
            t.Start();
            t.Join(30_000).Should().BeTrue();
            failure.Should().BeNull("a thread that never touched Python must be able to convert");
        }

        /// <summary>The quick-start block, run as written, with each transcript line asserted.</summary>
        [TestMethod]
        public void QuickStart_OneBufferBothSides()
        {
            var nd = np.arange(6).reshape(2, 3);

            NDArray copy, view;
            using (Py.GIL())
            {
                using var scope = Py.CreateScope();
                scope.Exec("import numpy as np");

                using (PyObject x = nd.ToNumpy())          // zero-copy view of NumSharp's buffer
                    scope.Set("x", x);

                scope.Exec("x[1, 2] = 99");                // Python writes...
                ReadAt<long>(nd, 1, 2).Should().Be(99, "...and NumSharp saw the write (np.arange is int64)");

                scope.Exec("r = np.sin(x / 3.0)");
                using PyObject r = scope.Get("r");
                copy = r.ToNDArray();                      // independent copy
                view = r.AsNDArray();                      // zero-copy view

                copy.typecode.Should().Be(NPTypeCode.Double, "np.sin computes float64");
                copy.shape.Should().Equal(2, 3);
                view.Shape.IsWriteable.Should().BeTrue("a view of a fresh numpy result is writable");

                WriteAt(view, -1.0, 0, 0);
                scope.Set("rr", r);
                using var probe = scope.Eval("float(rr[0, 0])");
                probe.As<double>().Should().BeApproximately(-1.0, 1e-12, "AsNDArray shares numpy's buffer");
            }

            copy.Dispose();
            view.Dispose();
        }

        // ============================  ## The four verbs  ========================================

        /// <summary>"A view shares later writes in both directions; a copy never does."</summary>
        [TestMethod]
        public void Verbs_ViewSharesAndCopyDetaches_BothDirections()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            using (Py.GIL())
            {
                using (PyObject v = nd.ToNumpy()) Scope.Set("v", v);
                using (PyObject c = nd.ToNumpyCopy()) Scope.Set("c", c);
                Scope.Exec("v[0] = 11.0");            // reaches nd
                Scope.Exec("c[1] = 22.0");            // does not
            }

            ReadAt<double>(nd, 0).Should().BeApproximately(11.0, 1e-12, "the view write reached NumSharp");
            ReadAt<double>(nd, 1).Should().BeApproximately(1.0, 1e-12, "the copy write did not");
            PyBool("np.shares_memory(v, c)").Should().BeFalse();

            // the import pair splits identically
            PyExec("src = np.arange(4, dtype='f8')");
            NDArray copied = ImportOf("src");
            copied.Shape.IsContiguous.Should().BeTrue("ToNDArray produces a fresh C-contiguous array");
            WriteAt(copied, -5.0, 0);
            PyFloat("float(src[0])").Should().BeApproximately(0.0, 1e-12, "ToNDArray does not share");
            copied.Dispose();

            NDArray leased = ViewOf("src");
            WriteAt(leased, -5.0, 0);
            PyFloat("float(src[0])").Should().BeApproximately(-5.0, 1e-12, "the view shares");
            leased.Dispose();
        }

        /// <summary>"`nd.ToNumpy(copy: true)` routes to `ToNumpyCopy`."</summary>
        [TestMethod]
        public void Verbs_CopyTrueRoutesToTheCopy()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                using (var p = nd.ToNumpy(copy: true)) Scope.Set("rc", p);
                using (var p = nd.ToNumpy()) Scope.Set("rv", p);
            }

            PyBool("np.shares_memory(rc, rv)").Should().BeFalse("copy: true means an independent array");
            PyExec("rc[0] = 8.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(0.0, 1e-12);
        }

        /// <summary>
        ///     "The naming follows numpy's own `array` / `asarray` split: `To…` copies, `As…` shares —
        ///     with `ToNumpy` as the deliberate exception" — plus `ToPython` as its alias.
        /// </summary>
        [TestMethod]
        public void Verbs_ToCopiesAndAsShares()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                using (var p = nd.ToPython()) Scope.Set("xp", p);
            }

            PyExec("xp[1] = 4.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(4.0, 1e-12, "ToPython aliases ToNumpy — the view exception");

            PyExec("py_src = np.arange(4, dtype='f8')");
            NDArray b, v, r;
            using (Gil())
            {
                using PyObject py = Scope.Eval("py_src");
                b = py.ToNDArray();                       // To… copies
                v = py.AsNDArray();                       // As… shares
                using PyObject ro = Scope.Eval("b'\\x01\\x02\\x03\\x04'");
                r = ro.AsNDArray(allowReadonly: true);    // a read-only exporter arrives non-writeable
            }

            WriteAt(b, -9.0, 0);
            PyFloat("float(py_src[0])").Should().BeApproximately(0.0, 1e-12, "To… copies");
            WriteAt(v, -9.0, 0);
            PyFloat("float(py_src[0])").Should().BeApproximately(-9.0, 1e-12, "As… shares");
            r.Shape.IsWriteable.Should().BeFalse();

            b.Dispose(); v.Dispose(); r.Dispose();
        }

        // ============================  ## What it costs  =========================================

        /// <summary>
        ///     "View verbs are flat in *n*; copy verbs are linear." Absolute timings are never gated
        ///     (spec §8.4) — the ordering across a 10⁴× size step is.
        /// </summary>
        [TestMethod]
        public void Costs_ViewVerbsAreFlat_CopyVerbsScale()
        {
            long BestTicks(Action a)
            {
                long best = long.MaxValue;
                for (int i = 0; i < 5; i++)
                {
                    var sw = Stopwatch.StartNew();
                    a();
                    sw.Stop();
                    if (i > 0) best = Math.Min(best, sw.ElapsedTicks);
                }

                return best;
            }

            var small = np.arange(1_000).astype(NPTypeCode.Double);
            var large = np.arange(10_000_000).astype(NPTypeCode.Double);
            using (Gil())
            {
                using (var p = small.ToNumpy()) Scope.Set("z_s", p);
                using (var p = large.ToNumpy()) Scope.Set("z_l", p);
            }

            long viewSmall = BestTicks(() => { using (Py.GIL()) { using var p = small.ToNumpy(); } });
            long viewLarge = BestTicks(() => { using (Py.GIL()) { using var p = large.ToNumpy(); } });
            long copySmall = BestTicks(() => { using (Py.GIL()) { using var p = small.ToNumpyCopy(); } });
            long copyLarge = BestTicks(() => { using (Py.GIL()) { using var p = large.ToNumpyCopy(); } });
            long leaseSmall = BestTicks(() => { using (Py.GIL()) { using var p = Scope.Eval("z_s"); using var v = NDArrayPythonInterop.ToNDArrayView(p, true); } });
            long leaseLarge = BestTicks(() => { using (Py.GIL()) { using var p = Scope.Eval("z_l"); using var v = NDArrayPythonInterop.ToNDArrayView(p, true); } });
            long importSmall = BestTicks(() => { using (Py.GIL()) { using var p = Scope.Eval("z_s"); using var c = NDArrayPythonInterop.ToNDArray(p); } });
            long importLarge = BestTicks(() => { using (Py.GIL()) { using var p = Scope.Eval("z_l"); using var c = NDArrayPythonInterop.ToNDArray(p); } });

            double ms(long t) => t * 1000.0 / Stopwatch.Frequency;
            string report = $"ToNumpy {ms(viewSmall):F4}->{ms(viewLarge):F4}, ToNumpyCopy {ms(copySmall):F4}->{ms(copyLarge):F4}, " +
                            $"ToNDArrayView {ms(leaseSmall):F4}->{ms(leaseLarge):F4}, ToNDArray {ms(importSmall):F4}->{ms(importLarge):F4} ms";

            (viewLarge < viewSmall * 20).Should().BeTrue($"ToNumpy must be flat in n ({report})");
            (leaseLarge < leaseSmall * 20).Should().BeTrue($"ToNDArrayView must be flat in n ({report})");
            (copyLarge > copySmall * 50).Should().BeTrue($"ToNumpyCopy must scale with n ({report})");
            (importLarge > importSmall * 50).Should().BeTrue($"ToNDArray must scale with n ({report})");

            PyExec("del z_s, z_l");
            large.Dispose();
        }

        // ============================  ## Exporting: every layout  ===============================

        /// <summary>Every row of the "NumSharp source → numpy sees" table.</summary>
        [TestMethod]
        public void Export_TableRows_KeepStridesAndFlags()
        {
            var b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);

            ExportTo("l_c", b);
            PyStr("l_c.strides").Should().Be("(48, 8)");
            PyBool("l_c.flags.c_contiguous").Should().BeTrue();

            ExportTo("l_rows", b["1:3"]);
            PyStr("l_rows.strides").Should().Be("(48, 8)");
            PyBool("np.shares_memory(l_rows, l_c)").Should().BeTrue("a row slice aliases the shared buffer");

            ExportTo("l_cols", b["1:3, 0:6:2"]);
            PyStr("l_cols.strides").Should().Be("(48, 16)");
            PyStr("l_cols.tolist()").Should().Be("[[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]");
            PyBool("np.shares_memory(l_cols, l_c)").Should().BeTrue();

            ExportTo("l_t", b.T);
            PyStr("l_t.strides").Should().Be("(8, 48)");
            PyBool("l_t.flags.f_contiguous").Should().BeTrue("a transpose of a C array is F-contiguous");

            ExportTo("l_r", b["::-1"]);
            PyBool("l_r.strides[0] < 0").Should().BeTrue("negative strides survive the export");
            PyStr("l_r[0].tolist()").Should().Be("[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]");

            ExportTo("l_f", np.asfortranarray(b));
            PyStr("l_f.strides").Should().Be("(8, 32)");
            PyBool("l_f.flags.f_contiguous").Should().BeTrue();
            PyBool("l_f.flags.c_contiguous").Should().BeFalse();

            ExportTo("l_b", np.broadcast_to(np.arange(3).astype(NPTypeCode.Double), new Shape(2, 3)));
            PyStr("l_b.strides").Should().Be("(0, 8)");
            PyBool("l_b.flags.writeable").Should().BeFalse("broadcast views are read-only on both sides");
            PyStr("l_b.tolist()").Should().Be("[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]");

            var scalar = np.array(2.5);
            ExportTo("l_0", scalar);
            PyLong("l_0.ndim").Should().Be(0);
            PyExec("l_0[()] = 7.5");
            ReadAt<double>(scalar).Should().BeApproximately(7.5, 1e-12, "a 0-d export writes through");

            ExportTo("l_e", np.zeros(new Shape(0, 3), NPTypeCode.Double));
            PyStr("l_e.shape").Should().Be("(0, 3)");
            PyStr("l_e.dtype").Should().Be("float64");
        }

        /// <summary>
        ///     "A numpy view chained onto a ctypes window over NumSharp's pointer" — the base chain,
        ///     `DummyArray` for strided exports, and `OWNDATA=False` everywhere.
        /// </summary>
        [TestMethod]
        public void Export_BaseChain_EndsAtTheCtypesWindow()
        {
            var b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);

            ExportTo("bc_full", b);
            PyStr("type(bc_full.base).__name__").Should().Be("ndarray", "reshape chains onto the flat frombuffer array");
            PyStr("type(bc_full.base.base).__name__").Should().Be("c_char_Array_192",
                "the deepest base is the ctypes window over NumSharp's pointer (24 x 8 bytes)");
            PyStr("bc_full.flags['OWNDATA']").Should().Be("False", "numpy agrees the memory is NumSharp's");

            ExportTo("bc_strided", b["1:3, 0:6:2"]);
            PyStr("type(bc_strided.base).__name__").Should().Be("DummyArray", "strided exports go through as_strided");
            PyStr("bc_strided.flags['OWNDATA']").Should().Be("False");
            PyBool("np.shares_memory(bc_full, bc_strided)").Should().BeTrue();
        }

        // ============================  ## Importing: three routes  ===============================

        /// <summary>"…takes one of three routes" — one representative per route, each proven a view by a crossing write.</summary>
        [TestMethod]
        public void Import_ThreeRoutes_EachYieldsAView()
        {
            // Route 1 — C-contiguous PEP 3118 exporter (any object)
            PyExec("r1 = bytearray(b'abcd')");
            NDArray v1 = ViewOf("r1");
            WriteAt(v1, (byte)90, 0);
            PyLong("r1[0]").Should().Be(90, "route 1: contiguous exporters view");
            v1.Dispose();

            // Route 2 — non-contiguous numpy array, via __array_interface__
            PyExec("r2 = np.arange(20, dtype='i8')");
            NDArray v2 = ViewOf("r2[::2]");
            v2.size.Should().Be(10);
            WriteAt(v2, -3L, 1);
            PyLong("int(r2[2])").Should().Be(-3, "route 2: strided numpy views via __array_interface__");
            v2.Dispose();

            // Route 3 — non-contiguous NON-numpy exporter (sliced memoryview) via PyBUF.STRIDED
            PyExec("r3 = bytearray(range(16))");
            NDArray v3 = ViewOf("memoryview(r3)[::2]");
            v3.size.Should().Be(8);
            WriteAt(v3, (byte)77, 3);       // logical 3 -> byte 6
            PyLong("r3[6]").Should().Be(77, "route 3: strided non-numpy exporters view via PyBUF.STRIDED");
            v3.Dispose();
        }

        /// <summary>
        ///     "The census, re-measured on every test run — 50 exporter varieties …: 47 view, 2 copy,
        ///     1 rejected." The totals the page quotes are what this test measures.
        /// </summary>
        [TestMethod]
        public void Import_Census_47Of50VarietiesView()
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
            Add("numpy layout", "read-only", "ro_src");
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
            // big-endian multi-byte is no longer a census REJECT — the copy path byteswaps it (covered by
            // ImportTests.Copy_BigEndianSource_ByteSwapsToNative + the viewability matrix). The census's one
            // rejection is now datetime64, whose element type has no NumSharp dtype at ANY byte order.
            Add("unviewable", "datetime64 (no NumSharp dtype)", "np.array(['2021-01-01'], dtype='M8[D]')");

            PyExec("ro_src = np.arange(4, dtype='f8')\nro_src.flags.writeable = False");

            var results = new List<(string Category, string Name, string Outcome)>();
            foreach (var (category, name, expr) in varieties)
                results.Add((category, name, Classify(expr)));

            string report = string.Join("\n", results.Select(r => $"  {r.Outcome,-8} {r.Category,-16} {r.Name}"));
            Console.WriteLine($"exporter census ({results.Count} varieties):\n{report}");

            int views = results.Count(r => r.Outcome == "view");
            int copies = results.Count(r => r.Outcome == "copy");
            int rejected = results.Count(r => r.Outcome == "rejected");

            // ---- the numbers the page quotes ----------------------------------------------------
            results.Count.Should().Be(50, "the page describes a census of 50 exporter varieties");
            views.Should().Be(47, "the page claims 47 of the 50 varieties share memory");
            copies.Should().Be(2, "the page claims exactly 2 varieties fall back to a copy");
            rejected.Should().Be(1, "datetime64 has no NumSharp dtype at any byte order");

            results.Single(r => r.Name == "numpy complex64").Outcome.Should().Be("copy", "widened");
            results.Single(r => r.Name == "sub-item stride (as_strided)").Outcome.Should().Be("copy", "linearized");
            results.Single(r => r.Name == "datetime64 (no NumSharp dtype)").Outcome.Should().Be("rejected", "no NumSharp dtype");
        }

        /// <summary>Runs one exporter through the decision order and reports which path took it.</summary>
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

        /// <summary>The read-only refusal, verbatim, and the non-writeable opt-in view.</summary>
        [TestMethod]
        public void Import_ReadonlyRefusedByDefault_OptInIsNonWriteable()
        {
            ((Action)(() => ViewOf("b'abcd'").Dispose()))
                .Should().Throw<InvalidOperationException>()
                .WithMessage("*the exporter's buffer is read-only*");

            NDArray ro = ViewOf("b'abcd'", allowReadonly: true);
            ro.Shape.IsWriteable.Should().BeFalse("numpy's writeable=False, carried across");
            ((Action)(() => ro[0] = (NDArray)(byte)5))
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");
            ro.Dispose();
        }

        /// <summary>
        ///     "An import view has numpy's `owndata == False` semantics … `np.require(view,
        ///     requirements: "O")` is the escape hatch."
        /// </summary>
        [TestMethod]
        public void ImportView_OwnsNothing_ResizeRefusesAndRequireOCopies()
        {
            PyExec("own = np.arange(4, dtype='f8')");
            NDArray view = ViewOf("own");

            ((Action)(() => view.resize(new Shape(8))))
                .Should().Throw<Exception>()
                .WithMessage("*cannot resize this array: it does not own its data*");

            NDArray owned = np.require(view, (Type)null, "O");
            owned.Should().NotBeSameAs(view);
            WriteAt(owned, -4.0, 0);
            PyFloat("float(own[0])").Should().BeApproximately(0.0, 1e-12, "the owning copy is detached");

            owned.Dispose();
            view.Dispose();
        }

        // ============================  ## Lifetime  ==============================================

        /// <summary>"Dispose the `NDArray`, drop the wrapper, run the GC — Python still reads and writes valid memory."</summary>
        [TestMethod]
        public void Lifetime_ExportOutlivesItsSource()
        {
            int pins = NDArrayPythonInterop.LiveExports;

            ExportAndForget();
            Pump();

            NDArrayPythonInterop.LiveExports.Should().Be(pins + 1, "the Python view pins the buffer");
            PyStr("lex.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0]");
            PyExec("lex[0] = 7.0");
            PyStr("lex.tolist()").Should().Be("[7.0, 1.0, 2.0, 3.0]", "still writable after every C# reference died");

            PyExec("del lex");
            WaitFor(() => NDArrayPythonInterop.LiveExports == pins).Should().BeTrue();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void ExportAndForget()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            ExportTo("lex", nd);
            nd.Dispose();
        }

        /// <summary>
        ///     "The lease is released when the last NumSharp view over the memory — *including
        ///     derived slices* — is disposed or collected."
        /// </summary>
        [TestMethod]
        public void Lifetime_LeaseIsHeldByTheLastView()
        {
            PyExec("lh = bytearray(range(8))");
            int baseline = NDArrayPythonInterop.LiveImports;

            NDArray original = ViewOf("lh");
            NDArray derived = original["2:"];

            original.Dispose();
            Pump();
            NDArrayPythonInterop.LiveImports.Should().Be(baseline + 1, "the derived slice still holds the lease");
            ReadAt<byte>(derived, 0).Should().Be(2);

            derived.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == baseline).Should().BeTrue();
        }

        /// <summary>"Both directions are observable" — the counters move by exactly one per conversion.</summary>
        [TestMethod]
        public void Lifetime_CountersTrackBothDirections()
        {
            int pinned = NDArrayPythonInterop.LiveExports;
            int leased = NDArrayPythonInterop.LiveImports;

            var nd = np.arange(4).astype(NPTypeCode.Double);
            ExportTo("lc", nd);
            NDArrayPythonInterop.LiveExports.Should().Be(pinned + 1);

            PyExec("lc_src = np.arange(4, dtype='f8')");
            NDArray v = ViewOf("lc_src");
            NDArrayPythonInterop.LiveImports.Should().Be(leased + 1);

            v.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == leased).Should().BeTrue();
            PyExec("del lc");
            WaitFor(() => NDArrayPythonInterop.LiveExports == pinned).Should().BeTrue();
        }

        /// <summary>
        ///     "Reallocation, on both sides": the `BufferError`, numpy's refcheck refusal, and
        ///     NumSharp's own resize refusal while exported — each verbatim.
        /// </summary>
        [TestMethod]
        public void Lifetime_ALiveConversionLocksResizing_BothSides()
        {
            // NumSharp view over a bytearray -> CPython blocks the resize
            PyExec("lba = bytearray(b'abcd')");
            NDArray held = ViewOf("lba");
            using (Gil())
                Scope.Exec("try:\n    lba.append(1)\n    lmsg = ''\nexcept BufferError as e:\n    lmsg = str(e)");
            PyStr("lmsg").Should().Be("Existing exports of data: object cannot be re-sized");

            // NumSharp view over a numpy array -> numpy's refcheck blocks the resize
            PyExec("lnp = np.arange(4, dtype='f8')");
            NDArray held2 = ViewOf("lnp");
            using (Gil())
                Scope.Exec("try:\n    lnp.resize((8,), refcheck=True)\n    rmsg = ''\nexcept ValueError as e:\n    rmsg = str(e)");
            PyStr("rmsg").Should().Contain("cannot resize an array that references or is referenced");

            held.Dispose();
            held2.Dispose();
            Pump();

            // mirror image: a Python export pins the NumSharp buffer against resize
            var nd = np.arange(8).astype(NPTypeCode.Double);
            ExportTo("lex2", nd);
            ((Action)(() => nd.resize(new Shape(16))))
                .Should().Throw<Exception>()
                .WithMessage("*cannot resize an array that references or is referenced*");
            PyExec("del lex2");
        }

        // ============================  ## The codec  =============================================

        /// <summary>"Register once, then every pythonnet boundary converts by itself" — the page block.</summary>
        [TestMethod]
        public void Codec_RegisterOnce_ThenSetAndAsJustWork()
        {
            NDArrayPythonInterop.RegisterCodec();          // once per engine session; idempotent
            NDArrayPythonInterop.RegisterCodec().Should().BeFalse("a second registration in one session is a no-op");

            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                Scope.Set("c", nd);                        // auto-encoded: a shared numpy view
                Scope.Exec("c[1] = 88.5");
                using PyObject r = Scope.Eval("np.sqrt(c)");
                Scope.Set("sq", r);
                using NDArray back = r.As<NDArray>();      // auto-decoded: a shared view of r
                back.typecode.Should().Be(NPTypeCode.Double);
                back.Shape.IsWriteable.Should().BeTrue();

                WriteAt(back, -1.0, 0);
            }

            PyStr("type(c).__name__").Should().Be("ndarray", "the encode produced a real numpy array");
            PyStr("c.flags['OWNDATA']").Should().Be("False", "…that borrows NumSharp's memory");
            ReadAt<double>(nd, 1).Should().BeApproximately(88.5, 1e-12, "the Python write reached NumSharp");
            PyFloat("float(sq[0])").Should().BeApproximately(-1.0, 1e-12, "the decode was a view of np.sqrt's result");
        }

        /// <summary>
        ///     "pythonnet caches the decoder lookup per (Python type, target type) pair, and it caches
        ///     misses." A ctypes array type's tp_name carries element AND length (`c_short_Array_7`),
        ///     so the probe pair below is touched by no other test — poisoning it, when this test runs
        ///     before any registration, harms nobody; and when the codec is already registered, the
        ///     same probe proves fresh-pair decoding instead. Both branches assert real behaviour.
        /// </summary>
        [TestMethod]
        public void Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned()
        {
            PyExec("import ctypes");

            string preRegistrationFailure = null;
            using (Gil())
            {
                using PyObject probe = Scope.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
                try
                {
                    using NDArray v = probe.As<NDArray>();
                }
                catch (InvalidCastException e)
                {
                    preRegistrationFailure = e.Message;
                }
            }

            bool weRegistered = NDArrayPythonInterop.RegisterCodec();

            if (preRegistrationFailure is null)
            {
                // Another test registered first, so the decode above already succeeded — the miss
                // cannot be reproduced this session, and registration must report already-done.
                weRegistered.Should().BeFalse("the probe decode can only have succeeded through a registered codec");
            }
            else
            {
                // We were first: the miss is cached. Registration does NOT heal the poisoned pair.
                weRegistered.Should().BeTrue();
                using (Gil())
                {
                    using PyObject again = Scope.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
                    ((Action)(() => again.As<NDArray>().Dispose()))
                        .Should().Throw<InvalidCastException>(
                            "pythonnet cached the pre-registration miss for this exact Python type");
                }
            }

            // A pair first touched AFTER registration decodes fine — c_short_Array_8 is a DIFFERENT
            // Python type from c_short_Array_7.
            using (Gil())
            {
                using PyObject fresh = Scope.Eval("(ctypes.c_short * 8)(1, 2, 3, 4, 5, 6, 7, 8)");
                using NDArray nd = fresh.As<NDArray>();
                nd.typecode.Should().Be(NPTypeCode.Int16);
                ReadAt<short>(nd, 7).Should().Be(8);
            }
        }

        /// <summary>
        ///     The modes table: `Auto` falls back to a copy, `View` declines, `Copy` detaches and
        ///     never locks — and `Decimal` encode converts to a float64 numpy array in every mode.
        /// </summary>
        [TestMethod]
        public void Codec_Modes_AutoViewCopy_AndTheDecimalConversion()
        {
            var auto = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Auto });
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });

            // Auto: complex64 has no view — the fallback copy takes it. View: declined.
            PyExec("cm = np.array([1+2j], dtype='c8')");
            using (Gil())
            {
                using PyObject p = Scope.Eval("cm");
                auto.TryDecode(p, out NDArray a).Should().BeTrue("Auto falls back to a copy");
                a.typecode.Should().Be(NPTypeCode.Complex);
                a.Dispose();

                viewOnly.TryDecode(p, out NDArray v).Should().BeFalse("View declines instead of silently copying");
                v.Should().BeNull();
            }

            // Copy: detached snapshot, no lock on the source.
            PyExec("cb = bytearray(b'abcd')");
            NDArray snap;
            using (Gil())
            {
                using PyObject p = Scope.Eval("cb");
                copyOnly.TryDecode(p, out snap).Should().BeTrue();
            }

            WriteAt(snap, (byte)90, 0);
            PyLong("cb[0]").Should().Be(97, "Copy mode never shares memory");
            PyExec("cb.append(1)");   // never locked at all
            snap.Dispose();

            // Decimal encode: converts to a float64 numpy array — in the instance codec and end to end.
            using (Gil())
            {
                using PyObject enc = auto.TryEncode(np.arange(3).astype(NPTypeCode.Decimal));
                enc.Should().NotBeNull("Decimal encodes as float64 — no numpy decimal dtype exists");
                Scope.Set("dcm", enc);
            }
            PyStr("dcm.dtype").Should().Be("float64");

            NDArrayPythonInterop.RegisterCodec();
            var dec = np.arange(3).astype(NPTypeCode.Decimal);
            using (Gil())
                Scope.Set("dput", dec);
            PyStr("type(dput).__name__").Should().Be("ndarray", "the codec crosses Decimal as a float64 numpy array");
            PyStr("dput.dtype").Should().Be("float64");
        }

        // ============================  ## The GIL  ===============================================

        /// <summary>The hot-loop block: one acquisition outside, `requireGIL: false` conversions inside.</summary>
        [TestMethod]
        public void Gil_OneAcquisitionManyConversions()
        {
            var batches = new[]
            {
                np.arange(4).astype(NPTypeCode.Double),
                np.arange(4, 8).astype(NPTypeCode.Double),
                np.arange(8, 12).astype(NPTypeCode.Double),
            };

            var seen = new List<double>();
            Action<PyObject> consumer = p => { using var s = p.InvokeMethod("sum"); seen.Add(s.As<double>()); };

            using (Py.GIL())                                              // ONE acquisition...
                foreach (var batch in batches)
                    using (PyObject p = batch.ToNumpy(requireGIL: false))  // ...N conversions inside
                        consumer.Invoke(p);

            seen.Should().Equal(new[] { 6.0, 22.0, 38.0 }, "every conversion and computation ran under one acquisition");
            NDArrayPythonInterop.RequireGIL.Should().BeTrue("the process-wide default is true");
        }

        // ============================  ## Dtypes  ================================================

        /// <summary>
        ///     Every row of the dtype table in both directions, plus the four specials: complex64
        ///     widens on copy, UCS-4 narrows on copy (BMP only), big-endian is refused with the fix,
        ///     and 8-byte long double views as Double.
        /// </summary>
        [TestMethod]
        public void Dtypes_EveryRowRoundTrips()
        {
            var table = new (NPTypeCode Ns, string Numpy)[]
            {
                (NPTypeCode.Boolean, "|b1"),
                (NPTypeCode.Byte, "|u1"), (NPTypeCode.SByte, "|i1"),
                (NPTypeCode.Int16, "<i2"), (NPTypeCode.UInt16, "<u2"),
                (NPTypeCode.Int32, "<i4"), (NPTypeCode.UInt32, "<u4"),
                (NPTypeCode.Int64, "<i8"), (NPTypeCode.UInt64, "<u8"),
                (NPTypeCode.Half, "<f2"), (NPTypeCode.Single, "<f4"), (NPTypeCode.Double, "<f8"),
                (NPTypeCode.Complex, "<c16"),
                (NPTypeCode.Char, "<u2"),
            };

            foreach (var (ns, numpy) in table)
            {
                NDArrayPythonInterop.ToNumpyDtypeStr(ns).Should().Be(numpy, $"the table maps {ns} -> {numpy}");
                if (ns != NPTypeCode.Char)   // Char shares <u2; the reverse map yields UInt16
                    NDArrayPythonInterop.FromNumpyDtypeStr(numpy).Should().Be(ns);
            }

            NDArrayPythonInterop.FromNumpyDtypeStr("<u2").Should().Be(NPTypeCode.UInt16);

            // complex64 widens on copy; no view.
            PyExec("dc8 = np.array([1+2j], dtype='c8')");
            NDArray widened = ImportOf("dc8");
            widened.typecode.Should().Be(NPTypeCode.Complex);
            widened.Dispose();

            // 2-byte wchar IS UTF-16 and views as Char; UCS-4 narrows on copy, astral refuses.
            NDArrayPythonInterop.FromBufferFormat("u", 2).Should().Be(NPTypeCode.Char);
            PyExec("du1 = np.array(['a', 'b'], dtype='U1')");
            NDArray narrowed = ImportOf("du1");
            narrowed.typecode.Should().Be(NPTypeCode.Char);
            ReadAt<char>(narrowed, 1).Should().Be('b');
            narrowed.Dispose();
            PyExec("dastral = np.array(['\\U0001F600'], dtype='U1')");
            ((Action)(() => ImportOf("dastral").Dispose()))
                .Should().Throw<NotSupportedException>("an astral code point needs a surrogate pair");

            // big-endian: rejected with the byte-swap fix — which works.
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">i4")))
                .Should().Throw<NotSupportedException>().WithMessage("*newbyteorder('<')*");
            PyExec("dbe = np.arange(4, dtype='>i4')\ndle = dbe.astype(dbe.dtype.newbyteorder('<'))");
            NDArray fixedUp = ImportOf("dle");
            fixedUp.typecode.Should().Be(NPTypeCode.Int32);
            ReadAt<int>(fixedUp, 2).Should().Be(2);
            fixedUp.Dispose();

            // long double: 8-byte (MSVC) is IEEE double; wider has no NumSharp dtype.
            NDArrayPythonInterop.FromBufferFormat("g", 8).Should().Be(NPTypeCode.Double);
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("g", 16)))
                .Should().Throw<NotSupportedException>().WithMessage("*astype(np.float64)*");
        }

        // ============================  ## Versions  ==============================================

        /// <summary>
        ///     "The mapping below is read out of each release's own `MaxSupportedVersion`" — the
        ///     page's table compared against the guard's own mapping, so they cannot drift apart.
        /// </summary>
        [TestMethod]
        public void Versions_TableIsTheGuardsOwnMapping()
        {
            var documented = new Dictionary<string, string>
            {
                ["3.7"] = "3.0.0", ["3.8"] = "3.0.0", ["3.9"] = "3.0.0", ["3.10"] = "3.0.0",
                ["3.11"] = "3.0.1",
                ["3.12"] = "3.0.3",
                ["3.13"] = "3.0.5",
                ["3.14"] = "3.1.0",
            };

            foreach (var (python, minimum) in documented)
                PythonRuntimeInterop.MinimumPythonnetFor(Version.Parse(python))
                    .Should().Be(minimum, $"the table maps Python {python} to pythonnet {minimum}");

            PythonRuntimeInterop.MinimumPythonnetFor(new Version(3, 99)).Should().BeNull("newer than anything known: no advice rather than wrong advice");
            PythonRuntimeInterop.MinimumPythonnetFor(new Version(4, 0)).Should().BeNull();
        }

        // ============================  ## Troubleshooting  =======================================

        /// <summary>
        ///     Every Troubleshooting row whose symptom quotes a message this bridge itself raises —
        ///     asserted verbatim, so a reworded exception cannot silently invalidate the table.
        /// </summary>
        [TestMethod]
        public void Troubleshooting_SymptomsAreVerbatim()
        {
            // "the exporter's buffer is read-only"
            ((Action)(() => ViewOf("b'ab'").Dispose()))
                .Should().Throw<InvalidOperationException>().WithMessage("*the exporter's buffer is read-only*");

            // "assignment destination is read-only"
            NDArray ro = ViewOf("b'ab'", allowReadonly: true);
            ((Action)(() => ro[0] = (NDArray)(byte)1))
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");
            ro.Dispose();

            // "Existing exports of data: object cannot be re-sized"
            PyExec("tba = bytearray(b'abcd')");
            NDArray held = ViewOf("tba");
            using (Gil())
                Scope.Exec("try:\n    tba.append(1)\n    tmsg = ''\nexcept BufferError as e:\n    tmsg = str(e)");
            PyStr("tmsg").Should().Be("Existing exports of data: object cannot be re-sized");
            held.Dispose();
            Pump();

            // "cannot resize an array that references or is referenced by another array in this way"
            // (the raised message carries NumPy's own line wrap after "referenced", so it is
            // asserted as its two fragments)
            var nd = np.arange(8).astype(NPTypeCode.Double);
            ExportTo("t_ex", nd);
            ((Action)(() => nd.resize(new Shape(16))))
                .Should().Throw<Exception>()
                .WithMessage("*cannot resize an array that references or is referenced*")
                .WithMessage("*by another array in this way*");
            PyExec("del t_ex");

            // "cannot resize this array: it does not own its data"
            PyExec("t_own = np.arange(4, dtype='f8')");
            NDArray importView = ViewOf("t_own");
            ((Action)(() => importView.resize(new Shape(8))))
                .Should().Throw<Exception>().WithMessage("*cannot resize this array: it does not own its data*");
            importView.Dispose();

            // "big-endian dtype '>f8' cannot be shared with a native-endian NumSharp buffer"
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8")))
                .Should().Throw<NotSupportedException>()
                .WithMessage("big-endian dtype '>f8' cannot be shared with a native-endian NumSharp buffer.*");

            // "decimal has no numpy dtype (16-byte, non-IEEE)"
            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>().WithMessage("decimal has no numpy dtype (16-byte, non-IEEE).*");

            // "the object does not export a PEP 3118 buffer"
            ((Action)(() => ViewOf("{'a': 1}", allowReadonly: true).Dispose()))
                .Should().Throw<NotSupportedException>().WithMessage("the object does not export a PEP 3118 buffer*");

            // "Python engine is not initialized" — cannot be triggered while the engine runs; its
            // wording is pinned by the source and exercised in a session-death context by
            // ShutdownLeakTests, so it is deliberately not re-asserted here.
        }
    }
}
