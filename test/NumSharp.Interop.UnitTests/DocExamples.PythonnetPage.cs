using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for every code example, table row and quoted error string on
    ///     <c>docs/website-src/docs/interop/pythonnet.md</c>.
    ///
    ///     <para>Documentation examples are what users copy-paste, so they are held to the same
    ///     standard as the library: each test below reproduces one doc snippet as literally as the
    ///     harness allows and asserts exactly what the surrounding prose promises. A doc claim that
    ///     stops being true fails here — that is the entire point of the file.</para>
    ///
    ///     <para>Test names carry the doc section they pin (<c>QuickStart_</c>, <c>FourVerbs_</c>,
    ///     <c>LayoutFidelity_</c>, ...) so a failure names the paragraph to fix.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_PythonnetPage : InteropTestBase
    {
        // ============================  ## Installation  ==========================================

        /// <summary>
        ///     The page opens with the three-line bootstrap. It cannot be re-run (CPython does not
        ///     support re-initialization), but <see cref="PythonSession"/> performs exactly those
        ///     three calls, so the documented sequence is what this very process booted with.
        /// </summary>
        [TestMethod]
        public void Installation_TheDocumentedBootstrap_IsWhatThisProcessRan()
        {
            // Runtime.PythonDLL = @"C:\Python312\python312.dll";
            // PythonEngine.Initialize();
            // PythonEngine.BeginAllowThreads();
            PythonEngine.IsInitialized.Should().BeTrue("PythonEngine.Initialize() is the documented entry point");
            Runtime.PythonDLL.Should().NotBeNullOrEmpty("the docs require pointing pythonnet at a CPython shared library");

            // BeginAllowThreads released the GIL from the init thread — so a fresh thread can convert
            // without the caller ever having touched Python, which is what the page promises under
            // "Threading & the GIL".
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

        /// <summary>
        ///     The "Your Python / Minimum pythonnet" table. Every row is read straight out of the
        ///     guard's own mapping, so the table cannot drift from the advice the library gives.
        /// </summary>
        [TestMethod]
        public void Installation_VersionTable_MatchesTheGuardsMapping()
        {
            // | Your Python | Minimum pythonnet |
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
                    .Should().Be(minimum, $"the docs table maps Python {python} to pythonnet {minimum}");

            // "newer than anything we know about" -> no advice rather than a wrong one.
            PythonRuntimeInterop.MinimumPythonnetFor(new Version(3, 99)).Should().BeNull();
            PythonRuntimeInterop.MinimumPythonnetFor(new Version(4, 0)).Should().BeNull();
        }

        // The page also quotes the guard's error verbatim in a fenced block. Pinning it
        // character-for-character would mean restructuring the guard so its message could be built for
        // a pairing this process cannot be in (a session runs one Python against one pythonnet) —
        // production code bent for a test. Its one drift-prone part, the version it tells you to
        // install, is pinned instead by DocExamples_DocIntegrity against the mapping above.

        // ============================  ## Quick Start  ===========================================

        /// <summary>The first runnable sample on the page, executed line for line.</summary>
        [TestMethod]
        public void QuickStart_ExplicitCalls_BehaveExactlyAsDocumented()
        {
            var nd = np.arange(6).reshape(2, 3);

            NDArray copy, view;
            using (Py.GIL())
            {
                using var scope = Py.CreateScope();
                scope.Exec("import numpy as np");

                // NumSharp -> numpy: a zero-copy VIEW of NumSharp's buffer
                using (PyObject x = nd.ToNumpy())
                    scope.Set("x", x);

                scope.Exec("x[1, 2] = 99");        // Python writes...
                // ...NumSharp sees it: nd now holds 99 at (1, 2) — same memory.
                // (np.arange is Int64 in NumSharp, so the buffer is int64 on both sides.)
                ReadAt<long>(nd, 1, 2).Should().Be(99, "the doc's central claim: same memory");

                // numpy -> NumSharp
                using PyObject result = scope.Eval("np.sin(x / 3.0)");
                copy = result.ToNDArray();      // independent copy
                view = result.AsNDArray();      // zero-copy view (shared mutation)

                // the comments' semantics, verified:
                scope.Set("r", result);
                view.Shape.IsWriteable.Should().BeTrue();
                WriteAt(view, -1.0, 0, 0);
                using var probe = scope.Eval("float(r[0, 0])");
                probe.As<double>().Should().BeApproximately(-1.0, 1e-12, "AsNDArray shares memory");

                using var probeCopy = scope.Eval("float(r[0, 1])");
                double pythonSecond = probeCopy.As<double>();
                ReadAt<double>(copy, 0, 1).Should().BeApproximately(pythonSecond, 1e-12);
                WriteAt(copy, -2.0, 0, 1);
                using var probeCopy2 = scope.Eval("float(r[0, 1])");
                probeCopy2.As<double>().Should().BeApproximately(pythonSecond, 1e-12,
                    "ToNDArray is an independent copy — writing it must not reach Python");
            }

            copy.Dispose();
            view.Dispose();
        }

        /// <summary>The codec form of the same sample — "skip the explicit calls entirely".</summary>
        [TestMethod]
        public void QuickStart_CodecForm_ConvertsAtEveryBoundary()
        {
            NDArrayPythonInterop.RegisterCodec();       // Auto: share when possible, copy when not

            var nd = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);
            NDArray back;
            using (Py.GIL())
            {
                Scope.Set("x", nd);                                 // NDArray -> numpy view, automatically
                using PyObject r = Scope.Eval("np.sin(x / 3.0)");
                back = r.As<NDArray>();                             // numpy -> NDArray, automatically
            }

            back.Should().NotBeNull("the registered codec decodes numpy arrays with no explicit call");
            back.shape.Should().BeEquivalentTo(new[] { 2, 3 });
            back.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(back, 0, 1).Should().BeApproximately(Math.Sin(1 / 3.0), 1e-12);

            // "scope.Set("x", nd)" really was a VIEW: a python write reaches NumSharp.
            PyExec("x[0, 0] = 12.5");
            ReadAt<double>(nd, 0, 0).Should().BeApproximately(12.5, 1e-12, "auto-encode is a shared view");

            back.Dispose();
        }

        // ============================  ## The Four Verbs  ========================================

        /// <summary>The four-row verb table: each row's stated semantics, asserted.</summary>
        [TestMethod]
        public void FourVerbs_EachRowOfTheTable_HasTheStatedSemantics()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            // Row 1 — ToNumpy: zero-copy numpy view, shared mutation.
            using (Gil()) { using var p = NDArrayPythonInterop.ToNumpy(nd); Scope.Set("v1", p); }
            PyExec("v1[0] = 11.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(11.0, 1e-12, "ToNumpy = zero-copy view, shared mutation");

            // Row 2 — ToNumpyCopy: independent, no shared memory.
            using (Gil()) { using var p = NDArrayPythonInterop.ToNumpyCopy(nd); Scope.Set("v2", p); }
            PyExec("v2[1] = 22.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(1.0, 1e-12, "ToNumpyCopy = independent numpy array");
            PyBool("np.shares_memory(v1, v2)").Should().BeFalse();

            // Row 3 — ToNDArray: COPY any PEP 3118 exporter into a fresh C-contiguous NDArray.
            PyExec("src = np.arange(4, dtype='f8')");
            NDArray copied = ImportOf("src");
            copied.Shape.IsContiguous.Should().BeTrue("ToNDArray produces a fresh C-contiguous array");
            WriteAt(copied, -5.0, 0);
            PyFloat("float(src[0])").Should().BeApproximately(0.0, 1e-12, "ToNDArray does not share memory");
            copied.Dispose();

            // Row 4 — ToNDArrayView: zero-copy NDArray view over Python memory, exporter leased.
            int leasesBefore = NDArrayPythonInterop.LiveImports;
            NDArray leased = ViewOf("src");
            NDArrayPythonInterop.LiveImports.Should().Be(leasesBefore + 1, "a view leases the exporter");
            WriteAt(leased, -5.0, 0);
            PyFloat("float(src[0])").Should().BeApproximately(-5.0, 1e-12, "ToNDArrayView = shared mutation");
            leased.Dispose();
        }

        /// <summary>"<c>ToNumpy(nd, copy: true)</c> is a routing overload equivalent to <c>ToNumpyCopy</c>."</summary>
        [TestMethod]
        public void FourVerbs_ToNumpyCopyTrue_IsEquivalentToToNumpyCopy()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil())
            {
                using (var p = nd.ToNumpy(copy: true)) Scope.Set("rc", p);
                using (var p = nd.ToNumpy()) Scope.Set("rv", p);
            }

            PyBool("np.shares_memory(rc, rv)").Should().BeFalse("copy:true routes to ToNumpyCopy");
            PyExec("rc[0] = 8.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(0.0, 1e-12);
        }

        /// <summary>The "Plus ..." paragraph: ToMemoryView and the four dtype maps.</summary>
        [TestMethod]
        public void FourVerbs_ToMemoryViewAndTheDtypeMaps_Exist()
        {
            var nd = np.arange(3).astype(NPTypeCode.Double);
            ExportMemoryViewTo("mv", nd);
            PyBool("isinstance(mv, memoryview)").Should().BeTrue("ToMemoryView returns a Python memoryview");
            PyBool("mv.readonly == False").Should().BeTrue("the docs promise a WRITABLE memoryview of raw bytes");
            PyLong("len(mv)").Should().Be(24, "raw bytes: 3 x float64");

            NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Double).Should().Be("<f8");
            NDArrayPythonInterop.FromNumpyDtypeStr("<f8").Should().Be(NPTypeCode.Double);
            NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Double).Should().Be("d");
            NDArrayPythonInterop.FromBufferFormat("d", 8).Should().Be(NPTypeCode.Double);
        }

        /// <summary>
        ///     The "Extension methods" block, run verbatim, plus the stated naming convention
        ///     (<c>To…</c> copies, <c>As…</c> shares — with <c>ToNumpy</c> as the documented exception).
        /// </summary>
        [TestMethod]
        public void FourVerbs_ExtensionMethodBlock_RunsVerbatim()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            using (Gil())
            {
                PyObject a = nd.ToNumpy();          // zero-copy view (alias: nd.ToPython())
                PyObject c = nd.ToNumpyCopy();      // independent copy
                PyObject m = nd.ToMemoryView();     // raw-bytes memoryview
                Scope.Set("xa", a); Scope.Set("xc", c); Scope.Set("xm", m);
                a.Dispose(); c.Dispose(); m.Dispose();
                using (var alias = nd.ToPython()) Scope.Set("xp", alias);
            }

            // "ToNumpy is the exception — the zero-copy view is the package's headline"
            PyBool("np.shares_memory(xa, xc)").Should().BeFalse("ToNumpyCopy is the explicit copy");
            PyExec("xa[0] = 3.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(3.0, 1e-12, "ToNumpy shares");
            PyExec("xp[1] = 4.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(4.0, 1e-12, "ToPython is an alias of ToNumpy");

            PyExec("py_src = np.arange(4, dtype='f8')");
            NDArray b, v, r;
            using (Gil())
            {
                using PyObject py = Scope.Eval("py_src");
                b = py.ToNDArray();        // copy
                v = py.AsNDArray();        // zero-copy view
                using PyObject ro = Scope.Eval("b'\\x01\\x02\\x03\\x04'");
                r = ro.AsNDArray(allowReadonly: true);   // NON-WRITEABLE view of a read-only exporter
            }

            WriteAt(b, -9.0, 0);
            PyFloat("float(py_src[0])").Should().BeApproximately(0.0, 1e-12, "To… copies");
            WriteAt(v, -9.0, 0);
            PyFloat("float(py_src[0])").Should().BeApproximately(-9.0, 1e-12, "As… shares");
            r.Shape.IsWriteable.Should().BeFalse("a view of a read-only exporter is NON-WRITEABLE");

            b.Dispose(); v.Dispose(); r.Dispose();
        }

        // ============================  ## Layout Fidelity  =======================================

        /// <summary>
        ///     Every row of the "NumSharp source -> numpy result" table. <c>ToNumpy</c> exports
        ///     <b>every</b> NumSharp layout zero-copy, and the numpy side must see the exact same
        ///     strided window.
        /// </summary>
        [TestMethod]
        public void LayoutFidelity_EveryRowOfTheTable_ExportsAsDocumented()
        {
            var b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);

            // Row 1 — C-contiguous -> C-contiguous array
            ExportTo("l_c", b);
            PyBool("l_c.flags.c_contiguous").Should().BeTrue();

            // Row 2 — Sliced view (nd["1:3, ::2"]), any offset -> strided view over the same buffer
            ExportTo("l_s", b["1:3, ::2"]);
            PyStr("l_s.tolist()").Should().Be("[[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]");
            PyBool("np.shares_memory(l_s, l_c)").Should().BeTrue("a slice must alias, not copy");
            PyBool("l_c.strides[1] * 2 == l_s.strides[1]").Should().BeTrue("step 2 doubles the inner stride");

            // Row 3 — Transposed / Fortran-order -> F-order strided view
            ExportTo("l_t", b.T);
            PyStr("l_t.shape").Should().Be("(6, 4)");
            PyBool("l_t.flags.f_contiguous").Should().BeTrue("a transpose of a C array is F-contiguous");
            PyBool("np.shares_memory(l_t, l_c)").Should().BeTrue();

            // Row 4 — Reversed (nd["::-1"], negative strides) -> negative-stride view
            ExportTo("l_r", b["::-1"]);
            PyBool("l_r.strides[0] < 0").Should().BeTrue("negative strides export as negative strides");
            PyStr("l_r[0].tolist()").Should().Be("[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]");

            // Row 5 — Broadcast (stride-0) view -> READ-ONLY array (flags.writeable == False)
            ExportTo("l_b", np.broadcast_to(np.arange(3), new Shape(2, 3)));
            PyBool("l_b.flags.writeable").Should().BeFalse("matching NumSharp's own write protection");
            PyStr("l_b.tolist()").Should().Be("[[0, 1, 2], [0, 1, 2]]");

            // Row 6 — Scalar (0-d) -> 0-d array
            ExportTo("l_0", np.mean(np.arange(5)));
            PyLong("l_0.ndim").Should().Be(0);

            // Row 7 — Empty -> empty array of the right shape/dtype
            ExportTo("l_e", new NDArray(NPTypeCode.Single, new Shape(0, 3)));
            PyStr("l_e.shape").Should().Be("(0, 3)");
            PyStr("l_e.dtype").Should().Be("float32");
        }

        /// <summary>
        ///     "Imports are symmetric. <c>ToNDArrayView</c> has three zero-copy routes" — one
        ///     representative of each bullet, each proven to be a view by a write that crosses.
        /// </summary>
        [TestMethod]
        public void LayoutFidelity_AllThreeImportRoutes_ProduceViews()
        {
            // Route 1 — C-contiguous PEP 3118 exporter (any object)
            PyExec("r1 = bytearray(b'abcd')");
            NDArray v1 = ViewOf("r1");
            WriteAt(v1, (byte)90, 0);
            PyLong("r1[0]").Should().Be(90, "route 1: contiguous exporters view");
            v1.Dispose();

            // Route 2 — non-contiguous NUMPY array, via __array_interface__
            PyExec("r2 = np.arange(20, dtype='i8')");
            NDArray v2 = ViewOf("r2[::2]");
            v2.size.Should().Be(10);
            WriteAt(v2, -3L, 1);
            PyLong("int(r2[2])").Should().Be(-3, "route 2: strided numpy views via __array_interface__");
            v2.Dispose();

            // Route 3 — non-contiguous NON-numpy exporter (sliced memoryview)
            PyExec("r3 = bytearray(range(16))");
            NDArray v3 = ViewOf("memoryview(r3)[::2]");
            v3.size.Should().Be(8);
            WriteAt(v3, (byte)77, 3);       // logical 3 -> byte 6
            PyLong("r3[6]").Should().Be(77, "route 3: strided non-numpy exporters view via PyBUF.STRIDED");
            v3.Dispose();
        }

        /// <summary>
        ///     "Read-only sources are refused for views by default ... Pass <c>allowReadonly: true</c>
        ///     to opt in: the view comes back non-writeable ... so guarded write paths raise
        ///     <c>assignment destination is read-only</c>."
        /// </summary>
        [TestMethod]
        public void LayoutFidelity_ReadOnlySources_RefusedByDefault_NonWriteableWhenOptedIn()
        {
            ((Action)(() => ViewOf("b'abcd'").Dispose()))
                .Should().Throw<InvalidOperationException>()
                .WithMessage("*read-only*", "the docs promise a refusal, not a silent read-only view");

            NDArray ro = ViewOf("b'abcd'", allowReadonly: true);
            ro.Shape.IsWriteable.Should().BeFalse("carried as Shape.IsWriteable == false");
            ((Action)(() => ro[0] = (NDArray)(byte)5))
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");
            ro.Dispose();
        }

        /// <summary>
        ///     "Import views also do not own their data — like <c>np.frombuffer(...)</c>: a size-changing
        ///     <c>ndarray.resize</c> refuses ... and <c>np.require(..., "O")</c> produces an owning copy."
        /// </summary>
        [TestMethod]
        public void LayoutFidelity_ImportViewsDoNotOwnTheirData()
        {
            PyExec("own = np.arange(8, dtype='f8')");
            NDArray view = ViewOf("own");

            ((Action)(() => view.resize(new Shape(16))))
                .Should().Throw<Exception>().WithMessage("*does not own its data*",
                    "exactly NumPy's message for a non-owning array");

            NDArray owned = np.require(view, (Type)null, "O");
            owned.Should().NotBeSameAs(view, "require('O') must produce an owning copy");
            WriteAt(owned, -4.0, 0);
            PyFloat("float(own[0])").Should().BeApproximately(0.0, 1e-12, "the owning copy is detached");

            view.Dispose();
        }

        // ============================  ## Lifetime & Memory Safety  ==============================

        /// <summary>
        ///     The three patterns in the "In practice this means..." block: cache an export past its
        ///     source, hand an imported view around after Python forgets the data, run kernels on it.
        /// </summary>
        [TestMethod]
        public void Lifetime_TheInPracticeBlock_Works()
        {
            var cache = new Dictionary<string, PyObject>();

            // Store exported PyObjects in a cache and read them much later — the source NDArrays can
            // be long gone:
            using (Gil()) cache["features"] = MakeAndForgetSource();
            Pump();

            using (Gil())
            {
                Scope.Set("cached", cache["features"]);
            }
            PyStr("cached.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0]",
                "the export roots the buffer even after every C# reference is gone");

            // Hand imported views across threads / queues / closures / awaits — Python can delete its
            // own references, the lease keeps the data alive:
            PyExec("big_dataset = np.arange(5, dtype='f8')");
            NDArray importedView = ViewOf("big_dataset");
            PyExec("del big_dataset");
            Pump();

            double total = (double)np.sum(importedView);   // NumSharp kernels over Python-owned memory
            total.Should().BeApproximately(10.0, 1e-12, "the lease keeps Python's buffer alive");

            importedView.Dispose();
            using (Gil()) { cache["features"].Dispose(); cache.Clear(); }
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        private static PyObject MakeAndForgetSource() => np.arange(4).astype(NPTypeCode.Double).ToNumpy();

        /// <summary>The two observability counters, and that they track conversions exactly.</summary>
        [TestMethod]
        public void Lifetime_ObservabilityCounters_TrackLiveConversions()
        {
            int pins = NDArrayPythonInterop.LiveExports;   // NumSharp buffers rooted by live Python views
            int leases = NDArrayPythonInterop.LiveImports; // Python buffers leased by live NumSharp views

            var nd = np.arange(4).astype(NPTypeCode.Double);
            using (Gil()) { using var p = nd.ToNumpy(); Scope.Set("cnt", p); }
            NDArrayPythonInterop.LiveExports.Should().Be(pins + 1);

            PyExec("cnt_src = np.arange(4, dtype='f8')");
            NDArray v = ViewOf("cnt_src");
            NDArrayPythonInterop.LiveImports.Should().Be(leases + 1);

            v.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == leases).Should().BeTrue();

            PyExec("del cnt");
            WaitFor(() => NDArrayPythonInterop.LiveExports == pins).Should().BeTrue();
        }

        // ============================  ## Auto-Marshaling (the Codec)  ===========================

        /// <summary>
        ///     The Auto-Marshaling sample, run verbatim: an <c>NDArray</c> goes into a Python callable
        ///     as an argument and the result comes back out, with no explicit conversion on either side.
        /// </summary>
        [TestMethod]
        public void Codec_AutoMarshalingSample_ConvertsArgumentsAndReturnValues()
        {
            NDArrayPythonInterop.RegisterCodec();   // once per engine session; idempotent

            PyExec("class _Model:\n" +
                   "    def predict(self, a):\n" +
                   "        return a * 2.0\n" +
                   "model = _Model()");

            var nd = np.arange(4).astype(NPTypeCode.Double);
            NDArray pred;
            using (Py.GIL())
            {
                Scope.Set("x", nd);                                   // auto-encoded as a numpy view
                dynamic model = Scope.Get("model");
                pred = ((PyObject)model.predict(nd)).As<NDArray>();   // in and out, no explicit calls
            }

            pred.Should().NotBeNull();
            pred.typecode.Should().Be(NPTypeCode.Double);
            ReadAt<double>(pred, 3).Should().BeApproximately(6.0, 1e-12, "the model saw the array's real values");
            pred.Dispose();
        }

        /// <summary>The <c>NumpyCodecOptions</c> table's three rows and their documented defaults.</summary>
        [TestMethod]
        public void Codec_OptionsTable_DefaultsAreAsDocumented()
        {
            var o = new NumpyCodecOptions();
            o.EncodeMode.Should().Be(NumpyCodecMode.Auto, "docs: EncodeMode default Auto");
            o.DecodeMode.Should().Be(NumpyCodecMode.Auto, "docs: DecodeMode default Auto");
            o.DecodeAnyBuffer.Should().BeTrue("docs: DecodeAnyBuffer default true");
        }

        /// <summary>
        ///     The "Auto decode shares memory" callout: a viewable source decodes as a zero-copy view,
        ///     a read-only source as a non-writeable view, and the view holds a Py_buffer lock.
        /// </summary>
        [TestMethod]
        public void Codec_AutoDecodeCallout_SharesMemory_AndLocksTheSource()
        {
            var codec = new NumpyCodec(NumpyCodecOptions.Default);

            PyExec("cd = bytearray(b'abcd')");
            NDArray nd;
            using (Gil())
            {
                using PyObject src = Scope.Eval("cd");
                codec.TryDecode(src, out nd).Should().BeTrue();
            }

            WriteAt(nd, (byte)65, 0);
            PyLong("cd[0]").Should().Be(65, "mutations flow both ways");

            // "a bytearray cannot be resized while it lives"
            PyExec("try:\n    cd.append(1)\n    locked = False\nexcept BufferError:\n    locked = True");
            PyBool("locked").Should().BeTrue("the view holds a Py_buffer lock");

            nd.Dispose();
            Pump();
            PyExec("cd.append(1)");   // ...now fine
            PyLong("len(cd)").Should().Be(5);

            // "Use DecodeMode = Copy for a detached snapshot."
            var copyCodec = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });
            PyExec("cd2 = bytearray(b'abcd')");
            NDArray snap;
            using (Gil())
            {
                using PyObject src = Scope.Eval("cd2");
                copyCodec.TryDecode(src, out snap).Should().BeTrue();
            }

            WriteAt(snap, (byte)90, 0);
            PyLong("cd2[0]").Should().Be(97, "Copy mode never shares memory");
            PyExec("cd2.append(1)");   // never locked at all
            snap.Dispose();
        }

        /// <summary>
        ///     "numpy ndarray subclasses decode via an __mro__ walk. Arrays with no numpy dtype
        ///     (decimal) fall back to pythonnet's default CLR-object wrapping."
        /// </summary>
        [TestMethod]
        public void Codec_SubclassesDecode_AndDecimalFallsBackToClrWrapping()
        {
            var codec = new NumpyCodec(NumpyCodecOptions.Default);
            PyExec("class MyArr(np.ndarray): pass\nsub = np.arange(4, dtype='f8').view(MyArr)");

            using (Gil())
            {
                using PyObject t = Scope.Eval("MyArr");
                codec.CanDecode(new PyType(t), typeof(NDArray)).Should().BeTrue("subclasses decode via the __mro__ walk");

                using PyObject src = Scope.Eval("sub");
                codec.TryDecode(src, out NDArray nd).Should().BeTrue();
                nd.Dispose();

                codec.TryEncode(np.arange(3).astype(NPTypeCode.Decimal))
                    .Should().BeNull("decimal falls back to pythonnet's CLR-object wrapping instead of failing");
            }
        }

        // ============================  ## Dtype Mapping  =========================================

        /// <summary>Every row of the dtype table, in both directions where the docs claim both.</summary>
        [TestMethod]
        public void DtypeMapping_EveryTableRow_MapsAsDocumented()
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
                (NPTypeCode.Char, "<u2"),        // UTF-16 code units — numpy has no char dtype
            };

            foreach (var (ns, numpy) in table)
            {
                NDArrayPythonInterop.ToNumpyDtypeStr(ns).Should().Be(numpy, $"docs map {ns} -> {numpy}");
                if (ns != NPTypeCode.Char)   // Char shares <u2 with UInt16; the reverse map yields UInt16
                    NDArrayPythonInterop.FromNumpyDtypeStr(numpy).Should().Be(ns);
            }

            NDArrayPythonInterop.FromNumpyDtypeStr("<u2").Should().Be(NPTypeCode.UInt16,
                "the docs note numpy has no char dtype, so <u2 comes back as UInt16");

            // Decimal — "No numpy equivalent (16-byte, non-IEEE); throws with guidance."
            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>()
                .WithMessage("*decimal has no numpy dtype*").WithMessage("*astype(NPTypeCode.Double)*");

            // complex64 — "sources copy-widen to Complex in ToNDArray (no zero-copy view)"
            PyExec("c8v = np.array([1+2j, 3+4j], dtype='c8')");
            NDArray widened = ImportOf("c8v");
            widened.typecode.Should().Be(NPTypeCode.Complex);
            widened.Dispose();

            // Char — "a 2-byte wchar buffer (PEP 3118 'u') views zero-copy as Char; UCS-4 text
            // (numpy <U1, 4-byte wchar, 'w') copy-narrows to Char in ToNDArray (BMP only)"
            NDArrayPythonInterop.FromBufferFormat("u", 2).Should().Be(NPTypeCode.Char,
                "the docs map 2-byte wchar text units onto Char");
            PyExec("u1v = np.array(['a', 'b'], dtype='U1')");
            NDArray narrowed = ImportOf("u1v");
            narrowed.typecode.Should().Be(NPTypeCode.Char);
            ReadAt<char>(narrowed, 1).Should().Be('b');
            narrowed.Dispose();
            PyExec("u1astral = np.array(['\\U0001F600'], dtype='U1')");
            ((Action)(() => ImportOf("u1astral").Dispose()))
                .Should().Throw<NotSupportedException>("the docs promise astral code points are refused");
        }

        /// <summary>"Big-endian buffers are rejected rather than silently byte-swapped."</summary>
        [TestMethod]
        public void DtypeMapping_BigEndian_IsRejectedWithByteSwapGuidance()
        {
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">i4")))
                .Should().Throw<NotSupportedException>().WithMessage("*newbyteorder('<')*");
            ((Action)(() => NDArrayPythonInterop.FromBufferFormat("!H", 2)))
                .Should().Throw<NotSupportedException>().WithMessage("*newbyteorder('<')*");

            // The documented Python-side fix actually works.
            PyExec("be = np.arange(4, dtype='>i4')\nle = be.astype(be.dtype.newbyteorder('<'))");
            NDArray fixedUp = ImportOf("le");
            fixedUp.typecode.Should().Be(NPTypeCode.Int32);
            ReadAt<int>(fixedUp, 2).Should().Be(2);
            fixedUp.Dispose();
        }

        // ============================  ## Threading & the GIL  ===================================

        /// <summary>The hot-loop opt-out block, run verbatim.</summary>
        [TestMethod]
        public void Gil_HotLoopOptOutBlock_RunsVerbatim()
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

            seen.Should().Equal(new[] { 6.0, 22.0, 38.0 },
                "the loop converted and computed under one acquisition");
        }

        /// <summary>"Every verb takes a nullable requireGIL; null follows the process-wide RequireGIL (default true)."</summary>
        [TestMethod]
        public void Gil_RequireGilDefault_IsTrue()
        {
            NDArrayPythonInterop.RequireGIL.Should().BeTrue("the docs state the process-wide default is true");
        }

        // ============================  ## Engine Lifecycle  ======================================

        /// <summary>
        ///     The shutdown opt-out snippet. It cannot run here (it would end the session), but
        ///     <see cref="PythonSession"/> performs exactly this at assembly cleanup — so this pins
        ///     that the two API members the snippet names still exist with the documented shape.
        /// </summary>
        [TestMethod]
        public void EngineLifecycle_ShutdownOptOutSnippet_NamesRealApi()
        {
            // RuntimeData.FormatterType = typeof(NoopFormatter);
            // PythonEngine.Shutdown();
            var property = typeof(RuntimeData).GetProperty(nameof(RuntimeData.FormatterType));
            property.Should().NotBeNull("the docs tell users to set RuntimeData.FormatterType");
            property.CanWrite.Should().BeTrue();
            property.PropertyType.Should().Be(typeof(Type));
            typeof(NoopFormatter).Should().NotBeNull("the docs name NoopFormatter as pythonnet's opt-out");
            typeof(PythonEngine).GetMethod(nameof(PythonEngine.Shutdown), Type.EmptyTypes).Should().NotBeNull();
        }

        // ============================  ## Troubleshooting  =======================================

        /// <summary>
        ///     Every Troubleshooting row whose "Symptom" quotes a message the bridge itself produces.
        ///     A reworded exception silently invalidates the table; this catches it.
        /// </summary>
        [TestMethod]
        public void Troubleshooting_QuotedSymptoms_AreTheMessagesActuallyThrown()
        {
            // "the exporter's buffer is read-only …"
            ((Action)(() => ViewOf("b'ab'").Dispose()))
                .Should().Throw<InvalidOperationException>().WithMessage("*the exporter's buffer is read-only*");

            // "assignment destination is read-only"
            NDArray ro = ViewOf("b'ab'", allowReadonly: true);
            ((Action)(() => ro[0] = (NDArray)(byte)1))
                .Should().Throw<Exception>().WithMessage("*assignment destination is read-only*");
            ro.Dispose();

            // "BufferError: Existing exports of data: object cannot be re-sized" (Python side)
            PyExec("tba = bytearray(b'abcd')");
            NDArray held = ViewOf("tba");
            using (Gil())
            {
                Scope.Exec("try:\n    tba.append(1)\n    tmsg = ''\nexcept BufferError as e:\n    tmsg = str(e)");
            }
            PyStr("tmsg").Should().Be("Existing exports of data: object cannot be re-sized");
            held.Dispose();
            Pump();

            // "ValueError: cannot resize an array that references or is referenced by …" (Python side)
            // Same cause as the row above, one exporter along: a live NumSharp view leases the numpy
            // array, and numpy's refcheck sees the reference.
            PyExec("tv = np.arange(4, dtype='f8')");
            NDArray importView = ViewOf("tv");
            using (Gil())
            {
                Scope.Exec("try:\n    tv.resize((8,), refcheck=True)\n    rmsg = ''\nexcept ValueError as e:\n    rmsg = str(e)");
            }
            PyStr("rmsg").Should().Contain("cannot resize an array that references or is referenced");

            // "cannot resize this array: it does not own its data" — the NumSharp side of the same
            // view, which (like np.frombuffer) does not own the memory it reads.
            ((Action)(() => importView.resize(new Shape(8))))
                .Should().Throw<Exception>().WithMessage("*cannot resize this array: it does not own its data*");
            importView.Dispose();

            // "big-endian dtype … cannot be shared"
            ((Action)(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8")))
                .Should().Throw<NotSupportedException>().WithMessage("*big-endian dtype*");

            // "decimal has no numpy dtype"
            ((Action)(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal)))
                .Should().Throw<NotSupportedException>().WithMessage("*decimal has no numpy dtype*");
        }

        // The last Troubleshooting row ("Access violation ... almost always requireGIL: false on a
        // thread that does not hold the GIL — including inside a Python -> .NET callback") rests on
        // pythonnet releasing the GIL around managed callback bodies. Proving it needs a raw
        // PyGILState_Check against the loaded interpreter, which
        // GilPolicyTests.PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere already
        // does; it is not duplicated here.
    }
}
