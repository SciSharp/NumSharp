using System;
using System.Runtime.InteropServices;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The GIL-management policy: every conversion verb takes a nullable <c>requireGIL</c>
    ///     parameter, <c>null</c> falls back to the process-wide <see cref="NDArrayPythonInterop.RequireGIL"/>
    ///     (default <c>true</c>), and an effective <c>false</c> replaces <see cref="Py.GIL"/> with a
    ///     shared no-op guard — the caller must already hold the GIL.
    ///
    ///     <para>Proof strategy: the policy factory (<c>AcquireGil</c>, internal) is asserted
    ///     precisely — singleton no-op guard vs real <see cref="Py.GILState"/> — because from inside
    ///     one thread a re-entrant <c>PyGILState_Ensure</c> is otherwise unobservable; that
    ///     <c>requireGIL:true</c> REALLY acquires is proven by cross-thread contention (the conversion
    ///     must block while another thread owns the GIL); and the end-to-end wiring of every verb is
    ///     exercised under a caller-held GIL. The base-class leak gate doubles every test as a
    ///     lifetime-accounting assertion.</para>
    /// </summary>
    [TestClass]
    public class GilPolicyTests : InteropTestBase
    {
        /// <summary>Belt-and-braces on top of each test's own <c>finally</c>: the global policy must
        /// never leak into other suites (runs before the base cleanup and its leak gate).</summary>
        [TestCleanup]
        public void RestoreGilPolicy() => NDArrayPythonInterop.RequireGIL = true;

        // ---- the policy surface -------------------------------------------------------------------

        [TestMethod]
        public void RequireGil_DefaultsToTrue()
        {
            // Every mutating test restores the global (finally + [TestCleanup]) and the assembly is
            // [DoNotParallelize], so observing true here IS observing the default.
            NDArrayPythonInterop.RequireGIL.Should().BeTrue("GIL management must be opt-OUT — the safe default acquires");
        }

        [TestMethod]
        public void AcquireGil_False_IsTheSharedNoOpGuard()
        {
            IDisposable g1 = NDArrayPythonInterop.AcquireGil(false);
            IDisposable g2 = NDArrayPythonInterop.AcquireGil(false);

            g1.Should().BeSameAs(g2, "the no-GIL guard is one shared instance — zero per-call allocation");
            g1.Should().NotBeAssignableTo<Py.GILState>("false must not touch PyGILState at all");

            g1.Dispose();
            g1.Dispose();   // the guard is shared and reused forever — disposal must be a true no-op
            g2.Dispose();
        }

        [TestMethod]
        public void AcquireGil_True_IsARealGilScope()
        {
            IDisposable g = NDArrayPythonInterop.AcquireGil(true);
            try
            {
                g.Should().BeAssignableTo<Py.GILState>("explicit true must actually take the GIL");
            }
            finally
            {
                g.Dispose();
            }
        }

        [TestMethod]
        public void AcquireGil_Null_FollowsTheGlobal()
        {
            IDisposable noGil = NDArrayPythonInterop.AcquireGil(false);

            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                NDArrayPythonInterop.AcquireGil(null).Should().BeSameAs(noGil, "null + RequireGIL=false → the no-op guard");
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = true;
            }

            IDisposable g = NDArrayPythonInterop.AcquireGil(null);
            try
            {
                g.Should().BeAssignableTo<Py.GILState>("null + RequireGIL=true → a real GIL scope");
            }
            finally
            {
                g.Dispose();
            }
        }

        [TestMethod]
        public void AcquireGil_ExplicitParameter_OverridesTheGlobal()
        {
            // global true (the default), explicit false → no-op guard
            NDArrayPythonInterop.AcquireGil(false).Should().BeSameAs(NDArrayPythonInterop.AcquireGil(false));
            NDArrayPythonInterop.AcquireGil(false).Should().NotBeAssignableTo<Py.GILState>(
                "explicit false must win over the global true");

            // global false, explicit true → real GIL scope
            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                IDisposable g = NDArrayPythonInterop.AcquireGil(true);
                try
                {
                    g.Should().BeAssignableTo<Py.GILState>("explicit true must win over the global false");
                }
                finally
                {
                    g.Dispose();
                }
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = true;
            }
        }

        // ---- true really acquires: cross-thread contention proof -----------------------------------

        [TestMethod]
        public void RequireGilTrue_ActuallyAcquires_ProvenByContention()
        {
            var nd = np.arange(3).astype(NPTypeCode.Double);
            using var gilHeld = new ManualResetEventSlim(false);
            using var releaseGil = new ManualResetEventSlim(false);
            using var converted = new ManualResetEventSlim(false);
            Exception failure = null;

            var holder = new Thread(() =>
            {
                using (Py.GIL())
                {
                    gilHeld.Set();
                    releaseGil.Wait(15_000);
                }
            }) { IsBackground = true, Name = "gil-holder" };

            var converter = new Thread(() =>
            {
                try
                {
                    gilHeld.Wait(15_000);
                    PyObject arr = nd.ToNumpy(requireGIL: true);   // must BLOCK: another thread owns the GIL
                    converted.Set();
                    using (Py.GIL())
                        arr.Dispose();
                }
                catch (Exception e)
                {
                    failure = e;
                }
            }) { IsBackground = true, Name = "gil-converter" };

            holder.Start();
            converter.Start();

            converted.Wait(500).Should().BeFalse(
                "with the GIL held on another thread, requireGIL:true must be stuck inside PyGILState_Ensure — " +
                "completing here would mean the conversion ran without acquiring");

            releaseGil.Set();
            converter.Join(15_000).Should().BeTrue("the conversion must finish once the GIL is free");
            holder.Join(15_000).Should().BeTrue();
            failure.Should().BeNull($"the conversion after GIL release must succeed, got: {failure}");
            converted.IsSet.Should().BeTrue("once the GIL was released the blocked acquisition must have proceeded");
        }

        // ---- the trap: Python -> .NET bodies do NOT hold the GIL -----------------------------------

        [TestMethod]
        public unsafe void PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere()
        {
            // Ground truth from the C-API itself: PyGILState_Check is safe to call WITHOUT the GIL
            // (that is its purpose), resolved from the exact library pythonnet loaded. This pins the
            // documented trap: pythonnet's method binder RELEASES the GIL around a Python->NET body
            // (probed on 3.0.5 and 3.1.0), so the one call site that LOOKS like it inherits the GIL
            // does not — requireGIL:false there is an access violation, requireGIL:true is mandatory.
            IntPtr lib = NativeLibrary.Load(Runtime.PythonDLL);
            try
            {
                var check = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(lib, "PyGILState_Check");

                int inBody = -1;
                string conversionInsideCallback = null;

                using (Gil())
                {
                    check().Should().Be(1, "the outer Py.GIL scope really holds the GIL");

                    Action cb = () =>
                    {
                        inBody = check();
                        try
                        {
                            PyObject p = np.arange(4).ToNumpy(requireGIL: true);   // must re-acquire itself
                            using (Py.GIL())
                                p.Dispose();   // pythonnet needs the GIL for Dispose (PyErr_Fetch)
                            conversionInsideCallback = "OK";
                        }
                        catch (Exception e)
                        {
                            conversionInsideCallback = $"{e.GetType().Name}: {e.Message}";
                        }
                    };
                    Scope.Set("gil_cb", cb);
                    Scope.Exec("gil_cb()");
                }

                inBody.Should().Be(0,
                    "pythonnet's method binder releases the GIL around Python->NET bodies; if this ever " +
                    "flips to 1 a pythonnet version changed the binder contract and the RequireGIL docs " +
                    "must be revisited");
                conversionInsideCallback.Should().Be("OK",
                    "requireGIL:true inside the callback re-acquires and must succeed");
            }
            finally
            {
                NativeLibrary.Free(lib);
            }
        }

        // ---- end-to-end: every verb runs GIL-free under a caller-held GIL ---------------------------

        [TestMethod]
        public void AllVerbs_WithExplicitNoGil_UnderCallerHeldGil()
        {
            var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            var flat = np.arange(4).astype(NPTypeCode.Int32);
            var empty = np.zeros(0);

            using (Gil())
            {
                // exports: view, copy, both alias forms, the copy-routing overload, raw memoryview,
                // and the empty-array fast path (np.empty through the cached callables).
                using (PyObject arr = nd.ToNumpy(requireGIL: false))
                    Scope.Set("ng", arr);
                using (PyObject copy = nd.ToNumpyCopy(requireGIL: false))
                    Scope.Set("ngc", copy);
                using (PyObject alias = nd.ToPython(requireGIL: false))
                    Scope.Set("ngp", alias);
                using (PyObject routed = nd.ToNumpy(copy: true, requireGIL: false))
                    Scope.Set("ngr", routed);
                using (PyObject mv = flat.ToMemoryView(requireGIL: false))
                    Scope.Set("ngmv", mv);
                using (PyObject e = empty.ToNumpy(requireGIL: false))
                    Scope.Set("nge", e);

                // import: copy
                using (PyObject src = Scope.Eval("np.array([1.5, 2.5, 3.5])"))
                {
                    NDArray copied = src.ToNDArray(requireGIL: false);
                    copied.size.Should().Be(3);
                    ReadAt<double>(copied, 1).Should().BeApproximately(2.5, 1e-12);
                }

                // import: zero-copy view + the As… alias (dispose deterministically; the lease
                // drain is machinery and manages the GIL itself)
                Scope.Exec("vsrc = np.arange(5.0)");
                using (PyObject vp = Scope.Eval("vsrc"))
                {
                    NDArray view = NDArrayPythonInterop.ToNDArrayView(vp, allowReadonly: false, requireGIL: false);
                    WriteAt(view, 42.0, 3);
                    view.Dispose();
                }

                using (PyObject vp = Scope.Eval("vsrc"))
                {
                    NDArray view = vp.AsNDArray(requireGIL: false);
                    ReadAt<double>(view, 3).Should().BeApproximately(42.0, 1e-12,
                        "the previous no-GIL view wrote through to Python's memory");
                    view.Dispose();
                }
            }

            // shared-buffer proofs through the no-GIL exports
            PyFloat("float(ng.sum())").Should().BeApproximately(15.0, 1e-9);
            PyExec("ng[0, 0] = 50.0");
            ReadAt<double>(nd, 0, 0).Should().BeApproximately(50.0, 1e-12, "ng is a zero-copy view");

            PyStr("type(ngp).__name__").Should().Be("ndarray");
            PyLong("ngmv.nbytes").Should().Be(16);
            PyLong("nge.size").Should().Be(0);

            PyExec("ngc[0, 0] = -1.0");
            ReadAt<double>(nd, 0, 0).Should().BeApproximately(50.0, 1e-12, "ngc is an independent copy");
        }

        [TestMethod]
        public void GlobalOptOut_NullParameters_RunUnderTheCallersGil()
        {
            var nd = np.arange(4).astype(NPTypeCode.Double);

            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                using (Gil())
                {
                    using (PyObject arr = nd.ToNumpy())          // null → global false → no-op guard
                        Scope.Set("g_ng", arr);

                    using (PyObject src = Scope.Eval("np.array([7, 8, 9], dtype='i4')"))
                    {
                        NDArray copied = src.ToNDArray();        // null → global false
                        ReadAt<int>(copied, 2).Should().Be(9);
                    }
                }
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = true;
            }

            PyExec("g_ng[1] = -1.0");
            ReadAt<double>(nd, 1).Should().BeApproximately(-1.0, 1e-12,
                "a conversion made under the global opt-out is a normal zero-copy view");
        }

        [TestMethod]
        public void CodecConversions_UnderGlobalOptOut_InheritThePythonnetCallersGil()
        {
            // pythonnet invokes codecs while the GIL is held (conversions happen inside Python
            // calls), so the global opt-out makes codec-mediated conversions skip re-acquisition.
            CodecTests.EnsureCodec();
            var nd = np.arange(4).astype(NPTypeCode.Double);

            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                using (Gil())
                {
                    Scope.Set("codec_ng", nd);                       // encoder → ToNumpy(null)
                    using PyObject doubled = Scope.Eval("codec_ng * 2");
                    NDArray back = doubled.As<NDArray>();            // decoder → ToNDArray(null)
                    ReadAt<double>(back, 3).Should().BeApproximately(6.0, 1e-12);
                }
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = true;
            }

            PyExec("codec_ng[0] = 9.0");
            ReadAt<double>(nd, 0).Should().BeApproximately(9.0, 1e-12, "the codec encoded a zero-copy view");
        }

        // ---- the machinery must be immune to the policy ---------------------------------------------

        [TestMethod]
        public void DeferredLeaseDrain_StillManagesTheGilItself_WhileGlobalOptOutActive()
        {
            // The ThreadPool drain and the shutdown drain run on threads that cannot inherit the
            // caller's GIL — they must keep acquiring regardless of the policy, or every disposal
            // under the opt-out would touch the C-API bare on a naked thread.
            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                int baseline = NDArrayPythonInterop.LiveImports;

                using (Gil())
                {
                    Scope.Exec("drainsrc = np.arange(8.0)");
                    using PyObject src = Scope.Eval("drainsrc");
                    NDArray view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: false, requireGIL: false);
                    NDArrayPythonInterop.LiveImports.Should().Be(baseline + 1);
                    view.Dispose();   // enqueues the lease; the drain takes the GIL ITSELF
                }

                WaitFor(() => NDArrayPythonInterop.LiveImports == baseline, 12_000).Should().BeTrue(
                    "the deferred drain manages the GIL itself and must be immune to the global opt-out");
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = true;
            }
        }

        // ---- overload & hot-loop pins ---------------------------------------------------------------

        [TestMethod]
        public void OverloadResolution_BoolLiteral_StillMeansCopy()
        {
            // nd.ToNumpy(true) has TWO candidates since the policy landed: (source, copy) and
            // (source, requireGIL). bool → bool is exact, bool → bool? needs lifting, so the copy
            // overload must keep winning — pinned by semantics (a copy shares nothing).
            var nd = np.arange(3).astype(NPTypeCode.Double);

            using (Gil())
            {
                using PyObject viaBool = nd.ToNumpy(true);
                Scope.Set("ovl", viaBool);
            }

            PyExec("ovl[0] = 123.0");
            ReadAt<double>(nd, 0).Should().Be(0.0,
                "nd.ToNumpy(true) must still bind to the (source, copy) overload and hand back an independent copy");
        }

        [TestMethod]
        public void DtypeGate_FiresBeforeAnyGilDecision()
        {
            // The Decimal gate throws before AcquireGil runs, so it must behave identically on a
            // GIL-free thread with management off — no Python work, no GIL requirement.
            var dec = np.arange(3).astype(NPTypeCode.Decimal);
            Action export = () => dec.ToNumpy(requireGIL: false);
            export.Should().Throw<NotSupportedException>().WithMessage("*decimal has no numpy dtype*");
        }

        [TestMethod]
        public void HotLoop_ManyConversionsUnderOneGil_NoPerCallGilManagement()
        {
            // The scenario the opt-out exists for: one outer acquisition, N conversions inside.
            var nd = np.arange(16).astype(NPTypeCode.Double);

            using (Gil())
            {
                for (int i = 0; i < 200; i++)
                {
                    using PyObject arr = nd.ToNumpy(requireGIL: false);
                    using PyObject copy = nd.ToNumpyCopy(requireGIL: false);
                    NDArray back = copy.ToNDArray(requireGIL: false);
                    if (back.size != 16)
                        Assert.Fail($"iteration {i}: round-trip size {back.size} != 16");
                }
            }

            // one value check outside the loop (helpers take their own GIL)
            using (Gil())
            {
                using PyObject arr = nd.ToNumpy(requireGIL: false);
                NDArray back = arr.ToNDArray(requireGIL: false);
                ReadAt<double>(back, 15).Should().BeApproximately(15.0, 1e-12);
            }
        }
    }
}
