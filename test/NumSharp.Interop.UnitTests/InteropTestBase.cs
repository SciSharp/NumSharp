using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Shared plumbing for all engine-backed interop tests.
    ///
    ///     <para><b>Leak gate:</b> every test captures the <see cref="PythonConvert.LiveExports"/> /
    ///     <see cref="PythonConvert.LiveImports"/> baseline on entry, and the cleanup FAILS the test
    ///     unless the counters return to that baseline — so every single test doubles as a
    ///     no-leak / no-premature-free assertion.</para>
    ///
    ///     <para><b>GIL discipline:</b> pythonnet 3.0.x requires the GIL for <c>PyObject.Dispose</c>
    ///     (the final decref runs <c>PyErr_Fetch</c>), so every helper here that touches a
    ///     <see cref="PyObject"/> creates AND disposes it inside its own <see cref="Py.GIL"/> scope;
    ///     tests handle NumSharp arrays and plain CLR values only, unless they explicitly open
    ///     <see cref="Gil"/>.</para>
    ///
    ///     <para><b>GC realism:</b> collection-dependent lifecycles must run inside
    ///     <c>[MethodImpl(MethodImplOptions.NoInlining)]</c> helpers — debug-build JIT keeps untracked
    ///     temps of a frame's references alive until the method returns, so an NDArray created and
    ///     dropped in the test method itself is NOT collectable until the test ends.</para>
    /// </summary>
    public abstract class InteropTestBase
    {
        protected PyModule Scope;
        private int _baseExports, _baseImports;

        [TestInitialize]
        public void InteropInit()
        {
            PythonSession.EnsureOrInconclusive();

            Settle();
            _baseExports = PythonConvert.LiveExports;
            _baseImports = PythonConvert.LiveImports;

            using (Py.GIL())
            {
                Scope = Py.CreateScope();
                Scope.Exec("import numpy as np\nimport gc, array");
            }
        }

        [TestCleanup]
        public void InteropCleanup()
        {
            if (Scope is null)
                return;   // engine unavailable — test was Inconclusive

            using (Py.GIL())
                Scope.Dispose();
            Scope = null;

            bool settled = WaitFor(() => PythonConvert.LiveExports <= _baseExports &&
                                         PythonConvert.LiveImports <= _baseImports, 12_000);
            Assert.IsTrue(settled,
                $"interop leaked conversions: LiveExports {_baseExports} -> {PythonConvert.LiveExports}, " +
                $"LiveImports {_baseImports} -> {PythonConvert.LiveImports}");
        }

        // ---- python helpers (each opens/closes its own GIL scope) --------------------------------

        protected IDisposable Gil() => Py.GIL();

        protected void PyExec(string code)
        {
            using (Py.GIL()) Scope.Exec(code);
        }

        protected long PyLong(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval(expr); return r.As<long>(); }
        }

        protected double PyFloat(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval(expr); return r.As<double>(); }
        }

        protected string PyStr(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval($"str({expr})"); return r.As<string>(); }
        }

        protected bool PyBool(string expr)
        {
            using (Py.GIL()) { using var r = Scope.Eval($"bool({expr})"); return r.As<bool>(); }
        }

        /// <summary>Zero-copy export of <paramref name="nd"/> bound to a scope name.</summary>
        protected void ExportTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = PythonConvert.ToNumpy(nd); Scope.Set(name, p); }
        }

        /// <summary>Independent-copy export bound to a scope name.</summary>
        protected void ExportCopyTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = PythonConvert.ToNumpyCopy(nd); Scope.Set(name, p); }
        }

        /// <summary>Raw-bytes memoryview export bound to a scope name.</summary>
        protected void ExportMemoryViewTo(string name, NDArray nd)
        {
            using (Py.GIL()) { using var p = PythonConvert.ToMemoryView(nd); Scope.Set(name, p); }
        }

        /// <summary>Copy-import the result of a python expression.</summary>
        protected NDArray ImportOf(string expr)
        {
            using (Py.GIL()) { using var p = Scope.Eval(expr); return PythonConvert.ToNDArray(p); }
        }

        /// <summary>Zero-copy view-import the result of a python expression.</summary>
        protected NDArray ViewOf(string expr, bool allowReadonly = false)
        {
            using (Py.GIL()) { using var p = Scope.Eval(expr); return PythonConvert.ToNDArrayView(p, allowReadonly); }
        }

        // ---- direct-memory readers/writers (prove aliasing at the pointer level) -----------------

        protected static unsafe void WriteAt<T>(NDArray nd, T value, params long[] coords) where T : unmanaged
        {
            var sh = nd.Shape;
            long off = sh.Offset;
            for (int i = 0; i < coords.Length; i++) off += coords[i] * sh.Strides[i];
            *((T*)nd.Storage.Address + off) = value;
        }

        protected static unsafe T ReadAt<T>(NDArray nd, params long[] coords) where T : unmanaged
        {
            var sh = nd.Shape;
            long off = sh.Offset;
            for (int i = 0; i < coords.Length; i++) off += coords[i] * sh.Strides[i];
            return *((T*)nd.Storage.Address + off);
        }

        // ---- GC / drain pumping -------------------------------------------------------------------

        /// <summary>
        ///     One full release turn: CLR GC + finalizers (NDArray finalizers release ARC refs and
        ///     enqueue lease disposals), pythonnet's deferred-decref flush, a Python GC pass, and one
        ///     trivial conversion to run the interop's inline drain.
        /// </summary>
        protected static void Pump()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            using (Py.GIL())
            {
                Finalizer.Instance.Collect();
                PythonEngine.RunSimpleString("import gc; gc.collect()");
                using var t = PythonConvert.ToNumpyCopy(np.arange(1));
            }
        }

        protected static bool WaitFor(Func<bool> condition, int timeoutMs = 10_000)
        {
            var sw = Stopwatch.StartNew();
            while (sw.ElapsedMilliseconds < timeoutMs)
            {
                if (condition()) return true;
                Pump();
                System.Threading.Thread.Sleep(20);
            }

            return condition();
        }

        /// <summary>Pump until the live counters stop changing (start-of-test quiescence).</summary>
        protected static void Settle()
        {
            int lastE = -1, lastI = -1;
            for (int i = 0; i < 40; i++)
            {
                int e = PythonConvert.LiveExports, im = PythonConvert.LiveImports;
                if (e == lastE && im == lastI && (e + im == 0 || i > 2))
                    return;
                lastE = e; lastI = im;
                Pump();
            }
        }
    }
}
