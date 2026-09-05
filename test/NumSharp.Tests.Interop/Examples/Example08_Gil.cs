using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/08-gil.cs</c>, section by section.</summary>
    [TestClass]
    public class Example08_Gil : ExampleTestBase
    {
        [TestMethod]
        public void TheDefault_VerbsAcquireTheGilThemselves()
        {
            NDArrayPythonInterop.RequireGIL.Should().BeTrue("the default");
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            PyObject bare = nd.ToNumpy();                                // no Py.GIL() on this thread
            using (Gil())
            {
                py.g1 = bare;
                bare.Dispose();
                py.g2 = nd.ToNumpy();                                    // re-entrant under an outer acquisition
                bool same = py.np.shares_memory(py.g1, py.g2);
                same.Should().BeTrue();
                py.Exec("del g1, g2");
            }
        }

        [TestMethod]
        public void TheHotLoop_OneAcquisition_ManyConversions()
        {
            using var scope = NDScope.Open();
            NDArray[] batches = Enumerable.Range(0, 200).Select(i => np.arange(16).astype(np.float64) + i).ToArray();
            double total = 0;
            using (Gil())
                foreach (NDArray batch in batches)
                {
                    dynamic p = batch.ToNumpy(requireGIL: false);
                    total += (double)p.sum();
                }

            double total2 = 0;
            foreach (NDArray batch in batches)
                using (Gil())
                {
                    dynamic p = batch.ToNumpy();
                    total2 += (double)p.sum();
                }

            total.Should().Be(200 * 120.0 + 16.0 * (199 * 200 / 2), "342400");
            total2.Should().Be(total);
        }

        [TestMethod]
        public void TheProcessWideOptOut_NullParametersFollowIt()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                using (Gil())
                {
                    py.g1 = nd.ToNumpy();
                    py.g3 = nd.ToNumpy();
                    using PyObject src = py.Eval("np.array([7, 8, 9], dtype='i4')");
                    NDArray copied = src.ToNDArray();
                    copied.GetInt32(2).Should().Be(9, "every verb ran on the caller's GIL");
                    bool same = py.np.shares_memory(py.g3, py.g1);
                    same.Should().BeTrue();
                    using PyObject forced = nd.ToNumpy(requireGIL: true);
                    forced.Should().NotBeNull("an explicit true still acquires, re-entrantly");
                    py.Exec("del g1, g3");
                }
            }
            finally { NDArrayPythonInterop.RequireGIL = true; }
        }

        [TestMethod]
        public void RequireGilTrue_BlocksWhileAnotherThreadOwnsTheGil()
        {
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            using var gilHeld = new ManualResetEventSlim(false);
            using var releaseGil = new ManualResetEventSlim(false);
            using var converted = new ManualResetEventSlim(false);
            var holder = new Thread(() => { using (Py.GIL()) { gilHeld.Set(); releaseGil.Wait(15_000); } }) { IsBackground = true };
            var converter = new Thread(() =>
            {
                gilHeld.Wait(15_000);
                PyObject arr = nd.ToNumpy(requireGIL: true);
                converted.Set();
                using (Py.GIL()) arr.Dispose();
            }) { IsBackground = true };
            holder.Start();
            converter.Start();
            bool finishedEarly = converted.Wait(400);
            releaseGil.Set();
            converter.Join(15_000).Should().BeTrue();
            holder.Join(15_000).Should().BeTrue();
            finishedEarly.Should().BeFalse("the conversion waited for the holder");
            converted.IsSet.Should().BeTrue("...and completed once the holder let go");
        }

        [TestMethod]
        public void WorkerThreads_EachTakesTheGilAroundItsOwnPythonWork()
        {
            PyExec("def weigh(a):\n    return float((a * 0.5).sum())");
            var failures = new ConcurrentQueue<string>();
            Thread[] workers = Enumerable.Range(0, 3).Select(id => new Thread(() =>
            {
                try
                {
                    using var workerScope = NDScope.Open();
                    for (int k = 0; k < 10; k++)
                    {
                        NDArray batch = np.arange(6).astype(np.float64) + id;
                        double weighed;
                        using (Py.GIL())
                            weighed = py.weigh(batch);
                        if (weighed != (15.0 + 6 * id) * 0.5) failures.Enqueue($"worker {id}: {weighed}");
                    }
                }
                catch (Exception e) { failures.Enqueue($"worker {id}: {e.GetType().Name}: {e.Message}"); }
            }) { IsBackground = true }).ToArray();
            foreach (Thread w in workers) w.Start();
            foreach (Thread w in workers) w.Join(30_000).Should().BeTrue();
            failures.Should().BeEmpty("3 workers x 10 calls, no deadlock, no wrong answer");
        }

        [TestMethod]
        public void TheTrap_APythonToNetCallbackBody_DoesNotHoldTheGil()
        {
            int inBody = -1;
            string inside = null;
            using (Gil())
            {
                PyGILState_Check().Should().Be(1, "inside our own Py.GIL() block");
                Action callback = () =>
                {
                    inBody = PyGILState_Check();
                    try
                    {
                        using NDArray tmp = np.arange(4).astype(np.float64);
                        PyObject p = tmp.ToNumpy(requireGIL: true);
                        using (Py.GIL()) p.Dispose();
                        inside = "OK";
                    }
                    catch (Exception e) { inside = $"{e.GetType().Name}: {e.Message}"; }
                };
                py.gil_cb = callback;
                py.Exec("gil_cb()");
                py.Exec("del gil_cb");
            }

            inBody.Should().Be(0, "pythonnet's binder released the GIL around the managed body");
            inside.Should().Be("OK", "requireGIL: true re-acquires inside the callback");
        }

        [TestMethod]
        public void TheMachinery_IsImmuneToThePolicy()
        {
            int imports0 = NDArrayPythonInterop.LiveImports;
            NDArrayPythonInterop.RequireGIL = false;
            try
            {
                using (Gil())
                {
                    py.Exec("drainsrc = np.arange(8.0)");
                    using PyObject src = py.drainsrc;
                    NDArray view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: false, requireGIL: false);
                    NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "a view taken GIL-less holds a lease");
                    view.Dispose();
                }

                WaitFor(() => NDArrayPythonInterop.LiveImports == imports0).Should().BeTrue("the drain took the GIL itself");
            }
            finally { NDArrayPythonInterop.RequireGIL = true; }
        }

        /// <summary>Ground truth from the C-API itself — safe to call without the GIL.</summary>
        private static unsafe int PyGILState_Check()
        {
            IntPtr lib = NativeLibrary.Load(Runtime.PythonDLL);
            var check = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(lib, "PyGILState_Check");
            return check();
        }
    }
}
