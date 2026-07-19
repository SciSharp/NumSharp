using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Reference-flow pass: conversions are PASSED AROUND (threads, queues, closures, awaits),
    ///     STORED (C# dictionaries of PyObjects, registries of NDArray views, Python containers) and
    ///     accessed LAZILY — long after the creating frame, the creating thread, or the other side's
    ///     references are gone. Every test hunts the two fatal failure modes: a reference chain that
    ///     silently breaks (premature free / use-after-free) and a reference that is never released
    ///     (leak — caught by the base-class baseline gate on every test).
    /// </summary>
    [TestClass]
    public class ReferenceFlowTests : InteropTestBase
    {
        // ------------------------------------------------------------------------------------------
        // storing references in C# and accessing lazily
        // ------------------------------------------------------------------------------------------

        /// <summary>
        ///     "Result repository": exported PyObjects are stored in a C# dictionary as the ONLY
        ///     reference anywhere (no scope name, source NDArrays dropped immediately). If the wrapper's
        ///     reference were lost or prematurely decref'd, the lazy reads would see dead objects.
        /// </summary>
        [TestMethod]
        public void PyObjectRepository_WrapperIsTheOnlyReference_LazyReadsStayValid()
        {
            int e0 = NDArrayPythonInterop.LiveExports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            Dictionary<string, PyObject> BuildRepository()
            {
                var repo = new Dictionary<string, PyObject>();
                for (int i = 0; i < 5; i++)
                {
                    var nd = np.arange(6).astype(NPTypeCode.Double) * (i + 1);   // dropped right away
                    repo[$"series-{i}"] = NDArrayPythonInterop.ToNumpy(nd);
                }

                return repo;
            }

            var repository = BuildRepository();
            Pump(); Pump(); Pump();   // every source NDArray is now collectable — buffers must survive

            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 5);
            for (int i = 0; i < 5; i++)
            {
                using (Gil())
                    Scope.Set("tmp", repository[$"series-{i}"]);
                PyFloat("float(tmp.sum())").Should().BeApproximately(15.0 * (i + 1), 1e-9,
                    $"series-{i} must still read valid memory long after its creator died");
            }

            using (Gil())
            {
                foreach (var stored in repository.Values)
                    stored.Dispose();
                Scope.Exec("del tmp");
            }
            repository.Clear();

            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue(
                "disposing the stored wrappers was the last reference chain — everything must drain");
        }

        /// <summary>
        ///     "Ingest registry": NDArray views over Python-owned arrays are stored in a registry;
        ///     Python deletes its names immediately. A consumer later runs NumSharp kernels over random
        ///     entries, then retires entries one by one — each retirement must release EXACTLY one lease.
        /// </summary>
        [TestMethod]
        public void ImportRegistry_LazyKernelAccess_RetirementReleasesStepwise()
        {
            int i0 = NDArrayPythonInterop.LiveImports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            Dictionary<int, NDArray> BuildRegistry()
            {
                var reg = new Dictionary<int, NDArray>();
                for (int k = 0; k < 4; k++)
                {
                    PyExec($"ing = np.arange(8, dtype='f8') * {k + 1}");
                    reg[k] = ViewOf("ing");
                    PyExec("del ing");   // python forgets immediately; the lease is the only keeper
                }

                return reg;
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            double LazySum(Dictionary<int, NDArray> reg, int k) => ReadAt<double>(np.sum(reg[k]));

            var registry = BuildRegistry();
            Pump(); Pump();
            NDArrayPythonInterop.LiveImports.Should().Be(i0 + 4);

            foreach (int k in new[] { 2, 0, 3, 1 })
                LazySum(registry, k).Should().BeApproximately(28.0 * (k + 1), 1e-9,
                    $"entry {k}: NumSharp kernels over the leased memory, long after python forgot it");

            for (int k = 0; k < 4; k++)
            {
                registry.Remove(k);
                WaitFor(() => NDArrayPythonInterop.LiveImports == i0 + 3 - k).Should().BeTrue(
                    $"retiring entry {k} must release exactly its own lease and nothing else");
            }
        }

        /// <summary>
        ///     Dropping a stored wrapper WITHOUT disposing it must not lose the release: pythonnet's
        ///     deferred finalization decrefs it, the numpy array dies, and the export pin drains —
        ///     no leak, no finalizer-thread crash.
        /// </summary>
        [TestMethod]
        public void DroppedWrapper_WithoutDispose_IsReclaimedSafely()
        {
            int e0 = NDArrayPythonInterop.LiveExports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            void CreateAndDrop()
            {
                var nd = np.arange(4).astype(NPTypeCode.Double);
                var wrapper = NDArrayPythonInterop.ToNumpy(nd);   // never disposed, never bound to a name
                GC.KeepAlive(wrapper);
            }

            CreateAndDrop();
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue(
                "an undisposed wrapper must be reclaimed through pythonnet's deferred finalization");
        }

        // ------------------------------------------------------------------------------------------
        // passing references between threads (GIL correctness with lazy consumption)
        // ------------------------------------------------------------------------------------------

        /// <summary>
        ///     "Pipeline handoff, export direction": a producer thread converts and enqueues PyObjects;
        ///     a separate consumer thread lazily reads and disposes them. Neither the source NDArrays
        ///     nor any scope name survive the handoff — the queued wrapper is the whole chain.
        /// </summary>
        [TestMethod]
        public void ProducerConsumer_PyObjectsAcrossThreads()
        {
            int e0 = NDArrayPythonInterop.LiveExports;
            var queue = new BlockingCollection<(PyObject Array, double ExpectedSum)>();
            var failures = new ConcurrentQueue<Exception>();

            var producer = new Thread(() =>
            {
                try
                {
                    for (int i = 0; i < 12; i++)
                    {
                        var nd = np.arange(6).astype(NPTypeCode.Double) + i;   // sum = 15 + 6i
                        queue.Add((NDArrayPythonInterop.ToNumpy(nd), 15.0 + 6 * i));
                    }
                }
                catch (Exception e) { failures.Enqueue(e); }
                finally { queue.CompleteAdding(); }
            }) { IsBackground = true };

            var consumer = new Thread(() =>
            {
                try
                {
                    foreach (var (array, expected) in queue.GetConsumingEnumerable())
                    {
                        Thread.Sleep(5);   // lazy: consume noticeably later than production
                        using (Py.GIL())
                        {
                            Scope.Set("qa", array);
                            array.Dispose();
                        }

                        PyFloat("float(qa.sum())").Should().BeApproximately(expected, 1e-9);
                        PyExec("del qa");
                    }
                }
                catch (Exception e) { failures.Enqueue(e); }
            }) { IsBackground = true };

            producer.Start();
            consumer.Start();
            producer.Join(30_000).Should().BeTrue("producer must not deadlock");
            consumer.Join(30_000).Should().BeTrue("consumer must not deadlock");

            failures.Should().BeEmpty(string.Join(" | ", failures));
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue("all handed-off exports must drain");
        }

        /// <summary>
        ///     "Pipeline handoff, import direction": the producer imports views over Python arrays and
        ///     deletes the Python names BEFORE enqueueing; the consumer thread reads/writes through the
        ///     views later and disposes them. The lease must carry the memory across the thread hop.
        /// </summary>
        [TestMethod]
        public void ProducerConsumer_ImportedViewsAcrossThreads()
        {
            int i0 = NDArrayPythonInterop.LiveImports;
            var queue = new BlockingCollection<(NDArray View, int Tag)>();
            var failures = new ConcurrentQueue<Exception>();

            var producer = new Thread(() =>
            {
                try
                {
                    for (int i = 0; i < 12; i++)
                    {
                        PyExec($"hv = np.arange(6, dtype='f8') * {i}");
                        var view = ViewOf("hv");
                        PyExec("del hv");   // python's reference is gone before the consumer ever runs
                        queue.Add((view, i));
                    }
                }
                catch (Exception e) { failures.Enqueue(e); }
                finally { queue.CompleteAdding(); }
            }) { IsBackground = true };

            var consumer = new Thread(() =>
            {
                try
                {
                    foreach (var (view, tag) in queue.GetConsumingEnumerable())
                    {
                        Thread.Sleep(5);
                        ReadAt<double>(view, 3).Should().BeApproximately(3.0 * tag, 1e-9,
                            $"item {tag}: the lease must have kept python's buffer alive across the handoff");
                        WriteAt(view, -1.0, 0);
                        ReadAt<double>(view, 0).Should().BeApproximately(-1.0, 1e-12);
                        view.Dispose();
                    }
                }
                catch (Exception e) { failures.Enqueue(e); }
            }) { IsBackground = true };

            producer.Start();
            consumer.Start();
            producer.Join(30_000).Should().BeTrue("producer must not deadlock");
            consumer.Join(30_000).Should().BeTrue("consumer must not deadlock");

            failures.Should().BeEmpty(string.Join(" | ", failures));
            WaitFor(() => NDArrayPythonInterop.LiveImports == i0).Should().BeTrue("all handed-off leases must drain");
        }

        // ------------------------------------------------------------------------------------------
        // closures and async flows
        // ------------------------------------------------------------------------------------------

        /// <summary>
        ///     "Lazy computation graph": thunks capture conversions in closures and are evaluated long
        ///     after the building method returned. The closure display classes are the only C#-side
        ///     references; dropping the thunk list must release everything.
        /// </summary>
        [TestMethod]
        public void LazyClosures_CaptureConversions_EvaluateAfterTheirCreatorDied()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;

            [MethodImpl(MethodImplOptions.NoInlining)]
            List<Func<double>> BuildThunks()
            {
                var thunks = new List<Func<double>>();

                PyExec("cl_src = np.arange(10, dtype='f8')");
                var view = ViewOf("cl_src");
                PyExec("del cl_src");
                thunks.Add(() => ReadAt<double>(view, 7));                       // captures the leased view

                ExportTo("cl_exp", np.arange(5).astype(NPTypeCode.Double) * 3);  // rooted by the scope name
                thunks.Add(() => PyFloat("float(cl_exp.sum())"));

                var reduced = np.sum(np.arange(4).astype(NPTypeCode.Double));    // plain NumSharp capture for contrast
                thunks.Add(() => ReadAt<double>(reduced));

                return thunks;
            }

            [MethodImpl(MethodImplOptions.NoInlining)]
            void EvaluatePhase()
            {
                var thunks = BuildThunks();
                Pump(); Pump();   // the building frame is gone; only the closures root the conversions

                thunks[0]().Should().BeApproximately(7.0, 1e-12, "closure-held import view");
                thunks[1]().Should().BeApproximately(30.0, 1e-9, "closure-evaluated export");
                thunks[2]().Should().BeApproximately(6.0, 1e-12, "closure-held NumSharp scalar");
                thunks[0]().Should().BeApproximately(7.0, 1e-12, "thunks must stay repeatable");
            }

            EvaluatePhase();
            PyExec("del cl_exp");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0)
                .Should().BeTrue("dropping the thunk list must release the captured conversions");
        }

        /// <summary>
        ///     "Async service flow": conversions created before awaits are consumed in continuations
        ///     that may resume on different ThreadPool threads, with GC churn in between. The async
        ///     state machine is the reference carrier.
        /// </summary>
        [TestMethod]
        public async Task AsyncPipeline_ConversionsSurviveAwaitBoundaries()
        {
            var nd = np.arange(8).astype(NPTypeCode.Double);
            ExportTo("async_x", nd);

            PyExec("async_src = np.arange(8, dtype='f8') * 2");
            var view = ViewOf("async_src");
            PyExec("del async_src");

            int beforeThread = Environment.CurrentManagedThreadId;
            await Task.Run(() => { Pump(); Pump(); });   // GC churn on a worker thread

            PyFloat("float(async_x.sum())").Should().BeApproximately(28.0, 1e-9,
            	"the export must survive the await boundary");
            ReadAt<double>(view, 5).Should().BeApproximately(10.0, 1e-12,
            	"the leased view must survive the await boundary");

            double fromWorker = await Task.Run(() =>
            {
                WriteAt(view, -9.0, 0);                  // lazy write from a ThreadPool thread
                return PyFloat("float(async_x[1])");     // lazy python read from a ThreadPool thread
            });
            fromWorker.Should().BeApproximately(1.0, 1e-12);

            await Task.Delay(25);
            ReadAt<double>(view, 0).Should().BeApproximately(-9.0, 1e-12, "the worker's write must persist");
            WriteAt(nd, 41.0, 2);
            PyFloat("float(async_x[2])").Should().BeApproximately(41.0, 1e-12);

            // Silence the "did the continuation actually hop" curiosity without asserting on it:
            // thread identity is scheduler business; correctness above must hold either way.
            _ = beforeThread;
        }

        // ------------------------------------------------------------------------------------------
        // reference-loss and premature-disposal edges
        // ------------------------------------------------------------------------------------------

        /// <summary>
        ///     A Python container is the LAST holder of an exported array after both the scope name and
        ///     the C# wrapper are gone; a second wrapper for the same object must survive disposal of
        ///     the first (independent references, not aliases).
        /// </summary>
        [TestMethod]
        public void PythonContainer_AndSecondWrapper_EachKeepTheExportAlive()
        {
            int e0 = NDArrayPythonInterop.LiveExports;
            PyExec("holder = []");

            var nd = np.arange(6).astype(NPTypeCode.Double);
            using (Gil())
            {
                using var p = NDArrayPythonInterop.ToNumpy(nd);
                Scope.Set("tmp_h", p);
            }
            PyExec("holder.append(tmp_h)\ndel tmp_h");   // wrapper disposed, scope name deleted
            Pump(); Pump();

            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 1, "the python list is a real reference");
            PyFloat("float(holder[0].sum())").Should().BeApproximately(15.0, 1e-9);

            // second-wrapper independence
            using (Gil())
            {
                var w1 = Scope.Eval("holder[0]");
                var w2 = Scope.Eval("holder[0]");
                w1.Dispose();
                Scope.Set("still", w2);
                w2.Dispose();
            }
            PyBool("still is holder[0]").Should().BeTrue("disposing one wrapper must not invalidate another");

            PyExec("holder.clear()\ndel still");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue(
                "emptying the container releases the last reference");
        }

        /// <summary>
        ///     Premature-disposal guard: explicitly disposing the ORIGINAL imported view must not free
        ///     the buffer under a derived view that shares it — NumSharp's refcount, not the disposal
        ///     order, decides when the lease ends. Double-dispose must be a no-op.
        /// </summary>
        [TestMethod]
        public void DisposingTheOriginal_DoesNotFreeUnderADerivedView()
        {
            int i0 = NDArrayPythonInterop.LiveImports;
            PyExec("pd_src = np.arange(8, dtype='f8')");

            [MethodImpl(MethodImplOptions.NoInlining)]
            void Lifecycle()
            {
                var original = ViewOf("pd_src");
                var derived = original["2:6"];

                original.Dispose();
                original.Dispose();   // idempotent
                original.IsDisposed.Should().BeTrue();

                Pump();
                NDArrayPythonInterop.LiveImports.Should().Be(i0 + 1,
                    "the lease must survive the original's explicit disposal while the derived view lives");

                ReadAt<double>(derived, 0).Should().BeApproximately(2.0, 1e-12, "derived data intact after the disposal");
                WriteAt(derived, -4.0, 1);
                PyFloat("float(pd_src[3])").Should().BeApproximately(-4.0, 1e-12, "derived view still aliases python");
            }

            Lifecycle();
            WaitFor(() => NDArrayPythonInterop.LiveImports == i0).Should().BeTrue(
                "the derived view's death is what ends the lease");
        }

        /// <summary>
        ///     The crown-jewel chain: python array → NumSharp view (lease) → re-exported numpy view
        ///     (pin) → stored in a python dict. Then EVERY other reference is destroyed — python's
        ///     original name, the C# NDArray reference, the C# wrapper. The dict entry alone must keep
        ///     the entire transitive chain alive; clearing it must tear everything down in order,
        ///     verified by a weakref on the root array.
        /// </summary>
        [TestMethod]
        public void TransitiveChain_DictEntryAlone_KeepsRootAlive_TeardownCascades()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;
            PyExec("import weakref\nchain_root = np.arange(5, dtype='f8') * 2\nwr_root = weakref.ref(chain_root)");

            [MethodImpl(MethodImplOptions.NoInlining)]
            void BuildChain()
            {
                var view = ViewOf("chain_root");             // lease on the root
                ExportTo("reexp", view);                     // pin on the view's buffer == the root's buffer
                PyExec("store = {'d': reexp}\ndel reexp\ndel chain_root");
                // Only chain now: store dict -> numpy re-export -> base buffer object -> export keeper
                //                -> NDArray view (held by the keeper) -> lease -> root array.
            }

            BuildChain();
            Pump(); Pump();

            PyBool("wr_root() is not None").Should().BeTrue("the dict entry must transitively root the original array");
            NDArrayPythonInterop.LiveExports.Should().Be(e0 + 1);
            NDArrayPythonInterop.LiveImports.Should().Be(i0 + 1, "the keeper holds the C# view alive, which holds the lease");
            PyFloat("float(store['d'].sum())").Should().BeApproximately(20.0, 1e-9, "lazy access straight through the chain");
            PyExec("store['d'][0] = -8.0");
            PyBool("wr_root() is not None").Should().BeTrue();

            PyExec("store.clear()");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0, 15_000)
                .Should().BeTrue("clearing the dict must cascade: keeper releases -> view collects -> lease releases");
            Pump();
            PyBool("wr_root() is None").Should().BeTrue("the root array itself must be freed at the end of the cascade");
        }

        /// <summary>
        ///     GIL-ordering regression: a thread that HOLDS the GIL disposes a view (release must only
        ///     enqueue, never block) and keeps converting (inline drains run re-entrantly under the held
        ///     GIL). The original drain design deadlocked exactly here.
        /// </summary>
        [TestMethod]
        public void HeldGil_DisposeAndConvert_NeverBlocks()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;
            PyExec("g_src = np.arange(4, dtype='f8')");
            var view = ViewOf("g_src");

            var sw = Stopwatch.StartNew();
            using (Gil())
            {
                view.Dispose();   // enqueue-only: must not wait for the drain gate or the GIL

                using var copy = NDArrayPythonInterop.ToNumpyCopy(np.arange(3));       // runs the inline drain under our GIL
                using var export = NDArrayPythonInterop.ToNumpy(np.arange(3));         // full rooting cycle under our GIL
                using var import = Scope.Eval("np.arange(3, dtype='i8')");
                NDArrayPythonInterop.ToNDArray(import).size.Should().Be(3);
            }

            sw.ElapsedMilliseconds.Should().BeLessThan(10_000, "no path may block against the held GIL");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0 && NDArrayPythonInterop.LiveImports == i0)
                .Should().BeTrue("the deferred disposal must drain once the GIL owner moved on");
        }
    }
}
