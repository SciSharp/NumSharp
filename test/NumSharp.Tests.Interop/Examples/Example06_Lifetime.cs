using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>
    ///     The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/06-lifetime.cs</c>, section by section.
    ///     Every block whose temporaries must be collectable before a counter is read runs in its own
    ///     <c>[MethodImpl(NoInlining)]</c> frame (the GC-realism rule of <see cref="InteropTestBase"/>).
    /// </summary>
    [TestClass]
    public class Example06_Lifetime : ExampleTestBase
    {
        [TestInitialize]
        public void ImportTheStdlib() => PyExec("import weakref, gc");

        [TestMethod]
        public void TheCounters_MoveByExactlyOnePerCrossing()
        {
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            using var scope = NDScope.Open();
            NDArray nd = np.arange(4).astype(np.float64);
            NDArray leased;
            using (Gil())
            {
                py.c1 = nd;
                NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 1, "one export");
                py.Exec("c2 = np.arange(3, dtype='i8')");
                using PyObject c2 = py.c2;
                leased = c2.AsNDArray();
                NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "one import view");
            }

            leased.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == imports0).Should().BeTrue("Dispose released the lease (the release is marshaled to the GIL)");
            PyExec("del c1");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0).Should().BeTrue("the last numpy name died, and the pin with it");
        }

        [TestMethod]
        public void AnExportOutlivesEveryManagedReference()
        {
            int exports0 = NDArrayPythonInterop.LiveExports;
            ExportAndForget();
            Drain();
            NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 1, "the pin is held for Python");
            PyStr("orphan.tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0]", "Python still reads valid memory");
            PyExec("orphan[0] = 7.0");
            PyStr("orphan.tolist()").Should().Be("[7.0, 1.0, 2.0, 3.0]", "...and still writes it");

            PyExec("child = orphan[2:]; del orphan");
            Drain();
            NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 1, "a derived view keeps the buffer rooted");
            PyStr("child.tolist()").Should().Be("[2.0, 3.0]");
            PyExec("del child");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0).Should().BeTrue("the last derived view released the pin");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void ExportAndForget()
        {
            using (Gil())
            using (NDScope.Open())
            {
                NDArray temp = np.arange(4).astype(np.float64);
                py.orphan = temp;                                   // Python's name is the only holder once the scope closes
            }
        }

        [TestMethod]
        public void AnImportLeaseOutlivesEveryPythonReference()
        {
            int imports0 = NDArrayPythonInterop.LiveImports;
            using var scope = NDScope.Open();
            PyExec("keep = np.arange(6, dtype='f8') * 1.5\nwr = weakref.ref(keep)");
            NDArray slice = TakeASliceAndLetPythonForget();
            Drain();
            NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "the slice alone keeps the lease; disposal order is irrelevant, the refcount decides");
            PyBool("wr() is not None").Should().BeTrue("NumSharp holds the exporter alive");
            slice.GetDouble(0).Should().Be(3.0);
            slice.GetDouble(2).Should().Be(6.0);

            slice.Dispose();
            WaitFor(() => NDArrayPythonInterop.LiveImports == imports0).Should().BeTrue("the last view released the lease");
            WaitFor(() => PyBool("wr() is None")).Should().BeTrue("...and Python freed the exporter");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private NDArray TakeASliceAndLetPythonForget()
        {
            NDArray slice;
            using (Gil())
            {
                using (var inner = NDScope.Open())
                {
                    using PyObject keep = py.keep;
                    NDArray whole = keep.AsNDArray();
                    slice = inner.Returns(whole["2:5"]);            // yielded to the outer scope; `whole` is released here
                }

                py.Exec("del keep; gc.collect()");
            }

            return slice;
        }

        [TestMethod]
        public void ALiveViewLocksReallocation_OnBothSides()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("ba = bytearray(b'abcd')\nnpa = np.arange(4, dtype='f8')");
                HoldBothAndTryToResize();
                Drain();
                py.Exec("ba.append(1)\nnpa.resize((8,), refcheck=True)");
                PyLong("len(ba)").Should().Be(5, "the lock lifted");
                PyLong("npa.size").Should().Be(8);

                NDArray pinned = np.arange(8).astype(np.float64);
                py.pin = pinned;
                Exception mirror = Throws(() => pinned.resize(new Shape(16)));
                mirror.Should().NotBeNull("NumSharp's resize refuses while Python holds a view");
                mirror.Message.Should().Contain("references or is referenced");
                py.Exec("del pin");
                Drain();
                pinned.resize(new Shape(16));
                pinned.size.Should().Be(16, "...and succeeds once the pin is gone");
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void HoldBothAndTryToResize()
        {
            using (NDScope.Open())
            {
                using PyObject ba = py.ba, npa = py.npa;
                NDArray held = ba.AsNDArray(), held2 = npa.AsNDArray();
                Throws(() => py.Exec("ba.append(1)")).Should().BeOfType<PythonException>()
                    .Which.Message.Should().Contain("Existing exports of data");
                Throws(() => py.Exec("npa.resize((8,), refcheck=True)")).Should().BeOfType<PythonException>()
                    .Which.Message.Should().Contain("references or is referenced");
            }
        }

        [TestMethod]
        public void Churn_EverythingReturnsToBaseline()
        {
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            for (int i = 0; i < 120; i++)
                ChurnOnce(i);
            PyExec("del h, hc, h2");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0 && NDArrayPythonInterop.LiveImports == imports0, 15_000)
                .Should().BeTrue($"exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports}");
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private void ChurnOnce(int i)
        {
            using (NDScope.Open())
            using (Gil())
            {
                NDArray a = np.arange(32).astype(np.float64) + i;
                py.h = a;
                py.hc = a.ToNumpyCopy();
                py.Exec("h2 = np.arange(16, dtype='i8')");
                NDArray view = py.h2;
                NDArray sub = view["4:12"];
                sub[0] = (long)i;
            }
        }

        [TestMethod]
        public void EngineShutdown_IsProvenByShutdownLeakTests_ThisPinsWhatGoesIn()
        {
            // The script leaves one export held only by Python and one import still referenced by C#, then shuts the
            // engine down. A shared-engine suite cannot shut it down mid-run: ShutdownLeakTests stages exactly those
            // two orphans and PythonSession.Stop asserts the sweep and the force-drain after the real Shutdown().
            // Pinned here: the state going in, and that letting go the ordinary way returns both counters to baseline.
            int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;
            NDArray survivor = StageTheOrphans(exports0, imports0);
            survivor.Dispose();
            PyExec("del shutdown_orphan, shutdown_import");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports0 && NDArrayPythonInterop.LiveImports == imports0)
                .Should().BeTrue();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private NDArray StageTheOrphans(int exports0, int imports0)
        {
            NDArray survivor;
            using (Gil())
            {
                using (NDScope.Open())
                {
                    NDArray orphanSource = np.arange(6).astype(np.float64);
                    py.shutdown_orphan = orphanSource;
                }

                py.Exec("shutdown_import = np.arange(8) * 2.0");
                using PyObject s = py.shutdown_import;
                survivor = s.AsNDArray();
            }

            Drain();
            NDArrayPythonInterop.LiveExports.Should().Be(exports0 + 1, "one orphan export going in");
            NDArrayPythonInterop.LiveImports.Should().Be(imports0 + 1, "one live import going in");
            return survivor;
        }
    }
}
