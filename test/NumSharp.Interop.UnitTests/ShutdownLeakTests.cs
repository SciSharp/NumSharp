using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     The engine-shutdown half of the lifetime contract. These tests deliberately leave
    ///     conversions ALIVE when the run ends; the assertions live in
    ///     <see cref="PythonSession.Stop"/>, right after the real <c>PythonEngine.Shutdown()</c> —
    ///     the only place a full engine teardown can be observed in-process (CPython + numpy cannot
    ///     re-initialize, so no ordinary test may shut the engine down).
    ///
    ///     <para><b>What is being proven</b> (probed on pythonnet 3.0.5 / CPython 3.12):
    ///     pythonnet's <c>Shutdown</c> never runs Python's <c>atexit</c> machinery, so
    ///     <c>weakref.finalize</c> callbacks for still-referenced exports DO NOT fire at engine
    ///     death — without a shutdown sweep, every export still held by Python leaks its NumSharp
    ///     buffer for the rest of the process. The sweep must release those pins right AFTER the
    ///     engine finishes dying (never during teardown, when Python could still read the memory),
    ///     and the import-side force-drain must leave later <c>Dispose</c> calls harmless.</para>
    ///
    ///     <para>Deliberately NOT derived from <see cref="InteropTestBase"/>: its per-test leak gate
    ///     would (correctly) fail a test that leaves a live conversion behind. Later tests are
    ///     unaffected — their gates baseline whatever is live when they start.</para>
    /// </summary>
    [TestClass]
    public class ShutdownLeakTests
    {
        /// <summary>Buffer of the orphaned export; Stop() asserts the sweep actually freed it.</summary>
        internal static IArraySlice OrphanExportSlice;

        /// <summary>Import view still referenced by C# when the engine dies; Stop() disposes it after.</summary>
        internal static NDArray OrphanImportView;
        internal static IArraySlice OrphanImportSlice;

        [TestMethod]
        public void OrphanExport_HeldOnlyByPython_AwaitsTheShutdownSweep()
        {
            PythonSession.EnsureOrInconclusive();
            int before = NDArrayPythonInterop.LiveExports;

            var nd = np.arange(6).astype(NPTypeCode.Double);
            using (Py.GIL())
            {
                using PyObject arr = NDArrayPythonInterop.ToNumpy(nd);
                using PyObject main = Py.Import("__main__");
                main.SetAttr("shutdown_orphan_export", arr);   // Python is now the ONLY holder
            }

            OrphanExportSlice = nd.Storage.InternalArray;
            nd.Dispose();   // C# reference gone; the export pin alone keeps the buffer alive

            NDArrayPythonInterop.LiveExports.Should().Be(before + 1, "the orphan must be pinned for Python");
            OrphanExportSlice.IsReleased.Should().BeFalse("the buffer must stay alive while Python can see it");
            // ... release is asserted in PythonSession.Stop, after PythonEngine.Shutdown().
        }

        [TestMethod]
        public void OrphanImport_StillReferencedByCSharp_AwaitsTheShutdownDrain()
        {
            PythonSession.EnsureOrInconclusive();
            int before = NDArrayPythonInterop.LiveImports;

            using (Py.GIL())
            {
                PythonEngine.RunSimpleString("import numpy as _np\nshutdown_orphan_import = _np.arange(8) * 2.0");
                using PyObject main = Py.Import("__main__");
                using PyObject src = main.GetAttr("shutdown_orphan_import");
                OrphanImportView = NDArrayPythonInterop.ToNDArrayView(src);
            }

            OrphanImportSlice = OrphanImportView.Storage.InternalArray;
            NDArrayPythonInterop.LiveImports.Should().Be(before + 1, "the orphan lease must be live");
            // ... the force-drain and the safe post-shutdown Dispose are asserted in PythonSession.Stop.
        }
    }
}
