using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharpPythonnetGhActionsCicd;

/// <summary>
///     Base class of every test that talks to Python. It gives each test its own Python namespace, and it
///     fails a test that leaves memory shared between NumSharp and Python after the test ended.
/// </summary>
/// <remarks>
///     <para><b>Namespace.</b> <see cref="py"/> is a new Python module scope for each test, with
///     <c>import numpy as np</c> already run. A name that one test binds cannot reach the next test.</para>
///     <para><b>Leak check.</b> A NumSharp array that Python views stays pinned until the last Python view
///     dies, and Python memory that NumSharp views stays leased until the last NumSharp view dies.
///     <see cref="NDArrayPythonInterop.LiveExports"/> and <see cref="NDArrayPythonInterop.LiveImports"/>
///     count both. After each test they must return to the value they had before it.</para>
/// </remarks>
public abstract class PythonTest
{
    /// <summary>
    ///     The Python namespace of the running test, as <c>dynamic</c>: <c>py.x = array</c> binds a NumSharp
    ///     array as a NumPy view, <c>NDArray a = py.x</c> reads one back, <c>py.Exec("...")</c> runs
    ///     statements, <c>py.Eval("...")</c> reads a value, and <c>py.np</c> is NumPy. Use it only inside a
    ///     <c>using (Py.GIL())</c> block.
    /// </summary>
    protected dynamic py = null!;

    /// <summary>MSTest sets this before each test. <see cref="MatplotlibTests"/> uses it to attach the figure it renders to the test results.</summary>
    public TestContext TestContext { get; set; } = null!;

    /// <summary>The module scope behind <see cref="py"/>; <c>null</c> when the test did not start (no engine).</summary>
    private PyModule? _namespace;

    /// <summary>The value of <see cref="NDArrayPythonInterop.LiveExports"/> before the test.</summary>
    private int _exportsBefore;

    /// <summary>The value of <see cref="NDArrayPythonInterop.LiveImports"/> before the test.</summary>
    private int _importsBefore;

    /// <summary>
    ///     Reports the test as Inconclusive when no engine runs (a developer machine without the variable),
    ///     then records the leak counters and opens the namespace.
    /// </summary>
    /// <exception cref="AssertInconclusiveException">No engine runs: <see cref="PythonEnvironment.Skipped"/> is the reason.</exception>
    [TestInitialize]
    public void OpenNamespace()
    {
        if (PythonEnvironment.Skipped is string reason)
            Assert.Inconclusive(reason);

        _exportsBefore = NDArrayPythonInterop.LiveExports;
        _importsBefore = NDArrayPythonInterop.LiveImports;

        using (Py.GIL())
        {
            _namespace = Py.CreateScope();
            _namespace.Exec("import numpy as np");
        }

        py = _namespace;
    }

    /// <summary>
    ///     Drops the namespace, then waits for both leak counters to return to their value before the test.
    /// </summary>
    /// <remarks>
    ///     The release is not immediate, which is why this waits instead of comparing once: an object that
    ///     <c>dynamic</c> code read is a .NET wrapper, and its Python reference is released only after the
    ///     .NET garbage collector finalized the wrapper and pythonnet processed its release queue.
    /// </remarks>
    /// <exception cref="AssertFailedException">
    ///     After 15 seconds of collecting, a NumSharp buffer is still pinned by Python or Python memory is
    ///     still leased by NumSharp: the test (or the code under test) keeps a reference it must release.
    /// </exception>
    [TestCleanup]
    public void CloseNamespaceAndCheckForLeaks()
    {
        if (_namespace is null)
            return;

        // A PyObject is disposed under the GIL, like every other use of it.
        using (Py.GIL())
            _namespace.Dispose();
        _namespace = null;

        var deadline = Stopwatch.StartNew();
        while (!CountersAreBack() && deadline.Elapsed < TimeSpan.FromSeconds(15))
            Collect();

        Assert.IsTrue(CountersAreBack(),
            $"The test left memory shared with Python: LiveExports {_exportsBefore} -> {NDArrayPythonInterop.LiveExports}, " +
            $"LiveImports {_importsBefore} -> {NDArrayPythonInterop.LiveImports}.");
    }

    /// <summary>Tells whether both leak counters are at, or below, their value before the test.</summary>
    /// <returns>True when nothing that this test shared is still alive.</returns>
    /// <remarks>"Or below" matters after a failed test: what it left behind is released during a later test.</remarks>
    private bool CountersAreBack() =>
        NDArrayPythonInterop.LiveExports <= _exportsBefore && NDArrayPythonInterop.LiveImports <= _importsBefore;

    /// <summary>
    ///     Runs both garbage collectors once: the .NET collector and its finalizers, pythonnet's queue of
    ///     deferred releases, Python's collector, and one small conversion, which makes the interop package
    ///     process the releases it deferred.
    /// </summary>
    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using (Py.GIL())
        {
            Finalizer.Instance.Collect();
            PythonEngine.RunSimpleString("import gc; gc.collect()");
            using NDArray probe = np.arange(1);
            using PyObject conversion = probe.ToNumpyCopy();
        }
    }
}
