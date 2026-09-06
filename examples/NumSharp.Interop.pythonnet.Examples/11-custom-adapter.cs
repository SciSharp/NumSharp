#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 11 — A custom IPythonArrayAdapter. Any Python library object that is not itself a buffer
//      exporter can be given a front door to the existing memory bridge: the adapter only picks the
//      library's canonical numpy/buffer object, and NDArrayPythonInterop still owns dtype, shape,
//      strides, writeability, leases, copy fallback, GIL and shutdown. Registered once, process-wide,
//      the adapter serves the direct verbs (AsNDArray / ToNDArray) AND the registered codec
//      (As<NDArray>(), `NDArray a = py.obj`). The built-in Torch and Pandas adapters are exactly this.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example11_CustomAdapter.cs.
//
//      Run:  dotnet run 11-custom-adapter.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

// A tiny "image library": Tile wraps pixels + metadata (no buffer protocol of its own), and
// LazyTile computes its pixels on demand — sharing them is impossible, materializing is a copy.
using (Py.GIL())
    py.Exec(
        "class Tile:\n" +
        "    def __init__(self, pixels, name):\n" +
        "        self.pixels = pixels\n" +
        "        self.name = name\n" +
        "\n" +
        "class LazyTile:\n" +
        "    def __init__(self, n):\n" +
        "        self.n = n\n" +
        "    def materialize(self):\n" +
        "        return np.arange(self.n, dtype='f8') ** 2\n" +
        "\n" +
        "tile = Tile(np.arange(6, dtype='f8').reshape(2, 3), 'north')\n" +
        "lazy = LazyTile(4)");

// ==== registration: process-wide, thread-safe, idempotent by Name ====================================
bool added = NDArrayPythonInterop.RegisterArrayAdapter(new TileAdapter());                   // true
bool addedTwice = PythonArrayAdapterRegistry.Register(new TileAdapter());                    // false: the same Name again (RegisterArrayAdapter is the same call)
bool torchBuiltIn = PythonArrayAdapterRegistry.Register(TorchPythonArrayAdapter.Instance);   // false: 'torch.Tensor' is already there...
bool pandasBuiltIn = PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance); // false: ...and so is 'pandas'
Exception? nameless = Throws(() => PythonArrayAdapterRegistry.Register(new Nameless()));    // ArgumentException: an adapter must have a non-empty Name

// ==== the direct verbs: the adapter selects `pixels`, the bridge does the rest =======================
using (Py.GIL())
{
    using PyObject tile = py.tile;
    NDArray view = tile.AsNDArray();                     // adapter (allowCopy: false) -> pixels -> the ordinary view bridge: a writable float64 (2,3) view
    view[1, 2] = 71.0;                                   // tile.pixels[1, 2] reads 71.0 — the adapter chose the object, the bridge shares its memory
    NDArray copy = tile.ToNDArray();                     // adapter (allowCopy: true) -> the ordinary copy bridge: detached
    copy[0, 0] = -8.0;                                   // tile.pixels[0, 0] still reads 0.0
}

// declining: `null` from Adapt means "not this one"; the registry tries the next adapter
using (Py.GIL())
{
    using PyObject lazy = py.lazy;
    Exception? cannotShare = Throws(() => lazy.AsNDArray());   // NotSupportedException: ... IPythonArrayAdapter — LazyTile declines for allowCopy: false and exports no buffer
    NDArray materialized = lazy.ToNDArray();                   // allowCopy: true -> materialize(): [0, 1, 4, 9]
}

// ==== the registered codec consults the same registry ================================================
using (Py.GIL())
{
    NDArray auto = py.tile;                              // Auto: View first — a shared view of tile.pixels
    auto[0, 1] = 5.5;                                    // tile.pixels[0, 1] reads 5.5
    NDArray lazyAuto = py.lazy;                          // Auto: View declined -> the Copy route -> materialize(): lazyAuto[2] == 4.0, writable

    var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
    using PyObject lazy = py.lazy;
    bool declined = !viewOnly.TryDecode(lazy, out NDArray none);   // true: a View-mode codec declines LazyTile instead of copying (none == null)

    var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
    using PyObject tile = py.tile;
    using PyObject t = tile.GetPythonType();
    using var tileType = new PyType(t);
    bool decodable = noAdapters.CanDecode(tileType, typeof(NDArray));   // false: DecodeArrayAdapters = false — Tile has no buffer of its own
}

// ==== robustness: a broken adapter cannot break the pipeline =========================================
bool broken = PythonArrayAdapterRegistry.Register(new ThrowingAdapter());   // true: an adapter whose CanAdapt THROWS is registered like any other...
using (Py.GIL())
{
    NDArray still = py.tile;                             // ...and skipped: the registry swallows its exception and TileAdapter still serves the object (still[0, 1] == 5.5)
}

// ==== who frees what ================================================================================
// As in the other examples: `scope` releases every view (the leases on tile.pixels go back), then the
// namespace, then the engine. LiveExports / LiveImports both read 0 afterwards.

// =====================================================================================================
//  Scaffolding — identical in every example (a single-file app cannot share source files). Nothing
//  below is specific to this script: Throws captures an exception the tutorial expects, PythonHost
//  finds CPython, runs the engine and owns a Python namespace with numpy imported.
// =====================================================================================================

static Exception? Throws(Action act) { try { act(); return null; } catch (Exception e) { return e; } }

/// <summary>
///     Finds a CPython that has numpy, starts the engine once, registers the codec, opens a Python
///     namespace with numpy imported, and shuts everything down crash-free on dispose.
/// </summary>
sealed class PythonHost : IDisposable
{
    /// <summary>
    ///     A Python module where <c>import numpy as np</c> already ran. Hold it as <c>dynamic py</c> and use
    ///     it under <c>Py.GIL()</c>: <c>py.x = nd</c> binds a NumSharp array as a zero-copy numpy view (the
    ///     codec runs ToNumpy), <c>NDArray a = py.x</c> decodes one back (AsNDArray whenever a view is
    ///     possible), <c>py.Exec("...")</c> runs statements, <c>double d = py.Eval("x.sum()")</c> reads a
    ///     value, and <c>py.np</c> is numpy itself.
    /// </summary>
    public PyModule Namespace { get; private set; } = null!;
    bool _down;

    /// <param name="codec">Register the codec at startup — the normal choice. Two examples pass <c>false</c>
    /// to show the registration order themselves.</param>
    public static PythonHost Start(bool codec = true)
    {
        Runtime.PythonDLL = Environment.GetEnvironmentVariable("PYTHONNET_PYDLL") ?? Discover();
        PythonEngine.Initialize();                          // once per process (CPython + numpy cannot re-initialize)
        PythonEngine.BeginAllowThreads();                   // release THIS thread's GIL so every thread can take it later
        if (codec) NDArrayPythonInterop.RegisterCodec();    // NDArray <-> numpy and C# tuples <-> Python tuples at every pythonnet boundary; before any As<NDArray>()
        var host = new PythonHost();
        using (Py.GIL()) { host.Namespace = Py.CreateScope(); host.Namespace.Exec("import numpy as np"); }
        return host;
    }

    public void Dispose()
    {
        if (_down) return;
        _down = true;
        using (Py.GIL()) Namespace.Dispose();               // a PyObject is disposed under the GIL — and never after Shutdown()
        RuntimeData.FormatterType = typeof(NoopFormatter);  // pythonnet's opt-out of BinaryFormatter stashing (.NET 8+)
        PythonEngine.Shutdown();                            // the interop's handler drains every lease first
    }

    /// <summary>
    ///     Runs both collectors to completion — CLR GC + finalizers, pythonnet's deferred decrefs, Python's
    ///     gc — plus one trivial conversion, which runs the interop's inline lease drain. The live counters
    ///     are exact after this.
    /// </summary>
    public static void Drain()
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        using (Py.GIL())
        {
            Finalizer.Instance.Collect();                   // wrappers the CLR GC finalized (`dynamic` temporaries, for instance) release their Python reference here
            PythonEngine.RunSimpleString("import gc; gc.collect()");
            using NDArray probe = np.arange(1);
            using PyObject conversion = probe.ToNumpyCopy();
        }
    }

    // Ask the `python` on PATH where its shared library lives (what PYTHONNET_PYDLL would name).
    static string Discover()
    {
        const string probe = "import sys,sysconfig,numpy;print(sys.base_prefix);print(sys.version_info.major);" +
                             "print(sys.version_info.minor);print(sysconfig.get_config_var('INSTSONAME') or '');" +
                             "print(sysconfig.get_config_var('LIBDIR') or '')";
        string[] lines = null!;
        foreach (string exe in new[] { "python", "python3" })                 // whichever spelling the platform puts on PATH
        {
            try
            {
                var psi = new ProcessStartInfo(exe, $"-c \"{probe}\"") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
                using Process p = Process.Start(psi)!;
                string[] output = p.StandardOutput.ReadToEnd().Split('\n', StringSplitOptions.TrimEntries);
                p.WaitForExit();
                if (p.ExitCode == 0 && output.Length >= 5) { lines = output; break; }
            }
            catch (System.ComponentModel.Win32Exception) { }                 // not on PATH — try the next spelling
        }
        if (lines is null)
            throw new InvalidOperationException("no `python`/`python3` on PATH that can `import numpy`; install requirements.txt or set PYTHONNET_PYDLL");

        string prefix = lines[0];
        int major = int.Parse(lines[1]), minor = int.Parse(lines[2]);
        if (OperatingSystem.IsWindows())
            return Path.Combine(prefix, $"python{major}{minor}.dll");

        string soname = lines[3], libdir = lines[4], ext = OperatingSystem.IsMacOS() ? "dylib" : "so";
        foreach (string candidate in new[]
                 {
                     Path.Combine(libdir, soname),                          // unix shared build
                     Path.Combine(prefix, Path.GetFileName(soname)),        // macOS framework build
                     Path.Combine(libdir, $"libpython{major}.{minor}.{ext}"),
                     Path.Combine(prefix, "lib", $"libpython{major}.{minor}.{ext}"),
                 })
            if (File.Exists(candidate))
                return candidate;
        throw new InvalidOperationException("libpython not found; set PYTHONNET_PYDLL to its full path");
    }
}

// =====================================================================================================
//  The adapter itself — the whole of what a library integration has to write.
// =====================================================================================================

/// <summary>Recognize the type, hand back the library's canonical array object, or decline.</summary>
sealed class TileAdapter : IPythonArrayAdapter
{
    public string Name => "examples.Tile";                                     // unique, stable — registration is idempotent by Name

    public bool CanAdapt(PyType objectType)                                    // type-level, under the GIL; must not throw
    {
        try
        {
            using PyObject name = objectType.GetAttr("__name__");
            string n = name.As<string>();
            return n == "Tile" || n == "LazyTile";
        }
        catch { return false; }
    }

    public PyObject Adapt(PyObject source, bool allowCopy)                     // a NEW owned wrapper the bridge disposes, or null to decline
    {
        using PyObject typeObj = source.GetPythonType();
        using var type = new PyType(typeObj);
        if (type.Name == "Tile")
            return source.GetAttr("pixels");                                   // shares memory either way: fine for View AND Copy
        return allowCopy ? source.InvokeMethod("materialize") : null!;         // LazyTile: only a copy can be honest
    }
}

sealed class ThrowingAdapter : IPythonArrayAdapter
{
    public string Name => "examples.Throwing";
    public bool CanAdapt(PyType objectType) => throw new InvalidOperationException("a third-party probe that fails");
    public PyObject Adapt(PyObject source, bool allowCopy) => null!;
}

sealed class Nameless : IPythonArrayAdapter
{
    public string Name => "";
    public bool CanAdapt(PyType objectType) => false;
    public PyObject Adapt(PyObject source, bool allowCopy) => null!;
}
