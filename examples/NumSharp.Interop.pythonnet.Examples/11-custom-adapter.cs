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
//      (As<NDArray>()). The built-in Torch and Pandas adapters are exactly this.
//
//      Run:  dotnet run 11-custom-adapter.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

// A tiny "image library": Tile wraps pixels + metadata (no buffer protocol of its own), and
// LazyTile computes its pixels on demand — sharing them is impossible, materializing is a copy.
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

// ==== registration ==================================================================================
check.Section("registration: process-wide, thread-safe, idempotent by Name");
check.That(NDArrayPythonInterop.RegisterArrayAdapter(new TileAdapter()), "RegisterArrayAdapter(new TileAdapter()) -> true");
check.That(!PythonArrayAdapterRegistry.Register(new TileAdapter()), "registering the same Name again -> false (PythonArrayAdapterRegistry.Register is the same call)");
check.That(!PythonArrayAdapterRegistry.Register(TorchPythonArrayAdapter.Instance) && !PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance),
           $"the built-ins are already there: '{TorchPythonArrayAdapter.Instance.Name}', '{PandasPythonArrayAdapter.Instance.Name}'");
check.Throws<ArgumentException>(() => PythonArrayAdapterRegistry.Register(new Nameless()), "an adapter must have a non-empty Name");

// ==== the direct verbs ==============================================================================
check.Section("tile.AsNDArray() / tile.ToNDArray(): the adapter selects `pixels`, the bridge does the rest");
using (Py.GIL())
{
    using PyObject tile = py.Get("tile");
    using NDArray view = tile.AsNDArray();                                     // adapter (allowCopy:false) -> pixels -> the ordinary view bridge
    check.That(view.shape.SequenceEqual(new long[] { 2, 3 }) && view.typecode == NPTypeCode.Double && view.Shape.IsWriteable, "AsNDArray: a writable float64 (2,3) view");
    view[1, 2] = 71.0;
    check.That(py.Float("tile.pixels[1, 2]") == 71.0, "a NumSharp write lands in tile.pixels — the adapter chose the object, the bridge shares its memory");
    using NDArray copy = tile.ToNDArray();                                     // adapter (allowCopy:true) -> the ordinary copy bridge
    copy[0, 0] = -8.0;
    check.That(py.Float("tile.pixels[0, 0]") == 0.0, "ToNDArray: an owning copy, detached");
}

check.Section("declining: `null` from Adapt means \"not this one\"; the registry tries the next adapter");
using (Py.GIL())
{
    using PyObject lazy = py.Get("lazy");
    check.Throws<NotSupportedException>(() => lazy.AsNDArray().Dispose(), "LazyTile cannot be SHARED (the adapter declines for allowCopy:false) and exports no buffer", "IPythonArrayAdapter");
    using NDArray materialized = lazy.ToNDArray();                             // allowCopy:true -> materialize()
    check.That(materialized.size == 4 && materialized.GetDouble(3) == 9.0, "ToNDArray materializes it: [0, 1, 4, 9]");
}

// ==== the codec =====================================================================================
check.Section("the registered codec consults the same registry");
NDArrayPythonInterop.RegisterCodec();
using (Py.GIL())
{
    using PyObject tile = py.Get("tile");
    using NDArray auto = tile.As<NDArray>();                                   // Auto: View first
    auto[0, 1] = 5.5;
    check.That(py.Float("tile.pixels[0, 1]") == 5.5, "tile.As<NDArray>() is a shared view");
    using PyObject lazy = py.Get("lazy");
    using NDArray lazyAuto = lazy.As<NDArray>();                               // Auto: View declined -> Copy route -> materialize()
    check.That(lazyAuto.GetDouble(2) == 4.0 && lazyAuto.Shape.IsWriteable, "lazy.As<NDArray>() falls back to the copy route");

    var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
    check.That(!viewOnly.TryDecode(lazy, out NDArray declined) && declined is null, "a View-mode codec declines LazyTile instead of copying");
    var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
    using PyObject t = tile.GetPythonType();
    using var tileType = new PyType(t);
    check.That(!noAdapters.CanDecode(tileType, typeof(NDArray)), "DecodeArrayAdapters = false: Tile is not decodable (it has no buffer of its own)");
}

check.Section("robustness: a broken adapter cannot break the pipeline");
check.That(PythonArrayAdapterRegistry.Register(new ThrowingAdapter()), "an adapter whose CanAdapt THROWS is registered like any other...");
using (Py.GIL())
{
    using PyObject tile = py.Get("tile");
    using NDArray still = tile.AsNDArray();
    check.That(still.GetDouble(0, 1) == 5.5, "...and is skipped: the registry swallows its exception and TileAdapter still serves the object");
}

check.Section("counters");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports == 0),
           $"every pin and lease released (exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports})");

return check.Exit();

static bool Settle(Func<bool> done)
{
    var sw = Stopwatch.StartNew();
    while (!done() && sw.ElapsedMilliseconds < 5000)
    {
        GC.Collect(); GC.WaitForPendingFinalizers();
        using (Py.GIL())
        {
            Finalizer.Instance.Collect();                                  // pythonnet's DEFERRED decrefs: wrappers the CLR GC finalized (e.g. `dynamic` call arguments) release their Python reference here
            PythonEngine.RunSimpleString("import gc; gc.collect()");
        }
        Thread.Sleep(20);
    }
    return done();
}

/// <summary>The whole adapter: recognize the type, hand back the canonical array object, or decline.</summary>
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

// =====================================================================================================
//  Shared scaffolding — identical in every example (a single-file app cannot share source files).
// =====================================================================================================

/// <summary>Finds a CPython that has numpy, starts the engine once, shuts it down crash-free on dispose.</summary>
sealed class PythonHost : IDisposable
{
    bool _down;

    public static PythonHost Start()
    {
        string dll = Environment.GetEnvironmentVariable("PYTHONNET_PYDLL") ?? Discover();
        Runtime.PythonDLL = dll;                 // pythonnet needs the shared library, not the interpreter
        PythonEngine.Initialize();               // once per process (CPython + numpy cannot re-initialize)
        PythonEngine.BeginAllowThreads();        // release THIS thread's GIL so every thread can take it later
        Console.WriteLine($"Python {PythonEngine.Version.Split(' ')[0]}  ({dll})");
        return new PythonHost();
    }

    public void Dispose()
    {
        if (_down) return;
        _down = true;
        RuntimeData.FormatterType = typeof(NoopFormatter);   // pythonnet's opt-out of BinaryFormatter stashing (.NET 8+)
        PythonEngine.Shutdown();                             // the interop's handler drains every lease first
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

/// <summary>A Python namespace with numpy imported, plus GIL-wrapped one-liners for reading it back.</summary>
sealed class PyScope : IDisposable
{
    public PyModule Module { get; }

    public PyScope(string setup = "import numpy as np")
    {
        using (Py.GIL()) { Module = Py.CreateScope(); Module.Exec(setup); }
    }

    public void Exec(string code) { using (Py.GIL()) Module.Exec(code); }
    public void Set(string name, PyObject value) { using (Py.GIL()) Module.Set(name, value); }
    public PyObject Get(string name) => Module.Get(name);                        // call under Py.GIL()
    public T Eval<T>(string expr) { using (Py.GIL()) { using PyObject r = Module.Eval(expr); return r.As<T>(); } }
    public string Str(string expr) => Eval<string>($"str({expr})");
    public bool Bool(string expr) => Eval<bool>($"bool({expr})");
    public long Long(string expr) => Eval<long>($"int({expr})");
    public double Float(string expr) => Eval<double>($"float({expr})");
    public void Dispose()
    {
        if (_disposed || !PythonEngine.IsInitialized) return;   // never touch the C-API after Shutdown()
        _disposed = true;
        using (Py.GIL()) Module.Dispose();
    }

    bool _disposed;
}

/// <summary>Evidence, not narration: one OK/FAIL line per claim; the exit code is 0 only when all hold.</summary>
sealed class Checklist
{
    int _failed;

    public void Section(string title) => Console.WriteLine($"\n-- {title} --");

    public bool That(bool ok, string claim)
    {
        Console.WriteLine($"  {(ok ? "OK  " : "FAIL")} {claim}");
        if (!ok) _failed++;
        return ok;
    }

    public void Throws<T>(Action act, string claim, string? containing = null) where T : Exception
    {
        try { act(); That(false, $"{claim} — expected {typeof(T).Name}, nothing was thrown"); }
        catch (T e) { That(containing is null || e.Message.Contains(containing), $"{claim}: {typeof(T).Name}: {FirstLine(e.Message)}"); }
        catch (Exception e) { That(false, $"{claim} — expected {typeof(T).Name}, got {e.GetType().Name}: {FirstLine(e.Message)}"); }
    }

    public int Exit()
    {
        Console.WriteLine(_failed == 0 ? "\nALL OK" : $"\n{_failed} FAILED");
        return _failed == 0 ? 0 : 1;
    }

    static string FirstLine(string s) { int i = s.IndexOfAny(new[] { '\r', '\n' }); return i < 0 ? s : s[..i]; }
}
