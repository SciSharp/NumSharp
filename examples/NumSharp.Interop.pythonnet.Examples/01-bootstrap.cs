#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 01 — Bootstrap: find CPython, start the engine ONCE per process, verify the pythonnet/Python
//      pairing, register the codec, share the first buffer, prove any thread can convert, and shut
//      down crash-free. Every other example starts the same way through PythonHost.Start() at the
//      bottom of the file — this one walks through what that call does.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example01_Bootstrap.cs.
//
//      Run:  dotnet run 01-bootstrap.cs        (needs `python` on PATH with numpy, or PYTHONNET_PYDLL)

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start(codec: false);   // Runtime.PythonDLL = ...; PythonEngine.Initialize(); PythonEngine.BeginAllowThreads(); a namespace with `import numpy as np`
using var scope = NDScope.Open();                  // every NDArray built on this thread is released when the scope closes — nothing is disposed by hand
dynamic py = host.Namespace;                       // the Python namespace; talk to it under Py.GIL()

// ---- 1. the pairing: every pythonnet release hard-caps the Python it can drive --------------------
Version pythonnet = typeof(PythonEngine).Assembly.GetName().Version!;           // 3.0.5
Version python = Version.Parse(PythonEngine.Version.Split(' ')[0]);             // the running Python, e.g. 3.12.12
Version floor = PythonEngine.MinSupportedVersion;                               // 3.7  — the same for the whole pythonnet 3 line
Version ceiling = PythonEngine.MaxSupportedVersion;                             // 3.13 — what moves from release to release
// PythonEngine.IsInitialized is true, and `python` lies inside [floor, ceiling]. The interop re-checks the
// range on first use and raises an actionable error naming the pythonnet to install when it does not.

// ---- 2. register the codec BEFORE the first As<NDArray>() anywhere in the process ------------------
// pythonnet caches decoder lookups per (Python type, CLR type) pair — and caches MISSES. A decode attempted
// before registration poisons that pair for the whole engine session. Register at startup; PythonHost.Start()
// does it for you in every other example.
bool registered = NDArrayPythonInterop.RegisterCodec();    // true: NDArray <-> numpy and C# tuples <-> Python tuples now convert at every pythonnet boundary
bool again = NDArrayPythonInterop.RegisterCodec();         // false: idempotent for the rest of the engine session

// ---- 3. one buffer, both sides ---------------------------------------------------------------------
NDArray nd = np.arange(6).reshape(2, 3);           // int64, C-contiguous, NumSharp-owned memory
NDArray copy, view;
using (Py.GIL())                                   // Python work runs under the GIL (the conversion verbs take it themselves, re-entrantly)
{
    py.x = nd;                                     // zero-copy: x is a numpy view over nd's bytes (the codec ran ToNumpy)
    py.Exec("x[1, 2] = 99");                       // Python writes...   nd is now [[0 1 2] [3 4 99]]
    py.Exec("r = np.sin(x / 3.0)");                // numpy computes a fresh float64 (2,3) array...
    using PyObject r = py.r;
    copy = r.ToNDArray();                          // ...an independent C-contiguous copy of it (copy.typecode == Double, copy.shape == (2,3))
    view = r.AsNDArray();                          // ...or a zero-copy view over numpy's buffer (view.Shape.IsWriteable == true)
    view[0, 0] = -1.0;                             // NumSharp writes through the view...
    double r00 = py.Eval("r[0, 0]");               // -1.0: numpy sees it
}

// ---- 4. BeginAllowThreads is what lets threads that never touched Python convert ------------------
double sum = 0;
var worker = new Thread(() =>
{
    using NDArray batch = np.arange(4).astype(np.float64);   // built on the worker: outside the main thread's scope, so disposed here
    using (Py.GIL())                                          // a thread that never touched Python takes the GIL like any other
    {
        dynamic p = batch.ToNumpy();
        sum = p.sum();                                        // 6.0
    }
});
worker.Start();
worker.Join();

// ---- 5. who frees what -----------------------------------------------------------------------------
// When this script ends: `scope` releases nd, copy and view in one sweep (the lease on r's buffer goes back to
// numpy); `host` drops the namespace, so x dies and the pin on nd's buffer with it, then shuts the engine down —
// RuntimeData.FormatterType = NoopFormatter; PythonEngine.Shutdown(). NDArrayPythonInterop.LiveExports and
// LiveImports both read 0 afterwards; 06-lifetime.cs watches them move.

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
