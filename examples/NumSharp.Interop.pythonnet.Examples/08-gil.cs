#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 08 — The GIL. Every conversion verb acquires the GIL itself (re-entrantly), so a bare call from
//      a thread that never touched Python works and nesting under your own Py.GIL() is fine. For a
//      hot loop, lift the per-call acquisition: one Py.GIL() outside, requireGIL:false inside. The
//      process-wide default is NDArrayPythonInterop.RequireGIL (true); the per-call parameter
//      overrides it, null follows it. The trap: a .NET body invoked FROM Python does NOT hold the
//      GIL — pythonnet's binder releases it around managed code — so leave management ON there.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example08_Gil.cs.
//
//      Run:  dotnet run 08-gil.cs

using System.Diagnostics;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

// ==== the default: verbs acquire the GIL themselves ==================================================
bool managed = NDArrayPythonInterop.RequireGIL;              // true by default
NDArray nd = np.arange(4).astype(np.float64);
PyObject bare = nd.ToNumpy();                                // no Py.GIL() on this thread — the verb took it (and released it) itself
using (Py.GIL())
{
    py.g1 = bare;
    bare.Dispose();                                          // YOUR PyObject work — including Dispose — still needs the GIL
    py.g2 = nd.ToNumpy();                                    // re-entrant under an outer acquisition
    bool same = py.np.shares_memory(py.g1, py.g2);           // True: both produced the same shared view
}

// ==== the hot loop: one acquisition, N conversions ==================================================
NDArray[] batches = Enumerable.Range(0, 200).Select(i => np.arange(16).astype(np.float64) + i).ToArray();
double total = 0;
using (Py.GIL())                                             // ONE acquisition...
    foreach (NDArray batch in batches)
    {
        dynamic p = batch.ToNumpy(requireGIL: false);        // ...N conversions inside, with no PyGILState_Ensure/Release pair per call
        total += (double)p.sum();
    }
// total == 200 * 120 + 16 * (0 + 1 + ... + 199) == 342400
double total2 = 0;
foreach (NDArray batch in batches)
    using (Py.GIL())                                         // the default shape: every call takes and releases the GIL (measurably slower per call)
    {
        dynamic p = batch.ToNumpy();
        total2 += (double)p.sum();                           // total2 == total
    }

// ==== the process-wide opt-out: RequireGIL = false (null parameters follow it) =====================
NDArrayPythonInterop.RequireGIL = false;                     // from now on EVERY call site in the process must hold the GIL
try
{
    using (Py.GIL())
    {
        py.g3 = nd.ToNumpy();                                // null -> global false -> the no-op guard; np.shares_memory(g3, g1) == True
        using PyObject src = py.Eval("np.array([7, 8, 9], dtype='i4')");
        NDArray copied = src.ToNDArray();                    // null -> global false; copied[2] == 9
        using PyObject forced = nd.ToNumpy(requireGIL: true);   // an explicit true still acquires (re-entrantly)
    }
}
finally { NDArrayPythonInterop.RequireGIL = true; }          // restore — the opt-out is a process-wide contract

// ==== proof that true REALLY acquires: contention ===================================================
using var gilHeld = new ManualResetEventSlim(false);
using var releaseGil = new ManualResetEventSlim(false);
using var converted = new ManualResetEventSlim(false);
var holder = new Thread(() => { using (Py.GIL()) { gilHeld.Set(); releaseGil.Wait(15_000); } }) { IsBackground = true };
var converter = new Thread(() =>
{
    gilHeld.Wait(15_000);
    PyObject arr = nd.ToNumpy(requireGIL: true);             // BLOCKS inside PyGILState_Ensure while the holder owns the GIL
    converted.Set();
    using (Py.GIL()) arr.Dispose();
}) { IsBackground = true };
holder.Start(); converter.Start();
bool finishedEarly = converted.Wait(400);                    // false: the conversion is still waiting for the holder
releaseGil.Set();
converter.Join(15_000); holder.Join(15_000);                 // converted.IsSet == true: it completed once the holder let go

// ==== worker threads with their own arrays ==========================================================
using (Py.GIL()) py.Exec("def weigh(a):\n    return float((a * 0.5).sum())");
var failures = new System.Collections.Concurrent.ConcurrentQueue<string>();
Thread[] workers = Enumerable.Range(0, 3).Select(id => new Thread(() =>
{
    using var workerScope = NDScope.Open();                  // a scope is per thread: the worker opens its own for its own arrays
    for (int k = 0; k < 10; k++)
    {
        NDArray batch = np.arange(6).astype(np.float64) + id;    // sum 15 + 6 * id
        double weighed;
        using (Py.GIL())                                         // each worker takes the GIL around its own Python work
            weighed = py.weigh(batch);                           // the codec encodes the argument; (15 + 6 * id) * 0.5 comes back
        if (weighed != (15.0 + 6 * id) * 0.5) failures.Enqueue($"worker {id}: {weighed}");
    }
}) { IsBackground = true }).ToArray();
foreach (Thread w in workers) w.Start();
foreach (Thread w in workers) w.Join(30_000);
// failures is empty: 3 workers x 10 calls, no deadlock, no wrong answer

// ==== the trap: a Python -> .NET callback body does NOT hold the GIL ================================
int inBody = -1;
string? inside = null;
using (Py.GIL())
{
    int here = PyGILState_Check();                           // 1: inside our own Py.GIL() block
    Action callback = () =>
    {
        inBody = PyGILState_Check();                         // 0: pythonnet's binder RELEASED the GIL around this managed body
        try
        {
            using NDArray tmp = np.arange(4).astype(np.float64);
            PyObject p = tmp.ToNumpy(requireGIL: true);      // so the conversion MUST manage the GIL itself here...
            using (Py.GIL()) p.Dispose();
            inside = "OK";                                   // ...and does; requireGIL: false in this spot would be an access violation
        }
        catch (Exception e) { inside = $"{e.GetType().Name}: {e.Message}"; }
    };
    py.gil_cb = callback;                                    // a .NET delegate becomes a Python callable
    py.Exec("gil_cb()");                                     // Python calls back into .NET
}

// ==== the machinery is immune to the policy =========================================================
NDArrayPythonInterop.RequireGIL = false;
try
{
    using (Py.GIL())
    {
        py.Exec("drainsrc = np.arange(8.0)");
        using PyObject src = py.drainsrc;
        NDArray view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: false, requireGIL: false);   // a view taken GIL-less holds a lease: LiveImports == 1
        view.Dispose();                                      // enqueues the lease release; the drain takes the GIL ITSELF
    }
    PythonHost.Drain();                                      // LiveImports == 0 — released without any caller-held GIL
}
finally { NDArrayPythonInterop.RequireGIL = true; }

// ==== who frees what ================================================================================
// As in the other examples: `scope`, then the namespace (g1, g2, g3 die with their names), then the engine.

// Ground truth from the C-API itself — PyGILState_Check() is safe to call without the GIL.
static unsafe int PyGILState_Check()
{
    IntPtr lib = NativeLibrary.Load(Runtime.PythonDLL!);
    var check = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(lib, "PyGILState_Check");
    return check();
}

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
