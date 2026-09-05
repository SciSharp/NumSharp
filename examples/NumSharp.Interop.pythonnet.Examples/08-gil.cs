#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
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
//      Run:  dotnet run 08-gil.cs

using System.Diagnostics;
using System.Runtime.InteropServices;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

// ==== the default: verbs acquire the GIL themselves ==================================================
check.Section("verbs acquire the GIL themselves");
check.That(NDArrayPythonInterop.RequireGIL, "NDArrayPythonInterop.RequireGIL defaults to true");
using NDArray nd = np.arange(4).astype(NPTypeCode.Double);
PyObject bare = nd.ToNumpy();                              // no Py.GIL() on this thread — the verb took it
using (Py.GIL())
{
    py.Module.Set("g1", bare);
    bare.Dispose();                                        // YOUR PyObject work (incl. Dispose) still needs the GIL
    using PyObject nested = nd.ToNumpy();                  // re-entrant under an outer acquisition
    py.Module.Set("g2", nested);
}
check.That(py.Bool("np.shares_memory(g1, g2)"), "a bare call and a nested call both produced the same shared view");

// ==== the hot loop: one acquisition, N conversions ==================================================
check.Section("hot loop: requireGIL:false under ONE outer Py.GIL()");
var batches = Enumerable.Range(0, 200).Select(i => np.arange(16).astype(NPTypeCode.Double) + i).ToArray();
double total = 0;
var sw = Stopwatch.StartNew();
using (Py.GIL())                                                        // ONE acquisition...
    foreach (NDArray batch in batches)
        using (PyObject p = batch.ToNumpy(requireGIL: false))            // ...N conversions inside, no PyGILState_Ensure/Release per call
        using (PyObject s = p.InvokeMethod("sum"))
            total += s.As<double>();
double perCallNoGil = sw.Elapsed.TotalMilliseconds * 1000 / batches.Length;
sw.Restart();
double total2 = 0;
foreach (NDArray batch in batches)
    using (Py.GIL())                                                    // the default shape: every call takes and releases the GIL
    using (PyObject p = batch.ToNumpy())
    using (PyObject s = p.InvokeMethod("sum"))
        total2 += s.As<double>();
double perCallDefault = sw.Elapsed.TotalMilliseconds * 1000 / batches.Length;
check.That(total == total2 && total == batches.Length * 120.0 + 16.0 * (199 * 200 / 2), $"both loops computed the same 200 sums ({perCallNoGil:F1} us/call lifted vs {perCallDefault:F1} us/call default)");
foreach (NDArray b in batches) b.Dispose();

check.Section("the process-wide opt-out: RequireGIL = false (null parameters follow it)");
NDArrayPythonInterop.RequireGIL = false;                                 // every call site in the process MUST hold the GIL from now on
try
{
    using (Py.GIL())
    {
        using (PyObject arr = nd.ToNumpy()) py.Module.Set("g3", arr);   // null -> global false -> no-op guard
        using PyObject src = py.Module.Eval("np.array([7, 8, 9], dtype='i4')");
        using NDArray copied = src.ToNDArray();                          // null -> global false
        check.That(copied.GetInt32(2) == 9 && py.Bool("np.shares_memory(g3, g1)"), "under the opt-out every verb runs on the caller's GIL");
        using PyObject forced = nd.ToNumpy(requireGIL: true);            // an explicit true still acquires (re-entrantly)
        check.That(forced is not null, "an explicit requireGIL:true overrides the global false");
    }
}
finally { NDArrayPythonInterop.RequireGIL = true; }                      // restore — the opt-out is a process-wide contract

// ==== proof that true REALLY acquires: contention ===================================================
check.Section("requireGIL:true blocks while another thread owns the GIL");
using var gilHeld = new ManualResetEventSlim(false);
using var releaseGil = new ManualResetEventSlim(false);
using var converted = new ManualResetEventSlim(false);
var holder = new Thread(() => { using (Py.GIL()) { gilHeld.Set(); releaseGil.Wait(15_000); } }) { IsBackground = true };
var converter = new Thread(() =>
{
    gilHeld.Wait(15_000);
    PyObject arr = nd.ToNumpy(requireGIL: true);                        // must BLOCK inside PyGILState_Ensure
    converted.Set();
    using (Py.GIL()) arr.Dispose();
}) { IsBackground = true };
holder.Start(); converter.Start();
bool finishedEarly = converted.Wait(400);
releaseGil.Set();
converter.Join(15_000); holder.Join(15_000);
check.That(!finishedEarly && converted.IsSet, "the conversion waited for the holder to release the GIL, then completed");

// ==== worker threads with their own arrays ==========================================================
check.Section("worker threads: each takes the GIL around its own Python work");
NDArrayPythonInterop.RegisterCodec();
py.Exec("def weigh(a):\n    return float((a * 0.5).sum())");
var failures = new System.Collections.Concurrent.ConcurrentQueue<string>();
var workers = Enumerable.Range(0, 3).Select(id => new Thread(() =>
{
    try
    {
        for (int k = 0; k < 10; k++)
        {
            using NDArray batch = np.arange(6).astype(NPTypeCode.Double) + id;   // sum 15 + 6*id
            double weighed;
            using (Py.GIL())
            {
                dynamic weigh = py.Get("weigh");
                weighed = (double)weigh(batch);                                   // the codec encodes the argument
            }
            if (weighed != (15.0 + 6 * id) * 0.5) failures.Enqueue($"worker {id}: {weighed}");
        }
    }
    catch (Exception e) { failures.Enqueue($"worker {id}: {e.GetType().Name}: {e.Message}"); }
}) { IsBackground = true }).ToArray();
foreach (var w in workers) w.Start();
foreach (var w in workers) w.Join(30_000);
check.That(failures.IsEmpty, $"3 workers x 10 calls, no deadlock, no wrong answer {string.Join(" | ", failures)}");

// ==== the trap: a Python -> .NET callback body does NOT hold the GIL ================================
check.Section("Python -> .NET callback bodies do not hold the GIL");
// Ground truth from the C-API itself: PyGILState_Check() is safe to call without the GIL.
IntPtr lib = NativeLibrary.Load(Runtime.PythonDLL!);
var gilCheck = Marshal.GetDelegateForFunctionPointer<PyGILStateCheck>(NativeLibrary.GetExport(lib, "PyGILState_Check"));
int inBody = -1;
string? inside = null;
using (Py.GIL())
{
    check.That(gilCheck() == 1, "PyGILState_Check() == 1 inside our own Py.GIL() block");
    Action callback = () =>
    {
        inBody = gilCheck();                                                     // pythonnet released the GIL around this body
        try
        {
            using NDArray tmp = np.arange(4).astype(NPTypeCode.Double);
            PyObject p = tmp.ToNumpy(requireGIL: true);                          // MUST manage the GIL itself here...
            using (Py.GIL()) p.Dispose();
            inside = "OK";
        }
        catch (Exception e) { inside = $"{e.GetType().Name}: {e.Message}"; }
    };
    py.Module.Set("gil_cb", callback);
    py.Module.Exec("gil_cb()");                                                  // Python calls back into .NET
}
check.That(inBody == 0, "PyGILState_Check() == 0 inside a .NET Action invoked from Python — the binder released it");
check.That(inside == "OK", $"requireGIL:true inside the callback re-acquires and succeeds ({inside}); requireGIL:false there would be an access violation");
NativeLibrary.Free(lib);

// ==== the machinery is immune to the policy =========================================================
check.Section("background lease disposal manages the GIL itself, even under the opt-out");
NDArrayPythonInterop.RequireGIL = false;
try
{
    using (Py.GIL())
    {
        py.Module.Exec("drainsrc = np.arange(8.0)");
        using PyObject src = py.Module.Eval("drainsrc");
        NDArray view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: false, requireGIL: false);
        check.That(NDArrayPythonInterop.LiveImports == 1, "a view taken GIL-less holds a lease");
        view.Dispose();                                                          // enqueues the lease; the drain thread takes the GIL ITSELF
    }
    check.That(Settle(() => NDArrayPythonInterop.LiveImports == 0), "the deferred drain released it without any caller-held GIL");
}
finally { NDArrayPythonInterop.RequireGIL = true; }

check.Section("counters");
py.Exec("del g1, g2, g3");
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

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate int PyGILStateCheck();

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
