#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 06 — Lifetime: two collectors, one buffer. The rule in both directions is the same — memory is
//      released when the LAST reference on the far side lets go, never when the near side tidies up
//      first. Exports take their own reference on the NumSharp buffer and hand the release to a
//      weakref.finalize on the numpy array's base object; imports lease the Python buffer through
//      NumSharp's memory-block refcount. Two counters make every live crossing observable, a live
//      view locks reallocation on both sides, and engine shutdown drains everything crash-free.
//
//      Run:  dotnet run 06-lifetime.cs

using System.Diagnostics;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

var host = PythonHost.Start();                       // disposed EXPLICITLY below — the last section observes the shutdown sweep
using var py = new PyScope("import numpy as np, weakref, gc");
var check = new Checklist();

// ==== the counters ====================================================================================
check.Section("LiveExports / LiveImports move by exactly one per crossing");
check.That(NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports == 0, "both start at 0");
using NDArray nd = np.arange(4).astype(NPTypeCode.Double);
using (Py.GIL()) { using PyObject p = nd.ToNumpy(); py.Set("c1", p); }
check.That(NDArrayPythonInterop.LiveExports == 1, "one export: LiveExports == 1");
py.Exec("c2 = np.arange(3, dtype='i8')");
NDArray leased;
using (Py.GIL()) { using PyObject c2 = py.Get("c2"); leased = c2.AsNDArray(); }
check.That(NDArrayPythonInterop.LiveImports == 1, "one import view: LiveImports == 1");
leased.Dispose();                                    // deterministic: Dispose releases the lease without waiting for a GC
check.That(Settle(() => NDArrayPythonInterop.LiveImports == 0), "Dispose() released the lease (LiveImports == 0)");
py.Exec("del c1");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0), "del of the last numpy name released the pin (LiveExports == 0)");

// ==== Python outlives the NDArray ===================================================================
check.Section("an export outlives every managed reference");
ExportAndForget();                                   // the NDArray AND the PyObject wrapper die inside; only Python's name remains
FullGc();
check.That(NDArrayPythonInterop.LiveExports == 1, "after Dispose() + GC the pin is still held for Python");
check.That(py.Str("orphan.tolist()") == "[0.0, 1.0, 2.0, 3.0]", $"Python still reads valid memory: {py.Str("orphan.tolist()")}");
py.Exec("orphan[0] = 7.0");
check.That(py.Str("orphan.tolist()") == "[7.0, 1.0, 2.0, 3.0]", "...and still writes it");
py.Exec("child = orphan[2:]\ndel orphan");           // a DERIVED numpy view chains to the same base object
FullGc();
check.That(NDArrayPythonInterop.LiveExports == 1 && py.Str("child.tolist()") == "[2.0, 3.0]", "a derived view (orphan[2:]) keeps the buffer rooted after the original array died");
py.Exec("del child");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0), "the last derived view releases the pin");

// ==== the NDArray outlives Python's object ==========================================================
check.Section("an import lease outlives every Python reference");
py.Exec("keep = np.arange(6, dtype='f8') * 1.5\nwr = weakref.ref(keep)");
NDArray derived = HoldWhilePythonForgets();          // returns only a SLICE of the imported view; original NDArray + Python name both dropped
FullGc();
check.That(NDArrayPythonInterop.LiveImports == 1, "the derived slice alone keeps the lease (disposal order is irrelevant; the refcount decides)");
check.That(py.Bool("wr() is not None"), "the numpy exporter is still alive — Python forgot it, NumSharp holds it");
check.That(derived.GetDouble(0) == 3.0 && derived.GetDouble(2) == 6.0, $"the slice reads valid memory: [{derived.GetDouble(0)}, {derived.GetDouble(1)}, {derived.GetDouble(2)}]");
derived.Dispose();
check.That(Settle(() => NDArrayPythonInterop.LiveImports == 0) && Settle(() => py.Bool("wr() is None")), "disposing the last view releases the lease AND lets Python free the exporter (no leak)");

// ==== what a live view forbids ========================================================================
check.Section("a live view locks reallocation on both sides");
py.Exec("ba = bytearray(b'abcd')");
NDArray held;
using (Py.GIL()) { using PyObject ba = py.Get("ba"); held = ba.AsNDArray(); }
check.Throws<PythonException>(() => py.Exec("ba.append(1)"), "CPython: BufferError while NumSharp leases the bytearray", "Existing exports of data: object cannot be re-sized");
py.Exec("npa = np.arange(4, dtype='f8')");
NDArray held2;
using (Py.GIL()) { using PyObject npa = py.Get("npa"); held2 = npa.AsNDArray(); }
check.Throws<PythonException>(() => py.Exec("npa.resize((8,), refcheck=True)"), "numpy: resize(refcheck=True) refuses while referenced", "cannot resize an array that references or is referenced");
held.Dispose(); held2.Dispose();
Settle(() => NDArrayPythonInterop.LiveImports == 0);
py.Exec("ba.append(1)\nnpa.resize((8,), refcheck=True)");
check.That(py.Long("len(ba)") == 5 && py.Long("npa.size") == 8, "both locks lift once the leases are released");

using NDArray pinned = np.arange(8).astype(NPTypeCode.Double);
using (Py.GIL()) { using PyObject p = pinned.ToNumpy(); py.Set("pin", p); }
check.Throws<Exception>(() => pinned.resize(new Shape(16)), "mirror image: NumSharp's resize refuses while Python holds a view", "cannot resize an array that references or is referenced");
py.Exec("del pin");
Settle(() => NDArrayPythonInterop.LiveExports == 0);
pinned.resize(new Shape(16));
check.That(pinned.size == 16, "...and succeeds once the pin is gone");

// ==== churn: everything returns to baseline ===========================================================
check.Section("120 rounds of export/import/slice churn drain to zero");
for (int i = 0; i < 120; i++) ChurnOnce(i);
py.Exec("del h");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports == 0, 15_000),
           $"exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports}");

// ==== engine shutdown ================================================================================
check.Section("engine shutdown: imports drained, orphaned exports swept");
// Leave one export held ONLY by Python and one import still referenced by C# when the engine dies.
NDArray orphanSource = np.arange(6).astype(NPTypeCode.Double);
using (Py.GIL()) { using PyObject p = orphanSource.ToNumpy(); py.Set("shutdown_orphan", p); }
orphanSource.Dispose();                              // Python is now the only holder of that buffer
py.Exec("shutdown_import = np.arange(8) * 2.0");
NDArray survivor;
using (Py.GIL()) { using PyObject s = py.Get("shutdown_import"); survivor = s.AsNDArray(); }
check.That(NDArrayPythonInterop.LiveExports == 1 && NDArrayPythonInterop.LiveImports == 1, "one orphan export + one live import going into Shutdown()");

host.Dispose();                                      // RuntimeData.FormatterType = NoopFormatter; PythonEngine.Shutdown() — the scope (and the orphan's only holder) dies WITH the interpreter
check.That(!PythonEngine.IsInitialized, "PythonEngine.Shutdown() completed without a crash");
check.That(NDArrayPythonInterop.LiveImports == 0, "the shutdown handler force-drained every import lease (LiveImports == 0)");
survivor.Dispose();                                  // must be a harmless no-op now — the lease was already claimed
check.That(true, "disposing the still-referenced import view after shutdown is safe");
var sw = Stopwatch.StartNew();
while (NDArrayPythonInterop.LiveExports != 0 && sw.ElapsedMilliseconds < 10_000) Thread.Sleep(25);
check.That(NDArrayPythonInterop.LiveExports == 0, "orphaned exports were swept right after the engine died (pythonnet runs no atexit pass, so weakref.finalize could never fire)");

return check.Exit();

[MethodImpl(MethodImplOptions.NoInlining)]           // debug JITs keep a frame's temporaries alive until it returns — so the dying references live in their OWN frame
void ExportAndForget()
{
    var temp = np.arange(4).astype(NPTypeCode.Double);
    using (Py.GIL()) { using PyObject p = temp.ToNumpy(); py.Set("orphan", p); }
    temp.Dispose();
}

[MethodImpl(MethodImplOptions.NoInlining)]
NDArray HoldWhilePythonForgets()
{
    NDArray original;
    using (Py.GIL()) { using PyObject k = py.Get("keep"); original = k.AsNDArray(); }
    NDArray slice = original["2:5"];                 // shares the memory block, so it extends the lease
    py.Exec("del keep");
    return slice;                                    // `original` dies with this frame
}

[MethodImpl(MethodImplOptions.NoInlining)]
void ChurnOnce(int i)
{
    var a = np.arange(32).astype(NPTypeCode.Double);
    using (Py.GIL()) { using PyObject p = a.ToNumpy(); }             // export; wrapper dropped immediately
    py.Exec("h = np.arange(16, dtype='i8')");
    NDArray view;
    using (Py.GIL()) { using PyObject h = py.Get("h"); view = h.AsNDArray(); }
    NDArray sub = view["4:12"];
    sub[0] = (long)i;
    using (Py.GIL()) { using PyObject c = a.ToNumpyCopy(); }
    if (i % 40 == 39) FullGc();
}

static void FullGc()
{
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    using (Py.GIL())
    {
        Finalizer.Instance.Collect();                                     // pythonnet's deferred decrefs
        PythonEngine.RunSimpleString("import gc; gc.collect()");
        using var t = np.arange(1).ToNumpyCopy();                         // any conversion runs the interop's inline drain
    }
}

static bool Settle(Func<bool> done, int timeoutMs = 5000)
{
    var sw = Stopwatch.StartNew();
    while (!done() && sw.ElapsedMilliseconds < timeoutMs) { FullGc(); Thread.Sleep(20); }
    return done();
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
