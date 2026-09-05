#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 01 — Bootstrap: find CPython, start the engine ONCE per process, verify the pythonnet/Python
//      pairing, share the first buffer, prove any thread can convert, and shut down crash-free.
//
//      Run:  dotnet run 01-bootstrap.cs        (needs `python` on PATH with numpy, or PYTHONNET_PYDLL)

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
var check = new Checklist();

// ---- 1. the pairing: every pythonnet release hard-caps the Python it can drive ------------------
check.Section("engine + version guard");
Version pythonnet = typeof(PythonEngine).Assembly.GetName().Version!;
Version python = Version.Parse(PythonEngine.Version.Split(' ')[0]);
Console.WriteLine($"  pythonnet {pythonnet.ToString(3)} drives Python {PythonEngine.MinSupportedVersion.ToString(2)} - " +
                  $"{PythonEngine.MaxSupportedVersion.ToString(2)}; running Python {python.ToString(2)}");
check.That(PythonEngine.IsInitialized, "PythonEngine.IsInitialized after Initialize()");
check.That(python >= new Version(PythonEngine.MinSupportedVersion.Major, PythonEngine.MinSupportedVersion.Minor) &&
           python <= new Version(PythonEngine.MaxSupportedVersion.Major, PythonEngine.MaxSupportedVersion.Minor),
           "the running Python is inside the loaded pythonnet's supported range (the interop re-checks this on first use)");

// ---- 2. register the codec BEFORE the first As<NDArray>() anywhere in the process ---------------
// pythonnet caches decoder lookups per (Python type, CLR type) pair — and caches MISSES. A decode
// attempted before registration poisons that pair for the whole engine session. Register at startup.
check.Section("codec registration order");
check.That(NDArrayPythonInterop.RegisterCodec(), "RegisterCodec() returns true the first time in a session");
check.That(!NDArrayPythonInterop.RegisterCodec(), "...and false afterwards (idempotent per session)");

// ---- 3. one buffer, both sides --------------------------------------------------------------------
check.Section("first shared array");
using NDArray nd = np.arange(6).reshape(2, 3);       // int64, C-contiguous, NumSharp-owned memory
NDArray copy, view;
using (Py.GIL())                                      // your own PyObject work still needs the GIL
{
    using var scope = Py.CreateScope();
    scope.Exec("import numpy as np");

    using (PyObject x = nd.ToNumpy())                 // zero-copy numpy view of NumSharp's buffer
        scope.Set("x", x);

    scope.Exec("x[1, 2] = 99");                       // Python writes...
    check.That(nd.GetInt64(1, 2) == 99, "a Python write through the view lands in NumSharp (nd[1,2] == 99)");

    scope.Exec("r = np.sin(x / 3.0)");                // numpy computes a fresh float64 array...
    using PyObject r = scope.Get("r");
    copy = r.ToNDArray();                             // ...an independent C-contiguous copy of it
    view = r.AsNDArray();                             // ...or a zero-copy view over numpy's buffer

    check.That(copy.typecode == NPTypeCode.Double && copy.shape.SequenceEqual(new long[] { 2, 3 }), "ToNDArray: float64 (2,3), detached");
    check.That(view.Shape.IsWriteable, "AsNDArray: writeable view over numpy's fresh result");

    view[0, 0] = -1.0;                                // NumSharp writes through the view...
    using PyObject probe = scope.Eval("float(r[0, 0])");
    check.That(probe.As<double>() == -1.0, "...and numpy sees it (r[0,0] == -1.0)");
}

Console.WriteLine(nd.ToString());
copy.Dispose();
view.Dispose();                                       // releases the lease on numpy's buffer

// ---- 4. BeginAllowThreads is what lets threads that never touched Python convert ----------------
check.Section("any thread converts");
Exception? failure = null;
var worker = new Thread(() =>
{
    try
    {
        using NDArray batch = np.arange(4).astype(NPTypeCode.Double);
        using (Py.GIL())
        using (PyObject p = batch.ToNumpy())
        using (PyObject s = p.InvokeMethod("sum"))
            if (s.As<double>() != 6.0) throw new Exception("wrong sum");
    }
    catch (Exception e) { failure = e; }
});
worker.Start();
worker.Join();
check.That(failure is null, $"a fresh thread exported + summed under its own Py.GIL() {(failure is null ? "" : failure.Message)}");

// ---- 5. observability: nothing leaks --------------------------------------------------------------
check.Section("counters");
check.That(NDArrayPythonInterop.LiveExports == 0, $"LiveExports == 0 after every Python view died (was {NDArrayPythonInterop.LiveExports})");
check.That(NDArrayPythonInterop.LiveImports == 0, $"LiveImports == 0 after every NumSharp view was disposed (was {NDArrayPythonInterop.LiveImports})");

return check.Exit();   // `using var host` shuts the engine down after this line

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
