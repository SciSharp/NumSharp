#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 02 — The four verbs. Everything else in the package is packaging over these:
//
//        NumSharp -> Python   nd.ToNumpy()        zero-copy numpy VIEW   (shared, source rooted)
//        NumSharp -> Python   nd.ToNumpyCopy()    independent numpy array
//        Python -> NumSharp   py.ToNDArray()      independent C-contiguous COPY
//        Python -> NumSharp   py.AsNDArray()      zero-copy NDArray VIEW (shared, exporter leased)
//
//      Naming follows numpy's array/asarray split: To… copies, As… shares — ToNumpy is the one
//      deliberate exception because the zero-copy view is the package's headline.
//
//      Run:  dotnet run 02-four-verbs.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

// ---- NumSharp -> Python -------------------------------------------------------------------------
check.Section("ToNumpy (view) vs ToNumpyCopy (copy)");
using NDArray nd = np.arange(4).astype(NPTypeCode.Double);            // [0 1 2 3]
using (Py.GIL())
{
    using (PyObject v = nd.ToNumpy()) py.Set("v", v);                 // numpy view over NumSharp's bytes
    using (PyObject c = nd.ToNumpyCopy()) py.Set("c", c);             // numpy owns a separate copy
}
py.Exec("v[0] = 11.0\nc[1] = 22.0");
check.That(nd.GetDouble(0) == 11.0, "view: Python's write reached NumSharp (nd[0] == 11)");
check.That(nd.GetDouble(1) == 1.0, "copy: Python's write did not (nd[1] still 1)");
check.That(!py.Bool("np.shares_memory(v, c)"), "np.shares_memory(v, c) is False");
check.That(py.Str("v.flags['OWNDATA']") == "False", "numpy agrees the view's memory is NumSharp's (OWNDATA=False)");
check.That(py.Str("c.flags['OWNDATA']") == "True", "...and that the copy is its own (OWNDATA=True)");

nd[2] = -5.0;                                                          // NumSharp writes...
check.That(py.Float("v[2]") == -5.0, "...the view shows it immediately");
check.That(py.Float("c[2]") == 2.0, "...the copy does not");

check.Section("the aliases");
using (Py.GIL())
{
    using (PyObject p = nd.ToPython()) py.Set("p", p);                // ToPython() == ToNumpy() (view)
    using (PyObject r = nd.ToNumpy(copy: true)) py.Set("r", r);       // copy:true routes to ToNumpyCopy
}
check.That(py.Bool("np.shares_memory(p, v)"), "nd.ToPython() is the zero-copy view");
check.That(!py.Bool("np.shares_memory(r, v)"), "nd.ToNumpy(copy: true) routes to the copy");

// ---- Python -> NumSharp -------------------------------------------------------------------------
check.Section("ToNDArray (copy) vs AsNDArray (view)");
py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5");
NDArray copied, leased;
using (Py.GIL())
{
    using PyObject src = py.Get("src");
    copied = src.ToNDArray();                                          // fresh, owning, C-contiguous
    leased = src.AsNDArray();                                          // == NDArrayPythonInterop.ToNDArrayView(src)
}
check.That(copied.Shape.IsContiguous && copied.typecode == NPTypeCode.Double, "ToNDArray: C-contiguous float64 copy");
copied[0, 0] = -9.0;
check.That(py.Float("src[0, 0]") == 0.0, "copy: NumSharp's write does not reach numpy");
leased[0, 0] = -9.0;
check.That(py.Float("src[0, 0]") == -9.0, "view: NumSharp's write reaches numpy");
py.Exec("src[1, 2] = 42.0");
check.That(leased.GetDouble(1, 2) == 42.0, "view: numpy's write reaches NumSharp");
check.That(NDArrayPythonInterop.LiveImports == 1, "one live import lease while the view is alive");

check.Section("non-contiguous sources: copies linearize, views keep strides");
NDArray strided;
using (Py.GIL())
{
    using PyObject t = py.Module.Eval("src.T");                       // (3,2) Fortran-strided view of src
    using NDArray tCopy = t.ToNDArray();
    strided = t.AsNDArray();
    check.That(tCopy.Shape.IsContiguous && tCopy.GetDouble(2, 1) == 42.0, "ToNDArray(src.T): C-contiguous, logical order preserved");
}
check.That(strided.shape.SequenceEqual(new long[] { 3, 2 }) && strided.Shape.Strides.SequenceEqual(new long[] { 1, 3 }),
           $"AsNDArray(src.T): shape {string.Join("x", strided.shape)}, element strides ({string.Join(",", strided.Shape.Strides)})");
check.That(strided.GetDouble(2, 1) == 42.0, "the strided view reads the right element");

check.Section("read-only sources");
using (Py.GIL())
{
    using PyObject ro = py.Module.Eval("b'\\x01\\x02\\x03\\x04'");     // bytes: immutable
    check.Throws<InvalidOperationException>(() => ro.AsNDArray().Dispose(),
        "AsNDArray refuses a read-only exporter by default", "the exporter's buffer is read-only");
    using NDArray view = ro.AsNDArray(allowReadonly: true);
    check.That(!view.Shape.IsWriteable && view.GetByte(1) == 2, "allowReadonly:true yields a NON-writeable view (Shape.IsWriteable == false)");
    check.Throws<Exception>(() => view[0] = (byte)7, "a guarded write through it throws instead of corrupting bytes", "read-only");
    using NDArray owned = ro.ToNDArray();
    check.That(owned.Shape.IsWriteable, "ToNDArray of the same bytes is a writable owning copy");
}

check.Section("an import view owns nothing (numpy's owndata == False semantics)");
check.Throws<Exception>(() => leased.resize(new Shape(12)), "resize refuses on a leased view", "does not own its data");
using (NDArray owning = np.require(leased, (Type?)null, "O"))         // the escape hatch: an owning copy
{
    owning[0, 0] = 123.0;
    check.That(py.Float("src[0, 0]") == -9.0, "np.require(view, null, \"O\") produced a detached owning copy");
}

check.Section("array-likes (list / tuple / scalar) — FromArrayLike goes through numpy.asarray");
using (Py.GIL())
{
    using PyObject list = py.Module.Eval("[[1, 2, 3], [4, 5, 6]]");
    using NDArray fromList = list.FromArrayLike();
    check.That(fromList.typecode == NPTypeCode.Int64 && fromList.GetInt64(1, 2) == 6, "a nested Python list becomes an owning int64 (2,3)");
    using PyObject scalar = py.Module.Eval("2.5");
    using NDArray fromScalar = scalar.FromArrayLike();
    check.That(fromScalar.ndim == 0 && fromScalar.GetDouble() == 2.5, "a Python float becomes a 0-d float64");
    check.Throws<NotSupportedException>(() => list.AsNDArray().Dispose(), "a list is not a PEP 3118 exporter, so AsNDArray refuses it", "does not export a PEP 3118 buffer");
}

// ---- release -------------------------------------------------------------------------------------
copied.Dispose();
leased.Dispose();
strided.Dispose();
check.Section("counters after disposal");
check.That(Settle(() => NDArrayPythonInterop.LiveImports == 0), "LiveImports drained to 0 (leases release on the last NumSharp view)");
py.Exec("del v, c, p, r");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0), "LiveExports drained to 0 (pins release when the last numpy view dies)");

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
