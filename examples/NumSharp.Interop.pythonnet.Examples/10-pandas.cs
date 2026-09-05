#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 10 — Pandas. The built-in PandasPythonArrayAdapter recognizes DataFrame, Series, Index,
//      ExtensionArray and their subclasses and hands their official to_numpy() projection to the
//      SAME memory bridge. Pandas documents that to_numpy(copy=False) does not guarantee a view, so
//      the View route proves two independent projections overlap before sharing; Copy/Auto take an
//      owning snapshot otherwise. Pandas 3's Copy-on-Write exposes shared projections READ-ONLY, so
//      view callers pass allowReadonly:true. NumSharp -> Pandas is the ordinary numpy encoder plus
//      an explicit copy=False.
//
//      Run:  dotnet run 10-pandas.cs        (self-skips when pandas is not installed)

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

try { py.Exec("import pandas as pd"); }
catch (PythonException) { Console.WriteLine("  (skip) python package 'pandas' is not installed — see requirements.txt"); return 0; }
Console.WriteLine($"pandas {py.Str("pd.__version__")}");

// ==== Pandas -> NumSharp ============================================================================
check.Section("Series: the ordinary verbs, through the adapter");
py.Exec("s = pd.Series(np.arange(6, dtype='i8'))");
using (Py.GIL())
{
    using PyObject s = py.Get("s");
    check.Throws<InvalidOperationException>(() => s.AsNDArray(requireGIL: false).Dispose(), "bare AsNDArray() refuses: Pandas 3 exposes the shared projection read-only", "allowReadonly:true");
    using NDArray view = s.AsNDArray(allowReadonly: true, requireGIL: false);
    check.That(view.typecode == NPTypeCode.Int64 && view.shape.SequenceEqual(new long[] { 6 }) && !view.Shape.IsWriteable, "AsNDArray(allowReadonly: true): a NON-writeable int64 (6,) view");
    check.That(py.Bool("s.to_numpy().ctypes.data == s.to_numpy().ctypes.data") && view.GetInt64(4) == 4, "...over Pandas' own stable storage (two to_numpy() projections overlap)");
    py.Exec("s.iloc[2] = 77");
    check.That(view.GetInt64(2) == 77, "a Pandas write is visible through the view");
    using NDArray zeros = np.zeros(new Shape(6), NPTypeCode.Int64);
    check.Throws<Exception>(() => np.copyto(view, zeros), "a NumSharp write through the read-only view is refused", "read-only");
    using NDArray copy = s.ToNDArray(requireGIL: false);
    copy[0] = 999L;
    check.That(copy.Shape.IsWriteable && py.Long("s.iloc[0]") == 0, "ToNDArray(): an owning, writable, detached snapshot");
}

check.Section("DataFrame: a homogeneous frame is ONE F-strided block");
NDArrayPythonInterop.RegisterCodec();
py.Exec("df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]}, dtype='i8')");
using (Py.GIL())
{
    using PyObject df = py.Get("df");
    using NDArray view = df.As<NDArray>();                                   // the codec: Auto = verified view first
    check.That(view.shape.SequenceEqual(new long[] { 3, 2 }) && view.Shape.Strides.SequenceEqual(new long[] { 1, 3 }) && view.Shape.IsFContiguous,
               $"As<NDArray>(): shape ({string.Join(",", view.shape)}), element strides ({string.Join(",", view.Shape.Strides)}) — Fortran order, as Pandas stores blocks");
    check.That(!view.Shape.IsWriteable && view.GetInt64(2, 1) == 30, "read-only (Copy-on-Write), values in place: [2,1] == 30");
    py.Exec("df.iloc[1, 1] = 77");
    check.That(view.GetInt64(1, 1) == 77, "a Pandas write is visible through the view");
}

check.Section("Index, RangeIndex and nullable extension arrays");
py.Exec("idx = pd.Index([4, 5, 6], dtype='i8')\nrng = pd.RangeIndex(5)\nnullable = pd.array([7, 8, 9], dtype='Int64')");
using (Py.GIL())
{
    using PyObject idx = py.Get("idx");
    using NDArray iv = idx.AsNDArray(allowReadonly: true, requireGIL: false);
    check.That(!iv.Shape.IsWriteable && iv.GetInt64(2) == 6, "pd.Index: read-only view (an Index is immutable)");
    using PyObject rng = py.Get("rng");
    using NDArray rv = rng.ToNDArray(requireGIL: false);
    check.That(rv.typecode == NPTypeCode.Int64 && rv.GetInt64(4) == 4, "pd.RangeIndex: recognized through the Index MRO; materializes to int64 [0..4]");
    using PyObject nullable = py.Get("nullable");
    using NDArray nv = nullable.AsNDArray(allowReadonly: true, requireGIL: false);
    check.That(nv.typecode == NPTypeCode.Int64 && nv.Shape.IsWriteable, "pd.array(dtype='Int64') without missing values exposes its stable typed buffer — writable");
    nv[1] = 71L;
    check.That(py.Long("nullable[1]") == 71, "...and a NumSharp write reaches the extension array");
}

check.Section("what materializes: mixed blocks, missing values, categoricals, text");
py.Exec("mixed = pd.DataFrame({'small': np.array([1, 2], dtype='i4'), 'real': np.array([3.5, 4.5], dtype='f8')})\n" +
        "with_na = pd.array([1, None, 3], dtype='Int64')\n" +
        "cat = pd.Series(pd.Categorical([3, 1, 3]))\n" +
        "text = pd.Series(['a', 'b'])");
using (Py.GIL())
{
    using PyObject mixed = py.Get("mixed");
    check.Throws<NotSupportedException>(() => mixed.AsNDArray(allowReadonly: true, requireGIL: false).Dispose(), "mixed dtypes: View declines (Pandas materialized a fresh array)", "cannot be proven");
    using NDArray upcast = mixed.ToNDArray(requireGIL: false);
    check.That(upcast.typecode == NPTypeCode.Double && upcast.GetDouble(1, 0) == 2.0 && upcast.GetDouble(0, 1) == 3.5, "ToNDArray(): Pandas' common-dtype upcast (int32 + float64 -> float64), detached");
    using NDArray auto = mixed.As<NDArray>();
    check.That(auto.Shape.IsWriteable && auto.typecode == NPTypeCode.Double, "the codec's Auto falls back to the owning copy");

    using PyObject withNa = py.Get("with_na");
    check.Throws<NotSupportedException>(() => withNa.AsNDArray(allowReadonly: true, requireGIL: false).Dispose(), "a nullable array WITH pd.NA: no stable buffer to share");
    using NDArray naCopy = withNa.ToNDArray(requireGIL: false);
    check.That(naCopy.typecode == NPTypeCode.Double && double.IsNaN(naCopy.GetDouble(1)), "...copies as float64 with NaN for the missing value");

    using PyObject cat = py.Get("cat");
    check.Throws<NotSupportedException>(() => cat.AsNDArray(allowReadonly: true, requireGIL: false).Dispose(), "categorical: View declines");
    using NDArray catCopy = cat.ToNDArray(requireGIL: false);
    check.That(catCopy.GetInt64(0) == 3 && catCopy.GetInt64(1) == 1, "...ToNDArray copies the category VALUES [3, 1, 3]");

    using PyObject text = py.Get("text");
    check.Throws<NotSupportedException>(() => text.ToNDArray(requireGIL: false).Dispose(), "object/string columns have no NumSharp dtype (both paths refuse)");
}

check.Section("labels are metadata: only the value matrix crosses");
py.Exec("labelled = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]), columns=['dup', 'dup'])");
using (Py.GIL())
{
    using PyObject labelled = py.Get("labelled");
    using NDArray values = labelled.As<NDArray>();
    check.That(values.shape.SequenceEqual(new long[] { 2, 2 }) && values.GetDouble(1, 0) == 3.0, "a MultiIndex + duplicate column names: still the (2,2) value block, row-major logical order");
}

// ==== NumSharp -> Pandas ============================================================================
check.Section("NumSharp -> Pandas: the numpy encoder + an explicit copy=False");
using NDArray nd = np.arange(6).astype(NPTypeCode.Double);
using (Py.GIL()) py.Module.Set("values", nd);                                       // the registered codec encodes a zero-copy numpy view
py.Exec("series = pd.Series(values, copy=False)\nframe = pd.DataFrame(values.reshape(-1, 2), copy=False)");
check.That(py.Bool("series.to_numpy().ctypes.data == values.ctypes.data"), "pd.Series(values, copy=False) sits on NumSharp's memory");
check.That(!py.Bool("series.to_numpy().flags.writeable"), "...Pandas 3 marks the shared projection read-only (its Copy-on-Write rule)");
nd[2] = 81.5;
check.That(py.Float("series.iloc[2]") == 81.5 && py.Float("frame.iloc[1, 0]") == 81.5, "a NumSharp write is visible in the Series AND the DataFrame");
py.Exec("series.iloc[2] = -7.25");
check.That(nd.GetDouble(2) == -7.25, "a Pandas write reaches NumSharp (copy=False is a deliberate shared-memory opt-in; CoW cannot detach an external owner)");
py.Exec("default_copy = pd.Series(values)");
check.That(!py.Bool("default_copy.to_numpy().ctypes.data == values.ctypes.data"), "without copy=False Pandas 3 copies numpy input by default");

check.Section("lifetime: a derived NumSharp view keeps Pandas storage alive after every Python name dies");
py.Exec("tmp = pd.Series(np.arange(8, dtype='f8') * 1.5)");
NDArray slice;
using (Py.GIL())
{
    using PyObject tmp = py.Get("tmp");
    using NDArray whole = tmp.AsNDArray(allowReadonly: true, requireGIL: false);
    slice = whole["2:5"];
}
py.Exec("del tmp\nimport gc; gc.collect()");
check.That(slice.GetDouble(0) == 3.0 && slice.GetDouble(2) == 6.0 && NDArrayPythonInterop.LiveImports == 1, "the slice still reads valid memory; the lease holds the projection");
slice.Dispose();

check.Section("registry: the adapter is built in");
check.That(!PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance) && PandasPythonArrayAdapter.Instance.Name == "pandas",
           "Register(PandasPythonArrayAdapter.Instance) returns false — already registered under the name 'pandas'");
var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
using (Py.GIL())
{
    using PyObject s = py.Get("s");
    using PyObject t = s.GetPythonType();
    using var seriesType = new PyType(t);
    check.That(!noAdapters.CanDecode(seriesType, typeof(NDArray)), "a codec with DecodeArrayAdapters = false does not decode a Series");
}

check.Section("counters");
py.Exec("del series, frame, default_copy, values");
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
