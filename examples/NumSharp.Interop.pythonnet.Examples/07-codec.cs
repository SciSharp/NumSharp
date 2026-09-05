#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 07 — The codec: conversions without calls. RegisterCodec() hooks NumSharp into pythonnet's
//      conversion pipeline, after which every boundary converts by itself — scope.Set("x", nd),
//      passing an NDArray to a Python callable, `dynamic` numpy calls with NDArray arguments, and
//      pyObj.As<NDArray>() / AsManagedObject(typeof(NDArray)) on the way back. NumpyCodecOptions
//      picks the policy per direction (Auto / View / Copy) and which Python types decode.
//
//      Run:  dotnet run 07-codec.cs

using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope("import numpy as np, ctypes");
var check = new Checklist();

// ==== the one ordering trap: register BEFORE the first As<NDArray>() ===============================
check.Section("register at startup — pythonnet caches decoder MISSES per (Python type, CLR type)");
using (Py.GIL())
{
    using PyObject probe = py.Module.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
    check.Throws<InvalidCastException>(() => probe.As<NDArray>().Dispose(), "before RegisterCodec: As<NDArray>() on c_short_Array_7 fails");
}
check.That(NDArrayPythonInterop.RegisterCodec(), "RegisterCodec() -> true");
using (Py.GIL())
{
    using PyObject again = py.Module.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
    check.Throws<InvalidCastException>(() => again.As<NDArray>().Dispose(), "after RegisterCodec: the SAME Python type still fails — the miss was cached for the session");
    using PyObject fresh = py.Module.Eval("(ctypes.c_short * 8)(1, 2, 3, 4, 5, 6, 7, 8)");
    using NDArray ok = fresh.As<NDArray>();
    check.That(ok.typecode == NPTypeCode.Int16 && ok.GetValue(7) is short and 8, "a pair first touched AFTER registration decodes fine (c_short_Array_8 is a different type)");
}
check.That(!NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy }),
           "a second RegisterCodec (even with other options) is a no-op for the session — the first registration's policy sticks");

// ==== encode: every pythonnet boundary ==============================================================
check.Section("encode: NDArray crosses as a zero-copy numpy view at every boundary");
using NDArray nd = np.arange(4).astype(NPTypeCode.Double);
using (Py.GIL()) py.Module.Set("c", nd);                                    // scope.Set with an NDArray — no ToNumpy() call anywhere
check.That(py.Str("type(c).__name__") == "ndarray" && py.Str("c.flags['OWNDATA']") == "False", "scope.Set(\"c\", nd): a real ndarray that borrows NumSharp's memory");
py.Exec("c[1] = 88.5");
check.That(nd.GetDouble(1) == 88.5, "...Python's write reached nd");

py.Exec("def fsum(a):\n    return float(a.sum())\n\ndef process(a):\n    return a * 2.0 + 1.0");
using (Py.GIL())
{
    dynamic fsum = py.Get("fsum");
    double s = (double)fsum(nd);                                             // the NDArray argument is encoded by the codec
    check.That(s == 88.5 + 5.0, $"a Python function receives the NDArray as numpy: fsum(nd) == {s}");

    dynamic numpy = Py.Import("numpy");
    using NDArray a = np.arange(4).astype(NPTypeCode.Double).reshape(2, 2);
    using NDArray b = (np.arange(4).astype(NPTypeCode.Double) + 1).reshape(2, 2);
    using NDArray product = ((PyObject)numpy.matmul(a, b)).As<NDArray>();     // NDArrays in, NDArray out
    using NDArray expected = np.matmul(a, b);
    check.That(product.GetDouble(1, 1) == expected.GetDouble(1, 1), $"dynamic numpy.matmul(a, b).As<NDArray>() == np.matmul: [1,1] = {product.GetDouble(1, 1)}");
}

// ==== decode: As<NDArray>() and AsManagedObject =====================================================
check.Section("decode: Auto shares when a view is possible");
py.Exec("mm = np.arange(6, dtype='f8').reshape(2, 3)");
using (Py.GIL())
{
    using PyObject mmPy = py.Get("mm");
    using NDArray dec = mmPy.As<NDArray>();
    using NDArray dec2 = (NDArray)mmPy.AsManagedObject(typeof(NDArray))!;
    dec[0, 0] = -1.0;
    check.That(py.Float("mm[0, 0]") == -1.0 && dec2.GetDouble(0, 0) == -1.0, "As<NDArray>() and AsManagedObject(typeof(NDArray)) are both zero-copy views of the ndarray");
    check.That(NDArrayPythonInterop.LiveImports == 2, "each holds its own lease (LiveImports == 2)");
}
check.That(Settle(() => NDArrayPythonInterop.LiveImports == 0), "leases released with the views");

check.Section("decode: what CanDecode accepts");
using (Py.GIL())
{
    using (PyObject mx = py.Module.Eval("np.matrix('1 2; 3 4').astype('i8')"))
    using (NDArray dmx = mx.As<NDArray>())
        check.That(dmx.shape.SequenceEqual(new long[] { 2, 2 }) && dmx.GetInt64(1, 1) == 4, "numpy.matrix / ndarray subclasses (an __mro__ walk)");
    using (PyObject mv = py.Module.Eval("memoryview(b'ab')"))
    using (NDArray dm = mv.As<NDArray>())
        check.That(dm.typecode == NPTypeCode.Byte && !dm.Shape.IsWriteable, "memoryview / bytes / bytearray / array.array by name (read-only sources become NON-writeable views)");
    using (PyObject ct = py.Module.Eval("(ctypes.c_double * 3)(0.5, 1.5, 2.5)"))
    using (NDArray dct = ct.As<NDArray>())
        check.That(dct.typecode == NPTypeCode.Double && dct.GetDouble(2) == 2.5, "any other exporter via the PEP 688 __buffer__ capability check (Python 3.12+): ctypes arrays");
    using (PyObject list = py.Module.Eval("[[1, 2], [3, 4]]"))
    using (NDArray dl = list.As<NDArray>())
        check.That(dl.typecode == NPTypeCode.Int64 && dl.GetInt64(1, 0) == 3 && dl.Shape.IsWriteable, "array-like builtins (list / tuple / int / float / bool / complex) via numpy.asarray — always an owning copy");
    using (PyObject sc = py.Module.Eval("3.5"))
    using (NDArray ds = sc.As<NDArray>())
        check.That(ds.ndim == 0 && ds.GetDouble() == 3.5, "a Python float decodes to a 0-d float64");
    using PyObject dict = py.Module.Eval("{'a': 1}");
    check.Throws<InvalidCastException>(() => dict.As<NDArray>().Dispose(), "a dict is neither a buffer nor array-like — declined");
}

// ==== the modes, through standalone codec instances =================================================
// RegisterCodec is process-global and sticky, so the alternatives are exercised through NumpyCodec
// objects directly (the same IPyObjectDecoder/IPyObjectEncoder pythonnet would call).
check.Section("modes: Auto falls back, View declines, Copy detaches");
var auto = new NumpyCodec(NumpyCodecOptions.Default);
var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View, EncodeMode = NumpyCodecMode.View });
var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy, EncodeMode = NumpyCodecMode.Copy });
py.Exec("c8 = np.array([1+2j, 3+4j], dtype='c8')\nba = bytearray(b'abcd')\nf8 = np.arange(4, dtype='f8')");
using (Py.GIL())
{
    using PyObject c8 = py.Get("c8");
    check.That(auto.TryDecode(c8, out NDArray autoC8) && autoC8.typecode == NPTypeCode.Complex, "Auto: complex64 has no view -> falls back to a widening copy");
    autoC8.Dispose();
    check.That(!viewOnly.TryDecode(c8, out NDArray declined) && declined is null, "View: complex64 -> DECLINED (returns false) rather than a silent copy");

    using PyObject f8 = py.Get("f8");
    check.That(viewOnly.TryDecode(f8, out NDArray strict) && strict.Shape.IsWriteable, "View: a contiguous float64 array -> a shared view");
    strict[0] = 5.5;
    check.That(py.Float("f8[0]") == 5.5, "...writes cross");
    strict.Dispose();

    using PyObject ba = py.Get("ba");
    check.That(copyOnly.TryDecode(ba, out NDArray snap) && snap.Shape.IsWriteable, "Copy: a bytearray -> an owning snapshot");
    snap[0] = (byte)90;
    check.That(py.Long("ba[0]") == 97, "...that never shares memory");
    py.Exec("ba.append(1)");
    check.That(py.Long("len(ba)") == 5, "...and never locks the source (append works while the snapshot lives)");
    snap.Dispose();

    using NDArray src = np.arange(3).astype(NPTypeCode.Double);
    using PyObject encV = auto.TryEncode(src);
    using PyObject encC = copyOnly.TryEncode(src);
    py.Set("encV", encV); py.Set("encC", encC);
    py.Exec("encV[0] = 10.0\nencC[1] = 20.0");
    check.That(src.GetDouble(0) == 10.0 && src.GetDouble(1) == 1.0, "encode: Auto/View -> a view (write reaches NumSharp); Copy -> a copy (write does not)");

    using NDArray dec = np.arange(3).astype(NPTypeCode.Decimal);
    using PyObject encD = viewOnly.TryEncode(dec);
    py.Set("encD", encD);
    check.That(encD is not null && py.Str("encD.dtype") == "float64", "Decimal encodes as float64 in EVERY mode (no numpy decimal dtype exists)");
}

check.Section("options: which Python types decode");
var noBuffers = new NumpyCodec(new NumpyCodecOptions { DecodeAnyBuffer = false });
var noArrayLike = new NumpyCodec(new NumpyCodecOptions { DecodeArrayLike = false });
var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
using (Py.GIL())
{
    check.That(auto.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray)) && !noBuffers.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray)),
               "DecodeAnyBuffer=false: memoryview/bytes/ctypes... decline; numpy.ndarray still decodes");
    check.That(auto.CanDecode(TypeOf("[1, 2]"), typeof(NDArray)) && !noArrayLike.CanDecode(TypeOf("[1, 2]"), typeof(NDArray)),
               "DecodeArrayLike=false: list/tuple/scalars decline (decode stays numpy-agnostic)");
    check.That(!auto.CanDecode(TypeOf("'text'"), typeof(NDArray)) && !auto.CanDecode(TypeOf("range(3)"), typeof(NDArray)),
               "str and range are never array-like");
    check.That(auto.CanDecode(TypeOf("np.zeros(1)"), typeof(NDArray)) && !auto.CanDecode(TypeOf("np.zeros(1)"), typeof(string)),
               "CanDecode is keyed on the CLR target: only typeof(NDArray)");
    try
    {
        py.Exec("import pandas as pd");
        check.That(auto.CanDecode(TypeOf("pd.Series([1])"), typeof(NDArray)) && !noAdapters.CanDecode(TypeOf("pd.Series([1])"), typeof(NDArray)),
                   "DecodeArrayAdapters=false: adapter-only objects (pandas.Series, torch.Tensor) decline; buffers and array-likes are unaffected");
    }
    catch (PythonException) { Console.WriteLine("  (skip) pandas is not installed — adapter example in 10-pandas.cs"); }
}

check.Section("counters");
py.Exec("del c, mm, encV, encC, encD");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports == 0),
           $"every pin and lease released (exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports})");

return check.Exit();

PyType TypeOf(string expr)                                                   // call under Py.GIL()
{
    using PyObject o = py.Module.Eval(expr);
    using PyObject t = o.GetPythonType();
    return new PyType(t);
}

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
