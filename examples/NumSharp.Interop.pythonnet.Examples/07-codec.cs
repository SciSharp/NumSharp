#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 07 — The codec: conversions without calls. RegisterCodec() hooks NumSharp into pythonnet's
//      conversion pipeline, after which every boundary converts by itself — `py.x = nd`, passing an
//      NDArray to a Python callable, `dynamic` numpy calls with NDArray arguments, `NDArray a = py.x`,
//      pyObj.As<NDArray>() / AsManagedObject(typeof(NDArray)) — and C# tuples cross as Python tuples
//      (shapes, axes, multi-indices) and back. NumpyCodecOptions picks the policy per direction
//      (Auto / View / Copy) and which Python types decode.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example07_Codec.cs.
//
//      Run:  dotnet run 07-codec.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start(codec: false);     // this example registers the codec itself, to show the ordering trap
using var scope = NDScope.Open();
dynamic py = host.Namespace;
using (Py.GIL()) py.Exec("import ctypes");

// ==== the one ordering trap: register BEFORE the first As<NDArray>() ===============================
using (Py.GIL())
{
    using PyObject probe = py.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
    Exception? before = Throws(() => probe.As<NDArray>());                 // InvalidCastException: no decoder yet — and pythonnet CACHES this miss for the pair (c_short_Array_7, NDArray)
    bool registered = NDArrayPythonInterop.RegisterCodec();                // true
    using PyObject again = py.Eval("(ctypes.c_short * 7)(1, 2, 3, 4, 5, 6, 7)");
    Exception? stillCached = Throws(() => again.As<NDArray>());            // InvalidCastException: the SAME Python type still fails — the miss is cached for the whole session
    using PyObject fresh = py.Eval("(ctypes.c_short * 8)(1, 2, 3, 4, 5, 6, 7, 8)");
    NDArray ok = fresh.As<NDArray>();                                      // a pair first touched AFTER registration decodes fine: Int16, ok[7] == 8
    bool second = NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });   // false: the first registration's policy sticks for the session
}

// ==== encode: every pythonnet boundary ==============================================================
NDArray nd = np.arange(4).astype(np.float64);
using (Py.GIL())
{
    py.c = nd;                                                             // a scope binding: a real ndarray borrowing NumSharp's memory (c.flags.owndata == False)
    py.Exec("c[1] = 88.5");                                                // nd[1] reads 88.5
    py.Exec("def fsum(a):\n    return float(a.sum())\n\ndef process(a):\n    return a * 2.0 + 1.0");
    double s = py.fsum(nd);                                                // 93.5 — a Python function receives the NDArray as numpy
    NDArray processed = py.process(nd);                                    // [1. 178. 5. 7.] — and its ndarray result comes back as an NDArray (a view of it)
    NDArray a = np.arange(4).astype(np.float64).reshape(2, 2);
    NDArray b = (np.arange(4).astype(np.float64) + 1).reshape(2, 2);
    NDArray product = py.np.matmul(a, b);                                  // NDArrays in, NDArray out: == np.matmul(a, b), product[1, 1] == 16
    NDArray zeros = py.np.zeros((2, 3));                                   // a C# tuple crossed as the shape: (2,3) float64
    (long, long) shape = py.np.zeros((2, 3)).shape;                        // (2, 3) — and a Python tuple came back as a C# one
}

// ==== decode: As<NDArray>(), AsManagedObject and the dynamic spelling =================================
using (Py.GIL())
{
    py.Exec("mm = np.arange(6, dtype='f8').reshape(2, 3)");
    using PyObject mmPy = py.mm;
    NDArray dec = mmPy.As<NDArray>();                                      // a zero-copy view (Auto mode)
    NDArray dec2 = (NDArray)mmPy.AsManagedObject(typeof(NDArray))!;        // the non-generic spelling: another view
    NDArray dec3 = py.mm;                                                  // the dynamic spelling: another view
    dec[0, 0] = -1.0;                                                      // mm[0, 0], dec2[0, 0] and dec3[0, 0] all read -1.0
    int leases = NDArrayPythonInterop.LiveImports;                         // 3 — each view holds its own lease
}

// what CanDecode accepts
using (Py.GIL())
{
    NDArray matrix = py.Eval("np.matrix('1 2; 3 4').astype('i8')");       // numpy.matrix / ndarray subclasses (an __mro__ walk): (2,2), matrix[1, 1] == 4
    NDArray mv = py.Eval("memoryview(b'ab')");                             // memoryview / bytes / bytearray / array.array by name — a read-only source is a NON-writeable view
    NDArray ct = py.Eval("(ctypes.c_double * 3)(0.5, 1.5, 2.5)");          // any other exporter via the PEP 688 __buffer__ capability check (Python 3.12+): ct[2] == 2.5
    NDArray list = py.Eval("[[1, 2], [3, 4]]");                            // array-like builtins (list / tuple / int / float / bool / complex) via numpy.asarray — always an owning copy
    NDArray scalar = py.Eval("3.5");                                       // a Python float: a 0-d float64
    using PyObject dict = py.Eval("{'a': 1}");
    Exception? notArrayLike = Throws(() => dict.As<NDArray>());            // InvalidCastException: neither a buffer nor array-like
}

// ==== the modes, through standalone codec instances =================================================
// RegisterCodec is process-global and sticky, so the alternatives are exercised through NumpyCodec objects
// directly — the same IPyObjectDecoder / IPyObjectEncoder pythonnet would call.
var auto = new NumpyCodec(NumpyCodecOptions.Default);
var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View, EncodeMode = NumpyCodecMode.View });
var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy, EncodeMode = NumpyCodecMode.Copy });
using (Py.GIL())
{
    using PyObject c8 = py.Eval("np.array([1+2j, 3+4j], dtype='c8')");
    bool autoDecoded = auto.TryDecode(c8, out NDArray widened);            // true  — Auto: complex64 has no view, so it falls back to a widening copy (typecode Complex)
    bool viewDecoded = viewOnly.TryDecode(c8, out NDArray declined);       // false — View: DECLINED rather than a silent copy (declined == null)

    py.Exec("f8 = np.arange(4, dtype='f8')");
    using PyObject f8 = py.f8;
    viewOnly.TryDecode(f8, out NDArray strict);                            // true: a contiguous float64 array -> a shared view
    strict[0] = 5.5;                                                       // f8[0] reads 5.5

    py.Exec("ba = bytearray(b'abcd')");
    using PyObject ba = py.ba;
    copyOnly.TryDecode(ba, out NDArray snapshot);                          // true: an owning snapshot
    snapshot[0] = (byte)90;                                                // ba[0] still reads 97: it never shares memory...
    py.Exec("ba.append(1)");                                               // ...and never locks the source: len(ba) == 5 while the snapshot lives

    NDArray src = np.arange(3).astype(np.float64);
    py.encV = auto.TryEncode(src);                                         // Auto / View encode: a view
    py.encC = copyOnly.TryEncode(src);                                     // Copy encode: a copy
    py.Exec("encV[0] = 10.0; encC[1] = 20.0");                             // src reads [10. 1. 2.]: the view's write arrived, the copy's did not
    NDArray dec = np.arange(3).astype(NPTypeCode.Decimal);
    py.encD = viewOnly.TryEncode(dec);                                     // Decimal encodes as float64 in EVERY mode (no numpy decimal dtype exists): encD.dtype == float64
}

// ==== options: which Python types decode ==============================================================
var noBuffers = new NumpyCodec(new NumpyCodecOptions { DecodeAnyBuffer = false });
var noArrayLike = new NumpyCodec(new NumpyCodecOptions { DecodeArrayLike = false });
using (Py.GIL())
{
    PyType TypeOf(string expr) { using PyObject o = py.Eval(expr); using PyObject t = o.GetPythonType(); return new PyType(t); }
    bool memoryviewByDefault = auto.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray));   // true
    bool memoryviewOff = noBuffers.CanDecode(TypeOf("memoryview(b'ab')"), typeof(NDArray));    // false — DecodeAnyBuffer = false: memoryview / bytes / ctypes ... decline; numpy.ndarray still decodes
    bool listByDefault = auto.CanDecode(TypeOf("[1, 2]"), typeof(NDArray));                    // true
    bool listOff = noArrayLike.CanDecode(TypeOf("[1, 2]"), typeof(NDArray));                   // false — DecodeArrayLike = false: list / tuple / scalars decline (decode stays numpy-agnostic)
    bool text = auto.CanDecode(TypeOf("'text'"), typeof(NDArray));                             // false — str and range are never array-like
    bool asString = auto.CanDecode(TypeOf("np.zeros(1)"), typeof(string));                     // false — CanDecode is keyed on the CLR target: only typeof(NDArray)
    // DecodeArrayAdapters = false makes adapter-only objects (pandas.Series, torch.Tensor) decline — 10-pandas.cs and 11-custom-adapter.cs show it;
    // ConvertTuples = false leaves C# tuples to pythonnet's default wrapping.
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every view above (their leases go back to numpy); `host` drops the
// namespace, so c, encV, encC and encD die with their names and the export pins with them; then the engine
// shuts down. LiveExports / LiveImports both read 0 afterwards.

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
