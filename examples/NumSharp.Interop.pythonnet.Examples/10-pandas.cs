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
//      view callers pass allowReadonly:true (the codec's Auto mode does). NumSharp -> Pandas is the
//      ordinary numpy encoder plus an explicit copy=False.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example10_Pandas.cs.
//
//      Run:  dotnet run 10-pandas.cs        (exits quietly when pandas is not installed)

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

bool HasModule(string name) { using (Py.GIL()) return Throws(() => py.Exec($"import {name}")) is null; }
if (!HasModule("pandas")) return;                             // see requirements.txt — Pandas 3
using (Py.GIL()) py.Exec("import pandas as pd");

// ==== Pandas -> NumSharp ============================================================================
// Series: the ordinary verbs, through the adapter
using (Py.GIL())
{
    py.Exec("s = pd.Series(np.arange(6, dtype='i8'))");
    using PyObject s = py.s;
    Exception? readOnly = Throws(() => s.AsNDArray());                        // InvalidOperationException: ... Pandas 3 exposes the shared projection read-only; pass allowReadonly:true
    NDArray view = s.AsNDArray(allowReadonly: true);                          // a NON-writeable Int64 (6,) view over Pandas' own stable storage (two to_numpy() projections overlap)
    py.Exec("s.iloc[2] = 77");                                                // view[2] reads 77
    Exception? guarded = Throws(() => np.copyto(view, np.zeros(new Shape(6), np.int64)));   // NumSharpException: assignment destination is read-only
    NDArray copy = s.ToNDArray();                                             // an owning, writable, detached snapshot
    copy[0] = 999L;                                                           // s.iloc[0] still reads 0
    NDArray viaCodec = py.s;                                                  // the codec's Auto mode: the read-only view, by itself
}

// DataFrame: a homogeneous frame is ONE F-strided block
using (Py.GIL())
{
    py.Exec("df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]}, dtype='i8')");
    NDArray view = py.df;                                                     // shape (3,2), element strides (1,3): Fortran order, as Pandas stores blocks; read-only (Copy-on-Write); view[2, 1] == 30
    py.Exec("df.iloc[1, 1] = 77");                                            // view[1, 1] reads 77
}

// Index, RangeIndex and nullable extension arrays
using (Py.GIL())
{
    py.Exec("idx = pd.Index([4, 5, 6], dtype='i8')\nrng = pd.RangeIndex(5)\nnullable = pd.array([7, 8, 9], dtype='Int64')");
    NDArray iv = py.idx;                                                      // pd.Index: a read-only view (an Index is immutable); iv[2] == 6
    NDArray rv = py.rng;                                                      // pd.RangeIndex: recognized through the Index MRO; no buffer to share, so Auto materializes int64 [0..4]
    NDArray nv = py.nullable;                                                 // pd.array(dtype='Int64') without missing values exposes its stable typed buffer — writable
    nv[1] = 71L;                                                              // nullable[1] reads 71
}

// what materializes: mixed blocks, missing values, categoricals, text
using (Py.GIL())
{
    py.Exec("mixed = pd.DataFrame({'small': np.array([1, 2], dtype='i4'), 'real': np.array([3.5, 4.5], dtype='f8')})\n" +
            "with_na = pd.array([1, None, 3], dtype='Int64')\n" +
            "cat = pd.Series(pd.Categorical([3, 1, 3]))\n" +
            "text = pd.Series(['a', 'b'])");
    using PyObject mixed = py.mixed;
    Exception? noView = Throws(() => mixed.AsNDArray(allowReadonly: true));   // NotSupportedException: ... cannot be proven — Pandas materialized a fresh array for the mixed blocks
    NDArray upcast = mixed.ToNDArray();                                       // Pandas' common-dtype upcast (int32 + float64 -> float64), detached: upcast[1, 0] == 2.0, upcast[0, 1] == 3.5
    NDArray auto = py.mixed;                                                  // the codec's Auto falls back to the owning copy (writable, Double)
    using PyObject withNa = py.with_na;
    Exception? noBuffer = Throws(() => withNa.AsNDArray(allowReadonly: true));   // NotSupportedException: a nullable array WITH pd.NA has no stable buffer to share
    NDArray naCopy = withNa.ToNDArray();                                      // float64 with NaN for the missing value
    using PyObject cat = py.cat;
    Exception? categorical = Throws(() => cat.AsNDArray(allowReadonly: true));   // NotSupportedException: categorical — View declines
    NDArray catCopy = cat.ToNDArray();                                        // the category VALUES: [3, 1, 3]
    using PyObject text = py.text;
    Exception? strings = Throws(() => text.ToNDArray());                      // NotSupportedException: object/string columns have no NumSharp dtype (both paths refuse)
}

// labels are metadata: only the value matrix crosses
using (Py.GIL())
{
    py.Exec("labelled = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)]), columns=['dup', 'dup'])");
    NDArray values = py.labelled;                                             // a MultiIndex + duplicate column names: still the (2,2) value block, row-major logical order; values[1, 0] == 3.0
}

// ==== NumSharp -> Pandas: the numpy encoder + an explicit copy=False ================================
NDArray nd = np.arange(6).astype(np.float64);
using (Py.GIL())
{
    py.values = nd;                                                           // the codec encodes a zero-copy numpy view
    py.Exec("series = pd.Series(values, copy=False)\nframe = pd.DataFrame(values.reshape(-1, 2), copy=False)");
    bool onNumSharpMemory = py.Eval("series.to_numpy().ctypes.data == values.ctypes.data");   // True: pd.Series(values, copy=False) sits on NumSharp's memory
    bool writeable = py.Eval("series.to_numpy().flags.writeable");            // False: Pandas 3 marks the shared projection read-only (its Copy-on-Write rule)
    nd[2] = 81.5;                                                             // series.iloc[2] and frame.iloc[1, 0] both read 81.5
    py.Exec("series.iloc[2] = -7.25");                                        // nd[2] reads -7.25: copy=False is a deliberate shared-memory opt-in; CoW cannot detach an external owner
    py.Exec("default_copy = pd.Series(values)");                              // without copy=False Pandas 3 copies numpy input by default: default_copy.to_numpy().ctypes.data != values.ctypes.data
}

// lifetime: a derived NumSharp view keeps Pandas storage alive after every Python name dies
NDArray slice;
using (Py.GIL())
{
    py.Exec("tmp = pd.Series(np.arange(8, dtype='f8') * 1.5)");
    using (var inner = NDScope.Open())
    {
        NDArray whole = py.tmp;                                               // a read-only view over the Series' projection
        slice = inner.Returns(whole["2:5"]);                                  // yielded to the outer scope; `whole` is released when `inner` closes
    }
    py.Exec("del tmp\nimport gc; gc.collect()");
}
double s0 = slice.GetDouble(0), s2 = slice.GetDouble(2);                      // 3.0, 6.0 — the slice still reads valid memory; its lease holds the projection (LiveImports == 1)
slice.Dispose();

// registry: the adapter is built in
bool alreadyThere = !PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance);   // true — already registered under the name "pandas"
var noAdapters = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
using (Py.GIL())
{
    using PyObject s = py.s;
    using PyObject t = s.GetPythonType();
    using var seriesType = new PyType(t);
    bool decodes = noAdapters.CanDecode(seriesType, typeof(NDArray));       // false: a codec with DecodeArrayAdapters = false does not decode a Series
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every view above (the leases on the Pandas projections go back);
// `host` drops the namespace, so values, series and frame die with their names and the export pin with them;
// then the engine shuts down. LiveExports / LiveImports both read 0 afterwards.

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
