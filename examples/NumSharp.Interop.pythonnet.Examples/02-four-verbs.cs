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
//      deliberate exception because the zero-copy view is the package's headline. With the codec
//      registered the two sharing verbs also run implicitly at every pythonnet boundary: `py.x = nd`
//      is ToNumpy, `NDArray a = py.x` is AsNDArray (a copy only when no view is possible).
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example02_FourVerbs.cs.
//
//      Run:  dotnet run 02-four-verbs.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();          // finds CPython + numpy, starts the engine, registers the codec
using var scope = NDScope.Open();             // every NDArray built below is released when the scope closes — nothing is disposed by hand
dynamic py = host.Namespace;                  // a Python module where `import numpy as np` already ran; talk to it under Py.GIL()

NDArray nd = np.arange(4).astype(np.float64); // [0. 1. 2. 3.]  — NumSharp work needs no GIL

// ---- NumSharp -> Python: ToNumpy (view) vs ToNumpyCopy (copy) -------------------------------------
using (Py.GIL())                              // Python work runs under the GIL; the conversion verbs take it themselves, re-entrantly
{
    py.v = nd.ToNumpy();                      // v: a numpy view over NumSharp's bytes
    py.c = nd.ToNumpyCopy();                  // c: numpy owns a separate copy
    py.Exec("v[0] = 11.0; c[1] = 22.0");      // Python writes into both...
    // nd is now [11. 1. 2. 3.]: the view's write arrived, the copy's did not

    bool shared = py.np.shares_memory(py.v, py.c);      // False
    bool viewOwnsData = py.v.flags.owndata;             // False — numpy agrees the view's memory is NumSharp's
    bool copyOwnsData = py.c.flags.owndata;             // True

    nd[2] = -5.0;                                       // NumSharp writes...
    double v2 = py.v[2];                                // -5.0: the view shows it
    double c2 = py.c[2];                                //  2.0: the copy does not
}

// ---- the aliases -----------------------------------------------------------------------------------
using (Py.GIL())
{
    py.p = nd.ToPython();                     // ToPython() == ToNumpy(): the view
    py.r = nd.ToNumpy(copy: true);            // copy: true routes to ToNumpyCopy
    bool pIsTheView = py.np.shares_memory(py.p, py.v);  // True
    bool rIsACopy = py.np.shares_memory(py.r, py.v);    // False
}

// ---- the everyday spelling: the codec runs ToNumpy for you -------------------------------------------
using (Py.GIL())
{
    py.w = nd;                                // no verb, no PyObject: binding an NDArray by name exports the view
    bool wIsTheView = py.np.shares_memory(py.w, py.v);  // True
}

// ---- Python -> NumSharp: ToNDArray (copy) vs AsNDArray (view) -------------------------------------
NDArray copied, leased;
using (Py.GIL())
{
    py.Exec("src = np.arange(6, dtype='f8').reshape(2, 3) * 0.5");
    using PyObject src = py.src;
    copied = src.ToNDArray();                 // fresh, owning, C-contiguous (copied.Shape.IsContiguous, copied.typecode == Double)
    leased = src.AsNDArray();                 // == NDArrayPythonInterop.ToNDArrayView(src): a lease on numpy's buffer

    copied[0, 0] = -9.0;                      // the copy never reaches numpy:              src[0, 0] is still 0.0
    leased[0, 0] = -9.0;                      // the view does:                             src[0, 0] is now -9.0
    py.Exec("src[1, 2] = 42.0");              // and numpy's write is visible through it:   leased[1, 2] == 42.0
    int live = NDArrayPythonInterop.LiveImports;        // 1 — one lease while the view is alive
}

// ---- the everyday spelling: the codec runs AsNDArray for you --------------------------------------------
using (Py.GIL())
{
    NDArray viaCodec = py.src;                // a view whenever one is possible (Auto mode), a copy otherwise; viaCodec[1, 2] == 42.0
}

// ---- non-contiguous sources: copies linearize, views keep strides ---------------------------------------
using (Py.GIL())
{
    using PyObject t = py.src.T;              // (3,2): a Fortran-strided view of src
    NDArray tCopy = t.ToNDArray();            // C-contiguous, logical order preserved: tCopy[2, 1] == 42.0
    NDArray strided = t.AsNDArray();          // the layout numpy has: shape (3,2), element strides (1,3)
}

// ---- read-only sources --------------------------------------------------------------------------------------
using (Py.GIL())
{
    using PyObject ro = py.Eval("b'\\x01\\x02\\x03\\x04'");     // bytes: immutable
    Exception? refused = Throws(() => ro.AsNDArray());          // InvalidOperationException: the exporter's buffer is read-only ... pass allowReadonly:true
    NDArray view = ro.AsNDArray(allowReadonly: true);           // opt in: a NON-writeable view (view.Shape.IsWriteable == false); view[1] == 2
    Exception? guarded = Throws(() => view[0] = (byte)7);       // NumSharpException: assignment destination is read-only — Python's bytes stay intact
    NDArray owned = ro.ToNDArray();                             // or copy: an owning, writable array (owned.Shape.IsWriteable == true)
}

// ---- an import view owns nothing (numpy's owndata == False) ------------------------------------------------
Exception? cannotResize = Throws(() => leased.resize(new Shape(12)));   // IncorrectShapeException: cannot resize this array: it does not own its data
NDArray owning = np.require(leased, (Type?)null, "O");                 // the escape hatch: an owning copy
owning[0, 0] = 123.0;                                                  // src[0, 0] is still -9.0

// ---- array-likes (list / tuple / scalar): FromArrayLike goes through numpy.asarray -----------------------
using (Py.GIL())
{
    using PyObject list = py.Eval("[[1, 2, 3], [4, 5, 6]]");
    NDArray fromList = list.FromArrayLike();                    // an owning int64 (2,3); fromList[1, 2] == 6
    using PyObject scalar = py.Eval("2.5");
    NDArray fromScalar = scalar.FromArrayLike();                // a 0-d float64
    Exception? notABuffer = Throws(() => list.AsNDArray());     // NotSupportedException: a list exports no PEP 3118 buffer
    NDArray implicitly = py.Eval("[[1, 2, 3], [4, 5, 6]]");     // the codec does the same by itself — always an owning copy, a list has no memory to share
}

// ---- who frees what ------------------------------------------------------------------------------------------
// When this script ends: `scope` disposes every NDArray built above in one sweep — the leases on src's buffer
// go back to numpy; `host` drops the namespace, so v, c, p, r and w die with their names and the export pins
// on nd's buffer with them; then the engine shuts down. NDArrayPythonInterop.LiveExports / LiveImports are
// the two counters that make every live crossing observable — 06-lifetime.cs is about them.

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
