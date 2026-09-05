#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 03 — Layouts. Exporting keeps every NumSharp layout's strides and flags intact (never a copy,
//      never a flattening); importing takes one of three zero-copy routes chosen by what the
//      exporter is. Each import route is shown to be a real view by a write crossing it.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example03_Layouts.cs.
//
//      Run:  dotnet run 03-layouts.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();                  // the slices, transposes and views below are all released when the scope closes
dynamic py = host.Namespace;

// ==== NumSharp -> numpy: every layout class, strides intact ========================================
NDArray b = np.arange(24).reshape(4, 6).astype(np.float64);          // C-contiguous (4,6) float64
NDArray scalar = np.array(2.5);

using (Py.GIL())
{
    py.l_c    = b;                                                   // each binding is a zero-copy numpy view (the codec)...
    py.l_rows = b["1:3"];                                            // ...carrying the NumSharp view's own strides and flags:
    py.l_cols = b["1:3, 0:6:2"];
    py.l_t    = b.T;
    py.l_r    = b["::-1"];
    py.l_f    = np.asfortranarray(b);
    py.l_b    = np.broadcast_to(np.arange(3).astype(np.float64), new Shape(2, 3));
    py.l_0    = scalar;
    py.l_e    = np.zeros(new Shape(0, 3), np.float64);

    //   name     NumSharp layout                shape    strides (bytes)   c_contiguous  f_contiguous  writeable  owndata
    //   l_c      b: C-contiguous                (4, 6)   (48, 8)           True          False         True       False
    //   l_rows   b["1:3"]: row slice            (2, 6)   (48, 8)           True          False         True       False
    //   l_cols   b["1:3, 0:6:2"]: 2-D slice     (2, 3)   (48, 16)          False         False         True       False
    //   l_t      b.T: transpose                 (6, 4)   (8, 48)           False         True          True       False
    //   l_r      b["::-1"]: reversed            (4, 6)   (-48, 8)          False         False         True       False
    //   l_f      asfortranarray(b)              (4, 6)   (8, 32)           False         True          True       False
    //   l_b      broadcast_to (3,)->(2,3)       (2, 3)   (0, 8)            False         False         False      False
    //   l_0      np.array(2.5): 0-d             ()       ()                True          True          True       False
    //   l_e      zeros((0, 3)): empty           (0, 3)   (0, 0)            True          True          True       True
    (long, long) cStrides = py.l_c.strides;                          // (48, 8)   — the tuple codec reads a Python tuple straight into a C# one
    (long, long) tStrides = py.l_t.strides;                          // (8, 48)
    (long, long) rStrides = py.l_r.strides;                          // (-48, 8)  — a negative stride survives
    (long, long) bStrides = py.l_b.strides;                          // (0, 8)    — a stride-0 axis survives, read-only on both sides

    bool rowsShare = py.np.shares_memory(py.l_rows, py.l_c);         // True: a slice is a window over the same export
    string cols = py.l_cols.tolist().ToString();                     // [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]
    string lastRowFirst = py.l_r[0].tolist().ToString();             // [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    Exception? readOnly = Throws(() => py.Exec("l_b[0, 0] = 1.0"));  // PythonException: assignment destination is read-only
    py.Exec("l_0[()] = 7.5");                                        // scalar.GetDouble() is now 7.5 — a 0-d view writes through
    string emptyDtype = py.l_e.dtype.name;                           // "float64" — shape and dtype survive; no bytes, so nothing is pinned

    // what the numpy array is made of:
    string baseType = py.Eval("type(l_c.base).__name__");            // "ndarray"          — a reshape over...
    string windowType = py.Eval("type(l_c.base.base).__name__");     // "c_char_Array_192" — ...np.frombuffer over a ctypes window of 24 x 8 bytes
    string stridedType = py.Eval("type(l_cols.base).__name__");      // "DummyArray"       — np.lib.stride_tricks.as_strided over the same window
}

// ==== numpy -> NumSharp: three zero-copy routes ======================================================
// route 1: a C-contiguous PEP 3118 exporter (any object)
using (Py.GIL())
{
    py.Exec("r1 = bytearray(b'abcd')");
    NDArray v1 = py.r1;                                              // a PyBUF.WRITABLE lease on the bytearray (the codec ran AsNDArray)
    v1[0] = (byte)90;                                                // r1[0] is now 90
    Exception? locked = Throws(() => py.Exec("r1.append(1)"));       // PythonException (BufferError): Existing exports of data: object cannot be re-sized
}

// route 2: non-contiguous numpy arrays via __array_interface__
using (Py.GIL())
{
    py.Exec("r2 = np.arange(20, dtype='i8')");
    NDArray v2 = py.Eval("r2[::2]");                                 // size 10, element stride 2: the layout is reproduced, not copied
    v2[1] = -3L;                                                     // lands at r2[2] — logical index 1 of the strided view

    NDArray vt = py.Eval("np.arange(12, dtype='f8').reshape(3, 4).T");                  // shape (4,3), element strides (1,4)
    NDArray vf = py.Eval("np.asfortranarray(np.arange(12, dtype='f8').reshape(3, 4))"); // IsFContiguous, not C-contiguous
    NDArray vr = py.Eval("np.arange(6, dtype='f8')[::-1]");                             // element stride -1, vr[0] == 5.0 (the window is normalized below numpy's data pointer)
    NDArray vb = py.Eval("np.broadcast_to(np.arange(3, dtype='f8'), (2, 3))");          // IsBroadcasted and NOT writeable — the codec's Auto mode takes the read-only view (AsNDArray would need allowReadonly: true)
    py.Exec("z0 = np.array(2.5)");                                   // (np.float64(2.5).reshape(()) would be READ-ONLY: a numpy scalar's buffer)
    NDArray v0 = py.z0;                                              // a 0-d view
    v0.SetValue(9.5);                                                // z0 reads 9.5
}

// route 3: non-contiguous NON-numpy exporters via PyBUF.STRIDED
using (Py.GIL())
{
    py.Exec("r3 = bytearray(range(16))");
    NDArray v3 = py.Eval("memoryview(r3)[::2]");                     // size 8, element stride 2: the buffer protocol hands over the pointer, the memoryview the strides
    v3[3] = (byte)77;                                                // r3[6] is now 77
    NDArray v3r = py.Eval("memoryview(r3)[::-1]");                   // element stride -1, v3r[0] == 15
}

// the only layouts that cannot be viewed decline with guidance (the codec's Auto mode copies them instead)
using (Py.GIL())
{
    using PyObject sub = py.Eval("np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), shape=(2,), strides=(2,))");   // a 2-byte stride over int32
    Exception? declined = Throws(() => sub.AsNDArray());            // NotSupportedException: stride 2 bytes is not a multiple of itemsize 4 ... Use ToNDArray (copy).
    NDArray linearized = sub.ToNDArray();                            // size 2, Int32 — linearized through memoryview.tobytes('C')
    NDArray auto = sub.As<NDArray>();                                // what the codec does for it: the same owning copy (auto.Shape.IsWriteable == true)
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every view above — the leases on r1, r2, r3 and the unnamed numpy
// temporaries go back; `host` drops the namespace, so l_c ... l_e die with their names and the export pins on
// b's buffer with them; then the engine shuts down. LiveExports / LiveImports both read 0 afterwards.

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
