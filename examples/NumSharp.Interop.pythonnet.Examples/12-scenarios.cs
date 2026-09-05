#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 12 — Scenarios: realistic application shapes driven end to end through the interop. An ML
//      inference hand-off, an in-place image pipeline, a long-lived telemetry ring, a Python-owned
//      dataset outliving its Python references under NumSharp kernels, a co-simulation with mixed
//      ownership, a codec-driven service loop, generators, Python exceptions, JSON, and faulty inputs.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example12_Scenarios.cs.
//
//      Run:  dotnet run 12-scenarios.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

// ==== 1. ML inference: C# owns feature engineering, Python owns the model ============================
NDArray weights = np.array(new double[] { 0.5, -1.0, 2.0, 0.25 });
using (Py.GIL())
{
    py.w = weights.ToNumpyCopy();                                   // the model's weights live on the Python side
    for (int b = 0; b < 10; b++)
    using (NDScope.Open())                                          // one scope per batch: the batch's temporaries are released as it ends
    {
        NDArray feats = np.arange(b, b + 12).astype(np.float64).reshape(3, 4) / 4.0;   // C# engineers the features...
        py.batch = feats;                                           // ...they cross zero-copy...
        NDArray pred = py.Eval("batch.dot(w) - batch.mean()");      // ...Python's model predicts, and the prediction comes back (a view of numpy's fresh result)
        NDArray expected = np.dot(feats, weights) - np.mean(feats); // == pred, to 1e-9, for every batch
    }
}

// ==== 2. Image pipeline: Python edits a C#-owned uint8 image IN PLACE ================================
NDArray img = np.arange(32 * 32 * 3).astype(np.uint8).reshape(32, 32, 3);
NDArray crop = img["8:24, 8:24, :"];
using (Py.GIL())
{
    py.img = img;
    py.Exec("img[:] = np.minimum(img.astype('i2') + 7, 255).astype('u1')");   // numpy writes NumSharp's pixels: every pixel is 7 brighter, in place, no copy anywhere
    py.crop = crop;                                                 // a strided window over the same buffer: np.shares_memory(crop, img) == True
    double cropMean = py.crop.mean();                               // == np.mean(crop), to 1e-9 — both sides agree
}

// ==== 3. Telemetry ring: hundreds of window exports, a bounded live count ============================
NDArray ring = np.zeros(new Shape(256));
var rnd = new Random(42);
using (Py.GIL())
{
    int baseline = NDArrayPythonInterop.LiveExports;               // batch, img and crop are still bound in the namespace
    for (int step = 0; step < 150; step++)
    using (NDScope.Open())
    {
        int start = step * 17 % 192;
        for (int k = 0; k < 16; k++) ring[start + k] = rnd.NextDouble() * 10;
        NDArray window = ring[$"{start}:{start + 16}"];
        py.win = window;                                            // rebinding `win` drops the previous export: the live count never grows unbounded (it peaks a few above the baseline)
        double std = py.Eval("win.std()");                          // == np.std(window), to 1e-9, every step — read through Eval: a dynamic `py.win` would hand out a wrapper that keeps the previous window's export alive until the GC gets to it
    }
    py.Exec("del win");
    PythonHost.Drain();                                             // LiveExports is back at the baseline
}

// ==== 4. A Python dataset outlives its Python references under NumSharp kernels =====================
NDArray x, y;
using (Py.GIL())
{
    py.Exec("ds = {'x': np.arange(12, dtype='f8').reshape(3, 4) * 0.5, 'y': np.arange(3, dtype='i8') * 10}");
    x = py.ds["x"];                                                 // two leases on Python's arrays
    y = py.ds["y"];
    py.Exec("del ds\nimport gc; gc.collect()");                     // only the leases keep the arrays alive now
}
double total = np.sum(x).GetDouble();                               // 33.0 — sum / mean kernels run over leased Python memory
double mean = np.mean(x).GetDouble();                               // 2.75
long ySum = np.sum(y).GetInt64();                                   // 30
NDArray scaled = x * 2.0;                                           // elementwise kernels read the lease and write fresh NumSharp memory: scaled[2, 3] == 11.0, x[2, 3] == 5.5

// ==== 5. Co-simulation: positions in C#, velocities in Python, both integrated in place =============
NDArray pos = np.zeros(new Shape(8));
NDArray vel;
using (Py.GIL())
{
    py.pos = pos;
    py.Exec("vel = np.arange(8, dtype='f8') - 3.5");
    vel = py.vel;
    for (int step = 0; step < 20; step++)
    using (NDScope.Open())
    {
        py.Exec("pos += vel * 0.1");                                // Python integrates C#-owned positions in place
        np.copyto(vel, vel * 0.9);                                  // C# damps Python-owned velocities in place
    }
}
// after 20 steps: pos[i] == (i - 3.5) * (1 - 0.9^20) and vel[i] == (i - 3.5) * 0.9^20, to 1e-9 — mixed ownership stays coherent

// ==== 6. A long-lived service through the codec: no explicit conversion calls ========================
using (Py.GIL())
{
    py.Exec("def process(a):\n    return a * 2.0 + 1.0");
    for (int i = 0; i < 40; i++)
    using (NDScope.Open())
    {
        NDArray batch = np.arange(6).astype(np.float64) + i;
        NDArray result = py.process(batch);                         // the batch is auto-encoded, the fresh ndarray auto-decoded as a view: result[j] == (j + i) * 2 + 1
    }
}

// ==== 7. Generators stream chunks into NumSharp =====================================================
using (Py.GIL())
{
    py.Exec("def chunks(n):\n    for i in range(n):\n        yield np.arange(3, dtype='f8') + 10 * i");
    var rows = new List<NDArray>();
    foreach (PyObject chunk in new PyIterable(py.chunks(4)))        // each yielded array...
        rows.Add(chunk.ToNDArray());                                // ...is copied out (the generator's temporaries do not outlive the loop)
    NDArray stacked = np.vstack(rows.ToArray());                    // (4,3); stacked[3, 2] == 32.0
}

// ==== 8. Exceptions, JSON, faulty inputs ============================================================
using (Py.GIL())
{
    py.ex = np.arange(3).astype(np.float64);
    Exception? raised = Throws(() => py.Exec("raise ValueError('bad shape: ' + str(ex.shape))"));   // PythonException: bad shape: (3,) — a Python raise is a .NET exception; the engine stays usable
    py.Exec("ex[0] = 7.5");
    double sum = py.ex.sum();                                       // 10.5: the same shared array keeps working afterwards

    py.jx = np.arange(5).astype(np.float64) * 1.5;
    py.Exec("import json\npayload = json.dumps(jx.tolist())");      // "[0.0, 1.5, 3.0, 4.5, 6.0]"
    NDArray back = py.Eval("np.asarray(json.loads(payload), dtype='f8')");   // json.loads -> np.asarray -> NDArray: back[4] == 6.0, lossless

    int exports = NDArrayPythonInterop.LiveExports, imports = NDArrayPythonInterop.LiveImports;
    using PyObject bytes = py.Eval("b'abcd'");
    Exception? readOnly = Throws(() => bytes.AsNDArray());          // InvalidOperationException: a read-only source without allowReadonly
    using PyObject dict = py.Eval("{'a': 1}");
    Exception? notArrayLike = Throws(() => dict.ToNDArray());       // NotSupportedException: a dict
    using PyObject rec = py.Eval("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])");
    Exception? structured = Throws(() => rec.ToNDArray());          // NotSupportedException: a structured dtype
    NDArray dead = np.arange(3);
    dead.Dispose();
    Exception? disposed = Throws(() => dead.ToNumpy());             // ObjectDisposedException: a disposed NDArray
    // LiveExports == exports and LiveImports == imports: no pin or lease survives a failed conversion
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every array above — x, y, vel, back and the rest, so their leases
// go back to Python; `host` drops the namespace, so w, batch, img, crop, pos, ex and jx die with their names
// and the export pins with them; then the engine shuts down. LiveExports / LiveImports both read 0 afterwards.

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
