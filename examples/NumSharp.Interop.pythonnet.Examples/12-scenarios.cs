#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 12 — Scenarios: realistic application shapes driven end to end through the interop, each
//      cross-checked against a pure-NumSharp recomputation and the live counters. An ML inference
//      hand-off, an in-place image pipeline, a long-lived telemetry ring, a Python-owned dataset
//      outliving its Python references under NumSharp kernels, a co-simulation with mixed ownership,
//      a codec-driven service loop, generators, Python exceptions, JSON, and faulty inputs.
//
//      Run:  dotnet run 12-scenarios.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();
NDArrayPythonInterop.RegisterCodec();

void Export(string name, NDArray nd) { using (Py.GIL()) { using PyObject p = nd.ToNumpy(); py.Set(name, p); } }
NDArray Import(string expr) { using (Py.GIL()) { using PyObject p = py.Module.Eval(expr); return p.ToNDArray(); } }
NDArray View(string expr) { using (Py.GIL()) { using PyObject p = py.Module.Eval(expr); return p.AsNDArray(); } }

// ==== 1. ML inference: C# owns feature engineering, Python owns the model ============================
check.Section("ML inference: features cross zero-copy, predictions come back, every batch cross-checked");
using NDArray weights = new NDArray(new double[] { 0.5, -1.0, 2.0, 0.25 });
using (Py.GIL()) { using PyObject w = weights.ToNumpyCopy(); py.Set("w", w); }
bool allBatches = true;
for (int b = 0; b < 10; b++)
{
    using NDArray feats = np.arange(b, b + 12).astype(NPTypeCode.Double).reshape(3, 4) / 4.0;
    Export("batch", feats);
    py.Exec("pred = batch.dot(w) - batch.mean()");
    using NDArray pred = Import("pred");
    using NDArray expected = np.dot(feats, weights) - np.mean(feats);
    for (int i = 0; i < 3; i++) allBatches &= Math.Abs(pred.GetDouble(i) - expected.GetDouble(i)) < 1e-9;
}
check.That(allBatches, "10 batches: python's batch.dot(w) - batch.mean() == NumSharp's np.dot/np.mean");

// ==== 2. Image pipeline: Python edits a C#-owned uint8 image IN PLACE ================================
check.Section("image pipeline: in-place brightness in Python, then crop statistics on a strided window");
using NDArray img = np.arange(32 * 32 * 3).astype(NPTypeCode.Byte).reshape(32, 32, 3);
Export("img", img);
py.Exec("img[:] = np.minimum(img.astype('i2') + 7, 255).astype('u1')");        // numpy writes NumSharp's pixels
byte original = (byte)(((5 * 32 + 7) * 3 + 1) % 256);
check.That(img.GetByte(5, 7, 1) == (byte)Math.Min(255, original + 7), "pixel (5,7,1) was brightened in place — no copy anywhere");
using NDArray crop = img["8:24, 8:24, :"];
Export("crop", crop);
check.That(py.Bool("np.shares_memory(crop, img)"), "the crop is a strided window over the same buffer");
check.That(Math.Abs(py.Float("crop.mean()") - np.mean(crop).GetDouble()) < 1e-9, "both sides agree on the crop's mean");

// ==== 3. Telemetry ring: hundreds of window exports, bounded live count ==============================
check.Section("telemetry ring: 150 sliding-window exports drain as the scope name is rebound");
using NDArray ring = np.zeros(new Shape(256));
var rnd = new Random(42);
int e0 = NDArrayPythonInterop.LiveExports;                                        // batch/img/crop are still bound in the scope
int peak = 0;
bool statsOk = true;
for (int step = 0; step < 150; step++)
{
    int start = step * 17 % 192;
    for (int k = 0; k < 16; k++) ring[start + k] = rnd.NextDouble() * 10;
    using NDArray window = ring[$"{start}:{start + 16}"];
    Export("w", window);                                                         // rebinding "w" drops the previous export
    statsOk &= Math.Abs(py.Float("w.std()") - np.std(window).GetDouble()) < 1e-9;
    peak = Math.Max(peak, NDArrayPythonInterop.LiveExports);
}
check.That(statsOk, "every window's std agrees with NumSharp's");
py.Exec("del w");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == e0) && peak < e0 + 8, $"live exports never grew unbounded (peak {peak - e0} above the {e0} still bound) and drained back");

// ==== 4. A Python dataset outlives its Python references under NumSharp kernels =====================
check.Section("dataset hand-off: Python builds it, NumSharp keeps it alive and computes over it");
py.Exec("ds = {'x': np.arange(12, dtype='f8').reshape(3, 4) * 0.5, 'y': np.arange(3, dtype='i8') * 10}");
using NDArray x = View("ds['x']");
using NDArray y = View("ds['y']");
py.Exec("del ds\nimport gc; gc.collect()");                                      // only our leases keep the arrays now
check.That(np.sum(x).GetDouble() == 33.0 && np.mean(x).GetDouble() == 2.75 && np.sum(y).GetInt64() == 30, "sum/mean kernels run over leased Python memory: 33.0, 2.75, 30");
using NDArray scaled = x * 2.0;
check.That(scaled.GetDouble(2, 3) == 11.0 && x.GetDouble(2, 3) == 5.5, "elementwise kernels read the lease and write fresh NumSharp memory");

// ==== 5. Co-simulation: positions in C#, velocities in Python, both integrated in place =============
check.Section("co-simulation: mixed ownership stays coherent over 20 steps");
using NDArray pos = np.zeros(new Shape(8));
Export("pos", pos);
py.Exec("vel = np.arange(8, dtype='f8') - 3.5");
using NDArray vel = View("vel");
for (int step = 0; step < 20; step++)
{
    py.Exec("pos += vel * 0.1");                                                 // Python integrates C#-owned positions in place
    using NDArray damped = vel * 0.9;
    np.copyto(vel, damped);                                                      // C# damps Python-owned velocities in place
}
double decay = Math.Pow(0.9, 20);
bool coherent = true;
for (int i = 0; i < 8; i++)
{
    double v0 = i - 3.5;
    coherent &= Math.Abs(pos.GetDouble(i) - v0 * (1 - decay)) < 1e-9 && Math.Abs(py.Float($"vel[{i}]") - v0 * decay) < 1e-9;
}
check.That(coherent, "positions (C# reads its own buffer) and velocities (Python reads its own) match the closed form");

// ==== 6. A long-lived service through the codec: no explicit conversion calls ========================
check.Section("service loop: 40 calls through the codec, results decoded back as views");
py.Exec("def process(a):\n    return a * 2.0 + 1.0");
bool loopOk = true;
for (int i = 0; i < 40; i++)
{
    using NDArray batch = np.arange(6).astype(NPTypeCode.Double) + i;
    using (Py.GIL())
    {
        dynamic process = py.Get("process");
        using PyObject r = (PyObject)process(batch);                             // batch auto-encoded, r a fresh ndarray
        using NDArray result = r.As<NDArray>();                                  // auto-decoded as a view of r
        for (int j = 0; j < 6; j++) loopOk &= result.GetDouble(j) == (j + i) * 2.0 + 1.0;
    }
}
check.That(loopOk, "every call's result is correct");

// ==== 7. Generators stream chunks into NumSharp =====================================================
check.Section("generators: each yielded array is copied out and stacked");
py.Exec("def chunks(n):\n    for i in range(n):\n        yield np.arange(3, dtype='f8') + 10 * i");
var rows = new List<NDArray>();
using (Py.GIL())
{
    using PyObject generator = py.Module.Eval("chunks(4)");
    using var iterator = PyIter.GetIter(generator);
    while (iterator.MoveNext())
    {
        using PyObject chunk = iterator.Current;
        rows.Add(chunk.ToNDArray());
    }
}
using NDArray stacked = np.vstack(rows.ToArray());
check.That(stacked.shape.SequenceEqual(new long[] { 4, 3 }) && stacked.GetDouble(3, 2) == 32.0, "4 chunks -> (4,3), last element 30 + 2");
foreach (NDArray r in rows) r.Dispose();

// ==== 8. Exceptions, JSON, faulty inputs ============================================================
check.Section("Python exceptions surface as PythonException; the engine stays usable");
Export("ex", np.arange(3).astype(NPTypeCode.Double));
check.Throws<PythonException>(() => py.Exec("raise ValueError('bad shape: ' + str(ex.shape))"), "a raise inside Python", "bad shape: (3,)");
py.Exec("ex[0] = 7.5");
check.That(py.Float("ex.sum()") == 10.5, "the same shared array keeps working afterwards");

check.Section("JSON round trip through the stdlib");
using NDArray series = np.arange(5).astype(NPTypeCode.Double) * 1.5;
Export("jx", series);
py.Exec("import json\npayload = json.dumps(jx.tolist())");
check.That(py.Str("payload") == "[0.0, 1.5, 3.0, 4.5, 6.0]", $"json.dumps(jx.tolist()) == {py.Str("payload")}");
using NDArray back = Import("np.asarray(json.loads(payload), dtype='f8')");
check.That(back.GetDouble(4) == 6.0, "json.loads -> np.asarray -> ToNDArray is lossless");

check.Section("faulty inputs leave no half-taken state");
int e1 = NDArrayPythonInterop.LiveExports, i1 = NDArrayPythonInterop.LiveImports;
check.Throws<InvalidOperationException>(() => View("b'abcd'").Dispose(), "a read-only source without allowReadonly");
check.Throws<NotSupportedException>(() => Import("{'a': 1}").Dispose(), "a dict");
check.Throws<NotSupportedException>(() => Import("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])").Dispose(), "a structured dtype");
check.Throws<ObjectDisposedException>(() => { var dead = np.arange(3); dead.Dispose(); using (Py.GIL()) dead.ToNumpy().Dispose(); }, "a disposed NDArray");
check.That(NDArrayPythonInterop.LiveExports == e1 && NDArrayPythonInterop.LiveImports == i1, "no pin or lease survived a failed conversion");

check.Section("counters");
py.Exec("del batch, pred, img, crop, pos, vel, ex, jx");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 0 && NDArrayPythonInterop.LiveImports <= 3),
           $"pins released; the only leases left are the three views this script still holds (exports {NDArrayPythonInterop.LiveExports}, imports {NDArrayPythonInterop.LiveImports})");

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
