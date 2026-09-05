#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 03 — Layouts. Exporting keeps every NumSharp layout's strides and flags intact (never a copy,
//      never a flattening); importing takes one of three zero-copy routes chosen by what the
//      exporter is. This script prints what numpy sees for each layout class and proves each
//      import route is a real view by crossing a write through it.
//
//      Run:  dotnet run 03-layouts.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

void Export(string name, NDArray nd)
{
    using (Py.GIL()) { using PyObject p = nd.ToNumpy(); py.Set(name, p); }
}

// ==== NumSharp -> numpy: every layout class, strides intact ========================================
check.Section("export: what numpy sees per NumSharp layout");
using NDArray b = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);   // C-contiguous (4,6) float64

Export("l_c", b);
check.That(py.Str("l_c.strides") == "(48, 8)" && py.Bool("l_c.flags.c_contiguous"), $"C-contiguous (4,6): strides {py.Str("l_c.strides")}, C_CONTIGUOUS");

Export("l_rows", b["1:3"]);                                                  // row slice: contiguous window at an offset
check.That(py.Str("l_rows.strides") == "(48, 8)" && py.Bool("np.shares_memory(l_rows, l_c)"), $"row slice b[\"1:3\"]: strides {py.Str("l_rows.strides")}, shares memory with the full export");

Export("l_cols", b["1:3, 0:6:2"]);                                           // column slice: strided in both axes
check.That(py.Str("l_cols.strides") == "(48, 16)", $"column slice b[\"1:3, 0:6:2\"]: strides {py.Str("l_cols.strides")}");
check.That(py.Str("l_cols.tolist()") == "[[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]", $"  tolist() == {py.Str("l_cols.tolist()")}");

Export("l_t", b.T);                                                          // transpose: F-contiguous
check.That(py.Str("l_t.strides") == "(8, 48)" && py.Bool("l_t.flags.f_contiguous"), $"transpose b.T: strides {py.Str("l_t.strides")}, F_CONTIGUOUS");

Export("l_r", b["::-1"]);                                                    // reversed: NEGATIVE stride survives
check.That(py.Long("l_r.strides[0]") == -48, $"reversed b[\"::-1\"]: strides {py.Str("l_r.strides")} — negative strides survive");
check.That(py.Str("l_r[0].tolist()") == "[18.0, 19.0, 20.0, 21.0, 22.0, 23.0]", $"  l_r[0] == {py.Str("l_r[0].tolist()")}");

Export("l_f", np.asfortranarray(b));                                         // Fortran order
check.That(py.Str("l_f.strides") == "(8, 32)" && py.Bool("l_f.flags.f_contiguous") && !py.Bool("l_f.flags.c_contiguous"),
           $"Fortran order: strides {py.Str("l_f.strides")}, F_CONTIGUOUS and not C_CONTIGUOUS");

using NDArray bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Double), new Shape(2, 3));
Export("l_b", bc);                                                           // broadcast: stride 0, READ-ONLY on both sides
check.That(py.Str("l_b.strides") == "(0, 8)" && !py.Bool("l_b.flags.writeable"), $"broadcast (3,)->(2,3): strides {py.Str("l_b.strides")}, WRITEABLE=False (as in NumSharp)");
check.Throws<PythonException>(() => py.Exec("l_b[0, 0] = 1.0"), "numpy refuses to write through the broadcast view", "read-only");

using NDArray scalar = np.array(2.5);
Export("l_0", scalar);                                                       // 0-d: shape ()
py.Exec("l_0[()] = 7.5");
check.That(py.Long("l_0.ndim") == 0 && scalar.GetDouble() == 7.5, "0-d scalar: shape (), l_0[()] = 7.5 writes through");

using NDArray empty = np.zeros(new Shape(0, 3), NPTypeCode.Double);
Export("l_e", empty);                                                        // empty: nothing to share, a fresh array
check.That(py.Str("l_e.shape") == "(0, 3)" && py.Str("l_e.dtype") == "float64", "empty (0,3): shape and dtype survive; no bytes, so no pin");

check.Section("export: what the numpy array is made of");
check.That(py.Str("type(l_c.base).__name__") == "ndarray" && py.Str("type(l_c.base.base).__name__") == "c_char_Array_192",
           $"contiguous: reshape -> frombuffer -> ctypes window ({py.Str("type(l_c.base.base).__name__")}, 24 x 8 bytes)");
check.That(py.Str("type(l_cols.base).__name__") == "DummyArray", "strided: np.lib.stride_tricks.as_strided over the same window (base is DummyArray)");
check.That(py.Str("l_c.flags['OWNDATA']") == "False", "OWNDATA=False everywhere — numpy agrees the memory is NumSharp's");

// ==== numpy -> NumSharp: three zero-copy routes ======================================================
check.Section("import route 1: C-contiguous PEP 3118 exporter (any object)");
py.Exec("r1 = bytearray(b'abcd')");
NDArray v1;
using (Py.GIL()) { using PyObject r1 = py.Get("r1"); v1 = r1.AsNDArray(); }
v1[0] = (byte)90;
check.That(py.Long("r1[0]") == 90, "bytearray: PyBUF.WRITABLE lease; the NumSharp write lands in Python (r1[0] == 90)");
check.Throws<PythonException>(() => py.Exec("r1.append(1)"), "while leased, CPython refuses to resize the bytearray", "Existing exports of data");
v1.Dispose();

check.Section("import route 2: non-contiguous numpy arrays via __array_interface__");
py.Exec("r2 = np.arange(20, dtype='i8')");
NDArray v2;
using (Py.GIL()) { using PyObject s = py.Module.Eval("r2[::2]"); v2 = s.AsNDArray(); }
check.That(v2.size == 10 && v2.Shape.Strides[0] == 2, $"r2[::2]: size {v2.size}, element stride {v2.Shape.Strides[0]} — the layout is reproduced, not copied");
v2[1] = -3L;
check.That(py.Long("r2[2]") == -3, "the write lands at r2[2] (logical index 1 of the strided view)");
v2.Dispose();

NDArray vt, vf, vr, vb, v0;
using (Py.GIL())
{
    using PyObject t = py.Module.Eval("np.arange(12, dtype='f8').reshape(3, 4).T"); vt = t.AsNDArray();
    using PyObject f = py.Module.Eval("np.asfortranarray(np.arange(12, dtype='f8').reshape(3, 4))"); vf = f.AsNDArray();
    using PyObject r = py.Module.Eval("np.arange(6, dtype='f8')[::-1]"); vr = r.AsNDArray();
    using PyObject bb = py.Module.Eval("np.broadcast_to(np.arange(3, dtype='f8'), (2, 3))"); vb = bb.AsNDArray(allowReadonly: true);
    using PyObject z = py.Module.Eval("np.array(2.5)"); v0 = z.AsNDArray();      // (np.float64(2.5).reshape(()) would be READ-ONLY: a numpy scalar's buffer)
    py.Set("z0", z);
}
v0.SetValue(9.5);
check.That(vt.shape.SequenceEqual(new long[] { 4, 3 }) && vt.Shape.Strides.SequenceEqual(new long[] { 1, 4 }), $"transpose: shape ({string.Join(",", vt.shape)}), element strides ({string.Join(",", vt.Shape.Strides)})");
check.That(vf.Shape.IsFContiguous && !vf.Shape.IsContiguous, "F-order numpy array: IsFContiguous, not C-contiguous");
check.That(vr.Shape.Strides[0] == -1 && vr.GetDouble(0) == 5.0, $"reversed numpy [::-1]: element stride {vr.Shape.Strides[0]}, first element 5.0 (the window is normalized below numpy's data pointer)");
check.That(vb.Shape.IsBroadcasted && !vb.Shape.IsWriteable, "broadcast numpy source: IsBroadcasted and NON-writeable (stride 0 must never be written blindly)");
check.That(v0.ndim == 0 && py.Float("z0") == 9.5, "0-d numpy array: a 0-d NumSharp view; v0[()] = 9.5 reaches numpy");
foreach (NDArray v in new[] { vt, vf, vr, vb, v0 }) v.Dispose();

check.Section("import route 3: non-contiguous NON-numpy exporters via PyBUF.STRIDED");
py.Exec("r3 = bytearray(range(16))");
NDArray v3, v3r;
using (Py.GIL())
{
    using PyObject s = py.Module.Eval("memoryview(r3)[::2]"); v3 = s.AsNDArray();          // strided memoryview, no __array_interface__
    using PyObject rr = py.Module.Eval("memoryview(r3)[::-1]"); v3r = rr.AsNDArray();       // reversed memoryview: negative stride
}
check.That(v3.size == 8 && v3.Shape.Strides[0] == 2, $"memoryview(bytearray)[::2]: size {v3.size}, stride {v3.Shape.Strides[0]} — the buffer protocol hands over the pointer, the memoryview the strides");
v3[3] = (byte)77;
check.That(py.Long("r3[6]") == 77, "the write lands at r3[6]");
check.That(v3r.Shape.Strides[0] == -1 && v3r.GetByte(0) == 15, "memoryview(bytearray)[::-1]: negative stride reconstructed, element 0 is byte 15");
v3.Dispose(); v3r.Dispose();

check.Section("the only layouts that cannot be viewed decline with guidance (Auto codec copies them instead)");
using (Py.GIL())
{
    using PyObject sub = py.Module.Eval("np.lib.stride_tricks.as_strided(np.arange(4, dtype='i4'), shape=(2,), strides=(2,))");
    check.Throws<NotSupportedException>(() => sub.AsNDArray().Dispose(), "a sub-element (2-byte) stride over int32", "not a multiple of itemsize");
    using NDArray linearized = sub.ToNDArray();
    check.That(linearized.size == 2 && linearized.typecode == NPTypeCode.Int32, "...but ToNDArray linearizes it through memoryview.tobytes('C')");
}

check.Section("counters");
py.Exec("del l_c, l_rows, l_cols, l_t, l_r, l_f, l_b, l_0, l_e, z0");
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
