#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 05 — Buffers and libraries. The import side is numpy-AGNOSTIC: anything exporting a PEP 3118
//      buffer (bytes, bytearray, memoryview, array.array, ctypes arrays, BytesIO.getbuffer(), mmap,
//      shared memory, PyArrow buffers, ...) becomes an NDArray. The export side offers ToMemoryView:
//      a writable Python memoryview of raw bytes for consumers that never touch numpy (struct,
//      hashlib, zlib, sockets, PIL.Image.frombuffer, pa.py_buffer, torch.frombuffer).
//      Third-party sections self-skip when the package is not installed (see requirements.txt).
//
//      Run:  dotnet run 05-buffers-and-libraries.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope("import numpy as np, array, ctypes, io, mmap, struct, hashlib, zlib, socket");
var check = new Checklist();

bool Installed(string module)
{
    try { py.Exec($"import {module}"); return true; }
    catch (PythonException) { Console.WriteLine($"  (skip) python package '{module}' is not installed"); return false; }
}

// ==== every builtin exporter is an NDArray =========================================================
check.Section("PEP 3118 exporters -> NDArray views (write-through proves the sharing)");
var exporters = new (string Expr, string Name, NPTypeCode Tc, string Probe)[]
{
    ("bytearray(b'abcd')",                       "bytearray",                   NPTypeCode.Byte,   "e[0]"),
    ("memoryview(bytearray(b'abcd'))",           "memoryview",                  NPTypeCode.Byte,   "e[0]"),
    ("io.BytesIO(b'abcd').getbuffer()",          "BytesIO.getbuffer()",         NPTypeCode.Byte,   "e[0]"),
    ("array.array('d', [1.5, 2.5, 3.5])",        "array.array('d')",            NPTypeCode.Double, "e[0]"),
    ("array.array('q', [1, 2, 3])",              "array.array('q')",            NPTypeCode.Int64,  "e[0]"),
    ("(ctypes.c_int32 * 4)(1, 2, 3, 4)",         "ctypes c_int32[4]",           NPTypeCode.Int32,  "e[0]"),
    ("(ctypes.c_double * 2)(0.5, 1.5)",          "ctypes c_double[2]",          NPTypeCode.Double, "e[0]"),
    ("mmap.mmap(-1, 8)",                         "anonymous mmap",              NPTypeCode.Byte,   "e[0]"),
    ("np.arange(4, dtype='u2')",                 "numpy uint16",                NPTypeCode.UInt16, "e[0]"),
};
foreach (var (expr, name, tc, probe) in exporters)
{
    py.Exec($"e = {expr}");
    NDArray v;
    using (Py.GIL()) { using PyObject e = py.Get("e"); v = e.AsNDArray(); }
    bool ok = v.typecode == tc && v.Shape.IsWriteable;
    v.SetValue(Convert.ChangeType(7, v.dtype), 0);                     // write through the view...
    ok &= py.Long(probe) == 7;                                           // ...and Python sees it
    v.Dispose();
    check.That(ok, $"{name,-24} -> {tc,-7} view; NumSharp writes 7, Python reads {probe} == 7");
}
py.Exec("del e");

check.Section("array.array: all twelve typecodes map by C type + itemsize");
bool allCodes = true;
foreach (char code in "bBhHiIlLqQfd")
{
    py.Exec($"aa = array.array('{code}', [1, 2, 3])");
    using (Py.GIL())
    {
        using PyObject aa = py.Get("aa");
        using NDArray v = aa.AsNDArray();
        long itemsize = py.Long("aa.itemsize");
        allCodes &= v.size == 3 && v.dtypesize == itemsize && Convert.ToDouble(v.GetValue(2)) == 3;
    }
}
check.That(allCodes, "'b','B','h','H','i','I','l','L','q','Q','f','d' all view (l/L resolve by the platform's itemsize)");

check.Section("read-only exporters: bytes");
using (Py.GIL())
{
    using PyObject bs = py.Module.Eval("b'hello'");
    check.Throws<InvalidOperationException>(() => bs.AsNDArray().Dispose(), "bytes refused by default", "read-only");
    using NDArray ro = bs.AsNDArray(allowReadonly: true);
    check.That(!ro.Shape.IsWriteable && ro.GetByte(1) == (byte)'e', "allowReadonly:true -> non-writeable Byte view (ro[1] == 'e')");
    using NDArray owned = bs.ToNDArray();
    check.That(owned.Shape.IsWriteable, "ToNDArray -> writable owning copy");
}

check.Section("shared memory across processes (multiprocessing.shared_memory)");
py.Exec("from multiprocessing import shared_memory\nshm = shared_memory.SharedMemory(create=True, size=32)");
NDArray shmView;
using (Py.GIL()) { using PyObject buf = py.Module.Eval("shm.buf"); shmView = buf.AsNDArray(); }
shmView[0] = (byte)42;
check.That(py.Long("shm.buf[0]") == 42, "a NumSharp view over a SharedMemory block writes straight into it");
shmView.Dispose();
Settle(() => NDArrayPythonInterop.LiveImports == 0);   // the lease must go before the block can be closed
py.Exec("shm.close(); shm.unlink()");

// ==== ToMemoryView: raw bytes for consumers that never touch numpy ==================================
check.Section("ToMemoryView: raw writable bytes (contiguous arrays only)");
using NDArray pcm = np.arange(8).astype(NPTypeCode.Int16);
using (Py.GIL()) { using PyObject mv = pcm.ToMemoryView(); py.Set("smv", mv); }
check.That(py.Str("smv.format") == "B" && py.Long("smv.nbytes") == 16, $"format 'B', nbytes {py.Long("smv.nbytes")} — the buffer, not a numpy array");
py.Exec("first4 = struct.unpack_from('<4h', smv)\ndigest = hashlib.sha256(smv).hexdigest()\npacked = zlib.compress(smv)");
check.That(py.Str("first4") == "(0, 1, 2, 3)", $"struct.unpack_from reads the raw bytes: {py.Str("first4")}");
check.That(py.Long("len(digest)") == 64 && py.Long("len(packed)") > 0, "hashlib and zlib consume it in place");
py.Exec("s1, s2 = socket.socketpair(); s1.sendall(smv); got = s2.recv(64); s1.close(); s2.close()");
check.That(py.Bool("got == bytes(smv)"), "a socket sends the memoryview's bytes directly");
py.Exec("smv.cast('h')[0] = 77");
check.That(pcm.GetValue(0) is short and 77, "memoryview.cast('h')[0] = 77 wrote back into NumSharp (pcm[0] == 77)");
py.Exec("a = np.frombuffer(smv, '<i2').reshape(2, 4)");                                                          // the recipe for any numpy consumer
check.That(py.Str("a[0].tolist()") == "[77, 1, 2, 3]" && py.Str("a.flags['OWNDATA']") == "False", "np.frombuffer(mv, dtype) is the recipe for reaching numpy-shaped APIs from raw bytes");
using NDArray sliced = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);
check.Throws<InvalidOperationException>(() => { using (Py.GIL()) sliced.T.ToMemoryView().Dispose(); }, "a non-contiguous array (transpose) refuses", "not C-contiguous");
using (Py.GIL())
{
    using PyObject rows = sliced["1:3"].ToMemoryView();                // contiguous window at an offset: exactly that window
    py.Set("rmv", rows);
}
check.That(py.Long("len(rmv)") == 96 && py.Str("np.frombuffer(rmv, '<f8')[0]") == "6.0", "a contiguous row window exports exactly [offset, offset+size): 96 bytes starting at 6.0");
py.Exec("del smv, rmv, a, packed");

// ==== third-party consumers, verbatim ==============================================================
if (Installed("PIL"))
{
    check.Section("Pillow");
    using NDArray img = np.zeros(new Shape(4, 6, 3), NPTypeCode.Byte);
    img[2, 3] = (NDArray)new byte[] { 10, 20, 30 };
    using (Py.GIL()) { using PyObject mv = img.ToMemoryView(); py.Set("pmv", mv); }
    py.Exec("from PIL import Image\nim = Image.frombuffer('RGB', (6, 4), pmv, 'raw', 'RGB', 0, 1)");
    check.That(py.Str("im.getpixel((3, 2))") == "(10, 20, 30)", $"Image.frombuffer reads NumSharp's bytes in place: getpixel((3,2)) == {py.Str("im.getpixel((3, 2))")}");
    // The reverse: a PIL Image exports no buffer and its __array_interface__ data is a bytes object,
    // so it cannot be VIEWED — np.asarray(im) first, and that array imports as a read-only view.
    using (Py.GIL())
    {
        using PyObject im = py.Get("im");
        check.Throws<NotSupportedException>(() => im.AsNDArray(allowReadonly: true).Dispose(), "an Image itself names no address to share", "(pointer, readonly) tuple");
        using PyObject asArray = py.Module.Eval("np.asarray(im)");
        using NDArray pixels = asArray.AsNDArray(allowReadonly: true);
        check.That(pixels.shape.SequenceEqual(new long[] { 4, 6, 3 }) && pixels.GetByte(2, 3, 2) == 30 && !pixels.Shape.IsWriteable, "np.asarray(im) -> read-only (4,6,3) uint8 view");
    }
    py.Exec("del im, pmv");
}

if (Installed("pyarrow"))
{
    check.Section("PyArrow");
    using NDArray col = np.arange(5).astype(NPTypeCode.Double);
    using (Py.GIL()) { using PyObject mv = col.ToMemoryView(); py.Set("amv", mv); }
    py.Exec("import pyarrow as pa\nabuf = pa.py_buffer(amv)\naarr = pa.Array.from_buffers(pa.float64(), 5, [None, abuf])");
    check.That(py.Bool("aarr.to_numpy(zero_copy_only=True).ctypes.data == abuf.address"), "pa.py_buffer wraps the memoryview zero-copy; the Arrow array sits on NumSharp's bytes");
    // And back: an Arrow buffer is itself a PEP 3118 exporter.
    py.Exec("arrow_native = pa.array([1.5, 2.5, 3.5], type=pa.float64())");
    using (Py.GIL())
    {
        using PyObject values = py.Module.Eval("arrow_native.buffers()[1]");
        using NDArray v = values.AsNDArray(allowReadonly: true);
        check.That(v.dtypesize == 1 && v.size >= 24 && v.size % 8 == 0,
                   $"an Arrow buffer imports as raw bytes ({v.typecode}, {v.size} B — pyarrow exports format 'b'; allocations may be padded to 64-byte multiples) — reinterpret with nd.view or np.frombuffer");
        using NDArray asDoubles = v.view(typeof(double));
        check.That(asDoubles.GetDouble(1) == 2.5 && asDoubles.GetDouble(2) == 3.5, "...nd.view(typeof(double)) reads the float64 values");
    }
    py.Exec("del aarr, abuf, amv, arrow_native");
}

if (Installed("torch"))
{
    check.Section("torch.frombuffer (the buffer route; 09-pytorch.cs covers the tensor bridge)");
    using NDArray f32 = np.arange(6).astype(NPTypeCode.Single);
    using (Py.GIL()) { using PyObject mv = f32.ToMemoryView(); py.Set("tmv", mv); }
    py.Exec("import torch\nt = torch.frombuffer(tmv, dtype=torch.float32)\nt[0] = 42.0");
    check.That(f32.GetSingle(0) == 42f, "torch.frombuffer(mv, dtype=torch.float32): a tensor write lands in the NDArray");
    check.That(py.Bool("t.data_ptr() == torch.from_numpy(np.frombuffer(tmv, '<f4')).data_ptr()"), "torch.frombuffer and np.frombuffer reach the identical address");
    py.Exec("del t, tmv");
}

check.Section("counters");
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
