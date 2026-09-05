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
//      The third-party sections run only when the package is installed (see requirements.txt).
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example05_BuffersAndLibraries.cs.
//
//      Run:  dotnet run 05-buffers-and-libraries.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;
using (Py.GIL()) py.Exec("import array, ctypes, io, mmap, struct, hashlib, zlib, socket");

bool HasModule(string name) { using (Py.GIL()) return Throws(() => py.Exec($"import {name}")) is null; }

// ==== every builtin exporter is an NDArray =========================================================
// A PEP 3118 exporter of any kind decodes to a writable view over its own memory; a NumSharp write lands in Python:
//   bytearray(b'abcd')                    Byte      memoryview(bytearray(b'abcd'))     Byte      io.BytesIO(b'abcd').getbuffer()    Byte
//   array.array('d', [1.5, 2.5, 3.5])     Double    array.array('q', [1, 2, 3])        Int64     (ctypes.c_int32 * 4)(1, 2, 3, 4)   Int32
//   (ctypes.c_double * 2)(0.5, 1.5)       Double    mmap.mmap(-1, 8)                   Byte      np.arange(4, dtype='u2')           UInt16
foreach (string exporter in new[]
         {
             "bytearray(b'abcd')", "memoryview(bytearray(b'abcd'))", "io.BytesIO(b'abcd').getbuffer()",
             "array.array('d', [1.5, 2.5, 3.5])", "array.array('q', [1, 2, 3])", "(ctypes.c_int32 * 4)(1, 2, 3, 4)",
             "(ctypes.c_double * 2)(0.5, 1.5)", "mmap.mmap(-1, 8)", "np.arange(4, dtype='u2')",
         })
using (Py.GIL())
{
    py.Exec($"e = {exporter}");
    NDArray v = py.e;                                                  // the codec ran AsNDArray: a writable view of the exporter's memory
    v.SetValue(Convert.ChangeType(7, v.dtype), 0);                     // NumSharp writes 7 into element 0...
    long e0 = py.Eval("int(e[0])");                                    // ...and Python reads 7
}

// array.array: all twelve typecodes map by C type + itemsize
foreach (char code in "bBhHiIlLqQfd")
using (Py.GIL())
{
    py.Exec($"aa = array.array('{code}', [1, 2, 3])");
    NDArray v = py.aa;                                                 // v.dtypesize == aa.itemsize ('l' / 'L' resolve by the platform's itemsize); v[2] == 3
}

// read-only exporters: bytes
using (Py.GIL())
{
    using PyObject bs = py.Eval("b'hello'");
    Exception? refused = Throws(() => bs.AsNDArray());                 // InvalidOperationException: the exporter's buffer is read-only ... pass allowReadonly:true
    NDArray ro = bs.AsNDArray(allowReadonly: true);                    // a NON-writeable Byte view; ro[1] == 'e'
    NDArray owned = bs.ToNDArray();                                    // an owning, writable copy
    NDArray implicitly = py.Eval("b'hello'");                          // the codec's Auto mode takes the read-only view by itself
}

// shared memory across processes (multiprocessing.shared_memory)
using (Py.GIL())
{
    py.Exec("from multiprocessing import shared_memory\nshm = shared_memory.SharedMemory(create=True, size=32)");
    using (NDScope.Open())                                             // a nested scope: the view's lease must be gone before the block can be closed
    {
        NDArray view = py.shm.buf;                                     // a view straight into the shared block
        view[0] = (byte)42;                                            // shm.buf[0] reads 42 — visible to every process attached to the block
    }
    PythonHost.Drain();                                                // the lease release is marshaled to the GIL; flush it before closing the block
    py.Exec("shm.close(); shm.unlink()");
}

// ==== ToMemoryView: raw bytes for consumers that never touch numpy ==================================
NDArray pcm = np.arange(8).astype(NPTypeCode.Int16);
using (Py.GIL())
{
    py.smv = pcm.ToMemoryView();                                       // a writable memoryview of the raw bytes: smv.format == 'B', smv.nbytes == 16 — the buffer, not a numpy array
    py.Exec("first4 = struct.unpack_from('<4h', smv)");                // struct reads the raw bytes in place
    (long, long, long, long) first4 = py.first4;                       // (0, 1, 2, 3) — the tuple codec reads Python's tuple into a C# one
    py.Exec("digest = hashlib.sha256(smv).hexdigest()");               // hashlib consumes it in place (64 hex characters)
    py.Exec("packed = zlib.compress(smv)");                            // so does zlib
    py.Exec("s1, s2 = socket.socketpair(); s1.sendall(smv); got = s2.recv(64); s1.close(); s2.close()");   // a socket sends the bytes directly: got == bytes(smv)
    py.Exec("smv.cast('h')[0] = 77");                                  // pcm[0] reads 77: the memoryview writes back into NumSharp
    py.Exec("a = np.frombuffer(smv, '<i2').reshape(2, 4)");            // the recipe for reaching numpy-shaped APIs from raw bytes: a[0] == [77, 1, 2, 3], a.flags.owndata == False

    NDArray matrix = np.arange(24).reshape(4, 6).astype(np.float64);
    Exception? notContiguous = Throws(() => matrix.T.ToMemoryView());  // InvalidOperationException: not C-contiguous — a memoryview of raw bytes needs one run of bytes
    py.rmv = matrix["1:3"].ToMemoryView();                             // a contiguous row window at an offset exports exactly [offset, offset + size)
    long windowBytes = py.Eval("len(rmv)");                            // 96
    double windowStart = py.Eval("np.frombuffer(rmv, '<f8')[0]");      // 6.0
}

// ==== third-party consumers, verbatim ==============================================================
if (HasModule("PIL"))
using (Py.GIL())
{
    NDArray img = np.zeros(new Shape(4, 6, 3), NPTypeCode.Byte);
    img[2, 3] = (NDArray)new byte[] { 10, 20, 30 };
    py.pmv = img.ToMemoryView();
    py.Exec("from PIL import Image\nim = Image.frombuffer('RGB', (6, 4), pmv, 'raw', 'RGB', 0, 1)");   // Image.frombuffer reads NumSharp's bytes in place
    (long, long, long) pixel = py.im.getpixel((3, 2));                 // (10, 20, 30) — a C# tuple crossed as the (x, y) argument, a Python tuple came back
    // The reverse: a PIL Image exports no buffer and its __array_interface__ data is a bytes object, so it cannot be VIEWED —
    using PyObject im = py.im;
    Exception? noAddress = Throws(() => im.AsNDArray(allowReadonly: true));   // NotSupportedException: ... (pointer, readonly) tuple — the Image names no address to share
    NDArray pixels = py.np.asarray(im);                                // np.asarray(im) first: a read-only (4,6,3) uint8 view; pixels[2, 3, 2] == 30
}

if (HasModule("pyarrow"))
using (Py.GIL())
{
    NDArray col = np.arange(5).astype(np.float64);
    py.amv = col.ToMemoryView();
    py.Exec("import pyarrow as pa\nabuf = pa.py_buffer(amv)\naarr = pa.Array.from_buffers(pa.float64(), 5, [None, abuf])");
    bool onNumSharpBytes = py.Eval("aarr.to_numpy(zero_copy_only=True).ctypes.data == abuf.address");   // True: pa.py_buffer wraps the memoryview zero-copy
    // And back: an Arrow buffer is itself a PEP 3118 exporter.
    py.Exec("arrow_native = pa.array([1.5, 2.5, 3.5], type=pa.float64())");
    NDArray raw = py.Eval("arrow_native.buffers()[1]");                // raw bytes — pyarrow exports format 'b' (SByte), and an allocation may be padded to a 64-byte multiple
    NDArray asDoubles = raw.view(typeof(double));                      // reinterpret with nd.view (or np.frombuffer): asDoubles[1] == 2.5, asDoubles[2] == 3.5
}

if (HasModule("torch"))
using (Py.GIL())
{
    NDArray f32 = np.arange(6).astype(np.float32);
    py.tmv = f32.ToMemoryView();
    py.Exec("import torch\nt = torch.frombuffer(tmv, dtype=torch.float32)\nt[0] = 42.0");   // a tensor over the same bytes: f32[0] reads 42 (09-pytorch.cs covers the tensor bridge itself)
    bool sameAddress = py.Eval("t.data_ptr() == torch.from_numpy(np.frombuffer(tmv, '<f4')).data_ptr()");   // True
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every view above (the leases on the exporters go back); `host` drops
// the namespace, so the memoryviews die with their names and the export pins with them; then the engine shuts
// down. LiveExports / LiveImports both read 0 afterwards.

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
