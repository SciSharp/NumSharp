#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 06 — Lifetime: two collectors, one buffer. The rule in both directions is the same — memory is
//      released when the LAST reference on the far side lets go, never when the near side tidies up
//      first. Exports take their own reference on the NumSharp buffer and hand the release to a
//      weakref.finalize on the numpy array's base object; imports lease the Python buffer through
//      NumSharp's memory-block refcount. Two counters make every live crossing observable, a live
//      view locks reallocation on both sides, and engine shutdown drains everything crash-free.
//
//      NDScope is the other half of the story here: a scope releases every NDArray built inside it
//      when it closes, so "the near side let go" is one `using` block instead of a Dispose per array.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example06_Lifetime.cs.
//
//      Run:  dotnet run 06-lifetime.cs

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

var host = PythonHost.Start();                       // disposed EXPLICITLY below — the last section watches the shutdown sweep
using var scope = NDScope.Open();
dynamic py = host.Namespace;
using (Py.GIL()) py.Exec("import weakref, gc");

// A note on `dynamic` in THIS example: every dynamic read (`py.c2`) hands out a CLR wrapper holding its own
// Python reference, released only when the garbage collector gets to it. That is fine everywhere else; here,
// where reference counts are the subject, the Python objects are read as `using PyObject` so the wrapper's
// reference ends deterministically and only the interop's own pins and leases remain in the picture.

// ==== the counters: LiveExports / LiveImports move by exactly one per crossing =========================
int exports0 = NDArrayPythonInterop.LiveExports, imports0 = NDArrayPythonInterop.LiveImports;   // 0, 0
NDArray nd = np.arange(4).astype(np.float64);
NDArray leased;
using (Py.GIL())
{
    py.c1 = nd;                                      // one export:      LiveExports == 1
    py.Exec("c2 = np.arange(3, dtype='i8')");
    using PyObject c2 = py.c2;
    leased = c2.AsNDArray();                         // one import view: LiveImports == 1
}
leased.Dispose();                                    // deterministic: Dispose releases the lease without waiting for a GC...
PythonHost.Drain();                                  // ...the release itself is marshaled to the GIL — LiveImports == 0 after the drain
using (Py.GIL()) py.Exec("del c1");                  // the last numpy name dies -> the pin is released: LiveExports == 0

// ==== an export outlives every managed reference ====================================================
using (Py.GIL())
using (NDScope.Open())                               // an inner scope: everything built in this block is released when it ends
{
    NDArray temp = np.arange(4).astype(np.float64);
    py.orphan = temp;                                // Python's name is now the only holder of the export
}                                                    // temp is disposed here — the NDArray AND its own reference on the buffer are gone
PythonHost.Drain();
int stillPinned = NDArrayPythonInterop.LiveExports;  // 1: the pin is held for Python
using (Py.GIL())
{
    string values = py.Eval("str(orphan.tolist())"); // [0.0, 1.0, 2.0, 3.0] — Python still reads valid memory...
    py.Exec("orphan[0] = 7.0");                      // ...and still writes it
    py.Exec("child = orphan[2:]; del orphan");       // a DERIVED numpy view chains to the same base object
}
PythonHost.Drain();
int derivedKeepsIt = NDArrayPythonInterop.LiveExports;   // 1: child alone keeps the buffer rooted; child.tolist() == [2.0, 3.0]
using (Py.GIL()) py.Exec("del child");
PythonHost.Drain();
int released = NDArrayPythonInterop.LiveExports;     // 0: the last derived view released the pin

// ==== an import lease outlives every Python reference ===============================================
NDArray slice;
using (Py.GIL())
{
    py.Exec("keep = np.arange(6, dtype='f8') * 1.5\nwr = weakref.ref(keep)");
    using (var inner = NDScope.Open())
    {
        using PyObject keep = py.keep;
        NDArray whole = keep.AsNDArray();            // the import view: a lease on numpy's buffer
        slice = inner.Returns(whole["2:5"]);         // yielded to the outer scope; `whole` itself is released when `inner` closes
    }
    py.Exec("del keep; gc.collect()");               // Python forgot the array...
}
PythonHost.Drain();
int leaseHeld = NDArrayPythonInterop.LiveImports;    // 1: the slice alone keeps the lease (disposal ORDER is irrelevant; the refcount decides)
double s0 = slice.GetDouble(0), s2 = slice.GetDouble(2);   // 3.0, 6.0 — the slice reads valid memory
bool exporterAlive;
using (Py.GIL()) exporterAlive = py.Eval("wr() is not None");   // true: NumSharp holds the exporter alive
slice.Dispose();                                     // the last view over the lease...
PythonHost.Drain();
int leaseReleased = NDArrayPythonInterop.LiveImports;   // 0: ...releases it, and Python frees the exporter (wr() is None now)

// ==== what a live view forbids ========================================================================
using (Py.GIL())
{
    py.Exec("ba = bytearray(b'abcd')\nnpa = np.arange(4, dtype='f8')");
    using (NDScope.Open())
    {
        using PyObject ba = py.ba, npa = py.npa;
        NDArray held = ba.AsNDArray(), held2 = npa.AsNDArray();
        Exception? bufferError = Throws(() => py.Exec("ba.append(1)"));                     // PythonException (BufferError): Existing exports of data: object cannot be re-sized
        Exception? refcheck = Throws(() => py.Exec("npa.resize((8,), refcheck=True)"));     // PythonException (ValueError): cannot resize an array that references or is referenced
    }
    PythonHost.Drain();                              // both leases are gone...
    py.Exec("ba.append(1)\nnpa.resize((8,), refcheck=True)");   // ...and both locks lift: len(ba) == 5, npa.size == 8

    NDArray pinned = np.arange(8).astype(np.float64);
    py.pin = pinned;
    Exception? mirror = Throws(() => pinned.resize(new Shape(16)));   // the mirror image — "cannot resize an array that references or is referenced" while Python holds a view
    py.Exec("del pin");
    PythonHost.Drain();
    pinned.resize(new Shape(16));                    // succeeds once the pin is gone: pinned.size == 16
}

// ==== churn: everything returns to baseline ===========================================================
for (int i = 0; i < 120; i++)
using (NDScope.Open())                               // one scope per round: the round's arrays are released as the round ends
using (Py.GIL())
{
    NDArray a = np.arange(32).astype(np.float64) + i;
    py.h = a;                                        // an export (rebinding `h` drops the previous one)
    py.hc = a.ToNumpyCopy();                         // a copy export
    py.Exec("h2 = np.arange(16, dtype='i8')");
    NDArray view = py.h2;                            // an import view
    NDArray sub = view["4:12"];                      // a derived slice over the lease
    sub[0] = (long)i;
}
using (Py.GIL()) py.Exec("del h, hc, h2");
PythonHost.Drain();
int exportsAfterChurn = NDArrayPythonInterop.LiveExports, importsAfterChurn = NDArrayPythonInterop.LiveImports;   // 0, 0

// ==== engine shutdown ================================================================================
// Leave one export held ONLY by Python and one import still referenced by C# when the engine dies.
NDArray survivor;
using (Py.GIL())
{
    using (NDScope.Open())
    {
        NDArray orphanSource = np.arange(6).astype(np.float64);
        py.shutdown_orphan = orphanSource;           // Python becomes the only holder of that buffer when the inner scope closes
    }
    py.Exec("shutdown_import = np.arange(8) * 2.0");
    using PyObject s = py.shutdown_import;
    survivor = s.AsNDArray();
    NDScope.Detach(survivor);                        // keep it out of the outer scope: it must still be referenced when the engine dies
}
int exportsGoingIn = NDArrayPythonInterop.LiveExports, importsGoingIn = NDArrayPythonInterop.LiveImports;   // 1, 1

host.Dispose();                                      // RuntimeData.FormatterType = NoopFormatter; PythonEngine.Shutdown() — the namespace (and the orphan's only holder) dies WITH the interpreter
bool down = !PythonEngine.IsInitialized;             // true: Shutdown() completed without a crash
int importsAfter = NDArrayPythonInterop.LiveImports; // 0: the shutdown handler force-drained every import lease
survivor.Dispose();                                  // a harmless no-op now — the lease was already claimed
// ...and within a moment LiveExports reads 0 too: the orphaned export is swept right after the engine dies (pythonnet runs
// no atexit pass, so weakref.finalize could never fire; the interop's own sweep waits until the interpreter is provably gone).

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
