#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 09 — PyTorch. The built-in TorchPythonArrayAdapter makes torch.Tensor a front door to the SAME
//      memory bridge: nd.ToTorch() is torch.from_numpy(nd.ToNumpy()), tensor.AsNDArray() is the
//      ordinary view verb over Tensor.numpy(), tensor.ToNDArray() the ordinary copy verb, and the
//      Torch-specific aliases add the `force` policy (detach + CPU transfer + resolve_conj/neg).
//      No TorchSharp, no compile-time torch dependency — torch is imported from the embedded CPython.
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example09_PyTorch.cs.
//
//      Run:  dotnet run 09-pytorch.cs        (exits quietly when torch is not installed)

using System.Diagnostics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

bool HasModule(string name) { using (Py.GIL()) return Throws(() => py.Exec($"import {name}")) is null; }
if (!HasModule("torch")) return;                              // see requirements.txt — any PyTorch >= 2.3

// ==== NumSharp -> PyTorch ============================================================================
NDArray matrix = np.arange(12).astype(np.float64).reshape(3, 4);
using (Py.GIL())
{
    py.t = matrix.ToTorch();                                  // torch.from_numpy(matrix.ToNumpy()): a zero-copy CPU tensor over NumSharp's buffer
    py.v = matrix;                                            // the numpy view of the same buffer, for comparison
    bool sameAddress = py.Eval("t.data_ptr() == v.ctypes.data");   // True: the tensor's data_ptr() IS NumSharp's buffer address
    py.Exec("t[1, 2] = 91.5");                                // matrix[1, 2] reads 91.5
    matrix[2, 1] = -7.25;                                     // t[2, 1].item() reads -7.25

    py.ts = matrix[":, ::2"].ToTorch();                       // a strided view keeps its layout
    (long, long) shape = py.ts.shape;                         // (3, 2)  — torch.Size is a tuple subclass, so the tuple codec reads it
    (long, long) stride = py.ts.stride();                     // (4, 2)
    py.Exec("ts.add_(1000)");                                 // in-place, columns 0 and 2 only: matrix[0] reads [1000, 1, 1002, 3]
}

// every NumSharp dtype maps to a torch dtype (Decimal converts to float64)
using (Py.GIL())
    foreach (var (tc, torchDtype) in new[]
             {
                 (NPTypeCode.Boolean, "torch.bool"), (NPTypeCode.Byte, "torch.uint8"), (NPTypeCode.SByte, "torch.int8"),
                 (NPTypeCode.Int16, "torch.int16"), (NPTypeCode.UInt16, "torch.uint16"), (NPTypeCode.Int32, "torch.int32"),
                 (NPTypeCode.UInt32, "torch.uint32"), (NPTypeCode.Int64, "torch.int64"), (NPTypeCode.UInt64, "torch.uint64"),
                 (NPTypeCode.Char, "torch.uint16"), (NPTypeCode.Half, "torch.float16"), (NPTypeCode.Single, "torch.float32"),
                 (NPTypeCode.Double, "torch.float64"), (NPTypeCode.Decimal, "torch.float64"), (NPTypeCode.Complex, "torch.complex128"),
             })
    {
        NDArray src = np.arange(3).astype(tc);
        dynamic tensor = src.ToTorch();
        string dtype = tensor.dtype.ToString();               // == torchDtype, 15 / 15
    }

// unsafe layouts need copy: true
NDArray reversed = matrix["::-1"];
NDArray broadcast = np.broadcast_to(matrix[0], new Shape(2, 4));
using (Py.GIL())
{
    Exception? negativeStrides = Throws(() => reversed.ToTorch());   // NotSupportedException: PyTorch tensors cannot express negative strides ... copy:true
    Exception? notWriteable = Throws(() => broadcast.ToTorch());     // InvalidOperationException: a non-writeable (broadcast) array — writing through it would be undefined behavior ... copy:true
    py.rc = reversed.ToTorch(copy: true);                     // materializes the logical values: rc[0] == v[2], rc[2] == v[0]
    py.bc = broadcast.ToTorch(copy: true);                    // tuple(bc.shape) == (2, 4)
    py.Exec("rc[0, 0] = -1.0; bc[0, 0] = -2.0");              // ...and detaches: matrix is untouched
    // a copy holds no export pin — only the live views (t, v, ts) do: LiveExports == 3 after a drain
}

// ==== PyTorch -> NumSharp ============================================================================
using (Py.GIL())
{
    py.Exec("src = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]");
    NDArray tv = py.src;                                      // the adapter hands Tensor.numpy() to the ordinary view bridge (== src.AsNDArray() == src.AsTorchNDArray()): shape (3,2), element strides (4,2)
    py.Exec("src[1, 1] = 77.5");                              // tv[1, 1] reads 77.5
    tv[2, 0] = -12.0;                                         // src[2, 0].item() reads -12.0
    py.Exec("del src; import gc; gc.collect()");              // the lease keeps the tensor storage alive: tv[2, 0] still reads -12.0
}

// autograd: a NumSharp-backed leaf tensor, gradient back as NumSharp
NDArray x = np.array(new[] { -2.0, 1.5, 3.0 });
using (Py.GIL())
{
    py.x = x.ToTorch();
    py.Exec("x.requires_grad_()\nloss = (x * x).sum()\nloss.backward()");
    double loss = py.loss.item();                             // 15.25
    NDArray dx = py.x.grad;                                   // [-4. 3. 6.] — x.grad decodes through the adapter (== x.grad.ToTorchNDArray())
}

// tensors that cannot be shared: `force: true` detaches, transfers and copies
using (Py.GIL())
{
    py.Exec("g = torch.tensor([1.5, -2.0, 4.25], dtype=torch.float64, requires_grad=True)");
    using PyObject g = py.g;
    Exception? requiresGrad = Throws(() => g.AsTorchNDArray());               // PythonException: ... requires grad — PyTorch itself refuses Tensor.numpy()
    NDArray forced = g.ToTorchNDArray(force: true);                           // detach().cpu().resolve_conj().resolve_neg().numpy(), then a NumSharp copy: [1.5, -2, 4.25]
    py.Exec("with torch.no_grad():\n    g[0] = 999.0");                       // forced[0] still reads 1.5: an independent snapshot
    NDArray viaCodec = g.As<NDArray>();                                       // the codec's Auto mode: View is impossible -> the force-copy route

    py.Exec("cj = torch.tensor([1+2j, 3-4j], dtype=torch.complex128).conj()");
    using PyObject cj = py.cj;
    Exception? lazyConjugate = Throws(() => cj.AsTorchNDArray());             // PythonException: ... resolve_conj() — a lazy conjugate bit cannot be viewed
    NDArray resolved = cj.ToTorchNDArray(force: true);                        // force resolves the bit: resolved[0] == (1, -2)

    bool cuda = py.Eval("torch.cuda.is_available()");
    if (cuda)
    {
        py.Exec("dev = torch.arange(6, dtype=torch.float64, device='cuda')");
        using PyObject dev = py.dev;
        Exception? onDevice = Throws(() => dev.AsTorchNDArray());             // PythonException: ... cpu() — a CUDA tensor cannot share CPU NumSharp memory
        NDArray onCpu = dev.ToTorchNDArray(force: true);                      // force: true transfers device -> CPU, then copies: onCpu[5] == 5.0
    }
}

// expand(): a stride-0 tensor imports as a NON-writeable NumSharp broadcast
using (Py.GIL())
{
    py.Exec("base = torch.arange(3, dtype=torch.float64)\nexp = base.expand(4, 3)");
    NDArray view = py.exp;                                                    // element strides (0, 1), IsBroadcasted, IsWriteable == false
    Exception? blindWrite = Throws(() => np.copyto(view, np.ones(new Shape(4, 3))));   // NumSharpException: assignment destination is read-only
}

// dtypes PyTorch cannot hand to numpy are directed errors; complex64 copy-widens
using (Py.GIL())
{
    using PyObject c64 = py.Eval("torch.tensor([1+2j], dtype=torch.complex64)");
    Exception? c64View = Throws(() => c64.AsTorchNDArray());                  // NotSupportedException: complex64 has no zero-copy view
    NDArray widened = c64.ToTorchNDArray();                                   // typecode Complex: (1, 2)
    using PyObject bf16 = py.Eval("torch.arange(3, dtype=torch.bfloat16)");
    Exception? bfloat16 = Throws(() => bf16.ToTorchNDArray(force: true));     // PythonException: ... BFloat16 — even with force
    using PyObject sparse = py.Eval("torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([3.0]), (2, 2))");
    Exception? sparseLayout = Throws(() => sparse.AsTorchNDArray());          // PythonException: ... to_dense()
    using PyObject meta = py.Eval("torch.empty(2, device='meta')");
    Exception? noData = Throws(() => meta.ToTorchNDArray(force: true));       // PythonException: ... meta — a meta tensor has no data
    Exception? notABuffer = Throws(() => py.Exec("memoryview(torch.arange(3))"));   // PythonException (TypeError): torch.Tensor is NOT a PEP 3118 exporter — the adapter goes through Tensor.numpy()
}

// ==== the codec and the ordinary verbs ==============================================================
NDArray source = np.arange(5).astype(np.float64);
using (Py.GIL())
{
    py.implicit_t = py.torch.from_numpy(source);              // torch.from_numpy(nd) with no explicit ToNumpy(): the codec encodes the NDArray argument
    py.sv = source;
    bool samePointer = py.Eval("implicit_t.data_ptr() == sv.ctypes.data");   // True: the tensor points at NumSharp's buffer
    py.Exec("implicit_t[1] = 88.0");                          // source[1] reads 88.0

    using PyObject tensor = py.implicit_t;
    NDArray viaGeneric = tensor.As<NDArray>();                // registered decoder -> Torch adapter -> the view bridge
    NDArray viaType = (NDArray)tensor.AsManagedObject(typeof(NDArray))!;
    NDArray viaVerb = tensor.AsNDArray();                     // the ordinary verb, same route
    NDArray viaDynamic = py.implicit_t;                       // the dynamic spelling, same route
    NDArray copy = tensor.ToNDArray();                        // the ordinary copy verb
    viaGeneric[2] = -12.0;                                    // viaType[2], viaVerb[2], viaDynamic[2] and implicit_t[2] all read -12.0
    copy[3] = -9.0;                                           // implicit_t[3] still reads 3.0: the copy is detached
}

// shared storage cannot be resized on either side
NDArray pinned = np.arange(4).astype(np.int64);
using (Py.GIL())
{
    py.pt = pinned.ToTorch();
    Exception? numsharpSide = Throws(() => pinned.resize(new Shape(8)));      // IncorrectShapeException: cannot resize an array that references or is referenced
    Exception? torchSide = Throws(() => py.Exec("pt.resize_(8)"));            // PythonException: ... storage is not resizable
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every NumSharp view of tensor storage (their leases go back);
// `host` drops the namespace, so t, v, ts, x, implicit_t, sv and pt die with their names and the export
// pins with them; then the engine shuts down. LiveExports / LiveImports both read 0 afterwards.

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
