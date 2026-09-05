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
//      Run:  dotnet run 09-pytorch.cs        (self-skips when torch is not installed)

using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

try { py.Exec("import torch"); }
catch (PythonException) { Console.WriteLine("  (skip) python package 'torch' is not installed — see requirements.txt"); return 0; }
Console.WriteLine($"torch {py.Str("torch.__version__")}  cuda={py.Bool("torch.cuda.is_available()")}");

// ==== NumSharp -> PyTorch ============================================================================
check.Section("nd.ToTorch(): a zero-copy CPU tensor over NumSharp's buffer");
using NDArray matrix = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
using (Py.GIL())
{
    using PyObject tensor = matrix.ToTorch();                          // torch.from_numpy(matrix.ToNumpy())
    using PyObject view = matrix.ToNumpy();
    py.Set("t", tensor); py.Set("v", view);
}
check.That(py.Bool("t.data_ptr() == v.ctypes.data"), "the tensor's data_ptr() IS NumSharp's buffer address");
py.Exec("t[1, 2] = 91.5");
check.That(matrix.GetDouble(1, 2) == 91.5, "a tensor write lands in the NDArray");
matrix[2, 1] = -7.25;
check.That(py.Float("t[2, 1].item()") == -7.25, "an NDArray write is visible through the tensor");

using NDArray everyOtherColumn = matrix[":, ::2"];                     // (3,2), element strides (4,2)
using (Py.GIL()) { using PyObject ts = everyOtherColumn.ToTorch(); py.Set("ts", ts); }
check.That(py.Str("tuple(ts.shape)") == "(3, 2)" && py.Str("ts.stride()") == "(4, 2)", $"a strided view keeps shape {py.Str("tuple(ts.shape)")} and stride {py.Str("ts.stride()")}");
py.Exec("ts.add_(1000)");
check.That(matrix.GetDouble(0, 0) == 1000 && matrix.GetDouble(0, 1) == 1 && matrix.GetDouble(0, 2) == 1002, "in-place add_ touched columns 0 and 2 only");

check.Section("every NumSharp dtype maps to a torch dtype (Decimal converts to float64)");
var dtypes = new (NPTypeCode Tc, string Torch)[]
{
    (NPTypeCode.Boolean, "torch.bool"), (NPTypeCode.Byte, "torch.uint8"), (NPTypeCode.SByte, "torch.int8"),
    (NPTypeCode.Int16, "torch.int16"), (NPTypeCode.UInt16, "torch.uint16"), (NPTypeCode.Int32, "torch.int32"),
    (NPTypeCode.UInt32, "torch.uint32"), (NPTypeCode.Int64, "torch.int64"), (NPTypeCode.UInt64, "torch.uint64"),
    (NPTypeCode.Char, "torch.uint16"), (NPTypeCode.Half, "torch.float16"), (NPTypeCode.Single, "torch.float32"),
    (NPTypeCode.Double, "torch.float64"), (NPTypeCode.Decimal, "torch.float64"), (NPTypeCode.Complex, "torch.complex128"),
};
bool allDtypes = true;
using (Py.GIL())
    foreach (var (tc, expected) in dtypes)
    {
        using NDArray src = np.arange(3).astype(tc);
        using PyObject tensor = src.ToTorch(requireGIL: false);
        using PyObject dtype = tensor.GetAttr("dtype");
        if (dtype.ToString() != expected) { allDtypes = false; Console.WriteLine($"    {tc} -> {dtype} (expected {expected})"); }
    }
check.That(allDtypes, "15/15: bool u8 i8 i16 u16 i32 u32 i64 u64 char->u16 f16 f32 f64 decimal->f64 c128");

check.Section("unsafe layouts need copy:true");
using NDArray reversed = matrix["::-1"];
using NDArray broadcast = np.broadcast_to(matrix[0], new Shape(2, 4));
check.Throws<NotSupportedException>(() => { using (Py.GIL()) reversed.ToTorch().Dispose(); }, "negative strides: PyTorch tensors cannot express them", "copy:true");
check.Throws<InvalidOperationException>(() => { using (Py.GIL()) broadcast.ToTorch().Dispose(); }, "a non-writeable (broadcast) array: writing through it would be undefined behavior", "copy:true");
using (Py.GIL())
{
    using PyObject rc = reversed.ToTorch(copy: true); py.Set("rc", rc);
    using PyObject bc = broadcast.ToTorch(copy: true); py.Set("bc", bc);
}
check.That(py.Bool("rc[0].tolist() == v[2].tolist() and rc[2].tolist() == v[0].tolist()") && py.Str("tuple(bc.shape)") == "(2, 4)", $"copy:true materializes the logical values: rc[0] == {py.Str("rc[0].tolist()")}");
double before20 = matrix.GetDouble(2, 0), before00 = matrix.GetDouble(0, 0);
py.Exec("rc[0, 0] = -1.0; bc[0, 0] = -2.0");
check.That(matrix.GetDouble(2, 0) == before20 && matrix.GetDouble(0, 0) == before00, "...and detaches the tensor from NumSharp");
check.That(Settle(() => NDArrayPythonInterop.LiveExports == 3), $"a copy holds no export pin — only the three live views (t, v, ts) do (LiveExports == {NDArrayPythonInterop.LiveExports})");

// ==== PyTorch -> NumSharp ============================================================================
check.Section("tensor.AsNDArray() / AsTorchNDArray(): a zero-copy NumSharp view of CPU storage");
py.Exec("src = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]");
NDArray tv;
using (Py.GIL()) { using PyObject src = py.Get("src"); tv = src.AsTorchNDArray(); }   // == src.AsNDArray()
check.That(tv.shape.SequenceEqual(new long[] { 3, 2 }) && tv.Shape.Strides.SequenceEqual(new long[] { 4, 2 }), "shape (3,2), element strides (4,2) preserved");
py.Exec("src[1, 1] = 77.5");
check.That(tv.GetDouble(1, 1) == 77.5, "a tensor write reaches the view");
tv[2, 0] = -12.0;
check.That(py.Float("src[2, 0].item()") == -12.0, "an NDArray write reaches the tensor");
py.Exec("del src\nimport gc; gc.collect()");
check.That(tv.GetDouble(2, 0) == -12.0, "the lease keeps the tensor storage alive after Python drops its name");
tv.Dispose();

check.Section("autograd: a NumSharp-backed leaf tensor, gradient back as NumSharp");
using NDArray x = np.array(new[] { -2.0, 1.5, 3.0 });
using (Py.GIL())
{
    using PyObject xt = x.ToTorch();
    py.Set("x", xt);
    py.Module.Exec("x.requires_grad_()\nloss = (x * x).sum()\nloss.backward()");
    check.That(py.Float("loss.item()") == 15.25, $"loss == {py.Float("loss.item()")}");
    using PyObject grad = py.Module.Eval("x.grad");
    using NDArray dx = grad.ToTorchNDArray();
    check.That(dx.GetDouble(0) == -4 && dx.GetDouble(1) == 3 && dx.GetDouble(2) == 6, $"x.grad.ToTorchNDArray() == [{dx.GetDouble(0)}, {dx.GetDouble(1)}, {dx.GetDouble(2)}]");
}

check.Section("tensors that cannot be shared: `force:true` detaches, transfers and copies");
py.Exec("g = torch.tensor([1.5, -2.0, 4.25], dtype=torch.float64, requires_grad=True)");
using (Py.GIL())
{
    using PyObject g = py.Get("g");
    check.Throws<PythonException>(() => g.AsTorchNDArray(requireGIL: false).Dispose(), "requires_grad: PyTorch itself refuses Tensor.numpy()", "requires grad");
    using NDArray forced = g.ToTorchNDArray(force: true, requireGIL: false);       // detach().cpu().resolve_conj().resolve_neg().numpy(), then a NumSharp copy
    check.That(forced.GetDouble(2) == 4.25, "ToTorchNDArray(force: true) copies the values");
    py.Module.Exec("with torch.no_grad():\n    g[0] = 999.0");
    check.That(forced.GetDouble(0) == 1.5, "...as an independent snapshot");
}
py.Exec("cj = torch.tensor([1+2j, 3-4j], dtype=torch.complex128).conj()");
using (Py.GIL())
{
    using PyObject cj = py.Get("cj");
    check.Throws<PythonException>(() => cj.AsTorchNDArray(requireGIL: false).Dispose(), "a lazy conjugate bit cannot be viewed", "resolve_conj");
    using NDArray resolved = cj.ToTorchNDArray(force: true, requireGIL: false);
    check.That((Complex)resolved.GetValue(0) == new Complex(1, -2), "force resolves the bit: (1-2j)");
}
if (py.Bool("torch.cuda.is_available()"))
{
    py.Exec("dev = torch.arange(6, dtype=torch.float64, device='cuda')");
    using (Py.GIL())
    {
        using PyObject dev = py.Get("dev");
        check.Throws<PythonException>(() => dev.AsTorchNDArray(requireGIL: false).Dispose(), "a CUDA tensor cannot share CPU NumSharp memory", "cpu()");
        using NDArray onCpu = dev.ToTorchNDArray(force: true, requireGIL: false);
        check.That(onCpu.GetDouble(5) == 5.0, "force: true transfers device -> CPU, then copies");
    }
}
else
    Console.WriteLine("  (skip) no CUDA device — the device->CPU copy section needs one");

check.Section("expand(): a stride-0 tensor imports as a NON-writeable NumSharp broadcast");
py.Exec("base = torch.arange(3, dtype=torch.float64)\nexp = base.expand(4, 3)");
using (Py.GIL())
{
    using PyObject exp = py.Get("exp");
    using NDArray view = exp.AsTorchNDArray(requireGIL: false);
    check.That(view.Shape.Strides.SequenceEqual(new long[] { 0, 1 }) && view.Shape.IsBroadcasted && !view.Shape.IsWriteable, "strides (0,1), IsBroadcasted, IsWriteable == false");
    using NDArray ones = np.ones(new Shape(4, 3));
    check.Throws<Exception>(() => np.copyto(view, ones), "a blind write is refused", "read-only");
}

check.Section("dtypes PyTorch cannot hand to numpy are directed errors, complex64 copy-widens");
using (Py.GIL())
{
    using PyObject c64 = py.Module.Eval("torch.tensor([1+2j], dtype=torch.complex64)");
    check.Throws<NotSupportedException>(() => c64.AsTorchNDArray(requireGIL: false).Dispose(), "complex64 view", "complex64");
    using NDArray widened = c64.ToTorchNDArray(requireGIL: false);
    check.That(widened.typecode == NPTypeCode.Complex && (Complex)widened.GetValue(0) == new Complex(1, 2), "ToTorchNDArray widens complex64 -> Complex");
    using PyObject bf16 = py.Module.Eval("torch.arange(3, dtype=torch.bfloat16)");
    check.Throws<PythonException>(() => bf16.ToTorchNDArray(force: true, requireGIL: false).Dispose(), "bfloat16 (even with force)", "BFloat16");
    using PyObject sparse = py.Module.Eval("torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([3.0]), (2, 2))");
    check.Throws<PythonException>(() => sparse.AsTorchNDArray(requireGIL: false).Dispose(), "sparse layout", "to_dense");
    using PyObject meta = py.Module.Eval("torch.empty(2, device='meta')");
    check.Throws<PythonException>(() => meta.ToTorchNDArray(force: true, requireGIL: false).Dispose(), "a meta tensor has no data", "meta");
    using PyObject t0 = py.Module.Eval("torch.arange(3, dtype=torch.float64)");
    check.Throws<PythonException>(() => py.Module.Exec("memoryview(torch.arange(3))"), "torch.Tensor is NOT a PEP 3118 exporter — the adapter goes through Tensor.numpy()", "Tensor");
}

// ==== the codec and the ordinary verbs ==============================================================
check.Section("the ordinary verbs and the registered codec see tensors through the adapter");
NDArrayPythonInterop.RegisterCodec();
using NDArray source = np.arange(5).astype(NPTypeCode.Double);
using (Py.GIL())
{
    dynamic torch = Py.Import("torch");
    using PyObject tensor = (PyObject)torch.from_numpy(source);              // the NDArray argument is encoded as its numpy view
    using PyObject sv = source.ToNumpy();
    py.Set("implicit_t", tensor); py.Set("sv", sv);
    check.That(py.Bool("implicit_t.data_ptr() == sv.ctypes.data") && py.Float("implicit_t[1].item()") == 1.0, "torch.from_numpy(nd) works with no explicit ToNumpy() call — and points at NumSharp's buffer");
    py.Module.Exec("implicit_t[1] = 88.0");
    check.That(source.GetDouble(1) == 88.0, "...and shares NumSharp's memory");

    using NDArray viaGeneric = tensor.As<NDArray>();                          // registered decoder -> Torch adapter -> view bridge
    using NDArray viaType = (NDArray)tensor.AsManagedObject(typeof(NDArray))!;
    using NDArray viaVerb = tensor.AsNDArray();                               // the ordinary verb, same route
    using NDArray copy = tensor.ToNDArray();                                  // the ordinary copy verb
    viaGeneric[2] = -12.0;
    check.That(viaType.GetDouble(2) == -12.0 && viaVerb.GetDouble(2) == -12.0 && py.Float("implicit_t[2].item()") == -12.0, "As<NDArray>(), AsManagedObject and AsNDArray() are all views of the tensor");
    copy[3] = -9.0;
    check.That(py.Float("implicit_t[3].item()") == 3.0, "ToNDArray() is detached");

    using PyObject gradT = py.Module.Eval("torch.arange(4, dtype=torch.float64, requires_grad=True)");
    using NDArray autoCopy = gradT.As<NDArray>();                             // Auto: View is impossible -> falls back to the force-copy route
    autoCopy[0] = 700.0;
    check.That(py.Bool("float(torch.arange(4, dtype=torch.float64)[0]) == 0.0") && autoCopy.GetDouble(0) == 700.0, "Auto decode of a requires_grad tensor is a detached copy (View would decline)");
}

check.Section("shared storage cannot be resized on either side");
using NDArray pinned = np.arange(4).astype(NPTypeCode.Int64);
using (Py.GIL())
{
    using PyObject pt = pinned.ToTorch();
    check.Throws<Exception>(() => pinned.resize(new Shape(8)), "NumSharp's resize refuses while the tensor pins the buffer", "references or is referenced");
    using PyObject eight = 8L.ToPython();
    check.Throws<PythonException>(() => pt.InvokeMethod("resize_", eight).Dispose(), "tensor.resize_ refuses: the storage is not resizable", "resizable");
}

check.Section("counters");
py.Exec("del t, v, ts, rc, bc, x, g, cj, base, exp, implicit_t, sv, loss");
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
