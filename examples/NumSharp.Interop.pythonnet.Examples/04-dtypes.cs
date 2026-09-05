#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 04 — Dtypes. Fourteen of NumSharp's fifteen dtypes cross bit-exactly in both directions; the
//      public maps (ToNumpyDtypeStr / FromNumpyDtypeStr / ToBufferFormat / FromBufferFormat) spell
//      the correspondence, and four element types get converted rather than misrepresented:
//      Decimal (-> float64 on export), complex64 (widened on copy), UCS-4 text (narrowed on copy),
//      big-endian (byte-swapped on copy). Every claim below is checked against live numpy.
//
//      Run:  dotnet run 04-dtypes.cs

using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var py = new PyScope();
var check = new Checklist();

// ==== the table: every dtype, both directions ======================================================
check.Section("14 dtypes round-trip; numpy names them as the table says");
var table = new (NPTypeCode Ns, string DtypeStr, string Pep, string NumpyName, NPTypeCode Back)[]
{
    (NPTypeCode.Boolean, "|b1",  "?",  "bool",       NPTypeCode.Boolean),
    (NPTypeCode.Byte,    "|u1",  "B",  "uint8",      NPTypeCode.Byte),
    (NPTypeCode.SByte,   "|i1",  "b",  "int8",       NPTypeCode.SByte),
    (NPTypeCode.Int16,   "<i2",  "h",  "int16",      NPTypeCode.Int16),
    (NPTypeCode.UInt16,  "<u2",  "H",  "uint16",     NPTypeCode.UInt16),
    (NPTypeCode.Int32,   "<i4",  "i",  "int32",      NPTypeCode.Int32),
    (NPTypeCode.UInt32,  "<u4",  "I",  "uint32",     NPTypeCode.UInt32),
    (NPTypeCode.Int64,   "<i8",  "q",  "int64",      NPTypeCode.Int64),
    (NPTypeCode.UInt64,  "<u8",  "Q",  "uint64",     NPTypeCode.UInt64),
    (NPTypeCode.Half,    "<f2",  "e",  "float16",    NPTypeCode.Half),
    (NPTypeCode.Single,  "<f4",  "f",  "float32",    NPTypeCode.Single),
    (NPTypeCode.Double,  "<f8",  "d",  "float64",    NPTypeCode.Double),
    (NPTypeCode.Complex, "<c16", "Zd", "complex128", NPTypeCode.Complex),
    (NPTypeCode.Char,    "<u2",  "H",  "uint16",     NPTypeCode.UInt16),   // a C# char is a UTF-16 code unit: uint16 on the numpy side
};

foreach (var (ns, dtypeStr, pep, numpyName, back) in table)
{
    bool ok = NDArrayPythonInterop.ToNumpyDtypeStr(ns) == dtypeStr && NDArrayPythonInterop.ToBufferFormat(ns) == pep;
    using NDArray src = ns == NPTypeCode.Char ? np.array(new[] { 'a', 'b', 'c' }) : np.arange(3).astype(ns);
    NDArray roundTrip;
    using (Py.GIL())
    {
        using PyObject x = src.ToNumpy();                                          // zero-copy view
        py.Set("x", x);
        roundTrip = x.ToNDArray();                                                 // copy it straight back
    }
    ok &= py.Str("x.dtype.str") == dtypeStr && py.Str("x.dtype.name") == numpyName;
    ok &= roundTrip.typecode == back;
    ok &= py.Bool("np.array_equal(x, np.frombuffer(x.tobytes(), x.dtype))");      // the bytes ARE the values
    roundTrip.Dispose();
    check.That(ok, $"{ns,-8} <-> {dtypeStr,-4} '{pep,-2}' numpy '{numpyName}'{(back != ns ? $" (imports as {back})" : "")}");
}

check.Section("the reverse maps");
check.That(NDArrayPythonInterop.FromNumpyDtypeStr("<f8") == NPTypeCode.Double && NDArrayPythonInterop.FromNumpyDtypeStr("=i4") == NPTypeCode.Int32 &&
           NDArrayPythonInterop.FromNumpyDtypeStr("|b1") == NPTypeCode.Boolean && NDArrayPythonInterop.FromNumpyDtypeStr("u2") == NPTypeCode.UInt16,
           "FromNumpyDtypeStr accepts '<', '=', '|' and bare codes");
check.That(NDArrayPythonInterop.FromBufferFormat("i", 4) == NPTypeCode.Int32 && NDArrayPythonInterop.FromBufferFormat("l", 8) == NPTypeCode.Int64 &&
           NDArrayPythonInterop.FromBufferFormat("l", 4) == NPTypeCode.Int32 && NDArrayPythonInterop.FromBufferFormat("", 1) == NPTypeCode.Byte,
           "FromBufferFormat disambiguates C 'l'/'i' by itemsize; an empty format is raw bytes");
check.That(NDArrayPythonInterop.FromBufferFormat("u", 2) == NPTypeCode.Char, "a 2-byte wchar_t unit ('u', Windows) IS a UTF-16 code unit — viewable as Char");
check.That(NDArrayPythonInterop.FromBufferFormat("g", 8) == NPTypeCode.Double, "MSVC's 8-byte long double ('g') is IEEE double — viewable");
check.Throws<NotSupportedException>(() => NDArrayPythonInterop.FromBufferFormat("g", 16), "a 16-byte long double has no NumSharp dtype", "astype(np.float64)");

// ==== bit-exactness: NaN payloads, signed zero, infinities, unsigned maxima ========================
check.Section("special values cross bit-for-bit");
using NDArray specials = np.array(new[]
{
    -0.0, double.NegativeInfinity, double.NaN,
    BitConverter.Int64BitsToDouble(unchecked((long)0xfff8123456789abc)),   // a NaN with a payload
    double.Epsilon, double.MaxValue,
});
using (Py.GIL()) { using PyObject p = specials.ToNumpy(); py.Set("sp", p); }
bool bitsOk = true;
for (int i = 0; i < specials.size; i++)
    bitsOk &= BitConverter.DoubleToInt64Bits(specials.GetDouble(i)) == py.Long($"sp.view('i8')[{i}]");
check.That(bitsOk, "every float64 bit pattern (incl. NaN payload, -0.0) reads identically through numpy");
using NDArray big = np.array(new[] { ulong.MaxValue, 0UL, 1UL << 63 });
using (Py.GIL()) { using PyObject p = big.ToNumpy(); py.Set("bg", p); }
check.That(py.Str("bg.tolist()") == "[18446744073709551615, 0, 9223372036854775808]", $"uint64 maxima: {py.Str("bg.tolist()")}");
using NDArray h = np.array(new[] { Half.NaN, Half.PositiveInfinity, (Half)(-0.0), Half.Epsilon });
using (Py.GIL()) { using PyObject p = h.ToNumpy(); py.Set("hf", p); }
check.That(py.Str("hf.dtype") == "float16" && py.Bool("np.isnan(hf[0]) and np.isposinf(hf[1]) and np.signbit(hf[2])"), "float16 specials: NaN, +inf, -0.0 as numpy sees them");
using NDArray cx = np.array(new[] { new Complex(1, -2), new Complex(double.NaN, double.PositiveInfinity) });
using (Py.GIL()) { using PyObject p = cx.ToNumpy(); py.Set("cx", p); }
check.That(py.Str("cx.dtype") == "complex128" && py.Str("cx[0]") == "(1-2j)" && py.Bool("np.isnan(cx[1].real) and np.isposinf(cx[1].imag)"), $"complex128: {py.Str("cx[0]")}, NaN+infj");

// ==== the four conversions ==========================================================================
check.Section("Decimal: no numpy dtype — the array verbs convert to float64, the low-level maps refuse");
using NDArray dec = np.array(new[] { 1.5m, -2.25m, 1234567890.123456789m });
check.Throws<NotSupportedException>(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal), "ToNumpyDtypeStr(Decimal)", "no numpy dtype");
check.Throws<NotSupportedException>(() => NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Decimal), "ToBufferFormat(Decimal)", "no PEP 3118 format");
using (Py.GIL())
{
    using PyObject p = dec.ToNumpy(); py.Set("dc", p);
    check.Throws<NotSupportedException>(() => dec.ToMemoryView().Dispose(), "ToMemoryView(Decimal) stays honest about the missing format", "no PEP 3118 format");
}
check.That(py.Str("dc.dtype") == "float64" && py.Float("dc[1]") == -2.25, $"ToNumpy(Decimal) -> float64 {py.Str("dc.tolist()")} (lossy past ~16 significant digits, by design)");
py.Exec("dc[0] = 99.0");
check.That((decimal)dec.GetValue(0) == 1.5m && py.Str("dc.flags['OWNDATA']") == "False",
           "...a numpy view over the CONVERTED float64 temporary (OWNDATA=False, kept alive by the export pin) — writes never reach the Decimal source");

check.Section("complex64: two float32 halves — widened to Complex (complex128) on copy, never viewed");
using (Py.GIL())
{
    using PyObject c8 = py.Module.Eval("np.array([1+2j, 3-4j], dtype='c8')");
    check.Throws<NotSupportedException>(() => c8.AsNDArray().Dispose(), "AsNDArray declines complex64", "complex64");
    using NDArray widened = c8.ToNDArray();
    check.That(widened.typecode == NPTypeCode.Complex && (Complex)widened.GetValue(1) == new Complex(3, -4), $"ToNDArray widens: {widened.GetValue(1)}");
    check.Throws<NotSupportedException>(() => NDArrayPythonInterop.FromNumpyDtypeStr("<c8"), "FromNumpyDtypeStr('<c8') says why", "widens");
}

check.Section("UCS-4 text: numpy '<U1' (4-byte code points) — narrowed to Char (UTF-16) on copy");
using (Py.GIL())
{
    using PyObject u1 = py.Module.Eval("np.array(['a', 'Z', '\\u00e9'], dtype='U1')");
    check.Throws<NotSupportedException>(() => u1.AsNDArray().Dispose(), "AsNDArray declines UCS-4", "UCS-4");
    using NDArray narrowed = u1.ToNDArray();
    check.That(narrowed.typecode == NPTypeCode.Char && (char)narrowed.GetValue(2) == 'é', $"ToNDArray narrows: '{narrowed.GetValue(0)}', '{narrowed.GetValue(1)}', '{narrowed.GetValue(2)}'");
    using PyObject astral = py.Module.Eval("np.array(['\\U0001F600'], dtype='U1')");
    check.Throws<NotSupportedException>(() => astral.ToNDArray().Dispose(), "a non-BMP code point needs a surrogate pair and is refused", "non-BMP");
}
// A 2-byte wchar_t buffer (array.array('u') on Windows) is UTF-16 already and views zero-copy as Char.
if (py.Long("__import__('array').array('u').itemsize") == 2)
{
    py.Exec("import array\nwide = array.array('u', 'hi!')");
    using (Py.GIL())
    {
        using PyObject wide = py.Get("wide");
        using NDArray chars = wide.AsNDArray();
        check.That(chars.typecode == NPTypeCode.Char && (char)chars.GetValue(1) == 'i', "array.array('u') (2-byte wchar_t) views zero-copy as Char");
        chars[0] = 'H';
        check.That(py.Str("wide.tounicode()") == "Hi!", $"the Char write is visible in Python: wide.tounicode() == '{py.Str("wide.tounicode()")}'");
    }
}
else
    Console.WriteLine("  (skip) array.array('u') is 4-byte on this platform — it takes the UCS-4 copy route above");

check.Section("big-endian: no zero-copy view (a native read would swap every value) — byte-reversed on copy");
check.Throws<NotSupportedException>(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8"), "FromNumpyDtypeStr('>f8') refuses with the fix", "newbyteorder('<')");
using (Py.GIL())
{
    using PyObject be = py.Module.Eval("np.array([1, -2, 300000], dtype='>i4')");
    check.Throws<NotSupportedException>(() => be.AsNDArray().Dispose(), "AsNDArray declines '>i4'", "big-endian");
    using NDArray swapped = be.ToNDArray();
    check.That(swapped.typecode == NPTypeCode.Int32 && swapped.GetInt32(2) == 300000 && swapped.GetInt32(1) == -2, $"ToNDArray byte-swaps: [{swapped.GetInt32(0)}, {swapped.GetInt32(1)}, {swapped.GetInt32(2)}]");
    using PyObject bec = py.Module.Eval("np.array([1+2j], dtype='>c16')");
    using NDArray swappedC = bec.ToNDArray();
    check.That((Complex)swappedC.GetValue(0) == new Complex(1, 2), "complex128 big-endian: each 8-byte half reversed independently");
    using PyObject be1 = py.Module.Eval("np.array([1, 2, 3], dtype='>i1')");
    using NDArray oneByte = be1.AsNDArray();
    check.That(oneByte.typecode == NPTypeCode.SByte && oneByte.GetValue(2) is sbyte and 3, "single-byte '>i1' views zero-copy regardless — byte order is meaningless at one byte");
}

check.Section("no NumSharp dtype at any byte order: refused on both paths");
using (Py.GIL())
{
    using PyObject dt = py.Module.Eval("np.array(['2021-01-01'], dtype='M8[D]')");
    check.Throws<NotSupportedException>(() => dt.AsNDArray(allowReadonly: true).Dispose(), "datetime64 view");
    check.Throws<NotSupportedException>(() => dt.ToNDArray().Dispose(), "datetime64 copy");
    using PyObject rec = py.Module.Eval("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])");
    check.Throws<NotSupportedException>(() => rec.ToNDArray().Dispose(), "a structured dtype");
}

check.Section("counters");
py.Exec("del x, sp, bg, hf, cx, dc");
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
