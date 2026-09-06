#:project ../../src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj
#:property PublishAot=false
// Outside this repository, replace the #:project line with the NuGet package:
//   #:package NumSharp.Interop.pythonnet@0.60.0
//
// 04 — Dtypes. Fourteen of NumSharp's fifteen dtypes cross bit-exactly in both directions; the
//      public maps (ToNumpyDtypeStr / FromNumpyDtypeStr / ToBufferFormat / FromBufferFormat) spell
//      the correspondence, and four element types get converted rather than misrepresented:
//      Decimal (-> float64 on export), complex64 (widened on copy), UCS-4 text (narrowed on copy),
//      big-endian (byte-swapped on copy).
//
//      Read it top to bottom; the comments state what each line leaves behind. The same steps, with
//      every statement asserted, are test/NumSharp.Tests.Interop/Examples/Example04_Dtypes.cs.
//
//      Run:  dotnet run 04-dtypes.cs

using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

using var host = PythonHost.Start();
using var scope = NDScope.Open();
dynamic py = host.Namespace;

// ==== the table: every dtype, both directions ======================================================
//   NumSharp   dtype.str   PEP 3118   numpy name    back as
//   Boolean    |b1         ?          bool          Boolean
//   Byte       |u1         B          uint8         Byte
//   SByte      |i1         b          int8          SByte
//   Int16      <i2         h          int16         Int16
//   UInt16     <u2         H          uint16        UInt16
//   Int32      <i4         i          int32         Int32
//   UInt32     <u4         I          uint32        UInt32
//   Int64      <i8         q          int64         Int64
//   UInt64     <u8         Q          uint64        UInt64
//   Half       <f2         e          float16       Half
//   Single     <f4         f          float32       Single
//   Double     <f8         d          float64       Double
//   Complex    <c16        Zd         complex128    Complex
//   Char       <u2         H          uint16        UInt16     (a C# char is a UTF-16 code unit: uint16 on the numpy side)
foreach (NPTypeCode tc in new[]
         {
             NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
             NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex, NPTypeCode.Char,
         })
using (Py.GIL())
{
    NDArray src = tc == NPTypeCode.Char ? np.array(new[] { 'a', 'b', 'c' }) : np.arange(3).astype(tc);
    string dtypeStr = NDArrayPythonInterop.ToNumpyDtypeStr(tc);                  // the dtype.str column
    string format = NDArrayPythonInterop.ToBufferFormat(tc);                     // the PEP 3118 column
    py.x = src;                                                                  // a zero-copy view...
    string numpyName = py.x.dtype.name;                                          // ...named as the numpy-name column says
    NDArray back = py.x;                                                         // ...and straight back: the back-as column
    bool bytesAreTheValues = py.np.array_equal(py.x, py.np.frombuffer(py.x.tobytes(), py.x.dtype));   // True for every row
}

// ==== the reverse maps =============================================================================
NPTypeCode f8 = NDArrayPythonInterop.FromNumpyDtypeStr("<f8");                  // Double  — '<', '=', '|' and bare codes are all accepted:
NPTypeCode i4 = NDArrayPythonInterop.FromNumpyDtypeStr("=i4");                  // Int32
NPTypeCode b1 = NDArrayPythonInterop.FromNumpyDtypeStr("|b1");                  // Boolean
NPTypeCode u2 = NDArrayPythonInterop.FromNumpyDtypeStr("u2");                   // UInt16
NPTypeCode cInt = NDArrayPythonInterop.FromBufferFormat("i", 4);                // Int32   — C 'l' / 'i' disambiguate by itemsize:
NPTypeCode cLong8 = NDArrayPythonInterop.FromBufferFormat("l", 8);              // Int64
NPTypeCode cLong4 = NDArrayPythonInterop.FromBufferFormat("l", 4);              // Int32
NPTypeCode raw = NDArrayPythonInterop.FromBufferFormat("", 1);                  // Byte    — an empty format is raw bytes
NPTypeCode wchar = NDArrayPythonInterop.FromBufferFormat("u", 2);               // Char    — a 2-byte wchar_t IS a UTF-16 code unit
NPTypeCode longDouble = NDArrayPythonInterop.FromBufferFormat("g", 8);          // Double  — MSVC's 8-byte long double is IEEE double
Exception? extended = Throws(() => NDArrayPythonInterop.FromBufferFormat("g", 16));   // NotSupportedException: a 16-byte long double has no NumSharp dtype. Convert first: arr.astype(np.float64).

// ==== bit-exactness: NaN payloads, signed zero, infinities, unsigned maxima ========================
NDArray specials = np.array(new[]
{
    -0.0, double.NegativeInfinity, double.NaN,
    BitConverter.Int64BitsToDouble(unchecked((long)0xfff8123456789abc)),   // a NaN with a payload
    double.Epsilon, double.MaxValue,
});
using (Py.GIL())
{
    py.sp = specials;
    long payload = py.Eval("int(sp.view('i8')[3])");                             // == unchecked((long)0xfff8123456789abc): the NaN payload survives...
    long negativeZero = py.Eval("int(sp.view('i8')[0])");                        // == unchecked((long)0x8000000000000000): ...and so does the sign of -0.0,
    // -inf (0xfff0000000000000), 5e-324 (0x0000000000000001) and 1.7976931348623157e+308 (0x7fefffffffffffff) — every bit pattern reads identically on both sides

    py.bg = np.array(new[] { ulong.MaxValue, 0UL, 1UL << 63 });
    string maxima = py.bg.tolist().ToString();                                    // [18446744073709551615, 0, 9223372036854775808]

    py.hf = np.array(new[] { Half.NaN, Half.PositiveInfinity, (Half)(-0.0), Half.Epsilon });
    string halfName = py.hf.dtype.name;                                           // "float16"; np.isnan(hf[0]), np.isposinf(hf[1]) and np.signbit(hf[2]) are all True

    py.cx = np.array(new[] { new Complex(1, -2), new Complex(double.NaN, double.PositiveInfinity) });
    string complexName = py.cx.dtype.name;                                        // "complex128"; cx[0] is (1-2j), cx[1] is (nan+infj)
}

// ==== the four conversions ==========================================================================
// Decimal: no numpy dtype — the array verbs convert to float64, the low-level maps refuse
NDArray dec = np.array(new[] { 1.5m, -2.25m, 1234567890.123456789m });
Exception? noDtype = Throws(() => NDArrayPythonInterop.ToNumpyDtypeStr(NPTypeCode.Decimal));   // NotSupportedException: decimal has no numpy dtype (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double).
Exception? noFormat = Throws(() => NDArrayPythonInterop.ToBufferFormat(NPTypeCode.Decimal));  // NotSupportedException: decimal has no PEP 3118 format ...
using (Py.GIL())
{
    Exception? noMemoryView = Throws(() => dec.ToMemoryView());                   // NotSupportedException: the same — ToMemoryView stays honest about the missing format
    py.dc = dec;                                                                  // ToNumpy(Decimal): a float64 view over a CONVERTED temporary
    string decName = py.dc.dtype.name;                                            // "float64"; dc.tolist() == [1.5, -2.25, 1234567890.1234567] — lossy past ~16 significant digits, by design
    py.Exec("dc[0] = 99.0");                                                      // dec[0] is still 1.5m: writes never reach the Decimal source
    bool ownsData = py.dc.flags.owndata;                                          // False — the temporary is kept alive by the export pin, not by numpy
}

// complex64: two float32 halves — widened to Complex (complex128) on copy, never viewed
using (Py.GIL())
{
    using PyObject c8 = py.Eval("np.array([1+2j, 3-4j], dtype='c8')");
    Exception? noView = Throws(() => c8.AsNDArray());                             // NotSupportedException: buffer format 'Zf' (complex64) has no exact NumSharp dtype. ToNDArray widens it ...
    NDArray widened = c8.ToNDArray();                                             // typecode Complex; widened[1] == (3, -4)
    Exception? noStr = Throws(() => NDArrayPythonInterop.FromNumpyDtypeStr("<c8"));   // the map says the same
}

// UCS-4 text: numpy '<U1' (4-byte code points) — narrowed to Char (UTF-16) on copy
using (Py.GIL())
{
    using PyObject u1 = py.Eval("np.array(['a', 'Z', '\\u00e9'], dtype='U1')");
    Exception? noView = Throws(() => u1.AsNDArray());                             // NotSupportedException: buffer format '1w' (itemsize 4) is UCS-4 text ... a zero-copy view is impossible
    NDArray narrowed = u1.ToNDArray();                                            // typecode Char: 'a', 'Z', 'é'
    using PyObject astral = py.Eval("np.array(['\\U0001F600'], dtype='U1')");
    Exception? nonBmp = Throws(() => astral.ToNDArray());                         // NotSupportedException: ... non-BMP code point U+1F600 ... needs a surrogate pair

    long wcharSize = py.Eval("__import__('array').array('u').itemsize");
    if (wcharSize == 2)                                                           // a 2-byte wchar_t buffer (Windows) is UTF-16 already: it views zero-copy as Char
    {
        py.Exec("import array\nwide = array.array('u', 'hi!')");
        NDArray chars = py.wide;                                                  // typecode Char; chars[1] == 'i'
        chars[0] = 'H';                                                           // wide.tounicode() is now "Hi!"
    }
    // (a 4-byte wchar_t platform takes the UCS-4 copy route above)
}

// big-endian: no zero-copy view (a native read would swap every value) — byte-reversed on copy
Exception? bigEndianStr = Throws(() => NDArrayPythonInterop.FromNumpyDtypeStr(">f8"));   // NotSupportedException: big-endian dtype '>f8' cannot be shared ... Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).
using (Py.GIL())
{
    using PyObject be = py.Eval("np.array([1, -2, 300000], dtype='>i4')");
    Exception? noView = Throws(() => be.AsNDArray());                             // NotSupportedException: big-endian buffer format '>l' cannot be mapped ...
    NDArray swapped = be.ToNDArray();                                             // typecode Int32: [1, -2, 300000]
    using PyObject bec = py.Eval("np.array([1+2j], dtype='>c16')");
    NDArray swappedComplex = bec.ToNDArray();                                     // (1, 2): each 8-byte half is reversed independently
    using PyObject be1 = py.Eval("np.array([1, 2, 3], dtype='>i1')");
    NDArray oneByte = be1.AsNDArray();                                            // typecode SByte, a zero-copy view: byte order is meaningless at one byte
}

// no NumSharp dtype at any byte order: refused on both paths
using (Py.GIL())
{
    using PyObject dt = py.Eval("np.array(['2021-01-01'], dtype='M8[D]')");
    Exception? datetimeView = Throws(() => dt.AsNDArray(allowReadonly: true));    // NotSupportedException: numpy dtype '<M8[D]' has no NumSharp dtype.
    Exception? datetimeCopy = Throws(() => dt.ToNDArray());                       // NotSupportedException: ... (cannot include dtype 'M' in a buffer)
    using PyObject rec = py.Eval("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])");
    Exception? structured = Throws(() => rec.ToNDArray());                        // NotSupportedException: buffer format 'T{l:x:=d:y:}' (itemsize 12) has no NumSharp dtype.
}

// ==== who frees what ================================================================================
// When this script ends: `scope` releases every array above (the Decimal export's float64 temporary included —
// it is owned by the export and freed the moment Python lets go of dc); `host` drops the namespace and the
// pins with it; then the engine shuts down. LiveExports / LiveImports both read 0 afterwards.

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
