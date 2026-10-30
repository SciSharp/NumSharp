#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Nullable=disable
#:property WarningLevel=0
#:property NoWarn=CS8632;CS0219;CS0168;NDW012;NDW001;NDW002;NDW003;NDW004;NDW005;NDW006;NDW007;NDW008;NDW009;NDW010;NDW011
// =============================================================================
// strings_full_bench.cs - NumSharp side of the full strings / chararray evaluation (benchmark/strings, the "full" sheet).
// Builds the SAME case list as strings_full_bench.py (identical keys {group}.{op}|{kind}|{n}) over the arrays the twin
// wrote with np.save (a StringDType array as NumPy's pickle, read with np.load_npy(allow_pickle: true)), and prints one
// line per case:   key \t ms \t sig     or     key \t REFUSED \t type: message
// sig is the result signature the twin computes too (dtype, shape and a content number: the total text length and NA
// count of a text result, the True count of a bool result, the sum of a numeric one), so the sheet proves both sides did
// the same work. Every case runs untimed once (refusal + signature), then an untimed warm-up pass over EVERY case (40
// calls at 1K, 2 at 100K) and a 1.5 s pause, then best-of-rounds timing with the twin's policy.
// NumPy's crash / stall cases (NUMPY_CRASHES / NUMPY_STALLS in the twin) are timed here anyway: NumSharp-only rows.
// Environment: NUMSHARP_STRINGS_FULL_DATA (the data directory; or args[0]), NUMSHARP_STRINGS_FULL_FILTER (comma key
//   prefixes, '$' = exact; or args[1]), NUMSHARP_STRINGS_FULL_SIZES (default 1000,100000), NUMSHARP_BENCHMARK_DEPTH
//   (pass | light | measure), NUMSHARP_BENCHMARK_AFFINITY (the host pin), NUMSHARP_STRINGS_FULL_LIST (print keys only).
// Driven by strings_full_sheet.py; standalone (after the twin's write mode):
//   NUMSHARP_STRINGS_FULL_DATA=<dir> dotnet run -c Release - < benchmark/strings/strings_full_bench.cs
// =============================================================================
using System.Collections;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using NumSharp;
using NumSharp.Backends;

var inv = CultureInfo.InvariantCulture;
var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core - rerun with -c Release"); return; }
if ((Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY")) is { Length: > 0 } aff)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(aff.StartsWith("0x") ? aff[2..] : aff, 16);
Console.Error.WriteLine($"affinity: 0x{(long)Process.GetCurrentProcess().ProcessorAffinity:X}");

string dataDir = args.Length > 0 ? args[0] : Environment.GetEnvironmentVariable("NUMSHARP_STRINGS_FULL_DATA");
if (string.IsNullOrEmpty(dataDir) || !Directory.Exists(dataDir))
{
    Console.Error.WriteLine("NUMSHARP_STRINGS_FULL_DATA (or args[0]) must name the directory strings_full_bench.py wrote into");
    return;
}
string filter = args.Length > 1 ? args[1] : Environment.GetEnvironmentVariable("NUMSHARP_STRINGS_FULL_FILTER");
string[] prefixes = string.IsNullOrEmpty(filter) || filter == "-" ? null : filter.Split(',');
int[] sizes = (Environment.GetEnvironmentVariable("NUMSHARP_STRINGS_FULL_SIZES") ?? "1000,100000").Split(',').Select(int.Parse).ToArray();
string depth = (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_DEPTH") ?? "measure").ToLowerInvariant();

var inputs = new HashSet<object>(ReferenceEqualityComparer.Instance);
var loaded = new Dictionary<string, NDArray>();
/// <summary>The array the twin saved as <c>{name}_{n}.npy</c>, loaded once and registered as an input (never disposed by
/// <see cref="Drop"/>).</summary>
/// <param name="name">The data kind (e.g. <c>u8</c>, <c>t40</c>, <c>ni8</c>, <c>mask</c>).</param>
/// <param name="n">The element count.</param>
/// <returns>The loaded array (a StringDType array is read from NumPy's pickle).</returns>
/// <exception cref="FileNotFoundException">The twin did not write that file.</exception>
NDArray Get(string name, int n)
{
    string key = $"{name}_{n}";
    if (!loaded.TryGetValue(key, out var a))
    {
        a = np.load_npy(Path.Combine(dataDir, key + ".npy"), allow_pickle: true);
        loaded[key] = a;
        inputs.Add(a);
    }
    return a;
}
/// <summary>Registers <paramref name="a"/> as a case input so <see cref="Drop"/> never disposes it (a call may return its
/// operand itself, and a disposed input would corrupt every later case).</summary>
/// <param name="a">The input array.</param>
/// <returns><paramref name="a"/>.</returns>
NDArray Keep(NDArray a) { inputs.Add(a); return a; }
/// <summary>The 2-D shape a 1-D data kind is viewed in for the transposed / broadcast cases (the twin's two_d).</summary>
/// <param name="n">The element count (1000 or 100000).</param>
/// <returns>(rows, columns) with rows * columns == n.</returns>
(int, int) TwoD(int n) => n == 1000 ? (25, 40) : (250, 400);
/// <summary>True for an <c>S</c> (bytes) array, whose text arguments are bytes in NumPy.</summary>
/// <param name="a">The operand.</param>
/// <returns>Whether <paramref name="a"/> holds <c>S</c> text.</returns>
bool IsS(NDArray a) => a.dtype.kind == 'S';
/// <summary>A text argument of <paramref name="a"/>'s class: a 0-d <c>S</c> array (Python bytes) for <c>S</c>, a 0-d
/// <c>U</c> array (Python str) otherwise. Arguments are built ONCE, outside the timed call (NumPy converts its Python
/// argument inside every call; the difference is the conversion of one scalar).</summary>
/// <param name="a">The operand whose text class the argument takes.</param>
/// <param name="s">The ASCII text.</param>
/// <returns>A registered 0-d <c>S</c> or <c>U</c> array of at least one character.</returns>
NDArray Nd(NDArray a, string s) => Keep(np.array(s, np.dtype((IsS(a) ? "S" : "U") + Math.Max(1, s.Length))));
/// <summary>A registered 0-d int64 array: the stand-in for a Python int argument (built once, outside the timed call).</summary>
/// <param name="v">The value.</param>
/// <returns>The 0-d array.</returns>
NDArray I(long v) => Keep(np.array(v));

// ------------------------------------------------------------------------------------------------ timing
/// <summary>Best-of-rounds milliseconds per call, the twin's policy: a warm-up whose timed tail is the pilot, ~200 ms of
/// batched rounds, and ~1 s of single-call rounds (5 to 100) for a call over 20 ms; one cold call at depth <c>pass</c>.</summary>
/// <param name="f">The timed call (it disposes its own result).</param>
/// <returns>The fastest round's milliseconds per call.</returns>
double Best(Action f)
{
    if (depth == "pass") { var once = Stopwatch.StartNew(); f(); return once.Elapsed.TotalMilliseconds; }
    int wm = depth == "light" ? 3 : 5;
    f();
    var pw = Stopwatch.StartNew();
    for (int i = 1; i < wm; i++) f();
    double perCall = pw.Elapsed.TotalMilliseconds / (wm - 1);
    bool slow = perCall > 20.0;
    int it = slow ? 1 : Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    double budget = 200.0 / (depth == "light" ? 6.0 : 1.0);
    int rd = slow ? (depth == "light" ? 3 : Math.Clamp((int)(1000.0 / perCall) + 1, 5, 100))
                  : Math.Max(1, (int)(budget / Math.Max(it * perCall, 1e-9)) + 1);
    double b = 1e9;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}

/// <summary>Disposes a case result so its buffer returns to the pool, as NumPy frees its result: an array, a list array, a
/// tuple of arrays or a unique result; a registered input (<see cref="Keep"/>) is left alone.</summary>
/// <param name="r">The result, or null for a call that returns nothing.</param>
void Drop(object r)
{
    switch (r)
    {
        case null: return;
        case NDArray a: if (!inputs.Contains(a)) a.Dispose(); return;
        case NDListArray la: la.Dispose(); return;
        case NDArray[] arr: foreach (var x in arr) if (x is not null && !inputs.Contains(x)) x.Dispose(); return;
        case ValueTuple<NDArray, NDArray, NDArray> t: Drop(t.Item1); Drop(t.Item2); Drop(t.Item3); return;
        case np.UniqueResult u: Drop((NDArray[])u); return;
    }
}

// ------------------------------------------------------------------------------------------------ signatures
/// <summary>The dtype's signature name: NumPy's kind plus item size (<c>U32</c> for <c>&lt;U8</c>), <c>T</c> for StringDType.</summary>
/// <param name="d">The descriptor.</param>
/// <returns>The name both twins print.</returns>
string DName(DType d) => d.kind == 'T' ? "T" : $"{d.kind}{d.itemsize}";
/// <summary>A number in the signature's format (six significant digits; the sheet compares numbers with a tolerance).</summary>
/// <param name="v">The value.</param>
/// <returns>The formatted value.</returns>
string Num(double v) => v.ToString("G6", inv);
/// <summary>The item count of a list result: an inner list counts its length, anything else (a string, a byte[]) one.</summary>
/// <param name="l">The outer list (<c>tolist()</c>, a split's lists).</param>
/// <returns>The total item count, as the twin's _list_items counts it.</returns>
long ListItems(IList l) { long t = 0; foreach (var x in l) t += x is IList li && x is not byte[] ? li.Count : 1; return t; }
/// <summary>The signature of an array result: a 0-d array reads as its scalar; otherwise dtype and shape plus a content
/// number - the total text length (code points for U / T, bytes for S) and the NA count of a text array, the True count
/// of a bool array, the sum of a numeric array.</summary>
/// <param name="a">The result array.</param>
/// <returns>The signature the twin computes for the same result.</returns>
string SigArray(NDArray a)
{
    char k = a.dtype.kind;
    if (a.ndim == 0)
    {
        if (k is 'S' or 'U' or 'T')
        {
            try { using var len = np.strings.str_len(a); return $"str:{len.GetAtIndex<long>(0)}"; }
            catch { return "num:nan"; }
        }
        if (k == 'b') return $"bool:{(a.GetAtIndex<bool>(0) ? 1 : 0)}";
        using var d = a.astype(np.complex128);
        var c = d.GetAtIndex<System.Numerics.Complex>(0);
        return $"num:{Num(c.Real + c.Imaginary)}";
    }
    string head = $"{DName(a.dtype)}[{string.Join("x", a.shape)}]";
    if (k is 'S' or 'U' or 'T')
    {
        if (a.size == 0) return $"{head}:0";
        if (k == 'T' && a.dtype.HasNAObject && a.dtype.na_object is double)
        {
            using var na = np.isnan(a);
            using var notna = np.logical_not(na);
            long cnt = np.count_nonzero(na);
            long tot = 0;
            if (cnt < a.size) { using var sub = a[notna]; using var ln = np.strings.str_len(sub); using var s = np.sum(ln); tot = s.GetAtIndex<long>(0); }
            return $"{head}:{tot}/{cnt}";
        }
        using var l = np.strings.str_len(a);
        using var sm = np.sum(l);
        return $"{head}:{sm.GetAtIndex<long>(0)}";
    }
    if (k == 'b') return $"{head}:{np.count_nonzero(a)}";
    if (k is 'i' or 'u')
    {
        if (a.size >= (1 << 20)) return $"{head}:n";
        using var f = a.astype(np.float64);
        using var s = np.sum(f);
        return $"{head}:{((long)s.GetAtIndex<double>(0)).ToString(inv)}";
    }
    if (k is 'f' or 'c')
    {
        using var cc = a.astype(np.complex128);
        double re = 0, im = 0;
        for (long i = 0; i < cc.size; i++) { var z = cc.GetAtIndex<System.Numerics.Complex>(i); if (!double.IsNaN(z.Real)) re += z.Real; if (!double.IsNaN(z.Imaginary)) im += z.Imaginary; }
        return $"{head}:{Num(re + im)}";
    }
    return head;
}
/// <summary>The signature of any case result (see <see cref="SigArray"/>); tuples join their parts with <c>;</c>.</summary>
/// <param name="r">The result.</param>
/// <returns>The signature.</returns>
string Sig(object r) => r switch
{
    null => "none",
    string s => $"str:{s.EnumerateRunes().Count()}",
    byte[] bs => $"str:{bs.Length}",
    bool b => $"bool:{(b ? 1 : 0)}",
    long or int or double or float or ulong => $"num:{Num(Convert.ToDouble(r, inv))}",
    NDListArray la => $"O8[{string.Join("x", la.shape)}]:{ListItems((IList)la.tolist())}",
    NDArray[] arr => string.Join(";", arr.Select(x => Sig(x))),
    ValueTuple<NDArray, NDArray, NDArray> t => $"{Sig(t.Item1)};{Sig(t.Item2)};{Sig(t.Item3)}",
    np.UniqueResult u => Sig((NDArray[])u),
    NDArray a => SigArray(a),
    DType d => $"dtype:{d}",
    IList l => $"list{l.Count}:{ListItems(l)}",
    _ => r.GetType().Name,
};

// ------------------------------------------------------------------------------------------------ cases
var sTable = new byte[256];
for (int i = 0; i < 256; i++) sTable[i] = (byte)i;
sTable['a'] = (byte)'x'; sTable['b'] = (byte)'y'; sTable['c'] = (byte)'z';
var uTable = new Dictionary<int, string> { [97] = "x", [98] = "y", [99] = "z" };
var abBytes = new byte[] { (byte)'a', (byte)'b' };

/// <summary>Every case at size <paramref name="n"/>, keyed <c>{group}.{op}|{kind}|{n}</c> exactly as the twin keys them,
/// with its arguments built here (outside the timed call).</summary>
/// <param name="n">The element count.</param>
/// <returns>The (key, call) pairs in the twin's order.</returns>
List<(string key, Func<object> fn)> Cases(int n)
{
    var o = new List<(string, Func<object>)>();
    /// <summary>Appends a case.</summary>
    /// <param name="op">The <c>{group}.{op}</c> part of the key.</param>
    /// <param name="k">The data kind part of the key.</param>
    /// <param name="fn">The call; it returns its result (or null) for the signature and the disposal.</param>
    void Add(string op, string k, Func<object> fn) => o.Add(($"{op}|{k}|{n}", fn));
    var (r, c) = TwoD(n);
    /// <summary>A data kind, possibly as a view: <c>name@2</c> is <c>[::2]</c> of the 2n-element base, <c>name@r</c>
    /// <c>[::-1]</c>, <c>name@T</c> the 2-D transposed view.</summary>
    /// <param name="k">The kind name.</param>
    /// <returns>The registered array or view.</returns>
    NDArray Kind(string k)
    {
        if (k.EndsWith("@2")) return Keep(Get(k[..^2] + "_2n", n)["::2"]);
        if (k.EndsWith("@r")) return Keep(Get(k[..^2], n)["::-1"]);
        if (k.EndsWith("@T")) return Keep(Get(k[..^2], n).reshape(r, c).T);
        return Get(k, n);
    }
    var mask = Get("mask", n); var idx = Get("idx", n); var cnt = Get("cnt", n);
    var T = np.dtypes.StringDType();

    // ---------------------------------------------------------------- cast
    string[] textSrc = { "s8", "u8", "t8", "tna", "u8n", "t8n", "s40", "u40", "t40" };
    var textDst = new (string name, DType dt)[]
    {
        ("S", np.dtype("S")), ("U", np.dtype("U")), ("T", T), ("S4", np.dtype("S4")), ("U4", np.dtype("U4")),
        ("S16", np.dtype("S16")), ("U16", np.dtype("U16")), ("same", null),
    };
    foreach (var k in textSrc)
        foreach (var (dn, dt) in textDst)
        {
            var a = Kind(k);
            Add($"cast.to_{dn}", k, () => a.astype(dt ?? a.dtype));
        }
    var numDst = new Dictionary<string, DType>
    {
        ["b1"] = np.bool_, ["i1"] = np.int8, ["i2"] = np.int16, ["i4"] = np.int32, ["i8"] = np.int64, ["u1"] = np.uint8,
        ["u2"] = np.uint16, ["u4"] = np.uint32, ["u8"] = np.uint64, ["f2"] = np.float16, ["f4"] = np.float32,
        ["f8"] = np.float64, ["c16"] = np.complex128,
    };
    foreach (var cls in new[] { "s", "u", "t" })
    {
        foreach (var (dn, dt) in numDst)
        {
            var a = Kind($"{cls}d2");
            Add($"cast.parse_{dn}", $"{cls}d2", () => a.astype(dt));
        }
        foreach (var dn in new[] { "i4", "i8", "u4", "u8", "f4", "f8" })
        {
            var a = Kind($"{cls}d6"); var dt = numDst[dn];
            Add($"cast.parse_{dn}", $"{cls}d6", () => a.astype(dt));
        }
        foreach (var dn in new[] { "f2", "f4", "f8", "c16" })
        {
            var a = Kind($"{cls}fl"); var dt = numDst[dn];
            Add($"cast.parse_{dn}", $"{cls}fl", () => a.astype(dt));
        }
    }
    foreach (var k in new[] { "nb1", "ni1", "ni4", "ni8", "nu8", "nf2", "nf4", "nf8", "nc16" })
        foreach (var (dn, dt) in new (string, DType)[] { ("S", np.dtype("S")), ("U", np.dtype("U")), ("T", T) })
        {
            var a = Kind(k);
            Add($"cast.fmt_{dn}", k, () => a.astype(dt));
        }
    foreach (var k in new[] { "u8@2", "u8@r", "u8@T" })
    {
        var a = Kind(k);
        Add("cast.to_S", k, () => a.astype(np.dtype("S")));
        Add("cast.to_T", k, () => a.astype(T));
    }
    foreach (var k in new[] { "t8@2", "t8@r", "t8@T" })
    {
        var a = Kind(k);
        Add("cast.to_U16", k, () => a.astype(np.dtype("U16")));
        Add("cast.to_S16", k, () => a.astype(np.dtype("S16")));
    }
    foreach (var k in new[] { "s8@2", "s8@r" })
    {
        var a = Kind(k);
        Add("cast.to_U", k, () => a.astype(np.dtype("U")));
    }
    { var a = Keep(Get("nf8", n)["::-1"]); Add("cast.fmt_U", "nf8@r", () => a.astype(np.dtype("U"))); }
    { var a = Keep(Get("ud6", n)["::-1"]); Add("cast.parse_f8", "ud6@r", () => a.astype(np.float64)); }

    // ---------------------------------------------------------------- copy (NDStringCopy routes)
    foreach (var k in new[] { "s8", "u8", "t8", "tna", "s40", "u40", "t40" })
    {
        var a = Kind(k);
        Add("copy.copy", k, () => a.copy());
        Add("copy.copy_rev", k, () => { using var v = a["::-1"]; return v.copy(); });
        Add("copy.copy_T", k, () => { using var m = a.reshape(r, c); using var t = m.T; return t.copy(); });
        var dst = Keep(np.empty_like(a));
        Add("copy.copyto", k, () => { np.copyto(dst, a); return null; });
        var dstw = Keep(np.empty_like(a));
        Add("copy.copyto_where", k, () => { np.copyto(dstw, a, where: mask); return null; });
        var dst2 = Keep(np.empty(new Shape(2L * n), a.dtype));
        Add("copy.set_step", k, () => { dst2["::2"] = a; return null; });
        var dstm = Keep(np.empty(new Shape(r, c), a.dtype));
        var row = Keep(a[$":{c}"].copy());
        Add("copy.set_bcast", k, () => { dstm["..."] = row; return null; });
        var dsts = Keep(np.empty_like(a));
        var xyz = Nd(a, "xyz");
        Add("copy.set_scalar", k, () => { dsts["..."] = xyz; return null; });
        var dstf = Keep(np.empty_like(a));
        Add("copy.fill", k, () => { dstf.fill("xyz"); return null; });
        var dsto = Keep(a.copy());
        Add("copy.shift1", k, () => { using var src = dsto[":-1"]; dsto["1:"] = src; return null; });
        var dstk = Keep(a.copy());
        var zz = Nd(a, "zz");
        Add("copy.set_mask", k, () => { dstk[mask] = zz; return null; });
    }
    foreach (var k in new[] { "s8", "u8", "t8" })
    {
        var a2 = Kind(k + "@2");
        Add("copy.copy_step", k + "@2", () => a2.copy());
        var aT = Kind(k + "@T");
        Add("copy.ascontig_T", k + "@T", () => np.ascontiguousarray(aT));
        // three effective axes: past the element-grid route (at most two), NDStringCopy's general unit-view copy
        var a = Kind(k);
        long[] shp3 = n == 1000 ? new long[] { 10, 10, 10 } : new long[] { 40, 50, 50 };
        Add("copy.copy_3d", k, () => { using var m = a.reshape(shp3); using var t = np.transpose(m, new[] { 2, 0, 1 }); return t.copy(); });
    }
    {
        var u8 = Get("u8", n); var s8 = Get("s8", n); var t8 = Get("t8", n);
        var pairs = new (string op, string k, NDArray a, NDArray dst, string casting)[]
        {
            ("copyto_U8toU16", "u8", u8, Keep(np.empty(new Shape(n), np.dtype("U16"))), "same_kind"),
            ("copyto_U8toU4", "u8", u8, Keep(np.empty(new Shape(n), np.dtype("U4"))), "unsafe"),
            ("copyto_S8toU8", "s8", s8, Keep(np.empty(new Shape(n), np.dtype("U8"))), "same_kind"),
            ("copyto_U8toS8", "u8", u8, Keep(np.empty(new Shape(n), np.dtype("S8"))), "unsafe"),
            ("copyto_S8toS16", "s8", s8, Keep(np.empty(new Shape(n), np.dtype("S16"))), "same_kind"),
            ("copyto_U8toT", "u8", u8, Keep(np.empty(new Shape(n), T)), "same_kind"),
            ("copyto_TtoU8", "t8", t8, Keep(np.empty(new Shape(n), np.dtype("U8"))), "unsafe"),
            ("copyto_S8toT", "s8", s8, Keep(np.empty(new Shape(n), T)), "same_kind"),
            ("copyto_TtoS8", "t8", t8, Keep(np.empty(new Shape(n), np.dtype("S8"))), "unsafe"),
            ("copyto_i8toU", "ni8", Get("ni8", n), Keep(np.empty(new Shape(n), np.dtype("U21"))), "unsafe"),
            ("copyto_f8toU", "nf8", Get("nf8", n), Keep(np.empty(new Shape(n), np.dtype("U32"))), "unsafe"),
            ("copyto_Utof8", "ufl", Get("ufl", n), Keep(np.empty(new Shape(n), np.float64)), "unsafe"),
            ("copyto_Utoi8", "ud6", Get("ud6", n), Keep(np.empty(new Shape(n), np.int64)), "unsafe"),
            ("copyto_Ttof8", "tfl", Get("tfl", n), Keep(np.empty(new Shape(n), np.float64)), "unsafe"),
            ("copyto_i8toT", "ni8", Get("ni8", n), Keep(np.empty(new Shape(n), T)), "unsafe"),
        };
        foreach (var (op, k, a, dst, casting) in pairs)
            Add($"copy.{op}", k, () => { np.copyto(dst, a, casting); return null; });
    }

    // ---------------------------------------------------------------- meta (resolve_descriptors / dtype layer)
    if (n == sizes[0])
    {
        DType dU8 = np.dtype("U8"), dS8 = np.dtype("S8"), dS3 = np.dtype("S3"), dU5 = np.dtype("U5"), dT = T, di8 = np.dtype("i8");
        DType dU8b = np.dtype("U8");
        var u8 = Get("u8", n); var s8 = Get("s8", n); var t8 = Get("t8", n);
        var meta = new (string, Func<object>)[]
        {
            ("dtype_U8", () => np.dtype("U8")), ("dtype_S8", () => np.dtype("S8")),
            ("dtype_T", () => np.dtypes.StringDType()), ("dtype_TNA", () => np.dtypes.StringDType(na_object: double.NaN)),
            ("can_cast_U8_S8", () => np.can_cast(dU8, dS8)), ("can_cast_S8_U8", () => np.can_cast(dS8, dU8)),
            ("can_cast_U8_T", () => np.can_cast(dU8, dT)), ("can_cast_T_U8_sk", () => np.can_cast(dT, dU8, "same_kind")),
            ("can_cast_i8_U8", () => np.can_cast(di8, dU8)), ("can_cast_U8_i8_u", () => np.can_cast(dU8, di8, "unsafe")),
            ("promote_S3_U5", () => np.promote_types(dS3, dU5)), ("promote_U8_T", () => np.promote_types(dU8, dT)),
            ("result_type_uS", () => np.result_type(u8, s8)), ("result_type_tU", () => np.result_type(t8, u8)),
            ("dtype_eq", () => dU8 == dU8b),
        };
        foreach (var (op, fn) in meta)
            o.Add(($"meta.{op}|-|0", fn));
    }

    // ---------------------------------------------------------------- ufunc
    var three = I(3);
    foreach (var k in new[] { "s8", "u8", "t8", "tna", "u8n", "t8n", "s40", "u40", "t40", "s8@2", "u8@2", "t8@2", "u8@r", "t8@r" })
    {
        var a = Kind(k);
        var b = Keep(np.roll(a, 1));
        var sc = Nd(a, "abc");
        var xy = Nd(a, "xy");
        Add("ufunc.eq", k, () => a == b);
        Add("ufunc.ne", k, () => a != b);
        Add("ufunc.lt", k, () => a < b);
        Add("ufunc.le", k, () => a <= b);
        Add("ufunc.gt", k, () => a > b);
        Add("ufunc.ge", k, () => a >= b);
        Add("ufunc.eq_sc", k, () => a == sc);
        Add("ufunc.add", k, () => np.add(a, b));
        Add("ufunc.add_sc", k, () => np.add(a, xy));
        Add("ufunc.mul3", k, () => np.strings.multiply(a, three));
        Add("ufunc.mul_arr", k, () => np.strings.multiply(a, cnt));
        Add("ufunc.maximum", k, () => np.maximum(a, b));
        Add("ufunc.minimum", k, () => np.minimum(a, b));
    }
    {
        var t8 = Get("t8", n); var u8 = Get("u8", n);
        Add("ufunc.eq_TU", "t8", () => t8 == u8);
        Add("ufunc.add_TU", "t8", () => np.add(t8, u8));
    }

    // ---------------------------------------------------------------- reduce
    string[] core = { "s8", "u8", "t8", "tna", "u40", "t40" };
    foreach (var k in core)
    {
        var a = Kind(k);
        Add("reduce.max", k, () => np.max(a));
        Add("reduce.min", k, () => np.min(a));
        Add("reduce.argmax", k, () => np.argmax(a));
        Add("reduce.argmin", k, () => np.argmin(a));
        Add("reduce.max_ax0", k, () => { using var m = a.reshape(r, c); return np.max(m, axis: 0); });
        Add("reduce.nonzero", k, () => np.nonzero(a));
        Add("reduce.count_nonzero", k, () => np.count_nonzero(a));
        Add("reduce.any", k, () => np.any(a));
        Add("reduce.all", k, () => np.all(a));
        if (n == sizes[0])
            Add("reduce.sum", k, () => np.sum(a));
    }

    // ---------------------------------------------------------------- sort / search / sets
    foreach (var k in core.Append("u8n"))
    {
        var a = Kind(k);
        var b = Keep(np.roll(a, 1));
        var hay = Keep(np.sort(a));
        var head = Keep(a[":100"]);
        Add("sort.sort", k, () => np.sort(a));
        Add("sort.argsort", k, () => np.argsort(a));
        Add("sort.sort_stable", k, () => np.sort(a, kind: "stable"));
        Add("sort.argsort_stable", k, () => np.argsort(a, kind: "stable"));
        Add("sort.partition", k, () => np.partition(a, n / 2));
        Add("sort.argpartition", k, () => np.argpartition(a, n / 2));
        Add("sort.searchsorted", k, () => np.searchsorted(hay, b));
        Add("sort.lexsort", k, () => np.lexsort(new[] { a, b }));
        Add("sort.unique", k, () => np.unique(a));
        Add("sort.unique_counts", k, () => np.unique(a, return_counts: true));
        Add("sort.unique_inverse", k, () => np.unique(a, return_inverse: true));
        Add("sort.isin", k, () => np.isin(a, head));
        Add("sort.intersect1d", k, () => np.intersect1d(a, b));
        Add("sort.union1d", k, () => np.union1d(a, b));
        Add("sort.setdiff1d", k, () => np.setdiff1d(a, b));
        Add("sort.setxor1d", k, () => np.setxor1d(a, b));
        Add("sort.intersect1d_au", k, () => np.intersect1d(a, b, true));
        Add("sort.setxor1d_au", k, () => np.setxor1d(a, b, assume_unique: true));
        Add("sort.quantile", k, () => np.quantile(a, 0.5, method: "lower"));
    }

    // ---------------------------------------------------------------- np.strings
    string[] strKinds = { "s8", "u8", "t8", "tna", "u40", "t40" };
    var w12 = I(12); var w10 = I(10); var i1 = I(1); var i3 = I(3); var i4 = I(4); var i5 = I(5); var im1 = I(-1); var i0 = I(0);
    foreach (var k in strKinds.Concat(new[] { "u8n", "t8n" }).Concat(new[] { "u8@2", "t8@2" }))
    {
        var a = Kind(k);
        bool full = strKinds.Contains(k);
        bool strided = k.EndsWith("@2");
        var nA = Nd(a, "a"); var nAb = Nd(a, "ab"); var nXy = Nd(a, "xy");
        Add("strings.str_len", k, () => np.strings.str_len(a));
        Add("strings.find", k, () => np.strings.find(a, nA));
        Add("strings.upper", k, () => np.strings.upper(a));
        Add("strings.isalpha", k, () => np.strings.isalpha(a));
        Add("strings.replace", k, () => np.strings.replace(a, nA, nXy));
        Add("strings.strip", k, () => np.strings.strip(a, nAb));
        Add("strings.center", k, () => np.strings.center(a, w12));
        Add("strings.startswith", k, () => np.strings.startswith(a, nA));
        Add("strings.slice", k, () => np.strings.slice(a, i1, i4));
        if (strided) continue;
        Add("strings.lower", k, () => np.strings.lower(a));
        Add("strings.swapcase", k, () => np.strings.swapcase(a));
        Add("strings.capitalize", k, () => np.strings.capitalize(a));
        Add("strings.title", k, () => np.strings.title(a));
        Add("strings.isdigit", k, () => np.strings.isdigit(a));
        Add("strings.isspace", k, () => np.strings.isspace(a));
        Add("strings.isalnum", k, () => np.strings.isalnum(a));
        Add("strings.islower", k, () => np.strings.islower(a));
        Add("strings.isupper", k, () => np.strings.isupper(a));
        Add("strings.istitle", k, () => np.strings.istitle(a));
        Add("strings.isdecimal", k, () => np.strings.isdecimal(a));
        Add("strings.isnumeric", k, () => np.strings.isnumeric(a));
        if (!IsS(a))
            Add("strings.encode", k, () => np.strings.encode(a, "utf-8"));
        if (!full) continue;
        var nStar = Nd(a, "*"); var nEmpty = Nd(a, "");
        Add("strings.rfind", k, () => np.strings.rfind(a, nA));
        Add("strings.find_rng", k, () => np.strings.find(a, nA, i1, i5));
        Add("strings.count", k, () => np.strings.count(a, nA));
        Add("strings.endswith", k, () => np.strings.endswith(a, nA));
        if (k != "tna")
        {
            var first = Keep(np.strings.slice(a, i0, i1));
            Add("strings.index", k, () => np.strings.index(a, first));
            Add("strings.rindex", k, () => np.strings.rindex(a, first));
            Add("strings.startswith_arr", k, () => np.strings.startswith(a, first));
            Add("strings.find_arr", k, () => np.strings.find(a, first));
        }
        Add("strings.lstrip", k, () => np.strings.lstrip(a, nAb));
        Add("strings.rstrip", k, () => np.strings.rstrip(a, nAb));
        Add("strings.strip_ws", k, () => np.strings.strip(a));
        Add("strings.ljust", k, () => np.strings.ljust(a, w12));
        Add("strings.rjust", k, () => np.strings.rjust(a, w12));
        Add("strings.center_fill", k, () => np.strings.center(a, w12, nStar));
        Add("strings.zfill", k, () => np.strings.zfill(a, w10));
        Add("strings.expandtabs", k, () => np.strings.expandtabs(a));
        Add("strings.replace_c1", k, () => np.strings.replace(a, nA, nXy, i1));
        Add("strings.replace_del", k, () => np.strings.replace(a, nA, nEmpty));
        Add("strings.partition", k, () => np.strings.partition(a, nA));
        Add("strings.rpartition", k, () => np.strings.rpartition(a, nA));
        Add("strings.slice_rev", k, () => np.strings.slice(a, null, null, im1));
        Add("strings.slice_stop", k, () => np.strings.slice(a, i3));
        Add("strings.translate", k, () => np.strings.translate(a, IsS(a) ? sTable : uTable));
        if (IsS(a))
        {
            Add("strings.translate_del", k, () => np.strings.translate(a, sTable, abBytes));
            Add("strings.decode", k, () => np.strings.decode(a, "utf-8"));
            Add("strings.decode_latin1", k, () => np.strings.decode(a, "latin-1"));
            var fmt = Keep(np.full(a.Shape, "<%s>", np.dtype("S4")));
            Add("strings.mod", k, () => np.strings.mod(fmt, a));
        }
        else
        {
            Add("strings.encode_ascii", k, () => np.strings.encode(a, "ascii"));
            var fmt = Keep(np.full(a.Shape, "<%s>", a.dtype.kind == 'T' ? a.dtype : np.dtype("U4")));
            Add("strings.mod", k, () => np.strings.mod(fmt, a));
        }
    }
    foreach (var cls in new[] { "s", "u", "t" })
    {
        var a = Kind($"{cls}d6");
        var fmt = Keep(np.full(new Shape(n), "%07d", cls == "s" ? np.dtype("S4") : cls == "u" ? np.dtype("U4") : T));
        var v = Get("ni4", n);
        Add("strings.mod_int", $"{cls}d6", () => np.strings.mod(fmt, v));
        Add("strings.isdigit", $"{cls}d6", () => np.strings.isdigit(a));
        Add("strings.isnumeric", $"{cls}d6", () => np.strings.isnumeric(a));
    }

    // ---------------------------------------------------------------- np.char
    var ch = np.@char;
    foreach (var k in new[] { "s8", "u8", "t8" })
    {
        var a = Kind(k);
        var b = Keep(np.roll(a, 1));
        var nA = Nd(a, "a"); var nAb = Nd(a, "ab"); var nXy = Nd(a, "xy"); var nDash = Nd(a, "-");
        Add("char.equal", k, () => ch.equal(a, b));
        Add("char.not_equal", k, () => ch.not_equal(a, b));
        Add("char.less", k, () => ch.less(a, b));
        Add("char.greater", k, () => ch.greater(a, b));
        Add("char.less_equal", k, () => ch.less_equal(a, b));
        Add("char.greater_equal", k, () => ch.greater_equal(a, b));
        Add("char.compare_chararrays", k, () => ch.compare_chararrays(a, b, "==", true));
        Add("char.add", k, () => ch.add(a, b));
        Add("char.multiply", k, () => ch.multiply(a, 3));
        Add("char.str_len", k, () => ch.str_len(a));
        Add("char.find", k, () => ch.find(a, nA));
        Add("char.count", k, () => ch.count(a, nA));
        Add("char.upper", k, () => ch.upper(a));
        Add("char.title", k, () => ch.title(a));
        Add("char.strip", k, () => ch.strip(a, nAb));
        Add("char.center", k, () => ch.center(a, 12));
        Add("char.zfill", k, () => ch.zfill(a, 10));
        Add("char.replace", k, () => ch.replace(a, nA, nXy));
        Add("char.isalpha", k, () => ch.isalpha(a));
        Add("char.partition", k, () => ch.partition(a, nA));
        Add("char.rpartition", k, () => ch.rpartition(a, nA));
        Add("char.join", k, () => ch.join(nDash, a));
        Add("char.array", k, () => ch.array(a));
        Add("char.asarray", k, () => ch.asarray(a));
        if (IsS(a)) Add("char.decode", k, () => ch.decode(a, "utf-8"));
        else Add("char.encode", k, () => ch.encode(a, "utf-8"));
    }
    foreach (var k in new[] { "sw", "uw", "tw" })
    {
        var a = Kind(k);
        var nA = Nd(a, "a");
        Add("char.split", k, () => ch.split(a));
        Add("char.rsplit", k, () => ch.rsplit(a));
        Add("char.split_sep", k, () => ch.split(a, nA));
    }
    foreach (var k in new[] { "sln", "uln", "tln" })
    {
        var a = Kind(k);
        Add("char.splitlines", k, () => ch.splitlines(a));
    }

    // ---------------------------------------------------------------- chararray members and operators
    foreach (var k in new[] { "s8", "u8", "s40", "u40", "sw", "uw" })
    {
        var a = Kind(k);
        var cc = (NDCharArray)Keep(ch.array(a));
        var cb = (NDCharArray)Keep(ch.array(Keep(np.roll(a, 1))));
        var nA = Nd(a, "a"); var nAb = Nd(a, "ab"); var nXy = Nd(a, "xy"); var nAbc = Nd(a, "abc");
        Add("carr.eq", k, () => cc == cb);
        Add("carr.lt", k, () => cc < cb);
        Add("carr.eq_sc", k, () => cc == nAbc);
        Add("carr.add", k, () => cc + cb);
        Add("carr.mul3", k, () => cc * 3);
        var cfmt = (NDCharArray)Keep(ch.array(Keep(np.full(new Shape(n), "<%s>", IsS(a) ? np.dtype("S4") : np.dtype("U4")))));
        Add("carr.mod", k, () => cfmt % a);
        Add("carr.upper", k, () => cc.upper());
        Add("carr.title", k, () => cc.title());
        Add("carr.strip", k, () => cc.strip(nAb));
        Add("carr.find", k, () => cc.find(nA));
        Add("carr.count", k, () => cc.count(nA));
        Add("carr.startswith", k, () => cc.startswith(nA));
        Add("carr.replace", k, () => cc.replace(nA, nXy));
        Add("carr.center", k, () => cc.center(12));
        Add("carr.zfill", k, () => cc.zfill(10));
        Add("carr.isalpha", k, () => cc.isalpha());
        Add("carr.partition", k, () => cc.partition(nA));
        Add("carr.split", k, () => cc.split());
        Add("carr.translate", k, () => cc.translate(IsS(a) ? sTable : uTable));
        if (IsS(a)) Add("carr.decode", k, () => cc.decode("utf-8"));
        else Add("carr.encode", k, () => cc.encode("utf-8"));
        Add("carr.getitem_fancy", k, () => cc[idx]);
        Add("carr.take", k, () => cc.take(idx));
        Add("carr.copy", k, () => cc.copy());
        Add("carr.astype_U16", k, () => cc.astype(np.dtype("U16")));
        Add("carr.argsort", k, () => cc.argsort());
        Add("carr.argmax", k, () => cc.argmax());
        Add("carr.flatten", k, () => cc.flatten());
        Add("carr.tolist", k, () => cc.tolist());
        Add("carr.iter", k, () =>
        {
            long tot = 0;
            foreach (var x in cc)
                tot += x switch { string s => s.EnumerateRunes().Count(), byte[] bs => bs.Length, _ => 1 };
            return tot;
        });
    }

    // ---------------------------------------------------------------- indexing / manipulation
    foreach (var k in core)
    {
        var a = Kind(k);
        var b = Keep(np.roll(a, 1));
        Add("index.take", k, () => np.take(a, idx));
        Add("index.fancy", k, () => a[idx]);
        Add("index.bool", k, () => a[mask]);
        Add("index.where", k, () => np.where(mask, a, b));
        Add("index.concat", k, () => np.concatenate((a, b)));
        Add("index.stack", k, () => np.stack(new[] { a, b }));
        Add("index.repeat", k, () => np.repeat(a, 2));
        Add("index.tile", k, () => np.tile(a, 2));
        Add("index.flip_copy", k, () => { using var f = np.flip(a); return f.copy(); });
        Add("index.roll", k, () => np.roll(a, 1));
        Add("index.ravel_T", k, () => { using var m = a.reshape(r, c); using var t = m.T; return t.ravel(); });
        Add("index.compress", k, () => np.compress(mask, a));
        var dstp = Keep(a.copy());
        Add("index.put", k, () => { np.put(dstp, idx, a); return null; });
    }

    // ---------------------------------------------------------------- creation / conversion / printing
    foreach (var k in core)
    {
        var a = Kind(k);
        Add("create.tolist", k, () => a.tolist());
        Add("create.array2string", k, () => np.array2string(a));
        Add("create.repr", k, () => np.array_repr(a));
        Add("create.full_like", k, () => np.full_like(a, "zz"));
        Add("create.zeros_like", k, () => np.zeros_like(a));
        Add("create.empty_like", k, () => np.empty_like(a));
    }
    foreach (var (k, dt) in new (string, DType)[] { ("u8", np.dtype("U8")), ("s8", np.dtype("S8")), ("t8", T), ("u40", np.dtype("U40")), ("t40", T) })
    {
        var strs = NDArray.AsStringArray(Get(k.EndsWith("8") ? "u8" : "u40", n));
        Add("create.from_list", k, () => np.array(strs, dt));
        Add("create.full", k, () => np.full(new Shape(n), "abc", dt));
        Add("create.zeros", k, () => np.zeros(new Shape(n), dt));
        Add("create.empty", k, () => np.empty(new Shape(n), dt));
    }
    {
        var strs = NDArray.AsStringArray(Get("u8", n));
        Add("create.from_list_infer", "u8", () => np.array(strs));
    }

    // ---------------------------------------------------------------- I/O
    foreach (var k in core)
    {
        var a = Kind(k);
        Add("io.save", k, () => { using var ms = new MemoryStream(); np.save(ms, a); return ms.Length; });
        byte[] raw;
        using (var ms = new MemoryStream()) { np.save(ms, a); raw = ms.ToArray(); }
        Add("io.load", k, () => np.load_npy(raw, allow_pickle: true));
        if (!IsS(a) && k != "tna")
        {
            Add("io.savetxt", k, () => { var sw = new StringWriter(); np.savetxt(sw, a, fmt: "%s"); return (long)sw.ToString().Length; });
            var sw0 = new StringWriter(); np.savetxt(sw0, a, fmt: "%s");
            string text = sw0.ToString();
            var dt = a.dtype;
            Add("io.loadtxt", k, () => np.loadtxt(new StringReader(text), dtype: dt.kind == 'T' ? np.dtypes.StringDType() : dt, comments: null, delimiter: "\t"));
        }
    }
    return o;
}

// ------------------------------------------------------------------------------------------------ run
var plan = new List<(int n, string key, Func<object> fn)>();
foreach (int n in sizes)
{
    List<(string key, Func<object> fn)> cs;
    try { cs = Cases(n); }
    catch (Exception e) { Console.Error.WriteLine($"setup n={n} failed: {e}"); return; }
    foreach (var (key, fn) in cs)
    {
        if (prefixes is not null && !prefixes.Any(p => p.EndsWith("$") ? key == p[..^1] : key.StartsWith(p))) continue;
        if (Environment.GetEnvironmentVariable("NUMSHARP_STRINGS_FULL_LIST") is { Length: > 0 }) { Console.WriteLine(key); continue; }
        object res;
        try { res = fn(); }
        catch (Exception e)
        {
            var m = e.Message.Replace('\n', ' ').Replace('\t', ' ');
            Console.WriteLine($"{key}\tREFUSED\t{e.GetType().Name}: {(m.Length > 120 ? m[..120] : m)}");
            continue;
        }
        string sig;
        try { sig = Sig(res); } catch (Exception e) { sig = "SIGERR:" + e.GetType().Name; }
        Drop(res);
        plan.Add((n, key + "\t" + sig, fn));
    }
}
Console.Out.Flush();
if (depth != "pass")
{
    foreach (var (n, _, fn) in plan)
        for (int i = 0, reps = n <= 1000 ? 40 : 2; i < reps; i++)
            Drop(fn());
    Thread.Sleep(1500);
}
foreach (var (n, keysig, fn) in plan)
{
    double ms;
    try { ms = Best(() => Drop(fn())); }
    catch (Exception e) { Console.WriteLine($"{keysig.Split('\t')[0]}\tREFUSED\tTIMING {e.GetType().Name}"); continue; }
    var parts = keysig.Split('\t');
    Console.WriteLine($"{parts[0]}\t{ms.ToString("G6", inv)}\t{parts[1]}");
    Console.Out.Flush();
}
