#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// expr_probe.cs — np.evaluate / NDExpr: the state of the DSL, measured.
//
// Run (from the repo, any cwd; pin to one P-core on a hybrid host):
//   NS_PROBE_AFFINITY=4 DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//     dotnet run -c Release benchmark/fusion/probes/expr_probe.cs [ABCDE]
//
// Sections:
//   A. semantics probes — suspected gaps/bugs, each printed as OK / DIVERGES / THROWS
//   B. dtype × op support sweep (all 15 dtypes × 12 expression shapes) -> a support matrix
//   C. fused vs unfused NumSharp, ms per call, 1K / 100K / 4M (twin: numpy_twins.py C)
//   D. fixed cost per call at n=8 (ns + managed bytes), kernel compile time
//   E. cache behaviour (kernel count vs distinct NDArray instances / constants)
//
// Twin: benchmark/fusion/probes/numpy_twins.py (same shapes, same data, NumPy 2.4.2).
// =============================================================================
using System.Diagnostics;
using System.Numerics;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

if ((Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY")
     ?? Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY")) is { Length: > 0 } hostAffinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(hostAffinity, 16);

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug-JITted NumSharp.Core — run with -c Release"); return; }
Console.WriteLine($"V256={Vector256.IsHardwareAccelerated} V512={Vector512.IsHardwareAccelerated} TC_CallCountingDelayMs={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs") ?? "(default 100)"} affinity={Process.GetCurrentProcess().ProcessorAffinity}");

string sections = args.Length > 0 ? args[0].ToUpperInvariant() : "ABCDE";

// ------------------------------------------------------------------ helpers
double BestMs(Action body, int rounds = 7)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    double best = double.MaxValue;
    for (int r = 0; r < rounds; r++) { var sw = Stopwatch.StartNew(); body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}
double BestNs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 60) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(3, (int)Math.Ceiling(150.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / it); }
    return best;
}
long AllocPerCall(Action body, int iters)
{
    for (int i = 0; i < 100; i++) body();
    long before = GC.GetAllocatedBytesForCurrentThread();
    for (int i = 0; i < iters; i++) body();
    return (GC.GetAllocatedBytesForCurrentThread() - before) / iters;
}
string Try(Func<string> f)
{
    try { return f(); }
    catch (Exception e) { return $"THROWS {e.GetType().Name}: {Head(e.Message, 110)}"; }
}
static string Head(string s, int n) { s = s.Replace("\r", " ").Replace("\n", " "); return s.Length <= n ? s : s[..n] + "…"; }
static long UlpDiffF64(double x, double y)
{
    if (x.Equals(y)) return 0;
    if (double.IsNaN(x) || double.IsNaN(y)) return long.MaxValue;
    return Math.Abs(BitConverter.DoubleToInt64Bits(x) - BitConverter.DoubleToInt64Bits(y));
}
static long UlpDiffF32(float x, float y)
{
    if (x.Equals(y)) return 0;
    if (float.IsNaN(x) || float.IsNaN(y)) return long.MaxValue;
    return Math.Abs((long)BitConverter.SingleToInt32Bits(x) - BitConverter.SingleToInt32Bits(y));
}
// Max ULP distance between two same-shaped float arrays (logical C-order).
static long MaxUlp(NDArray x, NDArray y)
{
    long m = 0;
    for (long i = 0; i < x.size; i++)
    {
        long d = x.typecode == NPTypeCode.Single
            ? UlpDiffF32(x.GetSingle(i), y.GetSingle(i))
            : UlpDiffF64(x.GetDouble(i), y.GetDouble(i));
        if (d > m) m = d;
    }
    return m;
}
static bool BitEqual(NDArray x, NDArray y)
{
    if (x.typecode != y.typecode || x.size != y.size) return false;
    for (long i = 0; i < x.size; i++)
        if (!Equals(x.GetAtIndex(i), y.GetAtIndex(i)) && !(x.GetAtIndex(i) is double dx && y.GetAtIndex(i) is double dy && dx.Equals(dy)))
            return false;
    return true;
}

// Deterministic bounded data (safe for exp/log/sqrt): values in [0,1) and [0.5,1.5).
(NDArray a, NDArray b, NDArray c) Data(long n)
{
    var idx = np.arange(n).astype(NPTypeCode.Double);
    var a = (idx % 977.0) / 977.0;
    var b = ((idx * 7.0) % 991.0) / 991.0 + 0.5;
    var c = ((idx * 13.0) % 983.0) / 983.0 * 0.5;
    return (a, b, c);
}

// =============================================================================
// A. semantics probes
// =============================================================================
if (sections.Contains('A'))
{
    Console.WriteLine("\n=== A. semantics probes ===");
    var (a, b, c) = Data(64);
    var f4a = a.astype(NPTypeCode.Single); var f4b = b.astype(NPTypeCode.Single);
    var f2a = a.astype(NPTypeCode.Half); var f2b = b.astype(NPTypeCode.Half);
    var i4 = np.arange(64).astype(NPTypeCode.Int32) - 32;
    var sb = np.arange(64).astype(NPTypeCode.SByte) - (sbyte)32;
    var i2 = np.arange(64).astype(NPTypeCode.Int16) - (short)32;
    var u1 = np.arange(64).astype(NPTypeCode.Byte);
    var z = np.array(new[] { new Complex(3, -4), new Complex(0, 0), new Complex(-1, 1), new Complex(double.NaN, 0) });
    var dec = np.arange(1, 9).astype(NPTypeCode.Decimal);

    void P(string id, string what, string got) => Console.WriteLine($"  [{id}] {what,-58} -> {got}");

    // A1/A2: sub-32-bit integer zero-push (WhereNode / LogicalNot / IsNaN-family use EmitPushZeroPublic)
    P("A1", "Where(int8 cond, f8, f8)", Try(() => { var r = np.evaluate(NDExpr.Where(NDExpr.Arr(sb), NDExpr.Arr(a), NDExpr.Arr(b))); return $"OK dtype={r.typecode} r[0]={r.GetDouble(0)} (cond[0]={sb.GetAtIndex(0)})"; }));
    P("A1b", "Where(int16 cond, f8, f8)", Try(() => { var r = np.evaluate(NDExpr.Where(NDExpr.Arr(i2), NDExpr.Arr(a), NDExpr.Arr(b))); return $"OK dtype={r.typecode}"; }));
    P("A2", "LogicalNot(int8)", Try(() => { var r = np.evaluate(NDExpr.LogicalNot(NDExpr.Arr(sb))); return $"OK dtype={r.typecode} r[32]={r.GetAtIndex(32)} (x[32]={sb.GetAtIndex(32)})"; }));
    P("A2b", "LogicalNot(uint8)", Try(() => { var r = np.evaluate(NDExpr.LogicalNot(NDExpr.Arr(u1))); return $"OK dtype={r.typecode} r[0]={r.GetAtIndex(0)}"; }));

    // A3/A4: complex unary typing
    P("A3", "Abs(complex128)  [NumPy: float64 magnitude 5,0,1.414,nan]", Try(() => { var r = np.evaluate(NDExpr.Abs(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r[0]={r.GetAtIndex(0)}"; }));
    P("A4", "Sign(complex128) [NumPy 2.x: z/|z|]", Try(() => { var r = np.evaluate(NDExpr.Sign(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r[0]={r.GetAtIndex(0)}"; }));
    P("A4b", "Square(complex128) vs np.square", Try(() => { var r = np.evaluate(NDExpr.Square(NDExpr.Arr(z))); var u = np.square(z); return BitEqual(r, u) ? "OK bit-equal" : $"DIVERGES r[0]={r.GetAtIndex(0)} u[0]={u.GetAtIndex(0)}"; }));
    P("A4c", "Exp(complex128) vs np.exp", Try(() => { var r = np.evaluate(NDExpr.Exp(NDExpr.Arr(z))); var u = np.exp(z); return BitEqual(r, u) ? "OK bit-equal" : $"DIVERGES r[0]={r.GetAtIndex(0)} u[0]={u.GetAtIndex(0)}"; }));
    P("A4d", "IsNaN(complex128) [NumPy: True iff either part NaN]", Try(() => { var r = np.evaluate(NDExpr.IsNaN(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r[3]={r.GetAtIndex(3)} r[0]={r.GetAtIndex(0)}"; }));

    // A5: negative integer exponent ARRAY (documented divergence: NumPy raises ValueError)
    P("A5", "Power(i4, i4 with negatives)  [NumPy: ValueError]", Try(() => { var r = np.evaluate(NDExpr.Power(NDExpr.Arr(i4), NDExpr.Arr(i4))); return $"OK(silently) dtype={r.typecode} r[0]={r.GetAtIndex(0)} (=(-32)^-32)"; }));

    // A6: complex reductions
    P("A6", "Sum(complex128) vs np.sum", Try(() => { var r = np.evaluate(NDExpr.Sum(NDExpr.Arr(z))); var u = np.sum(z); return $"OK dtype={r.typecode} r={r.GetAtIndex(0)} np={u.GetAtIndex(0)}"; }));
    P("A6b", "Prod(complex128)", Try(() => { var r = np.evaluate(NDExpr.Prod(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r={r.GetAtIndex(0)}"; }));
    P("A6c", "Min(complex128) [NumPy: lexicographic]", Try(() => { var r = np.evaluate(NDExpr.Min(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r={r.GetAtIndex(0)}"; }));
    P("A6d", "Mean(complex128)", Try(() => { var r = np.evaluate(NDExpr.Mean(NDExpr.Arr(z))); return $"OK dtype={r.typecode} r={r.GetAtIndex(0)}"; }));

    // A7: Half arithmetic tree — per-op f16 rounding parity vs unfused
    P("A7", "f2*f2+f2 fused vs unfused (per-op Half rounding)", Try(() => { var r = np.evaluate((NDExpr)f2a * f2b + f2a); var u = f2a * f2b + f2a; return BitEqual(r, u) ? $"OK bit-equal dtype={r.typecode}" : $"DIVERGES dtype={r.typecode} r[5]={r.GetAtIndex(5)} u[5]={u.GetAtIndex(5)}"; }));
    P("A7b", "Exp(f2) fused vs np.exp(f2)", Try(() => { var r = np.evaluate(NDExpr.Exp(NDExpr.Arr(f2a))); var u = np.exp(f2a); return BitEqual(r, u) ? $"OK bit-equal dtype={r.typecode}" : $"DIVERGES r[5]={r.GetAtIndex(5)} u[5]={u.GetAtIndex(5)}"; }));
    P("A7c", "Sqrt(f2) fused vs np.sqrt(f2)", Try(() => { var r = np.evaluate(NDExpr.Sqrt(NDExpr.Arr(f2a))); var u = np.sqrt(f2a); return BitEqual(r, u) ? $"OK bit-equal dtype={r.typecode}" : $"DIVERGES r[5]={r.GetAtIndex(5)} u[5]={u.GetAtIndex(5)}"; }));
    P("A7d", "Round(f2) / Floor(f2)", Try(() => { var r = np.evaluate(NDExpr.Round(NDExpr.Arr(f2b))); var f = np.evaluate(NDExpr.Floor(NDExpr.Arr(f2b))); return $"OK dtype={r.typecode} round[3]={r.GetAtIndex(3)} floor[3]={f.GetAtIndex(3)} (x[3]={f2b.GetAtIndex(3)})"; }));

    // A8/A9: float32 reduction accumulation vs the unfused (pairwise) NumSharp sum
    {
        var (A, B, _) = Data(100_003);
        var af = A.astype(NPTypeCode.Single); var bf = B.astype(NPTypeCode.Single);
        P("A8", "Sum(af*bf) fused vs np.sum(af*bf)  [f32 acc policy]", Try(() =>
        {
            float rf = np.evaluate(NDExpr.Sum((NDExpr)af * bf)).GetSingle(0);
            float uf = np.sum(af * bf).GetSingle(0);
            return $"fused=0x{BitConverter.SingleToInt32Bits(rf):X8} ({rf:R}) unfused=0x{BitConverter.SingleToInt32Bits(uf):X8} ({uf:R}) ulp={UlpDiffF32(rf, uf)}";
        }));
        P("A9", "Mean(af*bf) fused vs np.mean(af*bf)", Try(() =>
        {
            float rf = np.evaluate(NDExpr.Mean((NDExpr)af * bf)).GetSingle(0);
            float uf = np.mean(af * bf).GetSingle(0);
            return $"fused=0x{BitConverter.SingleToInt32Bits(rf):X8} unfused=0x{BitConverter.SingleToInt32Bits(uf):X8} ulp={UlpDiffF32(rf, uf)}";
        }));
        P("A9b", "Sum(a*b) f64 fused vs np.sum(a*b)", Try(() =>
        {
            double rd = np.evaluate(NDExpr.Sum((NDExpr)A * B)).GetDouble(0);
            double ud = np.sum(A * B).GetDouble(0);
            return $"fused={rd:R} unfused={ud:R} ulp={UlpDiffF64(rd, ud)}";
        }));
        P("A9c", "Sum(a*b, axis=0) 2-D fused vs np.sum(a*b, 0)", Try(() =>
        {
            var A2 = A["0:100000"].reshape(100, 1000); var B2 = B["0:100000"].reshape(100, 1000);
            var r = np.evaluate(NDExpr.Sum((NDExpr)A2 * B2, 0)); var u = np.sum(A2 * B2, 0);
            return $"maxulp={MaxUlp(r, u)}";
        }));
    }

    // A10: where promoting to complex
    P("A10", "Where(bool, complex, f8) -> complex", Try(() => { var cond = np.array(new[] { true, false, true, false }); var r = np.evaluate(NDExpr.Where(NDExpr.Arr(cond), NDExpr.Arr(z), NDExpr.Arr(a["0:4"]))); return $"OK dtype={r.typecode} r[1]={r.GetAtIndex(1)}"; }));

    // A11: uint64 vs int64 comparison (NEP50 promotes to float64 -> exactness lost past 2^53)
    P("A11", "Greater(u8[2^63+1], i8[2^63-1])  [exact: True]", Try(() => { var u = np.array(new ulong[] { 9223372036854775809UL }); var s = np.array(new long[] { 9223372036854775807L }); var r = np.evaluate(NDExpr.Greater(NDExpr.Arr(u), NDExpr.Arr(s))); var un = u > s; return $"fused={r.GetAtIndex(0)} unfused={un.GetAtIndex(0)}"; }));

    // A12: literals the DSL cannot spell (ulong / decimal / Complex / Half / bool constants)
    P("A12", "u8 + 18446744073709551615UL (binds to double -> f64)", Try(() => { var u = np.array(new ulong[] { 1UL }); var r = np.evaluate((NDExpr)u + 18446744073709551615UL); return $"dtype={r.typecode} r={r.GetAtIndex(0)}  [NumPy: uint64 wraps to 0]"; }));

    // A13: 0-d inputs / empty axis reduce / keepdims flat
    P("A13", "all 0-d inputs -> result shape", Try(() => { var s1 = NDArray.Scalar(2.0); var s2 = NDArray.Scalar(3.0); var r = np.evaluate((NDExpr)s1 * s2 + 1); return $"OK ndim={r.ndim} r={r.GetDouble(0)}"; }));
    P("A13b", "Sum(a2*b2, axis=0) over shape (0,3)", Try(() => { var e = np.zeros(new Shape(0, 3)); var r = np.evaluate(NDExpr.Sum((NDExpr)e * e, 0)); return $"OK shape=({string.Join(",", r.shape)}) r[0]={r.GetDouble(0)}"; }));
    P("A13c", "Sum(a2*b2, axis=(0,1)) — tuple axis", "not expressible (int axis only)");
    P("A13d", "Sum(x, keepdims) flat", "not expressible (keepdims only on the axis overload)");

    // A14: strided/overlapping out=
    P("A14", "out= aliasing a strided view of the input (overlap)", Try(() => { var x = np.arange(10).astype(NPTypeCode.Double); np.evaluate((NDExpr)x["0:5"] * 10.0, @out: x["5:10"]); return $"OK x={x}"; }));
    P("A14b", "out= into a transposed (F) target", Try(() => { var m = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3); var t = np.zeros(new Shape(3, 2)).T; np.evaluate((NDExpr)m * 2.0, @out: t); return $"OK t={t.ToString().Replace("\n", " ")}"; }));

    // A15: divide/mod/floordiv by zero at int dtype
    P("A15", "Divide(i4, 0) / Mod(i4, 0) / FloorDivide(i4, 0)", Try(() => { var zero = np.zeros(new Shape(64), np.int32); var d = np.evaluate(NDExpr.Divide(NDExpr.Arr(i4), NDExpr.Arr(zero))); var m = np.evaluate(NDExpr.Mod(NDExpr.Arr(i4), NDExpr.Arr(zero))); var f = np.evaluate(NDExpr.FloorDivide(NDExpr.Arr(i4), NDExpr.Arr(zero))); return $"div={d.typecode}:{d.GetAtIndex(1)} mod={m.typecode}:{m.GetAtIndex(1)} fdiv={f.typecode}:{f.GetAtIndex(1)}"; }));

    // A16: Decimal
    P("A16", "Sqrt(decimal) / Exp(decimal) / Mean(decimal)", Try(() => { var s = np.evaluate(NDExpr.Sqrt(NDExpr.Arr(dec))); var e = np.evaluate(NDExpr.Exp(NDExpr.Arr(dec))); var m = np.evaluate(NDExpr.Mean(NDExpr.Arr(dec))); return $"sqrt={s.typecode}:{s.GetAtIndex(1)} exp={e.typecode}:{e.GetAtIndex(0)} mean={m.typecode}:{m.GetAtIndex(0)}"; }));

    // A17: F-order result preservation
    P("A17", "(a.T * 2) result layout (NumPy keeps F)", Try(() => { var m = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3); var r = np.evaluate((NDExpr)m.T * 2.0); var u = m.T * 2.0; return $"fused F={r.Shape.IsFContiguous} C={r.Shape.IsContiguous} | unfused F={u.Shape.IsFContiguous} C={u.Shape.IsContiguous}"; }));

    // A18: predicates / logical on bool
    P("A18", "LogicalNot(bool) / IsNaN(bool) / Abs(bool)", Try(() => { var bb = np.array(new[] { true, false }); var l = np.evaluate(NDExpr.LogicalNot(NDExpr.Arr(bb))); var n = np.evaluate(NDExpr.IsNaN(NDExpr.Arr(bb))); var ab = np.evaluate(NDExpr.Abs(NDExpr.Arr(bb))); return $"lnot={l.typecode}:{l.GetAtIndex(0)} isnan={n.typecode}:{n.GetAtIndex(0)} abs={ab.typecode}:{ab.GetAtIndex(0)}"; }));

    // A19: comparison output written to a non-bool out (ufunc rule: bool casts same_kind to any numeric)
    P("A19", "Greater(a,b) into out=f8", Try(() => { var o = np.zeros(new Shape(64)); var r = np.evaluate(NDExpr.Greater(NDExpr.Arr(a), NDExpr.Arr(b)), @out: o); return $"OK dtype={r.typecode} r[0]={r.GetDouble(0)}"; }));

    // A20: mixed dtype where + typed compare
    P("A20", "Where(i4>0, f4, i4) dtype  [NumPy: f8]", Try(() => { var r = np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(i4), 0), NDExpr.Arr(f4a), NDExpr.Arr(i4))); return $"OK dtype={r.typecode}"; }));

    // A21: reduction over a broadcast (stride-0) input
    P("A21", "Sum(broadcast_to(x,(1000,64)) * b)", Try(() => { var bx = np.broadcast_to(a, new Shape(1000, 64)); var bb = np.broadcast_to(b, new Shape(1000, 64)); var r = np.evaluate(NDExpr.Sum((NDExpr)bx * bb)); var u = np.sum(bx * bb); return $"fused={r.GetDouble(0):R} unfused={u.GetDouble(0):R} ulp={UlpDiffF64(r.GetDouble(0), u.GetDouble(0))}"; }));

    // A22: Min/Max binary NaN policy vs np.maximum
    P("A22", "Max(a_with_nan, b) vs np.maximum", Try(() => { var an = a.copy(); an[3] = double.NaN; var r = np.evaluate(NDExpr.Max(NDExpr.Arr(an), NDExpr.Arr(b))); var u = np.maximum(an, b); return BitEqual(r, u) ? "OK bit-equal" : $"DIVERGES r[3]={r.GetDouble(3)} u[3]={u.GetDouble(3)}"; }));

    // A23: ConstNode + Boolean adoption
    P("A23", "bool_arr + 1 (NumPy: int64)", Try(() => { var bb = np.array(new[] { true, false }); var r = np.evaluate((NDExpr)bb + 1); return $"OK dtype={r.typecode} r={r}"; }));
    P("A23b", "bool_arr * 2.5 (NumPy: f64)", Try(() => { var bb = np.array(new[] { true, false }); var r = np.evaluate((NDExpr)bb * 2.5); return $"OK dtype={r.typecode}"; }));

    // A24: the engine used
    P("A24", "np.evaluate resolves BackendFactory.GetEngine(), not the operands' engine", "documented gap (ARCHITECTURE.md P2)");
}

// =============================================================================
// B. dtype × op support sweep
// =============================================================================
if (sections.Contains('B'))
{
    Console.WriteLine("\n=== B. dtype × op support sweep (OK:<result dtype> | X:<exception>) ===");
    var codes = new[]
    {
        NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
        NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
        NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex,
    };
    var shapes = new (string name, Func<NDArray, NDExpr> build)[]
    {
        ("a*a+1", x => (NDExpr)x * x + 1),
        ("a/2", x => (NDExpr)x / 2),
        ("Sqrt", x => NDExpr.Sqrt(NDExpr.Arr(x))),
        ("Exp", x => NDExpr.Exp(NDExpr.Arr(x))),
        ("Abs", x => NDExpr.Abs(NDExpr.Arr(x))),
        ("Negate", x => NDExpr.Negate(NDExpr.Arr(x))),
        ("Floor", x => NDExpr.Floor(NDExpr.Arr(x))),
        ("a>3", x => NDExpr.Greater(NDExpr.Arr(x), 3)),
        ("Where(a>3,a,a*2)", x => NDExpr.Where(NDExpr.Greater(NDExpr.Arr(x), 3), NDExpr.Arr(x), (NDExpr)x * 2)),
        ("Max(a,a*2)", x => NDExpr.Max(NDExpr.Arr(x), (NDExpr)x * 2)),
        ("IsNaN", x => NDExpr.IsNaN(NDExpr.Arr(x))),
        ("a&a", x => (NDExpr)x & x),
        ("a%3", x => (NDExpr)x % 3),
        ("Pow(a,2)", x => NDExpr.Power(NDExpr.Arr(x), 2)),
        ("Sum", x => NDExpr.Sum(NDExpr.Arr(x))),
        ("Min", x => NDExpr.Min(NDExpr.Arr(x))),
        ("Mean", x => NDExpr.Mean(NDExpr.Arr(x))),
        ("Sum(axis0)", x => NDExpr.Sum(NDExpr.Arr(x.reshape(4, 4)), 0)),
    };
    Console.Write($"{"dtype",-8}");
    foreach (var s in shapes) Console.Write($"|{s.name,-17}");
    Console.WriteLine();
    foreach (var tc in codes)
    {
        NDArray x;
        try { x = np.arange(1, 17).astype(tc); }
        catch (Exception e) { Console.WriteLine($"{tc,-8}| astype failed: {e.GetType().Name}"); continue; }
        Console.Write($"{tc,-8}");
        foreach (var s in shapes)
        {
            string cell;
            try
            {
                var r = np.evaluate(s.build(x));
                cell = "OK:" + r.typecode.AsNumpyDtypeName();
            }
            catch (Exception e) { cell = "X:" + e.GetType().Name.Replace("Exception", "").Replace("NotSupported", "NSE").Replace("InvalidOperation", "IOE").Replace("InvalidProgram", "IPE").Replace("Argument", "Arg"); }
            Console.Write($"|{Head(cell, 17),-17}");
        }
        Console.WriteLine();
    }
}

// =============================================================================
// C. fused vs unfused, ms per call
// =============================================================================
if (sections.Contains('C'))
{
    Console.WriteLine("\n=== C. fused vs unfused (ms/call, best-of); twin: numpy_twins.py C ===");
    Console.WriteLine("case\tN\tfused_ms\tunfused_ms\tunfused/fused");
    foreach (long n in new long[] { 1_000, 100_000, 4_000_000 })
    {
        var (a, b, c) = Data(n);
        var af = a.astype(NPTypeCode.Single); var bf = b.astype(NPTypeCode.Single);
        var f2a = a.astype(NPTypeCode.Half); var f2b = b.astype(NPTypeCode.Half);
        var i4 = (np.arange(n) % 1000).astype(NPTypeCode.Int32);
        var i8 = np.arange(n) % 1000;
        (long rows, long cols) = n switch { 1_000 => (10, 100), 100_000 => (100, 1000), _ => (2000, 2000) };
        var a2 = a.reshape(rows, cols); var b2 = b.reshape(rows, cols);
        var half = n / 2;
        var aS = a[$"0:{2 * half}:2"]; var bS = b[$"0:{2 * half}:2"]; var cS = c[$"0:{2 * half}:2"];

        var cases = new (string id, Action fused, Action unfused)[]
        {
            ("a*b+c f64",          () => np.evaluate((NDExpr)a * b + c),                                              () => { var _ = a * b + c; }),
            ("(a-b)/(a+b) f64",    () => np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)),                    () => { var _ = (a - b) / (a + b); }),
            ("sqrt(a*a+b*b) f64",  () => np.evaluate(NDExpr.Sqrt((NDExpr)a * a + (NDExpr)b * b)),                    () => { var _ = np.sqrt(a * a + b * b); }),
            ("where(a>b,a,b) f64", () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b))), () => { var _ = np.where(a > b, a, b); }),
            ("maximum(a,b) f64",   () => np.evaluate(NDExpr.Max(NDExpr.Arr(a), NDExpr.Arr(b))),                     () => { var _ = np.maximum(a, b); }),
            ("a>0.5 (bool out)",   () => np.evaluate(NDExpr.Greater(NDExpr.Arr(a), 0.5)),                            () => { var _ = a > 0.5; }),
            ("(a>0.2)&(b<0.8)",    () => np.evaluate(NDExpr.Greater(NDExpr.Arr(a), 0.2) & NDExpr.Less(NDExpr.Arr(b), 0.8)), () => { var _ = (a > 0.2) & (b < 0.8); }),
            ("leaky relu f64",     () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), 0.5), NDExpr.Arr(a), (NDExpr)a * 0.01)), () => { var _ = np.where(a > 0.5, a, a * 0.01); }),
            ("exp(a)*b f64",       () => np.evaluate(NDExpr.Exp(NDExpr.Arr(a)) * b),                                 () => { var _ = np.exp(a) * b; }),
            ("exp(af)*bf f32",     () => np.evaluate(NDExpr.Exp(NDExpr.Arr(af)) * bf),                               () => { var _ = np.exp(af) * bf; }),
            ("sin(a)*cos(a) f64",  () => np.evaluate(NDExpr.Sin(NDExpr.Arr(a)) * NDExpr.Cos(NDExpr.Arr(a))),        () => { var _ = np.sin(a) * np.cos(a); }),
            ("i4*2+f8 mixed",      () => np.evaluate((NDExpr)i4 * 2 + c),                                             () => { var _ = i4 * 2 + c; }),
            ("i8*i8+1 int64",      () => np.evaluate((NDExpr)i8 * i8 + 1),                                            () => { var _ = i8 * i8 + 1; }),
            ("f2*f2+f2 half",      () => np.evaluate((NDExpr)f2a * f2b + f2a),                                        () => { var _ = f2a * f2b + f2a; }),
            ("abs(a) f64",         () => np.evaluate(NDExpr.Abs(NDExpr.Arr(a))),                                     () => { var _ = np.abs(a); }),
            ("strided a*b+c",      () => np.evaluate((NDExpr)aS * bS + cS),                                           () => { var _ = aS * bS + cS; }),
            ("sum(a*b) f64",       () => np.evaluate(NDExpr.Sum((NDExpr)a * b)),                                     () => { var _ = np.sum(a * b); }),
            ("sum(af*bf) f32",     () => np.evaluate(NDExpr.Sum((NDExpr)af * bf)),                                   () => { var _ = np.sum(af * bf); }),
            ("mean((a-b)^2) f64",  () => np.evaluate(NDExpr.Mean((NDExpr.Arr(a) - b) * (NDExpr.Arr(a) - b))),        () => { var _ = np.mean((a - b) * (a - b)); }),
            ("max(a*b) f64",       () => np.evaluate(NDExpr.Max((NDExpr)a * b)),                                     () => { var _ = np.max(a * b); }),
            ("sum(a2*b2,ax0)",     () => np.evaluate(NDExpr.Sum((NDExpr)a2 * b2, 0)),                                () => { var _ = np.sum(a2 * b2, 0); }),
            ("sum(a2*b2,ax1)",     () => np.evaluate(NDExpr.Sum((NDExpr)a2 * b2, 1)),                                () => { var _ = np.sum(a2 * b2, 1); }),
        };
        foreach (var (id, fused, unfused) in cases)
        {
            string line;
            try
            {
                fused(); unfused();
                double tf = BestMs(fused); double tu = BestMs(unfused);
                line = $"{id}\t{n}\t{tf:F4}\t{tu:F4}\t{tu / tf:F2}";
            }
            catch (Exception e) { line = $"{id}\t{n}\tERR\t{e.GetType().Name}: {Head(e.Message, 80)}"; }
            Console.WriteLine(line);
        }
    }
}

// =============================================================================
// D. fixed cost per call (n = 8), compile time
// =============================================================================
if (sections.Contains('D'))
{
    Console.WriteLine("\n=== D. fixed cost (n=8): ns/call, managed B/call; twin: numpy_twins.py D ===");
    var (a, b, c) = Data(8);
    var expr = (NDExpr)a * b + c;              // prebuilt tree
    var o = np.empty_like(a);
    Console.WriteLine("case\tns\tB/call");
    void Row(string id, Action body) => Console.WriteLine($"{id}\t{BestNs(body):F0}\t{AllocPerCall(body, 20_000)}");
    Row("evaluate(a*b+c) rebuild tree", () => np.evaluate((NDExpr)a * b + c));
    Row("evaluate(expr) prebuilt", () => np.evaluate(expr));
    Row("evaluate(expr, out=)", () => np.evaluate(expr, @out: o));
    Row("unfused a*b+c", () => { var _ = a * b + c; });
    Row("unfused np.multiply/add out= (2 passes)", () => { np.multiply(a, b, @out: o); np.add(o, c, @out: o); });
    Row("evaluate(Sum(a*b))", () => np.evaluate(NDExpr.Sum((NDExpr)a * b)));
    Row("unfused np.sum(a*b)", () => { var _ = np.sum(a * b); });
    Row("evaluate(Where(a>b,a,b))", () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b))));
    Row("unfused np.where(a>b,a,b)", () => { var _ = np.where(a > b, a, b); });

    Console.WriteLine("\nkernel compile (first call of a never-seen tree shape/constant), ms:");
    for (int k = 0; k < 4; k++)
    {
        double konst = 1000.5 + k;                 // unique constant -> unique cache key -> fresh DynamicMethod
        var sw = Stopwatch.StartNew();
        np.evaluate((NDExpr)a * b + konst);
        sw.Stop();
        var sw2 = Stopwatch.StartNew();
        np.evaluate((NDExpr)a * b + konst);
        sw2.Stop();
        Console.WriteLine($"  a*b+{konst}: first {sw.Elapsed.TotalMilliseconds:F2} ms, second {sw2.Elapsed.TotalMilliseconds * 1000:F1} us");
    }
    {
        // a wide tree: degree-12 Horner polynomial (13 constants, 12 mul, 12 add)
        NDExpr Horner(NDExpr x, int deg) { NDExpr acc = 1.0 + deg; for (int i = deg - 1; i >= 0; i--) acc = acc * x + (1.0 + i + 0.25); return acc; }
        var sw = Stopwatch.StartNew(); np.evaluate(Horner(NDExpr.Arr(a), 12)); sw.Stop();
        Console.WriteLine($"  horner deg-12 first call: {sw.Elapsed.TotalMilliseconds:F2} ms");
    }
}

// =============================================================================
// E. cache behaviour
// =============================================================================
if (sections.Contains('E'))
{
    Console.WriteLine("\n=== E. kernel cache ===");
    var (a, b, c) = Data(16);
    int before = GeneratedDelegates.InnerLoopCount;
    np.evaluate((NDExpr)a * b + c);
    int after1 = GeneratedDelegates.InnerLoopCount;
    var (a2, b2, c2) = Data(32);
    np.evaluate((NDExpr)a2 * b2 + c2);            // same structure, different arrays/size
    int after2 = GeneratedDelegates.InnerLoopCount;
    np.evaluate((NDExpr)a2 * b2 + c2.astype(NPTypeCode.Single)); // different dtype signature
    int after3 = GeneratedDelegates.InnerLoopCount;
    np.evaluate((NDExpr)a2 * b2 + 2.0);           // literal instead of array: new kernel (constant baked in)
    np.evaluate((NDExpr)a2 * b2 + 3.0);
    int after4 = GeneratedDelegates.InnerLoopCount;
    Console.WriteLine($"  kernels: start {before} → a*b+c {after1} → same shape/other arrays {after2} → f4 operand {after3} → two literal variants {after4}");
    Console.WriteLine($"  (a*b+k with k in a loop compiles one kernel PER DISTINCT k: literal values are baked into the IL and the cache key)");
}

// =============================================================================
// F. 2-D operand layouts (the fusion gate's C/F/T/strided/bcast sweep, plus unary-only and
//    same-op controls so a layout cliff can be attributed to evaluate rather than to the engine)
// =============================================================================
if (sections.Contains('F'))
{
    Console.WriteLine("\n=== F. 2-D layouts, a*b+c f64 2000x2000 (ms/call); twin: numpy_twins.py F ===");
    const int LR = 2000, LC = 2000;
    var (a0, b0, c0) = Data(LR * LC);
    var a2 = a0.reshape(LR, LC); var b2 = b0.reshape(LR, LC); var c2 = c0.reshape(LR, LC);
    NDArray Lay(NDArray x, string l) => l switch
    {
        "C" => x, "F" => x.copy(order: 'F'), "T" => x.T,
        "strided" => x[":, ::2"],
        "rows2" => x["::2, :"],
        "bcast" => np.broadcast_to(x["0:1, :"], new Shape(LR, LC)),
        _ => throw new Exception(l),
    };
    Console.WriteLine("layout\tallStrictF\tfused_ms\tunfused_ms\tunfused/fused\tfused_mul_only_ms\tunfused_mul_only_ms");
    foreach (var l in new[] { "C", "F", "T", "strided", "rows2", "bcast" })
    {
        var al = Lay(a2, l); var bl = Lay(b2, l); var cl = Lay(c2, l);
        bool allF = DefaultEngine.AreAllInputsStrictFContig(new[] { al, bl, cl }, al.Shape);
        string line;
        try
        {
            np.evaluate((NDExpr)al * bl + cl);
            double tf = BestMs(() => np.evaluate((NDExpr)al * bl + cl));
            double tu = BestMs(() => { var _ = al * bl + cl; });
            double tf1 = BestMs(() => np.evaluate((NDExpr)al * bl));
            double tu1 = BestMs(() => { var _ = al * bl; });
            line = $"{l}\t{allF}\t{tf:F3}\t{tu:F3}\t{tu / tf:F2}\t{tf1:F3}\t{tu1:F3}";
        }
        catch (Exception e) { line = $"{l}\t{allF}\tERR {e.GetType().Name}: {Head(e.Message, 80)}"; }
        Console.WriteLine(line);
    }
    // Result layout the fused path produces per input layout (NumPy keeps K-order: F inputs -> F result).
    foreach (var l in new[] { "F", "T" })
    {
        var al = Lay(a2, l); var bl = Lay(b2, l); var cl = Lay(c2, l);
        var r = np.evaluate((NDExpr)al * bl + cl); var u = al * bl + cl;
        Console.WriteLine($"  [{l}] fused result F={r.Shape.IsFContiguous} C={r.Shape.IsContiguous} | unfused F={u.Shape.IsFContiguous} C={u.Shape.IsContiguous}");
    }
}

Console.WriteLine("\ndone.");
