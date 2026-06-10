#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// Variation-grid probe — production np.* routing vs NumPy across the layout/
// dispatch variation classes (docs/NPYITER_GAPS_AND_ROADMAP.md gates on this).
// Run: dotnet run -c Release - < benchmark/poc/variation_probe.cs
// Pair:                python benchmark/poc/variation_probe.py
// =============================================================================
using System.Diagnostics;
using System.Reflection;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("!! DEBUG BUILD — INVALID. use dotnet run -c Release"); return; }

const int ROUNDS = 7;
double Med(Func<double> f)
{
    var r = new double[ROUNDS];
    for (int i = 0; i < ROUNDS; i++) r[i] = f();
    Array.Sort(r);
    return r[ROUNDS / 2];
}
double T(Action f, int iters, int warm)
{
    for (int i = 0; i < warm; i++) f();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) f();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / iters;
}
void P(string name, double ms) => Console.WriteLine($"{name,-46} {ms,9:F3} ms");

int N = 4_000_000;
int side = 2000;

unsafe {
// ---------- operands ----------
var aC = np.arange(N).astype(np.float32) + 1f;                       // contig 1-D
var bC = np.arange(N).astype(np.float32) + 2f;
var a2 = (np.arange(side * side).astype(np.float32) + 1f).reshape(side, side);
var b2 = (np.arange(side * side).astype(np.float32) + 2f).reshape(side, side);
var row = (np.arange(side).astype(np.float32) + 3f).reshape(1, side);
var col = (np.arange(side).astype(np.float32) + 3f).reshape(side, 1);
var aF = a2.copy('F'); var bF = b2.copy('F');
var i32 = np.arange(N).astype(np.int32);
var f64 = np.arange(N).astype(np.float64) + 1.0;
var wide = np.arange(2 * N).astype(np.float32) + 1f;
var sa = wide["::2"];
var cond = (np.arange(N) % 3).astype(np.@bool);  // ~1/3 true
var x1 = aC; var y1 = bC;
var small1 = np.arange(1000).astype(np.float32) + 1f;
var small2 = np.arange(1000).astype(np.float32) + 2f;
var a5d = (np.arange(N).astype(np.float32) + 1f).reshape(10, 10, 10, 10, 400);

// ---------- correctness spot checks ----------
{
    var r = a2 + row;  // row broadcast
    if (r.GetSingle(5, 7) != a2.GetSingle(5, 7) + row.GetSingle(0, 7)) Console.WriteLine("!! row broadcast WRONG");
    r = a2 + col;
    if (r.GetSingle(5, 7) != a2.GetSingle(5, 7) + col.GetSingle(5, 0)) Console.WriteLine("!! col broadcast WRONG");
    var rev = aC["::-1"];
    var rr = np.sqrt(rev);
    if (Math.Abs(rr.GetSingle(0) - MathF.Sqrt(aC.GetSingle(N - 1))) > 1e-5) Console.WriteLine("!! neg-stride sqrt WRONG");
    var mix = i32 + f64;
    if (mix.typecode != NPTypeCode.Double) Console.WriteLine($"!! mixed dtype promotion WRONG: {mix.typecode}");
    if (Math.Abs(mix.GetDouble(7) - (7 + 8.0)) > 1e-12) Console.WriteLine("!! mixed dtype value WRONG");
}

// ---------- overlap-hazard correctness probe (write-ahead direction) ----------
// NumPy ufunc semantics: COPY_IF_OVERLAP + OVERLAP_ASSUME_ELEMENTWISE on all
// operands (ufunc_object.c:1070). Expected: out[i] = 2*original a[i] — no
// cascade. Before Wave 1.1 (no COPY_IF_OVERLAP implementation) this printed
// the corrupted [1, 2, 4, 6, 8, 16, 32, 64].
{
    var ov = np.arange(8).astype(np.float64) + 1.0;   // [1..8]
    var src = ov[":-1"]; var dst = ov["1:"];
    var elw = NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
    // NOTE: the write-back to the user array resolves at Dispose (NumPy's
    // WRITEBACKIFCOPY resolves at NpyIter_Deallocate) — check AFTER the using.
    using (var iter = NpyIterRef.MultiNew(3, new[] { src, src, dst },
        NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
        new[] { NpyIterPerOpFlags.READONLY | elw, NpyIterPerOpFlags.READONLY | elw, NpyIterPerOpFlags.WRITEONLY | elw }))
    {
        iter.ExecuteBinary(NumSharp.Backends.Kernels.BinaryOp.Add);
    }
    bool ovOk = ov.GetDouble(7) == 14.0 && ov.GetDouble(5) == 10.0;
    Console.WriteLine($"overlap probe (numpy: 1,2,4,6,8,10,12,14): {ov.ToString().Replace("\n", " ")}  {(ovOk ? "PASS" : "FAIL")}");
}

Console.WriteLine();
Console.WriteLine($"probe                                          NumSharp");
Console.WriteLine(new string('-', 60));

P("P1  contig binary  a+b f32 4M",            Med(() => T(() => { var _ = aC + bC; }, 30, 10)));
P("P2  row broadcast  (2k,2k)+(1,2k)",        Med(() => T(() => { var _ = a2 + row; }, 30, 10)));
P("P3  col broadcast  (2k,2k)+(2k,1)",        Med(() => T(() => { var _ = a2 + col; }, 30, 10)));
P("P4  scalar broadcast a+5",                 Med(() => T(() => { var _ = aC + 5f; }, 30, 10)));
P("P5  neg-stride unary sqrt(a[::-1])",       Med(() => T(() => { var _ = np.sqrt(aC["::-1"]); }, 30, 10)));
P("P6  neg-stride binary a[::-1]+b[::-1]",    Med(() => T(() => { var _ = aC["::-1"] + bC["::-1"]; }, 30, 10)));
P("P7  F-order binary aF+bF",                 Med(() => T(() => { var _ = aF + bF; }, 30, 10)));
P("P8  transposed binary a.T+b.T",            Med(() => T(() => { var _ = a2.T + b2.T; }, 30, 10)));
P("P9  mixed dtype i32+f64 4M",               Med(() => T(() => { var _ = i32 + f64; }, 30, 10)));
P("P10 astype strided a[::2]->f64",           Med(() => T(() => { var _ = sa.astype(np.float64); }, 30, 10)));
P("P11 where(cond,x,y) contig 4M",            Med(() => T(() => { var _ = np.where(cond, x1, y1); }, 30, 10)));
P("P12 sum axis=0 f32 (2k,2k)",               Med(() => T(() => { var _ = np.sum(a2, axis: 0); }, 30, 10)));
P("P13 sum axis=1 f32 (2k,2k)",               Med(() => T(() => { var _ = np.sum(a2, axis: 1); }, 30, 10)));
P("P14 5-D contig unary sqrt",                Med(() => T(() => { var _ = np.sqrt(a5d); }, 30, 10)));
double us = Med(() => T(() => { var _ = small1 + small2; }, 20000, 2000)) * 1000.0;
Console.WriteLine($"{"P15 small-N binary 1K (us/call)",-46} {us,9:F3} us");
P("P16 mean f32 contig 4M",                   Med(() => T(() => { var _ = np.mean(aC); }, 30, 10)));

// ---------- the genuinely-strided shapes (the Phase 2a holes) ----------
int n1 = 1_000_000;
var w1 = np.arange(2 * n1).astype(np.float32) + 1f;
var w2 = np.arange(2 * n1).astype(np.float32) + 2f;
var ss1 = w1["::2"]; var ss2 = w2["::2"];
var big2 = (np.arange(4 * n1).astype(np.float32) + 1f).reshape(2000, 2000);
var sv2d = big2["::2, ::2"];
double s1 = Med(() => T(() => { var _ = ss1 + ss2; }, 100, 30)) * 1000.0;
double s2 = Med(() => T(() => { var _ = np.sqrt(sv2d); }, 100, 30)) * 1000.0;
double s3 = Med(() => T(() => { var _ = np.sum(ss1); }, 100, 30)) * 1000.0;
Console.WriteLine($"{"S1  strided binary a[::2]+b[::2] f32 1M",-46} {s1,9:F0} us");
Console.WriteLine($"{"S2  strided 2-D sqrt(a[::2,::2]) f32 1M",-46} {s2,9:F0} us");
Console.WriteLine($"{"S3  strided sum(a[::2]) f32 1M",-46} {s3,9:F0} us");

Console.WriteLine("[done]");
}
