#:project ../../src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// route_audit_bench.cs — which not-yet-out= families ride NpyIter, and which
// would benefit from it. Run ONLY with: dotnet run -c Release route_audit_bench.cs
// Companion: route_audit_bench.py (NumPy numbers, same shapes, same box).
//
// Families & their audited routes (2026-06-10):
//   comparisons      trivial bypass → NpyIter Tier-3B → Direct   [ON NpyIter]
//   bitwise &|^      ExecuteBinaryOp ladder (NpyIter tier)        [ON NpyIter]
//   invert + unary   ExecuteUnaryOp ladder (NpyIter tier)         [ON NpyIter]
//   shifts           hand-rolled generic loops                    [NO NpyIter]
//   maximum/minimum  broadcast_arrays + clip → Direct flat Clip   [NO NpyIter]
//   reductions       Direct axis kernels / scalar NpyAxisIter     [NO NpyIter]
//
// For the NO-NpyIter families we also time a forced-NpyIter proxy where one
// exists (np.evaluate Max node = scalar-body lower bound; a custom Tier-3B
// ExecuteElementWise max with a real Vector256.Max body = faithful bound).
// =============================================================================
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

var asm = typeof(np).Assembly;
var dbg = (System.Diagnostics.DebuggableAttribute?)Attribute.GetCustomAttribute(asm, typeof(System.Diagnostics.DebuggableAttribute));
if (dbg is { IsJITOptimizerDisabled: true })
{
    Console.WriteLine("FATAL: NumSharp.Core built Debug — rerun with -c Release");
    return;
}

const int N = 4_000_000;

double Best(Action body, int rounds = 7)
{
    body(); // warmup / kernel compile
    double best = double.MaxValue;
    for (int i = 0; i < rounds; i++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        var sw = Stopwatch.StartNew();
        body();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
    }

    return best;
}

void Row(string name, double ms) => Console.WriteLine($"  {name,-44} {ms,8:F3} ms");

// ---- data ------------------------------------------------------------------
var af = (np.arange(N).astype(np.float32) % 997f) + 1f;
var bf = (np.arange(N).astype(np.float32) % 877f) + 2f;
var afS = af["::2"]; // 2M strided
var bfS = bf["::2"];
var m2d = np.arange(4_000_000).reshape(2000, 2000).astype(np.float32);
var col = np.arange(2000).reshape(2000, 1).astype(np.float32);
var ai = np.arange(N).astype(np.int32);
var bi = (np.arange(N).astype(np.int32) % 31);
var aiS = ai["::2"];
var biS = bi["::2"];
var ad = af.astype(np.float64)["0:2000000"];
var adS = af.astype(np.float64)["::4"];   // 1M strided f64

Console.WriteLine("== A. CONTROLS — families already ON NpyIter (expect ~NumPy parity) ==");
Row("less   f32 contig 4M        [NpyIter S/B]", Best(() => { var _ = af < bf; }));
Row("less   f32 strided 2M", Best(() => { var _ = afS < bfS; }));
Row("less   f32 bcast (2k,2k)<(2k,1)", Best(() => { var _ = m2d < col; }));
Row("and    i32 contig 4M        [NpyIter S/B]", Best(() => { var _ = ai & bi; }));
Row("and    i32 strided 2M", Best(() => { var _ = aiS & biS; }));
Row("invert i32 contig 4M        [NpyIter S]", Best(() => { var _ = np.invert(ai); }));
Row("invert i32 strided 2M", Best(() => { var _ = np.invert(aiS); }));
Row("sinh   f64 contig 2M        [NpyIter scalar-body]", Best(() => { var _ = np.sinh(ad); }));
Row("sinh   f64 strided 1M", Best(() => { var _ = np.sinh(adS); }));

Console.WriteLine("\n== B. maximum — NOT on NpyIter (broadcast_arrays + Direct Clip) ==");
Row("maximum f32 contig 4M       [current: clip]", Best(() => { var _ = np.maximum(af, bf); }));
Row("maximum f32 strided 2M", Best(() => { var _ = np.maximum(afS, bfS); }));
Row("maximum f32 bcast (2k,2k),(2k,1)", Best(() => { var _ = np.maximum(m2d, col); }));

// forced-NpyIter proxy #1: np.evaluate Max node (scalar Math.Max body — lower bound)
Row("maximum via np.evaluate     [iter, scalar body]", Best(() => { var _ = np.evaluate(NpyExpr.Max(NpyExpr.Arr(af), NpyExpr.Arr(bf))); }));
Row("maximum via np.evaluate strided 2M", Best(() => { var _ = np.evaluate(NpyExpr.Max(NpyExpr.Arr(afS), NpyExpr.Arr(bfS))); }));
Row("maximum via np.evaluate bcast", Best(() => { var _ = np.evaluate(NpyExpr.Max(NpyExpr.Arr(m2d), NpyExpr.Arr(col))); }));

// forced-NpyIter proxy #2: Tier-3B ExecuteElementWise with a REAL SIMD body
// (Vector256.Max — faithful to what a migrated route would emit).
var vecMaxF32 = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
    .First(m => m.Name == "Max" && m.IsGenericMethodDefinition && m.GetParameters().Length == 2)
    .MakeGenericMethod(typeof(float));
var mathMaxF32 = typeof(MathF).GetMethod("Max", new[] { typeof(float), typeof(float) })!;
Action<ILGenerator> maxScalar = il => il.EmitCall(OpCodes.Call, mathMaxF32, null);
Action<ILGenerator> maxVector = il => il.EmitCall(OpCodes.Call, vecMaxF32, null);

unsafe NDArray MaxViaIter(NDArray x, NDArray y)
{
    var result = new NDArray(NPTypeCode.Single, np.broadcast_arrays(x, y).Item1.Shape.Clean(), false);
    using var iter = NpyIterRef.MultiNew(3, new[] { x, y, result },
        NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
        new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
    iter.ExecuteElementWiseBinary(NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
        maxScalar, maxVector, "bench_max_f32_simd");
    return result;
}

Row("maximum via Tier-3B SIMD    [iter, V256.Max]", Best(() => { var _ = MaxViaIter(af, bf); }));
Row("maximum via Tier-3B SIMD strided 2M", Best(() => { var _ = MaxViaIter(afS, bfS); }));
Row("maximum via Tier-3B SIMD bcast", Best(() => { var _ = MaxViaIter(m2d, col); }));

Console.WriteLine("\n== C. shifts — NOT on NpyIter (hand-rolled loops) ==");
Row("a<<3   i32 contig 4M        [current]", Best(() => { var _ = np.left_shift(ai, (NDArray)3); }));
Row("a<<3   i32 strided 2M", Best(() => { var _ = np.left_shift(aiS, (NDArray)3); }));
Row("a<<b   i32 contig 4M", Best(() => { var _ = np.left_shift(ai, bi); }));
Row("a<<b   i32 strided 2M", Best(() => { var _ = np.left_shift(aiS, biS); }));

// forced-NpyIter proxy: Tier-3B with Shl scalar body + Avx2 variable-shift vector body
var shlVar = typeof(System.Runtime.Intrinsics.X86.Avx2).GetMethod("ShiftLeftLogicalVariable",
    new[] { typeof(Vector256<int>), typeof(Vector256<uint>) })!;
Action<ILGenerator> shlScalar = il =>
{
    il.Emit(OpCodes.Ldc_I4, 31);
    il.Emit(OpCodes.And);
    il.Emit(OpCodes.Shl);
};
Action<ILGenerator> shlVector = il =>
{
    var cast = typeof(Vector256).GetMethods(BindingFlags.Public | BindingFlags.Static)
        .First(m => m.Name == "AsUInt32" && m.IsGenericMethodDefinition)
        .MakeGenericMethod(typeof(int));
    il.EmitCall(OpCodes.Call, cast, null);
    il.EmitCall(OpCodes.Call, shlVar, null);
};

unsafe NDArray ShiftViaIter(NDArray x, NDArray y)
{
    var result = new NDArray(NPTypeCode.Int32, np.broadcast_arrays(x, y).Item1.Shape.Clean(), false);
    using var iter = NpyIterRef.MultiNew(3, new[] { x, y, result },
        NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.COPY_IF_OVERLAP,
        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
        new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
    iter.ExecuteElementWiseBinary(NPTypeCode.Int32, NPTypeCode.Int32, NPTypeCode.Int32,
        shlScalar, shlVector, "bench_shl_i32_simd");
    return result;
}

Row("a<<b via Tier-3B SIMD       [iter, vpsllvd]", Best(() => { var _ = ShiftViaIter(ai, bi); }));
Row("a<<b via Tier-3B SIMD strided 2M", Best(() => { var _ = ShiftViaIter(aiS, biS); }));

Console.WriteLine("\n== D. reductions — NOT on NpyIter (Direct axis kernels / scalar NpyAxisIter) ==");
Row("sum    axis=0 f32 (2k,2k)   [Direct kernel — control]", Best(() => { var _ = np.sum(m2d, 0); }));
Row("cumsum axis=0 f32 (2k,2k)   [NpyAxisIter scalar]", Best(() => { var _ = np.cumsum(m2d, 0); }));
Row("cumsum axis=1 f32 (2k,2k)", Best(() => { var _ = np.cumsum(m2d, 1); }));
Row("var    axis=0 f32 (2k,2k)   [NpyAxisIter scalar]", Best(() => { var _ = np.var(m2d, 0); }));
Row("var    axis=1 f32 (2k,2k)", Best(() => { var _ = np.var(m2d, 1); }));
Row("all    axis=0 f32 (2k,2k)   [NpyAxisIter scalar]", Best(() => { var _ = np.all(m2d, 0); }));
Row("any    axis=1 f32 (2k,2k)", Best(() => { var _ = np.any(m2d, 1); }));
Console.WriteLine("[done]");
