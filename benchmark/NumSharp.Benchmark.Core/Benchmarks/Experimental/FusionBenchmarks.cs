using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;

namespace NumSharp.Benchmark.Core.Experimental;

/// <summary>
/// Compares multi-pass (current NumSharp) vs fused kernel approaches.
/// This benchmark answers: "How much faster can compound expressions be with kernel fusion?"
///
/// Patterns tested:
/// - Pattern 1: c = a * a (DUP optimization)
/// - Pattern 2: c = a*a + 2*b (DUP + constant baking)
/// - Pattern 3: variance = sum((a-mean)²)/N (fused reduction)
/// - Pattern 4: c = a³ + a² + a (STLOC/LDLOC for 3+ uses)
/// - Pattern 5: c = sqrt(a² + b²) (multi-input DAG)
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
public unsafe class FusionBenchmarks
{
    private double* _a;
    private double* _b;
    private double* _c;
    private double* _t1;
    private double* _t2;
    private double* _t3;
    private double _mean;

    // Pre-compiled fused kernels
    private Action<nint, nint, int>? _fusedSquare;
    private Action<nint, nint, nint, int>? _fusedAaBb;
    private Func<nint, int, double>? _fusedVariance;
    private Action<nint, nint, int>? _fusedPolynomial;
    private Action<nint, nint, nint, int>? _fusedEuclidean;

    [Params(10_000_000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _a = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _b = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _c = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _t1 = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _t2 = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _t3 = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);

        var rng = new Random(42);
        double sum = 0;
        for (int i = 0; i < N; i++)
        {
            _a[i] = rng.NextDouble() * 10;
            _b[i] = rng.NextDouble() * 10;
            sum += _a[i];
        }
        _mean = sum / N;

        // Pre-compile all fused kernels
        _fusedSquare = EmitFusedSquare().CreateDelegate<Action<nint, nint, int>>();
        _fusedAaBb = EmitFusedAaBb().CreateDelegate<Action<nint, nint, nint, int>>();
        _fusedVariance = EmitFusedVariance(_mean).CreateDelegate<Func<nint, int, double>>();
        _fusedPolynomial = EmitFusedPolynomial().CreateDelegate<Action<nint, nint, int>>();
        _fusedEuclidean = EmitFusedEuclidean().CreateDelegate<Action<nint, nint, nint, int>>();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree(_a);
        NativeMemory.AlignedFree(_b);
        NativeMemory.AlignedFree(_c);
        NativeMemory.AlignedFree(_t1);
        NativeMemory.AlignedFree(_t2);
        NativeMemory.AlignedFree(_t3);
    }

    // ========================================================================
    // Pattern 1: c = a * a
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Multi-pass: t1=a*a, c=t1")]
    [BenchmarkCategory("Pattern1_Square")]
    public void Pattern1_MultiPass()
    {
        var a = _a;
        var c = _c;
        var n = N;
        // Two memory reads per element (compiler may or may not optimize)
        for (int i = 0; i < n; i++)
            c[i] = a[i] * a[i];
    }

    [Benchmark(Description = "Fused DynMethod (DUP)")]
    [BenchmarkCategory("Pattern1_Square")]
    public void Pattern1_Fused()
    {
        _fusedSquare!((nint)_a, (nint)_c, N);
    }

    // ========================================================================
    // Pattern 2: c = a*a + 2*b
    // ========================================================================

    [Benchmark(Baseline = true, Description = "3-pass: t1=a*a, t2=2*b, c=t1+t2")]
    [BenchmarkCategory("Pattern2_AaBb")]
    public void Pattern2_MultiPass()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var t1 = _t1;
        var t2 = _t2;
        var n = N;

        for (int i = 0; i < n; i++) t1[i] = a[i] * a[i];
        for (int i = 0; i < n; i++) t2[i] = 2.0 * b[i];
        for (int i = 0; i < n; i++) c[i] = t1[i] + t2[i];
    }

    [Benchmark(Description = "Single raw loop")]
    [BenchmarkCategory("Pattern2_AaBb")]
    public void Pattern2_SingleLoop()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        for (int i = 0; i < n; i++)
            c[i] = a[i] * a[i] + 2.0 * b[i];
    }

    [Benchmark(Description = "Fused DynMethod (DUP + const)")]
    [BenchmarkCategory("Pattern2_AaBb")]
    public void Pattern2_Fused()
    {
        _fusedAaBb!((nint)_a, (nint)_b, (nint)_c, N);
    }

    // ========================================================================
    // Pattern 3: variance = sum((a - mean)²) / N
    // ========================================================================

    [Benchmark(Baseline = true, Description = "3-pass: t1=a-mean, t2=t1², sum(t2)/N")]
    [BenchmarkCategory("Pattern3_Variance")]
    public double Pattern3_MultiPass()
    {
        var a = _a;
        var t1 = _t1;
        var t2 = _t2;
        var mean = _mean;
        var n = N;

        for (int i = 0; i < n; i++) t1[i] = a[i] - mean;
        for (int i = 0; i < n; i++) t2[i] = t1[i] * t1[i];
        double sum = 0;
        for (int i = 0; i < n; i++) sum += t2[i];
        return sum / n;
    }

    [Benchmark(Description = "Single raw loop")]
    [BenchmarkCategory("Pattern3_Variance")]
    public double Pattern3_SingleLoop()
    {
        var a = _a;
        var mean = _mean;
        var n = N;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            var diff = a[i] - mean;
            sum += diff * diff;
        }
        return sum / n;
    }

    [Benchmark(Description = "Fused DynMethod (baked mean + reduction)")]
    [BenchmarkCategory("Pattern3_Variance")]
    public double Pattern3_Fused()
    {
        return _fusedVariance!((nint)_a, N) / N;
    }

    // ========================================================================
    // Pattern 4: c = a³ + a² + a
    // ========================================================================

    [Benchmark(Baseline = true, Description = "4-pass: t1=a³, t2=a², t3=t1+t2, c=t3+a")]
    [BenchmarkCategory("Pattern4_Polynomial")]
    public void Pattern4_MultiPass()
    {
        var a = _a;
        var c = _c;
        var t1 = _t1;
        var t2 = _t2;
        var t3 = _t3;
        var n = N;

        for (int i = 0; i < n; i++) t1[i] = a[i] * a[i] * a[i];
        for (int i = 0; i < n; i++) t2[i] = a[i] * a[i];
        for (int i = 0; i < n; i++) t3[i] = t1[i] + t2[i];
        for (int i = 0; i < n; i++) c[i] = t3[i] + a[i];
    }

    [Benchmark(Description = "Single raw loop")]
    [BenchmarkCategory("Pattern4_Polynomial")]
    public void Pattern4_SingleLoop()
    {
        var a = _a;
        var c = _c;
        var n = N;
        for (int i = 0; i < n; i++)
        {
            var ai = a[i];
            c[i] = ai * ai * ai + ai * ai + ai;
        }
    }

    [Benchmark(Description = "Fused DynMethod (STLOC/LDLOC)")]
    [BenchmarkCategory("Pattern4_Polynomial")]
    public void Pattern4_Fused()
    {
        _fusedPolynomial!((nint)_a, (nint)_c, N);
    }

    // ========================================================================
    // Pattern 5: c = sqrt(a² + b²)
    // ========================================================================

    [Benchmark(Baseline = true, Description = "4-pass: t1=a², t2=b², t3=t1+t2, c=sqrt(t3)")]
    [BenchmarkCategory("Pattern5_Euclidean")]
    public void Pattern5_MultiPass()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var t1 = _t1;
        var t2 = _t2;
        var t3 = _t3;
        var n = N;

        for (int i = 0; i < n; i++) t1[i] = a[i] * a[i];
        for (int i = 0; i < n; i++) t2[i] = b[i] * b[i];
        for (int i = 0; i < n; i++) t3[i] = t1[i] + t2[i];
        for (int i = 0; i < n; i++) c[i] = Math.Sqrt(t3[i]);
    }

    [Benchmark(Description = "Single raw loop")]
    [BenchmarkCategory("Pattern5_Euclidean")]
    public void Pattern5_SingleLoop()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        for (int i = 0; i < n; i++)
        {
            var ai = a[i];
            var bi = b[i];
            c[i] = Math.Sqrt(ai * ai + bi * bi);
        }
    }

    [Benchmark(Description = "Fused DynMethod (multi-input STLOC)")]
    [BenchmarkCategory("Pattern5_Euclidean")]
    public void Pattern5_Fused()
    {
        _fusedEuclidean!((nint)_a, (nint)_b, (nint)_c, N);
    }

    // ========================================================================
    // DynamicMethod Builders
    // ========================================================================

    private static DynamicMethod EmitFusedSquare()
    {
        // c[i] = a[i] * a[i] using DUP
        var dm = new DynamicMethod("FusedSquare", typeof(void),
            [typeof(nint), typeof(nint), typeof(int)], typeof(FusionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int));

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        EmitPtrOfs8(il, 1);      // &c[i]
        EmitLoad8(il, 0);        // a[i]
        il.Emit(OpCodes.Dup);    // a[i], a[i] - ONE load, DUP on stack
        il.Emit(OpCodes.Mul);    // a[i]²
        il.Emit(OpCodes.Stind_R8);
        EmitIncr(il);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Blt, top);
        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static DynamicMethod EmitFusedAaBb()
    {
        // c[i] = a[i]*a[i] + 2*b[i]
        var dm = new DynamicMethod("FusedAaBb", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)], typeof(FusionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int));

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        EmitPtrOfs8(il, 2);      // &c[i]
        EmitLoad8(il, 0);        // a[i]
        il.Emit(OpCodes.Dup);    // a[i], a[i]
        il.Emit(OpCodes.Mul);    // a[i]²
        il.Emit(OpCodes.Ldc_R8, 2.0); // constant baked into IL
        EmitLoad8(il, 1);        // b[i]
        il.Emit(OpCodes.Mul);    // 2*b[i]
        il.Emit(OpCodes.Add);    // a[i]² + 2*b[i]
        il.Emit(OpCodes.Stind_R8);
        EmitIncr(il);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Blt, top);
        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static DynamicMethod EmitFusedVariance(double mean)
    {
        // returns sum((a[i] - mean)²) with mean baked as constant
        var dm = new DynamicMethod("FusedVariance", typeof(double),
            [typeof(nint), typeof(int)], typeof(FusionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int));     // i
        il.DeclareLocal(typeof(double));  // accumulator

        il.Emit(OpCodes.Ldc_R8, 0.0);
        il.Emit(OpCodes.Stloc_1);
        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        // diff = a[i] - mean
        EmitLoad8(il, 0);           // a[i]
        il.Emit(OpCodes.Ldc_R8, mean); // mean BAKED as constant
        il.Emit(OpCodes.Sub);       // diff
        il.Emit(OpCodes.Dup);       // diff, diff
        il.Emit(OpCodes.Mul);       // diff²
        // acc += diff²
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_1);
        EmitIncr(il);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_1);
        il.Emit(OpCodes.Blt, top);

        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static DynamicMethod EmitFusedPolynomial()
    {
        // c[i] = a[i]³ + a[i]² + a[i] using STLOC/LDLOC
        var dm = new DynamicMethod("FusedPolynomial", typeof(void),
            [typeof(nint), typeof(nint), typeof(int)], typeof(FusionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int));     // i
        il.DeclareLocal(typeof(double));  // cached a[i]

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        EmitPtrOfs8(il, 1);         // &c[i]
        EmitLoad8(il, 0);           // a[i] - ONE memory read
        il.Emit(OpCodes.Stloc_1);   // save to local (becomes CPU register)

        // a[i]³
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Mul);

        // + a[i]²
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Add);

        // + a[i]
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Add);

        il.Emit(OpCodes.Stind_R8);
        EmitIncr(il);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Blt, top);
        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static DynamicMethod EmitFusedEuclidean()
    {
        // c[i] = sqrt(a[i]² + b[i]²)
        var sqrtM = typeof(Math).GetMethod("Sqrt", [typeof(double)])!;

        var dm = new DynamicMethod("FusedEuclidean", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)], typeof(FusionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int));     // i
        il.DeclareLocal(typeof(double));  // a[i]
        il.DeclareLocal(typeof(double));  // b[i]

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        EmitPtrOfs8(il, 2);         // &c[i]

        // Cache both inputs
        EmitLoad8(il, 0);           // a[i]
        il.Emit(OpCodes.Stloc_1);
        EmitLoad8(il, 1);           // b[i]
        il.Emit(OpCodes.Stloc_2);

        // a² + b²
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Ldloc_2);
        il.Emit(OpCodes.Ldloc_2);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Add);

        // sqrt
        il.EmitCall(OpCodes.Call, sqrtM, null);
        il.Emit(OpCodes.Stind_R8);
        EmitIncr(il);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Blt, top);
        il.Emit(OpCodes.Ret);
        return dm;
    }

    // ========================================================================
    // IL Helpers
    // ========================================================================

    private static void EmitPtrOfs8(ILGenerator il, int argIdx)
    {
        il.Emit(OpCodes.Ldarg, argIdx);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Conv_I);
        il.Emit(OpCodes.Ldc_I4_8);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Add);
    }

    private static void EmitLoad8(ILGenerator il, int argIdx)
    {
        EmitPtrOfs8(il, argIdx);
        il.Emit(OpCodes.Ldind_R8);
    }

    private static void EmitIncr(ILGenerator il)
    {
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);
    }
}
