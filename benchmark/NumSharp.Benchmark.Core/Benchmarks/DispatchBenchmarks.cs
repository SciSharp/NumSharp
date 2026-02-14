using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;

namespace NumSharp.Benchmark.Core;

/// <summary>
/// Compares different dispatch mechanisms for binary operations.
/// This benchmark answers: "How should NumSharp dispatch arithmetic operations?"
///
/// Approaches tested:
/// - Raw pointer loop (inline arithmetic, no abstraction)
/// - Static method call (current NumSharp pattern via Operator.Add)
/// - Struct with interface (C++ template-like, IBinOp&lt;T&gt;)
/// - DynamicMethod scalar (IL emission, scalar loop)
/// - DynamicMethod SIMD (IL emission with Vector256)
/// - Hand-written SIMD (static code with Vector256)
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
public unsafe class DispatchBenchmarks
{
    private int* _a;
    private int* _b;
    private int* _c;

    private Action<nint, nint, nint, int>? _dynMethodScalar;
    private Action<nint, nint, nint, int>? _dynMethodSimd;

    [Params(1_000, 100_000, 10_000_000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _a = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 32);
        _b = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 32);
        _c = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 32);

        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            _a[i] = rng.Next(100);
            _b[i] = rng.Next(100);
        }

        // Pre-compile DynamicMethods
        _dynMethodScalar = EmitScalarPtrLoop().CreateDelegate<Action<nint, nint, nint, int>>();
        _dynMethodSimd = EmitSimdPtrLoop().CreateDelegate<Action<nint, nint, nint, int>>();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree(_a);
        NativeMemory.AlignedFree(_b);
        NativeMemory.AlignedFree(_c);
    }

    // ========================================================================
    // Benchmark Methods
    // ========================================================================

    [Benchmark(Description = "Raw ptr loop (inline a+b)")]
    [BenchmarkCategory("Baseline")]
    public void RawPtrLoop()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        for (int i = 0; i < n; i++)
            c[i] = a[i] + b[i];
    }

    [Benchmark(Baseline = true, Description = "Static Op.Add() [NumSharp current]")]
    [BenchmarkCategory("Current")]
    public void StaticOpAdd()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        for (int i = 0; i < n; i++)
            c[i] = Op.Add(a[i], b[i]);
    }

    [Benchmark(Description = "Struct<AddOp> via IBinOp<T>")]
    [BenchmarkCategory("Alternative")]
    public void StructDispatch()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        AddOp op = default;
        for (int i = 0; i < n; i++)
            c[i] = op.Execute(a[i], b[i]);
    }

    [Benchmark(Description = "DynamicMethod scalar ptr")]
    [BenchmarkCategory("Proposed")]
    public void DynMethodScalar()
    {
        _dynMethodScalar!((nint)_a, (nint)_b, (nint)_c, N);
    }

    [Benchmark(Description = "DynamicMethod SIMD (Vector256)")]
    [BenchmarkCategory("Proposed")]
    public void DynMethodSimd()
    {
        _dynMethodSimd!((nint)_a, (nint)_b, (nint)_c, N);
    }

    [Benchmark(Description = "Hand-written SIMD (Vector256)")]
    [BenchmarkCategory("Reference")]
    public void HandWrittenSimd()
    {
        var a = _a;
        var b = _b;
        var c = _c;
        var n = N;
        int i = 0;
        for (; i <= n - 8; i += 8)
            Vector256.Store(Vector256.Load(a + i) + Vector256.Load(b + i), c + i);
        for (; i < n; i++)
            c[i] = a[i] + b[i];
    }

    // ========================================================================
    // Support Types
    // ========================================================================

    private static class Op
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Add(int a, int b) => a + b;
    }

    private interface IBinOp<T> { T Execute(T a, T b); }

    private struct AddOp : IBinOp<int>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Execute(int a, int b) => a + b;
    }

    // ========================================================================
    // DynamicMethod Builders
    // ========================================================================

    private static DynamicMethod EmitScalarPtrLoop()
    {
        var dm = new DynamicMethod("ScalarPtr", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)],
            typeof(DispatchBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int)); // i

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        // c[i] = a[i] + b[i]
        EmitPtrOffset(il, 2); // &c[i]
        EmitPtrLoad(il, 0);   // a[i]
        EmitPtrLoad(il, 1);   // b[i]
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stind_I4);

        // i++
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Blt, top);

        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static DynamicMethod EmitSimdPtrLoop()
    {
        var loadM = typeof(Vector256).GetMethods()
            .First(m => m.Name == "Load" && m.IsGenericMethod)
            .MakeGenericMethod(typeof(int));
        var storeM = typeof(Vector256).GetMethods()
            .First(m => m.Name == "Store" && m.IsGenericMethod && m.GetParameters().Length == 2)
            .MakeGenericMethod(typeof(int));
        var addVecM = typeof(Vector256<int>).GetMethods()
            .First(m => m.Name == "op_Addition"
                && m.GetParameters()[0].ParameterType == typeof(Vector256<int>));

        var dm = new DynamicMethod("SimdPtr", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)],
            typeof(DispatchBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var sTop = il.DefineLabel();
        var sChk = il.DefineLabel();
        var tTop = il.DefineLabel();
        var tChk = il.DefineLabel();
        il.DeclareLocal(typeof(int)); // i
        il.DeclareLocal(typeof(int)); // simdEnd

        // simdEnd = n - 8
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Ldc_I4_8);
        il.Emit(OpCodes.Sub);
        il.Emit(OpCodes.Stloc_1);

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, sChk);

        // SIMD loop
        il.MarkLabel(sTop);
        EmitPtrOffset(il, 0);
        il.EmitCall(OpCodes.Call, loadM, null);
        EmitPtrOffset(il, 1);
        il.EmitCall(OpCodes.Call, loadM, null);
        il.EmitCall(OpCodes.Call, addVecM, null);
        EmitPtrOffset(il, 2);
        il.EmitCall(OpCodes.Call, storeM, null);

        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4_8);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);

        il.MarkLabel(sChk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ble, sTop);

        // Scalar tail
        il.Emit(OpCodes.Br, tChk);
        il.MarkLabel(tTop);
        EmitPtrOffset(il, 2);
        EmitPtrLoad(il, 0);
        EmitPtrLoad(il, 1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stind_I4);

        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);

        il.MarkLabel(tChk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Blt, tTop);

        il.Emit(OpCodes.Ret);
        return dm;
    }

    private static void EmitPtrOffset(ILGenerator il, int argIndex)
    {
        il.Emit(OpCodes.Ldarg, argIndex);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Conv_I);
        il.Emit(OpCodes.Ldc_I4_4);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Add);
    }

    private static void EmitPtrLoad(ILGenerator il, int argIndex)
    {
        EmitPtrOffset(il, argIndex);
        il.Emit(OpCodes.Ldind_I4);
    }
}
