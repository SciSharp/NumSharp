using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using NumSharp;

namespace NumSharp.Benchmark.GraphEngine;

/// <summary>
/// Benchmarks for GitHub Issue #544: Replace ~636K lines of generated math code with DynamicMethod IL emission.
///
/// This benchmark compares:
/// - NumSharp current (generated code via Regen templates)
/// - DynamicMethod scalar (IL-emitted pointer loops)
/// - DynamicMethod SIMD (IL-emitted Vector256 loops)
///
/// Success criteria from #544:
/// - Binary ops should be ≥2x faster than current on contiguous arrays
/// - All existing tests pass with DynamicMethod enabled
/// - Generated code files can be deleted
///
/// Related: #544 (this issue), #541 (successor - GraphEngine with fusion)
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
[GroupBenchmarksBy(BenchmarkDotNet.Configs.BenchmarkLogicalGroupRule.ByCategory)]
public unsafe class DynamicEmissionBenchmarks
{
    // ========================================================================
    // Data Arrays
    // ========================================================================

    // int32
    private int* _aInt;
    private int* _bInt;
    private int* _cInt;
    private NDArray _aIntND = null!;
    private NDArray _bIntND = null!;

    // float64
    private double* _aDouble;
    private double* _bDouble;
    private double* _cDouble;
    private NDArray _aDoubleND = null!;
    private NDArray _bDoubleND = null!;

    // Pre-compiled DynamicMethod delegates
    private Action<nint, nint, nint, int>? _dynAddInt32Scalar;
    private Action<nint, nint, nint, int>? _dynAddInt32Simd;
    private Action<nint, nint, nint, int>? _dynMulInt32Scalar;
    private Action<nint, nint, nint, int>? _dynMulInt32Simd;

    private Action<nint, nint, nint, int>? _dynAddFloat64Scalar;
    private Action<nint, nint, nint, int>? _dynAddFloat64Simd;
    private Action<nint, nint, nint, int>? _dynMulFloat64Scalar;
    private Action<nint, nint, nint, int>? _dynMulFloat64Simd;

    private Action<nint, nint, int>? _dynSqrtFloat64Scalar;

    [Params(1, 1_000, 10_000, 1_000_000, 10_000_000)]
    public int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        // Allocate native memory (aligned for SIMD)
        _aInt = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 64);
        _bInt = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 64);
        _cInt = (int*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(int)), 64);

        _aDouble = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _bDouble = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);
        _cDouble = (double*)NativeMemory.AlignedAlloc((nuint)(N * sizeof(double)), 64);

        // Initialize data
        var rng = new Random(42);
        for (int i = 0; i < N; i++)
        {
            _aInt[i] = rng.Next(1, 100);
            _bInt[i] = rng.Next(1, 100);
            _aDouble[i] = rng.NextDouble() * 10 + 0.1; // Avoid zero for sqrt
            _bDouble[i] = rng.NextDouble() * 10 + 0.1;
        }

        // Create NumSharp arrays
        np.random.seed(42);
        _aIntND = np.random.randint(1, 100, new Shape(N)).astype(np.int32);
        _bIntND = np.random.randint(1, 100, new Shape(N)).astype(np.int32);
        _aDoubleND = np.random.rand(N) * 10 + 0.1;
        _bDoubleND = np.random.rand(N) * 10 + 0.1;

        // Pre-compile DynamicMethod delegates
        _dynAddInt32Scalar = EmitBinaryScalar<int>(OpCodes.Add).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynAddInt32Simd = EmitBinarySimd<int>(AddVec<int>()).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynMulInt32Scalar = EmitBinaryScalar<int>(OpCodes.Mul).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynMulInt32Simd = EmitBinarySimd<int>(MulVec<int>()).CreateDelegate<Action<nint, nint, nint, int>>();

        _dynAddFloat64Scalar = EmitBinaryScalar<double>(OpCodes.Add).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynAddFloat64Simd = EmitBinarySimd<double>(AddVec<double>()).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynMulFloat64Scalar = EmitBinaryScalar<double>(OpCodes.Mul).CreateDelegate<Action<nint, nint, nint, int>>();
        _dynMulFloat64Simd = EmitBinarySimd<double>(MulVec<double>()).CreateDelegate<Action<nint, nint, nint, int>>();

        _dynSqrtFloat64Scalar = EmitUnarySqrt().CreateDelegate<Action<nint, nint, int>>();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        NativeMemory.AlignedFree(_aInt);
        NativeMemory.AlignedFree(_bInt);
        NativeMemory.AlignedFree(_cInt);
        NativeMemory.AlignedFree(_aDouble);
        NativeMemory.AlignedFree(_bDouble);
        NativeMemory.AlignedFree(_cDouble);
    }

    // ========================================================================
    // INT32 ADD: NumSharp vs DynamicMethod
    // ========================================================================

    [Benchmark(Baseline = true, Description = "NumSharp: a + b")]
    [BenchmarkCategory("Add_Int32")]
    public NDArray NumSharp_Add_Int32() => _aIntND + _bIntND;

    [Benchmark(Description = "DynMethod scalar: a + b")]
    [BenchmarkCategory("Add_Int32")]
    public void DynMethod_Add_Int32_Scalar()
    {
        _dynAddInt32Scalar!((nint)_aInt, (nint)_bInt, (nint)_cInt, N);
    }

    [Benchmark(Description = "DynMethod SIMD: a + b")]
    [BenchmarkCategory("Add_Int32")]
    public void DynMethod_Add_Int32_Simd()
    {
        _dynAddInt32Simd!((nint)_aInt, (nint)_bInt, (nint)_cInt, N);
    }

    // ========================================================================
    // INT32 MUL: NumSharp vs DynamicMethod
    // ========================================================================

    [Benchmark(Baseline = true, Description = "NumSharp: a * b")]
    [BenchmarkCategory("Mul_Int32")]
    public NDArray NumSharp_Mul_Int32() => _aIntND * _bIntND;

    [Benchmark(Description = "DynMethod scalar: a * b")]
    [BenchmarkCategory("Mul_Int32")]
    public void DynMethod_Mul_Int32_Scalar()
    {
        _dynMulInt32Scalar!((nint)_aInt, (nint)_bInt, (nint)_cInt, N);
    }

    [Benchmark(Description = "DynMethod SIMD: a * b")]
    [BenchmarkCategory("Mul_Int32")]
    public void DynMethod_Mul_Int32_Simd()
    {
        _dynMulInt32Simd!((nint)_aInt, (nint)_bInt, (nint)_cInt, N);
    }

    // ========================================================================
    // FLOAT64 ADD: NumSharp vs DynamicMethod
    // ========================================================================

    [Benchmark(Baseline = true, Description = "NumSharp: a + b")]
    [BenchmarkCategory("Add_Float64")]
    public NDArray NumSharp_Add_Float64() => _aDoubleND + _bDoubleND;

    [Benchmark(Description = "DynMethod scalar: a + b")]
    [BenchmarkCategory("Add_Float64")]
    public void DynMethod_Add_Float64_Scalar()
    {
        _dynAddFloat64Scalar!((nint)_aDouble, (nint)_bDouble, (nint)_cDouble, N);
    }

    [Benchmark(Description = "DynMethod SIMD: a + b")]
    [BenchmarkCategory("Add_Float64")]
    public void DynMethod_Add_Float64_Simd()
    {
        _dynAddFloat64Simd!((nint)_aDouble, (nint)_bDouble, (nint)_cDouble, N);
    }

    // ========================================================================
    // FLOAT64 MUL: NumSharp vs DynamicMethod
    // ========================================================================

    [Benchmark(Baseline = true, Description = "NumSharp: a * b")]
    [BenchmarkCategory("Mul_Float64")]
    public NDArray NumSharp_Mul_Float64() => _aDoubleND * _bDoubleND;

    [Benchmark(Description = "DynMethod scalar: a * b")]
    [BenchmarkCategory("Mul_Float64")]
    public void DynMethod_Mul_Float64_Scalar()
    {
        _dynMulFloat64Scalar!((nint)_aDouble, (nint)_bDouble, (nint)_cDouble, N);
    }

    [Benchmark(Description = "DynMethod SIMD: a * b")]
    [BenchmarkCategory("Mul_Float64")]
    public void DynMethod_Mul_Float64_Simd()
    {
        _dynMulFloat64Simd!((nint)_aDouble, (nint)_bDouble, (nint)_cDouble, N);
    }

    // ========================================================================
    // FLOAT64 SQRT (Unary): NumSharp vs DynamicMethod
    // ========================================================================

    [Benchmark(Baseline = true, Description = "NumSharp: np.sqrt(a)")]
    [BenchmarkCategory("Sqrt_Float64")]
    public NDArray NumSharp_Sqrt_Float64() => np.sqrt(_aDoubleND);

    [Benchmark(Description = "DynMethod scalar: sqrt(a)")]
    [BenchmarkCategory("Sqrt_Float64")]
    public void DynMethod_Sqrt_Float64_Scalar()
    {
        _dynSqrtFloat64Scalar!((nint)_aDouble, (nint)_cDouble, N);
    }

    // ========================================================================
    // DynamicMethod Emitters
    // ========================================================================

    private static DynamicMethod EmitBinaryScalar<T>(OpCode opcode) where T : unmanaged
    {
        var size = sizeof(T);
        var ldind = GetLdind<T>();
        var stind = GetStind<T>();

        var dm = new DynamicMethod($"BinaryScalar_{typeof(T).Name}", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)],
            typeof(DynamicEmissionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int)); // i

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        // c[i] = a[i] op b[i]
        EmitPtrOffset(il, 2, size); // &c[i]
        EmitPtrLoad(il, 0, size, ldind); // a[i]
        EmitPtrLoad(il, 1, size, ldind); // b[i]
        il.Emit(opcode);
        il.Emit(stind);

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

    private static DynamicMethod EmitBinarySimd<T>(MethodInfo vecOp) where T : unmanaged
    {
        var size = sizeof(T);
        var ldind = GetLdind<T>();
        var stind = GetStind<T>();
        var vecCount = Vector256<T>.Count;

        var loadM = typeof(Vector256).GetMethods()
            .First(m => m.Name == "Load" && m.IsGenericMethod)
            .MakeGenericMethod(typeof(T));
        var storeM = typeof(Vector256).GetMethods()
            .First(m => m.Name == "Store" && m.IsGenericMethod && m.GetParameters().Length == 2)
            .MakeGenericMethod(typeof(T));

        var dm = new DynamicMethod($"BinarySimd_{typeof(T).Name}", typeof(void),
            [typeof(nint), typeof(nint), typeof(nint), typeof(int)],
            typeof(DynamicEmissionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var sTop = il.DefineLabel();
        var sChk = il.DefineLabel();
        var tTop = il.DefineLabel();
        var tChk = il.DefineLabel();
        il.DeclareLocal(typeof(int)); // i
        il.DeclareLocal(typeof(int)); // simdEnd

        // simdEnd = n - vecCount
        il.Emit(OpCodes.Ldarg_3);
        il.Emit(OpCodes.Ldc_I4, vecCount);
        il.Emit(OpCodes.Sub);
        il.Emit(OpCodes.Stloc_1);

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, sChk);

        // SIMD loop
        il.MarkLabel(sTop);
        EmitPtrOffset(il, 0, size);
        il.EmitCall(OpCodes.Call, loadM, null);
        EmitPtrOffset(il, 1, size);
        il.EmitCall(OpCodes.Call, loadM, null);
        il.EmitCall(OpCodes.Call, vecOp, null);
        EmitPtrOffset(il, 2, size);
        il.EmitCall(OpCodes.Call, storeM, null);

        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4, vecCount);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);

        il.MarkLabel(sChk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldloc_1);
        il.Emit(OpCodes.Ble, sTop);

        // Scalar tail
        il.Emit(OpCodes.Br, tChk);
        il.MarkLabel(tTop);
        EmitPtrOffset(il, 2, size);
        EmitPtrLoad(il, 0, size, ldind);
        EmitPtrLoad(il, 1, size, ldind);
        il.Emit(OpCodes.Add);
        il.Emit(stind);

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

    private static DynamicMethod EmitUnarySqrt()
    {
        var sqrtM = typeof(Math).GetMethod("Sqrt", [typeof(double)])!;

        var dm = new DynamicMethod("UnarySqrt", typeof(void),
            [typeof(nint), typeof(nint), typeof(int)],
            typeof(DynamicEmissionBenchmarks).Module, true);
        var il = dm.GetILGenerator();
        var top = il.DefineLabel();
        var chk = il.DefineLabel();
        il.DeclareLocal(typeof(int)); // i

        il.Emit(OpCodes.Ldc_I4_0);
        il.Emit(OpCodes.Stloc_0);
        il.Emit(OpCodes.Br, chk);

        il.MarkLabel(top);
        // c[i] = sqrt(a[i])
        EmitPtrOffset(il, 1, 8); // &c[i]
        EmitPtrLoad(il, 0, 8, OpCodes.Ldind_R8); // a[i]
        il.EmitCall(OpCodes.Call, sqrtM, null);
        il.Emit(OpCodes.Stind_R8);

        // i++
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldc_I4_1);
        il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Stloc_0);

        il.MarkLabel(chk);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Ldarg_2);
        il.Emit(OpCodes.Blt, top);

        il.Emit(OpCodes.Ret);
        return dm;
    }

    // ========================================================================
    // IL Helpers
    // ========================================================================

    private static void EmitPtrOffset(ILGenerator il, int argIndex, int elemSize)
    {
        il.Emit(OpCodes.Ldarg, argIndex);
        il.Emit(OpCodes.Ldloc_0);
        il.Emit(OpCodes.Conv_I);
        il.Emit(OpCodes.Ldc_I4, elemSize);
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Add);
    }

    private static void EmitPtrLoad(ILGenerator il, int argIndex, int elemSize, OpCode ldind)
    {
        EmitPtrOffset(il, argIndex, elemSize);
        il.Emit(ldind);
    }

    private static OpCode GetLdind<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(int)) return OpCodes.Ldind_I4;
        if (typeof(T) == typeof(double)) return OpCodes.Ldind_R8;
        if (typeof(T) == typeof(float)) return OpCodes.Ldind_R4;
        if (typeof(T) == typeof(long)) return OpCodes.Ldind_I8;
        throw new NotSupportedException($"Type {typeof(T)} not supported");
    }

    private static OpCode GetStind<T>() where T : unmanaged
    {
        if (typeof(T) == typeof(int)) return OpCodes.Stind_I4;
        if (typeof(T) == typeof(double)) return OpCodes.Stind_R8;
        if (typeof(T) == typeof(float)) return OpCodes.Stind_R4;
        if (typeof(T) == typeof(long)) return OpCodes.Stind_I8;
        throw new NotSupportedException($"Type {typeof(T)} not supported");
    }

    private static MethodInfo AddVec<T>() where T : unmanaged =>
        typeof(Vector256<T>).GetMethods()
            .First(m => m.Name == "op_Addition" && m.GetParameters()[0].ParameterType == typeof(Vector256<T>));

    private static MethodInfo MulVec<T>() where T : unmanaged =>
        typeof(Vector256<T>).GetMethods()
            .First(m => m.Name == "op_Multiply" && m.GetParameters()[0].ParameterType == typeof(Vector256<T>));
}
