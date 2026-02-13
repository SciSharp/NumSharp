using System.Runtime.CompilerServices;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.Isolated;

/// <summary>
/// Measure the overhead of different dispatch mechanisms.
/// Goal: Determine acceptable dispatch overhead budget.
/// </summary>
public static unsafe class DispatchOverhead
{
    private const string Suite = "DispatchOverhead";

    /// <summary>
    /// Run all dispatch overhead benchmarks.
    /// </summary>
    public static List<BenchResult> RunAll(bool quick = false)
    {
        var results = new List<BenchResult>();
        var warmup = quick ? 3 : BenchFramework.DefaultWarmup;
        var measure = quick ? 15 : 30; // More iterations for small overhead measurement

        var sizes = quick
            ? new[] { 1_000, 100_000 }
            : new[] { 100, 1_000, 10_000, 100_000, 1_000_000 };

        BenchFramework.PrintHeader($"{Suite}: Dispatch Mechanism Overhead");

        foreach (var size in sizes)
        {
            BenchFramework.PrintDivider($"Size: {size:N0}");
            results.AddRange(MeasureDispatchOverhead(size, warmup, measure));
        }

        OutputFormatters.PrintSummary(results);
        return results;
    }

    /// <summary>
    /// Measure overhead of various dispatch mechanisms.
    /// </summary>
    public static List<BenchResult> MeasureDispatchOverhead(int size, int warmup, int measure)
    {
        var results = new List<BenchResult>();
        int elementBytes = 8;

        var lhs = SimdImplementations.AllocateAligned<double>(size);
        var rhs = SimdImplementations.AllocateAligned<double>(size);
        var result = SimdImplementations.AllocateAligned<double>(size);
        SimdImplementations.FillRandom(lhs, size, 42);
        SimdImplementations.FillRandom(rhs, size, 43);

        // Test 1: Direct function call (baseline)
        var baseline = BenchFramework.Run(
            () => SimdImplementations.AddFull_Float64(lhs, rhs, result, size),
            "Dispatch", "1_Direct", "float64", size, elementBytes, warmup, measure, Suite);
        results.Add(baseline);
        BenchFramework.PrintResult(baseline);

        // Test 2: Single if/else dispatch
        var ifElse = BenchFramework.Run(
            () => DispatchIfElse(lhs, rhs, result, size, isContiguous: true),
            "Dispatch", "2_IfElse", "float64", size, elementBytes, warmup, measure, Suite);
        ifElse = ifElse with { SpeedupVsBaseline = baseline.MeanUs / ifElse.MeanUs };
        results.Add(ifElse);
        BenchFramework.PrintResult(ifElse);

        // Test 3: Nested if/else dispatch (simulate shape checks)
        var nested = BenchFramework.Run(
            () => DispatchNested(lhs, rhs, result, size, isContiguous: true, isBroadcast: false, isScalar: false),
            "Dispatch", "3_Nested", "float64", size, elementBytes, warmup, measure, Suite);
        nested = nested with { SpeedupVsBaseline = baseline.MeanUs / nested.MeanUs };
        results.Add(nested);
        BenchFramework.PrintResult(nested);

        // Test 4: Switch dispatch
        var switchDisp = BenchFramework.Run(
            () => DispatchSwitch(lhs, rhs, result, size, scenario: 0),
            "Dispatch", "4_Switch", "float64", size, elementBytes, warmup, measure, Suite);
        switchDisp = switchDisp with { SpeedupVsBaseline = baseline.MeanUs / switchDisp.MeanUs };
        results.Add(switchDisp);
        BenchFramework.PrintResult(switchDisp);

        // Test 5: Virtual method dispatch
        IAddOperation operation = new SimdAddOperation();
        var virtualDisp = BenchFramework.Run(
            () => operation.Execute(lhs, rhs, result, size),
            "Dispatch", "5_Virtual", "float64", size, elementBytes, warmup, measure, Suite);
        virtualDisp = virtualDisp with { SpeedupVsBaseline = baseline.MeanUs / virtualDisp.MeanUs };
        results.Add(virtualDisp);
        BenchFramework.PrintResult(virtualDisp);

        // Test 6: Delegate invocation
        AddDelegate del = SimdImplementations.AddFull_Float64;
        var delegateDisp = BenchFramework.Run(
            () => del(lhs, rhs, result, size),
            "Dispatch", "6_Delegate", "float64", size, elementBytes, warmup, measure, Suite);
        delegateDisp = delegateDisp with { SpeedupVsBaseline = baseline.MeanUs / delegateDisp.MeanUs };
        results.Add(delegateDisp);
        BenchFramework.PrintResult(delegateDisp);

        // Test 7: Function pointer
        delegate* managed<double*, double*, double*, int, void> funcPtr = &SimdImplementations.AddFull_Float64;
        var funcPtrDisp = BenchFramework.Run(
            () => funcPtr(lhs, rhs, result, size),
            "Dispatch", "7_FuncPtr", "float64", size, elementBytes, warmup, measure, Suite);
        funcPtrDisp = funcPtrDisp with { SpeedupVsBaseline = baseline.MeanUs / funcPtrDisp.MeanUs };
        results.Add(funcPtrDisp);
        BenchFramework.PrintResult(funcPtrDisp);

        SimdImplementations.FreeAligned(lhs);
        SimdImplementations.FreeAligned(rhs);
        SimdImplementations.FreeAligned(result);

        return results;
    }

    #region Dispatch Implementations

    private delegate void AddDelegate(double* lhs, double* rhs, double* result, int count);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DispatchIfElse(double* lhs, double* rhs, double* result, int size, bool isContiguous)
    {
        if (isContiguous)
        {
            SimdImplementations.AddFull_Float64(lhs, rhs, result, size);
        }
        else
        {
            SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DispatchNested(double* lhs, double* rhs, double* result, int size,
        bool isContiguous, bool isBroadcast, bool isScalar)
    {
        if (isScalar)
        {
            // Scalar path (not used in this test)
            SimdImplementations.AddScalar_Float64(lhs, 0.0, result, size);
        }
        else if (isContiguous && !isBroadcast)
        {
            SimdImplementations.AddFull_Float64(lhs, rhs, result, size);
        }
        else if (isBroadcast)
        {
            // Broadcast path (not used in this test)
            SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size);
        }
        else
        {
            SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void DispatchSwitch(double* lhs, double* rhs, double* result, int size, int scenario)
    {
        switch (scenario)
        {
            case 0: // S1: Contiguous
                SimdImplementations.AddFull_Float64(lhs, rhs, result, size);
                break;
            case 1: // S2: Scalar broadcast
                SimdImplementations.AddScalar_Float64(lhs, 0.0, result, size);
                break;
            case 2: // S4: Row broadcast
                SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size);
                break;
            default:
                SimdImplementations.AddScalarLoop_Float64(lhs, rhs, result, size);
                break;
        }
    }

    private interface IAddOperation
    {
        void Execute(double* lhs, double* rhs, double* result, int size);
    }

    private class SimdAddOperation : IAddOperation
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Execute(double* lhs, double* rhs, double* result, int size)
        {
            SimdImplementations.AddFull_Float64(lhs, rhs, result, size);
        }
    }

    #endregion

    /// <summary>
    /// Measure overhead of property checks (simulated shape properties).
    /// </summary>
    public static List<BenchResult> MeasurePropertyCheckOverhead(int iterations, int warmup, int measure)
    {
        var results = new List<BenchResult>();

        BenchFramework.PrintDivider("Property Check Overhead");

        // Simulate shape-like object with properties
        var shape = new SimulatedShape(1000, 1000, isContiguous: true, isBroadcast: false);

        // Test 1: No property checks (baseline)
        bool result1 = false;
        var baseline = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    result1 = true; // Just assign
                }
            },
            "PropCheck", "0_None", "N/A", iterations, 1, warmup, measure, Suite);
        results.Add(baseline);
        BenchFramework.PrintResult(baseline);

        // Test 2: Single property check
        var single = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    result1 = shape.IsContiguous;
                }
            },
            "PropCheck", "1_Single", "N/A", iterations, 1, warmup, measure, Suite);
        single = single with { SpeedupVsBaseline = baseline.MeanUs / single.MeanUs };
        results.Add(single);
        BenchFramework.PrintResult(single);

        // Test 3: Multiple property checks
        var multi = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    result1 = shape.IsContiguous && !shape.IsBroadcast && shape.Size > 0;
                }
            },
            "PropCheck", "2_Multi", "N/A", iterations, 1, warmup, measure, Suite);
        multi = multi with { SpeedupVsBaseline = baseline.MeanUs / multi.MeanUs };
        results.Add(multi);
        BenchFramework.PrintResult(multi);

        // Test 4: Computed property
        var computed = BenchFramework.Run(
            () =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    result1 = shape.IsContiguousComputed;
                }
            },
            "PropCheck", "3_Computed", "N/A", iterations, 1, warmup, measure, Suite);
        computed = computed with { SpeedupVsBaseline = baseline.MeanUs / computed.MeanUs };
        results.Add(computed);
        BenchFramework.PrintResult(computed);

        // Keep result to prevent optimization
        _ = result1;

        return results;
    }

    private class SimulatedShape
    {
        private readonly int[] _dimensions;
        private readonly int[] _strides;
        private readonly bool _isContiguous;
        private readonly bool _isBroadcast;

        public SimulatedShape(int rows, int cols, bool isContiguous, bool isBroadcast)
        {
            _dimensions = [rows, cols];
            _strides = [cols, 1];
            _isContiguous = isContiguous;
            _isBroadcast = isBroadcast;
        }

        public bool IsContiguous => _isContiguous;
        public bool IsBroadcast => _isBroadcast;
        public int Size => _dimensions[0] * _dimensions[1];

        // Computed property that checks strides
        public bool IsContiguousComputed
        {
            get
            {
                int expected = 1;
                for (int i = _dimensions.Length - 1; i >= 0; i--)
                {
                    if (_strides[i] != expected) return false;
                    expected *= _dimensions[i];
                }
                return true;
            }
        }
    }
}
