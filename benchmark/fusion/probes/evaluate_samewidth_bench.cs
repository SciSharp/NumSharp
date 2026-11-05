#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaluate_samewidth_bench.cs — np.evaluate's P5.3 same-width vector plan (NDExpr.Vector.cs, TryPlanSameWidth).
//
// Seven trees whose non-bool dtypes share one element size but are not one dtype — the masks cross between them —
// timed three ways: vectorized (the default since P5.3), on NDExpr.ForceScalar (the scalar kernel every one of them ran
// before the plan existed: a scalar `where` branches per element) and as NumSharp's unfused statements. The NumPy twin
// (evaluate_samewidth_bench.py) times NumPy's statements over the same value distributions.
//   where(age>30,price,0)       int64 compare selecting float64 (W = float64)
//   (age>30)*1.5                a mask made 1.0 / 0.0 under a weak float literal
//   (age>30)&isnan(price)       a bool root: W = the first non-bool input's dtype (int64)
//   logical_and(age,price)      truthiness tested at each operand's own dtype
//   where(i4<5,f4,f4*2)         the 4-byte pair
//   m!=0                        a bool input against an int64 literal (NumPy types the compare int64)
//   where(m&(age>30),price,-1)  a streamed bool input in a same-width tree
// Measured 2026-11-05 (i9-13900K P-cores 0x14, .NET 10, Release; NumPy 2.4.2): vector vs the former scalar kernel
// 1.07-23.4x; NPY/NS 1.84-10.6x after (0.34-6.11x before), every cell at 1K / 100K / 1M.
//
// Run (Release only; internals: NDExpr.ForceScalar):
//   dotnet run -c Release --artifacts-path <its own folder> benchmark/fusion/probes/evaluate_samewidth_bench.cs
//   python benchmark/fusion/probes/evaluate_samewidth_bench.py
// Pinned to 0x14 (two P-cores); best of 7 after a paused double warm-up; every route checked equal first.
// =============================================================================
using System;
using System.Diagnostics;
using System.Threading;
using NumSharp;
using NumSharp.Backends.Iteration;

Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x14;

double Time(Func<NDArray> f, int n)
{
    for (int w = 0; w < 20; w++) f().Dispose();
    Thread.Sleep(100);
    for (int w = 0; w < 20; w++) f().Dispose();
    int reps = n <= 1000 ? 3000 : n <= 100_000 ? 200 : 25;
    double best = double.MaxValue;
    for (int r = 0; r < 7; r++)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) f().Dispose();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / reps);
    }
    return best;
}

NDArray Scalar(Func<NDArray> f)
{
    NDExpr.ForceScalar = true;
    try { return f(); }
    finally { NDExpr.ForceScalar = false; }
}

foreach (int n in new[] { 1_000, 100_000, 1_000_000 })
{
    var rnd = new Random(42);
    var ageV = new long[n];
    var priceV = new double[n];
    var i4V = new int[n];
    var f4V = new float[n];
    var mV = new bool[n];
    for (int i = 0; i < n; i++)
    {
        ageV[i] = rnd.Next(0, 97);
        priceV[i] = rnd.Next(0, 10) == 0 ? double.NaN : rnd.NextDouble() * 100;
        i4V[i] = rnd.Next(-10, 10);
        f4V[i] = (float)rnd.NextDouble();
        mV[i] = rnd.Next(2) == 0;
    }
    NDArray age = np.array(ageV), price = np.array(priceV), i4 = np.array(i4V), f4 = np.array(f4V), m = np.array(mV);

    var trees = new (string name, Func<NDArray> fused, Func<NDArray> unfused)[]
    {
        ("where(age>30,price,0)", () => np.evaluate(NDExpr.Where((NDExpr)age > 30, price, 0)),
            () => { using var c = age > 30; return np.where(c, price, 0); }),
        ("(age>30)*1.5", () => np.evaluate(NDExpr.Greater(age, 30) * 1.5),
            () => { using var c = age > 30; return c * 1.5; }),
        ("(age>30)&isnan(price)", () => np.evaluate(NDExpr.Greater(age, 30) & NDExpr.IsNaN(price)),
            () => { using var c = age > 30; using var nn = np.isnan(price); return c & nn; }),
        ("logical_and(age,price)", () => np.evaluate(NDExpr.LogicalAnd(age, price)),
            () => np.logical_and(age, price)),
        ("where(i4<5,f4,f4*2)", () => np.evaluate(NDExpr.Where(NDExpr.Less(i4, 5), f4, (NDExpr)f4 * 2)),
            () => { using var c = i4 < 5; using var d = f4 * 2; return np.where(c, f4, d); }),
        ("m!=0", () => np.evaluate(NDExpr.NotEqual(m, 0)),
            () => np.not_equal(m, 0)),
        ("where(m&(age>30),price,-1)", () => np.evaluate(NDExpr.Where(NDExpr.Arr(m) & NDExpr.Greater(age, 30), price, -1)),
            () => { using var c = age > 30; using var mm = m & c; return np.where(mm, price, -1); }),
    };
    foreach (var (name, fused, unfused) in trees)
    {
        using (var a = fused())
        using (var b = Scalar(fused))
        using (var u = unfused())
        {
            if (!np.array_equal(a, b, equal_nan: true) || !np.array_equal(a, u, equal_nan: true))
                throw new Exception($"{name} {n}: routes disagree");
        }
        double tv = Math.Min(Time(fused, n), Time(fused, n));
        double ts = Math.Min(Time(() => Scalar(fused), n), Time(() => Scalar(fused), n));
        double tu = Math.Min(Time(unfused, n), Time(unfused, n));
        Console.WriteLine($"{name,-28} n={n,9}: vector {tv,9:F1} us  scalar(before) {ts,9:F1} us  unfused {tu,9:F1} us  scalar/vector {ts / tv,5:F2}x  unfused/vector {tu / tv,5:F2}x");
    }
}
