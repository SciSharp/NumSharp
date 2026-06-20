#:property PublishAot=false
// Self-contained kernel microbench: how fast is .NET scalar Span.Sort vs NumPy's SIMD sort?
// Measures the 1-D line-sort kernel only (the dominant cost). No NumSharp needed.
using System;
using System.Collections.Generic;
using System.Diagnostics;

const int ROUNDS = 7;
int[] sizes = { 1_000, 100_000, 1_000_000, 10_000_000 };

static double BestMs(Action setup, Action timed)
{
    double best = 1e18;
    for (int r = 0; r < 7; r++)
    {
        setup();
        var sw = Stopwatch.StartNew();
        timed();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
    }
    return best;
}

var rng = new Random(42);

Console.WriteLine("== sort: Span.Sort() default vs struct NumPy-comparer ==");
foreach (int n in sizes)
{
    // int32
    var src32 = new int[n];
    for (int i = 0; i < n; i++) src32[i] = rng.Next();
    var work = new int[n];
    double def = BestMs(() => src32.CopyTo(work, 0), () => work.AsSpan().Sort());
    double cmp = BestMs(() => src32.CopyTo(work, 0), () => work.AsSpan().Sort(new IntCmp()));
    Console.WriteLine($"int32   n={n,9}  default {def,8:F3} ms ({n/def/1e3,7:F1} M/s) | structcmp {cmp,8:F3} ms ({n/cmp/1e3,7:F1} M/s)");
}
foreach (int n in sizes)
{
    var src = new double[n];
    for (int i = 0; i < n; i++) src[i] = rng.NextDouble();
    var work = new double[n];
    double def = BestMs(() => src.CopyTo(work, 0), () => work.AsSpan().Sort());
    double cmp = BestMs(() => src.CopyTo(work, 0), () => work.AsSpan().Sort(new DblCmp()));
    Console.WriteLine($"float64 n={n,9}  default {def,8:F3} ms ({n/def/1e3,7:F1} M/s) | structcmp {cmp,8:F3} ms ({n/cmp/1e3,7:F1} M/s)");
}
foreach (int n in sizes)
{
    var src = new float[n];
    for (int i = 0; i < n; i++) src[i] = (float)rng.NextDouble();
    var work = new float[n];
    double def = BestMs(() => src.CopyTo(work, 0), () => work.AsSpan().Sort());
    double cmp = BestMs(() => src.CopyTo(work, 0), () => work.AsSpan().Sort(new FltCmp()));
    Console.WriteLine($"float32 n={n,9}  default {def,8:F3} ms ({n/def/1e3,7:F1} M/s) | structcmp {cmp,8:F3} ms ({n/cmp/1e3,7:F1} M/s)");
}

Console.WriteLine("== argsort: co-sort keys+items vs indirect index sort ==");
foreach (int n in sizes)
{
    var keys0 = new int[n];
    for (int i = 0; i < n; i++) keys0[i] = rng.Next();
    var keys = new int[n];
    var idx = new long[n];
    // co-sort: keys.Sort(items) — unstable, mutates keys
    double co = BestMs(() => { keys0.CopyTo(keys, 0); for (long i = 0; i < n; i++) idx[i] = i; },
                       () => keys.AsSpan().Sort(idx.AsSpan()));
    // indirect: sort idx by keys0[idx] with tie-break -> stable
    double ind = BestMs(() => { for (long i = 0; i < n; i++) idx[i] = i; },
                        () => idx.AsSpan().Sort(new ArgCmp(keys0)));
    Console.WriteLine($"argint  n={n,9}  cosort {co,8:F3} ms ({n/co/1e3,7:F1} M/s) | indirect {ind,8:F3} ms ({n/ind/1e3,7:F1} M/s)");
}

readonly struct IntCmp : IComparer<int> { public int Compare(int a, int b) => a < b ? -1 : (a > b ? 1 : 0); }
readonly struct FltCmp : IComparer<float>
{
    public int Compare(float a, float b)
    {
        if (a < b) return -1; if (a > b) return 1;
        bool an = float.IsNaN(a), bn = float.IsNaN(b);
        if (an && bn) return 0; if (an) return 1; if (bn) return -1; return 0;
    }
}
readonly struct DblCmp : IComparer<double>
{
    public int Compare(double a, double b)
    {
        if (a < b) return -1; if (a > b) return 1;
        bool an = double.IsNaN(a), bn = double.IsNaN(b);
        if (an && bn) return 0; if (an) return 1; if (bn) return -1; return 0;
    }
}
readonly struct ArgCmp : IComparer<long>
{
    private readonly int[] _k;
    public ArgCmp(int[] k) { _k = k; }
    public int Compare(long i, long j)
    {
        int a = _k[i], b = _k[j];
        if (a < b) return -1; if (a > b) return 1;
        return i < j ? -1 : (i > j ? 1 : 0); // stable tie-break
    }
}
