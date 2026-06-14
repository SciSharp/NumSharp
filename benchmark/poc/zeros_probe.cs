#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Numerics;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;

static double Time(Action a, int iters, int warmup = 10)
{
    for (int i = 0; i < warmup; i++) a();
    // Drain finalizer backlog so a prior measurement's undisposed garbage
    // doesn't pollute this one (matches a fresh process / per-op steady state).
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) a();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / iters;
}

// ---- Perf across sizes & dtypes ----
Console.WriteLine("=== np.zeros timing (ms/op) ===");
Console.WriteLine($"{"N",-12}{"float64",12}{"int64",12}{"float32",12}{"int32",12}");
foreach (int N in new[] { 1_000, 100_000, 10_000_000 })
{
    int it = N >= 1_000_000 ? 50 : 200;
    double f64 = Time(() => np.zeros(new Shape(N), NPTypeCode.Double), it);
    double i64 = Time(() => np.zeros(new Shape(N), NPTypeCode.Int64), it);
    double f32 = Time(() => np.zeros(new Shape(N), NPTypeCode.Single), it);
    double i32 = Time(() => np.zeros(new Shape(N), NPTypeCode.Int32), it);
    Console.WriteLine($"{N,-12}{f64,12:F4}{i64,12:F4}{f32,12:F4}{i32,12:F4}");
}

Console.WriteLine($"\nPool zeroed-allocs counter: {SizeBucketedBufferPool.ZeroedAllocs}");

// ---- Correctness: all 15 dtypes return all zeros ----
Console.WriteLine("\n=== Correctness: all dtypes zeroed ===");
var codes = new[] { NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16,
    NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
    NPTypeCode.Char, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex };
bool allOk = true;
foreach (var c in codes)
{
    var z = np.zeros(new Shape(257), c); // 257 elems crosses SIMD tail
    bool ok = true;
    for (long i = 0; i < z.size; i++)
    {
        var v = z.GetAtIndex(i);
        // default(T) compare via Convert to double where possible
        switch (c)
        {
            case NPTypeCode.Boolean: ok &= ((bool)v == false); break;
            case NPTypeCode.Decimal: ok &= ((decimal)v == 0m); break;
            case NPTypeCode.Complex: ok &= ((Complex)v == Complex.Zero); break;
            case NPTypeCode.Half: ok &= ((Half)v == (Half)0); break;
            case NPTypeCode.Char: ok &= ((char)v == '\0'); break;
            default: ok &= (Convert.ToDouble(v) == 0d); break;
        }
        if (!ok) break;
    }
    allOk &= ok;
    Console.WriteLine($"  {c,-10}: {(ok ? "OK" : "FAIL")}");
}
Console.WriteLine($"All dtypes zeroed: {allOk}");

// ---- Edge cases ----
Console.WriteLine("\n=== Edge cases ===");
var empty = np.zeros(new Shape(0, 3), NPTypeCode.Double);
Console.WriteLine($"empty (0,3): size={empty.size}, shape=[{string.Join(",", empty.shape)}]");

var twoD = np.zeros(new Shape(3, 4), NPTypeCode.Int32);
Console.WriteLine($"2D (3,4): size={twoD.size}, all-zero={AllZero(twoD)}");

var defaultDtype = np.zeros(new Shape(5));
Console.WriteLine($"default dtype: {defaultDtype.dtype.Name} (expect Double)");

// ---- Writeable + OwnsData ----
var w = np.zeros(new Shape(10), NPTypeCode.Double);
Console.WriteLine($"\nIsWriteable={w.Shape.IsWriteable}, OwnsData(@base==null)={(w.@base == null)}");
w.SetData(5.0, 0);
Console.WriteLine($"after write [0]=5: {w.GetDouble(0)}");

// ---- Independence: two zeros do not alias ----
var a = np.zeros(new Shape(100), NPTypeCode.Double);
var b = np.zeros(new Shape(100), NPTypeCode.Double);
a.SetData(99.0, 0);
Console.WriteLine($"\nIndependence: a[0]={a.GetDouble(0)}, b[0]={b.GetDouble(0)} (b must stay 0)");

// ---- Reuse-after-dispose returns zeros (pool interaction) ----
for (int i = 0; i < 5; i++)
{
    using var t = np.zeros(new Shape(1000), NPTypeCode.Double);
    t.SetData(7.0, 0); // dirty it, then dispose -> returns to pool
}
var fresh = np.zeros(new Shape(1000), NPTypeCode.Double);
Console.WriteLine($"reuse-after-dispose fresh[0]={fresh.GetDouble(0)} (must be 0)");

// ---- zeros_like / eye correctness ----
var src = np.arange(12).reshape(3, 4).astype(NPTypeCode.Single);
var zl = np.zeros_like(src);
Console.WriteLine($"\nzeros_like: dtype={zl.dtype.Name}, shape=[{string.Join(",", zl.shape)}], all-zero={AllZero(zl)}");
var eye = np.eye(3);
Console.WriteLine($"eye(3) diag: [{eye.GetDouble(0,0)},{eye.GetDouble(1,1)},{eye.GetDouble(2,2)}], offdiag[0,1]={eye.GetDouble(0,1)}");

static bool AllZero(NDArray nd)
{
    for (long i = 0; i < nd.size; i++)
        if (Convert.ToDouble(nd.GetAtIndex(i)) != 0d) return false;
    return true;
}
