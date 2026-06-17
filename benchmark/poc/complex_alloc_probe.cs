#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using NumSharp;

static (double ms, long bytes) Measure(Action a, int iters)
{
    for (int i = 0; i < 3; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    long b0 = GC.GetAllocatedBytesForCurrentThread();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) a();
    sw.Stop();
    long b1 = GC.GetAllocatedBytesForCurrentThread();
    return (sw.Elapsed.TotalMilliseconds / iters, (b1 - b0) / iters);
}

void Run(int N)
{
    int rows = (int)Math.Sqrt(N), cols = N / rows;
    long total = (long)rows * cols;
    var a = (np.arange(total).astype(NPTypeCode.Double).reshape(rows, cols).astype(NPTypeCode.Complex)) + new Complex(1, 1);
    int it = N >= 1_000_000 ? 20 : 200;

    var (sumMs, sumB) = Measure(() => { using var _ = np.sum(a, axis: 0); }, it);
    var (meanMs, meanB) = Measure(() => { using var _ = np.mean(a, axis: 0); }, it);
    long elems = total; // elements visited per reduce
    Console.WriteLine($"N={N,-9} ({rows}x{cols}, {elems:N0} elems/op)");
    Console.WriteLine($"  np.sum  axis=0 : {sumMs,8:F4} ms | alloc {sumB,14:N0} B/op | {(double)sumB/elems,6:F1} B/elem");
    Console.WriteLine($"  np.mean axis=0 : {meanMs,8:F4} ms | alloc {meanB,14:N0} B/op | {(double)meanB/elems,6:F1} B/elem");
}

Console.WriteLine("=== Allocation per op (boxing leaves a fingerprint: ~24-72 B/elem) ===");
Run(100_000);
Run(10_000_000);

// Does the clean typed combiner support Complex today?
Console.WriteLine("\n=== Can Complex use the clean typed path? ===");
try {
    var m = typeof(NumSharp.Backends.Kernels.DirectILKernelGenerator)
        .GetMethod("AddScalar", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
    var g = m.MakeGenericMethod(typeof(Complex));
    var r = g.Invoke(null, new object[] { new Complex(1,2), new Complex(3,4) });
    Console.WriteLine($"  AddScalar<Complex>(1+2i, 3+4i) = {r}");
} catch (Exception e) {
    Console.WriteLine($"  AddScalar<Complex> THROWS: {e.InnerException?.GetType().Name ?? e.GetType().Name} -> '{e.InnerException?.Message ?? e.Message}'");
}
