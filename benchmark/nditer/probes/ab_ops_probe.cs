#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// ab_ops_probe.cs — A/B a broad set of NDIter-routed PRODUCTION ops between two commits.
//
// The #:project reference is RELATIVE to this file, so the same file measures whichever
// checkout it lives in. To A/B against a baseline commit:
//
//   git worktree add /tmp/ns-base <baseline-sha>
//   cp benchmark/nditer/probes/ab_ops_probe.cs /tmp/ns-base/benchmark/nditer/probes/
//   (cd /tmp/ns-base && DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//       dotnet run -c Release benchmark/nditer/probes/ab_ops_probe.cs) > before.tsv
//   DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//       dotnet run -c Release benchmark/nditer/probes/ab_ops_probe.cs > after.tsv
//   python benchmark/nditer/probes/numpy_twins.py ab > numpy.tsv     # NumPy side, same shapes
//   python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv
//
// Output rows are "id<TAB>ns". Run the two sides back-to-back on an idle host: a ±5 % spread
// between identical runs is normal (see docs/NDITER_PERF_DISCOVERY.md → "Measuring correctly").
// =============================================================================
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Generic;

double BestNs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 60) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(3, (int)Math.Ceiling(150.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / it); }
    return best;
}
void Row(string id, double ns) => Console.WriteLine($"{id}\t{ns:F1}");

unsafe
{
    foreach (int n in new[] { 1000, 100_000 })
    {
        string tag = n == 1000 ? "1K" : "100K";
        int R = n == 1000 ? 25 : 250, C = n == 1000 ? 40 : 400;
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(n), np.float64);
        var A = ((np.arange(n).astype(np.float64) % 97.0) + 1.0).reshape(R, C);
        var At = A.T;
        var row = (np.arange(C).astype(np.float64) % 5.0) + 1.0;
        var col = ((np.arange(R).astype(np.float64) % 5.0) + 1.0).reshape(R, 1);
        var a2 = (np.arange(2 * n).astype(np.float64) % 53.0) + 1.0;
        var sa = a2["::2"]; var sb = ((np.arange(2 * n).astype(np.float64) % 17.0) + 1.0)["::2"];
        var af32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
        var ai32 = (np.arange(n) % 1000).astype(np.int32);
        NDArray mask = (np.arange(n) % 2) == 0; var maskB = mask.MakeGeneric<bool>();
        NDArray sparse = (np.arange(n) % 97) == 0;
        var maskDst = a.copy(); var five = NDArray.Scalar(5.0, NPTypeCode.Double);
        var anan = a.copy(); anan[3] = double.NaN;
        var dstStrided = np.empty(new Shape(2 * n), np.float64)["::2"];
        var ob = np.empty(new Shape(n), np.bool_);
        var ipa = a.copy();
        var expr = (NDExpr)a * b + 2.0;

        Row($"add(A,row) bcast@{tag}", BestNs(() => { var r = np.add(A, row); r.Dispose(); }));
        Row($"add(A,col) bcast@{tag}", BestNs(() => { var r = np.add(A, col); r.Dispose(); }));
        Row($"add(sa,sb) strided@{tag}", BestNs(() => { var r = np.add(sa, sb); r.Dispose(); }));
        Row($"add(At,At) transposed@{tag}", BestNs(() => { var r = np.add(At, At); r.Dispose(); }));
        Row($"sqrt(sa) strided@{tag}", BestNs(() => { var r = np.sqrt(sa); r.Dispose(); }));
        Row($"sqrt(ai32) promote@{tag}", BestNs(() => { var r = np.sqrt(ai32); r.Dispose(); }));
        Row($"less(A,row) bcast@{tag}", BestNs(() => { var r = np.less(A, row); r.Dispose(); }));
        Row($"less(sa,sb) strided@{tag}", BestNs(() => { var r = np.less(sa, sb); r.Dispose(); }));
        Row($"left_shift(ai32,1)@{tag}", BestNs(() => { var r = np.left_shift(ai32, 1); r.Dispose(); }));
        Row($"add(a,b,out)@{tag}", BestNs(() => np.add(a, b, o)));
        Row($"multiply(a,b,out=a) inplace@{tag}", BestNs(() => np.multiply(ipa, b, ipa)));
        Row($"add(a,b,out,where)@{tag}", BestNs(() => np.add(a, b, o, where: mask)));
        Row($"less(a,b,out)@{tag}", BestNs(() => np.less(a, b, ob)));
        Row($"sqrt(a,out)@{tag}", BestNs(() => np.sqrt(a, o)));
        Row($"sqrt(af32,out f64) cast@{tag}", BestNs(() => np.sqrt(af32, o)));
        Row($"copyto(strided,src)@{tag}", BestNs(() => np.copyto(dstStrided, a)));
        Row($"At.copy()@{tag}", BestNs(() => { var r = At.copy(); r.Dispose(); }));
        Row($"ravel(At)@{tag}", BestNs(() => { var r = np.ravel(At); r.Dispose(); }));
        Row($"sa.astype(f32)@{tag}", BestNs(() => { var r = sa.astype(np.float32); r.Dispose(); }));
        Row($"concatenate([a,b])@{tag}", BestNs(() => { var r = np.concatenate(new[] { a, b }); r.Dispose(); }));
        Row($"pad(a,2)@{tag}", BestNs(() => { var r = np.pad(a, 2); r.Dispose(); }));
        Row($"diff(a)@{tag}", BestNs(() => { var r = np.diff(a); r.Dispose(); }));
        Row($"a.fill(1.0)@{tag}", BestNs(() => ipa.fill(1.0)));
        Row($"sum(A,axis=1)@{tag}", BestNs(() => { var r = np.sum(A, 1); r.Dispose(); }));
        Row($"sum(At,axis=1)@{tag}", BestNs(() => { var r = np.sum(At, 1); r.Dispose(); }));
        Row($"amin(A,axis=1)@{tag}", BestNs(() => { var r = np.amin(A, 1); r.Dispose(); }));
        Row($"sum(af32,dtype=f64)@{tag}", BestNs(() => { var r = np.sum(af32, NPTypeCode.Double); }));
        Row($"nanmax(anan)@{tag}", BestNs(() => { var r = np.nanmax(anan); }));
        Row($"nanmean(anan)@{tag}", BestNs(() => { var r = np.nanmean(anan); }));
        Row($"nanstd(anan)@{tag}", BestNs(() => { var r = np.nanstd(anan); }));
        Row($"any(sparse)@{tag}", BestNs(() => { var r = np.any(sparse); }));
        Row($"all(mask)@{tag}", BestNs(() => { var r = np.all(mask); }));
        Row($"cumsum(A,axis=1)@{tag}", BestNs(() => { var r = np.cumsum(A, 1); r.Dispose(); }));
        Row($"average(a,weights=b)@{tag}", BestNs(() => { var r = np.average(a, weights: b); }));
        Row($"where(cond,a,b)@{tag}", BestNs(() => { var r = np.where(mask, a, b); r.Dispose(); }));
        Row($"a[mask]@{tag}", BestNs(() => { var r = a[maskB]; r.Dispose(); }));
        Row($"a[mask]=5@{tag}", BestNs(() => maskDst[maskB] = five));
        Row($"count_nonzero(a)@{tag}", BestNs(() => { var r = np.count_nonzero(a); }));
        Row($"argwhere(mask)@{tag}", BestNs(() => { var r = np.argwhere(mask); r.Dispose(); }));
        Row($"nonzero(sparse)@{tag}", BestNs(() => { var r = np.nonzero(sparse); }));
        Row($"sort(A,axis=1)@{tag}", BestNs(() => { var r = np.sort(A, 1); r.Dispose(); }));
        Row($"argsort(A,axis=1)@{tag}", BestNs(() => { var r = np.argsort(A, 1); r.Dispose(); }));
        Row($"evaluate(a*b+2)@{tag}", BestNs(() => { var r = np.evaluate(expr); r.Dispose(); }));
        Row($"nditer<double> foreach@{tag}", BestNs(() => { double s = 0; foreach (ref double x in np.nditer<double>(a)) s += x; GC.KeepAlive(s); }));
        Row($"nditer_chunks<double>@{tag}", BestNs(() => { double s = 0; foreach (var c in np.nditer_chunks<double>(a)) s += c[0]; GC.KeepAlive(s); }));
        Row($"np.nditer class iternext@{tag}", BestNs(() => { using var it = np.nditer(a); while (!it.finished) it.iternext(); }));
        Row($"copy(a) contiguous@{tag}", BestNs(() => { var r = a.copy(); r.Dispose(); }));
    }
}
