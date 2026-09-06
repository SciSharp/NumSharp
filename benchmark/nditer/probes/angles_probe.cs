#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// angles_probe.cs — the candidate NEXT levers, each with the experiment that measures it.
//
//   DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//     dotnet run -c Release benchmark/nditer/probes/angles_probe.cs
//   python benchmark/nditer/probes/numpy_twins.py angles        # NumPy side
//
//   (1) narrow strided rows: per-row EXTERNAL_LOOP vs the BUFFERED window machinery
//   (2) where= masked driver across mask run lengths
//   (3) the fancy-index operator vs the take/put kernels
//   (4) the fresh-NDArray allocation floor
//   (5) 4K aliasing: the same V256 add with pooled buffers at their natural page offsets vs staggered,
//       interleaved and repeated — the demonstration of why a one-shot placement number is not evidence
//
// See docs/NDITER_PERF_DISCOVERY.md → "Where the next wins are".
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

double BestMs(Action body, int rounds = 7)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    double best = double.MaxValue;
    for (int r = 0; r < rounds; r++) { var sw = Stopwatch.StartNew(); body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}
double BestNs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(3, (int)Math.Ceiling(150.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / it); }
    return best;
}

var RO_WO = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY };
const NPY_ORDER KO = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;
const NDIterGlobalFlags EXL = NDIterGlobalFlags.EXTERNAL_LOOP;
const NDIterGlobalFlags BUFEXL = NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.GROWINNER;
var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };

unsafe
{
    Console.WriteLine("--- (1) narrow strided rows, 2M f64: per-row EXLOOP vs BUFFERED windows (ms) ---");
    foreach (int w in new[] { 4, 16, 64 })
    {
        const int TOTAL = 2_097_152; int rows = TOTAL / w;
        var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
        var sv = back[$":, :{w}"]; var dst = np.empty(new Shape(rows, w), np.float64); var ops = new[] { sv, dst };
        double tExl = BestMs(() => { using var it = NDIterRef.MultiNew(2, ops, EXL, KO, SAFE, RO_WO); it.ForEach(K.CopyF64); });
        double tBuf = BestMs(() => { using var it = NDIterRef.MultiNew(2, ops, BUFEXL, KO, SAFE, RO_WO, f64x2); it.ForEach(K.CopyF64); });
        Console.WriteLine($"w={w,4}: EXLOOP per-row {tExl:F3}  BUFFERED windows {tBuf:F3}  np.positive(out) {BestMs(() => np.positive(sv, dst)):F3}  np.sqrt(out) {BestMs(() => np.sqrt(sv, dst)):F3}");
    }

    Console.WriteLine("--- (2) where= masked driver, 100K f64 (us) ---");
    {
        int n = 100_000;
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0; var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0; var o = np.empty(new Shape(n), np.float64);
        NDArray alt = (np.arange(n) % 2) == 0;              // run length 1
        NDArray blocks = (np.floor_divide(np.arange(n), 64) % 2) == 0;   // run length 64 (integer division: `/` on an int array is NumPy true division)
        NDArray half = np.arange(n) < n / 2;                // one run
        Console.WriteLine($"where=alternating {BestNs(() => np.add(a, b, o, where: alt)) / 1e3:F1}  where=64-blocks {BestNs(() => np.add(a, b, o, where: blocks)) / 1e3:F1}  where=half {BestNs(() => np.add(a, b, o, where: half)) / 1e3:F1}  unmasked {BestNs(() => np.add(a, b, o)) / 1e3:F1}");
    }

    Console.WriteLine("--- (3) fancy index vs take/put, 100K f64 (us) ---");
    {
        int n = 100_000;
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var idx = ((np.arange(n).astype(np.int64) * 2654435761L) % n).astype(np.int32);
        var idx64 = idx.astype(np.int64);
        var vals = np.arange(n).astype(np.float64); var dst = a.copy();
        Console.WriteLine($"a[idx] {BestNs(() => { var r = a[idx]; r.Dispose(); }) / 1e3:F1}  np.take(a,idx) {BestNs(() => { var r = np.take(a, idx); r.Dispose(); }) / 1e3:F1}  a[idx]=v {BestNs(() => dst[idx] = vals) / 1e3:F1}  np.put(a,idx,v) {BestNs(() => np.put(dst, idx64, vals)) / 1e3:F1}");
    }

    Console.WriteLine("--- (4) allocation floor: fresh NDArray + Dispose (ns) ---");
    Console.WriteLine($"np.empty(1000) f64 {BestNs(() => { var r = np.empty(new Shape(1000), np.float64); r.Dispose(); }):F1}  np.empty(1) f64 {BestNs(() => { var r = np.empty(new Shape(1), np.float64); r.Dispose(); }):F1}  new Shape(1000) {BestNs(() => { var s = new Shape(1000); GC.KeepAlive(s.dimensions); }):F1}  np.empty((25,40)) {BestNs(() => { var r = np.empty(new Shape(25, 40), np.float64); r.Dispose(); }):F1}");

    // (5) is deliberately INTERLEAVED and repeated: a one-shot "natural then staggered" run once
    // read 0.460 → 0.355 ms (a spurious 1.30× "4K-aliasing win"); alternated on an idle host the
    // two placements sit inside the same noise band. Believe a placement effect only when every
    // rep agrees. See docs/NDITER_PERF_DISCOVERY.md → "Measuring correctly", trap 9.
    Console.WriteLine("--- (5) 1M elementwise add: page-offset stagger, 5 interleaved reps (ms) ---");
    {
        int n = 1_000_000;
        var A = (np.arange(n).astype(np.float64) % 97.0) + 1.0; var B = (np.arange(n).astype(np.float64) % 31.0) + 2.0; var O = np.empty(new Shape(n), np.float64);
        double* pa = (double*)A.Address, pb = (double*)B.Address, po = (double*)O.Address;
        Console.WriteLine($"page offsets mod 4096: A {(long)pa % 4096}  B {(long)pb % 4096}  O {(long)po % 4096}   (mod 64: {(long)pa % 64} {(long)pb % 64} {(long)po % 64})");
        long m = n - 1024;   // every shifted pointer stays inside its buffer
        for (int rep = 0; rep < 5; rep++)
        {
            double t0 = BestMs(() => K.AddRaw(pa, pb, po, m), 15);
            double t1 = BestMs(() => K.AddRaw(pa, pb + 16, po + 32, m), 15);
            double t2 = BestMs(() => K.AddRaw(pa, pb + 8, po + 16, m), 15);
            Console.WriteLine($"rep{rep}: natural (same) offsets {t0:F3}   B+128B/O+256B {t1:F3}   B+64B/O+128B {t2:F3}");
        }
    }
}

static unsafe class K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1];
        if (ss == 8 && so == 8) { Buffer.MemoryCopy(ps, po, count * 8, count * 8); return; }
        for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddRaw(double* pa, double* pb, double* po, long count)
    {
        long i = 0;
        for (; i + 8 <= count; i += 8)
        {
            Vector256.Store(Vector256.Load(pa + i) + Vector256.Load(pb + i), po + i);
            Vector256.Store(Vector256.Load(pa + i + 4) + Vector256.Load(pb + i + 4), po + i + 4);
        }
        for (; i < count; i++) po[i] = pa[i] + pb[i];
    }
}
