#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// fixed_cost_probe.cs — NDIter's FIXED cost, decomposed.
//
// Run (from the repo, any cwd):
//   DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
//     dotnet run -c Release benchmark/nditer/probes/fixed_cost_probe.cs
//
// Sections:
//   A. iterator construction + dispose (ns), and the native-allocator floor it sits on
//   B. per-chunk driver overhead on narrow strided rows (iternext-only / empty kernel / real kernel)
//   C. contiguous 1M / 10M elementwise through the iterator vs a raw pointer loop
//   D. production ufunc out= route fixed cost at n = 1 / 1K / 100K (+ managed bytes per call)
//   E. the individual fixed-cost components (string key vs packed key, Broadcast, closures)
//
// Twin: numpy_twins.py fixed  (same shapes, NumPy 2.4.2). See docs/NDITER_PERF_DISCOVERY.md.
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug-JITted NumSharp.Core — run with -c Release"); return; }
unsafe { Console.WriteLine($"V256={Vector256.IsHardwareAccelerated}  sizeof(NDIterState)={sizeof(NDIterState)}  TC_CallCountingDelayMs={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs") ?? "(default 100)"}"); }

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
double BestMs(Action body, int rounds = 7)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    double best = double.MaxValue;
    for (int r = 0; r < rounds; r++) { var sw = Stopwatch.StartNew(); body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
    return best;
}
long AllocPerCall(Action body, int iters)
{
    for (int i = 0; i < 100; i++) body();
    long before = GC.GetAllocatedBytesForCurrentThread();
    for (int i = 0; i < iters; i++) body();
    return (GC.GetAllocatedBytesForCurrentThread() - before) / iters;
}

var RO_WO = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY };
var RO_RO_WO = new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY };
const NPY_ORDER KO = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;
const NDIterGlobalFlags EXL = NDIterGlobalFlags.EXTERNAL_LOOP;
var ufuncFlags = EXL | NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER | NDIterGlobalFlags.DELAY_BUFALLOC | NDIterGlobalFlags.COPY_IF_OVERLAP | NDIterGlobalFlags.ZEROSIZE_OK;

unsafe
{
    // ---------------- A. construction ----------------
    {
        var a = np.arange(1000).astype(np.float64);
        var b = np.arange(1000).astype(np.float64) + 1.0;
        var o = np.empty(new Shape(1000), np.float64);
        var ops3 = new[] { a, b, o }; var ops2 = new[] { a, o };
        Console.WriteLine("--- A. construction + dispose (ns) ---");
        Console.WriteLine($"New(a) 1op                    : {BestNs(() => { using var it = NDIterRef.New(a); }):F1}");
        Console.WriteLine($"MultiNew 2op EXL              : {BestNs(() => { using var it = NDIterRef.MultiNew(2, ops2, EXL, KO, SAFE, RO_WO); }):F1}");
        Console.WriteLine($"MultiNew 3op EXL              : {BestNs(() => { using var it = NDIterRef.MultiNew(3, ops3, EXL, KO, SAFE, RO_RO_WO); }):F1}");
        Console.WriteLine($"MultiNew 3op ufunc flags      : {BestNs(() => { using var it = NDIterRef.MultiNew(3, ops3, ufuncFlags, KO, SAFE, RO_RO_WO); }):F1}");
        Console.WriteLine($"MultiNew 3op EXL+COPY_IF_OVERLAP: {BestNs(() => { using var it = NDIterRef.MultiNew(3, ops3, EXL | NDIterGlobalFlags.COPY_IF_OVERLAP, KO, SAFE, RO_RO_WO); }):F1}");
        Console.WriteLine($"cached state blocks on this thread now: {NDIterRef.CachedStateBlockCount}");
        nuint stateBytes = (nuint)sizeof(NDIterState), dimBytes = 8 + 8 + 3 * 8 + 8, opBytes = 9 * 3 * 8 + 2 * 3 * 4 + 8 + 8;
        Console.WriteLine($"allocator floor: 3x AllocZeroed+Free ({stateBytes}+{dimBytes}+{opBytes} B): {BestNs(() => { var p1 = NativeMemory.AllocZeroed(stateBytes); var p2 = NativeMemory.AllocZeroed(dimBytes); var p3 = NativeMemory.AllocZeroed(opBytes); NativeMemory.Free(p2); NativeMemory.Free(p3); NativeMemory.Free(p1); }):F1}   1x AllocZeroed+Free ({stateBytes + dimBytes + opBytes} B): {BestNs(() => { var p1 = NativeMemory.AllocZeroed(stateBytes + dimBytes + opBytes); NativeMemory.Free(p1); }):F1}");
    }

    // ---------------- B. per-chunk dispatch ----------------
    Console.WriteLine("--- B. chunk dispatch: strided rows (2M f64 total), ms [ns/chunk] ---");
    foreach (int w in new[] { 4, 16, 64 })
    {
        const int TOTAL = 2_097_152; int rows = TOTAL / w;
        var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
        var sv = back[$":, :{w}"]; var dst = np.empty(new Shape(rows, w), np.float64); var opsw = new[] { sv, dst };
        double tCopy = BestMs(() => { using var it = NDIterRef.MultiNew(2, opsw, EXL, KO, SAFE, RO_WO); it.ForEach(K.CopyF64); });
        double tEmpty = BestMs(() => { using var it = NDIterRef.MultiNew(2, opsw, EXL, KO, SAFE, RO_WO); it.ForEach(K.Empty); });
        double tNext = BestMs(() => { using var it = NDIterRef.MultiNew(2, opsw, EXL, KO, SAFE, RO_WO); var next = it.GetIterNext(); long n = 0; do { n++; } while (next(ref *it.RawState)); if (n != rows) throw new Exception("chunk count"); });
        double tProd = BestMs(() => np.positive(sv, dst));
        Console.WriteLine($"w={w,4} rows={rows,7}: ForEach+CopyF64 {tCopy:F3} [{tCopy * 1e6 / rows:F1}]  ForEach+Empty {tEmpty:F3} [{tEmpty * 1e6 / rows:F1}]  iternext-only {tNext:F3} [{tNext * 1e6 / rows:F1}]  np.positive(out) {tProd:F3} [{tProd * 1e6 / rows:F1}]");
    }

    // ---------------- C. contiguous 1M / 10M ----------------
    Console.WriteLine("--- C. contiguous elementwise 1M / 10M (ms) ---");
    foreach (int n in new[] { 1_000_000, 10_000_000 })
    {
        var A = (np.arange(n).astype(np.float64) % 97.0) + 1.0; var B = (np.arange(n).astype(np.float64) % 31.0) + 2.0; var O = np.empty(new Shape(n), np.float64);
        var add3 = new[] { A, B, O }; var copy2 = new[] { A, O };
        Console.WriteLine($"n={n,9}: ForEach add {BestMs(() => { using var it = NDIterRef.MultiNew(3, add3, EXL, KO, SAFE, RO_RO_WO); it.ForEach(K.AddF64); }):F3}  raw add {BestMs(() => K.AddRaw((double*)A.Address, (double*)B.Address, (double*)O.Address, n)):F3}  | ForEach sqrt {BestMs(() => { using var it = NDIterRef.MultiNew(2, copy2, EXL, KO, SAFE, RO_WO); it.ForEach(K.SqrtF64); }):F3}  | np.add(out) {BestMs(() => np.add(A, B, O)):F3}  np.sqrt(out) {BestMs(() => np.sqrt(A, O)):F3}");
    }

    // ---------------- D. production out= route ----------------
    foreach (int n in new[] { 1, 1000, 100_000 })
    {
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0; var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(n), np.float64); var ob = np.empty(new Shape(n), np.bool_);
        var a2 = (np.arange(2 * n).astype(np.float64) % 53.0) + 1.0; var b2 = (np.arange(2 * n).astype(np.float64) % 17.0) + 1.0;
        var sa = a2["::2"]; var sb = b2["::2"]; var ipa = a.copy();
        Console.WriteLine($"--- D. production routes n={n} (ns/call, managed B/call) ---");
        Console.WriteLine($"np.add(a,b,out=o)          : {BestNs(() => np.add(a, b, o)):F1}   alloc {AllocPerCall(() => np.add(a, b, o), 20_000)} B");
        Console.WriteLine($"np.multiply(a,b,out=a)     : {BestNs(() => np.multiply(ipa, b, ipa)):F1}   alloc {AllocPerCall(() => np.multiply(ipa, b, ipa), 20_000)} B");
        Console.WriteLine($"np.less(a,b,out=ob)        : {BestNs(() => np.less(a, b, ob)):F1}   alloc {AllocPerCall(() => np.less(a, b, ob), 20_000)} B");
        Console.WriteLine($"np.sqrt(a,out=o)           : {BestNs(() => np.sqrt(a, o)):F1}   alloc {AllocPerCall(() => np.sqrt(a, o), 20_000)} B");
        Console.WriteLine($"np.negative(a,out=o)       : {BestNs(() => np.negative(a, o)):F1}");
        Console.WriteLine($"np.positive(a,out=o)       : {BestNs(() => np.positive(a, o)):F1}");
        Console.WriteLine($"np.add(a,b) new+dispose    : {BestNs(() => { var r = np.add(a, b); r.Dispose(); }):F1}");
        Console.WriteLine($"np.add(sa,sb) strided new  : {BestNs(() => { var r = np.add(sa, sb); r.Dispose(); }):F1}");
    }

    // ---------------- E. components ----------------
    {
        var op = BinaryOp.Add; var t = NPTypeCode.Double;
        var A = np.arange(1000).astype(np.float64); var B = A + 1.0;
        Console.WriteLine("--- E. fixed-cost components (ns) ---");
        Console.WriteLine($"interpolated key string (legacy)   : {BestNs(() => { string k = $"npy_binop_{op}_{t}_{t}_{t}"; GC.KeepAlive(k); }):F1}   alloc {AllocPerCall(() => { string k = $"npy_binop_{op}_{t}_{t}_{t}"; GC.KeepAlive(k); }, 100_000)} B");
        var key = InnerLoopKernelKey.Binary(op, t, t, t);
        Console.WriteLine($"packed-key cache probe             : {BestNs(() => { DirectILKernelGenerator.TryGetInnerLoop(key, out var f); GC.KeepAlive(f); }):F1}");
        Console.WriteLine($"Broadcast(lhs,rhs) + Clean          : {BestNs(() => { var (l, r) = DefaultEngine.Broadcast(A.Shape, B.Shape); var c = l.Clean(); GC.KeepAlive(c.dimensions); }):F1}");
        Console.WriteLine($"two emit closures                  : {BestNs(() => { Action<System.Reflection.Emit.ILGenerator> s = il => DirectILKernelGenerator.EmitScalarOperation(il, op, t); Action<System.Reflection.Emit.ILGenerator> v = il => DirectILKernelGenerator.EmitVectorOperation(il, op, t); GC.KeepAlive(s); GC.KeepAlive(v); }):F1}");
        Console.WriteLine($"new NDArray[3] operand array       : {BestNs(() => { var arr = new NDArray[] { A, B, A }; GC.KeepAlive(arr); }):F1}");
        Console.WriteLine($"np.empty(1000) + Dispose           : {BestNs(() => { var r = np.empty(new Shape(1000), np.float64); r.Dispose(); }):F1}");
    }
}

static unsafe class K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Empty(void** dp, long* st, long count, void* aux) { }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1];
        if (ss == 8 && so == 8) { Buffer.MemoryCopy(ps, po, count * 8, count * 8); return; }
        for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SqrtF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1]; long i = 0;
        if (ss == 8 && so == 8)
            for (; i + 4 <= count; i += 4) { Vector256.Store(Vector256.Sqrt(Vector256.Load((double*)ps)), (double*)po); ps += 32; po += 32; }
        for (; i < count; i++) { *(double*)po = Math.Sqrt(*(double*)ps); ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddF64(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0]; byte* pb = (byte*)dp[1]; byte* po = (byte*)dp[2];
        long sa = st[0], sb = st[1], so = st[2]; long i = 0;
        if (so == 8 && sa == 8 && sb == 8)
            for (; i + 8 <= count; i += 8)
            {
                Vector256.Store(Vector256.Load((double*)pa) + Vector256.Load((double*)pb), (double*)po);
                Vector256.Store(Vector256.Load((double*)(pa + 32)) + Vector256.Load((double*)(pb + 32)), (double*)(po + 32));
                pa += 64; pb += 64; po += 64;
            }
        for (; i < count; i++) { *(double*)po = *(double*)pa + *(double*)pb; pa += sa; pb += sb; po += so; }
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
