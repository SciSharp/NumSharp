#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
#:property Nullable=disable
// complexity_probe.cs — O(1) proof + ratios + memory for ConcurrentOrderedDictionary and the recommended
// open-addressed compact table, across N = 1e3 .. 1e7 (four orders of magnitude).
//
//   DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/collections/probes/compact_ordered_dict_complexity.cs
//
// The proof has three independent legs:
//   A. WORK per op, hardware-independent — the average number of slots the probe visits per lookup,
//      counted directly (not timed). If it is flat across N, the algorithm is O(1) by definition.
//   B. TIME per op, cache-isolated — look up a SMALL fixed hot set (256 keys) drawn from an N-sized
//      table, repeatedly, so the probed lines stay in L1/L2. Flat across N ⇒ the per-op instruction/
//      memory work does not grow with N ⇒ O(1). (The full-random column alongside shows the real-world
//      cost including the DRAM step, which every contender — incl. the proven-O(1) Dictionary — pays
//      identically, so it is shared memory latency, not algorithmic growth.)
//   C. The O(n) aspects, stated honestly — order-preserving interior removal, and whole-collection
//      scans (enumerate/build) — measured to CONFIRM the linear scaling rather than hide it.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Collections;
using NumSharp.Collections.Concurrent;
using SysCd = System.Collections.Concurrent.ConcurrentDictionary<int, int>;

try { Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x4; } catch { }
{ var spin = Stopwatch.StartNew(); long x = 0; while (spin.ElapsedMilliseconds < 2500) x++; GC.KeepAlive(x); }

int[] sizes = { 1_000, 10_000, 100_000, 1_000_000, 10_000_000 };
Console.WriteLine($"pid {Environment.ProcessId}  tiered={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs")}");
Console.WriteLine("regime: build keys = random permutation of 0..N-1; lookups = independent permutation; one P-core; 150ms tier-1 warm; best-of.\n");

// ---- Leg A: average probe length (comparisons per lookup), OA + chained, across N -----------------
Console.WriteLine("## A. WORK per lookup — average slots visited (hardware-independent). Flat across N ⇒ O(1).");
Console.WriteLine("| N | OA hit | OA miss | chained hit | chained miss |");
Console.WriteLine("|---:|---:|---:|---:|---:|");
foreach (int n in sizes)
{
    var (bk, lk, miss) = Perms(n);
    var oa = new OA(n); var ch = new Chained(n);
    for (int i = 0; i < n; i++) { oa.Add(bk[i], i); ch.Add(bk[i], i); }
    Console.WriteLine($"| {n:N0} | {Avg(lk, oa.ProbeLen):F3} | {Avg(miss, oa.ProbeLen):F3} | {Avg(lk, ch.ProbeLen):F3} | {Avg(miss, ch.ProbeLen):F3} |");
}
Console.WriteLine("\nOther point ops are constant-work by construction: this[int]/GetKeyAt/TryGetAt = one array index;");
Console.WriteLine("Count = one field read; tail pop / swap-back = a fixed number of stores; IndexOf = the same probe as a hit.\n");

// ---- Leg B: time per op across N, cache-isolated hot set vs full random --------------------------
// Report each contender's ns/op at every size, HOT (256-key working set, isolates algorithm) and FULL
// (whole-table working set, includes the shared DRAM step). Ratios computed afterwards.
Console.WriteLine("## B. TIME per op across N (ns). HOT = 256-key hot set (algorithmic); FULL = whole-table (incl. DRAM).");
string[] ops = { "get-hit HOT", "get-hit FULL", "get-miss FULL", "add (amortized)", "this[int] FULL", "IndexOf HOT", "tail pop", "swap-back" };
var rows = new Dictionary<string, Dictionary<string, double[]>>(); // contender -> op -> [per size]
string[] names = { "List", "Dictionary", "ConcurrentDictionary", "COD (shipping)", "compact OA" };
foreach (var nm in names) { rows[nm] = new(); foreach (var op in ops) rows[nm][op] = new double[sizes.Length]; }

for (int si = 0; si < sizes.Length; si++)
{
    int n = sizes[si];
    var (bk, lk, miss) = Perms(n);
    int[] hot = new int[256]; for (int i = 0; i < 256; i++) hot[i] = bk[(int)((long)i * 2654435761u % (uint)n)];
    int reps = n <= 10_000 ? 200 : n <= 100_000 ? 40 : 8;

    // build the shared read fixtures
    var list = new List<int>(n); for (int i = 0; i < n; i++) list.Add(i * 2);
    var dict = new Dictionary<int, int>(n); for (int i = 0; i < n; i++) dict[bk[i]] = i * 2;
    var scd = new SysCd(1, n); for (int i = 0; i < n; i++) scd[bk[i]] = i * 2;
    var cod = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) cod.TryAdd(bk[i], i * 2);
    var oa = new OA(n); for (int i = 0; i < n; i++) oa.Add(bk[i], i * 2);

    // get-hit HOT
    rows["Dictionary"]["get-hit HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) { dict.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / 256;
    rows["ConcurrentDictionary"]["get-hit HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) { scd.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / 256;
    rows["COD (shipping)"]["get-hit HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) { cod.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / 256;
    rows["compact OA"]["get-hit HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) { oa.TryGet(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / 256;

    // get-hit FULL
    rows["Dictionary"]["get-hit FULL"][si] = Best(() => { long s = 0; foreach (int k in lk) { dict.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / n;
    rows["ConcurrentDictionary"]["get-hit FULL"][si] = Best(() => { long s = 0; foreach (int k in lk) { scd.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / n;
    rows["COD (shipping)"]["get-hit FULL"][si] = Best(() => { long s = 0; foreach (int k in lk) { cod.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / n;
    rows["compact OA"]["get-hit FULL"][si] = Best(() => { long s = 0; foreach (int k in lk) { oa.TryGet(k, out int v); s += v; } GC.KeepAlive(s); }, reps) / n;

    // get-miss FULL
    rows["Dictionary"]["get-miss FULL"][si] = Best(() => { long s = 0; foreach (int k in miss) if (dict.TryGetValue(k, out _)) s++; GC.KeepAlive(s); }, reps) / n;
    rows["ConcurrentDictionary"]["get-miss FULL"][si] = Best(() => { long s = 0; foreach (int k in miss) if (scd.TryGetValue(k, out _)) s++; GC.KeepAlive(s); }, reps) / n;
    rows["COD (shipping)"]["get-miss FULL"][si] = Best(() => { long s = 0; foreach (int k in miss) if (cod.TryGetValue(k, out _)) s++; GC.KeepAlive(s); }, reps) / n;
    rows["compact OA"]["get-miss FULL"][si] = Best(() => { long s = 0; foreach (int k in miss) if (oa.TryGet(k, out _)) s++; GC.KeepAlive(s); }, reps) / n;

    // add (amortized over a full unsized build)
    int addReps = Math.Max(3, reps / 6);
    rows["List"]["add (amortized)"][si] = Best(() => { var l = new List<int>(); for (int i = 0; i < n; i++) l.Add(i); GC.KeepAlive(l); }, addReps) / n;
    rows["Dictionary"]["add (amortized)"][si] = Best(() => { var d = new Dictionary<int, int>(); for (int i = 0; i < n; i++) d[bk[i]] = i; GC.KeepAlive(d); }, addReps) / n;
    rows["ConcurrentDictionary"]["add (amortized)"][si] = Best(() => { var d = new SysCd(); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i); GC.KeepAlive(d); }, addReps) / n;
    rows["COD (shipping)"]["add (amortized)"][si] = Best(() => { var d = new ConcurrentOrderedDictionary<int, int>(); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i); GC.KeepAlive(d); }, addReps) / n;
    rows["compact OA"]["add (amortized)"][si] = Best(() => { var d = new OA(0); for (int i = 0; i < n; i++) d.Add(bk[i], i); GC.KeepAlive(d); }, addReps) / n;

    // this[int] FULL (positional; index order 0..n-1 sequential is the natural scan)
    rows["List"]["this[int] FULL"][si] = Best(() => { long s = 0; for (int i = 0; i < n; i++) s += list[i]; GC.KeepAlive(s); }, reps) / n;
    rows["COD (shipping)"]["this[int] FULL"][si] = Best(() => { long s = 0; for (int i = 0; i < n; i++) s += cod[i]; GC.KeepAlive(s); }, reps) / n;
    rows["compact OA"]["this[int] FULL"][si] = Best(() => { long s = 0; for (int i = 0; i < n; i++) s += oa[i]; GC.KeepAlive(s); }, reps) / n;

    // IndexOf HOT (key->position; hot set)
    rows["COD (shipping)"]["IndexOf HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) s += cod.IndexOf(k); GC.KeepAlive(s); }, reps) / 256;
    rows["compact OA"]["IndexOf HOT"][si] = Best(() => { long s = 0; foreach (int k in hot) s += oa.IndexOf(k); GC.KeepAlive(s); }, reps) / 256;

    // tail pop (build+drain from the back; per-op)
    int drReps = Math.Max(3, reps / 8);
    rows["ConcurrentDictionary"]["tail pop"][si] = Best(() => { var d = new SysCd(1, n); for (int i = 0; i < n; i++) d.TryAdd(i, i); for (int i = n - 1; i >= 0; i--) d.TryRemove(i, out _); GC.KeepAlive(d); }, drReps) / (2.0 * n);
    rows["COD (shipping)"]["tail pop"][si] = Best(() => { var d = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(i, i); for (int i = n - 1; i >= 0; i--) d.TryRemove(d.GetKeyAt(d.Count - 1), out _); GC.KeepAlive(d); }, drReps) / (2.0 * n);
    rows["compact OA"]["tail pop"][si] = Best(() => { var d = new OA(n); for (int i = 0; i < n; i++) d.Add(i, i); for (int i = n - 1; i >= 0; i--) d.PopBack(); GC.KeepAlive(d); }, drReps) / (2.0 * n);

    // swap-back (front-order drain)
    rows["COD (shipping)"]["swap-back"][si] = Best(() => { var d = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(i, i); for (int i = 0; i < n; i++) d.TryRemoveSwapBack(i, out _); GC.KeepAlive(d); }, drReps) / (2.0 * n);
    rows["compact OA"]["swap-back"][si] = Best(() => { var d = new OA(n); for (int i = 0; i < n; i++) d.Add(i, i); for (int i = 0; i < n; i++) d.SwapBack(i); GC.KeepAlive(d); }, drReps) / (2.0 * n);

    GC.KeepAlive(list); GC.KeepAlive(dict); GC.KeepAlive(scd); GC.KeepAlive(cod); GC.KeepAlive(oa);
    GC.Collect();
}

foreach (var op in ops)
{
    Console.WriteLine($"\n### {op} — ns/op");
    Console.Write("| contender |");
    foreach (int n in sizes) Console.Write($" {n:N0} |");
    Console.WriteLine(" 1K→10M growth |");
    Console.Write("|---|"); foreach (var _ in sizes) Console.Write("---:|"); Console.WriteLine("---:|");
    foreach (var nm in names)
    {
        var a = rows[nm][op];
        if (a.All(v => v == 0)) continue;
        Console.Write($"| {nm} |");
        foreach (var v in a) Console.Write(v == 0 ? " — |" : $" {v:F2} |");
        double first = a.First(v => v > 0), last = a.Last(v => v > 0);
        Console.WriteLine($" {last / first:F1}× |");
    }
}

// ---- Ratios: COD and OA vs each baseline, per op, at 1M -------------------------------------------
Console.WriteLine("\n## Ratios at N = 1,000,000 (baseline_ns / contender_ns; >1 = contender faster).");
Console.WriteLine("| op | COD vs List | COD vs Dict | COD vs CD | OA vs List | OA vs Dict | OA vs CD |");
Console.WriteLine("|---|---:|---:|---:|---:|---:|---:|");
int m1 = Array.IndexOf(sizes, 1_000_000);
foreach (var op in ops)
{
    double cod = rows["COD (shipping)"][op][m1], oa = rows["compact OA"][op][m1];
    double list = rows["List"][op][m1], dict = rows["Dictionary"][op][m1], cd = rows["ConcurrentDictionary"][op][m1];
    string R(double baseline, double c) => (baseline > 0 && c > 0) ? $"{baseline / c:F2}×" : "—";
    Console.WriteLine($"| {op} | {R(list, cod)} | {R(dict, cod)} | {R(cd, cod)} | {R(list, oa)} | {R(dict, oa)} | {R(cd, oa)} |");
}

// ---- Leg C: the O(n) aspects, confirmed linear -----------------------------------------------------
Console.WriteLine("\n## C. The O(n) aspects (NOT O(1) — stated, then confirmed linear).");
Console.WriteLine("\n### Order-preserving interior removal — one removal at position p, ms. Cost ∝ (n − p): removing the FRONT is O(n), the TAIL is O(1).");
Console.WriteLine("| N | COD @0 | COD @n/2 | COD @n−1 | OA @0 | OA @n/2 | OA @n−1 |");
Console.WriteLine("|---:|---:|---:|---:|---:|---:|---:|");
foreach (int n in new[] { 100_000, 1_000_000, 10_000_000 })
{
    var (bk, _, _) = Perms(n);
    double C(int p, bool oa)
    {
        double best = double.MaxValue;
        for (int r = 0; r < 4; r++)
        {
            if (oa) { var d = new OA(n); for (int i = 0; i < n; i++) d.Add(i, i); var sw = Stopwatch.StartNew(); d.RemoveAt(p); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency); GC.KeepAlive(d); }
            else { var d = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(i, i); var sw = Stopwatch.StartNew(); d.RemoveAt(p); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency); GC.KeepAlive(d); }
        }
        return best;
    }
    Console.WriteLine($"| {n:N0} | {C(0, false):F3} | {C(n / 2, false):F3} | {C(n - 1, false):F3} | {C(0, true):F3} | {C(n / 2, true):F3} | {C(n - 1, true):F3} |");
}

// ---- Memory: bytes/entry across N, presized, full-GC deltas --------------------------------------
Console.WriteLine("\n## D. Memory — bytes per entry, presized, full-GC delta. Flat across N ⇒ O(1) space per entry (O(n) total).");
Console.WriteLine("| N | List | Dictionary | ConcurrentDictionary | COD (shipping) | compact OA | compact chained |");
Console.WriteLine("|---:|---:|---:|---:|---:|---:|---:|");
foreach (int n in sizes)
{
    var (bk, _, _) = Perms(n);
    double F(Func<object> b)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long before = GC.GetTotalMemory(true); object o = b(); long after = GC.GetTotalMemory(true); GC.KeepAlive(o);
        return (after - before) / (double)n;
    }
    double list = F(() => { var l = new List<int>(n); for (int i = 0; i < n; i++) l.Add(i); return l; });
    double dict = F(() => { var d = new Dictionary<int, int>(n); for (int i = 0; i < n; i++) d[bk[i]] = i; return d; });
    double cd = F(() => { var d = new SysCd(1, n); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i); return d; });
    double cod = F(() => { var d = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i); return d; });
    double oa = F(() => { var d = new OA(n); for (int i = 0; i < n; i++) d.Add(bk[i], i); return d; });
    double ch = F(() => { var d = new Chained(n); for (int i = 0; i < n; i++) d.Add(bk[i], i); return d; });
    Console.WriteLine($"| {n:N0} | {list:F1} | {dict:F1} | {cd:F1} | {cod:F1} | {oa:F1} | {ch:F1} |");
}

Sanity();
Console.WriteLine("\nsanity: OK");

static (int[] build, int[] lookup, int[] miss) Perms(int n)
{
    var r1 = new Random(42); var bk = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = r1.Next(i + 1); (bk[i], bk[j]) = (bk[j], bk[i]); }
    var r2 = new Random(4242); var lk = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = r2.Next(i + 1); (lk[i], lk[j]) = (lk[j], lk[i]); }
    return (bk, lk, lk.Select(k => k + n).ToArray());
}

static double Avg(int[] keys, Func<int, int> probe) { long p = 0; foreach (int k in keys) p += probe(k); return (double)p / keys.Length; }

static double Best(Action a, int reps)
{
    var warm = Stopwatch.StartNew(); do { a(); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); a(); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks); }
    return best * 1e9 / Stopwatch.Frequency;
}

static void Sanity()
{
    var oa = new OA(4); var ch = new Chained(4); var oracle = new List<int>(); var rng = new Random(7);
    for (int op = 0; op < 20_000; op++)
    {
        int kind = rng.Next(10);
        if (kind < 6 || oracle.Count == 0) { int k = rng.Next(5000); bool a = oa.Add(k, k * 3), b = ch.Add(k, k * 3); bool o = !oracle.Contains(k); if (o) oracle.Add(k); if (a != o || b != o) throw new Exception("add"); }
        else if (kind < 8) { int i = rng.Next(oracle.Count); oa.RemoveAt(i); ch.RemoveAt(i); oracle.RemoveAt(i); }
        else if (kind < 9) { int i = rng.Next(oracle.Count); int k = oracle[i]; if (!oa.SwapBack(k) || !ch.SwapBack(k)) throw new Exception("swap"); int last = oracle[^1]; oracle[i] = last; oracle.RemoveAt(oracle.Count - 1); }
        else { oa.PopBack(); ch.PopBack(); oracle.RemoveAt(oracle.Count - 1); }
        if (op % 500 == 0 || op > 19_900)
        {
            if (oa.Count != oracle.Count || ch.Count != oracle.Count) throw new Exception("count");
            for (int i = 0; i < oracle.Count; i++) { int k = oracle[i]; if (oa[i] != k * 3 || ch[i] != k * 3) throw new Exception("idx"); if (!oa.TryGet(k, out int v) || v != k * 3 || !ch.TryGet(k, out int w) || w != k * 3) throw new Exception("get"); if (oa.IndexOf(k) != i || ch.IndexOf(k) != i) throw new Exception("indexof"); }
        }
    }
}

// ================= the recommended open-addressed compact table (instrumented with ProbeLen) ========
sealed class OA
{
    sealed class T { public readonly long[] ix; public readonly int shift; public readonly int[] keys; public readonly int[] vals; public int count; public readonly int floor; public int dummies;
        public T(long[] ix, int[] k, int[] v, int c, int f, int d) { this.ix = ix; shift = 64 - System.Numerics.BitOperations.Log2((uint)ix.Length); keys = k; vals = v; count = c; floor = f; dummies = d; } }
    volatile T _t; readonly object _l = new();
    public OA(int cap) { int c = Math.Max(cap, 4); _t = new T(new long[Sz(c)], new int[c], new int[c], 0, 0, 0); }
    static int Sz(int c) => (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(8, (long)c * 3 / 2 + 1));
    public int Count => Volatile.Read(ref _t.count);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] static uint Home(T t, int k) => (uint)((ulong)(uint)k * 0x9E3779B97F4A7C15UL >> t.shift);
    static long Pack(int k, int s) => ((long)k << 32) | (uint)s;
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGet(int key, out int value)
    { T t = _t; long[] ix = t.ix; uint mask = (uint)ix.Length - 1; uint p = Home(t, key);
      while (true) { long e = Volatile.Read(ref ix[p]); int s = (int)e; if (s == 0) { value = 0; return false; } if (s > 0 && (int)(e >> 32) == key) { value = t.vals[s - 1]; if (Volatile.Read(ref ix[p]) == e) return true; p = Home(t, key); continue; } p = (p + 1) & mask; } }
    // counts the slots visited to resolve `key` (hit or miss) — the untimed O(1) proof
    public int ProbeLen(int key) { T t = _t; long[] ix = t.ix; uint mask = (uint)ix.Length - 1; uint p = Home(t, key); int n = 0; while (true) { n++; long e = ix[p]; int s = (int)e; if (s == 0) return n; if (s > 0 && (int)(e >> 32) == key) return n; p = (p + 1) & mask; } }
    public int IndexOf(int key) { T t = _t; int p = Probe(t, key); return p < 0 ? -1 : (int)t.ix[p] - 1; }
    static int Probe(T t, int key) { long[] ix = t.ix; uint mask = (uint)ix.Length - 1; uint p = Home(t, key); while (true) { long e = ix[p]; int s = (int)e; if (s == 0) return -1; if (s > 0 && (int)(e >> 32) == key) return (int)p; p = (p + 1) & mask; } }
    public int this[int i] { get { T t = _t; if ((uint)i >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(); return t.vals[i]; } }
    public bool Add(int key, int value) { lock (_l) {
        T t = _t; { long[] ix = t.ix; uint m = (uint)ix.Length - 1; uint q = Home(t, key); while (true) { long e = ix[q]; int sf = (int)e; if (sf == 0) break; if (sf > 0 && (int)(e >> 32) == key) return false; q = (q + 1) & m; } }
        int idx = t.count; T g = t; bool fresh = false;
        if (idx == t.keys.Length || idx < t.floor || (long)(idx + 1 + t.dummies) * 3 > (long)t.ix.Length * 2) { g = Copy(t, idx == t.keys.Length ? t.keys.Length * 2 : t.keys.Length); fresh = true; }
        long[] jx = g.ix; uint mask = (uint)jx.Length - 1; uint p = Home(g, key); while ((int)jx[p] != 0) p = (p + 1) & mask;
        g.keys[idx] = key; g.vals[idx] = value;
        if (fresh) { jx[p] = Pack(key, idx + 1); g.count = idx + 1; _t = g; } else { Volatile.Write(ref jx[p], Pack(key, idx + 1)); Volatile.Write(ref g.count, idx + 1); }
        return true; } }
    static T Copy(T t, int newCap) { int n = t.count; var k = new int[newCap]; var v = new int[newCap]; Array.Copy(t.keys, k, n); Array.Copy(t.vals, v, n); var ix = new long[Sz(newCap)]; var g = new T(ix, k, v, n, 0, 0); uint mask = (uint)ix.Length - 1; for (int j = 0; j < n; j++) { uint p = Home(g, k[j]); while ((int)ix[p] != 0) p = (p + 1) & mask; ix[p] = Pack(k[j], j + 1); } return g; }
    public bool PopBack() { lock (_l) { T t = _t; int n = t.count; if (n == 0) return false; int p = Probe(t, t.keys[n - 1]); t.ix[p] = Pack(t.keys[n - 1], -1); _t = new T(t.ix, t.keys, t.vals, n - 1, Math.Max(t.floor, n), t.dummies + 1); return true; } }
    public void RemoveAt(int i) { lock (_l) { T t = _t; int n = t.count; if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(); if (i == n - 1) { int p = Probe(t, t.keys[n - 1]); t.ix[p] = Pack(t.keys[n - 1], -1); _t = new T(t.ix, t.keys, t.vals, n - 1, Math.Max(t.floor, n), t.dummies + 1); return; }
        int cap = t.keys.Length; var k = new int[cap]; var v = new int[cap]; Array.Copy(t.keys, k, i); Array.Copy(t.keys, i + 1, k, i, n - i - 1); Array.Copy(t.vals, v, i); Array.Copy(t.vals, i + 1, v, i, n - i - 1);
        long[] old = t.ix; var ix = new long[old.Length]; int rs = i + 1; for (int p = 0; p < old.Length; p++) { long e = old[p]; int s = (int)e; if (s == rs) ix[p] = Pack((int)(e >> 32), -1); else if (s > rs) ix[p] = e - 1; else ix[p] = e; }
        _t = new T(ix, k, v, n - 1, 0, t.dummies + 1); } }
    public bool SwapBack(int key) { lock (_l) { T t = _t; int n = t.count; int p = Probe(t, key); if (p < 0) return false; int i = (int)t.ix[p] - 1;
        if (i == n - 1) { t.ix[p] = Pack(key, -1); _t = new T(t.ix, t.keys, t.vals, n - 1, Math.Max(t.floor, n), t.dummies + 1); return true; }
        int last = n - 1; t.ix[p] = Pack(key, -1); int kl = t.keys[last]; t.keys[i] = kl; t.vals[i] = t.vals[last]; int pl = Probe(t, kl); Volatile.Write(ref t.ix[pl], Pack(kl, i + 1)); _t = new T(t.ix, t.keys, t.vals, n - 1, Math.Max(t.floor, n), t.dummies + 1); return true; } }
}

// ================= chained no-hash compact table (context; also ProbeLen-instrumented) =============
sealed class Chained
{
    sealed class T { public readonly int[] buckets; public readonly ulong fm; public readonly int[] next; public readonly int[] keys; public readonly int[] vals; public int count; public readonly int floor;
        public T(int[] b, int[] nx, int[] k, int[] v, int c, int f) { buckets = b; fm = HashHelpers.GetFastModMultiplier((uint)b.Length); next = nx; keys = k; vals = v; count = c; floor = f; } }
    volatile T _t; readonly object _l = new();
    public Chained(int cap) { int c = Math.Max(cap, 4); _t = new T(new int[HashHelpers.GetPrime(c)], new int[c], new int[c], new int[c], 0, 0); }
    public int Count => Volatile.Read(ref _t.count);
    [MethodImpl(MethodImplOptions.AggressiveInlining)] static uint B(T t, int h) => HashHelpers.FastMod((uint)h, (uint)t.buckets.Length, t.fm);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGet(int key, out int value) { T t = _t; int i = Volatile.Read(ref t.buckets[B(t, key)]) - 1; while ((uint)i < (uint)t.keys.Length) { if (t.keys[i] == key) { value = t.vals[i]; return true; } i = Volatile.Read(ref t.next[i]); } value = 0; return false; }
    public int ProbeLen(int key) { T t = _t; int i = t.buckets[B(t, key)] - 1; int n = 1; while ((uint)i < (uint)t.keys.Length) { if (t.keys[i] == key) return n; i = t.next[i]; n++; } return n; }
    public int IndexOf(int key) { T t = _t; for (int i = Volatile.Read(ref t.buckets[B(t, key)]) - 1; (uint)i < (uint)t.keys.Length; i = Volatile.Read(ref t.next[i])) if (t.keys[i] == key) return i; return -1; }
    public int this[int i] { get { T t = _t; if ((uint)i >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(); return t.vals[i]; } }
    public bool Add(int key, int value) { lock (_l) { T t = _t; for (int i = t.buckets[B(t, key)] - 1; (uint)i < (uint)t.keys.Length; i = t.next[i]) if (t.keys[i] == key) return false;
        int idx = t.count; T g = t; bool fresh = false; if (idx == t.keys.Length || idx < t.floor) { g = Copy(t, idx == t.keys.Length ? t.keys.Length * 2 : t.keys.Length); fresh = true; }
        uint b = B(g, key); g.keys[idx] = key; g.vals[idx] = value; g.next[idx] = g.buckets[b] - 1; if (fresh) { g.buckets[b] = idx + 1; g.count = idx + 1; _t = g; } else { Volatile.Write(ref g.buckets[b], idx + 1); Volatile.Write(ref g.count, idx + 1); } return true; } }
    static T Copy(T t, int newCap) { int n = t.count; var nx = new int[newCap]; var k = new int[newCap]; var v = new int[newCap]; Array.Copy(t.keys, k, n); Array.Copy(t.vals, v, n); int[] b;
        if (newCap == t.keys.Length) { b = (int[])t.buckets.Clone(); Array.Copy(t.next, nx, n); } else { b = new int[HashHelpers.GetPrime(newCap)]; ulong fm = HashHelpers.GetFastModMultiplier((uint)b.Length); for (int j = 0; j < n; j++) { uint bb = HashHelpers.FastMod((uint)k[j], (uint)b.Length, fm); nx[j] = b[bb] - 1; b[bb] = j + 1; } } return new T(b, nx, k, v, n, 0); }
    static void Unlink(T t, int i) { uint b = B(t, t.keys[i]); int j = t.buckets[b] - 1; if (j == i) { t.buckets[b] = t.next[i] + 1; return; } while (j >= 0) { int nx = t.next[j]; if (nx == i) { t.next[j] = t.next[i]; return; } j = nx; } throw new InvalidOperationException(); }
    public bool PopBack() { lock (_l) { T t = _t; int n = t.count; if (n == 0) return false; Unlink(t, n - 1); _t = new T(t.buckets, t.next, t.keys, t.vals, n - 1, Math.Max(t.floor, n)); return true; } }
    public void RemoveAt(int i) { lock (_l) { T t = _t; int n = t.count; if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(); if (i == n - 1) { Unlink(t, i); _t = new T(t.buckets, t.next, t.keys, t.vals, n - 1, Math.Max(t.floor, n)); return; }
        int cap = t.keys.Length; var k = new int[cap]; var v = new int[cap]; var nx = new int[cap]; Array.Copy(t.keys, k, i); Array.Copy(t.keys, i + 1, k, i, n - i - 1); Array.Copy(t.vals, v, i); Array.Copy(t.vals, i + 1, v, i, n - i - 1);
        int skip = t.next[i]; int[] on = t.next; for (int j = 0; j < i; j++) nx[j] = Map(on[j], i, skip); for (int j = i + 1; j < n; j++) nx[j - 1] = Map(on[j], i, skip); int[] ob = t.buckets; var b = new int[ob.Length]; for (int q = 0; q < b.Length; q++) b[q] = Map(ob[q] - 1, i, skip) + 1; _t = new T(b, nx, k, v, n - 1, 0); } }
    public bool SwapBack(int key) { lock (_l) { T t = _t; int n = t.count; int i = IdxL(t, key); if (i < 0) return false; if (i == n - 1) { Unlink(t, i); _t = new T(t.buckets, t.next, t.keys, t.vals, n - 1, Math.Max(t.floor, n)); return true; }
        int last = n - 1; Unlink(t, i); int kl = t.keys[last]; t.keys[i] = kl; t.vals[i] = t.vals[last]; uint b = B(t, kl); t.next[i] = t.buckets[b] - 1; Volatile.Write(ref t.buckets[b], i + 1); Unlink(t, last); _t = new T(t.buckets, t.next, t.keys, t.vals, n - 1, Math.Max(t.floor, n)); return true; } }
    static int IdxL(T t, int key) { for (int i = t.buckets[B(t, key)] - 1; (uint)i < (uint)t.keys.Length; i = t.next[i]) if (t.keys[i] == key) return i; return -1; }
    [MethodImpl(MethodImplOptions.AggressiveInlining)] static int Map(int j, int r, int skip) { if (j == r) j = skip; return j > r ? j - 1 : j; }
}
