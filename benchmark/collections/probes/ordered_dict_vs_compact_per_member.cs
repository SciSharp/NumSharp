#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
#:property Nullable=disable
// ordered_dict_vs_compact_per_member.cs — measure EVERY public member of BOTH shipping types
// (ConcurrentOrderedDictionary<int,int> and its compact sibling ConcurrentOrderedCompactDictionary<int,int>) across
// N = 1e3 .. 1e7, classify each O(1)/amortized O(1)/O(n), and print them side by side.
//
//   DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/collections/probes/ordered_dict_vs_compact_per_member.cs
//
// Both types expose the IDENTICAL public surface (a drop-in sibling), so one loop-level adapter is
// implemented once per type and one driver times both through the same paths. Unit is PER CALL, held
// consistent so the class reads off the growth column: an O(1) member is flat across N (bounded only by
// the cache tier — a random hit that sits in L1 at 1e3 sits in DRAM at 1e7, the shared step every hash
// container incl. BCL Dictionary pays); an O(n) member grows ~10x per decade. Point reads are timed over
// a 256-key HOT set so the per-call figure is the member's own work, not cache-warming. The algorithmic
// O(1) (comparisons per lookup, flat at 1.00) is proven separately in compact_ordered_dict_complexity.cs.
//
// <int,int> is used throughout (it matches every prior table). It is ALSO the compact type's best case:
// a <=4-byte primitive key is bit-tagged into the index word, so a hit never touches keys[]. A wide or
// custom-compared key would cost the compact type one extra line per hit (still O(1)); noted, not swept.
using System.Diagnostics;
using NumSharp.Collections;

try { Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x4; } catch { }
{ var spin = Stopwatch.StartNew(); long x = 0; while (spin.ElapsedMilliseconds < 2500) x++; GC.KeepAlive(x); }

int[] sizes = { 1_000, 10_000, 100_000, 1_000_000, 10_000_000 };
Console.WriteLine($"pid {Environment.ProcessId}  tiered={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs")}");
Console.WriteLine("both shipping types, <int,int>; one P-core; 150ms tier-1 warm; best-of; per-CALL cost.\n");

ICon[] cons = { new CodCon(), new CocdCon() };
// results[typeName][member] = per-size per-call cost
var res = new Dictionary<string, Dictionary<string, double[]>>();
foreach (var c in cons) res[c.Name] = new();
void Put(string ty, string k, int si, double v) { var m = res[ty]; if (!m.ContainsKey(k)) m[k] = new double[sizes.Length]; m[k][si] = v; }

for (int si = 0; si < sizes.Length; si++)
{
    int n = sizes[si];
    var r1 = new Random(42); int[] bk = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = r1.Next(i + 1); (bk[i], bk[j]) = (bk[j], bk[i]); }
    int[] hot = new int[256]; for (int i = 0; i < 256; i++) hot[i] = bk[(int)((long)i * 2654435761u % (uint)n)];
    var pairs = new KeyValuePair<int, int>[n]; for (int i = 0; i < n; i++) pairs[i] = new(bk[i], i);
    int reps = n <= 10_000 ? 200 : n <= 100_000 ? 40 : 8;
    int drReps = Math.Max(2, reps / 12);

    foreach (var c in cons)
    {
        object d = c.Build(n, bk);
        void P(string k, double v) => Put(c.Name, k, si, v);

        P("TryGetValue", NsCall(() => Sink(c.Gets(d, hot)), 256, reps));
        P("ContainsKey", NsCall(() => Sink(c.Contains(d, hot)), 256, reps));
        P("IndexOf", NsCall(() => Sink(c.IndexOfs(d, hot)), 256, reps));
        P("this[int] get", NsCall(() => Sink(c.IdxGet(d)), 256, reps));
        P("GetKeyAt", NsCall(() => Sink(c.KeyAt(d)), 256, reps));
        P("TryGetAt", NsCall(() => Sink(c.TryAt(d)), 256, reps));
        P("Count", NsCall(() => Sink(c.Counts(d)), 256, reps));
        P("SetByKey (existing)", NsCall(() => c.RepSetByKey(d, hot), 256, reps));
        P("SetAt", NsCall(() => c.RepSetAt(d), 256, reps));
        P("TryUpdate", NsCall(() => c.RepTryUpdate(d, hot), 256, reps));
        P("Snapshot()", NsCall(() => Sink(c.Snap(d)), 1, reps));
        P("GetEnumerator()", NsCall(() => Sink(c.Enum(d)), 1, reps));

        P("TryAdd (amortized)", NsCall(() => { var t = c.NewEmpty(); c.FillKeys(t, bk); GC.KeepAlive(t); }, n, Math.Max(2, reps / 8)));
        P("TryRemove tail/pop", NsCall(() => { var t = c.NewEmpty(); c.FillSeq(t, n); c.PopAll(t, n); GC.KeepAlive(t); }, n, drReps, drain: true));
        P("TryRemoveSwapBack", NsCall(() => { var t = c.NewEmpty(); c.FillSeq(t, n); c.SwapAll(t, n); GC.KeepAlive(t); }, n, drReps, drain: true));

        P("TryRemove interior@0 (ms)", MsOneShot(() => { var t = c.NewEmpty(); c.FillSeq(t, n); return t; }, c.RemoveAt0));
        P("ToArray (ms)", MsRead(() => c.ToArr(d), reps));
        P("Keys (ms)", MsRead(() => c.KeysArr(d), reps));
        P("foreach (ms)", MsRead(() => c.Foreach(d), reps));
        P("Pairs (ms)", MsRead(() => c.PairsC(d), reps));
        P("CopyTo (ms)", MsRead(() => { var buf = new int[n]; return c.CopyToArr(d, buf); }, reps));
        P("AddRange (ms)", MsOneShot(() => (object)null, _ => { var t = c.AddRangeBuild(pairs); GC.KeepAlive(t); }));
        P("RemoveWhere all (ms)", MsOneShot(() => { var t = c.NewEmpty(); c.FillSeq(t, n); return t; }, t => c.RemoveWhereAll(t)));
        P("Clear presized (ms)", MsOneShot(() => c.Build(n, bk), c.Clear));
        P("Clear default-grown (ms)", MsOneShot(() => { var t = c.NewEmpty(); c.FillKeys(t, bk); return t; }, c.Clear));

        GC.KeepAlive(d); GC.Collect();
    }
    Console.Error.WriteLine($"  done N={n:N0}");
}

// ---- print: per member, both types side by side, with a per-type class ----
string[] order = {
    "TryGetValue","ContainsKey","IndexOf","this[int] get","GetKeyAt","TryGetAt","Count",
    "SetByKey (existing)","SetAt","TryUpdate","Snapshot()","GetEnumerator()",
    "TryAdd (amortized)","TryRemove tail/pop","TryRemoveSwapBack",
    "TryRemove interior@0 (ms)","ToArray (ms)","Keys (ms)","foreach (ms)","Pairs (ms)","CopyTo (ms)","AddRange (ms)","RemoveWhere all (ms)","Clear presized (ms)","Clear default-grown (ms)" };

Console.WriteLine("## Per-member per-call cost, COD vs COCD (ns unless ms). @1M column + 1K->10M growth + class.\n");
Console.WriteLine("| member | COD @1M | COD 1K→10M | COD class | COCD @1M | COCD 1K→10M | COCD class | COCD/COD @10M |");
Console.WriteLine("|---|---:|---:|---|---:|---:|---|---:|");
int m1 = Array.IndexOf(sizes, 1_000_000);
foreach (var k in order)
{
    var a = res["COD"][k]; var b = res["COCD"][k];
    Console.WriteLine($"| {k} | {a[m1]:F2} | {a[^1] / a[0]:F1}x | {Cls(k, a)} | {b[m1]:F2} | {b[^1] / b[0]:F1}x | {Cls(k, b)} | {(b[^1] > 0 && a[^1] > 0 ? (a[^1] / b[^1]).ToString("F2") + "x" : "-")} |");
}
Console.WriteLine("\n(COCD/COD @10M > 1 => the compact type is faster at 10M. ms rows: one whole call; ns rows: one call over a 256-key hot set.)");

// ---- per-N detail for the O(1) reads (prove flat on BOTH) ----
Console.WriteLine("\n## O(1) reads across N — flat both ways (ns/call).");
Console.WriteLine("| member | type | 1K | 10K | 100K | 1M | 10M |");
Console.WriteLine("|---|---|---:|---:|---:|---:|---:|");
foreach (var k in new[] { "TryGetValue", "this[int] get", "IndexOf", "SetByKey (existing)" })
    foreach (var ty in new[] { "COD", "COCD" })
    {
        var a = res[ty][k];
        Console.WriteLine($"| {k} | {ty} | {a[0]:F2} | {a[1]:F2} | {a[2]:F2} | {a[3]:F2} | {a[4]:F2} |");
    }

Sanity();
Console.WriteLine("\nsanity: OK");

static string Cls(string k, double[] a)
{
    double g = a[^1] / a[0];
    bool ms = k.Contains("ms");
    if (k.Contains("interior") || k.StartsWith("ToArray") || k.StartsWith("Keys") || k.StartsWith("foreach") || k.StartsWith("Pairs") || k.StartsWith("CopyTo") || k.StartsWith("RemoveWhere"))
        return "O(n)";
    if (k.StartsWith("AddRange")) return "O(m)";
    if (k.StartsWith("Clear presized")) return "O(cap)";
    if (k.StartsWith("Clear default")) return "O(1)";
    if (k.Contains("amortized")) return "amort O(1)";
    return "O(1)"; // reads/replaces/pop/swap: flat or DRAM-step, both O(1)
}

static void Sink(long v) => GC.KeepAlive(v); // int callers widen implicitly

static double NsCall(Action a, int callsPerInvoke, int reps, bool drain = false)
{
    var warm = Stopwatch.StartNew(); do { a(); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); a(); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks); }
    double perInvoke = best * 1e9 / Stopwatch.Frequency;
    return perInvoke / (drain ? 2.0 * callsPerInvoke : callsPerInvoke); // drain invoke = build + drain, amortized
}

static double MsRead(Func<long> a, int reps)
{
    var warm = Stopwatch.StartNew(); do { GC.KeepAlive(a()); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); GC.KeepAlive(a()); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks); }
    return best * 1000.0 / Stopwatch.Frequency;
}

static double MsOneShot(Func<object> build, Action<object> op)
{
    double best = double.MaxValue;
    for (int r = 0; r < 4; r++) { object t = build(); var sw = Stopwatch.StartNew(); op(t); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency); GC.KeepAlive(t); }
    return best;
}

// structural cross-check of both types against a List oracle (correctness gate for every number above)
static void Sanity()
{
    var cod = new ConcurrentOrderedDictionary<int, int>(4);
    var cocd = new ConcurrentOrderedCompactDictionary<int, int>(4);
    var oracle = new List<int>(); var rng = new Random(7);
    for (int op = 0; op < 20_000; op++)
    {
        int kind = rng.Next(10);
        if (kind < 6 || oracle.Count == 0) { int k = rng.Next(5000); bool a = cod.TryAdd(k, k * 3), b = cocd.TryAdd(k, k * 3); bool o = !oracle.Contains(k); if (o) oracle.Add(k); if (a != o || b != o) throw new Exception("add"); }
        else if (kind < 8) { int i = rng.Next(oracle.Count); cod.RemoveAt(i); cocd.RemoveAt(i); oracle.RemoveAt(i); }
        else if (kind < 9) { int i = rng.Next(oracle.Count); int k = oracle[i]; if (!cod.TryRemoveSwapBack(k, out _) || !cocd.TryRemoveSwapBack(k, out _)) throw new Exception("swap"); int last = oracle[^1]; oracle[i] = last; oracle.RemoveAt(oracle.Count - 1); }
        else { int k = cod.GetKeyAt(cod.Count - 1); cod.TryRemove(k, out _); cocd.TryRemove(cocd.GetKeyAt(cocd.Count - 1), out _); oracle.RemoveAt(oracle.Count - 1); }
        if (op % 500 == 0 || op > 19_900)
        {
            if (cod.Count != oracle.Count || cocd.Count != oracle.Count) throw new Exception("count");
            for (int i = 0; i < oracle.Count; i++) { int k = oracle[i]; if (!cod.TryGetValue(k, out int v) || v != k * 3 || !cocd.TryGetValue(k, out int w) || w != k * 3) throw new Exception("get"); if (cod.IndexOf(k) != i || cocd.IndexOf(k) != i) throw new Exception("indexof"); }
        }
    }
}

/// <summary>Loop-level adapter so each measured op runs inside a concretely-typed method (no per-element virtual call), letting one driver time both shipping types fairly.</summary>
interface ICon
{
    /// <summary>Row label for the printed tables.</summary>
    string Name { get; }
    /// <summary>Builds an instance presized to <paramref name="n"/> holding key=bk[i], value=i*2.</summary>
    object Build(int n, int[] bk);
    /// <summary>A fresh default-capacity (unsized) instance.</summary>
    object NewEmpty();
    /// <summary>Appends key=bk[i], value=i (the unsized-build workload).</summary>
    void FillKeys(object d, int[] bk);
    /// <summary>Appends key=i, value=i for i in [0,n) (so key == position; the drain fixtures).</summary>
    void FillSeq(object d, int n);
    /// <summary>Sums TryGetValue over the hot keys.</summary>
    long Gets(object d, int[] hot);
    /// <summary>Counts ContainsKey hits over the hot keys.</summary>
    long Contains(object d, int[] hot);
    /// <summary>Sums IndexOf over the hot keys.</summary>
    long IndexOfs(object d, int[] hot);
    /// <summary>Sums this[i] for i in [0,256).</summary>
    long IdxGet(object d);
    /// <summary>Sums GetKeyAt(i) for i in [0,256).</summary>
    long KeyAt(object d);
    /// <summary>Sums TryGetAt(i) for i in [0,256).</summary>
    long TryAt(object d);
    /// <summary>Reads Count 256 times (defeats hoisting via a data dependency in the caller sink).</summary>
    long Counts(object d);
    /// <summary>Replaces via SetByKey on the hot keys (existing).</summary>
    void RepSetByKey(object d, int[] hot);
    /// <summary>Replaces via SetAt(i,...) for i in [0,256).</summary>
    void RepSetAt(object d);
    /// <summary>Replaces via TryUpdate on the hot keys.</summary>
    void RepTryUpdate(object d, int[] hot);
    /// <summary>Takes a Snapshot() and returns its Count (structural-capture cost).</summary>
    int Snap(object d);
    /// <summary>Creates a GetEnumerator() and returns a constant (structural-capture cost).</summary>
    int Enum(object d);
    /// <summary>Drains every entry from the tail (pop-back).</summary>
    void PopAll(object d, int n);
    /// <summary>Drains every entry by swap-back in front order.</summary>
    void SwapAll(object d, int n);
    /// <summary>Removes the entry at position 0 (one order-preserving interior removal).</summary>
    void RemoveAt0(object d);
    /// <summary>Materializes ToArray, returns its length.</summary>
    long ToArr(object d);
    /// <summary>Materializes Keys, returns its length.</summary>
    long KeysArr(object d);
    /// <summary>Consumes the value enumerator, returns the value sum.</summary>
    long Foreach(object d);
    /// <summary>Consumes Pairs, returns the value sum.</summary>
    long PairsC(object d);
    /// <summary>CopyTo the buffer, returns its length.</summary>
    long CopyToArr(object d, int[] buf);
    /// <summary>Builds a fresh instance via AddRange(pairs).</summary>
    object AddRangeBuild(KeyValuePair<int, int>[] pairs);
    /// <summary>RemoveWhere(all), returns the count removed.</summary>
    int RemoveWhereAll(object d);
    /// <summary>Clears the instance.</summary>
    void Clear(object d);
}

/// <summary>Adapter for the shipping node-plus-arrays <see cref="ConcurrentOrderedDictionary{TKey,TValue}"/>.</summary>
sealed class CodCon : ICon
{
    /// <inheritdoc/>
    public string Name => "COD";
    /// <inheritdoc/>
    public object Build(int n, int[] bk) { var d = new ConcurrentOrderedDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i * 2); return d; }
    /// <inheritdoc/>
    public object NewEmpty() => new ConcurrentOrderedDictionary<int, int>();
    /// <inheritdoc/>
    public void FillKeys(object d, int[] bk) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < bk.Length; i++) t.TryAdd(bk[i], i); }
    /// <inheritdoc/>
    public void FillSeq(object d, int n) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < n; i++) t.TryAdd(i, i); }
    /// <inheritdoc/>
    public long Gets(object d, int[] hot) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; foreach (int k in hot) { t.TryGetValue(k, out int v); s += v; } return s; }
    /// <inheritdoc/>
    public long Contains(object d, int[] hot) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; foreach (int k in hot) if (t.ContainsKey(k)) s++; return s; }
    /// <inheritdoc/>
    public long IndexOfs(object d, int[] hot) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; foreach (int k in hot) s += t.IndexOf(k); return s; }
    /// <inheritdoc/>
    public long IdxGet(object d) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t[i]; return s; }
    /// <inheritdoc/>
    public long KeyAt(object d) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t.GetKeyAt(i); return s; }
    /// <inheritdoc/>
    public long TryAt(object d) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) { t.TryGetAt(i, out int v); s += v; } return s; }
    /// <inheritdoc/>
    public long Counts(object d) { var t = (ConcurrentOrderedDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t.Count; return s; }
    /// <inheritdoc/>
    public void RepSetByKey(object d, int[] hot) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.SetByKey(hot[i], i); }
    /// <inheritdoc/>
    public void RepSetAt(object d) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.SetAt(i, i); }
    /// <inheritdoc/>
    public void RepTryUpdate(object d, int[] hot) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.TryUpdate(hot[i], i + 1, i); }
    /// <inheritdoc/>
    public int Snap(object d) => ((ConcurrentOrderedDictionary<int, int>)d).Snapshot().Count;
    /// <inheritdoc/>
    public int Enum(object d) { var e = ((ConcurrentOrderedDictionary<int, int>)d).GetEnumerator(); GC.KeepAlive(e); return 1; }
    /// <inheritdoc/>
    public void PopAll(object d, int n) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = n - 1; i >= 0; i--) t.TryRemove(t.GetKeyAt(t.Count - 1), out _); }
    /// <inheritdoc/>
    public void SwapAll(object d, int n) { var t = (ConcurrentOrderedDictionary<int, int>)d; for (int i = 0; i < n; i++) t.TryRemoveSwapBack(i, out _); }
    /// <inheritdoc/>
    public void RemoveAt0(object d) => ((ConcurrentOrderedDictionary<int, int>)d).RemoveAt(0);
    /// <inheritdoc/>
    public long ToArr(object d) => ((ConcurrentOrderedDictionary<int, int>)d).ToArray().Length;
    /// <inheritdoc/>
    public long KeysArr(object d) => ((ConcurrentOrderedDictionary<int, int>)d).Keys.Length;
    /// <inheritdoc/>
    public long Foreach(object d) { long s = 0; foreach (int v in (ConcurrentOrderedDictionary<int, int>)d) s += v; return s; }
    /// <inheritdoc/>
    public long PairsC(object d) { long s = 0; foreach (var kv in ((ConcurrentOrderedDictionary<int, int>)d).Pairs) s += kv.Value; return s; }
    /// <inheritdoc/>
    public long CopyToArr(object d, int[] buf) { ((ConcurrentOrderedDictionary<int, int>)d).CopyTo(buf, 0); return buf.Length; }
    /// <inheritdoc/>
    public object AddRangeBuild(KeyValuePair<int, int>[] pairs) { var t = new ConcurrentOrderedDictionary<int, int>(); t.AddRange(pairs); return t; }
    /// <inheritdoc/>
    public int RemoveWhereAll(object d) => ((ConcurrentOrderedDictionary<int, int>)d).RemoveWhere((k, v) => true);
    /// <inheritdoc/>
    public void Clear(object d) => ((ConcurrentOrderedDictionary<int, int>)d).Clear();
}

/// <summary>Adapter for the shipping compact open-addressed <see cref="ConcurrentOrderedCompactDictionary{TKey,TValue}"/>.</summary>
sealed class CocdCon : ICon
{
    /// <inheritdoc/>
    public string Name => "COCD";
    /// <inheritdoc/>
    public object Build(int n, int[] bk) { var d = new ConcurrentOrderedCompactDictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(bk[i], i * 2); return d; }
    /// <inheritdoc/>
    public object NewEmpty() => new ConcurrentOrderedCompactDictionary<int, int>();
    /// <inheritdoc/>
    public void FillKeys(object d, int[] bk) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < bk.Length; i++) t.TryAdd(bk[i], i); }
    /// <inheritdoc/>
    public void FillSeq(object d, int n) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < n; i++) t.TryAdd(i, i); }
    /// <inheritdoc/>
    public long Gets(object d, int[] hot) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; foreach (int k in hot) { t.TryGetValue(k, out int v); s += v; } return s; }
    /// <inheritdoc/>
    public long Contains(object d, int[] hot) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; foreach (int k in hot) if (t.ContainsKey(k)) s++; return s; }
    /// <inheritdoc/>
    public long IndexOfs(object d, int[] hot) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; foreach (int k in hot) s += t.IndexOf(k); return s; }
    /// <inheritdoc/>
    public long IdxGet(object d) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t[i]; return s; }
    /// <inheritdoc/>
    public long KeyAt(object d) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t.GetKeyAt(i); return s; }
    /// <inheritdoc/>
    public long TryAt(object d) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) { t.TryGetAt(i, out int v); s += v; } return s; }
    /// <inheritdoc/>
    public long Counts(object d) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; long s = 0; for (int i = 0; i < 256; i++) s += t.Count; return s; }
    /// <inheritdoc/>
    public void RepSetByKey(object d, int[] hot) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.SetByKey(hot[i], i); }
    /// <inheritdoc/>
    public void RepSetAt(object d) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.SetAt(i, i); }
    /// <inheritdoc/>
    public void RepTryUpdate(object d, int[] hot) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < 256; i++) t.TryUpdate(hot[i], i + 1, i); }
    /// <inheritdoc/>
    public int Snap(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).Snapshot().Count;
    /// <inheritdoc/>
    public int Enum(object d) { var e = ((ConcurrentOrderedCompactDictionary<int, int>)d).GetEnumerator(); GC.KeepAlive(e); return 1; }
    /// <inheritdoc/>
    public void PopAll(object d, int n) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = n - 1; i >= 0; i--) t.TryRemove(t.GetKeyAt(t.Count - 1), out _); }
    /// <inheritdoc/>
    public void SwapAll(object d, int n) { var t = (ConcurrentOrderedCompactDictionary<int, int>)d; for (int i = 0; i < n; i++) t.TryRemoveSwapBack(i, out _); }
    /// <inheritdoc/>
    public void RemoveAt0(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).RemoveAt(0);
    /// <inheritdoc/>
    public long ToArr(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).ToArray().Length;
    /// <inheritdoc/>
    public long KeysArr(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).Keys.Length;
    /// <inheritdoc/>
    public long Foreach(object d) { long s = 0; foreach (int v in (ConcurrentOrderedCompactDictionary<int, int>)d) s += v; return s; }
    /// <inheritdoc/>
    public long PairsC(object d) { long s = 0; foreach (var kv in ((ConcurrentOrderedCompactDictionary<int, int>)d).Pairs) s += kv.Value; return s; }
    /// <inheritdoc/>
    public long CopyToArr(object d, int[] buf) { ((ConcurrentOrderedCompactDictionary<int, int>)d).CopyTo(buf, 0); return buf.Length; }
    /// <inheritdoc/>
    public object AddRangeBuild(KeyValuePair<int, int>[] pairs) { var t = new ConcurrentOrderedCompactDictionary<int, int>(); t.AddRange(pairs); return t; }
    /// <inheritdoc/>
    public int RemoveWhereAll(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).RemoveWhere((k, v) => true);
    /// <inheritdoc/>
    public void Clear(object d) => ((ConcurrentOrderedCompactDictionary<int, int>)d).Clear();
}
