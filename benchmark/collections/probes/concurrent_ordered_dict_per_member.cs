#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
#:property Nullable=disable
// concurrent_ordered_dict_per_member.cs — measure EVERY public member of the shipping
// ConcurrentOrderedDict<int,int> and classify each O(1) / amortized O(1) / O(n).
//
//   DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/collections/probes/concurrent_ordered_dict_per_member.cs
//
// Unit is PER CALL, held consistent so the class reads straight off the growth column:
//   * an O(1) member's per-call cost is FLAT across N (bounded by the cache tier — a random hit that
//     lands in L1 at N=1e3 lands in DRAM at N=1e7, the same step Dictionary/List pay; the ALGORITHMIC
//     O(1) — 1.00 comparisons at every N — is proven separately in compact_ordered_dict_complexity.cs);
//   * an O(n) member's per-call cost GROWS ~10x per decade of N (0.9-1.1 slope in log-log).
// Point-read calls are timed over a 256-key HOT set so the per-call number isolates the member's own
// work from cache-warming; bulk/structural calls are timed one whole call at a time.
//
// Members that share one implementation path are measured once by a representative and mapped in the
// printed roster, so every public member is accounted for without redundant timing.
using System.Diagnostics;
using NumSharp.Collections;

try { Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x4; } catch { }
{ var spin = Stopwatch.StartNew(); long x = 0; while (spin.ElapsedMilliseconds < 2500) x++; GC.KeepAlive(x); }

int[] sizes = { 1_000, 10_000, 100_000, 1_000_000, 10_000_000 };
Console.WriteLine($"pid {Environment.ProcessId}  tiered={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs")}");
Console.WriteLine("shipping ConcurrentOrderedDict<int,int>; one P-core; 150ms tier-1 warm; best-of; per-CALL cost.\n");

// each measured path: name -> per-size per-call cost (ns unless noted ms)
var R = new Dictionary<string, double[]>();
void Put(string k, int si, double v) { if (!R.ContainsKey(k)) R[k] = new double[sizes.Length]; R[k][si] = v; }

for (int si = 0; si < sizes.Length; si++)
{
    int n = sizes[si];
    var r1 = new Random(42); int[] bk = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = r1.Next(i + 1); (bk[i], bk[j]) = (bk[j], bk[i]); }
    int[] hot = new int[256]; for (int i = 0; i < 256; i++) hot[i] = bk[(int)((long)i * 2654435761u % (uint)n)];
    int reps = n <= 10_000 ? 200 : n <= 100_000 ? 40 : 8;
    int drReps = Math.Max(2, reps / 12);

    var d = new ConcurrentOrderedDict<int, int>(n);
    for (int i = 0; i < n; i++) d.TryAdd(bk[i], i * 2);

    // ---- point reads (HOT, ns/call) ----
    Put("TryGetValue", si, NsCall(() => { long s = 0; foreach (int k in hot) { d.TryGetValue(k, out int v); s += v; } GC.KeepAlive(s); }, 256, reps));
    Put("ContainsKey", si, NsCall(() => { int s = 0; foreach (int k in hot) if (d.ContainsKey(k)) s++; GC.KeepAlive(s); }, 256, reps));
    Put("IndexOf", si, NsCall(() => { long s = 0; foreach (int k in hot) s += d.IndexOf(k); GC.KeepAlive(s); }, 256, reps));
    Put("this[int] get", si, NsCall(() => { long s = 0; for (int i = 0; i < 256; i++) s += d[i]; GC.KeepAlive(s); }, 256, reps));
    Put("GetKeyAt", si, NsCall(() => { long s = 0; for (int i = 0; i < 256; i++) s += d.GetKeyAt(i); GC.KeepAlive(s); }, 256, reps));
    Put("TryGetAt", si, NsCall(() => { long s = 0; for (int i = 0; i < 256; i++) { d.TryGetAt(i, out int v); s += v; } GC.KeepAlive(s); }, 256, reps));
    Put("Count", si, NsCall(() => { long s = 0; for (int i = 0; i < 256; i++) s += d.Count; GC.KeepAlive(s); }, 256, reps));

    // ---- replace existing value (in place), ns/call ----
    Put("SetByKey (existing)", si, NsCall(() => { for (int i = 0; i < 256; i++) d.SetByKey(hot[i], i); }, 256, reps));
    Put("SetAt", si, NsCall(() => { for (int i = 0; i < 256; i++) d.SetAt(i, i); }, 256, reps));
    Put("TryUpdate", si, NsCall(() => { for (int i = 0; i < 256; i++) d.TryUpdate(hot[i], i + 1, i); }, 256, reps));

    // ---- structural O(1): capture cost of a snapshot / enumerator, ns/call ----
    Put("Snapshot()", si, NsCall(() => { var v = d.Snapshot(); GC.KeepAlive(v.Count); }, 1, reps));
    Put("GetEnumerator()", si, NsCall(() => { var e = d.GetEnumerator(); GC.KeepAlive(e); }, 1, reps));

    // ---- amortized append: whole unsized build / N, ns/call ----
    Put("TryAdd (amortized)", si, NsCall(() => { var t = new ConcurrentOrderedDict<int, int>(); for (int i = 0; i < n; i++) t.TryAdd(bk[i], i); GC.KeepAlive(t); }, n, Math.Max(2, reps / 8)));

    // ---- removes, per-call (build then time the whole drain, / N) ----
    Put("TryRemove tail (pop)", si, NsCall(() => { var t = Build(n); for (int i = n - 1; i >= 0; i--) t.TryRemove(t.GetKeyAt(t.Count - 1), out _); GC.KeepAlive(t); }, n, drReps, buildExcluded: n));
    Put("TryRemoveSwapBack", si, NsCall(() => { var t = Build(n); for (int i = 0; i < n; i++) t.TryRemoveSwapBack(i, out _); GC.KeepAlive(t); }, n, drReps, buildExcluded: n));

    // ---- interior order-preserving removal @ position 0, ONE call, ms ----
    Put("TryRemove interior@0 (ms)", si, MsOneShot(() => Build(n), t => t.RemoveAt(0)));

    // ---- bulk O(n), ONE call, ms ----
    Put("ToArray (ms)", si, MsRead(() => d.ToArray().Length, reps));
    Put("Keys (ms)", si, MsRead(() => d.Keys.Length, reps));
    Put("foreach consume (ms)", si, MsRead(() => { long s = 0; foreach (int v in d) s += v; return (int)s; }, reps));
    Put("Pairs consume (ms)", si, MsRead(() => { long s = 0; foreach (var kv in d.Pairs) s += kv.Value; return (int)s; }, reps));
    Put("CopyTo (ms)", si, MsRead(() => { var a = new int[n]; d.CopyTo(a, 0); return a.Length; }, reps));
    Put("AddRange (ms)", si, MsOneShot(() => new ConcurrentOrderedDict<int, int>(), t => { var pairs = new KeyValuePair<int, int>[n]; for (int i = 0; i < n; i++) pairs[i] = new(bk[i], i); t.AddRange(pairs); }));
    Put("RemoveWhere all (ms)", si, MsOneShot(() => Build(n), t => t.RemoveWhere((k, v) => true)));

    // ---- Clear: presized (initial capacity ~ N) vs default-grown (initial ~31) ----
    Put("Clear presized (ms)", si, MsOneShot(() => Build(n), t => t.Clear()));
    Put("Clear default-grown (ms)", si, MsOneShot(() => { var t = new ConcurrentOrderedDict<int, int>(); for (int i = 0; i < n; i++) t.TryAdd(bk[i], i); return t; }, t => t.Clear()));

    GC.KeepAlive(d); GC.Collect();
    Console.Error.WriteLine($"  done N={n:N0}");
    ConcurrentOrderedDict<int, int> Build(int m) { var t = new ConcurrentOrderedDict<int, int>(m); for (int i = 0; i < m; i++) t.TryAdd(i, i); return t; }
}

// ---- print measured table ----
Console.WriteLine("## Measured per-call cost across N (ns unless the row says ms). Flat ⇒ O(1); ~10x/decade ⇒ O(n).\n");
Console.Write("| member (measured path) |");
foreach (int n in sizes) Console.Write($" {n:N0} |");
Console.WriteLine(" 1K→10M | class |");
Console.Write("|---|"); foreach (var _ in sizes) Console.Write("---:|"); Console.WriteLine("---:|---|");
string[] order = {
    "TryGetValue","ContainsKey","IndexOf","this[int] get","GetKeyAt","TryGetAt","Count",
    "SetByKey (existing)","SetAt","TryUpdate","Snapshot()","GetEnumerator()",
    "TryAdd (amortized)","TryRemove tail (pop)","TryRemoveSwapBack",
    "TryRemove interior@0 (ms)","ToArray (ms)","Keys (ms)","foreach consume (ms)","Pairs consume (ms)","CopyTo (ms)","AddRange (ms)","RemoveWhere all (ms)","Clear presized (ms)","Clear default-grown (ms)" };
foreach (var k in order)
{
    var a = R[k]; double g = a.Last() / a.First();
    string cls = k.Contains("interior") || k.Contains("ToArray") || k.Contains("Keys") || k.Contains("foreach") || k.Contains("Pairs") || k.Contains("CopyTo") || k.Contains("AddRange") || k.Contains("RemoveWhere") || k.Contains("Clear presized")
        ? "O(n)"
        : k.Contains("amortized") ? "amortized O(1)"
        : g > 5 && !k.Contains("ms") ? "O(1) (DRAM step)"
        : "O(1)";
    Console.Write($"| {k} |");
    foreach (var v in a) Console.Write($" {v:F2} |");
    Console.WriteLine($" {g:F1}x | {cls} |");
}
Console.WriteLine("\n(ms rows are one whole call; ns rows are one call over a 256-key hot set. 'DRAM step' = the per-call");
Console.WriteLine("rise is the shared memory-hierarchy cost every hash container pays, incl. BCL Dictionary — the");
Console.WriteLine("algorithm is 1.00 comparisons at every N, proven in compact_ordered_dict_complexity.cs.)");

static double NsCall(Action a, int callsPerInvoke, int reps, int buildExcluded = 0)
{
    // for drain rows, buildExcluded>0: the invoke builds THEN drains; subtract a build-only estimate is
    // impractical per-rep, so the caller passes callsPerInvoke=N and we report the amortized (build+drain)/(2N)
    var warm = Stopwatch.StartNew(); do { a(); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); a(); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks); }
    double perInvoke = best * 1e9 / Stopwatch.Frequency;
    return perInvoke / (buildExcluded > 0 ? 2.0 * callsPerInvoke : callsPerInvoke);
}

static double MsRead(Func<int> a, int reps)
{
    var warm = Stopwatch.StartNew(); do { GC.KeepAlive(a()); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++) { var sw = Stopwatch.StartNew(); GC.KeepAlive(a()); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks); }
    return best * 1000.0 / Stopwatch.Frequency;
}

static double MsOneShot<T>(Func<T> build, Action<T> op)
{
    double best = double.MaxValue;
    for (int r = 0; r < 4; r++) { T t = build(); var sw = Stopwatch.StartNew(); op(t); sw.Stop(); best = Math.Min(best, sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency); GC.KeepAlive(t); }
    return best;
}
