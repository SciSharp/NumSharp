#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
#:property Nullable=disable
// =============================================================================
// compact_ordered_dict_probe.cs — "fold the Store into a compact table" discovery probe.
//
//   DOTNET_TC_CallCountingDelayMs=0 \
//     dotnet run -c Release benchmark/collections/probes/compact_ordered_dict_probe.cs -- 1000 100000 1000000 10000000
//   PROBE_KEYS=perm            build every table from a random permutation of 0..N-1 (the fair "hashed keys"
//                              regime — sequential int keys land in SEQUENTIAL buckets under mod-prime hashing
//                              and flatter the chaining layouts' builds/drains by 3x)
//   PROBE_ONLY=OA,no-hash,COD  contender filter (substring match on the row name)
//
// Findings + the design they support: src/NumSharp.Core/Collections/Concurrent/ConcurrentOrderedDict.COMPACT.md
//
// Five compact-layout prototypes (one copy of each key/value; slot == insertion-order index; no per-entry heap
// node) are measured against today's ConcurrentOrderedDict, the vendored ConcurrentDictionary clone, the BCL
// ConcurrentDictionary/Dictionary and List<T>: bytes/entry (presized + unsized), build, key hit/miss,
// foreach, this[int], interior order-preserving removal at three positions, pop-back and swap-back drains,
// allocation churn per op, and a 20K-op structural sanity check of every prototype against a List oracle.
//
// Every prototype uses the SAME publication discipline ConcurrentOrderedDict uses today (release-publish of the
// key-path link word, then of the count; readers acquire-read both; shrinking transitions publish a new
// holder; the floor rule freezes vacated tail slots), so the read paths measured here are representative of
// what a production port would run.
//
// Two harness traps this probe had to learn (both leave the numbers wrong by 3-5x if ignored):
//   * tier-0: a JIT-from-IL contender measured over a 2 ms window runs unoptimized code (the BCL types are
//     ReadyToRun and are NOT affected) — Best() warms each op for 150 ms before timing;
//   * key/slot correlation: the lookup permutation must be independent of the build permutation, or every
//     table is probed in its own allocation order and a random-access lookup turns into a prefetch stream.
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NumSharp.Collections;
using NumSharp.Collections.Concurrent;
using SysCd = System.Collections.Concurrent.ConcurrentDictionary<int, int>;
using CloneCd = NumSharp.Collections.Concurrent.ConcurrentDictionary<int, int>;

// One P-core (0x4 = logical CPU 2 on a hybrid part): an unpinned thread can land on an E-core and read 2-3x
// slower for everything. Then spin the clock up — the pinned core idles at a low P-state for the first ~100 ms
// of a process, which inflated every cache-resident row 4-5x in the first run.
try { Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)0x4; } catch { }
{
    var spin = Stopwatch.StartNew(); long x = 0; while (spin.ElapsedMilliseconds < 2500) x++; GC.KeepAlive(x);
}

int[] sizes = args.Length > 0 ? args.Select(int.Parse).ToArray() : new[] { 1_000, 100_000, 1_000_000 };
Console.WriteLine($"pid {Environment.ProcessId}  tiered={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs")}  sizes={string.Join(",", sizes)}");

foreach (int n in sizes)
{
    Console.WriteLine();
    Console.WriteLine($"## N = {n:N0}  (<int,int>)");
    var rng = new Random(42);
    int[] shuffled = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = rng.Next(i + 1); (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]); }
    // The lookup order is an INDEPENDENT permutation (seed 4242): with PROBE_KEYS=perm the build order is
    // `shuffled`, and probing a table in its own build order is a sequential walk, not a random one.
    var rng2 = new Random(4242);
    int[] lookup = Enumerable.Range(0, n).ToArray();
    for (int i = n - 1; i > 0; i--) { int j = rng2.Next(i + 1); (lookup[i], lookup[j]) = (lookup[j], lookup[i]); }
    int[] missing = lookup.Select(k => k + n).ToArray();
    Probe.BuildKeys = Environment.GetEnvironmentVariable("PROBE_KEYS") == "perm" ? shuffled : Enumerable.Range(0, n).ToArray();
    Console.WriteLine($"build keys: {(Probe.BuildKeys == shuffled ? "random permutation" : "sequential")}");

    int reps = n <= 1_000 ? 400 : n <= 100_000 ? 25 : 7;

    IContender[] contenders =
    [
        new ListC(), new DictC(), new SysCdC(), new CloneCdC(), new CodC(), new SoAC(), new SoANHC(), new AoSC(), new SplitC(), new OAC(),
    ];
    if (Environment.GetEnvironmentVariable("PROBE_ONLY") is { Length: > 0 } only)
        contenders = contenders.Where(c => only.Split(',').Any(o => c.Name.Contains(o, StringComparison.OrdinalIgnoreCase))).ToArray();

    // ---- footprint (presized exact N, full-GC deltas — the memory ledger's methodology) + the read/build rows
    Console.WriteLine();
    Console.WriteLine("| contender | B/entry presized | build ns/add | get-hit ns | get-miss ns | foreach ns/elem | this[i] ns/elem |");
    Console.WriteLine("|---|---:|---:|---:|---:|---:|---:|");
    foreach (var c in contenders)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        long before = GC.GetTotalMemory(true);
        c.Build(n);
        long after = GC.GetTotalMemory(true);
        double perEntry = (after - before) / (double)n;

        double buildNs = c.SupportsBuild ? Best(() => { var d = c.NewBuilt(n); GC.KeepAlive(d); }, Math.Max(3, reps / 3)) / n : double.NaN;
        double hitNs = c.SupportsKey ? Best(() => Sink(c.KeyGetSum(lookup)), reps) / n : double.NaN;
        double missNs = c.SupportsKey ? Best(() => Sink(c.KeyMissCount(missing)), reps) / n : double.NaN;
        double enumNs = Best(() => Sink(c.EnumSum()), reps) / n;
        double idxNs = c.SupportsIndex ? Best(() => Sink(c.IndexSum()), reps) / n : double.NaN;
        Console.WriteLine($"| {c.Name} | {perEntry:F1} | {F(buildNs)} | {F(hitNs)} | {F(missNs)} | {F(enumNs)} | {F(idxNs)} |");
        c.Drop();
    }

    // ---- removal: one interior order-preserving removal (ms) at the front / middle / near the tail, then a
    //      pop-back drain and a swap-back drain (ns per op, build included in the op count's denominator only)
    if (n >= 100_000)
    {
        Console.WriteLine();
        Console.WriteLine("| contender | remove @0 ms | remove @n/2 ms | remove @n-2 ms | pop-back drain ns/op | swap-back drain ns/op |");
        Console.WriteLine("|---|---:|---:|---:|---:|---:|");
        foreach (var c in contenders)
        {
            if (!c.SupportsRemove) continue;
            double r0 = BestRemove(c, n, 0), rMid = BestRemove(c, n, n / 2), rTail = BestRemove(c, n, n - 2);
            double drain = Best(() => { var d = c.NewBuilt(n); for (int i = n - 1; i >= 0; i--) c.PopBack(d); GC.KeepAlive(d); }, 3) / n;
            double swap = c.SupportsSwapBack ? Best(() => { var d = c.NewBuilt(n); for (int i = 0; i < n; i++) c.SwapBack(d, i); GC.KeepAlive(d); }, 3) / n : double.NaN;
            Console.WriteLine($"| {c.Name} | {r0:F3} | {rMid:F3} | {rTail:F3} | {F(drain)} | {F(swap)} |");
        }
    }
}

// ---- the ledger's wider-value rows (analytically compared in the findings)
Console.WriteLine();
Console.WriteLine("## Footprint B/entry at N = 1,000,000 for wider values (presized)");
Console.WriteLine("| shape | CloneCD | COD today | compact SoA | compact AoS | compact split | compact OA |");
Console.WriteLine("|---|---:|---:|---:|---:|---:|---:|");
Console.WriteLine($"| <int,long> | {Foot(1_000_000, () => { var d = new CloneCd(1, 1_000_000, null); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1}* | {Foot(1_000_000, () => { var d = new ConcurrentOrderedDict<int, long>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactSoA<int, long>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactAoS<int, long>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactSplit<int, long>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactOA<long>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} |");
Console.WriteLine($"| <int,decimal> | {Foot(1_000_000, () => { var d = new CloneCd(1, 1_000_000, null); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1}* | {Foot(1_000_000, () => { var d = new ConcurrentOrderedDict<int, decimal>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactSoA<int, decimal>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactAoS<int, decimal>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactSplit<int, decimal>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(1_000_000, () => { var d = new CompactOA<decimal>(1_000_000); for (int i = 0; i < 1_000_000; i++) d.TryAdd(i, i); return d; }):F1} |");
Console.WriteLine("(* CloneCD column is the <int,int> clone — the ledger's CD<int,decimal> is 57.3)");

// ---- unsized builds: ALL of a compact table's per-entry bytes sit in doubling arrays (up to 2x slack) where
//      COD's 40 B node is exact, so the worst case matters: 600K sits at 57% of the 1,048,576 doubling step
//      (near-worst), 1M at 95% (near-best).
Console.WriteLine();
Console.WriteLine("## Footprint B/entry, UNSIZED builds (<int,int>)");
Console.WriteLine("| N | CloneCD | COD today | compact SoA | compact AoS | compact split | compact OA |");
Console.WriteLine("|---|---:|---:|---:|---:|---:|---:|");
foreach (int un in new[] { 600_000, 1_000_000 })
{
    Console.WriteLine($"| {un:N0} | {Foot(un, () => { var d = new CloneCd(); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(un, () => { var d = new ConcurrentOrderedDict<int, int>(); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(un, () => { var d = new CompactSoA<int, int>(0); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(un, () => { var d = new CompactAoS<int, int>(0); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(un, () => { var d = new CompactSplit<int, int>(0); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} | {Foot(un, () => { var d = new CompactOA<int>(0); for (int i = 0; i < un; i++) d.TryAdd(i, i); return d; }):F1} |");
}

// ---- allocation churn per op (thread-local counter, min-of-rounds): a compact append allocates NOTHING (no
//      node); pops and swap-backs allocate one holder, like COD's Store; the interior removal shows the COW volume
Console.WriteLine();
Console.WriteLine("## Churn B/op (<int,int>, presized 70K holding 64K, min-of-rounds)");
Console.WriteLine("| op | COD today | compact SoA | compact OA |");
Console.WriteLine("|---|---:|---:|---:|");
{
    var cod = new ConcurrentOrderedDict<int, int>(70_000); var soa = new CompactSoA<int, int>(70_000); var oa = new CompactOA<int>(70_000);
    for (int i = 0; i < 64_000; i++) { cod.Add(i, i); soa.TryAdd(i, i); oa.TryAdd(i, i); }
    int k1 = 64_000, k2 = 64_000, k3 = 64_000;
    Console.WriteLine($"| append (in capacity) | {MinAlloc(() => cod.TryAdd(k1++, 1), 200)} | {MinAlloc(() => soa.TryAdd(k2++, 1), 200)} | {MinAlloc(() => oa.TryAdd(k3++, 1), 200)} |");
    Console.WriteLine($"| tail pop | {MinAlloc(() => cod.TryRemove(--k1, out _), 200)} | {MinAlloc(() => soa.TryRemoveLast(), 200)} | {MinAlloc(() => oa.TryRemoveLast(), 200)} |");
    int s1 = 100, s2 = 100, s3 = 100;
    Console.WriteLine($"| swap-back (in place) | {MinAlloc(() => cod.TryRemoveSwapBack(s1++, out _), 200)} | {MinAlloc(() => soa.TryRemoveSwapBack(s2++), 200)} | {MinAlloc(() => oa.TryRemoveSwapBack(s3++), 200)} |");
    Console.WriteLine($"| interior remove @1000 (COW) | {MinAlloc(() => cod.RemoveAt(1000), 20)} | {MinAlloc(() => soa.RemoveAt(1000), 20)} | {MinAlloc(() => oa.RemoveAt(1000), 20)} |");
}

// ---- single-threaded structural sanity of every prototype against a List oracle (the correctness gate
//      behind every number above: the renumbering pass, the swap-back protocols, tail pops, duplicate refusal)
Sanity();
Console.WriteLine("sanity: OK");

/// <summary>Formats a nanosecond figure, or an em dash for an operation the contender does not support.</summary>
/// <param name="v">The value; <see cref="double.NaN" /> means unsupported.</param>
/// <returns>The two-decimal text or "—".</returns>
static string F(double v) => double.IsNaN(v) ? "—" : v.ToString("F2");

/// <summary>Keeps a computed result observable so the JIT cannot dead-code the measured loop.</summary>
/// <param name="v">The value to retain.</param>
static void Sink(long v) => GC.KeepAlive(v);

/// <summary>
///     Best-of-<paramref name="reps" /> wall time of one invocation of <paramref name="a" />, in nanoseconds,
///     after warming the op for 150 ms so tier-1 code is installed BEFORE the timed reps (a 2 ms measurement
///     window over a JIT-from-IL method otherwise times tier-0 code — the trap that made every 1K row 5x slow).
/// </summary>
/// <param name="a">The operation to time.</param>
/// <param name="reps">How many timed invocations to take the minimum over.</param>
/// <returns>The minimum single-invocation time in nanoseconds.</returns>
static double Best(Action a, int reps)
{
    var warm = Stopwatch.StartNew();
    do { a(); } while (warm.ElapsedMilliseconds < 150);
    long best = long.MaxValue;
    for (int r = 0; r < reps; r++)
    {
        var sw = Stopwatch.StartNew();
        a();
        sw.Stop();
        best = Math.Min(best, sw.ElapsedTicks);
    }
    return best * 1e9 / Stopwatch.Frequency;
}

/// <summary>Best-of-4 time (ms) of ONE interior removal at <paramref name="index" /> on a freshly built table of <paramref name="n" /> entries (a removal mutates, so each sample rebuilds).</summary>
/// <param name="c">The contender.</param>
/// <param name="n">The entry count to build.</param>
/// <param name="index">The insertion-order position to remove.</param>
/// <returns>The minimum removal time in milliseconds; page-faulting the fresh COW arrays is part of the cost on every side.</returns>
static double BestRemove(IContender c, int n, int index)
{
    double best = double.MaxValue;
    for (int r = 0; r < 4; r++)
    {
        var d = c.NewBuilt(n);
        var sw = Stopwatch.StartNew();
        c.RemoveAt(d, index);
        sw.Stop();
        best = Math.Min(best, sw.ElapsedTicks * 1000.0 / Stopwatch.Frequency);
        GC.KeepAlive(d);
    }
    return best;
}

/// <summary>Steady-state bytes per entry of a built container: full-GC total-memory delta around <paramref name="build" />, divided by <paramref name="n" />.</summary>
/// <param name="n">The entry count the build produces.</param>
/// <param name="build">Constructs and fills the container; the returned object is kept alive across the measurement.</param>
/// <returns>Bytes per entry retained by the built container.</returns>
static double Foot(int n, Func<object> build)
{
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    long before = GC.GetTotalMemory(true);
    object d = build();
    long after = GC.GetTotalMemory(true);
    GC.KeepAlive(d);
    return (after - before) / (double)n;
}

/// <summary>Minimum managed bytes one invocation of <paramref name="op" /> allocates on this thread (the memory ledger's min-of-rounds pattern; warmup excluded).</summary>
/// <param name="op">The operation to measure.</param>
/// <param name="rounds">Rounds to take the minimum over.</param>
/// <returns>The minimum bytes allocated by a single invocation.</returns>
static long MinAlloc(Action op, int rounds)
{
    for (int i = 0; i < 3; i++) op();
    long best = long.MaxValue;
    for (int r = 0; r < rounds; r++)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        op();
        best = Math.Min(best, GC.GetAllocatedBytesForCurrentThread() - before);
    }
    return best;
}

/// <summary>
///     Drives 20,000 random adds / interior removals / swap-backs / tail pops through every prototype in lockstep
///     with a <see cref="List{T}" /> oracle and audits count, positional values, key lookups, absent keys and
///     <c>IndexOf</c> at every 500th step — throws on the first disagreement.
/// </summary>
/// <exception cref="Exception">A prototype disagreed with the oracle.</exception>
static void Sanity()
{
    var soa = new CompactSoA<int, int>(4);
    var aos = new CompactAoS<int, int>(4);
    var split = new CompactSplit<int, int>(4);
    var oa = new CompactOA<int>(4);
    var nh = new CompactSoANoHash<int, int>(4);
    var oracle = new List<int>();
    var rng = new Random(7);
    for (int op = 0; op < 20_000; op++)
    {
        int kind = rng.Next(10);
        if (kind < 6 || oracle.Count == 0)
        {
            int k = rng.Next(5_000);
            bool a = soa.TryAdd(k, k * 3), b = aos.TryAdd(k, k * 3), c = split.TryAdd(k, k * 3), e = oa.TryAdd(k, k * 3), f = nh.TryAdd(k, k * 3);
            bool o = !oracle.Contains(k);
            if (o) oracle.Add(k);
            if (a != o || b != o || c != o || e != o || f != o) throw new Exception("TryAdd disagree");
        }
        else if (kind < 8)
        {
            int i = rng.Next(oracle.Count);
            soa.RemoveAt(i); aos.RemoveAt(i); split.RemoveAt(i); oa.RemoveAt(i); nh.RemoveAt(i); oracle.RemoveAt(i);
        }
        else if (kind < 9)
        {
            int i = rng.Next(oracle.Count);
            int k = oracle[i];
            if (!soa.TryRemoveSwapBack(k) || !aos.TryRemoveSwapBack(k) || !split.TryRemoveSwapBack(k) || !oa.TryRemoveSwapBack(k) || !nh.TryRemoveSwapBack(k)) throw new Exception("swapback failed");
            int last = oracle[^1];
            oracle[i] = last; oracle.RemoveAt(oracle.Count - 1);
        }
        else
        {
            soa.TryRemoveLast(); aos.TryRemoveLast(); split.TryRemoveLast(); oa.TryRemoveLast(); nh.TryRemoveLast(); oracle.RemoveAt(oracle.Count - 1);
        }

        if (op % 500 == 0 || op > 19_900)
        {
            if (soa.Count != oracle.Count || aos.Count != oracle.Count || split.Count != oracle.Count || oa.Count != oracle.Count || nh.Count != oracle.Count) throw new Exception("count");
            for (int i = 0; i < oracle.Count; i++)
            {
                int k = oracle[i];
                if (soa[i] != k * 3 || aos[i] != k * 3 || split[i] != k * 3 || oa[i] != k * 3 || nh[i] != k * 3) throw new Exception($"index {i}");
                if (!soa.TryGetValue(k, out int v1) || v1 != k * 3) throw new Exception("soa get");
                if (!aos.TryGetValue(k, out int v2) || v2 != k * 3) throw new Exception("aos get");
                if (!split.TryGetValue(k, out int v3) || v3 != k * 3) throw new Exception("split get");
                if (!oa.TryGetValue(k, out int v4) || v4 != k * 3) throw new Exception("oa get");
                if (!nh.TryGetValue(k, out int v5) || v5 != k * 3) throw new Exception("no-hash get");
                if (soa.IndexOf(k) != i || aos.IndexOf(k) != i || split.IndexOf(k) != i || oa.IndexOf(k) != i || nh.IndexOf(k) != i) throw new Exception("indexof");
            }
            for (int k = 0; k < 5_000; k++)
            {
                bool present = oracle.Contains(k);
                if (soa.TryGetValue(k, out _) != present || aos.TryGetValue(k, out _) != present || split.TryGetValue(k, out _) != present || oa.TryGetValue(k, out _) != present || nh.TryGetValue(k, out _) != present) throw new Exception("absent");
            }
        }
    }
}

// ============================================================================================ harness

/// <summary>Process-wide probe settings read by the contenders.</summary>
static class Probe
{
    /// <summary>The key inserted at insertion-order position i by every build (the identity, or a random permutation under <c>PROBE_KEYS=perm</c>).</summary>
    public static int[] BuildKeys;
}

/// <summary>One measured container: the uniform surface the sweep drives, with capability flags for the operations a baseline lacks.</summary>
interface IContender
{
    /// <summary>The row label.</summary>
    string Name { get; }

    /// <summary>Whether the container has a key path (<see cref="KeyGetSum" />/<see cref="KeyMissCount" />).</summary>
    bool SupportsKey { get; }

    /// <summary>Whether the container has positional access (<see cref="IndexSum" />).</summary>
    bool SupportsIndex { get; }

    /// <summary>Whether a build is measurable (<see cref="NewBuilt" />).</summary>
    bool SupportsBuild { get; }

    /// <summary>Whether interior removal / pop-back drains apply.</summary>
    bool SupportsRemove { get; }

    /// <summary>Whether an order-breaking swap-back removal exists.</summary>
    bool SupportsSwapBack { get; }

    /// <summary>Builds and RETAINS the container the read benchmarks run over.</summary>
    /// <param name="n">Entry count.</param>
    void Build(int n);

    /// <summary>Builds a fresh, presized container of <paramref name="n" /> entries (the build benchmark's body and the removal benchmarks' fixture).</summary>
    /// <param name="n">Entry count.</param>
    /// <returns>The built container.</returns>
    object NewBuilt(int n);

    /// <summary>Releases the retained container so the next size starts from a clean heap.</summary>
    void Drop();

    /// <summary>Sums the values found for every key in <paramref name="keys" /> (all present).</summary>
    /// <param name="keys">The lookup order.</param>
    /// <returns>The sum (observed to defeat dead-code elimination).</returns>
    long KeyGetSum(int[] keys);

    /// <summary>Counts hits over <paramref name="keys" /> (all absent, so the count is the miss-path cost).</summary>
    /// <param name="keys">Absent keys.</param>
    /// <returns>The number found (0).</returns>
    long KeyMissCount(int[] keys);

    /// <summary>Sums every value by the container's natural enumeration.</summary>
    /// <returns>The sum.</returns>
    long EnumSum();

    /// <summary>Sums every value by positional index.</summary>
    /// <returns>The sum.</returns>
    long IndexSum();

    /// <summary>Removes the entry at insertion-order position <paramref name="index" />, preserving order.</summary>
    /// <param name="d">A container from <see cref="NewBuilt" />.</param>
    /// <param name="index">The position to remove.</param>
    void RemoveAt(object d, int index);

    /// <summary>Removes the last entry.</summary>
    /// <param name="d">A container from <see cref="NewBuilt" />.</param>
    void PopBack(object d);

    /// <summary>Removes <paramref name="key" /> by the order-breaking swap-back.</summary>
    /// <param name="d">A container from <see cref="NewBuilt" />.</param>
    /// <param name="key">The key to remove.</param>
    void SwapBack(object d, int key);
}

/// <summary>The list-path baseline: what an unkeyed, unsynchronized contiguous scan costs.</summary>
sealed class ListC : IContender
{
    List<int> _l;
    /// <inheritdoc />
    public string Name => "List<int>";
    /// <inheritdoc />
    public bool SupportsKey => false;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => false;
    /// <inheritdoc />
    public bool SupportsSwapBack => false;
    /// <inheritdoc />
    public void Build(int n) { _l = (List<int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var l = new List<int>(n); for (int i = 0; i < n; i++) l.Add(i * 2); return l; }
    /// <inheritdoc />
    public void Drop() => _l = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) => 0;
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) => 0;
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _l) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var l = _l; for (int i = 0; i < l.Count; i++) s += l[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) { }
    /// <inheritdoc />
    public void PopBack(object d) { }
    /// <inheritdoc />
    public void SwapBack(object d, int key) { }
}

/// <summary>The BCL <see cref="Dictionary{TKey,TValue}" /> — the compact chained layout being ported, as a calibration row (ReadyToRun-compiled, so immune to the tier-0 trap).</summary>
sealed class DictC : IContender
{
    Dictionary<int, int> _d;
    /// <inheritdoc />
    public string Name => "Dictionary<int,int>";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => false;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => false;
    /// <inheritdoc />
    public bool SupportsSwapBack => false;
    /// <inheritdoc />
    public void Build(int n) { _d = (Dictionary<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new Dictionary<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (var kv in _d) s += kv.Value; return s; }
    /// <inheritdoc />
    public long IndexSum() => 0;
    /// <inheritdoc />
    public void RemoveAt(object d, int index) { }
    /// <inheritdoc />
    public void PopBack(object d) { }
    /// <inheritdoc />
    public void SwapBack(object d, int key) { }
}

/// <summary>The framework <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}" /> — the key-path baseline (one stripe, presized).</summary>
sealed class SysCdC : IContender
{
    SysCd _d;
    /// <inheritdoc />
    public string Name => "SysCD";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => false;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => false;
    /// <inheritdoc />
    public bool SupportsSwapBack => false;
    /// <inheritdoc />
    public void Build(int n) { _d = (SysCd)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new SysCd(1, n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (var kv in _d) s += kv.Value; return s; }
    /// <inheritdoc />
    public long IndexSum() => 0;
    /// <inheritdoc />
    public void RemoveAt(object d, int index) { }
    /// <inheritdoc />
    public void PopBack(object d) { }
    /// <inheritdoc />
    public void SwapBack(object d, int key) { }
}

/// <summary>The vendored clone today's ordered dict is built on — isolates clone-vs-BCL drift from the ordered-dict layer.</summary>
sealed class CloneCdC : IContender
{
    CloneCd _d;
    /// <inheritdoc />
    public string Name => "CloneCD";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => false;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => false;
    /// <inheritdoc />
    public bool SupportsSwapBack => false;
    /// <inheritdoc />
    public void Build(int n) { _d = (CloneCd)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CloneCd(1, n, null); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (var kv in _d) s += kv.Value; return s; }
    /// <inheritdoc />
    public long IndexSum() => 0;
    /// <inheritdoc />
    public void RemoveAt(object d, int index) { }
    /// <inheritdoc />
    public void PopBack(object d) { }
    /// <inheritdoc />
    public void SwapBack(object d, int key) { }
}

/// <summary>Today's <see cref="ConcurrentOrderedDict{TKey,TValue}" /> — the design under review (node key path + parallel key/value arrays).</summary>
sealed class CodC : IContender
{
    ConcurrentOrderedDict<int, int> _d;
    /// <inheritdoc />
    public string Name => "COD today";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (ConcurrentOrderedDict<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new ConcurrentOrderedDict<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _d) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((ConcurrentOrderedDict<int, int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) { var c = (ConcurrentOrderedDict<int, int>)d; c.TryRemove(c.GetKeyAt(c.Count - 1), out _); }
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((ConcurrentOrderedDict<int, int>)d).TryRemoveSwapBack(key, out _);
}

/// <summary>Prototype: chained hash index over parallel <c>hashes/next/keys/values</c> arrays (structure-of-arrays; keeps every span surface).</summary>
sealed class SoAC : IContender
{
    CompactSoA<int, int> _d;
    /// <inheritdoc />
    public string Name => "compact SoA";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (CompactSoA<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CompactSoA<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _d.ValuesSpan()) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((CompactSoA<int, int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) => ((CompactSoA<int, int>)d).TryRemoveLast();
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((CompactSoA<int, int>)d).TryRemoveSwapBack(key);
}

/// <summary>Prototype: the SoA layout without a stored-hash array (a trivially-hashed key is compared directly — one cache line fewer per hit, 4 B/entry less).</summary>
sealed class SoANHC : IContender
{
    CompactSoANoHash<int, int> _d;
    /// <inheritdoc />
    public string Name => "compact SoA no-hash";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (CompactSoANoHash<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CompactSoANoHash<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _d.ValuesSpan()) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((CompactSoANoHash<int, int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) => ((CompactSoANoHash<int, int>)d).TryRemoveLast();
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((CompactSoANoHash<int, int>)d).TryRemoveSwapBack(key);
}

/// <summary>Prototype: the open-addressed index (CPython's <c>dk_indices</c> shape) with the key embedded in the 8-byte index word — a 2-line hit and a validated read.</summary>
sealed class OAC : IContender
{
    CompactOA<int> _d;
    /// <inheritdoc />
    public string Name => "compact OA (open addr.)";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (CompactOA<int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CompactOA<int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _d.ValuesSpan()) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((CompactOA<int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) => ((CompactOA<int>)d).TryRemoveLast();
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((CompactOA<int>)d).TryRemoveSwapBack(key);
}

/// <summary>Prototype: one interleaved <c>Entry[]</c> (array-of-structs, .NET <see cref="Dictionary{TKey,TValue}" />'s layout) — the fewest lines per hit, no value span, 4x the scan bandwidth.</summary>
sealed class AoSC : IContender
{
    CompactAoS<int, int> _d;
    /// <inheritdoc />
    public string Name => "compact AoS";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (CompactAoS<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CompactAoS<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() => _d.EnumSum();
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((CompactAoS<int, int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) => ((CompactAoS<int, int>)d).TryRemoveLast();
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((CompactAoS<int, int>)d).TryRemoveSwapBack(key);
}

/// <summary>Prototype: the hash index <c>{hash,next,key}</c> interleaved, the VALUES contiguous — value scans stay List-parity, keys are positional but not spannable.</summary>
sealed class SplitC : IContender
{
    CompactSplit<int, int> _d;
    /// <inheritdoc />
    public string Name => "compact split";
    /// <inheritdoc />
    public bool SupportsKey => true;
    /// <inheritdoc />
    public bool SupportsIndex => true;
    /// <inheritdoc />
    public bool SupportsBuild => true;
    /// <inheritdoc />
    public bool SupportsRemove => true;
    /// <inheritdoc />
    public bool SupportsSwapBack => true;
    /// <inheritdoc />
    public void Build(int n) { _d = (CompactSplit<int, int>)NewBuilt(n); }
    /// <inheritdoc />
    public object NewBuilt(int n) { var d = new CompactSplit<int, int>(n); for (int i = 0; i < n; i++) d.TryAdd(Probe.BuildKeys[i], i * 2); return d; }
    /// <inheritdoc />
    public void Drop() => _d = null;
    /// <inheritdoc />
    public long KeyGetSum(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) { d.TryGetValue(keys[i], out int v); s += v; } return s; }
    /// <inheritdoc />
    public long KeyMissCount(int[] keys) { long s = 0; var d = _d; for (int i = 0; i < keys.Length; i++) if (d.TryGetValue(keys[i], out _)) s++; return s; }
    /// <inheritdoc />
    public long EnumSum() { long s = 0; foreach (int v in _d.ValuesSpan()) s += v; return s; }
    /// <inheritdoc />
    public long IndexSum() { long s = 0; var d = _d; int n = d.Count; for (int i = 0; i < n; i++) s += d[i]; return s; }
    /// <inheritdoc />
    public void RemoveAt(object d, int index) => ((CompactSplit<int, int>)d).RemoveAt(index);
    /// <inheritdoc />
    public void PopBack(object d) => ((CompactSplit<int, int>)d).TryRemoveLast();
    /// <inheritdoc />
    public void SwapBack(object d, int key) => ((CompactSplit<int, int>)d).TryRemoveSwapBack(key);
}

// ============================================================================================ prototypes
// Shared conventions of the four CHAINED prototypes: buckets[] holds 1-based slot indices (0 = empty), chains
// run through `next` (-1 terminates), slot == insertion-order index (no holes ever: interior removal compacts
// into fresh arrays with a SEQUENTIAL renumbering pass — no hashing, no per-key walks), tail removal = chain
// unlink + lower-count holder + floor (COD's rule), swap-back = the 4-step in-place protocol from the findings
// (which the findings ALSO show cannot keep a chained key path tear-free — the reason the open-addressed
// prototype is the recommended one). Single-threaded: every mutator takes a Monitor so the publication order is
// the one a production port would use, but no reader-side stress is attempted here.

/// <summary>The slot renumbering behind compact-on-delete: a pure index map, so a removal repairs the hash index with a streaming pass instead of re-hashing or re-walking anything.</summary>
static class Renumber
{
    /// <summary>Maps an old slot index to its post-removal index when slot <paramref name="removed" /> is compacted away.</summary>
    /// <param name="j">The old slot index held by a bucket or a chain link (-1 = chain end, passed through).</param>
    /// <param name="removed">The slot being removed.</param>
    /// <param name="skip">The removed slot's own chain successor, so a link INTO the removed slot links past it.</param>
    /// <returns>The index the same entry has after the shift.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Map(int j, int removed, int skip)
    {
        if (j == removed) j = skip;
        return j > removed ? j - 1 : j;
    }
}

/// <summary>Chained compact table over five parallel arrays (buckets, hashes, next, keys, values) — one copy of each key/value, no per-entry node; every span surface of today's <c>ValuesView</c> survives.</summary>
/// <typeparam name="TKey">The key type (default comparer).</typeparam>
/// <typeparam name="TValue">The value type.</typeparam>
sealed class CompactSoA<TKey, TValue> where TKey : notnull
{
    /// <summary>One generation of the table: the arrays plus the monotonic live count and the append floor (COD's Store rules, extended to the hash index).</summary>
    sealed class Tables
    {
        public readonly int[] buckets; public readonly ulong fastMod;
        public readonly int[] hashes; public readonly int[] next; public readonly TKey[] keys; public readonly TValue[] values;
        public int count; public readonly int floor;
        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="buckets">1-based chain heads.</param>
        /// <param name="hashes">Stored hash per slot.</param>
        /// <param name="next">Chain link per slot (-1 = end).</param>
        /// <param name="keys">Keys in insertion order.</param>
        /// <param name="values">Values in insertion order.</param>
        /// <param name="count">Live entries at publication.</param>
        /// <param name="floor">First slot an in-place append may write.</param>
        public Tables(int[] buckets, int[] hashes, int[] next, TKey[] keys, TValue[] values, int count, int floor)
        { this.buckets = buckets; fastMod = HashHelpers.GetFastModMultiplier((uint)buckets.Length); this.hashes = hashes; this.next = next; this.keys = keys; this.values = values; this.count = count; this.floor = floor; }
    }

    volatile Tables _t;
    readonly object _lock = new();

    /// <summary>Creates a table presized for <paramref name="capacity" /> entries (prime bucket count, load ≤ 1).</summary>
    /// <param name="capacity">Entry capacity; at least 4.</param>
    public CompactSoA(int capacity)
    {
        int cap = Math.Max(capacity, 4);
        _t = new Tables(new int[HashHelpers.GetPrime(cap)], new int[cap], new int[cap], new TKey[cap], new TValue[cap], 0, 0);
    }

    /// <summary>The live entry count — one acquire read of the current generation's count.</summary>
    public int Count => Volatile.Read(ref _t.count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static uint Bucket(Tables t, int h) => HashHelpers.FastMod((uint)h, (uint)t.buckets.Length, t.fastMod);

    /// <summary>Lock-free lookup: acquire-read the chain head, walk hashes/keys, read the value — a hit touches four cache lines (buckets, hashes, keys, values).</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value on a hit.</param>
    /// <returns>Whether the key was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        int[] hashes = t.hashes;
        ref int hashes0 = ref MemoryMarshal.GetArrayDataReference(hashes);
        ref int next0 = ref MemoryMarshal.GetArrayDataReference(t.next);
        ref TKey keys0 = ref MemoryMarshal.GetArrayDataReference(t.keys);
        int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1;
        while ((uint)i < (uint)hashes.Length)
        {
            if (Unsafe.Add(ref hashes0, i) == h && EqualityComparer<TKey>.Default.Equals(Unsafe.Add(ref keys0, i), key))
            {
                value = Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(t.values), i);
                return true;
            }
            i = Volatile.Read(ref Unsafe.Add(ref next0, i));
        }
        value = default;
        return false;
    }

    /// <summary>The insertion-order position of <paramref name="key" /> — the slot number itself, no stored index field.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The position, or -1.</returns>
    public int IndexOf(TKey key)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1; (uint)i < (uint)t.hashes.Length; i = Volatile.Read(ref t.next[i]))
            if (t.hashes[i] == h && EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return i;
        return -1;
    }

    /// <summary>Positional read — a plain array read behind the count bound.</summary>
    /// <param name="index">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            if ((uint)index >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(nameof(index));
            return t.values[index];
        }
    }

    /// <summary>The live values as a span (the surface today's <c>ValuesView.AsSpan</c> offers).</summary>
    /// <returns>A read-only span over the live prefix of the values array.</returns>
    public ReadOnlySpan<TValue> ValuesSpan() { Tables t = _t; return new ReadOnlySpan<TValue>(t.values, 0, Volatile.Read(ref t.count)); }

    /// <summary>Appends if absent: one chain walk answers the duplicate question, then the slot is written and release-published through the chain head and the count — zero allocation in capacity.</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value.</param>
    /// <returns>Whether the entry was added.</returns>
    public bool TryAdd(TKey key, TValue value) { lock (_lock) return TryAddUnderLock(key, value); }

    bool TryAddUnderLock(TKey key, TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.hashes.Length; i = t.next[i])
            if (t.hashes[i] == h && EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return false;
        int idx = t.count;
        Tables target = t; bool fresh = false;
        if (idx == t.keys.Length || idx < t.floor) { target = Copy(t, idx == t.keys.Length ? t.keys.Length * 2 : t.keys.Length); fresh = true; }
        uint b = Bucket(target, h);
        target.hashes[idx] = h; target.keys[idx] = key; target.values[idx] = value; target.next[idx] = target.buckets[b] - 1;
        if (fresh) { target.buckets[b] = idx + 1; target.count = idx + 1; _t = target; }
        else { Volatile.Write(ref target.buckets[b], idx + 1); Volatile.Write(ref target.count, idx + 1); }
        return true;
    }

    // Copy-on-write into fresh arrays. Same capacity: slot indices are preserved, so buckets/next copy verbatim
    // (every slot >= count is already unlinked). Larger capacity: rebuild the chains sequentially from the
    // STORED hashes — no rehash, no node allocation (contrast CD.GrowTable, which re-creates every node).
    static Tables Copy(Tables t, int newCap)
    {
        int n = t.count;
        var hashes = new int[newCap]; var next = new int[newCap]; var keys = new TKey[newCap]; var values = new TValue[newCap];
        Array.Copy(t.hashes, hashes, n); Array.Copy(t.keys, keys, n); Array.Copy(t.values, values, n);
        int[] buckets;
        if (newCap == t.keys.Length) { buckets = (int[])t.buckets.Clone(); Array.Copy(t.next, next, n); }
        else
        {
            buckets = new int[HashHelpers.GetPrime(newCap)];
            ulong fm = HashHelpers.GetFastModMultiplier((uint)buckets.Length);
            for (int j = 0; j < n; j++) { uint b = HashHelpers.FastMod((uint)hashes[j], (uint)buckets.Length, fm); next[j] = buckets[b] - 1; buckets[b] = j + 1; }
        }
        return new Tables(buckets, hashes, next, keys, values, n, 0);
    }

    // Reader-safe unlink: a single store into a bucket or a predecessor's link; the unlinked slot's own fields
    // stay intact for any reader standing on it (the same shape as the framework dictionary's node unlink).
    static void Unlink(Tables t, int i)
    {
        uint b = Bucket(t, t.hashes[i]);
        int j = t.buckets[b] - 1;
        if (j == i) { t.buckets[b] = t.next[i] + 1; return; }
        while (j >= 0) { int nx = t.next[j]; if (nx == i) { t.next[j] = t.next[i]; return; } j = nx; }
        throw new InvalidOperationException("slot not in its chain");
    }

    /// <summary>O(1) tail removal: unlink, then publish a lower-count holder over the same arrays with the floor raised (no copy).</summary>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveLast()
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count; if (n == 0) return false;
            Unlink(t, n - 1);
            _t = new Tables(t.buckets, t.hashes, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    /// <summary>Order-preserving removal: the tail path when last, otherwise compact-on-delete into fresh arrays with the sequential <see cref="Renumber.Map" /> pass over next/buckets (O(n), no hashing).</summary>
    /// <param name="i">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public void RemoveAt(int i)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(nameof(i));
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.hashes, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n)); return; }
            int cap = t.keys.Length;
            var hashes = new int[cap]; var keys = new TKey[cap]; var values = new TValue[cap]; var next = new int[cap];
            Array.Copy(t.hashes, hashes, i); Array.Copy(t.hashes, i + 1, hashes, i, n - i - 1);
            Array.Copy(t.keys, keys, i); Array.Copy(t.keys, i + 1, keys, i, n - i - 1);
            Array.Copy(t.values, values, i); Array.Copy(t.values, i + 1, values, i, n - i - 1);
            int skip = t.next[i];
            int[] oldNext = t.next;
            for (int j = 0; j < i; j++) next[j] = Renumber.Map(oldNext[j], i, skip);
            for (int j = i + 1; j < n; j++) next[j - 1] = Renumber.Map(oldNext[j], i, skip);
            int[] oldB = t.buckets; var buckets = new int[oldB.Length];
            for (int b = 0; b < buckets.Length; b++) buckets[b] = Renumber.Map(oldB[b] - 1, i, skip) + 1;
            _t = new Tables(buckets, hashes, next, keys, values, n - 1, 0);
        }
    }

    /// <summary>
    ///     Order-breaking O(1) removal, the 4-step in-place protocol: (1) unlink i from its chain; (2) overwrite
    ///     slot i with the last entry while no chain references it; (3) release-link i as the head of key_last's
    ///     chain (key_last now resolves at both slots); (4) unlink the old tail slot; then publish count-1.
    /// </summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>Whether an entry was removed.</returns>
    /// <remarks>Step 2 rewrites a slot a lock-free reader may already stand on — a reader of the REMOVED key can pair its old key with the moved value. The chained layouts cannot close that window; the open-addressed one can (see the findings).</remarks>
    public bool TryRemoveSwapBack(TKey key)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            int i = IndexOfUnderLock(t, key); if (i < 0) return false;
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.hashes, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n)); return true; }
            int last = n - 1;
            Unlink(t, i);
            int hl = t.hashes[last];
            t.hashes[i] = hl; t.keys[i] = t.keys[last]; t.values[i] = t.values[last];
            uint b = Bucket(t, hl);
            t.next[i] = t.buckets[b] - 1;
            Volatile.Write(ref t.buckets[b], i + 1);
            Unlink(t, last);
            _t = new Tables(t.buckets, t.hashes, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    static int IndexOfUnderLock(Tables t, TKey key)
    {
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.hashes.Length; i = t.next[i])
            if (t.hashes[i] == h && EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return i;
        return -1;
    }
}

/// <summary>The SoA layout without a stored-hash array: a trivially-hashed key (int/long/enum under the default comparer) is compared directly, so a hit touches three lines and the entry shrinks by 4 B.</summary>
/// <typeparam name="TKey">The key type (default comparer; hashing must be cheap to recompute on growth).</typeparam>
/// <typeparam name="TValue">The value type.</typeparam>
sealed class CompactSoANoHash<TKey, TValue> where TKey : notnull
{
    /// <summary>One generation: buckets, next, keys, values, the monotonic count and the append floor.</summary>
    sealed class Tables
    {
        public readonly int[] buckets; public readonly ulong fastMod;
        public readonly int[] next; public readonly TKey[] keys; public readonly TValue[] values;
        public int count; public readonly int floor;
        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="buckets">1-based chain heads.</param>
        /// <param name="next">Chain link per slot (-1 = end).</param>
        /// <param name="keys">Keys in insertion order.</param>
        /// <param name="values">Values in insertion order.</param>
        /// <param name="count">Live entries at publication.</param>
        /// <param name="floor">First slot an in-place append may write.</param>
        public Tables(int[] buckets, int[] next, TKey[] keys, TValue[] values, int count, int floor)
        { this.buckets = buckets; fastMod = HashHelpers.GetFastModMultiplier((uint)buckets.Length); this.next = next; this.keys = keys; this.values = values; this.count = count; this.floor = floor; }
    }

    volatile Tables _t;
    readonly object _lock = new();

    /// <summary>Creates a table presized for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">Entry capacity; at least 4.</param>
    public CompactSoANoHash(int capacity)
    {
        int cap = Math.Max(capacity, 4);
        _t = new Tables(new int[HashHelpers.GetPrime(cap)], new int[cap], new TKey[cap], new TValue[cap], 0, 0);
    }

    /// <summary>The live entry count.</summary>
    public int Count => Volatile.Read(ref _t.count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static uint Bucket(Tables t, int h) => HashHelpers.FastMod((uint)h, (uint)t.buckets.Length, t.fastMod);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int Hash(TKey key) => EqualityComparer<TKey>.Default.GetHashCode(key);

    /// <summary>Lock-free lookup comparing keys directly along the chain (buckets, keys, values — three lines per hit).</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value on a hit.</param>
    /// <returns>Whether the key was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        Tables t = _t;
        TKey[] keys = t.keys;
        ref TKey keys0 = ref MemoryMarshal.GetArrayDataReference(keys);
        ref int next0 = ref MemoryMarshal.GetArrayDataReference(t.next);
        int i = Volatile.Read(ref t.buckets[Bucket(t, Hash(key))]) - 1;
        while ((uint)i < (uint)keys.Length)
        {
            if (EqualityComparer<TKey>.Default.Equals(Unsafe.Add(ref keys0, i), key))
            {
                value = Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(t.values), i);
                return true;
            }
            i = Volatile.Read(ref Unsafe.Add(ref next0, i));
        }
        value = default;
        return false;
    }

    /// <summary>The insertion-order position of <paramref name="key" /> (the slot number), or -1.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The position, or -1.</returns>
    public int IndexOf(TKey key)
    {
        Tables t = _t;
        for (int i = Volatile.Read(ref t.buckets[Bucket(t, Hash(key))]) - 1; (uint)i < (uint)t.keys.Length; i = Volatile.Read(ref t.next[i]))
            if (EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return i;
        return -1;
    }

    /// <summary>Positional read behind the count bound.</summary>
    /// <param name="index">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            if ((uint)index >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(nameof(index));
            return t.values[index];
        }
    }

    /// <summary>The live values as a span.</summary>
    /// <returns>A read-only span over the live prefix.</returns>
    public ReadOnlySpan<TValue> ValuesSpan() { Tables t = _t; return new ReadOnlySpan<TValue>(t.values, 0, Volatile.Read(ref t.count)); }

    /// <summary>Appends if absent (see <see cref="CompactSoA{TKey,TValue}.TryAdd" />); growth recomputes hashes from the keys.</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value.</param>
    /// <returns>Whether the entry was added.</returns>
    public bool TryAdd(TKey key, TValue value) { lock (_lock) return TryAddUnderLock(key, value); }

    bool TryAddUnderLock(TKey key, TValue value)
    {
        Tables t = _t;
        int h = Hash(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.keys.Length; i = t.next[i])
            if (EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return false;
        int idx = t.count;
        Tables target = t; bool fresh = false;
        if (idx == t.keys.Length || idx < t.floor) { target = Copy(t, idx == t.keys.Length ? t.keys.Length * 2 : t.keys.Length); fresh = true; }
        uint b = Bucket(target, h);
        target.keys[idx] = key; target.values[idx] = value; target.next[idx] = target.buckets[b] - 1;
        if (fresh) { target.buckets[b] = idx + 1; target.count = idx + 1; _t = target; }
        else { Volatile.Write(ref target.buckets[b], idx + 1); Volatile.Write(ref target.count, idx + 1); }
        return true;
    }

    static Tables Copy(Tables t, int newCap)
    {
        int n = t.count;
        var next = new int[newCap]; var keys = new TKey[newCap]; var values = new TValue[newCap];
        Array.Copy(t.keys, keys, n); Array.Copy(t.values, values, n);
        int[] buckets;
        if (newCap == t.keys.Length) { buckets = (int[])t.buckets.Clone(); Array.Copy(t.next, next, n); }
        else
        {
            buckets = new int[HashHelpers.GetPrime(newCap)];
            ulong fm = HashHelpers.GetFastModMultiplier((uint)buckets.Length);
            for (int j = 0; j < n; j++) { uint b = HashHelpers.FastMod((uint)Hash(keys[j]), (uint)buckets.Length, fm); next[j] = buckets[b] - 1; buckets[b] = j + 1; }
        }
        return new Tables(buckets, next, keys, values, n, 0);
    }

    static void Unlink(Tables t, int i)
    {
        uint b = Bucket(t, Hash(t.keys[i]));
        int j = t.buckets[b] - 1;
        if (j == i) { t.buckets[b] = t.next[i] + 1; return; }
        while (j >= 0) { int nx = t.next[j]; if (nx == i) { t.next[j] = t.next[i]; return; } j = nx; }
        throw new InvalidOperationException("slot not in its chain");
    }

    /// <summary>O(1) tail removal (unlink + lower-count holder + floor).</summary>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveLast()
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count; if (n == 0) return false;
            Unlink(t, n - 1);
            _t = new Tables(t.buckets, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    /// <summary>Order-preserving removal by compact-on-delete with the sequential renumber (see <see cref="CompactSoA{TKey,TValue}.RemoveAt" />).</summary>
    /// <param name="i">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public void RemoveAt(int i)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(nameof(i));
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n)); return; }
            int cap = t.keys.Length;
            var keys = new TKey[cap]; var values = new TValue[cap]; var next = new int[cap];
            Array.Copy(t.keys, keys, i); Array.Copy(t.keys, i + 1, keys, i, n - i - 1);
            Array.Copy(t.values, values, i); Array.Copy(t.values, i + 1, values, i, n - i - 1);
            int skip = t.next[i];
            int[] oldNext = t.next;
            for (int j = 0; j < i; j++) next[j] = Renumber.Map(oldNext[j], i, skip);
            for (int j = i + 1; j < n; j++) next[j - 1] = Renumber.Map(oldNext[j], i, skip);
            int[] oldB = t.buckets; var buckets = new int[oldB.Length];
            for (int b = 0; b < buckets.Length; b++) buckets[b] = Renumber.Map(oldB[b] - 1, i, skip) + 1;
            _t = new Tables(buckets, next, keys, values, n - 1, 0);
        }
    }

    /// <summary>The 4-step in-place swap-back (see <see cref="CompactSoA{TKey,TValue}.TryRemoveSwapBack" />, same key-path caveat).</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveSwapBack(TKey key)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            int i = IndexOfUnderLock(t, key); if (i < 0) return false;
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n)); return true; }
            int last = n - 1;
            Unlink(t, i);
            TKey kl = t.keys[last];
            t.keys[i] = kl; t.values[i] = t.values[last];
            uint b = Bucket(t, Hash(kl));
            t.next[i] = t.buckets[b] - 1;
            Volatile.Write(ref t.buckets[b], i + 1);
            Unlink(t, last);
            _t = new Tables(t.buckets, t.next, t.keys, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    static int IndexOfUnderLock(Tables t, TKey key)
    {
        for (int i = t.buckets[Bucket(t, Hash(key))] - 1; (uint)i < (uint)t.keys.Length; i = t.next[i])
            if (EqualityComparer<TKey>.Default.Equals(t.keys[i], key)) return i;
        return -1;
    }
}

/// <summary>
///     Open-addressed index table (CPython's <c>dk_indices</c> shape) with the KEY embedded in the 8-byte index
///     word, so a probe compares keys without touching the entries: a hit = index line + values line (two
///     lines, like a node walk) while keys/values stay dense and spannable. Word = (key &lt;&lt; 32) | (slot + 1);
///     0 = empty, slot field -1 = dummy (a deleted probe position — never reused in place, cleared by the next
///     rebuild — which is what makes the validated read ABA-free within a generation).
/// </summary>
/// <typeparam name="TValue">The value type; keys are <c>int</c> (the probe's key type — a reference key would embed its hash instead).</typeparam>
sealed class CompactOA<TValue>
{
    /// <summary>One generation: the index words, the dense keys/values, the monotonic count, the floor and the dummy count (writer-only).</summary>
    sealed class Tables
    {
        public readonly long[] index; public readonly int shift;
        public readonly int[] keys; public readonly TValue[] values;
        public int count; public readonly int floor; public int dummies;
        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="index">Power-of-two index words.</param>
        /// <param name="keys">Keys in insertion order.</param>
        /// <param name="values">Values in insertion order.</param>
        /// <param name="count">Live entries at publication.</param>
        /// <param name="floor">First slot an in-place append may write.</param>
        /// <param name="dummies">Dummied index positions carried by this generation.</param>
        public Tables(long[] index, int[] keys, TValue[] values, int count, int floor, int dummies)
        { this.index = index; shift = 64 - System.Numerics.BitOperations.Log2((uint)index.Length); this.keys = keys; this.values = values; this.count = count; this.floor = floor; this.dummies = dummies; }
    }

    volatile Tables _t;
    readonly object _lock = new();

    /// <summary>Creates a table presized for <paramref name="capacity" /> entries (index = next power of two above 1.5x, so load stays ≤ 2/3).</summary>
    /// <param name="capacity">Entry capacity; at least 4.</param>
    public CompactOA(int capacity)
    {
        int cap = Math.Max(capacity, 4);
        _t = new Tables(new long[IndexSizeFor(cap)], new int[cap], new TValue[cap], 0, 0, 0);
    }

    static int IndexSizeFor(int cap) => (int)System.Numerics.BitOperations.RoundUpToPowerOf2((uint)Math.Max(8, (long)cap * 3 / 2 + 1));

    /// <summary>The live entry count.</summary>
    public int Count => Volatile.Read(ref _t.count);

    // Fibonacci hashing: sequential and patterned int keys spread evenly over the power-of-two table.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static uint Home(Tables t, int key) => (uint)((ulong)(uint)key * 0x9E3779B97F4A7C15UL >> t.shift);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static long Pack(int key, int slotField) => ((long)key << 32) | (uint)slotField;

    /// <summary>
    ///     Lock-free VALIDATED lookup: probe the index words, and after reading the value re-read the word — if it
    ///     changed, the entry was dummied (removed) or re-pointed (swap-back) under us and the value may belong to
    ///     another key, so the probe restarts. This is what keeps the key path tear-free under in-place swap-back.
    /// </summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value on a hit.</param>
    /// <returns>Whether the key was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(int key, out TValue value)
    {
        Tables t = _t;
        long[] index = t.index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(t, key);
        while (true)
        {
            long e = Volatile.Read(ref index[p]);
            int s = (int)e;
            if (s == 0) { value = default; return false; }
            if (s > 0 && (int)(e >> 32) == key)
            {
                value = t.values[s - 1];
                if (Volatile.Read(ref index[p]) == e) return true;
                p = Home(t, key);
                continue;
            }
            p = (p + 1) & mask;
        }
    }

    /// <summary>The insertion-order position of <paramref name="key" /> (the slot field of its index word), or -1.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The position, or -1.</returns>
    public int IndexOf(int key)
    {
        Tables t = _t;
        int p = ProbeFor(t, key);
        return p < 0 ? -1 : (int)t.index[p] - 1;
    }

    // Returns the index position holding `key`, or -1.
    static int ProbeFor(Tables t, int key)
    {
        long[] index = t.index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(t, key);
        while (true)
        {
            long e = index[p];
            int s = (int)e;
            if (s == 0) return -1;
            if (s > 0 && (int)(e >> 32) == key) return (int)p;
            p = (p + 1) & mask;
        }
    }

    /// <summary>Positional read behind the count bound.</summary>
    /// <param name="index">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            if ((uint)index >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(nameof(index));
            return t.values[index];
        }
    }

    /// <summary>The live values as a span.</summary>
    /// <returns>A read-only span over the live prefix.</returns>
    public ReadOnlySpan<TValue> ValuesSpan() { Tables t = _t; return new ReadOnlySpan<TValue>(t.values, 0, Volatile.Read(ref t.count)); }

    /// <summary>Appends if absent: ONE probe pass finds the key or the first empty word; the slot is written, then the word and the count are release-published — zero allocation in capacity.</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value.</param>
    /// <returns>Whether the entry was added.</returns>
    public bool TryAdd(int key, TValue value) { lock (_lock) return TryAddUnderLock(key, value); }

    bool TryAddUnderLock(int key, TValue value)
    {
        Tables t = _t;
        {
            long[] ix = t.index; uint m = (uint)ix.Length - 1; uint q = Home(t, key);
            while (true)
            {
                long e = ix[q]; int sf = (int)e;
                if (sf == 0) break;                                  // first empty: the key is absent
                if (sf > 0 && (int)(e >> 32) == key) return false;   // duplicate
                q = (q + 1) & m;
            }
        }
        int idx = t.count;
        Tables target = t; bool fresh = false;
        // grow when the entries are full, when the floor blocks the slot, or when live+dummies would exceed 2/3 load
        if (idx == t.keys.Length || idx < t.floor || (long)(idx + 1 + t.dummies) * 3 > (long)t.index.Length * 2)
        {
            target = Copy(t, idx == t.keys.Length ? t.keys.Length * 2 : t.keys.Length);
            fresh = true;
        }
        long[] index = target.index;
        uint mask = (uint)index.Length - 1;
        uint p = Home(target, key);
        while ((int)index[p] != 0) p = (p + 1) & mask; // first EMPTY (dummies are not reused in place)
        target.keys[idx] = key; target.values[idx] = value;
        if (fresh) { index[p] = Pack(key, idx + 1); target.count = idx + 1; _t = target; }
        else { Volatile.Write(ref index[p], Pack(key, idx + 1)); Volatile.Write(ref target.count, idx + 1); }
        return true;
    }

    // Fresh arrays; the index is REBUILT from the dense keys (which also drops every dummy).
    static Tables Copy(Tables t, int newCap)
    {
        int n = t.count;
        var keys = new int[newCap]; var values = new TValue[newCap];
        Array.Copy(t.keys, keys, n); Array.Copy(t.values, values, n);
        var index = new long[IndexSizeFor(newCap)];
        var fresh = new Tables(index, keys, values, n, 0, 0);
        uint mask = (uint)index.Length - 1;
        for (int j = 0; j < n; j++)
        {
            uint p = Home(fresh, keys[j]);
            while ((int)index[p] != 0) p = (p + 1) & mask;
            index[p] = Pack(keys[j], j + 1);
        }
        return fresh;
    }

    /// <summary>O(1) tail removal: dummy the key's index word (readers probe past it) and publish a lower-count holder with the floor raised.</summary>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveLast()
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count; if (n == 0) return false;
            TryRemoveLastUnlocked(t);
            return true;
        }
    }

    /// <summary>Order-preserving removal: compact-on-delete of keys/values plus a sequential renumber of the slot fields (slot i → dummy, slots above i → minus one).</summary>
    /// <param name="i">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public void RemoveAt(int i)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(nameof(i));
            if (i == n - 1) { TryRemoveLastUnlocked(t); return; }
            int cap = t.keys.Length;
            var keys = new int[cap]; var values = new TValue[cap];
            Array.Copy(t.keys, keys, i); Array.Copy(t.keys, i + 1, keys, i, n - i - 1);
            Array.Copy(t.values, values, i); Array.Copy(t.values, i + 1, values, i, n - i - 1);
            long[] old = t.index; var index = new long[old.Length];
            int removedSlot = i + 1;
            for (int p = 0; p < old.Length; p++)
            {
                long e = old[p]; int s = (int)e;
                if (s == removedSlot) index[p] = Pack((int)(e >> 32), -1);
                else if (s > removedSlot) index[p] = e - 1;
                else index[p] = e;
            }
            _t = new Tables(index, keys, values, n - 1, 0, t.dummies + 1);
        }
    }

    void TryRemoveLastUnlocked(Tables t)
    {
        int n = t.count;
        int p = ProbeFor(t, t.keys[n - 1]);
        t.index[p] = Pack(t.keys[n - 1], -1);
        _t = new Tables(t.index, t.keys, t.values, n - 1, Math.Max(t.floor, n), t.dummies + 1);
    }

    /// <summary>Order-breaking O(1) removal: dummy key_i's word, overwrite slot i with the last entry (reachable only via the list path meanwhile), then ONE atomic index store re-points key_last from slot n-1 to slot i — a validated reader never pairs the wrong value.</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveSwapBack(int key)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            int p = ProbeFor(t, key); if (p < 0) return false;
            int i = (int)t.index[p] - 1;
            if (i == n - 1) { t.index[p] = Pack(key, -1); _t = new Tables(t.index, t.keys, t.values, n - 1, Math.Max(t.floor, n), t.dummies + 1); return true; }
            int last = n - 1;
            t.index[p] = Pack(key, -1);
            int kl = t.keys[last];
            t.keys[i] = kl; t.values[i] = t.values[last];
            int pl = ProbeFor(t, kl);
            Volatile.Write(ref t.index[pl], Pack(kl, i + 1));
            _t = new Tables(t.index, t.keys, t.values, n - 1, Math.Max(t.floor, n), t.dummies + 1);
            return true;
        }
    }
}

/// <summary>Interleaved <c>Entry[]</c> layout (.NET <see cref="Dictionary{TKey,TValue}" />'s): one line per probe, but no value span and every scan reads the whole entry.</summary>
/// <typeparam name="TKey">The key type (default comparer).</typeparam>
/// <typeparam name="TValue">The value type.</typeparam>
sealed class CompactAoS<TKey, TValue> where TKey : notnull
{
    struct Entry { public int hash; public int next; public TKey key; public TValue value; }

    /// <summary>One generation: buckets, entries, the monotonic count and the append floor.</summary>
    sealed class Tables
    {
        public readonly int[] buckets; public readonly ulong fastMod; public readonly Entry[] entries;
        public int count; public readonly int floor;
        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="buckets">1-based chain heads.</param>
        /// <param name="entries">Entries in insertion order.</param>
        /// <param name="count">Live entries at publication.</param>
        /// <param name="floor">First slot an in-place append may write.</param>
        public Tables(int[] buckets, Entry[] entries, int count, int floor)
        { this.buckets = buckets; fastMod = HashHelpers.GetFastModMultiplier((uint)buckets.Length); this.entries = entries; this.count = count; this.floor = floor; }
    }

    volatile Tables _t;
    readonly object _lock = new();

    /// <summary>Creates a table presized for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">Entry capacity; at least 4.</param>
    public CompactAoS(int capacity)
    {
        int cap = Math.Max(capacity, 4);
        _t = new Tables(new int[HashHelpers.GetPrime(cap)], new Entry[cap], 0, 0);
    }

    /// <summary>The live entry count.</summary>
    public int Count => Volatile.Read(ref _t.count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static uint Bucket(Tables t, int h) => HashHelpers.FastMod((uint)h, (uint)t.buckets.Length, t.fastMod);

    /// <summary>Lock-free lookup — bucket line + entry line per hit.</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value on a hit.</param>
    /// <returns>Whether the key was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        Entry[] entries = t.entries;
        int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1;
        while ((uint)i < (uint)entries.Length)
        {
            ref Entry e = ref entries[i];
            if (e.hash == h && EqualityComparer<TKey>.Default.Equals(e.key, key)) { value = e.value; return true; }
            i = Volatile.Read(ref e.next);
        }
        value = default;
        return false;
    }

    /// <summary>The insertion-order position of <paramref name="key" /> (the slot number), or -1.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The position, or -1.</returns>
    public int IndexOf(TKey key)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1; (uint)i < (uint)t.entries.Length; i = Volatile.Read(ref t.entries[i].next))
            if (t.entries[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.entries[i].key, key)) return i;
        return -1;
    }

    /// <summary>Positional read behind the count bound (a strided load).</summary>
    /// <param name="index">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            if ((uint)index >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(nameof(index));
            return t.entries[index].value;
        }
    }

    /// <summary>Sums the live values by scanning the entries (the int-specialized scan the probe's foreach column measures).</summary>
    /// <returns>The sum.</returns>
    public long EnumSum()
    {
        Tables t = _t; Entry[] e = t.entries; int n = Volatile.Read(ref t.count); long s = 0;
        for (int i = 0; i < n; i++) s += Unsafe.As<TValue, int>(ref e[i].value);
        return s;
    }

    /// <summary>Appends if absent (see <see cref="CompactSoA{TKey,TValue}.TryAdd" />).</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value.</param>
    /// <returns>Whether the entry was added.</returns>
    public bool TryAdd(TKey key, TValue value) { lock (_lock) return TryAddUnderLock(key, value); }

    bool TryAddUnderLock(TKey key, TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.entries.Length; i = t.entries[i].next)
            if (t.entries[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.entries[i].key, key)) return false;
        int idx = t.count;
        Tables target = t; bool fresh = false;
        if (idx == t.entries.Length || idx < t.floor) { target = Copy(t, idx == t.entries.Length ? t.entries.Length * 2 : t.entries.Length); fresh = true; }
        uint b = Bucket(target, h);
        ref Entry e = ref target.entries[idx];
        e.hash = h; e.key = key; e.value = value; e.next = target.buckets[b] - 1;
        if (fresh) { target.buckets[b] = idx + 1; target.count = idx + 1; _t = target; }
        else { Volatile.Write(ref target.buckets[b], idx + 1); Volatile.Write(ref target.count, idx + 1); }
        return true;
    }

    static Tables Copy(Tables t, int newCap)
    {
        int n = t.count;
        var entries = new Entry[newCap];
        Array.Copy(t.entries, entries, n);
        int[] buckets;
        if (newCap == t.entries.Length) buckets = (int[])t.buckets.Clone();
        else
        {
            buckets = new int[HashHelpers.GetPrime(newCap)];
            ulong fm = HashHelpers.GetFastModMultiplier((uint)buckets.Length);
            for (int j = 0; j < n; j++) { uint b = HashHelpers.FastMod((uint)entries[j].hash, (uint)buckets.Length, fm); entries[j].next = buckets[b] - 1; buckets[b] = j + 1; }
        }
        return new Tables(buckets, entries, n, 0);
    }

    static void Unlink(Tables t, int i)
    {
        uint b = Bucket(t, t.entries[i].hash);
        int j = t.buckets[b] - 1;
        if (j == i) { t.buckets[b] = t.entries[i].next + 1; return; }
        while (j >= 0) { int nx = t.entries[j].next; if (nx == i) { t.entries[j].next = t.entries[i].next; return; } j = nx; }
        throw new InvalidOperationException("slot not in its chain");
    }

    /// <summary>O(1) tail removal (unlink + lower-count holder + floor).</summary>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveLast()
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count; if (n == 0) return false;
            Unlink(t, n - 1);
            _t = new Tables(t.buckets, t.entries, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    /// <summary>Order-preserving removal by compact-on-delete with the sequential renumber over entries' links and buckets.</summary>
    /// <param name="i">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public void RemoveAt(int i)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(nameof(i));
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.entries, n - 1, Math.Max(t.floor, n)); return; }
            int cap = t.entries.Length;
            var entries = new Entry[cap];
            Array.Copy(t.entries, entries, i); Array.Copy(t.entries, i + 1, entries, i, n - i - 1);
            int skip = t.entries[i].next;
            for (int j = 0; j < n - 1; j++) entries[j].next = Renumber.Map(entries[j].next, i, skip);
            int[] oldB = t.buckets; var buckets = new int[oldB.Length];
            for (int b = 0; b < buckets.Length; b++) buckets[b] = Renumber.Map(oldB[b] - 1, i, skip) + 1;
            _t = new Tables(buckets, entries, n - 1, 0);
        }
    }

    /// <summary>The 4-step in-place swap-back (see <see cref="CompactSoA{TKey,TValue}.TryRemoveSwapBack" />, same key-path caveat).</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveSwapBack(TKey key)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            int i = IndexOfUnderLock(t, key); if (i < 0) return false;
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.entries, n - 1, Math.Max(t.floor, n)); return true; }
            int last = n - 1;
            Unlink(t, i);
            ref Entry el = ref t.entries[last];
            ref Entry ei = ref t.entries[i];
            ei.hash = el.hash; ei.key = el.key; ei.value = el.value;
            uint b = Bucket(t, el.hash);
            ei.next = t.buckets[b] - 1;
            Volatile.Write(ref t.buckets[b], i + 1);
            Unlink(t, last);
            _t = new Tables(t.buckets, t.entries, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    static int IndexOfUnderLock(Tables t, TKey key)
    {
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.entries.Length; i = t.entries[i].next)
            if (t.entries[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.entries[i].key, key)) return i;
        return -1;
    }
}

/// <summary>Hybrid: the hash index <c>{hash,next,key}</c> interleaved (one line per probe), the VALUES contiguous and spannable; keys positional but not spannable.</summary>
/// <typeparam name="TKey">The key type (default comparer).</typeparam>
/// <typeparam name="TValue">The value type.</typeparam>
sealed class CompactSplit<TKey, TValue> where TKey : notnull
{
    struct IndexEntry { public int hash; public int next; public TKey key; }

    /// <summary>One generation: buckets, index entries, values, the monotonic count and the append floor.</summary>
    sealed class Tables
    {
        public readonly int[] buckets; public readonly ulong fastMod; public readonly IndexEntry[] index; public readonly TValue[] values;
        public int count; public readonly int floor;
        /// <summary>Wraps the arrays of one generation.</summary>
        /// <param name="buckets">1-based chain heads.</param>
        /// <param name="index">Index entries in insertion order.</param>
        /// <param name="values">Values in insertion order.</param>
        /// <param name="count">Live entries at publication.</param>
        /// <param name="floor">First slot an in-place append may write.</param>
        public Tables(int[] buckets, IndexEntry[] index, TValue[] values, int count, int floor)
        { this.buckets = buckets; fastMod = HashHelpers.GetFastModMultiplier((uint)buckets.Length); this.index = index; this.values = values; this.count = count; this.floor = floor; }
    }

    volatile Tables _t;
    readonly object _lock = new();

    /// <summary>Creates a table presized for <paramref name="capacity" /> entries.</summary>
    /// <param name="capacity">Entry capacity; at least 4.</param>
    public CompactSplit(int capacity)
    {
        int cap = Math.Max(capacity, 4);
        _t = new Tables(new int[HashHelpers.GetPrime(cap)], new IndexEntry[cap], new TValue[cap], 0, 0);
    }

    /// <summary>The live entry count.</summary>
    public int Count => Volatile.Read(ref _t.count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static uint Bucket(Tables t, int h) => HashHelpers.FastMod((uint)h, (uint)t.buckets.Length, t.fastMod);

    /// <summary>Lock-free lookup — bucket line + index line + values line per hit.</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value on a hit.</param>
    /// <returns>Whether the key was found.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(TKey key, out TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        IndexEntry[] index = t.index;
        int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1;
        while ((uint)i < (uint)index.Length)
        {
            ref IndexEntry e = ref index[i];
            if (e.hash == h && EqualityComparer<TKey>.Default.Equals(e.key, key)) { value = Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(t.values), i); return true; }
            i = Volatile.Read(ref e.next);
        }
        value = default;
        return false;
    }

    /// <summary>The insertion-order position of <paramref name="key" /> (the slot number), or -1.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The position, or -1.</returns>
    public int IndexOf(TKey key)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = Volatile.Read(ref t.buckets[Bucket(t, h)]) - 1; (uint)i < (uint)t.index.Length; i = Volatile.Read(ref t.index[i].next))
            if (t.index[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.index[i].key, key)) return i;
        return -1;
    }

    /// <summary>Positional read behind the count bound.</summary>
    /// <param name="index">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public TValue this[int index]
    {
        get
        {
            Tables t = _t;
            if ((uint)index >= (uint)Volatile.Read(ref t.count)) throw new ArgumentOutOfRangeException(nameof(index));
            return t.values[index];
        }
    }

    /// <summary>The live values as a span.</summary>
    /// <returns>A read-only span over the live prefix.</returns>
    public ReadOnlySpan<TValue> ValuesSpan() { Tables t = _t; return new ReadOnlySpan<TValue>(t.values, 0, Volatile.Read(ref t.count)); }

    /// <summary>Appends if absent (see <see cref="CompactSoA{TKey,TValue}.TryAdd" />).</summary>
    /// <param name="key">The key.</param>
    /// <param name="value">The value.</param>
    /// <returns>Whether the entry was added.</returns>
    public bool TryAdd(TKey key, TValue value) { lock (_lock) return TryAddUnderLock(key, value); }

    bool TryAddUnderLock(TKey key, TValue value)
    {
        Tables t = _t;
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.index.Length; i = t.index[i].next)
            if (t.index[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.index[i].key, key)) return false;
        int idx = t.count;
        Tables target = t; bool fresh = false;
        if (idx == t.index.Length || idx < t.floor) { target = Copy(t, idx == t.index.Length ? t.index.Length * 2 : t.index.Length); fresh = true; }
        uint b = Bucket(target, h);
        ref IndexEntry e = ref target.index[idx];
        e.hash = h; e.key = key; e.next = target.buckets[b] - 1; target.values[idx] = value;
        if (fresh) { target.buckets[b] = idx + 1; target.count = idx + 1; _t = target; }
        else { Volatile.Write(ref target.buckets[b], idx + 1); Volatile.Write(ref target.count, idx + 1); }
        return true;
    }

    static Tables Copy(Tables t, int newCap)
    {
        int n = t.count;
        var index = new IndexEntry[newCap]; var values = new TValue[newCap];
        Array.Copy(t.index, index, n); Array.Copy(t.values, values, n);
        int[] buckets;
        if (newCap == t.index.Length) buckets = (int[])t.buckets.Clone();
        else
        {
            buckets = new int[HashHelpers.GetPrime(newCap)];
            ulong fm = HashHelpers.GetFastModMultiplier((uint)buckets.Length);
            for (int j = 0; j < n; j++) { uint b = HashHelpers.FastMod((uint)index[j].hash, (uint)buckets.Length, fm); index[j].next = buckets[b] - 1; buckets[b] = j + 1; }
        }
        return new Tables(buckets, index, values, n, 0);
    }

    static void Unlink(Tables t, int i)
    {
        uint b = Bucket(t, t.index[i].hash);
        int j = t.buckets[b] - 1;
        if (j == i) { t.buckets[b] = t.index[i].next + 1; return; }
        while (j >= 0) { int nx = t.index[j].next; if (nx == i) { t.index[j].next = t.index[i].next; return; } j = nx; }
        throw new InvalidOperationException("slot not in its chain");
    }

    /// <summary>O(1) tail removal (unlink + lower-count holder + floor).</summary>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveLast()
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count; if (n == 0) return false;
            Unlink(t, n - 1);
            _t = new Tables(t.buckets, t.index, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    /// <summary>Order-preserving removal by compact-on-delete with the sequential renumber.</summary>
    /// <param name="i">The insertion-order position.</param>
    /// <exception cref="ArgumentOutOfRangeException">Out of range.</exception>
    public void RemoveAt(int i)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            if ((uint)i >= (uint)n) throw new ArgumentOutOfRangeException(nameof(i));
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.index, t.values, n - 1, Math.Max(t.floor, n)); return; }
            int cap = t.index.Length;
            var index = new IndexEntry[cap]; var values = new TValue[cap];
            Array.Copy(t.index, index, i); Array.Copy(t.index, i + 1, index, i, n - i - 1);
            Array.Copy(t.values, values, i); Array.Copy(t.values, i + 1, values, i, n - i - 1);
            int skip = t.index[i].next;
            for (int j = 0; j < n - 1; j++) index[j].next = Renumber.Map(index[j].next, i, skip);
            int[] oldB = t.buckets; var buckets = new int[oldB.Length];
            for (int b = 0; b < buckets.Length; b++) buckets[b] = Renumber.Map(oldB[b] - 1, i, skip) + 1;
            _t = new Tables(buckets, index, values, n - 1, 0);
        }
    }

    /// <summary>The 4-step in-place swap-back (see <see cref="CompactSoA{TKey,TValue}.TryRemoveSwapBack" />, same key-path caveat).</summary>
    /// <param name="key">The key to remove.</param>
    /// <returns>Whether an entry was removed.</returns>
    public bool TryRemoveSwapBack(TKey key)
    {
        lock (_lock)
        {
            Tables t = _t; int n = t.count;
            int i = IndexOfUnderLock(t, key); if (i < 0) return false;
            if (i == n - 1) { Unlink(t, i); _t = new Tables(t.buckets, t.index, t.values, n - 1, Math.Max(t.floor, n)); return true; }
            int last = n - 1;
            Unlink(t, i);
            ref IndexEntry el = ref t.index[last];
            ref IndexEntry ei = ref t.index[i];
            ei.hash = el.hash; ei.key = el.key; t.values[i] = t.values[last];
            uint b = Bucket(t, el.hash);
            ei.next = t.buckets[b] - 1;
            Volatile.Write(ref t.buckets[b], i + 1);
            Unlink(t, last);
            _t = new Tables(t.buckets, t.index, t.values, n - 1, Math.Max(t.floor, n));
            return true;
        }
    }

    static int IndexOfUnderLock(Tables t, TKey key)
    {
        int h = EqualityComparer<TKey>.Default.GetHashCode(key);
        for (int i = t.buckets[Bucket(t, h)] - 1; (uint)i < (uint)t.index.Length; i = t.index[i].next)
            if (t.index[i].hash == h && EqualityComparer<TKey>.Default.Equals(t.index[i].key, key)) return i;
        return -1;
    }
}
