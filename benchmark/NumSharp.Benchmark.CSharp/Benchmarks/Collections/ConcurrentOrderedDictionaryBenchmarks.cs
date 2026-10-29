using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using NumSharp.Collections;
using SysCd = System.Collections.Concurrent.ConcurrentDictionary<int, int>;
using CloneCd = NumSharp.Collections.Concurrent.ConcurrentDictionary<int, int>;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Collections;

/// <summary>
///     Fair micro-benchmarks for <see cref="ConcurrentOrderedDictionary{TKey,TValue}" /> against its two speed targets
///     (see <c>src/NumSharp.Core/Collections/Concurrent/ConcurrentOrderedDictionary.TODO.md</c>): <see cref="List{T}" />
///     for the list-path reads (enumerate, ToArray, index get) and the framework
///     <see cref="System.Collections.Concurrent.ConcurrentDictionary{TKey,TValue}" /> for the key-path operations
///     (get, add, replace, remove). The vendored clone the ordered dict is built on is measured alongside, so a
///     gap can be attributed to the ordered-dict layer vs. clone-vs-BCL drift.
/// </summary>
/// <remarks>
///     <para>
///         <b>Fairness rules encoded here.</b> (1) Every group carries a <c>Baseline = true</c> method so
///         BenchmarkDotNet emits the ratio column per logical group. (2) Build benchmarks construct the container
///         <b>inside</b> the measured body on every side — allocation and growth are part of the workload being
///         compared, identically for all contenders. (3) Read benchmarks run over prebuilt containers filled with
///         identical data, and key gets use one shared shuffled key order (fixed seed) so all sides pay the same
///         cache-miss pattern. (4) Remove benchmarks are build+remove composites (a removal benchmark cannot run
///         repeatedly over a prebuilt container without an unmeasurable per-iteration reset), again identical on
///         all sides.
///     </para>
///     <para>
///         This suite has no NumPy twin and is therefore <b>experimental</b> (not part of the official
///         run_benchmark.py matrix), like the Allocation suites. Run it directly:
///         <c>dotnet run -c Release -- --filter "*ConcurrentOrderedDictionary*"</c>.
///     </para>
/// </remarks>
[BenchmarkCategory("Collections", "ConcurrentOrderedDictionary")]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
public class ConcurrentOrderedDictionaryBenchmarks
{
    /// <summary>Element count per case: the house 1K / 100K tiers plus 1M (10M builds would dominate wall time for no extra signal on a hash-map workload).</summary>
    [Params(1_000, 100_000, 1_000_000)]
    public int N { get; set; }

    /// <summary>Prebuilt list-path baseline holding the same values as <see cref="_cod" />, in the same order.</summary>
    private List<int> _list = null!;

    /// <summary>Prebuilt framework dictionary — the key-path baseline.</summary>
    private SysCd _sysCd = null!;

    /// <summary>Prebuilt vendored-clone dictionary — isolates clone-vs-BCL drift from ordered-dict overhead.</summary>
    private CloneCd _cloneCd = null!;

    /// <summary>Prebuilt ordered dict under test.</summary>
    private ConcurrentOrderedDictionary<int, int> _cod = null!;

    /// <summary>Prebuilt compact ordered dict (open-addressed index over dense key/value arrays, no per-entry node) — the memory-lean sibling measured on every row.</summary>
    private ConcurrentOrderedCompactDictionary<int, int> _cocd = null!;

    /// <summary>The keys 0..N-1 in one fixed shuffled order, so every key-get contender pays the same cache-miss pattern.</summary>
    private int[] _shuffledKeys = null!;

    /// <summary>Prebuilt pair array for the AddRange path (materialized so the batch API's known-count presize engages, its intended use).</summary>
    private KeyValuePair<int, int>[] _pairs = null!;

    /// <summary>Builds the shared fixtures once per (N) case.</summary>
    [GlobalSetup]
    public void Setup()
    {
        // Fixed seed: identical shuffle for every contender and every run. (System-qualified: this project has
        // its own Benchmarks.Random namespace that would otherwise shadow the BCL type.)
        var rng = new System.Random(42);
        _shuffledKeys = new int[N];
        for (int i = 0; i < N; i++)
            _shuffledKeys[i] = i;
        for (int i = N - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (_shuffledKeys[i], _shuffledKeys[j]) = (_shuffledKeys[j], _shuffledKeys[i]);
        }

        _pairs = new KeyValuePair<int, int>[N];
        for (int i = 0; i < N; i++)
            _pairs[i] = new KeyValuePair<int, int>(i, i * 2);

        _list = new List<int>(N);
        _sysCd = new SysCd(Environment.ProcessorCount, N);
        _cloneCd = new CloneCd(1, N, null);
        _cod = new ConcurrentOrderedDictionary<int, int>(N);
        _cocd = new ConcurrentOrderedCompactDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
        {
            _list.Add(i * 2);
            _sysCd[i] = i * 2;
            _cloneCd[i] = i * 2;
            _cod.Add(i, i * 2);
            _cocd.Add(i, i * 2);
        }
    }

    /// <summary>Releases the fixtures so the next (N) case starts from a clean heap.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _list = null!;
        _sysCd = null!;
        _cloneCd = null!;
        _cod = null!;
        _cocd = null!;
        _shuffledKeys = null!;
        _pairs = null!;
        GC.Collect();
    }

    // ------------------------------------------------------------------ Build (baseline: framework dictionary)

    /// <summary>Baseline: sequential TryAdd into an unsized framework dictionary.</summary>
    [Benchmark(Baseline = true, Description = "SysCD.TryAdd (unsized)")]
    [BenchmarkCategory("Build")]
    public int SysCd_TryAdd()
    {
        var d = new SysCd();
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Presized framework dictionary — the fair peer for the presized ordered dict.</summary>
    [Benchmark(Description = "SysCD.TryAdd (presized)")]
    [BenchmarkCategory("Build")]
    public int SysCd_TryAdd_Presized()
    {
        var d = new SysCd(Environment.ProcessorCount, N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>The vendored clone, unsized — attributes any build gap to clone-vs-BCL drift vs. ordered-dict overhead.</summary>
    [Benchmark(Description = "CloneCD.TryAdd (unsized)")]
    [BenchmarkCategory("Build")]
    public int CloneCd_TryAdd()
    {
        var d = new CloneCd();
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Context row: what a plain (non-thread-safe, unkeyed) append costs.</summary>
    [Benchmark(Description = "List.Add (unsized)")]
    [BenchmarkCategory("Build")]
    public int List_Add()
    {
        var l = new List<int>();
        for (int i = 0; i < N; i++)
            l.Add(i);
        return l.Count;
    }

    /// <summary>Sequential TryAdd into an unsized ordered dict — the gap-B2 headline cell.</summary>
    [Benchmark(Description = "COD.TryAdd (unsized)")]
    [BenchmarkCategory("Build")]
    public int Cod_TryAdd()
    {
        var d = new ConcurrentOrderedDictionary<int, int>();
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Presized ordered dict — no array growth, the recommended build pattern.</summary>
    [Benchmark(Description = "COD.TryAdd (presized)")]
    [BenchmarkCategory("Build")]
    public int Cod_TryAdd_Presized()
    {
        var d = new ConcurrentOrderedDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Batch upsert under one lock — the bulk-load API (amortizes lock, growth and publication).</summary>
    [Benchmark(Description = "COD.AddRange (batch)")]
    [BenchmarkCategory("Build")]
    public int Cod_AddRange()
    {
        var d = new ConcurrentOrderedDictionary<int, int>();
        d.AddRange(_pairs);
        return d.Count;
    }

    /// <summary>The pairs constructor — the optimal build path: pre-sizes BOTH the hash map and the arrays from the countable source, so the build allocates exactly the live bytes (no growth churn on either representation).</summary>
    [Benchmark(Description = "COD ctor(pairs) (presizes both)")]
    [BenchmarkCategory("Build")]
    public int Cod_CtorFromPairs() => new ConcurrentOrderedDictionary<int, int>(_pairs).Count;

    /// <summary>Sequential TryAdd into an unsized compact dict — no per-add node; growth copies arrays and rebuilds the index from its own words.</summary>
    [Benchmark(Description = "COCD.TryAdd (unsized)")]
    [BenchmarkCategory("Build")]
    public int Cocd_TryAdd()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>();
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Presized compact dict — an in-capacity append allocates nothing at all.</summary>
    [Benchmark(Description = "COCD.TryAdd (presized)")]
    [BenchmarkCategory("Build")]
    public int Cocd_TryAdd_Presized()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.Count;
    }

    /// <summary>Batch upsert into the compact dict under one lock.</summary>
    [Benchmark(Description = "COCD.AddRange (batch)")]
    [BenchmarkCategory("Build")]
    public int Cocd_AddRange()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>();
        d.AddRange(_pairs);
        return d.Count;
    }

    /// <summary>The compact dict's pairs constructor — pre-sizes the arrays and the index from the countable source.</summary>
    [Benchmark(Description = "COCD ctor(pairs) (presizes both)")]
    [BenchmarkCategory("Build")]
    public int Cocd_CtorFromPairs() => new ConcurrentOrderedCompactDictionary<int, int>(_pairs).Count;

    // ------------------------------------------------------------------ Key get (baseline: framework dictionary)

    /// <summary>Baseline: lock-free TryGetValue over the shuffled key order.</summary>
    [Benchmark(Baseline = true, Description = "SysCD.TryGetValue")]
    [BenchmarkCategory("KeyGet")]
    public long SysCd_TryGetValue()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
        {
            _sysCd.TryGetValue(keys[i], out int v);
            s += v;
        }

        return s;
    }

    /// <summary>The vendored clone's TryGetValue — the clone-vs-BCL attribution row.</summary>
    [Benchmark(Description = "CloneCD.TryGetValue")]
    [BenchmarkCategory("KeyGet")]
    public long CloneCd_TryGetValue()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
        {
            _cloneCd.TryGetValue(keys[i], out int v);
            s += v;
        }

        return s;
    }

    /// <summary>Ordered-dict key get — the gap-B1 headline cell (value stored inline in the hash node).</summary>
    [Benchmark(Description = "COD.TryGetValue")]
    [BenchmarkCategory("KeyGet")]
    public long Cod_TryGetValue()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
        {
            _cod.TryGetValue(keys[i], out int v);
            s += v;
        }

        return s;
    }

    /// <summary>Key→index mapping — same walk as the key get, reading the inline index field instead.</summary>
    [Benchmark(Description = "COD.IndexOf")]
    [BenchmarkCategory("KeyGet")]
    public long Cod_IndexOf()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
            s += _cod.IndexOf(keys[i]);
        return s;
    }

    /// <summary>Compact-dict key get — the validated open-addressed probe (index line + values line for int keys).</summary>
    [Benchmark(Description = "COCD.TryGetValue")]
    [BenchmarkCategory("KeyGet")]
    public long Cocd_TryGetValue()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
        {
            _cocd.TryGetValue(keys[i], out int v);
            s += v;
        }

        return s;
    }

    /// <summary>Compact-dict key→index — the slot number carried by the index word, no stored position field.</summary>
    [Benchmark(Description = "COCD.IndexOf")]
    [BenchmarkCategory("KeyGet")]
    public long Cocd_IndexOf()
    {
        long s = 0;
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
            s += _cocd.IndexOf(keys[i]);
        return s;
    }

    // ------------------------------------------------------------------ Enumerate (baseline: List)

    /// <summary>Baseline: foreach over a list.</summary>
    [Benchmark(Baseline = true, Description = "List foreach")]
    [BenchmarkCategory("Enumerate")]
    public long List_Foreach()
    {
        long s = 0;
        foreach (int v in _list)
            s += v;
        return s;
    }

    /// <summary>Ordered-dict struct enumerator over the contiguous value snapshot.</summary>
    [Benchmark(Description = "COD foreach")]
    [BenchmarkCategory("Enumerate")]
    public long Cod_Foreach()
    {
        long s = 0;
        foreach (int v in _cod)
            s += v;
        return s;
    }

    /// <summary>Snapshot-view foreach — a span enumerator over the captured array (the hot-loop form).</summary>
    [Benchmark(Description = "COD snapshot foreach (span)")]
    [BenchmarkCategory("Enumerate")]
    public long Cod_SnapshotForeach()
    {
        long s = 0;
        foreach (int v in _cod.Snapshot())
            s += v;
        return s;
    }

    /// <summary>Compact-dict struct enumerator over the dense values array.</summary>
    [Benchmark(Description = "COCD foreach")]
    [BenchmarkCategory("Enumerate")]
    public long Cocd_Foreach()
    {
        long s = 0;
        foreach (int v in _cocd)
            s += v;
        return s;
    }

    /// <summary>Compact-dict snapshot-view foreach (span enumerator).</summary>
    [Benchmark(Description = "COCD snapshot foreach (span)")]
    [BenchmarkCategory("Enumerate")]
    public long Cocd_SnapshotForeach()
    {
        long s = 0;
        foreach (int v in _cocd.Snapshot())
            s += v;
        return s;
    }

    /// <summary>Context row: the framework dictionary's enumerator (unordered, node-walking — structurally slower by design, shown for scale).</summary>
    [Benchmark(Description = "SysCD foreach (unordered)")]
    [BenchmarkCategory("Enumerate")]
    public long SysCd_Foreach()
    {
        long s = 0;
        foreach (KeyValuePair<int, int> kv in _sysCd)
            s += kv.Value;
        return s;
    }

    // ------------------------------------------------------------------ ToArray (baseline: List)

    /// <summary>Baseline: list snapshot copy.</summary>
    [Benchmark(Baseline = true, Description = "List.ToArray")]
    [BenchmarkCategory("ToArray")]
    public int List_ToArray() => _list.ToArray().Length;

    /// <summary>Ordered-dict snapshot copy — a single contiguous block copy of the live prefix.</summary>
    [Benchmark(Description = "COD.ToArray")]
    [BenchmarkCategory("ToArray")]
    public int Cod_ToArray() => _cod.ToArray().Length;

    /// <summary>Compact-dict snapshot copy — the same contiguous block copy of the live prefix.</summary>
    [Benchmark(Description = "COCD.ToArray")]
    [BenchmarkCategory("ToArray")]
    public int Cocd_ToArray() => _cocd.ToArray().Length;

    // ------------------------------------------------------------------ Index get (baseline: List)

    /// <summary>Baseline: list indexer loop.</summary>
    [Benchmark(Baseline = true, Description = "List[i] loop")]
    [BenchmarkCategory("IndexGet")]
    public long List_IndexLoop()
    {
        long s = 0;
        for (int i = 0; i < N; i++)
            s += _list[i];
        return s;
    }

    /// <summary>Live indexer loop — pays one volatile snapshot read per call (the documented single-call cost).</summary>
    [Benchmark(Description = "COD[i] loop (live)")]
    [BenchmarkCategory("IndexGet")]
    public long Cod_IndexLoop()
    {
        long s = 0;
        for (int i = 0; i < N; i++)
            s += _cod[i];
        return s;
    }

    /// <summary>Snapshot-view indexer loop — captures the volatile state once, then plain array reads (gap A1's fix).</summary>
    [Benchmark(Description = "COD.Snapshot()[i] loop")]
    [BenchmarkCategory("IndexGet")]
    public long Cod_SnapshotIndexLoop()
    {
        long s = 0;
        ConcurrentOrderedDictionary<int, int>.ValuesView view = _cod.Snapshot();
        for (int i = 0; i < view.Count; i++)
            s += view[i];
        return s;
    }

    /// <summary>Span loop over the snapshot — the vectorizable form.</summary>
    [Benchmark(Description = "COD.Snapshot().AsSpan() loop")]
    [BenchmarkCategory("IndexGet")]
    public long Cod_SpanLoop()
    {
        long s = 0;
        ReadOnlySpan<int> span = _cod.Snapshot().AsSpan();
        for (int i = 0; i < span.Length; i++)
            s += span[i];
        return s;
    }

    /// <summary>Compact-dict live indexer loop (one volatile generation read per call).</summary>
    [Benchmark(Description = "COCD[i] loop (live)")]
    [BenchmarkCategory("IndexGet")]
    public long Cocd_IndexLoop()
    {
        long s = 0;
        for (int i = 0; i < N; i++)
            s += _cocd[i];
        return s;
    }

    /// <summary>Compact-dict snapshot-view indexer loop.</summary>
    [Benchmark(Description = "COCD.Snapshot()[i] loop")]
    [BenchmarkCategory("IndexGet")]
    public long Cocd_SnapshotIndexLoop()
    {
        long s = 0;
        ConcurrentOrderedCompactDictionary<int, int>.ValuesView view = _cocd.Snapshot();
        for (int i = 0; i < view.Count; i++)
            s += view[i];
        return s;
    }

    // ------------------------------------------------------------------ Replace existing (baseline: framework dictionary)

    /// <summary>Baseline: indexer upsert of existing keys (the framework dictionary's in-place atomic update).</summary>
    [Benchmark(Baseline = true, Description = "SysCD[k]=v (existing)")]
    [BenchmarkCategory("Replace")]
    public int SysCd_IndexerReplace()
    {
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
            _sysCd[keys[i]] = i;
        return keys.Length;
    }

    /// <summary>Ordered-dict upsert of existing keys — in-place atomic field stores on both paths, position preserved.</summary>
    [Benchmark(Description = "COD.SetByKey (existing)")]
    [BenchmarkCategory("Replace")]
    public int Cod_SetByKey()
    {
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
            _cod.SetByKey(keys[i], i);
        return keys.Length;
    }

    /// <summary>Compact-dict upsert of existing keys — one in-place store into the single value copy.</summary>
    [Benchmark(Description = "COCD.SetByKey (existing)")]
    [BenchmarkCategory("Replace")]
    public int Cocd_SetByKey()
    {
        int[] keys = _shuffledKeys;
        for (int i = 0; i < keys.Length; i++)
            _cocd.SetByKey(keys[i], i);
        return keys.Length;
    }

    // ------------------------------------------------------------------ Remove (baseline: framework dictionary; composites)

    /// <summary>Baseline composite: build N then remove every key (the framework dictionary's O(1)-per-remove reference).</summary>
    [Benchmark(Baseline = true, Description = "SysCD build+removeAll")]
    [BenchmarkCategory("Remove")]
    public int SysCd_BuildRemoveAll()
    {
        var d = new SysCd(Environment.ProcessorCount, N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        for (int i = N - 1; i >= 0; i--)
            d.TryRemove(i, out _);
        return d.Count;
    }

    /// <summary>Build N then pop every entry from the back — exercises the O(1) tail-removal fast path.</summary>
    [Benchmark(Description = "COD build+popBackAll")]
    [BenchmarkCategory("Remove")]
    public int Cod_BuildPopBackAll()
    {
        var d = new ConcurrentOrderedDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        for (int i = N - 1; i >= 0; i--)
            d.TryRemove(i, out _);
        return d.Count;
    }

    /// <summary>Build N then swap-back-remove every entry in FRONT order — the O(1) order-breaking removal for arbitrary positions.</summary>
    [Benchmark(Description = "COD build+swapBackAll (front order)")]
    [BenchmarkCategory("Remove")]
    public int Cod_BuildSwapBackAll()
    {
        var d = new ConcurrentOrderedDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        for (int i = 0; i < N; i++)
            d.TryRemoveSwapBack(i, out _);
        return d.Count;
    }

    /// <summary>Build N then remove everything in ONE compaction pass — the bulk-removal API.</summary>
    [Benchmark(Description = "COD build+RemoveWhere(all)")]
    [BenchmarkCategory("Remove")]
    public int Cod_BuildRemoveWhereAll()
    {
        var d = new ConcurrentOrderedDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.RemoveWhere(static (k, v) => true);
    }

    /// <summary>Compact dict: build N then pop every entry from the back (one dummied index word + one holder per pop).</summary>
    [Benchmark(Description = "COCD build+popBackAll")]
    [BenchmarkCategory("Remove")]
    public int Cocd_BuildPopBackAll()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        for (int i = N - 1; i >= 0; i--)
            d.TryRemove(i, out _);
        return d.Count;
    }

    /// <summary>Compact dict: build N then swap-back-remove every entry in FRONT order (in place: dummy, slot overwrite, one re-pointing word store).</summary>
    [Benchmark(Description = "COCD build+swapBackAll (front order)")]
    [BenchmarkCategory("Remove")]
    public int Cocd_BuildSwapBackAll()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        for (int i = 0; i < N; i++)
            d.TryRemoveSwapBack(i, out _);
        return d.Count;
    }

    /// <summary>Compact dict: build N then remove everything in ONE pass (fresh arrays + one sequential index renumber).</summary>
    [Benchmark(Description = "COCD build+RemoveWhere(all)")]
    [BenchmarkCategory("Remove")]
    public int Cocd_BuildRemoveWhereAll()
    {
        var d = new ConcurrentOrderedCompactDictionary<int, int>(N);
        for (int i = 0; i < N; i++)
            d.TryAdd(i, i);
        return d.RemoveWhere(static (k, v) => true);
    }
}
