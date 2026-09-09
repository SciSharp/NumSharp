/* ============================================================================================
 * RESULTS — measured 2026-09-09 (regenerate: dotnet run -c Release -f net10.0 -- --filter "*Iteration*")
 *
 * BenchmarkDotNet v0.15.8 · .NET 10.0.1, X64 RyuJIT x86-64-v3 (AVX2)
 * 13th Gen Intel Core i9-13900K · Windows 11 (10.0.26200)
 * Config: OfficialBenchmarkConfig — InProcessEmit · 50 iterations · 25 ms iteration cap · 5 warmups
 *         · OutlierMode.RemoveAll · MemoryDiagnoser ON.
 * Host stability: CPU clock LOCKED (turbo boost Disabled, restored after) + process pinned to one
 *         P-core (logical CPU 2). Timing basis = Min (best window). 208 cases measured; the 2 NA
 *         cells are np.nditer_chunks<T> × Strided (a Span<T> can't span a non-unit inner stride —
 *         it throws NotSupportedException, by design).
 *
 * Read "Allocated" as the headline: it is the whole reason MemoryDiagnoser is on. The typed/by-ref
 * walks allocate a FLAT 64 B for the entire walk at any N and any layout; the boxed/parity forms
 * allocate MEGABYTES (a boxed object or an NDArray view per element).
 *
 * ── MIN time — N = 1 (fixed per-call / iterator-setup overhead) ──────────────────────────────
 * | technique                          |  Contiguous |  Transposed |     Strided |    Reversed |   Broadcast |
 * | foreach (row) foreach (x)          |    608.7 ns |    619.2 ns |    617.1 ns |    619.1 ns |    607.0 ns |
 * | a.flat (boxed)                     |    625.3 ns |    504.7 ns |    508.3 ns |    539.8 ns |    506.7 ns |
 * | a.flatiter (boxed, any layout)     |    120.2 ns |     22.8 ns |     21.6 ns |     22.0 ns |     22.2 ns |
 * | np.flat<T> (ref, C-order)          |    215.7 ns |    181.7 ns |    181.5 ns |    180.5 ns |    181.0 ns |
 * | np.nditer<T> (ref, memory-order)   |    218.7 ns |    192.3 ns |    191.9 ns |    190.4 ns |    190.2 ns |
 * | np.nditer_chunks<T> (Span)         |    263.0 ns |    241.9 ns |    187.7 ns |    189.2 ns |    187.4 ns |
 * | np.ndenumerate<T> (by-value)       |    156.9 ns |     66.3 ns |     66.4 ns |     67.6 ns |     66.2 ns |
 * | np.ndenumerate<T>().AsRef()        |    309.1 ns |    237.5 ns |    238.1 ns |    239.3 ns |    242.6 ns |
 * | nd.AsEnumerable<T>()               |    112.6 ns |     18.7 ns |     18.5 ns |     18.8 ns |     18.4 ns |
 * | np.ndindex + GetValue (long[])     |     93.2 ns |     93.1 ns |     93.2 ns |     93.0 ns |     93.0 ns |
 * | np.ndindex().AsSpans() + GetValue  |     49.6 ns |     50.2 ns |     50.0 ns |     50.1 ns |     50.7 ns |
 * | np.nditer (boxed object)           |     1.55 us |     1.58 us |     1.53 us |     1.55 us |     1.56 us |
 * | np.nested_iters                    |     2.32 us |     2.30 us |     2.57 us |     2.56 us |     2.61 us |
 * | np.broadcast (2-operand)           |     1.47 us |     1.45 us |     1.47 us |     1.45 us |     1.44 us |
 *
 * ── MIN time — N = 1,000 ─────────────────────────────────────────────────────────────────────
 * | technique                          |  Contiguous |  Transposed |     Strided |    Reversed |   Broadcast |
 * | foreach (row) foreach (x)          |    36.82 us |    34.17 us |    33.83 us |    31.30 us |    25.90 us |
 * | a.flat (boxed)                     |     6.78 us |     9.13 us |     9.24 us |     8.88 us |     7.99 us |
 * | a.flatiter (boxed, any layout)     |     6.03 us |    20.37 us |    19.81 us |    19.87 us |    19.70 us |
 * | np.flat<T> (ref, C-order)          |     2.98 us |     2.96 us |     2.95 us |     2.96 us |     2.96 us |
 * | np.nditer<T> (ref, memory-order)   |     2.97 us |     2.97 us |     2.97 us |     2.98 us |     2.96 us |
 * | np.nditer_chunks<T> (Span)         |    892.3 ns |    890.6 ns |         n/a |    896.2 ns |    880.5 ns |
 * | np.ndenumerate<T> (by-value)       |    65.28 us |    50.19 us |    50.95 us |    50.39 us |    49.75 us |
 * | np.ndenumerate<T>().AsRef()        |     4.62 us |     4.70 us |     4.63 us |     4.68 us |     4.70 us |
 * | nd.AsEnumerable<T>()               |     1.74 us |    16.63 us |    16.45 us |    16.44 us |    16.50 us |
 * | np.ndindex + GetValue (long[])     |    69.94 us |    69.72 us |    69.23 us |    69.74 us |    68.59 us |
 * | np.ndindex().AsSpans() + GetValue  |     7.99 us |     8.00 us |     7.93 us |     7.97 us |     8.03 us |
 * | np.nditer (boxed object)           |   800.68 us |   799.05 us |   802.83 us |   804.31 us |     1.17 ms |
 * | np.nested_iters                    |   764.76 us |   532.68 us |   537.17 us |   799.20 us |   774.06 us |
 * | np.broadcast (2-operand)           |    93.18 us |   100.42 us |   100.79 us |   100.63 us |   100.54 us |
 *
 * ── MIN time — N = 100,000 (throughput) ──────────────────────────────────────────────────────
 * | technique                          |  Contiguous |  Transposed |     Strided |    Reversed |   Broadcast |
 * | foreach (row) foreach (x)          |   757.62 us |     1.04 ms |     1.03 ms |     1.08 ms |   736.44 us |
 * | a.flat (boxed)                     |     2.28 ms |     3.11 ms |     2.26 ms |     3.11 ms |     2.25 ms |
 * | a.flatiter (boxed, any layout)     |   561.16 us |     1.96 ms |     1.96 ms |     1.97 ms |     2.01 ms |
 * | np.flat<T> (ref, C-order)          |   278.67 us |   282.51 us |   278.89 us |   278.84 us |   279.57 us |
 * | np.nditer<T> (ref, memory-order)   |   278.59 us |   278.53 us |   278.79 us |   278.56 us |   277.71 us |
 * | np.nditer_chunks<T> (Span)         |    69.83 us |    69.80 us |         n/a |    69.76 us |    69.57 us |
 * | np.ndenumerate<T> (by-value)       |     7.13 ms |     5.04 ms |     5.07 ms |     4.99 ms |     4.97 ms |
 * | np.ndenumerate<T>().AsRef()        |   390.80 us |   391.59 us |   377.43 us |   378.41 us |   376.85 us |
 * | nd.AsEnumerable<T>()               |   191.38 us |     1.62 ms |     1.62 ms |     1.63 ms |     1.63 ms |
 * | np.ndindex + GetValue (long[])     |     7.68 ms |     7.57 ms |     7.70 ms |     7.52 ms |     7.49 ms |
 * | np.ndindex().AsSpans() + GetValue  |   699.40 us |   712.86 us |   702.44 us |   696.78 us |   707.66 us |
 * | np.nditer (boxed object)           |   114.59 ms |   114.14 ms |   114.18 ms |   113.13 ms |    39.63 ms |
 * | np.nested_iters                    |   125.12 ms |   109.78 ms |   110.36 ms |   110.70 ms |   108.67 ms |
 * | np.broadcast (2-operand)           |     3.28 ms |     4.45 ms |     4.52 ms |     4.63 ms |     4.64 ms |
 *
 * ── ALLOCATED per full walk — N = 1,000 ──────────────────────────────────────────────────────
 * | technique                          |  Contiguous |  Transposed |     Strided |    Reversed |   Broadcast |
 * | foreach (row) foreach (x)          |     47.9 KB |     48.3 KB |     48.3 KB |     48.3 KB |     47.5 KB |
 * | a.flat (boxed)                     |     31.8 KB |     32.2 KB |     32.2 KB |     32.2 KB |     32.2 KB |
 * | a.flatiter (boxed, any layout)     |     23.5 KB |     62.6 KB |     62.6 KB |     62.6 KB |     62.6 KB |
 * | np.flat<T> (ref, C-order)          |        65 B |        65 B |        64 B |        65 B |        64 B |
 * | np.nditer<T> (ref, memory-order)   |        65 B |        65 B |        64 B |        65 B |        64 B |
 * | np.nditer_chunks<T> (Span)         |        64 B |        64 B |         n/a |        64 B |        64 B |
 * | np.ndenumerate<T> (by-value)       |     39.2 KB |     78.2 KB |     78.2 KB |     78.2 KB |     78.2 KB |
 * | np.ndenumerate<T>().AsRef()        |       201 B |       200 B |       201 B |       200 B |       200 B |
 * | nd.AsEnumerable<T>()               |        64 B |     39.1 KB |     39.1 KB |     39.1 KB |     39.1 KB |
 * | np.ndindex + GetValue (long[])     |     39.2 KB |     39.2 KB |     39.2 KB |     39.2 KB |     39.2 KB |
 * | np.ndindex().AsSpans() + GetValue  |     39.1 KB |     39.1 KB |     39.1 KB |     39.1 KB |     39.1 KB |
 * | np.nditer (boxed object)           |    453.3 KB |    453.3 KB |    453.3 KB |    453.3 KB |    453.4 KB |
 * | np.nested_iters                    |    864.1 KB |    864.1 KB |    864.1 KB |    864.1 KB |    864.1 KB |
 * | np.broadcast (2-operand)           |    126.5 KB |    165.5 KB |    165.5 KB |    165.5 KB |    165.5 KB |
 *
 * ── ALLOCATED per full walk — N = 100,000 (the headline) ─────────────────────────────────────
 * | technique                          |  Contiguous |  Transposed |     Strided |    Reversed |   Broadcast |
 * | foreach (row) foreach (x)          |     3.12 MB |     3.12 MB |     3.12 MB |     3.12 MB |     3.12 MB |
 * | a.flat (boxed)                     |     3.05 MB |     3.05 MB |     3.05 MB |     3.05 MB |     3.05 MB |
 * | a.flatiter (boxed, any layout)     |     2.29 MB |     6.10 MB |     6.10 MB |     6.10 MB |     6.10 MB |
 * | np.flat<T> (ref, C-order)          |        64 B |        64 B |        64 B |        64 B |        64 B |
 * | np.nditer<T> (ref, memory-order)   |        64 B |        64 B |        64 B |        64 B |        64 B |
 * | np.nditer_chunks<T> (Span)         |        64 B |        64 B |         n/a |        64 B |        64 B |
 * | np.ndenumerate<T> (by-value)       |     3.82 MB |     7.63 MB |     7.63 MB |     7.63 MB |     7.63 MB |
 * | np.ndenumerate<T>().AsRef()        |       200 B |       200 B |       259 B |       200 B |       200 B |
 * | nd.AsEnumerable<T>()               |        64 B |     3.81 MB |     3.81 MB |     3.81 MB |     3.81 MB |
 * | np.ndindex + GetValue (long[])     |     3.81 MB |     3.82 MB |     3.82 MB |     3.82 MB |     3.82 MB |
 * | np.ndindex().AsSpans() + GetValue  |     3.81 MB |     3.81 MB |     3.81 MB |     3.81 MB |     3.81 MB |
 * | np.nditer (boxed object)           |    44.26 MB |    44.26 MB |    44.26 MB |    44.26 MB |    44.26 MB |
 * | np.nested_iters                    |    83.21 MB |    83.21 MB |    83.21 MB |    83.21 MB |    83.21 MB |
 * | np.broadcast (2-operand)           |    12.21 MB |    16.02 MB |    16.02 MB |    16.02 MB |    16.02 MB |
 *
 * ── Takeaways ────────────────────────────────────────────────────────────────────────────────
 * - Allocation-free hot-loop walks: np.flat<T> / np.nditer<T> / np.nditer_chunks<T> allocate a FLAT
 *   64 B for the entire walk regardless of N or layout; np.ndenumerate<T>().AsRef() ~200 B (reused
 *   buffers). Use these when a loop runs.
 * - Fastest whole-array walk: np.nditer_chunks<T> (Span) ~70 us @100K — 4x the ref element walks —
 *   because it hands a contiguous Span<T> ready to vectorize. (n/a on Strided: a Span can't describe
 *   a stepped inner loop; use np.nditer<T> or a .copy() there.)
 * - Robust element walks: np.flat<T> / np.nditer<T> ~278 us @100K, layout-independent (views are
 *   coalesced), flat 64 B.
 * - nd.AsEnumerable<T>() is the LINQ bridge: on CONTIGUOUS it is the fastest per-element walk here
 *   (191 us @100K, 64 B) — BDN's sustained hot loop lets Dynamic PGO devirtualize+inline the
 *   IEnumerable<T> enumerator — but on ANY non-contiguous layout it costs ~8x (1.62 ms) AND allocates
 *   ~40 B/element (3.81 MB), because GetAtIndex<T> does per-element stride math. LINQ, not hot loops.
 * - Boxed / parity forms allocate MBs and run 10-1700x slower (boxed object or NDArray view per
 *   element): a.flat 3 MB, ndenumerate(by-value) 4-8 MB, ndindex 3.8 MB, np.nditer(boxed) 44 MB/~114
 *   ms, np.nested_iters 83 MB/~110 ms, np.broadcast 12-16 MB. Keep them for NumPy parity / flags only.
 *   (np.nditer(boxed) over the stride-0 Broadcast is ~39 ms vs ~114 ms elsewhere — the 'K'-order walk
 *   re-visits the single backing row rather than 100K distinct addresses.)
 * ============================================================================================ */

using System;
using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Iteration;

/// <summary>
/// The five array KINDS iterated. Each is a 2-D <c>float64</c> array of ~N elements in a distinct
/// memory layout, so the benchmark exercises every iteration code path (contiguous fast path, F-order,
/// non-unit stride, negative stride, and stride-0 broadcast).
/// </summary>
public enum ArrayKind
{
    /// <summary>C-contiguous, unit stride — the fast path.</summary>
    Contiguous,
    /// <summary>Transpose view (F-contiguous) — non-contiguous in C order, coalescable in memory order.</summary>
    Transposed,
    /// <summary>Stepped view <c>[:, ::2]</c> — non-unit inner stride (the case <c>nditer_chunks&lt;T&gt;</c> cannot span).</summary>
    Strided,
    /// <summary>Reversed view <c>[:, ::-1]</c> — negative inner stride.</summary>
    Reversed,
    /// <summary>Broadcast view (a row stretched over every row) — stride-0, read-only.</summary>
    Broadcast,
}

/// <summary>
/// A fair, self-verifying comparison of every way to iterate / enumerate an <see cref="NDArray"/>,
/// across 5 array kinds × 3 sizes (Scalar / 1K / 100K). Companion to the docs page
/// <c>docs/website-src/docs/iterating-and-enumerating.md</c>.
///
/// <para>
/// <b>What is measured.</b> Each <c>[Benchmark]</c> walks the WHOLE array with one technique and folds
/// every element into a <c>double</c> checksum (returned, so nothing is dead-code-eliminated). Row-
/// and chunk-yielding techniques (<c>foreach</c> over axis 0, <c>nditer_chunks</c>) accumulate over the
/// yielded row/chunk, so they touch all N elements too — the honest cost of using that technique to
/// process the whole array. <c>MemoryDiagnoser</c> reports allocations, which is where the boxed
/// techniques (flat / flatiter / ndenumerate / boxed nditer / ndindex / foreach) pay and the typed
/// ones (<c>nditer&lt;T&gt;</c> / <c>nditer_chunks&lt;T&gt;</c>) do not.
/// </para>
///
/// <para>
/// <b>Boxed vs unboxed pairs.</b> Where a technique has both a boxed and a typed/by-<c>ref</c> form, BOTH
/// are measured so the allocation delta is visible head-to-head: <c>a.flat</c>/<c>a.flatiter</c> vs
/// <c>np.flat&lt;T&gt;</c>; <c>np.ndenumerate&lt;T&gt;</c> (by-value, fresh <c>long[]</c>/step) vs
/// <c>np.ndenumerate&lt;T&gt;().AsRef()</c> (<c>ReadOnlySpan&lt;long&gt;</c> + <c>ref T</c>);
/// <c>np.ndindex</c> (fresh <c>long[]</c>/step) vs <c>np.ndindex().AsSpans()</c> (reused buffer); plus
/// <c>nd.AsEnumerable&lt;T&gt;()</c> (unboxed <c>IEnumerable&lt;T&gt;</c>, one enumerator allocation).
/// </para>
///
/// <para>
/// <b>Fairness.</b> Every element is an <c>arange</c> value, and an integer-valued <c>double</c> sum is
/// exact and order-independent up to 2^53 (max here ≈ 1e10), so all single-operand techniques must
/// produce the IDENTICAL checksum regardless of traversal order (C-order vs memory-order). <see cref="Verify"/>
/// asserts that in <c>[GlobalSetup]</c> and fails the cell otherwise, so a technique that skipped or
/// double-counted elements can never masquerade as fast.
/// </para>
///
/// <para>
/// <b>Two cells are legitimately N/A.</b> <c>np.nditer_chunks&lt;T&gt;</c> needs a unit-stride inner loop
/// (a <c>Span&lt;T&gt;</c> is contiguous by definition), so on the <see cref="ArrayKind.Strided"/> kind
/// (inner stride 2) it throws <c>NotSupportedException</c> and BenchmarkDotNet reports the cell as NA —
/// the truthful outcome, not a fabricated timing. <c>np.broadcast</c> is a TWO-operand technique, so its
/// checksum differs from the single-operand ones and it is excluded from the equality check.
/// </para>
///
/// Experimental (NumSharp-internal, no NumPy twin): run with
/// <c>dotnet run -c Release -- --filter "*Iteration*"</c>.
/// </summary>
[BenchmarkCategory("Iteration")]
public class IterationBenchmarks : BenchmarkBase
{
    /// <summary>Element-count tiers — Scalar (1), 1K (1000), 100K (100000).</summary>
    [Params(ArraySizeSource.Scalar, ArraySizeSource.Small, ArraySizeSource.Medium)]
    public override int N { get; set; }

    /// <summary>The array layout under test.</summary>
    [Params(ArrayKind.Contiguous, ArrayKind.Transposed, ArrayKind.Strided, ArrayKind.Reversed, ArrayKind.Broadcast)]
    public ArrayKind Kind { get; set; }

    private NDArray _a = null!;   // the array-of-kind (float64, 2-D, ~N elements)
    private NDArray _b = null!;   // a (1, cols) partner for the np.broadcast (two-operand) case

    // Row/col factorization of each size tier into a 2-D shape.
    private static (long r, long c) Factor(int n) => n switch
    {
        ArraySizeSource.Scalar => (1, 1),       // 1     → (1, 1)   — the fixed-overhead tier
        ArraySizeSource.Small => (25, 40),      // 1000  → (25, 40)
        ArraySizeSource.Medium => (100, 1000),  // 100000→ (100, 1000)
        _ => (1, n),
    };

    private static NDArray BuildKind(ArrayKind kind, int n)
    {
        var (r, c) = Factor(n);
        return kind switch
        {
            ArrayKind.Contiguous => np.arange(r * c).astype(np.float64).reshape(r, c),
            ArrayKind.Transposed => np.arange(r * c).astype(np.float64).reshape(c, r).T,          // F-contiguous view
            ArrayKind.Strided => np.arange(r * 2 * c).astype(np.float64).reshape(r, 2 * c)[":, ::2"], // inner stride 2
            ArrayKind.Reversed => np.arange(r * c).astype(np.float64).reshape(r, c)[":, ::-1"],   // inner stride -1
            ArrayKind.Broadcast => np.broadcast_to(np.arange(c).astype(np.float64).reshape(1, c), (r, c)), // stride-0 rows
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
    }

    [GlobalSetup]
    public void Setup()
    {
        _a = BuildKind(Kind, N);
        long cols = _a.ndim > 1 ? _a.shape[1] : 1;
        _b = np.arange(cols).astype(np.float64).reshape(1, cols);   // broadcasts over (rows, cols)
        Verify();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a?.Dispose();
        _b?.Dispose();
        _a = null!;
        _b = null!;
        GC.Collect();
    }

    // ----------------------------------------------------------------- techniques

    [Benchmark(Description = "foreach (row) foreach (x)")]
    [BenchmarkCategory("row")]
    public double ForEachAxis0()
    {
        double s = 0;
        foreach (NDArray row in _a)
            foreach (var x in row)
                s += (double)x;
        return s;
    }

    [Benchmark(Description = "a.flat (boxed)")]
    [BenchmarkCategory("element")]
    public double Flat()
    {
        double s = 0;
        foreach (var x in _a.flat)   // raveled NDArray — COPIES when non-contiguous
            s += (double)x;
        return s;
    }

    [Benchmark(Description = "a.flatiter (boxed, any layout)")]
    [BenchmarkCategory("element")]
    public double FlatIter()
    {
        double s = 0;
        var f = _a.flatiter;         // write-through view, no copy
        foreach (var x in f)
            s += (double)x;
        return s;
    }

    [Benchmark(Description = "np.flat<T> (ref, C-order)")]
    [BenchmarkCategory("element")]
    public double FlatTyped()
    {
        double s = 0;
        foreach (ref double x in np.flat<double>(_a))   // unboxed, logical C-order (== nditer<T> order:'C')
            s += x;
        return s;
    }

    [Benchmark(Description = "np.nditer<T> (ref, memory-order)")]
    [BenchmarkCategory("element")]
    public double NdIterTyped()
    {
        double s = 0;
        foreach (ref double x in np.nditer<double>(_a))
            s += x;
        return s;
    }

    [Benchmark(Description = "np.nditer_chunks<T> (Span)")]
    [BenchmarkCategory("chunk")]
    public double NdIterChunks()
    {
        // N/A for the Strided kind (non-unit inner stride): np.nditer_chunks<T> throws
        // NotSupportedException at GetEnumerator, and BenchmarkDotNet reports the cell as NA.
        double s = 0;
        foreach (Span<double> chunk in np.nditer_chunks<double>(_a))
            for (int i = 0; i < chunk.Length; i++)
                s += chunk[i];
        return s;
    }

    [Benchmark(Description = "np.ndenumerate<T> (by-value)")]
    [BenchmarkCategory("element")]
    public double NdEnumerate()
    {
        double s = 0;
        foreach (var (_, v) in np.ndenumerate<double>(_a))   // T unboxed, but a fresh long[] index per step
            s += v;
        return s;
    }

    [Benchmark(Description = "np.ndenumerate<T>().AsRef()")]
    [BenchmarkCategory("element")]
    public double NdEnumerateRef()
    {
        double s = 0;
        foreach (var e in np.ndenumerate<double>(_a).AsRef())   // ReadOnlySpan<long> index + ref T value, no alloc/step
            s += e.Value;
        return s;
    }

    [Benchmark(Description = "nd.AsEnumerable<T>()")]
    [BenchmarkCategory("element")]
    public double AsEnumerableT()
    {
        // Unboxed IEnumerable<T>, C-order, one enumerator allocation. It DOES consume every element
        // (yield-iterates GetAtIndex<T>(0..size-1); the Verify() guard proves its checksum matches).
        // READ THE CONTIGUOUS NUMBER WITH CARE: this is an INTERFACE-dispatched enumerator, so under a
        // sustained hot loop (BDN's ~12,500 invocations) Dynamic PGO devirtualizes + inlines it and it
        // reaches ~100µs/100K — FASTER here than the ref-struct flat<T>/nditer<T> (~148µs, statically
        // dispatched, PGO-insensitive). In a lightly-called / cold loop it is ~1.2–1.8× SLOWER, and on a
        // NON-contiguous layout it is ~10× slower (GetAtIndex does per-element stride math). Prefer
        // np.flat<T>/np.nditer<T> for robust speed; use AsEnumerable<T> for LINQ / IEnumerable<T> APIs.
        double s = 0;
        foreach (double x in _a.AsEnumerable<double>())
            s += x;
        return s;
    }

    [Benchmark(Description = "np.ndindex + GetValue (long[])")]
    [BenchmarkCategory("index")]
    public double NdIndex()
    {
        double s = 0;
        foreach (var idx in np.ndindex(_a.shape))   // fresh long[] index per step
            s += _a.GetValue<double>(idx);
        return s;
    }

    [Benchmark(Description = "np.ndindex().AsSpans() + GetValue")]
    [BenchmarkCategory("index")]
    public double NdIndexSpans()
    {
        // Every array here is 2-D, so idx[0]/idx[1] index it directly. .AsSpans() reuses one
        // coordinate buffer (no per-step index allocation); GetValue's params long[] still allocates.
        double s = 0;
        foreach (var idx in np.ndindex(_a.shape).AsSpans())   // ReadOnlySpan<long> over a reused buffer
            s += _a.GetValue<double>(idx[0], idx[1]);
        return s;
    }

    [Benchmark(Description = "np.nditer (boxed object)")]
    [BenchmarkCategory("boxed")]
    public double NdIterBoxed()
    {
        double s = 0;
        using var it = np.nditer(_a);
        foreach (var v in it)
            s += (double)v[0];
        return s;
    }

    [Benchmark(Description = "np.nested_iters")]
    [BenchmarkCategory("row")]
    public double NestedIters()
    {
        double s = 0;
        var levels = np.nested_iters(_a, new[] { new[] { 0 }, new[] { 1 } });
        using var outer = levels[0];
        using var inner = levels[1];
        foreach (var _ in outer)
            foreach (var __ in inner)
                s += (double)inner[0];
        return s;
    }

    [Benchmark(Description = "np.broadcast (2-operand)")]
    [BenchmarkCategory("broadcast")]
    public double Broadcast()
    {
        double s = 0;
        var bc = np.broadcast(_a, _b);
        foreach (object[] v in bc)
            s += (double)v[0] + (double)v[1];
        return s;
    }

    // ----------------------------------------------------------------- fairness guard

    /// <summary>
    /// Assert every single-operand technique visits exactly the same elements (identical exact checksum),
    /// so no method can look fast by skipping work. Runs once per (N, Kind) in <c>[GlobalSetup]</c>;
    /// throwing here fails the cell, which is the intended signal that the benchmark is not fair.
    /// <c>nditer_chunks</c> is included only where it applies; <c>np.broadcast</c> (two operands) is not.
    /// </summary>
    private void Verify()
    {
        double reference = ForEachAxis0();
        var checks = new (string name, Func<double> f)[]
        {
            ("flat", Flat), ("flatiter", FlatIter), ("flat<T>", FlatTyped), ("nditer<T>", NdIterTyped),
            ("AsEnumerable<T>", AsEnumerableT),
            ("ndenumerate", NdEnumerate), ("ndenumerate.AsRef", NdEnumerateRef),
            ("ndindex", NdIndex), ("ndindex.AsSpans", NdIndexSpans),
            ("nditer(boxed)", NdIterBoxed), ("nested_iters", NestedIters),
        };
        foreach (var (name, f) in checks)
        {
            double got = f();
            if (got != reference)
                throw new InvalidOperationException(
                    $"Iteration benchmark unfair: {name} checksum {got} != reference {reference} " +
                    $"for Kind={Kind}, N={N}.");
        }

        // chunks only where the inner loop is unit-stride (all kinds except Strided at N>1).
        try
        {
            double chunks = NdIterChunks();
            if (chunks != reference)
                throw new InvalidOperationException(
                    $"Iteration benchmark unfair: nditer_chunks checksum {chunks} != reference {reference} " +
                    $"for Kind={Kind}, N={N}.");
        }
        catch (NotSupportedException)
        {
            // Expected on the Strided kind — the benchmark cell will report NA.
        }
    }
}
