# Benchmarks: NumSharp vs NumPy

This is where we present the evidence behind a deliberately ambitious claim: a **managed
.NET array library can keep pace with — and in places outrun — NumPy**, the C/Fortran
reference implementation that has set the bar for array computing for two decades.

The lever that makes this possible is [Runtime IL Generation](il-generation.md). NumSharp
does not interpret array operations through generic loops; for each operation it emits a
specialized, SIMD-vectorized kernel as machine code at runtime, caches it, and reuses it
forever. The pages are paired on purpose: **[IL Generation](il-generation.md) explains *how*;
this page shows *what it buys you* against NumPy, head to head.**

> **The numbers on this page are auto-generated.** Every published release triggers the
> [`Benchmark` workflow](https://github.com/SciSharp/NumSharp/blob/master/.github/workflows/benchmark.yml),
> which runs the whole NumSharp-vs-NumPy suite on one machine and commits the refreshed
> report and the two cards below straight to `master`. The cards always reflect the
> latest committed run — they are not screenshots pasted into this doc.

---

## The headline

<p align="center">
  <img src="https://raw.githubusercontent.com/SciSharp/NumSharp/master/benchmark/nditer/cards/ops.png" alt="NDIter vs NumPy — operations: geomean speedup by array-size tier and by operation class" width="400" height="300">
  &nbsp;
  <img src="https://raw.githubusercontent.com/SciSharp/NumSharp/master/benchmark/nditer/cards/cat.png" alt="NDIter — the IL-generation dividends: iterator construction vs np.nditer, expression fusion, kernel reuse, parallel inner loop" width="400" height="300">
</p>

The **left card** is the head-to-head against NumPy — geomean speedup by array-size
tier and by operation class. The **right card** is the IL-generation *dividend*:
iterator machinery NumPy has no structural equivalent for — cheaper construction than
`np.nditer`, one-pass expression fusion (`np.evaluate`), kernel reuse, and a parallel
inner loop.

Both cards report a single ratio:

```
speedup   = NumPy time ÷ NumSharp time    (> 1.0×  ⇒  NumSharp is faster)
%NumPy🕐 = NumSharp ÷ NumPy × 100         (share of NumPy's time NumSharp uses;
                                           30% = takes only 30% as long; <100% = faster)
```

The cards intentionally show **ratios only, never absolute milliseconds**. Absolute timings
drift with hardware — and the CI runner that produces these is shared, variable silicon — but
the *same-runner* ratio of NumPy to NumSharp stays meaningful from one run to the next, because
both libraries are measured back-to-back on the identical machine against a pinned NumPy
(`2.4.2`).

### What the latest committed run shows

These figures come from the iterator benchmark sheet
([`benchmark/nditer/nditer_results.md`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/nditer/nditer_results.md)) —
the source of truth that the cards are rendered from.

| Operation class | Speedup (NumPy ÷ NumSharp) | %NumPy🕐 | Reading |
|---|---:|---:|---|
| **Reductions** (`sum`, `cumsum`, `any`, axis sums) | **≈ 1.8×** | ≈ 56% | NumSharp's horizontal-SIMD + tree-reduction kernels lead clearly |
| **Dtype-specialized loops** (`int8`, `complex`, …) | **≈ 1.6×** | ≈ 63% | Per-type emitted kernels beat NumPy's generic ufunc loops on narrow types |
| **Elementwise** (`add`, `sqrt`, copy, broadcast) | **≈ 1.1×** | ≈ 89% | Roughly parity — both are memory-bandwidth bound at scale |
| **Copy / cast** (`flatten`, `astype`, `ravel.T`) | **≈ 0.65×** | ≈ 154% | NumSharp's small-N tax; closes to parity (or wins) by 1M+ |
| **Index math** (`unravel_index`, `ravel_multi_index`) | **≈ 0.7×** | ≈ 143% | Scalar-bound; a known laggard, tracked as a canary |

Across the whole operation matrix the geomean lands a little above parity (NumPy ÷ NumSharp
≈ **1.17×**, a majority of cells in NumSharp's favor). The story is not "uniformly faster" —
it is **"faster where the kernels are SIMD-rich, parity where memory bandwidth dominates, and a
small-N tax on a couple of pointer-shuffling operations we have not vectorized yet."** We
publish the losses as loudly as the wins.

---

## Reading the result, class by class

### Reductions — the clearest win

`sum`, `cumsum`, `any`/`all`, and axis reductions are where IL generation pays off most. The
emitted kernels use the [techniques documented on the IL page](il-generation.md#simd-optimization-techniques):
4× loop unrolling, multiple independent accumulators, tree reduction to combine them, and
SIMD early-exit for boolean reductions. A boolean `any()` that finds its hit early returns in
a handful of vector compares — often **20×+** faster than scanning the array.

The honest counter-case is in the same family: `any()` over an **all-false** array can't
early-exit, and at large N NumSharp currently trails NumPy badly there (a known scan gap, on
the list to fix). Both extremes are visible in the published sheet — we don't average them away.

### Dtype-specialized loops — beating generic ufuncs

NumPy dispatches most elementwise work through generic ufunc loops. NumSharp emits a *distinct*
kernel per `(operation, dtype, layout)` and caches it. For narrow integer types like `int8`,
where many elements pack into one SIMD register, the specialized loop can run **several times**
faster than NumPy at cache-resident sizes. This is the structural advantage of generating code
*after* you know the exact type, rather than ahead of time for all of them.

### Elementwise — parity, and why that's the right answer

`add`, `sqrt`, contiguous copy, broadcast — these hover around 1.0×. That is expected and
correct: for large contiguous arrays the operation is **memory-bandwidth bound**, not
compute-bound. Once both libraries saturate the memory bus, the winner is whoever copies the
fewest bytes, and a well-emitted SIMD loop is already at the bandwidth ceiling. Matching a
mature C implementation at the hardware limit *is* the achievement.

### Copy/cast and index-math — the taxes we still pay

`flatten`, `astype`, `ravel().T`, `unravel_index` lag at small N (≈ 0.5–0.7×). These are
pointer-and-bookkeeping heavy rather than arithmetic-heavy, so SIMD buys little and per-call
overhead dominates. They recover toward (and often past) parity once arrays are large enough to
amortize that overhead. They are tracked explicitly as **canaries** in the report so a
regression can't hide.

---

## The dividends NumPy can't structurally match

Some advantages don't come from a faster loop — they come from *owning the code generator*.
These are measured in the **Dividends** section of the report and have no NumPy equivalent
better than "the closest thing NumPy can do."

- **Expression fusion** (`np.evaluate`) — a chained expression like `a*b + c*d - 2` compiles to
  **one** inner-loop pass that reads each operand once and allocates no intermediates, the way
  [`numexpr`](https://github.com/pydata/numexpr) works in the Python ecosystem. Against NumPy's
  unavoidable temporaries it runs **up to ~13×** faster on small/cache-resident chains and stays
  ahead even at 10M.
- **Kernel reuse** — because kernels are cached by operation key, the second and later calls pay
  *zero* generation cost. NumPy re-enters its generic machinery every call.
- **Parallel inner loop** — the iterator can fan a strided workload across cores; the report's
  `par8` row shows **up to ~8×** over the single-threaded path on the same machine.

The iterator itself is also cheap to stand up: building and tearing down an `NDIter` runs
roughly **2–3× faster than constructing `np.nditer`** in NumPy (see the **Construction**
section). For more on the iterator, see [NDIter](NDIter.md).

---

## How the numbers are produced

Two complementary harnesses run under one entry point
([`benchmark/run_benchmark.py`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/run_benchmark.py)),
and the [`Benchmark` workflow](https://github.com/SciSharp/NumSharp/blob/master/.github/workflows/benchmark.yml)
runs that after every release and commits the results:

1. **The operation matrix** — BenchmarkDotNet measures NumSharp across *op × dtype × N*; NumPy
   is measured for the matching cells; the two are merged into one per-cell ratio report
   ([`benchmark/benchmark-report.md`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/benchmark-report.md)).
   This is the broad coverage: every dtype, every operation family, three cache tiers.

2. **The iterator benchmark** — the harness behind the cards
   ([`benchmark/nditer/`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/nditer/README.md)).
   Its result model is *aspect × cache-tier* (construction, traversal, reductions, selection,
   dtypes, pathologies, dividends) rather than op/dtype/N, so it is **appended** to the report,
   not merged. It isolates the iterator machinery the op matrix can't see.

A few methodology points worth knowing when you read the sheet:

- **Always measured under `-c Release`.** Ad-hoc `dotnet run` scripts build *Debug* by default,
  which disables JIT optimization and silently ~2× inflates hand-written kernels. The harness
  asserts the JIT optimizer is on for both assemblies before it records a single number.
- **Best-of-rounds.** Each cell is the fastest of several rounds after warm-up, so the JIT tax
  for first-time kernel generation never pollutes a measurement.
- **Ratios, not absolutes.** Only NumPy ÷ NumSharp on the same runner is reported; raw
  milliseconds are deliberately kept off the cards.
- **AccessViolation → `NA`.** NumSharp has a known *intermittent* unmanaged-storage lifetime bug
  that can crash a heavily allocating section. Rather than mask it, the harness runs each section
  in its own subprocess and reports a crashing section as **`NA` / IGNORED** with a header,
  excluding it from every geomean. An `NA` block in the sheet is that policy firing — not a
  silent omission.

For the full harness internals see
[`benchmark/nditer/README.md`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/nditer/README.md)
and the development guide at
[`benchmark/CLAUDE.md`](https://github.com/SciSharp/NumSharp/blob/master/benchmark/CLAUDE.md).

---

## Reproduce it yourself

```bash
# Full suite — operation matrix + iterator benchmark + cards (matches CI)
python benchmark/run_benchmark.py

# Iterator benchmark only (renders the two cards)
python benchmark/nditer/nditer_sheet.py
python benchmark/nditer/nditer_cards.py
```

Both write their reports under `benchmark/` and the cards to `benchmark/nditer/cards/`. The
absolute numbers will differ on your hardware; the ratios are what carry over.

---

## Read the full reports

Both are rendered as searchable pages on this site, refreshed every release:

- **Iterator benchmark sheet** (drives the cards) → [Iterator sheet (full)](benchmark-iterator.md)
- **Operation matrix** (op × dtype × N) → [Operation matrix (full)](benchmark-matrix.md)
- **How the kernels that produce these numbers are generated** → [IL Generation](il-generation.md)

The raw generated files live in the repo under
[`benchmark/`](https://github.com/SciSharp/NumSharp/tree/master/benchmark) on `master`.
