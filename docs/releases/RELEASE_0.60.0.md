# NumSharp 0.60.0 — The nditer Release

The first **stable** (non-prerelease) NumSharp in the 0.x line. 0.60.0 graduates and consolidates the entire prerelease run since `0.50.0` into one release: a from-scratch port of NumPy 2.4.2's `nditer` engine, a fused-expression DSL (`np.evaluate`), **full advanced-indexing parity** with NumPy (down to the memory-safety and error-text level), **byte-exact NumPy array printing**, C/F/A/K memory-layout support wired through the whole API, stride-native matmul, NumPy-seed-compatible `np.random` (MT19937), 36+ new `np.*` functions, deterministic memory management, and a differential-fuzz pipeline that proves bit-exactness against NumPy.

**731 commits since `0.50.0`** — 617 of them the nditer engine branch. The net is a large body of new engine code offset by an equally large deletion of *legacy generated code* (the Regen template engine, `NDIterator`, and `MultiIterator` are all gone). Everything below was validated against **NumPy 2.4.2** ground truth — by a ~40,000-case differential corpus, 566 iterator-parity scenarios, a 12,000+-case index oracle, ~18,000 array-print fuzz cases, and per-feature battle tests run on actual NumPy output.

> This release supersedes and absorbs the unreleased `0.51.0-prerelease` draft.

---

## TL;DR

- **`NDIter` — full port of NumPy 2.4.2's `nditer`** (~12.5K lines): all iteration orders (C/F/A/K), all indexing modes, buffered casting, buffered-reduce double-loop, masking, memory-overlap protection (`COPY_IF_OVERLAP`), windowed buffering (`DELAY_BUFALLOC`), unlimited operands and dimensions. It is the production execution engine for the elementwise/reduce core, **at or faster than NumPy on every probed aspect**.
- **`NDExpr` DSL + three-tier custom-op API**, exposed as the public **`np.evaluate`** — write your own ufunc (raw IL / element-wise SIMD / composable expression trees) and run fused expressions **3.2–6.1× faster than NumPy** (which can't fuse), with per-node NumPy `result_type` typing and fused reductions.
- **Full advanced-indexing parity.** A faithful port of NumPy's `prepare_index` + unified advanced-index gather/scatter takes the get/set surface from ~697 → **0 divergences** (full parity) against a new committed **differential index oracle**, closes a class of **memory-safety bugs** (out-of-bounds gather/scatter on exotic mixed indices), and aligns every `IndexError`/`ValueError` text with NumPy.
- **Byte-exact NumPy array printing.** `NDArray.ToString()` is now a 1-to-1 port of NumPy 2.4.2's `array_str`/`array_repr`/`array2string` + Dragon4 float formatting; new public `np.array2string`/`np.array_repr`/`np.array_str`/`np.set_printoptions`/`np.printoptions`/`np.format_float_positional`/`np.format_float_scientific`. ~18,000 fuzz cases byte-identical to NumPy.
- **`out=` / `where=` / `dtype=` ufunc kwargs across the elementwise API** — binary, unary-math, comparison, predicate, and bitwise families with exact NumPy broadcast/cast/error-text semantics. Plus `np.bitwise_and/or/xor`, `np.positive`, and first-class `np.maximum/minimum/fmax/fmin`.
- **36+ new `np.*` APIs** — `sort`/`argsort`, `pad` (11 modes), `tile`, `median`/`percentile`/`quantile` (all 13 interpolation methods) + `nan*` variants, `average`, `ptp`, `take`/`put`/`place`, `extract`/`compress`, `diagonal`/`trace`, `argwhere`/`flatnonzero`, `unravel_index`/`ravel_multi_index`/`indices`, `delete`/`insert`/`append`, `split`/`array_split`/`hsplit`/`vsplit`/`dsplit`, `diff`/`ediff1d`, `asfortranarray`/`ascontiguousarray`, `np.multithreading`, plus the printing APIs above.
- **`np.random` rebuilt for NumPy RNG parity** — the legacy Knuth subtractive generator is replaced by **MT19937** (NumPy's Mersenne Twister) for 1-to-1 seed/state compatibility, plus **24 new distribution samplers** (`weibull`, `vonmises`, `pareto`, `laplace`, `gumbel`, `dirichlet`, `multivariate_normal`, `noncentral_chisquare`/`noncentral_f`, `standard_t`/`standard_cauchy`/`standard_gamma`/`standard_exponential`, `triangular`, `zipf`, `logseries`, `rayleigh`, `wald`, `power`, `f`, `logistic`, `hypergeometric`, `multinomial`, `negative_binomial`). Output is byte-identical to NumPy 2.4.2 at a given seed.
- **C/F/A/K order support wired through the whole API** — `Shape` understands F-contiguity, `OrderResolver` resolves NumPy order modes, ~68 layout bugs fixed.
- **Stride-native matmul/dot** — BLIS-style GEBP GEMM absorbs arbitrary strides for all dtypes (kills a ~100× penalty on transposed inputs); fused 1-D dot 3.5–9× faster with zero GC; opt-in multithreaded dot ~2× faster than NumPy's default on 1M vectors.
- **Type casting (`astype`) faster than NumPy across the board** — the entire copy/retype/cast surface is unified on one `NDIter.CopyAs` core (the 2,226-line legacy per-element cast loop deleted), then a SIMD campaign took the full 15×8×15 `astype` matrix from **716 → 129 lagging cells of 1,568 comparable** (1,439 winning ≥1.0× vs NumPy 2.4.2).
- **NumPy-parity benchmark: geomean 1.26× at 10M elements** (397 faster / 150 close / 42 slower of 615 ops; 1.14× at 1K, 0.90× at 100K), from a committed, reproducible BenchmarkDotNet-vs-NumPy harness — an op/dtype/N matrix over all 15 dtypes plus five appended subsystems (iterator, memory-layout, operand, cast, fusion) with per-release provenance snapshots.
- **Deterministic memory management** — atomic reference counting + `IDisposable` on `NDArray`, plus a tcache-style buffer pool (1 B – 64 MiB window).
- **Differential fuzzing vs NumPy** — ~40,000 bit-exact corpus cases across 25+ tiers, a seeded random fuzzer with shrinker, a CI `FuzzMatrix` gate, and a nightly soak workflow.
- **Legacy stacks deleted outright** — `MultiIterator`, `NDIterator` (interface + class + `AsIterator`), and the **Regen template engine** (87 inline `#if _REGEN` blocks across 35 files) are all gone.
- **Cross-platform** — macOS/Apple-Silicon (ARM64) signed-zero + integer-widening reduction parity fixed.
- **Test suite: ~10,980 passed / 0 failed** on net8.0 + net10.0, plus the differential-fuzz corpora replayed by the `FuzzMatrix` gate. **177 formerly-`[OpenBugs]` reproductions** were promoted into regular CI tests as their bugs were fixed.

---

## 1. `NDIter` — full NumPy `nditer` port

From-scratch C# port of NumPy 2.4.2's iterator machinery under `src/NumSharp.Core/Backends/Iterators/` (~12,557 lines), promoted to **public API** with NDArray overloads. The public surface includes the NumPy-named flag enums (`NDIterFlags`/`NDIterOpFlags`/`NDIterGlobalFlags`, `NPY_ORDER`, `NPY_CASTING`), the `NDIterRef` kernel handle, and the `NDInnerLoopFunc` per-chunk delegate; the standalone flat iterator is `NDFlatIterator` (drives `np.broadcast(...).iters`).

| Capability | Detail |
|---|---|
| Iteration orders | C, F, A, K — incl. NEGPERM negative-stride handling, axis reordering + coalescing to full 1-D collapse |
| Indexing modes | `MULTI_INDEX`, `C_INDEX`, `F_INDEX`, `RANGE` (parallel chunking), `GotoIndex` / `GotoMultiIndex` / `GotoIterIndex` |
| Buffering | Buffered casting with all 5 casting rules, **windowed buffered iteration**, `DELAY_BUFALLOC`, buffered-reduce double-loop (incl. `bufferSize < coreSize`) |
| Reductions | `op_axes` with `-1` reduction axes, `REDUCE_OK`, `IsFirstVisit`, `REUSE_REDUCE_LOOPS` slab accumulation |
| Overlap safety | **`COPY_IF_OVERLAP`** via a port of NumPy's `mem_overlap` solver (`NDMemOverlap.cs`) — overlapping in/out operands no longer silently corrupt |
| Masking | `WRITEMASKED` + `ARRAYMASK` **executed** — the buffered window flush writes back only mask-nonzero elements; `VIRTUAL` operands construct with NumPy 2.x semantics |
| Operands / dims | **Unlimited operands** (NumPy caps at `NPY_MAXARGS=64`) and **unlimited dimensions** (NumPy caps at `NPY_MAXDIMS=64`) via dynamic allocation |
| APIs | `Copy`, `GetIterView`, `RemoveAxis`, `RemoveMultiIndex`, `ResetBasePointers`, `IterRange`, `DebugPrint`, fixed/axis stride queries, `GetValue<T>`/`SetValue<T>`, … |
| Casting parity | `NDIterCasting.CanCast` matches NumPy's `safe`/`same_kind` lattice exactly |

Validated by a dedicated battletest harness: **566 scenarios** replayed against NumPy 2.4.2 byte-for-byte, plus a permanent variation-probe harness. Dozens of parity bugs found and fixed against NumPy ground truth (negative-stride flipping, `NO_BROADCAST` enforcement, `F_INDEX` coalescing, buffered-reduction stride inversion, K-order on broadcast inputs, the size-1 stride-0 invariant, `op_axes` out-of-bounds reads on stretched size-1 axes, write-broadcast validation, unit-axis absorption) — each reproduced against NumPy first, then fixed by adopting NumPy's constructor structure.

### Execution at NumPy speed

`NDIter` isn't just correct — it is the production execution engine: `DefaultEngine`'s binary, unary, and comparison ops (same- and mixed-dtype) route through the NDIter Tier-3B shell.

| Aspect (float32) | NumSharp | NumPy | Ratio (NPY/NS) |
|---|---|---|---|
| contig sqrt 10M | 2.98 ms | 3.24 ms | 1.09× |
| contig add 10M | 3.91 ms | 4.09 ms | 1.05× |
| strided add 1M | 319 µs | 416 µs | 1.30× |
| strided sqrt 1M | 206 µs | 374 µs | 1.82× |
| strided sum 1M | 109 µs | 205 µs | 1.88× |
| **fused** `a*b+c` 10M | 4.77 ms | 13.38 ms | **2.81×** |
| **fused** `(a-b)/(a+b)` 10M | 4.12 ms | 22.33 ms | **5.42×** |

Key mechanisms: an O(1) **trivial-loop bypass** that skips iterator construction for contiguous operands, identity-broadcast fast paths, **AVX2 hardware-gather** (`vgatherdps`) strided SIMD in the Tier-3B shell (NumPy uses scalar loops for strided binary/reduce — its floors are beatable), and strided-reduction kernels.

## 2. `NDExpr` DSL + `np.evaluate` (fusion)

User-extensible kernel layer on top of `NDIter` — the public answer to "how do I write my own ufunc":

- **Tier 3A — `ExecuteRawIL`**: emit raw IL against the NumPy ufunc signature `void(void** dataptrs, long* strides, long count, void* aux)`.
- **Tier 3B — `ExecuteElementWise`**: provide scalar + vector IL; the shell supplies a 4×-unrolled SIMD loop, remainder vector, scalar tail, and strided fallback.
- **Tier 3C — `ExecuteExpression`**: compose `NDExpr` trees with C# operators (`(a - b) / (a + b)`), 50+ node types (arithmetic, trig, exp/log, rounding, predicates, comparisons, `Min/Max/Clamp/Where`), plus **`Call()`** to splice any delegate/`MethodInfo` into a fused kernel. Compiled once, cached by structural key, ~5 ns dispatch.

Exposed publicly as **`np.evaluate(expr[, operands][, out])`**:

- **Per-node NumPy `result_type` typing** — every node resolves to its NumPy 2.4.2 dtype, so mixed trees wrap correctly: `(i4*i4)+f8` wraps the multiply in int32 (→ `1410065408`) before promoting. Strong-strong NEP50 (incl. int/float tier crossing), weak python-scalar literals (`i4+2 → i4`, `i4/2 → f8`) with NumPy's exact `OverflowError`, and special resolvers (`true_divide`, `arctan2`, negative-integer-literal `power` → `ValueError`, bool `add`=OR/`multiply`=AND).
- **Fused reductions** — `NDExpr.Sum/Prod/Min/Max/Mean` compile a one-pass inner loop; `sum(a*b)` reads `a` and `b` once and never materializes the product. NumPy reduction dtypes (int→i64, uint→u64, mean→f64).
- **`out=` joins via the ufunc rules** (same_kind validation, reference identity, overlap-safe aliasing through `COPY_IF_OVERLAP`); an `EXTERNAL_LOOP` guard prevents the silent `count==1` slow path.
- **Measured** (Release, 4M f64, NumPy 2.4.2): `a*b+c` **3.2×**, `(a-b)/(a+b)` **6.1×**, `sum(a*b)` **3.6×**, `sum f32` 2.9×, `i4*2+f8` 3.5× faster. Permanent gate in `benchmark/fusion/`.
- A runtime+type-aware SIMD gate decides per-node whether the rounding family (`Floor`/`Ceil`/`Round`/`Truncate`) vectorizes (float/double only, and only where the BCL provides `Vector{N}.<op>` at the active width), so fused integer rounding no longer crashes and SIMD is used wherever it exists on the running runtime.

## 3. Full advanced-indexing parity + memory-safety

A differential index oracle (NumPy 2.4.2 as the sole oracle) proved the common get/set surface was bit-exact but exposed ~697 divergences across **exotic mixed advanced-index combinations** (boolean-array + fancy, multi-dim fancy + slice, 0-d-bool + fancy, multi-fancy, empty combos) **plus a flaky heap-corruption crash**. This release replaces the ad-hoc per-shape "Try\*" patchwork with a faithful port of NumPy's `mapping.c` model and drives the oracle from ~697 → **0 divergences** (full parity). The final fixes closed the remaining divergence categories — value-broadcast on empty/scalar selections, 0-d-base over-indexing, empty-advanced gather, empty-slice assignment, and non-consecutive 0-d-bool axis placement (which uncovered and fixed a latent `Shape.Broadcast` hash collision) — all pinned by a new `Indexing.CombinatorialParity` `FuzzMatrix` gate; the curated, dtype, and seeded-random oracle tiers are bit-exact. The one known open item is a flaky teardown heap-corruption crash on a single corpus shape (`Index_Random` stays `[OpenBugs]` for the *crash*, not for any parity gap).

- **`PrepareIndex`** (`Selection/NDArray.Indexing.PrepareIndex.cs`) — a port of NumPy's `prepare_index`: one up-front pass that classifies the whole index tuple (HAS_* bitmask) and **validates** it before any kernel runs. Raises NumPy-verbatim texts for too-many-indices, boolean-array-dim mismatch, integer/array value out-of-bounds, and un-broadcastable advanced blocks.
- **`TryBuildMultiAdvancedGrid`** — NumPy's general advanced-index algorithm (block broadcast + consec-aware axis placement). **All** advanced tuples (single or multiple fancy, fancy + slice/newaxis, 0-d bool joining the block, pure-advanced) now route through one gather/scatter; only pure-basic tuples fall through to the view path. Correct axis placement for ≥2 advanced indices mixed with slices (consecutive → in place; separated → advanced axes to front).
- **Memory-safety fixes** (the heap-corruption class is closed): negative scalar-index assignment no longer writes one element *before* the buffer; fancy negative-OOB no longer reads OOB; per-axis integer OOB is validated (`a[0,4]` on a 3×4 now raises instead of returning a neighbour); too-many advanced indices raise instead of walking strides past the array; the negative-stride gather offset bound was corrected (valid rows on `a[::-1]` no longer rejected); block-copy bounds guards + an opt-in Windows page-heap (`NUMSHARP_GUARD_PAGES=1`) backstop the gather/scatter.
- **Combined boolean + advanced indexing** — a boolean mask mixed with int/slice/array (`arr[mask, int]`, `arr[:, mask]`, `arr[mask, 1:3]`, leading k-D mask + basic) now works for get **and** set, expanding each mask to its `nonzero()` components like NumPy.
- **Assignment validation** — value-broadcast is checked on assignment (partial writes rejected with NumPy's `could not broadcast …` ValueError); a valid smaller value tiles like NumPy; a broadcastable value scatters correctly into a ≥2-D fancy subspace; empty-into-non-empty raises, empty-into-empty no-ops.
- **Input-form parity** — raw `bool[]`/`bool[,]` (any rank) and `IEnumerable<bool>` are recognized as masks; C# tuples (`nd[(1,2)]`) spread to coordinates; `List<int>`/`ArrayList`/any `IEnumerable` coerce to a fancy index; `uint64` scalar indices and `int8` fancy-index dtype are accepted (all eight integer dtypes now index identically).
- **Differential index oracle** committed as a `FuzzMatrix` gate (`test/oracle/gen_index_oracle.py` + `Fuzz/IndexOracleTests.cs`): `index_curated.jsonl` (2,265) + `index_dtype.jsonl` (104) run in CI bit-exact, and the 10,000-case seeded random tier is now **0 divergences** too (replayed in soak; held out of the per-PR gate only by the teardown crash above). The combinatorial fixes are additionally pinned by the `Indexing.CombinatorialParity` gate.

> **Breaking:** a raw `int[]`/`long[]` used as the **sole** index is now **fancy** indexing, not coordinate access — see Breaking Changes.

## 4. Byte-exact NumPy array printing

`NDArray.ToString()` is refactored from the legacy Python-list output (`[0, 1, 2]`, no alignment/precision/summarization/dtype) to a 1-to-1 port of NumPy 2.4.2's array printing (`numpy/_core/arrayprint.py` + `dragon4.c`).

- `ToString()` / `ToString(false)` → `np.array_str` → `"[0 1 2]"` (the `str()` form).
- `ToString(true)` → `np.array_repr` → `"array([0, 1, 2], dtype=…)"` (the `repr()` form).
- New subsystem `src/NumSharp.Core/Backends/Printing/`: `PrintOptions` (AsyncLocal context mirroring NumPy's `format_options`), `Dragon4` (positional + scientific float→string), `ElementFormatters` (bool/int/float/complex with the two-pass exp-format decision, per-dtype cutoffs, column sizing, nan/inf fields), `ArrayFormatter` (recursive layout, line-wrap at `linewidth`, summarization with `edgeitems`, the 0-d `str`-vs-`repr` asymmetry, repr dtype/shape suffixes).
- Float digit generation uses .NET's shortest round-trip `ToString("R")` (== Dragon4 unique) but routes **all rounding** through `ToString("F"|"E"+precision)` (rounds the true binary value, IEEE half-to-even) — never the shortest string (which diverges ~50% on adversarial ties).
- Public API (`APIs/np.array2string.cs`): `np.array2string`, `np.array_str`, `np.array_repr`, `np.set_printoptions`, `np.get_printoptions`, `np.printoptions` (IDisposable context), `np.format_float_positional`, `np.format_float_scientific`.
- NumSharp's `Char` dtype (string storage, no NumPy equivalent) keeps its legacy rendering, so `GetString`/`AsString` are unaffected.

Validated by ~18,000 differential-fuzz cases byte-identical to NumPy 2.4.2 across all dtypes (incl. float16/32/64 scientific, adversarial ties/carries), 1-D–5-D, sliced/broadcast/transposed views, complex, nan/inf, summarization, line-wrap, and print options; ~174 strict parity tests on net8.0 + net10.0.

> **Breaking:** any code parsing the old `ToString()` format must update — see Breaking Changes.

## 5. C/F/A/K memory-layout support

- `Shape` now tracks **F-contiguity** with NumPy-convention contiguity computation; new `OrderResolver` resolves `C`/`F`/`A`/`K` for every API with an `order` parameter.
- Order support wired through: `copy`, `array`, `asarray`, `asanyarray`, `*_like`, `astype`, `flatten`, `ravel`, `reshape`, `eye`, `concatenate`, `cumsum`, `argsort`, `tile`, plus **post-hoc F-contig preservation across the IL-kernel dispatchers**.
- New: `np.asfortranarray`, `np.ascontiguousarray`.
- `np.where` selects C/F output layout the way NumPy does; `ravel('F')` of an F-contig source returns a **view** (was a 3,000× copy).
- ~68 layout bugs fixed across 9 TDD fix groups, backed by ~3,300 lines of order tests (reductions/keepdims, matmul/dot/outer/convolve, broadcasting-from-F, manipulation, file I/O `fortran_order`, Decimal scalar path, fancy-write isolation, …).

## 6. New & completed `np.*` APIs

**New functions:**

| Area | APIs |
|---|---|
| Fused / ufunc | `np.evaluate` (fused expressions — §2), `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `np.positive`, first-class `np.maximum`/`np.minimum`/`np.fmax`/`np.fmin` |
| Printing | `np.array2string`, `np.array_str`, `np.array_repr`, `np.set_printoptions`, `np.get_printoptions`, `np.printoptions`, `np.format_float_positional`, `np.format_float_scientific` |
| Sorting | `np.sort` (+ `ndarray.sort`; `np.argsort` reimplemented) — radix line-kernel on NDIter, stable, NaN-last, all axes / orders (closes a long-standing Missing Function) |
| Manipulation | `np.pad` (all 11 NumPy modes + callable), `np.tile`, `np.delete`, `np.insert`, `np.append` |
| Splitting | `np.split`, `np.array_split` (uneven splits), `np.hsplit`, `np.vsplit`, `np.dsplit` |
| Indexing/selection | `np.take`, `np.put`, `np.place`, `np.extract`, `np.compress`, `np.argwhere`, `np.flatnonzero`, `np.diagonal`, `np.trace`, `np.unravel_index`, `np.ravel_multi_index`, `np.indices` |
| Statistics | `np.median`, `np.percentile`, `np.quantile` (**all 13 interpolation methods**, tuple axis, `out=`, `keepdims`, QuickSelect), `np.average` (`weights`, `returned`, tuple-axis), `np.ptp`, `np.nanmedian`, `np.nanpercentile`, `np.nanquantile` |
| Math | `np.diff`, `np.ediff1d` |
| Creation | `np.asfortranarray`, `np.ascontiguousarray` |
| Runtime | `np.multithreading(enabled, max_threads)` — opt-in threaded kernels |

**Rebuilt to full NumPy 2.x parity:** `np.clip` (`min=`/`max=` aliases, None bounds, 2.x promotion, `out=`), `np.unique` (5 missing kwargs, NaN partitioning, up to 43× faster), `np.searchsorted` (`side=`/`sorter=`, IL binary-search 5–25× faster), `np.copyto` (`casting=`/`where=`, overlap-safe), `np.asarray` (`copy=`/`like=`/`device=`/string dtype), `np.concatenate`, `np.all`/`np.any` (tuple-axis, `out=`, `where=`), `np.expand_dims` (tuple axis), `np.repeat` (`axis=`), `np.power` (integer-power semantics + crash fix), `np.broadcast` (N-operand `0..∞`, live cursor, lazy `.iters`/`.numiter`). Engine completeness: bool/char `max`/`min`, Complex quantile, `IsInf` implemented, and the six **Complex transcendentals** `sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan` (were `NotSupportedException`).

**`np.maximum`/`minimum`/`fmax`/`fmin`** are now first-class binary ufuncs (NEP50 promotion, broadcasting, `out=`, `where=`, every execution path) instead of `clip`-based shims — which fixes a real correctness bug: **`fmax`/`fmin` now ignore NaN** (return the finite operand) while `maximum`/`minimum` propagate it.

### `out=` / `where=` / `dtype=` ufunc kwargs (NumPy parity)

The kwargs on every NumPy ufunc now span the elementwise core — binary (`add`, `subtract`, `multiply`, `divide`, `true_divide`, `mod`, `power`, `floor_divide`), unary-math (`sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `abs`/`absolute`, `negative`, `square`, …), the six comparisons, predicates (`isnan`/`isfinite`/`isinf`), bitwise, `invert`, `arctan2` — each as **one NumPy-shaped overload**, every rule pinned against NumPy 2.4.2:

- `out` joins the broadcast but **never stretches** (mismatched/stretchable `out` raise NumPy's exact texts, trailing space included); loop dtype resolved from inputs (NEP50); `out` only needs a same_kind cast; the provided instance is returned (reference identity).
- `where` must be exactly `bool`; it broadcasts over operands **and** participates in output shape; mask-false slots keep prior `out` contents.
- `out` **aliasing an input is well-defined** via `COPY_IF_OVERLAP`.
- `dtype=` **computes in the loop dtype** (`subtract(300, 5, dtype=i16) = 295`), with the bool `add`→OR / `multiply`→AND remap keyed off the **final** loop dtype.

### Random sampling — NumPy RNG parity (`np.random`)

`np.random` is rebuilt around NumPy's own bit generator and legacy distribution algorithms, so a seeded NumSharp stream is byte-identical to NumPy 2.4.2.

- **MT19937 bit generator** replaces the legacy Knuth subtractive generator — a full Mersenne-Twister port (`Seed(uint)` / `SeedByArray(uint[])`, 53-bit `NextDouble`, rejection-sampled bounded ints, 624-word state `Clone`/`SetState`) with **1-to-1 seed/state compatibility**: `get_state()`/`set_state()` use NumPy's state tuple format (`Algorithm`, `Key[624]`, `Pos`, `HasGauss`, `CachedGaussian`). Gaussian generation moved from Box-Muller to the **Marsaglia polar method** with the `_hasGauss`/`_gaussCache` carry, matching NumPy's `random_standard_normal` (and its cached state) exactly.
- **24 new distribution samplers** — `dirichlet`, `f`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `logseries`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_t`, `triangular`, `vonmises`, `wald`, `weibull`, `zipf` — joining the existing 16 (`rand`, `randn`, `randint`, `uniform`, `normal`, `bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `exponential`, `gamma`, `geometric`, `lognormal`, `poisson`, `permutation`/`shuffle`).
- **Legacy-algorithm alignment** — the pre-existing samplers were corrected to NumPy's `RandomState` algorithms so they consume the RNG in the same order and emit byte-identical values: `geometric` (NumPy's search algorithm; previously returned negatives), `beta` (Jöhnk's algorithm for `a,b ≤ 1`), `chisquare` (`SampleStandardGamma` with Vaduva's algorithm for `shape < 1`), the `lognormal(mean=0)` NaN fix, and `size=0` empty-array support across every sampler.
- **`multivariate_normal`** uses an SVD transform (Jacobi eigendecomposition, eigenvalues sorted descending) matching NumPy's implementation; identity covariance is an exact sequence match.
- **Integer distributions return `int64`** — `poisson`, `binomial`, `geometric`, `hypergeometric`, `zipf`, `logseries`, `negative_binomial` (matching NumPy dtypes); scalar overloads return a 0-d `NDArray`; `long` is the canonical size/index type throughout (all `int` downcasts removed). Size/axis/seed validation, the return contract (`size=None` → scalar, `size=()` → 0-d), and the dual-overload API shape all follow NumPy.

> **Breaking:** the RNG swap means a given seed now produces a **different sequence** than prior NumSharp versions — intentional, because the new sequence matches NumPy. See Breaking Changes.

## 7. Linear algebra

- **Stride-native GEMM for all 12 numeric dtypes** — BLIS-style GEBP with stride-aware packers; the 8×16 `Vector256` FMA micro-kernel reads packed panels, so transposed/sliced inputs cost nothing extra. Eliminates the ~100× fallback penalty (`np.dot(x.T, grad)`: 240 ms → ~1 ms) and the boxing `GetValue` fallback chain.
- **Full `matmul` gufunc semantics** — batched stacking, 1-D promotion/squeeze rules, validated by a differential matrix (816 cases).
- **Fused single-pass 1-D dot** — 3.5–9× faster, **zero GC** (was up to 446 gen-0 collections per call at 100K).
- **`np.multithreading`** — opt-in parallel 1-D dot (1M float dot 172 → 60 µs, ~2× NumPy's default). Off by default; bitwise-identical summation order when off.

## 8. Performance (engine, reductions, shifts, sort, casts)

| Op | Improvement (NPY/NS, >1 = NumSharp faster) |
|---|---|
| Axis reductions, narrow ints | **Widening SIMD** (int16→int32 accum etc.): `sum(int16, axis=1)` 1058 ms → 2.7 ms (**389×**); also fixes a uint32 axis-sum **corruption** bug |
| `mean`/`var`/`std` (axis) | mean **217×**, var/std 21×; `count_nonzero` 20× |
| Flat `min`/`max` (f64/f32) | raw `Avx.Min/Max` drops the JIT's redundant NaN fixup with a separate finite-mask + cold scan: 0.64×/0.69× → **1.55×/1.73×** @100K; broadcast/neg-stride axis min/max SIMD-routed (was 0.07–0.17×) |
| `np.left_shift`/`right_shift` | reworked into first-class binary ufuncs + SIMD (variable-shift `VPSLLV*`/`VPSRAV*` via Tier-3B): common cases **2–4×** NumPy (was 0.05–0.34× scalar); fixes 7 correctness bugs incl. NEP50 promotion, negative-count, bool→int8, sbyte SIMD |
| `np.sort` / `argsort` | radix line-kernel + **insertion-sort fast path** for short lines (fixes a 25–35× short-line regression) + single-pass multi-histogram radix; int8/16 **6–9×**, argsort 1.3–8× on most dtypes |
| `np.cumsum` | reimplemented as NDIter-driven `add.accumulate` with KEEPORDER output (C-src→C, F-src→F); 1.6–11.8× by axis; fixes empty/0-d/size-1 NEP50 widening + negative-axis validation |
| Boolean mask get+set | unified on one NDIter gather/scatter — tall-thin axis-0 select **372.9 ms / 3.86 GB → 9.1 ms / ~0 MB** (~41×) |
| `np.nonzero` | IL SIMD kernel closes an **8–241×** gap |
| Broadcast-reduce | stride-0 axes folded algebraically — `sum(broadcast_to(...))` **~534–700×** faster, bit-exact |
| `sum`/`mean` (float) | bit-exact NumPy **pairwise summation** (matches `np.add.reduce` bit-for-bit; unblocks float32) |
| `np.any`/`np.all` (bool/char) | reinterpret to byte/ushort → integer SIMD path (was 5–12× scalar); fixes a latent AVX2 32-lane mask-overflow bug |
| `np.zeros` | `calloc`/demand-zero — O(1) (10M f64: 14.3 ms → ~0.01 ms) |
| Casts (`astype`) | full 15×8×15 SIMD campaign: **716 → 129 lagging cells of 1,568 comparable (1,439 winning ≥1.0×)** vs NumPy — per-src geomean f16→narrow **3.8×**, f32→narrow **2.0×**, f64→narrow **1.4×** (architecture below) |
| `np.abs` | exact `dtype=` parity (complex magnitude `|z|→f64`, unsigned cast-then-abs, complex128-output rejection); int32 3.7× / uint32 6.5× |

### High-performance type casting

The single largest engineering effort in this release rebuilt the `astype`/copy/retype machinery from the ground up.

- **One unified core.** Every copy, retype, and cast now routes through **`NDIter.CopyAs(dstType, src, order)`** — it resolves `C`/`F`/`A`/`K` via `OrderResolver`, allocates the destination, and fills it through the in-place `Copy` primitive. `NDArray.copy()` and `DefaultEngine.Cast` (astype) both fold into it; the old scalar / `(1,)` / same-type-`Clone` / F-contig-special / `CastCrossType` branch maze and the **2,226-line per-element `Converts.FindConverter` cast loop are deleted**. A `TryCopySameType` fast path fills scalar-broadcast and gap-free contiguous destinations with one typed `InitBlock`/memset/SIMD fill (6–8× for 1-byte, ~2× wider).
- **SIMD campaign over the whole matrix.** A Phase-0 discovery sweep benchmarked all **15 src × 8 layouts × 15 dst** `astype` combinations at 1M against NumPy 2.4.2, producing a lagging-cell worklist that successive waves drove from **716 → 129 of 1,568 comparable cells** (1,439 now win ≥1.0×). The kernels: `float→narrow-int` (`cvtt` + truncating `Narrow`), `float/int→bool` (`!=0` compare), **Half↔ via the Giesen bit-fiddle** (widen/narrow, sNaN-preserving), `complex→int/bool` (real deinterleave), sub-word strided/reversed lane shuffles (`VPSHUFB`/`VPACKUS`), **fused `VPGATHER` whole-array kernels** for strided float→narrow, an IL-emitted scalar cast kernel for the Vector-less dtypes (direct `Converts.ToX`, 0.65 → 1.5–2.6×), and a KEEPORDER same-type copy. Per-src geomeans: f16→narrow **3.8×**, f32→narrow **2.0×**, f64→narrow **1.4×**.
- **Correctness rounds.** Alongside the speed work, multiple rounds of NumPy-parity fixes closed precision-boundary bugs in the `double→int` converters, `ToUInt32(double)` overflow → 0, `DateTime`/`TimeSpan` conversions, and the Half/Complex/char converter paths, and replaced the `IConvertible` constraint with a `Converts<T>` path — all pinned by the cast tier of the differential-fuzz corpus (full 15×15).

**Also in this release:** `np.where` gained an IL-generated AVX2/SSE4.1 (Neon-safe) SIMD kernel — **5.4× over the old call-based kernel and ~3.9× faster than NumPy** at 1M — plus NEP50 weak-scalar type promotion; `np.asanyarray` now accepts **every built-in C# collection** (`List`/`HashSet`/`Queue`/`Stack`/`ImmutableArray`/`LinkedList`/`Memory<T>`/`ArraySegment`/LINQ results, non-generic `IEnumerable`/`IEnumerator` like `ArrayList`, `Tuple`/`ValueTuple`, and mixed `object[]`) with NumPy-like type promotion; and the test suite was migrated to MSTest v3.

### Benchmark harness

Every figure in this release is backed by a committed, reproducible NumSharp-vs-NumPy comparison driven by one entry point (`benchmark/run_benchmark.py`, NumPy 2.4.2 pinned). It runs the BenchmarkDotNet op/dtype/N matrix (`benchmark/NumSharp.Benchmark.CSharp`) — 1K / 100K / 10M elements × all 15 dtypes, ~615 ops per size (1,851 measured cells: ✅ 792 / 🟡 357 / 🟠 177 / 🔴 72) — joined per `(op, dtype, N)` to a warm NumPy process, then appends **five subsystems that fill the axes the matrix can't express**:

| Subsystem | What it isolates | Headline (NPY/NS) |
|---|---|---|
| **iterator** (`benchmark/nditer`) | the `NDIter` machinery itself — construction, traversal, reductions, selection, dtypes, pathologies, dividends — vs `np.nditer`, aspect × cache-tier | 1.18× geomean; build+dispose **3.3× faster than `np.nditer`**; `bcast_reduce` 538× |
| **layout** (`benchmark/layout`) | reduction / copy / elementwise across 8 memory layouts (C/F/T/strided/sliced/negrow/negcol/bcast) — the op matrix is C-contiguous only | at 1M copy ~2–3× & elementwise ~1.2–1.8×; the per-call dispatch tax shows at 100K |
| **operand** (`benchmark/operand`) | 1-D / scalar / mixed-operand (C+F, C+T) / binary-broadcast layouts | ~1.3–2.2× per case |
| **cast** (`benchmark/cast`) | the full 15×15 `astype` src→dst × 8 layouts at 1M (no op-matrix coverage) | 1,439 / 1,568 comparable cells win |
| **fusion** (`benchmark/fusion`) | `np.evaluate` fused single-pass vs unfused np.* chains | several-fold over NumPy (§2) |

Each run writes a committable `benchmark/history/<date>_<sha>/` snapshot (report + every subsystem sheet + cards + a provenance MANIFEST) and repoints a `latest` symlink; a decoupled post-release `benchmark.yml` regenerates and commits it (matrix + iterator/layout/operand/cast figures above are the `2026-06-23` snapshot on an i9-13900K). The reported convention is **NPY/NS** throughout — ratio = NumPy ÷ NumSharp time, **>1 = NumSharp faster**.

## 9. Memory management — ARC + `IDisposable`

- `NDArray` implements **`IDisposable`** backed by **atomic reference counting** on the unmanaged block: CAS-driven `TryAddRef`/`Release`, idempotent `Dispose`, finalizer safety net, immortal non-owning wraps. Views keep parents alive; parent disposal never invalidates live views.
- Hammered by a 15-case lifecycle suite incl. 32-thread × 1,000-op concurrency races and 50-way parallel dispose — zero corruption.
- A tcache-style **size-bucketed buffer pool** with a **1 B – 64 MiB** window (covers both small-N ufunc results and 4M+ outputs); deterministic release plus the pool removes most steady-state GC pressure (`dot` at 100K: 446 collections → 0).
- Native allocations now register **GC memory pressure** (`GC.AddMemoryPressure`/`RemoveMemoryPressure`) so the runtime sees the true unmanaged footprint and collects on time — fixing a runaway-growth bug (a 1M-array loop peaked at 10+ GB → ~54 MB).

## 10. Differential fuzzing vs NumPy

- **~40,000 bit-exact corpus cases** across 25+ JSONL tiers generated from real NumPy 2.4.2 outputs: casts (full 15×15), binary arith (NEP50), div/mod/power, comparisons, unary (incl. float16 + narrow ints), reductions (incl. negative-stride layouts), NaN-aware reductions, cumulative, statistics, logic/extrema, bitwise+shift, where/place, manipulation, matmul, modf multi-output, sorting/searching, `copyto`, the **index oracle** (curated + dtype + the 0-divergence combinatorial-parity gate), SIMD-tail boundaries, operand aliasing, and error-parity.
- **Seeded random fuzzer** with an element-wise shrinker for minimal repros; a metamorphic-invariant tier.
- **CI integration:** the `FuzzMatrix` gate in `build-and-release.yml` + a nightly **fuzz-soak** workflow. Generators live in `test/oracle/`; the committed corpus is replayed with **no Python at test time**.

## 11. Legacy stacks deleted

- `MultiIterator` **deleted**; all callers migrated to `NDIter.Copy` / multi-operand execution.
- `NDIterator` (interface + `NDIterator<T>` + `AsIterator` extensions) **deleted entirely**; production iteration runs through `NDIter` / `NDIterRef` / `GetAtIndex` / `NDFlatIterator`.
- The **Regen template engine** is fully purged: 87 inline `#if _REGEN … #else <generated> #endif` blocks across 35 files collapsed to the generated code with the template DSL preserved as reference comments; all `*.template.cs` / `.tt` / `GenerateCode.ps1` and their dead `<Compile Remove>` csproj guards are gone. `~400` per-dtype `NPTypeCode` switch sites were replaced by a generic `NpFunc` dispatch utility, and dead-code sweeps removed the 24 `[Obsolete(error:true)]` tombstones plus 10 confirmed-dead private methods.

## 12. New primitives — `Char8` and `DateTime64`

Two NumPy-parity primitive types, both adapted from vendored .NET BCL sources under `src/dotnet/` and **standalone conversion-helpers** for now (not yet wired into `NPTypeCode`):

- **`Char8`** — the NumPy `S1` / Python `bytes(1)` equivalent (`readonly struct`, 1 byte): conversions, operators, span helpers, and **100% Python `bytes` API parity** validated against a Python oracle.
- **`DateTime64`** — the NumPy `datetime64` equivalent (`readonly struct` over a `long` tick count): full `long` range and a **`NaT` sentinel** (`long.MinValue`) with IEEE-NaN-style propagation and NumPy comparison semantics (any ordering involving `NaT` is `false`, `!=` is `true`, while `Equals` stays hash-contract-compliant). Implicit interop from `DateTime`/`DateTimeOffset`/`long`, checked conversion back, and `Converts.ToDateTime64`/`ToX(DateTime64)` matching NumPy exactly; calendar arithmetic delegates to `System.DateTime`.

## 13. Examples — trainable MNIST MLP

New `examples/NeuralNetwork.NumSharp`: a 2-layer MLP with a naive and a **fused** implementation (single-`NDIter` bias+ReLU fusion, fused softmax-cross-entropy backward, Adam optimizer). The stride-native GEMM made the old "copy transposed views before `np.dot`" workaround unnecessary; converges to >99% test accuracy in the bundled demo.

## 14. Cross-platform

macOS / Apple-Silicon (ARM64) reduction parity fixed (6 failing tests on `macos-latest`): `maximum`/`minimum` signed-zero ties (ARM `FMAX`/`FMINNM` vs x86 `MAXPS`/`MINPS`) now resolve to the second operand via an explicit strict-compare + `ConditionalSelect`; `negate(+0.0)` is an explicit sign-bit XOR (was `0 - x` → `+0.0` on ARM); narrow-int `sum`/`prod` reductions use exact integer accumulators on ARM (the AVX2-gated widening kernel fell back to a saturating double path). Reproduced under linux/arm64 via QEMU and pinned as a committed parity suite.

## 15. Tests & CI

- **~10,980 passed / 0 failed** on net8.0 + net10.0, with zero regressions.
- **177 formerly-`[OpenBugs]` reproductions promoted into regular CI tests** as their bugs were fixed (each asserts NumPy-2.4.2-correct behavior). Deliberately kept flagged: 7 AVX-512-only and 2 timing-dependent repros.
- New/expanded suites: the differential index oracle + exhaustive get/set parity matrices (basic/fancy/edge/layout/combined/boolean-mask), array-print parity (`np.ArrayPrint.ParityTests`), cumsum parity (54), abs parity (33), shift parity, `np.evaluate`/`out=`/`where=`/`dtype=` parity, NDIter battletests (566), order-support sections, ARC lifecycle, and the macOS/ARM64 signed-zero parity suite.
- CI: the `FuzzMatrix` gate runs on Windows/Ubuntu/macOS; nightly `fuzz-soak.yml`; a decoupled post-release `benchmark.yml` runs the whole NumSharp-vs-NumPy harness (op/dtype/N matrix + the five iterator/layout/operand/cast/fusion subsystems — §8), renders the DocFX benchmark pages, and auto-commits the refreshed report + cards + a committable `benchmark/history/<date>_<sha>/` provenance snapshot (and its `latest` symlink) to master.
- **Known remaining gaps** (checked in as failing-by-design tests rather than ignored): the still-unimplemented NumPy functions `flip`/`fliplr`/`flipud`/`rot90`, `diag`, `gradient`, and `round` (the function form; `np.round_`/`np.around` exist); the benchmark-surfaced slower paths, all tracked in the committed sheets — small-N (~1K) per-call dispatch overhead, the scalar `Half`/`Decimal` element paths (no BCL `Vector<Half>`/`Vector<decimal>`), large-N `np.any` full-scan, comparison→`bool` stores, and fancy gather/scatter; and a handful of iterator/indexing edge cases pinned as `[OpenBugs]`.

---

## Key highlights since 0.40.0

0.60.0 caps a three-release arc that rebuilt NumSharp's compute core from the ground up and aligned it with NumPy 2.x. For anyone upgrading from **0.40.0**, the cumulative picture across the prerelease line:

### 0.41.0 — IL Kernel Generator (the compute-core rewrite, Mar 2026)

- **Runtime IL emission** (`System.Reflection.Emit.DynamicMethod`) replaced the ~600 K-line Regen template engine with ~19 K lines — a **net −533 K lines** — with `Vector128/256/512` SIMD and runtime width detection across every op.
- **NEP50 (NumPy 2.x) type promotion**; single-threaded **deterministic** execution (SIMD in place of `Parallel.*`).
- **35 new functions** — the `nan*` reductions, `cbrt`, `floor_divide`, `left/right_shift`, `cumprod`, `count_nonzero`, `isnan`/`isfinite`/`isinf`/`isclose`, and the `np.comparison` + `np.logical` modules — plus the comparison/bitwise **operators** (`==` … `>=`, `&`/`|`/`^`) implemented for the first time.
- **MatMul 35–100× faster** (cache-blocked SIMD, 20+ GFLOPS); boolean indexing and axis reductions rewritten on SIMD.
- 60+ NumPy-parity bug fixes; **+4,200 tests**; **no breaking changes**.

### 0.50.0 — Long Indexing (>2 GB arrays + the type system, Apr 2026)

- **Int64/long indexing** migrated across `Shape`/`NDArray`/`Storage`/iterators/IL kernels — arrays beyond **2.1 billion elements (>2 GB)** now work; `np.argmax`/`np.nonzero` return `long`. New `UnmanagedSpan<T>` (long-length `Span` parity), `LongIntroSort`, and unmanaged long index buffers.
- **12 type-introspection APIs** — `can_cast`, `promote_types`, `result_type`, `min_scalar_type`, `common_type`, `issubdtype`, `finfo`, `iinfo`, `isreal`/`iscomplex`/`isrealobj`/`iscomplexobj`.
- **NumPy 2.x type system** — `np.arange(10)` → `int64`, `NPTypeHierarchy` (bool **not** under Number), and **0-D scalar arrays** (`np.array(5)` → 0-D).
- **Python container protocol** — `__contains__`/`__len__`/`__iter__`/`__getitem__`/`__setitem__`, plus `tolist()`/`item()`; `np.frombuffer` rewritten to the full NumPy signature (`count`/`offset`/big-endian/view semantics).
- `ValueType`→`object` scalar migration; operator-overload cleanup (−74%); **600+ battle tests**.

### 0.60.0 — nditer (this release — the first stable)

- Full NumPy **`nditer` port** as the execution engine; **`np.evaluate`** fusion (3.2–6.1×); **full advanced-indexing parity** + a differential index oracle; **byte-exact array printing**; **`out=`/`where=`/`dtype=`** ufunc kwargs; **C/F/A/K** memory layout; **36+ new `np.*` APIs** (sort, pad, percentile/quantile, take/put, split, …); **`np.random`** rebuilt on **MT19937** for NumPy seed parity + **24 new distributions**; **stride-native matmul** and an **`astype` SIMD campaign** that both beat NumPy; **ARC** memory management + buffer pool; differential fuzzing vs NumPy; and the legacy iterator stack **and** the Regen engine deleted outright. *(All detailed in the sections above.)*

---

## Breaking changes

| Change | Impact | Migration |
|---|---|---|
| **Raw `int[]`/`long[]` as the sole index is now FANCY** | `nd[new int[]{0,2}]` selects rows 0 and 2 (shape `(2,…)`), not the element at coordinate `(0,2)` | Use `nd.GetData(0, 2)` for coordinate access. `nd[0,2]` (separate ints) is unchanged; `NDArray<T>.this[int[]]` is unchanged |
| **`NDArray.ToString()` format changed** to NumPy `array_str`/`array_repr` | `[0 1 2]` (str) / `array([0, 1, 2], dtype=int64)` (repr) instead of `[0, 1, 2]` | Update any code parsing `ToString()` output; use the typed accessors/`GetData` for values |
| **`np.left_shift`/`right_shift` result dtype** is now `result_type(lhs, rhs)` | `int8 << int32` → `int32` (was `int8`, overflowing); `bool << bool` → `int8` | — (matches NumPy) |
| **`np.fmax`/`fmin` now ignore NaN** | return the finite operand; `maximum`/`minimum` still propagate NaN | — (matches NumPy; fixes a prior correctness bug) |
| **`np.cumsum` of empty / 0-d / size-1 integer input** widens to int64 and never returns 0-d | `cumsum(empty int32)` → fresh `int64`; `cumsum(0-d)` → `(1,)` | — (matches `np.add.accumulate`) |
| **Boolean-mask axis-0/partial set with a 1-D count-length value** now raises `IncorrectShapeException` | was silently "one scalar per selected row" | Use a `(count, 1)` value to fill one value per selected row |
| **Over-indexing with slices** now raises `IndexError` (too many indices) | `A[:, :, :]` on a 2-D array | Drop the extra indices (matches NumPy) |
| **Per-axis / fancy out-of-bounds now raise `IndexError`** | `A[0,4]` on a 3×4, fancy `-7` on a size-6 axis (was wrong value / OOB read) | — (correctness + memory safety) |
| **`np.full` argument order flipped** to `np.full(shape, fill_value, dtype)` | matches NumPy (was `np.full(fill_value, shape, dtype)`) | Swap the first two arguments; `dtype` stays third |
| `bool - bool`, `-bool`, `np.negative(bool)` now throw | Matches NumPy | Use `^` / cast to int first |
| NaN `<=` / `>=` returns `False` | Matches IEEE & NumPy | Use `np.isnan` explicitly |
| `floor_divide`/`mod` divide-by-zero & floored results; `np.negative(uint)` wraps | Matches NumPy | — |
| `np.power(int, negative int)` raises `ValueError` | Matches NumPy | Use float exponents |
| Cast edge cases (overflow/NaN/`complex→bool`/`float→int` truncation); transcendental NEP50 width promotion; `np.clip`/quantile dtype promotion | Return values/dtypes may change | — |
| **`np.random` seed sequences changed** (Knuth subtractive generator → MT19937) | Same seed now yields a different sequence | — (intentional; output is now byte-identical to NumPy 2.4.2 at a given seed) |
| **Integer `np.random` distributions return `int64`** | `poisson`/`binomial`/`geometric`/`hypergeometric`/`zipf`/`logseries`/`negative_binomial` were `double` | — (matches NumPy dtypes) |
| Broadcast views are read-only; broadcasting keeps rank for 1-D `[1]` | Matches NumPy | `.copy()` to write |
| `MultiIterator` **and** `NDIterator` (+ `NDIterator<T>`, `AsIterator`) removed | Public types removed (threw at runtime anyway) | Use `NDIter` / `NDIter.Copy` / `NDFlatIterator` |
| `NDIter`: `MaxOperands=8` and 64-dim limits removed | None (loosening) | — |
| `np.copyto` unwriteable-destination error type corrected | Exception type change | — |

---

*Everything above was validated against NumPy 2.4.2 ground truth — by ~40,000 differential corpus cases, 566 iterator parity scenarios, a 12,000+-case index oracle, ~18,000 array-print fuzz cases, and per-feature battle tests run on actual NumPy output.*

---

**Closes:** #435 #439 #456 #477 #480 #495 #501 #508 #515 #542 #567 #568 #604 #605 #608
