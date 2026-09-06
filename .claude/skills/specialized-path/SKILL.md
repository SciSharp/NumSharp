---
name: specialized-path
description: >-
  How to discover, validate, build, and honestly verify a SPECIALIZED path for ANY function — a
  cheaper route for a recognizable SUBSET of an operation's inputs, sitting behind the general
  implementation with a gate + fallback so the general case never regresses and semantics stay
  bit-identical. The cheaper route can be a different KERNEL (SIMD vs scalar, gemv/syrk/stencil), a
  different KERNEL CONTRACT (one call per block instead of per row), a REPRESENTATION trick (bit-level
  f16, byte-lane bool, widen-compute-narrow), a different ALGORITHM (hash vs sort, with an
  observed-growth bailout), a VIEW instead of a copy (O(1) stride tricks), an EARLY EXIT (all-false /
  empty / k==0 / provably-constant output), a FIXED-COST BYPASS (skip iterator/broadcast setup for the
  trivial subset), a MEMORY strategy (cpblk, prefetch, write-once, privatized accumulators, retained
  scratch), a FUSED pass, a composition of EXISTING validated kernels, wrapper-glue removal on a
  parity-bound op, or a native BACKEND seam. Use this whenever you make an np.* op faster or cheaper,
  ask "why is <op> slow / can we beat NumPy / why is 1K slow", add a fast path / special-case /
  view-return / short-circuit, optimize a composition, decide whether a speedup is even reachable, or
  need to prove a fast path actually ENGAGED. Trigger on: "specialized path", "fast path", "special
  case", "make <op> faster/cheaper", "why is <op> slow", "beat numpy", "return a view instead of a
  copy", "hash vs sort", "gemv/syrk/stencil/fused", "short-circuit / early exit",
  "degenerate/tall-skinny/narrow shape", "is this memory-bound", "fixed cost / small-N floor",
  "bit-level kernel", "widen-compute-narrow", "floor probe", "did the fast path engage", "measure the
  speedup". Reach for it before hand-writing a special case OR claiming a win.
---

# Specialized paths (for any function)

A **specialized path** is a cheaper route for a recognizable SUBSET of an operation's inputs, sitting
behind the general implementation: `if (subset recognized) cheaper(); else general();`. The general
implementation stays correct for everything; the specialized route only has to be correct+cheaper for
the subset it claims, and **must fall back cleanly (general case unchanged) and stay bit-identical in
semantics.** That invariant is the whole game — everything below serves it.

The cycle is **DISCOVER → EXPERIMENT → CREATE → MEASURE → VERIFY.** Most of the value (and the failure
modes) is in EXPERIMENT (prove the win before building) and MEASURE/VERIFY (prove it after). Depth is
in `references/`; this file is the map + the hard-won rules. **`references/catalog.md` lists every
specialized path that ships in NumSharp today** — by kind, with its file, gate condition, measured
crossover and commit — copy the nearest exemplar rather than inventing a shape.

**Hard rule (Eli): NumSharp kernels are single-threaded.** No `Parallel.For`/`Task`/threads in any
kernel or np.* path, ever — a parallel int min/max that was bit-exact and 5× NumPy was reverted
(`1c709bf0` → `7b8dac64`). Parity with single-core NumPy on a memory-bound op is the ACCEPTED end
state, not a defect to beat with threads. The only exception is the opt-in, off-by-default
`np.multithreading`, which must not be extended.

## The KINDS of specialized path (recognize which you're building)

The "cheaper route" is not always a SIMD kernel. Each kind below is the same pattern — recognize a
subset, take a cheaper route, gate + fallback — and each has a live NumSharp example (files + gates in
`references/catalog.md`):

| kind | recognized subset | cheaper route | NumSharp example |
|------|-------------------|---------------|------------------|
| **kernel swap** | shape / layout / dtype | a tighter compute kernel | gemv/gevm (`N==1`/`M==1`), syrk (`A@A.T`), `diff` stencil, f32 `argmax` tournament, MAXPS skip-NaN `nanmax`, i64→f64 axis widening |
| **kernel contract** | narrow inner axis × many rows | the kernel loops the outer axis ITSELF — prologue + driver once per BLOCK, not per row | `ND2DElementwiseKernel` (+ masked form), flat gather/scatter for `a[idx]` |
| **representation** | a dtype whose op is defined on its BIT PATTERN, or a narrow lane | integer lane algebra on raw bits; byte lanes; widen-compute-narrow | f16 min/max/cmp/clip/round on the `bits ^ (0x8000 \| asr15)` order key, f16 ±×÷ via f32 SIMD, bool `min(v,1)` byte lanes, `sum(bool)` = popcount, Half sum in an f32 shadow |
| **algorithm swap** | a data property (cardinality, value range) — possibly only OBSERVABLE while running | a different algorithm, or a mid-run bailout | `unique` hash(int)/radix(float) + bailout to radix at 2^17 distinct; `isin` table vs searchsorted by value range; `np.block` concat vs single-alloc; `kron` direct vs tile |
| **compose existing kernels** | a subset two already-validated kernels serve together | route through them — zero new SIMD IL, zero fuzz risk | `bool<<scalar` = `astype` + the SIMD shift kernel ≤ 262144 elems; gevm = a widened `SimpleStrided` condition |
| **view vs copy** | contiguity / stride relationship | an O(1) view, no data movement | `flip`/`rot90` (stride negation), `trim_zeros` (bounding box), `unstack`, reshape-when-contig |
| **early exit / constant output** | a structural state that decides the result | skip the work, or just allocate the answer | `all`/`any` early exit, `where(cond)` all-false prescan, matmul `k==0`, bool argmax find-first, `bool >> s≠0` → lazy zeros |
| **fixed-cost bypass** | the TRIVIAL subset of a route whose per-call SETUP dominates at small N | skip the iterator / broadcast construction entirely | same-layout copy hoisted above `CreateCopyState`; small single-broadcast → direct SimdChunk (≤ 8192); same-dims skip of `Broadcast` in the `*UfuncInto` routes |
| **memory strategy** | a size / layout / microarchitecture threshold | a cheaper allocation or access pattern | `cpblk` contiguous copy, `take` prefetch above 2 MiB, `np.tri` write-once below 64 MiB, `bincount` privatized accumulators ≤ 2048 bins, `fillZeros:false` for fully-written outputs, LAPACK scratch retention pool |
| **fused pass** | contiguous / no-cast shape | collapse a composition to one pass | `np.select` fused chain, `np.evaluate`, `ediff1d`, the fused bool shift body |
| **wrapper-glue removal** | a PARITY-BOUND op (both sides share a native floor) | strip managed overhead around the kernel | `cond`'s two string-slice parses (1.7 µs of Regex EACH) → `Slice[]`; `MakeGeneric` alias → `AsGeneric`; per-call `long[0]` → `Array.Empty` |
| **backend seam** | operands a backend implements | a native route, else managed | `IBlasBackend.Try*` (false → managed), `ISlidingDotBackend.DotBatch` (dispatch once per call, not per position) |

## THE meta-rule: prove the win on the subset BEFORE you build

Not every specialized path is worth it. For **view / early-exit** kinds the win is asymptotic (O(1) vs
O(N)) — obviously worth it, but you still must prove *semantics are identical* (see VERIFY). For
**kernel / algorithm / memory** kinds the win is empirical — you must measure it, and often it only
wins in a *band* (a crossover). Two pre-checks decide whether there is anything to win:

1. **The floor probe (decisive).** Hand-write the ideal loop for the subset as a raw-pointer routine
   (a static method, `AggressiveOptimization`, no captures) and time production vs floor vs NumPy on
   the same buffers. `floor ≈ production ≈ NumPy` → memory-bound, STOP. `floor ≪ production` → the
   gap is overhead and the win is real: the 2-D block kernel's floor was **1.7–2.1× NumPy** while
   production sat 2× off its own floor (`docs/stale-docs/NDITER_2D_BLOCK_KERNEL.md` §2.2).
2. **Ceiling analysis.** Measure the general route's **GB/s vs the hardware floor**
   (`references/discover.md` §4) and place it in the taxonomy below.

| ceiling | how you recognize it | what is still reachable |
|---------|----------------------|-------------------------|
| **pathological general path** | far below the bandwidth floor: strided-transpose GEMM, length-1 panel packing, all-coordinate materialization, a prologue per row, quadratic re-reads | a big win (cov syrk 10×, gemv 8.6× over old, 2-D block 0.82× → 2.5×) |
| **memory-bound at the floor** | GB/s ≈ per-core DRAM (~36–48 GB/s) or an L2/L3 stream, and NumPy is already there | parity (0.8–1.1×); FMA / unrolling / prefetch measured as no-ops or regressions |
| **shared native floor** | both sides call the SAME BLAS/LAPACK routine | parity; only wrapper glue is removable (`full − engine − raw` decomposition) |
| **allocation floor (small N)** | an undisposed 1K op pays ~0.5–1 µs of NDArray + pool + finalizer vs NumPy's ~0.3 µs refcounted C call | disposed (`using`/NDScope) already BEATS NumPy at 1K; undisposed 1K is unwinnable and sits under the report's 1 µs credibility floor (▫ negligible) |
| **memcpy ceiling** | a byte-moving op at 100K; memcpy on this box = 146 GB/s | 1.5× is physically impossible (bool `&` would need 210 GB/s); ~1.2× is the cap |
| **CRT-parity ceiling** | a transcendental NumPy computes with the scalar CRT (asinh/acosh/atanh, `exp2` on win-amd64, f16 T3 unaries) | parity; a SIMD polynomial would be faster AND diverge — rejected |
| **sort-core gap** | a FLAT ratio at every size (sort/partition 0.2–0.4×; BCL `Array.Sort` alone is 3.4× slower than NumPy's vqsort) | a shared sort-core port — not a specialized path |
| **allocator asymmetry** | wins with `Dispose()`, loses without; NumPy's refcount-prompt warm-block reuse | a pool/ARC concern (pool pacing landed in `160ecbba`), not fixable inside the op |

**Fingerprints — read the size profile before touching code:**

| profile | meaning | lever |
|---------|---------|-------|
| slow small, fast large (0.3× at 1K → 1× at 10M) | per-call FIXED cost (NDIter setup 62–113 ns, Shape/NDArray allocs, Regex slice parse) | fixed-cost bypass / wrapper-glue removal |
| flat gap at every size | algorithmic (the sort core) | not a specialization |
| bad ONLY at 100K | allocator / page-fault regime, the leaked-output asymmetry, or a contaminated snapshot | re-measure idle, min-of-N, `+dispose` vs drop |
| bad ONLY at 10M | a bandwidth wall, or GC-storm / host contamination | isolate the cell, `GC.Collect()` between cases |
| identical µs at 100K and 10M | a lazy short-circuit — commit-OOMs under BDN batching unless disposed | dispose per invocation in the bench |
| "before ≈ after" on a route you just built | the fast path did NOT engage (stale DLL, wrong overload, dead gate) | prove engagement (below) |

Worked instances (all single-thread-pinned, NPY/NS):

| path | commit | vs OLD | vs single-thread NumPy | why |
|------|--------|--------|------------------------|-----|
| cov/corrcoef **syrk** | d64f3df5 | 10× | **2.88× (beats)** | general route was a strided-transpose GEMM (553 µs) — pathological |
| **gemv/gevm** | 4f666fc8 | 3–17× | ~0.8–1.0× (**parity**) | memory-bound; cblas dgemv already at the floor — no pathology |
| diff/ediff1d **stencil** | 6e94dbcc | 8% + 4× fewer allocs | already faster in isolation | the "slowness" was a benchmark leaked-alloc artifact, not the kernel |
| **2-D block kernel** (narrow rows) | af25a746 → ccadeef4 | 2× | 0.82× → **1.4–2.5×** | a prologue + odometer PER ROW; the floor probe proved 1.7–2.1× was reachable |
| fancy `a[idx]` **flat gather** | 09d1fc59 | 5.6–7× | 0.81× → **4.97×** (int32 idx) | the general take kernel (0.75 ns/elem, runtime-size imuls) could NOT beat NumPy's trivial path (0.57); a compile-time-width kernel did |
| f16 **bit-level min/max** | a05b5b1e | — | **6–60×** | the pathology is on NumPy's side: its HALF loops are scalar branchy bit-compares |
| f32 **argmax tournament** | 48f894ea | 7× | 0.15× → **1.06×** | scalar NaN-aware walk → NumPy's own 4×-unrolled port; 10M is the bandwidth ceiling, two "improvements" regressed it |
| bool **bitwise** | — | none possible | 0.6× at 100K, **STOP** | the kernel alone is 1.36× faster than NumPy's WHOLE op (190 GB/s); the rest is the allocation floor |

Do NOT conflate "faster than the OLD path" with "faster than NumPy" — that mistake inflated a "6–7×"
claim that was really parity. Report both baselines, explicitly.

## DISCOVER — where the subset hides

Find where the general path handles a recognizable subset suboptimally. The subset can be keyed on:
**shape** (a degenerate/small/tall-skinny/narrow dimension), **layout** (contiguity, stride,
transpose, aliasing), **dtype** (SIMD-capable vs not, int vs float, a bit-pattern-defined op),
**data property** (cardinality, value range, all-false, empty — sometimes knowable only while
running), **size** (a threshold where a different allocation/algorithm wins), or an **aliasing
relationship** (`A@A.T`, `a[1:]` vs `a[:-1]`).

- **Grep for compositions over a general primitive** — that's where structured operands appear:
  `grep -rn "np\.dot(\|np\.matmul(\|\.Sum(\|argwhere\|concatenate\|NDIter" src/NumSharp.Core --include=*.cs | grep -v test`.
  Also grep for setup-heavy routes reached by trivial inputs (`CreateCopyState`, `MultiNew`,
  `Broadcast(`), string slicing in hot wrappers (`["..., 0"]` = a Regex per call), blanket upcasts
  (`astype(NPTypeCode.Double)` — isclose did 2× the memory work at f32), and `MakeGeneric<`.
- **Profile the composition** to confirm which component dominates — specialize only the dominant one
  (cov's `dot` was 553 µs of 762 µs; at 10M concat + centering dominate instead). For a parity-bound
  op decompose by LAYER: `full − engine = wrapper`, `engine − raw = broadcast/alloc`, `raw = the
  shared floor` (how `cond`/`vdot`/`eig`/`vecmat` went from 0.88–0.93× to parity, `a0a6a439`).
- **Audit engagement before assuming absence.** A dtype-specialized helper may already exist and be
  UNREACHABLE: `ArgMaxBoolHelper` sat behind a gate that rejected Boolean, so bool argmax ran a 24 ms
  full scan for an answer at index ~1 (`d967d99b`). Trace the gate to the helper; add a counter.
- **Read NumPy's own dispatch** (`docs/numpy/PERFORMANCE_EXECUTION_PATHS.md`, `refs/numpy/`) — every
  route NumPy specializes (`small_correlate` ≤ 11, `BOOL_argmax`, `simd_argmax`, the unconditional
  `npyiter_coalesce_axes`) is a subset it already decided was worth it; porting the SELECTION keeps
  edge behaviour aligned for free.
- **Check "already optimal"** — `dot(1d,1d)`/`norm(2)` are at the SimdDot floor; `outer` is a broadcast
  multiply near its write floor. Measure GB/s against the hardware floor first.
- **Probe the regimes a report omitted before accepting its ceiling.** The first 2-D-kernel report
  measured only f64, 2M, w ≥ 4 and ops with a vector body — and hid three LOSSES on the kernel's own
  shapes (w < lanes 0.41×, inner-broadcast 1.15×, `exp` 0.84×). Checklist: sub-vector widths,
  cache-resident sizes, other dtypes (bool/f16/int), ops without a vector body, broadcast shapes,
  non-contiguous inputs, `out=` vs fresh alloc, disposed vs dropped.

Recipes + the structural-pattern table: `references/discover.md`.

## EXPERIMENT — validate before building

1. **Floor probe + ceiling analysis first** (above). Then **prototype the cheaper route (throwaway)
   and prove it wins on the subset** — a hand kernel raced against the general path (`-c Release`,
   fresh filename, `PublishAot=false`, pinned); or, for a view/algorithm path, an asymptotic argument
   + a micro-check. Not clearly better → **STOP**.
2. **Try composition before new IL.** Route the subset through EXISTING validated kernels (astype +
   SIMD shift, a widened dispatch condition) — byte-exact by construction and fuzz-risk-free. Write
   new SIMD IL only when composition cannot reach the floor (the general take kernel could not beat
   NumPy's trivial gather; only a compile-time-width kernel did).
3. **Find the CROSSOVER → set the gate.** The route usually wins only in a band (cov Gram `M<=16`,
   loses above; broadcast-direct wins to ~16–32K, gated at 8192 with margin; bool shift ≤ 262144 =
   an L2-resident 2 MB int64 temp; bincount ≤ 2048 bins = 64 KB of accumulators in L2, a LOSS at
   3072). The gate is part of the design, not an afterthought — firing outside the band is a
   regression.
4. **Prove the prototype ENGAGED.** Watch a kernel-cache count grow (`_innerLoop2DCache.Count`), add
   a debug counter, or check for a distinctive timing signature. Two traps manufacture "no change":
   a stale runfile DLL and a gate wired into only one of two overloads.
5. **Any placement / alignment / prefetch claim needs the interleaved, repeated form.** A one-shot
   "4K-aliasing" win of 1.30× vanished when the two placements were alternated 5× best-of-15.

## CREATE — integrate cleanly

- **Gate at the op's DISPATCH, not a per-caller hack** — so every consumer benefits (gemv went into the
  matmul dispatch and lifted `matvec`/`inner`/`dot`/`einsum`/`tensordot` at once). Shape it as a
  `Try…()` that returns `null`/`false` on any miss and falls through to the general path. Make the
  gate CHEAP and type-independent (`Is2DElementwiseShape` reads flags and strides only; callers build
  operand arrays only after it returns true).
- **The gate + fallback is the contract.** On ANY unmet condition (dtype out of scope, non-contiguous
  reduction axis, size outside the band, structure not recognized) → general path, byte-for-byte and
  perf-for-perf unchanged. Never widen a fast path to a shape you didn't measure.
- **A threshold is a named `const`/`static readonly` with the measured crossover in its comment and an
  env override for tuning** (`DirectBroadcastMaxElements` 8192 / `NS_BROADCAST_DIRECT_MAX`,
  `BoolSimdScalarShiftMaxElements` 262144 / `NS_BOOL_SHIFT_SIMD_MAX`, `WriteOnceMaxBytes`,
  `GramCovMaxVars`, `HashBailoutThreshold`). Sit the cutoff safely BELOW the crossover — regimes drift.
- **Know the dispatch ORDER.** An earlier tier can shadow your path: `np.add(arr, 0-d)` rides Tier 3B
  (`TryExecuteBinaryOpViaNDIter`) BEFORE the MixedType gate, so the f16 arith kernels had to make Tier
  3B decline `(Half,Half)`. And wire EVERY entry point — the masked ufunc routes call the PACKED-key
  `ExecuteElementWise` overload; a gate on the string-key overload alone is never reached.
- **Widening a gate onto an older path AUDITS that path.** The direct SimdChunk route had latent bugs
  masked while NDIter was primary (NEP50 mixed dtype, kron-style double broadcast left outputs
  UNWRITTEN, trailing-size-1 strided views, f16 specials) — caught only by fuzz + the full suite; the
  gate now excludes each (`leftShape.IsBroadcasted ^ rightShape.IsBroadcasted`, `IsContiguousCorF`,
  `CanUseSimdBinary`).
- **For KERNEL kinds, stay dtype-dynamic without a per-dtype switch** — a generic helper +
  `GetGenericHelper(name, GetClrType(dt)).CreateDelegate<Kernel>()` (the `GramHelperSameType` shape) or
  an NPTypeCode-keyed IL emit (the `AdjacentDiff` shape); reuse the SAME emit bodies as the general
  kernel wherever possible (the 2-D block kernel is byte-identical BY CONSTRUCTION because it wraps the
  per-chunk kernel's own scalar/vector bodies). The kernel-cache key must carry every specialization
  (`copyKind`, `prefetch`, `idx32`) or the cache serves the wrong variant. For VIEW / ALGORITHM /
  EARLY-EXIT kinds it's a plain C# branch — no IL.
- **Lifecycle hygiene is part of the path.** Dispose a temp the instant the kernel finishes (the
  widened astype temp in the bool shift is what keeps small-N at parity), mark entries `[NDScoped]`
  (helpers reached only under a scope: `[NDScopedCovered]`), wrap a fresh typed result with
  `AsGeneric<T>() ?? throw` (a `MakeGeneric` alias is a second storage allocation), and use
  `fillZeros:false` ONLY when the kernel provably writes 100% of the output — the analyzer (NDW012)
  and the oracle's raw-allocation chokepoint gate both fire on violations.
- **Pointer/stride discipline** (kernel kinds): logical base is
  `(byte*)nd.Address + nd.Shape.offset * elemsize` — right for BOTH a contiguous slice (re-seated
  `Storage.Address`, offset 0) and a strided view (base address, non-zero offset); gate SIMD on the
  reduction/inner axis being unit-stride. **`nd.shape[i]` is `long`** — cast or use
  `nd.Shape.dimensions[i]`. Never `np.empty(view.Shape)` for an output: it inherits the view's strides.
- **JIT rules:** `[MethodImpl(AggressiveOptimization)]` on leaf hot loops (tier-0 leaves ran 12×
  slow inside short LAPACK calls), NEVER on IL emitters (one-shot, already tier-1 output) and NOT on
  inline-hot helpers (it blocks inlining — correlate went 2× slower). `[SkipLocalsInit]` where a
  per-row `stackalloc` is zeroed. Generic hot compares take POINTERS (`LtPtr<T>`), never by-value
  `T` (the `&a` spill made the block partition 3× slower).

Patterns per kind + snippets + the SIMD-emission traps: `references/implement.md`.

## MEASURE — honestly (traps that will burn you)

Perf-measurement traps (apply to kernel/algorithm/memory kinds; a view/early-exit path is verified
asymptotically + a micro-check that no hidden copy sneaks in). The reusable harness is
`benchmark/nditer/probes/` (`*_probe.cs` + `numpy_twins.py <twin>` + `join`), which already encodes
every rule below:

- **`dotnet run -c Release` ALWAYS** — Debug taints hand loops ~2× (IL-emitted kernels look normal, so
  Debug mis-ranks candidates). The probes assert `IsJITOptimizerDisabled == false` on Core and refuse.
- **`#:property PublishAot=false` in every kernel-touching script** — without it `DynamicMethod` is
  disabled, EVERY IL kernel returns null and silently falls back 10–17× slower — a fake "SetData
  bottleneck". Any kernel-vs-scalar conclusion from an AOT script is invalid.
- **`DOTNET_TC_CallCountingDelayMs=0`** — tiered compilation promotes to tier-1 only after a 100 ms
  window with no new tier-0 JIT; a bench keeps jitting DynamicMethods, so the FIRST rows of a process
  are timed at tier-0 (`less(out=)` n=1: 484 ns default vs 124 with the flag). Timed loops go in a
  `static class` method — script local functions/capturing lambdas stay tier-0 (56–70 vs 186 GB/s).
- **Pin BOTH sides to one P-core** (`NS_PROBE_AFFINITY=0x4` here) and never run them concurrently —
  an unpinned thread on this hybrid host reads 2–3× slower UNIFORMLY and swings run to run (411 µs
  pinned vs 800–2600 unpinned). Uniform slowness across every row is the tell.
- **FRESH FILENAME per rebuild** — the runfile cache keys on filename+content; re-running the same
  `bench.cs` after a rebuild reuses the STALE old DLL (NEW read byte-identical to OLD).
- **Never A/B the live tree.** Both sides in detached worktrees (`git worktree add --detach
  <scratch>/pre <sha>`), interleaved old→new→old→new, min of ≥ 2 rounds per side. Machine regime
  drifts hourly (allocator-heavy cells swing 1.7–4×); cross-run ratios lie.
- **Pin EVERY threading backend to 1 — on BOTH sides — or the comparison is meaningless.** NumPy
  multi-threads BLAS by default (an unpinned dgemv hit 171 GB/s = impossible single-core), and the
  NumSharp OpenBLAS backend threads too (its result BITS depend on the count, so >1 also breaks
  byte-parity gates). Export in the SHELL before the process starts (C# `SetEnvironmentVariable` never
  reaches a native `getenv`): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE`;
  NumSharp-side leave `np.multithreading` OFF and `OpenBlasEngine.Enable(threads: 1)` if the backend is on.
- **Min-of-N on BOTH sides, same session, same load.** A single standalone NumPy reading can be a
  fluke (`left_shift(bool)` 100K read 0.2025 ms once, 0.0425 robust — a fake 3× "win"); a stale
  reference from a quieter run faked a 0.61× Half regression that was really 2.3× in NumSharp's favour.
- **Compare kernels on `out=` cells or at 100K.** NumPy's 1M allocating cells page-fault a fresh 8 MB
  result per call while our pool does not — those cells FLATTER NumSharp.
- **SHORT benches, `GC.Collect()` between cases** — a 26-case run contaminated later cases ~2×; four
  dtype×op variants at 10M in one loop GC-stormed a phantom 4223 µs argmin (1766 isolated).
- **Probe-script traps:** `/` on an integer NDArray is TRUE division (build masks with
  `np.floor_divide` — the first `where=` numbers compared different masks); `"label " + nd` binds to
  `NDArray.op_Addition` (broadcast error whose left shape is the string length); `GetDouble` on a
  float32 array reinterprets bytes — use the matching accessor.

Full playbook + harness: `references/measure-verify.md`.

## VERIFY — the specialized route must be SEMANTICALLY IDENTICAL on the subset

Bit-identical *values* and matching *metadata* (dtype, shape, view writeable/read-only + owns-data
flags, NaN sign/payload/-0 handling, error type AND text, partial-write behaviour) to what the general
path produces.

1. **Know what the oracle PINS.** Grep the corpus (`test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl`) for
   the op — usually byte-exact for SMALL sizes, tolerance for large. Design so the pinned regime is
   byte-identical: a SIMD reduction's scalar tail IS a sequential dot for small K, matching the general
   path — that's why gemv/Gram stay green; a returned VIEW must carry the same writeable/broadcast flags
   the copy path would (a `flip` of a read-only broadcast stays read-only). The oracle does NOT
   enumerate widths 1–40, the ≥ 2^17-distinct bailout regime, or the boundary of your threshold —
   those need their own tests.
2. **A reorder/different algorithm is only safe if the pinned regime still matches — VERIFY, don't
   assume.** A 64-bit-int → double sum can ROUND, so the widening axis kernel keeps innermost/pinned
   cases on the exact scalar helper (`wideSeqOnly`) and streams only the leading-axis case.
3. **Self-consistency probe for a kernel change:** every strided result byte-equal to the same call on
   contiguous copies AND to an independent composition (`np.where` for masks, `np.take` for gathers)
   across widths × dtypes × layouts (the 2-D kernel's 16,831-check probe: 0 mismatches). When the
   domain is small, sweep it EXHAUSTIVELY (all 65,536 f16 patterns; all 2³² f32 inputs for the ported
   kernels).
4. **Pin the gate itself:** a test on each side of the threshold (`Tril_AboveTheWriteOnceThreshold_…`
   crosses 64 MiB; the bool shift is bit-exact across its boundary), the bailout regime
   (`Unique_HighCardinality_RadixBailout`), and the FALLBACK (env knob forcing all sizes, or a
   non-eligible layout). Pin route semantics beyond values: verbatim error text and ORDER, no partial
   write on a bad index (NumPy validates BEFORE storing — pre-scan), view-offset conventions
   (`Indexing.KernelRoute.Tests`).
5. **NaN is a contract, not noise.** The oracle tokenizes float NaN, so a NaN SIGN/payload regression
   hides there — the `nan` tier (host-gated) and per-family unit tests pin first-NaN-wins, payload
   verbatim, canonical `0x7fc00000`/`0x7ff8…` where NumPy canonicalizes, and ±0 tie sign. Never turn
   a portable cell host-dependent (MSVC FMA contraction / operand order / CRT libm are PINNED classes —
   `Fuzz/README.md` → "Host-dependent values").
6. **Run the gates:** `dotnet test --filter "TestCategory=FuzzMatrix"` (byte-parity), the op's
   unit/battle tests, the full suite with the CI filter, the NDW analyzer build (no new leaks), and a
   differential sweep vs NumPy across shapes × dtypes × layouts — contiguous, F-contig, transposed,
   negative-stride, strided, AND the fallback path. See the `oracle` skill.

## Critical gotchas (learned the hard way)

- **The "slowness" may be a benchmark artifact, not the code.** diff/ediff1d were already faster than
  NumPy in isolation; the reported 0.4× was the leaked-output allocator asymmetry + a contaminated
  snapshot. The committed nditer sheet's `add@10M` 0.33× re-ran alone at 0.93×. Reproduce the REAL
  regime (idle, pinned, min-of-N, `+dispose` and drop) before optimizing.
- **A report's ceiling is only as wide as its probe.** Probe the omitted regimes (DISCOVER) before
  accepting "done" — the v1 2-D kernel report looked finished and was losing on RGB/xyz-width rows.
- **Reuse the general kernel for the pinned regime** instead of re-deriving it — gevm just *widened a
  dispatch condition* to route `M==1` to the existing `SimpleStrided`, so the pinned small cases changed
  zero bytes; only the unpinned large path moved.
- **Match the algorithm SELECTION, not just the values.** `unique` hashes ints and sorts floats
  *because NumPy does* — mirroring the selection keeps edge behaviour (order, NaN) aligned for free;
  the bailout only catches the cardinality NumPy's own hash thrashes on.
- **Memory-bound loops punish cleverness.** On the 10M argmax stream `Sse.Prefetch0` was 2× slower
  (bandwidth contention) and a deferred-NaN accumulator 2.3× slower (loop-carried dependency); an
  in-loop `isMax` branch cost 2.26× — split max/min into separate methods, keep loop-INVARIANT flags
  only. The straight NumPy port is the ceiling.
- **`nd.shape[i]` returns `long`** — cast or use `nd.Shape.dimensions[i]` (a real build-error trap).

## References

- `references/catalog.md` — **every shipped specialized path by kind**: file, gate condition, measured
  crossover, commit. Start here for an exemplar.
- `references/discover.md` — where subsets hide (shape/layout/dtype/value-range/size/aliasing), grep
  recipes, layer decomposition, the floor probe, ceiling analysis, the regimes checklist, engagement audit.
- `references/implement.md` — integration + the gate/fallback contract, per-kind patterns (kernel IL/
  generic-delegate, kernel contract, representation, view stride-tricks, algorithm-swap + bailout,
  early-exit, fixed-cost bypass, memory-strategy, backend seam), gate construction rules, JIT rules,
  SIMD-emission traps, lifecycle hygiene.
- `references/measure-verify.md` — the honest-measurement harness (the probe pattern + env line + the
  worktree A/B protocol) + every trap, and the semantic-parity checklist.
- Design records (moved to `docs/stale-docs/` on 2026-09-04, still the canonical write-ups):
  `NDITER_PERF_DISCOVERY.md` (§3 = 13 measuring traps with evidence), `NDITER_PERF_CONTINUATION.md`
  (protocol + ranked open levers L1–L8), `NDITER_2D_BLOCK_KERNEL.md` (§2.2 floor probe, §11 the review),
  `FLOAT16_DESIGN.md` (three techniques picked by one question), `UNIQUE_DESIGN.md` (engine selection +
  bailout), `GEMM_PARITY.md`. NumPy's own path selection: `docs/numpy/PERFORMANCE_EXECUTION_PATHS.md`.
- Sibling skills: **`benchmark`** (the official harness + NPY/NS), **`oracle`** (the byte-parity gate).
- Worked commits: `d64f3df5` (cov syrk), `4f666fc8` (gemv/gevm), `6e94dbcc` (diff stencil),
  `7bfc27a2` (copy fixed-cost bypass), `15154b00` (NDIter fixed cost), `af25a746`/`ccadeef4`/`09d1fc59`
  (2-D block kernel, coalescing, fancy-index kernels), `a05b5b1e`…`c8babf28` (the seven f16 families),
  `46dbb9c6`/`d967d99b` (bool byte lanes), `48f894ea` (argmax), `8c09dc15` (widening + nanmax),
  `bd655743` (compose-existing bool shift), `5df10897` (unique radix + bailout), `a13238ac` (DiagWrite),
  `fbbda5f0` (isclose result_type), `a0a6a439`/`e89d6d8f`/`a150e4e9` (wrapper glue, scratch pool,
  fillZeros), `160ecbba` (pool pacing), `8a1376ff`/`75a1d873` (block partition).
