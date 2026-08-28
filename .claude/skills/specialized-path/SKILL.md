---
name: specialized-path
description: >-
  How to discover, validate, build, and honestly measure a SPECIALIZED high-performance path — a
  faster route for a recognizable SUBSET of an operation's inputs (a degenerate/symmetric/strided/
  overlapping shape the general kernel handles badly), gated with a fallback so the general case never
  regresses. Use this whenever you're asked to make an np.* op faster, "why is <op> slow / how do we
  beat NumPy", add a fast path / gemv / syrk / stencil / tall-skinny / small-N kernel, optimize a
  composition that internally calls np.dot / np.matmul / a reduction / an iterator, or decide whether
  a speedup is even reachable. Trigger on: "specialized path", "fast path", "make <op> faster", "why
  is <op> slower than numpy", "beat numpy", "gemv/gevm/syrk/gram/stencil", "degenerate shape",
  "tall-skinny", "optimize matmul/reduction for <shape>", "is this memory-bound", "measure the
  speedup". Reach for it before hand-writing a kernel OR claiming a perf win.
---

# Specialized high-performance paths

A **specialized path** is a faster route for a recognizable SUBSET of an operation's inputs, sitting
behind the general kernel: `if (structure matches) fast(); else general();`. The general kernel stays
correct for everything; the specialized path only has to be correct+fast for the shape it claims, and
**must fall back cleanly so the general case never regresses**.

The cycle is **DISCOVER → EXPERIMENT → CREATE → MEASURE → VERIFY**. Do them in order — most of the
value (and most of the failure modes) is in EXPERIMENT and MEASURE, *before* and *after* you write the
kernel. Depth lives in `references/`; this file is the map + the hard-won rules.

## THE meta-rule: not every specialized path beats NumPy — measure to find out

You **beat** NumPy when the general path is *pathological* for the shape (strided-transpose GEMM,
degenerate length-1 packing, quadratic re-reads). You reach **parity** when the op is
*memory-bandwidth bound* and NumPy's reference (cblas/OpenBLAS) is already at the floor — there's no
pathology left to exploit. Tell them apart by measuring the general path's **GB/s vs the hardware
floor** (`references/measure-verify.md` → "ceiling analysis"). Real examples from one session:

| path | commit | vs OLD managed | vs single-thread NumPy | why |
|------|--------|----------------|------------------------|-----|
| cov/corrcoef **syrk** (`A@A.T`) | d64f3df5 | 10× | **2.88× (beats)** | general path was a strided-transpose GEMM (553µs) — pathological |
| **gemv/gevm** (matrix·vector) | 4f666fc8 | 3–17× | ~0.8× gemv / ~1.0–1.85× gevm (**parity**) | memory-bound; cblas dgemv already at the floor — no pathology |
| diff/ediff1d **stencil** (`a[1:]-a[:-1]`) | 6e94dbcc | 8% + 4× fewer allocs | already faster in isolation | the "slowness" was a benchmark leaked-alloc artifact, not the kernel |

Do NOT conflate "faster than the OLD managed path" with "faster than NumPy" — that mistake inflated a
"6-7×" claim that was really parity. Always report both, explicitly.

## DISCOVER — where opportunities hide

1. **Find compositions over a general primitive.** Grep the source for internal callers of the general
   kernel — that's where operands with exploitable structure appear:
   `grep -rn "np\.dot(\|np\.matmul(\|\.Sum(\|NDIter" src/NumSharp.Core --include=*.cs | grep -v test`.
2. **Recognize the structural patterns** (each defeats a general kernel in a specific way):
   - **Degenerate dimension** — `N==1` (gemv), `M==1` (gevm): the general kernel's inner SIMD loop or
     packing collapses to scalar/overhead.
   - **Symmetric** — `A@A.T` (syrk): compute the upper triangle, mirror (half the work).
   - **Tall-skinny / small-M / small-K** — packing/blocking fixed cost dominates the actual FLOPs.
   - **Overlapping/adjacent views** — `a[1:]-a[:-1]` is a *stencil* (one source, overlapping loads), not
     a binary op on two aliased views + slice allocations.
   - **Transposed/strided operand** routed through a general path that materializes a copy or drops to a
     scalar inner loop.
3. **Profile to CONFIRM the dominant cost** before building anything. Break the composition into
   components, time each in `-c Release`. The specialized path only matters if it attacks the *dominant*
   component (cov's dot was 553µs of a 762µs call; its concat/center were the rest).
4. **Check "already optimal" so you don't waste effort.** Measure the general path's GB/s: if it's near
   the memory floor, there's nothing to win. `dot(1d,1d)`/`norm(2)` are already SimdDot; `outer` is a
   broadcast multiply — both near-floor, no opportunity.

Full recipes: `references/discover.md`.

## EXPERIMENT — validate BEFORE you build

1. **Write a THROWAWAY hand kernel and race it** against the general path (`dotnet run -c Release`,
   fresh filename). Not clearly faster → **STOP**, the opportunity isn't there.
2. **Sweep for the CROSSOVER.** The fast path wins only in a regime — find its edge and gate to it
   (cov Gram: M≤16, loses above; gemv: N==1 with the contraction axis contiguous). Everything else
   falls back.
3. **Do ceiling analysis.** Memory-bound + tuned reference = parity ceiling (don't over-invest chasing
   1.5×). Pathological general path = big win available. See `references/discover.md`.

## CREATE — integrate cleanly

- **Integrate at the general primitive's DISPATCH, not a per-function hack** — so every consumer
  benefits. gemv went into `TryMatMulSimd`/`SimdMatMul` dispatch, which lifted `np.matvec`,
  `np.inner`, `np.dot(m,v)`, and transitively `einsum`/`tensordot`/`multi_dot`.
- **dtype-dynamic WITHOUT a per-dtype switch** (project rule): generic helper +
  `GetGenericHelper(name, GetClrType(dt)).CreateDelegate<Kernel>()` (the `CumSumHelperSameType` /
  `GramHelperSameType` pattern), or an NPTypeCode-keyed IL emit (the `AdjacentDiff` pattern using
  `EmitVectorLoad/Operation/Store`). Reuse existing SIMD primitives (`GramDot`, `SimdDot`,
  `SimpleStrided`) rather than re-deriving.
- **GATE + FALLBACK is the contract.** Check the structural condition; on ANY miss return
  `null` / fall through to the general kernel. The general case must be byte-for-byte and
  perf-for-perf unchanged.
- **Pointer discipline:** logical base is `(byte*)nd.Address + nd.Shape.offset * elemsize`; gate the
  SIMD path on the *reduction axis* being contiguous (`stride == 1`), else fall back.

Patterns + snippets: `references/implement.md`.

## MEASURE — honestly (the traps that will burn you)

- **`dotnet run -c Release` ALWAYS** — Debug taints hand-written loops ~2× (IL-emitted kernels look
  normal, so a Debug run mis-ranks the two).
- **FRESH FILENAME per benchmark** — the runfile cache keys on filename+content, so re-running the
  same `bench.cs` after a rebuild reuses the STALE old DLL (made a NEW build read identical to OLD).
- **git-stash before/after in the SAME session** — the only reliable improvement number; machine
  regime drifts and cross-run comparisons lie.
- **Pin NumPy to one thread** — `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. NumPy
  multi-threads BLAS by default (an unpinned dgemv hit 171 GB/s = impossible single-core); the official
  harness pins, so single-thread is the fair comparison.
- **SHORT benches, `GC.Collect()` between cases** — a long combined bench contaminates later cases ~2×
  (regime pressure accumulates; a gevm group measured after a gemv group read 2× slow).
- best-of-N (**min**), warm, correctness-checked before the timed rows.

Full playbook + copy-paste harness: `references/measure-verify.md`.

## VERIFY — parity is non-negotiable

1. **Know what the oracle PINS.** Grep the corpus (`test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl`)
   for the op's dtypes/shapes — usually **byte-exact for SMALL sizes, tolerance (allclose) for large**.
   Design so the pinned regime is byte-identical: a SIMD reduction's scalar tail IS a sequential dot
   for small K, matching the general path — that's why gemv/Gram stay green.
2. **A large-K accumulation reorder (SIMD multi-accumulator) is OK only if the corpus doesn't pin it**
   (managed float GEMM was never byte-exact with cblas at large K anyway) — but VERIFY, don't assume.
3. **Run the gates:** `dotnet test --filter "TestCategory=FuzzMatrix"` (byte-parity), the op's
   unit/battle tests, and a differential sweep vs NumPy across shapes/dtypes/layouts (contiguous + F +
   transposed + strided + the fallback path). See the `oracle` skill for the corpus machinery.

## Critical gotchas (learned the hard way)

- **`nd.shape[i]` returns `long`**, not int — cast, or use `nd.Shape.dimensions[i]`. (Cost me two
  build errors.)
- **The "slowness" may be a benchmark artifact, not the kernel.** diff/ediff1d were already faster than
  NumPy in isolation; the reported 0.4× was the leaked-output allocator asymmetry (GC lazy-finalize vs
  NumPy eager refcount-free) + a contaminated snapshot. Reproduce the real regime (dispose vs
  no-dispose) BEFORE optimizing — see the `benchmark` skill's regime-volatility notes.
- **Symmetry mirroring is bit-exact even for `A@B.T` when the result is mathematically symmetric**
  (cov's weighted Gram): `dot(A[i],B[j]) == dot(A[j],B[i])` term-by-term, so compute the upper triangle
  and mirror — matches NumPy's independent per-entry dots.
- **Reuse the general kernel for the byte-exact regime instead of re-deriving it.** gevm just *widened a
  dispatch condition* to route `M==1` to the existing `SimpleStrided` daxpy — zero new bytes for the
  pinned small cases, only the large (unpinned) path moved off blocked.

## References

- `references/discover.md` — discovery patterns, grep recipes, profiling a composition, crossover +
  ceiling analysis (memory-bound vs pathological).
- `references/implement.md` — integration point, generic-delegate vs IL-emit dtype dispatch, gate +
  fallback, pointer/stride discipline, reusable primitives.
- `references/measure-verify.md` — the honest-measurement harness + every trap, and the parity/gate
  checklist.
- Sibling skills: **`benchmark`** (the official harness + NPY/NS), **`oracle`** (the byte-parity gate).
- Worked commits: `d64f3df5` (cov syrk), `4f666fc8` (gemv/gevm), `6e94dbcc` (diff stencil).
