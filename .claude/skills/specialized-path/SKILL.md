---
name: specialized-path
description: >-
  How to discover, validate, build, and honestly verify a SPECIALIZED path for ANY function — a cheaper
  route for a recognizable SUBSET of an operation's inputs, sitting behind the general implementation
  with a gate + fallback so the general case never regresses and semantics stay bit-identical. The
  cheaper route can be a different KERNEL (SIMD vs scalar, gemv/syrk/stencil), a different ALGORITHM
  (hash vs sort), a VIEW instead of a copy (O(1) stride tricks), an EARLY EXIT (all-false / empty /
  k==0), a MEMORY strategy (cpblk, prefetch, write-once), a FUSED pass, or a native BACKEND seam. Use
  this whenever you make an np.* op faster or cheaper, ask "why is <op> slow / can we beat NumPy",
  add a fast path / special-case / view-return / short-circuit, optimize a composition, or decide
  whether a speedup is even reachable. Trigger on: "specialized path", "fast path", "special case",
  "make <op> faster/cheaper", "why is <op> slow", "beat numpy", "return a view instead of a copy",
  "hash vs sort", "gemv/syrk/stencil/fused", "short-circuit / early exit", "degenerate/tall-skinny
  shape", "is this memory-bound", "measure the speedup". Reach for it before hand-writing a special
  case OR claiming a win.
---

# Specialized paths (for any function)

A **specialized path** is a cheaper route for a recognizable SUBSET of an operation's inputs, sitting
behind the general implementation: `if (subset recognized) cheaper(); else general();`. The general
implementation stays correct for everything; the specialized route only has to be correct+cheaper for
the subset it claims, and **must fall back cleanly (general case unchanged) and stay bit-identical in
semantics.** That invariant is the whole game — everything below serves it.

The cycle is **DISCOVER → EXPERIMENT → CREATE → MEASURE → VERIFY.** Most of the value (and the failure
modes) is in EXPERIMENT (prove the win before building) and MEASURE/VERIFY (prove it after). Depth is
in `references/`; this file is the map + the hard-won rules.

## The KINDS of specialized path (recognize which you're building)

The "cheaper route" is not always a SIMD kernel. Each kind below is the same pattern — recognize a
subset, take a cheaper route, gate + fallback — and each has a live NumSharp example:

| kind | recognized subset | cheaper route | NumSharp example |
|------|-------------------|---------------|------------------|
| **kernel swap** | shape / layout / dtype | a tighter compute kernel | gemv/gevm (`N==1`/`M==1`), syrk (`A@A.T`), contiguous stencil (`diff`), SimdFull vs General |
| **algorithm swap** | a data property (cardinality, value-range) | a different algorithm | `unique` hash(int)/sort(float); `isin` table vs sort+searchsorted by value-range; `np.block` concat vs single-alloc by size |
| **view vs copy** | contiguity / stride relationship | an O(1) view, no data movement | `flip`/`rot90` (stride negation), `trim_zeros` (bounding box), reshape-when-contig |
| **early exit / short-circuit** | a structural state | skip the work entirely | `all`/`any` mask early-exit, `where(cond)` all-false prescan, matmul `k==0`→zeros, `diff n==0`→input |
| **memory strategy** | size / layout threshold | a cheaper allocation/access | `cpblk` for contiguous copy, `take` prefetch above 2 MiB footprint, `np.tri` write-once below 64 MiB vs zero-pages above |
| **fused pass** | contiguous / no-cast shape | collapse a composition to one pass | `np.select` fused chain (`TrySelectFused`), `np.evaluate`, `ediff1d` fused subtract |
| **backend seam** | operands a backend implements | a native route, else managed | `IBlasBackend.Try*` (returns false → managed fallback) |

## THE meta-rule: prove the win on the subset BEFORE you build

Not every specialized path is worth it. For **view / early-exit** kinds the win is asymptotic (O(1) vs
O(N)) — obviously worth it, but you still must prove *semantics are identical* (see VERIFY). For
**kernel / algorithm / memory** kinds the win is empirical — you must measure it, and often it only
wins in a *band* (a crossover). The specific pre-check for a compute path is **ceiling analysis**:

You **beat** the reference (NumPy) when the general route is *pathological* for the subset
(strided-transpose GEMM, degenerate length-1 packing, materialize-all-coordinates, quadratic
re-reads). You reach **parity** when the op is *memory-bandwidth bound* and the reference is already at
the floor — no pathology to exploit. Tell them apart by measuring the general route's **GB/s vs the
hardware floor** (`references/discover.md` → "ceiling analysis"). Worked instances from one session:

| path | commit | vs OLD | vs single-thread NumPy | why |
|------|--------|--------|------------------------|-----|
| cov/corrcoef **syrk** | d64f3df5 | 10× | **2.88× (beats)** | general route was a strided-transpose GEMM (553µs) — pathological |
| **gemv/gevm** | 4f666fc8 | 3–17× | ~0.8–1.0× (**parity**) | memory-bound; cblas dgemv already at the floor — no pathology |
| diff/ediff1d **stencil** | 6e94dbcc | 8% + 4× fewer allocs | already faster in isolation | the "slowness" was a benchmark leaked-alloc artifact, not the kernel |

Do NOT conflate "faster than the OLD path" with "faster than NumPy" — that mistake inflated a "6–7×"
claim that was really parity. Report both baselines, explicitly.

## DISCOVER — where the subset hides

Find where the general path handles a recognizable subset suboptimally. The subset can be keyed on:
**shape** (a degenerate/small/tall-skinny dimension), **layout** (contiguity, stride, transpose,
aliasing), **dtype** (SIMD-capable vs not, int vs float), **data property** (cardinality, value-range,
all-false, empty), **size** (a threshold where a different allocation/algorithm wins), or an
**aliasing relationship** (`A@A.T`, `a[1:]` vs `a[:-1]`).

- **Grep for compositions over a general primitive** — that's where structured operands appear:
  `grep -rn "np\.dot(\|np\.matmul(\|\.Sum(\|argwhere\|concatenate\|NDIter" src/NumSharp.Core --include=*.cs | grep -v test`.
- **Profile the composition** to confirm which component dominates — specialize only the dominant one
  (cov's `dot` was 553µs of 762µs; its concat/center were the rest, and at 10M they flip to dominate).
- **Check "already optimal"** so you don't waste effort — `dot(1d,1d)`/`norm(2)` are already at the
  SimdDot floor; `outer` is a broadcast multiply near its floor. Measure the general route's GB/s
  against the hardware floor first.

Recipes + the structural-pattern table: `references/discover.md`.

## EXPERIMENT — validate before building

1. **Prototype the cheaper route (throwaway) and prove it wins on the subset.** A hand kernel raced
   against the general path (`-c Release`, fresh filename); or, for a view/algorithm path, an asymptotic
   argument + a micro-check. Not clearly better → **STOP**.
2. **Find the CROSSOVER → set the gate.** The route usually wins only in a band (cov Gram: `M<=16`,
   loses above; `unique` hash: integers only; `np.tri` write-once: `<64 MiB`). The gate is part of the
   design, not an afterthought — firing outside the winning band is a regression.
3. **Ceiling analysis for compute paths** (memory-bound → parity ceiling; pathological → big win).

## CREATE — integrate cleanly

- **Gate at the op's DISPATCH, not a per-caller hack** — so every consumer benefits (gemv went into the
  matmul dispatch and lifted `matvec`/`inner`/`dot`/`einsum`/`tensordot` at once). Shape it as a
  `Try…()` that returns `null`/`false` on any miss and falls through to the general path.
- **The gate + fallback is the contract.** On ANY unmet condition (dtype out of scope, non-contiguous
  reduction axis, size outside the band, structure not recognized) → general path, byte-for-byte and
  perf-for-perf unchanged. Never widen a fast path to a shape you didn't measure.
- **For KERNEL kinds, stay dtype-dynamic without a per-dtype switch** — a generic helper +
  `GetGenericHelper(name, GetClrType(dt)).CreateDelegate<Kernel>()` (the `GramHelperSameType` /
  `CumSumHelperSameType` shape) or an NPTypeCode-keyed IL emit (the `AdjacentDiff` shape). Reuse
  primitives (`GramDot`, `SimdDot`, `SimpleStrided`). For VIEW/ALGORITHM/EARLY-EXIT kinds it's a plain
  C# branch — no IL.
- **Pointer/stride discipline** (kernel kinds): logical base is
  `(byte*)nd.Address + nd.Shape.offset * elemsize`; gate SIMD on the reduction/inner axis being
  unit-stride. **`nd.shape[i]` is `long`** — cast or use `nd.Shape.dimensions[i]`.

Patterns per kind + snippets: `references/implement.md`.

## MEASURE — honestly (traps that will burn you)

Perf-measurement traps (apply to kernel/algorithm/memory kinds; a view/early-exit path is verified
asymptotically + a micro-check that no hidden copy sneaks in):

- **`dotnet run -c Release` ALWAYS** — Debug taints hand loops ~2× (IL-emitted kernels look normal, so
  Debug mis-ranks candidates).
- **FRESH FILENAME per rebuild** — the runfile cache keys on filename+content; re-running the same
  `bench.cs` after a rebuild reuses the STALE old DLL.
- **git-stash before/after in the SAME session** — the only reliable improvement number; machine regime
  drifts hourly and cross-run ratios lie.
- **Pin EVERY threading backend to 1 — on BOTH sides — or the comparison is meaningless.** NumPy
  multi-threads BLAS by default (an unpinned dgemv hit 171 GB/s = impossible single-core → your kernel
  looks ~5× worse than it is), and the NumSharp OpenBLAS backend threads too (its result BITS depend on
  the thread count, so >1 also breaks byte-parity gates). The full env set, exported **in the shell
  before the process starts** (not via C# `Environment.SetEnvironmentVariable` — that never reaches a
  native `getenv`): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE`. NumSharp-side:
  leave `np.multithreading` OFF (default) and `OpenBlasEngine.Enable(threads: 1)` if the backend is on.
  Full recipe: `references/measure-verify.md` → "Single-thread pinning".
- **`#:property PublishAot=false` in every kernel-touching script** — without it `dotnet run` disables
  `DynamicMethod`, so EVERY IL kernel (`GetPutKernel`, NDIter.Copy, casts, …) returns null and silently
  falls back to a 10–17× slower scalar path — a "SetData bottleneck" that vanishes with the flag. Any
  kernel-vs-scalar conclusion from an AOT script is invalid.
- **SHORT benches, `GC.Collect()` between cases** — long combined benches contaminate later cases ~2×.

Full playbook + harness: `references/measure-verify.md`.

## VERIFY — the specialized route must be SEMANTICALLY IDENTICAL on the subset

Bit-identical *values* and matching *metadata* (dtype, shape, view writeable/read-only + owns-data
flags, NaN/-0 handling) to what the general path produces.

1. **Know what the oracle PINS.** Grep the corpus (`test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl`) for
   the op — usually byte-exact for SMALL sizes, tolerance for large. Design so the pinned regime is
   byte-identical: a SIMD reduction's scalar tail IS a sequential dot for small K, matching the general
   path — that's why gemv/Gram stay green; a returned VIEW must carry the same writeable/broadcast flags
   the copy path would (a `flip` of a read-only broadcast stays read-only).
2. **A reorder/different algorithm is only safe if the pinned regime still matches — VERIFY, don't
   assume.** (Large-K SIMD reorder is fine only because the corpus doesn't pin it; a hash `unique` must
   still return the same SET as the sort path.)
3. **Run the gates:** `dotnet test --filter "TestCategory=FuzzMatrix"` (byte-parity), the op's
   unit/battle tests, and a differential sweep vs NumPy across shapes × dtypes × layouts — contiguous,
   F-contig, transposed, negative-stride, strided, AND the fallback path. See the `oracle` skill.

## Critical gotchas (learned the hard way)

- **The "slowness" may be a benchmark artifact, not the code.** diff/ediff1d were already faster than
  NumPy in isolation; the reported 0.4× was the leaked-output allocator asymmetry (GC lazy-finalize vs
  NumPy eager refcount-free) + a contaminated snapshot. Reproduce the REAL regime before optimizing.
- **Reuse the general kernel for the pinned regime** instead of re-deriving it — gevm just *widened a
  dispatch condition* to route `M==1` to the existing `SimpleStrided`, so the pinned small cases changed
  zero bytes; only the unpinned large path moved.
- **Match the algorithm SELECTION, not just the values.** `unique` hashes ints and sorts floats
  *because NumPy does* — mirroring the selection keeps edge behaviour (order, NaN) aligned for free.
- **`nd.shape[i]` returns `long`** — cast or use `nd.Shape.dimensions[i]` (a real build-error trap).

## References

- `references/discover.md` — where subsets hide (shape/layout/dtype/value-range/size/aliasing), grep
  recipes, profiling a composition, ceiling analysis.
- `references/implement.md` — integration + the gate/fallback contract, per-kind patterns (kernel IL/
  generic-delegate, view stride-tricks, algorithm-swap, early-exit, memory-strategy, backend seam),
  pointer/stride discipline.
- `references/measure-verify.md` — the honest-measurement harness + every trap, and the semantic-parity
  checklist.
- Sibling skills: **`benchmark`** (the official harness + NPY/NS), **`oracle`** (the byte-parity gate).
- Worked commits: `d64f3df5` (cov syrk), `4f666fc8` (gemv/gevm), `6e94dbcc` (diff stencil).
