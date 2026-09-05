# Discover & validate a specialized-path opportunity

## 0. What keys the subset

A specialized path recognizes a subset of inputs. The subset can be keyed on any of these — learn to
spot all of them, not just shape:

- **shape** — a degenerate (`N==1`), small, tall-skinny, or NARROW dimension (a 3-wide inner axis).
- **layout** — contiguity, stride, transpose, or an aliasing relationship (`A@A.T`, `a[1:]` vs `a[:-1]`).
- **dtype** — SIMD-capable vs not, integer vs float, or an op that is defined on the BIT PATTERN
  (f16 ordering, bool truthiness) — the representation kinds.
- **data property** — cardinality, value range, all-false mask, empty, `k==0`. Some are only
  OBSERVABLE while running (distinct count) — that is the bailout pattern.
- **size** — a byte threshold where a different allocation/algorithm/access pattern wins, or where a
  per-call FIXED cost stops mattering.

## 1. Find compositions over a general primitive (kernel/algorithm kinds)

Opportunities cluster where a function *composes over* a general primitive whose operands acquire
special structure. Grep for internal callers (exclude the primitive's own impl + tests):

```bash
grep -rn "np\.dot(\|np\.matmul(\|\.Sum(\|argwhere\|concatenate\|NDIter" src/NumSharp.Core --include=*.cs \
  | grep -viE "test|benchmark|/np.dot|/np.matmul|SimdMatMul|MatMul.cs"
```

The last GEMM sweep surfaced `np.cov`/`np.corrcoef` (`dot(Xc,Xc.T)` — syrk), `np.polyfit`
(`inv(dot(lhs.T,lhs))` — column-Gram), `np.linalg.multi_dot`/`matrix_power`/`norm`, `np.einsum`
(`matmul(left,right)`). Also look for **materialize-then-reduce** (`np.trim_zeros` avoided
`argwhere.min/max`'s all-coordinate materialization via a separable bounding box) and
**copy-when-a-view-would-do** (`flip`/`rot90`).

Four more greps, each of which found a real win:

```bash
# setup-heavy routes reached by trivial inputs → fixed-cost bypass (copy hoist 7bfc27a2, broadcast-direct)
grep -rn "CreateCopyState\|MultiNew(\|Broadcast(" src/NumSharp.Core --include=*.cs | grep -v test
# string slicing in a hot wrapper = Slice.ParseSlices = a Regex per call (~1.7 µs) → Slice[] (a0a6a439)
grep -rnE '\["[^"]*(\.\.\.|:)[^"]*"\]' src/NumSharp.Core/LinearAlgebra src/NumSharp.Core/Statistics --include=*.cs
# blanket upcasts doing 2× the memory work at f32 (isclose fbbda5f0)
grep -rn "astype(NPTypeCode.Double)" src/NumSharp.Core --include=*.cs | grep -v test
# a second storage allocation per typed result (predicates 4dfe7619)
grep -rn "\bMakeGeneric<" src/NumSharp.Core/Backends --include=*.cs
```

## 2. The recognizable patterns (any kind), and why the general path is suboptimal

| pattern (subset) | example | why the general path is bad | cheaper route (kind) |
|---|---|---|---|
| **degenerate dim** | matrix·vector `(M,K)@(K,1)` | inner SIMD loop over N=1 → scalar; BLIS packs a length-1 panel | SIMD dot per row (kernel: `GemvHelper`) |
| **symmetric** | `A @ A.T` (Gram/syrk) | computes all M² entries; ignores `C[i,j]==C[j,i]` | upper triangle + mirror (kernel: `GramHelperSameType`) |
| **tall-skinny** | `(2,K)@(K,2)`, K huge | tiny M,N → packing fixed cost dominates; transposed 2nd operand strided | pairwise SIMD dot, gated small-M (kernel) |
| **narrow rows** | `m[:, :4]`, RGB/xyz triples, `x - mean[:, None]` | the driver's odometer AND the kernel's SIMD-viability prologue run PER ROW for 4 elements of work | the kernel loops the outer axis itself (**kernel contract**: `ND2DElementwiseKernel`) |
| **overlapping views** | `a[1:]-a[:-1]` (diff) | binary op on two slice-view NDArrays + NDIter | stencil, one source (kernel: `AdjacentDiff`) |
| **bit-pattern op** | f16 min/max/cmp/round, bool and/or/xor, `sum(bool)` | converting each lane to float (no `Vector<Half>`) or running a scalar branchy compare; NumPy's HALF loops are scalar | integer lane algebra on raw bits, byte lanes, popcount (**representation**) |
| **narrow-lane arithmetic** | f16 `+ − × ÷`, Half `sum` | a scalar double bridge that also double-rounds differently from NumPy | widen-compute-narrow in f32 SIMD, an f32 shadow accumulator (**representation**) |
| **value-range / cardinality** | `unique`, `isin` | one algorithm can't be best for both int-dense and float-sparse; NumPy's hash thrashes at high cardinality | hash vs sort / table vs searchsorted, plus a bailout when a counter crosses 2^17 (**algorithm swap**) |
| **stride relationship** | `flip`, `rot90`, `matrix_transpose`, `unstack` | materializing a copy when strides can express it | O(1) stride-negation/permute view (**view vs copy**) |
| **all-coordinate build** | `trim_zeros` | `argwhere` materializes every non-zero coord | separable first/last non-zero (**view/algorithm**) |
| **structural state** | `all`/`any`, `where(cond)`, matmul `k==0`, `bool>>s≠0` | runs the full loop over a decided outcome | mask early-exit / all-false prescan / zeros / lazy-zeros (**early exit**) |
| **trivial input, heavy setup** | `copyto` of same-layout arrays, `matrix + row` at 1K | ~62–113 ns of NDIter construction (was 161) + broadcast/Shape allocs on a 9 ns memcpy | bypass the iterator for the trivial subset (**fixed-cost bypass**) |
| **size threshold** | `np.tri`, `take`, contiguous copy, `bincount` | one memory strategy across all sizes | write-once vs zero-pages / prefetch / `cpblk` / privatized accumulators (**memory strategy**) |
| **contiguous no-cast** | `np.select`, `np.evaluate` | a multi-pass composition re-reads/re-writes | one fused pass (**fused pass**) |
| **already-validated neighbours** | `bool << scalar` | a scalar-only route while a bit-exact `astype` and a SIMD shift kernel already exist | compose the two below an L2-sized threshold (**compose existing**) |
| **shared native floor** | `cond`, `vdot`, `eig`, `vecmat` with OpenBLAS | the routine IS NumPy's; the gap is managed glue (Regex slices, params arrays, alias allocs, tier-0 leaves) | strip the glue (**wrapper-glue removal**) |

## 3. Profile the composition to confirm the dominant cost

Never optimize a component that isn't the bottleneck. Break the call into pieces and time each in
Release (see `measure-verify.md` for the harness). Example that decided cov's scope:

```
cov(a,b) 100K whole      762 us
  concat[a,b]            218 us   ← real, but not the GEMM
  average               35 us
  center X-avg          56 us
  np.dot(Xc, Xt) [GEMM] 553 us   ← THE dominant cost → specialize this
```

At 10M the same profile flips (concat 68ms + center 49ms dominate, GEMM only 12ms) — so the
specialized GEMM helps 100K enormously but 10M stays composition-bound. **The dominant component moves
with size; profile at the size you care about.**

**Decompose by LAYER when the op is parity-bound** (the OpenBLAS wrapper method, `a0a6a439`): a
friend-assembly script (`AssemblyName=NumSharp.DotNetRunScript` + signed with `Open.snk`) calls the
engine member (`OpenBlasEngine.TryVecmat/TryVdot/TryEig/TrySvd`) and the raw native routine
(`OpenBlasNative.Dgemv/Ddot`) directly, best-of-many warm:

```
full − engine  = the np.* wrapper (validation, slicing, allocs)   ← removable
engine − raw   = broadcast / result alloc / linearize             ← partly removable
raw            = the shared native floor NumPy also pays          ← the ceiling
```

Interleave a noisy pair (`full − svdvals` back to back) so the native noise cancels.

**Attribute cost to the right layer** with a three-way probe (`benchmark/nditer/probes/fixed_cost_probe.cs`
section C): the same arithmetic through `NDIterRef.ForEach`, as a raw pointer loop, and through
`np.add(out=)`. If all three agree (10M: 8.25 / 8.23 / 8.62 ms, NumPy 8.06) the iterator is not the cost.

**Separate the kernel from the per-call floor** with the `+dispose` vs drop pair: time the op with
`using`/`Dispose()` on its result AND with the result dropped (`GC.KeepAlive`, BDN-faithful). If the
disposed form beats NumPy and the dropped form does not, the gap is the leaked-output allocator
asymmetry (GC lazy-finalize vs NumPy's refcount-prompt reuse) — not the kernel. That is how diff,
bool bitwise and the f32 predicates were diagnosed; calling the kernel directly with a reused output
(bool AND: 1.573 µs @100K = 190 GB/s, faster than NumPy's whole 2.14 µs op) closes the case.

## 4. Ceiling analysis — CAN you even win? (do this before building)

**The floor probe is the decisive measurement.** Hand-write the ideal loop for the subset as a raw
pointer routine in a `static class` (the probes' `Floors.W4(...)` etc. in
`benchmark/nditer/probes/narrow_probe.cs` section F): one prologue, the inner SIMD run, a per-row
pointer bump — nothing else. Time production vs floor vs NumPy on the SAME buffers:

| 2M f64 `(rows, w)` | floor | production (before) | NumPy | floor vs NumPy |
|---|---:|---:|---:|---:|
| w=4 | 1.705 ms | 3.629 | 2.994 | 1.76× |
| w=64 | 1.120 | 1.792 | 2.005 | 1.79× |

Floor ≈ production ≈ NumPy → memory-bound, STOP. Floor ≪ production → the gap is overhead, the win is
real and its size is known BEFORE any production code exists.

Then place the op in the ceiling taxonomy by measuring the general path's effective bandwidth:

```
GB/s = bytes_touched / time
```

- **General path is PATHOLOGICAL** (far below floor, cache-hostile, packing/strided/quadratic, a
  prologue per row) → large win available. cov's `dot(Xc,Xc.T)` did 553µs for a 2×2 result over
  K=100000 — a strided transpose read; a plain Gram did it in 53µs (**10×**). gemv's OLD path packed a
  length-1 panel: 188µs→22µs at 512 (**8.6×**). NumPy's own HALF loops are scalar bit-compares, so f16
  min/max on raw keys is **6–60×**.
- **General path is MEMORY-BOUND at the floor** and the reference is tuned (OpenBLAS/cblas, or a
  streaming ufunc) → **parity only**. gemv at 1000² reads 8MB from L3 at ~75 GB/s; NumPy's dgemv gets
  ~83 GB/s. Per-core DRAM on this host is ~36–48 GB/s (16 accumulators, prefetch: no change). You can
  reach ~0.8–1.1× but not 1.5× — the bytes must move regardless, FMA won't help (tested).
- **Shared native floor** (both sides call the same LAPACK/BLAS) → parity; only the wrapper glue is
  yours (layer decomposition above).
- **Allocation floor** (undisposed 1K: ~0.5–1 µs of NDArray/pool/finalizer vs NumPy's ~0.3 µs) → the
  op is under the report's 1 µs credibility floor and marked ▫ negligible; disposed already beats NumPy.
- **memcpy ceiling**: this box copies at 146 GB/s, so a 100K byte-moving op (bool `&` moves 300–400 KB)
  caps at ~1.2× — 1.5× would need 210 GB/s.
- **CRT-parity ceiling**: an op NumPy computes through the scalar CRT (asinh/acosh/atanh, `exp2` on
  win-amd64, the f16 transcendentals) is bit-exact ONLY through the same scalar calls; parity is the
  correct ceiling and a SIMD polynomial is a divergence, not a win.
- **Already at the floor** → no opportunity. `dot(1d,1d)` 1M = 100µs = 80 GB/s = SimdDot at the RAM
  floor; `outer` is a broadcast multiply near its write-bandwidth floor. Skip.

## 5. Validate with a throwaway prototype (before production code)

- **Compose first.** If two already-validated kernels serve the subset (a bit-exact `astype` + an
  existing SIMD kernel; a dispatch condition widened onto an existing strided kernel), prototype THAT —
  it is byte-exact by construction and carries no fuzz risk. Write new SIMD IL only when composition
  cannot reach the floor (the general take kernel at 0.75 ns/elem could not beat NumPy's 0.57 trivial
  gather; a compile-time-width flat kernel at 45 µs vs 57 did).
- **Compute kinds (kernel / algorithm / memory):** race a hand prototype against the general path in a
  `-c Release` script (fresh filename, `PublishAot=false`, pinned); verify values match, then time both.
  Not clearly faster across the intended band → STOP. cov's hand Gram (39µs vs `np.dot` 553µs) and
  gemv's `y[i]=dot(A[i],x)` (31µs vs 189µs) proved the win before a line of production code. The diff
  stencil proved the *opposite* usefully — only 8% faster in isolation, revealing the real cost was
  allocation, which reframed the task.
- **View / early-exit kinds:** the win is asymptotic (O(1) view vs O(N) copy; skip vs full loop), so
  there's nothing to race — but you MUST prototype the SEMANTICS: does the view carry the right
  writeable/broadcast/owns-data flags? does the short-circuit produce the identical result the full
  path would? That check is the validation (see `measure-verify.md` → parity).
- **Prove the prototype ENGAGED.** A kernel-cache count that grows (`_innerLoop2DCache.Count`), a debug
  counter, or a distinctive signature (the 2-D floor probe read "2× off floor" once — the stale runfile
  DLL had never engaged the kernel). "Before ≈ after" means not-engaged until proven otherwise.

## 6. Find the crossover → set the gate

Sweep the structural parameter; the fast path wins only in a band. Gate to it, fall back outside, and
sit the cutoff BELOW the measured crossover — regimes drift hour to hour.

- cov Gram vs `np.dot`, sweeping M (K=100000): M=2 **9.8×**, M=8 2.0×, M=16 1.6×, M=32 **0.87×
  (loses)** — pairwise re-reads exceed cache above ~16 vars → gate `M <= 16`.
- gemv: wins whenever `N==1` and the contraction axis is contiguous in both operands → that IS the
  gate; transposed A / strided x fall through.
- broadcast-direct: the direct SimdChunk beats NDIter up to ~16–32K elements → gated at 8192 with
  margin (`DirectBroadcastMaxElements`, env-tunable).
- bool `<< scalar`: `astype(int64)` + SIMD shift wins while the temp is L2-resident → 262144 elements
  (2 MB); above it the two passes over an 80 MB temp lose to the fused scalar route.
- bincount privatized accumulators: 4 lanes × bins × 8 B must fit L2 → ≤ 2048 bins (a wash at 2048, a
  LOSS at 3072), and `len ≥ 4·bins`, `len ≥ 4096` to amortize the merge.
- `np.tri` write-once vs zero-pages: 64 MiB (64 MB: 6.0 vs 11.0 ms; 72 MB: 18.9 vs 13.5 reversed).
- `take` prefetch: only above a 2 MiB gathered footprint (below it 1.6× SLOWER at 100K).
- unique hash → radix bailout: 2^17 distinct (the table leaves L2) — a threshold on a counter observed
  DURING the run, because no cheap up-front sample distinguishes 100K-unique-in-1M from all-unique.

The gate is part of the design, not an afterthought — a fast path that fires outside its winning
regime is a regression.

## 7. The regimes a report must cover before you accept its ceiling

The first 2-D block kernel report measured f64, 2M elements, w ≥ 4 and ops with a vector body — and
declared the lever done while the kernel's own target shapes were LOSING (w=3 `sqrt` 0.41×, `add(A,col)`
1.15×, `exp` w=4 0.84×). Before calling a path finished, probe:

- **widths below one vector** (1, 2, 3 lanes) — a scalar tail per row is 6× slower per element;
- **cache-resident sizes** (100K) as well as DRAM-bound (2M–10M) — every kernel converges at DRAM;
- **other dtypes**: bool, f16, int (`np.arange` is int64 — an "int32 argmax" that was really int64
  looked 0.15× and was a false alarm), f32 (a 2× lane-width story of its own);
- **ops without a vector body** (exp/log/power/mod, mixed dtype) — they pay the per-row route too;
- **the broadcast shapes** (`add(A, col)`, `view * scalar`, a 0-d operand) — an iterator that only
  coalesces when nothing broadcasts iterates a `(250000,4)` array one row per kernel call;
- **non-contiguous inputs** (F, transposed, reversed, strided, offset) AND the fallback path;
- **`out=` vs fresh allocation**, **disposed vs dropped** — different floors, both real usage.

## 8. Read NumPy's own dispatch

NumPy already decided which subsets pay: `docs/numpy/PERFORMANCE_EXECUTION_PATHS.md` (its path
selection per op family — ONE kernel per dtype with runtime stride branches, not separate kernels) and
the vendored source under `refs/numpy/`. Examples that became NumSharp paths by porting the SELECTION:
`small_correlate` (n2 ≤ 11), `BOOL_argmax`/memchr `BOOL_argmin`, `simd_argmax`'s tournament,
`npyiter_coalesce_axes` running unconditionally, `_unique_hash` for ints but not floats,
`mapiter_trivial_set` validating every index before its first store. Mirroring the selection keeps
edge behaviour (order, NaN, partial writes) aligned for free; the bailouts are where NumSharp goes
beyond (unique's hash thrash).

## 9. Audit engagement — a helper may exist and be unreachable

`ArgMaxBoolHelper`/`ArgMinBoolHelper` (early-exit block scans) existed but `CanUseReductionSimd`
rejected Boolean, so bool argmax ran the scalar FULL scan: 24 ms on a 10M array whose first True sat at
index ~1 (`d967d99b`). `positive` had no vector body, so it was scalar even contiguously and could not
enter the 2-D block route. When a dtype-specialized helper exists, trace the gate to it and prove the
route with a counter — the same "prove engagement" rule as §5.
