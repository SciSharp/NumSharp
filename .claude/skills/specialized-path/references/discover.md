# Discover & validate a specialized-path opportunity

## 0. What keys the subset

A specialized path recognizes a subset of inputs. The subset can be keyed on any of these — learn to
spot all of them, not just shape:

- **shape** — a degenerate (`N==1`), small, or tall-skinny dimension.
- **layout** — contiguity, stride, transpose, or an aliasing relationship (`A@A.T`, `a[1:]` vs `a[:-1]`).
- **dtype** — SIMD-capable vs not, integer vs float (drives both kernel and algorithm choice).
- **data property** — cardinality, value-range, all-false mask, empty, `k==0`.
- **size** — a byte threshold where a different allocation/algorithm/access pattern wins.

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

## 2. The recognizable patterns (any kind), and why the general path is suboptimal

| pattern (subset) | example | why the general path is bad | cheaper route (kind) |
|---|---|---|---|
| **degenerate dim** | matrix·vector `(M,K)@(K,1)` | inner SIMD loop over N=1 → scalar; BLIS packs a length-1 panel | SIMD dot per row (kernel: `GemvHelper`) |
| **symmetric** | `A @ A.T` (Gram/syrk) | computes all M² entries; ignores `C[i,j]==C[j,i]` | upper triangle + mirror (kernel: `GramHelperSameType`) |
| **tall-skinny** | `(2,K)@(K,2)`, K huge | tiny M,N → packing fixed cost dominates; transposed 2nd operand strided | pairwise SIMD dot, gated small-M (kernel) |
| **overlapping views** | `a[1:]-a[:-1]` (diff) | binary op on two slice-view NDArrays + NDIter | stencil, one source (kernel: `AdjacentDiff`) |
| **value-range / cardinality** | `unique`, `isin` | one algorithm can't be best for both int-dense and float-sparse | hash vs sort / table vs searchsorted (**algorithm swap**) |
| **stride relationship** | `flip`, `rot90`, `matrix_transpose` | materializing a copy when strides can express it | O(1) stride-negation/permute view (**view vs copy**) |
| **all-coordinate build** | `trim_zeros` | `argwhere` materializes every non-zero coord | separable first/last non-zero (**view/algorithm**) |
| **structural state** | `all`/`any`, `where(cond)`, matmul `k==0` | runs the full loop over a decided outcome | mask early-exit / all-false prescan / zeros (**early exit**) |
| **size threshold** | `np.tri`, `take`, contiguous copy | one memory strategy across all sizes | write-once vs zero-pages / prefetch / `cpblk` (**memory strategy**) |
| **contiguous no-cast** | `np.select`, `np.evaluate` | a multi-pass composition re-reads/re-writes | one fused pass (**fused pass**) |

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

## 4. Ceiling analysis — CAN you even win? (do this before building)

The single most important pre-check. Measure the general path's effective bandwidth and compare to the
hardware floor:

```
GB/s = bytes_touched / time
```

- **General path is PATHOLOGICAL** (far below floor, cache-hostile, packing/strided/quadratic) →
  large win available. cov's `dot(Xc,Xc.T)` did 553µs for a 2×2 result over K=100000 — a strided
  transpose read; a plain Gram did it in 53µs (**10×**). gemv's OLD path packed a length-1 panel:
  188µs→22µs at 512 (**8.6×**).
- **General path is MEMORY-BOUND at the floor** and the reference is tuned (OpenBLAS/cblas) →
  **parity only**. gemv at 1000² reads 8MB from L3 at ~75 GB/s; NumPy's dgemv gets ~83 GB/s. You can
  reach ~0.8–0.9× but not 1.5× — the bytes must move regardless, and FMA won't help (tested: no
  change, it's bandwidth not compute). Don't over-invest.
- **Already at the floor** → no opportunity. `dot(1d,1d)` 1M = 100µs = 80 GB/s = SimdDot at the RAM
  floor; `outer` is a broadcast multiply near its write-bandwidth floor. Skip.

## 5. Validate with a throwaway prototype (before production code)

- **Compute kinds (kernel / algorithm / memory):** race a hand prototype against the general path in a
  `-c Release` script (fresh filename); verify values match, then time both. Not clearly faster across
  the intended band → STOP. cov's hand Gram (39µs vs `np.dot` 553µs) and gemv's `y[i]=dot(A[i],x)`
  (31µs vs 189µs) proved the win before a line of production code. The diff stencil proved the
  *opposite* usefully — only 8% faster in isolation, revealing the real cost was allocation, which
  reframed the task.
- **View / early-exit kinds:** the win is asymptotic (O(1) view vs O(N) copy; skip vs full loop), so
  there's nothing to race — but you MUST prototype the SEMANTICS: does the view carry the right
  writeable/broadcast/owns-data flags? does the short-circuit produce the identical result the full
  path would? That check is the validation (see `measure-verify.md` → parity).

## 6. Find the crossover → set the gate

Sweep the structural parameter; the fast path wins only in a band. Gate to it, fall back outside.

- cov Gram vs `np.dot`, sweeping M (K=100000): M=2 **9.8×**, M=8 2.0×, M=16 1.6×, M=32 **0.87×
  (loses)** — pairwise re-reads exceed cache above ~16 vars → gate `M <= 16`.
- gemv: wins whenever `N==1` and the contraction axis is contiguous in both operands → that IS the
  gate; transposed A / strided x fall through.

The gate is part of the design, not an afterthought — a fast path that fires outside its winning
regime is a regression.
