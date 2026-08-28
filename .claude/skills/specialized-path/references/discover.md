# Discover & validate a specialized-path opportunity

## 1. Find compositions over a general primitive

Specialized-path opportunities live where a function *composes over* a general kernel whose operands
end up with special structure. Grep for internal callers (exclude the primitive's own impl + tests):

```bash
# GEMM consumers
grep -rn "np\.dot(\|np\.matmul(\|\.Matmul\|MultiplyMatrix\|\.dot(" src/NumSharp.Core --include=*.cs \
  | grep -viE "test|benchmark|/np.dot|/np.matmul|SimdMatMul|MatMul.cs"
# reduction / iterator consumers
grep -rn "\.Sum(\|ReduceAdd\|NDIterRef\|\.Reduce(" src/NumSharp.Core --include=*.cs | grep -v test
```

What the last GEMM sweep surfaced (2026-08): `np.cov`/`np.corrcoef` (`dot(Xc, Xc.T)` — syrk),
`np.polyfit` (`inv(dot(lhs.T, lhs))` — column-Gram), `np.linalg.multi_dot`/`matrix_power`/`norm`,
`np.einsum.Contract` (`matmul(left,right)`). Each internal call is a candidate.

## 2. The structural patterns (and why each defeats a general kernel)

| pattern | example | why the general kernel is bad | specialized kernel |
|---------|---------|-------------------------------|--------------------|
| **degenerate dim** | matrix·vector `(M,K)@(K,1)` | inner SIMD loop over N=1 → scalar; BLIS packs a length-1 panel (7/8 wasted) | SIMD dot per output row (`GemvHelper`→`GramDot`) |
| **symmetric** | `A @ A.T` (Gram/syrk) | computes all M² entries; ignores `C[i,j]==C[j,i]` | upper triangle + mirror (`GramHelperSameType`) |
| **tall-skinny** | `(2,K)@(K,2)`, K huge | tiny M,N → blocking/packing fixed cost dominates; transposed 2nd operand is strided | pairwise SIMD dot over K, gated to small M |
| **overlapping views** | `a[1:] - a[:-1]` (diff) | binary op on two *slice-view NDArrays* + NDIter; two operand streams | stencil: one source, overlapping loads (`AdjacentDiff`) |
| **transposed/strided** | `dot(Xc, Xc.T)` 2nd operand | strided access / materialized copy → cache-hostile | read the C-contiguous operand row-wise instead |

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

## 5. Validate with a throwaway hand kernel

Before writing production code, race a hand kernel against the general path. If it's not clearly
faster across the intended regime, STOP.

```bash
# in a -c Release dotnet_run script (fresh filename):
# hand kernel vs np.dot for the target shape, verify values match, then time both
```

cov's hand Gram (39µs) vs `np.dot` (553µs) and gemv's hand `y[i]=dot(A[i],x)` (31µs vs 189µs) both
proved the opportunity before a line of production code. The diff stencil hand kernel proved the
*opposite* usefully — it was only 8% faster in isolation, revealing the real cost was elsewhere
(allocation), which reframed the whole task.

## 6. Find the crossover → set the gate

Sweep the structural parameter; the fast path wins only in a band. Gate to it, fall back outside.

- cov Gram vs `np.dot`, sweeping M (K=100000): M=2 **9.8×**, M=8 2.0×, M=16 1.6×, M=32 **0.87×
  (loses)** — pairwise re-reads exceed cache above ~16 vars → gate `M <= 16`.
- gemv: wins whenever `N==1` and the contraction axis is contiguous in both operands → that IS the
  gate; transposed A / strided x fall through.

The gate is part of the design, not an afterthought — a fast path that fires outside its winning
regime is a regression.
