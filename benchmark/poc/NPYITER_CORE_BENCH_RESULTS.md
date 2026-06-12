# NpyIter core bench — the iterator itself vs NumPy, and standalone

**Date:** 2026-06-12 · **Machine:** i9-13900K (AVX2, 32 threads) · **NumPy:** 2.4.2 · **Branch:** `nditer`
**Scripts:** `npyiter_core_bench.{cs,py}` (same aspect ids both sides; merged below). Supplements run inline (S/G series, reproduced at the bottom).

Unlike `npyiter_parity_poc` (end-to-end op throughput) and `variation_probe` (production `np.*` grid), this bench measures **the iterator machinery itself**: construction across flag configurations, traversal orchestration across chunk profiles, buffering windows, index tracking, the per-element protocol, and small-N pipeline scaling. Every kernel is deliberately trivial and matched to NumPy's loop family (memcpy / scalar-strided / V256-contig) so iterator costs dominate.

## Verdict

**The iterator is at parity or faster than NumPy on every like-for-like aspect.** Construction beats `np.nditer` 1.4–3.7× in every multi-operand configuration. Full-iterator traversal matches NumPy's real ufunc nditer at tiny chunks (10.1 vs 10.3 ns/chunk) and beats it 2× at wide strided chunks. Buffered mixed-dtype, broadcast, transposed, and reduction traversals are 0.64–0.86× NumPy. The remaining daylight is **(a)** ~200 ns of `np.*` glue above the raw iterator at small N, **(b)** NumPy's *stripped raw-copy walker* (`np.copyto`) at 4 ns/chunk vs our chunked-callback 7 ns — which our production whole-array copy route already matches, and **(c)** one hollow flag (GROWINNER, +8.4% on a measurable config). The iterator-reuse measurement (54.7 ns/call) shows a path to small-N dispatch **5–7× under both sides' per-call floors** — a structural lever NumPy cannot reach from Python.

## C — Construction (construct + dispose only, 1K f64 operands)

| id | configuration | NumSharp | NumPy `np.nditer` | NS/NP |
|---|---|--:|--:|--:|
| C1 | 1-op, no flags | 289 ns | 275 ns | 1.05 |
| C2 | 2-op `[a,out]` | 298 ns | 541 ns | **0.55** |
| C3 | 3-op `[a,b,out]` | 308 ns | 622 ns | **0.50** |
| C4 | 3-op EXTERNAL_LOOP | 304 ns | 701 ns | **0.43** |
| C5 | 3-op broadcast `(32,32)+(32,)` | 696 ns | 719 ns | 0.97 |
| C6 | 2-op BUFFERED cast f32→f64, eager fill | 559 ns | 1 420 ns | **0.39** |
| C7 | C6 + DELAY_BUFALLOC | 309 ns | 1 140 ns | **0.27** |
| C8 | 1-op MULTI_INDEX (32,32) | 315 ns | 421 ns | 0.75 |
| C9 | 1-op C_INDEX (32,32) | 317 ns | 426 ns | 0.74 |
| C10 | 3-op full ufunc config (EXL\|BUF\|GROW\|DELAY\|CIO\|ZS) | 343 ns | 1 000 ns | **0.34** |
| C11 | 8-op contig | 385 ns | 1 140 ns | **0.34** |
| C12 | 2-op 4-D contig (coalesce 4 axes) | 379 ns | 662 ns | 0.57 |
| C13 | 2-op strided 2-D view | 327 ns | 621 ns | 0.53 |

Honest caveats: the NumPy rows include one Python object allocation (~150–250 ns — visible as the C1 floor), so the *C-internal* nditer prep is cheaper than these absolutes; conversely NumPy's ufunc layer **skips full nditer construction entirely** for trivial cases (`check_for_trivial_loop`) — its true small-N floor is the H-series (286 ns e2e at N=8).

Construction decomposition (NumSharp standalone):
- Flags are essentially free: C2→C4/C8/C9/C10 spread is 298→343 ns. Per-op cost ≈ 13 ns (C2 298 → C11 8-op 385).
- **Broadcast is the construction outlier: +388 ns over same-shape (C5 696 vs C3 308)** — rank-mismatched operands fall off the same-shape fast path into `np.broadcast_to` (Shape allocation per ctor). NumPy pays only +97 ns for the same case (719 vs 622).
- Eager buffer alloc + first window fill (1000-elem cast) costs 250 ns over DELAY (C6 559 vs C7 309) — the Wave-2.4 pooled buffers working.

## T — Traversal / orchestration (trivial matched kernels)

| id | aspect | NumSharp | NumPy | NS/NP | note |
|---|---|--:|--:|--:|---|
| T1 | copy contig f64 10M (coalesces to 1 chunk) | 4.372 ms | 4.377 ms | 1.00 | both 18+18 GB/s — DRAM roofline |
| T2.4 | copy strided rows, w=4 (524 288 chunks) | 3.929 ms · 7 ns/chunk | 2.234 ms · 4 ns/chunk | 1.76 | **vs raw-copy walker, not nditer** (see S.4) |
| T2g | same via `ExecuteGeneric` (struct kernel) | 3.249 ms · 6 ns/chunk | — | — | delegate tax ≈ 1.3 ns/chunk |
| T2c | same via `NpyIter.Copy` (production route) | 2.335 ms · 4 ns/chunk | 2.234 ms | 1.05 | whole-array kernel ⇒ **parity with their raw walker** |
| T2.16 | w=16 (131 072 chunks) | 2.010 ms | 1.637 ms | 1.23 | |
| T2.64 | w=64 (32 768 chunks) | 1.521 ms | 1.365 ms | 1.11 | |
| T2.256 | w=256 (8 192 chunks) | 1.297 ms | 1.285 ms | 1.01 | |
| T2.1024 | w=1024 (2 048 chunks) | 0.858 ms | 1.039 ms | **0.83** | |
| T2x | w=4 via *Python* nditer chunk loop | — | 93.1 ms · 178 ns/chunk | — | what Python users pay for custom iterator code |
| T3 | copy transposed (1448²) → contig | 1.592 ms | 1.895 ms | **0.84** | K-order resolution |
| T4r | add row-bcast (2000,2000)+(2000,) f32 | 715 µs | 960 µs | **0.74** | stride-0 outer reset |
| T4c | add col-bcast (2000,2000)+(2000,1) f32 | 699 µs | 868 µs | **0.81** | stride-0 inner |
| T5 | buffered cast copy f32→f64 4M | 2.502 ms | 2.259 ms (`copyto`) | 1.11 | NumPy `copyto` casts in ONE pass (direct transfer fn, no buffer round-trip) |
| T5i | — same through NumPy's *buffered nditer* | — | 3.214 ms | **0.78** | like-for-like window machinery: we win |
| T5n | T5 without GROWINNER | 2.525 ms | — | — | **no effect — GROWINNER is hollow (G-series)** |
| T5b | T5 bufferSize=65536 | 2.475 ms | — | — | window size ≈ irrelevant at DRAM sizes |
| T6 | buffered mixed add f32+f64→f64 4M | 3.695 ms | 4.287 ms | **0.86** | ufunc-style windowed buffering |
| T7a | per-element walk (1000²) [coalesced] | 2.5 ns/elem | 39.9 ns/elem | — | NumPy row = Python protocol cost (context) |
| T7b | + C_INDEX | 3.1 ns/elem | 40.3 | — | flat-index tracking +24% |
| T7c | + MULTI_INDEX (2-D walk, no coalesce) | 2.5 ns/elem | 40.3 | — | multi-index walk costs nothing extra |
| T8 | reduce sum f64 10M contig | 3.152 ms | 4.924 ms | **0.64** | `ExecuteReducing` 4-acc V256 vs pairwise |
| T8s | reduce sum f64 1M strided `[::2]` | 240 µs | 282 µs | **0.85** | |

### S — full iterator vs full iterator (strided-row ADD: NumPy pays its real ufunc nditer)

`np.copyto` (T2) is NOT nditer-driven — it uses NumPy's stripped raw-array walker. The honest per-chunk comparison is a strided binary ufunc, which constructs and drives the full nditer on both sides:

| w | NS `ForEach` | NS production `np.add(out=)` | NumPy `np.add(out=)` | ForEach / NP | prod / NP |
|--:|--:|--:|--:|--:|--:|
| 4 | 5.285 ms · 10.1 ns/chunk | 4.625 ms · 8.8 | 5.417 ms · 10.3 | **0.98** | **0.85** |
| 16 | 3.965 ms | 3.038 ms | 3.823 ms | 1.04 | **0.79** |
| 64 | 2.898 ms | 3.048 ms | 3.060 ms | **0.95** | 1.00 |
| 1024 | 1.449 ms | 1.537 ms | 2.837 ms | **0.51** | **0.54** |

**At the tiny-chunk extreme our full iterator is at parity with NumPy's; at wide strided chunks we are 2× faster** (their strided binary loops are scalar; ours gather/SIMD). The production route even beats raw ForEach at w≤16 (its Tier-3B kernel amortizes better). The only machine that does 4 ns/chunk is NumPy's *raw copy walker* — and `NpyIter.Copy` (whole-array kernel, no per-chunk callbacks) matches it exactly.

## H — small-N pipeline scaling (ctor + run + dispose per call vs `np.add(out=)`)

| N | NS raw iterator pipeline | NumPy `np.add(out=)` e2e | NS/NP |
|--:|--:|--:|--:|
| 8 | 308 ns | 286 ns | 1.08 |
| 64 | 317 ns | 300 ns | 1.06 |
| 512 | 355 ns | 384 ns | **0.92** |
| 1 000 (production `np.add(out=)`, H0) | **648 ns** | 429 ns | **1.51** |
| 4 096 | 947 ns | 981 ns | 0.96 |
| 32 768 | 5.64 µs | 5.28 µs | 1.07 |
| 262 144 | 85.5 µs | 83.6 µs | 1.02 |
| 2 097 152 | 1.397 ms | 1.399 ms | 1.00 |
| **HR512: REUSED iterator (Reset+ForEach only)** | **54.7 ns** | n/a | **0.14 vs NP@512** |

Reading:
- The **raw iterator pipeline tracks NumPy's whole ufunc dispatch within ±8% at every size** — the iterator is NOT the small-N problem.
- Interpolating raw at N=1000 ≈ 430–450 ns vs production H0 648 ns: **~200 ns of `np.*` routing glue** sits above the iterator (operand arrays, ladder, validation) — the remaining P15-class gap (NumPy e2e 429 ns).
- **HR512 is the headline: with construction amortized away, dispatch+kernel for 512 f64 adds costs 54.7 ns** — 6.5× under our own per-call number and 7× under NumPy's floor. NumPy cannot reuse a ufunc's iterator across calls from Python; we can (Reset/ResetBasePointers are public, PARALLEL_SAFE/RANGED machinery already exists).

## G — hollow GROWINNER quantified (same-dtype add f64 4M)

| config | time | windows |
|---|--:|--:|
| EXLOOP plain | 3.187 ms | 1 |
| EXLOOP\|BUFFERED\|GROWINNER | 3.454 ms | 512 × 8192 |
| full ufunc config (…\|DELAY\|CIO\|ZS) | 3.393 ms | 512 |

`NpyIter.cs:751` sets the GROWINNER bit; **nothing reads it** — `NpyIterBufferManager.ComputeTransferSize` caps every fill at `BufferSize` unconditionally. NumPy's growinner expands the transfer to the whole remaining iteration when no operand actually needs buffer copies; here that's **+8.4% tax** on any same-dtype iterator constructed with the NumPy-default ufunc config (and worse at cache-resident sizes).

## Findings (bugs / gaps surfaced by this bench)

1. **GROWINNER is declared-but-hollow** (G-series above): flag stored, never consumed; 512 needless windows ⇒ +8.4% on 4M same-dtype buffered traversal. One condition in `ComputeTransferSize` fixes it (grow `cap` to `remaining` when every operand is linear/unbuffered — NumPy's exact rule; buffered-op presence keeps the cap).
2. **Broadcast construction pays +388 ns over same-shape** (C5 vs C3) because rank-mismatched operands take the `np.broadcast_to` path (`NpyIter.cs:480`); the same-shape fast path (`NpyIter.cs:457`) doesn't cover them. NumPy's delta is +97 ns. Right-aligned stride mapping inline would close ~300 ns on every broadcast ufunc call at small N.
3. **Per-chunk overhead is 7 ns (delegate) / 6 ns (struct) vs NumPy's 4 ns raw walker** — three sources, all in the chunked-callback path: managed delegate invocation (~1.3 ns, T2.4 vs T2g), `ExternalLoopNext`'s per-axis-per-op `GetStride(axis,op) * SrcElementSizes[op]` multiply (NumPy stores byte strides — no imul in the hot step), and the `InvokeInner` mask-resolution branch. Against NumPy's *full* nditer (S-series) we're already at parity — this only matters for raw-walker-class workloads, where `NpyIter.Copy`-style whole-array lowering already wins (T2c = 4 ns/chunk).
4. **~200 ns of production glue above the raw iterator at small N** (H0 648 vs raw ≈ 440): the known Wave-2.2/2.3 territory, now isolated numerically from iterator cost (which is competitive on its own).
5. *(non-bug)* C_INDEX tracking costs +0.6 ns/elem on the per-element protocol; MULTI_INDEX costs nothing over the coalesced walk. Window size (8192 vs 65536) is irrelevant at DRAM-bound sizes. T5-class pure cast-copies are better served by direct one-pass transfer functions (NumPy's `copyto` design, our `TryGetStridedCastKernel`) than by the buffer round-trip — production `astype` already does this; only iterator-driven cast traversal pays the 11%.

## How NpyIter becomes the best — prioritized by measured headroom

1. **Iterator reuse / state pooling (small-N killer).** HR512 proves the run-only cost is 54.7 ns; ctor+dispose is ~300 ns of every call, and NumPy's own floor is 286 ns. Pool `NpyIterState` allocations keyed (NOp, NDim) — or cache fully-constructed iterators keyed (dtype, layout signature) for repeated call sites (`np.evaluate` trees, engine inner routes) and re-arm via `ResetBasePointers`. Target: ≤150 ns e2e at N≤1K ⇒ **2–3× under NumPy where NumPy structurally cannot follow** (Python ufuncs can't hold iterators across calls). Extends roadmap Wave 2.3.
2. **Cut the ~200 ns production glue** (H0 648 → ~450 target): operand-array reuse strategy (`_operands` ownership, Wave 2.2 leftover), flag/dtype array interning, ladder shortcuts when `out=` is provided. Combined with (1): production small-N goes from 1.51× behind to decisively ahead.
3. **Implement GROWINNER** (one condition in `ComputeTransferSize`): immediate +8.4% on same-dtype buffered traversal, protects the NumPy-default ufunc config route (Wave 4) from a permanent window tax.
4. **Broadcast-ctor fast path** for rank-mismatch (finding 2): ~300 ns off every small-N broadcast call; C5 696 → ~400 ns class.
5. **Byte-stride axisdata + specialized iternext.** Premultiply per-axis strides to bytes at construction (kill the per-step imul per op per axis in `ExternalLoopNext`/`Advance`/`Goto*` — NumPy's NAD_STRIDES are bytes for exactly this reason); optionally IL-emit iternext specialized per (ndim≤3, nop≤4) like NumPy's `nditer_templ.c` macro instances. Headroom: 7 → ~4–5 ns/chunk on tiny-chunk traversal (T2.4-class -30–40%), S.4 from parity to ~1.3× ahead.
6. **Tiny-chunk lowering:** when post-coalesce inner width is small (≲16) and the layout is regular 2-D, hand the whole iteration to a whole-array kernel instead of per-chunk callbacks — `NpyIter.Copy` already demonstrates 4 ns/chunk this way (T2c). The chunked-callback contract stays for everything else.
7. **Parallel ForEach (Wave 6.2) stays the unmatched dividend:** PARALLEL_SAFE is wired and free; DRAM-bound rows (T1) won't gain, but compute-bound strided/transcendental traversals will scale on 8 P-cores where NumPy never threads its iterator.

What needs **no** work, now proven at the iterator level: construction beats `nditer` across every config (C2–C13); buffered mixed-dtype, broadcast, transposed, reduction traversal all faster (T3–T8s); full-iterator strided traversal at parity-to-2× (S); per-element protocol 2.5 ns/elem with free MULTI_INDEX; contiguous traversal at the DRAM roofline (T1).

## Reproduce

```bash
python benchmark/poc/npyiter_core_bench.py                          # NumPy side
dotnet run -c Release - < benchmark/poc/npyiter_core_bench.cs       # NumSharp side (Release is MANDATORY)
```

S/G supplements were run as inline variants of the same harness (strided-row add 3-op at w∈{4,16,64,1024} both sides; same-dtype 4M add under EXLOOP vs BUFFERED|GROWINNER vs full ufunc config). All aspects correctness-checked in-script before timing; both scripts assert the JIT optimizer is enabled (Debug builds refuse to print).
