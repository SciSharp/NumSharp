# NDIter performance discovery

How to find out where NumSharp's iterator spends its time, reproduce the 2026-09-03 numbers,
prove a change did not regress them, and locate the next win. Everything here is measured on
the same host in the same session (Windows 11, i9-13900K class, AVX2, .NET 10, NumPy 2.4.2
with `OPENBLAS_NUM_THREADS=1`), and every number has a probe you can re-run. Ratios follow the
house convention **NPY/NS = NumPy time ÷ NumSharp time, higher = NumSharp faster**.

Companion material: `benchmark/nditer/probes/` (the probes), `benchmark/nditer/README.md`
(the canonical sheet and its findings ledger), the NDIter section of `.claude/CLAUDE.md`.

---

## 1. TL;DR

The iterator's throughput at 1M–10M elements was already at parity with NumPy. What was not
at parity was a **fixed, size-invariant tax** paid on every NDIter-routed call, plus one
scalar-only kernel on the comparison `out=` route. Six changes (commit `15154b00`) removed most
of the tax:

| Call (n = 1, ns per call)            | Before | After | NumPy 2.4.2 |
|--------------------------------------|-------:|------:|------------:|
| `np.add(a, b, out=o)`                |    411 |   168 |         289 |
| `np.less(a, b, out=ob)`              |    366 |   140 |         300 |
| `np.sqrt(a, out=o)`                  |    338 |   183 |         258 |
| `np.negative(a, out=o)`              |    329 |   153 |         267 |
| `np.positive(a, out=o)`              |    332 |   141 |         257 |
| `np.add(a, b, out=a)` in-place       |    282 |   154 |         570 |
| managed garbage per `add(out=)` call |  552 B | 200 B |             |

| Iterator construction + dispose (ns) | Before | After | `np.nditer(...)` |
|--------------------------------------|-------:|------:|-----------------:|
| 1 operand                            |    146 |    62 |              258 |
| 3 operands, EXTERNAL_LOOP            |    161 |    93 |              698 |
| 3 operands, ufunc flag set           |    197 |   113 |              923 |

Two things a developer must know before trusting any number in this area:

- **The C# benchmark process must run with `DOTNET_TC_CallCountingDelayMs=0`.** Without it
  the first rows of every section are timed at JIT tier-0 and read 2–4× slower than the same
  call one row later (`lessbool@1` 484 ns vs 124 ns). Section 3 has the evidence.
- **The committed 2026-08-29 nditer sheet's 1M/10M elementwise cells are not real.** The
  section re-run alone reads parity where the sheet says 0.23–0.50×. Section 3 has the evidence.

---

## 2. The cost model

Think of every NDIter-driven op as

    time ≈ FIXED  +  rows × PER_CHUNK  +  elements × PER_ELEMENT

- **FIXED** is paid once per call and does not depend on the array: iterator construction
  (broadcast shape, stride fill, coalescing, contiguity flags, allocation of the state),
  route glue (validation, kernel-cache lookup, closures, operand arrays), and a fresh result
  array when the op allocates. It dominates below ~10K elements.
- **PER_CHUNK** is paid once per inner loop the kernel is handed: the outer-odometer advance
  plus the kernel dispatch. A contiguous op is ONE chunk. A `(rows, w)` strided view is
  `rows` chunks, so for narrow rows this term is the whole story.
- **PER_ELEMENT** is the kernel itself. Above the L2 size it is memory bandwidth; NumPy and
  NumSharp are at parity there, and the only lever left is memory placement (section 6).

Measured decomposition of the three-operand construction, before the change, on the 1-D case
that every contiguous ufunc call hits:

| Component of `MultiNew(3 ops, EXTERNAL_LOOP)`                 | ns | Share |
|---------------------------------------------------------------|---:|------:|
| 3 × `NativeMemory.AllocZeroed` + 3 × `Free` (state, dim block, op block) | 78 | 48 % |
| 1 × `AllocZeroed` + `Free` of the same 616 bytes, for comparison | 29 |       |
| everything else (broadcast shape, stride fill, flags, validation loops) | ~83 | 52 % |
| **total**                                                     | **161** |   |

And of the production `out=` route glue around it (before):

| Component                                                          |    ns | Garbage |
|--------------------------------------------------------------------|------:|--------:|
| interpolated kernel-cache key `$"npy_binop_{op}_{l}_{r}_{res}"` + dictionary probe | 45 | 96 B |
| the same probe on a prebuilt key                                   |   9.5 |     0 B |
| `Broadcast(lhs.Shape, rhs.Shape)` + `Clean()` on identical dims    |    58 | 2 Shapes|
| two emit closures                                                  |   3.7 |    ~96 B|
| `new NDArray[] { lhs, rhs, out }`                                  |   8.6 |    48 B |

The per-chunk term, measured on 2M float64 elements as `(524288, 4)` strided rows:

| Driver configuration (ns per chunk)                        | Before | After |
|------------------------------------------------------------|-------:|------:|
| iternext only (delegate-driven, no kernel)                 |    3.1 |   2.9 |
| `ForEach` with an empty kernel (advance + dispatch)        |    5.6 |   4.5 |
| `ForEach` with a 32-byte memcpy kernel                     |    7.8 |   7.4 |
| raw C# loop doing the same 32-byte memcpy per row          |    4.9 |       |
| NumPy `np.positive(sv, out=dst)` on the same view          |    5.9 |       |

---

## 3. Measuring correctly

Every one of these was a wrong number first.

1. **Release only.** `dotnet run -c Release ...` (file-based apps default to Debug, which taints
   hand-written kernels ~2×). The probes assert `IsJITOptimizerDisabled == false` on
   NumSharp.Core and refuse to run otherwise.

2. **`DOTNET_TC_CallCountingDelayMs=0`.** Tiered compilation promotes a method to tier-1 only
   after a 100 ms window in which no NEW method was jitted at tier-0. A benchmark process keeps
   jitting DynamicMethod kernels and first-touched routes for most of a 200 ms row, so the
   window never opens during the first rows and they are timed at tier-0. NumPy has no JIT, so
   this only ever inflates the NumSharp side. `nditer_sheet.py` now sets it for the C# process;
   set it yourself for any hand-run `dotnet run` timing below ~10K elements.

   | `np.less(a, b, out=ob)`, n = 1, first row of the process | ns |
   |----------------------------------------------------------|---:|
   | default runtime settings                                 | 484 |
   | `DOTNET_TieredPGO=0`                                     | 349 |
   | `DOTNET_TC_CallCountingDelayMs=0`                        | 124 |
   | the same call measured as the 5th row of the same process | 130 |
   | `astype@1`, default vs delay 0                           | 755 → 346 |

3. **A fresh filename per rebuild for `dotnet run` scripts.** A file-based app caches its
   build; after rebuilding NumSharp.Core it can keep running the OLD `#:project` output. The
   checked-in probes reference the csproj relatively and are rebuilt when it changes, but if
   in doubt copy the probe to a new name. `--no-build` is only safe when nothing changed.

4. **`#:property PublishAot=false`.** Without it the IL kernel generators are disabled and every
   kernel falls to a scalar fallback, 10–17× slower. Every probe carries it.

5. **Single-threaded on both sides.** `OPENBLAS_NUM_THREADS=1` (and the `OMP_`/`MKL_` twins if
   set) for the NumPy process. NumSharp's kernels are single-threaded by design; the comparison
   is single core vs single core.

6. **NumPy in the same session, same shapes, same policy.** Best-of-rounds, warm, ~150 ms of
   samples per row; take the minimum. Two identical back-to-back runs of the same binary spread
   about ±5 % at 1K–100K, so a 1.05× is noise and a 1.10× is a signal only if it reproduces.

7. **Never benchmark while anything else runs.** A background build on the same host doubled a
   NumPy 1M add from 0.42 ms to 0.71 ms and reversed the sign of a 1.30× effect. The committed
   sheet was taken under such contamination:

   | Committed sheet (2026-08-29) vs the section re-run alone | NumSharp ms | NumPy ms | NPY/NS |
   |----------------------------------------------------------|------------:|---------:|-------:|
   | `add@10M`, committed                                     |       22.08 |     7.36 |  0.33× |
   | `add@10M`, re-run alone                                  |        8.64 |     8.06 |  0.93× |
   | `sqrt@10M`, committed                                    |       28.44 |     6.46 |  0.23× |
   | `sqrt@10M`, re-run alone                                 |        6.64 |     6.96 |  1.05× |
   | `sqrt@1M`, committed (both sides 5× too slow)            |        2.81 |     2.81 |  1.00× |
   | `sqrt@1M`, re-run alone                                  |        0.55 |     0.55 |  1.00× |

8. **Attribute cost to the right layer.** `fixed_cost_probe.cs` section C runs the same
   arithmetic three ways on the same buffers: through `NDIterRef.ForEach`, as a raw pointer
   loop, and through `np.add(out=)`. If all three agree, the iterator is not the cost. At 10M:
   ForEach add 8.25 ms, raw add 8.23, `np.add(out=)` 8.62, NumPy 8.06.

9. **Interleave and repeat before believing a placement effect.** A one-shot probe measured the
   same V256 add on the same 1M buffers at 0.460 ms with the pooled buffers at their natural
   page offsets and 0.355 ms with two of them staggered by 64/128 bytes, which looked like a
   1.30× 4K-aliasing win. Alternating the two placements five times, best-of-15 each, on an idle
   host gives 0.388–0.411 ms natural against 0.383–0.454 ms staggered at 1M and 3.08–3.63 vs
   2.89–3.31 at 4M: no effect outside the noise band. The first measurement ran cold and first.
   Any claim about memory placement, alignment or prefetch needs the interleaved form
   (`angles_probe.cs` section 5 does it that way now).

10. **A/B against a real baseline, not a memory.** Build the previous commit in a worktree and run
   the same probe file from both checkouts back to back:

       git worktree add /tmp/ns-base <baseline-sha>
       cp benchmark/nditer/probes/ab_ops_probe.cs /tmp/ns-base/benchmark/nditer/probes/
       (cd /tmp/ns-base && DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 dotnet run -c Release benchmark/nditer/probes/ab_ops_probe.cs) > before.tsv
       DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 dotnet run -c Release benchmark/nditer/probes/ab_ops_probe.cs > after.tsv
       OPENBLAS_NUM_THREADS=1 python benchmark/nditer/probes/numpy_twins.py ab > numpy.tsv
       python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv
       git worktree remove --force /tmp/ns-base

11. **Pin to one P-core on a hybrid part.** This host has E-cores, and an unpinned benchmark thread
   lands on one often enough that a whole run reads 2–3× slower UNIFORMLY and the same cell swings
   run to run (1M f64 `positive` on 4-wide rows: 800–2600 µs unpinned, 411 µs pinned — the
   first-generation 2-D-kernel numbers were taken unpinned and did not reproduce). Uniform
   slowness across every row of a probe is the tell; a "before" and an "after" taken in different
   scheduling states compare nothing. `narrow_probe.cs` and `numpy_twins.py` honour
   `NS_PROBE_AFFINITY=<hex mask>` (`0x4` = one P-core here); give both sides the same mask and never
   run them concurrently. Cache-resident strided-stream cells (100K) additionally move ±20 % with the
   same binary from stream placement alone — believe only effects well outside that band there, and
   take the min of at least two rounds per side.

---

## 4. The probes

All in `benchmark/nditer/probes/`. Each `.cs` is a .NET 10 file-based app with a relative
`#:project`, so it runs from any checkout; each prints one line per measurement.

| Probe | What it isolates | Twin |
|-------|------------------|------|
| `fixed_cost_probe.cs` | A: construction + dispose vs the allocator floor. B: per-chunk driver on narrow rows. C: 1M/10M through the iterator vs raw. D: production `out=` routes at n = 1/1K/100K with managed bytes per call. E: the individual glue components. | `numpy_twins.py fixed` |
| `ab_ops_probe.cs` | 48 NDIter-routed production ops at 1K and 100K, `id<TAB>ns` rows for the A/B recipe above. | `numpy_twins.py ab`, `numpy_twins.py join` |
| `angles_probe.cs` | The candidate next levers, each as the experiment that measures it (section 6). | `numpy_twins.py angles` |
| `../nditer_bench.cs` via `nditer_sheet.py` | The canonical sheet: 33 families × 5 tiers, construction, chunk width, pathology, dividends. Section-addressable. | built in |

Run any probe as

    DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 dotnet run -c Release benchmark/nditer/probes/<name>.cs

Reading the output: ns are per call unless the line says ms; `[n]` after a chunk row is
ns per chunk; `alloc N B` is managed bytes allocated per call (`GC.GetAllocatedBytesForCurrentThread`
over 20K calls), which is the cheapest leak detector there is: a route that allocates 500 B per
call on a 1K array is spending as long in the allocator as in the kernel.

---

## 5. What changed on 2026-09-03 and where it lives

| # | Lever | Where | Mechanism | How to see it |
|---|-------|-------|-----------|---------------|
| 1 | One recycled native block per iterator | `NDIter.State.cs` (`AttachInlineArena`, `ResetForRecycle`, the carve in `AllocateDimArrays`); `NDIter.cs` (`AllocateStateBlock`, `ReleaseStateBlock`, `StateArenaBytes = 1536`, `StateBlockCacheSlots = 4`) | The header and a 1536-byte arena are one allocation; the dimension/operand arrays are carved from the arena; `Dispose`/`FreeState` park the block in a bounded `[ThreadStatic]` cache after re-zeroing the header and only the carved bytes. Arena overflow (ndim × nop beyond ~8 × 8) falls back to separate blocks; stack states (`CreateCopyState`) have no arena. | `fixed_cost_probe` A: 3-op construction 161 → 93 ns, `cached state blocks` count non-zero after a dispose |
| 2 | Packed kernel-cache key | `Kernels/InnerLoopKernelKey.cs`; `DirectILKernelGenerator.InnerLoop.cs` (`_innerLoopKeyCache`, `TryGetInnerLoop`); `NDIter.Execution.Custom.cs` (packed-key `ExecuteElementWise*`); the nine call sites in `DefaultEngine.{BinaryOp,CompareOp,UnaryOp,UfuncOut}.cs`, `Default.Shift.cs` | A struct front cache over the string-keyed cache; `ToCacheKey()` rebuilds the legacy string byte-exactly on a miss so DynamicMethod names and `GeneratedDelegates.InnerLoopCount` are unchanged. | `fixed_cost_probe` E: 45 ns + 96 B → 9.5 ns + 0 B |
| 3 | Direct EXTERNAL_LOOP advance | `NDIter.Execution.cs` (`ForEach`, `ExecuteGenericMulti`, `ExecuteReducing`); `NDIter.cs` (`ExternalLoopNext`, `IsPlainExternalLoopAdvance`) | The unbuffered EXTERNAL_LOOP driver calls the advancer directly, inlined, with the state's pointer fields hoisted into locals; the delegate path stays for buffered and non-EXLOOP iterators. | `fixed_cost_probe` B: `ForEach+Empty` 5.6 → 4.5 ns/chunk; `ab_ops_probe`: `add(A,row)` at 100K 17.2 → 13.4 µs |
| 4 | SIMD comparison for `out=` | `NDIter.Execution.cs` (`TryExecuteComparison`); `DefaultEngine.UfuncOut.cs` (`ExecuteComparisonUfuncInto`) | Same-dtype inputs into a bool out that is contiguous in iteration order run the whole-array `Vector.Compare` + mask-pack kernel through the iterator's post-coalesce strides; mixed dtypes, cast outs and strided/F outs keep the scalar body. | `fixed_cost_probe` D: `less(out=)` at 100K 40.0 → 9.3 µs (NumPy 11.4) |
| 5 | Same-dims fast path | `DefaultEngine.UfuncOut.cs` (`SameDims`, the three `*UfuncInto` routes) | Identical dims on every operand make the broadcast and the out join identity transforms that cannot raise (`Shape.Equals` compares dims only), so they are skipped. | `fixed_cost_probe` E: 58 ns + 2 Shapes per call gone from `add(out=)` |
| 6 | Eager COPY_IF_OVERLAP temp dispose | `NDIter.cs` (`ResolveWritebacks`); `NDIter.Detach.cs` (`ResolveDetachedWritebacks`) | After write-back the operand slot reverts to the original and the forced-copy temp is disposed instead of dropped to the finalizer. | `fixed_cost_probe` D: `multiply(a,b,out=a)` 282 → 154 ns at n = 1 |

Also in the same commit: `nditer_sheet.py` sets `DOTNET_TC_CallCountingDelayMs=0`; the README
ledger records the resolved items and the contaminated cells; three test classes pin the
mechanisms (`NDIterStateBlockTests`, `InnerLoopKernelKeyTests`, `NDIterComparisonOutRouteTests`).

---

## 6. Results

### 6.1 Construction vs `np.nditer` (official `construction` section, ns)

| Config        | NumSharp | `np.nditer` | NPY/NS | Committed sheet |
|---------------|---------:|------------:|-------:|----------------:|
| 1op           |     57.5 |         258 |  4.5×  | 1.18× |
| 3op_exl       |     81.5 |         698 |  8.6×  | 4.09× |
| ufunc         |     99.6 |         923 |  9.3×  | 3.30× |
| bufcast       |      320 |        1438 |  4.5×  | 3.47× |
| multiindex    |     74.5 |         410 |  5.5×  | 2.05× |
| 8op           |      143 |        1196 |  8.4×  | 5.71× |
| 4d            |      141 |         662 |  4.7×  | 1.84× |
| 8d            |      241 |         768 |  3.2×  | 2.24× |
| strided2d     |       94 |         629 |  6.7×  | 3.18× |
| **geomean**   |          |             | **5.8×** | 2.73× |

### 6.2 Production `out=` routes (`fixed_cost_probe` D; ns per call, µs where marked)

| Op                     | n = 1 before → after (NumPy) | n = 1K before → after (NumPy) | n = 100K before → after (NumPy) |
|------------------------|------------------------------|-------------------------------|---------------------------------|
| `add(a,b,out=o)`       | 411 → 168 (289)              | 510 → 233 (438)               | 25.6 → 25.3 µs (24.3)           |
| `less(a,b,out=ob)`     | 366 → 140 (300)              | 794 → 232 (417)               | 40.0 → 9.3 µs (11.4)            |
| `sqrt(a,out=o)`        | 338 → 183 (258)              | 895 → 741 (860)               | 56.5 → 56.5 µs (56.2)           |
| `negative(a,out=o)`    | 329 → 153 (267)              | 382 → 178 (402)               | 13.3 → 13.3 µs (12.5)           |
| `positive(a,out=o)`    | 332 → 141 (257)              | 533 → 323 (509)               | 20.2 → 20.7 µs (21.7)           |
| `add(a,b)` allocating  | 239 → 252 (320)              | 303 → 336 (503)               | 25.5 → 25.1 µs (42.7)           |
| `add(sa,sb)` strided   | 281 → 292 (315)              | 1024 → 847 (721)              | 48.3 → 49.3 µs (63.6)           |
| `add(a,b1)` broadcast  | 239 → 247 (317)              | 286 → 333 (776)               | 14.4 → 13.1 µs (34.0)           |

The allocating rows did not take the `out=` levers and move within the ±5–10 % run-to-run
spread; they are the control group.

### 6.3 Per-chunk (official `chunkwidth` section, 2M float64 as strided rows of width w, ms)

| w    | NumSharp | NumPy `np.positive` | NPY/NS | Committed sheet |
|------|---------:|--------------------:|-------:|----------------:|
| 4    |     3.68 |                3.02 |  0.82× | 0.48× |
| 16   |     1.98 |                2.25 |  1.13× | 1.02× |
| 64   |     1.41 |                1.66 |  1.17× | 1.00× |
| 256  |     1.37 |                1.78 |  1.30× | 1.23× |
| 1024 |     0.91 |                1.27 |  1.39× | 1.40× |

### 6.4 The broad A/B (`ab_ops_probe`, same file against the previous commit; ns, µs where marked)

| Op | 1K before → after | 100K before → after | NumPy 100K |
|----|-------------------|---------------------|-----------:|
| `less(a,b,out=)`                | 744 → 208 (3.6×)  | 37.1 → 10.1 µs (3.7×) | 11.4 |
| `add(a,b,out=)`                 | 534 → 231 (2.3×)  | 25.2 → 22.5 µs (1.12×) | 23.2 |
| `multiply(a,b,out=a)` in-place  | 464 → 229 (2.0×)  | 11.2 → 10.9 µs | |
| `nditer_chunks<double>`         | 148 → 71 (2.1×)   | 147 → 71 (2.1×) | |
| `nditer<double>` foreach        | 560 → 450 (1.24×) | 43.3 → 41.8 µs | |
| `sqrt(f32, out=f64)`            | 747 → 522 (1.43×) | parity | |
| `sqrt(a, out=)`                 | 862 → 669 (1.29×) | parity | |
| `add(sa,sb)` strided            | 943 → 755 (1.25×) | parity | |
| `less(A,row)` / `less(sa,sb)`   | 1581 → 1388 / 1223 → 1039 | parity | |
| `sum(A, axis=1)`                | 1121 → 928 (1.21×) | 10.5 → 10.1 µs | 16.3 |
| `nanmean` / `nanstd`            | 745 → 642 / 1290 → 1110 (1.16×) | parity | |
| `a[mask] = v`                   | 1110 → 924 (1.20×) | 42.9 → 38.9 µs (1.10×) | |
| `diff(a)`                       | 317 → 287 (1.10×) | 17.5 → 15.5 µs (1.13×) | |
| `add(A,row)` / `add(A,col)`     | parity | 17.2 → 13.4 / 16.6 → 13.3 µs (1.25–1.28×) | 24.8 / 25.7 |
| `add(A.T, A.T)`                 | parity | 14.4 → 12.0 µs (1.20×) | 12.9 |
| `average(a, weights=b)`         | parity | 11.6 → 9.9 µs (1.17×) | |
| `where`, `a[mask]`, `argwhere`, `nonzero`, `sort`, `argsort`, `cumsum`, `nanmax`, `amin` axis, `sum(dtype=)`, strided `copyto`, `At.copy`, `ravel(At)`, `concatenate`, `pad`, `fill` | unchanged (0.95–1.07×) | unchanged | |

The 100K broadcast gains are lever 3: the old delegate path reloaded the state's pointer
fields per operand per axis, about 15 ns per chunk with three operands.

### 6.5 Official `copycast` and `pathology` sections after the change (NumSharp vs NumPy)

| Row | 1 | 1K | 100K | 1M | 10M |
|-----|---|----|------|----|-----|
| `lessbool` (ns / µs / ms) | 136 vs 281 | 213 vs 387 | 8.3 vs 10.4 µs | 0.205 vs 0.217 ms | 5.39 vs 5.55 ms |
| `inplace`                 | 152 vs 570 | 235 vs 400 | 12.4 vs 13.2 µs | 0.236 vs 0.246 ms | 5.57 vs 6.18 ms |
| `astype`                  | 349 vs 284 | 546 vs 591 | 14.7 vs 23.5 µs | 0.180 vs 0.665 ms | 4.92 vs 10.08 ms |
| `flatten`                 | 128 vs 345 | 300 vs 445 | 19.7 vs 175.9 µs | 0.207 vs 0.981 ms | 11.5 vs 13.0 ms |
| `ravelT`                  | 150 vs 305 | 811 vs 1099 | 32.5 vs 55.4 µs | 0.600 vs 1.545 ms | 47.5 vs 52.8 ms |
| `path.zerodim`            | 164 vs 530 ns | | | | |
| `path.overlap_copy`       | 4.38 vs 8.24 ms | | | | |
| `path.forder_out`         | 3.47 vs 4.41 ms | | | | |

`astype@1` is the one cell still behind: it is the fresh-result allocation (section 7, angle 5),
not the cast.

---

## 7. Where the next wins are

Each angle below was measured with `angles_probe.cs` and its NumPy twin on an idle host. They
are ranked by expected payoff; none needs threads. One candidate was checked and rejected:
**4K aliasing of pooled buffers.** Every large pooled buffer (NumPy's too) sits at page offset
64 mod 4096, and a one-shot measurement suggested staggering them was worth 1.30×; the
interleaved, repeated form (section 3, trap 9) shows no effect at 1M or 4M. Do not build a
pool-stagger on that number.

1. **Route the fancy-index operator to the take/put kernels.** `a[idx]` and `a[idx] = v` walk the
   older MapIter-style path; `np.take`/`np.put` have prefetching typed-MOV kernels.

   | 100K float64, 100K random int32 indices | NumSharp | NumPy |
   |-----------------------------------------|---------:|------:|
   | `a[idx]`                                |   252 µs | 168 µs |
   | `np.take(a, idx)`                       |  99.6 µs | 366 µs |
   | `a[idx] = v`                            |   294 µs | 193 µs |
   | `np.put(a, idx, v)`                     |  82.7 µs | 205 µs |

   Dispatching the 1-D integer-index-on-axis-0 case of the indexer to those kernels is
   2.5–3.5× and turns the sheet's largest remaining family loss (0.61× geomean) into ~1.7–2.3×.

2. **A 2-D kernel contract for narrow rows — LANDED (2026-09-03).** Per row the per-chunk route
   pays the driver (~4.5 ns odometer + dispatch) AND the kernel's own prologue (snapshot ptrs,
   then a runtime SIMD-viability dispatch — stride==elemSize checks per operand) on *every* row,
   for a handful of elements of actual work. Routing narrow rows through the BUFFERED window
   machinery does NOT help (the gather/scatter costs as much as it saves — the inner runs are
   already SIMD-able in place):

   | 2M float64 as `(rows, w)` strided rows | w = 4 | w = 16 | w = 64 |
   |----------------------------------------|------:|-------:|-------:|
   | EXTERNAL_LOOP, one kernel call per row | 3.95 ms | 2.27 | 1.92 |
   | BUFFERED windows (gather + one call per 8192) | 4.45 ms | 2.93 | 2.60 |
   | hand-written 2-D loop (the FLOOR)      | 1.71 ms | 1.23 | 1.12 |
   | NumPy `np.positive(sv, out=)`          | 2.99 ms | 1.98 | 1.82 |

   The floor (one prologue, an inner SIMD run, a per-row pointer bump) is ~1.7-2.1× NumPy — the
   whole gap was per-row overhead, not memory traffic. The fix is a kernel variant
   (`ND2DElementwiseKernel`, `DirectILKernelGenerator.InnerLoop2D.cs`) that loops the outer axis
   itself with per-operand outer byte strides, called once per coalesced 2-D block, so neither the
   odometer nor the prologue runs per row. It reuses the SAME scalar/vector emit bodies as the
   per-chunk kernel (byte-identical results). The dispatcher (`NDIterRef.Is2DElementwiseShape` /
   `TryExecute2DElementwise`, wired into the three packed-key `ExecuteElementWise*` entry points)
   gates on: unbuffered, no `where=` mask, EXTERNAL_LOOP, inner axis element-contiguous for every
   operand, all operands the same SIMD-capable dtype. **NumSharp's NDIter does not coalesce the
   outer axes** (a 3-D `x[:, :, :w]` stays NDim=3), so the gate ALSO flattens *mutually contiguous*
   outer axes (`stride[d] == stride[d+1]*shape[d+1]`, true for any trailing-narrow N-D slice) into
   one `(outerCount, outerStride)` and reuses the same 2-D kernel — which additionally routes those
   N-D cases off the pre-existing malformed-output crash in the ForEach 3-D odometer path.

   Two pieces rode along: `positive` had **no** vector body (`CanUseUnarySimd(Positive)` was false),
   so it was scalar even contiguously and could not engage this path — an identity vector body was
   added (`EmitUnaryVectorOperation` Positive branch), which also gives contiguous positive a SIMD
   copy (1.59× NumPy). Comparisons self-exclude (mixed dtype → bool) and already have the Direct
   whole-array strided kernel.

   **Measured (NPY/NS, 2M f64, no-out fresh alloc):** 2-D positive **1.95-2.18×**, sqrt
   **2.11-2.52×**, add **1.82-2.29×**; 3-D trailing-narrow positive **1.66-2.07×**, sqrt
   **1.51-2.26×**, add **1.37-2.17×** — every cell a win, up from the 0.82× (positive w=4) it
   started at. Gates: FuzzMatrix (all tiers, incl. `out_where` across every strided layout) 98/98,
   main suite 14446/0, plus a 700-case 2-D and 123-case N-D self-consistency check (2-D/N-D strided
   result bit-identical to the contiguous-copy result across ops × dtypes × widths × layouts).
   **Second generation (same day, `docs/NDITER_2D_BLOCK_KERNEL.md` §11).** A review of the report
   above with a wider probe (`narrow_probe.cs` + `numpy_twins.py narrow`: widths 1–64 at
   100K/1M/2M, f32/u8/i32, ops without a vector body, the broadcast shapes, 3-D) found three
   regimes where the kernel's own shapes were LOSING to NumPy — sub-vector rows (`sqrt` on
   `m[:, :3]` **0.41×**: the scalar tail per row where NumPy finishes with a masked vector),
   inner-broadcast (`add(A, col)` **1.15×** vs 3.3× for `add(A, row)`), and ops without a vector
   body (`exp` on 4-wide rows **0.84×**) — plus a general iterator defect: NDIter only coalesced
   axes when every operand was contiguous and nothing was broadcast, so a C-contiguous
   `(250000, 4)` array times a 0-d scalar iterated 2-D, one kernel call per 4-element row
   (1.7 ms → 251 µs once fixed). All four closed: NDIter now merges axes in iteration order
   exactly as NumPy's unconditional `npyiter_coalesce_axes` does
   (`NDIterCoalescing.CoalesceAxesIterationOrder`; `a[:, ::2]` is one chunk like NumPy's, and the
   oracle's K1 excuse shrank to the `'A'`-order transposed-3-D case); the kernel gained
   inner-broadcast modes {C,S}/{CC,SC,CS} dispatched once per call, AVX2 masked sub-vector rows
   and tails (`vmaskmov`, integer masked-off lanes filled with 1), a once-per-call row-shape
   dispatch (generic / one-vector rows / masked rows), a scalar-only block for ops without a
   vector body, and a per-BLOCK odometer for non-foldable leading axes. Pinned, min-of-two,
   NPY/NS: `sqrt` w=2/3 0.53/0.41 → **1.58/1.84** (100K) and 0.60/0.52 → 1.78/2.24 (2M);
   `add(A, col)` c=4 1.15 → **6.5**; `multiply(view, 2.0)` 0.48 → **1.97**; `add(view, col)`
   1.63 → **6.0**; `exp` w=4 0.84 → 1.22; `x[::2, :, :4]` positive 0.70 → **2.12**. Gates:
   16,831-check self-consistency 0 mismatches, FuzzMatrix 98/98, main suite 14446/0. Still open:
   copy-class ops at w=2/3 gain only 1.1–1.3× (a half-vector store would add ~1.3× — a per-width
   specialization, not done); hosts without 256-bit AVX2 keep the scalar tail; `where=` and
   buffered-cast routes are untouched. And `np.empty(view.Shape)` inheriting the view's strides
   fabricates a malformed (size-N buffer, addressing-beyond-N strides) output that crashes any
   kernel that honors it — a pre-existing footgun, unrelated to this lever.

3. **SIMD run detection in the `where=` masked driver.** `InvokeInner` finds mask-true runs one
   byte at a time. Short runs are already ahead of NumPy; long runs lose to the scan:

   | `np.add(a, b, out=o, where=m)`, 100K | NumSharp | NumPy |
   |--------------------------------------|---------:|------:|
   | alternating mask (run length 1)      |   247 µs | 288 µs |
   | 64-element blocks                    |  38.5 µs | 26.9 µs |
   | one half-array run                   |  34.4 µs | 21.9 µs |
   | unmasked                             |  23.3 µs | 23.2 µs |

   A `Vector256` compare over 32 mask bytes at a time to locate the next transition brings the
   long-run cases to parity.

4. **The fresh-NDArray floor.** `np.empty(1000)` costs 216 ns and `np.empty(1)` 211 ns against
   NumPy's 183 / 161; `new Shape(1000)` is 5.5 ns of that, so the cost is the storage object,
   the pool take and finalizer registration. About 100 ns on every allocating small op, and the
   reason `astype@1` is 0.81× while every `out=` op is ahead.

5. **Kernel-level gaps, for a kernel pass rather than an iterator pass** (100K, NumSharp vs NumPy):
   `sum(f32, dtype=f64)` 75.6 vs 37.0 µs (buffered widen then reduce; wants a fused
   widen-accumulate), `amin(A, axis=1)` 19.7 vs 14.7 µs, `sqrt(int32)` 110 vs 78 µs (buffered
   int→f64 cast then sqrt; the binary mixed-dtype path shows the fused convert-in-kernel wins).

6. **Iterator-internal trims, ~80 ns on tiny ops.** The remaining 93 ns construction is ~30 ns of
   allocation-free bookkeeping and ~60 ns of validation loops that the plain ufunc flag set never
   needs (mask pairing pre-check, cast validation, the three legacy contiguity flags,
   PARALLEL_SAFE); the ConcurrentDictionary probe (9.5 ns) could be a direct-mapped slot; the
   two emit closures and the operand array (~15 ns) are only needed on a cache miss; and the
   allocating binary route still pays `Broadcast` + `Clean` (55 ns) on same-dims strided inputs.

7. **`np.nditer`'s per-element view.** `it[0]` allocates an NDArray per element (0.16× NumPy).
   One `UnmanagedStorage` method that re-seats every address cache together would make the 0-d
   view allocation-free; the earlier attempt failed because the caches were updated one at a
   time (see the NDIter section of `.claude/CLAUDE.md`).

---

## 8. Gates to run after touching this code

    # the iterator classes (fast, ~1 s)
    dotnet test --no-build -f net10.0 --filter "FullyQualifiedName~Iterators|FullyQualifiedName~NDIter|FullyQualifiedName~nditer|FullyQualifiedName~NDExpr|FullyQualifiedName~Evaluate"
    # the whole suite with the CI filter (~2 min)
    dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
    # the NumPy differential gate — every ufunc out=/where= layout is bit-compared (out_where.jsonl)
    (cd test/NumSharp.Tests.Oracle && dotnet test --no-build -f net10.0 --filter "TestCategory=FuzzMatrix")
    # both targets still build
    dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0

Five `AuditV2_Iterators` tests fail before and after; they are `[OpenBugs]` and excluded by the
CI filter. Then re-run `fixed_cost_probe.cs` and compare against section 6: the numbers that
should not move are the 100K/1M/10M rows (kernel-bound); the ones that show a change are the
n = 1 and 1K rows.
