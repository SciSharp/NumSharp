# NpyIter Performance Migration — Handover

**Branch:** `nditer`  ·  **Goal:** make `NpyIter` the single execution driver for every `np.*` op, where every memory layout (contiguous / strided / broadcast / widening / mixed-dtype) hits its performance roofline and matches or beats NumPy 2.x.

This document is written so the next session can resume cold. It records the architecture, what is already done (with commits), the **measured evidence**, the **approaches already proven to fail** (do not re-attempt them), and an extensive per-phase plan describing *how each phase reaches peak performance*.

> **TL;DR of the whole investigation:** NpyIter's SIMD code-generation is *already* NumPy-class (verified: identical to the Direct kernels on contiguous data). What's missing is (a) the ~1 µs setup tax that forces contiguous ops to bypass NpyIter, and (b) two SIMD *coverage holes* — strided inner loops and narrow-int widening — that drop to scalar. Close those and migrate the kernels, and there is one fast path instead of two.

---

## 0. How to use this doc

- Each phase has: **Goal → Peak-performance mechanism → Evidence → Implementation → Files → Expected win → Verification → Risk.**
- The **"Peak-performance mechanism"** sections are the heart of the request: they explain, in roofline terms, *why* the change reaches the hardware limit.
- Re-run the microbenchmarks in [§9 Verification toolkit](#9-verification-toolkit) before/after every change. The cast-matrix and full-suite gates are non-negotiable.
- **Proven-failed approaches are in [§4](#4-proven-facts--hard-won-lessons). Read it first.** Two plausible "fixes" for the strided gap actually lose or wash; only one wins.
- **Cold-start fast path:** §1 (architecture) → §4 (lessons) → §11 (runbook: orient/build/test in ~5 min) → pick a phase from §10. §12 has paste-able bench/gate scripts, §13 the regression bisect, §14 the Tier-3C fusion endgame, §15 a file/symbol index.

---

## 1. Architecture primer

```
np.op(a)
   │
   ├── DefaultEngine.<Op>.cs            ← dispatch / routing per op
   │       │
   │       ├── trivial-contiguous bypass ──► DirectILKernelGenerator   (whole-array kernel)
   │       │                                   ~50 partials in Backends/Kernels/Direct/
   │       │                                   ONE call walks the whole array itself.
   │       │
   │       └── strided / broadcast ──────► NpyIterRef  (per-chunk driver)
   │                                          Backends/Iterators/NpyIter*.cs
   │                                          iterator advances pointers; kernel does one chunk.
   │
   └── (1-D strided unary only) ────────► TryStridedSimdUnaryOp  (fused raw-ptr SIMD, pre-NpyIter)
```

**Two kernel generators, two contracts (named for the contract):**

| | `DirectILKernelGenerator` | `ILKernelGenerator` |
|---|---|---|
| Location | `Backends/Kernels/Direct/*.cs` (~50 partials) | `Backends/Kernels/ILKernelGenerator*.cs` (root) |
| Contract | **whole-array** — one call processes everything; kernel walks dims/strides | **per-chunk** — `NpyInnerLoopFunc(dataptrs, strides, count, aux)`; iterator drives |
| Status | **LEGACY** but ~95 % of hot paths today | **TARGET** (mostly placeholder) |

**The Tier-3B per-chunk SIMD shell** — `DirectILKernelGenerator.CompileInnerLoop` in `Direct/DirectILKernelGenerator.InnerLoop.cs` — is what NpyIter uses for custom element-wise kernels. It emits, at runtime:

```
contig-inner?  ── yes ─► 4×-unrolled SIMD loop  +  1-vector remainder  +  scalar tail
   │                       (widest vector: VectorBits = 512/256/128)
   │                       + binary-broadcast variant (Vector.Create the scalar once)
   └── no (strided) ─────► EmitScalarStridedLoop     ◄── THE GAP (Phase 2a)
```

**Key routing facts (verified this session):**

- **Contiguous element-wise** → Direct whole-array kernel (NpyIter is bypassed to dodge its setup cost).
- **Flat reduction**: contiguous → Direct; non-contiguous → `NpyIterRef.New(a).ExecuteReduction<T>(op)` (`DefaultEngine.ReductionOp.cs:63`).
- **Axis reduction** (`sum`/`min`/`max`/`prod`): `Default.Reduction.Add.cs:ExecuteAxisReduction` → `DirectIL.TryGetAxisReductionKernel` **only**, throws if null (no NpyIter fallback). `NpyAxisIter` (scalar) is used for `var`/`std`/`cumsum`/`cumprod`/`all`/`any` axis — **not** sum/min/max/prod.
- `ExecuteReduction` / `ExecuteUnary` / `ExecuteBinary` in `NpyIter.Execution.cs` **delegate to the same Direct kernels** → on contiguous data NpyIter cannot beat Direct, it ties it.

---

## 2. Done so far

| Phase | Commit | Summary |
|---|---|---|
| **0 Hygiene** | `96a5ffcd` | Deleted dead `*_axis_simd` / `TryExecuteAxisReductionSimd` cluster (~125 lines). Fixed `Iternext()` EXLOOP over-iteration: it called `Advance()` (one-element ripple over *all* axes) unconditionally; now routes through `GetIterNext()` (`ExternalLoopNext`/`SingleIterationNext`/`StandardNext`). `StandardNext` == old behavior for the common case → strict correction. **Suite: 9447 pass.** |
| **4 Buffered cast + multi-output** | `cb0a0720` | Verified the machinery already works end-to-end (strided cast, strided copyto, `np.place`, `copyto(where=)`). Fixed the one real parity bug: `NpyIterCasting.CanCast` diverged from NumPy. Rewrote `IsSameKindCast`/`IsSafeCast` → **338/338 cast-matrix cells now byte-identical to NumPy 2.x** (both `safe` and `same_kind`). Corrected a mislabeled "parity" test that asserted the opposite of NumPy. **Suite: 9447 pass.** |
| **2b Widening-int SIMD** | `845f5e0b` | **DONE.** Narrow-int axis sums now at/beyond NumPy: int16 axis0 1068.6→4.61 ms (1.01× NumPy), axis1 →2.72 ms (0.84× = faster than NumPy), uint8 axis0 →4.03 ms (0.91×). int32 axis0 24.2→7.78 ms. mean(int16) 1354→40.6 ms. **Fixed silent uint32 axis-0 sum corruption** (UnpackLow/High per-lane interleave swapped output columns 2,3↔4,5 per group of 8). Design: 3 tiers — concrete per-pair Sum loops (blocked row-streaming, int32/uint32 chunked scratch for 8/16-bit), generic static-abstract tier for Prod/Min/Max/Mean(→double), typed scalar fallback. **Bisect verdict: NO regression existed** — 6038990f measures identically; the snapshot report's NumSharp axis numbers are pipeline artifacts (~19.5× off). **Suite: 9477 pass (13 new tests).** |
| **POC: NpyIter ≥ NumPy** | `0c8a5d6f` + this session | `benchmark/poc/` drives every layout through real `NpyIterRef` vs NumPy 2.4.2. After fixing the Debug-build pitfall (§4.12) and switching strided kernels to **AVX2 hardware gather**: **every aspect at or faster than NumPy** — contig sqrt/add 0.92×/0.96×, strided add/sqrt/sum **0.77×/0.55×/0.53×** (1.3–1.9× faster), fusion **2.8×/5.4× faster**, small-N dispatch 0.40 vs 0.44 µs/call. NumPy strided ground truth read from source (binary=scalar-only, reduce=scalar-8acc, unary=hw-gather). Full table: `benchmark/poc/POC_RESULTS.md`. |

**Phase 0 leftover — bug (b), deferred (test-first):** `Advance()` / `ExternalLoopNext` (`NpyIter.State.cs:730`, `NpyIter.cs:1511/1521`) advance pointers by `Strides × ElementSizes`. After a **buffered cast**, `ElementSizes` becomes the *buffer* dtype size while `Strides` are still *source* strides → wrong byte delta across a multi-buffer-fill iteration. The bridge avoids it by construction today (so the suite is green), but it must be fixed before Phase 3 routes buffered casts through the raw advancer. **Fix:** advance buffered operands by `BufStrides` (already byte-sized). **Gate:** a multi-fill (N > 8192) buffered-cast test that currently has no coverage.

---

## 3. The remaining phases at a glance

```
⬜ 1   Kill the ~1 µs setup tax            unlocks routing contiguous through NpyIter (removes the dual path)
                                            (Release re-measure: full MultiNew+exec+dispose already 0.40 µs vs NumPy 0.44 —
                                             re-baseline the gap under -c Release before sizing this phase)
⬜ 2a  Fused strided-SIMD in Tier-3B        REAL gaps re-proven on a[::2,::2] (2.48×) + strided binary (2.2×); the doc's
                                            "transposed unary" probe was an ARTIFACT (2× element count + F-contig = parity).
                                            POC kernels now BEAT NumPy 1.3–1.9× via AVX2 hardware gather (§4.12) — port them.
✅ 2b  Widening-int SIMD (8/16-bit)         DONE 845f5e0b — narrow ints at/beyond NumPy; uint32 corruption bug fixed
⬜ 3   Migrate kernels → per-chunk model    one driver; retire ~50 Direct partials; enables fusion
⬜ b   Buffered-cast Advance (test-first)   correctness prerequisite for Phase 3 buffered paths
```

Dependency order: **1 → 2a → 3**, with **b** landed before 3's buffered paths.

**2b leftovers (optional tightening, not blockers):** int32/uint32 axis sums at 1.3–1.5× NumPy (widen+add uop-bound; was 4.6×); mean(int) via the generic tier ~40 ms (concrete →double Sum loops would take it to ~5–8 ms); innermost int16/uint8 Sum could use PMADDWD/PSADBW for a further ~2×.

---

## 4. Proven facts & hard-won lessons (READ BEFORE CODING)

These came out of direct measurement this session. They will save you from dead ends.

1. **NpyIter's SIMD codegen == Direct's, on contiguous data.** Measured `abs`/`sqrt`/`square`/`negate` at 10M through the real Tier-3B path vs `np.*` (Direct): **0.96–1.02×**, at NumPy parity. *Corollary:* you will **never** speed up a contiguous op by "routing it through NpyIter" — it calls the same kernel. The wins are elsewhere (setup tax, strided, widening).

2. **The strided gap is real but its fix is implementation-sensitive.** sqrt f32 strided (stride-2), 1 M elements:

   | path | ms | vs ceiling |
   |---|--:|:--:|
   | NpyIter strided (scalar fallback) | 1.55 | 2.2× ← the gap |
   | **production fused `TryStridedSimdUnaryOp` (raw-ptr `Vector.Create`)** | **0.706** | **1.01× = NumPy** |
   | contiguous SIMD ceiling | 0.699 | 1.0× |
   | NumPy strided (reference) | 0.684 | 1.0× |

   **Which production shapes actually hit that scalar-fallback row** (re-measured on current HEAD, 1M f32, §12.1): a 1-D `a[::2]` unary does **not** — `TryStridedSimdUnaryOp` rescues it to **1.11×**, and `m[:,0]` collapses to 1-D and is rescued to **1.08×**. The 2.2× scalar-fallback is reached in production only by a **≥2-D strided unary (transpose) = 1.99×** or a **strided binary (`a[::2]+b[::2]`) = 2.45×** (the fused path is unary-only). *Always benchmark Phase 2a with the transpose/binary shapes, never the 1-D slice — the 1-D slice will tell you "no gap" and it's lying.*

3. **Two "obvious" strided fixes FAIL — do not use them:**
   - **Buffered (gather a 4096-tile to scratch, then SIMD the tile):** *slower* than scalar (0.63–0.94×). The scratch write+read round-trip costs more than SIMD saves.
   - **Managed fused (`Vector.Create` from a `stackalloc Span`):** a *wash* (0.96–1.04×, and 0.67× on a heavy poly) — bounds-checks + span store/reload stalls eat the gain.
   - **Only the raw-pointer fused gather (no scratch round-trip, tight IL) wins.** That's `TryStridedSimdUnaryOp`. Phase 2a = port *that exact technique* into the shell.

4. **SIMD-on-strided only helps when compute ≥ gather.** At large N the strided case becomes memory-bound (reading ~2× the cache lines) and converges to the strided-memory floor — which NumPy also hits (4M: fused 1.23× vs ceiling; NumPy 1.21×). So 2a is a **strict win-or-tie, never a loss** — unlike buffering.

5. **`NpyAxisIter` is scalar.** Routing narrow-int axis-sum through it would land at the slow scalar number, not the SIMD one. The lever for 2b is a **widening SIMD kernel**, not the iterator.

6. **Narrow-int axis-sum is catastrophically scalar.** int16 `sum(axis=…)` ≈ 1150 ms vs int32 ≈ 22 ms on identical 10M shapes → **25–57×** across axes/shapes. Root cause: `Reduction.Axis.Widening.cs` covers only int32/uint32/float; byte/int16/uint16/uint8 → int64 fall to `CreateAxisReductionKernelScalar` (cache-hostile, non-tiled, non-SIMD).

7. **The ~1 µs setup tax is the *only* reason contiguous bypasses NpyIter.** At N=1K, NpyIter is 1.06–1.58× slower than Direct purely from `MultiNew` construction; invisible at 10M. Kill it and the dual path can collapse.

8. **Suspicious benchmark rows (re-validate, don't trust at face value):** `np.copy` float64 10M = 0.00× (likely dead-code-eliminated / returns a view, not an 80 MB copy); `searchsorted` = 0.00× (scalar `v` vs NumPy's array `v` — apples-to-oranges); `np.sum` float64 100K = 11.83× (lone outlier among green rows). The `amin`-vs-`amax` 2–3× asymmetry from the report **did not reproduce** in a clean in-process measurement (both ≈ equal) — treat that "priority" as report noise.

9. ~~**Possible regression to bisect**~~ **RESOLVED (Phase 2b): there was NO regression.** Checkout of `6038990f` measures identically to HEAD (int16 axis0 ≈ 1137 ms; int32 ≈ 25.7 ms; same shapes, same Int64 output dtype). The snapshot report's NumSharp axis numbers (58 ms / 6.2 ms) are **benchmark-pipeline artifacts** (~19.5× too low); the NumPy column in the same report matches reality. Treat *all* NumSharp axis-reduction rows in `benchmark/history/2026-06-05_6038990f/` as unreliable until the report pipeline is audited.

10. **JIT lesson from 2b (extends the `Add256<T>` post-mortem in `Reduction.Axis.Simd.cs`):** in widen-density hot loops (one widen per 4 elements), **every** form of abstraction has a measured cost on .NET 10: one interface hop (static abstract *or* instance, `byte*` or `Vector256` params) ≈ +50%; two hops ≈ +104%; generic structs / static generic helpers with `typeof(T)` chains ≈ 3.5–7× (the chains do NOT fold on this call shape). Hot kernels must be fully concrete per pair. A generic-pointee pointer (`TIn*`) in an interface signature is worst of all — it defeats constrained-call devirtualization entirely (boxed virtual call per invocation, ~10 cycles/element).

11. **Silent-corruption bug class (found & fixed in 2b):** `Avx2.UnpackLow/UnpackHigh` interleave **per 128-bit lane** — using them to zero-extend for *positionally stored* accumulators scrambles element order (uint32 axis-0 sums had columns 2,3 ↔ 4,5 swapped per group of 8, undetected by 9447 tests). Order-sensitive widening must use PMOVSX/PMOVZX/CVT element-order loads. Audit any other Unpack* uses against positional stores.

12. **THE BENCHMARK-INVALIDATOR (found 2026-06-10): `dotnet run` file-based scripts build DEBUG by default — the script assembly *and* the `#:project`-referenced NumSharp.Core** (`DebuggableAttribute(DisableOptimizations)` = flags 263; the JIT honors it even over `[MethodImpl(AggressiveOptimization)]`). Measured: hand-written C# hot loops ~2× slower (POC C 789→319 µs, D 470→206, E 368→109); `MultiNew` construction ~40% slower (H 0.69→0.40 µs/call). `DynamicMethod`-emitted kernels are **immune** (emitted IL is always JIT-optimized) — which is why contiguous/IL-kernel/fusion numbers always looked fine while hand-written strided kernels looked mysteriously 2× slow, and why the cause survived scheduling-, GC-, stride-codegen- and alignment-elimination. `#:property Optimize=true` fixes only the script (Core stays Debug); `#:property Configuration=Release` changes output paths but binaries stay unoptimized; **only command-line `dotnet run -c Release` optimizes both**. Verify with `IsJITOptimizerDisabled` on both assemblies — the POC asserts it at startup.
    **Corrections this forces onto earlier conclusions:**
    - **§4.3's "only raw-pointer insert-gather wins; compaction loses 2×" is amended.** Under Release (interleaved, runtime strides, 1M f32): stride-2 `vshufps`/`vpermd` compaction 314 µs ≈ **AVX2 hardware gather (`Avx2.GatherVector256`) 334 µs** < insert-gather 387 < scalar-8× 362–373 < plain scalar 396. **Hardware gather is the production strided-load technique**: stride-general (compaction is stride-2-only), index vector hoisted so the hot loop is stride-agnostic, and it is NumPy's own strided-unary technique (`npyv_loadn_f32` = `_mm256_i32gather_ps`, `simd/avx2/memory.h`). Guard: gather-capable dtype (f32/f64/i32/i64), `|7·stride| ≤ int.MaxValue`, `Avx2.IsSupported`; insert-gather is the fallback (also for gather-slow cores: Zen 2/3, pre-Skylake).
    - **NumPy strided ground truth** (from `src/numpy/`): strided *binary* float ops have **NO SIMD path** (`loops_arithm_fp.dispatch.c.src` → `goto loop_scalar`); strided *reduce* is a **scalar 8-accumulator** loop (`loops_utils.h.src pairwise_sum`); only strided *unary* uses hardware gather (`loops_unary_fp.dispatch.c.src`, 4×-unrolled). NumPy's strided floors are therefore beatable — POC (Release, back-to-back): strided add **1.30×**, strided sqrt **1.82×**, strided sum **1.88× FASTER than NumPy**.
    - The §4.2/§4.3 measurements predate this discovery (Debug NumSharp.Core); the *directional* verdicts held up (fused-gather ≫ scalar/buffered/managed-span), but **re-validate any §4 ratio under `-c Release` before building on it**. §4.10's abstraction-cost magnitudes likewise need a Release re-validation pass before being treated as law. The committed 2b production kernels are unaffected (they live in NumSharp.Core and tests/benchmarks of *production* builds compile Release) — their documented wins are conservative.

---

## 5. Phase 1 — Kill the ~1 µs setup tax

**Goal.** Make `NpyIterRef.New` / `MultiNew` construction effectively free (target < ~50 ns for the common single-/dual-operand, contiguous, C-order case) so contiguous ops can route through NpyIter with zero penalty — the precondition for collapsing the Direct/NpyIter dual path.

**Peak-performance mechanism.** Today every op below a few-thousand elements is **dispatch-bound, not compute-bound** — the work is over before the pipeline warms, so fixed overhead dominates the roofline. NpyIter construction currently does, per call: heap-allocate `NpyIterState`, copy shapes/strides, run dimension **coalescing**, resolve **C/F/A/K order**, and (when buffered) allocate aligned buffers. For a 1-D contiguous operand none of that changes the answer. Peak is reached by *eliminating the setup from the critical path of the trivial case*:

1. **Fast-path constructor** for `NOp ≤ 2`, all-contiguous, C-order, no cast, no buffer: skip coalescing/order-resolution, fill a minimal state directly (one axis = total size, stride = elemSize). This is NumPy's own `check_for_trivial_loop` idea applied to construction.
2. **Stack-allocate the state** for small `NOp`/`NDim` (the dominant case) instead of heap — removes the alloc + GC pressure that shows up as the 1 µs.
3. **Cache/pool** the resolved state keyed by `(shape-signature, NOp, order)` so a hot loop calling the same op shape reuses it (NumPy reuses the iterator across `iternext`; we pay per call).

**Evidence.** N=1K, NpyIter/Direct = 1.06–1.58× (abs 1.21×, sqrt 1.58×); the absolute delta is ~1 µs, flat across ops → it's construction, not kernel.

**Implementation.** `NpyIter.cs` (`New`/`MultiNew`, `NpyIter.State.cs` allocation). Add a `TryNewTrivial(...)` path; benchmark each lever independently.

**Expected win.** N=1K NpyIter/Direct → ~1.0×. No change at 10M (already 1.0×). Unblocks Phase 3.

**Verification.** Re-run the unary micro-bench at N=1K and N=1M (template in §9 / script in §12.1); the small-N ratio should collapse to ≈1.0. Full suite green.

**Risk.** Low–medium: touches the constructor used everywhere. The fast path must *exactly* reproduce the general path's state for the trivial case (assert-equal the two in a debug test across a shape sweep).

---

## 6. Phase 2a — Fused strided-SIMD in the Tier-3B shell  *(highest-confidence next step)*

> **POC UPDATE (2026-06-10, §4.12 + `benchmark/poc/`).** The strided inner loops are written and proven — as hand-written per-chunk kernels (`PocKernels.AddF32/SqrtF32/SumF32` in `npyiter_parity_poc.cs`) driven by `NpyIterRef.ForEach`, they measure **FASTER than NumPy**: strided binary 319 vs 416 µs, strided 2-D unary 206 vs 374 µs, strided reduce 109 vs 205 µs (1M f32, Release, back-to-back). The technique hierarchy (Release, interleaved): **AVX2 hardware gather** (`Avx2.GatherVector256`, scale=1, byte-offset index vector built once per chunk — hot loop is stride-agnostic) for gather-capable dtypes f32/f64/i32/i64 with `|7·stride| ≤ int.MaxValue`; **insert-gather** (`Vector.Create` from strided lanes — the §4.2/§4.3 technique) as the fallback and for non-gatherable dtypes. Phase 2a = make `EmitFusedStridedSimdLoop` emit exactly those two bodies. Emitted `DynamicMethod` IL is immune to the §4.12 Debug pitfall by construction. Unroll factors that won: binary 2× (2 gathers/iter already saturate), unary 4× (NumPy uses 4× too), reduce 4 independent accumulators.

**Peak-performance mechanism.** A strided element-wise op has the same number of scalar *loads* as a contiguous one (you must touch each element), but the scalar fallback also does the *compute* one element at a time, leaving the SIMD ALUs idle — so it runs at the **scalar-compute roofline**, ~`lanes×` below the SIMD roofline. The fused gather raises it to the SIMD roofline *without adding memory traffic*:

```
for each vc-wide chunk:
    v = Vector.Create(*p, *(p+s), *(p+2s), …, *(p+(vc-1)s))   ← vc scalar loads, straight into a register
    r = vectorBody(v)                                          ← ONE SIMD op for vc elements
    store r contiguously (or strided for the output)          ← no scratch buffer, no round-trip
```

The decisive subtlety (see §4.3): the gather is **register-direct** — the same `vc` loads the scalar loop already issues, assembled into a vector via `Vector.Create`, with **no scratch write+read**. Measurement shows that assembly is nearly free relative to the loads, so the per-element cost drops from "scalar compute" to "SIMD compute / lanes". When the op is memory-bound at large N, it gracefully converges to the strided-memory floor (which NumPy also hits) — hence **win-or-tie, never loss**.

**Evidence.** §4.2 table: 1.55 ms → 0.706 ms = 2.2×, lands exactly on the contiguous ceiling and on NumPy. §4.3: buffering and managed-span both fail — must be raw-ptr IL.

**Exactly where it falls to scalar today** (read `CompileInnerLoop`, `Direct/DirectILKernelGenerator.InnerLoop.cs:127`):
- **Binary** (`InnerLoop.cs:242-258`): SIMD covers only `CC` (both contig), `SL` (lhs **stride-0 broadcast**, rhs contig), `SR` (lhs contig, rhs stride-0). ⚠️ `SL`/`SR` are *broadcast*, **not** genuine strided — `a[::2] OP b[::2]` (both stride = 2·elem) hits neither and branches to `lblScalarStrided`.
- **Unary / ternary** (`InnerLoop.cs:262-272`): SIMD **only** when every operand stride == its elemSize; *any* stride ≠ elemSize → scalar. This is the whole multi-D strided-unary case.
- **Mixed-dtype / non-SIMD** (`InnerLoop.cs:277-301`): contig-scalar or strided-scalar; out of scope for 2a (leave as-is).
- All three converge on `EmitScalarStridedLoop` at `InnerLoop.cs:304` — **that is the single site Phase 2a replaces with a fused-gather SIMD loop** (keeping it as the final fallback for the mixed/non-SIMD/ndim-edge cases).

**Implementation.**
1. In `CompileInnerLoop`, at the SIMD-capable `lblScalarStrided` site (`InnerLoop.cs:275/300/304`), add `EmitFusedStridedSimdLoop` gated on: same dtype across operands, SIMD-capable (the existing `canSimd` predicate), at least one stride a non-unit multiple of elemSize. Keep `EmitScalarStridedLoop` as the final fallback (mixed dtype, non-SIMD types like Decimal/Half/Complex, ndim edge cases).
2. **Primary body (per the POC update above): AVX2 hardware gather** for f32/f64/i32/i64 — emit the per-operand byte-offset index-vector construction *before* the loop (`Vector256.Create(0, s, 2s, …, 7s)` as int32) and `Avx2.GatherVector256(base, idx, 1)` per lane-group inside it; advance bases by `lanes·stride`. Reference implementations to transcribe: `PocKernels.AddF32/SqrtF32/SumF32` (`benchmark/poc/npyiter_parity_poc.cs`) and the existing gather usage in `Direct/DirectILKernelGenerator.Reduction.Axis.Simd.cs:741`. Gate on `Avx2.IsSupported && |7·stride| ≤ int.MaxValue`.
3. **Fallback body: insert-gather** — port the emission from `TryStridedSimdUnaryOp` / `GetStridedUnaryKernel` in `Direct/DirectILKernelGenerator.Unary.Strided.cs` (register-direct `Vector.Create` from strided lanes). Generalize from 1 input to N inputs, and to an output that may itself be strided (strided store, lane-by-lane).
4. Cover the genuine multi-stride binary case (`a[::2] OP b[::2]`, `a[::2] OP b` contiguous, transposed-non-F) that `CC`/`SL`/`SR` miss — they only handle contig + stride-0-broadcast, not stride-k.

**Files.** `Direct/DirectILKernelGenerator.InnerLoop.cs` (new `EmitFusedStridedSimdLoop` emitter at the `lblScalarStrided` site), `Direct/DirectILKernelGenerator.Unary.Strided.cs` (reuse the lane-gather IL helpers — **note: it's `Unary.Strided.cs`, not `Unary.Vector.cs`**), `NpyIter.Execution.Custom.cs` (no change — it already drives the shell).

**Expected win.** Strided multi-D + strided-binary: 2.2× → ~1.0–1.23× of contiguous SIMD; vs NumPy, the POC-proven targets are **0.77× (binary) / 0.55× (unary) / 0.53× (reduce)** — faster than NumPy, because NumPy has no SIMD at all for strided binary/reduce (§4.12 ground truth).

**Verification.** New micro-bench: `a[::2,::2] + b[::2,::2]` and a transposed-non-F unary, through the *real* NpyIter path, vs Direct and NumPy. Confirm ≈2.2× over the old scalar and parity with NumPy. Full suite green. (A 1-D strided unary already gets rescued by `TryStridedSimdUnaryOp` upstream — test multi-D/binary specifically, which is where the shell actually runs.)

**Risk.** Low: additive (new branch; scalar fallback retained). Correctness gate: results must equal the scalar path bit-for-bit across a strided/broadcast shape sweep.

---

## 7. Phase 2b — Widening-int SIMD (8/16-bit → 64-bit)  ✅ DONE (`845f5e0b`)

> **Outcome (2026-06-09).** Shipped. Narrow-int axis sums at/beyond NumPy 2.4.2: int16 axis0 1068.6→**4.61 ms** (1.01×), axis1 →**2.72 ms** (0.84×, faster than NumPy), uint8 axis0 →**4.03 ms** (0.91×), int32 axis0 24.2→7.78 ms, mean(int16) 1354→40.6 ms. Also **fixed silent uint32 axis-0 corruption** (§4.11) and added the missing SByte/Boolean/Char typed-scalar rows. Architecture: concrete per-pair Sum loops (blocked row-streaming; int32/uint32 chunked scratch for 8/16-bit inputs — see §4.10 for why generics are banned there) + generic static-abstract tier for Prod/Min/Max/Mean(→double) + typed scalar fallback. Flat reductions were measured FINE (4.3 ms ≈ int32) — the hole was exclusively the axis path. 13 regression tests in `AxisReductionWideningTests.cs`; suite 9477 green. The section below is kept for historical context.

**Goal.** Give byte/int16/uint16/uint8 reductions (flat *and* axis sum/prod) the same widening-SIMD treatment int32/uint32/float already have, eliminating the 25–57× scalar penalty.

**Peak-performance mechanism.** `sum(int16)` must accumulate into int64 (NEP50). The current scalar kernel (`CreateAxisReductionKernelScalar<short,long>`) is doubly off the roofline: (1) **no SIMD** (one add per element), and (2) **cache-hostile** for `axis=0` (output-outer, input-inner stride → a cache miss per element). The fix mirrors the existing int32 path in `Reduction.Axis.Widening.cs`:

```
load Vector256<short> (16 lanes)
   ├─ widen lower 8 → Vector256<long>  (sign/zero-extend)
   └─ widen upper 8 → Vector256<long>
accumulate into register-resident Vector256<long> column tiles (touched once at the end)
```

This hits the SIMD roofline for the adds **and** the memory roofline via the column-tiled accumulator (register-resident across all rows, output written once) — the same pattern that took int32 from 491× scalar down to ~1×. Narrow types win *more* per vector (16 int16 lanes vs 8 int32) once widened.

**Evidence.** int16 ≈1150 ms vs int32 ≈22 ms, identical 10M shapes, all axes → 25–57×. `Reduction.Axis.Widening.cs:56-62` `covered` list is exactly {int32→int64, uint32→int64/uint64, float→double}; the comment even says *"Falls through to scalar for other widening pairs (byte/short/…)"*.

**Implementation.**
1. Extend `TryGetAxisReductionWideningKernel`'s `covered` set and add the byte/sbyte/int16/uint16 → int64/uint64 specializations using `Avx2` widen intrinsics (`Avx2.ConvertToVector256Int32`/`Int16` widen, or `Vector256.Widen`).
2. Do the same for the **flat** widening reduction if it shares the scalar gap (check `ExecuteElementReduction` narrow-int path).
3. **First**, `git bisect` the axis-reduction code vs snapshot `6038990f` (§4.9) — a fast path may have regressed; restoring it could be the real fix or a complement.

**Files.** `Direct/DirectILKernelGenerator.Reduction.Axis.Widening.cs`, `Direct/DirectILKernelGenerator.Reduction.cs` (flat), `Direct/DirectILKernelGenerator.Reduction.Axis.cs` (dispatch).

**Expected win.** int16/uint16/uint8 axis-sum: ~1150 ms → ~22 ms (int32-class), 25–57× → ~1×.

**Verification.** Narrow-int axis-sum vs int32 same shape (should converge); correctness vs scalar across signed/unsigned/overflow cases; full suite; benchmark re-run (and confirm/deny the regression).

**Risk.** Medium: widening intrinsics + sign-extension correctness for signed narrow ints; overflow/wrap parity with NumPy's int64 accumulator. Strong test matrix required.

---

## 8. Phase 3 — Migrate kernels to the per-chunk model

**Goal.** Route each np.* family through `NpyIterRef.Execute*` (per-chunk) and delete its `Direct/DirectILKernelGenerator.<X>.cs` partial, leaving **one driver** where the now-complete Tier-3B shell (Phases 1/2a/2b) gives every layout its roofline.

**Peak-performance mechanism.** Peak here is *architectural*, not micro: today the same op has two implementations that can drift, and the strided/widening/F-order cases each need bespoke handling per partial. With one driver, **coalescing, order resolution, broadcasting, and the SIMD/strided/widening path selection happen once, centrally**, so every op automatically gets the best path — and the door opens to **cross-op fusion** via the Tier-3C `NpyExpr` DSL (e.g. `a*b + c` as one buffered SIMD pass instead of three array round-trips), which is the next roofline jump beyond per-op SIMD (it removes intermediate memory traffic entirely).

**Order (NumPy-priority):** reductions → binary arith → comparison → unary → scan → copy → multi-output (`Modf`) → selection (`Where`/`Place`, enabling VIRTUAL/WRITEMASKED operands).

**Implementation pattern (per family).** (1) Wire the call site to `NpyIterRef.Execute…`; (2) benchmark contiguous + strided + widening vs the Direct kernel — must be ≥ parity (this is why 1/2a/2b come first); (3) full suite green; (4) delete the `Direct/` partial. Never delete before the benchmark + suite gate.

**Files.** `DefaultEngine.*.cs` call sites; `NpyIter.Execution.cs`; remove `Direct/DirectILKernelGenerator.<X>.cs` per family.

**Expected win.** No single-op speedup (parity is the bar) — the payoff is eliminated divergence, ~50 fewer partials to maintain, and the fusion runway.

**Verification.** Per-family benchmark non-regression + full suite, before each partial deletion.

**Risk.** Medium-high cumulative — but incremental and individually revertible. Gate hard.

---

## 9. Verification toolkit

**Micro-bench (file-based script).** Template proven this session — reference the project with internal access. **⚠️ MUST run as `dotnet run -c Release - < script.cs`** — plain `dotnet_run` builds the script AND NumSharp.Core in Debug and silently doubles hand-written hot-loop times (§4.12):

```csharp
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// TimeMs(Action, iters, warmup); compare Direct (np.*) vs NpyIter path; print ms + ratio.
// Always test BOTH a throughput size (10M, roofline) AND a small size (1K, dispatch tax).
// Assert at startup that IsJITOptimizerDisabled == false on BOTH assemblies (§4.12).
```

**Cast-matrix gate (Phase 4 style).** Dump `NpyIterCasting.CanCast` for the 13 core types × {safe, same_kind} and diff against `np.can_cast` — must be 338/338 identical.

**Full suite (CI filter).**
```bash
cd test/NumSharp.UnitTest
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0
dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
# gate: Passed: 9447  Failed: 0
```

**Benchmark re-run.** `python run_benchmark.py --suites …` then diff the per-(op,dtype,N) ratios in `benchmark/benchmark-report.md` (watch the suspicious rows in §4.8).

**Roofline rule of thumb.** Before optimizing, classify the op at the target size: *compute-bound* (small/cache-resident, or heavy op) → SIMD width is the ceiling; *memory-bound* (large, cheap op) → bandwidth is the ceiling and SIMD won't help beyond it. 2a/2b raise the compute ceiling; nothing raises the memory ceiling (match NumPy and stop).

---

## 10. Open decisions — **what should the next session do?**

Pick one (my recommendation in **bold**):

- **A. Phase 2a (fused strided-SIMD).** Highest confidence: 2.2×→parity already proven, reference impl in-tree to port, additive/low-risk, moves real production numbers (multi-D strided + binary). *Recommended first.*
- **B. Phase 2b (widening-int SIMD).** Biggest raw number (25–57×) but a fresh SIMD kernel + sign-extension correctness; do the `git bisect` vs `6038990f` first to see if it's partly a regression.
- **C. Phase 1 (setup tax).** Smaller direct win but unlocks Phase 3 and removes the dual path; do it before 3 regardless.
- **D. Bug (b) (buffered-cast Advance), test-first.** Smallest, pure correctness; prerequisite for Phase 3's buffered paths.
- **E. Investigate the suspicious benchmark rows / regression (§4.8–4.9)** before more optimization, so the numbers we chase are real.

**Sequencing recommendation:** **A** now (visible win, low risk) → **B** (with the bisect) → **C** → **D** → **3**. Each lands as its own verified commit.

---

## 11. Session quickstart runbook (orient in ~5 min)

A cold session should run these in order before touching code.

```bash
# 0. Where am I
cd K:/source/NumSharp
git branch --show-current          # expect: nditer
git log --oneline -4               # expect cb0a0720 (Phase 4) and 96a5ffcd (Phase 0) near HEAD

# 1. Build the library (net10.0, silent-on-success)
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0 src/NumSharp.Core/NumSharp.Core.csproj

# 2. Baseline the suite (this is the green line every phase must hold)
cd test/NumSharp.UnitTest
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0
dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory" \
  2>&1 | grep -E "^(Passed!|Failed!|  total:|  failed:|  succeeded:|  skipped:)"
# GATE: succeeded 9447, failed 0
```

Then read, in order: **§1** (the two-generator split + the Tier-3B shell), **§4** (don't re-walk the failed paths), the phase you picked from **§10**. The single most load-bearing fact: *NpyIter shares Direct's kernels on contiguous data, so wins live only in setup-tax (§5), strided (§6), widening (§7) — never in "route contiguous through NpyIter."*

**If picking Phase 2a (recommended):** open `Direct/DirectILKernelGenerator.InnerLoop.cs:304` (`EmitScalarStridedLoop` — the one site to replace) and `Direct/DirectILKernelGenerator.Unary.Strided.cs` (the proven raw-ptr gather IL to generalize). Reproduce the gap with the §12.1 script *first* (confirm 2.2×), then make it parity, then hold the §12.2 gate.

---

## 12. Verification scripts (paste-able, not just templates)

### 12.1 Strided-SIMD micro-bench — prove the gap, then prove parity

> ⚠️ **The shape is the whole experiment. Get it wrong and the gap vanishes.** A 1-D `a[::2]` unary does **not** reach the shell's scalar fallback — `TryStridedSimdUnaryOp` rescues it upstream (and `m[:,0]` collapses to 1-D and is rescued too). Those read ~1.1× and look fine, hiding the real gap.
>
> ⚠️⚠️ **CORRECTION (2026-06-09 re-proof): the `reshape(2,n).T` "transposed unary" probe below is an ARTIFACT — do not use it.** It has **2× the elements** of the 1M baseline *and* is F-contiguous; its ≈2× ratio is element count, not a gap (NumPy shows the identical 2.02× on it, and an equal-count F-transpose runs at **0.99× = parity** in NumSharp). The real ≥2-D strided unary probe is **`a[::2, ::2]` of a (2000, 2000)** — equal element count, genuinely strided, stays 2-D. Re-proven gaps on current HEAD (1M f32, equal counts): **2-D strided unary `a[::2,::2]` 2.48×** (NumPy itself: 1.38× — its strided-memory floor), **strided+strided binary 2.24×** (NumPy 1.15×), **strided+contig binary 2.21×** (NumPy 1.02×). In absolute terms NumSharp is 1.78–1.97× slower than NumPy on these. Post-2a targets: match NumPy's floors (1.0–1.4×), not 1.0×.

Run **before** Phase 2a (expect transposed ≈ 2.0×, binary ≈ 2.4× over the contiguous ceiling) and **after** (expect both ≈ 1.0–1.3×). Use **~1M** (compute-bound, gap widest); at 10M the op is memory-bound and converges to the ~1.2× strided-memory floor that NumPy also hits (§4.4) — that's the floor, not the gap.

> ⚠️⚠️⚠️ **`-c Release` is mandatory (§4.12).** The ratios in the correction above were measured before the Debug-default discovery — both numerator and denominator were Debug-inflated, so the *ratios* survived, but absolute numbers and NumPy comparisons did not. Re-baseline under `dotnet run -c Release` before/after 2a.

```bash
dotnet run -c Release - <<'EOF'
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System.Diagnostics;
using NumSharp;

static double TimeMs(Action f, int iters, int warmup) {
    for (int i = 0; i < warmup; i++) f();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) f();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / iters;
}

int n = 1_000_000, it = 400;                                  // compute-bound size
var contig = np.arange(n).astype(np.float32) + 1f;            // contiguous baseline

// (trap) 1-D strided unary — rescued by TryStridedSimdUnaryOp, NOT the shell:
var wide   = np.arange(2 * n).astype(np.float32) + 1f;
var s1d    = wide["::2"];

// (real) ≥2-D strided unary — transpose keeps it ≥2-D → shell scalar fallback:
var mT     = (np.arange(2 * n).astype(np.float32) + 1f).reshape(2, n).T;

// (real) strided binary — fused path is unary-only → shell scalar fallback:
var wide2  = np.arange(2 * n).astype(np.float32) + 2f;
var b1     = wide["::2"]; var b2 = wide2["::2"];

double cU = TimeMs(() => { var _ = np.sqrt(contig); }, it, 100);
double cB = TimeMs(() => { var _ = contig + contig; }, it, 100);
Console.WriteLine($"contig sqrt (unary base)   {cU:F4} ms   1.00x");
Console.WriteLine($"1-D strided sqrt  (trap)   {TimeMs(()=>{var _=np.sqrt(s1d);},it,100):F4} ms   {TimeMs(()=>{var _=np.sqrt(s1d);},it,50)/cU:F2}x   <- rescued, ignore");
Console.WriteLine($"transposed sqrt   (REAL)   {TimeMs(()=>{var _=np.sqrt(mT);},it/2,50):F4} ms   {TimeMs(()=>{var _=np.sqrt(mT);},it/2,30)/cU:F2}x   <- Phase 2a target");
Console.WriteLine($"contig add (binary base)   {cB:F4} ms   1.00x");
Console.WriteLine($"strided add       (REAL)   {TimeMs(()=>{var _=b1+b2;},it,100):F4} ms   {TimeMs(()=>{var _=b1+b2;},it,50)/cB:F2}x   <- Phase 2a target");
Console.Error.WriteLine("[done]");
EOF
```

Notes: `np.sqrt(x)` inside a `() => { ... }` lambda is fine; the CS0123 method-group error only bites when passing `np.sqrt` *as* a delegate — always wrap in a lambda. Swap `np.sqrt` for `np.square` (heavier → bigger gap) or `np.abs` (lighter → smaller). After Phase 2a, the two `(REAL)` rows must drop to ≈1.0–1.3× while the suite (§12.2) stays green.

### 12.2 Full-suite gate (the non-negotiable green line)

```bash
cd K:/source/NumSharp/test/NumSharp.UnitTest
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0
dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory" \
  2>&1 | grep -E "^(Passed!|Failed!|  total:|  failed:|  succeeded:|  skipped:)"
# GATE: succeeded 9447, failed 0   (any drop = stop and diff)
```

### 12.3 Cast-matrix gate (only if you touch `NpyIterCasting`)

Phase 4 made `CanCast` byte-identical to NumPy for 13 numeric dtypes × {safe, same_kind} = 338 cells. If you go near casting again, re-run both dumps and `diff` — they must match exactly.

```bash
# (a) NumSharp dump → /tmp/ns_cast.csv
dotnet_run <<'EOF'
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
using NumSharp;
using NumSharp.Backends.Iteration;   // NpyIterCasting, NPY_CASTING  (note: namespace is Iteration, not Iterators — folder ≠ namespace here)
var types = new[] {
    NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
    NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
    NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex,
};
foreach (var rule in new[] { ("safe", NPY_CASTING.NPY_SAFE_CASTING), ("same_kind", NPY_CASTING.NPY_SAME_KIND_CASTING) })
  foreach (var s in types)
    foreach (var d in types)
      Console.WriteLine($"{rule.Item1},{s.AsNumpyDtypeName()},{d.AsNumpyDtypeName()},{(NpyIterCasting.CanCast(s, d, rule.Item2) ? 1 : 0)}");
EOF

# (b) NumPy reference dump → /tmp/np_cast.csv, then diff
python_run <<'EOF'
import numpy as np
T = ['bool','uint8','int8','int16','uint16','int32','uint32','int64','uint64','float16','float32','float64','complex128']
for rule in ('safe','same_kind'):
    for s in T:
        for d in T:
            print(f"{rule},{s},{d},{1 if np.can_cast(np.dtype(s), np.dtype(d), rule) else 0}")
EOF
# Pipe each to a file and: diff <(sort ns_cast.csv) <(sort np_cast.csv)   → MUST be empty (338/338)
# dtype-name mapping: AsNumpyDtypeName() yields bool/uint8/int8/.../float16/float32/float64/complex128.
```

---

## 13. Regression bisect runbook (the §4.9 narrow-int slowdown)  ✅ RESOLVED — no regression

> **Outcome (2026-06-09).** Executed during Phase 2b. The endpoints are identical (HEAD 1136 ms vs `6038990f` 1137 ms, int16 axis0, benchmark's own 3162² shape, Int64 output both) — **no bisect needed, nothing regressed**. The snapshot report's "58 ms"/"6.2 ms" NumSharp numbers are report-pipeline artifacts (§4.9). The scalar axis path was always ~1130 ms; Phase 2b's fresh widening kernels were the right fix, with NumPy (not the bogus 58 ms) as the bar. Runbook kept below as a template for future bisects.

```bash
cd K:/source/NumSharp
# 1. A tiny, fast probe that returns the int16 axis-sum time (used as the bisect oracle).
cat > /tmp/axissum_probe.sh <<'SH'
#!/usr/bin/env bash
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0 -f net10.0 src/NumSharp.Core/NumSharp.Core.csproj >/dev/null 2>&1 || exit 125
dotnet run -c Release - <<'EOF'
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
using System.Diagnostics; using NumSharp;
var a = (np.arange(10_000_000).astype(np.int16) % 7).reshape(1000, 10000);
for (int i=0;i<3;i++){ var _=np.sum(a, axis:0); }      // warmup
var sw=Stopwatch.StartNew(); for(int i=0;i<10;i++){ var _=np.sum(a, axis:0);} sw.Stop();
double ms = sw.Elapsed.TotalMilliseconds/10;
Console.WriteLine($"int16 axis0 sum = {ms:F1} ms");
// bisect oracle: "good" if fast (<200ms), "bad" if regressed (>=200ms)
Environment.Exit(ms < 200 ? 0 : 1);
EOF
SH
chmod +x /tmp/axissum_probe.sh

# 2. Drive git bisect with the probe as the automatic oracle.
git bisect start
git bisect bad HEAD
git bisect good 6038990f
git bisect run /tmp/axissum_probe.sh
# → prints "<sha> is the first bad commit". Inspect its diff to Reduction.Axis.* / NEP50 widening.
git bisect reset
```

If the first-bad commit is the int64-widening correctness fix, the regression is *expected* (correctness traded the fast path) → Phase 2b's widening-SIMD kernel is the right restoration, now with a known-good target time. If it's something incidental, you may recover most of the win by repairing that path directly. Either way you now chase a *real* number, not report noise (§4.8).

---

## 14. North star beyond Phase 3 — Tier-3C fusion (`NpyExpr`)

> **POC-PROVEN (2026-06-09, upgraded 2026-06-10, `benchmark/poc/`).** The end-to-end POC drives real `NpyIterRef` execution head-to-head with NumPy 2.4.2 and is now **at or faster than NumPy on every aspect** (Release, back-to-back): contiguous sqrt/add **0.92×/0.96×**, strided add/sqrt/sum **0.77×/0.55×/0.53×** via AVX2 hardware-gather per-chunk kernels (§4.12), small-N dispatch **0.40 vs 0.44 µs/call** incl. full `MultiNew`, and **fusion 2.8×/5.4× FASTER** (`a*b+c` 4.77 vs 13.38 ms; `(a-b)/(a+b)` 4.12 vs 22.33 ms, f32 10M). Full table, methodology, and dead-ends in `benchmark/poc/POC_RESULTS.md`. Operational findings: the original POC's strided numbers (789/470/368 µs) were **Debug-build artifacts** (§4.12 — `dotnet run` file-based scripts disable the JIT optimizer by default; the 2026-06-09 "remaining ≤2× stride-2 gap vs RyuJIT" narrative is retracted); `ExecuteExpression` needs `EXTERNAL_LOOP` or `ForEach` degrades to per-element delegate calls (~38× slower); per-call `np.empty` costs ~0.3–0.4 ms in soft page faults at 4 MB (GC-deferred frees keep pages cold) — preallocate or pool for fair benchmarks.

Phase 3 isn't the finish line; it's what *unlocks* the next roofline jump. The DSL already exists: **`Backends/Iterators/NpyExpr.cs`** — an algebraic AST over NpyIter operands that compiles to a single `NpyInnerLoopFunc` via the same `CompileInnerLoop` shell (so it inherits every SIMD/strided/widening win from Phases 1/2a/2b for free).

**Why it's the real endgame (roofline).** Per-op SIMD raises the *compute* ceiling, but a composite like `a*b + c` evaluated as three separate ufuncs pays the *memory* roofline three times — two full-array reads + write per op, plus two materialized temporaries. `NpyExpr` fuses the tree into **one inner loop**: inputs are read once, the whole expression is computed in-register (scalar body and `Vector{W}<T>` body per node), the result written once. That removes ~⅔ of the memory traffic for a 3-op chain — and at large N every cheap elementwise op is *memory-bound*, so this is the only lever left after SIMD is maxed.

**Current state (from the file header).** Intermediate compute happens in the output dtype (inputs auto-promote on load); the SIMD path turns on iff every input type == output type and every node's op supports SIMD, else the compiled kernel carries a scalar body and the shell's strided fallback covers it. Compilation is cache-keyed by the tree's structural signature.

**What's missing to make it pay off (post-Phase-3 work, not yet scoped into a phase):**
1. **A capture surface** — let `a * b + c` on `NDArray` build an `NpyExpr` lazily (operator overloads returning expression nodes, or an explicit `np.evaluate(...)` / `ne.evaluate`-style entry) instead of eagerly calling three ufuncs.
2. **A materialization boundary** — decide when a lazy tree forces (assignment, indexing, reduction input) and route the force through one `NpyIterRef.Execute` over the compiled `NpyExpr`.
3. **Mixed-dtype promotion inside the tree** matching NumPy's casting-by-subexpression (the header notes the current "all-compute-in-output-dtype" simplification diverges from NumPy for some mixed chains — verify against `np.result_type` before exposing publicly).
4. **Bench the fusion win** — `a*b+c` fused vs three ops at 10M; expect ≈ the memory-traffic ratio (~2–3× for a 3-op chain), converging to NumExpr-class throughput.

This is the architectural reason Phase 3 ("one driver") is worth the migration cost: once every op is one per-chunk kernel over `NpyIter`, fusing N of them into one `NpyExpr` pass is a natural extension rather than a rewrite.

---

## 15. File & symbol index (where things actually live)

| Symbol / concern | File:line | Role |
|---|---|---|
| `CompileInnerLoop` | `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop.cs:127` | Tier-3B per-chunk SIMD shell (4× unroll + remainder + tail) |
| `EmitSimdContigLoop` | `…/DirectILKernelGenerator.InnerLoop.cs:378` | All-contig SIMD body (the path NpyIter ties Direct on) |
| `EmitScalarStridedLoop` | `…/DirectILKernelGenerator.InnerLoop.cs:725` | **The gap (Phase 2a)** — replace at the `lblScalarStrided` site (`:304`) |
| Binary `CC`/`SL`/`SR` branches | `…/DirectILKernelGenerator.InnerLoop.cs:242-258` | contig + **stride-0 broadcast** only — genuine multi-stride still scalar |
| `TryStridedSimdUnaryOp` | `Backends/Default/Math/DefaultEngine.UnaryOp.cs` | 1-D strided unary fused gather (the proven-win technique to generalize) |
| `GetStridedUnaryKernel` (gather IL) | `Backends/Kernels/Direct/DirectILKernelGenerator.Unary.Strided.cs` | Raw-ptr `Vector.Create`-from-lanes emitter to port into the shell |
| Axis-reduction widening `covered` set | `Backends/Kernels/Direct/DirectILKernelGenerator.Reduction.Axis.Widening.cs:56-62` | **Phase 2b** — only int32/uint32/float; add byte/sbyte/int16/uint16 |
| `ExecuteAxisReduction` (live axis path) | `Backends/Default/Math/Reduction/Default.Reduction.Add.cs:50` | Direct-only; throws if no kernel (no NpyIter fallback) |
| `ExecuteElementReduction` (flat routing) | `Backends/Default/Math/DefaultEngine.ReductionOp.cs:63` | contig→Direct, non-contig→`NpyIterRef…ExecuteReduction` |
| `ExecuteUnaryOp` dispatch | `Backends/Default/Math/DefaultEngine.UnaryOp.cs:73-116` | trivial-contig→Direct; strided→fused / buffered / NpyIter Tier-3B |
| `New` / `MultiNew` (setup tax) | `Backends/Iterators/NpyIter.cs` | **Phase 1** — add a trivial fast-path constructor here |
| `Advance` / buffered advance | `Backends/Iterators/NpyIter.State.cs:712 / 778` | **Bug (b)** — buffered operands must advance by `BufStrides`, not `Strides×ElementSizes` |
| `Iternext` (Phase 0 fix) | `Backends/Iterators/NpyIter.cs:2043` | now routes through `GetIterNext()` (was unconditional `Advance()`) |
| `CanCast` (Phase 4) | `Backends/Iterators/NpyIterCasting.cs:17` | `CanCast(src, dst, NPY_CASTING)` — 338/338 NumPy-identical |
| Tier-3B driver | `Backends/Iterators/NpyIter.Execution.Custom.cs` | `ExecuteElementWise` → `CompileInnerLoop` |
| `NpyExpr` (Tier-3C fusion DSL) | `Backends/Iterators/NpyExpr.cs` | **§14 north star** — AST → one fused `NpyInnerLoopFunc` |
| Execution bridge | `Backends/Iterators/NpyIter.Execution.cs` | `ForEach`/`ExecuteGeneric`/`ExecuteReduction`/`ExecuteUnary`/`ExecuteBinary` |
