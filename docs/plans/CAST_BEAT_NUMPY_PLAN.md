# Cast Path — Final Optimization Plan: Beat NumPy at Every Execution

**Scope:** `astype` / every cross-dtype conversion and same-dtype copy, across all 15 dtypes,
all output dtypes, all memory layouts and parameter variations. **Goal:** `NPY/NS ≥ 1.0` (ideally
`≥ 1.5`) in **every** cell of the 15×15×layout matrix, with bit-exact NumPy 2.4.2 semantics.

> Convention (house): `ratio = NumPy_ms / NumSharp_ms`. `>1` = NumSharp faster. Icons ✅ `≥1.0`
> · 🟡 `≥0.5` · 🟠 `≥0.2` · 🔴 `<0.2`. All timing via `dotnet run -c Release - < script.cs`,
> best-of-rounds, correctness checked before every timed row.

---

## 0. Status snapshot (what is already won)

Delivered this arc (commits `d628c7cc` → `a2b6cba7`):

- **IL scalar cast kernel** (`DirectILKernelGenerator.Cast.Scalar.cs`): emits a direct
  `call Converts.To{Dst}` inner loop per (src,dst) pair — JIT-inlined, **0.99–1.00× of a
  hand-inlined static call**, vs the `Func<>` delegate it replaced which was **1.1–4.8× slower**.
  Drives all 225 pairs through `NpyIterCasting.CopyStridedToStridedWithCast` with the validated
  incremental-coord outer walk.
- **Same-dtype strided copy** (`CopyGeneralSameType`): memcpy-per-row + incremental coord.
- **Result:** every integer/char/bool/complex **widening** cast now beats NumPy (1.5–2.6×);
  sub-word **strided** casts beat NumPy too (the per-element conversion is heavy enough that the
  scalar IL is competitive — the AVX2-gather limitation that hurts elementwise binary ops does
  **not** bite casts).

**Validated:** 1575 addressing checks (15×15 × 7 layouts, view-cast == materialized-cast,
bit-exact) + 672 NumPy edge values incl. complex re+im (NaN/±inf/overflow/wrap/negatives) —
0 diverged. Full suite 9864 pass / 0 fail.

### Measured baseline (1M, NP/NS) — the cells that still lag

| src → dst | C | strided | mechanism gap |
|-----------|---|---------|---------------|
| **f32 → i32** | ✅ 1.92 (`TryGetFloatToInt32Kernel` cvtt) | 🟠 **0.25** | **strided f32→i32 has no cvtt kernel** (only double does) → IL scalar |
| **f64 → i32** | ✅ 1.90 (cvtt) | 🟡 0.78 | partial (`TryGetDoubleToInt32StridedKernel`, unit/reversed/gather only) |
| **f16 → i32** | 🟡 0.67 | 🟡 0.63 | Half→int is scalar (no F16C vector widen) |
| **c128 → i32** | 🟡 0.63 | 🟡 0.60 | Complex→int is scalar (16-byte load + real extract + cvtt) |

Everything else measured (all sources → f64, all int→int, →bool, →char) is ✅. **Phase 0 below
must enumerate the *full* matrix** — `→f16`, `→i8/i16/i64/u*`, `→complex` targets were sampled,
not swept, and are the most likely place additional lagging cells hide.

---

## 1. Definition of Done

A cast execution is "done" when, for the 1M-element benchmark:

1. **Perf:** `NPY/NS ≥ 1.0` on contiguous **and** every non-contiguous layout
   (`F`, `T`, `sliced`, `negrow`, `negcol`, `[:, ::2]` strided-inner, `broadcast`).
2. **Correctness:** bit-exact vs NumPy 2.4.2 (or, for Decimal which has no NumPy dtype, bit-exact
   vs the pre-change `Converts` table). Verified by `StridedCastParityTests` (addressing) + the
   NumPy edge-value harness (semantics).
3. **No regression:** full unit suite green on net8.0 + net10.0; the 10 already-winning SIMD dtypes
   keep their contiguous kernels.

**The project is done when every one of the 15×15×7 ≈ 1575 cells is ✅.**

---

## 2. The cast execution space (taxonomy)

Three orthogonal axes define every execution:

**A. Conversion family** (determines the per-element cost and the SIMD technique):
| Family | Examples | NumPy technique | NumSharp today |
|--------|----------|-----------------|----------------|
| int→int widen | i16→i64 | vector sign/zero-extend | IL scalar (✅ wins) |
| int→int narrow (wrap) | i64→u8 | vector truncate | IL scalar (✅ wins) |
| int→float | i32→f64 | vector cvt | IL scalar (✅ wins) |
| float→float | f32→f64, f64→f32 | vector cvt | IL scalar / SIMD (✅) |
| **float→int (cvtt)** | **f32→i32** | **`vcvttps2dq` + sentinel** | **scalar (🟠 strided)** |
| to/from **Half** | f16→f64, f64→f16 | **F16C `vcvtph2ps`/`vcvtps2ph`** | **scalar (🟡 to-int / to-f16)** |
| to/from **Complex** | c128→f64, f64→c128 | deinterleave / interleave | scalar (✅ →real, 🟡 →int) |
| →bool / →char | f64→bool | compare / truncate | IL scalar (✅ wins) |
| Decimal (no NumPy) | f64→dec | — | IL scalar (n/a) |

**B. Layout** (determines addressing + whether SIMD load is possible):
| Layout | Has stride-1 axis? | SIMD load |
|--------|-------------------|-----------|
| C-contiguous | yes (inner) | contiguous vector |
| F / transpose / sliced / negrow | yes (some axis) | iterator reorders → contiguous inner vector |
| **`[:, ::2]` strided-inner** | **no** | **gather (4/8-byte) or scalar** |
| broadcast (stride-0) | n/a | splat / materialize |

**C. SIMD capability** of the *source* element width:
- **gather-able:** 4-byte (i32,u32,f32) and 8-byte (i64,u64,f64) — `VPGATHERDD`/`VPGATHERQQ`.
- **not gather-able:** 1/2-byte (bool,byte,sbyte,i16,u16,char,Half) — must scalar-walk or
  shuffle-deinterleave the strided-inner case. *(For casts this is a non-issue — see §4.D.)*

---

## 3. The unifying insight

**The three remaining cliffs all bottleneck on the same core — `float→int` via `vcvttps2dq` —
plus a cheap front-end widen:**

```
f32 → i32   =                        cvtt(f32)            ← Cliff 1 (core)
f16 → i32   =   F16C widen(f16→f32) ; cvtt(f32)           ← Cliff 2 = widen + core
c128 → i32  =   deinterleave real(f64) ; cvtt(f64)        ← Cliff 3 = extract + core
```

So the plan is: **build the `float→int` cvtt SIMD kernel once (W1), then feed it from Half (W2,
F16C) and Complex (W3, deinterleave).** Everything else already wins via the IL scalar kernel.
`cvttps2dq`/`cvttpd2dq` return `0x8000_0000` (INT_MIN) for NaN/±inf/overflow — **which is exactly
NumPy's float→int32 sentinel** (this is why the existing contiguous `TryGetFloatToInt32Kernel` is
documented "NumPy-faithful cvtt"). So the core is correct *for free* on `→i32`.

---

## 4. Work items (prioritized)

### Phase 0 — Full-matrix discovery (do first; ~1 script)
Sweep **all 15 src × 15 dst × {C, F, T, sliced, negrow, negcol, `[:, ::2]`, broadcast}** at 1M,
NumPy vs NumSharp, and emit the complete ✅/🟡/🟠/🔴 matrix + a sorted list of every `<1.0` cell.
This converts "the 3 cliffs" into the *authoritative* worklist (the `→f16`, `→i8/i16/i64`,
`→complex` columns are unswept). Harness: extend `benchmark/poc/` (model on the cast matrix already
used this session). **Exit:** a checked-in `cast_matrix.md` that every later phase is measured
against.

### Phase 1 — `float → int32` strided cvtt+gather kernel  *(Cliff 1, widest win)*
**Cells fixed:** `f32→i32` strided (0.25 → ~1.5), `f64→i32` strided (0.78 → ~1.5), and the
front-end for W2/W3.

- **Mechanism:** for the strided-inner layout, `VPGATHERDD` (f32, 4-byte) / `VPGATHERQQ` (f64,
  8-byte) load 8/4 lanes, then `cvttps2dq` / `cvttpd2dq` truncate to i32, store contiguous (output
  of a cast is always a fresh C/F-contig owning array). For contiguous-inner layouts the existing
  contiguous cvtt kernel already wins — only the no-stride-1-axis case needs gather.
- **.NET:** `Avx2.GatherVector256(float*, Vector256<int> idx, 4)`,
  `Sse2/Avx.ConvertToVector*Int32WithTruncation`. Reuse `TryGetGatherSupport` (already gates
  4/8-byte). Generalize the existing `TryGetDoubleToInt32StridedKernel` to a
  `TryGetFloatLikeToInt32StridedKernel` covering both f32 and f64.
- **Plug-in:** `TryGetStridedCastKernel` strategy resolution (it already special-cases
  double→int32 at `Cast.cs:142`). Add the float variant beside it.
- **Tail/OOB:** gather reads whole lanes; the last partial vector + the elements where
  `gatherIdx*stride` could exceed the buffer fall to the scalar IL inner kernel (already exists).
- **Correctness:** cvtt sentinel == NumPy INT_MIN (proven by existing contiguous kernel + the
  672-value harness, which already covers f64→i32 NaN/inf/overflow). Re-run the harness for f32.
- **Expected:** 0.25 → ~1.5× (match/beat NumPy's own cvtt loop).

### Phase 2 — `float → {i8, i16, i64, u8…u64}` (the non-i32 int targets)
**Cells fixed:** any `float→narrow-int` / `float→i64` / `float→unsigned` that Phase 0 flags `<1.0`.

- NumPy float→int8/16 = truncate-to-int then wrap; float→int64 = `cvttpd2qq` (AVX-512) or scalar;
  float→unsigned has its own saturation rules. **These are NOT the i32 cvtt sentinel** — verify
  each against NumPy before SIMD-izing.
- **Cheapest correct path:** `cvtt → i32` (SIMD) then **vector narrow/wrap** to the target width
  for i8/i16; for i64/u64 confirm whether NumPy truncates via 64-bit cvtt and only then SIMD-ize,
  else keep the IL scalar (which already matches NumPy bit-exactly and may already be ✅ — Phase 0
  decides).
- **Also retire the legacy `{i8,i16,i32}→u64` path:** these 3 pairs are the *only* ones still on
  `CastCrossType`'s `DivergesFromNumpyCast` Clone+CastTo branch. Measure in Phase 0; route them
  through `NpyIter.Copy` → IL kernel (already bit-exact via `Converts.ToUInt64`), deleting the
  legacy branch entirely once confirmed ≥ its current speed.
- **Plug-in:** extend the cvtt kernel family; fall through to IL scalar when no faithful SIMD form
  exists (correct, and the IL scalar already wins on many of these).

### Phase 3 — Half SIMD via F16C  *(Cliff 2)*
**Cells fixed:** `f16→i32` (0.63), and any `f16→{f32,f64,int}` / `{f32,f64,int}→f16` Phase 0 flags.

- **Mechanism:** `vcvtph2ps` widens 8 halves → 8 floats; `vcvtps2ph` narrows with round-to-nearest
  (NumPy's float16 rounding). Compose: `f16→i32` = `vcvtph2ps` then `cvttps2dq` (reuses W1);
  `f16→f64` = widen→f32→f64; `f64→f16` = →f32→`vcvtps2ph`.
- **.NET caveat (must verify first):** confirm the F16C intrinsic surface in the target runtime
  (`System.Runtime.Intrinsics.X86`). The **scalar** `(float)half` already JITs to `vcvtph2ps`, but
  the **vectorized** 8-at-once path needs the explicit intrinsic. **If F16C is not publicly
  exposed**, fall back to a vectorized IEEE half↔float **bit-fiddle** over `Vector<ushort>`↔
  `Vector<uint>` (widen exponent/mantissa, branchless subnormal/inf/nan handling — more code, but
  no intrinsic dependency). Gate either path on hardware support; scalar IL remains the fallback.
- **Note:** `f16→f64` already measures **1.00×** via the IL scalar, so the Half wins here are
  mainly `f16→int` and `→f16` (narrowing). Scope to those; don't regress the already-even ones.
- **Rounding correctness:** `vcvtps2ph` round mode must match NumPy (round-half-to-even). Validate
  against NumPy on the full subnormal/overflow/NaN-payload set (reuse the `HalfNegateBitFlipTests`
  sample vector style).

### Phase 4 — Complex deinterleave  *(Cliff 3)*
**Cells fixed:** `c128→i32` (0.60), any `c128→{int}` and `{real}→c128` Phase 0 flags.

- `Complex[N]` is `double[2N]` interleaved `(re,im,re,im…)`. `c128→real` = deinterleave the even
  lane (`Avx.UnpackLow/Permute` on pairs) → reals vector; `c128→int` = deinterleave real then cvtt
  (reuses W1). `real→c128` = interleave `(x, 0)`.
- **`c128→f64` already wins (1.91×)** via IL scalar (`ldobj Complex` + `c.Real`) — only `c128→int`
  needs the deinterleave+cvtt. Keep scope tight.
- **Semantics:** `c→real` drops imaginary (NumPy ComplexWarning); `c→bool` is `re!=0 || im!=0`
  (do NOT route bool through the real-only deinterleave — keep it on the IL scalar). Already pinned
  by `StridedCastParityTests.StridedCast_ComplexEdges_MatchNumPy`.

### Phase 5 — Sub-word strided & broadcast (verify, likely already ✅)
- **Sub-word strided casts** (e.g. `char[:, ::2]→f64`) already measure **1.85×** — the heavy
  per-element conversion makes the scalar IL competitive, so the AVX2-gather-is-4/8-byte-only
  limitation (which cripples elementwise binary ops, see `DirectILKernelGenerator.InnerLoop`
  `TryGetGatherSupport`) is **a non-issue for casts**. Phase 0 confirms; if any sub-word cast cell
  is `<1.0`, only then consider shuffle-deinterleave.
- **Broadcast-source casts** (stride-0): materialize-once-then-cast, or cast-then-broadcast. Verify
  Phase 0; the incremental-coord walk already handles stride-0 (the inner kernel re-reads the same
  element), which is correct but may redo work — if flagged, cast the unique row once.

---

## 5. Routing map (where each kernel plugs in)

```
astype(nd, dtype)                                  APIs → DefaultEngine.Cast
  └─ CastCrossType (Default.Cast.cs)
       ├─ DivergesFromNumpyCast?  ── true ONLY for {i8,i16,i32}→u64 (3 pairs) ─────────┐
       │     → legacy Clone+Array.CastTo  (audit in Phase 0; route to IL kernel)       │
       └─ NpyIter.Copy (cross-dtype)   [all other pairs, incl. ALL float→int]          │
            ├─ IsContiguousCopy → TryGetCastKernel        ← TryGetFloatToInt32Kernel    │
            │                                               (cvtt contig, DONE) + SIMD  │
            ├─ TryGetStridedCastKernel                    ← TryGetDoubleToInt32Strided  │
            │                                               (DONE); ADD float (W1),     │
            │                                               F16C (W3), deinterleave (W4)│
            └─ CopyStridedToStridedWithCast               ← IL scalar kernel (DONE)  ◄──┘
                 └─ TryGetInnerCastKernel  (call Converts.ToX, all 225 pairs)
```

**Precise gate facts:** `DivergesFromNumpyCast(src,dst)` returns true **only** for
`{SByte,Int16,Int32} → UInt64` — *not* for float→int. So every float→int already flows through
`NpyIter.Copy`: contiguous f32/f64→i32 hit the cvtt `TryGetFloatToInt32Kernel` (✅ 1.9×); strided
f64→i32 hits `TryGetDoubleToInt32StridedKernel` (🟡 0.78); strided **f32→i32 has no kernel** and
lands on the IL scalar (🟠 0.25). The 3 signed→u64 pairs are the only ones on the legacy
Clone+CastTo path (Phase 0 measures them; route to the IL kernel if lagging).

**Design rule:** each new SIMD kernel is added to the `TryGetStridedCastKernel` /
`TryGetCastKernel` strategy resolution and returns `null` when it cannot be NumPy-faithful — the
chain then falls to the **IL scalar kernel**, which is already correct and already beats NumPy on
most cells. **No cast ever falls below the IL scalar floor** (measured: for f32→i32 strided the IL
floor 1.07× the legacy Clone+CastTo, so the legacy branch is safe to retire).

---

## 6. Validation & benchmark gates (every phase)

1. **Addressing:** `StridedCastParityTests` — every 15×15 pair × {strided,sliced,negrow,negcol,F,T}
   view-cast == materialized-cast, bit-exact. (Already green; extend layouts if a phase adds a
   path.)
2. **Semantics vs NumPy:** the edge-value harness — int/float/complex sources with
   {0, ±1, max, min, 2^k boundaries, NaN, ±inf, overflow, subnormal} × all targets, on a strided
   view, diffed against NumPy 2.4.2 (re+im for complex). **0 diverge** is the gate.
3. **Perf:** the Phase-0 full matrix re-run; **no cell may regress** and the targeted cells must
   reach ✅. Geomean per dtype tracked in `cast_matrix.md`.
4. **Suite:** `dotnet test` net8.0 + net10.0, `TestCategory!=OpenBugs&!=HighMemory`, 0 fail.

---

## 7. Correctness invariants (must hold across all SIMD kernels)

- **float→int32 sentinel:** NaN/±inf/overflow → `INT_MIN` (= `cvtt` hardware result). Verified.
- **float→int (other widths):** NumPy truncates toward zero then wraps modulo the width — confirm
  per target; do **not** assume saturation.
- **signed→unsigned / narrowing:** modular wrap (`-1→255`), not clamp.
- **Half rounding:** round-half-to-even on `vcvtps2ph`; subnormal flush matches NumPy; NaN sign +
  payload preserved.
- **Complex→real:** drop imaginary; **Complex→bool:** truthy on either part (separate path).
- **Decimal:** no NumPy dtype — pin against the `Converts` table (self-consistent), never regress.
- **Fallback chain is always correct:** SIMD kernel `null` → IL scalar (`Converts.To{Dst}`) →
  `ConvertValue` (IL-disabled). All three share the `Converts` table, so they are bit-identical.

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| F16C not in public .NET API for target TFM | Vectorized IEEE bit-fiddle fallback over `Vector<ushort>`/`Vector<uint>`; scalar IL floor regardless. |
| Gather OOB read at strided tail | Gather only the in-bounds bulk; scalar-IL the tail (the existing inner kernel). |
| cvtt semantics differ for non-i32 targets | Verify each target vs NumPy in Phase 0/2; keep IL scalar where no faithful SIMD exists. |
| Hand-written IL/intrinsic bugs | Every kernel gated by the addressing + NumPy-edge harness *before* it ships; kernel returns `null` on emit failure → safe fallback. |
| Concurrent reduction work on branch | Cast files are disjoint from reduction files; rebase-clean. |

---

## 9. Sequencing & exit criteria

| Phase | Deliverable | Exit |
|-------|-------------|------|
| 0 | `cast_matrix.md` full sweep + lagging-cell worklist | every `<1.0` cell enumerated |
| 1 | `float→int32` strided cvtt+gather kernel | f32/f64→i32 strided ✅; harness 0-diff |
| 2 | `float→{i8,i16,i64,u*}` faithful SIMD or confirmed IL-✅ | all float→int cells ✅ |
| 3 | Half F16C (or bit-fiddle) widen/narrow | all f16↔ cells ✅; rounding 0-diff |
| 4 | Complex deinterleave→cvtt | all c128→int cells ✅ |
| 5 | Sub-word/broadcast verify | matrix 100% ✅ |

**Project exit:** the full 15×15×7 matrix is **100% ✅**, both TFMs green, both harnesses 0-diff.
At that point NumSharp beats NumPy at **every** cast execution.

---

## 10. Appendix — key code references

| Component | Location |
|-----------|----------|
| IL scalar cast kernel (DONE) | `Backends/Kernels/Direct/DirectILKernelGenerator.Cast.Scalar.cs` |
| Strided/contig SIMD cast + strategy | `…/DirectILKernelGenerator.Cast.cs` (`TryGetStridedCastKernel`, `TryGetFloatToInt32Kernel`, `TryGetDoubleToInt32StridedKernel`, `DivergesFromNumpyCast`) |
| Gather support (4/8-byte gate) | `…/DirectILKernelGenerator.InnerLoop.cs` (`TryGetGatherSupport`) |
| Typed load/store IL helpers | `…/DirectILKernelGenerator.cs` (`EmitLoadIndirect`, `EmitStoreIndirect`, `GetTypeSize`, `GetClrType`) |
| Cross-dtype strided driver | `Backends/Iterators/NpyIterCasting.cs` (`CopyStridedToStridedWithCast`, `CastStridedScalar`, `ConvertValue`) |
| astype routing | `Backends/Default/ArrayManipulation/Default.Cast.cs` (`CastCrossType`) |
| Bit-exact conversion table | `Utilities/Converts.cs` (all 225 `To{Dst}({Src})`) |
| Parity tests | `test/…/Backends/Kernels/StridedCastParityTests.cs`, `StridedCopySameTypeParityTests.cs` |
| Same-type copy (DONE) | `…/DirectILKernelGenerator.Copy.cs` (`CopyGeneralSameType`) |
