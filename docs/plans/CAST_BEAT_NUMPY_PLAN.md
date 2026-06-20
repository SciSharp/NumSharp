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

## 0.1 Phase 0 RESULTS (measured 2026-06-20) — the matrix reprioritizes the plan

Full sweep checked in: **`benchmark/cast/cast_results.md`** (15 src × 8 layouts × 15 dst, 1M,
NumPy 2.4.2; the `benchmark/cast` subsystem — `cast_matrix_bench.{cs,py}` + `cast_sheet.py`). **716 / 1568
comparable cells lag (<1.0); 852 win.** The sweep **overturns the §3 framing**: `float→i32` is
*already won* (contiguous cvtt kernel: f32→i32 **1.69**, f64→i32 **1.56**) — only **strided**
`f32→i32` (**0.24**, 4 cells) remains of old "Cliff 1". The real fire is one family:

| Severity | Cells | Dominant family |
|----------|-------|-----------------|
| 🔴 `<0.2` | 46 | **45 = `float/cplx → narrow-int` (→bool/u8/i8/i16/u16/char)** + 1 bcast u8→u8 |
| 🟠 `0.2–0.5` | 120 | 72 = float/cplx→narrow · 12 = `*→bool` · 9 = same-type diag · 8 = bool→f16 · **4 = f32→i32 strided** |
| 🟡 `0.5–1.0` | 550 | 160 = `int→sub-word (narrow)` · 67 = `*→bool` · 66 = float/cplx→narrow |

**`float/complex → narrow-int` geomean by src: f32→narrow `0.21`, c128→narrow `0.40`, f64→narrow
`0.38`, f16→narrow `0.48`.** f32→i8 bottoms the whole matrix at **0.09** (10.8× slower: 2.6 ms vs
0.24 ms) — no SIMD kernel exists for narrowing float→sub-word, so it falls to the IL scalar
(`Converts.ToSByte((float)x)` per element). **NumPy does *not* vectorize these casts either** —
its generic cast loop is scalar-speed (measured 4M: f32→i8 **1.31 ms**, f64→i32 **3.26 ms**), which
is precisely the headroom: a SIMD `cvtt`+narrow kernel beats NumPy outright.

> **⚠ Correction (proven 2026-06-20, supersedes earlier framing):** the back-end is **NOT a
> saturating pack** (`vpackssdw`/`vpackuswb`). NumPy float→narrow-int **WRAPS** (low bits), it does
> **not** saturate — oracle: f32→i8 `128.5 → -128` (wrap), a saturating pack gives `+127` (wrong).
> Benchmarked head-to-head at 4M: a `cvtt`+**saturate-pack** kernel produced **3.47 M diffs** vs
> NumPy. The correct primitive is `cvtt` + **truncating narrow** = mask-to-width (`&0xFF`/`&0xFFFF`)
> + unsigned-pack (saturation becomes a *no-op* because masked values are already in-range) +
> cross-lane permute. See §0.2 for the measured shootout.

### Reprioritized phase order (supersedes §4 sequencing)

1. **P1′ (was P2) — `float/complex → narrow-int` SIMD `cvtt + truncating-narrow`.** *Highest value:
   45/46 reds, ~183 cells.* **Mechanism proven in §0.2** (f32/f64 source already benchmarked
   3.4–4.1× / 1.9–2.0× vs NumPy, 0-diff): `f32→i8` = `cvttps2dq` (W1 core) **then mask-to-width +
   unsigned-pack + per-width permute** (`vpermd` for 8-bit, `vpermq` for 16-bit) — **NOT** a
   saturating pack. Same cvtt core; the missing piece is the truncating narrow, now designed and
   measured. f16 front-ends via F16C/bit-fiddle (P3 dep), c128 via deinterleave (P4 dep) — so build
   the f32/f64→narrow kernel first (the §0.2 prototypes drop straight into the kernel family).
2. **P2′ (was P1) — strided `f32→i32` cvtt+gather.** Now only ~4 amber cells (contiguous won).
3. **`*→bool`** (≈79 cells, mild 🟡): `v != 0` vector compare → packed bool. Cheap, broad.
4. **`int→sub-word (narrow)`** (160 🟡): int→{u8/i8/i16/u16/char} vector pack (no cvtt — direct
   truncate/pack). Mild lag; vector pack closes it.
5. **same-type 1-byte copy** (u8/i8/bool diagonal 0.23–0.31; `bcast u8→u8` 0.15🔴): the contiguous
   `astype(copy)` same-type path isn't hitting cpblk for 1-byte at 1M — audit the routing.
6. **f16 (F16C/bit-fiddle)** and **c128 (deinterleave)** front-ends feed P1′ for their narrow targets.

The unifying core (`cvtt`) is unchanged — but the **truncating-narrow back-end** (int32 → low
bytes of int16/int8) is the new shared primitive every narrowing cast needs, and is built once
in P1′. It is a **mask + unsigned-pack + permute**, NOT a saturating pack (see §0.2).

---

## 0.2 float→narrow-int implementation shootout (PROVEN, 2026-06-20)

Five implementations of the worst cell (`f32→i8`, 4M, best-of-7) benchmarked for **both**
correctness (vs the NumPy-faithful `Converts` scalar, which the parity suite pins to NumPy 2.4.2)
**and** speed. Harness: `/tmp/cast_f32i8*.cs` style (reproducible; uses `Avx`/`Avx2` intrinsics,
`#:project NumSharp.Core`). This is the empirical proof the SIMD back-end is sound:

| # | Implementation | ms | ×scalar | diffs vs NumPy | verdict |
|---|----------------|----|---------|----------------|---------|
| V1 | scalar `Converts.ToSByte((float)x)` (current IL kernel) | 10.9 | 1.0 | 0 (baseline) | correct, **the 0.09× cliff** |
| V2 | `cvttps2dq` (SIMD) + **scalar** low-byte narrow | 1.52 | 7.0 | 0 | correct; cvtt alone clears the cliff |
| V3 | `cvttps2dq` + `&0xFF` + 2×`vpackuswb` + **`vpermd`** | 0.38 | 18–29 | **0** | **correct + production** |
| V5 | `cvttps2dq` + `vpshufb` low-byte gather + 64-bit moves | 0.58 | 19 | 0 | correct alt |
| V4 | `cvttps2dq` + **saturating** `vpackssdw`/`vpacksswb` | 0.32 | 34 | **3.47 M** | **WRONG** (saturates `128.5→127`) |

**Per-dtype optimization is real and measured** (each target validated 0-diff vs NumPy, NPY/NS):

| cast | kernel (per width) | NS ms (4M) | NumPy ms | NPY/NS |
|------|--------------------|-----------|----------|--------|
| f32→i8  | `&0xFF`  + 2×`vpackuswb` + **`vpermd`** | 0.378 | 1.305 | **3.45×** |
| f32→u8  | *(same kernel — bit-identical)*        | 0.378 | 1.565 | **4.14×** |
| f32→i16 | `&0xFFFF` + 1×`vpackusdw` + **`vpermq 0xD8`** *(cheaper)* | 0.595 | 2.110 | **3.55×** |
| f32→u16 | *(same kernel — bit-identical)*        | 0.595 | 2.008 | **3.37×** |
| f64→i32 | `cvttpd2dq` + store *(no narrow)*       | 1.750 | 3.256 | **1.86×** |
| f64→i16 | `cvttpd2dq` + 128-bit `vpackusdw` *(no lane cross)* | 1.321 | 2.633 | **1.99×** |

**Findings that drive the implementation:**
- **`cvtt` is the engine; the narrow is cheap.** Even V2 (scalar narrow) is 7× — but only the full
  SIMD narrow (V3) beats NumPy with margin. The cliff is the *absence* of `cvtt`, not the narrow.
- **8-bit vs 16-bit need different lane fixups** (the per-dtype win): the 2-level pack for i8/u8
  interleaves at 4-byte granularity → needs **`vpermd`** (`[0,4,1,5,2,6,3,7]`); the 1-level pack
  for i16/u16 interleaves at 8-byte granularity → needs only **`vpermq 0xD8`** (cheaper cross-lane
  op). `Permute4x64` on the 2-level result is *insufficient* (1.99 M diffs — it cannot split a
  quadword that holds two operands' bytes).
- **i8≡u8 and i16≡u16 share one kernel** (low-bytes are bit-identical; only the C# element type
  differs) → 4 targets, 2 kernels.
- **f64 source** uses `cvttpd2dq` (4 doubles→4 i32, INT_MIN sentinel for out-of-i32 range — matches
  NumPy: `3e9→i16→0` is low-16 of INT_MIN); its narrow uses **128-bit** packs (no lane crossing).
- **Memory-bound ceiling:** f32→i8 writes 4 MB + reads 16 MB; at ~0.38 ms that is ~52 GB/s — near
  the bandwidth floor, so further kernel tuning yields little. The win is already captured.

**The saturate-pack trap (V4) is the single most important correction to this plan** — it is the
fastest *and wrongest* option; any implementer reaching for the "obvious" `vpackss` must use the
mask+`vpackus` truncate instead.

---

## 0.3 ALL remaining cliff families — PROVEN (2026-06-20)

Every lagging family the Phase-0 matrix flagged has now been prototyped, **correctness-gated
0-diff** vs the NumPy-faithful `Converts` scalar (which the parity suite pins to NumPy 2.4.2), and
timed at 4M best-of-7. Harnesses checked in under `benchmark/poc/cast_*.cs`. **Result: a SIMD
kernel beats NumPy on every family** — NumPy does not vectorize any of these casts (all its
baselines are scalar-speed, the headroom).

| Family | Technique (new primitive) | cast | NS ms | NumPy ms | NPY/NS | diffs |
|--------|---------------------------|------|-------|----------|--------|-------|
| **float→narrow** (§0.2) | `cvtt` + mask + `vpackus` + `vpermd`/`vpermq` | f32→i8 | 0.33 | 1.31 | **3.95** | 0 |
| | | f32→i16 | 0.42 | 2.11 | **5.04** | 0 |
| | | f64→i32 | 1.56 | 3.26 | **2.09** | 0 |
| **complex→int** | **deinterleave** (`vunpcklpd`+`vpermq`) + `cvtt` | c128→i32 | 3.24 | 5.03 | **1.55** | 0 |
| `benchmark/poc/cast_complex_deinterleave.cs` | | c128→i8 | 2.51 | 3.46 | **1.38** | 0 |
| **int→sub-word narrow** | mask + `vpackus` + permute (**no cvtt**) | i32→i8 | 0.34 | 1.24 | **3.69** | 0 |
| `benchmark/poc/cast_int_narrow.cs` | + i64→i32 **dword-extract** (`vpermd [0,2,4,6]`) | i32→i16 | 0.44 | 1.61 | **3.67** | 0 |
| | | i64→i32 | 1.48 | 3.47 | **2.35** | 0 |
| | | i64→i16 | 1.43 | 2.36 | **1.65** | 0 |
| **\*→bool** | `~CompareEqual(v,0) & 1` (`AndNot`) + narrow | f32→bool | 0.33 | 1.93 | **5.89** | 0 |
| `benchmark/poc/cast_to_bool.cs` | IEEE: `-0.0→False`, `NaN→True` (Ordered eq) | f64→bool | 1.02 | 2.36 | **2.31** | 0 |
| | | i32→bool | 0.34 | 1.14 | **3.36** | 0 |
| **f16→narrow** | **Giesen** half→float bit-fiddle (F16C-free) + `cvtt` | f16→i32 | 2.76 | 5.10 | **1.85** | 0 |
| `benchmark/poc/cast_half_giesen.cs` | `vpmovzxwd` + magic-mul + inf/nan `cmpgt` | f16→i8 | 2.85 | 4.04 | **1.42** | 0 |
| | | f16→f32 | 2.69 | 4.12 | **1.53** | 0* |

**Technique notes (each a reusable kernel primitive):**
- **Complex deinterleave:** `Complex[N]` is `double[2N]` `(re,im,…)`; `vunpcklpd(a,b)` + `vpermq 0xD8`
  extracts 4 contiguous reals → `cvttpd2dq`. `c128→narrow` = deinterleave then the §0.2 narrow.
  Modest margin (1.4×) because 16-byte elements are memory-bound (64 MB read at 4M).
- **int→narrow = §0.2 minus the cvtt** (input is already int). i64 sources add a **dword-extract**
  front-end: `vpermd [0,2,4,6]` picks each long's low dword → i32, then the standard narrow.
- **\*→bool:** `boolmask = AndNot(CompareEqual(v,0), one)` → 0/1 ints, then the §0.2 narrow to bytes.
  Float uses **OrderedEqualNonSignaling** so `-0.0` equals `0` (→False) and `NaN` is unordered
  (→True) — bit-exact NumPy `!= 0`. f64→bool needs the i64→i32 dword-extract (8:1 narrow).
- **f16 (no F16C in this .NET — verified: `Avx512FP16` absent, no `Vector256<Half>`, no vectorized
  `ConvertToSingle`):** use **Giesen's** branchless half→float (`expmant<<13` × magic reconstructs
  normals *and* subnormals exactly; `cmpgt` injects inf/nan exponent) over `vpmovzxwd`-widened
  halves, then `cvtt`. **Scalar `(float)half` is a trap** — measured *slower* than the current
  scalar (10.9 ms, it double-converts). `*` f16→f32 has **NaN-payload-only** diffs (Giesen's NaN
  bits ≠ BCL's; both are NaN) — irrelevant for →int (NaN→`cvtt`→INT_MIN→0), but f16→f32 should
  stay on the already-even (1.00×) scalar path **or** get a NaN-canonicalize before it ships as a
  user-facing float cast.

**Net:** all five flagged families (≈1568-cell matrix's entire `<1.0` population except the
1-byte same-type copy / `bcast u8→u8` routing issue, §0.4) have a proven, correctness-gated SIMD
design that beats NumPy. The implementation is now de-risked end-to-end; what remains is wiring
these prototypes into `TryGetStridedCastKernel`/`TryGetCastKernel` (§5 routing).

---

## 0.4 Last cliff — same-type 1-byte copy & broadcast: ROOT CAUSE + FIX (PROVEN 2026-06-20)

The only non-SIMD-kernel `<1.0` cells. **Not a missing kernel — a routing/overhead issue.** The
same-type `astype(copy)` path is `Cast` → `nd.Clone()` → `Storage.Clone()` →
{`InternalArray.Clone()` for raw-layout | `CloneData()`→`NpyIter.Copy` for broadcast/sliced}.
Harnesses: `benchmark/poc/cast_sametype_breakdown.cs`, `cast_bcast_fill.cs`.

**Decomposition (proven by component breakdown, 1M u8):**

| Component | ms (warm) | finding |
|-----------|-----------|---------|
| `Buffer.MemoryCopy` 1 MB | 0.0142 | ≈ NumPy's own 0.015 — **the copy itself is already at parity** |
| pool `Take(1MB)` (warm) | 0.0001 | free |
| `new Shape(_shape)` | ~0 | free |
| ideal clone (Take+Copy+Return) | 0.0159 | ≈ NumPy |
| **full `astype(same)` (warm)** | **0.034** | +0.018 ms = **~5 managed-object allocs** (`UnmanagedMemoryBlock`+`Disposer`+`Storage`+`Shape`+`NDArray`) |

So the contiguous cliff is **two small, size-independent taxes**, only visible when the copy is
tiny: (a) **buffer-pool cold-start** — a *fresh* 1 MB+ buffer faults on first touch (a best-of-N
loop allocating a new output each rep hits cold buffers → the 0.15 ms "10×" the Phase-0 matrix saw;
warm steady-state is 0.034 ms); (b) **~5 managed-object allocations** per clone (NumPy returns one
C struct). At **4M these vanish — same-type 1-byte already WINS** (u8→u8 1.20×, bool→bool 3.87×,
i8→i8 2.95×). **Verdict: low priority** — the copy is at parity; only the per-call tax lags, and
only sub-0.05 ms. Optional micro-fix: a same-type contiguous clone fast-path that takes the pool
buffer + `Buffer.MemoryCopy` + minimal wrapper, skipping the general `Storage.Clone` machinery.

**The broadcast cell is the real, cleanly-fixable one.** `bcast u8→u8` routes a *broadcast source*
(stride-0) through `CloneData()` → `NpyIter.Copy` (general strided iterator) to materialize — but a
same-type broadcast clone is just a **fill** (replicate the scalar / tile the broadcast row).
**Proven** (`InitBlock` memset for the scalar-broadcast case):

| N | current (`NpyIter.Copy`) | direct fill (`InitBlock`) | speedup | NPY/NS current → fill |
|---|--------------------------|---------------------------|---------|------------------------|
| 1M | 0.215 ms | 0.035 ms | **6.1×** | 0.06 → 0.34 |
| 4M | 0.859 ms | 0.126 ms | **6.8×** | 0.83 → **5.69** |

At 4M the fill (32 GB/s) crushes NumPy's own broadcast-materialize (5.6 GB/s); at 1M NumPy's
0.012 ms is cache-resident (near-unbeatable at 1 MB) but the fill is still 6× the current path.

**Fix (broadcast):** in `UnmanagedStorage.CloneData()`, before the `NpyIter.Copy` fallback, detect
`_shape.IsBroadcasted` **and** same-type and dispatch a **broadcast-aware fill**:
- scalar-broadcast (all strides 0) → `Unsafe.InitBlockUnaligned` per element-width (1/2/4/8-byte
  splat; for >1-byte, fill one element then exponentially double-copy, or splat a `Vector<T>`).
- partial/row broadcast (some stride 0) → materialize the non-broadcast tile once, then
  exponentially double-copy (`memcpy` the filled prefix to itself) to cover N — O(N) bytes, one
  pass, no per-element iteration.
Keep `NpyIter.Copy` as the fallback for the genuinely strided (non-broadcast) sliced case.

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
used this session). **Exit:** a checked-in `benchmark/cast/cast_results.md` that every later phase is measured
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
- **Cheapest correct path (PROVEN — §0.2):** `cvtt → i32` (SIMD) then **truncating narrow** =
  mask-to-width + unsigned-pack + `vpermd`(8-bit)/`vpermq`(16-bit). f32→{i8,u8,i16,u16} and
  f64→{i32,i16} are already benchmarked 0-diff and 1.9–4.1× vs NumPy; these prototypes are the
  kernel bodies. For i64/u64 confirm whether NumPy truncates via 64-bit cvtt (`cvttpd2qq` is
  AVX-512 only — **unavailable here**, Avx512F=False) and only then SIMD-ize, else keep the IL
  scalar (which already matches NumPy bit-exactly and may already be ✅ — Phase 0 decides).
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
- **Same-type 1-byte copy & broadcast (ROOT-CAUSED — see §0.4):** the diagonal 1-byte cells are
  per-call overhead (cold pool + ~5 object allocs), invisible at 4M (already ✅); low priority. The
  `bcast u8→u8` 🔴 is the real fix: route same-type broadcast clones through a direct fill
  (`InitBlock`/tiled-copy) in `UnmanagedStorage.CloneData()` instead of `NpyIter.Copy` — **proven
  6–8× (4M: 0.83→5.69 NPY/NS)**.

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
   reach ✅. Geomean per dtype tracked in `benchmark/cast/cast_results.md`.
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
| 0 | `benchmark/cast/cast_results.md` full sweep + lagging-cell worklist | every `<1.0` cell enumerated |
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
