# Cast Matrix — Continuation Plan for the Remaining 133 Cells

**Continuation of** `docs/plans/CAST_BEAT_NUMPY_PLAN.md` (see its §11 for the Wave-15 history).
**Scoreboard:** `benchmark/cast/cast_results.md` (full per-cell worklist) + `cast_results.tsv`.
**Convention:** ratio = `NumPy_ms / NumSharp_ms`, **>1.0 = NumSharp faster** (higher is better).

---

## 0. Where we are

> **UPDATE — Wave 16 (done): §2 is CLOSED.** The "one large SIMD-shuffle opportunity" below
> (sub-word `strided`/`negcol`) was implemented and shipped — `DirectILKernelGenerator.Cast.SubwordCopy.cs`
> (same-type + same-size bit-reinterpret copies) and `.Cast.SubwordNarrow.cs` (2B-int → {1B,bool}
> narrow), via VPACKUS deinterleave / VPSHUFB reverse lane shuffles (NO gather, NO staging). All ~65
> sub-word strided/negcol/sliced/negrow cells now win **1.1–7.9×** (best-of-7), bit-exact vs NumPy
> (208/208 hashes), full suite green. See `CAST_BEAT_NUMPY_PLAN.md` §12. The §6 noise hypothesis was
> also **confirmed**: clean best-of-7 spot checks show most borderline 0.8–0.89 large-type cells
> (`f32/f64/c128 → i64/u64`) measure ≥0.93–0.99 → true ≥0.9 is **~96%**, not the sweep's 91–93%.
> **Remaining real laggards:** §3 (alloc-bound contig, the lone 🔴 `bool|F|bool`), §4 (`i64/u64→f16`,
> AVX512-gated), §5 (`i64/u64→narrow` strided, memory-bound), and a couple genuine large-type cells
> (`c128|C|u64` 0.79, `c128|strided|u64` 0.87).

At the time of writing the cast matrix was **1435 / 1568 cells ≥ 0.9 (91.5%)**, 1351 (86.2%) win ≥ 1.0,
overall geomean **1.70**. Waves 15–15e closed the float/complex→unsigned cliff and the f16
laggards on AVX2. **133 cells remained < 0.9.** This document enumerates them by *root cause*,
states whether each is addressable, and sketches the approach where one exists.

The headline: **every family that was tractable as an AVX2 cast kernel is done.** What's left
splits into (a) best-of-3 measurement noise, (b) genuinely hardware-blocked families needing
AVX512 or an allocator change, and (c) ~~one large SIMD-shuffle opportunity (stride-2 / negcol
of sub-word types)~~ **— DONE in Wave 16.**

### The 133, by family

| # | Family | Ratio | Class |
|---|--------|-------|-------|
| 40 | 2-byte/1-byte → narrow **strided** (`i16/u16/char/u8/i8/bool → narrow`, `strided`+`negcol`) | 0.56–0.90 | 🟠 partly tractable (§2) |
| ~30 | scattered / noise (`f32/f64/c128→i64` contig, `float→narrow` contig, some `→u64`) | 0.72–0.89 | 🔵 noise (§6) |
| 16 | `i64/u64 → f16` | 0.67–0.89 | 🚫 AVX512-gated (§4) |
| ~12 | `float/c128 → u64` negcol/strided (real subset) | 0.70–0.90 | 🟡 memory-bound (§5) |
| 15 | same-type copy (`x|strided|x` + alloc-bound `x|C/F/T|x`) | 0.17–0.90 | 🚫/🟠 split (§2, §3) |
| 8 | `i64/u64 → narrow` **strided** | 0.83–0.90 | 🟠 memory-bound (§5) |
| 5 | `2-byte → bool` strided | 0.61–0.84 | 🚫 non-gatherable (§2) |

(Counts overlap the buckets in §1's analysis script; treat them as approximate — best-of-3
reshuffles the borderline cells run-to-run.)

---

## 1. READ THIS FIRST — methodology & hard-won gotchas

Everything below was learned the slow way during Waves 11–15. Re-learning it costs hours.

### Measuring
- **best-of-3 is NOISY** (the full-sweep harness). Individual cells swing ±0.2–0.5 and dip
  in/out of `<0.9` run-to-run. **Always reliably re-measure a target cell with best-of-7
  back-to-back** before believing it lags, and again after a change. ~30–40 of the 133 are
  noise — they measure ≥0.9 under best-of-7.
- Timing scripts **must** run `dotnet run -c Release - < script.cs` (Debug taints hand-written
  SIMD ~2×; see `benchmark/CLAUDE.md`).
- Correctness gate ≠ value check: the sweep's gate only checks `.size`. Bit-exactness is a
  **separate** obligation — test each kernel against NumPy directly (see "Testing" below).

### Running file-based scripts (Windows)
- **`dotnet run <file.cs>` with a multi-target `#:project` reference HANGS** (multi-TFM
  ambiguity). Use the **stdin form from the repo root**: `cd <repo> && dotnet run -c Release - < /path/script.cs`. Standalone scripts (no `#:project`) run fine as `dotnet run file.cs`.
- **bash `/tmp` ≠ .NET `/tmp`.** Git-bash `/tmp` is MSYS temp; .NET resolves `/tmp/x` to
  `K:\tmp\x`. For any file shared between a Python and a C# process, use an **absolute Windows
  path** (e.g. `K:/source/NumSharp/benchmark/poc/_scratch/...`), not `/tmp`.
- `cast_sheet.py` pipes the `.cs` via UTF-8 now (fixed in `bench_common.py`); the bench `.cs`
  has non-ASCII (`× +→`) in comments, so any new piping must set `encoding="utf-8"`.

### Testing (the bit-exactness bar)
- Build the kernel's math **standalone first** (pure intrinsics, no project ref) and prove it
  0-diff vs a scalar reference over a large random+edge sample (and **exhaustively** when the
  domain is small — f16 has only 65536 values; test all of them).
- Then validate the **integrated `astype` path** vs NumPy across **all 8 layouts** (C, F, T,
  sliced, negrow, negcol, strided, bcast) by sharing a binary input file and comparing bits.
- The scalar **tail and the strided `conv`** must use the same NumPy-faithful reference as the
  SIMD bulk (`Converts.To{X}`), or bulk/tail diverge on edge values.

### SIMD facts that gate the whole problem
- **`VPGATHER` is 4/8-byte only.** 2-byte and 1-byte sources are **not gatherable** — this is
  *the* reason the 40-cell family is hard.
- **AVX512 `cvttps2udq`/`cvttpd2udq` SATURATE**, which does *not* match NumPy's modular wrap
  (`5e9 → 705032704`, not `0xFFFFFFFF`). Even on AVX512 the faithful path is `cvttpd2qq → narrow`.
  Don't "fix" u32/u64 by reaching for those instructions.
- **BCL `(float)Half` / `(Half)x` quiet signaling NaNs**; NumPy doesn't. Any Half path must use
  the bit-fiddle (Giesen widen / round-to-odd narrow), not the BCL cast.
- **Gather→convert stalls** on gather latency when the convert is a long dependency chain.
  **Stage** (gather → contig buffer → convert) to let gathers pipeline — 2.5× for f32→u32
  strided. BUT staging **regresses a cheap convert** (the 1-cycle truncate of 2-byte→narrow):
  the staging copy costs more than the narrow it feeds. Stage only heavy converts.
- This machine is **AVX2-only** (no AVX512F/DQ/BW; AVX2 + F16C-absent-in-.NET, 32 cores).

---

## 2. 🟠 2-byte/1-byte → narrow strided (40) + same-type strided (9) + 2-byte→bool (5) — the big SIMD-shuffle opportunity

**Cells:** `{i16,u16,char,u8,i8,bool} → {bool,u8,i8,i16,u16,char}` in the `strided` and `negcol`
blocks (e.g. `i16|strided|i8` 0.85, `char|strided|char` 0.59, `u16|strided|bool` 0.67,
`i16|negcol|bool`), plus same-type `x|strided|x` (`u8` 0.56, `i8` 0.53, `i16` 0.59, `char` 0.59,
`f16` 0.55) and `x|negcol|x`.

**Root cause:** sub-word source, non-unit inner stride → no `VPGATHER`, so the generic emitter
falls to a scalar inner loop. It's pure strided-read-bound; the convert (truncate/narrow/copy)
is ~free. NumPy's tight C strided loop wins by ~1.3–1.8×.

**Why the obvious fixes fail:** staging strided→contig buffer is itself the scalar strided copy
that's the bottleneck (Wave 13 measured a regression). General `VPGATHER` doesn't exist for 2B.

**The opportunity — specialise the two strided layouts that dominate the sweep:**
1. **`strided` = `[:, ::2]` (inner stride 2).** The wanted elements are *contiguous with gaps*:
   load 2 full vectors (32× i16, or 32 bytes), **deinterleave the even lanes** with `VPSHUFB`
   (byte-level, within 128-bit lanes) + `VPERMQ`/`PERM2I128` to repack across lanes, then narrow.
   This is the i16/byte analogue of the f32-stride-2 permute that measured **0.21 ms** vs gather's
   0.69 (see `/tmp/f32strided_variants.cs` history). Covers ~half the family.
2. **`negcol` = `[:, ::-1]` (inner stride −1).** The data is **contiguous, just reversed**: load
   forward `Vector256<short>`/`<byte>`, reverse lanes with `VPSHUFB`+`VPERMQ`, narrow, store.
   No gather. The pattern already exists for doubles in `CastDoubleToInt32Strided` (the `ss==-1`
   branch) — generalise it to 2-byte/1-byte and to same-type copy.
3. General strides (ss = 3, 5, …) and same-type: stride-2/negcol specialisations apply; the rest
   stay scalar (rare).

**Same-type strided copy** is the *easiest* sub-case (no convert at all — just the
deinterleave/reverse + store). Do it first as the proving ground, then add the narrow.

**Effort/payoff:** Medium-high effort (cross-lane AVX2 shuffles, per-width `VPSHUFB` masks for
1B/2B, separate even/odd and reverse paths). Highest cell count (~30–40 reachable of the ~54).
Owner file: extend `DirectILKernelGenerator.Cast.ShortNarrow.cs` (currently char→byte contig
only) into a strided stride-2/reverse kernel; route from `TryGetStridedCastKernel` ahead of the
generic emitter, and from `TryCopySameType` for the same-type strided case.

**Verdict:** the only large remaining win available on AVX2. Worth doing if pushing past ~93%.

---

## 3. 🚫 same-type CONTIG copy at 1M — allocation-bound (~6) — needs a buffer pool

**Cells:** `bool|F|bool` **0.17** (lone 🔴), `i8|C|i8` 0.65, `i8|F|i8` 0.62, `i8|T|i8` 0.59,
`u8|…`, etc. — small-type, *contiguous*, same-type copies at 1M.

**Root cause:** the copy itself is already `cpblk`/memcpy — it's not the kernel. The 1 MB fresh
output buffer is **cold** (page-fault on first touch). NumPy reuses **pooled** warm buffers
(`~0.0143 ms`); NumSharp allocates fresh unmanaged memory each `astype`/`clone` (`~0.085 ms`
cold). The copy-kernel improvements only show past L3 (10M).

**Fix:** a **warm buffer pool** for `astype`/`clone`/cast outputs — recycle freed
`UnmanagedMemoryBlock`s by size class instead of returning them to the OS, so the common
"allocate 1 MB, fill, drop" cycle hits warm pages. Highest *leverage* item here: it fixes these
6 cells **and** speeds every same-size copy/cast output **and** cuts allocation churn broadly.

**Risk/scope:** architectural — touches `UnmanagedStorage`/`UnmanagedMemoryBlock` allocation and
lifetime. Must not break view aliasing / `OwnsData` semantics or leak. This is an **allocator
project, not a cast kernel.** Prototype: a thread-safe free-list keyed by rounded byte size with
a cap; benchmark `astype(copy:true)` same-type at 1M (target: match NumPy ~0.015 ms) and confirm
no regression at 10M and no test failures.

**Verdict:** do this if you want the 🔴 gone and a broad allocation win; budget it as its own task.

---

## 4. 🚫 i64/u64 → f16 (16) — AVX512-gated, low value

**Cells:** all 8 layouts × {i64, u64} → f16, 0.67–0.89 (currently scalar `Converts.To{Half}`).

**Root cause:** needs `i64→double` (or `i64→f32`); `cvtqq2pd`/`cvtqq2ps` are AVX512DQ.

**Approach if pursued:** synthesise `i64→double` on AVX2 (inverse of `DoubleToU64x4`'s hi/lo
split: low-32 via the `2^52` magic, high-32 signed via `2^84` magic, add), then `double→f16`
round-to-odd (already have `DoubleToHalfBits`). **Shortcut:** f16 max finite is 65504, so any
`|v| > 65520 → ±inf`; only tiny `|v|` produce finite f16 and those are exact in double — so the
synthesis only needs to be correct in a narrow band + saturate-to-inf elsewhere. Still moderate
SIMD work for a family that's mostly producing inf.

**Verdict:** tractable but complex and low practical value. Defer unless chasing 100%.

---

## 5. 🟡 memory-bound: float/c128→u64 negcol/strided (~12) + i64/u64→narrow strided (8)

**float/c128→u64 negcol/strided** (f64 negcol 0.84, f64 strided 0.89, f32 negcol 0.89): 16 MB
read+write traffic + the heavy hi/lo `DoubleToU64x4` convert. Already staged (Wave 15d).
- *Micro-opportunity:* `negcol` (ss=−1) currently **gathers**; the data is contiguous-reversed,
  so a contiguous load + lane-reverse (as in §2.2) would beat the gather. Might lift negcol over
  0.9. `strided` (ss=2) deinterleave-doubles likewise.

**i64/u64→narrow strided** (0.83–0.90): 8-byte gatherable, already uses `VPGATHERQQ` + Narrow.
Staging the gather→buffer→narrow measured only **~16%** (0.20→0.17 ms for u64→u8 strided) — not
enough to clear 0.9; NumPy's cache-friendly scalar loop wins. (See `/tmp/u64narrow.cs` history.)

**Verdict:** a few % squeezable via contiguous-reverse on negcol; unlikely to clear 0.9
reliably. Low priority.

---

## 6. 🔵 Noise (~30) — not real laggards

Cells like `f32|C|i64`, `f64|F|i64`, `c128|C|i64`, scattered `float→narrow` contig, and some
`→u64` show 0.72–0.89 in this best-of-3 run but measure **≥1.0 under best-of-7** (verified for
several during Wave 15 — e.g. `c128|strided|u32` shows 0.80 here but is 1.21 reliably; `f32|C|i64`
shows 0.85 here but ~1.05). No code change needed.

**Action:** before touching any cell from this bucket, re-measure best-of-7. Most will already
pass. To shrink the apparent count permanently, raise the sweep to best-of-5/7 (slower) or report
≥0.9 as the headline (which already absorbs the jitter band).

---

## 7. Recommended order of attack (if continuing)

| Priority | Item | Cells | Effort | Notes |
|---|---|---|---|---|
| 1 | **best-of-7 re-measure** the noise bucket (§6) | ~30 | XS | Confirms true ≥0.9 is ~93%; free. |
| 2 | **Buffer pool** for cast/copy outputs (§3) | 6 + broad | L | Highest leverage; allocator project. |
| 3 | **stride-2 + negcol SIMD shuffle** for sub-word narrow & same-type strided (§2) | ~30–40 | M-H | Biggest AVX2 cell count; start with same-type (no convert). |
| 4 | **negcol contiguous-reverse** for float→u64 / i64u64→narrow (§5) | ~6 | M | A few % each; may not clear 0.9. |
| 5 | **i64/u64→f16 synthesis** (§4) | 16 | M-H | Complex, low value (mostly →inf). |

Do **not** attempt: general 2-byte `VPGATHER` (doesn't exist), AVX512 `cvttps2udq` for u32/u64
(saturates ≠ wrap), BCL Half casts (quiet sNaN).

## 8. Definition of done for any cell here
1. Bit-exact vs NumPy 2.4.2 (standalone large-sample/exhaustive **and** integrated `astype` ×
   all 8 layouts), scalar tail/`conv` sharing `Converts.To{X}`.
2. best-of-7 ratio ≥ 0.9 (ideally ≥ 1.0) vs NumPy, and **no regression** on neighbouring cells.
3. Full suite green (`dotnet test --no-build -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"` → 9956/0).
4. Refresh `benchmark/cast/cast_results.md` (`python benchmark/cast/cast_sheet.py --skip-build`)
   and note the wave in `CAST_BEAT_NUMPY_PLAN.md` §11.

## 9. Code references (Wave-15 kernels to mirror)

| What | Where |
|---|---|
| float/c128 → u32/u64 (mod-2³² reduce, hi/lo split, stage) | `DirectILKernelGenerator.Cast.FloatToUInt.cs` |
| f16 widen (NaN-exact) / f16→i64 / Half routing | `DirectILKernelGenerator.Cast.Half.cs` |
| complex deinterleave (`ComplexReals4`, Fused gather) | `DirectILKernelGenerator.Cast.Complex.cs` |
| char→byte contig narrow (extend for stride-2/reverse) | `DirectILKernelGenerator.Cast.ShortNarrow.cs` |
| routing chains | `DirectILKernelGenerator.Cast.cs` (`TryGetCastKernel`, `TryGetStridedCastKernel`) |
| same-type copy path (extend for strided shuffle) | `NpyIter.cs` (`TryCopySameType`, `IsSameFlatLayout`), `DirectILKernelGenerator.Copy.cs` |
| `ss==-1` contiguous-reverse precedent | `DirectILKernelGenerator.Cast.cs` (`InnerCastDoubleToInt32`) |
| NumPy-faithful scalar reference | `Utilities/Converts.Native.cs` (`To{Dst}` overloads) |
| allocation (buffer-pool target) | `Backends/Unmanaged/UnmanagedStorage.cs`, `UnmanagedMemoryBlock` |
