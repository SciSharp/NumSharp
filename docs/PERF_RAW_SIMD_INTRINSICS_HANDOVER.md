# Handover — Raw x86 SIMD intrinsics vs `Vector256.*` JIT fixups (the min/max lesson, generalized)

Status date: 2026-06-27 · Branch: `nditer`

## TL;DR

On **.NET 9+**, `Vector256.Min` / `Vector256.Max` for **floating-point** do **not** lower to a
single `vminpd`/`vmaxpd`. The JIT appends an **IEEE NaN-propagation fixup** (a compare + a
blend) so the BCL method matches `Math.Min` NaN semantics. That fixup roughly **doubles** the
instruction count of the hot loop. Integer min/max have no NaN domain, so they get the single
(or, on AVX2, the emulated) instruction and were already fast — which is exactly why
`min(i64)` beat NumPy while `min(f64)`/`min(f32)` lost: **same kernel, only the element type
differed.** A genuine memory/bandwidth wall would have capped all three dtypes equally.

**The fix pattern:** when you are already doing (or can cheaply do) your own NaN handling,
use the **raw** `Avx.Min` / `Avx.Max` intrinsic (which *drops* NaN, like hardware `MINPD`)
and track "did any NaN appear" in a separate **finite mask**, taking a **cold scalar scan**
for the exact NaN bits only when one is present. This skips the per-element fixup entirely.

Measured on this op (f64/f32 flat min/max, NumPy 2.4.2, `>1` = NumSharp faster):

| dtype | 100K before | 100K after | 10K after | 1K after |
|-------|-------------|------------|-----------|----------|
| f64   | **0.64×**   | **1.55×**  | 2.01×     | 2.03×    |
| f32   | **0.69×**   | **1.73×**  | 2.14×     | 1.95×    |
| i64   | 1.26× (n/a) | 1.29×      | 1.88×     | 2.23×    |

→ A ~2.4× kernel speedup on the float paths; NumPy now beats NumSharp on **none** of the
three dtypes for min/max.

---

## 0. What's already done (do not redo)

| Op | File / kernel | Commit |
|----|---------------|--------|
| **flat (axis=None) f64/f32 min/max** | `Backends/Default/Math/DefaultEngine.ReductionOp.cs` → `FlatMinMaxF64Avx` / `FlatMinMaxF32Avx`, routed from `ExecuteElementReduction` ahead of the IL kernel, gated on `Avx.IsSupported` | `f73eba6e` |
| **axis f64/f32 min/max on broadcast / negative-stride** (the earlier G1 routing) | `ILKernelGenerator.Reduction.cs` → `SimdMinMaxSameType` (still uses `Vector256.Min/Max` — see target T3) | `3a34aebf` |

The `FlatMinMaxF64Avx` / `FlatMinMaxF32Avx` kernels are the **reference implementation** of
the pattern — copy their structure.

---

## 1. The mechanism (why `Vector256.Min/Max` is ~2× the raw instruction)

- `EmitVectorNaNPropagatingMinMax` (`Direct/DirectILKernelGenerator.Reduction.cs:599`) has a
  `#if NET8_0` split. On **net8.0** the BCL `Vector256.Min/Max` were raw `MINPS/MAXPS`
  (NaN-dropping), so it hand-rolled a 6-op NaN wrapper. On **net9.0+** it emits the bare
  `Vector256.Min/Max` — believing the single intrinsic is free. **It is not**: the net9+ JIT
  lowers `Vector256.Min` to `vminpd` **plus** `cmpunordpd` + `blendvpd` to re-introduce NaN.
- Raw `Avx.Min(a,b)` (= `VMINPD`) returns the **second** operand when either is NaN (Intel
  SDM "MINPD"). It is one instruction, but it does **not** propagate NaN — so it is only safe
  when you handle NaN yourself.
- The raw intrinsic is already wired up: `VectorMethodCache.BinaryX86(simdBits, "Min"|"Max",
  clrType)` (`Backends/Kernels/VectorMethodCache.cs:518`) returns `Avx.Min`/`Avx.Max` for
  float/double (and `Avx2.*` for ints). The file header (line 83) already notes raw Avx is
  "1.8–2× ... code than `System.Runtime.Intrinsics.X86.Avx.*`" — this handover is the concrete
  payoff for the min/max family that was bypassing it for NaN-correctness.

**Diagnostic that proves it for any candidate op** (drop into a `dotnet run` probe):

```csharp
// identical 8-accumulator loop, only the combine differs:
a_k = Vector256.Min(a_k, v_k);   // variant A: BCL (fixup)
a_k = Avx.Min(a_k, v_k);         // variant B: raw VMINPD
// If B is ~2× A on f64/f32 but ~equal on i64 → the fixup is your bottleneck.
```

---

## 2. The fix pattern (validated template)

Copy `FlatMinMaxF64Avx` (`DefaultEngine.ReductionOp.cs`). The shape:

```csharp
if (!Avx.IsSupported || n < W) { /* portable/scalar fallback, NaN-propagating, bit-exact */ }

var seed = Vector256.Create(max ? -inf : +inf);
var a0..a7 = seed;                      // 8 explicit accumulators (saturate the 2 min/max ports)
var fin = Vector256<T>.AllBitsSet;      // finite mask
for (stripe of 8*W) {
    var v0..v7 = Load(...);
    a_k = max ? Avx.Max(a_k, v_k) : Avx.Min(a_k, v_k);     // RAW — drops NaN
    fin &= Equals(v0,v0) & ... & Equals(v7,v7);            // NaN lane ⇒ that lane goes 0
}
// horizontal-fold a0..a7 with raw Avx, then W-remainder, then scalar tail
bool anyNaN = ExtractMostSignificantBits(fin) != (1<<W)-1;
if (anyNaN) for (k in 0..n) if (IsNaN(d[k])) return d[k];  // COLD: exact first-NaN bits
return acc;
```

Non-negotiable correctness rules (all baked into the reference kernel):

1. **Avx.IsSupported gate.** Non-x86 (ARM `AdvSimd`) must fall through to the existing portable
   path — never ship an x86-only kernel without a fallback.
2. **NaN bits, not `T.NaN`.** Propagate the **input** NaN's exact bits (NumPy copies the input
   NaN; `.NET`'s `double.NaN` is the *negative* `0xFFF8…` and will fail a bit-exact diff). The
   cold scan returns `d[k]`, which is correct by construction.
3. **±0 / ±inf.** Raw min/max + the scalar fold already match NumPy (verified). Don't add a
   "clever" signed-zero branch — it isn't needed and risks divergence.
4. **8 accumulators** for a contiguous hot loop (saturates the two ALU ports); the finite mask
   ANDs cheaply alongside. More than 8 spills YMM registers and *regresses* — measured.

---

## 3. Targets (ranked) — where the same lesson applies next

> All use the **float** `Vector256.Min/Max` fixup today. Each is an independent, shippable win.

### T1 — axis f64/f32 min/max, contiguous (Direct kernel)  **[HIGH]**
- **Where:** `Direct/DirectILKernelGenerator.Reduction.Axis.Simd.cs` → `NaNAwareMinMax256` /
  `NaNAwareMinMax128` (~line 999–1012); used by the contiguous-axis SIMD reduce.
- **Why:** axis min/max on contiguous arrays is the common 2-D case and pays the full fixup.
- **How:** swap the `Vector256.Max/Min(a,b)` inside `NaNAwareMinMax*` for `BinaryX86(...,
  "Min"/"Max", clrType)` (raw `Avx`) **and** add a finite-mask accumulator threaded through the
  axis reduce + a cold per-output-row NaN scan. This is the only target needing IL/structural
  work (the mask must persist across the unrolled body + horizontal merge). Reference: how
  `FlatMinMaxF64Avx` keeps `fin` and scans cold.
- **DoD:** bit-exact (edge diff + FuzzMatrix), axis min/max f64/f32 within the flat ratios.

### T2 — `np.clip` f64/f32  **[HIGH, easy]**
- **Where:** `Direct/DirectILKernelGenerator.Clip.cs` → `EmitVectorMinOrMax(il, …,
  propagateNaN: true)` (~line 237, 249). Loop is `Min(Max(src, lo), hi)`.
- **Why:** clip is a hot elementwise op and emits the fixup'd min/max twice per element.
- **Subtlety:** `np.clip` propagates NaN in the **value** (clip(nan)=nan) but the bounds
  `lo`/`hi` are scalars. With raw `Avx.Max(src, lo)` then `Avx.Min(., hi)`: a NaN `src` is
  dropped by both → wrong. Use the per-lane NaN-keep blend (see `NaNFoldVec` in
  `ILKernelGenerator.Reduction.cs`) **or** a finite mask + cold fix on the value lanes. Verify
  against `np.clip([nan, …], lo, hi)` explicitly.
- **DoD:** clip f64/f32 ratio up; FuzzMatrix `clip` corpus + a NaN clip case green.

### T3 — axis f64/f32 min/max on broadcast / negative-stride (my own kernel)  **[MED, easy]**
- **Where:** `ILKernelGenerator.Reduction.cs` → `SimdMinMaxSameType` (PINNED hot loop +
  `NaNFoldVec` SLAB). It **already** tracks NaN via a finite mask / NaN-keep blend, yet the
  hot loop still calls `Vector256.Min/Max` — so the fixup is **pure redundant cost** here.
- **How:** swap `Vector256.Min/Max` → `Avx.Min/Max` (guarded by `Avx.IsSupported`; keep the
  current calls as the non-x86 fallback). No NaN-logic change needed — the mask already exists.
  Lowest-risk of all the targets (my code, no structural change).
- **DoD:** negcol/bcast axis min/max ratios rise; edge diff + FuzzMatrix green.

### T4 — elementwise `maximum` / `minimum` / `fmax` / `fmin` f64/f32  **[MED — verify first]**
- **Where:** the binary-op min/max kernels (search `KernelOp` `Maximum`/`Minimum`/`FMax`/`FMin`
  → their emit in `Direct/DirectILKernelGenerator.*` / `MixedType`). Confirm they use
  `Vector256.Min/Max`; if so, same fix.
- **Subtlety:** `maximum`/`minimum` **propagate** NaN; `fmax`/`fmin` **ignore** NaN (return the
  non-NaN operand). For `fmax`/`fmin`, raw `Avx.Max`/`Avx.Min` is *almost* right but the
  operand-order-on-NaN differs — test `fmax(nan, 1)` and `fmax(1, nan)` both.

### T5 — `nanmin` / `nanmax` (NaN-ignoring)  **[LOW — different semantics]**
- **Where:** `Reduction.NaN.cs`, `Reduction.Axis.NaN.cs`, `Masking.NaN.cs`.
- **Note:** these *skip* NaN (opposite of min/max). Raw `Avx.Min/Max` actually fits the
  "ignore NaN" intent better than the fixup, but the all-NaN edge (NumPy returns NaN + a
  RuntimeWarning) must be preserved. Measure before touching.

---

## 4. General methodology — finding more of these

The lesson is **not** "replace every `Vector256.*` with `Avx.*`." Most binary arithmetic
(`Add`/`Sub`/`Mul`/`Div`) already routes through `VectorMethodCache.BinaryX86`, and those map
1:1 to hardware with no fixup. The win is specifically where a `Vector256.*` method carries a
**JIT-inserted IEEE/edge-case fixup** that you are *already handling yourself* (or don't need):

1. **Grep** `Vector256.(Min|Max)` / `Vector128.*` / `Vector512.*` in `Backends/Kernels/`
   (44 hits today — see the survey in this branch's history).
2. For each, **probe** the raw-vs-BCL A/B (template in §1). Flag float ops where raw is ~2×.
3. Check whether the surrounding kernel **already** does the edge handling the fixup duplicates
   (finite mask, cold scan, scalar fallback). If yes → free swap (like T3). If no → add the
   cheap edge handling, then swap (like T1/T2).
4. Other fixup-prone BCL vectors worth A/B-ing on float: none confirmed beyond min/max yet —
   `Abs`/`Sqrt`/`Floor`/`Ceiling`/`Add`/`Mul` map cleanly. **Measure, don't assume.**

---

## 5. No-harm validation protocol (gate every change)

```bash
# 1. build Core Release
dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0

# 2. bit-exact edge differential vs NumPy 2.4.2 — MUST include NaN / ±inf / ±0, and sizes that
#    hit the scalar tail (n<W), one SIMD block, the 8× unrolled body, and the cold NaN scan.
#    Feed the SAME NaN bits both sides (NumPy float('nan') == 0x7FF8…; .NET double.NaN is 0xFFF8…
#    — use BitConverter.Int64BitsToDouble(0x7FF8000000000000L) in C# or the diff is a false red).

# 3. differential gate + suites, BOTH frameworks (the fix is Avx.IsSupported-gated, runs on net8 too)
cd test/NumSharp.UnitTest
dotnet test --no-build -c Release --framework net10.0 --filter "TestCategory=FuzzMatrix"
dotnet test --no-build -c Release --framework net8.0  --filter "TestCategory=FuzzMatrix"
dotnet test --no-build -c Release --framework net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&(ClassName~Reduction|ClassName~Statistic|ClassName~Clip|ClassName~Math|ClassName~Amax|ClassName~Amin)"
```

Add a corpus case for any new kernel (the min/max routing added `negstride_*` to
`REDUCE_LAYOUTS` in `test/oracle/gen_oracle.py`; clip/maximum/fmax have their own corpus modes).

---

## 6. Caveats & non-targets

- **i64 (and u64) min/max is NOT this lesson.** AVX2 has no `vpminsq`/`vpmaxsq` (64-bit packed
  min/max) — `Vector256.Min<long>` is *emulated* (compare + blend) with or without a fixup, so
  there is no raw single-instruction win. i64 min/max already runs ~1.2–2.2× NumPy; its 100K
  ~1.29× is the AVX2 emulation ceiling, not a fixup. (Same story for i64/u64 `Multiply`: no
  `vpmullq` on AVX2.) AVX-512 would add these, but **`Vector512.IsHardwareAccelerated == false`
  on the dev box** — don't bank on it.
- **Memory-bound sizes.** This lesson is a *compute/fixup* win, visible at cache-resident sizes
  (≤~1 MB working set). At tens-of-MB the op is memory-bandwidth-bound (~30 GB/s single core)
  and both NumSharp and single-threaded NumPy sit near that wall — the raw-intrinsic swap helps
  little there. (Multi-core would, but reductions are intentionally single-threaded here.)
- **Always keep the portable path.** Gate on `Avx.IsSupported`; ARM/`AdvSimd` keeps the
  existing `Vector256.*` kernel (which is correct, just carries the fixup).
- **NaN bit-exactness is the trap.** Every false red in this work was a probe feeding
  `double.NaN` (negative) against NumPy's positive `float('nan')`. The kernels are correct;
  match the input NaN bits in tests.
