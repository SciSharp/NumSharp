# Float16 (Half) performance design — how f16 is fast in NumSharp

NumSharp's float16 ops beat NumPy 2.4.2 by 2–60× on most of the ufunc surface while staying
bit-exact, on a runtime that has **no vectorized F16C**. This document records the constraint
that shapes everything, the three techniques that work (chosen by one question per op), the
sign-magnitude order map they share, the NumPy semantics contract each kernel reproduces, the
dispatch seams, the measured scoreboard, and the traps that cost real time to learn.

All numbers are NPY/NS (NumPy_ms / NumSharp_ms — higher = NumSharp faster), best-of-9 warm,
Release, measured through the real `np.*` path against NumPy 2.4.2 (win-amd64 wheel) on the
same host. Commits: `a05b5b1e` `f3405659` `3a46cfec` `c9e141c5` `498a68f0` `c8b0573d`
`c8babf28` (plus the earlier sign/negate/abs and widen-unary work they build on).

## 1. The constraint

.NET on this runtime exposes **no vector float16 arithmetic and no vector F16C conversion**:

- `Vector128<Half>`/`Vector256<Half>` throw `NotSupportedException` for every arithmetic op;
  there is no F16C intrinsic class and `Avx10v1.IsSupported` is false.
- The **scalar** conversions are hardware (`[Intrinsic] op_Explicit` → `vcvtph2ps`/`vcvtps2ph`,
  fast per call) but latency-bound in a loop; `TensorPrimitives.ConvertToSingle` measures
  *slower* than the scalar converts.
- SIMD f16↔f32 conversion therefore means the **software Giesen bit-fiddle**
  (`HalfBitsToFloatExact` / `FloatToHalfBits` in the Cast.Half/Cast.ToHalf kernels — proven
  0-diff vs `npy_half_to_float`/`npy_float_to_half` over all 65,536 patterns), which costs
  real vector work per 8 lanes.

The opening on the other side: **NumPy's own HALF loops are scalar C on win-amd64.** The
`loops_half.dispatch` AVX-512-FP16 path needs Sapphire Rapids and is absent from the wheel;
minimum/maximum/comparisons are branchy bit-compares (`npy_half_ge` via `half.hpp`), arithmetic
is `npy_float_to_half(npy_half_to_float(a) op npy_half_to_float(b))` per element, and the
roundings are `narrow(floorf(widen(h)))` per element. Measured on 2.4.2 per 100K: `maximum`
512 µs, `clip` 795 µs, `rint` 723 µs, `less` 414 µs, `floor` 440 µs, `add` 285 µs, `max`
reduce 150 µs — 25–100× slower than NumPy's own f32 SIMD for the same ops. Every technique
below exploits exactly this asymmetry.

## 2. Three techniques, picked by one question

**"Does this op need real float math?"**

| Answer | Technique | Result class |
|---|---|---|
| No — defined on the bit pattern | **T1: bit-level AVX2 on raw ushort lanes** (zero conversions) | **Beats NumPy 2–60×**, bit-exact |
| Yes, but +−×÷ only | **T2: SIMD widen-compute-narrow** (Giesen f16→f32 → one vector op → Giesen RTNE narrow) | Beats NumPy 2.2–3.6×, bit-exact |
| Yes, a transcendental | **T3: scalar F16C widen-unary** (`(Half)MathF.X((float)h)` — `EmitUnaryHalfViaFloat`) | Conversion-bound: ≈ parity (sqrt 1.1–1.2×) |

### T1 — bit-level kernels (the class that wins)

f16 is small enough that whole ops collapse to integer lane algebra:

- **sign / negate / abs** (`DirectILKernelGenerator.Unary.Decimal.cs`): `bits ^ 0x8000`,
  `bits & 0x7fff`, and a 3-blend select chain. The original proof that conversion-free wins.
- **Ordering** — everything else in T1 rides one proven map (see §3): flat `min`/`max`/`ptp`
  (`Reduction.MinMax.Half.cs`), elementwise `maximum`/`minimum`/`fmax`/`fmin` and `clip`
  (`Binary.MinMax.Half.cs`), the six comparisons (`Comparison.Half.cs`), `nanmin`/`nanmax`
  (NaN lanes blended to the identity key).
- **Roundings** — `floor`/`ceil`/`trunc`/`rint` (`Unary.Round.Half.cs`): every result is
  exactly f16-representable (all f16 ≥ 2¹⁰ are already integers), so NumPy's
  widen→`roundf`→narrow is *identically* exponent-based mantissa masking:
  `fracBits = 25 − exp`, truncate = `bits & ~mask`, floor/ceil add `mask+1` on the fractional
  side (the bit-step carries across exponent boundaries exactly like value+1), rint =
  `(bits + mask/2 + intLSB) & ~mask` (RTNE). Verified **exhaustively**: all 65,536 patterns ×
  4 ops vs live NumPy, 0 diffs.

### T2 — SIMD widen-compute-narrow (`Binary.Arith.Half.cs`)

`add`/`subtract`/`multiply`/`divide` run NumPy's exact pipeline 8 lanes at a time:
`HalfBitsToFloatExact` (NaN-payload-exact widen) → one `vaddps`/`vsubps`/`vmulps`/`vdivps` →
`FloatToHalfBits` (RTNE narrow, payload-truncating) → `PACKUSDW`. Two things make it *bit*-exact
rather than close:

- The compute is **float32, never double**. A double bridge single-rounds where NumPy's f32
  path double-rounds (exponent-gap sums like `65504 + tiny`), and the hardware `(double)Half`
  widen **quiets sNaN before the op** — both real divergence classes the old emitted-IL route
  had.
- NaN-payload **priority is an explicit post-op blend**, never instruction operand order
  (see trap #4): add/multiply resolve in2's NaN first, subtract/divide in1's, with
  `quiet(x) = x | 0x0200`.

The same widen primitives power the f16 **sum reduce** (`ILKernelGenerator.Reduction.Half.cs`):
NumPy's pairwise f32 fold reproduced in a float32 *shadow accumulator* with a per-inner-loop
`RoundToF16` — saturating exactly like NumPy's per-call narrow without the 16-bit round-trip.

### T3 — scalar widen-unary (`EmitUnaryHalfViaFloat` and the BCL `Half.X` statics)

Sinh/Cosh/Tanh/ASin/ACos/ATan/Asinh/Acosh/Atanh/Tan/Cbrt/Log2/Log10/Exp2/Reciprocal/Square emit
`(Half)Xf((float)h)` via `EmitUnaryHalfViaFloat`; sqrt/sin/cos and friends call the BCL `Half.X`
statics, which are internally the same shape. Either way it is the scalar hardware F16C converts
around the CRT float function — NumPy's own `npy_half` model
(`npy_float_to_half(npy_Xf(npy_half_to_float(h)))`), so it is the *correct* implementation — and
it is conversion-latency-bound, so it lands at ≈ parity (sqrt measures 1.1–1.2×), not at a win. That ceiling is physical: NumPy's compiled loop overlaps
the same scalar converts with compute; there is nothing left to vectorize without a vector f16
transcendental that matches the CRT bit-for-bit (none exists). Two ops deliberately stay on the
double bridge: `expm1`/`log1p`, because `float.ExpM1/LogP1` lose small-|x| precision (16,386
finite diffs over the f16 domain).

## 3. The order map (T1's shared core)

```
key = bits ^ (0x8000 | (bits >>ARITH 15))     // positives: bits | 0x8000,  negatives: ~bits
```

Proven exhaustively: strictly order-preserving over all 63,490 non-NaN patterns (sorted-order
comparison against float order), round-trips all 65,536, and **every NaN key lands strictly
outside the finite range** — +NaN keys above `key(+inf) = 0xfc00`, −NaN keys below
`key(−inf) = 0x03ff`. Consequences the kernels lean on:

- Unsigned `pmaxuw`/`pminuw` over keys **is** float max/min; unsigned key compares **are**
  `npy_half_lt/le` (raw form) for every bit pattern including NaNs.
- Tracking *both* extremes in one loop makes NaN presence decidable from the accumulator pair
  alone (one extra `pminuw` per vector — no separate NaN mask), and a NaN-*skipping* reduce
  (`nanmax`) needs only NaN lanes blended to the identity key: an accumulator still outside
  the finite key band afterwards *proves* the input was all-NaN.
- The single seam where the key order disagrees with IEEE: `−0 < +0` by key, equal by
  `npy_half` ge/le. Every kernel carries a `bothZero` repair mask (elementwise) or a rare-path
  first-zero rescan (reductions).
- Horizontal ushort reduction is one instruction: `PHMINPOSUW` (`Sse41.MinHorizontal`); max
  reduces through it on the complement.

## 4. The NumPy semantics contract (what "bit-exact" required)

All probed live on 2.4.2 and pinned by tests; the fold analysis was verified against
`numpy/_core/src/npymath/halffloat.cpp` + `half.hpp` + `loops.c.src`:

- **`npy_half_ge/le` are the NaN-GUARDED operators** (`operator<=` =
  `!(isnan||isnan) && raw compare`), *not* the `_nonan` raw compares. A reference model built
  on the raw compares mispredicts every `(finite, −NaN)` maximum (trap #3).
- **Reduce folds keep the accumulator on ties and on NaN** (`(ge(acc,x) || isnan(acc)) ? acc : x`):
  `np.max` returns the **first NaN verbatim** (payload + sign — even a negative NaN `0xfe22`),
  and a ±0 extremum takes the **first zero's sign**. `nanmax` is `fmax.reduce`: NaNs skipped,
  all-NaN returns the **first element verbatim**.
- **Elementwise min/max**: maximum/minimum return the NaN *operand* (in1 preferred when both);
  fmax/fmin return the non-NaN operand (both-NaN → in1); ±0 ties → in1. `clip` is
  `_NPY_MIN(_NPY_MAX(x, lo), hi)` — a NaN *bound* wins its stage for every input (the guarded
  compare is false against NaN, so the bound is taken and rides through), `lo > hi` resolves
  to `hi` (min applied last).
- **Comparisons**: ±0 equal in every ordered op; NaN makes everything false except `!=`
  (true — including two NaNs with identical bits).
- **Arithmetic NaN payloads flow** (widen `payload << 13`, op, truncate back), sNaN quieted by
  the FP op (`| 0x0200` at f16 level), 0/0 and inf−inf produce the x86 default qNaN `0xfe00`.
  Payload **priority is a host pin**: the wheel's MSVC build compiled the commutative ops with
  reversed operands (`addss dst=in2`) — probed: `add(qNaN_a, sNaN_b) = quieted b`,
  `sub(qNaN_a, sNaN_b) = a`. Same pin class as the MSVC cast kernels
  (`Fuzz/README.md` → "Host-dependent values").
- **Roundings**: NaN → `bits | 0x0200`; `ceil(−0.2) = −0` (sign preserved); `rint(±0.5) = ±0`
  (ties-to-even below 1); `|x| ≥ 1024` is identity (inf included).

Fixed along the way (real divergences the kernels replaced): flat max's last-NaN-wins scan;
`HalfMaxNaN/HalfMinNaN` returning canonical `Half.NaN` (payload destroyed — also broke clip);
`nanmax` all-NaN canonical NaN; a clip precedence bug
(`(f32||f64) && IsNaN(min) || IsNaN(max)`) that returned a **float32** all-NaN fill for *any*
dtype with a NaN max-bound; the arithmetic double bridge.

## 5. Dispatch map — where the gates live

Every gate is `(Half → Half)`-keyed so other dtypes keep their existing SIMD paths, and every
non-served layout falls to the pre-existing route unchanged.

| Surface | Gate | Kernels | Fallback (unchanged) |
|---|---|---|---|
| unary sign/negate/abs + floor/ceil/trunc/rint (`around`/`round` share Round) | `GenerateUnaryKernel` (contiguous keys) | `Unary.Decimal.cs`, `Unary.Round.Half.cs` | strided → scalar IL |
| flat min/max (→ ptp) | `Max/MinElementwiseHalfFallback` (contiguous) | `Reduction.MinMax.Half.cs` | non-contig → `HalfMinMaxViaNDIter` |
| nanmin/nanmax | `NanMin/NanMaxHalfHelper` bodies | same file (NaN-blend cores) | non-contig → `NanReductionScalar` |
| elementwise maximum/minimum/fmax/fmin + add/sub/mul/div | `GenerateMixedTypeKernel` (SimdFull + both scalar-broadcast paths) | `Binary.MinMax.Half.cs`, `Binary.Arith.Half.cs` | SimdChunk/General → emitted IL |
| comparisons ==/!=/</<=/>/>= | `GenerateComparisonKernel` (same three paths) | `Comparison.Half.cs` | SimdChunk/General → scalar IL |
| clip (every mode × scalar/array bounds) | `Clip.Generate` | clip cores in `Binary.MinMax.Half.cs` | strided → `ClipStridedT` |
| sum reduce | `ReduceKernelKey(Sum, Half, Single)` | `ILKernelGenerator.Reduction.Half.cs` | — |

One dispatch subtlety worth knowing: `np.add(arr, 0-d scalar)` reaches the binary Tier 3B
NDIter route *before* the MixedType kernel factory — `TryExecuteBinaryOpViaNDIter` explicitly
**declines** `(Half,Half)→Half` add/sub/mul/div so scalar broadcasts land on the
`SimdScalarLeft/Right` kernels (contiguous same-shape pairs already took the trivial bypass).

## 6. Scoreboard (NPY/NS, 1K / 100K / 10M)

| Family | 1K | 100K | 10M | Exactness |
|---|---|---|---|---|
| `max`/`min` (flat) | 6.3× | 57× | 40× | bit-exact (first-NaN verbatim) |
| `ptp` | 3.4× | 51× | 35× | bit-exact (composition) |
| `maximum`/`fmax` (elementwise) | 5.6× | 56× | 27× | bit-exact |
| `less` / `equal` | 2.7× / 1.8× | 40× / 9.7× | 24× / 5.4× | bit-exact |
| `nanmax`/`nanmin` | 8.4× | 43× | 26× | bit-exact (all-NaN verbatim) |
| `clip` | 4.3× | 48× | 40× | bit-exact (bound-NaN bits) |
| `add`/`mul`/`div` | 2.2× | 3.6× | 3.6× | bit-exact (f32 pipeline + payload pin) |
| `floor`/`ceil` | ~6× | ~15× | ~14× | exhaustively bit-exact (65,536×) |
| `trunc` / `rint` | ~6× | 22× / 26× | 21× / 25× | exhaustively bit-exact |
| `sign` / `negative` / `abs` | 1.5–2× | 8–14× | 4.3–7.9× | bit-exact (65,536×) |
| `sqrt` (T3 class) | 1.2× | 1.14× | 1.19× | matches NumPy's CRT model |

10M cells are memory-bandwidth-bound (e.g. flat max streams at ~54 GB/s) — the correct ceiling,
not a shortfall. 1K cells are bounded by the per-call allocation/dispatch floor, not the kernels.

## 7. Traps (each cost real debugging time — do not relearn)

1. **The portable `Vector256` API does not lower** for ushort compare/select — a sign kernel on
   `Vector256.GreaterThan<ushort>`/`ConditionalSelect` measured ~30× slower than the same logic
   in raw `Avx2.*` intrinsics. Always use `Avx2.*` directly for f16 bit kernels (signed
   `CompareGreaterThan` is safe on magnitudes since both sides ≤ 0x7fff).
2. **AVX2 has no 16-bit variable shift** (`VPSLLVW` is AVX-512BW) — per-lane `fracBits` work
   (the roundings) widens 8 ushorts to an i32 vector and uses `VPSLLVD`/`VPSRLVD`.
3. **`npy_half_ge/le` are NaN-guarded**, not the raw `_nonan` compares (`halffloat.cpp` exports
   the guarded operators). Building a reference on the raw compares silently flips every
   `(finite, −NaN)` maximum and `(−NaN, finite)` fmin.
4. **RyuJIT re-swaps commutative intrinsic operands.** `Avx.Add(fb, fa)` does *not* guarantee
   `vaddps dst, fb, fa` — observed: the carefully-ordered operands came out swapped, resolving
   the wrong operand's NaN. Any semantics that depend on operand order (x86 NaN priority) must
   be an explicit mask/blend, never instruction order.
5. **An in-loop data-dependent branch costs ~2.26×** (the argmax lesson) — but *loop-invariant*
   branches (`isMax`, `nanIgnore`, `op`) are perfectly predicted and effectively free; one
   shared core with invariant flags beats duplicated per-op loops.
6. **Min and max accumulate together for free-ish** — the second accumulator doubles as the
   opposite-sign NaN detector; a dedicated NaN mask (and+cmp+or per vector) is strictly worse.
7. **Measure only in the real `np.*` path** with `#:property PublishAot=false` (else every
   DynamicMethod kernel is null → 10–17× slow scalar garbage) and a fresh script filename per
   rebuild (stale runfile cache). Script-lambda/cross-assembly microbenches mislead.
8. **Prove first, at full width**: unaries get all-65,536 sweeps; reductions get a scalar fold
   transcribed from NumPy's C *plus* a live-NumPy differential (the source-derived model was
   wrong once — trap #3 — and only the live probe caught it).

## 8. Ceilings and what remains

- **T3 is capped at ≈ parity by physics** — conversion-bound; matching NumPy's CRT results
  requires the same scalar converts NumPy uses. A vector polynomial (SLEEF-style) would be
  faster and *diverge*; rejected.
- **`np.power(f16)`** still routes `Half→double→Math.Pow→Half` (NumPy parity via the double
  bridge; not yet on T3's float path).
- **The small-op floor**: at 1K, alloc/dispatch (~0.5–1 µs) dominates every kernel; improving
  it is cross-cutting NDArray-lifecycle work, not f16 work.
- **Non-contiguous paths** keep the older fallbacks: the strided/NDIter f16 min/max fallback
  returns canonical `Half.NaN` (payload lost) instead of first-NaN verbatim; f16 axis
  reductions ride the generic driver. Both pre-existing, both documented above.
- **Don't touch `median`/`sort`/`unique` as f16 work** — their gaps are sort-algorithm gaps
  (scalar radix vs NumPy's vqsort), not dtype gaps.

Gates: `HalfMinMaxBitKernelTests`, `HalfPairMinMaxKernelTests`, `HalfComparisonKernelTests`,
`HalfArithKernelTests`, `HalfRoundKernelTests` (test/NumSharp.Tests/Math/) + the FuzzMatrix
oracle corpus; dev-time evidence per family: 1,200–13,000-case reference fuzz + 250–30,600-check
live-NumPy differentials, 0 fails throughout.
