# Differential-Fuzz Coverage Campaign — Bug Ledger

Bugs surfaced while closing the "all × all" coverage gaps (dtype holes, missing op tiers,
operand-relationship flags, parameters, error-parity, metamorphic). Each is **documented in
`MisalignedRegistry`** (so the bit-exact gate stays green) and **left unfixed** per the campaign
directive ("bugs are skipped after marking as Misaligned"). NumPy 2.4.2 is the oracle.

Severity: 🔴 memory-safety / crash · 🟠 wrong value · 🟡 wrong dtype / throws-where-NumPy-succeeds.

---

## W1 — dtype expansion (float16 as input + narrow integers)

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W1-A | 🟠 | `floor_divide`/`mod` → **float16** | floored quotient / IEEE ±inf | `-0.0` / `NaN` | `NpyDivision` (F1) ported SByte..UInt64/Single/Double but **not Half**; Half falls back to a generic path. |
| W1-B | 🔴 | `power(uint64,int64)` | promote→float64, compute | `ArgumentException "Integers to negative integer powers"` **or `DebugAssert "index < Count, Memory corruption"`** | NEP50 `uint64+int64→float64` not applied; stays on integer-power path → OOB index in kernel. |
| W1-C | 🟡 | `power(float16,*)` scalar path | float16 power | `InvalidCastException` (Half→IConvertible) | `System.Half` does not implement `IConvertible`; scalar power helper assumes it does. |
| W1-D | 🟡 | `dot(int8,int8)` 1-D | int8 (modular) | `NotSupportedException "IL kernel not available for Sum(SByte)->SByte"` | 1-D dot routes through `ReduceAdd(int8)->int8`; no int8→int8 reduction kernel emitted. 2-D int8 GEMM is fine. |
| W1-E | 🟡 | `where(int8,…)` scalar-broadcast | select | `NotSupportedException "Zero-push unsupported for SByte"` | `NpyExpr.EmitPushZero` gained Complex/Half (F4) but not the sub-32-bit ints. |
| W1-F | 🟡 | `power(int8\|uint8, float16)` | **float16** | **float64** | power-specific NEP50 promotion widens past float16 (add/sub/mul/divide on the same pair promote correctly). |

**Coverage added (W1):** unary 2574→4914, reduce 3640→6760, binary_arith 720→1368,
binary_divmod_power 430→866, comparison 1080→2052, where 40→70, matmul 408→816 cases — all 13
NumPy-representable dtypes now gated through every existing tier.

---

## W2 — T9 bitwise + shift (`bitwise.jsonl`, 655 cases)

**No bugs.** bitwise_and/or/xor, invert, left_shift/right_shift are 655/655 bit-exact with
NumPy across all integer+bool dtypes, including the overflow-shift semantics (shift ≥ width →
0 / −1) and arithmetic-vs-logical right shift on negative operands.

---

## W3 — unary stragglers (`unary_extra.jsonl`, 4654 cases)

14 transcendental/hyperbolic/inverse-trig/angle ufuncs that had no differential coverage.
`expm1/log2/log10/log1p/positive` are clean (bit-exact or ≤2 ULP). Three bug classes:

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W3-A | 🟡 | `sinh/cosh/tanh/arcsin/arccos/arctan/deg2rad/rad2deg` on Half-promoting input (bool/int8/uint8/float16) | float16 result | `NotSupportedException "… not supported for Half"` | No Half kernel emitted for these 8 ufuncs. |
| W3-B | 🟡 | `sinh/cosh/tanh/arcsin/arccos/arctan` on complex128 | complex result | `NotSupportedException "… not supported for Complex"` | No Complex kernel (NumPy computes complex hyperbolic/inverse-trig). |
| W3-C | 🔴 | `exp2(int16\|uint16\|float32)` | float32 result | **`InvalidProgramException` (CLR rejected the emitted IL)** | The float32-output `exp2` kernel emits a malformed IL method body. `exp2(float64)`/Half are fine — isolated to the Single emitter. |

---

## W4 — NaN-aware reductions (`nanreduce.jsonl`, 2040 cases)

The nan* family is **broadly broken** — 526/2040 cells diverge. `nanmax/nanmin/nanprod` are
clean; `nansum/nanmean/nanstd/nanvar/nanmedian` are not.

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W4-A | 🟠 | `nanmean/nanstd/nanvar` (Shape) | scalar `[]`; `keepdims` honored | `[1]` on 1-D axis-0; `keepdims` **ignored on int input** | reduction collapses/keepdims logic in the QuantileEngine/mean path is wrong for these shapes. |
| W4-B | 🟠 | `nansum/nanmean` on strided axis | all-NaN slice → `0`; correct count | garbage (`2³¹`) / wrong mean (32.0625 vs 32.0) | NaN masking & divisor-count wrong on the strided/axis path. |
| W4-C | 🟠 | `nanmedian` with NaNs | ignores NaN → non-NaN median (±inf etc.) | **`NaN`** | nanmedian does not strip NaN before the median — propagates it. |
| W4-D | 🟡 | `nansum(complex128, axis)` 1-D | complex sum | `InvalidOperationException "NDCoordinatesAxisIncrementor … vector shape"` | shared complex-1D-axis-reduction defect (same class as #12). |
| W4-E | 🟡 | `nanmean/nanstd/nanvar` empty float16, axis=None | `NaN` + warning | `InvalidOperationException "NDIterator … empty shape"` | empty-array path not handled for the float16 nan-reduce. |

---

## W5 — cumulative (`scan.jsonl`, 544 cases)

`diff` is **fully clean** (bit-exact across n=1,2 / axis 0,last / all dtypes). One cumsum/cumprod bug:

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W5-A | 🟡 | `cumsum`/`cumprod` on a **size-1** int16/int32/uint8/uint16 array | int64 / uint64 (NEP50 accumulator) | input dtype preserved | the one-element fast path skips the accumulator widening that the size>1 path applies correctly. |

---

## W6 — statistics (`stat.jsonl`, 2304 cases)

`ptp` and `count_nonzero` are **clean**. `median/percentile/quantile` (shared QuantileEngine),
`average`, and `clip` have bugs:

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W6-A | 🟠 | `median/percentile/quantile` on a slice with ±inf/NaN | clean value or NaN per IEEE | NaN where NumPy isn't (and vice-versa) | partition + linear interpolation `(a+b)/2` / `a+(b−a)·f` mishandles non-finite operands (e.g. `(+inf+−inf)/2`). |
| W6-B | 🟠 | `percentile/quantile(int…, axis)` | interpolated float64 | **gross** wrong value (sign flips: +8192 vs −8191) | genuine QuantileEngine defect on the integer axis interpolation path. |
| W6-C | 🟡 | `average` over large-magnitude slice | pairwise sum | naive-sum drift | summation order differs from NumPy. |
| W6-D | 🟠 | `clip(NaN, lo, hi)` | `NaN` (passthrough) | `lo` (−10) | clip's min/max comparisons sort NaN below the lower bound → clamps NaN to a_min instead of preserving it. |

---

## W7 — logic + element-wise extrema (`logic.jsonl`, 828 cases)

`isnan/isinf/isfinite` are **clean**. `maximum/minimum/fmax/fmin/isclose` have bugs:

| # | Severity | Op · cell | NumPy | NumSharp | Root cause |
|---|----------|-----------|-------|----------|------------|
| W7-A | 🟠 | `maximum/minimum/fmax/fmin` with an **F-contiguous/strided** operand | element-wise logical pairing | **scrambled** (reads memory order) | the extrema kernel ignores the operand's strides and walks it C-contiguously. C-contig is bit-exact; add/sub/mul handle the same F-contig operand correctly, so it's extrema-specific. |
| W7-B | 🟠 | `fmax/fmin(x, NaN)` | `x` (ignore NaN) | `NaN` | fmax/fmin propagate NaN — they behave identically to maximum/minimum instead of skipping NaN. |
| W7-C | 🟡 | `isclose` on F-contiguous complex | element-wise bool | wrong bool | same strided-pairing family as W7-A on the complex path. |
