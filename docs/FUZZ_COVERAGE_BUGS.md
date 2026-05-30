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
