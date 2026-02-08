# NumPy 1.x Deprecation Audit — Findings

## Status: Phase 1-7 Complete

This document contains the consolidated results of the systematic audit comparing NumSharp against NumPy 2.x (v2.4.2). See `numpy-1x-deprecation-audit.md` for the investigation plan.

---

## Executive Summary

| Category | Count | Severity |
|----------|-------|----------|
| Deprecated/removed APIs present in NumSharp | 5 functions | High |
| Deprecated aliases/constants | 13 symbols | Medium |
| Type promotion table mismatches (arr-arr) | 0 of 100 | None |
| Type promotion table mismatches (arr-scalar) | 12 of 80 | **Critical** |
| Behavioral divergences from NumPy 2.x | 6 areas | High |
| Bugs discovered during audit | 8 bugs | Mixed |
| NUMPY_NUMSHARP_MAP.md corrections needed | 5 entries | Medium |
| Outdated documentation URLs | 385 across 121 files | Low |

---

## 1. Deprecated/Removed APIs Still Present

### 1.1 Functions

| # | Function | File | NumPy Status | Severity | Migration |
|---|----------|------|-------------|----------|-----------|
| 1 | `np.asscalar()` (6 overloads) | `Manipulation/np.asscalar.cs` | Removed in 1.23 | **High** | Replace with `ndarray.item()`. Used internally by `NDArray.amin<T>()` and `NDArray.amax<T>()`. |
| 2 | `np.find_common_type()` (7+ overloads) | `Logic/np.find_common_type.cs` | Removed in 2.0 | **High** | Replace public API with `np.result_type()` / `np.promote_types()`. Internal `_FindCommonType` stays but needs NEP 50 update. Has ~25 dedicated test cases. |
| 3 | `np.round_()` (4 overloads) | `Math/np.round.cs` | Removed in 2.0 | **Medium** | Mark `[Obsolete]`. `np.round()` already exists alongside. No tests call `round_` directly. |
| 4 | `DType.newbyteorder()` | `Creation/np.dtype.cs:114` | Removed in 2.0 | **Low** | Already throws `NotSupportedException`. Consider removing entirely. |
| 5 | `np.asscalar` reference in comments | `LinearAlgebra/np.linalg.norm.cs` | — | **Low** | Commented-out Python code references `asfarray` (removed in 2.0). Clean up comments. |

### 1.2 Aliases & Constants

All defined in `APIs/np.cs`:

| # | Symbol | Line | NumPy 2.0 Status | Action |
|---|--------|------|-----------------|--------|
| 1 | `np.NaN` | 67 | Removed (use `np.nan`) | Mark `[Obsolete]` |
| 2 | `np.NAN` | 66 | Removed | Mark `[Obsolete]` |
| 3 | `np.Inf` | 73 | Removed (use `np.inf`) | Mark `[Obsolete]` |
| 4 | `np.Infinity` | 76 | Removed (use `np.inf`) | Mark `[Obsolete]` |
| 5 | `np.infinity` | 77 | Removed | Mark `[Obsolete]` |
| 6 | `np.infty` | 72 | Removed (use `np.inf`) | Mark `[Obsolete]` |
| 7 | `np.NINF` | 74 | Removed (use `-np.inf`) | Mark `[Obsolete]` |
| 8 | `np.PINF` | 75 | Removed (use `np.inf`) | Mark `[Obsolete]` |
| 9 | `np.float_` | 50 | Removed (use `np.float64`) | Mark `[Obsolete]` |
| 10 | `np.complex_` | 54 | Removed (use `np.complex128`) | Mark `[Obsolete]` |
| 11 | `np.bool8` | 22 | Removed | Remove |
| 12 | `np.int0` | 42 | Removed | Remove |
| 13 | `np.uint0` | 45 | Removed | Remove |

### 1.3 Confirmed Absent (Clean)

These NumPy 1.x deprecated functions are NOT present in NumSharp:
`alltrue`, `sometrue`, `cumproduct`, `product`, `asfarray`, `msort`, `row_stack`, `in1d`, `trapz`, `cast`, `ptp` (method), `setitem` (method), `issubsctype`, `issubclass_`, `maximum_sctype`, `set_string_function`, `set_numeric_ops`, `fastCopyAndTranspose`, `get_array_wrap`, `safe_eval`, `nbytes` (function), `AxisError`, `ComplexWarning`, `VisibleDeprecationWarning`.

---

## 2. Type Promotion Table Analysis

### 2.1 Array-Array Promotion (`_typemap_arr_arr`)

**Result: 100/100 entries match NumPy 2.x — perfect correspondence.**

NumSharp's `_typemap_arr_arr` dictionary in `Logic/np.find_common_type.cs` was compared against `np.result_type(dtype1, dtype2)` for all 100 comparable pairs (10 types x 10 types, excluding complex64/decimal/char which are NumSharp-specific). Every entry matches.

### 2.2 Array-Scalar Promotion (`_typemap_arr_scalar`) — 12 MISMATCHES

**Result: 68/80 entries match NEP 50, 12 diverge.**

All 12 divergences follow one pattern: **unsigned integer array + signed integer scalar**. Under NumPy 1.x, the result widens to accommodate both ranges. Under NEP 50 (NumPy 2.x), the array dtype wins because the scalar is a "weak" integer of the same kind.

| Line | NumSharp (1.x) | NumPy 2.x (NEP 50) | Array Type | Scalar Type |
|------|---------------|---------------------|------------|-------------|
| 258 | (uint8, int16) → int16 | → **uint8** | uint8 | int16 |
| 260 | (uint8, int32) → int32 | → **uint8** | uint8 | int32 |
| 262 | (uint8, int64) → int64 | → **uint8** | uint8 | int64 |
| 297 | (uint16, int16) → int32 | → **uint16** | uint16 | int16 |
| 299 | (uint16, int32) → int32 | → **uint16** | uint16 | int32 |
| 301 | (uint16, int64) → int64 | → **uint16** | uint16 | int64 |
| 323 | (uint32, int16) → int64 | → **uint32** | uint32 | int16 |
| 325 | (uint32, int32) → int64 | → **uint32** | uint32 | int32 |
| 327 | (uint32, int64) → int64 | → **uint32** | uint32 | int64 |
| 349 | (uint64, int16) → float64 | → **uint64** | uint64 | int16 |
| 351 | (uint64, int32) → float64 | → **uint64** | uint64 | int32 |
| 353 | (uint64, int64) → float64 | → **uint64** | uint64 | int64 |

**Fix**: Update these 12 entries so the result equals the array's unsigned type (the "array wins" rule).

### 2.3 Key NEP 50 Behavioral Changes (from Python verification)

Verified against NumPy 2.4.2:

| Expression | NumPy 1.x | NumPy 2.x | NumSharp |
|-----------|----------|----------|----------|
| `float32(3) + 3.0` | float64 | **float32** | Needs testing |
| `float32_arr + float64_scalar` | float32 | **float64** | Needs testing |
| `int8_arr + python_int` | int16+ | **int8** | Needs testing |
| `uint8_arr + python_int` | int16+ | **uint8** | Needs testing |
| `bool_arr + bool_arr` | int (value=2) | **bool** | bool (OR semantics — see Bug #5) |

---

## 3. Behavioral Divergences

### 3.1 floor/ceil on integer arrays — BREAKING

**NumPy 2.1+**: `np.floor(int_array)` returns the array with the same integer dtype (no-op).
**NumSharp**: Always casts to `Double`. The chain: `ResolveUnaryReturnType` → `GetComputingType()` → any integer → `NPTypeCode.Double`. The `Default.Floor.cs` switch only handles Double/Single/Decimal and throws for integer types.

**Location**: `Backends/Default/Math/Default.Floor.cs`, `Default.Ceil.cs`, `NPTypeCode.cs:577`
**Impact**: Any `np.floor` or `np.ceil` call on integer arrays produces float64 output instead of preserving dtype.

### 3.2 np.unique return_inverse — MISSING FEATURE

**NumPy 2.x**: `return_inverse` shape matches input array shape (was 1D in 1.x).
**NumSharp**: `np.unique` accepts only `(NDArray a)` — no `return_index`, `return_inverse`, or `return_counts` parameters exist at all.

### 3.3 np.asarray — MISSING copy PARAMETER

**NumPy 2.x**: `np.asarray(existing_array)` returns a view; `copy=False` raises if copy needed.
**NumSharp**: `np.asarray` only accepts scalars and managed arrays (`T[]`), never an existing `NDArray`. Always creates a new `NDArray` and copies data. No `copy` parameter.

### 3.4 bool + bool semantics — WRONG

**NumPy 2.x**: `np.array([True]) + np.array([True])` returns `int` value `2`.
**NumSharp**: Uses `Operator.Add(bool, bool)` which is `lhs || rhs` (OR), so `True + True = True`. Loses information: `True + True` should be `2`, not `True`.

**Location**: `Utilities/Maths/Operator.cs:32`

### 3.5 np.any(axis) — COMPLETELY BROKEN

Two bugs make `np.any(nd, axis)` always throw `InvalidOperationException`:

1. **Bug**: `ComputeAnyPerAxis<T>` returns `false` (line 162 of `np.any.cs`), causing the caller to always throw.
2. **Bug**: The inner logic implements `all()` instead of `any()` — initializes `currentResult = true` and detects zero values, which is the inverse of correct `any()` behavior.

### 3.6 np.random — LEGACY API ONLY

**NumPy 2.x**: Recommends `np.random.default_rng()` Generator API with PCG64/Philox/SFC64 bit generators. Legacy `RandomState` is deprecated.
**NumSharp**: Uses exclusively the legacy `RandomState` API. The `NumPyRandom` class docs explicitly say "serves as numpy.random.RandomState". No Generator API, no `default_rng()`, no modern bit generators. The underlying `Randomizer` class is a serializable clone of `System.Random`.

---

## 4. Bugs Discovered During Audit

| # | Bug | Location | Severity | Category |
|---|-----|----------|----------|----------|
| 1 | `np.roll` static returns `int` instead of `NDArray` | `APIs/np.array_manipulation.cs:16` | **High** | Implementation bug |
| 2 | `np.fmax`/`np.fmin` have identical implementation to `np.maximum`/`np.minimum` — NaN-ignoring behavior not implemented | `Math/np.maximum.cs:57-89`, `Math/np.minimum.cs:57-89` | **Medium** | Missing behavior |
| 3 | `np.fmin` docstrings say "Element-wise maximum" and "NaNs are propagated" — should say "minimum" and "NaNs are ignored" | `Math/np.minimum.cs:57-89` | **Low** | Wrong docs |
| 4 | `stardard_normal` method name is misspelled (missing 'd') | `RandomSampling/np.random.randn.cs` | **Low** | Typo |
| 5 | `bool + bool` uses OR (`||`) instead of integer addition | `Utilities/Maths/Operator.cs:32` | **Medium** | Wrong semantics |
| 6 | `np.any(axis)` always throws — `ComputeAnyPerAxis` returns false + has inverted logic | `Logic/np.any.cs:130-162` | **High** | Two bugs |
| 7 | `np.convolve` always returns null — Regen template never generated | `Math/NdArray.Convolve.cs` | **Medium** | Dead code |
| 8 | `floor`/`ceil` cast integer inputs to Double | `Backends/Default/Math/Default.Floor.cs`, `NPTypeCode.cs:577` | **Medium** | NumPy 1.x behavior |

---

## 5. NUMPY_NUMSHARP_MAP.md Corrections

### 5.1 Status Corrections

| Function | Current Status | Correct Status | Reason |
|----------|---------------|----------------|--------|
| `np.asscalar` (§23) | `Y` | `Y` **(deprecated)** | Removed in NumPy 1.23 |
| `np.find_common_type` (§11) | `Y` | `Y` **(deprecated)** | Removed in NumPy 2.0 |
| `np.fmax`/`np.fmin` (§6) | `Y` | `~` | NaN-ignoring behavior not actually different from `maximum`/`minimum` |
| `nd.tofile()` (§16) | `Y` at `APIs/np.save.cs` | `Y` at `APIs/np.tofile.cs` | Wrong file path |
| `np.random.normal`/`standard_normal` (§15) | "Part of randn" | Independent impl + typo | `normal()` is standalone; `stardard_normal` is misspelled |

### 5.2 Missing NumPy 2.x Functions (add to map)

| Function | NumPy Version | Section |
|----------|--------------|---------|
| `np.matvec` | New in 2.2 | §13 Linear Algebra |
| `np.vecmat` | New in 2.2 | §13 Linear Algebra |
| `np.bitwise_count` | New in 2.0 | §6 Math |
| `np.isdtype` | New in 2.0 | §11 Logic |

### 5.3 Deprecation Notes (add to map)

These entries in the map should note they correspond to deprecated NumPy APIs:
- `np.row_stack` (§2): "Deprecated alias for vstack, removed in NumPy 2.0"
- `np.trapezoid` (§21): "Renamed from np.trapz in NumPy 2.0"

---

## 6. Documentation URL Audit

### 6.1 Old scipy-hosted URLs

**385 occurrences across 121 files** of `https://docs.scipy.org/doc/numpy/...` in XML doc `<remarks>` tags.

**Fix**: Bulk replace `https://docs.scipy.org/doc/numpy/` → `https://numpy.org/doc/stable/`

### 6.2 Version-pinned NumPy 1.x URLs

**54 occurrences across 22 files** referencing specific 1.x versions:

| Version | Files | Count |
|---------|-------|-------|
| `numpy-1.14.0` | np.random.binomial.cs | 2 |
| `numpy-1.15.0` | np.random.gamma.cs, np.random.exponential.cs, np.random.chisquare.cs, np.clip.cs, np.linspace.cs, np.exp.cs | 16 |
| `numpy-1.16.0` | np.asscalar.cs, np.array_equal.cs, np.dtype.cs, np.reshape.cs, NDArray.Equals.cs, np.cs | 16 |
| `numpy-1.16.1` | Default.Broadcasting.cs, np.random.cs, np.dtype.cs | 11 |
| `numpy-1.17.0` | NDArray.Indexing*.cs, np.cs | 5 |

**Fix**: Replace all with `https://numpy.org/doc/stable/` (unversioned stable docs).

---

## 7. Priority Action Items

### Critical (must fix for NumPy 2.x compatibility)

1. **Update 12 entries in `_typemap_arr_scalar`** — unsigned int array + signed int scalar should return the array's type under NEP 50.
2. **Implement `np.result_type()` and `np.promote_types()`** — replacements for the removed `np.find_common_type()`.
3. **Fix `np.any(axis)`** — two bugs: return value and inverted logic.

### High Priority

4. **Mark `np.asscalar` as `[Obsolete]`** — add `NDArray.item()` method as replacement.
5. **Fix `np.roll` static return type** — should return `NDArray`, not `int`.
6. **Fix `bool + bool` semantics** — should be integer addition, not OR.
7. **Fix `floor`/`ceil` on integer arrays** — should return same dtype, not cast to Double.

### Medium Priority

8. **Mark deprecated aliases `[Obsolete]`** — 13 symbols in `np.cs`.
9. **Mark `np.round_()` as `[Obsolete]`** — 4 overloads.
10. **Fix `np.fmax`/`np.fmin` NaN handling** — should ignore NaN unlike `maximum`/`minimum`.
11. **Fix `np.fmin` docstrings** — says "maximum" instead of "minimum".
12. **Fix `stardard_normal` typo** — rename to `standard_normal`.
13. **Update NUMPY_NUMSHARP_MAP.md** — 5 corrections + 4 new functions.

### Low Priority

14. **Update 385 doc URLs** — bulk replace `docs.scipy.org` → `numpy.org/doc/stable/`.
15. **Remove `DType.newbyteorder()`** — already throws, just dead weight.
16. **Clean up `asfarray` reference in `np.linalg.norm.cs` comments**.
17. **Remove `np.bool8`, `np.int0`, `np.uint0` aliases**.
