# Plan: NumPy 1.x Deprecated Feature Audit in NumSharp

## Status: Investigation Complete — See `numpy-1x-deprecation-findings.md` for results

NumSharp was originally built targeting NumPy 1.x APIs. NumPy 2.0 (June 2024) introduced ~100 API removals, behavioral changes, and type promotion overhauls. This plan documents a systematic investigation to identify every NumSharp feature that corresponds to deprecated, removed, or behaviorally-changed NumPy 1.x functionality, using the NumPy v2.4.2 source tree at `src/numpy/` as the authoritative reference.

## Objectives

1. **Identify deprecated/removed APIs** present in NumSharp that no longer exist in NumPy 2.x
2. **Identify behavioral differences** where NumSharp follows NumPy 1.x semantics that changed in 2.x
3. **Identify type aliases and constants** that were removed or renamed in NumPy 2.0
4. **Categorize findings** by severity: breaking (must fix), warning (should fix), cosmetic (can defer)
5. **Produce actionable tickets** for each finding with migration path

## Reference Sources

| Source | Location | Contents |
|--------|----------|----------|
| NumPy 2.0 Migration Guide | `src/numpy/doc/source/numpy_2_0_migration_guide.rst` | Comprehensive migration instructions, ~100 removed members table |
| NumPy 2.0.0 Release Notes | `src/numpy/doc/source/release/2.0.0-notes.rst` | All removals, deprecations, expired deprecations, behavioral changes |
| NumPy 2.1.0 Release Notes | `src/numpy/doc/source/release/2.1.0-notes.rst` | Post-2.0 deprecations |
| NumPy 2.2.0 Release Notes | `src/numpy/doc/source/release/2.2.0-notes.rst` | Further deprecations |
| NumPy _methods.py | `src/numpy/numpy/_core/_methods.py` | Canonical reduction implementations (mean, var, std) |
| NumPy fromnumeric.py | `src/numpy/numpy/_core/fromnumeric.py` | Function signatures and delegation patterns |
| NumPy numeric.py | `src/numpy/numpy/_core/numeric.py` | Creation functions, element-wise ops |
| NumPy _type_aliases.py | `src/numpy/numpy/_core/_type_aliases.py` | Type hierarchy and promotion |
| **NUMPY_NUMSHARP_MAP.md** | `src/numpy/NUMPY_NUMSHARP_MAP.md` | Complete 1-to-1 API map (394 functions, ~40% coverage) — **must be cross-checked by this audit** |

---

## Phase 1: Removed APIs Still Present in NumSharp

### 1.1 Functions Removed in NumPy 2.0

Cross-reference every function listed in the NumPy 2.0 migration guide removal table against NumSharp's public API.

#### Already Identified

| NumSharp API | NumPy Status | File | Action Required |
|-------------|-------------|------|-----------------|
| `np.asscalar()` | **Removed in NumPy 1.23** (deprecated 1.16) | `Manipulation/np.asscalar.cs` | Replace with `ndarray.item()`. Note: NumSharp references `numpy-1.16.0` docs in remarks. |
| `np.find_common_type()` | **Removed in NumPy 2.0** | `Logic/np.find_common_type.cs` | Replace public API with `np.result_type()` or `np.promote_types()`. Internal `_FindCommonType` can stay but must match NumPy 2.x `result_type` semantics. |
| `np.NaN` (capitalized) | **Removed in NumPy 2.0** | `APIs/np.cs:67` | Keep as alias but mark `[Obsolete]`. NumPy 2.0 only has `np.nan`. |
| `np.Inf` (capitalized) | **Removed in NumPy 2.0** | `APIs/np.cs:73` | Keep as alias but mark `[Obsolete]`. NumPy 2.0 only has `np.inf`. |
| `np.Infinity` | **Removed in NumPy 2.0** | `APIs/np.cs:76` | Mark `[Obsolete]`, use `np.inf`. |
| `np.infty` | **Removed in NumPy 2.0** | `APIs/np.cs:72` | Mark `[Obsolete]`, use `np.inf`. |
| `np.NINF` | **Removed in NumPy 2.0** | `APIs/np.cs:74` | Remove or mark `[Obsolete]`, use `-np.inf`. |
| `np.PINF` | **Removed in NumPy 2.0** | `APIs/np.cs:75` | Remove or mark `[Obsolete]`, use `np.inf`. |
| `np.float_` | **Removed in NumPy 2.0** (use `np.float64`) | `APIs/np.cs:50` | Mark `[Obsolete]`. |
| `np.complex_` | **Removed in NumPy 2.0** (use `np.complex128`) | `APIs/np.cs:54` | Mark `[Obsolete]`. |
| `np.bool8` | **Removed in NumPy 2.0** (dtype alias `bool8` removed) | `APIs/np.cs:22` | Remove. |
| `np.int0` | **Removed in NumPy 2.0** (dtype alias `int0` removed) | `APIs/np.cs:42` | Remove. |
| `np.uint0` | **Removed in NumPy 2.0** (dtype alias `uint0` removed) | `APIs/np.cs:45` | Remove. |

#### Investigation Needed

Search NumSharp for any usage of these NumPy 1.x functions that were removed in 2.0:

| Removed in NumPy 2.0 | Search Pattern | Expected in NumSharp? |
|----------------------|----------------|----------------------|
| `np.alltrue` | `alltrue` | Unlikely (NumSharp uses `np.all`) |
| `np.sometrue` | `sometrue` | Unlikely (NumSharp uses `np.any`) |
| `np.cumproduct` | `cumproduct` | Unlikely (NumSharp uses `np.cumsum` but check for `cumprod`) |
| `np.product` | `product` | Unlikely (NumSharp uses `np.prod`) |
| `np.asfarray` | `asfarray` | Unlikely |
| `np.round_` | `round_` | Check if alias exists alongside `np.round` |
| `np.msort` | `msort` | Unlikely |
| `np.cast` | `np.cast` | Unlikely |
| `np.row_stack` | `row_stack` | Unlikely (deprecated alias for vstack) |
| `np.in1d` | `in1d` | Unlikely (deprecated, use `np.isin`) |
| `np.trapz` | `trapz` | Unlikely |

**Task**: Run grep across `src/NumSharp.Core/` for each pattern above to confirm presence/absence.

### 1.2 ndarray/scalar Methods Removed in NumPy 2.0

| Removed Method | Replacement | Check in NumSharp |
|---------------|-------------|-------------------|
| `ndarray.newbyteorder()` | `arr.view(arr.dtype.newbyteorder(order))` | Search for `newbyteorder` |
| `ndarray.ptp()` | `np.ptp(arr)` | Search for `ptp` |
| `ndarray.setitem()` | `arr[index] = value` | Search for `setitem` |

**Task**: Grep for these method names in NDArray class files.

---

## Phase 2: Type Promotion Changes (NEP 50)

This is the most impactful behavioral change. NumPy 2.0 overhauled type promotion via NEP 50.

### 2.1 Key Changes

| Behavior | NumPy 1.x | NumPy 2.x | NumSharp Current |
|----------|----------|----------|-----------------|
| `float32(3) + 3.0` | float64 | **float32** | Investigate |
| `array([3], dtype=float32) + float64(3)` | float32 | **float64** | Investigate |
| Scalar precision preserved | No (data-dependent) | **Yes** (dtype-dependent) | Investigate |
| Windows default int | int32 | **int64** | NumSharp uses int64 (may already match) |

### 2.2 Investigation Steps

1. **Read NumSharp's type promotion tables** in `np.find_common_type.cs` — the `_typemap_arr_arr` and `_typemap_arr_scalar` dictionaries encode NumPy 1.x promotion rules.
2. **Compare each entry** against NumPy 2.x `result_type` behavior by running Python scripts:
   ```python
   import numpy as np
   # For each combination:
   print(np.result_type(np.float32, np.float64))
   print(np.result_type(np.array([1], dtype=np.float32), np.float64(1.0)))
   ```
3. **Document all differences** between NumSharp's tables and NumPy 2.x `result_type`.
4. **Decide migration strategy**: update tables to match 2.x, or provide both modes.

### 2.3 Specific Promotion Rules to Verify

The `_typemap_arr_scalar` dictionary (lines 242-425 in `np.find_common_type.cs`) encodes the old `find_common_type` behavior with separate array/scalar semantics. NumPy 2.0 replaced this with unified `result_type`. Key entries to verify:

- `(float32, float64_scalar)` → NumSharp says float32; NumPy 2.x says float64 (scalar precision preserved)
- `(int32, int64_scalar)` → NumSharp says int32; NumPy 2.x says int64
- All `(array_type, scalar_type)` combinations where scalar has higher precision

**Task**: Write a Python script that generates the full NumPy 2.x promotion table for all dtype pairs, then compare against NumSharp's hardcoded tables.

---

## Phase 3: Behavioral Changes

### 3.1 Sorting Algorithm Changes

NumPy 2.0 uses SIMD-accelerated sorting (Intel x86-simd-sort, Google Highway). Unstable sorts may return different orderings for equal elements.

**Task**: Check if NumSharp's `argsort` tests assume specific ordering for equal elements.

### 3.2 `np.unique` Return Shape Changes

NumPy 2.0 changed `return_inverse` shape for multi-dimensional inputs.

**Task**: Compare NumSharp's `np.unique` behavior with NumPy 2.x for multi-dim arrays.

### 3.3 `floor`, `ceil`, `trunc` Integer Input Behavior

NumPy 2.1: `floor`, `ceil`, `trunc` no longer cast integer input to float.

**Task**: Check if NumSharp's `np.floor`/`np.ceil` cast integer arrays to float.

### 3.4 `np.any`/`np.all` Object Array Returns

NumPy 2.0: `any`/`all` now return booleans for object arrays (previously returned objects).

**Task**: Not directly applicable to NumSharp (no object dtype), but verify bool return type consistency.

### 3.5 Copy Keyword Behavior

NumPy 2.0 changed `np.array(..., copy=False)` and `np.asarray(...)` copy semantics.

**Task**: Check if NumSharp's `np.array` and `np.asarray` have `copy` parameter and if behavior matches.

### 3.6 Empty Array Truthiness

NumPy 2.0: `bool(np.array([]))` now raises an error (previously returned False).

**Task**: Check NumSharp's behavior for `bool` conversion of empty arrays.

---

## Phase 4: Dead Code Tied to NumPy 1.x

From CLAUDE.md, these are known dead code items. Verify if any correspond to APIs that were restructured in NumPy 2.x:

| Dead Code | Status | NumPy 2.x Status |
|-----------|--------|-------------------|
| `np.linalg.norm` (private) | Dead — declared `private static` | Still exists in NumPy 2.x as `np.linalg.norm` |
| `nd.inv()` → returns null | Dead | `np.linalg.inv` still exists |
| `nd.qr()` → returns default | Dead | `np.linalg.qr` still exists |
| `nd.svd()` → returns default | Dead | `np.linalg.svd` still exists |
| `nd.lstsq()` → named `lstqr`, returns null | Dead | `np.linalg.lstsq` still exists |
| `nd.multi_dot()` → returns null | Dead | `np.linalg.multi_dot` still exists |
| `np.isnan` → engine returns null | Dead | Still exists in NumPy 2.x |
| `np.isfinite` → engine returns null | Dead | Still exists in NumPy 2.x |
| `np.isclose` → engine returns null | Dead | Still exists in NumPy 2.x |
| `np.allclose` → depends on isclose | Dead | Still exists in NumPy 2.x |
| `NDArray & (AND)` → returns null | Dead | Still exists in NumPy 2.x |
| `NDArray \| (OR)` → returns null | Dead | Still exists in NumPy 2.x |
| `nd.delete()` → returns null | Dead | Still exists in NumPy 2.x |
| `nd.roll()` → partial | Partial | Still exists, no API change |

**Conclusion**: These are not NumPy 1.x deprecations — they're unfinished implementations. They should be fixed (implemented properly) or removed, but that's a separate concern from the deprecation audit.

---

## Phase 5: Documentation Link Audit

Many NumSharp files contain XML doc `<remarks>` links pointing to NumPy 1.x documentation URLs.

### Examples Found

- `np.asscalar.cs`: References `numpy-1.16.0` docs
- `np.cs`: References `numpy-1.17.0` and `numpy-1.16.0` docs
- Various other files likely reference `scipy.org/doc/numpy-1.xx.x/`

**Task**: Grep for `numpy-1.` across the codebase to find all 1.x doc references and update to NumPy 2.x URLs (which use `numpy.org/doc/stable/`).

---

## Phase 6: Automated Investigation Scripts

### 6.1 Python Script: Generate NumPy 2.x Type Promotion Table

```python
# Run with NumPy 2.x installed
import numpy as np
import itertools

dtypes = [np.bool_, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
          np.int64, np.uint64, np.float32, np.float64]

print("# Array-Array promotion (np.result_type)")
for d1, d2 in itertools.product(dtypes, repeat=2):
    result = np.result_type(d1, d2)
    print(f"({np.dtype(d1).name}, {np.dtype(d2).name}) -> {result}")

print("\n# Array-Scalar promotion")
for d1 in dtypes:
    for scalar in [True, 0, 0.0]:
        arr = np.array([1], dtype=d1)
        result = np.result_type(arr, type(scalar))
        print(f"array({np.dtype(d1).name}) + {type(scalar).__name__} -> {result}")
```

### 6.2 Grep Commands for Phase 1 Audit

```bash
# Removed function names
rg -i "asscalar|find_common_type|alltrue|sometrue|cumproduct|asfarray" src/NumSharp.Core/ --type cs
rg -i "msort|row_stack|in1d|trapz" src/NumSharp.Core/ --type cs
rg -i "newbyteorder|\.ptp\b|\.setitem\b" src/NumSharp.Core/ --type cs

# Old doc URLs
rg "numpy-1\." src/NumSharp.Core/ --type cs

# Deprecated constant/alias usage in test code
rg "np\.NaN|np\.Inf\b|np\.Infinity|np\.infty|np\.NINF|np\.PINF|np\.float_|np\.complex_|np\.bool8|np\.int0|np\.uint0" src/ --type cs
```

### 6.3 C# Script: Compare NumSharp Promotion Tables Against NumPy 2.x

Write a `dotnet run` script that:
1. Loads NumSharp's `_nptypemap_arr_arr` and `_nptypemap_arr_scalar` tables
2. Compares against expected NumPy 2.x `result_type` values (hardcoded from Python output)
3. Reports all mismatches

---

## Phase 7: Cross-Check NUMPY_NUMSHARP_MAP.md

The file `src/numpy/NUMPY_NUMSHARP_MAP.md` is a comprehensive 394-function API map with status codes (`Y`/`~`/`-`), file locations, and notes. This audit should validate and correct that map.

### 7.1 Deprecation-Related Corrections

Items in the map that are NumPy 1.x-only and need annotation:

| Map Entry | Section | Current Status | Correction Needed |
|-----------|---------|----------------|-------------------|
| `np.asscalar` | §23 Other | `Y` | Should be marked `Y (deprecated)` — removed in NumPy 1.23, replace with `ndarray.item()` |
| `np.find_common_type` | §11 Logic | `Y` | Should be marked `Y (deprecated)` — removed in NumPy 2.0, replace with `np.result_type` |
| `np.row_stack` | §2 Stacking | `-` | Correctly absent. Was deprecated alias for `vstack`, removed in NumPy 2.0. Should note "Removed in NumPy 2.0, was alias for vstack" |
| `np.trapezoid` | §21 Advanced | `-` | Correctly listed as NumPy 2.0 name. Note: old name `np.trapz` also not in NumSharp (clean) |

### 7.2 Map Accuracy Verification — Status Claims

For each `Y` (implemented) entry in the map, verify the claim by checking:
1. The file exists at the stated path
2. The function is public and callable (not dead code)
3. The function signature matches NumPy 2.x (not just NumPy 1.x)

Priority spot-checks (items that may have NumPy 1.x → 2.x signature changes):

| Function | Map Claim | What to Verify |
|----------|-----------|---------------|
| `np.round` / `np.around` | `Y` | Does NumSharp also expose `np.round_`? (Yes — 4 overloads, deprecated in NumPy 2.0) |
| `np.convolve` | `-` (dead) | Map says "Dead code: Regen not generated, returns null". Verify this is accurate. |
| `np.any(axis)` | `~` | Map says "with-axis always throws". Verify against CLAUDE.md dead code list. |
| `np.savez` | `~` | Map says "As `Save_Npz`". Verify this matches NumPy 2.x `savez` signature. |
| `np.roll` | `~` | Map says "static returns int (bug)". Verify. |

### 7.3 Missing Deprecation Annotations

The map should flag these NumPy 2.0 items that NumSharp should NOT implement (since they were removed):

| Should NOT Be Added | Reason |
|---------------------|--------|
| `np.alltrue` | Removed in 2.0, use `np.all` |
| `np.sometrue` | Removed in 2.0, use `np.any` |
| `np.cumproduct` | Removed in 2.0, use `np.cumprod` |
| `np.product` | Removed in 2.0, use `np.prod` |
| `np.asfarray` | Removed in 2.0 |
| `np.msort` | Removed in 2.0 |
| `np.in1d` | Deprecated in 2.0, use `np.isin` |

These are already absent from the map, which is correct.

### 7.4 New NumPy 2.x Functions to Track

The map already identifies some NumPy 2.0+ additions. Verify completeness:

| New in NumPy 2.x | In Map? | Notes |
|-------------------|---------|-------|
| `np.cumulative_sum` | Yes (§9) | New in 2.0 |
| `np.cumulative_prod` | Yes (§9) | New in 2.0 |
| `np.matrix_transpose` | Yes (§5) | New in 2.0 |
| `np.unstack` | Yes (§2) | New in 2.1 |
| `np.unique_all` | Yes (§18) | New in 2.0 |
| `np.unique_counts` | Yes (§18) | New in 2.0 |
| `np.unique_inverse` | Yes (§18) | New in 2.0 |
| `np.unique_values` | Yes (§18) | New in 2.0 |
| `np.linalg.matrix_norm` | Yes (§13) | New in 2.0 |
| `np.linalg.vector_norm` | Yes (§13) | New in 2.0 |
| `np.vecdot` | Yes (§13) | New in 2.0 |
| `np.matvec` | No | New in 2.2 — add to map |
| `np.vecmat` | No | New in 2.2 — add to map |
| `np.bitwise_count` | No | New in 2.0 — add to map |
| `np.astype` (top-level) | Yes (§23) | New in 2.0 |
| `np.from_dlpack` | Yes (§1) | New in 2.0 |
| `np.isdtype` | No | New in 2.0 — add to map |
| `np.trapezoid` | Yes (§21) | Renamed from `trapz` in 2.0 |

### 7.5 Summary Statistics Validation

The map claims **152 implemented, 5 partial, 237 missing** out of **394 total (~40% coverage)**. This audit should:
1. Subtract deprecated functions from the "implemented" count (asscalar, find_common_type → these still count as implemented but need annotation)
2. Verify the dead code items are correctly marked `-` (not `Y`)
3. Add missing NumPy 2.0+ functions to the total

---

## Execution Order

| Phase | Description | Effort | Priority |
|-------|-------------|--------|----------|
| 1.1 | Grep for removed APIs | Low | High |
| 1.2 | Check removed ndarray methods | Low | High |
| 2.1-2.3 | Type promotion comparison | Medium | **Critical** |
| 3.1-3.6 | Behavioral change audit | Medium | High |
| 4 | Dead code review (already known) | Low | Low |
| 5 | Doc URL audit | Low | Low |
| 6 | Write and run automated scripts | Medium | High |
| 7.1-7.2 | Cross-check NUMPY_NUMSHARP_MAP.md deprecations & accuracy | Medium | High |
| 7.3-7.5 | Update map with 2.0 annotations & new functions | Low | Medium |

## Output Artifacts

Each phase produces:
1. A findings table (present/absent, matches/differs)
2. Migration recommendations (fix, deprecate, remove)
3. Test cases to verify corrections

Final deliverables:
- `docs/plans/numpy-1x-deprecation-findings.md` — all results consolidated
- Updated `src/numpy/NUMPY_NUMSHARP_MAP.md` — corrected with deprecation annotations, new 2.x functions, verified status claims

---

## Summary of Known Issues (Pre-Investigation)

### Confirmed Deprecated APIs in NumSharp

| # | Item | Severity | Migration |
|---|------|----------|-----------|
| 1 | `np.asscalar()` | Breaking | Replace with `ndarray.item()` method |
| 2 | `np.find_common_type()` public API | Breaking | Replace with `np.result_type()` / `np.promote_types()` |
| 3 | `np.NaN`, `np.Inf`, `np.Infinity`, `np.infty`, `np.NINF`, `np.PINF` | Warning | Mark `[Obsolete]`, keep for compat |
| 4 | `np.float_`, `np.complex_` | Warning | Mark `[Obsolete]` |
| 5 | `np.bool8`, `np.int0`, `np.uint0` | Warning | Remove aliases |
| 6 | `np.round_()` (4 public overloads) | Warning | Mark `[Obsolete]`, keep `np.round()` as primary. `np.round_` was removed in NumPy 2.0 (use `np.round`). |
| 7 | `DType.newbyteorder()` | Warning | Throws `NotSupportedException` already. `ndarray.newbyteorder()` was removed in NumPy 2.0. Consider removing. |
| 8 | Type promotion tables | **Critical** | Must update to match NEP 50 / `result_type` semantics |
| 9 | Old documentation URLs (54 occurrences across 22 files) | Cosmetic | Bulk find-replace `numpy-1.xx.x` → `numpy.org/doc/stable/` |

### Grep Results Summary (Phase 1 Bootstrap)

**Deprecated functions found in NumSharp:**
- `np.asscalar()` — `Manipulation/np.asscalar.cs` (6 overloads, references numpy-1.16.0 docs)
- `np.round_()` — `Math/np.round.cs` (4 public overloads alongside `np.round`)
- `np.find_common_type()` — `Logic/np.find_common_type.cs` (public API + internal)
- `DType.newbyteorder()` — `Creation/np.dtype.cs:114` (throws NotSupportedException)

**Deprecated functions NOT found in NumSharp (clean):**
- `alltrue`, `sometrue`, `cumproduct`, `product`, `asfarray`, `msort`, `row_stack`, `in1d`, `trapz`, `cast`, `ptp`, `setitem`

**Old documentation URLs:** 54 references to `numpy-1.x` docs across 22 .cs files

**Deprecated aliases/constants used in test code:** No test files use the deprecated aliases (clean)

### Requires Python Verification

| # | Area | Method |
|---|------|--------|
| 1 | Full type promotion table comparison | Run Python script with NumPy 2.x |
| 2 | `floor`/`ceil` integer behavior | Run `np.floor(np.array([1,2,3]))` in NumPy 2.x |
| 3 | `np.unique` return_inverse shape | Run multi-dim unique in NumPy 2.x |
| 4 | Copy semantics of `np.array`/`np.asarray` | Compare `copy=` parameter behavior |
