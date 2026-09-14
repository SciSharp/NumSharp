# `numpy.ma` (masked-array) module — parity audit & gap register

**Scope:** NumSharp `np.ma` / `MaskedArray` vs **NumPy 2.4.2**.
**Source of truth:** NumPy 2.4.2 (`refs/numpy/numpy/ma/{core,extras}.py`), probed live.
**Implementation:** `src/NumSharp.Core/Ma/MaskedArray.cs` (single file). Gate: `test/NumSharp.Tests/Ma/MaskedArrayTests.cs`.

This is a living checklist. Techniques used are recorded so a later pass can re-run them. **Nothing here is "done" until it is bit-exact vs NumPy 2.4.2 and gated.**

---

## 0. Headline coverage

| Surface | NumSharp | NumPy 2.4.2 | Gap |
|---|---|---|---|
| `np.ma.*` module names (`__all__`) | 164 | 226 | 62 names |
| `MaskedArray` instance members | 25 | 94 | 69 members |

**Behavioral correctness of what IS implemented:** a 186-case differential (3 dtypes × reductions[×3 axes]/ufuncs/constructors/extras/sort/unique/set-ops) found **0 real value/mask/dtype mismatches**. The only divergence is the intentional scalar-vs-0d return-type choice (§5.1). Set operations (`intersect1d`/`union1d`/`setxor1d`/`setdiff1d`, added 2026-09-14, commit 566fd303) are bit-exact.

> **Audit-harness trap (do not re-chase):** reading an op result lazily — `nd.astype(f64).GetDouble(i)` in a loop, interleaved with other allocations — returns recycled-buffer GARBAGE (non-deterministic `e+257` values). This is the *harness's* fault, not the library: an inline check of the identical op sequence showed 0 corruption, and materializing each result with `.ToArray<T>()` **before the next allocation** made the diff clean. Rule: differential harnesses MUST snapshot each result to a managed array immediately.

---

## 1. MISSING module functions (62)

### A. Trivial aliases of already-implemented ops — quick wins
| Missing | Alias of | Notes |
|---|---|---|
| `amax` | `max` | not `is`-identical in NumPy but functionally equal |
| `amin` | `min` | |
| `alltrue` | `all` | deprecated in NumPy, still exported |
| `sometrue` | `any` | deprecated |
| `round_` | `round` | true alias |
| `row_stack` | `vstack` | true alias |
| `innerproduct` | `inner` | true alias |
| `outerproduct` | `outer` | true alias |
| `isMA`, `isarray` | `isMaskedArray` | true aliases |

### B. Real functional gaps — implementable now (compositions over existing np.*/np.ma.*)
| Missing | What it does | Note |
|---|---|---|
| `is_masked` | **True iff ANY element is masked** | DISTINCT from `isMaskedArray` (a type check). `np.ma.is_masked(nomask array)` → False. |
| `common_fill_value` | common fill of two arrays, else null | |
| `set_fill_value` | set `.fill_value` in place | needs settable fill_value |
| `make_mask` | coerce array-like → bool mask (shrink) | |
| `make_mask_none` | all-False mask of a shape | |
| `mask_or` | OR two masks (nomask-aware) | we have private `Or`; expose it |
| `fix_invalid` | mask nan/inf, keep data/fill | like `masked_invalid` but returns filled data at masked |
| `ndim`, `shape`, `size` | module-level shape queries | exist only as instance props |
| `copy` | `ma.copy(a)` | |
| `diff` | masked `np.diff` (n, axis, prepend, append) | |
| `append` | masked `np.append` | |
| `left_shift`, `right_shift` | bitwise-shift ufuncs | binary ufunc family |
| `clip` | masked clip | |
| `choose` | masked choose | |
| `compress` | masked compress (base of compress_rows/cols) | |
| `diagonal` | masked diagonal | |
| `nonzero` | masked nonzero | |
| `trace` | masked trace | |
| `put` | in-place scatter (mask-aware) | needs instance indexing or data/mask write |
| `putmask` | in-place masked put | |
| `resize` | masked resize | |
| `hsplit` | masked hsplit | (we have vsplit? verify) |
| `frombuffer` | masked frombuffer | |
| `fromfunction` | masked fromfunction | |
| `ids` | `(data ptr, mask ptr)` | low value |
| `ndenumerate` | `(index, value)` skipping masked | iterator |

### C. Machinery-blocked (deferred — need infra NumSharp lacks)
| Missing | Blocker |
|---|---|
| `apply_along_axis`, `apply_over_axes` | callable-over-axis machinery |
| `cov`, `corrcoef` | pairwise-complete masked covariance |
| `polyfit` | LAPACK lstsq (backend-only) |
| `convolve`, `correlate` | `propagate_mask` sliding masks |
| `notmasked_edges`, `notmasked_contiguous` | needs MaskedArray slicing/indexing |
| `compress_nd`, `compress_rowcols`, `mask_rowcols` | axis-boolean-index on NDArray |
| `median` (explicit axis on MASKED input) | masked sort core |
| `mr_`, `mvoid`, `fromflex`, `flatten_mask`, `flatten_structured_array`, `make_mask_descr` | structured/record dtypes — NumSharp has NONE |

### D. Types / exceptions / config (not functions)
`MaskedArray` (class ref — exists as a type, not on `np.ma`), `MAError`, `MaskError` (exception types — missing), `bool_` (dtype alias), `masked_print_option` (print config), `core`/`extras` (submodule refs — N/A in C#).

---

## 2. MaskedArray INSTANCE-surface gaps (25 present of 94)

**Biggest gap — no indexer.** `MaskedArray` has NO `this[...]` get/set. NumPy: `x[1]`→`masked`, `x[0]`→scalar, `x[::2]`→sub-masked-array, `x[mask]`→compressed, `x[i]=v` / `x[i]=masked`. This blocks `notmasked_contiguous`, `mask_rowcols`, `put`, and idiomatic use.

**Present (25):** `all any anom argmax argmin astype count cumprod cumsum data dtype filled mask max mean min ndim prod ptp shape size std sum typecode var` (+ operators).

**Missing instance members that exist as `np.ma.*` module funcs (wire through):** `T mT flat flatten ravel reshape transpose swapaxes squeeze repeat take compress compressed sort argsort clip round conj conjugate dot trace nonzero put harden_mask soften_mask shrink_mask`.

**Missing instance members with no module equivalent:** `fill_value` (get/set), `get_fill_value`/`set_fill_value`, `real`, `imag`, `item`, `tolist`, `tobytes`, `view`, `count` (have), `ids`, `iscontiguous`, `hardmask`/`sharedmask`/`recordmask` flags, `torecords`/`toflex` (structured).

---

## 3. Parameter-parity gaps on IMPLEMENTED functions

| Function | NumSharp params | NumPy params | Missing |
|---|---|---|---|
| `average` | `a, axis, weights` | `a, axis, weights, returned, keepdims` | **`returned`**, `keepdims` |
| `median` | `a, axis` | `a, axis, out, overwrite_input, keepdims` | **`keepdims`**, `out`, `overwrite_input` |
| `sum`/`prod`/`mean`/… | `a, axis, dtype, keepdims` | `+ out` | `out=` |
| `array`/`masked_array` | `data, mask, fill_value, copy` | `+ dtype, order, keep_mask, hard_mask, shrink, ndmin, subok` | **`dtype`**, `keep_mask`, `hard_mask`, `shrink`, `ndmin` |

---

## 4. Behavioral divergences (verified, intentional)

### 4.1 Flat reductions return a 0-d MaskedArray, NumPy returns a bare scalar
`np.ma.sum/mean/min/max/prod/ptp/std/var/median(x)` (no axis) → NumSharp yields a 0-d `MaskedArray`; NumPy yields a NumPy scalar. **Values identical.** 27/186 differential cases. Documented as a deliberate type-consistency choice, but it is observable (`type()`, scalar-expecting call sites).

---

## 5. Additional findings — deeper techniques (round 2)

### 5.1 Behavioral parameter gaps (from full signature sweep of all 148 shared functions)
Meaningful, result-changing params NumSharp lacks (excludes the library-wide policy gaps in §5.5):
| Function | Missing param | Effect |
|---|---|---|
| `sort`, `argsort` | **`endwith`** (+`kind`,`stable`,`order`) | `endwith=False` sorts masked to the FRONT; NumSharp only does the `True` default |
| `isin` | **`assume_unique`** | speed hint (result same); NumSharp lacks the overload |
| `std`, `var` | **`mean`** (NumPy 2.0) | precomputed-mean fast path |
| `take` | **`mode`** | `clip`/`wrap`/`raise` out-of-bounds behavior |
| `dot` | **`strict`** | strict mask propagation |
| `squeeze` | **`axis`** | squeeze a specific axis only |
| `power` | **`third`** | 3-arg `pow(a,b,mod)` |
| `median` | `keepdims`,`out`,`overwrite_input` | (also §3) |
| `average` | `returned`,`keepdims` | (also §3) |

**Parameter NAME mismatches (a NumPy-spelled keyword call won't compile):**
- `vander(x, N=…)` — NumPy spells it **`n`** (lowercase). `np.ma.vander(x, n=3)` fails in NumSharp.
- `reshape(a, shape=…)` — NumPy spells it **`new_shape`**.

### 5.2 Operator-surface gaps on `MaskedArray`
| Operator | Status | NumPy | Note |
|---|---|---|---|
| `%` | **THROWS** (no operator) | elementwise mod | `np.ma.mod`/`remainder` methods exist |
| `&` `|` `^` | **THROW** (no operators) | bitwise elementwise | `np.ma.bitwise_and`/`or`/`xor` methods exist |
| `~` | **THROWS** (no operator) | bitwise invert | no `np.ma.invert` function EITHER |
| `==` `!=` | **REFERENCE equality → scalar bool** | elementwise **masked bool array** | **Silent porting footgun.** Deliberately not overloaded (avoids the `==null` elementwise trap); use `np.ma.equal`/`not_equal`. NumPy `ma1==ma2` returns a masked bool array. |
| `//` `**` | n/a (C# has no such operator) | — | `np.ma.floor_divide`/`power` cover them |

### 5.3 Complex fill values return a REAL scalar, not complex
| Call | NumSharp | NumPy 2.4.2 |
|---|---|---|
| `default_fill_value(complex)` | `1e20` (double) | `(1e20+0j)` |
| `minimum_fill_value(complex)` | `inf` (double) | `(inf+infj)` |
| `maximum_fill_value(complex)` | `-inf` (double) | `(-inf-infj)` |

**Latent effect:** a complex masked `min`/`max`/`filled` fills masked slots with `inf+0j` (from the real `inf`) where NumPy uses `inf+infj` — can diverge for complex masked reductions whose extremum sits under the mask. **Worth fixing.** (int64/uint64 min/max fills were verified EXACT — an earlier "diff" was double-rounding in the audit formatter.)

### 5.4 Error-type divergences (message verbatim, .NET type differs — house convention)
- `negative(bool)` → `NotSupportedException` vs NumPy `TypeError` (**message byte-identical**).
- shape mismatch → `IncorrectShapeException` vs NumPy `ValueError` (message identical; documented house convention).
- bad axis → `AxisError` (matches, plus the `(Parameter 'axis')` suffix .NET appends).

### 5.5 Library-wide policy gaps (consistent, lower priority — NumSharp models none of these anywhere)
`out=` absent on every reduction/ufunc; `subok`/`order`/`device`/`like`/`casting` absent on creation/stack funcs; NumPy's `*args,**kwargs` ufunc passthrough is not a real param.

### 5.6 Confirmed CORRECT across all 13 NumPy dtypes — do NOT re-flag
`sum/prod/mean/std/var/min/max/count/anom/cumsum/abs/negative/add/multiply/divide/sqrt/power/compressed/filled/average` are bit-exact for bool/int8/uint8/int16/uint16/int32/uint32/int64/uint64/float16/float32/float64/complex128 (modulo §4.1 scalar-vs-0d and the excused float16 `std`/`var` prefer-precise divergence — NumSharp is *more* accurate there). masked-singleton propagation (`add(masked,x)`, `sqrt(masked)`, all-masked reduction → `masked`) is correct.

---

## 6. Prioritized backlog (recommended order)

1. **`is_masked`** + module `ndim`/`shape`/`size` + `copy` + alias tier (§1.A) — trivial, high-parity-yield.
2. **Parameter parity** on hot functions: `average.returned`/`keepdims`, `median.keepdims`, `std`/`var`/`sort`/`argsort` (`endwith`), `array.dtype`, `isin.assume_unique`, and the `vander` `N`→`n` / `reshape` `shape`→`new_shape` name fixes (§5.1).
3. **Operators** `%` `&` `|` `^` `~` on MaskedArray (§5.2) — wire to existing `np.ma.*` funcs. Decide `==`/`!=` policy (document loudly or add a named method).
4. **Complex fill values** → complex (§5.3).
5. **Instance surface**: `fill_value` (get/set) + wire `T`/`flatten`/`ravel`/`reshape`/`sort`/`compressed`/`round`/`clip`/`conj`/`real`/`imag`/`tolist`/`item` through to the module funcs (§2).
6. **The `MaskedArray` indexer** `this[...]` get/set (§2) — unlocks `notmasked_contiguous`, `mask_rowcols`, `put`.
7. Real functional gaps §1.B (`fix_invalid`, `diff`, `clip`, `choose`, `compress`, `diagonal`, `nonzero`, `trace`, `make_mask*`, `mask_or`, `common_fill_value`, `set_fill_value`, `left_shift`/`right_shift`, …).
8. Machinery-blocked §1.C (deferred): `cov`/`corrcoef`, `polyfit`, `apply_along/over_axes`, `convolve`/`correlate`, `notmasked_*`, structured/record family.

---

## 7. Technique log (reproducible; scripts in session scratchpad)
1. **Surface diff** — `dir(np.ma.__all__)` (226) vs C# reflection over `MaskedArrayModule` (164); `MaskedArray` members 25 vs 94; also `dir(np.ma)` − `__all__` (nothing but `test`).
2. **Full signature parity** — `inspect.signature` per NumPy func vs C# reflected param names (148 shared).
3. **Behavioral differential** — 186 cases (int64/float64/float32 × reductions[×3 axes]/ufuncs/constructors/extras/sort/unique) — **0 real mismatches** with defensive reads.
4. **All-13-dtype + edge sweep** — 314 cases (every NumPy dtype × core ops + empty/0-d/all-masked/nomask/neg-axis/keepdims/broadcast).
5. **Operators** — `+ - * / % & | ^ ~ == != < > <= >=` on MaskedArray.
6. **Constants** — `masked` singleton propagation; `nomask`.
7. **Error parity** — bad axis / shape mismatch / bool-negative.
8. **Exact fill values** — `default_fill_value`/`minimum_fill_value`/`maximum_fill_value` × 15 dtypes.

**Harness rule (learned the hard way):** every NumSharp op result MUST be snapshot to a managed array (`.ToArray<T>()`) **before the next allocation** — a lazy `nd.astype(f64).GetDouble(i)` loop straddling later allocations reads recycled-buffer GARBAGE (non-deterministic). This is a *harness* pitfall, verified NOT a library bug (identical op sequences checked inline showed 0 corruption).

