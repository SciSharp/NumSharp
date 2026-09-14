# `numpy.ma` (masked-array) module — parity audit & gap register

**Scope:** NumSharp `np.ma` / `MaskedArray` vs **NumPy 2.4.2**.
**Source of truth:** NumPy 2.4.2 (`refs/numpy/numpy/ma/{core,extras}.py`), probed live.
**Implementation:** `src/NumSharp.Core/Ma/MaskedArray.cs` (single file). Gate: `test/NumSharp.Tests/Ma/MaskedArrayTests.cs`.

This is a living checklist. Techniques used are recorded so a later pass can re-run them. **Nothing here is "done" until it is bit-exact vs NumPy 2.4.2 and gated.**

> **Progress — 2026-09-14 (two passes).** **Pass 1 (backlog §6 items 1–5, 7):** the predicate/shape-query/copy/
> alias tier (§1.A), complex fill values (§5.3), `make_mask`/`make_mask_none`/`mask_or`, the fill-value surface
> (`fill_value` get/set, `common_fill_value`, `set_fill_value`, `fix_invalid`), `left_shift`/`right_shift`/`clip`/
> `choose`/`compress`/`diagonal`/`nonzero`/`trace`/`diff`/`append`, the parameter-parity fixes
> (`average.keepdims`+`average_returned`, `median.keepdims`, `sort`/`argsort` `endwith`, `array.dtype`,
> `isin.assume_unique`, `vander` `N`→`n`, `reshape` `new_shape`), the `%`/`&`/`|`/`^`/`~` operators (§5.2), a
> broad instance surface, and `default_fill_value` made instance. **Pass 2 (§6 item 6 — THE KEYSTONE):** the
> `MaskedArray this[...]` indexer (get: scalar/masked-singleton/sub-array VIEW-or-COPY; set: value-unmasks /
> `= masked` masks / masked-array propagates / nomask→mask promotion), and everything it unblocks —
> `put`/`putmask`/`resize`, `mask_rowcols`, `compress_rowcols`/`compress_nd`, `notmasked_edges`/`notmasked_contiguous`
> (axis=None / 1-D), and `ids`. All probed against NumPy 2.4.2, verified by 33 more differential checks (0 fail),
> gated by 8 more `MaskedArrayTests` cases (**53 total**, green net8.0/net10.0).
> **Pass 3 (quick wins):** `frombuffer`, `fromfunction` (Func-of-index-grids + a 2-D convenience), `hsplit`
> (data+mask split), `ndenumerate` (unmasked `(index, value)` pairs), and the exception types `MAError`/`MaskError`
> (`MaskError : MAError : Exception`, NumPy's hierarchy). Gated by 2 more `MaskedArrayTests` cases (**55 total**).
> **Still open — all either machinery-blocked or N/A:** masked `cov`/`corrcoef` (pairwise-complete),
> `polyfit` (LAPACK lstsq — backend-only), `apply_along/over_axes` (callable-over-axis), `convolve`/`correlate`
> (sliding masks), N-D-axis `notmasked_*` / masked-axis `median` (masked sort core), the structured/record
> family (`mvoid`/`mr_`/`fromflex`/`flatten_mask`/`flatten_structured_array`/`make_mask_descr` — NumSharp has no
> structured dtypes), and the config ref `masked_print_option` / submodule refs `core`/`extras` (N/A in C#).

---

## 0. Headline coverage

| Surface | NumSharp | NumPy 2.4.2 | Gap |
|---|---|---|---|
| `np.ma.*` module names (`__all__`) | **207** (of 211 public) | 226 | **19 names** (was 62; the rest machinery-blocked / structured / type-or-submodule refs) |
| `MaskedArray` instance members | **57** (+ the indexer) | 94 | **~37 members** (was 69) |

**Behavioral correctness of what IS implemented:** a 186-case differential (3 dtypes × reductions[×3 axes]/ufuncs/constructors/extras/sort/unique/set-ops) found **0 real value/mask/dtype mismatches**, plus 79 + 33 checks of every 2026-09-14 addition against probed NumPy 2.4.2 output (0 failures). The only divergence is the intentional scalar-vs-0d return-type choice (§5.1). Set operations (`intersect1d`/`union1d`/`setxor1d`/`setdiff1d`, added 2026-09-14, commit 566fd303) are bit-exact.

> **Audit-harness trap (do not re-chase):** reading an op result lazily — `nd.astype(f64).GetDouble(i)` in a loop, interleaved with other allocations — returns recycled-buffer GARBAGE (non-deterministic `e+257` values). This is the *harness's* fault, not the library: an inline check of the identical op sequence showed 0 corruption, and materializing each result with `.ToArray<T>()` **before the next allocation** made the diff clean. Rule: differential harnesses MUST snapshot each result to a managed array immediately.

---

## 1. MISSING module functions (was 62 → now 32)

### A. Trivial aliases of already-implemented ops — ✅ DONE (2026-09-14)
| Was missing | Alias of | Status |
|---|---|---|
| `amax` | `max` | ✅ (with `axis`/`fill_value`/`keepdims`) |
| `amin` | `min` | ✅ |
| `alltrue` | `all` | ✅ (deprecated, exported) |
| `sometrue` | `any` | ✅ |
| `round_` | `round` | ✅ |
| `row_stack` | `vstack` | ✅ |
| `innerproduct` | `inner` | ✅ |
| `outerproduct` | `outer` | ✅ |
| `isMA`, `isarray` | `isMaskedArray` | ✅ |
| `masked_singleton` | `masked` (same object) | ✅ |
| `bool_` | `MaskType` (the bool dtype) | ✅ |

### B. Real functional gaps — mostly ✅ DONE (2026-09-14)
| Was missing | What it does | Status |
|---|---|---|
| `is_masked` | **True iff ANY element is masked** (distinct from the `isMaskedArray` type check) | ✅ |
| `common_fill_value` | common fill of two arrays, else null | ✅ |
| `set_fill_value` | set `.fill_value` in place | ✅ (`_fill_value` made settable; `fill_value` get/set property) |
| `make_mask` | coerce array-like → bool mask (shrink) | ✅ |
| `make_mask_none` | all-False mask of a shape | ✅ |
| `mask_or` | OR two masks (nomask-aware) | ✅ |
| `fix_invalid` | mask nan/inf, write fill into data | ✅ |
| `ndim`, `shape`, `size` | module-level shape queries | ✅ |
| `copy` | `ma.copy(a)` (deep-copies data+mask) | ✅ |
| `diff` | masked `np.diff` (n, axis, prepend, append) | ✅ (data via `np.diff`; mask via n-fold adjacent-OR) |
| `append` | masked `np.append` | ✅ (= masked `concatenate`; axis=None flattens) |
| `left_shift`, `right_shift` | bitwise-shift ufuncs | ✅ |
| `clip` | masked clip | ✅ |
| `choose` | masked choose | ✅ (mask chosen by the same index; OR'd with the index's mask) |
| `compress` | masked compress (base of compress_rows/cols) | ✅ |
| `diagonal` | masked diagonal | ✅ |
| `nonzero` | masked nonzero (masked→0, excluded) | ✅ |
| `trace` | masked trace | ✅ (float64 by default — NumPy's `astype(None)` quirk) |
| `put` | in-place scatter (mask-aware) | ✅ (masked values mask the slots, plain values unmask, cycles short values, shrinks) |
| `putmask` | in-place masked put | ✅ (copyto-broadcast; masked values mask, plain values unmask) |
| `resize` | masked resize | ✅ (tiles data AND mask; new array, not in place) |
| `ids` | `(data ptr, mask ptr)` | ✅ (buffer addresses; mask 0 for nomask) |
| `hsplit` | masked hsplit | ✅ (data+mask split → `MaskedArray[]`; `vsplit`/`array_split`/`split`/`dsplit` still absent) |
| `frombuffer` | masked frombuffer | ✅ (unmasked wrapper over `np.frombuffer`) |
| `fromfunction` | masked fromfunction | ✅ (`Func<NDArray[],NDArray>` of index grids + a 2-D `Func<NDArray,NDArray,NDArray>` convenience) |
| `ndenumerate` | `(index, value)` skipping masked | ✅ (`IEnumerable<(long[], object)>`, unmasked only) |

### C. Machinery-blocked (deferred — need infra NumSharp lacks)
| Missing | Blocker |
|---|---|
| `apply_along_axis`, `apply_over_axes` | callable-over-axis machinery |
| `cov`, `corrcoef` | pairwise-complete masked covariance |
| `polyfit` | LAPACK lstsq (backend-only) |
| `convolve`, `correlate` | `propagate_mask` sliding masks |
| `median` (explicit axis on MASKED input) | masked sort core |
| `mr_`, `mvoid`, `fromflex`, `flatten_mask`, `flatten_structured_array`, `make_mask_descr` | structured/record dtypes — NumSharp has NONE |

**✅ Moved OUT of machinery-blocked (2026-09-14, pass 2 — the indexer unblocked them):** `compress_nd`,
`compress_rowcols`, `mask_rowcols` (done via mask-broadcast + `np.compress`), and `notmasked_edges`/
`notmasked_contiguous` for the axis=None / 1-D case (an explicit axis on >1-D — the per-line list-of-lists — is
the one remaining piece, `NotSupportedException` with a clear message).

### D. Types / exceptions / config (not functions)
`MaskedArray` (class ref — exists as a type). ✅ **`MAError`/`MaskError` NOW EXIST** as C# exception types
(`Exceptions/MAError.cs`, `MaskError : MAError : Exception` — NumPy's hierarchy; not yet raised by any `np.ma`
op, so present for type parity). ✅ `bool_` (dtype alias) added. Still absent: `masked_print_option` (print
config) and `core`/`extras` (submodule refs — N/A in C#).

---

## 2. MaskedArray INSTANCE-surface gaps (57 present of 94, + the indexer)

**✅ THE INDEXER LANDED (2026-09-14, pass 2).** `object this[params object[]] { get; set; }`: GET returns the bare
scalar for an unmasked scalar index, the `masked` singleton for a masked one, or a sub-`MaskedArray` (a VIEW for a
basic slice — writes through — a COPY for fancy/boolean); SET writes the data and reconciles the mask (a plain
value UNMASKS, `= masked` masks in place leaving the data, a masked-array value PROPAGATES its mask, and a
previously-`nomask` array gains a real mask on the first masking write). Returns `object` because NumPy's indexer
is polymorphic and a C# indexer can't switch its return type on the runtime index — a slice read is cast to
`MaskedArray`. This unblocked `put`/`putmask`/`resize`/`mask_rowcols`/`compress_rowcols`/`notmasked_*`.

**Present (57):** `all any anom argmax argmin argsort astype clip compress compressed conj conjugate count cumprod cumsum data diagonal dot dtype fill_value filled flatten harden_mask imag item mask max mean min mT ndim nonzero prod ptp put ravel real repeat reshape round shape shrink_mask size soften_mask sort squeeze std sum swapaxes T take trace transpose typecode var` (+ the `this[...]` indexer + operators `+ - * / % & | ^ ~ < > <= >=`).

**✅ Wired through 2026-09-14:** `T mT flatten ravel reshape transpose swapaxes squeeze repeat take compress compressed sort argsort clip round conj conjugate dot trace diagonal nonzero real imag item fill_value(get/set) harden_mask soften_mask shrink_mask`. NumPy's METHOD forms `get_fill_value()`/`set_fill_value()` are deliberately NOT offered — a C# property named `fill_value` reserves those exact accessor names (CS0082), so the property + the module-level `set_fill_value(a, v)` cover them.

**Still missing instance members:** `flat` (would collide with NumSharp's raveled-array `.flat`), `tolist`, `tobytes`, `view`, `ids`, `iscontiguous`, `put`/`putmask`/`resize` (need mutation), the `hardmask`/`sharedmask`/`recordmask` flags (NumSharp masks carry no hard/soft state — `harden_mask`/`soften_mask` are accepted no-ops), `torecords`/`toflex` (structured/record — NumSharp has none).

---

## 3. Parameter-parity gaps on IMPLEMENTED functions

| Function | Status |
|---|---|
| `average` | ✅ `keepdims` added; `returned` covered by the sibling `average_returned` → `(avg, sumOfWeights)` tuple (the base-library convention, since C# can't switch return type on a flag) |
| `median` | ✅ `keepdims` added (flat + unmasked-axis paths). `out`/`overwrite_input` still absent; masked-axis still `NotSupportedException` (needs the masked sort core) |
| `sum`/`prod`/`mean`/… | `out=` still absent (the library-wide policy gap, §5.5 — NumSharp models no `out=` on any masked reduction) |
| `array`/`masked_array` | ✅ **`dtype`** added; `keep_mask`/`hard_mask`/`shrink`/`ndmin`/`order`/`subok` still absent (hard/soft-mask and ndmin have no NumSharp analog) |

---

## 4. Behavioral divergences (verified, intentional)

### 4.1 Flat reductions return a 0-d MaskedArray, NumPy returns a bare scalar
`np.ma.sum/mean/min/max/prod/ptp/std/var/median(x)` (no axis) → NumSharp yields a 0-d `MaskedArray`; NumPy yields a NumPy scalar. **Values identical.** 27/186 differential cases. Documented as a deliberate type-consistency choice, but it is observable (`type()`, scalar-expecting call sites).

---

## 5. Additional findings — deeper techniques (round 2)

### 5.1 Behavioral parameter gaps (from full signature sweep of all 148 shared functions)
Meaningful, result-changing params — most now closed (2026-09-14):
| Function | Param | Status |
|---|---|---|
| `sort`, `argsort` | **`endwith`** | ✅ `endwith=False` fills masked with `maximum_fill_value` → sorts to FRONT (mirror of the `True` default) |
| `isin` | **`assume_unique`** | ✅ added (speed hint; result unchanged) |
| `array` | **`dtype`** | ✅ added |
| `average` | `returned`,`keepdims` | ✅ (`keepdims` + `average_returned`) |
| `median` | `keepdims` | ✅ (flat/unmasked-axis) |
| `std`, `var` | **`mean`** (NumPy 2.0) | ⛔ precomputed-mean fast path — not added (low value) |
| `take` | **`mode`** | ⛔ `clip`/`wrap`/`raise` — not added |
| `dot` | **`strict`** | ⛔ strict mask propagation — not added |
| `squeeze` | **`axis`** | ⛔ squeeze a specific axis — not added |
| `power` | **`third`** | ⛔ 3-arg `pow(a,b,mod)` — not added |
| `sort`/`argsort` | `kind`,`stable`,`order` | ⛔ not added (NumSharp sort is always stable radix) |

**Parameter NAME mismatches — ✅ FIXED (2026-09-14):**
- `vander(x, n=…)` — was `N`; now lowercase **`n`**, so `np.ma.vander(x, n: 3)` ports verbatim.
- `reshape(a, new_shape=…)` — the param is now **`new_shape`** (`np.ma.reshape(a, new_shape: …)`).

### 5.2 Operator-surface gaps on `MaskedArray` — ✅ MOSTLY FIXED (2026-09-14)
| Operator | Status |
|---|---|
| `%` | ✅ added → `np.ma.remainder` (division-domain: div-by-zero masked) |
| `&` `|` `^` | ✅ added → `np.ma.bitwise_and`/`or`/`xor` (mask = OR of operands') |
| `~` | ✅ added → internal `Invert` = `np.bitwise_not` on the data (NumPy has no `np.ma.invert` module function, so the operator is the only spelling) |
| `==` `!=` | **STILL reference equality → scalar bool** — deliberately not overloaded (avoids the `==null` elementwise trap); use `np.ma.equal`/`not_equal`. Documented porting footgun, unchanged. |
| `//` `**` | n/a (C# has no such operator) — `np.ma.floor_divide`/`power` cover them |

All the mixed `(MaskedArray, NDArray)` / `(…, object)` overloads are provided for each new operator, matching the existing arithmetic-operator pattern (avoids the CS0034 ambiguity the implicit `NDArray→MaskedArray` conversion would otherwise cause).

### 5.3 Complex fill values — ✅ FIXED (2026-09-14): now COMPLEX
| Call | Now returns | NumPy 2.4.2 |
|---|---|---|
| `default_fill_value(complex)` | `new Complex(1e20, 0)` | `(1e20+0j)` ✅ |
| `minimum_fill_value(complex)` | `new Complex(+inf, +inf)` | `(inf+infj)` ✅ |
| `maximum_fill_value(complex)` | `new Complex(-inf, -inf)` | `(-inf-infj)` ✅ |

The fix closes the latent divergence: a complex masked `min`/`max` now fills masked slots with `inf+infj` (both components +inf), so a masked slot sorts strictly last in NumPy's real-then-imag lexicographic order and can never be wrongly selected as the extremum when a real `inf+kj` value is present. `default_fill_value` was also made an INSTANCE method (was `static`), so `np.ma.default_fill_value(dtype)` now works and it's consistent with `minimum/maximum_fill_value`.

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

1. ✅ **DONE** — `is_masked` + module `ndim`/`shape`/`size` + `copy` + alias tier (§1.A).
2. ✅ **DONE** — Parameter parity: `average.keepdims`+`average_returned`, `median.keepdims`, `sort`/`argsort` `endwith`, `array.dtype`, `isin.assume_unique`, `vander` `N`→`n`, `reshape` `new_shape` (§5.1). *(Left: `std`/`var`.`mean`, `take.mode`, `dot.strict`, `squeeze.axis`, `power.third` — low-value.)*
3. ✅ **DONE** — Operators `%` `&` `|` `^` `~` on MaskedArray (§5.2). `==`/`!=` kept as documented reference-equality (use `np.ma.equal`/`not_equal`).
4. ✅ **DONE** — Complex fill values → complex (§5.3).
5. ✅ **DONE** — Instance surface: `fill_value` (get/set) + `T`/`mT`/`real`/`imag`/`ravel`/`flatten`/`reshape`/`transpose`/`swapaxes`/`squeeze`/`repeat`/`take`/`compressed`/`compress`/`sort`/`argsort`/`clip`/`round`/`conj`/`dot`/`diagonal`/`trace`/`nonzero`/`item` (§2). *(Left: `tolist`, `tobytes`, `view`.)*
6. ✅ **DONE (pass 2)** — the `MaskedArray this[...]` indexer get/set (§2), plus the mutation trio `put`/`putmask`/
   `resize`, `mask_rowcols`, `compress_rowcols`/`compress_nd`, `notmasked_edges`/`notmasked_contiguous` (axis=None/1-D),
   and `ids`.
7. ✅ **DONE** — Real functional gaps §1.B: `fix_invalid`, `diff`, `append`, `clip`, `choose`, `compress`, `diagonal`, `nonzero`, `trace`, `make_mask`/`make_mask_none`, `mask_or`, `common_fill_value`, `set_fill_value`, `left_shift`/`right_shift`, `put`/`putmask`/`resize`, `ids`, `frombuffer`, `fromfunction`, `hsplit`, `ndenumerate`, plus the `MAError`/`MaskError` types. *(Left: the `vsplit`/`array_split`/`split`/`dsplit` split family.)*
8. ⛔ **NEXT / Machinery-blocked §1.C** (deferred — need infra NumSharp lacks): masked `cov`/`corrcoef` (pairwise-complete covariance), `polyfit` (LAPACK lstsq — backend-only), `apply_along/over_axes` (callable-over-axis), `convolve`/`correlate` (sliding masks), N-D-axis `notmasked_*`, masked-axis `median` (masked sort core), and the structured/record family (`mvoid`/`mr_`/`fromflex`/`flatten_mask`/`flatten_structured_array`/`make_mask_descr` — no structured dtypes in NumSharp).

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

9. **2026-09-14 verification pass** — every addition was cross-checked with a 79-assertion `dotnet run` script (`#:project NumSharp.Core`, signed with `Open.snk`, `AssemblyName=NumSharp.DotNetRunScript` to reach internals) comparing against the probed NumPy 2.4.2 values (0 failures), then gated by 16 new MSTest cases. **Parallel-session isolation trap:** the shared working tree held an untracked, half-written `src/NumSharp.Core/APIs/np.vectorize.cs` (a different session's WIP) that broke the whole-Core test build with an unrelated CS1503 — the *incremental* Core builds skipped it, but the test project's fuller rebuild hit it. The fix: build/test in a throwaway `git worktree add <scratch> HEAD` (untracked files are NOT carried into a new worktree), copy in only the two edited files (`Ma/MaskedArray.cs`, `Ma/MaskedArrayTests.cs`), and run there — never touch the other session's file.

