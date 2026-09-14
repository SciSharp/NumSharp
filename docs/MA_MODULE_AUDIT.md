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
> **Pass 4:** masked `cov` / `corrcoef` — a port of NumPy's `_covhelper`+`cov`+`corrcoef`: each variable is
> centered by its own UNMASKED mean and each covariance entry divides by the count of observations where BOTH
> variables are unmasked (PAIRWISE-COMPLETE), masking any entry with no complete pair; `corrcoef` normalizes by
> the outer product of the per-variable stds. Composed over `dot`/`filled`/masked-`mean`/`diagonal`/`outer` —
> GEMM-bound like the base `np.cov` (byte-exact small, allclose large). Verified against NumPy 2.4.2 (uniform +
> pairwise-mask + bias + rowvar), 1 more `MaskedArrayTests` case (**56 total**).
> **Pass 5:** `convolve`/`correlate` (mask propagated by sliding the boolean masks against ones — the exact
> `_convolve_or_correlate` composition, `propagate_mask` both ways) and `apply_over_axes` (repeated masked
> reduction over each axis, re-expanding a keepdims-less result).
> **Pass 6:** `apply_along_axis` — hands each 1-D masked slice (via the indexer) to the function and assembles
> the results; a scalar per-slice result drops the axis (an all-masked slice → `masked`), a 1-D result replaces
> it (mask preserved), a rank ≥ 2 result throws (needs the object-array assembly NumSharp lacks).
> **✅ THE np.ma MODULE IS NOW FUNCTIONALLY COMPLETE FOR NumSharp.Core (59 `MaskedArrayTests`, green).** The 13
> remaining `__all__` names are NONE-of-them-a-functional-gap: `MAError`/`MaskError`/`MaskedArray` all EXIST as C#
> types; `core`/`extras`/`masked_print_option` are submodule/config refs (N/A in C#); the structured/record family
> (`mvoid`/`mr_`/`fromflex`/`flatten_mask`/`flatten_structured_array`/`make_mask_descr`) needs structured dtypes
> NumSharp has no analog for; and `polyfit` is blocked on `np.polyfit` (which doesn't exist in Core — it needs
> the backend-only LAPACK lstsq, a Core-linalg gap, not an ma gap). The only non-`__all__` deferrals are N-D-axis
> `notmasked_edges`/`notmasked_contiguous` (list-of-lists) and masked-axis `median` (masked sort core).
>
> **Progress — 2026-09-14 (pass 7 — closing the remaining tractable findings).** Two premises in the earlier
> passes had gone STALE and their findings are now CLOSED, plus three real parameter gaps: **`ma.polyfit`**
> (`np.polyfit` DOES exist in Core — `Polynomial/np.polyfit.cs`; `ma.polyfit` is now the faithful ~30-line NumPy
> port — union the operand masks, drop the masked observations, delegate — and inherits `np.polyfit`'s backend
> requirement, throwing the identical `OpenBlasMissingBackendException` with no LAPACK present, never a silent
> wrong fit); **masked-axis `median`** (a port of NumPy's `_median` axis branch composed over the EXISTING
> `ma.sort`/`count`/`take_along_axis`/`where` — no new masked-sort core was needed after all: masked entries sort
> behind the valid ones, the low/high middle of each slice's unmasked count is averaged, an all-masked slice →
> `masked`, an unmasked NaN forces the slice's median to NaN, keepdims/negative-axis/1-D/int→float64 all match);
> **`dot.strict`** (mask propagated along the contracted axes first — NumPy's `_mask_propagate`); **`take.mode`**
> ("raise"/"wrap"/"clip", applied to the data AND mask gather); **`squeeze.axis`** (drop one named size-1 axis).
> The **split-family** note ("`vsplit`/`array_split`/`split`/`dsplit` still absent") was a DOCUMENTATION ERROR:
> `numpy.ma.__all__` exports ONLY `hsplit` from that family (verified 2.4.2), so NumSharp already matches NumPy
> and there is nothing to add. All five closures verified bit-exact against probed NumPy 2.4.2 (22/22 differential
> checks, 0 fail) and gated by **5 more `MaskedArrayTests` (64 total, green net8.0/net10.0)**. The remaining
> N-D-axis `notmasked_edges`/`notmasked_contiguous` are NOT machinery-blocked (NumSharp has the machinery) — they
> are RETURN-TYPE-impedance deferrals: NumPy returns a polymorphic Python list (`ndarray`-or-list /
> list-or-list-of-lists) that C# cannot express without either an `object` return that breaks the already-gated
> typed flat form (`long[]`/`Slice[]`) or a result-struct over-engineered for two niche functions.
>
> **Progress — 2026-09-14 (pass 8 — the return-type-impedance and last-param findings CLOSED).** Chose the honest
> mirror of NumPy's own polymorphism (the project accepts breaking changes for parity) and closed the flagship
> remaining functional gap: **N-D-axis `notmasked_edges` and 2-D-axis `notmasked_contiguous`** now return
> `object` — the flat/1-D case still yields the same `long[]`/`Slice[]` VALUES (boxed; the one flat test casts),
> and the axis case yields `NDArray[][]` (`{mins, maxs}`, one compressed coord array per dimension) /
> `Slice[][]` (one run-list per line), byte-for-byte NumPy's `[tuple(mins), tuple(maxs)]` /
> list-of-lists (`>2-D` raises NumPy's "Currently limited to at most 2D array."). Plus the last two real param
> gaps: **`power`'s third (modulus) arg** (rejected with NumPy's `MaskError "3-argument power not supported."` —
> the `MaskError` type added pass 3 now has its first raiser), and **`var`/`std`'s NumPy-2.0 `mean`** — `var`
> centers by the supplied mean, while **`std` ACCEPTS but IGNORES it** (a genuine NumPy quirk: `ma.std` forwards
> only `keepdims` to `var`, so `ma.std(mean=X)` is the PLAIN std — reproduced exactly). All verified bit-exact
> against probed NumPy 2.4.2 (7 N-D/power + 8 var-mean checks, 0 fail) and gated by **4 more `MaskedArrayTests`
> (68 total, green net8.0/net10.0)**. **What is left is now ONLY the genuinely-blocked or policy items:** the
> structured/record family (no structured dtypes in NumSharp), the library-wide `out=`/`subok`/`order` gaps, and
> the low-value instance `tolist`/`tobytes`/`view`/`flat` (semantic mismatches — `tolist` needs object/None
> arrays, `view` a dtype reinterpret, `tobytes` a bytes path — none a masked-array behavior).
>
> **Progress — 2026-09-14 (pass 9 — the instance conversion/export surface + REAL hard-mask semantics CLOSED).**
> The "low-value instance members" and "hard/soft mask is a no-op" findings were both closeable after all — the
> first three of the four so-called semantic mismatches turned out to be plain value exports, and hard-mask has a
> clean snapshot/restore implementation. Added, every value probed bit-exact vs NumPy 2.4.2: **`tolist`**
> (nested `object[]` with `null` at masked / the fill / a bare scalar for 0-D — walks by logical C-order index so
> any layout works), **`tobytes`** (`filled(fill).tobytes(order)`, byte-identical), **`tofile`** (throws NumPy's
> `NotImplementedError` — a masked array can't be binary-serialized), **`view`** (dtype reinterpret; same-itemsize
> carries the mask, an itemsize change with a live mask is refused), **`ids`**/**`iscontiguous`**/**`putmask`**
> (instance) / **`resize`** (instance — throws NumPy's verbatim "cannot be resized" refusal, pointing at
> `np.ma.resize`), **`flat`** (a raveled masked VIEW — NumSharp's `NDArray.flat` house convention, not NumPy's
> iterator), and the flag surface **`recordmask`**/**`baseclass`**/**`unshare_mask`** (which now genuinely copies
> the mask, since a view/`real`/`imag`/`view` DOES alias it). **`sharedmask` is deliberately NOT offered** — it is
> the one flag with no faithful value (NumPy reports True right after `array(data, mask=…)`, False after an op) and
> a bogus `False` would invite mutating a shared mask, so the guidance is just to call `unshare_mask()`. **And the
> keystone: `harden_mask`/`soften_mask` are now REAL, not no-ops** — a hard mask makes a plain-value assignment
> (indexer / `put`) skip masked slots entirely and never unmask, `= masked` still masks, and `putmask` under hard
> WRITES the data even at a masked slot but FREEZES the mask (NumPy's three distinct branches, all ported and
> probed: `put` drops masked indices, `putmask` does `a.mask |= …`). Implemented by snapshot+restore around the
> soft write (correct for every index kind because the fancy/boolean SET scatters through). Verified by 25 + 19
> signed `dotnet run` checks (0 fail) and gated by **8 more `MaskedArrayTests` (76 total, green net8.0/net10.0)**.
> **THE ONLY REMAINDERS ARE NOW INFRA- OR POLICY-BLOCKED, none a closeable functional gap:** the structured/record
> family (no structured dtypes — `mvoid`/`mr_`/`fromflex`/`flatten_*`/`make_mask_descr`/`torecords`/`toflex`), the
> `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — a C# `fill_value` property reserves those exact
> names; the property + module `set_fill_value` cover the behavior), `sharedmask` (unmodelable), and the
> library-wide `out=`/`subok`/`order` gaps.

---

## 0. Headline coverage

| Surface | NumSharp | NumPy 2.4.2 | Gap |
|---|---|---|---|
| `np.ma.*` module names (`__all__`) | **214** (of 218 public; +`polyfit`) | 226 | **12 names** — NONE a functional gap: 3 are C# types that exist (`MAError`/`MaskError`/`MaskedArray`), 3 are N/A submodule/config refs, 6 need structured dtypes. (`polyfit` CLOSED pass 7 — it composes on the now-present `np.polyfit`.) |
| `MaskedArray` instance members | **70** (+ the indexer) | 94 | **~24 members** — all infra/policy-blocked: structured/record `torecords`/`toflex`, the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 property-name collision), `sharedmask` (unmodelable), and NumPy-internal members. (Pass 9 added `tolist`/`tobytes`/`tofile`/`view`/`ids`/`iscontiguous`/`putmask`/`resize`/`flat`/`recordmask`/`baseclass`/`unshare_mask` + REAL `hardmask`.) |

**Behavioral correctness of what IS implemented:** a 186-case differential (3 dtypes × reductions[×3 axes]/ufuncs/constructors/extras/sort/unique/set-ops) found **0 real value/mask/dtype mismatches**, plus 79 + 33 + 22 + 25 + 19 checks of every 2026-09-14 addition (incl. the pass-7 `polyfit`/masked-axis `median`/`dot.strict`/`take.mode`/`squeeze.axis` and the pass-9 conversion/export + hard-mask surface) against probed NumPy 2.4.2 output (0 failures). The only divergence is the intentional scalar-vs-0d return-type choice (§5.1). Set operations (`intersect1d`/`union1d`/`setxor1d`/`setdiff1d`, added 2026-09-14, commit 566fd303) are bit-exact.

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
| `hsplit` | masked hsplit | ✅ (data+mask split → `MaskedArray[]`). NOTE: `hsplit` is the ONLY split `numpy.ma.__all__` exports (2.4.2, verified) — `vsplit`/`array_split`/`split`/`dsplit` are NOT `np.ma` API, so NumSharp already matches NumPy; adding them would VIOLATE 1-to-1 parity. |
| `frombuffer` | masked frombuffer | ✅ (unmasked wrapper over `np.frombuffer`) |
| `fromfunction` | masked fromfunction | ✅ (`Func<NDArray[],NDArray>` of index grids + a 2-D `Func<NDArray,NDArray,NDArray>` convenience) |
| `ndenumerate` | `(index, value)` skipping masked | ✅ (`IEnumerable<(long[], object)>`, unmasked only) |

### C. Machinery-blocked (deferred — need infra NumSharp lacks)
| Missing | Blocker |
|---|---|
| `mr_`, `mvoid`, `fromflex`, `flatten_mask`, `flatten_structured_array`, `make_mask_descr` | structured/record dtypes — NumSharp has NONE |

**✅ Moved OUT of the return-type-impedance deferral (2026-09-14, pass 8):** N-D-axis `notmasked_edges` and
2-D-axis `notmasked_contiguous` now RETURN `object` — the honest mirror of NumPy's own polymorphic list return.
The flat/1-D/axis=None case still yields the typed `long[]`/`Slice[]` VALUES (boxed), and the axis case yields
`NDArray[][]` (`{mins, maxs}`) / `Slice[][]` (list-of-lists), verified byte-for-byte against NumPy 2.4.2
(`>2-D` raises NumPy's "Currently limited to at most 2D array."). The one flat-form test casts the boxed return;
the breaking change is acceptable per the project's parity-over-ergonomics policy for such niche functions.

**✅ Moved OUT of machinery-blocked (2026-09-14, pass 7):** `polyfit` (the audit premise was STALE — `np.polyfit`
DOES exist in `Polynomial/np.polyfit.cs`; `ma.polyfit` is now the faithful NumPy port that composes on it, so it
computes wherever `np.polyfit` does and throws the identical `OpenBlasMissingBackendException` where the LAPACK
backend is absent — never a silent wrong fit), and masked-axis **`median`** (no bespoke masked-sort core was needed
after all — it composes over the EXISTING `ma.sort` + `count` + `take_along_axis` + `where`, a line-for-line port of
NumPy's `_median` axis branch).

**✅ Moved OUT of machinery-blocked (2026-09-14):** `cov`/`corrcoef` (pass 4 — composable over `dot`/`filled`/
masked-`mean`/`diagonal`/`outer`), `convolve`/`correlate` and `apply_over_axes` (pass 5 — the machinery
`np.convolve`/`np.correlate`/`np.apply_over_axes` already existed, and the mask propagation is a clean composition),
and `apply_along_axis` (pass 6 — the indexer made extracting the masked 1-D slices possible; scalar and 1-D
per-slice results assemble via `stack`/`reshape`/`moveaxis`, no object dtype needed).

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

## 2. MaskedArray INSTANCE-surface gaps (70 present of 94, + the indexer)

**✅ THE INDEXER LANDED (2026-09-14, pass 2).** `object this[params object[]] { get; set; }`: GET returns the bare
scalar for an unmasked scalar index, the `masked` singleton for a masked one, or a sub-`MaskedArray` (a VIEW for a
basic slice — writes through — a COPY for fancy/boolean); SET writes the data and reconciles the mask (a plain
value UNMASKS, `= masked` masks in place leaving the data, a masked-array value PROPAGATES its mask, and a
previously-`nomask` array gains a real mask on the first masking write). Returns `object` because NumPy's indexer
is polymorphic and a C# indexer can't switch its return type on the runtime index — a slice read is cast to
`MaskedArray`. This unblocked `put`/`putmask`/`resize`/`mask_rowcols`/`compress_rowcols`/`notmasked_*`.

**Present (70):** `all any anom argmax argmin argsort astype baseclass clip compress compressed conj conjugate count cumprod cumsum data diagonal dot dtype fill_value filled flat flatten harden_mask hardmask ids imag iscontiguous item mask max mean min mT ndim nonzero prod ptp put putmask ravel real recordmask repeat reshape resize round shape shrink_mask size soften_mask sort squeeze std sum swapaxes T take tobytes tofile tolist trace transpose typecode unshare_mask var view` (+ the `this[...]` indexer + operators `+ - * / % & | ^ ~ < > <= >=`).

**✅ Pass 9 (2026-09-14) — conversion/export + flag surface + REAL hard mask:**
- **`tolist(fill_value=null)`** → nested `object[]` (`null` at masked / the fill / a bare scalar for 0-D); walks by logical C-order index so any layout is handled without a copy.
- **`tobytes(fill_value=null, order='C')`** = `filled(fill).tobytes(order)`, byte-identical to NumPy (the `order='A'` corner resolves via the C-contiguous filled copy — a documented masked nuance).
- **`tofile`** throws NumPy's `NotImplementedError` (a masked array can't be binary-serialized).
- **`view(dtype=null)`** reinterprets the data dtype — a same-itemsize/no-dtype view keeps the shape and carries the mask (aliased); an itemsize-changing view of a MASKED array is refused (the bool mask can't be reinterpreted).
- **`ids()`/`iscontiguous()`/`putmask(mask,values)`** instance forms; **`resize(new_shape)`** throws NumPy's verbatim "does not own its data … cannot be resized" refusal (use `np.ma.resize`).
- **`flat`** returns a raveled masked VIEW — NumSharp's `NDArray.flat` house convention (a view for contiguous, a copy otherwise), NOT NumPy's `flat` iterator object.
- **`hardmask`/`recordmask`/`baseclass`** flag properties + **`unshare_mask()`** (which genuinely COPIES the mask, since `view`/`real`/`imag` alias it). **`sharedmask` deliberately omitted** — no faithful value (NumPy: True after `array(data,mask=…)`, False after an op) and a bogus `False` would invite mutating a shared mask; call `unshare_mask()` instead.
- **REAL hard mask (`harden_mask`/`soften_mask` no longer no-ops):** a hard mask makes a plain-value assignment (indexer / `put`) skip masked slots and never unmask; `= masked` still masks; `putmask` under hard writes data everywhere but freezes the mask (`a.mask |= …`). All three NumPy branches ported (snapshot+restore for the indexer/`put`, the three-way mask branch for `putmask`), probed bit-exact.

**✅ Earlier wired-through (2026-09-14):** `T mT flatten ravel reshape transpose swapaxes squeeze repeat take compress compressed sort argsort clip round conj conjugate dot trace diagonal nonzero real imag item fill_value(get/set) harden_mask soften_mask shrink_mask`. NumPy's METHOD forms `get_fill_value()`/`set_fill_value()` are deliberately NOT offered — a C# property named `fill_value` reserves those exact accessor names (CS0082), so the property + the module-level `set_fill_value(a, v)` cover them.

**Still missing instance members (all infra/policy-blocked):** `torecords`/`toflex` and the `mvoid` record machinery (structured/record dtypes — NumSharp has none); the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — covered by the `fill_value` property + module `set_fill_value`); `sharedmask` (unmodelable, see above); and the NumPy-internal/policy members (`out=` on reductions, `_optinfo`, etc.).

---

## 3. Parameter-parity gaps on IMPLEMENTED functions

| Function | Status |
|---|---|
| `average` | ✅ `keepdims` added; `returned` covered by the sibling `average_returned` → `(avg, sumOfWeights)` tuple (the base-library convention, since C# can't switch return type on a flag) |
| `median` | ✅ `keepdims` added (flat + unmasked-axis paths); ✅ **masked-axis DONE (pass 7)** — the per-slice masked median (`_median` axis-branch port). `out`/`overwrite_input` still absent (the library-wide `out=` policy gap, §5.5) |
| `sum`/`prod`/`mean`/… | `out=` still absent (the library-wide policy gap, §5.5 — NumSharp models no `out=` on any masked reduction) |
| `array`/`masked_array` | ✅ **`dtype`** added; ✅ **`hard_mask`** added (pass 9 — bakes the hard mask in at construction, now that hard mask is real); `keep_mask`/`shrink`/`ndmin`/`order`/`subok` still absent (`ndmin`/`order`/`subok` have no NumSharp analog) |

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
| `std`, `var` | **`mean`** (NumPy 2.0) | ✅ added (pass 8) — `var` centers by the supplied mean; `std` ACCEPTS but IGNORES it (NumPy's own quirk: `ma.std` forwards only `keepdims` to `var`, so `ma.std(mean=X)` is the PLAIN std — reproduced) |
| `take` | **`mode`** | ✅ `clip`/`wrap`/`raise` added (pass 7) — applied to the data AND mask gather alike |
| `dot` | **`strict`** | ✅ added (pass 7) — masks propagated along the contracted axes (NumPy's `_mask_propagate`) before the product |
| `squeeze` | **`axis`** | ✅ added (pass 7) — drop one named size-1 axis (mask squeezed alike) |
| `power` | **`third`** | ✅ added (pass 8) — the modulus slot exists for signature parity and rejects a non-null value with NumPy's `MaskError "3-argument power not supported."` |
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
2. ✅ **DONE** — Parameter parity: `average.keepdims`+`average_returned`, `median.keepdims`, `sort`/`argsort` `endwith`, `array.dtype`, `isin.assume_unique`, `vander` `N`→`n`, `reshape` `new_shape` (§5.1), (pass 7) `take.mode`, `dot.strict`, `squeeze.axis`, and (pass 8) `std`/`var`.`mean` + `power.third`. *(The only param NOT added is `sort`/`argsort`'s `kind`/`stable`/`order` — NumSharp's sort is always stable radix, so they're inexpressible/no-op.)*
3. ✅ **DONE** — Operators `%` `&` `|` `^` `~` on MaskedArray (§5.2). `==`/`!=` kept as documented reference-equality (use `np.ma.equal`/`not_equal`).
4. ✅ **DONE** — Complex fill values → complex (§5.3).
5. ✅ **DONE** — Instance surface: `fill_value` (get/set) + `T`/`mT`/`real`/`imag`/`ravel`/`flatten`/`reshape`/`transpose`/`swapaxes`/`squeeze`/`repeat`/`take`/`compressed`/`compress`/`sort`/`argsort`/`clip`/`round`/`conj`/`dot`/`diagonal`/`trace`/`nonzero`/`item` (§2). *(The rest — `tolist`/`tobytes`/`tofile`/`view`/`ids`/`iscontiguous`/`putmask`/`resize`/`flat`/`recordmask`/`baseclass`/`unshare_mask` — DONE pass 9.)*
6. ✅ **DONE (pass 2)** — the `MaskedArray this[...]` indexer get/set (§2), plus the mutation trio `put`/`putmask`/
   `resize`, `mask_rowcols`, `compress_rowcols`/`compress_nd`, `notmasked_edges`/`notmasked_contiguous` (axis=None/1-D),
   and `ids`.
7. ✅ **DONE** — Real functional gaps §1.B: `fix_invalid`, `diff`, `append`, `clip`, `choose`, `compress`, `diagonal`, `nonzero`, `trace`, `make_mask`/`make_mask_none`, `mask_or`, `common_fill_value`, `set_fill_value`, `left_shift`/`right_shift`, `put`/`putmask`/`resize`, `ids`, `frombuffer`, `fromfunction`, `hsplit`, `ndenumerate`, plus the `MAError`/`MaskError` types. *(The `vsplit`/`array_split`/`split`/`dsplit` "split family" was a non-gap — `numpy.ma` exports only `hsplit`; corrected pass 7.)*
8. ✅ **DONE (pass 7)** — `polyfit` (the `np.polyfit`-doesn't-exist premise was stale — it does; `ma.polyfit` composes on it), masked-axis `median` (a composition over `ma.sort`/`count`/`take_along_axis`, no bespoke masked-sort core needed), `dot.strict`, `take.mode`, `squeeze.axis`.
9. ✅ **DONE (pass 8)** — N-D-axis `notmasked_edges` + 2-D-axis `notmasked_contiguous` (return `object`, the honest mirror of NumPy's polymorphic list; flat form still typed-but-boxed), `power.third` (rejects with `MaskError`), and `std`/`var`.`mean` (var uses it; std reproduces NumPy's ignore-it quirk).
10. ✅ **DONE (pass 9)** — the instance conversion/export surface (`tolist`/`tobytes`/`tofile`/`view`/`ids`/`iscontiguous`/`putmask`/`resize`/`flat`/`recordmask`/`baseclass`/`unshare_mask`) + the `array`/`masked_array` `hard_mask` param, AND **REAL hard-mask semantics** — `harden_mask`/`soften_mask` are no longer no-ops: a hard mask blocks unmasking on a plain-value assignment (indexer/`put`), `= masked` still masks, and `putmask` under hard writes data but freezes the mask (all three NumPy branches ported, probed bit-exact). The `tolist`/`tobytes`/`view`/`flat` "semantic mismatch" premise was mostly stale — only `flat`'s NumPy-iterator-vs-NumSharp-raveled-view difference is real, and `flat` follows the house convention.
11. ⛔ **Genuinely remaining — all infra-blocked or by-policy, NONE a closeable functional gap:** the structured/record family (`mvoid`/`mr_`/`fromflex`/`flatten_mask`/`flatten_structured_array`/`make_mask_descr`/`torecords`/`toflex` — no structured dtypes in NumSharp); `sharedmask` (no faithful value — unmodelable; use `unshare_mask()` instead); the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — the `fill_value` property + module `set_fill_value` cover the behavior); and the library-wide `out=`/`subok`/`order` policy gaps (§5.5). **Every other backlog item is DONE — the module is functionally complete for NumSharp.Core bar structured dtypes.**

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

10. **2026-09-14 pass 7** — before writing anything, probed `numpy.ma.__all__` (only `hsplit` in the split family) and confirmed `np.polyfit` exists in Core, invalidating two stale premises. Every pass-7 addition (`polyfit`, masked-axis `median` incl. NaN-propagation/keepdims/neg-axis/int→float64/1-D, `dot.strict`, `take.mode`, `squeeze.axis`) was captured as a NumPy 2.4.2 reference block and bit-compared by a 22-assertion signed `dotnet run` script — **22/22 pass** — then gated by 5 new `MaskedArrayTests` (64 total, green net8.0 + net10.0). `polyfit`'s live gate enables the OpenBLAS backend per-test (Inconclusive if no LAPACK-capable BLAS), mirroring `LapackFactorisationTests.RequireLapack`. This session's parallel-WIP files (`NDExpr.Combinators.cs` + its test) compiled cleanly, so the in-place build was safe — the worktree quarantine from technique 9 was not needed this pass, but stays the fallback if a sibling's WIP ever fails to compile.

11. **2026-09-14 pass 8** — closed the return-type-impedance and last-param findings. Key discovery from `inspect.getsource(numpy.ma.MaskedArray.std)`: **`ma.std` accepts `mean` but never forwards it to `var`** (its `kwargs` carries only `keepdims`), so `ma.std(mean=X)` silently returns the PLAIN std — a NumPy quirk that a naive `sqrt(var(…, mean))` would have gotten WRONG (my first cut did; the probe `sqrt(var(mean=2))=1.414` ≠ `std(mean=2)=1.247` caught it). Reproduced by making `std` ignore `mean`. The N-D `notmasked_*` return `object` (NumPy's own return there is a dynamically-typed Python list, so `object` is the faithful mirror, not a compromise); the flat test casts the boxed `long[]`/`Slice[]`. Verified by two signed `dotnet run` scripts (7 N-D/power checks + 8 var-mean checks, all pass) against NumPy 2.4.2 reference blocks, gated by 4 new `MaskedArrayTests` (68 total, green both TFMs). Probed the indexer forms (`ma[i, Slice.All]` / `ma[Slice.All, i]`), `np.indices`, and masked `min(axis)`+`compressed` up front to confirm the composition before writing it.

12. **2026-09-14 pass 9** — closed the instance conversion/export surface and REAL hard-mask semantics. Two of the "genuinely remaining" findings were re-examined and fell: `tolist`/`tobytes`/`view` are plain value exports (only `flat`'s iterator-vs-raveled-view difference is a true semantic mismatch, resolved by the house convention), and `harden_mask`/`soften_mask` being no-ops is a fixable *bug*, not a missing analog. The hard-mask semantics were read out of `refs/numpy/numpy/ma/core.py` — the two probes that shaped the impl: **`put` and `putmask` under hard are DIFFERENT** (`put` at lines 4896–4903 drops the masked indices before writing, so masked data is untouched; the module `putmask` at 7579–7588 does `np.copyto(a._data, valdata, where=mask)` UNCONDITIONALLY and only `a.mask |= m` — so it WRITES data at masked slots but never unmasks). My first putmask cut copied `put`'s "restore masked data" and was wrong; the probe `hard_putmask → [50,60,3,4]` (data 60 at the masked slot) caught it. Snapshot+restore for the indexer/`put` is correct for every index kind because the fancy/boolean SET scatters through the array (unlike the COPY the getter returns). Verified by 25 (export/flags) + 19 (hard-mask) signed `dotnet run` checks against probed NumPy 2.4.2 (0 fail), gated by 8 new `MaskedArrayTests` (**76 total**, green net8.0 + net10.0). No worktree quarantine needed — the only untracked `.cs` in the tree (`ConcurrentOrderedDictRandomizedConcurrencyTests.cs`) compiled clean.

