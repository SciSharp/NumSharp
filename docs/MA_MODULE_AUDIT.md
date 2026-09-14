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
>
> **Progress — 2026-09-14 (pass 10 — three "structured/record" names were mis-categorized; CLOSED).** The
> blanket "structured/record family — blocked" (§1.C) was too coarse: it swept up three `__all__` names that are
> NOT structured-dependent, only mis-filed. All three now landed, every value probed bit-exact vs NumPy 2.4.2:
> **`flatten_mask`** (ravels any mask to a fresh 1-D bool array — NumPy's structured recursion NEVER branches
> without structured dtypes, so this is exactly its non-structured path: "ravel in C-order, coerce `!= 0`";
> 0-D → length 1, empty → length 0); **`make_mask_descr`** (returns the boolean dtype for EVERY data dtype —
> the recursion NumPy runs over a structured dtype has nothing to descend into here, so every dtype maps to the
> single scalar `bool` descriptor NumPy returns for a non-structured dtype); and **`mr_`** — the masked
> counterpart of `np.r_` (NumPy's `MAxisConcatenator`), which is structured ONLY in its de-activated
> matrix-string path — its real job is masked concatenation. `mr_` composes over the EXISTING `np.r_`: the result
> DATA is `np.r_[key]` over the entries (a masked entry contributes its `data`), the result MASK is `np.r_[maskKey]`
> where each entry is replaced by its boolean-mask contribution of the same natural shape (masked → its full mask,
> plain/scalar → all-False via `getmaskarray`, colon-slice → all-False of the arange's shape, directive string →
> passes through), so every directive (axis / ndmin / `r`/`c` matrix) lays the mask out to match the data; NO mask
> anywhere → `nomask` fast path; a LONE string → `MAError "Unavailable for masked array."` (NumPy's rejection,
> and `MAError`'s SECOND raiser after `power`'s modulus). It follows NumSharp's library-wide colon-string =
> slice convention (raw NumPy's `mr_["0:3"]` would be a failed directive since Python slices come from `:` syntax),
> with ONE documented divergence: a colon-slice combined with a `trans1d != -1` directive lays data via `np.r_`'s
> slice branch but the mask via its array branch (their axis permutations differ only in that rare case).
> Verified by 26 + 5 signed `dotnet run` checks (0 fail) — data/mask/shape/dtype/nomask-fast-path/lone-string,
> plus the lone-masked-array and `"r"` matrix-directive edges — and gated by **3 more `MaskedArrayTests` (79
> total, green net8.0/net10.0)**. **THE ONLY REMAINDERS ARE NOW GENUINELY BLOCKED OR BY-POLICY:** the true
> structured/record family (`mvoid`/`fromflex`/`flatten_structured_array`/`torecords`/`toflex`/`mr_class`-matrix
> — no structured dtypes); `MAError`/`MaskError`/`MaskedArray` (C# types that already exist, caught as
> `catch (MAError)`); `core`/`extras`/`masked_print_option` (submodule/print-style refs — `masked_print_option`
> configures NumPy's aligned inline `[1 -- 3]` repr, a printing style NumSharp deliberately does not emulate);
> `sharedmask` (unmodelable — use `unshare_mask()`); the `get_fill_value`/`set_fill_value` METHOD spellings
> (CS0082); and the library-wide `out=`/`subok`/`order` gaps.
>
> **Progress — 2026-09-14 (pass 11 — the "printing style NumSharp does not emulate" premise was STALE; masked
> printing + `masked_print_option` CLOSED, plus the last tractable instance members).** The pass-10 note called
> `masked_print_option` "a printing style NumSharp deliberately does not emulate, so the option would have nothing
> to affect" — the exact class of rationalization every prior pass has overturned, and it fell here too. Masked
> `str`/`repr` are now a **BYTE-EXACT port of NumPy 2.4.2's `MaskedArray.__str__`/`__repr__`**: when a mask is
> present and the print option is enabled, NumPy renders the data as an OBJECT array (each element its Python-scalar
> repr, NO numeric column alignment) with the display token (`--`) substituted at masked slots — which is why a
> masked print is UNALIGNED (`[1 -- 3 --]`) versus the padded numeric print of the same data. This reused NumSharp's
> existing byte-exact array-printing engine: the recursive layout in `ArrayFormatter.Recurse` gained an optional
> lockstep `mask`/`display` hook (numeric path provably unchanged — `mask==null` is the old code), and the object
> element strings come from the pre-existing `ScalarStr`/`PythonFloatRepr` (the object-array formatter). The
> `repr()` template — one-row-vs-2-D layout, the `dtype=` line (non-implied/all-masked/empty), and the
> `fill_value=` line including NumPy 2.x's `np.<type>(value)` wrapper for a promoted fill dtype — all match. And
> **`masked_print_option` is now a live global** (`display`/`set_display`/`enabled`/`enable`): `set_display("N/A")`
> changes the token and `enable(false)` switches masked slots to the fill-value (aligned) rendering — both
> observable through `ToString()`. `ToString()` now returns the `str` form and `ToString(true)` the `repr` form,
> aligning with `NDArray`'s convention (a deliberate, accepted change from the old readable-approximation repr).
> Also closed the last tractable INSTANCE members (NumPy inherits these from ndarray, so they see through the mask,
> and none was structured-blocked — only unenumerated): **`fill`** (overwrites data, mask untouched), **`searchsorted`**
> (on the data, returns a plain array), **`byteswap`** (swaps data, carries the mask), **`partition`**/**`argpartition`**
> (NumPy's own mask-oblivious footgun — reorders only the data, leaving the mask positional; NumPy warns, NumSharp
> has no warning channel), and the data-describing **`itemsize`**/**`nbytes`**/**`strides`** (mirror the data buffer,
> mask excluded — matching NumPy's `nbytes`). All verified byte-exact against probed NumPy 2.4.2 (40 printing +
> option + instance-member checks, then 3 more property checks, 0 fail) and gated by **6 more `MaskedArrayTests`
> (85 total, green net8.0/net10.0)**; the shared numeric printing path is unregressed (241 printing unit tests +
> the 2094-case `dtype_text` oracle differential tier, both green). **THE ONLY REMAINDERS ARE NOW GENUINELY
> BLOCKED OR BY-POLICY** (none a closeable masked-array-behavior gap): the true structured/record family
> (`mvoid`/`fromflex`/`flatten_structured_array`/`getfield`/`setfield`/`torecords`/`toflex` — no structured
> dtypes); `sharedmask` (no faithful value — unmodelable; use `unshare_mask()`); the
> `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — the `fill_value` property + module `set_fill_value`
> cover the behavior); the ndarray infra mirrors with no masked semantics (`base`/`ctypes`/`flags`/`setflags`),
> pickle (`dump`/`dumps`) and array-API device (`device`/`to_device`), all N/A in NumSharp; `core`/`extras`
> (submodule refs); and the library-wide `out=`/`subok`/`order` policy gaps.
>
> **Progress — 2026-09-14 (pass 12 — a fresh multi-technique re-audit found 3 surface gaps + 1 real behavioral
> divergence; ALL CLOSED).** Rather than trust "functionally complete", re-ran the audit with independent
> techniques. **Technique 1 — reflection surface diff** (enumerate NumPy's `__all__`/members, reflect NumSharp's
> `MaskedArrayModule`/`MaskedArray`, set-difference): found THREE genuine INSTANCE-method gaps that existed only as
> MODULE functions, not instance methods — **`copy()`** (deep-copies data+mask), **`choose(choices, mode)`** (select
> by this array's masked indices), **`product()`** (alias of `prod`) — plus the data-describing `itemsize`/`nbytes`/
> `strides` from pass 11. (`set_fill_value` flagged by the diff was a false positive — the `set_` accessor-name
> filter hid the real module function.) **Technique 2 — a 527-case behavioral differential fuzz** (4 dtypes × 5
> memory layouts {C,F,transposed,reversed,strided} × 28 ops + edge masks, layout-independent per-element
> bit-token comparison vs NumPy 2.4.2): found **ONE genuine divergence uncaught by every prior pass** — a masked
> binary op with a BARE SCALAR operand did not widen like NumPy. numpy.ma wraps a scalar as
> `np.asarray(scalar)` — a STRONG default-width array (python int→int64, float→float64) — so `ma.multiply(int32, 2)`
> is **int64**, whereas NumSharp's library-wide weak-scalar model kept it **int32** (the prior audit's "multiply
> bit-exact all dtypes" only ever tested array×array). The rule is exact and uniform:
> `result_type(array_dtype, strong_default(scalar))`, verified across every dtype × op. **CLOSED** by
> `PromoteMaScalar` (widens the INPUT data arrays — required so a value overflowing the narrow dtype is computed
> wide, e.g. `ma.multiply(int32 2e9, 2) == 4_000_000_000`), wired into the two shared cores `Binary`/`DomainedBinary`
> (activates ONLY when a bare scalar is present — array×array is provably untouched), `power`, and `where` (which
> transitively fixes `maximum`/`minimum`). Comparisons/logical ops in `Binary` are unaffected (bool output). After
> the fix the fuzz is **527/527 bit-exact**. Verified + gated by **2 more `MaskedArrayTests` (87 total, green
> net8.0/net10.0)** and the full 527-case differential; no regression (all changes confined to `Ma/MaskedArray.cs`,
> which only `np.ma` consumes). One SMALL documented remainder stands: the scalar operand's width follows the C#
> literal's kind (int→int64, float→float64) — a bare `(byte)`/`(short)` scalar (no python-literal analog) maps to
> int64 like a python int, and the C#-`int`-is-int32 vs python-`int`-is-int64 language gap is unchanged for array
> literals (documented library-wide). **The module remains functionally complete for NumSharp.Core; the genuinely
> remaining names are still only the structured/record family, the CS0082 method spellings, `sharedmask`, the N/A
> ndarray-infra/pickle/device mirrors, `core`/`extras`, and the library-wide `out=`/`subok`/`order` gaps.**

---

## 0. Headline coverage

| Surface | NumSharp | NumPy 2.4.2 | Gap |
|---|---|---|---|
| `np.ma.*` module names (`__all__`) | **218** (of 219 public; +`polyfit`, +`masked_print_option`) | 226 | **8 names** — NONE a functional gap: 3 are C# types that exist (`MAError`/`MaskError`/`MaskedArray`), 2 are N/A submodule refs (`core`/`extras`), 3 need structured dtypes (`mvoid`/`fromflex`/`flatten_structured_array`). (`polyfit` CLOSED pass 7; `flatten_mask`/`make_mask_descr`/`mr_` CLOSED pass 10; **`masked_print_option` CLOSED pass 11** — now a real, live option object driving byte-exact masked `str`/`repr`.) |
| `MaskedArray` instance members | **81** (+ the indexer) | 94 | **~13 members** — all infra/policy-blocked: structured/record `getfield`/`setfield`/`torecords`/`toflex`, the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 property-name collision), `sharedmask` (unmodelable), the ndarray infra mirrors with no masked semantics (`base`/`ctypes`/`flags`/`setflags`), pickle (`dump`/`dumps`) and array-API device (`device`/`to_device`) — all N/A. (Pass 9 added `tolist`/`tobytes`/`tofile`/`view`/`ids`/`iscontiguous`/`putmask`/`resize`/`flat`/`recordmask`/`baseclass`/`unshare_mask` + REAL `hardmask`; pass 11 added `fill`/`searchsorted`/`byteswap`/`partition`/`argpartition`/`itemsize`/`nbytes`/`strides`; **pass 12 added `copy`/`choose`/`product`**.) |

**Behavioral correctness of what IS implemented:** a 186-case differential (3 dtypes × reductions[×3 axes]/ufuncs/constructors/extras/sort/unique/set-ops) found **0 real value/mask/dtype mismatches**, plus 79 + 33 + 22 + 25 + 19 + (26 + 5) + (40 + 3) checks of every 2026-09-14 addition (incl. the pass-7 `polyfit`/masked-axis `median`/`dot.strict`/`take.mode`/`squeeze.axis`, the pass-9 conversion/export + hard-mask surface, the pass-10 `flatten_mask`/`make_mask_descr`/`mr_`, and the **pass-11 byte-exact masked `str`/`repr` + `masked_print_option` + `fill`/`searchsorted`/`byteswap`/`partition`/`argpartition`/`itemsize`/`nbytes`/`strides`**) against probed NumPy 2.4.2 output (0 failures). The only divergence is the intentional scalar-vs-0d return-type choice (§5.1). Set operations (`intersect1d`/`union1d`/`setxor1d`/`setdiff1d`, added 2026-09-14, commit 566fd303) are bit-exact. The shared numeric array-printing path is unregressed by the masked-print hook (241 printing unit tests + the 2094-case `dtype_text` oracle differential tier, both green). **A pass-12 fresh 527-case behavioral differential (4 dtypes × 5 memory layouts × 28 ops + edges, layout-independent per-element bit comparison vs NumPy 2.4.2) is 527/527 bit-exact** after the scalar-promotion fix (§5.8). Gate: **87 `MaskedArrayTests`, green net8.0/net10.0.**

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
| `mvoid`, `fromflex`, `flatten_structured_array` (+ instance `torecords`/`toflex`) | structured/record dtypes — NumSharp has NONE |

**✅ Moved OUT of machinery-blocked (2026-09-14, pass 10 — three names were MIS-FILED here):** `flatten_mask`,
`make_mask_descr`, and `mr_` are NOT structured-dependent. `flatten_mask` is exactly NumPy's non-structured path
(ravel C-order → coerce `!= 0` → bool), `make_mask_descr` returns the scalar `bool` descriptor NumPy gives every
non-structured dtype, and `mr_` (NumPy's `MAxisConcatenator`) is structured only in its de-activated matrix-string
path — its real job, masked concatenation, composes over the existing `np.r_` (data via `np.r_[key]`, mask via
`np.r_[maskKey]` with each entry's boolean-mask contribution). All three land bit-exact vs NumPy 2.4.2, gated by 3
new `MaskedArrayTests`. See the pass-10 headline note.

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
`MaskedArray` (class ref — exists as a type). ✅ **`MAError`/`MaskError` EXIST** as C# exception types
(`Exceptions/MAError.cs`, `MaskError : MAError : Exception` — NumPy's hierarchy). Both are now RAISED, not just
present for type parity: `MaskError` by `power`'s modulus arg (pass 8) and `MAError` by `mr_`'s lone-string
rejection (pass 10). Ported code catches them idiomatically as `catch (MAError)` — exposing `np.ma.MAError` as a
`Type` property would be non-idiomatic and unusable in a `catch`, so the namespace types ARE the parity. ✅ `bool_`
(dtype alias) added. ✅ **`masked_print_option` DONE (pass 11)** — the stale premise (NumSharp "deliberately does
not emulate" NumPy's inline `[1 -- 3]` masked repr) is overturned: `MaskedArray.ToString()`/`ToString(true)` are
now a byte-exact port of NumPy's `__str__`/`__repr__`, and `np.ma.masked_print_option` is a live
`MaskedPrintOption` object (`display`/`set_display`/`enabled`/`enable`) that drives them — changing the token or
disabling the substitution is observable through `ToString()`, exactly as in NumPy. Still absent: `core`/`extras`
(submodule refs — N/A in C#).

---

## 2. MaskedArray INSTANCE-surface gaps (78 present of 94, + the indexer)

**✅ THE INDEXER LANDED (2026-09-14, pass 2).** `object this[params object[]] { get; set; }`: GET returns the bare
scalar for an unmasked scalar index, the `masked` singleton for a masked one, or a sub-`MaskedArray` (a VIEW for a
basic slice — writes through — a COPY for fancy/boolean); SET writes the data and reconciles the mask (a plain
value UNMASKS, `= masked` masks in place leaving the data, a masked-array value PROPAGATES its mask, and a
previously-`nomask` array gains a real mask on the first masking write). Returns `object` because NumPy's indexer
is polymorphic and a C# indexer can't switch its return type on the runtime index — a slice read is cast to
`MaskedArray`. This unblocked `put`/`putmask`/`resize`/`mask_rowcols`/`compress_rowcols`/`notmasked_*`.

**Present (78):** `all any anom argmax argmin argpartition argsort astype baseclass byteswap clip compress compressed conj conjugate count cumprod cumsum data diagonal dot dtype fill fill_value filled flat flatten harden_mask hardmask ids imag iscontiguous item itemsize mask max mean min mT nbytes ndim nonzero partition prod ptp put putmask ravel real recordmask repeat reshape resize round searchsorted shape shrink_mask size soften_mask sort squeeze std strides sum swapaxes T take tobytes tofile tolist trace transpose typecode unshare_mask var view` (+ the `this[...]` indexer + operators `+ - * / % & | ^ ~ < > <= >=`).

**✅ Pass 11 (2026-09-14) — masked printing + `masked_print_option` + the last tractable instance members:**
- **`ToString()` / `ToString(bool)`** — a BYTE-EXACT port of NumPy's `MaskedArray.__str__`/`__repr__`. `ToString()` is the `str` form (`[1 -- 3 --]`), `ToString(true)` the `repr` form (`masked_array(data=…, mask=…, fill_value=…[, dtype=…])`) — a deliberate, accepted change from the old readable-approximation repr, aligning with `NDArray.ToString(bool)`. When a mask is present and `masked_print_option` is enabled, the data prints as an OBJECT array (Python-scalar repr per element, NO numeric alignment) with `--` at masked slots; a `nomask` array (or a disabled option, which fills the slots) prints numeric-aligned. Implemented by threading an optional lockstep `mask`/`display` hook through the existing `ArrayFormatter.Recurse` (numeric path provably unchanged) + reusing `ScalarStr`; the `fill_value=` line reproduces NumPy 2.x's `np.<type>(value)` wrapper for a promoted fill dtype.
- **`fill(value)`** — overwrites the DATA in place, mask untouched (NumPy's `fill` is a data op, not an unmask).
- **`searchsorted(v, side, sorter)`** — on the DATA (ignores the mask), returns a PLAIN `NDArray`.
- **`byteswap(inplace=false)`** — swaps the DATA bytes, carries the mask into a fresh masked array (or in place).
- **`partition`/`argpartition`** — NumPy's own mask-oblivious FOOTGUN (it warns): reorders only the data, leaving the mask in its ORIGINAL positions. Reproduced (NumSharp has no warning channel), clearly documented; prefer `sort`/`argsort`, which move the mask with the data.
- **`itemsize`/`nbytes`/`strides`** — describe the DATA buffer, EXCLUDING the mask (matching NumPy's `nbytes`; strides in bytes).

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

**Still missing instance members (all infra/policy-blocked, none a masked-array-behavior gap):** `getfield`/`setfield`/`torecords`/`toflex` and the `mvoid` record machinery (structured/record dtypes — NumSharp has none); the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — covered by the `fill_value` property + module `set_fill_value`); `sharedmask` (unmodelable, see above); the ndarray infra mirrors that carry no masked semantics (`base`/`ctypes`/`flags`/`setflags`), pickle (`dump`/`dumps`) and array-API device (`device`/`to_device`), all N/A in NumSharp; and the NumPy-internal/policy members (`out=` on reductions, `_optinfo`, etc.). `get_real`/`get_imag` are covered by the `real`/`imag` properties.

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

### 5.7 Masked printing — ✅ BYTE-EXACT (pass 11)
`MaskedArray.ToString()` (NumPy `str`) and `ToString(true)` (NumPy `repr`) are a byte-exact port of NumPy 2.4.2's
`MaskedArray.__str__`/`__repr__`, verified across int/float/complex/bool × 0-D/1-D/2-D × masked/nomask/all-masked/
all-False/empty/summarized, plus every repr subtlety (one-row-vs-2-D indent layout, the `dtype=` line for
non-implied/all-masked/empty dtypes, and the `fill_value=` line including NumPy 2.x's `np.<type>(value)` wrapper for
a promoted fill dtype). `masked_print_option` (`display`/`set_display`/`enabled`/`enable`) is a live global that
drives them. Two behaviors are load-bearing and easy to get wrong:
- **Masked prints are UNALIGNED.** With a mask present + option enabled, NumPy converts the data to an OBJECT array
  (each element its Python-scalar repr, no shared column width) and substitutes `--` at masked slots — so
  `[1 -- 3 --]`, NOT the padded numeric form. A `nomask` array, or a DISABLED print option (which fills the masked
  slots with the fill value), prints the numeric array with normal alignment. The branch is on `_mask is null` /
  `masked_print_option.enabled()`, so NumSharp's `_mask` null-vs-array state must match NumPy's nomask-vs-array state
  (it does — construction keeps the mask as given, no shrink).
- **The `fill_value` display dtype ≠ the fill VALUE.** For a DEFAULT fill the effective dtype follows NumPy's
  `_check_fill_value` promotion (signed→int64, unsigned→uint64, float→float64), which differs from the data dtype
  only for a non-implied data dtype — exactly the arrays that also show `dtype=` — and there NumPy wraps the value as
  `np.<promoted>(value)`. A caller-SET custom fill of a mismatched dtype is the one edge rendered against the data
  dtype (may differ in that single line); the default-fill case (the realistic one) is byte-exact for every dtype.

### 5.8 Masked binary ops with a scalar operand PROMOTE like NumPy — ✅ FIXED (pass 12)
Found by a fresh 527-case differential fuzz (uncaught by every prior pass, which only tested array×array binary
ops). numpy.ma does NOT treat a bare scalar operand as a weak NEP50 scalar — it wraps it as `np.asarray(scalar)`,
a STRONG default-width array (python int→int64, float→float64, bool→bool, complex→complex128) — so
`ma.multiply(int32, 2)` is **int64**, `ma.floor_divide(int8, 2)` is int64, `ma.bitwise_and(int32, 2)` is int64,
whereas the plain-np `np.multiply(int32, 2)` stays int32 (NumPy's ma diverges from its OWN plain path here).
NumSharp's library-wide weak-scalar / 0-d model previously kept the array dtype. The rule is exact and uniform:
`result_type(array_dtype, strong_default(scalar))`, verified across every dtype × op. `PromoteMaScalar`
(`Ma/MaskedArray.cs`) widens the INPUT data arrays before the op — casting the inputs, not the result, is required
so a value that overflows the narrow dtype is computed wide (`ma.multiply(int32 2e9, 2) == 4_000_000_000`, not the
int32-wrapped value). Wired into `Binary` + `DomainedBinary` (add/subtract/multiply/arctan2/hypot/bitwise/divide/
floor_divide/remainder/mod/fmod; activates ONLY when a bare scalar is present, so array×array is untouched),
`power`, and `where` (which transitively fixes `maximum`/`minimum`). Comparisons/logical ops in `Binary` are
unaffected (bool output). **Two small documented residues:** the scalar's width follows the C# literal's kind, so a
bare `(byte)`/`(short)` scalar (no python-literal analog) maps to int64 like a python int; and the
C#-`int`-is-int32 vs python-`int`-is-int64 language gap remains for array literals (`np.array(new[]{1,2,3})` is
int32) — both are the documented library-wide int-width convention, not ma-specific.

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
11. ✅ **DONE (pass 10)** — `flatten_mask`, `make_mask_descr`, and `mr_` — three names MIS-FILED under the
    structured/record family that are NOT actually structured-dependent (§1.C). `flatten_mask` = NumPy's
    non-structured path (ravel C-order → `!= 0` → bool), `make_mask_descr` = the scalar `bool` descriptor NumPy
    returns for every non-structured dtype, `mr_` = the masked `np.r_` (composed over the existing `np.r_` for
    both the data and mask streams, lone-string → `MAError`). Bit-exact vs NumPy 2.4.2, gated by 3 new
    `MaskedArrayTests` (79 total).
12. ✅ **DONE (pass 11)** — byte-exact masked `str`/`repr` (`ToString()`/`ToString(true)`, §5.7) + **`masked_print_option`**
    (the stale "printing style NumSharp does not emulate" premise overturned), plus the last tractable instance
    members `fill`/`searchsorted`/`byteswap`/`partition`/`argpartition` and the data-describing
    `itemsize`/`nbytes`/`strides`. Gated by 6 new `MaskedArrayTests` (85 total); the shared numeric printing path is
    unregressed (241 printing unit tests + the 2094-case `dtype_text` oracle tier).
13. ✅ **DONE (pass 12)** — a fresh multi-technique re-audit: reflection surface diff → instance `copy`/`choose`/`product`
    (existed only as module functions); a 527-case behavioral differential fuzz → the masked binary-op scalar-strong
    promotion divergence (§5.8), fixed in `Binary`/`DomainedBinary`/`power`/`where`. Fuzz 527/527 bit-exact; gated by 2
    new `MaskedArrayTests` (87 total).
14. ⛔ **Genuinely remaining — all infra-blocked or by-policy, NONE a closeable masked-array-behavior gap:** the TRUE
    structured/record family (`mvoid`/`fromflex`/`flatten_structured_array` + instance
    `getfield`/`setfield`/`torecords`/`toflex` — no structured dtypes in NumSharp); `sharedmask` (no faithful value —
    unmodelable; use `unshare_mask()` instead); the `get_fill_value`/`set_fill_value` METHOD spellings (CS0082 — the
    `fill_value` property + module `set_fill_value` cover the behavior); the ndarray infra mirrors with no masked
    semantics (`base`/`ctypes`/`flags`/`setflags`), pickle (`dump`/`dumps`) and array-API device
    (`device`/`to_device`), all N/A in NumSharp; `core`/`extras` (submodule refs — N/A in C#); and the
    library-wide `out=`/`subok`/`order` policy gaps (§5.5). **Every other backlog item is DONE — the module is
    functionally complete for NumSharp.Core bar structured dtypes.**

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

13. **2026-09-14 pass 10** — re-audited the "structured/record family — blocked" bucket (§1.C) and found THREE names mis-filed there that are not structured-dependent at all: `flatten_mask`, `make_mask_descr`, `mr_`. Probed `numpy.ma` up front to pin the non-structured behavior — `flatten_mask` of a 2-D int/float/bool/scalar/empty (`[[0,2],[0,0]] → [F,T,F,F]`, `False → [F]`, `[] → []`), `make_mask_descr(float64|int32|complex128) → dtype('bool')` — then read `MAxisConcatenator` in `refs/numpy` to confirm `mr_` is structured ONLY in its de-activated matrix-string path. **Key discovery — the string-slice convention forces `mr_` to follow NumSharp's `np.r_`, not raw NumPy:** `ma.mr_['0:3', a]` FAILS in NumPy (the leading STRING `'0:3'` is read as a directive and `int('0:3')` throws — real NumPy slices come from Python `:` syntax, e.g. `mr_[0:3, a]`), while NumSharp spells every slice as a colon-string library-wide, so `mr_` delegates to `np.r_`'s grammar. The mask stream reuses `np.r_` too, replacing each entry with its boolean-mask contribution of the same natural shape (masked → `getmaskarray`, colon-slice → `zeros(np.r_[slice].Shape)`, directive → passthrough). The one shape-consistency risk (FromSlice's `swapaxes(-1,trans1d)` vs FromArrayLike's defaxes permutation) is confined to `trans1d != -1` — documented, never hit in realistic use. Verified by 26 + 5 signed `dotnet run` checks (data/mask/shape/dtype/nomask/lone-string + lone-masked-array + `'r'` matrix directive; 0 fail) against NumPy 2.4.2 reference blocks, gated by 3 new `MaskedArrayTests` (**79 total**, green net8.0 + net10.0). **Test bug caught:** first cut compared `mr_` DATA via `filled(0L)` (zeros masked slots) against NumPy's `.data` (keeps the underlying value) — the mask checks all passed but 4 data checks "failed"; reading `.data` (not `filled`) fixed the comparison, confirming the underlying data at masked slots is preserved as NumPy does. No worktree quarantine needed — only my two edited files were dirty (`Ma/MaskedArray.cs`, `Ma/MaskedArrayTests.cs`).

14. **2026-09-14 pass 11** — overturned the "printing style NumSharp does not emulate" premise and closed masked printing + `masked_print_option`, plus the last tractable instance members. Read NumPy's `MaskedArray.__str__`/`__repr__` + `_insert_masked_print` out of `refs/numpy/numpy/ma/core.py`; the load-bearing discovery is that masked printing **converts the data to an OBJECT array** and prints unaligned (Python-scalar repr per element), NOT the numeric-aligned formatter — probed directly (`str(ma.array([1,200,3,40000],mask=[0,1,0,1])) == '[1 -- 3 --]'`, no width-padding of the `--` or the values). Reused NumSharp's existing byte-exact array-printing engine rather than writing a second one: threaded an optional lockstep `mask`/`display` hook through `ArrayFormatter.Recurse` (the numeric path is provably unchanged — `mask==null` is exactly the old code, confirmed by 241 printing unit tests + the 2094-case `dtype_text` oracle tier both green) and reused the pre-existing `ScalarStr`/`PythonFloatRepr` as the object-array element formatter. Decoded the `__repr__` template arithmetic (`is_one_row` = all-but-last dim == 1; `indents[k] = max(2, len("masked_array(data") - len(k))`; `dtype_needed` = non-implied OR `np.all(mask)` OR empty) and the `fill_value` branch (`np.<promoted>(value)` when the fill's `_check_fill_value` dtype ≠ data dtype). **Test bug caught (the int32-vs-int64 trap):** C# `int[]` → NumSharp int32, but NumPy's Python-list ints are int64, so my first repr expectations (assuming int64) "failed" against a correct int32 render (`fill_value=np.int64(999999), dtype=int32`) — the implementation was right, the test data wrong; casting the int cases with `.astype(np.int64)` fixed it (7 "failures" → 0). Also probed `partition`/`argpartition` under `-W ignore` to confirm NumPy's mask-oblivious behavior (data reordered, mask left positional) and `byteswap`/`fill`/`searchsorted`. Verified by a 40-check signed `dotnet run` script (str/repr across dtype × shape × masked-state, the option toggle, and the 5 instance members) + 3 property checks, **43/43 pass**, gated by 6 new `MaskedArrayTests` (**85 total**, green net8.0 + net10.0). No worktree quarantine needed — Core built clean with the untracked `ConcurrentPointingDict.cs` present, and only my two edited files were dirty.

15. **2026-09-14 pass 12** — a fresh multi-technique re-audit (do NOT assume "functionally complete" means "audited"). **Technique A — reflection surface diff:** dumped NumPy's `__all__` (226) + `MaskedArray` members (94) from Python, reflected NumSharp's `MaskedArrayModule`/`MaskedArray` public members via a signed `dotnet run` script, and set-differenced. Found instance `copy`/`choose`/`product` present only as MODULE functions → added the instance wrappers. **TRAP — the reflection filter's `set_`/`get_` accessor-name drop produced a FALSE POSITIVE for `set_fill_value`** (a real module function whose name starts with `set_`); always cross-check a flagged name against the source before "closing" it. **Technique B — a 527-case behavioral differential fuzz** (Python oracle emits per-element bit-tokens — `"M"` at masked slots, else the value's little-endian bit pattern — across 4 dtypes × 5 layouts × 28 ops + edges; C# replays and bit-compares, made layout-independent by `.copy()`-ing any non-C-contiguous result before `GetAtIndex`). Found the scalar-strong promotion divergence (§5.8). **Three harness TRAPS hit, all documented elsewhere but re-encountered:** (1) `NDArray != null` is an ELEMENTWISE compare returning `NDArray<bool>` (CS0266) — use `is not null`; (2) `GetInt64(long)` binds the COORDINATE overload (`long[]`), not a flat index → use `GetAtIndex(long)` for flat C-order reads; (3) `GetAtIndex(0)` on a 0-D result works where `GetInt64(0)` (coordinate) throws. **Discovery that made the fix tractable:** NumPy's ma scalar promotion is EXACTLY `result_type(array_dtype, np.asarray(scalar).dtype)` (verified predicted==actual across dtypes×scalars), and it must widen the INPUTS not the result (probed `ma.multiply(int32 2e9, 2) == 4e9`, so casting the result up would give the int32-wrapped value). Gated by 2 new `MaskedArrayTests` (**87 total**, green net8.0 + net10.0); post-fix fuzz 527/527. No worktree quarantine needed — only `Ma/MaskedArray.cs` + its test changed, and the ma change cannot affect non-ma code (single-file boundary).

