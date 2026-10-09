# Fixed-width strings, phase 2 — computing on text (S / U): implementation plan

> Plan for Stage D phase 2 of `docs/plans/dtype-system.md`: every computation NumPy 2.4.2 runs on fixed-width text
> (`bytes_` = `S`, `str_` = `U`). Phase 1 (branch `strings`, commits `25709496` … `77b68d89`) stores, moves, casts
> between string dtypes, saves and prints text byte for byte, and defers every computation with
> `StringComputationNotSupportedException` (91 call sites). This plan replaces those deferrals with computation.
>
> It splits the work into eight subphases, each a family of functions that share one technique (§4). Every technique
> choice rests on a measurement taken for this plan (§2): NumPy's own timings, and C# prototypes of the candidate
> kernels on the same data and the same pinned cores.
>
> Source of truth: `refs/numpy/` (**2.4.2**) — `numpy/_core/src/umath/string_ufuncs.cpp`, `string_buffer.h`,
> `numpy/_core/strings.py`, `numpy/_core/defchararray.py`, `numpy/_core/src/npysort/{quicksort,mergesort,timsort,
> heapsort}.cpp`, `numpy/_core/src/multiarray/{arraytypes.c.src,scalartypes.c.src}`, `numpy/lib/_arraysetops_impl.py`.
> All behaviors and timings below were probed against `numpy==2.4.2` on CPython 3.12.12 (win-amd64, i9-13900K,
> two P-cores pinned with `start /affinity 14`) on 2026-10-09.

---

## 0. Where phase 1 leaves us

| Part | State |
|---|---|
| Storage, dtypes, movement, string↔string casts, `.npy`, printing | Done, byte-identical (phase 1). |
| Iteration | `np.nditer` and `NDIter` carry string operands with NumPy's descriptors (`5477074f`): element sizes, string buffers, S↔U buffer casts. This is the iteration base of every phase-2 kernel. |
| Refusals | Where NumPy has no string loop, NumSharp raises NumPy's type and text, after NumPy's earlier checks in NumPy's order (`Backends/Strings/NDStringUfuncs{,.Precedence,.ShapeDecided}.cs`). These stay. |
| Shape-decided answers | Empty inputs, lines of one element and pairs the `==` operator cannot compare are answered from the shape (bool stand-in). These stay. |
| Deferrals | 91 code sites throw `StringComputationNotSupportedException` (counted at `40f2d20f`): 48 compute calls (`NDStringUfuncs.NextPhase`), 29 text↔number casts (`NDStringCasting.TextConversion`), 6 number-to-text stores (`NDStringStorage.NumberToText`), 2 mixed copies (`NDStringCopy.MixedKinds`), 6 direct throws. Appendix A maps each site to a subphase. |
| The gate | The string oracle (`test/oracle/gen_strings_oracle.py` → `NumSharp.Tests.Oracle/Strings/corpus/strings.jsonl`, 27,308 cases at `40f2d20f`, 14,427 of them NumPy errors). 3,342 of them are phase 2: they pass as "deferred" only while NumSharp throws the deferral exception, and become strict as soon as a site computes. Appendix B breaks them down. |
| Phase-1 leftovers | Closed by other sessions on this branch: `np.nditer` over strings (`5477074f`), the argwhere / flatnonzero / unravel_index views (`b87ff919`), an empty QR / SVD without a backend (`2202cdac`), `ma.put` with a hard mask (`23a51194`), numpy.ma fill values (`de07ca81`). |

---

## 1. What NumPy 2.4.2 does

### 1.1 The ufunc loops (`string_ufuncs.cpp`, `string_buffer.h`)

Every loop builds a `Buffer<enc>` per element (`ASCII` for `S`, `UTF32` for `U`) and works one code unit at a time.

| Ufunc | Loops | Output | Per-element algorithm |
|---|---|---|---|
| `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` | (S,S), (U,U) only | bool | `strcmp`: compare the common prefix unit by unit as UNSIGNED values (`npy_ucs4` / `unsigned char`), then any nonzero unit in the longer operand's tail makes it greater. So trailing NULs never count (`'a' == 'a\0'`), embedded NULs do. |
| `add` | (S,S), (U,U) | the first class, width `w1 + w2` | copy `num_codepoints(a)` units, then `num_codepoints(b)` units, zero-fill the rest. |
| `multiply` | (S,int64), (int64,S), same for U | `out=` required | repeat `num_codepoints(a)` units `reps` times, truncated to the out width; `reps < 1` → empty. |
| `str_len` | S, U | intp | `num_codepoints` = index of the last nonzero unit + 1. |
| `isalpha`, `isdigit`, `isspace`, `isalnum`, `islower`, `isupper`, `istitle` | S (ASCII rules), U (`Py_UNICODE_IS*`) | bool | every code point satisfies the predicate; an empty string is False; `islower` / `isupper` / `istitle` use CPython's cased logic. |
| `isdecimal`, `isnumeric` | U only | bool | as above; no S loop. |
| `find`, `rfind`, `index`, `rindex`, `count` | (S,S,i64,i64), (U,U,i64,i64) | intp | CPython's `adjust_offsets` on `start` / `end`, then `fastsearch` (results only depend on the match semantics). `index` raises `ValueError("substring not found")`. |
| `startswith`, `endswith` | as `find` | bool | `tailmatch`. |
| `_lstrip_whitespace` … `_strip_chars` | S, U | the input dtype | ASCII whitespace for S, `Py_UNICODE_ISSPACE` for U. |
| `_replace`, `_expandtabs(_length)`, `_center`, `_ljust`, `_rjust`, `_zfill`, `_partition_index`, `_rpartition_index`, `_slice` | S, U (`_center` family also mixes an S buffer with a U fill char) | `out=` required (the Python wrappers size it) | per element; `_slice` follows `PySlice_AdjustIndices`. |

The comparison loops are also used by `compare_chararrays` with `rstrip=True` (strip trailing NULs and ASCII
whitespace first), which `np.char.equal` … `np.char.less_equal` call.

### 1.2 The Python layer (`numpy/_core/strings.py`)

Most `np.strings` functions are TWO passes of ufuncs: a length pass and the main loop into an output of the exact
width. Example, `multiply`: `a_len = str_len(a)`; an overflow check; `buffersizes = a_len * i`;
`out = np.empty_like(a, shape=buffersizes.shape, dtype=f"{a.dtype.char}{buffersizes.max()}")`; then the ufunc with
`out=`. `center`, `zfill`, `replace`, `expandtabs` and `partition` follow the same shape. So the output width is
DATA-dependent (`np.strings.multiply(['a', 'bb'], [3, 2])` is `<U4`).

Nine `np.strings` functions and `np.char.join` run per element through CPython's own `str` / `bytes` methods
(`_vec_string`):
- `upper`, `lower`, `swapcase`, `capitalize`, `title` and `translate` write into the INPUT dtype, so CPython's full
  case mapping is truncated: `upper('ß')` is `'SS'`, and `'ßa'` in `<U2` becomes `'SS'`;
- `mod`, `encode`, `decode` and `np.char.join` go through an object array, and `_to_bytes_or_str_array` sizes the
  result from the longest element.

### 1.3 `np.char` (legacy)

`np.char` re-exports 36 `np.strings` functions. Its own: the six rstrip comparisons, `compare_chararrays`,
`multiply`, `join`, `partition` / `rpartition` (one array with a last axis of 3, not a tuple), `split` / `rsplit` /
`splitlines` (object arrays), `array`, `asarray`, `chararray`.

### 1.4 Sorting and searching

- `np.sort` / `np.argsort`: `string_quicksort_` / `string_aquicksort_` (`npysort/quicksort.cpp`): introsort
  (median of three, `SMALL_QUICKSORT = 15`, insertion sort, `string_aheapsort_` past depth `2·msb(n)`), comparing with
  `STRING_LT` / `UNICODE_LT` (unsigned units over the item width). It is NOT stable: with duplicates the index order
  is the algorithm's (probed: `[29, 15, 13, 21, …]` where a stable sort gives `[1, 3, 5, 7, …]`). `kind='heapsort'` gives
  the same permutation as the default (probed). `kind='stable'` / `'mergesort'` use the stable string sorts.
- `np.partition` / `np.argpartition`: no selection loop for strings, so `PyArray_ArgPartition` falls back to the
  GENERIC `npy_aquicksort` with the dtype's compare function (`item_selection.c`). Its scans carry an extra
  `pi < pj` guard, which stops them exactly where `string_aquicksort_`'s unguarded scans stop (the element at the old
  `pj` is never below the pivot), so the permutation is the same: probed, `argpartition` returns the default
  `argsort`'s permutation. One port serves both.
- `np.argmax` / `np.argmin`: `STRING_argmax` / `UNICODE_argmax`, first occurrence. `np.max` / `np.min` stay
  refusals (no `maximum` loop).
- `np.searchsorted`: the generic binary search with the dtype's compare function.
- `np.unique`: values only → a hash table (`unique_string`, `unique.cpp`), then a sort; `return_index` → a stable
  argsort; `return_inverse` / `return_counts` → independent of the tie order. Set routines, `isin`, `lexsort`
  (stable argsorts) and `digitize` compose these.

### 1.5 Casts between text and numbers

Legacy per-element casts through Python objects (`getitem` → `setitem`):

- Number → text is the NumPy scalar's `str()`: Dragon4 shortest digits, positional from `1e-4` up to `1e3` for
  float16, `1e6` for float32, `1e16` for float64 (`scalartypes.c.src`), scientific otherwise
  (`'1e+04'`, `'6.55e+04'`, `'1.6777216e+07'`, `'1e+20'`); complex is Python's repr (`'(1+2j)'`, `'(-0-1j)'`,
  `'1e+20j'`); bool is `'True'` / `'False'`. The result is cut to the target width (`astype('U3')` of `123.456` is
  `'123'`). Default widths: bool 5, int8 4, int64 21, uint64 20, floats 32, complex 64.
- Text → number is CPython's `int()` / `float()` / `complex()` on an `np.str_` / `np.bytes_`: whitespace,
  underscores (`'1_000'`) and Unicode digits (`'١٢'` → 12) are accepted, `'1e400'` → inf. Errors:
  `invalid literal for int() with base 10: np.str_('1.5')`, `could not convert string to float: np.bytes_(b'\xff')`,
  `complex() arg is a malformed string`, and the pack check's `OverflowError: Python integer 300 out of bounds for
  int8`.
- Text → bool is truthiness: non-empty is True (`'\0'` and `''` are False, `'False'` is True).

### 1.6 Truth values

`astype(bool)`, `np.nonzero`, `count_nonzero`, `np.any` / `np.all` and `logical_and` / `or` / `xor` (any operand mix)
all read a string's truth value through the text → bool cast. `logical_not` has no loop (TypeError, probed).

### 1.7 Unicode data

`Py_UNICODE_IS*` and the case mappings come from the running CPython's database: 3.12.12 → Unicode **15.0.0**.

---

## 2. Measurements that decide the techniques

Data: N random lowercase strings of 1–8 characters as `<U8` (32 bytes) and `|S8`; a 100-value categorical set for the
duplicate-heavy cases. NumPy timed with `time.perf_counter` (best of rounds); C# prototypes are standalone
`dotnet run -c Release` programs on raw unmanaged buffers, `DOTNET_TieredCompilation=0`, the same two pinned cores.
The prototypes exclude NumSharp's fixed per-call cost (an NDArray result and an iterator: about 0.2–0.5 µs).

### 2.1 Elementwise text kernels (µs; NPY/NS = NumPy ÷ prototype)

| Operation | NumPy 1K / 100K / 1M | Prototype 1K / 100K / 1M | NPY/NS |
|---|---|---|---|
| `u == v` (`<U8`) | 16.4 / 2,224 / 22,888 | 1.1 / 165 / 3,062 | 15 / 13.5 / 7.5 |
| `u < v` (`<U8`) | 16.4 / 2,292 / 23,994 | 2.0 / 207 / 3,319 | 8.2 / 11 / 7.2 |
| `u == 'abcd'` | 17.3 / 1,675 / 17,099 | 0.8 / 108 / 1,634 | 22 / 15.5 / 10.5 |
| `<U8 == <U4` | 16.4 / 2,213 / 22,830 | 1.2 / 144 / 2,360 | 14 / 15 / 9.7 |
| `u[::2] == v[::2]` | 8.8 / 1,102 / 11,765 | 0.7 / 148 / 2,835 | 13 / 7.4 / 4.2 |
| `s == t` (`\|S8`) | 16.3 / 2,219 / 22,877 | 0.8 / 81 / 830 | 20 / 27 / 28 |
| `s < t` (`\|S8`) | 16.3 / 2,290 / 23,650 | 1.0 / 104 / 1,057 | 16 / 22 / 22 |
| outer `(1000,1) == (1,1000)` | 22,705 | 881 | 26 |
| `u.astype(bool)` | 211 / 21,790 / 214,503 | 0.8 / 108 / 1,847 | 260 / 200 / 116 |
| `str_len(u)` | 3.5 / 1,408 / 16,765 | 1.4 / 146 / 2,418 | 2.5 / 9.6 / 6.9 |
| `np.add(u, v)` | 18.9 / 5,018 / 60,494 | 2.1 / 383 / 5,876 | 9.0 / 13 / 10 |

The prototype kernels compare a whole element at once: a `<U8` element is exactly one `Vector256`, so equality is one
compare and one `EqualsAll`, ordering one compare plus a `movemask` / trailing-zero count to the first differing unit,
the length one compare with zero plus a leading-zero count, `add` two loads and three stores. NumPy's loop builds a
`Buffer` per element, scans trailing NULs and compares one unit per step (~22 ns per element).

### 2.2 Sorting and searching (µs)

| Operation | NumPy 1K / 100K / 1M | Prototype 1K / 100K / 1M | NPY/NS |
|---|---|---|---|
| `np.argsort(u)`, plain port of `string_aquicksort_` | 30.8 / 16,911 / 239,562 | 11.9 / 12,795 / 190,366 | 2.6 / **1.32** / **1.26** |
| same, the port run over (64-bit key, index) records | | 12.1 / 6,803 / 81,182 | 2.5 / 2.5 / 3.0 |
| `np.argsort(categorical)`, plain port | 42.4 / 12,117 / 157,723 | 10.9 / 8,661 / 115,107 | 3.9 / **1.40** / **1.37** |
| same, over key records | | 11.4 / 3,968 / 42,369 | 3.7 / 3.1 / 3.7 |
| `np.argsort(u, kind='stable')`, LSD radix over the keys | 55.1 / 21,167 / 362,879 | 190 / 2,342 / 28,325 | **0.29** / 9.0 / 12.8 |
| `np.sort(u)` (NumPy only) | 70.6 / 17,004 / 200,803 | | |
| `np.argmax(u)`, first-unit filter | 3.0 / 180 / 2,172 | 0.3 / 94 / 1,686 | 10 / 1.9 / **1.29** |
| `np.searchsorted(sorted, v)` (NumPy only) | 81 / 34,981 / 1,385,798 | | |
| `np.unique(categorical)` / `np.unique(u)` (NumPy only) | 43 / 3,893 / 41,929 · 147 / 35,098 / 526,976 | | |

Findings:
- A faithful introsort port is NOT enough (1.26–1.40× at 100K / 1M): the time goes to random access into the string
  buffer through the index array, not to the compare.
- Running the SAME introsort over contiguous (key, index) records, where the key is the string's order-preserving
  64-bit prefix, answers every comparison identically, so the permutation is NumPy's. The prototype checked this
  index for index against the plain port on both data sets. It runs at 2.5–3.7×.
- A stable radix sort over the keys reaches 9–13× at 100K / 1M. Its 16-bit buckets cost 190 µs at 1K, so small inputs
  need another algorithm (§3 D3).
- `argmax` at 1M reads every cache line on both sides: 1.29× is the memory-bandwidth floor of a `<U8` array.

### 2.3 Casts (µs)

| Operation | NumPy 1K / 100K / 1M | Prototype 1K / 100K / 1M | NPY/NS |
|---|---|---|---|
| int64 → `<U21` (native itoa) | 193 / 22,045 / 222,899 | 26.6 / 3,010 / 30,358 | 7.3 / 7.3 / 7.3 |
| `<U21` → int64 (native parse) | 313 / 31,850 / 325,014 | 19.0 / 2,322 / 23,845 | 16 / 14 / 14 |
| float64 → `<U32` (.NET `TryFormat`, shortest) | 510 / 59,000 / 589,224 | 318 / 34,731 / 348,501 | **1.6 / 1.7 / 1.7** |
| `<U32` → float64 (.NET `double.Parse`) | 550 / 56,462 / 568,200 | 301 / 32,133 / 321,623 | **1.8 / 1.8 / 1.8** |

The .NET formatter and parser cost 300–350 ns per float64 with 17–18 significant digits, which leaves no room for
NumPy's layout rules and CPython's parse grammar on top. The float casts need a fast shortest-digit algorithm and a
fast correctly rounded parser (§3 D6).

### 2.4 Unicode tables

CPython's string predicates against .NET's nearest APIs, all 1,112,064 non-surrogate code points (count that differ):

| Predicate | .NET 8.0.29 (Unicode 15.0) | .NET 10.0.1 (Unicode 16.0) |
|---|---|---|
| `isalpha` / `isalnum` | 0 / 0 | 4,924 / 5,004 |
| `isdecimal` | 0 | 80 |
| `isdigit` / `isnumeric` | 128 / 81 | 208 / 161 |
| `isspace` | 4 (`U+001C`–`U+001F`) | 4 |
| `islower` / `isupper` | 311 / 120 | 336 / 147 |
| `upper()` / `lower()` (one-character mappings) | 103 / 1 | 103 / 1 |

CPython defines `islower` / `isupper` by the derived Lowercase / Uppercase properties, `isdigit` / `isnumeric` by
`Numeric_Type`, and `upper` / `lower` by the full mappings (SpecialCasing); .NET uses the general category and simple
mappings. And the two target frameworks ship different Unicode versions, so .NET's APIs would make net8.0 and
net10.0 disagree with each other.

---

## 3. Technique decisions

Phase 2 adds text kernels and parsers; everything around them already exists and is fast:

| Existing component | Phase-2 use |
|---|---|
| `NDIter` with string operands (`5477074f`): broadcasting, coalescing, buffering, `out=` / `where=`, overlap copies | every elementwise text kernel (D1) and every cast loop (D6) |
| The Tier-3A inner-loop contract and the once-per-call stride-pattern dispatch (`DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs`) | the text kernels' contiguous / broadcast / strided variants |
| The trivial-loop bypass (`TryTrivialBinaryOpInto`), `Shape.NpyIterOutputShape` | the fixed cost at 1K, NumPy's result layouts |
| Phase 1's `ElementCopy` / `WhereBytes` kernels, unit views, `NDStringCopy` | concatenation copies, the narrower-`out=` truncation, `np.where` over text |
| `RadixSort.ArgSortU64`, `AxisSort.DriveAllButAxis` | the stable kinds, sorts along an axis (D3) |
| The bool machinery: `nonzero` (count + bit scan), `count_nonzero`, the `any` / `all` reductions and their prologue, `np.where`, `compress` | truth values (D5) |
| The numeric ufuncs (`maximum`, `where`, `multiply`, `max`) | the `np.strings` size passes (D7) |
| `RandomSampling/PythonInt.cs` (CPython `int()`), the Dragon4 / `format_float_*` ports (`Backends/Printing`), `PyRepr`, `PrintfFormatter` (Python `%`) | text → int, the number → text layout rules, `np.str_('…')` in error texts, `np.strings.mod` |
| The text↔number `CastingImpl`s (`DTypes/StringCastingImpls.cs`, `DTypes/CastingImpls.cs`: descriptor resolution) and the `ArrayMethod.GetStridedLoop` seam | the cast loops (D6) |
| The set-routine and `unique` compositions (`np.setops.cs`, `NDArray.unique*.cs`) | P2.2, over the new text sort / compare primitives |

**D1. Elementwise text work = IL-emitted inner loops driven by NDIter.** One `DynamicMethod` per (operation,
encoding, operand widths, inner-stride pattern), with the NDIter inner-loop contract (`NDInnerLoopFunc(dataptrs,
strides, count, auxdata)`) and `EXTERNAL_LOOP`, so broadcasting, every layout, coalescing, `out=` / `where=` and
overlap handling are NDIter's. Element widths are EMIT-TIME constants (as phase 1's `ElementCopy` kernel), so a
`<U8` compare is one `Vector256` instruction sequence and an `|S8` one a 64-bit compare. The inner-stride pattern of
each operand (contiguous = the item size, broadcast = 0, strided) is dispatched once per call; a broadcast operand's
vector and length are hoisted out of the loop. Widths up to a cap (proposed 256 bytes) are specialized; wider elements
run one runtime-width kernel per operation, which keeps the kernel cache bounded. A call whose operands are trivially
iterable skips NDIter (as `TryTrivialBinaryOpInto` does for numbers). A fresh result takes NumPy's ufunc output layout
(`Shape.NpyIterOutputShape`).

**D2. Shared element primitives.** Every family builds on six emitters, tested once over every width and both
encodings: equal (prefix + zero tail), compare (first differing unit, unsigned), length (`num_codepoints`), truth (any
nonzero unit), copy / zero-fill, and the 64-bit order-preserving key.

**D3. Sorting = 64-bit order-preserving prefix keys.** The key packs the first units big-endian: 8 bytes of an `S`
element; for `U`, 8, 4, 3 or 2 units by the largest code unit of the operand (below 2⁸, 2¹⁶, 2²¹, or any 32-bit value;
one SIMD pass finds it). Key order is string order on the prefix; equal keys of strings longer than the prefix compare
the rest.
- The default / `quicksort` / `heapsort` kinds: NumPy's `string_aquicksort_` and `string_quicksort_` (with the
  `string_aheapsort_` fallback) ported statement for statement, run over (key, index) records. The permutation is
  NumPy's by construction, ties included (§2.2: 2.5–3.7×).
- The `stable` / `mergesort` kinds: any stable algorithm gives the one stable order. Use `RadixSort.ArgSortU64` over
  the keys plus a tie pass on equal keys; below a size cutoff, a merge sort over the records.
- `np.sort` (values only): equal strings are equal bytes at one width, so the fastest order-correct algorithm wins.
- Lanes along an axis run through `AxisSort.DriveAllButAxis`.

This also gives a way to close the existing NUMERIC argsort tie-order excuse later: the numeric argsort could run
NumPy's own introsort over (key, index) records.

**D4. Searching.** `argmax` / `argmin`: gather the first unit of each element, compare it with the current best's,
fully compare only the candidates (§2.2). `searchsorted`: keys of the sorted haystack, a binary search on the keys,
a full compare on equal keys, the monotonic-bound carry for ascending needles. `unique`, `isin`, the set routines,
`lexsort`, `digitize` and the histogram family run NumPy's own statements over these primitives.

**D5. Truth values = the text → bool cast, then the existing bool machinery.** This is NumPy's own structure (§1.6).
The extra pass writes 1 byte per element against the 4–32 bytes read, and it reuses the SIMD `nonzero`,
`count_nonzero`, `any` / `all` reductions (axis, keepdims, `out=`, `where=`), `np.where`, `compress` and the logical
ufuncs unchanged.

**D6. Casts = NDIter-driven cast loops with per-element format / parse helpers that read and write code units
directly** (no managed `string` per element). The loop comes from the NEP 43 seam phase 1 already declared and left
unused: `ArrayMethod.GetStridedLoop` of the text↔number `CastingImpl`s, which already resolve the descriptors
(`ToStringCastingImpl` in `DTypes/StringCastingImpls.cs` for number → text, `LegacyWrappingCastingImpl` in
`DTypes/CastingImpls.cs` for text → number, both registered by `DTypeRegistry.InitializeStringCasts`). Its consumers
are the routes the string↔string casts take today: `NDIter.CopyAs`
(`NDStringCasting.AsType`) and the NDIter buffer manager. So `astype`, `copyto`, setitem, `concatenate`, `where`,
`choose`, `place`, `putmask`, the ufunc buffered casts, `np.array` discovery and the numpy.ma fill values all reach the
same loops.
- Integers: CPython `int()` grammar (whitespace, sign, underscores, Unicode decimal digits from the tables of D9),
  NumPy's pack range check and texts; a digit-pair itoa for formatting.
- Floats: a Ryu-family shortest-digit formatter for float64 / float32 and float16's own shortest digits, laid out by
  NumPy's scalar `str()` rules; an Eisel–Lemire parser with the .NET parser as the fallback, behind CPython's
  `float()` grammar. Verify exhaustively for float16 and float32 and on a random + boundary set for float64 (the
  existing printing port already treats .NET's shortest digits as Dragon4's unique mode).
- Complex: CPython's `complex()` grammar and Python's complex repr. Bool: truthiness and `'True'` / `'False'`.
- `Char` and `Decimal` have no NumPy analog: document their rules (proposed: `Char` as one code unit, `Decimal`
  through `decimal.ToString` / `decimal.Parse` with the invariant culture).

**D7. `np.strings` = NumPy's two-pass Python statements over text kernels.** The length / count / find passes are D1
kernels; the size arithmetic runs on the existing numeric ufuncs exactly as NumPy writes it (`maximum`, `where`,
`max`), which keeps NumPy's error order and dtypes; the main loop is a D1 kernel writing into the right-width `out=`.
Search uses a vector first-character scan plus verification.

**D8. NDExpr fusion is not used in phase 2.** The text kernels dominate each call (4–32 bytes read per element); the
fusable parts are the int64 size passes of D7 and bool masks, worth at most 20–30 % of a call, while NumPy's
unfused statements give its error order for free. Revisit after phase 2 with measurements: text predicate leaves
(compare with a scalar, `str_len`, truth) in `np.evaluate` would fuse `(names == "x") & (age > 30)`-style filters.

**D9. Unicode tables generated from CPython.** A generator script (proposed `tools/gen_unicode_tables.py`) emits
compact two-stage tables from CPython 3.12's `unicodedata` 15.0.0: the type-record flags (alpha, decimal, digit,
numeric, lower, upper, title, space, cased, case-ignorable), decimal / digit values, and the full case mappings
(lower, upper, title, with SpecialCasing expansions and the final-sigma rule). A pin test asserts the Unicode
version and a table hash. `PyRepr.IsPrintable` moves to the same tables (its documented version caveat goes away).

**D10. C# text scalars.** `NDArray.Scalar("abc")` is NumPy's `str` scalar (a 0-d `<U3`). Today a lone C# `string`
operand of an `object`-typed operator / `np.*` parameter goes through the legacy implicit `NDArray(string)` conversion
(a char array or a parsed matrix literal), so `u == "abc"` would not compare against the text. Phase 2 must make a C#
`string` operand NumPy's `str` scalar (0-d `U`) — at least in text contexts, preferably in `np.asanyarray(string)`
itself, which is a breaking change of the legacy conversion. A `byte[]` stays a numeric `uint8` array except in
parameters that are text by definition (`np.strings` `sub`, `prefix`, `suffix`, `chars`, `old`, `new`, `sep`,
`fillchar`), where it is NumPy's `bytes`. **Owner decision needed** (§6).

---

## 4. Subphases

Each subphase ships its kernels, its NumPy entry points with NumPy's parameters, unit tests from NumPy output, a new
string-oracle section with zero excuses, the phase-2 cases it unlocks turned strict, benchmark rows at 1K / 100K / 1M
across layouts, and its documentation. Every oracle section covers the ndarray kinds of the relayout tiers (C, F,
permuted, stepped, reversed, offset views, row / column / outer broadcasts, 0-d, one-element, empty, empty views)
up to 5-D, both encodings, widths from 1 to past the specialization cap, and the special units (embedded and trailing
NULs, bytes ≥ 0x80, U units above `U+10FFFF`).

### P2.0 Foundations (no new public API)

- The D2 element primitives (`Backends/Kernels/Direct/DirectILKernelGenerator.TextElement.cs`).
- The text ufunc executor: NDIter construction for `(x1[, x2], out[, where])` with string operands, kernel selection
  by (operation, encoding, widths, stride pattern, out dtype), the trivial-loop bypass, result allocation, 0-d results.
- The D9 table generator, the runtime lookup (`Backends/Strings/PyUnicodeData`), the pin test.
- Gate: primitive tests over widths 1–64 (and a few beyond the cap) × S / U × random, embedded / trailing NULs,
  bytes ≥ 0x80, U units > 0x10FFFF (reachable through views; NumPy compares them unsigned), against a scalar
  reference of NumPy's `Buffer` semantics.
- Size: medium. Prerequisite of every other subphase.

### P2.1 Comparisons and truth values

- `np.equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` for (S,S) and (U,U), with `out=`, `where=`,
  `dtype=bool`; the operators `==` `!=` `<` `<=` `>` `>=` (C# string operands per D10); broadcasting and N-D; the
  rstrip comparisons of `np.char` (`compare_chararrays`, `np.char.equal`, …).
- The text → bool cast and through it (D5): `astype(bool)`, `nonzero`, `count_nonzero`, `argwhere`, `flatnonzero`,
  `np.where(cond)`, `any` / `all`, `logical_and` / `or` / `xor`, `array_equal`, `array_equiv`, `trim_zeros`,
  `compress` / `extract`, the `place` / `putmask` masks, the histogram bin-order checks, the numpy.ma truth members.
- Kernels: 6 comparisons × 2 encodings × widths × stride patterns; the truth kernel.
- Targets (§2.1): ≥ 4× at every size (memory-bound floor at 1M strided ~4×).
- Size: medium.

### P2.2 Sorting, searching, set routines

- `np.sort` / `ndarray.sort`, `np.argsort` (all kinds), `np.lexsort`, `np.partition` / `argpartition` (full sort, as
  NumPy), `np.argmax` / `argmin` / `nanargmax` / `nanargmin` (axis, keepdims, `out=`), `np.searchsorted` (`side`,
  `sorter`), `np.unique` (all flags) and the Array-API forms, `np.isin` / `in1d`, `intersect1d`, `union1d`,
  `setdiff1d`, `setxor1d`, `np.digitize`, `np.histogram` / `histogram_bin_edges` / `histogramdd` / `histogram2d` with
  text bins, `np.sort_complex` (needs P2.4's parse).
- Technique: D3 / D4.
- Targets: argsort 2.5–3.7× (exact unstable), stable kinds and `np.sort` ≥ 7× from 100K (a small-size algorithm below
  the radix cutoff), argmax ≥ 1.9× to 100K and the 1.29× bandwidth floor at 1M (documented), searchsorted ≥ 10×
  (NumPy 1.4 µs per needle at 1M).
- Size: large.

### P2.3 Concatenation and repetition

- `np.add`, `+` (incl. `u + "xy"` and `"p" + u`), `np.strings.add`, `np.char.add`: width `w1 + w2`, `out=` narrower
  (truncation through the buffered cast, probed) or wider (zero pad), `where=`, the S + U TypeError, the
  "addition result string of length … is too large" TypeError.
- `np.multiply(text, int, out=)`, `np.strings.multiply` (the two-pass wrapper, `Overflow encountered in string
  multiply`), `np.char.multiply`.
- Kernels: concatenation (two loads, three stores at `<U8`; §2.1 9–13×), repetition (a one-character fill fast path).
- Size: small.

### P2.4 Casts between text and numbers

- Every numeric dtype ↔ S / U (D6) through `astype` and the casting rules; `np.array` of mixed lists (`['a', 1]` →
  `<U21`, `['a', 1.5, True]` → `<U32`); `np.ones` / `np.full` / `ones_like` with a text dtype (`'1'`); numbers stored
  into text arrays (`u[0] = 5` → `'5'`, `1.5` into `<U2` → `'1.'`) and text into numeric ones; `copyto`,
  `concatenate`, `where`, `choose`, `place`, `putmask`, `fill`, `np.pad` constants, the NDIter buffered casts,
  `ToMuliDimArray`, the implicit C# conversions of a 0-d text array, the numpy.ma fill values (`filled(5)`).
- The numeric functions NumPy feeds parsed text: `linalg.norm`, `cov`, `quantile` / `percentile`, `i0`, `poly` /
  `roots`, the random integer bounds, `SeedSequence` / Philox / `default_rng` seeds, `np.fft`.
- Targets: integers ≥ 7× (§2.3); floats ≥ 3× once the fast formatter and parser land (the .NET routines alone give
  1.6–1.8×).
- Size: large (the float formatter and parser are self-contained sub-units with their own exhaustive tests).

### P2.5 `np.strings`: length, search, strip, pad, replace, partition, slice

- `str_len`, `find`, `rfind`, `index`, `rindex`, `count`, `startswith`, `endswith`, `strip` / `lstrip` / `rstrip`
  (whitespace and `chars`), `replace` (`count`), `expandtabs`, `center` / `ljust` / `rjust` (the S-buffer / U-fill
  rule), `zfill`, `partition` / `rpartition` (tuple), `slice`; the wrappers' TypeErrors and ValueErrors.
- Technique: D7. U whitespace needs P2.0's tables.
- Targets (NumPy at 100K per element): `str_len` 14 ns, `find` 29, `startswith` 15, `strip` 36, `replace` 112,
  `center` 76, `multiply` 63 → ≥ 3× each.
- Size: large (many small functions on one engine).

### P2.6 `np.strings`: Unicode properties and case; the object-path functions; `np.char` extras

- `isalpha`, `isdigit`, `isspace`, `isalnum`, `islower`, `isupper`, `istitle`, `isdecimal`, `isnumeric` (S: ASCII
  rules; no S loop for the last two); `upper`, `lower`, `swapcase`, `capitalize`, `title` (CPython's full mappings,
  final sigma, the cut to the input width; NumPy's SystemError for U units > 0x10FFFF); `translate`; `mod` (the
  existing `PrintfFormatter`, values broadcast; tuple values need object arrays and stay out); `encode` / `decode`
  (the codecs .NET has natively under Python's names, `strict` / `ignore` / `replace` at least, Python's
  UnicodeError texts, `LookupError: unknown encoding` for the rest); `np.char.join`, `np.char.partition` /
  `rpartition` (stacked), `np.char.multiply`.
- Technique: per-element scalar transforms over the D9 tables, with ASCII fast paths (all units < 0x80: SIMD case
  conversion).
- Targets: NumPy's `_vec_string` costs ~310 ns per element (`upper`) → ≥ 10×.
- Size: large.

### P2.7 numpy.ma over text, the remaining sites, the switch-over

- The numpy.ma deferrals (masked sort keys, `median`, truth members, number fill values formatted into text) — most
  unlock with P2.1–P2.4; the rest are numpy.ma's own statements.
- The remaining sites (Appendix A), including `np.evaluate` with a text operand (a NumSharp extension: refuse with a
  clear message or support text leaves, D8).
- The whole corpus strict: zero deferred phase-2 cases. `StringComputationNotSupportedException` stays only for what
  remains out of scope (§5).
- Docs (`docs/website-src/docs/fundamentals/strings-and-bytes.md` "Computing on text", `.claude/CLAUDE.md`,
  `docs/plans/dtype-system.md` §5), a `benchmark/strings/` subsystem with the NPY/NS sheet, the release notes.
- Size: medium.

### Order

`P2.0 → P2.1 → { P2.2, P2.3, P2.4 } → P2.5 → P2.6 → P2.7`. P2.2, P2.3 and P2.4 are independent once P2.1's
compare primitives exist; P2.2's `sort_complex` and P2.4's int parser need P2.0's digit tables.

---

## 5. Out of scope

- NEP 55 `StringDType` (variable width, UTF-8, missing values) — phase 3.
- The object-array functions `np.char.split`, `rsplit`, `splitlines`, and `np.strings.mod` with tuple values:
  NumSharp has no object dtype.
- The `np.char.chararray` subclass (auto-rstrip semantics): NumSharp has no ndarray subclasses.
- `np.max` / `np.min` / `np.sum` / … of text: NumPy refuses them too (phase 1 answers).

---

## 6. Risks and open questions

1. **D10, the C# `string` operand (owner decision).** Changing `np.asanyarray(string)` to a 0-d `U` array is
   NumPy's answer and breaks the legacy `NDArray x = "[1, 2, 3]"` conversion.
2. **Exact `argsort` tie order.** The plan ports NumPy's introsort (§3 D3) rather than following the numeric precedent
   (a stable radix plus the oracle's tie-order excuse); the measurements show no speed cost.
3. **The Unicode version follows the oracle's Python.** CPython 3.12 → Unicode 15.0.0; moving the oracle to 3.13 or
   3.14 regenerates the tables and the oracle together.
4. **Kernel cache growth.** One kernel per (operation, widths) pair; the specialization cap (D1) bounds it.
5. **NumPy answers that depend on call history** (the ufunc dispatch cache, phase 1 trap): the corpus records a fresh
   process.
6. **Codec coverage** for `encode` / `decode` (P2.6) — which codecs and error handlers ship.
7. **Float formatter / parser exactness** (P2.4): the shortest digits must equal Dragon4's unique mode on every input
   NumPy can format; exhaustive float16 / float32 checks are cheap, float64 needs a large random + boundary set.

---

## 7. State log

- **2026-10-09 — P2.3: concatenation and repetition** (this commit). Implemented, every statement probed against
  NumPy 2.4.2:
  1. **`np.add` over two strings of one class** (NumPy's `string_add`) and its spellings: the `+` operator with an array
     or a C# string on either side (a string on the left stays the FIRST operand: Python's `__radd__` is
     `np.add(s, a)`, unlike the comparisons' reflection), and `np.strings.add` (`is np.add`, plus `string` overloads).
     The width is `w1 + w2`; trailing NULs never count, embedded ones do. NumPy's string add promoter ignores `dtype=`
     for two strings of one class (`IsIgnoredStringRequest`). The TypeErrors: the bytes / str mix, a string with a
     number, and "addition result string of length … is too large" (NumPy prints the length in BYTES for both classes:
     `descr->type` is compared with `NPY_UNICODE`, a char against a type number). An `out=` of the loop's descriptor is
     written in place; any other text `out=` receives NumPy's output cast (a loop-descriptor temporary copied in by
     `np.copyto(…, 'same_kind', where)`: truncation, padding, the ASCII decode of S into U). `where=`, NumPy's output
     layout, the trivial loop, NDIter with COPY_IF_OVERLAP.
  2. **`np.multiply(text, int, out=)`** (NumPy's `string_multiply`): the count cast to int64 (a uint64 past 2^63 wraps
     negative and repeats nothing), a one-unit text is a fill, a count below 1 or an empty text gives an empty element,
     `reps * len` past 2^63 - 1 leaves that element as it was and raises "Overflow detected in string multiply" AFTER the
     loop. The loop descriptor is `out=`'s read in the text's class (a float out reports `<U32` in the cast TypeError).
     Without `out=` NumPy's TypeError ("The 'out' kwarg is necessary …"), for the ufunc and both `*` operator orders.
  3. **`np.strings.multiply`**, the two-pass wrapper statement for statement: the integer check first ("unsupported type
     float64 for operand 'i'"; bool is not an integer), `maximum(i, 0)`, `str_len`, the float64 overflow test
     ("Overflow encountered in string multiply"), `buffersizes = len * i` (float64 after a uint64 count, so the
     descriptor string is `'U6.0'` and NumPy 2.4.2 raises "data type 'U6.0' not understood" — reproduced), the max (a
     zero-size result is the reduction ValueError), and `np.empty_like(a, shape=…)` (a's memory order).
  4. **Kernels** (`Backends/Kernels/ILKernelGenerator.Text.Concat.cs`, managed fallbacks in
     `Backends/Strings/NDStringArithmetic.cs`): concatenation (A's chunks copied and scanned for len(A) in one pass, the B
     part zeroed, B stored at len(A)), `str_len`, and the repetition in two forms. The first repetition kernel called a
     C# helper per element (NumPy's loop shape: variable-size copies, then a zero fill) and ran 0.70–1.16× NumPy: its
     copies change size element to element, which the branch predictor cannot follow (30–40 ns per `<U8` element; the
     length scan is 3.5 ns of it). The NARROW kernel (text ≤ 128 bytes, output ≤ 512 bytes) holds the text's chunks in
     locals, stores them once per copy at `k * len` into a zeroed stack row one text width wider than the output, and
     copies the row out at the output width — fixed-size stores only, 2.0–3.9× NumPy. Wider shapes keep the helper
     (doubling copies, bandwidth-bound).
  Oracle: section `strarith` (2,046 cases, zero excuses: values, widths, the operators, keyword forms, every
  catalog layout, broadcasts, `out=` of every width / class, masks, overflow, special units); 171 phase-2 cases of the
  other sections are now held strict (deferred 1,141 → 970); `strprec`'s out operands moved to `target()` (21 OWNDATA
  facets; the operands are byte-identical). `NUMSHARP_STRING_ORACLE_DEFERRED=1` lists the deferred cases with their
  refusal. Unit tests: `StringArithmeticTests` (32, incl. the kernels against a naive reference with the widths
  straddling the narrow limits, the managed route end to end; 8 planted kernel bugs, all caught) and the doc tests
  (`Computing_ConcatenationAndRepetition`). Benchmarks (two pinned P-cores, NPY/NS, 22 operations × 1K / 100K / 1M):
  geomean 5.0×; concatenation 3.3–44× (`S8` 42–44×), repetition into `out=` 2.0–3.4×, `np.strings.multiply` 1.6–4.9×.
  Under 1.5×: `add_U100` at 1M, a fresh 80 MB result (1.32–1.50×: both sides pay 12–14 ms of page faults; into a
  reused `out=` 1.52×, the one-core bandwidth ceiling at ~27 GB/s), and `add_strided_U8` at 1K in the matrix run
  (1.14–1.63×, measured before tier-1; 2.1–2.3× warmed). Open: `np.char.add` / `np.char.multiply` wait for the
  `np.char` module (C-D1; `np.char.multiply` is NOT `np.strings.multiply`: a float count is its ValueError "Can only
  multiply by integers"); `np.strings.str_len` goes public with P2.5; numbers formatted as text (`np.add(f8, f8,
  dtype='U')`) and bool results into a text `out=` wait for P2.4. C# note: `u += v` rebinds `u` to the wider result;
  NumPy's in-place form is `np.add(u, v, out: u)` (truncated to u's width, u returned).
- **2026-10-09 — P2.1's zero-excuse oracle section, and the four gaps it found** (commit cc8b103f). `gen_strings_oracle.py`
  gained section `strcomp` (8,796 cases, 90 of them NumPy errors): the six comparisons over every catalog layout and
  five U / four S width pairs, every special unit pair through an outer broadcast (U on both sides of U+10FFFF and the
  BMP, S across 0x80), the C# operators with a string on either side, keyword ufuncs (`out=` of five dtypes, `where=`,
  `dtype=`), and the truth readers (astype every order, any / all / count_nonzero / nonzero / flatnonzero / argwhere,
  any / all / count_nonzero along every axis with keepdims, the logical ufuncs, units above U+10FFFF). It replays with
  zero excuses after four fixes:
  1. **A string on the LEFT is Python's reflection.** `"" < s` asks `str.__lt__` first, which declines, so Python runs
     `s > ""` — NumPy's error names `greater` with the array's dtype first. The `(string, NDArray)` operators now evaluate
     the reflected operator on the array (same values, NumPy's texts).
  2. **The cast raises only for a LONE unit above U+10FFFF.** CPython's `_PyUnicode_FromUCS4` builds a one-unit string
     through `unicode_char` → `PyUnicode_New(1, ch)` (SystemError), but caps a longer string's maximum character at
     U+10FFFF (`find_max_char`) and builds it without an error: `[0x110000, 'A']` casts to True, `[0x110000, 0]` raises
     (probed over 15 patterns). The truth kernel's cast flavor now tests "first unit above U+10FFFF and every later unit
     zero" (one scalar compare per element, the rest re-read only on that rare branch); the managed kernel, `np.copyto`'s
     masked conversion and `ma.flatten_mask` (whose recursion builds the first non-empty element's first character: a
     first unit above U+10FFFF raises either way) follow.
  3. **`np.any` / `np.all` / `np.count_nonzero` along an axis keep the input's memory order** (api-parameter-parity X10,
     which the strides facet exposed through the truth stand-ins): the logical reductions allocate with
     `Shape.ReductionOutputShape`, `NDAxisIter.ReduceBool` writes through the result's strides, keepdims uses
     `ExpandDimension`; `count_nonzero(a, axis)` is NumPy's `a.astype(bool).sum(axis, dtype=intp)` (15–100× faster than
     the scalar loop it replaced at 100K–1M); bool C/F operands of ≥ 16K elements run np.evaluate's Any / All axis
     stream (72 cells geomean 4.2× NumPy; F-slab "read everything" cells 0.88–1.38×, L2-bound at 1M on both sides).
  4. **np.sum's bool count route kept np.evaluate's keepdims view** (OWNDATA false; NumPy owns): it now re-inserts the
     axis on the fresh counts.
  The C# replay ops live in `StringOracleOps.Compute.cs` (`operator_text`, `truth_axis`). Tests:
  `StringComparisonTruthTests` (+2: the lone-unit matrix, the reflected operators; the kernel-vs-reference test now
  reaches the failing branch and asserts it did) and `Logic/np.all_any.ResultLayout.Test.cs` (6). The string oracle
  is 36,104 cases (20 sections), zero failures, zero excuses in `strcomp`. Benchmarks (two pinned P-cores, NPY/NS): the
  truth readers unchanged or faster after the new cast test (astype(bool) U8 145–221×, U40 42–107×), text
  count_nonzero along an axis 7–18×. Open in P2.1: unchanged (np.char's rstrip comparisons, the numpy.ma members that
  fill text with `True`).
- **2026-10-09 — P2.0's kernels and the core of P2.1** (commit 7c817644). The text kernels are
  `Backends/Kernels/ILKernelGenerator.Text{,.Emit}.cs`: one IL kernel per (operation, unit, widths, operand order), the
  chunk plan of §3 D1 (unrolled up to 9 chunks, a loop beyond), broadcast-pattern stages that hoist a stride-0 comparand,
  and the truth kernel with the cast flavor's U+10FFFF flag. `Backends/Strings/NDStringComparison.cs` runs the six
  comparisons (NumPy's output layout, a trivial-loop bypass that takes any 1-D operand, NDIter for the rest, buffered
  casts into a non-bool `out=`, masks); `Backends/Strings/NDStringTruth.cs` gives the two truth routes (the bool cast,
  which raises SystemError for a U unit above U+10FFFF, and the nonzero function, which never raises). Wired: the
  comparison guard, the C# operators with a `string` operand (text context only, D10's non-breaking half),
  `astype(bool)`, `any` / `all` / `nonzero` / `count_nonzero` / `argwhere` / `flatnonzero` (their stand-ins are now the
  truth arrays), the logical ufuncs, `np.where(c, x, y)` with a text condition, `np.copyto` into bool (NumPy's masked
  conversion rule), `np.place` / `np.putmask` masks (cast before the size check, NumPy's order), `np.trim_zeros`,
  `np.array_equal` / `np.array_equiv`, the histogram bin-order checks, `ma.flatten_mask` over text (NumPy's RecursionError
  for non-empty U, byte iteration for S) and `ma.mask_or`'s data view. A text `out=` stays deferred (NumPy formats
  `'True'` / `'False'`: P2.4). Results: the string oracle went from 3,342 deferred cases to 1,141, zero failures; IL
  kernels checked against the managed reference over 4.5 M comparisons; NPY/NS over 78 benchmark cells (26 operations ×
  1K / 100K / 1M, two pinned P-cores) min 1.56×, geomean 12.3× — comparisons 4–35×, the truth readers 100–220×, `nonzero`
  2.4–3.1×, `count_nonzero` 1.6–2.2×. Deferral sites 91 → 83. Moved: the Unicode tables (P2.0) go to the subphase that
  first reads them (P2.4's digits, P2.5's whitespace). Open in P2.1: `np.char`'s rstrip comparisons (with the `np.char`
  module, P2.5), the numpy.ma members that fill text with `True` (`alltrue` …: the bool → text cast, P2.4), and a
  zero-excuse oracle section for comparisons and truth over the relayout kinds.

## Appendix A — deferral sites by subphase (91)

| Subphase | Sites |
|---|---|
| P2.1 | `NDStringUfuncs.cs` compute answers for the comparisons and logical ufuncs; `NDStringUfuncs.Precedence.cs` computing loops (shared with P2.3); `np.array_equal`, `np.array_equiv`; the masks of `np.place` / `np.putmask`; `np.histogram` / `np.histogramdd` bin-order checks ("greater"); `np.trim_zeros`. |
| P2.2 | `AxisSort` ("sort"), `AxisPartition` (×2), `NDArray.unique`, `np.isin` (×2), `np.intersect1d`, `np.setdiff1d` (×3), `np.setxor1d` (×3), `np.union1d`, `np.digitize`, `np.lexsort`, `np.nanargmax` family, `np.searchsorted`, `np.sort_complex`, `np.histogram` text data (×3), `np.histogramdd` searchsorted. |
| P2.3 | Done (2026-10-09): `NDStringUfuncs.cs` `multiply` with `out=`; the `add` computing answer. (`np.add` of numbers with a string `dtype=` stays deferred: it formats numbers as text, P2.4.) |
| P2.4 | All 29 `TextConversion` sites (`NDStringCasting`, `NDIterBufferManager`, `np.flatiter`, `NDStringArrays` ×2, `ToMuliDimArray` ×2, `np.concatenate.Strings` ×2, `np.vander`, `np.choose`, `np.place` ×2, `np.putmask`, `NDArray.fill`, `np.copyto(.Strings)` ×5, the indexing setter ×2, `NDMaskedArray` ×7); all 6 `NumberToText` sites; both `MixedKinds` sites; all 6 direct throws (`np.ones`, `StringDTypeMeta`, `NDStringArrays` ×2, `NDArray.Indexing.Strings`, `np.pad` with a number constant); `np.ones_like`; the implicit conversions; the parse-fed numerics (`linalg.norm`, `cov`, `i0`, `poly`, `roots`, `np.fft`). |
| P2.7 | `np.evaluate` (×2), `np.pad` `linear_ramp`, `NDMaskedArray` median, anything left. |

## Appendix B — the 3,342 phase-2 oracle cases

| Group (generator op) | Cases | Unlocked by |
|---|---|---|
| numpy.ma (`ma0`, `ma1`, `ma2`) | 1,352 | P2.1 / P2.2 / P2.4 / P2.7 (truth members, set routines, number fill values) |
| `hist`, `hdd_bins` | 467 | P2.1 (bin order) + P2.2 (searchsorted) |
| `reduce`, `shape_reduce`, `call`, `index_search`, `argreduce_out` | 506 | P2.1 (any / all / count_nonzero / nonzero) + P2.2 (sorts, argmax family) |
| random bounds and seeds (`randint`, `seed_seq`, `philox_counter`, `seed_spawn`, `default_rng`, `randint_onearg`) | 279 | P2.4 (CPython `int()` on text) |
| `linalg1`, `quantile`, `cov`, `poly`, `einsum`, `interp`, `reduce_out` | 192 | P2.4 (text parsed to float) |
| `ufunc`, `ufunc_dtype`, `ufunc_kw`, `operator`, `equality` | 162 | P2.1 + P2.3 |
| sort / set / search (`shape_sort`, `setop`, `unique`, `unique_api`, `isin`, `searchsorted_kw`, `partition`, `lexsort`, `sort_complex`, `digitize`, `trim_zeros`) | 158 | P2.2 |
| casts and stores (`pad`, `astype`, `ones_like`, `ones`, `full`, `setitem`, `setitem_array`, `copyto`, `concat_kw`, `concatenate`, `where`, `where3`, `place_arr`, `putmask_arr`, `choose_mode`, `array`, `repeat_arr`, `insert_obj`, `delete_obj`, `nditer_fromnumber`, `nditer_tonumber`, `nditer_pair`, `to_scalar`, `can_cast_arr`, `min_scalar_type`, `common_type`, `unravel_ix`) | 226 | P2.4 |
