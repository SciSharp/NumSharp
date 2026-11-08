# Strings and chararray: performance evaluation against NumPy 2.4.2

_2026-10-30, branch `strings` at `5ec7033b` (P3 batch 9b's `get_strided_loop` conversion included), with a before/after
A/B against `f103ab22` (the commit before the casting impls had strided loops). Numbers: `strings_full_results.md`
(+ `.tsv`), produced by `strings_full_sheet.py`. ratio = NumPy_ms / NumSharp_ms, **>1 = NumSharp faster**._

## What was measured

Every text operation over the three string dtypes, `S` (bytes), `U` (UCS-4) and `StringDType` (`T`, also with a NaN NA
object), at 1K and 100K elements, on the same arrays on both sides (the NumPy twin writes them with `np.save`, NumSharp
loads them with `np.load_npy`). Data kinds separate the cases the kernels branch on: 1–8 ASCII characters, 20 %
non-ASCII, 20–40 characters (arena storage for `T`), 10 % NA, digits and float texts, words for `split`, and stepped,
reversed and transposed views. 2,865 keys in twelve groups; 2,687 have a comparand on both sides (the rest are refused by
both, as NumPy refuses them, or crash NumPy). Each side computes a result signature (dtype, shape, total text length /
True count / sum) so a cell is only counted when both did the same work; the refusals agree on every key.

Method: both twins pinned to two P-cores (`0x14`), turbo boost disabled, best of rounds after a warm-up (NumSharp: an
untimed warm-up pass over every case first, `DOTNET_TC_CallCountingDelayMs=0`). NumPy's own run-to-run spread over all
2,687 cells: geomean 0.993, p5–p95 0.94–1.03. Every cell under 1.5×, and every cell that moved more than 1.25× between
the two builds, was re-timed in an interleaved pass (NumPy, before, tip, twice each; tip A/A p5–p95 0.85–1.10): the slow
cells reproduce (67 under 1.0×, 201 under 1.5× of the 229 re-timed).

## Headline

| | Cells | Geomean | <1.0× | <1.5× |
|---|---:|---:|---:|---:|
| All | 2,687 | **5.00×** | 76 | 207 |
| cast (astype, the casting impls' families) | 342 | 4.42× | 7 | 35 |
| **copy (NDStringCopy routes, stores)** | 202 | **2.03×** | **49** | **91** |
| meta (dtype layer, O(1)) | 15 | 7.87× | 0 | 0 |
| ufunc | 336 | 4.70× | 0 | 0 |
| reduce | 93 | 5.47× | 1 | 5 |
| sort / search / sets | 251 | 5.69× | 0 | 2 |
| np.strings (all 46 names) | 648 | 5.82× | 0 | 8 |
| np.char | 154 | 6.95× | 0 | 5 |
| chararray members / operators | 336 | 6.14× | 2 | 11 |
| index / manipulation | 156 | 3.94× | 10 | 30 |
| create / convert / print | 114 | 6.53× | 3 | 13 |
| io (.npy, text) | 40 | 3.67× | 4 | 7 |

By dtype: S 5.25×, U 5.19×, T 4.62×. By size: 1K 4.48×, 100K 5.56×.

**Verdict.** The computing surface is in very good shape: the ufuncs (min 1.64×), all of `np.strings` (min 1.20×),
`np.char`, the chararray members, sorting and the set routines are 4.7–7× NumPy; after re-timing none of their cells is
under 1.0×, and their cells under 1.5× are `S8` copies at the allocation / memcpy floor, a few 1K calls dominated by
fixed costs, and the known `U` argsort above U+FFFF. The weak layer is **data movement into fixed-width text**: the copy
/ store routes around `NDStringCopy` hold 49 of the 76 cells under 1.0× and 91 of the 207 under 1.5×, and two allocation
and indexing mechanisms below them explain most of the rest. All of them are implementation choices with a known fix,
not physical limits; the cells that are near parity for physical reasons are listed separately (section 4).

## 1. The classes named in the request

### NDStringCopy (`Backends/Strings/NDStringCopy.cs`)

| Route (how it is reached) | Cells | Result |
|---|---|---|
| `TryCopyDenseBlock` — same class and width, same layout (`copy`, `copyto`, `astype(same)`) | copy / copyto / to_same | `U` 1.0–7.4× (a 3.2 MB `copyto` is memcpy-bound on both sides, 1.0×), `T` 2.6–11×. `S8`: 0.93–1.06× — memcpy on both sides at 100K, the NDArray allocation floor (~0.35–0.5 µs) at 1K. |
| `TryAssignRaw1D` — overlapping 1-D (`a[1:] = a[:-1]`) | shift1 | 0.91–4.1×; `S8` 1K 0.91×, `U40` 1K 0.91×. |
| `TryCopyElementGrid` — ≤ 2 effective axes, any widths of a class | copy_rev / copy_step / copy_T / set_step / set_bcast / copyto_U8toU16 / U8toU4 | `U`: 1.1–6.5×. **`S8`: 0.75–1.10×; broadcast rows (`set_bcast`): 0.39–0.51× at 1K, `S8` 0.42× at 100K.** |
| Unit-view copy (`CopyUnits` over `NDIter.Copy`) — 3+ effective axes | copy_3d | `U8` 1.7–5.7×, `T` 5–9×, `S8` 0.92–1.05×. |
| S ↔ U with the ASCII check | copyto_S8toU8 / U8toS8, cast.to_U / to_S | contiguous 49–87×, stepped / reversed 9–16× (NumPy decodes / encodes through Python objects). |
| Numbers ↔ text (`NDTextConversion`) | copyto_i8toU / f8toU / Utof8 / Utoi8 | 5.3–7.3×. |
| 0-d source (`dst[...] = scalar`) | set_scalar | **`S8` 0.46× / 0.21×**, `U8` 0.90× / 1.51×, `T` 0.62–0.76×. |

Two mechanisms make the slow rows, both measured in isolation (`GetElementCopyKernel` over raw buffers, 1,000
elements):

1. **The element-copy kernel moves one element per iteration even when both rows are contiguous.** `cpblk(d, s, COPY)`
   per element costs 0.43 ns for an 8- or 16-byte element; one `Buffer.MemoryCopy` of the same run costs 0.08 / 0.13 ns
   an element. A broadcast row (`dst2d[...] = row`) is two contiguous runs per row, so it runs at the element rate:
   44 µs for 100K `S8` elements where NumPy's row memmove takes 17.8 µs. The stepped / reversed `S8` copies are at
   NumPy's own 8-byte loop (0.8–0.94×), while NumSharp's numeric int64 copies of the same layouts beat it with SIMD
   gathers and lane reverses.
2. **A 0-d source of another width is widened once per element.** `dst[...] = b'xyz'` into `S8` takes the element-grid
   route with `COPY = 3, ZERO = 5` (a `cpblk` and an `initblk` per element): 84 µs for 100K. With an `S8` scalar it is
   40 µs (point 1), and `np.copyto(dst, b'xyz')` — which converts the scalar once and fills — takes 17.6 µs, NumPy's
   time. `np.full(n, 'abc', 'S8')` (86 µs, 0.24×) and `np.full_like` (0.34×) share the route.

**Fix direction:** in `TryCopyElementGrid`, a row whose two inner strides equal the element size and `ZERO == 0` is one
block move; a stride-0 source is a fill of a pre-widened element (or delegate the 0-d case to the copyto scalar route);
the 1/2/4/8/16-byte elements can reuse the numeric strided-copy kernels (SIMD gather / reverse). Expected: `set_bcast`
and `set_scalar` 3–5× faster (to 1.5–2.5× NumPy), stepped `S8` copies past 1.5×.

### The casting impls and `get_strided_loop` (`DTypes/StringCastingImpls.cs`, `VStringCastingImpls.cs`, `CastingImpls.cs`)

At `f103ab22` every `get_strided_loop` threw (`ArrayMethod`'s base: "Stage C"); at `5ec7033b` `2535625f` gives every
storage class's casting impl its loop and `DTypeCasting.GetDTypeTransferFunction` hands them out, and `c1e5791a` moves
the converting walks onto them (the numbers ↔ text raw assignment, the iterator's string buffer transfers, the text-number
reductions, the scalar store of a number into `S` / `U`).

- **Performance-neutral, measured.** NumSharp before/after over all 2,702 cells: geomean 1.001 (cast 1.003, copy 1.016,
  text → number 1.003, number → text 0.989, every group within ±2.5 %). A buffered `np.nditer` walk with a casting
  `op_dtype` (the iterator's string buffer transfer) is unchanged too: `S8 → U8` 424 → 451 µs, `U8 → S8` 1,072 →
  1,090 µs, `U8 → U16` 171 → 177 µs, `i8 → U21` 3.61 → 3.87 ms (100K).
- **The handout is cheap.** `GetDTypeTransferFunction` (impl lookup + `resolve_descriptors` + `get_strided_loop`):
  0.08–0.20 µs a call.
- **The loops themselves** (one contiguous 100K run through `NDCastInfo.Run`):

  | Loop | ns / element | astype of the same data |
  |---|---:|---:|
  | `S8 → U8` | 4.22 | 1.81 (vectorized unit cast) |
  | `U8 → S8` | 10.72 | 2.99 |
  | `U8 → U16` | 3.01 | 3.86 |
  | `i8 → U21` / `f8 → U32` | 35.9 / 76.3 | same as the loop |
  | `U(float text) → f8` / `S(int text) → i8` | 61.2 / 43.8 | same |
  | `T → S16` / `T → U16` | 21.6 / 17.9 | same |
  | `U8 → T` / `S8 → T` / `i8 → T` | 27.4 / 23.1 / 53.7 | 27.1 / 22.9 / 70.5 |
  | `T(float text) → f8` | 47.6 | same |

  The numbers ↔ text loops ARE the conversion (CPython's `int()` / `float()` / `str()` ports) and are 2–9× NumPy. The
  **fixed-width `S ↔ U` loops are per-element and 2.3–3.6× slower than `astype`'s own vectorized unit-view route**;
  nothing on the measured surface runs them yet beyond the buffered transfers (which ran the same per-element routine
  before), but a caller moved onto them would inherit it. **Fix direction:** have the `S ↔ U` and zero-pad loops call
  the unit cast / element-copy kernels for a contiguous run before more walks are converted.

The cast families against NumPy (cast group):

| Family (casting impl) | Range | Note |
|---|---|---|
| `S ↔ U`, `S → S`/`U → U` other widths (`StringToStringCastingImpl`, `ToStringCastingImpl`) | 1.3–87× | same-dtype copies (`astype('S')` of `S8`, `astype(same)`) 0.94–1.14×: memcpy + allocation floor |
| number → `S` / `U` (`ToStringCastingImpl`) | 4.2–13× | |
| `S` / `U` text → number (`LegacyWrappingCastingImpl`) | 2.1–9.6×; text → bool 42–111× | |
| `T → T` | 2.5–10.8× | |
| `S` / `U` / number → `T` (`AnyToVStringCastingImpl`) | **`S8 → T` 1.22–1.29×** (`S40` 4.5–8.7×), `U → T` 1.6–4.3×, number → `T` 2.4–7.1×, **bool → `T` 1.12–1.43×** | per-element encode; no inline-cell fast path |
| `T → S` / `U` (`VStringToFixedWidthCastingImpl`) | **`T → S` 0.97–1.47×**, `T → U` 1.0–11.7× | per-element decode / encode |
| `T` → number / bool (`VStringToNumberCastingImpl`) | 1.8–4×; **`T → bool` 1.42–1.62×** | |

**Fix direction for `T`:** a short text (≤ 15 bytes) lives inline in its 16-byte cell. ASCII `S → T`, `T → S` and
`T → U` of inline cells are byte moves plus the size byte (one vector per cell), and `bool → T` / `T → bool` need two
constant cells / the size byte; the measured 17–27 ns an element is the general encode / decode path.

## 2. The mechanisms behind the other slow cells

### 2a. Boolean-mask stores over `S` / `U` (`copy.set_mask` 0.27–0.43×)

`a[mask] = v` with a mask of `a`'s shape reaches `NDArray.StringAdvancedSet` (`Selection/NDArray.Indexing.Strings.cs`),
which runs the store on the UNIT view: an `(n, chars)` `uint8` / `uint32` array with the `(n,)` mask broadcast across the
character axis, scattered one unit at a time by the branchy `BooleanMaskScatterKernel`. 100K `S8`: **1,923 µs**; the same
work through `np.copyto(a, v, where=mask)` (the masked-assign SIMD blend): **18.7 µs**; NumPy 800 µs. The numeric path
has the same shape at a smaller scale: `int64 a[mask] = 7` takes 553 µs, `np.copyto(where=)` 18.4 µs.
**Fix direction:** a mask of the array's shape with a 0-d / one-element value is `np.copyto(a, v, where=mask)` (NumPy's
`array_assign_boolean_subscript` makes the same reduction); route it there for every dtype. Expected 30–100×.

### 2b. Masked `copyto` of wide elements (`copy.copyto_where` U8 / S40 / U40: 0.47–0.70×)

`DirectILKernelGenerator.MaskedAssignAnyWidth` copies an element wider than its largest native width as LANES on an
extra innermost axis, so every element becomes its own run of 2–10 lanes with the run stage's setup: `U8` (32 bytes)
1,305 µs for 100K where `S8` (one native lane) takes 18.7 µs (8.9–26.9× NumPy). **Fix direction:** a per-element
vector select for 16 / 32 / 64-byte elements (one or two `Vector256` per element, the mask byte broadcast), or lanes on
an outer axis. Expected 0.5–0.7× → 3–10×.

### 2c. Zero-filled `U` allocations of 128 KiB and more fault on every call (`index.repeat` / `index.roll` U40 1K 0.20–0.21×, `io.load` U40 1K 0.56×)

A `U` buffer is always zero-filled (NumPy's `NPY_NEEDS_INIT`: `np.empty(3, 'U3')` is three empty strings), unless the
caller passes `overwriteAll` (`UnmanagedStorage.Allocate`). Zero-filled buffers bypass the pool
(`SizeBucketedBufferPool.TakeZeroed`) and, from `OsVirtualMemory.ThresholdBytes` = 128 KiB, come from `VirtualAlloc` as
fresh demand-zero pages that are released on dispose — so a result that is then written in full pays the first-touch
fault of every page on every call. Measured, allocate + write + dispose of 256 KiB: `U16` 96–121 µs (any route:
`new NDArray`, `fillZeros: false`, `np.empty`), `S16` / `float64` with `fillZeros: false` or `np.empty` 8–11 µs, and
`float64` / `S16` through the zeroing constructor or `np.zeros` 106–116 µs; 64 KiB costs 4 µs. `np.repeat(u40_1k, 2)`:
121 µs = 11 µs kernel + 3 µs allocation + ~105 µs page faults (`concatenate` and `copy`, which pass `overwriteAll`, are
fine). **Fix direction:** zero mid-size buffers with a pooled `Take` + `Clear` (as `VStringArraySlice.Allocate` already
does up to 64 MiB) instead of fresh pages, and let producers that overwrite everything (`repeat`, `roll`, `np.load`, …)
pass `overwriteAll` / allocate with `fillZeros: false`. The numeric zeroing constructor `new NDArray(dtype, shape)` that
`np.repeat` uses has the same cost for every dtype.

### 2d. `StringDType` scalar stores (`copy.set_scalar` T 0.62–0.76×, `copy.fill` T 0.99–1.28×)

`dst[...] = 'xyz'` costs 10 ns an element and `dst.fill('xyz')` 5 ns (NumPy 6.3 / 6.6). The value is the same for every
element; a short text is an inline cell, so the store is one 16-byte pattern after releasing the destination's arena /
heap texts. **Fix direction:** pack the cell once and broadcast it (5–10×).

### 2e. `.npy` of `S` (`io.save` / `io.load` S8 100K: 0.35× / 0.36×)

0.20 / 0.30 ms for an 800 KB payload where NumPy takes 0.07 / 0.11 ms (`U8` is at parity: 1.04 vs 1.12 ms). Not
investigated further; the `U40` 1K load is section 2c.

### 2f. Odd-width gathers and scatters, truth of long `U`

`index.put` U8 100K 0.61×, `index.compress` U40 100K 0.76×, `index.where` S8 100K 0.78×. `reduce.count_nonzero` U40
0.86–1.08×: the truth of a `U` element scans all 40 units, where NumPy's `UNICODE_nonzero` stops at the first nonzero
unit (one unit for a text that does not start with NUL).

## 3. NumPy 2.4.2 defects met by the twin

- **Segfault:** `np.intersect1d` / `np.setxor1d` over `StringDType` (every `T` kind, both sizes): the concatenation of
  two sorted runs sends the default-kind introsort to its heapsort fallback, which crashes for `T`. NumSharp computes them
  (0.18–0.34 ms at 1K, 26–50 ms at 100K: NumSharp-only rows); the `assume_unique` forms give the comparand: 3.8–7.6×.
- **Quadratic:** `np.setdiff1d` over `StringDType` → `np.isin`, which treats `T` as an object dtype and loops `ar1 == a`
  over every element of `ar2`: 11.7–19.7 ms at 1K where NumSharp takes 0.16–0.22 ms (71–89×), ~135 s a call at 100K
  (not run; NumSharp 27–35 ms).
- **Wrong answers:** `np.searchsorted` over arena-stored `T` texts (20–40 characters) reads the haystack with the keys'
  allocator (index sum 5.06e9 where the correct 0..n-1 sum is 5.0e9; NumSharp's is correct); `StringDType → complex128`
  writes the real part into the imaginary part too. Both are the 6 of the 8 signature mismatches; the other 2 are
  `np.empty` of `S`, which NumPy leaves uninitialized.

## 4. Cells near parity for physical reasons

- **Allocation floor at 1K:** a fresh small NDArray costs ~0.35–0.5 µs; `astype(same)` / `copy` / `flatten` /
  `np.empty` of `S8` at 1K (0.47–0.5 µs both sides) read 0.93–1.4×.
- **memcpy-bound 100K `S8` copies:** `copy` / `copyto` / `astype(same)` of 800 KB, 20 µs on both sides (1.0–1.05×).
- **`U` argsort with units above U+FFFF** (`sort.argsort` u8n 1.36–1.43×): three 21-bit units a radix key (known).

## 5. Priorities

| # | Change | Cells it lifts | Expected |
|---|---|---|---|
| 1 | `a[mask] = scalar` → masked assign (all dtypes) | set_mask ×8 (0.27–0.43×) | 30–100× faster |
| 2 | 0-d store into `S` / `U`: widen once, fill (setitem `...`, `np.full`, `full_like`) | set_scalar, full, full_like | 4–5× |
| 3 | Pooled `Take` + `Clear` for zero-filled mid-size buffers; `overwriteAll` in full-overwrite producers | repeat / roll / load U40 1K, every ≥128 KiB `U` result | 5–10× on those cells |
| 4 | Masked assign of 16+-byte elements without a lane axis | copyto_where U / S40 | 5–15× |
| 5 | Element-copy kernel: row block moves, stride-0 fill, numeric SIMD copies for 1–16-byte elements | set_bcast, stepped / reversed `S8` | 2–5× |
| 6 | `T` scalar store / fill: broadcast one packed cell | set_scalar / fill T | 5–10× |
| 7 | Inline-cell ASCII fast paths for `S/U ↔ T`, `bool ↔ T` | to_T, to_S/U from T, copyto S→T | 3–10× |
| 8 | Vectorize the `S ↔ U` / zero-pad strided loops before more walks use them | (future callers of `get_strided_loop`) | 2–4× |
| 9 | `U` truth stops at the first nonzero unit | count_nonzero / nonzero / astype(bool) of long `U` | 2–10× |

## Measurement notes

- `strings.str_len` over `t40` at 1K is bimodal per process (11.1 µs or 16.7 µs, seen in both builds); it is not a
  regression of the tip.
- Not covered: texts of hundreds of characters (the 2026-10-17 size-class survey covers them), sizes past 100K, 2-D and
  3-D layouts outside the copy / cast groups, `np.char` re-exports that are thin forwards to `np.strings` (a
  representative subset is timed), `np.char.chararray` construction from a shape.
- Reproduce: `python benchmark/strings/strings_full_sheet.py` (~40 minutes; `--filter` takes key prefixes, `$`-suffixed
  entries match one key, `--sizes` limits the sizes).
