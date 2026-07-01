# Differential-Fuzz Coverage Gaps — `np.*` op audit

Audit of which NumSharp `np.*` functions are / are not exercised by the differential-fuzz
**op corpus** (`gen_oracle.py` + `gen_decimal_oracle.cs` + `OpRegistry.cs`). The gate's contract
is **operand bytes in → result bytes out, bit-compared against an oracle**, so a function's
testability is decided by its shape. Gaps fall into three kinds:

- **A** — array→array(/scalar) ops that FIT the design but are not wired yet (the addressable gaps).
- **B** — covered by a SIBLING gate instead (indexing / printing / evaluate / metamorphic / random).
- **C** — structurally out of scope (creation / I-O / dtype-metadata / list·tuple·bool returns).

Authoritative covered set (op corpus, as of this doc): add subtract multiply divide floor_divide
mod power · negative abs sign sqrt cbrt square reciprocal floor ceil trunc sin cos tan exp log exp2
expm1 log2 log10 log1p sinh cosh tanh arcsin arccos arctan deg2rad rad2deg positive · equal not_equal
less greater less_equal greater_equal · bitwise_and/or/xor invert left_shift right_shift · sum prod
min max mean std var argmax argmin all any median average ptp percentile quantile count_nonzero ·
cumsum cumprod diff · nansum nanprod nanmax nanmin nanmean nanstd nanvar nanmedian · isnan isinf
isfinite · maximum minimum fmax fmin isclose · where place clip copyto · argsort searchsorted nonzero ·
matmul dot outer · astype modf · ravel transpose reshape squeeze expand_dims roll repeat tile swapaxes
moveaxis delete atleast_1d/2d/3d concatenate stack hstack vstack dstack pad. Decimal oracle adds:
power var std matmul astype (+ the arith/compare/reduce/scan/unary already shared).

---

## Group A — array→array ops NOT wired (addressable gaps) — WORK LIST

Status: `[ ]` todo · `[~]` in progress · `[x]` wired (green or carved+[OpenBugs]).

### Batch 1 — elementwise (oracle-trivial)  ✅ DONE (all green)
- [x] `logical_and` `logical_or` `logical_xor` — binary → bool · green
- [x] `logical_not` — unary → bool · green
- [x] `arctan2` — binary trig → float · green

### Batch 2 — reductions / rounding / linalg  ✅ DONE (3 bugs → [OpenBugs])
- [x] `sort` — value sort, axis {-1,0,1} · green
- [x] `round_` / `around` — green at decimals {0,1,2}; **BUG**: dec=-1 (int throws / float wrong) + float16 fractional → carved, `OpenBugsDtypeCoverageTests.Round_NegativeDecimals_Broken` / `Round_Float16_Fractional_Diverges`
- [x] `trace` — green for signed/float/complex; **BUG**: trace(unsigned) → Int64 not uint64 → carved, `Trace_Unsigned_WrongResultDtype`
- [x] `diagonal` — green
- [x] `ediff1d` — green
- [x] `nanpercentile` / `nanquantile` — green on finite+NaN data (inf-interpolation edge out of scope)

### Batch 3 — selection / search / set  ✅ PARTIAL (2 bugs → [OpenBugs])
- [x] `argwhere` `flatnonzero` — green
- [x] `unique` — green on contiguous+finite (raw-offset views = #11 unreachable-via-API; inf/NaN complex ordering out of scope)
- [x] `allclose` `array_equal` — whole-array → 0-D bool · green
- [x] `iscomplex` `isreal` — **BUG**: ignore the imaginary part (complex → all-real) + garbage on strided → carved, `IsComplex_IgnoresImaginaryPart` / `IsReal_IgnoresImaginaryPart`
- [x] `take` `put` `compress` `extract` — green (groupa tier)
- [x] `ravel_multi_index` `unravel_index` — green (groupa tier). `indices` = creation-shaped (no operand) → Group C
- [x] `convolve` — 1-D convolution (full/same/valid) · green

### Batch 4 — shape (single-array)  ✅ DONE (all green, groupa tier)
- [x] `flatten` — C-order copy (contiguous + non-contiguous source) · green
- [x] `rollaxis` — green
- [x] `append` `insert` — green

### Batch 5 — multi-output (per-output-piece op, like modf's `_frac`/`_int`)  ✅ DONE (all green, groupa tier)
- [x] `split` `hsplit` `vsplit` `dsplit` — one case per output piece · green (`array_split` == split mechanism)

### Batch 6 — whole-array predicates (→ scalar/elementwise bool)  ✅ DONE
- [x] `allclose` `array_equal` — whole-array → 0-D bool · green
- [x] `iscomplex` `isreal` — **BUG** (see Batch 3): carved → [OpenBugs]

> **np.\* Group A is complete** (33 ops wired: 7 bugs → [OpenBugs], the rest green). Only remaining
> item is the **decimal-specific ops** (extend `gen_decimal_oracle.cs`: floor/ceil/trunc/clip/where/
> sort/median/ptp/percentile/quantile/diff/manip/nan*).

> `flip` is intentionally NOT here: NumSharp exposes no `np.flip` (use `[::-1]` slicing). Skipped.

### Decimal-specific A-gaps (decimal rides its own C# oracle, currently 8 tiers)
- [ ] decimal `floor`/`ceil`/`trunc`, `clip`, `where`, `sort`, `median`/`ptp`/`percentile`/`quantile`,
      `diff`, the manip ops, `nan*` reductions — not in `gen_decimal_oracle.cs` yet.

---

## Group B — covered by a SIBLING gate (not the op corpus, by design)

| Area | Owner |
|------|-------|
| Indexing — getter/setter, fancy/boolean/slice (covers `take`/`compress`-style access) | `IndexOracleTests` (`gen_index_oracle.py`) |
| `np.evaluate` (fused NDExpr) | `NDEvaluateTests.cs` |
| Printing — `array2string`, `array_repr`/`array_str`, `format_float_*`, `printoptions` | dedicated ~18K-case `ToString` fuzz suite |
| Round-trips / involutions / identities | `MetamorphicTests.cs` |
| `np.random.*` | seed/state parity (MT19937) tests |

---

## Group C — structurally out of scope (operand-replay differential can't fit these)

| Group | Functions | Why |
|-------|-----------|-----|
| Creation | `arange` `array` `as*array` `copy` `empty(_like)` `eye` `frombuffer` `full(_like)` `identity` `linspace` `meshgrid` `mgrid` `ones(_like)` `zeros(_like)` | no input operand (`empty` also non-deterministic) |
| Dtype/promotion metadata | `can_cast` `common_type` `find_common_type` `result_type` `promote_types` `min_scalar_type` `mintypecode` `issubdtype` `isdtype` `finfo` `iinfo` `sctype2char` `maximum_sctype` | return dtype objects / bools / info structs |
| Broadcasting helpers | `broadcast` `broadcast_arrays` `broadcast_to` `are_broadcastable` | return views/tuples/bools; broadcasting is exercised via the broadcast LAYOUTS |
| Object-level predicates | `isscalar` `iscomplexobj` `isrealobj` `array_equiv` | scalar bool about the object |
| File I/O | `load` `save` `fromfile` `tofile` (+ `Npz`) | disk I/O; own round-trip tests |
| Misc | `asscalar` `size` `multithreading` | scalar / config |

---

## Group D — parameter/mode gaps WITHIN covered ops

- ufunc `out=` / `where=` / `dtype=` — only `out=`/overlap via the small `aliasing`/`copyto` tiers; not the full `where=`-mask / `dtype=`-loop matrix per ufunc.
- Reductions — `keepdims=True`, tuple/multi-axis, `ddof` beyond {0,1} only partial (`params` tier).
- Scans — `cumsum`/`cumprod` `axis=` lightly covered (mostly flattened).
- `clip` — only scalar lo/hi; no array-broadcast bounds or one-sided.
- `searchsorted` — `sorter=` not covered.
- `pad` — limited `mode=` set.
