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
expm1 log2 log10 log1p sinh cosh tanh arcsin arccos arctan deg2rad rad2deg positive rint · equal
not_equal less greater less_equal greater_equal · bitwise_and/or/xor invert left_shift right_shift ·
sum prod min max mean std var argmax argmin all any median average ptp percentile quantile
count_nonzero · cumsum cumprod diff ediff1d · nansum nanprod nanmax nanmin nanmean nanstd nanvar
nanmedian nanpercentile nanquantile · isnan isinf isfinite iscomplex isreal · maximum minimum fmax
fmin isclose allclose array_equal logical_and/or/xor/not arctan2 · where (incl. non-bool cond)
place clip (incl. complex) copyto · argsort sort (incl. NaN + strided) searchsorted nonzero
flatnonzero argwhere unique · matmul (incl. bool semiring, negstride, k=0) dot outer trace diagonal
(incl. strided/offset) · astype modf round_ convolve · ravel flatten transpose reshape squeeze
expand_dims roll rollaxis repeat tile swapaxes moveaxis delete append insert split hsplit vsplit
dsplit take compress extract put ravel_multi_index unravel_index atleast_1d/2d/3d concatenate stack
hstack vstack dstack pad. Decimal oracle (12 tiers, 695 cases) adds: power (int exponents −2…3) var
std matmul astype (↔int·float·bool·int16·uint64) · floor ceil trunc · diff · clip median ptp
percentile quantile · where · sort · ravel transpose reshape · axis reductions + keepdims + empty +
flat argmax/argmin/all/any/count_nonzero (+ the arith/compare/reduce/scan/unary already shared).

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
- [x] `iscomplex` `isreal` — **BUG**: ignore the imaginary part (complex → all-real) + garbage on strided → carved, `IsComplex_IgnoresImaginaryPart` / `IsReal_IgnoresImaginaryPart`. (2026-07-07: the GREEN side — 12 real dtypes × 5 contiguous layouts — now actually generates 120 corpus cells in `logic.jsonl`; before that the OpRegistry wiring existed but ZERO cases were emitted, so "wired" was aspirational)
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

> **np.\* Group A is complete** (33 ops wired: 7 bugs → [OpenBugs], the rest green), **including the
> decimal-specific ops** (below). Group A is fully closed.

> `flip` is intentionally NOT here: NumSharp exposes no `np.flip` (use `[::-1]` slicing). Skipped.

### Decimal-specific A-gaps (decimal rides its own C# oracle) — ✅ DONE (all green, 12 tiers)
- [x] `floor`/`ceil`/`trunc` — added to the `decimal_unary` tier (`decimal.Floor/Ceiling/Truncate`,
      exact in base-10). `DecimalUnary` gate.
- [x] `diff` (n=1,2 along last axis) — added to the `decimal_scan` tier (`DiffAxis` oracle). `DecimalScan` gate.
- [x] `clip` + order stats `median`/`ptp`/`percentile`/`quantile` — new `decimal_stat.jsonl`
      (170 cases). Oracle: `Max(lo,Min(hi,x))` + naive-sort + NumPy 'linear' interpolation in EXACT
      decimal (`Quantile`/`Median`). Validated bit-identical to NumSharp (even & odd n, q∈{0,.25,.5,.75,1}). `DecimalStat` gate.
- [x] `where(cond,a,b)` — new `decimal_where.jsonl`; the 16-byte conditional-copy kernel over
      contiguous + strided decimal. `DecimalWhere` gate.
- [x] `sort` (axis, 1-D/2-D, contiguous + strided) — new `decimal_sort.jsonl` (`SortAxis` oracle). `DecimalSort` gate.
- [x] manip `ravel`/`transpose`/`reshape` — new `decimal_manip.jsonl`; value-preserving reindex forces
      the strided decimal materialize/copy path (compared C-contiguous via `ascontiguousarray`). `DecimalManip` gate.
- [x] `nan*` reductions — **intentionally skipped**: `System.Decimal` cannot represent NaN, so
      `nansum`/`nanmax`/… are byte-identical to plain `sum`/`max`/… (already covered by `decimal_reduce`).
      Verified `np.nansum(decimal)` == `np.sum(decimal)`.

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

Still open:
- ufunc `out=` / `where=` / `dtype=` — only `out=`/overlap via the small `aliasing`/`copyto` tiers; not the full `where=`-mask / `dtype=`-loop matrix per ufunc (the ufunc surface itself is covered by `Math/UfuncDtypeOverloadTests.cs`, outside this corpus).
- Reductions — tuple/multi-axis, `ddof` beyond {0,1} only partial (`params` tier).
- Scans — `cumsum`/`cumprod` `axis=` lightly covered (mostly flattened).
- `clip` — only scalar lo/hi; no array-broadcast bounds or one-sided.
- `searchsorted` — `sorter=` not covered.
- `pad` — limited `mode=` set.
- `put`/`take` — no `mode=` (raise/wrap/clip) axis.

Closed by the 2026-07-07 remediation (`COMPLETENESS_PLAN.md`, branch `worktree-fuzz-completeness`):
- ~~`clip` excluded complex128 on a false premise~~ — complex clip covered, bit-exact (G1).
- ~~`round_` missing bool/int8/uint16/uint32/uint64/complex~~ — widened; 2 real bugs carved+pinned (G2, ledger L2/L3).
- ~~`matmul` missing bool~~ — AND/OR-semiring cases green (G3); negstride + k=0 edges added (G14).
- ~~`where` cond always bool~~ — int32/float64/uint8/complex128 truthiness conds green (G4).
- ~~`iscomplex`/`isreal` zero cases~~ — 120 green cells + carves pointing at the existing pins (G5).
- ~~index oracle `E03` (empty base) dead, V0 absent from random pools~~ — revived (G6); cross-dtype setters added (G15, `index_setter_dtype.jsonl`).
- ~~decimal negative power dead code~~ — exponents −2…3 live (G7); decimal axis/keepdims/empty/argmax-family/astype-widening (G8).
- ~~Char absent from where/logic/matmul/rounding/copyto~~ — woven, +459 cells; round_(char) FIXED, dot(char) 1-D carved+pinned (G9, ledger L8/L9).
- ~~random fuzzer 7 dtypes / 4 kinds~~ — 13 dtypes × 6 kinds incl. flat reduce + astype; the nightly soak sweeps the widened envelope (G10).
- ~~sort: no NaN, no strided~~ — NaN-contract + strided/negstride sort/argsort, zero carves (G11).
- ~~reductions never saw positive-offset views~~ — 4 offset layouts added; caught + FIXED the all/any Half/Complex offset bug (G12, ledger L4).
- ~~errors tier 10 specs~~ — 16 incl. invert(float) (post-crash-guard), min/max/argmax(empty), floor(complex), searchsorted(2-D) (G13).
- ~~trace/diagonal contiguous-only~~ — strided/offset views added (G14).
- Registry excuse over-breadth (int reductions, complex-binary blanket, cumprod/modf/hyperbolic/isclose/complex-NaN) — every branch scoped to its documented cell + tightness self-tests (B1–B7); `RunCorpus` per-file minimum-count floors (B9); invert crash guard verified + pinned (B8).
