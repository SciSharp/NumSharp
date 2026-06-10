# Plan — Finish #2 (44-variation matrix C/D/E) + build #3 (NpyIter behaviors)

Where we are after Plan A:

- **#2 — 44-variation matrix:** sections **A** (25 single-array layouts) and **B** (6 pairwise
  paths) are done. Sections **C** (8 per-operand flags), **D** (8 iteration flags), **E** (4
  composite execution paths) — ~20 variations — are **missing**.
- **#3 — NpyIter behaviors** (order/coalescing, buffered casting, op_axes, ranged iteration):
  **not started**. This is the white-box half and overlaps section D heavily.

Two parts. **Part 1** (C + E) extends the existing black-box differential harness — cheap, finds
more bugs in the same model. **Part 2** (D + #3) is white-box NpyIter introspection and needs new
internal accessors. Do Part 1 first; gate Part 2 on a coverage ledger.

---

## Part 1 — #2 sections C (per-operand flags) + E (composite paths)

Black-box: still NumPy-oracle, bit-exact replay through `FuzzCorpus`. New work is **operand
relationships** (aliasing, overlap, in-place, masking) and the **`out=` / `where=` parameters**,
which the corpus model doesn't exercise yet (every case is a pure `op → fresh result`).

### P1.0 — Harness prerequisites
- **OpRegistry:** add an optional pre-existing output operand and a mask operand. Function forms
  (`np.add(a, b, out=c, where=m)`, reductions' `out=`, `np.all/any(@out, @where)`) — audit which
  `np.*` expose `out=`/`where=`; where the operator form can't, use the function form. **Missing
  `out=`/`where=` support on an op is itself a finding** (feature gap → `[OpenBugs]`).
- **Corpus schema:** add `"out"` (a writable operand reconstructed like any other, whose *prior*
  bytes matter) and `"where"` (bool mask) to the case. Expected = NumPy's post-op `out` buffer
  (so unmasked positions must retain the prior value). `OpRegistry` returns the `out` array.
- **Identity + no-corruption checks:** after an `out=` op, assert the returned array **is** `out`
  (same storage) and that bytes outside the written/broadcast region are untouched.

### P1.1 — C1 in-place output (`out=`)
`np.add(a, b, out=c)`, reductions with `out=`. Vary `out` layout: contiguous, strided/sliced
(feeds **E2 source-contig + dest-strided**), broadcast-shaped reduction target
(feeds **E1 source-broadcast + dest-contig**), cross-dtype (feeds **E3 buffer-required**).

### P1.2 — C2 aliased operands
`a + a`, `np.add(a, a, out=a)`, `np.multiply(a, a, out=a)`. Generator emits 2–3 operand descriptors
pointing at the **same base buffer**. Read-before-write must not corrupt.

### P1.3 — C3 overlapping views
`a[1:] op a[:-1]`, `np.add(a[:-1], 1, out=a[1:])` (partial overlap). NumPy guarantees correct
results (it buffers when it detects overlap); assert bit-exact. This is the classic clobber hazard.

### P1.4 — C4 write-masked operand (`where=`)
`np.add(a, b, out=c, where=mask)` — only `True` positions written, the rest keep `c`'s prior bytes.
Exercises `WRITEMASKED`. Generator emits prior-`c` + `mask`; expected = NumPy's masked `c`.

### P1.5 — C5 read-only / broadcast-write protection (error parity)
Assert writing into a broadcast / non-writeable operand throws the **same** way NumPy does
(`ValueError: assignment destination is read-only`). Couples with Part 2's B5.

### P1.6 — E composite paths
Fall out of combining the above:
- **E1** source-broadcast + dest-contig — reduction with `out=` contiguous.
- **E2** source-contig + dest-strided — `out=` a sliced view.
- **E3** buffer-required — cross-dtype `out=` (forces a temp).
- **E4** reused reduce loops — chained/strided-output reductions writing successive positions.

Add these as named `out`/pair layouts (`oc_*` in `layout_catalog.py`).

**Deliverables:** `out=`/`where=` in OpRegistry + corpus schema; operand-relationship layouts
(aliased / overlap / masked / out-strided / out-broadcast / out-crossdtype); new corpora;
classifier entries for any documented divergences. **Acceptance:** every new layout bit-exact or
`[Misaligned]`; identity + no-corruption asserted.

**Risk:** the corpus reconstruction must now seed `out`'s *prior* bytes and compare its *post* bytes
(not a fresh result) — a small `FuzzCorpus`/`RunCorpus` extension. Aliased/overlap cases need the
generator to share one base across operand descriptors (the self-validation guard already proves
containment).

---

## Part 2 — #3 NpyIter behaviors (= section D, white-box)

Section D's flags (coalesce, IDENTPERM/NEGPERM, EXLOOP, RANGE, GROWINNER, GATHER,
PARALLEL_SAFE — EARLY_EXIT was deleted in Wave 1.4; early exit is a kernel property) are
**chosen by the iterator**, not selectable via `np.*`. They need white-box tests that
assert the iterator's *chosen plan*, ported from NumPy's own `test_nditer.py`
(`src/numpy/numpy/_core/tests/test_nditer.py`, ~106 functions). Land under
`test/NumSharp.UnitTest/Backends/Iterators/Parity/`.

### B0 — Coverage ledger
Map all `test_iter_*` → `{covered, gap, N/A, feature-missing}` in `docs/NPYITER_PARITY.md`.
N/A = object arrays, Python refcount (replace with NumSharp unmanaged-mem equivalents). This sizes
Part 2 and says exactly what to port vs build.

### B1 — Expose iterator state
Add internal read-only accessors on `NpyIterState` / `NpyIterRef` (via `InternalsVisibleTo`):
`ChosenOrder` (C/F/optimal), `CoalescedNDim`, `IsBuffered`, `InnerStrides`, `IterSize`. No behavior
change — introspection only.

### B2 — Order + coalescing (highest value)
Guards the KEEPORDER stride-sort the cast win relied on. Port `iter_best_order_{c,f,multi_index}`,
`dim_coalescing`, `no_inner_dim_coalescing`, `iter_best_order_multi_index_*`:
- C-contiguous → C order; F-contiguous → F order; mixed strides → stride-sorted optimal.
- Contiguous N-D collapses the iterated ndim (coalescing); non-contiguous does not.

### B3 — Buffered casting
Port `write_buffering`, `copy_casts`, `nbo_align_contig`: a dtype/alignment mismatch forces a temp
buffer; assert correctness **and** that the buffered path engaged, including partial-buffer chunk
boundaries.

### B4 — op_axes / remove_axis / ranged
Port `op_axes` (+errors), `remove_axis`, `remove_multi_index_inner_loop`,
`iterindex` / `iterrange` / `itershape` (ranged/partial traversal). **Triage first:** any not
implemented in `NpyIter` → file a `gh` feature issue + `[OpenBugs]` placeholder (document the gap,
don't fake it).

### B5 — nditer-config error parity (also covers test-kind #4)
Port `flags_errors`, `op_axes_errors`, `reduction_error`, `scalar_cast_errors`, `too_large`
(+ multi-index): assert NumSharp raises the matching exception for invalid iterator configs
(conflicting flags, broadcast-write to read-only, oversize).

**Deliverables:** parity ledger; state accessors; ported B2–B5 suites; feature-gap issues filed.
**Acceptance:** every NumPy `test_iter_*` is green, an explicit `[Misaligned]`, or a tracked
`[OpenBugs]` feature-gap — **nothing untracked**.

**Sequencing:** B0 ledger → B1 accessors → B2 (order/coalesce, highest value) → B3 (buffering) →
B4/B5. B2–B5 depend only on B1.

---

## Adjacent gaps (noted, not in scope of these two rows)

For a complete picture beyond #2/#3:

- **Op breadth (T7+):** manipulation (concatenate/stack/reshape/pad/repeat/roll), **matmul/dot**,
  bitwise, nan-aware, cumsum/cumprod, clip, median/percentile/quantile, sorting (argsort/nonzero/
  searchsorted). ~75 transformation ops still untested.
- **Parameters:** `order=` (C/F/A/K), `dtype=` accumulator override, `ddof`, middle/tuple/negative
  axis.
- **Large / SIMD-tail shapes:** sizes at the V128/V256/V512 boundaries (7/8/15/16/31/32/…) where
  kernels switch SIMD↔scalar tail.
- **Error parity (#4):** today the generators *skip* NumPy-raising cases — fold assertions into B5
  + C5 so "NumPy raises ⇒ NumSharp raises the same" is checked, not skipped.
- **Unmanaged leak/cleanup (#5), perf gates (#6 — the original 1.5×-NumPy goal), metamorphic
  invariants (#7):** separate workstreams.

## Effort shape

| Slice | Depends on | Relative size | Yield |
|-------|-----------|---------------|-------|
| P1.0 harness (`out=`/`where=`) | — | M | unblocks C/E |
| P1.1–P1.6 (C + E) | P1.0 | M | high (new bug surface: aliasing/overlap/masking) |
| B0 ledger | — | S | sizes Part 2 |
| B1 accessors | — | S | unblocks B2–B5 |
| B2 order/coalesce | B1 | M | highest (guards stride-sort) |
| B3 buffering | B1 | M | medium |
| B4 op_axes/ranged | B1 | M–L | medium (may surface feature gaps) |
| B5 error parity | B1 | M | high (closes test-kind #4) |
