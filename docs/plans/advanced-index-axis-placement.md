# Multiple advanced indices mixed with slices (axis placement)

**Status: RESOLVED.** Implemented by `TryBuildMultiAdvancedGrid` in
`src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Getter.cs` (shared by the
getter and the setter `NDArray.Indexing.Selection.Setter.cs`). The recommended
"build the index grid in final order" approach (§4) was taken — no transpose needed.

Every case in the truth table below is now bit-exact with NumPy 2.4.2 (GET and SET,
including ints-as-advanced, newaxis separators, negative wrap, 2-D-broadcast advanced
indices and 4-D placements). Verified by differential probing and pinned as tests:
`test/NumSharp.UnitTest/Selection/IndexingProbeMatrix.Tests.cs` (`c32`–`c50`,
`cs14`–`cs19`, `cse03`) and `CombinedIndexing.MatrixTests.cs`
(`Get_TwoMasks_SeparatedBySlice_AdvancedAxesToFront`,
`Get_TwoArrays_ContiguousThenSlice_BlockInPlace`,
`Get_SliceThenTwoArrays_BlockAfterSlice`,
`Set_TwoArrays_SeparatedBySlice_GridValueBroadcast`). The full CI suite and the
`FuzzMatrix` differential gate stay green.

The document below is retained as the design record (the NumPy rule, references and the
implementation it describes).

**Owner of the surrounding work:** the combined boolean+advanced indexing fix
(commits `17c467d9`, `718c5a04`, `5c537104` on branch `nditer`).

---

## 1. What works today (do not regress)

Combined indexing — a boolean mask / integer array mixed with basic indices — was
fixed for the cases below. They must keep working when this gap is closed.

| Shape of the index tuple | Handler | File |
|---|---|---|
| **One** advanced index (1-D array, or 1-D mask → `nonzero`) + slices / ints / newaxis / ellipsis | `TryFetchSliceWithSingleAdvanced` (get, *slice-then-`np.take`*) / `TrySetSliceWithSingleAdvanced` (set, *meshgrid scatter*) | `Selection/NDArray.Indexing.Selection.Getter.cs`, `.Setter.cs` |
| **Leading k-D mask** + trailing basic only (`b[mask2d, 1:3]`, `b[mask2d, :]`) | `TryFetchLeadingMaskWithBasic` / inline setter via `TryBuildLeadingMaskBasicIndex` (*slice-then-partial-mask* through the unified `BooleanMask`/`BooleanMaskSet`) | same files |
| All-advanced tuples (no slice): `arr[mask,int]`, `arr[mask,mask]`, `arr[2dmask,int]`, … | masks expand to `np.nonzero()` in the `_NDArrayFound` loop, then the existing broadcast/gather path | same files |

Coverage: `CombinedIndexing.MatrixTests.cs` (~50 cases) and
`IndexingProbeMatrix.Tests.cs` (92 data-driven cases), all pinned to NumPy 2.4.2.

---

## 2. The gap

**Two or more *advanced* indices (masks/arrays) in the same tuple as a *slice*.**
All of these currently throw `IncorrectShapeException` or return the wrong shape,
because every handler above bails (`advCount != 1`, or `nz.Length != 1`, or the
trailing items are not all basic), and the fallback broadcast path treats the slice
as just another advanced index (it converts slices to index arrays via
`GetIndicesFromSlice` and broadcasts everything together).

Verified against NumPy 2.4.2 (`b = np.arange(24).reshape(2,3,4)`):

| Expression | NumPy result | NumSharp today |
|---|---|---|
| `b[ia, ib, :]`  (contiguous arrays, then slice) | `(2,4)` `[0,1,2,3,20,21,22,23]` | **ERR** |
| `b[:, ia, ib]`  (slice, then contiguous arrays) | `(2,2)` `[0,9,12,21]` | **wrong** `(2,)` `[0,21]` |
| `b[ia, :, ib]`  (arrays separated by a slice) | `(2,3)` `[0,4,8,14,18,22]` | **ERR** |
| `b[mask, :, mask2]` (masks separated by a slice) | `(2,3)` `[0,4,8,2,6,10]` | **ERR** ← the OpenBugs test |
| `b[mask, mask2, :]` (contiguous masks, then slice) | `(2,?)` | **ERR** |
| `b[ia2d, :, ib]` (separated, advanced broadcasts to 2-D) | `(2,2,3)` | **ERR** |

`ia = np.array([0,1])`, `ib = np.array([0,2])`, `mask = [T,F]`, `mask2 = [T,F,T,F]`.

### The rule (NumPy)

1. Each boolean mask becomes its `nonzero()` integer arrays (NumSharp already does
   this in the `_NDArrayFound` loop). After this everything is integer-array advanced
   indices + basic (slice/int/newaxis).
2. **All advanced indices broadcast together** into one result shape `bshape`
   (they share output axes — this is *not* an outer product with each other).
3. **Slices/ints produce their own output axes** (outer product with the advanced
   block).
4. **Axis placement of the advanced block** depends on whether the advanced indices
   are *consecutive* in the tuple:
   - **Consecutive** (no slice between them) → the advanced `bshape` axes stay
     **in place**, where the advanced block sits. `b[ia,ib,:]` → `(bshape) + (slice)`
     = `(2,4)`. `b[:,ia,ib]` → `(slice) + (bshape)` = `(2,2)`.
   - **Non-consecutive** (a slice separates them) → the advanced `bshape` axes move
     to the **front**, slices follow. `b[ia,:,ib]` → `(bshape) + (slice)` = `(2,3)`;
     `b[ia2d,:,ib]` → `(2,2) + (3)` = `(2,2,3)`.

---

## 3. NumPy reference

`src/numpy/numpy/_core/src/multiarray/mapping.c`:

- `prepare_index(...)` — boolean → integer-array conversion (NumSharp parallel: the
  `np.nonzero()` expansion in the `_NDArrayFound` loops). Already done.
- `PyArray_MapIterNew(...)` — computes `mit->consec` (lines ~2486–2513): walks the
  index ops; `consec_status` becomes `2` and `mit->consec = 0` when advanced indices
  are **non-consecutive** → advanced axes go to front; otherwise `mit->consec` is the
  axis where the contiguous advanced block sits.
- `_get_transpose(fancy_ndim, consec, ndim, getmap, dims)` (line 62) — builds the
  permutation that places the advanced (`fancy`) axes back to `consec` for a *get*,
  and the inverse for a *set*. This is the transpose to apply after the flat gather.

---

## 4. Recommended implementation

Generalize the existing setter meshgrid (`TrySetSliceWithSingleAdvanced`) into a
unified handler used by **both** get and set, replacing the single-advanced branch
when `advCount >= 2` (keep the simpler `take` / slice-then-mask fast paths for the
1-advanced and leading-k-D-mask cases — they are correct and cheaper).

Sketch (per source axis, after masks → `nonzero`):

1. Classify items → `advanced[]` (integer arrays, in tuple order) and `basic` (slices
   → `GetIndicesFromSlice` ranges; ints → length-1; newaxis → output-only).
2. `adv = np.broadcast_arrays(advanced)` → common `bshape` (advanced share these axes).
3. Decide placement: `consecutive = (max advItemIdx - min advItemIdx + 1) == advCount`.
   - Output grid axes = either `[bshape..., sliceAxes...]` (non-consecutive → front) or
     slices/advanced interleaved in source order (consecutive → in place).
4. Reshape each per-axis index array to occupy its grid position(s): advanced arrays
   span the `bshape` dims; each slice spans one own dim; ints stay length-1.
   `np.broadcast_arrays(...)` to the full grid, then **one** `FetchIndices` (get) /
   `SetIndices` (set) — both already handle N broadcast index arrays.
5. For the **consecutive** case, the simplest correct route is: build the grid as
   `[bshape..., sliceAxes...]` (advanced-first) for the gather, then `transpose` the
   result so the advanced block lands at the first-advanced-index position — i.e.
   port `_get_transpose`. For **non-consecutive**, advanced-first is already the final
   order (no transpose).

Set mirrors get: same index grid, broadcast the value to the grid shape (see the
existing `TrySetSliceWithSingleAdvanced` value-broadcast step), then `SetIndices`.

### Why this is tractable

Every primitive already exists: `np.nonzero`, `np.broadcast_arrays`,
`GetIndicesFromSlice`, `FetchIndices`/`SetIndices` (N-array advanced gather/scatter),
and `transpose`. The new logic is (a) advanced-share-vs-slice-outer-product reshaping
and (b) the `consec` transpose. No new kernels.

### Risk / effort

Moderate. It touches core indexing (`FetchIndices(object[])` / `SetIndices(object[])`),
so it must be gated behind the existing single-advanced fast paths and validated
against the full differential matrix before landing. Estimate: ~1 focused session
plus a test pass.

---

## 5. How to verify (TDD)

1. **Probe NumPy** for the truth table (reuse the scratchpad pattern):
   `b[ia,ib,:]`, `b[:,ia,ib]`, `b[ia,:,ib]`, `b[mask,:,mask2]`, `b[mask,mask2,:]`,
   `b[ia2d,:,ib]`, plus the set mirrors and 4-D variants and broadcast shapes.
2. **Replay in NumSharp**, diff shapes + C-order values.
3. **Promote** the cases into `IndexingProbeMatrix.Tests.cs` (the `CombGet`/`CombSet`
   arrays) and remove `[OpenBugs]` from
   `Get_TwoMasks_SeparatedBySlice_AdvancedAxesToFront_Unsupported` once it passes.
4. Gate: full CI-style suite green (`TestCategory!=OpenBugs&!=HighMemory&!=LargeMemoryTest`)
   and the `FuzzMatrix` differential gate.

---

## 6. Out of scope here (separate gaps, not this one)

- Raw `bool[]` / `bool[,]` (not `NDArray<bool>`) as an index — not recognized as a
  mask. Tracked by the `MaskSetter` `[OpenBugs]` test. Independent of axis placement.
