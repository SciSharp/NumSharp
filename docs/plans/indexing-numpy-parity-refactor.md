# Handover: refactor `NDArray` get[]/set[] indexing to 1-to-1 NumPy parity

**Status:** proposed. This is a design + migration plan, not yet implemented.
**Scope:** the `this[...]` getters/setters and their dispatch — `FetchIndices` /
`SetIndices` and the `Try*` handler cascade — re-architected to mirror NumPy 2.4.2's
`prepare_index` model, while respecting .NET/C# nuances.
**Non-goal:** rewriting the low-level gather/scatter kernels (the unified `BooleanMask`,
`FetchIndicesND`, `SimdMatMul`, etc. stay) — this is about the *dispatch/classification*
layer that sits above them.

---

## 0. TL;DR

NumPy parses any index into **one normalized list of typed index items** (`prepare_index`),
collapses that into a single **`index_type` bitmask**, and routes on that bitmask to exactly
one of a handful of execution paths. NumSharp instead has **fragmented entry points** and an
**~8-stage `Try*` cascade** inside the `object[]` path, each stage re-deriving the same facts
(is there a mask? a slice? how many advanced axes? where's the ellipsis?). The fix is to
introduce a NumSharp `PrepareIndex` that produces the same normalized item list + `index_type`,
then a thin router — and to funnel every entry point through it.

The current code only normalizes **one** thing (booleans → mask, `NormalizeBooleanMaskIndices`);
everything else (ellipsis fill, newaxis, 0-d scalar arrays, bool→nonzero expansion, fancy
broadcast, axis placement) is re-discovered ad hoc per handler. That is the root of the
recurring single-case bugs fixed this cycle (0-d advanced index, multi-advanced placement,
raw bool arrays, stale `(1,)` implied shapes).

---

## 1. NumPy's model (the target architecture)

Reference: `src/numpy/numpy/_core/src/multiarray/mapping.c` and `mapping.h`.

### 1a. `index_type` flags (`mapping.h:7-21`)

```
HAS_INTEGER       1     a python int / scalar  (consumes 1 axis, adds 0)
HAS_NEWAXIS       2     None / np.newaxis      (consumes 0, adds 1, size-1)
HAS_SLICE         4     a slice object         (consumes 1, adds 1)
HAS_ELLIPSIS      8     ...                     (expands to N full slices)
HAS_FANCY         16    an integer index array  (advanced; broadcasts)
HAS_BOOL          32    a single full-shape boolean mask (the whole tuple)
HAS_SCALAR_ARRAY  64    a 0-d integer array      (acts like int, but forces a copy)
HAS_0D_BOOL  (16|128)   a 0-d boolean scalar     (adds a size-0/1 axis like newaxis)
```

`index_type` is the **OR of every item's flag**. The whole routing decision is "switch on
`index_type`".

### 1b. `prepare_index` (`mapping.c:262-769`) — the single classifier

Walks the (unpacked) index tuple once and produces `indices[]` (an array of
`{type, object, value}`), plus `index_num`, the result `ndim`, and `fancy_ndim`. The cascade
**per item** (the order matters — it is the spec):

1. `...` → `HAS_ELLIPSIS` (at most one; its slice-count is resolved at the end).
2. `None` → `HAS_NEWAXIS` (adds a size-1 output axis, consumes no source axis).
3. a slice → `HAS_SLICE`.
4. **a python int (or anything that converts to one and is not an array)** → `HAS_INTEGER`,
   stores the value (this is the "scalar" case; *consumes an axis, adds none*).
5. otherwise it is array-like → coerce with `PyArray_FROM_O` (this is where **lists**,
   `bool[]`-equivalents, etc. become arrays). Empty → defaults to intp.
   - boolean array:
     - if it is the **only** index and its shape == the array's shape → `HAS_BOOL`
       (the single-full-mask fast path) and `break`.
     - if it is **0-d** → `HAS_0D_BOOL` (adds a newaxis-like axis; `True`→size 1, `False`→0).
     - else → expand to its `nonzero()` integer arrays, each `HAS_FANCY` (one per mask dim).
   - integer array:
     - if **0-d** → `HAS_INTEGER | HAS_SCALAR_ARRAY` (acts like an int but flags a forced copy).
     - else → `HAS_FANCY`.
   - anything else → `IndexError("only integers, slices ...")`.
6. After the loop: if `used_ndim < array_ndims`, fill the ellipsis (or **append a trailing
   one**); if `> array_ndims` → "too many indices". `array[()]` on 0-d → `HAS_INTEGER`.
7. `HAS_SCALAR_ARRAY` cleanup: drop the flag if the tuple also has fancy, or if it is a *pure*
   integer index (`HAS_INTEGER | HAS_SCALAR_ARRAY`) — i.e. a lone 0-d int array behaves exactly
   like an int.
8. Re-walk to validate boolean-mask axis sizes ("boolean index did not match ... along axis").

### 1c. Routing (getitem `mapping.c:1520-1644`, setitem `~1874-2050`)

```
index_type == HAS_INTEGER                         -> scalar item (get_item_pointer)
index_type == HAS_BOOL                            -> array_boolean_subscript  (single full mask)
index_type == HAS_ELLIPSIS                        -> view (whole array)
index_type & (SLICE|NEWAXIS|ELLIPSIS|INTEGER)     -> get_view_from_index (basic view)
   then if index_type & HAS_SCALAR_ARRAY          ->   PyArray_NewCopy(view)   (force a copy)
!(index_type & HAS_FANCY)                          -> return the view
index_type == HAS_FANCY && index_num == 1          -> trivial 1-D fancy fast path
otherwise                                          -> PyArray_MapIterNew (full advanced)
```

Advanced indexing (`PyArray_MapIterNew`): broadcast all fancy arrays together; slices/newaxis
become their own subspace axes; the **fancy block goes in place when the fancy indices are
consecutive, else to the front** (`mit->consec`, lines ~2486-2513), realized by
`_get_transpose` (line 62). Setitem mirrors getitem with `array_assign_boolean_subscript` /
the same mapiter scatter.

**Takeaway:** one classifier, one bitmask, one `switch`. Every behavior NumSharp re-derives
per-handler is a single field of `index_info` in this model.

---

## 2. NumSharp's current state

### 2a. Entry points are fragmented (`Selection/NDArray.Indexing.cs:32-111`)

| Indexer | Routes to | Notes |
|---|---|---|
| `this[long* dims, int ndims]` | `Storage.GetData` | pointer coords |
| `this[params NDArray<int>[] selection]` | `FetchIndices(this, …)` | fancy via typed arrays |
| `this[string slice]` | `Storage.GetView(ParseSlices)` | **basic only**, separate path |
| `this[params Slice[] slice]` | `Storage.GetView(slice)` | **basic only**, separate path |
| `this[params object[] indicesObjects]` | `FetchIndices(object[])` / `SetIndices(object[])` | the catch-all cascade |
| `NDArray<T>.this[int[]]`, `[long[]]`, `[string]`, `[Slice[]]` | own copies | generic mirror |

There is **no single classifier**. `this[string]` and `this[Slice[]]` can never see a fancy
index; the `object[]` path re-implements slice/int/ellipsis handling that `GetView` already does.

### 2b. The `object[]` cascade (`…Getter.cs:60+`, `…Setter.cs` mirror)

In order, the getter currently does:

1. `NormalizeBooleanMaskIndices` — **the only normalization** (bool array-like → `NDArray<bool>`).
2. `indicesLen == 1` switch — `NDArray`(bool→mask / else→`FetchIndices`), `int`→`GetData`,
   `bool`→newaxis/empty, **`int[]`/`long[]`→`GetData(coords)` (coordinate access!)**,
   `NDArray[]`→`this[nds]`, `object[]`→`this[objs]`, `string`→`GetView`, `null`→throw.
3. `TryFetchLeadingMaskWithBasic` — leading k-D mask + trailing basic.
4. `TryFetchSliceWithSingleAdvanced` — exactly one advanced index + slices (uses `np.take`).
5. `TryBuildMultiAdvancedGrid` — ≥2 advanced axes + an explicit slice/newaxis (meshgrid grid;
   ports the `consec` placement rule). Has its own `MixKind` mini-classifier.
6. scan loop — first `NDArray`/`int[]`/`long[]` ⇒ `goto _NDArrayFound`; else tally ints/bools/slices.
7. all-ints → `GetData(coords)`; all-bools → mask; else → `Slice[]` → `GetView`.
8. `_NDArrayFound` — the broadcast advanced path: masks→`nonzero`, slices→`GetIndicesFromSlice`,
   "premature slicing" fallback when not broadcastable, then `FetchIndices(@this, indicesArray)`.

Stages 3-5 and 8 each re-scan the tuple and re-derive "where are the masks / slices / advanced
axes". `TryBuildMultiAdvancedGrid` already contains a *local* `prepare_index`-shaped pass
(`MixKind { Adv, Int, Slice, NewAxis }`, ellipsis expansion, consec test) — evidence that the
right abstraction wants to exist; it is just trapped inside one handler.

### 2c. The gather/scatter core (kept as-is)

- `FetchIndices<T>` (`…Getter.cs`): single-index fast lane + multi-index broadcast →
  `PrepareIndexGetters` → offsets → `FetchIndicesND` (contiguous) / `FetchIndicesNDNonLinear`.
- `SetIndices<T>` (`…Setter.cs`): mirror → `SetIndicesND`. **`SetIndicesNDNonLinear` throws
  `NotImplementedException`** (fancy-set into a non-contiguous subspace is unimplemented —
  the `T1_15a/15b` `[OpenBugs]`).
- Helpers (`…Selection.cs`): `GetIndicesFromSlice`, `NormalizeIndexArray`, `PrepareIndexGetters`,
  `ExpandEllipsis`; plus `ExpandEllipsisForMixed`, `MixKind`, `TryBuildMultiAdvancedGrid` in the
  getter. Boolean engine: `Backends/Default/Indexing/Default.BooleanMask.cs` +
  `TensorEngine.BooleanMask`/`BooleanMaskSet`.

---

## 3. Divergence & quirk ledger (truth table of what differs from NumPy)

| # | Area | NumSharp today | NumPy 2.4.2 | Disposition |
|---|---|---|---|---|
| D1 | **`nd[int[]]` / `nd[long[]]`** | **coordinate access** — `nd[new int[]{0,2}]` → element `(0,2)` | **fancy** — rows `[0,2]` | **DECIDED (§5): → fancy (B).** Small blast radius; coordinate access preserved via `nd.GetData(coords)`. |
| D2 | Classifier | none; ~8-stage cascade per entry | one `prepare_index` | the refactor itself |
| D3 | Entry points | `string`/`Slice[]` bypass fancy entirely | every index funnels through `prepare_index` | unify (§4) |
| D4 | Only normalization | `NormalizeBooleanMaskIndices` | full per-item classification | the refactor |
| D5 | 0-d integer array | fixed this cycle (acts like int) — but via two ad-hoc patches | `HAS_INTEGER\|HAS_SCALAR_ARRAY` → int + forced copy | fold into classifier; **note the *copy* nuance is not modelled** (we return views) |
| D6 | 0-d boolean scalar `nd[np.array(True)]` mixed in a tuple | partial (`case bool` handles literal; 0-d bool array path murky) | `HAS_0D_BOOL` adds a newaxis-like axis | model explicitly |
| D7 | Fancy-set into non-contiguous subspace | `SetIndicesNDNonLinear` throws `NotImplemented` (`T1_15a/15b`) | works (mapiter scatter) | implement during/after refactor |
| D8 | newaxis + advanced edge cases | mostly fixed; some still latent (e.g. multiple newaxis around a 0-d advanced) | precise per `prepare_index` | classifier removes the class of bug |
| D9 | `HAS_SCALAR_ARRAY` forced copy | we return a view where NumPy returns a copy | copy | low impact; document or honor |
| D10 | Error message text/order | partially matched (the `IndexError` string is reproduced) | specific strings + priority | pin in classifier |
| D11 | `array[()]` (0-d / empty tuple) | ad hoc | `HAS_INTEGER` → scalar | classifier |
| D12 | Lists / `IEnumerable<int>` as fancy | not handled (only bool via `IEnumerable<bool>`) | `PyArray_FROM_O` coerces any sequence | see §5 (entangled with D1) |

---

## 4. Proposed architecture

Introduce a NumSharp analogue of `prepare_index`, shared by **all** entry points.

### 4a. `IndexItem` + `IndexPlan`

```csharp
internal enum IndexKind : byte { Integer, NewAxis, Slice, Ellipsis, Fancy, Bool, ZeroDBool }

internal readonly struct IndexItem
{
    public readonly IndexKind Kind;
    public readonly long      Value;   // integer value, or ellipsis slice-count, or bool-axis size
    public readonly Slice     Slice;   // for Slice
    public readonly NDArray   Array;   // for Fancy / Bool (already int64 / bool)
}

internal sealed class IndexPlan
{
    public IndexItem[] Items;          // normalized, ellipsis-resolved, masks already -> nonzero
    public int IndexType;              // OR of HAS_* flags
    public int ResultNDim;             // new_ndim + fancy_ndim
    public int FancyNDim;
}
```

### 4b. `PrepareIndex(Shape, object[]) : IndexPlan`

A direct port of §1b: one pass, same per-item cascade, same flag rules, same ellipsis fill,
same `HAS_SCALAR_ARRAY` cleanup, same validation + error strings. This **absorbs**:
`NormalizeBooleanMaskIndices`, `ExpandEllipsis`/`ExpandEllipsisForMixed`, the `MixKind` pass
inside `TryBuildMultiAdvancedGrid`, the 0-d handling, the bool→`nonzero` expansion, and the
"premature slicing" detection.

### 4c. `Route(IndexPlan)` — one switch mirroring §1c

```
HAS_INTEGER                          -> Storage.GetData(coords)               (scalar/element)
HAS_BOOL                             -> TensorEngine.BooleanMask / …Set        (single full mask)
HAS_ELLIPSIS (alone)                 -> view
& (SLICE|NEWAXIS|ELLIPSIS|INTEGER), !FANCY -> Storage.GetView(slices)          (basic view)
HAS_FANCY (consecutive / separated)  -> the meshgrid gather already in TryBuildMultiAdvancedGrid,
                                        generalized to consume IndexPlan.Items directly
```

The existing `TryBuildMultiAdvancedGrid` becomes the *single* advanced-index executor (it
already handles broadcast-together + slice-as-own-axis + consec placement, get & set). The
`TryFetchLeadingMaskWithBasic` / `TryFetchSliceWithSingleAdvanced` fast paths can stay as
*optimizations* selected from the plan (e.g. `FANCY && index_num == 1` → 1-D fast path, mirroring
NumPy's own special-case), but they are no longer the classifier.

### 4d. Funnel the typed indexers

`this[string]` and `this[Slice[]]` keep their fast `GetView` path **only when the plan is purely
basic** (which they always are today — a `Slice[]` cannot carry a fancy index, a string parses to
slices), so they can stay as-is or call `Route` with a pre-built basic plan. The `object[]` and
`NDArray<int>[]` indexers call `PrepareIndex` → `Route`. Net effect: one code path, many sugar
entry points.

---

## 5. The `int[]` decision — **DECIDED: (B) full NumPy, `int[]` = fancy**

`nd[new int[]{0,2}]` means **coordinate access** today in NumSharp (element at `(0,2)`), but
**fancy** in NumPy (rows `0` and `2`). **Decision: align to NumPy** — raw `int[]`/`long[]` as an
index becomes **fancy indexing**, per the project's "breaking changes OK to match NumPy" rule.

**Blast radius (measured) is small:**
- Implementing code: the getter `case int[] coords` / `case long[] coords` →`GetData(coords)`
  (`…Getter.cs:81-84`) and the setter mirror (`…Setter.cs:192-195`). These reroute to the fancy
  executor. The *all-scalar-int* path (`ints == indicesLen` → `GetData(coords)`, `…Getter.cs:150`)
  is the `nd[0,2]` form, which is **coordinate in NumPy too** (`a[0,2]` is `HAS_INTEGER`) — it
  **stays**. Only the *array* form moves.
- Call sites: an exhaustive scan found **no** library code and only **one** test
  (`MaskIndex_BoolArrayLikeForms`, asserting `nd[new int[]{0,2}]` → scalar) relying on the
  coordinate behavior. `Filter1D` / `NDArrayByNDArray` / `Filter2D` use `np.array(...)` or the
  implicit `NDArray nd = new int[]{…}` conversion → they are **already fancy** and do not change.
- **Coordinate access is preserved** — the shim already exists and is public:
  `NDArray.GetData(int[])` / `GetData(long[])` (`Backends/NDArray.cs:812,819`) and
  `Storage.GetData(params …)`. Migration note for users: replace `nd[new int[]{i,j}]` (was
  coordinate) with `nd.GetData(i, j)`.

**Migration step (do first, it is contained):** flip the `case int[]/long[]` branches from
`GetData(coords)` to the fancy path, update the one test, add a `[Misaligned]`-style note +
differential cases proving `nd[int[]]` now matches NumPy fancy, and document the `GetData`
coordinate shim in the indexing section of `.claude/CLAUDE.md`.

---

## 6. C# / .NET nuances the refactor must encode

1. **`params object[]` covariance.** `bool[][]` / `int[][]` (jagged, reference-element) are
   *spread* into per-row arguments before any indexer runs (`bool[][]` is assignable to
   `object[]`). So a jagged array can never arrive as one k-D index — only rectangular `T[,]`
   survives as a single item. The classifier must treat `T[,]` (rectangular) as one index and
   accept that jagged is pre-spread. (Documented in `NormalizeBooleanMaskIndices`.)
2. **No `operator=`.** `view *= 2` cannot write back through a view (C# reassigns the local).
   NumPy's `a[i] += 1` in-place semantics are unreachable for the *compound-assign-on-a-view*
   form; only `a[i] = …` (our setter) is interceptable. Keep `Slice2x2Mul_AssignmentChangesOriginal`
   `[OpenBugs]` as the documented limit.
3. **Raw arrays vs `NDArray` vs `NDArray<T>`.** Three representations reach the indexer
   (`int[]`, `NDArray`, `NDArray<int>`). The classifier should normalize to `NDArray` (int64 for
   fancy) early — but mind D1 (raw `int[]` may be coordinate, not fancy).
4. **Implicit conversions / boxing.** `NDArray nd = new double[,]{…}` works via an implicit
   operator, but inside `params object[]` the element is the *raw* array (boxed), not yet an
   `NDArray`. Conversion must happen in the classifier (`np.array(Array)`), as the bool path does.
5. **Views vs copies.** NumSharp returns **views** for basic indexing (shared memory) and fresh
   owning arrays for fancy/bool — matching NumPy *except* `HAS_SCALAR_ARRAY` (D9), where NumPy
   forces a copy and we return a view.
6. **Scalar return type.** NumPy returns a 0-d *array scalar*; NumSharp returns a 0-d `NDArray`
   (true 0-d supported). Keep returning 0-d `NDArray`.
7. **`IConvertible` / `Half` / `Complex` index values.** The cascade currently coerces these to
   `Slice.Index`. Fold into the classifier's "converts to an int" step (§1b step 4).

---

## 7. Phased migration (incremental, always green)

Each phase keeps the full suite + `FuzzMatrix` green and is independently committable.

- **P0 — characterize & freeze.** Expand the differential matrices (`IndexingProbeMatrix`,
  `CombinedIndexing.MatrixTests`) until every routing branch in §1c has a pinned case (get + set).
  This is the safety net for the rewrite.
- **P1 — `PrepareIndex` (read-only).** Implement the classifier + `IndexPlan`; assert it produces
  the right `index_type`/`ndim` for the P0 corpus *without changing routing yet* (a parallel
  oracle test). No behavior change.
- **P2 — route getter through the plan.** Replace the getter cascade with `PrepareIndex → Route`,
  reusing the existing executors (`GetView`, `BooleanMask`, `TryBuildMultiAdvancedGrid`-as-executor,
  `GetData`). Delete `TryFetchLeadingMaskWithBasic` / `TryFetchSliceWithSingleAdvanced` /
  `ExpandEllipsisForMixed` once their cases are covered by the plan (keep any as pure perf fast paths).
- **P3 — route setter through the plan** (mirror), and **implement `SetIndicesNDNonLinear`**
  (D7) so fancy-set into non-contiguous subspaces works (un-mark `T1_15a/15b`).
- **P4 — unify entry points.** Funnel `this[string]`/`this[Slice[]]`/`NDArray<int>[]` and the
  `NDArray<T>` generic indexers through the shared classifier/router (or document the basic-only
  fast paths as deliberate).
- **P5 — close residual divergences.** D6 (0-d bool in a tuple), D9 (scalar-array copy), D10
  (error text/order), and the §5 `int[]` decision if (B) is chosen.

---

## 8. Test & verification strategy

- **Differential, NumPy-as-oracle.** Every change verified by running the expression in NumPy
  2.4.2 and bit-comparing shape + C-order values (the workflow used all cycle:
  `python <<EOF … EOF` oracle vs a `dotnet run --file` replay).
- **The committed matrices** are the regression net: `IndexingProbeMatrix.Tests.cs`
  (`BoolGet/Set`, `CombGet/Set`, multi-advanced `c32–c50`/`cs14–cs19`),
  `CombinedIndexing.MatrixTests.cs`, `BooleanMasking.MatrixTests.cs`,
  `NDArray.Indexing.Test.cs`. Extend per P0.
- **`FuzzMatrix` gate** (`test/oracle` → `Fuzz/corpus/*.jsonl`) must stay bit-exact; add
  `astype`/`where`-style oracle modes for indexing if a generator is added.
- **Build/test discipline (this repo):** always `mutex-capture build -- <cmd>`; the
  stale-binary gremlin is real — use `--no-incremental` before trusting a `dotnet run --file`
  probe after editing `NumSharp.Core` (hit twice this cycle).

---

## 9. Risks & open decisions

- **R1 (decided):** the §5 `int[]` fork is **resolved to (B) fancy** — measured blast radius is
  one test + two dispatch branches; coordinate access stays available via `nd.GetData`.
- **R2:** perf — `PrepareIndex` must not regress the hot `nd[i]` / `nd["1:3"]` paths. Mitigate
  by keeping the typed `this[string]`/`this[Slice[]]` fast paths and a trivial-fancy short-circuit
  (NumPy itself special-cases these).
- **R3:** the setter's non-contiguous scatter (D7) is genuinely unimplemented; P3 is real work,
  not just routing.
- **R4:** scope creep into the kernels. Hold the line: this refactor is the *dispatch* layer.
- **R5:** concurrent-agent churn in `Backends/` — keep commits scoped to `Selection/` + tests +
  this doc, as done all cycle.

---

## 10. Reference map

**NumPy (`src/numpy/numpy/_core/src/multiarray/`):**
- `mapping.h:7-21` — `HAS_*` flags.
- `mapping.c:262-769` — `prepare_index_noarray` (the classifier; the spec for §1b).
- `mapping.c:1520-1644` — getitem routing (§1c).
- `mapping.c:~1874-2050` — setitem routing; `array_assign_boolean_subscript`.
- `mapping.c:62` `_get_transpose`, `:~2486-2513` `mit->consec` — advanced-axis placement.

**NumSharp (`src/NumSharp.Core/`):**
- `Selection/NDArray.Indexing.cs:32-111` — the five `this[...]` entry points.
- `Generics/NDArray\`1.cs:184-247` — generic mirror indexers.
- `Selection/NDArray.Indexing.Selection.Getter.cs` — `FetchIndices(object[])` cascade (`:60+`),
  `NormalizeBooleanMaskIndices`, `MixKind`/`TryBuildMultiAdvancedGrid`, `FetchIndices<T>` core.
- `Selection/NDArray.Indexing.Selection.Setter.cs` — `SetIndices(object[])` mirror, `SetIndices<T>`,
  `SetIndicesND`, **`SetIndicesNDNonLinear` (throws — D7)**.
- `Selection/NDArray.Indexing.Selection.cs` — `GetIndicesFromSlice`, `NormalizeIndexArray`,
  `PrepareIndexGetters`, `ExpandEllipsis`.
- `Backends/Default/Indexing/Default.BooleanMask.cs`, `TensorEngine.BooleanMask`/`BooleanMaskSet`.
- `View/Slice.cs`, `Storage.GetView` — basic-view machinery.

**Prior handover (subsumed by this one):** `docs/plans/advanced-index-axis-placement.md` (the
multi-advanced placement rule, now implemented in `TryBuildMultiAdvancedGrid`).

---

## 11. Definition of done

- `PrepareIndex` produces NumPy-identical `index_type`/`ndim` for the P0 corpus.
- Both getter and setter route through it; the `Try*` cascade is gone (or demoted to perf fast
  paths selected from the plan).
- `SetIndicesNDNonLinear` implemented; `T1_15a/15b` un-marked.
- Every routing branch in §1c has pinned get+set tests; full suite + `FuzzMatrix` green on
  net8.0 and net10.0.
- The `int[]` nuance (§5) is aligned to NumPy fancy (B); `nd.GetData(coords)` is documented as
  the coordinate-access replacement in code comments, this doc, and `.claude/CLAUDE.md`.
