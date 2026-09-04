# NDIter / fancy-index performance — continuation document (2026-09-03)

The entry point for the next session on this line of work. It records where the tree stands after
Tier 1 of `docs/NDITER_PERF_DISCOVERY.md` §7 closed, the exact protocol every number here was
taken with, the gaps that remain **measured** (not remembered), and a ranked plan with a concrete
design per lever — so the next session starts by re-running one probe and picking a lever, not by
rediscovering the cost model.

Read in this order: §1 (state), §2 (protocol — non-negotiable), §3 (what is still slow), §4 (the
plan), §6 (gates). §5 is the trap list; every trap in it cost this line of work at least one
measurement round.

---

## 1. Where the tree stands

Branch `journey3`, on top of `ccadeef4` (the 2-D block kernel second generation + NumPy-style axis
coalescing):

| commit | what |
|--------|------|
| `09d1fc59` | Tier 1 closed: fancy index on take/put flat kernels, `where=` SIMD run scan, `where=` admitted into the 2-D block kernel, coalescer reach verified |
| `bce0a2fc` | the int32-index-in-place kernel variants kept on measurement (probe section D) |

**Gates at `bce0a2fc`:** main suite 14476 / 0 (CI filter), FuzzMatrix 98 / 98,
`Indexing.KernelRoute.Tests` 30 / 30, net8.0 build clean, the masked 2-D kernel's 8,358-check
self-consistency probe 0 mismatches.

### 1.1 What landed, and where it lives

| lever | code | pinned result (NPY/NS, higher = NumSharp faster) |
|-------|------|--------------------------------------------------|
| Fancy get/set on the take/put kernels — ONE integer index array over a C-contiguous source of any rank (`a[idx]`, `a[idx] = v`, `m[ridx]`, `m[ridx] = row`) | `Selection/FancyIndexKernels.cs` (route, SIMD bounds pre-scan), `Backends/Kernels/Direct/DirectILKernelGenerator.GatherFlat.cs` (flat gather/scatter, compile-time width), `DirectILKernelGenerator.{Take,Put}.cs` (`idx32` variants), the gates in `Selection/NDArray.Indexing.Selection.{Getter,Setter}.cs` (`FetchIndices<T>` / `SetIndices<T>` / `TryScatterKernel<T>`), `Indexing/np.{take,put}.cs` (int32 in place, flat kernel for RAISE) | 100K f64: `a[idx32]` 0.81 → **4.97**, `a[idx64]` 0.22 → **1.26**, `a[idx64] = v` 0.37 → **1.13**, `m[ridx] = row` 2.1 → **6.3** (10M: 0.33 → 1.27), `np.take(a, idx32)` 3.6 → **9.5** |
| `where=` masked driver run scan (one movemask per 32 mask bytes, runs walked by trailing-zero counts) | `Backends/Iterators/NDIter.Execution.cs` → `InvokeInner`, `SkipMaskFalse`, `SkipMaskTrue` | 100K add: all-false 27.7 → 1.5 µs (3.5×), half 34 → 8.2 (2.8×), 64-runs 43 → 16.9 (1.4×), run-length-1 unchanged (1.1×) |
| `where=` admitted into the 2-D block kernel (mask as the trailing operand; byte mask → lane mask via pmovzx + compare; row-gated `(rows,1)` masks; masked scalar block otherwise) | `DirectILKernelGenerator.InnerLoop2D.Masked.cs`; `NDIter.Execution.Custom.cs` → `Is2DMaskedElementwiseShape`, `TryExecute2DMasked` (string AND packed-key overloads), `Run2DMaskedBlocks` | 1M f64 `(rows, w)` views: w=3 add 0.66 → **1.67**, w=4 sqrt 1.26 → **6.24**, w=4 column mask 2.62 → **10.5**; unmasked rows unchanged |
| Coalescer reach on copy / cast / reduce — verified, nothing to do | `Backends/Iterators/NDIterCoalescing.cs` (unchanged) | pre-coalescer `fb6e34cd` vs `ccadeef4`: reduce 0.997×, copy 1.031×, elementwise 1.068×, cast 0.992× |

Full tables: `docs/NDITER_PERF_DISCOVERY.md` §6.6; the masked kernel's design: `docs/NDITER_2D_BLOCK_KERNEL.md` §12;
the one-paragraph summary: `.claude/CLAUDE.md` → NDIter section.

### 1.2 Decisions made on this line (do not re-litigate without new numbers)

- **NumSharp indexes with int64/long everywhere; a narrower index width is admissible only where it
  is measured faster.** The `idx32` kernel variants stay because the in-place read beat the widen-
  to-int64-temp alternative 1.7–2.5× at 1K, 1.1–1.8× at 100K, 1.2× at 10M, while an int32 index
  costs the same as a native int64 one (0.97–1.13×). All index ARITHMETIC is int64; only the load
  width and the cursor step differ. `fancy_where_probe.cs` section D reproduces it; drop the
  variants if that ever stops being true.
- **A fancy assignment never writes partially.** NumPy's `mapiter_trivial_set` validates every
  index before its first store; `TryScatterKernel<T>` pre-scans with `Vector<T>` for the same
  reason. Do not trade the scan for the kernel's fail-return.
- **Kernel results must be byte-identical to the route they replace.** Every block kernel reuses
  the per-chunk kernel's emit bodies; the fancy route is pinned element-for-element against the
  old route's semantics by `Indexing.KernelRoute.Tests`.

---

## 2. Measurement protocol (every number in this document)

    # C# side — pinned to one P-core, tiered-compilation delay off, single-threaded BLAS
    NS_PROBE_AFFINITY=0x4 DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
      dotnet run -c Release benchmark/nditer/probes/<probe>.cs [sections] > after.tsv
    # NumPy side — same core, never at the same time
    NS_PROBE_AFFINITY=0x4 python benchmark/nditer/probes/numpy_twins.py <twin> [sections] > numpy.tsv
    # join (b/a = before/after speedup, NPY/NS = NumPy time / NumSharp time)
    python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv

| probe | twin | sections |
|-------|------|----------|
| `fancy_where_probe.cs` | `fancy_where` | A fancy get/set vs take/put (1K/100K/10M, int32 + int64) · B `where=` run lengths · C `where=` narrow rows · D int32-in-place vs widen (C# only) |
| `neighbours_probe.cs` | `neighbours` | E fancy-index shapes still on the old route · F `where=` neighbours · G the Tier 2 claims |
| `narrow_probe.cs` | `narrow` | the unmasked 2-D block kernel (widths, dtypes, broadcast shapes, 3-D, floors) |
| `fixed_cost_probe.cs`, `ab_ops_probe.cs`, `angles_probe.cs` | `fixed`, `ab`, `angles` | construction cost, the broad A/B, the original angle measurements |

**A "before" is a detached worktree, never a memory and never the live tree:**

    git worktree add --detach <scratch>/pre  <before-sha>
    git worktree add --detach <scratch>/post <after-sha>
    # rewrite the probe's `#:project` line to <scratch>/pre/src/NumSharp.Core/NumSharp.Core.csproj
    # (a fresh filename per rebuild — the runfile cache keys on the file), run, then the same for post
    git worktree remove --force <scratch>/pre   # when done

For the official subsystem benches (`benchmark/layout`, `benchmark/cast`) the same A/B was driven by
injecting the `NS_PROBE_AFFINITY` pin after the last `using` line of each `*_bench.cs` and after the
`import numpy` line of each `*_bench.py`, then running them with `dotnet run -c Release -` on stdin
exactly as `benchmark/scripts/bench_common.py` does. **`bench_common.py` itself does not pin** — the
first thing to do before the next official run is to add that (the harness-level item of the
original status list; not done).

Min-of-rounds, ~250 ms per cell, warm. 100K strided-stream cells move ±20 % from placement alone;
believe effects well outside that band and take at least two rounds per side.

---

## 3. What is still slow — measured (`neighbours_probe.cs`, pinned, NumPy 2.4.2)

### 3.1 The fancy-index shapes still on the delegate route (section E)

The kernel route covers exactly ONE index array over a C-contiguous source. Everything adjacent is
still `PrepareIndexGetters`'s delegate-per-element walk (two delegate calls + `GetOffset` per
element for two index arrays, a materialised offset array, a second pass) — the largest loss left
in the library:

| case (f64) | 100K NumSharp | 100K NumPy | NPY/NS 100K | NPY/NS 1M |
|------------|--------------:|-----------:|------------:|----------:|
| `m[ri, ci]` — two index arrays | 2147 µs | 191 µs | **0.09** | 0.23 |
| `m[:, rk]` — slice + index array (column gather) | 635 µs | 34 µs | **0.05** | 0.34 |
| `m[rk, :2]` — index array + slice | 10.2 µs | 2.6 µs | 0.26 | 0.51 |
| `mT[rk]` — rows of an F-order (non-contiguous) source | 187 µs | 35 µs | 0.18 | 0.69 |
| `a[idx[::2]]` — strided index view | 290 µs | 60 µs | 0.21 | 1.11 |
| `m[ri, ci] = v` | 1810 µs | 203 µs | **0.11** | 0.21 |
| `m[:, rk] = v` | 567 µs | 84 µs | 0.15 | 1.87 |
| `view[::2][idx] = v` — non-contiguous destination | 328 µs | 284 µs | 0.87 | 2.44 |

Already ahead in the same family (no action): boolean masks `a[mask]` 2.2–5.8×, `a[mask] = v`
5.2–5.4×, `a[mask] = scalar` 16–17×, `np.compress` 8–12×, `count_nonzero` 3.3×, `np.take(axis=1)`
4.9–7.5×, `np.put` into a strided view 2.4–3.1×.

### 3.2 `where=` neighbours (section F)

| case | 100K NumSharp | 100K NumPy | NPY/NS 100K | NPY/NS 1M |
|------|--------------:|-----------:|------------:|----------:|
| `add(f32, f32, out=f64, where=m)` on 1-D (buffered cast out) | 48 µs | 146 µs | 3.0 | 2.7 |
| the same on 4-wide strided rows | 429 µs | 140 µs | **0.33** | 0.62 |
| `less(a, b, out=, where=)` | 24 µs | 49 µs | 2.0 | 2.1 |
| `copyto(o, a, where=m)` | 37 µs | 32 µs | 0.86 | 1.08 |
| `np.where(m, a, b)` | 31 µs | 67 µs | 2.2 | 3.9 |
| `add(a, b, where=m)` (no out, allocates) | 18 µs | 49 µs | 2.7 | 3.8 |

Only the cast-`out` masked narrow rows lose: the buffered (windowed cast) route is excluded from the
block kernels by the `BUFFER` flag, so it is back on the per-row masked driver.

### 3.3 The Tier 2 claims, remeasured (section G)

| claim in the status list | measured, pinned | verdict |
|--------------------------|------------------|---------|
| "buffered-cast unary routes: `sqrt(int32)` 0.71×" | `sqrt(i32, out=f64)` **1.03×** (76 vs 78 µs), allocating `sqrt(i32)` 1.75× / 3.7×, `exp(i32)` 1.3× / 1.6×, `negative(i32, out=f64)` 2.0×, `add(i32, f64)` 2.2× / 5.1× | parity or ahead — the 0.71× was an unpinned number; the ONE buffered loss left is the cast-out masked narrow rows above |
| "`sum(f32, dtype=f64)` 0.49×" | **1.04× / 1.00×** (75.5 vs 78.2 µs) — but our own same-dtype `sum(f32)` is **3.1 µs** (8× NumPy) | parity with NumPy; a 24× self-relative opportunity (the widening flat reduction is a converting scalar loop) |
| "narrow inner-axis reductions, unmeasured" | trailing axis `sum(x, axis=1)`: w3 1.12× / 1.26×, w4 1.10× / 1.27×, w8 1.19×, w16 1.31×, w64 1.76×; `max` 1.5–3.9×; `mean` w3/w4/w8 **0.95× / 0.95× / 1.02×** | trailing axis is fine (mean at parity) — the gap is the LEADING axis with few rows: `sum(xt, axis=0)` on (3, N) **0.52×** (61 vs 32 µs at 100K), (4, N) **0.58×**, (8, N) 0.91×, (64, N) 0.79×; at 1M 1.16× / 1.18× / **0.83×** / 1.21× / 1.49× |
| "iterator fixed cost on tiny ops, ~80 ns" | n = 1 … 1000: `add(out)` 0.2 vs 0.4 µs, `add` allocating 0.2–0.4 vs 0.7–1.4, `sqrt(out)` 0.1–0.7 vs 0.3–3.2, `sum` 0.2 vs 3.7–4.2, `less` 0.2–0.3 vs 0.7–1.1 | **2–23× ahead of NumPy**; the ~80 ns trims would not move a single ratio — skip for parity |

Two things to know when reading section G: NumPy's allocating cells at 1M (e.g. `sqrt(f64)` 3566 µs
vs our 548) are dominated by its allocator page-faulting an 8 MB result every call, which our pool
avoids — the honest kernel comparison is the 100K cell (`sqrt(f64)` 1.01×) or an `out=` cell. And
our leading-axis kernel runs at a flat ~0.6–0.75 ns per element whether the operand is L2-resident
or not — which says per-row / per-block overhead, not memory traffic.

---

## 4. The ranked plan

Ordered by measured loss × breadth, with the concrete design. Effort: S = a session-hour, M = a
session, L = more.

### L1 — one index array + slices → the take/put kernels (`m[:, rk]`, `m[rk, :2]`) — **M**

**Today:** `FetchIndices(object[])` (`NDArray.Indexing.Selection.Getter.cs`) validates the tuple,
then `TryBuildMultiAdvancedGrid` expands the slices into one FULL integer index array per source
axis (an `arange` grid broadcast to the result shape) and hands the whole grid to the multi-index
delegate path — for `m[:, rk]` that is `side × len(rk)` pairs of delegate calls. The setter does
the same in `SetIndices(object[], values)`.

**NumPy's rule makes this a `take`:** with exactly one advanced index the result keeps every slice
axis in place and puts the index array's dims where the indexed axis was — precisely
`np.take(a, idx, axis=k)`'s shape `a.shape[:k] + idx.shape + a.shape[k+1:]`, for a 1-D or N-D index.

**Design:**
1. In `FetchIndices(object[])`, before `TryBuildMultiAdvancedGrid`: if the tuple is one integer
   index array (an `NDArray` of an integer dtype, or `int[]`/`long[]`) plus only `Slice`s, no
   newaxis, no ellipsis left after expansion, no 0-d bool → `k` = the array's position.
2. If every slice is `Slice.All` → return `np.take(this, idx, axis: k)` (the source is C-contiguous
   in the common case; `np.take` already routes a non-contiguous source through
   `ascontiguousarray`, which is O(source) — acceptable until L3 lands, then replace it with the
   strided-slab gather).
3. Otherwise → build the sliced VIEW first (`this[slices with Slice.All at k]`), then gather along
   axis `k` of the view: with L3's strided-slab take kernel this is one call; without it, fall back
   to today's grid path so nothing regresses.
4. Setter: `m[:, rk] = v` needs a **put along an axis**, which the put kernels lack: mirror the take
   kernel's `(outerSize, maxItem, innerSize)` factorisation in a `PutAxisKernel` — for each outer
   plane, for each `j`: copy the `innerSize`-byte slab of `values[outer, j]` to
   `dst[outer, idx[j]]` (the RAISE pre-scan runs once on the index array, as in
   `TryScatterKernel<T>`). Value broadcasting stays as today (materialise to the selection shape,
   or one slab with a wrapping cursor).
5. Keep the verbatim errors: a bad index is `IndexError "index {v} is out of bounds for axis {k}
   with size {d}"` (note: axis `k`, not 0); a value mismatch is the existing `shape mismatch`
   text. Pin with `Indexing.FancyParity.MatrixTests` + new cases in `Indexing.KernelRoute.Tests`.

**Expected:** `m[:, rk]` 635 → ~75 µs (our `take(axis=1)` measured 73.6) ≈ 0.05 → **0.45×**
NumPy at 100K and ~3× at 1M; `m[rk, :2]` and the setter similar. NumPy's column gather at 100K is
34 µs because it walks a 2-D strided copy per index — matching it needs L3, not L1.

### L2 — two or more index arrays → an offset kernel + the flat gather (`m[ri, ci]`) — **M**

**Today:** `FetchIndices<T>` broadcasts the index arrays (`np.broadcast_arrays`, fine), then for
every element calls one delegate per index array and `srcShape.GetOffset` — 2147 µs at 100K, the
worst ratio measured on this line (0.09×).

**Design:** replace the per-element loop with ONE kernel that turns `k` index arrays into a flat
int64 element-offset array, then run the existing flat gather over the source buffer with those
offsets:
- `ComputeOffsets(void** idxPtrs, bool[] idx32, long* dims, long* elemStrides, long count, long baseOffset, long* offsets)`:
  per axis, `v = idx[i]; if (v < 0) v += dim; if ((ulong)v >= (ulong)dim) → fail(axis, i)`;
  `off += v * elemStride` — SIMD-able on int64 lanes with `Vector<long>` (compare, add, multiply
  by a broadcast stride), scalar fallback; the first failing (axis, position) reproduces
  `PrepareIndexGetters`'s message verbatim.
- Gather: the flat kernel with `maxItem = source buffer element count` over `Storage.Address`
  (offsets are absolute in elements; `baseOffset = Shape.offset` folded in), width = element
  size; sub-shaped (`m[ri, ci]` on a 3-D `m`) uses the general take kernel with a slab.
- Because offsets carry the source's element strides, this route serves ANY source layout (the
  `mT[ri, ci]` case too), and the setter is the same kernel + the put kernel after the scan.
- The index arrays are already `.flat`/contiguous after broadcasting; a broadcast index array
  (stride 0 from `broadcast_arrays`) must be materialised or handled with a stride argument —
  check what `np.broadcast_arrays` returns before assuming contiguity.

**Expected:** 2147 → ~120 µs at 100K (one pass of ~0.4 ns/element for the offsets + the 40 µs
gather) ≈ 0.09 → **1.5×**; the setter the same.

### L3 — a strided-slab gather (non-contiguous source rows, `mT[rk]`) — **S–M**

**Today:** `FetchIndicesNDNonLinear` — an odometer per sub-array element through the view's strides
(187 µs for 100K elements; NumPy 35).

**Design:** a variant of the general take kernel that takes `(slabCount, srcElemStrideBytes)` and
copies each selected sub-array element by element with the typed MOV (the flat kernel's copy) from
`src + idx * slabStrideBytes + e * srcElemStrideBytes` into the contiguous result — one kernel call
for the whole gather, the odometer folded into two strides (a 2-D sub-array needs a second
stride/count pair; stop at 2-D and keep the odometer route beyond). This is what makes L1 step 3
a single call and what `np.take` on a non-contiguous source should use instead of
`ascontiguousarray` (which copies the WHOLE source to gather a few rows).

### L4 — a strided index view → copy it, then the kernel (`a[idx[::2]]`) — **S (trivial)**

**Today:** `NormalizeIndexArray` keeps a non-contiguous 1-D index array as is, so
`FancyIndexKernels.TryGetIndexPointer` declines and the delegate route runs (290 vs 60 µs).

**Design:** in `NormalizeIndexArray` (or just before the kernel gate), `if
(!idxs.Shape.IsContiguous) idxs = np.ascontiguousarray(idxs)` — a ~10 µs copy at 100K that pays
for itself 20 times over. Same in the setter. Pin: `Get1D_StridedIndexView_*` already exists and
keeps passing; add a timing assertion nowhere (timing is the probe's job).

### L5 — leading-axis reductions with few rows (`sum(axis=0)` on (3…8, N)) — **M, root cause unknown**

**Today:** 0.52× / 0.58× at 100K for 3 / 4 rows, 0.83× at 1M for 8 rows. The kernel is
`AxisReductionLeadingTyped` (`DirectILKernelGenerator.Reduction.Axis.Simd.cs`; the widening twin
in `.Reduction.Axis.Widening.cs` documents the 8192-column block scheme: init the output block to
identity, then stream each input row into it). At ~0.6–0.75 ns per element REGARDLESS of cache
residency, it is not bandwidth-bound — suspect the per-block prologue and the per-row loop
overhead dominating when there are only 3 rows per block, or a scalar path being taken for the
short-row shape.

**First step, before any change:** a floor probe — a hand-written `Vector256<double>` loop that
adds 3 rows of 33 333 doubles into an output (the same shape) — next to the kernel's number. If
the floor is ~10 µs (it should be: 100K elements read once from L2), the gap is the kernel's
structure and the fix is a "few-row" shape that keeps the accumulator in registers per column
group and walks the rows innermost. Gate: `benchmark/layout/reduce_layout_bench` (pinned A/B) and
the `reduce` FuzzMatrix tier — summation ORDER must not change for float (bit parity: the tier
compares bytes).

### L6 — the cast-`out` `where=` on narrow rows — **S**

**Today:** `ExecuteBinaryUfuncInto` sets `BUFFERED | GROWINNER | DELAY_BUFALLOC` when
`target.typecode != resultType`; `Is2DMaskedElementwiseShape` rejects `BUFFER`, so the per-row
masked driver runs: 429 vs 140 µs at 100K (0.33×).

**Design (two passes, both already fast):** when `outNeedsCast && where != null` and the unbuffered
shape would qualify for the block kernel, compute into a fresh loop-dtype temp with the masked
block kernel (masked-off temp slots hold garbage, never read), then `np.copyto(out, temp, where:
where, casting: "unsafe")` — the masked cast kernel (`Cast.Masked.cs`) — which writes only the
mask-true slots, so masked-off `out` slots keep their prior contents exactly as today. Expected
~48 + ~30 µs ≈ 80 µs at 100K (**1.75×**) from 0.33×. The fused alternative (convert-on-store inside
the masked kernel) is cleaner but is IL work in `InnerLoop2D.Masked.cs`; do it only if the two-pass
number disappoints. The unmasked cast-out narrow-row case has the same structure and the same
two-pass answer (the unmasked block kernel + a plain cast copy).

### L7 — a fused widening flat sum (`sum(f32, dtype=f64)`, `sum(i32, dtype=f64)`) — **M, parity risk**

**Today:** parity with NumPy (75 vs 78 µs) but 24× slower than our `sum(f32)` (3.1 µs):
`ExecuteElementReduction<TResult>` with `inputType != accumType` runs the contiguous IL reduction
"scalar loop with type conversion" (`Reduction.Axis.cs` header). The axis kernels already have the
widen-in-register form (`VCVTPS2PD` / `VPMOVSX` in `.Reduction.Axis.Widening.cs`).

**Design:** a flat widening SIMD reduction — widen 4 lanes, accumulate in the wide type — **but
reproduce NumPy's summation order**: a buffered NumPy reduction is pairwise WITHIN each 8192-element
buffer window and sequential ACROSS windows, which is not the whole-array pairwise the same-dtype
kernel does. Before touching the order, check whether the oracle's `reduce` tier
(`test/oracle/gen_oracle.py reduce`) has `dtype=` widening cells; if it does not, add them first
so the change is gated. Integer accumulators are order-free (modular), so `sum(i32, dtype=f64)` —
1.87× today — is the safe first half.

### L8 — `copyto(where=)` run scan — **S**

0.86× at 100K, 1.08× at 1M. Check whether the masked cast kernel (`Cast.Masked.cs`) finds runs
byte-by-byte; if so, lift the movemask + trailing-zero run walk from `InvokeInner` into it.

### Skip (measured, not worth a session)

- Iterator fixed-cost trims on tiny ops: 2–23× ahead of NumPy at n ≤ 1000.
- Buffered-cast unary routes: parity or ahead (`sqrt(i32, out=f64)` 1.03×).
- Trailing-axis narrow-row reductions: ahead (`sum` 1.1–1.8×, `max` 1.5–3.9×); `mean` at 0.95–1.0×
  is the sum + a divide pass — only worth fusing if L5's kernel work touches the same file anyway.
- Copy-class ops at w = 2/3 on the unmasked block kernel (1.1–1.3×): a per-width half-vector
  store specialization, documented in `NDITER_2D_BLOCK_KERNEL.md` §11.6.

---

## 5. Traps — each cost a measurement round

1. **The masked ufunc routes call the PACKED-key `ExecuteElementWise` overload.** A gate added only
   to the string-key overload is never reached; "before ≈ after" on a route you just built is the
   tell. (`NDIter.Execution.Custom.cs` has two overloads; wire both.)
2. **`/` on an integer NDArray is NumPy true division.** `(ar / 64 % 2) == 0` is a sparse mask (one
   true element per 128), not 64-runs. Build masks with `np.floor_divide`; a NumSharp/NumPy pair
   whose timings differ in SHAPE across run lengths is the tell.
3. **Never A/B the live tree.** A chain of `dotnet run` benches compiles the tree as it is when each
   run starts; editing during the chain hands later runs a half-edited tree (one run failed to
   build, another silently measured a partial state). Both sides in detached worktrees.
4. **Pin.** Unpinned runs on this hybrid host swing 2–3× uniformly and do not reproduce
   (`NDITER_PERF_DISCOVERY.md` §3 trap 11). Pin BOTH sides, never concurrently.
5. **NumPy's 1M allocating cells flatter NumSharp.** NumPy page-faults a fresh 8 MB result per
   call; our pool does not. Compare kernels on `out=` cells or at 100K.
6. **`dotnet run` runfile cache.** A rebuilt project behind an unchanged script file can serve the
   OLD assembly; give the probe a fresh filename (or one changed comment line) per rebuild.
7. **`np.take` on a non-contiguous source copies the whole source** (`ascontiguousarray`) before
   gathering — fine for `take` on its own, wrong as a building block for L1 on a sliced view;
   that is what L3 is for.
8. **A contiguous slice re-seats `Storage.Address` and keeps `Shape.offset == 0`; a strided view
   keeps the base address and a non-zero offset.** `Storage.Address + Shape.offset × itemsize` is
   right for both; the old fancy route was NOT offset-buggy (probed) — do not re-claim it.
9. **Unary "SIMD" with a promoting input is a buffered cast + the wide kernel** (`bufferedPromoting`
   in `DefaultEngine.UfuncOut.cs`), not a fused convert; it measured at parity, so leave it.

---

## 6. Gates to run after touching any of this

    # the kernel-route + where= pins (fast)
    dotnet test test/NumSharp.Tests --no-build -c Release -f net10.0 --filter "ClassName~Indexing_KernelRoute_Tests|FullyQualifiedName~NumSharp.Tests.Selection|FullyQualifiedName~NumSharp.Tests.Indexing|Name~Where"
    # the whole suite with the CI filter (~2 min)
    dotnet test test/NumSharp.Tests --no-build -c Release -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
    # the NumPy differential gate — every ufunc out=/where= layout, index get/set oracle, reductions
    dotnet test test/NumSharp.Tests.Oracle --no-build -c Release -f net10.0 --filter "TestCategory=FuzzMatrix"
    # both targets
    dotnet build src/NumSharp.Core/NumSharp.Core.csproj -c Release -f net8.0

Five `AuditV2` tests and `Slice2x2Mul_AssignmentChangesOriginal` fail before and after; they are
`[OpenBugs]` and excluded by the CI filter. For a kernel change, ALSO run a self-consistency probe
in the style of the masked kernel's (every strided result byte-equal to the same call on
contiguous copies AND to an independent composition — `np.where` for masks, `np.take` for
gathers) before the differential gate; the oracle corpus does not enumerate widths 1–40.

---

## 7. File map for orientation

| area | files |
|------|-------|
| fancy get/set dispatch | `Selection/NDArray.Indexing.cs` (indexers), `Selection/NDArray.Indexing.Selection.Getter.cs` (`FetchIndices(object[])`, `FetchIndices<T>`, `FetchIndicesND`, `FetchIndicesNDNonLinear`, `TryBuildMultiAdvancedGrid`), `Selection/NDArray.Indexing.Selection.Setter.cs` (`SetIndices(object[], …)`, `SetIndices<T>`, `TryScatterKernel<T>`, `SetIndicesND`, `SetIndicesNDNonLinear`), `Selection/NDArray.Indexing.Selection.cs` (`PrepareIndexGetters`, `NormalizeIndexArray`) |
| fancy kernels | `Selection/FancyIndexKernels.cs`, `Backends/Kernels/Direct/DirectILKernelGenerator.{GatherFlat,Take,Put}.cs`, `Indexing/np.{take,put}.cs` |
| `where=` routes | `Backends/Default/Math/DefaultEngine.UfuncOut.cs` (`ExecuteBinaryUfuncInto`, `ExecuteUnaryUfuncInto`, the comparison twin), `Backends/Iterators/NDIter.Execution.cs` (`ForEach`, `InvokeInner`), `NDIter.Execution.Custom.cs` (the 2-D gates and drivers), `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D{,.Masked}.cs` |
| reductions named above | `Backends/Default/Math/DefaultEngine.ReductionOp.cs` (`ExecuteElementReduction<T>`), `Backends/Default/Math/Reduction/Default.Reduction.Add.cs`, `Backends/Kernels/Direct/DirectILKernelGenerator.Reduction.Axis{,.Simd,.Widening}.cs` |
| tests | `test/NumSharp.Tests/Selection/Indexing.KernelRoute.Tests.cs` (the Tier 1 pins), `Indexing.FancyParity.MatrixTests.cs`, `Indexing.CombinatorialParity.MatrixTests.cs`, `IndexingProbeMatrix.Tests.cs`, `test/NumSharp.Tests.Oracle/Fuzz/IndexOracleTests.cs` |
| probes | `benchmark/nditer/probes/{fancy_where,neighbours,narrow,fixed_cost,ab_ops,angles}_probe.cs` + `numpy_twins.py` |
| documents | `docs/NDITER_PERF_DISCOVERY.md` (cost model, traps, §6.6 results, §7 levers), `docs/NDITER_2D_BLOCK_KERNEL.md` (the block kernel, §12 masked form), this file |
