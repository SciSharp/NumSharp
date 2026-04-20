# NDIter but with IL generation — kerneling your NDArray

NumPy's `nditer` is the unsung workhorse of NumPy. Every ufunc, every reduction, every broadcasted operation is scheduled by `nditer` under the covers. It decides which axes to iterate, which to coalesce, whether to buffer, how to walk strided memory — then it hands those decisions to a typed C inner loop generated from C++ templates.

NumSharp has to reach the same destination from the other direction. We have no templates. What we have is `System.Reflection.Emit.DynamicMethod` and a JIT that eagerly autovectorizes tight loops. This page explains how NumSharp's port of `nditer` (`NpyIter`) works, why we diverge from NumPy in a few places, and — most importantly — how `NpyIter.Execution.cs` glues the iterator to `ILKernelGenerator` so a single call like `ExecuteBinary(Add)` cashes out to the same kind of native SIMD loop that NumPy's C++ emits at compile time, but generated at your first call and cached forever after.

Read this page end-to-end if you're writing a new `np.*` function, porting a ufunc, or trying to squeeze more performance out of an existing operation.

## Table of Contents

- [Overview](#overview)
- [What NpyIter Is](#what-npyiter-is)
- [Divergences from NumPy](#divergences-from-numpy)
- [Iterator State](#iterator-state)
- [Construction](#construction)
- [Coalescing, Reordering, and Flipping](#coalescing-reordering-and-flipping)
- [Iteration Mechanics](#iteration-mechanics)
- [Buffering](#buffering)
- [Buffered Reduction: The Double Loop](#buffered-reduction-the-double-loop)
- [Kernel Integration Layer](#kernel-integration-layer)
  - [Quick reference](#quick-reference)
  - [Decision tree](#decision-tree)
  - [Measured behavior](#measured-behavior)
  - [Cache state — two lifetimes to know about](#cache-state--two-lifetimes-to-know-about)
  - [Layer 1 — Canonical Inner-Loop API](#layer-1--canonical-inner-loop-api)
  - [Layer 2 — Struct-Generic Dispatch](#layer-2--struct-generic-dispatch)
  - [Layer 3 — Typed ufunc Dispatch](#layer-3--typed-ufunc-dispatch)
  - [Custom Operations (Tier A / B / C)](#custom-operations-tier-a--b--c)
    - [Tier A — Raw IL](#tier-a--raw-il)
    - [Tier B — Templated Inner Loop](#tier-b--templated-inner-loop)
    - [Tier C — Expression DSL](#tier-c--expression-dsl)
      - [Node catalog](#node-catalog)
      - [Operator overloads](#operator-overloads)
      - [Call — invoke any .NET method](#call--invoke-any-net-method)
      - [Type discipline](#type-discipline)
      - [SIMD coverage rules](#simd-coverage-rules)
      - [Caching and auto-keys](#caching-and-auto-keys)
      - [Memory model and lifetime](#memory-model-and-lifetime)
      - [Validation and errors](#validation-and-errors)
      - [Gotchas](#gotchas)
      - [Debugging compiled kernels](#debugging-compiled-kernels)
      - [When to use Tier C](#when-to-use-tier-c)
- [Path Detection](#path-detection)
- [Worked Examples](#worked-examples)
- [Performance](#performance)
  - [JIT Warmup Caveat](#jit-warmup-caveat)
  - [Implementation Notes](#implementation-notes)
  - [When Does Each Layer Pay Off?](#when-does-each-layer-pay-off)
  - [Allocations](#allocations)
- [Known Bugs and Workarounds](#known-bugs-and-workarounds)
- [Summary](#summary)

---

## Overview

### What Is An Iterator?

An array is just a pointer plus a shape plus strides. Iterating "through" it means producing, one element (or chunk of elements) at a time, the byte offset into the buffer. For a contiguous row-major 3×4 array this is trivial — walk from 0 to 11 with stride 1. For a transposed view, a sliced view, a broadcasted view, or two arrays with mismatched strides, it is not.

`NpyIter` takes that tangle and produces a single linear schedule of pointer advances. Once you have it, you can write one loop — `do { kernel(dataptrs, strides, count); } while (iternext); ` — and it runs correctly for every memory layout NumSharp supports.

### Why Build Our Own?

NumPy's `nditer` is C99 with templates mixed in through macro expansion. We can't take it verbatim. At the same time we want every one of its capabilities: coalescing, reordering, negative-stride flipping, ALLOCATE, COPY_IF_OVERLAP, buffered casting, buffered reduction with the double-loop trick, C/F/K ordering, per-operand flags, op_axes with explicit reduction encoding. These are features users rely on without realizing it — `np.sum(a, axis=0)` quietly benefits from four of them.

NumSharp implements all of it in managed code with `NativeMemory.AllocZeroed` for unmanaged state and `ILKernelGenerator` for the typed inner loops. The bridge that wires them together is `NpyIter.Execution.cs`, which this page centers on.

---

## What NpyIter Is

`NpyIter` is a `ref partial struct` living in `NumSharp.Backends.Iteration`. Concretely:

```
NpyIterRef (ref partial struct)                ← public handle (~3000 lines across 2 partials)
    ├── _state: NpyIterState*                  ← heap-allocated unmanaged state
    ├── _operands: NDArray[]                   ← kept alive by GC root
    └── _cachedIterNext: NpyIterNextFunc?      ← memoized iterate-advance delegate

NpyIterState (unmanaged struct)                ← ~30 fields, all dynamically sized
    ├── Scalars: NDim, NOp, IterSize, IterIndex, ItFlags, ...
    ├── Dim arrays (size = NDim): Shape*, Coords*, Strides*, Perm*
    ├── Op arrays (size = NOp):   DataPtrs*, ResetDataPtrs*, BufStrides*,
    │                              InnerStrides*, BaseOffsets*, OpDTypes*, ...
    └── Reduction arrays:         ReduceOuterStrides*, ReduceOuterPtrs*,
                                  ArrayWritebackPtrs*, CoreSize, CorePos, ...
```

The public struct is cheap to pass around; the heavy state lives behind one pointer so we can allocate it exactly once, on the heap, sized to the problem. Dispose frees it.

### The Files

| File | What lives there |
|------|------------------|
| `NpyIter.cs` | Construction, iteration wrappers, debug dump, `Copy`, `Dispose` (~3000 lines) |
| `NpyIter.State.cs` | `NpyIterState` definition, allocation, `Advance`, `Reset`, `GotoIterIndex`, `BufferedReduceAdvance` |
| `NpyIter.Execution.cs` | **Kernel integration layer** — `ForEach`, `ExecuteGeneric`, `Execute{Binary,Unary,Reduction,Comparison,Scan,Copy}` (~600 lines) |
| `NpyIterFlags.cs` | `NpyIterFlags`, `NpyIterOpFlags`, `NpyIterGlobalFlags`, `NpyIterPerOpFlags`, casting/order enums |
| `NpyIterCoalescing.cs` | `CoalesceAxes`, `ReorderAxesForCoalescing`, `FlipNegativeStrides` |
| `NpyIterCasting.cs` | Safe/same-kind/unsafe cast rules, `ConvertValue`, `FindCommonDtype` |
| `NpyIterBufferManager.cs` | Aligned buffer allocation, copy-in/copy-out, `GROWINNER`, `BUF_REUSABLE` |
| `NpyIterKernels.cs` | Abstract kernel interfaces (`INpyIterKernel`, path selectors) |
| `NpyAxisIter.cs`, `NpyAxisIter.State.cs` | Specialized axis-reduction iterator (simpler API, fewer features) |
| `NpyLogicalReductionKernels.cs` | Generic boolean/numeric axis-reduction kernel structs |

---

## Divergences from NumPy

NumPy's `nditer` has two hard-coded limits that NumSharp drops:

| Limit | NumPy | NumSharp |
|-------|-------|----------|
| `NPY_MAXDIMS` | 64 | unlimited (dynamic alloc, soft limit ≈ 300k from `stackalloc`) |
| `NPY_MAXARGS` | 64 | unlimited (dynamic alloc) |

NumPy uses fixed arrays inside `NpyIter_InternalIterator`. NumSharp allocates everything via `NativeMemory.AllocZeroed` sized to the actual `(ndim, nop)` the caller passes. The trade is marginally more setup cost in exchange for no artificial ceilings and no wasted memory on a 2-operand 1-D iter.

Other deliberate differences:

- **Flag bit layout.** NumSharp reserves low bits 0-7 for legacy compat (`SourceBroadcast`, `SourceContiguous`, `DestinationContiguous`). NumPy-parity flags (`IDENTPERM`, `HASINDEX`, `REDUCE`, ...) sit at bits 8-15. Transfer flags pack into the top byte at shift 24. Semantics match NumPy; positions do not.
- **Element strides everywhere internally.** NumPy stores byte strides in `NAD_STRIDES`. NumSharp stores element strides in `state.Strides` and multiplies by `ElementSizes[op]` at use. This matches NumSharp's `Shape.strides` convention.
- **No Python object support.** `REFS_OK`, garbage collection hooks, and `NpyIter_GetBufferNeedsAPI` are no-ops. All cast routines are written assuming the data is plain unmanaged bytes.
- **Int64 indexing.** Every iteration counter is `long`. Arrays > 2 GB are first-class, unlike NumPy which still uses `npy_intp` (platform-dependent).

---

## Iterator State

A couple of fields deserve a closer look because every later section refers to them.

### Shape, Coords, Strides

```csharp
public long* Shape;             // [NDim]        — post-coalesce dimension sizes
public long* Coords;            // [NDim]        — current position, 0..Shape[d]
public long* Strides;           // [NOp * NDim]  — element stride per (op, axis)
public sbyte* Perm;             // [NDim]        — Perm[internal] = original_axis
                                //                 negative means axis was flipped
```

After coalescing, `NDim` can shrink. `StridesNDim` captures the stride allocation width so `GetStride(axis, op) = Strides[op * StridesNDim + axis]` still works.

`Perm[internal_axis] = original_axis` records how internal axes relate to the axes the caller passed in. If `FlipNegativeStrides` rewrote an axis, `Perm[d] = -1 - original_axis` encodes the flip. `GetMultiIndex` uses Perm to translate internal coords back into caller-space.

### DataPtrs vs ResetDataPtrs vs BaseOffsets

```csharp
public long* ResetDataPtrs;     // base pointer per operand; start of iteration
public long* BaseOffsets;       // byte accumulator from FlipNegativeStrides
public long* DataPtrs;          // live pointer; moves every Advance()
```

`Reset()` copies `ResetDataPtrs` into `DataPtrs`. When the iterator flips an axis it walks the data pointer to the end-of-axis first (since we'll iterate backwards in original memory, forwards in flipped-coord space) and records the byte delta in `BaseOffsets`. `ResetBasePointers(newPtrs)` lets the caller swap the array out while keeping the iteration schedule: new reset = new base + stored offset.

### Buffering Fields

```csharp
public long  BufferSize;        // elements per operand buffer (default 8192)
public long  BufIterEnd;        // how far into the buffer we're iterating
public long* Buffers;           // aligned-64 buffer pointer per operand (0 = no buffer)
public long* BufStrides;        // inner-loop stride per operand in bytes
                                //   == ElementSizes[op] for buffered operands
```

When buffering is active, an operand's `DataPtrs[op]` points into `Buffers[op]`, not into the original NDArray. The kernel sees a contiguous buffer at the buffer dtype; `NpyIterBufferManager` handles the strided copy-in and copy-out.

### Reduction Fields (double-loop)

```csharp
public int  OuterDim;           // which internal axis is the reduce axis
public long CoreSize;           // elements per output slot (inner-loop length)
public long CorePos;            // position within core, 0..CoreSize
public long ReduceOuterSize;    // number of output slots in current buffer
public long ReducePos;          // position within outer loop

public long* ReduceOuterStrides;  // stride per op, advances to next output slot
public long* ReduceOuterPtrs;     // saved pointer at start of current output slot
public long* ArrayWritebackPtrs;  // array-space pointer for flushing output buffer
```

These only come into play when the iterator has both `BUFFER` and `REDUCE` flags. They're explained in detail in [Buffered Reduction: The Double Loop](#buffered-reduction-the-double-loop).

---

## Construction

Creating an iterator looks like this:

```csharp
using var iter = NpyIterRef.MultiNew(
    nop: 3,
    op: new[] { a, b, out },
    flags: NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_SAFE_CASTING,
    opFlags: new[] {
        NpyIterPerOpFlags.READONLY,
        NpyIterPerOpFlags.READONLY,
        NpyIterPerOpFlags.WRITEONLY },
    opDtypes: new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double });
```

Behind the scenes:

```
1. Pre-check WRITEMASKED/ARRAYMASK pairing    (state-free validation)
2. Resolve broadcast shape                    (ResolveReturnShape; respects op_axes)
3. Allocate ALLOCATE operands with result dtype
4. state.AllocateDimArrays(ndim, nop)         (one big NativeMemory.AllocZeroed)
5. Set MaskOp from ARRAYMASK flag
6. Find common dtype if COMMON_DTYPE
7. For each operand:
      - SetOpSrcDType (array dtype)
      - SetOpDType (buffer dtype; equals array dtype when not casting)
      - Translate NpyIterPerOpFlags → NpyIterOpFlags
      - Mark CAST if dtypes differ
      - Compute strides (respecting op_axes or broadcast)
      - Set data pointer = arr.Address + offset * elemSize
      - Mark SourceBroadcast if any dim has stride 0 with Shape > 1
8. Validate casting requires BUFFERED flag
9. NpyIterCasting.ValidateCasts(ref state, casting)
10. Apply op_axes reduction flags (detects implicit + explicit reduction axes)
11. FlipNegativeStrides (K-order only; skipped for C/F/A)
12. If NDim > 1: ReorderAxesForCoalescing → CoalesceAxes
    (but only when MULTI_INDEX and C_INDEX/F_INDEX are both off)
13. Set EXLOOP, GROWINNER, HASMULTIINDEX, HASINDEX flags per request
14. InitializeFlatIndex() if HASINDEX
15. UpdateInnerStrides()  (cache inner stride per op for fast access)
16. UpdateContiguityFlags()  (sets CONTIGUOUS if every operand is contiguous)
17. If BUFFERED: allocate buffers, prime them with CopyToBuffer
18. If BUFFERED + REDUCE: SetupBufferedReduction (double-loop)
19. If IterSize <= 1: set ONEITERATION
```

The result is a state machine ready to produce pointers.

### The Flag Families

There are four mostly-disjoint flag enums. A quick reference:

**`NpyIterGlobalFlags` — passed at construction, affect the whole iterator.**

| Flag | Meaning |
|------|---------|
| `C_INDEX`, `F_INDEX` | Track a flat index in C or F order |
| `MULTI_INDEX` | Track per-dim coords (needed for `GetMultiIndex`) |
| `EXTERNAL_LOOP` | Caller handles inner dim — iterator returns inner-dim-sized chunks |
| `COMMON_DTYPE` | Find common dtype across all operands and cast to it |
| `REDUCE_OK` | Allow reduction operands (needed for axis reductions) |
| `BUFFERED` | Enable operand buffering (required with cross-type casting) |
| `GROWINNER` | Make inner loop as large as possible within buffer |
| `DELAY_BUFALLOC` | Defer buffer alloc until first `Reset` |
| `DONT_NEGATE_STRIDES` | Suppress `FlipNegativeStrides` |
| `COPY_IF_OVERLAP` | Copy operand if it overlaps another in memory |
| `RANGED` | Iterator covers a sub-range |

**`NpyIterPerOpFlags` — passed per operand, affect just that operand.**

| Flag | Meaning |
|------|---------|
| `READONLY`, `WRITEONLY`, `READWRITE` | Direction |
| `COPY`, `UPDATEIFCOPY` | Force copy / update on dealloc |
| `ALLOCATE` | `op[i]` is null — iterator allocates using `opDtypes[i]` |
| `CONTIG` | Require contiguous view (may force buffering) |
| `NO_BROADCAST` | Error if this operand would need to broadcast |
| `WRITEMASKED`, `ARRAYMASK` | Writemask pair for masked writes |

**`NpyIterFlags` — internal state, set/cleared during iteration.** (`IDENTPERM`, `NEGPERM`, `HASINDEX`, `BUFFER`, `REDUCE`, `ONEITERATION`, etc.) These flow from construction decisions.

**`NpyIterOpFlags` — per-operand internal state.** (`READ`, `WRITE`, `CAST`, `REDUCE`, `VIRTUAL`, `WRITEMASKED`, `BUF_REUSABLE`, `CONTIG`.)

---

## Coalescing, Reordering, and Flipping

The single biggest performance lever the iterator has is **reducing NDim**. A 3-D contiguous array should iterate in one flat loop, not in three nested ones.

### Coalescing Rule

Two adjacent axes `d` and `d+1` can merge if, for **every** operand:

```
stride[op][d] * shape[d] == stride[op][d+1]
```

...or either axis is size 1 with stride 0 (broadcast pass-through). When that holds, the pair is collapsed: the new shape is `shape[d] * shape[d+1]`, the new stride is `stride[op][d]` (the inner one).

A contiguous 2×3×4 float32 array has strides `[12, 4, 1]` in elements. The coalescing check succeeds at both boundaries, and `CoalesceAxes` reduces NDim from 3 to 1 with shape 24 and stride 1. One flat SIMD loop, exactly.

### Reordering

Coalescing only works if adjacent axes are *already* stride-ordered. `ReorderAxesForCoalescing` sorts axes by minimum absolute stride (smallest innermost) when the requested order allows it:

```
C-order:  last axis innermost (no reorder — identity perm)
F-order:  first axis innermost (reverse axes)
K-order:  smallest stride innermost (insertion sort by stride)
A-order:  behaves like K-order
```

For K-order on a non-contiguous broadcast array, stride-based sorting produces the wrong iteration order, so the iterator falls back to C-order. This guard rail lives in the construction logic around `effectiveOrder`.

### Negative-Stride Flipping

`FlipNegativeStrides` only runs under K-order (not C/F/A — those are "forced orders" that preserve logical iteration direction). For each axis where *all* operands have zero or negative strides, the iterator:

1. Negates the stride.
2. Accumulates `(shape[d] - 1) * old_stride * elem_size` into `BaseOffsets[op]`.
3. Marks the axis flipped via `Perm[d] = (sbyte)(-1 - Perm[d])`.

The effect: a reversed slice still iterates contiguous memory in ascending order, which the SIMD kernels can chew on. Later, `GetMultiIndex` decodes the flip so the caller sees original coordinates.

### Interaction with MULTI_INDEX and HASINDEX

If `MULTI_INDEX` is set we **reorder but don't coalesce** — coalescing would lose the mapping from internal to original axes. Same for `C_INDEX`/`F_INDEX`, which need original axis structure to compute the flat index.

---

## Iteration Mechanics

Three flavors of `iternext` exist, and `GetIterNext()` returns the right one for the current flag set:

| Flavor | Picked when | Behavior |
|--------|-------------|----------|
| `SingleIterationNext` | `ONEITERATION` | One shot, done |
| `ExternalLoopNext` | `EXLOOP` | Advance *outer* coords only; inner dim is the caller's problem |
| `StandardNext` | otherwise | Full ripple-carry advance, one element at a time |

`state.Advance()` is the ripple-carry primitive. For each axis from innermost to outermost:

```
for axis in (NDim-1 ... 0):
    coord[axis]++
    if coord[axis] < shape[axis]:
        dataptrs[op] += stride[op][axis] * elem_size[op]   for every op
        return
    // carry: reset this axis
    coord[axis] = 0
    dataptrs[op] -= stride[op][axis] * (shape[axis] - 1) * elem_size[op]
// fell through: iteration complete
```

Straightforward, but note the rewind on carry: when axis 2 wraps, we subtract `stride*(shape-1)*size` so the pointer lands back at the axis-2 start, then axis 1 will add one stride. The net effect is identical to `dataptr = base + sum(coord[d] * stride[d][op]) * size`, but computed incrementally.

### GetInnerLoopSizePtr()

Ideally the inner loop processes many elements per `iternext` call. The iterator exposes this via:

```csharp
long* size = iter.GetInnerLoopSizePtr();
```

- When `BUFFER` is set: returns `&state.BufIterEnd` (whatever fit in the current buffer fill).
- Otherwise: returns `&state.Shape[NDim-1]` (the innermost dimension size).

With `EXTERNAL_LOOP` set and the array coalesced to 1-D, one `iternext` call returns the entire array size — a single kernel invocation processes everything.

---

## Buffering

Buffering solves two problems:

1. **Casting.** If the caller wants to see doubles but the NDArray is int32, the iterator copies into a double buffer, runs the kernel against the buffer, writes back on dispose.
2. **Non-contiguous + SIMD.** If the operand is strided (sliced, transposed), copying to a contiguous buffer lets a SIMD kernel work efficiently.

`NpyIterBufferManager.AllocateBuffers` allocates 64-byte-aligned blocks (AVX-512-friendly) per operand that needs buffering. Default buffer size is 8192 elements; this can be tuned per call.

```
strided array (stride=5, size=24)       aligned 64-byte buffer (size ≤ 8192)
┌─────┬─────┬─────┬─────┐               ┌──┬──┬──┬──┬──┬──┬──┐
│ a[0]│  ?  │  ?  │  ?  │  CopyToBuffer │a0│a5│a10│...         │
│  ?  │  ?  │  ?  │ a[5]│    ────────▶  └──┴──┴──┴──┴──┴──┴──┘
│  ?  │  ?  │a[10]│  ?  │                   ^
│     ...            │                      DataPtrs[op] points here
└─────────────────────┘                     BufStrides[op] = sizeof(T)
```

Once the buffer is filled, `DataPtrs[op]` moves into the buffer and every inner-loop kernel treats it as a flat contiguous array. When iteration advances past `BufIterEnd`, `NpyIterBufferManager.CopyFromBuffer` writes output back into the original array (respecting original strides) and `CopyToBuffer` refills input buffers for the next chunk.

### GROWINNER

When `GROWINNER` is set the iterator tries to inline as many outer axes as will fit in the buffer into the inner loop. On a 5×6 contiguous array with buffer size 8192, the entire 30-element array fits in one pass; the reported inner loop size becomes 30 instead of 6. More work per kernel call, less loop overhead.

### BUF_REUSABLE

For reductions, the same input block may be read multiple times (e.g. `mean` when accumulator type differs). The `BUF_REUSABLE` flag tells the iterator "the buffer contents are still valid, skip the copy." `CopyToBufferIfNeeded` honors it.

---

## Buffered Reduction: The Double Loop

When you do `np.sum(a, axis=0)` on a 2-D array, the output has one fewer axis than the input. The iterator must visit every input but accumulate into a fixed output position while the reduction axis is scanned. The efficient way to do this with buffering is NumPy's **double loop**:

```
CoreSize    = length of reduce axis              ("how many inputs per output")
ReduceOuterSize = other-axes length fitted into buffer   ("how many output slots")

For each buffer fill:
    for outer in 0..ReduceOuterSize:              ← advance ReduceOuterPtrs by ReduceOuterStrides
        for core in 0..CoreSize:                  ← advance DataPtrs by BufStrides
            kernel(dataptrs, bufstrides, 1)       ← accumulate into output
        // reset inner, move outer pointer to next output slot
```

The trick: reduce operands have `BufStrides[op] = 0`, so inside the core loop their pointer stays pinned. The kernel keeps adding into the same output slot until the reduce axis is exhausted; the outer loop then moves to the next output slot.

`NpyIterState.BufferedReduceAdvance()` returns:
- `1` — more elements in current buffer (inner or outer)
- `0` — buffer exhausted, caller must refill
- `-1` — iteration complete, caller must flush

The bridge's `BufferedReduce` method drives this explicitly.

### IsFirstVisit

Reduction kernels must initialize the output before accumulating. `iter.IsFirstVisit(op)` returns `true` only when every reduction-axis coordinate is zero *and* `CorePos == 0` in buffered mode. Kernels check this once at each output slot to emit identity-write semantics:

```csharp
if (iter.IsFirstVisit(reduceOp)) *(double*)ptrs[reduceOp] = 0.0;
*(double*)ptrs[reduceOp] += *(double*)ptrs[inputOp];
```

---

## Kernel Integration Layer

Everything up to this point describes `NpyIter`'s scheduling machinery. What `NpyIter.Execution.cs` adds is the connection between that schedule and the SIMD kernels `ILKernelGenerator` emits.

The layer is a partial declaration of `NpyIterRef` that exposes **seven entry points** arranged along an ergonomics-vs-control axis. Pick the one that matches your use case; they all share the same compiled-kernel cache and all run through the same `ForEach` driver at the bottom.

```
           ergonomics                                                     control
              ▲                                                              ▲
              │                                                              │
  Layer 3     │  ExecuteBinary / Unary / Reduction / Comparison / Scan      │  90% case
              │  "one call, NumPy-style — one line per op"                   │
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier C      │  ExecuteExpression(NpyExpr)                                  │  compose
              │  "build a tree with operators; no IL in caller"              │  with DSL
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier C+Call │  NpyExpr.Call(Math.X / Func / MethodInfo, args)              │  inject any
              │  "invoke arbitrary managed method per element"               │  BCL / user op
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier B      │  ExecuteElementWiseBinary(scalarBody, vectorBody)            │  hand-tune
              │  "write per-element IL; factory wraps the unroll shell"      │  the vector body
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Tier A      │  ExecuteRawIL(emit, key, aux)                                │  emit
              │  "emit the whole inner-loop body including ret"              │  everything
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Layer 2     │  ExecuteGeneric<TKernel> / ExecuteReducing<TKernel, TAccum>  │  struct-
              │  "zero-alloc; JIT specializes per struct; early-exit reduce" │  generic
  ──────────  │  ─────────────────────────────────────────────────────────  │  ──────────
  Layer 1     │  ForEach(NpyInnerLoopFunc kernel, void* aux)                 │  delegate,
              │  "closest to NumPy's C API; closures welcome"                │  anything goes
              │                                                              │
              ▼                                                              ▼
           NpyIter state (Shape, Strides, DataPtrs, Buffers, ...)
                                  │
                                  ▼
              ILKernelGenerator (DynamicMethod + V128/V256/V512)
```

### Quick reference

| # | Entry point | When to reach for it | Per-call cost |
|---|-------------|----------------------|---------------|
| 1 | `ExecuteBinary` / `Unary` / `Reduction` / `Comparison` / `Scan` | The op is a standard NumPy ufunc. 90% of cases. | Cache hit after first call |
| 2 | `ExecuteExpression(NpyExpr)` | Compose a fused ufunc from DSL nodes (`Add`, `Sqrt`, `Where`, `Exp`, comparisons, `Min`/`Max`/`Clamp`, …). SIMD when dtypes align. | Cache hit after first compile |
| 3 | `ExecuteExpression(NpyExpr.Call(...))` | DSL doesn't expose the op you want (`Math.BitIncrement`, custom activation, reflected plugin method). | +5-10 ns / element for non-static delegates |
| 4 | `ExecuteElementWiseBinary` / `Unary` / `Ternary` / `ExecuteElementWise` (array form) | You want SIMD + 4× unroll for a fused or non-standard op; the DSL doesn't compose to it, but the loop shape is still element-wise. Hand-write the scalar + vector body. | Cache hit after first compile |
| 5 | `ExecuteRawIL(emit, key, aux)` | Non-rectangular loop: gather/scatter, cross-element deps, branch-on-auxdata. You emit every opcode. | Cache hit after first compile |
| 6 | `ExecuteGeneric<TKernel>` / `ExecuteReducing<TKernel, TAccum>` | Custom kernel in struct form. Zero allocation; JIT specializes. **Only** path with early-exit reductions. | No delegate indirection |
| 7 | `ForEach(NpyInnerLoopFunc)` | Exploratory; one-off fused kernels; anything a closure makes natural. | Delegate allocation per call |

### Decision tree

```
Is the op a standard NumPy ufunc already in ExecuteBinary/Unary/Reduction?
  yes → Layer 3 (baked). Fastest, zero work. Done.
  no ↓

Can I express it as a tree of DSL nodes (Add, Sqrt, Where, Exp, …)?
  yes → Tier C. Fused, SIMD-or-scalar automatic, no IL.
  no ↓

Is the missing piece a BCL method (Math.X, user activation, reflected plugin)?
  yes → Tier C + Call. Scalar-only but fused. Done.
  no ↓

Do I need V256/V512 intrinsics the DSL doesn't wrap (Fma, Shuffle, Gather, …)?
  yes → Tier B. Hand-write the vector body; factory wraps the shell.
  no ↓

Is the loop shape non-rectangular (gather/scatter, cross-element deps)?
  yes → Tier A. Emit the whole inner-loop IL yourself.
  no ↓

Do I need an early-exit reduction (Any / All / find-first)?
  yes → Layer 2 ExecuteReducing. Returns false from the kernel to bail out.
  no ↓

Just exploring or writing a one-off?
       → Layer 1 ForEach. Delegate per call; flexible.
```

### Measured behavior

Benchmarked on 1M-element arrays, post-warmup, via the showcase script in this doc's `/demos/` sibling (not checked in — recreate with the snippet in each tier's section below):

| Technique | Operation | Time / run | Notes |
|-----------|-----------|-----------:|-------|
| Layer 3 | `a + b` (f32) | 0.58 ms | baked, 4×-unrolled V256, cache hit |
| Tier B | `2a + 3b` hand V256 (f32) | 0.61 ms | within ~7% of baked — same shell |
| Layer 2 reduction | `AnyNonZero` early-exit (hit @ 500) | 0.001 ms | returns `false` from kernel, bridge bails |
| Tier A | `abs(a - b)` raw IL (i32) | 1.27 ms | scalar loop, JIT autovectorizes post tier-1 |
| Call | `GELU` via captured lambda (f64) | 8.08 ms | `Math.Tanh` dominates |
| Tier C | stable sigmoid via `Where` (f64) | 13.6 ms | 3 × `Math.Exp` per element |

Layer 1 and Layer 2 element-wise kernels have a tier-0 JIT caveat: when run from a dynamic host (ephemeral script, `dotnet_run`, first-call cold start) they can look 30-50× slower than production code. Post-tier-1 promotion (~100 hot-loop iterations) brings them within 2-3 ms for hypot on 1M f32. See [JIT Warmup Caveat](#jit-warmup-caveat).

### Cache state — two lifetimes to know about

The full integration layer shares two process-lifetime caches. Inspect them via the internal hooks (need `[InternalsVisibleTo]` or the `AssemblyName=NumSharp.DotNetRunScript` script directive):

```csharp
int kernels = ILKernelGenerator.InnerLoopCachedCount;   // compiled DynamicMethods
int slots   = DelegateSlots.RegisteredCount;            // registered delegates + targets

ILKernelGenerator.ClearInnerLoopCache();                // test-only
DelegateSlots.Clear();                                   // test-only — pair with above!
```

After running the full showcase (Layer 3 + Tiers A-C + Call across 130 warmup+timed iterations), typical counts are:

```
ILKernelGenerator.InnerLoopCachedCount = 4     ← one per unique cache key across all tiers
DelegateSlots.RegisteredCount          = 131   ← one per Call(lambda) construction
```

The `131` is the documented gotcha from the [Memory model and lifetime](#memory-model-and-lifetime) section — every `NpyExpr.Call(lambda, …)` constructor call re-registers the delegate, even if the kernel is reused via an explicit `cacheKey`. Users expecting steady-state slot growth should register delegates once at startup (`static readonly Func<…>`), see the [registration-once pattern](#memory-model-and-lifetime).

### Layer 1 — Canonical Inner-Loop API

This is the NumPy-in-C pattern. You hand the iterator a function pointer (a delegate in C#), and it runs the canonical loop:

```csharp
public void ForEach(NpyInnerLoopFunc kernel, void* auxdata = null);

public unsafe delegate void NpyInnerLoopFunc(
    void** dataptrs, long* strides, long count, void* auxdata);
```

One call per *inner loop*, not per element. The iterator decides what "inner loop" means:

| Scenario | Call count | Count per call |
|----------|-----------|----------------|
| Fully coalesced + contiguous, with `EXTERNAL_LOOP` | 1 | `IterSize` |
| Non-coalesced with `EXTERNAL_LOOP` | outer product | `Shape[NDim-1]` |
| Buffered | `ceil(IterSize / BufferSize)` | `BufIterEnd` |
| Neither `EXTERNAL_LOOP` nor `BUFFERED` | `IterSize` | 1 |

The strides passed to the kernel are always in **bytes** — the bridge converts from element strides for the non-buffered path. This matches NumPy's convention and makes the kernel body identical whether or not the iterator is buffering.

**Performance note.** Post-tier-1 the JIT autovectorizes both byte-pointer and typed-pointer loops into Vector256. To get there faster and to keep the fast path as simple as possible, branch on stride at the top and drop to typed pointers:

```csharp
using var iter = NpyIterRef.MultiNew(3, new[] { a, b, c },
    NpyIterGlobalFlags.EXTERNAL_LOOP,
    NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_NO_CASTING,
    new[] { NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.WRITEONLY });

iter.ForEach((ptrs, strides, count, _) => {
    // Fast branch: contiguous, element stride == sizeof(float).
    // The JIT autovectorizes this to Vector256 sqrt.
    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
        float* a = (float*)ptrs[0], b = (float*)ptrs[1], c = (float*)ptrs[2];
        for (long i = 0; i < count; i++)
            c[i] = MathF.Sqrt(a[i] * a[i] + b[i] * b[i]);
        return;
    }
    // Slow branch: strided / broadcast. Correct but scalar.
    long sA = strides[0], sB = strides[1], sC = strides[2];
    byte* pA = (byte*)ptrs[0]; byte* pB = (byte*)ptrs[1]; byte* pC = (byte*)ptrs[2];
    for (long i = 0; i < count; i++) {
        float av = *(float*)(pA + i * sA);
        float bv = *(float*)(pB + i * sB);
        *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
    }
});
```

Use this when you're writing a one-off operation that doesn't fit the standard ufunc shape, or when you want to fuse several operations into a single pass to avoid temporaries.

### Layer 2 — Struct-Generic Dispatch

Delegates have an indirect call. For hot inner loops, that hurts. Layer 2 trades a delegate for a struct type parameter:

```csharp
public interface INpyInnerLoop
{
    void Execute(void** dataptrs, long* strides, long count);
}

public interface INpyReducingInnerLoop<TAccum> where TAccum : unmanaged
{
    bool Execute(void** dataptrs, long* strides, long count, ref TAccum accumulator);
}

public void ExecuteGeneric<TKernel>(TKernel kernel)
    where TKernel : struct, INpyInnerLoop;

public TAccum ExecuteReducing<TKernel, TAccum>(TKernel kernel, TAccum init)
    where TKernel : struct, INpyReducingInnerLoop<TAccum>
    where TAccum : unmanaged;
```

Because `TKernel` is constrained to `struct`, the JIT specializes one copy of `ExecuteGeneric` per struct type at codegen time and inlines `kernel.Execute(...)` at the call site. No vtable, no delegate, no boxing. It's the closest managed C# gets to C++ templates.

The bridge splits `ExecuteGeneric` internally so the single-inner-loop case (the common case: coalesced contig + `EXTERNAL_LOOP`, `ONEITERATION`, or buffered-fits-in-one-fill) goes through `ExecuteGenericSingle` — a tiny `[AggressiveInlining]` method with one `kernel.Execute` call and no `do/while`. That's what lets the JIT autovectorize the kernel's body. The multi-loop path keeps the canonical `do { kernel.Execute(...); } while (iternext); ` driver.

```csharp
readonly unsafe struct HypotKernel : INpyInnerLoop
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Execute(void** p, long* s, long n)
    {
        // Fast branch — typed pointers so the JIT autovectorizes.
        if (s[0] == 4 && s[1] == 4 && s[2] == 4) {
            float* a = (float*)p[0], b = (float*)p[1], c = (float*)p[2];
            for (long i = 0; i < n; i++)
                c[i] = MathF.Sqrt(a[i] * a[i] + b[i] * b[i]);
            return;
        }
        // Slow branch — any stride, scalar.
        long sA = s[0], sB = s[1], sC = s[2];
        byte* pA = (byte*)p[0]; byte* pB = (byte*)p[1]; byte* pC = (byte*)p[2];
        for (long i = 0; i < n; i++) {
            float av = *(float*)(pA + i * sA);
            float bv = *(float*)(pB + i * sB);
            *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
        }
    }
}

iter.ExecuteGeneric(default(HypotKernel));  // zero-alloc, inlined
```

For early-exit reductions, the kernel returns `false` to abort:

```csharp
readonly unsafe struct AnyNonZero : INpyReducingInnerLoop<bool>
{
    public bool Execute(void** p, long* s, long n, ref bool acc)
    {
        long st = s[0]; byte* pt = (byte*)p[0];
        for (long i = 0; i < n; i++)
            if (*(int*)(pt + i * st) != 0) { acc = true; return false; }  // stop
        return true;
    }
}

bool found = iter.ExecuteReducing<AnyNonZero, bool>(default, false);
```

On a 1M-element array with a non-zero near the start, this returns after one kernel call.

### Layer 3 — Typed ufunc Dispatch

Layer 3 is what you reach for 90% of the time: "run a standard ufunc, pick the best kernel." The bridge inspects the iterator's post-coalesce stride picture, constructs the right cache key for `ILKernelGenerator`, materializes a SIMD kernel, and invokes it.

```csharp
public void ExecuteBinary(BinaryOp op);       // [in0, in1, out]
public void ExecuteUnary(UnaryOp op);          // [in, out]
public void ExecuteComparison(ComparisonOp);   // [in0, in1, bool out]
public TResult ExecuteReduction<TResult>(ReductionOp op);  // [in] → T
public void ExecuteScan(ReductionOp op);       // [in, out]
public void ExecuteCopy();                     // [src, dst]
public void BufferedReduce<K, T>(K kernel);    // explicit BUFFER+REDUCE double-loop
```

Under the hood each helper does four things:

1. **Validate.** Throw if operand count or flags are wrong.
2. **Detect path.** Scan operand strides, pick `SimdFull` / `SimdScalarRight` / `SimdScalarLeft` / `SimdChunk` / `General`.
3. **Prepare args.** `stackalloc` one stride array per operand, fill with element strides, grab `_state->Shape` and data pointers.
4. **Invoke.** `ILKernelGenerator.GetMixedTypeKernel(key)(...)` — cache hit returns the cached delegate, cache miss emits IL and caches.

For buffered paths, `ExecuteBinary` dispatches to `RunBufferedBinary`, which runs the kernel against `_state->Buffers` using `BufStrides` (which are always element-sized for the buffer dtype) rather than the original-array strides. This sidesteps a known issue with the in-state pointer-advance, discussed in [Known Bugs](#known-bugs-and-workarounds).

### Custom Operations (Tier A / B / C)

The enum-driven `Execute{Binary,Unary,Reduction,...}` methods cover every primitive NumPy ufunc, but they're a closed set. The moment you want `a*b + c` as one pass, or `sqrt(a² + b²)` without materializing intermediates, or a brand-new op that isn't in `BinaryOp`/`UnaryOp`, you're outside the baked catalog.

The Custom Operations extension solves this by letting the bridge **IL-generate a kernel specialized for any user-defined computation** while preserving Layer 3's 4×-unrolled SIMD shell. Three tiers trade control for convenience:

```
              ┌─────────────────── You provide ────────────────────┐
 Tier A       │  the entire inner-loop IL body                     │  Maximum control
 Tier B       │  per-element scalar + (optional) vector IL body    │  Shared unroll shell
 Tier C       │  an expression tree (NpyExpr)                      │  No IL required
              └────────────────────────────────────────────────────┘
                           │
                           ▼
               ILKernelGenerator.CompileInnerLoop  (new partial)
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Contig SIMD path      Scalar strided path
          (4× unroll + V256      (per-element, stride-aware
           + 1-vec remainder      pointer walk)
           + scalar tail)
                 └─────────┬─────────┘
                           ▼
                  NpyInnerLoopFunc delegate (cached)
                           │
                           ▼
                 NpyIterRef.ForEach → do { kernel(...); } while (iternext)
```

All three tiers produce the same delegate shape (`NpyInnerLoopFunc`) and funnel through `ForEach`. The factory emits a runtime contig check at the top of the kernel: if every operand's byte stride equals its element size, take the SIMD path; otherwise fall into the scalar-strided loop. Cache keys are user-supplied strings; Tier C derives a structural signature automatically if you don't provide one.

| Method on `NpyIterRef` | Tier | What you supply |
|------------------------|------|------------------|
| `ExecuteRawIL(emit, key, aux)` | A | `Action<ILGenerator>` — the entire method, including `ret` |
| `ExecuteElementWise(operandTypes, scalarBody, vectorBody, key)` | B | Two `Action<ILGenerator>` — per-element scalar and vector |
| `ExecuteElementWiseUnary/Binary/Ternary(...)` | B | Typed convenience overloads |
| `ExecuteExpression(expr, inputTypes, outputType, key?)` | C | An `NpyExpr` tree |

#### Tier A — Raw IL

You emit everything. Arguments are the canonical inner-loop shape: `arg0 = void** dataptrs`, `arg1 = long* byteStrides`, `arg2 = long count`, `arg3 = void* auxdata`. Your body must emit its own `ret`. Cached by the string key you pass — same key returns the same compiled delegate.

```csharp
iter.ExecuteRawIL(il =>
{
    // Pull out pointers and strides once.
    var p0 = il.DeclareLocal(typeof(byte*));
    var p1 = il.DeclareLocal(typeof(byte*));
    var p2 = il.DeclareLocal(typeof(byte*));
    // ... load dataptrs[0..2], strides[0..2] ...

    // for (i = 0; i < count; i++) *p2 = *p0 + *p1
    var i = il.DeclareLocal(typeof(long));
    il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);

    var top = il.DefineLabel(); var end = il.DefineLabel();
    il.MarkLabel(top);
    il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, end);
    // compute p2[i*s2] = p0[i*s0] + p1[i*s1]
    // ...
    il.Emit(OpCodes.Ldloc, i); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, i);
    il.Emit(OpCodes.Br, top);
    il.MarkLabel(end);
    il.Emit(OpCodes.Ret);
}, cacheKey: "my_int32_add");
```

Use when: you need a loop shape the templated shell can't express (gather, scatter, cross-element dependencies, non-rectangular write patterns).

#### Tier B — Templated Inner Loop

Supply only the per-element work; the factory wraps it in the standard 4×-unrolled SIMD + 1-vector remainder + scalar tail + scalar-strided fallback. The two `Action<ILGenerator>` callbacks are stack-based:

- **`scalarBody`** — on entry, stack holds N input scalars in order (operand 0 deepest, operand N-1 on top); on exit, stack must hold one value of the output dtype.
- **`vectorBody`** — same contract but with `Vector{W}<T>` values. Optional — pass `null` for scalar-only. If non-null **and** all operand dtypes are identical **and** the type is SIMD-capable, the factory emits the fast path.

```csharp
// out = a*b + 1 on 16 float32s, fused in one pass.
iter.ExecuteElementWiseBinary(
    NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
    scalarBody: il =>
    {
        // Stack: [a, b] -> [a*b + 1]
        il.Emit(OpCodes.Mul);
        il.Emit(OpCodes.Ldc_R4, 1.0f);
        il.Emit(OpCodes.Add);
    },
    vectorBody: il =>
    {
        // Stack: [va, vb] -> [va*vb + 1]
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);
        il.Emit(OpCodes.Ldc_R4, 1.0f);
        ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
    },
    cacheKey: "fma_f32_c1");
```

The `ILKernelGenerator.Emit*` helpers (`EmitVectorOperation`, `EmitVectorCreate`, `EmitVectorLoad`, `EmitVectorStore`, `EmitScalarOperation`, `EmitConvertTo`, `EmitLoadIndirect`, `EmitStoreIndirect`, `EmitUnaryScalarOperation`, `EmitUnaryVectorOperation`) are exposed as `internal` so you can compose primitives without reinventing IL emission. The same helpers power the baked `ExecuteBinary`/`ExecuteUnary` kernels.

Convenience overloads exist for common arities:

```csharp
iter.ExecuteElementWiseUnary(inType, outType, scalarBody, vectorBody, key);
iter.ExecuteElementWiseBinary(lhs, rhs, outType, scalarBody, vectorBody, key);
iter.ExecuteElementWiseTernary(a, b, c, outType, scalarBody, vectorBody, key);
```

For arity > 3 or variable operand counts, use the array form `ExecuteElementWise(NPTypeCode[], ...)`.

**When SIMD is skipped.** The factory emits the vector path only when `CanSimdAllOperands(operandTypes)` returns true — every operand's dtype must be identical and SIMD-capable (i.e. not `Boolean`, `Char`, or `Decimal`). If either condition fails, only the scalar path is emitted. Mixed-type ufuncs (e.g. `int32 + float32 → float32`) use the scalar path with the user's `EmitConvertTo` inside the body.

**Contig runtime check.** The kernel's first act is to compare each operand's stride with its element size. If any differ, control jumps to the scalar-strided loop — inner-axis iteration that advances pointers by their declared byte strides. This means a single kernel handles both contiguous and sliced inputs without recompiling.

Use when: you want SIMD + 4× unrolling for a fused or non-standard op but don't want to hand-roll the whole loop.

#### Tier C — Expression DSL

The expression DSL lets you compose ops with C# operator syntax, and `Compile()` emits the IL for you. No `ILGenerator` exposure in your code.

```csharp
// out = sqrt(a² + b²)
var expr = NpyExpr.Sqrt(NpyExpr.Square(NpyExpr.Input(0)) +
                        NpyExpr.Square(NpyExpr.Input(1)));

iter.ExecuteExpression(expr,
    inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
    outputType: NPTypeCode.Single);
```

##### Node catalog

**Leaves.**

| Factory | Semantics | NumPy |
|---------|-----------|-------|
| `NpyExpr.Input(i)` | Reference operand `i` (0-based input index). Auto-converts to output dtype on load. | — |
| `NpyExpr.Const(value)` | Literal — `int / long / float / double` overloads. Emitted at the output dtype. | — |

**Binary arithmetic.**

| Factory | Operator | SIMD | NumPy equivalent | Notes |
|---------|----------|:----:|------------------|-------|
| `Add(a,b)` | `a + b` | ✓ | `np.add` | |
| `Subtract(a,b)` | `a - b` | ✓ | `np.subtract` | |
| `Multiply(a,b)` | `a * b` | ✓ | `np.multiply` | |
| `Divide(a,b)` | `a / b` | ✓ | `np.divide` | True-division for floats; integer division for ints. |
| `Mod(a,b)` | `a % b` | — | `np.mod` | Floored modulo — result sign follows divisor (like Python `%`), unlike C# `%` which truncates toward zero. |
| `Power(a,b)` | — | — | `np.power` | Routed through `Math.Pow(double, double)`; integer operands are promoted to double and the result converted back. |
| `FloorDivide(a,b)` | — | — | `np.floor_divide` | Floor toward negative infinity. For signed int operands, correctly returns `-4` (not `-3`) for `-10 // 3`. |
| `ATan2(y,x)` | — | — | `np.arctan2` | Four-quadrant arctan via `Math.Atan2`. |

**Binary bitwise.** Integer types only; floating-point operands are a compile-time IL emission error.

| Factory | Operator | SIMD | NumPy equivalent |
|---------|----------|:----:|------------------|
| `BitwiseAnd(a,b)` | `a & b` | ✓ | `np.bitwise_and` |
| `BitwiseOr(a,b)` | `a \| b` | ✓ | `np.bitwise_or` |
| `BitwiseXor(a,b)` | `a ^ b` | ✓ | `np.bitwise_xor` |

**Scalar-branchy combinators** (scalar path only).

| Factory | Semantics | NumPy equivalent |
|---------|-----------|------------------|
| `Min(a,b)` | Delegates to `Math.Min` — NaN-propagating per IEEE 754. | `np.minimum` (**not** `np.fmin`) |
| `Max(a,b)` | Delegates to `Math.Max` — NaN-propagating per IEEE 754. | `np.maximum` (**not** `np.fmax`) |
| `Clamp(x,lo,hi)` | `Min(Max(x,lo),hi)` — sugar, shares the compiled kernel structure with the underlying pair. | `np.clip` |
| `Where(cond,a,b)` | Branchy ternary select: if `cond != 0` return `a` else `b`. `cond` is evaluated in the output dtype, so floats, integers, and decimals all work uniformly. | `np.where` (with eager eval of both branches) |

> `Where`'s branches are **both emitted** into the kernel but only the taken one runs per element — the `brfalse` branches past the untaken side. If one side is much more expensive (e.g. `Exp`), the cost is only paid on elements where it's selected, making `Where` a real optimization over `cond * a + (1-cond) * b` for expensive alternatives.

**Unary — arithmetic.**

| Factory | Operator | SIMD | NumPy equivalent |
|---------|----------|:----:|------------------|
| `Negate(x)` | unary `-x` | ✓ | `np.negative` |
| `Abs(x)` | — | ✓ | `np.abs` / `np.absolute` |
| `Sqrt(x)` | — | ✓ | `np.sqrt` |
| `Square(x)` | — | ✓ | `np.square` |
| `Reciprocal(x)` | — | ✓ | `np.reciprocal` |
| `Cbrt(x)` | — | — | `np.cbrt` |
| `Sign(x)` | — | — | `np.sign` |

**Unary — exp / log.** All route through `Math.Exp / Log / ...` (or `MathF` for `Single`); integer inputs are auto-promoted to double around the call and cast back at the end.

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Exp(x)` | eˣ | — | `np.exp` |
| `Exp2(x)` | 2ˣ | — | `np.exp2` |
| `Expm1(x)` | eˣ − 1 (accurate for small x) | — | `np.expm1` |
| `Log(x)` | ln x | — | `np.log` |
| `Log2(x)` | log₂ x | — | `np.log2` |
| `Log10(x)` | log₁₀ x | — | `np.log10` |
| `Log1p(x)` | ln(1 + x) (accurate for small x) | — | `np.log1p` |

**Unary — trigonometric.**

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Sin(x)`, `Cos(x)`, `Tan(x)` | Standard trig | — | `np.sin / cos / tan` |
| `Sinh(x)`, `Cosh(x)`, `Tanh(x)` | Hyperbolic | — | `np.sinh / cosh / tanh` |
| `ASin(x)`, `ACos(x)`, `ATan(x)` | Inverse | — | `np.arcsin / arccos / arctan` |
| `Deg2Rad(x)` | x · π/180 | ✓ | `np.deg2rad` / `np.radians` |
| `Rad2Deg(x)` | x · 180/π | ✓ | `np.rad2deg` / `np.degrees` |

**Unary — rounding.**

| Factory | Semantics | SIMD | NumPy equivalent |
|---------|-----------|:----:|------------------|
| `Floor(x)` | ⌊x⌋ | ✓ | `np.floor` |
| `Ceil(x)` | ⌈x⌉ | ✓ | `np.ceil` |
| `Round(x)` | Banker's rounding (half-to-even) | — | `np.rint` (matches NumPy's half-to-even default) |
| `Truncate(x)` | Toward zero | — | `np.trunc` |

> `Round` and `Truncate` have a working SIMD path on .NET 9+, but NumSharp's library targets .NET 8 as well, where `Vector256.Round/Truncate` don't exist. NpyExpr gates them to the scalar path unconditionally so the compiled kernel works on both frameworks. Other contiguous rounding ops autovectorize after tier-1 JIT promotion.

**Unary — bitwise / logical / predicates.**

| Factory | Operator | SIMD | NumPy equivalent | Notes |
|---------|----------|:----:|------------------|-------|
| `BitwiseNot(x)` | `~x` | ✓ | `np.invert` / `np.bitwise_not` | Integer types only. |
| `LogicalNot(x)` | `!x` | — | `np.logical_not` | Returns 1 if `x == 0` else 0. Routes through `EmitComparisonOperation(Equal, outType)` — correct for all dtypes including Int64, Single, Double, Decimal (see [Gotchas](#gotchas)). |
| `IsNaN(x)` | — | — | `np.isnan` | Returns 0/1 at output dtype. For integer types: always 0. |
| `IsFinite(x)` | — | — | `np.isfinite` | Returns 0/1 at output dtype. For integer types: always 1. |
| `IsInf(x)` | — | — | `np.isinf` | Returns 0/1 at output dtype. For integer types: always 0. |

**Comparisons** (produce numeric 0 or 1 at output dtype; scalar path only).

| Factory | Semantics | NumPy equivalent |
|---------|-----------|------------------|
| `Equal(a,b)` | `a == b` | `np.equal` |
| `NotEqual(a,b)` | `a != b` | `np.not_equal` |
| `Less(a,b)` | `a < b` | `np.less` |
| `LessEqual(a,b)` | `a <= b` | `np.less_equal` |
| `Greater(a,b)` | `a > b` | `np.greater` |
| `GreaterEqual(a,b)` | `a >= b` | `np.greater_equal` |

Unlike NumPy's comparison ufuncs (which return `bool` arrays), Tier C's single-output-dtype model collapses comparisons to `0 or 1` at the output dtype. This composes cleanly with arithmetic — e.g. ReLU becomes `(x > 0) * x`.

NaN semantics match IEEE 754: any comparison involving NaN produces 0 (false). `NaN == NaN → 0`, `NaN < 5 → 0`, `NaN >= 5 → 0`. To test for NaN, use `IsNaN(x)`.

**Call — invoke any .NET method.** The escape hatch for math not in the node catalog. Scalar path only.

| Factory | Semantics |
|---------|-----------|
| `Call<T1…Tn, TR>(Func<T1…Tn, TR> f, NpyExpr a1, …)` | Typed generic overloads for arity 0–4. Accept method groups without cast (`NpyExpr.Call(Math.Sqrt, x)`, `NpyExpr.Call(Math.Pow, x, y)`). |
| `Call(Delegate func, params NpyExpr[] args)` | Catch-all for pre-constructed delegates. Use when the arity exceeds 4 or when the typed overload is ambiguous. |
| `Call(MethodInfo staticMethod, params NpyExpr[] args)` | Invoke a reflection-obtained static method. |
| `Call(MethodInfo instanceMethod, object target, params NpyExpr[] args)` | Invoke a reflection-obtained instance method against `target`. |

See [Call — invoke any .NET method](#call--invoke-any-net-method) below for dispatch paths, auto-conversion rules, supported signatures, performance envelope, and overload-disambiguation guidance.

##### Operator overloads

An expression tree reads like ordinary C#:

```csharp
// (a + b) * c + 1
var linear = (NpyExpr.Input(0) + NpyExpr.Input(1)) * NpyExpr.Input(2) + NpyExpr.Const(1.0f);

// ReLU via comparison × input
var relu = NpyExpr.Greater(NpyExpr.Input(0), NpyExpr.Const(0.0f)) * NpyExpr.Input(0);

// Clamp with no named method call
var clamped = NpyExpr.Min(NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Const(0f)), NpyExpr.Const(1f));
```

Overloads: `+ - * /` (arithmetic), `%` (NumPy mod), `& | ^` (bitwise), unary `-` (negate), `~` (bitwise not), `!` (logical not). No overloads for `<`, `>`, `==`, `!=` (those need to return `bool` in C#, which would collide with `object.Equals` and similar) — use the factory methods (`Less`, `Greater`, `Equal`, `NotEqual`, `LessEqual`, `GreaterEqual`) for comparisons.

##### Call — invoke any .NET method

The DSL's built-in catalog covers most element-wise math. `Call` is the escape hatch for everything else: user-defined activations, BCL helpers without a dedicated node (e.g. `Math.BitDecrement`, `Math.CopySign`), plugin methods discovered through reflection, captured-state business logic. It trades SIMD for universality.

**One node, four factory shapes, three dispatch paths.** All four factories construct the same `CallNode`; the node inspects its input and picks the cheapest dispatch at construction:

```
                      ┌─────────────────────────┐
  NpyExpr.Call(...)   │       CallNode          │
  ─────────────▶      │  _kind ∈ {              │
                      │    StaticMethod,        │ ← call <methodinfo>
                      │    BoundTarget,         │ ← load target, callvirt
                      │    Delegate             │ ← load delegate, Invoke
                      │  }                      │
                      └─────────────────────────┘
```

**Path A — static methods (zero indirection).**

```csharp
// Func<T...> overload: compiler infers delegate signature, no cast needed
// for non-overloaded methods.
NpyExpr.Call(Math.Sqrt,   NpyExpr.Input(0));
NpyExpr.Call(Math.Pow,    NpyExpr.Input(0), NpyExpr.Input(1));
NpyExpr.Call(MathF.Tanh,  NpyExpr.Input(0));

// MethodInfo overload: useful when reflecting.
var mi = typeof(Math).GetMethod("BitIncrement", new[] { typeof(double) });
NpyExpr.Call(mi, NpyExpr.Input(0));
```

Emit: one `call <methodinfo>` opcode after the arguments are pushed. The JIT may inline the target when it's small and visible. No DelegateSlots entry, no runtime lookup. This is the fast path and is what you get automatically whenever the delegate has no captured state.

**Path B — bound instance methods (one indirection).**

```csharp
class Activations
{
    public double Temperature { get; set; }
    public double Softmax(double x) => Math.Exp(x / Temperature);
}

var inst = new Activations { Temperature = 1.5 };
var mi = typeof(Activations).GetMethod("Softmax");

NpyExpr.Call(mi, inst, NpyExpr.Input(0));
```

Emit: the kernel first loads the target object from a process-wide `DelegateSlots` registry by integer ID, casts it to the method's declaring type, pushes the arguments, then `callvirt <methodinfo>`. Cost is one dictionary lookup (~5 ns) plus a virtual call. The target object's state is live — mutations are visible to subsequent kernel invocations.

**Path C — captured delegates (one indirection).**

```csharp
// Works uniformly for lambdas with captures, instance-method-bound delegates,
// or any pre-constructed Delegate instance.
Func<double, double> swish = x => x / (1.0 + Math.Exp(-x));
NpyExpr.Call(swish, NpyExpr.Input(0));

// Pre-constructed delegate with explicit type (no method-group cast needed here).
Delegate d = swish;
NpyExpr.Call(d, NpyExpr.Input(0));
```

Emit: the kernel loads the delegate from `DelegateSlots`, casts it to its concrete runtime type (e.g. `Func<double, double>`), pushes arguments, then `callvirt Invoke`. Same ~5-10 ns overhead as Path B, plus the `Delegate.Invoke` dispatch stub (single virtual call).

**Auto-conversion at the call boundary.**

The node respects the DSL's single-output-dtype invariant:

```
        ctx.OutputType        param dtype       return dtype      ctx.OutputType
  ┌──────────────────┐  ┌─────────────┐   ┌───────────────┐   ┌──────────────────┐
  │ args evaluated   │─▶│ convert via │──▶│ method runs   │──▶│ convert via      │
  │ in outputType    │  │ EmitConvertTo│   │               │   │ EmitConvertTo    │
  └──────────────────┘  └─────────────┘   └───────────────┘   └──────────────────┘
```

So `NpyExpr.Call(Math.Sqrt, Input(0))` with an `Int32` input and a `Double` output works end-to-end: the int gets loaded, converted to double at `InputNode`, arrives at the call as double (no further conversion needed for a `Double` param), `Math.Sqrt` runs, the double return flows out to the `Double` output slot. Flip the output dtype to `Single` and you'd get an extra `Conv_R4` after the call.

**Overload disambiguation.**

Non-overloaded static methods bind to the typed `Func<...>` overload via method-group conversion — no cast needed:

```csharp
NpyExpr.Call(Math.Sqrt, x);        // ✓ Func<double,double>
NpyExpr.Call(Math.Cbrt, x);        // ✓ same
NpyExpr.Call(MathF.Tanh, x);       // ✓ Func<float,float>
NpyExpr.Call(Math.Pow, x, y);      // ✓ Func<double,double,double>
```

Methods with multiple overloads (same name, different signatures) need a cast to disambiguate which one you want:

```csharp
// ERROR: 'Math.Abs' has 9 overloads.
// NpyExpr.Call(Math.Abs, x);
//                ^^^^^^^^
// CS0121: The call is ambiguous between ...

// Cast to the concrete Func<...> you want:
NpyExpr.Call((Func<double, double>)Math.Abs, x);       // ✓ picks Math.Abs(double)
NpyExpr.Call((Func<float, float>)MathF.Abs,  x);       // ✓ picks MathF.Abs(float)
NpyExpr.Call((Func<long, long>)Math.Abs,     x);       // ✓ picks Math.Abs(long)
```

Alternatively, use the `MethodInfo` overload to pick by signature explicitly:

```csharp
var mi = typeof(Math).GetMethod(nameof(Math.Abs), new[] { typeof(double) });
NpyExpr.Call(mi, x);  // unambiguous — the MethodInfo is already picked
```

**Thread safety.**

`DelegateSlots` registration uses `Interlocked.Increment` for ID generation and `ConcurrentDictionary` for storage, so concurrent `Call` construction from multiple threads is safe. Kernel compilation itself happens under the `ConcurrentDictionary.GetOrAdd` atomicity for the inner-loop cache — one compilation per key, even under contention. Once compiled, kernels are re-entrant (they only read the delegate/target from their immutable slot).

**Performance envelope.**

Per-element cost of the three paths, measured against a built-in DSL op on a post-warmup 1M-element double array:

| Path | Relative to built-in Sqrt | Notes |
|------|--------------------------|-------|
| Static method (Path A) | ~1.5× slower | One managed call per element; JIT may inline small targets |
| Bound instance (Path B) | ~2-3× slower | Dict lookup + castclass + virtual call |
| Captured delegate (Path C) | ~2-4× slower | Same lookup + castclass + `Delegate.Invoke` stub |

These ratios assume the user's method does comparable arithmetic to `Math.Sqrt`. If your target does substantially more work (e.g. three `Math.Exp` calls), the ratio collapses toward 1 — the call overhead becomes negligible compared to the math.

##### Type discipline

Every intermediate value flows through the output dtype: `Input(i)` loads the i-th operand's dtype and auto-converts (via `EmitConvertTo`) to the output dtype; constants are emitted directly in the output dtype. This **single-type intermediate invariant** keeps the DSL simple — you don't need to reason about mixed-type arithmetic inside the tree.

**Concrete example — integer to float promotion.**

```csharp
// Input is int32, output is float64. The DSL handles the promotion automatically.
var a = np.array(new int[] { 1, 4, 9, 16, 25 });
var r = np.empty(new Shape(5), np.float64);

using var iter = NpyIterRef.MultiNew(2, new[] { a, r }, ...);
iter.ExecuteExpression(NpyExpr.Sqrt(NpyExpr.Input(0)),
    inputTypes: new[] { NPTypeCode.Int32 }, outputType: NPTypeCode.Double);
// r = [1.0, 2.0, 3.0, 4.0, 5.0]
```

What the emitted IL does per element: load `int32`, `Conv_R8` (promote to double), call `Math.Sqrt(double)`, store `double`. The conversion is emitted at the `Input` node, not at the `Sqrt` node — all subsequent operations see the output-dtype value.

**SIMD gate.** The vector path is enabled only when **every** input dtype equals the output dtype (so a single `Vector<T>` instantiation covers the whole tree) **and every node in the tree has a SIMD emit**. If any node lacks a SIMD path, the whole compilation falls back to scalar — correctness preserved, but no 4× unroll. For mixed-dtype work you're in the scalar-strided fallback regardless.

##### SIMD coverage rules

A node's `SupportsSimd` determines whether Tier C emits the vector body:

- **Yes:** `Input`, `Const`, the four arithmetic binary ops (`+ - * /`), the three bitwise binary ops (`& | ^`), and the unary ops `Negate`, `Abs`, `Sqrt`, `Floor`, `Ceil`, `Square`, `Reciprocal`, `Deg2Rad`, `Rad2Deg`, `BitwiseNot`.
- **No:** `Mod`, `Power`, `FloorDivide`, `ATan2`, `Min`/`Max`/`Clamp`/`Where`, all comparisons, `Round`, `Truncate` (no net8 SIMD method), all trig (except `Deg2Rad`/`Rad2Deg`), all log/exp, `Sign`, `Cbrt`, `LogicalNot`, predicates (`IsNaN`/`IsFinite`/`IsInf`), `Call` (user methods are always scalar — there is no vectorization path for arbitrary managed calls).

**Predicate / LogicalNot result handling.** Predicates (`IsNaN`/`IsFinite`/`IsInf`) and `LogicalNot` emit an I4 0/1 on the stack, not a value of the output dtype. `UnaryNode` detects these ops and inserts a trailing `EmitConvertTo(Int32, outType)` so the factory's final `Stind` matches. `LogicalNot` in particular routes through `EmitComparisonOperation(Equal, outType)` with an output-dtype zero literal, because the default `ILKernelGenerator` emit path uses `Ldc_I4_0 + Ceq` which is only correct when the value fits in I4 — broken for Int64, Single, Double, Decimal. NpyExpr takes the safer route.

A tree's `SupportsSimd` is true only if **every** node in it does. One unsupported node demotes the whole tree to scalar-only — which is usually still autovectorized by the JIT after tier-1 promotion, just without the 4× unroll.

##### Caching and auto-keys

Pass `cacheKey` to share the compiled delegate across iterators; omit it and the compiler auto-derives one from the tree's structural signature plus input/output dtypes. Actual examples (verified against `NpyExpr.AppendSignature`):

```
NpyExpr:Add(Multiply(In[0],Const[2]),Const[3]):in=Double:out=Double
NpyExpr:Sqrt(Add(Square(In[0]),Square(In[1]))):in=Single,Single:out=Single
NpyExpr:Where(CmpGreater(In[0],Const[0]),In[0],Multiply(Const[0.1],In[0])):in=Double:out=Double
NpyExpr:Min(In[0],In[1]):in=Int32,Int32:out=Int32
NpyExpr:IsNan(In[0]):in=Double:out=Double
NpyExpr:LogicalNot(In[0]):in=Double:out=Double
NpyExpr:BitwiseNot(In[0]):in=Int32:out=Int32
NpyExpr:Mod(In[0],Const[3]):in=Double:out=Double
NpyExpr:Sqrt(In[0]):in=Int32:out=Double           ← int input, double output
NpyExpr:Call[System.Math.Sqrt#100663308@<guid>](In[0]):in=Double:out=Double
NpyExpr:Call[MyApp.Activations.Swish#167772171@<guid>,target#7](In[0]):in=Double:out=Double
```

Enum names appear verbatim (e.g. `Multiply`, not `Mul`; `IsNan`, not `IsNaN` — the enum is spelled `IsNan`).

Two trees with identical structure and types get the same auto-derived key and share a cached kernel. Each node class contributes a distinct signature prefix:

| Node class | Signature fragment |
|------------|--------------------|
| `InputNode` | `In[i]` |
| `ConstNode` | `Const[value]` (integer form if constructed from int/long; decimal form for float/double) |
| `BinaryNode` | `<BinaryOp>(L,R)` (e.g. `Add(...)`, `Mod(...)`, `ATan2(...)`) |
| `UnaryNode` | `<UnaryOp>(C)` (e.g. `Sqrt(...)`, `IsNan(...)`, `BitwiseNot(...)`) |
| `ComparisonNode` | `Cmp<Op>(L,R)` (e.g. `CmpEqual(...)`, `CmpGreater(...)`) |
| `MinMaxNode` | `Min(L,R)` or `Max(L,R)` |
| `WhereNode` | `Where(C,A,B)` |
| `CallNode` | `Call[<declaringType>.<name>#<metadataToken>@<moduleGuid>](args)` — for instance methods, additionally `,target#<slotId>` |

> **Constant value sensitivity.** Two trees that differ only in a constant value (e.g. `x + 1` vs `x + 2`) generate distinct keys — the constant is part of the signature, because it's baked into the emitted IL. If you need many kernels parameterized by a scalar, consider passing the scalar as a second input operand (as a 0-d `NDArray` or a broadcast view) rather than a compile-time constant.
>
> **Integer/float const collision.** `NpyExpr.Const(1)` and `NpyExpr.Const(1.0)` both serialize to `Const[1]` when the `double` value is whole. With the same output dtype they produce identical IL, so sharing a cache entry is correct. If you need to distinguish — say, to force a specific integer vs float constant interpretation — construct both trees separately and supply an explicit `cacheKey`.

##### Memory model and lifetime

Three things live longer than you might expect when you use Tier C. Knowing what they are, where they hide, and how long they stick around is enough to avoid every subtle memory-creep footgun in practice.

**1. Compiled kernels (`_innerLoopCache`).**

Every unique `(structural signature, inputTypes, outputType)` triple produces a `DynamicMethod` that's JIT-compiled once and cached in a process-wide `ConcurrentDictionary<string, NpyInnerLoopFunc>` keyed by the cache-key string. The cache is append-only within the process lifetime. Cache keys are strings, so GC collects the old tree nodes once compilation completes, but the compiled delegate itself holds its `DynamicMethod` handle indefinitely.

Typical memory profile:
- Each compiled kernel is ~2-5 KB of native code + its metadata in the runtime's dynamic-method table.
- Typical application: a few dozen unique expressions → ~100-200 KB of steady-state cache.
- Pathological: a hot loop constructing new-per-call trees → linear growth. Reuse expression objects or pass explicit cache keys.

To inspect or reset during tests:
```csharp
ILKernelGenerator.InnerLoopCachedCount;   // count of compiled kernels
ILKernelGenerator.ClearInnerLoopCache();  // wipe for fresh-start testing
```

Both are `internal`, so scripts need the `AssemblyName=NumSharp.DotNetRunScript` override.

**2. Registered delegates and bound targets (`DelegateSlots`).**

Paths B and C of `Call` stash a managed reference in a static `ConcurrentDictionary<int, Delegate>` or `ConcurrentDictionary<int, object>` so the emitted IL can look it up at runtime. The reference is **strong** — entries live for the process lifetime. This is necessary: if the reference were weak, the GC could collect the delegate while a compiled kernel still holds its slot ID, and the next lookup would throw.

The cost is small per registration (~16-32 bytes for the dictionary entry plus whatever the delegate captures), but unbounded across registrations. Registering one delegate per kernel is fine; registering one delegate per iteration of a loop is a leak.

| Pattern | Registrations | Memory impact |
|--------|---------------|---------------|
| Static method (Path A) | zero | none |
| Cached delegate reused every iter | one | negligible |
| Per-call lambda | one per call | linear in call count |

Test hook:
```csharp
DelegateSlots.RegisteredCount;  // strong-ref count across both dicts
DelegateSlots.Clear();          // wipe for testing (invalidates kernels that reference it!)
```

> Calling `DelegateSlots.Clear()` while a kernel that references a slot is compiled is a footgun — the next call will throw `KeyNotFoundException` from inside the generated IL. Only use in test setup/teardown where you also clear the inner-loop cache.

**3. NDArrays referenced by the iterator.**

Orthogonal to Tier C, but worth mentioning in the same section for completeness: `NpyIterRef` holds a managed `NDArray[]` field so the operands' backing memory isn't collected mid-iteration. The field is released when you `Dispose()` the ref — the `using var iter = ...` pattern handles this automatically. Forgetting to dispose keeps the NDArrays alive for however long the iterator lives.

**Registration-once pattern.**

For `Call`-based activations or user kernels used in hot loops, the idiomatic pattern is:

```csharp
static class MyActivations
{
    // One delegate instance, registered once when the static class is first touched.
    public static readonly Func<double, double> Swish =
        x => x / (1.0 + Math.Exp(-x));

    public static readonly Func<double, double> GELU =
        x => 0.5 * x * (1.0 + Math.Tanh(
            Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}

// Usage — reuses the same slot + cached kernel every time:
var swished = NpyExpr.Call(MyActivations.Swish, NpyExpr.Input(0));
var gelud   = NpyExpr.Call(MyActivations.GELU,  NpyExpr.Input(0));
```

##### Validation and errors

The DSL fails fast at tree-construction time for structural errors and at compile time for type-mismatch or arity errors:

| Error condition | Where | Exception |
|----------------|-------|-----------|
| `NpyExpr.Input(-1)` | Factory | `ArgumentOutOfRangeException` |
| `NpyExpr.Sqrt(null)` | Node constructor | `ArgumentNullException` |
| `NpyExpr.Add(null, x)` / `Add(x, null)` | Node constructor | `ArgumentNullException` |
| `ExecuteExpression(expr, null, outType)` | Bridge entry | `ArgumentNullException` |
| `ExecuteExpression(expr, inputTypes, outType)` with too-few inputs vs operand count | Bridge entry | `ArgumentException` |
| `Input(5)` when tree compiled with 2 inputs | Compile-time IL emission | `InvalidOperationException` — message: `"Input(5) out of range; compile provided 2 inputs."` |
| Tree calls a vector-only path on a non-SIMD type (shouldn't happen via public API) | Compile-time | `NotSupportedException` |

Runtime errors depend on the op and dtype:

- `Divide` / `Mod` / `FloorDivide` with a zero integer divisor → `DivideByZeroException` from the CLI. Float division by zero produces `±Infinity` / `NaN` per IEEE 754, no exception.
- `Power(neg, fractional)` → `NaN` via `Math.Pow`, no exception.
- Overflow during `Conv_*` from a float that's outside the target integer range → silently wraps or saturates per the CLI's conv opcode semantics (matches `unchecked {}` casts in C#). Use `Conv_Ovf_*` if you need checked behavior — not exposed through the DSL.

##### Gotchas

A non-exhaustive list of pitfalls worth internalizing:

- **NaN propagation in `Min`/`Max` matches `np.minimum`/`np.maximum`, not `np.fmin`/`np.fmax`.** If you need NaN-skipping min/max, compose with `IsNaN` and `Where`:
  ```csharp
  // fmin(a, b): return non-NaN if one is NaN, else min
  var fmin = NpyExpr.Where(NpyExpr.IsNaN(a),
      b,
      NpyExpr.Where(NpyExpr.IsNaN(b), a, NpyExpr.Min(a, b)));
  ```

- **`Mod` doesn't match C# `%` for negative operands.** C# truncates toward zero (`-10 % 3 == -1`); NumPy (and `NpyExpr.Mod`) floor toward negative infinity (`-10 mod 3 == 2`). This matches Python `%`.

- **Integer division by zero throws.** `Divide(int_arr, int_arr_with_zero)` raises `DivideByZeroException` at runtime. Float division is silent (produces `±Infinity`/`NaN`).

- **Constants widen to the output dtype.** `NpyExpr.Const(1_000_000_000) + NpyExpr.Input(0)` where the output is `Byte` will emit `Ldc_I4 1000000000` followed by `Conv_U1` — the billion wraps to a small byte. The DSL won't check that the constant fits; you get silent truncation.

- **Bitwise ops require integer output dtype.** `NpyExpr.Input(0) & NpyExpr.Input(1)` with `outputType = Double` is a malformed tree — `EmitScalarOperation(BitwiseAnd, Double)` doesn't emit `And` for floats. You'll get an `InvalidOperationException` or unverifiable IL at compile time. Use an integer output dtype, or convert through `BitwiseNot`/`BitwiseAnd` in integer land and cast to float separately.

- **`LogicalNot` is `x == 0`, not `x != 0`.** It returns 1 when the input is zero and 0 otherwise. Same as Python's `not` applied to a numeric value. If you want "non-zero as 1", use `NpyExpr.NotEqual(x, NpyExpr.Const(0))`.

- **Input dtype mismatch is silent.** If your `inputTypes[]` says `Int32` but the actual NDArray operand is `Int16`, the kernel reads 4 bytes starting at the int16 pointer — garbage. The iterator's buffer/cast machinery only kicks in with `BUFFERED | NPY_*_CASTING`. For ad-hoc Tier C use, make sure `inputTypes[i]` matches the actual NDArray dtype, or run the iterator with casting flags.

- **Comparisons in non-float arithmetic can be off-by-one.** For integer-output trees, `NpyExpr.Greater(x, Const(0.5))` with `x` as `Int32` will compare two integers — `Const(0.5)` gets emitted as `Ldc_I4 0`, because `ConstNode.EmitLoadTyped` converts the literal to the output dtype's CLI type. `Greater(int_x, 0)` is almost never what you intended. Use an explicit `Const(1)` with the correct integer threshold, or change the output dtype to a float.

- **`Where` duplicates both branches in IL.** The true-branch IL and false-branch IL are emitted sequentially with a `br` skipping the false side when cond is true. Deeply-nested `Where`s quadruple IL size (1 → 2 → 4 → 8 branches). For more than ~10 levels of nesting, consider flattening with a lookup table via Tier B.

- **`Call` delegates are held forever.** `CallNode` stashes captured delegates and bound instance targets in a process-wide `DelegateSlots` dictionary so the emitted IL can look them up. There is no eviction. If you call `NpyExpr.Call(x => x * scale, in0)` inside a hot loop (creating a new closure each iteration), the dictionary grows without bound. Register delegates once at startup — a `static readonly Func<double, double>` field or a DI singleton — and reuse them.

- **`Call` method-group ambiguity.** `NpyExpr.Call(Math.Abs, x)` fails to compile because `Math.Abs` has nine overloads (`double`, `float`, `int`, `long`, etc.) and the compiler can't pick one. Cast to the specific `Func<...>` you want: `NpyExpr.Call((Func<double, double>)Math.Abs, x)`. Single-overload methods like `Math.Sqrt`, `Math.Cbrt`, `Math.Log` bind without cast.

- **`Call` runs at scalar speed.** A managed method call per element forfeits SIMD. For a sustained throughput-critical op, it's ~30-50% slower than the equivalent built-in DSL node because the call itself has overhead beyond just computing the result. Use `Call` for math the DSL doesn't expose (user-defined activations, `MathNet.Numerics` routines, lookup tables via a method), not for things like `Sqrt` where `NpyExpr.Sqrt(x)` is the right answer.

- **`Call` return type widening is lossy for NaN.** If a delegate returns `int` and the tree output is `double`, `Math.Floor(NaN) = NaN` gets cast to `int` (yielding `0` or some CPU-dependent value), which widens back to the float representation of that integer. NaN information is lost across integer-returning calls. Match return dtype to output dtype when NaN correctness matters.

##### Debugging compiled kernels

Tier C kernels are `DynamicMethod` delegates — you can't step into their IL with a debugger as-is. What you *can* do:

- **Inspect the kernel cache.** `ILKernelGenerator.InnerLoopCachedCount` (internal; use `[InternalsVisibleTo]` or a `dotnet_run` script with `AssemblyName=NumSharp.DotNetRunScript`) gives you a count. `ILKernelGenerator.ClearInnerLoopCache()` (internal) lets you force recompilation in a test.
- **Inspect the delegate slot registry** (only relevant when `Call` is in play). `DelegateSlots.RegisteredCount` (internal) returns the sum of registered delegates + registered instance targets. Growing unboundedly means a per-call lambda or target allocation somewhere — find it by comparing counts before and after your suspected hot path. `DelegateSlots.Clear()` wipes the registry; always pair with `ClearInnerLoopCache()` because cleared-but-cached kernels will throw `KeyNotFoundException` on their next invocation.
- **Print the auto-derived cache key.** Construct the tree, call `new StringBuilder().Also(e => node.AppendSignature(sb))` (`AppendSignature` is internal). The printed signature is exactly what goes into the cache key — useful for diagnosing "why aren't these two trees sharing a kernel?". For `Call` nodes in particular, the signature includes `MetadataToken` and `ModuleVersionId` — if those differ across two calls of what you thought was the same method, the compiler loaded the method from different assemblies or modules.
- **Reduce to a minimal tree.** If a compiled kernel misbehaves, isolate the failing subtree by compiling just that fragment against a tiny input (1-3 elements). `ExecuteExpression` on a 3-element array still exercises the scalar path; crashes become reproducible in a few lines.
- **Watch the output dtype.** `ExecuteExpression` expects `outputType` to match the output NDArray's dtype. If they disagree, the kernel reads/writes wrong byte counts. Double-check both.
- **Diagnose "method group ambiguous" errors.** If you see `CS0121: The call is ambiguous between the following methods` when writing `NpyExpr.Call(Math.X, ...)`, the method has multiple overloads (e.g. `Math.Abs` has 9). Cast to the specific `Func<...>` you want, or use the `MethodInfo` overload with an explicit parameter-types array to `GetMethod`.
- **Diagnose "Method X returns void"** errors — you passed a method with no return value to `Call`. Tier C requires every node to contribute a value to the output dtype.
- **Diagnose "Target is X, method declares Y"** errors — your instance `MethodInfo` call received a target that isn't an instance of the method's declaring type. Confirm both the method and the target came from the same type, especially if you're reflecting across a plugin boundary.
- **Enable IL dumps** by emitting into a persistent assembly instead of `DynamicMethod` — not a supported build configuration, but `ILKernelGenerator.InnerLoop.cs` is a single partial file you can modify in a workspace-only diff if you need to dump bytes during development.

##### When to use Tier C

Reach for Tier C when you want Layer 3 ergonomics for fused or custom ops and you're not chasing the last 15% of throughput. The DSL covers arithmetic, bitwise, rounding, transcendentals (exp/log/trig/hyperbolic/inverse-trig), predicates (IsNaN/IsFinite/IsInf), comparisons, Min/Max/Clamp/Where, and common compositions (ReLU, Leaky ReLU, sigmoid, clamp, hypot, linear, FMA, piecewise functions) without writing IL. For absolute peak perf on a hot ufunc — or for ops outside the DSL's node catalog (e.g. intrinsics the runtime exposes but the DSL doesn't wrap) — drop to Tier B and hand-tune the vector body.

**Decision tree: which tier do I need?**

```
Is the op a standard NumPy ufunc already in ExecuteBinary/Unary/Reduction?
  yes → Layer 3 (baked). Fastest, zero work. Done.
  no ↓

Can I express it as a tree of DSL nodes (Add, Sqrt, Where, Exp, etc.)?
  yes → Tier C. Fused, SIMD-or-scalar automatic, no IL.
  no ↓

Is the missing piece a BCL method (Math.X, user activation, reflected plugin)?
  yes → Tier C with Call. Scalar but fused. Done.
  no ↓

Do I need V256/V512 intrinsics the DSL doesn't wrap (Fma, Shuffle, ...)?
  yes → Tier B. Hand-write the vector body; factory wraps the shell.
  no ↓

Is the loop shape non-rectangular (gather/scatter, cross-element deps)?
  yes → Tier A. Emit the whole inner-loop IL yourself.
```

**Caching is shared across all tiers.** All three write into the same `_innerLoopCache` inside `ILKernelGenerator.InnerLoop.cs`. The first `ExecuteRawIL("k")` call JIT-compiles; every subsequent call with the same key returns the cached delegate immediately. `InnerLoopCachedCount` (internal) exposes the size for tests.

---

## Path Detection

`DetectExecutionPath()` is the heart of Layer 3. It looks at the iterator *after* coalescing and negative-stride flipping, and picks:

```csharp
if (CONTIGUOUS flag set)                                return SimdFull;
if (NDim == 0)                                          return SimdFull;
if (op1 is scalar AND op0 is contiguous)                return SimdScalarRight;
if (op0 is scalar AND op1 is contiguous)                return SimdScalarLeft;
if (every operand's innermost stride ∈ {0, 1})          return SimdChunk;
otherwise                                               return General;
```

"Scalar" here means every stride is 0 across every dimension — the operand is a 0-d array or a fully broadcasted view. "Contiguous" uses the standard backward stride check.

The resulting `ExecutionPath` is baked into the `MixedTypeKernelKey`:

```csharp
var key = new MixedTypeKernelKey(LhsType, RhsType, ResultType, Op, Path);
```

Different paths get different IL. `SimdFull` emits a flat 4× unrolled SIMD loop. `SimdScalarRight` broadcasts the scalar into a vector once, then runs a SIMD loop against only the LHS. `SimdChunk` processes the inner dim as a chunk within an outer coordinate loop. `General` does full coordinate-based iteration in IL. All of that machinery already lives in `ILKernelGenerator`; Layer 3's job is just to pick the right key.

---

## Worked Examples

Seventeen worked examples grouped by API tier.

**Layers 1–3 (baked kernels):**
1. [Three-operand binary over a 3-D contiguous array](#1-three-operand-binary-over-a-3-d-contiguous-array)
2. [Array × scalar with broadcast detection](#2-array--scalar-with-broadcast-detection)
3. [Sliced view — non-contiguous input](#3-sliced-view--non-contiguous-input)
4. [Fused hypot via Layer 1](#4-fused-hypot-via-layer-1)
5. [Early-exit Any over 1M elements](#5-early-exit-any-over-1m-elements)

**Tier B (templated scalar + vector bodies):**

6. [Fused hypot via Tier C expression](#6-fused-hypot-via-tier-c-expression)
7. [Fused linear transform via Tier B with vector body](#7-fused-linear-transform-via-tier-b-with-vector-body)

**Tier C (expression DSL):**

8. [ReLU via Tier C comparison-multiply](#8-relu-via-tier-c-comparison-multiply)
9. [Clamp with Min/Max](#9-clamp-with-minmax)
10. [Softmax-ish: exp then divide-by-sum](#10-softmax-ish-exp-then-divide-by-sum)
11. [Sigmoid via Where for numerical stability](#11-sigmoid-via-where-for-numerical-stability)
12. [NaN-replacement using IsNaN + Where](#12-nan-replacement-using-isnan--where)
13. [Leaky ReLU via piecewise Where](#13-leaky-relu-via-piecewise-where)
14. [Manual abs via comparison + Where](#14-manual-abs-via-comparison--where)
15. [Heaviside step function](#15-heaviside-step-function)
16. [Polynomial evaluation via Horner's method](#16-polynomial-evaluation-via-horners-method)
17. [Piecewise: absolute value of sine (abs(sin(x)))](#17-piecewise-absolute-value-of-sine-abssinx)
18. [User-defined activation via NpyExpr.Call](#18-user-defined-activation-via-npyexprcall)
19. [Reflected MethodInfo with an instance method](#19-reflected-methodinfo-with-an-instance-method)

### 1. Three-operand binary over a 3-D contiguous array

```csharp
var a = np.arange(24, dtype: np.float32).reshape(2, 3, 4);
var b = (np.arange(24, dtype: np.float32).reshape(2, 3, 4) * 2f).astype(np.float32);
var c = np.zeros(new Shape(2, 3, 4), np.float32);

using var iter = NpyIterRef.MultiNew(
    nop: 3, op: new[] { a, b, c },
    flags: NpyIterGlobalFlags.None,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_NO_CASTING,
    opFlags: new[] { NpyIterPerOpFlags.READONLY,
                     NpyIterPerOpFlags.READONLY,
                     NpyIterPerOpFlags.WRITEONLY });

iter.ExecuteBinary(BinaryOp.Add);
// NDim = 1 after coalesce, Path = SimdFull
// ILKernelGenerator emits a 4×-unrolled V256 add loop
// c[1,2,3] = 69
```

One call. 3-D → 1-D coalesce → one SIMD kernel runs over 24 elements. The generated IL is the same regardless of whether `a` and `b` started as 3-D, 4-D, or flat — as long as they're contiguous.

### 2. Array × scalar with broadcast detection

```csharp
var vec = np.arange(8, dtype: np.float32);
var sc  = np.full(new Shape(), 100f, NPTypeCode.Single);   // 0-d scalar
var res = np.zeros(new Shape(8), np.float32);

using var iter = NpyIterRef.MultiNew(3, new[] { vec, sc, res }, ...);

Console.WriteLine(iter.DetectExecutionPath());  // SimdScalarRight
iter.ExecuteBinary(BinaryOp.Multiply);
// res = vec * 100
```

The 0-d scalar comes through with all strides equal to 0, so `DetectExecutionPath` picks `SimdScalarRight`. The kernel loads the scalar once, splats it into a V256 register, and multiplies the whole LHS against it.

### 3. Sliced view — non-contiguous input

```csharp
var big   = np.arange(20, dtype: np.float32).reshape(4, 5);
var slice = big[":, 1:4"];     // 4×3 view, strides = [5, 1]
var dst   = np.zeros(new Shape(4, 3), np.float32);

using var iter = NpyIterRef.MultiNew(2, new[] { slice, dst }, ...);
iter.ExecuteUnary(UnaryOp.Sqrt);
// dst[3,2] = sqrt(big[3,3]) = sqrt(18) ≈ 4.243
```

The slice can't coalesce (stride 5 on outer axis, stride 1 on inner) so NDim stays at 2 and `IsContiguous` is false. Layer 3 picks the strided `UnaryKernel`, which computes `offset = sum(coord[d] * stride[d])` at each element.

### 4. Fused hypot via Layer 1

```csharp
using var iter = NpyIterRef.MultiNew(3, new[] { a, b, result },
    NpyIterGlobalFlags.EXTERNAL_LOOP, ...);

iter.ForEach((ptrs, strides, count, _) => {
    if (strides[0] == 4 && strides[1] == 4 && strides[2] == 4) {
        float* pa = (float*)ptrs[0], pb = (float*)ptrs[1], pc = (float*)ptrs[2];
        for (long i = 0; i < count; i++)
            pc[i] = MathF.Sqrt(pa[i] * pa[i] + pb[i] * pb[i]);   // JIT → V256
    } else {
        byte* pA = (byte*)ptrs[0], pB = (byte*)ptrs[1], pC = (byte*)ptrs[2];
        long sA = strides[0], sB = strides[1], sC = strides[2];
        for (long i = 0; i < count; i++) {
            float av = *(float*)(pA + i * sA);
            float bv = *(float*)(pB + i * sB);
            *(float*)(pC + i * sC) = MathF.Sqrt(av * av + bv * bv);
        }
    }
});
```

Without Layer 1 this operation would be `sqrt(a * a + b * b)` — three Layer 3 calls and three temporary arrays. Fused into one kernel, it runs in a single pass with zero intermediates. The stride branch is the idiom that lets the JIT autovectorize the tight case while the outer shape keeps the kernel correct for strided inputs.

### 5. Early-exit Any over 1M elements

```csharp
var data = np.zeros(new Shape(1_000_000), NPTypeCode.Int32);
data[500] = 1;

using var iter = NpyIterRef.New(data, flags: NpyIterGlobalFlags.EXTERNAL_LOOP);
bool found = iter.ExecuteReducing<AnyNonZero, bool>(default, false);
// found = true, after exactly one ForEach call (SIMD early exit inside kernel).
```

### 6. Fused hypot via Tier C expression

The same hypot operation written as an expression tree — no IL, no hand-written stride branch. The factory emits a 4×-unrolled V256 kernel on the contiguous path and a scalar-strided fallback on non-contiguous input.

```csharp
using var iter = NpyIterRef.MultiNew(3, new[] { a, b, result },
    NpyIterGlobalFlags.EXTERNAL_LOOP, ...);

var expr = NpyExpr.Sqrt(NpyExpr.Square(NpyExpr.Input(0)) +
                        NpyExpr.Square(NpyExpr.Input(1)));

iter.ExecuteExpression(expr,
    inputTypes: new[] { NPTypeCode.Single, NPTypeCode.Single },
    outputType: NPTypeCode.Single);
// result[i] = sqrt(a[i]² + b[i]²), fused in one pass, SIMD-vectorized
```

Compare with example 4 — same output, same performance envelope, no IL emission visible in your code. The tree's structural signature `"Sqrt(Add(Square(In[0]),Square(In[1])))"` becomes the cache key, so every iterator that runs the same expression reuses the same compiled delegate.

### 7. Fused linear transform via Tier B with vector body

When you want the Tier C ergonomics but also want the vector body under your control (e.g. to insert a Vector256 intrinsic the DSL doesn't expose):

```csharp
iter.ExecuteElementWiseBinary(
    NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single,
    scalarBody: il =>
    {
        // Stack: [a, b] → [a*2 + b*3]
        il.Emit(OpCodes.Ldc_R4, 2.0f); il.Emit(OpCodes.Mul);  // a*2
        var tmp = il.DeclareLocal(typeof(float));
        il.Emit(OpCodes.Stloc, tmp);                           // stash a*2
        il.Emit(OpCodes.Ldc_R4, 3.0f); il.Emit(OpCodes.Mul);  // b*3
        il.Emit(OpCodes.Ldloc, tmp); il.Emit(OpCodes.Add);    // a*2 + b*3
    },
    vectorBody: il =>
    {
        // Stack: [va, vb]
        il.Emit(OpCodes.Ldc_R4, 2.0f); ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);  // va*2
        var tmp = il.DeclareLocal(ILKernelGenerator.GetVectorType(typeof(float)));
        il.Emit(OpCodes.Stloc, tmp);
        il.Emit(OpCodes.Ldc_R4, 3.0f); ILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Single);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Multiply, NPTypeCode.Single);  // vb*3
        il.Emit(OpCodes.Ldloc, tmp);
        ILKernelGenerator.EmitVectorOperation(il, BinaryOp.Add, NPTypeCode.Single);
    },
    cacheKey: "linear_2a_3b_f32");
```

Single pass, no temporaries, SIMD-unrolled. Conceptually the same as `2*a + 3*b` written via Tier C, but lets you drop in `Vector256.Fma` or similar intrinsics if you ever need them.

### 8. ReLU via Tier C comparison-multiply

ReLU in one fused kernel, leveraging Tier C's "comparison returns 0/1 at output dtype" semantics:

```csharp
using var iter = NpyIterRef.MultiNew(2, new[] { input, output },
    NpyIterGlobalFlags.EXTERNAL_LOOP, ...);

var relu = NpyExpr.Greater(NpyExpr.Input(0), NpyExpr.Const(0.0f)) * NpyExpr.Input(0);
iter.ExecuteExpression(relu,
    new[] { NPTypeCode.Single }, NPTypeCode.Single);
// output[i] = max(input[i], 0) for every i
```

No branch, no intermediate array. The comparison node emits an I4 0/1, gets converted to float, and the multiply folds it into the final value. Scalar path only (comparisons don't SIMD), but the JIT autovectorizes the resulting tight loop post-tier-1.

### 9. Clamp with Min/Max

```csharp
var clamped = NpyExpr.Clamp(NpyExpr.Input(0), NpyExpr.Const(-1.0), NpyExpr.Const(1.0));
iter.ExecuteExpression(clamped,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
// output[i] = min(max(input[i], -1), 1)
```

`Clamp` is just sugar for `Min(Max(x, lo), hi)` — both map to branchy scalar selects that propagate NaN (matching `np.minimum` / `np.maximum` rather than `np.fmin` / `np.fmax`).

### 10. Softmax-ish: exp then divide-by-sum

Tier C is element-wise; reductions (like summing all elements) aren't expressible directly. But the element-wise half of softmax is:

```csharp
// out = exp(x - max_x) / sum_exp   — where max_x and sum_exp are precomputed scalars.
var shifted = NpyExpr.Subtract(NpyExpr.Input(0), NpyExpr.Const(maxX));
var numerator = NpyExpr.Exp(shifted);
var result = numerator / NpyExpr.Const(sumExp);
iter.ExecuteExpression(result,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Scalar path only (Exp isn't in the vector emit set), but the tree fuses three operations into one kernel — versus three Layer 3 calls with two temporary arrays.

### 11. Sigmoid via Where for numerical stability

The naive `1 / (1 + exp(-x))` overflows for very negative `x` (exp of a large positive number). A numerically stable form uses two branches:

```csharp
//           { 1 / (1 + exp(-x))   if x >= 0
// sigmoid = { exp(x) / (1 + exp(x)) if x < 0
var x = NpyExpr.Input(0);
var pos = NpyExpr.Const(1.0) / (NpyExpr.Const(1.0) + NpyExpr.Exp(-x));
var neg = NpyExpr.Exp(x) / (NpyExpr.Const(1.0) + NpyExpr.Exp(x));
var stable = NpyExpr.Where(NpyExpr.GreaterEqual(x, NpyExpr.Const(0.0)), pos, neg);

iter.ExecuteExpression(stable,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Every branch computes three `Exp` calls in the worst case, but only the taken branch's values are materialized — `Where` emits actual `brfalse` + jump IL, not a branchless blend. For large arrays, branch prediction handles a sign-bit pattern well. If your input is already known to be mostly positive or mostly negative, this is noticeably cheaper than the naive `1/(1+exp(-x))` kernel.

### 12. NaN-replacement using `IsNaN` + `Where`

```csharp
// replace NaN with 0
var x = NpyExpr.Input(0);
var clean = NpyExpr.Where(NpyExpr.IsNaN(x), NpyExpr.Const(0.0), x);
iter.ExecuteExpression(clean,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

`IsNaN(x)` emits a `double.IsNaN` call that leaves an I4 0/1 on the stack, and `UnaryNode` inserts an implicit `EmitConvertTo(Int32, Double)` so `Where`'s condition-normalizer gets the right dtype. The whole tree is scalar-only but fuses NaN-detection and replacement into a single pass.

### 13. Leaky ReLU via piecewise Where

```csharp
// leaky_relu(x, alpha=0.1) = x if x > 0 else alpha * x
var x = NpyExpr.Input(0);
var leaky = NpyExpr.Where(
    NpyExpr.Greater(x, NpyExpr.Const(0.0)),
    x,
    NpyExpr.Const(0.1) * x);
iter.ExecuteExpression(leaky,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Contrast with the "branchless" ReLU (`(x > 0) * x`): that works for plain ReLU because the false branch is zero, but doesn't handle Leaky ReLU's non-zero negative side. `Where` is the general escape hatch.

### 14. Manual abs via comparison + Where

A worked example of combining comparisons with `Where` for pedagogical purposes (the DSL's `Abs` is faster — it has a SIMD path):

```csharp
var x = NpyExpr.Input(0);
var manualAbs = NpyExpr.Where(
    NpyExpr.Less(x, NpyExpr.Const(0.0)),
    -x,           // operator overload for Negate
    x);
iter.ExecuteExpression(manualAbs,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

This is ~10% slower than `NpyExpr.Abs(x)` because it runs the scalar-only `Where` instead of the SIMD-vectorized `Abs`. Use the built-in where possible; `Where` is the generalization when no built-in fits.

### 15. Heaviside step function

```csharp
// heaviside(x, h0) = 0 if x < 0, h0 if x == 0, 1 if x > 0
// NumPy's np.heaviside(x, 0.5) is the default "midpoint" convention.
var x = NpyExpr.Input(0);
var step = NpyExpr.Where(
    NpyExpr.Less(x, NpyExpr.Const(0.0)),
    NpyExpr.Const(0.0),
    NpyExpr.Where(
        NpyExpr.Greater(x, NpyExpr.Const(0.0)),
        NpyExpr.Const(1.0),
        NpyExpr.Const(0.5)));   // h0 value at x == 0

iter.ExecuteExpression(step,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Three-way nested `Where` flattens to linear IL — two `brfalse` branches at runtime. The auto-derived cache key becomes `Where(CmpLess(In[0],Const[0]),Const[0],Where(CmpGreater(In[0],Const[0]),Const[1],Const[0.5]))`. Reused automatically across iterators.

### 16. Polynomial evaluation via Horner's method

Evaluate `p(x) = 1·x⁴ + 2·x³ + 3·x² + 4·x + 5` with optimal multiplications:

```csharp
// ((((1·x + 2)·x + 3)·x + 4)·x + 5
var x = NpyExpr.Input(0);
var poly = (((NpyExpr.Const(1.0) * x + NpyExpr.Const(2.0)) * x
             + NpyExpr.Const(3.0)) * x
             + NpyExpr.Const(4.0)) * x
             + NpyExpr.Const(5.0);
iter.ExecuteExpression(poly,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

Four `Multiply`s, four `Add`s — all SIMD-capable. Whole tree emits the 4×-unrolled V256 path. For a degree-N polynomial this stays in registers end-to-end, with no intermediate array allocations. Compare with the naïve `1*x*x*x*x + 2*x*x*x + 3*x*x + 4*x + 5` — ten multiplications, same IL size after constant folding by the JIT, but less readable.

### 17. Piecewise: absolute value of sine (abs(sin(x)))

Combine the two unary SIMD-capable ops for the pattern `|sin x|`:

```csharp
var expr = NpyExpr.Abs(NpyExpr.Sin(NpyExpr.Input(0)));
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

`Sin` is scalar-only, so the whole tree runs scalar (no 4× unroll). But both ops fuse into one pass — a single `Math.Sin` call + `Math.Abs` per element. The alternative — two Layer 3 calls on three arrays — would allocate a `sin(x)` temporary.

### 18. User-defined activation via `NpyExpr.Call`

Say you want **Swish** (`x * sigmoid(x)`, used in EfficientNet and family) but Tier C doesn't have a `Sigmoid` node. Drop to `Call`:

```csharp
// Registered once at startup — static readonly field, not a per-call lambda.
static readonly Func<double, double> SwishActivation =
    x => x / (1.0 + Math.Exp(-x));

// Tree: out = Swish(x) + bias  (bias is a broadcast-scalar Input, not a Const)
var expr = NpyExpr.Call(SwishActivation, NpyExpr.Input(0)) + NpyExpr.Input(1);
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double, NPTypeCode.Double }, NPTypeCode.Double);
```

The `SwishActivation` delegate is registered exactly once into `DelegateSlots`; every subsequent iter reuses the same slot ID and the same compiled kernel (auto-derived cache key is stable because it's keyed by `MethodInfo.MetadataToken`, not delegate identity). Runtime overhead is ~5 ns per element for the slot lookup + one `Delegate.Invoke` call per element — still single-pass, still zero intermediates.

For maximum speed, if your activation is hot enough to matter, compose it out of DSL primitives:
```csharp
var x = NpyExpr.Input(0);
var swish = x / (NpyExpr.Const(1.0) + NpyExpr.Exp(-x));   // same op, no Call overhead
```

### 19. Reflected MethodInfo with an instance method

Sometimes you're calling a method you discovered via reflection (e.g. an op registered through a plugin system). Use the `MethodInfo + target` overload:

```csharp
var provider = new PluginActivations { Temperature = 1.5 };
var method = provider.GetType().GetMethod("ApplyTempered")!;
// ApplyTempered(double x) => Math.Pow(x, 1.0 / Temperature);

var expr = NpyExpr.Call(method, provider, NpyExpr.Input(0));
iter.ExecuteExpression(expr,
    new[] { NPTypeCode.Double }, NPTypeCode.Double);
```

The `provider` object's state (`Temperature`) is captured into the compiled kernel via a `DelegateSlots` slot ID. Mutating `provider.Temperature` between calls is visible to subsequent invocations — the slot holds the reference, not a snapshot.

---

## Performance

Benchmarking 1M `sqrt` on a contiguous float32 array after 300 warmup iterations, Ryzen-class CPU:

| Approach | Time | ns/elem | Notes |
|----------|------|---------|-------|
| `ForEach` with byte-ptr scalar | 2.82 ms | 2.82 | JIT autovectorizes V256 sqrt, no unroll |
| `ExecuteGeneric<Scalar>` byte-ptr | 2.54 ms | 2.54 | Same, no delegate indirection |
| `ExecuteGeneric<Scalar>` typed-ptr branch | 2.79 ms | 2.79 | `if (stride == 4) float*` branch |
| `ExecuteGeneric<V256+4x>` hand-SIMD | **0.86 ms** | 0.86 | User-written Vector256 + 4× unroll |
| `ExecuteUnary(Sqrt)` IL kernel | **0.75 ms** | 0.75 | `ILKernelGenerator`'s 4×-unrolled V256 |

**Layer 3 is ~3.7× faster than Layer 1/2 scalar code** — the gap is entirely explained by loop unrolling, since the JIT does autovectorize a typed-pointer loop into V256 but doesn't issue the four independent vectors per iteration that `ILKernelGenerator` emits. A user who writes Vector256 + 4× unroll by hand closes the gap to 15% (0.86 vs 0.75 ms).

Layer 1 and Layer 2 give you control and fusion. For any standard elementwise ufunc, **Layer 3 is the right default**. Drop to Layer 1/2 when fusing several ops (one pass, zero temporaries), when the op isn't in `ILKernelGenerator`, or when your kernel has a structure the generator can't express.

**Custom ops (Tier B / Tier C) hit the Layer 3 envelope.** Because the factory wraps user bodies in the same 4×-unrolled SIMD + remainder + scalar-tail shell, a Tier B or Tier C kernel for sqrt lands within rounding distance of `ExecuteUnary(Sqrt)` — the only overhead is the runtime contig check (a few stride comparisons at kernel entry). Fused ops like `sqrt(a² + b²)` via Tier C are typically faster than composing three Layer 3 calls, because there are no intermediate arrays and the whole computation stays in V256 registers between operations.

**Custom op overhead breakdown.** Tier A and Tier B kernels share the same `NpyInnerLoopFunc` delegate shape as the baked ufuncs; call overhead is identical. Tier C adds:

| Overhead source | When | Cost |
|----------------|------|------|
| Compile (first call per key) | First `ExecuteExpression` with a given cache key | 1-10 ms one-time (IL emission + JIT) |
| Auto-key derivation | When `cacheKey: null` | ~O(tree size) StringBuilder walk — typically < 1 μs |
| Runtime contig check | Every inner-loop entry | 2-4 stride comparisons (~ns) |
| Scalar-strided fallback | When any operand has non-contig inner stride | Per-element pointer arithmetic; JIT autovectorizes post-tier-1 |
| `Call` dispatch (Path A) | Every element — static method | One `call <methodinfo>`; JIT may inline |
| `Call` dispatch (Path B/C) | Every element — instance or delegate | `ldc.i4 + DelegateSlots.Lookup + castclass + callvirt` (~5-10 ns) |

**When fusion pays off.** Fusing `sqrt(a² + b²)` into one Tier C kernel avoids materializing the `a²` and `a² + b²` intermediates. For 1M float32 elements, that's 8 MB of memory traffic saved per temporary — on a typical 30-GB/s RAM bandwidth, that's ~300 μs per avoided temporary. Fusing 3 ops into one Tier C kernel can beat 3 baked Layer 3 calls by 1-2× when memory-bound.

**When Call pays off.** If the user-supplied method does nontrivial work (e.g. three `Math.Exp` calls for a numerically-stable sigmoid), the dispatch overhead is a few-percent tax on something that was never going to SIMD anyway. If the method is trivial (`x => x * 2`), composing out of DSL primitives (`NpyExpr.Input(0) * NpyExpr.Const(2.0)`) keeps the SIMD path and runs 3-5× faster. Pick Call when the method is the cheapest thing to write and the kernel isn't a hot path; pick DSL composition when the kernel is profiled and matters.

### JIT Warmup Caveat

**Critical gotcha for benchmarking.** .NET uses tiered compilation: methods first compile to unoptimized tier-0 code, then get promoted to tier-1 after ~100+ calls. Until tier-1 kicks in, **autovectorization doesn't happen**. A scalar kernel that eventually runs at 2.5 ms/iter will look like 70+ ms/iter if you only warm up 10 times.

Symptoms of under-warmed benchmarks:
- Layer 2 scalar shows 50-80 ms instead of 2-5 ms
- `ExecuteGeneric` looks slower than `ForEach` (it isn't, post-warmup)
- Reusing a single iterator looks 50× faster than constructing fresh ones (the reuse path warmed up faster because it kept hitting the same call site)

Benchmark with ≥200 warmup iterations per variant, not just a few. Production code doesn't see this effect because long-running loops are always past tier-1.

### Implementation Notes

The bridge is tuned for the JIT in two ways:

1. **Fast-path split.** `ExecuteGeneric` dispatches to `ExecuteGenericSingle` (1 call, inlineable) or `ExecuteGenericMulti` (do/while driver). Small single-call bodies are what the autovectorizer needs to do its job — a do/while with a delegate inside prevents tier-1 SIMD promotion.

2. **`AggressiveInlining + AggressiveOptimization`.** Both attributes sit on the fast path so the JIT doesn't punt on inlining due to method size and immediately promotes to tier-1 once discovered hot.

Without these, `ExecuteGeneric` gets stuck at tier-0 in micro-benchmarks and looks 30× slower than it actually is.

### When Does Each Layer Pay Off?

| Layer | Good for | Drawback |
|-------|----------|----------|
| Layer 1 (`ForEach`) | Exploration, one-off fused kernels, non-standard ops | Delegate allocation per call; no loop unrolling |
| Layer 2 (`ExecuteGeneric`) | Same as Layer 1 in a hot path | No delegate cost, otherwise same — no loop unrolling |
| Layer 3 (`Execute*`) | Standard ufuncs already in `ILKernelGenerator` | No fusion; one kernel per call |
| `BufferedReduce` | Axis reductions with casting | Double-loop only worth it with `BUFFER + REDUCE` |

To reach Layer 3 parity in Layer 2, keep a typed-pointer fast branch and add the 4× unroll yourself. The typed-pointer contiguous branch helps the JIT tier up faster and gives the autovectorizer a trivial pattern to match:

```csharp
public void Execute(void** p, long* s, long n) {
    if (s[0] == sizeof(float) && s[1] == sizeof(float)) {
        float* src = (float*)p[0]; float* dst = (float*)p[1];
        for (long i = 0; i < n; i++) dst[i] = MathF.Sqrt(src[i]);  // JIT → V256
    } else {
        byte* p0 = (byte*)p[0]; byte* p1 = (byte*)p[1];
        long s0 = s[0], s1 = s[1];
        for (long i = 0; i < n; i++)
            *(float*)(p1 + i * s1) = MathF.Sqrt(*(float*)(p0 + i * s0));
    }
}
```

For maximum throughput, write the 4×-unrolled V256 version in the fast branch — you'll land within 15% of the IL kernel.

### Allocations

Layer 3 allocates exactly once per call: the stackalloc stride arrays (NDim longs each). No heap allocation. Layer 2 inlines the entire kernel body into the JIT's codegen of `ExecuteGeneric` — no allocation at all, not even a delegate. Layer 1 allocates a single delegate per call (closure if it captures anything).

**Custom-op tiers:**

| Tier | Per-call allocation | One-time allocation |
|------|--------------------|--------------------|
| Tier A (`ExecuteRawIL`) | stackalloc strides + the user's `Action<ILGenerator>` closure on first compile | compiled `DynamicMethod` cached by key; stays live for process lifetime (~2-5 KB native + runtime metadata) |
| Tier B (`ExecuteElementWise`) | stackalloc strides + (on first compile) two `Action<ILGenerator>` closures | compiled kernel cached by key |
| Tier C (`ExecuteExpression`) | stackalloc strides + (on first compile) an NpyExpr tree allocated by the caller + StringBuilder for the auto-key | compiled kernel cached by key |
| Tier C with `Call` | same as Tier C, plus one `DelegateSlots` entry per unique captured delegate / bound target | registered references live for process lifetime; see [Memory model and lifetime](#memory-model-and-lifetime) |

The one case where allocations grow without bound is the anti-pattern of constructing a new `Call` delegate per iteration — each new delegate reference gets a new slot ID and a new cache entry. Register delegates once at startup to avoid this.

---

## Known Bugs and Workarounds

While building `NpyIter.Execution.cs` we surfaced two bugs in the iterator that callers should know about. Both are documented in the source of `NpyIter.Execution.cs` and both are worked around by the bridge.

### Bug A: `Iternext()` ignores `EXTERNAL_LOOP`

`NpyIterRef.Iternext()` calls `state.Advance()` unconditionally. `Advance()` is the per-element ripple-carry advance — it doesn't know about `EXLOOP`. The correct advance for `EXLOOP` is `ExternalLoopNext`, which `GetIterNext()` returns based on flags but `Iternext()` bypasses.

**Symptom.** A caller using `Iternext()` with `EXTERNAL_LOOP` set reads past the end of each inner chunk and iterates `NDim - 1` extra times.

**Workaround in the bridge.** `ForEach`, `ExecuteGeneric`, and `ExecuteReducing` call `GetIterNext()` directly:

```csharp
var iternext = GetIterNext();
do {
    kernel(...);
} while (iternext(ref *_state));
```

### Bug B: Buffered + Cast pointer advance

When `BUFFERED` is set and the operand dtype differs from the array dtype, `NpyIterBufferManager.CopyToBuffer` fills a contiguous buffer at the *buffer dtype* (e.g. 8 bytes per element for `double`). But `state.Strides[op]` still contains the array's element-count strides — `Advance()` then computes `Strides[op] * ElementSizes[op]`, where `ElementSizes[op]` is now the buffer dtype's size. The product is the wrong byte delta.

**Symptom.** Buffered casts silently return garbage. A minimal repro:

```csharp
var i32 = np.arange(10, dtype: np.int32);
var f64 = np.zeros(new Shape(10), np.float64);

using var iter = NpyIterRef.MultiNew(2, new[] { i32, f64 },
    NpyIterGlobalFlags.BUFFERED, NPY_ORDER.NPY_KEEPORDER,
    NPY_CASTING.NPY_SAFE_CASTING,
    new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY },
    opDtypes: new[] { NPTypeCode.Double, NPTypeCode.Double });

// Iterating with iter.Iternext() returns wrong values.
```

**Workaround in the bridge.** `ExecuteBinary` routes buffered paths through `RunBufferedBinary`, which uses `_state->BufStrides` (which `NpyIterBufferManager` correctly sets to `GetElementSize(op)` = buffer-dtype size) instead of `state.Strides`. The bridge also uses `GetInnerLoopByteStrides()` for Layer 1/2 — it returns `BufStrides` when `BUFFER` is set and converts element strides to byte strides otherwise.

Both bugs are fixable inside `NpyIter.cs`. Until they are, the bridge is the only way to use buffered iteration correctly — any direct use of `iter.Iternext()` with these flag combinations will be wrong.

### Bug C (fixed): `NpyExpr.Where` now works

Historically `WhereNode.EmitScalar` had an incomplete prelude that threw `InvalidOperationException("WhereNode prelude needs redesign")` at IL-compile time. The rewritten node evaluates `cond` in the output dtype, compares it to zero via `EmitComparisonOperation(NotEqual, outType)` (which yields a verifiable I4 0/1), and branches on that. Works uniformly across integer, float, and decimal output dtypes.

### Bug D (core, fixed): `NPTypeCode.SizeOf(Decimal)` disagreed with `InfoOf<decimal>.Size`

Historically `NPTypeCode.SizeOf(Decimal)` returned **32** while the actual `decimal` type is 16 bytes (verified: `UnmanagedStorage` lays decimals out at 16-byte stride). The iterator used `NPTypeCode.SizeOf` for `ElementSizes`, so any custom-op kernel that multiplied element strides by `ElementSizes` read at 32-byte offsets into 16-byte-stride storage, producing `System.OverflowException` when the garbage happened to decode as a huge decimal.

Fixed in the commit that introduced the custom-op API (`32 → 16`). All decimal-using code benefits, not just the bridge.

### Bug E (fixed): predicates silently wrote garbage to the output slot

`IsNaN` / `IsFinite` / `IsInf` emit via `double.IsNaN(x)` etc., which leaves a `bool` (I4 0/1) on the evaluation stack. The factory's `Stind` takes the output dtype — storing an I4 into an 8-byte double slot reinterprets the bit pattern as a tiny denormal (0.0 or ~4.94e-324), not as the intended 0.0 or 1.0 result. Output arrays filled with near-zero garbage looked "mostly correct" for mixed inputs, hiding the bug in casual use.

**Fix:** `UnaryNode.EmitScalar` inspects the op and emits a trailing `EmitConvertTo(Int32, outType)` for predicate results. The I4 0/1 becomes a properly-typed 0.0 or 1.0.

**Caught by:** `NpyExprExtensiveTests.IsNaN_Double` — a test deliberately run early in the battletest phase, because NaN behavior is usually the first thing to go wrong.

### Bug F (fixed): `LogicalNot` broken for Int64 / float / decimal

`EmitUnaryScalarOperation(UnaryOp.LogicalNot, outType)` in `ILKernelGenerator` emits `Ldc_I4_0` + `Ceq` — correct when the operand is I4-sized (bool, byte, int16, int32), broken when the operand is anything else. For a `Double` on the stack, the comparison `ceq(double, I4_0)` is type-mismatched IL that produces undefined output (in practice, always-1 on our test hardware).

**Fix:** `UnaryNode.EmitScalar` special-cases `UnaryOp.LogicalNot`: it routes through `EmitComparisonOperation(Equal, outType)` with a properly-typed zero literal (emitted by `EmitPushZero` — `Ldc_R8 0.0` for Double, `Ldc_I8 0L` for Int64, `decimal.Zero` for Decimal, etc.), then converts the I4 result to the output dtype. The underlying `ILKernelGenerator` emit path is still broken for direct use; NpyExpr simply doesn't use it for this op.

**Caught by:** `LogicalNot_Double_Operator` test — all outputs came back as `1.0` regardless of input, because the type-mismatched `ceq` always returned true on this CPU.

### Bug G (library, exposed): `Vector256.Round/Truncate` don't exist on .NET 8

`ILKernelGenerator.CanUseUnarySimd` lists `UnaryOp.Round` and `UnaryOp.Truncate` as SIMD-supported, and `EmitUnaryVectorOperation` looks up `Vector256.Round(Vector256<double>)` and `Vector256.Truncate(Vector256<double>)` at compile time. Those methods exist in .NET 9+ but **not in .NET 8** — the lookup returns null and throws `InvalidOperationException("Could not find Round/Truncate for Vector256\`1")`.

The existing Unary kernel cache never hit this bug because production `np.round` / `np.trunc` paths are exercised mostly in tests and tests are usually run against one framework. Tier C exercises every op for every SIMD-eligible dtype, and surfaces it immediately.

**Fix (in NpyExpr only, not in `ILKernelGenerator`):** `NpyExpr.UnaryNode.IsSimdUnary` excludes `Round` and `Truncate`, routing them to the scalar path on both net8 and net9+. Scalar rounding is still JIT-autovectorized post-tier-1, so the practical performance delta is small.

**Caught by:** `Truncate_Double` in the extensive tests — crashed at compile time on net8 with the "Could not find" error.

**Upstream fix would be:** conditionally compile `ILKernelGenerator.CanUseUnarySimd` to exclude `Round`/`Truncate` on `#if !NET9_0_OR_GREATER`, or explicitly check `method != null` with a fallback emit.

### Bug H (fixed): `MinMaxNode` didn't propagate NaN

Originally `MinMaxNode` emitted a branchy select via `EmitComparisonOperation(LessEqual / GreaterEqual, outType)`. IEEE 754 says any comparison with NaN is false, so `Min(NaN, 3.0)` with the branchy approach returned `3.0` — but NumPy's `np.minimum(np.nan, 3.0)` returns `NaN`. The implementation matched C# `<=` semantics rather than NumPy.

**Fix:** `MinMaxNode.EmitBranchy` delegates to `Math.Min` / `Math.Max` via reflection lookup on `typeof(Math)`. Those methods explicitly propagate NaN per IEEE 754 (any NaN operand yields NaN), matching NumPy's `np.minimum`/`np.maximum`. For `Char` / `Boolean` outputs, where no `Math.Min(Char, Char)` overload exists, the node falls back to the branchy path (NaN propagation irrelevant for those types).

**Caught by:** `Min_Double_NaNPropagation` test — expected NaN, got the non-NaN operand.

> NumPy has two variants: `np.minimum` (NaN-propagating, our choice) and `np.fmin` (NaN-skipping). If you need `fmin`/`fmax` semantics, compose with `IsNaN` and `Where` — see the [Gotchas](#gotchas) section.

---

## Summary

NpyIter is how NumSharp turns "iterate these three arrays of possibly-different shapes, types, and strides" into a deterministic schedule of pointer advances. `NpyIter.Execution.cs` is how that schedule becomes a SIMD kernel call.

**The core idea.** NumPy's C++ templates compile `for (i = 0; i < n; i++) c[i] = a[i] + b[i]` ahead of time, specialized per type. NumSharp cannot. Instead it emits that same loop as IL via `DynamicMethod` the first time you ask for it, then caches the JIT-compiled delegate forever. `NpyIter` handles the *layout* problem (what offsets, in what order), `ILKernelGenerator` handles the *type* problem (what opcodes, with what SIMD intrinsics), and `NpyIter.Execution.cs` hands the one to the other.

**Three layers.** `ExecuteBinary / Unary / Reduction / ...` for standard ufuncs (this is what you want 90% of the time — it's ~3.7× faster than a JIT-autovectorized scalar loop and ~1.15× faster than hand-written Vector256 + 4× unroll). `ExecuteGeneric<TKernel>` for custom kernels that need zero dispatch overhead. `ForEach` with a `NpyInnerLoopFunc` delegate when you're exploring, fusing, or writing something exotic.

**Custom ops extend Layer 3.** When a baked ufunc doesn't match your problem, three tiers let you reach the same SIMD-unrolled performance envelope without leaving the bridge: `ExecuteRawIL` (you emit the whole body), `ExecuteElementWise` (you supply per-element scalar + vector IL; factory wraps the unroll shell), `ExecuteExpression` (compose with `NpyExpr` — no IL required). Each tier is cached, reuses `ILKernelGenerator`'s emit primitives, and runs through the same `ForEach` driver as baked ops.

**Coalesce first.** A 3-D contiguous array should run as one flat SIMD loop, not a triple-nested loop. The iterator does this for you — as long as you don't set flags that disable it (`MULTI_INDEX`, `C_INDEX`, `F_INDEX`).

**Buffer when casting or when non-contiguous + SIMD-critical.** The iterator will copy strided input into aligned contiguous buffers, run the kernel there, and write back. Just be aware of Bug B above if you're working around the bridge.

**Struct-generic is a template substitute.** Constraining a type parameter to `struct` lets the JIT specialize the method per concrete type at codegen time. For hot inner loops this is indistinguishable from a hand-inlined function. Use it — but remember that **scalar kernel code only autovectorizes after tier-1 JIT promotion**, which takes ~100+ hot-loop iterations. Microbenchmarks that warm up 10 times will wildly under-report Layer 1/2 performance. Production code never sees this effect.

**Simple kernels autovectorize after warmup.** Post-tier-1, the JIT autovectorizes both byte-pointer `*(float*)(p + i*s) = ...` and typed-pointer `dst[i] = ...` loops into Vector256. If you care about every microsecond, a stride-equality branch with typed pointers in the fast path is slightly more robust and reaches tier-1 faster, but it's not the order-of-magnitude difference you might expect — the Vector256 + 4×-unroll hand-kernel is.

Everything else — flag enums, op_axes encoding, negative-stride flipping, the double-loop reduction schedule — exists to handle corner cases NumPy users write every day without thinking. NumSharp handles them the same way, just translated into a language where we emit IL instead of expanding templates.

## See Also

- [IL Generation](il-generation.md) — the kernel side of the bridge
- [Broadcasting](broadcasting.md) — stride-0 iteration
- [Buffering & Memory](buffering.md) — buffer allocation and lifetime
