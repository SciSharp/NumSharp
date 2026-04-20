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
  - [Layer 1 — Canonical Inner-Loop API](#layer-1--canonical-inner-loop-api)
  - [Layer 2 — Struct-Generic Dispatch](#layer-2--struct-generic-dispatch)
  - [Layer 3 — Typed ufunc Dispatch](#layer-3--typed-ufunc-dispatch)
  - [Custom Operations (Tier A / B / C)](#custom-operations-tier-a--b--c)
    - [Tier A — Raw IL](#tier-a--raw-il)
    - [Tier B — Templated Inner Loop](#tier-b--templated-inner-loop)
    - [Tier C — Expression DSL](#tier-c--expression-dsl)
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

The layer is a partial declaration of `NpyIterRef` that exposes three layers of progressively higher abstraction. Pick the one that matches your use case.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3: ExecuteBinary / Unary / Reduction / Comparison / Scan      │ ← 90% case
│           "I want to add two arrays, please pick the best kernel"     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2: ExecuteGeneric<TKernel> / ExecuteReducing<TKernel, TAccum> │ ← custom kernel,
│           struct-generic, JIT-inlined zero-alloc                      │   perf-critical
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1: ForEach(NpyInnerLoopFunc kernel, void* aux)                │ ← raw power users,
│           delegate-based, closest to NumPy's C API                    │   experimentation
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
           NpyIter state (Shape, Strides, DataPtrs, Buffers, ...)
                                  │
                                  ▼
              ILKernelGenerator (DynamicMethod + V128/V256/V512)
```

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

| Factory | Semantics |
|---------|-----------|
| `NpyExpr.Input(i)` | Reference operand `i` (0-based input index). Auto-converts to output dtype on load. |
| `NpyExpr.Const(value)` | Literal — `int / long / float / double` overloads. Emitted at the output dtype. |

**Binary arithmetic.**

| Factory | Operator | SIMD | Notes |
|---------|----------|:----:|-------|
| `Add(a,b)` | `a + b` | ✓ | |
| `Subtract(a,b)` | `a - b` | ✓ | |
| `Multiply(a,b)` | `a * b` | ✓ | |
| `Divide(a,b)` | `a / b` | ✓ | |
| `Mod(a,b)` | `a % b` | — | NumPy floored modulo (result sign follows divisor, not dividend). |
| `Power(a,b)` | — | — | `Math.Pow` via scalar path. |
| `FloorDivide(a,b)` | — | — | NumPy floor-toward-negative-infinity. |
| `ATan2(y,x)` | — | — | Four-quadrant arctan. |

**Binary bitwise.**

| Factory | Operator | SIMD |
|---------|----------|:----:|
| `BitwiseAnd(a,b)` | `a & b` | ✓ |
| `BitwiseOr(a,b)` | `a \| b` | ✓ |
| `BitwiseXor(a,b)` | `a ^ b` | ✓ |

**Scalar-branchy combinators** (scalar path only).

| Factory | Semantics |
|---------|-----------|
| `Min(a,b)` | Delegates to `Math.Min` — matches `np.minimum` (propagates NaN per IEEE 754). |
| `Max(a,b)` | Delegates to `Math.Max` — matches `np.maximum` (propagates NaN per IEEE 754). |
| `Clamp(x,lo,hi)` | `Min(Max(x,lo),hi)` — sugar. |
| `Where(cond,a,b)` | Branchy ternary select: if `cond != 0` return `a` else `b`. `cond` is evaluated in the output dtype, so floats, integers, and decimals all work uniformly. |

**Unary — arithmetic.**

| Factory | Operator | SIMD |
|---------|----------|:----:|
| `Negate(x)` | unary `-x` | ✓ |
| `Abs(x)` | — | ✓ |
| `Sqrt(x)` | — | ✓ |
| `Square(x)` | — | ✓ |
| `Reciprocal(x)` | — | ✓ |
| `Cbrt(x)` | — | — |
| `Sign(x)` | — | — |

**Unary — exp / log.**

| Factory | Semantics | SIMD |
|---------|-----------|:----:|
| `Exp(x)` | eˣ | — |
| `Exp2(x)` | 2ˣ | — |
| `Expm1(x)` | eˣ − 1 | — |
| `Log(x)` | ln x | — |
| `Log2(x)` | log₂ x | — |
| `Log10(x)` | log₁₀ x | — |
| `Log1p(x)` | ln(1 + x) | — |

**Unary — trigonometric.**

| Factory | Semantics | SIMD |
|---------|-----------|:----:|
| `Sin(x)`, `Cos(x)`, `Tan(x)` | Standard trig | — |
| `Sinh(x)`, `Cosh(x)`, `Tanh(x)` | Hyperbolic | — |
| `ASin(x)`, `ACos(x)`, `ATan(x)` | Inverse | — |
| `Deg2Rad(x)` | x · π/180 | ✓ |
| `Rad2Deg(x)` | x · 180/π | ✓ |

**Unary — rounding.**

| Factory | Semantics | SIMD |
|---------|-----------|:----:|
| `Floor(x)` | ⌊x⌋ | ✓ |
| `Ceil(x)` | ⌈x⌉ | ✓ |
| `Round(x)` | Banker's rounding | — |
| `Truncate(x)` | Toward zero | — |

> `Round` and `Truncate` have a working SIMD path on .NET 9+, but NumSharp's library targets .NET 8 as well, where `Vector256.Round/Truncate` don't exist. NpyExpr gates them to the scalar path unconditionally so the compiled kernel works on both frameworks. Other contiguous rounding ops autovectorize after tier-1 JIT promotion.

**Unary — bitwise / logical / predicates.**

| Factory | Operator | SIMD | Notes |
|---------|----------|:----:|-------|
| `BitwiseNot(x)` | `~x` | ✓ | |
| `LogicalNot(x)` | `!x` | — | Boolean NOT. |
| `IsNaN(x)` | — | — | Returns 0/1 at output dtype. |
| `IsFinite(x)` | — | — | Returns 0/1 at output dtype. |
| `IsInf(x)` | — | — | Returns 0/1 at output dtype. |

**Comparisons** (produce numeric 0 or 1 at output dtype; scalar path only).

| Factory | Semantics |
|---------|-----------|
| `Equal(a,b)` | `a == b` |
| `NotEqual(a,b)` | `a != b` |
| `Less(a,b)` | `a < b` |
| `LessEqual(a,b)` | `a <= b` |
| `Greater(a,b)` | `a > b` |
| `GreaterEqual(a,b)` | `a >= b` |

Unlike NumPy's comparison ufuncs (which return `bool` arrays), Tier C's single-output-dtype model collapses comparisons to `0 or 1` at the output dtype. This composes cleanly with arithmetic — e.g. ReLU becomes `(x > 0) * x`.

**Operator overloads.** An expression tree reads like ordinary C#:

```csharp
// (a + b) * c + 1
var linear = (NpyExpr.Input(0) + NpyExpr.Input(1)) * NpyExpr.Input(2) + NpyExpr.Const(1.0f);

// ReLU via comparison × input
var relu = NpyExpr.Greater(NpyExpr.Input(0), NpyExpr.Const(0.0f)) * NpyExpr.Input(0);

// Clamp with no named method call
var clamped = NpyExpr.Min(NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Const(0f)), NpyExpr.Const(1f));
```

Overloads: `+ - * /` (arithmetic), `%` (NumPy mod), `& | ^` (bitwise), unary `-` (negate), `~` (bitwise not), `!` (logical not). No overloads for `<`, `>`, `==`, `!=` (those need to return `bool` in C#) — use the factory methods for comparisons.

##### Type discipline

Every intermediate value flows through the output dtype: `Input(i)` loads the i-th operand's dtype and auto-converts (via `EmitConvertTo`) to the output dtype; constants are emitted directly in the output dtype. The vector path is enabled only when **every** input dtype equals the output dtype (so a single `Vector<T>` instantiation covers the whole tree) **and every node in the tree has a SIMD emit**. If any node (e.g. `Min`, `Sin`, any comparison) lacks a SIMD path, the whole compilation falls back to scalar — correctness preserved, but no 4× unroll.

##### SIMD coverage rules

A node's `SupportsSimd` determines whether Tier C emits the vector body:

- **Yes:** `Input`, `Const`, the four arithmetic binary ops (`+ - * /`), the three bitwise binary ops (`& | ^`), and the unary ops `Negate`, `Abs`, `Sqrt`, `Floor`, `Ceil`, `Square`, `Reciprocal`, `Deg2Rad`, `Rad2Deg`, `BitwiseNot`.
- **No:** `Mod`, `Power`, `FloorDivide`, `ATan2`, `Min`/`Max`/`Clamp`/`Where`, all comparisons, `Round`, `Truncate` (no net8 SIMD method), all trig (except `Deg2Rad`/`Rad2Deg`), all log/exp, `Sign`, `Cbrt`, `LogicalNot`, predicates (`IsNaN`/`IsFinite`/`IsInf`).

**Predicate / LogicalNot result handling.** Predicates (`IsNaN`/`IsFinite`/`IsInf`) and `LogicalNot` emit an I4 0/1 on the stack, not a value of the output dtype. `UnaryNode` detects these ops and inserts a trailing `EmitConvertTo(Int32, outType)` so the factory's final `Stind` matches. `LogicalNot` in particular routes through `EmitComparisonOperation(Equal, outType)` with an output-dtype zero literal, because the default `ILKernelGenerator` emit path uses `Ldc_I4_0 + Ceq` which is only correct when the value fits in I4 — broken for Int64, Single, Double, Decimal. NpyExpr takes the safer route.

A tree's `SupportsSimd` is true only if **every** node in it does. One unsupported node demotes the whole tree to scalar-only — which is usually still autovectorized by the JIT after tier-1 promotion, just without the 4× unroll.

##### Caching

Pass `cacheKey` to share the compiled delegate across iterators; omit it and the compiler auto-derives one from the tree's structural signature plus input/output dtypes:

```
NpyExpr:Add(Mul(In[0],Const[2]),Const[3]):in=Double:out=Double
```

Two trees with identical structure and types get the same auto-derived key and share a cached kernel. Comparisons appear as `Cmp<Op>(...)`, Min/Max as `Min(...)`/`Max(...)`, and Where as `Where(...)` — all influence the cache key.

##### When to use Tier C

Reach for Tier C when you want Layer 3 ergonomics for fused or custom ops and you're not chasing the last 15% of throughput. The DSL covers arithmetic, bitwise, rounding, transcendentals (exp/log/trig/hyperbolic/inverse-trig), predicates (IsNaN/IsFinite/IsInf), comparisons, Min/Max/Clamp/Where, and common compositions (ReLU, Leaky ReLU, sigmoid, clamp, hypot, linear, FMA, piecewise functions) without writing IL. For absolute peak perf on a hot ufunc — or for ops outside the DSL's node catalog — drop to Tier B and hand-tune the vector body.

**Shared caching.** All three tiers write into the same `_innerLoopCache` inside `ILKernelGenerator.InnerLoop.cs`. The first `ExecuteRawIL("k")` call JIT-compiles; every subsequent call with the same key returns the cached delegate immediately. `InnerLoopCachedCount` (internal) exposes the size for tests.

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
