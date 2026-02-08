# Plan: Rewrite Offset Model to Match NumPy's `base_offset + strides` Architecture

## Status: Investigation Required

This document is the entry point for planning the rewrite of NumSharp's view/offset resolution system to match NumPy's architecture exactly. It identifies what needs to change, what needs to be investigated first, and the risks involved.

## Problem Statement

NumSharp uses a chain-based offset model (ViewInfo + BroadcastInfo) to resolve memory offsets for sliced and broadcast arrays. NumPy uses a flat model: `base_offset + sum(stride[i] * coord[i])`. NumSharp's model is the root cause of most broadcast/slice bugs and creates O(chain_depth) overhead per element access.

### NumPy's Model (target)

Every ndarray view stores exactly:
```
data        → pointer to first element of THIS view (not the allocation base)
shape       → int[ndim] dimensions
strides     → int[ndim] bytes per step (0 for broadcast, negative for reversed)
base        → reference to parent array (for memory management / refcounting only)
```

Offset computation: `byte_offset = sum(strides[i] * coords[i])` — one loop, no branching, no chain resolution.

When creating a slice `a[2:8:2, ::-1]`:
```python
new.data    = a.data + 2*a.strides[0] + (a.shape[1]-1)*a.strides[1]
new.shape   = [3, original_cols]
new.strides = [a.strides[0]*2, -a.strides[1]]
new.base    = a.base or a
```

When broadcasting `broadcast_to(a, target_shape)`:
```python
new.data    = a.data
new.shape   = target_shape
new.strides = [0 if dim was stretched, else a.strides[i]]
new.base    = a
```

Both operations compose naturally: slicing a broadcast produces correct strides via arithmetic on the existing strides. No special cases.

### NumSharp's Current Model (to be replaced)

```
Shape
├── dimensions[]
├── strides[]        ← may be broadcast strides (0) OR original strides
├── ViewInfo         ← chain of slice history
│   ├── OriginalShape    ← the unsliced root shape
│   ├── ParentShape      ← intermediate sliced shape (for recursive slicing)
│   ├── Slices[]         ← SliceDef per dimension (start, stop, step)
│   └── UnreducedShape   ← shape before dimension reduction
└── BroadcastInfo    ← broadcast history
    ├── OriginalShape            ← shape before broadcasting
    └── UnreducedBroadcastedShape ← lazy-loaded resolved shape
```

Offset computation: 6+ code paths in `GetOffset`, `GetOffset_1D`, `GetOffset_broadcasted`, `GetOffset_broadcasted_1D`, `GetOffset_IgnoreViewInfo`, `resolveUnreducedBroadcastedShape`, plus recursive calls through `ParentShape.GetOffset`.

## Target Architecture

### Shape (after rewrite)

```csharp
public struct Shape
{
    internal int[] dimensions;
    internal int[] strides;     // fully resolved — absorbs slicing, broadcasting
    internal int base_offset;   // offset to first element within InternalArray
    internal int size;           // product of dimensions
    // ViewInfo and BroadcastInfo: REMOVED
    // IsSliced, IsBroadcasted: derived from strides (stride=0 → broadcast)
}
```

### GetOffset (after rewrite)

```csharp
public int GetOffset(params int[] coords)
{
    int offset = base_offset;
    for (int i = 0; i < coords.Length; i++)
        offset += strides[i] * coords[i];
    return offset;
}
```

### View Creation (after rewrite)

```csharp
// Slicing: a[start:stop:step] along axis
new_base_offset = old_base_offset + start * old_strides[axis];
new_strides[axis] = old_strides[axis] * step;
new_dimensions[axis] = (stop - start + step - 1) / step;  // ceiling division

// Negative step (reversal): a[::-1] along axis
new_base_offset = old_base_offset + (old_dimensions[axis] - 1) * old_strides[axis];
new_strides[axis] = -old_strides[axis];

// Broadcasting: stretch dim from 1 to N
new_strides[axis] = 0;  // that's it
new_dimensions[axis] = N;

// Index reduction: a[3] along axis — removes dimension, adjusts base_offset
new_base_offset = old_base_offset + 3 * old_strides[axis];
// remove axis from dimensions[] and strides[]
```

## Investigation Checklist

Before writing a full implementation plan, the following must be investigated:

### 1. Catalog all Shape consumers

Every place that reads `ViewInfo`, `BroadcastInfo`, `IsSliced`, `IsBroadcasted`, `IsRecursive` needs to be identified and mapped to the new model.

- [ ] `Shape.cs` — GetOffset variants, Slice(), TransformOffset, GetCoordinates, Clean, Clone
- [ ] `Shape.Unmanaged.cs` — unmanaged pointer access paths
- [ ] `Shape.Reshaping.cs` — reshape on sliced/broadcast views
- [ ] `NDIterator.cs` — iteration with AutoReset for broadcasting
- [ ] `MultiIterator.cs` — paired/broadcast iteration
- [ ] `UnmanagedStorage.Slicing.cs` — GetViewInternal, the Bug 17 materializing fix
- [ ] `UnmanagedStorage.Getters.cs` — element access
- [ ] `Default.Broadcasting.cs` — Broadcast() creates ViewInfo/BroadcastInfo
- [ ] Generated math templates (`Default.Add.*.cs`, `Default.Subtract.*.cs`, etc.) — ~24 files, ~200K lines that reference ViewInfo/BroadcastInfo for fast-path decisions
- [ ] `NDArray.Indexing.cs` — indexing dispatch
- [ ] Selection/masking code
- [ ] `np.reshape.cs` — reshape interacts with views

### 2. Understand IArraySlice bounds checking

NumPy adjusts the `data` pointer to point at the first view element. NumSharp uses `IArraySlice` with bounds-checked access via `Count`. Questions:

- [ ] Can `base_offset` be negative? (yes, for reversed views the first logical element may precede the parent's first element in memory — NumPy handles this with pointer arithmetic, NumSharp would need signed offset)
- [ ] Does `IArraySlice` support negative indexing or must we use raw pointers?
- [ ] Should we keep `IArraySlice` bounds checking (safety) or move to unchecked pointer access (performance)?
- [ ] How does `InternalArray.Address` interact with offset-based access?

### 3. Understand NDIterator's relationship to strides

NDIterator has fast paths for contiguous arrays and slow paths for sliced/broadcast. Questions:

- [ ] With the new model, is NDIterator still needed or can it be simplified to `pointer + stride` walking?
- [ ] Does AutoReset (for broadcast smaller-into-larger iteration) still work with flat strides?
- [ ] What is the performance impact of removing the contiguous fast path (since all arrays now use the same stride-based access)?

### 4. Understand reshape-after-slice interactions

`np.reshape(sliced_view)` in NumPy either returns a view (if the slice is contiguous) or copies. NumSharp's `IsRecursive` flag and `ParentShape` chain handle this. Questions:

- [ ] Can reshape on a non-contiguous view be handled with just `base_offset + strides`?
- [ ] When must reshape force a copy? (NumPy: when the data is not contiguous in the requested order)
- [ ] How to detect contiguity from strides alone? (check: `strides[i] == strides[i+1] * dimensions[i+1]` for all i)

### 5. Understand generated template code

The `Regen` templates generate type-specific math code that checks `IsBroadcasted` / `IsSliced` for fast-path branching.

- [ ] What does the template source look like? (`Default.Op.General.template.cs`)
- [ ] Can the fast-path decisions be made from strides alone? (e.g., contiguous = strides match row-major; broadcast = any stride is 0)
- [ ] How to regenerate after changes?

### 6. Map `IsSliced` / `IsBroadcasted` to stride-based checks

Current flags:
- `IsSliced` → `ViewInfo != null`
- `IsBroadcasted` → `BroadcastInfo != null`
- `IsContiguous` → complex check involving ViewInfo

New derivations:
- `IsBroadcasted` → `any(strides[i] == 0 && dimensions[i] > 1)`
- `IsContiguous` → `strides == row_major_strides(dimensions)` (can precompute)
- `IsSliced` → no longer a meaningful concept; all views are just `base_offset + strides`

### 7. Memory management / aliasing

- [ ] How does NumSharp track that a view shares memory with a parent? (`InternalArray` reference)
- [ ] Does `base_offset` change anything about GC pinning or unmanaged memory lifetime?
- [ ] The `Alias()` method creates views — confirm it can work with `base_offset`

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking the 1601 passing tests | High | Incremental approach — keep old model behind a flag initially |
| Generated template code (~200K lines) | High | Must understand Regen before touching templates |
| NDIterator redesign cascade | Medium | May need to rewrite iteration entirely |
| Performance regression | Medium | Benchmark before/after; the new model should be faster (fewer branches) |
| Reshape-after-slice edge cases | Medium | Port NumPy's contiguity check exactly |
| Negative base_offset for reversed views | Medium | Verify IArraySlice can handle it or use pointer arithmetic |

## Suggested Investigation Approach

1. Start with investigation items 1-3 above — catalog every consumer and understand the constraints
2. Build a prototype `Shape2` struct with `base_offset + strides` alongside the existing `Shape`
3. Add a `ToShape2()` conversion and verify offset parity on all test cases
4. Once parity is confirmed, plan the incremental migration file by file

## NumPy Reference Files

These files in `src/numpy/` contain the authoritative implementation:

| File | What to study |
|------|---------------|
| `numpy/_core/include/numpy/ndarraytypes.h` | `PyArrayObject` struct — `data`, `dimensions`, `strides`, `base` |
| `numpy/_core/src/multiarray/getitem.c` | Element access via strides |
| `numpy/_core/src/multiarray/mapping.c` | Slice/index view creation, stride computation |
| `numpy/_core/src/multiarray/ctors.c` | Array construction, contiguity checks |
| `numpy/_core/src/multiarray/shape.c` | Reshape — when to copy vs view |
| `numpy/_core/src/multiarray/nditer_constr.c` | Iterator setup, stride=0 for broadcast |
| `numpy/lib/_stride_tricks_impl.py` | `as_strided`, `broadcast_to`, `broadcast_arrays` |
