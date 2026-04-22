# NpyIter Implementation Fixes Required

**To:** Developer implementing NpyIter parity
**From:** Architecture review
**Date:** 2026-04-15
**Priority:** High
**Reference:** NumPy source at `src/numpy/numpy/_core/src/multiarray/nditer_*.c`

---

## Executive Summary

The current NpyIter implementation provides a working foundation but diverges from NumPy's behavior in several critical ways. These differences will cause NumSharp operations to produce different results than NumPy in edge cases, break code ported from Python, and prevent proper integration with IL kernels that expect NumPy-compatible iteration patterns.

This document details each fix required, why it matters, and how to implement it correctly.

---

## Fix #1: Coalescing Must Always Run

### Current Behavior (Wrong)
```csharp
// In NpyIterRef.Initialize()
if ((flags & NpyIterGlobalFlags.EXTERNAL_LOOP) != 0)
{
    _state->ItFlags |= (uint)NpyIterFlags.EXLOOP;
    NpyIterCoalescing.CoalesceAxes(ref *_state);
}
```

Coalescing only runs when `EXTERNAL_LOOP` is requested.

### NumPy Behavior (Correct)
```c
// In nditer_constr.c, line 395-396
if (ndim > 1 && !(itflags & NPY_ITFLAG_HASMULTIINDEX)) {
    npyiter_coalesce_axes(iter);
}
```

NumPy **always** coalesces axes after construction unless multi-index tracking is enabled.

### Why This Matters

1. **Performance**: Without coalescing, a contiguous (2, 3, 4) array iterates with 3 nested loops instead of 1 flat loop. This is 3x more loop overhead.

2. **SIMD Eligibility**: IL kernels check `NDim == 1` to enable SIMD fast paths. Without coalescing, contiguous arrays miss this optimization.

3. **Behavioral Parity**: NumPy code like `np.nditer([a, b])` produces a 1D iterator for contiguous arrays. NumSharp would produce a 3D iterator for the same input.

4. **External Loop Contracts**: When `EXTERNAL_LOOP` is set, callers expect the innermost dimension to be as large as possible. Without prior coalescing, this assumption breaks.

### Required Fix

```csharp
// In NpyIterRef.Initialize(), replace the coalescing block with:

// Apply coalescing unless multi-index tracking is requested
// NumPy: nditer_constr.c line 395-396
if (_state->NDim > 1 && (flags & NpyIterGlobalFlags.MULTI_INDEX) == 0)
{
    NpyIterCoalescing.CoalesceAxes(ref *_state);
}

// Then handle external loop flag separately
if ((flags & NpyIterGlobalFlags.EXTERNAL_LOOP) != 0)
{
    _state->ItFlags |= (uint)NpyIterFlags.EXLOOP;
}
```

### Test Case
```csharp
var arr = np.arange(24).reshape(2, 3, 4);  // Contiguous

// NumPy: ndim=1 after coalescing (shape=[24])
// Current NumSharp: ndim=3 (shape=[2,3,4]) - WRONG
using var iter = NpyIterRef.New(arr);
Assert.AreEqual(1, iter.NDim);  // Must pass
```

---

## Fix #2: Stride Layout Incompatibility

### Current Layout (Problematic)
```csharp
// NpyIterState.cs
public fixed long Strides[MaxDims * MaxOperands];  // [op0_axis0, op0_axis1, ..., op1_axis0, ...]

// Access pattern
public long GetStride(int axis, int op)
{
    return Strides[op * MaxDims + axis];  // op-major layout
}
```

### NumPy Layout
```c
// NumPy uses per-axis NpyIter_AxisData structures
struct NpyIter_AxisData_tag {
    npy_intp shape, index;
    Py_intptr_t ad_flexdata;  // Strides for all operands at this axis
};
// Access: NAD_STRIDES(axisdata)[op]  // axis-major layout
```

### Why This Matters

1. **GetInnerStrideArray() Contract**: NumPy's `NpyIter_GetInnerStrideArray()` returns a contiguous array of inner strides for all operands: `[op0_inner_stride, op1_inner_stride, ...]`. The current layout requires gathering these from scattered locations.

2. **Cache Efficiency**: When iterating, you access strides for all operands at the same axis together. Axis-major layout has better cache locality.

3. **Coalescing Algorithm**: The coalescing algorithm compares strides across operands at the same axis. Current layout requires pointer arithmetic.

### Required Fix

Either:

**Option A: Change layout to axis-major (Recommended)**
```csharp
// Strides[axis * MaxOperands + op] - axis-major
public long GetStride(int axis, int op)
{
    return Strides[axis * MaxOperands + op];
}
```

**Option B: Add inner stride cache**
```csharp
// Add separate array for inner strides (gathered from main array)
public fixed long InnerStrides[MaxOperands];

// Update when NDim changes
public void UpdateInnerStrides()
{
    int innerAxis = NDim - 1;
    for (int op = 0; op < NOp; op++)
        InnerStrides[op] = GetStride(innerAxis, op);
}
```

### Impact Assessment

Option A requires updating:
- `NpyIterState.GetStride()` / `SetStride()`
- `NpyIterState.GetStridesPointer()`
- `NpyIterCoalescing.CoalesceAxes()`
- `NpyIterRef.Initialize()`
- Static `NpyIter.CoalesceAxes()`

Option B is less invasive but adds memory overhead.

---

## Fix #3: op_axes Parameter Not Implemented

### Current State
```csharp
public static NpyIterRef AdvancedNew(
    ...
    int opAxesNDim = -1,      // Ignored
    int[][]? opAxes = null,   // Ignored
    ...
)
```

### NumPy Behavior
```c
// op_axes allows remapping operand dimensions to iterator dimensions
// Example: iterate over columns of a 2D array
int op_axes[2] = {1, 0};  // Swap axes
NpyIter_AdvancedNew(1, &arr, ..., 2, &op_axes, ...);
```

### Why This Matters

1. **Reduction Operations**: `np.sum(arr, axis=1)` uses `op_axes` to mark axis 1 as the reduction axis while iterating over axis 0.

2. **Transpose Iteration**: Iterating over transposed views without copying requires axis remapping.

3. **Broadcasting Control**: `op_axes` with `-1` entries marks dimensions for broadcasting.

4. **NumPy API Parity**: Many NumPy ufuncs internally use `op_axes` for complex operations.

### Required Implementation

```csharp
private void ApplyOpAxes(int opAxesNDim, int[][] opAxes)
{
    if (opAxes == null || opAxesNDim < 0)
        return;

    for (int op = 0; op < _state->NOp; op++)
    {
        if (opAxes[op] == null)
            continue;

        var opAxisMap = opAxes[op];
        var originalStrides = new long[opAxesNDim];

        // Gather original strides
        var stridePtr = _state->GetStridesPointer(op);
        for (int i = 0; i < opAxesNDim; i++)
            originalStrides[i] = stridePtr[i];

        // Apply remapping
        for (int iterAxis = 0; iterAxis < opAxesNDim; iterAxis++)
        {
            int opAxis = opAxisMap[iterAxis];
            if (opAxis < 0)
            {
                // -1 means broadcast this dimension (stride = 0)
                stridePtr[iterAxis] = 0;
            }
            else
            {
                stridePtr[iterAxis] = originalStrides[opAxis];
            }
        }
    }
}
```

### Test Case
```csharp
// Sum along axis 1: result shape (3,) from input (3, 4)
var arr = np.arange(12).reshape(3, 4);
var result = np.empty(3);

// op_axes: input uses all axes, output broadcasts axis 1
int[][] opAxes = { null, new[] { 0, -1 } };  // -1 = reduction axis

using var iter = NpyIterRef.AdvancedNew(
    nop: 2,
    op: new[] { arr, result },
    opAxesNDim: 2,
    opAxes: opAxes,
    ...);
```

---

## Fix #4: Missing Multi-Index Support

### Current State
Multi-index tracking (`HASMULTIINDEX` flag) is defined but never set or used.

### NumPy Behavior
```c
// Construction with MULTI_INDEX flag
NpyIter_New(arr, NPY_ITER_MULTI_INDEX, ...);

// Access current multi-index
npy_intp multi_index[NPY_MAXDIMS];
NpyIter_GetMultiIndex(iter, multi_index);

// Jump to specific multi-index
NpyIter_GotoMultiIndex(iter, multi_index);
```

### Why This Matters

1. **Coordinate Tracking**: Operations like `np.where()` need to know the coordinates of each element, not just the flat index.

2. **Sparse Operations**: Building sparse arrays requires coordinate tracking.

3. **Debugging**: Multi-index is essential for debugging iteration order.

4. **RemoveAxis() Prerequisite**: NumPy's `RemoveAxis()` requires multi-index tracking.

### Required Implementation

```csharp
// In NpyIterRef
public void GetMultiIndex(Span<long> outCoords)
{
    if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
        throw new InvalidOperationException("Iterator not tracking multi-index");

    for (int d = 0; d < _state->NDim; d++)
        outCoords[d] = _state->Coords[d];
}

public void GotoMultiIndex(ReadOnlySpan<long> coords)
{
    if ((_state->ItFlags & (uint)NpyIterFlags.HASMULTIINDEX) == 0)
        throw new InvalidOperationException("Iterator not tracking multi-index");

    // Validate coordinates
    for (int d = 0; d < _state->NDim; d++)
    {
        if (coords[d] < 0 || coords[d] >= _state->Shape[d])
            throw new IndexOutOfRangeException($"Coordinate {coords[d]} out of range for axis {d}");
    }

    // Update coordinates and compute linear index
    long iterIndex = 0;
    long multiplier = 1;

    for (int d = _state->NDim - 1; d >= 0; d--)
    {
        _state->Coords[d] = coords[d];
        iterIndex += coords[d] * multiplier;
        multiplier *= _state->Shape[d];
    }

    _state->IterIndex = iterIndex;

    // Update data pointers
    for (int op = 0; op < _state->NOp; op++)
    {
        long offset = 0;
        for (int d = 0; d < _state->NDim; d++)
            offset += coords[d] * _state->GetStride(d, op);

        _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * _state->ElementSizes[op];
    }
}
```

### Construction Change
```csharp
// In Initialize()
if ((flags & NpyIterGlobalFlags.MULTI_INDEX) != 0)
{
    _state->ItFlags |= (uint)NpyIterFlags.HASMULTIINDEX;
    // Do NOT coalesce when multi-index is tracked
}
```

---

## Fix #5: Ranged Iteration Not Implemented

### Current State
`IterStart` and `IterEnd` are defined but always set to `0` and `IterSize`.

### NumPy Behavior
```c
// Iterate only elements 100-200
NpyIter_ResetToIterIndexRange(iter, 100, 200);

// Or construct with range
NpyIter_AdvancedNew(..., NPY_ITER_RANGED, ...);
```

### Why This Matters

1. **Parallel Chunking**: Divide iteration among threads by giving each a range.

2. **Lazy Evaluation**: Process only needed elements.

3. **Memory Efficiency**: Avoid loading entire arrays when only a subset is needed.

### Required Implementation

```csharp
public bool ResetToIterIndexRange(long start, long end)
{
    if (start < 0 || end > _state->IterSize || start > end)
        return false;

    _state->IterStart = start;
    _state->IterEnd = end;
    _state->ItFlags |= (uint)NpyIterFlags.RANGE;

    GotoIterIndex(start);
    return true;
}
```

---

## Fix #6: Buffer Copy Lacks Type Generality

### Current State
```csharp
// Type-specific methods
public static void CopyToBuffer<T>(...) where T : unmanaged
```

Requires compile-time type knowledge.

### NumPy Behavior
```c
// Runtime dtype dispatch
npyiter_copy_to_buffers(iter, prev_dataptrs);
// Handles any dtype via NPY_cast_info
```

### Why This Matters

1. **Generic Iteration**: Can't write dtype-agnostic iteration code.

2. **Type Casting**: NumPy supports iteration with automatic type promotion.

3. **IL Kernel Integration**: Kernels expect dtype-dispatched copy.

### Required Implementation

```csharp
public static void CopyToBuffer(ref NpyIterState state, int op, long count)
{
    var dtype = state.GetOpDType(op);

    switch (dtype)
    {
        case NPTypeCode.Boolean: CopyToBuffer<bool>(ref state, op, count); break;
        case NPTypeCode.Byte: CopyToBuffer<byte>(ref state, op, count); break;
        case NPTypeCode.Int16: CopyToBuffer<short>(ref state, op, count); break;
        case NPTypeCode.UInt16: CopyToBuffer<ushort>(ref state, op, count); break;
        case NPTypeCode.Int32: CopyToBuffer<int>(ref state, op, count); break;
        case NPTypeCode.UInt32: CopyToBuffer<uint>(ref state, op, count); break;
        case NPTypeCode.Int64: CopyToBuffer<long>(ref state, op, count); break;
        case NPTypeCode.UInt64: CopyToBuffer<ulong>(ref state, op, count); break;
        case NPTypeCode.Single: CopyToBuffer<float>(ref state, op, count); break;
        case NPTypeCode.Double: CopyToBuffer<double>(ref state, op, count); break;
        case NPTypeCode.Decimal: CopyToBuffer<decimal>(ref state, op, count); break;
        case NPTypeCode.Char: CopyToBuffer<char>(ref state, op, count); break;
        default: throw new NotSupportedException($"Buffer copy not supported for {dtype}");
    }
}
```

---

## Fix #7: Iterator Flag Bit Positions

### Current State
```csharp
// NpyIterFlags.cs - flags at shifted positions
IDENTPERM = 0x0001 << 8,  // = 0x0100
NEGPERM = 0x0002 << 8,    // = 0x0200
```

### NumPy Layout
```c
#define NPY_ITFLAG_IDENTPERM    (1 << 0)  // = 0x0001
#define NPY_ITFLAG_NEGPERM      (1 << 1)  // = 0x0002
```

### Why This Matters

While the flags work internally, the bit positions don't match NumPy. This matters for:

1. **Debugging**: Can't compare flag values between implementations.
2. **Serialization**: If iterator state is ever serialized/logged.
3. **Interop**: Any future C interop would have mismatched flags.

### Required Fix

The current design reserves lower bits for legacy flags (`SourceBroadcast`, `SourceContiguous`, `DestinationContiguous`). Two options:

**Option A: Remove legacy flags (Breaking Change)**
```csharp
// Match NumPy exactly
IDENTPERM = 1 << 0,
NEGPERM = 1 << 1,
// Remove SourceBroadcast etc.
```

**Option B: Document the difference (Acceptable)**
Keep current layout but document that NumSharp uses different bit positions for internal reasons. The static `NpyIter` class maintains backward compatibility.

---

## Fix #8: MaxDims Too Small

### Current State
```csharp
internal const int MaxDims = 32;
```

### NumPy
```c
#define NPY_MAXDIMS 64
```

### Why This Matters

While 32 dimensions covers most cases, NumPy supports 64. Ported code with high-dimensional arrays will fail.

### Required Fix

```csharp
internal const int MaxDims = 64;  // Match NPY_MAXDIMS
```

**Impact**: Increases `NpyIterState` size from ~10KB to ~20KB. For stack-allocated states, this may cause stack overflow in deeply recursive code. Consider heap allocation for states with ndim > 16.

---

## Implementation Order

1. **Fix #1 (Coalescing)** - Critical, easy, high impact
2. **Fix #7 (Flags)** - Decide on approach
3. **Fix #6 (Buffer dispatch)** - Required for buffered iteration
4. **Fix #2 (Stride layout)** - Medium complexity, affects many files
5. **Fix #4 (Multi-index)** - Required for advanced features
6. **Fix #3 (op_axes)** - Complex, enables reductions
7. **Fix #5 (Ranged)** - Nice to have, enables parallelism
8. **Fix #8 (MaxDims)** - Simple but has memory impact

---

## Testing Requirements

After each fix, verify:

1. **All existing tests pass** (5652 tests)
2. **New edge case tests** for the specific fix
3. **NumPy comparison tests** - run same operations in both, compare results

Example NumPy comparison test pattern:
```csharp
[Test]
public void Coalescing_MatchesNumPy()
{
    // NumPy output (verified manually):
    // >>> import numpy as np
    // >>> arr = np.arange(24).reshape(2,3,4)
    // >>> it = np.nditer(arr)
    // >>> it.ndim
    // 1

    var arr = np.arange(24).reshape(2, 3, 4);
    using var iter = NpyIterRef.New(arr);
    Assert.AreEqual(1, iter.NDim, "Must match NumPy ndim after coalescing");
}
```

---

## Questions for Clarification

1. Should we maintain backward compatibility with existing `NpyIter` static class, or can we deprecate it?

2. Is heap allocation acceptable for large state structs (ndim > 16)?

3. Should we prioritize op_axes (enables reductions) or multi-index (enables coordinate tracking)?

4. Should failing coalescing tests block CI, or should they be marked as known differences?

---

## References

- `src/numpy/numpy/_core/src/multiarray/nditer_impl.h` - Data structures and flags
- `src/numpy/numpy/_core/src/multiarray/nditer_constr.c` - Construction logic
- `src/numpy/numpy/_core/src/multiarray/nditer_api.c` - API functions and coalescing
- `docs/NPYITER_PARITY_ANALYSIS.md` - Full parity comparison table
