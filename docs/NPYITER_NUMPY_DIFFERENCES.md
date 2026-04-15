# NumPy nditer vs NumSharp NpyIter: Complete Differences Analysis

**Generated from NumPy source analysis**
**Reference files:**
- `src/numpy/numpy/_core/src/multiarray/nditer_impl.h`
- `src/numpy/numpy/_core/src/multiarray/nditer_constr.c`
- `src/numpy/numpy/_core/src/multiarray/nditer_api.c`

---

## 1. Memory Layout Differences

### NumPy: Flexible Data Structure
```c
struct NpyIter_InternalOnly {
    npy_uint32 itflags;
    npy_uint8 ndim;
    int nop, maskop;
    npy_intp itersize, iterstart, iterend;
    npy_intp iterindex;
    char iter_flexdata[];  // Variable-sized flexible array
};
```

NumPy uses a **flexible array member** (`iter_flexdata[]`) that contains:
1. `perm[NPY_MAXDIMS]` - axis permutation
2. `dtypes[nop]` - dtype pointers
3. `resetdataptr[nop+1]` - reset data pointers (+1 for index)
4. `baseoffsets[nop+1]` - base offsets
5. `operands[nop]` - PyArrayObject pointers
6. `opitflags[nop]` - per-operand flags
7. `bufferdata` (if buffered)
8. `dataptrs[nop+1]` - current data pointers
9. `userptrs[nop+1]` - user-visible pointers
10. `axisdata[ndim]` - per-axis data structures

### NumSharp: Fixed + Dynamic Structure
```csharp
struct NpyIterState {
    uint ItFlags;
    int NDim, NOp, MaskOp;
    long IterSize, IterIndex, IterStart, IterEnd;

    // Dynamic (allocated via NativeMemory)
    sbyte* Perm;      // size = NDim
    long* Shape;      // size = NDim
    long* Coords;     // size = NDim
    long* Strides;    // size = NDim * NOp

    // Fixed arrays (MaxOperands = 8)
    fixed long DataPtrs[8];
    fixed long ResetDataPtrs[8];
    // ... etc
}
```

### Key Difference: Per-Axis Data Structure

**NumPy uses `NpyIter_AxisData` per axis:**
```c
struct NpyIter_AxisData_tag {
    npy_intp shape, index;
    Py_intptr_t ad_flexdata;  // Contains strides for all operands + index stride
};
// Access: NAD_STRIDES(axisdata)[op] = strides[axis][op]
```

**NumSharp uses flat stride array:**
```csharp
// Strides[op * StridesNDim + axis] = strides[op][axis]
// Inverted layout: op-major vs axis-major
```

| Aspect | NumPy | NumSharp |
|--------|-------|----------|
| Stride layout | `[axis][op]` (axis-major) | `[op][axis]` (op-major) |
| Index stride | Stored with operand strides | Separate FlatIndex field |
| Per-axis index | `NAD_INDEX(axisdata)` | `Coords[axis]` |
| Per-axis shape | `NAD_SHAPE(axisdata)` | `Shape[axis]` |

---

## 2. Index Tracking Differences

### NumPy: Index as Extra "Operand"
NumPy tracks the flat index by storing it as an additional stride/pointer alongside operand data:

```c
#define NAD_NSTRIDES() ((nop) + ((itflags&NPY_ITFLAG_HASINDEX) ? 1 : 0))

// Index pointer is stored after operand pointers
npy_intp *NpyIter_GetIndexPtr(iter) {
    return (npy_intp*)(NpyIter_GetDataPtrArray(iter) + nop);
}

// Index strides are computed and stored in NAD_STRIDES(axisdata)[nop]
npyiter_compute_index_strides(iter, flags);
```

### NumSharp: Separate FlatIndex Field
NumSharp uses a dedicated `FlatIndex` field that's computed on demand:

```csharp
public long FlatIndex;
public bool IsCIndex;  // true for C-order, false for F-order

// Computed in ComputeFlatIndex() based on Coords
```

| Aspect | NumPy | NumSharp |
|--------|-------|----------|
| Storage | Extra entry in data pointer array | Separate field |
| Index stride | Pre-computed per axis | Computed from coords |
| Update method | Stride-based during iteration | Incremented or recomputed |
| Memory overhead | Per-axis stride storage | Single long field |

---

## 3. Coalescing Algorithm Differences

### NumPy Coalescing (lines 1644-1700 in nditer_api.c)
```c
void npyiter_coalesce_axes(NpyIter *iter) {
    // Clears IDENTPERM and HASMULTIINDEX flags
    NIT_ITFLAGS(iter) &= ~(NPY_ITFLAG_IDENTPERM|NPY_ITFLAG_HASMULTIINDEX);

    for (idim = 0; idim < ndim-1; ++idim) {
        // Check if shape0*stride0 == stride1 for ALL strides (including index)
        for (istrides = 0; istrides < nstrides; ++istrides) {
            if (!((shape0 == 1 && strides0[istrides] == 0) ||
                  (shape1 == 1 && strides1[istrides] == 0)) &&
                 (strides0[istrides]*shape0 != strides1[istrides])) {
                can_coalesce = 0;
                break;
            }
        }
        // If coalescing, multiply shapes and take non-zero stride
    }
    // Update ndim, reset perm to identity
}
```

### NumSharp Coalescing (NpyIterCoalescing.cs)
```csharp
public static void CoalesceAxes(ref NpyIterState state) {
    // Similar logic but:
    // 1. Operates on separate Shape/Strides arrays
    // 2. Doesn't handle index stride (separate FlatIndex)
    // 3. Clears HASMULTIINDEX, sets IDENTPERM
}
```

| Aspect | NumPy | NumSharp |
|--------|-------|----------|
| Index stride handling | Coalesces index stride too | Index handled separately |
| Perm reset | Resets to identity after coalescing | Same |
| When called | After axis ordering, before buffer setup | Same timing |

---

## 4. Axis Ordering Differences

### NumPy: Best Axis Ordering
NumPy has sophisticated axis ordering in `npyiter_find_best_axis_ordering()`:
1. Sorts axes by absolute stride magnitude
2. Handles negative strides (flipped axes)
3. Uses permutation array to track original axis mapping
4. Considers all operands when determining order

### NumSharp: Stride-Based Reordering
```csharp
public static void ReorderAxesForCoalescing(ref NpyIterState state, NPY_ORDER order) {
    // Simple insertion sort by minimum absolute stride across operands
    // No negative stride handling (separate from axis order)
}
```

| Aspect | NumPy | NumSharp |
|--------|-------|----------|
| Negative strides | Handled via `npyiter_flip_negative_strides()` | Not handled in reordering |
| Sort algorithm | Complex multi-criteria | Simple insertion sort |
| C/F order | Forces specific axis ordering | Forces via order parameter |

---

## 5. Missing NumPy Features in NumSharp

### 5.1 RemoveAxis()
NumPy allows removing an axis from iteration dynamically:
```c
int NpyIter_RemoveAxis(NpyIter *iter, int axis);
```
**NumSharp status:** NOT IMPLEMENTED

### 5.2 RemoveMultiIndex()
NumPy allows removing multi-index tracking and coalescing afterwards:
```c
int NpyIter_RemoveMultiIndex(NpyIter *iter);
```
**NumSharp status:** NOT IMPLEMENTED

### 5.3 GotoIndex() with Index Tracking
NumPy's `GotoIndex()` converts flat index to multi-index using pre-computed index strides:
```c
int NpyIter_GotoIndex(NpyIter *iter, npy_intp flat_index);
// Uses NAD_STRIDES(axisdata)[nop] to decompose flat_index
```
**NumSharp status:** NOT IMPLEMENTED (has GotoIterIndex but not GotoIndex)

### 5.4 GetIterView()
NumPy provides array views with iterator's internal axis ordering:
```c
PyArrayObject *NpyIter_GetIterView(NpyIter *iter, npy_intp i);
```
**NumSharp status:** NOT IMPLEMENTED

### 5.5 IsFirstVisit()
For reduction operations, NumPy tracks whether each element is being visited for the first time:
```c
npy_bool NpyIter_IsFirstVisit(NpyIter *iter, int iop);
```
**NumSharp status:** NOT IMPLEMENTED

### 5.6 Reduction Support
NumPy has full reduction support with:
- `NPY_ITFLAG_REDUCE` flag
- `NPY_OP_ITFLAG_REDUCE` per-operand flag
- `NBF_REDUCE_POS`, `NBF_REDUCE_OUTERSIZE`, `NBF_OUTERDIM` in buffer data
- Special reduce loop handling

**NumSharp status:** PARTIAL (flags exist but not fully implemented)

### 5.7 Cast/Type Conversion During Iteration
NumPy supports automatic type casting via `NpyIter_TransferInfo`:
```c
struct NpyIter_TransferInfo_tag {
    NPY_cast_info read;   // For copying array -> buffer
    NPY_cast_info write;  // For copying buffer -> array
    NPY_traverse_info clear;
};
```
**NumSharp status:** NOT IMPLEMENTED (only same-type copy)

### 5.8 Object Array Support
NumPy tracks reference counting for object arrays:
- `NPY_ITEM_REFCOUNT` flag
- `NpyIter_IterationNeedsAPI()` for GIL requirements

**NumSharp status:** N/A (no object arrays in NumSharp)

---

## 6. Flag Bit Position Differences

### NumPy Internal Flags (bits 0-12)
```c
#define NPY_ITFLAG_IDENTPERM     (1 << 0)   // 0x0001
#define NPY_ITFLAG_NEGPERM       (1 << 1)   // 0x0002
#define NPY_ITFLAG_HASINDEX      (1 << 2)   // 0x0004
#define NPY_ITFLAG_HASMULTIINDEX (1 << 3)   // 0x0008
#define NPY_ITFLAG_FORCEDORDER   (1 << 4)   // 0x0010
#define NPY_ITFLAG_EXLOOP        (1 << 5)   // 0x0020
#define NPY_ITFLAG_RANGE         (1 << 6)   // 0x0040
#define NPY_ITFLAG_BUFFER        (1 << 7)   // 0x0080
#define NPY_ITFLAG_GROWINNER     (1 << 8)   // 0x0100
#define NPY_ITFLAG_ONEITERATION  (1 << 9)   // 0x0200
#define NPY_ITFLAG_DELAYBUF      (1 << 10)  // 0x0400
#define NPY_ITFLAG_REDUCE        (1 << 11)  // 0x0800
#define NPY_ITFLAG_REUSE_REDUCE_LOOPS (1 << 12) // 0x1000
```

### NumSharp Internal Flags (bits 0-7 legacy, 8-15 NumPy-aligned)
```csharp
// Legacy (bits 0-2)
SourceBroadcast = 1 << 0,
SourceContiguous = 1 << 1,
DestinationContiguous = 1 << 2,

// NumPy-equivalent (bits 8-15, shifted by 8)
IDENTPERM = 0x0001 << 8,      // 0x0100
NEGPERM = 0x0002 << 8,        // 0x0200
HASINDEX = 0x0004 << 8,       // 0x0400
// etc.
```

**Impact:** Flag values don't match between implementations. Cannot directly compare or serialize.

---

## 7. Buffer Management Differences

### NumPy Buffer Data Structure
```c
struct NpyIter_BufferData_tag {
    npy_intp buffersize, size, bufiterend,
             reduce_pos, coresize, outersize, coreoffset, outerdim;
    Py_intptr_t bd_flexdata;  // strides, outerptrs, buffers, transferinfo
};
```

### NumSharp Buffer Fields
```csharp
public long BufferSize;
public long BufIterEnd;
public fixed long Buffers[MaxOperands];
public fixed long BufStrides[MaxOperands];
```

| Aspect | NumPy | NumSharp |
|--------|-------|----------|
| Reduce support | Full (pos, outersize, outerdim) | Not implemented |
| Transfer functions | NPY_cast_info per operand | Type switch dispatch |
| Stride storage | In bd_flexdata | Fixed array |
| Core/outer loop | Separate coresize, outersize | Not implemented |

---

## 8. MaxDims and MaxOperands

| Limit | NumPy | NumSharp |
|-------|-------|----------|
| MaxDims | 64 (NPY_MAXDIMS) | Unlimited (dynamic allocation) |
| MaxOperands | Unlimited | 8 (MaxOperands) |
| AxisData size | Variable per ndim | N/A (uses separate arrays) |

---

## 9. API Completeness Matrix

| API Function | NumPy | NumSharp | Notes |
|--------------|-------|----------|-------|
| `New()` | Yes | Yes | |
| `MultiNew()` | Yes | Yes | |
| `AdvancedNew()` | Yes | Yes | |
| `Reset()` | Yes | Yes | |
| `ResetBasePointers()` | Yes | No | |
| `ResetToIterIndexRange()` | Yes | Yes | |
| `GotoMultiIndex()` | Yes | Yes | |
| `GotoIndex()` | Yes | No | Uses flat index |
| `GotoIterIndex()` | Yes | Yes | |
| `GetIterIndex()` | Yes | Yes | |
| `GetMultiIndex()` | Yes | Yes | |
| `RemoveAxis()` | Yes | No | |
| `RemoveMultiIndex()` | Yes | No | |
| `EnableExternalLoop()` | Yes | Yes | |
| `GetNDim()` | Yes | Yes | Property |
| `GetNOp()` | Yes | Yes | Property |
| `GetIterSize()` | Yes | Yes | Property |
| `GetIterIndexRange()` | Yes | Yes | |
| `GetShape()` | Yes | No | |
| `GetDescrArray()` | Yes | Yes | |
| `GetOperandArray()` | Yes | Yes | |
| `GetIterView()` | Yes | No | |
| `GetDataPtrArray()` | Yes | Yes | |
| `GetInitialDataPtrArray()` | Yes | No | |
| `GetIndexPtr()` | Yes | No | Uses GetIndex() |
| `GetInnerStrideArray()` | Yes | Yes | |
| `GetInnerLoopSizePtr()` | Yes | Yes | |
| `GetInnerFixedStrideArray()` | Yes | No | |
| `GetBufferSize()` | Yes | No | Property |
| `HasDelayedBufAlloc()` | Yes | No | |
| `HasExternalLoop()` | Yes | Yes | Property |
| `HasMultiIndex()` | Yes | Yes | Property |
| `HasIndex()` | Yes | Yes | Property |
| `RequiresBuffering()` | Yes | Yes | Property |
| `IsBuffered()` | Yes | Yes | |
| `IsGrowInner()` | Yes | Yes | Property |
| `IsFirstVisit()` | Yes | No | |
| `IterationNeedsAPI()` | Yes | No | N/A (no GIL) |
| `Deallocate()` | Yes | Yes | Dispose pattern |
| `Copy()` | Yes | No | |
| `DebugPrint()` | Yes | No | |

---

## 10. Behavioral Differences Summary

| Behavior | NumPy | NumSharp |
|----------|-------|----------|
| Coalescing trigger | `ndim > 1 && !HASMULTIINDEX` | Same |
| Axis reordering | Before coalescing | Same |
| Negative stride handling | Via permutation with negative entries | Not fully implemented |
| Index computation | Pre-computed strides | On-demand from coords |
| Buffer GROWINNER | Grows inner loop across axes | Implemented but simpler |
| Reduction iteration | Double-loop with reduce_pos | Basic support via op_axes and IsFirstVisit |
| Type casting | Via NPY_cast_info | Full support via BUFFERED + op_dtypes |
| Error handling | Python exceptions | C# exceptions |

---

## 11. Implementation Status (Updated 2026-04-16)

### Implemented
- **RemoveMultiIndex()** - Enable coalescing after construction (calls ReorderAxes + Coalesce)
- **RemoveAxis()** - Dynamic axis removal with itersize recalculation
- **Finished property** - Check if iteration is complete
- **Shape property** - Get current iterator shape after coalescing
- **IterRange property** - Get (Start, End) tuple
- **Iternext()** - Advance and return whether more elements exist
- **GetValue<T>() / SetValue<T>()** - Type-safe value access
- **GetDataPtr()** - Raw pointer access to current operand data

### All Major Features Complete

NpyIter now has full NumPy parity for the features needed by NumSharp operations.

### Recently Completed (2026-04-16)

- **Reduction support** - Basic reduction via op_axes with -1 entries. REDUCE_OK flag validation
  for READWRITE operands. IsFirstVisit(operand) checks if current element is first visit
  (for initialization). IsReduction and IsOperandReduction() properties. REDUCE flags set
  on iterator and operands. Proper op_axes handling for stride calculation. 7 new tests.
- **Cast support** - Full NumPy parity: Type conversion during buffered iteration via
  BUFFERED flag, op_dtypes parameter, and COMMON_DTYPE flag. Supports all casting rules
  (no_casting, equiv, safe, same_kind, unsafe). NpyIterCasting validates casts and performs
  type conversion via double intermediate. Fixed critical bug: Dispose was freeing aligned
  buffers with wrong function (Free vs AlignedFree). 13 new NumPy parity tests.
- **GetIterView()** - Returns NDArray view with iterator's internal axes ordering. A C-order
  iteration of the view matches the iterator's iteration order. Not available when buffering
  is enabled. 8 new NumPy parity tests.
- **Negative stride flipping** - Full NumPy parity: FlipNegativeStrides() negates all-negative
  axes, adjusts base pointers, marks axes with negative Perm entries, sets NEGPERM flag.
  GetMultiIndex/GotoMultiIndex/GotoIndex/ComputeFlatIndex all handle NEGPERM correctly.
  DONT_NEGATE_STRIDES flag supported. 13 new NumPy parity tests.
- **Copy()** - Create independent copy of iterator at current position
- **GotoIndex()** - Jump to flat C/F index position (full NumPy parity)
- **ComputeFlatIndex fix** - Uses Perm to compute index in original coordinate order
- **F-order with MULTI_INDEX** - Full NumPy parity: first axis changes fastest
- **K-order with MULTI_INDEX** - Full NumPy parity: follows memory layout (smallest stride innermost)
- **Axis permutation tracking** - Perm array correctly maps internal to original coordinates
- **forCoalescing parameter** - Conditional axis sorting for coalescing vs iteration
