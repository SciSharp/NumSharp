# NpyIter NumPy Parity Analysis

**Source of Truth:** `numpy/_core/src/multiarray/nditer_impl.h`, `nditer_constr.c`, `nditer_api.c`

---

## Data Structures

### NpyIter_InternalOnly (NumPy)

```c
struct NpyIter_InternalOnly {
    npy_uint32 itflags;
    npy_uint8 ndim;
    int nop, maskop;
    npy_intp itersize, iterstart, iterend;
    npy_intp iterindex;
    char iter_flexdata[];  // Variable-length: perm, dtypes, resetdataptr, baseoffsets, etc.
};
```

### NpyIterState (NumSharp)

| Field | NumPy | NumSharp | Parity |
|-------|-------|----------|--------|
| `itflags` | `npy_uint32` | `uint ItFlags` | ✅ Match |
| `ndim` | `npy_uint8` | `int NDim` | ✅ Match (wider type OK) |
| `nop` | `int` | `int NOp` | ✅ Match |
| `maskop` | `int` | `int MaskOp` | ✅ Match |
| `itersize` | `npy_intp` | `long IterSize` | ✅ Match |
| `iterstart` | `npy_intp` | `long IterStart` | ✅ Match |
| `iterend` | `npy_intp` | `long IterEnd` | ✅ Match |
| `iterindex` | `npy_intp` | `long IterIndex` | ✅ Match |
| `perm[]` | Variable in flexdata | `fixed sbyte Perm[32]` | ⚠️ Fixed size (32 vs NPY_MAXDIMS=64) |
| `dtypes[]` | Variable in flexdata | `fixed byte OpDTypes[8]` | ⚠️ NPTypeCode enum vs PyArray_Descr* |
| `resetdataptr[]` | Variable in flexdata | `fixed long ResetDataPtrs[8]` | ✅ Match |
| `baseoffsets[]` | Variable in flexdata | `fixed long BaseOffsets[8]` | ✅ Match |
| `operands[]` | Variable in flexdata | `NDArray[]? _operands` (in NpyIterRef) | ✅ Match |
| `opitflags[]` | Variable in flexdata | `fixed ushort OpItFlags[8]` | ✅ Match |
| `dataptrs[]` | Variable in flexdata | `fixed long DataPtrs[8]` | ✅ Match |
| `bufferdata` | Conditional | `BufferSize`, `BufIterEnd`, `Buffers[]` | ✅ Match |
| `axisdata[]` | Per-axis struct | Flattened into `Shape[]`, `Coords[]`, `Strides[]` | ⚠️ Different layout |

**Assessment:** Core fields match. NumSharp uses fixed-size arrays (MaxDims=32, MaxOperands=8) vs NumPy's variable-length flexdata. This limits NumSharp to 32 dimensions and 8 operands max.

---

## Iterator Flags

### NPY_ITFLAG_* (NumPy) vs NpyIterFlags (NumSharp)

| Flag | NumPy Value | NumSharp Value | Parity |
|------|-------------|----------------|--------|
| `IDENTPERM` | `1 << 0` | `0x0100` | ⚠️ Different bit position |
| `NEGPERM` | `1 << 1` | `0x0200` | ⚠️ Different bit position |
| `HASINDEX` | `1 << 2` | `0x0400` | ⚠️ Different bit position |
| `HASMULTIINDEX` | `1 << 3` | `0x0800` | ⚠️ Different bit position |
| `FORCEDORDER` | `1 << 4` | `0x1000` | ⚠️ Different bit position |
| `EXLOOP` | `1 << 5` | `0x2000` | ⚠️ Different bit position |
| `RANGE` | `1 << 6` | `0x4000` | ⚠️ Different bit position |
| `BUFFER` | `1 << 7` | `0x8000` | ⚠️ Different bit position |
| `GROWINNER` | `1 << 8` | `0x010000` | ⚠️ Different bit position |
| `ONEITERATION` | `1 << 9` | `0x020000` | ⚠️ Different bit position |
| `DELAYBUF` | `1 << 10` | `0x040000` | ⚠️ Different bit position |
| `REDUCE` | `1 << 11` | `0x080000` | ⚠️ Different bit position |
| `REUSE_REDUCE_LOOPS` | `1 << 12` | `0x100000` | ⚠️ Different bit position |

**Assessment:** Flag values differ but functionality is equivalent. NumSharp reserves lower bits for legacy compatibility flags.

### NPY_OP_ITFLAG_* (NumPy) vs NpyIterOpFlags (NumSharp)

| Flag | NumPy Value | NumSharp Value | Parity |
|------|-------------|----------------|--------|
| `WRITE` | `0x0001` | `0x0001` | ✅ Match |
| `READ` | `0x0002` | `0x0002` | ✅ Match |
| `CAST` | `0x0004` | `0x0004` | ✅ Match |
| `BUFNEVER` | `0x0008` | `0x0008` | ✅ Match |
| `BUF_SINGLESTRIDE` | `0x0010` | `0x0010` | ✅ Match |
| `REDUCE` | `0x0020` | `0x0020` | ✅ Match |
| `VIRTUAL` | `0x0040` | `0x0040` | ✅ Match |
| `WRITEMASKED` | `0x0080` | `0x0080` | ✅ Match |
| `BUF_REUSABLE` | `0x0100` | `0x0100` | ✅ Match |
| `FORCECOPY` | `0x0200` | `0x0200` | ✅ Match |
| `HAS_WRITEBACK` | `0x0400` | `0x0400` | ✅ Match |
| `CONTIG` | `0x0800` | `0x0800` | ✅ Match |

**Assessment:** Per-operand flags match exactly.

---

## Factory Methods

### NumPy API

| Function | Parameters | NumSharp Equivalent | Parity |
|----------|------------|---------------------|--------|
| `NpyIter_New` | `op, flags, order, casting, dtype` | `NpyIterRef.New()` | ✅ Implemented |
| `NpyIter_MultiNew` | `nop, op[], flags, order, casting, op_flags[], op_dtypes[]` | `NpyIterRef.MultiNew()` | ✅ Implemented |
| `NpyIter_AdvancedNew` | `nop, op[], flags, order, casting, op_flags[], op_dtypes[], oa_ndim, op_axes[][], itershape[], buffersize` | `NpyIterRef.AdvancedNew()` | ⚠️ Partial |

### AdvancedNew Parameters

| Parameter | NumPy | NumSharp | Status |
|-----------|-------|----------|--------|
| `nop` | int | int | ✅ |
| `op_in` | PyArrayObject** | NDArray[] | ✅ |
| `flags` | npy_uint32 | NpyIterGlobalFlags | ✅ |
| `order` | NPY_ORDER | NPY_ORDER | ✅ |
| `casting` | NPY_CASTING | NPY_CASTING | ✅ |
| `op_flags` | npy_uint32* | NpyIterPerOpFlags[] | ✅ |
| `op_request_dtypes` | PyArray_Descr** | NPTypeCode[]? | ⚠️ Simpler (no descr objects) |
| `oa_ndim` | int | int (not used) | ❌ Not implemented |
| `op_axes` | int** | int[][]? (not used) | ❌ Not implemented |
| `itershape` | npy_intp* | long[]? (not used) | ❌ Not implemented |
| `buffersize` | npy_intp | long | ✅ |

---

## API Methods

### Iteration Control

| NumPy Function | NumSharp Method | Status |
|----------------|-----------------|--------|
| `NpyIter_GetIterNext()` | `GetIterNext()` | ✅ Implemented |
| `NpyIter_GetDataPtrArray()` | `GetDataPtrArray()` | ✅ Implemented |
| `NpyIter_GetInnerStrideArray()` | `GetInnerStrideArray()` | ⚠️ Layout differs |
| `NpyIter_GetInnerLoopSizePtr()` | `GetInnerLoopSizePtr()` | ✅ Implemented |
| `NpyIter_GetIterSize()` | `IterSize` property | ✅ Implemented |
| `NpyIter_GetIterIndex()` | `IterIndex` property | ✅ Implemented |
| `NpyIter_GetNOp()` | `NOp` property | ✅ Implemented |
| `NpyIter_GetNDim()` | `NDim` property | ✅ Implemented |
| `NpyIter_Reset()` | `Reset()` | ✅ Implemented |
| `NpyIter_GotoIterIndex()` | `GotoIterIndex()` | ✅ Implemented |
| `NpyIter_GotoMultiIndex()` | - | ❌ Not implemented |
| `NpyIter_GetMultiIndexFunc()` | - | ❌ Not implemented |

### Configuration

| NumPy Function | NumSharp Method | Status |
|----------------|-----------------|--------|
| `NpyIter_RemoveAxis()` | `RemoveAxis()` | ⚠️ Partial (no perm handling) |
| `NpyIter_RemoveMultiIndex()` | - | ❌ Not implemented |
| `NpyIter_EnableExternalLoop()` | `EnableExternalLoop()` | ✅ Implemented |
| `NpyIter_IterationNeedsAPI()` | - | ❌ N/A (no Python API) |
| `NpyIter_RequiresBuffering()` | `RequiresBuffering` property | ✅ Implemented |

### Buffer Management

| NumPy Function | NumSharp Method | Status |
|----------------|-----------------|--------|
| `npyiter_allocate_buffers()` | `NpyIterBufferManager.AllocateBuffers()` | ✅ Implemented |
| `npyiter_copy_to_buffers()` | `CopyToBuffer<T>()` | ⚠️ Type-specific only |
| `npyiter_copy_from_buffers()` | `CopyFromBuffer<T>()` | ⚠️ Type-specific only |
| `npyiter_clear_buffers()` | `FreeBuffers()` | ✅ Implemented |

### Introspection

| NumPy Function | NumSharp Method | Status |
|----------------|-----------------|--------|
| `NpyIter_GetOperandArray()` | `GetOperandArray()` | ✅ Implemented |
| `NpyIter_GetDescrArray()` | `GetDescrArray()` | ✅ Implemented (returns NPTypeCode[]) |
| `NpyIter_GetShape()` | - | ❌ Not implemented |
| `NpyIter_GetReadFlags()` | `GetOpFlags()` | ✅ Via state |
| `NpyIter_GetWriteFlags()` | `GetOpFlags()` | ✅ Via state |

---

## Core Algorithms

### Axis Coalescing

| Aspect | NumPy | NumSharp | Parity |
|--------|-------|----------|--------|
| Algorithm | Merge adjacent axes with compatible strides | Same algorithm | ✅ Match |
| Condition | `(shape0==1 && stride0==0) || (shape1==1 && stride1==0) || (stride0*shape0==stride1)` | Same condition | ✅ Match |
| Per-operand check | Checks all operands + index stride | Checks all operands | ✅ Match |
| Updates perm | Resets to identity after coalescing | Resets to identity | ✅ Match |
| When called | After construction, before buffering | On EXTERNAL_LOOP flag | ⚠️ Different trigger |

### Broadcasting

| Aspect | NumPy | NumSharp | Parity |
|--------|-------|----------|--------|
| Shape calculation | Right-align, broadcast 1s | Same | ✅ Match |
| Stride mapping | stride=0 for broadcast dims | Same | ✅ Match |
| NO_BROADCAST flag | Prevents broadcasting for operand | Implemented | ✅ Match |
| Error handling | IncorrectShapeException equivalent | IncorrectShapeException | ✅ Match |

### GotoIterIndex

| Aspect | NumPy | NumSharp | Parity |
|--------|-------|----------|--------|
| Coordinate calculation | Divide-mod from innermost | Same | ✅ Match |
| Pointer update | Add coord * stride for each axis | Same | ✅ Match |
| Buffered mode | Updates buffer position | Not fully implemented | ⚠️ Partial |

---

## Feature Gaps (NumSharp Missing)

### Critical for Full Parity

1. **op_axes parameter**: Custom axis mapping for operands
2. **itershape parameter**: Explicit iteration shape
3. **Multi-index tracking**: `HASMULTIINDEX` flag and `GetMultiIndex()`
4. **Index tracking**: `HASINDEX` flag and flat index access
5. **Ranged iteration**: `RANGE` flag, `iterstart`/`iterend` control

### Nice to Have

1. **Axis removal with permutation**: Current `RemoveAxis()` doesn't handle permuted axes
2. **GROWINNER optimization**: Dynamic inner loop sizing
3. **Type casting during iteration**: `NPY_cast_info` integration
4. **Buffer reuse**: `BUF_REUSABLE` optimization

### Not Applicable to NumSharp

1. **Python API checks**: `IterationNeedsAPI()` - no GIL
2. **Reference counting**: Object arrays not supported
3. **Fortran order**: NumSharp is C-order only

---

## Behavioral Differences

### Coalescing Trigger

- **NumPy**: Always coalesces after construction unless `HASMULTIINDEX`
- **NumSharp**: Only coalesces when `EXTERNAL_LOOP` flag is set

**Impact**: NumSharp may have more dimensions than NumPy for same input without external loop.

### Stride Layout

- **NumPy**: Per-axis data in `NpyIter_AxisData` structs
- **NumSharp**: Flat arrays `Strides[op * MaxDims + axis]`

**Impact**: Different memory access patterns, but same logical data.

### Buffer Copy

- **NumPy**: Generic dtype-aware copy with cast support
- **NumSharp**: Type-specific `CopyToBuffer<T>` methods

**Impact**: No type casting during iteration (must match types beforehand).

---

## Recommendations

### Priority 1: Complete Core API

1. Implement `op_axes` parameter for axis remapping
2. Add `GotoMultiIndex()` for multi-index navigation
3. Fix coalescing to always run (match NumPy behavior)

### Priority 2: Buffer Improvements

1. Add dtype-aware buffer copy (not just type-specific)
2. Implement `GROWINNER` for dynamic sizing
3. Add buffer reuse tracking

### Priority 3: Advanced Features

1. Implement ranged iteration
2. Add index tracking
3. Support axis permutation in `RemoveAxis()`

---

## Test Coverage

| Feature | NumPy Tests | NumSharp Tests | Status |
|---------|-------------|----------------|--------|
| Single operand | Extensive | 4 tests | ⚠️ Need more |
| Multi operand | Extensive | 3 tests | ⚠️ Need more |
| Broadcasting | Extensive | 2 tests | ⚠️ Need more |
| Coalescing | Moderate | 1 test | ⚠️ Need more |
| Buffering | Extensive | 1 test | ⚠️ Need more |
| External loop | Moderate | 2 tests | ⚠️ Need more |
| Error cases | Extensive | 1 test | ⚠️ Need more |
