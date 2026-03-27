# Int64 Migration Issues

Tracking document for migrating NumSharp from int32-based indexing to int64 for >2GB array support.

## Status Legend
- `[x]` Fixed
- `[ ]` Not started
- `[!]` Won't fix (platform limitation)

---

## Phase 1: Critical Path (COMPLETE)

### Span<T> Constructor (.NET limitation: int length)

| Status | File | Code | Resolution |
|--------|------|------|------------|
| [x] | `UnmanagedStorage.cs:188` | `new Span<T>(Address, (int)Count)` | Overflow check throws |
| [x] | `ArraySlice`1.cs:333` | `new Span<T1>(VoidAddress, (int)Count)` | Overflow check throws |
| [x] | `SimdMatMul.cs:48` | `new Span<float>(C, (int)outputSize).Clear()` | Falls back to loop |

### InitBlockUnaligned (.NET limitation: uint byte count)

| Status | File | Code | Resolution |
|--------|------|------|------------|
| [x] | `UnmanagedMemoryBlock`1.cs:597` | `InitBlockUnaligned(..., (uint)Count)` | Guarded by Count < uint.MaxValue |
| [x] | `UnmanagedMemoryBlock`1.cs:655` | `InitBlockUnaligned(..., (uint)count)` | Guarded by Count < uint.MaxValue |
| [x] | `ArraySlice`1.cs:183` | `InitBlockUnaligned(..., byteLen)` | Falls back to loop |

### Managed Array Allocation

| Status | File | Code | Resolution |
|--------|------|------|------------|
| [x] | `UnmanagedStorage.cs:1472` | `new T[Shape.Size]` | Overflow check - ToArray() is platform-limited |

### Explicit Truncation

| Status | File | Code | Resolution |
|--------|------|------|------------|
| [x] | `Shape.cs:1101` | `return (int)shape.Size` | Overflow check in explicit operator |
| [x] | `np.random.choice.cs:23` | `int arrSize = (int)a.size` | Overflow check at line 21-22 |

---

## Phase 2: Public API (MOSTLY COMPLETE)

### NDArray Constructors

| Status | Signature | Notes |
|--------|-----------|-------|
| [x] | `NDArray(Type dtype, long size)` | Added |
| [x] | `NDArray(Type dtype, long size, bool fillZeros)` | Added |
| [x] | `NDArray(NPTypeCode dtype, long size)` | Already existed |
| [x] | `NDArray(NPTypeCode dtype, long size, bool fillZeros)` | Already existed |
| [x] | `NDArray<T>(long size)` | Already existed |
| [x] | `NDArray<T>(long size, bool fillZeros)` | Already existed |

### Allocation Methods

| Status | File | Notes |
|--------|------|-------|
| [x] | `ArraySlice.cs` | int overloads delegate to long |
| [x] | `UnmanagedMemoryBlock.cs` | int overloads delegate to long |

### Remaining API Issues

| Status | File | Signature |
|--------|------|-----------|
| [ ] | `Arrays.cs:384` | `Create(Type, int length)` |
| [ ] | `Arrays.cs:414` | `Create<T>(int length)` |
| [ ] | `Arrays.cs:425` | `Create(NPTypeCode, int length)` |
| [ ] | `np.array.cs:105` | `array<T>(IEnumerable<T>, int size)` |

---

## Phase 3: IL Emission

### Stride Truncation (COMPLETE)

| Status | File | Code | Resolution |
|--------|------|------|------------|
| [x] | `Reduction.Axis.Simd.cs:392` | `int strideInt = (int)stride` | Stride check at dispatch |
| [x] | `Reduction.Axis.Simd.cs:433` | `int strideInt = (int)stride` | AVX2 gather requires int32 (hardware) |

### ArgMax/ArgMin Return Type

NumPy returns int64 for argmax/argmin. NumSharp incorrectly returns int32.

| Status | File | Line | Code |
|--------|------|------|------|
| [ ] | `DefaultEngine.ReductionOp.cs` | 237-247 | `ExecuteElementReduction<int>(..., ReductionOp.ArgMax, ...)` |
| [ ] | `DefaultEngine.ReductionOp.cs` | 270-280 | `ExecuteElementReduction<int>(..., ReductionOp.ArgMin, ...)` |
| [ ] | `ILKernelGenerator.Reduction.cs` | 653 | `Conv_I4` truncates index to int32 |

---

## Phase 4: reshape Methods

Shape uses `long[]` internally but reshape APIs accept `int[]`:

| Status | File | Signature |
|--------|------|-----------|
| [ ] | `NdArray.ReShape.cs:51` | `reshape(int[] shape)` |
| [ ] | `NdArray.ReShape.cs:116` | `reshape_unsafe(int[] shape)` |
| [ ] | `NdArray`1.ReShape.cs:41` | `reshape(int[] shape)` |
| [ ] | `NdArray`1.ReShape.cs:97` | `reshape_unsafe(int[] shape)` |
| [ ] | `np.reshape.cs:12` | `reshape(NDArray nd, params int[] shape)` |

---

## Phase 5: ArrayConvert.cs

40+ type conversion loops use `int` counter over array length:

```csharp
for (int i = 0; i < length; i++) output[i] = Converts.To...(sourceArray[i]);
```

**Location**: `Utilities/ArrayConvert.cs` lines 1168-1837+

**Fix**: Change to `for (long i = 0; i < length; i++)` and ensure arrays use unmanaged allocation.

---

## Phase 6: Statistics Output Allocation (COMPLETE)

| Status | File | Resolution |
|--------|------|------------|
| [x] | `np.nanmean.cs` | Uses `new NDArray(NPTypeCode, Shape)` |
| [x] | `np.nanvar.cs` | Uses `new NDArray(NPTypeCode, Shape)` |
| [x] | `np.nanstd.cs` | Uses `new NDArray(NPTypeCode, Shape)` |

---

## Phase 7: Loop Counters

### Fixed

| Status | File | Location |
|--------|------|----------|
| [x] | `UnmanagedMemoryBlock`1.cs:755` | `GetEnumerator()` |
| [x] | `UnmanagedMemoryBlock`1.cs:768` | `Contains()` |
| [x] | `UnmanagedMemoryBlock`1.cs:780` | `CopyTo()` |

### Remaining

| Status | File | Pattern |
|--------|------|---------|
| [!] | `np.frombuffer.cs` | Input is `byte[]` which is int-limited |
| [ ] | `ArrayConvert.cs` | See Phase 5 |

---

## Phase 8: Array Creation Parameters

| Status | File | Signature | Notes |
|--------|------|-----------|-------|
| [ ] | `np.arange.cs:234` | `arange(int stop)` | Output can exceed int.MaxValue |
| [ ] | `np.arange.cs:248` | `arange(int start, int stop, int step)` | |
| [ ] | `np.linspace.cs:21` | `linspace(..., int num, ...)` | num controls output size |
| [ ] | `np.linspace.cs:37` | `linspace(..., int num, ...)` | |
| [ ] | `np.linspace.cs:54` | `linspace(..., int num, ...)` | |
| [ ] | `np.linspace.cs:71` | `linspace(..., int num, ...)` | |

---

## Won't Fix (.NET Platform Limitations)

### String Length

.NET `string` has `int` length. Cannot create strings >2GB.

| Status | File | Code |
|--------|------|------|
| [!] | `NDArray.String.cs:33` | `new string((char*)arr.Address, 0, (int)arr.size)` |
| [!] | `NDArray.String.cs:87` | `new string('\0', (int)src.Count)` |
| [!] | `NDArray.String.cs:100` | `new string((char*)src.Address, 0, (int)src.Count)` |

### Collection Count Property

.NET `ICollection<T>.Count` returns `int`. Custom interface would be needed.

| Status | File | Property |
|--------|------|----------|
| [!] | `Hashset`1.cs:361` | `public int Count` |
| [!] | `ConcurrentHashset`1.cs:310` | `public int Count` |

### stackalloc

`stackalloc` requires `int` size. Stack space is limited anyway.

| Status | File | Code |
|--------|------|------|
| [!] | `Hashset`1.cs:1374` | `stackalloc long[(int)intArrayLength]` |
| [!] | `Hashset`1.cs:1473` | `stackalloc long[(int)intArrayLength]` |
| [!] | `Hashset`1.cs:1631` | `stackalloc long[(int)intArrayLength]` |

---

## Confirmed Correct (No Changes Needed)

| Category | Examples | Reason |
|----------|----------|--------|
| Dimension counter | `locD`, `int d` | ndim is always small (<32) |
| Vector count | `vectorCount` | SIMD width (4-64) |
| Tile sizes | `KC`, `MC`, `NR` | Cache optimization constants |
| Element conversion | `Conv_I4` in `EmitConvertTo` | Converting values, not indices |
| Bit operations | `bitWidth`, shift amounts | Always small integers |
| Shape iteration | `for (int i = 0; i < dims.Length; i++)` | dims.Length is ndim (small) |
| ndim parameters | `int ndim`, `int axis` | Number of dimensions is small |

---

## Files Already Using long Correctly

| File | Evidence |
|------|----------|
| `ValueCoordinatesIncrementor.cs` | `long[] dimensions`, `long[] Index` |
| `Shape.cs` | `long[] dimensions`, `long[] strides`, `long size` |
| `UnmanagedMemoryBlock<T>` | `long Count`, `long BytesCount` |
| `ArraySlice<T>` | `long Count`, indexers use `long` |
| `SimdMatMul.cs` | `long M, N, K`, loop vars are `long` |
| `NDIterator.cs` | Uses `ValueCoordinatesIncrementor` with `long[]` |

---

## Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Critical path (Span, InitBlock, allocation) | **COMPLETE** |
| 2 | Public API constructors/allocators | **MOSTLY COMPLETE** |
| 3 | IL emission (stride, argmax/argmin) | Stride done, argmax TODO |
| 4 | reshape methods | TODO |
| 5 | ArrayConvert.cs loops | TODO |
| 6 | Statistics output allocation | **COMPLETE** |
| 7 | Loop counters | Partially done |
| 8 | Array creation parameters | TODO |

**Remaining work**: ~60 code locations across 4 phases.
