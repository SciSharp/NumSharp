# Int64 Migration Issues

This document tracks potential issues when migrating from int32-based indexing to int64 for >2GB array support.

## Status Legend
- `[ ]` Not started
- `[~]` In progress
- `[x]` Fixed
- `[!]` Won't fix (documented limitation)

---

## CRITICAL - Would fail for >2GB arrays

### Span<T> Constructor Limitation
.NET `Span<T>` constructor takes `int length`, limiting to ~2 billion elements.

| Status | File | Line | Code |
|--------|------|------|------|
| [x] | `UnmanagedStorage.cs` | 188 | `new Span<T>(Address, (int)Count)` | Has overflow check, throws for >2GB |
| [x] | `ArraySlice\`1.cs` | 333 | `new Span<T1>(VoidAddress, (int)Count)` | Has overflow check, throws for >2GB |
| [x] | `SimdMatMul.cs` | 48 | `new Span<float>(C, (int)outputSize).Clear()` | Has overflow check, falls back to loop for >2GB |

**Mitigation**: Has overflow checks - throws or falls back to loop-based iteration for >2GB.

### Explicit Size Truncation

| Status | File | Line | Code | Notes |
|--------|------|------|------|-------|
| [x] | `Shape.cs` | 1101 | `return (int)shape.Size` | Has overflow check, explicit operator throws for >2GB |
| [x] | `np.random.choice.cs` | 23 | `int arrSize = (int)a.size` | Has overflow check at line 21-22 |

### Array Allocation with Truncated Size

| Status | File | Line | Code |
|--------|------|------|------|
| [x] | `UnmanagedStorage.cs` | 1467 | `new T[Shape.Size]` | Added overflow check - ToArray() returns managed T[], platform limitation |

### Unsafe.InitBlockUnaligned Limitation
`Unsafe.InitBlockUnaligned` takes `uint` for byte count, limiting to 4GB.

| Status | File | Line | Code |
|--------|------|------|------|
| [x] | `UnmanagedMemoryBlock\`1.cs` | 597 | `Unsafe.InitBlockUnaligned(..., (uint)Count)` | Only called when Count < uint.MaxValue |
| [x] | `UnmanagedMemoryBlock\`1.cs` | 655 | `Unsafe.InitBlockUnaligned(..., (uint)count)` | Only called when Count < uint.MaxValue |
| [x] | `ArraySlice\`1.cs` | 183 | `Unsafe.InitBlockUnaligned(..., byteLen)` | Has check, falls back to loop for >4GB |

---

## HIGH - Stride/Offset Truncation

### IL Emission - Stride as int32

| Status | File | Line | Code | Notes |
|--------|------|------|------|-------|
| [x] | `Reduction.Axis.Simd.cs` | 392 | `int strideInt = (int)stride` | Added stride check at dispatch; falls back to scalar for stride > int.MaxValue |
| [x] | `Reduction.Axis.Simd.cs` | 433 | `int strideInt = (int)stride` | AVX2 gather requires int32 indices - hardware limitation |

### ArgMax/ArgMin Return Type

| Status | File | Line | Code | Notes |
|--------|------|------|------|-------|
| [ ] | `Reduction.cs` | 653 | `Conv_I4` | NumPy returns int64 for argmax/argmin |
| [ ] | `DefaultEngine.ReductionOp.cs` | 237-247 | `ExecuteElementReduction<int>` | Comment says int64 but code uses int |
| [ ] | `DefaultEngine.ReductionOp.cs` | 270-280 | `ExecuteElementReduction<int>` | Comment says int64 but code uses int |

---

## HIGH - ArrayConvert.cs Loop Counters

40+ instances of `for (int i = 0; i < length; i++)` in type conversion loops.
These iterate over array length which could exceed int.MaxValue.

| Status | File | Lines | Pattern |
|--------|------|-------|---------|
| [ ] | `ArrayConvert.cs` | 1168-1837+ | `for (int i = 0; i < length; i++) output[i] = Converts.To...` |

**Fix needed**: Change to `for (long i = 0; i < length; i++)` and ensure output array uses unmanaged allocation.

---

## HIGH - reshape with int[] Parameters

| Status | File | Line | Signature |
|--------|------|------|-----------|
| [ ] | `NdArray.ReShape.cs` | 51 | `reshape(int[] shape)` |
| [ ] | `NdArray.ReShape.cs` | 116 | `reshape_unsafe(int[] shape)` |
| [ ] | `NdArray\`1.ReShape.cs` | 41 | `reshape(int[] shape)` |
| [ ] | `NdArray\`1.ReShape.cs` | 97 | `reshape_unsafe(int[] shape)` |
| [ ] | `np.reshape.cs` | 12 | `reshape(NDArray nd, params int[] shape)` |

**Note**: Shape now uses `long[]` internally. These int[] overloads need long[] counterparts.

---

## MEDIUM - Axis Reduction Output Allocation

These allocate output arrays using truncated sizes:

| Status | File | Line | Code |
|--------|------|------|------|
| [x] | `np.nanvar.cs` | - | Changed to use unmanaged NDArray allocation |
| [x] | `np.nanstd.cs` | - | Changed to use unmanaged NDArray allocation |
| [x] | `np.nanmean.cs` | - | Changed to use unmanaged NDArray allocation |

**Fix**: Replaced managed array allocation with `new NDArray(NPTypeCode, Shape)` which uses unmanaged memory.

---

## LOW - .NET Platform Limitations

### String Length Limitation
.NET strings have `int` length. These would fail for char arrays >2GB:

| Status | File | Line | Code |
|--------|------|------|------|
| [!] | `NDArray.String.cs` | 33 | `new string((char*)arr.Address, 0, (int)arr.size)` |
| [!] | `NDArray.String.cs` | 87 | `new string('\0', (int)src.Count)` |
| [!] | `NDArray.String.cs` | 100 | `new string((char*)src.Address, 0, (int)src.Count)` |

**Note**: .NET string limitation, not NumSharp issue.

---

## HIGH - Public API with int Parameters

These public APIs now have `long` overloads (int versions kept for backward compatibility):

### NDArray Constructors

| Status | File | Line | Signature |
|--------|------|------|-----------|
| [x] | `NDArray.cs` | - | `NDArray(Type dtype, long size)` added |
| [x] | `NDArray.cs` | - | `NDArray(Type dtype, long size, bool fillZeros)` added |
| [x] | `NDArray.cs` | - | `NDArray(NPTypeCode dtype, long size)` already exists |
| [x] | `NDArray.cs` | - | `NDArray(NPTypeCode dtype, long size, bool fillZeros)` already exists |
| [x] | `NDArray\`1.cs` | - | `NDArray(long size, bool fillZeros)` already exists |
| [x] | `NDArray\`1.cs` | - | `NDArray(long size)` already exists |

### ArraySlice/MemoryBlock Allocation

| Status | File | Line | Signature |
|--------|------|------|-----------|
| [x] | `ArraySlice.cs` | - | int overloads delegate to long overloads (already done) |
| [x] | `UnmanagedMemoryBlock.cs` | - | int overloads delegate to long overloads (already done) |

### Array Creation APIs

| Status | File | Line | Signature |
|--------|------|------|-----------|
| [ ] | `np.array.cs` | 105 | `array<T>(IEnumerable<T>, int size)` |
| [ ] | `Arrays.cs` | 384 | `Create(Type, int length)` |
| [ ] | `Arrays.cs` | 414 | `Create<T>(int length)` |
| [ ] | `Arrays.cs` | 425 | `Create(NPTypeCode, int length)` |

---

## MEDIUM - for Loops with int Counter

These use `int` loop counter over potentially large sizes:

| Status | File | Line | Pattern |
|--------|------|------|---------|
| [!] | `np.frombuffer.cs` | 15+ | Input is byte[] which is int-limited - OK as-is |
| [ ] | `ArrayConvert.cs` | 1168+ | `for (int i = 0; i < length; i++)` (30+ instances) |
| [x] | `UnmanagedMemoryBlock\`1.cs` | 755 | Changed to `for (long i = 0; i < Count; i++)` |
| [x] | `UnmanagedMemoryBlock\`1.cs` | 768 | Fixed Contains() loop counter to long |
| [x] | `UnmanagedMemoryBlock\`1.cs` | 780 | Fixed CopyTo() loop counter to long |

---

## MEDIUM - stackalloc with int Cast

| Status | File | Line | Code |
|--------|------|------|------|
| [ ] | `Hashset\`1.cs` | 1374 | `stackalloc long[(int)intArrayLength]` |
| [ ] | `Hashset\`1.cs` | 1473 | `stackalloc long[(int)intArrayLength]` |
| [ ] | `Hashset\`1.cs` | 1631 | `stackalloc long[(int)intArrayLength]` |

---

## MEDIUM - arange/linspace with int Parameters

| Status | File | Line | Signature |
|--------|------|------|-----------|
| [ ] | `np.arange.cs` | 234 | `arange(int stop)` |
| [ ] | `np.arange.cs` | 248 | `arange(int start, int stop, int step = 1)` |
| [ ] | `np.linspace.cs` | 21,37,54,71 | `linspace(..., int num, ...)` |

**Note**: These control output size. `num` should support long for >2B element arrays.

---

## LOW - Hashset/Dictionary Count Property

| Status | File | Line | Property |
|--------|------|------|----------|
| [!] | `Hashset\`1.cs` | 361 | `public int Count` |
| [!] | `ConcurrentHashset\`1.cs` | 310 | `public int Count` |

**Note**: .NET collection pattern uses int Count. Would require custom interface.

---

## OK - Confirmed Correct as int32

These use int32 correctly and don't need migration:

| Category | Examples | Reason |
|----------|----------|--------|
| Dimension counter | `locD` in strided loops | ndim is always small (<32) |
| Vector count | `vectorCount` (4-64) | SIMD vector size |
| Tile sizes | KC, MC, NR in SimdMatMul | Small constants for cache optimization |
| Type conversion | `Conv_I4`/`Conv_U4` in EmitConvertTo | Converting element values, not indices |
| Bit widths | `bitWidth` in Shift.cs | 8, 16, 32, 64 |
| Shift amounts | Constants like 15, 31, 63 | IL shift ops require int32 |
| Shape/stride loops | `for (int i = 0; i < dims.Length; i++)` | Iterating over dimensions (small) |
| Coordinate arrays | `for (int i = 0; i < indices.Length; i++)` | Iterating over coordinates (small) |
| ndim parameter | `int ndim` in method signatures | Number of dimensions is small |

---

## Search Patterns Used

```bash
# Explicit casts to int for size/count/index
grep -rn "(int)(.*size|.*count|.*length|.*offset|.*index)" --include="*.cs"

# Span constructor with int cast
grep -rn "new (Span|Memory)<.*>.*\(int\)" --include="*.cs"

# IL locals declared as int for potential indices
grep -rn "DeclareLocal(typeof(int)).*//.*[Ii]ndex\|[Oo]ffset\|[Cc]ount" --include="*.cs"

# Conv_I4 in IL emission (potential truncation)
grep -rn "Conv_I4\|Conv_U4" --include="*.cs"

# int variable assigned from cast
grep -rn "int\s+\w+\s*=\s*\(int\)" --include="*.cs"

# Ldc_I4 with size variables
grep -rn "Ldc_I4, (inputSize|outputSize|elementSize)" --include="*.cs"
```

---

## Additional Patterns to Search

```bash
# Array indexer with potential overflow
grep -rn "\[\s*(int)\s*" --include="*.cs"

# for loops with int counter over size
grep -rn "for\s*\(\s*int\s+\w+.*[Ss]ize\|[Ll]ength\|[Cc]ount" --include="*.cs"

# Multiplication that could overflow
grep -rn "\*\s*(int)" --include="*.cs"

# Method parameters that should be long
grep -rn "int\s+(size|count|length|offset|index|stride)" --include="*.cs"

# Unsafe.InitBlockUnaligned with uint cast
grep -rn "InitBlockUnaligned.*\(uint\)" --include="*.cs"

# Buffer.MemoryCopy (takes long, but check callers)
grep -rn "Buffer\.(BlockCopy|MemoryCopy)" --include="*.cs"

# Public API with int size/count parameters
grep -rn "public.*\(.*int (size|count|length|offset)\b" --include="*.cs"

# checked/unchecked casts (potential overflow points)
grep -rn "checked\s*\(|unchecked\s*\(" --include="*.cs"
```

---

## Summary Statistics

| Category | Count | Status | Priority |
|----------|-------|--------|----------|
| Span<T> limitation | 3 | DONE | CRITICAL |
| Explicit truncation | 2 | DONE | CRITICAL |
| Array allocation | 1 | DONE | CRITICAL |
| InitBlockUnaligned | 3 | DONE | CRITICAL |
| Stride truncation | 2 | DONE | HIGH |
| ArgMax/ArgMin return | 3 | TODO | HIGH |
| ArrayConvert.cs loops | 40+ | TODO | HIGH |
| reshape int[] params | 5 | TODO | HIGH |
| Public API (int params) | 14+ | MOSTLY DONE | HIGH |
| Output allocation | 6 | DONE | MEDIUM |
| arange/linspace params | 4 | TODO | MEDIUM |
| stackalloc | 3 | TODO | MEDIUM |
| String limitation | 3 | WON'T FIX | LOW |
| Hashset Count | 2 | WON'T FIX | LOW |

---

## Recommended Fix Order

1. **Phase 1 - Critical path**: ~~Fix Span<T>, array allocation, InitBlockUnaligned~~ DONE
2. **Phase 2 - Public API**: ~~Add long overloads to constructors and Allocate methods~~ MOSTLY DONE
3. **Phase 3 - IL emission**: ~~Fix stride truncation~~, argmax/argmin return type
4. **Phase 4 - reshape**: Add long[] overloads to reshape methods
5. **Phase 5 - ArrayConvert**: Fix 40+ loop counters, use unmanaged allocation
6. **Phase 6 - Loops**: Convert remaining int loop counters to long where needed
7. **Document**: String/Hashset limitations as known platform constraints

---

## Files Already Using long Correctly

These files are confirmed to use long-based indexing:

| File | Notes |
|------|-------|
| `ValueCoordinatesIncrementor.cs` | Uses `long[]` for dimensions and Index |
| `Shape.cs` | Internal dimensions/strides are `long[]` |
| `UnmanagedMemoryBlock\`1.cs` | Count is `long`, address arithmetic uses long |
| `ArraySlice\`1.cs` | Count is `long`, indexers use long |
| `SimdMatMul.cs` | Loop variables are long, pointer arithmetic uses long |
