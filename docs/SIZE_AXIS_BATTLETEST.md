# NumPy Size/Axis Parameter Battle Test Results

**Date**: 2026-03-24
**NumPy Version**: 2.4.2
**Platform**: Windows 11 (64-bit)

This document captures exact NumPy behavior for `size` and `axis` parameters to ensure NumSharp matches 100%.

---

## Executive Summary

| Area | NumPy Behavior | NumSharp Current | Action Required |
|------|---------------|------------------|-----------------|
| Size input types | Accepts any integer type | `int[]` only | Accept `long`, validate |
| Axis input types | Accepts any integer type | `int?` only | OK (int sufficient) |
| Negative size | ValueError | Silently accepts? | Add validation |
| Float size | TypeError | Compiles (implicit cast) | Add overload rejection |
| Seed range | 0 to 2^32-1 only | `int` (allows negative) | Add validation |
| randint bounds | dtype-specific | Casts to `(int)` | Support int64 ranges |
| Return types | Python `int` | C# `int` | Already correct |

---

## Test 1: Size Parameter Type Acceptance

### Accepted Types
```python
# All of these work in NumPy:
np.random.rand(5)                    # Python int
np.random.rand(int(5))               # explicit int
np.random.rand(np.int8(5))           # numpy int8
np.random.rand(np.int16(5))          # numpy int16
np.random.rand(np.int32(5))          # numpy int32
np.random.rand(np.int64(5))          # numpy int64
np.random.rand(np.uint8(5))          # numpy uint8
np.random.rand(np.uint16(5))         # numpy uint16
np.random.rand(np.uint32(5))         # numpy uint32
np.random.rand(np.uint64(5))         # numpy uint64
np.random.rand(np.intp(5))           # platform pointer type

# Objects with __index__ method work:
class MyInt:
    def __index__(self): return 3
np.random.rand(MyInt())              # Works! shape=(3,)
```

### Rejected Types
```python
np.random.rand(5.0)
# TypeError: 'float' object cannot be interpreted as an integer

np.random.rand(-1)
# ValueError: negative dimensions are not allowed
```

### Multi-dimensional Size
```python
np.random.uniform(0, 1, size=(2, 3))           # tuple of ints
np.random.uniform(0, 1, size=[2, 3])           # list works too
np.random.uniform(0, 1, size=np.array([2, 3])) # ndarray works
np.random.uniform(0, 1, size=(np.int64(2), np.int64(3)))  # tuple of int64

# Special cases:
np.random.uniform(0, 1, size=None)   # Returns Python float (not ndarray!)
np.random.uniform(0, 1, size=())     # Returns 0-d ndarray, shape=()
np.random.uniform(0, 1, size=(2,0,3)) # Valid! Creates empty array

np.random.uniform(0, 1, size=(2, -1))
# ValueError: negative dimensions are not allowed
```

---

## Test 2: Axis Parameter Type Acceptance

### Accepted Types
```python
arr = np.arange(24).reshape(2, 3, 4)

np.sum(arr, axis=1)                  # Python int
np.sum(arr, axis=np.int32(1))        # numpy int32
np.sum(arr, axis=np.int64(1))        # numpy int64
np.sum(arr, axis=np.uint64(1))       # numpy uint64
np.sum(arr, axis=-1)                 # negative (wraps)
np.sum(arr, axis=np.int64(-1))       # negative int64
np.sum(arr, axis=(0, 2))             # tuple of axes
np.sum(arr, axis=(np.int64(0), np.int64(2)))  # tuple of int64
np.sum(arr, axis=None)               # reduce all axes
```

### Rejected Types
```python
np.sum(arr, axis=1.0)
# TypeError: 'float' object cannot be interpreted as an integer

np.sum(arr, axis=5)   # ndim=3, valid axes are 0,1,2
# numpy.exceptions.AxisError: axis 5 is out of bounds for array of dimension 3

np.sum(arr, axis=-4)  # ndim=3, valid negative axes are -1,-2,-3
# numpy.exceptions.AxisError: axis -4 is out of bounds for array of dimension 3
```

### Axis Normalization
```
For ndim=3 array (axes 0, 1, 2):
  axis=0  -> axis 0
  axis=1  -> axis 1
  axis=2  -> axis 2
  axis=-1 -> axis 2 (ndim + axis = 3 + (-1) = 2)
  axis=-2 -> axis 1
  axis=-3 -> axis 0
  axis=3  -> AxisError (out of bounds)
  axis=-4 -> AxisError (out of bounds)
```

---

## Test 3: Return Types

### Array Properties
```python
arr = np.arange(24).reshape(2, 3, 4)

type(arr.shape[0])   # <class 'int'>  (Python int, NOT np.int64)
type(arr.strides[0]) # <class 'int'>
type(arr.size)       # <class 'int'>
type(arr.ndim)       # <class 'int'>
type(arr.nbytes)     # <class 'int'>
type(arr.itemsize)   # <class 'int'>

isinstance(arr.shape[0], int)         # True
isinstance(arr.shape[0], np.integer)  # False
```

### Index Return Types
```python
arr = np.arange(10)

result = np.argmax(arr)
type(result)  # <class 'numpy.int64'>  # Note: np.int64, not Python int!

result = np.argmax(arr.reshape(2,5), axis=0)
result.dtype  # dtype('int64')

indices = np.nonzero(arr > 5)
indices[0].dtype  # dtype('int64')

indices = np.where(arr > 5)
indices[0].dtype  # dtype('int64')
```

---

## Test 4: randint Behavior

### Default dtype
```python
r = np.random.randint(0, 10, size=5)
r.dtype  # dtype('int32')  <-- Default is int32!
```

### With dtype parameter
```python
np.random.randint(0, 10, dtype=np.int32)   # Works
np.random.randint(0, 10, dtype=np.int64)   # Works
np.random.randint(0, 256, dtype=np.uint8)  # Works
```

### Bounds validation
```python
# High value must fit in dtype:
np.random.randint(0, 2**32, size=5)  # Default dtype=int32
# ValueError: high is out of bounds for int32

np.random.randint(0, 2**32, size=5, dtype=np.int64)  # Works!

np.random.randint(0, 1000, dtype=np.uint8)  # uint8 max is 255
# ValueError: high is out of bounds for uint8
```

### Large ranges with int64
```python
np.random.seed(42)
np.random.randint(0, 2**62, size=5, dtype=np.int64)
# array([145689414457766657, 4229063510710445413, ...], dtype=int64)

# Near int64 max:
np.random.randint(2**63-1000, 2**63, size=5, dtype=np.int64)  # Works!
```

---

## Test 5: Seed Validation

### Accepted Values
```python
np.random.seed(0)            # OK
np.random.seed(42)           # OK
np.random.seed(2**32 - 1)    # OK (4294967295)
np.random.seed(np.int32(42)) # OK
np.random.seed(np.int64(42)) # OK
np.random.seed(np.uint32(42))# OK
np.random.seed(np.uint64(42))# OK
np.random.seed(None)         # OK (uses entropy)
np.random.seed([1, 2, 3, 4]) # OK (array seed)
```

### Rejected Values
```python
np.random.seed(-1)
# ValueError: Seed must be between 0 and 2**32 - 1

np.random.seed(2**32)        # 4294967296
# ValueError: Seed must be between 0 and 2**32 - 1

np.random.seed(2**33 + 42)
# ValueError: Seed must be between 0 and 2**32 - 1

np.random.seed(2**100)
# ValueError: Seed must be between 0 and 2**32 - 1
```

---

## Test 6: Reshape -1 Special Case

```python
arr = np.arange(12)

arr.reshape(-1, 3)   # shape=(4, 3) - infers first dim
arr.reshape(2, -1)   # shape=(2, 6) - infers second dim
arr.reshape(-1)      # shape=(12,) - flatten

arr.reshape(-1, -1)
# ValueError: can only specify one unknown dimension

# Note: -1 in reshape is DIFFERENT from -1 in size!
# reshape(-1) = infer dimension
# size=-1 = ValueError (negative dimensions not allowed)
```

---

## NumSharp Implementation Requirements

### 1. Size Parameter Validation

```csharp
// Add validation for size parameters in random functions:
private static void ValidateSize(int[] size)
{
    if (size == null) return;
    foreach (var dim in size)
    {
        if (dim < 0)
            throw new ValueError("negative dimensions are not allowed");
    }
}

// For accepting long values:
public NDArray uniform(double low, double high, params long[] size)
{
    // Convert long[] to int[] with validation
    var intSize = new int[size.Length];
    for (int i = 0; i < size.Length; i++)
    {
        if (size[i] < 0)
            throw new ValueError("negative dimensions are not allowed");
        if (size[i] > int.MaxValue)
            throw new ValueError("array is too big");
        intSize[i] = (int)size[i];
    }
    return uniform(low, high, intSize);
}
```

### 2. Axis Validation

```csharp
// Normalize and validate axis:
public static int NormalizeAxis(int axis, int ndim)
{
    if (axis < 0)
        axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw new AxisError($"axis {axis} is out of bounds for array of dimension {ndim}");
    return axis;
}
```

### 3. Seed Validation

```csharp
// Add overloads and validation:
public void seed(uint seed)  // Primary - matches NumPy's uint32 range
{
    Seed = (int)seed;
    randomizer = new MT19937(seed);
    _hasGauss = false;
    _gaussCache = 0.0;
}

public void seed(int seed)
{
    if (seed < 0)
        throw new ValueError("Seed must be between 0 and 2**32 - 1");
    this.seed((uint)seed);
}

public void seed(long seed)
{
    if (seed < 0 || seed > uint.MaxValue)
        throw new ValueError("Seed must be between 0 and 2**32 - 1");
    this.seed((uint)seed);
}

public void seed(ulong seed)
{
    if (seed > uint.MaxValue)
        throw new ValueError("Seed must be between 0 and 2**32 - 1");
    this.seed((uint)seed);
}
```

### 4. randint Int64 Support

```csharp
public NDArray randint(long low, long high = -1, Shape size = default, NPTypeCode? dtype = null)
{
    var typeCode = dtype ?? NPTypeCode.Int32;

    if (high == -1)
    {
        high = low;
        low = 0;
    }

    // Validate bounds against dtype
    var (min, max) = GetTypeRange(typeCode);
    if (high > max + 1)
        throw new ValueError($"high is out of bounds for {typeCode.AsNumpyDtypeName()}");
    if (low < min)
        throw new ValueError($"low is out of bounds for {typeCode.AsNumpyDtypeName()}");

    // Use appropriate random method based on range
    if (typeCode == NPTypeCode.Int64 || typeCode == NPTypeCode.UInt64)
    {
        // Use NextLong for int64 ranges
        return GenerateRandintLong(low, high, size, typeCode);
    }
    else
    {
        // Use Next for int32 ranges
        return GenerateRandintInt((int)low, (int)high, size, typeCode);
    }
}
```

---

## Platform Considerations

### .NET Array Limitations
- `Array.Length` is `int` (not `long`)
- Maximum array size is ~2^31 elements
- Shape dimensions should remain `int[]` (this is correct)

### Platform Pointer Type
```python
# NumPy uses np.intp for platform-specific pointer size:
np.intp          # <class 'numpy.int64'> on 64-bit
np.dtype(np.intp).itemsize  # 8 bytes on 64-bit
```

In NumSharp, `nint` (native int) is the C# equivalent, but since .NET arrays are int32-indexed, this is mostly irrelevant.

---

## Verification Commands

```python
# Test seed compatibility:
np.random.seed(42)
print(np.random.randint(0, 100, size=5))  # [51, 92, 14, 71, 60]

# Test with different dtypes:
np.random.seed(42)
print(np.random.randint(0, 100, size=5, dtype=np.int32))  # [51, 92, 14, 71, 60]
np.random.seed(42)
print(np.random.randint(0, 100, size=5, dtype=np.int64))  # [51, 92, 14, 71, 60]
```

---

## Exception Types

| Condition | NumPy Exception | NumSharp Should Throw |
|-----------|-----------------|----------------------|
| Negative size | `ValueError` | `ValueError` |
| Float as size | `TypeError` | `TypeError` (or ArgumentException) |
| Axis out of bounds | `numpy.exceptions.AxisError` | `AxisError` (custom) |
| Seed out of range | `ValueError` | `ValueError` |
| randint high out of bounds | `ValueError` | `ValueError` |
| Array too big | `ValueError` | `ValueError` (or OutOfMemoryException) |

---

## Appendix A: NumSharp Current Behavior (Gaps Identified)

### Tested 2026-03-24

#### 1. Seed Validation - GAPS FOUND

```
Test: seed(-1)
  NumSharp: ACCEPTED (no error)
  NumPy:    ValueError: Seed must be between 0 and 2**32 - 1
  STATUS:   MISMATCH - must reject negative seeds

Test: seed(long)
  NumSharp: Not available - signature is seed(int) only
  NumPy:    Accepts any integer, validates 0 to 2^32-1
  STATUS:   API MISMATCH - add overloads with validation
```

#### 2. Size Validation - GAPS FOUND

```
Test: rand(-1)
  NumSharp: OutOfMemoryException
  NumPy:    ValueError: negative dimensions are not allowed
  STATUS:   MISMATCH - should throw ValueError before allocation

Test: rand(0)
  NumSharp: InvalidOperationException: Can't construct ValueCoordinatesIncrementor with an empty shape
  NumPy:    Returns empty array shape=(0,), size=0
  STATUS:   MISMATCH - zero dimensions are valid

Test: uniform(0, 1, size=())
  NumSharp: Returns shape=(1), ndim=1
  NumPy:    Returns shape=(), ndim=0 (0-d array)
  STATUS:   MISMATCH - empty shape should create 0-d array
```

#### 3. Axis Validation - GAPS FOUND

```
Test: sum(arr, axis=3) where ndim=3
  NumSharp: ArgumentOutOfRangeException
  NumPy:    AxisError: axis 3 is out of bounds for array of dimension 3
  STATUS:   OK behavior, wrong exception type

Test: sum(arr, axis=-4) where ndim=3
  NumSharp: Returns result (silently normalizes to valid axis)
  NumPy:    AxisError: axis -4 is out of bounds for array of dimension 3
  STATUS:   MISMATCH - must validate negative axis bounds

Test: sum(arr, axis=-100) where ndim=3
  NumSharp: Returns result (silently normalizes)
  NumPy:    AxisError
  STATUS:   MISMATCH - bug in negative axis normalization
```

#### 4. randint Bounds - GAPS FOUND

```
Test: randint(0, 2^32, dtype=int32)
  NumSharp: Returns array of zeros
  NumPy:    ValueError: high is out of bounds for int32
  STATUS:   MISMATCH - must validate bounds against dtype
```

#### 5. Working Correctly

```
Test: randint(0, 100, size=5) with seed=42
  NumSharp: [51, 92, 14, 71, 60]
  NumPy:    [51, 92, 14, 71, 60]
  STATUS:   MATCH

Test: Valid positive axes (0, 1, 2)
  STATUS:   MATCH

Test: Valid negative axes (-1, -2, -3)
  STATUS:   MATCH

Test: reshape(-1, 3) dimension inference
  STATUS:   MATCH
```

---

## Appendix B: Priority Fix List

### P0 - Critical (Wrong behavior, silent corruption)

1. **Negative axis normalization bug**: `axis=-4` on 3D array silently works instead of throwing
2. **randint bounds**: Large high values silently produce zeros instead of throwing

### P1 - High (Wrong exceptions)

3. **Negative size**: Should throw `ValueError`, throws `OutOfMemoryException`
4. **Negative seed**: Should throw `ValueError`, silently accepts
5. **Zero size**: Should work, throws `InvalidOperationException`

### P2 - Medium (API parity)

6. **seed() overloads**: Add `uint`, `long`, `ulong` overloads with validation
7. **AxisError exception**: Create custom `AxisError` exception type
8. **size=() scalar**: Should return ndim=0, returns ndim=1

### P3 - Low (Nice to have)

9. **Error messages**: Match NumPy's exact error message text
