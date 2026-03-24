# NEP 50 - Promotion Rules for Python Scalars

**Status:** Final
**NumSharp Impact:** CRITICAL - Major behavioral change in NumPy 2.0

## Summary

NumPy 2.0 changes how Python scalars (int, float, complex) interact with NumPy arrays in type promotion. Values no longer influence result types.

## The Two Key Problems Solved

### Problem 1: Value-Based Promotion (ELIMINATED)

**Old behavior inspected values:**
```python
np.result_type(np.int8, 1) == np.int8      # 1 fits in int8
np.result_type(np.int8, 255) == np.int16   # 255 doesn't fit - UPCASTED!
```

**New behavior ignores values:**
```python
np.result_type(np.int8, 1) == np.int8      # int8, regardless of value
np.result_type(np.int8, 255) == np.int8    # Still int8!
```

### Problem 2: Inconsistent 0-D vs N-D Arrays (FIXED)

**Old inconsistency:**
```python
np.result_type(np.array(1, dtype=np.uint8), 1) == np.int64  # 0-D
np.result_type(np.array([1], dtype=np.uint8), 1) == np.uint8  # 1-D different!
```

**New consistency:**
```python
# Both now return uint8
np.result_type(np.array(1, dtype=np.uint8), 1) == np.uint8
np.result_type(np.array([1], dtype=np.uint8), 1) == np.uint8
```

## New "Weak" Scalar Promotion Rules

Python `int`, `float`, `complex` are treated as "weakly typed":

```python
np.uint8(1) + 1 == np.uint8(2)           # Result is uint8
np.int16(2) + 2 == np.int16(4)           # Result is int16
np.uint16(3) + 3.0 == np.float64(6.0)    # Float promotes to default
np.float32(5) + 5j == np.complex64(5+5j) # Same precision preserved
```

### Kind Hierarchy

**boolean < integral < inexact**

Cross-kind promotion uses default precision:
- `boolean` + `integral` → `int64`
- `integral` + `inexact` → `float64` or `complex128`
- `float32` + `complex` → `complex64`

## Breaking Changes Table

| Expression | NumPy 1.x | NumPy 2.x | Note |
|---|---|---|---|
| `uint8(1) + 2` | `int64(3)` | `uint8(3)` | Honors uint8 |
| `array([1], uint8) + int64(1)` | `uint8` | `int64` | Respects int64 |
| `array([1.], float32) + float64(1.)` | `float32` | `float64` | Respects float64 |
| `uint8(1) + 300` | `int64(301)` | **Exception** | 300 > uint8 max |
| `uint8(100) + 200` | `int64(300)` | `uint8(44)` | Overflow warning |

## NumSharp Implementation Requirements

### Current Behavior Audit

Check NumSharp's `np._FindCommonType` and arithmetic operators:

```csharp
var a = np.array(new byte[] { 1 });  // uint8
var b = 300;  // C# int
var c = a + b;  // What happens?
```

### Required Changes

1. **Value-Independent Promotion:**
   ```csharp
   // Don't inspect scalar value, only type
   NPTypeCode Promote(NPTypeCode arrayType, Type scalarType) {
       // scalarType is int/long/float/double, not the actual value
   }
   ```

2. **Weak Scalar Treatment:**
   - C# `int` → weakly typed (defers to array dtype if same kind)
   - C# `double` → weakly typed for floating arrays
   - NumPy scalar types → strongly typed

3. **Overflow Handling:**
   ```csharp
   // When Python scalar doesn't fit in target dtype:
   // Option A: Throw (matches NumPy for literals)
   // Option B: Overflow with warning (matches NumPy for operations)
   ```

4. **Consistent 0-D and N-D:**
   - Zero-rank arrays should behave same as N-D arrays in promotion

### Type Promotion Matrix

```csharp
// Weak scalar (C# int) + NumPy array
uint8 + int  → uint8   // int defers to array
int16 + int  → int16
float32 + int → float32

// NumPy scalar + NumPy array
uint8 + int64 → int64  // int64 is strong
float32 + float64 → float64

// Cross-kind (weak scalar)
uint8 + float → float64  // Default float
int32 + complex → complex128  // Default complex
```

## Migration Detection

Test with environment variable (NumPy 1.24+):
```bash
NPY_PROMOTION_STATE=weak_and_warn
```

## References

- [NEP 50 Full Text](https://numpy.org/neps/nep-0050-scalar-promotion.html)
- `src/NumSharp.Core/Utilities/np.find_common_type.cs`
- `src/NumSharp.Core/Operations/Elementwise/`
