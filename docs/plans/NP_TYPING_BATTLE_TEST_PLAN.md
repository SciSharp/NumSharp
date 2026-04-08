# Plan: Battle Test np Typing Functions

**Goal:** Thoroughly battle test each typing function implementation, ensure C#-friendly APIs with proper overloads, and cover all edge cases in tests.

---

## Part 1: API Overload Analysis and Gaps

### 1.1 Current State Audit

| Function | Current Overloads | Missing C#-Friendly Overloads |
|----------|-------------------|-------------------------------|
| `iinfo` | `NPTypeCode`, `Type` | `iinfo<T>()`, `iinfo(NDArray)`, `iinfo(string dtype)` |
| `finfo` | `NPTypeCode`, `Type` | `finfo<T>()`, `finfo(NDArray)`, `finfo(string dtype)` |
| `can_cast` | `NPTypeCode`, `Type`, `int`, `long`, `double`, `object`, `NDArray` | `byte`, `short`, `ushort`, `uint`, `ulong`, `float`, `decimal`, `bool`, generic `can_cast<TFrom, TTo>()` |
| `result_type` | `object[]`, `NPTypeCode[]`, `NDArray[]` | Two-arg convenience `result_type(NPTypeCode, NPTypeCode)`, `result_type(Type, Type)` |
| `promote_types` | `NPTypeCode`, `Type` | Generic `promote_types<T1, T2>()` |
| `min_scalar_type` | `object` | Typed overloads for common types |
| `issubdtype` | `NPTypeCode+string`, `NPTypeCode+NPTypeCode`, `Type+string` | `NDArray+string`, `Type+Type` |
| `common_type` | `NDArray[]` | `NPTypeCode[]`, `Type[]` |
| `issctype` | `object` | `Type`, `NPTypeCode`, `NDArray` explicit overloads |
| `isdtype` | `NPTypeCode+string`, `NPTypeCode+string[]` | `NDArray+string`, `Type+string` |
| `isreal/iscomplex` | `NDArray` | Scalar overloads |
| `isrealobj/iscomplexobj` | `NDArray` | Scalar overloads |

### 1.2 Implementation Tasks

- [ ] Add generic overloads for type-safe C# usage
- [ ] Add NDArray overloads where missing
- [ ] Add dtype string overloads (e.g., `iinfo("int32")`)
- [ ] Add scalar-specific overloads to avoid boxing
- [ ] Add two-argument convenience overloads
- [ ] Ensure consistent null handling across all functions

---

## Part 2: iinfo Battle Tests

### 2.1 Edge Cases

```csharp
// All supported integer types
[TestCase(NPTypeCode.Boolean)]
[TestCase(NPTypeCode.Byte)]
[TestCase(NPTypeCode.Int16)]
[TestCase(NPTypeCode.UInt16)]
[TestCase(NPTypeCode.Int32)]
[TestCase(NPTypeCode.UInt32)]
[TestCase(NPTypeCode.Int64)]
[TestCase(NPTypeCode.UInt64)]
[TestCase(NPTypeCode.Char)]

// Error cases
np.iinfo(NPTypeCode.Single)   // Should throw
np.iinfo(NPTypeCode.Double)   // Should throw
np.iinfo(NPTypeCode.Decimal)  // Should throw
np.iinfo(NPTypeCode.Complex)  // Should throw
np.iinfo(NPTypeCode.Empty)    // Should throw
np.iinfo((Type)null)          // Should throw

// C# friendly
np.iinfo<int>()               // New: generic
np.iinfo(np.array(new int[]{1})) // New: from array
np.iinfo("int32")             // New: from string
```

### 2.2 NumPy Alignment Verification

```python
# Run in Python to verify
import numpy as np
for dt in [np.bool_, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
    info = np.iinfo(dt)
    print(f"{dt}: bits={info.bits}, min={info.min}, max={info.max}, kind={info.kind}")
```

### 2.3 Implementation Checklist

- [ ] Add `iinfo<T>()` generic overload
- [ ] Add `iinfo(NDArray arr)` overload
- [ ] Add `iinfo(string dtype)` overload
- [ ] Verify UInt64 max handling (exceeds long.MaxValue)
- [ ] Test `ToString()` output matches NumPy format
- [ ] Test equality/comparison semantics

---

## Part 3: finfo Battle Tests

### 3.1 Edge Cases

```csharp
// All supported float types
np.finfo(NPTypeCode.Single)
np.finfo(NPTypeCode.Double)
np.finfo(NPTypeCode.Decimal)  // Partial support

// Error cases
np.finfo(NPTypeCode.Int32)    // Should throw
np.finfo(NPTypeCode.Boolean)  // Should throw
np.finfo(NPTypeCode.Empty)    // Should throw
np.finfo((Type)null)          // Should throw

// C# friendly
np.finfo<float>()             // New: generic
np.finfo<double>()
np.finfo(np.array(new float[]{1})) // New: from array
np.finfo("float64")           // New: from string
```

### 3.2 NumPy Alignment Verification

```python
import numpy as np
for dt in [np.float32, np.float64]:
    info = np.finfo(dt)
    print(f"{dt}:")
    print(f"  bits={info.bits}")
    print(f"  eps={info.eps}")
    print(f"  epsneg={info.epsneg}")
    print(f"  max={info.max}")
    print(f"  min={info.min}")
    print(f"  tiny={info.tiny}")
    print(f"  smallest_normal={info.smallest_normal}")
    print(f"  smallest_subnormal={info.smallest_subnormal}")
    print(f"  precision={info.precision}")
    print(f"  resolution={info.resolution}")
    print(f"  maxexp={info.maxexp}")
    print(f"  minexp={info.minexp}")
```

### 3.3 Implementation Checklist

- [ ] Add `finfo<T>()` generic overload
- [ ] Add `finfo(NDArray arr)` overload
- [ ] Add `finfo(string dtype)` overload
- [ ] Verify eps calculation matches NumPy exactly
- [ ] Verify smallest_normal/smallest_subnormal values
- [ ] Test `ToString()` output format

---

## Part 4: can_cast Battle Tests

### 4.1 Comprehensive Type Matrix

Test all 12x12 type combinations for "safe" casting:

```csharp
// Generate test matrix
foreach (var from in AllTypeCodes)
    foreach (var to in AllTypeCodes)
        foreach (var casting in new[] { "no", "equiv", "safe", "same_kind", "unsafe" })
            TestCanCast(from, to, casting);
```

### 4.2 Scalar Value Tests

```csharp
// Boundary values
np.can_cast(0, NPTypeCode.Boolean)      // True
np.can_cast(1, NPTypeCode.Boolean)      // True
np.can_cast(2, NPTypeCode.Boolean)      // False

np.can_cast(255, NPTypeCode.Byte)       // True
np.can_cast(256, NPTypeCode.Byte)       // False

np.can_cast(-1, NPTypeCode.Byte)        // False
np.can_cast(-1, NPTypeCode.Int16)       // True

np.can_cast(int.MaxValue, NPTypeCode.Int32)   // True
np.can_cast(int.MaxValue + 1L, NPTypeCode.Int32) // False

np.can_cast(long.MaxValue, NPTypeCode.UInt64) // False (negative when viewed as signed)
np.can_cast((ulong)long.MaxValue, NPTypeCode.UInt64) // True

// Float edge cases
np.can_cast(double.MaxValue, NPTypeCode.Single) // False
np.can_cast(float.MaxValue, NPTypeCode.Double)  // True
np.can_cast(double.NaN, NPTypeCode.Single)      // ?
np.can_cast(double.PositiveInfinity, NPTypeCode.Single) // ?
```

### 4.3 Missing Overloads to Add

```csharp
// Add specific overloads to avoid enum confusion
public static bool can_cast(byte value, NPTypeCode to, string casting = "safe")
public static bool can_cast(short value, NPTypeCode to, string casting = "safe")
public static bool can_cast(ushort value, NPTypeCode to, string casting = "safe")
public static bool can_cast(uint value, NPTypeCode to, string casting = "safe")
public static bool can_cast(ulong value, NPTypeCode to, string casting = "safe")
public static bool can_cast(float value, NPTypeCode to, string casting = "safe")
public static bool can_cast(decimal value, NPTypeCode to, string casting = "safe")
public static bool can_cast(bool value, NPTypeCode to, string casting = "safe")

// Generic overload
public static bool can_cast<TFrom, TTo>(string casting = "safe")
```

### 4.4 NumPy Alignment

```python
import numpy as np

# Test matrix
types = [np.bool_, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
         np.int64, np.uint64, np.float32, np.float64]

for from_t in types:
    for to_t in types:
        for casting in ['no', 'equiv', 'safe', 'same_kind', 'unsafe']:
            result = np.can_cast(from_t, to_t, casting=casting)
            print(f"can_cast({from_t}, {to_t}, '{casting}') = {result}")

# Scalar tests
print(np.can_cast(100, np.uint8))    # True
print(np.can_cast(1000, np.uint8))   # False
print(np.can_cast(-1, np.uint8))     # False
```

### 4.5 Implementation Checklist

- [ ] Add all primitive type overloads (byte, short, ushort, uint, ulong, float, decimal, bool)
- [ ] Add `can_cast<TFrom, TTo>()` generic overload
- [ ] Add `can_cast(NDArray from, NDArray to)` overload
- [ ] Test invalid casting string throws ArgumentException
- [ ] Test null NDArray throws ArgumentNullException
- [ ] Verify "same_kind" allows signed<->signed and unsigned<->unsigned downcasting
- [ ] Generate full 12x12x5 type matrix tests

---

## Part 5: result_type Battle Tests

### 5.1 Edge Cases

```csharp
// Single type
np.result_type(NPTypeCode.Int32)  // Int32

// Two types
np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)    // Int64
np.result_type(NPTypeCode.Int32, NPTypeCode.Single)   // Single or Double?
np.result_type(NPTypeCode.Int64, NPTypeCode.Double)   // Double
np.result_type(NPTypeCode.UInt32, NPTypeCode.Int32)   // Int64 (to hold both ranges)

// Many types
np.result_type(NPTypeCode.Int8, NPTypeCode.Int16, NPTypeCode.Int32)  // Int32

// Arrays
var a = np.array(new int[]{1});
var b = np.array(new float[]{1f});
np.result_type(a, b)  // Single or Double

// Mixed arrays and types
np.result_type(a, NPTypeCode.Double)  // Double

// Empty throws
np.result_type()  // Should throw
```

### 5.2 Missing Overloads

```csharp
// Two-argument convenience (avoid params overhead)
public static NPTypeCode result_type(NPTypeCode type1, NPTypeCode type2)
public static NPTypeCode result_type(Type type1, Type type2)
public static NPTypeCode result_type(NDArray arr1, NDArray arr2)
```

### 5.3 NumPy Alignment

```python
import numpy as np

print(np.result_type(np.int32, np.int64))     # int64
print(np.result_type(np.int32, np.float32))   # float64
print(np.result_type(np.uint32, np.int32))    # int64
print(np.result_type(np.float32, np.float64)) # float64

a = np.array([1], dtype=np.int32)
b = np.array([1.0], dtype=np.float32)
print(np.result_type(a, b))  # float64
```

### 5.4 Implementation Checklist

- [ ] Add two-argument convenience overloads
- [ ] Verify promotion matches NumPy 2.x NEP50 rules
- [ ] Test with all 12 type combinations
- [ ] Test empty input throws
- [ ] Test null input handling
- [ ] Test scalar NDArray handling

---

## Part 6: promote_types Battle Tests

### 6.1 Full Type Matrix

```csharp
// Test all 12x12 combinations
foreach (var t1 in AllTypeCodes)
    foreach (var t2 in AllTypeCodes)
        TestPromoteTypes(t1, t2);
```

### 6.2 NumPy Alignment

```python
import numpy as np

types = [np.bool_, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
         np.int64, np.uint64, np.float32, np.float64]

for t1 in types:
    for t2 in types:
        result = np.promote_types(t1, t2)
        print(f"promote_types({t1}, {t2}) = {result}")
```

### 6.3 Implementation Checklist

- [ ] Add `promote_types<T1, T2>()` generic overload
- [ ] Test symmetric: `promote_types(a, b) == promote_types(b, a)`
- [ ] Test all type pairs
- [ ] Verify matches NumPy exactly

---

## Part 7: min_scalar_type Battle Tests

### 7.1 Boundary Tests

```csharp
// Unsigned boundaries
np.min_scalar_type(0)          // Byte
np.min_scalar_type(255)        // Byte
np.min_scalar_type(256)        // UInt16
np.min_scalar_type(65535)      // UInt16
np.min_scalar_type(65536)      // UInt32
np.min_scalar_type(uint.MaxValue)  // UInt32
np.min_scalar_type((ulong)uint.MaxValue + 1) // UInt64

// Signed boundaries
np.min_scalar_type(-1)         // Int16 (no Int8 in NumSharp)
np.min_scalar_type(-128)       // Int16
np.min_scalar_type(-129)       // Int16
np.min_scalar_type(-32768)     // Int16
np.min_scalar_type(-32769)     // Int32
np.min_scalar_type(int.MinValue)   // Int32
np.min_scalar_type((long)int.MinValue - 1) // Int64

// Float
np.min_scalar_type(1.0f)       // Single
np.min_scalar_type(1.0)        // Single or Double?
np.min_scalar_type(1e100)      // Double
np.min_scalar_type(float.MaxValue * 2.0) // Double

// Special values
np.min_scalar_type(float.NaN)           // Single
np.min_scalar_type(double.NaN)          // Single or Double
np.min_scalar_type(float.PositiveInfinity)  // Single
np.min_scalar_type(true)                // Boolean
np.min_scalar_type(false)               // Boolean
```

### 7.2 NumPy Alignment

```python
import numpy as np

values = [0, 255, 256, 65535, 65536, -1, -128, -129, -32768, -32769,
          1.0, 1e100, float('nan'), float('inf'), True, False]

for v in values:
    result = np.min_scalar_type(v)
    print(f"min_scalar_type({v}) = {result}")
```

### 7.3 Implementation Checklist

- [ ] Test all boundary values
- [ ] Verify float precision detection
- [ ] Test special float values (NaN, Inf)
- [ ] Add typed overloads for performance

---

## Part 8: issubdtype Battle Tests

### 8.1 Type Hierarchy Tests

```csharp
// Generic
foreach (var t in AllTypeCodes)
    Assert(np.issubdtype(t, "generic") == true);

// Number (excludes bool, char, string)
np.issubdtype(NPTypeCode.Int32, "number")    // True
np.issubdtype(NPTypeCode.Boolean, "number")  // False (NumPy 2.x)
np.issubdtype(NPTypeCode.Char, "number")     // False?

// Integer
np.issubdtype(NPTypeCode.Int32, "integer")   // True
np.issubdtype(NPTypeCode.Byte, "integer")    // True
np.issubdtype(NPTypeCode.Boolean, "integer") // False (NumPy 2.x)
np.issubdtype(NPTypeCode.Double, "integer")  // False

// Signed/Unsigned
np.issubdtype(NPTypeCode.Int32, "signedinteger")    // True
np.issubdtype(NPTypeCode.UInt32, "signedinteger")   // False
np.issubdtype(NPTypeCode.UInt32, "unsignedinteger") // True
np.issubdtype(NPTypeCode.Int32, "unsignedinteger")  // False

// Floating
np.issubdtype(NPTypeCode.Double, "floating")  // True
np.issubdtype(NPTypeCode.Single, "floating")  // True
np.issubdtype(NPTypeCode.Decimal, "floating") // True
np.issubdtype(NPTypeCode.Int32, "floating")   // False

// Inexact
np.issubdtype(NPTypeCode.Double, "inexact")   // True
np.issubdtype(NPTypeCode.Complex, "inexact")  // True
np.issubdtype(NPTypeCode.Int32, "inexact")    // False
```

### 8.2 Missing Overloads

```csharp
// From NDArray
public static bool issubdtype(NDArray arr, string category)

// Type to Type
public static bool issubdtype(Type arg1, Type arg2)

// NPTypeCode to Type
public static bool issubdtype(NPTypeCode arg1, Type arg2)
```

### 8.3 NumPy Alignment

```python
import numpy as np

types = [np.bool_, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
         np.int64, np.uint64, np.float32, np.float64]

categories = ['generic', 'number', 'integer', 'signedinteger',
              'unsignedinteger', 'inexact', 'floating']

for t in types:
    for cat in categories:
        try:
            result = np.issubdtype(t, getattr(np, cat))
            print(f"issubdtype({t}, {cat}) = {result}")
        except:
            print(f"issubdtype({t}, {cat}) = ERROR")
```

### 8.4 Implementation Checklist

- [ ] Add `issubdtype(NDArray, string)` overload
- [ ] Add `issubdtype(Type, Type)` overload
- [ ] Verify bool is NOT integer subtype (NumPy 2.x)
- [ ] Test all type/category combinations
- [ ] Test invalid category string handling

---

## Part 9: common_type Battle Tests

### 9.1 Tests

```csharp
// Always returns float type
np.common_type_code(np.array(new int[]{1}))     // Double
np.common_type_code(np.array(new byte[]{1}))    // Double
np.common_type_code(np.array(new float[]{1f}))  // Single
np.common_type_code(np.array(new double[]{1.0})) // Double

// Multiple arrays
np.common_type_code(
    np.array(new int[]{1}),
    np.array(new float[]{1f})
)  // Single? Double?

// Empty throws
np.common_type_code()  // Should throw
```

### 9.2 Missing Overloads

```csharp
// From type codes
public static NPTypeCode common_type_code(params NPTypeCode[] types)

// From Types
public static Type common_type(params Type[] types)
```

### 9.3 NumPy Alignment

```python
import numpy as np

print(np.common_type(np.array([1], dtype=np.int32)))    # <class 'numpy.float64'>
print(np.common_type(np.array([1.0], dtype=np.float32))) # <class 'numpy.float32'>
print(np.common_type(
    np.array([1], dtype=np.int32),
    np.array([1.0], dtype=np.float32)
))  # ?
```

### 9.4 Implementation Checklist

- [ ] Add `common_type_code(params NPTypeCode[])` overload (already exists)
- [ ] Verify integer arrays always return Double
- [ ] Test single float32 array returns Single
- [ ] Test mixed types

---

## Part 10: Type Checking Functions Battle Tests

### 10.1 issctype Tests

```csharp
// Valid scalar types
np.issctype(typeof(int))        // True
np.issctype(typeof(float))      // True
np.issctype(typeof(double))     // True
np.issctype(NPTypeCode.Int32)   // True

// Invalid
np.issctype(typeof(NDArray))    // False
np.issctype(typeof(string))     // False?
np.issctype(typeof(object))     // False
np.issctype(null)               // False
np.issctype(123)                // False (not a type)
```

### 10.2 isdtype Tests

```csharp
// Single category
np.isdtype(NPTypeCode.Int32, "integral")       // True
np.isdtype(NPTypeCode.Int32, "real floating")  // False
np.isdtype(NPTypeCode.Double, "real floating") // True
np.isdtype(NPTypeCode.Boolean, "bool")         // True
np.isdtype(NPTypeCode.Boolean, "numeric")      // False

// Multiple categories
np.isdtype(NPTypeCode.Int32, new[] {"integral", "real floating"}) // True
np.isdtype(NPTypeCode.Boolean, new[] {"integral", "real floating"}) // False
```

### 10.3 Missing Overloads

```csharp
// From Type
public static bool isdtype(Type type, string kind)
public static bool isdtype(Type type, string[] kinds)

// From NDArray
public static bool isdtype(NDArray arr, string kind)
public static bool isdtype(NDArray arr, string[] kinds)
```

### 10.4 sctype2char Tests

```csharp
// All types
np.sctype2char(NPTypeCode.Boolean)  // 'b'
np.sctype2char(NPTypeCode.Byte)     // 'B'
np.sctype2char(NPTypeCode.Int16)    // 'h'
np.sctype2char(NPTypeCode.UInt16)   // 'H'
np.sctype2char(NPTypeCode.Int32)    // 'i'
np.sctype2char(NPTypeCode.UInt32)   // 'I'
np.sctype2char(NPTypeCode.Int64)    // 'q'
np.sctype2char(NPTypeCode.UInt64)   // 'Q'
np.sctype2char(NPTypeCode.Single)   // 'f'
np.sctype2char(NPTypeCode.Double)   // 'd'
np.sctype2char(NPTypeCode.Complex)  // 'D'

// Invalid
np.sctype2char(NPTypeCode.Empty)    // '?'
np.sctype2char(NPTypeCode.String)   // '?'
```

### 10.5 maximum_sctype Tests

```csharp
// Signed integers -> Int64
np.maximum_sctype(NPTypeCode.Int16)  // Int64
np.maximum_sctype(NPTypeCode.Int32)  // Int64
np.maximum_sctype(NPTypeCode.Int64)  // Int64

// Unsigned integers -> UInt64
np.maximum_sctype(NPTypeCode.Byte)   // UInt64
np.maximum_sctype(NPTypeCode.UInt16) // UInt64
np.maximum_sctype(NPTypeCode.UInt32) // UInt64
np.maximum_sctype(NPTypeCode.UInt64) // UInt64

// Floats -> Double
np.maximum_sctype(NPTypeCode.Single) // Double
np.maximum_sctype(NPTypeCode.Double) // Double

// Special
np.maximum_sctype(NPTypeCode.Boolean) // Boolean
np.maximum_sctype(NPTypeCode.Decimal) // Decimal
```

### 10.6 Implementation Checklist

- [ ] Add `isdtype(Type, string)` overload
- [ ] Add `isdtype(NDArray, string)` overload
- [ ] Verify sctype2char matches NumPy character codes
- [ ] Test all types for maximum_sctype

---

## Part 11: isreal/iscomplex Battle Tests

### 11.1 Array Tests

```csharp
// Non-complex arrays - all real
var intArr = np.array(new int[] {1, 2, 3});
np.isreal(intArr)       // All True
np.iscomplex(intArr)    // All False
np.isrealobj(intArr)    // True
np.iscomplexobj(intArr) // False

var floatArr = np.array(new double[] {1.0, 2.0, 3.0});
np.isreal(floatArr)     // All True
np.iscomplex(floatArr)  // All False
np.isrealobj(floatArr)  // True
np.iscomplexobj(floatArr) // False

// Scalar arrays
var scalar = np.array(5);
np.isreal(scalar)       // True
np.isrealobj(scalar)    // True
```

### 11.2 Missing Overloads

```csharp
// Scalar overloads
public static bool isreal(object scalar)
public static bool iscomplex(object scalar)

// These return bool, not array
public static bool isrealobj(object x)
public static bool iscomplexobj(object x)
```

### 11.3 NumPy Alignment

```python
import numpy as np

a = np.array([1, 2, 3])
print(np.isreal(a))      # [True, True, True]
print(np.iscomplex(a))   # [False, False, False]
print(np.isrealobj(a))   # True
print(np.iscomplexobj(a)) # False

# Scalar
print(np.isreal(1.0))    # True
print(np.iscomplex(1.0)) # False
```

### 11.4 Implementation Checklist

- [ ] Add scalar overloads for isreal/iscomplex
- [ ] Verify element-wise behavior for arrays
- [ ] Test all dtype combinations
- [ ] Test scalar NDArray (0-dim)

---

## Part 12: Test File Structure

### 12.1 Test Organization

```
test/NumSharp.UnitTest/APIs/
├── np.iinfo.BattleTest.cs       # iinfo comprehensive tests
├── np.finfo.BattleTest.cs       # finfo comprehensive tests
├── np.can_cast.BattleTest.cs    # can_cast comprehensive tests
├── np.result_type.BattleTest.cs # result_type tests
├── np.promote_types.BattleTest.cs
├── np.min_scalar_type.BattleTest.cs
├── np.issubdtype.BattleTest.cs
├── np.common_type.BattleTest.cs
├── np.type_checks.BattleTest.cs  # issctype, isdtype, sctype2char, maximum_sctype
├── np.isreal_iscomplex.BattleTest.cs
└── np.typing.Test.cs            # Existing basic tests (keep for quick sanity)
```

### 12.2 Test Patterns

Each battle test file should include:

1. **All 12 dtype coverage** - Test with every supported dtype
2. **Boundary values** - Min/max values for each type
3. **Error cases** - Null, invalid types, empty arrays
4. **NumPy alignment** - Tests based on actual NumPy output
5. **C# idiom tests** - Generic overloads, common usage patterns
6. **Performance notes** - Document any boxing/allocation concerns

---

## Part 13: Execution Order

1. **Add Missing Overloads** (Parts 1.2)
   - [ ] iinfo: generic, NDArray, string
   - [ ] finfo: generic, NDArray, string
   - [ ] can_cast: all primitive types, generic
   - [ ] result_type: two-arg convenience
   - [ ] issubdtype: NDArray, Type+Type
   - [ ] isdtype: Type, NDArray
   - [ ] isreal/iscomplex: scalar

2. **Create Battle Test Files** (Part 12)
   - [ ] np.iinfo.BattleTest.cs
   - [ ] np.finfo.BattleTest.cs
   - [ ] np.can_cast.BattleTest.cs
   - [ ] np.result_type.BattleTest.cs
   - [ ] np.promote_types.BattleTest.cs
   - [ ] np.min_scalar_type.BattleTest.cs
   - [ ] np.issubdtype.BattleTest.cs
   - [ ] np.common_type.BattleTest.cs
   - [ ] np.type_checks.BattleTest.cs
   - [ ] np.isreal_iscomplex.BattleTest.cs

3. **Run and Fix**
   - [ ] Run all battle tests
   - [ ] Fix any NumPy alignment issues
   - [ ] Fix any C# API issues
   - [ ] Document any intentional differences

---

## Summary Checklist

| Part | Focus | Status |
|------|-------|--------|
| 1 | API Overload Analysis | [ ] |
| 2 | iinfo Battle Tests | [ ] |
| 3 | finfo Battle Tests | [ ] |
| 4 | can_cast Battle Tests | [ ] |
| 5 | result_type Battle Tests | [ ] |
| 6 | promote_types Battle Tests | [ ] |
| 7 | min_scalar_type Battle Tests | [ ] |
| 8 | issubdtype Battle Tests | [ ] |
| 9 | common_type Battle Tests | [ ] |
| 10 | Type Checking Functions | [ ] |
| 11 | isreal/iscomplex Battle Tests | [ ] |
| 12 | Test File Structure | [ ] |
| 13 | Execute and Fix | [ ] |
