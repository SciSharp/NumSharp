# Plan: Implement and Align np Typing Functions

**Issue:** #599 - [Core] Implement and align np typing functions
**Goal:** Implement NumPy's type introspection/promotion functions and consolidate scattered type info

---

## Part 1: Consolidate Internal Type Info

### 1.1 Alignment Research

```bash
# Compare existing implementations
grep -rn "SizeOf\|GetTypeSize" src/NumSharp.Core/ --include="*.cs"
grep -rn "AsType\|GetClrType" src/NumSharp.Core/ --include="*.cs"
grep -rn "IsUnsigned\|IsSigned" src/NumSharp.Core/ --include="*.cs"
```

**Files to audit:**
- `src/NumSharp.Core/Backends/NPTypeCode.cs`
- `src/NumSharp.Core/Backends/Kernels/TypeRules.cs`
- `src/NumSharp.Core/Backends/Kernels/ReductionKernel.cs`
- `src/NumSharp.Core/Utilities/NumberInfo.cs`
- `src/NumSharp.Core/Utilities/InfoOf.cs`

### 1.2 Implementation

- [ ] Remove `TypeRules.GetTypeSize()` - use `NPTypeCode.SizeOf()`
- [ ] Remove `TypeRules.GetClrType()` - use `NPTypeCode.AsType()`
- [ ] Remove `TypeRules.IsUnsigned()` - use `NPTypeCode.IsUnsigned()`
- [ ] Remove `TypeRules.IsSigned()` - use `NPTypeCode.IsSigned()`
- [ ] Remove `TypeRules.GetAccumulatingType()` - use `NPTypeCode.GetAccumulatingType()`
- [ ] Move `ReductionTypeExtensions.GetOneValue()` to `NPTypeCode` extensions
- [ ] Keep `ReductionTypeExtensions.GetMinValue()`/`GetMaxValue()` (different semantics - uses infinity)
- [ ] Update all callers to use consolidated APIs
- [ ] Delete `TypeRules.cs` if empty

### 1.3 Battle Test

```bash
dotnet build src/NumSharp.Core
dotnet test test/NumSharp.UnitTest --no-build -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"
```

- [ ] All tests pass
- [ ] No regressions in reduction operations
- [ ] No regressions in type promotion

---

## Part 2: np.iinfo (Integer Type Info)

### 2.1 Alignment Research

```python
# NumPy behavior to match
import numpy as np

print(np.iinfo(np.int32).bits)   # 32
print(np.iinfo(np.int32).min)    # -2147483648
print(np.iinfo(np.int32).max)    # 2147483647
print(np.iinfo(np.int32).dtype)  # int32
print(np.iinfo(np.int32).kind)   # 'i'

print(np.iinfo(np.uint8).bits)   # 8
print(np.iinfo(np.uint8).min)    # 0
print(np.iinfo(np.uint8).max)    # 255
print(np.iinfo(np.uint8).kind)   # 'u'

# Edge cases
np.iinfo(np.float64)  # ValueError: Invalid integer data type 'float64'
np.iinfo(np.bool_)    # Works in NumPy (bits=8, min=0, max=1)
```

**NumPy source:** `src/numpy/numpy/_core/getlimits.py:355-460`

### 2.2 Implementation

Create `src/NumSharp.Core/APIs/np.iinfo.cs`:

```csharp
public class iinfo
{
    public int bits { get; }
    public long min { get; }
    public long max { get; }  // Use long to hold uint64.MaxValue
    public NPTypeCode dtype { get; }
    public char kind { get; }  // 'i' or 'u'

    public iinfo(NPTypeCode typeCode) { ... }
    public iinfo(Type type) { ... }

    public override string ToString() { ... }
}
```

**Supported types:** Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char

### 2.3 Battle Test

Create `test/NumSharp.UnitTest/APIs/np.iinfo.Test.cs`:

```csharp
[Test] public async Task IInfo_Int32_Bits() => Assert.That(np.iinfo(NPTypeCode.Int32).bits).IsEqualTo(32);
[Test] public async Task IInfo_Int32_Min() => Assert.That(np.iinfo(NPTypeCode.Int32).min).IsEqualTo(int.MinValue);
[Test] public async Task IInfo_Int32_Max() => Assert.That(np.iinfo(NPTypeCode.Int32).max).IsEqualTo(int.MaxValue);
[Test] public async Task IInfo_UInt8_Min() => Assert.That(np.iinfo(NPTypeCode.Byte).min).IsEqualTo(0);
[Test] public async Task IInfo_Bool_Bits() => Assert.That(np.iinfo(NPTypeCode.Boolean).bits).IsEqualTo(8);
[Test] public async Task IInfo_Float_Throws() => Assert.That(() => np.iinfo(NPTypeCode.Double)).Throws<ArgumentException>();
```

---

## Part 3: np.finfo (Float Type Info)

### 3.1 Alignment Research

```python
import numpy as np

f64 = np.finfo(np.float64)
print(f64.bits)              # 64
print(f64.eps)               # 2.220446049250313e-16
print(f64.epsneg)            # 1.1102230246251565e-16
print(f64.max)               # 1.7976931348623157e+308
print(f64.min)               # -1.7976931348623157e+308
print(f64.tiny)              # 2.2250738585072014e-308 (smallest_normal)
print(f64.smallest_normal)   # 2.2250738585072014e-308
print(f64.smallest_subnormal)# 5e-324
print(f64.precision)         # 15
print(f64.resolution)        # 1e-15
print(f64.maxexp)            # 1024
print(f64.minexp)            # -1021
print(f64.dtype)             # float64

f32 = np.finfo(np.float32)
print(f32.bits)              # 32
print(f32.eps)               # 1.1920929e-07
print(f32.precision)         # 6

# Edge cases
np.finfo(np.int32)           # TypeError: data type int32 not inexact
np.finfo(np.complex128)      # Returns finfo for float64 component
```

**NumPy source:** `src/numpy/numpy/_core/getlimits.py:60-350`

### 3.2 Implementation

Create `src/NumSharp.Core/APIs/np.finfo.cs`:

```csharp
public class finfo
{
    public int bits { get; }
    public double eps { get; }
    public double epsneg { get; }
    public double max { get; }
    public double min { get; }
    public double tiny { get; }  // alias for smallest_normal
    public double smallest_normal { get; }
    public double smallest_subnormal { get; }
    public int precision { get; }
    public double resolution { get; }
    public int maxexp { get; }
    public int minexp { get; }
    public NPTypeCode dtype { get; }

    public finfo(NPTypeCode typeCode) { ... }
    public finfo(Type type) { ... }

    public override string ToString() { ... }
}
```

**Supported types:** Single, Double, Decimal (partial - no subnormal)

**C# constants to use:**
- `double.Epsilon` = smallest_subnormal (5e-324)
- `double.MaxValue`, `double.MinValue`
- `Math.BitIncrement(1.0) - 1.0` = eps

### 3.3 Battle Test

```csharp
[Test] public async Task FInfo_Float64_Bits() => Assert.That(np.finfo(NPTypeCode.Double).bits).IsEqualTo(64);
[Test] public async Task FInfo_Float64_Eps() => Assert.That(np.finfo(NPTypeCode.Double).eps).IsCloseTo(2.220446049250313e-16, 1e-30);
[Test] public async Task FInfo_Float32_Precision() => Assert.That(np.finfo(NPTypeCode.Single).precision).IsEqualTo(6);
[Test] public async Task FInfo_Int_Throws() => Assert.That(() => np.finfo(NPTypeCode.Int32)).Throws<ArgumentException>();
```

---

## Part 4: np.can_cast

### 4.1 Alignment Research

```python
import numpy as np

# Safe casting
print(np.can_cast(np.int32, np.int64))        # True
print(np.can_cast(np.int64, np.int32))        # False (loses precision)
print(np.can_cast(np.float32, np.float64))    # True
print(np.can_cast(np.int32, np.float64))      # True
print(np.can_cast(np.float64, np.int32))      # False

# Casting modes
print(np.can_cast(np.int32, np.int16, casting='safe'))      # False
print(np.can_cast(np.int32, np.int16, casting='same_kind')) # True
print(np.can_cast(np.int32, np.int16, casting='unsafe'))    # True
print(np.can_cast(np.int32, np.float32, casting='same_kind'))# False (different kind)

# With values (scalar context)
print(np.can_cast(1, np.int8))                # True (value fits)
print(np.can_cast(1000, np.int8))             # False (value doesn't fit)
print(np.can_cast(np.array([1]), np.int8))    # False (array context)
```

**NumPy source:** `src/numpy/numpy/_core/multiarray.py:601-660`

### 4.2 Implementation

Create `src/NumSharp.Core/Logic/np.can_cast.cs`:

```csharp
public static bool can_cast(NPTypeCode from, NPTypeCode to, string casting = "safe")
public static bool can_cast(Type from, Type to, string casting = "safe")
public static bool can_cast(object from_value, NPTypeCode to, string casting = "safe")
public static bool can_cast(NDArray from, NPTypeCode to, string casting = "safe")
```

**Casting modes:**
- `"no"` - no casting allowed
- `"equiv"` - only byte-order changes
- `"safe"` - no precision loss
- `"same_kind"` - safe within kind (int→int, float→float)
- `"unsafe"` - any cast allowed

### 4.3 Battle Test

```csharp
[Test] public async Task CanCast_Int32ToInt64_Safe() => Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Int64)).IsTrue();
[Test] public async Task CanCast_Int64ToInt32_Safe() => Assert.That(np.can_cast(NPTypeCode.Int64, NPTypeCode.Int32)).IsFalse();
[Test] public async Task CanCast_Int32ToFloat32_SameKind() => Assert.That(np.can_cast(NPTypeCode.Int32, NPTypeCode.Single, "same_kind")).IsFalse();
[Test] public async Task CanCast_ScalarFits() => Assert.That(np.can_cast(100, NPTypeCode.Byte)).IsTrue();
[Test] public async Task CanCast_ScalarOverflow() => Assert.That(np.can_cast(1000, NPTypeCode.Byte)).IsFalse();
```

---

## Part 5: np.result_type

### 5.1 Alignment Research

```python
import numpy as np

# Array + array
print(np.result_type(np.int32, np.int64))     # int64
print(np.result_type(np.int32, np.float32))   # float64
print(np.result_type(np.float32, np.float64)) # float64

# Scalar + array
print(np.result_type(1, np.array([1.0])))     # float64
print(np.result_type(1.0, np.array([1])))     # float64

# Multiple types
print(np.result_type(np.int8, np.int16, np.int32))  # int32

# With actual arrays
a = np.array([1, 2], dtype=np.int32)
b = np.array([1.0, 2.0], dtype=np.float32)
print(np.result_type(a, b))  # float64
```

**NumPy source:** `src/numpy/numpy/_core/multiarray.py:711-780`

### 5.2 Implementation

Make `_FindCommonType` public as `result_type`:

```csharp
public static NPTypeCode result_type(params object[] arrays_and_dtypes)
public static NPTypeCode result_type(params NPTypeCode[] types)
public static NPTypeCode result_type(params NDArray[] arrays)
```

### 5.3 Battle Test

```csharp
[Test] public async Task ResultType_Int32Int64() => Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Int64)).IsEqualTo(NPTypeCode.Int64);
[Test] public async Task ResultType_Int32Float32() => Assert.That(np.result_type(NPTypeCode.Int32, NPTypeCode.Single)).IsEqualTo(NPTypeCode.Double);
[Test] public async Task ResultType_Arrays() {
    var a = np.array(new int[] {1, 2});
    var b = np.array(new float[] {1.0f, 2.0f});
    Assert.That(np.result_type(a, b)).IsEqualTo(NPTypeCode.Double);
}
```

---

## Part 6: np.promote_types

### 6.1 Alignment Research

```python
import numpy as np

print(np.promote_types(np.int32, np.float32))   # float64
print(np.promote_types(np.int16, np.uint16))    # int32
print(np.promote_types(np.int8, np.int8))       # int8
print(np.promote_types(np.bool_, np.int32))     # int32
print(np.promote_types(np.float32, np.float64)) # float64

# Difference from result_type: doesn't consider values
# promote_types always returns the smallest safe type
```

**NumPy source:** C implementation, but behavior documented

### 6.2 Implementation

```csharp
public static NPTypeCode promote_types(NPTypeCode type1, NPTypeCode type2)
public static NPTypeCode promote_types(Type type1, Type type2)
```

Uses existing `_FindCommonType` logic but simplified for two types.

### 6.3 Battle Test

```csharp
[Test] public async Task PromoteTypes_Int32Float32() => Assert.That(np.promote_types(NPTypeCode.Int32, NPTypeCode.Single)).IsEqualTo(NPTypeCode.Double);
[Test] public async Task PromoteTypes_Int16UInt16() => Assert.That(np.promote_types(NPTypeCode.Int16, NPTypeCode.UInt16)).IsEqualTo(NPTypeCode.Int32);
[Test] public async Task PromoteTypes_Same() => Assert.That(np.promote_types(NPTypeCode.Int32, NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int32);
```

---

## Part 7: np.min_scalar_type

### 7.1 Alignment Research

```python
import numpy as np

print(np.min_scalar_type(10))         # uint8
print(np.min_scalar_type(-10))        # int8
print(np.min_scalar_type(1000))       # uint16
print(np.min_scalar_type(100000))     # uint32
print(np.min_scalar_type(1.0))        # float16 (or float32 if no float16)
print(np.min_scalar_type(1e100))      # float64
print(np.min_scalar_type(True))       # bool
```

**NumPy source:** `src/numpy/numpy/_core/multiarray.py:663-710`

### 7.2 Implementation

```csharp
public static NPTypeCode min_scalar_type(object value)
```

Logic:
- For integers: find smallest int/uint that holds value
- For floats: find smallest float that represents value exactly
- For bool: return Boolean

### 7.3 Battle Test

```csharp
[Test] public async Task MinScalarType_SmallPositive() => Assert.That(np.min_scalar_type(10)).IsEqualTo(NPTypeCode.Byte);
[Test] public async Task MinScalarType_SmallNegative() => Assert.That(np.min_scalar_type(-10)).IsEqualTo(NPTypeCode.Int16); // No Int8 in NumSharp
[Test] public async Task MinScalarType_Large() => Assert.That(np.min_scalar_type(100000)).IsEqualTo(NPTypeCode.UInt32);
[Test] public async Task MinScalarType_Bool() => Assert.That(np.min_scalar_type(true)).IsEqualTo(NPTypeCode.Boolean);
```

---

## Part 8: np.issubdtype

### 8.1 Alignment Research

```python
import numpy as np

# Integer hierarchy
print(np.issubdtype(np.int32, np.integer))      # True
print(np.issubdtype(np.int32, np.signedinteger))# True
print(np.issubdtype(np.uint32, np.unsignedinteger)) # True
print(np.issubdtype(np.int32, np.floating))     # False

# Float hierarchy
print(np.issubdtype(np.float32, np.floating))   # True
print(np.issubdtype(np.float32, np.inexact))    # True
print(np.issubdtype(np.float32, np.number))     # True

# Generic types
print(np.issubdtype(np.int32, np.number))       # True
print(np.issubdtype(np.bool_, np.integer))      # False (in NumPy 2.x)
print(np.issubdtype(np.bool_, np.generic))      # True
```

**NumPy source:** `src/numpy/numpy/_core/numerictypes.py:476-540`

### 8.2 Implementation

```csharp
public static bool issubdtype(NPTypeCode arg1, string arg2)  // "integer", "floating", etc.
public static bool issubdtype(NPTypeCode arg1, NPTypeCode arg2)
public static bool issubdtype(Type arg1, string arg2)
```

**Type categories:**
- `"generic"` - all types
- `"number"` - all numeric
- `"integer"` - all integers
- `"signedinteger"` - signed integers
- `"unsignedinteger"` - unsigned integers
- `"inexact"` - floats and complex
- `"floating"` - float types
- `"complexfloating"` - complex types

### 8.3 Battle Test

```csharp
[Test] public async Task IsSubdtype_Int32Integer() => Assert.That(np.issubdtype(NPTypeCode.Int32, "integer")).IsTrue();
[Test] public async Task IsSubdtype_Int32Floating() => Assert.That(np.issubdtype(NPTypeCode.Int32, "floating")).IsFalse();
[Test] public async Task IsSubdtype_Float64Number() => Assert.That(np.issubdtype(NPTypeCode.Double, "number")).IsTrue();
[Test] public async Task IsSubdtype_BoolInteger() => Assert.That(np.issubdtype(NPTypeCode.Boolean, "integer")).IsFalse();
```

---

## Part 9: np.common_type

### 9.1 Alignment Research

```python
import numpy as np

# Returns scalar type (not dtype)
print(np.common_type(np.array([1, 2], dtype=np.int32)))        # <class 'numpy.float64'>
print(np.common_type(np.array([1.0]), np.array([1+0j])))       # <class 'numpy.complex128'>
print(np.common_type(np.array([1], dtype=np.float32),
                     np.array([1], dtype=np.float64)))         # <class 'numpy.float64'>

# Integers always promote to at least float64
print(np.common_type(np.array([1], dtype=np.int8)))            # <class 'numpy.float64'>
```

**NumPy source:** `src/numpy/numpy/lib/_type_check_impl.py:658-720`

### 9.2 Implementation

```csharp
public static Type common_type(params NDArray[] arrays)
public static NPTypeCode common_type_code(params NDArray[] arrays)
```

Note: Always returns float type (minimum float64 for integers).

### 9.3 Battle Test

```csharp
[Test] public async Task CommonType_Int32() => Assert.That(np.common_type_code(np.array(new int[] {1, 2}))).IsEqualTo(NPTypeCode.Double);
[Test] public async Task CommonType_Float32Float64() {
    var a = np.array(new float[] {1.0f});
    var b = np.array(new double[] {1.0});
    Assert.That(np.common_type_code(a, b)).IsEqualTo(NPTypeCode.Double);
}
```

---

## Part 10: Type Checking Functions

### 10.1 Alignment Research

```python
import numpy as np

# issctype - is scalar type
print(np.issctype(np.int32))     # True
print(np.issctype(int))          # True
print(np.issctype(np.ndarray))   # False
print(np.issctype([1, 2, 3]))    # False

# isdtype - check dtype kind (NumPy 2.0+)
print(np.isdtype(np.int32, 'integral'))    # True
print(np.isdtype(np.float64, 'real floating')) # True
print(np.isdtype(np.int32, ('integral', 'real floating'))) # True

# issubsctype
print(np.issubsctype(np.int32, np.integer)) # True

# sctype2char
print(np.sctype2char(np.int32))  # 'i' (or 'l' on some platforms)
print(np.sctype2char(np.float64))# 'd'

# maximum_sctype
print(np.maximum_sctype(np.int32))   # <class 'numpy.int64'>
print(np.maximum_sctype(np.float32)) # <class 'numpy.float128'> or float64
```

### 10.2 Implementation

```csharp
public static bool issctype(object rep)
public static bool isdtype(NPTypeCode dtype, string kind)
public static bool isdtype(NPTypeCode dtype, string[] kinds)
public static bool issubsctype(NPTypeCode arg1, NPTypeCode arg2)
public static char sctype2char(NPTypeCode sctype)
public static NPTypeCode maximum_sctype(NPTypeCode t)
```

### 10.3 Battle Test

```csharp
[Test] public async Task IsSctype_Int32() => Assert.That(np.issctype(typeof(int))).IsTrue();
[Test] public async Task IsDtype_Int32Integral() => Assert.That(np.isdtype(NPTypeCode.Int32, "integral")).IsTrue();
[Test] public async Task Sctype2Char_Int32() => Assert.That(np.sctype2char(NPTypeCode.Int32)).IsEqualTo('i');
[Test] public async Task MaximumSctype_Int32() => Assert.That(np.maximum_sctype(NPTypeCode.Int32)).IsEqualTo(NPTypeCode.Int64);
```

---

## Part 11: Array Type Checking (isreal, iscomplex)

### 11.1 Alignment Research

```python
import numpy as np

# isreal - element-wise check for no imaginary part
print(np.isreal(np.array([1, 2, 3])))           # [True, True, True]
print(np.isreal(np.array([1+0j, 2+1j])))        # [True, False]
print(np.isreal(1.0))                           # True

# iscomplex - element-wise check for imaginary part
print(np.iscomplex(np.array([1+0j, 2+1j])))     # [False, True]

# isrealobj - check if object is non-complex type
print(np.isrealobj(np.array([1, 2])))           # True
print(np.isrealobj(np.array([1+0j])))           # False (dtype is complex)

# iscomplexobj - check if object is complex type
print(np.iscomplexobj(np.array([1+0j])))        # True
print(np.iscomplexobj(np.array([1, 2])))        # False
```

### 11.2 Implementation

```csharp
// For now, without complex support:
public static NDArray<bool> isreal(NDArray a)      // Always True for non-complex
public static NDArray<bool> iscomplex(NDArray a)   // Always False for non-complex
public static bool isrealobj(NDArray a)            // True if not complex dtype
public static bool iscomplexobj(NDArray a)         // False if not complex dtype
```

### 11.3 Battle Test

```csharp
[Test] public async Task IsReal_IntArray() {
    var a = np.array(new int[] {1, 2, 3});
    Assert.That(np.isreal(a).all()).IsTrue();
}
[Test] public async Task IsRealObj_IntArray() => Assert.That(np.isrealobj(np.array(new int[] {1, 2}))).IsTrue();
[Test] public async Task IsComplexObj_IntArray() => Assert.That(np.iscomplexobj(np.array(new int[] {1, 2}))).IsFalse();
```

---

## Summary Checklist

| Part | Feature | Status |
|------|---------|--------|
| 1 | Consolidate Internal Type Info | [ ] |
| 2 | np.iinfo | [ ] |
| 3 | np.finfo | [ ] |
| 4 | np.can_cast | [ ] |
| 5 | np.result_type | [ ] |
| 6 | np.promote_types | [ ] |
| 7 | np.min_scalar_type | [ ] |
| 8 | np.issubdtype | [ ] |
| 9 | np.common_type | [ ] |
| 10 | Type Checking (issctype, isdtype, etc.) | [ ] |
| 11 | Array Type Checking (isreal, iscomplex) | [ ] |

---

## Execution Order

**Recommended order:**
1. Part 1 (consolidate) - foundation cleanup
2. Part 2 (iinfo) - simple, high value
3. Part 3 (finfo) - simple, high value
4. Part 8 (issubdtype) - needed by can_cast
5. Part 4 (can_cast) - depends on issubdtype
6. Part 5 (result_type) - expose existing internal
7. Part 6 (promote_types) - uses result_type logic
8. Part 7 (min_scalar_type) - standalone
9. Part 9 (common_type) - uses promote logic
10. Part 10 (type checking functions) - standalone
11. Part 11 (isreal/iscomplex) - last, needs complex support later

**Per-part workflow:**
1. Research NumPy behavior (run Python, read source)
2. Implement in NumSharp
3. Write tests based on NumPy output
4. Run tests, fix issues
5. Commit with descriptive message
