# NumSharp-NumPy Alignment Plan

**Document Version**: 1.0
**Created**: 2024
**Target**: NumPy 2.x API parity
**Scope**: All operator, function, and protocol gaps

---

## Executive Summary

NumSharp currently implements **34 of 54** NumPy operator/protocol features (63%). This plan covers closing the remaining **20 feature gaps** organized into 6 phases over an estimated 4-6 weeks of focused development.

### Gap Breakdown

| Category | Missing | Effort | Feasibility |
|----------|---------|--------|-------------|
| In-place operators | 12 | N/A | ❌ C# limitation |
| Float functions | 7 | Low | ✅ Straightforward |
| Math functions | 3 | Low | ✅ Straightforward |
| Ufunc infrastructure | 5 | High | ⚠️ Architecture change |
| Container protocol | 2 | Medium | ✅ Doable |
| Type conversions | 2 | Medium | ⚠️ Complex type needed |

---

## Phase 0: In-Place Operators (NOT IMPLEMENTABLE)

### Why C# Cannot Support In-Place Operators

C# does not allow overloading compound assignment operators (`+=`, `-=`, etc.). When you write:

```csharp
arr += 5;
```

The compiler expands this to:

```csharp
arr = arr + 5;
```

This means:
1. A **new** NDArray is created by `arr + 5`
2. The reference `arr` is reassigned to point to the new array
3. The **original** array is unchanged (and may be garbage collected)

### NumPy Behavior

```python
arr = np.array([1, 2, 3])
original_id = id(arr)
arr += 5
assert id(arr) == original_id  # Same object, modified in-place!
```

### NumSharp Workarounds

#### Option A: Explicit Methods (Recommended)

```csharp
// Instead of: arr += 5
arr.AddInPlace(5);
arr.SubtractInPlace(5);
arr.MultiplyInPlace(5);
// etc.
```

#### Option B: Extension Methods with ref

```csharp
public static void AddInPlace(ref NDArray arr, object value)
{
    np.add(arr, value, out: arr);  // Requires out parameter support
}

// Usage:
arr.AddInPlace(ref arr, 5);  // Awkward syntax
```

#### Option C: Document the Difference

Simply document that `arr += 5` creates a new array in NumSharp (unlike NumPy).

### Recommendation

**Option A** - Add explicit in-place methods:

```csharp
public partial class NDArray
{
    public void AddInPlace(object value) => np.add(this, value, out: this);
    public void SubtractInPlace(object value) => np.subtract(this, value, out: this);
    public void MultiplyInPlace(object value) => np.multiply(this, value, out: this);
    public void DivideInPlace(object value) => np.divide(this, value, out: this);
    public void ModInPlace(object value) => np.mod(this, value, out: this);
    public void PowerInPlace(object value) => np.power(this, value, out: this);
    public void FloorDivideInPlace(object value) => np.floor_divide(this, value, out: this);
    public void BitwiseAndInPlace(object value) => np.bitwise_and(this, value, out: this);
    public void BitwiseOrInPlace(object value) => np.bitwise_or(this, value, out: this);
    public void BitwiseXorInPlace(object value) => np.bitwise_xor(this, value, out: this);
    public void LeftShiftInPlace(object value) => np.left_shift(this, value, out: this);
    public void RightShiftInPlace(object value) => np.right_shift(this, value, out: this);
}
```

**Prerequisite**: Requires `out` parameter support in ufuncs (Phase 5).

### Status: DEFERRED to Phase 5

---

## Phase 1: Float-Specific Functions

**Estimated Effort**: 2-3 days
**Priority**: Low (rarely used, but easy wins)
**Dependencies**: None

### 1.1 `np.signbit(x)`

Returns True where the sign bit is set (negative).

```python
np.signbit([-1.0, 0.0, 1.0, -0.0, np.inf, -np.inf, np.nan])
# [True, False, False, True, False, True, False]
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Logic/np.signbit.cs
public static NDArray<bool> signbit(NDArray x)
{
    if (x.typecode != NPTypeCode.Single && x.typecode != NPTypeCode.Double)
        throw new ArgumentException("signbit requires float dtype");

    var result = new NDArray<bool>(x.shape);
    // Use BitConverter to check sign bit
    // float: bit 31, double: bit 63
}
```

**Test Cases**:
- Positive numbers → False
- Negative numbers → True
- Positive zero → False
- Negative zero → True (!)
- +inf → False
- -inf → True
- NaN → Implementation-defined (check sign bit)

### 1.2 `np.isposinf(x)` and `np.isneginf(x)`

Test for positive/negative infinity specifically.

```python
np.isposinf([np.inf, -np.inf, 0, 1, np.nan])  # [True, False, False, False, False]
np.isneginf([np.inf, -np.inf, 0, 1, np.nan])  # [False, True, False, False, False]
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Logic/np.isinf_extended.cs
public static NDArray<bool> isposinf(NDArray x)
{
    return np.logical_and(np.isinf(x), np.greater(x, 0));
}

public static NDArray<bool> isneginf(NDArray x)
{
    return np.logical_and(np.isinf(x), np.less(x, 0));
}
```

**Test Cases**:
- +inf → isposinf=True, isneginf=False
- -inf → isposinf=False, isneginf=True
- Finite numbers → Both False
- NaN → Both False

### 1.3 `np.copysign(x, y)`

Copy sign of `y` to magnitude of `x`.

```python
np.copysign(1.0, -2.0)   # -1.0
np.copysign(-1.0, 2.0)   # 1.0
np.copysign([1, -1], -1) # [-1, -1]
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.copysign.cs
public static NDArray copysign(NDArray x, NDArray y)
{
    // Use Math.CopySign for scalar, vectorize for arrays
    // Handles: float, double
    // Broadcasting: yes
}
```

**Edge Cases**:
- `copysign(NaN, -1)` → -NaN (sign bit set)
- `copysign(inf, -1)` → -inf
- `copysign(0, -1)` → -0.0

### 1.4 `np.fabs(x)`

Absolute value for floating-point only (unlike `np.abs` which works on all types).

```python
np.fabs([-1.5, 0.0, 1.5])  # [1.5, 0.0, 1.5]
np.fabs(np.array([-1], dtype=np.int32))  # TypeError!
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.fabs.cs
public static NDArray fabs(NDArray x)
{
    if (x.typecode != NPTypeCode.Single && x.typecode != NPTypeCode.Double)
        throw new TypeError("fabs requires float dtype");

    return np.abs(x);  // Delegate to existing abs
}
```

### 1.5 `np.nextafter(x, y)`

Returns the next representable floating-point value after `x` toward `y`.

```python
np.nextafter(1.0, 2.0)  # 1.0000000000000002
np.nextafter(1.0, 0.0)  # 0.9999999999999999
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.nextafter.cs
public static NDArray nextafter(NDArray x, NDArray y)
{
    // Use BitConverter to increment/decrement mantissa
    // Or use Math.BitIncrement/BitDecrement (.NET 5+)
}
```

### 1.6 `np.spacing(x)`

Returns the distance to the next representable floating-point value.

```python
np.spacing(1.0)      # 2.220446049250313e-16 (machine epsilon at 1.0)
np.spacing(1e10)     # 1.9073486328125e-06 (larger spacing at larger values)
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.spacing.cs
public static NDArray spacing(NDArray x)
{
    // spacing(x) = nextafter(x, inf) - x
    return np.subtract(np.nextafter(x, np.inf), x);
}
```

### Phase 1 Deliverables

| File | Functions |
|------|-----------|
| `np.signbit.cs` | `signbit` |
| `np.isinf_extended.cs` | `isposinf`, `isneginf` |
| `np.copysign.cs` | `copysign` |
| `np.fabs.cs` | `fabs` |
| `np.nextafter.cs` | `nextafter` |
| `np.spacing.cs` | `spacing` |

### Phase 1 Tests

```
test/NumSharp.UnitTest/Math/
├── SignbitTests.cs
├── IsposinfTests.cs
├── IsneginfTests.cs
├── CopysignTests.cs
├── FabsTests.cs
├── NextafterTests.cs
└── SpacingTests.cs
```

---

## Phase 2: Math Functions

**Estimated Effort**: 1-2 days
**Priority**: Low
**Dependencies**: None

### 2.1 `np.divmod(x, y)`

Returns tuple of `(quotient, remainder)` = `(x // y, x % y)`.

```python
np.divmod(7, 3)           # (2, 1)
np.divmod([7, 8, 9], 3)   # (array([2, 2, 3]), array([1, 2, 0]))
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.divmod.cs
public static (NDArray quotient, NDArray remainder) divmod(NDArray x, NDArray y)
{
    return (np.floor_divide(x, y), np.mod(x, y));
}

// Tuple return matches NumPy's behavior
```

**Note**: C# tuples work well here. NumPy returns a tuple of two arrays.

### 2.2 `np.ldexp(x, i)`

Computes `x * 2^i` efficiently using floating-point representation.

```python
np.ldexp(1.0, 3)   # 8.0 (1.0 * 2^3)
np.ldexp(0.5, 4)   # 8.0 (0.5 * 2^4)
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.ldexp.cs
public static NDArray ldexp(NDArray x, NDArray i)
{
    // Use Math.ScaleB (.NET 5+) or manual bit manipulation
    // ldexp(x, i) = x * 2^i
}
```

### 2.3 `np.frexp(x)`

Decomposes floating-point into `(mantissa, exponent)` where `x = mantissa * 2^exponent`.

```python
np.frexp(8.0)   # (0.5, 4) because 0.5 * 2^4 = 8
np.frexp([1, 2, 4])  # (array([0.5, 0.5, 0.5]), array([1, 2, 3]))
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.frexp.cs
public static (NDArray mantissa, NDArray exponent) frexp(NDArray x)
{
    // Use bit manipulation to extract mantissa and exponent
    // Or use Math operations
}
```

### Phase 2 Deliverables

| File | Functions |
|------|-----------|
| `np.divmod.cs` | `divmod` |
| `np.ldexp.cs` | `ldexp` |
| `np.frexp.cs` | `frexp` |

---

## Phase 3: Container Protocol

**Estimated Effort**: 1-2 days
**Priority**: Medium
**Dependencies**: None

### 3.1 `__contains__` (the `in` operator)

NumPy supports:

```python
2 in np.array([1, 2, 3])  # True
5 in np.array([1, 2, 3])  # False
```

**Challenge**: C# doesn't have an `in` operator that can be overloaded.

**Options**:

#### Option A: ICollection<T> Interface (Not Recommended)

```csharp
public partial class NDArray : ICollection<object>
{
    public bool Contains(object item) { ... }
}
```

**Problem**: NDArray is not generic, and `ICollection<T>` requires a single type.

#### Option B: Explicit Contains Method (Recommended)

```csharp
// File: src/NumSharp.Core/Backends/NDArray.Contains.cs
public partial class NDArray
{
    /// <summary>
    /// Returns true if value is found in the array (linear search).
    /// Equivalent to NumPy's `value in arr`.
    /// </summary>
    public bool Contains(object value)
    {
        var scalar = np.asanyarray(value);
        var comparison = this == scalar;
        return np.any(comparison);
    }
}
```

**Usage**:

```csharp
// NumPy: 2 in arr
// NumSharp: arr.Contains(2)
if (arr.Contains(2)) { ... }
```

### 3.2 `__hash__` (Proper Behavior)

NumPy arrays are **unhashable** because they're mutable:

```python
hash(np.array([1, 2, 3]))  # TypeError: unhashable type: 'numpy.ndarray'
```

NumSharp currently returns a hash, which is **incorrect**.

**Fix**:

```csharp
// File: src/NumSharp.Core/Backends/NDArray.cs
public override int GetHashCode()
{
    throw new NotSupportedException(
        "NDArray is unhashable because it is mutable. " +
        "Use arr.tobytes().GetHashCode() for a hashable representation, " +
        "or convert to a tuple with arr.ToArray().");
}
```

**Breaking Change**: Code using NDArray as dictionary keys will break.

**Migration**:

```csharp
// Before (broken):
var dict = new Dictionary<NDArray, int>();
dict[arr] = 5;

// After (correct):
var dict = new Dictionary<string, int>();
dict[arr.tobytes()] = 5;  // Use bytes as key

// Or use reference equality:
var dict = new Dictionary<NDArray, int>(ReferenceEqualityComparer.Instance);
```

### Phase 3 Deliverables

| File | Changes |
|------|---------|
| `NDArray.Contains.cs` | Add `Contains(object)` method |
| `NDArray.cs` | Override `GetHashCode()` to throw |

---

## Phase 4: Type Conversions

**Estimated Effort**: 3-5 days (complex number support is substantial)
**Priority**: Medium
**Dependencies**: None

### 4.1 Complex Number Support

NumPy has full complex number support:

```python
c = np.array([1+2j, 3+4j])
c.dtype        # complex128
c.real         # array([1., 3.])
c.imag         # array([2., 4.])
np.abs(c)      # array([2.236, 5.0])
c * c          # array([-3+4j, -7+24j])
complex(np.array(1+2j))  # (1+2j)
```

**Implementation Scope**:

#### 4.1.1 Add NPTypeCode.Complex128

```csharp
// File: src/NumSharp.Core/Utilities/NPTypeCode.cs
public enum NPTypeCode
{
    // ... existing types ...
    Complex128 = 14,  // System.Numerics.Complex
}
```

#### 4.1.2 Add Complex Storage Support

```csharp
// Modify UnmanagedStorage to handle Complex
// Complex is 16 bytes (two doubles)
```

#### 4.1.3 Add Complex Operators

All arithmetic operators need complex support:
- `+`, `-`, `*`, `/` (complex arithmetic)
- `abs()` (returns magnitude)
- `conjugate()` (complex conjugate)
- `.real`, `.imag` properties

#### 4.1.4 Add Complex Conversion

```csharp
// File: src/NumSharp.Core/Casting/Implicit/NdArray.Implicit.ValueTypes.cs
public static implicit operator NDArray(Complex d) => NDArray.Scalar(d);

public static explicit operator Complex(NDArray nd)
{
    if (nd.ndim != 0)
        throw new IncorrectShapeException();
    return Converts.ChangeType<Complex>(nd.Storage.GetAtIndex(0));
}
```

**Effort**: This is a substantial change affecting:
- Type system
- Storage
- All arithmetic operations
- Serialization

**Recommendation**: Create separate issue/PR for complex number support.

### 4.2 `__index__` Protocol

NumPy's `__index__` allows scalar arrays to be used as indices:

```python
idx = np.array(2)
lst = [10, 20, 30]
lst[idx]  # 30 - uses __index__
```

**C# Equivalent**: Implicit conversion to `int` or `long`.

**Current State**: Already works via explicit conversion:

```csharp
int idx = (int)np.array(2);
var value = list[idx];
```

**Enhancement**: Could add implicit conversion for 0-d integer arrays only:

```csharp
// This is controversial - implicit conversions can be surprising
public static implicit operator int(NDArray nd)
{
    if (nd.ndim != 0)
        throw new IncorrectShapeException("Only 0-d arrays can be used as indices");
    if (!nd.IsIntegerType)
        throw new TypeError("Only integer arrays can be used as indices");
    return (int)nd;
}
```

**Recommendation**: Keep as explicit conversion. Document the pattern.

### Phase 4 Deliverables

| Feature | Scope | Recommendation |
|---------|-------|----------------|
| Complex numbers | Large | Separate PR |
| `__index__` | Small | Document only |

---

## Phase 5: Ufunc Infrastructure

**Estimated Effort**: 2-4 weeks
**Priority**: High (enables many features)
**Dependencies**: None

### 5.1 `out` Parameter

The `out` parameter allows writing results to a pre-allocated array:

```python
out = np.zeros(3)
np.add([1,2,3], [4,5,6], out=out)
# out is now [5, 7, 9]
```

**Benefits**:
- Avoids allocation
- Enables in-place operations
- Required for `+=` workaround

**Implementation**:

#### 5.1.1 Function Signature Changes

```csharp
// Current:
public static NDArray add(NDArray x1, NDArray x2)

// New:
public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null)
{
    var result = @out ?? new NDArray(resultType, resultShape);
    // Write to result instead of allocating new
    return result;
}
```

#### 5.1.2 TensorEngine Changes

```csharp
// Current:
public abstract NDArray Add(NDArray lhs, NDArray rhs);

// New:
public abstract NDArray Add(NDArray lhs, NDArray rhs, NDArray @out = null);
```

#### 5.1.3 ILKernelGenerator Changes

Kernels need to accept output pointer:

```csharp
// Current kernel signature:
void Kernel(void* lhs, void* rhs, void* result, long size)

// No change needed - already writes to result pointer
// Just need to pass out.Address instead of allocating
```

**Affected Functions** (all ufuncs):
- Arithmetic: `add`, `subtract`, `multiply`, `divide`, `mod`, `power`, `floor_divide`
- Comparison: `equal`, `not_equal`, `less`, `greater`, `less_equal`, `greater_equal`
- Bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift`
- Unary: `negative`, `positive`, `abs`, `sqrt`, `exp`, `log`, etc.

### 5.2 `where` Parameter

Conditional application of ufunc:

```python
np.add([1,2,3], 10, where=[True, False, True])
# Result: [11, 0, 13] - only applies where mask is True
```

**Implementation**:

```csharp
public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
{
    var result = @out ?? new NDArray(resultType, resultShape);

    if (where != null)
    {
        // Only compute where mask is True
        // Leave other elements unchanged (or zero if out is null)
    }

    return result;
}
```

**Complexity**: Requires masked iteration in kernels.

### 5.3 Ufunc Object Model

NumPy ufuncs are objects with methods:

```python
np.add.reduce([1,2,3,4,5])      # 15 (sum)
np.add.accumulate([1,2,3,4,5])  # [1,3,6,10,15] (cumsum)
np.add.outer([1,2], [10,20,30]) # [[11,21,31], [12,22,32]]
np.add.at(arr, [0,2], 1)        # In-place add at indices
```

**Implementation Options**:

#### Option A: Static Methods (Simple)

```csharp
public static class np
{
    public static NDArray add_reduce(NDArray arr, int? axis = null) => np.sum(arr, axis);
    public static NDArray add_accumulate(NDArray arr, int? axis = null) => np.cumsum(arr, axis);
    public static NDArray add_outer(NDArray a, NDArray b) => np.outer(a, b);
    public static void add_at(NDArray arr, NDArray indices, NDArray values) { ... }
}
```

**Con**: Doesn't scale - need separate methods for each ufunc.

#### Option B: Ufunc Class (NumPy-like)

```csharp
public class Ufunc
{
    public string Name { get; }
    public Func<NDArray, NDArray, NDArray> BinaryOp { get; }

    public NDArray reduce(NDArray arr, int? axis = null) { ... }
    public NDArray accumulate(NDArray arr, int? axis = null) { ... }
    public NDArray outer(NDArray a, NDArray b) { ... }
    public void at(NDArray arr, NDArray indices, NDArray values) { ... }
}

public static class np
{
    public static Ufunc add = new Ufunc("add", (a, b) => TensorEngine.Add(a, b));
    public static Ufunc multiply = new Ufunc("multiply", (a, b) => TensorEngine.Multiply(a, b));
    // ...
}

// Usage:
np.add.reduce(arr);
np.multiply.accumulate(arr);
```

**Pro**: Matches NumPy API exactly.
**Con**: Significant refactoring.

#### Option C: Hybrid (Recommended)

Keep current static methods, add ufunc-style methods where needed:

```csharp
// Existing (keep):
np.sum(arr)       // add.reduce
np.cumsum(arr)    // add.accumulate
np.outer(a, b)    // multiply.outer

// Add missing:
np.add_at(arr, indices, values)  // In-place at indices
```

### 5.4 `ufunc.at()` - In-Place at Indices

The most useful missing ufunc method:

```python
arr = np.array([1, 2, 3, 4, 5])
np.add.at(arr, [0, 2, 4], 10)
# arr is now [11, 2, 13, 4, 15]

# Unlike arr[[0,2,4]] += 10, this handles repeated indices correctly:
arr = np.zeros(3)
np.add.at(arr, [0, 0, 0], 1)
# arr is [3, 0, 0] - added 1 three times to index 0
```

**Implementation**:

```csharp
// File: src/NumSharp.Core/Math/np.ufunc_at.cs
public static void add_at(NDArray arr, NDArray indices, NDArray values)
{
    // Validate inputs
    // Iterate over indices, apply operation at each
    // Handle repeated indices correctly (unlike fancy indexing)
}

public static void subtract_at(NDArray arr, NDArray indices, NDArray values) { ... }
public static void multiply_at(NDArray arr, NDArray indices, NDArray values) { ... }
// etc.
```

### Phase 5 Deliverables

| Feature | Files | Effort |
|---------|-------|--------|
| `out` parameter | All ufunc files | 1 week |
| `where` parameter | All ufunc files | 3-5 days |
| `*_at` methods | New file | 2-3 days |
| Ufunc object model | Major refactor | 1-2 weeks (optional) |

### Phase 5 Recommended Order

1. `out` parameter (enables in-place workaround)
2. `*_at` methods (commonly needed)
3. `where` parameter (nice to have)
4. Ufunc object model (future consideration)

---

## Phase 6: NumPy 2.x Compatibility

**Estimated Effort**: 1-2 days
**Priority**: High (correctness)
**Dependencies**: None

### 6.1 Boolean Array Restrictions

NumPy 2.x removed support for:

| Operation | NumPy 1.x | NumPy 2.x | NumSharp Should |
|-----------|-----------|-----------|-----------------|
| `bool - bool` | Worked | TypeError | Throw |
| `-bool` | Worked | TypeError | Throw |

**Implementation**:

```csharp
// File: src/NumSharp.Core/Operations/Elementwise/NDArray.Primitive.cs
public static NDArray operator -(NDArray x)
{
    if (x.typecode == NPTypeCode.Boolean)
        throw new TypeError(
            "The numpy boolean negative, the `-` operator, is not supported, " +
            "use the `~` operator or the logical_not function instead.");

    return x.TensorEngine.Negate(x);
}

// Also in TensorEngine.Subtract for bool - bool
```

### 6.2 Boolean Arithmetic Returns Bool

NumPy 2.x changed:

```python
# NumPy 1.x:
np.array([True]) + np.array([True])  # array([2], dtype=int64)

# NumPy 2.x:
np.array([True]) + np.array([True])  # array([True], dtype=bool)
```

**Note**: Verify NumSharp's current behavior and align if needed.

---

## Implementation Schedule

### Week 1: Float & Math Functions (Phase 1 + 2)

| Day | Tasks |
|-----|-------|
| 1 | `signbit`, `isposinf`, `isneginf` |
| 2 | `copysign`, `fabs` |
| 3 | `nextafter`, `spacing` |
| 4 | `divmod`, `ldexp`, `frexp` |
| 5 | Tests for all above |

### Week 2: Container Protocol + NumPy 2.x (Phase 3 + 6)

| Day | Tasks |
|-----|-------|
| 1 | `Contains()` method |
| 2 | `GetHashCode()` throws + migration docs |
| 3 | Boolean operator restrictions |
| 4 | Boolean arithmetic dtype fix |
| 5 | Tests + documentation |

### Week 3-4: Out Parameter (Phase 5.1)

| Day | Tasks |
|-----|-------|
| 1-2 | Design out parameter flow |
| 3-5 | Implement for binary ops |
| 6-7 | Implement for unary ops |
| 8-9 | Implement for comparisons |
| 10 | Tests |

### Week 5: At Methods + Where (Phase 5.2 + 5.4)

| Day | Tasks |
|-----|-------|
| 1-2 | `add_at`, `subtract_at`, etc. |
| 3-4 | `where` parameter |
| 5 | Tests |

### Week 6: Complex Numbers (Phase 4 - Optional)

| Day | Tasks |
|-----|-------|
| 1 | NPTypeCode.Complex128 |
| 2 | Storage support |
| 3 | Arithmetic operators |
| 4 | Functions (abs, conjugate) |
| 5 | Tests |

---

## Test Plan

### Test File Structure

```
test/NumSharp.UnitTest/
├── Math/
│   ├── SignbitTests.cs
│   ├── CopysignTests.cs
│   ├── NextafterTests.cs
│   ├── SpacingTests.cs
│   ├── DivmodTests.cs
│   ├── LdexpTests.cs
│   └── FrexpTests.cs
├── Logic/
│   ├── IsposinfTests.cs
│   └── IsneginfTests.cs
├── Container/
│   ├── ContainsTests.cs
│   └── HashTests.cs
├── Ufunc/
│   ├── OutParameterTests.cs
│   ├── WhereParameterTests.cs
│   └── AtMethodTests.cs
└── Compatibility/
    └── NumPy2xTests.cs
```

### Test Categories

```csharp
[Category("Phase1_FloatFunctions")]
[Category("Phase2_MathFunctions")]
[Category("Phase3_Container")]
[Category("Phase4_TypeConversions")]
[Category("Phase5_Ufunc")]
[Category("Phase6_NumPy2x")]
```

### Running Phase Tests

```bash
# Run specific phase
dotnet test -- --treenode-filter "/*/*/*/*[Category=Phase1_FloatFunctions]"
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| `GetHashCode` breaking change | High | Document migration path |
| Complex numbers scope creep | Medium | Separate PR |
| Out parameter performance | Low | Benchmark before/after |
| Ufunc object model complexity | Medium | Defer to future |

---

## Success Criteria

### Phase Completion Checklist

- [ ] **Phase 1**: All 7 float functions implemented and tested
- [ ] **Phase 2**: All 3 math functions implemented and tested
- [ ] **Phase 3**: `Contains()` works, `GetHashCode()` throws
- [ ] **Phase 4**: Complex numbers deferred or implemented
- [ ] **Phase 5**: `out` parameter works for all ufuncs
- [ ] **Phase 6**: NumPy 2.x boolean restrictions enforced

### Final Gap Count

| Before | After | Reduction |
|--------|-------|-----------|
| 20 missing | 8 missing | 60% closed |

*Remaining 8 = 12 in-place operators (C# limitation) - 4 complex (deferred)*

---

## Appendix: File Locations

### New Files to Create

```
src/NumSharp.Core/
├── Logic/
│   ├── np.signbit.cs
│   ├── np.isposinf.cs
│   └── np.isneginf.cs
├── Math/
│   ├── np.copysign.cs
│   ├── np.fabs.cs
│   ├── np.nextafter.cs
│   ├── np.spacing.cs
│   ├── np.divmod.cs
│   ├── np.ldexp.cs
│   ├── np.frexp.cs
│   └── np.ufunc_at.cs
└── Backends/
    └── NDArray.Contains.cs
```

### Files to Modify

```
src/NumSharp.Core/
├── Backends/NDArray.cs                    # GetHashCode override
├── Operations/Elementwise/NDArray.Primitive.cs  # Boolean restrictions
├── Math/np.add.cs                         # out parameter
├── Math/np.subtract.cs                    # out parameter
├── Math/np.multiply.cs                    # out parameter
├── Math/np.divide.cs                      # out parameter
└── [All other ufunc files]                # out + where parameters
```
