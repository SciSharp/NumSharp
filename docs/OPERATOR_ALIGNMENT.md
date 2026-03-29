# NumSharp Operator Alignment with NumPy

This document maps all NumPy operators to NumSharp, documenting expected behaviors and implementation status.

## Overview

NumPy defines operators via:
1. **PyNumberMethods** (`number.c`) - Arithmetic, bitwise, conversions
2. **tp_richcompare** (`arrayobject.c`) - Comparison operators
3. **NDArrayOperatorsMixin** (`mixins.py`) - Operator to ufunc mapping

All operators delegate to ufuncs for actual computation.

---

## 1. Arithmetic Operators

| Python | C# | NumPy ufunc | NumSharp Status |
|--------|-----|-------------|-----------------|
| `+` (binary) | `+` | `np.add` | ✅ object pattern (3 overloads) |
| `-` (binary) | `-` | `np.subtract` | ✅ object pattern (3 overloads) |
| `*` | `*` | `np.multiply` | ✅ object pattern (3 overloads) |
| `/` | `/` | `np.true_divide` | ✅ object pattern (3 overloads) |
| `//` | N/A | `np.floor_divide` | ✅ np.floor_divide() only |
| `%` | `%` | `np.remainder` | ✅ object pattern (3 overloads) |
| `**` | N/A | `np.power` | ✅ np.power() only |
| `@` | N/A | `np.matmul` | ✅ np.matmul() only |
| `-` (unary) | `-` | `np.negative` | ✅ Implemented |
| `+` (unary) | `+` | `np.positive` | ✅ Implemented (returns copy) |
| `abs()` | N/A | `np.absolute` | ✅ np.abs() only |
| `divmod()` | N/A | `np.divmod` | ❌ Not implemented |

### Behavior Matrix

```
Operation       | int32 + int32  | int32 + float  | int32 / int32
----------------|----------------|----------------|---------------
NumPy           | int32          | float64        | float64
NumSharp        | int32          | float64        | float64
```

### Reflected Operators (`__radd__`, `__rsub__`, etc.)

NumPy: `5 + arr` calls `arr.__radd__(5)` → returns `ndarray`

NumSharp: Uses `operator +(object left, NDArray right)` → `np.asanyarray(left) + right`

This matches NumPy's behavior exactly - any type is accepted and converted via `np.asanyarray`.

---

## 2. Comparison Operators

| Python | C# | NumPy ufunc | NumSharp Status |
|--------|-----|-------------|-----------------|
| `==` | `==` | `np.equal` | ✅ object pattern, returns `NDArray<bool>` |
| `!=` | `!=` | `np.not_equal` | ✅ object pattern, returns `NDArray<bool>` |
| `<` | `<` | `np.less` | ✅ object pattern, returns `NDArray<bool>` |
| `<=` | `<=` | `np.less_equal` | ✅ object pattern, returns `NDArray<bool>` |
| `>` | `>` | `np.greater` | ✅ object pattern, returns `NDArray<bool>` |
| `>=` | `>=` | `np.greater_equal` | ✅ object pattern, returns `NDArray<bool>` |

### Identity vs Equality

```python
# NumPy
arr == None    # Returns array([False, False, False]) - element-wise!
arr is None    # Returns False - identity check
```

```csharp
// NumSharp (correct pattern)
arr == null    // Returns NDArray<bool> - element-wise!
arr is null    // Returns bool - identity check (use this for null checks)
```

**Key Insight**: Using `== null` for null checking is semantically wrong in both NumPy and NumSharp. Both should use identity checks (`is None` / `is null`).

---

## 3. Bitwise Operators

| Python | C# | NumPy ufunc | NumSharp Status |
|--------|-----|-------------|-----------------|
| `&` | `&` | `np.bitwise_and` | ✅ object pattern (3 overloads) |
| `\|` | `\|` | `np.bitwise_or` | ✅ object pattern (3 overloads) |
| `^` | `^` | `np.bitwise_xor` | ✅ object pattern (3 overloads) |
| `~` | `~` | `np.invert` | ✅ Implemented (unary) |
| `<<` | `<<` | `np.left_shift` | ✅ np.left_shift() function |
| `>>` | `>>` | `np.right_shift` | ✅ np.right_shift() function |

### Boolean Array Behavior

```python
# NumPy
arr_bool & True   # [True, False, True] - element-wise AND
~arr_bool         # [False, True, False] - element-wise NOT
```

The `~` operator works on both integer arrays (bitwise NOT) and boolean arrays (logical NOT).

---

## 4. Type Conversion Operators

| Python | C# | NumPy Behavior | NumSharp Status |
|--------|-----|----------------|-----------------|
| `int(arr)` | `(int)arr` | Explicit, scalar only | ✅ `explicit` (implemented) |
| `float(arr)` | `(double)arr` | Explicit, scalar only | ✅ `explicit` (implemented) |
| `bool(arr)` | `(bool)arr` | Explicit, scalar only, raises for multi-element | ✅ `explicit` (implemented) |
| `complex(arr)` | N/A | Explicit, scalar only | ❌ No complex support |
| `__index__` | N/A | For array indexing | ❌ Not implemented |

**All 12 supported types** have:
- `implicit` conversion: scalar → NDArray (safe, creates 0-d array)
- `explicit` conversion: NDArray → scalar (requires 0-d, throws `IncorrectShapeException` otherwise)

### Scalar Extraction Rules

```python
# NumPy
scalar = np.array(5)     # 0-dimensional array
int(scalar)              # 5 - OK
float(scalar)            # 5.0 - OK
bool(scalar)             # True - OK

arr = np.array([1, 2])   # 1-dimensional array
int(arr)                 # ValueError!
bool(arr)                # ValueError: ambiguous, use .any() or .all()
```

**NumSharp Alignment**: FROM-NDArray conversions are `explicit` to match NumPy's `int(arr)` pattern.

---

## 5. In-Place Operators

| Python | C# | NumPy ufunc | NumSharp Status |
|--------|-----|-------------|-----------------|
| `+=` | N/A | `np.add` with `out=` | ❌ Not supported |
| `-=` | N/A | `np.subtract` with `out=` | ❌ Not supported |
| `*=` | N/A | `np.multiply` with `out=` | ❌ Not supported |
| `/=` | N/A | `np.true_divide` with `out=` | ❌ Not supported |
| `//=` | N/A | `np.floor_divide` with `out=` | ❌ Not supported |
| `%=` | N/A | `np.remainder` with `out=` | ❌ Not supported |
| `**=` | N/A | `np.power` with `out=` | ❌ Not supported |
| `&=` | N/A | `np.bitwise_and` with `out=` | ❌ Not supported |
| `\|=` | N/A | `np.bitwise_or` with `out=` | ❌ Not supported |
| `^=` | N/A | `np.bitwise_xor` with `out=` | ❌ Not supported |
| `<<=` | N/A | `np.left_shift` with `out=` | ❌ Not supported |
| `>>=` | N/A | `np.right_shift` with `out=` | ❌ Not supported |

**Note**: C# doesn't support overloading compound assignment operators. These would need to be explicit method calls like `arr.AddInPlace(value)`.

---

## 6. Matrix Multiplication

| Python | C# | NumPy ufunc | NumSharp Status |
|--------|-----|-------------|-----------------|
| `@` | N/A | `np.matmul` | ✅ `np.matmul()` function |
| `@=` | N/A | `np.matmul` with `out=` | ❌ Not supported |

**Note**: C# doesn't have a `@` operator. Matrix multiplication is via `np.matmul()` or `np.dot()`.

---

## 7. Implicit vs Explicit Conversions

### NumPy Behavior

```python
# NumPy requires EXPLICIT scalar extraction
scalar_arr = np.array(5)
x = int(scalar_arr)      # ✅ Explicit - works
x = scalar_arr           # ❌ x is still ndarray, not int!

# In typed contexts (e.g., function expecting int)
def foo(x: int): pass
foo(scalar_arr)          # Type error or implicit conversion
```

### NumSharp Alignment

| Direction | NumPy | NumSharp | Rationale |
|-----------|-------|----------|-----------|
| scalar → NDArray | Implicit via `np.asarray()` | `implicit` | Always safe, no data loss |
| NDArray → scalar | Explicit `int(arr)` | `explicit` | May fail, requires 0-dim |

```csharp
// TO-NDArray: implicit (matches NumPy's flexible input handling)
NDArray arr = 5;           // OK - creates scalar array

// FROM-NDArray: explicit (matches NumPy's int(arr) pattern)
int x = arr;               // ❌ Compile error
int x = (int)arr;          // ✅ Explicit cast required
```

---

## 8. Operator Overload Strategy

### NumPy Approach

NumPy operators delegate to ufuncs which handle type conversion internally via `PyArray_FromAny`:

```python
def __add__(self, other):
    return np.add(self, other)  # ufunc calls PyArray_FromAny(other) internally
```

```c
// numpy/_core/src/umath/ufunc_object.c line 643
out_op[i] = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
```

This is equivalent to `np.asanyarray()` - converts scalars to 0-d arrays, passes NDArrays through.

### NumSharp Pattern (Implemented)

We match NumPy's behavior by using `np.asanyarray` at the operator level:

```csharp
// 3 overloads per binary operator (instead of 25+)
public static NDArray operator +(NDArray x, NDArray y) => x.TensorEngine.Add(x, y);
public static NDArray operator +(NDArray left, object right) => left + np.asanyarray(right);
public static NDArray operator +(object left, NDArray right) => np.asanyarray(left) + right;
```

**Why this works:**
1. `np.asanyarray` handles all types: scalars become 0-d arrays, NDArrays pass through, arrays wrap
2. `object` parameter accepts any C# type (boxing is transparent)
3. Matches NumPy's internal `PyArray_FromAny` behavior exactly
4. Implicit `T → NDArray` conversions exist for all 12 supported types

**Implementation status:**
| Operator | Pattern | Status |
|----------|---------|--------|
| `+` | object-based | ✅ Implemented |
| `-` | object-based | ✅ Implemented |
| `*` | object-based | ✅ Implemented |
| `/` | object-based | ✅ Implemented |
| `%` | object-based | ✅ Implemented |

### Legacy Approach (Explicit Type Overloads)

The old pattern used 24 explicit overloads per operator:

```csharp
// 12 types × 2 directions = 24 overloads per binary operator
public static NDArray operator +(NDArray left, int right) => np.add(left, Scalar(right));
public static NDArray operator +(int left, NDArray right) => np.add(Scalar(left), right);
// ... repeated for bool, byte, short, ushort, int, uint, long, ulong, char, float, double, decimal
```

This is being phased out in favor of the object-based pattern above.

---

## 9. Implementation Plan

### Priority 1: Migrate Remaining Arithmetic Operators

Migrate `-`, `*`, `/`, `%` from explicit overloads to object pattern (like `+`):

```csharp
// Pattern for each operator
public static NDArray operator -(NDArray x, NDArray y) => x.TensorEngine.Subtract(x, y);
public static NDArray operator -(NDArray left, object right) => left - np.asanyarray(right);
public static NDArray operator -(object left, NDArray right) => np.asanyarray(left) - right;
```

### Priority 2: Add Missing Bitwise Operators

| Operator | Symbol | Implementation |
|----------|--------|----------------|
| Bitwise NOT | `~` | `operator ~(NDArray) => np.invert(arr)` |
| XOR | `^` | `operator ^(NDArray, NDArray) => np.bitwise_xor()` |
| Left Shift | `<<` | `operator <<(NDArray, int) => np.left_shift()` |
| Right Shift | `>>` | `operator >>(NDArray, int) => np.right_shift()` |

### Priority 3: Power and Floor Division

| Operation | NumPy | NumSharp |
|-----------|-------|----------|
| `arr ** n` | `__pow__` operator | Use `np.power()` - no C# operator |
| `arr // n` | `__floordiv__` operator | Use `np.floor_divide()` - no C# operator |

### Priority 4: Matrix Multiplication

C# has no `@` operator. Use `np.matmul()` function.

---

## 10. Edge Cases and Behaviors

### Empty Arrays

```python
np.array([]) + 1        # array([]) - preserves empty shape
np.array([]) == 1       # array([], dtype=bool)
np.array([]).sum()      # 0.0 - identity element
np.array([]).max()      # ValueError - no identity for max/min
```

### 0-Dimensional (Scalar) Arrays

```python
scalar = np.array(5)
scalar.shape            # ()
scalar.ndim             # 0
scalar.size             # 1
scalar + scalar         # np.int64(10) - returns 0-dim array
scalar.item()           # 5 - extracts Python scalar
```

### Broadcasting

```python
# 3x1 + 3 → 3x3
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
a + b  # [[11, 21, 31], [12, 22, 32], [13, 23, 33]]

# Incompatible shapes raise
np.array([1, 2, 3]) + np.array([[1], [2]])  # ValueError!
```

### NaN Handling

```python
np.nan == np.nan        # False (IEEE 754)
np.array([np.nan]) == np.array([np.nan])  # [False]
np.isnan(np.nan)        # True
```

### Type Promotion

```python
int32 / int32           → float64  (true division always floats)
int32 // int32          → int32    (floor division preserves int)
int32 + float64         → float64  (promotes to wider type)
int32 ** 2              → int32    (integer power preserves int)
int32 ** 0.5            → float64  (fractional power → float)
```

### Division by Zero

```python
int / 0                 → inf (with warning)
int // 0                → 0 (platform-dependent, with warning)
float / 0               → inf
```

---

## 11. NumPy 2.x Breaking Changes (Boolean Arrays)

NumPy 2.x removed support for some boolean array operations:

| Operation | NumPy 1.x | NumPy 2.x | Alternative |
|-----------|-----------|-----------|-------------|
| `bool_arr - bool_arr` | Worked | ❌ TypeError | Use `^` or `logical_xor` |
| `-bool_arr` (unary minus) | Worked | ❌ TypeError | Use `~` or `logical_not` |
| `bool_arr + bool_arr` | int result | bool result | Explicit cast if needed |
| `bool_arr * bool_arr` | int result | bool result | Explicit cast if needed |

**NumSharp should match NumPy 2.x behavior**.

---

## 12. Container Protocol Methods

| Python | C# | NumPy Behavior | NumSharp Status |
|--------|-----|----------------|-----------------|
| `len(arr)` | `.Length` / `.Count` | First dimension size | ✅ via `.shape[0]` |
| `x in arr` | N/A | Linear search | ❌ Not implemented |
| `iter(arr)` | `GetEnumerator()` | Iterates first axis | ✅ Implemented |
| `hash(arr)` | `GetHashCode()` | Raises TypeError | ⚠️ Returns hash (differs!) |

### `__contains__` (the `in` operator)

```python
2 in np.array([1, 2, 3])  # True - linear search
```

NumSharp doesn't implement `in` operator support.

### `__len__` vs `.size`

```python
len(arr)     # First dimension (arr.shape[0])
arr.size     # Total elements (product of all dims)
```

---

## 13. Ufunc Advanced Features

### `out` Parameter

```python
out = np.zeros(3)
np.add([1,2,3], [4,5,6], out=out)  # Writes to existing array
```

NumSharp: ⚠️ Partial support (some functions have `out` parameter)

### `where` Parameter

```python
np.add([1,2,3], 10, where=[True, False, True])  # Conditional operation
```

NumSharp: ❌ Not implemented

### Ufunc Methods

| Method | NumPy | NumSharp |
|--------|-------|----------|
| `ufunc.reduce()` | `np.add.reduce(arr)` = sum | ❌ No ufunc objects |
| `ufunc.accumulate()` | `np.add.accumulate(arr)` = cumsum | ❌ No ufunc objects |
| `ufunc.outer()` | `np.add.outer(a, b)` | ✅ `np.outer()` function |
| `ufunc.at()` | `np.add.at(arr, idx, val)` | ❌ Not implemented |

---

## 14. Logical Functions (not operators)

| NumPy Function | NumSharp Status |
|----------------|-----------------|
| `np.logical_and(a, b)` | ✅ Implemented |
| `np.logical_or(a, b)` | ✅ Implemented |
| `np.logical_xor(a, b)` | ✅ Implemented |
| `np.logical_not(a)` | ✅ Implemented |
| `np.all(a)` | ✅ Implemented |
| `np.any(a)` | ✅ Implemented |

---

## 15. Missing NumPy Functions

### Float-specific Functions

| Function | Description | NumSharp |
|----------|-------------|----------|
| `np.signbit(x)` | True for negative | ❌ Missing |
| `np.copysign(x, y)` | Copy sign of y to x | ❌ Missing |
| `np.nextafter(x, y)` | Next float toward y | ❌ Missing |
| `np.spacing(x)` | Distance to next float | ❌ Missing |
| `np.fabs(x)` | Absolute (float only) | ❌ Missing |
| `np.isposinf(x)` | Test for +inf | ❌ Missing |
| `np.isneginf(x)` | Test for -inf | ❌ Missing |

### Element-wise Min/Max

| Function | Description | NumSharp |
|----------|-------------|----------|
| `np.maximum(a, b)` | Element-wise max | ✅ Implemented |
| `np.minimum(a, b)` | Element-wise min | ✅ Implemented |
| `np.fmax(a, b)` | Max ignoring NaN | ✅ Implemented |
| `np.fmin(a, b)` | Min ignoring NaN | ✅ Implemented |

### Other Missing

| Function | Description | NumSharp |
|----------|-------------|----------|
| `np.divmod(a, b)` | (quotient, remainder) | ❌ Missing |
| `np.modf(x)` | (fractional, integer) | ✅ Implemented |
| `np.ldexp(x, i)` | x * 2^i | ❌ Missing |
| `np.frexp(x)` | (mantissa, exponent) | ❌ Missing |

---

## 16. Views vs Copies

```python
# View (shares memory)
view = arr[1:4]
np.shares_memory(arr, view)  # True

# Copy (independent)
copy = arr[1:4].copy()
np.shares_memory(arr, copy)  # False
```

NumSharp: ✅ Correct view semantics implemented

---

## 17. Test Cases Required

### Basic Operations
- [ ] All 12 dtypes × all operators
- [ ] NDArray × NDArray
- [ ] NDArray × scalar (all 12 types)
- [ ] scalar × NDArray (all 12 types)

### Broadcasting
- [ ] Different shapes that broadcast
- [ ] Incompatible shapes (should throw)

### Edge Cases
- [ ] Empty arrays
- [ ] Scalar arrays (0-dim)
- [ ] NaN values
- [ ] Inf values
- [ ] Division by zero
- [ ] Type promotion rules

### Null/None Handling
- [ ] `arr == null` returns `NDArray<bool>`
- [ ] `arr is null` returns `bool`
- [ ] `null == arr` returns `NDArray<bool>`

### NumPy 2.x Compatibility
- [ ] Boolean subtraction raises
- [ ] Boolean negation raises
- [ ] Boolean arithmetic returns bool dtype

---

## 18. Summary Table

| Category | NumPy Count | NumSharp Implemented | Missing |
|----------|-------------|---------------------|---------|
| Arithmetic Binary | 8 | 5 | `//`, `**`, `@` (C# has no operators) |
| Arithmetic Unary | 3 | 2 | `abs()` (function only) |
| Comparison | 6 | 6 | - |
| Bitwise Binary | 5 | 5 | - |
| Bitwise Unary | 1 | 1 | - |
| Type Conversion | 4 | 12 | `complex`, `__index__` |
| In-place | 12 | 0 | All (C# limitation) |
| Container | 4 | 2 | `__contains__`, proper `__hash__` |
| Ufunc methods | 4 | 1 | `reduce`, `accumulate`, `at` |
| Float functions | 7 | 0 | All |
| **TOTAL** | 54 | 34 | 20 |

### Operator Pattern Migration Status

| Category | Operators | Pattern | Status |
|----------|-----------|---------|--------|
| Arithmetic | `+`, `-`, `*`, `/`, `%` | object (3 each) | ✅ Complete |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` | object (3 each) | ✅ Complete |
| Bitwise | `&`, `\|`, `^` | object (3 each) | ✅ Complete |
| Unary | `-`, `+`, `~`, `!` | NDArray only | ✅ Complete |

**File size reduction:** `NDArray.Primitive.cs` reduced from 159 lines to 42 lines (74% reduction).

**Total savings:** ~150+ explicit overloads replaced by ~40 object-based overloads.

---

## 19. Implementation Priority

### Phase 1: Migrate Arithmetic Operators ✅ COMPLETE
All arithmetic operators now use the object pattern:
- `+`, `-`, `*`, `/`, `%` → 3 overloads each (NDArray×NDArray, NDArray×object, object×NDArray)
- Removed 110 explicit scalar overloads
- Removed Regen template

### Phase 2: Bitwise Operators ✅ COMPLETE
All bitwise operators implemented:
- `&`, `|`, `^` → object pattern (3 overloads each)
- `~` → unary operator
- `<<`, `>>` → via `np.left_shift()` / `np.right_shift()` functions

### Phase 2.5: Remove ValueType ✅ COMPLETE
All public APIs now use `object` instead of `ValueType`:
- `np.power(arr, object)` - scalar or array-like exponent
- `np.floor_divide(arr, object)` - scalar or array-like divisor
- `np.left_shift(arr, object)` - scalar or array-like shift amount
- `np.right_shift(arr, object)` - scalar or array-like shift amount
- `np.full(object, ...)` - scalar fill value
- `np.equal/less/greater/...` - object pattern overloads
- TensorEngine: removed ValueType overloads for Power, FloorDivide, Clip, LeftShift, RightShift
- `NDArray.Scalar(ValueType)` removed (use `Scalar(object)`)

### Phase 3: Float Functions (Medium Priority)
1. `np.signbit`
2. `np.isposinf`, `np.isneginf`
3. `np.copysign`

### Phase 4: Container Protocol (Low Priority)
1. `__contains__` support
2. Proper `__hash__` (should raise like NumPy)

### Phase 5: Ufunc Infrastructure (Future)
1. Ufunc object model
2. `reduce`, `accumulate`, `at` methods
3. `out` and `where` parameters

---

## Appendix: NumPy Source References

- **Number Protocol**: `numpy/_core/src/multiarray/number.c`
- **Input Conversion**: `PyArray_FromAny()` in `numpy/_core/src/multiarray/ctors.c` (line 1520)
- **Ufunc Input Handling**: `numpy/_core/src/umath/ufunc_object.c` (line 643)
- **Rich Compare**: `numpy/_core/src/multiarray/arrayobject.c`
- **Operator Mixin**: `numpy/lib/mixins.py` (lines 60-181)
- **Ufunc Definitions**: `numpy/_core/umath.py`

### Key NumPy Implementation Detail

NumPy operators delegate to ufuncs, which convert inputs via `PyArray_FromAny`:

```c
// ufunc_object.c line 643 - converts any Python object to ndarray
out_op[i] = (PyArrayObject *)PyArray_FromAny(obj, NULL, 0, 0, 0, NULL);
```

This is equivalent to `np.asanyarray()` with flags=0:
- NDArray → returns as-is (no copy)
- Scalar → wraps in 0-d array
- List/tuple → converts to array
- Subclass → preserves subclass

NumSharp's `np.asanyarray` matches this behavior exactly.
