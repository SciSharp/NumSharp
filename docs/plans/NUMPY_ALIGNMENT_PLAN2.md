# NumSharp-NumPy Alignment Plan (Revised)

**Document Version**: 2.0
**Created**: 2024
**Revised**: 2026-04-08
**Target**: NumPy 2.x API parity
**Scope**: All operator, function, and protocol gaps

---

## Executive Summary

NumSharp currently implements **34 of 54** NumPy operator/protocol features (63%). This revised plan accounts for NumSharp's actual architecture: **ILKernelGenerator** for optimized operations rather than NumPy's generalized ufunc system.

### Architecture Reality Check

NumSharp does NOT have a generalized ufunc infrastructure. Instead:

| NumPy | NumSharp |
|-------|----------|
| `numpy.ufunc` objects with `.reduce()`, `.accumulate()`, `.at()`, `.outer()` | Individual `np.*` static methods |
| `out` parameter on all ufuncs | `out` parameter on select functions only (`clip`, `minimum`, `maximum`) |
| `where` parameter on all ufuncs | Not implemented |
| Runtime-generated ufuncs | `ILKernelGenerator` - compile-time IL emission with SIMD |

**Key Insight**: Adding `out`/`where` parameters requires modifying each operation individually through `TensorEngine` → `DefaultEngine` → `ILKernelGenerator`, not a single ufunc infrastructure change.

### Revised Gap Breakdown

| Category | Missing | Effort | Feasibility |
|----------|---------|--------|-------------|
| In-place operators | 12 | N/A | C# limitation |
| Float functions | 7 | Low | Straightforward |
| Math functions | 3 | Low | Straightforward |
| `out` parameter | ~40 functions | Medium | Per-function change |
| `where` parameter | ~40 functions | High | Requires kernel changes |
| Container protocol | 2 | Low | Straightforward |
| Type conversions | 2 | High | Complex number support |

---

## NumSharp Architecture Overview

### Operation Flow

```
np.add(x1, x2)
    -> TensorEngine.Add(lhs, rhs)           [abstract method]
        -> DefaultEngine.Add(lhs, rhs)      [concrete implementation]
            -> ExecuteBinaryOp(lhs, rhs, BinaryOp.Add)
                -> ILKernelGenerator.GetMixedTypeKernel(key)
                    -> IL-generated SIMD kernel (cached)
```

### ILKernelGenerator Structure

27 partial class files generating optimized IL at runtime:

| Category | Files | Operations |
|----------|-------|------------|
| Core | `ILKernelGenerator.cs`, `.Scalar.cs` | Type mapping, SIMD detection |
| Binary | `.Binary.cs`, `.MixedType.cs`, `.Shift.cs` | Add, Sub, Mul, Div, Power, FloorDivide, Bitwise |
| Unary | `.Unary.cs`, `.Unary.Math.cs`, `.Unary.Decimal.cs`, `.Unary.Vector.cs`, `.Unary.Predicate.cs` | Negate, Abs, Sqrt, Trig, Exp, Log, Sign, Floor, Ceil |
| Comparison | `.Comparison.cs` | ==, !=, <, >, <=, >= |
| Reduction | `.Reduction.cs`, `.Reduction.Arg.cs`, `.Reduction.Boolean.cs`, `.Reduction.Axis.cs` | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any |
| Scan | `.Scan.cs` | CumSum, CumProd |

### Supported Operations (via BinaryOp/UnaryOp enums)

**BinaryOp**: Add, Subtract, Multiply, Divide, Mod, BitwiseAnd, BitwiseOr, BitwiseXor, Power, FloorDivide, LeftShift, RightShift, ATan2

**UnaryOp**: Negate, Abs, Sqrt, Exp, Log, Sin, Cos, Tan, Exp2, Expm1, Log2, Log10, Log1p, Sinh, Cosh, Tanh, ASin, ACos, ATan, Sign, Ceil, Floor, Round, Truncate, Square, Reciprocal, Cbrt, Deg2Rad, Rad2Deg, Invert, IsNaN, IsInf, IsFinite

---

## Phase 0: In-Place Operators (NOT IMPLEMENTABLE)

**Status**: DEFERRED PERMANENTLY

C# does not allow overloading compound assignment operators (`+=`, `-=`, etc.). The compiler expands `arr += 5` to `arr = arr + 5`, creating a new array.

### Workaround: Explicit In-Place Methods

```csharp
// File: src/NumSharp.Core/Backends/NDArray.InPlace.cs
public partial class NDArray
{
    public void AddInPlace(NDArray other) { /* modify this in-place */ }
    public void SubtractInPlace(NDArray other) { /* modify this in-place */ }
    // etc.
}
```

**Implementation**: These would call `np.copyto(this, np.add(this, other))` or use direct memory copy with kernel execution.

**Effort**: 1-2 days
**Priority**: Low (workaround exists)

---

## Phase 1: Float-Specific Functions

**Estimated Effort**: 2-3 days
**Priority**: Low (rarely used, but easy wins)
**Dependencies**: None

All implementations are pure C# using `System.Math` or bit manipulation - no kernel changes needed.

### 1.1 `np.signbit(x)`

```csharp
// File: src/NumSharp.Core/Logic/np.signbit.cs
public static NDArray<bool> signbit(NDArray x)
{
    // Use BitConverter to check sign bit
    // float: bit 31, double: bit 63
    // Handles -0.0 correctly (returns True)
}
```

### 1.2 `np.isposinf(x)` and `np.isneginf(x)`

```csharp
// File: src/NumSharp.Core/Logic/np.isinf_extended.cs
public static NDArray<bool> isposinf(NDArray x)
    => np.logical_and(np.isinf(x), np.greater(x, 0));

public static NDArray<bool> isneginf(NDArray x)
    => np.logical_and(np.isinf(x), np.less(x, 0));
```

### 1.3 `np.copysign(x, y)`

```csharp
// File: src/NumSharp.Core/Math/np.copysign.cs
public static NDArray copysign(NDArray x, NDArray y)
{
    // Use Math.CopySign (available in .NET 8+)
    // Broadcast x and y, apply element-wise
}
```

### 1.4 `np.fabs(x)`

```csharp
// File: src/NumSharp.Core/Math/np.fabs.cs
public static NDArray fabs(NDArray x)
{
    if (x.GetTypeCode != NPTypeCode.Single && x.GetTypeCode != NPTypeCode.Double)
        throw new TypeError("fabs requires float dtype");
    return np.abs(x);
}
```

### 1.5 `np.nextafter(x, y)`

```csharp
// File: src/NumSharp.Core/Math/np.nextafter.cs
public static NDArray nextafter(NDArray x, NDArray y)
{
    // Use Math.BitIncrement/BitDecrement (.NET 8+)
    // Or manual bit manipulation for cross-platform
}
```

### 1.6 `np.spacing(x)`

```csharp
// File: src/NumSharp.Core/Math/np.spacing.cs
public static NDArray spacing(NDArray x)
    => np.subtract(np.nextafter(x, np.inf), x);
```

### Phase 1 Deliverables

| File | Functions | Effort |
|------|-----------|--------|
| `np.signbit.cs` | `signbit` | 2h |
| `np.isinf_extended.cs` | `isposinf`, `isneginf` | 1h |
| `np.copysign.cs` | `copysign` | 2h |
| `np.fabs.cs` | `fabs` | 30m |
| `np.nextafter.cs` | `nextafter` | 2h |
| `np.spacing.cs` | `spacing` | 30m |

---

## Phase 2: Math Functions

**Estimated Effort**: 1-2 days
**Priority**: Low
**Dependencies**: None

### 2.1 `np.divmod(x, y)`

```csharp
// File: src/NumSharp.Core/Math/np.divmod.cs
public static (NDArray quotient, NDArray remainder) divmod(NDArray x, NDArray y)
    => (np.floor_divide(x, y), np.mod(x, y));
```

### 2.2 `np.ldexp(x, i)`

```csharp
// File: src/NumSharp.Core/Math/np.ldexp.cs
public static NDArray ldexp(NDArray x, NDArray i)
{
    // Use Math.ScaleB (.NET 8+)
    // ldexp(x, i) = x * 2^i
}
```

### 2.3 `np.frexp(x)`

```csharp
// File: src/NumSharp.Core/Math/np.frexp.cs
public static (NDArray mantissa, NDArray exponent) frexp(NDArray x)
{
    // Extract mantissa and exponent from IEEE 754 representation
    // mantissa in [0.5, 1.0), x = mantissa * 2^exponent
}
```

### Phase 2 Deliverables

| File | Functions | Effort |
|------|-----------|--------|
| `np.divmod.cs` | `divmod` | 30m |
| `np.ldexp.cs` | `ldexp` | 2h |
| `np.frexp.cs` | `frexp` | 2h |

---

## Phase 3: Container Protocol

**Estimated Effort**: 1 day
**Priority**: Medium
**Dependencies**: None

### 3.1 `NDArray.Contains(object value)`

```csharp
// File: src/NumSharp.Core/Backends/NDArray.Contains.cs
public partial class NDArray
{
    /// <summary>
    /// Returns true if value is found in the array.
    /// Equivalent to NumPy's `value in arr`.
    /// </summary>
    public bool Contains(object value)
    {
        var scalar = np.asanyarray(value);
        return np.any(this == scalar);
    }
}
```

**Usage**: `if (arr.Contains(2)) { ... }` instead of NumPy's `2 in arr`

### 3.2 `GetHashCode()` - Make Unhashable

NumPy arrays are unhashable because they're mutable. NumSharp should match.

```csharp
// File: src/NumSharp.Core/Backends/NDArray.cs
public override int GetHashCode()
{
    throw new NotSupportedException(
        "NDArray is unhashable because it is mutable. " +
        "Use arr.tobytes().GetHashCode() for a hashable representation.");
}
```

**Breaking Change**: Code using NDArray as dictionary keys will break.

**Migration**:
```csharp
// Before (broken):
var dict = new Dictionary<NDArray, int>();

// After:
var dict = new Dictionary<NDArray, int>(ReferenceEqualityComparer.Instance);
// Or use arr.tobytes() as key
```

---

## Phase 4: `out` Parameter Support

**Estimated Effort**: 1-2 weeks
**Priority**: Medium
**Dependencies**: None

### Current State

Only these functions have `out` parameter:
- `np.clip(a, min, max, out)`
- `np.minimum(x1, x2, out)` / `np.maximum(x1, x2, out)`
- `np.fmin(x1, x2, out)` / `np.fmax(x1, x2, out)`

### Implementation Strategy

Adding `out` requires changes at 3 layers:

1. **np.* API** - Add overload with `NDArray @out = null`
2. **TensorEngine** - Add `out` parameter to abstract method
3. **DefaultEngine** - Pass `out` to kernel execution

### Example: Adding `out` to `np.add`

```csharp
// 1. API Layer: src/NumSharp.Core/Math/np.math.cs
public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null)
    => x1.TensorEngine.Add(x1, x2, @out);

// 2. TensorEngine: src/NumSharp.Core/Backends/TensorEngine.cs
public abstract NDArray Add(NDArray lhs, NDArray rhs, NDArray @out = null);

// 3. DefaultEngine: src/NumSharp.Core/Backends/Default/Math/Default.Add.cs
public override NDArray Add(NDArray lhs, NDArray rhs, NDArray @out = null)
{
    return ExecuteBinaryOp(lhs, rhs, BinaryOp.Add, @out);
}

// 4. ExecuteBinaryOp modification
internal NDArray ExecuteBinaryOp(NDArray lhs, NDArray rhs, BinaryOp op, NDArray @out = null)
{
    // ... type promotion, broadcasting ...

    // Allocate result OR use provided out array
    var result = @out ?? new NDArray(resultType, resultShape, false);

    // Validate out array if provided
    if (@out != null)
    {
        if (!@out.Shape.Equals(resultShape))
            throw new ArgumentException("out array has wrong shape");
        if (@out.GetTypeCode != resultType)
            throw new ArgumentException("out array has wrong dtype");
    }

    // ... kernel execution writes to result ...
    return result;
}
```

### Functions to Add `out` Parameter

**Binary Operations** (highest priority):
- `np.add`, `np.subtract`, `np.multiply`, `np.divide`
- `np.mod`, `np.power`, `np.floor_divide`
- `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`
- `np.left_shift`, `np.right_shift`

**Unary Operations**:
- `np.negative`, `np.abs`, `np.sqrt`, `np.square`
- `np.exp`, `np.log`, `np.log2`, `np.log10`
- `np.sin`, `np.cos`, `np.tan`
- `np.floor`, `np.ceil`, `np.trunc`, `np.round`

**Comparison Operations**:
- `np.equal`, `np.not_equal`
- `np.less`, `np.greater`, `np.less_equal`, `np.greater_equal`

### Phase 4 Deliverables

| Category | Functions | Files to Modify | Effort |
|----------|-----------|-----------------|--------|
| Binary arithmetic | 7 | `np.math.cs`, `TensorEngine.cs`, `Default.*.cs` | 2 days |
| Binary bitwise | 5 | Same | 1 day |
| Unary math | 15+ | Same | 3 days |
| Comparisons | 6 | Same | 1 day |

---

## Phase 5: `*_at` Methods (Indexed In-Place Operations)

**Estimated Effort**: 3-5 days
**Priority**: Medium
**Dependencies**: None

NumPy's `ufunc.at()` performs unbuffered in-place operations at specified indices, correctly handling repeated indices.

```python
# NumPy
arr = np.zeros(3)
np.add.at(arr, [0, 0, 0], 1)  # arr = [3, 0, 0] - added 1 three times
```

### Implementation

Since NumSharp doesn't have ufunc objects, implement as static methods:

```csharp
// File: src/NumSharp.Core/Math/np.ufunc_at.cs
public static partial class np
{
    /// <summary>
    /// Performs unbuffered in-place operation at specified indices.
    /// Unlike fancy indexing, correctly handles repeated indices.
    /// </summary>
    public static void add_at(NDArray a, NDArray indices, NDArray b)
    {
        // For each index in indices:
        //   a[index] = a[index] + b[corresponding]
        // Key: don't buffer - apply each operation immediately
    }

    public static void subtract_at(NDArray a, NDArray indices, NDArray b) { ... }
    public static void multiply_at(NDArray a, NDArray indices, NDArray b) { ... }
    public static void divide_at(NDArray a, NDArray indices, NDArray b) { ... }
    // etc.
}
```

### Phase 5 Deliverables

| File | Functions | Effort |
|------|-----------|--------|
| `np.ufunc_at.cs` | `add_at`, `subtract_at`, `multiply_at`, `divide_at`, `minimum_at`, `maximum_at` | 3-5 days |

---

## Phase 6: NumPy 2.x Compatibility

**Estimated Effort**: 1 day
**Priority**: High (correctness)
**Dependencies**: None

### 6.1 Boolean Array Restrictions

NumPy 2.x removed support for:

| Operation | NumPy 1.x | NumPy 2.x | NumSharp Should |
|-----------|-----------|-----------|-----------------|
| `bool - bool` | Worked | TypeError | Throw |
| `-bool_array` | Worked | DeprecationWarning | Throw |

```csharp
// File: src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs
// In ExecuteBinaryOp:
if (op == BinaryOp.Subtract && lhsType == NPTypeCode.Boolean && rhsType == NPTypeCode.Boolean)
    throw new TypeError("numpy boolean subtract is not supported");

// File: src/NumSharp.Core/Operations/Elementwise/NDArray.Primitive.cs
public static NDArray operator -(NDArray x)
{
    if (x.GetTypeCode == NPTypeCode.Boolean)
        throw new TypeError(
            "The numpy boolean negative, the `-` operator, is not supported, " +
            "use the `~` operator or the logical_not function instead.");
    return x.TensorEngine.Negate(x);
}
```

---

## Phase 7: `where` Parameter (DEFERRED)

**Estimated Effort**: 2-3 weeks
**Priority**: Low
**Dependencies**: Phase 4 (`out` parameter)

The `where` parameter applies operations conditionally:

```python
np.add([1,2,3], 10, where=[True, False, True])
# Result: [11, 0, 13] - only applies where mask is True
```

### Why Deferred

This requires significant kernel changes:
1. All IL-generated kernels need conditional execution path
2. Need to handle `out` array initialization for masked elements
3. Complex interaction with SIMD (mask operations)

### Alternative

Users can achieve same result with:
```csharp
var result = arr.copy();
var mask = np.array(new[] { true, false, true });
result[mask] = np.add(arr, 10)[mask];
```

---

## Phase 8: Complex Number Support (DEFERRED)

**Estimated Effort**: 2-4 weeks
**Priority**: Low
**Dependencies**: None

### Scope

1. Add `NPTypeCode.Complex128` (System.Numerics.Complex)
2. Update `UnmanagedStorage` for 16-byte complex type
3. Add complex arithmetic to all binary operations
4. Add `.real`, `.imag` properties to NDArray
5. Add `np.conjugate()`, `np.angle()`

### Why Deferred

- Substantial change affecting type system, storage, all operations
- Low user demand (most NumSharp users don't need complex numbers)
- Can be added incrementally later

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
| 3-4 | Boolean operator restrictions |
| 5 | Tests + documentation |

### Week 3-4: `out` Parameter (Phase 4)

| Day | Tasks |
|-----|-------|
| 1-2 | Binary arithmetic ops (`add`, `subtract`, `multiply`, `divide`) |
| 3-4 | Binary ops (`mod`, `power`, `floor_divide`, bitwise) |
| 5-7 | Unary ops (most common: `abs`, `sqrt`, `exp`, `log`) |
| 8-10 | Remaining unary + comparison ops |

### Week 5: `*_at` Methods (Phase 5)

| Day | Tasks |
|-----|-------|
| 1-2 | `add_at`, `subtract_at` |
| 3-4 | `multiply_at`, `divide_at`, `minimum_at`, `maximum_at` |
| 5 | Tests |

---

## Success Criteria

### Phase Completion Checklist

- [ ] **Phase 1**: All 7 float functions implemented and tested
- [ ] **Phase 2**: All 3 math functions implemented and tested
- [ ] **Phase 3**: `Contains()` works, `GetHashCode()` throws
- [ ] **Phase 4**: `out` parameter on binary arithmetic ops
- [ ] **Phase 5**: `add_at` and `multiply_at` work correctly
- [ ] **Phase 6**: NumPy 2.x boolean restrictions enforced

### Deferred (Future Work)

- [ ] **Phase 7**: `where` parameter (complex kernel changes)
- [ ] **Phase 8**: Complex number support (System.Numerics.Complex)
- [ ] In-place operator methods (`AddInPlace`, etc.)

### Final Gap Count

| Before | After Phase 6 | After All |
|--------|---------------|-----------|
| 20 missing | ~10 missing | ~5 missing |

*Remaining = in-place operators (C# limitation) + where parameter + complex numbers*

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
    ├── NDArray.Contains.cs
    └── NDArray.InPlace.cs (optional)
```

### Files to Modify

```
src/NumSharp.Core/
├── Backends/
│   ├── NDArray.cs                           # GetHashCode override
│   ├── TensorEngine.cs                      # Add out parameter to methods
│   └── Default/Math/
│       ├── DefaultEngine.BinaryOp.cs        # out parameter, bool restrictions
│       ├── Default.Add.cs                   # out parameter
│       ├── Default.Subtract.cs              # out parameter
│       └── [all other Default.*.cs]         # out parameter
├── Operations/Elementwise/
│   └── NDArray.Primitive.cs                 # Boolean negate restriction
└── Math/
    └── np.math.cs                           # out parameter overloads
```

---

## Appendix: Architecture Diagram

```
User Code
    |
    v
np.add(x1, x2, out=result)          <-- Static API (np.*.cs)
    |
    v
TensorEngine.Add(lhs, rhs, out)     <-- Abstract interface
    |
    v
DefaultEngine.Add(lhs, rhs, out)    <-- Concrete implementation
    |
    v
ExecuteBinaryOp(lhs, rhs, Add, out) <-- Dispatch logic
    |
    +---> Type promotion (np._FindCommonType)
    +---> Shape broadcasting (Broadcast)
    +---> Path classification (SimdFull/SimdScalar/General)
    |
    v
ILKernelGenerator.GetMixedTypeKernel(key)   <-- IL emission
    |
    +---> Cache lookup (ConcurrentDictionary)
    +---> Generate DynamicMethod if not cached
    +---> Emit SIMD loops (Vector256/Vector128)
    |
    v
kernel(lhs*, rhs*, result*, strides, shape, ndim, size)
    |
    v
Result written to 'out' or newly allocated NDArray
```
