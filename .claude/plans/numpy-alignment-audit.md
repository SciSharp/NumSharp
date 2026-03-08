# NumPy Alignment Audit Plan

## Objective

Systematically audit NumSharp's NonZero, ArgMax/ArgMin, and Boolean Masking implementations against NumPy 2.4.2 to:
1. Document exact NumPy behavior for all edge cases
2. Identify and catalog misalignments
3. Expand test coverage to 100% NumPy parity
4. Prioritize and fix discrepancies

---

## Methodology

### Phase 1: NumPy Behavior Extraction

For each operation, create a Python script that exhaustively tests:

```python
# Template for behavior extraction
import numpy as np
import json

def test_operation(name, func, inputs):
    """Run operation and capture all outputs."""
    results = []
    for inp in inputs:
        try:
            out = func(inp)
            results.append({
                "input": repr(inp),
                "output": repr(out),
                "dtype": str(out.dtype) if hasattr(out, 'dtype') else type(out).__name__,
                "shape": list(out.shape) if hasattr(out, 'shape') else None,
            })
        except Exception as e:
            results.append({
                "input": repr(inp),
                "error": type(e).__name__,
                "message": str(e)
            })
    return results
```

### Phase 2: NumSharp Verification

Create corresponding `dotnet run` scripts that:
1. Run identical inputs
2. Compare outputs exactly
3. Generate diff reports

### Phase 3: Gap Analysis

For each discrepancy:
- Root cause (missing feature, wrong algorithm, edge case)
- Severity (crash, wrong result, missing dtype)
- Fix complexity (trivial, moderate, architectural)

### Phase 4: Fix Implementation

Prioritized by:
1. Crashes / exceptions
2. Wrong results
3. Missing features
4. Performance differences

---

## Operation 1: NonZero (`np.nonzero`)

### NumPy Reference
- Source: `src/numpy/numpy/_core/tests/test_numeric.py`
- API: `numpy/_core/fromnumeric.py:nonzero()`
- C impl: `numpy/_core/src/multiarray/item_selection.c`

### Test Matrix

| Category | Test Case | NumPy Behavior | NumSharp Status |
|----------|-----------|----------------|-----------------|
| **Empty Arrays** | `np.nonzero([])` | Returns `(array([]),)` | FAIL: throws "size > 0" |
| **0D Arrays** | `np.nonzero(np.array(5))` | Raises ValueError | TODO: verify |
| **All Zeros** | `np.nonzero(zeros(N))` | Returns empty indices | PASS |
| **All NonZero** | `np.nonzero(ones(N))` | Returns all indices | PASS |
| **NaN Values** | `np.nonzero([0, nan, 1])` | NaN is nonzero | PASS |
| **Inf Values** | `np.nonzero([0, inf, -inf])` | Inf is nonzero | TODO |
| **Negative Zero** | `np.nonzero([-0.0, 0.0])` | Both are zero | TODO |
| **Boolean** | `np.nonzero([True, False])` | Works | PASS |
| **2D** | Row/col indices | Correct order | PASS |
| **3D+** | N arrays of indices | Correct | PASS |
| **Non-contiguous** | Sliced arrays | Works | TODO |
| **Broadcast views** | `broadcast_to` result | Works | TODO |

### Dtype Coverage

| Dtype | Supported | Tested | Notes |
|-------|-----------|--------|-------|
| bool | Yes | Yes | |
| int8 (sbyte) | **NO** | Yes | FAIL: not supported |
| uint8 (byte) | Yes | No | TODO |
| int16 | Yes | No | TODO |
| uint16 | Yes | Yes | |
| int32 | Yes | Yes | |
| uint32 | Yes | No | TODO |
| int64 | Yes | No | TODO |
| uint64 | Yes | No | TODO |
| float32 | Yes | No | TODO |
| float64 | Yes | Yes | |
| complex64 | No | - | Not supported by NumSharp |
| complex128 | No | - | Not supported by NumSharp |

### Known Misalignments

1. **Empty array throws exception**
   - NumPy: Returns `(array([], dtype=int64),)` for 1D empty
   - NumSharp: Throws `Debug.Assert(size > 0)`
   - Fix: Remove assertion, handle size=0 case

2. **sbyte (int8) not supported**
   - NumPy: Full int8 support
   - NumSharp: `UnmanagedStorage` doesn't support sbyte
   - Fix: Add sbyte to supported types (larger change)

3. **0D scalar behavior**
   - NumPy: Raises `ValueError: Calling nonzero on 0d arrays is not allowed`
   - NumSharp: TODO - verify current behavior

### Python Extraction Script

```python
#!/usr/bin/env python3
"""Extract np.nonzero behavior for all edge cases."""
import numpy as np

print("=== NonZero Edge Cases ===\n")

# Empty arrays
print("# Empty Arrays")
for dtype in [np.int32, np.float64, np.bool_]:
    a = np.array([], dtype=dtype)
    r = np.nonzero(a)
    print(f"nonzero(empty {dtype.__name__}): {[x.tolist() for x in r]}, shapes={[x.shape for x in r]}")

# 0D scalar
print("\n# 0D Scalar")
try:
    np.nonzero(np.array(5))
except ValueError as e:
    print(f"nonzero(scalar): ValueError: {e}")

# Special float values
print("\n# Special Float Values")
for val in [np.nan, np.inf, -np.inf, -0.0, 0.0]:
    a = np.array([0.0, val, 1.0])
    r = np.nonzero(a)
    print(f"nonzero([0, {val}, 1]): {r[0].tolist()}")

# Non-contiguous
print("\n# Non-contiguous Arrays")
a = np.arange(10)[::2]  # [0, 2, 4, 6, 8]
print(f"nonzero(arange(10)[::2]): {np.nonzero(a)[0].tolist()}")
print(f"  contiguous: {a.flags['C_CONTIGUOUS']}")

# Transposed
a = np.arange(6).reshape(2, 3).T
print(f"nonzero(transposed): row={np.nonzero(a)[0].tolist()}, col={np.nonzero(a)[1].tolist()}")
print(f"  contiguous: {a.flags['C_CONTIGUOUS']}")
```

---

## Operation 2: ArgMax/ArgMin (`np.argmax`, `np.argmin`)

### NumPy Reference
- Source: `src/numpy/numpy/_core/tests/test_multiarray.py`
- API: `numpy/_core/fromnumeric.py:argmax()`, `argmin()`
- C impl: `numpy/_core/src/multiarray/calculation.c`

### Test Matrix

| Category | Test Case | NumPy Behavior | NumSharp Status |
|----------|-----------|----------------|-----------------|
| **Basic 1D** | Simple array | First occurrence | PASS |
| **Ties** | Multiple max/min | First occurrence | PASS |
| **NaN** | Array with NaN | Returns first NaN index | **FAIL** |
| **Inf** | +Inf, -Inf | Correct | PASS |
| **Empty** | `argmax([])` | Raises ValueError | TODO |
| **Boolean** | `[True, False]` | Raises NotSupportedError | **FAIL** (should work) |
| **2D no axis** | Flattened index | Correct | PASS |
| **2D axis=0** | Per-column | Correct | PASS |
| **2D axis=1** | Per-row | Correct | PASS |
| **Negative axis** | axis=-1 | Same as last axis | PASS |
| **keepdims** | Shape preserved | Not implemented | TODO |
| **out parameter** | Output array | Not implemented | TODO |

### NaN Handling Deep Dive

NumPy's NaN behavior is specific:
```python
>>> np.argmax([1.0, np.nan, 3.0])
1  # First NaN wins, regardless of other values

>>> np.argmin([1.0, np.nan, 3.0])
1  # Same - first NaN wins

>>> np.argmax([np.nan, np.nan, 3.0])
0  # First NaN
```

**Root Cause in NumSharp:**
The comparison `val > max` returns `False` when either operand is NaN.
NumPy uses a different comparison that treats NaN as "maximum".

**Fix Required:**
```csharp
// Current (wrong):
if (val > max) { max = val; maxAt = idx; }

// Should be:
if (val > max || double.IsNaN(val) && !double.IsNaN(max))
{ max = val; maxAt = idx; }
```

### Dtype Coverage

| Dtype | argmax | argmin | Notes |
|-------|--------|--------|-------|
| bool | FAIL | FAIL | Should work (True=1, False=0) |
| byte | PASS | PASS | |
| int16 | PASS | PASS | |
| int32 | PASS | PASS | |
| int64 | PASS | PASS | |
| float32 | PASS* | PASS* | *NaN handling wrong |
| float64 | PASS* | PASS* | *NaN handling wrong |

### Known Misalignments

1. **NaN propagation incorrect**
   - NumPy: First NaN always wins
   - NumSharp: Ignores NaN, returns index of actual max/min
   - Impact: Silent wrong results for NaN-containing data
   - Fix: Special NaN handling in comparison loop

2. **Boolean type not supported**
   - NumPy: Treats as 0/1, returns index
   - NumSharp: Throws `NotSupportedException`
   - Fix: Add Boolean case to type switch

3. **Empty array behavior**
   - NumPy: Raises `ValueError: attempt to get argmax of an empty sequence`
   - NumSharp: TODO - verify behavior
   - Fix: Add empty check with appropriate exception

### Python Extraction Script

```python
#!/usr/bin/env python3
"""Extract np.argmax/argmin behavior for all edge cases."""
import numpy as np

print("=== ArgMax/ArgMin Edge Cases ===\n")

# NaN behavior (critical)
print("# NaN Behavior")
cases = [
    [1.0, np.nan, 3.0],
    [np.nan, 1.0, 3.0],
    [1.0, 3.0, np.nan],
    [np.nan, np.nan, 1.0],
]
for c in cases:
    print(f"argmax({c}): {np.argmax(c)}, argmin: {np.argmin(c)}")

# Empty array
print("\n# Empty Array")
try:
    np.argmax([])
except ValueError as e:
    print(f"argmax([]): ValueError: {e}")

# Boolean
print("\n# Boolean")
a = np.array([False, True, False, True])
print(f"argmax(bool): {np.argmax(a)}, argmin: {np.argmin(a)}")

# 2D with axis
print("\n# 2D with Axis")
a = np.array([[1, 5, 3], [4, 2, 6]])
print(f"argmax(axis=0): {np.argmax(a, axis=0).tolist()}")
print(f"argmax(axis=1): {np.argmax(a, axis=1).tolist()}")
print(f"argmax(axis=-1): {np.argmax(a, axis=-1).tolist()}")
print(f"argmax(axis=-2): {np.argmax(a, axis=-2).tolist()}")

# keepdims
print("\n# keepdims")
print(f"argmax(axis=0, keepdims=True): shape={np.argmax(a, axis=0, keepdims=True).shape}")
print(f"argmax(axis=1, keepdims=True): shape={np.argmax(a, axis=1, keepdims=True).shape}")
```

---

## Operation 3: Boolean Masking (`a[mask]`)

### NumPy Reference
- Source: `src/numpy/numpy/_core/tests/test_indexing.py`
- API: `numpy/_core/src/multiarray/mapping.c`

### Test Matrix

| Category | Test Case | NumPy Behavior | NumSharp Status |
|----------|-----------|----------------|-----------------|
| **1D explicit mask** | `a[[T,F,T]]` | Filtered array | **FAIL** |
| **1D condition** | `a[a > 3]` | Filtered array | PASS |
| **All True** | Full selection | All elements | **FAIL** |
| **All False** | Empty selection | Empty array, shape (0,) | **FAIL** |
| **2D row mask** | 1D mask on 2D | Row selection | **FAIL** |
| **2D element mask** | 2D mask on 2D | Flattened | **FAIL** |
| **Mask shape mismatch** | Wrong size | Raises IndexError | TODO |
| **Non-bool mask** | Int array | Fancy indexing | Different operation |
| **Dtype preservation** | Result dtype | Same as input | PASS (when working) |

### Critical Finding: Two Indexing Paths

NumSharp has two different code paths:

1. **Condition-based** (`a[a > 3]`): Creates mask internally via comparison
   - Goes through `NDArray.Indexing.cs` general indexing
   - **Works correctly**

2. **Explicit mask** (`a[np.array([True, False, True])]`): Uses `NDArray<bool>` indexer
   - Goes through `NDArray.Indexing.Masking.cs`
   - **Broken**: Returns all elements instead of filtering

### Root Cause Analysis

Looking at `NDArray.Indexing.Masking.cs`:

```csharp
public unsafe NDArray this[NDArray<bool> mask]
{
    get
    {
        // SIMD fast path
        if (ILKernelGenerator.Enabled && ...)
        {
            return BooleanMaskFastPath(mask);  // <-- This path is broken
        }

        // Fallback
        return FetchIndices(this, np.nonzero(mask), null, true);
    }
}
```

The SIMD fast path `BooleanMaskFastPath` appears to have bugs in the helpers.

### Test Categories

| Test Type | Description | Priority |
|-----------|-------------|----------|
| Shape tests | Empty result, 0D, multi-D | High |
| Dtype tests | All 12 types | Medium |
| Memory tests | Contiguous vs sliced | High |
| Edge cases | All True, All False | High |
| 2D+ tests | Row/element selection | High |

### Python Extraction Script

```python
#!/usr/bin/env python3
"""Extract boolean masking behavior for all edge cases."""
import numpy as np

print("=== Boolean Masking Edge Cases ===\n")

# Explicit mask vs condition
print("# Explicit Mask vs Condition")
a = np.array([1, 2, 3, 4, 5, 6])
mask = np.array([True, False, True, False, True, False])
print(f"a[explicit_mask]: {a[mask].tolist()}")
print(f"a[a % 2 == 1]: {a[a % 2 == 1].tolist()}")  # Should be same

# All True / All False
print("\n# All True / All False")
a = np.array([1, 2, 3])
print(f"a[[T,T,T]]: {a[np.array([True, True, True])].tolist()}")
print(f"a[[F,F,F]]: {a[np.array([False, False, False])].tolist()}")
print(f"a[[F,F,F]].shape: {a[np.array([False, False, False])].shape}")

# 2D row selection
print("\n# 2D Row Selection")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = np.array([True, False, True])
print(f"a[row_mask]:\n{a[mask]}")
print(f"shape: {a[mask].shape}")

# 2D element mask (flattens)
print("\n# 2D Element Mask (Flattens)")
a = np.array([[1, 2], [3, 4]])
mask = np.array([[True, False], [False, True]])
print(f"a[2D_mask]: {a[mask].tolist()}")
print(f"shape: {a[mask].shape}")

# Shape mismatch
print("\n# Shape Mismatch")
a = np.array([1, 2, 3, 4, 5])
mask = np.array([True, False])  # Wrong size
try:
    result = a[mask]
except IndexError as e:
    print(f"IndexError: {e}")

# Dtype preservation
print("\n# Dtype Preservation")
for dtype in [np.int16, np.float32, np.float64]:
    a = np.array([1, 2, 3], dtype=dtype)
    mask = np.array([True, False, True])
    result = a[mask]
    print(f"{dtype.__name__}: result.dtype = {result.dtype}")
```

---

## Prioritized Fix Roadmap

### Priority 1: Crashes & Exceptions (Immediate)

| Issue | Operation | Severity | Effort |
|-------|-----------|----------|--------|
| Empty array throws | NonZero | Crash | Low |
| Boolean mask returns all elements | Masking | Wrong results | Medium |

### Priority 2: Wrong Results (High)

| Issue | Operation | Severity | Effort |
|-------|-----------|----------|--------|
| NaN handling in argmax/argmin | ArgMax/Min | Silent wrong | Medium |
| 2D row mask selection | Masking | Wrong results | Medium |
| 2D element mask flattening | Masking | Wrong results | Medium |

### Priority 3: Missing Features (Medium)

| Issue | Operation | Impact | Effort |
|-------|-----------|--------|--------|
| Boolean dtype for argmax/argmin | ArgMax/Min | Feature gap | Low |
| sbyte (int8) support | NonZero | Feature gap | High |
| keepdims parameter | ArgMax/Min | Feature gap | Medium |
| out parameter | ArgMax/Min | Feature gap | Medium |

### Priority 4: Test Coverage (Ongoing)

| Area | Current | Target |
|------|---------|--------|
| NonZero dtypes | 6/12 | 12/12 |
| ArgMax dtypes | 8/12 | 12/12 |
| Boolean masking paths | 1/2 | 2/2 |
| Edge cases | ~50% | 100% |

---

## Execution Plan

### Week 1: Deep Analysis

1. Run all Python extraction scripts
2. Capture exact NumPy outputs to JSON
3. Create comprehensive dotnet run verification scripts
4. Generate full diff report

### Week 2: Fix Critical Issues

1. Fix NonZero empty array handling
2. Fix Boolean masking SIMD path
3. Add Boolean dtype to ArgMax/ArgMin

### Week 3: Fix Wrong Results

1. Implement NaN handling for ArgMax/ArgMin
2. Fix 2D boolean masking scenarios
3. Add missing edge case handling

### Week 4: Expand Coverage

1. Add all remaining dtype tests
2. Add non-contiguous array tests
3. Add broadcast view tests
4. Document all Misaligned behaviors

---

## Test File Organization

```
test/NumSharp.UnitTest/
├── Backends/Kernels/
│   ├── SimdOptimizationTests.cs       # Current: 72 tests
│   ├── NonZeroEdgeCaseTests.cs        # NEW: Expanded nonzero
│   ├── ArgMaxMinEdgeCaseTests.cs      # NEW: Expanded argmax/argmin
│   └── BooleanMaskingTests.cs         # NEW: Comprehensive masking
├── Indexing/
│   └── np_nonzero_tests.cs            # Existing (2 basic tests)
└── Selection/
    └── NDArray.Indexing.Test.cs       # Existing (mixed)
```

---

## Success Criteria

1. **Zero crashes** on valid NumPy inputs
2. **100% result parity** for supported dtypes
3. **Clear documentation** for unsupported features
4. **Regression tests** for all fixed bugs
5. **OpenBugs removed** as fixes land
