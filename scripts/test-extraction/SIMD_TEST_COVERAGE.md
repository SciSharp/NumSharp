# SIMD Optimization Test Coverage

Tests ported from NumPy 2.4.2 test suite to verify NumSharp correctness.

## Test File

`test/NumSharp.UnitTest/Backends/Kernels/SimdOptimizationTests.cs`

## Source Tests

Test cases derived from:
- `src/numpy/numpy/_core/tests/test_numeric.py` - NonZero tests
- `src/numpy/numpy/_core/tests/test_multiarray.py` - ArgMax/ArgMin, Masking tests
- `src/numpy/numpy/_core/tests/test_indexing.py` - Boolean indexing tests

## Coverage Summary

| Feature | Total Tests | Passing | OpenBugs |
|---------|------------|---------|----------|
| NonZero | 17 | 15 | 2 |
| ArgMax/ArgMin | 37 | 33 | 4 |
| Boolean Masking | 18 | 7 | 11 |
| **Total** | **72** | **55** | **17** |

## NonZero Tests (17 total)

### Passing (15)
- `NonZero_1D_Basic` - Basic 1D array
- `NonZero_2D_Basic` - 2D array returns row/col indices
- `NonZero_AllZeros` - All zeros returns empty
- `NonZero_AllNonzero` - All nonzero returns all indices
- `NonZero_Boolean` - Boolean array support
- `NonZero_Float` - Float array with zeros
- `NonZero_Large_SparseValues` - Large array (SIMD path)
- `NonZero_3D` - 3D array multi-dimensional indices
- `NonZero_NaN_IsNonzero` - NaN treated as nonzero
- `NonZero_EyeMatrix` - Identity matrix diagonal
- `NonZero_UInt16` - UInt16 dtype
- `NonZero_SparsePattern` - Sparse boolean pattern
- `NonZero_FromNumPyTest_Onedim` - NumPy test case
- `NonZero_FromNumPyTest_Twodim` - NumPy test case

### OpenBugs (2)
- `NonZero_Empty` - Empty array throws "size > 0" error
- `NonZero_Int8` - sbyte (int8) not supported by NumSharp

## ArgMax/ArgMin Tests (37 total)

### Passing (33)
- Basic 1D operations
- Ties return first occurrence
- Single element arrays
- Negative values
- Infinity handling
- 2D flattened (no axis)
- 2D with axis=0, axis=1
- Large arrays (SIMD path)
- UInt8, Int16, Int64, Float32 dtypes
- Negative axis
- All same values
- Decreasing order

### OpenBugs (4)
- `ArgMax_NaN_FirstNaNWins` - NumSharp doesn't propagate NaN correctly (returns max value index instead of first NaN)
- `ArgMin_NaN_FirstNaNWins` - Same NaN handling issue
- `ArgMax_Boolean` - Boolean type not supported by ArgMax
- `ArgMin_Boolean` - Boolean type not supported by ArgMin

## Boolean Masking Tests (18 total)

### Passing (7)
- `BooleanMask_Condition` - `a[a > 3]` works correctly
- `BooleanMask_Float` - Float condition `a[a > 2.0]`
- `BooleanMask_Large_SIMDPath` - Large array with condition
- `BooleanMask_ComplexCondition` - `a[(a > 3) & (a < 8)]`
- `BooleanMask_Int32_Condition` - Int32 with condition
- `BooleanMask_Float64_Condition` - Float64 with condition
- `BooleanMask_EvenNumbers` - `a[a % 2 == 0]`
- `BooleanMask_2D_Condition_Flattens` - 2D with condition flattens

### OpenBugs (11)
**Root Cause: Explicit boolean mask arrays don't work**

NumSharp only supports condition-based masking like `a[a > 3]`.
Explicit mask arrays like `a[np.array([True, False, True])]` return all elements instead of filtered elements.

Affected tests:
- `BooleanMask_1D_Basic` - Explicit mask returns all elements
- `BooleanMask_AllTrue` - Returns wrong values
- `BooleanMask_AllFalse` - Should return empty, returns all
- `BooleanMask_EmptyResult_Shape` - Wrong shape
- `BooleanMask_2D_RowSelection` - Row selection fails
- `BooleanMask_2D_Flattens` - 2D mask doesn't flatten
- `BooleanMask_Int16_PreservesDtype` - Returns all elements
- `BooleanMask_FromNumPyTest_Basic` - From NumPy test
- `BooleanMask_FromNumPyTest_2D_RowMask` - From NumPy test
- `BooleanMask_FromNumPyTest_2D_ElementMask` - 2D element mask fails
- `BooleanMask_UInt8` - Returns all elements

## Dtypes Tested

| Dtype | NonZero | ArgMax/ArgMin | Boolean Mask |
|-------|---------|---------------|--------------|
| bool | PASS | FAIL (not supported) | n/a |
| byte (uint8) | PASS | PASS | PASS* |
| sbyte (int8) | FAIL (not supported) | - | - |
| short (int16) | - | PASS | PASS* |
| ushort (uint16) | PASS | - | - |
| int (int32) | PASS | PASS | PASS |
| long (int64) | - | PASS | - |
| float (float32) | - | PASS | - |
| double (float64) | PASS | PASS | PASS |

*Only with condition-based masking, not explicit mask arrays

## Running Tests

```bash
# All SimdOptimizationTests (including OpenBugs)
dotnet test -- --treenode-filter "/*/NumSharp.UnitTest.Backends.Kernels/SimdOptimizationTests/*"

# Exclude OpenBugs (CI-style)
dotnet test -- --treenode-filter "/*/NumSharp.UnitTest.Backends.Kernels/SimdOptimizationTests/*[Category!=OpenBugs]"

# Run ONLY OpenBugs (verify fixes)
dotnet test -- --treenode-filter "/*/NumSharp.UnitTest.Backends.Kernels/SimdOptimizationTests/*[Category=OpenBugs]"
```

## NumPy Python Commands Used

Expected values generated using:

```python
import numpy as np

# NonZero
np.nonzero([0, 1, 0, 3, 0, 5])  # [[1, 3, 5]]
np.nonzero([[0, 1, 0], [3, 0, 5]])  # [[0, 1, 1], [1, 0, 2]]

# ArgMax/ArgMin
np.argmax([3, 1, 4, 1, 5, 9, 2, 6])  # 5
np.argmin([3, 1, 4, 1, 5, 9, 2, 6])  # 1
np.argmax([1.0, np.nan, 3.0])  # 1 (first NaN wins)

# Boolean Masking
a = np.array([1, 2, 3, 4, 5, 6])
a[a > 3]  # [4, 5, 6]
a[np.array([True, False, True, False, True, False])]  # [1, 3, 5]
```
