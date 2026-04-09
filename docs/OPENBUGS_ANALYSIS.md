# OpenBugs Test Analysis

Generated: 2026-04-09

## Summary
- **Total OpenBugs tests:** 294 (129 unique test methods)
- **Passing:** 215 runs (81 unique methods) - CAN REMOVE [OpenBugs]
- **Failing:** 79 runs (79 unique methods) - Still broken

---

## Tests That Can Have [OpenBugs] Removed (81 tests)

These tests are now passing and the [OpenBugs] attribute should be removed:

```
Abs_Int32
Add_SlicedWithScalar
Allocate_2GB
Allocate_4GB
ArgMax_Boolean
ArgMin_Boolean
Assignment_BooleanMask
Assignment_FancyIndex
ATan2_ReversedArrays
ATan2_SlicedArrays
ATan2_TransposedArrays
BooleanIndex_RowSelection
BooleanIndex_Simple
BooleanMask_1D_Basic
BooleanMask_2D_Flattens
BooleanMask_2D_RowSelection
BooleanMask_AllFalse
BooleanMask_AllTrue
BooleanMask_EmptyResult_Shape
BooleanMask_FromNumPyTest_2D_ElementMask
BooleanMask_FromNumPyTest_2D_RowMask
BooleanMask_FromNumPyTest_Basic
BooleanMask_Int16_PreservesDtype
BooleanMask_UInt8
BoolTwo1D_NDArrayAND
BoolTwo1D_NDArrayOR
BoolTwo2D_NDArrayAND
BoolTwo2D_NDArrayOR
Broadcast_Add_AllocatesFullOutput
Broadcast_Square_AllocatesFullOutput
BroadcastArrayWriteThrows
Byte1D_NDArrayAND
Case1
Case2
Case2_Axis_minus1
Case2_Axis_minus1_keepdims
Case2_Axis1
Case2_Axis4
Divide_Int32_ReturnsDouble
EmptyArray_ReturnsEmptyIndices
Fmin_TwoArraysWithOut_Compiles
GreaterThan_SlicedWithScalar
IndexNDArray_Set_Case2
IndexNDArray_Set_Case3
IndexNDArray_Set_Case4
IndexNDArray_sliced3dreshaped_indexed_by_1d_1d
Mean_EmptyArray
Minimum_TwoArraysWithOut_Compiles
Multiply_SlicedWithScalar
Negative_Float64
Negative_Int32
NonZero_Empty
Prod_EmptyAlongAxis
Ravel_ContiguousRowSlice2D_ShouldBeView
Ravel_ContiguousSlice1D_ShouldBeView
Repeat_PerElement
ReproducesIssue123
Reshape_1DToScalar
Roll_MultiAxis_BothNegative
Roll_MultiAxis_BothShift1
Roll_MultiAxis_NegativeShift_TupleAxis
Roll_MultiAxis_SameAxis_Twice
Roll_MultiAxis_SameAxis1_Twice
Roll_MultiAxis_ScalarShift_TupleAxis
Roll_MultiAxis_TupleShift_TupleAxis
Roll_MultiAxis_ZeroAndNegative
Roll_MultiAxis_ZeroAndOne
Searchsorted_AfterAll
Searchsorted_BeforeAll
Searchsorted_Simple
Sign_EdgeCases
Sign_Int32
Sign_NaN_ReturnsNaN
Slice2x2Mul
Std_WithDdof
Sum_EmptyAlongAxis0
Sum_EmptyAlongAxis1
Sum_SingleColumnMatrix_Axis1
Sum_SingleRowMatrix_Axis0
Unique_Float_WithNaN
Unique_ReturnsSorted
```

---

## Tests That Are Still Failing (79 tests)

### Category 1: np.isinf not implemented (11 tests)
- `np_isinf_1D`
- `np_isinf_2D`
- `np_isinf_EmptyArray`
- `np_isinf_Float32`
- `np_isinf_IntegerTypes_AlwaysFalse`
- `np_isinf_MaxValueIsNotInfinity`
- `np_isinf_NaNIsNotInfinity`
- `np_isinf_Scalar`
- `np_isinf_SlicedArray`
- `NDArray_Isinf_WorksWithLargeArray`
- `Case24_IsInf_Mask`

### Category 2: Int8/SByte not supported (5 tests)
- `Add_Int8_Overflow_Wraps`
- `Subtract_Int8_Underflow_Wraps`
- `Multiply_Int8_Overflow_Wraps`
- `Abs_Int8_MinValue_Overflow`
- `Bug7_SByte_ArrayCreation`

### Category 3: Unsigned negative/overflow (2 tests)
- `Negative_UInt8_Wraps`
- `Negative_Byte_Overflow`

### Category 4: Bitmap/GDI related - Windows only (11 tests)
- `Bug1a_ToNDArray_CopyTrue_OddWidth24bpp_WrongShape`
- `Bug1b_ToNDArray_CopyTrue_5pxWide24bpp_ExtraPaddingBytes`
- `Bug2_ToNDArray_CopyTrue_2pxWide24bpp_WrongBpp`
- `Bug3a_AsNDArray_FlatTrue_IndexOutOfRange`
- `Bug3b_AsNDArray_FlatTrue_DiscardAlpha_IndexOutOfRange`
- `Bug4_ToBitmap_SlicedNDArray_BroadcastMismatch`
- `Bug5_ToBitmap_NonByteDtype_InvalidCastException`
- `Bug6a_ToBitmap_1pxWide24bpp_StridePaddingCrash`
- `Bug6b_ToBitmap_5pxWide24bpp_StridePaddingCrash`
- `Bug7_ToNDArray_CopyTrue_2pxWide24bpp_RoundTripCorruption`
- `Bug8_ToBitmap_UnsupportedBpp_UnhelpfulError`

### Category 5: np.random.choice replace=False ignored (5 tests)
- `Bug32_Choice_ReplaceFalse_NoDuplicates`
- `Bug32_Choice_ReplaceFalse_SizeExceedsPopulation_ShouldThrow`
- `Bug32_Choice_NDArray_ReplaceFalse`
- `NonUniformSampleWithoutReplace`
- `UniformSampleWithoutReplace`

### Category 6: Matmul/Dot not fully implemented (7 tests)
- `Matmul_1D_1D_DotProduct`
- `Matmul_1D_2D`
- `Matmul_2D_1D`
- `Matmul_3D_2D_Broadcasting`
- `Matmul_Broadcasting_3D_2D`
- `Dot_MatrixVector_Compiles`
- `Dot_ND_1D`

### Category 7: NestedView SetData corruption (3 tests)
- `NestedView_1D`
- `NestedView_1D_Stepping`
- `NestedView_2D`

### Category 8: View/Transpose returns copy instead of view (3 tests)
- `Bug_SwapAxes_0DScalar_ShouldThrow`
- `Bug_SwapAxes_ReturnsPhysicalCopy_ShouldBeView`
- `Bug_Transpose_ReturnsPhysicalCopy_ShouldBeView`

### Category 9: Boolean/Fancy indexing bugs (6 tests)
- `MaskSetter`
- `Masking_2D_over_3D`
- `Combining_IndexArrays_with_Slices`
- `Combining_MaskArrays_with_Slices`
- `Bug69_BooleanMaskGetter_ReturnsSelection`
- `SetItem_BroadcastRowTo2D`

### Category 10: Broadcast/Slice bugs (7 tests)
- `Bug_SliceBroadcast_StrideMismatch_SlicedSourceRows`
- `Bug_ToString_DoubleSlicedBroadcast`
- `Bug_ToString_ReversedSliceBroadcast`
- `Bug_ToString_SlicedColumnBroadcast`
- `Bug_ToString_StepSliceBroadcast`
- `Broadcast_SliceWithLargeIndices_Limited`
- `Broadcast_Sum_InternalError`

### Category 11: Miscellaneous (19 tests)
- `Add_0DScalars` - 0D scalar addition returns wrong ndim
- `Allocate_44GB` - Large allocation fails (expected on most systems)
- `Base_ReductionKeepdims_Size1Axis_ReturnsView` - keepdims view semantics
- `Bug12_Searchsorted_ArrayInput_WrongResults` - searchsorted with array input
- `Bug84_Prod_EmptyBool_ReturnsInt64` - prod dtype for empty bool
- `Compare` - Comparison edge case
- `Exp2_WithNPTypeCode_Compiles` - exp2 overload missing
- `Exp2_WithType_Compiles` - exp2 overload missing
- `HashHelpersLong_ExpandPrime_ProgressiveGrowthTest` - Bad primes in table
- `HashHelpersLong_GetPrime` - Bad primes in table
- `IndexNDArray_Get_Case7` - Advanced indexing edge case
- `IndexNDArray_Get_Case7_Broadcasted` - Advanced indexing edge case
- `IndexNDArray_Get_Case8_Broadcasted` - Advanced indexing edge case
- `IndexNDArray_NewAxis_Case2` - np.newaxis indexing
- `NegateBoolean_ReversedArray` - Boolean negation on sliced array
- `NegateBoolean_SlicedArray` - Boolean negation on sliced array
- `Roll_Empty2D_Axis1` - Roll on empty array
- `Slice2x2Mul_AssignmentChangesOriginal` - Assignment through slice
- `Var_Float32_NumPyReturnsFloat64` - Variance dtype promotion

---

## Recommendations

### Quick Wins (remove [OpenBugs] from 81 passing tests)
Run the following to find and edit files:
```bash
grep -rln "\[OpenBugs\]" test/NumSharp.UnitTest --include="*.cs" | xargs -I{} sed -i 's/\[OpenBugs\].*\n.*void Abs_Int32/void Abs_Int32/g' {}
```

### Priority Fixes
1. **np.isinf** - Implement in `Default.IsInf.cs` (11 tests)
2. **Int8/SByte support** - Add to type switch in UnmanagedStorage (5 tests)
3. **Matmul broadcasting** - Fix dimension handling (7 tests)
4. **Boolean indexing** - Fix mask setter/getter (6 tests)
