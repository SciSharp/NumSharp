# Test Distribution for `in` Parameter Removal

This document tracks the distribution of test writing across 3 agents to verify all overloads compile correctly after removing the `in` parameter modifier.

## Overview

- **Total overloads modified**: 199
- **Agents**: 3
- **Test file**: `test/NumSharp.UnitTest/InRemovalOverloadTests.cs`

Each agent will:
1. Run Python/NumPy to verify expected behavior for each function
2. Write C# tests that call each overload and verify the output matches NumPy

---

## Agent 1: Unary Math Operations (~70 overloads)

**Test Class**: `InRemovalOverloadTests_UnaryMath`

### Absolute/Sign (8)
- `np.absolute(a)`, `np.absolute(a, dtype)`, `np.absolute(a, NPTypeCode)`
- `np.abs(a)`, `np.abs(a, dtype)`, `np.abs(a, NPTypeCode)`
- `np.sign(a)`, `np.sign(a, dtype)`

### Roots/Powers (5)
- `np.sqrt(a)`, `np.sqrt(a, dtype)`
- `np.cbrt(a)`, `np.cbrt(a, dtype)`
- `np.square(a)`

### Rounding (12)
- `np.ceil(a)`, `np.ceil(a, dtype)`
- `np.floor(a)`, `np.floor(a, dtype)`
- `np.trunc(a)`, `np.trunc(a, dtype)`
- `np.round_(a)`, `np.round_(a, decimals)`, `np.round_(a, dtype)`, `np.round_(a, decimals, dtype)`
- `np.around(a)`, `np.around(a, decimals)`, `np.around(a, dtype)`, `np.around(a, decimals, dtype)`

### Exponential (9)
- `np.exp(a)`, `np.exp(a, dtype)`, `np.exp(a, NPTypeCode)`
- `np.exp2(a)`, `np.exp2(a, dtype)`, `np.exp2(a, NPTypeCode)`
- `np.expm1(a)`, `np.expm1(a, dtype)`, `np.expm1(a, NPTypeCode)`

### Logarithmic (12)
- `np.log(a)`, `np.log(a, dtype)`, `np.log(a, NPTypeCode)`
- `np.log2(a)`, `np.log2(a, dtype)`, `np.log2(a, NPTypeCode)`
- `np.log10(a)`, `np.log10(a, dtype)`, `np.log10(a, NPTypeCode)`
- `np.log1p(a)`, `np.log1p(a, dtype)`, `np.log1p(a, NPTypeCode)`

### Trigonometric (18)
- `np.sin(a)`, `np.sin(a, dtype)`
- `np.cos(a)`, `np.cos(a, dtype)`
- `np.tan(a)`, `np.tan(a, dtype)`
- `np.arcsin(a)`, `np.arcsin(a, dtype)`
- `np.arccos(a)`, `np.arccos(a, dtype)`
- `np.arctan(a)`, `np.arctan(a, dtype)`
- `np.sinh(a)`, `np.sinh(a, dtype)`
- `np.cosh(a)`, `np.cosh(a, dtype)`
- `np.tanh(a)`, `np.tanh(a, dtype)`

### Angle Conversion (8)
- `np.deg2rad(a)`, `np.deg2rad(a, dtype)`
- `np.rad2deg(a)`, `np.rad2deg(a, dtype)`
- `np.radians(a)`, `np.radians(a, dtype)`
- `np.degrees(a)`, `np.degrees(a, dtype)`

### Other Unary (4)
- `np.positive(a)`
- `np.negative(a)`
- `np.reciprocal(a)`, `np.reciprocal(a, dtype)`

### Modf (2)
- `np.modf(a)`, `np.modf(a, dtype)`

---

## Agent 2: Binary Math & Reductions (~70 overloads)

**Test Class**: `InRemovalOverloadTests_BinaryReductions`

### Basic Binary Ops (8)
- `np.add(a, b)`
- `np.subtract(a, b)`
- `np.multiply(a, b)`
- `np.divide(a, b)`
- `np.true_divide(a, b)`
- `np.mod(a, b)`, `np.mod(a, scalar)`

### Power (7)
- `np.power(a, scalar)`, `np.power(a, scalar, dtype)`, `np.power(a, scalar, NPTypeCode)`
- `np.power(a, b)`, `np.power(a, b, dtype)`, `np.power(a, b, NPTypeCode)`

### Floor Division (6)
- `np.floor_divide(a, b)`, `np.floor_divide(a, b, dtype)`, `np.floor_divide(a, b, NPTypeCode)`
- `np.floor_divide(a, scalar)`, `np.floor_divide(a, scalar, dtype)`, `np.floor_divide(a, scalar, NPTypeCode)`

### Min/Max (12)
- `np.maximum(a, b)`, `np.maximum(a, b, dtype)`, `np.maximum(a, b, out)`
- `np.minimum(a, b)`, `np.minimum(a, b, dtype)`, `np.minimum(a, b, out)`
- `np.fmax(a, b)`, `np.fmax(a, b, dtype)`, `np.fmax(a, b, out)`
- `np.fmin(a, b)`, `np.fmin(a, b, dtype)`, `np.fmin(a, b, out)`

### Clip (3)
- `np.clip(a, min, max)`, `np.clip(a, min, max, dtype)`, `np.clip(a, min, max, out)`

### Bitwise (8)
- `np.left_shift(a, b)`, `np.left_shift(a, scalar)`
- `np.right_shift(a, b)`, `np.right_shift(a, scalar)`
- `np.invert(a)`, `np.invert(a, dtype)`
- `np.bitwise_not(a)`, `np.bitwise_not(a, dtype)`

### Arctan2 (2)
- `np.arctan2(y, x)`, `np.arctan2(y, x, dtype)`

### Sum Reductions (10)
- `np.sum(a)`, `np.sum(a, axis)`, `np.sum(a, keepdims)`
- `np.sum(a, axis, keepdims)`, `np.sum(a, axis, keepdims, dtype)`, `np.sum(a, axis, keepdims, NPTypeCode)`
- `np.sum(a, axis, dtype)`, `np.sum(a, axis, NPTypeCode)`
- `np.sum(a, dtype)`, `np.sum(a, NPTypeCode)`

### Other Reductions (12)
- `np.prod(a, axis, dtype, keepdims)`
- `np.mean(a)`, `np.mean(a, axis)`, `np.mean(a, keepdims)`, `np.mean(a, axis, dtype, keepdims)`, `np.mean(a, axis, NPTypeCode, keepdims)`, `np.mean(a, axis, keepdims)`
- `np.nansum(a, axis, keepdims)`
- `np.nanprod(a, axis, keepdims)`
- `np.nanmean(a, axis, keepdims)`
- `np.nanmin(a, axis, keepdims)`
- `np.nanmax(a, axis, keepdims)`

### Std/Var (remaining overloads ~8)
- `np.std(a, keepdims, ddof, dtype)`, `np.std(a, axis, dtype, keepdims, ddof)`, `np.std(a, axis, NPTypeCode, keepdims, ddof)`, `np.std(a, axis, keepdims, ddof, dtype)`
- `np.var(a, keepdims, ddof, dtype)`, `np.var(a, axis, dtype, keepdims, ddof)`, `np.var(a, axis, NPTypeCode, keepdims, ddof)`, `np.var(a, axis, keepdims, ddof, dtype)`
- `np.nanstd(a, axis, keepdims, ddof)`
- `np.nanvar(a, axis, keepdims, ddof)`

---

## Agent 3: Logic, Comparison, Manipulation & Linear Algebra (~59 overloads)

**Test Class**: `InRemovalOverloadTests_LogicManipulation`

### Comparison (18)
- `np.equal(a, b)`, `np.equal(a, scalar)`, `np.equal(scalar, a)`
- `np.not_equal(a, b)`, `np.not_equal(a, scalar)`, `np.not_equal(scalar, a)`
- `np.less(a, b)`, `np.less(a, scalar)`, `np.less(scalar, a)`
- `np.greater(a, b)`, `np.greater(a, scalar)`, `np.greater(scalar, a)`
- `np.less_equal(a, b)`, `np.less_equal(a, scalar)`, `np.less_equal(scalar, a)`
- `np.greater_equal(a, b)`, `np.greater_equal(a, scalar)`, `np.greater_equal(scalar, a)`

### Logical (4)
- `np.logical_and(a, b)`
- `np.logical_or(a, b)`
- `np.logical_not(a)`
- `np.logical_xor(a, b)`

### Axis Manipulation (8)
- `np.moveaxis(a, source, dest)`, `np.moveaxis(a, sources[], dest)`, `np.moveaxis(a, source, dests[])`, `np.moveaxis(a, sources[], dests[])`
- `np.rollaxis(a, axis, start)`
- `np.swapaxes(a, axis1, axis2)`
- `np.transpose(a)`, `np.transpose(a, permute)`

### Unique (1)
- `np.unique(a)`

### Counting/Indexing (3)
- `np.count_nonzero(a)`, `np.count_nonzero(a, axis, keepdims)`
- `np.nonzero(a)`

### Linear Algebra (3)
- `np.dot(a, b)`
- `np.matmul(a, b)`
- `np.outer(a, b)`

### Min with dtype (3)
- `np.amin<T>(a)`
- `np.amin(a, axis, keepdims, dtype)`
- `np.min(a, axis, keepdims, dtype)`

---

## Test File Structure

```
test/NumSharp.UnitTest/
└── InRemoval/
    ├── InRemovalOverloadTests_UnaryMath.cs      (Agent 1)
    ├── InRemovalOverloadTests_BinaryReductions.cs (Agent 2)
    └── InRemovalOverloadTests_LogicManipulation.cs (Agent 3)
```

## Status

| Agent | Category | Status | Tests Written |
|-------|----------|--------|---------------|
| 1 | Unary Math | Pending | 0/70 |
| 2 | Binary & Reductions | Pending | 0/70 |
| 3 | Logic & Manipulation | Pending | 0/59 |

---

## Instructions for Agents

1. **Run NumPy in Python** to get expected outputs for representative inputs
2. **Write TUnit tests** that:
   - Call each overload with the same inputs
   - Assert the output matches NumPy
   - Use `[Test]` attribute (TUnit framework)
3. **Test naming**: `{FunctionName}_{Overload}_{Scenario}`
4. **Group related tests** in the same test class
5. **Focus on compilation** - the primary goal is to verify the overload compiles and can be called
