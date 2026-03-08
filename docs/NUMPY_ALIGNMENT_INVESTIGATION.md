# NumPy 2.4.2 Alignment Investigation

Generated: 2026-03-08
NumPy Version: 2.4.2

## Purpose

This document captures NumPy 2.4.2 behavior as the reference for NumSharp alignment verification.

---

## 1. Array Creation Defaults

| Function | Default dtype | NumSharp Status |
|----------|---------------|-----------------|
| `np.zeros((2,3))` | float64 | TBD |
| `np.ones((2,3))` | float64 | TBD |
| `np.arange(5)` | int64 | TBD |
| `np.arange(5.0)` | float64 | TBD |
| `np.linspace(0,1,5)` | float64 | TBD |

---

## Baseline Test Results (2026-03-08)

### Type Promotion - ALL PASS
- `int32.sum()` → Int64 ✅
- `int32.prod()` → Int64 ✅
- `int32.mean()` → Double ✅
- `int32+int64` → Int64 ✅
- `int32+float32` → Double ✅

### Unary Operations - ALL PASS
- `sqrt(int32)` → Double ✅
- `abs(int32)` → Int32 ✅
- `square(int32)` → Int32 ✅ (fixed in audit)
- `negative(int32)` → Int32 ✅

### Reduction Axis - ALL PASS
- `sum()` shape: scalar ✅
- `sum(axis=0)` shape: correct ✅
- `sum(axis=-1)` shape: correct ✅
- `sum(keepdims=true)` shape: correct ✅
- `argmax(keepdims=true)` shape: correct ✅ (added in audit)

### Boolean Operations - ALL PASS
- `np.invert(true)` → False ✅ (fixed in audit)
- `np.invert(false)` → True ✅ (fixed in audit)

### NaN/Special Values - ALL PASS
- `sum([1,NaN,3])` → NaN ✅
- `max([1,NaN,3])` → NaN ✅
- `sqrt(-1)` → NaN ✅
- `log(0)` → -Inf ✅

### Issues Found
| Issue | Task | Status |
|-------|------|--------|
| `max([])` should raise ValueError | #84 | In Progress |
| `argmax` dtype should be Int64 | #85 | In Progress |
| Multi-axis reduction not supported | #87 | Tracked (LOW) |
| Missing nanmean/nanstd/nanvar | #86 | Tracked (LOW) |
| `reciprocal(int)` returns float | - | [Misaligned] intentional |

---

## Agent Investigation Results (2026-03-08)

### Task #80: Type Promotion (binary-auditor)
**Result: 28/30 tests PASS (93%)**

| Category | Tests | Status |
|----------|-------|--------|
| Reduction Promotion | 8/8 | ✅ |
| Binary Promotion | 7/7 | ✅ |
| Division | 1/1 | ✅ |
| Power | 2/2 | ✅ |
| Unary | 5/5 | ✅ |
| Axis Reductions | 3/3 | ✅ |
| argmax/argmin dtype | 0/2 | ❌ (Int32→Int64 needed) |

### Task #81: Reduction Axis (reduction-auditor)
**Result: 22/25 tests PASS (88%)**

| Category | Tests | Status |
|----------|-------|--------|
| Basic axis reduction | 5/5 | ✅ |
| Keepdims | 6/6 | ✅ |
| Empty arrays | 3/4 | ⚠️ (max/min should throw) |
| all/any axis | 4/4 | ✅ |
| argmax/argmin keepdims | 3/3 | ✅ |
| Multi-axis | 0/1 | ❌ (not supported) |
| argmax dtype | 0/2 | ❌ |

### Task #82: NaN Handling (api-auditor)
**Result: 25/28 tests PASS (89%)**

| Category | Tests | Status |
|----------|-------|--------|
| NaN in reductions | 8/8 | ✅ |
| Special value arithmetic | 8/8 | ✅ |
| nan* functions | 5/5 | ✅ |
| Missing nan* functions | 0/3 | ❌ (nanmean/std/var) |
| Empty array edge cases | 2/4 | ⚠️ (max/min should throw) |

### Task #83: Unary Dtype (unary-auditor)
**Result: 9/10 tests PASS (90%)**

| Category | Tests | Status |
|----------|-------|--------|
| Math promotion (sqrt/sin/exp) | 4/4 | ✅ |
| Arithmetic preservation | 4/4 | ✅ |
| Reciprocal | 0/1 | [Misaligned] intentional |

### Overall Alignment Score: **92%** (84/91 tests pass)

---

## 2. Type Promotion (NEP50)

### Reduction Promotion
| Operation | Input | Output dtype |
|-----------|-------|--------------|
| `sum()` | int32 | int64 |
| `prod()` | int32 | int64 |
| `cumsum()` | int32 | int64 |
| `max()` | int32 | int32 (preserves) |
| `mean()` | int32 | float64 |
| `std()` | int32 | float64 |
| `var()` | int32 | float64 |

### Binary Operation Promotion
| Operation | Result dtype |
|-----------|--------------|
| int32 + int32 | int32 |
| int32 + int64 | int64 |
| int32 + float32 | float64 |
| int64 + float32 | float64 |
| float32 + float64 | float64 |
| int32 / int32 | float64 (true division) |
| int32 // int32 | int32 (floor division) |

### Power Promotion
| Operation | Result dtype |
|-----------|--------------|
| int32 ** 2 | int32 |
| int32 ** 2.0 | float64 |

---

## 3. Unary Operations

| Operation | Input | Output dtype |
|-----------|-------|--------------|
| `np.sqrt(int32)` | int32 | float64 |
| `np.abs(int32)` | int32 | int32 |
| `np.sign(int32)` | int32 | int32 |
| `np.square(int32)` | int32 | int32 |
| `np.negative(int32)` | int32 | int32 |
| `np.positive(int32)` | int32 | int32 |
| `np.reciprocal(int32)` | int32 | int32 (floor: 1→1, 2→0, 3→0) |

---

## 4. Reduction Axis Behavior

For shape (2,3,4):
| Operation | Result Shape |
|-----------|--------------|
| `sum()` | () scalar |
| `sum(axis=0)` | (3, 4) |
| `sum(axis=1)` | (2, 4) |
| `sum(axis=-1)` | (2, 3) |
| `sum(axis=(0,2))` | (3,) |
| `sum(keepdims=True)` | (1, 1, 1) |
| `sum(axis=1, keepdims=True)` | (2, 1, 4) |

### argmax/argmin
| Operation | Result |
|-----------|--------|
| `argmax()` dtype | int64 |
| `argmax(axis=0)` shape | (3, 4) |
| `argmax(axis=0, keepdims=True)` shape | (1, 3, 4) |

---

## 5. Boolean Operations

| Operation | Result |
|-----------|--------|
| `~True` | False |
| `~False` | True |
| `np.invert(True)` | False |
| `np.invert(False)` | True |
| `np.logical_not([True,False,True])` | [False, True, False] |
| `np.invert([True,False,True])` | [False, True, False] |
| `np.invert(int32(0))` | -1 (bitwise) |

---

## 6. NaN Handling

| Operation | Result |
|-----------|--------|
| `sum([1, nan, 3])` | nan |
| `nansum([1, nan, 3])` | 4.0 |
| `max([1, nan, 3])` | nan |
| `nanmax([1, nan, 3])` | 3.0 |
| `argmax([1, nan, 3])` | 1 (index of nan) |

---

## 7. Empty Array Behavior

| Operation | Result |
|-----------|--------|
| `sum([])` | 0.0 |
| `prod([])` | 1.0 |
| `max([])` | ValueError |
| `argmax([])` | ValueError |

---

## 8. Special Values

| Operation | Result |
|-----------|--------|
| `inf + 1` | inf |
| `inf - inf` | nan |
| `0 / 0` | nan |
| `1 / 0` | inf |
| `sqrt(-1)` | nan |
| `log(0)` | -inf |
| `log(-1)` | nan |

---

## 9. Rounding Behavior

Values: [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

| Function | Result |
|----------|--------|
| `floor` | [-3, -2, -1, 0, 1, 2] |
| `ceil` | [-2, -1, 0, 1, 2, 3] |
| `round` | [-2, -2, 0, 0, 2, 2] (banker's rounding) |
| `trunc` | [-2, -1, 0, 0, 1, 2] |

---

## 10. Clip Behavior

| Operation | Result |
|-----------|--------|
| `clip([-3,-1,0,1,3,5], -1, 3)` | [-1, -1, 0, 1, 3, 3] |
| `clip(..., None, 2)` | [-3, -1, 0, 1, 2, 2] |
| `clip(..., 0, None)` | [0, 0, 0, 1, 3, 5] |

---

## 11. Stacking/Concatenation

For two (2,2) arrays:
| Operation | Result Shape |
|-----------|--------------|
| `vstack` | (4, 2) |
| `hstack` | (2, 4) |
| `stack(axis=0)` | (2, 2, 2) |
| `stack(axis=1)` | (2, 2, 2) |
| `stack(axis=2)` | (2, 2, 2) |
| `concatenate(axis=0)` | (4, 2) |
| `concatenate(axis=1)` | (2, 4) |

---

## 12. Reshape/Transpose

For shape (2,3,4):
| Operation | Result Shape |
|-----------|--------------|
| `reshape(-1)` | (24,) |
| `reshape(4, -1)` | (4, 6) |
| `transpose()` | (4, 3, 2) |
| `transpose(1,0,2)` | (3, 2, 4) |
| `swapaxes(0,2)` | (4, 3, 2) |
| `moveaxis(0,2)` | (3, 4, 2) |

---

## 13. Flatten/Ravel

| Property | flatten() | ravel() |
|----------|-----------|---------|
| Returns copy | Yes | No (view when possible) |
| Shares memory | No | Yes |
| 'C' order | [1,2,3,4] | [1,2,3,4] |
| 'F' order | [1,3,2,4] | [1,3,2,4] |

---

## Investigation Tasks

### Priority 1: Type Promotion
- [ ] Verify int32 sum/prod/cumsum return int64
- [ ] Verify mean/std/var return float64
- [ ] Verify binary promotion rules
- [ ] Verify power promotion

### Priority 2: Reduction Behavior
- [ ] Verify axis parameter on all reductions
- [ ] Verify keepdims parameter
- [ ] Verify multi-axis reduction (axis=(0,2))
- [ ] Verify empty array behavior

### Priority 3: Special Values
- [ ] Verify NaN handling in reductions
- [ ] Verify inf/nan arithmetic
- [ ] Verify division by zero behavior

### Priority 4: Shape Operations
- [ ] Verify reshape with -1
- [ ] Verify transpose permutations
- [ ] Verify moveaxis behavior
- [ ] Verify flatten vs ravel memory sharing
