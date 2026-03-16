# NumSharp Kernel & API Audit - Definition of Done

## Purpose

This document tracks the audit and alignment of NumSharp kernel operations against NumPy 2.x API specifications. Every operation must satisfy the Definition of Done (DOD) criteria before being considered complete.

## Definition of Done (DOD) Criteria

Every np.* function and DefaultEngine operation MUST:

### 1. Memory Layout Support
- [ ] **Contiguous arrays**: Works correctly with C-contiguous memory
- [ ] **Non-contiguous arrays**: Works correctly with sliced/strided/transposed views
- [ ] **Broadcast arrays**: Works correctly with stride=0 dimensions (read-only)
- [ ] **Sliced views**: Correctly handles Shape.offset for base address calculation

### 2. Dtype Support (12 NumSharp types)
- [ ] Boolean (bool)
- [ ] Byte (uint8)
- [ ] Int16 (int16)
- [ ] UInt16 (uint16)
- [ ] Int32 (int32)
- [ ] UInt32 (uint32)
- [ ] Int64 (int64)
- [ ] UInt64 (uint64)
- [ ] Char (char - NumSharp extension)
- [ ] Single (float32)
- [ ] Double (float64)
- [ ] Decimal (decimal - NumSharp extension)

### 3. NumPy API Parity
- [ ] Function signature matches NumPy (parameter names, order, defaults)
- [ ] Type promotion matches NumPy 2.x (NEP50)
- [ ] Edge cases match NumPy (empty arrays, scalars, NaN handling, broadcasting)
- [ ] Return dtype matches NumPy exactly

### 4. Testing
- [ ] Unit tests based on actual NumPy output
- [ ] Edge case tests (empty, scalar, broadcast, strided)
- [ ] Dtype coverage tests for all 12 types

---

## Audit Status by Category

### Binary Operations (ILKernelGenerator.Binary.cs, ILKernelGenerator.MixedType.cs)

| Operation | Contiguous | Non-Contiguous | All dtypes | API Match | Tests |
|-----------|------------|----------------|------------|-----------|-------|
| Add | ✅ | ✅ | ✅ | ✅ | ✅ |
| Subtract | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multiply | ✅ | ✅ | ✅ | ✅ | ✅ |
| Divide | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mod | ✅ | ✅ | ✅ | ⚠️ | 🔲 |
| Power | ✅ | ✅ | ✅ | ✅ | ✅ |
| FloorDivide | ✅ | ✅ | ✅ | ✅ | ✅ |
| BitwiseAnd | ✅ | ✅ | ⚠️ int only | ✅ | 🔲 |
| BitwiseOr | ✅ | ✅ | ⚠️ int only | ✅ | 🔲 |
| BitwiseXor | ✅ | ✅ | ⚠️ int only | ✅ | 🔲 |
| LeftShift | ✅ | ✅ | ⚠️ int only | ⚠️ | ✅ |
| RightShift | ✅ | ✅ | ⚠️ int only | ⚠️ | ✅ |
| ATan2 | ✅ | ✅ | ⚠️ float only | ✅ | ✅ |

Legend: ✅ Complete | ⚠️ Partial | 🔲 Not verified | ❌ Missing/Bug

**Audit Notes (2026-03-08):**
- Add/Sub/Mul/Div/Mod/Power/FloorDivide: Use ILKernelGenerator with proper path classification
- BitwiseAnd/Or/Xor: Integer types only (correct behavior)
- LeftShift/RightShift: Use `.copy()` pattern to materialize non-contiguous before processing
- ATan2: FIXED - Now uses IL kernels with proper stride/offset/broadcast handling

### Unary Operations (ILKernelGenerator.Unary.cs)

| Operation | Contiguous | Non-Contiguous | All dtypes | API Match | Tests |
|-----------|------------|----------------|------------|-----------|-------|
| Negate | ✅ | ❌ **BUG** (bool) | ✅ | ✅ | 🔲 |
| Abs | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Sign | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Sqrt | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Cbrt | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Square | ✅ | ✅ | ⚠️ promotes | ⚠️ | 🔲 |
| Reciprocal | ✅ | ✅ | ⚠️ promotes | ⚠️ | 🔲 |
| Floor | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Ceil | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Truncate | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Sin | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Cos | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Tan | ✅ | ✅ | ✅ | ✅ | 🔲 |
| ASin | ✅ | ✅ | ✅ | ✅ | 🔲 |
| ACos | ✅ | ✅ | ✅ | ✅ | 🔲 |
| ATan | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Sinh | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Cosh | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Tanh | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Exp | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Exp2 | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Expm1 | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Log | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Log2 | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Log10 | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Log1p | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Deg2Rad | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Rad2Deg | ✅ | ✅ | ✅ | ✅ | 🔲 |
| BitwiseNot | ✅ | ✅ | ⚠️ int only | ⚠️ bool wrong | ✅ |

**Audit Notes (2026-03-08):**
- All unary ops use ILKernelGenerator with IsContiguous flag in kernel key
- Shape.offset correctly applied at kernel invocation (DefaultEngine.UnaryOp.cs:148)
- Strided iteration uses correct coordinate decomposition
- **NegateBoolean BUG**: Bypasses IL kernel, uses linear indexing without strides (Task #74)

### Comparison Operations (ILKernelGenerator.Comparison.cs)

| Operation | Contiguous | Non-Contiguous | All dtypes | API Match | Tests |
|-----------|------------|----------------|------------|-----------|-------|
| Equal | ✅ | ✅ | ✅ | ✅ | 🔲 |
| NotEqual | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Less | ✅ | ✅ | ✅ | ✅ | 🔲 |
| LessEqual | ✅ | ✅ | ✅ | ✅ | 🔲 |
| Greater | ✅ | ✅ | ✅ | ✅ | 🔲 |
| GreaterEqual | ✅ | ✅ | ✅ | ✅ | 🔲 |

### Reduction Operations (ILKernelGenerator.Reduction.cs)

| Operation | Contiguous | Non-Contiguous | All dtypes | API Match | Tests |
|-----------|------------|----------------|------------|-----------|-------|
| Sum | ✅ | ✅ (axis) | ✅ | ✅ | 🔲 |
| Prod | ✅ | ✅ (axis) | ✅ | ✅ | 🔲 |
| Min | ✅ | ✅ (axis) | ✅ | ✅ | 🔲 |
| Max | ✅ | ✅ (axis) | ✅ | ✅ | 🔲 |
| Mean | ✅ | ✅ (axis) | ✅ | ✅ | 🔲 |
| ArgMax | ✅ | ✅ (axis) | ✅ | ⚠️ no keepdims | ✅ |
| ArgMin | ✅ | ✅ (axis) | ✅ | ⚠️ no keepdims | ✅ |
| All | ✅ | ✅ (axis) | ✅ | ✅ | ✅ |
| Any | ✅ | ✅ (axis) | ✅ | ✅ | ✅ |
| Std | ✅ | ✅ | ✅ | ✅ ddof works | 🔲 |
| Var | ✅ | ✅ | ✅ | ✅ ddof works | 🔲 |
| CumSum | ✅ | ✅ | ✅ | ✅ | 🔲 |

**Audit Notes (2026-03-08):**
- All reductions use IL kernels for axis=None (SIMD), iterator-based for axis reductions
- Type promotion matches NEP50 (int32 sum -> int64)
- **ArgMax/ArgMin BUG**: Missing `keepdims` parameter (Task #76)
- Std/Var: SIMD helpers exist but unused in element-wise path (performance gap)
- Multiple axes `axis=(0,2)` NOT supported

### Standalone Operations (Default.*.cs)

| Operation | Contiguous | Non-Contiguous | All dtypes | API Match | Tests |
|-----------|------------|----------------|------------|-----------|-------|
| Clip | ✅ | ⚠️ | ✅ | ✅ | 🔲 |
| Modf | ✅ | ⚠️ | ✅ | ✅ | 🔲 |
| NonZero | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Known Misalignments

Document behavioral differences with `[Misaligned]` tests:

| Function | NumSharp Behavior | NumPy Behavior | Status |
|----------|-------------------|----------------|--------|
| `np.square(int)` | Returns double | Returns int | [Misaligned] documented |
| `np.reciprocal(int)` | Returns double(0.5) | Returns int(0) | [Misaligned] documented |
| `np.invert(bool)` | Bitwise NOT (~1=0xFE) | Logical NOT | [Misaligned] documented |

---

## API Parameter Audit

NumPy functions have standardized parameters. Verify NumSharp matches:

### Ufunc Standard Parameters
```python
numpy.add(x1, x2, /, out=None, *, where=True, casting='same_kind',
          order='K', dtype=None, subok=True)
```

NumSharp SHOULD support (minimum viable):
- `x1, x2` - operands ✅
- `dtype` - output type ✅ (via overloads)
- `out` - output array 🔲 (not implemented)
- `where` - boolean mask 🔲 (not implemented)

### Reduction Standard Parameters
```python
numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=0, where=True)
```

NumSharp SHOULD support (minimum viable):
- `a` - input array ✅
- `axis` - reduction axis ✅
- `dtype` - accumulator type ✅
- `keepdims` - preserve dimensions ✅
- `out` - output array 🔲
- `initial` - starting value 🔲
- `where` - mask 🔲

---

## Implementation Gaps

### High Priority (blocks common use cases)
1. `out` parameter support for in-place operations
2. `where` parameter support for conditional operations
3. `axis` parameter support for all reductions

### Medium Priority (alignment with NumPy)
1. Type preservation for `square`, `reciprocal` on integers
2. Logical NOT for boolean `invert`
3. `ddof` parameter for `std`/`var`

### Low Priority (edge cases)
1. `order` parameter (NumSharp is C-only)
2. `casting` parameter
3. `subok` parameter (no subclassing support)

---

## Verification Process

For each operation, verify:

1. **Read NumPy docs** - Check exact signature and behavior
2. **Run NumPy tests** - Execute actual Python code for edge cases
3. **Compare output** - Verify NumSharp matches NumPy exactly
4. **Document differences** - Create [Misaligned] tests for intentional differences

```python
# Example verification script
import numpy as np

# Test contiguous
a = np.array([1, 2, 3])
print(f"contiguous: {np.sqrt(a)}, dtype={np.sqrt(a).dtype}")

# Test non-contiguous (sliced)
b = np.array([1, 2, 3, 4, 5])[::2]
print(f"non-contiguous: {np.sqrt(b)}, dtype={np.sqrt(b).dtype}")

# Test broadcast
c = np.array([[1], [2], [3]])
d = np.array([1, 2, 3])
print(f"broadcast: {np.add(c, d)}")
```

---

## Bugs Found During Audit

| Task | Bug | File | Priority | Status |
|------|-----|------|----------|--------|
| #73 | ATan2 ignores strides/offset/broadcast | Default.ATan2.cs | HIGH | **Resolved** |
| #74 | NegateBoolean ignores strides | Default.Negate.cs | HIGH | **Resolved** |
| #76 | ArgMax/ArgMin missing keepdims | Default.Reduction.ArgMax/Min.cs | MEDIUM | **Resolved** |
| #75 | outType vs dtype naming inconsistency | Various np.*.cs | LOW | **Resolved** |
| #77 | np.power only accepts scalar exponent | np.power.cs | MEDIUM | **Resolved** |
| #78 | Missing np.logical_and/or/not/xor | np.logical.cs | LOW | **Resolved** |
| #79 | Missing np.equal/less/greater functions | np.comparison.cs | LOW | **Resolved** |

---

---

## API Signature Gaps

### Missing NumPy Parameters (acceptable to omit for now)
- `out` - In-place output array
- `where` - Conditional mask
- `initial` - Starting value for reductions
- `casting`, `order`, `subok` - Advanced type control

### NumSharp Extensions (not in NumPy)
- `amax`/`amin` have `dtype` parameter (NumPy doesn't)

### Naming Inconsistencies
- ~~Some functions use `outType`, others use `dtype`~~ (Task #75 - **Resolved**: all renamed to `dtype`)

### Missing Function APIs (operators exist, functions don't)
| NumPy Function | NumSharp Equivalent | Task |
|----------------|---------------------|------|
| `np.equal(x1, x2)` | `x1 == x2` operator | #79 |
| `np.not_equal(x1, x2)` | `x1 != x2` operator | #79 |
| `np.less(x1, x2)` | `x1 < x2` operator | #79 |
| `np.greater(x1, x2)` | `x1 > x2` operator | #79 |
| `np.less_equal(x1, x2)` | `x1 <= x2` operator | #79 |
| `np.greater_equal(x1, x2)` | `x1 >= x2` operator | #79 |
| `np.logical_and(x1, x2)` | None | #78 |
| `np.logical_or(x1, x2)` | None | #78 |
| `np.logical_not(x)` | None | #78 |
| `np.logical_xor(x1, x2)` | None | #78 |

### Functional Gaps
| Function | Issue | Task |
|----------|-------|------|
| `np.power(arr, arr)` | Only supports scalar exponent, not NDArray | #77 |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-08 | **AUDIT COMPLETE**: All 17 tasks done, build passes, 3058/3283 tests pass | coordinator |
| 2026-03-08 | Task #75: Renamed outType to dtype in 19 np.*.cs files | api-auditor |
| 2026-03-08 | Fixed: ATan2 (#73), NegateBoolean (#74), ArgMax/Min keepdims (#76) | agents |
| 2026-03-08 | Added: np.power(NDArray,NDArray) (#77), np.logical_* (#78), np.equal/less/etc (#79) | agents |
| 2026-03-08 | API supplementary: power, comparison, logical gaps (#77-79) | api-auditor |
| 2026-03-08 | Reduction/API audit complete, bugs #75 #76 found | reduction-auditor, api-auditor |
| 2026-03-08 | Binary/Unary audit complete, bugs #73 #74 found | binary-auditor, unary-auditor |
| 2026-03-08 | Initial audit document created | Claude |
