# Kernel Completion Plan

> Technical orchestration document for completing ILKernelGenerator coverage in NumSharp.
> This is the source of truth for kernel implementation work.

## Current State Summary

| Category | Implemented | Missing | Coverage |
|----------|-------------|---------|----------|
| Binary Operations | 8 | 4 | 67% |
| Unary Operations | 23 | 7 | 77% |
| Comparison Operations | 6 | 0 | 100% |
| Reduction Operations | 10 | 6 | 63% |
| Interface Routing | 15 methods | 6 SIMD helpers | 71% |

**Total estimated Regen/legacy code to replace:** ~68,000 lines

---

## Work Streams

### Stream 1: Interface Completion (Priority: HIGH)

Route all SIMD helpers through IKernelProvider interface for backend abstraction.

| Helper Method | Current Location | Called From | Action |
|---------------|------------------|-------------|--------|
| `AllSimdHelper<T>` | Reduction.cs:663 | np.all.cs | Add to interface |
| `AnySimdHelper<T>` | Reduction.cs:737 | np.any.cs | Add to interface |
| `NonZeroSimdHelper<T>` | Reduction.cs:1107 | Default.NonZero.cs | Add to interface |
| `ConvertFlatIndicesToCoordinates` | Reduction.cs:1191 | Default.NonZero.cs | Add to interface |
| `CountTrueSimdHelper` | Reduction.cs:1234 | NDArray.Indexing.Masking.cs | Add to interface |
| `CopyMaskedElementsHelper<T>` | Reduction.cs:1307 | NDArray.Indexing.Masking.cs | Add to interface |

**Interface additions required:**
```csharp
// Boolean reductions (contiguous fast path)
bool All<T>(T* data, int size) where T : unmanaged;
bool Any<T>(T* data, int size) where T : unmanaged;

// NonZero support
void FindNonZero<T>(T* data, int size, List<int> indices) where T : unmanaged;
NDArray<int>[] ConvertFlatToCoordinates(List<int> flatIndices, int[] shape);

// Boolean masking support
int CountTrue(bool* data, int size);
void CopyMasked<T>(T* src, bool* mask, T* dest, int size) where T : unmanaged;
```

---

### Stream 2: Missing Binary Operations (Priority: HIGH)

| Operation | BinaryOp Enum | SIMD Feasibility | Implementation Notes |
|-----------|---------------|------------------|---------------------|
| **Power** | `Power` | Partial | Float: SIMD possible; Int: use scalar Math.Pow |
| **FloorDivide** | `FloorDivide` | Yes | Integer division + floor for floats |
| **LeftShift** | `LeftShift` | Yes | `Vector.ShiftLeft` intrinsic |
| **RightShift** | `RightShift` | Yes | `Vector.ShiftRightArithmetic/Logical` |

**np.* API status:**
- `np.power` - EXISTS, uses DefaultEngine
- `np.floor_divide` - MISSING, needs creation
- `np.left_shift` - MISSING, needs creation
- `np.right_shift` - MISSING, needs creation

---

### Stream 3: Missing Unary Operations (Priority: MEDIUM)

| Operation | UnaryOp Enum | SIMD Feasibility | Implementation Notes |
|-----------|--------------|------------------|---------------------|
| **Truncate** | `Truncate` | Yes | `Vector.Truncate` available |
| **Reciprocal** | `Reciprocal` | Yes | `Vector.Divide(1, x)` |
| **Square** | `Square` | Yes | `Vector.Multiply(x, x)` |
| **Cbrt** | `Cbrt` | No | Scalar `Math.Cbrt` |
| **Deg2Rad** | `Deg2Rad` | Yes | `x * (π/180)` |
| **Rad2Deg** | `Rad2Deg` | Yes | `x * (180/π)` |
| **BitwiseNot** | `BitwiseNot` | Yes | `Vector.OnesComplement` |

**np.* API status:**
- `np.trunc` - EXISTS (uses np.fix)
- `np.reciprocal` - MISSING
- `np.square` - EXISTS (uses np.power)
- `np.cbrt` - MISSING
- `np.deg2rad` - MISSING
- `np.rad2deg` - MISSING
- `np.bitwise_not` / `np.invert` - MISSING

---

### Stream 4: Missing Reduction Operations (Priority: MEDIUM)

| Operation | ReductionOp Enum | SIMD Feasibility | Implementation Notes |
|-----------|------------------|------------------|---------------------|
| **Std** | `Std` | Partial | Two-pass: mean, then variance |
| **Var** | `Var` | Partial | Single-pass Welford or two-pass |
| **NanSum** | `NanSum` | Yes | Replace NaN with 0, then sum |
| **NanProd** | `NanProd` | Yes | Replace NaN with 1, then prod |
| **NanMin** | `NanMin` | Partial | Skip NaN in comparison |
| **NanMax** | `NanMax` | Partial | Skip NaN in comparison |

**np.* API status:**
- `np.std`, `np.var` - EXIST, use DefaultEngine (iterator path)
- `np.nansum`, `np.nanprod`, `np.nanmin`, `np.nanmax` - MISSING

---

### Stream 5: Quick Wins - Element-wise Ops (Priority: HIGH)

Operations with existing DefaultEngine that are simple SIMD candidates.

| Operation | File | Lines | SIMD Approach |
|-----------|------|-------|---------------|
| **Clip** | Default.Clip.cs | ~1,200 | `Vector.Min(Vector.Max(x, min), max)` |
| **Modf** | Default.Modf.cs | ~86 | `Vector.Truncate` + subtract |
| **ATan2** | Default.ATan2.cs | ~128 | Scalar only (transcendental) + **FIX BUG** |

**ATan2 Bug:** Uses `byte*` for x operand regardless of actual type (line ~35).

---

### Stream 6: Axis Reductions (Priority: LOW - Complex)

These use iterator paths and are complex to migrate.

| Operation | File | Lines | Notes |
|-----------|------|-------|-------|
| Sum (axis) | Default.Reduction.Add.cs | ~4,000 | Regen template |
| Prod (axis) | Similar | ~4,000 | Regen template |
| Mean (axis) | Similar | ~4,000 | Uses Sum internally |
| Max/Min (axis) | Similar | ~4,000 | Regen template |
| CumSum (axis) | Default.Reduction.CumAdd.cs | ~4,500 | Sequential dependency |
| Std/Var (axis) | Default.Reduction.Std/Var.cs | ~20,000 | Two-pass |
| ArgMax/ArgMin (axis) | Default.Reduction.ArgMax/Min.cs | ~1,000 | Index tracking |

**Strategy:** SIMD optimization for inner-axis when contiguous, iterator fallback otherwise.

---

### Stream 7: MatMul Optimization (Priority: LOW - Major Effort)

| File | Lines | Current Approach |
|------|-------|------------------|
| Default.MatMul.2D2D.cs | 19,924 | Regen 12×12 type combinations |

**Options:**
1. IL kernel with SIMD inner products
2. Integrate with BLAS library (MKL, OpenBLAS)
3. Keep Regen but optimize inner loop

---

## Implementation Order

### Phase 1: Interface & Quick Wins (Week 1)
1. ✅ Route SIMD helpers through IKernelProvider
2. ✅ Implement Clip kernel
3. ✅ Fix ATan2 bug + implement

### Phase 2: Binary Operations (Week 2)
4. Implement Power kernel
5. Implement FloorDivide kernel + np.floor_divide API
6. Implement LeftShift/RightShift kernels + APIs

### Phase 3: Unary Operations (Week 2-3)
7. Implement Truncate, Reciprocal, Square kernels
8. Implement Deg2Rad, Rad2Deg kernels + APIs
9. Implement BitwiseNot kernel + np.invert API
10. Implement Cbrt kernel + np.cbrt API

### Phase 4: Reduction Operations (Week 3)
11. Implement Var/Std kernels (element-wise first)
12. Implement NanSum/NanProd kernels + APIs
13. Implement NanMin/NanMax kernels + APIs

### Phase 5: Axis Operations (Week 4+)
14. Axis reduction SIMD optimization
15. CumSum axis optimization
16. MatMul optimization (if prioritized)

---

## Files Reference

### ILKernelGenerator Partial Files
| File | Responsibility |
|------|----------------|
| `ILKernelGenerator.cs` | Core, SIMD detection, type utilities |
| `ILKernelGenerator.Binary.cs` | Same-type binary ops |
| `ILKernelGenerator.MixedType.cs` | Mixed-type binary, `ClearAll()` |
| `ILKernelGenerator.Unary.cs` | Unary element-wise |
| `ILKernelGenerator.Comparison.cs` | Comparison returning bool |
| `ILKernelGenerator.Reduction.cs` | Element reductions, SIMD helpers |

### Key Enum Files
| File | Contains |
|------|----------|
| `KernelOp.cs` | BinaryOp, UnaryOp, ReductionOp, ComparisonOp enums |
| `KernelSignatures.cs` | Delegate types for kernels |
| `IKernelProvider.cs` | Interface definition |

### DefaultEngine Files to Migrate
| File | Operation | Priority |
|------|-----------|----------|
| `Default.Clip.cs` | Clip | HIGH |
| `Default.Power.cs` | Power | HIGH |
| `Default.ATan2.cs` | ATan2 | MEDIUM (has bug) |
| `Default.Modf.cs` | Modf | MEDIUM |
| `Default.Reduction.Std.cs` | Std | MEDIUM |
| `Default.Reduction.Var.cs` | Var | MEDIUM |

---

## Success Criteria

1. All SIMD helpers routed through IKernelProvider interface
2. All BinaryOp enum values have kernel implementations
3. All UnaryOp enum values have kernel implementations
4. All ReductionOp enum values have kernel implementations
5. Missing np.* APIs created for new operations
6. Tests passing for all new kernels
7. Benchmarks showing SIMD speedup vs scalar

---

## Team Assignments

| Agent | Stream | Focus |
|-------|--------|-------|
| **interface-agent** | Stream 1 | IKernelProvider interface additions |
| **binary-agent** | Stream 2 | Binary operation kernels |
| **unary-agent** | Stream 3 | Unary operation kernels |
| **reduction-agent** | Stream 4 | Reduction operation kernels |
| **quickwin-agent** | Stream 5 | Clip, Modf, ATan2 fix |

---

*Last updated: 2026-03-07*
