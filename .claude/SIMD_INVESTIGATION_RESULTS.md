# SIMD Optimization Investigation Results

## Executive Summary

The investigation revealed NumSharp already has optimal SIMD scalar paths for **same-type operations** (via C# SimdKernels), but **mixed-type operations** fell back to scalar loops in IL kernels. **This has now been fixed.**

### Implementation Complete ✅

SIMD scalar paths have been added to the IL kernel generator for mixed-type operations where the array type equals the result type (no per-element conversion needed).

**Final Benchmark Results:**
```
Array size: 10,000,000 elements

Same-type operations (C# SIMD baseline):
  double + double_scalar                      15.29 ms  [C# SIMD]
  float + float_scalar                         8.35 ms  [C# SIMD]

Mixed-type with IL SIMD (LHS type == Result type):
  double + int_scalar                         14.96 ms  [IL SIMD ✓]  <- NOW OPTIMIZED
  float + int_scalar                           7.18 ms  [IL SIMD ✓]  <- NOW OPTIMIZED

Mixed-type without SIMD (requires conversion):
  int + double_scalar                         15.84 ms  [Scalar loop]
```

**Tests:** All 2597 tests pass, 0 failures.

---

## Hardware Detection Results

| Feature | Supported |
|---------|-----------|
| SSE | Yes |
| SSE2 | Yes |
| SSE3 | Yes |
| SSSE3 | Yes |
| SSE4.1 | Yes |
| SSE4.2 | Yes |
| AVX | Yes |
| AVX2 | Yes |
| **AVX-512** | **No** |
| Vector256 | Yes (hardware accelerated) |
| Vector512 | No |

**Conclusion**: This machine (and most consumer CPUs) only supports up to AVX2/Vector256. AVX-512 hardware detection should be added but has lower priority since adoption is limited.

---

## Scalar SIMD Benchmark Results

```
Benchmark: array[10,000,000] + scalar

1. Scalar Loop           :    25.42 ms
2. SIMD Hoisted          :    16.28 ms  (1.56x faster)
3. SIMD In-Loop          :    22.42 ms  (JIT doesn't fully hoist)
```

**Key Findings:**
- SIMD with hoisted `Vector256.Create(scalar)` is **1.56x faster** than scalar loop
- JIT does NOT fully hoist `Vector256.Create` - explicit hoisting gains another **1.38x**
- Explicit hoisting before the loop is critical for performance

---

## NumSharp Current State Analysis

### Execution Path Dispatch

```
Operation Type    | Path Classification | Kernel Used          | SIMD Scalar?
------------------|---------------------|----------------------|-------------
double + double   | SimdScalarRight     | C# SimdKernels       | YES (optimal)
int + double      | SimdScalarRight     | IL MixedTypeKernel   | NO (scalar loop)
int + int         | SimdScalarRight     | C# SimdKernels       | YES (for int/double/float/long)
byte + float      | SimdScalarRight     | IL MixedTypeKernel   | NO (scalar loop)
```

### Performance Comparison

```
Benchmark: array[10,000,000] + scalar

Same-type (double+double): 14.26 ms  (C# SIMD kernel)
Mixed-type (int+double):   18.07 ms  (IL scalar kernel)

Performance gap: ~27%
```

### Code Analysis

**C# SimdKernels.cs (lines 217-231)** - Optimal implementation:
```csharp
private static unsafe void SimdScalarRight_Add_Double(double* lhs, double scalar, double* result, int totalSize)
{
    var scalarVec = Vector256.Create(scalar);  // Hoisted!
    int i = 0;
    int vectorEnd = totalSize - Vector256<double>.Count;

    for (; i <= vectorEnd; i += Vector256<double>.Count)
    {
        var vl = Vector256.Load(lhs + i);
        Vector256.Store(vl + scalarVec, result + i);  // SIMD!
    }

    for (; i < totalSize; i++)
        result[i] = lhs[i] + scalar;  // Remainder
}
```

**ILKernelGenerator.cs (lines 912-970)** - Suboptimal implementation:
```csharp
private static void EmitScalarRightLoop(ILGenerator il, MixedTypeKernelKey key, ...)
{
    // Line 916-925: Hoist scalar value to local (good!)
    var locRhsVal = il.DeclareLocal(GetClrType(key.ResultType));
    il.Emit(OpCodes.Ldarg_1); // rhs
    EmitLoadIndirect(il, key.RhsType);
    EmitConvertTo(il, key.RhsType, key.ResultType);
    il.Emit(OpCodes.Stloc, locRhsVal);

    // Lines 938-960: Scalar operations only, NO SIMD!
    for (int i = 0; i < totalSize; i++)
    {
        result[i] = lhs[i] + rhsVal;  // Scalar add
    }
}
```

---

## Recommendations

### Priority 1: Add SIMD to IL Scalar Paths (HIGH IMPACT)

**Why**: 27% speedup for mixed-type scalar operations.

**Implementation**:
1. Modify `EmitScalarRightLoop()` to emit SIMD code for supported types
2. Hoist `Vector256.Create(scalar)` before the loop
3. Add Vector256 load/add/store in the main loop
4. Keep scalar remainder loop for sizes not divisible by vector count

**Target types**: float, double (already have Vector256 support)

**Files to modify**:
- `ILKernelGenerator.cs`: Add `EmitSimdScalarRightLoop()` method
- Update `GenerateSimdScalarRightKernel()` to choose SIMD vs scalar based on type

### Priority 2: Hardware Detection (LOW PRIORITY)

**Why**: AVX-512 adoption is limited. Most CPUs (including this dev machine) only support AVX2.

**Implementation** (when AVX-512 becomes common):
1. Add static readonly flags in `SimdThresholds.cs`:
   ```csharp
   public static readonly bool HasAvx512 = Vector512.IsHardwareAccelerated;
   public static readonly int PreferredVectorWidth = HasAvx512 ? 512 : 256;
   ```
2. Add Vector512 code paths alongside Vector256
3. Use runtime dispatch based on `HasAvx512`

**Expected benefit**: 2x throughput on AVX-512 hardware (16 floats vs 8 floats per instruction)

---

## Implementation Checklist

### Phase 1: SIMD Scalar for IL Kernels ✅ COMPLETE

- [x] Add `EmitSimdScalarRightLoop()` for float/double
- [x] Add `EmitSimdScalarLeftLoop()` for float/double
- [x] Add `EmitVectorCreate()` helper for Vector256.Create(scalar)
- [x] Update `GenerateSimdScalarRightKernel()` to choose SIMD path
- [x] Update `GenerateSimdScalarLeftKernel()` to choose SIMD path
- [x] Verify correctness with small arrays
- [x] Run full test suite (2597 passed, 0 failed)
- [x] Benchmark before/after

### Phase 2: Hardware Detection (Defer)

- [ ] Add `SimdCapabilities` static class
- [ ] Cache detection results at startup
- [ ] Add Vector512 code paths (when adopting)
- [ ] Runtime dispatch mechanism

---

## Files Modified

- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs`:
  - Added `EmitSimdScalarRightLoop()` method (lines 1063-1178)
  - Added `EmitSimdScalarLeftLoop()` method (lines 1180-1295)
  - Added `EmitVectorCreate()` helper (lines 1900-1914)
  - Updated `GenerateSimdScalarRightKernel()` to check SIMD eligibility
  - Updated `GenerateSimdScalarLeftKernel()` to check SIMD eligibility

---

## Appendix: Raw Benchmark Data

### Test 1: Hardware Detection
```
X86 Intrinsics:
  Sse:        True
  Sse2:       True
  Avx:        True
  Avx2:       True
  Avx512F:    False

Generic Vector Types:
  Vector256<float>:  True
  Vector512<float>:  False
```

### Test 2: Scalar vs SIMD
```
array[10,000,000] + scalar

1. Scalar Loop           :    25.42 ms
2. SIMD Hoisted          :    16.28 ms
3. SIMD In-Loop          :    22.42 ms
```

### Test 3: NumSharp Same-type vs Mixed-type
```
Same-type (double+double): 14.26 ms
Mixed-type (int+double):   18.07 ms
```

---

## Conclusion

The investigation confirmed:
1. **Scalar SIMD** with hoisted broadcast provides **1.56x speedup** over scalar loops
2. NumSharp's C# SimdKernels already implement this optimally for same-type operations
3. ~~**IL MixedTypeKernels lack SIMD for scalar paths**~~ **FIXED** ✅
4. AVX-512 hardware detection is low priority due to limited adoption

**Status**: SIMD scalar paths have been implemented for IL kernels. Mixed-type operations like `double_array + int_scalar` now use SIMD when the array type equals the result type.

**Remaining work**: Hardware detection for AVX-512 (deferred until adoption increases).
