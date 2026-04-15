# New Dtypes Implementation - Developer Handoff

## Overview

This document provides guidance for completing the remaining work on the new dtype implementation (SByte/int8, Half/float16, Complex/complex128). The core implementation is complete and functional, but 6 files remain that need updates for full coverage.

## Current State

**Build Status:** ✅ Passes
**Runtime Status:** ✅ Functional for basic operations
**Test Verification:** ✅ Array creation, zeros, dtype parsing all work

The new types work correctly for most operations. However, certain performance-critical paths and type conversion utilities still have incomplete switch statements that will throw `NotSupportedException` when hit.

---

## Files Requiring Updates

### 1. `Utilities/Converts.cs` (HIGH PRIORITY)

**Why it matters:** This file contains type conversion logic used throughout NumSharp. When you call `.astype()`, cast between types, or perform mixed-type arithmetic, this code is invoked.

**What's missing:** The `ChangeType<T>` and related methods have switch statements that don't include SByte, Half, or Complex.

**Pattern to follow:**
```csharp
// Find switches like this:
case NPTypeCode.Byte:
    return Converts.ToByte(Unsafe.As<T, byte>(ref value));

// Add after Byte:
case NPTypeCode.SByte:
    return Converts.ToSByte(Unsafe.As<T, sbyte>(ref value));

// For Half (no IConvertible):
case NPTypeCode.Half:
    return (Half)Convert.ToDouble(Unsafe.As<T, Half>(ref value));

// For Complex (no IConvertible):
case NPTypeCode.Complex:
    return Unsafe.As<T, System.Numerics.Complex>(ref value);
```

**Gotcha:** Half and Complex don't implement `IConvertible`, so you can't use `Convert.ToXxx()` directly. For Half, cast through double. For Complex, direct reinterpret or construct from real part.

**Discovery command:**
```bash
grep -n "case NPTypeCode.Byte:" Utilities/Converts.cs | head -20
```

---

### 2. `Utilities/ArrayConvert.cs` (HIGH PRIORITY)

**Why it matters:** Handles array-to-array type conversions. Used when converting entire arrays between dtypes.

**What's missing:** Switch statements for bulk array conversion don't include new types.

**Pattern:** Same as Converts.cs - find Byte cases, add SByte/Half/Complex after them.

---

### 3. `Backends/Kernels/ILKernelGenerator.cs` (MEDIUM PRIORITY)

**Why it matters:** This is the core IL code generation infrastructure. It contains type mappings that tell the IL emitter what opcodes to use for each type.

**What's missing:** Type-to-IL mappings for SByte, Half, Complex.

**What happens without it:** Operations fall back to slower iterator-based paths instead of SIMD-optimized kernels.

**Key areas to update:**

1. **Type size mapping:**
```csharp
// Look for patterns like:
typeof(byte) => 1,
// Add:
typeof(sbyte) => 1,
typeof(Half) => 2,
typeof(System.Numerics.Complex) => 16,
```

2. **SIMD capability:**
```csharp
// SByte IS SIMD capable (same as byte)
// Half is NOT SIMD capable (no Vector<Half> support)
// Complex is NOT SIMD capable (16 bytes, complex arithmetic)
```

3. **Load/Store opcodes:**
```csharp
// SByte uses Ldind_I1 / Stind_I1
// Half uses Ldind_I2 / Stind_I2 (but treated as non-SIMD)
// Complex uses custom 16-byte load/store
```

---

### 4. `Backends/Kernels/ILKernelGenerator.Reduction.cs` (MEDIUM PRIORITY)

**Why it matters:** Generates IL kernels for reduction operations (sum, prod, min, max, mean).

**What's missing:** Type dispatch for new types in reduction kernel generation.

**Pattern:**
```csharp
// Find:
case NPTypeCode.Byte: return GenerateReductionKernel<byte>(...);

// Add:
case NPTypeCode.SByte: return GenerateReductionKernel<sbyte>(...);
case NPTypeCode.Half: return null; // Fall back to iterator path
case NPTypeCode.Complex: return null; // Fall back to iterator path
```

**Note:** For Half and Complex, returning `null` from the kernel generator causes the caller to use the iterator-based fallback, which works correctly but is slower.

---

### 5. `Backends/Kernels/ILKernelGenerator.Reduction.Axis.cs` (MEDIUM PRIORITY)

**Why it matters:** Generates IL kernels for axis-based reductions (e.g., `np.sum(arr, axis=0)`).

**Same pattern as ILKernelGenerator.Reduction.cs** - add SByte cases, return null for Half/Complex.

---

### 6. `Backends/Kernels/ILKernelGenerator.Unary.Math.cs` (LOW PRIORITY)

**Why it matters:** Generates IL for unary math operations (abs, sqrt, exp, log, sin, cos, etc.).

**What's missing:** Type dispatch for new types.

**Special considerations:**

- **SByte:** Most math operations should work (abs, sign, etc.)
- **Half:** Math operations need to go through double: `(Half)Math.Sqrt((double)value)`
- **Complex:** Has dedicated `Complex.Sqrt()`, `Complex.Exp()`, etc. in `System.Numerics`

**Pattern:**
```csharp
// For Half - emit conversion to double, call Math.*, convert back
// For Complex - emit call to System.Numerics.Complex static methods
```

---

## Type-Specific Considerations

### SByte (int8)
- **Difficulty:** Easy
- **Pattern:** Copy byte cases, change type name
- **SIMD:** Yes, fully supported
- **IConvertible:** Yes
- **Math operations:** Standard integer math

### Half (float16)
- **Difficulty:** Medium
- **Pattern:** Copy float/Single cases, but handle conversion through double
- **SIMD:** No - `Vector<Half>` doesn't exist in .NET
- **IConvertible:** No - must cast through double
- **Math operations:** Convert to double, compute, convert back
- **Special values:** Has NaN, Infinity, works like float

### Complex (complex128)
- **Difficulty:** Hard
- **Pattern:** Unique - not similar to other types
- **SIMD:** No - 16 bytes, complex arithmetic semantics
- **IConvertible:** No
- **Math operations:** Use `System.Numerics.Complex` static methods
- **Comparison:** Not supported (complex numbers aren't orderable)
- **Excluded from:** `unique()`, `clip()`, `shift operations`, `randint`

---

## Testing Strategy

### Quick Smoke Test
```bash
cd K:/source/NumSharp/.claude/worktrees/half
dotnet_run <<'EOF'
#:project K:/source/NumSharp/.claude/worktrees/half/src/NumSharp.Core
#:property PublishAot=false

using NumSharp;
using NumSharp.Backends;

// Test the operation you just fixed
var arr = np.array(new sbyte[] { 1, 2, 3 });
var result = np.sum(arr);  // or whatever operation
Console.WriteLine($"Result: {result}");
EOF
```

### Finding Missing Cases
```bash
cd src/NumSharp.Core

# Find files with Byte but missing SByte
grep -l "case NPTypeCode.Byte:" --include="*.cs" -r | while read f; do
  grep -q "case NPTypeCode.SByte:" "$f" || echo "$f"
done
```

### Verification After Changes
```bash
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

---

## Common Pitfalls

### 1. Half Conversion
```csharp
// WRONG - Half doesn't implement IConvertible
Converts.ToSingle(halfValue)  // Throws!

// CORRECT
(float)(double)halfValue
// or
(float)Convert.ToDouble(halfValue)  // Also throws!

// ACTUALLY CORRECT
(float)(Half)value  // Direct cast works
```

### 2. Complex Comparison
```csharp
// WRONG - Complex doesn't implement IComparable
if (c1 < c2)  // Compile error!

// Complex numbers cannot be ordered
// Skip Complex in: unique(), clip(), argmin(), argmax(), sort()
```

### 3. Complex Arithmetic vs Real
```csharp
// Complex + real number
Complex c = new Complex(1, 2);
double d = 3.0;
Complex result = c + d;  // Works - implicit conversion

// But for type switches, handle separately
case NPTypeCode.Complex:
    // Use System.Numerics.Complex operations
```

### 4. Switch Fall-Through
```csharp
// Don't forget the break!
case NPTypeCode.SByte:
    DoSomething<sbyte>();
    break;  // <-- Don't forget this!
case NPTypeCode.Int16:
```

---

## Definition of Done

1. **Build passes:** `dotnet build` succeeds with no errors
2. **Grep check:** Running the discovery command returns no files
3. **Smoke tests pass:** Basic operations work for all three types
4. **No NotSupportedException:** Using new types doesn't throw in common paths

---

## Priority Order

1. **Converts.cs** - Unlocks type conversion, highest impact
2. **ArrayConvert.cs** - Unlocks array conversion
3. **ILKernelGenerator.cs** - Core type mapping
4. **ILKernelGenerator.Reduction.cs** - Sum/prod/min/max performance
5. **ILKernelGenerator.Reduction.Axis.cs** - Axis reduction performance
6. **ILKernelGenerator.Unary.Math.cs** - Math function performance

---

## Questions?

If you encounter issues:
1. Check if Half/Complex need special handling (they usually do)
2. Verify the operation makes sense for the type (e.g., no Complex comparison)
3. Return `null` from IL kernel generators to fall back to iterator path
4. Test with a simple script before running full test suite
