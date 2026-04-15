# New Dtypes Implementation Status

This document tracks the implementation of three new NumPy-compatible data types in NumSharp:
- **SByte** (int8) - `NPTypeCode.SByte = 5`
- **Half** (float16) - `NPTypeCode.Half = 16`
- **Complex** (complex128) - `NPTypeCode.Complex = 128`

## Completed Work

### Core Type System (✓ Complete)

| File | Status | Notes |
|------|--------|-------|
| `NPTypeCode.cs` | ✓ | Added enum values, updated all extension methods |
| `InfoOf.cs` | ✓ | Added Size cases for new types |
| `NumberInfo.cs` | ✓ | Added MaxValue/MinValue for new types |
| `np.dtype.cs` | ✓ | Added kind mapping and dtype string parsing |

### Memory Management (✓ Complete)

| File | Status | Notes |
|------|--------|-------|
| `UnmanagedMemoryBlock.cs` | ✓ | Added FromArray and Allocate cases |
| `ArraySlice.cs` | ✓ | Added all Scalar and Allocate cases |
| `UnmanagedStorage.cs` | ✓ | Added typed fields and SetInternalArray cases |

### Updated NPTypeCode Extension Methods

All extension methods in `NPTypeCode.cs` have been updated:
- `GetTypeCode(Type)` - Handles `Half` type
- `AsType()` - Returns correct Type for new codes
- `SizeOf()` - Returns 1/2/16 for SByte/Half/Complex
- `IsRealNumber()` - Half and Complex return true
- `IsUnsigned()` - SByte returns false
- `IsSigned()` - SByte and Half return true
- `GetGroup()` - SByte in group 1, Half in group 3, Complex in group 10
- `GetPriority()` - Correct priority for type promotion
- `ToTypeCode()` / `ToTYPECHAR()` - NPY_TYPECHAR conversions
- `AsNumpyDtypeName()` - Returns "int8", "float16", "complex128"
- `GetAccumulatingType()` - Returns appropriate accumulator types
- `GetDefaultValue()` - Returns default for each type
- `GetOneValue()` - Returns multiplicative identity (1)
- `IsFloatingPoint()` - Half returns true
- `IsInteger()` - SByte returns true
- `IsSimdCapable()` - SByte true, Half false, Complex false
- `IsNumerical()` - All three return true

## Remaining Work

### Files Needing Switch Statement Updates

The following files have switch statements that handle NPTypeCode but don't yet include the new types.
These will throw `NotSupportedException` at runtime when using new types:

#### High Priority (Core Functionality)
- `Backends/Unmanaged/UnmanagedStorage.Getters.cs`
- `Backends/Unmanaged/UnmanagedStorage.Setters.cs`
- `Backends/Unmanaged/UnmanagedStorage.Cloning.cs`
- `Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs`
- `Backends/NDArray.cs`

#### Iterators
- `Backends/Iterators/NDIterator.cs`
- `Backends/Iterators/NDIteratorExtensions.cs`
- `Backends/Iterators/MultiIterator.cs`

#### DefaultEngine Operations
- `Backends/Default/ArrayManipulation/Default.NDArray.cs`
- `Backends/Default/Indexing/Default.BooleanMask.cs`
- `Backends/Default/Indexing/Default.NonZero.cs`
- `Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs`
- `Backends/Default/Math/Default.Clip.cs`
- `Backends/Default/Math/Default.ClipNDArray.cs`
- `Backends/Default/Math/Default.Shift.cs`
- `Backends/Default/Math/Reduction/Default.Reduction.CumAdd.cs`
- `Backends/Default/Math/Reduction/Default.Reduction.CumMul.cs`
- `Backends/Default/Math/Reduction/Default.Reduction.Std.cs`
- `Backends/Default/Math/Reduction/Default.Reduction.Var.cs`

#### ILKernelGenerator (Performance Critical)
- `Backends/Kernels/ILKernelGenerator.cs`
- `Backends/Kernels/ILKernelGenerator.Reduction.cs`
- `Backends/Kernels/ILKernelGenerator.Reduction.Axis.cs`
- `Backends/Kernels/ILKernelGenerator.Unary.Math.cs`

#### Creation APIs
- `APIs/np.fromfile.cs`
- `Creation/np.arange.cs`
- `Creation/np.frombuffer.cs`
- `Creation/np.linspace.cs`

#### Other
- `Casting/Implicit/NdArray.Implicit.Array.cs`
- `Manipulation/NDArray.unique.cs`

## Special Considerations

### Half Type
- `System.Half` doesn't implement `IConvertible`, so conversion methods need special handling
- SIMD support is limited - marked as not SIMD-capable
- Conversions go through `double` intermediate: `(Half)Convert.ToDouble(value)`

### Complex Type
- `System.Numerics.Complex` doesn't implement `IConvertible`
- Complex uses 16 bytes (two 64-bit doubles)
- Many math operations may need special handling for complex arithmetic
- Already had `NPTypeCode.Complex = 128` defined, but wasn't implemented in most switches

### SByte Type
- Straightforward to implement - same pattern as `byte`
- Full SIMD support
- Maps to NumPy's `int8` / `np.int8`

## Testing

Basic tests are in `test/NumSharp.UnitTest/NewDtypes/NewDtypesBasicTests.cs`:
- Array creation with new types
- `np.zeros` with new type codes
- NPTypeCode property verification
- dtype string parsing

## Migration Guide

To add support for a new type to an existing switch statement:

```csharp
// Pattern for SByte
case NPTypeCode.SByte:
{
    // Use sbyte type
    break;
}

// Pattern for Half
case NPTypeCode.Half:
{
    // Use Half type
    // Note: No IConvertible support
    break;
}

// Pattern for Complex
case NPTypeCode.Complex:
{
    // Use System.Numerics.Complex type
    // Note: No IConvertible support
    break;
}
```

## Build Status

The project builds successfully with all changes. Runtime support depends on which operations are used.
