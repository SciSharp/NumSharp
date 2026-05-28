# NpyIter Implementation Plan

**Status:** Design Phase
**Target:** 100% NumPy nditer parity + NumSharp IL optimization integration
**Reference:** `numpy/_core/src/multiarray/nditer_impl.h`, `nditer_constr.c`, `nditer_api.c`

---

## Table of Contents

1. [Overview](#overview)
2. [NumPy nditer Analysis](#numpy-nditer-analysis)
3. [Architecture Design](#architecture-design)
4. [Data Structures](#data-structures)
5. [Flags and Enumerations](#flags-and-enumerations)
6. [Core Operations](#core-operations)
7. [Execution Paths](#execution-paths)
8. [IL Kernel Integration](#il-kernel-integration)
9. [Buffering System](#buffering-system)
10. [Axis Coalescing](#axis-coalescing)
11. [API Surface](#api-surface)
12. [Implementation Phases](#implementation-phases)
13. [Testing Strategy](#testing-strategy)
14. [Performance Targets](#performance-targets)

---

## Overview

### Purpose

NpyIter is the core iteration infrastructure for multi-operand array operations. It handles:
- Synchronized iteration over multiple arrays with different shapes/strides
- Broadcasting alignment
- Memory layout optimization (buffering, coalescing)
- Type casting during iteration
- Reduction axis handling

### Scope

This document covers the **complete** NpyIter implementation matching NumPy's capabilities:

| Component | NumPy | NumSharp Target |
|-----------|-------|-----------------|
| Single-operand iteration | `NpyIter_New` | `NpyIter.New` |
| Multi-operand iteration | `NpyIter_MultiNew` | `NpyIter.MultiNew` |
| Advanced iteration | `NpyIter_AdvancedNew` | `NpyIter.AdvancedNew` |
| Buffered iteration | `NPY_ITER_BUFFERED` | Full support |
| External loop | `NPY_ITER_EXTERNAL_LOOP` | Full support |
| Axis coalescing | `npyiter_coalesce_axes` | Full support |
| Type casting | `NPY_cast_info` | Via IL kernels |
| Reduction support | `NPY_ITER_REDUCE_OK` | Full support |

### Non-Goals

- Python-specific features (pickle, `__array_wrap__`)
- Object dtype iteration (NumSharp doesn't support object arrays)
- Fortran-order preference (NumSharp is C-order only)

---

## NumPy nditer Analysis

### Source Files

| File | Purpose | Lines |
|------|---------|-------|
| `nditer_impl.h` | Internal structures, macros, flags | ~400 |
| `nditer_constr.c` | Construction, validation, setup | ~2000 |
| `nditer_api.c` | Public API, iteration, buffer management | ~1800 |
| `nditer_templ.c.src` | Templated iteration functions | ~500 |
| `nditer_pywrap.c` | Python wrapper | ~1200 |

### Core Data Structure (NumPy)

```c
struct NpyIter_InternalOnly {
    npy_uint32 itflags;           // Iterator flags
    npy_uint8 ndim;               // Number of dimensions (after coalescing)
    int nop, maskop;              // Number of operands, mask operand index
    npy_intp itersize;            // Total iteration count
    npy_intp iterstart, iterend;  // Range for ranged iteration
    npy_intp iterindex;           // Current iteration index
    char iter_flexdata[];         // Variable-length data (see below)
};

// iter_flexdata layout:
// - perm[NPY_MAXDIMS]           : Axis permutation
// - dtypes[nop]                 : Operand dtypes
// - resetdataptr[nop+1]         : Reset data pointers
// - baseoffsets[nop+1]          : Base offsets
// - operands[nop]               : Operand array references
// - opitflags[nop]              : Per-operand flags
// - bufferdata (if buffered)    : Buffer management
// - dataptrs[nop+1]             : Current data pointers
// - userptrs[nop+1]             : User-visible pointers
// - axisdata[ndim]              : Per-axis data (shape, index, strides)
```

### Per-Axis Data (NumPy)

```c
struct NpyIter_AxisData_tag {
    npy_intp shape;               // Size of this axis
    npy_intp index;               // Current index along this axis
    Py_intptr_t ad_flexdata;      // Strides for each operand
};
```

### Key Functions (NumPy)

| Function | Purpose |
|----------|---------|
| `NpyIter_AdvancedNew` | Full constructor with all options |
| `NpyIter_MultiNew` | Simplified multi-operand constructor |
| `NpyIter_New` | Single-operand constructor |
| `NpyIter_GetIterNext` | Get iteration function pointer |
| `NpyIter_GetDataPtrArray` | Get current data pointers |
| `NpyIter_GetInnerStrideArray` | Get inner loop strides |
| `NpyIter_GetInnerLoopSizePtr` | Get inner loop size |
| `NpyIter_Reset` | Reset to beginning |
| `NpyIter_GotoIterIndex` | Jump to specific index |
| `NpyIter_RemoveAxis` | Remove axis from iteration |
| `NpyIter_EnableExternalLoop` | Enable external loop handling |
| `npyiter_coalesce_axes` | Merge compatible axes |
| `npyiter_copy_to_buffers` | Fill buffers from operands |
| `npyiter_copy_from_buffers` | Flush buffers to operands |

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Public API                                      │
│  NpyIter.New() / NpyIter.MultiNew() / NpyIter.AdvancedNew()                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NpyIter (ref struct)                               │
│                                                                              │
│  Properties:                                                                 │
│    - NpyIterState* State        // Pointer to state struct                  │
│    - bool IsValid               // Whether iterator is valid                 │
│    - int NDim                   // Dimensions after coalescing              │
│    - int NOp                    // Number of operands                        │
│    - long IterSize              // Total iterations                          │
│                                                                              │
│  Methods:                                                                    │
│    - GetIterNext()              // Returns NpyIterNextFunc delegate         │
│    - GetDataPtrArray()          // Returns void** to current pointers       │
│    - GetInnerStrideArray()      // Returns long* to inner strides           │
│    - GetInnerLoopSizePtr()      // Returns long* to inner size              │
│    - Reset()                    // Reset to beginning                        │
│    - GotoIterIndex(index)       // Jump to index                            │
│    - RemoveAxis(axis)           // Remove axis, enable coalescing           │
│    - RemoveMultiIndex()         // Drop multi-index tracking                │
│    - EnableExternalLoop()       // Caller handles inner loop                │
│    - Dispose()                  // Free resources                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
         ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
         │  NpyIterState    │ │ NpyIterAxis  │ │ NpyIterBuffer    │
         │  (fixed struct)  │ │ (per-axis)   │ │ (if buffered)    │
         └──────────────────┘ └──────────────┘ └──────────────────┘
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────┐
         │                  Execution Paths                        │
         ├────────────────────────────────────────────────────────┤
         │  Contiguous    │  Buffered     │  Strided    │ General │
         │  ───────────   │  ────────     │  ───────    │ ─────── │
         │  Direct SIMD   │  Copy→Buffer  │  Gather     │ Coords  │
         │  IL Kernels    │  SIMD on buf  │  SIMD       │ Loop    │
         │                │  Buffer→Copy  │             │         │
         └────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Zero Allocation Hot Path**: State structs use fixed-size buffers, no heap allocation during iteration
2. **Stack Allocation**: `NpyIterState` is a struct that can live on stack for small operand counts
3. **IL Kernel Integration**: Seamless handoff to `ILKernelGenerator` for optimized inner loops
4. **NumPy API Parity**: Method names and semantics match NumPy exactly
5. **Execution Path Detection**: Automatically select optimal path based on operand layout

---

## Data Structures

### NpyIterState

The core state structure, designed for stack allocation with fixed-size buffers.

```csharp
/// <summary>
/// Core iterator state. Stack-allocated with fixed-size buffers.
/// Matches NumPy's NpyIter_InternalOnly layout conceptually.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct NpyIterState
{
    // =========================================================================
    // Constants
    // =========================================================================

    /// <summary>Maximum supported dimensions (matches NPY_MAXDIMS).</summary>
    public const int MaxDims = 32;

    /// <summary>Maximum supported operands.</summary>
    public const int MaxOperands = 8;

    // =========================================================================
    // Core Fields (fixed size: 32 bytes)
    // =========================================================================

    /// <summary>Iterator flags (NpyIterFlags bitmask).</summary>
    public uint ItFlags;

    /// <summary>Number of dimensions after coalescing.</summary>
    public int NDim;

    /// <summary>Number of operands.</summary>
    public int NOp;

    /// <summary>Mask operand index (-1 if none).</summary>
    public int MaskOp;

    /// <summary>Total number of iterations.</summary>
    public long IterSize;

    /// <summary>Current iteration index.</summary>
    public long IterIndex;

    // =========================================================================
    // Fixed Arrays (stack-allocated)
    // =========================================================================

    /// <summary>Axis permutation (maps iterator axis to original axis).</summary>
    public fixed sbyte Perm[MaxDims];

    /// <summary>Shape after coalescing.</summary>
    public fixed long Shape[MaxDims];

    /// <summary>Current coordinates.</summary>
    public fixed long Coords[MaxDims];

    /// <summary>
    /// Strides for each operand along each axis.
    /// Layout: [axis0_op0, axis0_op1, ..., axis1_op0, axis1_op1, ...]
    /// Access: Strides[axis * NOp + opIndex]
    /// </summary>
    public fixed long Strides[MaxDims * MaxOperands];

    /// <summary>Current data pointers for each operand.</summary>
    public fixed long DataPtrs[MaxOperands];  // IntPtr stored as long

    /// <summary>Reset data pointers (base + offset).</summary>
    public fixed long ResetDataPtrs[MaxOperands];

    /// <summary>Base offsets for each operand.</summary>
    public fixed long BaseOffsets[MaxOperands];

    /// <summary>Per-operand flags.</summary>
    public fixed ushort OpItFlags[MaxOperands];

    /// <summary>Operand dtypes.</summary>
    public fixed byte OpDTypes[MaxOperands];  // NPTypeCode as byte

    // =========================================================================
    // Buffer Data (when BUFFERED flag is set)
    // =========================================================================

    /// <summary>Buffer size (elements per buffer).</summary>
    public long BufferSize;

    /// <summary>Current buffer fill size.</summary>
    public long BufIterEnd;

    /// <summary>Buffer pointers for each operand.</summary>
    public fixed long Buffers[MaxOperands];  // IntPtr stored as long

    /// <summary>Buffer strides (always 1 for contiguous buffers).</summary>
    public fixed long BufStrides[MaxOperands];

    // =========================================================================
    // Accessor Methods
    // =========================================================================

    /// <summary>Get pointer to Shape array.</summary>
    public long* GetShapePtr()
    {
        fixed (long* p = Shape) return p;
    }

    /// <summary>Get pointer to Coords array.</summary>
    public long* GetCoordsPtr()
    {
        fixed (long* p = Coords) return p;
    }

    /// <summary>Get stride for operand at axis.</summary>
    public long GetStride(int axis, int op)
    {
        fixed (long* p = Strides) return p[axis * NOp + op];
    }

    /// <summary>Set stride for operand at axis.</summary>
    public void SetStride(int axis, int op, long value)
    {
        fixed (long* p = Strides) p[axis * NOp + op] = value;
    }

    /// <summary>Get current data pointer for operand.</summary>
    public void* GetDataPtr(int op)
    {
        fixed (long* p = DataPtrs) return (void*)p[op];
    }

    /// <summary>Set current data pointer for operand.</summary>
    public void SetDataPtr(int op, void* ptr)
    {
        fixed (long* p = DataPtrs) p[op] = (long)ptr;
    }

    /// <summary>Get operand dtype.</summary>
    public NPTypeCode GetOpDType(int op)
    {
        fixed (byte* p = OpDTypes) return (NPTypeCode)p[op];
    }

    /// <summary>Get operand flags.</summary>
    public NpyIterOpFlags GetOpFlags(int op)
    {
        fixed (ushort* p = OpItFlags) return (NpyIterOpFlags)p[op];
    }
}
```

### NpyIterAxisData

Per-axis data for multi-index tracking.

```csharp
/// <summary>
/// Per-axis iteration data.
/// Used when multi-index tracking is enabled.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct NpyIterAxisData
{
    /// <summary>Size of this axis.</summary>
    public long Shape;

    /// <summary>Current index along this axis.</summary>
    public long Index;

    /// <summary>
    /// Strides for each operand along this axis.
    /// Inline array, actual size depends on NOp.
    /// </summary>
    public fixed long Strides[NpyIterState.MaxOperands];
}
```

### NpyIterBufferData

Buffer management for non-contiguous operands.

```csharp
/// <summary>
/// Buffer management data for buffered iteration.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct NpyIterBufferData
{
    /// <summary>Buffer size in elements.</summary>
    public long BufferSize;

    /// <summary>Current fill size.</summary>
    public long Size;

    /// <summary>End of buffer iteration.</summary>
    public long BufIterEnd;

    /// <summary>Reduce position (for reduction operations).</summary>
    public long ReducePos;

    /// <summary>Core size (for external loop).</summary>
    public long CoreSize;

    /// <summary>Outer size (for external loop).</summary>
    public long OuterSize;

    /// <summary>Core offset.</summary>
    public long CoreOffset;

    /// <summary>Outer dimension index.</summary>
    public long OuterDim;

    /// <summary>Buffer strides per operand.</summary>
    public fixed long Strides[NpyIterState.MaxOperands];

    /// <summary>Outer strides for reduce.</summary>
    public fixed long ReduceOuterStrides[NpyIterState.MaxOperands];

    /// <summary>Outer pointers for reduce.</summary>
    public fixed long ReduceOuterPtrs[NpyIterState.MaxOperands];

    /// <summary>Buffer pointers per operand.</summary>
    public fixed long Buffers[NpyIterState.MaxOperands];
}
```

---

## Flags and Enumerations

### NpyIterFlags (Iterator Flags)

```csharp
/// <summary>
/// Iterator-level flags. Matches NumPy's NPY_ITFLAG_* constants.
/// </summary>
[Flags]
public enum NpyIterFlags : uint
{
    None = 0,

    // =========================================================================
    // Permutation Flags
    // =========================================================================

    /// <summary>The axis permutation is identity.</summary>
    IDENTPERM = 0x0001,

    /// <summary>The permutation has negative entries (flipped axes).</summary>
    NEGPERM = 0x0002,

    // =========================================================================
    // Index Tracking Flags
    // =========================================================================

    /// <summary>Iterator is tracking a flat index.</summary>
    HASINDEX = 0x0004,

    /// <summary>Iterator is tracking a multi-index.</summary>
    HASMULTIINDEX = 0x0008,

    // =========================================================================
    // Order and Loop Flags
    // =========================================================================

    /// <summary>Iteration order was forced on construction.</summary>
    FORCEDORDER = 0x0010,

    /// <summary>Inner loop is handled outside the iterator.</summary>
    EXLOOP = 0x0020,

    /// <summary>Iterator is ranged (subset iteration).</summary>
    RANGE = 0x0040,

    // =========================================================================
    // Buffering Flags
    // =========================================================================

    /// <summary>Iterator uses buffering.</summary>
    BUFFER = 0x0080,

    /// <summary>Grow the buffered inner loop when possible.</summary>
    GROWINNER = 0x0100,

    /// <summary>Single iteration, can specialize iternext.</summary>
    ONEITERATION = 0x0200,

    /// <summary>Delay buffer allocation until first Reset.</summary>
    DELAYBUF = 0x0400,

    // =========================================================================
    // Reduction Flags
    // =========================================================================

    /// <summary>Iteration includes reduction operands.</summary>
    REDUCE = 0x0800,

    /// <summary>Reduce loops don't need recalculation.</summary>
    REUSE_REDUCE_LOOPS = 0x1000,

    // =========================================================================
    // NumSharp Extensions (above NumPy's range)
    // =========================================================================

    /// <summary>All operands are contiguous (SIMD eligible).</summary>
    CONTIGUOUS = 0x00010000,

    /// <summary>Can use AVX2 gather for strided access.</summary>
    GATHER_ELIGIBLE = 0x00020000,

    /// <summary>Operation supports early exit (boolean ops).</summary>
    EARLY_EXIT = 0x00040000,

    /// <summary>Parallel outer loop is safe.</summary>
    PARALLEL_SAFE = 0x00080000,
}
```

### NpyIterOpFlags (Per-Operand Flags)

```csharp
/// <summary>
/// Per-operand flags. Matches NumPy's NPY_OP_ITFLAG_* constants.
/// </summary>
[Flags]
public enum NpyIterOpFlags : ushort
{
    None = 0,

    // =========================================================================
    // Read/Write Flags
    // =========================================================================

    /// <summary>Operand will be written to.</summary>
    WRITE = 0x0001,

    /// <summary>Operand will be read from.</summary>
    READ = 0x0002,

    /// <summary>Operand is read-write.</summary>
    READWRITE = READ | WRITE,

    // =========================================================================
    // Buffering Flags
    // =========================================================================

    /// <summary>Operand needs type conversion/byte swapping/alignment.</summary>
    CAST = 0x0004,

    /// <summary>Operand never needs buffering.</summary>
    BUFNEVER = 0x0008,

    /// <summary>Buffer filling can use single stride.</summary>
    BUF_SINGLESTRIDE = 0x0010,

    // =========================================================================
    // Reduction Flags
    // =========================================================================

    /// <summary>Operand is being reduced.</summary>
    REDUCE = 0x0020,

    /// <summary>Operand is virtual (no backing array).</summary>
    VIRTUAL = 0x0040,

    /// <summary>Operand requires masking when copying buffer to array.</summary>
    WRITEMASKED = 0x0080,

    // =========================================================================
    // Buffer State Flags
    // =========================================================================

    /// <summary>Buffer is fully filled and ready for reuse.</summary>
    BUF_REUSABLE = 0x0100,

    /// <summary>Operand must be copied.</summary>
    FORCECOPY = 0x0200,

    /// <summary>Operand has temporary data, write back at dealloc.</summary>
    HAS_WRITEBACK = 0x0400,

    /// <summary>User requested contiguous operand.</summary>
    CONTIG = 0x0800,
}
```

### NpyIterGlobalFlags (Construction Flags)

```csharp
/// <summary>
/// Global flags passed to iterator construction.
/// Matches NumPy's NPY_ITER_* constants.
/// </summary>
[Flags]
public enum NpyIterGlobalFlags : uint
{
    None = 0,

    // =========================================================================
    // Index Tracking
    // =========================================================================

    /// <summary>Track a C-order flat index.</summary>
    C_INDEX = 0x0001,

    /// <summary>Track an F-order flat index.</summary>
    F_INDEX = 0x0002,

    /// <summary>Track a multi-index.</summary>
    MULTI_INDEX = 0x0004,

    // =========================================================================
    // Loop Control
    // =========================================================================

    /// <summary>Expose inner loop to external code.</summary>
    EXTERNAL_LOOP = 0x0008,

    /// <summary>Don't negate strides for axes iterated in reverse.</summary>
    DONT_NEGATE_STRIDES = 0x0010,

    // =========================================================================
    // Buffering
    // =========================================================================

    /// <summary>Enable buffering.</summary>
    BUFFERED = 0x0020,

    /// <summary>Grow inner loop when possible.</summary>
    GROWINNER = 0x0040,

    /// <summary>Delay buffer allocation until Reset.</summary>
    DELAY_BUFALLOC = 0x0080,

    // =========================================================================
    // Safety and Compatibility
    // =========================================================================

    /// <summary>Allow zero-size arrays.</summary>
    ZEROSIZE_OK = 0x0100,

    /// <summary>Allow object dtype arrays.</summary>
    REFS_OK = 0x0200,

    /// <summary>Allow reduction operands.</summary>
    REDUCE_OK = 0x0400,

    /// <summary>Enable ranged iteration.</summary>
    RANGED = 0x0800,

    // =========================================================================
    // Type Handling
    // =========================================================================

    /// <summary>Find common dtype for all operands.</summary>
    COMMON_DTYPE = 0x1000,

    /// <summary>Copy operands if they overlap in memory.</summary>
    COPY_IF_OVERLAP = 0x2000,

    /// <summary>Assume elementwise access for overlap detection.</summary>
    OVERLAP_ASSUME_ELEMENTWISE = 0x4000,
}
```

### NpyIterPerOpFlags (Per-Operand Construction Flags)

```csharp
/// <summary>
/// Per-operand flags passed to iterator construction.
/// Matches NumPy's NPY_ITER_* per-operand constants.
/// </summary>
[Flags]
public enum NpyIterPerOpFlags : uint
{
    None = 0,

    // =========================================================================
    // Read/Write Mode
    // =========================================================================

    /// <summary>Operand is read-only.</summary>
    READONLY = 0x0001,

    /// <summary>Operand is write-only.</summary>
    WRITEONLY = 0x0002,

    /// <summary>Operand is read-write.</summary>
    READWRITE = 0x0004,

    // =========================================================================
    // Allocation and Copying
    // =========================================================================

    /// <summary>Copy operand data.</summary>
    COPY = 0x0008,

    /// <summary>Update original if copy is made.</summary>
    UPDATEIFCOPY = 0x0010,

    /// <summary>Allocate output array if null.</summary>
    ALLOCATE = 0x0020,

    /// <summary>Don't allocate with subtype.</summary>
    NO_SUBTYPE = 0x0040,

    // =========================================================================
    // Broadcasting Control
    // =========================================================================

    /// <summary>Don't broadcast this operand.</summary>
    NO_BROADCAST = 0x0080,

    // =========================================================================
    // Memory Layout
    // =========================================================================

    /// <summary>Require contiguous data.</summary>
    CONTIG = 0x0100,

    /// <summary>Require aligned data.</summary>
    ALIGNED = 0x0200,

    /// <summary>Require native byte order.</summary>
    NBO = 0x0400,

    // =========================================================================
    // Masking
    // =========================================================================

    /// <summary>This operand is an array mask.</summary>
    ARRAYMASK = 0x0800,

    /// <summary>Write only where mask is true.</summary>
    WRITEMASKED = 0x1000,

    // =========================================================================
    // Reduction
    // =========================================================================

    /// <summary>Mark as a reduction axis.</summary>
    REDUCTION_AXIS = unchecked((uint)(-1)),  // Special marker for op_axes
}
```

---

## Core Operations

### Construction

```csharp
public ref struct NpyIter
{
    private NpyIterState* _state;
    private bool _ownsState;

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /// <summary>
    /// Create iterator for a single operand.
    /// Equivalent to NumPy's NpyIter_New.
    /// </summary>
    public static NpyIter New(
        NDArray op,
        NpyIterGlobalFlags flags = NpyIterGlobalFlags.None,
        NPY_ORDER order = NPY_ORDER.NPY_KEEPORDER,
        NPY_CASTING casting = NPY_CASTING.NPY_SAFE_CASTING,
        NPTypeCode? dtype = null)
    {
        var opFlags = new[] { NpyIterPerOpFlags.READONLY };
        var dtypes = dtype.HasValue ? new[] { dtype.Value } : null;
        return AdvancedNew(1, new[] { op }, flags, order, casting, opFlags, dtypes);
    }

    /// <summary>
    /// Create iterator for multiple operands.
    /// Equivalent to NumPy's NpyIter_MultiNew.
    /// </summary>
    public static NpyIter MultiNew(
        int nop,
        NDArray[] op,
        NpyIterGlobalFlags flags,
        NPY_ORDER order,
        NPY_CASTING casting,
        NpyIterPerOpFlags[] opFlags,
        NPTypeCode[]? opDtypes = null)
    {
        return AdvancedNew(nop, op, flags, order, casting, opFlags, opDtypes);
    }

    /// <summary>
    /// Create iterator with full control over all parameters.
    /// Equivalent to NumPy's NpyIter_AdvancedNew.
    /// </summary>
    public static NpyIter AdvancedNew(
        int nop,
        NDArray[] op,
        NpyIterGlobalFlags flags,
        NPY_ORDER order,
        NPY_CASTING casting,
        NpyIterPerOpFlags[] opFlags,
        NPTypeCode[]? opDtypes = null,
        int opAxesNDim = -1,
        int[][]? opAxes = null,
        long[]? iterShape = null,
        long bufferSize = 0)
    {
        // Implementation follows NumPy's npyiter_construct flow:
        // 1. Validate inputs
        // 2. Calculate broadcast shape
        // 3. Determine iteration order
        // 4. Apply axis permutation
        // 5. Calculate strides in iteration space
        // 6. Apply axis coalescing
        // 7. Allocate buffers if needed
        // 8. Initialize state

        // ... (see Implementation Phases)
    }
}
```

### Iteration Functions

```csharp
public ref struct NpyIter
{
    // =========================================================================
    // Iteration Control
    // =========================================================================

    /// <summary>
    /// Get the iteration-advance function.
    /// Returns a delegate that advances to next iteration.
    /// </summary>
    public NpyIterNextFunc GetIterNext()
    {
        // Select specialized function based on flags
        var itflags = (NpyIterFlags)_state->ItFlags;

        if ((itflags & NpyIterFlags.BUFFER) != 0)
            return GetBufferedIterNext();

        if ((itflags & NpyIterFlags.EXLOOP) != 0)
            return GetExternalLoopIterNext();

        if ((itflags & NpyIterFlags.ONEITERATION) != 0)
            return GetSingleIterationIterNext();

        return GetStandardIterNext();
    }

    /// <summary>
    /// Get array of current data pointers.
    /// </summary>
    public void** GetDataPtrArray()
    {
        fixed (long* p = _state->DataPtrs)
            return (void**)p;
    }

    /// <summary>
    /// Get array of inner loop strides.
    /// </summary>
    public long* GetInnerStrideArray()
    {
        // Inner strides are the strides for axis 0 (fastest varying)
        fixed (long* p = _state->Strides)
            return p;
    }

    /// <summary>
    /// Get pointer to inner loop size.
    /// </summary>
    public long* GetInnerLoopSizePtr()
    {
        // For buffered: return buffer size
        // For unbuffered: return shape[0]
        if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
        {
            // Point to buffer size field
            return &_state->BufIterEnd;
        }
        else
        {
            fixed (long* p = _state->Shape)
                return p;
        }
    }

    /// <summary>
    /// Get the total iteration size.
    /// </summary>
    public long GetIterSize() => _state->IterSize;

    /// <summary>
    /// Get the current iteration index.
    /// </summary>
    public long GetIterIndex() => _state->IterIndex;

    /// <summary>
    /// Reset iterator to the beginning.
    /// </summary>
    public bool Reset()
    {
        _state->IterIndex = 0;

        // Reset coordinates
        for (int d = 0; d < _state->NDim; d++)
            _state->Coords[d] = 0;

        // Reset data pointers to reset positions
        for (int op = 0; op < _state->NOp; op++)
            _state->DataPtrs[op] = _state->ResetDataPtrs[op];

        // If buffered, prepare first buffer
        if ((_state->ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
            return PrepareBuffers();

        return true;
    }

    /// <summary>
    /// Jump to a specific iteration index.
    /// </summary>
    public void GotoIterIndex(long iterindex)
    {
        _state->IterIndex = iterindex;

        // Calculate coordinates from linear index
        long remaining = iterindex;
        for (int d = _state->NDim - 1; d >= 0; d--)
        {
            long shape = _state->Shape[d];
            _state->Coords[d] = remaining % shape;
            remaining /= shape;
        }

        // Update data pointers
        for (int op = 0; op < _state->NOp; op++)
        {
            long offset = 0;
            for (int d = 0; d < _state->NDim; d++)
            {
                offset += _state->Coords[d] * _state->GetStride(d, op);
            }
            _state->DataPtrs[op] = _state->ResetDataPtrs[op] + offset * GetElementSize(op);
        }
    }
}
```

### Delegate Types

```csharp
/// <summary>
/// Function to advance iterator to next position.
/// Returns true if more iterations remain.
/// </summary>
public unsafe delegate bool NpyIterNextFunc(ref NpyIterState state);

/// <summary>
/// Function to get multi-index at current position.
/// </summary>
public unsafe delegate void NpyIterGetMultiIndexFunc(ref NpyIterState state, long* outCoords);

/// <summary>
/// Inner loop kernel called by iterator.
/// </summary>
public unsafe delegate void NpyIterInnerLoopFunc(
    void** dataptrs,
    long* strides,
    long count,
    void* auxdata);
```

---

## Execution Paths

### Path Selection Logic

```csharp
internal static class NpyIterPathSelector
{
    /// <summary>
    /// Determine the optimal execution path based on operand layout.
    /// </summary>
    public static NpyIterExecutionPath SelectPath(ref NpyIterState state)
    {
        // Check if all operands are contiguous
        bool allContiguous = true;
        bool anyBroadcast = false;
        bool canGather = true;

        for (int op = 0; op < state.NOp; op++)
        {
            // Check inner stride
            long innerStride = state.GetStride(0, op);

            if (innerStride != 1)
                allContiguous = false;

            if (innerStride == 0)
                anyBroadcast = true;

            // Gather requires stride fits in int32 and is positive
            if (innerStride < 0 || innerStride > int.MaxValue)
                canGather = false;
        }

        // Select path
        if (allContiguous)
            return NpyIterExecutionPath.Contiguous;

        if (anyBroadcast || !canGather)
        {
            // Need buffering for broadcast or large strides
            if ((state.ItFlags & (uint)NpyIterFlags.BUFFER) != 0)
                return NpyIterExecutionPath.Buffered;
            else
                return NpyIterExecutionPath.General;
        }

        // Can use gather for strided access
        if (Avx2.IsSupported)
            return NpyIterExecutionPath.Strided;

        return NpyIterExecutionPath.General;
    }
}

public enum NpyIterExecutionPath
{
    /// <summary>All operands contiguous, use direct SIMD.</summary>
    Contiguous,

    /// <summary>Strided but gather-compatible, use AVX2 gather.</summary>
    Strided,

    /// <summary>Copy to contiguous buffers, SIMD on buffers.</summary>
    Buffered,

    /// <summary>Coordinate-based iteration, scalar operations.</summary>
    General,
}
```

### Contiguous Path

```csharp
internal static class NpyIterContiguousPath
{
    /// <summary>
    /// Execute contiguous iteration with SIMD kernel.
    /// </summary>
    public static unsafe void Execute<TKernel>(
        ref NpyIterState state,
        TKernel kernel)
        where TKernel : INpyIterKernel
    {
        void** dataptrs = (void**)state.GetDataPtr(0);
        long count = state.IterSize;

        // Get contiguous kernel from IL generator
        var innerKernel = kernel.GetInnerKernel(NpyIterExecutionPath.Contiguous);

        // Execute in single call (no iteration needed)
        fixed (long* strides = state.Strides)
        {
            innerKernel(dataptrs, strides, count);
        }
    }
}
```

### Buffered Path

```csharp
internal static class NpyIterBufferedPath
{
    /// <summary>
    /// Execute buffered iteration.
    /// </summary>
    public static unsafe void Execute<TKernel>(
        ref NpyIterState state,
        TKernel kernel)
        where TKernel : INpyIterKernel
    {
        long bufferSize = state.BufferSize;
        var innerKernel = kernel.GetInnerKernel(NpyIterExecutionPath.Contiguous);

        // Allocate aligned buffers
        Span<IntPtr> buffers = stackalloc IntPtr[state.NOp];
        for (int op = 0; op < state.NOp; op++)
        {
            buffers[op] = AllocateAlignedBuffer(bufferSize, state.GetOpDType(op));
        }

        try
        {
            long remaining = state.IterSize;

            while (remaining > 0)
            {
                long batchSize = Math.Min(remaining, bufferSize);

                // Copy from operands to buffers
                CopyToBuffers(ref state, buffers, batchSize);

                // Execute kernel on buffers
                void** bufPtrs = stackalloc void*[state.NOp];
                for (int op = 0; op < state.NOp; op++)
                    bufPtrs[op] = (void*)buffers[op];

                long* bufStrides = stackalloc long[state.NOp];
                for (int op = 0; op < state.NOp; op++)
                    bufStrides[op] = 1;  // Buffers are contiguous

                innerKernel(bufPtrs, bufStrides, batchSize);

                // Copy from buffers back to operands (for write operands)
                CopyFromBuffers(ref state, buffers, batchSize);

                // Advance state
                AdvanceBy(ref state, batchSize);
                remaining -= batchSize;
            }
        }
        finally
        {
            // Free buffers
            for (int op = 0; op < state.NOp; op++)
            {
                if (buffers[op] != IntPtr.Zero)
                    FreeAlignedBuffer(buffers[op]);
            }
        }
    }

    private static unsafe void CopyToBuffers(
        ref NpyIterState state,
        Span<IntPtr> buffers,
        long count)
    {
        for (int op = 0; op < state.NOp; op++)
        {
            var opFlags = state.GetOpFlags(op);
            if ((opFlags & NpyIterOpFlags.READ) == 0)
                continue;  // Write-only, skip

            var dtype = state.GetOpDType(op);
            void* src = state.GetDataPtr(op);
            void* dst = (void*)buffers[op];

            // Get strided→contiguous copy kernel
            var copyKernel = ILKernelGenerator.GetStridedToContiguousCopyKernel(dtype);

            // Execute copy
            fixed (long* strides = state.Strides)
            fixed (long* shape = state.Shape)
            {
                copyKernel(src, dst, strides + op, shape, state.NDim, count);
            }
        }
    }
}
```

### General Path (Coordinate Iteration)

```csharp
internal static class NpyIterGeneralPath
{
    /// <summary>
    /// Execute general coordinate-based iteration.
    /// </summary>
    public static unsafe void Execute<TKernel>(
        ref NpyIterState state,
        TKernel kernel)
        where TKernel : INpyIterKernel
    {
        // Process element by element
        for (long i = 0; i < state.IterSize; i++)
        {
            // Get current data pointers
            void** dataptrs = (void**)Unsafe.AsPointer(ref state.DataPtrs[0]);

            // Process single element
            kernel.ProcessElement(dataptrs);

            // Advance to next position
            Advance(ref state);
        }
    }

    /// <summary>
    /// Advance iterator by one position.
    /// </summary>
    private static unsafe void Advance(ref NpyIterState state)
    {
        state.IterIndex++;

        // Update coordinates and data pointers (ripple carry)
        for (int axis = state.NDim - 1; axis >= 0; axis--)
        {
            state.Coords[axis]++;

            if (state.Coords[axis] < state.Shape[axis])
            {
                // Advance data pointers along this axis
                for (int op = 0; op < state.NOp; op++)
                {
                    long stride = state.GetStride(axis, op);
                    state.DataPtrs[op] += stride * GetElementSize(state.GetOpDType(op));
                }
                return;
            }

            // Carry: reset this axis, continue to next
            state.Coords[axis] = 0;

            // Reset data pointers for this axis
            for (int op = 0; op < state.NOp; op++)
            {
                long stride = state.GetStride(axis, op);
                long shape = state.Shape[axis];
                state.DataPtrs[op] -= stride * (shape - 1) * GetElementSize(state.GetOpDType(op));
            }
        }
    }
}
```

---

## IL Kernel Integration

### Kernel Interface

```csharp
/// <summary>
/// Interface for kernels that work with NpyIter.
/// </summary>
public interface INpyIterKernel
{
    /// <summary>
    /// Get the inner loop function for the specified execution path.
    /// </summary>
    NpyIterInnerLoopFunc GetInnerKernel(NpyIterExecutionPath path);

    /// <summary>
    /// Process a single element (for general path).
    /// </summary>
    unsafe void ProcessElement(void** dataptrs);

    /// <summary>
    /// Whether this kernel supports early exit.
    /// </summary>
    bool SupportsEarlyExit { get; }

    /// <summary>
    /// Required alignment for buffers (0 for no requirement).
    /// </summary>
    int RequiredAlignment { get; }
}
```

### Kernel Registration

```csharp
/// <summary>
/// Factory for creating NpyIter-compatible kernels.
/// </summary>
public static class NpyIterKernelFactory
{
    /// <summary>
    /// Create a binary operation kernel.
    /// </summary>
    public static INpyIterKernel CreateBinaryKernel(BinaryOp op, NPTypeCode dtype)
    {
        return new BinaryOpKernel(op, dtype);
    }

    /// <summary>
    /// Create a reduction kernel.
    /// </summary>
    public static INpyIterKernel CreateReductionKernel(ReductionOp op, NPTypeCode dtype)
    {
        return new ReductionOpKernel(op, dtype);
    }

    /// <summary>
    /// Create a unary operation kernel.
    /// </summary>
    public static INpyIterKernel CreateUnaryKernel(UnaryOp op, NPTypeCode inputType, NPTypeCode outputType)
    {
        return new UnaryOpKernel(op, inputType, outputType);
    }
}

/// <summary>
/// Binary operation kernel implementation.
/// </summary>
internal class BinaryOpKernel : INpyIterKernel
{
    private readonly BinaryOp _op;
    private readonly NPTypeCode _dtype;
    private readonly NpyIterInnerLoopFunc _contiguousKernel;
    private readonly NpyIterInnerLoopFunc _stridedKernel;

    public BinaryOpKernel(BinaryOp op, NPTypeCode dtype)
    {
        _op = op;
        _dtype = dtype;

        // Get IL-generated kernels
        _contiguousKernel = CreateContiguousKernel(op, dtype);
        _stridedKernel = CreateStridedKernel(op, dtype);
    }

    public NpyIterInnerLoopFunc GetInnerKernel(NpyIterExecutionPath path)
    {
        return path switch
        {
            NpyIterExecutionPath.Contiguous => _contiguousKernel,
            NpyIterExecutionPath.Strided => _stridedKernel,
            NpyIterExecutionPath.Buffered => _contiguousKernel,  // Buffers are contiguous
            _ => throw new NotSupportedException($"Path {path} not supported")
        };
    }

    public unsafe void ProcessElement(void** dataptrs)
    {
        // Single element processing for general path
        // Delegate to scalar operation
        ILKernelGenerator.InvokeBinaryScalar(_op, _dtype, dataptrs[0], dataptrs[1], dataptrs[2]);
    }

    public bool SupportsEarlyExit => false;
    public int RequiredAlignment => 32;  // AVX2 alignment

    private static unsafe NpyIterInnerLoopFunc CreateContiguousKernel(BinaryOp op, NPTypeCode dtype)
    {
        // Wrap IL-generated kernel
        var kernel = ILKernelGenerator.GetMixedTypeKernel(
            new MixedTypeKernelKey(op, dtype, dtype, dtype, BinaryExecutionPath.SimdFull));

        return (dataptrs, strides, count, auxdata) =>
        {
            kernel(dataptrs[0], dataptrs[1], dataptrs[2],
                   strides[0], strides[1], strides[2],
                   null, null, null, 0, count);
        };
    }
}
```

---

## Buffering System

### Buffer Allocation

```csharp
internal static class NpyIterBufferManager
{
    /// <summary>
    /// Default buffer size (number of elements).
    /// </summary>
    public const long DefaultBufferSize = 8192;

    /// <summary>
    /// Required alignment for SIMD operations.
    /// </summary>
    public const int Alignment = 64;  // Cache line size, good for AVX-512

    /// <summary>
    /// Allocate aligned buffer.
    /// </summary>
    public static unsafe void* AllocateAligned(long elements, NPTypeCode dtype)
    {
        long bytes = elements * InfoOf.GetSize(dtype);
        return NativeMemory.AlignedAlloc((nuint)bytes, Alignment);
    }

    /// <summary>
    /// Free aligned buffer.
    /// </summary>
    public static unsafe void FreeAligned(void* buffer)
    {
        NativeMemory.AlignedFree(buffer);
    }

    /// <summary>
    /// Determine optimal buffer size based on array sizes and cache.
    /// </summary>
    public static long DetermineBufferSize(ref NpyIterState state, long requestedSize)
    {
        if (requestedSize > 0)
            return requestedSize;

        // Use L2 cache size heuristic
        const long L2CacheSize = 256 * 1024;  // 256 KB

        long totalElementSize = 0;
        for (int op = 0; op < state.NOp; op++)
        {
            totalElementSize += InfoOf.GetSize(state.GetOpDType(op));
        }

        // Target: buffers fit in L2 cache
        long maxElements = L2CacheSize / totalElementSize;

        // Round down to SIMD vector multiple
        int vectorSize = 32;  // AVX2
        maxElements = (maxElements / vectorSize) * vectorSize;

        return Math.Max(vectorSize, Math.Min(maxElements, DefaultBufferSize));
    }
}
```

### Buffer Copy Kernels

```csharp
internal static class NpyIterBufferCopy
{
    /// <summary>
    /// Copy strided data to contiguous buffer.
    /// </summary>
    public static unsafe void StridedToContiguous<T>(
        T* src,
        T* dst,
        long* strides,
        long* shape,
        int ndim,
        long count)
        where T : unmanaged
    {
        if (ndim == 1 && strides[0] == 1)
        {
            // Already contiguous: memcpy
            Unsafe.CopyBlock(dst, src, (uint)(count * sizeof(T)));
            return;
        }

        // Use IL-generated copy kernel
        var kernel = ILKernelGenerator.TryGetCopyKernel(
            new CopyKernelKey(InfoOf<T>.NPTypeCode, CopyExecutionPath.General));

        if (kernel != null)
        {
            long* dstStrides = stackalloc long[ndim];
            ComputeContiguousStrides(shape, ndim, dstStrides);
            kernel(src, dst, strides, dstStrides, shape, ndim, count);
        }
        else
        {
            // Fallback scalar copy
            CopyStridedScalar(src, dst, strides, shape, ndim, count);
        }
    }

    /// <summary>
    /// Copy contiguous buffer to strided destination.
    /// </summary>
    public static unsafe void ContiguousToStrided<T>(
        T* src,
        T* dst,
        long* strides,
        long* shape,
        int ndim,
        long count)
        where T : unmanaged
    {
        if (ndim == 1 && strides[0] == 1)
        {
            Unsafe.CopyBlock(dst, src, (uint)(count * sizeof(T)));
            return;
        }

        var kernel = ILKernelGenerator.TryGetCopyKernel(
            new CopyKernelKey(InfoOf<T>.NPTypeCode, CopyExecutionPath.General));

        if (kernel != null)
        {
            long* srcStrides = stackalloc long[ndim];
            ComputeContiguousStrides(shape, ndim, srcStrides);
            kernel(src, dst, srcStrides, strides, shape, ndim, count);
        }
        else
        {
            CopyStridedScalar(src, dst, strides, shape, ndim, count);
        }
    }
}
```

---

## Axis Coalescing

### Algorithm

```csharp
internal static class NpyIterCoalescing
{
    /// <summary>
    /// Coalesce adjacent axes that have compatible strides.
    /// Reduces ndim, improving iteration efficiency.
    /// </summary>
    public static unsafe void CoalesceAxes(ref NpyIterState state)
    {
        if (state.NDim <= 1)
            return;

        int writeAxis = 0;
        int newNDim = 1;

        for (int readAxis = 0; readAxis < state.NDim - 1; readAxis++)
        {
            int nextAxis = readAxis + 1;
            long shape0 = state.Shape[writeAxis];
            long shape1 = state.Shape[nextAxis];

            // Check if all operands can be coalesced
            bool canCoalesce = true;
            for (int op = 0; op < state.NOp; op++)
            {
                long stride0 = state.GetStride(writeAxis, op);
                long stride1 = state.GetStride(nextAxis, op);

                // Can coalesce if:
                // - Either axis has shape 1 (trivial dimension)
                // - Strides are compatible: stride0 * shape0 == stride1
                bool opCanCoalesce =
                    (shape0 == 1 && stride0 == 0) ||
                    (shape1 == 1 && stride1 == 0) ||
                    (stride0 * shape0 == stride1);

                if (!opCanCoalesce)
                {
                    canCoalesce = false;
                    break;
                }
            }

            if (canCoalesce)
            {
                // Merge nextAxis into writeAxis
                state.Shape[writeAxis] *= shape1;

                // Update strides (take non-zero stride)
                for (int op = 0; op < state.NOp; op++)
                {
                    long stride0 = state.GetStride(writeAxis, op);
                    long stride1 = state.GetStride(nextAxis, op);

                    if (stride0 == 0)
                        state.SetStride(writeAxis, op, stride1);
                }
            }
            else
            {
                // Move to next write position
                writeAxis++;
                if (writeAxis != nextAxis)
                {
                    state.Shape[writeAxis] = state.Shape[nextAxis];
                    for (int op = 0; op < state.NOp; op++)
                    {
                        state.SetStride(writeAxis, op, state.GetStride(nextAxis, op));
                    }
                }
                newNDim++;
            }
        }

        // Update state
        state.NDim = newNDim;

        // Reset permutation to identity
        for (int d = 0; d < newNDim; d++)
            state.Perm[d] = (sbyte)d;

        // Clear IDENTPERM/HASMULTIINDEX flags
        state.ItFlags &= ~(uint)(NpyIterFlags.IDENTPERM | NpyIterFlags.HASMULTIINDEX);
    }
}
```

### Coalescing Examples

```
Before coalescing:
  Shape: [2, 3, 4, 5]
  Strides (op0): [60, 20, 5, 1]  (C-contiguous)

After coalescing:
  Shape: [120]
  Strides (op0): [1]
  NDim: 1

Before coalescing:
  Shape: [2, 3, 4]
  Strides (op0): [12, 4, 1]   (C-contiguous)
  Strides (op1): [1, 0, 0]    (broadcast from scalar)

After coalescing:
  Shape: [2, 12]
  Strides (op0): [12, 1]
  Strides (op1): [1, 0]       (broadcast dimension preserved)
  NDim: 2
```

---

## API Surface

### Public API

```csharp
namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// High-performance multi-operand iterator.
    /// Matches NumPy's nditer API.
    /// </summary>
    public ref struct NpyIter
    {
        // =====================================================================
        // Factory Methods
        // =====================================================================

        /// <summary>Create single-operand iterator.</summary>
        public static NpyIter New(
            NDArray op,
            NpyIterGlobalFlags flags = NpyIterGlobalFlags.None,
            NPY_ORDER order = NPY_ORDER.NPY_KEEPORDER,
            NPY_CASTING casting = NPY_CASTING.NPY_SAFE_CASTING,
            NPTypeCode? dtype = null);

        /// <summary>Create multi-operand iterator.</summary>
        public static NpyIter MultiNew(
            int nop,
            NDArray[] op,
            NpyIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NpyIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null);

        /// <summary>Create iterator with full control.</summary>
        public static NpyIter AdvancedNew(
            int nop,
            NDArray[] op,
            NpyIterGlobalFlags flags,
            NPY_ORDER order,
            NPY_CASTING casting,
            NpyIterPerOpFlags[] opFlags,
            NPTypeCode[]? opDtypes = null,
            int opAxesNDim = -1,
            int[][]? opAxes = null,
            long[]? iterShape = null,
            long bufferSize = 0);

        // =====================================================================
        // Properties
        // =====================================================================

        /// <summary>Number of operands.</summary>
        public int NOp { get; }

        /// <summary>Number of dimensions after coalescing.</summary>
        public int NDim { get; }

        /// <summary>Total iteration count.</summary>
        public long IterSize { get; }

        /// <summary>Whether iterator requires buffering.</summary>
        public bool RequiresBuffering { get; }

        /// <summary>Whether iteration needs Python API.</summary>
        public bool IterationNeedsAPI { get; }

        /// <summary>Get operand arrays.</summary>
        public NDArray[] GetOperandArray();

        /// <summary>Get operand dtypes.</summary>
        public NPTypeCode[] GetDescrArray();

        // =====================================================================
        // Iteration Methods
        // =====================================================================

        /// <summary>Get iteration advance function.</summary>
        public NpyIterNextFunc GetIterNext();

        /// <summary>Get current data pointer array.</summary>
        public unsafe void** GetDataPtrArray();

        /// <summary>Get inner loop stride array.</summary>
        public unsafe long* GetInnerStrideArray();

        /// <summary>Get pointer to inner loop size.</summary>
        public unsafe long* GetInnerLoopSizePtr();

        /// <summary>Reset to beginning.</summary>
        public bool Reset();

        /// <summary>Jump to iteration index.</summary>
        public void GotoIterIndex(long iterindex);

        // =====================================================================
        // Configuration Methods
        // =====================================================================

        /// <summary>Remove axis from iteration.</summary>
        public bool RemoveAxis(int axis);

        /// <summary>Remove multi-index tracking.</summary>
        public bool RemoveMultiIndex();

        /// <summary>Enable external loop handling.</summary>
        public bool EnableExternalLoop();

        // =====================================================================
        // Multi-Index Methods
        // =====================================================================

        /// <summary>Get function to retrieve multi-index.</summary>
        public NpyIterGetMultiIndexFunc GetGetMultiIndex();

        /// <summary>Goto specific multi-index.</summary>
        public void GotoMultiIndex(params long[] multiIndex);

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /// <summary>Deallocate iterator resources.</summary>
        public void Dispose();
    }
}
```

### Usage Examples

```csharp
// Example 1: Simple element-wise addition
using var iter = NpyIter.MultiNew(
    nop: 3,
    op: new[] { a, b, result },
    flags: NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_SAFE_CASTING,
    opFlags: new[] {
        NpyIterPerOpFlags.READONLY,
        NpyIterPerOpFlags.READONLY,
        NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.ALLOCATE
    });

var iternext = iter.GetIterNext();
var dataptrs = iter.GetDataPtrArray();
var strides = iter.GetInnerStrideArray();
var countptr = iter.GetInnerLoopSizePtr();

do
{
    // Inner loop handled by SIMD kernel
    AddKernel(dataptrs[0], dataptrs[1], dataptrs[2], strides, *countptr);
} while (iternext(ref iter._state));


// Example 2: Reduction (sum)
using var iter = NpyIter.AdvancedNew(
    nop: 2,
    op: new[] { input, output },
    flags: NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.REDUCE_OK,
    order: NPY_ORDER.NPY_KEEPORDER,
    casting: NPY_CASTING.NPY_SAFE_CASTING,
    opFlags: new[] {
        NpyIterPerOpFlags.READONLY,
        NpyIterPerOpFlags.READWRITE | NpyIterPerOpFlags.ALLOCATE
    },
    opAxes: new[] {
        null,  // input: all axes
        new[] { -1, -1, 0 }  // output: reduction axes marked with -1
    });

// ... iterate with reduction kernel
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2) ✅ COMPLETED

**Goal:** Basic single-operand iteration working

- [x] `NpyIterState` struct with fixed buffers
- [x] `NpyIterFlags` and `NpyIterOpFlags` enums
- [x] `NpyIter.New()` for single operand
- [x] Basic `GetIterNext()` returning standard iterator
- [x] `GetDataPtrArray()`, `GetInnerStrideArray()`, `GetInnerLoopSizePtr()`
- [x] `Reset()` and `GotoIterIndex()`
- [x] Unit tests for single-operand iteration

**Deliverables:**
- `NpyIter.cs` - Main ref struct (`NpyIterRef`)
- `NpyIterState.cs` - State struct (enhanced with full accessor methods)
- `NpyIterFlags.cs` - All flag enums (complete NumPy parity)
- `NpyIterRefTests.cs` - Basic tests

### Phase 2: Multi-Operand Support (Week 3-4) ✅ COMPLETED

**Goal:** Multi-operand iteration with broadcasting

- [x] `NpyIter.MultiNew()` implementation
- [x] Broadcasting shape calculation
- [x] Stride calculation in broadcast space
- [x] `NpyIter.AdvancedNew()` with op_axes support
- [x] Multi-operand coordinate tracking
- [x] Unit tests for broadcasting scenarios

**Deliverables:**
- Broadcasting logic integrated in `NpyIterRef`
- Multi-operand tests in `NpyIterRefTests.cs`

### Phase 3: Axis Coalescing (Week 5) ⚠️ PARTIAL

**Goal:** Automatic axis optimization

- [x] `npyiter_coalesce_axes()` implementation
- [x] Integration with construction
- [x] `RemoveAxis()` API
- [ ] `RemoveMultiIndex()` API (not implemented)
- [x] Tests verifying coalescing behavior

**Notes:** Coalescing works for 2-operand copy scenarios. Multi-operand coalescing needs refinement.

**Deliverables:**
- `NpyIterCoalescing.cs` - Full coalescing logic
- Coalescing tests (basic)

### Phase 4: External Loop (Week 6) ✅ COMPLETED

**Goal:** Expose inner loop to callers

- [x] `EXTERNAL_LOOP` flag handling
- [x] `EnableExternalLoop()` API
- [x] Inner stride and size calculation
- [ ] Integration with ILKernelGenerator (partial - kernel interfaces defined)
- [ ] Performance tests

**Deliverables:**
- External loop support
- Kernel integration tests

### Phase 5: Buffering (Week 7-8)

**Goal:** Full buffering support

- [ ] `NpyIterBufferData` struct
- [ ] Buffer allocation with alignment
- [ ] `CopyToBuffers()` - strided to contiguous
- [ ] `CopyFromBuffers()` - contiguous to strided
- [ ] Buffer size optimization
- [ ] `DELAY_BUFALLOC` support
- [ ] `GROWINNER` support

**Deliverables:**
- `NpyIterBufferManager.cs`
- `NpyIterBufferCopy.cs`
- Buffering tests

### Phase 6: Type Casting (Week 9)

**Goal:** Type conversion during iteration

- [ ] Cast info structure
- [ ] Integration with IL type conversion kernels
- [ ] Common dtype detection (`COMMON_DTYPE`)
- [ ] Safe/unsafe casting modes

**Deliverables:**
- Type casting support
- Casting tests

### Phase 7: Reduction Support (Week 10)

**Goal:** Full reduction axis support

- [ ] `REDUCE_OK` flag handling
- [ ] Reduction axis marking in op_axes
- [ ] Reduce position tracking
- [ ] Integration with reduction kernels

**Deliverables:**
- Reduction support
- Reduction tests

### Phase 8: Optimization Integration (Week 11-12)

**Goal:** Connect all IL optimizations

- [ ] Execution path selection
- [ ] Contiguous path with SIMD
- [ ] Strided path with AVX2 gather
- [ ] Buffered path optimization
- [ ] Parallel outer loop (where safe)
- [ ] Performance benchmarks

**Deliverables:**
- `NpyIterPathSelector.cs`
- Path-specific execution
- Benchmark suite

### Phase 9: API Parity Verification (Week 13)

**Goal:** Verify NumPy compatibility

- [ ] Compare with NumPy test suite
- [ ] Edge case testing
- [ ] Error handling parity
- [ ] Documentation

**Deliverables:**
- NumPy parity tests
- API documentation

---

## Testing Strategy

### Unit Test Categories

| Category | Tests | Priority |
|----------|-------|----------|
| Construction | Validate all factory methods | P0 |
| Single-operand | Basic iteration patterns | P0 |
| Multi-operand | Broadcasting, sync | P0 |
| Coalescing | Axis merging | P1 |
| Buffering | Copy correctness | P1 |
| External loop | Kernel integration | P1 |
| Reduction | Axis reduction | P1 |
| Edge cases | Empty, scalar, 0-stride | P2 |

### Test Patterns

```csharp
[Test]
public void NpyIter_SingleOperand_Contiguous()
{
    var arr = np.arange(24).reshape(2, 3, 4);

    using var iter = NpyIter.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);

    Assert.That(iter.NDim, Is.EqualTo(1));  // Coalesced to 1D
    Assert.That(iter.IterSize, Is.EqualTo(24));

    var iternext = iter.GetIterNext();
    var dataptrs = iter.GetDataPtrArray();
    var count = *iter.GetInnerLoopSizePtr();

    Assert.That(count, Is.EqualTo(24));  // All in one inner loop
}

[Test]
public void NpyIter_MultiOperand_Broadcasting()
{
    var a = np.arange(12).reshape(3, 4);
    var b = np.arange(4);  // Will broadcast
    var c = np.empty((3, 4));

    using var iter = NpyIter.MultiNew(
        nop: 3,
        op: new[] { a, b, c },
        flags: NpyIterGlobalFlags.EXTERNAL_LOOP,
        order: NPY_ORDER.NPY_KEEPORDER,
        casting: NPY_CASTING.NPY_SAFE_CASTING,
        opFlags: new[] {
            NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.WRITEONLY
        });

    Assert.That(iter.IterSize, Is.EqualTo(12));

    // Verify strides account for broadcasting
    var strides = iter.GetInnerStrideArray();
    Assert.That(strides[1], Is.EqualTo(1));  // b: inner stride
    // Note: outer stride for b should be 0 (broadcast)
}
```

### NumPy Comparison Tests

```csharp
[Test]
public void NpyIter_MatchesNumPy_BroadcastStrides()
{
    // Run equivalent in NumPy:
    // >>> a = np.arange(12).reshape(3, 4)
    // >>> b = np.arange(4)
    // >>> it = np.nditer([a, b])
    // >>> it.operands[1].strides
    // Expected output from NumPy

    var a = np.arange(12).reshape(3, 4);
    var b = np.arange(4);

    using var iter = NpyIter.MultiNew(...);

    // Compare strides with NumPy output
    Assert.That(actualStrides, Is.EqualTo(expectedFromNumPy));
}
```

---

## Performance Targets

### Benchmarks

| Operation | NumPy Time | Target Time | Ratio |
|-----------|------------|-------------|-------|
| Sum 1M contiguous | 0.5ms | 0.5ms | 1.0x |
| Sum 1M strided | 2.0ms | 1.5ms | 0.75x (gather) |
| Binary 1M contiguous | 0.3ms | 0.3ms | 1.0x |
| Binary 1M broadcast | 1.0ms | 0.8ms | 0.8x |
| Reduce axis (1000x1000) | 1.5ms | 1.2ms | 0.8x |

### Optimization Targets

1. **Zero allocation** in hot path (iteration)
2. **SIMD utilization** > 90% for contiguous paths
3. **Buffer reuse** across iterations
4. **Parallel outer loop** for large reductions
5. **Early exit** for boolean operations

---

## References

- NumPy source: `numpy/_core/src/multiarray/nditer_*.c`
- NumPy NEP-10: New Iterator/UFunc Proposal
- NumSharp ILKernelGenerator architecture
- Intel AVX2 intrinsics documentation
