# NumPy Shared Infrastructure

This document describes the common infrastructure that NumPy operations share. Understanding this architecture enables building similarly reusable infrastructure in NumSharp.

## Overview: Code Reuse Statistics

| Component | Lines | Shared % | Used By |
|-----------|-------|----------|---------|
| NpyIter | 8,425 | 100% | All multi-dim ops |
| ufunc_object.c | 6,805 | 90% | All ufuncs |
| fast_loop_macros.h | 380 | 100% | All inner loops |
| type_resolution.c | 2,258 | 100% | All type dispatch |
| mem_overlap.c | 923 | 100% | All write ops |
| SIMD infrastructure | 818 | 100% | All vectorized ops |

**Effective reuse**: ~70% of code is shared infrastructure used by 100+ operations.

---

## 1. Ufunc Execution Skeleton

All ufuncs flow through a common execution pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UFUNC EXECUTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│  1. PyUFunc_GenericFunction()     ← Entry point for all ufuncs      │
│  2. convert_ufunc_arguments()     ← Parse Python arguments          │
│  3. promote_and_get_info()        ← Type resolution (shared table)  │
│     └─ Cache lookup (O(1) identity hash)                            │
│  4. try_trivial_single_output_loop() ← Fast path for simple cases   │
│     └─ If contiguous: skip iterator, direct loop call               │
│  5. execute_ufunc_loop()          ← Full NpyIter path               │
│     ├─ NpyIter_AdvancedNew()      ← Iterator construction           │
│     ├─ Broadcasting resolution    ← Automatic shape matching        │
│     ├─ Memory overlap check       ← Automatic safety                │
│     └─ Buffered iteration         ← If casting needed               │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Operations only implement the inner loop. Everything else is shared.

---

## 2. Fast Loop Macros

Location: `numpy/_core/src/umath/fast_loop_macros.h`

### Basic Loop Skeletons

These macros provide the iteration structure that operations plug into:

```c
// Unary loop skeleton
#define UNARY_LOOP \
    char *ip1 = args[0], *op1 = args[1]; \
    npy_intp is1 = steps[0], os1 = steps[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1)

// Binary loop skeleton
#define BINARY_LOOP \
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2]; \
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)

// Reduction loop skeleton
#define BINARY_REDUCE_LOOP(TYPE) \
    char *iop1 = args[0]; \
    TYPE io1 = *(TYPE *)iop1; \
    npy_intp is2 = steps[1]; \
    char *ip2 = args[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip2 += is2)
```

### Contiguity Detection Macros

```c
// Check if unary operation has contiguous input/output
#define IS_UNARY_CONT(tin, tout) \
    (steps[0] == sizeof(tin) && steps[1] == sizeof(tout))

// Check if binary operation has contiguous operands
#define IS_BINARY_CONT(tin, tout) \
    (steps[0] == sizeof(tin) && steps[1] == sizeof(tin) && steps[2] == sizeof(tout))

// Scalar operand detection
#define IS_BINARY_CONT_S1(tin, tout)  // First operand is scalar (step=0)
#define IS_BINARY_CONT_S2(tin, tout)  // Second operand is scalar

// SIMD eligibility (alignment + separation)
#define IS_BLOCKABLE_BINARY(esize, vsize) \
    (IS_BINARY_CONT(esize, esize) && \
     npy_is_aligned(args[0], esize) && \
     npy_is_aligned(args[1], esize) && \
     npy_is_aligned(args[2], esize) && \
     (abs_ptrdiff(args[2], args[0]) >= vsize || args[2] == args[0]))
```

### Fast Loop Variants

The `*_FAST` macros automatically dispatch to the best path:

```c
#define BINARY_LOOP_FAST(tin, tout, op) \
    do { \
        if (IS_BINARY_CONT(tin, tout)) { \
            /* Contiguous path - SIMD friendly */ \
            BINARY_LOOP { op; } \
        } \
        else if (IS_BINARY_CONT_S1(tin, tout)) { \
            /* Scalar + array path */ \
            BINARY_LOOP_S1 { op; } \
        } \
        else if (IS_BINARY_CONT_S2(tin, tout)) { \
            /* Array + scalar path */ \
            BINARY_LOOP_S2 { op; } \
        } \
        else { \
            /* Generic strided path */ \
            BINARY_LOOP { op; } \
        } \
    } while (0)
```

---

## 3. NpyIter Infrastructure

Location: `numpy/_core/src/multiarray/nditer_*.c` (~8,425 lines)

### Core Structure

```c
struct NpyIter_InternalOnly {
    npy_uint32 itflags;           // NPY_ITFLAG_* flags
    npy_uint8 ndim;               // Number of dimensions
    int nop;                      // Number of operands
    npy_intp itersize;            // Total iteration count
    npy_intp iterindex;           // Current position
    char iter_flexdata[];         // Variable-length: perm, dtypes, axisdata
};
```

### Common Iterator Setup

```c
NpyIter *iter = NpyIter_AdvancedNew(
    nop, op_arrays,
    NPY_ITER_EXTERNAL_LOOP |      // We handle inner loop
    NPY_ITER_BUFFERED |           // Enable type casting buffers
    NPY_ITER_ZEROSIZE_OK |        // Allow empty arrays
    NPY_ITER_GROWINNER |          // Maximize inner loop size
    NPY_ITER_REDUCE_OK,           // Allow reduction operands
    order, casting,
    op_flags, op_dtypes,
    -1, NULL, NULL, buffersize
);

// Get iteration pointers
NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
char **dataptr = NpyIter_GetDataPtrArray(iter);
npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

// Main iteration
do {
    inner_loop(dataptr, *countptr, strideptr);
} while (iternext(iter));
```

### Key Iterator Features

| Feature | Flag | Purpose |
|---------|------|---------|
| External loop | `NPY_ITER_EXTERNAL_LOOP` | We control inner loop |
| Buffering | `NPY_ITER_BUFFERED` | Type casting support |
| Grow inner | `NPY_ITER_GROWINNER` | Maximize SIMD runs |
| Reduce OK | `NPY_ITER_REDUCE_OK` | Allow stride=0 writes |
| Copy if overlap | `NPY_ITER_COPY_IF_OVERLAP` | Safety for in-place |

### Axis Optimization

The iterator automatically:
1. **Reorders axes** by stride for cache locality
2. **Coalesces axes** with compatible strides into larger dimensions
3. **Tracks first-visit** for no-identity reductions

---

## 4. Type Resolution

Location: `numpy/_core/src/umath/ufunc_type_resolution.c`

### Dispatch Cache

O(1) lookup using identity hash on dtype tuple:

```c
PyObject *info = PyArrayIdentityHash_GetItem(
    ufunc->_dispatch_cache,
    (PyObject **)op_dtypes
);
if (info != NULL) {
    return info;  // Cache hit
}
// Cache miss: resolve and cache
```

### Common Type Resolvers

| Resolver | Pattern | Used By |
|----------|---------|---------|
| `SimpleUniform` | XX -> X | add, sub, mul |
| `SimpleBinaryComparison` | XX -> bool | <, >, ==, != |
| `Default` | Linear search | Complex ops |

### Casting Validation

```c
// Compile-time table: can type A safely cast to type B?
static const bool _npy_can_cast_safely_table[NPY_NTYPES][NPY_NTYPES] = {
    // bool < int8 < int16 < int32 < int64 < float32 < float64
    ...
};
```

---

## 5. Template System (.c.src)

Location: `numpy/_core/src/umath/loops*.c.src`

### Template Syntax

```c
/**begin repeat
 * #TYPE = FLOAT, DOUBLE#
 * #type = npy_float, npy_double#
 * #sfx = f32, f64#
 */
NPY_NO_EXPORT void @TYPE@_sqrt(char **args, ...) {
    UNARY_LOOP_FAST(@type@, @type@, *out = npy_sqrt@sfx@(in));
}
/**end repeat**/
```

Generates:
```c
NPY_NO_EXPORT void FLOAT_sqrt(char **args, ...) {
    UNARY_LOOP_FAST(npy_float, npy_float, *out = npy_sqrtf32(in));
}
NPY_NO_EXPORT void DOUBLE_sqrt(char **args, ...) {
    UNARY_LOOP_FAST(npy_double, npy_double, *out = npy_sqrtf64(in));
}
```

### Nested Templates

```c
/**begin repeat
 * #sfx = s8, u8, s16, u16, s32, u32, s64, u64, f32, f64#
 */
/**begin repeat1
 * #kind = add, subtract, multiply#
 * #OP = +, -, *#
 */
void @sfx@_@kind@(...) {
    BINARY_LOOP_FAST(..., *out = in1 @OP@ in2);
}
/**end repeat1**/
/**end repeat**/
```

Generates 10 types x 3 operations = 30 functions from ~5 lines.

---

## 6. SIMD Kernel Templates

### Unary Operation Template

From `loops_unary_fp.dispatch.c.src`:

```c
/**begin repeat1
 * #kind = sqrt, abs, square, reciprocal, floor, ceil, trunc, rint#
 * #intr = sqrt, abs, square, recip,      floor, ceil, trunc, rint#
 */
/**begin repeat2
 * #STYPE = CONTIG, NCONTIG#
 * #DTYPE = CONTIG, NCONTIG#
 * #unroll = 4, 2#
 */
static void simd_@TYPE@_@kind@_@STYPE@_@DTYPE@(...)
{
    const int vstep = npyv_nlanes_@sfx@;

    // Unrolled SIMD loop (structure is IDENTICAL for all operations)
    for (; len >= @unroll@*vstep; len -= @unroll@*vstep) {
        npyv_@sfx@ v0 = LOAD_@STYPE@(src, 0);
        #if @unroll@ > 1
        npyv_@sfx@ v1 = LOAD_@STYPE@(src, 1);
        #endif
        // ... more loads if unroll > 2

        // THE ONLY PART THAT DIFFERS:
        npyv_@sfx@ r0 = npyv_@intr@_@sfx@(v0);
        #if @unroll@ > 1
        npyv_@sfx@ r1 = npyv_@intr@_@sfx@(v1);
        #endif

        STORE_@DTYPE@(dst, 0, r0);
        #if @unroll@ > 1
        STORE_@DTYPE@(dst, 1, r1);
        #endif
    }
    // Tail handling (shared)
}
/**end repeat2**/
/**end repeat1**/
```

This generates **8 operations x 4 layouts x 2 types = 64 functions** from one template.

### Binary Operation Template

```c
/**begin repeat1
 * #kind = add, subtract, multiply, divide#
 * #VOP = add, sub, mul, div#
 */
static void simd_binary_@kind@_@sfx@(char **args, npy_intp len)
{
    // Load (shared)
    npyv_@sfx@ a = npyv_load_@sfx@(src1);
    npyv_@sfx@ b = npyv_load_@sfx@(src2);

    // Operation (ONLY DIFFERENCE)
    npyv_@sfx@ r = npyv_@VOP@_@sfx@(a, b);

    // Store (shared)
    npyv_store_@sfx@(dst, r);
}
/**end repeat1**/
```

### Comparison Template (with pack-to-bool)

```c
/**begin repeat1
 * #kind = equal, not_equal, less, less_equal, greater, greater_equal#
 * #VOP = cmpeq, cmpneq, cmplt, cmple, cmpgt, cmpge#
 */
static void simd_compare_@kind@_@sfx@(...)
{
    // Compare (generates boolean vector)
    npyv_b@width@ c = npyv_@VOP@_@sfx@(a, b);

    // Pack to u8 (shared across all comparisons)
    #if @width@ == 32
    npyv_u8 r = npyv_pack_b8_b32(c0, c1, c2, c3);
    #elif @width@ == 64
    npyv_u8 r = npyv_pack_b8_b64(c0, c1, c2, c3, c4, c5, c6, c7);
    #endif

    npyv_store_u8(dst, npyv_and_u8(r, truemask));
}
/**end repeat1**/
```

---

## 7. Memory Overlap Detection

Location: `numpy/_core/src/common/mem_overlap.c`

### Algorithm

NumPy solves a bounded Diophantine equation:

```
sum(stride_a[i] * x_a[i]) - sum(stride_b[i] * x_b[i]) == offset

where: 0 <= x_a[i] < shape_a[i]
       0 <= x_b[i] < shape_b[i]
```

### Fast Paths

1. **Different base pointers**: Check if memory extents overlap
2. **Same array**: Check internal overlap via stride patterns
3. **Simple cases**: Direct bounds comparison

### Usage

```c
// Check before in-place operations
if (solve_may_share_memory(output, input, 1) != MEM_OVERLAP_NO) {
    // Allocate temporary, copy, operate, copy back
    temp = PyArray_Copy(input);
    operate(temp, output);
    PyArray_CopyInto(output, temp);
}
```

---

## 8. How to Add a New Operation

To add a new ufunc leveraging shared infrastructure:

### Step 1: Define Inner Loop

```c
// In loops_custom.c.src
/**begin repeat
 * #TYPE = FLOAT, DOUBLE#
 * #type = float, double#
 */
NPY_NO_EXPORT void @TYPE@_myop(
    char **args, npy_intp const *dimensions,
    npy_intp const *steps, void *func)
{
    // Use shared macro - only provide the operation
    BINARY_LOOP_FAST(@type@, @type@, *out = custom_op(in1, in2));
}
/**end repeat**/
```

### Step 2: Optional SIMD Optimization

```c
#if NPY_SIMD
static void simd_myop_@sfx@(char **args, npy_intp len)
{
    // Use shared SIMD skeleton
    const int vstep = npyv_nlanes_@sfx@;
    for (; len >= vstep; len -= vstep, ...) {
        npyv_@sfx@ a = npyv_load_@sfx@(src1);
        npyv_@sfx@ b = npyv_load_@sfx@(src2);
        npyv_@sfx@ r = /* your SIMD operation */;
        npyv_store_@sfx@(dst, r);
    }
}
#endif

NPY_NO_EXPORT void @TYPE@_myop(...) {
#if NPY_SIMD
    if (IS_BLOCKABLE_BINARY(sizeof(@type@), NPY_SIMD_WIDTH)) {
        simd_myop_@sfx@(args, dimensions[0]);
        return;
    }
#endif
    BINARY_LOOP_FAST(@type@, @type@, *out = custom_op(in1, in2));
}
```

### What You Get for Free

- Broadcasting (via NpyIter)
- Type casting (via buffered iteration)
- Memory safety (via overlap detection)
- `out=` parameter support
- Reduction support (`ufunc.reduce()`)
- Accumulate support (`ufunc.accumulate()`)
- Multi-threading coordination

---

## Summary: Infrastructure Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Python Interface                                           │
│ - Argument parsing, output allocation                               │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2: Type Dispatch (shared)                                     │
│ - Cache lookup, type resolution, casting decisions                  │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3: Iterator (shared)                                          │
│ - Broadcasting, axis reordering, buffering                          │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 4: Loop Selection (shared macros)                             │
│ - IS_BLOCKABLE_*, contiguity detection                              │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 5: SIMD Kernel (shared skeleton)                              │
│ - Peel/main/tail, unrolling, partial loads                          │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 6: Operation (5% unique code)                                 │
│ - One intrinsic or operator per operation                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scope: Which Systems Use This Infrastructure

**This shared infrastructure applies to ufuncs (Systems 1-4).** NumPy has 10 total computation systems:

### Systems Using Ufunc Infrastructure (This Document)

| System | Uses NpyIter | Uses Loop Macros | Uses SIMD Templates |
|--------|--------------|------------------|---------------------|
| 1. Ufunc Binary | Yes | Yes | Yes |
| 2. Ufunc Unary | Yes | Yes | Yes |
| 3. Ufunc Reductions | Yes | Yes | Partial (pairwise) |
| 4. Ufunc Comparisons | Yes | Yes | Yes + pack-to-bool |

### Systems with SEPARATE Infrastructure

| System | Infrastructure | Location |
|--------|----------------|----------|
| 5. Linear Algebra | BLAS dispatch | `cblasfuncs.c`, `matmul.c.src` |
| 6. Sorting | SIMD quicksort | `npysort/quicksort.cpp` |
| 7. Searching | Binary search templates | `npysort/binsearch.cpp` |
| 8. Indexing | MapIterator | `mapping.c`, `item_selection.c` |
| 9. Histogram | Direct accumulation | `compiled_base.c` |
| 10. Partition | Introselect | `npysort/selection.cpp` |

### Key Insight: ONE Kernel per Dtype with Runtime Branches

The shared infrastructure produces **ONE function per operation per dtype** with **runtime branches**:

```c
// Generated from template: ONE function, multiple branches
void DOUBLE_add(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    // Branch selection happens at RUNTIME based on steps[] values
    if (IS_BINARY_REDUCE) {
        // Pairwise reduction branch
    }
    else if (IS_BLOCKABLE_BINARY(sizeof(double), NPY_SIMD_WIDTH)) {
        // SIMD contiguous branch
    }
    else if (IS_BINARY_CONT_S1(double, double)) {
        // Scalar + array branch
    }
    else if (IS_BINARY_CONT_S2(double, double)) {
        // Array + scalar branch
    }
    else {
        // General strided branch
    }
}
```

**This is NOT multiple separate kernels - it's runtime dispatch within ONE kernel.**

### NumSharp Alignment

NumSharp's ILKernelGenerator should mirror this architecture:
- Generate ONE method per operation+dtype
- Include runtime branches for contiguity/scalar/overlap checks
- Share loop skeleton code across operations
- Only the actual operation differs (one intrinsic or operator)

See `PERFORMANCE_NUMSHARP_RECOMMENDATIONS.md` for the complete 10-system architecture and implementation priorities.
