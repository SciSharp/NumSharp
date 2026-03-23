# NumPy Execution Paths

This document details the different execution paths NumPy operations can take, when each path is selected, and what optimizations apply to each.

## Overview: Path Selection

Every NumPy operation selects from multiple execution paths based on input characteristics:

```
                              Input Analysis
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
       Contiguous?            Scalar operand?          Overlap?
            │                       │                       │
            ▼                       ▼                       ▼
    ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
    │ SIMD Fast Path│      │ Scalar Broad- │      │ Copy-Operate  │
    │ (4x unrolled) │      │ cast Path     │      │ Path          │
    └───────────────┘      └───────────────┘      └───────────────┘
```

---

## 1. Binary Elementwise Operations (add, sub, mul, div)

### Path Decision Tree

```
np.add(a, b):
│
├─ IS_BINARY_REDUCE? (accumulator: a += b)
│   └─ YES: Pairwise summation path
│       ├─ 8 accumulators
│       ├─ Prefetch 512B ahead
│       └─ O(log n) rounding error
│
├─ Memory overlap with output?
│   └─ YES: Allocate temp, operate, copy back
│
├─ SIMD eligible? (NPY_SIMD && len > 2*vwidth)
│   │
│   ├─ Both contiguous? (IS_BINARY_CONT)
│   │   └─ **Path A: SIMD Contiguous**
│   │       ├─ 2x unrolled vector loop
│   │       ├─ Masked tail handling
│   │       └─ ~8x faster than scalar
│   │
│   ├─ First operand scalar? (IS_BINARY_CONT_S1)
│   │   └─ **Path B: SIMD Scalar+Array**
│   │       ├─ Broadcast scalar to vector once
│   │       ├─ 2x unrolled vector loop
│   │       └─ ~6x faster than scalar
│   │
│   ├─ Second operand scalar? (IS_BINARY_CONT_S2)
│   │   └─ **Path C: SIMD Array+Scalar**
│   │       └─ Same as Path B
│   │
│   └─ None of above
│       └─ **Path D: Scalar Strided**
│
└─ SIMD not available
    └─ **Path D: Scalar Strided**
        ├─ Generic loop with arbitrary strides
        └─ No vectorization
```

### Path Characteristics

| Path | Conditions | Unroll | SIMD | Speed |
|------|------------|--------|------|-------|
| A: SIMD Contiguous | `steps == sizeof(type)` | 2x | Yes | 1.0x (baseline) |
| B: SIMD Scalar+Array | `steps[0]==0`, others contiguous | 2x | Yes | 1.1x |
| C: SIMD Array+Scalar | `steps[1]==0`, others contiguous | 2x | Yes | 1.1x |
| D: Scalar Strided | Any other case | None | No | 4-8x slower |
| Reduce | Accumulator pattern | 8 accum | Partial | Varies |

---

## 2. Unary Elementwise Operations (sqrt, sin, abs)

### Path Decision Tree

```
np.sqrt(a):
│
├─ Memory overlap (in-place)?
│   └─ YES: **Path E: No-Unroll**
│       └─ Single-element SIMD for FP consistency
│
├─ Loadable stride? (stride fits in gather)
│   └─ NO: **Path E: No-Unroll**
│
├─ SIMD eligible?
│   │
│   ├─ Input contiguous, output contiguous?
│   │   └─ **Path F: CONTIG_CONTIG**
│   │       ├─ 4x unrolled vector loop
│   │       ├─ npyv_load_till for tail
│   │       └─ Maximum throughput
│   │
│   ├─ Input strided, output contiguous?
│   │   └─ **Path G: NCONTIG_CONTIG**
│   │       ├─ 4x unrolled with gather loads
│   │       └─ Slower due to gather
│   │
│   ├─ Input contiguous, output strided?
│   │   └─ **Path H: CONTIG_NCONTIG**
│   │       ├─ 2x unrolled with scatter stores
│   │       └─ Scatter is expensive
│   │
│   └─ Both strided?
│       └─ **Path I: NCONTIG_NCONTIG**
│           ├─ 2x unrolled gather/scatter
│           └─ Slowest SIMD path
│
└─ SIMD not available
    └─ **Path J: Scalar**
        └─ c_sqrt() or npy_sqrt()
```

### Path Characteristics

| Path | In Stride | Out Stride | Unroll | Notes |
|------|-----------|------------|--------|-------|
| F: CONTIG_CONTIG | 1 | 1 | 4x | Best path |
| G: NCONTIG_CONTIG | N | 1 | 4x | Gather loads |
| H: CONTIG_NCONTIG | 1 | N | 2x | Scatter stores |
| I: NCONTIG_NCONTIG | N | N | 2x | Both gather/scatter |
| E: No-Unroll | Any | Any | None | Overlap safety |
| J: Scalar | Any | Any | None | Fallback |

**Note**: NumPy forces SSE128 even on AVX512 for some unary ops because gather/scatter instructions are too expensive at wider widths.

---

## 3. Reduction Operations (sum, prod, min, max)

### Path Decision Tree

```
np.sum(a, axis=1):
│
├─ axis=None? (full array reduction)
│   │
│   └─ IS_BINARY_REDUCE && contiguous?
│       └─ **Path K: Pairwise Summation**
│           ├─ Recursive divide-and-conquer
│           ├─ Block size 128
│           ├─ 8 independent accumulators
│           ├─ Prefetch 512B ahead
│           └─ O(log n) rounding error
│
├─ Axis reduction (axis=0, 1, ...)?
│   │
│   ├─ Contiguous along reduction axis?
│   │   └─ **Path L: Contiguous Axis Reduce**
│   │       ├─ May use SIMD for inner reduction
│   │       └─ Iterator handles outer dims
│   │
│   └─ Non-contiguous reduction axis?
│       └─ **Path M: Iterator Reduce**
│           ├─ NpyIter with EXTERNAL_LOOP
│           ├─ Axis reordering for cache
│           └─ Axis coalescing when possible
│
└─ Has identity? (0 for sum, 1 for prod)
    ├─ YES: Initialize result to identity
    └─ NO: Copy first element, skip in loop
```

### Pairwise Summation Detail

```
n < 8:      Scalar loop (r = -0.0 init for sign preservation)
n <= 128:   8-way unrolled, tree reduction at end
n > 128:    Recursive split at n/2 (aligned to 8)

                    pairwise_sum(0..n)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      pairwise_sum(0..n/2)    pairwise_sum(n/2..n)
              │                         │
         [8 accum]                 [8 accum]
              │                         │
    ((r0+r1)+(r2+r3))         ((r0+r1)+(r2+r3))
    +((r4+r5)+(r6+r7))        +((r4+r5)+(r6+r7))
```

---

## 4. Comparison Operations (<, >, ==, !=)

### Path Decision Tree

```
np.greater(a, b):
│
└─ Implemented as: np.less(b, a)  // Swap arguments!

np.less(a, b):
│
├─ Memory overlap?
│   └─ YES: Scalar fallback
│
├─ SIMD eligible?
│   │
│   ├─ Scalar + Array? (IS_BINARY_CONT_S1)
│   │   └─ **Path N: Scalar Broadcast Compare**
│   │
│   ├─ Array + Scalar? (IS_BINARY_CONT_S2)
│   │   └─ **Path O: Array Scalar Compare**
│   │
│   └─ Both contiguous? (IS_BINARY_CONT)
│       └─ **Path P: SIMD Compare**
│           ├─ Vector comparison → boolean vector
│           ├─ Pack to u8 (see below)
│           └─ Unroll factor depends on type width
│
└─ Fallback
    └─ **Path Q: Scalar Compare**
```

### Pack-to-Bool Pattern

Comparisons produce wide boolean vectors that must be packed to u8:

```c
// 64-bit types: 8x unroll to fill one u8 vector
npyv_b64 c0 = npyv_cmplt_f64(a0, b0);  // 4 bools in 256 bits
npyv_b64 c1 = npyv_cmplt_f64(a1, b1);
// ... c2..c7
npyv_u8 packed = npyv_pack_b8_b64(c0, c1, c2, c3, c4, c5, c6, c7);

// 32-bit types: 4x unroll
npyv_u8 packed = npyv_pack_b8_b32(c0, c1, c2, c3);

// 16-bit types: 2x unroll
npyv_u8 packed = npyv_pack_b8_b16(c0, c1);

// 8-bit types: 1x (already u8 width)
npyv_u8 packed = npyv_cvt_u8_b8(c0);
```

| Input Type | Unroll Factor | Why |
|------------|---------------|-----|
| int8/uint8 | 1x | Already byte-width booleans |
| int16/uint16 | 2x | Pack 2 comparisons → 1 u8 vector |
| int32/uint32/float32 | 4x | Pack 4 comparisons → 1 u8 vector |
| int64/uint64/float64 | 8x | Pack 8 comparisons → 1 u8 vector |

---

## 5. Matrix Multiplication (matmul, dot)

### Path Decision Tree

```
np.matmul(a, b):
│
├─ BLAS available && (float32|float64|complex)?
│   │
│   ├─ Too big or zero dimension?
│   │   └─ **Path R: No-BLAS Triple Loop**
│   │
│   ├─ Special shapes?
│   │   │
│   │   ├─ Scalar output (1xN @ Nx1)?
│   │   │   └─ **Path S: BLAS Dot**
│   │   │       └─ cblas_ddot / cblas_sdot
│   │   │
│   │   ├─ Matrix @ Vector?
│   │   │   └─ **Path T: BLAS GEMV**
│   │   │       └─ cblas_dgemv / cblas_sgemv
│   │   │
│   │   └─ Vector @ Matrix?
│   │       └─ **Path T: BLAS GEMV** (transposed)
│   │
│   ├─ A @ A.T detected? (same array, transposed)
│   │   └─ **Path U: BLAS SYRK**
│   │       ├─ Only compute upper triangle
│   │       ├─ Mirror to lower
│   │       └─ ~50% fewer operations
│   │
│   ├─ Both BLAS-able layout?
│   │   └─ **Path V: BLAS GEMM**
│   │       └─ cblas_dgemm / cblas_sgemm
│   │
│   └─ Need to copy for BLAS?
│       └─ **Path W: Copy + GEMM**
│           ├─ Copy non-contiguous to temp
│           └─ Then GEMM
│
└─ No BLAS (integer, half, object)
    └─ **Path R: Triple Loop**
```

### BLAS-Ability Check

```c
bool is_blasable2d(array) {
    // Inner (fast) dimension must be contiguous
    if (strides[1] != itemsize) return false;

    // Outer stride must be >= inner size
    if (strides[0] < shape[1] * itemsize) return false;

    // Stride must fit in BLAS integer
    if (strides[0] > BLAS_MAXSIZE) return false;

    return true;
}
```

---

## 6. Edge Case Paths

### Memory Overlap Path

When output overlaps input:

```
1. Detect overlap (Diophantine solver, max_work=1)
2. If overlap:
   a. Allocate temporary array
   b. Operate into temporary
   c. Copy temporary to output (WRITEBACKIFCOPY)
3. Cost: +100% memory, +50% time
```

### Type Casting Path

When dtypes don't match:

```
1. Allocate buffer (size = min(8192, array_size))
2. For each buffer chunk:
   a. Cast input chunk to buffer
   b. Operate on buffer
   c. Cast buffer to output
3. Cost: +copy overhead per buffer
```

### Empty Array Path

When `size == 0`:

```
1. Check NPY_ITER_ZEROSIZE_OK flag
2. If reduction without identity:
   → Raise ValueError
3. Else:
   → Return empty array (no computation)
```

### Division Edge Cases

```c
// Division by zero
if (divisor == 0) {
    npy_set_floatstatus_divbyzero();
    return 0;  // Don't crash
}

// Overflow case (MIN_INT / -1)
if (dividend == MIN_INT && divisor == -1) {
    npy_set_floatstatus_overflow();
    return MIN_INT;  // Avoid x86 SIGFPE
}
```

### NaN Handling Paths

| Function | NaN Behavior | Implementation |
|----------|--------------|----------------|
| `max/min` | Propagate | `(a >= b \|\| isnan(a)) ? a : b` |
| `fmax/fmin` | Ignore | Use `fmax()/fmin()` C functions |
| `nanmax/nanmin` | Skip | Filter NaN before reduction |

---

## Path Selection Summary Table

| Operation | # Paths | Fast Path Condition | Slowest Path |
|-----------|---------|---------------------|--------------|
| Binary (add) | 5 | Both contiguous | Strided (4-8x slower) |
| Unary (sqrt) | 6 | Both contiguous | Gather+scatter (3x slower) |
| Reduction (sum) | 4 | Contiguous, axis=None | Iterator (2x slower) |
| Comparison | 4 | Both contiguous | Strided (4x slower) |
| Matmul | 7 | BLAS-able layout | Triple loop (10-100x slower) |

---

## Critical Clarification: Runtime Branches, Not Separate Kernels

**The paths described above are NOT separate compiled kernels.** They are **runtime branches within ONE kernel per dtype**.

### NumPy's Actual Structure

```c
// ONE function with if/else branches - NOT multiple functions
void DOUBLE_add(char **args, npy_intp *dimensions, npy_intp *steps, void *func)
{
    // Path selection happens HERE at RUNTIME via stride checks
    if (IS_BINARY_REDUCE) {
        // Branch 1: Pairwise reduction
    }
    else if (IS_BLOCKABLE_BINARY(sizeof(double), NPY_SIMD_WIDTH)) {
        // Branch 2: SIMD contiguous
    }
    else if (IS_BINARY_CONT_S1(double, double)) {
        // Branch 3: Scalar + array
    }
    else if (IS_BINARY_CONT_S2(double, double)) {
        // Branch 4: Array + scalar
    }
    else {
        // Branch 5: General strided
    }
}
```

### Implications for NumSharp

NumSharp's ILKernelGenerator should generate **ONE method per operation+dtype** with runtime branches:

```csharp
// Generated IL method - ONE method with branches
void Generated_Add_Float64(double* lhs, double* rhs, double* result,
                           int lhsStride, int rhsStride, int resultStride, int length)
{
    // Runtime branch selection (mirrors NumPy's if/else)
    if (lhsStride == 0 && lhs == result) {
        // Pairwise reduction branch
    }
    else if (lhsStride == 1 && rhsStride == 1 && resultStride == 1) {
        // SIMD contiguous branch
    }
    else if (lhsStride == 0 && rhsStride == 1) {
        // Scalar + array branch
    }
    else if (rhsStride == 0 && lhsStride == 1) {
        // Array + scalar branch
    }
    else {
        // General strided branch
    }
}
```

---

## 7. Sorting (sort, argsort)

### Path Decision Tree

```
np.sort(a):
│
├─ SIMD dispatch available?
│   │
│   ├─ x86 platform?
│   │   ├─ sizeof(T) == 2 (16-bit)?
│   │   │   └─ **Path: x86_simd_qsort_16bit**
│   │   └─ sizeof(T) == 4 or 8 (32/64-bit)?
│   │       └─ **Path: x86_simd_qsort** (AVX512)
│   │
│   └─ ARM/other platform?
│       └─ **Path: highway_qsort** (portable SIMD)
│
├─ SIMD dispatch returns false → Introsort
│   │
│   ├─ Recursion depth exceeded (cdepth < 0)?
│   │   └─ **Path: Heapsort** (O(n log n) guaranteed)
│   │
│   ├─ Partition size > SMALL_QUICKSORT (15)?
│   │   └─ **Path: Quicksort partition**
│   │       ├─ (pi - pl) < (pr - pi) → push right, recurse left
│   │       └─ (pi - pl) >= (pr - pi) → push left, recurse right
│   │
│   └─ Partition size <= 15?
│       └─ **Path: Insertion sort**
│
├─ String type?
│   └─ **Path: string_quicksort_** (length-aware)
│
└─ Generic type (custom comparator)?
    └─ **Path: npy_quicksort_impl** with PyArray_CompareFunc
```

### Branch Count: 9

| Branch | Condition | Algorithm |
|--------|-----------|-----------|
| 1 | x86 + 16-bit | SIMD quicksort (16-bit) |
| 2 | x86 + 32/64-bit | SIMD quicksort (AVX512) |
| 3 | ARM/other | Highway SIMD quicksort |
| 4 | Depth exceeded | Heapsort fallback |
| 5 | Large partition | Quicksort partition |
| 6 | Small partition (≤15) | Insertion sort |
| 7 | String type | String-aware quicksort |
| 8 | Generic type | Comparator-based sort |
| 9 | Cygwin | SIMD disabled entirely |

---

## 8. Searching (searchsorted, nonzero)

### Path Decision Tree

```
np.searchsorted(sorted_arr, values, side='left'):
│
├─ side == 'left'?
│   └─ **Path: Binary search with < comparator**
│
├─ side == 'right'?
│   └─ **Path: Binary search with <= comparator**
│
├─ Empty key array (key_len == 0)?
│   └─ **Path: Early return** (nothing to search)
│
├─ Sorted key optimization?
│   ├─ last_key < current_key?
│   │   └─ Adjust min_idx based on previous result
│   └─ Skip binary search portion
│
├─ Has sorter array (indirect search)?
│   └─ **Path: argbinsearch**
│       ├─ Validate index bounds
│       └─ Indirect array access
│
└─ Generic type?
    └─ **Path: npy_binsearch** with PyArray_CompareFunc
```

### np.nonzero(a)

```
np.nonzero(a):
│
├─ Count nonzero elements (first pass)
│   └─ Iterate, count where a[i] != 0
│
├─ Allocate output arrays (one per dimension)
│   └─ Shape: (count,) × ndim
│
└─ Fill indices (second pass)
    └─ Iterate, store coordinates where a[i] != 0
```

### Branch Count: 7

| Branch | Condition | Operation |
|--------|-----------|-----------|
| 1 | side='left' | Use `<` comparator |
| 2 | side='right' | Use `<=` comparator |
| 3 | Empty keys | Early return |
| 4 | Sorted keys | Skip redundant search |
| 5 | Has sorter | Indirect search |
| 6 | Invalid index | Return -1 (error) |
| 7 | Generic type | Custom comparator |

---

## 9. Indexing (boolean, fancy, take/put)

### Path Decision Tree

```
a[index]:
│
├─ Structured array field access?
│   └─ **Path: _get_field_view**
│
├─ Full integer index (all dims specified)?
│   └─ **Path: Scalar return**
│       └─ get_item_pointer + PyArray_Scalar
│
├─ Boolean array index?
│   └─ **Path: Boolean subscript**
│       ├─ count_boolean_trues (first pass)
│       ├─ Allocate 1D result
│       └─ Copy matching elements (second pass)
│
├─ Single ellipsis (...)?
│   └─ **Path: View return** (PyArray_View)
│
├─ Slices/newaxis/ellipsis/integer combo?
│   └─ **Path: get_view_from_index**
│       ├─ HAS_SCALAR_ARRAY → PyArray_NewCopy
│       └─ !HAS_FANCY → return view
│
├─ Simple 1D fancy (single index array)?
│   └─ Check trivial conditions:
│       ├─ TRIVIALLY_ITERABLE?
│       ├─ ITEMSIZE == sizeof(intp)?
│       ├─ Integer kind + aligned?
│       └─ → **Path: mapiter_trivial_get** (optimized)
│
├─ Complex fancy indexing?
│   └─ **Path: PyArray_MapIterNew**
│       ├─ Multiple index arrays → MapIterCheckIndices
│       ├─ subspace_iter != NULL → subspace iteration
│       └─ subspace_iter == NULL → direct copy
│
└─ Assignment (a[index] = value)?
    ├─ Integer → get_item_pointer + PyArray_Pack
    ├─ Boolean → array_assign_boolean_subscript
    ├─ Ellipsis → self==op check, PyArray_CopyObject
    ├─ Subclass → PyObject_GetItem
    └─ Fancy → mapiter_set
```

### Branch Count: 15

| Branch | Condition | Operation |
|--------|-----------|-----------|
| 1 | Field access | Structured array field |
| 2 | Full integer | Return scalar |
| 3 | Boolean array | Count + gather |
| 4 | Single ellipsis | Return view |
| 5 | Slice combo | Strided view |
| 6 | Scalar array | Copy required |
| 7 | Simple 1D fancy | Trivial mapiter |
| 8 | Complex fancy | Full MapIterNew |
| 9 | Subspace iter | Subspace iteration |
| 10 | Direct copy | No subspace |
| 11 | Assign integer | Pack scalar |
| 12 | Assign boolean | Boolean scatter |
| 13 | Assign ellipsis | Full copy |
| 14 | Assign subclass | Python protocol |
| 15 | Assign fancy | mapiter_set |

---

## 10. Histogram/Bincount

### Path Decision Tree

```
np.bincount(x, weights=None, minlength=0):
│
├─ Empty input (len == 0)?
│   └─ **Path: Return zeros(minlength)**
│
├─ Negative values in x?
│   └─ **Path: Raise ValueError**
│
├─ minlength handling
│   ├─ minlength == None → Error (use 0)
│   ├─ minlength specified → ans_size = max(max(x)+1, minlength)
│   └─ minlength not specified → ans_size = max(x) + 1
│
├─ weights == None (unweighted)?
│   └─ **Path: Integer accumulation**
│       └─ for (i=0; i<len; i++) result[x[i]] += 1
│
├─ weights provided?
│   └─ **Path: Weighted accumulation**
│       ├─ Validate weights.size == x.size
│       └─ for (i=0; i<len; i++) result[x[i]] += weights[i]
│
└─ Input type checking
    ├─ PyArray_Check → direct processing
    ├─ !PyArray_Check → PyArray_FromAny conversion
    ├─ IsInteger → NPY_ARRAY_FORCECAST
    └─ Non-integer → deprecation warning
```

### Branch Count: 8

| Branch | Condition | Operation |
|--------|-----------|-----------|
| 1 | Empty input | Return zeros |
| 2 | Negative values | Raise error |
| 3 | minlength=None | Error |
| 4 | minlength specified | Adjust size |
| 5 | Unweighted | Integer accumulate |
| 6 | Weighted | Float accumulate |
| 7 | Non-array input | Convert first |
| 8 | Non-integer input | Warning + cast |

---

## 11. Partition/Selection

### Path Decision Tree

```
np.partition(a, kth):
│
├─ Introselect algorithm
│   │
│   ├─ Partition size small?
│   │   └─ **Path: Insertion sort** (finish)
│   │
│   ├─ Recursion depth exceeded?
│   │   └─ **Path: Heapselect** (O(n) guaranteed)
│   │
│   └─ Normal case
│       └─ **Path: Quickselect partition**
│           ├─ Partition around pivot
│           ├─ kth in left partition? → recurse left
│           ├─ kth in right partition? → recurse right
│           └─ kth at pivot? → done
│
├─ Multiple kth values?
│   └─ Sort kth, then select each in order
│
└─ np.argpartition?
    └─ Same as above but track indices
```

### Branch Count: 5 (separate from sorting)

| Branch | Condition | Operation |
|--------|-----------|-----------|
| 1 | Small partition | Insertion sort |
| 2 | Depth exceeded | Heapselect |
| 3 | kth in left | Recurse left |
| 4 | kth in right | Recurse right |
| 5 | kth at pivot | Return |

---

## Complete Branch Summary (All 10 Systems)

| System | Branches | Key Decision Points |
|--------|----------|---------------------|
| 1. Binary Elementwise | 10 | SIMD, scalar, aliasing |
| 2. Unary Elementwise | 8 | 4 layouts, overlap, stride |
| 3. Reductions | 9 | Pairwise, axis, masked, identity |
| 4. Comparisons | 4+ | Pack-to-bool varies by width |
| 5. Linear Algebra | 12 | BLAS level, layout, syrk |
| 6. Sorting | 9 | SIMD, introsort, insertion |
| 7. Searching | 7 | Side, sorter, generic |
| 8. Indexing | 15 | Boolean, fancy, trivial |
| 9. Histogram | 8 | Empty, weighted, types |
| 10. Partition | 5 | Select vs sort paths |
| **TOTAL** | **~87** | |

---

## NumSharp Implementation Status

**This document covers ALL 10 NumPy computation systems with ~87 total runtime branches.**

| System | Status | Notes |
|--------|--------|-------|
| 1. Binary Elementwise | DONE | Missing: pairwise reduce, scalar broadcast |
| 2. Unary Elementwise | DONE | Missing: strided SIMD branches |
| 3. Reductions | DONE | Missing: pairwise algorithm, axis reorder |
| 4. Comparisons | DONE | Missing: scalar broadcast branches |
| 5. Linear Algebra | Partial | Basic matmul done, einsum missing |
| 6. Sorting | **MISSING** | HIGH PRIORITY |
| 7. Searching | DONE | searchsorted, argmax, argmin, nonzero |
| 8. Indexing | DONE | Boolean masking, fancy indexing, take |
| 9. Histogram | **MISSING** | MEDIUM PRIORITY |
| 10. Partition | **MISSING** | MEDIUM PRIORITY |

See `PERFORMANCE_NUMSHARP_RECOMMENDATIONS.md` for implementation priorities and code examples.
