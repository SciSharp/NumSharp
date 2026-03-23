# NumPy Optimization Strategies

This document analyzes how NumPy achieves high performance through various optimization strategies. Understanding these patterns is essential for improving NumSharp's performance.

## Overview

NumPy's performance comes from a combination of strategies working together:

| Strategy | Impact | Where Applied |
|----------|--------|---------------|
| SIMD Vectorization | 4-16x speedup | All elementwise ops |
| Memory Layout Optimization | 2-5x speedup | Iteration, reductions |
| Temporary Elision | 4-6x speedup | Chained operations |
| Pairwise Summation | 3-5x speedup | Reductions |
| BLAS Integration | 10-100x speedup | Linear algebra |
| Small Array Cache | Reduced allocation | Arrays < 1KB |

---

## 1. SIMD Vectorization

### Universal Intrinsics (npyv_*)

NumPy abstracts SIMD across platforms with a unified API:

```c
// Same code compiles to SSE, AVX2, AVX-512, NEON, VSX
npyv_f32 v = npyv_load_f32(ptr);      // Load vector
npyv_f32 r = npyv_add_f32(v1, v2);    // Vector operation
npyv_store_f32(dst, r);                // Store vector
```

**Platform Support**:
| Platform | Width | Lanes (f32) |
|----------|-------|-------------|
| SSE2 | 128-bit | 4 |
| AVX2 | 256-bit | 8 |
| AVX-512 | 512-bit | 16 |
| ARM NEON | 128-bit | 4 |
| PowerPC VSX | 128-bit | 4 |

### Multi-Target Compilation

NumPy compiles the same source multiple times for different CPU targets:

```c
// Generated at build time:
void add_AVX512_SKX(const float *src, float *dst, size_t n);
void add_AVX2(const float *src, float *dst, size_t n);
void add_SSE2(const float *src, float *dst, size_t n);  // baseline

// Runtime dispatch:
if (NPY_CPU_HAVE(AVX512_SKX)) return add_AVX512_SKX;
if (NPY_CPU_HAVE(AVX2)) return add_AVX2;
return add_SSE2;
```

### Standard SIMD Loop Pattern

All SIMD operations follow a 3-phase pattern:

```c
// Phase 1: Peel to alignment
for (i = 0; i < peel; i++) {
    dst[i] = scalar_op(src[i]);
}

// Phase 2: Main SIMD loop (4x unrolled)
for (; i < end - 4*vstep; i += 4*vstep) {
    npyv_f32 v0 = npyv_load_f32(src + i + 0*vstep);
    npyv_f32 v1 = npyv_load_f32(src + i + 1*vstep);
    npyv_f32 v2 = npyv_load_f32(src + i + 2*vstep);
    npyv_f32 v3 = npyv_load_f32(src + i + 3*vstep);

    npyv_f32 r0 = npyv_sqrt_f32(v0);
    npyv_f32 r1 = npyv_sqrt_f32(v1);
    npyv_f32 r2 = npyv_sqrt_f32(v2);
    npyv_f32 r3 = npyv_sqrt_f32(v3);

    npyv_store_f32(dst + i + 0*vstep, r0);
    npyv_store_f32(dst + i + 1*vstep, r1);
    npyv_store_f32(dst + i + 2*vstep, r2);
    npyv_store_f32(dst + i + 3*vstep, r3);
}

// Phase 3: Tail with partial vector operations
if (remaining > 0) {
    npyv_f32 v = npyv_load_till_f32(src + i, remaining, 0);
    npyv_store_till_f32(dst + i, remaining, npyv_sqrt_f32(v));
}
```

**Key aspects**:
- 4x unrolling maximizes instruction-level parallelism
- `npyv_load_till` / `npyv_store_till` handle remainders without scalar fallback
- Cleanup call (`npyv_cleanup()`) required for AVX→SSE transitions

---

## 2. Memory Management

### Small Array Cache

NumPy caches small allocations to avoid malloc overhead:

```c
#define NBUCKETS 1024      // Arrays up to 1KB
#define NCACHE 7           // 7 cached pointers per bucket

// Allocation: check cache first
if (size < NBUCKETS && cache[size].available > 0) {
    return cache[size].ptrs[--cache[size].available];
}
return malloc(size);

// Deallocation: cache if bucket not full
if (size < NBUCKETS && cache[size].available < NCACHE) {
    cache[size].ptrs[cache[size].available++] = ptr;
} else {
    free(ptr);
}
```

### Temporary Elision

For chained operations like `a + b + c`, NumPy detects when intermediate results can be reused:

**Conditions for elision**:
| Condition | Check |
|-----------|-------|
| Unique reference | `refcount == 1` |
| Large enough | `size >= 256KB` |
| Owns data | `OWNDATA` flag set |
| Writeable | `WRITEABLE` flag set |
| Same/safe dtype | Safe casting allowed |
| From interpreter | Backtrace validates caller |

**Result**: `a + b + c + d` allocates once instead of three times (4-6x speedup).

### Huge Pages

For arrays >= 4MB, NumPy advises the kernel to use huge pages:

```c
if (size >= (1u << 22u)) {  // 4MB
    madvise(ptr, size, MADV_HUGEPAGE);
}
```

This reduces TLB misses for large array traversals.

---

## 3. Pairwise Summation

For floating-point reductions, NumPy uses pairwise summation for numerical stability:

```c
#define PW_BLOCKSIZE 128

double pairwise_sum(double *a, size_t n, size_t stride) {
    if (n < 8) {
        // Direct accumulation
        double r = -0.0;  // Preserves -0 semantics
        for (i = 0; i < n; i++) r += a[i * stride];
        return r;
    }
    else if (n <= PW_BLOCKSIZE) {
        // 8 independent accumulators
        double r[8];
        // Initialize r[0..7] from first 8 elements

        for (i = 8; i < n - (n % 8); i += 8) {
            NPY_PREFETCH(a + (i + 64) * stride, 0, 3);  // L1 prefetch
            r[0] += a[(i+0) * stride];
            r[1] += a[(i+1) * stride];
            // ... r[2..7]
        }

        // Tree reduction preserves pairing
        return ((r[0]+r[1]) + (r[2]+r[3])) + ((r[4]+r[5]) + (r[6]+r[7]));
    }
    else {
        // Recursive split
        size_t n2 = (n / 2) - ((n / 2) % 8);  // Keep aligned to 8
        return pairwise_sum(a, n2, stride) +
               pairwise_sum(a + n2*stride, n - n2, stride);
    }
}
```

**Benefits**:
- O(log n) rounding error instead of O(n)
- 8 accumulators enable superscalar execution
- Prefetch hints improve cache behavior
- Block size (128) fits in L1 cache

---

## 4. BLAS Integration

For linear algebra, NumPy dispatches to optimized BLAS libraries:

### Tiered Dispatch

| Operation | BLAS Level | Function |
|-----------|------------|----------|
| Inner product | Level 1 | `cblas_ddot` |
| Matrix-vector | Level 2 | `cblas_dgemv` |
| Matrix-matrix | Level 3 | `cblas_dgemm` |
| Symmetric A@A.T | Level 3 | `cblas_dsyrk` |

### Blasable Detection

```c
bool is_blasable2d(PyArrayObject *arr) {
    // Inner dimension must be contiguous
    if (strides[1] != itemsize) return false;

    // Outer stride must be reasonable
    if (strides[0] < shape[1] * itemsize) return false;
    if (strides[0] > BLAS_MAXSIZE) return false;

    return true;
}
```

### Symmetric Optimization

When computing `A @ A.T`, NumPy detects this pattern and uses `syrk`:

```c
if (ip1 == ip2 && same_shape && opposite_transpose) {
    // Only compute upper triangle, then mirror
    cblas_dsyrk(CblasRowMajor, CblasUpper, trans, n, k,
                alpha, A, lda, beta, C, ldc);
    // Mirror upper to lower
}
```

This halves the computation for symmetric results.

---

## 5. Division Optimization

### Division by Constant

Integer division by a constant uses multiply-high trick:

```c
// Precompute magic numbers once
npyv_s32x3 divisor = npyv_divisor_s32(scalar);

// In loop: multiply-high instead of divide
for (; len >= vstep; len -= vstep) {
    npyv_s32 a = npyv_load_s32(src);
    npyv_s32 q = npyv_divc_s32(a, divisor);  // No actual division!
    npyv_store_s32(dst, q);
}
```

The `npyv_divc_s32` uses the magic multiplier + shifts instead of hardware division.

### Edge Case Handling

```c
// Avoid hardware exceptions
if (divisor == 0) {
    npy_set_floatstatus_divbyzero();
    return 0;  // Don't crash
}
if (dividend == MIN_INT && divisor == -1) {
    npy_set_floatstatus_overflow();
    return MIN_INT;  // Avoid x86 SIGFPE
}
```

---

## 6. Compiler Hints

NumPy uses various compiler hints to enable better optimization:

### Loop Optimization

```c
// Tell compiler no loop-carried dependencies
#if __GNUC__ >= 6
#define IVDEP_LOOP _Pragma("GCC ivdep")
#endif

IVDEP_LOOP
for (i = 0; i < n; i++) {
    dst[i] = src1[i] + src2[i];
}
```

### Branch Prediction

```c
#define NPY_LIKELY(x)   __builtin_expect(!!(x), 1)
#define NPY_UNLIKELY(x) __builtin_expect(!!(x), 0)

if (NPY_UNLIKELY(divisor == 0)) {
    handle_error();
}
```

### Force Optimization

```c
// Force O3 for hot functions
#define NPY_GCC_OPT_3 __attribute__((optimize("O3")))

NPY_GCC_OPT_3
static void hot_inner_loop(...) { ... }
```

### Prefetch

```c
// Prefetch 512 bytes ahead into L1 cache
NPY_PREFETCH(ptr + 64, 0, 3);  // 0=read, 3=L1 locality
```

---

## 7. Axis Reduction Optimization

### Axis Reordering

NumPy sorts axes by stride for better cache locality:

```c
// Before: sum along axis 0 of C-order array
// shape=[1000, 1000], strides=[8000, 8] (row-major)
// Naive: 1000 cache misses per column

// After reordering: inner loop on contiguous axis
// Process rows (stride=8) in inner loop
// 1000x fewer cache misses
```

### Axis Coalescing

Adjacent axes with compatible strides are merged:

```c
// shape=[100, 100, 100], strides=[80000, 800, 8]
// All contiguous: coalesce to shape=[1000000], strides=[8]
// Single large SIMD reduction instead of nested loops
```

---

## Summary: Optimization Checklist

When implementing NumSharp operations, verify:

1. **SIMD**: Is the hot path vectorized with 4x unrolling?
2. **Contiguity**: Is there a fast path for contiguous arrays?
3. **Scalar broadcast**: Is scalar+array optimized (broadcast once)?
4. **Reductions**: Using pairwise summation with 8 accumulators?
5. **Allocation**: Can temporaries be elided? Small array cache?
6. **Edge cases**: Division by zero, overflow handled without crash?
7. **Cache**: Axes reordered for locality? Prefetch for large arrays?

---

## Key Files in NumPy Source

| Area | File |
|------|------|
| SIMD abstraction | `numpy/_core/src/common/simd/` |
| Loop macros | `numpy/_core/src/umath/fast_loop_macros.h` |
| Pairwise sum | `numpy/_core/src/umath/loops_utils.h.src` |
| BLAS dispatch | `numpy/_core/src/common/cblasfuncs.c` |
| Memory alloc | `numpy/_core/src/multiarray/alloc.c` |
| Temp elision | `numpy/_core/src/multiarray/temp_elide.c` |

---

## Optimization Strategy by Computation System

NumPy has **10 core computation systems**. Here's which optimizations apply to each:

| System | SIMD | Pairwise | BLAS | Cache Reorder | Temp Elision |
|--------|------|----------|------|---------------|--------------|
| 1. Ufunc Binary | Yes (4x unroll) | Reduce only | - | - | Yes |
| 2. Ufunc Unary | Yes (4x unroll) | - | - | - | Yes |
| 3. Ufunc Reductions | Yes | **Yes** | - | **Yes** | - |
| 4. Ufunc Comparisons | Yes + pack-to-bool | - | - | - | Yes |
| 5. Linear Algebra | - | - | **Yes** | - | - |
| 6. Sorting | Yes (x86-simd-sort) | - | - | - | - |
| 7. Searching | - | - | - | - | - |
| 8. Indexing | Partial | - | - | - | - |
| 9. Histogram | - | - | - | - | - |
| 10. Partition | - | - | - | - | - |

### Key Insight: ONE Kernel with Runtime Branches

Each system generates **ONE kernel per dtype** with **runtime branches** for different memory layouts:

```
Binary kernel (e.g., DOUBLE_add):
├─ Branch 1: IS_REDUCE → pairwise accumulator
├─ Branch 2: IS_CONTIGUOUS → SIMD 4x unrolled loop
├─ Branch 3: IS_SCALAR1 → broadcast first operand
├─ Branch 4: IS_SCALAR2 → broadcast second operand
└─ Branch 5: GENERAL → scalar strided loop
```

**This is NOT multiple separate kernels - it's ONE function with if/else branches.**

### NumSharp Alignment

NumSharp's ILKernelGenerator should:
1. Generate ONE method per operation+dtype
2. Include runtime branches matching NumPy's patterns
3. Apply appropriate optimizations per branch

See `PERFORMANCE_NUMSHARP_RECOMMENDATIONS.md` for the complete 10-system architecture.
