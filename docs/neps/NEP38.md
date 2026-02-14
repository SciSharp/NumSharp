# NEP 38 - Using SIMD Optimization Instructions for Performance

**Status:** Final
**NumSharp Impact:** HIGH - Relevant to issues #544, #545 (SIMD optimization)

## Summary

Defines NumPy's SIMD infrastructure using "universal intrinsics" that abstract SIMD operations across CPU architectures, with runtime dispatch to select optimal implementations.

## Three-Stage Mechanism

1. **Infrastructure:** Abstract intrinsics in code, extending ufunc machinery
2. **Compile-time:** Compiler macros convert abstract intrinsics to concrete calls
3. **Runtime:** CPU detection selects optimal loop for each ufunc

## Universal Intrinsics Concept

Abstract SIMD operations that map to platform-specific implementations:

| Universal Intrinsic | ARM NEON | x86 AVX2 | x86 AVX-512 |
|---------------------|----------|----------|-------------|
| `npyv_load_u32` | `vld1q_u32` | `_mm256_loadu_si256` | `_mm512_loadu_si512` |
| `npyv_add_f32` | `vaddq_f32` | `_mm256_add_ps` | `_mm512_add_ps` |
| `npyv_mul_f64` | `vmulq_f64` | `_mm256_mul_pd` | `_mm512_mul_pd` |

## Supported Instruction Sets

### x86_64 (Default baseline: SSE3)
```
SSE3, SSSE3, SSE41, SSE42, POPCNT, AVX, F16C, XOP, FMA4, FMA3,
AVX2, AVX512F, AVX512CD, AVX512_KNL, AVX512_KNM, AVX512_SKX,
AVX512_CLX, AVX512_CNL, AVX512_ICL
```

### Other Architectures
- **ARM:** NEON
- **PowerPC:** VSX
- **s390x:** VX/VXE/VXE2

## Runtime Dispatch

### CPU Feature Detection
```c
// C-level boolean array
extern bool npy__cpu_have[NPY_CPU_FEATURE_MAX];

// Query function
bool npy_cpu_have(int feature_id);
```

### Python API
```python
import numpy as np
np.__cpu_features__  # Dict of feature → bool
# {'SSE': True, 'SSE2': True, 'AVX2': True, 'AVX512F': False, ...}
```

### Loop Selection
At import time, NumPy selects the best available loop for each ufunc based on runtime CPU capabilities.

## Build Configuration

### `--cpu-baseline`
Minimum required features (default x86_64: SSE3)

### `--cpu-dispatch`
Additional features to compile dispatch variants for

```bash
# Build with specific dispatch targets
python setup.py build --cpu-dispatch="AVX2 AVX512F"
```

## NumSharp Relevance

### Current State
NumSharp uses scalar loops. SIMD would significantly improve:
- Element-wise operations (+, -, *, /)
- Reductions (sum, mean, min, max)
- Comparisons
- Mathematical functions

### Implementation Options for C#

#### Option 1: System.Numerics.Vector<T>
```csharp
// .NET's portable SIMD
using System.Numerics;

public static void Add(float[] a, float[] b, float[] result) {
    int simdLength = Vector<float>.Count;
    int i = 0;
    for (; i <= a.Length - simdLength; i += simdLength) {
        var va = new Vector<float>(a, i);
        var vb = new Vector<float>(b, i);
        (va + vb).CopyTo(result, i);
    }
    // Scalar remainder
    for (; i < a.Length; i++) {
        result[i] = a[i] + b[i];
    }
}
```

#### Option 2: System.Runtime.Intrinsics (.NET Core 3.0+)
```csharp
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

public static unsafe void AddAvx(float* a, float* b, float* result, int length) {
    if (Avx.IsSupported) {
        int i = 0;
        for (; i <= length - 8; i += 8) {
            var va = Avx.LoadVector256(a + i);
            var vb = Avx.LoadVector256(b + i);
            Avx.Store(result + i, Avx.Add(va, vb));
        }
        // Scalar remainder
    }
}
```

#### Option 3: Native Libraries
- Intel MKL via P/Invoke
- OpenBLAS
- Custom C/C++ with SIMD

### Priority Operations for SIMD

| Operation | Potential Speedup | Complexity |
|-----------|-------------------|------------|
| Element-wise arithmetic | 4-8x | Low |
| Reductions (sum, mean) | 2-4x | Medium |
| Dot product / matmul | 4-16x | High |
| Transcendentals (exp, log) | 2-4x | High |

## Design Principles (from NumPy)

| Aspect | NumPy Approach |
|--------|----------------|
| **Correctness** | Max 1-3 ULP accuracy loss |
| **Code Bloat** | Minimize source and binary size |
| **Maintainability** | Prefer universal intrinsics |
| **Performance** | Require benchmarks showing significant boost |

## References

- [NEP 38 Full Text](https://numpy.org/neps/nep-0038-SIMD-optimizations.html)
- [NumPy SIMD docs](https://numpy.org/doc/stable/reference/simd/index.html)
- [.NET Vector<T> docs](https://docs.microsoft.com/dotnet/api/system.numerics.vector-1)
- [.NET Intrinsics docs](https://docs.microsoft.com/dotnet/api/system.runtime.intrinsics)

## Related Issues

- #544 - SIMD optimization tracking
- #545 - SIMD implementation
