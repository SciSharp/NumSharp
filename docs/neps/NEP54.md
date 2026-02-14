# NEP 54 - SIMD Infrastructure Evolution: Google Highway

**Status:** Accepted
**NumSharp Impact:** INFORMATIONAL - Shows modern SIMD library design patterns

## Summary

NumPy is adopting Google Highway as its SIMD framework, replacing the C-based Universal Intrinsics with a C++ solution supporting sizeless SIMD (ARM SVE, RISC-V RVV).

## What is Google Highway?

Google Highway is a modern C++ SIMD library providing:
- Portable intrinsics across CPU architectures
- Sizeless SIMD support (SVE, RVV)
- Clean, readable API
- Apache 2.0 / BSD-3 dual license
- Used by Chromium, JPEG XL

## Why Move from Universal Intrinsics to Highway?

### Code Readability Improvement

**Old C code (Universal Intrinsics):**
```c
npyv_@sfx@ a5 = npyv_load_@sfx@(src1 + npyv_nlanes_@sfx@ * 4);
```

**New C++ (Highway):**
```cpp
auto a5 = Load(src1 + nlanes * 4);
```

### Key Motivations

1. **Sizeless SIMD:** ARM SVE and RISC-V RVV not possible with C intrinsics
2. **Better Documentation:** Highway has extensive docs
3. **Wider Testing:** Used by other projects, more bug detection
4. **Maintainability:** Clearer C++ code reduces regressions

## Comparison: Highway vs Universal Intrinsics

| Aspect | Highway | Universal Intrinsics |
|--------|---------|----------------------|
| **Language** | C++ | C |
| **Compilation** | Single unit, preprocessing | Multiple units per feature |
| **Runtime Dispatch** | One-time dynamic | Multiple linked functions |
| **SVE/RVV Support** | Yes | No |
| **Z-system Support** | Limited | Full (VX/VXE/VXE2) |

## Supported Platforms

### Highway Supports:
- x86/x86-64: SSE4.2, AVX, AVX2, AVX-512
- ARM: NEON, SVE, SVE2
- PowerPC
- RISC-V RVV
- s390x (limited)

### NumPy Already Implemented:
Sorting functions (sort, argsort, partition, argpartition) use:
- Intel x86-simd-sort
- Google Highway
- Large speedups reported

## NumSharp Relevance

### Learning from Highway's Design

Highway's API patterns are applicable to .NET SIMD:

**Highway (C++):**
```cpp
HWY_API Vec128<float> Add(Vec128<float> a, Vec128<float> b) {
    return Vec128<float>{_mm_add_ps(a.raw, b.raw)};
}
```

**Equivalent .NET pattern:**
```csharp
public static Vector128<float> Add(Vector128<float> a, Vector128<float> b) {
    if (Sse.IsSupported)
        return Sse.Add(a, b);
    // Fallback
    return SoftwareFallback(a, b);
}
```

### .NET SIMD Options

| Option | Pros | Cons |
|--------|------|------|
| `Vector<T>` | Portable, simple | Limited operations |
| `Vector128/256<T>` | Full control, all intrinsics | More code, manual dispatch |
| `Span<T>` + vectorization | Compiler may vectorize | Unpredictable |

### Potential NumSharp Architecture

```csharp
public interface ISimdBackend {
    void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result);
    void Multiply(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result);
    float Sum(ReadOnlySpan<float> a);
}

public class AvxBackend : ISimdBackend {
    public void Add(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result) {
        // AVX implementation
    }
}

public class ScalarBackend : ISimdBackend {
    // Fallback implementation
}

// Runtime selection
ISimdBackend backend = Avx2.IsSupported ? new Avx2Backend()
                     : Avx.IsSupported ? new AvxBackend()
                     : new ScalarBackend();
```

## Performance in NumPy 2.0

### Sorting Speedups (VQSort via Highway)
- `np.sort`, `np.argsort`
- `np.partition`, `np.argpartition`
- Hardware-specific large speedups

### Math Routines
- Highway has limited, low-precision math
- NumPy keeps existing routines, may use Highway primitives internally

## References

- [NEP 54 Full Text](https://numpy.org/neps/nep-0054-simd-cpp-highway.html)
- [Google Highway GitHub](https://github.com/google/highway)
- [NumPy Roadmap](https://numpy.org/neps/roadmap.html)
- [.NET Hardware Intrinsics](https://docs.microsoft.com/dotnet/standard/simd)

## Related Issues

- #544 - SIMD optimization tracking
- #545 - SIMD implementation
