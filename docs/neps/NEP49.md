# NEP 49 - Data Allocation Strategies

**Status:** Final
**NumSharp Impact:** LOW - NumSharp uses unmanaged memory with custom patterns

## Summary

Provides mechanism to override memory management strategy for `ndarray.data` with user-provided alternatives.

## Custom Allocator API

### C Structure
```c
typedef struct {
    void *ctx;
    void* (*malloc) (void *ctx, size_t size);
    void* (*calloc) (void *ctx, size_t nelem, size_t elsize);
    void* (*realloc) (void *ctx, void *ptr, size_t new_size);
    void (*free) (void *ctx, void *ptr, size_t size);
} PyDataMemAllocator;
```

### Python API
```python
# Set/get allocation handler
np.core.multiarray.set_handler(handler)
np.core.multiarray.get_handler_name(arr)
```

## Use Cases

1. **Data Alignment:** 64-byte alignment for SIMD (40x performance gain)
2. **NUMA Pinning:** Pin to specific cores on multi-socket systems
3. **Memory Profiling:** Integration with tracing tools
4. **Specialized Hardware:** FPGA DMA operations
5. **Huge Pages:** Linux `madvise` for huge page allocation

## NumSharp Relevance

### Current Architecture

NumSharp already uses custom memory management:

```csharp
// UnmanagedMemoryBlock allocates via Marshal.AllocHGlobal
public unsafe class UnmanagedMemoryBlock<T> {
    private T* _address;

    public UnmanagedMemoryBlock(int length) {
        _address = (T*)Marshal.AllocHGlobal(length * sizeof(T));
    }
}
```

### Potential Enhancements

If NumSharp wanted similar flexibility:

```csharp
public interface IMemoryAllocator {
    IntPtr Allocate(int size);
    IntPtr Reallocate(IntPtr ptr, int newSize);
    void Free(IntPtr ptr);
}

public class AlignedAllocator : IMemoryAllocator {
    private readonly int _alignment;  // e.g., 64 bytes for AVX-512

    public IntPtr Allocate(int size) {
        // Use NativeMemory.AlignedAlloc in .NET 6+
        return (IntPtr)NativeMemory.AlignedAlloc((nuint)size, (nuint)_alignment);
    }
}
```

### Current Priority

**LOW** - NumSharp's current allocation works. Custom allocators would be useful for:
- SIMD-optimized operations (future)
- GPU memory (if CUDA support added)
- Memory-mapped files

## Key Design Decisions

- **Handler Lifetime:** Each array carries its allocator
- **Context Variables:** Thread-safe per-coroutine configuration
- **Backward Compatible:** Doesn't break existing code

## References

- [NEP 49 Full Text](https://numpy.org/neps/nep-0049-data-allocation-strategies.html)
- `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.cs`
