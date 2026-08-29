# Extending Libraries

NumSharp is designed to integrate with the broader .NET ecosystem. Extension packages bridge NumSharp arrays with platform-specific features and external libraries.

## Official Extensions

| Package | Purpose |
|---------|---------|
| [NumSharp.Bitmap](bitmap.md) | Image ↔ NDArray conversion via `System.Drawing` |
| [NumSharp.Interop.OpenBLAS](../interop/openblas.md) | OpenBLAS/LAPACK backend for the matrix products and `np.linalg` factorisations — byte-identical to NumPy 2.4.2 |
| [NumSharp.Interop.pythonnet](../interop/pythonnet-numpy.md) | Zero-copy NumSharp ↔ Python/NumPy exchange via Python.NET |
| [NumSharp.Build](../ndscoped.md) | Build-time `[NDScoped]` IL weaver — deterministic `NDArray` memory reclamation (a development dependency, never a runtime one; the companion Roslyn analyzer ships inside the `NumSharp` package itself) |

## Build Your Own

NumSharp exposes low-level memory access for integration with native libraries, GPU frameworks, or domain-specific formats:

```csharp
// Access raw memory for interop
byte* ptr = (byte*)ndarray.Unsafe.Address;

// Wrap external memory as NDArray
var nd = new NDArray(new ArraySlice<byte>(
    new UnmanagedMemoryBlock<byte>(ptr, length, onDispose)
));
```

Have an extension to share? [Open a PR](https://github.com/SciSharp/NumSharp) to add it to this list.
