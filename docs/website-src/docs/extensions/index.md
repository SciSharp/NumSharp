# Extending Libraries

NumSharp is designed to integrate with the broader .NET ecosystem. Extension packages bridge NumSharp arrays with platform-specific features and external libraries.

## Official Extensions

| Package | Purpose |
|---------|---------|
| [NumSharp.Bitmap](bitmap.md) | Image ↔ NDArray conversion via `System.Drawing` |

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
