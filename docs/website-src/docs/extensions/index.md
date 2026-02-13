# Extending Libraries

NumSharp's core functionality can be extended with additional packages that integrate with platform-specific features or external libraries.

---

## Available Extensions

| Package | Description | Platform |
|---------|-------------|----------|
| [NumSharp.Bitmap](bitmap.md) | Convert between `System.Drawing.Bitmap` and `NDArray` | Windows |

---

## NumSharp.Bitmap

Seamless image-to-array conversion for image processing and ML workflows.

```csharp
using System.Drawing;
using NumSharp;

// Image → Array
var bitmap = new Bitmap("photo.jpg");
var pixels = bitmap.ToNDArray();  // (1, height, width, channels)

// Process pixels with NumSharp operations...

// Array → Image
var result = processedPixels.ToBitmap();
result.Save("output.jpg");
```

**Key Features:**
- Zero-copy mode for performance-critical code
- Supports 24bpp and 32bpp images
- Alpha channel handling
- Round-trip support (load → process → save)

[Full documentation →](bitmap.md)

---

## Installing Extensions

Extensions are separate NuGet packages:

```bash
# Core library (always required)
dotnet add package NumSharp

# Extensions (optional, add as needed)
dotnet add package NumSharp.Bitmap
```

---

## Creating Your Own Extensions

NumSharp is designed to be extensible. The key integration points:

### Working with NDArray Memory

```csharp
// Get raw pointer to contiguous data
unsafe
{
    byte* ptr = (byte*)ndarray.Unsafe.Address;
    // Use with P/Invoke, native libraries, etc.
}
```

### Creating NDArray from External Memory

```csharp
// Wrap existing unmanaged memory as NDArray
unsafe
{
    var slice = new ArraySlice<byte>(
        new UnmanagedMemoryBlock<byte>(ptr, length, disposeCallback)
    );
    var nd = new NDArray(slice);
}
```

### Extension Method Pattern

Follow the NumSharp.Bitmap pattern for consistent API:

```csharp
public static class YourExtensions
{
    public static NDArray ToNDArray(this YourType source, ...)
    {
        // Convert to NDArray
    }

    public static YourType ToYourType(this NDArray nd, ...)
    {
        // Convert from NDArray
    }
}
```
