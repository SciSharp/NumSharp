# NumSharp.Bitmap

The **NumSharp.Bitmap** package provides seamless conversion between `System.Drawing.Bitmap` and `NDArray`. If you're working with images in .NET—loading them, processing pixels, applying filters, or feeding them to ML models—this extension makes it easy to move data between the image world and the array world.

---

## Installation

NumSharp.Bitmap is a separate NuGet package:

```bash
dotnet add package NumSharp.Bitmap
```

> **Platform Note:** This extension uses `System.Drawing.Common`, which is only fully supported on Windows. On Linux/macOS, you'll need additional setup (libgdiplus) or consider alternatives like ImageSharp.

---

## Quick Start

```csharp
using System.Drawing;
using NumSharp;

// Load an image and convert to NDArray
var bitmap = new Bitmap("photo.jpg");
var pixels = bitmap.ToNDArray();
// Shape: (1, height, width, channels)
// e.g., (1, 480, 640, 3) for a 640x480 RGB image

// Manipulate the pixel data
var brightened = (pixels.astype(NPTypeCode.Int32) + 50).clip(0, 255).astype(NPTypeCode.Byte);

// Convert back to Bitmap
var result = brightened.ToBitmap();
result.Save("brightened.jpg");
```

---

## Converting Bitmaps to NDArrays

### `Bitmap.ToNDArray()`

The primary method for converting images to arrays.

```csharp
public static NDArray ToNDArray(
    this Bitmap image,
    bool flat = false,
    bool copy = true,
    bool discardAlpha = false
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flat` | `false` | If `true`, returns 1-D array of pixels: `R1G1B1R2G2B2...` |
| `copy` | `true` | If `true`, copies pixel data. If `false`, wraps bitmap memory directly. |
| `discardAlpha` | `false` | If `true`, strips the alpha channel (4th channel) from 32bpp images. |

**Return Shape:**

- `flat=false`: `(1, height, width, channels)` — 4-D tensor suitable for ML models
- `flat=true`: `(height * width * channels,)` — 1-D array of raw pixel bytes

### Examples

**Standard conversion (recommended for most uses):**

```csharp
var bitmap = new Bitmap("image.png");
var nd = bitmap.ToNDArray();

Console.WriteLine(nd.shape);  // e.g., (1, 480, 640, 4) for 32bpp ARGB
Console.WriteLine(nd.dtype);  // Byte
```

**Discard alpha channel:**

```csharp
// 32bpp ARGB → 3 channels (RGB only)
var rgb = bitmap.ToNDArray(discardAlpha: true);
Console.WriteLine(rgb.shape);  // (1, 480, 640, 3)
```

**Flat pixel array:**

```csharp
// For algorithms that expect 1-D input
var flat = bitmap.ToNDArray(flat: true);
Console.WriteLine(flat.ndim);  // 1
```

**Zero-copy mode (advanced):**

```csharp
// Wraps bitmap memory directly — faster but risky
var wrapped = bitmap.ToNDArray(copy: false);
// WARNING: The NDArray becomes invalid if the bitmap is disposed
// or modified. The bitmap remains locked until the NDArray is GC'd.
```

### Memory Layout

The pixel data is in **BGR/BGRA order** (Windows GDI convention), not RGB:

```csharp
var nd = bitmap.ToNDArray();
// nd[0, y, x, 0] = Blue
// nd[0, y, x, 1] = Green
// nd[0, y, x, 2] = Red
// nd[0, y, x, 3] = Alpha (if 32bpp)
```

If you need RGB order for ML models, swap the channels:

```csharp
// BGRA → RGBA
var rgba = nd[Slice.All, Slice.All, Slice.All, new int[] {2, 1, 0, 3}];
```

---

## Converting NDArrays to Bitmaps

### `NDArray.ToBitmap()`

Converts an NDArray back to a Bitmap.

```csharp
public static Bitmap ToBitmap(
    this NDArray nd,
    int width,
    int height,
    PixelFormat format = PixelFormat.DontCare
)

// Overload that infers dimensions from shape
public static Bitmap ToBitmap(
    this NDArray nd,
    PixelFormat format = PixelFormat.DontCare
)
```

**Requirements:**

- NDArray must be 4-D: `(1, height, width, channels)`
- First dimension must be 1 (single image)
- dtype should be `Byte`
- Channels must match the pixel format (3 for 24bpp, 4 for 32bpp)

### Examples

**Basic conversion:**

```csharp
var nd = np.zeros(1, 100, 200, 3).astype(NPTypeCode.Byte);
var bitmap = nd.ToBitmap();
// Infers: 200x100 image, 24bpp RGB
```

**Explicit format:**

```csharp
var nd = np.zeros(1, 100, 200, 4).astype(NPTypeCode.Byte);
var bitmap = nd.ToBitmap(200, 100, PixelFormat.Format32bppArgb);
```

**From flat array:**

```csharp
// If you have a 1-D array, provide dimensions and format
var flat = np.arange(0, 200 * 100 * 3).astype(NPTypeCode.Byte);
var bitmap = flat.ToBitmap(200, 100, PixelFormat.Format24bppRgb);
```

### Supported Pixel Formats

| Format | Channels | Bytes/Pixel |
|--------|----------|-------------|
| `Format24bppRgb` | 3 | 3 |
| `Format32bppArgb` | 4 | 4 |
| `Format32bppPArgb` | 4 | 4 |
| `Format32bppRgb` | 4 | 4 |
| `Format48bppRgb` | 3 | 6 |
| `Format64bppArgb` | 4 | 8 |
| `Format64bppPArgb` | 4 | 8 |

---

## Working with BitmapData Directly

For performance-critical code, you can work with `BitmapData` directly.

### `BitmapData.AsNDArray()`

Wraps locked bitmap data as an NDArray without copying.

```csharp
var bitmap = new Bitmap("image.png");
var bmpData = bitmap.LockBits(
    new Rectangle(0, 0, bitmap.Width, bitmap.Height),
    ImageLockMode.ReadOnly,
    bitmap.PixelFormat
);

try
{
    var nd = bmpData.AsNDArray(flat: false, discardAlpha: false);
    // Process pixels...
    // WARNING: nd is only valid while bits are locked!
}
finally
{
    bitmap.UnlockBits(bmpData);
}
```

> **Warning:** The NDArray points directly to bitmap memory. If you call `UnlockBits()`, the NDArray becomes invalid and accessing it causes undefined behavior.

---

## Common Patterns

### Image Preprocessing for ML

```csharp
// Load and normalize for neural network input
var bitmap = new Bitmap("input.jpg");
var nd = bitmap.ToNDArray(discardAlpha: true);  // (1, H, W, 3)

// Normalize to [0, 1] range
var normalized = nd.astype(NPTypeCode.Single) / 255.0f;

// Resize would require additional libraries (not built into NumSharp)
```

### Grayscale Conversion

```csharp
var bitmap = new Bitmap("color.jpg");
var rgb = bitmap.ToNDArray(discardAlpha: true);  // (1, H, W, 3)

// Luminance formula: 0.299*R + 0.587*G + 0.114*B
// Note: GDI uses BGR order, so channels are [B, G, R]
var b = rgb[Slice.All, Slice.All, Slice.All, 0].astype(NPTypeCode.Single);
var g = rgb[Slice.All, Slice.All, Slice.All, 1].astype(NPTypeCode.Single);
var r = rgb[Slice.All, Slice.All, Slice.All, 2].astype(NPTypeCode.Single);

var gray = (0.114f * b + 0.587f * g + 0.299f * r).astype(NPTypeCode.Byte);
// Shape: (1, H, W) - single channel
```

### Batch Processing

```csharp
// Process multiple images
var files = Directory.GetFiles("images/", "*.jpg");
var batch = new List<NDArray>();

foreach (var file in files)
{
    using var bitmap = new Bitmap(file);
    var nd = bitmap.ToNDArray(discardAlpha: true);
    batch.Add(nd);
}

// Stack into batch: (N, H, W, 3)
// Note: All images must have same dimensions
var batchArray = np.concatenate(batch.ToArray(), axis: 0);
```

### Round-Trip (Load, Process, Save)

```csharp
// Load
var original = new Bitmap("photo.jpg");
var nd = original.ToNDArray();

// Process: invert colors
var inverted = (255 - nd.astype(NPTypeCode.Int32)).clip(0, 255).astype(NPTypeCode.Byte);

// Save
var result = inverted.ToBitmap();
result.Save("inverted.jpg", ImageFormat.Jpeg);
```

---

## Known Limitations

### Platform Support

`System.Drawing.Common` is Windows-only in .NET 6+. On other platforms:

```csharp
// This throws PlatformNotSupportedException on Linux/macOS
var bitmap = new Bitmap("image.png");
```

**Workarounds:**
- Use `libgdiplus` on Linux (limited compatibility)
- Use ImageSharp or SkiaSharp (different API, not covered by this extension)

### Stride Padding

Bitmaps may have stride padding (row alignment to 4-byte boundaries). The extension handles this in most cases, but odd-width 24bpp images may have issues with `copy: true`. Use `copy: false` for odd-width images.

### Color Order

Windows bitmaps use BGR/BGRA byte order, not RGB. If your ML model expects RGB, you need to swap channels manually.

### No Resize

NumSharp doesn't include image resizing. You'll need to resize in `System.Drawing` before converting, or use a library like ImageSharp.

---

## API Reference

### Extension Methods

| Method | Description |
|--------|-------------|
| `Bitmap.ToNDArray(...)` | Convert Bitmap to NDArray |
| `Image.ToNDArray(...)` | Convert Image to NDArray (creates Bitmap internally) |
| `BitmapData.AsNDArray(...)` | Wrap locked BitmapData as NDArray (no copy) |
| `NDArray.ToBitmap(...)` | Convert NDArray to Bitmap |

### Helper Methods

| Method | Description |
|--------|-------------|
| `PixelFormat.ToBytesPerPixel()` | Get bytes per pixel for a format |
