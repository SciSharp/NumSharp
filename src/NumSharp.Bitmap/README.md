# NumSharp.Bitmap

Windows-only extensions that bridge `System.Drawing.Bitmap` and NumSharp's `NDArray`, with or
without copying pixel data. Uses GDI+ via `System.Drawing.Common`.

This is an extension package for [NumSharp](https://www.nuget.org/packages/NumSharp) — it depends
on and builds on the core library.

## Install

```bash
dotnet add package NumSharp.Bitmap
```

> **Windows only.** `System.Drawing.Common` is supported on Windows only; on other platforms these
> APIs throw at runtime.

## Quick start

```csharp
using System.Drawing;
using NumSharp;
using NumSharp.Bitmap;

using var bmp = (Bitmap)Image.FromFile("input.png");

// Bitmap -> NDArray (shape [height, width, channels]); discardAlpha drops the alpha channel.
NDArray pixels = bmp.ToNDArray(copy: true, discardAlpha: true);

// ... manipulate pixels with np.* ...

// NDArray -> Bitmap
using Bitmap outBmp = pixels.ToBitmap();
```

Every conversion method takes a `discardAlpha` argument, and `ToNDArray` takes `flat` and `copy`
flags to control raveling and whether pixel data is shared or copied.

## Links

- Documentation: <https://scisharp.github.io/NumSharp/docs/extensions/bitmap.html>
- Source: <https://github.com/SciSharp/NumSharp>

Licensed under the Apache License 2.0.
