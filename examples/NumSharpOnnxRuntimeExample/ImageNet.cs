using NumSharp;
using SkiaSharp;

namespace NumSharpOnnxRuntimeExample;

/// <summary>
///     The input every ImageNet classifier of the ONNX model zoo expects (ResNet, MobileNet, SqueezeNet, ...), made
///     with NumSharp array operations instead of the per-pixel loop of ONNX Runtime's C# tutorial.
/// </summary>
internal static class ImageNet
{
    /// <summary>The side of the square the models read.</summary>
    private const int Side = 224;

    /// <summary>
    ///     ImageNet's per-channel mean, shaped (3, 1, 1) so it broadcasts over a (3, 224, 224) image. Float32 on
    ///     purpose: a float64 constant would turn the image into float64, which a float32 model input refuses under
    ///     the bridge's default casting rule ("safe"), as Python's onnxruntime refuses it.
    /// </summary>
    private static readonly NDArray Mean = np.array(new[] { 0.485f, 0.456f, 0.406f }).reshape(3, 1, 1);

    /// <summary>ImageNet's per-channel standard deviation, shaped and typed like <see cref="Mean"/>.</summary>
    private static readonly NDArray Std = np.array(new[] { 0.229f, 0.224f, 0.225f }).reshape(3, 1, 1);

    /// <summary>
    ///     Decode a photo and turn it into a normalized (3, 224, 224) float32 image: SkiaSharp crops the largest centered
    ///     square and scales it to 224 x 224, then NumSharp drops the alpha channel, scales the bytes to [0, 1],
    ///     subtracts the mean, divides by the standard deviation and moves the channels in front of the rows.
    /// </summary>
    /// <remarks>
    ///     Every NumSharp step is a whole-array operation; the only per-pixel work is SkiaSharp's own resampling. The
    ///     intermediate arrays are disposed when the method returns (each <c>using</c>), so their pooled buffers go
    ///     back to NumSharp at once instead of waiting for the garbage collector.
    /// </remarks>
    /// <param name="path">A photo in a format SkiaSharp decodes (JPEG, PNG, WebP, BMP, GIF, ICO, ...).</param>
    /// <returns>A new (3, 224, 224) float32 array that the caller owns and disposes.</returns>
    /// <exception cref="InvalidDataException">The file does not exist or is not an image SkiaSharp can decode.</exception>
    public static NDArray Preprocess(string path)
    {
        byte[] rgba;
        using (SKBitmap photo = SKBitmap.Decode(path) ?? throw new InvalidDataException($"'{path}' is not an image SkiaSharp can decode."))
        {
            // The largest centered square, scaled to the model's side: what the tutorial's "resize, then crop the
            // center" produces, done in the other order.
            int square = Math.Min(photo.Width, photo.Height);
            using var crop = new SKBitmap();
            photo.ExtractSubset(crop, SKRectI.Create((photo.Width - square) / 2, (photo.Height - square) / 2, square, square));
            var target = new SKImageInfo(Side, Side, SKColorType.Rgba8888, SKAlphaType.Opaque);
            using SKBitmap resized = crop.Resize(target, new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear));
            rgba = resized.Bytes;                                  // row after row, four bytes (R, G, B, A) per pixel
        }

        using NDArray bytes = np.frombuffer(rgba, NPTypeCode.Byte);
        using NDArray hwc = bytes.reshape(Side, Side, 4);           // (height, width, channel), as the bitmap is laid out
        using NDArray rgb = hwc[":, :, :3"];                        // drop alpha: a view, nothing moves yet
        using NDArray chw = np.transpose(rgb, [2, 0, 1]);           // (channel, height, width): still a view
        using NDArray unit = chw.astype(NPTypeCode.Single);         // the one copy, laid out channel after channel
        using NDArray scaled = unit / 255f;                         // [0, 1]
        using NDArray centered = scaled - Mean;
        return centered / Std;
    }
}
