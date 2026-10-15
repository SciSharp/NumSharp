// Gist: https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14
// Pinned source: https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14/6c17b8dd160d1212f11a7918d00c7c0c0c85e812
// Sources: googlenet.py, lrn.py, pool_helper.py; Joel Louis Marino.
// No license declaration was found in the pinned gist. Independently implemented mathematical
// routines. This does not load weights, resize/decode images or claim to run the GoogLeNet model.
using System;

namespace NumSharp.Examples.Gist;

public static class GoogLeNet
{
    /// <summary>
    /// Prepare an already-resized HWC RGB image: cast to float32, subtract RGB means,
    /// swap R/B, transpose to CHW and add a batch axis. The model requires 224x224; the
    /// numerical transform itself also accepts other spatial sizes for reusable preprocessing.
    /// The result owns its data and never changes the input.
    /// </summary>
    public static NDArray PreprocessRgb(NDArray rgb)
    {
        if (rgb is null) throw new ArgumentNullException(nameof(rgb));
        if (rgb.ndim != 3 || rgb.shape[2] != 3)
            throw new ArgumentException("Expected an HWC RGB image with three channels.", nameof(rgb));
        using var image = rgb.astype(NPTypeCode.Single, copy: true);
        float[] means = { 123.68f, 116.779f, 103.939f };
        for (int channel = 0; channel < 3; channel++)
        {
            using var view = image[$":,:,{channel}"];
            using var adjusted = view - means[channel];
            image[$":,:,{channel}"] = adjusted;
        }
        using var bgr = image[":,:,::-1"];
        using var chw = np.transpose(bgr, new[] { 2, 0, 1 });
        using var batch = np.expand_dims(chw, 0);
        return batch.copy();
    }

    /// <summary>Across-channel LRN from the companion: x/(k + alpha/n * window_sum(x²))^beta.</summary>
    public static NDArray LocalResponseNormalize(NDArray x, double alpha = 0.0001,
        double k = 1, double beta = 0.75, int n = 5)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.ndim != 4 || (x.typecode != NPTypeCode.Single && x.typecode != NPTypeCode.Double))
            throw new ArgumentException("Expected an NCHW float32 or float64 tensor.", nameof(x));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        int half = n / 2;
        using var square = np.square(x);
        using var padded = np.pad(square, new[,] { { 0, 0 }, { half, half }, { 0, 0 }, { 0, 0 } });
        using var scale = np.full(x.shape, k, dtype: x.dtype);
        for (int i = 0; i < n; i++)
        {
            using var region = padded[$":,{i}:{i + x.shape[1]},:,:"];
            using var contribution = (alpha / n) * region;
            np.add(scale, contribution, @out: scale);
        }
        using var divisor = np.power(scale, beta);
        return x / divisor;
    }

    /// <summary>Remove the leading padded row/column before Caffe-compatible pooling; a shared view.</summary>
    public static NDArray PoolHelper(NDArray x)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (x.ndim != 4) throw new ArgumentException("Expected an NCHW tensor.", nameof(x));
        return x[":,:,1:,1:"];
    }

    /// <summary>The flattened first-maximum class selection in the gist's inference demonstration.</summary>
    public static long PredictedLabel(NDArray probabilities)
    {
        if (probabilities is null) throw new ArgumentNullException(nameof(probabilities));
        return np.argmax(probabilities); // this overload already returns a C# scalar
    }

    public static void Demo()
    {
        using var rgb = np.array(new byte[,,] { { { 255, 128, 0 }, { 0, 64, 255 } } });
        using var input = PreprocessRgb(rgb);
        using var normalized = LocalResponseNormalize(input);
        using var probabilities = np.array(new[] { 0.1f, 0.7f, 0.2f });
        Console.WriteLine($"GoogLeNet input shape: ({string.Join(",", input.shape)}); BGR[0]={input.item<float>(0):F3}; normalized={normalized.item<float>(0):F3}; class={PredictedLabel(probabilities)}");
    }
}
