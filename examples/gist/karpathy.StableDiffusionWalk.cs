// Gist: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
// Pinned source: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355/cfec3bb84900cc3abfe5539d6d69e3fb1f2921b8#file-stablediffusionwalk-py
// Source: stablediffusionwalk.py, Andrej Karpathy; its slerp credits @xsteenbrugge.
// No license declaration was found in the pinned gist. Original numerical implementation below;
// no source comments or docstrings are reproduced. This is not a Stable Diffusion model runner.
using System;
using System.Linq;

namespace NumSharp.Examples.Gist;

/// <summary>Array operations from the latent-space walk, without PyTorch, checkpoints or image IO.</summary>
public static class StableDiffusionWalk
{
    /// <summary>
    /// Interpolate equally shaped float32/float64 tensors. The nearly parallel/antiparallel branch
    /// deliberately uses linear interpolation, as the gist does. Zero norms produce NaNs.
    /// The gist's uninitialized inputs_are_torch variable for ndarray input has no C# equivalent:
    /// this entry point accepts NDArray directly and never moves an array between devices.
    /// </summary>
    public static NDArray Slerp(double t, NDArray v0, NDArray v1, double dotThreshold = 0.9995)
    {
        RequireMatching(v0, v1);
        using var n0 = np.linalg.norm(v0);
        using var n1 = np.linalg.norm(v1);
        using var denominator = n0 * n1;
        using var products = v0 * v1;
        using var normalized = products / denominator;
        using var dot = np.sum(normalized);
        if (Math.Abs(dot.item<double>()) > dotThreshold)
        {
            using var left = (1 - t) * v0;
            using var right = t * v1;
            return left + right;
        }

        using var theta = np.arccos(dot);
        using var sinTheta = np.sin(theta);
        // NumPy's float32 scalar * Python float stays float32 (NEP50). A C# double combined
        // with a zero-dimensional NDArray would instead widen this scalar intermediate.
        using var fraction = v0.typecode == NPTypeCode.Single ? np.asarray((float)t) : np.asarray(t);
        using var thetaT = theta * fraction;
        using var sinThetaT = np.sin(thetaT);
        using var remainder = theta - thetaT;
        using var sinRemainder = np.sin(remainder);
        using var s0 = sinRemainder / sinTheta;
        using var s1 = sinThetaT / sinTheta;
        using var weighted0 = s0 * v0;
        using var weighted1 = s1 * v1;
        return weighted0 + weighted1;
    }

    /// <summary>Combine unconditional and text-conditioned noise predictions before a scheduler step.</summary>
    public static NDArray ClassifierFreeGuidance(NDArray unconditional, NDArray conditional, double scale = 7.5)
    {
        RequireMatching(unconditional, conditional);
        using var difference = conditional - unconditional;
        using var correction = scale * difference;
        return unconditional + correction;
    }

    /// <summary>
    /// Convert the first decoded NCHW image to HWC uint8. Input is the VAE output, not a latent:
    /// divide by two, add one half, clamp, multiply by 255 and truncate exactly as the gist does.
    /// The returned image is owned. Decoding a latent still requires an actual VAE elsewhere.
    /// </summary>
    public static NDArray DecodedImageToBytes(NDArray decoded)
    {
        RequireFloat(decoded, nameof(decoded));
        if (decoded.ndim != 4 || decoded.shape[0] == 0)
            throw new ArgumentException("Expected a nonempty NCHW batch.", nameof(decoded));
        using var half = decoded / 2.0;
        using var shifted = half + 0.5;
        using var clipped = np.clip(shifted, 0.0, 1.0);
        using var first = clipped["0,:,:,:"];
        using var hwc = np.transpose(first, new[] { 1, 2, 0 });
        using var scaled = hwc * 255;
        return scaled.astype(NPTypeCode.Byte);
    }

    public static void Demo()
    {
        using var a = np.array(new[] { 1.0, 0.0, 0.0 });
        using var b = np.array(new[] { 0.0, 1.0, 0.0 });
        using var midway = Slerp(0.5, a, b);
        using var guided = ClassifierFreeGuidance(a, b);
        Console.WriteLine($"Latent arc midpoint: [{midway.item<double>(0):F6}, {midway.item<double>(1):F6}]; guided noise[0]={guided.item<double>(0):F2}");
    }

    private static void RequireMatching(NDArray a, NDArray b)
    {
        RequireFloat(a, nameof(a));
        RequireFloat(b, nameof(b));
        if (a.typecode != b.typecode || !a.shape.SequenceEqual(b.shape))
            throw new ArgumentException("The two tensors must have the same shape and dtype.");
    }

    private static void RequireFloat(NDArray value, string name)
    {
        if (value is null) throw new ArgumentNullException(name);
        if (value.typecode != NPTypeCode.Single && value.typecode != NPTypeCode.Double)
            throw new ArgumentException("Expected float32 or float64.", name);
    }
}
