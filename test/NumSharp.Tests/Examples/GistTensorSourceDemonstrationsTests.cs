using System;
using System.Security.Cryptography;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Examples;

// Source-demonstration facts and explicit framework exclusions: examples/gist/coverage-tensor.json.
[TestClass]
public class GistTensorSourceDemonstrationsTests
{
    [TestMethod]
    public void StableWalk_Source200PointTimeline_OnFull64By64Latents()
    {
        using var scope = NDScope.Open();
        // The shape/timeline are the gist's. RandomState is a deterministic fixture, NOT Torch's RNG.
        var rng = np.random.RandomState(1337);
        var a = rng.randn(1, 4, 64, 64).astype(NPTypeCode.Single);
        var b = rng.randn(1, 4, 64, 64).astype(NPTypeCode.Single);
        var timeline = np.linspace(0.0, 1.0, 200);
        Assert.AreEqual(200L, timeline.size);
        Assert.AreEqual(0.0, timeline.item<double>(0));
        Assert.AreEqual(1.0, timeline.item<double>(199));
        for (int frame = 0; frame < 200; frame++)
        {
            using var iteration = NDScope.Open();
            var latent = StableDiffusionWalk.Slerp(timeline.item<double>(frame), a, b);
            CollectionAssert.AreEqual(new long[] { 1, 4, 64, 64 }, latent.shape);
            Assert.AreEqual(NPTypeCode.Single, latent.typecode);
            Assert.IsTrue(np.all(np.isfinite(latent)), $"frame {frame}");
            if (frame == 0) Assert.IsTrue(np.array_equal(a, latent));
            if (frame == 199) Assert.IsTrue(np.array_equal(b, latent));
            if (frame == 99)
            {
                // Captured NumPy 2.4.2 values; the managed norm may sum in another order.
                Assert.AreEqual(-.8654261827468872, latent.item<double>(0), 2e-6);
                Assert.AreEqual(-1.510780692100525, latent.item<double>(1), 2e-6);
            }
            Assert.IsTrue(np.linalg.norm(latent).item<double>() > 120);
            Assert.IsTrue(np.linalg.norm(latent).item<double>() < 135);
        }
    }

    [TestMethod]
    public void StableWalk_SourceGuidanceAnd512PixelConversion_CoverEveryElement()
    {
        using var scope = NDScope.Open();
        var rawNoise = new float[4 * 64 * 64];
        for (int i = 0; i < rawNoise.Length; i++) rawNoise[i] = (i % 257 - 128) / 32.0f;
        var unconditional = np.array(rawNoise).reshape(1, 4, 64, 64);
        var conditional = unconditional * -0.5f;
        var guided = StableDiffusionWalk.ClassifierFreeGuidance(unconditional, conditional, 7.5);
        var guidedValues = guided.ToArray<float>();
        for (int i = 0; i < guidedValues.Length; i++)
            Assert.AreEqual(rawNoise[i] + 7.5f * (-.5f * rawNoise[i] - rawNoise[i]), guidedValues[i]);

        var decodedValues = new float[3 * 512 * 512];
        for (int i = 0; i < decodedValues.Length; i++) decodedValues[i] = (i % 1025 - 512) / 256.0f;
        var decoded = np.array(decodedValues).reshape(1, 3, 512, 512);
        using var bytes = StableDiffusionWalk.DecodedImageToBytes(decoded);
        CollectionAssert.AreEqual(new long[] { 512, 512, 3 }, bytes.shape);
        Assert.AreEqual("52749DB215039D0EE4BAA6CB29E2D9912CD90F739D2BE85F478D993B0632FEEE",
            Convert.ToHexString(SHA256.HashData(bytes.ToArray<byte>())));
        Assert.AreEqual(100041344L, np.sum(bytes).item<long>());
    }

    [DataTestMethod]
    [DataRow(1)]
    [DataRow(2)]
    public void Timeseries_SourceThousandSampleDemonstrations_AssertAllWindowsAndSplits(int seriesCount)
    {
        using var scope = NDScope.Open();
        var t = np.arange(1000).astype(NPTypeCode.Int64);
        var original = seriesCount == 1 ? t : np.stack(new[] { t, -t }).T;
        using var normalized = TimeseriesCnn.AsTimeMajorMatrix(original);
        using var data = TimeseriesCnn.MakeInstances(normalized, 50);
        using var split = TimeseriesCnn.Split(data, (int)(.01 * normalized.shape[0]));
        CollectionAssert.AreEqual(new long[] { 950, 50, seriesCount }, data.X.shape);
        CollectionAssert.AreEqual(new long[] { 950, seriesCount }, data.Y.shape);
        CollectionAssert.AreEqual(new long[] { 1, 50, seriesCount }, data.Query.shape);
        CollectionAssert.AreEqual(new long[] { 940, 50, seriesCount }, split.XTrain.shape);
        CollectionAssert.AreEqual(new long[] { 10, 50, seriesCount }, split.XTest.shape);
        CollectionAssert.AreEqual(new long[] { 940, seriesCount }, split.YTrain.shape);
        CollectionAssert.AreEqual(new long[] { 10, seriesCount }, split.YTest.shape);
        long[] x = data.X.ToArray<long>(), y = data.Y.ToArray<long>(), q = data.Query.ToArray<long>();
        for (int sample = 0; sample < 950; sample++)
            for (int channel = 0; channel < seriesCount; channel++)
            {
                long sign = channel == 0 ? 1 : -1;
                Assert.AreEqual(sign * (sample + 50), y[sample * seriesCount + channel]);
                for (int offset = 0; offset < 50; offset++)
                    Assert.AreEqual(sign * (sample + offset), x[(sample * 50 + offset) * seriesCount + channel]);
            }
        for (int offset = 0; offset < 50; offset++)
            for (int channel = 0; channel < seriesCount; channel++)
                Assert.AreEqual((channel == 0 ? 1 : -1) * (950L + offset), q[offset * seriesCount + channel]);
        Assert.AreEqual(989L, split.YTrain.item<long>(939, 0));
        Assert.AreEqual(990L, split.YTest.item<long>(0, 0));
        Assert.AreEqual(999L, split.YTest.item<long>(9, 0));
    }

    [TestMethod]
    public void GoogLeNet_Source224RgbPreparation_AssertEntireTensor()
    {
        using var scope = NDScope.Open();
        var pixels = new byte[224 * 224 * 3];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = (byte)(i % 256);
        var rgb = np.array(pixels).reshape(224, 224, 3);
        using var input = GoogLeNet.PreprocessRgb(rgb);
        CollectionAssert.AreEqual(new long[] { 1, 3, 224, 224 }, input.shape);
        Assert.AreEqual(NPTypeCode.Single, input.typecode);
        var values = input.ToArray<float>();
        var rawBytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, rawBytes, 0, rawBytes.Length);
        Assert.AreEqual("53E786FEFD6197BA82154680EAF45AD176A071C81D3E91144F4A22D25D74A458",
            Convert.ToHexString(SHA256.HashData(rawBytes)));
        CollectionAssert.AreEqual(pixels, rgb.ToArray<byte>(), "preprocessing must not overwrite the original RGB input");
    }

    [DataTestMethod]
    [DataRow(64, 3.5928385257720947)]
    [DataRow(192, 2.78086256980896)]
    public void GoogLeNet_SourceLrnStageShapes_NormalizeAllChannels(int channels, double last)
    {
        using var scope = NDScope.Open();
        var values = new float[channels * 56 * 56];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 257 - 128) / 32.0f;
        var x = np.array(values).reshape(1, channels, 56, 56);
        using var normalized = GoogLeNet.LocalResponseNormalize(x);
        CollectionAssert.AreEqual(x.shape, normalized.shape);
        Assert.AreEqual(NPTypeCode.Single, normalized.typecode);
        Assert.IsTrue(np.all(np.isfinite(normalized)));
        Assert.IsTrue(np.all(np.abs(normalized) <= np.abs(x)));
        Assert.AreEqual(-3.998668670654297, normalized.item<double>(0), 5e-7);
        Assert.AreEqual(last, normalized.item<double>(normalized.size - 1), 5e-7);
    }

    [TestMethod]
    public void GoogLeNet_AllFourPoolCropsAndThirdClassifierSelection_UseSourceDimensions()
    {
        using var scope = NDScope.Open();
        // Shapes derive from the pinned convolution/padding/pooling parameters; no CNN is executed.
        foreach (var stage in new[] { (64, 114), (192, 58), (480, 30), (832, 16) })
        {
            using var iteration = NDScope.Open();
            var x = np.arange((long)stage.Item1 * stage.Item2 * stage.Item2).reshape(1, stage.Item1, stage.Item2, stage.Item2);
            var cropped = GoogLeNet.PoolHelper(x);
            CollectionAssert.AreEqual(new long[] { 1, stage.Item1, stage.Item2 - 1, stage.Item2 - 1 }, cropped.shape);
            Assert.AreEqual((long)stage.Item2 + 1, cropped.item<long>(0));
            Assert.AreEqual(x.item<long>(x.size - 1), cropped.item<long>(cropped.size - 1));
            x[0, 0, 1, 1] = -99L;
            Assert.AreEqual(-99L, cropped.item<long>(0), "PoolHelper must keep view semantics");
        }
        // The source chooses out[2], not an average across its three 1000-class heads.
        var first = np.zeros((1, 1000), dtype: np.float32);
        var second = np.zeros((1, 1000), dtype: np.float32);
        var final = np.zeros((1, 1000), dtype: np.float32);
        first[0, 3] = .9f; second[0, 6] = .9f; final[0, 281] = .9f;
        NDArray[] output = { first, second, final };
        Assert.AreEqual(281L, GoogLeNet.PredictedLabel(output[2]));
    }
}
