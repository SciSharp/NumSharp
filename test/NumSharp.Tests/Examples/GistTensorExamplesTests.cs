using System;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Examples;

[TestClass]
public class GistTensorExamplesTests
{
    [TestMethod]
    public void Slerp_OrthogonalAndLinearBranches_PreserveDtypeAndInputs()
    {
        foreach (var dtype in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            using var raw0 = np.array(new[] { 1.0, 0.0, 0.0 });
            using var raw1 = np.array(new[] { 0.0, 1.0, 0.0 });
            using var a = raw0.astype(dtype);
            using var b = raw1.astype(dtype);
            using var arc = StableDiffusionWalk.Slerp(0.5, a, b);
            Assert.AreEqual(dtype, arc.typecode);
            Assert.AreEqual(System.Math.Sqrt(0.5), arc.item<double>(0), 1e-7);
            Assert.AreEqual(System.Math.Sqrt(0.5), arc.item<double>(1), 1e-7);
            Assert.AreEqual(0.0, arc.item<double>(2));
            using var parallel = StableDiffusionWalk.Slerp(0.25, a, a);
            Assert.IsTrue(np.array_equal(a, parallel));
            using var negative = -a;
            using var antiparallel = StableDiffusionWalk.Slerp(0.5, a, negative);
            Assert.AreEqual(0.0, antiparallel.item<double>(0));
            Assert.AreEqual(1.0, a.item<double>(0));
            Assert.AreEqual(1.0, b.item<double>(1));
        }
    }

    [TestMethod]
    public void Slerp_ZeroNorm_PropagatesNaNLikeTheFormula()
    {
        using var a = np.zeros(new Shape(3));
        using var b = np.ones(new Shape(3));
        using var result = StableDiffusionWalk.Slerp(0.5, a, b);
        for (int i = 0; i < 3; i++) Assert.IsTrue(double.IsNaN(result.item<double>(i)));
    }

    [TestMethod]
    public void GuidanceAndImagePostprocessing_AreRunnableWithoutAModel()
    {
        using var uncond = np.array(new[] { 1.0f, 2.0f });
        using var cond = np.array(new[] { 2.0f, -1.0f });
        using var guided = StableDiffusionWalk.ClassifierFreeGuidance(uncond, cond, 2);
        Assert.AreEqual(3.0f, guided.item<float>(0));
        Assert.AreEqual(-4.0f, guided.item<float>(1));
        using var values = np.array(new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 0.5f });
        using var decoded = values.reshape(1, 3, 1, 2);
        using var image = StableDiffusionWalk.DecodedImageToBytes(decoded);
        CollectionAssert.AreEqual(new long[] { 1, 2, 3 }, image.shape);
        Assert.AreEqual(NPTypeCode.Byte, image.typecode);
        CollectionAssert.AreEqual(new byte[] { 0, 127, 255, 0, 255, 191 }, image.ToArray<byte>());
    }

    [TestMethod]
    public void Timeseries_VectorAndMatrix_KeepOriginalShapesAndOwnership()
    {
        using var vector = np.arange(6).astype(NPTypeCode.Int64);
        using var data = TimeseriesCnn.MakeInstances(vector, 3);
        CollectionAssert.AreEqual(new long[] { 3, 3, 1 }, data.X.shape);
        CollectionAssert.AreEqual(new long[] { 3 }, data.Y.shape);
        CollectionAssert.AreEqual(new long[] { 1, 3, 1 }, data.Query.shape);
        Assert.AreEqual(2L, data.X.item<long>(2, 0, 0));
        Assert.AreEqual(5L, data.Query.item<long>(0, 2, 0));
        vector[3] = 90L;
        Assert.AreEqual(90L, data.Y.item<long>(0)); // the original returns a target view
        Assert.AreEqual(3L, data.X.item<long>(1, 2, 0));
        Assert.AreEqual(3L, data.Query.item<long>(0, 0, 0));

        using var all = np.arange(12).astype(NPTypeCode.Int64);
        using var matrix = all.reshape(6, 2);
        using var multi = TimeseriesCnn.MakeInstances(matrix, 3);
        CollectionAssert.AreEqual(new long[] { 3, 3, 2 }, multi.X.shape);
        CollectionAssert.AreEqual(new long[] { 3, 2 }, multi.Y.shape);
        Assert.AreEqual(9L, multi.X.item<long>(2, 2, 1));
    }

    [TestMethod]
    public void Timeseries_ReversedInputAndChronologicalSplit_RespectPythonSlices()
    {
        using var raw = np.arange(10).astype(NPTypeCode.Int64);
        using var reversed = raw["::-1"];
        using var instances = TimeseriesCnn.MakeInstances(reversed, 2);
        using var split = TimeseriesCnn.Split(instances, 2);
        Assert.AreEqual(6L, split.XTrain.shape[0]);
        Assert.AreEqual(2L, split.XTest.shape[0]);
        Assert.AreEqual(9L, split.XTrain.item<long>(0));
        Assert.AreEqual(3L, split.XTest.item<long>(0));
        using var zero = TimeseriesCnn.Split(instances, 0);
        Assert.AreEqual(0L, zero.XTrain.shape[0]);
        Assert.AreEqual(8L, zero.XTest.shape[0]);
        using var oversized = TimeseriesCnn.Split(instances, 100);
        Assert.AreEqual(0L, oversized.XTrain.shape[0]);
        Assert.AreEqual(8L, oversized.XTest.shape[0]);
        using var column = TimeseriesCnn.AsTimeMajorMatrix(raw);
        CollectionAssert.AreEqual(new long[] { 10, 1 }, column.shape);
    }

    [TestMethod]
    public void GoogLeNet_PreprocessIsFloat32BgrAndDoesNotMutateRgb()
    {
        using var rgb = np.array(new byte[,,] { { { 255, 128, 0 }, { 0, 64, 255 } } });
        using var result = GoogLeNet.PreprocessRgb(rgb);
        CollectionAssert.AreEqual(new long[] { 1, 3, 1, 2 }, result.shape);
        Assert.AreEqual(NPTypeCode.Single, result.typecode);
        Assert.AreEqual(-103.939f, result.item<float>(0, 0, 0, 0));
        Assert.AreEqual(151.061f, result.item<float>(0, 0, 0, 1));
        Assert.AreEqual(131.32f, result.item<float>(0, 2, 0, 0));
        Assert.AreEqual((byte)255, rgb.item<byte>(0, 0, 0));
    }

    [TestMethod]
    public void GoogLeNet_LrnAndPoolHelper_CoverChannelEdgesAndViews()
    {
        using var raw = np.arange(1, 25).astype(NPTypeCode.Single);
        using var input = raw.reshape(1, 6, 2, 2);
        using var result = GoogLeNet.LocalResponseNormalize(input);
        Assert.AreEqual(0.9983980059623718, result.item<double>(0), 1e-7);
        Assert.AreEqual(23.56583023071289, result.item<double>(23), 2e-6);
        using var identity = GoogLeNet.LocalResponseNormalize(input, alpha: 0);
        Assert.IsTrue(np.array_equal(input, identity));
        using var cropped = GoogLeNet.PoolHelper(input);
        CollectionAssert.AreEqual(new long[] { 1, 6, 1, 1 }, cropped.shape);
        Assert.AreEqual(4.0f, cropped.item<float>(0));
        input[0, 0, 1, 1] = 99.0f;
        Assert.AreEqual(99.0f, cropped.item<float>(0));
        using var tied = np.array(new[] { 0.2f, 0.8f, 0.8f });
        Assert.AreEqual(1L, GoogLeNet.PredictedLabel(tied));
    }

    [TestMethod]
    public void InvalidTensorShapesAndWindows_AreExplicitErrors()
    {
        using var a = np.arange(6);
        using var f = a.astype(NPTypeCode.Double);
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => TimeseriesCnn.MakeInstances(a, 0));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => TimeseriesCnn.MakeInstances(a, 6));
        Assert.ThrowsException<ArgumentException>(() => GoogLeNet.PreprocessRgb(a));
        Assert.ThrowsException<ArgumentException>(() => GoogLeNet.PoolHelper(a));
        Assert.ThrowsException<ArgumentException>(() => GoogLeNet.LocalResponseNormalize(f));
        Assert.ThrowsException<ArgumentException>(() => StableDiffusionWalk.DecodedImageToBytes(f));
        Assert.ThrowsException<ArgumentException>(() => StableDiffusionWalk.Slerp(0.5, a, a));
    }
}
