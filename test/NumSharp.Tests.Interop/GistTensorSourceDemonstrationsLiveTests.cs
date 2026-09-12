using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistTensorSourceDemonstrationsLiveTests : InteropTestBase
{
    [TestMethod]
    public void StableWalk_All200FullSizeLatentFrames_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(1337);
        var a = random.randn(1, 4, 64, 64).astype(NPTypeCode.Single);
        var b = random.randn(1, 4, 64, 64).astype(NPTypeCode.Single);
        ExportTo("a", a); ExportTo("b", b);
        PyExec("""
            # Fixture uses shared NumSharp buffers; it does not claim to reproduce torch.randn.
            def source_arc(t, a, b):
                c=np.sum(a*b/(np.linalg.norm(a)*np.linalg.norm(b)))
                if abs(c)>.9995: return (1-t)*a+t*b
                q=np.arccos(c)
                return np.sin(q-q*t)/np.sin(q)*a+np.sin(q*t)/np.sin(q)*b
            timeline=np.linspace(0,1,200)
            """);
        var timeline = np.linspace(0.0, 1.0, 200);
        Exact(timeline, "timeline", "source 200-frame timeline");
        for (int i = 0; i < 200; i++)
        {
            using var frame = StableDiffusionWalk.Slerp(timeline.item<double>(i), a, b);
            Exact(frame, $"source_arc(float(timeline[{i}]),a,b)", $"source latent frame {i}/199");
        }
    }

    [TestMethod]
    public void StableWalk_SourceGuidanceAnd512DecodedPixels_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var noise = new float[4 * 64 * 64];
        for (int i = 0; i < noise.Length; i++) noise[i] = (i % 257 - 128) / 32.0f;
        var a = np.array(noise).reshape(1, 4, 64, 64);
        var b = a * -0.5f;
        ExportTo("a", a); ExportTo("b", b);
        var guided = StableDiffusionWalk.ClassifierFreeGuidance(a, b, 7.5);
        Exact(guided, "a+7.5*(b-a)", "source guidance_scale=7.5, full latent shape");
        var values = new float[3 * 512 * 512];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 1025 - 512) / 256.0f;
        var decoded = np.array(values).reshape(1, 3, 512, 512);
        ExportTo("decoded", decoded);
        var bytes = StableDiffusionWalk.DecodedImageToBytes(decoded);
        Exact(bytes, "(np.clip(decoded/2+.5,0,1)[0].transpose(1,2,0)*255).astype(np.uint8)", "all 512x512 decoded RGB bytes");
    }

    [DataTestMethod]
    [DataRow(1)]
    [DataRow(2)]
    public void Timeseries_BothThousandSampleSourceDatasets_ExactLiveNumpy(int seriesCount)
    {
        using var scope = NDScope.Open();
        var t = np.arange(1000).astype(NPTypeCode.Int64);
        var source = seriesCount == 1 ? t : np.stack(new[] { t, -t }).T;
        ExportTo("source", source);
        using var matrix = TimeseriesCnn.AsTimeMajorMatrix(source);
        using var data = TimeseriesCnn.MakeInstances(matrix, 50);
        using var split = TimeseriesCnn.Split(data, (int)(.01 * matrix.shape[0]));
        PyExec("""
            series=np.atleast_2d(source)
            if series.shape[0]==1: series=series.T
            features=np.atleast_3d(np.array([series[i:i+50] for i in range(len(series)-50)]))
            targets=series[50:]
            query=np.atleast_3d([series[-50:]])
            ntest=int(.01*len(series))
            """);
        Exact(matrix, "series", "source time-major normalization");
        Exact(data.X, "features", "all 950 source lookback windows");
        Exact(data.Y, "targets", "all 950 next-step targets");
        Exact(data.Query, "query", "source final query window");
        Exact(split.XTrain, "features[:-ntest]", "source 940 training windows");
        Exact(split.XTest, "features[-ntest:]", "source 10 test windows");
        Exact(split.YTrain, "targets[:-ntest]", "source 940 training targets");
        Exact(split.YTest, "targets[-ntest:]", "source 10 test targets");
    }

    [TestMethod]
    public void GoogLeNet_Source224RgbImageAndThirdHeadArgmax_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var pixels = new byte[224 * 224 * 3];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = (byte)(i % 256);
        var rgb = np.array(pixels).reshape(224, 224, 3);
        ExportTo("rgb", rgb);
        PyExec("""
            image=rgb.astype(np.float32)
            for channel,mean in enumerate((123.68,116.779,103.939)): image[:,:,channel]-=mean
            image[:,:,[0,1,2]]=image[:,:,[2,1,0]]
            prepared=np.expand_dims(image.transpose(2,0,1),0)
            """);
        using var actual = GoogLeNet.PreprocessRgb(rgb);
        Exact(actual, "prepared", "all 224x224 source RGB preprocessing values");
        var heads = np.zeros((3, 1000), dtype: np.float32);
        heads[0, 3] = .9f; heads[1, 6] = .9f; heads[2, 281] = .9f;
        var final = heads["2:3,:"];
        ExportTo("heads", heads);
        Assert.AreEqual(PyLong("int(np.argmax(heads[2]))"), GoogLeNet.PredictedLabel(final));
    }

    [DataTestMethod]
    [DataRow(64)]
    [DataRow(192)]
    public void GoogLeNet_BothSourceLrnStageShapes_ExactLiveNumpy(int channels)
    {
        using var scope = NDScope.Open();
        var values = new float[channels * 56 * 56];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 257 - 128) / 32.0f;
        var input = np.array(values).reshape(1, channels, 56, 56);
        ExportTo("x", input);
        PyExec("""
            squared=np.pad(x*x,((0,0),(2,2),(0,0),(0,0)))
            scale=1
            for i in range(5): scale+=(.0001/5)*squared[:,i:i+x.shape[1]]
            expected=x/scale**.75
            """);
        using var result = GoogLeNet.LocalResponseNormalize(input);
        Exact(result, "expected", $"source LRN stage {channels}x56x56");
    }

    [TestMethod]
    public void GoogLeNet_AllFourSourcePoolHelperStages_ExactLiveNumpy()
    {
        foreach (var stage in new[] { (64, 114), (192, 58), (480, 30), (832, 16) })
        {
            using var scope = NDScope.Open();
            var x = np.arange((long)stage.Item1 * stage.Item2 * stage.Item2).reshape(1, stage.Item1, stage.Item2, stage.Item2);
            ExportTo("x", x);
            using var cropped = GoogLeNet.PoolHelper(x);
            Exact(cropped, "x[:,:,1:,1:]", $"source PoolHelper {stage.Item1}x{stage.Item2}x{stage.Item2}");
            PyExec("del x");
        }
    }

    private void Exact(NDArray actual, string expression, string label)
    {
        using (Gil())
        {
            using var expected = Scope.Eval(expression);
            GistParity.AssertExact(actual, expected, label);
        }
    }
}
