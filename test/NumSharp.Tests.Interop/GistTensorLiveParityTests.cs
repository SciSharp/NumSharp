using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

/// <summary>Live independent NumPy formulae over zero-copy exported inputs; no external model downloads.</summary>
[TestClass]
public class GistTensorLiveParityTests : InteropTestBase
{
    [TestMethod]
    public void Slerp_OrthogonalAndNearParallel_ByteExact()
    {
        PyExec("""
            def arc(t, a, b):
                c = np.sum(a*b / (np.linalg.norm(a)*np.linalg.norm(b)))
                if abs(c) > .9995:
                    return (1-t)*a + t*b
                angle = np.arccos(c)
                return np.sin(angle-angle*t)/np.sin(angle)*a + np.sin(angle*t)/np.sin(angle)*b
            """);
        foreach (var dtype in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            using var data0 = np.array(new[] { 1.0, 0.0, 0.0 });
            using var data1 = np.array(new[] { 0.0, 1.0, 0.0 });
            using var a = data0.astype(dtype);
            using var b = data1.astype(dtype);
            ExportTo("a", a);
            ExportTo("b", b);
            foreach (double t in new[] { 0.0, 0.25, 0.5, 1.0 })
            {
                using var result = StableDiffusionWalk.Slerp(t, a, b);
                AssertExact(result, $"arc({t.ToString(System.Globalization.CultureInfo.InvariantCulture)}, a, b)", $"spherical interpolation {dtype}, t={t}");
            }
            using var nearbyRaw = np.array(new[] { 1.0, 0.0001, 0.0 });
            using var nearby = nearbyRaw.astype(dtype);
            ExportTo("nearby", nearby);
            using var nearParallel = StableDiffusionWalk.Slerp(0.37, a, nearby);
            AssertExact(nearParallel, "arc(.37, a, nearby)", "near-parallel interpolation");
            using var parallel = StableDiffusionWalk.Slerp(0.25, a, a);
            AssertExact(parallel, "arc(.25, a, a)", "parallel interpolation");
            using var opposite = -a;
            using var antiparallel = StableDiffusionWalk.Slerp(0.5, a, opposite);
            AssertExact(antiparallel, "arc(.5, a, -a)", "antiparallel interpolation");
        }
    }

    [TestMethod]
    public void Slerp_StridedTensorAndZeroNorm_VsLiveNumpy()
    {
        PyExec("""
            def arc(t, a, b):
                c = np.sum(a*b / (np.linalg.norm(a)*np.linalg.norm(b)))
                if abs(c) > .9995: return (1-t)*a+t*b
                q = np.arccos(c)
                return np.sin(q-q*t)/np.sin(q)*a+np.sin(q*t)/np.sin(q)*b
            """);
        using var values = np.array(new double[,] { { 1, 8, 3, 9 }, { -2, 7, 4, 6 } });
        using var a = values[":,::2"];
        using var b = a["::-1,::-1"];
        ExportTo("a", a);
        ExportTo("b", b);
        using var result = StableDiffusionWalk.Slerp(0.3, a, b);
        AssertExact(result, "arc(.3, a, b)", "strided latent arc");

        using var zero = np.zeros(new Shape(2, 2));
        using var nan = StableDiffusionWalk.Slerp(0.5, zero, a);
        ExportTo("z", zero);
        ExportTo("actual", nan);
        PyExec("with np.errstate(all='ignore'): expected = arc(.5, z, a)");
        Assert.IsTrue(PyBool("actual.dtype == expected.dtype and actual.shape == expected.shape and np.isnan(actual).all() and np.isnan(expected).all()"));
        // NaN payload/sign is not a portable arithmetic contract for a zero-norm interpolation.
    }

    [TestMethod]
    public void GuidanceAndDecodedPixels_ByteExact()
    {
        foreach (var dtype in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            using var raw = np.array(new[] { -2.0, -1.0, -0.001, 0.0, 0.1, 0.5, 0.9, 1.0, 2.0, 0.25, -0.5, 0.75 });
            using var typed = raw.astype(dtype);
            using var decoded = typed.reshape(1, 3, 2, 2);
            using var reverse = decoded[":,:,::-1,::-1"];
            ExportTo("x", reverse);
            using var image = StableDiffusionWalk.DecodedImageToBytes(reverse);
            AssertExact(image, "(np.clip(x/2+.5,0,1)[0].transpose(1,2,0)*255).astype(np.uint8)", "decoded image bytes");
            using var prediction = reverse * 0.5;
            ExportTo("conditioned", prediction);
            using var guidance = StableDiffusionWalk.ClassifierFreeGuidance(reverse, prediction, 7.5);
            AssertExact(guidance, "x+7.5*(conditioned-x)", "classifier-free guidance");
        }
    }

    [TestMethod]
    public void Timeseries_WindowsTargetsQueryAndSplit_ByteExact()
    {
        using var raw = np.arange(24).astype(NPTypeCode.Int64);
        using var matrix = raw.reshape(12, 2);
        using var reversed = matrix["::-1,::-1"];
        foreach (var input in new[] { raw, reversed })
        {
            ExportTo("s", input);
            using var data = TimeseriesCnn.MakeInstances(input, 3);
            PyExec("features=np.atleast_3d(np.array([s[i:i+3] for i in range(len(s)-3)]))\ntargets=s[3:]\nquery=np.atleast_3d([s[-3:]])");
            AssertExact(data.X, "features", "lookback windows");
            AssertExact(data.Y, "targets", "next-step targets");
            AssertExact(data.Query, "query", "query");
            foreach (int count in new[] { 0, 2, 100 })
            {
                using var split = TimeseriesCnn.Split(data, count);
                AssertExact(split.XTrain, $"features[:-{count}]", "train windows");
                AssertExact(split.XTest, $"features[-{count}:]", "test windows");
                AssertExact(split.YTrain, $"targets[:-{count}]", "train targets");
                AssertExact(split.YTest, $"targets[-{count}:]", "test targets");
            }
            using var normalized = TimeseriesCnn.AsTimeMajorMatrix(input);
            AssertExact(normalized, "np.atleast_2d(s).T if np.atleast_2d(s).shape[0]==1 else np.atleast_2d(s)", "time-major normalization");
        }
    }

    [TestMethod]
    public void GoogLeNet_PreprocessingPoolAndArgmax_ByteExact()
    {
        using var values = np.arange(36).astype(NPTypeCode.Byte);
        using var source = values.reshape(3, 4, 3);
        using var rgb = source["::-1,::2,:"];
        ExportTo("rgb", rgb);
        PyExec("""
            image = rgb.astype(np.float32)
            image[:,:,0] -= 123.68
            image[:,:,1] -= 116.779
            image[:,:,2] -= 103.939
            image[:,:,[0,1,2]] = image[:,:,[2,1,0]]
            expected = image.transpose(2,0,1)[None]
            """);
        using var prepared = GoogLeNet.PreprocessRgb(rgb);
        AssertExact(prepared, "expected", "RGB to centered BGR/NCHW");
        ExportTo("prepared", prepared);
        using var cropped = GoogLeNet.PoolHelper(prepared);
        AssertExact(cropped, "prepared[:,:,1:,1:]", "pool helper crop");
        using var probabilities = np.array(new[] { .1f, .6f, .6f, .3f });
        ExportTo("probabilities", probabilities);
        Assert.AreEqual(PyLong("int(np.argmax(probabilities))"), GoogLeNet.PredictedLabel(probabilities));
    }

    [TestMethod]
    public void GoogLeNet_LocalResponseNormalization_ByteExact()
    {
        PyExec("""
            def lrn(x, n):
                padded=np.pad(x*x,((0,0),(n//2,n//2),(0,0),(0,0)))
                scale=1
                for offset in range(n):
                    scale += (.0001/n)*padded[:,offset:offset+x.shape[1],:,:]
                return x/scale**.75
            """);
        foreach (var dtype in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            using var raw = np.arange(1, 25).astype(dtype);
            using var tensor = raw.reshape(1, 6, 2, 2);
            using var reversed = tensor[":,::-1,::-1,:"];
            ExportTo("x", reversed);
            foreach (int n in new[] { 1, 4, 5, 9 })
            {
                using var result = GoogLeNet.LocalResponseNormalize(reversed, n: n);
                AssertExact(result, $"lrn(x,{n})", "channel normalization");
            }
        }
    }

    private void AssertExact(NDArray actual, string expression, string context)
    {
        using (Gil())
        {
            using var expected = Scope.Eval(expression);
            GistParity.AssertExact(actual, expected, context);
        }
    }
}
