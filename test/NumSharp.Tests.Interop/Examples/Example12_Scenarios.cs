using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/12-scenarios.cs</c>, scenario by scenario.</summary>
    [TestClass]
    public class Example12_Scenarios : ExampleTestBase
    {
        [TestMethod]
        public void Scenario1_MlInference_FeaturesCrossZeroCopy_PredictionsComeBack()
        {
            using var scope = NDScope.Open();
            NDArray weights = np.array(new double[] { 0.5, -1.0, 2.0, 0.25 });
            using (Gil())
            {
                py.w = weights.ToNumpyCopy();
                for (int b = 0; b < 10; b++)
                using (NDScope.Open())
                {
                    NDArray feats = np.arange(b, b + 12).astype(np.float64).reshape(3, 4) / 4.0;
                    py.batch = feats;
                    NDArray pred = py.Eval("batch.dot(w) - batch.mean()");
                    NDArray expected = np.dot(feats, weights) - np.mean(feats);
                    for (int i = 0; i < 3; i++)
                        pred.GetDouble(i).Should().BeApproximately(expected.GetDouble(i), 1e-9, $"batch {b}, element {i}");
                }

                py.Exec("del w, batch");
            }
        }

        [TestMethod]
        public void Scenario2_ImagePipeline_PythonEditsACSharpOwnedImageInPlace()
        {
            using var scope = NDScope.Open();
            NDArray img = np.arange(32 * 32 * 3).astype(np.uint8).reshape(32, 32, 3);
            NDArray crop = img["8:24, 8:24, :"];
            byte original = (byte)(((5 * 32 + 7) * 3 + 1) % 256);
            using (Gil())
            {
                py.img = img;
                py.Exec("img[:] = np.minimum(img.astype('i2') + 7, 255).astype('u1')");
                img.GetByte(5, 7, 1).Should().Be((byte)Math.Min(255, original + 7), "brightened in place, no copy anywhere");
                py.crop = crop;
                PyBool("np.shares_memory(crop, img)").Should().BeTrue("a strided window over the same buffer");
                double cropMean = py.crop.mean();
                cropMean.Should().BeApproximately(np.mean(crop).GetDouble(), 1e-9, "both sides agree");
                py.Exec("del img, crop");
            }
        }

        [TestMethod]
        public void Scenario3_TelemetryRing_ABoundedLiveCount()
        {
            using var scope = NDScope.Open();
            NDArray ring = np.zeros(new Shape(256));
            var rnd = new Random(42);
            int baseline = NDArrayPythonInterop.LiveExports;
            int peak = 0;
            using (Gil())
            {
                for (int step = 0; step < 150; step++)
                using (NDScope.Open())
                {
                    int start = step * 17 % 192;
                    for (int k = 0; k < 16; k++) ring[start + k] = rnd.NextDouble() * 10;
                    NDArray window = ring[$"{start}:{start + 16}"];
                    py.win = window;
                    double std = py.Eval("win.std()");               // through Eval: a dynamic `py.win` would hand out a wrapper pinning the previous window until a GC
                    std.Should().BeApproximately(np.std(window).GetDouble(), 1e-9, $"step {step}");
                    peak = Math.Max(peak, NDArrayPythonInterop.LiveExports);
                }

                py.Exec("del win");
            }

            WaitFor(() => NDArrayPythonInterop.LiveExports == baseline).Should().BeTrue("the live count drains back");
            (peak - baseline).Should().BeLessThan(8, "rebinding `win` drops the previous export: the count never grows unbounded");
        }

        [TestMethod]
        public void Scenario4_APythonDatasetOutlivesItsPythonReferences_UnderNumSharpKernels()
        {
            using var scope = NDScope.Open();
            NDArray x, y;
            using (Gil())
            {
                py.Exec("ds = {'x': np.arange(12, dtype='f8').reshape(3, 4) * 0.5, 'y': np.arange(3, dtype='i8') * 10}");
                x = py.ds["x"];
                y = py.ds["y"];
                py.Exec("del ds\nimport gc; gc.collect()");
            }

            np.sum(x).GetDouble().Should().Be(33.0, "sum runs over leased Python memory");
            np.mean(x).GetDouble().Should().Be(2.75);
            np.sum(y).GetInt64().Should().Be(30);
            NDArray scaled = x * 2.0;
            scaled.GetDouble(2, 3).Should().Be(11.0, "elementwise kernels read the lease and write fresh NumSharp memory");
            x.GetDouble(2, 3).Should().Be(5.5);
        }

        [TestMethod]
        public void Scenario5_CoSimulation_MixedOwnershipStaysCoherent()
        {
            using var scope = NDScope.Open();
            NDArray pos = np.zeros(new Shape(8));
            NDArray vel;
            using (Gil())
            {
                py.pos = pos;
                py.Exec("vel = np.arange(8, dtype='f8') - 3.5");
                vel = py.vel;
                for (int step = 0; step < 20; step++)
                using (NDScope.Open())
                {
                    py.Exec("pos += vel * 0.1");
                    np.copyto(vel, vel * 0.9);
                }

                double decay = Math.Pow(0.9, 20);
                for (int i = 0; i < 8; i++)
                {
                    double v0 = i - 3.5;
                    pos.GetDouble(i).Should().BeApproximately(v0 * (1 - decay), 1e-9, $"pos[{i}]: C# reads its own buffer");
                    PyFloat($"vel[{i}]").Should().BeApproximately(v0 * decay, 1e-9, $"vel[{i}]: Python reads its own");
                }

                py.Exec("del pos");
            }
        }

        [TestMethod]
        public void Scenario6_ALongLivedServiceThroughTheCodec()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("def process(a):\n    return a * 2.0 + 1.0");
                for (int i = 0; i < 40; i++)
                using (NDScope.Open())
                {
                    NDArray batch = np.arange(6).astype(np.float64) + i;
                    NDArray result = py.process(batch);
                    for (int j = 0; j < 6; j++)
                        result.GetDouble(j).Should().Be((j + i) * 2.0 + 1.0, $"call {i}, element {j}");
                }
            }
        }

        [TestMethod]
        public void Scenario7_GeneratorsStreamChunksIntoNumSharp()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("def chunks(n):\n    for i in range(n):\n        yield np.arange(3, dtype='f8') + 10 * i");
                var rows = new List<NDArray>();
                foreach (PyObject chunk in new PyIterable(py.chunks(4)))
                    rows.Add(chunk.ToNDArray());
                NDArray stacked = np.vstack(rows.ToArray());
                stacked.shape.Should().Equal(4, 3);
                stacked.GetDouble(3, 2).Should().Be(32.0);
            }
        }

        [TestMethod]
        public void Scenario8_Exceptions_Json_FaultyInputs()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.ex = np.arange(3).astype(np.float64);
                Throws(() => py.Exec("raise ValueError('bad shape: ' + str(ex.shape))")).Should().BeOfType<PythonException>()
                    .Which.Message.Should().Contain("bad shape: (3,)");
                py.Exec("ex[0] = 7.5");
                double sum = py.ex.sum();
                sum.Should().Be(10.5, "the engine stays usable after a Python raise");

                py.jx = np.arange(5).astype(np.float64) * 1.5;
                py.Exec("import json\npayload = json.dumps(jx.tolist())");
                PyStr("payload").Should().Be("[0.0, 1.5, 3.0, 4.5, 6.0]");
                NDArray back = py.Eval("np.asarray(json.loads(payload), dtype='f8')");
                back.GetDouble(4).Should().Be(6.0, "json.loads -> np.asarray -> NDArray is lossless");

                int exports = NDArrayPythonInterop.LiveExports, imports = NDArrayPythonInterop.LiveImports;
                using PyObject bytes = py.Eval("b'abcd'");
                Throws(() => bytes.AsNDArray()).Should().BeOfType<InvalidOperationException>();
                using PyObject dict = py.Eval("{'a': 1}");
                Throws(() => dict.ToNDArray()).Should().BeOfType<NotSupportedException>();
                using PyObject rec = py.Eval("np.zeros(2, dtype=[('x', 'i4'), ('y', 'f8')])");
                Throws(() => rec.ToNDArray()).Should().BeOfType<NotSupportedException>();
                NDArray dead = np.arange(3);
                dead.Dispose();
                Throws(() => dead.ToNumpy()).Should().BeOfType<ObjectDisposedException>();
                NDArrayPythonInterop.LiveExports.Should().Be(exports, "no pin survives a failed conversion");
                NDArrayPythonInterop.LiveImports.Should().Be(imports, "no lease survives a failed conversion");
                py.Exec("del ex, jx");
            }
        }
    }
}
