using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Made-up but realistic applications driven end-to-end through the interop. Each scenario
    ///     stresses a usage shape a real app would hit; the base-class cleanup gate additionally fails
    ///     any scenario that leaks a conversion or frees memory early.
    /// </summary>
    [TestClass]
    public class ScenarioTests : InteropTestBase
    {
        /// <summary>
        ///     "ML inference": C# owns feature engineering, Python owns the model. Features cross
        ///     zero-copy, predictions come back as NumSharp, and every batch is cross-checked against
        ///     a pure-NumSharp recomputation of the same math.
        /// </summary>
        [TestMethod]
        public void MlInference_PythonModelMatchesNumSharpMath_AcrossBatches()
        {
            var weights = new NDArray(new double[] { 0.5, -1.0, 2.0, 0.25 });
            ExportCopyTo("w", weights);

            for (int b = 0; b < 10; b++)
            {
                var feats = np.arange(b, b + 12).astype(NPTypeCode.Double).reshape(3, 4) / 4.0;
                ExportTo("batch", feats);
                PyExec("pred = batch.dot(w) - batch.mean()");

                var pred = ImportOf("pred");
                var expected = np.dot(feats, weights) - np.mean(feats);

                for (int i = 0; i < 3; i++)
                    ReadAt<double>(pred, i).Should().BeApproximately(ReadAt<double>(expected, i), 1e-9,
                        $"batch {b}, row {i}: python-computed prediction must equal the NumSharp recomputation");
            }
        }

        /// <summary>
        ///     "Image editor": a C#-owned uint8 image is brightened IN PLACE by Python (no copies
        ///     anywhere), then C# crops a center region by slicing and Python computes statistics on the
        ///     exported crop — still the same single buffer.
        /// </summary>
        [TestMethod]
        public void ImagePipeline_InPlaceBrightness_ThenCropStatistics()
        {
            var img = np.arange(32 * 32 * 3).astype(NPTypeCode.Byte).reshape(32, 32, 3);
            ExportTo("img", img);

            PyExec("img[:] = np.minimum(img.astype('i2') + 7, 255).astype('u1')");

            foreach (var (r, c, ch) in new[] { (5L, 7L, 1L), (0L, 0L, 0L), (31L, 31L, 2L) })
            {
                byte original = (byte)(((r * 32 + c) * 3 + ch) % 256);
                byte expected = (byte)Math.Min(255, original + 7);
                ReadAt<byte>(img, r, c, ch).Should().Be(expected, $"pixel ({r},{c},{ch}) after python's in-place edit");
            }

            var crop = img["8:24, 8:24, :"];
            ExportTo("crop", crop);
            PyBool("np.shares_memory(crop, img)").Should().BeTrue("the crop is a strided window, not a copy");

            double pyMean = PyFloat("float(crop.mean())");
            double nsMean = ReadAt<double>(np.mean(crop));
            pyMean.Should().BeApproximately(nsMean, 1e-9, "both sides must agree on the crop statistics");
        }

        /// <summary>
        ///     "Telemetry service": a long-lived C# ring buffer feeds sliding windows to a Python
        ///     anomaly detector for hundreds of iterations. Window exports are churned constantly;
        ///     live-export count must stay bounded and drain to zero.
        /// </summary>
        [TestMethod]
        public void TelemetryRing_SlidingWindowAnalytics_NoLeakGrowth()
        {
            int e0 = NDArrayPythonInterop.LiveExports;
            var ring = np.zeros(new Shape(256));
            var rnd = new Random(42);

            for (int step = 0; step < 150; step++)
            {
                int start = step * 17 % 192;
                for (int k = 0; k < 16; k++)
                    WriteAt(ring, rnd.NextDouble() * 10, start + k);

                var window = ring[$"{start}:{start + 16}"];
                ExportTo("w", window);

                PyFloat("float(w.mean())").Should().BeApproximately(ReadAt<double>(np.mean(window)), 1e-9, $"step {step} mean");
                PyFloat("float(w.std())").Should().BeApproximately(ReadAt<double>(np.std(window)), 1e-9, $"step {step} std");

                if (step % 50 == 49)
                {
                    Pump();
                    NDArrayPythonInterop.LiveExports.Should().BeLessThan(e0 + 8,
                        "rebinding the scope name must release prior window exports — no unbounded growth");
                }
            }

            PyExec("del w");
            WaitFor(() => NDArrayPythonInterop.LiveExports == e0).Should().BeTrue("all window exports must drain");
        }

        /// <summary>
        ///     "Dataset handoff": Python builds a dataset, C# takes zero-copy views, and Python then
        ///     forgets every reference. The leases must keep the data alive, and NumSharp's own compute
        ///     kernels (sum/mean/elementwise) must run correctly over the Python-owned memory.
        /// </summary>
        [TestMethod]
        public void PythonDataset_OutlivesItsPythonReferences_UnderNumSharpKernels()
        {
            PyExec("ds = {'x': np.arange(12, dtype='f8').reshape(3, 4) * 0.5, 'y': np.arange(3, dtype='i8') * 10}");

            var x = ViewOf("ds['x']");
            var y = ViewOf("ds['y']");

            PyExec("del ds");
            Pump(); Pump();   // python-side GC: nothing but our leases keeps the arrays now

            ReadAt<double>(np.sum(x)).Should().BeApproximately(33.0, 1e-9, "sum kernel over leased python memory");
            ReadAt<double>(np.mean(x)).Should().BeApproximately(2.75, 1e-9, "mean kernel over leased python memory");
            ReadAt<long>(np.sum(y)).Should().Be(30, "integer reduction over leased python memory");

            var scaled = x * 2.0;   // elementwise kernel reading the lease, writing a fresh NumSharp array
            ReadAt<double>(scaled, 2, 3).Should().BeApproximately(11.0, 1e-9);

            ReadAt<double>(x, 2, 3).Should().BeApproximately(5.5, 1e-12, "the dataset memory was never freed early");
        }

        /// <summary>
        ///     "Physics co-simulation": positions live in C# (exported to Python), velocities live in
        ///     Python (imported into C#). Each step Python integrates positions in place and C# damps
        ///     velocities in place through the lease (np.copyto into Python-owned memory). After 20
        ///     steps both sides must agree with the closed-form result.
        /// </summary>
        [TestMethod]
        public void SimulationLoop_MixedOwnershipState_StaysCoherent()
        {
            var pos = np.zeros(new Shape(8));
            ExportTo("pos", pos);
            PyExec("vel = np.arange(8, dtype='f8') - 3.5");
            var velNd = ViewOf("vel");

            for (int step = 0; step < 20; step++)
            {
                PyExec("pos += vel * 0.1");          // python writes C#-owned memory in place
                np.copyto(velNd, velNd * 0.9);       // C# writes python-owned memory in place
            }

            double decay = Math.Pow(0.9, 20);
            for (int i = 0; i < 8; i++)
            {
                double v0 = i - 3.5;
                ReadAt<double>(pos, i).Should().BeApproximately(v0 * (1 - decay), 1e-9, $"position {i} (C# reads its own buffer)");
                PyFloat($"float(vel[{i}])").Should().BeApproximately(v0 * decay, 1e-9, $"velocity {i} (python reads its own buffer)");
            }
        }

        /// <summary>
        ///     "Long-lived service with the codec": NDArrays flow into a Python callable and results
        ///     flow back with zero explicit conversion calls, many times over.
        /// </summary>
        [TestMethod]
        public void PythonService_CodecCallLoop_StaysCorrectAndBalanced()
        {
            CodecTests.EnsureCodec();
            PyExec("def process(a):\n    return a * 2.0 + 1.0");

            for (int i = 0; i < 40; i++)
            {
                var batch = np.arange(6).astype(NPTypeCode.Double) + i;

                NDArray result;
                using (Gil())
                {
                    dynamic process = Scope.Get("process");
                    using var r = (PyObject)process(batch);   // batch auto-encoded by the codec
                    result = r.As<NDArray>();                 // result auto-decoded by the codec
                }

                for (int j = 0; j < 6; j++)
                    ReadAt<double>(result, j).Should().BeApproximately((j + i) * 2.0 + 1.0, 1e-9, $"call {i}, element {j}");
            }
        }

        /// <summary>
        ///     "Bad inputs happen": every rejection path must throw cleanly, leave ZERO half-taken
        ///     leases or pins behind, and the engine must remain fully usable afterwards.
        /// </summary>
        [TestMethod]
        public void FaultyInputs_LeaveNoHalfTakenState_EngineStaysHealthy()
        {
            int e0 = NDArrayPythonInterop.LiveExports, i0 = NDArrayPythonInterop.LiveImports;

            ((Action)(() => NDArrayPythonInterop.ToNumpy(np.arange(3).astype(NPTypeCode.Decimal))))
                .Should().Throw<NotSupportedException>();
            ((Action)(() => ViewOf("b'abcd'")))
                .Should().Throw<InvalidOperationException>();
            ((Action)(() => ImportOf("{'a': 1}")))
                .Should().Throw<NotSupportedException>();
            ((Action)(() => ImportOf("np.arange(3).astype('>i4')")))
                .Should().Throw<NotSupportedException>();

            Pump();
            NDArrayPythonInterop.LiveExports.Should().Be(e0, "failed conversions must not pin buffers");
            NDArrayPythonInterop.LiveImports.Should().Be(i0, "failed conversions must not take leases");

            // The engine and the interop must be fully functional after the failures.
            var nd = np.arange(4).astype(NPTypeCode.Double);
            ExportTo("ok", nd);
            PyExec("ok[0] = 5.5");
            ReadAt<double>(nd, 0).Should().BeApproximately(5.5, 1e-12);

            PyExec("okv = np.arange(4, dtype='i8')");
            var v = ViewOf("okv");
            WriteAt(v, 9L, 0);
            PyLong("int(okv[0])").Should().Be(9);
        }
    }
}
