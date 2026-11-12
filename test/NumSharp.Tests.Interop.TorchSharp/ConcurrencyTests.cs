using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     Exports and imports from many threads at once: every lease is released exactly once, whichever thread libtorch or
    ///     NumSharp releases it on, and every crossing reads its own values.
    /// </summary>
    [TestClass]
    public class ConcurrencyTests : TorchTestBase
    {
        /// <summary>Eight threads export the same array and their own arrays, and dispose the tensors, 200 times each.</summary>
        [TestMethod]
        public void ParallelExports_ReleaseEveryLease()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray shared = np.arange(256).astype(np.float32);
            var failures = new ConcurrentBag<string>();
            Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, worker =>
            {
                for (int i = 0; i < 200; i++)
                {
                    using (Tensor t = shared.AsTensor(TorchCodecMode.View))
                        if (At(t, 255) != 255)
                            failures.Add($"worker {worker}: a shared export read the wrong value");
                    using NDArray own = np.full(new Shape(64), (float)worker, np.float32);
                    using (Tensor t = own.T.AsTensor())
                        if (At(t, 63) != worker)
                            failures.Add($"worker {worker}: a private export read another thread's value");
                }
            });
            failures.Should().BeEmpty();
            NDArrayTorchInterop.LiveExports.Should().Be(before);
        }

        /// <summary>Eight threads view the same tensor and their own tensors, and dispose the views, 200 times each.</summary>
        [TestMethod]
        public void ParallelImports_ReleaseEveryLease()
        {
            int before = NDArrayTorchInterop.LiveImports;
            using Tensor shared = torch.arange(256, dtype: ScalarType.Float64);
            var failures = new ConcurrentBag<string>();
            Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, worker =>
            {
                for (int i = 0; i < 200; i++)
                {
                    using (NDArray nd = shared.AsNDArray(TorchCodecMode.View))
                        if (nd.GetDouble(255) != 255)
                            failures.Add($"worker {worker}: a shared view read the wrong value");
                    using Tensor own = torch.full(new long[] { 8, 8 }, (double)worker);
                    using (var scope = Scope())
                    using (NDArray nd = own.t().AsNDArray())
                        if (nd.GetDouble(7, 7) != worker)
                            failures.Add($"worker {worker}: a private view read another thread's value");
                }
            });
            failures.Should().BeEmpty();
            NDArrayTorchInterop.LiveImports.Should().Be(before);
        }

        /// <summary>
        ///     The release can run on another thread than the export: a tensor exported on one thread and disposed on another
        ///     releases its lease exactly once.
        /// </summary>
        [TestMethod]
        public void CrossThreadDispose_ReleasesOnce()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray nd = np.ones(new Shape(32), np.float32);
            var tensors = new Tensor[64];
            for (int i = 0; i < tensors.Length; i++)
                tensors[i] = nd.AsTensor().DetachFromDisposeScope();
            NDArrayTorchInterop.LiveExports.Should().Be(before + tensors.Length);
            Parallel.For(0, tensors.Length, i => tensors[i].Dispose());
            NDArrayTorchInterop.LiveExports.Should().Be(before);
        }
    }
}
