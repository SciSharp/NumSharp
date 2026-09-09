using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The <c>LiveExports</c> / <c>LiveImports</c> counters and the ARC refcount are lock-free
    ///     (<c>Interlocked</c>); this hammers them from many threads at once and asserts they settle exactly
    ///     back to baseline — a create/dispose race would leave a non-zero count or an orphaned pin.
    /// </summary>
    [TestClass]
    public class ConcurrencyTests : OnnxTestBase
    {
        private const int Iterations = 400;

        [TestMethod]
        public void ParallelExportsOnOneSharedBuffer_EachHoldsItsOwnPin_AllReleased()
        {
            using NDArray shared = Arange(NPTypeCode.Double, 64);
            var errors = new ConcurrentBag<Exception>();

            Parallel.For(0, Iterations, i =>
            {
                try
                {
                    using OrtTensor t = shared.AsOrtValue();
                    double v = t.Value.GetTensorDataAsSpan<double>()[i % 64];
                    if (v != (i % 64))
                        throw new Exception($"read {v} at {i % 64}");
                }
                catch (Exception e) { errors.Add(e); }
            });

            errors.Should().BeEmpty();
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "every parallel export handle was disposed");
            shared.GetData().IsUniquelyReferenced.Should().BeTrue("all pins released; the array owns its buffer alone again");
        }

        [TestMethod]
        public void ParallelExportsOnDistinctBuffers_SettleToZero()
        {
            var errors = new ConcurrentBag<Exception>();

            Parallel.For(0, Iterations, i =>
            {
                try
                {
                    using NDArray nd = Arange(NPTypeCode.Int32, 3, 4);
                    using OrtTensor t = nd.AsOrtValue();
                    if (t.Value.GetTensorDataAsSpan<int>()[11] != 11)
                        throw new Exception("bad tail");
                }
                catch (Exception e) { errors.Add(e); }
            });

            errors.Should().BeEmpty();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ParallelImportsOverAManagedTensor_LeasesAllReleased()
        {
            var tensor = new DenseTensor<float>(new float[] { 0, 1, 2, 3, 4, 5 }, new[] { 2, 3 });
            var errors = new ConcurrentBag<Exception>();

            Parallel.For(0, Iterations, i =>
            {
                try
                {
                    using NDArray view = tensor.AsNDArray();   // pins the managed buffer for the view's life
                    if (view.GetSingle(1, 2) != 5f)
                        throw new Exception("bad element");
                }
                catch (Exception e) { errors.Add(e); }
            });

            errors.Should().BeEmpty();
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "every parallel view released its lease");
        }

        [TestMethod]
        public void ParallelMixedExportAndImport_SettleToZero()
        {
            using NDArray shared = Arange(NPTypeCode.Single, 32);
            var errors = new ConcurrentBag<Exception>();

            Parallel.For(0, Iterations, i =>
            {
                try
                {
                    if ((i & 1) == 0)
                    {
                        using OrtTensor t = shared.AsOrtValue();
                        _ = t.Value.GetTensorDataAsSpan<float>()[0];
                    }
                    else
                    {
                        using OrtTensor t = shared.AsOrtValue();
                        using NDArray view = t.Value.AsNDArray();   // a view over the exported buffer -> a lease
                        _ = view.GetSingle(31);
                    }
                }
                catch (Exception e) { errors.Add(e); }
            });

            errors.Should().BeEmpty();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }
    }
}
