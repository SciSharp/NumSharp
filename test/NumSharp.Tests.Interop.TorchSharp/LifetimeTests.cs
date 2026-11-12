using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     Who keeps what alive: an exported tensor keeps the NumSharp buffer until libtorch frees the storage (dispose, scope,
    ///     finalizer, the last torch view); an imported view keeps the torch storage until the last NumSharp view goes, through a
    ///     <c>detach()</c> that survives the caller's dispose, <c>set_</c> and DisposeScope.
    /// </summary>
    [TestClass]
    public class LifetimeTests : TorchTestBase
    {
        /// <summary>The array may be disposed while torch reads it: the tensor's reference keeps the buffer out of the pool.</summary>
        [TestMethod]
        public void Export_SourceDisposedWhileTensorReads()
        {
            NDArray nd = np.arange(1 << 20).astype(np.float32);   // 4 MB: a pooled size another array would reuse at once
            using Tensor t = nd.AsTensor(TorchCodecMode.View);
            nd.Dispose();
            using NDArray thief = np.full(new Shape(1 << 20), 7f, np.float32);
            At(t, 12345).Should().Be(12345, "the buffer was not handed to the next same-size array");
        }

        /// <summary>A torch view of an exported tensor keeps the buffer after the tensor is disposed; the last one releases it.</summary>
        [TestMethod]
        public void Export_TorchViewsKeepTheBuffer()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray nd = np.arange(12).astype(np.float64).reshape(3, 4);
            Tensor t = nd.AsTensor();
            Tensor row = t[1];
            Tensor transposed = t.t();
            t.Dispose();
            NDArrayTorchInterop.LiveExports.Should().Be(before + 1, "two torch views still read the storage");
            At(row, 2).Should().Be(6);
            row.Dispose();
            NDArrayTorchInterop.LiveExports.Should().Be(before + 1, "the transposed view still reads it");
            transposed.Dispose();
            NDArrayTorchInterop.LiveExports.Should().Be(before, "the last tensor over the storage is gone");
        }

        /// <summary>A forgotten exported tensor releases the buffer when its finalizer frees the storage.</summary>
        [TestMethod]
        public void Export_ForgottenTensor_ReleasedByTheFinalizer()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray nd = np.ones(new Shape(64), np.float32);
            ExportAndForget(nd);
            WaitFor(() => NDArrayTorchInterop.LiveExports == before, 10_000).Should().BeTrue("TorchSharp's ~Tensor freed the storage");
        }

        /// <summary>Exports a tensor and drops it without disposing it (a frame of its own, so nothing roots it afterwards).</summary>
        /// <param name="nd">The array to export.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ExportAndForget(NDArray nd) => nd.AsTensor().DetachFromDisposeScope();

        /// <summary>NumPy's resize refcheck refuses to move a buffer torch still reads, and allows it once the tensor is gone.</summary>
        [TestMethod]
        public void Export_ResizeRefusedWhileTorchReads()
        {
            using NDArray nd = np.arange(8).astype(np.float64);
            using (Tensor t = nd.AsTensor())
            {
                Action resize = () => nd.resize(new Shape(16));
                resize.Should().Throw<Exception>().WithMessage("*referenced*");
            }
            Settle();
        }

        /// <summary>A view survives the tensor's dispose, and the lease ends with the last NumSharp view.</summary>
        [TestMethod]
        public void Import_TensorDisposedWhileViewReads()
        {
            int before = NDArrayTorchInterop.LiveImports;
            Tensor t = torch.arange(1 << 20, dtype: ScalarType.Float32);
            NDArray nd = t.AsNDArray();
            t.Dispose();
            NDArrayTorchInterop.LiveImports.Should().Be(before + 1);
            using (Tensor thief = torch.full(new long[] { 1 << 20 }, 7f))
                nd.GetSingle(54321).Should().Be(54321f, "the view's detach() kept the storage alive");
            nd.Dispose();
            NDArrayTorchInterop.LiveImports.Should().Be(before, "the last view released its handle");
        }

        /// <summary>set_ rebinds the caller's TensorImpl; the view's own detach() still holds the old storage.</summary>
        [TestMethod]
        public void Import_SetRebindsTheTensorNotTheView()
        {
            using Tensor t = torch.arange(1 << 21, dtype: ScalarType.Float32);   // 8 MB: freed memory is unmapped, not kept
            using NDArray nd = t.AsNDArray();
            using (Tensor other = torch.zeros(4))
                t.set_(other);
            t.numel().Should().Be(4, "the caller's tensor now reads other storage");
            nd.GetSingle(1_000_000).Should().Be(1_000_000f, "the view still reads the storage it was made over");
        }

        /// <summary>A view made inside a DisposeScope outlives the scope that disposed the tensor.</summary>
        [TestMethod]
        public void Import_ViewOutlivesTheTensorsDisposeScope()
        {
            NDArray nd;
            using (var scope = Scope())
            {
                Tensor t = torch.arange(10, dtype: ScalarType.Float64);
                nd = t.AsNDArray();
            }
            using (nd)
                ((double)np.sum(nd)).Should().Be(45, "the bridge's handle belongs to no scope");
        }

        /// <summary>A slice of a view keeps the lease after the view itself is disposed.</summary>
        [TestMethod]
        public void Import_DerivedSliceKeepsTheLease()
        {
            int before = NDArrayTorchInterop.LiveImports;
            using Tensor t = torch.arange(10, dtype: ScalarType.Int64);
            NDArray nd = t.AsNDArray();
            NDArray tail = nd["5:"];
            nd.Dispose();
            NDArrayTorchInterop.LiveImports.Should().Be(before + 1, "the slice is a NumSharp view of the same block");
            tail.GetInt64(0).Should().Be(5);
            tail.Dispose();
            WaitFor(() => NDArrayTorchInterop.LiveImports == before, 10_000).Should().BeTrue();
        }

        /// <summary>A forgotten view releases its lease when NumSharp's finalizers run.</summary>
        [TestMethod]
        public void Import_ForgottenView_ReleasedByTheFinalizer()
        {
            int before = NDArrayTorchInterop.LiveImports;
            using Tensor t = torch.ones(64);
            ImportAndForget(t);
            WaitFor(() => NDArrayTorchInterop.LiveImports == before, 10_000).Should().BeTrue();
        }

        /// <summary>Imports a view and drops it without disposing it.</summary>
        /// <param name="t">The tensor to view.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ImportAndForget(Tensor t) => t.AsNDArray();

        /// <summary>A view books its tensor's whole storage with NumSharp's memory accounting until it is released.</summary>
        [TestMethod]
        public void Import_BooksTheWholeStorage()
        {
            using Tensor big = torch.zeros(1 << 18, dtype: ScalarType.Float32);   // 1 MiB of storage
            using Tensor small = big.narrow(0, 10, 4);
            long before = SizeBucketedBufferPool.OutstandingForeignBytes;
            using (NDArray nd = small.AsNDArray())
                (SizeBucketedBufferPool.OutstandingForeignBytes - before).Should().Be(1 << 20, "the 4-element view keeps the whole storage alive");
            SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(before, "the release gave the booking back");
        }

        /// <summary>An Auto import of a private intermediate (here: a resolved conjugation) adopts it and releases it with the view.</summary>
        [TestMethod]
        public void Import_AdoptedIntermediate_IsReleasedWithTheView()
        {
            int before = NDArrayTorchInterop.LiveImports;

            using var scope = Scope();
            Tensor c = torch.complex(torch.tensor(new double[] { 1 }), torch.tensor(new double[] { 2 }));
            Tensor cj = c.conj();
            NDArray nd = cj.AsNDArray();
            NDArrayTorchInterop.LiveImports.Should().Be(before + 1, "the resolved copy is viewed, not copied again");
            nd.ToArray<System.Numerics.Complex>()[0].Should().Be(new System.Numerics.Complex(1, -2));
            nd.Dispose();
            NDArrayTorchInterop.LiveImports.Should().Be(before);
        }
    }
}
