using System;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The failure and cleanup paths no ordinary crossing takes, driven through the package's internal seams: every one
    ///     must leave the lease counters balanced - a reference given back exactly once, never twice, never not at all.
    /// </summary>
    [TestClass]
    public class FailurePathTests : TorchTestBase
    {
        /// <summary>
        ///     A TorchSharp whose THSTensor_new form the bridge does not know: every export is refused with the reason, and the
        ///     reference the export took is given back (no lease left open, the array intact).
        /// </summary>
        [TestMethod]
        public void UnknownNativeAbi_RefusesEveryExport_AndReleasesTheReference()
        {
            using NDArray nd = np.arange(6).astype(np.float32);
            int before = NDArrayTorchInterop.LiveExports;
            int saved = LibTorchSharpNative._abi;
            LibTorchSharpNative._abi = (int)LibTorchSharpNative.NewTensorAbi.Unknown;
            try
            {
                foreach (TorchCodecMode mode in StateMatrixSupport.Modes)
                {
                    Action act = () => nd.AsTensor(mode);
                    act.Should().Throw<NotSupportedException>().WithMessage("*THSTensor_new this bridge does not know*");
                    NDArrayTorchInterop.LiveExports.Should().Be(before, $"{mode}: the refused export gave its reference back");
                }
            }
            finally
            {
                LibTorchSharpNative._abi = saved;
            }
            nd.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f, 4f, 5f);
            using Tensor ok = nd.AsTensor();   // the real form works again
            nd.MayShareMemory(ok).Should().BeTrue();
        }

        /// <summary>libtorch refusing from_blob (a negative size): the lease is released once, and libtorch's own message surfaces.</summary>
        [TestMethod]
        public unsafe void FromBlobRefused_ReleasesTheLeaseOnce()
        {
            using NDArray nd = np.arange(4).astype(np.float32);
            int before = NDArrayTorchInterop.LiveExports;
            IArraySlice slice = nd.Storage.InternalArray;
            slice.TryAddRef().Should().BeTrue();
            var lease = new ExportLease(slice);
            NDArrayTorchInterop.LiveExports.Should().Be(before + 1);
            Action act = () => NDArrayTorchInterop.FromBlob(slice.Address, new long[] { -1 }, ScalarType.Float32, lease);
            act.Should().Throw<ExternalException>("libtorch validates the sizes and records its error");
            lease.IsReleased.Should().BeTrue();
            NDArrayTorchInterop.LiveExports.Should().Be(before, "released exactly once");
            nd.ToArray<float>().Should().Equal(new[] { 0f, 1f, 2f, 3f }, "the array kept its own reference");
        }

        /// <summary>An export lease released twice (the deleter after a cleanup, say) gives its reference back once.</summary>
        [TestMethod]
        public void ExportLease_ReleaseIsIdempotent()
        {
            using NDArray nd = np.ones(new Shape(3), np.float64);
            int before = NDArrayTorchInterop.LiveExports;
            int live = ExportLease.LiveCount;
            nd.Storage.InternalArray.TryAddRef().Should().BeTrue();
            var lease = new ExportLease(nd.Storage.InternalArray);
            ExportLease.LiveCount.Should().Be(live + 1, "the lease roots its deleter until it is released");
            lease.Release();
            lease.Release();
            lease.IsReleased.Should().BeTrue();
            NDArrayTorchInterop.LiveExports.Should().Be(before);
            ExportLease.LiveCount.Should().Be(live);
            ((double)np.sum(nd)).Should().Be(3, "one reference given back, not two");
        }

        /// <summary>An import lease released twice disposes its handle once and gives its booking back once.</summary>
        [TestMethod]
        public void ImportLease_ReleaseIsIdempotent()
        {
            using Tensor t = torch.ones(8);
            int before = NDArrayTorchInterop.LiveImports;
            long booked = NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.OutstandingForeignBytes;
            IntPtr handle = LibTorchSharpNative.THSTensor_detach(t.Handle);
            var lease = new ImportLease(handle, 32);
            lease.Release();
            lease.Release();
            lease.IsReleased.Should().BeTrue();
            NDArrayTorchInterop.LiveImports.Should().Be(before);
            NumSharp.Backends.Unmanaged.Pooling.SizeBucketedBufferPool.OutstandingForeignBytes.Should().Be(booked);
            t.sum().item<float>().Should().Be(8, "the caller's tensor is untouched by the lease's handle");
        }

        /// <summary>A handle whose tensor has no addressable storage (a meta tensor's): refused, the handle disposed, nothing leased.</summary>
        [TestMethod]
        public void AdoptHandle_WithoutStorage_RefusesAndDisposesTheHandle()
        {
            using Tensor meta = torch.empty(new long[] { 3 }, device: new torch.Device(DeviceType.META));
            int before = NDArrayTorchInterop.LiveImports;
            IntPtr handle = LibTorchSharpNative.THSTensor_detach(meta.Handle);
            Action act = () => NDArrayTorchInterop.AdoptHandle(handle, NPTypeCode.Single, new long[] { 3 }, new long[] { 1 }, readOnly: false);
            act.Should().Throw<NotSupportedException>().WithMessage("*no addressable storage*");
            NDArrayTorchInterop.LiveImports.Should().Be(before);
        }

        /// <summary>The transient view of the copy paths refuses a tensor without addressable storage the same way.</summary>
        [TestMethod]
        public void Transient_WithoutStorage_Refuses()
        {
            using Tensor meta = torch.empty(new long[] { 2 }, device: new torch.Device(DeviceType.META));
            Action act = () => NDArrayTorchInterop.Transient(meta, NPTypeCode.Single, new long[] { 2 }, new long[] { 1 });
            act.Should().Throw<NotSupportedException>().WithMessage("*no addressable storage*");
        }

        /// <summary>An Auto import of an EMPTY conjugated tensor: its resolved intermediate has nothing to view, so a fresh empty array.</summary>
        [TestMethod]
        public void EmptyConjugatedTensor_ImportsAsAFreshEmptyArray()
        {
            using var scope = Scope();
            Tensor c = torch.zeros(new long[] { 0, 3 }, dtype: ScalarType.ComplexFloat64).conj();
            c.is_conj().Should().BeTrue();
            using NDArray nd = c.AsNDArray();
            nd.shape.Should().Equal(new long[] { 0, 3 });
            nd.flags.owndata.Should().BeTrue();
        }

        /// <summary>ToNDArray is AsNDArray(Copy): an owning, contiguous, independent array.</summary>
        [TestMethod]
        public void ToNDArray_IsTheCopyMode()
        {
            using Tensor t = torch.arange(6, dtype: ScalarType.Int32).reshape(2, 3).t();
            using NDArray nd = t.ToNDArray();
            nd.flags.owndata.Should().BeTrue();
            nd.flags.c_contiguous.Should().BeTrue();
            nd.MayShareMemory(t).Should().BeFalse();
            SameBits(nd, t).Should().BeTrue();
        }

        /// <summary>MayShareMemory refuses nulls and disposed operands, and answers false for empty or off-CPU tensors.</summary>
        [TestMethod]
        public void MayShareMemory_Arguments()
        {
            using NDArray nd = np.arange(3).astype(np.float32);
            using Tensor t = torch.ones(3);
            ((Action)(() => ((NDArray)null).MayShareMemory(t))).Should().Throw<ArgumentNullException>();
            ((Action)(() => nd.MayShareMemory(null))).Should().Throw<ArgumentNullException>();
            Tensor gone = torch.ones(2);
            gone.Dispose();
            ((Action)(() => nd.MayShareMemory(gone))).Should().Throw<ObjectDisposedException>();
            NDArray released = np.arange(2).astype(np.float32);
            released.Dispose();
            ((Action)(() => released.MayShareMemory(t))).Should().Throw<ObjectDisposedException>();
            using Tensor meta = torch.empty(new long[] { 3 }, device: new torch.Device(DeviceType.META));
            nd.MayShareMemory(meta).Should().BeFalse("a meta tensor has no memory NumSharp can address");
            using Tensor empty = torch.empty(new long[] { 0 });
            nd.MayShareMemory(empty).Should().BeFalse();
        }

        /// <summary>The reach of an empty layout is 0 (a view over it would address nothing).</summary>
        [TestMethod]
        public void ReachOfAnEmptyLayout_IsZero()
        {
            NDArrayTorchInterop.ReachFrom(new long[] { 2, 0, 3 }, new long[] { 3, 3, 1 }).Should().Be(0);
            NDArrayTorchInterop.ReachFrom(new long[] { 3, 4 }, new long[] { 1, 1 }).Should().Be(6, "overlapping windows reach 1 + 2 + 3 elements");
            NDArrayTorchInterop.ReachFrom(Array.Empty<long>(), Array.Empty<long>()).Should().Be(1, "a 0-d layout holds one element");
        }
    }
}
