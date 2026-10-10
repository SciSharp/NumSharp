using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The small internal shape / memory helpers, tested in isolation — including the
    ///     <see cref="NDArrayOnnxInterop.IsCpuAccessible"/> device-refusal branch that a GPU test would
    ///     otherwise be needed for (a constructed <see cref="OrtMemoryInfo"/> exercises it with no device).
    /// </summary>
    [TestClass]
    public class HelperEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void ShapeFrom_EmptyOrNull_IsTheScalar_AndDimsRoundTrip()
        {
            NDArrayOnnxInterop.ShapeFrom(null).NDim.Should().Be(0, "no dims == the 0-d scalar");
            NDArrayOnnxInterop.ShapeFrom(Array.Empty<long>()).NDim.Should().Be(0);

            Shape s = NDArrayOnnxInterop.ShapeFrom(new long[] { 2, 3, 4 });
            s.NDim.Should().Be(3);
            s.Dimensions.Should().Equal(2L, 3L, 4L);

            Shape z = NDArrayOnnxInterop.ShapeFrom(new long[] { 0, 4 });
            z.Dimensions.Should().Equal(0L, 4L);
            z.size.Should().Be(0);
        }

        [TestMethod]
        public void ShapeFrom_NegativeDim_IsRefused_ConcreteTensorsHaveNoSymbolicDims()
        {
            new Action(() => NDArrayOnnxInterop.ShapeFrom(new long[] { -1, 3 })).Should().Throw<ArgumentException>()
                .WithMessage("*negative dimension*");
            new Action(() => NDArrayOnnxInterop.ShapeFrom(new long[] { 3, -1 })).Should().Throw<ArgumentException>()
                .WithMessage("*negative dimension*");
        }

        [TestMethod]
        public void LongDims_And_IntDims_And_FortranStrides()
        {
            NDArrayOnnxInterop.LongDims(Shape.Scalar).Should().BeEmpty("0-d -> empty dims");
            NDArrayOnnxInterop.LongDims(new Shape(2, 3)).Should().Equal(2L, 3L);

            NDArrayOnnxInterop.IntDims(Shape.Scalar, "x").Should().BeEmpty();
            NDArrayOnnxInterop.IntDims(new Shape(2, 3, 4), "x").Should().Equal(2, 3, 4);

            // column-major (Fortran) element strides: first axis fastest
            NDArrayOnnxInterop.FortranStrides(new long[] { 2, 3 }).Should().Equal(1L, 2L);
            NDArrayOnnxInterop.FortranStrides(new long[] { 4, 5, 6 }).Should().Equal(1L, 4L, 20L);
        }

        [TestMethod]
        public void IsCpuAccessible_AcceptsCpuMemory_RefusesDeviceMemory_NoGpuNeeded()
        {
            NDArrayOnnxInterop.IsCpuAccessible(OrtMemoryInfo.DefaultInstance).Should().BeTrue("the default CPU allocator");

            // the arena "Cpu" allocator is NOT DefaultInstance but must still be accepted (the memInfo.Name branch)
            using (var arenaCpu = new OrtMemoryInfo("Cpu", OrtAllocatorType.ArenaAllocator, 0, OrtMemType.Default))
                NDArrayOnnxInterop.IsCpuAccessible(arenaCpu).Should().BeTrue("a CPU arena allocator is CPU-addressable");

            // host-accessible device memory (CUDA pinned) is fine — it is CpuOutput / CpuInput memory
            using (var cudaPinned = new OrtMemoryInfo("CudaPinned", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuOutput))
                NDArrayOnnxInterop.IsCpuAccessible(cudaPinned).Should().BeTrue("CpuOutput memory of a device EP is host-accessible");
            using (var cudaPinnedIn = new OrtMemoryInfo("CudaPinned", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.CpuInput))
                NDArrayOnnxInterop.IsCpuAccessible(cudaPinnedIn).Should().BeTrue();

            // genuine device memory is NOT CPU-addressable — this is the branch ToNDArray/AsNDArray refuse on
            using (var cuda = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default))
                NDArrayOnnxInterop.IsCpuAccessible(cuda).Should().BeFalse("a device-resident tensor's memory the CPU cannot address");
        }

        /// <summary>
        ///     Builds an <see cref="OrtMemoryInfo"/> for an allocator NAME with default memory type, or returns <c>null</c>
        ///     when the loaded ONNX Runtime does not know the name (it refuses to describe a device it has no entry for;
        ///     <c>CpuAligned4K</c> and <c>QnnHtpShared</c> are newer than the floor version).
        /// </summary>
        /// <param name="name">ORT's allocator name.</param>
        /// <returns>The memory info, which the caller disposes, or <c>null</c>.</returns>
        private static OrtMemoryInfo TryDescribe(string name)
        {
            try
            {
                return new OrtMemoryInfo(name, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);
            }
            catch (OnnxRuntimeException)
            {
                return null;
            }
        }

        /// <summary>
        ///     Every allocator the CPU can read is accepted, by the rule that fits it: a CPU device's own memory by its
        ///     NAME (<c>Cpu</c>, and <c>CpuAligned4K</c>, which was refused), memory a device shares with the host by ORT's
        ///     own <c>HOST_ACCESSIBLE</c> answer or, on an ONNX Runtime too old to ask, by name (<c>QnnHtpShared</c>, which
        ///     was refused; <c>CudaPinned</c> / <c>HipPinned</c> even when their memory type is reported as default). The
        ///     default memory of a GPU provider stays refused. A name the loaded ORT cannot describe is skipped.
        /// </summary>
        [TestMethod]
        public void IsCpuAccessible_AcceptsEveryHostReadableAllocator_AndRefusesDeviceDefaults()
        {
            var seen = new System.Collections.Generic.List<string>();
            foreach (string name in new[] { "Cpu", "CpuAligned4K", "QnnHtpShared", "CudaPinned", "HipPinned" })
            {
                using OrtMemoryInfo info = TryDescribe(name);
                if (info is null)
                    continue;
                NDArrayOnnxInterop.IsCpuAccessible(info).Should().BeTrue($"'{name}' memory is CPU-addressable");
                seen.Add(name);
            }
            seen.Should().Contain("Cpu").And.Contain("CudaPinned", "every supported ONNX Runtime describes these two");
            Console.WriteLine("host-readable allocators this ONNX Runtime describes: " + string.Join(", ", seen));

            int refused = 0;
            foreach (string name in new[] { "Cuda", "DML", "Hip", "OpenVINO_GPU" })
            {
                using OrtMemoryInfo info = TryDescribe(name);
                if (info is null)
                    continue;
                NDArrayOnnxInterop.IsCpuAccessible(info).Should().BeFalse($"'{name}' is a device's own memory");
                refused++;
            }
            refused.Should().BeGreaterThan(0, "at least 'Cuda' is describable on every supported ONNX Runtime");
        }

        /// <summary>
        ///     The refusal end to end, with no GPU: a tensor whose memory info CLAIMS device memory is refused by both
        ///     imports in every mode (no mode makes device memory readable) and is left alive, even when the import was
        ///     asked to own it. A tensor that claims the 4 KB-aligned CPU allocator is read. Without the check an import
        ///     would read a device pointer through the CPU and fault.
        /// </summary>
        [TestMethod]
        public unsafe void TensorClaimingDeviceMemory_IsRefusedEndToEnd_AndAlignedCpuMemoryIsRead()
        {
            using NDArray backing = Arange(NPTypeCode.Single, 4);
            var address = (IntPtr)backing.Storage.InternalArray.Address;

            using (var cuda = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default))
            using (OrtValue onDevice = OrtValue.CreateTensorValueWithData(cuda, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, new long[] { 4 }, address, 16))
            {
                new Action(() => onDevice.ToNDArray()).Should().Throw<InvalidOperationException>().WithMessage("*ToNDArray*'Cuda'*cannot address*");
                new Action(() => onDevice.AsNDArray()).Should().Throw<InvalidOperationException>().WithMessage("*AsNDArray*'Cuda'*cannot address*");
                new Action(() => onDevice.AsNDArray(ownsValue: true)).Should().Throw<InvalidOperationException>();
                new Action(() => onDevice.AsNDArray(mode: OnnxCodecMode.Copy)).Should().Throw<InvalidOperationException>("a copy would read the device pointer too");
                new Action(() => onDevice.AsNDArray(ownsValue: true, mode: OnnxCodecMode.Copy)).Should().Throw<InvalidOperationException>();
                onDevice.GetTensorTypeAndShape().ElementCount.Should().Be(4, "a refused owning import must not dispose the value");
            }

            using (OrtMemoryInfo aligned = TryDescribe("CpuAligned4K"))
            {
                if (aligned is not null)
                {
                    using OrtValue onCpu = OrtValue.CreateTensorValueWithData(aligned, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, new long[] { 4 }, address, 16);
                    using NDArray copy = onCpu.ToNDArray();
                    copy.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f);
                    using NDArray view = onCpu.AsNDArray();
                    view.GetSingle(3).Should().Be(3f);
                }
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0);
            GC.KeepAlive(backing);
        }
    }
}
