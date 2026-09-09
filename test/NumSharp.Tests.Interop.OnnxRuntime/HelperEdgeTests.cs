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
    }
}
