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

        /// <summary>
        ///     Regression pin (found 2026-11-09 by KokoroSharp's device probe on a Galaxy S23 Ultra, ONNX Runtime 1.22.0 for
        ///     Android): the CPU device's memory under a provider's OWN allocator name is read. XNNPACK, the CPU accelerator of
        ///     ONNX Runtime's Android and iOS packages, describes its outputs as <c>XnnpackExecutionProvider</c> memory on the
        ///     CPU device with default memory; every import of an XNNPACK session's output (<c>AsNDArray</c>,
        ///     <c>ToNDArray</c>, <c>session.Run(…, zeroCopyOutputs: true)</c>) threw "the tensor lives in
        ///     'XnnpackExecutionProvider' memory (device 0, Default), which the CPU cannot address" while ONNX Runtime's own
        ///     <c>GetTensorDataAsSpan</c> read the same tensor. Described here with no XNNPACK through ONNX Runtime 1.23's
        ///     <c>CreateMemoryInfo_V2</c> (any name, a device type); Inconclusive on an older ONNX Runtime (the floor, 1.16.0,
        ///     cannot describe it). The real provider end to end: <c>NumSharp.Tests.Interop.OnnxRuntime.Android</c>.
        /// </summary>
        [TestMethod]
        public unsafe void IsCpuAccessible_CpuDeviceMemoryUnderAProvidersOwnName_IsRead()
        {
            using OrtMemoryInfo xnnpack = TryDescribe("XnnpackExecutionProvider", "CPU", "DEFAULT");
            if (xnnpack is null)
                Assert.Inconclusive("the loaded ONNX Runtime has no OrtMemoryInfo(name, OrtMemoryInfoDeviceType, …) constructor (CreateMemoryInfo_V2, 1.23+)");

            NDArrayOnnxInterop.IsCpuAccessible(xnnpack).Should().BeTrue("XNNPACK's outputs are the CPU device's own memory");

            using NDArray backing = Arange(NPTypeCode.Single, 4);
            var address = (IntPtr)backing.Storage.InternalArray.Address;
            using (OrtValue output = OrtValue.CreateTensorValueWithData(xnnpack, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, new long[] { 4 }, address, 16))
            {
                output.GetTensorDataAsSpan<float>()[3].Should().Be(3f, "ONNX Runtime itself reads it");
                using NDArray view = output.AsNDArray();
                view.GetSingle(3).Should().Be(3f);
                using NDArray copy = output.ToNDArray();
                copy.ToArray<float>().Should().Equal(0f, 1f, 2f, 3f);
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0);
            GC.KeepAlive(backing);
        }

        /// <summary>
        ///     Every execution provider's allocator, described the way the provider registers it in ONNX Runtime's sources
        ///     (device type, memory type, name): the CPU-device ones are read, end to end through <c>AsNDArray</c> and
        ///     <c>ToNDArray</c> over a tensor in that memory; a GPU's or NPU's own memory stays refused, under its real name
        ///     and under a name the bridge has never seen. Inconclusive on an ONNX Runtime without <c>CreateMemoryInfo_V2</c>
        ///     (before 1.23).
        /// </summary>
        /// <remarks>
        ///     The rows follow the audit recorded on <c>NDArrayOnnxInterop.CpuAllocatorNames</c>. A refused row never reads
        ///     its tensor (the memory here is ordinary host memory, which a wrong acceptance would read without faulting, so
        ///     the assertion is on the decision and the thrown message, not on a fault).
        /// </remarks>
        [TestMethod]
        public unsafe void IsCpuAccessible_EveryProvidersAllocator_CpuDeviceRead_DeviceMemoryRefused()
        {
            var accepted = new (string Name, string Device, string Provider)[]
            {
                ("Cpu", "CPU", "CPU, ACL, DNNL, NNAPI, CoreML, RKNPU, SNPE, Azure, VitisAI"),
                ("CpuAligned4K", "CPU", "VSINPU"),
                ("XnnpackExecutionProvider", "CPU", "XNNPACK"),
            };
            var refused = new (string Name, string Device, string Provider)[]
            {
                ("Cuda", "GPU", "CUDA, TensorRT, NV TensorRT RTX"),
                ("CUDAMemPoolArena", "GPU", "CUDA (mempool arena)"),
                ("DML", "GPU", "DirectML"),
                ("WebGPU_Buf", "GPU", "WebGPU, JS (ORT 1.25+)"),
                ("WebGPU_Buffer", "GPU", "WebGPU, JS (before 1.25)"),
                ("WebNN_Ten", "GPU", "WebNN (ORT 1.25+)"),
                ("WebNN_Tensor", "GPU", "WebNN (before 1.25)"),
                ("MIGraphX", "GPU", "MIGraphX"),
                ("Cann", "NPU", "CANN"),
                ("OpenVINO_GPU", "GPU", "OpenVINO"),
                ("OpenVINO_NPU", "NPU", "OpenVINO"),
                ("SomeFutureGpuProvider", "GPU", "an unknown GPU provider"),
            };

            using NDArray backing = Arange(NPTypeCode.Single, 4);
            var address = (IntPtr)backing.Storage.InternalArray.Address;
            foreach (var (name, device, provider) in accepted)
            {
                using OrtMemoryInfo info = TryDescribe(name, device, "DEFAULT");
                if (info is null)
                    Assert.Inconclusive("the loaded ONNX Runtime has no OrtMemoryInfo(name, OrtMemoryInfoDeviceType, …) constructor (CreateMemoryInfo_V2, 1.23+)");
                NDArrayOnnxInterop.IsCpuAccessible(info).Should().BeTrue($"'{name}' is {provider}'s CPU-device memory");
                using OrtValue output = OrtValue.CreateTensorValueWithData(info, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, new long[] { 4 }, address, 16);
                using (NDArray view = output.AsNDArray())
                    view.GetSingle(2).Should().Be(2f, name);
                using (NDArray copy = output.ToNDArray())
                    copy.ToArray<float>().Should().Equal(new[] { 0f, 1f, 2f, 3f }, name);
            }
            foreach (var (name, device, provider) in refused)
            {
                using OrtMemoryInfo info = TryDescribe(name, device, "DEFAULT");
                NDArrayOnnxInterop.IsCpuAccessible(info).Should().BeFalse($"'{name}' is {provider}'s own {device} memory");
                using OrtValue output = OrtValue.CreateTensorValueWithData(info, Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float, new long[] { 4 }, address, 16);
                new Action(() => output.AsNDArray()).Should().Throw<InvalidOperationException>().WithMessage($"*'{name}'*cannot address*");
                new Action(() => output.ToNDArray()).Should().Throw<InvalidOperationException>().WithMessage($"*'{name}'*cannot address*");
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0);
            GC.KeepAlive(backing);
        }

        /// <summary>
        ///     Describes memory of a device type under <paramref name="name"/> through ONNX Runtime 1.23's name-and-device-type
        ///     constructor, found by reflection because the suite compiles against the 1.16.0 floor, which has neither the
        ///     constructor nor its two enums.
        /// </summary>
        /// <param name="name">Any allocator name; the constructor accepts names ORT does not know.</param>
        /// <param name="deviceType">An <c>OrtMemoryInfoDeviceType</c> member name: <c>CPU</c>, <c>GPU</c>, <c>FPGA</c>, <c>NPU</c>.</param>
        /// <param name="deviceMemoryType">An <c>OrtDeviceMemoryType</c> member name: <c>DEFAULT</c> or <c>HOST_ACCESSIBLE</c>.</param>
        /// <returns>The memory info, which the caller disposes, or <c>null</c> when the loaded ONNX Runtime lacks the constructor.</returns>
        /// <exception cref="System.Reflection.TargetInvocationException">ONNX Runtime refused the description.</exception>
        private static OrtMemoryInfo TryDescribe(string name, string deviceType, string deviceMemoryType)
        {
            System.Reflection.Assembly ort = typeof(OrtMemoryInfo).Assembly;
            Type deviceTypeEnum = ort.GetType("Microsoft.ML.OnnxRuntime.OrtMemoryInfoDeviceType");
            Type deviceMemoryTypeEnum = ort.GetType("Microsoft.ML.OnnxRuntime.OrtDeviceMemoryType");
            if (deviceTypeEnum is null || deviceMemoryTypeEnum is null)
                return null;
            var constructor = typeof(OrtMemoryInfo).GetConstructor(new[] { typeof(string), deviceTypeEnum, typeof(uint), typeof(int), deviceMemoryTypeEnum, typeof(ulong), typeof(OrtAllocatorType) });
            // The enum members are looked up by name, never assumed to be numbers.
            return (OrtMemoryInfo)constructor?.Invoke(new object[] { name, Enum.Parse(deviceTypeEnum, deviceType), 0u, 0, Enum.Parse(deviceMemoryTypeEnum, deviceMemoryType), 0UL, OrtAllocatorType.DeviceAllocator });
        }
    }
}
