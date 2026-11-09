using System;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Every CPU-side execution provider the loaded ONNX Runtime carries, end to end through the bridge
    ///     (<see cref="ProviderEndToEnd"/>): a session with the provider appended runs <c>matmul_const_f32</c> and its output
    ///     is imported zero-copy, copied, and returned by the bridge's own Run. Desktop ONNX Runtime packages carry only the
    ///     CPU provider (XNNPACK and NNAPI are in the Android and iOS packages), so the mobile providers run in
    ///     <c>NumSharp.Tests.Interop.OnnxRuntime.Android</c>, which calls the same code on a device.
    /// </summary>
    [TestClass]
    public class ProviderEndToEndTests : OnnxTestBase
    {
        /// <summary>
        ///     The CPU provider: its MatMul output lives in <c>Cpu</c> memory, and every import route gives X @ W exactly.
        /// </summary>
        [TestMethod]
        public void CpuProvider_ConstantMatMul_ImportsEndToEnd()
        {
            string report = ProviderEndToEnd.Run(File.ReadAllBytes(Model("matmul_const_f32")), "CPUExecutionProvider");
            report.Should().Contain("'Cpu' memory");
        }

        /// <summary>
        ///     Every provider of <see cref="ProviderEndToEnd.Known"/> this ONNX Runtime reports as available runs end to end;
        ///     the ones it lacks are named in the output (a desktop build lists only the CPU provider).
        /// </summary>
        [TestMethod]
        public void EveryAvailableCpuSideProvider_ImportsEndToEnd()
        {
            string[] available = ProviderEndToEnd.Available();
            available.Should().Contain("CPUExecutionProvider", "every ONNX Runtime build carries the CPU provider");
            byte[] model = File.ReadAllBytes(Model("matmul_const_f32"));
            foreach (string provider in available)
                Console.WriteLine(ProviderEndToEnd.Run(model, provider));
            var missing = ProviderEndToEnd.Known.Select(k => k.Provider).Except(available).ToArray();
            if (missing.Length > 0)
                Console.WriteLine($"not in this ONNX Runtime build (run on Android: NumSharp.Tests.Interop.OnnxRuntime.Android): {string.Join(", ", missing)}");
        }

        /// <summary>
        ///     Every GPU provider of <see cref="ProviderEndToEnd.DeviceMemoryProviders"/> this ONNX Runtime carries has its
        ///     device-bound output refused by both imports. A desktop CPU package carries none (WebGPU ships in the mobile
        ///     packages), so here the test only reports that; the Android host runs it on a device.
        /// </summary>
        [TestMethod]
        public void EveryAvailableGpuProvider_DeviceBoundOutput_IsRefused()
        {
            var available = OrtEnv.Instance().GetAvailableProviders();
            byte[] model = File.ReadAllBytes(Model("matmul_const_f32"));
            int ran = 0;
            foreach (var (provider, _, _, _) in ProviderEndToEnd.DeviceMemoryProviders)
            {
                if (!available.Contains(provider))
                    continue;
                Console.WriteLine(ProviderEndToEnd.RunRefused(model, provider));
                ran++;
            }
            if (ran == 0)
                Console.WriteLine("no GPU provider in this ONNX Runtime build; the refusal of device memory is covered here by " +
                                  "IsCpuAccessible_EveryProvidersAllocator_CpuDeviceRead_DeviceMemoryRefused and on a device by the Android host");
        }
    }
}
