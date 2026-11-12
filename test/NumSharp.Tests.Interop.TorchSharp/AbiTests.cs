using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The native seam: the bridge must recognize the <c>THSTensor_new</c> form of the TorchSharp it runs against, or every
    ///     export would be refused. 0.100.x declares six arguments, 0.101.0 and later nine.
    /// </summary>
    [TestClass]
    public class AbiTests : TorchTestBase
    {
        /// <summary>The detected form matches the loaded TorchSharp's version line.</summary>
        [TestMethod]
        public void Abi_IsDetectedForTheLoadedTorchSharp()
        {
            Version loaded = typeof(torch).Assembly.GetName().Version;
            LibTorchSharpNative.NewTensorAbi abi = LibTorchSharpNative.DetectAbi();
            Console.WriteLine($"TorchSharp {loaded}, libtorch {torch.__version__}: THSTensor_new {abi}");
            abi.Should().Be(loaded.Minor <= 100 ? LibTorchSharpNative.NewTensorAbi.Args6 : LibTorchSharpNative.NewTensorAbi.Args9,
                "TorchSharp 0.100.x declares the 6-argument THSTensor_new, every later release the 9-argument one");
            LibTorchSharpNative.Abi.Should().Be(abi, "the cached answer is the detected one");
        }

        /// <summary>
        ///     The package is compiled against the floor, TorchSharp 0.100.0, whatever TorchSharp this leg loads: a run against a
        ///     newer TorchSharp is then the gate that every member the package binds kept its signature.
        /// </summary>
        [TestMethod]
        public void Package_IsCompiledAgainstTheFloor()
        {
            AssemblyNameReference();
            Version referenced = Array.Find(typeof(NDArrayTorchInterop).Assembly.GetReferencedAssemblies(), a => a.Name == "TorchSharp").Version;
            referenced.Should().Be(new Version(0, 100, 0, 0), "the package's PackageReference is the floor, 0.100.0");
            Console.WriteLine($"package compiled against TorchSharp {referenced}, running on {typeof(torch).Assembly.GetName().Version}");
        }

        /// <summary>Touches the package assembly so its references are loaded (a no-op guard for the reflection above).</summary>
        private static void AssemblyNameReference() => _ = NDArrayTorchInterop.LiveExports;

        /// <summary>A from_blob export through the detected form shares the buffer and releases it when the tensor dies.</summary>
        [TestMethod]
        public void DetectedForm_SharesTheBufferAndReleasesIt()
        {
            int before = NDArrayTorchInterop.LiveExports;
            using NDArray nd = np.arange(6).astype(np.float32);
            using (Tensor t = nd.AsTensor(TorchCodecMode.View))
            {
                NDArrayTorchInterop.LiveExports.Should().Be(before + 1);
                t.requires_grad.Should().BeFalse("the bridge never asks libtorch to track gradients");
                t.dtype.Should().Be(ScalarType.Float32);
                nd.MayShareMemory(t).Should().BeTrue();
            }
            NDArrayTorchInterop.LiveExports.Should().Be(before, "disposing the only tensor frees the storage and runs the deleter");
        }
    }
}
