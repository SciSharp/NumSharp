using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Live proof against the latest stable PyTorch pinned when this bridge was added: 2.13.0.
    ///     These tests are optional on hosts without that exact PyTorch feature line, but strict when
    ///     it is present. They exercise PyTorch's real storage, striding and autograd behavior rather
    ///     than a mock or a value-only serialization round-trip.
    /// </summary>
    [TestClass]
    public class PyTorchInteropTests : InteropTestBase
    {
        private const string ValidatedTorchLine = "2.13.";

        [TestMethod]
        public void Runtime_IsLatestStablePyTorch213()
        {
            RequireValidatedPyTorch();
            PyStr("torch.__version__").Should().StartWith(ValidatedTorchLine);
        }

        [TestMethod]
        public unsafe void NumSharpToTorch_Contiguous_IsZeroCopyAndMutatesBothWays()
        {
            RequireValidatedPyTorch();

            using NDArray source = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
            long expectedAddress = (long)source.Storage.Address + source.Shape.Offset * source.dtypesize;

            using (Gil())
            using (PyObject tensor = source.ToTorch())
            {
                Scope.Set("shared_tensor", tensor);
                PyLong("shared_tensor.data_ptr()").Should().Be(expectedAddress,
                    "torch.from_numpy must point at NumSharp's actual unmanaged buffer");

                Scope.Exec("shared_tensor[1, 2] = 91.5");
                ReadAt<double>(source, 1, 2).Should().Be(91.5,
                    "a PyTorch write must land in the NumSharp array");

                WriteAt(source, -7.25, 2, 1);
                using PyObject value = Scope.Eval("shared_tensor[2, 1].item()");
                value.As<double>().Should().Be(-7.25,
                    "a NumSharp write must be visible through the tensor");
            }
        }

        [TestMethod]
        public void NumSharpToTorch_PositiveStridedView_PreservesShapeStrideAndStorage()
        {
            RequireValidatedPyTorch();

            using NDArray owner = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
            using NDArray strided = owner[":, ::2"];

            using (Gil())
            using (PyObject tensor = strided.ToTorch())
            {
                Scope.Set("strided_tensor", tensor);
                PyStr("tuple(strided_tensor.shape)").Should().Be("(3, 2)");
                PyStr("strided_tensor.stride()").Should().Be("(4, 2)");

                Scope.Exec("strided_tensor.add_(1000)");
                ReadAt<double>(owner, 0, 0).Should().Be(1000);
                ReadAt<double>(owner, 0, 1).Should().Be(1,
                    "the skipped column must not be touched by a strided tensor operation");
                ReadAt<double>(owner, 0, 2).Should().Be(1002);
            }
        }

        [TestMethod]
        public unsafe void NumSharpToTorch_All15Dtypes_MapOnPyTorch213()
        {
            RequireValidatedPyTorch();

            (NPTypeCode type, string torchDtype)[] cases =
            {
                (NPTypeCode.Boolean, "torch.bool"),
                (NPTypeCode.Byte, "torch.uint8"),
                (NPTypeCode.SByte, "torch.int8"),
                (NPTypeCode.Int16, "torch.int16"),
                (NPTypeCode.UInt16, "torch.uint16"),
                (NPTypeCode.Int32, "torch.int32"),
                (NPTypeCode.UInt32, "torch.uint32"),
                (NPTypeCode.Int64, "torch.int64"),
                (NPTypeCode.UInt64, "torch.uint64"),
                (NPTypeCode.Char, "torch.uint16"),
                (NPTypeCode.Half, "torch.float16"),
                (NPTypeCode.Single, "torch.float32"),
                (NPTypeCode.Double, "torch.float64"),
                (NPTypeCode.Decimal, "torch.float64"),
                (NPTypeCode.Complex, "torch.complex128"),
            };

            using (Gil())
            {
                foreach (var (type, expected) in cases)
                {
                    using NDArray source = np.arange(3).astype(type);
                    using PyObject tensor = source.ToTorch(requireGIL: false);
                    using PyObject dtype = tensor.GetAttr("dtype");
                    dtype.ToString().Should().Be(expected, type.ToString());

                    if (type == NPTypeCode.Decimal)
                    {
                        using PyObject dataPointer = tensor.InvokeMethod("data_ptr");
                        dataPointer.As<long>().Should().NotBe((long)source.Storage.Address,
                            "Decimal has no NumPy/PyTorch dtype and must convert to independent float64 storage");
                    }
                }
            }
        }

        [TestMethod]
        public void NumSharpToTorch_UnsafeLayoutsRequireExplicitCopy()
        {
            RequireValidatedPyTorch();

            using NDArray owner = np.arange(6).astype(NPTypeCode.Double);
            using NDArray reversed = owner["::-1"];
            using NDArray broadcast = np.broadcast_to(owner["0:3"], new Shape(2, 3));

            new Action(() =>
                {
                    using (Gil())
                    using (PyObject _ = reversed.ToTorch()) { }
                })
                .Should().Throw<NotSupportedException>()
                .WithMessage("*negative strides*copy:true*");

            new Action(() =>
                {
                    using (Gil())
                    using (PyObject _ = broadcast.ToTorch()) { }
                })
                .Should().Throw<InvalidOperationException>()
                .WithMessage("*non-writeable*undefined behavior*copy:true*");

            using (Gil())
            using (PyObject reversedCopy = reversed.ToTorch(copy: true))
            using (PyObject broadcastCopy = broadcast.ToTorch(copy: true))
            {
                Scope.Set("reversed_copy", reversedCopy);
                Scope.Set("broadcast_copy", broadcastCopy);
                PyStr("reversed_copy.tolist()").Should().Be("[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]");
                PyStr("tuple(broadcast_copy.shape)").Should().Be("(2, 3)");

                Scope.Exec("reversed_copy[0] = -999; broadcast_copy[0, 0] = -888");
                ReadAt<double>(owner, 5).Should().Be(5,
                    "copy:true must detach PyTorch writes from the reversed source");
                ReadAt<double>(owner, 0).Should().Be(0,
                    "copy:true must detach PyTorch writes from the broadcast source");
            }
        }

        [TestMethod]
        public void TorchToNumSharp_PositiveStridedCpuTensor_IsZeroCopyAndSurvivesTensorWrapper()
        {
            RequireValidatedPyTorch();
            PyExec("torch_source = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]");

            NDArray view;
            using (Gil())
            {
                using PyObject tensor = Scope.Get("torch_source");
                view = tensor.AsTorchNDArray();
            }

            using (view)
            {
                view.shape.Should().Equal(3, 2);
                view.Shape.Strides.Should().Equal(4L, 2L);

                PyExec("torch_source[1, 1] = 77.5");
                ReadAt<double>(view, 1, 1).Should().Be(77.5,
                    "a tensor write must reach the NumSharp view");

                WriteAt(view, -12.0, 2, 0);
                PyFloat("torch_source[2, 0].item()").Should().Be(-12.0,
                    "a NumSharp write must reach the tensor");

                PyExec("del torch_source; gc.collect()");
                ReadAt<double>(view, 2, 0).Should().Be(-12.0,
                    "the NumSharp lease must keep tensor storage alive after Python drops its name");
            }
        }

        [TestMethod]
        public void TorchToNumSharp_CopyIsDetached_AndForceHandlesAutogradTensor()
        {
            RequireValidatedPyTorch();
            PyExec("grad_tensor = torch.tensor([1.5, -2.0, 4.25], dtype=torch.float64, requires_grad=True)");

            using (Gil())
            using (PyObject tensor = Scope.Get("grad_tensor"))
            {
                new Action(() => { using var _ = tensor.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>()
                    .WithMessage("*requires grad*detach*");

                using NDArray copy = tensor.ToTorchNDArray(force: true, requireGIL: false);
                using NDArray expected = np.array(new[] { 1.5, -2.0, 4.25 });
                ByteContract.NsBytes(copy).Should().Equal(ByteContract.NsBytes(expected),
                    "force follows Tensor.numpy(force=True) before making an independent NumSharp copy");

                Scope.Exec("with torch.no_grad():\n    grad_tensor[0] = 999.0");
                ReadAt<double>(copy, 0).Should().Be(1.5,
                    "ToTorchNDArray is an independent snapshot");
            }
        }

        [TestMethod]
        public void NumSharpToTorch_AutogradCompute_GradientReturnsToNumSharp()
        {
            RequireValidatedPyTorch();
            using NDArray source = np.array(new[] { -2.0, 1.5, 3.0 });

            using (Gil())
            using (PyObject tensor = source.ToTorch())
            {
                Scope.Set("x", tensor);
                Scope.Exec("x.requires_grad_()\nloss = (x * x).sum()\nloss.backward()");
                PyFloat("loss.item()").Should().Be(15.25);

                using PyObject gradient = Scope.Eval("x.grad");
                using NDArray back = gradient.ToTorchNDArray();
                using NDArray expected = np.array(new[] { -4.0, 3.0, 6.0 });
                ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(expected),
                    "PyTorch autograd's 2*x result must return exactly to NumSharp");
            }
        }

        [TestMethod]
        public void TorchTensor_IsNotABufferExporter_TheNumpyAdapterIsIntentional()
        {
            RequireValidatedPyTorch();
            PyExec("plain_tensor = torch.arange(3, dtype=torch.float64)");

            PyBool("hasattr(plain_tensor, '__array__')").Should().BeTrue(
                "Tensor.numpy/__array__ is the supported CPU adapter");
            using (Gil())
            {
                new Action(() => { using var _ = Scope.Eval("memoryview(plain_tensor)"); })
                    .Should().Throw<PythonException>()
                    .WithMessage("*bytes-like object*Tensor*");
            }
        }

        private void RequireValidatedPyTorch()
        {
            SkipUnless("torch");
            using (Gil())
            {
                Scope.Exec("import torch");
                using PyObject versionObject = Scope.Eval("str(torch.__version__)");
                string version = versionObject.As<string>();
                if (!version.StartsWith(ValidatedTorchLine, StringComparison.Ordinal))
                    Assert.Inconclusive(
                        $"PyTorch {version} is installed; this live compatibility gate is pinned to the " +
                        "latest stable 2.13.x line validated by this change.");
            }
        }
    }
}
