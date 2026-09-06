using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>The proof of <c>examples/NumSharp.Interop.pythonnet.Examples/09-pytorch.cs</c>, section by section.</summary>
    [TestClass]
    public class Example09_PyTorch : ExampleTestBase
    {
        [TestInitialize]
        public void RequireTorch() => PyTorchTestGate.Require(Scope);

        [TestMethod]
        public void NumSharpToPyTorch_ToTorchIsAZeroCopyTensor()
        {
            using var scope = NDScope.Open();
            NDArray matrix = np.arange(12).astype(np.float64).reshape(3, 4);
            using (Gil())
            {
                py.t = matrix.ToTorch();
                py.v = matrix;
                bool sameAddress = py.Eval("t.data_ptr() == v.ctypes.data");
                sameAddress.Should().BeTrue("the tensor's data_ptr() IS NumSharp's buffer address");
                py.Exec("t[1, 2] = 91.5");
                matrix.GetDouble(1, 2).Should().Be(91.5);
                matrix[2, 1] = -7.25;
                PyFloat("t[2, 1].item()").Should().Be(-7.25);

                py.ts = matrix[":, ::2"].ToTorch();
                (long, long) shape = py.ts.shape;
                (long, long) stride = py.ts.stride();
                shape.Should().Be((3L, 2L), "torch.Size is a tuple subclass");
                stride.Should().Be((4L, 2L));
                py.Exec("ts.add_(1000)");
                matrix.GetDouble(0, 0).Should().Be(1000);
                matrix.GetDouble(0, 1).Should().Be(1);
                matrix.GetDouble(0, 2).Should().Be(1002);
                py.Exec("del t, v, ts");
            }
        }

        [TestMethod]
        public void EveryNumSharpDtype_MapsToATorchDtype()
        {
            using var scope = NDScope.Open();
            using (Gil())
                foreach (var (tc, torchDtype) in new[]
                         {
                             (NPTypeCode.Boolean, "torch.bool"), (NPTypeCode.Byte, "torch.uint8"), (NPTypeCode.SByte, "torch.int8"),
                             (NPTypeCode.Int16, "torch.int16"), (NPTypeCode.UInt16, "torch.uint16"), (NPTypeCode.Int32, "torch.int32"),
                             (NPTypeCode.UInt32, "torch.uint32"), (NPTypeCode.Int64, "torch.int64"), (NPTypeCode.UInt64, "torch.uint64"),
                             (NPTypeCode.Char, "torch.uint16"), (NPTypeCode.Half, "torch.float16"), (NPTypeCode.Single, "torch.float32"),
                             (NPTypeCode.Double, "torch.float64"), (NPTypeCode.Decimal, "torch.float64"), (NPTypeCode.Complex, "torch.complex128"),
                         })
                {
                    NDArray src = np.arange(3).astype(tc);
                    dynamic tensor = src.ToTorch();
                    string dtype = tensor.dtype.ToString();
                    dtype.Should().Be(torchDtype, tc.ToString());
                }
        }

        [TestMethod]
        public void UnsafeLayouts_NeedCopyTrue()
        {
            using var scope = NDScope.Open();
            NDArray matrix = np.arange(12).astype(np.float64).reshape(3, 4);
            NDArray reversed = matrix["::-1"];
            NDArray broadcast = np.broadcast_to(matrix[0], new Shape(2, 4));
            using (Gil())
            {
                py.v = matrix;
                Throws(() => reversed.ToTorch()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("copy:true");
                Throws(() => broadcast.ToTorch()).Should().BeOfType<InvalidOperationException>().Which.Message.Should().Contain("copy:true");
                py.rc = reversed.ToTorch(copy: true);
                py.bc = broadcast.ToTorch(copy: true);
                PyBool("rc[0].tolist() == v[2].tolist() and rc[2].tolist() == v[0].tolist()").Should().BeTrue("the logical values were materialized");
                PyStr("tuple(bc.shape)").Should().Be("(2, 4)");
                py.Exec("rc[0, 0] = -1.0; bc[0, 0] = -2.0");
                matrix.GetDouble(2, 0).Should().Be(8.0, "a copy is detached from NumSharp");
                matrix.GetDouble(0, 0).Should().Be(0.0);
                py.Exec("del v, rc, bc");
            }
        }

        [TestMethod]
        public void PyTorchToNumSharp_AZeroCopyViewOfCpuStorage()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("src = torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]");
                NDArray tv = py.src;
                tv.shape.Should().Equal(3, 2);
                tv.Shape.Strides.Should().Equal(new long[] { 4, 2 });
                py.Exec("src[1, 1] = 77.5");
                tv.GetDouble(1, 1).Should().Be(77.5);
                tv[2, 0] = -12.0;
                PyFloat("src[2, 0].item()").Should().Be(-12.0);
                py.Exec("del src; import gc; gc.collect()");
                tv.GetDouble(2, 0).Should().Be(-12.0, "the lease keeps the tensor storage alive");
            }
        }

        [TestMethod]
        public void Autograd_ANumSharpBackedLeaf_GradientBackAsNumSharp()
        {
            using var scope = NDScope.Open();
            NDArray x = np.array(new[] { -2.0, 1.5, 3.0 });
            using (Gil())
            {
                py.x = x.ToTorch();
                py.Exec("x.requires_grad_()\nloss = (x * x).sum()\nloss.backward()");
                double loss = py.loss.item();
                loss.Should().Be(15.25);
                NDArray dx = py.x.grad;
                dx.GetDouble(0).Should().Be(-4);
                dx.GetDouble(1).Should().Be(3);
                dx.GetDouble(2).Should().Be(6);
                py.Exec("del x, loss");
            }
        }

        [TestMethod]
        public void TensorsThatCannotBeShared_ForceDetachesTransfersAndCopies()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("g = torch.tensor([1.5, -2.0, 4.25], dtype=torch.float64, requires_grad=True)");
                using PyObject g = py.g;
                Throws(() => g.AsTorchNDArray()).Should().BeOfType<PythonException>().Which.Message.Should().Contain("requires grad");
                NDArray forced = g.ToTorchNDArray(force: true);
                forced.GetDouble(2).Should().Be(4.25);
                py.Exec("with torch.no_grad():\n    g[0] = 999.0");
                forced.GetDouble(0).Should().Be(1.5, "an independent snapshot");
                NDArray viaCodec = g.As<NDArray>();
                viaCodec.GetDouble(0).Should().Be(999.0, "Auto decode took the force-copy route");

                py.Exec("cj = torch.tensor([1+2j, 3-4j], dtype=torch.complex128).conj()");
                using PyObject cj = py.cj;
                Throws(() => cj.AsTorchNDArray()).Should().BeOfType<PythonException>().Which.Message.Should().Contain("resolve_conj");
                NDArray resolved = cj.ToTorchNDArray(force: true);
                ((Complex)resolved.GetValue(0)).Should().Be(new Complex(1, -2));

                bool cuda = py.Eval("torch.cuda.is_available()");
                if (cuda)
                {
                    py.Exec("dev = torch.arange(6, dtype=torch.float64, device='cuda')");
                    using PyObject dev = py.dev;
                    Throws(() => dev.AsTorchNDArray()).Should().BeOfType<PythonException>().Which.Message.Should().Contain("cpu()");
                    NDArray onCpu = dev.ToTorchNDArray(force: true);
                    onCpu.GetDouble(5).Should().Be(5.0);
                    py.Exec("del dev");
                }

                py.Exec("del g, cj");
            }
        }

        [TestMethod]
        public void Expand_AStrideZeroTensor_ImportsAsANonWriteableBroadcast()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                py.Exec("base = torch.arange(3, dtype=torch.float64)\nexp = base.expand(4, 3)");
                NDArray view = py.exp;
                view.Shape.Strides.Should().Equal(new long[] { 0, 1 });
                view.Shape.IsBroadcasted.Should().BeTrue();
                view.Shape.IsWriteable.Should().BeFalse();
                Exception blindWrite = Throws(() => np.copyto(view, np.ones(new Shape(4, 3))));
                blindWrite.Should().NotBeNull();
                blindWrite.Message.Should().Contain("read-only");
                py.Exec("del base, exp");
            }
        }

        [TestMethod]
        public void DtypesPyTorchCannotHandToNumpy_AreDirectedErrors_Complex64CopyWidens()
        {
            using var scope = NDScope.Open();
            using (Gil())
            {
                using PyObject c64 = py.Eval("torch.tensor([1+2j], dtype=torch.complex64)");
                Throws(() => c64.AsTorchNDArray()).Should().BeOfType<NotSupportedException>().Which.Message.Should().Contain("complex64");
                NDArray widened = c64.ToTorchNDArray();
                widened.typecode.Should().Be(NPTypeCode.Complex);
                ((Complex)widened.GetValue(0)).Should().Be(new Complex(1, 2));
                using PyObject bf16 = py.Eval("torch.arange(3, dtype=torch.bfloat16)");
                Throws(() => bf16.ToTorchNDArray(force: true)).Should().BeOfType<PythonException>().Which.Message.Should().Contain("BFloat16");
                using PyObject sparse = py.Eval("torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([3.0]), (2, 2))");
                Throws(() => sparse.AsTorchNDArray()).Should().BeOfType<PythonException>().Which.Message.Should().Contain("to_dense");
                using PyObject meta = py.Eval("torch.empty(2, device='meta')");
                Throws(() => meta.ToTorchNDArray(force: true)).Should().BeOfType<PythonException>().Which.Message.Should().Contain("meta");
                Throws(() => py.Exec("memoryview(torch.arange(3))")).Should().BeOfType<PythonException>("torch.Tensor is not a PEP 3118 exporter");
            }
        }

        [TestMethod]
        public void TheCodecAndTheOrdinaryVerbs_SeeTensorsThroughTheAdapter()
        {
            using var scope = NDScope.Open();
            NDArray source = np.arange(5).astype(np.float64);
            using (Gil())
            {
                py.implicit_t = py.torch.from_numpy(source);
                py.sv = source;
                bool samePointer = py.Eval("implicit_t.data_ptr() == sv.ctypes.data");
                samePointer.Should().BeTrue("torch.from_numpy(nd) received the codec's numpy view");
                py.Exec("implicit_t[1] = 88.0");
                source.GetDouble(1).Should().Be(88.0);

                using PyObject tensor = py.implicit_t;
                NDArray viaGeneric = tensor.As<NDArray>();
                NDArray viaType = (NDArray)tensor.AsManagedObject(typeof(NDArray));
                NDArray viaVerb = tensor.AsNDArray();
                NDArray viaDynamic = py.implicit_t;
                NDArray copy = tensor.ToNDArray();
                viaGeneric[2] = -12.0;
                viaType.GetDouble(2).Should().Be(-12.0);
                viaVerb.GetDouble(2).Should().Be(-12.0);
                viaDynamic.GetDouble(2).Should().Be(-12.0);
                PyFloat("implicit_t[2].item()").Should().Be(-12.0);
                copy[3] = -9.0;
                PyFloat("implicit_t[3].item()").Should().Be(3.0, "the copy is detached");
                py.Exec("del implicit_t, sv");
            }
        }

        [TestMethod]
        public void SharedStorage_CannotBeResizedOnEitherSide()
        {
            using var scope = NDScope.Open();
            NDArray pinned = np.arange(4).astype(np.int64);
            using (Gil())
            {
                py.pt = pinned.ToTorch();
                Exception numsharpSide = Throws(() => pinned.resize(new Shape(8)));
                numsharpSide.Should().NotBeNull();
                numsharpSide.Message.Should().Contain("references or is referenced");
                Throws(() => py.Exec("pt.resize_(8)")).Should().BeOfType<PythonException>().Which.Message.Should().Contain("resizable");
                py.Exec("del pt");
            }
        }
    }
}
