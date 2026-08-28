using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Rare-scenario matrix for the PyTorch bridge. The core test class proves the public contract;
    ///     this class attacks its boundaries: degenerate shapes, storage offsets, every boundary value,
    ///     reverse dtype mapping, special tensor state, unsupported layouts/dtypes, ownership, GIL and
    ///     concurrent churn. Every Python-side answer is produced by live PyTorch 2.13.x.
    /// </summary>
    [TestClass]
    public class PyTorchInteropEdgeCaseTests : InteropTestBase
    {
        [TestMethod]
        public unsafe void NumSharpShapes_ScalarEmptyOffsetFortranAndSingletonAxes()
        {
            RequirePyTorch();

            using NDArray scalar = np.array(3.5);
            using NDArray empty = np.empty(new Shape(0, 3), NPTypeCode.Double);
            using NDArray owner = np.arange(30).astype(NPTypeCode.Double).reshape(5, 6);
            using NDArray offset = owner["1:4, 1::2"];
            using NDArray fortran = np.asfortranarray(np.arange(12).astype(NPTypeCode.Double).reshape(3, 4));
            using NDArray singleton = np.arange(3).astype(NPTypeCode.Double).reshape(1, 1, 3);

            using (Gil())
            using (PyObject ts = scalar.ToTorch())
            using (PyObject te = empty.ToTorch())
            using (PyObject to = offset.ToTorch())
            using (PyObject tf = fortran.ToTorch())
            using (PyObject tn = singleton.ToTorch())
            {
                Scope.Set("edge_ts", ts);
                Scope.Set("edge_te", te);
                Scope.Set("edge_to", to);
                Scope.Set("edge_tf", tf);
                Scope.Set("edge_tn", tn);

                PyStr("tuple(edge_ts.shape)").Should().Be("()");
                PyLong("edge_ts.data_ptr()").Should().Be((long)scalar.Storage.Address);
                Scope.Exec("edge_ts.fill_(8.25)");
                ReadAt<double>(scalar).Should().Be(8.25);

                PyStr("tuple(edge_te.shape)").Should().Be("(0, 3)");
                PyLong("edge_te.numel()").Should().Be(0);

                PyStr("tuple(edge_to.shape)").Should().Be("(3, 3)");
                PyStr("edge_to.stride()").Should().Be("(6, 2)");
                PyLong("edge_to.data_ptr()").Should().Be(
                    (long)owner.Storage.Address + offset.Shape.Offset * offset.dtypesize);

                PyStr("edge_tf.stride()").Should().Be("(1, 3)");
                PyBool("edge_tf.is_contiguous()").Should().BeFalse();
                Scope.Exec("edge_tf[2, 3] = -9.0");
                ReadAt<double>(fortran, 2, 3).Should().Be(-9.0);

                PyStr("tuple(edge_tn.shape)").Should().Be("(1, 1, 3)");
                PyStr("edge_tn.stride()").Should().Be("(3, 3, 1)");
            }
        }

        [TestMethod]
        public unsafe void TorchShapes_ScalarEmptyTransposeOffsetAndExpanded()
        {
            RequirePyTorch();
            int imports = NDArrayPythonInterop.LiveImports;
            PyExec(
                "edge_scalar = torch.tensor(3.5, dtype=torch.float64)\n" +
                "edge_empty = torch.empty((0, 3), dtype=torch.float64)\n" +
                "edge_transpose = torch.arange(12, dtype=torch.float64).reshape(3, 4).t()\n" +
                "edge_offset = torch.arange(30, dtype=torch.float64).reshape(5, 6)[1:4, 1::2]\n" +
                "edge_expanded_base = torch.arange(3, dtype=torch.float64)\n" +
                "edge_expanded = edge_expanded_base.expand(2, 3)");

            NDArray scalar;
            NDArray empty;
            NDArray transpose;
            NDArray offset;
            NDArray expanded;
            using (Gil())
            {
                using PyObject ps = Scope.Get("edge_scalar");
                using PyObject pe = Scope.Get("edge_empty");
                using PyObject pt = Scope.Get("edge_transpose");
                using PyObject po = Scope.Get("edge_offset");
                using PyObject px = Scope.Get("edge_expanded");
                scalar = ps.AsTorchNDArray();
                empty = pe.AsTorchNDArray();
                transpose = pt.AsTorchNDArray();
                offset = po.AsTorchNDArray();
                expanded = px.AsTorchNDArray();
            }

            using (scalar)
            using (empty)
            using (transpose)
            using (offset)
            using (expanded)
            {
                scalar.ndim.Should().Be(0);
                WriteAt(scalar, 6.75);
                PyFloat("edge_scalar.item()").Should().Be(6.75);

                empty.shape.Should().Equal(0, 3);
                empty.size.Should().Be(0);
                NDArrayPythonInterop.LiveImports.Should().Be(imports + 4,
                    "an empty tensor has no bytes to lease; the other four views do");

                transpose.shape.Should().Equal(4, 3);
                transpose.Shape.Strides.Should().Equal(1L, 4L);
                ((long)transpose.Storage.Address + transpose.Shape.Offset * transpose.dtypesize)
                    .Should().Be(PyLong("edge_transpose.data_ptr()"));

                offset.shape.Should().Equal(3, 3);
                offset.Shape.Strides.Should().Equal(6L, 2L);
                ReadAt<double>(offset, 0, 0).Should().Be(7);
                ReadAt<double>(offset, 2, 2).Should().Be(23);

                expanded.Shape.IsBroadcasted.Should().BeTrue();
                expanded.Shape.IsWriteable.Should().BeFalse(
                    "PyTorch exposes expand() as writable stride-0 NumPy memory, but NumSharp must not " +
                    "let two logical coordinates blindly overwrite one element");
                using NDArray zeros = np.zeros(new Shape(2, 3));
                new Action(() => np.copyto(expanded, zeros))
                    .Should().Throw<NumSharpException>().WithMessage("*read-only*");
                PyStr("edge_expanded_base.tolist()").Should().Be("[0.0, 1.0, 2.0]");
            }
        }

        [TestMethod]
        public void All15Dtypes_BoundaryValuesRoundTripThroughTorch()
        {
            RequirePyTorch();

            var cases = new (string name, NDArray source, NPTypeCode returnedType)[]
            {
                ("bool", np.array(new[] { false, true }), NPTypeCode.Boolean),
                ("u8", np.array(new byte[] { 0, byte.MaxValue }), NPTypeCode.Byte),
                ("i8", np.array(new[] { sbyte.MinValue, (sbyte)-1, sbyte.MaxValue }), NPTypeCode.SByte),
                ("i16", np.array(new[] { short.MinValue, (short)-1, short.MaxValue }), NPTypeCode.Int16),
                ("u16", np.array(new ushort[] { 0, ushort.MaxValue }), NPTypeCode.UInt16),
                ("i32", np.array(new[] { int.MinValue, -1, int.MaxValue }), NPTypeCode.Int32),
                ("u32", np.array(new uint[] { 0, uint.MaxValue }), NPTypeCode.UInt32),
                ("i64", np.array(new[] { long.MinValue, -1L, long.MaxValue }), NPTypeCode.Int64),
                ("u64", np.array(new ulong[] { 0, ulong.MaxValue }), NPTypeCode.UInt64),
                ("char", np.array(new[] { '\0', '\ud800', '\uffff' }), NPTypeCode.UInt16),
                ("f16", np.array(new[]
                {
                    BitConverter.UInt16BitsToHalf(0x8000),
                    BitConverter.UInt16BitsToHalf(0x7c00),
                    BitConverter.UInt16BitsToHalf(0xfe55),
                }), NPTypeCode.Half),
                ("f32", np.array(new[]
                {
                    -0.0f,
                    float.PositiveInfinity,
                    BitConverter.Int32BitsToSingle(unchecked((int)0xffc12345)),
                }), NPTypeCode.Single),
                ("f64", np.array(new[]
                {
                    -0.0,
                    double.NegativeInfinity,
                    BitConverter.Int64BitsToDouble(unchecked((long)0xfff8123456789abc)),
                }), NPTypeCode.Double),
                ("decimal", np.array(new[] { decimal.MinValue, 1.234567890123456789m, decimal.MaxValue }), NPTypeCode.Double),
                ("complex", np.array(new[]
                {
                    new Complex(-0.0, double.PositiveInfinity),
                    new Complex(BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000042)), -0.0),
                }), NPTypeCode.Complex),
            };

            try
            {
                using (Gil())
                {
                    foreach (var (name, source, returnedType) in cases)
                    {
                        using PyObject tensor = source.ToTorch(requireGIL: false);
                        using NDArray back = tensor.ToTorchNDArray(requireGIL: false);
                        back.typecode.Should().Be(returnedType, name);

                        if (source.typecode == NPTypeCode.Decimal)
                        {
                            using NDArray expected = source.astype(NPTypeCode.Double);
                            ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(expected), name);
                        }
                        else
                        {
                            ByteContract.NsBytes(back).Should().Equal(ByteContract.NsBytes(source), name);
                        }
                    }
                }
            }
            finally
            {
                foreach (var c in cases)
                    c.source.Dispose();
            }
        }

        [TestMethod]
        public unsafe void TorchDtypes_AllNativelyViewableTypesShare_Complex64CopiesAndWidens()
        {
            RequirePyTorch();
            (string torchDtype, NPTypeCode numSharpType)[] viewable =
            {
                ("torch.bool", NPTypeCode.Boolean),
                ("torch.uint8", NPTypeCode.Byte),
                ("torch.int8", NPTypeCode.SByte),
                ("torch.int16", NPTypeCode.Int16),
                ("torch.uint16", NPTypeCode.UInt16),
                ("torch.int32", NPTypeCode.Int32),
                ("torch.uint32", NPTypeCode.UInt32),
                ("torch.int64", NPTypeCode.Int64),
                ("torch.uint64", NPTypeCode.UInt64),
                ("torch.float16", NPTypeCode.Half),
                ("torch.float32", NPTypeCode.Single),
                ("torch.float64", NPTypeCode.Double),
                ("torch.complex128", NPTypeCode.Complex),
            };

            using (Gil())
            {
                foreach (var (torchDtype, numSharpType) in viewable)
                {
                    using PyObject tensor = Scope.Eval($"torch.zeros(3, dtype={torchDtype})");
                    using NDArray view = tensor.AsTorchNDArray(requireGIL: false);
                    view.typecode.Should().Be(numSharpType, torchDtype);
                    long first = (long)view.Storage.Address + view.Shape.Offset * view.dtypesize;
                    using PyObject ptr = tensor.InvokeMethod("data_ptr");
                    first.Should().Be(ptr.As<long>(), $"{torchDtype} must be a real view");
                }

                using PyObject complex64 = Scope.Eval("torch.tensor([1+2j, -3+4j], dtype=torch.complex64)");
                new Action(() => { using var _ = complex64.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<NotSupportedException>().WithMessage("*complex64*copy*");

                using NDArray widened = complex64.ToTorchNDArray(requireGIL: false);
                widened.typecode.Should().Be(NPTypeCode.Complex);
                ReadAt<Complex>(widened, 0).Should().Be(new Complex(1, 2));
                ReadAt<Complex>(widened, 1).Should().Be(new Complex(-3, 4));
            }
        }

        [TestMethod]
        public void ConjugateAndNegativeBits_RequireForce_ThenResolveExactly()
        {
            RequirePyTorch();
            PyExec(
                "edge_conj = torch.tensor([1+2j, 3-4j], dtype=torch.complex128).conj()\n" +
                "edge_neg = torch._neg_view(torch.tensor([1.0, -2.0], dtype=torch.float64))");

            using (Gil())
            using (PyObject conjugated = Scope.Get("edge_conj"))
            using (PyObject negative = Scope.Get("edge_neg"))
            {
                new Action(() => { using var _ = conjugated.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*conjugate bit*resolve_conj*");
                new Action(() => { using var _ = negative.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*negative bit*resolve_neg*");

                using NDArray conjCopy = conjugated.ToTorchNDArray(force: true, requireGIL: false);
                using NDArray negCopy = negative.ToTorchNDArray(force: true, requireGIL: false);
                ReadAt<Complex>(conjCopy, 0).Should().Be(new Complex(1, -2));
                ReadAt<Complex>(conjCopy, 1).Should().Be(new Complex(3, 4));
                ReadAt<double>(negCopy, 0).Should().Be(-1);
                ReadAt<double>(negCopy, 1).Should().Be(2);

                Scope.Exec("edge_conj = torch.zeros(2, dtype=torch.complex128); edge_neg = torch.zeros(2)");
                ReadAt<Complex>(conjCopy, 0).Should().Be(new Complex(1, -2), "force returns a detached copy");
                ReadAt<double>(negCopy, 0).Should().Be(-1, "force returns a detached copy");
            }
        }

        [TestMethod]
        public void UnsupportedTorchDtypesLayoutsAndDevices_FailWithDirectedPyTorchErrors()
        {
            RequirePyTorch();
            using (Gil())
            {
                AssertCannotCross("torch.arange(3, dtype=torch.bfloat16)", "*unsupported ScalarType BFloat16*");
                AssertCannotCross("torch.zeros(2, dtype=torch.float8_e4m3fn)", "*unsupported ScalarType Float8_e4m3fn*");
                AssertCannotCross(
                    "torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([3.0]), (2, 2))",
                    "*Sparse layout*tensor to numpy*to_dense*");
                AssertCannotCross(
                    "torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), 0.1, 10, torch.quint8)",
                    "*unsupported ScalarType QUInt8*");

                using PyObject meta = Scope.Eval("torch.empty(2, device='meta')");
                new Action(() => { using var _ = meta.ToTorchNDArray(force: true, requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*Cannot copy out of meta tensor*no data*");
            }
        }

        [TestMethod]
        public void ExpandedTensor_IsReadOnlyInNumSharp_AndRequiresCopyBackToTorch()
        {
            RequirePyTorch();
            PyExec("edge_expand_owner = torch.arange(3, dtype=torch.float64)\nedge_expand = edge_expand_owner.expand(4, 3)");

            using (Gil())
            using (PyObject tensor = Scope.Get("edge_expand"))
            using (NDArray view = tensor.AsTorchNDArray(requireGIL: false))
            {
                view.Shape.Strides.Should().Equal(0L, 1L);
                view.Shape.IsBroadcasted.Should().BeTrue();
                view.Shape.IsWriteable.Should().BeFalse();

                using NDArray ones = np.ones(new Shape(4, 3));
                new Action(() => np.copyto(view, ones))
                    .Should().Throw<NumSharpException>().WithMessage("*read-only*");
                new Action(() => { using var _ = view.ToTorch(requireGIL: false); })
                    .Should().Throw<InvalidOperationException>().WithMessage("*non-writeable*copy:true*");

                using PyObject copy = view.ToTorch(copy: true, requireGIL: false);
                Scope.Set("edge_expand_copy", copy);
                Scope.Exec("edge_expand_copy[0, 0] = 999");
                PyStr("edge_expand_owner.tolist()").Should().Be("[0.0, 1.0, 2.0]");
            }
        }

        [TestMethod]
        public void TorchDerivedViews_KeepNumSharpExportAliveUntilTheLastTensorDies()
        {
            RequirePyTorch();
            int exports = NDArrayPythonInterop.LiveExports;

            using (Gil())
            {
                NDArray source = np.arange(10).astype(NPTypeCode.Double);
                using PyObject tensor = source.ToTorch(requireGIL: false);
                Scope.Set("edge_torch_owner", tensor);
                Scope.Exec("edge_torch_child = edge_torch_owner[3:8]\ndel edge_torch_owner");
                source.Dispose();
            }

            Pump();
            NDArrayPythonInterop.LiveExports.Should().Be(exports + 1);
            PyFloat("edge_torch_child.sum().item()").Should().Be(25);
            PyExec("del edge_torch_child");
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports).Should().BeTrue(
                "the tensor's retained NumPy owner must release the export after the last derived tensor dies");
        }

        [TestMethod]
        public void NumSharpDerivedViews_KeepTorchStorageAliveAfterTheOriginalTensorWrapperDies()
        {
            RequirePyTorch();
            int imports = NDArrayPythonInterop.LiveImports;
            PyExec(
                "import weakref\n" +
                "edge_torch_source = torch.arange(8, dtype=torch.float64)\n" +
                "edge_torch_weak = weakref.ref(edge_torch_source)");

            NDArray derived;
            using (Gil())
            {
                using PyObject tensor = Scope.Get("edge_torch_source");
                using NDArray original = tensor.AsTorchNDArray(requireGIL: false);
                derived = original["2:6"];
                Scope.Exec("del edge_torch_source");
            }

            using (derived)
            {
                Pump();
                NDArrayPythonInterop.LiveImports.Should().Be(imports + 1);
                PyBool("edge_torch_weak() is None").Should().BeTrue(
                    "Tensor.numpy() roots storage through a separate NumPy-base Tensor object, not the " +
                    "original Python Tensor wrapper");
                ReadAt<double>(derived, 0).Should().Be(2);
                ReadAt<double>(derived, 3).Should().Be(5);
                WriteAt(derived, -44.0, 1);

                using (Gil())
                using (PyObject rehydrated = derived.ToTorch())
                {
                    Scope.Set("edge_rehydrated", rehydrated);
                    PyFloat("edge_rehydrated[1].item()").Should().Be(-44.0,
                        "the leased Torch storage must remain valid after its original wrapper dies");
                }
                PyExec("del edge_rehydrated");
            }

            WaitFor(() => NDArrayPythonInterop.LiveImports == imports).Should().BeTrue();
        }

        [TestMethod]
        public void SharedStorage_CannotResizeOnEitherSide_CopyStorageCan()
        {
            RequirePyTorch();
            int exports = NDArrayPythonInterop.LiveExports;
            using NDArray source = np.arange(4).astype(NPTypeCode.Int64);

            using (Gil())
            using (PyObject tensor = source.ToTorch())
            {
                new Action(() => source.resize(new Shape(8)))
                    .Should().Throw<Exception>().Which.GetType().Name.Should().Be("IncorrectShapeException");
                using PyObject eight = 8L.ToPython();
                new Action(() => { using var _ = tensor.InvokeMethod("resize_", eight); })
                    .Should().Throw<PythonException>().WithMessage("*storage*not resizable*");
            }

            WaitFor(() => NDArrayPythonInterop.LiveExports == exports).Should().BeTrue();
            source.resize(new Shape(8));
            source.size.Should().Be(8);

            PyExec("edge_resize_tensor = torch.arange(4, dtype=torch.int64)");
            using (Gil())
            using (PyObject tensor = Scope.Get("edge_resize_tensor"))
            using (NDArray view = tensor.AsTorchNDArray(requireGIL: false))
            {
                new Action(() => view.resize(new Shape(8)))
                    .Should().Throw<Exception>().WithMessage("*does not own its data*");
                using NDArray copy = tensor.ToTorchNDArray(requireGIL: false);
                copy.resize(new Shape(8));
                copy.size.Should().Be(8);
            }
        }

        [TestMethod]
        public void NullDisposedAndNonTensorInputs_FailCleanlyWithoutLeaking()
        {
            RequirePyTorch();
            int exports = NDArrayPythonInterop.LiveExports;
            int imports = NDArrayPythonInterop.LiveImports;

            NDArray nullArray = null;
            PyObject nullObject = null;
            new Action(() => TorchInterop.ToTorch(nullArray)).Should().Throw<ArgumentNullException>();
            new Action(() => TorchInterop.AsTorchNDArray(nullObject)).Should().Throw<ArgumentNullException>();
            new Action(() => TorchInterop.ToTorchNDArray(nullObject)).Should().Throw<ArgumentNullException>();

            var disposed = np.arange(3).astype(NPTypeCode.Double);
            disposed.Dispose();
            new Action(() => { using (Gil()) using (PyObject _ = disposed.ToTorch()) { } })
                .Should().Throw<ObjectDisposedException>();

            using (Gil())
            using (PyObject list = Scope.Eval("[1, 2, 3]"))
            {
                new Action(() => { using var _ = list.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<NotSupportedException>().WithMessage("*does not export a PEP 3118 buffer*IPythonArrayAdapter*");
                new Action(() => { using var _ = list.ToTorchNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*list*object has no attribute*numpy*");
            }

            Pump();
            NDArrayPythonInterop.LiveExports.Should().Be(exports);
            NDArrayPythonInterop.LiveImports.Should().Be(imports);
        }

        [TestMethod]
        public void ExplicitNoGilPolicy_WorksUnderOneOuterGil_ForEveryTorchVerb()
        {
            RequirePyTorch();
            using NDArray source = np.arange(6).astype(NPTypeCode.Double);

            using (Gil())
            using (PyObject tensor = source.ToTorch(requireGIL: false))
            using (NDArray view = tensor.AsTorchNDArray(requireGIL: false))
            using (NDArray copy = tensor.ToTorchNDArray(requireGIL: false))
            {
                view.Shape.IsWriteable.Should().BeTrue();
                WriteAt(view, 123.0, 4);
                ReadAt<double>(source, 4).Should().Be(123.0);
                ReadAt<double>(copy, 4).Should().Be(4.0,
                    "the copy was taken before the shared view mutation");
            }

            bool previous = NDArrayPythonInterop.RequireGIL;
            try
            {
                NDArrayPythonInterop.RequireGIL = false;
                using (Gil())
                using (PyObject tensor = source.ToTorch())
                using (NDArray view = tensor.AsTorchNDArray())
                using (NDArray copy = tensor.ToTorchNDArray())
                {
                    ReadAt<double>(view, 4).Should().Be(123.0);
                    ReadAt<double>(copy, 4).Should().Be(123.0);
                }
            }
            finally
            {
                NDArrayPythonInterop.RequireGIL = previous;
            }
        }

        [TestMethod]
        public void ToTorchCopy_IsDetachedBothWays_AndDoesNotKeepAnExportPin()
        {
            RequirePyTorch();
            int exports = NDArrayPythonInterop.LiveExports;
            using NDArray source = np.arange(5).astype(NPTypeCode.Double);

            using (Gil())
            using (PyObject tensor = source.ToTorch(copy: true))
                Scope.Set("edge_owned_copy", tensor);

            WaitFor(() => NDArrayPythonInterop.LiveExports == exports).Should().BeTrue(
                "copy:true may use a temporary export to copy bytes, but the owned tensor must not pin NumSharp");

            WriteAt(source, 77.0, 0);
            PyFloat("edge_owned_copy[0].item()").Should().Be(0.0,
                "NumSharp writes after the copy must not reach the tensor");
            PyExec("edge_owned_copy[1] = -88.0");
            ReadAt<double>(source, 1).Should().Be(1.0,
                "tensor writes after the copy must not reach NumSharp");
            PyExec("del edge_owned_copy");
        }

        [TestMethod]
        public void ConcurrentTorchRoundTrips_AreThreadSafeAndReturnCountersToBaseline()
        {
            RequirePyTorch();
            int exports = NDArrayPythonInterop.LiveExports;
            int imports = NDArrayPythonInterop.LiveImports;
            var failures = new ConcurrentQueue<Exception>();

            void Worker(int id)
            {
                try
                {
                    for (int k = 0; k < 12; k++)
                    {
                        using NDArray source = np.arange(16).astype(NPTypeCode.Double) + id;
                        PyObject tensor = source.ToTorch();
                        try
                        {
                            using NDArray view = tensor.AsTorchNDArray();
                            using NDArray copy = tensor.ToTorchNDArray();
                            ReadAt<double>(view, 5).Should().Be(5 + id);
                            ReadAt<double>(copy, 15).Should().Be(15 + id);
                        }
                        finally
                        {
                            using (Py.GIL())
                                tensor.Dispose();
                        }
                    }
                }
                catch (Exception e)
                {
                    failures.Enqueue(e);
                }
            }

            var threads = new Thread[4];
            for (int i = 0; i < threads.Length; i++)
            {
                int id = i;
                threads[i] = new Thread(() => Worker(id)) { IsBackground = true };
                threads[i].Start();
            }

            foreach (Thread thread in threads)
                thread.Join(60_000).Should().BeTrue("PyTorch interop workers must not deadlock");

            failures.Should().BeEmpty(string.Join(" | ", failures));
            WaitFor(() => NDArrayPythonInterop.LiveExports == exports &&
                          NDArrayPythonInterop.LiveImports == imports, 20_000).Should().BeTrue(
                $"all Torch crossings must drain (exports={NDArrayPythonInterop.LiveExports}, " +
                $"imports={NDArrayPythonInterop.LiveImports})");
        }

        [TestMethod]
        public void CodecModes_UseTheTorchAdapterForViewCopyAutoAndExplicitDisable()
        {
            RequirePyTorch();
            var auto = new NumpyCodec(NumpyCodecOptions.Default);
            var viewOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
            var copyOnly = new NumpyCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });
            var disabled = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });

            PyExec(
                "codec_tensor = torch.arange(5, dtype=torch.float64)\n" +
                "codec_grad = torch.arange(5, dtype=torch.float64, requires_grad=True)");

            using (Gil())
            using (PyObject tensor = Scope.Get("codec_tensor"))
            using (PyObject grad = Scope.Get("codec_grad"))
            {
                using PyObject typeObject = tensor.GetPythonType();
                using var tensorType = new PyType(typeObject);
                auto.CanDecode(tensorType, typeof(NDArray)).Should().BeTrue();
                disabled.CanDecode(tensorType, typeof(NDArray)).Should().BeFalse();
                disabled.TryDecode(tensor, out NDArray declined).Should().BeFalse();
                declined.Should().BeNull();

                auto.TryDecode(tensor, out NDArray autoView).Should().BeTrue();
                viewOnly.TryDecode(tensor, out NDArray strictView).Should().BeTrue();
                copyOnly.TryDecode(tensor, out NDArray snapshot).Should().BeTrue();
                using (autoView)
                using (strictView)
                using (snapshot)
                {
                    WriteAt(autoView, 55.0, 1);
                    ReadAt<double>(strictView, 1).Should().Be(55.0);
                    PyFloat("codec_tensor[1].item()").Should().Be(55.0);
                    WriteAt(snapshot, -7.0, 2);
                    PyFloat("codec_tensor[2].item()").Should().Be(2.0,
                        "Copy mode must remain detached even for a viewable CPU Tensor");
                }

                viewOnly.TryDecode(grad, out NDArray impossibleView).Should().BeFalse(
                    "View mode may not detach a gradient-tracking Tensor");
                impossibleView.Should().BeNull();

                auto.TryDecode(grad, out NDArray autoCopy).Should().BeTrue(
                    "Auto falls back through the adapter's allow-copy route");
                copyOnly.TryDecode(grad, out NDArray explicitCopy).Should().BeTrue();
                using (autoCopy)
                using (explicitCopy)
                {
                    WriteAt(autoCopy, 900.0, 0);
                    WriteAt(explicitCopy, 800.0, 1);
                    PyStr("codec_grad.detach().tolist()").Should().Be("[0.0, 1.0, 2.0, 3.0, 4.0]",
                        "implicit copy modes must not mutate autograd storage");
                }

                using PyObject parameter = Scope.Eval(
                    "torch.nn.Parameter(torch.arange(3, dtype=torch.float64))");
                using PyObject parameterTypeObject = parameter.GetPythonType();
                using var parameterType = new PyType(parameterTypeObject);
                auto.CanDecode(parameterType, typeof(NDArray)).Should().BeTrue(
                    "the Torch adapter walks MRO and recognizes Tensor subclasses");
                auto.TryDecode(parameter, out NDArray parameterCopy).Should().BeTrue();
                using (parameterCopy)
                {
                    parameterCopy.typecode.Should().Be(NPTypeCode.Double);
                    WriteAt(parameterCopy, -1.0, 0);
                    Scope.Set("codec_parameter", parameter);
                    PyFloat("codec_parameter.detach()[0].item()").Should().Be(0.0);
                }
            }
        }

        [TestMethod]
        public void CustomAdapter_AlsoFeedsDirectVerbsAndCodecWithoutNewMemoryCode()
        {
            NDArrayPythonInterop.RegisterArrayAdapter(DecliningArrayAdapter.Instance).Should().BeTrue();
            NDArrayPythonInterop.RegisterArrayAdapter(CustomArrayAdapter.Instance).Should().BeTrue();
            NDArrayPythonInterop.RegisterArrayAdapter(CustomArrayAdapter.Instance).Should().BeFalse(
                "adapter registration is idempotent by name");
            PyExec(
                "class NumSharpAdapterProbe:\n" +
                "    def __init__(self):\n" +
                "        self.payload = np.arange(6, dtype='f8')\n" +
                "class NumSharpAdapterBufferProbe(np.ndarray):\n" +
                "    pass\n" +
                "custom_adapter_source = NumSharpAdapterProbe()\n" +
                "custom_buffer_source = np.arange(4, dtype='f8').view(NumSharpAdapterBufferProbe)\n" +
                "custom_buffer_source.payload = np.arange(100, 104, dtype='f8')");

            using (Gil())
            using (PyObject wrapper = Scope.Get("custom_adapter_source"))
            using (NDArray view = wrapper.AsNDArray())
            using (NDArray copy = wrapper.ToNDArray())
            {
                var codec = new NumpyCodec();
                using PyObject typeObject = wrapper.GetPythonType();
                using var wrapperType = new PyType(typeObject);
                codec.CanDecode(wrapperType, typeof(NDArray)).Should().BeTrue();

                WriteAt(view, 71.0, 4);
                PyFloat("custom_adapter_source.payload[4]").Should().Be(71.0,
                    "the custom adapter only selected payload; the standard bridge owns sharing");
                WriteAt(copy, -8.0, 5);
                PyFloat("custom_adapter_source.payload[5]").Should().Be(5.0,
                    "the standard copy bridge remains detached");
            }

            using (Gil())
            using (PyObject bufferSource = Scope.Get("custom_buffer_source"))
            {
                var disabled = new NumpyCodec(new NumpyCodecOptions { DecodeArrayAdapters = false });
                using PyObject typeObject = bufferSource.GetPythonType();
                using var bufferType = new PyType(typeObject);
                disabled.CanDecode(bufferType, typeof(NDArray)).Should().BeTrue(
                    "disabling adapters must not disable a source's own ndarray/buffer capability");
                disabled.TryDecode(bufferSource, out NDArray nativeView).Should().BeTrue();
                using (nativeView)
                {
                    ReadAt<double>(nativeView, 0).Should().Be(0.0,
                        "the adapter's different payload must be bypassed when adapters are disabled");
                    WriteAt(nativeView, 77.0, 1);
                    PyFloat("custom_buffer_source[1]").Should().Be(77.0);
                    PyFloat("custom_buffer_source.payload[1]").Should().Be(101.0);
                }
            }
        }

        [TestMethod]
        public void AcceleratorTensor_ForceCopiesToCpu_WhenAnAcceleratorIsAvailable()
        {
            RequirePyTorch();
            string device;
            using (Gil())
            {
                if (PyBool("torch.cuda.is_available()")) device = "cuda";
                else if (PyBool("hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()")) device = "mps";
                else
                {
                    Assert.Inconclusive("no CUDA or MPS device is available on this host");
                    return;
                }

                using PyObject tensor = Scope.Eval($"torch.arange(6, dtype=torch.float64, device='{device}')");
                new Action(() => { using var _ = tensor.AsTorchNDArray(requireGIL: false); })
                    .Should().Throw<PythonException>().WithMessage("*CPU*");

                using NDArray copy = tensor.ToTorchNDArray(force: true, requireGIL: false);
                copy.typecode.Should().Be(NPTypeCode.Double);
                copy.shape.Should().Equal(6);
                for (int i = 0; i < 6; i++) ReadAt<double>(copy, i).Should().Be(i);
            }
        }

        private void RequirePyTorch() => PyTorchTestGate.Require(Scope);

        private void AssertCannotCross(string expression, string message)
        {
            using PyObject tensor = Scope.Eval(expression);
            new Action(() => { using var _ = tensor.AsTorchNDArray(requireGIL: false); })
                .Should().Throw<PythonException>().WithMessage(message);
            new Action(() => { using var _ = tensor.ToTorchNDArray(force: true, requireGIL: false); })
                .Should().Throw<PythonException>().WithMessage(message);
        }

        private sealed class CustomArrayAdapter : IPythonArrayAdapter
        {
            internal static CustomArrayAdapter Instance { get; } = new CustomArrayAdapter();

            public string Name => "NumSharp.Tests.Interop.NumSharpAdapterProbe";

            public bool CanAdapt(PyType objectType)
            {
                try
                {
                    using PyObject name = objectType.GetAttr("__name__");
                    string value = name.As<string>();
                    return value == "NumSharpAdapterProbe" || value == "NumSharpAdapterBufferProbe";
                }
                catch
                {
                    return false;
                }
            }

            public PyObject Adapt(PyObject source, bool allowCopy)
                => source.GetAttr("payload");
        }

        private sealed class DecliningArrayAdapter : IPythonArrayAdapter
        {
            internal static DecliningArrayAdapter Instance { get; } = new DecliningArrayAdapter();

            public string Name => "NumSharp.Tests.Interop.DecliningProbe";

            public bool CanAdapt(PyType objectType)
            {
                using PyObject name = objectType.GetAttr("__name__");
                return name.As<string>() == "NumSharpAdapterProbe";
            }

            public PyObject Adapt(PyObject source, bool allowCopy) => null;
        }
    }

    internal static class PyTorchTestGate
    {
        internal const string ValidatedTorchLine = "2.13.";

        internal static void Require(PyModule scope)
        {
            using (Py.GIL())
            {
                try
                {
                    using PyObject module = Py.Import("torch");
                }
                catch (PythonException)
                {
                    Assert.Inconclusive("python package 'torch' is not installed");
                }

                scope.Exec("import torch");
                using PyObject versionObject = scope.Eval("str(torch.__version__)");
                string version = versionObject.As<string>();
                if (!version.StartsWith(ValidatedTorchLine, StringComparison.Ordinal))
                    Assert.Inconclusive(
                        $"PyTorch {version} is installed; this live compatibility gate is pinned to the " +
                        "latest stable 2.13.x line validated by this change.");
            }
        }
    }
}
