using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     torch → NumSharp: every dtype and strided layout views in place; requires_grad views are read-only; devices, sparse,
    ///     conjugated and negated tensors and the two dtypes NumSharp lacks copy under Auto and throw under View.
    /// </summary>
    [TestClass]
    public class ImportTests : TorchTestBase
    {
        /// <summary>A tensor of each shared dtype: Auto and View share it, Copy owns an independent copy, all with its bits.</summary>
        [TestMethod]
        public void SharedDtypes_ViewInAutoAndView_CopyInCopy()
        {
            foreach (NPTypeCode code in SharedDtypes)
            {
                using NDArray source = Ramp(code, 3, 4);
                using Tensor t = source.ToTensor();   // a tensor that owns its memory
                foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
                {
                    using NDArray nd = t.AsNDArray(mode);
                    nd.typecode.Should().Be(code, $"{code} {mode}");
                    SameBits(nd, t).Should().BeTrue($"{code} {mode}");
                    nd.MayShareMemory(t).Should().Be(mode != TorchCodecMode.Copy, $"{code} {mode}");
                    nd.Shape.IsWriteable.Should().BeTrue($"{code} {mode}");
                    if (mode == TorchCodecMode.Copy)
                        nd.Shape.IsContiguous.Should().BeTrue();
                }
            }
        }

        /// <summary>A view writes through both ways.</summary>
        [TestMethod]
        public void View_WritesThroughBothWays()
        {
            using Tensor t = torch.arange(12, dtype: ScalarType.Float32).reshape(3, 4);
            using NDArray nd = t.AsNDArray();
            nd.SetSingle(-5f, 1, 2);
            At(t, 1, 2).Should().Be(-5, "a NumSharp write shows in the tensor");
            t.mul_(2);
            nd.GetSingle(2, 3).Should().Be(22f, "torch's in-place multiply shows in the view");
        }

        /// <summary>Strided tensors - transposed, narrowed, stepped, permuted, overlapping - are views with torch's strides.</summary>
        [TestMethod]
        public void StridedTensors_AreViewsWithTorchsStrides()
        {
            using var scope = Scope();
            Tensor b = torch.arange(60, dtype: ScalarType.Float64).reshape(3, 4, 5);
            Tensor[] layouts =
            {
                b.permute(2, 0, 1),
                b.narrow(1, 1, 2),
                b[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)],
                b.transpose(0, 2),
                b.as_strided(new long[] { 4, 3 }, new long[] { 1, 1 }, 2),   // overlapping windows
                b[1],
            };
            foreach (Tensor t in layouts)
            {
                using NDArray nd = t.AsNDArray(TorchCodecMode.View);
                SameBits(nd, t).Should().BeTrue();
                nd.MayShareMemory(t).Should().BeTrue();
                nd.Shape.strides.Should().Equal(t.stride(), "NumSharp's element strides are torch's");
            }
        }

        /// <summary>An expanded (stride-0) tensor is a read-only broadcast view in NumSharp, NumPy's rule.</summary>
        [TestMethod]
        public void ExpandedTensor_IsAReadOnlyBroadcastView()
        {
            using Tensor row = torch.arange(3, dtype: ScalarType.Float32);
            using Tensor ex = row.expand(4, 3);
            using NDArray nd = ex.AsNDArray(TorchCodecMode.View);
            nd.Shape.IsBroadcasted.Should().BeTrue();
            nd.Shape.IsWriteable.Should().BeFalse();
            ((double)np.sum(nd)).Should().Be(12);
        }

        /// <summary>A tensor that requires grad is viewed READ-ONLY (its detach() is the writeable route); Copy is writeable.</summary>
        [TestMethod]
        public void RequiresGrad_IsAReadOnlyView()
        {
            using Tensor w = torch.ones(new long[] { 2, 3 }, requires_grad: true);
            using (NDArray nd = w.AsNDArray())
            {
                nd.MayShareMemory(w).Should().BeTrue("the memory is shared...");
                nd.Shape.IsWriteable.Should().BeFalse("...and NumSharp refuses writes autograd could not see");
                Action write = () => nd["0, 0"] = NDArray.Scalar(5.0);
                write.Should().Throw<Exception>("NumSharp refuses a write into a read-only array");
                ((double)np.sum(nd)).Should().Be(6, "reads are always safe");
                using NDArray slice = nd["0"];
                slice.Shape.IsWriteable.Should().BeFalse("a view derived from the read-only view stays read-only");
            }
            using (NDArray view = w.AsNDArray(TorchCodecMode.View))
                view.Shape.IsWriteable.Should().BeFalse();
            using (Tensor d = w.detach())
            using (NDArray writeable = d.AsNDArray())
            {
                writeable.Shape.IsWriteable.Should().BeTrue("detach() opts out of autograd: a writeable view");
                writeable.MayShareMemory(w).Should().BeTrue();
            }
            using (NDArray copy = w.AsNDArray(TorchCodecMode.Copy))
            {
                copy.Shape.IsWriteable.Should().BeTrue();
                copy.MayShareMemory(w).Should().BeFalse();
            }
        }

        /// <summary>A pending conjugation copies under Auto (resolved values) and is refused under View.</summary>
        [TestMethod]
        public void ConjugateBit_CopiesResolved_ViewThrows()
        {
            using var scope = Scope();
            Tensor c = torch.complex(torch.tensor(new double[] { 1, 2 }), torch.tensor(new double[] { 3, -4 }));
            Tensor cj = c.conj();
            cj.is_conj().Should().BeTrue("conj() is lazy in libtorch");
            using (NDArray nd = cj.AsNDArray())
            {
                var values = nd.ToArray<System.Numerics.Complex>();
                values[0].Should().Be(new System.Numerics.Complex(1, -3));
                values[1].Should().Be(new System.Numerics.Complex(2, 4));
                nd.MayShareMemory(cj).Should().BeFalse();
            }
            Action view = () => cj.AsNDArray(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*pending conjugation*");
        }

        /// <summary>A pending negation (the imaginary part of a conjugated view) copies resolved under Auto, throws under View.</summary>
        [TestMethod]
        public void NegativeBit_CopiesResolved_ViewThrows()
        {
            using var scope = Scope();
            Tensor c = torch.complex(torch.tensor(new double[] { 1, 2 }), torch.tensor(new double[] { 3, -4 }));
            Tensor im = c.conj().imag;
            im.is_neg().Should().BeTrue();
            using (NDArray nd = im.AsNDArray())
                nd.ToArray<double>().Should().Equal(-3.0, 4.0);
            Action view = () => im.AsNDArray(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*pending negation*");
        }

        /// <summary>complex64 widens exactly to complex128 (signed zeros, NaN, strided input), and View refuses it.</summary>
        [TestMethod]
        public void Complex64_WidensExactly()
        {
            using var scope = Scope();
            Tensor re = torch.tensor(new float[] { 1.5f, -0.0f, float.NaN, 3e38f, float.Epsilon, -2f });
            Tensor imag = torch.tensor(new float[] { -0.0f, 2f, 0.25f, float.NegativeInfinity, 1f, 0f });
            Tensor c64 = torch.complex(re, imag).reshape(2, 3);
            c64.dtype.Should().Be(ScalarType.ComplexFloat32);
            foreach (Tensor t in new[] { c64, c64.t() })
            {
                using NDArray nd = t.AsNDArray();
                nd.typecode.Should().Be(NPTypeCode.Complex);
                nd.shape.Should().Equal(t.shape);
                Tensor wide = t.to_type(ScalarType.ComplexFloat64);   // torch's own widening, the oracle
                SameBits(nd, wide).Should().BeTrue("float32 to float64 is exact, so NumSharp's widening equals torch's bit for bit");
            }
            Action view = () => c64.AsNDArray(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*ComplexFloat32*widened*");
        }

        /// <summary>bfloat16 widens exactly to float32 (the 16 bits shifted into the top half), and View refuses it.</summary>
        [TestMethod]
        public void BFloat16_WidensExactly()
        {
            using var scope = Scope();
            Tensor f = torch.tensor(new float[] { 1.5f, -0.0f, float.NaN, 3.0e38f, float.NegativeInfinity, 1e-40f, 65504f, -7.25f }).reshape(2, 4);
            Tensor bf = f.to_type(ScalarType.BFloat16);
            foreach (Tensor t in new[] { bf, bf.t() })
            {
                using NDArray nd = t.AsNDArray(TorchCodecMode.Copy);
                nd.typecode.Should().Be(NPTypeCode.Single);
                Tensor wide = t.to_type(ScalarType.Float32);
                SameBits(nd, wide).Should().BeTrue("the shift is exact, so NumSharp's widening equals torch's bit for bit");
            }
            Action view = () => bf.AsNDArray(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*BFloat16*widened*");
        }

        /// <summary>A meta tensor holds no values: refused in every mode.</summary>
        [TestMethod]
        public void MetaTensor_IsRefusedInEveryMode()
        {
            using Tensor meta = torch.empty(new long[] { 3 }, device: new torch.Device(DeviceType.META));
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                Action act = () => meta.AsNDArray(mode);
                act.Should().Throw<NotSupportedException>().WithMessage("*meta tensor holds no values*");
            }
        }

        /// <summary>A sparse tensor densifies under Auto and Copy and is refused under View (to_sparse exists from TorchSharp 0.107).</summary>
        [TestMethod]
        public void SparseTensor_Densifies_ViewThrows()
        {
            using var scope = Scope();
            MethodInfo toSparse = typeof(Tensor).GetMethod("to_sparse", Type.EmptyTypes);
            if (toSparse is null)
                Assert.Inconclusive($"TorchSharp {typeof(torch).Assembly.GetName().Version} has no Tensor.to_sparse() to make a sparse tensor with.");
            Tensor dense = torch.eye(3);
            var sparse = (Tensor)toSparse.Invoke(dense, null);
            sparse.is_sparse.Should().BeTrue();
            using (NDArray nd = sparse.AsNDArray())
                SameBits(nd, dense).Should().BeTrue();
            Action view = () => sparse.AsNDArray(TorchCodecMode.View);
            view.Should().Throw<InvalidOperationException>().WithMessage("*sparse*");
        }

        /// <summary>A quantized tensor is refused in every mode, naming dequantize() and int_repr() (quantize_per_tensor needs 0.107).</summary>
        [TestMethod]
        public void QuantizedTensor_IsRefusedInEveryMode()
        {
            using var scope = Scope();
            MethodInfo quantize = typeof(torch).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .FirstOrDefault(m => m.Name == "quantize_per_tensor" && m.GetParameters().Length == 4 && m.GetParameters()[0].ParameterType == typeof(Tensor));
            if (quantize is null)
                Assert.Inconclusive($"TorchSharp {typeof(torch).Assembly.GetName().Version} has no public quantize_per_tensor.");
            var q = (Tensor)quantize.Invoke(null, new object[] { torch.ones(4), 0.5, 0L, (ScalarType)12 });
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                Action act = () => q.AsNDArray(mode);
                act.Should().Throw<NotSupportedException>().WithMessage("*quantized*dequantize()*");
            }
        }

        /// <summary>libtorch's own uint16 (c10 code 27, unnamed in TorchSharp) imports as a NumSharp uint16 view (libtorch 2.3+).</summary>
        [TestMethod]
        public unsafe void LibtorchUInt16_ImportsAsUInt16()
        {
            if (typeof(torch).Assembly.GetName().Version.Minor < 103)
                Assert.Inconclusive("libtorch has uint16 from 2.3 (TorchSharp 0.103) on.");
            using NDArray source = np.array(new ushort[] { 1, 2, 65535 });
            // Lend the buffer to libtorch as code 27 through the bridge's own native seam, then read it back.
            using Tensor u16 = UnsignedTensor(source, 27);
            u16.dtype.Should().Be((ScalarType)27);
            using NDArray nd = u16.AsNDArray(TorchCodecMode.View);
            nd.typecode.Should().Be(NPTypeCode.UInt16);
            nd.ToArray<ushort>().Should().Equal((ushort)1, (ushort)2, (ushort)65535);
            nd.MayShareMemory(u16).Should().BeTrue();
            nd.SetUInt16(7, 0);
            source.GetUInt16(0).Should().Be(7, "the view reads the very buffer the uint16 tensor was made over");
        }

        /// <summary>Empty tensors keep their shape; a 0-d tensor is a 0-d view.</summary>
        [TestMethod]
        public void EmptyAndZeroD_Tensors()
        {
            using Tensor empty = torch.empty(new long[] { 2, 0, 3 });
            foreach (TorchCodecMode mode in new[] { TorchCodecMode.Auto, TorchCodecMode.View, TorchCodecMode.Copy })
            {
                using NDArray nd = empty.AsNDArray(mode);
                nd.shape.Should().Equal(new long[] { 2, 0, 3 }, $"{mode}");
            }
            using Tensor zeroD = torch.tensor(6.5);
            using NDArray z = zeroD.AsNDArray(TorchCodecMode.View);
            z.ndim.Should().Be(0);
            z.GetDouble().Should().Be(6.5);
            zeroD.fill_(1.25);
            z.GetDouble().Should().Be(1.25, "the 0-d view shares the tensor's element");
        }

        /// <summary>A disposed tensor, a null tensor and an undefined mode are refused.</summary>
        [TestMethod]
        public void BadArguments_AreRefused()
        {
            Tensor t = torch.ones(2);
            t.Dispose();
            Action disposed = () => t.AsNDArray();
            disposed.Should().Throw<ObjectDisposedException>();
            Action nul = () => ((Tensor)null).AsNDArray();
            nul.Should().Throw<ArgumentNullException>();
            using Tensor ok = torch.ones(2);
            Action mode = () => ok.AsNDArray((TorchCodecMode)7);
            mode.Should().Throw<ArgumentOutOfRangeException>();
        }

        /// <summary>A tensor libtorch made over the given NumSharp buffer with an arbitrary c10 dtype code (a test-only seam).</summary>
        /// <param name="source">The array whose buffer the tensor reads (its dtype's size must match the code's).</param>
        /// <param name="code">The c10 ScalarType ordinal.</param>
        /// <returns>The tensor; libtorch's deleter releases the buffer when it dies.</returns>
        private static unsafe Tensor UnsignedTensor(NDArray source, int code)
        {
            var slice = source.Storage.InternalArray;
            slice.TryAddRef().Should().BeTrue();
            var lease = new ExportLease(slice);
            IntPtr h = LibTorchSharpNative.NewTensor(slice.Address, lease.FunctionPointer, source.shape, (ScalarType)code);
            if (h == IntPtr.Zero)
            {
                lease.Release();
                torch.CheckForErrors();
            }
            return Tensor.UnsafeCreateTensor(h);
        }
    }
}
