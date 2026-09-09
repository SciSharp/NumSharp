using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Degenerate and rare inputs to the legacy <c>DenseTensor</c> surface: 0-d scalars, empty arrays, the
    ///     1-D reverse-stride shortcut, a Char round-trip, the BFloat16 refusal on the direct generic verbs, a
    ///     string tensor through <see cref="NamedOnnxValue"/>, and a full 12-dtype round-trip sweep.
    /// </summary>
    [TestClass]
    public class DenseTensorEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void AsDenseTensor_Scalar_Is0dTensorOfLengthOne()
        {
            using NDArray scalar = NDArray.Scalar(2.5);
            using (OrtTensor<double> h = scalar.AsDenseTensor<double>())
            {
                h.Tensor.Dimensions.Length.Should().Be(0, "a 0-d array is a 0-d tensor");
                h.Tensor.Length.Should().Be(1);
                h.Tensor.GetValue(0).Should().Be(2.5);
                NDArrayOnnxInterop.LiveExports.Should().Be(1);
            }
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            DenseTensor<double> copy = scalar.ToDenseTensor<double>();
            copy.Dimensions.Length.Should().Be(0);
            copy.Length.Should().Be(1);
            copy.GetValue(0).Should().Be(2.5);
        }

        [TestMethod]
        public void AsDenseTensor_And_ToDenseTensor_Empty_HaveZeroLength()
        {
            using NDArray empty = np.zeros(new Shape(0, 3), typeof(float));
            using (OrtTensor<float> h = empty.AsDenseTensor<float>())
            {
                h.Tensor.Dimensions.ToArray().Should().Equal(0, 3);
                h.Tensor.Length.Should().Be(0);
                h.Memory.Length.Should().Be(0);
                NDArrayOnnxInterop.LiveExports.Should().Be(1, "even an empty share pins the buffer");
            }
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            DenseTensor<float> copy = empty.ToDenseTensor<float>();
            copy.Dimensions.ToArray().Should().Equal(0, 3);
            copy.Length.Should().Be(0);
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "the copy pins nothing (the size==0 path skips the pin)");
        }

        [TestMethod]
        public void DenseTensor_Empty_ToNDArray_And_AsNDArray_AreEmpty_NoLease()
        {
            var t = new DenseTensor<int>(Array.Empty<int>(), new[] { 0, 4 });

            using NDArray copy = t.ToNDArray();
            copy.shape.Should().Equal(0, 4);
            copy.typecode.Should().Be(NPTypeCode.Int32);

            using NDArray view = t.AsNDArray();
            view.shape.Should().Equal(0, 4);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "nothing to share -> no pin/lease");
        }

        [TestMethod]
        public void DenseTensor_1D_ReverseStride_TakesTheSimplePath_BothImportVerbs()
        {
            // reverseStride is a no-op for a 1-D tensor: memory order == logical order.
            var t = new DenseTensor<int>(new[] { 10, 20, 30, 40 }, new[] { 4 }, reverseStride: true);
            t.IsReversedStride.Should().BeTrue();

            using NDArray copy = t.ToNDArray();
            copy.Shape.IsContiguous.Should().BeTrue("the dims.Length < 2 branch skips the transpose");
            copy.shape.Should().Equal(4);
            copy.GetInt32(0).Should().Be(10);
            copy.GetInt32(3).Should().Be(40);

            using NDArray view = t.AsNDArray();
            view.Shape.IsContiguous.Should().BeTrue("1-D is both C- and F-contiguous, so no Fortran alias is built");
            view.GetInt32(1).Should().Be(20);
        }

        [TestMethod]
        public void DenseTensor_Char_RoundTripsAsChar_Directional()
        {
            // exporting Char -> ushort is directional (UInt16 comes back UInt16); but a genuine DenseTensor<char>
            // is unambiguously chars, so TypeCodeOf<char> == Char.
            using NDArray chars = np.array("héy".ToCharArray());
            using OrtTensor<ushort> asUnits = chars.AsDenseTensor<ushort>();
            asUnits.Tensor[1].Should().Be((ushort)'é');
            using NDArray backAsUnits = asUnits.Tensor.AsNDArray();
            backAsUnits.typecode.Should().Be(NPTypeCode.UInt16, "a ushort tensor is ushort on the way back");

            var charTensor = new DenseTensor<char>(new[] { 'a', 'b', 'c' }, new[] { 3 });
            using NDArray backAsChars = charTensor.ToNDArray();
            backAsChars.typecode.Should().Be(NPTypeCode.Char);
            backAsChars.GetValue<char>(2).Should().Be('c');
        }

        [TestMethod]
        public void DenseTensor_BFloat16_IsRefused_ByBothDirectGenericVerbs()
        {
            var t = new DenseTensor<BFloat16>(new[] { new BFloat16(0x3f80), new BFloat16(0x4000) }, new[] { 2 });
            new Action(() => t.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*bfloat16*");
            new Action(() => t.AsNDArray()).Should().Throw<NotSupportedException>().WithMessage("*bfloat16*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a refused import leases nothing");
        }

        [TestMethod]
        public void NamedOnnxValue_StringTensor_IsRefused_WithTheReadStringsHint()
        {
            var t = new DenseTensor<string>(new[] { "a", "b" }, new[] { 2 });
            var value = NamedOnnxValue.CreateFromTensor("s", t);
            new Action(() => value.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*GetStringTensorAsArray*");
        }

        [TestMethod]
        public void DenseTensor_RoundTrip_EveryZeroCopyDtype()
        {
            DenseSweep<bool>(NPTypeCode.Boolean);
            DenseSweep<byte>(NPTypeCode.Byte);
            DenseSweep<sbyte>(NPTypeCode.SByte);
            DenseSweep<short>(NPTypeCode.Int16);
            DenseSweep<ushort>(NPTypeCode.UInt16);
            DenseSweep<int>(NPTypeCode.Int32);
            DenseSweep<uint>(NPTypeCode.UInt32);
            DenseSweep<long>(NPTypeCode.Int64);
            DenseSweep<ulong>(NPTypeCode.UInt64);
            DenseSweep<Float16>(NPTypeCode.Half);
            DenseSweep<float>(NPTypeCode.Single);
            DenseSweep<double>(NPTypeCode.Double);
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        private void DenseSweep<T>(NPTypeCode code) where T : unmanaged
        {
            using NDArray nd = Arange(code, 2, 3);
            using (OrtTensor<T> h = nd.AsDenseTensor<T>())
            {
                h.Tensor.Dimensions.ToArray().Should().Equal(new[] { 2, 3 }, $"{code} shape");
                using NDArray copy = h.Tensor.ToNDArray();
                copy.typecode.Should().Be(code, $"{code} DenseTensor->NDArray");
                BytesOf(copy).Should().Equal(BytesOf(nd), $"{code} DenseTensor->NDArray bytes");
                using NDArray view = h.Tensor.AsNDArray();
                view.typecode.Should().Be(code, $"{code} DenseTensor->view");
                BytesOf(view).Should().Equal(BytesOf(nd), $"{code} view bytes");
            }

            DenseTensor<T> managedCopy = nd.ToDenseTensor<T>();
            using NDArray back = managedCopy.ToNDArray();
            BytesOf(back).Should().Equal(BytesOf(nd), $"{code} ToDenseTensor->NDArray bytes");
        }
    }
}
