using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>The legacy surface: <c>AsDenseTensor&lt;T&gt;</c> / <c>ToDenseTensor&lt;T&gt;</c> and the DenseTensor → NDArray twins.</summary>
    [TestClass]
    public class DenseTensorTests : OnnxTestBase
    {
        [TestMethod]
        public void AsDenseTensor_SharesTheBuffer_MutationsBothWays()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            using OrtTensor<float> h = nd.AsDenseTensor<float>();

            DenseTensor<float> t = h.Tensor;
            t.Dimensions.ToArray().Should().Equal(2, 3);
            t.IsReversedStride.Should().BeFalse();
            t[1, 2].Should().Be(5f);

            t[0, 1] = 42f;
            nd.GetSingle(0, 1).Should().Be(42f, "the tensor writes NumSharp's memory");
            nd[1, 0] = -3f;
            t[1, 0].Should().Be(-3f, "NumSharp writes the memory the tensor reads");
            h.Memory.Length.Should().Be(6);
        }

        [TestMethod]
        public void AsDenseTensor_FortranOrder_SharesColumnMajor_WithReverseStride()
        {
            using NDArray c = Arange(NPTypeCode.Int32, 3, 4);
            using NDArray f = np.asfortranarray(c);
            f.Shape.IsFContiguous.Should().BeTrue();
            f.Shape.IsContiguous.Should().BeFalse();

            using OrtTensor<int> h = f.AsDenseTensor<int>();
            h.Tensor.IsReversedStride.Should().BeTrue("column-major memory is read in place with reversed strides");
            for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                h.Tensor[i, j].Should().Be(c.GetInt32(i, j), $"[{i},{j}]");

            h.Tensor[2, 1] = 99;
            f.GetInt32(2, 1).Should().Be(99);
        }

        [TestMethod]
        public void AsDenseTensor_WrongElementType_IsRefused_NotReinterpreted()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 4);
            new Action(() => nd.AsDenseTensor<int>()).Should().Throw<ArgumentException>().WithMessage("*DenseTensor<Single>*");
            new Action(() => nd.ToDenseTensor<double>()).Should().Throw<ArgumentException>().WithMessage("*DenseTensor<Single>*");

            using NDArray half = Arange(NPTypeCode.Half, 4);
            new Action(() => half.AsDenseTensor<Half>()).Should().Throw<ArgumentException>()
                .WithMessage("*Float16*", "ORT's DenseTensor spells float16 as Microsoft.ML.OnnxRuntime.Float16");
            using OrtTensor<Float16> ok = half.AsDenseTensor<Float16>();
            ((float)ok.Tensor[3]).Should().Be(3f);

            using NDArray chars = np.array("xy".ToCharArray());
            using OrtTensor<ushort> asUnits = chars.AsDenseTensor<ushort>();
            asUnits.Tensor[1].Should().Be((ushort)'y');
        }

        [TestMethod]
        public void AsDenseTensor_StridedView_IsRefused()
        {
            using NDArray m = Arange(NPTypeCode.Double, 4, 4);
            using NDArray stepped = m["::2"];
            new Action(() => stepped.AsDenseTensor<double>()).Should().Throw<InvalidOperationException>().WithMessage("*ToDenseTensor*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void ToDenseTensor_IsAnIndependentManagedCopy_InLogicalOrder()
        {
            using NDArray m = Arange(NPTypeCode.Int64, 2, 3);
            using NDArray t = m.T;   // (3,2), non-contiguous
            DenseTensor<long> copy = t.ToDenseTensor<long>();
            copy.Dimensions.ToArray().Should().Equal(3, 2);
            copy.Buffer.ToArray().Should().Equal(0L, 3L, 1L, 4L, 2L, 5L);
            copy[0, 0] = 77;
            m.GetInt64(0, 0).Should().Be(0L, "a copy shares nothing");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "a copy pins nothing");
        }

        [TestMethod]
        public void Decimal_ConvertsToDouble_ForBothDenseTensorVerbs()
        {
            using NDArray dec = np.array(new[] { 1.25m, 2m });
            using OrtTensor<double> h = dec.AsDenseTensor<double>();
            h.Tensor.Buffer.ToArray().Should().Equal(1.25, 2.0);
            h.Source.typecode.Should().Be(NPTypeCode.Double);
            dec.ToDenseTensor<double>().Buffer.ToArray().Should().Equal(1.25, 2.0);
        }

        [TestMethod]
        public void DenseTensor_ToNDArray_And_AsNDArray_RowMajor()
        {
            var tensor = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 });

            using NDArray copy = tensor.ToNDArray();
            copy.typecode.Should().Be(NPTypeCode.Single);
            copy.shape.Should().Equal(2, 3);
            copy.GetSingle(1, 2).Should().Be(6f);
            copy[0, 0] = 100f;
            tensor[0, 0].Should().Be(1f, "ToNDArray copies");

            using (NDArray view = tensor.AsNDArray())
            {
                view.shape.Should().Equal(2, 3);
                view.GetSingle(1, 0).Should().Be(4f);
                view[1, 0] = -4f;
                tensor[1, 0].Should().Be(-4f, "AsNDArray shares the tensor's (pinned) buffer");
                NDArrayOnnxInterop.LiveImports.Should().Be(1);
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "disposing the view released the pin");
        }

        [TestMethod]
        public void DenseTensor_ColumnMajor_ToNDArray_TransposesBack_AsNDArray_IsAFortranView()
        {
            // memory 1..6 read column-major as (2,3): [[1,3,5],[2,4,6]]
            var tensor = new DenseTensor<int>(new[] { 1, 2, 3, 4, 5, 6 }, new[] { 2, 3 }, reverseStride: true);
            tensor[0, 1].Should().Be(3);

            using NDArray copy = tensor.ToNDArray();
            copy.Shape.IsContiguous.Should().BeTrue("ToNDArray always yields a C-contiguous owning copy");
            copy.GetInt32(0, 1).Should().Be(3);
            copy.GetInt32(1, 2).Should().Be(6);

            using NDArray view = tensor.AsNDArray();
            view.Shape.IsFContiguous.Should().BeTrue("the view keeps the tensor's column-major layout — no copy");
            view.GetInt32(0, 1).Should().Be(3);
            view.GetInt32(1, 0).Should().Be(2);
            view[1, 2] = 60;
            tensor[1, 2].Should().Be(60);
        }

        [TestMethod]
        public void Float16Tensor_ReadsBackAsHalf()
        {
            var tensor = new DenseTensor<Float16>(new[] { (Float16)1.5f, (Float16)(-2f) }, new[] { 2 });
            using NDArray nd = tensor.ToNDArray();
            nd.typecode.Should().Be(NPTypeCode.Half);
            ((float)nd.GetHalf(0)).Should().Be(1.5f);
            ((float)nd.GetHalf(1)).Should().Be(-2f);
        }

        [TestMethod]
        public void AsDenseTensor_FeedsTheLegacyNamedOnnxValueRun_AndTheResultReadsBack()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            using OrtTensor<float> h = nd.AsDenseTensor<float>();

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("X", h.Tensor) };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
            results.Count.Should().Be(1);
            results[0].Name.Should().Be("Y");

            using NDArray back = results[0].ToNDArray();   // runtime dispatch on the tensor's element type
            back.typecode.Should().Be(NPTypeCode.Single);
            back.shape.Should().Equal(2, 3);
            BytesOf(back).Should().Equal(BytesOf(nd));
        }

        [TestMethod]
        public void NamedOnnxValue_NonTensor_IsRefused()
        {
            var seq = NamedOnnxValue.CreateFromSequence("s", new List<float> { 1f });
            new Action(() => seq.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*not a tensor*");
        }
    }
}
