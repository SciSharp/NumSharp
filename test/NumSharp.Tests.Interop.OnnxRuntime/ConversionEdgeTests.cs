using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Zero-copy conversion corners: a reduced-dimension 0-d view, a contiguous 1-D window at a non-zero
    ///     offset, and the empty / scalar cases of the standalone-<see cref="OrtValue"/> import verbs
    ///     (<c>ToNDArray</c> / <c>AsNDArray</c> with and without <c>ownsValue</c>).
    /// </summary>
    [TestClass]
    public class ConversionEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void ReducedDimensionScalarView_ExportsAsA0dTensor_ReadingTheRightElement()
        {
            using NDArray arr = Arange(NPTypeCode.Single, 6);
            using NDArray scalarView = arr["2"];   // 0-d, addresses element 2
            scalarView.ndim.Should().Be(0);
            scalarView.Shape.IsContiguous.Should().BeTrue("a 0-d view is contiguous");

            using OrtTensor t = scalarView.AsOrtValue();
            t.Value.GetTensorTypeAndShape().Shape.Should().BeEmpty("0-d tensor");
            t.Value.GetTensorDataAsSpan<float>()[0].Should().Be(2f, "the re-seated address points at element 2");

            using NDArray back = t.Value.ToNDArray();
            back.ndim.Should().Be(0);
            back.GetSingle().Should().Be(2f);
        }

        [TestMethod]
        public void ContiguousWindowAtOffset_SharesExactlyItsSlice()
        {
            using NDArray arr = Arange(NPTypeCode.Int64, 8);
            using NDArray window = arr["3:6"];   // contiguous, offset 3
            using OrtTensor t = window.AsOrtValue();
            t.Value.GetTensorTypeAndShape().Shape.Should().Equal(3L);
            t.Value.GetTensorDataAsSpan<long>().ToArray().Should().Equal(3L, 4L, 5L);

            t.Value.GetTensorMutableDataAsSpan<long>()[0] = -1;
            arr.GetInt64(3).Should().Be(-1L, "the window is the parent's own memory");
        }

        [TestMethod]
        public void StandaloneOrtValue_Empty_ToNDArray_IsEmpty()
        {
            using NDArray empty = np.zeros(new Shape(0, 4), typeof(int));
            using OrtValue v = empty.ToOrtValue();
            using NDArray back = v.ToNDArray();
            back.shape.Should().Equal(0, 4);
            back.typecode.Should().Be(NPTypeCode.Int32);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a copy holds no lease");
        }

        [TestMethod]
        public void StandaloneOrtValue_Empty_AsNDArray_NonOwning_IsEmpty_NoLease()
        {
            using NDArray empty = np.zeros(new Shape(0, 4), typeof(byte));
            using OrtValue v = empty.ToOrtValue();
            using NDArray view = v.AsNDArray();   // ownsValue: false — the value stays ours
            view.shape.Should().Equal(0, 4);
            view.typecode.Should().Be(NPTypeCode.Byte);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "nothing to share");
        }

        [TestMethod]
        public void StandaloneOrtValue_Empty_AsNDArray_Owning_DisposesTheValue_NoLease()
        {
            using NDArray empty = np.zeros(new Shape(0), typeof(double));
            OrtValue v = empty.ToOrtValue();
            // ownsValue + empty: the value is disposed inside AsNDArray and no lease is created — do NOT dispose v again.
            using NDArray view = v.AsNDArray(ownsValue: true);
            view.shape.Should().Equal(0);
            view.typecode.Should().Be(NPTypeCode.Double);
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void StandaloneOrtValue_AsNDArray_Owning_LastViewDisposesTheValue()
        {
            using NDArray src = Arange(NPTypeCode.Single, 6);
            OrtValue v = src.ToOrtValue();   // an independent ORT-allocated copy
            NDArray view = v.AsNDArray(ownsValue: true);
            NDArrayOnnxInterop.LiveImports.Should().Be(1);

            view.GetSingle(5).Should().Be(5f, "the view reads ORT's own copy, not the source");
            view.Dispose();
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "the last (only) view released the lease and disposed the OrtValue");
        }

        [TestMethod]
        public void StandaloneOrtValue_Scalar_RoundTrips()
        {
            using NDArray scalar = NDArray.Scalar(-7.5);
            using OrtValue v = scalar.ToOrtValue();
            v.GetTensorTypeAndShape().Shape.Should().BeEmpty();
            using NDArray back = v.ToNDArray();
            back.ndim.Should().Be(0);
            back.GetDouble().Should().Be(-7.5);
        }
    }
}
