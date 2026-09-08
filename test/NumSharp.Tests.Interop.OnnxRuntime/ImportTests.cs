using System;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>ORT → NumSharp on REAL session outputs: <c>ToNDArray</c> (copy) and <c>AsNDArray</c> (zero-copy lease).</summary>
    [TestClass]
    public class ImportTests : OnnxTestBase
    {
        private static IDisposableReadOnlyCollection<OrtValue> RunIdentity(InferenceSession session, OrtTensor input)
        {
            using var options = new RunOptions();
            return session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "Y" });
        }

        [TestMethod]
        public void ToNDArray_FromARealOutput_BitExact_EveryDtype_IncludingScalarAndEmpty()
        {
            foreach (NPTypeCode code in ZeroCopyDtypes)
            {
                using InferenceSession session = OpenSession(IdentityModel(code));
                foreach (int[] shape in new[] { new[] { 2, 3 }, new int[0], new[] { 0, 2 } })
                {
                    using NDArray nd = shape.Length == 0 ? NDArray.Scalar(5, code) : Arange(code, shape);
                    using OrtTensor input = nd.AsOrtValue();
                    using IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);
                    using NDArray back = outputs[0].ToNDArray();

                    back.typecode.Should().Be(code, $"{code} {nd.Shape}");
                    back.shape.Should().Equal(Dims(shape), $"{code} {nd.Shape}");
                    BytesOf(back).Should().Equal(BytesOf(nd), $"{code} {nd.Shape}");
                    back.Shape.IsContiguous.Should().BeTrue();
                }
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "copies hold no lease");
        }

        [TestMethod]
        public void AsNDArray_View_SharesOrtsOutputBuffer_BothWays()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray nd = Arange(NPTypeCode.Single, 2, 3);
            using OrtTensor input = nd.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);

            using (NDArray view = outputs[0].AsNDArray())
            {
                view.shape.Should().Equal(2, 3);
                view.GetSingle(1, 1).Should().Be(4f);
                NDArrayOnnxInterop.LiveImports.Should().Be(1);

                outputs[0].GetTensorMutableDataAsSpan<float>()[0] = 99f;
                view.GetSingle(0, 0).Should().Be(99f, "ORT's write is visible through the view");
                view[1, 2] = -5f;
                outputs[0].GetTensorDataAsSpan<float>()[5].Should().Be(-5f, "the view's write is visible to ORT");

                view.Shape.IsContiguous.Should().BeTrue();
                view.Storage.IsView.Should().BeTrue("a foreign buffer: owndata == false");
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void AsNDArray_DoesNotOwnItsData_ResizeRefuses()
        {
            using InferenceSession session = OpenSession("identity_int32");
            using NDArray nd = Arange(NPTypeCode.Int32, 6);
            using OrtTensor input = nd.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);
            using NDArray view = outputs[0].AsNDArray();

            new Action(() => view.resize(12)).Should().Throw<Exception>().WithMessage("*does not own its data*");
        }

        [TestMethod]
        public void AsNDArray_OwnsValue_TheLastView_DisposesTheOrtValue()
        {
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray nd = Arange(NPTypeCode.Double, 8);
            using OrtTensor input = nd.AsOrtValue();
            IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);   // NOT disposed: the view owns the value

            NDArray view = outputs[0].AsNDArray(ownsValue: true);
            NDArray tail = view["4:"];   // a derived slice extends the lease
            NDArrayOnnxInterop.LiveImports.Should().Be(1);

            view.Dispose();
            NDArrayOnnxInterop.LiveImports.Should().Be(1, "the slice still references the block");
            tail.GetDouble(0).Should().Be(4.0, "still readable through the slice");

            tail.Dispose();
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "the last view released the lease and disposed the OrtValue");

            // The OrtValue is now disposed (ORT's OrtValue exposes no IsDisposed and touching a disposed value is a
            // native access violation, not an exception — so it is NOT probed here). ORT's Dispose is idempotent,
            // which is what makes the never-disposed collection inert if it is disposed later.
            outputs.Dispose();
        }

        [TestMethod]
        public void AsNDArray_NonOwning_RootsTheOrtValue_AgainstCollection()
        {
            using InferenceSession session = OpenSession("identity_int64");
            using NDArray nd = Arange(NPTypeCode.Int64, 5);
            using OrtTensor input = nd.AsOrtValue();

            using NDArray view = MakeViewAndDropTheCollection(session, input);
            Settle();
            Settle();
            view.GetInt64(4).Should().Be(4L, "the lease's strong reference kept the OrtValue (and its buffer) alive across GC");
            NDArrayOnnxInterop.LiveImports.Should().Be(1);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray MakeViewAndDropTheCollection(InferenceSession session, OrtTensor input)
        {
            IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);
            return outputs[0].AsNDArray();   // `outputs` becomes unreachable when this frame returns
        }

        [TestMethod]
        public void AsNDArray_OfAnEmptyOutput_IsAnEmptyArray_NoLease()
        {
            using InferenceSession session = OpenSession("identity_uint8");
            using NDArray nd = np.zeros(new Shape(0, 4), typeof(byte));
            using OrtTensor input = nd.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = RunIdentity(session, input);
            using NDArray view = outputs[0].AsNDArray();
            view.shape.Should().Equal(0, 4);
            view.typecode.Should().Be(NPTypeCode.Byte);
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "nothing to share");
        }

        [TestMethod]
        public void NonTensorOutput_IsRefused_WithTheOrtValueApiHint()
        {
            using InferenceSession session = OpenSession("sequence_out");
            using NDArray nd = Arange(NPTypeCode.Single, 3);
            using OrtTensor input = nd.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "seq" });

            outputs[0].OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_SEQUENCE);
            new Action(() => outputs[0].ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*SEQUENCE*GetValueCount*");
            new Action(() => outputs[0].AsNDArray()).Should().Throw<NotSupportedException>().WithMessage("*SEQUENCE*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void ViewOfAnInputTensor_IsAViewOfTheNumSharpSource()
        {
            // an OrtValue built by AsOrtValue viewed back: the same memory, three names
            using NDArray nd = Arange(NPTypeCode.Int16, 4);
            using OrtTensor t = nd.AsOrtValue();
            using NDArray again = t.Value.AsNDArray();
            again[0] = (short)-1;
            nd.GetValue<short>(0).Should().Be((short)-1);
        }

        [TestMethod]
        public void NullValue_Throws()
        {
            OrtValue none = null;
            new Action(() => none.ToNDArray()).Should().Throw<ArgumentNullException>();
            new Action(() => none.AsNDArray()).Should().Throw<ArgumentNullException>();
        }
    }
}
