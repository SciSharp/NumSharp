using System;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Non-tensor OrtValues — the outputs ORT produces that are not a dense tensor: sequences of tensors,
    ///     the scikit-learn ZipMap <c>sequence(map(int64, float))</c>, and string tensors — read through the
    ///     new <c>ToNDArrays</c> / <c>ToMap</c> / <c>ToMaps</c> / <c>ReadStringTensor</c> verbs; plus the two
    ///     model-INPUT rejections (a string-typed input, a sequence input) that Tier-2 <c>Run</c> refuses.
    /// </summary>
    [TestClass]
    public class NonTensorTests : OnnxTestBase
    {
        private static IDisposableReadOnlyCollection<OrtValue> Run(InferenceSession session, string inName, OrtValue input, string outName)
        {
            using var options = new RunOptions();
            return session.Run(options, new[] { inName }, new[] { input }, new[] { outName });
        }

        [TestMethod]
        public void SequenceOfTensors_ReadWith_ToNDArrays()
        {
            using InferenceSession session = OpenSession("sequence_out");
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            using OrtTensor t = x.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "seq");
            OrtValue seq = outputs[0];
            seq.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_SEQUENCE);

            NDArray[] tensors = seq.ToNDArrays();
            tensors.Length.Should().Be(1);
            using (tensors[0])
            {
                tensors[0].shape.Should().Equal(3);
                tensors[0].typecode.Should().Be(NPTypeCode.Single);
                tensors[0].GetSingle(2).Should().Be(3f);
                tensors[0].flags.owndata.Should().BeTrue("sequence elements are read as owning copies");
            }

            // a sequence is not a dense tensor / not a sequence-of-maps
            new Action(() => seq.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*SEQUENCE*ToNDArrays*");
            new Action(() => seq.ToMaps()).Should().Throw<NotSupportedException>().WithMessage("*element 0 is ONNX_TYPE_TENSOR*ToNDArrays*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "the readers copy — no lease");
        }

        [TestMethod]
        public void ZipMap_SequenceOfMaps_ReadWith_ToMaps()
        {
            using InferenceSession session = OpenSession("zipmap_int64");
            using NDArray probs = np.array(new[,] { { 0.1f, 0.7f, 0.2f }, { 0.5f, 0.3f, 0.2f } });
            using OrtTensor t = probs.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "Z");
            OrtValue z = outputs[0];
            z.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_SEQUENCE, "ZipMap emits a sequence of maps");

            (NDArray keys, NDArray values)[] maps = z.ToMaps();
            using (maps[0].keys)
            using (maps[0].values)
            using (maps[1].keys)
            using (maps[1].values)
            {
                maps.Length.Should().Be(2, "one map per input row");
                maps[0].keys.typecode.Should().Be(NPTypeCode.Int64, "class labels are int64");
                maps[0].keys.GetInt64(0).Should().Be(10L);
                maps[0].keys.GetInt64(2).Should().Be(30L);
                maps[0].values.typecode.Should().Be(NPTypeCode.Single);
                maps[0].values.GetSingle(1).Should().BeApproximately(0.7f, 1e-6f, "class 20's probability, row 0");
                maps[1].values.GetSingle(0).Should().BeApproximately(0.5f, 1e-6f);
            }
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        [TestMethod]
        public void Map_Element_ReadWith_ToMap_And_TypeGuards()
        {
            using InferenceSession session = OpenSession("zipmap_int64");
            using NDArray probs = np.array(new[,] { { 0.25f, 0.25f, 0.5f } });
            using OrtTensor t = probs.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "Z");
            OrtValue z = outputs[0];

            // pull one map element out of the sequence (the documented GetValueCount / GetValue flow)
            z.GetValueCount().Should().Be(1);
            using OrtValue m = z.GetValue(0, OrtAllocator.DefaultInstance);
            m.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_MAP);

            (NDArray keys, NDArray values) = m.ToMap();
            using (keys)
            using (values)
            {
                keys.GetInt64(0).Should().Be(10L);
                values.GetSingle(2).Should().BeApproximately(0.5f, 1e-6f);
            }

            // type guards: ToMap wants a map, ToNDArrays/ToNDArray want tensors/sequences
            new Action(() => z.ToMap()).Should().Throw<NotSupportedException>().WithMessage("*ToMap*SEQUENCE*");
            new Action(() => m.ToNDArrays()).Should().Throw<NotSupportedException>().WithMessage("*sequence*MAP*");
            new Action(() => m.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*MAP*");
        }

        [TestMethod]
        public void StringTensorOutput_ReadWith_ReadStringTensor()
        {
            using InferenceSession session = OpenSession("string_io");
            using OrtValue input = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 3 });
            input.StringTensorSetElementAt("héllo".AsSpan(), 0);
            input.StringTensorSetElementAt("wörld".AsSpan(), 1);
            input.StringTensorSetElementAt("".AsSpan(), 2);
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", input, "Y");
            OrtValue y = outputs[0];
            y.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_TENSOR);
            y.GetTensorTypeAndShape().IsString.Should().BeTrue();

            y.ReadStringTensor().Should().Equal("héllo", "wörld", "");

            // a string tensor is refused by the numeric verbs; a numeric tensor is refused by ReadStringTensor
            new Action(() => y.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*ReadStringTensor*");
            using NDArray num = np.array(new[] { 1f, 2f });
            using OrtTensor numT = num.AsOrtValue();
            new Action(() => numT.Value.ReadStringTensor()).Should().Throw<NotSupportedException>().WithMessage("*STRING tensors only*");
        }

        [TestMethod]
        public void NonTensorReaders_NullArguments_Throw()
        {
            OrtValue none = null;
            new Action(() => none.ToNDArrays()).Should().Throw<ArgumentNullException>();
            new Action(() => none.ToMap()).Should().Throw<ArgumentNullException>();
            new Action(() => none.ToMaps()).Should().Throw<ArgumentNullException>();
            new Action(() => none.ReadStringTensor()).Should().Throw<ArgumentNullException>();
            new Action(() => none.GetMapKeys()).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void StringTypedModelInput_IsRefused_WithTheReadStringsHint()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray nd = np.array(new[] { 1f, 2f });
            new Action(() => session.Run(nd)).Should().Throw<NotSupportedException>()
                .WithMessage("*'X'*String*ReadStringTensor*", "the model's input dtype has no NumSharp analog");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        [TestMethod]
        public void SequenceModelInput_IsRefused_WithTheOrtValueApiHint()
        {
            using InferenceSession session = OpenSession("sequence_input");
            using NDArray nd = np.array(new[] { 1f, 2f, 3f });
            new Action(() => session.Run(nd)).Should().Throw<NotSupportedException>()
                .WithMessage("*'seq'*SEQUENCE*not a tensor*OrtValue*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
