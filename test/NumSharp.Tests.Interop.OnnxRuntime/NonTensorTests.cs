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

        /// <summary>
        ///     A sequence of tensors: each element comes back as a view that OWNS ORT's copy of the element (one copy per
        ///     element, where the reader used to copy ORT's copy again), so it reports <c>owndata == False</c>, counts as a
        ///     live import until it is disposed, and frees ORT's copy then.
        /// </summary>
        [TestMethod]
        public void SequenceOfTensors_ReadWith_ToNDArrays()
        {
            using InferenceSession session = OpenSession("sequence_out");
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            using OrtTensor t = x.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "seq");
            OrtValue seq = outputs[0];
            seq.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_SEQUENCE);
            long pressure = NDArrayOnnxInterop.ImportPressureBytes;

            NDArray[] tensors = seq.ToNDArrays();
            tensors.Length.Should().Be(1);
            using (tensors[0])
            {
                tensors[0].shape.Should().Equal(3);
                tensors[0].typecode.Should().Be(NPTypeCode.Single);
                tensors[0].GetSingle(2).Should().Be(3f);
                tensors[0].flags.owndata.Should().BeFalse("the array is a view over ORT's copy of the element");
                tensors[0].flags.writeable.Should().BeTrue("the copy is the array's alone");
                NDArrayOnnxInterop.LiveImports.Should().Be(1, "the view owns the element until it is disposed");
                NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure + 12, "an owning view reports ORT's 3 floats to the GC");
            }
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "disposing the array freed ORT's copy of the element");
            NDArrayOnnxInterop.ImportPressureBytes.Should().Be(pressure, "and took the report back");

            // a sequence is not a dense tensor / not a sequence-of-maps
            new Action(() => seq.ToNDArray()).Should().Throw<NotSupportedException>().WithMessage("*SEQUENCE*ToNDArrays*");
            new Action(() => seq.ToMaps()).Should().Throw<NotSupportedException>().WithMessage("*element 0 is ONNX_TYPE_TENSOR*ToNDArrays*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a refused reader leaves no lease behind");
        }

        /// <summary>
        ///     The arrays a reader returns do not point into the sequence or the map they were read from: with the run's
        ///     outputs disposed first, the values are still there (and after more runs reuse ORT's memory).
        /// </summary>
        [TestMethod]
        public void ReaderResults_OutliveTheValueTheyWereReadFrom()
        {
            using InferenceSession sequenceSession = OpenSession("sequence_out");
            using InferenceSession mapSession = OpenSession("zipmap_int64");
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            using NDArray probs = np.array(new[,] { { 0.1f, 0.7f, 0.2f } });

            NDArray[] tensors;
            (NDArray keys, NDArray values)[] maps;
            using (OrtTensor t = x.AsOrtValue())
            using (IDisposableReadOnlyCollection<OrtValue> outputs = Run(sequenceSession, "X", t.Value, "seq"))
                tensors = outputs[0].ToNDArrays();
            using (OrtTensor t = probs.AsOrtValue())
            using (IDisposableReadOnlyCollection<OrtValue> outputs = Run(mapSession, "X", t.Value, "Z"))
                maps = outputs[0].ToMaps();

            // More runs of the same shapes: ORT reuses the memory the disposed outputs gave back.
            using (NDArray other = np.array(new[] { -7f, -7f, -7f }))
            using (OrtTensor t = other.AsOrtValue())
                for (int k = 0; k < 4; k++)
                    using (IDisposableReadOnlyCollection<OrtValue> again = Run(sequenceSession, "X", t.Value, "seq"))
                    {
                    }

            using (tensors[0])
            using (maps[0].keys)
            using (maps[0].values)
            {
                tensors[0].ToArray<float>().Should().Equal(1f, 2f, 3f);
                maps[0].keys.ToArray<long>().Should().Equal(10L, 20L, 30L);
                maps[0].values.GetSingle(1).Should().BeApproximately(0.7f, 1e-6f);
                NDArrayOnnxInterop.LiveImports.Should().Be(3);
            }
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }

        /// <summary>
        ///     The scikit-learn ZipMap shape, a sequence of maps: one (keys, values) pair per row, each array a view that
        ///     owns ORT's copy of its tensor.
        /// </summary>
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
                maps[0].keys.flags.owndata.Should().BeFalse("each array is a view over ORT's copy");
                NDArrayOnnxInterop.LiveImports.Should().Be(4, "two maps, a keys and a values array each");
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

            // the same tensor as a text array; a numeric tensor is refused by ReadStringTensor
            using (NDArray text = y.ToNDArray())
                NDArray.AsStringArray(text).Should().Equal("héllo", "wörld", "");
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

        /// <summary>
        ///     A string model input takes a text array and refuses a numeric one: NumSharp does not convert numbers to
        ///     text yet, so no casting rule admits the pair. (Every array used to be refused here, because NumSharp had no
        ///     text dtype.)
        /// </summary>
        [TestMethod]
        public void StringTypedModelInput_RefusesANumericArray_UnderEveryCastingRule()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray nd = np.array(new[] { 1f, 2f });
            foreach (string casting in new[] { "safe", "same_kind", "unsafe" })
                new Action(() => session.Run(nd, casting: casting)).Should().Throw<InvalidCastException>(casting)
                    .WithMessage("*'X'*string tensor*Single*text array*");
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

        /// <summary>
        ///     <c>GetMapKeys</c> hands out ONNX Runtime's own copy of a map's keys tensor: the int64 class labels of a
        ///     ZipMap, and the string labels of a string-keyed one (read with <c>ReadStringTensor</c>). The caller owns
        ///     the value; a value that is not a map is refused.
        /// </summary>
        [TestMethod]
        public void GetMapKeys_ReturnsTheKeysTensor_OfAnIntKeyedAndAStringKeyedMap()
        {
            using NDArray probs = np.array(new[,] { { 0.25f, 0.25f, 0.5f } });
            using OrtTensor t = probs.AsOrtValue();

            using (InferenceSession session = OpenSession("zipmap_int64"))
            using (IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "Z"))
            using (OrtValue map = outputs[0].GetValue(0, OrtAllocator.DefaultInstance))
            {
                using (OrtValue keys = map.GetMapKeys())
                {
                    keys.OnnxType.Should().Be(OnnxValueType.ONNX_TYPE_TENSOR);
                    keys.GetTensorDataAsSpan<long>().ToArray().Should().Equal(10L, 20L, 30L);
                    using NDArray asArray = keys.ToNDArray();
                    asArray.ToArray<long>().Should().Equal(10L, 20L, 30L);
                }

                // not a map: the sequence itself, and a tensor
                new Action(() => outputs[0].GetMapKeys()).Should().Throw<NotSupportedException>().WithMessage("GetMapKeys needs a map*SEQUENCE*");
                new Action(() => t.Value.GetMapKeys()).Should().Throw<NotSupportedException>().WithMessage("GetMapKeys needs a map*TENSOR*");
            }

            // The model's labels are cat, dog, bird. ONNX Runtime keeps a map ordered by key, so the keys of a
            // string-keyed map come back sorted, not in the order of the labels.
            using (InferenceSession session = OpenSession("zipmap_string"))
            using (IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "Z"))
            using (OrtValue map = outputs[0].GetValue(0, OrtAllocator.DefaultInstance))
            using (OrtValue keys = map.GetMapKeys())
                keys.ReadStringTensor().Should().Equal("bird", "cat", "dog");

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "GetMapKeys returns a bare value: no lease");
        }

        /// <summary>
        ///     Each reader names what it needs when it is given another kind of value: a sequence of MAPS to the
        ///     tensor-sequence reader (with the reader that takes it), a tensor or a map to the map-sequence reader, a
        ///     sequence to the string reader. A refused call leaves no array and no lease behind.
        /// </summary>
        [TestMethod]
        public void Readers_GivenAnotherKindOfValue_RefuseAndNameTheRightReader()
        {
            using InferenceSession session = OpenSession("zipmap_int64");
            using NDArray probs = np.array(new[,] { { 0.1f, 0.7f, 0.2f }, { 0.5f, 0.3f, 0.2f } });
            using OrtTensor t = probs.AsOrtValue();
            using IDisposableReadOnlyCollection<OrtValue> outputs = Run(session, "X", t.Value, "Z");
            OrtValue sequenceOfMaps = outputs[0];
            using OrtValue map = sequenceOfMaps.GetValue(0, OrtAllocator.DefaultInstance);

            new Action(() => sequenceOfMaps.ToNDArrays()).Should().Throw<NotSupportedException>()
                .WithMessage("ToNDArrays: sequence element 0 is ONNX_TYPE_MAP*ToMaps*");
            new Action(() => t.Value.ToNDArrays()).Should().Throw<NotSupportedException>().WithMessage("ToNDArrays needs a sequence*TENSOR*");
            new Action(() => t.Value.ToMaps()).Should().Throw<NotSupportedException>().WithMessage("ToMaps needs a sequence*TENSOR*");
            new Action(() => map.ToMaps()).Should().Throw<NotSupportedException>().WithMessage("ToMaps needs a sequence*MAP*");
            new Action(() => sequenceOfMaps.ReadStringTensor()).Should().Throw<NotSupportedException>().WithMessage("ReadStringTensor needs a tensor*SEQUENCE*");
            new Action(() => map.ReadStringTensor()).Should().Throw<NotSupportedException>().WithMessage("ReadStringTensor needs a tensor*MAP*");

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a refused read leaves no lease");
        }

        /// <summary>
        ///     A sequence of tensors of DIFFERENT shapes: <c>ToNDArrays</c> returns one array per element, in sequence
        ///     order, each with its own shape. And a sequence of tensors given to the map reader is refused with the
        ///     reader that takes it.
        /// </summary>
        [TestMethod]
        public void SequenceOfTwoTensors_ReadsInOrder_EachWithItsOwnShape()
        {
            using InferenceSession session = OpenSession("seq_two");
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArray b = np.array(new float[] { 7, 8, 9, 10 });
            using OrtTensor ta = a.AsOrtValue();
            using OrtTensor tb = b.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, new[] { "A", "B" }, new[] { ta.Value, tb.Value }, new[] { "S" });
            OrtValue sequence = outputs[0];
            sequence.GetValueCount().Should().Be(2);

            NDArray[] elements = sequence.ToNDArrays();
            using (elements[0])
            using (elements[1])
            {
                elements.Length.Should().Be(2);
                elements[0].shape.Should().Equal(2, 3);
                elements[0].ToArray<float>().Should().Equal(0f, 1f, 2f, 3f, 4f, 5f);
                elements[1].shape.Should().Equal(4);
                elements[1].ToArray<float>().Should().Equal(7f, 8f, 9f, 10f);
                NDArrayOnnxInterop.LiveImports.Should().Be(2, "each element is a view that owns ORT's copy of it");
            }

            new Action(() => sequence.ToMaps()).Should().Throw<NotSupportedException>()
                .WithMessage("ToMaps: sequence element 0 is ONNX_TYPE_TENSOR*ToNDArrays*");
            NDArrayOnnxInterop.LiveImports.Should().Be(0);
        }
    }
}
