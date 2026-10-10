using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Text across the bridge: NumSharp's fixed-width string arrays (<c>U</c>, code points, and <c>S</c>, bytes) and
    ///     ONNX Runtime's string tensors. Every crossing is a copy in both directions, so these tests check values, shapes
    ///     and dtypes against NumSharp's own reads and ORT's own reads, what each mode does, and that nothing is leased or
    ///     left behind. The one documented loss (a string that ends in NUL characters) is pinned too, so a change in it
    ///     is noticed.
    /// </summary>
    [TestClass]
    public class TextTests : OnnxTestBase
    {
        /// <summary>Strings that exercise the encodings: ASCII, Latin-1, a BMP character, an astral one (a surrogate pair), and the empty string.</summary>
        private static readonly string[] Sample = { "a", "héllo", "", "日本", "😀x", "end" };

        /// <summary>
        ///     Both fixed-width classes map to the ONNX string element type, sized or not, and back the string element type
        ///     maps to the string storage lane; the legacy tensor's element type is <see cref="string"/>.
        /// </summary>
        [TestMethod]
        public void DtypeMaps_MapBothTextClasses_ToTheStringElementType()
        {
            using NDArray unicode = np.array(Sample);
            using NDArray bytes = np.array(new[] { "ab", "c" }, np.dtype("S"));
            foreach (DType dtype in new[] { np.dtype("U"), np.dtype("<U5"), np.dtype("S"), np.dtype("S3"), unicode.dtype, bytes.dtype })
            {
                NDArrayOnnxInterop.TryToTensorElementType(dtype, out TensorElementType type).Should().BeTrue(dtype.ToString(true));
                type.Should().Be(TensorElementType.String, dtype.ToString(true));
                NDArrayOnnxInterop.ToTensorElementType(dtype).Should().Be(TensorElementType.String);
                NDArrayOnnxInterop.ToTensorElementClrType(dtype).Should().Be(typeof(string));
            }
        }

        /// <summary>
        ///     <c>ToOrtValue</c> of a <c>U</c> array is a string tensor of the array's shape holding every string, for every
        ///     shape (1-D, 2-D, 0-d, empty with and without a second axis) and every layout (stepped, reversed, transposed,
        ///     broadcast: read in logical order). The expected strings are NumSharp's own read of the array.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_UnicodeArray_CopiesEveryString_ForEveryShapeAndLayout()
        {
            using NDArray flat = np.array(Sample);
            using NDArray grid = np.array(Sample);
            grid.resize(2, 3);
            using NDArray scalar = np.array("solo");
            using NDArray empty = np.array(Array.Empty<string>());
            using NDArray emptyGrid = np.array(Array.Empty<string>());
            emptyGrid.resize(0, 3);
            using NDArray stepped = flat["::2"];
            using NDArray reversed = flat["::-1"];
            using NDArray transposed = grid.T;
            using NDArray row = np.array(new[] { "p", "qq" });
            using NDArray broadcast = np.broadcast_to(row, new Shape(3, 2));

            var cases = new (string Name, NDArray Array)[]
            {
                ("1-D", flat), ("2-D", grid), ("0-d", scalar), ("empty (0,)", empty), ("empty (0,3)", emptyGrid),
                ("stepped", stepped), ("reversed", reversed), ("transposed", transposed), ("broadcast", broadcast),
            };
            foreach (var (name, array) in cases)
            {
                using OrtValue value = array.ToOrtValue();
                OrtTensorTypeAndShapeInfo info = value.GetTensorTypeAndShape();
                info.IsString.Should().BeTrue(name);
                info.Shape.Should().Equal(array.shape, name);
                value.GetStringTensorAsArray().Should().Equal(NDArray.AsStringArray(array), $"{name}: the tensor holds the array's strings in logical order");
            }

            using (OrtValue value = transposed.ToOrtValue())
                value.GetStringTensorAsArray().Should().Equal("a", "日本", "héllo", "😀x", "", "end");
            NDArrayOnnxInterop.LiveExports.Should().Be(0, "ToOrtValue returns a bare value: nothing is pinned");
        }

        /// <summary>
        ///     An <c>S</c> array crosses its BYTES: ORT takes them as UTF-8 without checking them, which is what Python's
        ///     onnxruntime does with a bytes array. ASCII bytes read back as that text, and bytes that are UTF-8 for a
        ///     non-ASCII character read back as that character.
        /// </summary>
        [TestMethod]
        public void ToOrtValue_BytesArray_CrossesItsBytesAsUtf8()
        {
            using NDArray ascii = np.array(new[] { "ab", "c", "" }, np.dtype("S"));
            using (OrtValue value = ascii.ToOrtValue())
                value.GetStringTensorAsArray().Should().Equal("ab", "c", "");

            using NDArray utf8 = np.array(new[] { new byte[] { 0xC3, 0xA9 }, new byte[] { 0x41 } }, np.dtype("S"));
            using (OrtValue value = utf8.ToOrtValue())
            {
                value.GetStringElementAsSpan(0).ToArray().Should().Equal(new byte[] { 0xC3, 0xA9 }, "the bytes are stored as they are");
                value.GetStringTensorAsArray().Should().Equal("é", "A");
            }

            using NDArray stepped = ascii["::2"];
            using (OrtTensor handle = stepped.AsOrtValue())
                handle.Value.GetStringTensorAsArray().Should().Equal("ab", "");
        }

        /// <summary>
        ///     <c>AsOrtValue</c> of a text array: under the default mode and under <see cref="OnnxCodecMode.Copy"/> the handle
        ///     holds a string tensor ORT owns (not a view, no reference on the array's buffer, <c>Source</c> is the array),
        ///     and a change on ORT's side does not reach the array; <see cref="OnnxCodecMode.View"/> refuses, because text
        ///     cannot be shared.
        /// </summary>
        [TestMethod]
        public void AsOrtValue_TextArray_IsACopyOrtOwns_AndViewIsRefused()
        {
            using NDArray text = np.array(Sample);
            foreach (OnnxCodecMode mode in new[] { OnnxCodecMode.Auto, OnnxCodecMode.Copy })
            {
                using (OrtTensor handle = text.AsOrtValue(mode))
                {
                    handle.IsView.Should().BeFalse($"{mode}: text is always a copy");
                    handle.ElementType.Should().Be(TensorElementType.String);
                    handle.Shape.Should().Equal(6L);
                    handle.Source.Should().BeSameAs(text, "there is no NumSharp array ORT reads: Source is the array the strings came from");
                    handle.Value.GetStringTensorAsArray().Should().Equal(Sample);
                    text.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("the handle holds no reference on the array's buffer");
                    NDArrayOnnxInterop.LiveExports.Should().Be(1);

                    // The value reads back through both import verbs as a string tensor ORT owns: a text handle has no
                    // NumSharp buffer, so the "value this package made" shortcut of the numeric handles must not be taken.
                    using (NDArray back = handle.Value.ToNDArray())
                        NDArray.AsStringArray(back).Should().Equal(Sample, $"{mode}: ToNDArray of the handle's value");
                    using (NDArray back = handle.Value.AsNDArray())
                        NDArray.AsStringArray(back).Should().Equal(Sample, $"{mode}: AsNDArray of the handle's value");

                    handle.Value.StringTensorSetElementAt("changed".AsSpan(), 0);
                    NDArray.AsStringArray(text)[0].Should().Be("a", "ORT changed its own copy");
                }

                NDArrayOnnxInterop.LiveExports.Should().Be(0);
            }

            new Action(() => text.AsOrtValue(OnnxCodecMode.View)).Should().Throw<InvalidOperationException>()
                .WithMessage("*OnnxCodecMode.View*text*only as a copy*OnnxCodecMode.Auto*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
            text.IsDisposed.Should().BeFalse("a handle that does not own its source never disposes it");
        }

        /// <summary>
        ///     <c>ToNDArray</c> of a string tensor is an owning, C-contiguous <c>U</c> array of the tensor's shape, as wide
        ///     as the longest string counted in code points (an astral character is one), at least 1.
        /// </summary>
        [TestMethod]
        public void ToNDArray_StringTensor_IsAnOwningUnicodeArray_AsWideAsItsLongestString()
        {
            using (OrtValue tensor = StringTensor(new long[] { 2, 3 }, Sample))
            using (NDArray text = tensor.ToNDArray())
            {
                text.typecode.Should().Be(NPTypeCode.String);
                text.dtype.kind.Should().Be('U');
                text.dtype.itemsize.Should().Be(5 * 4, "<U5: 'héllo' has 5 code points, 4 bytes each");
                text.shape.Should().Equal(2, 3);
                text.flags.owndata.Should().BeTrue("the result owns its buffer");
                text.flags.c_contiguous.Should().BeTrue();
                NDArray.AsStringArray(text).Should().Equal(Sample);
            }

            using (OrtValue astral = StringTensor(new long[] { 1 }, "😀😀"))
            using (NDArray text = astral.ToNDArray())
                text.dtype.itemsize.Should().Be(2 * 4, "two code points, although four UTF-16 units");

            using (OrtValue scalar = StringTensor(Array.Empty<long>(), "solo"))
            using (NDArray text = scalar.ToNDArray())
            {
                text.ndim.Should().Be(0);
                NDArray.AsStringArray(text).Should().Equal("solo");
            }

            using (OrtValue allEmpty = StringTensor(new long[] { 2 }, "", ""))
            using (NDArray text = allEmpty.ToNDArray())
            {
                text.dtype.itemsize.Should().Be(4, "<U1: the width is at least 1");
                NDArray.AsStringArray(text).Should().Equal("", "");
            }

            foreach (long[] dims in new[] { new long[] { 0 }, new long[] { 0, 3 }, new long[] { 2, 0 } })
            {
                using OrtValue empty = StringTensor(dims);
                using NDArray text = empty.ToNDArray();
                text.shape.Should().Equal(dims);
                text.size.Should().Be(0);
                text.typecode.Should().Be(NPTypeCode.String);
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "a text import is a copy");
        }

        /// <summary>
        ///     <c>AsNDArray</c> of a string tensor cannot be a view: the default mode returns the copy, the strict mode
        ///     throws, and an owning import disposes the value once the copy exists (and leaves it alive when it refuses).
        /// </summary>
        [TestMethod]
        public void AsNDArray_StringTensor_CopiesUnderAuto_ThrowsUnderView_AndOwningDisposesTheValue()
        {
            using (OrtValue tensor = StringTensor(new long[] { 2 }, "x", "yy"))
            {
                using (NDArray text = tensor.AsNDArray())
                {
                    text.flags.owndata.Should().BeTrue("a copy, not a view");
                    NDArray.AsStringArray(text).Should().Equal("x", "yy");
                }
                using (NDArray text = tensor.AsNDArray(mode: OnnxCodecMode.Copy))
                    NDArray.AsStringArray(text).Should().Equal("x", "yy");
                new Action(() => tensor.AsNDArray(mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>().WithMessage("*AsNDArray*text*only as a copy*");
                new Action(() => tensor.AsNDArray(ownsValue: true, mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>();
                OrtProbe.IsDisposed(tensor).Should().BeFalse("a refused owning import leaves the value alive");
                NDArrayOnnxInterop.LiveImports.Should().Be(0);
            }

            OrtValue owned = StringTensor(new long[] { 1 }, "mine");
            using (NDArray text = owned.AsNDArray(ownsValue: true))
            {
                NDArray.AsStringArray(text).Should().Equal("mine");
                OrtProbe.IsDisposed(owned).Should().BeTrue("the copy needs nothing from the value, so owning it means disposing it");
            }

            // a text value the package made: owning it disposes its handle
            using NDArray source = np.array(new[] { "p", "q" });
            OrtTensor handle = source.AsOrtValue();
            using (NDArray text = handle.Value.AsNDArray(ownsValue: true))
            {
                NDArray.AsStringArray(text).Should().Equal("p", "q");
                handle.IsDisposed.Should().BeTrue();
            }
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     The documented loss of the fixed-width storage: a string that ENDS in NUL characters comes back without them
        ///     through a text array, because the padding of a fixed-width element is NUL too. A NUL inside a string
        ///     survives, and <c>ReadStringTensor</c> returns the strings exactly as ORT holds them. NumSharp needs a
        ///     variable-width string dtype to remove this (the TODO in NDArrayOnnxInterop.Text.cs).
        /// </summary>
        [TestMethod]
        public void TrailingNul_IsLostInATextArray_AndKeptByReadStringTensor()
        {
            using OrtValue tensor = StringTensor(new long[] { 3 }, "ab\0", "a\0b", "\0");
            tensor.ReadStringTensor().Should().Equal("ab\0", "a\0b", "\0");
            using NDArray text = tensor.ToNDArray();
            NDArray.AsStringArray(text).Should().Equal(new[] { "ab", "a\0b", "" }, "trailing NULs are indistinguishable from the element's padding");
        }

        /// <summary>
        ///     A real string model end to end: a text array feeds the string input, and the string output comes back as a
        ///     text array of the same shape and strings, for a <c>U</c> and an <c>S</c> input, a 2-D input, a view, the
        ///     dictionary form and "zero-copy" outputs (a text output is a copy there too). The strict input mode refuses
        ///     text, and a string output cannot be pre-allocated.
        /// </summary>
        [TestMethod]
        public void Run_StringModel_TakesAndReturnsTextArrays()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray text = np.array(Sample);
            using (NDArray echo = session.Run(text))
            {
                echo.typecode.Should().Be(NPTypeCode.String);
                echo.shape.Should().Equal(6);
                NDArray.AsStringArray(echo).Should().Equal(Sample);
            }

            using NDArray grid = np.array(Sample);
            grid.resize(3, 2);
            using NDArray transposed = grid.T;
            using (NDArray echo = session.Run(transposed))
            {
                echo.shape.Should().Equal(2, 3);
                NDArray.AsStringArray(echo).Should().Equal(NDArray.AsStringArray(transposed), "a view is fed in logical order");
            }

            using NDArray bytes = np.array(new[] { "ab", "c" }, np.dtype("S"));
            using (NDArray echo = session.Run(bytes, casting: "no"))   // text into a string input needs no cast rule
            {
                echo.dtype.kind.Should().Be('U', "a string output always reads back as Unicode");
                NDArray.AsStringArray(echo).Should().Equal("ab", "c");
            }

            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "X", text } }, zeroCopyOutputs: true);
            using (NDArray echo = outputs["Y"])
            {
                echo.flags.owndata.Should().BeTrue("a text output cannot be a view over ORT's strings");
                NDArray.AsStringArray(echo).Should().Equal(Sample);
                NDArrayOnnxInterop.LiveImports.Should().Be(0);
            }

            new Action(() => session.Run(text, mode: OnnxCodecMode.View)).Should().Throw<InvalidOperationException>().WithMessage("*'X'*text*OnnxCodecMode.Auto*");
            using NDArray target = np.array(new[] { "......", "......", "......", "......", "......", "......" });
            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", text } }, new Dictionary<string, NDArray> { { "Y", target } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y'*string tensor*Fetch it with Run*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     Text and numbers do not convert into each other yet: a text array into a numeric input is refused under every
        ///     casting rule (also "unsafe", which <c>np.can_cast</c> admits but <c>astype</c> cannot do), and a pre-allocated
        ///     text array for a numeric output is refused with a message about the dtype.
        /// </summary>
        [TestMethod]
        public void Run_TextIntoANumericInput_IsRefused_UnderEveryCastingRule()
        {
            using InferenceSession session = OpenSession("identity_float32");
            using NDArray text = np.array(new[] { "1.5", "2.5" });
            foreach (string casting in new[] { "no", "safe", "same_kind", "unsafe" })
                new Action(() => session.Run(text, casting: casting)).Should().Throw<InvalidCastException>(casting)
                    .WithMessage("*'X'*Single*text*parse*");

            using NDArray x = np.array(new[] { 1.5f, 2.5f });
            new Action(() => session.RunInto(new Dictionary<string, NDArray> { { "X", x } }, new Dictionary<string, NDArray> { { "Y", text } }))
                .Should().Throw<InvalidOperationException>().WithMessage("*'Y'*Float*text*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     One run with a text input next to a numeric input, and a text output next to a numeric output; the text
        ///     input's declared rank is validated like any input's.
        /// </summary>
        [TestMethod]
        public void Run_MixedModel_FeedsTextNextToNumbers()
        {
            using InferenceSession session = OpenSession("mixed_string_float");
            using NDArray text = np.array(new[] { "left", "right", "" });
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            IReadOnlyDictionary<string, NDArray> outputs = session.Run(new Dictionary<string, NDArray> { { "text", text }, { "x", x } });
            using (NDArray echo = outputs["echo"])
            using (NDArray y = outputs["y"])
            {
                NDArray.AsStringArray(echo).Should().Equal("left", "right", "");
                y.ToArray<float>().Should().Equal(1f, 2f, 3f);
            }

            using NDArray grid = np.array(new[] { "a", "b", "c", "d" });
            grid.resize(2, 2);
            new Action(() => session.Run(new Dictionary<string, NDArray> { { "text", grid }, { "x", x } }))
                .Should().Throw<ArgumentException>().WithMessage("*'text'*rank 1*rank 2*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     A classifier with string class labels, the usual scikit-learn export: the label output is a string tensor
        ///     (a text array through <c>Run</c> and through <c>ToNDArray</c>), and the probabilities are a sequence of maps
        ///     keyed by the label strings, which the map readers return as text keys next to float values. String-keyed
        ///     maps used to be refused.
        /// </summary>
        [TestMethod]
        public void Classifier_WithStringLabels_ReadsTheLabels_AndTheStringKeyedMaps()
        {
            using InferenceSession session = OpenSession("classifier_str");
            using NDArray scores = np.array(new[,] { { 0.1f, 0.7f, 0.2f }, { 0.5f, 0.3f, 0.2f } });

            using (NDArray labels = session.Run("X", scores, "label"))
                NDArray.AsStringArray(labels).Should().Equal("dog", "cat");

            using OrtTensor input = scores.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "label", "probabilities" });
            using (NDArray labels = outputs[0].ToNDArray())
                NDArray.AsStringArray(labels).Should().Equal("dog", "cat");

            (NDArray keys, NDArray values)[] maps = outputs[1].ToMaps();
            try
            {
                maps.Length.Should().Be(2);
                for (int row = 0; row < 2; row++)
                {
                    string[] names = NDArray.AsStringArray(maps[row].keys);
                    float[] probabilities = maps[row].values.ToArray<float>();
                    names.Should().BeEquivalentTo("cat", "dog", "bird");
                    probabilities[Array.IndexOf(names, "dog")].Should().BeApproximately(scores.GetSingle(row, 1), 1e-6f, $"row {row}: 'dog' is class 1");
                    probabilities[Array.IndexOf(names, "bird")].Should().BeApproximately(scores.GetSingle(row, 2), 1e-6f);
                    maps[row].keys.dtype.kind.Should().Be('U');
                }
            }
            finally
            {
                foreach ((NDArray keys, NDArray values) in maps)
                {
                    keys.Dispose();
                    values.Dispose();
                }
            }

            NDArrayOnnxInterop.LiveImports.Should().Be(0, "the text keys are copies and the float values were disposed");
        }

        /// <summary>The string-keyed ZipMap model through the sequence-of-maps reader, and one of its maps through the map reader.</summary>
        [TestMethod]
        public void ZipMapWithStringLabels_ReadWith_ToMaps_And_ToMap()
        {
            using InferenceSession session = OpenSession("zipmap_string");
            using NDArray scores = np.array(new[,] { { 0.25f, 0.5f, 0.25f } });
            using OrtTensor input = scores.AsOrtValue();
            using var options = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> outputs = session.Run(options, new[] { "X" }, new[] { input.Value }, new[] { "Z" });

            (NDArray keys, NDArray values)[] maps = outputs[0].ToMaps();
            using (maps[0].keys)
            using (maps[0].values)
            {
                string[] names = NDArray.AsStringArray(maps[0].keys);
                names.Should().BeEquivalentTo("cat", "dog", "bird");
                maps[0].values.ToArray<float>()[Array.IndexOf(names, "dog")].Should().Be(0.5f);
            }

            using OrtValue one = outputs[0].GetValue(0, OrtAllocator.DefaultInstance);
            (NDArray k, NDArray v) = one.ToMap();
            using (k)
            using (v)
                NDArray.AsStringArray(k).Should().HaveCount(3);
        }

        /// <summary>A sequence whose elements are string tensors reads as text arrays (copies: nothing is leased), each of its own shape and width.</summary>
        [TestMethod]
        public void SequenceOfStringTensors_ReadsAsTextArrays()
        {
            // The sequence owns the element values it is given, so they are not disposed here.
            OrtValue first = StringTensor(new long[] { 2 }, "a", "bcd");
            OrtValue second = StringTensor(new long[] { 1, 1 }, "z");
            using OrtValue sequence = OrtValue.CreateSequence(new List<OrtValue> { first, second });

            NDArray[] arrays = sequence.ToNDArrays();
            using (arrays[0])
            using (arrays[1])
            {
                arrays.Length.Should().Be(2);
                NDArray.AsStringArray(arrays[0]).Should().Equal("a", "bcd");
                arrays[0].dtype.itemsize.Should().Be(12, "<U3");
                arrays[1].shape.Should().Equal(1, 1);
                NDArray.AsStringArray(arrays[1]).Should().Equal("z");
                NDArrayOnnxInterop.LiveImports.Should().Be(0, "text elements are copies");
            }
        }

        /// <summary>
        ///     The two <c>DenseTensor&lt;T&gt;</c> export verbs refuse a text array (a string is not an unmanaged T) and say
        ///     which verbs take text and how to build the legacy tensor by hand; the refusal comes before anything is read.
        /// </summary>
        [TestMethod]
        public void DenseTensorVerbs_RefuseText_NamingTheVerbsThatTakeIt()
        {
            using NDArray text = np.array(Sample);
            new Action(() => text.AsDenseTensor<int>()).Should().Throw<NotSupportedException>().WithMessage("*AsDenseTensor*text*ToOrtValue*DenseTensor<string>*");
            new Action(() => text.ToDenseTensor<ushort>()).Should().Throw<NotSupportedException>().WithMessage("*ToDenseTensor*text*ToOrtValue*DenseTensor<string>*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);

            // what the message describes works
            var tensor = new DenseTensor<string>(NDArray.AsStringArray(text), new[] { 6 });
            using NDArray back = NamedOnnxValue.CreateFromTensor("t", tensor).ToNDArray();
            NDArray.AsStringArray(back).Should().Equal(Sample);
        }

        /// <summary>A text array whose buffer was released is refused by every text export, like a numeric one: nothing is read from freed memory.</summary>
        [TestMethod]
        public void ReleasedTextArray_IsRefused_ByEveryExport()
        {
            using InferenceSession session = OpenSession("string_io");
            foreach (string kind in new[] { "U", "S" })
            {
                NDArray released = np.array(new[] { "gone", "too" }, np.dtype(kind));
                released.Dispose();
                new Action(() => released.ToOrtValue()).Should().Throw<ObjectDisposedException>($"ToOrtValue, {kind}");
                new Action(() => released.AsOrtValue()).Should().Throw<ObjectDisposedException>($"AsOrtValue, {kind}");
                new Action(() => session.Run(released)).Should().Throw<ObjectDisposedException>($"Run, {kind}");
            }
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>Builds an ORT string tensor of a shape from its strings in row-major order (ORT's own factory and setter).</summary>
        /// <param name="dims">The shape.</param>
        /// <param name="items">The strings; their count is the shape's element count.</param>
        /// <returns>The tensor, which the caller disposes.</returns>
        private static OrtValue StringTensor(long[] dims, params string[] items)
        {
            OrtValue value = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, dims);
            for (int i = 0; i < items.Length; i++)
                value.StringTensorSetElementAt(items[i].AsSpan(), i);
            return value;
        }
    }
}
