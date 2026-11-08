using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Text crossing the System.Numerics.Tensors bridge, always as a COPY: <c>ToStringTensor()</c> copies a <c>U</c> /
    ///     <c>S</c> / <c>StringDType</c> array into a <c>Tensor&lt;string&gt;</c>, and <c>ToNDArray</c> over a
    ///     <c>Tensor&lt;string&gt;</c> or a string span copies it into a <c>StringDType(na_object=None)</c> array. No string can
    ///     be shared between NumSharp's text storage (UCS-4 code units, bytes, a UTF-8 arena) and managed string objects, so
    ///     under the bridges' Auto rule (a view when possible, otherwise a copy) text always takes the copy.
    /// </summary>
    /// <remarks>
    ///     NumPy is the reference for each element: the text numpy's <c>arr.ravel().tolist()</c> gives — a <c>U</c> element
    ///     without trailing NULs, an <c>S</c> element's UTF-8 decode (<c>np.strings.decode</c>), a <c>StringDType</c> text,
    ///     with an NA whose object is not a string (None, NaN) as <c>null</c> — and on the way back
    ///     <c>np.array(texts, dtype=StringDType(na_object=None))</c>, a <c>null</c> as the NA.
    /// </remarks>
    [TestClass]
    public class TextCrossingTests : TensorsTestBase
    {
        /// <summary>U+1F600: one code point, two UTF-16 units, four UTF-8 bytes.</summary>
        private static readonly string Emoji = char.ConvertFromUtf32(0x1F600);

        /// <summary>U+00E9, two UTF-8 bytes.</summary>
        private static readonly string Acute = ((char)0xE9).ToString();

        /// <summary>
        ///     The text arrays: <c>U1</c> (truncated), <c>U5</c>, an <c>S</c> array of UTF-8 bytes (<c>np.strings.encode</c>),
        ///     <c>StringDType()</c>, and <c>StringDType</c> with each NA object (None, NaN, the string "NA") holding an NA — each
        ///     in C 2-D order, F order, transposed, reversed (a negative stride), stepped to the end, broadcast, 0-d, empty and
        ///     3-D.
        /// </summary>
        /// <returns>(label, array) pairs.</returns>
        private static IEnumerable<(string Name, NDArray Array)> TextArrays()
        {
            string[] texts = { "alpha", "be", "", Acute + "t" + Acute, Emoji, "x\0y" };
            var bases = new (string, NDArray)[]
            {
                ("U1", np.array(texts).astype(np.dtype("U1"))),
                ("U5", np.array(texts)),
                ("S (UTF-8 bytes)", np.strings.encode(np.array(texts))),
                ("StringDType()", np.array(texts, np.dtypes.StringDType())),
                ("StringDType(na_object=None) + NA", np.array(new object[] { "alpha", null, "", Acute, Emoji, "x\0y" }, np.dtypes.StringDType((object)null))),
                ("StringDType(na_object=nan) + NA", np.array(new object[] { "alpha", double.NaN, "", Acute, Emoji, "x\0y" }, np.dtypes.StringDType(double.NaN))),
                ("StringDType(na_object='NA') + NA", np.array(new object[] { "alpha", "NA", "", Acute, Emoji, "x\0y" }, np.dtypes.StringDType("NA"))),
            };
            foreach (var (name, b) in bases)
            {
                NDArray grid = b.reshape(2, 3);
                yield return ($"{name} C", grid);
                yield return ($"{name} F", np.asfortranarray(grid));
                yield return ($"{name} transposed", grid.T);
                yield return ($"{name} reversed", b["::-1"]);
                yield return ($"{name} stepped", b["1::2"]);
                yield return ($"{name} broadcast", np.broadcast_to(b["0:3"], new Shape(2, 3)));
                yield return ($"{name} 0-d", b["3"]);
                yield return ($"{name} empty", b["3:3"]);
                yield return ($"{name} 3-D", b.reshape(1, 2, 3));
            }
        }

        /// <summary>
        ///     numpy's <c>arr.ravel().tolist()</c> of a text array as strings: bytes decoded as UTF-8, an NA object that is not
        ///     a string as <c>null</c>. Read through <c>tolist()</c>, a path the verb does not take for <c>U</c> / <c>S</c>.
        /// </summary>
        /// <param name="text">A text array.</param>
        /// <returns>The expected tensor elements.</returns>
        private static string[] NumpysTexts(NDArray text)
            => ((IEnumerable)text.flatten().tolist()).Cast<object>()
                .Select(o => o is byte[] bytes ? System.Text.Encoding.UTF8.GetString(bytes) : o as string)
                .ToArray();

        /// <summary>The elements of a <c>StringDType</c> array in C order (<c>null</c> for the NA).</summary>
        /// <param name="nd">The array.</param>
        /// <returns>Its texts.</returns>
        private static object[] Elements(NDArray nd)
        {
            using NDArray flat = nd.flatten();
            return Enumerable.Range(0, (int)flat.size).Select(i => flat.GetValue(i)).ToArray();
        }

        /// <summary>
        ///     Every text class × layout (63 cells: 0-d, empty, broadcast, reversed and 3-D included) copies into a
        ///     <c>Tensor&lt;string&gt;</c> of the array's shape — a 0-d array as the vector <c>[1]</c>, as the numeric verbs —
        ///     holding numpy's <c>tolist()</c> texts in C order: <c>S</c> decoded as UTF-8, a None / NaN NA as <c>null</c>, a
        ///     string NA as its text.
        /// </summary>
        [TestMethod]
        public void ToStringTensor_EveryClassAndLayout_IsNumpysTolist()
        {
            int cells = 0;
            foreach (var (name, text) in TextArrays())
            {
                cells++;
                Tensor<string> tensor = text.ToStringTensor();
                long[] lengths = tensor.Lengths.ToArray().Select(l => (long)l).ToArray();
                lengths.Should().Equal(text.ndim == 0 ? new long[] { 1 } : text.shape, $"{name}: the tensor has the array's shape");
                var elements = new string[tensor.FlattenedLength];
                tensor.AsReadOnlyTensorSpan().FlattenTo(elements);
                elements.Should().Equal(NumpysTexts(text), $"{name}: numpy's tolist() in C order");
            }

            cells.Should().Be(7 * 9, "every class × layout ran");
        }

        /// <summary>
        ///     A <c>Tensor&lt;string&gt;</c> copies into <c>StringDType(na_object=None)</c> of its lengths — every text whole
        ///     (empty, astral, an embedded NUL), a <c>null</c> as the NA — through the tensor, its read-only span and its
        ///     mutable span alike; a slice reads its own elements; <c>Tensor&lt;string&gt;.Empty</c> (rank 0) is the empty
        ///     vector <c>(0,)</c> and a zero-length axis keeps its shape.
        /// </summary>
        [TestMethod]
        public void ToNDArray_TensorOfString_IsStringDType_NullIsTheNA()
        {
            string[] values = { "a", null, "", Emoji, "x\0y", Acute };
            Tensor<string> tensor = Tensor.Create(values, new nint[] { 2, 3 });
            foreach (var (route, nd) in new (string, NDArray)[]
                     {
                         ("Tensor", tensor.ToNDArray()),
                         ("ReadOnlyTensorSpan", tensor.AsReadOnlyTensorSpan().ToNDArray()),
                         ("TensorSpan", tensor.AsTensorSpan().ToNDArray()),
                     })
            {
                nd.dtype.ToString().Should().Be("StringDType(na_object=None)", route);
                nd.shape.Should().Equal(new long[] { 2, 3 }, route);
                Elements(nd).Should().Equal(values.Cast<object>(), route);
            }

            Tensor<string> column = tensor.Slice(new[] { NRange.All, new NRange(1, 2) });
            NDArray sliced = column.ToNDArray();
            sliced.shape.Should().Equal(new long[] { 2, 1 }, "a slice keeps its lengths");
            Elements(sliced).Should().Equal(new object[] { null, "x\0y" }, "a slice reads its own elements");

            Tensor<string>.Empty.ToNDArray().shape.Should().Equal(new long[] { 0 }, "rank 0 holds no element: the empty vector");
            Tensor.Create(new string[0], new nint[] { 2, 0 }).ToNDArray().shape.Should().Equal(new long[] { 2, 0 });
        }

        /// <summary>
        ///     A text array round-trips through <c>Tensor&lt;string&gt;</c> as the same texts in <c>StringDType</c>: a <c>U</c>
        ///     array's values, a <c>StringDType(na_object=None)</c> array's NA at its place, and a 0-d array as the vector
        ///     <c>(1,)</c> (the BCL has no rank 0).
        /// </summary>
        [TestMethod]
        public void TextArray_RoundTripsThroughTensorOfString()
        {
            using NDArray u = np.array(new[] { "alpha", "be", Emoji, "x\0y" }).reshape(2, 2);
            NDArray back = u.ToStringTensor().ToNDArray();
            back.shape.Should().Equal(new long[] { 2, 2 });
            Elements(back).Should().Equal(NDArray.AsStringArray(u).Cast<object>(), "a U array's texts");

            using NDArray withNa = np.array(new object[] { "a", null, "c" }, np.dtypes.StringDType((object)null));
            Elements(withNa.ToStringTensor().ToNDArray()).Should().Equal(new object[] { "a", null, "c" }, "the NA comes back as the NA");

            using NDArray scalar = np.array(new[] { "zero-d" })["0"];
            NDArray one = scalar.ToStringTensor().ToNDArray();
            one.shape.Should().Equal(new long[] { 1 }, "a 0-d array crosses as the vector [1]");
            Elements(one).Should().Equal(new object[] { "zero-d" });
        }

        /// <summary>
        ///     What the text verbs refuse: a numeric array (<c>ToTensor&lt;T&gt;</c> copies those, and the message says which
        ///     <c>T</c>), a null array or tensor, an <c>S</c> element that is not UTF-8 (numpy's <c>UnicodeDecodeError</c> from
        ///     <c>np.strings.decode</c>), and a text holding a lone surrogate on the way in (it has no UTF-8 form: numpy's
        ///     <c>UnicodeEncodeError</c>).
        /// </summary>
        [TestMethod]
        public void TextVerbs_RefuseNumbersNullsInvalidUtf8AndLoneSurrogates()
        {
            using NDArray numbers = np.arange(3);
            new Action(() => numbers.ToStringTensor()).Should().Throw<ArgumentException>()
                .WithMessage("ToStringTensor copies a text array (U, S or StringDType); this is a NumSharp Int64 array, which ToTensor<Int64>() copies.*");
            new Action(() => ((NDArray)null).ToStringTensor()).Should().Throw<ArgumentNullException>();
            new Action(() => ((Tensor<string>)null).ToNDArray()).Should().Throw<ArgumentNullException>();

            using NDArray notUtf8 = np.array(new object[] { new byte[] { 0x61 }, new byte[] { 0xFF } }, np.dtype("S"));
            new Action(() => notUtf8.ToStringTensor()).Should().Throw<UnicodeDecodeError>()
                .WithMessage("*'utf-8' codec can't decode byte 0xff in position 0: invalid start byte*");

            Tensor<string> surrogate = Tensor.Create(new[] { "a", ((char)0xD800).ToString() }, new nint[] { 2 });
            new Action(() => surrogate.ToNDArray()).Should().Throw<UnicodeEncodeError>()
                .WithMessage("*surrogates not allowed*");
        }
    }
}
