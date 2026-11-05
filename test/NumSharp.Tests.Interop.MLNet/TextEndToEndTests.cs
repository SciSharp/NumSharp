using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Text and the ML.NET bridge END TO END. The bridge maps numbers and booleans only, so every NumSharp text array —
    ///     fixed-width <c>U</c> / <c>S</c> of any width, <c>StringDType</c> with or without an NA object, in every layout —
    ///     must be refused by every export verb with the bridge's text message (before anything is read or leased), and
    ///     the two routes that message names must work end to end through real ML.NET text pipelines: NumSharp text into
    ///     ML.NET through <c>NDArray.AsStringArray</c>, and an ML.NET text column into NumSharp through
    ///     <c>GetColumn&lt;string&gt;</c> → <c>np.array</c>.
    /// </summary>
    /// <remarks>
    ///     NumPy is the reference where an answer has a NumPy form: the keys ML.NET's <c>MapValueToKey</c> assigns in value
    ///     order are <c>np.unique(text, return_inverse=True)</c>'s inverse plus one (ML.NET reserves key 0 for a missing
    ///     value), in first-occurrence order the rank of each value's first index; the <c>U</c> array the import route
    ///     builds has <c>np.array(list)</c>'s width (the longest text in code points, at least 1). The texts hold BMP
    ///     characters only where ML.NET orders them, since ML.NET compares UTF-16 code units (an astral character would sort
    ///     before U+E000..U+FFFF there and after it in numpy's code-point order).
    /// </remarks>
    [TestClass]
    public class TextEndToEndTests : MLNetTestBase
    {
        /// <summary>One text column of the POCO loader.</summary>
        private sealed class TextRow
        {
            /// <summary>The text.</summary>
            public string T { get; set; }
        }

        /// <summary>
        ///     The text arrays every verb refuses: <c>U</c> of widths 1 / 5 / 100, <c>S</c> of widths 1 / 5, <c>StringDType()</c>,
        ///     <c>StringDType(na_object=None)</c> holding an NA and <c>StringDType(na_object=nan)</c>, each in C order, F
        ///     order, transposed, reversed, stepped to the end, broadcast, 0-d, empty and 3-D.
        /// </summary>
        /// <returns>(label, array) pairs.</returns>
        private static IEnumerable<(string Name, NDArray Array)> TextArrays()
        {
            string[] texts = { "alpha", "be", "", "été", "x", "zz" };
            var bases = new List<(string, NDArray)>
            {
                ("U1", np.array(texts).astype(np.dtype("U1"))),
                ("U5", np.array(texts).astype(np.dtype("U5"))),
                ("U100", np.array(texts).astype(np.dtype("U100"))),
                ("S1", np.array(new[] { "a", "b", "", "c", "d", "e" }, np.dtype("S1"))),
                ("S5", np.array(new[] { "alpha", "be", "", "q\0r", "x", "zz" }, np.dtype("S5"))),
                ("StringDType()", np.array(texts, np.dtypes.StringDType())),
                ("StringDType(na_object=None) + NA", np.array(new object[] { "alpha", null, "", "été", "x", "zz" }, np.dtypes.StringDType((object)null))),
                ("StringDType(na_object=nan)", np.array(texts, np.dtypes.StringDType(double.NaN))),
            };
            foreach (var (name, b) in bases)
            {
                NDArray grid = b.reshape(2, 3);
                yield return ($"{name} C 1-D", b);
                yield return ($"{name} C 2-D", grid);
                yield return ($"{name} F 2-D", np.asfortranarray(grid));
                yield return ($"{name} transposed", grid.T);
                yield return ($"{name} reversed", b["::-1"]);
                yield return ($"{name} stepped to the end", b["1::2"]);
                yield return ($"{name} broadcast", np.broadcast_to(b["0:3"], new Shape(2, 3)));
                yield return ($"{name} 0-d", b["3"]);
                yield return ($"{name} empty", b["2:2"]);
                yield return ($"{name} 3-D", b.reshape(1, 2, 3));
            }
        }

        /// <summary>
        ///     Every text array (8 classes and widths × 10 layouts, 0-d / empty / 3-D included) is refused by every export
        ///     verb and the dtype maps with the bridge's TEXT message — the refusal comes first, before a shape or rank
        ///     check could answer with a numeric message — and the refusal leases nothing and leaves the array readable.
        /// </summary>
        [TestMethod]
        public void TextArrays_EveryClassAndLayout_RefusedByEveryVerb_LeasingNothing()
        {
            int exports = NDArrayMLNetInterop.LiveExports;
            int cells = 0;
            foreach (var (name, text) in TextArrays())
            {
                string[] names = text.ndim == 2 ? Enumerable.Range(0, (int)text.shape[1]).Select(i => $"c{i}").ToArray() : new[] { "c0" };
                var verbs = new (string Verb, Action Call)[]
                {
                    ("AsDataView(column)", () => text.AsDataView("Text")),
                    ("AsDataView(columns)", () => text.AsDataView(names)),
                    ("ToDataView(column)", () => text.ToDataView("Text")),
                    ("ToDataView(columns)", () => text.ToDataView(names)),
                    ("ToVBuffer<ushort>", () => text.ToVBuffer<ushort>()),
                    ("ToVBuffer<float>", () => text.ToVBuffer<float>()),
                    ("ToDataViewType", () => NDArrayMLNetInterop.ToDataViewType(text.dtype)),
                };
                foreach (var (verb, call) in verbs)
                {
                    cells++;
                    string cell = $"{name} / {verb}";
                    NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(call, cell);
                    StringAssert.Contains(ex.Message, "text arrays (fixed-width U / S and StringDType)", cell);
                    StringAssert.Contains(ex.Message, "NDArray.AsStringArray", $"{cell}: the message names the route that works");
                }

                Assert.IsFalse(NDArrayMLNetInterop.TryToDataViewType(text.dtype, out _), name);
                Assert.IsNull(NDArrayMLNetInterop.ToDataViewClrType(text.dtype), name);
                Assert.AreEqual(exports, NDArrayMLNetInterop.LiveExports, $"{name}: a refusal leases nothing");
                Assert.AreEqual(text.size, NDArray.AsStringArray(text.dtype.kind == 'S' ? text.astype(np.dtype("U5")) : text.dtype.kind == 'T' ? StringDTypeAsU(text) : text).Length,
                    $"{name}: the array is still readable after the refusals");
            }

            Assert.AreEqual(8 * 10 * 7, cells, "every class × layout × verb ran");
        }

        /// <summary>A <c>StringDType</c> array as <c>U</c> for reading (an NA reads as its text form, <c>None</c> / <c>nan</c>).</summary>
        /// <param name="text">A <c>StringDType</c> array.</param>
        /// <returns>Its sized <c>U</c> copy.</returns>
        private static NDArray StringDTypeAsU(NDArray text)
            => np.array(Enumerable.Range(0, (int)text.size).Select(i => Convert.ToString(text.flatten().GetValue(i)) ?? "None").ToArray());

        /// <summary>
        ///     NumSharp text into an ML.NET text pipeline through the route the refusal names (<c>NDArray.AsStringArray</c>
        ///     into ML.NET's own loader), back out through the bridge: <c>MapValueToKey</c> in VALUE order assigns
        ///     <c>np.unique(text, return_inverse=True)</c>'s inverse + 1, in OCCURRENCE order the rank of each value's first
        ///     index + 1 — for a <c>U</c>, an <c>S</c> (decoded) and a <c>StringDType</c> source of any layout, and the keys
        ///     cross back as <c>UInt32</c>.
        /// </summary>
        [TestMethod]
        public void NumSharpText_IntoMapValueToKey_KeysAreNumpysUniqueInverse()
        {
            string[] texts = { "pear", "apple", "éclair", "fig", "apple", "Fig", "pear", "z", "apple", "a b", "fig", "éclair" };
            using NDArray u = np.array(texts);
            using NDArray grid = u.reshape(3, 4);
            var sources = new (string Name, NDArray Text)[]
            {
                ("U C", u), ("U transposed", grid.T), ("U reversed", u["::-1"]), ("U stepped", u["1::2"]),
                ("S C", np.array(texts.Select(t => t.Replace("é", "e")).ToArray(), np.dtype("S"))),
                ("StringDType C", u.astype(np.dtypes.StringDType())),
            };
            foreach (var (name, text) in sources)
            {
                // The route: the array's texts in C order (S decoded) into ML.NET's own POCO loader.
                string[] strings = text.dtype.kind == 'U' ? NDArray.AsStringArray(text)
                    : text.dtype.kind == 'S' ? NDArray.AsStringArray(text.astype(np.dtype("U20")))
                    : NDArray.AsStringArray(text.astype(np.dtype("U20")));
                IDataView view = Ml.Data.LoadFromEnumerable(strings.Select(s => new TextRow { T = s }));

                using NDArray flat = np.array(strings);
                var unique = np.unique(flat, return_index: true, return_inverse: true);

                foreach (var ordinality in new[] { ValueToKeyMappingEstimator.KeyOrdinality.ByValue, ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence })
                {
                    IDataView keyed = Ml.Transforms.Conversion.MapValueToKey("Key", "T", keyOrdinality: ordinality).Fit(view).Transform(view);
                    using NDArray keys = keyed.ToNDArray("Key");
                    Assert.AreEqual(NPTypeCode.UInt32, keys.typecode, $"{name} / {ordinality}");

                    // numpy's twin: by value, the unique inverse; by occurrence, the rank of each unique value's first
                    // index among the first indices (argsort of the first indices, inverted).
                    long[] inverse = unique.inverse_indices.ravel().ToArray<long>();
                    long[] expected;
                    if (ordinality == ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                        expected = inverse.Select(k => k + 1).ToArray();
                    else
                    {
                        long[] firsts = unique.indices.ToArray<long>();
                        long[] rankOfUnique = new long[firsts.Length];
                        long[] order = Enumerable.Range(0, firsts.Length).OrderBy(i => firsts[i]).Select(i => (long)i).ToArray();
                        for (int r = 0; r < order.Length; r++)
                            rankOfUnique[order[r]] = r;
                        expected = inverse.Select(k => rankOfUnique[k] + 1).ToArray();
                    }

                    CollectionAssert.AreEqual(expected.Select(k => (uint)k).ToArray(), keys.ToArray<uint>(), $"{name} / {ordinality}");
                }
            }
        }

        /// <summary>
        ///     An ML.NET text column into NumSharp through the route the import refusal names
        ///     (<c>np.array(view.GetColumn&lt;string&gt;("col").ToArray())</c>): the <c>U</c> array has numpy's
        ///     <c>np.array(list)</c> width — the longest text in CODE POINTS (an astral character is one), at least 1 for an
        ///     all-empty column — and NumSharp's text operations over it give what ML.NET's own text transform gives for the
        ///     case-mapping ML.NET and numpy agree on (ASCII and Latin-1 letters, upper case).
        /// </summary>
        [TestMethod]
        public void MLNetTextColumn_IntoNumSharp_IsNumpysArrayOfTheTexts()
        {
            var cases = new (string Name, string[] Texts, string Width)[]
            {
                ("ascii and latin-1", new[] { "hello", "wörld!", "" }, "<U6"),
                ("astral counts one", new[] { char.ConvertFromUtf32(0x1F600) + "ab", "x" }, "<U3"),
                ("all empty", new[] { "", "" }, "<U1"),
                ("embedded NUL", new[] { "a\0b", "c" }, "<U3"),
            };
            foreach (var (name, texts, width) in cases)
            {
                IDataView view = Ml.Data.LoadFromEnumerable(texts.Select(t => new TextRow { T = t }));
                Assert.ThrowsException<NotSupportedException>(() => view.ToNDArray("T"), $"{name}: a text column is refused");
                using NDArray read = np.array(view.GetColumn<string>("T").ToArray());
                Assert.AreEqual(width, read.dtype.str, $"{name}: np.array(list)'s width");
                CollectionAssert.AreEqual(texts, NDArray.AsStringArray(read), name);
                using NDArray asT = np.array(view.GetColumn<string>("T").ToArray(), np.dtypes.StringDType());
                CollectionAssert.AreEqual(texts, NDArray.AsStringArray(asT.astype(np.dtype("U10"))), $"{name}: as StringDType");
            }

            // ML.NET's own upper-casing (NormalizeText, CaseMode.Upper, keeping diacritics) against np.strings.upper over
            // the texts read through the route: the same strings for ASCII and Latin-1 letters.
            string[] words = { "hello", "wörld", "MiXeD 42", "" };
            IDataView source = Ml.Data.LoadFromEnumerable(words.Select(t => new TextRow { T = t }));
            IDataView upper = Ml.Transforms.Text.NormalizeText("U", "T", TextNormalizingEstimator.CaseMode.Upper,
                keepDiacritics: true, keepPunctuations: true, keepNumbers: true).Fit(source).Transform(source);
            using NDArray mlUpper = np.array(upper.GetColumn<string>("U").ToArray());
            using NDArray npUpper = np.strings.upper(np.array(source.GetColumn<string>("T").ToArray()));
            CollectionAssert.AreEqual(NDArray.AsStringArray(npUpper), NDArray.AsStringArray(mlUpper), "ML.NET's upper case is np.strings.upper's");
        }

        /// <summary>
        ///     A whole text pipeline whose NUMBERS cross back: NumSharp text → ML.NET (the route) → <c>FeaturizeText</c> →
        ///     the bridge's import of the float feature vectors: a <c>(rows, n)</c> float32 array, finite, with identical
        ///     rows for identical texts and different rows for different texts — the bridge reads ML.NET's vector column
        ///     exactly as ML.NET's own cursor does.
        /// </summary>
        [TestMethod]
        public void FeaturizeText_FromNumSharpText_FeaturesCrossBackAsFloat32Rows()
        {
            using NDArray text = np.array(new[] { "the cat sat", "a dog ran", "the cat sat", "été chaud" });
            IDataView view = Ml.Data.LoadFromEnumerable(NDArray.AsStringArray(text).Select(t => new TextRow { T = t }));
            IDataView featurized = Ml.Transforms.Text.FeaturizeText("F", "T").Fit(view).Transform(view);
            using NDArray features = featurized.ToNDArray("F");
            Assert.AreEqual(NPTypeCode.Single, features.typecode);
            Assert.AreEqual(2, features.ndim);
            Assert.AreEqual(4L, features.shape[0]);
            Assert.IsTrue((bool)np.all(np.isfinite(features)), "every feature is finite");
            Assert.IsTrue(np.array_equal(features[0], features[2]), "identical texts featurize identically");
            Assert.IsFalse(np.array_equal(features[0], features[1]), "different texts featurize differently");

            // The bridge's read equals ML.NET's own cursor read of the same column.
            float[][] cursorRows = featurized.GetColumn<float[]>("F").ToArray();
            for (int r = 0; r < cursorRows.Length; r++)
                CollectionAssert.AreEqual(cursorRows[r], features[r].ToArray<float>(), $"row {r}");
        }
    }
}
