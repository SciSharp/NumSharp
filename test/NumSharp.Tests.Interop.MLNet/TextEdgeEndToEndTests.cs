using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The edges of text crossing between NumSharp and ML.NET, end to end through real ML.NET pipelines: empty and missing
    ///     texts as keys, a NumSharp <c>StringDType</c> NA on the way in, vector-of-text columns (ragged and fixed) on the way
    ///     out, and a CSV whose numbers cross the bridge while its text takes the route the refusal names — each held to NumPy
    ///     (numpy's <c>np.unique</c> / <c>np.char.split</c> / <c>np.array</c> / <c>np.loadtxt</c> over the same data, run
    ///     through NumSharp's ports and checked against numpy 2.4.2's recorded answers).
    /// </summary>
    /// <remarks>
    ///     ML.NET reads a <c>null</c> string as an EMPTY text and treats the empty text as the MISSING value of a text column:
    ///     <c>MapValueToKey</c> gives it key 0 (and <c>MapKeyToValue</c> gives back <c>""</c>), numbering only the non-empty
    ///     texts. numpy has no such notion (<c>np.unique</c> counts <c>''</c> as the smallest value), so the twin below is
    ///     numpy's unique inverse over the NON-EMPTY texts, with 0 at the empty and missing ones.
    /// </remarks>
    [TestClass]
    public class TextEdgeEndToEndTests : MLNetTestBase
    {
        /// <summary>One text column of the POCO loader.</summary>
        private sealed class TextRow
        {
            /// <summary>The text (a <c>null</c> reads as an empty text in ML.NET).</summary>
            public string T { get; set; }
        }

        /// <summary>Two text columns of the POCO loader, concatenated into a fixed-size text vector.</summary>
        private sealed class PairRow
        {
            /// <summary>The first text.</summary>
            public string A { get; set; }

            /// <summary>The second text.</summary>
            public string B { get; set; }
        }

        /// <summary>
        ///     The keys numpy's twin assigns: 0 for an empty or missing text, otherwise <c>np.unique</c>'s inverse over the
        ///     non-empty texts + 1 (by value) or the rank of the value's first occurrence + 1 (by occurrence).
        /// </summary>
        /// <param name="texts">The texts as ML.NET receives them (<c>null</c> for a NumSharp NA).</param>
        /// <param name="byValue">True for <see cref="ValueToKeyMappingEstimator.KeyOrdinality.ByValue"/>, false for by occurrence.</param>
        /// <returns>The expected key per text.</returns>
        private static uint[] NumpyKeys(string[] texts, bool byValue)
        {
            string[] present = texts.Where(t => !string.IsNullOrEmpty(t)).ToArray();
            using NDArray values = np.array(present);
            var unique = np.unique(values, return_index: true, return_inverse: true);
            long[] inverse = unique.inverse_indices.ravel().ToArray<long>();
            long[] firsts = unique.indices.ToArray<long>();

            // By occurrence, a unique value's key is the rank of its first index among all first indices.
            long[] rankOfUnique = new long[firsts.Length];
            int[] order = Enumerable.Range(0, firsts.Length).OrderBy(i => firsts[i]).ToArray();
            for (int r = 0; r < order.Length; r++)
                rankOfUnique[order[r]] = r;

            var keys = new uint[texts.Length];
            int next = 0;
            for (int i = 0; i < texts.Length; i++)
            {
                if (string.IsNullOrEmpty(texts[i]))
                    continue;   // the missing key 0
                long u = inverse[next++];
                keys[i] = (uint)((byValue ? u : rankOfUnique[u]) + 1);
            }

            return keys;
        }

        /// <summary>
        ///     <c>MapValueToKey</c> over texts holding empty strings, a <c>null</c> and a NumSharp <c>StringDType(na_object=None)</c>
        ///     NA (which <c>NDArray.AsStringArray</c> hands over as <c>null</c>): the empty and missing texts take key 0, the
        ///     others numpy's unique inverse over the non-empty texts + 1 (by value) or their first-occurrence rank + 1 (by
        ///     occurrence); the keys cross back as <c>UInt32</c>; and <c>MapKeyToValue</c> gives back each text, <c>""</c> for
        ///     the empty and missing ones.
        /// </summary>
        [TestMethod]
        public void MapValueToKey_EmptyAndMissingTexts_AreKeyZero_TheRestAreNumpysUniqueInverse()
        {
            using NDArray withNa = np.array(new object[] { "b", "", "a", null, "b", "", "é", "a", "zz" }, np.dtypes.StringDType((object)null));
            string[] texts = NDArray.AsStringArray(withNa);
            Assert.IsNull(texts[3], "a StringDType(na_object=None) NA reads as null through AsStringArray");

            IDataView view = Ml.Data.LoadFromEnumerable(texts.Select(t => new TextRow { T = t }));
            CollectionAssert.AreEqual(texts.Select(t => t ?? "").ToArray(), view.GetColumn<string>("T").ToArray(), "ML.NET reads null as an empty text");
            foreach (var ordinality in new[] { ValueToKeyMappingEstimator.KeyOrdinality.ByValue, ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence })
            {
                IDataView keyed = Ml.Transforms.Conversion.MapValueToKey("K", "T", keyOrdinality: ordinality).Fit(view).Transform(view);
                using NDArray keys = keyed.ToNDArray("K");
                Assert.AreEqual(NPTypeCode.UInt32, keys.typecode, $"{ordinality}");
                CollectionAssert.AreEqual(NumpyKeys(texts, ordinality == ValueToKeyMappingEstimator.KeyOrdinality.ByValue), keys.ToArray<uint>(), $"{ordinality}");

                IDataView back = Ml.Transforms.Conversion.MapKeyToValue("V", "K").Fit(keyed).Transform(keyed);
                CollectionAssert.AreEqual(texts.Select(t => t ?? "").ToArray(), back.GetColumn<string>("V").ToArray(), $"{ordinality}: key 0 maps back to the empty text");
            }
        }

        /// <summary>
        ///     A <c>StringDType(na_object=nan)</c> NA cannot take the <c>AsStringArray</c> route: it is refused with
        ///     <see cref="InvalidCastException"/>, naming the element and <c>tolist()</c>. Through <c>tolist()</c> the NA comes back
        ///     as the NaN object; mapped to <c>null</c> it reaches ML.NET as the missing key 0, the texts beside it keyed as numpy's
        ///     twin keys them.
        /// </summary>
        [TestMethod]
        public void StringDTypeNanNA_IsRefusedByAsStringArray_AndTolistIsTheRoute()
        {
            using NDArray nanNa = np.array(new object[] { "b", double.NaN, "", "a" }, np.dtypes.StringDType(double.NaN));
            InvalidCastException refusal = Assert.ThrowsException<InvalidCastException>(() => NDArray.AsStringArray(nanNa));
            Assert.AreEqual("Element 1 of the StringDType(na_object=nan) array is the NA object nan, which is not a string; read it with tolist().",
                refusal.Message);

            object[] items = ((IEnumerable)nanNa.tolist()).Cast<object>().ToArray();
            Assert.IsTrue(items[1] is double d && double.IsNaN(d), "tolist() returns the NA object itself");
            string[] texts = items.Select(o => o as string).ToArray();
            IDataView view = Ml.Data.LoadFromEnumerable(texts.Select(t => new TextRow { T = t }));
            // ML.NET's default ordinality is BY OCCURRENCE; by value is the order numpy's np.unique sorts in.
            IDataView keyed = Ml.Transforms.Conversion.MapValueToKey("K", "T", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Fit(view).Transform(view);
            using NDArray keys = keyed.ToNDArray("K");
            CollectionAssert.AreEqual(NumpyKeys(texts, byValue: true), keys.ToArray<uint>());
            CollectionAssert.AreEqual(new uint[] { 2, 0, 0, 1 }, keys.ToArray<uint>(), "b → 2, the NA and '' → 0, a → 1");
        }

        /// <summary>
        ///     ML.NET's <c>TokenizeIntoWords</c> makes a VARIABLE-length vector-of-text column, whose rows are numpy's
        ///     <c>np.char.split(x).tolist()</c> for space-separated texts (a run of spaces is one separator, an empty text
        ///     no word). The bridge refuses the column, naming the route <c>np.array(view.GetColumn&lt;string[]&gt;("W").ToArray())</c>;
        ///     over ragged rows that route is refused as numpy refuses <c>np.array</c> of ragged lists (<see cref="ValueError"/>,
        ///     numpy's text), and over rows of one length — tokens of equal count, or a fixed-size vector from
        ///     <c>Concatenate</c> — it is numpy's <c>(rows, width)</c> <c>U</c> array, the width the longest text.
        /// </summary>
        [TestMethod]
        public void VectorOfText_TokenRowsAreNumpysCharSplit_RaggedRefusedAsNumpy_FixedIsNumpysUArray()
        {
            string[] sentences = { "the cat sat", "a  dog", "", "one" };
            IDataView view = Ml.Data.LoadFromEnumerable(sentences.Select(t => new TextRow { T = t }));
            IDataView words = Ml.Transforms.Text.TokenizeIntoWords("W", "T").Fit(view).Transform(view);
            string[][] rows = words.GetColumn<string[]>("W").ToArray();

            using NDArray x = np.array(sentences);
            using NDListArray split = np.@char.split(x);
            string[][] numpyRows = ((IEnumerable)split.tolist()).Cast<IEnumerable>().Select(r => r.Cast<string>().ToArray()).ToArray();
            Assert.AreEqual(numpyRows.Length, rows.Length);
            for (int i = 0; i < rows.Length; i++)
                CollectionAssert.AreEqual(numpyRows[i], rows[i], $"row {i} ('{sentences[i]}') is np.char.split's");

            NotSupportedException refusal = Assert.ThrowsException<NotSupportedException>(() => words.ToNDArray("W"));
            StringAssert.Contains(refusal.Message, "np.array(view.GetColumn<string[]>(\"W\").ToArray())", "the refusal names the route");
            ValueError ragged = Assert.ThrowsException<ValueError>(() => np.array(rows));
            Assert.AreEqual("setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. " +
                            "The detected shape was (4,) + inhomogeneous part.", ragged.Message, "numpy's np.array of ragged lists");

            IDataView even = Ml.Data.LoadFromEnumerable(new[] { "ab c", "d efg" }.Select(t => new TextRow { T = t }));
            string[][] evenRows = Ml.Transforms.Text.TokenizeIntoWords("W", "T").Fit(even).Transform(even).GetColumn<string[]>("W").ToArray();
            using NDArray grid = np.array(evenRows);
            Assert.AreEqual("<U3", grid.dtype.str, "np.array([['ab', 'c'], ['d', 'efg']])'s width");
            CollectionAssert.AreEqual(new long[] { 2, 2 }, grid.shape);
            CollectionAssert.AreEqual(new[] { "ab", "c", "d", "efg" }, NDArray.AsStringArray(grid));

            IDataView pairs = Ml.Data.LoadFromEnumerable(new[] { new PairRow { A = "x", B = "yy" }, new PairRow { A = "", B = "z" } });
            IDataView joined = Ml.Transforms.Concatenate("AB", "A", "B").Fit(pairs).Transform(pairs);
            Assert.ThrowsException<NotSupportedException>(() => joined.ToNDArray("AB"), "a fixed-size text vector is refused too");
            using NDArray fixedGrid = np.array(joined.GetColumn<string[]>("AB").ToArray());
            Assert.AreEqual("<U2", fixedGrid.dtype.str, "np.array([['x', 'yy'], ['', 'z']])'s width");
            CollectionAssert.AreEqual(new[] { "x", "yy", "", "z" }, NDArray.AsStringArray(fixedGrid));
        }

        /// <summary>
        ///     One CSV read two ways: ML.NET's <c>TextLoader</c> (quoting on) with the bridge importing its numeric columns, and
        ///     NumSharp's <c>np.loadtxt</c> (numpy's reader, <c>quotechar='"'</c>). The float32 column is the same bits both ways
        ///     and numpy 2.4.2's (<c>1.5</c>, <c>-0.0</c> with its sign, NaN, the smallest subnormal <c>1e-45</c>), the int32
        ///     column the same values (<c>int.MaxValue</c> included), the boolean column ML.NET's parse of <c>true</c> /
        ///     <c>false</c>; the bridge refuses the quoted text column, and the route it names reads numpy's <c>&lt;U8</c> array
        ///     (a quoted comma kept, an empty field empty, a doubled quote one quote, a non-ASCII letter whole).
        /// </summary>
        [TestMethod]
        public void TextLoaderCsv_NumbersCrossTheBridgeAsNpLoadtxtReadsThem_TextByTheRoute()
        {
            string csv = Path.Combine(Path.GetTempPath(), $"nsml_text_{Guid.NewGuid():N}.csv");
            File.WriteAllText(csv, "label,name,score,count\n" +
                                   "true,\"Smith, J\",1.5,3\n" +
                                   "false,,-0.0,0\n" +
                                   "true,\"é \"\"q\"\"\",NaN,-7\n" +
                                   "false,plain,1e-45,2147483647\n");
            try
            {
                var loader = Ml.Data.CreateTextLoader(new TextLoader.Options
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    AllowQuoting = true,
                    Columns = new[]
                    {
                        new TextLoader.Column("label", DataKind.Boolean, 0), new TextLoader.Column("name", DataKind.String, 1),
                        new TextLoader.Column("score", DataKind.Single, 2), new TextLoader.Column("count", DataKind.Int32, 3),
                    },
                });
                IDataView loaded = loader.Load(csv);

                using NDArray score = loaded.ToNDArray("score");
                using NDArray npScore = np.loadtxt(csv, dtype: np.float32, delimiter: ",", skiprows: 1, usecols: new[] { 2 }, quotechar: "\"");
                Assert.AreEqual(NPTypeCode.Single, score.typecode);
                float[] viaBridge = score.ToArray<float>(), viaLoadtxt = npScore.ToArray<float>();
                int[] numpyBits = { 0x3FC00000, unchecked((int)0x80000000), 0x7FC00000, 0x00000001 };   // numpy 2.4.2's tobytes()
                for (int i = 0; i < numpyBits.Length; i++)
                {
                    if (i == 2)
                    {
                        // A NaN's sign is the parser's (.NET parses NaN negative, numpy positive): NaN-ness is the value.
                        Assert.IsTrue(float.IsNaN(viaBridge[i]) && float.IsNaN(viaLoadtxt[i]), "row 2 is NaN both ways");
                        continue;
                    }

                    Assert.AreEqual(numpyBits[i], BitConverter.SingleToInt32Bits(viaBridge[i]), $"row {i}: ML.NET + the bridge");
                    Assert.AreEqual(numpyBits[i], BitConverter.SingleToInt32Bits(viaLoadtxt[i]), $"row {i}: np.loadtxt");
                }

                using NDArray count = loaded.ToNDArray("count");
                using NDArray npCount = np.loadtxt(csv, dtype: np.int32, delimiter: ",", skiprows: 1, usecols: new[] { 3 }, quotechar: "\"");
                Assert.AreEqual(NPTypeCode.Int32, count.typecode);
                CollectionAssert.AreEqual(new[] { 3, 0, -7, int.MaxValue }, count.ToArray<int>());
                CollectionAssert.AreEqual(npCount.ToArray<int>(), count.ToArray<int>(), "np.loadtxt's int32 column");

                using NDArray label = loaded.ToNDArray("label");
                Assert.AreEqual(NPTypeCode.Boolean, label.typecode);
                CollectionAssert.AreEqual(new[] { true, false, true, false }, label.ToArray<bool>());

                NotSupportedException refusal = Assert.ThrowsException<NotSupportedException>(() => loaded.ToNDArray("name"));
                StringAssert.Contains(refusal.Message, "np.array(view.GetColumn<string>(\"name\").ToArray())", "the refusal names the route");
                string[] numpyNames = { "Smith, J", "", "é \"q\"", "plain" };
                using NDArray names = np.array(loaded.GetColumn<string>("name").ToArray());
                using NDArray npNames = np.loadtxt(csv, dtype: np.dtype("U"), delimiter: ",", skiprows: 1, usecols: new[] { 1 }, quotechar: "\"");
                Assert.AreEqual("<U8", names.dtype.str, "numpy's width: 'Smith, J' is eight characters");
                Assert.AreEqual(names.dtype.str, npNames.dtype.str);
                CollectionAssert.AreEqual(numpyNames, NDArray.AsStringArray(names), "the route over ML.NET's quoted-CSV parse");
                CollectionAssert.AreEqual(numpyNames, NDArray.AsStringArray(npNames), "np.loadtxt's quoted-CSV parse");
            }
            finally
            {
                File.Delete(csv);
            }
        }
    }
}
