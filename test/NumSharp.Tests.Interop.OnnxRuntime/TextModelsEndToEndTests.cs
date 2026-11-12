using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Text END TO END through ONNX Runtime's own string kernels: a NumSharp text array (fixed-width <c>U</c> with
    ///     accented, astral, empty and NUL-holding elements, or ASCII <c>S</c> bytes) of every layout feeds a real model —
    ///     Concat, Transpose, Gather, Reshape, Slice, Unique, Where, Expand, Tile, StringNormalizer, LabelEncoder
    ///     (<c>test/oracle/gen_onnx_models.py</c>, each self-checked there against numpy through Python's
    ///     <c>onnxruntime</c>) — and the output that reads back must be the array numpy's version of the same operation
    ///     gives: the same shape, the same strings in C order, and the width numpy's <c>np.array(list)</c> gives a
    ///     list of those strings (the longest one in code points, at least 1).
    /// </summary>
    /// <remarks>
    ///     <para>The expected array is computed by NumSharp's own <c>np.*</c> twin of each operation over the same input
    ///     (<c>np.concatenate</c>, <c>np.transpose</c>, <c>np.take</c>, <c>np.unique</c>, <c>np.where</c>,
    ///     <c>np.strings.upper</c>, …): the pythonnet tier (<c>TextOpsLiveParityTests</c>) holds every one of those to
    ///     LIVE numpy over every layout, so equality here is equality with numpy without a Python at test time. The
    ///     edge behaviours of Python's own <c>onnxruntime</c> (1.29.0 with numpy 2.4.2, probed) are pinned beside the
    ///     bridge's where the two differ: Python's numpy conversion truncates a <c>U</c> / <c>S</c> element at its first
    ///     NUL and raises on a lone surrogate or on non-UTF-8 output, where this bridge keeps the NUL and writes U+FFFD
    ///     (its documented lossy edges).</para>
    /// </remarks>
    [TestClass]
    public class TextModelsEndToEndTests : OnnxTestBase
    {
        /// <summary>
        ///     The twelve texts every 3 × 4 layout is built from: ASCII (with a repeated element, so Unique has a count of
        ///     2), the empty string, a combining sequence, an astral character, an embedded NUL, a space, and a ten-letter
        ///     element (the widest).
        /// </summary>
        /// <returns>The texts.</returns>
        private static string[] Texts() => new[]
        {
            "b", "a", "c", "a", "dd", "", "e" + (char)0x0301, char.ConvertFromUtf32(0x1F600) + "x", "a\0b", "Title Case",
            "xxxxxxxxxx", "zz",
        };

        /// <summary>The ASCII twelve the <c>S</c> (bytes) inputs and the StringNormalizer are built from.</summary>
        /// <returns>The texts.</returns>
        private static string[] AsciiTexts() => new[]
        {
            "b", "a", "c", "a", "dd", "", "Mixed", "x1 z", "q\0r", "Title Case", "xxxxxxxxxx", "zz",
        };

        /// <summary>
        ///     The inputs every model meets: a 3 × 4 <c>U</c> array in C order, F order, transposed (4 × 3), with its rows
        ///     reversed, stepped along the columns (3 × 2), broadcast from one row (stride 0, read-only), empty (0 × 4), and
        ///     the ASCII texts as <c>S</c> bytes in C order.
        /// </summary>
        /// <returns>(name, array) pairs; the caller disposes nothing (each array is a fresh, collectable object).</returns>
        private static IEnumerable<(string Name, NDArray Array)> Inputs()
        {
            NDArray grid = np.array(Texts()).reshape(3, 4);
            yield return ("U C 3x4", grid);
            yield return ("U F 3x4", np.asfortranarray(grid));
            yield return ("U transposed 4x3", grid.T);
            yield return ("U reversed rows", grid["::-1"]);
            yield return ("U stepped columns 3x2", grid[":, ::2"]);
            yield return ("U broadcast row 3x4", np.broadcast_to(grid["1:2"], new Shape(3, 4)));
            yield return ("U empty 0x4", grid["0:0"]);
            yield return ("S C 3x4", np.array(AsciiTexts(), np.dtype("S")).reshape(3, 4));
        }

        /// <summary>The number of code points in a .NET string (a surrogate pair is one, as in numpy's <c>U</c>).</summary>
        /// <param name="s">The string.</param>
        /// <returns>Its length in code points.</returns>
        private static int CodePoints(string s)
        {
            int n = 0;
            for (int i = 0; i < s.Length; i++, n++)
            {
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    i++;
            }

            return n;
        }

        /// <summary>
        ///     The text a NumSharp reference array holds, as a string per element in C order: a <c>U</c> element as its text,
        ///     an <c>S</c> element decoded as UTF-8 (what a string tensor reads its bytes as).
        /// </summary>
        /// <param name="nd">A fixed-width text array.</param>
        /// <returns>The strings.</returns>
        private static string[] Strings(NDArray nd)
        {
            if (nd.dtype.kind == 'U')
                return NDArray.AsStringArray(nd);
            var result = new string[nd.size];
            using NDArray flat = nd.flatten();
            for (int i = 0; i < result.Length; i++)
                result[i] = Encoding.UTF8.GetString((byte[])flat.GetValue(i));
            return result;
        }

        /// <summary>
        ///     Holds an ORT text output to numpy's array for the same operation: the shape, the strings in C order, and the
        ///     <c>U</c> width numpy's <c>np.array(list)</c> of those strings has (the longest in code points, at least 1).
        /// </summary>
        /// <param name="actual">The output the bridge read back.</param>
        /// <param name="expected">NumSharp's twin of the operation over the same input.</param>
        /// <param name="because">The cell label.</param>
        private static void AssertSameText(NDArray actual, NDArray expected, string because)
        {
            actual.typecode.Should().Be(NPTypeCode.String, because);
            actual.dtype.kind.Should().Be('U', $"{because}: a string tensor reads back as Unicode");
            actual.shape.Should().Equal(expected.shape, because);
            string[] strings = Strings(expected);
            NDArray.AsStringArray(actual).Should().Equal(strings, because);
            int width = Math.Max(1, strings.Length == 0 ? 0 : strings.Max(CodePoints));
            actual.dtype.Should().Be(np.dtype($"<U{width}"), $"{because}: np.array(list) of the strings is <U{width}");
        }

        /// <summary>
        ///     Concat (axis 0) of every input with its rows reversed: the strings of <c>np.concatenate((a, a[::-1]))</c>.
        /// </summary>
        [TestMethod]
        public void Concat_EveryLayout_IsNumpysConcatenate()
        {
            using InferenceSession session = OpenSession("string_concat");
            foreach (var (name, a) in Inputs())
            {
                using NDArray b = a["::-1"];
                var outputs = session.Run(new Dictionary<string, NDArray> { ["A"] = a, ["B"] = b });
                using NDArray c = outputs["C"];
                using NDArray expected = np.concatenate((a, b));
                AssertSameText(c, expected, name);
            }
        }

        /// <summary>Transpose (reversed axes) of every input: the strings of <c>a.T</c>, in its logical order.</summary>
        [TestMethod]
        public void Transpose_EveryLayout_IsNumpysTranspose()
        {
            using InferenceSession session = OpenSession("string_transpose");
            foreach (var (name, a) in Inputs())
            {
                using NDArray y = session.Run(a);
                AssertSameText(y, a.T, name);
            }
        }

        /// <summary>
        ///     Gather on axis 1 with a negative index: <c>np.take(a, [1, 0, -1], axis=1)</c> (the model gathers on axis 1:
        ///     onnxruntime 1.29's string Gather on axis 0 of a 2-D tensor copies one string per row, see the generator).
        /// </summary>
        [TestMethod]
        public void Gather_EveryLayout_IsNumpysTake()
        {
            using InferenceSession session = OpenSession("string_gather");
            using NDArray indices = np.array(new long[] { 1, 0, -1 });
            foreach (var (name, a) in Inputs())
            {
                var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = a, ["I"] = indices });
                using NDArray y = outputs["Y"];
                using NDArray expected = np.take(a, indices, axis: 1);
                AssertSameText(y, expected, name);
            }
        }

        /// <summary>Reshape to a flat vector and to two rows (a <c>-1</c> extent): <c>a.reshape(-1)</c> / <c>a.reshape(2, -1)</c>.</summary>
        [TestMethod]
        public void Reshape_EveryLayout_IsNumpysReshape()
        {
            using InferenceSession session = OpenSession("string_reshape");
            foreach (var (name, a) in Inputs())
            {
                foreach (long[] shape in new[] { new long[] { -1 }, new long[] { 2, -1 } })
                {
                    var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = a, ["S"] = np.array(shape) });
                    using NDArray y = outputs["Y"];
                    using NDArray expected = a.reshape(shape);
                    AssertSameText(y, expected, $"{name} reshape({string.Join(", ", shape)})");
                }
            }
        }

        /// <summary>
        ///     Slice with a negative step on axis 0 and a step of 2 from 1 on axis 1: <c>a[::-1, 1:4:2]</c> (ends past the
        ///     extent are clamped, as numpy's slicing clamps them).
        /// </summary>
        [TestMethod]
        public void Slice_EveryLayout_IsNumpysSlicing()
        {
            using InferenceSession session = OpenSession("string_slice");
            foreach (var (name, a) in Inputs())
            {
                var outputs = session.Run(new Dictionary<string, NDArray>
                {
                    ["X"] = a,
                    ["starts"] = np.array(new long[] { -1, 1 }),
                    ["ends"] = np.array(new long[] { -1000, 4 }),
                    ["axes"] = np.array(new long[] { 0, 1 }),
                    ["steps"] = np.array(new long[] { -1, 2 }),
                });
                using NDArray y = outputs["Y"];
                AssertSameText(y, a["::-1, 1:4:2"], name);
            }
        }

        /// <summary>
        ///     Unique (sorted) over the flattened input: the four results of
        ///     <c>np.unique(a, return_index=True, return_inverse=True, return_counts=True)</c> — ORT orders by UTF-8 bytes,
        ///     which is numpy's code-point order, astral and combining text included; the inverse is the flattened one.
        /// </summary>
        [TestMethod]
        public void Unique_EveryLayout_IsNumpysUnique()
        {
            using InferenceSession session = OpenSession("string_unique");
            foreach (var (name, a) in Inputs())
            {
                var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = a });
                using NDArray y = outputs["Y"];
                using NDArray indices = outputs["indices"];
                using NDArray inverse = outputs["inverse_indices"];
                using NDArray counts = outputs["counts"];
                var expected = np.unique(a, return_index: true, return_inverse: true, return_counts: true);
                AssertSameText(y, expected.values, name);
                indices.ToArray<long>().Should().Equal(expected.indices.ToArray<long>(), $"{name}: first occurrences");
                inverse.ToArray<long>().Should().Equal(expected.inverse_indices.ravel().ToArray<long>(), $"{name}: inverse");
                counts.ToArray<long>().Should().Equal(expected.counts.ToArray<long>(), $"{name}: counts");
            }
        }

        /// <summary>
        ///     Where with a condition computed from the text itself (<c>str_len(a[:1]) &gt; 1</c>, one row broadcast over
        ///     the rows) and a one-element text as the other branch: <c>np.where(cond, a, ['y'])</c>.
        /// </summary>
        [TestMethod]
        public void Where_ConditionFromTheText_IsNumpysWhere()
        {
            using InferenceSession session = OpenSession("string_where");
            using NDArray other = np.array(new[] { "y" });
            foreach (var (name, a) in Inputs())
            {
                // The first row (an empty input is its own first "row"); not disposed: it may be the input itself.
                NDArray firstRow = a.shape[0] > 0 ? a["0:1"] : a;
                using NDArray cond = np.strings.str_len(firstRow) > 1;
                var outputs = session.Run(new Dictionary<string, NDArray> { ["cond"] = cond, ["X"] = a, ["Y"] = other });
                using NDArray z = outputs["Z"];
                using NDArray expected = np.where(cond, a, other);
                AssertSameText(z, expected, name);
            }
        }

        /// <summary>Expand one row to 3 rows: <c>np.broadcast_to(a[0], (3, n))</c>, materialized by ORT.</summary>
        [TestMethod]
        public void Expand_OneRow_IsNumpysBroadcastTo()
        {
            using InferenceSession session = OpenSession("string_expand");
            foreach (var (name, a) in Inputs().Where(i => i.Array.shape[0] > 0))
            {
                using NDArray row = a["0"];
                var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = row, ["S"] = np.array(new long[] { 3, row.shape[0] }) });
                using NDArray y = outputs["Y"];
                using NDArray expected = np.broadcast_to(row, new Shape(3, row.shape[0]));
                AssertSameText(y, expected, name);
            }
        }

        /// <summary>Tile twice along the columns: <c>np.tile(a, (1, 2))</c>.</summary>
        [TestMethod]
        public void Tile_EveryLayout_IsNumpysTile()
        {
            using InferenceSession session = OpenSession("string_tile");
            using NDArray repeats = np.array(new long[] { 1, 2 });
            foreach (var (name, a) in Inputs())
            {
                var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = a, ["R"] = repeats });
                using NDArray y = outputs["Y"];
                using NDArray expected = np.tile(a, (1, 2));
                AssertSameText(y, expected, name);
            }
        }

        /// <summary>
        ///     StringNormalizer's UPPER over 1-D ASCII text of three layouts (C, reversed, stepped) and of both classes:
        ///     <c>np.strings.upper(x)</c>. The text is ASCII because ORT changes case through the C locale's wide
        ///     characters, which on Windows replaces a non-ASCII character with U+FFFD (Python's onnxruntime 1.29 does the
        ///     same) — ORT's behaviour, not the bridge's.
        /// </summary>
        [TestMethod]
        public void StringNormalizer_Upper_IsNumpysUpper()
        {
            using InferenceSession session = OpenSession("string_normalizer");
            using NDArray u = np.array(AsciiTexts().Where(t => !t.Contains('\0')).ToArray());
            using NDArray s = u.astype(np.dtype("S"));
            foreach (var (name, x) in new[] { ("U C", u), ("U reversed", u["::-1"]), ("U stepped", u["1::2"]), ("S C", s), ("S reversed", s["::-1"]) })
            {
                using NDArray y = session.Run(x);
                using NDArray expected = np.strings.upper(x);
                AssertSameText(y, expected, name);
            }
        }

        /// <summary>
        ///     LabelEncoder decodes text categories to their ids, so a category must arrive EXACTLY: <c>cat</c> / <c>dog</c> /
        ///     <c>bird</c> map to 0 / 1 / 2 from a <c>U</c> and an <c>S</c> array of any layout, while a near miss — a
        ///     different case, a trailing space, an embedded NUL, the empty string, an unknown word — gets the default -1.
        /// </summary>
        [TestMethod]
        public void LabelEncoder_TextCategories_ArriveExactly()
        {
            using InferenceSession session = OpenSession("string_label_encoder");
            string[] words = { "cat", "dog", "bird", "Cat", "dog ", "ca\0t", "", "fish" };
            long[] ids = { 0, 1, 2, -1, -1, -1, -1, -1 };
            using NDArray u = np.array(words).reshape(2, 4);
            using NDArray s = np.array(words, np.dtype("S")).reshape(2, 4);
            foreach (var (name, x, expected) in new[]
                     {
                         ("U C", u, ids), ("U transposed", u.T, new long[] { 0, -1, 1, -1, 2, -1, -1, -1 }),
                         ("S C", s, ids), ("U reversed", u["::-1, ::-1"], Enumerable.Reverse(ids).ToArray()),
                     })
            {
                using NDArray y = session.Run(x);
                y.typecode.Should().Be(NPTypeCode.Int64, name);
                y.shape.Should().Equal(x.shape, name);
                y.ToArray<long>().Should().Equal(expected, name);
            }
        }

        /// <summary>
        ///     Identity of an empty tensor and of an all-empty one: both read back as <c>&lt;U1</c>, which is numpy's width
        ///     for <c>np.array([], dtype=str)</c> and <c>np.array(['', ''])</c> (a fixed-width text has at least one code
        ///     point of room); a 0-d text and a 6-D one keep their shapes.
        /// </summary>
        [TestMethod]
        public void Identity_EmptyAllEmptyZeroDAndHighRank_KeepNumpysShapeAndWidth()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray empty = np.array(Array.Empty<string>());
            using NDArray emptyGrid = np.array(Array.Empty<string>()).reshape(0, 3);
            using NDArray allEmpty = np.array(new[] { "", "" });
            using NDArray zeroD = np.array("solo");
            using NDArray highRank = np.array(Enumerable.Range(0, 64).Select(i => i.ToString()).ToArray()).reshape(2, 2, 2, 2, 2, 2);
            foreach (var (name, x) in new[] { ("empty (0,)", empty), ("empty (0,3)", emptyGrid), ("all empty", allEmpty), ("0-d", zeroD), ("6-D", highRank) })
            {
                using NDArray y = session.Run(x);
                AssertSameText(y, x, name);
            }
        }

        /// <summary>
        ///     Volume and length: 20,000 strings of 0 – 40 code points (astral ones included) and one string of a million
        ///     code points cross both ways unchanged, the width being the longest string's.
        /// </summary>
        [TestMethod]
        public void Identity_ManyAndLongStrings_RoundTripExactly()
        {
            using InferenceSession session = OpenSession("string_io");
            var random = new Random(20261105);
            string astral = char.ConvertFromUtf32(0x1F600);
            string[] many = Enumerable.Range(0, 20_000).Select(i =>
            {
                var sb = new StringBuilder();
                int n = random.Next(0, 41);
                for (int k = 0; k < n; k++)
                    sb.Append(random.Next(10) == 0 ? astral : ((char)('a' + random.Next(26))).ToString());
                return sb.ToString();
            }).ToArray();
            using NDArray manyArray = np.array(many);
            using (NDArray y = session.Run(manyArray))
                AssertSameText(y, manyArray, "20,000 strings");

            using NDArray longOne = np.array(new[] { new string('L', 1_000_000), "s" });
            using (NDArray y = session.Run(longOne))
                AssertSameText(y, longOne, "one string of a million code points");
        }

        /// <summary>
        ///     An embedded NUL survives the round trip in a <c>U</c> and in an <c>S</c> element — ORT's strings carry their
        ///     length — where Python's <c>onnxruntime</c> 1.29 (numpy 2.4.2) truncates a <c>U</c> / <c>S</c> array's element at
        ///     its first NUL (<c>['a\x00b']</c> comes back <c>['a']</c>; only an object array keeps it). The bridge keeps
        ///     what numpy's array holds.
        /// </summary>
        [TestMethod]
        [TestCategory("Misaligned")]
        public void Identity_EmbeddedNul_Survives_WherePythonOrtTruncates()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray u = np.array(new[] { "a\0b", "\0lead", "mid\0\0dle" });
            using NDArray s = np.array(new[] { "a\0b", "\0lead" }, np.dtype("S"));
            using (NDArray y = session.Run(u))
                NDArray.AsStringArray(y).Should().Equal("a\0b", "\0lead", "mid\0\0dle");
            using (NDArray y = session.Run(s))
                NDArray.AsStringArray(y).Should().Equal("a\0b", "\0lead");
        }

        /// <summary>
        ///     A lone surrogate in a <c>U</c> element has no UTF-8 form: the bridge writes U+FFFD (its documented lossy
        ///     edge, <c>NDArrayOnnxInterop.TextToOrtValue</c>), where Python's <c>onnxruntime</c> raises
        ///     <c>UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 1: surrogates not
        ///     allowed</c>. Pinned so a change to either side is noticed.
        /// </summary>
        [TestMethod]
        [TestCategory("Misaligned")]
        public void Identity_LoneSurrogate_BecomesTheReplacementCharacter_WherePythonOrtRaises()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray u = np.array(new[] { "a" + (char)0xD800, "ok" });
            using NDArray y = session.Run(u);
            NDArray.AsStringArray(y).Should().Equal("a" + (char)0xFFFD, "ok");
        }

        /// <summary>
        ///     <c>S</c> bytes that are not UTF-8 enter ORT as they are (as with Python's onnxruntime) and read back with each
        ///     invalid sequence as U+FFFD (the bridge's documented decode, <c>NDArrayOnnxInterop.TextArray</c>), where
        ///     Python's onnxruntime raises <c>UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0:
        ///     invalid start byte</c> when it reads the output. Pinned so a change to either side is noticed.
        /// </summary>
        [TestMethod]
        [TestCategory("Misaligned")]
        public void Identity_InvalidUtf8Bytes_ReadBackAsReplacementCharacters_WherePythonOrtRaises()
        {
            using InferenceSession session = OpenSession("string_io");
            using NDArray s = np.array(new[] { new byte[] { 0x80, 0xFF }, new byte[] { (byte)'o', (byte)'k' } }, np.dtype("S2"));
            using NDArray y = session.Run(s);
            string[] read = NDArray.AsStringArray(y);
            read[1].Should().Be("ok");
            read[0].Should().MatchRegex("^�+$", "each invalid byte reads as the replacement character");
        }
    }
}
