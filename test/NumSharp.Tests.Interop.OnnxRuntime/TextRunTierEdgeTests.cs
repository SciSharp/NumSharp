using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The Run tier's edges with TEXT: a text input feeding a numeric output (LabelEncoder) through every run form —
    ///     <c>Run</c>, <c>RunInto</c> with a pre-allocated output (the right dtype, a wrong dtype, a wrong shape), and
    ///     <c>RunAsync</c> / <c>RunIntoAsync</c> on ONNX Runtime 1.30 and newer — the declared-rank validation of a text
    ///     input, StringNormalizer's stopword removal (fewer outputs than inputs, <c>['']</c> when everything is removed,
    ///     ORT's own refusal of an empty tensor), and eight threads running string models at once, every result held to its
    ///     NumSharp (numpy-verified) twin.
    /// </summary>
    [TestClass]
    public class TextRunTierEdgeTests : OnnxTestBase
    {
        /// <summary>LabelEncoder's categories in the committed model: cat / dog / bird → 0 / 1 / 2, anything else -1.</summary>
        private static readonly Dictionary<string, long> Categories = new() { ["cat"] = 0, ["dog"] = 1, ["bird"] = 2 };

        /// <summary>The ids LabelEncoder gives the texts of an array, in C order (the model's own mapping, computed in C#).</summary>
        /// <param name="text">A text array (an <c>S</c> array read as its decoded strings).</param>
        /// <returns>The ids.</returns>
        private static long[] ExpectedIds(NDArray text)
        {
            string[] strings = text.dtype.kind == 'S' ? NDArray.AsStringArray(text.astype(np.dtype("U8"))) : NDArray.AsStringArray(text);
            return strings.Select(s => Categories.TryGetValue(s, out long id) ? id : -1L).ToArray();
        }

        /// <summary>
        ///     The text inputs of the LabelEncoder cases: <c>U</c> in C order, transposed, reversed, broadcast and empty, and
        ///     <c>S</c> bytes.
        /// </summary>
        /// <returns>(label, array) pairs.</returns>
        private static IEnumerable<(string Name, NDArray Text)> CategoryInputs()
        {
            NDArray u = np.array(new[] { "cat", "fish", "dog", "bird", "", "Cat" }).reshape(2, 3);
            yield return ("U C", u);
            yield return ("U transposed", u.T);
            yield return ("U reversed", u["::-1, ::-1"]);
            yield return ("U broadcast", np.broadcast_to(u["0:1"], new Shape(3, 3)));
            yield return ("U empty", u["0:0"]);
            yield return ("S C", np.array(new[] { "dog", "bird", "x" }, np.dtype("S")));
        }

        /// <summary>
        ///     <c>RunInto</c> with a TEXT input and a pre-allocated NUMERIC output: the int64 output receives the ids for
        ///     every input layout (the pre-allocated array itself, not a copy); an int32 output is refused before the run with
        ///     the output's name in the message, and an output of the wrong shape — which the bridge cannot know before the
        ///     run, since the model declares no shape for <c>Y</c> — is ONNX Runtime's own refusal during the run.
        /// </summary>
        [TestMethod]
        public void RunInto_TextInput_FillsAPreallocatedNumericOutput()
        {
            using InferenceSession session = OpenSession("string_label_encoder");
            foreach (var (name, text) in CategoryInputs())
            {
                using NDArray y = np.full(new Shape(text.shape), -99L, np.dtype("int64"));
                session.RunInto(new Dictionary<string, NDArray> { ["X"] = text }, new Dictionary<string, NDArray> { ["Y"] = y });
                y.ToArray<long>().Should().Equal(ExpectedIds(text), name);
            }

            using NDArray input = np.array(new[] { "cat", "dog" });
            using NDArray int32Out = np.zeros(new Shape(2), np.dtype("int32"));
            new Action(() => session.RunInto(new Dictionary<string, NDArray> { ["X"] = input }, new Dictionary<string, NDArray> { ["Y"] = int32Out }))
                .Should().Throw<Exception>().WithMessage("*'Y'*");
            // The model declares no shape for Y, so the bridge cannot validate an output of the wrong shape before the run:
            // ONNX Runtime refuses it while executing the node, and nothing stays leased.
            using NDArray wrongShape = np.zeros(new Shape(3), np.dtype("int64"));
            new Action(() => session.RunInto(new Dictionary<string, NDArray> { ["X"] = input }, new Dictionary<string, NDArray> { ["Y"] = wrongShape }))
                .Should().Throw<OnnxRuntimeException>().WithMessage("*LabelEncoder*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     <c>Run</c> with a text input and a numeric output gives the ids as an owning int64 array of the input's shape
        ///     for every layout, and the dictionary form gives the same.
        /// </summary>
        [TestMethod]
        public void Run_TextInput_GivesTheNumericOutput()
        {
            using InferenceSession session = OpenSession("string_label_encoder");
            foreach (var (name, text) in CategoryInputs())
            {
                using NDArray y = session.Run(text);
                y.typecode.Should().Be(NPTypeCode.Int64, name);
                y.shape.Should().Equal(text.shape, name);
                y.ToArray<long>().Should().Equal(ExpectedIds(text), name);
                var outputs = session.Run(new Dictionary<string, NDArray> { ["X"] = text });
                using NDArray fromDictionary = outputs["Y"];
                fromDictionary.ToArray<long>().Should().Equal(ExpectedIds(text), $"{name}: the dictionary form");
            }
        }

        /// <summary>The session options of an asynchronous run: two intra-op threads, the least ORT accepts for one.</summary>
        private static readonly Lazy<SessionOptions> TwoThreads = new(() => new SessionOptions { IntraOpNumThreads = 2, InterOpNumThreads = 1 });

        /// <summary>
        ///     <c>RunAsync</c> and <c>RunIntoAsync</c> with a text INPUT and a numeric output (ONNX Runtime 1.30 and newer;
        ///     Inconclusive on the 1.16.0 floor, whose refusal <c>SessionRunAsyncTests</c> holds). Over the fixed-shape
        ///     LabelEncoder (<c>[2, 3]</c> → <c>[2, 3]</c>) both forms give the synchronous ids for every 2 × 3 layout (C,
        ///     reversed, broadcast, <c>S</c>); over the rank-agnostic one, <c>RunAsync</c> refuses — it allocates the outputs
        ///     before the run and <c>Y</c> declares no dimensions — while <c>RunIntoAsync</c> fills a pre-allocated output.
        /// </summary>
        [TestMethod]
        public async Task RunAsync_TextInput_GivesTheSynchronousIds()
        {
            if (!InferenceSessionExtensions.IsRunAsyncSupported)
                Assert.Inconclusive($"ONNX Runtime {OrtEnv.Instance().GetVersionString()}: the asynchronous run needs 1.30.0 or newer.");

            using NDArray grid = np.array(new[] { "cat", "fish", "dog", "bird", "", "Cat" }).reshape(2, 3);
            var inputs = new (string Name, NDArray Text)[]
            {
                ("U C", grid), ("U reversed", grid["::-1, ::-1"]), ("U broadcast", np.broadcast_to(grid["0:1"], new Shape(2, 3))),
                ("S C", np.array(new[] { "dog", "bird", "x", "cat", "cat", "" }, np.dtype("S")).reshape(2, 3)),
            };
            using (var fixedSession = new InferenceSession(Model("string_label_encoder_fixed"), TwoThreads.Value))
            {
                foreach (var (name, text) in inputs)
                {
                    IReadOnlyDictionary<string, NDArray> outputs = await fixedSession.RunAsync(new Dictionary<string, NDArray> { ["X"] = text });
                    using (NDArray y = outputs["Y"])
                        y.ToArray<long>().Should().Equal(ExpectedIds(text), $"{name}: RunAsync");

                    using NDArray into = np.full(new Shape(2, 3), -99L, np.dtype("int64"));
                    await fixedSession.RunIntoAsync(new Dictionary<string, NDArray> { ["X"] = text }, new Dictionary<string, NDArray> { ["Y"] = into });
                    into.ToArray<long>().Should().Equal(ExpectedIds(text), $"{name}: RunIntoAsync");
                }
            }

            using var anyRank = new InferenceSession(Model("string_label_encoder"), TwoThreads.Value);
            Func<Task> allocateBefore = () => anyRank.RunAsync(new Dictionary<string, NDArray> { ["X"] = grid });
            (await allocateBefore.Should().ThrowAsync<NotSupportedException>()).WithMessage("RunAsync: output 'Y' is declared without dimensions*");
            using NDArray preallocated = np.full(new Shape(2, 3), -99L, np.dtype("int64"));
            await anyRank.RunIntoAsync(new Dictionary<string, NDArray> { ["X"] = grid }, new Dictionary<string, NDArray> { ["Y"] = preallocated });
            preallocated.ToArray<long>().Should().Equal(ExpectedIds(grid), "RunIntoAsync over the rank-agnostic model");
        }

        /// <summary>
        ///     A text input's declared rank is validated like a numeric one's: a 2-D text array into StringNormalizer's 1-D
        ///     input is refused by the bridge before the run, naming the input and its declared shape.
        /// </summary>
        [TestMethod]
        public void TextInput_DeclaredRank_IsValidated()
        {
            using InferenceSession session = OpenSession("string_normalizer");
            using NDArray grid = np.array(new[] { "a", "b", "c", "d" }).reshape(2, 2);
            new Action(() => session.Run(grid)).Should().Throw<Exception>().WithMessage("*'X'*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     StringNormalizer's stopword removal (LOWER, case-insensitive stopwords <c>the</c> / <c>A</c>): the output has
        ///     fewer elements than the input (<c>['The', 'Cat', 'a', 'MAT']</c> gives <c>['cat', 'mat']</c>), every removal in
        ///     any case leaves ONE empty string (<c>['THE', 'the', 'The']</c> gives <c>['']</c>, as Python's onnxruntime
        ///     1.29 does), a text that only contains a stopword is kept, and an EMPTY tensor is ORT's own refusal ("Single
        ///     dimension value must be greater than 0").
        /// </summary>
        [TestMethod]
        public void StringNormalizer_Stopwords_RemoveElements_AndAllRemovedIsOneEmptyString()
        {
            using InferenceSession session = OpenSession("string_normalizer_stop");
            var cases = new (string[] Input, string[] Expected)[]
            {
                (new[] { "The", "Cat", "a", "MAT" }, new[] { "cat", "mat" }),
                (new[] { "THE", "the", "The" }, new[] { "" }),
                (new[] { "x" }, new[] { "x" }),
                (new[] { "A b", "a" }, new[] { "a b" }),
            };
            foreach (var (input, expected) in cases)
            {
                using NDArray text = np.array(input);
                using NDArray y = session.Run(text);
                y.shape.Should().Equal(new long[] { expected.Length }, string.Join(",", input));
                NDArray.AsStringArray(y).Should().Equal(expected, string.Join(",", input));
                using NDArray bytes = np.array(input, np.dtype("S"));
                using NDArray fromBytes = session.Run(bytes);
                NDArray.AsStringArray(fromBytes).Should().Equal(expected, $"{string.Join(",", input)} as S");
            }

            using NDArray empty = np.array(Array.Empty<string>());
            new Action(() => session.Run(empty)).Should().Throw<OnnxRuntimeException>().WithMessage("*Single dimension value must be greater than 0*");
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }

        /// <summary>
        ///     Eight threads, each running string models (Concat and Unique) 25 times over its OWN texts at once, on one shared
        ///     session per model: every result equals its NumSharp twin (no text is read from another thread's array, no
        ///     string tensor is shared), and nothing is leased afterwards.
        /// </summary>
        [TestMethod]
        public void ConcurrentStringRuns_EightThreads_EqualTheirTwins()
        {
            using InferenceSession concat = OpenSession("string_concat");
            using InferenceSession unique = OpenSession("string_unique");
            var failures = new ConcurrentBag<string>();
            Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, thread =>
            {
                for (int iteration = 0; iteration < 25; iteration++)
                {
                    string[] texts = Enumerable.Range(0, 12).Select(i => $"t{thread}-{(i * 7 + iteration) % 5}-{new string('x', i % 4)}").ToArray();
                    using NDArray a = np.array(texts).reshape(3, 4);
                    using NDArray b = a["::-1"];
                    var concatenated = concat.Run(new Dictionary<string, NDArray> { ["A"] = a, ["B"] = b });
                    using (NDArray c = concatenated["C"])
                    using (NDArray twin = np.concatenate((a, b)))
                    {
                        if (!NDArray.AsStringArray(c).SequenceEqual(NDArray.AsStringArray(twin)))
                            failures.Add($"thread {thread} iteration {iteration}: Concat");
                    }

                    var uniques = unique.Run(new Dictionary<string, NDArray> { ["X"] = a });
                    using (NDArray y = uniques["Y"])
                    using (uniques["indices"])
                    using (uniques["inverse_indices"])
                    using (NDArray counts = uniques["counts"])
                    {
                        var expected = np.unique(a, return_counts: true);
                        if (!NDArray.AsStringArray(y).SequenceEqual(NDArray.AsStringArray(expected.values)) ||
                            !counts.ToArray<long>().SequenceEqual(expected.counts.ToArray<long>()))
                            failures.Add($"thread {thread} iteration {iteration}: Unique");
                    }
                }
            });
            failures.Should().BeEmpty();
            NDArrayOnnxInterop.LiveExports.Should().Be(0);
        }
    }
}
