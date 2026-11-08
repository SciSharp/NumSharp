using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Holds the two corpus fast paths to the parser they replaced: <see cref="CorpusFile"/> must yield exactly the
    ///     cases the old <c>File.ReadLines</c> + <c>IsNullOrWhiteSpace</c> + string-deserialize loader produced, and every
    ///     <see cref="CorpusSurvey"/> header record must equal what the coverage gates used to compute from the fully
    ///     parsed <see cref="FuzzCorpus.Case"/>. Without these, a scanner bug would not fail a coverage gate — it would
    ///     silently change what the gate believes the corpus covers.
    /// </summary>
    [TestClass]
    public class CorpusSurveyTests
    {
        /// <summary>
        ///     Real tiers that together carry every header feature the scanner reads (checked by
        ///     <see cref="RepresentativeTiers_CarryEveryHeaderFeature"/>): string/number/list/null params incl. dist,
        ///     axes of every value kind, every expected kind, truth, error objects and expects_throw, 0-d operands,
        ///     empty params, operand-less cases, the masked-array and advanced-indexing schemas, and lines written with
        ///     <c>", "</c>/<c>": "</c> separators. ~4 MB, 8,366 cases — cheap next to the eight whole-corpus parses the
        ///     survey replaced. (Verified once against the parser over the WHOLE corpus — 222,682 cases, 0 field
        ///     mismatches on net10.0 and net8.0 — when the survey was introduced; this subset keeps it honest after.)
        /// </summary>
        private static readonly string[] RepresentativeTiers =
        {
            "dtype_text.jsonl",     // error/throw, kinds dtype/scalar/text/tuple, no-operand cases, dtype param, spaces
            "random_parity.jsonl",  // params.dist (rnd), text kind
            "products.jsonl",       // expected.truth, int axes
            "fft.jsonl",            // null and list axes
            "ma_extras.jsonl",      // masked-array schema, masked kind
            "index_curated.jsonl",  // advanced-indexing schema read through the case DTO (op key only)
            "index_dtype.jsonl",    // advanced-indexing lines with no op key at all
            "errors_full.jsonl",    // raising cells with verbatim error objects
        };

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Survey_MatchesTheCaseParser_OnRepresentativeTiers()
        {
            var survey = CorpusSurvey.Files.ToDictionary(f => f.Name, StringComparer.Ordinal);
            var mismatches = new List<string>();
            int compared = 0;
            foreach (string tier in RepresentativeTiers)
            {
                Assert.IsTrue(survey.TryGetValue(tier, out var scanned), $"the survey has no '{tier}'");
                using var file = FuzzCorpus.Open(tier);
                int i = 0;
                foreach (var c in file)
                {
                    Assert.IsTrue(i < scanned.Cases.Length, $"{tier}: the survey holds fewer cases than the parser yields");
                    Compare($"{tier} case #{i + 1}", Reference(c), scanned.Cases[i], mismatches);
                    i++;
                }
                Assert.AreEqual(scanned.Cases.Length, i, $"{tier}: survey and parser disagree on the case count");
                compared += i;
            }
            // ~90 % of the 2026-09-24 count (8,366): the comparison must stay non-vacuous as the tiers regenerate.
            Assert.IsTrue(compared > 7_500, $"only {compared} cases compared — representative tiers shrank?");
            Assert.AreEqual(0, mismatches.Count, "survey vs parser:\n  " + string.Join("\n  ", mismatches.Take(40)));
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void RepresentativeTiers_CarryEveryHeaderFeature()
        {
            // The agreement test above only proves what its tiers exercise: if a regeneration stopped emitting a
            // feature in all of them, the check would keep passing while testing less. Pin the features.
            var byName = CorpusSurvey.Files.ToDictionary(f => f.Name, StringComparer.Ordinal);
            var cases = RepresentativeTiers.SelectMany(t => byName[t].Cases).ToArray();
            var missing = new List<string>();
            void Require(string feature, bool present)
            {
                if (!present)
                    missing.Add(feature);
            }
            Require("rnd dist", cases.Any(c => c.Op == "rnd" && c.Dist != null));
            Require("dtype param", cases.Any(c => c.ParamDtype != null));
            Require("axes param", cases.Any(c => c.HasAxesParam));
            Require("truth", cases.Any(c => c.HasTruth));
            Require("error", cases.Any(c => c.IsError));
            Require("0-d operand", cases.Any(c => c.HasZeroDOperand));
            Require("empty params", cases.Any(c => c.ParamSignature.Length == 0));
            Require("no operands", cases.Any(c => c.OperandDtypes.Length == 0));
            Require("no op key", cases.Any(c => c.Op == null));
            foreach (string kind in new[] { "array", "scalar", "dtype", "text", "tuple", "masked" })
                Require("expected kind " + kind, cases.Any(c => c.ExpectedKind == kind));
            Assert.AreEqual(0, missing.Count, "representative tiers no longer exercise: " + string.Join(", ", missing));
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Survey_CoversEveryNonHostCorpusFile()
        {
            string dir = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            var expected = Directory.EnumerateFiles(dir, "*.jsonl").Select(Path.GetFileName)
                .Where(n => !n.EndsWith(".host.jsonl", StringComparison.Ordinal))
                .OrderBy(n => n, StringComparer.Ordinal).ToArray();
            CollectionAssert.AreEqual(expected, CorpusSurvey.Files.Select(f => f.Name).ToArray());
            Assert.IsTrue(CorpusSurvey.Files.Sum(f => f.Cases.Length) > 200_000, "the survey lost most of the corpus");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Survey_HonorsTheParserQuirks_OnSyntheticLines()
        {
            // Every quirk the fidelity contract names, on lines no generator writes today: case-insensitive property
            // names but case-sensitive param keys, last-wins repeats, null vs {} vs absent params, raw param text with
            // spaces and escapes, non-string ufunc/dtype/dist values, null/absent/empty-kind expected, 0-d vs null
            // shapes, unknown nested properties, and CRLF / lone-CR / whitespace-only separators.
            string[] lines =
            {
                """{"op":"add","params":{"b":[1, 2],"a":{"k": "v"},"s":"x\"y","u":"\u00e9"},"operands":[{"dtype":"int32","shape":[2,3]},{"dtype":null,"shape":[]}],"expected":{"kind":"","dtype":"int64","truth":null},"layout":"C","valueclass":"small"}""",
                """{"OP":"rnd","Params":{"dist":"normal","seed":1},"Operands":null,"EXPECTED":{"KIND":"text","Truth":"00ff"}}""",
                """{"op":"x","params":null}""",
                """{"op":"x","params":{}}""",
                """{"op":"x"}""",
                """{"op":"first","op":"second","params":{"k":1,"k":2}}""",
                """{"op":"out_binary","params":{"ufunc":"add","dtype":"float32","axes":null},"valueclass":"outwhere","error":{"type":"ValueError","text":"x"},"expects_throw":true}""",
                """{"op":"y","params":{"ufunc":5,"dtype":["f4"],"dist":7},"error":null,"expects_throw":false}""",
                """   {"op":"z","operands":[{"shape":null,"dtype":"bool","buffer":"00"}]}   """,
                """{"op":"get","base":"V6","tokens":[["int",0]],"tag":"basic1","np":{"ok":true,"shape":[],"vals":[0]},"id":"get/V6/basic1/0"}""",
                """{"op":"t","expected":{"kind":"tuple","slots":[{"kind":"array","dtype":"int8"}],"dtype":"float64"}}""",
                """{"op":"w","params":{"Dist":"normal","axes":[0,1]}}""",
                """{"id":"only-id","alias":true}""",
                """{"op" : "spaced" , "params" : { "q" : 0.5 , "axis" : -1 } , "expected" : null}""",
            };
            string content = string.Join("\n", lines.Take(5)) + "\r\n\r\n   \t\n" + string.Join("\r", lines.Skip(5).Take(4))
                             + "\n\n" + string.Join("\r\n", lines.Skip(9));
            byte[] utf8 = Encoding.UTF8.GetBytes(content);

            var scanned = CorpusSurvey.ScanContent("synthetic.jsonl", utf8).Cases;
            var parsed = new List<FuzzCorpus.Case>();
            foreach (ReadOnlySpan<byte> line in new CorpusFile.LineEnumerator(utf8))
                parsed.Add(FuzzCorpus.ParseLine(line));

            Assert.AreEqual(lines.Length, parsed.Count, "the line walker split the synthetic content wrongly");
            Assert.AreEqual(parsed.Count, scanned.Length);
            var mismatches = new List<string>();
            for (int i = 0; i < parsed.Count; i++)
                Compare($"synthetic line {i + 1}", Reference(parsed[i]), scanned[i], mismatches);
            Assert.AreEqual(0, mismatches.Count, "survey vs parser:\n  " + string.Join("\n  ", mismatches));

            // Spot-check the quirks themselves, so a shared misreading on BOTH sides cannot pass unnoticed.
            Assert.AreEqual("a={\"k\": \"v\"}|b=[1, 2]|s=\"x\\\"y\"|u=\"\\u00e9\"", scanned[0].ParamSignature);
            Assert.AreEqual("array", scanned[0].ExpectedKind);
            Assert.IsTrue(scanned[0].HasZeroDOperand);
            CollectionAssert.AreEqual(new[] { "int32", null }, scanned[0].OperandDtypes);
            Assert.AreEqual("normal", scanned[1].Dist);
            Assert.IsTrue(scanned[1].HasTruth);
            Assert.AreEqual("{}", scanned[2].ParamSignature);
            Assert.AreEqual("", scanned[3].ParamSignature);
            Assert.AreEqual("{}", scanned[4].ParamSignature);
            Assert.AreEqual("second", scanned[5].Op);
            Assert.AreEqual("k=2", scanned[5].ParamSignature);
            Assert.IsTrue(scanned[6].IsError);
            Assert.AreEqual("add", scanned[6].Ufunc);
            Assert.IsNull(scanned[7].Ufunc);
            Assert.IsNull(scanned[7].ParamDtype);
            Assert.IsFalse(scanned[8].HasZeroDOperand);
            Assert.AreEqual("get", scanned[9].Op);
            Assert.AreEqual("float64", scanned[10].ExpectedDtype);
            Assert.IsNull(scanned[11].Dist);
            Assert.IsTrue(scanned[11].HasAxesParam);
            Assert.IsNull(scanned[12].Op);
            Assert.AreEqual("axis=-1|q=0.5", scanned[13].ParamSignature);
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Survey_RejectsMalformedHeaders()
        {
            // A header field of the wrong type or a line that is not one JSON object fails the scan, as it fails the
            // deserializer; so do the two shapes the deserializer accepts but every gate then crashed on — a null
            // operand and a random-stream case whose sampler is not a string.
            string[] bad =
            {
                """{"op":5}""",
                """{"op":"a","params":[1]}""",
                """{"op":"a","operands":[null]}""",
                """{"op":"a","operands":[{"shape":[1.5]}]}""",
                """{"op":"a","expected":{"kind":3}}""",
                """{"op":"a","expects_throw":null}""",
                """{"op":"a","error":"text"}""",
                """{"op":"rnd","params":{"dist":null}}""",
                """{"op":"a"} {"op":"b"}""",
                """{"op":"a" """,
                """[1,2]""",
            };
            foreach (string line in bad)
            {
                try
                {
                    CorpusSurvey.ScanContent("bad.jsonl", Encoding.UTF8.GetBytes(line));
                }
                catch (JsonException)
                {
                    // Expected — including the reader's own JsonException subtype for syntax errors.
                    continue;
                }
                Assert.Fail($"the survey accepted a malformed line: {line}");
            }
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void CorpusFile_YieldsExactlyTheCasesOfTheReadLinesLoader()
        {
            // BOM, CRLF, a lone CR, whitespace-only lines (ASCII, NBSP, U+2028) and surrounding blanks: the pooled
            // UTF-8 reader must produce the same cases, in the same order, as the string loader it replaced.
            string text = "\uFEFF{\"op\":\"a\",\"id\":\"1\"}\r\n\r\n  \t \n{\"op\":\"b\",\"id\":\"2\"}\r{\"op\":\"c\",\"id\":\"3\"}\n"
                          + "\u00A0\u00A0\n\u2028\n  {\"op\":\"d\",\"id\":\"4\"}  \n\n";
            string path = Path.Combine(Path.GetTempPath(), "ns_corpusfile_" + Guid.NewGuid().ToString("N") + ".jsonl");
            File.WriteAllBytes(path, Encoding.UTF8.GetBytes(text));
            try
            {
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                var reference = File.ReadLines(path).Where(l => !string.IsNullOrWhiteSpace(l))
                    .Select(l => JsonSerializer.Deserialize<FuzzCorpus.Case>(l, options)).ToList();

                using var file = new CorpusFile("synthetic.jsonl", path);
                var streamed = file.ToList();
                Assert.AreEqual(4, reference.Count, "reference loader sanity");
                Assert.AreEqual(reference.Count, file.Count, "Count must equal the loader's case count");
                CollectionAssert.AreEqual(reference.Select(c => c.Op + "/" + c.Id).ToArray(),
                                          streamed.Select(c => c.Op + "/" + c.Id).ToArray());

                // Enumerating after Dispose must refuse, not read a buffer the pool may have handed out again.
                file.Dispose();
                Assert.ThrowsException<ObjectDisposedException>(() => file.GetEnumerator());
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     The header record exactly as the coverage gates derived it from a parsed case before the survey existed —
        ///     the reference each scanned record must equal.
        /// </summary>
        /// <param name="c">A case parsed by <see cref="FuzzCorpus.ParseLine"/>.</param>
        /// <returns>The expected header record.</returns>
        private static SurveyCase Reference(FuzzCorpus.Case c)
        {
            var operands = c.Operands ?? Array.Empty<FuzzCorpus.Operand>();
            string StringParam(string key)
                => c.Params != null && c.Params.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.String
                    ? v.GetString()
                    : null;
            return new SurveyCase(
                c.Op, c.Layout, c.Valueclass,
                operands.Select(o => o.Dtype).ToArray(),
                string.Join(",", operands.Select(o => o.Dtype ?? "")),
                operands.Any(o => o.Shape != null && o.Shape.Length == 0),
                c.Params == null
                    ? "{}"
                    : string.Join("|", c.Params.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                                               .Select(kv => kv.Key + "=" + kv.Value.GetRawText())),
                StringParam("dist"), StringParam("ufunc"), StringParam("dtype"),
                c.Params != null && c.Params.ContainsKey("axes"),
                c.Expected?.KindOrArray ?? "array",
                c.Expected?.Dtype,
                c.Expected?.Truth != null,
                c.Error != null || c.Expects_Throw);
        }

        /// <summary>
        ///     Records every field on which <paramref name="actual"/> differs from <paramref name="expected"/>.
        /// </summary>
        /// <param name="where">The case's location, for the message.</param>
        /// <param name="expected">The reference record.</param>
        /// <param name="actual">The scanned record.</param>
        /// <param name="mismatches">Collects one line per differing field.</param>
        private static void Compare(string where, SurveyCase expected, SurveyCase actual, List<string> mismatches)
        {
            void Field<T>(string name, T want, T got)
            {
                if (!EqualityComparer<T>.Default.Equals(want, got))
                    mismatches.Add($"{where}: {name} expected <{want}> got <{got}>");
            }
            Field("Op", expected.Op, actual.Op);
            Field("Layout", expected.Layout, actual.Layout);
            Field("Valueclass", expected.Valueclass, actual.Valueclass);
            if (!expected.OperandDtypes.SequenceEqual(actual.OperandDtypes))
                mismatches.Add($"{where}: OperandDtypes expected [{string.Join(",", expected.OperandDtypes)}] " +
                               $"got [{string.Join(",", actual.OperandDtypes)}]");
            Field("DtypeSignature", expected.DtypeSignature, actual.DtypeSignature);
            Field("HasZeroDOperand", expected.HasZeroDOperand, actual.HasZeroDOperand);
            Field("ParamSignature", expected.ParamSignature, actual.ParamSignature);
            Field("Dist", expected.Dist, actual.Dist);
            Field("Ufunc", expected.Ufunc, actual.Ufunc);
            Field("ParamDtype", expected.ParamDtype, actual.ParamDtype);
            Field("HasAxesParam", expected.HasAxesParam, actual.HasAxesParam);
            Field("ExpectedKind", expected.ExpectedKind, actual.ExpectedKind);
            Field("ExpectedDtype", expected.ExpectedDtype, actual.ExpectedDtype);
            Field("HasTruth", expected.HasTruth, actual.HasTruth);
            Field("IsError", expected.IsError, actual.IsError);
        }
    }
}
