using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Anti-coverage-theater gates. Surface coverage proves an op is named; these checks prove
    ///     it is represented by a small matrix rather than one duplicate happy-path row.
    /// </summary>
    [TestClass]
    public class OracleCoverageStrengthTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void EveryOrdinaryOracleOp_HasFourCasesAndAChangingAxis()
        {
            var byOp = new Dictionary<string, Strength>(StringComparer.Ordinal);
            string directory = Path.GetDirectoryName(FuzzCorpus.CorpusPath("unused"));
            foreach (string path in Directory.EnumerateFiles(directory, "*.jsonl"))
            {
                string file = Path.GetFileName(path);
                if (file.EndsWith(".host.jsonl", StringComparison.Ordinal) ||
                    file.StartsWith("index_", StringComparison.Ordinal))
                    continue; // host metadata and the separate advanced-indexing schema

                foreach (var c in FuzzCorpus.Load(file))
                {
                    if (string.IsNullOrEmpty(c.Op))
                        continue;
                    if (!byOp.TryGetValue(c.Op, out var s))
                        byOp[c.Op] = s = new Strength();
                    s.Count++;
                    s.Layouts.Add(c.Layout ?? "");
                    s.ValueClasses.Add(c.Valueclass ?? "");
                    s.DtypeSignatures.Add(string.Join(",", (c.Operands ?? Array.Empty<FuzzCorpus.Operand>())
                        .Select(o => o.Dtype ?? "")));
                    s.ParameterSignatures.Add(c.Params == null ? "{}" : string.Join("|",
                        c.Params.OrderBy(kv => kv.Key, StringComparer.Ordinal)
                            .Select(kv => kv.Key + "=" + kv.Value.GetRawText())));
                    s.Outcomes.Add(c.Error != null || c.Expects_Throw
                        ? "error"
                        : c.Expected?.KindOrArray ?? "array");
                }
            }

            var thin = byOp.Where(kv => kv.Value.Count < 4)
                .Select(kv => $"{kv.Key}={kv.Value.Count}").OrderBy(x => x).ToArray();
            var duplicateOnly = byOp.Where(kv => !kv.Value.HasVariation)
                .Select(kv => kv.Key).OrderBy(x => x).ToArray();

            Assert.AreEqual(0, thin.Length,
                "Every ordinary op needs at least four committed cases; thin: " + string.Join(", ", thin));
            Assert.AreEqual(0, duplicateOnly.Length,
                "Every ordinary op must vary layout, params, operand dtype, value class, or outcome; duplicates only: " +
                string.Join(", ", duplicateOnly));

            int underTen = byOp.Count(kv => kv.Value.Count < 10);
            int oneLayout = byOp.Count(kv => kv.Value.Layouts.Count == 1);
            int oneDtypeSignature = byOp.Count(kv => kv.Value.DtypeSignatures.Count == 1);
            int errorCovered = byOp.Count(kv => kv.Value.Outcomes.Contains("error"));
            Console.WriteLine($"[OracleStrength] ops={byOp.Count}, min_cases={byOp.Min(kv => kv.Value.Count)}, " +
                              $"under10={underTen}, one_layout={oneLayout}, " +
                              $"one_dtype_signature={oneDtypeSignature}, error_covered={errorCovered}");
        }

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void AdvancedIndexingCorpora_KeepTheirMatrixFloors()
        {
            var floors = new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["index_curated.jsonl"] = 2_000,
                ["index_dtype.jsonl"] = 100,
                ["index_setter_dtype.jsonl"] = 10,
                ["index_random_20240626.jsonl"] = 10_000,
            };

            foreach (var pair in floors)
            {
                int count = File.ReadLines(FuzzCorpus.CorpusPath(pair.Key)).Count(line =>
                    !string.IsNullOrWhiteSpace(line));
                Assert.IsTrue(count >= pair.Value,
                    $"{pair.Key} has {count} cases; expected at least {pair.Value}");
            }
        }

        private sealed class Strength
        {
            public int Count;
            public readonly HashSet<string> Layouts = new(StringComparer.Ordinal);
            public readonly HashSet<string> ValueClasses = new(StringComparer.Ordinal);
            public readonly HashSet<string> DtypeSignatures = new(StringComparer.Ordinal);
            public readonly HashSet<string> ParameterSignatures = new(StringComparer.Ordinal);
            public readonly HashSet<string> Outcomes = new(StringComparer.Ordinal);

            public bool HasVariation => Layouts.Count > 1 || ValueClasses.Count > 1 ||
                                        DtypeSignatures.Count > 1 || ParameterSignatures.Count > 1 ||
                                        Outcomes.Count > 1;
        }
    }
}
