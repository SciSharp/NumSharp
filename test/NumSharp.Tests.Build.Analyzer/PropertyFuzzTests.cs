using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     COVERAGE_PLAN RB-5 — the property-based invariant gate. A deterministic seeded generator
    ///     composes method bodies from a grammar of leak-neutral and leak-positive statements and
    ///     asserts the analyzer's NDW012 count equals the EXACT number of leaky statements — i.e. the
    ///     two invariants "a value that is disposed/using/stored/handed-off is never flagged" and "a
    ///     dropped/discarded produced value is always flagged", preserved under arbitrary interleaving.
    ///     Seeded <see cref="Random"/> is stable, so every run replays the same 100 bodies; a failure
    ///     prints the seed and the generated method for direct reproduction.
    /// </summary>
    [TestClass]
    public class PropertyFuzzTests
    {
        // (statement template, expected NDW012 count) — {0} is a unique local suffix.
        private static readonly (string Stmt, int Leaks)[] Grammar =
        {
            ("var t{0} = a + b;", 1),                          // dropped temp
            ("_ = a + b;", 1),                                 // explicit discard
            ("var t{0} = (a + 1.0, b - 1.0);", 1),             // dropped tuple literal
            ("var g{0} = new[] {{ a + b }};", 1),              // dropped array literal
            ("var t{0} = a + b; t{0}.Dispose();", 0),          // disposed
            ("using var t{0} = a + b;", 0),                    // using declaration
            ("var t{0} = a + b; Sink(t{0});", 0),              // handed to a foreign sink
            ("_field = a + b;", 0),                            // stored to a field
            ("var t{0} = (a, b);", 0),                         // tuple of aliases (owns nothing)
        };

        [TestMethod]
        public async Task RandomizedBodies_FlagExactlyTheLeakyStatements()
        {
            const int seeds = 100;
            var src = new StringBuilder();
            src.AppendLine("using NumSharp;");
            src.AppendLine("public static class F {");
            src.AppendLine("  static NDArray _field;");
            src.AppendLine("  static void Sink(NDArray x) { }");

            int expectedTotal = 0;
            var perMethod = new List<(int Seed, int Expected, string Body)>();
            for (int seed = 0; seed < seeds; seed++)
            {
                var rng = new Random(seed);
                int count = 3 + rng.Next(6); // 3..8 statements
                var body = new StringBuilder();
                int expected = 0;
                for (int i = 0; i < count; i++)
                {
                    var (stmt, leaks) = Grammar[rng.Next(Grammar.Length)];
                    body.Append("    ").AppendFormat(stmt, seed + "_" + i).AppendLine();
                    expected += leaks;
                }
                perMethod.Add((seed, expected, body.ToString()));
                expectedTotal += expected;
                src.AppendLine($"  public static void M{seed}(NDArray a, NDArray b) {{");
                src.Append(body);
                src.AppendLine("  }");
            }
            src.AppendLine("}");

            var r = await AnalyzerTestHarness.RunAsync(src.ToString(), "property_fuzz.cs");
            Assert.IsTrue(r.CompileErrors.IsEmpty,
                "generated source must compile:\n" + string.Join("\n", r.CompileErrors));
            Assert.IsTrue(expectedTotal > 0, "the grammar must have produced leaky statements (non-vacuous)");

            if (r.CountOf("NDW012") == expectedTotal)
                return;

            // Mismatch: re-run per method to pinpoint the seed, and report its exact body.
            foreach (var (seed, expected, body) in perMethod)
            {
                var one = "using NumSharp;\npublic static class F {\n  static NDArray _field;\n" +
                          "  static void Sink(NDArray x) { }\n" +
                          $"  public static void M(NDArray a, NDArray b) {{\n{body}  }}\n}}";
                var rr = await AnalyzerTestHarness.RunAsync(one, $"property_fuzz_seed{seed}.cs");
                Assert.AreEqual(expected, rr.CountOf("NDW012"),
                    $"seed {seed}: expected {expected} NDW012 for the body:\n{body}");
            }
            Assert.Fail($"total NDW012 {r.CountOf("NDW012")} != expected {expectedTotal}, " +
                        "but every per-seed replay matched — investigate cross-method interference");
        }
    }
}
