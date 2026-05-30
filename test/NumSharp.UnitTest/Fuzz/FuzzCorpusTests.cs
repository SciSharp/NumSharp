using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Fuzz;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Differential matrix: replay every committed NumPy oracle case through NumSharp and
    ///     bit-compare. One test per corpus file; a failure lists every divergent cell so the
    ///     whole matrix is visible at once (not first-failure-wins). Runs in CI under FuzzMatrix.
    /// </summary>
    [TestClass]
    public class FuzzCorpusTests
    {
        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Astype_Smoke() => RunCorpus("astype_smoke.jsonl");

        [TestMethod]
        [TestCategory("FuzzMatrix")]
        public void Astype_Full() => RunCorpus("astype_full.jsonl");

        private static void RunCorpus(string file)
        {
            var cases = FuzzCorpus.Load(file);
            Assert.IsTrue(cases.Count > 0, $"corpus '{file}' has no cases (was it generated/copied?)");

            var failures = new List<string>();
            foreach (var c in cases)
            {
                try
                {
                    var operands = new NumSharp.NDArray[c.Operands.Length];
                    for (int i = 0; i < operands.Length; i++)
                        operands[i] = FuzzCorpus.Reconstruct(c.Operands[i]);

                    var result = OpRegistry.Apply(c.Op, c.Params, operands);
                    var actual = FuzzCorpus.ResultBytes(result);
                    var expected = FuzzCorpus.FromHex(c.Expected.Buffer);
                    var tc = FuzzCorpus.DtypeToTC(c.Expected.Dtype);

                    var diffs = BitDiff.Compare(expected, actual, tc);
                    if (diffs.Count > 0)
                        failures.Add($"{c.Id} [{c.Layout}]: " +
                            string.Join(", ", diffs.Take(3).Select(d => $"@{d.Index} exp {d.Expected} act {d.Actual}")) +
                            (diffs.Count > 3 ? $" (+{diffs.Count - 3} more)" : ""));
                }
                catch (Exception e)
                {
                    failures.Add($"{c.Id} [{c.Layout}]: THREW {e.GetType().Name}: {e.Message}");
                }
            }

            if (failures.Count > 0)
                Assert.Fail($"{failures.Count}/{cases.Count} cases diverged from NumPy:\n  " +
                            string.Join("\n  ", failures.Take(60)));
        }
    }
}
