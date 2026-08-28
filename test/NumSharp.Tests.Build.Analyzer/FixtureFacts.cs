using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     Content-based fixture assertions shared by the scenario test classes — "the warning landed on
    ///     the line that STARTS WITH this text" / "no warning landed on a line CONTAINING this text".
    ///     Robust to line shifts (unlike the exact-match tag gate, which pins the line number), so the
    ///     two together catch both a mis-placed diagnostic and a diagnostic on the wrong statement.
    /// </summary>
    internal static class FixtureFacts
    {
        public static async Task<List<string>> FlaggedLineTexts(string fileName, string id = "NDW012")
        {
            var result = await AnalyzerTestHarness.RunFileAsync(fileName);
            Assert.IsTrue(result.CompileErrors.IsEmpty,
                $"fixture '{fileName}' must compile:\n  " +
                string.Join("\n  ", result.CompileErrors.Select(e => e.ToString())));
            var lines = File.ReadAllLines(AnalyzerTestHarness.FixturePath(fileName));
            return result.Ndw.Where(d => d.Id == id)
                .Select(d => lines[AnalyzerTestHarness.LineOf(d) - 1].Trim())
                .ToList();
        }

        public static void AnyStartsWith(List<string> flagged, string prefix, string what)
            => Assert.IsTrue(flagged.Any(t => t.StartsWith(prefix, StringComparison.Ordinal)),
                $"expected a warning on {what} ('{prefix}'); flagged: [{string.Join(" | ", flagged)}]");

        public static void NoneContains(List<string> flagged, string needle, string what)
            => Assert.IsFalse(flagged.Any(t => t.Contains(needle)),
                $"{what} ('{needle}') must NOT draw a warning; flagged: [{string.Join(" | ", flagged)}]");
    }
}
