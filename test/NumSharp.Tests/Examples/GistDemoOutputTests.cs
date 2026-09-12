using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Examples;

// Console and the BLAS seam are process-global. Restore both even when a contract fails.
[TestClass, DoNotParallelize]
public class GistDemoOutputTests
{
    [DataTestMethod]
    [DataRow(1)] [DataRow(2)] [DataRow(3)] [DataRow(4)] [DataRow(5)]
    [DataRow(6)] [DataRow(7)] [DataRow(8)] [DataRow(9)] [DataRow(10)]
    public void EveryPortDemo_ReproducesAllItsPublishedMeasurements(int rank)
    {
        Action[] demos = { ForecastingMetrics.Demo, RankingMetrics.Demo, FrequencyEstimation.Demo,
            StableDiffusionWalk.Demo, CatchReinforcementLearning.Demo, NaturalEvolutionStrategies.Demo,
            PeakDetection.Demo, TensorboardHistogram.Demo, TimeseriesCnn.Demo, GoogLeNet.Demo };
        string golden = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "GistDemoOutput.txt")).Replace("\r\n", "\n");
        var sections = Regex.Matches(golden, @"(?ms)^([0-9]{2})\. [^\n]*\n(.*?)(?=^[0-9]{2}\. |\z)");
        Assert.AreEqual(10, sections.Count, "The checked demo transcript must cover all ten implementations.");
        var expected = sections[rank - 1].Groups[2].Value;
        using var probe = np.zeros(0);
        var engine = probe.TensorEngine;
        var previousBackend = engine.Blas;
        var previousWriter = Console.Out;
        var previousCulture = CultureInfo.CurrentCulture;
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        try
        {
            OpenBlasEngine.Enable(threads: 1);
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            Console.SetOut(output);
            demos[rank - 1](); // the actual executable Demo, not a rewritten C# algorithm
        }
        finally
        {
            Console.SetOut(previousWriter);
            CultureInfo.CurrentCulture = previousCulture;
            engine.Blas = previousBackend;
        }
        string actual = output.ToString();
        Assert.IsFalse(string.IsNullOrWhiteSpace(actual));
        Assert.IsFalse(actual.Contains("NaN", StringComparison.OrdinalIgnoreCase), "These demonstration fixtures all have finite observables.");
        Assert.AreEqual(Normalize(expected), Normalize(actual), $"Port #{rank}: all reported values, not merely 'did not throw'.");
    }

    private static string Normalize(string text) => Regex.Replace(text, @"\s+", " ").Trim();
}
