using System;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using NumSharp.Examples.Gist.Karpathy;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Examples;

[TestClass, DoNotParallelize]
public class KarpathyDemoTests
{
    [DataTestMethod]
    [DataRow("rnn")] [DataRow("pong")] [DataRow("lstm")]
    [DataRow("microgpt")] [DataRow("nes")] [DataRow("walk")]
    public void EveryActualDemo_PrintsItsMeasuredResultsAndSamples(string name)
    {
        string golden = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "KarpathyDemoOutput.txt")).Replace("\r\n", "\n");
        var sections = Regex.Matches(golden, @"(?ms)^--- ([a-z]+) ---\n(.*?)(?=^--- [a-z]+ ---|\z)");
        Assert.AreEqual(6, sections.Count);
        string? expected = null;
        foreach (Match section in sections) if (section.Groups[1].Value == name) expected = section.Groups[2].Value;
        Assert.IsNotNull(expected);
        Action demo = name switch { "rnn" => MinimalCharacterRnn.Demo, "pong" => PongPolicyGradient.Demo,
            "lstm" => BatchedLstm.Demo, "microgpt" => MicroGpt.Demo, "nes" => NaturalEvolutionStrategies.Demo,
            "walk" => StableDiffusionWalk.Demo, _ => throw new ArgumentException(name) };
        using var probe = np.zeros(0);
        var engine = probe.TensorEngine; var previousBackend = engine.Blas;
        var previousWriter = Console.Out; var previousCulture = CultureInfo.CurrentCulture;
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        try
        {
            OpenBlasEngine.Enable(threads: 1);
            Console.SetOut(output); CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            demo();
        }
        finally { Console.SetOut(previousWriter); CultureInfo.CurrentCulture = previousCulture; engine.Blas = previousBackend; }
        // Only normalize line endings and outer whitespace; sample characters are not altered.
        Assert.AreEqual(expected!.Trim(), output.ToString().Replace("\r\n", "\n").Trim());
    }
}
