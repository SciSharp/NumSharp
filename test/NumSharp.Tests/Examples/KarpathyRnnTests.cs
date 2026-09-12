using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Examples;

[TestClass]
public class KarpathyRnnTests
{
    [TestMethod]
    public void Vocabulary_OrderingRoundtripAndUnknownCharacters_AreExplicit()
    {
        var vocabulary = new MinimalCharacterRnn.Vocabulary("cabca\n");
        Assert.AreEqual("\nabc", vocabulary.Characters);
        CollectionAssert.AreEqual(new[] { 3, 1, 2 }, vocabulary.Encode("cab"));
        Assert.AreEqual("cab\n", vocabulary.Decode(new[] { 3, 1, 2, 0 }));
        Assert.ThrowsException<ArgumentException>(() => vocabulary.Encode("z"));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => vocabulary.Decode(new[] { 4 }));
    }

    [TestMethod]
    public void SmallForwardAndBptt_MatchPrecomputedNumpy_AndDoNotMutateInputs()
    {
        using var model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
        using var hidden = np.array(new double[,] { { .02 }, { -.01 }, { .03 } });
        using var result = model.LossAndGradients(new[] { 0, 1, 2, 3 }, new[] { 1, 2, 3, 0 }, hidden);
        Assert.AreEqual(5.54474603545919, result.Loss, 2e-14);
        double[] expectedHidden = { .004091387759317059, -.017556930803244476, -.0018173108166513062 };
        for (int i = 0; i < 3; i++) Assert.AreEqual(expectedHidden[i], result.LastHidden.item<double>(i, 0), 2e-16);
        CollectionAssert.AreEqual(new long[] { 4, 3 }, result.HiddenStates.shape);
        CollectionAssert.AreEqual(new long[] { 4, 4 }, result.Probabilities.shape);
        for (int t = 0; t < 4; t++)
        {
            using var row = result.Probabilities[$"{t},:"];
            using var sum = np.sum(row, axis: 0);
            Assert.AreEqual(1.0, sum.item<double>(), 3e-16);
        }
        Assert.AreEqual(.02, hidden.item<double>(0, 0));
        foreach (var memory in model.AdagradMemory.Arrays) Assert.IsTrue(np.all(memory == 0));
    }

    [TestMethod]
    public void Bptt_AllFortySmallModelParameters_PassIndependentFiniteDifferences()
    {
        using var model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
        using var hidden = np.array(new double[,] { { .02 }, { -.01 }, { .03 } });
        int[] inputs = { 0, 1, 2, 3 }, targets = { 1, 2, 3, 0 };
        using var analytic = model.LossAndGradients(inputs, targets, hidden, clipGradients: false);
        const double epsilon = 1e-5;
        int checkedParameters = 0;
        for (int group = 0; group < 5; group++)
        {
            var parameter = model.Parameters.Arrays[group];
            for (long i = 0; i < parameter.size; i++)
            {
                long row = i / parameter.shape[1], column = i % parameter.shape[1];
                double original = parameter.item<double>(row, column);
                double plus, minus;
                try
                {
                    parameter[row, column] = original + epsilon;
                    using (var result = model.LossAndGradients(inputs, targets, hidden, false)) plus = result.Loss;
                    parameter[row, column] = original - epsilon;
                    using (var result = model.LossAndGradients(inputs, targets, hidden, false)) minus = result.Loss;
                }
                finally { parameter[row, column] = original; }
                double numerical = (plus - minus) / (2 * epsilon);
                Assert.AreEqual(numerical, analytic.Gradients.Arrays[group].item<double>(row, column), 1e-8, $"group {group}, ({row},{column})");
                checkedParameters++;
            }
        }
        Assert.AreEqual(40, checkedParameters);
    }

    [TestMethod]
    public void OriginalHidden100AndSequence25_ComputeFullTracesAndAllFiveGradients()
    {
        using var model = new MinimalCharacterRnn.Model(5, seed: 42);
        using var hidden = np.zeros((100, 1), dtype: np.float64);
        using var result = model.LossAndGradients(Enumerable.Range(0, 25).Select(i => i % 5).ToArray(),
            Enumerable.Range(0, 25).Select(i => (i + 1) % 5).ToArray(), hidden);
        Assert.AreEqual(100, model.HiddenSize);
        Assert.AreEqual(40.226833494060116, result.Loss, 2e-12);
        CollectionAssert.AreEqual(new long[] { 25, 100 }, result.HiddenStates.shape);
        CollectionAssert.AreEqual(new long[] { 25, 5 }, result.Probabilities.shape);
        for (int i = 0; i < 5; i++)
        {
            var gradient = result.Gradients.Arrays[i];
            CollectionAssert.AreEqual(model.Parameters.Arrays[i].shape, gradient.shape);
            Assert.IsTrue(np.all(np.isfinite(gradient)));
            Assert.IsTrue(np.all(np.abs(gradient) <= 5.0));
        }
    }

    [TestMethod]
    public void ClippingAndAdagrad_UpdateEveryParameterAndPersistentAccumulator()
    {
        using var scope = NDScope.Open();
        using var model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
        using var gradients = model.Parameters.Copy();
        foreach (var gradient in gradients.Arrays)
        {
            gradient[":"] = .25;
            gradient[0, 0] = -9.0;
            gradient[-1, -1] = 9.0;
        }
        MinimalCharacterRnn.ClipGradients(gradients);
        foreach (var gradient in gradients.Arrays)
        {
            Assert.AreEqual(-5.0, gradient.item<double>(0, 0));
            Assert.AreEqual(5.0, gradient.item<double>(gradient.shape[0] - 1, gradient.shape[1] - 1));
        }
        using var before = model.Parameters.Copy();
        model.ApplyAdagrad(gradients);
        for (int group = 0; group < 5; group++)
        for (long i = 0; i < gradients.Arrays[group].size; i++)
        {
            double g = gradients.Arrays[group].item<double>(i);
            Assert.AreEqual(g * g, model.AdagradMemory.Arrays[group].item<double>(i));
            double expected = before.Arrays[group].item<double>(i) - .1 * g / System.Math.Sqrt(g * g + 1e-8);
            Assert.AreEqual(expected, model.Parameters.Arrays[group].item<double>(i), 3e-17);
        }
    }

    [TestMethod]
    public void BoundedTraining_LearnsTinyCorpus_ResetsAndSamplesAtSpecifiedIntervals()
    {
        string corpus = string.Concat(Enumerable.Repeat("hello world\n", 40));
        var vocabulary = new MinimalCharacterRnn.Vocabulary(corpus);
        using var model = new MinimalCharacterRnn.Model(vocabulary.Count, 16, seed: 1);
        using var result = model.Train(vocabulary.Encode(corpus), 40, sampleEvery: 20, sampleLength: 40);
        Assert.AreEqual(40, result.Steps.Count);
        Assert.AreEqual(54.929665653107975, result.Steps[0].Loss, 2e-12);
        Assert.IsTrue(result.Steps[^1].Loss < 1.0, "Real training must reduce the specified batch loss below one nat.");
        Assert.AreEqual(.6994428739514988, result.Steps[^1].Loss, 1e-10);
        Assert.AreEqual(53.138829143652835, result.Steps[^1].SmoothLoss, 1e-10);
        CollectionAssert.AreEqual(new[] { 0, 19, 38 }, result.Steps.Where(s => s.ResetHidden).Select(s => s.Iteration).ToArray());
        CollectionAssert.AreEqual(new[] { 0, 20 }, result.Samples.Select(s => s.Iteration).ToArray());
        Assert.AreEqual("hereeo hdlrhrhl ewlohhr\nlwh rdohwlr\nw\ndw", vocabulary.Decode(result.Samples[0].Tokens));
        Assert.AreEqual("elo worllo world\nlell\nhhello world\nhello", vocabulary.Decode(result.Samples[1].Tokens));
        Assert.IsTrue(result.Samples.All(s => s.Tokens.Length == 40));
    }

    [TestMethod]
    public void Sampling_IsSeededFunctionalAndDoesNotMutateInitialState()
    {
        using var model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
        using var hidden = np.zeros((3, 1), dtype: np.float64);
        int[] expected = { 0, 3, 0, 0, 1, 0, 2, 3, 3, 2, 0, 2, 2, 0, 0, 1, 0, 2, 1, 2 };
        CollectionAssert.AreEqual(expected, model.Sample(hidden, 0, 20, np.random.RandomState(19)));
        Assert.AreEqual(0, model.Sample(hidden, 0, 0).Length);
        Assert.IsTrue(np.all(hidden == 0));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => model.Sample(hidden, 4, 1));
        Assert.ThrowsException<ArgumentException>(() => model.LossAndGradients(Array.Empty<int>(), Array.Empty<int>(), hidden));
    }

    [TestMethod]
    public void ModelAndReturnedResults_OutliveCallerTemporaryScope()
    {
        MinimalCharacterRnn.Model model;
        MinimalCharacterRnn.LossResult result;
        using (var scope = NDScope.Open())
        {
            model = new MinimalCharacterRnn.Model(4, 3, seed: 7);
            result = model.LossAndGradients(new[] { 0, 1 }, new[] { 1, 2 }, np.zeros((3, 1), dtype: np.float64));
        }
        using (model) using (result)
        {
            Assert.IsTrue(double.IsFinite(model.Parameters.Wxh.item<double>(0, 0)));
            Assert.IsTrue(double.IsFinite(result.LastHidden.item<double>(0, 0)));
            model.ApplyAdagrad(result.Gradients);
            Assert.IsTrue(double.IsFinite(model.AdagradMemory.Wxh.item<double>(0, 0)));
        }
    }

    [TestMethod, DoNotParallelize]
    public void Demo_PrintsTheActualMeasuredTrainingAndSamples()
    {
        var previous = Console.Out; var culture = CultureInfo.CurrentCulture;
        using var writer = new StringWriter(CultureInfo.InvariantCulture);
        try { Console.SetOut(writer); CultureInfo.CurrentCulture = CultureInfo.InvariantCulture; MinimalCharacterRnn.Demo(); }
        finally { Console.SetOut(previous); CultureInfo.CurrentCulture = culture; }
        string output = writer.ToString();
        StringAssert.Contains(output, "characters=480, vocabulary=9, hidden=16, updates=40");
        StringAssert.Contains(output, "Sample 0: hereeo hdlrhrhl ewlohhr\\nlwh rdohwlr\\nw\\ndw");
        StringAssert.Contains(output, "Sample 20: elo worllo world\\nlell\\nhhello world\\nhello");
        string lossLine = output.Split('\n').Single(line => line.StartsWith("Loss:"));
        string[] fields = lossLine.Trim().Replace("Loss: initial=", "").Split(", final=");
        Assert.AreEqual(54.929665653107975, double.Parse(fields[0], CultureInfo.InvariantCulture), 2e-12);
        Assert.AreEqual(.6994428739514988, double.Parse(fields[1], CultureInfo.InvariantCulture), 1e-10);
    }
}
