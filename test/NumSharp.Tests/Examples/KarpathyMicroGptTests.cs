using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Examples;

[TestClass]
public class KarpathyMicroGptTests
{
    [TestMethod]
    public void Tokenizer_UsesCodePointOrderAndSharedBosEndMarker()
    {
        var tokenizer = new MicroGpt.Tokenizer(new[] { "b😀a", "ab" });
        CollectionAssert.AreEqual(new[] { 3, 1, 2, 0, 3 }, tokenizer.Encode("b😀a"));
        Assert.AreEqual("b😀a", tokenizer.Decode(tokenizer.Encode("b😀a")));
        Assert.ThrowsException<ArgumentException>(() => tokenizer.Encode("z"));
    }

    [TestMethod]
    public void OriginalDefaultDimensions_Have4192ParametersFor27Tokens_AndLiveBeyondCallerScope()
    {
        MicroGpt.Model model;
        using (NDScope.Open()) model = new MicroGpt.Model(27);
        using (model)
        {
            Assert.AreEqual(16, model.EmbeddingSize);
            Assert.AreEqual(4, model.HeadCount);
            Assert.AreEqual(16, model.BlockSize);
            Assert.AreEqual(1, model.LayerCount);
            Assert.AreEqual(4192L, model.ParameterCount);
            Assert.AreEqual(9, model.Parameters.Count);
            using var logits = model.Forward(new[] { 26, 0, 1 });
            CollectionAssert.AreEqual(new long[] { 3, 27 }, logits.shape);
            Assert.IsTrue(np.all(np.isfinite(logits)));
        }
    }

    [TestMethod]
    public void CausalAttention_FutureTokensCannotChangeEarlierOutputs()
    {
        using var model = new MicroGpt.Model(5, seed: 8);
        int[] original = Enumerable.Range(0, 16).Select(i => i % 5).ToArray();
        int[] changed = original.Select((x, i) => i < 8 ? x : (x + 1) % 5).ToArray();
        using var a = model.Forward(original);
        using var b = model.Forward(changed);
        using var prefixA = a[":8,:"];
        using var prefixB = b[":8,:"];
        Assert.IsTrue(np.array_equal(prefixA, prefixB), "Later keys and values must not leak through the causal mask.");
        Assert.IsFalse(np.array_equal(a, b));
    }

    [TestMethod]
    public void EveryParameterGradient_AgreesWithCentralDifferences()
    {
        using var model = new MicroGpt.Model(3, embeddingSize: 4, headCount: 2, blockSize: 3, seed: 7);
        int[] inputs = { 2, 0, 1 }, targets = { 0, 1, 2 };
        using var batch = model.LossAndGradients(inputs, targets);
        double worst = 0;
        int checkedElements = 0;
        const double delta = 1e-5;
        foreach (var (name, parameter) in model.Parameters)
        {
            for (long i = 0; i < parameter.size; i++)
            {
                long row = i / parameter.shape[1], column = i % parameter.shape[1];
                double original = parameter.item<double>(row, column);
                double positive, negative;
                try
                {
                    parameter[row, column] = original + delta;
                    positive = model.Loss(inputs, targets);
                    parameter[row, column] = original - delta;
                    negative = model.Loss(inputs, targets);
                }
                finally { parameter[row, column] = original; }
                double numeric = (positive - negative) / (2 * delta);
                double analytic = batch.Parameters[name].item<double>(row, column);
                double error = System.Math.Abs(numeric - analytic) / (1 + System.Math.Abs(numeric) + System.Math.Abs(analytic));
                worst = System.Math.Max(worst, error);
                Assert.IsTrue(error < 2e-6, $"{name}[{row},{column}] analytic={analytic:R}, numeric={numeric:R}, normalized error={error:R}");
                checkedElements++;
            }
        }
        Assert.AreEqual(model.ParameterCount, (long)checkedElements);
        Console.WriteLine($"MICROGPT_GRADCHECK parameters={checkedElements}, max_normalized_error={worst:R}, delta={delta}");
    }

    [TestMethod]
    public void ActualTrainingLowersCorpusLoss_AndProducesTwentyBoundedSamples()
    {
        var tokenizer = new MicroGpt.Tokenizer(MicroGpt.TinyNames);
        using var model = new MicroGpt.Model(tokenizer.Size);
        double CorpusLoss() => MicroGpt.TinyNames.Average(doc =>
        {
            var tokens = tokenizer.Encode(doc);
            return model.Loss(tokens.Take(tokens.Length - 1).ToArray(), tokens.Skip(1).ToArray());
        });
        double before = CorpusLoss();
        using var trace = model.Train(tokenizer, MicroGpt.TinyNames, steps: 200);
        double after = CorpusLoss();
        Assert.AreEqual(200, model.OptimizerStep);
        Assert.AreEqual(200L, trace.size);
        Assert.IsTrue(np.all(np.isfinite(trace)));
        Assert.IsTrue(after < before * .75, $"Actual corpus loss must fall materially: {before:R} -> {after:R}");
        var samples = model.Generate(tokenizer, samples: 20, temperature: .5, seed: 42);
        var repeated = model.Generate(tokenizer, samples: 20, temperature: .5, seed: 42);
        Assert.AreEqual(20, samples.Length);
        CollectionAssert.AreEqual(samples, repeated);
        Assert.IsTrue(samples.Any(x => x.Length > 0));
        foreach (string sample in samples)
        {
            Assert.IsTrue(sample.Length <= 16);
            tokenizer.Encode(sample); // every emitted character belongs to the trained vocabulary
        }
        Console.WriteLine($"MICROGPT_TRAIN corpus_loss={before:R}->{after:R}; samples={string.Join(',', samples)}");
    }

    [TestMethod]
    public void ContextTruncationAndBadArguments_AreExplicit()
    {
        var tokenizer = new MicroGpt.Tokenizer(new[] { "ab" });
        using var model = new MicroGpt.Model(tokenizer.Size, embeddingSize: 4, headCount: 2, blockSize: 3);
        using var trace = model.Train(tokenizer, new[] { "abababababab" }, steps: 1);
        Assert.IsTrue(double.IsFinite(trace.item<double>()));
        Assert.ThrowsException<ArgumentException>(() => model.Forward(Array.Empty<int>()));
        Assert.ThrowsException<ArgumentException>(() => model.Forward(new[] { 0, 0, 0, 0 }));
        Assert.ThrowsException<ArgumentException>(() => model.Generate(tokenizer, temperature: 0));
        Assert.ThrowsException<ArgumentException>(() => new MicroGpt.Model(3, embeddingSize: 5, headCount: 2));
    }

    [TestMethod]
    public void OriginalDefaultThousandStepSchedule_TrainsEveryStepAndImprovesLocalCorpus()
    {
        var tokenizer = new MicroGpt.Tokenizer(MicroGpt.TinyNames);
        using var model = new MicroGpt.Model(tokenizer.Size);
        double CorpusLoss() => MicroGpt.TinyNames.Average(doc =>
        {
            var tokens = tokenizer.Encode(doc);
            return model.Loss(tokens.Take(tokens.Length - 1).ToArray(), tokens.Skip(1).ToArray());
        });
        double before = CorpusLoss();
        int observedSteps = 0;
        using var trace = model.Train(tokenizer, MicroGpt.TinyNames, progress: (step, loss) =>
        {
            Assert.AreEqual(observedSteps++, step);
            Assert.IsTrue(double.IsFinite(loss));
        });
        double after = CorpusLoss();
        Assert.AreEqual(1000, observedSteps);
        Assert.AreEqual(1000, model.OptimizerStep);
        Assert.AreEqual(1000L, trace.size);
        Assert.IsTrue(after < before * .75, $"Default 1000-step training: {before:R} -> {after:R}");
        foreach (var parameter in model.Parameters.Values) Assert.IsTrue(np.all(np.isfinite(parameter)));
        Console.WriteLine($"MICROGPT_DEFAULT_1000 corpus_loss={before:R}->{after:R}; first={trace.item<double>(0):R}; last={trace.item<double>(999):R}");
    }

    [TestMethod]
    public void OwningGradientBatch_SurvivesCallerScopeAndCanUpdateModel()
    {
        MicroGpt.Model model;
        MicroGpt.GradientBatch batch;
        using (NDScope.Open())
        {
            model = new MicroGpt.Model(3, embeddingSize: 4, headCount: 2, blockSize: 3);
            batch = model.LossAndGradients(new[] { 2, 0, 1 }, new[] { 0, 1, 2 });
        }
        using (model) using (batch)
        {
            Assert.IsTrue(np.all(np.isfinite(batch.Logits)));
            foreach (var gradient in batch.Parameters.Values) Assert.IsTrue(np.all(np.isfinite(gradient)));
            model.ApplyAdam(batch, .01);
            Assert.AreEqual(1, model.OptimizerStep);
            Assert.IsTrue(np.all(np.isfinite(model.Parameters["wte"])));
        }
    }
}
