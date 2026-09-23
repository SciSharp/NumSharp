using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Examples;

[TestClass]
public class GistLearningSourceDemonstrationsTests
{
    [TestMethod]
    public void Nes_AllFifteenPublishedProgressMeasurements_MatchPrintedPrecision()
    {
        // Numerical observations printed at nes.py lines 53–67, not invented success criteria.
        // Parameters are the source's seed=0, 300 updates, population=50, sigma=.1, alpha=.001.
        double[,] published = {
            { 1.76405235, .40015721, .97873798, -3.323094 },
            { 1.63796944, .36987244, .84497941, -2.678783 },
            { 1.50042904, .33577052, .70329169, -2.063040 },
            { 1.36438269, .29247833, .56990397, -1.540938 },
            { 1.22573280, .25622233, .43607161, -1.092895 },
            { 1.08819889, .22827364, .30415088, -.727430 },
            { .95675286, .19282042, .16682465, -.435164 },
            { .82214521, .16161165, .03600742, -.220475 },
            { .70282088, .12935569, -.09779598, -.082885 },
            { .58380424, .11579811, -.21083135, -.015224 },
            { .52089064, .09897718, -.27612250, -.001008 },
            { .50861791, .10220363, -.29023563, -.000174 },
            { .50428202, .10834192, -.29828744, -.000091 },
            { .50147991, .10445590, -.30255291, -.000029 },
            { .50208135, .09867220, -.29841024, -.000009 }
        };
        using var solution = np.array(new[] { .5, .1, -.3 });
        int checkedCheckpoints = 0, observedStates = 0;
        using var final = NaturalEvolutionStrategies.OptimizeQuadratic(solution, progress: (iteration, weights, reward) =>
        {
            observedStates++;
            if (iteration == 300 || iteration % 20 != 0) return;
            int index = iteration / 20;
            for (int j = 0; j < 3; j++)
                Assert.AreEqual(published[index, j], weights.item<double>(j), 5.1e-9, $"printed iter {iteration}, weight {j}");
            Assert.AreEqual(published[index, 3], reward, 5.1e-7, $"printed iter {iteration} reward");
            checkedCheckpoints++;
        });
        Assert.AreEqual(15, checkedCheckpoints);
        Assert.AreEqual(301, observedStates);
        Assert.IsTrue(NaturalEvolutionStrategies.QuadraticReward(final, solution) > -.001);
    }

    [TestMethod]
    public void Catch_SourceScaleEpisodes_ActionSelectionReplayAndTargetsAreFunctional()
    {
        // Source numerical settings, with a deterministic fixed-Q test predictor, NOT Keras training.
        using var game = new CatchReinforcementLearning.CatchGame(10);
        using var replay = new CatchReinforcementLearning.ReplayMemory(500, .9);
        using var q = np.array(new double[,] { { .25, .5, -.25 } });
        var random = np.random.RandomState(7);
        int wins = 0, moves = 0;
        for (int episode = 0; episode < 1000; episode++)
        {
            game.Reset(random);
            while (!game.IsOver)
            {
                using var before = game.Observe();
                int action = CatchReinforcementLearning.SelectAction(q, .1, random);
                var step = game.Act(action);
                using var after = step.Observation;
                replay.Remember(before, action, step.Reward, after, step.GameOver);
                if (step.Reward == 1) wins++;
                moves++;
                CollectionAssert.AreEqual(new long[] { 1, 100 }, after.shape);
            }
        }
        Assert.AreEqual(9000, moves);
        Assert.AreEqual(330, wins, "Observed from the independently run seeded NumPy fixed-Q numerical loop.");
        Assert.AreEqual(500, replay.Count);
        var (inputs, targets) = replay.GetBatch(_ => q.copy(), 3, 50, np.random.RandomState(11));
        using (inputs) using (targets)
        {
            CollectionAssert.AreEqual(new long[] { 50, 100 }, inputs.shape);
            CollectionAssert.AreEqual(new long[] { 50, 3 }, targets.shape);
            Assert.IsTrue(np.all(np.isfinite(targets)));
        }
    }

    [TestMethod]
    public void Catch_EpsilonSelectorValidatesAndPreservesFirstArgmaxTie()
    {
        using var q = np.array(new double[,] { { 1, 1, 0 } });
        var random = np.random.RandomState(2);
        Assert.AreEqual(0, CatchReinforcementLearning.SelectAction(q, 0, random));
        for (int i = 0; i < 100; i++)
            Assert.IsTrue(CatchReinforcementLearning.SelectAction(q, 1, random) is >= 0 and < 3);
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => CatchReinforcementLearning.SelectAction(q, double.NaN, random));
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Double)]
    [DataRow(NPTypeCode.Single)]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.Int64)]
    public void Histogram_SourceDefaultThousandBinsAndAllNumericFields_AreFunctional(NPTypeCode dtype)
    {
        using var scope = NDScope.Open();
        var values = np.arange(4096).astype(dtype).reshape(64, 64)["::-1,::2"];
        using var summary = TensorboardHistogram.Create(values); // source bins=1000, not a four-bin stand-in
        Assert.AreEqual(1000L, summary.Buckets.size);
        Assert.AreEqual(1000L, summary.BucketLimits.size);
        Assert.AreEqual(2048L, summary.Num);
        Assert.AreEqual(summary.Num, np.sum(summary.Buckets).item<long>());
        Assert.AreEqual(0.0, summary.Min);
        Assert.AreEqual(4094.0, summary.Max);
        Assert.AreEqual(4192256.0, summary.Sum);
        // Integer squaring is done in the original input dtype; these fixture squares do not overflow.
        Assert.IsTrue(summary.SumSquares > 0);
        Assert.IsTrue(np.all(np.diff(summary.BucketLimits) > 0));
    }
}
