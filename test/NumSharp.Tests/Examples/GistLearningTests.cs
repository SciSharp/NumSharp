using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Examples;

[TestClass]
public class GistLearningTests
{
    [TestMethod]
    public void Catch_TransitionsAndTerminalReward()
    {
        using var game = new CatchReinforcementLearning.CatchGame(5);
        game.Reset(2, 2);
        for (int i = 0; i < 4; i++)
        {
            var result = game.Act(1);
            using var observation = result.Observation;
            CollectionAssert.AreEqual(new long[] { 1, 25 }, observation.shape);
            Assert.AreEqual(i == 3 ? 1 : 0, result.Reward);
            Assert.AreEqual(i == 3, result.GameOver);
            Assert.AreEqual(1.0, observation.item<double>((i + 1) * 5 + 2));
        }
        Assert.ThrowsException<InvalidOperationException>(() => game.Act(0));
        game.Reset(0, 4);
        using var frame = game.Observe();
        Assert.AreEqual(1.0, frame.item<double>(23));
        Assert.AreEqual(1.0, frame.item<double>(24));
        Assert.AreEqual(0.0, frame.item<double>(22));
    }

    [TestMethod]
    public void Catch_SeededReset_UsesOriginalDrawBounds()
    {
        using var game = new CatchReinforcementLearning.CatchGame(10, seed: 0);
        using var state = game.State();
        Assert.AreEqual(5L, state.item<long>(0, 1));
        Assert.AreEqual(1L, state.item<long>(0, 2));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => game.Act(3));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => game.Reset(-1, 2));
    }

    [TestMethod]
    public void Replay_TargetsBootstrapOnlySelectedNonterminalAction()
    {
        using var scope = NDScope.Open();
        using var replay = new CatchReinforcementLearning.ReplayMemory(capacity: 2);
        var a = np.array(new double[,] { { 1, 2 } });
        var b = np.array(new double[,] { { 3, 4 } });
        replay.Remember(a, 1, .5, b, false);
        replay.Remember(b, 2, -1, a, true);
        a[0, 0] = 99.0; // replay owns a snapshot, not this subsequently changed buffer.
        NDArray Predict(NDArray state) => np.array(new double[,] { { state.item<double>(0, 0), state.item<double>(0, 1), 1 } });
        var (inputs, targets) = replay.GetBatchAt(Predict, 3, 0, 1);
        Assert.AreEqual(1.0, inputs.item<double>(0, 0));
        Assert.AreEqual(4.1, targets.item<double>(0, 1));
        Assert.AreEqual(-1.0, targets.item<double>(1, 2));
        Assert.AreEqual(3.0, targets.item<double>(1, 0));
        replay.Remember(b, 0, 1, b, false);
        Assert.AreEqual(2, replay.Count);
    }

    [TestMethod]
    public void Replay_EmptyAndInvalidPredictionFailExplicitly()
    {
        using var scope = NDScope.Open();
        using var replay = new CatchReinforcementLearning.ReplayMemory();
        Assert.ThrowsException<InvalidOperationException>(() => replay.GetBatch(_ => np.zeros((1, 3)), 3));
        replay.Remember(np.zeros((1, 2)), 1, 0, np.ones((1, 2)), false);
        Assert.ThrowsException<ArgumentException>(() => replay.GetBatchAt(_ => np.zeros((2, 3)), 3, 0));
    }

    [TestMethod]
    public void CatchAndReplay_OwnStateBeyondAnInnerTemporaryScope()
    {
        CatchReinforcementLearning.CatchGame game;
        using var replay = new CatchReinforcementLearning.ReplayMemory();
        using (NDScope.Open())
        {
            game = new CatchReinforcementLearning.CatchGame(5);
            game.Reset(2, 2);
            var state = game.Observe();
            var step = game.Act(1);
            replay.Remember(state, 1, step.Reward, step.Observation, false);
        }
        using (game)
        {
            using var state = game.State();
            Assert.AreEqual(1L, state.item<long>(0, 0));
            var (inputs, targets) = replay.GetBatchAt(_ => np.ones((1, 3)), 3, 0);
            using (inputs) using (targets)
            {
                Assert.AreEqual(1.0, inputs.item<double>(0, 2));
                Assert.AreEqual(.9, targets.item<double>(0, 1));
            }
        }
    }

    [TestMethod]
    public void Nes_OneStep_MatchesCapturedNumPy()
    {
        using var scope = NDScope.Open();
        var w = np.array(new[] { .25, -.5 });
        var solution = np.array(new[] { .5, .25 });
        var noise = np.array(new double[,] { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } });
        var rewards = NaturalEvolutionStrategies.Rewards(w, noise, trial => NaturalEvolutionStrategies.QuadraticReward(trial, solution));
        var actual = NaturalEvolutionStrategies.Step(w, noise, rewards, alpha: .05);
        Assert.AreEqual(.3618033988749895, actual.item<double>(0), 1e-15);
        Assert.AreEqual(-.16458980337503154, actual.item<double>(1), 1e-15);
        Assert.AreEqual(.25, w.item<double>(0));
        Assert.ThrowsException<ArgumentException>(() => NaturalEvolutionStrategies.Step(w, noise, np.ones(4)));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => NaturalEvolutionStrategies.Step(w, noise, rewards, sigma: 0));
    }

    [TestMethod]
    public void Nes_FullOptimization_IsFunctionalAndDeterministic()
    {
        using var solution = np.array(new[] { .5, .1, -.3 });
        using var first = NaturalEvolutionStrategies.OptimizeQuadratic(solution);
        using var second = NaturalEvolutionStrategies.OptimizeQuadratic(solution);
        Assert.IsTrue(NaturalEvolutionStrategies.QuadraticReward(first, solution) > -.001);
        Assert.IsTrue(np.array_equal(first, second));
    }

    [TestMethod]
    public void Histogram_BucketsStatisticsAndConstantRange()
    {
        using var scope = NDScope.Open();
        using var result = TensorboardHistogram.Create(np.array(new[] { 0.0, 1, 1, 2, 3, 4 }), bins: 4);
        Assert.IsTrue(np.array_equal(result.Buckets, np.array(new long[] { 1, 2, 1, 2 })));
        Assert.IsTrue(np.array_equal(result.BucketLimits, np.array(new[] { 1.0, 2, 3, 4 })));
        Assert.AreEqual(6L, result.Num);
        Assert.AreEqual(11.0, result.Sum);
        Assert.AreEqual(31.0, result.SumSquares);
        using var constant = TensorboardHistogram.Create(np.array(new[] { 2.0, 2, 2 }), bins: 4);
        Assert.IsTrue(np.array_equal(constant.Buckets, np.array(new long[] { 0, 0, 3, 0 })));
        Assert.AreEqual(2.5, constant.BucketLimits.item<double>(3));
    }

    [TestMethod]
    public void Histogram_StridedInputAndRejectedRanges()
    {
        using var scope = NDScope.Open();
        var backing = np.array(new[] { 0.0, 99, 1, 99, 2, 99, 3, 99 });
        using var result = TensorboardHistogram.Create(backing["::2"], bins: 3);
        Assert.IsTrue(np.array_equal(result.Buckets, np.array(new long[] { 1, 1, 2 })));
        Assert.ThrowsException<ArgumentException>(() => TensorboardHistogram.Create(np.array(new[] { double.NaN })));
        Assert.ThrowsException<ArgumentException>(() => TensorboardHistogram.Create(np.zeros(0)));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => TensorboardHistogram.Create(backing, 0));
        Assert.ThrowsException<ArgumentException>(() => TensorboardHistogram.Create(np.array(new[] { 9007199254740992.0, 9007199254740994.0 }), 2));
    }

    [TestMethod]
    public void Histogram_Float32StatisticsRetainOriginalReductionDtype()
    {
        using var values = np.array(new[] { .1f, .2f, .2f, .3f, .4f, .5f });
        using var result = TensorboardHistogram.Create(values, 4);
        Assert.AreEqual(NPTypeCode.Single, result.BucketLimits.typecode);
        Assert.AreEqual(1.7000000476837158, result.Sum);
        Assert.AreEqual(.5900000333786011, result.SumSquares);
    }
}
