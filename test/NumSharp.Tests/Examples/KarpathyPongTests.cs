using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist.Karpathy;

namespace NumSharp.Tests.Examples;

[TestClass]
public class KarpathyPongTests
{
    [TestMethod]
    public void Preprocess_ExactCropStrideBackgroundAndSourceMutation()
    {
        using var scope = NDScope.Open();
        var frame = np.full(new Shape(210, 160, 3), (byte)144, dtype: np.uint8);
        frame[35, 0, 0] = (byte)255;
        frame[35, 2, 0] = (byte)109;
        frame[193, 158, 0] = (byte)30;
        frame[34, 0, 0] = (byte)255;
        var input = PongPolicyGradient.PreprocessFrame(frame);
        CollectionAssert.AreEqual(new long[] { 6400 }, input.shape);
        Assert.AreEqual(NPTypeCode.Double, input.typecode);
        Assert.AreEqual(2.0, np.sum(input).item<double>());
        Assert.AreEqual(1.0, input.item<double>(0));
        Assert.AreEqual(1.0, input.item<double>(6399));
        Assert.AreEqual((byte)1, frame.item<byte>(35, 0, 0));
        Assert.AreEqual((byte)0, frame.item<byte>(35, 2, 0));
        Assert.AreEqual((byte)255, frame.item<byte>(34, 0, 0));
        Assert.AreEqual((byte)144, frame.item<byte>(35, 0, 1));
        Assert.AreEqual((byte)144, frame.item<byte>(36, 0, 0));
        input[0] = 7.0;
        Assert.AreEqual((byte)1, frame.item<byte>(35, 0, 0), "Returned float vector owns a copy.");
        Assert.ThrowsException<ArgumentException>(() => PongPolicyGradient.PreprocessFrame(np.zeros((210, 160, 3))));
    }

    [TestMethod]
    public void FrameDifferences_ActionsAndSigmoidHaveExplicitBoundaries()
    {
        using var scope = NDScope.Open();
        var current = np.array(new[] { 1.0, 0, 1 });
        var previous = np.array(new[] { 0.0, 1, 1 });
        CollectionAssert.AreEqual(new[] { 0.0, 0, 0 }, PongPolicyGradient.FrameDifference(current, null).ToArray<double>());
        CollectionAssert.AreEqual(new[] { 1.0, -1, 0 }, PongPolicyGradient.FrameDifference(current, previous).ToArray<double>());
        Assert.AreEqual(.5, PongPolicyGradient.Sigmoid(0));
        Assert.AreEqual(0, PongPolicyGradient.Sigmoid(double.NegativeInfinity));
        Assert.AreEqual(1, PongPolicyGradient.Sigmoid(double.PositiveInfinity));
        Assert.AreEqual(2, PongPolicyGradient.SampleAction(.5, .499));
        Assert.AreEqual(3, PongPolicyGradient.SampleAction(.5, .5));
        Assert.AreEqual(.25, PongPolicyGradient.ActionLogGradient(2, .75));
        Assert.AreEqual(-.75, PongPolicyGradient.ActionLogGradient(3, .75));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => PongPolicyGradient.SampleAction(double.NaN, .5));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => PongPolicyGradient.ActionLogGradient(1, .5));
    }

    [TestMethod]
    public void DiscountRewards_ResetAtEachPointAndPreserveColumnShape()
    {
        using var scope = NDScope.Open();
        var rewards = np.array(new[] { 0.0, 0, 1, 0, -1, 0, 0, 1 });
        double[] expected = { .25, .5, 1, -.5, -1, .25, .5, 1 };
        CollectionAssert.AreEqual(expected, PongPolicyGradient.DiscountRewards(rewards, .5).ToArray<double>());
        var column = rewards.reshape(-1, 1);
        var actual = PongPolicyGradient.DiscountRewards(column, .5);
        CollectionAssert.AreEqual(new long[] { 8, 1 }, actual.shape);
        CollectionAssert.AreEqual(expected, actual.ToArray<double>());
        Assert.AreEqual(0L, PongPolicyGradient.DiscountRewards(np.array(Array.Empty<double>())).size);
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => PongPolicyGradient.DiscountRewards(rewards, -1));
    }

    [TestMethod]
    public void Standardization_IsCenteredUnitVarianceAndRejectsPoisoningUpdates()
    {
        using var scope = NDScope.Open();
        var advantages = PongPolicyGradient.StandardizeReturns(np.array(new[] { .99, 1.0, -1 }));
        CollectionAssert.AreEqual(new long[] { 3 }, advantages.shape);
        Assert.AreEqual(.70178357663704, advantages.item<double>(0), 1e-14);
        Assert.AreEqual(.7124166611315407, advantages.item<double>(1), 1e-14);
        Assert.AreEqual(-1.414200237768581, advantages.item<double>(2), 1e-14);
        Assert.AreEqual(0.0, np.mean(advantages, axis: 0).item<double>(), 1e-15);
        Assert.AreEqual(1.0, np.std(advantages, axis: 0).item<double>(), 1e-14);
        Assert.ThrowsException<ArgumentException>(() => PongPolicyGradient.StandardizeReturns(np.zeros(4)));
        Assert.ThrowsException<ArgumentException>(() => PongPolicyGradient.StandardizeReturns(np.array(new[] { 0.0, double.NaN })));
    }

    [TestMethod]
    public void Backward_AllFifteenParametersMatchFiniteDifferences()
    {
        using var scope = NDScope.Open();
        using var model = SmallModel();
        var inputs = Inputs();
        var advantage = PongPolicyGradient.StandardizeReturns(np.array(new[] { .99, 1.0, -1 }));
        int[] actions = { 2, 3, 2 };
        var hiddenRows = new NDArray[3]; var weighted = np.zeros((3, 1));
        for (int t = 0; t < 3; t++)
        {
            var prediction = PongPolicyGradient.Forward(model, inputs[$"{t},:"]);
            hiddenRows[t] = prediction.Hidden;
            weighted[t, 0] = PongPolicyGradient.ActionLogGradient(actions[t], prediction.Probability) * advantage.item<double>(t);
        }
        var hidden = np.vstack(hiddenRows);
        var gradients = PongPolicyGradient.Backward(model, inputs, hidden, weighted);
        int compared = 0;
        foreach (var pair in new[] { (model.W1, gradients.W1), (model.W2, gradients.W2) })
        {
            var flat = pair.Item1.reshape(-1);
            for (long i = 0; i < flat.size; i++)
            {
                double old = flat.item<double>(i), step = 1e-6;
                flat[i] = old + step;
                double plus = LogObjective(model, inputs, actions, advantage);
                flat[i] = old - step;
                double minus = LogObjective(model, inputs, actions, advantage);
                flat[i] = old;
                double numerical = (plus - minus) / (2 * step);
                Assert.IsTrue(double.IsFinite(numerical) && double.IsFinite(pair.Item2.item<double>(i)),
                    $"Finite-difference comparison must not accept a NaN/Infinity at parameter {compared}.");
                Assert.AreEqual(numerical, pair.Item2.item<double>(i), 5e-8, $"parameter {compared}");
                compared++;
            }
        }
        Assert.AreEqual(15, compared);
    }

    [TestMethod]
    public void TenEpisodeBatch_AccumulatesThenUpdatesAndClearsGradients()
    {
        using var scope = NDScope.Open();
        using var model = SmallModel();
        using var trainer = new PongPolicyGradient.Trainer(model);
        var inputs = Inputs(); var rows = new NDArray[3]; var logGrad = np.zeros((3, 1));
        for (int t = 0; t < 3; t++)
        {
            var f = PongPolicyGradient.Forward(model, inputs[$"{t},:"]);
            rows[t] = f.Hidden;
            logGrad[t, 0] = PongPolicyGradient.ActionLogGradient(t == 1 ? 3 : 2, f.Probability);
        }
        var hidden = np.vstack(rows); var rewards = np.array(new[] { 0.0, 1, -1 }).reshape(-1, 1);
        var original = model.W1.copy();
        for (int episode = 1; episode <= 10; episode++)
        {
            Assert.AreEqual(episode == 10, trainer.TrainEpisode(inputs, hidden, logGrad, rewards));
            if (episode < 10) Assert.IsTrue(np.array_equal(original, model.W1));
        }
        Assert.AreEqual(10, trainer.Episodes); Assert.AreEqual(1, trainer.Updates);
        Assert.AreEqual(0.0, np.sum(np.abs(trainer.GradientW1)).item<double>());
        Assert.AreEqual(0.0, np.sum(np.abs(trainer.GradientW2)).item<double>());
        Assert.IsTrue(np.any(trainer.CacheW1 > 0.0));
        Assert.IsFalse(np.array_equal(original, model.W1));
        Assert.AreEqual(0.0, trainer.RunningReward);
    }

    [TestMethod]
    public void ModelAndOptimizerState_OutliveCallerScopesAndCopyInputs()
    {
        PongPolicyGradient.Model model;
        PongPolicyGradient.Trainer trainer;
        using (var scope = NDScope.Open())
        {
            var w1 = np.ones((2, 3)); var w2 = np.array(new[] { .5, -.25 });
            model = new PongPolicyGradient.Model(w1, w2);
            trainer = new PongPolicyGradient.Trainer(model);
            w1[0, 0] = 999.0;
            Assert.AreEqual(1.0, model.W1.item<double>(0, 0));
        }
        using (model)
        using (trainer)
        using (var scope = NDScope.Open())
        {
            var prediction = PongPolicyGradient.Forward(model, np.array(new[] { 1.0, 0, 0 }));
            Assert.IsTrue(prediction.Probability > .5);
            Assert.AreEqual(0.0, trainer.CacheW1.item<double>(0, 0));
            trainer.Accumulate(np.ones((2, 3)), np.ones(2));
            Assert.AreEqual(1.0, trainer.GradientW1.item<double>(0, 0));
        }
    }

    [TestMethod]
    public void FullSourceDimensions_BoundedSyntheticTrainingActuallyUpdatesWeights()
    {
        var result = PongPolicyGradient.RunSyntheticTraining(episodes: 2, stepsPerEpisode: 6, seed: 7);
        Assert.AreEqual(6400, PongPolicyGradient.InputSize);
        Assert.AreEqual(200, PongPolicyGradient.HiddenUnits);
        Assert.AreEqual(2, result.Episodes); Assert.AreEqual(1, result.Updates);
        Assert.IsTrue(double.IsFinite(result.ParameterChangeL2) && result.ParameterChangeL2 > 0);
        Assert.IsTrue(result.LastProbability > 0 && result.LastProbability < 1);
        Assert.IsTrue(System.Math.Abs(result.RunningReward) <= 1);
        // This proves a real update on invented observations/rewards, not Pong competence.
    }

    [TestMethod]
    public void Validation_RejectsIncompatibleNetworkAndOptimizerInputs()
    {
        using var scope = NDScope.Open();
        using var model = SmallModel();
        Assert.ThrowsException<ArgumentException>(() => new PongPolicyGradient.Model(np.ones((2, 3)), np.ones(3)));
        Assert.ThrowsException<ArgumentException>(() => PongPolicyGradient.Forward(model, np.ones(3)));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => new PongPolicyGradient.Trainer(model, batchSize: 0));
        Assert.ThrowsException<ArgumentException>(() => PongPolicyGradient.RmsPropUpdate(np.ones(3), np.ones(2), np.zeros(3)));
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => PongPolicyGradient.RmsPropUpdate(np.ones(3), np.ones(3), np.zeros(3), decayRate: 1));
    }

    private static PongPolicyGradient.Model SmallModel()
    {
        using var w1 = np.array(new double[,] { { .2, -.3, .4, .1 }, { -.5, .2, .1, .3 }, { .1, .3, -.2, .4 } });
        using var w2 = np.array(new[] { .2, -.4, .3 });
        return new PongPolicyGradient.Model(w1, w2);
    }

    private static NDArray Inputs() => np.array(new double[,] { { 1, -.5, .25, 0 }, { -.3, .5, 1, -.25 }, { .2, .1, -.5, .7 } });

    private static double LogObjective(PongPolicyGradient.Model model, NDArray inputs, int[] actions, NDArray advantages)
    {
        using var scope = NDScope.Open();
        double objective = 0;
        for (int t = 0; t < actions.Length; t++)
        {
            var (p, _) = PongPolicyGradient.Forward(model, inputs[$"{t},:"]);
            objective += advantages.item<double>(t) * System.Math.Log(actions[t] == 2 ? p : 1 - p);
        }
        return objective;
    }
}
