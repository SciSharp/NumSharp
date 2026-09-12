using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;
using Python.Runtime;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistLearningLiveParityTests : InteropTestBase
{
    [TestMethod]
    public void Catch_GridTransitions_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        using var game = new CatchReinforcementLearning.CatchGame(5);
        game.Reset(2, 2);
        ExportTo("initial", game.State());
        PyExec("state=initial.copy()\ngrid=5");
        foreach (int action in new[] { 0, 2, 1, 1 })
        {
            var step = game.Act(action);
            using var observation = step.Observation;
            PyExec($"state[0,0]+=1\nstate[0,2]=np.clip(state[0,2]+{action - 1},1,grid-1)\ncanvas=np.zeros((grid,grid))\ncanvas[state[0,0],state[0,1]]=1\ncanvas[-1,state[0,2]-1:state[0,2]+2]=1\nobservation=canvas.reshape(1,-1)");
            using (Gil())
            {
                using var expected = Scope.Eval("observation");
                GistParity.AssertExact(observation, expected, "catch grid");
            }
            Assert.AreEqual(PyBool("state[0,0]==grid-1"), step.GameOver);
            Assert.AreEqual((int)PyLong("int((1 if abs(state[0,1]-state[0,2])<=1 else -1) if state[0,0]==grid-1 else 0)"), step.Reward);
        }
    }

    [TestMethod]
    public void Replay_BellmanTargets_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        using var replay = new CatchReinforcementLearning.ReplayMemory();
        var a = np.array(new double[,] { { 1, 2 } });
        var b = np.array(new double[,] { { 3, 4 } });
        replay.Remember(a, 1, .5, b, false);
        replay.Remember(b, 2, -1, a, true);
        NDArray Predict(NDArray input) => np.concatenate(new[] { input, np.ones((1, 1)) }, axis: 1);
        var (inputs, targets) = replay.GetBatchAt(Predict, 3, 0, 1);
        ExportTo("a", a); ExportTo("b", b);
        PyExec("inputs=np.vstack([a,b])\ntargets=np.concatenate([inputs,np.ones((2,1))],axis=1)\nnext_q=np.concatenate([b,np.ones((1,1))],axis=1)\ntargets[0,1]=.5+.9*np.max(next_q[0])\ntargets[1,2]=-1");
        using (Gil())
        {
            using var expectedInputs = Scope.Eval("inputs");
            using var expectedTargets = Scope.Eval("targets");
            GistParity.AssertExact(inputs, expectedInputs, "replay inputs");
            GistParity.AssertExact(targets, expectedTargets, "replay targets");
        }
    }

    [TestMethod]
    public void Nes_SeededNoiseAndOneStep_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var random = np.random.RandomState(0);
        var noise = random.randn(4, 2);
        var weights = np.array(new[] { .25, -.5 });
        var solution = np.array(new[] { .5, .25 });
        var rewards = NaturalEvolutionStrategies.Rewards(weights, noise, x => NaturalEvolutionStrategies.QuadraticReward(x, solution));
        var next = NaturalEvolutionStrategies.Step(weights, noise, rewards, alpha: .05);
        ExportTo("w", weights); ExportTo("solution", solution); ExportTo("noise", noise);
        PyExec("rng=np.random.RandomState(0)\nexpected_noise=rng.randn(4,2)\nr=np.array([-np.sum(np.square(solution-(w+.1*n))) for n in noise])\na=(r-np.mean(r))/np.std(r)\nnext_w=w+.05/(4*.1)*np.dot(noise.T,a)");
        using (Gil())
        {
            using var expectedNoise = Scope.Eval("expected_noise");
            using var expectedRewards = Scope.Eval("r");
            using var expectedNext = Scope.Eval("next_w");
            GistParity.AssertExact(noise, expectedNoise, "NES RNG");
            GistParity.AssertExact(rewards, expectedRewards, "NES rewards");
            GistParity.AssertExact(next, expectedNext, "NES step");
        }
    }

    [TestMethod]
    public void Nes_Complete300IterationOptimization_ByteExactLiveNumpy()
    {
        using var solution = np.array(new[] { .5, .1, -.3 });
        using var actual = NaturalEvolutionStrategies.OptimizeQuadratic(solution);
        ExportTo("solution", solution);
        PyExec("""
            rng=np.random.RandomState(0)
            weights=rng.randn(3)
            for iteration in range(300):
                noise=rng.randn(50,3)
                rewards=np.array([-np.sum(np.square(solution-(weights+.1*row))) for row in noise])
                advantages=(rewards-np.mean(rewards))/np.std(rewards)
                weights=weights+.001/(50*.1)*np.dot(noise.T,advantages)
            """);
        using (Gil())
        {
            using var expected = Scope.Eval("weights");
            Console.WriteLine($"NES final C#={Convert.ToHexString(ByteContract.NsBytes(actual))}, Python={Convert.ToHexString(expected.bytes_c())}");
            GistParity.AssertUlps(actual, expected, 0, "NES 300-step final weights");
        }
    }

    [DataTestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void Histogram_BucketsAndMetadata_ExactLiveNumpy(bool constant)
    {
        using var scope = NDScope.Open();
        var values = constant ? np.array(new[] { 2.0, 99, 2, 99, 2, 99 })["::2"]
                              : np.array(new[] { 0.0, 99, 1, 99, 1, 99, 2, 99, 3, 99, 4, 99 })["::2"];
        using var actual = TensorboardHistogram.Create(values, bins: 4);
        ExportTo("values", values);
        PyExec("counts,edges=np.histogram(values,bins=4)\ncounts=counts.astype(np.int64)\nlimits=edges[1:]\nstats=np.array([values.min(),values.max(),values.size,values.sum(),np.sum(values**2)],dtype=np.float64)");
        var stats = np.array(new[] { actual.Min, actual.Max, (double)actual.Num, actual.Sum, actual.SumSquares });
        using (Gil())
        {
            using var counts = Scope.Eval("counts");
            using var limits = Scope.Eval("limits");
            using var expectedStats = Scope.Eval("stats");
            GistParity.AssertExact(actual.Buckets, counts, "histogram counts");
            GistParity.AssertExact(actual.BucketLimits, limits, "histogram bucket limits");
            GistParity.AssertExact(stats, expectedStats, "histogram statistics");
        }
    }

    [TestMethod]
    public void Histogram_Float32EdgesAndStatistics_ExactLiveNumpy()
    {
        using var scope = NDScope.Open();
        var values = np.array(new[] { .1f, .2f, .2f, .3f, .4f, .5f });
        using var actual = TensorboardHistogram.Create(values, bins: 4);
        ExportTo("values", values);
        PyExec("counts,edges=np.histogram(values,bins=4)\ncounts=counts.astype(np.int64)\nlimits=edges[1:]\nstats=np.array([values.min(),values.max(),values.size,values.sum(),np.sum(values**2)],dtype=np.float64)");
        var stats = np.array(new[] { actual.Min, actual.Max, (double)actual.Num, actual.Sum, actual.SumSquares });
        using (Gil())
        {
            using var counts = Scope.Eval("counts");
            using var limits = Scope.Eval("limits");
            using var expectedStats = Scope.Eval("stats");
            GistParity.AssertExact(actual.Buckets, counts, "float32 histogram counts");
            GistParity.AssertExact(actual.BucketLimits, limits, "float32 histogram limits");
            GistParity.AssertExact(stats, expectedStats, "float32 histogram statistics");
        }
    }
}
