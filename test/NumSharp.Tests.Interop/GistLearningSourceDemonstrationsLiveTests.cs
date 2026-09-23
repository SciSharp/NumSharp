using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Interop;

[TestClass]
public class GistLearningSourceDemonstrationsLiveTests : InteropTestBase
{
    /// <summary>
    /// Every NES state and reward across the original 300 updates, matched against live NumPy on the x64
    /// reference architecture. On arm64 it is inconclusive: the macos-latest runner measured a 1-ULP drift
    /// in the 301-state trace (element 542), the cross-architecture class
    /// <see cref="InteropTestBase.SkipByteExactOnArm64"/> documents.
    /// </summary>
    [TestMethod]
    public void Nes_EveryStateAndRewardAcrossOriginal300Updates_MatchesLiveNumpy()
    {
        SkipByteExactOnArm64("Nes_EveryStateAndRewardAcrossOriginal300Updates (1-ULP trace drift measured on macos-latest arm64)");
        using var solution = np.array(new[] { .5, .1, -.3 });
        using var trajectory = np.zeros((301, 4), dtype: np.float64);
        using var final = NaturalEvolutionStrategies.OptimizeQuadratic(solution, progress: (iteration, weights, reward) =>
        {
            for (int j = 0; j < 3; j++) trajectory[iteration, j] = weights.item<double>(j);
            trajectory[iteration, 3] = reward;
        });
        ExportTo("solution", solution);
        PyExec("""
            rng=np.random.RandomState(0)
            weights=rng.randn(3)
            trace=np.empty((301,4))
            for i in range(301):
                trace[i,:3]=weights
                trace[i,3]=-np.sum(np.square(solution-weights))
                if i==300: break
                noise=rng.randn(50,3)
                rewards=np.array([-np.sum(np.square(solution-(weights+.1*row))) for row in noise])
                a=(rewards-rewards.mean())/rewards.std()
                weights=weights+.001/(50*.1)*np.dot(noise.T,a)
            """);
        using (Gil())
        {
            using var expected = Scope.Eval("trace");
            GistParity.AssertUlps(trajectory, expected, 0, "NES full 301-state/reward trace");
        }
    }

    [TestMethod]
    public void Catch_ThousandEpisodeNumericalLoopAndSampledReplay_ExactLiveNumpy()
    {
        // No neural-network training is claimed. Both sides use the same fixed Q values.
        using var q = np.array(new double[,] { { .25, .5, -.25 } });
        using var trace = np.zeros((9000, 6), dtype: np.int64);
        using var replay = new CatchReinforcementLearning.ReplayMemory(500);
        using var game = new CatchReinforcementLearning.CatchGame(10);
        var random = np.random.RandomState(7);
        int at = 0;
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
                using var state = game.State();
                trace[at, 0] = (long)episode;
                for (int j = 0; j < 3; j++) trace[at, j + 1] = state.item<long>(0, j);
                trace[at, 4] = (long)action;
                trace[at++, 5] = (long)step.Reward;
            }
        }
        var (inputs, targets) = replay.GetBatch(_ => q.copy(), 3, 50, np.random.RandomState(11));
        using (inputs) using (targets)
        {
            ExportTo("q", q);
            PyExec("""
                rng=np.random.RandomState(7)
                history=[]
                memory=[]
                def draw(row,fruit,basket):
                    image=np.zeros((10,10))
                    image[row,fruit]=1
                    image[-1,basket-1:basket+2]=1
                    return image.reshape(1,100)
                for episode in range(1000):
                    fruit=rng.randint(0,9)
                    basket=rng.randint(1,8)
                    for row in range(1,10):
                        before=draw(row-1,fruit,basket)
                        action=int(rng.randint(0,3)) if rng.rand()<=.1 else int(np.argmax(q[0]))
                        basket=int(np.clip(basket+action-1,1,9))
                        reward=(1 if abs(fruit-basket)<=1 else -1) if row==9 else 0
                        after=draw(row,fruit,basket)
                        history.append([episode,row,fruit,basket,action,reward])
                        memory.append((before,action,reward,after,row==9))
                        if len(memory)>500: del memory[0]
                history=np.array(history,dtype=np.int64)
                samples=np.random.RandomState(11).randint(0,len(memory),size=50)
                inputs=np.zeros((50,100)); targets=np.zeros((50,3))
                for i,index in enumerate(samples):
                    state,action,reward,next_state,terminal=memory[index]
                    inputs[i:i+1]=state
                    targets[i]=q[0]
                    targets[i,action]=reward if terminal else reward+.9*np.max(q[0])
                """);
            using (Gil())
            {
                using var expectedTrace = Scope.Eval("history");
                using var expectedInputs = Scope.Eval("inputs");
                using var expectedTargets = Scope.Eval("targets");
                GistParity.AssertExact(trace, expectedTrace, "Catch all 9000 actions/states/rewards");
                GistParity.AssertExact(inputs, expectedInputs, "Catch sampled replay inputs");
                GistParity.AssertExact(targets, expectedTargets, "Catch sampled Bellman targets");
            }
        }
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Double)]
    [DataRow(NPTypeCode.Single)]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.Int64)]
    public void Histogram_SourceDefaultBinsEveryField_ExactLiveNumpy(NPTypeCode dtype)
    {
        using var scope = NDScope.Open();
        var values = np.arange(4096).astype(dtype).reshape(64, 64)["::-1,::2"];
        using var actual = TensorboardHistogram.Create(values);
        ExportTo("v", values);
        PyExec("counts,edges=np.histogram(v,bins=1000)\ncounts=counts.astype(np.int64)\nlimits=edges[1:]\nstats=np.array([v.min(),v.max(),v.size,v.sum(),np.sum(v**2)],dtype=np.float64)");
        var stats = np.array(new[] { actual.Min, actual.Max, (double)actual.Num, actual.Sum, actual.SumSquares });
        using (Gil())
        {
            using var expectedCounts = Scope.Eval("counts");
            using var expectedLimits = Scope.Eval("limits");
            using var expectedStats = Scope.Eval("stats");
            GistParity.AssertExact(actual.Buckets, expectedCounts, "default histogram buckets");
            GistParity.AssertExact(actual.BucketLimits, expectedLimits, "default histogram limits");
            GistParity.AssertExact(stats, expectedStats, "default histogram statistics");
        }
    }
}
