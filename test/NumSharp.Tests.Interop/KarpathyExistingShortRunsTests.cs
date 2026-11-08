using System;
using System.Globalization;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Examples.Gist;

namespace NumSharp.Tests.Interop;

[TestClass]
public class KarpathyExistingShortRunsTests : InteropTestBase
{
    /// <summary>
    /// Twelve NES updates driven by the hash-pinned original objective <c>f</c>. All 13 states and rewards
    /// must be byte-identical to NumSharp's <see cref="NaturalEvolutionStrategies.OptimizeQuadratic"/>.
    /// </summary>
    /// <remarks>
    /// The seeded Gaussians come from <c>legacy_randn</c> (<see cref="InteropTestBase.DefineLegacyRandn"/>).
    /// On x64 that is NumPy's own <c>rng.randn</c>; on a NumPy that fuses <c>legacy_gauss</c> (arm64) it is
    /// the literal evaluation of the same stream. With raw <c>rng.randn</c> this passed on macos-latest only
    /// because twelve steps end before the fused stream's first visible effect. The 300-step trace of the
    /// same loop diverged at iteration 135.
    /// </remarks>
    [TestMethod, TestCategory("KarpathyShortRun"), TestCategory("KarpathyByteParity")]
    public void NesOriginalObjective_TwelveCompleteUpdates_AllStatesByteExact()
    {
        using var solution = np.array(new[] { .5, .1, -.3 });
        ExportTo("solution", solution);
        KarpathyOriginalSource.Load(Scope, "nes");
        DefineLegacyRandn();
        PyExec("""
            original['solution']=solution
            rng=np.random.RandomState(0); w=legacy_randn(rng,3); trace=np.empty((13,4))
            for step in range(13):
                trace[step,:3]=w; trace[step,3]=original['f'](w)
                if step==12: break
                noise=legacy_randn(rng,50,3)
                rewards=np.array([original['f'](w+.1*row) for row in noise])
                advantages=(rewards-np.mean(rewards))/np.std(rewards)
                w=w+.001/(50*.1)*np.dot(noise.T,advantages)
            """);
        using var actual = np.zeros((13, 4), dtype: np.float64);
        using var result = NaturalEvolutionStrategies.OptimizeQuadratic(solution, iterations: 12, progress: (step, weights, reward) =>
        {
            for (int column = 0; column < 3; column++) actual[step, column] = weights.item<double>(column);
            actual[step, 3] = reward;
        });
        using (Gil()) { using var expected = Scope.Eval("trace"); GistParity.AssertExact(actual, expected, "NES original short trajectory"); }
        Console.WriteLine("KARPATHY_ORIGINAL_AND_BYTES NES: 12updates,13states,52float64 values; original objective, byte-exact trajectory.");
    }

    [DataTestMethod, TestCategory("KarpathyShortRun"), TestCategory("KarpathyByteParity")]
    [DataRow(false)] [DataRow(true)]
    public void WalkOriginalSlerp_FiveFullLatentFrames_ByteExact(bool nearParallel)
    {
        using var scope = NDScope.Open();
        var rng = np.random.RandomState(1337);
        var a = rng.randn(1, 4, 64, 64).astype(np.float32);
        var b = nearParallel ? (a * np.array(1.001f)) : rng.randn(1, 4, 64, 64).astype(np.float32);
        ExportTo("a", a); ExportTo("b", b);
        KarpathyOriginalSource.Load(Scope, "walk");
        for (int frame = 0; frame < 5; frame++)
        {
            double t = frame / 4.0;
            PyExec($"expected=original['slerp']({t.ToString("R", CultureInfo.InvariantCulture)},a,b)");
            using var actual = StableDiffusionWalk.Slerp(t, a, b);
            using (Gil()) { using var expected = Scope.Eval("expected"); GistParity.AssertExact(actual, expected, $"original Slerp frame {frame}, nearParallel={nearParallel}"); }
        }
        Console.WriteLine($"KARPATHY_ORIGINAL_AND_BYTES walk: frames=5, shape1x4x64x64, nearParallel={nearParallel}; dispatch-flag repair only, no diffusion model.");
    }
}
