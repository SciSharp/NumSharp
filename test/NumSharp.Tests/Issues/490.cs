using System.Linq;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #490, "np.random.choice with replace: false produces duplicates"
    ///     (https://github.com/SciSharp/NumSharp/issues/490): since 0.80.0 the legacy <c>choice</c> follows NumPy's
    ///     <c>RandomState.choice</c>, so a draw without replacement never repeats and a seeded draw is NumPy's stream.
    /// </summary>
    /// <remarks>
    ///     <c>[DoNotParallelize]</c>: the answer's example seeds the GLOBAL <c>np.random</c> and expects NumPy's stream, and
    ///     the parallel phase shares that singleton with every other test that draws from it (TestParallelization's rule).
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class Issue490
    {
        /// <summary>
        ///     The issue's own call, <c>choice(71, 40, replace: false)</c>, returns 40 distinct values in [0, 71) for every
        ///     one of 200 seeds; before 0.80.0 <c>replace: false</c> was ignored and draws repeated.
        /// </summary>
        [TestMethod]
        public void ChoiceWithoutReplacement_NeverRepeats()
        {
            for (int seed = 0; seed < 200; seed++)
            {
                np.random.seed(seed);
                var draw = np.random.choice(71, new Shape(40), replace: false).ToArray<long>();
                draw.Distinct().Count().Should().Be(40, $"seed {seed} must draw 40 distinct values");
                draw.Should().OnlyContain(v => v >= 0 && v < 71);
            }
        }

        /// <summary>
        ///     <c>seed(42)</c> then <c>choice(71, 40, replace: false)</c> gives NumPy 2.4.2's exact 40 values (the answer's
        ///     example). The values are int64: NumSharp's legacy integers are LP64 C <c>long</c>, as on Linux NumPy.
        /// </summary>
        [TestMethod]
        public void ChoiceWithoutReplacement_Seed42_IsNumPysStream()
        {
            np.random.seed(42);
            var r = np.random.choice(71, new Shape(40), replace: false);

            r.typecode.Should().Be(NPTypeCode.Int64);
            r.ToArray<long>().Should().Equal(
                22, 0, 49, 4, 55, 18, 10, 33, 46, 12, 31, 9, 61, 5, 53, 30, 59, 35, 58, 42,
                16, 34, 44, 28, 7, 54, 50, 45, 47, 19, 57, 40, 25, 39, 13, 70, 3, 17, 38, 8);
        }

        /// <summary>
        ///     Re-seeding reproduces the draw (the issue also reported that "setting the seed doesn't work").
        /// </summary>
        [TestMethod]
        public void ChoiceWithoutReplacement_SameSeed_SameDraw()
        {
            np.random.seed(7);
            var first = np.random.choice(71, new Shape(40), replace: false).ToArray<long>();
            np.random.seed(7);
            var second = np.random.choice(71, new Shape(40), replace: false).ToArray<long>();

            second.Should().Equal(first);
        }
    }
}
