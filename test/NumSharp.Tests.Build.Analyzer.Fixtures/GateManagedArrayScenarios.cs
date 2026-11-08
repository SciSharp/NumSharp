namespace NumSharp.Tests.Build.Analyzer.Fixtures
{
    /// <summary>
    ///     OPEN BUG (found by Bigtree.Algorithm's upgrade to NumSharp 0.80.0, 2026-11-08): the <c>[NDScoped]</c> target
    ///     gate refuses a method that returns a managed array of a type that can never hold an NDArray (<c>double[]</c>,
    ///     <c>int[]</c>, <c>string[]</c>, <c>double[][]</c>) with NDW003, "an unsupported carrier the weaver cannot see
    ///     every NDArray through". Such a return carries no NDArray, so it is a value like <c>double</c> and must stay
    ///     CLEAN: every method here is untagged, and <c>GateAnalyzerTests.ManagedArrayReturns_AreNotGated</c> fails while
    ///     the gate treats them as carriers. Consumer workaround today: open the scope by hand
    ///     (<c>using var scope = NDScope.Open();</c>).
    /// </summary>
    public abstract class GateManagedArrayScenarios
    {
        /// <summary>Bigtree's shape: a NumSharp computation whose result leaves as a managed <c>double[]</c> table.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>The doubled values, copied out.</returns>
        [NDScoped]
        public static double[] ScoresAsDoubles(NDArray a)
            => (a * 2.0).ToArray<double>();

        /// <summary>A computation whose result leaves as <c>int[]</c>.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>1 where an element is positive, else 0.</returns>
        [NDScoped]
        public static int[] CountsAsInts(NDArray a)
            => (a > 0).astype(np.int32).ToArray<int>();

        /// <summary>A computation whose result leaves as <c>string[]</c>.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>The text of the shifted array.</returns>
        [NDScoped]
        public static string[] Names(NDArray a)
            => new[] { (a + 1.0).ToString() };

        /// <summary>A computation whose result leaves as a jagged <c>double[][]</c>.</summary>
        /// <param name="a">Any array.</param>
        /// <returns>One row: the shifted values, copied out.</returns>
        [NDScoped]
        public static double[][] Jagged(NDArray a)
            => new[] { (a - 1.0).ToArray<double>() };
    }
}
