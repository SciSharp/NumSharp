using System.Collections.Generic;
using System.Text.Json;
using NumSharp.Tests.Fuzz;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's op for section <c>evaltext</c>: np.evaluate over text (docs/plans/evaluate-text.md, phases E1
    ///     and E2). The case's tree is the numeric evaluate tier's prefix grammar (plus the <c>ls:</c> Python-str literal and
    ///     numpy.strings' function names), so it is built and evaluated by that tier's own replay (<see cref="OpRegistry"/>'s
    ///     evaluate parser, with its <c>reduce</c> / <c>dtype</c> / <c>order</c> parameters), here over the string corpus'
    ///     operands (S, U and StringDType views rebuilt exactly).
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>Replays one np.evaluate case, or reports that <paramref name="op"/> is not this section's.</summary>
        /// <param name="op">The corpus op key.</param>
        /// <param name="p">The case parameters (<c>expr</c>, and optionally <c>reduce</c>, <c>dtype</c>, <c>order</c>).</param>
        /// <param name="ops">The reconstructed operands, <c>in0</c> … in order.</param>
        /// <param name="result">np.evaluate's result.</param>
        /// <returns>True when <paramref name="op"/> is <c>evaluate</c>.</returns>
        private static bool TryInvokeEvaluate(string op, Dictionary<string, JsonElement> p, NDArray[] ops, out object result)
        {
            if (op != "evaluate")
            {
                result = null;
                return false;
            }

            result = OpRegistry.EvaluateFromCorpus(p, ops);
            return true;
        }
    }
}
