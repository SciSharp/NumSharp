using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// DefaultEngine.Evaluate.Text.cs — the per-call half of np.evaluate over text (docs/plans/evaluate-text.md)
// =============================================================================
//
// A program whose inputs include text carries a plan instead of a kernel (NDExpr.Text.cs): the stages that compute
// on text, in evaluation order, and the numeric remainder over their results. A call runs each stage as the engine's
// own text call over whole arrays — NumPy's statement for that node, its errors included — then hands the numeric
// inputs and the stage results to the remainder's program, resolved through the positional program cache (its dtype
// signature and 0-d parameters are those of THIS call's stage results), which EvaluateCore runs with the call's
// keywords exactly as for any numeric tree. The stage results are this call's temporaries and are released here,
// except a bare stage result that is the evaluation's whole answer.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Evaluates a program over text inputs: its stages, then its numeric remainder (see the file header).
        /// </summary>
        /// <param name="program">A program with a <see cref="NDExprProgram.TextPlan"/>.</param>
        /// <param name="inputs">The call's inputs, in the program's input order (text arrays included).</param>
        /// <param name="out">The caller's result array, or null for a fresh result.</param>
        /// <param name="options">The call's resolved keywords.</param>
        /// <returns>The evaluation's result: <paramref name="out"/> itself, the remainder's fresh result, or — for a tree
        ///     that is one comparison over text with no result-shaping keyword — the comparison's own result array.</returns>
        /// <exception cref="NotSupportedException">A reduction tree given <c>dtype=</c>, <c>order=</c> or <c>where=</c>
        ///     (checked before any text is read).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text <paramref name="out"/> or a text <c>dtype=</c>
        ///     (a text result, which this phase does not produce).</exception>
        /// <exception cref="TypeError">A stage NumPy refuses (S with U, text with a number), with NumPy's text.</exception>
        /// <exception cref="IncorrectShapeException">A stage's operands, or the remainder's, do not broadcast.</exception>
        /// <exception cref="ValueError">A StringDType ordering meets a <c>None</c>-like NA, or the remainder's own checks
        ///     fail (NumPy's texts).</exception>
        /// <exception cref="ArgumentException">The remainder's <paramref name="out"/> / <c>where=</c> checks fail.</exception>
        /// <remarks>
        ///     <para>
        ///     Ordering. NumPy evaluates the same tree as statements: each comparison over its own operands, then the numeric
        ///     ufuncs over the results. So the stages run first, in evaluation order (a stage's broadcast or refusal is raised
        ///     before any later statement runs), and the remainder's <c>out=</c> / <c>where=</c> checks come after them, as
        ///     NumPy's final statement makes them. Only the keyword rules that depend on the tree alone run first.
        ///     </para>
        ///     <para>
        ///     Ownership. Every stage result is a fresh array of this call. The remainder writes a fresh result or the
        ///     caller's <paramref name="out"/>, never into a stage result, so all of them are released when the call ends —
        ///     except the bare stage result handed out as the answer.
        ///     </para>
        /// </remarks>
        private NDArray EvaluateTextPlan(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options)
        {
            var plan = program.TextPlan;
            ThrowIfReductionKeywords(plan.IsReduction, options);
            // Every E1 result is numeric (a bool or a number computed from the comparisons); a text result is phase E3.
            if (@out is not null && NDStringUfuncs.IsText(@out))
                throw NDExprTextLowering.NotYet("a text result (a text out=)" /* plan: E3, text results */);
            if (options.Dtype == NPTypeCode.String)
                throw NDExprTextLowering.NotYet("a text result (a text dtype=)" /* plan: E3, text results */);

            var results = new NDArray[plan.Stages.Length];
            NDArray handedOut = null;
            try
            {
                for (int s = 0; s < results.Length; s++)
                    results[s] = plan.Stages[s].Run(inputs);

                // np.evaluate(NDExpr.Equal(a, b)): the comparison's own array — NumPy's np.equal result, dtype, shape and
                // layout — is the answer; copying it through the remainder would only add a pass. Any keyword that shapes the
                // result (out=, where=, dtype=, order=) takes the remainder, which applies it.
                if (@out is null && !options.HasWhere && !options.HasDtype && !options.HasExplicitOrder
                    && plan.IsBareStageResult(out int stage))
                    return handedOut = results[stage];

                var remainderInputs = plan.RemainderInputs(inputs, results);
                var remainder = plan.Remainder.GetProgram(remainderInputs);
                return EvaluateCore(remainder, remainderInputs, @out, options);
            }
            finally
            {
                // The stage results are this call's temporaries (the remainder never writes into them); the one handed out
                // as the answer belongs to the caller now.
                foreach (var r in results)
                    if (r is not null && !ReferenceEquals(r, handedOut))
                        r.Dispose();
            }
        }
    }
}
