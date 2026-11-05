using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// DefaultEngine.Evaluate.Text.cs — the per-call half of np.evaluate over text (docs/plans/evaluate-text.md)
// =============================================================================
//
// A program whose inputs include text (or that holds a numpy.strings node) carries a plan instead of a kernel
// (NDExpr.Text.cs): the stages that compute on text, in evaluation order, and the numeric remainder over their results.
// A call resolves each stage's operands — an input as it is, a literal as NumPy's Python scalar, an earlier stage's
// result, a computed numeric sub-tree evaluated here into the temporary NumPy's unfused sequence would hold — runs the
// stage's statement (the engine's own text call over whole arrays: NumPy's statement for that node, its errors
// included), then hands the numeric inputs and the stage results to the remainder's program, resolved through the
// positional program cache (its dtype signature and 0-d parameters are those of THIS call's stage results), which
// EvaluateCore runs with the call's keywords exactly as for any numeric tree. The stage results are this call's
// temporaries and are released here, except a bare stage result that is the evaluation's whole answer.
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
        ///     that is one stage with no result-shaping keyword — the stage's own result array (NumPy's statement's).</returns>
        /// <exception cref="NotSupportedException">A reduction tree given <c>dtype=</c>, <c>order=</c> or <c>where=</c>
        ///     (checked before any text is read).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text <paramref name="out"/> or a text <c>dtype=</c>
        ///     (a text result, which np.evaluate does not produce).</exception>
        /// <exception cref="TypeError">A stage NumPy refuses (S with U, text with a number, a numpy.strings function over
        ///     the wrong class), with NumPy's text.</exception>
        /// <exception cref="IncorrectShapeException">A stage's operands, or the remainder's, do not broadcast.</exception>
        /// <exception cref="ValueError">A StringDType NA a stage's rule refuses, a parse a cast stage refuses, an
        ///     <c>index</c> / <c>rindex</c> without a match, or the remainder's own checks (NumPy's texts).</exception>
        /// <exception cref="ArgumentException">The remainder's <paramref name="out"/> / <c>where=</c> checks fail, or a
        ///     search's <c>start</c> / <c>end</c> cannot reach int64.</exception>
        /// <remarks>
        ///     <para>
        ///     Ordering. NumPy evaluates the same tree as statements: each text node over its own operands, then the numeric
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
            // Every result np.evaluate computes from text is numeric (a bool, a count, a parsed number, a value the remainder
            // computes from them); a text result is phase E3.
            if (@out is not null && NDStringUfuncs.IsText(@out))
                throw NDExprTextLowering.NotYet("a text result (a text out=)" /* plan: E3, text results */);
            if (options.Dtype == NPTypeCode.String)
                throw NDExprTextLowering.NotYet("a text result (a text dtype=)" /* plan: E3, text results */);

            var results = new NDArray[plan.Stages.Length];
            NDArray handedOut = null;
            try
            {
                for (int s = 0; s < results.Length; s++)
                    results[s] = RunTextStage(plan, plan.Stages[s], inputs, results);

                // np.evaluate(NDExpr.Equal(a, b)), np.evaluate(NDExpr.StrLen(a)): the stage's own array — NumPy's statement
                // result, dtype, shape and layout — is the answer; copying it through the remainder would only add a pass.
                // Any keyword that shapes the result (out=, where=, dtype=, order=) takes the remainder, which applies it.
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

        /// <summary>
        ///     Runs one stage for one call: resolves its operands (<see cref="ResolveStageOperand"/>), runs its statement and
        ///     releases the operand arrays this call made.
        /// </summary>
        /// <param name="plan">The program's text plan.</param>
        /// <param name="stage">The stage.</param>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call so far (the stage reads only earlier ones).</param>
        /// <returns>The stage's fresh result (owned by the caller).</returns>
        /// <exception cref="System.Exception">Whatever the statement or a computed operand raises (NumPy's errors; see
        ///     <see cref="EvaluateTextPlan"/>): the operands made here are released first.</exception>
        private NDArray RunTextStage(NDExprTextPlan plan, NDExprTextStage stage, NDArray[] inputs, NDArray[] results)
        {
            var operands = stage.Operands;
            var args = new NDArray[operands.Length];
            // Per operand, up to two arrays this call made: a literal's 0-d, or a computed operand's view and its buffer.
            var made = new NDArray[2 * operands.Length];
            NDArray result = null;
            try
            {
                for (int i = 0; i < operands.Length; i++)
                    args[i] = ResolveStageOperand(plan, operands[i], inputs, results, out made[2 * i], out made[2 * i + 1]);
                return result = stage.Execute(args);
            }
            finally
            {
                // A view before the buffer it views (made[2i] before made[2i+1]); the statement's result is never one of
                // them (every stage returns a fresh array), but it is checked so a future statement that hands an operand
                // back cannot have it released under the caller.
                for (int i = 0; i < made.Length; i++)
                    if (made[i] is not null && !ReferenceEquals(made[i], result))
                        made[i].Dispose();
            }
        }

        /// <summary>
        ///     One stage operand for this call: an input as it is, a literal as the 0-d operand NumPy's function sees (a Python
        ///     scalar), an omitted optional argument as null (the function's default), an earlier stage's result, or a
        ///     computed numeric sub-tree evaluated over exactly the remainder inputs it reads into the fresh array NumPy's own
        ///     ufunc would allocate for it (<see cref="MaterializeChildNumPyLayout"/>: the K-order layout of its operands), so
        ///     the stage meets NumPy's temporary.
        /// </summary>
        /// <param name="plan">The program's text plan (it maps a remainder input to an input or a stage result).</param>
        /// <param name="operand">The operand's description.</param>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call so far.</param>
        /// <param name="owned1">Receives an array this call created and must dispose (a literal, a constant's value, or the
        ///     computed view), or null.</param>
        /// <param name="owned2">Receives the buffer behind a view in <paramref name="owned1"/>, or null.</param>
        /// <returns>The operand array, or null for an omitted optional argument.</returns>
        /// <exception cref="IncorrectShapeException">A computed sub-tree's operands do not broadcast.</exception>
        private NDArray ResolveStageOperand(NDExprTextPlan plan, in NDExprStageOperand operand, NDArray[] inputs,
                                           NDArray[] results, out NDArray owned1, out NDArray owned2)
        {
            owned1 = owned2 = null;
            switch (operand.Kind)
            {
                case NDExprStageOperandKind.Input:
                    return inputs[operand.Index];
                case NDExprStageOperandKind.Literal:
                    return owned1 = operand.Value.ToOperandArray();
                case NDExprStageOperandKind.Absent:
                    return null;
                case NDExprStageOperandKind.StageResult:
                    return results[operand.Index];
            }

            if (operand.Inputs.Length == 0)
            {
                // A constant-only sub-tree reads no array, and a program needs one to bind: it is evaluated over a 0-d dummy it
                // never reads (a 0-d result), and marked as a Python scalar when NumPy would see one.
                using var dummy = NDArray.Scalar(false);
                var one = new[] { dummy };
                var constant = EvaluateCore(operand.Tree.GetProgram(one), one, null);
                owned1 = operand.WeakConstant ? NDArray.MarkAsCSharpLiteral(constant) : constant;
                return owned1;
            }

            var ops = new NDArray[operand.Inputs.Length];
            for (int i = 0; i < ops.Length; i++)
                ops[i] = plan.RemainderInput(operand.Inputs[i], inputs, results);
            var childProgram = operand.Tree.GetProgram(ops);
#pragma warning disable NDW012 // not a leak: IteratorOperands hands back the call's own operands (a subset of ops), which the caller holds
            var view = MaterializeChildNumPyLayout(childProgram, ops, childProgram.IteratorOperands(ops), out var buffer);
#pragma warning restore NDW012
            owned1 = view;
            if (!ReferenceEquals(view, buffer))
                owned2 = buffer;
            return view;
        }
    }
}
