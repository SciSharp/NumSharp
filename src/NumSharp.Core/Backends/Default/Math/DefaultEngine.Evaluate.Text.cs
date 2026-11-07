using System;
using NumSharp.Backends.Iteration;

// =============================================================================
// DefaultEngine.Evaluate.Text.cs — the per-call half of np.evaluate over text (docs/plans/evaluate-text.md)
// =============================================================================
//
// A program whose inputs include text (or that holds a node only a stage runs: a numpy.strings function, a cast to text)
// carries a plan instead of a kernel (NDExpr.Text.cs): the stages that compute on text, in evaluation order, and either
// the numeric remainder over their numeric results or the TEXT ROOT — the text value that is the whole answer. A call
// resolves each stage's operands — an input as it is, a literal as NumPy's Python scalar, an earlier stage's result, a
// computed numeric sub-tree evaluated here into the temporary NumPy's unfused sequence would hold — runs the stage's
// statement (the engine's own member for NumPy's statement, its errors included), then:
//   - for a numeric result, hands the numeric inputs and the stage results to the remainder's program, resolved through
//     the positional program cache (its dtype signature and 0-d parameters are those of THIS call's stage results), which
//     EvaluateCore runs with the call's keywords exactly as for any numeric tree;
//   - for a text result, applies the keywords itself (FinishResult): dtype= is NumPy's astype of the result, out= takes
//     it under np.evaluate's casting rule and the ufunc shape rules, where= masks a fresh result, order= lays it out.
// A root ufunc over text (np.add / np.multiply / np.maximum / np.minimum) takes out= / where= itself instead — NumPy's own
// call — which is how an S / U repeat exists at all (NumPy computes it only into an out=).
//
// The same finishing serves a numeric result asked for as TEXT (a text out= or dtype= over any tree): the fused kernels
// write numbers only, so the result is computed at its natural dtype and then cast, NumPy's formatting cut to the width.
// The stage results are this call's temporaries and are released here, except the one handed out as the answer.
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Evaluates a program over text: its stages, then its numeric remainder or its text root (see the file header).
        /// </summary>
        /// <param name="program">A program with a <see cref="NDExprProgram.TextPlan"/>.</param>
        /// <param name="inputs">The call's inputs, in the program's input order (text arrays included).</param>
        /// <param name="out">The caller's result array, or null for a fresh result.</param>
        /// <param name="options">The call's resolved keywords.</param>
        /// <returns>The evaluation's result: <paramref name="out"/> itself, the remainder's fresh result, a text result (cast
        ///     into the keywords), or — for a tree that is one stage with no result-shaping keyword — the stage's own result
        ///     array (NumPy's statement's).</returns>
        /// <exception cref="NotSupportedException">A reduction tree given <c>dtype=</c>, <c>order=</c> or <c>where=</c>
        ///     (checked before any text is read).</exception>
        /// <exception cref="TypeError">A stage NumPy refuses (no loop for the operand classes, no common dtype for np.where's
        ///     branches, a numpy.strings function over the wrong class), with NumPy's text.</exception>
        /// <exception cref="IncorrectShapeException">A stage's operands, or the remainder's, do not broadcast.</exception>
        /// <exception cref="ValueError">A StringDType NA a stage's rule refuses, a parse a cast stage refuses, an
        ///     <c>index</c> / <c>rindex</c> without a match, a flat StringDType reduction over several axes, or the remainder's
        ///     own checks (NumPy's texts).</exception>
        /// <exception cref="TypeError">The result cannot be cast into <paramref name="out"/> under the casting rule, or
        ///     a search's <c>start</c> / <c>end</c> cannot reach int64.</exception>
        /// <exception cref="ArgumentException">The result does not broadcast into <paramref name="out"/>; a non-bool
        ///     <c>where=</c>.</exception>
        /// <exception cref="OverflowException">A StringDType repeat by a negative count or past 2^63 - 1 characters, or a parse
        ///     out of the target's range.</exception>
        /// <remarks>
        ///     <para>
        ///     Ordering. NumPy evaluates the same tree as statements: each text node over its own operands, then the numeric
        ///     ufuncs over the results. So the stages run first, in evaluation order (a stage's broadcast or refusal is raised
        ///     before any later statement runs), and the result's <c>out=</c> / <c>where=</c> checks come after them, as
        ///     NumPy's final statement makes them. Only the keyword rules that depend on the tree alone run first.
        ///     </para>
        ///     <para>
        ///     Ownership. Every stage result is a fresh array of this call. The remainder writes a fresh result or the
        ///     caller's <paramref name="out"/>, never into a stage result, so all of them are released when the call ends —
        ///     except the stage result handed out as the answer (a bare stage result, a text root, or <paramref name="out"/>
        ///     itself when the root ufunc wrote it).
        ///     </para>
        /// </remarks>
        private NDArray EvaluateTextPlan(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options)
        {
            var plan = program.TextPlan;
            ThrowIfReductionKeywords(plan.IsReduction, options);

            // The stage that IS the whole evaluation, when there is one: a text root's stage, or the stage a bare numeric
            // remainder reads. A root ufunc over text takes out= / where= as NumPy's own call does (an S / U repeat exists only
            // into an out=), unless dtype= or a non-default casting= asks for np.evaluate's cast of the computed result.
            int rootStage = plan.TextRoot.Kind == NDExprStageOperandKind.StageResult ? plan.TextRoot.Index
                : plan.IsBareStageResult(out int bare) ? bare : -1;
            bool rootTakesKeywords = rootStage >= 0 && plan.Stages[rootStage].TakesOutAndWhere
                                     && (@out is not null || options.HasWhere) && !options.HasDtype
                                     && options.Casting == NPY_CASTING.NPY_SAME_KIND_CASTING;

            var results = new NDArray[plan.Stages.Length];
            NDArray handedOut = null;
            try
            {
                for (int s = 0; s < results.Length; s++)
                    results[s] = s == rootStage && rootTakesKeywords
                        ? RunTextStage(plan, plan.Stages[s], inputs, results, @out, options.Where, intoKeywords: true)
                        : RunTextStage(plan, plan.Stages[s], inputs, results, null, null, intoKeywords: false);

                if (rootTakesKeywords)
                {
                    // NumPy's ufunc wrote out= (returned as it is) or a fresh masked result, which order= may still lay out.
                    handedOut = results[rootStage];
                    if (@out is not null || !options.HasExplicitOrder)
                        return handedOut;
                    var laidOut = RelayoutResult(handedOut, options.Order);
                    if (!ReferenceEquals(laidOut, handedOut))
                        results[rootStage] = null;   // RelayoutResult released the stage's array
                    return handedOut = laidOut;
                }

                if (plan.HasTextRoot)
                {
                    // A text stage's array is this call's own; a text INPUT (a bare text leaf as the whole expression) is the
                    // caller's, so its answer is a copy (FinishResult copies a borrowed natural result).
                    bool owned = plan.TextRoot.Kind == NDExprStageOperandKind.StageResult;
                    var natural = owned ? results[plan.TextRoot.Index] : inputs[plan.TextRoot.Index];
                    var reduction = plan.RootReductionStage >= 0 ? plan.Stages[plan.RootReductionStage] : null;
                    if (owned)
                        results[plan.TextRoot.Index] = null;   // FinishResult owns it now (hands it out or releases it)
                    return handedOut = FinishResult(natural, owned, @out, options, inputs, ReductionOutRule(reduction));
                }

                if (plan.RootReductionStage >= 0)
                {
                    // A numeric reduction of text (np.argmax, np.mean of a NaN-like NA's sum): its result is the stage's own,
                    // and out= follows the reduction's rules (exact rank, np.evaluate's casting rule), not an elementwise copy.
                    var reduced = results[plan.RootReductionStage];
                    var stage = plan.Stages[plan.RootReductionStage];
                    results[plan.RootReductionStage] = null;
                    return handedOut = FinishResult(reduced, owned: true, @out, options, inputs, ReductionOutRule(stage));
                }

                // np.evaluate(NDExpr.Equal(a, b)), np.evaluate(NDExpr.StrLen(a)): the stage's own array — NumPy's statement
                // result, dtype, shape and layout — is the answer; copying it through the remainder would only add a pass.
                // Any keyword that shapes the result (out=, where=, dtype=, order=) takes the remainder, which applies it.
                if (@out is null && !options.HasWhere && !options.HasDtype && !options.HasExplicitOrder && rootStage >= 0)
                    return handedOut = results[rootStage];

                var remainderInputs = plan.RemainderInputs(inputs, results);
                var remainder = plan.Remainder.GetProgram(remainderInputs);
                return EvaluateCore(remainder, remainderInputs, @out, options);
            }
            finally
            {
                // The stage results are this call's temporaries (the remainder never writes into them); the one handed out
                // as the answer belongs to the caller now, and a caller's out= is never released.
                foreach (var r in results)
                    if (r is not null && !ReferenceEquals(r, handedOut) && !ReferenceEquals(r, @out))
                        r.Dispose();
            }
        }

        /// <summary>
        ///     A numeric tree whose result is asked for as TEXT — a text <c>out=</c> or a text <c>dtype=</c>: the fused kernels
        ///     write numbers only, so the result is computed at its natural dtype (with only the layout request) and then
        ///     finished into the keywords (<see cref="FinishResult"/>): NumPy's formatting of each number, cut to the width.
        /// </summary>
        /// <param name="program">A numeric program (no text plan).</param>
        /// <param name="inputs">The call's inputs.</param>
        /// <param name="out">The caller's result array, or null.</param>
        /// <param name="options">The call's resolved keywords (a text <c>out=</c> or <c>dtype=</c> among them).</param>
        /// <returns><paramref name="out"/>, or a fresh text result.</returns>
        /// <exception cref="TypeError">The natural result cannot be cast into <paramref name="out"/> under the casting
        ///     rule.</exception>
        /// <exception cref="ArgumentException">The natural result does not broadcast into <paramref name="out"/>; a
        ///     non-bool <c>where=</c>.</exception>
        /// <remarks>Compute-then-cast gives NumPy's values for every selected element; a masked-off element of
        ///     <paramref name="out"/> keeps its text (np.evaluate's masked write never reads <paramref name="out"/>).</remarks>
        private NDArray EvaluateIntoText(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options)
        {
#pragma warning disable NDW012 // not a leak: FinishResult takes the natural result (owned: true) and returns it or releases it on every path
            var natural = EvaluateCore(program, inputs, null, options.ForNaturalResult());
#pragma warning restore NDW012
            var rule = program.Reduce is { } reduce ? new ReductionOut(ReduceUfuncName(reduce.Kind), reduce.Axis is null, reduce.Keepdims)
                : program.Average is { } average ? new ReductionOut("add", average.Axis is null, average.Keepdims)
                : default;
#pragma warning disable NDW012 // not a leak: IteratorOperands hands back the call's own operands (a subset of inputs), named by an error only
            return FinishResult(natural, owned: true, @out, options, program.IteratorOperands(inputs), rule);
#pragma warning restore NDW012
        }

        /// <summary>
        ///     How a root REDUCTION's result takes <c>out=</c>: NumPy names its reduce ufunc in a rank refusal, and a flat
        ///     reduction's out has the exact rank (0, or the input's with keepdims, one element); an axis reduction's extents
        ///     are checked by the copy, as the numeric axis route checks them. <c>default</c> (a null <see cref="Ufunc"/>)
        ///     for an elementwise result, which follows the ufunc shape rules instead.
        /// </summary>
        /// <param name="Ufunc">The reduce ufunc NumPy names (<c>add</c>, <c>maximum</c>, <c>argmax</c>, ...), or null.</param>
        /// <param name="Flat">Whether the reduction is flat (no axis).</param>
        /// <param name="Keepdims">Whether it keeps the reduced axes as length 1.</param>
        private readonly record struct ReductionOut(string Ufunc, bool Flat, bool Keepdims);

        /// <summary>
        ///     The <c>out=</c> rule of a text plan's root reduction stage (a reduction or a weighted average of text), or
        ///     <c>default</c> for any other stage or none.
        /// </summary>
        /// <param name="stage">The root reduction's stage, or null.</param>
        /// <returns>The rule.</returns>
        private static ReductionOut ReductionOutRule(NDExprTextStage stage) => stage switch
        {
            NDExprReduceStage r => new ReductionOut(r.ReduceUfunc, r.Axis is null, r.Keepdims),
            NDExprAverageStage a => new ReductionOut("add", a.Axis is null, a.Keepdims),
            _ => default,
        };

        /// <summary>
        ///     Applies the call's result keywords to a computed NATURAL result (a text result, or a numeric one asked for as
        ///     text): <c>dtype=</c> is NumPy's <c>astype</c> of it; <c>out=</c> takes it under np.evaluate's casting rule
        ///     (<c>same_kind</c> unless <c>casting=</c> says otherwise) and the ufunc shape rules (a reduction's: its exact
        ///     rank), copied under <c>where=</c>; <c>where=</c> alone makes a fresh result written only where the mask holds;
        ///     <c>order=</c> lays a fresh result out. With no keyword the natural result is the answer (a borrowed one is
        ///     copied: np.evaluate answers with an array of its own).
        /// </summary>
        /// <param name="natural">The computed result.</param>
        /// <param name="owned">Whether <paramref name="natural"/> belongs to this call (released when not the answer) or to the
        ///     caller (a text input as the whole expression: copied, never released).</param>
        /// <param name="out">The caller's result array, or null.</param>
        /// <param name="options">The call's resolved keywords.</param>
        /// <param name="operands">The call's operands, named by a broadcast error the way the numeric route names them.</param>
        /// <param name="reduction">For a root reduction, how its result takes <c>out=</c>; <c>default</c> for an elementwise
        ///     result.</param>
        /// <returns><paramref name="out"/>, or a fresh array the caller owns.</returns>
        /// <exception cref="TypeError">The cast into <paramref name="out"/> is not allowed under the casting rule
        ///     (<c>Cannot cast ufunc 'evaluate' output from ... with casting rule ...</c>).</exception>
        /// <exception cref="ArgumentException">The result does not broadcast into <paramref name="out"/> /
        ///     <c>where=</c> or would stretch <paramref name="out"/>; a reduction's <paramref name="out"/> of the wrong
        ///     rank; a non-bool <c>where=</c>.</exception>
        /// <exception cref="ValueError">A text result that does not parse into a numeric <paramref name="out"/> /
        ///     <c>dtype=</c> (NumPy's parse, possible only under <c>casting='unsafe'</c> or an explicit dtype).</exception>
        private NDArray FinishResult(NDArray natural, bool owned, NDArray @out, in NDEvaluateOptions options, NDArray[] operands,
                                     ReductionOut reduction)
        {
            NDArray current = natural;
            bool currentOwned = owned;
            try
            {
                if (@out is null && !options.HasWhere && !options.HasDtype && !options.HasExplicitOrder)
                {
                    if (currentOwned)
                    {
                        currentOwned = false;   // handed out
                        return current;
                    }
                    return current.copy('K');
                }

                // dtype= is an implicit root cast (an astype of the computed result, NumPy's unsafe astype rule): a text
                // descriptor formats numbers or recasts text; it keeps the result's layout (K order).
                if (options.HasDtype)
                {
                    var cast = current.astype(options.DtypeDescr, true, 'K', "unsafe");
                    if (currentOwned)
                        current.Dispose();
                    current = cast;
                    currentOwned = true;
                }

                if (@out is not null)
                {
                    // np.evaluate's out= rule: the result's dtype must reach out's under the casting rule (by descriptor, so a
                    // text width or class decides as in NumPy: U5 → U3 is same_kind, U → S is not), checked before any write.
                    if (!DTypeCasting.CanCastTypeTo(current.dtype, @out.dtype, options.Casting))
                        throw new TypeError(
                            $"Cannot cast ufunc 'evaluate' output from {current.dtype.ToString(true)} to " +
                            $"{@out.dtype.ToString(true)} with casting rule '{DTypeCasting.CastingToString(options.Casting)}'");
                    if (reduction.Ufunc is not null)
                    {
                        // A flat reduction's out has the result's exact rank (0, or the input's under keepdims, one element:
                        // the rule the numeric reductions apply); an axis reduction's extents are checked by the copy, as the
                        // numeric axis route checks them.
                        if (reduction.Flat)
                            ValidateFlatReduceOut(@out, current.ndim, reduction.Keepdims, reduction.Ufunc);
                    }
                    else
                    {
                        ValidateWhereMask(options.Where);
                        ResolveUfuncIterationShape(current.Shape.Clean(), operands, @out, options.Where);
                    }
                    // The cast rule was checked above: the copy itself casts without a rule of its own.
                    UnmanagedStorage.Copyto(@out, current, NPY_CASTING.NPY_UNSAFE_CASTING, options.Where);
                    return @out;
                }

                if (options.HasWhere)
                {
                    // A fresh result written only where the mask holds (NumPy leaves the rest uninitialized; here it is zero —
                    // empty text, or 0): the mask broadcasts with the result, as a ufunc's where= does.
                    ValidateWhereMask(options.Where);
                    var dims = ResolveUfuncIterationShape(current.Shape.Clean(), operands, null, options.Where).dimensions;
                    char order = ResolveFreshOrder(current, options.Order);
                    var fresh = new NDArray(current.dtype, new Shape((long[])dims.Clone(), order, copy: false), true);
                    try
                    {
                        UnmanagedStorage.Copyto(fresh, current, NPY_CASTING.NPY_UNSAFE_CASTING, options.Where);
                    }
                    catch
                    {
                        fresh.Dispose();
                        throw;
                    }
                    return fresh;
                }

                // order= alone (or after dtype=): a fresh result laid out as asked; a borrowed result is copied first.
                var laid = RelayoutResult(currentOwned ? current : current.copy('K'), options.Order);
                currentOwned = false;   // RelayoutResult took it (returned it, or released it for its copy)
                return laid;
            }
            finally
            {
                if (currentOwned && !ReferenceEquals(current, @out))
                    current.Dispose();
            }
        }

        /// <summary>
        ///     The memory order of a fresh result np.evaluate makes from a computed <paramref name="result"/>: <c>'C'</c> /
        ///     <c>'F'</c> as asked, otherwise the result's own (F when it is F-contiguous and not C-contiguous, as the numeric
        ///     route keeps an all-F evaluation F).
        /// </summary>
        /// <param name="result">The computed result.</param>
        /// <param name="order">The resolved order request ('C', 'F', 'A' or 'K').</param>
        /// <returns><c>'C'</c> or <c>'F'</c>.</returns>
        private static char ResolveFreshOrder(NDArray result, char order)
        {
            char o = char.ToUpperInvariant(order);
            if (o == 'C' || o == 'F')
                return o;
            return result.Shape.IsFContiguous && !result.Shape.IsContiguous ? 'F' : 'C';
        }

        /// <summary>
        ///     Lays a fresh result out as <c>order=</c> asks: <c>'C'</c> / <c>'F'</c> copy it unless it already is so,
        ///     <c>'A'</c> keeps a C- or F-contiguous result and copies anything else in C order, <c>'K'</c> keeps it.
        /// </summary>
        /// <param name="fresh">A fresh result this call owns.</param>
        /// <param name="order">The resolved order request.</param>
        /// <returns><paramref name="fresh"/> itself, or its copy (then <paramref name="fresh"/> is released).</returns>
        private static NDArray RelayoutResult(NDArray fresh, char order)
        {
            char o = char.ToUpperInvariant(order);
            bool keep = o switch
            {
                'C' => fresh.Shape.IsContiguous,
                'F' => fresh.Shape.IsFContiguous,
                'A' => fresh.Shape.IsContiguous || fresh.Shape.IsFContiguous,
                _ => true,
            };
            if (keep)
                return fresh;
            using (fresh)
                return fresh.copy(o == 'F' ? 'F' : 'C');
        }

        /// <summary>
        ///     Runs one stage for one call: resolves its operands (<see cref="ResolveStageOperand"/>), runs its statement —
        ///     with the evaluation's <c>out=</c> / <c>where=</c> when it is the root ufunc that takes them — and releases the
        ///     operand arrays this call made.
        /// </summary>
        /// <param name="plan">The program's text plan.</param>
        /// <param name="stage">The stage.</param>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="results">The stage results of this call so far (the stage reads only earlier ones).</param>
        /// <param name="out">The evaluation's <c>out=</c> for the root ufunc, or null.</param>
        /// <param name="where">The evaluation's <c>where=</c> for the root ufunc, or null.</param>
        /// <param name="intoKeywords">True to run the stage as the root ufunc with <paramref name="out"/> /
        ///     <paramref name="where"/> (<see cref="NDExprTextStage.ExecuteInto"/>).</param>
        /// <returns>The stage's fresh result (owned by the caller), or <paramref name="out"/>.</returns>
        /// <exception cref="System.Exception">Whatever the statement or a computed operand raises (NumPy's errors; see
        ///     <see cref="EvaluateTextPlan"/>): the operands made here are released first.</exception>
        private NDArray RunTextStage(NDExprTextPlan plan, NDExprTextStage stage, NDArray[] inputs, NDArray[] results,
                                     NDArray @out, NDArray where, bool intoKeywords)
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
                return result = intoKeywords ? stage.ExecuteInto(this, args, @out, where) : stage.Execute(this, args);
            }
            finally
            {
                // A view before the buffer it views (made[2i] before made[2i+1]); the statement's result is never one of
                // them (every stage returns a fresh array or out=), but it is checked so a future statement that hands an
                // operand back cannot have it released under the caller.
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
