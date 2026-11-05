using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // np.evaluate's staged float fmax / fmin nodes (NDExpr.FMaxStages.cs, parity QA-20)
        // =============================================================================
        //
        // Each stage is NumPy's own np.fmax / np.fmin call: its children are evaluated as NumPy's unfused sequence would
        // hold them (a leaf as it is, a literal as a Python scalar, a computed subtree in the fresh layout NumPy's ufunc
        // allocates), then ExecuteBinaryOp runs them with NumPy's position-dependent answer. The remainder over the stage
        // results is an ordinary program. When the expression IS one fmax / fmin, its stage takes the call's out= and
        // where= itself, because NumPy's masked or out= call has positions of its own.
        // =============================================================================

        /// <summary>
        /// Evaluates a program whose float32 / float64 fmax / fmin nodes were staged (see the section header): the stages in
        /// order, then the remainder over the original inputs and the stage results, with the call's keywords.
        /// </summary>
        /// <param name="program">The staged program (<see cref="NDExprProgram.FMaxPlan"/> non-null).</param>
        /// <param name="inputs">The call's inputs, in the program's input order.</param>
        /// <param name="out">The caller's out= array, or null.</param>
        /// <param name="options">The call's resolved keywords.</param>
        /// <returns>The result: <paramref name="out"/> when given, else a fresh array.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (a stage's or the remainder's), or out=
        /// does not have the result's shape.</exception>
        /// <exception cref="ArgumentException">out= cannot hold the result's dtype under the casting rule, or where= is
        /// not a bool array.</exception>
        private NDArray EvaluateFMaxPlan(NDExprProgram program, NDArray[] inputs, NDArray @out, in NDEvaluateOptions options)
        {
            var plan = program.FMaxPlan;
            int n = plan.InputCount, stages = plan.Stages.Length;
            var all = new NDArray[n + stages];
            Array.Copy(inputs, all, n);

            // A root fmax / fmin takes out= and where= itself, NumPy's call with those keywords, when nothing else shapes
            // the result: no dtype= (a result cast), no explicit order=, the default same_kind casting the ufunc applies.
            bool keywordsToStage = plan.RootIsLastStage && (@out is not null || options.HasWhere)
                                   && !options.HasDtype && !options.HasExplicitOrder
                                   && options.Casting == NPY_CASTING.NPY_SAME_KIND_CASTING;

            NDArray returned = null;
            try
            {
                for (int k = 0; k < stages; k++)
                {
                    var stage = plan.Stages[k];
                    NDArray left = null, right = null, ownL1 = null, ownL2 = null, ownR1 = null, ownR2 = null;
                    try
                    {
                        left = StageOperand(stage.Left, all, out ownL1, out ownL2);
                        right = StageOperand(stage.Right, all, out ownR1, out ownR2);
                        // dtype= is the node's NEP 50 dtype: the fused typing reads a 0-d array as strong, which the
                        // engine (that takes any 0-d operand for a Python scalar) would not; it never changes NumPy's loop.
                        bool last = k == stages - 1;
                        if (keywordsToStage && last)
                            all[n + k] = ExecuteBinaryOp(left, right, stage.Op, @out, options.Where, stage.Dtype);
                        else
                        {
                            var r = ExecuteBinaryOp(left, right, stage.Op, null, null, stage.Dtype);
                            // A result read by later stages or the remainder must be the temporary NumPy holds: its
                            // layout decides a later np.fmax's calls and a reduction's order. The root's own result is
                            // shaped by the call's keywords instead.
                            all[n + k] = last && plan.RootIsLastStage ? r : AsNumPyUfuncResultLayout(r, left, right);
                        }
                    }
                    finally
                    {
                        // A view before the buffer it views.
                        ownL1?.Dispose();
                        ownL2?.Dispose();
                        ownR1?.Dispose();
                        ownR2?.Dispose();
                    }
                }

                if (keywordsToStage)
                    return returned = all[n + stages - 1];

                // A plain root stage is already the answer when it has the layout the fused evaluation would allocate;
                // otherwise (and for every non-root stage) the remainder runs with the call's keywords.
                if (plan.RootIsLastStage && @out is null && !options.HasWhere && !options.HasDtype
                    && !options.HasExplicitOrder && HasEvaluateLayout(all[n + stages - 1], program.IteratorOperands(inputs)))
                    return returned = all[n + stages - 1];

                var remainderOps = new NDArray[plan.RemainderInputs.Length];
                for (int i = 0; i < remainderOps.Length; i++)
                    remainderOps[i] = all[plan.RemainderInputs[i]];
                var remainder = plan.Remainder.GetProgram(remainderOps);
                return returned = EvaluateCore(remainder, remainderOps, @out, options);
            }
            finally
            {
                // The stage results are this call's temporaries, except the one returned (an out= is the caller's).
                for (int k = 0; k < stages; k++)
                {
                    var r = all[n + k];
                    if (r is not null && !ReferenceEquals(r, returned) && !ReferenceEquals(r, @out))
                        r.Dispose();
                }
            }
        }

        /// <summary>
        /// One stage operand for this call: a positional input as it is, a literal as the 0-d operand NumPy's ufunc sees
        /// (a Python scalar), or a computed subtree evaluated over exactly the operands it reads into the fresh array
        /// NumPy's own ufunc would allocate for it (<see cref="MaterializeChildNumPyLayout"/>: the K-order layout of its
        /// operands), so the stage's np.fmax sees NumPy's temporary — its layout decides NumPy's calls.
        /// </summary>
        /// <param name="child">The operand's description.</param>
        /// <param name="all">The staged evaluation's inputs so far (original inputs, then earlier stage results).</param>
        /// <param name="owned1">Receives an array this call created and must dispose (a literal, or the operand view), or null.</param>
        /// <param name="owned2">Receives the buffer behind a view in <paramref name="owned1"/>, or null.</param>
        /// <returns>The operand.</returns>
        /// <exception cref="IncorrectShapeException">A computed subtree's operands do not broadcast.</exception>
        private NDArray StageOperand(NDExprFMaxStageChild child, NDArray[] all, out NDArray owned1, out NDArray owned2)
        {
            owned1 = owned2 = null;
            if (child.Input >= 0)
                return all[child.Input];
            if (child.Literal is not null)
                return owned1 = child.Literal.ToOperandArray();

            var ops = new NDArray[child.Inputs.Length];
            for (int i = 0; i < ops.Length; i++)
                ops[i] = all[child.Inputs[i]];
            var childProgram = child.Tree.GetProgram(ops);
            var view = MaterializeChildNumPyLayout(childProgram, ops, childProgram.IteratorOperands(ops), out var buffer);
            owned1 = view;
            if (!ReferenceEquals(view, buffer))
                owned2 = buffer;
            return view;
        }

        /// <summary>
        /// A stage's result in the layout NumPy's ufunc allocates for that call (<see cref="Shape.NpyIterOutputShape"/>:
        /// its operands' memory order, C on ties), copied there when the engine laid it out otherwise (a permuted operand),
        /// so the next stage or the remainder reads the very temporary NumPy's unfused sequence holds.
        /// </summary>
        /// <param name="result">The stage's result (fresh, owned by the caller); disposed when it is replaced.</param>
        /// <param name="left">The stage's first operand.</param>
        /// <param name="right">The stage's second operand.</param>
        /// <returns><paramref name="result"/> itself, or a copy in NumPy's layout.</returns>
        private NDArray AsNumPyUfuncResultLayout(NDArray result, NDArray left, NDArray right)
        {
            if (result.ndim < 2)
                return result;   // 0-d and 1-D layouts are fixed by the shape
            var want = Shape.NpyIterOutputShape((long[])result.Shape.dimensions.Clone(), new[] { left.Shape, right.Shape });
            var have = result.Shape.strides;
            bool same = true;
            for (int k = 0; k < have.Length && same; k++)
                same = result.Shape.dimensions[k] == 1 || have[k] == want.strides[k];
            if (same)
                return result;

            var relaid = NewArray(result.typecode, want, false);
            try
            {
                // The array object's assignment (np.copyto's body below its argument parsing): same dtype, so no cast.
                UnmanagedStorage.Copyto(relaid, result, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
            }
            catch
            {
                relaid.Dispose();
                throw;
            }
            result.Dispose();
            return relaid;
        }

        /// <summary>
        /// Whether a root stage's result already has the layout the fused evaluation allocates for a fresh result with the
        /// default order: F when every non-scalar input is strictly F-contiguous (the 'K' heuristic,
        /// <see cref="AreAllInputsStrictFContig"/>), C otherwise. When it does, the stage's array is returned as the
        /// result; when not, the remainder copies it into that layout.
        /// </summary>
        /// <param name="result">The root stage's result.</param>
        /// <param name="iteratorOperands">The program's streamed inputs (its 0-d parameters excluded).</param>
        /// <returns>True when the result is laid out as the fused evaluation would lay it out.</returns>
        private static bool HasEvaluateLayout(NDArray result, NDArray[] iteratorOperands)
        {
            bool wantF = AreAllInputsStrictFContig(iteratorOperands, result.Shape);
            return wantF ? result.Shape.IsFContiguous : result.Shape.IsContiguous;
        }
    }
}
