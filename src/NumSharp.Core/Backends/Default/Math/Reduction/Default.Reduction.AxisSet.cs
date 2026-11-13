using System;

// =============================================================================
// Default.Reduction.AxisSet.cs — the reductions over a TUPLE axis (API-parity batch 5)
// =============================================================================
//
// NumPy's reductions take an axis TUPLE: np.sum(a, (0, 2)), np.max(a, ()), np.nanvar(a, (1, 2)). PyUFunc_GenericReduction
// turns the tuple into reduction flags (each item normalized in order - an AxisError for one out of bounds, the 0-d
// allowance of an integer 0 / -1 not applying to an item - then PyUFunc_Reduce's "duplicate value in 'axis'"), and the
// reduction iterator runs over every flagged axis at once. A tuple is NOT a chain of single-axis reductions: the iterator's
// call schedule (its axis sort and coalescing, the reduce form per inner-loop call) decides the float bits.
//
// The engine members route a tuple three ways (ReductionArguments.TryIntegerForm):
//   * a set of ONE axis reduces exactly like that integer, and a set of EVERY axis exactly like None (NumPy builds the
//     same flags), so those take the member's integer / None route with its exact schedules and its text routes;
//   * every other tuple - two axes or more short of every axis, the empty tuple of an input with dimensions, or a tuple
//     NumPy refuses - takes the member's NumPy statements over the axis set (add / multiply / maximum / minimum / fmax /
//     fmin.reduce through NumPyReduceIterator; _mean / _var and the nan functions composed of those reductions), which
//     raise in NumPy's order because they are NumPy's statements.
//
// Text under such a set. Fixed-width text and every StringDType loop NumPy refuses answer whatever the axes (the loop
// resolution runs before the reduction looks at them), so the None route's refusal is theirs. StringDType's add / maximum /
// minimum are NOT reorderable: two axes or more is PyUFunc_ReduceWrapper's "at most one axis may be specified" (also the
// None route's answer, since such a set implies three axes or more), and the empty tuple reduces nothing - each output is
// its own element (a copy, or initial= combined with it), which is the reduction along an appended axis of length 1
// (VStringReduceOverNoAxis).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Whether <paramref name="axis"/> is a tuple that reduces NO axis of an input with dimensions (the empty tuple):
        ///     every output then reduces exactly its own element.
        /// </summary>
        /// <param name="axis">The axis argument.</param>
        /// <param name="flags">The reduced-axis flags it gave (<see cref="ReductionArguments.AxisFlags(int, NDAxis?)"/>).</param>
        /// <returns>True for a tuple with no flagged axis over a rank of 1 or more; false for an integer, None, or a 0-d
        ///     input's empty tuple (which is None: every axis of a 0-d input).</returns>
        internal static bool IsNoAxisSet(NDAxis? axis, bool[] flags)
            => axis is { IsTuple: true } && flags.Length > 0 && Array.IndexOf(flags, true) < 0;

        /// <summary>
        ///     The integer form a TEXT route takes for a validated axis argument: an integer or None as given, a one-axis set
        ///     as that axis, and any other set as None.
        /// </summary>
        /// <param name="axis">The axis argument, already validated (its flags built without an error).</param>
        /// <param name="flags">The reduced-axis flags it gave.</param>
        /// <returns>The single reduced axis, or null.</returns>
        /// <remarks>
        ///     None stands for a set of two axes or more because every text answer for one is the None route's: the loop
        ///     refusals do not read the axes, and StringDType's "not reorderable" counts the reduced axes, which are two or more
        ///     both ways (a set short of every axis implies an input of three axes or more). The empty tuple of an input with
        ///     dimensions is the one set None does not stand for (<see cref="IsNoAxisSet"/>): a caller that computes text
        ///     answers it first, and a refusal (the only other text answer) does not read the axes.
        /// </remarks>
        internal static int? TextAxis(NDAxis? axis, bool[] flags)
        {
            if (axis is not { IsTuple: true })
                return ReductionArguments.IntegerAxis(axis);
            int first = Array.IndexOf(flags, true);
            // Exactly one flagged axis: its integer route.
            if (first >= 0 && Array.IndexOf(flags, true, first + 1) < 0)
                return first;
            return null;
        }

        /// <summary>
        ///     A StringDType reduction (add, maximum or minimum) over the empty tuple of an input with dimensions: NumPy reduces
        ///     no axis, so every output is its own element - a copy, or <paramref name="initial"/> combined with it, a
        ///     <paramref name="where"/> mask selecting which outputs combine - which is exactly the reduction along an appended
        ///     axis of length 1. The loop resolution and the out / mask shapes are checked against the input's own shape first,
        ///     so a refusal or a shape error is NumPy's and names NumPy's shapes.
        /// </summary>
        /// <param name="arr">The StringDType input (rank 1 or more).</param>
        /// <param name="flags">Its all-false reduced-axis flags.</param>
        /// <param name="keepdims">The caller's keepdims (no reduced axis to keep: the result has the input's shape either way;
        ///     only the out's rank check reads it).</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="initial">The caller's initial value, or null (only the reduction reads it).</param>
        /// <param name="where">The caller's mask, or null; a 0-d true mask is Python's <c>where=True</c>, no mask.</param>
        /// <param name="ufunc">The ufunc NumPy names (<c>add</c>, <c>maximum</c>, <c>minimum</c>).</param>
        /// <param name="reduceLane">The single-axis reduction of the lane view along its last axis (keepdims false, the
        ///     caller's out and initial), given the lane and the lane mask (null for no mask).</param>
        /// <returns>The reduction's result (the out when given), the input's shape.</returns>
        /// <exception cref="TypeError">NumPy's loop refusal (an out of another class, incompatible NA objects).</exception>
        /// <exception cref="ValueError">A misshapen out, a mask with more axes than the input, a mask without
        ///     <paramref name="initial"/> (the loop has no identity), an initial value the loop refuses.</exception>
        /// <exception cref="IncorrectShapeException">The out or the mask does not broadcast against the input.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the lane view and the lane mask are this call's own and released; the result
        ///     is yielded (the reductions make a fresh result or write into the caller's out, never return the lane).</remarks>
        [NDScoped]
        internal static NDArray VStringReduceOverNoAxis(NDArray arr, bool[] flags, bool keepdims, NDArray @out, object initial,
            NDArray @where, string ufunc, Func<NDArray, NDArray, NDArray> reduceLane)
        {
            // NumPy's order: the loop resolution on (out, input), then the iterator's shape checks over the input's shape.
            string op = ufunc switch { "add" => "sum", "maximum" => "max", _ => "min" };
            if (Strings.NDStringUfuncs.VStringReduceLoopAnswer(op, arr, null, @out) is { } refusal)
                throw refusal;
            ReductionArguments.ValidateOperands(arr, flags, keepdims, @out, @where, ufunc);

            // a[..., np.newaxis]: every element alone on a reduced axis of length 1, the mask stretched to the input's shape
            // and given the same axis (Python's where=True stays no mask, as the reduction itself reads it). The array
            // object's expand_dims (a view), not the np facade: an engine file calls no np.* (the layering ratchet R2).
            var lane = NDArray.expand_dims(arr, -1);
            NDArray mask = null;
            if (@where is not null && !IsTrueScalarMask(@where))
                mask = NDArray.expand_dims(NDArray.broadcast_to(@where, arr.Shape), -1);
            return reduceLane(lane, mask);
        }

        /// <summary>
        ///     <c>maximum.reduce</c> / <c>minimum.reduce</c> over a tuple axis that is neither one axis nor every axis (or a
        ///     tuple NumPy refuses): the keyword path's statements (<see cref="MaxMinNumPy"/>) over the whole set at once.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="tuple">The tuple axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The engine member's loop dtype, or null for the input's (np.max / np.min have no
        ///     <c>dtype=</c>; only a direct engine call passes one).</param>
        /// <param name="out">NumPy's <c>out=</c>, or null.</param>
        /// <param name="initial">NumPy's <c>initial=</c>, or null.</param>
        /// <param name="where">NumPy's <c>where=</c>, or null.</param>
        /// <param name="op"><see cref="MinMaxOp.Max"/> or <see cref="MinMaxOp.Min"/>.</param>
        /// <returns><paramref name="out"/> itself when given, else the fresh reduction (NumPy's layout).</returns>
        /// <exception cref="AxisError">An item of the tuple is out of bounds.</exception>
        /// <exception cref="ValueError">A repeated axis; an empty reduced axis or a mask without <paramref name="initial"/>; a
        ///     <c>StringDType</c> reduction over two axes or more (not reorderable); text that does not parse into a numeric
        ///     <paramref name="dtype"/>; a read-only or misshapen out.</exception>
        /// <exception cref="TypeError">Fixed-width text or a string <paramref name="dtype"/> (NumPy's loop refusal); a C# scalar
        ///     as out; an initial value the loop cannot take.</exception>
        /// <exception cref="ArgumentException">A mask that is not bool.</exception>
        /// <exception cref="IncorrectShapeException">The out or the mask does not broadcast.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the cast copy a <paramref name="dtype"/> makes is released; the result is
        ///     yielded.</remarks>
        [NDScoped]
        private NDArray MaxMinOverAxisSet(NDArray arr, NDAxis tuple, bool keepdims, DType dtype, NDArray @out, object initial,
            NDArray @where, MinMaxOp op)
        {
            if (dtype is null)
                return MaxMinNumPy(arr, tuple, @out, keepdims, initial, @where, op);

            // dtype=: ufunc.reduce's converters and the axis first, then the loop - maximum.reduce casts every element into it
            // (a parse for a numeric dtype over text) before it compares, and a selection over the cast elements visited in
            // the same order is the cast copy's (astype's 'K' keeps the memory order the iterator walks).
            ReductionArguments.ValidateParsedArguments(@out, @where);
            ReductionArguments.AxisFlags(arr.ndim, tuple);
            string name = op == MinMaxOp.Max ? "max" : "min";
            if (Strings.NDStringUfuncs.ParseForNumericReduce(name, arr, null, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => MaxMinNumPy(p, tuple, @out, keepdims, initial, @where, op));
            Strings.NDStringUfuncs.ThrowIfReduce(name, arr, (int?)null, dtype);
            var cast = arr.astype(dtype);
            return MaxMinNumPy(cast, tuple, @out, keepdims, initial, @where, op);
        }
    }
}
