using System;
using NumSharp.Backends.Iteration;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc/>
        public override NDArray All(NDArray a, int? axis, NDArray @out, bool keepdims = false, NDArray where = null)
            => LogicalReduceNumPy(a, axis, @out, keepdims, where, reduceAll: true);

        /// <inheritdoc/>
        public override NDArray All(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray where = null)
            => LogicalReduceNumPy(a, axis, @out, keepdims, where, reduceAll: true);

        /// <inheritdoc/>
        public override NDArray Any(NDArray a, int? axis, NDArray @out, bool keepdims = false, NDArray where = null)
            => LogicalReduceNumPy(a, axis, @out, keepdims, where, reduceAll: false);

        /// <inheritdoc/>
        public override NDArray Any(NDArray a, int[] axis, NDArray @out, bool keepdims = false, NDArray where = null)
            => LogicalReduceNumPy(a, axis, @out, keepdims, where, reduceAll: false);

        /// <summary>
        ///     NumPy's <c>np.all</c> / <c>np.any</c> with an <c>int?</c> axis, the whole call:
        ///     <c>logical_and.reduce</c> / <c>logical_or.reduce(a, axis, bool, out, keepdims=, where=)</c> in NumPy's order.
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axis, or null for every axis.</param>
        /// <param name="out">The caller's <c>out=</c>, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">The caller's <c>where=</c> bool mask, or null.</param>
        /// <param name="reduceAll">True for <c>all</c>, false for <c>any</c>.</param>
        /// <returns><paramref name="out"/> after the store, or the fresh result (read-only when 0-d: NumPy's scalar).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">The reduce prologue's out= / where= refusals (<see cref="ReductionArguments"/>).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or an out of another shape.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="IncorrectShapeException">The mask or the out does not broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>Moved from the public <c>np.all</c> / <c>np.any(a, int? axis, out, keepdims, where)</c> overloads by the
        ///     np-layer dtype plan's P1b, statement for statement; the two bodies were the same but for the ufunc's name and
        ///     the mask's identity.</remarks>
        [NDScoped] // reclaims the where-mask effective array and the reduced temp on the out= path
        private NDArray LogicalReduceNumPy(NDArray a, int? axis, NDArray @out, bool keepdims, NDArray @where, bool reduceAll)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy's reduce prologue, in NumPy's order (read-only out, mask dtype, axis, out rank, iterator shapes), before
            // any work. Without out= and where= it can raise nothing the reduction below does not raise itself, so the plain
            // call skips it.
            if (@out is not null || @where is not null)
                ReductionArguments.Validate(a, axis, keepdims, @out, @where, reduceAll ? "logical_and" : "logical_or");
            // Text: after NumPy's reduce prologue its logical_and / logical_or.reduce reads the strings' truth values
            // through the bool cast: the stand-in is the truth array (an empty input's too), and the bool path below gives
            // NumPy's identity result, keepdims shape, out= write and where= handling on it.
            if (Strings.NDStringUfuncs.ReduceStandIn(reduceAll ? "all" : "any", a, axis) is { } standIn)
            {
                using (standIn)
                    return LogicalReduceNumPy(standIn, axis, @out, keepdims, @where, reduceAll);
            }

            NDArray effective = @where is null ? a : reduceAll ? ApplyWhereForAll(a, @where) : ApplyWhereForAny(a, @where);
            NDArray<bool> reduced = reduceAll ? All(effective, axis, keepdims) : Any(effective, axis, keepdims);
            // A fresh 0-d result is a numpy SCALAR (np.bool_) at the boundary: read-only (PyArray_Return semantics); out=
            // returns the writeable out array itself.
            return @out is null ? reduced.MarkReductionScalar() : WriteLogicalToOut(reduced, @out);
        }

        /// <summary>
        ///     NumPy's <c>np.all</c> / <c>np.any</c> with a tuple axis, the whole call (see the <c>int?</c> form).
        /// </summary>
        /// <param name="a">The array (any dtype, any layout).</param>
        /// <param name="axis">The reduced axes.</param>
        /// <param name="out">The caller's <c>out=</c>, or null.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="where">The caller's <c>where=</c> bool mask, or null.</param>
        /// <param name="reduceAll">True for <c>all</c>, false for <c>any</c>.</param>
        /// <returns><paramref name="out"/> after the store, or the fresh result (read-only when 0-d).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="axis"/> is null.</exception>
        /// <exception cref="ValueError">The reduce prologue's refusals, or an axis that repeats
        ///     (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>, or an out of another shape.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="IncorrectShapeException">The mask or the out does not broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="SystemError">A <c>U</c> array holding a unit above U+10FFFF.</exception>
        /// <remarks>Moved from the public tuple-axis <c>np.all</c> / <c>np.any(a, int[] axis, out, keepdims, where)</c>
        ///     overloads by the np-layer dtype plan's P1b, statement for statement; the 0-d marking the facade's tuple overload
        ///     added to the reduced result is applied here, where the call returns it.</remarks>
        [NDScoped] // reclaims the where-mask effective array and the reduced temp on the out= path
        private NDArray LogicalReduceNumPy(NDArray a, int[] axis, NDArray @out, bool keepdims, NDArray @where, bool reduceAll)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (axis is null)
                throw new ArgumentNullException(nameof(axis));

            // NumPy's reduce prologue for a tuple axis (see the int? form).
            if (@out is not null || @where is not null)
                ReductionArguments.Validate(a, axis, keepdims, @out, @where, reduceAll ? "logical_and" : "logical_or");
            // Text: the truth array (see the int? form); a tuple is checked item by item and for repeats first.
            if (Strings.NDStringUfuncs.ReduceStandIn(reduceAll ? "all" : "any", a, axis) is { } standIn)
            {
                using (standIn)
                    return LogicalReduceNumPy(standIn, axis, @out, keepdims, @where, reduceAll);
            }

            NDArray effective = @where is null ? a : reduceAll ? ApplyWhereForAll(a, @where) : ApplyWhereForAny(a, @where);
            NDArray<bool> reduced = reduceAll ? All(effective, axis, keepdims) : Any(effective, axis, keepdims);
            // Every axis reduced away: a numpy SCALAR at the boundary (read-only); out= returns the out array itself.
            return @out is null ? reduced.MarkReductionScalar() : WriteLogicalToOut(reduced, @out);
        }

        /// <summary>
        ///     The effective bool array of <c>all</c> under a <c>where=</c> mask: the identity of <c>all</c> is True, so a
        ///     masked-off element contributes True (<c>effective = !mask | bool(a)</c>, broadcast by the operators).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="where">The mask (already validated as bool by the reduce prologue).</param>
        /// <returns>A fresh bool array of the broadcast shape.</returns>
        [NDScopedCovered] // only caller: LogicalReduceNumPy, [NDScoped] (the !mask and truth temporaries are scope-reclaimed)
        private static NDArray ApplyWhereForAll(NDArray a, NDArray @where)
        {
            NDArray<bool> maskBool = TruthOf(@where);
            NDArray<bool> aBool = TruthOf(a);
            // (!maskBool) | aBool — broadcasting is handled by the operators.
            return !maskBool | aBool;
        }

        /// <summary>
        ///     The effective bool array of <c>any</c> under a <c>where=</c> mask: the identity of <c>any</c> is False, so a
        ///     masked-off element contributes False (<c>effective = mask &amp; bool(a)</c>).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="where">The mask (already validated as bool by the reduce prologue).</param>
        /// <returns>A fresh bool array of the broadcast shape.</returns>
        [NDScopedCovered] // only caller: LogicalReduceNumPy, [NDScoped] (the truth temporaries are scope-reclaimed)
        private static NDArray ApplyWhereForAny(NDArray a, NDArray @where)
        {
            NDArray<bool> maskBool = TruthOf(@where);
            NDArray<bool> aBool = TruthOf(a);
            return maskBool & aBool;
        }

        /// <summary>
        ///     NumPy's truth value of every element (non-zero is True; NaN and infinities are True): a bool array as a typed
        ///     wrapper over its own storage, any other array through <c>!= 0</c>.
        /// </summary>
        /// <param name="nd">The array (numbers or bool; text has been replaced by its truth array before this runs).</param>
        /// <returns>A bool array of <paramref name="nd"/>'s shape.</returns>
        private static NDArray<bool> TruthOf(NDArray nd)
        {
            if (nd.GetTypeCode == NPTypeCode.Boolean)
                return nd.MakeGeneric<bool>();
            return nd != 0;
        }

        /// <summary>
        ///     The <c>out=</c> store of <c>np.all</c> / <c>np.any</c>: <paramref name="out"/> keeps its dtype (a float out
        ///     receives 0.0 / 1.0) and must have the reduction's own shape; the bool result is stored with unsafe casting.
        /// </summary>
        /// <param name="reduced">The bool result.</param>
        /// <param name="out">The caller's out (the reduce prologue has already refused a read-only or misshapen one).</param>
        /// <returns><paramref name="out"/> itself.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="out"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/>'s shape is not the result's.</exception>
        /// <remarks>Moved from <c>np.all</c>'s <c>WriteToOut</c> by the np-layer dtype plan's P1b. The store still goes
        ///     through <c>np.copyto</c>, a structural call into the facade that the plan's P6 replaces with an engine copy;
        ///     the up-call ratchet pins it with that reason.</remarks>
        private static NDArray WriteLogicalToOut(NDArray<bool> reduced, NDArray @out)
        {
            if (@out is null)
                throw new ArgumentNullException(nameof(@out));
            if (!@out.Shape.Equals(reduced.Shape))
                throw new ArgumentException(
                    $"output shape mismatch: expected {reduced.Shape}, got {@out.Shape}",
                    nameof(@out));

            UnmanagedStorage.Copyto(@out, reduced, NPY_CASTING.NPY_UNSAFE_CASTING);
            return @out;
        }
    }
}
