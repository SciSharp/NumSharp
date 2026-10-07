using System;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The calls NumPy computes on text but answers from the SHAPE alone: a truth test, a count, an index search, a sort
    ///     or an arg-reduction that reads no element (an empty input), or that orders a single element (a sort axis of
    ///     length one, a unique over one element). NumPy's own loops return early for those shapes
    ///     (<c>_new_sortlike</c> / <c>_new_argsortlike</c> start with <c>if (N &lt;= 1 || PyArray_SIZE(op) == 0)</c>, a
    ///     reduction over no element is its identity), so the answer is final behavior for the string dtypes, not a gap of
    ///     the string support.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>How a call computes such an answer: a bool stand-in.</b> Over these shapes NumPy's answer does not depend on the
    ///     values: the shapes of the results, an index output (all zeros along a length-one axis), a reduction's identity, an
    ///     <c>out=</c> write and every error raised after the text check are the same for any dtype. So the caller runs its
    ///     NUMERIC path on <see cref="BoolStandIn"/>, a bool array of zeros with the input's shape, and gets NumPy's answer
    ///     without a string lane in any kernel. A call that returns VALUES (a sort, a partition, a unique) takes the values
    ///     from the input itself: over these shapes they are the input's elements in their input order.
    ///     </para>
    ///     <para>
    ///     <b>Order.</b> The checks NumPy makes before it reads an element run first, with the text operand, exactly as
    ///     for a call that computes (<see cref="ValidateReduceAxis"/>, the arg-reductions' empty-sequence error, the 0-d
    ///     <c>nonzero</c> error), so a bad axis raises NumPy's AxisError even when the shape alone would decide the answer.
    ///     </para>
    /// </remarks>
    internal static partial class NDStringUfuncs
    {
        /// <summary>
        ///     NumPy's refusal of <c>nonzero</c> on a 0-d array (<c>PyArray_Nonzero</c>), for any dtype but bool: NumPy raises
        ///     before it reads the element, so a 0-d string array gets it too. (NumPy's bool text adds the
        ///     <c>arr[nonzero(cond)]</c> hint; the numeric 0-d path of NumSharp keeps its NumPy 1.x promotion, see
        ///     <see cref="DefaultEngine.NonZero"/>.)
        /// </summary>
        internal const string NonZero0dMessage =
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.";

        /// <summary>
        ///     A bool array of zeros with <paramref name="a"/>'s dimensions (C order, owning): the stand-in a call runs its
        ///     numeric path on when NumPy's answer over the text in <paramref name="a"/> is decided by the shape.
        /// </summary>
        /// <param name="a">The text operand.</param>
        /// <returns>A fresh bool array the caller owns; dispose it once the numeric call returned (no result aliases it:
        ///     every function that takes a stand-in returns new arrays).</returns>
        internal static NDArray BoolStandIn(NDArray a)
        {
            var dims = (long[])a.Shape.dimensions.Clone();
            return np.zeros(dims.Length == 0 ? Shape.NewScalar() : Shape.Adopt(dims), NPTypeCode.Boolean);
        }

        /// <summary>
        ///     Whether NumPy's <paramref name="op"/> over the text in <paramref name="a"/> reads no element, or orders one
        ///     element with nothing, so its answer is decided by the shape.
        /// </summary>
        /// <param name="op">The NumPy function name.</param>
        /// <param name="a">The text operand (its axis already validated).</param>
        /// <param name="axis">The <c>axis</c> argument, or null for every axis / the flattened array.</param>
        /// <returns>True when the shape decides the answer.</returns>
        /// <remarks>
        ///     The truth tests and the index searches (<c>any</c>, <c>all</c>, <c>count_nonzero</c>, <c>nonzero</c>,
        ///     <c>flatnonzero</c>, <c>argwhere</c>) read every element, so only an EMPTY input qualifies. The sorts and the
        ///     arg-reductions compare elements along one line, so an empty input or a line of at most one element
        ///     qualifies; a 0-d input of the ravelling functions (<c>argmax</c>, <c>argsort</c>, …) is one element.
        /// </remarks>
        internal static bool ShapeDecides(string op, NDArray a, int? axis)
        {
            if (a.size == 0)
                return op is "any" or "all" or "count_nonzero" or "nonzero" or "flatnonzero" or "argwhere" or "argmax"
                    or "argmin" or "nanargmax" or "nanargmin" or "sort" or "argsort" or "partition" or "argpartition";
            switch (op)
            {
                case "argmax":
                case "argmin":
                case "nanargmax":
                case "nanargmin":
                case "sort":
                case "argsort":
                case "partition":
                case "argpartition":
                    // axis=None and a 0-d input (the ravelling functions) order the flattened elements.
                    long line = axis is not { } ax || a.ndim == 0
                        ? a.size
                        : a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
                    return line <= 1;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     The guard of a function NumPy computes on text but answers from the shape when it reads no element: runs NumPy's
        ///     checks in NumPy's order, then returns normally when the shape decides the answer and throws when NumPy would
        ///     compute on text.
        /// </summary>
        /// <param name="op">The NumPy function name (see <see cref="ShapeDecides"/>).</param>
        /// <param name="a">The operand; a numeric array returns at once.</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <exception cref="AxisError">The axis is out of bounds (the rule of <see cref="AxisRuleOf"/>).</exception>
        /// <exception cref="ValueError">An arg-reduction over an empty line (<c>attempt to get argmax of an empty
        ///     sequence</c>), or <c>nonzero</c> of a 0-d array (<see cref="NonZero0dMessage"/>).</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text (the next phase).</exception>
        /// <remarks>Use it where the numeric path that follows handles the shape-decided text itself (a sort whose copy moves
        ///     the strings and whose line kernel returns for a line of one element); otherwise take <see cref="ReduceStandIn(string, NDArray, int?)"/>.</remarks>
        internal static void ThrowIfComputes(string op, NDArray a, int? axis = null)
        {
            if (!IsString(a))
                return;
            ValidateReduceAxis(op, a.ndim, axis);
            if (EmptyArgReductionError(op, a, axis) is { } empty)
                throw empty;
            if (op is "nonzero" && a.ndim == 0)
                throw new ValueError(NonZero0dMessage);
            if (!ShapeDecides(op, a, axis))
                throw ReduceAnswer(op, a, null, false);
        }

        /// <summary>
        ///     <see cref="ThrowIfComputes"/>, then the stand-in: null for a numeric input (the caller's path continues with it),
        ///     a <see cref="BoolStandIn"/> of a text input whose answer the shape decides (the caller runs its numeric path on
        ///     it and disposes it).
        /// </summary>
        /// <param name="op">The NumPy function name (see <see cref="ShapeDecides"/>).</param>
        /// <param name="a">The operand.</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <returns>Null for a numeric input; the bool stand-in otherwise.</returns>
        /// <exception cref="AxisError">The axis is out of bounds.</exception>
        /// <exception cref="ValueError">An arg-reduction over an empty line, or <c>nonzero</c> of a 0-d array.</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy computes this call on text (the next phase).</exception>
        internal static NDArray ReduceStandIn(string op, NDArray a, int? axis = null)
        {
            if (!IsString(a))
                return null;
            ThrowIfComputes(op, a, axis);
            return BoolStandIn(a);
        }

        /// <summary>
        ///     <see cref="ReduceStandIn(string, NDArray, int?)"/> for a tuple <c>axis</c> (<c>any</c> / <c>all</c>): the tuple
        ///     is checked item by item and for repeats; only an EMPTY input reads no element (an empty tuple still casts every
        ///     element to bool).
        /// </summary>
        /// <param name="op">The NumPy function name (<c>"any"</c> or <c>"all"</c>).</param>
        /// <param name="a">The operand.</param>
        /// <param name="axes">The axes, or null.</param>
        /// <returns>Null for a numeric input; the bool stand-in otherwise.</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">An axis appears twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="StringComputationNotSupportedException">NumPy reads the strings' truth values (the next phase).</exception>
        internal static NDArray ReduceStandIn(string op, NDArray a, int[] axes)
        {
            if (!IsString(a))
                return null;
            ValidateReduceAxes(op, a.ndim, axes);
            if (a.size != 0)
                throw ReduceAnswer(op, a, null, false);
            return BoolStandIn(a);
        }

        /// <summary>
        ///     Whether NumPy's <c>equal</c> / <c>not_equal</c> has a loop for the pair, i.e. compares the text: two strings of
        ///     ONE class. Any other pair with a string operand has no loop, and the <c>==</c> / <c>!=</c> operators answer
        ///     all-False / all-True without reading an element (<see cref="FailedComparison"/>).
        /// </summary>
        /// <param name="d1">The first descriptor.</param>
        /// <param name="d2">The second descriptor.</param>
        /// <returns>True for <c>S</c> with <c>S</c> or <c>U</c> with <c>U</c> (any widths).</returns>
        internal static bool ComparesText(DType d1, DType d2)
            => d1.Meta is StringDTypeMeta && d2.Meta is StringDTypeMeta && d1.Meta == d2.Meta;

        /// <summary>
        ///     Whether a concatenation of <paramref name="a"/> and <paramref name="b"/> can fail decoding bytes into str: one
        ///     operand is <c>U</c> (so the result is <c>U</c>) and the other is an <c>S</c> array holding a byte at or above
        ///     0x80. NumPy's ASCII decode raises there, naming the FIRST failing element in the order it converts them — after
        ///     a <c>unique</c>, the sorted order of the text — so a set routine whose answer otherwise follows from the shapes
        ///     depends on the text there and defers. All-ASCII bytes decode without fail, and the shape-decided answer stands.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <returns>True when non-ASCII <c>S</c> text meets <c>U</c> text.</returns>
        internal static bool ConcatMayFailDecoding(NDArray a, NDArray b)
        {
            static bool IsKind(NDArray x, char kind) => IsString(x) && x.dtype.kind == kind;
            static bool MayFail(NDArray bytes) => bytes.size != 0 && !NDStringArrays.AllAscii(bytes);
            return IsKind(a, 'U') && IsKind(b, 'S') && MayFail(b) || IsKind(b, 'U') && IsKind(a, 'S') && MayFail(a);
        }
    }
}
