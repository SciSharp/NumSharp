using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.expand_dims(a, axis)</c>: a view of <paramref name="a"/> with a new axis of length one at every
        ///     position <paramref name="axis"/> names in the RESULT.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="axis">NumPy's <c>axis</c> (required, no None): one position (<c>0</c>, <c>-1</c>) or a tuple of
        ///     positions (<c>(0, 3)</c> or <c>new[] { 0, 3 }</c>), each in the result's rank <c>a.ndim + Count</c>;
        ///     negative positions count from the end of the result. An empty tuple inserts nothing. A
        ///     <c>List&lt;int&gt;</c> needs <c>.ToArray()</c>.</param>
        /// <returns>A NEW view of <paramref name="a"/> with the new axes (an empty tuple included, NumPy's
        ///     <c>a.reshape(shape)</c>): it shares memory with <paramref name="a"/>. A <c>numpy.matrix</c> operand gives a
        ///     plain array (NumPy converts it with <c>asarray</c> first); a <c>numpy.char.chararray</c> operand gives a
        ///     chararray over the result (NumPy creates it as the subclass), owning its data exactly when the plain result
        ///     would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null, or <paramref name="axis"/> was converted
        ///     from a null <c>int[]</c> (a non-nullable <see cref="NDAxis"/> has no None).</exception>
        /// <exception cref="AxisError">A position is out of range for the result's rank: <c>axis 4 is out of bounds for
        ///     array of dimension 4</c>, the position as given (every position is range-checked before the repeat
        ///     check).</exception>
        /// <exception cref="ValueError">Two positions are the same after normalization (<c>repeated axis</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html. NumPy's statements are the array
        ///     object's (<c>NDArray.expand_dims</c>): <c>out_ndim = len(axis) + a.ndim</c>,
        ///     <c>axis = normalize_axis_tuple(axis, out_ndim)</c>, then the reshape.
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int</c>, <c>int[]</c> and
        ///     <c>IEnumerable&lt;int&gt;</c> overloads (the last one let a null sequence return <paramref name="a"/> itself,
        ///     where NumPy has no None for this argument).</para>
        /// </remarks>
        public static NDArray expand_dims(NDArray a, NDAxis axis)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy's statements are the array object's: one position takes the single-axis statement (the output rank is
            // a.ndim + 1), a tuple the tuple statement (a.ndim + len(axis), every range before the repeat check).
            return axis.IsTuple
                ? NDArray.expand_dims(a, axis.ToArray())
                : NDArray.expand_dims(a, axis[0]);
        }

        /// <summary>
        ///     <see cref="expand_dims(NDArray, NDAxis)"/> with the axis as one <c>int</c>: the overload a C# integer
        ///     binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so every
        ///     answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="expand_dims(NDArray, NDAxis)"/>
        public static NDArray expand_dims(NDArray a, int axis)
            => expand_dims(a, (NDAxis)axis);

        /// <summary>
        ///     <see cref="expand_dims(NDArray, NDAxis)"/> with the axes as an <c>int[]</c>: the overload an
        ///     <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis),
        ///     forwarding to the NDAxis form as NumPy's tuple (<c>new int[0]</c> is <c>()</c>). NumPy's expand_dims has
        ///     no None, so a null array throws, exactly as the NDAxis form's conversion does.
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple (not null).</param>
        /// <exception cref="ArgumentNullException"><paramref name="axis"/> is null (expand_dims has no None).</exception>
        /// <inheritdoc cref="expand_dims(NDArray, NDAxis)"/>
        public static NDArray expand_dims(NDArray a, int[] axis)
            => expand_dims(a, (NDAxis)axis);
    }
}
