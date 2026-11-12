using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.roll(a, shift, axis=None)</c>: roll array elements along the given axis or axes. Elements that
        ///     roll beyond the last position are re-introduced at the first.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="shift">The number of places by which elements are shifted: a C# integer (it converts to a 0-d array),
        ///     or an array of one dimension (<c>new[] { 1, 2 }</c>) broadcast against the axes - one shift for every axis, or one
        ///     per axis. Each element is read as Python's <c>int()</c> reads it (a float truncated, a bool 0 / 1), and repeated
        ///     axes add up their shifts.</param>
        /// <param name="axis">The axis, or the axes (<c>(0, 1)</c>), along which elements are shifted. <c>null</c> (the
        ///     default) rolls the flattened array, after which the original shape is restored.</param>
        /// <returns>Output array, with the same shape as <paramref name="a"/>: <paramref name="a"/>'s memory order along axes,
        ///     C order for the flattened roll. A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy
        ///     creates it as the subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="shift"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>'s rank.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="shift"/> does not broadcast against the axes (NumPy's
        ///     ValueError <c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with
        ///     shape (3,) and arg 1 with shape (2,).</c>).</exception>
        /// <exception cref="ValueError"><paramref name="shift"/> and the axes broadcast to two dimensions or more (<c>'shift'
        ///     and 'axis' should be scalars or 1D sequences</c>); a NaN shift; a text shift that is no int literal; a 0-d
        ///     <paramref name="a"/> with the empty tuple (NumPy's <c>not enough values to unpack (expected 2, got 0)</c>); a
        ///     <c>numpy.matrix</c> operand whose result cannot be a matrix (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="OverflowError">An infinite shift.</exception>
        /// <remarks>
        ///     NumPy's algorithm (<see cref="NDArray"/>'s internal roll): empty_like plus one slice copy per combination of
        ///     the rolled axes' body and tail. One signature, NumPy's: the <c>int</c> / <c>long</c> shift overloads went with
        ///     API-parity batch 5, as both convert to the array shift. https://numpy.org/doc/stable/reference/generated/numpy.roll.html
        /// </remarks>
        public static NDArray roll(NDArray a, NDArray shift, NDAxis? axis = null)
        {
            // NumPy's statements (the flat roll, or empty_like and the slice copies along the axes) are the array object's.
            return NDArray.roll(a, shift, axis);
        }

        /// <summary>
        ///     <see cref="roll(NDArray, NDArray, NDAxis?)"/> with the axis as one <c>int</c>: the overload a C# integer
        ///     binds (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so every
        ///     answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="roll(NDArray, NDArray, NDAxis?)"/>
        public static NDArray roll(NDArray a, NDArray shift, int axis)
            => roll(a, shift, (NDAxis?)axis);

        /// <summary>
        ///     <see cref="roll(NDArray, NDArray, NDAxis?)"/> with the axes as an <c>int[]</c>: the overload an
        ///     <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis),
        ///     forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple (<c>new
        ///     int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="roll(NDArray, NDArray, NDAxis?)"/>
        public static NDArray roll(NDArray a, NDArray shift, int[] axis)
            => roll(a, shift, (NDAxis?)axis);
    }
}
