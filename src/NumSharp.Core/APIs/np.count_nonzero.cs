using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Counts the number of non-zero values in the array.
        /// </summary>
        /// <param name="a">The array for which to count non-zeros.</param>
        /// <returns>Number of non-zero values in the array (a text element counts when non-empty); 0 for an empty
        ///     array.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html — the count is the engine's
        ///     (<c>TensorEngine.CountNonZero</c>). The C# <see cref="long"/> of NumPy's <c>np.count_nonzero(a)</c> (its
        ///     <c>np.intp</c> scalar), kept beside the axis signature below as C# programs read it; with any argument after
        ///     <paramref name="a"/> C# binds the axis signature, whose answer is an array.</remarks>
        public static long count_nonzero(NDArray a)
        {
            // NumPy's count_nonzero over every element (multiarray.count_nonzero), text and empty arrays included, is the
            // engine's: DefaultEngine.CountNonZero.
            return a.TensorEngine.CountNonZero(a);
        }

        /// <summary>
        ///     NumPy's <c>np.count_nonzero(a, axis=None, *, keepdims=False)</c>: the number of non-zero values along the given
        ///     axes (every axis by default).
        /// </summary>
        /// <param name="a">The array for which to count non-zeros (any dtype; a text element counts when non-empty).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) counts every element; one axis
        ///     (<c>0</c>, <c>-1</c>) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) counts along those axes, and the empty
        ///     tuple counts each element on its own (0 or 1). Negative axes count from the end. A 0-d <paramref name="a"/>
        ///     takes the integer axis 0 or -1 (the sum's rule) and the empty tuple.</param>
        /// <param name="keepdims">If True, the counted axes are left in the result as dimensions with size one (keyword-only
        ///     in NumPy; C# takes it positionally too).</param>
        /// <returns>The int64 counts (NumPy's <c>intp</c>), laid out as NumPy's sum allocates them; a read-only 0-d array
        ///     (NumPy's scalar) when every axis is counted without <paramref name="keepdims"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of bounds (the first one in order, as given; every axis is
        ///     range-checked before the repeat check).</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html. NumPy's statements:
        ///     <c>if axis is None and not keepdims: return multiarray.count_nonzero(a)</c>, else
        ///     <c>a.astype(bool, copy=False).sum(axis=axis, dtype=np.intp, keepdims=keepdims)</c> (text through
        ///     <c>a != ''</c>) - the whole call is the engine's (<see cref="TensorEngine.CountNonZero(NDArray, NDAxis?, bool)"/>).
        ///     The counts are integers, so they are exact in any reduction order.
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces <c>count_nonzero(a, int axis,
        ///     keepdims)</c>, which had no tuple and no <c>axis=None</c> with <c>keepdims</c>.</para>
        /// </remarks>
        public static NDArray count_nonzero(NDArray a, NDAxis? axis = null, bool keepdims = false)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            // NumPy's count_nonzero(a, axis, keepdims=) - None, an int or a tuple, with the empty array, the text truth values
            // and the 0-d rules - is the engine's whole call.
            return a.TensorEngine.CountNonZero(a, axis, keepdims);
        }
    }
}
