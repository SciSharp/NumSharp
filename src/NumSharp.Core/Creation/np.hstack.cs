using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence horizontally (column wise).
        ///     This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis.Rebuilds arrays divided by hsplit.
        ///     This function makes most sense for arrays with up to 3 dimensions.For instance, for pixel-data with a height(first axis), width(second axis), 
        ///     and r/g/b channels(third axis). The functions concatenate, stack and block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the second axis, except 1-D arrays which can be any length.</param>
        /// <returns>The array formed by stacking the given arrays.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.hstack.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate").</exception>
        public static NDArray hstack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (atleast_1d(*tup), then concatenate along axis 0 for 1-d arrays, axis 1 otherwise) are the
            // array object's.
            return NDArray.hstack_(tup);
        }

        /// <summary>
        ///     NumPy's <c>hstack(tup, *, dtype=None, casting='same_kind')</c>: stack the arrays in sequence horizontally,
        ///     with np.concatenate's <c>dtype</c> / <c>casting</c> - along the first axis for 1-D arrays, the second
        ///     otherwise.
        /// </summary>
        /// <param name="tup">The arrays (NumPy's sequence): the same shape along all but the second axis, except 1-D arrays,
        ///     which can be any length.</param>
        /// <param name="dtype">The result dtype handed to np.concatenate, or null for the arrays' promoted dtype.</param>
        /// <param name="casting">The casting rule handed to np.concatenate (<c>'no'</c>, <c>'equiv'</c>, <c>'safe'</c>,
        ///     <c>'same_kind'</c> or <c>'unsafe'</c>).</param>
        /// <returns>The array formed by stacking the given arrays (a <c>numpy.matrix</c> operand gives a matrix, NumPy's
        ///     subtype rule; a chararray a plain array).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate"); a
        ///     <c>numpy.matrix</c> result of more than two long axes.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to <paramref name="dtype"/> under
        ///     <paramref name="casting"/> (np.concatenate's refusal with NumPy's text; NumPy's type is TypeError, pinned in
        ///     <c>OpenBugs.ApiParity.cs</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
        ///     <para>NumPy's keywords beside NumSharp's <c>params</c> form, which the owner kept (API-parity batch 3): C#
        ///     allows nothing after a <c>params</c> array, so <c>np.hstack(a, b)</c> stays and <c>np.hstack(new[] { a, b },
        ///     dtype: np.float32)</c> (C# 12: <c>np.hstack([a, b], dtype: np.float32)</c>) binds here. A plain array argument
        ///     with no keyword binds here too (C# prefers the overload with more declared parameters); both answer
        ///     alike.</para></remarks>
        public static NDArray hstack(NDArray[] tup, DType dtype = null, string casting = "same_kind")
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (atleast_1d(*tup), then concatenate with dtype / casting) are the array object's.
            return NDArray.hstack_(tup, dtype, casting);
        }
    }
}
