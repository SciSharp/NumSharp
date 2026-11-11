using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays of shape(N,) have been reshaped to(1, N). Rebuilds arrays divided by vsplit.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the first axis. 1-D arrays must have the same length.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 2-D.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vstack.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate").</exception>
        public static NDArray vstack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (atleast_2d(*tup), then concatenate along axis 0) are the array object's.
            return NDArray.vstack_(tup);
        }

        /// <summary>
        ///     NumPy's <c>vstack(tup, *, dtype=None, casting='same_kind')</c>: stack the arrays in sequence vertically (1-D
        ///     arrays become rows), with np.concatenate's <c>dtype</c> / <c>casting</c>.
        /// </summary>
        /// <param name="tup">The arrays (NumPy's sequence): the same shape along all but the first axis; 1-D arrays the same
        ///     length.</param>
        /// <param name="dtype">The result dtype handed to np.concatenate, or null for the arrays' promoted dtype.</param>
        /// <param name="casting">The casting rule handed to np.concatenate (<c>'no'</c>, <c>'equiv'</c>, <c>'safe'</c>,
        ///     <c>'same_kind'</c> or <c>'unsafe'</c>).</param>
        /// <returns>The array formed by stacking the given arrays, at least 2-D (a <c>numpy.matrix</c> operand gives a
        ///     matrix; a chararray a plain array).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate"); a
        ///     <c>numpy.matrix</c> result of more than two long axes.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to <paramref name="dtype"/> under
        ///     <paramref name="casting"/> (np.concatenate's refusal with NumPy's text; NumPy's type is TypeError, pinned in
        ///     <c>OpenBugs.ApiParity.cs</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.vstack.html
        ///     <para>NumPy's keywords beside NumSharp's <c>params</c> form, which the owner kept (API-parity batch 3): C#
        ///     allows nothing after a <c>params</c> array, so <c>np.vstack(a, b)</c> stays and <c>np.vstack(new[] { a, b },
        ///     dtype: np.int8, casting: "unsafe")</c> binds here. A plain array argument with no keyword binds here too;
        ///     both answer alike.</para></remarks>
        public static NDArray vstack(NDArray[] tup, DType dtype = null, string casting = "same_kind")
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (atleast_2d(*tup), then concatenate along axis 0 with dtype / casting) are the array object's.
            return NDArray.vstack_(tup, dtype, casting);
        }
    }
}
