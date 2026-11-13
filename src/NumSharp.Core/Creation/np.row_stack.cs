using System;

namespace NumSharp
{
    public static partial class np
    {
        // =============================================================================================================
        // np.row_stack - NumPy 2.4's deprecated alias of np.vstack (numpy/_core/shape_base.py: `warnings.warn(
        // "`row_stack` alias is deprecated. Use `np.vstack` directly.", DeprecationWarning)` then `return vstack(tup,
        // dtype=dtype, casting=casting)`). NumSharp has no warnings channel, so the alias answers exactly as np.vstack
        // and its remarks carry the deprecation. The three forms are vstack's: the params form the owner kept for
        // vstack / hstack, NumPy's keyword form, and the object-array-of-lists form.
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>row_stack(tup)</c>, the deprecated alias of <see cref="vstack(NDArray[])"/>: stacks the arrays in
        ///     sequence vertically (1-D arrays become rows) - concatenation along the first axis after <c>atleast_2d</c>.
        /// </summary>
        /// <param name="tup">The arrays: the same shape along all but the first axis; 1-D arrays the same length.</param>
        /// <returns>The stacked array, at least 2-D (np.vstack's: a <c>numpy.matrix</c> operand gives a matrix, a chararray
        ///     a plain array).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty (<c>need at least one array to concatenate</c>);
        ///     arrays whose other axes differ; a <c>numpy.matrix</c> result of more than two long axes.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.row_stack.html - DEPRECATED in NumPy 2.0 (NumPy warns
        ///     "`row_stack` alias is deprecated. Use `np.vstack` directly."; NumSharp has no warnings channel): prefer
        ///     <see cref="vstack(NDArray[])"/>. API-parity batch 12.
        /// </remarks>
        public static NDArray row_stack(params NDArray[] tup)
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // vstack's statements (atleast_2d(*tup), then concatenate along axis 0), the array object's.
            return NDArray.vstack_(tup);
        }

        /// <summary>
        ///     NumPy's <c>row_stack(tup, *, dtype=None, casting='same_kind')</c>, the deprecated alias of
        ///     <see cref="vstack(NDArray[], DType, string)"/>: the arrays stacked vertically with np.concatenate's
        ///     <c>dtype</c> / <c>casting</c>.
        /// </summary>
        /// <param name="tup">The arrays (NumPy's sequence).</param>
        /// <param name="dtype">The result dtype handed to np.concatenate, or null for the arrays' promoted dtype.</param>
        /// <param name="casting">The casting rule handed to np.concatenate (<c>'no'</c>, <c>'equiv'</c>, <c>'safe'</c>,
        ///     <c>'same_kind'</c> or <c>'unsafe'</c>).</param>
        /// <returns>The stacked array, at least 2-D.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty; a <c>numpy.matrix</c> result of more than two long
        ///     axes.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to <paramref name="dtype"/> under
        ///     <paramref name="casting"/> (np.vstack's refusal with NumPy's text; NumPy's type is TypeError).</exception>
        /// <remarks>DEPRECATED in NumPy 2.0: prefer <see cref="vstack(NDArray[], DType, string)"/>. API-parity batch 12.</remarks>
        public static NDArray row_stack(NDArray[] tup, DType dtype = null, string casting = "same_kind")
        {
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // vstack's statements with dtype / casting, the array object's.
            return NDArray.vstack_(tup, dtype, casting);
        }

        /// <summary>
        ///     NumPy's <c>row_stack(tup)</c> over object arrays of Python lists (<see cref="NDListArray"/>): the deprecated
        ///     alias of <see cref="vstack(NDListArray[])"/>, the lists shared as NumPy's object pointers are.
        /// </summary>
        /// <param name="tup">The list arrays (NumPy's sequence).</param>
        /// <returns>A new object array of the lists, at least 2-D.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> or one of its arrays is null.</exception>
        /// <exception cref="ValueError">No array; arrays whose other axes differ.</exception>
        /// <exception cref="NotSupportedException">A bytes list beside str lists.</exception>
        /// <exception cref="ObjectDisposedException">One of the arrays was disposed.</exception>
        /// <remarks>DEPRECATED in NumPy 2.0: prefer <see cref="vstack(NDListArray[])"/>. A sequence parameter, as vstack's
        ///     (a second params form would make <c>np.row_stack()</c> ambiguous). API-parity batch 12 (the owner: "support
        ///     that too").</remarks>
        public static NDListArray row_stack(NDListArray[] tup) => NDListArray.VStack(tup);
    }
}
