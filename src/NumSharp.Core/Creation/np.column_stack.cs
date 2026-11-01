using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack 1-D arrays as columns into a 2-D array.
        ///     Take a sequence of 1-D arrays and stack them as columns to make a
        ///     single 2-D array. 2-D arrays are stacked as-is, just like with
        ///     <see cref="hstack"/>. 1-D arrays are turned into 2-D columns first.
        /// </summary>
        /// <param name="tup">
        ///     Arrays to stack. All of them must have the same first dimension.
        /// </param>
        /// <returns>The 2-D array formed by stacking the given arrays.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null, or one of its entries is (the parameter
        ///     is then named <c>tup[i]</c>).</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate").</exception>
        /// <exception cref="IncorrectShapeException">Arrays whose ranks or first dimensions differ once turned into
        ///     columns.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype.</exception>
        public static NDArray column_stack(params NDArray[] tup)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(tup))
                return NDMatrixBoundary.Wrapped(tup, static s => column_stack(s));
            // a numpy.char.chararray operand: NumPy's column_stack ends with np.concatenate, which drops the class; the rule's
            // region keeps its inner view and transpose calls plain (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(tup))
                return NDCharArrayBoundary.Plain(tup, static s => column_stack(s));
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (each array below two dimensions as a column view, the join along axis 1) are the array
            // object's.
            return NDArray.column_stack(tup);
        }
    }
}
