using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Stack arrays in sequence depth wise (along third axis).
        ///     This is equivalent to concatenation along the third axis after 2-D arrays of shape(M, N) have been reshaped to(M, N,1) and 1-D arrays of shape(N,) have been reshaped to(1, N,1).
        ///     Rebuilds arrays divided by dsplit.
        ///     This function makes most sense for arrays with up to 3 dimensions.For instance, for pixel-data with a height(first axis), width(second axis), and r/g/b channels(third axis). The functions concatenate, stack and block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">The arrays must have the same shape along all but the third axis. 1-D or 2-D arrays must have the same shape.</param>
        /// <returns>The array formed by stacking the given arrays, will be at least 3-D.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.dstack.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="tup"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="tup"/> is empty ("need at least one array to concatenate").</exception>
        /// <exception cref="NullReferenceException">An entry of <paramref name="tup"/> is null.</exception>
        /// <exception cref="IncorrectShapeException">Arrays whose ranks, or dimensions other than the third, differ once
        ///     made three-dimensional.</exception>
        /// <exception cref="InvalidCastException">An array that does not cast to the result dtype.</exception>
        public static NDArray dstack(params NDArray[] tup)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(tup))
                return NDMatrixBoundary.Wrapped(tup, static s => dstack(s));
            // a numpy.char.chararray operand: NumPy's dstack ends with np.concatenate, which drops the class; the rule's
            // region keeps its inner atleast_3d calls plain (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(tup))
                return NDCharArrayBoundary.Plain(tup, static s => dstack(s));
            if (tup == null)
                throw new ArgumentNullException(nameof(tup));

            // NumPy's statements (atleast_3d(*tup), then concatenate along axis 2) are the array object's; its name carries
            // the trailing underscore because NDArray.dstack is the public instance method.
            return NDArray.dstack_(tup);
        }
    }
}
