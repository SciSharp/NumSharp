using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Lower triangle of an array — a copy of <paramref name="m"/> with all elements
        ///     <em>above</em> the <paramref name="k"/>-th diagonal zeroed.
        /// </summary>
        /// <param name="m">Input array. For arrays with <c>ndim &gt; 2</c>, <c>tril</c> applies to the final two axes.</param>
        /// <param name="k">Diagonal above which to zero elements. <c>k = 0</c> (the default) is the main diagonal, <c>k &lt; 0</c> is below it and <c>k &gt; 0</c> is above.</param>
        /// <returns>Lower triangle of <paramref name="m"/>, of the same dtype. Always a fresh, writeable, C-contiguous array
        ///     (2-D for a 1-D <paramref name="m"/>, whose row is broadcast down a square matrix).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril.html</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="m"/> is 0-d (NumPy's leaked text: <c>tri() missing 1
        ///     required positional argument: 'N'</c>).</exception>
        public static NDArray tril(NDArray m, int k = 0)
        {
            using var plain = NDMatrixBoundary.PlainFor(m);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's statements (the 0-d refusal, the row-run blit or `where(tri(...), m, zeros(1, m.dtype))`) are the
            // array object's.
            return NDArray.tril(m, k);
        }

        /// <summary>
        ///     Upper triangle of an array — a copy of <paramref name="m"/> with all elements
        ///     <em>below</em> the <paramref name="k"/>-th diagonal zeroed.
        /// </summary>
        /// <param name="m">Input array. For arrays with <c>ndim &gt; 2</c>, <c>triu</c> applies to the final two axes.</param>
        /// <param name="k">Diagonal below which to zero elements. <c>k = 0</c> (the default) is the main diagonal, <c>k &lt; 0</c> is below it and <c>k &gt; 0</c> is above.</param>
        /// <returns>Upper triangle of <paramref name="m"/>, of the same dtype. Always a fresh, writeable, C-contiguous array
        ///     (2-D for a 1-D <paramref name="m"/>, whose row is broadcast down a square matrix).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu.html</remarks>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="m"/> is 0-d (NumPy's leaked text: <c>tri() missing 1
        ///     required positional argument: 'N'</c>).</exception>
        public static NDArray triu(NDArray m, int k = 0)
        {
            using var plain = NDMatrixBoundary.PlainFor(m);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's statements (the 0-d refusal, the row-run blit or `where(tri(..., k-1), zeros(1, m.dtype), m)`) are
            // the array object's.
            return NDArray.triu(m, k);
        }
    }
}
