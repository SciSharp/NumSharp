using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Transposes a matrix (or a stack of matrices) <paramref name="x"/>.<br></br>
        ///     Swaps the two innermost dimensions, i.e. an array of shape <c>(..., M, N)</c> becomes <c>(..., N, M)</c>.<br></br>
        ///     Equivalent to <c>np.swapaxes(x, -1, -2)</c>. This function is Array API compatible.
        /// </summary>
        /// <param name="x">Input array having shape <c>(..., M, N)</c> and whose two innermost dimensions form <c>MxN</c> matrices.</param>
        /// <returns>An array containing the transpose for each matrix and having shape <c>(..., N, M)</c>. A view is returned whenever possible.</returns>
        /// <exception cref="ArgumentException">If <paramref name="x"/> has fewer than 2 dimensions.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.matrix_transpose.html</remarks>
        public static NDArray matrix_transpose(NDArray x)
        {
            if (x.ndim < 2)
                throw new ArgumentException($"Input array must be at least 2-dimensional, but it is {x.ndim}");

            return x.TensorEngine.SwapAxes(x, -1, -2);
        }
    }
}
