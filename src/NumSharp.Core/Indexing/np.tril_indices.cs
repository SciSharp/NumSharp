using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices for the lower-triangle of an <c>(n, m)</c> array.
        /// </summary>
        /// <param name="n">The row dimension of the arrays for which the returned indices will be valid; a negative count
        ///     is an empty axis.</param>
        /// <param name="k">Diagonal offset. <c>k = 0</c> (default) is the main diagonal.</param>
        /// <param name="m">The column dimension. By default <paramref name="m"/> is taken equal to <paramref name="n"/>.</param>
        /// <returns>The row and column indices of the lower triangle, in C (row-major) order: two owning int64
        ///     arrays (OWNDATA True, as NumPy's boolean-mask gathers are).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril_indices.html
        ///     <para>NumPy's statements are the array object's (<c>NDArray.tril_indices</c>), which emits the triangle's row
        ///     runs in O(output) and releases its column ladder.</para></remarks>
        public static NDArray<long>[] tril_indices(int n, int k = 0, int? m = null)
            => NDArray.tril_indices(n, k, m);

        /// <summary>
        ///     Return the indices for the upper-triangle of an <c>(n, m)</c> array.
        /// </summary>
        /// <param name="n">The size of the arrays for which the returned indices will be valid; a negative count is an
        ///     empty axis.</param>
        /// <param name="k">Diagonal offset. <c>k = 0</c> (default) is the main diagonal.</param>
        /// <param name="m">The column dimension. By default <paramref name="m"/> is taken equal to <paramref name="n"/>.</param>
        /// <returns>The row and column indices of the upper triangle, in C (row-major) order: two owning int64
        ///     arrays (OWNDATA True, as NumPy's boolean-mask gathers are).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html
        ///     <para>NumPy's statements are the array object's (<c>NDArray.triu_indices</c>).</para></remarks>
        public static NDArray<long>[] triu_indices(int n, int k = 0, int? m = null)
            => NDArray.triu_indices(n, k, m);

        /// <summary>
        ///     Return the indices for the lower-triangle of <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">The 2-D array whose shape supplies the dimensions.</param>
        /// <param name="k">Diagonal offset.</param>
        /// <returns>The row and column indices of <paramref name="arr"/>'s lower triangle (see
        ///     <see cref="tril_indices"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException"><c>input array must be 2-d</c> (NumPy <c>ValueError</c>, verbatim).</exception>
        /// <exception cref="OverflowException">A dimension of <paramref name="arr"/> exceeds <see cref="int.MaxValue"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tril_indices_from.html
        ///     <para>Only the shape is read, so no class rule applies (a numpy.matrix has two axes).</para></remarks>
        public static NDArray<long>[] tril_indices_from(NDArray arr, int k = 0)
            => NDArray.tril_indices_from(arr, k);

        /// <summary>
        ///     Return the indices for the upper-triangle of <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">The 2-D array whose shape supplies the dimensions.</param>
        /// <param name="k">Diagonal offset.</param>
        /// <returns>The row and column indices of <paramref name="arr"/>'s upper triangle (see
        ///     <see cref="triu_indices"/>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> is null.</exception>
        /// <exception cref="ArgumentException"><c>input array must be 2-d</c> (NumPy <c>ValueError</c>, verbatim).</exception>
        /// <exception cref="OverflowException">A dimension of <paramref name="arr"/> exceeds <see cref="int.MaxValue"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.triu_indices_from.html
        ///     <para>Only the shape is read, so no class rule applies (a numpy.matrix has two axes).</para></remarks>
        public static NDArray<long>[] triu_indices_from(NDArray arr, int k = 0)
            => NDArray.triu_indices_from(arr, k);

        /// <summary>
        ///     Return the indices to access <c>(n, n)</c> arrays, given a masking function.
        /// </summary>
        /// <param name="n">The returned indices will be valid to access arrays of shape <c>(n, n)</c>.</param>
        /// <param name="mask_func">
        ///     A function whose call signature is <c>(arr, k)</c> and which returns
        ///     <paramref name="n"/>-by-<paramref name="n"/> masked arrays — e.g.
        ///     <see cref="triu"/> or <see cref="tril"/>.
        /// </param>
        /// <param name="k">An optional argument passed through to <paramref name="mask_func"/>.</param>
        /// <returns>The <c>N</c> index arrays of the locations where the mask function is non-zero.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="mask_func"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.mask_indices.html
        ///     <para>
        ///     The returned arity follows <paramref name="mask_func"/>'s output rank, not
        ///     <c>n</c>: passing <see cref="diag"/> — which reduces a 2-D input to its 1-D
        ///     diagonal — yields a <b>single</b> index array, matching NumPy (probed).
        ///     </para>
        ///     <para>
        ///     It stays a facade, NumPy's three statements here: the callback's result, of whatever class the callback
        ///     returns, goes through np.not_equal and np.nonzero, whose class rules NumPy's <c>a != 0</c> and
        ///     <c>nonzero(a)</c> apply to that class (np.apply_over_axes keeps its statements for the same reason).
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray<long>[] mask_indices(int n, Func<NDArray, int, NDArray> mask_func, int k = 0)
        {
            if (mask_func is null) throw new ArgumentNullException(nameof(mask_func));

            var m = np.ones(Shape.Matrix(n < 0 ? 0 : n, n < 0 ? 0 : n), NPTypeCode.Int64);
            var a = mask_func(m, k);

            return np.nonzero(np.not_equal(a, np.zeros(new Shape(1), a.typecode)));
        }
    }
}
