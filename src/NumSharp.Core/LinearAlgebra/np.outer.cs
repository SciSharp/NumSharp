namespace NumSharp
{
    public partial class np
    {

        /// <summary>
        ///     Compute the outer product of two vectors.
        ///     Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the outer product[R60] is:
        /// </summary>
        /// <param name="a">First input vector. Input is flattened if not already 1-dimensional.</param>
        /// <param name="b">Second input vector. Input is flattened if not already 1-dimensional.</param>
        /// <param name="out">
        ///     A location into which the result is stored. Its shape must be <c>(a.size, b.size)</c>.
        ///     Returned as-is when given. NumPy computes <c>outer</c> as a single <c>multiply</c>, so
        ///     <paramref name="out"/> follows the ufunc rules exactly — it joins the broadcast (a
        ///     shape mismatch is the ufunc broadcast error) and accepts a same_kind cast from the
        ///     product dtype.
        /// </param>
        /// <returns>out[i, j] = a[i] * b[j]</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="TypeError">Operands the multiply has no loop for (text with numbers, two texts).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="System.ArgumentException">An <paramref name="out"/> the product does not reach under same_kind,
        ///     or one the products do not broadcast into.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.outer.html
        ///     <para>NumPy's statement <c>multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)</c> is
        ///     <see cref="TensorEngine.Outer"/>, on <paramref name="a"/>'s engine, called inside the matrix boundary's plain
        ///     region (a numpy.matrix operand is computed as a plain array).</para>
        /// </remarks>
        public static NDArray outer(NDArray a, NDArray b, NDArray @out = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return a.TensorEngine.Outer(a, b, @out);
        }
    }
}
