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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.outer.html</remarks>
        [NDScoped]
        public static NDArray outer(NDArray a, NDArray b, NDArray @out = null)
        {
            //multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)
            return multiply(np.expand_dims(a.ravel(), -1), np.expand_dims(b.ravel(), 0), @out);
        }
    }
}
