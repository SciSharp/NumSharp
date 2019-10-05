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
        /// <returns>out[i, j] = a[i] * b[j]</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.outer.html</remarks>
        public static NDArray outer(in NDArray a, in NDArray b)
        {
            //multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)
            return multiply(np.expand_dims(a.ravel(), -1), np.expand_dims(b.ravel(), 0));
        }
    }
}
