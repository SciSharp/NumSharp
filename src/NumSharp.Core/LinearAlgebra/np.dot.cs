namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        ///     Dot product of two arrays. See remarks.
        /// </summary>
        /// <param name="a">Lhs, First argument.</param>
        /// <param name="b">Rhs, Second argument.</param>
        /// <returns>Returns the dot product of a and b. If a and b are both scalars or both 1-D arrays then a scalar is returned; otherwise an array is returned. If out is given, then it is returned.</returns>
        /// <remarks>
        ///     https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html<br></br>
        ///     Specifically,<br></br>
        ///         - If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).<br></br>
        ///         - If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.<br></br>
        ///         - If either a or b is 0-D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a* b is preferred.<br></br>
        ///         - If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.<br></br>
        ///         - If a is an N-D array and b is an M-D array(where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:<br></br>
        ///           dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// </remarks>
        public static NDArray dot(in NDArray a, in NDArray b)
            => a.TensorEngine.Dot(a, b);
    }
}
