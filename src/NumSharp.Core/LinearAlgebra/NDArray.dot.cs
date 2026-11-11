namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Dot product of two arrays. See remarks.
        /// </summary>
        /// <param name="b">Rhs, Second argument.</param>
        /// <returns>Returns the dot product of a and b. If a and b are both scalars or both 1-D arrays then a scalar is returned; otherwise an array is returned. If out is given, then it is returned.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.dot.html<br></br>
        ///     Specifically,<br></br>
        ///         - If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).<br></br>
        ///         - If both a and b are 2-D arrays, it is matrix multiplication, but using matmul or a @ b is preferred.<br></br>
        ///         - If either a or b is 0-D(scalar), it is equivalent to multiply and using numpy.multiply(a, b) or a* b is preferred.<br></br>
        ///         - If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.<br></br>
        ///         - If a is an N-D array and b is an M-D array(where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b:<br></br>
        ///           dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
        /// </remarks>
        /// <param name="b">The second operand (NumPy's positional-only <c>other</c>).</param>
        /// <param name="out">NumPy's <c>out=</c>: the C-contiguous destination of the product, of exactly the result's
        ///     shape and dtype (<see cref="np.dot(NDArray, NDArray, NDArray)"/>'s rules), returned itself; <c>null</c>
        ///     allocates.</param>
        /// <returns>The dot product, or <paramref name="out"/> itself.</returns>
        /// <exception cref="ValueError">The shapes are not aligned, or <paramref name="out"/> is not acceptable (NumPy's
        ///     texts).</exception>
        public NDArray dot(NDArray b, NDArray @out = null)
            // NumPy's a.dot(other, out=None): without out the engine product as before, with it np.dot's out= path.
            => @out is null ? TensorEngine.Dot(this, b) : np.dot(this, b, @out);
    }
}
