namespace NumSharp
{
    public static partial class np
    {

        /// <summary>
        ///     Dot product of two arrays. See remarks.
        /// </summary>
        /// <param name="a">Lhs, First argument.</param>
        /// <param name="b">Rhs, Second argument.</param>
        /// <param name="out">
        ///     Output argument. Unlike a ufunc's <c>out</c>, <c>np.dot</c>'s is STRICT (its
        ///     <c>new_array_for_sum</c> in <c>common.c</c>): it must have the EXACT result dtype, the
        ///     exact number of dimensions, be C-contiguous and writeable — no casting or broadcasting.
        ///     A dtype / ndim / non-C-array mismatch is <c>"output array is not acceptable (must have
        ///     the right datatype, number of dimensions, and be a C-Array)"</c>; a same-ndim shape
        ///     mismatch is <c>"output array has wrong dimensions"</c>. Returned as-is when given.
        /// </param>
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
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes
        ///     longer than 1 (<c>shape too large to be a matrix.</c>); an out array the strict check refuses (the two texts
        ///     above).</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands (NumPy's <c>shapes (…) and (…) not aligned</c>
        ///     ValueError text).</exception>
        [NDScoped]
        public static NDArray dot(NDArray a, NDArray b, NDArray @out = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a, b))
                return NDMatrixBoundary.Wrapped((a, b, @out), static s => dot(s.a, s.b, s.@out), @out);
            // NumPy's dot is PyArray_MatrixProduct2 (C): the text answers, the product and the out= check are the
            // engine's (TensorEngine.Dot).
            return a.TensorEngine.Dot(a, b, @out);
        }
    }
}
