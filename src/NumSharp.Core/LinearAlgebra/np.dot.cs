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
        public static NDArray dot(NDArray a, NDArray b, NDArray @out = null)
        {
            var result = a.TensorEngine.Dot(a, b);
            if (@out is null)
                return result;

            // Port of NumPy's new_array_for_sum (numpy/_core/src/multiarray/common.c): the ndim,
            // dtype and C-array checks fire together as one message; only then is each dimension
            // compared. ISCARRAY = C-contiguous AND writeable (ALIGNED is always true here).
            if (@out.ndim != result.ndim
                || @out.typecode != result.typecode
                || !@out.Shape.IsContiguous
                || !@out.Shape.IsWriteable)
                throw new ValueError(
                    "output array is not acceptable (must have the right datatype, " +
                    "number of dimensions, and be a C-Array)");

            for (int d = 0; d < result.ndim; d++)
                if (result.shape[d] != @out.shape[d])
                    throw new ValueError("output array has wrong dimensions");

            // dtype already validated equal, so this is a straight copy of the freshly-computed
            // (and therefore non-overlapping) result into out.
            np.copyto(@out, result, "unsafe");
            return @out;
        }
    }
}
