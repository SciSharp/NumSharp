namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     True for a 0-d array that the caller built as an ARRAY (<c>np.array(2)</c>, <c>np.asarray(2)</c>): NumPy's
        ///     0-d <c>ndarray</c>, as opposed to the 0-d array that stands for a NumPy SCALAR (the result of
        ///     <c>a.sum()</c>, <c>np.dot(v, v)</c>, <c>a[0, 0]</c>, <see cref="Scalar{T}(T)"/>).
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumSharp has no scalar type: a 0-d <see cref="NDArray"/> plays both roles, and almost every operation
        ///     treats the two alike, as NumPy 2 does (NEP 50 promotes them the same way). One NumPy operator tells
        ///     them apart: <c>matrix.__mul__</c> tests <c>isinstance(other, ndarray)</c> BEFORE <c>isscalar(other)</c>.
        ///     So <c>m * np.array(2)</c> is <c>np.dot(m, asmatrix(np.array(2)))</c>, the product with a 1 x 1 matrix,
        ///     which raises "shapes (2,3) and (1,1) not aligned" unless the matrix has one column, while
        ///     <c>m * np.float64(2)</c> and <c>m * m.sum()</c> scale. <see cref="NDMatrix"/> reads this mark to make
        ///     the same choice; nothing else reads it.
        ///     </para>
        ///     <para>
        ///     <b>Unmarked means scalar.</b> Only the two explicit spellings set the mark, so a 0-d result of any
        ///     computation keeps scaling a matrix. The functions NumSharp's own kernels allocate their 0-d results
        ///     with (<c>np.zeros</c>, <c>np.empty</c>, <see cref="Scalar{T}(T)"/>) do NOT set it: marking there would
        ///     turn computed scalars into arrays and make <c>m * np.linalg.norm(a)</c> raise. The cost: a 0-d array
        ///     built another way (<c>np.zeros(())</c>, <c>x.reshape(())</c>, <c>np.squeeze(x)</c>) still scales a
        ///     matrix, where NumPy raises.
        ///     </para>
        ///     <para>
        ///     The mark belongs to this one object, not to its storage: a view, a copy, a cast or an operation
        ///     result is a new array without it. <c>np.array(nd)</c> and <c>np.asarray(nd)</c> of a marked array
        ///     carry it over, as NumPy returns a 0-d array from both.
        ///     </para>
        /// </remarks>
        internal bool IsExplicitZeroDimArray { get; private set; }

        /// <summary>
        ///     Marks this array as an explicit 0-d ARRAY (see <see cref="IsExplicitZeroDimArray"/>). Call it only on an
        ///     array a public creation function is about to return for a scalar argument; never on a computed result.
        /// </summary>
        /// <returns>This array, for chaining. An array with one or more dimensions is returned unmarked: the mark
        ///     has a meaning for 0-d arrays only.</returns>
        internal NDArray MarkExplicitZeroDimArray()
        {
            if (Shape.IsScalar)
                IsExplicitZeroDimArray = true;
            return this;
        }
    }
}
