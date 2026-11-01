using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range: NumPy reads <c>ary.shape[axis]</c> first,
        ///     a Python tuple index ("tuple index out of range"), so a 0-d array refuses every axis.</exception>
        /// <exception cref="ValueError">The sections do not divide the axis ("array split does not result in an equal
        ///     division"), or are not positive ("number sections must be larger than 0.").</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray[] split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices_or_sections, axis), static s => split(s.ary, s.indices_or_sections, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices_or_sections, axis), static s => split(s.ary, s.indices_or_sections, s.axis));
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's statements (the axis read first, the count's checks, the equal views) are the array object's.
            return NDArray.split(ary, indices_or_sections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// The split points along <paramref name="axis"/>: <c>[2, 3]</c> gives <c>ary[:2]</c>, <c>ary[2:3]</c>, <c>ary[3:]</c>.
        /// Each point is clamped as a slice bound (a negative one counts from the end), and unsorted points give empty
        /// sub-arrays.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns><c>indices.Length + 1</c> sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null, or <paramref name="indices"/> is.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).
        /// </remarks>
        public static NDArray[] split(NDArray ary, long[] indices, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices, axis), static s => split(s.ary, s.indices, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices, axis), static s => split(s.ary, s.indices, s.axis));
            // np.array_split's check, which this overload reached first when it called that facade.
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's split with split points is array_split; the array object's checks the points for null.
            return NDArray.split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// The split points along <paramref name="axis"/>: <c>[2, 3]</c> gives <c>ary[:2]</c>, <c>ary[2:3]</c>, <c>ary[3:]</c>.
        /// Each point is clamped as a slice bound (a negative one counts from the end), and unsorted points give empty
        /// sub-arrays.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns><c>indices.Length + 1</c> sub-arrays as views into ary.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null, or <paramref name="indices"/> is.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).
        /// </remarks>
        public static NDArray[] split(NDArray ary, int[] indices, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices, axis), static s => split(s.ary, s.indices, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices, axis), static s => split(s.ary, s.indices, s.axis));
            // np.array_split's check, which this overload reached first when it called that facade.
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's split with split points is array_split; the array object's checks the points for null.
            return NDArray.split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <remarks>
        /// The only difference between split and array_split is that array_split allows
        /// indices_or_sections to be an integer that does not equally divide the axis.
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">The number of sections (sizes differ by at most one).</param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>The sub-arrays, views into <paramref name="ary"/>.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null.</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range": NumPy's
        ///     <c>Ntotal = ary.shape[axis]</c> runs first).</exception>
        /// <exception cref="ValueError"><paramref name="indices_or_sections"/> is not positive ("number sections must be
        ///     larger than 0.").</exception>
        public static NDArray[] array_split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices_or_sections, axis), static s => array_split(s.ary, s.indices_or_sections, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices_or_sections, axis), static s => array_split(s.ary, s.indices_or_sections, s.axis));
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's statements (`Ntotal = ary.shape[axis]`, the count's check, the sections) are the array object's.
            return NDArray.array_split(ary, indices_or_sections, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// The split points along <paramref name="axis"/>: <c>[2, 3]</c> gives <c>ary[:2]</c>, <c>ary[2:3]</c>, <c>ary[3:]</c>.
        /// Each point is clamped as a slice bound (a negative one counts from the end), and unsorted points give empty
        /// sub-arrays.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns><c>indices.Length + 1</c> sub-arrays, views into <paramref name="ary"/>.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null, or <paramref name="indices"/> is (checked
        ///     in that order, before the axis).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, long[] indices, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices, axis), static s => array_split(s.ary, s.indices, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices, axis), static s => array_split(s.ary, s.indices, s.axis));
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's statements are the array object's; it checks the points for null (every split-points route, the
            // h / v / d splits after their rank check, reaches that check there).
            return NDArray.array_split(ary, indices, axis);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices">
        /// The split points along <paramref name="axis"/>: <c>[2, 3]</c> gives <c>ary[:2]</c>, <c>ary[2:3]</c>, <c>ary[3:]</c>.
        /// Each point is clamped as a slice bound (a negative one counts from the end), and unsorted points give empty
        /// sub-arrays.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns><c>indices.Length + 1</c> sub-arrays, views into <paramref name="ary"/>.
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null, or <paramref name="indices"/> is (checked
        ///     in that order, before the axis).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range ("tuple index out of range"; a 0-d array
        ///     refuses every axis).</exception>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives chararrays over the results (NumPy creates them as the
        ///     subclass).
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int[] indices, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(ary))
                return NDMatrixBoundary.WrappedAll((ary, indices, axis), static s => array_split(s.ary, s.indices, s.axis));
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(ary))
                return NDCharArrayBoundary.KeptAll((ary, indices, axis), static s => array_split(s.ary, s.indices, s.axis));
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // NumPy's statements are the array object's; it checks the points for null (see the long[] overload).
            return NDArray.array_split(ary, indices, axis);
        }
    }
}
