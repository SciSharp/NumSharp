using System;
using NumSharp.Backends.Iteration;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Converts a flat index or an array of flat indices into a tuple of coordinate arrays — NumPy's
        ///     <c>arr_unravel_index</c> (<c>numpy/_core/src/multiarray/compiled_base.c</c>), statement for statement.
        ///     Inverse of <see cref="ravel_multi_index(NDArray[], int[], string, char)"/>.
        /// </summary>
        /// <param name="indices">
        ///     An integer (or bool) array of flat indices into an array of dimensions <paramref name="shape"/>, of any
        ///     rank and layout. Read as int64: a uint64 index at or past 2^63 wraps negative and is out of bounds, as
        ///     NumPy's same_kind cast to intp makes it.
        /// </param>
        /// <param name="shape">
        ///     The dimensions to unravel into. An empty shape (NumPy's <c>()</c>) is legal: its one valid index is 0, and
        ///     only a 0-d index array may use it.
        /// </param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case.</param>
        /// <returns>
        ///     One int64 array per dimension of <paramref name="shape"/>, each of <paramref name="indices"/>' shape —
        ///     NumPy's <c>ret_arr[..., j]</c>: the COLUMNS of one interleaved buffer of shape
        ///     <c>indices.shape + (len(shape),)</c>, so every array is a strided VIEW (OWNDATA false, element stride
        ///     <c>len(shape)</c> along the fastest axis) and they share that buffer. The buffer is laid out along
        ///     <paramref name="indices"/>' iteration order: C for a C-ordered, broadcast or reversed index array, F for an
        ///     F-ordered one, the same permutation for a transposed one. A 0-d <paramref name="indices"/> gives NumPy's
        ///     scalars instead — fresh read-only 0-d arrays that own their value. An empty <paramref name="shape"/> with a
        ///     0-d index gives an empty array (NumPy's <c>()</c>).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> or <paramref name="shape"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     In NumPy's order: <paramref name="order"/> is not one of C / F / A / K in either case (<c>order must be one of
        ///     'C', 'F', 'A', or 'K' (got 'X')</c>, the argument parse's error); the dimensions' product overflows int64 or
        ///     is exactly -1 (<c>dimensions are too large; …</c>); then — after the <see cref="TypeError"/> below — an
        ///     <c>'A'</c> or <c>'K'</c> order (<c>only 'C' or 'F' order is permitted</c>); an index outside
        ///     <c>[0, prod(shape))</c>, the FIRST one in the iteration order (<c>index N is out of bounds for array with
        ///     size M</c>; a shape with a zero or an odd number of negative dimensions has no valid index); and, for an
        ///     empty <paramref name="shape"/>, an index array of rank 1 or more (<c>multiple indices are not supported for
        ///     0d arrays</c>, checked after the indices).
        /// </exception>
        /// <exception cref="TypeError"><paramref name="indices"/> is neither integer nor bool (<c>only int indices
        ///     permitted</c>; a float is refused, never truncated, and text is refused, never parsed).</exception>
        /// <exception cref="NotSupportedException">Runtime code generation is unavailable (the IL kernel cannot be
        ///     emitted).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html
        ///     <para><b>BREAKING (NumPy parity):</b> the coordinate arrays used to be separate owning C-contiguous arrays.
        ///     They are now views with an element stride of <c>len(shape)</c>, so code that reads their raw buffer
        ///     (<c>Storage.Address</c>, <c>Unsafe.Span</c>) must go through the strides or take
        ///     <c>np.ascontiguousarray</c> first. An out-of-range index raises <see cref="ValueError"/> (NumPy's type)
        ///     where it raised <see cref="ArgumentOutOfRangeException"/>, and a lowercase order letter is accepted.</para>
        ///     <para><b>Lifetime:</b> the shared buffer lives while any coordinate array does (each holds a counted
        ///     reference); disposing one coordinate array leaves the others valid.</para>
        ///     <para><b>Iteration order:</b> NumPy walks the indices with <c>NpyIter</c> (KEEPORDER,
        ///     <c>DONT_NEGATE_STRIDES</c>) and writes the coordinates of the k-th element it visits into the k-th row of
        ///     the buffer, which <c>NpyIter_CreateCompatibleStrides</c> lays out along that same order. NumSharp copies a
        ///     non-int64 or differently laid out index array into that layout (an int64 array already laid out so —
        ///     every C-contiguous one — is read in place) and runs one IL kernel over it linearly, so the rows, the
        ///     strides and the out-of-range index that is reported first are NumPy's.</para>
        ///     <para>The body is <see cref="TensorEngine.UnravelIndex"/> (the np-layer dtype plan): this facade computes
        ///     a <c>numpy.matrix</c> operand as a plain array and calls the operand's engine.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray<long>[] unravel_index(NDArray indices, int[] shape, char order = 'C')
        {
            using var plain = NDMatrixBoundary.PlainFor(indices);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return np.EngineOf(indices).UnravelIndex(indices, shape, order);
        }

        /// <summary>
        ///     Scalar convenience overload — converts a single flat index into its coordinates, the values NumPy's
        ///     <c>np.unravel_index(index, shape, order)</c> returns as a tuple of scalars, without NDArray wrapping.
        /// </summary>
        /// <param name="index">The flat index.</param>
        /// <param name="shape">The dimensions to unravel into; an empty shape (NumPy's <c>()</c>) accepts the index 0 and
        ///     returns no coordinate.</param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case.</param>
        /// <returns>One coordinate per dimension of <paramref name="shape"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="shape"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: an order letter that is not C / F / A / K; a dimensions' product
        ///     that overflows int64 or is exactly -1; an <c>'A'</c> or <c>'K'</c> order (<c>only 'C' or 'F' order is
        ///     permitted</c>); an index outside <c>[0, prod(shape))</c> (<c>index N is out of bounds for array with size
        ///     M</c>).</exception>
        public static long[] unravel_index(long index, int[] shape, char order = 'C')
        {
            if (shape is null) throw new ArgumentNullException(nameof(shape));
            NPY_ORDER npyOrder = OrderResolver.Convert(order);
            long unravelSize = Backends.DefaultEngine.UnravelSize(shape);
            bool cOrder = OrderResolver.RequireCOrFOrder(npyOrder);

            if (index < 0 || index >= unravelSize)
                throw new ValueError($"index {index} is out of bounds for array with size {unravelSize}");

            int ndim = shape.Length;
            // shape=(): index 0 (the only valid one) has no coordinate — NumPy's empty tuple.
            if (ndim == 0)
                return Array.Empty<long>();

            var coords = new long[ndim];
            if (cOrder)
            {
                // C-order: from innermost dim down to outermost; skip last divmod
                // because val < dims[0] after extracting the previous coords.
                long val = index;
                for (int d = ndim - 1; d > 0; d--)
                {
                    long m = shape[d];
                    coords[d] = val % m;
                    val /= m;
                }
                coords[0] = val;
            }
            else
            {
                long val = index;
                for (int d = 0; d < ndim - 1; d++)
                {
                    long m = shape[d];
                    coords[d] = val % m;
                    val /= m;
                }
                coords[ndim - 1] = val;
            }

            return coords;
        }

    }
}
