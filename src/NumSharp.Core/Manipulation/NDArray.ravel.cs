namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a slice, the a copy will be made.</remarks>
        public NDArray ravel()
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return ravel(this, 'C');
        }

        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major, 'F' - column-major,
        ///     'A' - 'F' if F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - memory order.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        public NDArray ravel(char order)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            return ravel(this, order);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Ravel</c> (<c>numpy/_core/src/multiarray/shape.c</c>), the body of np.ravel and of
        ///     ndarray.ravel: a 1-D VIEW whenever the requested reading order walks <paramref name="a"/>'s memory front
        ///     to back, otherwise a copy read in that order (<c>PyArray_Flatten</c>).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="order"><c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>, in either case (see
        ///     <see cref="np.ravel(NDArray, char)"/>).</param>
        /// <returns>A view that shares <paramref name="a"/>'s memory (and its writeability), or a fresh copy; never
        ///     <paramref name="a"/> itself.</returns>
        /// <exception cref="ArgumentException">An order code other than C, F, A or K (the house order
        ///     error).</exception>
        /// <remarks>No class boundary runs here: np.ravel applies its numpy.char.chararray rule and ndarray.ravel its
        ///     matrix region before they call this, and an engine caller sits inside its public function's
        ///     boundary.</remarks>
        internal static NDArray ravel(NDArray a, char order)
        {
            var s = a.Shape;
            // PyArray_Ravel's order resolution: 'K' is C / F for a C- / F-contiguous array (0-d and size<=1 included);
            // 'A' is F only for a strictly F-contiguous array.
            char o;
            switch (order)
            {
                case 'C':
                case 'c':
                    o = 'C';
                    break;
                case 'F':
                case 'f':
                    o = 'F';
                    break;
                case 'A':
                case 'a':
                    o = s.IsFContiguous && !s.IsContiguous ? 'F' : 'C';
                    break;
                case 'K':
                case 'k':
                    o = s.IsContiguous ? 'C' : s.IsFContiguous ? 'F' : 'K';
                    break;
                default:
                    // The house error for an unknown order code.
                    OrderResolver.Resolve(order, s);
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }

            // PyArray_Ravel reshapes to (-1,), never to the array's own size: -1 skips PyArray_Newshape's same-shape shortcut
            // (a plain view that keeps the strides), so the view gets fresh C strides — a one-element array with a negative
            // or a stepped stride (np.arange(1)[::-1]) ravels to stride one element, as in NumPy.
            if (o == 'C' && s.IsContiguous)
                return a.reshape(-1L);

            // An F-contiguous array read column-major, or an array dense in another axis order read in memory order, is
            // its buffer from the view's first element on: one 1-D view with stride one element.
            if ((o == 'F' && s.IsFContiguous) || (o == 'K' && s.IsDenseInSomeAxisOrder()))
            {
                var vec = new Shape(new long[] { a.size }, new long[] { 1 }, s.offset, s.bufferSize);
                // NumPy builds the view with the input's flags: a read-only input gives a read-only view.
                if (!s.IsWriteable)
                    vec = vec.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                return new NDArray(a.Storage.Alias(vec)) { TensorEngine = a.TensorEngine };
            }

            // PyArray_Flatten: a copy, read in the resolved order.
            return a.flatten(o);
        }
    }
}
