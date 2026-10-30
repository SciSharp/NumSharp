namespace NumSharp
{
    // ============================== _core/shape_base.py ==============================
    // NumPy's numpy/_core/shape_base.py statements as members of the array object: atleast_1d's and atleast_2d's for
    // one array, and the ndmin padding (AtLeastNdView) np.array, np.block's _atleast_nd and np.r_ share. The
    // np.atleast_1d facades and np.atleast_2d's NDArray facade keep their numpy.char.chararray rule and
    // their null checks and call the members; engine code (np.poly, np.roots, the polynomial arithmetic, np.cov) calls
    // them directly, inside its public function's boundary. np.atleast_2d(object) keeps its own statements: an array
    // handed to it as an object passes through np.asanyarray, which keeps a chararray's class, and its np.expand_dims
    // then applies the chararray rule, which these plain statements do not.
    public partial class NDArray
    {
        // ---------------------------- np.atleast_1d ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.atleast_1d(a)</c> for one array after the facade's numpy.char.chararray rule: the array
        ///     itself when it has one dimension or more, else its reshape to <c>(1,)</c>.
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= 1</c> (so a caller must not dispose the result unless
        ///     it is not <paramref name="a"/>), else a <c>(1,)</c> VIEW of the 0-d array's element: it shares the
        ///     buffer, keeps it alive, and writes through.</returns>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     facade applies its rules before it calls this.</remarks>
        internal static NDArray AtLeast1d(NDArray a)
        {
            switch (a.ndim)
            {
                case 0:
                    return a.reshape(1);
                default:
                    return a;
            }
        }

        // ---------------------------- np.atleast_2d ----------------------------

        /// <summary>
        ///     NumPy's <c>numpy.atleast_2d(a)</c> for one array after the facade's numpy.char.chararray rule: the array
        ///     itself when it has two dimensions or more, a <c>(1, 1)</c> reshape of a 0-d array, and a <c>(1, N)</c>
        ///     view of a 1-d one (NumPy's <c>ary[newaxis, :]</c>).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= 2</c> (so a caller must not dispose the result
        ///     unless it is not <paramref name="a"/>), else a VIEW of <paramref name="a"/>'s elements with the leading
        ///     axes added: it shares the buffer, keeps it alive, and writes through.</returns>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     facade applies its rules before it calls this. The 1-d case is np.expand_dims(a, 0)'s statement
        ///     (<see cref="ExpandDims(NDArray, int)"/>) without that facade's chararray rule.</remarks>
        internal static NDArray AtLeast2d(NDArray a)
        {
            switch (a.ndim)
            {
                case 0:
                    return a.reshape(1, 1);
                case 1:
                    return ExpandDims(a, 0);
                default:
                    return a;
            }
        }

        // ---------------------------- ndmin (np.array, np.block, np.r_) ----------------------------

        /// <summary>
        ///     NumPy's <c>array(a, ndmin=ndim, copy=None)</c> over an array: <paramref name="a"/> with ones prepended to its
        ///     shape up to <paramref name="ndim"/> dimensions, always as a VIEW of its buffer (np.array's ndmin,
        ///     np.block's <c>_atleast_nd</c>, np.r_ / np.c_'s directive ndmin).
        /// </summary>
        /// <param name="a">The array (not null).</param>
        /// <param name="ndim">The minimum number of dimensions; 0 or less, or no more than <paramref name="a"/>'s, adds
        ///     none. It can come from a user's text (np.r_'s directive), so the padding costs O(ndim), never more.</param>
        /// <returns><paramref name="a"/> itself when <c>a.ndim &gt;= ndim</c> (so a caller must not dispose the result unless
        ///     it is not <paramref name="a"/>), else a VIEW of <paramref name="a"/>'s buffer with the leading length-1 axes
        ///     added: it shares the buffer, keeps it alive and writes through. A non-empty <paramref name="a"/> keeps its
        ///     strides, flags and writeability (one alias over <see cref="Shape.PrependDimensions"/>); an empty one is
        ///     reshaped to the padded dimensions.</returns>
        /// <exception cref="System.OutOfMemoryException">An <paramref name="ndim"/> too large to allocate a shape for, the
        ///     answer any other oversized NumSharp request gives.</exception>
        /// <remarks>No class boundary runs here: an engine caller sits inside its public function's boundary, and the
        ///     callers np.array's conversion, np.block (a matrix leaf passes unpadded) and np.r_ / np.c_ apply their
        ///     rules before they call this.</remarks>
        internal static NDArray AtLeastNdView(NDArray a, int ndim)
        {
            if (a.ndim >= ndim)
                return a;

            if (a.size == 0)
            {
                // expand_dims early-returns empty arrays unchanged; reshape an
                // empty directly instead (no data, layout is irrelevant).
                var dims = new long[ndim];
                int pad = ndim - a.ndim;
                for (int i = 0; i < pad; i++)
                    dims[i] = 1;
                for (int i = 0; i < a.ndim; i++)
                    dims[pad + i] = a.Shape.dimensions[i];
                return a.reshape(Shape.Adopt(dims));
            }

            // ONE alias, not one per axis. `ndim` can be user-supplied (np.r_'s ndmin comes
            // from a typed directive string), and expand_dims-per-axis clones dims+strides
            // every step — quadratic, 27.6 s at ndim=100 000. PrependDimensions is the
            // closed form of that loop, so the strides, flags and writeable bit are
            // unchanged; only the cost is.
            return new NDArray(a.Storage.Alias(a.Shape.PrependDimensions(ndim - a.ndim)))
                { TensorEngine = a.TensorEngine };
        }
    }
}
