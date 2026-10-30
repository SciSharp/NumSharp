namespace NumSharp
{
    // ============================== _core/shape_base.py ==============================
    // NumPy's numpy/_core/shape_base.py statements as members of the array object: atleast_1d's and atleast_2d's for
    // one array. The np.atleast_1d facades and np.atleast_2d's NDArray facade keep their numpy.char.chararray rule and
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
    }
}
