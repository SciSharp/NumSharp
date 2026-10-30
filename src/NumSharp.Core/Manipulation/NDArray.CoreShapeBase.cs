namespace NumSharp
{
    // ============================== _core/shape_base.py ==============================
    // NumPy's numpy/_core/shape_base.py statements as members of the array object: atleast_1d's for one array. The
    // np.atleast_1d facades keep their numpy.char.chararray rule and their null checks and call the member; engine code
    // (np.poly, np.roots, the polynomial arithmetic) calls it directly, inside its public function's boundary.
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
    }
}
