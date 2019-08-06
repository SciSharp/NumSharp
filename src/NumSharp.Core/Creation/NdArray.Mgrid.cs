namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     nd_grid instance which returns a dense multi-dimensional “meshgrid”.
        ///     An instance of numpy.lib.index_tricks.nd_grid which returns an dense (or fleshed out) mesh-grid when indexed, so that each returned argument has the same shape.
        ///     The dimensions and number of the output arrays are equal to the number of indexing dimensions.If the step length is not a complex number, then the stop is not inclusive.
        /// </summary>
        /// <param name="rhs"></param>
        /// <returns>mesh-grid `ndarrays` all of the same dimensions</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.mgrid.html</remarks>
        public (NDArray, NDArray) mgrid(NDArray rhs)
        {
            return np.mgrid(this, rhs);
        }
    }
}
