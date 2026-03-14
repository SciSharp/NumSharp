namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return coordinate matrices from coordinate vectors.
        /// Make N-D coordinate arrays for vectorized evaluations of
        /// N-D scalar/vector fields over N-D grids, given
        /// one-dimensional coordinate arrays x1, x2,..., xn.
        /// .. versionchanged:: 1.9
        /// 1-D and 0-D cases are allowed.
        /// </summary>
        /// <param name="x1"> 1-D arrays representing the coordinates of a grid</param>
        /// /// <param name="x2"> 1-D arrays representing the coordinates of a grid</param>
        /// <returns></returns>
        public static (NDArray, NDArray) meshgrid(NDArray x1, NDArray x2, Kwargs kwargs = null)
        {
            if (kwargs == null)
            {
                kwargs = new Kwargs();
            }

            int ndim = 2;
            var s0 = (1, 1);
            var output = new NDArray[] {x1.reshape(x1.size, 1), x2.reshape(1, x2.size)};

            if (kwargs.indexing == "xy" && ndim > 1)
            {
                // Switch first and second axis
                output = new NDArray[] {x1.reshape(1, x1.size), x2.reshape(x2.size, 1)};
            }

            if (!kwargs.sparse)
            {
                // Return the full N-D matrix(not only the 1 - D vector)
                output = np.broadcast_arrays(output[0], output[1], true);
            }

            if (kwargs.copy)
            { }

            return (output[0], output[1]);
        }
    }

    public class Kwargs
    {
        public string indexing { get; set; }
        public bool sparse { get; set; }
        public bool copy { get; set; }

        /// <summary>
        /// Kwargs constructor
        /// </summary>
        /// <param name="indexing"> {'xy', 'ij'}, optional Cartesian('xy', default) or matrix('ij') indexing of output.</param>
        /// <param name="sparse">If True a sparse grid is returned in order to conserve memory. Default is False.</param>
        /// <param name="copy">If False, a view into the original arrays are returned in order to conserve memory.
        /// Default is True.Please note that sparse= False, copy= False`` will likely return non-contiguous arrays.  
        /// Furthermore, more than one element of a broadcast array may refer to a single memory location.
        /// If you need to write to the arrays, make copies first.</param>
        public Kwargs(string indexing, bool sparse, bool copy)
        {
            this.indexing = indexing;
            this.sparse = sparse;
            this.copy = copy;
        }

        public Kwargs()
        {
            this.indexing = "xy";
            this.sparse = false;
            this.copy = true;
        }
    }
}
