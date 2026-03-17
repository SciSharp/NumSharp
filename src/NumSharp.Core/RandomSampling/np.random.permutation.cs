namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an integer, randomly permute np.arange(x).</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        ///     <br/>
        ///     If x is an integer, randomly permute np.arange(x).
        ///     If x is an array, make a copy and shuffle the elements randomly.
        /// </remarks>
        public NDArray permutation(int x)
        {
            var nd = np.arange(x);
            shuffle(nd);
            return nd;
        }

        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an array, make a copy and shuffle the elements randomly.</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        ///     <br/>
        ///     If x is a multi-dimensional array, it is only shuffled along its first index.
        /// </remarks>
        public NDArray permutation(NDArray x)
        {
            x = x.copy();
            shuffle(x);
            return x;
        }
    }
}
