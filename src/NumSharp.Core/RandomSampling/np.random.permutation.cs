namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an integer, randomly permute np.arange(x).</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html</remarks>
        public NDArray permutation(int x)
        {
            var nd = np.arange(x);
            np.random.shuffle(nd);

            return nd;
        }

        /// <summary>
        ///     Randomly permute a sequence, or return a permuted range.
        /// </summary>
        /// <param name="x">If x is an integer, randomly permute np.arange(x).</param>
        /// <returns>Permuted sequence or array range.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html</remarks>
        public NDArray permutation(NDArray x)
        {
            x = x.copy();
            np.random.shuffle(x);

            return x;
        }
    }
}
