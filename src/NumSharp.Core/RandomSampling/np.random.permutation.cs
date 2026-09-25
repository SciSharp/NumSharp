namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Randomly permute <c>np.arange(x)</c>.
        /// </summary>
        /// <param name="x">The length of the range.</param>
        /// <returns>
        ///     A shuffled range of dtype <c>np.result_type(x, np.long)</c> — int32 on the win-amd64 reference build
        ///     (a Python int is weak, so the C <c>long</c> decides).
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        /// </remarks>
        public NDArray permutation(int x)
        {
            var nd = np.arange((double)x, np.int32);
            shuffle(nd);
            return nd;
        }

        /// <summary>
        ///     Randomly permute <c>np.arange(x)</c> (a C# <c>long</c> is a weak integer, like the Python int it stands for).
        /// </summary>
        /// <param name="x">The length of the range.</param>
        /// <returns>A shuffled int32 range (int64 when <paramref name="x"/> exceeds the int32 range, where NumPy's int32 arange would fail).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        /// </remarks>
        public NDArray permutation(long x)
        {
            if (x <= int.MaxValue)
                return permutation((int)x);
            var nd = np.arange(x);
            shuffle(nd);
            return nd;
        }

        /// <summary>
        ///     Randomly permute a sequence (a copy of <paramref name="x"/>; for a multi-dimensional array only the first
        ///     axis is permuted).
        /// </summary>
        /// <param name="x">The array to permute (at least 1-D).</param>
        /// <returns>A permuted copy; <paramref name="x"/> is not modified.</returns>
        /// <exception cref="IndexError"><paramref name="x"/> is 0-d (<c>x must be an integer or at least 1-dimensional</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html
        ///     <br/>
        ///     NumPy's legacy form: a 1-D input is copied and shuffled in place; an N-D input shuffles an intp index
        ///     vector (the same <c>random_interval</c> draws) and returns <c>x[idx]</c>.
        /// </remarks>
        [NDScoped] // reclaims the N-D path's index vector; the gathered copy is yielded
        public NDArray permutation(NDArray x)
        {
            if (x.ndim < 1)
                throw new IndexError("x must be an integer or at least 1-dimensional");

            if (x.ndim == 1)
            {
                var arr = x.copy();
                shuffle(arr);
                return arr;
            }

            var idx = np.arange(x.shape[0]);
            shuffle(idx);
            return np.take(x, idx, axis: 0);
        }
    }
}
