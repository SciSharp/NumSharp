using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters of the distribution (k elements, each &gt; 0; NaN is accepted and
        ///     samples NaN, as in NumPy; an empty vector draws nothing).</param>
        /// <param name="size">Output shape. The output has shape (*size, k) where k is the length of alpha; null (NumPy's
        ///     <c>None</c>) draws one vector of shape (k,).</param>
        /// <returns>Drawn samples from the Dirichlet distribution (float64). Each row sums to 1.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null (<c>object of type 'NoneType' has no len()</c>).</exception>
        /// <exception cref="ValueError">An element of <paramref name="alpha"/> is <c>&lt;= 0</c> (<c>alpha &lt;= 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html
        ///     <br/>
        ///     The Dirichlet distribution is a distribution over vectors x that fulfil:
        ///     - x_i > 0
        ///     - sum(x) = 1
        ///     <br/>
        ///     The probability density function is:
        ///     p(x) = (1/B(alpha)) * prod(x_i^(alpha_i - 1))
        ///     <br/>
        ///     Algorithm (NumPy's legacy <c>dirichlet</c>): for each row draw <c>Y_i = legacy_standard_gamma(alpha_i)</c>,
        ///     then MULTIPLY by the reciprocal of their sum (<c>Y_i * (1 / sum)</c> — not a division, which rounds
        ///     differently). A row whose gammas all underflow to 0 therefore becomes NaN (<c>0 * inf</c>), as in NumPy.
        ///     Byte-identical to <c>np.random.RandomState(seed).dirichlet</c>; holds the bit generator's lock for the draws.
        /// </remarks>
        public unsafe NDArray dirichlet(double[] alpha, Shape? size = null)
        {
            if (alpha is null)
                throw new TypeError("object of type 'NoneType' has no len()");

            fixed (double* alphaData = alpha)
                return DirichletCore(alphaData, alpha.Length, size);
        }

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters as a 1-D NDArray of any numeric dtype and layout (cast to float64).</param>
        /// <param name="size">Output shape; the output has shape (*size, k). Null (NumPy's <c>None</c>) draws one vector.</param>
        /// <returns>Drawn samples from the Dirichlet distribution (float64).</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null (<c>object of type 'NoneType' has no len()</c>) or 0-d
        ///     (<c>len() of unsized object</c>).</exception>
        /// <exception cref="ValueError"><paramref name="alpha"/> has more than one dimension (<c>object too deep for desired
        ///     array</c>), an element is <c>&lt;= 0</c> (<c>alpha &lt;= 0</c>), or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     NumPy converts alpha with <c>PyArray_FROMANY(alpha, NPY_DOUBLE, 1, 1)</c> after taking <c>len(alpha)</c>, so a
        ///     0-d array fails the <c>len</c> and a 2-D array the depth check. The float64 copy is a raw pooled buffer private
        ///     to this call, returned to the pool on every path.
        /// </remarks>
        public unsafe NDArray dirichlet(NDArray alpha, Shape? size = null)
        {
            if (alpha is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            if (alpha.ndim == 0)
                throw new TypeError("len() of unsized object");
            if (alpha.ndim > 1)
                throw new ValueError("object too deep for desired array");

            long k = alpha.size;

            // Copy alpha (any layout, any numeric dtype) into a flat double buffer via NDIter.Copy — handles
            // strided/broadcast alpha + any->double cast. This is a RAW pooled buffer (not an NDArray), invisible to
            // [NDScoped]; free it in the finally.
            var alphaBlock = new UnmanagedMemoryBlock<double>(k);
            var alphaSlice = new ArraySlice<double>(alphaBlock);
            try
            {
                var alphaStorage = new UnmanagedStorage(alphaSlice, new Shape(k));
                NDIter.Copy(alphaStorage, alpha.Storage);
                return DirichletCore((double*)alphaSlice.Address, k, size);
            }
            finally
            {
                alphaSlice.DangerousFree();   // return the raw alpha-copy buffer to the pool
            }
        }

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null.</exception>
        /// <exception cref="ValueError">An element of <paramref name="alpha"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        public NDArray dirichlet(double[] alpha, int[] size)
            => dirichlet(alpha, new Shape(size));

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null.</exception>
        /// <exception cref="ValueError">An element of <paramref name="alpha"/> is <c>&lt;= 0</c>, or a size dimension is negative.</exception>
        public NDArray dirichlet(double[] alpha, long[] size)
            => dirichlet(alpha, new Shape(size));

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters.</param>
        /// <param name="size">Number of samples to draw.</param>
        /// <returns>Drawn samples from the Dirichlet distribution.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null.</exception>
        /// <exception cref="ValueError">An element of <paramref name="alpha"/> is <c>&lt;= 0</c>, or <paramref name="size"/> is negative.</exception>
        public NDArray dirichlet(double[] alpha, int size)
            => dirichlet(alpha, new int[] { size });

        /// <summary>
        ///     The shared body of the <c>dirichlet</c> overloads — NumPy's legacy loop over a contiguous float64 alpha.
        /// </summary>
        /// <param name="alpha">The k concentration parameters (read-only here).</param>
        /// <param name="k">The number of categories (0 allowed: the output is empty and nothing is drawn).</param>
        /// <param name="size">The leading output shape, or null for a single vector.</param>
        /// <returns>The float64 output of shape (*size, k).</returns>
        /// <exception cref="ValueError">An element is <c>&lt;= 0</c> (<c>alpha &lt;= 0</c>) or a size dimension is negative.</exception>
        private unsafe NDArray DirichletCore(double* alpha, long k, Shape? size)
        {
            // NumPy's np.any(np.less_equal(alpha_arr, 0)): NaN compares false and is accepted.
            for (long j = 0; j < k; j++)
                if (alpha[j] <= 0)
                    throw new ValueError("alpha <= 0");

            // Output shape: (k,) for size=None, else (*size, k).
            long[] outputDims;
            // A default Shape is NumSharp's spelling of NumPy's size=None, like a null one.
            if (size is null || size.Value.IsEmpty)
            {
                outputDims = new[] { k };
            }
            else
            {
                var sizeVal = size.Value;
                outputDims = new long[sizeVal.NDim + 1];
                for (int d = 0; d < sizeVal.NDim; d++)
                    outputDims[d] = sizeVal.dimensions[d];
                outputDims[sizeVal.NDim] = k;
            }

            var ret = new NDArray(NPTypeCode.Double, new Shape(outputDims), false);
            var val = (double*)ret.Address;
            long totsize = ret.size;

            lock (randomizer.@lock)
            {
                // totsize is a multiple of k (k == 0 means totsize == 0), so the row stride never overruns.
                for (long i = 0; i < totsize; i += k)
                {
                    double acc = 0.0;
                    for (long j = 0; j < k; j++)
                    {
                        val[i + j] = LegacyStandardGamma(alpha[j]);
                        acc = acc + val[i + j];
                    }
                    double invacc = 1 / acc;
                    for (long j = 0; j < k; j++)
                        val[i + j] = val[i + j] * invacc;
                }
            }

            return ret;
        }
    }
}
