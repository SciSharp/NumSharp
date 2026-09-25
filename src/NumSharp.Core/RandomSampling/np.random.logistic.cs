namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution. Default is 0.</param>
        /// <param name="scale">Scale parameter (must be &gt;= 0). Default is 1.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        public NDArray logistic(double loc = 0.0, double scale = 1.0) => logistic(loc, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a logistic distribution.
        /// </summary>
        /// <param name="loc">Mean of the distribution. Default is 0.</param>
        /// <param name="scale">Scale parameter (must be &gt;= 0). Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized logistic distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html
        ///     <br/>
        ///     The logistic distribution is used in extreme value problems, finance,
        ///     and for growth modeling. It is similar to the normal distribution but
        ///     has heavier tails.
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; μ, s) = exp(-(x-μ)/s) / (s * (1 + exp(-(x-μ)/s))^2)
        ///     <br/>
        ///     Mean = loc, Variance = scale^2 * pi^2 / 3
        ///     <br/>
        ///     NumPy's <c>random_logistic</c>: <c>loc + scale * log(U / (1 - U))</c>, redrawing only <c>U == 0</c>.
        ///     <c>scale == 0</c> STILL consumes a uniform per value (the former shortcut skipped the draw and desynchronized
        ///     every later value). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray logistic(double loc, double scale, Shape size)
        {
            RandomConstraints.Check(scale, "scale", ConstraintType.CONS_NON_NEGATIVE);

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(Distributions.RandomLogistic(ref one, loc, scale));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws at least one uniform.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = Distributions.RandomLogistic(ref src, loc, scale);
                    }
            }

            return ret;
        }
    }
}
