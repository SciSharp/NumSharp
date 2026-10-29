namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from the Laplace distribution.
        /// </summary>
        /// <param name="loc">The position of the distribution peak.</param>
        /// <param name="scale">The exponential decay. Must be non-negative.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        /// <remarks>
        ///     Every distribution argument given, no size: one draw (a 0-d array, NumPy's size <c>()</c>, which draws
        ///     exactly what <c>None</c> draws). NumPy's defaults live on the size overload, which carries NumPy's whole
        ///     <c>laplace(loc=0.0, scale=1.0, size=None)</c> signature — so <c>laplace()</c>, <c>laplace(size: 3)</c>
        ///     and any argument left out bind there. This overload has no defaults on purpose: two overloads that both
        ///     need defaults filled in are ambiguous to C#, and the size-only call would not compile.
        /// </remarks>
        public NDArray laplace(double loc, double scale) => laplace(loc, scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from the Laplace or double exponential distribution with
        ///     specified location (or mean) and scale (decay).
        /// </summary>
        /// <param name="loc">The position of the distribution peak. Default is 0.</param>
        /// <param name="scale">The exponential decay. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized Laplace distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html
        ///     <br/>
        ///     The Laplace distribution is similar to the Gaussian/normal distribution,
        ///     but is sharper at the peak and has fatter tails. It represents the
        ///     difference between two independent, identically distributed exponential
        ///     random variables.
        ///     <br/>
        ///     The probability density function is:
        ///     f(x; μ, λ) = (1/2λ) * exp(-|x - μ| / λ)
        ///     <br/>
        ///     where μ is the location parameter and λ is the scale parameter.
        ///     <br/>
        ///     NumPy's <c>random_laplace</c> (shared by RandomState and Generator): <c>loc - scale*log(2 - 2U)</c> for
        ///     <c>U &gt;= 0.5</c>, <c>loc + scale*log(2U)</c> for <c>0 &lt; U &lt; 0.5</c>, redrawing <c>U == 0</c>.
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray laplace(double loc = 0.0, double scale = 1.0, Shape size = default)
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
                        return NDArray.Scalar(Distributions.RandomLaplace(ref one, loc, scale));
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
                        dst[i] = Distributions.RandomLaplace(ref src, loc, scale);
                    }
            }

            return ret;
        }
    }
}
