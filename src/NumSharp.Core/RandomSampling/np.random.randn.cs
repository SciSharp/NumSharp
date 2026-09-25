using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Return a sample (or samples) from the "standard normal" distribution.
        /// </summary>
        /// <param name="shape">Dimensions of the returned array (d0, d1, ..., dn).</param>
        /// <returns>
        ///     Array of floating-point samples from the standard normal distribution.
        /// </returns>
        /// <exception cref="ValueError">A dimension is negative.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
        ///     <br/>
        ///     NumPy signature: randn(d0, d1, ..., dn) where d0..dn are dimension sizes.
        ///     <br/>
        ///     For random samples from N(μ, σ²), use: σ * np.random.randn(...) + μ
        /// </remarks>
        public NDArray randn(params long[] shape)
        {
            if (shape.Length == 0)
                return standard_normal();
            return standard_normal(new Shape(shape));
        }

        /// <summary>
        ///     Return a scalar sample from the standard normal distribution.
        /// </summary>
        /// <typeparam name="T">The desired output type.</typeparam>
        /// <returns>A single random value.</returns>
        /// <remarks>Draws one cached-polar normal (NumPy's <c>legacy_gauss</c>) under the bit generator's lock, then converts it.</remarks>
        public T randn<T>()
        {
            double draw;
            lock (randomizer.@lock)
                draw = NextGaussian();
            return (T)Converts.ChangeType(draw, InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     Draw a single sample from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean ("centre") of the distribution. Default is 0.</param>
        /// <param name="scale">Standard deviation of the distribution. Must be non-negative. Default is 1.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>).</exception>
        public NDArray normal(double loc = 0.0, double scale = 1.0) => normal(loc, scale, Shape.Scalar);

        /// <summary>
        ///     Draw random samples from a normal (Gaussian) distribution.
        /// </summary>
        /// <param name="loc">Mean ("centre") of the distribution. Default is 0.</param>
        /// <param name="scale">Standard deviation (spread or "width") of the distribution. Must be non-negative. Default is 1.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized normal distribution (float64).</returns>
        /// <exception cref="ValueError"><paramref name="scale"/> is negative, including <c>-0.0</c> (<c>scale &lt; 0</c>), or
        ///     <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        ///     <br/>
        ///     The probability density function of the normal distribution, first derived
        ///     by De Moivre and 200 years later by both Gauss and Laplace independently,
        ///     is often called the bell curve because of its characteristic shape.
        ///     <br/>
        ///     NumPy's <c>legacy_normal</c>: <c>loc + scale * legacy_gauss</c> (the cached polar method). Holds the bit
        ///     generator's lock for the draws.
        /// </remarks>
        public NDArray normal(double loc, double scale, Shape size)
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
                        return NDArray.Scalar(LegacyNormal(ref one, loc, scale));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator), consumed by the two-phase polar fill (LegacyGaussFill);
                // then legacy_normal's loc + scale * g of each, in place — the same expression, so the same bits.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    LegacyGaussFill(ref src, dst, n);
                for (long i = 0; i < n; i++)
                    dst[i] = loc + scale * dst[i];
            }

            return ret;
        }

        /// <summary>
        ///     Draw a single sample from a standard Normal distribution.
        /// </summary>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        public NDArray standard_normal() => standard_normal(Shape.Scalar);

        /// <summary>
        ///     Draw samples from a standard Normal distribution (mean=0, stdev=1).
        /// </summary>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>A floating-point array of shape size of drawn samples.</returns>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
        ///     <br/>
        ///     NumPy's <c>legacy_gauss</c> directly (not <c>0 + 1 * gauss</c>: <c>standard_normal</c> returns the draw itself,
        ///     so a <c>-0.0</c> normal keeps its sign). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray standard_normal(Shape size)
        {
            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(LegacyGauss(ref one));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator), consumed by the two-phase polar fill (LegacyGaussFill).
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    LegacyGaussFill(ref src, dst, n);
            }

            return ret;
        }
    }
}
