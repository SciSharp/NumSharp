using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random values in a given shape.
        /// </summary>
        /// <param name="d0">Dimension(s) of the returned array.</param>
        /// <returns>Random values in the shape (d0, d1, ..., dn).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        ///     <br/>
        ///     Create an array of the given shape and populate it with random samples
        ///     from a uniform distribution over [0, 1).
        ///     <br/>
        ///     This is a convenience function for users porting code from Matlab.
        ///     For new code, use np.random.random_sample instead.
        /// </remarks>
        public NDArray rand(params int[] d0)
        {
            return rand(new Shape(d0));
        }

        /// <summary>
        ///     Random values in a given shape.
        /// </summary>
        /// <param name="shape">Shape of the returned array.</param>
        /// <returns>Random values.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        ///     <br/>
        ///     Create an array of the given shape and populate it with random samples
        ///     from a uniform distribution over [0, 1).
        /// </remarks>
        public NDArray rand(Shape shape)
        {
            NDArray ret = new NDArray(typeof(double), shape, false);

            unsafe
            {
                var addr = (double*)ret.Address;
                var incr = new ValueCoordinatesIncrementor(ref shape);
                do
                {
                    *(addr + shape.GetOffset(incr.Index)) = randomizer.NextDouble();
                } while (incr.Next() != null);
            }

            return ret;
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Array of random floats of shape size.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html
        ///     <br/>
        ///     Results are from the "continuous uniform" distribution over the stated interval.
        ///     To sample Unif[a, b), b > a, multiply the output by (b-a) and add a.
        /// </remarks>
        public NDArray random_sample(params int[] size)
        {
            return rand(size);
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Array of random floats.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html
        /// </remarks>
        public NDArray random_sample(Shape size)
        {
            return rand(size);
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Array of random floats.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
        ///     <br/>
        ///     Alias for random_sample. This is the preferred function for new code.
        /// </remarks>
        public NDArray random(params int[] size)
        {
            return random_sample(size);
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Array of random floats.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
        ///     <br/>
        ///     Alias for random_sample. This is the preferred function for new code.
        /// </remarks>
        public NDArray random(Shape size)
        {
            return random_sample(size);
        }
    }
}
