using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random values in a given shape.
        /// </summary>
        /// <param name="shape">Dimensions of the returned array (d0, d1, ..., dn).</param>
        /// <returns>Random values.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        ///     <br/>
        ///     Create an array of the given shape and populate it with random samples
        ///     from a uniform distribution over [0, 1).
        ///     <br/>
        ///     NumPy signature: rand(d0, d1, ..., dn) where d0..dn are dimension sizes.
        /// </remarks>
        public NDArray rand(params long[] shape)
        {
            if (shape.Length == 0)
                return NDArray.Scalar(randomizer.NextDouble());
            return rand(new Shape(shape));
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

            // Handle empty arrays (any dimension is 0)
            if (shape.size == 0)
                return ret;

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
        public NDArray random_sample(params long[] size) => rand(size);

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Array of random floats.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
        ///     <br/>
        ///     Alias for random_sample.
        /// </remarks>
        public NDArray random(params long[] size) => random_sample(size);
    }
}
