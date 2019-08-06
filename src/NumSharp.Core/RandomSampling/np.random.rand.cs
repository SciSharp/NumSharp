using NumSharp.Utilities;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Random values in a given shape.
        ///     Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        /// </summary>
        public NDArray rand(params int[] size)
        {
            return rand(new Shape(size));
        }

        /// <summary>
        ///     Random values in a given shape.
        ///     Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        /// </summary>
        public NDArray rand(Shape shape)
        {
            NDArray ret = new NDArray(typeof(double), shape, false);

            unsafe
            {
                var addr = (double*)ret.Address;
                var incr = new NDCoordinatesIncrementor(ref shape);
                do
                {
                    *(addr + shape.GetOffset(incr.Index)) = randomizer.NextDouble();
                } while (incr.Next() != null);
            }

            return ret;
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        ///     Results are from the “continuous uniform” distribution over the stated interval. To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
        /// </summary>
        /// <param name="size">The samples</param>
        /// <returns></returns>
        public NDArray random_sample(params int[] size)
        {
            return rand(size);
        }

        /// <summary>
        ///     Return random floats in the half-open interval [0.0, 1.0).
        ///     Results are from the “continuous uniform” distribution over the stated interval. To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
        /// </summary>
        /// <param name="shape">The shape to randomize</param>
        /// <returns></returns>
        public NDArray random_sample(Shape shape)
        {
            return rand(shape);
        }
    }
}
