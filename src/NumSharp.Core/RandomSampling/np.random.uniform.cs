using System;
using System.Linq;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a uniform distribution.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be >= low. Default is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be &lt; high. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized uniform distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
        ///     <br/>
        ///     Samples are uniformly distributed over the half-open interval [low, high)
        ///     (includes low, but excludes high). In other words, any value within the
        ///     given interval is equally likely to be drawn by uniform.
        /// </remarks>
        public NDArray uniform(double low, double high, Shape size)
        {
            return uniform(low, high, size.dimensions);
        }

        /// <summary>
        ///     Draw samples from a uniform distribution.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be >= low. Default is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be &lt; high. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized uniform distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
        ///     <br/>
        ///     Samples are uniformly distributed over the half-open interval [low, high)
        ///     (includes low, but excludes high). In other words, any value within the
        ///     given interval is equally likely to be drawn by uniform.
        /// </remarks>
        public NDArray uniform(double low, double high, params int[] size) => uniform(low, high, Shape.ComputeLongShape(size));

        /// <summary>
        ///     Draw samples from a uniform distribution.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be >= low. Default is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be &lt; high. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized uniform distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
        ///     <br/>
        ///     Samples are uniformly distributed over the half-open interval [low, high)
        ///     (includes low, but excludes high). In other words, any value within the
        ///     given interval is equally likely to be drawn by uniform.
        /// </remarks>
        public NDArray uniform(double low, double high, params long[] size)
        {
            if (size == null || size.Length == 0)
            {
                var ret = new NDArray<double>(new Shape(1));
                var data = new double[] { low + randomizer.NextDouble() * (high - low) };
                ret.ReplaceData(data);
                return ret;
            }

            var result = new NDArray<double>(size);
            ArraySlice<double> resultArray = result.Data<double>();

            double diff = high - low;
            for (long i = 0; i < result.size; ++i)
                resultArray[i] = low + randomizer.NextDouble() * diff;

            result.ReplaceData(resultArray);
            return result;
        }

        /// <summary>
        ///     Draw samples from a uniform distribution with array boundaries.
        /// </summary>
        /// <param name="low">Lower boundary array.</param>
        /// <param name="high">Upper boundary array.</param>
        /// <param name="dtype">The dtype of the output NDArray.</param>
        /// <returns>Drawn samples.</returns>
        public NDArray uniform(NDArray low, NDArray high, Type dtype = null)
        {
            if (!low.shape.SequenceEqual(high.shape))
                throw new IncorrectShapeException();
            dtype = dtype ?? (low.dtype == high.dtype ? low.dtype : throw new IncorrectTypeException());

            var ret = low + rand(low.shape).astype(dtype) * (high - low);
            return dtype != null ? ret.astype(dtype) : ret;
        }
    }
}
