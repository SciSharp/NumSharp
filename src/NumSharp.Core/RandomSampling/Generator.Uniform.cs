using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Draw samples from a uniform distribution over <c>[low, high)</c>.
        /// </summary>
        /// <param name="low">Lower boundary (inclusive). Default 0.</param>
        /// <param name="high">Upper boundary (exclusive). Default 1.</param>
        /// <param name="size">Output shape.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
        ///     <br/><c>low + (high - low) * next_double()</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray uniform(double low = 0.0, double high = 1.0, Shape size = default)
        {
            double range = high - low;
            if (double.IsInfinity(range) || double.IsNaN(range))
                throw new OverflowException("high - low range exceeds valid bounds");

            if (IsNoSize(size))
                return NDArray.Scalar(low + range * _bitGenerator.NextDouble());

            return FillDoubleDist(size, () => low + range * _bitGenerator.NextDouble());
        }
    }
}
