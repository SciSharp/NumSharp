using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from an exponential distribution.
        /// </summary>
        public NDArray exponential(double scale = 1.0) => exponential(scale, Shape.Scalar);

        /// <summary>
        ///     Draw samples from an exponential distribution.
        /// </summary>
        /// <param name="scale">The scale parameter, β = 1/λ. Must be non-negative. Default is 1.0.</param>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the parameterized exponential distribution.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
        ///     <br/>
        ///     The exponential distribution is a continuous analogue of the geometric distribution.
        ///     It describes many common situations, such as the size of raindrops measured over
        ///     many rainstorms, or the time between page requests to Wikipedia.
        /// </remarks>
        public NDArray exponential(double scale, Shape size)
        {
            if (size.IsScalar || size.IsEmpty)
                return NDArray.Scalar(-Math.Log(1 - randomizer.NextDouble()) * scale);

            // Every step but the final `* scale` is an owning intermediate consumed exactly once.
            // Release each synchronously (`using`) instead of letting it ride the finalizer queue —
            // in a tight exponential() loop the un-disposed uniform / (1-u) / negate buffers
            // (≈400 KB each at 50K float64) accumulated as live allocations until GC, growing the
            // process working set (np.random.exponential leak guard). The trailing `* scale`
            // produces the fresh NDArray returned to (and owned by) the caller.
            using var u = uniform(0, 1, size);   // U(0,1)
            using var oneMinusU = 1 - u;         // 1 - U
            using var x = np.log(oneMinusU);     // log(1 - U)
            using var negX = np.negative(x);     // -log(1 - U)
            return negX * scale;                 // β · (-log(1 - U))
        }
    }
}
