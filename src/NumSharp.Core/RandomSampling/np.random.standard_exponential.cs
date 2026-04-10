using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard exponential distribution (scale=1).</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_exponential.html
        ///     <br/>
        ///     The standard exponential distribution is the exponential distribution with scale=1.
        ///     It has mean=1 and variance=1.
        ///     <br/>
        ///     Equivalent to: exponential(scale=1.0, size=size)
        ///     <br/>
        ///     Uses inverse transform: X = -log(1 - U) where U ~ Uniform(0, 1)
        /// </remarks>
        public NDArray standard_exponential(Shape size = default)
        {
            if (size.IsEmpty)
            {
                return NDArray.Scalar(SampleStandardExponential());
            }

            var ret = new NDArray<double>(size);
            ArraySlice<double> data = ret.Data<double>();

            for (int i = 0; i < ret.size; i++)
            {
                data[i] = SampleStandardExponential();
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples from the standard exponential distribution.</returns>
        public NDArray standard_exponential(int[] size)
            => standard_exponential(new Shape(size));

        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape.</param>
        /// <returns>Drawn samples from the standard exponential distribution.</returns>
        public NDArray standard_exponential(params long[] size)
            => standard_exponential(new Shape(size));

        /// <summary>
        ///     Draw samples from the standard exponential distribution.
        /// </summary>
        /// <param name="size">Output shape as single int.</param>
        /// <returns>Drawn samples from the standard exponential distribution.</returns>
        public NDArray standard_exponential(int size)
            => standard_exponential(new Shape(size));

        /// <summary>
        ///     Sample a single value from the standard exponential distribution.
        /// </summary>
        /// <remarks>
        ///     Based on NumPy's random_standard_exponential in distributions.c:
        ///     X = -log(1 - U) where U ~ Uniform(0, 1)
        ///     Avoids U=0 to prevent -log(1) = 0 and U=1 to prevent -log(0) = infinity.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double SampleStandardExponential()
        {
            double U;
            // Ensure U is in (0, 1) to avoid log(0) = -infinity
            do
            {
                U = randomizer.NextDouble();
            } while (U == 0.0);

            return -Math.Log(1.0 - U);
        }
    }
}
