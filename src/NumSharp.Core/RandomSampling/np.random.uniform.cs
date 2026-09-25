using System;
using System.Linq;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw a single sample from a uniform distribution.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. Default is 0.</param>
        /// <param name="high">Upper boundary of the output interval. Default is 1.0.</param>
        /// <returns>A 0-d float64 array holding the draw.</returns>
        /// <exception cref="OverflowException"><c>high - low</c> is not finite (NumPy's <c>OverflowError: Range exceeds valid bounds</c>).</exception>
        public NDArray uniform(double low = 0.0, double high = 1.0) => uniform(low, high, Shape.Scalar);

        /// <summary>
        ///     Draw samples from a uniform distribution.
        /// </summary>
        /// <param name="low">Lower boundary of the output interval. All values generated will be >= low. Default is 0.</param>
        /// <param name="high">Upper boundary of the output interval. All values generated will be &lt; high. Default is 1.0.</param>
        /// <param name="size">Output shape; <c>default</c> (NumPy's <c>None</c>) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized uniform distribution (float64).</returns>
        /// <exception cref="OverflowException"><c>high - low</c> is infinite or NaN — NumPy's <c>OverflowError('Range exceeds
        ///     valid bounds')</c>, raised before anything is drawn (a NaN bound included).</exception>
        /// <exception cref="ValueError"><paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
        ///     <br/>
        ///     Samples are uniformly distributed over the half-open interval [low, high)
        ///     (includes low, but excludes high). In other words, any value within the
        ///     given interval is equally likely to be drawn by uniform.
        ///     <br/>
        ///     NumPy's <c>random_uniform</c>: <c>low + (high - low) * U</c>, one uniform per value (bulk-filled and
        ///     transformed in place). <c>high &lt; low</c> is legal and samples <c>(high, low]</c>, as in NumPy. Holds the bit
        ///     generator's lock for the draws.
        /// </remarks>
        public NDArray uniform(double low, double high, Shape size)
        {
            double range = high - low;
            if (!double.IsFinite(range))
                throw new OverflowException("Range exceeds valid bounds");

            if (IsScalarDraw(size))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.Scalar(Distributions.RandomUniform(ref one, low, range));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Double, size);
            unsafe
            {
                var dst = (double*)ret.Address;
                long n = ret.size;
                lock (randomizer.@lock)
                    randomizer.FillDouble(dst, n);
                for (long i = 0; i < n; i++)
                    dst[i] = low + range * dst[i];
            }

            return ret;
        }

        /// <summary>
        ///     Draw samples from a uniform distribution with array boundaries.
        /// </summary>
        /// <param name="low">Lower boundary array.</param>
        /// <param name="high">Upper boundary array.</param>
        /// <param name="dtype">The dtype of the output NDArray.</param>
        /// <returns>Drawn samples.</returns>
        /// <exception cref="IncorrectShapeException"><paramref name="low"/> and <paramref name="high"/> differ in shape.</exception>
        /// <exception cref="IncorrectTypeException"><paramref name="dtype"/> is null and the bounds differ in dtype.</exception>
        [NDScoped] // reclaims the rand draw, its astype, the (high-low) diff and the pre-cast ret
        public NDArray uniform(NDArray low, NDArray high, DType dtype = null)
        {
            if (!low.shape.SequenceEqual(high.shape))
                throw new IncorrectShapeException();
            dtype ??= low.typecode == high.typecode ? low.dtype : throw new IncorrectTypeException();

            var ret = low + rand(low.shape).astype(dtype) * (high - low);
            return dtype != null ? ret.astype(dtype) : ret;
        }
    }
}
