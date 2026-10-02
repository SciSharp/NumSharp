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
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <returns>The float64 draws.</returns>
        /// <exception cref="OverflowException"><c>high - low</c> is not finite (NumPy's <c>OverflowError</c>).</exception>
        /// <exception cref="ValueError"><c>high - low</c> has its sign bit set (NumPy's <c>CONS_NON_NEGATIVE</c>: <c>-0.0</c> included).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
        ///     <br/><c>low + (high - low) * next_double()</c>, byte-identical to NumPy.
        /// </remarks>
        public NDArray uniform(double low = 0.0, double high = 1.0, Shape size = default)
        {
            double range = high - low;
            if (double.IsInfinity(range) || double.IsNaN(range))
                throw new OverflowException("high - low range exceeds valid bounds");
            CheckNonNegative(range, "high - low"); // NumPy CONS_NON_NEGATIVE on 'high - low'

            lock (_bitGenerator.@lock)
            {
                if (IsNoSize(size))
                    return NDArray.Scalar(low + range * _bitGenerator.NextDouble());

                // One bulk uniform fill, then low + range * u in place — NumPy's per-element expression.
                var ret = new NDArray(typeof(double), size, false);
                unsafe
                {
                    var p = (double*)ret.Address;
                    long n = ret.size;
                    _bitGenerator.FillDouble(p, n);
                    for (long i = 0; i < n; i++)
                        p[i] = low + range * p[i];
                }
                return ret;
            }
        }
    }
}
