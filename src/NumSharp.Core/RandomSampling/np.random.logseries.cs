using System.Runtime.CompilerServices;
namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a logarithmic series distribution.
        /// </summary>
        /// <param name="p">Shape parameter for the distribution. Must be in the range [0, 1).</param>
        /// <param name="size">Output shape; <c>null</c> (NumPy's <c>None</c>, the default) draws a single value.</param>
        /// <returns>Drawn samples from the parameterized logarithmic series distribution (int64).</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is outside <c>[0, 1)</c> or NaN
        ///     (<c>p &lt; 0, p &gt;= 1 or p is NaN</c>), or <paramref name="size"/> has a negative dimension.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html: P(k) = -p^k / (k *
        ///     ln(1-p)), positive integers (k >= 1) - the log series distribution Fisher, Corbet and Williams proposed in
        ///     1943 for species richness and occurrence.
        ///     <br/>
        ///     NumPy's <c>legacy_logseries</c> (Kemp's LK generator with the legacy <c>log(1 - p)</c> / <c>1 - exp(r*U)</c>
        ///     spellings) — byte-identical to <c>np.random.RandomState(seed).logseries</c>, including the rejection of a
        ///     count whose <c>floor</c> overflows. <c>p = 0</c> (and <c>-0.0</c>) returns 1 after one draw. int64 output
        ///     (NumPy returns C <c>long</c>). Holds the bit generator's lock for the draws.
        /// </remarks>
        public NDArray logseries(double p, Shape? size = null)
        {
            RandomConstraints.Check(p, "p", ConstraintType.CONS_BOUNDED_LT_0_1);

            // The per-call setup NumPy recomputes for every value, evaluated once (the same expressions — bit-neutral).
            var setup = new LegacyLogseriesSetup(p);

            if (IsScalarDraw(size.GetValueOrDefault()))
            {
                unsafe
                {
                    // A one-double buffer IS NumPy's per-draw call sequence.
                    double word;
                    var one = new DrawBufferDouble(randomizer, &word, 1);
                    lock (randomizer.@lock)
                        return NDArray.WritableScalar(LegacyLogseries(ref one, in setup));
                }
            }

            var ret = LegacyOutput(NPTypeCode.Int64, size.GetValueOrDefault());
            unsafe
            {
                var dst = (long*)ret.Address;
                long n = ret.size;
                // Read-ahead draws (bulk-filled by the bit generator): every value draws at least one uniform.
                double* storage = stackalloc double[DrawBufferDouble.Capacity];
                var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
                lock (randomizer.@lock)
                    for (long i = 0; i < n; i++)
                    {
                        src.Owed = n - i;
                        dst[i] = LegacyLogseries(ref src, in setup);
                    }
            }

            return ret;
        }

    }
}
