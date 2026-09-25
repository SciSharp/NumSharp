using System;
using System.Collections.Generic;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Generates a random sample from a given 1-D array (or from <c>np.arange(a)</c> when <paramref name="a"/>
        ///     is a 0-d integer).
        /// </summary>
        /// <param name="a">A 1-D array to sample from, or a 0-d integer population size.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value; <c>()</c> returns a 0-d array.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">Optional 1-D probabilities, one per entry of <paramref name="a"/>. If not given, the sample is uniform.</param>
        /// <returns>The drawn elements of <paramref name="a"/> (or, for an integer population, the drawn int32 indices).</returns>
        /// <exception cref="ValueError">
        ///     A 0-d population that is not an integer (<c>a must be 1-dimensional or an integer</c>) or not positive with
        ///     samples requested; a multi-dimensional or empty <paramref name="a"/>; a <paramref name="p"/> that is not 1-D,
        ///     has the wrong size, contains NaN or negatives, or does not sum to 1; a sample larger than the population
        ///     without replacement; fewer non-zero probabilities than samples without replacement.
        /// </exception>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d (<c>len() of unsized object</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        ///     <br/>
        ///     Byte-identical port of NumPy's legacy <c>RandomState.choice</c> (<c>mtrand.pyx</c>): with replacement it is
        ///     <c>randint(0, pop, shape)</c> (dtype <c>'l'</c>, int32 on win-amd64) or the <paramref name="p"/>-weighted
        ///     CDF search <c>searchsorted(random_sample(shape), side='right')</c>; without replacement it is
        ///     <c>permutation(pop)[:size]</c> or the weighted rounds. The messages are the legacy module's (they differ
        ///     from <see cref="Generator.choice(NDArray, Shape?, bool, NDArray, int, bool)"/>'s).
        /// </remarks>
        [NDScoped]
        public NDArray choice(NDArray a, Shape size = default, bool replace = true, NDArray p = null)
        {
            // `shape == None` means `shape == ()`, with the sample unpacked at the end (NumPy's is_scalar).
            bool isScalar = size.IsEmpty;
            Shape shape = isScalar ? Shape.Scalar : size;
            long count = isScalar ? 1 : shape.size; // np.prod(size); prod(()) == 1

            // ---- resolve the population (NumPy: a = np.asarray(a); legacy: 1-D or an integer only) ----
            long popSize;
            bool popExceedsInt64 = false;
            if (a.ndim == 0)
            {
                popSize = Generator.ScalarPopulation(a, "a must be 1-dimensional or an integer", out popExceedsInt64);
                if (!popExceedsInt64 && popSize <= 0 && count != 0)
                    throw new ValueError("a must be greater than 0 unless no samples are taken");
            }
            else if (a.ndim != 1)
            {
                throw new ValueError("a must be 1-dimensional");
            }
            else
            {
                popSize = a.shape[0];
                if (popSize == 0 && count != 0)
                    throw new ValueError("'a' cannot be empty unless no samples are taken");
            }

            double[] pw = null;
            if (p is not null)
                pw = Generator.ValidateChoiceProbabilities(p, popSize, legacyMessages: true);

            NDArray idx;
            lock (randomizer.@lock)
            {
                if (replace)
                {
                    if (pw is not null)
                    {
                        // cdf = p.cumsum(); cdf /= cdf[-1]; idx = cdf.searchsorted(random_sample(shape), 'right')
                        // then np.asarray(idx).astype(np.long, casting='unsafe') — int32 on win-amd64.
                        var cdf = np.cumsum(np.array(pw));
                        double total = Convert.ToDouble(cdf.GetAtIndex(cdf.size - 1));
                        cdf = cdf / total;
                        var uniformSamples = rand(shape);
                        idx = np.searchsorted(cdf, uniformSamples, "right").astype(np.int32);
                    }
                    else if (popExceedsInt64)
                    {
                        // randint(0, pop_size) with dtype 'l' (int32): the high cannot fit.
                        throw new ValueError("high is out of bounds for int32");
                    }
                    else
                    {
                        idx = randint(0, popSize, shape);
                    }
                }
                else
                {
                    if (!popExceedsInt64 && count > popSize)
                        throw new ValueError("Cannot take a larger sample than population when 'replace=False'");
                    if (count < 0)
                        throw new ValueError("Negative dimensions are not allowed");

                    if (pw is not null)
                    {
                        idx = LegacyChoiceNoReplaceWeighted(pw, count, shape);
                    }
                    else
                    {
                        // idx = self.permutation(pop_size)[:size]; idx.shape = shape
                        if (popExceedsInt64)
                            throw new OverflowException("Python int too large to convert to C long");
                        var perm = permutation(popSize);
                        idx = perm[$":{count}"].reshape(shape);
                    }
                }
            }

            // Use samples as indices for a if a is array-like (a 0-d idx takes one element -> 0-d result, which also
            // covers NumPy's `not is_scalar and idx.ndim == 0` special case).
            if (a.ndim == 0)
                return idx;
            return np.take(a, idx, axis: 0);
        }

        /// <summary>
        ///     Generates a random sample from <c>np.arange(a)</c>.
        /// </summary>
        /// <param name="a">The population size.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">Optional 1-D probabilities of length <paramref name="a"/>.</param>
        /// <returns>The drawn int32 indices.</returns>
        /// <exception cref="ValueError">See <see cref="choice(NDArray, Shape, bool, NDArray)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        /// </remarks>
        [NDScoped]
        public NDArray choice(long a, Shape size = default, bool replace = true, NDArray p = null)
            => choice(NDArray.Scalar(a), size, replace, p);

        /// <summary>
        ///     Generates a random sample from <c>np.arange(a)</c>.
        /// </summary>
        /// <param name="a">The population size.</param>
        /// <param name="size">Output shape. Default (NumPy's <c>None</c>) returns a single value.</param>
        /// <param name="replace">Whether the sample is with or without replacement. Default is True.</param>
        /// <param name="p">Optional 1-D probabilities of length <paramref name="a"/>.</param>
        /// <returns>The drawn int32 indices.</returns>
        /// <exception cref="ValueError">See <see cref="choice(NDArray, Shape, bool, NDArray)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
        /// </remarks>
        public NDArray choice(int a, Shape size = default, bool replace = true, NDArray p = null)
            => choice((long)a, size, replace, p);

        /// <summary>
        ///     The legacy weighted sampling without replacement (NumPy's <c>mtrand.pyx</c> choice, the
        ///     <c>p is not None</c> branch): rounds of <c>rand(size - n_uniq)</c> mapped through the found-zeroed
        ///     normalized CDF with <c>searchsorted(side='right')</c>, keeping each round's first-occurrence-unique hits.
        /// </summary>
        /// <param name="pw">The validated float64 probabilities (a private copy — found entries are zeroed in place).</param>
        /// <param name="size">The number of distinct samples.</param>
        /// <param name="shape">The output shape.</param>
        /// <returns>The int32 (<c>np.long</c>) indices shaped to <paramref name="shape"/>.</returns>
        /// <exception cref="ValueError">Fewer non-zero probabilities than <paramref name="size"/>.</exception>
        /// <remarks>The caller holds the bit generator's lock.</remarks>
        private unsafe NDArray LegacyChoiceNoReplaceWeighted(double[] pw, long size, Shape shape)
        {
            long d = pw.LongLength;
            long nonzero = 0;
            for (long i = 0; i < d; i++)
                if (pw[i] > 0) nonzero++;
            if (nonzero < size)
                throw new ValueError("Fewer non-zero entries in p than size");

            var found = new int[size]; // np.zeros(shape, dtype=np.long): int32 on win-amd64
            var seen = new HashSet<long>();
            var cdf = new double[d];
            long nUniq = 0;
            while (nUniq < size)
            {
                long need = size - nUniq;
                // x = self.rand(size - n_uniq): the legacy next_double draws.
                using NDArray x = rand(new Shape(need));
                double* xp = (double*)x.Address;

                for (long k = 0; k < nUniq; k++) pw[found[k]] = 0.0;
                double acc = 0.0;
                for (long i = 0; i < d; i++) { acc += pw[i]; cdf[i] = acc; }
                double last = cdf[d - 1];
                for (long i = 0; i < d; i++) cdf[i] /= last;

                for (long t = 0; t < need && nUniq < size; t++)
                {
                    long ins = BisectRight(cdf, xp[t]);
                    if (seen.Add(ins))
                        found[nUniq++] = (int)ins;
                }
            }

            return np.array(found).reshape(shape);
        }

        /// <summary>bisect_right on a sorted double[] (NumPy's <c>searchsorted(side='right')</c>): the first index whose value exceeds <paramref name="v"/>.</summary>
        /// <param name="a">The ascending CDF.</param>
        /// <param name="v">The probe value.</param>
        /// <returns>The insertion index.</returns>
        private static long BisectRight(double[] a, double v)
        {
            long lo = 0, hi = a.Length;
            while (lo < hi)
            {
                long mid = (lo + hi) >> 1;
                if (v < a[mid]) hi = mid;
                else lo = mid + 1;
            }
            return lo;
        }
    }
}
