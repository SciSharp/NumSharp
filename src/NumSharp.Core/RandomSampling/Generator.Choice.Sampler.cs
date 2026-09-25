using System;
using System.Collections.Generic;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Generates a random sample from a given array (or <c>arange(a)</c> when <paramref name="a"/>
        ///     is an integer population size).
        /// </summary>
        /// <param name="a">The population: a 0-d integer (or bool) array for <c>arange(a)</c>, else the array sampled along <paramref name="axis"/>.</param>
        /// <param name="size">Output shape; null is NumPy's <c>size=None</c> (one sample, the element/sub-array itself), <c>()</c> a 0-d request.</param>
        /// <param name="replace">Whether a population member may be drawn more than once.</param>
        /// <param name="p">Optional 1-D probabilities, one per population member; must sum to 1 within a tolerance tied to its float dtype.</param>
        /// <param name="axis">The axis of <paramref name="a"/> sampled along (Python-style negative indexing).</param>
        /// <param name="shuffle">Without replacement and without <paramref name="p"/>: whether the selection is shuffled (false keeps Floyd's order).</param>
        /// <returns>
        ///     For an integer population, the int64 indices (0-d for <c>size=None</c>/<c>()</c>); otherwise
        ///     <c>a.take(indices, axis)</c> — a single element (0-d) or one sub-array when <c>size=None</c>.
        /// </returns>
        /// <exception cref="ValueError">
        ///     A 0-d population that is not an integer; a non-positive population or an empty <paramref name="a"/>
        ///     with samples requested; a <paramref name="p"/> that is not 1-D, has the wrong size, contains NaN or
        ///     negatives, or does not sum to 1; a sample larger than the population without replacement; fewer
        ///     non-zero probabilities than samples without replacement.
        /// </exception>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d (NumPy evaluates <c>len(p)</c> first).</exception>
        /// <exception cref="IndexError"><paramref name="axis"/> is outside <c>[-a.ndim, a.ndim)</c> (NumPy reads <c>a.shape[axis]</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
        ///     <br/>Byte-identical to NumPy on every sampling path: with replacement (uniform integers or the
        ///     <paramref name="p"/>-weighted CDF search), and without replacement (Floyd's algorithm / tail
        ///     shuffle, or the weighted rounds). The whole call holds the bit generator's lock.
        /// </remarks>
        [NDScoped]
        public NDArray choice(NDArray a, Shape? size = null, bool replace = true, NDArray p = null, int axis = 0, bool shuffle = true)
        {
            // `shape == None` means `shape == ()`, with the sample unpacked at the end (NumPy's is_scalar).
            bool isScalar = size == null;
            Shape shape = isScalar ? Shape.Scalar : size.Value;
            long count = isScalar ? 1 : shape.size; // np.prod(size); prod(()) == 1

            // ---- resolve the population (NumPy: a = np.asarray(a)) ----
            bool aIsScalarPop = a.ndim == 0;
            long popSize;
            bool popExceedsInt64 = false;
            if (aIsScalarPop)
            {
                popSize = ScalarPopulation(a, out popExceedsInt64);
                if (!popExceedsInt64 && popSize <= 0 && count != 0)
                    throw new ValueError("a must be a positive integer unless no samples are taken");
            }
            else
            {
                // NumPy reads the population with `a.shape[axis]`, i.e. Python tuple indexing: any axis in
                // [-ndim, ndim) is legal (negative counts from the end), anything else is an IndexError — not
                // an AxisError, because the axis is never normalized on this path.
                if (axis < -a.ndim || axis >= a.ndim)
                    throw new IndexError("tuple index out of range");
                if (axis < 0)
                    axis += a.ndim;
                popSize = a.shape[axis];
                if (popSize == 0 && count != 0)
                    throw new ValueError("a cannot be empty unless no samples are taken");
            }

            // ---- p validation (NumPy's order: len(p), atol, cast, ndim, size, NaN, sign, sum) ----
            double[] pw = null;
            if (p is not null)
                pw = ValidateChoiceProbabilities(p, popSize);

            NDArray idx;
            lock (_bitGenerator.@lock)
            {
                if (replace)
                {
                    if (pw is not null)
                    {
                        // cdf = cumsum(p); cdf /= cdf[-1]; idx = cdf.searchsorted(random(shape), 'right')
                        var cdf = np.cumsum(np.array(pw));
                        double total = Convert.ToDouble(cdf.GetAtIndex(cdf.size - 1));
                        cdf = cdf / total;
                        var uniform = random(shape);
                        idx = np.searchsorted(cdf, uniform, "right").astype(np.int64);
                    }
                    else if (popExceedsInt64)
                    {
                        // NumPy's integers(0, pop_size, dtype=int64) — the high cannot fit int64.
                        throw new ValueError("high is out of bounds for int64");
                    }
                    else
                    {
                        idx = integers(0, popSize, shape, np.int64);
                    }
                }
                else
                {
                    if (!popExceedsInt64 && count > popSize)
                        throw new ValueError("Cannot take a larger sample than population when replace is False");
                    if (count < 0)
                        throw new ValueError("negative dimensions are not allowed");

                    if (pw is not null)
                        idx = ChoiceNoReplaceWeighted(pw, count, shape, isScalar);
                    else if (popExceedsInt64)
                        // NumPy assigns pop_size to a C int64 here: OverflowError.
                        throw new OverflowException("Python int too large to convert to C long");
                    else
                        idx = ChoiceNoReplaceUniform(popSize, count, shuffle, shape, isScalar);
                }
            }

            // ---- map indices back onto the population ----
            if (aIsScalarPop)
                return idx; // integer population -> the drawn indices themselves (0-d for size=None/())

            // NumPy returns `a.take(idx, axis)`: a 0-d idx (size=None, or size=() with 1-D a) takes a single
            // element / sub-array, so the result is 0-d for 1-D a and a.shape without `axis` otherwise.
            return np.take(a, idx, axis: axis);
        }

        /// <summary>choice(int population, ...) convenience: draws from <c>arange(a)</c>.</summary>
        /// <param name="a">The population size.</param>
        /// <param name="size">Output shape; null is NumPy's <c>size=None</c> (a 0-d int64 result).</param>
        /// <param name="replace">Whether an index may be drawn more than once.</param>
        /// <param name="p">Optional 1-D probabilities of length <paramref name="a"/>.</param>
        /// <param name="shuffle">Without replacement and without <paramref name="p"/>: whether the selection is shuffled.</param>
        /// <returns>The int64 indices.</returns>
        /// <exception cref="ValueError">See <see cref="choice(NDArray, Shape?, bool, NDArray, int, bool)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d.</exception>
        [NDScoped]
        public NDArray choice(long a, Shape? size = null, bool replace = true, NDArray p = null, bool shuffle = true)
            => choice(NDArray.Scalar(a), size, replace, p, 0, shuffle);

        /// <summary>
        ///     NumPy's <c>pop_size = operator.index(a.item())</c> for a 0-d population.
        /// </summary>
        /// <param name="a">The 0-d population array.</param>
        /// <param name="exceedsInt64">Set when a uint64 population lies above <see cref="long.MaxValue"/> (NumPy keeps the
        /// Python int; the caller reproduces the error NumPy then raises on each sampling path).</param>
        /// <returns>The population size (bool counts as the integer 0/1, as in Python).</returns>
        /// <exception cref="ValueError">The scalar is not an integer (float, complex, decimal): <c>operator.index</c>
        /// raises <c>TypeError</c>, which NumPy re-raises as this ValueError naming the argument's type.</exception>
        private static long ScalarPopulation(NDArray a, out bool exceedsInt64)
        {
            exceedsInt64 = false;
            switch (a.typecode)
            {
                case NPTypeCode.Boolean:
                    return a.GetAtIndex<bool>(0) ? 1 : 0;
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.Char:
                    return Convert.ToInt64(a.GetAtIndex(0));
                case NPTypeCode.UInt64:
                    ulong u = a.GetAtIndex<ulong>(0);
                    exceedsInt64 = u > long.MaxValue;
                    return exceedsInt64 ? long.MaxValue : (long)u;
                default:
                    // NumSharp only ever passes an ndarray here, so NumPy's message names numpy.ndarray.
                    throw new ValueError("a must be a sequence or an integer, not <class 'numpy.ndarray'>");
            }
        }

        /// <summary>
        ///     Validates <c>choice</c>'s probabilities exactly as NumPy does and returns them as a float64 copy.
        /// </summary>
        /// <param name="p">The user's probabilities.</param>
        /// <param name="popSize">The population size they must match.</param>
        /// <returns>The probabilities converted to float64 (a private copy the weighted paths may modify).</returns>
        /// <exception cref="TypeError"><paramref name="p"/> is 0-d (<c>len() of unsized object</c>).</exception>
        /// <exception cref="ValueError">Not 1-D, wrong size, NaN sum, a negative entry, or a sum farther than the tolerance from 1.</exception>
        /// <remarks>
        ///     The tolerance is <c>sqrt(finfo(float64).eps)</c>, widened to <c>sqrt(finfo(p.dtype).eps)</c> for a
        ///     floating <paramref name="p"/> (float32: ~3.45e-4, float16: 0.03125) — so probabilities normalized in
        ///     single or half precision are accepted. The sum is NumPy's Kahan (compensated) sum, not a plain one.
        /// </remarks>
        private static unsafe double[] ValidateChoiceProbabilities(NDArray p, long popSize)
        {
            // d = len(p): a 0-d array has no length.
            if (p.ndim == 0)
                throw new TypeError("len() of unsized object");

            double atol = Math.Sqrt(2.220446049250313e-16); // sqrt(finfo(float64).eps)
            switch (p.typecode)
            {
                // atol = max(atol, np.sqrt(np.finfo(p.dtype).eps)) — computed in p's own precision, as NumPy's
                // np.sqrt of the dtype's eps scalar is.
                case NPTypeCode.Single: atol = Math.Max(atol, (double)MathF.Sqrt(1.1920929E-07f)); break;
                case NPTypeCode.Half: atol = Math.Max(atol, (double)Half.Sqrt((Half)0.0009765625)); break;
            }

            if (p.ndim != 1)
                throw new ValueError("p must be 1-dimensional");
            if (p.size != popSize)
                throw new ValueError("a and p must have same size");

            using var pd = p.astype(np.float64);
            var pw = new double[pd.size];
            double* src = (double*)(pd.Storage.Address + pd.Shape.offset * sizeof(double));
            for (long i = 0; i < pw.LongLength; i++)
                pw[i] = src[i];

            double pSum = KahanSum(pw);
            if (double.IsNaN(pSum))
                throw new ValueError("Probabilities contain NaN");
            foreach (double v in pw)
                if (v < 0)
                    throw new ValueError("Probabilities are not non-negative");
            if (Math.Abs(pSum - 1.0) > atol)
                throw new ValueError("Probabilities do not sum to 1. See Notes section of docstring for more information.");
            return pw;
        }

        /// <summary>NumPy's <c>kahan_sum</c> (<c>_common.pyx</c>): a compensated left-to-right sum.</summary>
        /// <param name="values">The values to add.</param>
        /// <returns>The compensated sum (0 for an empty array).</returns>
        private static double KahanSum(double[] values)
        {
            if (values.Length == 0)
                return 0.0;
            double sum = values[0], c = 0.0;
            for (int i = 1; i < values.Length; i++)
            {
                double y = values[i] - c;
                double t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            return sum;
        }

        /// <summary>
        ///     <c>replace=False, p=None</c>: Floyd's algorithm (small samples) or a tail partial-shuffle (large
        ///     samples of a large population), then the optional full shuffle — NumPy's heuristic and draws.
        /// </summary>
        /// <param name="popSize">The population size.</param>
        /// <param name="size">The number of samples (validated &lt;= <paramref name="popSize"/>).</param>
        /// <param name="shuffle">Whether to shuffle the Floyd selection.</param>
        /// <param name="shape">The output shape.</param>
        /// <param name="isScalar">Whether the request was <c>size=None</c> (0-d result).</param>
        /// <returns>The int64 indices shaped to <paramref name="shape"/>.</returns>
        /// <remarks>Both branches draw with <c>random_bounded_uint64(0, j, 0, 0)</c> == <see cref="BoundedUInt64Scalar"/>. The caller holds the lock.</remarks>
        private NDArray ChoiceNoReplaceUniform(long popSize, long size, bool shuffle, Shape shape, bool isScalar)
        {
            long[] result;
            int cutoff = shuffle ? 50 : 20;

            if (popSize > 10000 && size > popSize / cutoff)
            {
                // Tail-shuffle 'size' elements out of arange(pop_size).
                var idxAll = new long[popSize];
                for (long k = 0; k < popSize; k++)
                    idxAll[k] = k;
                ShuffleIntBuffer(idxAll, popSize, Math.Max(popSize - size, 1));
                result = new long[size];
                Array.Copy(idxAll, popSize - size, result, 0, size);
            }
            else
            {
                // Floyd's algorithm.
                result = new long[size];
                ulong setSize = (ulong)(1.2 * size);
                ulong mask = GenMask(setSize);
                setSize = 1 + mask;
                var hashSet = new ulong[setSize];
                for (ulong t = 0; t < setSize; t++)
                    hashSet[t] = ulong.MaxValue;

                for (long j = popSize - size; j < popSize; j++)
                {
                    ulong val = BoundedUInt64Scalar((ulong)j);
                    ulong loc = val & mask;
                    while (hashSet[loc] != ulong.MaxValue && hashSet[loc] != val)
                        loc = (loc + 1) & mask;
                    if (hashSet[loc] == ulong.MaxValue)
                    {
                        hashSet[loc] = val;
                        result[j - popSize + size] = (long)val;
                    }
                    else
                    {
                        loc = (ulong)j & mask;
                        while (hashSet[loc] != ulong.MaxValue)
                            loc = (loc + 1) & mask;
                        hashSet[loc] = (ulong)j;
                        result[j - popSize + size] = j;
                    }
                }

                if (shuffle)
                    ShuffleIntBuffer(result, size, 1);
            }

            var arr = np.array(result);
            if (isScalar)
                return arr.reshape(Shape.Scalar); // 0-d
            return arr.reshape(shape);
        }

        /// <summary>
        ///     <c>replace=False, p!=None</c>: weighted sampling without replacement (NumPy's
        ///     <c>_generator.pyx</c> choice, the <c>p is not None</c> branch).
        /// </summary>
        /// <param name="pw">The validated float64 probabilities (a private copy — found entries are zeroed in place).</param>
        /// <param name="size">The number of distinct samples.</param>
        /// <param name="shape">The output shape.</param>
        /// <param name="isScalar">Whether the request was <c>size=None</c> (0-d result).</param>
        /// <returns>The int64 indices shaped to <paramref name="shape"/>.</returns>
        /// <exception cref="ValueError">Fewer non-zero probabilities than <paramref name="size"/>.</exception>
        /// <remarks>
        ///     Each round draws <c>size - n_uniq</c> uniforms, maps them through the current (found-zeroed)
        ///     normalized CDF via <c>searchsorted(side='right')</c>, and keeps the batch's first-occurrence-unique
        ///     hits, until <paramref name="size"/> distinct indices are found. Composed from the byte-exact
        ///     <c>random</c> draw + sequential cumsum + bisect_right, so the stream matches
        ///     <c>default_rng(seed).choice(..., replace=False, p=...)</c> bit-for-bit. The caller holds the lock.
        /// </remarks>
        private unsafe NDArray ChoiceNoReplaceWeighted(double[] pw, long size, Shape shape, bool isScalar)
        {
            long d = pw.LongLength;
            long nonzero = 0;
            for (long i = 0; i < d; i++)
                if (pw[i] > 0) nonzero++;
            if (nonzero < size)
                throw new ValueError("Fewer non-zero entries in p than size");

            var found = new long[size];
            var seen = new HashSet<long>();
            var cdf = new double[d];
            long nUniq = 0;

            while (nUniq < size)
            {
                long need = size - nUniq;
                // x = self.random((need,)) — the byte-exact float64 draw.
                using NDArray x = random(new Shape(need));
                double* xp = (double*)x.Address;

                // Zero the probabilities of already-found indices, then rebuild the normalised CDF.
                for (long k = 0; k < nUniq; k++) pw[found[k]] = 0.0;
                double acc = 0.0;
                for (long i = 0; i < d; i++) { acc += pw[i]; cdf[i] = acc; }
                double last = cdf[d - 1];
                for (long i = 0; i < d; i++) cdf[i] /= last;

                // searchsorted(cdf, x, side='right'); keep first-occurrence-unique hits in order.
                for (long t = 0; t < need && nUniq < size; t++)
                {
                    long ins = BisectRight(cdf, xp[t]);
                    if (seen.Add(ins))
                        found[nUniq++] = ins;
                }
            }

            var arr = np.array(found);
            return isScalar ? arr.reshape(Shape.Scalar) : arr.reshape(shape);
        }

        /// <summary>bisect_right on a sorted double[] (numpy searchsorted side='right'): first index i with a[i] &gt; v.</summary>
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

        /// <summary>NumPy's <c>_shuffle_int</c>: Fisher-Yates over an int64 buffer using <c>random_bounded_uint64(0, i, 0, 0)</c>.</summary>
        /// <param name="data">The buffer shuffled in place.</param>
        /// <param name="n">The number of elements considered.</param>
        /// <param name="first">The smallest index swapped (1 shuffles everything; larger values shuffle only the tail).</param>
        private void ShuffleIntBuffer(long[] data, long n, long first)
        {
            for (long i = n - 1; i >= first; i--)
            {
                ulong j = BoundedUInt64Scalar((ulong)i);
                long tmp = data[j];
                data[j] = data[i];
                data[i] = tmp;
            }
        }

        /// <summary>numpy random_bounded_uint64(bitgen, off=0, rng, mask=0, use_masked=0) — the scalar Lemire draw.</summary>
        /// <param name="rng">The closed-interval range.</param>
        /// <returns>A value in <c>[0, rng]</c>.</returns>
        private ulong BoundedUInt64Scalar(ulong rng)
        {
            if (rng == 0)
                return 0;
            if (rng <= 0xFFFFFFFFUL)
            {
                if (rng == 0xFFFFFFFFUL)
                    return _bitGenerator.NextUInt32();
                return LemireUint32((uint)rng);
            }
            if (rng == 0xFFFFFFFFFFFFFFFFUL)
                return _bitGenerator.NextUInt64();
            return LemireUint64(rng);
        }

        /// <summary>NumPy's <c>_gen_mask</c>: the smallest all-ones bit mask covering <paramref name="max"/>.</summary>
        /// <param name="max">The value to cover.</param>
        /// <returns>The mask.</returns>
        private static ulong GenMask(ulong max)
        {
            ulong mask = max;
            mask |= mask >> 1;
            mask |= mask >> 2;
            mask |= mask >> 4;
            mask |= mask >> 8;
            mask |= mask >> 16;
            mask |= mask >> 32;
            return mask;
        }
    }
}
