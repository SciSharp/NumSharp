using System;

namespace NumSharp
{
    public sealed partial class Generator
    {
        /// <summary>
        ///     Generates a random sample from a given array (or <c>arange(a)</c> when <paramref name="a"/>
        ///     is an integer population size).
        /// </summary>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html
        ///     <br/>Byte-identical to NumPy for the common paths: with-replacement (uniform or
        ///     <paramref name="p"/>-weighted) and without-replacement uniform (Floyd's algorithm +
        ///     optional shuffle). Without-replacement WITH weights is not yet ported.
        /// </remarks>
        public NDArray choice(NDArray a, Shape? size = null, bool replace = true, NDArray p = null, int axis = 0, bool shuffle = true)
        {
            // ---- resolve population size ----
            long popSize;
            bool aIsScalarPop = a.ndim == 0;
            if (aIsScalarPop)
                popSize = Convert.ToInt64(a.GetAtIndex(0));
            else
            {
                int ax = axis < 0 ? axis + a.ndim : axis;
                if (ax < 0 || ax >= a.ndim)
                    throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {a.ndim}");
                axis = ax;
                popSize = a.shape[axis];
            }

            bool isScalar = size == null;
            Shape shape = isScalar ? Shape.Scalar : size.Value;
            long count = isScalar ? 1 : shape.size;

            if (aIsScalarPop && popSize <= 0 && count != 0)
                throw new ValueError("a must be a positive integer unless no samples are taken");
            if (!aIsScalarPop && popSize == 0 && count != 0)
                throw new ValueError("a cannot be empty unless no samples are taken");

            // ---- p validation (numpy: shape, NaN, non-negative, sum-to-1 within atol) ----
            if (p is not null)
            {
                if (p.ndim != 1)
                    throw new ValueError("p must be 1-dimensional");
                if (p.size != popSize)
                    throw new ValueError("a and p must have same size");

                var pd = p.astype(np.float64);
                double atol = Math.Sqrt(2.220446049250313e-16); // sqrt(finfo(float64).eps)
                double pSum = 0.0;
                bool anyNeg = false;
                for (long t = 0; t < pd.size; t++)
                {
                    double pv = Convert.ToDouble(pd.GetAtIndex(t));
                    pSum += pv;
                    if (pv < 0) anyNeg = true;
                }
                if (double.IsNaN(pSum))
                    throw new ValueError("Probabilities contain NaN");
                if (anyNeg)
                    throw new ValueError("Probabilities are not non-negative");
                if (Math.Abs(pSum - 1.0) > atol)
                    throw new ValueError("Probabilities do not sum to 1. See Notes section of docstring for more information.");
            }

            NDArray idx;
            if (replace)
            {
                if (p is not null)
                {
                    // cdf = cumsum(p); cdf /= cdf[-1]; idx = cdf.searchsorted(random(shape), 'right')
                    var cdf = np.cumsum(p.astype(np.float64));
                    double total = Convert.ToDouble(cdf.GetAtIndex(cdf.size - 1));
                    cdf = cdf / total;
                    var uniform = isScalar ? random() : random(shape);
                    idx = np.searchsorted(cdf, uniform, "right").astype(np.int64);
                }
                else
                {
                    idx = isScalar
                        ? integers(0, popSize, Shape.Scalar, np.int64)
                        : integers(0, popSize, shape, np.int64);
                }
            }
            else
            {
                if (count > popSize)
                    throw new ValueError("Cannot take a larger sample than population when replace is False");
                if (count < 0)
                    throw new ValueError("negative dimensions are not allowed");

                if (p is not null)
                    throw new NotSupportedException("choice(replace=false) with p (weighted sampling without replacement) is not yet ported in NumSharp.");

                idx = ChoiceNoReplaceUniform(popSize, count, shuffle, shape, isScalar);
            }

            // ---- map indices back onto the population ----
            if (aIsScalarPop)
                return idx; // integer population -> return the drawn indices directly

            var taken = np.take(a, idx.ndim == 0 ? idx.reshape(1) : idx, axis: axis);
            if (isScalar)
                return taken; // NumPy unpacks to a scalar; NumSharp returns the 0-d/1-elem result
            return taken;
        }

        /// <summary>choice(int population, ...) convenience: draws from <c>arange(a)</c>.</summary>
        public NDArray choice(long a, Shape? size = null, bool replace = true, NDArray p = null, bool shuffle = true)
            => choice(NDArray.Scalar(a), size, replace, p, 0, shuffle);

        // replace=False, p=None : Floyd's algorithm (small) / tail partial-shuffle (large), then
        // optional full shuffle. Both use random_bounded_uint64(0, j, 0, 0) == BoundedUInt64Scalar.
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

        // numpy _shuffle_int : Fisher-Yates over int64[] using random_bounded_uint64(0, i, 0, 0).
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

        // numpy random_bounded_uint64(bitgen, off=0, rng, mask=0, use_masked=0) — the scalar Lemire draw.
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
