using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments (&gt;= 0) — NumPy's <c>long n</c>, 64-bit in NumSharp's LP64 model, so a count
        ///     past <c>2**31</c> is legal (NumPy's Windows build, with a 32-bit <c>long</c>, rejects it).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes, each in <c>[0, 1]</c>; all but the last must
        ///     sum to at most 1 (the last category takes the remainder, whatever <c>pvals[-1]</c> says). May be empty.</param>
        /// <param name="size">Output shape. Result will have shape (*size, k); null (NumPy's <c>None</c>) gives shape (k,).</param>
        /// <returns>Drawn samples with shape (*size, k), where each row sums to n (int64 — NumPy's C <c>long</c>
        ///     <c>np.zeros(shape, dtype=np.long)</c>, modelled LP64: <see cref="LegacyLong"/>).</returns>
        /// <exception cref="TypeError"><paramref name="pvals"/> is null (<c>pvals must be a 1-d sequence</c>).</exception>
        /// <exception cref="ValueError">
        ///     In NumPy's order: a probability outside <c>[0, 1]</c> or NaN (<c>pvals &lt; 0, pvals &gt; 1 or pvals contains NaNs</c>),
        ///     <c>sum(pvals[:-1]) &gt; 1.0</c> (beyond a <c>1e-12</c> tolerance, Kahan-summed), a negative size dimension, and
        ///     only then <c>n &lt; 0</c>.
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
        ///     <br/>
        ///     The multinomial distribution is a multivariate generalization of the binomial
        ///     distribution. Each sample represents n experiments, where each experiment
        ///     results in one of k possible outcomes.
        ///     <br/>
        ///     NumPy's <c>legacy_random_multinomial</c>: one conditional binomial per category through the MODERN
        ///     <c>random_binomial</c> (log1p inversion / BTPE), sharing this RandomState's binomial setup cache with
        ///     <see cref="binomial(long, double, Shape)"/> — byte-identical to <c>np.random.RandomState(seed).multinomial</c>.
        ///     Holds the bit generator's lock for the draws.
        /// </remarks>
        public unsafe NDArray multinomial(long n, double[] pvals, Shape? size = null)
        {
            if (pvals is null)
                throw new TypeError("pvals must be a 1-d sequence");

            fixed (double* pix = pvals)
                return MultinomialCore(n, pix, pvals.Length, size, castMessage: false);
        }

        /// <summary>
        ///     Draw samples from a multinomial distribution, the probabilities given as a 1-D NDArray of any numeric dtype and
        ///     layout (cast to float64, as NumPy's <c>PyArray_FROMANY(pvals, NPY_DOUBLE, 0, 1)</c> does).
        /// </summary>
        /// <param name="n">Number of experiments (&gt;= 0; NumPy's <c>long n</c>, 64-bit here).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes (1-D).</param>
        /// <param name="size">Output shape. Result will have shape (*size, k); null gives shape (k,).</param>
        /// <returns>Drawn samples with shape (*size, k), where each row sums to n (int64, NumPy's C <c>long</c> modelled LP64).</returns>
        /// <exception cref="TypeError"><paramref name="pvals"/> is null or 0-d (<c>pvals must be a 1-d sequence</c>), or
        ///     complex (<c>Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule
        ///     'safe'</c>; <c>scalar</c> for a 0-d array, which fails the cast before the 1-d test).</exception>
        /// <exception cref="ValueError"><paramref name="pvals"/> has more than one dimension (<c>object too deep for desired
        ///     array</c>, reported before the cast), or a validation of <see cref="multinomial(long, double[], Shape?)"/> fails.</exception>
        /// <remarks>
        ///     When <paramref name="pvals"/> is a float16/float32 array whose own-precision sum is below <c>1.0001</c> but whose
        ///     float64 cast fails the sum check, the error carries NumPy's longer message explaining the cast.
        /// </remarks>
        public unsafe NDArray multinomial(long n, NDArray pvals, Shape? size = null)
        {
            // np.array(None, float64) is a 0-d NaN: the 1-d test below would reject it.
            if (pvals is null)
                throw new TypeError("pvals must be a 1-d sequence");
            // PyArray_FROMANY(pvals, NPY_DOUBLE, 0, 1) tests an ARRAY's depth first ("object too deep" — the "setting an
            // array element with a sequence" text belongs to nested Python lists, which C# spells as NDArray too late to
            // tell apart), then the safe cast (complex fails), and only then does multinomial test for 0-d.
            if (pvals.ndim > 1)
                throw new ValueError("object too deep for desired array");
            RandomConstraints.CheckSafeCast(pvals, NPTypeCode.Double);
            if (pvals.ndim == 0)
                throw new TypeError("pvals must be a 1-d sequence");

            long d = pvals.size;
            // NumPy's alternate message applies when `pvals.sum() < 1.0001` for a non-float64 FLOAT array. Under NEP 50 the
            // Python literal 1.0001 is weak, so the comparison happens IN the array's dtype: the threshold is 1.0001
            // rounded to float16/float32 (float16's is exactly 1.0, so a float16 sum of 1.0 does NOT qualify).
            bool castMessage = false;
            if (pvals.typecode == NPTypeCode.Half || pvals.typecode == NPTypeCode.Single)
                using (var s = np.sum(pvals))
                    // GetAtIndex boxes the element in its own dtype (GetDouble would REINTERPRET a float32 slot as 8 bytes).
                    castMessage = s.GetAtIndex(0) switch
                    {
                        System.Half h => h < (System.Half)1.0001,
                        float f => f < 1.0001f,
                        _ => false,
                    };

            // Flat float64 copy (any layout/dtype) — a raw pooled buffer private to this call, freed on every path.
            var block = new UnmanagedMemoryBlock<double>(d);
            var slice = new ArraySlice<double>(block);
            try
            {
                NDIter.Copy(new UnmanagedStorage(slice, new Shape(d)), pvals.Storage);
                return MultinomialCore(n, (double*)slice.Address, d, size, castMessage);
            }
            finally
            {
                slice.DangerousFree();
            }
        }

        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments (&gt;= 0; NumPy's <c>long n</c>).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes. Must sum to ~1.</param>
        /// <param name="size">Output shape as int array.</param>
        /// <returns>Drawn samples with shape (*size, k), where each row sums to n (int64).</returns>
        /// <exception cref="TypeError"><paramref name="pvals"/> is null.</exception>
        /// <exception cref="ValueError">A validation of <see cref="multinomial(long, double[], Shape?)"/> fails.</exception>
        public NDArray multinomial(long n, double[] pvals, int[] size)
            => multinomial(n, pvals, new Shape(size));

        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments (&gt;= 0; NumPy's <c>long n</c>).</param>
        /// <param name="pvals">Probabilities of each of the k different outcomes. Must sum to ~1.</param>
        /// <param name="size">Number of samples to draw — NumPy's integer <c>size</c>, an npy_intp (int64) dimension.</param>
        /// <returns>Drawn samples with shape (size, k), where each row sums to n (int64).</returns>
        /// <exception cref="TypeError"><paramref name="pvals"/> is null.</exception>
        /// <exception cref="ValueError">A validation of <see cref="multinomial(long, double[], Shape?)"/> fails.</exception>
        public NDArray multinomial(long n, double[] pvals, long size)
            => multinomial(n, pvals, new Shape(size));

        /// <summary>
        ///     The shared body of the <c>multinomial</c> overloads — NumPy's legacy <c>multinomial</c> after the pvals
        ///     conversion: validate, allocate the zeroed output, validate <c>n</c>, then one row per sample.
        /// </summary>
        /// <param name="n">Number of experiments.</param>
        /// <param name="pix">The d probabilities as contiguous float64.</param>
        /// <param name="d">The number of categories.</param>
        /// <param name="size">The leading output shape, or null for a single row.</param>
        /// <param name="castMessage">Use NumPy's longer float64-cast message for a failed sum check (a float16/float32
        ///     NDArray input whose own-dtype sum is below 1.0001).</param>
        /// <returns>The int64 counts (NumPy's C <c>long</c>, modelled LP64), shape (*size, d).</returns>
        /// <exception cref="ValueError">Any of NumPy's validations fails (see <see cref="multinomial(long, double[], Shape?)"/>).</exception>
        private unsafe NDArray MultinomialCore(long n, double* pix, long d, Shape? size, bool castMessage)
        {
            // check_array_constraint(parr, 'pvals', CONS_BOUNDED_0_1): the NaN-rejecting negated comparisons.
            for (long i = 0; i < d; i++)
                if (!(pix[i] >= 0) || !(pix[i] <= 1))
                    throw new ValueError("pvals < 0, pvals > 1 or pvals contains NaNs");

            // Only check if pvals is non-empty due no checks in kahan_sum
            if (d != 0 && Distributions.KahanSum(pix, d - 1) > (1.0 + 1e-12))
            {
                // NumPy words it differently when a low-precision float array sums below 1.0001 in its own dtype.
                if (castMessage)
                    throw new ValueError("sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals array is cast to 64-bit "
                                         + "floating point prior to checking the sum. Precision changes when casting may "
                                         + "cause problems even if the sum of the original pvals is valid.");
                throw new ValueError("sum(pvals[:-1]) > 1.0");
            }

            // shape = (d,) for size=None, else tuple(size) + (d,)  — np.zeros, so untouched categories stay 0.
            long[] dims;
            // A default Shape is NumSharp's spelling of NumPy's size=None, like a null one.
            if (size is null || size.Value.IsEmpty)
            {
                dims = new[] { d };
            }
            else
            {
                var sizeVal = size.Value;
                dims = new long[sizeVal.NDim + 1];
                for (int k = 0; k < sizeVal.NDim; k++)
                    dims[k] = sizeVal.dimensions[k];
                dims[sizeVal.NDim] = d;
            }
            var multin = new NDArray(LegacyLong.typecode, new Shape(dims), true);

            // NumPy validates n only after the output exists (so a bad size is reported first); the output is released
            // when n is refused, so that error path leaves no pooled buffer behind.
            try
            {
                RandomConstraints.Check(n, "n", ConstraintType.CONS_NON_NEGATIVE);
            }
            catch
            {
                multin.Dispose();
                throw;
            }

            var mnix = (long*)multin.Address;
            long sz = multin.size;
            // gh-20483: Avoids divide by 0
            long niter = d != 0 ? sz / d : 0;
            // Read-ahead draws (bulk-filled by the bit generator). Every row has the same parameters, so the rows are
            // all-or-nothing: a row's binomials skip the stream only for n == 0 or p == 0, and when n > 0 its first category
            // j < d-1 with pvals[j] != 0 runs with the full n and p = pvals[j] / 1.0 (remaining_p only ever lost zeros
            // before it), so it draws. Either every row draws at least once — the rows still owed bound the draws still to
            // come — or no row ever touches the stream and the buffer never refills; a full read-ahead is exact either way.
            double* storage = stackalloc double[DrawBufferDouble.Capacity];
            var src = new DrawBufferDouble(randomizer, storage, DrawBufferDouble.Capacity);
            lock (randomizer.@lock)
            {
                // A row cycles through one binomial key per category, on which NumPy's single-entry cache never hits; a
                // memo of those setups for the fill skips the recomputation without changing a bit (see BinomialState).
                // It is installed and removed under the lock, so no other sampler ever sees it.
                if (niter >= MultinomialMemoMinRows)
                    _binomial.Memo = new BinomialSetup[1 << BinomialState.MemoBits];
                try
                {
                    long offset = 0;
                    for (long i = 0; i < niter; i++)
                    {
                        src.Owed = niter - i;
                        Distributions.RandomMultinomial(ref src, n, mnix + offset, pix, d, _binomial);
                        offset += d;
                    }
                }
                finally
                {
                    _binomial.Memo = null;
                }
            }

            return multin;
        }

        /// <summary>
        ///     The smallest multinomial fill (in rows) that installs the binomial-setup memo: below it, allocating the memo
        ///     costs more than the recomputed setups it saves.
        /// </summary>
        private const long MultinomialMemoMinRows = 8;
    }
}
