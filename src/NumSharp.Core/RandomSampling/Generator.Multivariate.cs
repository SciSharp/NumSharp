using System;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public sealed partial class Generator
    {
        // =============================================================================================================
        // multinomial
        // =============================================================================================================

        /// <summary>
        ///     Draw samples from a multinomial distribution.
        /// </summary>
        /// <param name="n">Number of experiments, non-negative.</param>
        /// <param name="pvals">Probabilities of each of the <c>d</c> outcomes, each in <c>[0, 1]</c>; <c>sum(pvals[:-1])</c>
        ///     must not exceed 1 (the last entry takes whatever probability is left).</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one vector of shape <c>(d,)</c>.</param>
        /// <returns>The int64 counts, shape <c>(*size, d)</c>.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="pvals"/> is empty; an element is outside
        ///     <c>[0, 1]</c> or NaN; <c>sum(pvals[:-1]) &gt; 1.0</c>; then (after the output exists) <paramref name="n"/> is
        ///     negative (<c>n &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html
        ///     <br/>NumPy's <c>random_multinomial</c> — one conditional binomial per category over this Generator's
        ///     binomial cache. Byte-identical to <c>default_rng(seed).multinomial</c>.
        /// </remarks>
        public NDArray multinomial(long n, double[] pvals, Shape size = default)
        {
            // NumPy converts pvals first; None becomes a 0-d array, which fails the dimension test below. Both wrappers are
            // private to this call (the core returns a fresh array and retains neither).
            using var on = NDArray.Scalar(n);
            using var parr = pvals is null ? NDArray.Scalar(double.NaN) : np.array(pvals);
            return MultinomialCore(on, parr, castMessage: false, size);
        }

        /// <summary>
        ///     Draw samples from a multinomial distribution, with array-valued probabilities (the last axis is the outcomes).
        /// </summary>
        /// <param name="n">Number of experiments, non-negative.</param>
        /// <param name="pvals">Probabilities, any numeric dtype and layout (cast to float64); rows along the last axis.</param>
        /// <param name="size">Leading output shape; must equal the broadcast of <paramref name="pvals"/>' leading shape.</param>
        /// <returns>The int64 counts, shape <c>(*broadcast, d)</c>.</returns>
        /// <exception cref="ValueError">See <see cref="multinomial(NDArray, NDArray, Shape)"/>.</exception>
        /// <exception cref="TypeError"><paramref name="pvals"/> is complex (not safely castable to float64).</exception>
        public NDArray multinomial(long n, NDArray pvals, Shape size = default)
        {
            using var on = NDArray.Scalar(n); // private to this call: the result is a fresh array
            return multinomial(on, pvals, size);
        }

        /// <summary>
        ///     Draw samples from a multinomial distribution with broadcasting trial counts and probability rows.
        /// </summary>
        /// <param name="n">Number of experiments: a scalar or an array broadcast against <paramref name="pvals"/>' leading
        ///     shape. Its dtype must reach int64 under the <c>'safe'</c> rule (bool, the signed integers, uint8-uint32).</param>
        /// <param name="pvals">Probabilities; the last axis holds the <c>d</c> outcomes, the leading axes broadcast. Any real
        ///     dtype and layout (cast to float64); null reads as NumPy's <c>None</c> (a 0-d NaN, rejected as dimensionless).</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) uses the broadcast shape of
        ///     <paramref name="n"/> and <paramref name="pvals"/>' leading axes.</param>
        /// <returns>The int64 counts, shape <c>(*broadcast, d)</c>.</returns>
        /// <exception cref="TypeError">In NumPy's order: <paramref name="n"/> is null, or its dtype is not safely castable to
        ///     int64 (a float or uint64 count: <c>Cannot cast array data from dtype('float64') to dtype('int64') according to
        ///     the rule 'safe'</c>, <c>scalar</c> for a 0-d array); <paramref name="pvals"/> is complex.</exception>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="pvals"/> is 0-d or has an empty last axis; an
        ///     element is outside <c>[0, 1]</c> or NaN; a row's <c>sum(pvals[...,:-1]) &gt; 1.0</c> (NumPy's alternate text for
        ///     a float16/float32 array whose own sum is below 1.0001); an element of <paramref name="n"/> is negative; the
        ///     inputs and <paramref name="size"/> do not broadcast (NumPy's <c>shape mismatch</c> text) or broadcast to a shape
        ///     other than <paramref name="size"/> (<c>Output size ... is not compatible ...</c>).</exception>
        /// <remarks>
        ///     With a 0-d <paramref name="n"/> and 1-D <paramref name="pvals"/> this is the plain path (one row per size
        ///     position); otherwise NumPy's vector path, which pairs every broadcast position with its own count and
        ///     probability row, in C order.
        /// </remarks>
        public NDArray multinomial(NDArray n, NDArray pvals, Shape size = default)
        {
            // NumPy converts n first (PyArray_FROM_OTF(n, NPY_INT64)): None fails inside int(), a float/uint64 array fails
            // the safe cast — both before pvals is looked at.
            if (n is null)
                throw new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'NoneType'");
            RandomConstraints.CheckSafeCast(n, NPTypeCode.Int64);

            // The C-contiguous int64 / float64 working copies are made only when the caller's arrays are not already
            // that; a copy is private to this call and released on every path (the caller's arrays are never touched).
            bool ownOn = !(n.typecode == NPTypeCode.Int64 && n.Shape.IsContiguous), ownParr = false;
            NDArray on = ownOn ? n.astype(np.int64) : n, parr = null;
            try
            {
                // pvals=None converts to a 0-d NaN array, which the dimension check rejects.
                if (pvals is null)
                {
                    ownParr = true;
                    parr = NDArray.Scalar(double.NaN);
                    return MultinomialCore(on, parr, castMessage: false, size);
                }
                RandomConstraints.CheckSafeCast(pvals, NPTypeCode.Double);

                // NumPy's alternate sum message needs `pvals.sum() < 1.0001` IN the array's own float dtype (NEP 50).
                bool castMessage = false;
                if (pvals.typecode == NPTypeCode.Half || pvals.typecode == NPTypeCode.Single)
                    using (var s = np.sum(pvals))
                        castMessage = s.GetAtIndex(0) switch
                        {
                            Half h => h < (Half)1.0001,
                            float fs => fs < 1.0001f,
                            _ => false,
                        };

                ownParr = !(pvals.typecode == NPTypeCode.Double && pvals.Shape.IsContiguous);
                parr = ownParr ? pvals.astype(np.float64) : pvals;
                return MultinomialCore(on, parr, castMessage, size);
            }
            finally
            {
                if (ownOn)
                    on.Dispose();
                if (ownParr)
                    parr?.Dispose();
            }
        }

        /// <summary>
        ///     The body of every <c>multinomial</c> overload: NumPy's <c>Generator.multinomial</c> over an int64
        ///     <paramref name="on"/> and a C-contiguous float64 <paramref name="parr"/>.
        /// </summary>
        /// <param name="on">The trial counts (0-d for a scalar), int64.</param>
        /// <param name="parr">The probabilities, float64, C-contiguous.</param>
        /// <param name="castMessage">Whether a failing sum reports NumPy's float-cast text.</param>
        /// <param name="size">The leading output shape, or <c>default</c> for NumPy's <c>None</c>.</param>
        /// <returns>The int64 counts.</returns>
        /// <exception cref="ValueError">A NumPy validation fails (see the public overloads).</exception>
        private unsafe NDArray MultinomialCore(NDArray on, NDArray parr, bool castMessage, Shape size)
        {
            int ndim = parr.ndim;
            long d = ndim >= 1 ? parr.shape[ndim - 1] : 0;
            if (d == 0)
                throw new ValueError("pvals must have at least 1 dimension and the last dimension of pvals must be greater than 0.");

            var pix = (double*)((byte*)parr.Storage.Address + parr.Shape.offset * sizeof(double));
            long sz = parr.size;
            // check_array_constraint(parr, 'pvals', CONS_BOUNDED_0_1): every element in [0, 1], NaN rejected.
            for (long i = 0; i < sz; i++)
                if (!(pix[i] >= 0) || !(pix[i] <= 1))
                    throw new ValueError("pvals < 0, pvals > 1 or pvals contains NaNs");
            for (long offset = 0; offset < sz; offset += d)
            {
                if (Distributions.KahanSum(pix + offset, d - 1) > (1.0 + 1e-12))
                {
                    string sliceRepr = ndim == 1 ? "[:-1]" : "[...,:-1]";
                    if (castMessage)
                        throw new ValueError($"sum(pvals{sliceRepr}.astype(np.float64)) > 1.0. The pvals array is cast to 64-bit floating"
                                             + " point prior to checking the sum. Precision changes when casting may cause problems even"
                                             + " if the sum of the original pvals is valid.");
                    throw new ValueError($"sum(pvals{sliceRepr}) > 1.0");
                }
            }

            if (on.ndim != 0 || ndim > 1)
                return MultinomialVector(on, parr, pix, d, size);

            // The plain path: shape (d,) or (*size, d), np.zeros so untouched categories stay 0.
            long[] dims = OutputDimsWithTrailing(size, d);
            var multin = new NDArray(NPTypeCode.Int64, new Shape(dims), true);
            long ni = on.GetInt64();
            // NumPy validates n only after the output exists (so a bad size is reported first); the output never leaves
            // this call when the check fails, so it is released before the error propagates.
            try
            {
                RandomConstraints.Check((double)ni, "n", ConstraintType.CONS_NON_NEGATIVE);
            }
            catch
            {
                multin.Dispose();
                throw;
            }

            var mnix = (long*)multin.Address;
            long rows = multin.size / d;
            MultinomialRows(ni, mnix, pix, d, rows);
            return multin;
        }

        /// <summary>
        ///     Draws <paramref name="rows"/> multinomial rows with one fixed count and probability row — through a read-ahead
        ///     and, for fills long enough, the binomial-setup memo (bit-neutral, see <see cref="BinomialState"/>).
        /// </summary>
        /// <param name="ni">The trial count.</param>
        /// <param name="mnix">The zeroed output rows.</param>
        /// <param name="pix">The probability row.</param>
        /// <param name="d">The categories per row.</param>
        /// <param name="rows">The number of rows.</param>
        private unsafe void MultinomialRows(long ni, long* mnix, double* pix, long d, long rows)
        {
            // Every row has the same parameters, so the rows are all-or-nothing: either every row draws at least once and
            // the rows still owed bound the draws still to come, or no row ever touches the stream (the buffer never
            // refills). A full read-ahead is exact either way.
            double* storage = stackalloc double[DrawBufferDouble.Capacity];
            var src = new DrawBufferDouble(_bitGenerator, storage, DrawBufferDouble.Capacity);
            lock (_bitGenerator.@lock)
            {
                if (rows >= 8)
                    _binomial.Memo = new BinomialSetup[1 << BinomialState.MemoBits];
                try
                {
                    long offset = 0;
                    for (long i = 0; i < rows; i++)
                    {
                        src.Owed = rows - i;
                        Distributions.RandomMultinomial(ref src, ni, mnix + offset, pix, d, _binomial);
                        offset += d;
                    }
                }
                finally
                {
                    _binomial.Memo = null;
                }
            }
        }

        /// <summary>
        ///     NumPy's vector multinomial path: the counts and the probability rows broadcast (with <paramref name="size"/>
        ///     when given), and every broadcast position draws one row with its own count and probabilities, in C order.
        /// </summary>
        /// <param name="on">The int64 counts.</param>
        /// <param name="parr">The float64 C-contiguous probabilities.</param>
        /// <param name="pix">Their data.</param>
        /// <param name="d">The categories per row.</param>
        /// <param name="size">The requested leading shape, or <c>default</c>.</param>
        /// <returns>The int64 counts, shape <c>(*broadcast, d)</c>.</returns>
        /// <exception cref="ValueError">A count is negative, or the shapes do not broadcast to the requested size.</exception>
        private unsafe NDArray MultinomialVector(NDArray on, NDArray parr, double* pix, long d, Shape size)
        {
            // check_array_constraint(on, 'n', CONS_NON_NEGATIVE)
            using (var lt = on < 0)
                if (np.any(lt))
                    throw new ValueError("n < 0");

            // offsets = arange(0, parr.size, d).reshape(parr.shape[:-1]) — each broadcast position's row start.
            var rowShape = new long[parr.ndim - 1];
            for (int i = 0; i < rowShape.Length; i++)
                rowShape[i] = parr.shape[i];

            Shape itShape;
            try
            {
                itShape = IsNoSize(size)
                    ? np.broadcast_shapes(on.Shape, new Shape(rowShape))
                    : np.broadcast_shapes(on.Shape, new Shape(rowShape), new Shape(size.dimensions));
            }
            catch (IncorrectShapeException e)
            {
                // NumPy's MultiIter raises ValueError with this exact text.
                throw new ValueError(e.Message);
            }
            if (!IsNoSize(size) && !SameDims(itShape.dimensions, size.dimensions))
                throw new ValueError($"Output size {PyTuple(size.dimensions)} is not compatible with broadcast dimensions of inputs {PyTuple(itShape.dimensions)}.");

            long positions = itShape.size;
            var outDims = new long[itShape.NDim + 1];
            Array.Copy(itShape.dimensions, outDims, itShape.NDim);
            outDims[itShape.NDim] = d;

            // Each position's count and probability-row offset are read in C order straight through the broadcast views'
            // strides (NumPy's MultiIter walk) — nothing is materialized. Every intermediate here is private to this call
            // and released when it returns (or throws).
            using var offsets = np.arange(0L, parr.size, d);
            using var offsetRows = offsets.reshape(new Shape(rowShape));
            using var nView = np.broadcast_to(on, itShape);
            using var oView = np.broadcast_to(offsetRows, itShape);

            var multin = new NDArray(NPTypeCode.Int64, new Shape(outDims), true);
            if (positions == 0)
                return multin;
            var mnix = (long*)multin.Address;

            // A read-ahead needs every position to draw at least once: a positive count and a nonzero probability before
            // the row's last. Broadcasting drops no element, so every count and every row reaches some position, and
            // checking the operands themselves decides all positions at once.
            var onData = (long*)on.Storage.Address + on.Shape.offset;
            bool allDraw = true;
            for (long i = 0, cnt = on.size; i < cnt && allDraw; i++)
                allDraw = onData[i] > 0;
            for (long row = 0, rows = parr.size / d; row < rows && allDraw; row++)
            {
                bool rowDraws = false;
                for (long j = 0; j < d - 1 && !rowDraws; j++)
                    rowDraws = pix[row * d + j] != 0.0;
                allDraw = rowDraws;
            }

            int nd = itShape.NDim;
            long* dims = stackalloc long[nd];
            long* nStride = stackalloc long[nd];
            long* oStride = stackalloc long[nd];
            long* coord = stackalloc long[nd];
            for (int ax = 0; ax < nd; ax++)
            {
                dims[ax] = itShape.dimensions[ax];
                nStride[ax] = nView.Shape.strides[ax];
                oStride[ax] = oView.Shape.strides[ax];
                coord[ax] = 0;
            }
            var nBase = (long*)nView.Storage.Address;
            var oBase = (long*)oView.Storage.Address;
            long nAt = nView.Shape.offset, oAt = oView.Shape.offset;

            double* storage = stackalloc double[DrawBufferDouble.Capacity];
            var src = new DrawBufferDouble(_bitGenerator, storage, allDraw ? DrawBufferDouble.Capacity : 1);
            lock (_bitGenerator.@lock)
            {
                // Recurring (count, probability) keys thrash NumPy's single binomial cache; the memo keeps every recurring
                // key's setup for the fill (bit-neutral, see BinomialState) and is removed before the lock is released. A
                // key is memoized only from its second miss on, so positions that never repeat a key cost no allocation.
                if (positions >= 8)
                    _binomial.Memo = new BinomialSetup[1 << BinomialState.MemoBits];
                try
                {
                    long offset = 0;
                    for (long i = 0; i < positions; i++)
                    {
                        src.Owed = positions - i;
                        Distributions.RandomMultinomial(ref src, nBase[nAt], mnix + offset, pix + oBase[oAt], d, _binomial);
                        offset += d;
                        // C-order odometer over the broadcast shape: bump the last axis, carrying into the earlier ones.
                        for (int ax = nd - 1; ax >= 0; ax--)
                        {
                            if (++coord[ax] < dims[ax])
                            {
                                nAt += nStride[ax];
                                oAt += oStride[ax];
                                break;
                            }
                            coord[ax] = 0;
                            nAt -= nStride[ax] * (dims[ax] - 1);
                            oAt -= oStride[ax] * (dims[ax] - 1);
                        }
                    }
                }
                finally
                {
                    _binomial.Memo = null;
                }
            }
            return multin;
        }

        // =============================================================================================================
        // dirichlet
        // =============================================================================================================

        /// <summary>
        ///     Draw samples from the Dirichlet distribution.
        /// </summary>
        /// <param name="alpha">Concentration parameters (length k, each non-negative; NaN accepted and samples NaN).</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one vector of shape <c>(k,)</c>.</param>
        /// <returns>The float64 draws, shape <c>(*size, k)</c>; each row sums to 1.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null (<c>object of type 'NoneType' has no len()</c>).</exception>
        /// <exception cref="ValueError">An element of <paramref name="alpha"/> is negative (<c>alpha &lt; 0</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html
        ///     <br/>NumPy's Generator algorithm: when every alpha is below 0.1, the stick-breaking construction over
        ///     <c>random_beta</c> (so all-tiny alphas never produce the 0/0 NaN of the gamma normalization); otherwise
        ///     standard gammas normalized by the reciprocal of their sum. Unlike RandomState, zeros are allowed.
        ///     Byte-identical to <c>default_rng(seed).dirichlet</c>.
        /// </remarks>
        public unsafe NDArray dirichlet(double[] alpha, Shape size = default)
        {
            if (alpha is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            fixed (double* a = alpha)
                return DirichletCore(a, alpha.Length, size);
        }

        /// <summary>
        ///     Draw samples from the Dirichlet distribution with the concentration given as a 1-D array.
        /// </summary>
        /// <param name="alpha">Concentration parameters, 1-D, any numeric dtype and layout (cast to float64).</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one vector.</param>
        /// <returns>The float64 draws, shape <c>(*size, k)</c>.</returns>
        /// <exception cref="TypeError"><paramref name="alpha"/> is null or 0-d (<c>len() of unsized object</c>), or complex
        ///     (<c>Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule 'safe'</c>).</exception>
        /// <exception cref="ValueError"><paramref name="alpha"/> has more than one dimension (<c>object too deep for desired
        ///     array</c>) or a negative element (<c>alpha &lt; 0</c>).</exception>
        /// <remarks>
        ///     NumPy's order: <c>len(alpha)</c> (fails for 0-d), then <c>PyArray_FROMANY(alpha, NPY_DOUBLE, 1, 1)</c>, whose
        ///     depth test precedes its safe-cast test — so a 2-D complex array reports the depth.
        /// </remarks>
        public unsafe NDArray dirichlet(NDArray alpha, Shape size = default)
        {
            if (alpha is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            if (alpha.ndim == 0)
                throw new TypeError("len() of unsized object");
            if (alpha.ndim > 1)
                throw new ValueError("object too deep for desired array");
            RandomConstraints.CheckSafeCast(alpha, NPTypeCode.Double);

            long k = alpha.size;
            var block = new UnmanagedMemoryBlock<double>(k);
            var slice = new ArraySlice<double>(block);
            try
            {
                NDIter.Copy(new UnmanagedStorage(slice, new Shape(k)), alpha.Storage);
                return DirichletCore((double*)slice.Address, k, size);
            }
            finally
            {
                slice.DangerousFree(); // the raw float64 copy is private to this call
            }
        }

        /// <summary>The body of the <c>dirichlet</c> overloads: NumPy's Generator loop over a contiguous float64 alpha.</summary>
        /// <param name="alpha">The k concentration parameters.</param>
        /// <param name="k">The number of categories (0 allowed: the output is empty and nothing is drawn).</param>
        /// <param name="size">The leading output shape, or <c>default</c>.</param>
        /// <returns>The float64 output, shape <c>(*size, k)</c>.</returns>
        /// <exception cref="ValueError">An element is negative.</exception>
        private unsafe NDArray DirichletCore(double* alpha, long k, Shape size)
        {
            // np.any(np.less(alpha_arr, 0)): NaN compares false and is accepted.
            for (long j = 0; j < k; j++)
                if (alpha[j] < 0)
                    throw new ValueError("alpha < 0");

            var diric = new NDArray(NPTypeCode.Double, new Shape(OutputDimsWithTrailing(size, k)), true);
            var val = (double*)diric.Address;
            long totsize = diric.size;
            if (totsize == 0)
                return diric;
            long rows = totsize / k;

            // alpha_arr.max() < 0.1 — a NaN maximum (any NaN element) compares false.
            bool smallAlpha = k > 0;
            for (long j = 0; j < k && smallAlpha; j++)
                if (!(alpha[j] < 0.1))
                    smallAlpha = false;

            // Per-row draws are all-or-nothing (every row has the same alphas), so rows still owed bound the draws.
            ulong* storage = stackalloc ulong[DrawBuffer64.Capacity];
            var src = new DrawBuffer64(_bitGenerator, storage, DrawBuffer64.Capacity);

            if (smallAlpha)
            {
                // Stick-breaking over random_beta. csum[j] = sum(alpha[j:]), accumulated right to left as NumPy does.
                var csum = new double[k];
                double c = 0.0;
                for (long j = k - 1; j >= 0; j--)
                {
                    c += alpha[j];
                    csum[j] = c;
                }
                // csum == 0 means every alpha is 0: nothing to draw, the zeros stand.
                if (c > 0)
                {
                    var betas = new BetaSetup[Math.Max(0, k - 1)];
                    for (long j = 0; j < k - 1; j++)
                        betas[j] = new BetaSetup(alpha[j], csum[j + 1]);
                    lock (_bitGenerator.@lock)
                        for (long i = 0, row = 0; i < totsize; i += k, row++)
                        {
                            src.Owed = rows - row;
                            double acc = 1.0;
                            for (long j = 0; j < k - 1; j++)
                            {
                                double v = Beta(ref src, in betas[j]);
                                val[i + j] = acc * v;
                                acc *= (1.0 - v);
                                // csum[j + 1] == 0: v must be 1, acc is now 0, and every later element stays 0.
                                if (csum[j + 1] == 0)
                                    break;
                            }
                            val[i + k - 1] = acc;
                        }
                }
            }
            else
            {
                // Standard gammas normalized by the reciprocal of their sum; one setup per alpha, evaluated once.
                var gammas = new GammaSetup[k];
                for (long j = 0; j < k; j++)
                    gammas[j] = new GammaSetup(alpha[j]);
                lock (_bitGenerator.@lock)
                    for (long i = 0, row = 0; i < totsize; i += k, row++)
                    {
                        src.Owed = rows - row;
                        double acc = 0.0;
                        for (long j = 0; j < k; j++)
                        {
                            val[i + j] = StandardGamma(ref src, in gammas[j]);
                            acc = acc + val[i + j];
                        }
                        double invacc = 1.0 / acc;
                        for (long j = 0; j < k; j++)
                            val[i + j] = val[i + j] * invacc;
                    }
            }
            return diric;
        }

        // =============================================================================================================
        // multivariate_hypergeometric
        // =============================================================================================================

        /// <summary>
        ///     Generate variates from a multivariate hypergeometric distribution.
        /// </summary>
        /// <param name="colors">The number of each type of item in the collection (non-negative integers).</param>
        /// <param name="nsample">The number of items selected, <c>0 &lt;= nsample &lt;= sum(colors)</c>.</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one variate of shape <c>(len(colors),)</c>.</param>
        /// <param name="method"><c>"marginals"</c> (default; successive univariate hypergeometric draws, <c>sum(colors) &lt; 10^9</c>)
        ///     or <c>"count"</c> (a partial Fisher-Yates shuffle over an array of <c>sum(colors)</c> entries).</param>
        /// <returns>The int64 variates, shape <c>(*size, len(colors))</c>.</returns>
        /// <exception cref="ValueError">In NumPy's order: an unknown <paramref name="method"/>; a negative
        ///     <paramref name="nsample"/>; a negative color; a sum of colors past int64; <c>"marginals"</c> with
        ///     <c>sum(colors) &gt;= 10^9</c>; <c>"count"</c> with a sum too large to index; <c>nsample &gt; sum(colors)</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>"count"</c> cannot allocate its <c>sum(colors)</c>-entry work array.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_hypergeometric.html
        ///     <br/>NumPy's <c>random_multivariate_hypergeometric_count</c> / <c>_marginals</c>. Byte-identical to
        ///     <c>default_rng(seed).multivariate_hypergeometric</c>.
        /// </remarks>
        public NDArray multivariate_hypergeometric(long[] colors, long nsample, Shape size = default, string method = "marginals")
        {
            if (method != "count" && method != "marginals")
                throw new ValueError("method must be \"count\" or \"marginals\".");
            if (nsample < 0)
                throw new ValueError("nsample must be nonnegative.");
            if (colors is null)
                throw new ValueError($"colors must be a one-dimensional sequence of nonnegative integers not exceeding {long.MaxValue}.");
            for (int i = 0; i < colors.Length; i++)
                if (colors[i] < 0)
                    throw new ValueError($"colors must be a one-dimensional sequence of nonnegative integers not exceeding {long.MaxValue}.");
            return MultivariateHypergeometricCore(colors, nsample, size, method);
        }

        /// <summary>
        ///     Generate variates from a multivariate hypergeometric distribution, the colors given as an array.
        /// </summary>
        /// <param name="colors">1-D array of non-negative integers (any integer dtype; an empty array of any dtype is allowed).</param>
        /// <param name="nsample">The number of items selected.</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one variate.</param>
        /// <param name="method"><c>"marginals"</c> (default) or <c>"count"</c>.</param>
        /// <returns>The int64 variates, shape <c>(*size, len(colors))</c>.</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_hypergeometric(long[], long, Shape, string)"/>; also a
        ///     non-1-D or non-integer <paramref name="colors"/>.</exception>
        /// <exception cref="OutOfMemoryException"><c>"count"</c> cannot allocate its work array.</exception>
        public NDArray multivariate_hypergeometric(NDArray colors, long nsample, Shape size = default, string method = "marginals")
        {
            if (method != "count" && method != "marginals")
                throw new ValueError("method must be \"count\" or \"marginals\".");
            if (nsample < 0)
                throw new ValueError("nsample must be nonnegative.");
            string invalid = $"colors must be a one-dimensional sequence of nonnegative integers not exceeding {long.MaxValue}.";
            if (colors is null || colors.ndim != 1)
                throw new ValueError(invalid);
            // An empty array of any dtype passes; a non-empty one must be an integer dtype.
            if (colors.size > 0 && !IsIntegerTypeCode(colors.typecode))
                throw new ValueError(invalid);
            long[] c;
            if (colors.size == 0)
            {
                c = Array.Empty<long>();
            }
            else
            {
                using var c64 = colors.astype(np.int64); // private working copy
                c = c64.ToArray<long>();
            }
            for (int i = 0; i < c.Length; i++)
                if (c[i] < 0 || (colors.typecode == NPTypeCode.UInt64 && (ulong)colors.GetAtIndex<ulong>(i) > long.MaxValue))
                    throw new ValueError(invalid);
            return MultivariateHypergeometricCore(c, nsample, size, method);
        }

        /// <summary>Whether <paramref name="tc"/> is one of NumPy's integer dtypes (bool is not).</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>True for the signed/unsigned integer dtypes (Char, NumSharp's uint16-like, counts too).</returns>
        private static bool IsIntegerTypeCode(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32
                or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char => true,
            _ => false,
        };

        /// <summary>The validated body of <c>multivariate_hypergeometric</c> (from the sum checks on).</summary>
        /// <param name="colors">The validated non-negative colors.</param>
        /// <param name="nsample">The validated non-negative sample size.</param>
        /// <param name="size">The leading output shape, or <c>default</c>.</param>
        /// <param name="method"><c>"count"</c> or <c>"marginals"</c>.</param>
        /// <returns>The int64 variates.</returns>
        /// <exception cref="ValueError">A sum check fails.</exception>
        /// <exception cref="OutOfMemoryException">The count method's work array cannot be allocated.</exception>
        private unsafe NDArray MultivariateHypergeometricCore(long[] colors, long nsample, Shape size, string method)
        {
            // _safe_sum_nonneg_int64: the total, or an error when it overflows int64.
            long total = 0;
            for (int i = 0; i < colors.Length; i++)
            {
                if (colors[i] > long.MaxValue - total)
                    throw new ValueError($"sum(colors) must not exceed the maximum value of a 64 bit signed integer ({long.MaxValue})");
                total += colors[i];
            }
            if (method == "marginals" && total >= 1000000000)
                throw new ValueError("When method is \"marginals\", sum(colors) must be less than 1000000000.");
            // The count method allocates total * sizeof(size_t); NumPy caps the index so that product cannot overflow.
            const long MaxIndex = long.MaxValue / sizeof(ulong);
            if (method == "count" && total > MaxIndex)
                throw new ValueError($"When method is 'count', sum(colors) must not exceed {MaxIndex}");
            if (nsample > total)
                throw new ValueError("nsample > sum(colors)");

            long numColors = colors.Length;
            var variates = new NDArray(NPTypeCode.Int64, new Shape(OutputDimsWithTrailing(size, numColors)), true);
            if (numColors == 0)
                return variates;

            long numVariates = variates.size / numColors;
            var v = (long*)variates.Address;
            fixed (long* c = colors)
            {
                if (method == "count")
                    MultivariateHypergeometricCount(total, numColors, c, nsample, numVariates, v);
                else
                    MultivariateHypergeometricMarginals(total, numColors, c, nsample, numVariates, v);
            }
            return variates;
        }

        /// <summary>
        ///     NumPy's <c>random_multivariate_hypergeometric_count</c>: a partial Fisher-Yates shuffle (over
        ///     <c>random_interval</c>) of an array holding each color index once per item, counted per variate.
        /// </summary>
        /// <param name="total">The number of items.</param>
        /// <param name="numColors">The number of colors.</param>
        /// <param name="colors">The items per color.</param>
        /// <param name="nsample">The items selected per variate.</param>
        /// <param name="numVariates">The number of variates.</param>
        /// <param name="variates">The zeroed output, <paramref name="numVariates"/> rows of <paramref name="numColors"/>.</param>
        /// <exception cref="OutOfMemoryException">The <paramref name="total"/>-entry work array cannot be allocated.</exception>
        private unsafe void MultivariateHypergeometricCount(long total, long numColors, long* colors, long nsample,
                                                            long numVariates, long* variates)
        {
            if ((total == 0) || (nsample == 0) || (numVariates == 0))
                return;

            // The work array goes through NumSharp's pooled allocator (a raw native allocation would bypass the pool
            // accounting); its failure is NumPy's MemoryError, with NumPy's text.
            ArraySlice<ulong> work;
            try
            {
                work = new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>(total));
            }
            catch (OutOfMemoryException)
            {
                throw new OutOfMemoryException($"Insufficient memory for multivariate_hypergeometric with method='count' and sum(colors)={total}");
            }
            ulong* choices = (ulong*)work.Address;
            try
            {
                // If colors contains [3 2 5], choices holds [0 0 0 1 1 2 2 2 2 2].
                for (long i = 0, k = 0; i < numColors; ++i)
                    for (long j = 0; j < colors[i]; ++j)
                        choices[k++] = (ulong)i;

                bool moreThanHalf = nsample > (total / 2);
                if (moreThanHalf)
                    nsample = total - nsample;

                // Every Fisher-Yates step draws at least one word: its bound total - j - 1 is at least 1, because
                // nsample <= total / 2 after the complement. When every bound also fits 32 bits (total - 1 <= 2^32 - 1)
                // the steps draw next_uint32 words only, so a 32-bit read-ahead owed the steps still to come never draws
                // past NumPy's stream. Larger urns draw 64-bit words for their first steps and stay per-draw.
                // The read-ahead's cursor lives in locals (a by-reference buffer would keep its position in memory, and
                // every step would wait on a store-to-load round trip); its refill rule is DrawBuffer32's.
                bool readAhead = total - 1 <= 0xFFFFFFFFL;
                uint* buf = stackalloc uint[DrawBuffer32.Capacity];
                int pos = 0, avail = 0;
                lock (_bitGenerator.@lock)
                    for (long i = 0, row = 0; i < numVariates * numColors; i += numColors, row++)
                    {
                        long rowsAfter = numVariates - row - 1;
                        // Fisher-Yates over the first nsample entries only: choices[:nsample] is then a random sample.
                        for (long j = 0; j < nsample; ++j)
                        {
                            long k;
                            if (readAhead)
                            {
                                // random_interval(total - j - 1): the bound is at least 1 (see above), so gen_mask is one lzcnt.
                                ulong max = (ulong)(total - j - 1);
                                ulong mask = ulong.MaxValue >> BitOperations.LeadingZeroCount(max);
                                ulong value;
                                do
                                {
                                    if (pos == avail)
                                    {
                                        // The steps still owed, this one included: (nsample - j) now + nsample per later
                                        // row, saturated (only a lower bound matters; a refill takes at most the capacity).
                                        long owed = nsample - j;
                                        owed = rowsAfter > (long.MaxValue - owed) / nsample ? long.MaxValue : owed + rowsAfter * nsample;
                                        avail = owed < DrawBuffer32.Capacity ? (int)owed : DrawBuffer32.Capacity;
                                        _bitGenerator.FillUInt32(buf, avail);
                                        pos = 0;
                                    }
                                    value = buf[pos++] & mask;
                                } while (value > max);
                                k = j + (long)value;
                            }
                            else
                            {
                                k = j + (long)BoundedIntegers.RandomInterval(_bitGenerator, (ulong)(total - j - 1));
                            }
                            ulong tmp = choices[k];
                            choices[k] = choices[j];
                            choices[j] = tmp;
                        }
                        // Count each color in choices[:nsample].
                        for (long j = 0; j < nsample; ++j)
                            variates[i + (long)choices[j]] += 1;
                        if (moreThanHalf)
                            for (long k = 0; k < numColors; ++k)
                                variates[i + k] = colors[k] - variates[i + k];
                    }
            }
            finally
            {
                work.DangerousFree(); // the raw pooled buffer is private to this call
            }
        }

        /// <summary>
        ///     NumPy's <c>random_multivariate_hypergeometric_marginals</c>: one univariate hypergeometric per color against
        ///     the colors that remain.
        /// </summary>
        /// <param name="total">The number of items.</param>
        /// <param name="numColors">The number of colors.</param>
        /// <param name="colors">The items per color.</param>
        /// <param name="nsample">The items selected per variate.</param>
        /// <param name="numVariates">The number of variates.</param>
        /// <param name="variates">The zeroed output.</param>
        /// <remarks>
        ///     <para>
        ///     The univariate draws switch between the urn walk (32-bit <c>random_interval</c> words) and HRUA (doubles) as
        ///     the remaining counts change, so no read-ahead may cross a draw: a buffered word would be reordered against
        ///     the other kind. Inside one HRUA draw it is exact, though — every attempt consumes exactly one U/V pair — so the
        ///     doubles come a pair per refill, and the buffer is empty whenever an HRUA draw returns.
        ///     </para>
        ///     <para>
        ///     The <c>(good, bad, sample)</c> triples repeat across variates (the first color's always, the later colors' over
        ///     the few remainders the earlier draws leave), so a fill keeps their HRUA setups — and candidate memos — in a
        ///     small cache instead of re-evaluating NumPy's per-call statements. Both are deterministic functions of the
        ///     triple, so the draws are NumPy's bits.
        ///     </para>
        /// </remarks>
        private unsafe void MultivariateHypergeometricMarginals(long total, long numColors, long* colors, long nsample,
                                                                long numVariates, long* variates)
        {
            if ((total == 0) || (nsample == 0) || (numVariates == 0))
                return;

            bool moreThanHalf = nsample > (total / 2);
            if (moreThanHalf)
                nsample = total - nsample;

            // When every univariate draw the fill can make routes to HRUA (decided from the counts alone, see
            // AllMarginalsUseHrua), the fill draws only doubles, at least one U/V pair per variate: a full read-ahead owed
            // two doubles per variate still to come is then exact. Otherwise the urn walk's 32-bit words may interleave,
            // and the doubles come a pair at a time — exact inside one HRUA draw, where every attempt is one pair.
            bool allHrua = AllMarginalsUseHrua(total, numColors, colors, nsample);
            double* dbuf = stackalloc double[DrawBufferDouble.Capacity];
            var hrua = allHrua
                ? new DrawBufferDouble(_bitGenerator, dbuf, DrawBufferDouble.Capacity)
                : new DrawBufferDouble(_bitGenerator, dbuf, 2) { Owed = 2 };
            // Short fills would not repay the cache's setup; they evaluate each draw's setup like NumPy.
            var cache = numVariates >= 16 ? new HruaSetupCache() : null;
            lock (_bitGenerator.@lock)
                for (long i = 0, row = 0; i < numVariates * numColors; i += numColors, row++)
                {
                    long numToSample = nsample;
                    long remaining = total;
                    // Doubles still owed at any point of this variate: this draw's first pair plus a pair per later variate
                    // (a draw that might not happen owes nothing, so only the one being made counts).
                    if (allHrua)
                        hrua.Owed = 2 + 2 * (numVariates - row - 1);
                    for (long j = 0; (numToSample > 0) && (j + 1 < numColors); ++j)
                    {
                        remaining -= colors[j];
                        long good = colors[j];
                        long r;
                        // NumPy's random_hypergeometric dispatch: HRUA for 10 <= sample <= good + bad - 10.
                        if ((numToSample >= 10) && (numToSample <= good + remaining - 10))
                        {
                            if (cache != null)
                            {
                                ref readonly HruaSetup setup = ref cache.Get(good, remaining, numToSample, out double[] gpMemo);
                                r = HypergeometricHrua(ref hrua, in setup, gpMemo);
                            }
                            else
                            {
                                var setup = new HruaSetup(good, remaining, numToSample);
                                r = HypergeometricHrua(ref hrua, in setup, null);
                            }
                        }
                        else
                        {
                            r = HypergeometricSample(_bitGenerator, good, remaining, numToSample);
                        }
                        variates[i + j] = r;
                        numToSample -= r;
                    }
                    if (numToSample > 0)
                        variates[i + numColors - 1] = numToSample;
                    if (moreThanHalf)
                        for (long k = 0; k < numColors; ++k)
                            variates[i + k] = colors[k] - variates[i + k];
                }
        }

        /// <summary>
        ///     Whether every univariate draw the marginals method can make for these counts routes to HRUA
        ///     (<c>10 &lt;= sample &lt;= good + bad - 10</c>) — decided from the counts alone, before any draw.
        /// </summary>
        /// <param name="total">The number of items.</param>
        /// <param name="numColors">The number of colors.</param>
        /// <param name="colors">The items per color.</param>
        /// <param name="nsample">The items selected per variate (after NumPy's complement).</param>
        /// <returns>True when no possible draw takes the urn walk.</returns>
        /// <remarks>
        ///     Before draw <c>j</c> the items still to sample lie in an interval <c>[lo, hi]</c>: <c>[nsample, nsample]</c> for
        ///     the first, and a draw from <c>(good, bad, s)</c> takes between <c>max(0, s - bad)</c> and <c>min(s, good)</c>, so
        ///     the next interval is <c>[max(0, lo - good), min(hi, bad)]</c>. A draw with nothing left to sample never
        ///     happens (the loop stops), so only the positive part of each interval must route to HRUA.
        /// </remarks>
        private static unsafe bool AllMarginalsUseHrua(long total, long numColors, long* colors, long nsample)
        {
            long lo = nsample, hi = nsample, remaining = total;
            for (long j = 0; j + 1 < numColors; ++j)
            {
                long good = colors[j];
                remaining -= good;
                long positiveLo = lo > 1 ? lo : 1;
                if (positiveLo <= hi && (positiveLo < 10 || hi > good + remaining - 10))
                    return false;
                lo = lo - good > 0 ? lo - good : 0;
                hi = hi < remaining ? hi : remaining;
                if (hi <= 0)
                    break;
            }
            return true;
        }

        /// <summary>
        ///     A direct-mapped cache of HRUA setups (and their candidate memos) for one multivariate-hypergeometric fill,
        ///     keyed by the univariate draw's <c>(good, bad, sample)</c>.
        /// </summary>
        /// <remarks>
        ///     A collision simply re-evaluates the slot for the new triple (NumPy's per-call statements, so nothing but time
        ///     is at stake). Memos are kept only for up to 4096 candidates, capping the cache at 64 x 32 KB.
        /// </remarks>
        private sealed class HruaSetupCache
        {
            /// <summary>log2 of the slot count.</summary>
            private const int SlotBits = 6;

            /// <summary>The slots' triples; a slot is empty until <see cref="_used"/> marks it.</summary>
            private readonly long[] _keys = new long[3 << SlotBits];

            /// <summary>Whether each slot holds a setup.</summary>
            private readonly bool[] _used = new bool[1 << SlotBits];

            /// <summary>The slots' setups.</summary>
            private readonly HruaSetup[] _setups = new HruaSetup[1 << SlotBits];

            /// <summary>The slots' candidate memos (null where not kept).</summary>
            private readonly double[][] _memos = new double[1 << SlotBits][];

            /// <summary>
            ///     The setup for <c>(good, bad, sample)</c> — cached, or evaluated into the triple's slot (evicting a colliding
            ///     triple) — and its candidate memo.
            /// </summary>
            /// <param name="good">Good items.</param>
            /// <param name="bad">Bad items.</param>
            /// <param name="sample">Items drawn (the caller routed the triple to HRUA).</param>
            /// <param name="gpMemo">The triple's NaN-initialized candidate memo, or null when it has more than 4096 candidates.</param>
            /// <returns>A reference to the slot's setup, valid until the slot is next evicted.</returns>
            internal ref readonly HruaSetup Get(long good, long bad, long sample, out double[] gpMemo)
            {
                ulong h = ((ulong)good * 0x9E3779B97F4A7C15UL) ^ ((ulong)bad * 0xC2B2AE3D27D4EB4FUL) ^ (ulong)sample;
                int slot = (int)((h * 0xBF58476D1CE4E5B9UL) >> (64 - SlotBits));
                int k = slot * 3;
                if (!_used[slot] || _keys[k] != good || _keys[k + 1] != bad || _keys[k + 2] != sample)
                {
                    _used[slot] = true;
                    _keys[k] = good;
                    _keys[k + 1] = bad;
                    _keys[k + 2] = sample;
                    _setups[slot] = new HruaSetup(good, bad, sample);
                    double b = _setups[slot].B;
                    double[] memo = null;
                    if (b <= 4096)
                    {
                        memo = new double[(int)b];
                        Array.Fill(memo, double.NaN);
                    }
                    _memos[slot] = memo;
                }
                gpMemo = _memos[slot];
                return ref _setups[slot];
            }
        }

        // =============================================================================================================
        // multivariate_normal
        // =============================================================================================================

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution.
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution (1-D).</param>
        /// <param name="cov">Covariance matrix (N x N, symmetric positive-semidefinite for proper sampling).</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one sample of shape <c>(N,)</c>.</param>
        /// <param name="check_valid">Behaviour when <paramref name="cov"/> is not positive-semidefinite: <c>"warn"</c> (NumPy
        ///     warns; NumSharp has no warnings channel, so this is silent), <c>"raise"</c> or <c>"ignore"</c>.</param>
        /// <param name="tol">Tolerance of the positive-semidefiniteness check.</param>
        /// <param name="method"><c>"svd"</c> (default), <c>"eigh"</c> or <c>"cholesky"</c>: the factorisation of <paramref name="cov"/>.</param>
        /// <returns>The float64 samples, shape <c>(*size, N)</c>.</returns>
        /// <exception cref="ValueError">In NumPy's order: an unknown <paramref name="method"/>; a non-1-D (or null) mean,
        ///     non-square (or null) cov or mismatched lengths; an invalid <paramref name="check_valid"/> (not checked for
        ///     cholesky); a covariance that is not positive-semidefinite under <c>"raise"</c>.</exception>
        /// <exception cref="TypeError"><paramref name="mean"/> or <paramref name="cov"/> is complex (checked before the
        ///     shapes, as NumPy does).</exception>
        /// <exception cref="LinAlgError"><c>"cholesky"</c> of a covariance that is not positive definite.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
        ///     <br/>NumPy's algorithm: the standard normals are drawn FIRST, then <c>mean + x @ factor.T</c> with
        ///     <c>factor = u * sqrt(s)</c> (svd), <c>u * sqrt(|s|)</c> (eigh) or <c>L</c> (cholesky). Byte-identical to
        ///     NumPy when <c>NumSharp.Interop.OpenBLAS</c> supplies LAPACK (the same <c>gesdd</c>/<c>syevd</c>/<c>potrf</c>);
        ///     without it managed factorisations stand in (Jacobi eigen for svd/eigh, Cholesky-Banachiewicz), which give a
        ///     valid sample but can differ in a singular vector's sign.
        /// </remarks>
        public NDArray multivariate_normal(NDArray mean, NDArray cov, Shape size = default, string check_valid = "warn",
                                           double tol = 1e-8, string method = "svd")
        {
            if (method != "eigh" && method != "svd" && method != "cholesky")
                throw new ValueError("method must be one of {'eigh', 'svd', 'cholesky'}");
            // np.array(None) is a 0-d object array: never complex, and it fails the shape checks below — so a null
            // mean or cov reports the complex check of the other operand first, then its own shape message.
            if ((mean is not null && mean.typecode == NPTypeCode.Complex) || (cov is not null && cov.typecode == NPTypeCode.Complex))
                throw new TypeError("mean and cov must not be complex");
            if (mean is null)
                throw new ValueError("mean must be 1 dimensional");
            if (cov is null)
            {
                if (mean.ndim != 1)
                    throw new ValueError("mean must be 1 dimensional");
                throw new ValueError("cov must be 2 dimensional and square");
            }
            return MultivariateNormalCore(mean, cov, size, check_valid, tol, method);
        }

        /// <summary>
        ///     Draw random samples from a multivariate normal distribution (array arguments).
        /// </summary>
        /// <param name="mean">Mean of the N-dimensional distribution.</param>
        /// <param name="cov">Covariance matrix, N x N.</param>
        /// <param name="size">Leading output shape. Default (NumPy's <c>None</c>) draws one sample.</param>
        /// <param name="check_valid"><c>"warn"</c>, <c>"raise"</c> or <c>"ignore"</c>.</param>
        /// <param name="tol">Tolerance of the positive-semidefiniteness check.</param>
        /// <param name="method"><c>"svd"</c>, <c>"eigh"</c> or <c>"cholesky"</c>.</param>
        /// <returns>The float64 samples, shape <c>(*size, N)</c>.</returns>
        /// <exception cref="ValueError">See <see cref="multivariate_normal(NDArray, NDArray, Shape, string, double, string)"/>.</exception>
        /// <exception cref="LinAlgError"><c>"cholesky"</c> of a covariance that is not positive definite.</exception>
        public NDArray multivariate_normal(double[] mean, double[,] cov, Shape size = default, string check_valid = "warn",
                                           double tol = 1e-8, string method = "svd")
        {
            // np.array(mean) / np.array(cov): owned copies that die with this call (the result is a fresh array).
            using var meanArr = mean is null ? null : np.array(mean);
            using var covArr = cov is null ? null : np.array(cov);
            return multivariate_normal(meanArr, covArr, size, check_valid, tol, method);
        }

        /// <summary>The body of <c>multivariate_normal</c>: shape checks, the normals, the factorisation, the transform.</summary>
        /// <param name="mean">The mean (1-D expected).</param>
        /// <param name="cov">The covariance (square 2-D expected).</param>
        /// <param name="size">The leading output shape, or <c>default</c>.</param>
        /// <param name="check_valid">The PSD policy.</param>
        /// <param name="tol">The PSD tolerance.</param>
        /// <param name="method">The factorisation.</param>
        /// <returns>The samples.</returns>
        /// <exception cref="ValueError">A shape, check_valid or PSD check fails.</exception>
        /// <exception cref="LinAlgError">Cholesky of a non positive-definite covariance.</exception>
        [NDScoped] // reclaims the normals, the float64 cov, the factors and the product temporaries
        private NDArray MultivariateNormalCore(NDArray mean, NDArray cov, Shape size, string check_valid, double tol, string method)
        {
            if (mean.ndim != 1)
                throw new ValueError("mean must be 1 dimensional");
            if (cov.ndim != 2 || cov.shape[0] != cov.shape[1])
                throw new ValueError("cov must be 2 dimensional and square");
            if (mean.shape[0] != cov.shape[0])
                throw new ValueError("mean and cov must have same length");

            long n = mean.shape[0];
            long[] finalShape = OutputDimsWithTrailing(size, n);
            // The normals are drawn BEFORE the factorisation, so a later failure has already advanced the stream.
            NDArray normals = standard_normal(new Shape(finalShape));
            // x.reshape(-1, 0) cannot infer -1 from a zero-size array: NumPy leaks that reshape's ValueError (after the
            // empty draw), and NumSharp's own reshape would raise its house IncorrectShapeException with the same text.
            if (n == 0)
                throw new ValueError("cannot reshape array of size 0 into shape (0)");
            NDArray x = normals.reshape(-1, n);

            // GH10839, ensure double to make tol meaningful
            NDArray covD = cov.astype(np.float64, copy: true);
            NDArray u = null, s = null, vh = null, l = null;
            if (method == "svd")
                (u, s, vh) = SvdForMultivariateNormal(covD, n);
            else if (method == "eigh")
                (s, u) = EighForMultivariateNormal(covD, n);
            else
                l = CholeskyForMultivariateNormal(covD, n);

            // check_valid is ignored for cholesky: the decomposition would already have failed for an invalid cov.
            if (check_valid != "ignore" && method != "cholesky")
            {
                if (check_valid != "warn" && check_valid != "raise")
                    throw new ValueError("check_valid must equal 'warn', 'raise', or 'ignore'");
                bool psd = method == "svd"
                    ? np.allclose(np.dot(vh.T * s, vh), covD, rtol: tol, atol: tol)
                    : !(bool)np.any(s < -tol);
                // NumPy warns (RuntimeWarning) for "warn"; NumSharp has no warnings channel, so only "raise" is observable.
                if (!psd && check_valid == "raise")
                    throw new ValueError("covariance is not symmetric positive-semidefinite.");
            }

            NDArray factor = method == "cholesky" ? l
                : method == "eigh" ? u * np.sqrt(np.abs(s))
                : u * np.sqrt(s);

            NDArray result = mean + np.matmul(x, factor.T);
            return result.reshape(new Shape(finalShape));
        }

        /// <summary>
        ///     <c>(u, s, vh)</c> of NumPy's <c>svd(cov)</c>: the installed LAPACK backend's <c>gesdd</c> when there is one,
        ///     otherwise a managed Jacobi eigendecomposition read as the SVD of a symmetric matrix.
        /// </summary>
        /// <param name="covD">The float64 covariance, N x N.</param>
        /// <param name="n">N.</param>
        /// <returns><c>u</c> (N x N), the singular values (descending) and <c>vh</c> (N x N).</returns>
        /// <remarks>
        ///     For a symmetric <c>cov = V diag(λ) V^T</c> the SVD is <c>U diag(|λ|) V^T</c> with <c>U</c>'s columns those of
        ///     <c>V</c> signed by <c>λ</c>; the fallback builds exactly that from the legacy sampler's Jacobi routines.
        /// </remarks>
        private static (NDArray u, NDArray s, NDArray vh) SvdForMultivariateNormal(NDArray covD, long n)
        {
            if (n == 0)
                return (np.empty(new Shape(0L, 0L), np.float64), np.empty(new Shape(0L), np.float64), np.empty(new Shape(0L, 0L), np.float64));
            var blas = covD.TensorEngine.Blas; // read once: a concurrent Disable() must not null it between test and call
            if (blas != null && blas.TrySvd(covD, true, true, out var u, out var s, out var vh))
                return (u, s, vh);

            var (lambda, v) = ManagedSymmetricEigen(covD, n, descending: true);
            var su = new NDArray(NPTypeCode.Double, new Shape(n), false);
            var uu = new NDArray(NPTypeCode.Double, new Shape(n, n), false);
            var vv = new NDArray(NPTypeCode.Double, new Shape(n, n), false);
            unsafe
            {
                var sp = (double*)su.Address;
                var up = (double*)uu.Address;
                var vp = (double*)vv.Address;
                for (long i = 0; i < n; i++)
                {
                    sp[i] = Math.Abs(lambda[i]);
                    double sign = lambda[i] < 0 ? -1.0 : 1.0;
                    for (long j = 0; j < n; j++)
                    {
                        up[j * n + i] = sign * v[j * n + i]; // u[:, i] = sign(λ_i) * V[:, i]
                        vp[i * n + j] = v[j * n + i];        // vh[i, :] = V[:, i]
                    }
                }
            }
            return (uu, su, vv);
        }

        /// <summary>
        ///     <c>(w, v)</c> of NumPy's <c>eigh(cov)</c>: the backend's <c>syevd</c>, else a managed Jacobi
        ///     eigendecomposition (eigenvalues ascending, as <c>eigh</c> returns them).
        /// </summary>
        /// <param name="covD">The float64 covariance.</param>
        /// <param name="n">N.</param>
        /// <returns>The eigenvalues (ascending) and eigenvectors (columns).</returns>
        private static (NDArray s, NDArray u) EighForMultivariateNormal(NDArray covD, long n)
        {
            if (n == 0)
                return (np.empty(new Shape(0L), np.float64), np.empty(new Shape(0L, 0L), np.float64));
            var blas = covD.TensorEngine.Blas;
            if (blas != null && blas.TryEigh(covD, 'L', true, out var w, out var vecs))
                return (w, vecs);

            var (lambda, v) = ManagedSymmetricEigen(covD, n, descending: false);
            var ws = new NDArray(NPTypeCode.Double, new Shape(n), false);
            var vs = new NDArray(NPTypeCode.Double, new Shape(n, n), false);
            unsafe
            {
                var wp = (double*)ws.Address;
                var vp = (double*)vs.Address;
                for (long i = 0; i < n; i++)
                    wp[i] = lambda[i];
                for (long i = 0; i < n * n; i++)
                    vp[i] = v[i];
            }
            return (ws, vs);
        }

        /// <summary>
        ///     The lower Cholesky factor NumPy's <c>cholesky(cov)</c> returns: the backend's <c>potrf</c>, else the managed
        ///     Cholesky-Banachiewicz factorisation.
        /// </summary>
        /// <param name="covD">The float64 covariance.</param>
        /// <param name="n">N.</param>
        /// <returns><c>L</c>, lower triangular with <c>L @ L.T == cov</c>.</returns>
        /// <exception cref="LinAlgError">The covariance is not positive definite (<c>Matrix is not positive definite</c>).</exception>
        private static NDArray CholeskyForMultivariateNormal(NDArray covD, long n)
        {
            if (n == 0)
                return np.empty(new Shape(0L, 0L), np.float64);
            var blas = covD.TensorEngine.Blas;
            if (blas != null && blas.TryCholesky(covD, false, out var lf))
                return lf;

            var L = new NDArray(NPTypeCode.Double, new Shape(n, n), true);
            unsafe
            {
                var lp = (double*)L.Address;
                for (long i = 0; i < n; i++)
                {
                    for (long j = 0; j <= i; j++)
                    {
                        double sum = covD.GetDouble(i, j);
                        for (long k = 0; k < j; k++)
                            sum -= lp[i * n + k] * lp[j * n + k];
                        if (i == j)
                        {
                            // LAPACK's potrf fails on a non-positive (or NaN) pivot; NumPy reports it as below.
                            if (!(sum > 0))
                                throw new LinAlgError("Matrix is not positive definite");
                            lp[i * n + i] = Math.Sqrt(sum);
                        }
                        else
                        {
                            lp[i * n + j] = sum / lp[j * n + j];
                        }
                    }
                }
            }
            return L;
        }

        /// <summary>
        ///     A managed symmetric eigendecomposition over the legacy sampler's Jacobi routines: eigenvalues sorted
        ///     (descending for the SVD reading, ascending for <c>eigh</c>) with their eigenvector columns.
        /// </summary>
        /// <param name="covD">The float64 symmetric matrix, N x N (N &gt; 0).</param>
        /// <param name="n">N.</param>
        /// <param name="descending">Sort the eigenvalues descending (else ascending).</param>
        /// <returns>The eigenvalues and the row-major N x N eigenvector matrix (column i pairs with eigenvalue i).</returns>
        private static (double[] lambda, double[] v) ManagedSymmetricEigen(NDArray covD, long n, bool descending)
        {
            var work = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n * n));
            var vecs = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n * n));
            var vals = new ArraySlice<double>(new UnmanagedMemoryBlock<double>(n));
            try
            {
                for (long i = 0; i < n; i++)
                    for (long j = 0; j < n; j++)
                    {
                        work[i * n + j] = covD.GetDouble(i, j);
                        vecs[i * n + j] = i == j ? 1.0 : 0.0;
                    }
                NumPyRandom.JacobiEigendecomposition(work, vecs, vals, n, 100, 1e-12);
                // Both sorts are stable insertion sorts, so tied eigenvalues keep Jacobi's column order — which is what
                // makes an identity (or any diagonal) covariance come back as the untouched basis, exactly like LAPACK.
                // (Reversing the descending order instead would swap tied columns and permute the samples.)
                if (descending)
                    NumPyRandom.SortEigenDescending(vals, vecs, n);
                else
                    SortEigenAscending(vals, vecs, n);
                NumPyRandom.NormalizeEigenvectorSigns(vecs, n);

                var lambda = new double[n];
                var v = new double[n * n];
                for (long i = 0; i < n; i++)
                {
                    lambda[i] = vals[i];
                    for (long r = 0; r < n; r++)
                        v[r * n + i] = vecs[r * n + i];
                }
                return (lambda, v);
            }
            finally
            {
                work.DangerousFree();
                vecs.DangerousFree();
                vals.DangerousFree();
            }
        }

        /// <summary>
        ///     Sorts eigenvalues ascending (<c>eigh</c>'s order), moving their eigenvector columns with them — a stable
        ///     insertion sort, the ascending twin of <see cref="NumPyRandom.SortEigenDescending"/>.
        /// </summary>
        /// <param name="eigenvalues">The n eigenvalues, reordered in place.</param>
        /// <param name="eigenvectors">The row-major n x n eigenvector matrix; column i pairs with eigenvalue i.</param>
        /// <param name="n">The matrix order (small: a covariance's dimension).</param>
        private static void SortEigenAscending(ArraySlice<double> eigenvalues, ArraySlice<double> eigenvectors, long n)
        {
            for (long i = 1; i < n; i++)
            {
                double keyVal = eigenvalues[i];
                long j = i - 1;
                // Strict '>' keeps equal eigenvalues in their current order (stability).
                while (j >= 0 && eigenvalues[j] > keyVal)
                {
                    eigenvalues[j + 1] = eigenvalues[j];
                    eigenvalues[j] = keyVal;
                    for (long k = 0; k < n; k++)
                    {
                        double temp = eigenvectors[k * n + (j + 1)];
                        eigenvectors[k * n + (j + 1)] = eigenvectors[k * n + j];
                        eigenvectors[k * n + j] = temp;
                    }
                    j--;
                }
            }
        }

        // =============================================================================================================
        // shared shape helpers
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>shape = (k,)</c> for <c>size=None</c>, else <c>tuple(size) + (k,)</c> — the output of the vector
        ///     samplers.
        /// </summary>
        /// <param name="size">The leading shape, or <c>default</c> for <c>None</c>.</param>
        /// <param name="k">The trailing (vector) extent.</param>
        /// <returns>The output dimensions.</returns>
        private static long[] OutputDimsWithTrailing(Shape size, long k)
        {
            if (IsNoSize(size))
                return new[] { k };
            var dims = new long[size.NDim + 1];
            for (int i = 0; i < size.NDim; i++)
                dims[i] = size.dimensions[i];
            dims[size.NDim] = k;
            return dims;
        }

        /// <summary>Whether two dimension vectors are equal (rank and extents).</summary>
        /// <param name="a">The first dimensions.</param>
        /// <param name="b">The second dimensions.</param>
        /// <returns>True when equal.</returns>
        private static bool SameDims(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>A Python tuple repr of dimensions: <c>()</c>, <c>(3,)</c>, <c>(2, 3)</c> — NumPy's error-message spelling.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The repr.</returns>
        private static string PyTuple(long[] dims)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(dims[i]);
            }
            if (dims.Length == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }
    }
}
