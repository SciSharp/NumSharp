// NumPy's np.bincount (arr_bincount, numpy/_core/src/multiarray/compiled_base.c) as DefaultEngine's member: the depth and
// dtype checks in NumPy's order, a fused min / max scan, and the count and weighted kernels. Its zero-filled results
// are the allocation constructor (np.zeros's statement), so the engine calls no np.* function.

using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // Privatized-accumulator tuning for the (unweighted) count kernel. The counting scatter
        // `ans[x[i]]++` stalls on store-to-load forwarding when the SAME bin repeats in quick
        // succession (the histogram-like common case: few bins, many samples). Splitting the
        // scatter across K independent accumulator arrays (round-robin by unroll lane) breaks that
        // carried dependency, then a single SIMD pass merges the partials. Integer addition is
        // associative and cannot overflow a count (≤ len < 2^63), so this is BIT-EXACT with the
        // sequential loop regardless of K — privatization is a pure throughput lever here.
        //
        // Gated because the merge + the extra zeroed buffers cost O(K·ansSize): privatization only
        // pays when the bin range is small AND there are enough samples to collide. For a sparse
        // range (ansSize ≫ len) the scatter is memory-latency bound, collisions are rare, and the
        // merge is pure overhead — so we fall through to the plain single-accumulator scatter, the
        // same wall NumPy's own scalar loop hits. Constants tuned by benchmark/bincount.
        // K=4 measured the sweet spot (ties K=8 at ≤8 bins, beats it by ~1024). The 2048-bin
        // ceiling is the cache gate: 4 accumulators × 2048 × 8 B = 64 KB fits L2, and the win is
        // present through ~1536 bins, a wash at 2048, and a net LOSS above (the K arrays spill L2 —
        // measured 0.72× at 3072 bins). The 4096-element floor + `len ≥ 4·ansSize` gate is the
        // amortization guard: below it the alloc+zero+merge overhead outweighs the stall savings.
        private const int BincountAccumulators = 4;          // K (round-robin lanes)
        private const long BincountPrivMaxBins = 2048;        // ansSize ceiling (L2 cache gate)
        private const long BincountPrivMinLen = 4096;         // below this, overhead > win

        /// <inheritdoc />
        [NDScoped] // the int64 / float64 copies of x and the weights are this call's own
        public override NDArray Bincount(NDArray x, NDArray weights = null, int minlength = 0)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));

            // --- x must be exactly 1-D (NumPy PyArray_FromAny with min_depth == max_depth == 1). ---
            // The verbatim NumPy ValueError texts, mapped to the house dimension-error exception.
            if (x.ndim == 0)
                throw new IncorrectShapeException("object of too small depth for desired array");
            if (x.ndim > 1)
                throw new IncorrectShapeException("object too deep for desired array");

            // --- x dtype gate. NumPy casts `list` to intp; an actual float/complex ndarray fails the
            // 'safe' cast and raises TypeError (only the Python-list path is deprecated-but-accepted).
            // NumSharp inputs are always array-like, so we take the ndarray branch and raise — the
            // same stance np.take/np.put take for a non-integer index array. Integer family + bool +
            // char (a uint16 code unit) are the accepted, castable inputs. ---
            NPTypeCode xtc = x.typecode;
            if (!(xtc.IsInteger() || xtc == NPTypeCode.Boolean || xtc == NPTypeCode.Char))
                throw new InvalidCastException(
                    $"Cannot cast array data from {x.dtype.ToString(true)} " +
                    $"to dtype('int64') according to the rule 'safe'");

            if (minlength < 0)
                throw new ValueError("'minlength' must not be negative");

            // Materialize x as a contiguous, offset-free int64 buffer. `.Address` ignores
            // Shape.offset, so a sliced/broadcast view must be copied (the astype already produces a
            // fresh contiguous array for a non-int64 dtype; uint64 → int64 wraps, exactly as NumPy's
            // FORCECAST does, so a value ≥ 2^63 becomes negative and is caught below).
            NDArray xi = xtc == NPTypeCode.Int64 ? x : x.astype(NPTypeCode.Int64);
            if (xi.Shape.IsSliced || xi.Shape.IsBroadcasted)
                xi = xi.copy();

            long len = xi.size;

            // Empty input returns zeros(minlength) as int64 — even when weights are present and
            // mismatched, NumPy returns here before ever looking at weights.
            if (len == 0)
                return NewArray(NPTypeCode.Int64, Shape.Vector(minlength), true);

            MinMaxInt64(xi, len, out long mn, out long mx);
            if (mn < 0)
                throw new ValueError("'list' argument must have no negative elements");

            long ansSize = mx + 1;
            if (ansSize < minlength) ansSize = minlength;

            if (weights is null)
                return BincountCount(xi, len, ansSize);

            // --- weights: 1-D, castable to float64, same length as x (validated only AFTER the
            // negative-element check, matching NumPy's order). ---
            if (weights.ndim == 0)
                throw new IncorrectShapeException("object of too small depth for desired array");
            if (weights.ndim > 1)
                throw new IncorrectShapeException("object too deep for desired array");

            NPTypeCode wtc = weights.typecode;
            if (wtc == NPTypeCode.Complex || wtc == NPTypeCode.String)
                throw new InvalidCastException(
                    $"Cannot cast array data from {weights.dtype.ToString(true)} " +
                    $"to dtype('float64') according to the rule 'safe'");

            NDArray w64 = wtc == NPTypeCode.Double ? weights : weights.astype(NPTypeCode.Double);
            if (w64.Shape.IsSliced || w64.Shape.IsBroadcasted)
                w64 = w64.copy();

            if (w64.size != len)
                throw new ValueError("The weights and list don't have the same length.");

            return BincountWeighted(xi, w64, len, ansSize);
        }

        /// <summary>
        ///     Fused single-pass SIMD min+max over a contiguous int64 buffer. bincount needs the min
        ///     (for the non-negative check) and the max (to size the output) together, so both fold
        ///     into one pass — the same scan NumPy's <c>minmax()</c> does, but vectorized.
        /// </summary>
        /// <param name="xi">The contiguous, offset-free int64 values.</param>
        /// <param name="n">How many (at least one).</param>
        /// <param name="mn">The smallest value.</param>
        /// <param name="mx">The largest value.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MinMaxInt64(NDArray xi, long n, out long mn, out long mx)
        {
            long* p = (long*)xi.Address;
            long min = p[0];
            long max = p[0];
            long i = 1;

            if (Vector.IsHardwareAccelerated && n >= (long)Vector<long>.Count * 2)
            {
                int w = Vector<long>.Count;
                Vector<long> vmin = Unsafe.ReadUnaligned<Vector<long>>(p);
                Vector<long> vmax = vmin;
                long vend = n - (n % w);
                for (long k = w; k < vend; k += w)
                {
                    Vector<long> v = Unsafe.ReadUnaligned<Vector<long>>((void*)(p + k));
                    vmin = Vector.Min(vmin, v);
                    vmax = Vector.Max(vmax, v);
                }
                for (int j = 0; j < w; j++)
                {
                    if (vmin[j] < min) min = vmin[j];
                    if (vmax[j] > max) max = vmax[j];
                }
                i = vend;
            }

            for (; i < n; i++)
            {
                long v = p[i];
                if (v < min) min = v;
                else if (v > max) max = v;
            }

            mn = min;
            mx = max;
        }

        /// <summary>
        ///     Unweighted count kernel: <c>ans[x[i]]++</c>. Uses privatized accumulators (see the
        ///     tuning constants above) to break the store-forwarding stall on repeated bins, then a
        ///     SIMD merge. Bit-exact with the sequential scatter for every K (integer add).
        /// </summary>
        /// <param name="xi">The contiguous, offset-free, non-negative int64 values (each below
        ///     <paramref name="ansSize"/>).</param>
        /// <param name="len">How many.</param>
        /// <param name="ansSize">The number of bins.</param>
        /// <returns>A new int64 array of <paramref name="ansSize"/> counts.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private unsafe NDArray BincountCount(NDArray xi, long len, long ansSize)
        {
            NDArray ans = NewArray(NPTypeCode.Int64, Shape.Vector(ansSize), true);
            long* xp = (long*)xi.Address;
            long* a0 = (long*)ans.Address;

            if (ansSize <= BincountPrivMaxBins && len >= BincountPrivMinLen &&
                len >= ansSize * 4 && BincountAccumulators == 4)
            {
                // K=4 privatized accumulators. The primary output IS accumulator 0; the other three
                // live in a zeroed unmanaged scratch (never a managed >85 KB LOH allocation).
                long extraLen = ansSize * 3;
                long* ex = (long*)NativeMemory.AllocZeroed((nuint)(extraLen * sizeof(long)));
                try
                {
                    long* a1 = ex;
                    long* a2 = ex + ansSize;
                    long* a3 = ex + 2 * ansSize;

                    long i = 0;
                    long limit = len - (len & 3);
                    for (; i < limit; i += 4)
                    {
                        a0[xp[i]]++;
                        a1[xp[i + 1]]++;
                        a2[xp[i + 2]]++;
                        a3[xp[i + 3]]++;
                    }
                    for (; i < len; i++)
                        a0[xp[i]]++;

                    MergeAccumulators(a0, a1, a2, a3, ansSize);
                }
                finally
                {
                    NativeMemory.Free(ex);
                }
            }
            else
            {
                for (long i = 0; i < len; i++)
                    a0[xp[i]]++;
            }

            return ans;
        }

        /// <summary>SIMD merge of the three extra count accumulators back into the primary output.</summary>
        /// <param name="a0">The primary accumulator (the result), added into.</param>
        /// <param name="a1">The second accumulator.</param>
        /// <param name="a2">The third accumulator.</param>
        /// <param name="a3">The fourth accumulator.</param>
        /// <param name="n">The number of bins.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MergeAccumulators(long* a0, long* a1, long* a2, long* a3, long n)
        {
            long i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int w = Vector<long>.Count;
                long vend = n - (n % w);
                for (; i < vend; i += w)
                {
                    Vector<long> v0 = Unsafe.ReadUnaligned<Vector<long>>((void*)(a0 + i));
                    Vector<long> v1 = Unsafe.ReadUnaligned<Vector<long>>((void*)(a1 + i));
                    Vector<long> v2 = Unsafe.ReadUnaligned<Vector<long>>((void*)(a2 + i));
                    Vector<long> v3 = Unsafe.ReadUnaligned<Vector<long>>((void*)(a3 + i));
                    Unsafe.WriteUnaligned((void*)(a0 + i), v0 + v1 + v2 + v3);
                }
            }
            for (; i < n; i++)
                a0[i] += a1[i] + a2[i] + a3[i];
        }

        /// <summary>
        ///     Weighted kernel: <c>ans[x[i]] += w[i]</c>, in strict logical order. This path is
        ///     deliberately a single sequential accumulator: float addition is non-associative, so
        ///     privatizing would reorder the per-bin sums and diverge from NumPy at the ULP level.
        ///     Sequential keeps it BYTE-EXACT with NumPy's own loop (verified incl. catastrophic
        ///     cancellation and NaN/inf propagation).
        /// </summary>
        /// <param name="xi">The contiguous, offset-free, non-negative int64 values (each below
        ///     <paramref name="ansSize"/>).</param>
        /// <param name="w64">The contiguous, offset-free float64 weights, one per value.</param>
        /// <param name="len">How many values.</param>
        /// <param name="ansSize">The number of bins.</param>
        /// <returns>A new float64 array of <paramref name="ansSize"/> weighted sums.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private unsafe NDArray BincountWeighted(NDArray xi, NDArray w64, long len, long ansSize)
        {
            NDArray ans = NewArray(NPTypeCode.Double, Shape.Vector(ansSize), true);
            long* xp = (long*)xi.Address;
            double* wp = (double*)w64.Address;
            double* dans = (double*)ans.Address;

            for (long i = 0; i < len; i++)
                dans[xp[i]] += wp[i];

            return ans;
        }
    }
}
