using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged.Pooling;

// =============================================================================
// DirectILKernelGenerator.Histogram.Range.cs — the equal-width index kernels for float16 / float32 data counted over a
// RANGE.
//
// RESPONSIBILITY
//   NumPy's index statements are the same with or without a range,
//
//       f_indices = (_unsigned_subtract(tmp_a, first_edge) / norm_denom) * n_equal_bins
//
//   but their dtypes are not. Without a range the outer edges are the data's own float16 / float32 scalars, so every
//   operand is in the data's dtype and the quotient is too (DirectILKernelGenerator.Histogram.cs). With a range the
//   outer edges are Python floats: NEP 50 rounds them to the data's dtype where they meet the data (the keep mask and
//   the subtraction), but norm_denom = _unsigned_subtract(last_edge, first_edge) subtracts the two Python floats into an
//   np.float64 scalar, a STRONG operand, so the division and the product with n_equal_bins are float64.
//
// WHY IT IS OBSERVABLE
//   When the range is about as narrow as the dtype's spacing, the rounded edges sit farther apart than the range itself,
//   and a kept sample's float64 quotient can pass n_equal_bins by more than the one-bin corrections reach: NumPy's
//   fancy index `bin_edges[indices]` then raises (float16 [10080, 10064], bins=2, range=(10066.28, 10076.73): "index 3
//   is out of bounds for axis 0 with size 3"). Computing the quotient in the data's width gives an index in range and
//   counts the samples.
//
// THE INDEX
//   .astype(np.intp) is C's double-to-int64 conversion, which x86 turns into INT64_MIN for a NaN or an out-of-range value
//   (a quotient of a tiny norm_denom); .NET's conversion saturates instead, so the kernels spell x86's result. The
//   refusal names the first such sample in the data's order, as NumPy's fancy index does within the first block that
//   holds one; the blocks before it change nothing NumPy returns, so no partial result is observable.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        private static readonly ConcurrentDictionary<NPTypeCode, HistogramCountKernel> _histRangeCountCache = new();
        private static readonly ConcurrentDictionary<NPTypeCode, HistogramWeightedKernel> _histRangeWeightedCache = new();

        /// <summary>
        ///     Returns the cached fused count kernel for float16 / float32 data counted over a range: the
        ///     <see cref="HistogramCountKernel"/> contract, with NumPy's float64 quotient (see the file header).
        /// </summary>
        /// <param name="binDtype">The data's and the edges' dtype: <see cref="NPTypeCode.Half"/> or
        ///     <see cref="NPTypeCode.Single"/> (float64 data's quotient is float64 either way: use
        ///     <see cref="GetHistogramCountKernel"/>).</param>
        /// <returns>The cached kernel. It throws <see cref="IndexError"/> for a kept sample whose index passes the edges.</returns>
        /// <exception cref="NotSupportedException">Another dtype.</exception>
        public static unsafe HistogramCountKernel GetHistogramRangeCountKernel(NPTypeCode binDtype) =>
            _histRangeCountCache.GetOrAdd(binDtype, static b => b switch
            {
                NPTypeCode.Single => static (s, c, e, f, l, n, o) => HistogramRangeCountBody<float>(s, c, e, f, l, n, o),
                NPTypeCode.Half => static (s, c, e, f, l, n, o) => HistogramRangeCountBody<Half>(s, c, e, f, l, n, o),
                _ => throw new NotSupportedException($"Histogram range kernels are float16 / float32 only (got {b})."),
            });

        /// <summary>
        ///     Returns the cached fused weighted-scatter kernel for float16 / float32 data counted over a range: the
        ///     <see cref="HistogramWeightedKernel"/> contract (one block, float64 totals), with NumPy's float64 quotient.
        /// </summary>
        /// <param name="binDtype"><see cref="NPTypeCode.Half"/> or <see cref="NPTypeCode.Single"/>.</param>
        /// <returns>The cached kernel. It throws <see cref="IndexError"/> for a kept sample whose index passes the edges.</returns>
        /// <exception cref="NotSupportedException">Another dtype.</exception>
        public static unsafe HistogramWeightedKernel GetHistogramRangeWeightedKernel(NPTypeCode binDtype) =>
            _histRangeWeightedCache.GetOrAdd(binDtype, static b => b switch
            {
                NPTypeCode.Single => static (s, w, c, e, f, l, n, o) => HistogramRangeWeightedBody<float>(s, w, c, e, f, l, n, o),
                NPTypeCode.Half => static (s, w, c, e, f, l, n, o) => HistogramRangeWeightedBody<Half>(s, w, c, e, f, l, n, o),
                _ => throw new NotSupportedException($"Histogram range kernels are float16 / float32 only (got {b})."),
            });

        /// <summary>
        ///     The count loop over a range for one dtype (float or Half), with the privatized accumulators of
        ///     <see cref="HistogramCountBody{T}"/> (four round-robin count buffers merged at the end; the three extra ones
        ///     come from <see cref="SizeBucketedBufferPool"/> zeroed and go back to it, an exception included).
        /// </summary>
        /// <typeparam name="T">float or Half: the JIT folds the other branch away.</typeparam>
        /// <param name="samplesP">The contiguous samples in the dtype.</param>
        /// <param name="count">The number of samples.</param>
        /// <param name="edgesP">The contiguous <c>nbins + 1</c> edges in the dtype.</param>
        /// <param name="first">The range's lower Python float.</param>
        /// <param name="last">The range's upper Python float.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <param name="counts">The pre-zeroed int64 counts, <paramref name="nbins"/> of them.</param>
        /// <exception cref="IndexError">A kept sample whose index passes the edges (the first, in the data's order).</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistogramRangeCountBody<T>(
            void* samplesP, long count, void* edgesP, double first, double last, int nbins, long* counts) where T : unmanaged
        {
            // norm_denom: the two Python floats subtracted in float64.
            double nd = last - first;
            bool privatize = nbins <= HistCountPrivMaxBins && count >= HistCountPrivMinLen && count >= (long)nbins * 4;
            long* a1 = null, a2 = null, a3 = null;
            long extraBytes = (long)nbins * 3 * sizeof(long);
            if (privatize)
            {
                long* ex = (long*)SizeBucketedBufferPool.TakeZeroed(extraBytes);
                a1 = ex; a2 = ex + nbins; a3 = ex + 2 * nbins;
            }
            try
            {
                if (typeof(T) == typeof(float))
                {
                    float* s = (float*)samplesP; float* e = (float*)edgesP;
                    // NEP 50: the Python floats meet the float32 data rounded to float32.
                    float fe = (float)first, le = (float)last;
                    if (privatize)
                    {
                        long i = 0, lim = count - (count & 3);
                        for (; i < lim; i += 4)
                        {
                            HistRangeScatterF(counts, s[i], e, fe, le, nd, nbins);
                            HistRangeScatterF(a1, s[i + 1], e, fe, le, nd, nbins);
                            HistRangeScatterF(a2, s[i + 2], e, fe, le, nd, nbins);
                            HistRangeScatterF(a3, s[i + 3], e, fe, le, nd, nbins);
                        }
                        for (; i < count; i++) HistRangeScatterF(counts, s[i], e, fe, le, nd, nbins);
                    }
                    else
                        for (long i = 0; i < count; i++) HistRangeScatterF(counts, s[i], e, fe, le, nd, nbins);
                }
                else if (typeof(T) == typeof(Half))
                {
                    Half* s = (Half*)samplesP; Half* e = (Half*)edgesP;
                    float fe = (float)(Half)first, le = (float)(Half)last;
                    if (privatize)
                    {
                        long i = 0, lim = count - (count & 3);
                        for (; i < lim; i += 4)
                        {
                            HistRangeScatterH(counts, s[i], e, fe, le, nd, nbins);
                            HistRangeScatterH(a1, s[i + 1], e, fe, le, nd, nbins);
                            HistRangeScatterH(a2, s[i + 2], e, fe, le, nd, nbins);
                            HistRangeScatterH(a3, s[i + 3], e, fe, le, nd, nbins);
                        }
                        for (; i < count; i++) HistRangeScatterH(counts, s[i], e, fe, le, nd, nbins);
                    }
                    else
                        for (long i = 0; i < count; i++) HistRangeScatterH(counts, s[i], e, fe, le, nd, nbins);
                }
                else
                    throw new NotSupportedException($"HistogramRangeCountBody: unsupported dtype {typeof(T)}.");

                if (privatize)
                {
                    // Fold the three extra accumulators into the primary counts (integer adds: any order is exact).
                    long i = 0;
                    if (System.Numerics.Vector.IsHardwareAccelerated)
                    {
                        int w = System.Numerics.Vector<long>.Count;
                        long vend = nbins - (nbins % w);
                        for (; i < vend; i += w)
                        {
                            var v0 = Unsafe.ReadUnaligned<System.Numerics.Vector<long>>((void*)(counts + i));
                            var v1 = Unsafe.ReadUnaligned<System.Numerics.Vector<long>>((void*)(a1 + i));
                            var v2 = Unsafe.ReadUnaligned<System.Numerics.Vector<long>>((void*)(a2 + i));
                            var v3 = Unsafe.ReadUnaligned<System.Numerics.Vector<long>>((void*)(a3 + i));
                            Unsafe.WriteUnaligned((void*)(counts + i), v0 + v1 + v2 + v3);
                        }
                    }
                    for (; i < nbins; i++) counts[i] += a1[i] + a2[i] + a3[i];
                }
            }
            finally
            {
                // The refusal of a sample's index leaves through here too: the accumulators go back either way, with the
                // size they were taken with (the pool's contract).
                if (privatize) SizeBucketedBufferPool.Return((IntPtr)a1, extraBytes);
            }
        }

        /// <summary>
        ///     The weighted scatter over a range for one dtype and one block: each kept sample's float64 weight added to its
        ///     bin's float64 total in the data's order (float addition is order-sensitive, so no privatization).
        /// </summary>
        /// <typeparam name="T">float or Half.</typeparam>
        /// <param name="samplesP">The block's contiguous samples in the dtype.</param>
        /// <param name="weightsP">The block's contiguous float64 weights.</param>
        /// <param name="count">The block's length.</param>
        /// <param name="edgesP">The contiguous <c>nbins + 1</c> edges in the dtype.</param>
        /// <param name="first">The range's lower Python float.</param>
        /// <param name="last">The range's upper Python float.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <param name="totals">The pre-zeroed float64 totals, <paramref name="nbins"/> of them.</param>
        /// <exception cref="IndexError">A kept sample whose index passes the edges (the first, in the block's order).</exception>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistogramRangeWeightedBody<T>(
            void* samplesP, void* weightsP, long count, void* edgesP, double first, double last, int nbins, double* totals)
            where T : unmanaged
        {
            double* wp = (double*)weightsP;
            double nd = last - first;
            if (typeof(T) == typeof(float))
            {
                float* s = (float*)samplesP; float* e = (float*)edgesP;
                float fe = (float)first, le = (float)last;
                for (long i = 0; i < count; i++)
                {
                    long idx = HistRangeIndexF(s[i], e, fe, le, nd, nbins);
                    if (idx >= 0)
                        totals[idx] += wp[i];
                }
            }
            else if (typeof(T) == typeof(Half))
            {
                Half* s = (Half*)samplesP; Half* e = (Half*)edgesP;
                float fe = (float)(Half)first, le = (float)(Half)last;
                for (long i = 0; i < count; i++)
                {
                    long idx = HistRangeIndexH(s[i], e, fe, le, nd, nbins);
                    if (idx >= 0)
                        totals[idx] += wp[i];
                }
            }
            else
                throw new NotSupportedException($"HistogramRangeWeightedBody: unsupported dtype {typeof(T)}.");
        }

        /// <summary>Counts one float32 sample over a range (skips a sample outside the outer edges, NaN included).</summary>
        /// <param name="acc">The accumulator to increment.</param>
        /// <param name="x">The sample.</param>
        /// <param name="e">The float32 edges.</param>
        /// <param name="fe">The lower edge rounded to float32.</param>
        /// <param name="le">The upper edge rounded to float32.</param>
        /// <param name="nd">The float64 norm_denom.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <exception cref="IndexError">The sample's index passes the edges.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HistRangeScatterF(long* acc, float x, float* e, float fe, float le, double nd, int nbins)
        {
            long idx = HistRangeIndexF(x, e, fe, le, nd, nbins);
            if (idx >= 0)
                acc[idx]++;
        }

        /// <summary>Counts one float16 sample over a range (skips a sample outside the outer edges, NaN included).</summary>
        /// <param name="acc">The accumulator to increment.</param>
        /// <param name="x">The sample.</param>
        /// <param name="e">The float16 edges.</param>
        /// <param name="fe">The lower edge rounded to float16, widened to float32 (exact).</param>
        /// <param name="le">The upper edge rounded to float16, widened to float32 (exact).</param>
        /// <param name="nd">The float64 norm_denom.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <exception cref="IndexError">The sample's index passes the edges.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HistRangeScatterH(long* acc, Half x, Half* e, float fe, float le, double nd, int nbins)
        {
            long idx = HistRangeIndexH(x, e, fe, le, nd, nbins);
            if (idx >= 0)
                acc[idx]++;
        }

        /// <summary>
        ///     NumPy's index of one float32 sample over a range: keep, <c>tmp_a - first_edge</c> in float32, <c>/ norm_denom *
        ///     n_equal_bins</c> in float64, <c>astype(np.intp)</c>, the end-of-range decrement, the fancy index's bounds, the
        ///     one-ULP corrections against the float32 edges.
        /// </summary>
        /// <param name="x">The sample.</param>
        /// <param name="e">The float32 edges.</param>
        /// <param name="fe">The lower edge rounded to float32.</param>
        /// <param name="le">The upper edge rounded to float32.</param>
        /// <param name="nd">The float64 norm_denom.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <returns>The bin index, or -1 for a sample outside the outer edges.</returns>
        /// <exception cref="IndexError">The index passes the edges.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long HistRangeIndexF(float x, float* e, float fe, float le, double nd, int nbins)
        {
            if (!(x >= fe && x <= le))
                return -1;
            long idx = HistRangeTruncate((double)(x - fe) / nd * nbins, nbins);
            if (x < e[idx]) idx--;
            if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
            return idx;
        }

        /// <summary>
        ///     NumPy's index of one float16 sample over a range: keep, <c>tmp_a - first_edge</c> as half arithmetic (float32,
        ///     rounded to float16), <c>/ norm_denom * n_equal_bins</c> in float64 (the float16 difference widens), then as
        ///     <see cref="HistRangeIndexF"/>.
        /// </summary>
        /// <param name="xh">The sample.</param>
        /// <param name="e">The float16 edges.</param>
        /// <param name="fe">The lower edge rounded to float16, widened to float32.</param>
        /// <param name="le">The upper edge rounded to float16, widened to float32.</param>
        /// <param name="nd">The float64 norm_denom.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <returns>The bin index, or -1 for a sample outside the outer edges.</returns>
        /// <exception cref="IndexError">The index passes the edges.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long HistRangeIndexH(Half xh, Half* e, float fe, float le, double nd, int nbins)
        {
            float x = (float)xh;
            if (!(x >= fe && x <= le))
                return -1;
            float diff = (float)(Half)(x - fe);
            long idx = HistRangeTruncate(diff / nd * nbins, nbins);
            if (x < (float)e[idx]) idx--;
            if (idx != nbins - 1 && x >= (float)e[idx + 1]) idx++;
            return idx;
        }

        /// <summary>
        ///     <c>f_indices.astype(np.intp)</c> as x86 computes it (INT64_MIN for a NaN or a value at or past 2^63; a kept
        ///     sample's quotient is never negative), then <c>indices[indices == n_equal_bins] -= 1</c>, then the bounds of the
        ///     fancy index <c>bin_edges[indices]</c>.
        /// </summary>
        /// <param name="f">The float64 quotient.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <returns>The index, in [0, nbins - 1].</returns>
        /// <exception cref="IndexError">An index past the edges: <c>index 3 is out of bounds for axis 0 with size 3</c>.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long HistRangeTruncate(double f, int nbins)
        {
            long idx = f < 9.2233720368547758E18 ? (long)f : long.MinValue;
            if (idx == nbins) idx--;
            // idx is >= 0 or INT64_MIN here: one unsigned comparison catches both refusals.
            if ((ulong)idx > (ulong)nbins)
                ThrowHistogramIndexOutOfEdges(idx, nbins);
            return idx;
        }

        /// <summary>NumPy's fancy-index refusal of an index past the <c>nbins + 1</c> edges (kept out of the hot loop).</summary>
        /// <param name="idx">The index.</param>
        /// <param name="nbins">The number of bins.</param>
        /// <exception cref="IndexError">Always.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowHistogramIndexOutOfEdges(long idx, int nbins) =>
            throw new IndexError($"index {idx} is out of bounds for axis 0 with size {(long)nbins + 1}");
    }
}
