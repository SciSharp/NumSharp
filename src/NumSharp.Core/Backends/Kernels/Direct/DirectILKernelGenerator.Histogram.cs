using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Histogram.cs — fused bin-index kernel for np.histogram.
//
// RESPONSIBILITY
//   The ONE novel hot loop of the histogram family: turning each sample into the
//   index of the equal-width bin it falls into. This is the exact port of the
//   fast-path index computation in NumPy's `histogram` (numpy/lib/_histograms_impl.py):
//
//       f_indices = ((tmp_a - first_edge) / norm_denom) * n_equal_bins
//       indices   = f_indices.astype(intp)          # truncate toward zero
//       indices[indices == n_equal_bins] -= 1        # value == last_edge
//       indices[tmp_a <  bin_edges[indices]]   -= 1   # ~1 ULP low  correction
//       indices[(tmp_a >= bin_edges[indices+1]) & (indices != nbins-1)] += 1  # ~1 ULP high
//
//   Every sample OUTSIDE [first_edge, last_edge] is mapped to the sentinel bin
//   `n_equal_bins` (NumPy drops these with a `keep` mask; we let np.bincount's
//   trailing sentinel bin collect them and slice it off — identical counts,
//   identical weighted-sum ORDER, and it reuses bincount's privatized scatter).
//
// WHY A KERNEL AND NOT COMPOSITION
//   NumPy computes the five lines above as ~8 separate vectorized passes over
//   full temp arrays. Fusing them into ONE scalar pass (the corrections read the
//   already-sorted `bin_edges`, so they cannot be vectorized cheaply) reads each
//   sample once and writes one int64 index — the whole reason np.histogram can
//   beat NumPy rather than merely match its per-pass work.
//
// COMPUTE DTYPE B (Half / Single / Double)
//   The arithmetic runs in the SAME dtype NumPy uses for `bin_edges` — float16
//   for float16 input, float32 for float32, float64 for everything else. That
//   dtype (B) is baked into the kernel body: `typeof(T)==` branches let the JIT
//   monomorphize one specialized loop per B (the WeightedSum idiom), with the
//   float16 path computing through float32 exactly as NumPy's npy_half ufuncs do
//   (each op = widen→op→narrow), which is what makes float16 histograms
//   bit-identical rather than merely close.
//
//   Samples are pre-cast to B by the caller, so the kernel reads one contiguous
//   B* buffer — no per-element T→B dispatch inside the hot loop.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Fused histogram bin-index kernel: maps each sample (pre-cast to the bin dtype B)
        ///     to its equal-width bin index, or to the sentinel bin <c>nbins</c> when out of range.
        /// </summary>
        /// <param name="samples">Contiguous buffer of samples in dtype B (<c>count</c> elements).</param>
        /// <param name="count">Number of samples.</param>
        /// <param name="binEdges">Contiguous buffer of <c>nbins+1</c> bin edges in dtype B.</param>
        /// <param name="first">Lower outer edge (<c>first_edge</c>), always passed as double.</param>
        /// <param name="last">Upper outer edge (<c>last_edge</c>), always passed as double.</param>
        /// <param name="nbins">Number of equal-width bins; the sentinel out-of-range bin is this value.</param>
        /// <param name="outIndices">Output buffer of <c>count</c> int64 bin indices in <c>[0, nbins]</c>.</param>
        public unsafe delegate void HistogramIndicesKernel(
            void* samples, long count, void* binEdges,
            double first, double last, int nbins, long* outIndices);

        private static readonly ConcurrentDictionary<NPTypeCode, HistogramIndicesKernel> _histIndicesCache = new();

        /// <summary>
        ///     Returns the cached fused bin-index kernel for the given compute dtype B.
        /// </summary>
        /// <param name="binDtype">
        ///     The bin/compute dtype — one of <see cref="NPTypeCode.Half"/>, <see cref="NPTypeCode.Single"/>
        ///     or <see cref="NPTypeCode.Double"/>. Any other value throws, because np.histogram resolves
        ///     the bin dtype to exactly these three float widths before reaching the kernel.
        /// </param>
        /// <returns>The cached <see cref="HistogramIndicesKernel"/> for <paramref name="binDtype"/>.</returns>
        /// <exception cref="NotSupportedException">The dtype is not one of the three float compute widths.</exception>
        public static HistogramIndicesKernel GetHistogramIndicesKernel(NPTypeCode binDtype) =>
            _histIndicesCache.GetOrAdd(binDtype, CreateHistogramIndicesKernel);

        private static unsafe HistogramIndicesKernel CreateHistogramIndicesKernel(NPTypeCode binDtype) =>
            binDtype switch
            {
                // One closure per B; the JIT compiles a specialized HistogramIndicesBody<T> for each,
                // and the dead typeof() branches inside fold away — the WeightedSum monomorphization idiom.
                NPTypeCode.Double => static (s, c, e, f, l, n, o) => HistogramIndicesBody<double>(s, c, e, f, l, n, o),
                NPTypeCode.Single => static (s, c, e, f, l, n, o) => HistogramIndicesBody<float>(s, c, e, f, l, n, o),
                NPTypeCode.Half   => static (s, c, e, f, l, n, o) => HistogramIndicesBody<Half>(s, c, e, f, l, n, o),
                _ => throw new NotSupportedException(
                    $"Histogram bin dtype must be Half, Single or Double (got {binDtype}).")
            };

        /// <summary>
        ///     The specialized fused index loop for one compute dtype. Not called directly — reached only
        ///     through <see cref="GetHistogramIndicesKernel"/>, which pins <typeparamref name="T"/> to
        ///     double/float/Half so exactly one <c>typeof</c> branch survives per instantiation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistogramIndicesBody<T>(
            void* samplesP, long count, void* edgesP,
            double first, double last, int nbins, long* outIdx) where T : unmanaged
        {
            // --- float64 path: int/bool/char/uint/int64/float64 inputs (the common case). ---
            if (typeof(T) == typeof(double))
            {
                double* s = (double*)samplesP;
                double* e = (double*)edgesP;
                double fe = first, le = last;
                double nd = le - fe; // norm_denom in double; no signed-integer overflow because both are doubles
                for (long i = 0; i < count; i++)
                {
                    double x = s[i];
                    // Out-of-range samples land in the sentinel bin (NumPy drops them via `keep`).
                    // The !(>= && <=) form also routes NaN here (NaN fails both comparisons).
                    if (!(x >= fe && x <= le)) { outIdx[i] = nbins; continue; }
                    double f = (x - fe) / nd * nbins;
                    long idx = (long)f; // truncate toward zero; f >= 0 here so this is floor
                    if (idx == nbins) idx--;                       // value == last_edge
                    if (x < e[idx]) idx--;                          // ~1 ULP low correction
                    if (idx != nbins - 1 && x >= e[idx + 1]) idx++; // ~1 ULP high correction
                    outIdx[i] = idx;
                }
                return;
            }

            // --- float32 path: float32 inputs, arithmetic in float32 to match NumPy's bin dtype. ---
            if (typeof(T) == typeof(float))
            {
                float* s = (float*)samplesP;
                float* e = (float*)edgesP;
                float fe = (float)first, le = (float)last;
                float nd = le - fe;
                for (long i = 0; i < count; i++)
                {
                    float x = s[i];
                    if (!(x >= fe && x <= le)) { outIdx[i] = nbins; continue; }
                    float f = (x - fe) / nd * nbins;
                    long idx = (long)f;
                    if (idx == nbins) idx--;
                    if (x < e[idx]) idx--;
                    if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
                    outIdx[i] = idx;
                }
                return;
            }

            // --- float16 path: each op widens to float32, computes, narrows back — the exact model
            //     NumPy's npy_half ufuncs use, so the ~1 ULP boundary decisions match bit-for-bit. ---
            if (typeof(T) == typeof(Half))
            {
                Half* s = (Half*)samplesP;
                Half* e = (Half*)edgesP;
                Half fe = (Half)first, le = (Half)last;
                Half nd = (Half)((float)le - (float)fe);
                float ndF = (float)nd;
                int nbF = nbins;
                for (long i = 0; i < count; i++)
                {
                    Half x = s[i];
                    float xf = (float)x;
                    if (!(xf >= (float)fe && xf <= (float)le)) { outIdx[i] = nbins; continue; }
                    Half t1 = (Half)(xf - (float)fe);       // subtract  (half)
                    Half t2 = (Half)((float)t1 / ndF);      // divide    (half)
                    Half fH = (Half)((float)t2 * nbF);      // multiply  (half)
                    long idx = (long)(float)fH;             // astype(intp) — truncate toward zero
                    if (idx == nbins) idx--;
                    if (xf < (float)e[idx]) idx--;
                    if (idx != nbins - 1 && xf >= (float)e[idx + 1]) idx++;
                    outIdx[i] = idx;
                }
                return;
            }

            throw new NotSupportedException($"HistogramIndicesBody: unsupported compute dtype {typeof(T)}.");
        }

        /// <summary>
        ///     Fused unweighted histogram count kernel: for each sample compute its bin index (as in
        ///     <see cref="HistogramIndicesKernel"/>) and increment the count directly — no int64 indices array
        ///     and no separate <c>bincount</c> pass. Out-of-range/NaN samples are skipped.
        /// </summary>
        /// <param name="samples">Contiguous samples in dtype B.</param>
        /// <param name="count">Number of samples.</param>
        /// <param name="binEdges">Contiguous <c>nbins+1</c> bin edges in dtype B.</param>
        /// <param name="first">Lower outer edge (double).</param>
        /// <param name="last">Upper outer edge (double).</param>
        /// <param name="nbins">Number of bins.</param>
        /// <param name="counts">Output int64 count buffer of length <paramref name="nbins"/> (pre-zeroed).</param>
        public unsafe delegate void HistogramCountKernel(
            void* samples, long count, void* binEdges,
            double first, double last, int nbins, long* counts);

        private static readonly ConcurrentDictionary<NPTypeCode, HistogramCountKernel> _histCountCache = new();

        /// <summary>Returns the cached fused count kernel for compute dtype B (Half/Single/Double).</summary>
        /// <param name="binDtype">The compute dtype B.</param>
        /// <returns>The cached <see cref="HistogramCountKernel"/>.</returns>
        /// <exception cref="NotSupportedException">The dtype is not a supported float compute width.</exception>
        public static HistogramCountKernel GetHistogramCountKernel(NPTypeCode binDtype) =>
            _histCountCache.GetOrAdd(binDtype, CreateHistogramCountKernel);

        private static unsafe HistogramCountKernel CreateHistogramCountKernel(NPTypeCode binDtype) =>
            binDtype switch
            {
                NPTypeCode.Double => static (s, c, e, f, l, n, o) => HistogramCountBody<double>(s, c, e, f, l, n, o),
                NPTypeCode.Single => static (s, c, e, f, l, n, o) => HistogramCountBody<float>(s, c, e, f, l, n, o),
                NPTypeCode.Half   => static (s, c, e, f, l, n, o) => HistogramCountBody<Half>(s, c, e, f, l, n, o),
                _ => throw new NotSupportedException(
                    $"Histogram bin dtype must be Half, Single or Double (got {binDtype}).")
            };

        // Privatized-accumulator tuning, mirroring np.bincount: the counting scatter `counts[idx]++`
        // stalls on store-to-load forwarding when the SAME bin repeats (the histogram norm: few bins,
        // many samples). Four round-robin accumulators break that carried dependency; a final SIMD
        // merge folds them back. Integer add is associative, so this is bit-identical to the sequential
        // scatter. Gated: privatization only pays when bins fit L2 and there are enough samples to
        // collide — otherwise the extra zeroed buffers + merge are pure overhead.
        private const int HistCountAccumulators = 4;
        private const long HistCountPrivMaxBins = 2048;
        private const long HistCountPrivMinLen = 4096;

        /// <summary>The specialized fused count loop for one compute dtype (double/float/Half).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistogramCountBody<T>(
            void* samplesP, long count, void* edgesP,
            double first, double last, int nbins, long* counts) where T : unmanaged
        {
            // Decide whether to privatize (same gate as bincount): small bin range + enough samples.
            bool privatize = nbins <= HistCountPrivMaxBins && count >= HistCountPrivMinLen && count >= (long)nbins * 4;
            long* a1 = null, a2 = null, a3 = null;
            if (privatize)
            {
                // Three extra zeroed accumulators in one unmanaged block (never a managed LOH array).
                long* ex = (long*)System.Runtime.InteropServices.NativeMemory.AllocZeroed((nuint)((long)nbins * 3 * sizeof(long)));
                a1 = ex; a2 = ex + nbins; a3 = ex + 2 * nbins;
            }
            try
            {
                if (typeof(T) == typeof(double))
                {
                    double* s = (double*)samplesP; double* e = (double*)edgesP;
                    double fe = first, le = last, nd = le - fe;
                    if (privatize)
                    {
                        long i = 0, lim = count - (count & 3);
                        for (; i < lim; i += 4)
                        {
                            HistCountScatterD(counts, s[i],   e, fe, le, nd, nbins);
                            HistCountScatterD(a1,     s[i+1], e, fe, le, nd, nbins);
                            HistCountScatterD(a2,     s[i+2], e, fe, le, nd, nbins);
                            HistCountScatterD(a3,     s[i+3], e, fe, le, nd, nbins);
                        }
                        for (; i < count; i++) HistCountScatterD(counts, s[i], e, fe, le, nd, nbins);
                    }
                    else
                        for (long i = 0; i < count; i++) HistCountScatterD(counts, s[i], e, fe, le, nd, nbins);
                }
                else if (typeof(T) == typeof(float))
                {
                    float* s = (float*)samplesP; float* e = (float*)edgesP;
                    float fe = (float)first, le = (float)last, nd = le - fe;
                    if (privatize)
                    {
                        long i = 0, lim = count - (count & 3);
                        for (; i < lim; i += 4)
                        {
                            HistCountScatterF(counts, s[i],   e, fe, le, nd, nbins);
                            HistCountScatterF(a1,     s[i+1], e, fe, le, nd, nbins);
                            HistCountScatterF(a2,     s[i+2], e, fe, le, nd, nbins);
                            HistCountScatterF(a3,     s[i+3], e, fe, le, nd, nbins);
                        }
                        for (; i < count; i++) HistCountScatterF(counts, s[i], e, fe, le, nd, nbins);
                    }
                    else
                        for (long i = 0; i < count; i++) HistCountScatterF(counts, s[i], e, fe, le, nd, nbins);
                }
                else if (typeof(T) == typeof(Half))
                {
                    Half* s = (Half*)samplesP; Half* e = (Half*)edgesP;
                    Half fe = (Half)first, le = (Half)last;
                    float ndF = (float)((Half)((float)le - (float)fe));
                    for (long i = 0; i < count; i++) HistCountScatterH(counts, s[i], e, fe, le, ndF, nbins);
                }
                else
                    throw new NotSupportedException($"HistogramCountBody: unsupported dtype {typeof(T)}.");

                if (privatize)
                {
                    // SIMD merge of the three extra accumulators back into the primary count buffer.
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
                if (privatize) System.Runtime.InteropServices.NativeMemory.Free(a1);
            }
        }

        /// <summary>Compute one double sample's bin index and increment its accumulator slot (skip if out of range/NaN).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HistCountScatterD(long* acc, double x, double* e, double fe, double le, double nd, int nbins)
        {
            if (!(x >= fe && x <= le)) return;
            double f = (x - fe) / nd * nbins;
            long idx = (long)f;
            if (idx == nbins) idx--;
            if (x < e[idx]) idx--;
            if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
            acc[idx]++;
        }

        /// <summary>Compute one float sample's bin index and increment its accumulator slot.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HistCountScatterF(long* acc, float x, float* e, float fe, float le, float nd, int nbins)
        {
            if (!(x >= fe && x <= le)) return;
            float f = (x - fe) / nd * nbins;
            long idx = (long)f;
            if (idx == nbins) idx--;
            if (x < e[idx]) idx--;
            if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
            acc[idx]++;
        }

        /// <summary>
        ///     Fused weighted histogram scatter for equal-width bins: for each in-range sample, add its (float64)
        ///     weight to the bin's running float64 total. Runs over ONE block; the caller casts the total to the
        ///     weight dtype and folds it into the result, replicating NumPy's per-block
        ///     <c>n += bincount(block, w).astype(ntype)</c> — no int64 indices array, no separate bincount pass.
        /// </summary>
        /// <param name="samples">Contiguous block of samples in dtype B.</param>
        /// <param name="weights">Contiguous block of float64 weights (same length).</param>
        /// <param name="count">Block length.</param>
        /// <param name="binEdges">Contiguous <c>nbins+1</c> bin edges in dtype B.</param>
        /// <param name="first">Lower outer edge (double).</param>
        /// <param name="last">Upper outer edge (double).</param>
        /// <param name="nbins">Number of bins.</param>
        /// <param name="totals">Output float64 per-bin totals of length <paramref name="nbins"/> (pre-zeroed by the caller).</param>
        public unsafe delegate void HistogramWeightedKernel(
            void* samples, void* weights, long count, void* binEdges,
            double first, double last, int nbins, double* totals);

        private static readonly ConcurrentDictionary<NPTypeCode, HistogramWeightedKernel> _histWeightedCache = new();

        /// <summary>Returns the cached fused weighted-scatter kernel for compute dtype B (Half/Single/Double).</summary>
        /// <param name="binDtype">The compute dtype B.</param>
        /// <returns>The cached <see cref="HistogramWeightedKernel"/>.</returns>
        /// <exception cref="NotSupportedException">The dtype is not a supported float compute width.</exception>
        public static HistogramWeightedKernel GetHistogramWeightedKernel(NPTypeCode binDtype) =>
            _histWeightedCache.GetOrAdd(binDtype, CreateHistogramWeightedKernel);

        private static unsafe HistogramWeightedKernel CreateHistogramWeightedKernel(NPTypeCode binDtype) =>
            binDtype switch
            {
                NPTypeCode.Double => static (s, w, c, e, f, l, n, o) => HistogramWeightedBody<double>(s, w, c, e, f, l, n, o),
                NPTypeCode.Single => static (s, w, c, e, f, l, n, o) => HistogramWeightedBody<float>(s, w, c, e, f, l, n, o),
                NPTypeCode.Half   => static (s, w, c, e, f, l, n, o) => HistogramWeightedBody<Half>(s, w, c, e, f, l, n, o),
                _ => throw new NotSupportedException($"Histogram bin dtype must be Half, Single or Double (got {binDtype}).")
            };

        /// <summary>The specialized fused weighted-scatter loop for one compute dtype (sequential — float add is order-sensitive).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistogramWeightedBody<T>(
            void* samplesP, void* weightsP, long count, void* edgesP,
            double first, double last, int nbins, double* totals) where T : unmanaged
        {
            double* wp = (double*)weightsP;
            if (typeof(T) == typeof(double))
            {
                double* s = (double*)samplesP; double* e = (double*)edgesP;
                double fe = first, le = last, nd = le - fe;
                for (long i = 0; i < count; i++)
                {
                    double x = s[i];
                    if (!(x >= fe && x <= le)) continue;
                    double f = (x - fe) / nd * nbins;
                    long idx = (long)f;
                    if (idx == nbins) idx--;
                    if (x < e[idx]) idx--;
                    if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
                    totals[idx] += wp[i];
                }
            }
            else if (typeof(T) == typeof(float))
            {
                float* s = (float*)samplesP; float* e = (float*)edgesP;
                float fe = (float)first, le = (float)last, nd = le - fe;
                for (long i = 0; i < count; i++)
                {
                    float x = s[i];
                    if (!(x >= fe && x <= le)) continue;
                    float f = (x - fe) / nd * nbins;
                    long idx = (long)f;
                    if (idx == nbins) idx--;
                    if (x < e[idx]) idx--;
                    if (idx != nbins - 1 && x >= e[idx + 1]) idx++;
                    totals[idx] += wp[i];
                }
            }
            else if (typeof(T) == typeof(Half))
            {
                Half* s = (Half*)samplesP; Half* e = (Half*)edgesP;
                Half fe = (Half)first, le = (Half)last;
                float ndF = (float)((Half)((float)le - (float)fe));
                for (long i = 0; i < count; i++)
                {
                    Half xh = s[i]; float xf = (float)xh;
                    if (!(xf >= (float)fe && xf <= (float)le)) continue;
                    Half t1 = (Half)(xf - (float)fe);
                    Half t2 = (Half)((float)t1 / ndF);
                    Half fH = (Half)((float)t2 * nbins);
                    long idx = (long)(float)fH;
                    if (idx == nbins) idx--;
                    if (xf < (float)e[idx]) idx--;
                    if (idx != nbins - 1 && xf >= (float)e[idx + 1]) idx++;
                    totals[idx] += wp[i];
                }
            }
            else
                throw new NotSupportedException($"HistogramWeightedBody: unsupported dtype {typeof(T)}.");
        }

        /// <summary>Compute one Half sample's bin index (via float32, npy_half model) and increment its slot.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void HistCountScatterH(long* acc, Half xh, Half* e, Half fe, Half le, float ndF, int nbins)
        {
            float xf = (float)xh;
            if (!(xf >= (float)fe && xf <= (float)le)) return;
            Half t1 = (Half)(xf - (float)fe);
            Half t2 = (Half)((float)t1 / ndF);
            Half fH = (Half)((float)t2 * nbins);
            long idx = (long)(float)fH;
            if (idx == nbins) idx--;
            if (xf < (float)e[idx]) idx--;
            if (idx != nbins - 1 && xf >= (float)e[idx + 1]) idx++;
            acc[idx]++;
        }

        // =====================================================================
        //  Non-uniform (arbitrary edges) fused search+count
        // =====================================================================

        /// <summary>
        ///     Fused non-uniform histogram count kernel: for each sample, binary-search its bin among arbitrary
        ///     (monotonic) edges and increment the count — replacing a whole-array <c>searchsorted</c> + comparison
        ///     + <c>astype</c> + <c>bincount</c> chain (5 passes and several full-size temps) with ONE pass.
        /// </summary>
        /// <param name="samples">Contiguous samples in the compare dtype C.</param>
        /// <param name="count">Number of samples.</param>
        /// <param name="edges">Contiguous <c>nbins+1</c> monotonic edges in the compare dtype C.</param>
        /// <param name="nbins">Number of bins (<c>edges.Length - 1</c>).</param>
        /// <param name="counts">Output int64 count buffer of length <paramref name="nbins"/> (pre-zeroed).</param>
        public unsafe delegate void HistogramSearchCountKernel(void* samples, long count, void* edges, int nbins, long* counts);

        private static readonly ConcurrentDictionary<NPTypeCode, HistogramSearchCountKernel> _histSearchCache = new();

        /// <summary>Returns the cached fused non-uniform search+count kernel for the given compare dtype.</summary>
        /// <param name="compareDtype">The dtype both samples and edges are cast to (their common type).</param>
        /// <returns>The cached kernel, or null when the dtype has no ordering (Complex — caller composes instead).</returns>
        public static HistogramSearchCountKernel GetHistogramSearchCountKernel(NPTypeCode compareDtype) =>
            _histSearchCache.GetOrAdd(compareDtype, CreateHistogramSearchCountKernel);

        private static unsafe HistogramSearchCountKernel CreateHistogramSearchCountKernel(NPTypeCode c) => c switch
        {
            NPTypeCode.Double  => static (s, n, e, nb, o) => HistSearchCountF64(s, n, e, nb, o),
            NPTypeCode.Single  => static (s, n, e, nb, o) => HistSearchCountF32(s, n, e, nb, o),
            NPTypeCode.Half    => static (s, n, e, nb, o) => HistSearchCountGeneric<Half>(s, n, e, nb, o),
            NPTypeCode.Byte    => static (s, n, e, nb, o) => HistSearchCountGeneric<byte>(s, n, e, nb, o),
            NPTypeCode.SByte   => static (s, n, e, nb, o) => HistSearchCountGeneric<sbyte>(s, n, e, nb, o),
            NPTypeCode.Int16   => static (s, n, e, nb, o) => HistSearchCountGeneric<short>(s, n, e, nb, o),
            NPTypeCode.UInt16  => static (s, n, e, nb, o) => HistSearchCountGeneric<ushort>(s, n, e, nb, o),
            NPTypeCode.Int32   => static (s, n, e, nb, o) => HistSearchCountGeneric<int>(s, n, e, nb, o),
            NPTypeCode.UInt32  => static (s, n, e, nb, o) => HistSearchCountGeneric<uint>(s, n, e, nb, o),
            NPTypeCode.Int64   => static (s, n, e, nb, o) => HistSearchCountGeneric<long>(s, n, e, nb, o),
            NPTypeCode.UInt64  => static (s, n, e, nb, o) => HistSearchCountGeneric<ulong>(s, n, e, nb, o),
            NPTypeCode.Char    => static (s, n, e, nb, o) => HistSearchCountGeneric<char>(s, n, e, nb, o),
            NPTypeCode.Boolean => static (s, n, e, nb, o) => HistSearchCountGeneric<byte>(s, n, e, nb, o),
            NPTypeCode.Decimal => static (s, n, e, nb, o) => HistSearchCountGeneric<decimal>(s, n, e, nb, o),
            _ => null // Complex has no ordering — caller falls back to the composed cumulative path.
        };

        // The 'right'-side binary search returns p = number of edges <= x (so a value equal to an interior edge
        // falls in the higher bin, matching NumPy's half-open [l, r) bins). A value equal to the LAST edge is
        // pulled back one (the inclusive right edge). bin = p-1 is counted when it lands in [0, nbins). Because
        // the search uses `x < edges[mid]` (never a CompareTo), a NaN sample fails every branch and walks off the
        // top end → skipped as an outlier, exactly as NumPy drops it.
        // Below this many edges the branchless SIMD "count edges <= x" beats the binary search — whose
        // `x < e[mid]` compare mispredicts on every step for random data (~15-cycle penalty), so it stays
        // slower than an O(k/lanes) vector scan up to a surprisingly large k. Measured crossover on this host
        // is ~250 edges (binary is 3-6x SLOWER at k≤200); above ~256 the vector scan finally costs more.
        // Beyond it the binary search runs — still correct, just the documented slow tail for many-bin
        // non-uniform edges (bounded by the absence of a fast SIMD sort, the "sort family" perf class).
        private const int HistSearchSimdMaxEdges = 256;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchCountF64(void* samplesP, long count, void* edgesP, int nbins, long* counts)
        {
            double* s = (double*)samplesP; double* e = (double*)edgesP;
            int k = nbins + 1;
            double last = e[nbins];

            if (Vector256.IsHardwareAccelerated && k <= HistSearchSimdMaxEdges)
            {
                // Branchless: p = #edges <= x via vector compares + popcount. Tail lanes padded with NaN
                // (NaN <= anything is false, so padding is never counted — and it keeps +inf samples correct).
                int nvec = (k + 3) >> 2;                 // Vector256<double> = 4 lanes
                Vector256<double>* ev = stackalloc Vector256<double>[nvec];
                for (int v = 0; v < nvec; v++)
                {
                    double l0 = 4*v+0 < k ? e[4*v+0] : double.NaN;
                    double l1 = 4*v+1 < k ? e[4*v+1] : double.NaN;
                    double l2 = 4*v+2 < k ? e[4*v+2] : double.NaN;
                    double l3 = 4*v+3 < k ? e[4*v+3] : double.NaN;
                    ev[v] = Vector256.Create(l0, l1, l2, l3);
                }
                for (long i = 0; i < count; i++)
                {
                    var xv = Vector256.Create(s[i]);
                    int p = 0;
                    for (int v = 0; v < nvec; v++)
                        p += System.Numerics.BitOperations.PopCount(
                            Vector256.LessThanOrEqual(ev[v], xv).ExtractMostSignificantBits());
                    if (s[i] == last) p--;
                    int bin = p - 1;
                    if ((uint)bin < (uint)nbins) counts[bin]++;
                }
                return;
            }

            int hiInit = k;
            for (long i = 0; i < count; i++)
            {
                double x = s[i];
                int lo = 0, hi = hiInit;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x < e[mid]) hi = mid; else lo = mid + 1; }
                if (x == last) lo--;
                int bin = lo - 1;
                if ((uint)bin < (uint)nbins) counts[bin]++;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchCountF32(void* samplesP, long count, void* edgesP, int nbins, long* counts)
        {
            float* s = (float*)samplesP; float* e = (float*)edgesP;
            int k = nbins + 1;
            float last = e[nbins];

            if (Vector256.IsHardwareAccelerated && k <= HistSearchSimdMaxEdges)
            {
                int nvec = (k + 7) >> 3;                 // Vector256<float> = 8 lanes
                Vector256<float>* ev = stackalloc Vector256<float>[nvec];
                for (int v = 0; v < nvec; v++)
                {
                    float* baseE = e + 8 * v;
                    float l0 = 8*v+0 < k ? baseE[0] : float.NaN, l1 = 8*v+1 < k ? baseE[1] : float.NaN;
                    float l2 = 8*v+2 < k ? baseE[2] : float.NaN, l3 = 8*v+3 < k ? baseE[3] : float.NaN;
                    float l4 = 8*v+4 < k ? baseE[4] : float.NaN, l5 = 8*v+5 < k ? baseE[5] : float.NaN;
                    float l6 = 8*v+6 < k ? baseE[6] : float.NaN, l7 = 8*v+7 < k ? baseE[7] : float.NaN;
                    ev[v] = Vector256.Create(l0, l1, l2, l3, l4, l5, l6, l7);
                }
                for (long i = 0; i < count; i++)
                {
                    var xv = Vector256.Create(s[i]);
                    int p = 0;
                    for (int v = 0; v < nvec; v++)
                        p += System.Numerics.BitOperations.PopCount(
                            Vector256.LessThanOrEqual(ev[v], xv).ExtractMostSignificantBits());
                    if (s[i] == last) p--;
                    int bin = p - 1;
                    if ((uint)bin < (uint)nbins) counts[bin]++;
                }
                return;
            }

            int hiInit = k;
            for (long i = 0; i < count; i++)
            {
                float x = s[i];
                int lo = 0, hi = hiInit;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x < e[mid]) hi = mid; else lo = mid + 1; }
                if (x == last) lo--;
                int bin = lo - 1;
                if ((uint)bin < (uint)nbins) counts[bin]++;
            }
        }

        /// <summary>
        ///     Generic non-uniform search+count for every ordered dtype except double/float (which have their own
        ///     NaN-correct specializations). Uses <see cref="IComparable{T}"/> — safe here because the only float
        ///     type routed through it is <see cref="Half"/> (whose NaN is a rare edge value), and the integer/decimal
        ///     types have no NaN at all.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchCountGeneric<C>(void* samplesP, long count, void* edgesP, int nbins, long* counts)
            where C : unmanaged, IComparable<C>
        {
            C* s = (C*)samplesP; C* e = (C*)edgesP;
            C last = e[nbins];
            int hiInit = nbins + 1;
            for (long i = 0; i < count; i++)
            {
                C x = s[i];
                int lo = 0, hi = hiInit;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x.CompareTo(e[mid]) < 0) hi = mid; else lo = mid + 1; }
                if (x.CompareTo(last) == 0) lo--;
                int bin = lo - 1;
                if ((uint)bin < (uint)nbins) counts[bin]++;
            }
        }

        // =====================================================================
        //  Non-uniform search INDEX (for histogramdd's per-dimension binning)
        // =====================================================================

        /// <summary>
        ///     Fused per-dimension bin-index kernel for histogramdd: for each sample writes
        ///     <c>searchsorted(edges, x, 'right')</c> with a value on the last edge pulled one bin left — the exact
        ///     <c>Ncount</c> histogramdd needs — using the same branchless SIMD/binary search as the count kernels.
        ///     Replaces the general (slow) <c>np.searchsorted</c> call in the D-dimensional hot loop.
        /// </summary>
        /// <param name="samples">Contiguous samples in the compare dtype C.</param>
        /// <param name="count">Number of samples.</param>
        /// <param name="edges">Contiguous <paramref name="numEdges"/> monotonic edges in dtype C.</param>
        /// <param name="numEdges">Number of edges (<c>len(edges)</c>).</param>
        /// <param name="outIdx">Output int64 bin numbers of length <paramref name="count"/> (range <c>[0, numEdges]</c>).</param>
        public unsafe delegate void HistogramSearchIndexKernel(void* samples, long count, void* edges, int numEdges, long* outIdx);

        private static readonly ConcurrentDictionary<NPTypeCode, HistogramSearchIndexKernel> _histSearchIdxCache = new();

        /// <summary>Returns the cached fused search-index kernel for the given compare dtype (null for Complex).</summary>
        /// <param name="compareDtype">The dtype samples and edges are cast to (their common type).</param>
        /// <returns>The cached kernel, or null when the dtype has no ordering.</returns>
        public static HistogramSearchIndexKernel GetHistogramSearchIndexKernel(NPTypeCode compareDtype) =>
            _histSearchIdxCache.GetOrAdd(compareDtype, CreateHistogramSearchIndexKernel);

        private static unsafe HistogramSearchIndexKernel CreateHistogramSearchIndexKernel(NPTypeCode c) => c switch
        {
            NPTypeCode.Double  => static (s, n, e, k, o) => HistSearchIndexF64(s, n, e, k, o),
            NPTypeCode.Single  => static (s, n, e, k, o) => HistSearchIndexF32(s, n, e, k, o),
            NPTypeCode.Half    => static (s, n, e, k, o) => HistSearchIndexGeneric<Half>(s, n, e, k, o),
            NPTypeCode.Byte    => static (s, n, e, k, o) => HistSearchIndexGeneric<byte>(s, n, e, k, o),
            NPTypeCode.SByte   => static (s, n, e, k, o) => HistSearchIndexGeneric<sbyte>(s, n, e, k, o),
            NPTypeCode.Int16   => static (s, n, e, k, o) => HistSearchIndexGeneric<short>(s, n, e, k, o),
            NPTypeCode.UInt16  => static (s, n, e, k, o) => HistSearchIndexGeneric<ushort>(s, n, e, k, o),
            NPTypeCode.Int32   => static (s, n, e, k, o) => HistSearchIndexGeneric<int>(s, n, e, k, o),
            NPTypeCode.UInt32  => static (s, n, e, k, o) => HistSearchIndexGeneric<uint>(s, n, e, k, o),
            NPTypeCode.Int64   => static (s, n, e, k, o) => HistSearchIndexGeneric<long>(s, n, e, k, o),
            NPTypeCode.UInt64  => static (s, n, e, k, o) => HistSearchIndexGeneric<ulong>(s, n, e, k, o),
            NPTypeCode.Char    => static (s, n, e, k, o) => HistSearchIndexGeneric<char>(s, n, e, k, o),
            NPTypeCode.Boolean => static (s, n, e, k, o) => HistSearchIndexGeneric<byte>(s, n, e, k, o),
            NPTypeCode.Decimal => static (s, n, e, k, o) => HistSearchIndexGeneric<decimal>(s, n, e, k, o),
            _ => null
        };

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchIndexF64(void* samplesP, long count, void* edgesP, int numEdges, long* outIdx)
        {
            double* s = (double*)samplesP; double* e = (double*)edgesP;
            double last = e[numEdges - 1];
            if (Vector256.IsHardwareAccelerated && numEdges <= HistSearchSimdMaxEdges)
            {
                int nvec = (numEdges + 3) >> 2;
                Vector256<double>* ev = stackalloc Vector256<double>[nvec];
                for (int v = 0; v < nvec; v++)
                    ev[v] = Vector256.Create(
                        4*v+0 < numEdges ? e[4*v+0] : double.NaN, 4*v+1 < numEdges ? e[4*v+1] : double.NaN,
                        4*v+2 < numEdges ? e[4*v+2] : double.NaN, 4*v+3 < numEdges ? e[4*v+3] : double.NaN);
                for (long i = 0; i < count; i++)
                {
                    var xv = Vector256.Create(s[i]);
                    int p = 0;
                    for (int v = 0; v < nvec; v++)
                        p += System.Numerics.BitOperations.PopCount(Vector256.LessThanOrEqual(ev[v], xv).ExtractMostSignificantBits());
                    if (s[i] == last) p--;
                    outIdx[i] = p;
                }
                return;
            }
            for (long i = 0; i < count; i++)
            {
                double x = s[i];
                int lo = 0, hi = numEdges;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x < e[mid]) hi = mid; else lo = mid + 1; }
                if (x == last) lo--;
                outIdx[i] = lo;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchIndexF32(void* samplesP, long count, void* edgesP, int numEdges, long* outIdx)
        {
            float* s = (float*)samplesP; float* e = (float*)edgesP;
            float last = e[numEdges - 1];
            if (Vector256.IsHardwareAccelerated && numEdges <= HistSearchSimdMaxEdges)
            {
                int nvec = (numEdges + 7) >> 3;
                Vector256<float>* ev = stackalloc Vector256<float>[nvec];
                for (int v = 0; v < nvec; v++)
                {
                    float* b = e + 8 * v;
                    ev[v] = Vector256.Create(
                        8*v+0 < numEdges ? b[0] : float.NaN, 8*v+1 < numEdges ? b[1] : float.NaN,
                        8*v+2 < numEdges ? b[2] : float.NaN, 8*v+3 < numEdges ? b[3] : float.NaN,
                        8*v+4 < numEdges ? b[4] : float.NaN, 8*v+5 < numEdges ? b[5] : float.NaN,
                        8*v+6 < numEdges ? b[6] : float.NaN, 8*v+7 < numEdges ? b[7] : float.NaN);
                }
                for (long i = 0; i < count; i++)
                {
                    var xv = Vector256.Create(s[i]);
                    int p = 0;
                    for (int v = 0; v < nvec; v++)
                        p += System.Numerics.BitOperations.PopCount(Vector256.LessThanOrEqual(ev[v], xv).ExtractMostSignificantBits());
                    if (s[i] == last) p--;
                    outIdx[i] = p;
                }
                return;
            }
            for (long i = 0; i < count; i++)
            {
                float x = s[i];
                int lo = 0, hi = numEdges;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x < e[mid]) hi = mid; else lo = mid + 1; }
                if (x == last) lo--;
                outIdx[i] = lo;
            }
        }

        /// <summary>Generic ordered-dtype search-index (integers/decimal/Half) via <see cref="IComparable{T}"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void HistSearchIndexGeneric<C>(void* samplesP, long count, void* edgesP, int numEdges, long* outIdx)
            where C : unmanaged, IComparable<C>
        {
            C* s = (C*)samplesP; C* e = (C*)edgesP;
            C last = e[numEdges - 1];
            for (long i = 0; i < count; i++)
            {
                C x = s[i];
                int lo = 0, hi = numEdges;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (x.CompareTo(e[mid]) < 0) hi = mid; else lo = mid + 1; }
                if (x.CompareTo(last) == 0) lo--;
                outIdx[i] = lo;
            }
        }
    }
}
