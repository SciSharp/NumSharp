using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp.Backends;
using SRCS = System.Runtime.CompilerServices;

namespace NumSharp
{
    public partial class NDArray
    {
        // Shared sliding multiply-accumulate engine for np.correlate and np.convolve.
        //
        // This is a direct port of NumPy's _pyarray_correlate (numpy/_core/src/multiarray/
        // multiarraymodule.c) + small_correlate (arraytypes.c.src): given a data array `a`
        // (length n1) and a kernel `k` (length n2, with n1 >= n2), read the kernel FORWARD and
        // compute the mode-length output as three contiguous regions —
        //
        //   left ramp  : outputs [0, nLeft)          — partial overlap, dot length grows
        //   middle     : outputs [nLeft, nLeft+mid)  — full overlap, uniform length n2
        //   right ramp : outputs [nLeft+mid, outLen) — partial overlap, dot length shrinks
        //
        // where mid = n1 - n2 + 1 (the fully-overlapping positions) and nLeft/nRight/outLen come
        // from the requested mode. This is exactly NumPy's region split: small_correlate handles
        // the middle, the scalar `dot` handles the ramps.
        //
        // np.correlate reads `k` forward (= NumPy correlate2, whose engine reads the kernel
        // forward and conjugates v beforehand for complex). np.convolve is
        // correlate(a, v[::-1]) — it reverses v into `k` before calling here, so the kernel is
        // still read forward. Both accumulate a-index increasing / kernel-index increasing, which
        // is the SAME per-output accumulation order NumPy uses, so the values are bit-identical to
        // NumPy for the small-kernel regime (see the ULP notes in np.correlate.cs).

        internal enum SlidingMode { Valid = 0, Same = 1, Full = 2 }

        /// <summary>
        /// Parse a convolve/correlate mode string exactly like NumPy's correlatemode_parser
        /// (numpy/_core/src/multiarray/conversion_utils.c): case-SENSITIVE exact match, with
        /// NumPy's two distinct near-miss messages reproduced verbatim.
        /// </summary>
        internal static SlidingMode ParseSlidingMode(string mode)
        {
            switch (mode)
            {
                case "valid": return SlidingMode.Valid;
                case "same": return SlidingMode.Same;
                case "full": return SlidingMode.Full;
            }

            // A near-miss (right first letter, wrong case/spelling) gets NumPy's parser message;
            // anything else gets the string-converter message that quotes the input.
            char c0 = mode.Length > 0 ? mode[0] : '\0';
            if (c0 == 'v' || c0 == 'V' || c0 == 's' || c0 == 'S' || c0 == 'f' || c0 == 'F')
                throw new ArgumentException("Use one of 'valid', 'same', or 'full' for convolve/correlate mode", nameof(mode));
            throw new ArgumentException($"mode must be one of 'valid', 'same', or 'full' (got '{mode}')", nameof(mode));
        }

        /// <summary>
        /// Core sliding multiply-accumulate. <paramref name="data"/> and <paramref name="kernel"/>
        /// MUST be contiguous, offset-0, of dtype <paramref name="retType"/>, with
        /// <c>data.size &gt;= kernel.size</c>. The kernel is read forward. Returns a fresh,
        /// C-contiguous, owning result of the mode-appropriate length.
        /// </summary>
        /// <param name="data">The data operand (contiguous, offset 0, of <paramref name="retType"/>).</param>
        /// <param name="kernel">The kernel operand, read forward (contiguous, offset 0, of <paramref name="retType"/>).</param>
        /// <param name="retType">The dtype of both operands and of the result.</param>
        /// <param name="mode">Valid, same or full.</param>
        /// <param name="complexDotViaBlas">Whether NumPy's complex dotfunc reaches cblas (both operand strides positive as
        ///     its dot sees them — <see cref="DotOperandBlasable"/>); false runs CDOUBLE_dot's plain loop instead.</param>
        /// <returns>The correlation.</returns>
        internal static NDArray SlidingCorrelate(NDArray data, NDArray kernel, NPTypeCode retType, SlidingMode mode, bool complexDotViaBlas = true)
        {
            long n1 = data.size;
            long n2 = kernel.size;

            // No zero-fill: the three regions (left ramp + middle + right ramp) write EVERY one of
            // the outLen positions exactly once (nLeft + mid + nRight == outLen), so a zeroing pass
            // would only be redundant writes on the memory-bound common case.
            var result = new NDArray(retType, Shape.Vector(SlidingOutputLength(n1, n2, mode)), false);
            unsafe
            {
                SlidingCorrelateInto((void*)data.Address, n1, (void*)kernel.Address, n2, (void*)result.Address, retType, mode,
                    data.TensorEngine.Blas as ISlidingDotBackend, complexDotViaBlas);
            }
            return result;
        }

        /// <summary>
        ///     Whether one operand of NumPy's <c>_pyarray_correlate</c> reaches its dotfunc with a stride cblas accepts
        ///     (<c>blas_stride</c>: positive and a multiple of the itemsize). <c>PyArray_Correlate(2)</c> takes each operand
        ///     through <c>PyArray_FromAny(op, dtype, 1, 1, NPY_ARRAY_DEFAULT)</c>, which COPIES — into a fresh, positive
        ///     stride — an operand of another dtype or one that is not C-contiguous; but an array of ONE element is always
        ///     C-contiguous whatever its stride, so it is passed through as it is, and np.convolve's kernel is <c>v[::-1]</c>,
        ///     whose stride is the negation of <c>v</c>'s. Only such a one-element operand can therefore be unblasable.
        /// </summary>
        /// <param name="x">The operand as the caller received it.</param>
        /// <param name="retType">The dtype NumPy converts it to.</param>
        /// <param name="reversed">The operand is np.convolve's kernel, which NumPy reverses before the conversion.</param>
        /// <returns>False when NumPy's dot sees a non-positive stride (a one-element operand of the final dtype whose
        ///     stride, after the reversal, is not positive).</returns>
        /// <remarks>Only the complex dotfunc behaves differently on the two paths (cblas' zdotu builds its result with C99
        ///     complex arithmetic, which turns a non-finite imaginary part's <c>im*0</c> into a NaN real part — see
        ///     <see cref="ZdotuResult"/>); a real dot of the single term such an operand allows is identical on both.</remarks>
        internal static bool DotOperandBlasable(NDArray x, NPTypeCode retType, bool reversed)
        {
            // Another dtype is a cast (a fresh array), more than one element a copy unless already C-contiguous — both a
            // positive stride.
            if (x.GetTypeCode != retType || x.size != 1)
                return true;
            long stride = x.Shape.strides.Length == 0 ? 1 : x.Shape.strides[0];
            return (reversed ? -stride : stride) > 0;
        }

        /// <summary>
        ///     The output length of a sliding correlate / convolve of an <paramref name="n1"/>-element data array with an
        ///     <paramref name="n2"/>-element kernel (<c>n1 &gt;= n2 &gt;= 1</c>): <c>n1 - n2 + 1</c> ('valid'), <c>n1</c>
        ///     ('same'), <c>n1 + n2 - 1</c> ('full').
        /// </summary>
        /// <param name="n1">Data length.</param>
        /// <param name="n2">Kernel length.</param>
        /// <param name="mode">The mode.</param>
        /// <returns>The number of output elements.</returns>
        internal static long SlidingOutputLength(long n1, long n2, SlidingMode mode) => mode switch
        {
            SlidingMode.Valid => n1 - n2 + 1,
            SlidingMode.Same => n1,
            _ => n1 + n2 - 1,
        };

        /// <summary>
        ///     The pointer-level core of <see cref="SlidingCorrelate"/>: NumPy's <c>_pyarray_correlate</c> over a contiguous
        ///     data array and a contiguous kernel read FORWARD, into a caller-owned output — so a caller holding its operands
        ///     in raw memory (numpy.polynomial's series arena, whose <c>np.convolve</c> calls build their reversed kernels
        ///     there) runs exactly the arithmetic <c>np.correlate</c> / <c>np.convolve</c> run, the byte-parity BLAS route
        ///     included.
        /// </summary>
        /// <param name="a">Data element 0 (<paramref name="n1"/> contiguous elements of <paramref name="retType"/>).</param>
        /// <param name="n1">Data length.</param>
        /// <param name="k">Kernel element 0 (<paramref name="n2"/> contiguous elements, read forward).</param>
        /// <param name="n2">Kernel length (1 ≤ n2 ≤ n1).</param>
        /// <param name="o">Output (<see cref="SlidingOutputLength"/> elements; every one is written, none read).</param>
        /// <param name="retType">The dtype of all three buffers.</param>
        /// <param name="mode">Valid, same or full.</param>
        /// <param name="slidingBlas">The installed byte-parity backend (<c>TensorEngine.Blas as ISlidingDotBackend</c>), or
        ///     null for NumSharp's managed kernels.</param>
        /// <param name="complexDotViaBlas">For complex128: whether NumPy's CDOUBLE_dot reaches <c>cblas_zdotu_sub</c> — both
        ///     operands' strides positive as its dot sees them (<see cref="DotOperandBlasable"/>). False (a one-element
        ///     operand passed through with a non-positive stride, e.g. np.convolve's reversed one-element kernel) runs its
        ///     plain loop instead, NOT the backend: that loop builds no C99 complex, so an infinite imaginary part leaves the
        ///     real part alone.</param>
        /// <exception cref="NotSupportedException"><paramref name="retType"/> has no sliding kernel.</exception>
        internal static unsafe void SlidingCorrelateInto(void* a, long n1, void* k, long n2, void* o, NPTypeCode retType, SlidingMode mode,
            ISlidingDotBackend slidingBlas, bool complexDotViaBlas = true)
        {
            long nLeft, nRight;
            switch (mode)
            {
                case SlidingMode.Valid:
                    nLeft = nRight = 0;
                    break;
                case SlidingMode.Same:
                    nLeft = n2 / 2;
                    nRight = n2 - nLeft - 1;
                    break;
                default: // Full
                    nLeft = nRight = n2 - 1;
                    break;
            }

            long mid = n1 - n2 + 1; // fully-overlapping outputs (>= 1 since n1 >= n2)

            // Accumulator per dtype family mirrors NumPy's *_dot inner loops (arraytypes.c.src):
            // float32/float64 accumulate in their own precision (matching small_correlate); Half
            // accumulates float products in float32 (one final round to Half); complex sums each
            // component in double; every integer accumulates modularly (native-lane SIMD wrap and
            // the scalar ramp both equal NumPy's accumulate-in-wide-then-truncate); bool is
            // BOOL_dot's OR-of-ANDs with early exit; decimal (no NumPy analog) accumulates in
            // decimal for full precision. Char rides the ushort SIMD kernel (same 2-byte modular
            // arithmetic).
            // OpenBLAS priority: when a byte-parity BLAS backend is installed (TensorEngine.Blas also
            // implements ISlidingDotBackend), route the EXACT positions NumPy sends through cblas — every
            // ramp, plus the middle when small_correlate declines — through that backend's ?dot, so
            // np.correlate/np.convolve match NumPy to the last bit on the long float32/float64 and
            // complex128 kernels the managed reduction reorders. small_correlate (arraytypes.c.src)
            // covers real float kernels of length <= 11 with a plain sequential sum, and the managed
            // outer-vec middle + small (<= 11) ramps are already byte-identical to NumPy there, so only
            // real n2 > 11 and complex (small_correlate never applies to complex) need the backend — and
            // for those NumPy routes every ramp AND middle position through cblas, so there is no mixed
            // managed/native path within one call. A backend that declines this dtype (e.g. a real-only
            // CBLAS handed complex) leaves the managed kernel in place. The caller reads the engine's
            // Blas property ONCE (a local) and hands it in — GEMM_PARITY §9's rule for a settable backend.
            bool useBlas = slidingBlas != null
                           && (retType == NPTypeCode.Single || retType == NPTypeCode.Double || retType == NPTypeCode.Complex)
                           && slidingBlas.SupportsDot(retType)
                           && (retType == NPTypeCode.Complex || n2 > SmallCorrelateMaxKernel);

            {
                switch (retType)
                {
                    case NPTypeCode.Byte:   SlidingSimd((byte*)a,   (byte*)k,   (byte*)o,   n2, nLeft, nRight, mid); break;
                    case NPTypeCode.SByte:  SlidingSimd((sbyte*)a,  (sbyte*)k,  (sbyte*)o,  n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Int16:  SlidingSimd((short*)a,  (short*)k,  (short*)o,  n2, nLeft, nRight, mid); break;
                    case NPTypeCode.UInt16: SlidingSimd((ushort*)a, (ushort*)k, (ushort*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Char:   SlidingSimd((ushort*)a, (ushort*)k, (ushort*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Int32:  SlidingSimd((int*)a,    (int*)k,    (int*)o,    n2, nLeft, nRight, mid); break;
                    case NPTypeCode.UInt32: SlidingSimd((uint*)a,   (uint*)k,   (uint*)o,   n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Int64:  SlidingSimd((long*)a,   (long*)k,   (long*)o,   n2, nLeft, nRight, mid); break;
                    case NPTypeCode.UInt64: SlidingSimd((ulong*)a,  (ulong*)k,  (ulong*)o,  n2, nLeft, nRight, mid); break;
                    // Without the backend, a kernel long enough to reach OpenBLAS's vector dot block takes the blocked
                    // long-product kernels (NDArray.SlidingDot.Long.cs) for exactly the positions NumPy sums with that
                    // vector kernel; every shorter position keeps its byte-exact scalar-regime sum. The Try* methods
                    // decline (writing nothing) on a host without AVX/FMA or for a kernel holding a non-finite value.
                    case NPTypeCode.Single:
                        if (useBlas) SlidingBlas(slidingBlas, NPTypeCode.Single, (float*)a, (float*)k, (float*)o, n2, nLeft, nRight, mid);
                        else if (!TrySlidingSingleLong((float*)a, (float*)k, (float*)o, n2, nLeft, nRight, mid))
                            SlidingSingle((float*)a, (float*)k, (float*)o, n2, nLeft, nRight, mid);
                        break;
                    case NPTypeCode.Double:
                        if (useBlas) SlidingBlas(slidingBlas, NPTypeCode.Double, (double*)a, (double*)k, (double*)o, n2, nLeft, nRight, mid);
                        else if (!TrySlidingDoubleLong((double*)a, (double*)k, (double*)o, n2, nLeft, nRight, mid))
                            SlidingSimd((double*)a, (double*)k, (double*)o, n2, nLeft, nRight, mid);
                        break;
                    // float16 has no BLAS route in NumPy: HALF_dot is a sequential float32 sum at every length, which the
                    // blocked exact kernel reproduces for every position (declining only as above).
                    case NPTypeCode.Half:
                        if (!TrySlidingHalfExact((Half*)a, (Half*)k, (Half*)o, n2, nLeft, nRight, mid))
                            SlidingHalf((Half*)a, (Half*)k, (Half*)o, n2, nLeft, nRight, mid);
                        break;
                    case NPTypeCode.Complex:
                        // An operand NumPy's dot cannot hand to cblas takes CDOUBLE_dot's own loop — ahead of the backend,
                        // which NumPy does not call then either.
                        if (!complexDotViaBlas) SlidingComplexPlain((Complex*)a, (Complex*)k, (Complex*)o, n2, nLeft, nRight, mid);
                        else if (useBlas) SlidingBlas(slidingBlas, NPTypeCode.Complex, (Complex*)a, (Complex*)k, (Complex*)o, n2, nLeft, nRight, mid);
                        else if (!TrySlidingComplexLong((Complex*)a, (Complex*)k, (Complex*)o, n2, nLeft, nRight, mid))
                            SlidingComplex((Complex*)a, (Complex*)k, (Complex*)o, n2, nLeft, nRight, mid);
                        break;
                    case NPTypeCode.Decimal: SlidingDecimal((decimal*)a, (decimal*)k, (decimal*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Boolean: SlidingBoolean((bool*)a, (bool*)k, (bool*)o, n2, nLeft, nRight, mid); break;
                    default:
                        throw new NotSupportedException($"Type {retType} is not supported for correlate/convolve.");
                }
            }
        }

        // --------------------------------------------------------------------------------------
        //  SIMD path — the 10 Vector<T>-capable dtypes (byte/sbyte/i16/u16/i32/u32/i64/u64/f32/f64;
        //  char reinterpreted as ushort). Middle is outer-vectorized over OUTPUT positions with
        //  SEPARATE multiply then add — each output lane still accumulates sum_t a[i+t]*k[t] in
        //  t-order, so it is bit-identical to the scalar sequential sum (verified across all these
        //  dtypes). FMA is deliberately NOT used: it would fuse the mul+add and change float
        //  rounding vs NumPy's small_correlate.
        // --------------------------------------------------------------------------------------
        // Below this dot length the ramps / few-output tail stay scalar. This keeps the
        // small-kernel regime bit-identical to NumPy (every ramp dot of a nv<=31 kernel is < 32
        // and stays sequential), and only genuinely large kernels — where NumPy itself routes the
        // dot through cblas and reorders — take the reordered SIMD reduction. The main middle loop
        // is always outer-vectorized (scalar per-lane order), so it is unaffected by this gate.
        private const long InnerDotSimdMin = 32;

        // NumPy's small_correlate (arraytypes.c.src) handles the fully-overlapping MIDDLE with a plain
        // sequential sum ONLY for uniform real float32/float64 kernels of length <= 11 ("Calling a BLAS
        // dot product for the inner loop is overkill for small kernels"); anything longer, and every
        // complex kernel, falls to the cblas dotfunc. The managed outer-vec middle reproduces that
        // sequential sum bit-for-bit, so at or below this the managed path already matches NumPy without
        // a backend — this is the boundary above which the ISlidingDotBackend route takes over.
        internal const long SmallCorrelateMaxKernel = 11;

        // --------------------------------------------------------------------------------------
        //  Byte-parity BLAS path — float32 / float64 / complex128 when a byte-parity backend is
        //  installed AND NumPy routes the reduction through cblas (real kernel longer than 11, or any
        //  complex kernel). Every ramp AND middle position goes through the backend's ?dot, exactly the
        //  positions and exactly the summation (double-accumulated chunked cblas ?dot / ?dotu) NumPy's
        //  _pyarray_correlate uses when small_correlate declines — so the result is byte-identical to
        //  NumPy. The index arithmetic is IDENTICAL to SlidingSimd/SlidingComplex below (same ramp/
        //  middle/ramp positions); only the inner dot is swapped from the reordered managed reduction to
        //  the backend's. One backend call per output position mirrors NumPy's own per-position dotfunc
        //  call: this trades the managed SIMD speed for NumPy parity, the opt-in the backend exists for.
        // --------------------------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingBlas<T>(ISlidingDotBackend blas, NPTypeCode tc,
            T* a, T* k, T* o, long n2, long nLeft, long nRight, long mid)
            where T : unmanaged
        {
            // Left ramp: out[j] = sum_t a[t] * k[(nLeft - j) + t], length (n2 - nLeft + j).
            for (long j = 0; j < nLeft; j++)
                blas.Dot(tc, a, 1, k + (nLeft - j), 1, o + j, n2 - nLeft + j);

            // Middle: out[nLeft + i] = sum_{t=0}^{n2-1} a[i+t] * k[t], i in [0, mid). This is the
            // bulk of the work (mid ~= n1 output positions vs. only nLeft+nRight ramp positions), so
            // it goes through DotBatch — ONE interface call for the whole uniform region. The backend
            // then runs the tight per-position ?dot loop with no per-output virtual dispatch, which is
            // bit-identical to calling blas.Dot per position but hoists the ~15 ns/position dispatch
            // (~1.45 ms on a 100K signal) out of the loop. The ramps below keep the per-position Dot:
            // there are at most (n2-1) of them each side, negligible against mid.
            T* om = o + nLeft;
            if (mid > 0)
                blas.DotBatch(tc, a, k, om, mid, n2);

            // Right ramp: out[nLeft + mid + j] = sum_t a[(mid + j) + t] * k[t], length (n2 - 1 - j).
            T* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                blas.Dot(tc, a + mid + j, 1, k, 1, orr + j, n2 - 1 - j);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingSimd<T>(T* a, T* k, T* o, long n2, long nLeft, long nRight, long mid)
            where T : unmanaged, IMultiplyOperators<T, T, T>, IAdditionOperators<T, T, T>
        {
            // Left ramp: out[j] = sum_{t} a[t] * k[(nLeft - j) + t], length (n2 - nLeft + j).
            for (long j = 0; j < nLeft; j++)
                o[j] = DotSimd(a, k + (nLeft - j), n2 - nLeft + j);

            SlidingMiddle(a, k, o + nLeft, n2, mid);

            // Right ramp: out[nLeft + mid + j] = sum_{t} a[(mid + j) + t] * k[t], length (n2-1-j).
            T* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = DotSimd(a + mid + j, k, n2 - 1 - j);
        }

        /// <summary>
        ///     The fully-overlapping middle of <see cref="SlidingSimd{T}"/>:
        ///     <c>om[i] = sum_{t=0}^{n2-1} a[i+t] * k[t]</c>, i in [0, mid), vectorized over OUTPUT positions so each lane keeps
        ///     the sequential t-order sum in T — NumPy's small_correlate (<c>s += d*k</c>, no FMA) bit for bit.
        /// </summary>
        /// <param name="a">Data element 0.</param>
        /// <param name="k">Kernel element 0 (read forward).</param>
        /// <param name="om">The middle's first output.</param>
        /// <param name="n2">Kernel length.</param>
        /// <param name="mid">Output positions.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingMiddle<T>(T* a, T* k, T* om, long n2, long mid)
            where T : unmanaged, IMultiplyOperators<T, T, T>, IAdditionOperators<T, T, T>
        {
            // Middle: out[nLeft + i] = sum_{t=0}^{n2-1} a[i+t] * k[t], i in [0, mid). Vectorized
            // over OUTPUT positions (each lane keeps sequential t-order → bit-exact vs scalar).
            int W = Vector<T>.Count;
            long i = 0;

            if (Vector.IsHardwareAccelerated && mid >= W)
            {
                // 4x unrolled over output strips: four independent accumulators break the
                // per-strip carried dependency (each k[t] is broadcast once and reused).
                long fourW = 4L * W;
                for (; i <= mid - fourW; i += fourW)
                {
                    var acc0 = Vector<T>.Zero;
                    var acc1 = Vector<T>.Zero;
                    var acc2 = Vector<T>.Zero;
                    var acc3 = Vector<T>.Zero;
                    for (long t = 0; t < n2; t++)
                    {
                        var kv = new Vector<T>(k[t]);
                        T* ap = a + i + t;
                        acc0 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(ap) * kv;
                        acc1 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(ap + W) * kv;
                        acc2 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(ap + 2 * W) * kv;
                        acc3 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(ap + 3 * W) * kv;
                    }
                    SRCS.Unsafe.WriteUnaligned(om + i, acc0);
                    SRCS.Unsafe.WriteUnaligned(om + i + W, acc1);
                    SRCS.Unsafe.WriteUnaligned(om + i + 2 * W, acc2);
                    SRCS.Unsafe.WriteUnaligned(om + i + 3 * W, acc3);
                }
                // 1x vector remainder.
                for (; i <= mid - W; i += W)
                {
                    var acc = Vector<T>.Zero;
                    for (long t = 0; t < n2; t++)
                        acc += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + i + t) * new Vector<T>(k[t]);
                    SRCS.Unsafe.WriteUnaligned(om + i, acc);
                }
            }

            // Tail (< W outputs, or the whole middle when mid < W — the few-outputs-long-kernel
            // regime). Each is one inner dot, SIMD-reduced when the kernel is long.
            for (; i < mid; i++)
                om[i] = DotSimd(a + i, k, n2);
        }

        // --------------------------------------------------------------------------------------
        //  float32 — FLOAT_dot's cblas_sdot sums in DOUBLE
        // --------------------------------------------------------------------------------------
        // NumPy's float32 dotfunc (FLOAT_dot, the ramps always, the middle once small_correlate declines a
        // kernel longer than 11) calls cblas_sdot, and scipy-openblas' sdot below its vector threshold
        // rounds each product to float32 and sums in DOUBLE, returning the float32 of that sum (probed
        // 2.4.2, np.dot of 1..31 random float32 terms: 0 of 9,300 differ from this model; a float32 sum
        // differs on up to 72%). From SdotVectorMin terms the vector kernel reorders the sum, which only the
        // byte-parity backend reproduces. The middle of a kernel of at most 11 stays small_correlate's
        // float32 sequential sum (SlidingMiddle).
        private const long SdotVectorMin = 32;

        // The complex twin: scipy-openblas' zdotu runs its scalar four-sum tail below 8 terms (see SlidingComplex).
        private const long ZdotuVectorMin = 8;

        /// <summary>
        ///     float32 sliding engine: NumPy's region split with its float32 arithmetic per region — ramps (and the middle of a
        ///     kernel longer than 11) through <see cref="SdotManaged"/>, the middle of a short kernel through small_correlate's
        ///     float32 sum.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="o">Output element 0.</param><param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param><param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Middle positions.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingSingle(float* a, float* k, float* o, long n2, long nLeft, long nRight, long mid)
        {
            for (long j = 0; j < nLeft; j++)
                o[j] = SdotManaged(a, k + (nLeft - j), n2 - nLeft + j);

            float* om = o + nLeft;
            if (n2 <= SmallCorrelateMaxKernel || n2 >= SdotVectorMin)
                SlidingMiddle(a, k, om, n2, mid);   // small_correlate (or a reordered sdot nobody reproduces without the backend)
            else
                SlidingMiddleSdot(a, k, om, n2, mid);

            float* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = SdotManaged(a + mid + j, k, n2 - 1 - j);
        }

        /// <summary>
        ///     NumPy's float32 dotfunc for one position: <c>(float)(Σ (double)(float)(a[t]*k[t]))</c> below
        ///     <see cref="SdotVectorMin"/> terms — cblas_sdot's scalar sum, reproduced exactly — and the reordered managed
        ///     reduction beyond (the vector kernel, bounded-ULP without the backend).
        /// </summary>
        /// <param name="a">First operand.</param><param name="k">Second operand.</param><param name="len">Terms.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float SdotManaged(float* a, float* k, long len)
        {
            if (len >= SdotVectorMin)
                return DotSimd(a, k, len);
            double d = 0.0;
            for (long t = 0; t < len; t++)
                d += a[t] * k[t];   // a float32 product, rounded, then widened into the double sum
            return (float)d;
        }

        /// <summary>
        ///     The middle for a float32 kernel of 12 … 31 terms (small_correlate declines, sdot's scalar sum runs): every
        ///     position is <see cref="SdotManaged"/>'s double-accumulated sum, vectorized over OUTPUT positions — eight float32
        ///     products per step widened into two double accumulators, each lane keeping the sequential t-order sum.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="om">The middle's first output.</param><param name="n2">Kernel length (12 … 31).</param>
        /// <param name="mid">Middle positions.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingMiddleSdot(float* a, float* k, float* om, long n2, long mid)
        {
            long i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 8 <= mid; i += 8)
                {
                    var lo = Vector256<double>.Zero;
                    var hi = Vector256<double>.Zero;
                    for (long t = 0; t < n2; t++)
                    {
                        var p = Vector256.Load(a + i + t) * Vector256.Create(k[t]);   // eight rounded float32 products
                        var (pl, ph) = Vector256.Widen(p);
                        lo += pl;
                        hi += ph;
                    }
                    Vector256.Narrow(lo, hi).Store(om + i);
                }
            }
            for (; i < mid; i++)
                om[i] = SdotManaged(a + i, k, n2);
        }

        /// <summary>
        /// Inner dot product <c>sum_{t=0}^{len-1} a[t]*k[t]</c> for a ramp / few-output tail. Short
        /// dots (len &lt; <see cref="InnerDotSimdMin"/>) run the sequential scalar sum (bit-exact vs
        /// NumPy's small-kernel path). Long dots take a 4-accumulator SIMD reduction — a REORDERED
        /// sum. For every integer dtype the reorder is exact (modular add is associative AND
        /// commutative), so this stays bit-identical to NumPy's scalar INT_dot even for large
        /// kernels; for float32/float64 it is bounded-ULP, the same regime NumPy reorders via cblas.
        /// (SEPARATE multiply-then-add, not FMA — <c>Vector.FusedMultiplyAdd</c> is net10-only, and
        /// keeping one path makes net8 and net10 identical.)
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe T DotSimd<T>(T* a, T* k, long len)
            where T : unmanaged, IMultiplyOperators<T, T, T>, IAdditionOperators<T, T, T>
        {
            long t = 0;
            T s = default;
            int W = Vector<T>.Count;

            if (Vector.IsHardwareAccelerated && len >= InnerDotSimdMin)
            {
                var acc0 = Vector<T>.Zero;
                var acc1 = Vector<T>.Zero;
                var acc2 = Vector<T>.Zero;
                var acc3 = Vector<T>.Zero;
                long fourW = 4L * W;
                for (; t <= len - fourW; t += fourW)
                {
                    acc0 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + t) * SRCS.Unsafe.ReadUnaligned<Vector<T>>(k + t);
                    acc1 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + t + W) * SRCS.Unsafe.ReadUnaligned<Vector<T>>(k + t + W);
                    acc2 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + t + 2 * W) * SRCS.Unsafe.ReadUnaligned<Vector<T>>(k + t + 2 * W);
                    acc3 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + t + 3 * W) * SRCS.Unsafe.ReadUnaligned<Vector<T>>(k + t + 3 * W);
                }
                for (; t <= len - W; t += W)
                    acc0 += SRCS.Unsafe.ReadUnaligned<Vector<T>>(a + t) * SRCS.Unsafe.ReadUnaligned<Vector<T>>(k + t);
                s = Vector.Sum((acc0 + acc1) + (acc2 + acc3));
            }

            for (; t < len; t++)
                s = s + a[t] * k[t];
            return s;
        }

        // --------------------------------------------------------------------------------------
        //  Scalar paths — Half / Complex / Decimal / Boolean have no Vector<T> arithmetic, so
        //  they stay scalar (matching the CLAUDE.md perf-notes policy). Each reproduces NumPy's
        //  *_dot accumulator exactly.
        // --------------------------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingHalf(Half* a, Half* k, Half* o, long n2, long nLeft, long nRight, long mid)
        {
            // HALF_dot: products and accumulation in float32, one final round to Half.
            for (long j = 0; j < nLeft; j++)
            {
                long len = n2 - nLeft + j, koff = nLeft - j;
                float s = 0f;
                for (long t = 0; t < len; t++) s += (float)a[t] * (float)k[koff + t];
                o[j] = (Half)s;
            }
            Half* om = o + nLeft;
            for (long i = 0; i < mid; i++)
            {
                float s = 0f;
                for (long t = 0; t < n2; t++) s += (float)a[i + t] * (float)k[t];
                om[i] = (Half)s;
            }
            Half* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
            {
                long len = n2 - 1 - j; Half* ar = a + mid + j;
                float s = 0f;
                for (long t = 0; t < len; t++) s += (float)ar[t] * (float)k[t];
                orr[j] = (Half)s;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingComplex(Complex* a, Complex* k, Complex* o, long n2, long nLeft, long nRight, long mid)
        {
            for (long j = 0; j < nLeft; j++)
                o[j] = ZdotuManaged(a, k + (nLeft - j), n2 - nLeft + j);
            Complex* om = o + nLeft;
            for (long i = 0; i < mid; i++)
                om[i] = ZdotuManaged(a + i, k, n2);
            Complex* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = ZdotuManaged(a + mid + j, k, n2 - 1 - j);
        }

        /// <summary>
        ///     The complex128 sliding engine when NumPy's CDOUBLE_dot does NOT reach cblas (an operand whose stride its dot
        ///     cannot pass on — <see cref="DotOperandBlasable"/>): every position through CDOUBLE_dot's own loop,
        ///     <see cref="CdoubleDotPlain"/>. Same positions as <see cref="SlidingComplex"/>.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="o">Output element 0.</param><param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param><param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingComplexPlain(Complex* a, Complex* k, Complex* o, long n2, long nLeft, long nRight, long mid)
        {
            for (long j = 0; j < nLeft; j++)
                o[j] = CdoubleDotPlain(a, k + (nLeft - j), n2 - nLeft + j);
            Complex* om = o + nLeft;
            for (long i = 0; i < mid; i++)
                om[i] = CdoubleDotPlain(a + i, k, n2);
            Complex* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = CdoubleDotPlain(a + mid + j, k, n2 - 1 - j);
        }

        /// <summary>
        ///     CDOUBLE_dot's loop without cblas (arraytypes.c.src): <c>sumr += ip1r*ip2r - ip1i*ip2i; sumi += ip1r*ip2i +
        ///     ip1i*ip2r</c> from +0, the first operand's parts first — plain double arithmetic, so a non-finite imaginary
        ///     part does not reach the real part (compare <see cref="ZdotuResult"/>).
        /// </summary>
        /// <param name="ap">First operand (NumPy's <c>ip1</c>, the data).</param>
        /// <param name="kp">Second operand (<c>ip2</c>, the kernel; not conjugated).</param>
        /// <param name="len">Terms.</param>
        /// <returns>The dot product.</returns>
        private static unsafe Complex CdoubleDotPlain(Complex* ap, Complex* kp, long len)
        {
            double sumr = 0.0, sumi = 0.0;
            for (long t = 0; t < len; t++)
            {
                var x = ap[t]; var y = kp[t];
                sumr += x.Real * y.Real - x.Imaginary * y.Imaginary;
                sumi += x.Real * y.Imaginary + x.Imaginary * y.Real;
            }
            return new Complex(sumr, sumi);
        }

        /// <summary>
        ///     NumPy's complex dotfunc for one position. CDOUBLE_dot calls cblas_zdotu_sub, and scipy-openblas' zdotu below
        ///     its vector threshold keeps FOUR plain double sums — re = Σ ar*br − Σ ai*bi, im = Σ ar*bi + Σ ai*br — which
        ///     CDOUBLE_dot adds into a zeroed double pair (probed 2.4.2, np.dot of 1..7 random complex terms: 0 of 2,100
        ///     differ from this model; the per-term naive product differs on up to 90%). From <see cref="ZdotuVectorMin"/>
        ///     terms the vector kernel reorders the sums (byte parity through the backend only); those keep the naive per-term
        ///     product. Both regimes end in zdotu's C99 result construction (<see cref="ZdotuResult"/>), which is what makes a
        ///     non-finite imaginary part poison the real part exactly as NumPy's does.
        /// </summary>
        /// <param name="ap">First operand.</param>
        /// <param name="kp">Second operand (not conjugated).</param>
        /// <param name="len">Terms.</param>
        /// <returns>The dot product.</returns>
        private static unsafe Complex ZdotuManaged(Complex* ap, Complex* kp, long len)
        {
            if (len < ZdotuVectorMin)
            {
                double d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (long t = 0; t < len; t++)
                {
                    var x = ap[t]; var y = kp[t];
                    d0 += x.Real * y.Real;
                    d1 += x.Imaginary * y.Imaginary;
                    d2 += x.Real * y.Imaginary;
                    d3 += x.Imaginary * y.Real;
                }
                return ZdotuResult(d0 - d1, d2 + d3);
            }
            double sr = 0, si = 0;
            for (long t = 0; t < len; t++)
            {
                var x = ap[t]; var y = kp[t];
                sr += x.Real * y.Real - x.Imaginary * y.Imaginary;
                si += x.Real * y.Imaginary + x.Imaginary * y.Real;
            }
            return ZdotuResult(sr, si);
        }

        /// <summary>
        ///     zdotu's result as CDOUBLE_dot receives it, then added into CDOUBLE_dot's zeroed double pair: scipy-openblas is
        ///     built with C99 complex, so zdot_compute returns <c>openblas_make_complex_double(re, im)</c> =
        ///     <c>re + im * _Complex_I</c>, and C's real-by-complex product <c>im * (0 + 1i)</c> is <c>(im*0) + (im*1)i</c> —
        ///     the real part becomes <c>re + im*0</c>.
        /// </summary>
        /// <param name="re">The real sum <c>dot[0] - dot[1]</c>.</param>
        /// <param name="im">The imaginary sum <c>dot[2] + dot[3]</c>.</param>
        /// <returns>The dot product NumPy stores.</returns>
        /// <remarks>
        ///     Invisible for finite values (<c>im*0</c> is a zero, and <c>re</c> is never -0.0 here: every sum starts at +0),
        ///     but an infinite or NaN imaginary part makes <c>im*0</c> NaN, so the real part is NaN too: NumPy's
        ///     <c>np.convolve([1+0j], [inf+0j])</c> is <c>nan+nanj</c> and a product holding <c>inf+infj</c> reads
        ///     <c>nan+infj</c>, where the plain sums give <c>inf+nanj</c> and <c>inf+infj</c> (probed against the DLL NumPy
        ///     2.4.2 loads: <c>scipy_cblas_zdotu_sub64_</c> on 1..8-term operands). Neither Roslyn nor the JIT folds a
        ///     floating-point <c>x * 0.0</c> (it is not an identity under IEEE rules), and the unit tests pin the NaN.
        /// </remarks>
        private static Complex ZdotuResult(double re, double im)
        {
            double sum0 = 0, sum1 = 0;   // CDOUBLE_dot's `sum += tmp` over its one chunk
            sum0 += re + im * 0.0;
            sum1 += im;
            return new Complex(sum0, sum1);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingDecimal(decimal* a, decimal* k, decimal* o, long n2, long nLeft, long nRight, long mid)
        {
            static decimal Dot(decimal* ap, decimal* kp, long len)
            {
                decimal s = 0m;
                for (long t = 0; t < len; t++) s += ap[t] * kp[t];
                return s;
            }
            for (long j = 0; j < nLeft; j++)
                o[j] = Dot(a, k + (nLeft - j), n2 - nLeft + j);
            decimal* om = o + nLeft;
            for (long i = 0; i < mid; i++)
                om[i] = Dot(a + i, k, n2);
            decimal* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = Dot(a + mid + j, k, n2 - 1 - j);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingBoolean(bool* a, bool* k, bool* o, long n2, long nLeft, long nRight, long mid)
        {
            // BOOL_dot: OR of ANDs with early exit.
            static bool Dot(bool* ap, bool* kp, long len)
            {
                for (long t = 0; t < len; t++)
                    if (ap[t] && kp[t]) return true;
                return false;
            }
            for (long j = 0; j < nLeft; j++)
                o[j] = Dot(a, k + (nLeft - j), n2 - nLeft + j);
            bool* om = o + nLeft;
            for (long i = 0; i < mid; i++)
                om[i] = Dot(a + i, k, n2);
            bool* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = Dot(a + mid + j, k, n2 - 1 - j);
        }

        /// <summary>
        /// Materialize <paramref name="x"/> to a contiguous, offset-0 array of dtype
        /// <paramref name="retType"/> (the sliding kernels walk the raw buffer from Address, which
        /// does not include Shape.offset). astype of a differing dtype already yields a fresh
        /// contiguous array; a sliced/strided/broadcast/reversed view is copied.
        /// </summary>
        internal static NDArray MaterializeForSliding(NDArray x, NPTypeCode retType)
        {
            NDArray t = x.GetTypeCode == retType ? x : x.astype(retType);
            if (t.Shape.IsSliced || t.Shape.IsBroadcasted)
                t = t.copy();
            return t;
        }
    }
}
