using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused single-pass "are ALL imaginary parts strictly within <c>tol</c> of zero?" reduction
    ///     used by <see cref="np.real_if_close(NDArray, double)"/> to decide whether a complex128 array
    ///     collapses to its real lane.
    ///
    ///     NumPy computes this as <c>np.all(np.absolute(a.imag) &lt; tol)</c> — a strided read of the
    ///     imaginary lane, a full <c>absolute</c> temp, a full boolean <c>&lt;</c> temp and a second
    ///     reduce pass. We fuse all of that into ONE streaming pass over the imaginary lane with no
    ///     intermediate allocation and an early exit on the first out-of-band element (the
    ///     <see cref="FiniteScan"/> pattern).
    ///
    ///     <para>
    ///     The predicate per element is <c>|imag| &lt; tol</c>, evaluated so that the answer matches
    ///     NumPy's <c>absolute(x) &lt; tol</c> bit-for-bit on every value: a <b>NaN</b> imaginary part
    ///     gives <c>|NaN| &lt; tol == false</c> (any comparison with NaN is false), and <b>±inf</b> gives
    ///     <c>inf &lt; tol == false</c> — so a NaN/inf anywhere prevents the collapse, exactly as NumPy's
    ///     <c>np.all</c> over the same mask does. The comparison is STRICT (<c>&lt;</c>), so an imaginary
    ///     part exactly equal to <c>tol</c> — and <b>every</b> element when <c>tol &lt;= 0</c>, since
    ///     <c>|imag| &gt;= 0</c> can never be <c>&lt; 0</c> — fails, matching NumPy (probed 2.4.2).
    ///     </para>
    ///
    ///     <para>
    ///     Because NumSharp has only complex128, the imaginary lane is ALWAYS float64, so this kernel is
    ///     float64-only (no per-dtype switch). A <see cref="System.Numerics.Complex"/> is laid out as two
    ///     contiguous doubles (real then imaginary), so a C- or F-contiguous complex array is a gap-free
    ///     run of interleaved <c>[re,im,re,im,...]</c> doubles: the dense fast path loads them as
    ///     <see cref="Vector256{T}"/> pairs and deinterleaves the imaginary lanes with a single
    ///     <c>vshufpd</c> (all four imaginary parts per shuffle, order irrelevant for a band reduction),
    ///     then abs + <see cref="Vector256.LessThanAll{T}"/>. Strided / transposed / negative-stride /
    ///     broadcast views take an incremental-offset odometer whose innermost run either reuses the dense
    ///     scan (unit / reversed complex stride) or gathers the strided imaginary doubles (AVX2), falling
    ///     to a scalar walk for pathological strides that overflow the int32 gather index.
    ///     </para>
    /// </summary>
    internal static unsafe class ImagCloseScan
    {
        /// <summary>
        ///     Returns <c>true</c> iff EVERY imaginary part of the complex128 array <paramref name="a"/>
        ///     is strictly within <paramref name="tol"/> of zero (i.e. <c>|imag| &lt; tol</c> for all
        ///     elements) — the condition under which <see cref="np.real_if_close(NDArray, double)"/>
        ///     collapses <paramref name="a"/> to its real lane.
        /// </summary>
        /// <param name="a">
        ///     The input array. MUST be complex128 (<see cref="NPTypeCode.Complex"/>) — the caller checks
        ///     the dtype before calling, so this method reads the imaginary lane of every element through
        ///     <paramref name="a"/>'s own shape/strides/offset and supports any memory layout.
        /// </param>
        /// <param name="tol">
        ///     The absolute tolerance already resolved by the caller (NumPy multiplies the raw
        ///     <c>tol</c> by the float64 machine epsilon when it exceeds 1). A NaN or non-positive
        ///     <paramref name="tol"/> makes every non-empty array fail the band test (nothing collapses).
        /// </param>
        /// <returns>
        ///     <c>true</c> if the array is empty (an empty <c>all</c> is vacuously true, so an empty
        ///     complex array collapses to an empty real array, matching NumPy) or if every imaginary part
        ///     satisfies <c>|imag| &lt; tol</c>; otherwise <c>false</c> (some imaginary part is
        ///     &gt;= <paramref name="tol"/>, NaN, or infinite).
        /// </returns>
        internal static bool AllImagWithinTol(NDArray a, double tol)
        {
            long n = a.size;
            if (n == 0)
                return true; // all([]) is True (vacuous) -> an empty complex array collapses to real

            var shape = a.Shape;
            // First LOGICAL complex element, honouring a sliced view's offset (dtypesize(Complex)==16).
            Complex* cbase = (Complex*)((byte*)a.Address + shape.offset * (long)a.dtypesize);

            // A C- or F-contiguous complex array is a dense, gap-free run of `n` complex values; the
            // band reduction is order-independent, so either layout scans as one linear run.
            if (shape.IsContiguous || shape.IsFContiguous)
                return ScanContig(cbase, n, tol);

            // Strided / transposed / negative-stride / broadcast: walk all-but-inner axes by incremental
            // offset advance and scan each innermost run (strides are in COMPLEX elements).
            return ScanStrided(cbase, shape.dimensions, shape.strides, a.ndim, tol);
        }

        /// <summary>
        ///     Scan a CONTIGUOUS run of <paramref name="n"/> complex values starting at
        ///     <paramref name="p"/>, returning <c>false</c> at the first imaginary part that is not
        ///     strictly within <paramref name="tol"/> (early exit). AVX deinterleave fast path
        ///     (8 complex / iteration) with a scalar tail; <see cref="Vector256"/> is JIT-baked so no
        ///     runtime width branch is taken.
        /// </summary>
        /// <param name="p">Pointer to the first complex value of the run.</param>
        /// <param name="n">Number of complex values in the run.</param>
        /// <param name="tol">Absolute tolerance (see <see cref="AllImagWithinTol"/>).</param>
        /// <returns><c>true</c> iff every imaginary part in the run satisfies <c>|imag| &lt; tol</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool ScanContig(Complex* p, long n, double tol)
        {
            long i = 0;

            // 8 complex = 16 interleaved doubles per iteration. The imaginary lanes are deinterleaved
            // with vshufpd control 0b1111: Shuffle(x, y, 0xF) = [x[1], y[1], x[3], y[3]] picks the odd
            // (imaginary) lane of each 128-bit half, so two shuffles surface all eight imaginary parts.
            // Order within the vector is irrelevant — every lane is band-checked identically.
            if (Avx.IsSupported && n >= 8)
            {
                var tolv = Vector256.Create(tol);
                long end = n - 8;
                for (; i <= end; i += 8)
                {
                    var a0 = Avx.LoadVector256((double*)(p + i));      // [re_i,   im_i,   re_i+1, im_i+1]
                    var a1 = Avx.LoadVector256((double*)(p + i + 2));  // [re_i+2, im_i+2, re_i+3, im_i+3]
                    var a2 = Avx.LoadVector256((double*)(p + i + 4));
                    var a3 = Avx.LoadVector256((double*)(p + i + 6));
                    var im01 = Avx.Shuffle(a0, a1, 0b1111);            // 4 imaginary parts of complex i..i+3
                    var im23 = Avx.Shuffle(a2, a3, 0b1111);            // 4 imaginary parts of complex i+4..i+7
                    // LessThanAll is false when any lane is >= tol OR NaN (NaN comparisons are false),
                    // so a NaN/inf/out-of-band imaginary part exits immediately — matching np.all.
                    if (!Vector256.LessThanAll(Vector256.Abs(im01), tolv))
                        return false;
                    if (!Vector256.LessThanAll(Vector256.Abs(im23), tolv))
                        return false;
                }
            }

            // Scalar tail (and the whole run when AVX is unavailable or n < 8). The `!(< tol)` shape —
            // rather than `>= tol` — keeps NaN correct: |NaN| < tol is false, so !false returns false.
            for (; i < n; i++)
            {
                double im = p[i].Imaginary;
                if (!(Math.Abs(im) < tol))
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     Walk every axis but the innermost by incremental offset advance (no per-element div/mod),
        ///     delegating each innermost run to <see cref="ScanRun"/>. Returns <c>false</c> as soon as any
        ///     run reports an out-of-band imaginary part (early exit).
        /// </summary>
        /// <param name="cbase">Pointer to the first logical complex element (offset already applied).</param>
        /// <param name="dims">Per-axis extents.</param>
        /// <param name="strides">Per-axis strides in COMPLEX elements (stride 0 for a broadcast axis).</param>
        /// <param name="ndim">Rank of the array.</param>
        /// <param name="tol">Absolute tolerance (see <see cref="AllImagWithinTol"/>).</param>
        /// <returns><c>true</c> iff every imaginary part across all runs satisfies <c>|imag| &lt; tol</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool ScanStrided(Complex* cbase, long[] dims, long[] strides, int ndim, double tol)
        {
            int inner = ndim - 1;
            long innerLen = dims[inner];
            long innerStride = strides[inner]; // complex elements

            long outerCount = 1;
            for (int d = 0; d < inner; d++)
                outerCount *= dims[d];

            Span<long> coord = ndim <= 16 ? stackalloc long[ndim] : new long[ndim];
            coord.Clear();

            long elemOffset = 0; // complex elements relative to cbase
            for (long o = 0; o < outerCount; o++)
            {
                if (!ScanRun(cbase + elemOffset, innerLen, innerStride, tol))
                    return false;

                // Advance the outer odometer, carrying by subtracting the wrapped axis' full extent.
                for (int d = inner - 1; d >= 0; d--)
                {
                    elemOffset += strides[d];
                    if (++coord[d] < dims[d])
                        break;
                    coord[d] = 0;
                    elemOffset -= strides[d] * dims[d];
                }
            }
            return true;
        }

        /// <summary>
        ///     Scan one innermost run of <paramref name="len"/> complex values whose consecutive elements
        ///     are <paramref name="complexStride"/> complex elements apart, returning <c>false</c> at the
        ///     first out-of-band imaginary part.
        /// </summary>
        /// <param name="rowBase">Pointer to the run's first complex element.</param>
        /// <param name="len">Number of complex values in the run.</param>
        /// <param name="complexStride">Stride between consecutive elements, in complex elements (may be 0 for a broadcast inner axis, or negative for a reversed view).</param>
        /// <param name="tol">Absolute tolerance (see <see cref="AllImagWithinTol"/>).</param>
        /// <returns><c>true</c> iff every imaginary part in the run satisfies <c>|imag| &lt; tol</c>.</returns>
        [MethodImpl(OptimizeAndInline)]
        private static bool ScanRun(Complex* rowBase, long len, long complexStride, double tol)
        {
            // Unit stride is a contiguous run; a -1 stride is the SAME block walked backward, and the
            // band check is order-independent, so scan it forward from its low address (no gather).
            if (complexStride == 1)
                return ScanContig(rowBase, len, tol);
            if (complexStride == -1)
                return ScanContig(rowBase - (len - 1), len, tol);

            long k = 0;
            // AVX2-gather the strided imaginary doubles: element k's imaginary part sits at double index
            // k*(complexStride*2) from imag0 (the imaginary of element 0). The four gather indices
            // [0, ds, 2ds, 3ds] must fit the int32 index budget after scaling by 8 bytes.
            long ds = complexStride * 2; // double-element stride between consecutive imaginary parts
            if (Avx2.IsSupported && len >= 4 && FitsGatherIndex(ds, 3))
            {
                double* imag0 = (double*)rowBase + 1; // imaginary of element 0
                var idx = Vector128.Create(0, (int)ds, (int)(2 * ds), (int)(3 * ds));
                var tolv = Vector256.Create(tol);
                long end = len - 4;
                for (; k <= end; k += 4)
                {
                    var vg = Avx2.GatherVector256(imag0 + k * ds, idx, sizeof(double));
                    if (!Vector256.LessThanAll(Vector256.Abs(vg), tolv))
                        return false;
                }
            }

            for (; k < len; k++)
            {
                double im = rowBase[k * complexStride].Imaginary;
                if (!(Math.Abs(im) < tol))
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     True when the largest gather index (<paramref name="maxLane"/>·<paramref name="doubleStride"/>)
        ///     scaled by the 8-byte element size stays within the signed 32-bit byte-offset budget the AVX2
        ///     gather instruction uses; otherwise the strided run must stay scalar.
        /// </summary>
        /// <param name="doubleStride">Stride between consecutive imaginary parts, in double elements.</param>
        /// <param name="maxLane">The highest lane index used by the gather (3 for a 4-wide double gather).</param>
        /// <returns><c>true</c> if the gather index budget is not exceeded.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool FitsGatherIndex(long doubleStride, int maxLane)
        {
            long maxByte = Math.Abs(doubleStride) * maxLane * sizeof(double);
            return maxByte <= int.MaxValue;
        }
    }
}
