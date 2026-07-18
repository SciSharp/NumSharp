using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Fused single-pass "are all elements finite?" reduction used by <see cref="np.asarray_chkfinite(NDArray, Type, char)"/>.
    ///
    ///     NumPy computes this as <c>np.isfinite(a).all()</c> — a full boolean temp plus a second
    ///     reduce pass. We fuse both into ONE streaming pass with no intermediate allocation:
    ///     <list type="bullet">
    ///       <item>
    ///         <b>NaN-poison accumulation</b> (float/double): for a finite value <c>x - x == +0</c>,
    ///         but for ±inf/NaN <c>x - x == NaN</c>. Accumulating <c>acc += (v - v)</c> across the array
    ///         keeps <c>acc</c> at exactly +0 while everything is finite and poisons it to NaN the moment
    ///         any lane is non-finite (NaN is absorbing under <c>+</c>). One <c>EqualsAll(acc, 0)</c> at the
    ///         end decides the whole array — no per-vector branch, so the loop streams at memory bandwidth.
    ///         4 independent accumulators break the FP-add dependency chain.
    ///       </item>
    ///       <item>
    ///         <b>Half</b>: no <c>Vector&lt;Half&gt;</c> arithmetic in the BCL, so scan the raw 16-bit
    ///         pattern — a half is ±inf/NaN iff its 5 exponent bits are all set (<c>(bits &amp; 0x7C00) == 0x7C00</c>).
    ///       </item>
    ///       <item>
    ///         <b>Complex</b>: <c>isfinite(z)</c> ⟺ real and imag both finite. A contiguous Complex[N] is
    ///         2·N contiguous doubles (real, imag interleaved), so it reinterprets to the double scan directly.
    ///       </item>
    ///     </list>
    ///
    ///     Layout coverage (NumSharp DOD): a dense C- or F-contiguous buffer takes the flat SIMD path;
    ///     strided / transposed / negative-stride / broadcast views take an incremental-offset odometer
    ///     that still runs the SIMD run-scanner on any contiguous inner axis (stride==1) and a scalar
    ///     walk otherwise. Integer / bool / char / decimal can never be inf/NaN, so they short-circuit to true.
    /// </summary>
    internal static unsafe class FiniteScan
    {
        /// <summary>
        ///     Returns false iff <paramref name="a"/> contains any inf or NaN. Only the float family
        ///     (Half/Single/Double/Complex) can, so every other dtype trivially returns true — matching
        ///     NumPy's <c>a.dtype.char in typecodes['AllFloat']</c> gate.
        /// </summary>
        internal static bool IsAllFinite(NDArray a)
        {
            long n = a.size;
            if (n == 0)
                return true; // .all() of an empty mask is True (vacuous)

            var shape = a.Shape;
            byte* baseb = (byte*)a.Address + shape.offset * (long)a.dtypesize;
            // A C- or F-contiguous array is a dense, gap-free run of `size` elements from `offset`;
            // finiteness is order-independent so either layout scans linearly.
            bool dense = shape.IsContiguous || shape.IsFContiguous;

            switch (a.typecode)
            {
                case NPTypeCode.Single:
                    return dense
                        ? AllFiniteContig<float>((float*)baseb, n)
                        : ScanStrided(baseb, shape.dimensions, shape.strides, a.ndim, sizeof(float), &ScanRunSingle);
                case NPTypeCode.Double:
                    return dense
                        ? AllFiniteContig<double>((double*)baseb, n)
                        : ScanStrided(baseb, shape.dimensions, shape.strides, a.ndim, sizeof(double), &ScanRunDouble);
                case NPTypeCode.Half:
                    return dense
                        ? AllFiniteHalfContig((ushort*)baseb, n)
                        : ScanStrided(baseb, shape.dimensions, shape.strides, a.ndim, sizeof(ushort), &ScanRunHalf);
                case NPTypeCode.Complex:
                    return dense
                        ? AllFiniteContig<double>((double*)baseb, n * 2) // 2 doubles per complex, interleaved
                        : ScanStrided(baseb, shape.dimensions, shape.strides, a.ndim, sizeof(double) * 2, &ScanRunComplex);
                default:
                    return true; // integers / bool / char / decimal are always finite
            }
        }

        /// <summary>
        ///     Contiguous NaN-poison SIMD scan for float/double, 4× unrolled with a scalar tail.
        ///     Generic over <typeparamref name="T"/> so one body serves both element widths — the JIT
        ///     bakes the vector width (V128/V256/V512) per instantiation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool AllFiniteContig<T>(T* src, long n) where T : unmanaged, INumber<T>
        {
            long i = 0;

            if (Vector.IsHardwareAccelerated && Vector<T>.IsSupported)
            {
                int w = Vector<T>.Count;
                long step = (long)w * 4;
                if (n >= step)
                {
                    Vector<T> a0 = Vector<T>.Zero, a1 = a0, a2 = a0, a3 = a0;
                    long end = n - step;
                    for (; i <= end; i += step)
                    {
                        var v0 = Vector.Load(src + i);
                        var v1 = Vector.Load(src + i + w);
                        var v2 = Vector.Load(src + i + 2 * w);
                        var v3 = Vector.Load(src + i + 3 * w);
                        // (v - v) is +0 for finite lanes, NaN for inf/NaN lanes; NaN is absorbing under +.
                        a0 += v0 - v0;
                        a1 += v1 - v1;
                        a2 += v2 - v2;
                        a3 += v3 - v3;
                    }

                    if (!Vector.EqualsAll((a0 + a1) + (a2 + a3), Vector<T>.Zero))
                        return false; // some lane poisoned to NaN → a non-finite element exists
                }
            }

            for (; i < n; i++)
            {
                T x = src[i];
                if (x - x != T.Zero) // NaN for non-finite, +0 (== T.Zero) for finite
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     Contiguous SIMD scan for Half via its raw 16-bit pattern: ±inf/NaN ⟺ the 5 exponent
        ///     bits are all set. <c>Vector&lt;ushort&gt;</c> is JIT-vectorizable where <c>Vector&lt;Half&gt;</c> is not.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool AllFiniteHalfContig(ushort* src, long n)
        {
            const ushort ExpMask = 0x7C00;
            long i = 0;

            if (Vector.IsHardwareAccelerated && Vector<ushort>.IsSupported && n >= Vector<ushort>.Count)
            {
                int w = Vector<ushort>.Count;
                var mask = new Vector<ushort>(ExpMask);
                long end = n - w;
                for (; i <= end; i += w)
                {
                    var v = Vector.Load(src + i);
                    if (Vector.EqualsAny(v & mask, mask))
                        return false; // any lane with all exponent bits set → inf/NaN
                }
            }

            for (; i < n; i++)
            {
                if ((src[i] & ExpMask) == ExpMask)
                    return false;
            }
            return true;
        }

        // ── Strided odometer ────────────────────────────────────────────────
        // Run-scanner contract (function pointer): (rowBase, innerLen, innerStrideElements) → all finite in this run?

        /// <summary>
        ///     Walk every axis but the last by incremental offset advance (no per-element div/mod),
        ///     delegating each innermost run to <paramref name="scanRun"/>. The run-scanner takes the
        ///     SIMD fast path when the inner axis is contiguous (stride==1) and a scalar walk otherwise,
        ///     so row-contiguous strided views (the common slice) still vectorize.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static bool ScanStrided(byte* baseb, long[] dims, long[] strides, int ndim, int elemSize,
            delegate*<byte*, long, long, bool> scanRun)
        {
            int inner = ndim - 1;
            long innerLen = dims[inner];
            long innerStride = strides[inner]; // in elements

            long outerCount = 1;
            for (int d = 0; d < inner; d++)
                outerCount *= dims[d];

            Span<long> coord = ndim <= 16 ? stackalloc long[ndim] : new long[ndim];
            coord.Clear();

            long byteOffset = 0;
            for (long o = 0; o < outerCount; o++)
            {
                if (!scanRun(baseb + byteOffset, innerLen, innerStride))
                    return false;

                // Advance the outer odometer, carrying by subtracting the wrapped axis' full extent.
                for (int d = inner - 1; d >= 0; d--)
                {
                    byteOffset += strides[d] * elemSize;
                    if (++coord[d] < dims[d])
                        break;
                    coord[d] = 0;
                    byteOffset -= strides[d] * dims[d] * elemSize;
                }
            }
            return true;
        }

        private static bool ScanRunSingle(byte* p, long len, long stride)
        {
            float* fp = (float*)p;
            if (stride == 1)
                return AllFiniteContig<float>(fp, len);
            // |stride|==1 is a contiguous run; stride==-1 is the SAME block walked backward, and
            // finiteness is order-independent — scan it forward from its low address, no gather.
            if (stride == -1)
                return AllFiniteContig<float>(fp - (len - 1), len);

            long i = 0;
            // AVX2-gather NaN-poison for the inner-strided run (any stride, incl. negative). The
            // 8 element indices [0,s,…,7s] must fit the int32 gather budget after scaling by 4.
            if (Avx2.IsSupported && len >= 8 && FitsGatherIndex(stride, 7, sizeof(float)))
            {
                var idx = Vector256.Create(0, (int)stride, (int)(2 * stride), (int)(3 * stride),
                                           (int)(4 * stride), (int)(5 * stride), (int)(6 * stride), (int)(7 * stride));
                var acc = Vector256<float>.Zero;
                long end = len - 8;
                for (; i <= end; i += 8)
                {
                    var vg = Avx2.GatherVector256(fp + i * stride, idx, sizeof(float));
                    acc += vg - vg;
                }
                if (!Vector256.EqualsAll(acc, Vector256<float>.Zero))
                    return false;
            }
            for (; i < len; i++)
            {
                float x = fp[i * stride];
                if (x - x != 0f)
                    return false;
            }
            return true;
        }

        private static bool ScanRunDouble(byte* p, long len, long stride)
        {
            double* dp = (double*)p;
            if (stride == 1)
                return AllFiniteContig<double>(dp, len);
            if (stride == -1)
                return AllFiniteContig<double>(dp - (len - 1), len);

            long i = 0;
            if (Avx2.IsSupported && len >= 4 && FitsGatherIndex(stride, 3, sizeof(double)))
            {
                var idx = Vector128.Create(0, (int)stride, (int)(2 * stride), (int)(3 * stride));
                var acc = Vector256<double>.Zero;
                long end = len - 4;
                for (; i <= end; i += 4)
                {
                    var vg = Avx2.GatherVector256(dp + i * stride, idx, sizeof(double));
                    acc += vg - vg;
                }
                if (!Vector256.EqualsAll(acc, Vector256<double>.Zero))
                    return false;
            }
            for (; i < len; i++)
            {
                double x = dp[i * stride];
                if (x - x != 0d)
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     True when the largest gather index (<paramref name="maxLane"/>·<paramref name="stride"/>)
        ///     scaled by the element size stays within the signed 32-bit byte-offset budget the AVX2
        ///     gather instruction uses; otherwise the strided run must stay scalar.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool FitsGatherIndex(long stride, int maxLane, int elemSize)
        {
            long maxByte = Math.Abs(stride) * maxLane * elemSize;
            return maxByte <= int.MaxValue;
        }

        private static bool ScanRunHalf(byte* p, long len, long stride)
        {
            ushort* up = (ushort*)p;
            if (stride == 1)
                return AllFiniteHalfContig(up, len);
            if (stride == -1)
                return AllFiniteHalfContig(up - (len - 1), len);
            for (long k = 0; k < len; k++)
            {
                if ((up[k * stride] & 0x7C00) == 0x7C00)
                    return false;
            }
            return true;
        }

        private static bool ScanRunComplex(byte* p, long len, long stride)
        {
            if (stride == 1)
                return AllFiniteContig<double>((double*)p, len * 2); // contiguous complex run = 2·len interleaved doubles
            if (stride == -1)
                return AllFiniteContig<double>((double*)((Complex*)p - (len - 1)), len * 2); // reversed contiguous block

            Complex* cp = (Complex*)p;
            for (long k = 0; k < len; k++)
            {
                var z = cp[k * stride];
                double re = z.Real, im = z.Imaginary;
                if (re - re != 0d || im - im != 0d)
                    return false;
            }
            return true;
        }
    }
}
