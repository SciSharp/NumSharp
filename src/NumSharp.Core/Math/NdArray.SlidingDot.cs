using System;
using System.Numerics;
using System.Runtime.CompilerServices;
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
        internal static NDArray SlidingCorrelate(NDArray data, NDArray kernel, NPTypeCode retType, SlidingMode mode)
        {
            long n1 = data.size;
            long n2 = kernel.size;

            long nLeft, nRight, outLen;
            switch (mode)
            {
                case SlidingMode.Valid:
                    nLeft = nRight = 0;
                    outLen = n1 - n2 + 1;
                    break;
                case SlidingMode.Same:
                    nLeft = n2 / 2;
                    nRight = n2 - nLeft - 1;
                    outLen = n1;
                    break;
                default: // Full
                    nLeft = nRight = n2 - 1;
                    outLen = n1 + n2 - 1;
                    break;
            }

            long mid = n1 - n2 + 1; // fully-overlapping outputs (>= 1 since n1 >= n2)

            // No zero-fill: the three regions (left ramp + middle + right ramp) write EVERY one of
            // the outLen positions exactly once (nLeft + mid + nRight == outLen), so a zeroing pass
            // would only be redundant writes on the memory-bound common case.
            var result = new NDArray(retType, Shape.Vector(outLen), false);

            // Accumulator per dtype family mirrors NumPy's *_dot inner loops (arraytypes.c.src):
            // float32/float64 accumulate in their own precision (matching small_correlate); Half
            // accumulates float products in float32 (one final round to Half); complex sums each
            // component in double; every integer accumulates modularly (native-lane SIMD wrap and
            // the scalar ramp both equal NumPy's accumulate-in-wide-then-truncate); bool is
            // BOOL_dot's OR-of-ANDs with early exit; decimal (no NumPy analog) accumulates in
            // decimal for full precision. Char rides the ushort SIMD kernel (same 2-byte modular
            // arithmetic).
            unsafe
            {
                void* a = (void*)data.Address;
                void* k = (void*)kernel.Address;
                void* o = (void*)result.Address;

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
                    case NPTypeCode.Single: SlidingSimd((float*)a,  (float*)k,  (float*)o,  n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Double: SlidingSimd((double*)a, (double*)k, (double*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Half:    SlidingHalf((Half*)a, (Half*)k, (Half*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Complex: SlidingComplex((Complex*)a, (Complex*)k, (Complex*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Decimal: SlidingDecimal((decimal*)a, (decimal*)k, (decimal*)o, n2, nLeft, nRight, mid); break;
                    case NPTypeCode.Boolean: SlidingBoolean((bool*)a, (bool*)k, (bool*)o, n2, nLeft, nRight, mid); break;
                    default:
                        throw new NotSupportedException($"Type {retType} is not supported for correlate/convolve.");
                }
            }

            return result;
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

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingSimd<T>(T* a, T* k, T* o, long n2, long nLeft, long nRight, long mid)
            where T : unmanaged, IMultiplyOperators<T, T, T>, IAdditionOperators<T, T, T>
        {
            // Left ramp: out[j] = sum_{t} a[t] * k[(nLeft - j) + t], length (n2 - nLeft + j).
            for (long j = 0; j < nLeft; j++)
                o[j] = DotSimd(a, k + (nLeft - j), n2 - nLeft + j);

            // Middle: out[nLeft + i] = sum_{t=0}^{n2-1} a[i+t] * k[t], i in [0, mid). Vectorized
            // over OUTPUT positions (each lane keeps sequential t-order → bit-exact vs scalar).
            T* om = o + nLeft;
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

            // Right ramp: out[nLeft + mid + j] = sum_{t} a[(mid + j) + t] * k[t], length (n2-1-j).
            T* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = DotSimd(a + mid + j, k, n2 - 1 - j);
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

        private static unsafe void SlidingComplex(Complex* a, Complex* k, Complex* o, long n2, long nLeft, long nRight, long mid)
        {
            // CDOUBLE_dot: component sums in double via the naive complex product, so NaN in either
            // component propagates into BOTH result components exactly like NumPy.
            static Complex Dot(Complex* ap, Complex* kp, long len)
            {
                double sr = 0, si = 0;
                for (long t = 0; t < len; t++)
                {
                    var x = ap[t]; var y = kp[t];
                    sr += x.Real * y.Real - x.Imaginary * y.Imaginary;
                    si += x.Real * y.Imaginary + x.Imaginary * y.Real;
                }
                return new Complex(sr, si);
            }
            for (long j = 0; j < nLeft; j++)
                o[j] = Dot(a, k + (nLeft - j), n2 - nLeft + j);
            Complex* om = o + nLeft;
            for (long i = 0; i < mid; i++)
                om[i] = Dot(a + i, k, n2);
            Complex* orr = o + nLeft + mid;
            for (long j = 0; j < nRight; j++)
                orr[j] = Dot(a + mid + j, k, n2 - 1 - j);
        }

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
