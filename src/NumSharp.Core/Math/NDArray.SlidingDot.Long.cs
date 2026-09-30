using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Kernels;

namespace NumSharp
{
    public partial class NDArray
    {
        // ======================================================================================================
        //  Long sliding products without the byte-parity backend
        // ======================================================================================================
        //
        // WHY. NumPy reduces every correlate / convolve output position with its dtype's dotfunc (the ramps always, the
        // middle once small_correlate declines a kernel longer than 11 or any complex kernel), and scipy-openblas' ?dot
        // runs a scalar sum below its vector block — ddot below 16 terms, sdot below 32 (float32 products summed in a
        // double), zdotu below 8 (four separate sums) — and a VECTOR kernel from there on. The managed engine reproduces
        // the scalar regime byte for byte (DotSimd's sequential sum, SdotManaged, ZdotuManaged); the vector regime's
        // summation order is reproduced only through NumSharp.Interop.OpenBLAS (ISlidingDotBackend). So a position of that
        // length is free to be summed in ANY order without the backend — which the per-position dots here did not use:
        // one DotSimd call per position loads two vectors per four products and re-reduces horizontally every time, so a
        // 1000 x 1000 convolution (ramps only) ran at ~0.9x NumPy, and complex long products at 0.25x (a scalar loop).
        //
        // HOW. Output p of the correlation is  Σ_s a[p - nLeft + s] · k[s]  over the terms whose data index lies inside
        // `a` — the ramps are the positions whose windows run off an end. A block of B consecutive positions shares ONE
        // loop over s across the union of its lanes' valid terms, accumulated outer-vectorized: per step, one broadcast of
        // k[s] and one load per vector (half the loads of a per-position dot, and no horizontal reduction), fused
        // multiply-adds into eight independent accumulator chains (two sets over alternating s, so the FMA latency hides).
        // The union splits three ways: the CORE, where every lane's index is inside `a` and the loads read `a` itself; and
        // at most B - 1 HEAD / TAIL steps, where some lanes' index runs off an end — those read two tiny windows of the
        // data's ends padded with zeros (≈ 2B elements each, filled once per call), so an out-of-range lane multiplies a
        // zero. Measured alternatives: masking the edge lanes (masked load + product AND) costs ~4x a core step per edge
        // step, 12 % slower at 1000 x 1000; a padded copy of each whole ramp costs O(n2) copies per call.
        //
        // THE ONE REQUIREMENT. An out-of-range lane adds 0 · k[s], which is 0 only for a FINITE k[s] (0 · inf = NaN would
        // reach a position whose true sum never met that term). A kernel holding an infinity or a NaN declines the blocked
        // route (the per-position kernels run, as before); data values may be anything — a padding zero is never
        // multiplied by data.
        //
        // WHAT STAYS EXACT. The positions shorter than the vector block (the first and last ramp positions) keep the
        // scalar-regime kernels, so every position NumPy computes without the vector kernel stays byte-identical.
        //
        // float16 is the exception to "any order": NumPy's HALF_dot is a plain float32 sum at EVERY length (no BLAS), so
        // its blocked kernel (TrySlidingHalfExact) keeps each lane's sequential s-order sum with a separate multiply and
        // add — the head, core and tail steps run in s order, and a padding product (±0) added to a running sum that
        // started at +0 (never -0) changes nothing — so it is exact at every position, over float32 copies of the operands
        // widened once by the house cast kernel.
        //
        // THE CEILING. The float64 / float32 kernels do one fused multiply-add per four / eight products, where OpenBLAS's
        // ddot / sdot need a second load per vector (both operands vary): ~2x NumPy at 1000 x 1000. complex128 is FMA-bound
        // on both sides — a complex product is four real multiply-adds, one 256-bit FMA per product either way — so the
        // blocked kernel wins only NumPy's per-position call overhead: ~1.5x at 1000 x 1000, less as the dots grow.

        /// <summary>ddot's vector block: positions of this many terms or more are summed by OpenBLAS's vector kernel.</summary>
        private const long DdotVectorMin = 16;

        /// <summary>Positions per float64 block: four Vector256 of four.</summary>
        private const int DoubleBlock = 16;

        /// <summary>Positions per float32 block (and per float16 block, computed in float32): four Vector256 of eight.</summary>
        private const int SingleBlock = 32;

        /// <summary>Positions per complex128 block: four Vector256 of two complex values.</summary>
        private const int ComplexBlock = 8;

        /// <summary>
        ///     Whether the blocked long-product kernels can run: x86-64 with AVX and FMA — every other host keeps the
        ///     per-position kernels, which are correct everywhere.
        /// </summary>
        private static bool LongProductKernelsSupported => Fma.IsSupported && Avx.IsSupported;

        /// <summary>
        ///     The float64 correlation of a kernel of <see cref="DdotVectorMin"/> or more terms without the backend: the
        ///     positions shorter than ddot's vector block through <see cref="DotSimd{T}"/>'s sequential sum (NumPy's bytes),
        ///     every other position through the blocked FMA kernel.
        /// </summary>
        /// <param name="a">Data element 0 (n1 = mid + n2 - 1 elements).</param>
        /// <param name="k">Kernel element 0 (read forward).</param>
        /// <param name="o">Output element 0.</param>
        /// <param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param>
        /// <param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        /// <returns>False (nothing written) when the host lacks AVX/FMA, the kernel is shorter than the vector block, or the
        ///     kernel holds a non-finite value — the caller then runs the per-position kernels.</returns>
        internal static unsafe bool TrySlidingDoubleLong(double* a, double* k, double* o, long n2, long nLeft, long nRight, long mid)
        {
            if (!LongProductKernelsSupported || n2 < DdotVectorMin || !AllFinite(k, n2))
                return false;
            long n1 = mid + n2 - 1;
            var (pF0, pF1) = VectorRegime(n2, nLeft, nRight, mid, DdotVectorMin);
            for (long p = 0; p < pF0; p++)                       // left ramp, shorter than the vector block
                o[p] = DotSimd(a, k + (nLeft - p), n2 - nLeft + p);
            SlidingBlocks(a, n1, k, n2, o, nLeft, pF0, pF1, DoubleBlock, &FmaBlockDouble);
            long outLen = nLeft + mid + nRight;
            for (long p = pF1; p < outLen; p++)                  // right ramp, shorter than the vector block
            {
                long j = p - nLeft - mid;
                o[p] = DotSimd(a + mid + j, k, n2 - 1 - j);
            }
            return true;
        }

        /// <summary>
        ///     The float32 correlation of a kernel of <see cref="SdotVectorMin"/> or more terms without the backend: the
        ///     positions shorter than sdot's vector block through <see cref="SdotManaged"/> (NumPy's double-accumulated
        ///     scalar sum, byte for byte), every other position through the blocked float32 FMA kernel.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="o">Output element 0.</param><param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param><param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        /// <returns>False (nothing written) when the blocked route does not apply (see <see cref="TrySlidingDoubleLong"/>).</returns>
        internal static unsafe bool TrySlidingSingleLong(float* a, float* k, float* o, long n2, long nLeft, long nRight, long mid)
        {
            if (!LongProductKernelsSupported || n2 < SdotVectorMin || !AllFinite(k, n2))
                return false;
            long n1 = mid + n2 - 1;
            var (pF0, pF1) = VectorRegime(n2, nLeft, nRight, mid, SdotVectorMin);
            for (long p = 0; p < pF0; p++)
                o[p] = SdotManaged(a, k + (nLeft - p), n2 - nLeft + p);
            SlidingBlocks(a, n1, k, n2, o, nLeft, pF0, pF1, SingleBlock, &FmaBlockSingle);
            long outLen = nLeft + mid + nRight;
            for (long p = pF1; p < outLen; p++)
            {
                long j = p - nLeft - mid;
                o[p] = SdotManaged(a + mid + j, k, n2 - 1 - j);
            }
            return true;
        }

        /// <summary>
        ///     The complex128 correlation of a kernel of <see cref="ZdotuVectorMin"/> or more terms without the backend: the
        ///     positions shorter than zdotu's vector block through <see cref="ZdotuManaged"/> (the four separate sums, byte
        ///     for byte for finite values), every other position through the blocked complex FMA kernel.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="o">Output element 0.</param><param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param><param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        /// <returns>False (nothing written) when the blocked route does not apply (see <see cref="TrySlidingDoubleLong"/>).</returns>
        internal static unsafe bool TrySlidingComplexLong(Complex* a, Complex* k, Complex* o, long n2, long nLeft, long nRight, long mid)
        {
            if (!LongProductKernelsSupported || n2 < ZdotuVectorMin || !AllFinite((double*)k, 2 * n2))
                return false;
            long n1 = mid + n2 - 1;
            var (pF0, pF1) = VectorRegime(n2, nLeft, nRight, mid, ZdotuVectorMin);
            for (long p = 0; p < pF0; p++)
                o[p] = ZdotuManaged(a, k + (nLeft - p), n2 - nLeft + p);
            SlidingBlocks(a, n1, k, n2, o, nLeft, pF0, pF1, ComplexBlock, &FmaBlockComplex);
            long outLen = nLeft + mid + nRight;
            for (long p = pF1; p < outLen; p++)
            {
                long j = p - nLeft - mid;
                o[p] = ZdotuManaged(a + mid + j, k, n2 - 1 - j);
            }
            return true;
        }

        /// <summary>
        ///     The float16 correlation, EXACT at every position (NumPy's HALF_dot: float32 products summed in float32 from
        ///     +0, one final RTNE to float16): both operands widened to float32 once by the house cast kernel, every position
        ///     summed by the blocked kernel in its own sequential s-order with a separate multiply and add, the float32 sums
        ///     narrowed by the house cast kernel.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="k">Kernel element 0 (forward).</param>
        /// <param name="o">Output element 0.</param><param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param><param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        /// <returns>False (nothing written) when the host lacks AVX, a house cast kernel is unavailable, or the kernel holds a
        ///     non-finite value — the caller then runs the scalar HALF_dot loop.</returns>
        internal static unsafe bool TrySlidingHalfExact(Half* a, Half* k, Half* o, long n2, long nLeft, long nRight, long mid)
        {
            if (!Avx.IsSupported)
                return false;
            var widen = DirectILKernelGenerator.TryGetCastKernel(NPTypeCode.Half, NPTypeCode.Single);
            var narrow = DirectILKernelGenerator.TryGetCastKernel(NPTypeCode.Single, NPTypeCode.Half);
            if (widen == null || narrow == null)
                return false;
            long n1 = mid + n2 - 1;
            long outLen = nLeft + mid + nRight;
            // One pooled float buffer: the kernel (n2), the data (n1) and the float32 sums (outLen).
            float[] rented = ArrayPool<float>.Shared.Rent(checked((int)(n2 + n1 + outLen)));
            try
            {
                fixed (float* buf = rented)
                {
                    float* k32 = buf, a32 = buf + n2, s32 = a32 + n1;
                    widen(k, k32, n2);
                    if (!AllFinite(k32, n2))
                        return false;   // a padding zero times an infinity would reach lanes that never met it
                    widen(a, a32, n1);
                    SlidingBlocks(a32, n1, k32, n2, s32, nLeft, 0, outLen, SingleBlock, &ExactBlockSingle);
                    narrow(s32, o, outLen);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
            return true;
        }

        /// <summary>
        ///     The positions NumPy sums with OpenBLAS's vector kernel — every position of at least
        ///     <paramref name="vectorMin"/> terms — as one contiguous range: the ramps' dot lengths grow (left) and shrink
        ///     (right) by one per position, the middle's is <paramref name="n2"/> ≥ <paramref name="vectorMin"/>.
        /// </summary>
        /// <param name="n2">Kernel length.</param>
        /// <param name="nLeft">Left-ramp positions.</param>
        /// <param name="nRight">Right-ramp positions.</param>
        /// <param name="mid">Fully overlapping positions.</param>
        /// <param name="vectorMin">The dot's vector-block threshold (terms).</param>
        /// <returns>[first, end) of the vector-regime positions.</returns>
        private static (long first, long end) VectorRegime(long n2, long nLeft, long nRight, long mid, long vectorMin)
        {
            // Left ramp position p sums n2 - nLeft + p terms; right ramp position nLeft + mid + j sums n2 - 1 - j.
            long first = Math.Max(0, nLeft + vectorMin - n2);
            long end = nLeft + mid + Math.Min(nRight, Math.Max(0, n2 - vectorMin));
            return (first, end);
        }

        /// <summary>
        ///     Positions [<paramref name="p0"/>, <paramref name="p1"/>) of the correlation, B at a time. For the block at p
        ///     (data offset q0 = p - nLeft, so lane i's term s reads a[q0 + i + s]): [sLo, sHi) is the union of the lanes'
        ///     valid terms and [coreLo, coreHi) the terms valid for every lane; the head steps (before the core) read the
        ///     left edge window, the core reads the data, the tail steps (after it) the right edge window. A partial last
        ///     block is computed into a scratch and only its real positions copied out.
        /// </summary>
        /// <typeparam name="T">The element type (float, double, Complex).</typeparam>
        /// <param name="a">Data element 0 (<paramref name="n1"/> elements).</param>
        /// <param name="n1">Data length.</param>
        /// <param name="k">Kernel element 0 (<paramref name="n2"/> finite elements).</param>
        /// <param name="n2">Kernel length.</param>
        /// <param name="o">Output element 0 (position 0).</param>
        /// <param name="nLeft">Left-ramp positions (the data offset of position 0 is -nLeft).</param>
        /// <param name="p0">First position to compute.</param>
        /// <param name="p1">End of the positions to compute.</param>
        /// <param name="B">Positions per block (the block kernel's width).</param>
        /// <param name="block">The block kernel <c>(a, head, tail, q0, k, sLo, coreLo, coreHi, sHi, dst)</c>:
        ///     <c>dst[i] = Σ_s a[q0 + i + s] · k[s]</c> over the terms whose index is inside the data, i in [0, B), where
        ///     <c>head[j]</c> / <c>tail[j]</c> are data index j with zeros outside the data (see the windows below).</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void SlidingBlocks<T>(T* a, long n1, T* k, long n2, T* o, long nLeft, long p0, long p1, int B,
            delegate*<T*, T*, T*, long, T*, long, long, long, long, T*, void> block)
            where T : unmanaged
        {
            if (p1 <= p0)
                return;
            // The edge windows. A head step reads data indices within [-(B - 1), B - 1) and a tail step within
            // [max(0, n1 - B + 1) - ..., n1 + B - 1): stored as zero-padded copies so an out-of-range lane reads a zero.
            // head[j] is data index j - (B - 1) for j in [0, 2B); tail[j] is data index tailStart + j for j in [0, 3B).
            long tailStart = Math.Max(0, n1 - 2 * B);
            T* win = stackalloc T[5 * B];
            T* head = win, tail = win + 2 * B;
            FillEdge(a, n1, head, -(B - 1), 2 * B);
            FillEdge(a, n1, tail, tailStart, 3 * B);
            T* scratch = stackalloc T[B];
            // Rebased pointers: headBase[x] and tailBase[x] are data index x (valid only for the indices each window holds).
            T* headBase = head + (B - 1), tailBase = tail - tailStart;
            for (long p = p0; p < p1; p += B)
            {
                long q0 = p - nLeft;
                // Lane i's valid terms: 0 <= q0 + i + s < n1, i.e. [-q0 - i, n1 - q0 - i). The union over the block starts
                // at lane B - 1's first and ends at lane 0's last; the core, where all lanes are valid, starts at lane 0's
                // first and ends at lane B - 1's last. All four are clipped to the kernel [0, n2).
                long sLo = Math.Max(0, -q0 - (B - 1));
                long sHi = Math.Min(n2, n1 - q0);
                long coreLo = Math.Max(sLo, -q0);
                long coreHi = Math.Min(sHi, n1 - q0 - B + 1);
                if (p + B <= p1)
                {
                    block(a, headBase, tailBase, q0, k, sLo, coreLo, coreHi, sHi, o + p);
                }
                else
                {
                    block(a, headBase, tailBase, q0, k, sLo, coreLo, coreHi, sHi, scratch);
                    new ReadOnlySpan<T>(scratch, (int)(p1 - p)).CopyTo(new Span<T>(o + p, (int)(p1 - p)));
                }
            }
        }

        /// <summary>
        ///     Writes data indices [<paramref name="start"/>, <paramref name="start"/> + <paramref name="len"/>) of
        ///     <paramref name="a"/> into <paramref name="dst"/>, zeros where the index is outside [0, <paramref name="n1"/>).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="a">Data element 0.</param>
        /// <param name="n1">Data length.</param>
        /// <param name="dst">Destination (<paramref name="len"/> elements).</param>
        /// <param name="start">First data index (may be negative).</param>
        /// <param name="len">Elements.</param>
        private static unsafe void FillEdge<T>(T* a, long n1, T* dst, long start, long len) where T : unmanaged
        {
            long lo = Math.Max(start, 0), hi = Math.Min(start + len, n1);
            new Span<T>(dst, (int)len).Clear();
            if (hi > lo)
                new ReadOnlySpan<T>(a + lo, (int)(hi - lo)).CopyTo(new Span<T>(dst + (lo - start), (int)(hi - lo)));
        }

        /// <summary>Whether every one of <paramref name="n"/> doubles is finite: x − x is 0 for a finite x and NaN for an
        ///     infinity or a NaN, and a NaN survives any sum, so one vector sum answers for the whole array.</summary>
        /// <param name="p">First element.</param><param name="n">Count.</param>
        /// <returns>True when all are finite.</returns>
        private static unsafe bool AllFinite(double* p, long n)
        {
            var acc = Vector256<double>.Zero;
            long i = 0;
            for (; i + 4 <= n; i += 4)
            {
                var v = Vector256.Load(p + i);
                acc += v - v;
            }
            double s = 0;
            for (; i < n; i++)
                s += p[i] - p[i];
            return double.IsFinite(Vector256.Sum(acc) + s);
        }

        /// <summary>Whether every one of <paramref name="n"/> floats is finite (the float twin of the double scan).</summary>
        /// <param name="p">First element.</param><param name="n">Count.</param>
        /// <returns>True when all are finite.</returns>
        private static unsafe bool AllFinite(float* p, long n)
        {
            var acc = Vector256<float>.Zero;
            long i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Load(p + i);
                acc += v - v;
            }
            float s = 0;
            for (; i < n; i++)
                s += p[i] - p[i];
            return float.IsFinite(Vector256.Sum(acc) + s);
        }

        /// <summary>
        ///     Sixteen float64 positions: <c>dst[i] = Σ_s a[q0 + i + s] · k[s]</c> — fused multiply-adds over two accumulator
        ///     sets on alternating s (eight independent chains), the head / tail steps reading the zero-padded edge windows, the
        ///     sets summed at the end: an order OpenBLAS's vector kernel does not share either; these positions are never
        ///     byte-compared without the backend.
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="head">Left edge window (indexed by data index).</param>
        /// <param name="tail">Right edge window (indexed by data index).</param><param name="q0">Lane 0's data offset.</param>
        /// <param name="k">Kernel element 0.</param><param name="sLo">First term of any lane.</param>
        /// <param name="coreLo">First term of every lane.</param><param name="coreHi">End of every lane's terms.</param>
        /// <param name="sHi">End of any lane's terms.</param><param name="dst">Sixteen outputs.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void FmaBlockDouble(double* a, double* head, double* tail, long q0, double* k, long sLo, long coreLo,
            long coreHi, long sHi, double* dst)
        {
            Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector256<double> b0 = default, b1 = default, b2 = default, b3 = default;
            StepsDouble(head + q0, k, sLo, Math.Min(coreLo, sHi), ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            StepsDouble(a + q0, k, coreLo, coreHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            // The tail starts at the core's end — or at its start, when the block is so short that no term is valid for
            // every lane (then the head covered the terms below coreLo).
            StepsDouble(tail + q0, k, Math.Max(coreLo, coreHi), sHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            (a0 + b0).Store(dst);
            (a1 + b1).Store(dst + 4);
            (a2 + b2).Store(dst + 8);
            (a3 + b3).Store(dst + 12);
        }

        /// <summary>
        ///     Terms [<paramref name="from"/>, <paramref name="to"/>) of <see cref="FmaBlockDouble"/>: <c>x[s + i]</c> times
        ///     <c>k[s]</c> into the sixteen lanes, alternate terms into the second accumulator set.
        /// </summary>
        /// <param name="x">Lane 0's element for term 0 (x[s + i] is lane i's term s).</param><param name="k">Kernel element 0.</param>
        /// <param name="from">First term.</param><param name="to">End of the terms.</param>
        /// <param name="a0">Set A, lanes 0-3.</param><param name="a1">Set A, lanes 4-7.</param>
        /// <param name="a2">Set A, lanes 8-11.</param><param name="a3">Set A, lanes 12-15.</param>
        /// <param name="b0">Set B, lanes 0-3.</param><param name="b1">Set B, lanes 4-7.</param>
        /// <param name="b2">Set B, lanes 8-11.</param><param name="b3">Set B, lanes 12-15.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StepsDouble(double* x, double* k, long from, long to,
            ref Vector256<double> a0, ref Vector256<double> a1, ref Vector256<double> a2, ref Vector256<double> a3,
            ref Vector256<double> b0, ref Vector256<double> b1, ref Vector256<double> b2, ref Vector256<double> b3)
        {
            long s = from;
            for (; s + 1 < to; s += 2)
            {
                var k0 = Vector256.Create(k[s]);
                var k1 = Vector256.Create(k[s + 1]);
                double* v = x + s;
                a0 = Fma.MultiplyAdd(Vector256.Load(v), k0, a0);
                a1 = Fma.MultiplyAdd(Vector256.Load(v + 4), k0, a1);
                a2 = Fma.MultiplyAdd(Vector256.Load(v + 8), k0, a2);
                a3 = Fma.MultiplyAdd(Vector256.Load(v + 12), k0, a3);
                b0 = Fma.MultiplyAdd(Vector256.Load(v + 1), k1, b0);
                b1 = Fma.MultiplyAdd(Vector256.Load(v + 5), k1, b1);
                b2 = Fma.MultiplyAdd(Vector256.Load(v + 9), k1, b2);
                b3 = Fma.MultiplyAdd(Vector256.Load(v + 13), k1, b3);
            }
            if (s < to)
            {
                var k0 = Vector256.Create(k[s]);
                double* v = x + s;
                a0 = Fma.MultiplyAdd(Vector256.Load(v), k0, a0);
                a1 = Fma.MultiplyAdd(Vector256.Load(v + 4), k0, a1);
                a2 = Fma.MultiplyAdd(Vector256.Load(v + 8), k0, a2);
                a3 = Fma.MultiplyAdd(Vector256.Load(v + 12), k0, a3);
            }
        }

        /// <summary>
        ///     Thirty-two float32 positions — <see cref="FmaBlockDouble"/> with eight lanes per vector (float32 accumulators,
        ///     as OpenBLAS's sdot vector kernel keeps them).
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="head">Left edge window.</param>
        /// <param name="tail">Right edge window.</param><param name="q0">Lane 0's data offset.</param>
        /// <param name="k">Kernel element 0.</param><param name="sLo">First term of any lane.</param>
        /// <param name="coreLo">First term of every lane.</param><param name="coreHi">End of every lane's terms.</param>
        /// <param name="sHi">End of any lane's terms.</param><param name="dst">Thirty-two outputs.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void FmaBlockSingle(float* a, float* head, float* tail, long q0, float* k, long sLo, long coreLo,
            long coreHi, long sHi, float* dst)
        {
            Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector256<float> b0 = default, b1 = default, b2 = default, b3 = default;
            StepsSingle(head + q0, k, sLo, Math.Min(coreLo, sHi), ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            StepsSingle(a + q0, k, coreLo, coreHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            StepsSingle(tail + q0, k, Math.Max(coreLo, coreHi), sHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            (a0 + b0).Store(dst);
            (a1 + b1).Store(dst + 8);
            (a2 + b2).Store(dst + 16);
            (a3 + b3).Store(dst + 24);
        }

        /// <summary>Terms [<paramref name="from"/>, <paramref name="to"/>) of <see cref="FmaBlockSingle"/> (see
        ///     <see cref="StepsDouble"/>).</summary>
        /// <param name="x">Lane 0's element for term 0.</param><param name="k">Kernel element 0.</param>
        /// <param name="from">First term.</param><param name="to">End of the terms.</param>
        /// <param name="a0">Set A, lanes 0-7.</param><param name="a1">Set A, lanes 8-15.</param>
        /// <param name="a2">Set A, lanes 16-23.</param><param name="a3">Set A, lanes 24-31.</param>
        /// <param name="b0">Set B, lanes 0-7.</param><param name="b1">Set B, lanes 8-15.</param>
        /// <param name="b2">Set B, lanes 16-23.</param><param name="b3">Set B, lanes 24-31.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StepsSingle(float* x, float* k, long from, long to,
            ref Vector256<float> a0, ref Vector256<float> a1, ref Vector256<float> a2, ref Vector256<float> a3,
            ref Vector256<float> b0, ref Vector256<float> b1, ref Vector256<float> b2, ref Vector256<float> b3)
        {
            long s = from;
            for (; s + 1 < to; s += 2)
            {
                var k0 = Vector256.Create(k[s]);
                var k1 = Vector256.Create(k[s + 1]);
                float* v = x + s;
                a0 = Fma.MultiplyAdd(Vector256.Load(v), k0, a0);
                a1 = Fma.MultiplyAdd(Vector256.Load(v + 8), k0, a1);
                a2 = Fma.MultiplyAdd(Vector256.Load(v + 16), k0, a2);
                a3 = Fma.MultiplyAdd(Vector256.Load(v + 24), k0, a3);
                b0 = Fma.MultiplyAdd(Vector256.Load(v + 1), k1, b0);
                b1 = Fma.MultiplyAdd(Vector256.Load(v + 9), k1, b1);
                b2 = Fma.MultiplyAdd(Vector256.Load(v + 17), k1, b2);
                b3 = Fma.MultiplyAdd(Vector256.Load(v + 25), k1, b3);
            }
            if (s < to)
            {
                var k0 = Vector256.Create(k[s]);
                float* v = x + s;
                a0 = Fma.MultiplyAdd(Vector256.Load(v), k0, a0);
                a1 = Fma.MultiplyAdd(Vector256.Load(v + 8), k0, a1);
                a2 = Fma.MultiplyAdd(Vector256.Load(v + 16), k0, a2);
                a3 = Fma.MultiplyAdd(Vector256.Load(v + 24), k0, a3);
            }
        }

        /// <summary>
        ///     Eight complex128 positions, each Vector256 holding two as [re, im, re, im]: per term, the data vector times the
        ///     broadcast real part of k[s] into one accumulator ([re*kr, im*kr]) and times the broadcast imaginary part into a
        ///     second ([re*ki, im*ki]) — eight chains, two fused multiply-adds per two complex products, the machine's peak —
        ///     combined once at the end: the second set's lanes swapped ([im*ki, re*ki]) and add-subtracted from the first,
        ///     [re*kr − im*ki, im*kr + re*ki]. (Swapping the DATA per term instead would add a shuffle per load on the one port
        ///     that executes shuffles.)
        /// </summary>
        /// <param name="a">Data element 0.</param><param name="head">Left edge window.</param>
        /// <param name="tail">Right edge window.</param><param name="q0">Lane 0's data offset.</param>
        /// <param name="k">Kernel element 0.</param><param name="sLo">First term of any lane.</param>
        /// <param name="coreLo">First term of every lane.</param><param name="coreHi">End of every lane's terms.</param>
        /// <param name="sHi">End of any lane's terms.</param><param name="dst">Eight outputs.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void FmaBlockComplex(Complex* a, Complex* head, Complex* tail, long q0, Complex* k, long sLo,
            long coreLo, long coreHi, long sHi, Complex* dst)
        {
            Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default;
            Vector256<double> b0 = default, b1 = default, b2 = default, b3 = default;
            StepsComplex((double*)(head + q0), (double*)k, sLo, Math.Min(coreLo, sHi), ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            StepsComplex((double*)(a + q0), (double*)k, coreLo, coreHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            StepsComplex((double*)(tail + q0), (double*)k, Math.Max(coreLo, coreHi), sHi, ref a0, ref a1, ref a2, ref a3, ref b0, ref b1, ref b2, ref b3);
            double* d = (double*)dst;
            // 0b0101 swaps the two doubles of each 128-bit lane ([re*ki, im*ki] -> [im*ki, re*ki]); addsubpd then takes
            // even lanes a - b (re*kr - im*ki) and odd lanes a + b (im*kr + re*ki).
            Avx.AddSubtract(a0, Avx.Permute(b0, 0b0101)).Store(d);
            Avx.AddSubtract(a1, Avx.Permute(b1, 0b0101)).Store(d + 4);
            Avx.AddSubtract(a2, Avx.Permute(b2, 0b0101)).Store(d + 8);
            Avx.AddSubtract(a3, Avx.Permute(b3, 0b0101)).Store(d + 12);
        }

        /// <summary>Terms [<paramref name="from"/>, <paramref name="to"/>) of <see cref="FmaBlockComplex"/>.</summary>
        /// <param name="x">Lane 0's real part for term 0 (as doubles: term s of complex lane i at x[2(s + i)]).</param>
        /// <param name="ks">Kernel element 0 as doubles.</param>
        /// <param name="from">First term.</param><param name="to">End of the terms.</param>
        /// <param name="a0">Real-part set, complex lanes 0-1.</param><param name="a1">Lanes 2-3.</param>
        /// <param name="a2">Lanes 4-5.</param><param name="a3">Lanes 6-7.</param>
        /// <param name="b0">Imaginary-part set, complex lanes 0-1.</param><param name="b1">Lanes 2-3.</param>
        /// <param name="b2">Lanes 4-5.</param><param name="b3">Lanes 6-7.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StepsComplex(double* x, double* ks, long from, long to,
            ref Vector256<double> a0, ref Vector256<double> a1, ref Vector256<double> a2, ref Vector256<double> a3,
            ref Vector256<double> b0, ref Vector256<double> b1, ref Vector256<double> b2, ref Vector256<double> b3)
        {
            for (long s = from; s < to; s++)
            {
                var kr = Vector256.Create(ks[2 * s]);
                var ki = Vector256.Create(ks[2 * s + 1]);
                double* v = x + 2 * s;
                var v0 = Vector256.Load(v);
                var v1 = Vector256.Load(v + 4);
                var v2 = Vector256.Load(v + 8);
                var v3 = Vector256.Load(v + 12);
                a0 = Fma.MultiplyAdd(v0, kr, a0);
                a1 = Fma.MultiplyAdd(v1, kr, a1);
                a2 = Fma.MultiplyAdd(v2, kr, a2);
                a3 = Fma.MultiplyAdd(v3, kr, a3);
                b0 = Fma.MultiplyAdd(v0, ki, b0);
                b1 = Fma.MultiplyAdd(v1, ki, b1);
                b2 = Fma.MultiplyAdd(v2, ki, b2);
                b3 = Fma.MultiplyAdd(v3, ki, b3);
            }
        }

        /// <summary>
        ///     Thirty-two float32 positions summed EXACTLY as NumPy's HALF_dot sums each one: from +0, term s after term s - 1,
        ///     a rounded float32 product added in float32 — a separate multiply and add per lane (four chains of eight lanes),
        ///     never fused. The head, core and tail steps run in s order, and a padding product (±0) leaves a sum that
        ///     started at +0 unchanged.
        /// </summary>
        /// <param name="a">Data element 0 (float32).</param><param name="head">Left edge window.</param>
        /// <param name="tail">Right edge window.</param><param name="q0">Lane 0's data offset.</param>
        /// <param name="k">Kernel element 0 (float32).</param><param name="sLo">First term of any lane.</param>
        /// <param name="coreLo">First term of every lane.</param><param name="coreHi">End of every lane's terms.</param>
        /// <param name="sHi">End of any lane's terms.</param><param name="dst">Thirty-two float32 sums.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ExactBlockSingle(float* a, float* head, float* tail, long q0, float* k, long sLo, long coreLo,
            long coreHi, long sHi, float* dst)
        {
            Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
            ExactStepsSingle(head + q0, k, sLo, Math.Min(coreLo, sHi), ref a0, ref a1, ref a2, ref a3);
            ExactStepsSingle(a + q0, k, coreLo, coreHi, ref a0, ref a1, ref a2, ref a3);
            ExactStepsSingle(tail + q0, k, Math.Max(coreLo, coreHi), sHi, ref a0, ref a1, ref a2, ref a3);
            a0.Store(dst);
            a1.Store(dst + 8);
            a2.Store(dst + 16);
            a3.Store(dst + 24);
        }

        /// <summary>Terms [<paramref name="from"/>, <paramref name="to"/>) of <see cref="ExactBlockSingle"/>, in s order:
        ///     <c>acc += x[s + i] * k[s]</c>, the product rounded before the add.</summary>
        /// <param name="x">Lane 0's element for term 0.</param><param name="k">Kernel element 0.</param>
        /// <param name="from">First term.</param><param name="to">End of the terms.</param>
        /// <param name="a0">Lanes 0-7.</param><param name="a1">Lanes 8-15.</param>
        /// <param name="a2">Lanes 16-23.</param><param name="a3">Lanes 24-31.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ExactStepsSingle(float* x, float* k, long from, long to,
            ref Vector256<float> a0, ref Vector256<float> a1, ref Vector256<float> a2, ref Vector256<float> a3)
        {
            for (long s = from; s < to; s++)
            {
                var kv = Vector256.Create(k[s]);
                float* v = x + s;
                a0 += Vector256.Load(v) * kv;
                a1 += Vector256.Load(v + 8) * kv;
                a2 += Vector256.Load(v + 16) * kv;
                a3 += Vector256.Load(v + 24) * kv;
            }
        }
    }
}
