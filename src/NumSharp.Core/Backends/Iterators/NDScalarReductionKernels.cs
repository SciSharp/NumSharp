using System;
using System.Numerics;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Iteration
{
    // =========================================================================
    // Scalar (axis=None) Reduction Kernels — Half & Complex
    //
    // These struct kernels implement INDReducingInnerLoop<TAccum> and drive the
    // flat Half/Complex reductions that the IL reduction kernel cannot emit:
    //   • Half  — OpCodes.Bgt/Blt don't apply to the Half struct, and there is
    //             no Vector<Half> arithmetic in the BCL.
    //   • Complex — System.Numerics.Complex has no total ordering; min/max/arg
    //             use a lexicographic (real, then imaginary) compare.
    //
    // They replace the per-element AsIterator fallbacks AND the earlier
    // ForEach(delegate) helpers: the struct-generic ExecuteReducing path is
    // devirtualized + inlined by the JIT (the accumulator stays in a register
    // rather than the per-element memory round-trip the delegate forced), and
    // the contiguous-chunk branches add SIMD (Complex sum) / 4-way unroll
    // (Half min/max). Measured vs the ForEach helpers on 4M elements:
    //   Complex.sum 2.4×, Complex.prod 1.5×, Half.sum 1.6×, Half.max 1.3×.
    //
    // Layout-aware: NDIter (EXTERNAL_LOOP) hands the kernel per-inner-loop byte
    // strides, so contiguous, sliced, broadcast, and transposed arrays all work.
    // The contiguous-chunk fast paths gate on stride == sizeof(element); any
    // other stride takes the scalar branch.
    //
    // ORDER CONTRACT
    //   • sum / prod / min / max  → order-independent value ⇒ NPY_KEEPORDER
    //     (the cache-friendly default of NDIterRef.New). For Complex min/max
    //     this also matches NumPy's reduce, which propagates the FIRST NaN in
    //     MEMORY order (not C order) — see ComplexMinMaxAccumulator.
    //   • argmin / argmax         → the returned flat index must be the C-order
    //     first occurrence ⇒ callers MUST build the iterator with NPY_CORDER so
    //     the running index the kernel keeps is the C-order position.
    //
    // Call pattern:
    //     using var iter = NDIterRef.New(arr, NDIterGlobalFlags.EXTERNAL_LOOP);
    //     double s = iter.ExecuteReducing<HalfSumKernel, double>(default, 0.0);
    // =========================================================================

    // -------------------------------------------------------------------------
    // Accumulators
    // -------------------------------------------------------------------------

    /// <summary>
    /// Min/Max accumulator for Half: running extremum (held in double, the
    /// precision the codebase's f16 reductions already use), a "seen any value"
    /// flag for the empty/first-element guard, and a NaN flag. Any NaN ⇒ result
    /// NaN (NumPy: min/max with NaN propagates), so the kernel aborts on the
    /// first NaN it sees.
    /// </summary>
    public struct HalfMinMaxAccumulator
    {
        public double Best;
        public bool Seen;
        public bool SawNaN;
    }

    /// <summary>
    /// Min/Max accumulator for Complex: the running lexicographic extremum, a
    /// "seen any value" flag, and a NaN flag. On the first element whose real OR
    /// imaginary part is NaN the kernel stores that element VERBATIM in
    /// <see cref="Best"/> and aborts — matching NumPy's minimum/maximum, which
    /// return the NaN-bearing operand as-is (e.g. min([1+1j, nan+0j]) → (nan,0),
    /// not (nan,nan)). Because the iterator runs in NPY_KEEPORDER, the element
    /// captured is the first NaN in MEMORY order, which is exactly what NumPy's
    /// reduce returns for a non-C-contiguous (e.g. transposed) array.
    /// </summary>
    public struct ComplexMinMaxAccumulator
    {
        public Complex Best;
        public bool Seen;
        public bool SawNaN;
    }

    /// <summary>
    /// ArgMin/ArgMax accumulator for Half. <see cref="Cur"/> is the running
    /// C-order flat index of the chunk's first element (callers MUST use
    /// NPY_CORDER). <see cref="BestIdx"/> starts at -1 as the "no value yet"
    /// sentinel. <see cref="SawNaNIdx"/> is the flat index of the first NaN
    /// (NumPy: argmin/argmax of an array containing NaN return the first NaN's
    /// index); -1 until a NaN is seen.
    /// </summary>
    public struct HalfArgAccumulator
    {
        public double Best;
        public long BestIdx;
        public long Cur;
        public long SawNaNIdx;
    }

    /// <summary>
    /// ArgMin/ArgMax accumulator for Complex — lexicographic best plus the same
    /// running-index / NaN-index bookkeeping as <see cref="HalfArgAccumulator"/>.
    /// </summary>
    public struct ComplexArgAccumulator
    {
        public Complex Best;
        public long BestIdx;
        public long Cur;
        public long SawNaNIdx;
    }

    // -------------------------------------------------------------------------
    // Sum / Prod — Half accumulates in double, then the caller narrows to Half
    // (a large sum saturates to ±inf exactly like NumPy's float16 reduce).
    // -------------------------------------------------------------------------

    public readonly struct HalfSumKernel : INDReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double acc = sum;
            for (long i = 0; i < count; i++)
                acc += (double)*(Half*)(p + i * stride);
            sum = acc;
            return true;
        }
    }

    public readonly struct HalfProdKernel : INDReducingInnerLoop<double>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref double prod)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double acc = prod;
            for (long i = 0; i < count; i++)
                acc *= (double)*(Half*)(p + i * stride);
            prod = acc;
            return true;
        }
    }

    /// <summary>
    /// Complex sum. When the inner loop is contiguous (stride == 16 bytes = one
    /// Complex) and Vector256 is hardware-accelerated, the chunk is summed as a
    /// flat double stream with two Vector256&lt;double&gt; lanes (real/imag
    /// interleaved survive the lane reduction), then the tail is added scalar.
    /// Non-contiguous chunks add scalar. The SIMD reassociation differs from a
    /// strict left fold only at ULP level (same class as the codebase's pairwise
    /// reductions).
    /// </summary>
    public readonly struct ComplexSumKernel : INDReducingInnerLoop<Complex>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref Complex sum)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            if (stride == 16 && Vector256.IsHardwareAccelerated)
            {
                double* d = (double*)p;
                long m = count * 2;
                Vector256<double> v0 = Vector256<double>.Zero, v1 = Vector256<double>.Zero;
                long i = 0;
                for (; i + 8 <= m; i += 8)
                {
                    v0 += Vector256.Load(d + i);
                    v1 += Vector256.Load(d + i + 4);
                }
                var v = v0 + v1;
                double re = sum.Real + v.GetElement(0) + v.GetElement(2);
                double im = sum.Imaginary + v.GetElement(1) + v.GetElement(3);
                for (; i < m; i += 2) { re += d[i]; im += d[i + 1]; }
                sum = new Complex(re, im);
            }
            else
            {
                Complex acc = sum;
                for (long i = 0; i < count; i++)
                    acc += *(Complex*)(p + i * stride);
                sum = acc;
            }
            return true;
        }
    }

    /// <summary>
    /// Complex product. The cross-term multiply (a+bi)(c+di) cannot be expressed
    /// as an independent-lane SIMD reduction, so this is a scalar fold; the win
    /// over the delegate path is devirtualization + a register-held accumulator.
    /// </summary>
    public readonly struct ComplexProdKernel : INDReducingInnerLoop<Complex>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref Complex prod)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex acc = prod;
            for (long i = 0; i < count; i++)
                acc *= *(Complex*)(p + i * stride);
            prod = acc;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Min / Max — Half (4-way unrolled when contiguous), Complex (lexicographic)
    // -------------------------------------------------------------------------

    /// <summary>
    /// Half max. Contiguous chunks (stride == 2) run a 4-accumulator unroll that
    /// breaks the per-element dependency chain (~1.3× the scalar fold); other
    /// strides take the scalar branch. NaN propagates: the kernel aborts the
    /// moment a NaN is seen so the caller returns Half.NaN.
    /// </summary>
    public readonly struct HalfMaxKernel : INDReducingInnerLoop<HalfMinMaxAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref HalfMinMaxAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double best = a.Seen ? a.Best : double.NegativeInfinity;
            bool seen = a.Seen;
            if (stride == 2)
            {
                Half* h = (Half*)p;
                double b0 = best, b1 = double.NegativeInfinity, b2 = b1, b3 = b1;
                bool nan = false;
                long i = 0;
                for (; i + 4 <= count; i += 4)
                {
                    double v0 = (double)h[i], v1 = (double)h[i + 1], v2 = (double)h[i + 2], v3 = (double)h[i + 3];
                    nan |= double.IsNaN(v0) | double.IsNaN(v1) | double.IsNaN(v2) | double.IsNaN(v3);
                    if (v0 > b0) b0 = v0;
                    if (v1 > b1) b1 = v1;
                    if (v2 > b2) b2 = v2;
                    if (v3 > b3) b3 = v3;
                }
                best = Math.Max(Math.Max(b0, b1), Math.Max(b2, b3));
                seen |= count > 0;
                for (; i < count; i++)
                {
                    double v = (double)h[i];
                    if (double.IsNaN(v)) { a.SawNaN = true; a.Best = best; a.Seen = true; return false; }
                    if (v > best) best = v;
                }
                if (nan) { a.SawNaN = true; a.Best = best; a.Seen = seen; return false; }
            }
            else
            {
                for (long i = 0; i < count; i++)
                {
                    double v = (double)*(Half*)(p + i * stride);
                    if (double.IsNaN(v)) { a.SawNaN = true; a.Seen = true; return false; }
                    if (!seen || v > best) { best = v; seen = true; }
                }
            }
            a.Best = best;
            a.Seen = seen;
            return true;
        }
    }

    /// <summary>
    /// Half min — mirror of <see cref="HalfMaxKernel"/> with the comparison and
    /// the unroll seeds inverted (+inf).
    /// </summary>
    public readonly struct HalfMinKernel : INDReducingInnerLoop<HalfMinMaxAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref HalfMinMaxAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double best = a.Seen ? a.Best : double.PositiveInfinity;
            bool seen = a.Seen;
            if (stride == 2)
            {
                Half* h = (Half*)p;
                double b0 = best, b1 = double.PositiveInfinity, b2 = b1, b3 = b1;
                bool nan = false;
                long i = 0;
                for (; i + 4 <= count; i += 4)
                {
                    double v0 = (double)h[i], v1 = (double)h[i + 1], v2 = (double)h[i + 2], v3 = (double)h[i + 3];
                    nan |= double.IsNaN(v0) | double.IsNaN(v1) | double.IsNaN(v2) | double.IsNaN(v3);
                    if (v0 < b0) b0 = v0;
                    if (v1 < b1) b1 = v1;
                    if (v2 < b2) b2 = v2;
                    if (v3 < b3) b3 = v3;
                }
                best = Math.Min(Math.Min(b0, b1), Math.Min(b2, b3));
                seen |= count > 0;
                for (; i < count; i++)
                {
                    double v = (double)h[i];
                    if (double.IsNaN(v)) { a.SawNaN = true; a.Best = best; a.Seen = true; return false; }
                    if (v < best) best = v;
                }
                if (nan) { a.SawNaN = true; a.Best = best; a.Seen = seen; return false; }
            }
            else
            {
                for (long i = 0; i < count; i++)
                {
                    double v = (double)*(Half*)(p + i * stride);
                    if (double.IsNaN(v)) { a.SawNaN = true; a.Seen = true; return false; }
                    if (!seen || v < best) { best = v; seen = true; }
                }
            }
            a.Best = best;
            a.Seen = seen;
            return true;
        }
    }

    /// <summary>
    /// Complex max via lexicographic (real, then imaginary) compare. On the
    /// first NaN-bearing element the kernel stores it verbatim and aborts (see
    /// <see cref="ComplexMinMaxAccumulator"/>).
    /// </summary>
    public readonly struct ComplexMaxKernel : INDReducingInnerLoop<ComplexMinMaxAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref ComplexMinMaxAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex best = a.Best;
            bool seen = a.Seen;
            for (long i = 0; i < count; i++)
            {
                Complex v = *(Complex*)(p + i * stride);
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary))
                {
                    a.Best = v;
                    a.SawNaN = true;
                    a.Seen = true;
                    return false;
                }
                if (!seen || v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary))
                {
                    best = v;
                    seen = true;
                }
            }
            a.Best = best;
            a.Seen = seen;
            return true;
        }
    }

    public readonly struct ComplexMinKernel : INDReducingInnerLoop<ComplexMinMaxAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref ComplexMinMaxAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex best = a.Best;
            bool seen = a.Seen;
            for (long i = 0; i < count; i++)
            {
                Complex v = *(Complex*)(p + i * stride);
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary))
                {
                    a.Best = v;
                    a.SawNaN = true;
                    a.Seen = true;
                    return false;
                }
                if (!seen || v.Real < best.Real || (v.Real == best.Real && v.Imaginary < best.Imaginary))
                {
                    best = v;
                    seen = true;
                }
            }
            a.Best = best;
            a.Seen = seen;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // ArgMin / ArgMax — running C-order index (callers MUST use NPY_CORDER).
    // First occurrence wins (strict compare); first NaN's index propagates.
    // -------------------------------------------------------------------------

    public readonly struct HalfArgMaxKernel : INDReducingInnerLoop<HalfArgAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref HalfArgAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double best = a.Best;
            long bi = a.BestIdx;
            long cur = a.Cur;
            for (long i = 0; i < count; i++)
            {
                double v = (double)*(Half*)(p + i * stride);
                if (double.IsNaN(v)) { a.SawNaNIdx = cur + i; return false; }
                if (bi < 0 || v > best) { best = v; bi = cur + i; }
            }
            a.Best = best;
            a.BestIdx = bi;
            a.Cur = cur + count;
            return true;
        }
    }

    public readonly struct HalfArgMinKernel : INDReducingInnerLoop<HalfArgAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref HalfArgAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            double best = a.Best;
            long bi = a.BestIdx;
            long cur = a.Cur;
            for (long i = 0; i < count; i++)
            {
                double v = (double)*(Half*)(p + i * stride);
                if (double.IsNaN(v)) { a.SawNaNIdx = cur + i; return false; }
                if (bi < 0 || v < best) { best = v; bi = cur + i; }
            }
            a.Best = best;
            a.BestIdx = bi;
            a.Cur = cur + count;
            return true;
        }
    }

    public readonly struct ComplexArgMaxKernel : INDReducingInnerLoop<ComplexArgAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref ComplexArgAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex best = a.Best;
            long bi = a.BestIdx;
            long cur = a.Cur;
            for (long i = 0; i < count; i++)
            {
                Complex v = *(Complex*)(p + i * stride);
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) { a.SawNaNIdx = cur + i; return false; }
                if (bi < 0 || v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary))
                {
                    best = v;
                    bi = cur + i;
                }
            }
            a.Best = best;
            a.BestIdx = bi;
            a.Cur = cur + count;
            return true;
        }
    }

    public readonly struct ComplexArgMinKernel : INDReducingInnerLoop<ComplexArgAccumulator>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref ComplexArgAccumulator a)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            Complex best = a.Best;
            long bi = a.BestIdx;
            long cur = a.Cur;
            for (long i = 0; i < count; i++)
            {
                Complex v = *(Complex*)(p + i * stride);
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) { a.SawNaNIdx = cur + i; return false; }
                if (bi < 0 || v.Real < best.Real || (v.Real == best.Real && v.Imaginary < best.Imaginary))
                {
                    best = v;
                    bi = cur + i;
                }
            }
            a.Best = best;
            a.BestIdx = bi;
            a.Cur = cur + count;
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Any / All — early-exit predicate kernels for the non-contiguous Half/Complex
    // paths (contiguous arrays keep their direct-pointer scan). Any: truthy is
    // "!= 0" (NaN is truthy). All: falsy is "== 0" (NaN is truthy, never breaks
    // All). The bool accumulator is the result; the kernel aborts on the first
    // decisive element (Any → found, All → a zero).
    // -------------------------------------------------------------------------

    public readonly struct HalfAnyKernel : INDReducingInnerLoop<bool>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref bool found)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
                if (*(Half*)(p + i * stride) != Half.Zero) { found = true; return false; }
            return true;
        }
    }

    public readonly struct ComplexAnyKernel : INDReducingInnerLoop<bool>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref bool found)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
                if (*(Complex*)(p + i * stride) != Complex.Zero) { found = true; return false; }
            return true;
        }
    }

    public readonly struct HalfAllKernel : INDReducingInnerLoop<bool>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref bool allTrue)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
                if (*(Half*)(p + i * stride) == Half.Zero) { allTrue = false; return false; }
            return true;
        }
    }

    public readonly struct ComplexAllKernel : INDReducingInnerLoop<bool>
    {
        public unsafe bool Execute(void** dataptrs, long* strides, long count, ref bool allTrue)
        {
            byte* p = (byte*)dataptrs[0];
            long stride = strides[0];
            for (long i = 0; i < count; i++)
                if (*(Complex*)(p + i * stride) == Complex.Zero) { allTrue = false; return false; }
            return true;
        }
    }
}
