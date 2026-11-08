using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Comparison.Complex.cs — complex128 comparisons
// (==, !=, <, <=, >, >=) with 256-bit vectors
// =============================================================================
//
// NumPy orders complex numbers lexicographically, with a NaN rule of its own
// (numpy/_core/src/umath/loops.c.src):
//
//     CLT(x, y) = (xr <  yr && !isnan(xi) && !isnan(yi)) || (xr == yr && xi <  yi)
//     CLE(x, y) = (xr <  yr && !isnan(xi) && !isnan(yi)) || (xr == yr && xi <= yi)
//     CGT(x, y) = (xr >  yr && !isnan(xi) && !isnan(yi)) || (xr == yr && xi >  yi)
//     CGE(x, y) = (xr >  yr && !isnan(xi) && !isnan(yi)) || (xr == yr && xi >= yi)
//     CEQ(x, y) =  xr == yr && xi == yi
//     CNE(x, y) =  xr != yr || xi != yi
//
// NumPy runs these one element at a time (CDOUBLE_less and friends are plain
// BINARY_LOOPs: 0.8 - 1.5 ns an element measured on 2.4.2). A complex128 is two
// doubles, [re, im], so a 256-bit vector holds two elements.
//
// EQUALITY (== and !=)
//   One ordered-equal compare of the whole vector gives four bits
//   (re0, im0, re1, im1); an element is equal when both of its bits are set.
//   That "and" is done on the mask bits of four vectors at once, with integer
//   instructions. CNE is exactly !CEQ (also with NaN), so != inverts the bits.
//
// ORDER (<, <=, >, >=)
//   The rule mixes compares of the real parts with compares of the imaginary
//   parts, so two vectors (four elements) are de-interleaved first into one
//   vector of real parts and one of imaginary parts (vunpcklpd / vunpckhpd; the
//   lane order they give, 0 2 1 3, is the same for both operands and is put
//   right on the four mask bits). The rule is then lane-wise:
//
//       less = (re < re  &  im ordered)  |  (re == re  &  im < im)
//
//   CGT(x, y) is CLT(y, x) and CGE(x, y) is CLE(y, x), term for term, so the two
//   "greater" operators run the "less" loop with the operands exchanged.
//
//   Measured, 1,000 elements in L1 (ns an element): this form 0.59, the scalar
//   rule 1.34. A form that kept the elements interleaved and swapped the
//   imaginary results onto the real lanes (three compares on all four lanes of
//   every vector) measured 0.85.
//
// AN OPERAND THAT IS ONE VALUE is read from a four-double buffer holding it
// twice, with an element step of 0, so the loops have no form of their own for
// the array-against-value cases.
//
// Hosts without AVX (and the last 0-7 elements) run the scalar rule above,
// which is also the reference the unit tests hold the vector loops to.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     The contiguous complex128 comparison kernel for one operator and one operand form, or null for a form
        ///     this file does not serve (the strided forms, which keep the emitted per-element loop).
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <param name="path">
        ///     <see cref="ExecutionPath.SimdFull"/> (both operands are contiguous runs),
        ///     <see cref="ExecutionPath.SimdScalarRight"/> (the right operand is one element) or
        ///     <see cref="ExecutionPath.SimdScalarLeft"/> (the left operand is one element).
        /// </param>
        /// <returns>The kernel, or null. The kernel ignores its stride, shape and ndim arguments: it reads
        ///     <c>totalSize</c> elements from each contiguous operand (one from an operand that is a value) and writes
        ///     <c>totalSize</c> bytes.</returns>
        /// <remarks>
        ///     The comparison cache keeps one delegate per key, so each lambda is built once. The "greater" operators
        ///     pass the operands in exchanged order (see the file header).
        /// </remarks>
        internal static unsafe ComparisonKernel TryGetComplexComparisonKernel(ComparisonOp op, ExecutionPath path)
        {
            bool lValue, rValue;
            switch (path)
            {
                case ExecutionPath.SimdFull: lValue = false; rValue = false; break;
                case ExecutionPath.SimdScalarRight: lValue = false; rValue = true; break;
                case ExecutionPath.SimdScalarLeft: lValue = true; rValue = false; break;
                default: return null;
            }

            switch (op)
            {
                case ComparisonOp.Equal:
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexEquality((double*)l, lValue, (double*)r, rValue, (byte*)res, n, negate: false);
                case ComparisonOp.NotEqual:
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexEquality((double*)l, lValue, (double*)r, rValue, (byte*)res, n, negate: true);
                case ComparisonOp.Less:
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexOrder((double*)l, lValue, (double*)r, rValue, (byte*)res, n, inclusive: false);
                case ComparisonOp.LessEqual:
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexOrder((double*)l, lValue, (double*)r, rValue, (byte*)res, n, inclusive: true);
                case ComparisonOp.Greater:        // a > b  is  b < a
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexOrder((double*)r, rValue, (double*)l, lValue, (byte*)res, n, inclusive: false);
                default:                          // a >= b  is  b <= a
                    return (l, r, res, ls, rs, sh, nd, n) => ComplexOrder((double*)r, rValue, (double*)l, lValue, (byte*)res, n, inclusive: true);
            }
        }

        /// <summary>
        ///     NumPy's comparison of two complex128 values (the <c>CLT</c> / <c>CLE</c> / <c>CGT</c> / <c>CGE</c> /
        ///     <c>CEQ</c> / <c>CNE</c> macros): lexicographic by real part then imaginary part, with NumPy's NaN rule.
        /// </summary>
        /// <param name="ar">Real part of the left value.</param>
        /// <param name="ai">Imaginary part of the left value.</param>
        /// <param name="br">Real part of the right value.</param>
        /// <param name="bi">Imaginary part of the right value.</param>
        /// <param name="op">The comparison.</param>
        /// <returns>The result of <c>a op b</c>.</returns>
        /// <remarks>
        ///     For the four ordered operators a NaN in ANY of the four parts makes the result False: when the real
        ///     parts order the pair the imaginary parts must both be numbers, and when they are equal the imaginary
        ///     compare decides. So <c>(1, nan) &lt; (2, 0)</c> is False although 1 &lt; 2.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ComplexCompareScalarRule(double ar, double ai, double br, double bi, ComparisonOp op)
        {
            switch (op)
            {
                case ComparisonOp.Equal:
                    return ar == br && ai == bi;
                case ComparisonOp.NotEqual:
                    return ar != br || ai != bi;
                case ComparisonOp.Less:
                    return (ar < br && !double.IsNaN(ai) && !double.IsNaN(bi)) || (ar == br && ai < bi);
                case ComparisonOp.LessEqual:
                    return (ar < br && !double.IsNaN(ai) && !double.IsNaN(bi)) || (ar == br && ai <= bi);
                case ComparisonOp.Greater:
                    return (ar > br && !double.IsNaN(ai) && !double.IsNaN(bi)) || (ar == br && ai > bi);
                default: // ComparisonOp.GreaterEqual
                    return (ar > br && !double.IsNaN(ai) && !double.IsNaN(bi)) || (ar == br && ai >= bi);
            }
        }

        /// <summary>
        ///     <c>a &lt; b</c> (or <c>a &lt;= b</c>) for four complex128 elements of each operand, as four bits: bit k
        ///     is the answer of element k.
        /// </summary>
        /// <param name="a">Left operand: 8 doubles (4 elements), or the value's pair buffer when the step is 0.</param>
        /// <param name="aStep">Doubles per left element: 2, or 0 for one value.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="bStep">Doubles per right element: 2, or 0 for one value.</param>
        /// <param name="inclusive">True for <c>&lt;=</c>. Every call site passes a constant, so the inlined body holds
        ///     one imaginary compare and no test.</param>
        /// <returns>The four answer bits.</returns>
        /// <remarks>
        ///     The unpacks give the real parts as <c>[re0, re2, re1, re3]</c> and the imaginary parts in the same
        ///     order, so the mask's bits 1 and 2 are exchanged at the end. Every compare is the ORDERED form (False
        ///     when a lane holds a NaN).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ComplexLessFour(double* a, long aStep, double* b, long bStep, bool inclusive)
        {
            var a0 = Avx.LoadVector256(a);
            var a1 = Avx.LoadVector256(a + 2 * aStep);
            var b0 = Avx.LoadVector256(b);
            var b1 = Avx.LoadVector256(b + 2 * bStep);
            var ar = Avx.UnpackLow(a0, a1);
            var ai = Avx.UnpackHigh(a0, a1);
            var br = Avx.UnpackLow(b0, b1);
            var bi = Avx.UnpackHigh(b0, b1);
            var imaginary = inclusive ? Avx.CompareLessThanOrEqual(ai, bi) : Avx.CompareLessThan(ai, bi);
            // (re < re and both imaginary parts are numbers) or (re == re and the imaginary parts decide).
            var mask = Avx.Or(
                Avx.And(Avx.CompareLessThan(ar, br), Avx.CompareOrdered(ai, bi)),
                Avx.And(Avx.CompareEqual(ar, br), imaginary));
            uint k = (uint)Avx.MoveMask(mask);                          // answers of elements 0, 2, 1, 3
            return (k & 0b1001) | ((k & 0b0010) << 1) | ((k >> 1) & 0b0010);
        }

        /// <summary>
        ///     <c>a == b</c> for four complex128 elements of each operand, as four bits: bit k is the answer of
        ///     element k.
        /// </summary>
        /// <param name="a">Left operand: 8 doubles (4 elements), or the value's pair buffer when the step is 0.</param>
        /// <param name="aStep">Doubles per left element: 2, or 0 for one value.</param>
        /// <param name="b">Right operand.</param>
        /// <param name="bStep">Doubles per right element: 2, or 0 for one value.</param>
        /// <returns>The four answer bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint ComplexEqualFour(double* a, long aStep, double* b, long bStep)
        {
            // One bit per double: (re, im) of elements 0 and 1, then of elements 2 and 3.
            uint low = (uint)Avx.MoveMask(Avx.CompareEqual(Avx.LoadVector256(a), Avx.LoadVector256(b)));
            uint high = (uint)Avx.MoveMask(Avx.CompareEqual(Avx.LoadVector256(a + 2 * aStep), Avx.LoadVector256(b + 2 * bStep)));
            uint parts = low | (high << 4);
            // An element is equal when its real bit (even position) and its imaginary bit (the next one) are both set.
            return CompressEvenBits(parts & (parts >> 1));
        }

        /// <summary>The bits at the even positions 0, 2, 4, 6 of <paramref name="x"/>, packed into bits 0 to 3.</summary>
        /// <param name="x">The source bits (only the low 8 are read).</param>
        /// <returns>The four packed bits.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static uint CompressEvenBits(uint x)
        {
            if (Bmi2.IsSupported)
                return Bmi2.ParallelBitExtract(x, 0x55);
            x &= 0x55;
            x = (x | (x >> 1)) & 0x33;
            return (x | (x >> 2)) & 0x0F;
        }

        /// <summary>
        ///     Turns the low 8 bits of <paramref name="bits"/> into 8 bytes, bit k into byte k, each 0 or 1.
        /// </summary>
        /// <param name="bits">The bits (only the low 8 are read).</param>
        /// <returns>The 8 bytes as a little-endian <c>ulong</c>.</returns>
        /// <remarks>
        ///     With BMI2: one PDEP into the low bit of each byte. Without: <see cref="SpreadBitsToBytesPortable"/>.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong SpreadBitsToBytes(ulong bits)
        {
            if (Bmi2.X64.IsSupported)
                return Bmi2.X64.ParallelBitDeposit(bits, 0x0101010101010101UL);
            return SpreadBitsToBytesPortable(bits);
        }

        /// <summary>
        ///     <see cref="SpreadBitsToBytes"/> without BMI2 (an AVX host older than Haswell, or BMI2 switched off):
        ///     bit k of the low byte into byte k, each 0 or 1, by integer arithmetic.
        /// </summary>
        /// <param name="bits">The bits (only the low 8 are read).</param>
        /// <returns>The 8 bytes as a little-endian <c>ulong</c>.</returns>
        /// <remarks>
        ///     The multiply by <c>0x0101010101010101</c> copies the byte into all eight bytes; the mask
        ///     <c>0x8040201008040201</c> keeps bit k of copy k, so byte k holds 0 or <c>1 &lt;&lt; k</c>. Adding 0x7F to
        ///     each byte sets its top bit exactly when the byte is not zero (the largest byte, 0x80, gives 0xFF: no
        ///     carry leaves a byte), and the shift by 7 moves that top bit to the byte's low bit. A multiply that
        ///     places bit k directly (the "0x8040201008040201 then mask 0x80…" form) writes the bytes in REVERSE order,
        ///     which is why this is not that form.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ulong SpreadBitsToBytesPortable(ulong bits)
        {
            ulong onePerByte = ((bits & 0xFF) * 0x0101010101010101UL) & 0x8040201008040201UL;
            return ((onePerByte + 0x7F7F7F7F7F7F7F7FUL) >> 7) & 0x0101010101010101UL;
        }

        /// <summary>
        ///     <c>a &lt; b</c> (or <c>a &lt;= b</c>) over <paramref name="n"/> complex128 elements, each operand a
        ///     contiguous run or one value.
        /// </summary>
        /// <param name="pa">Left operand: 2n doubles, or 2 doubles when <paramref name="aIsValue"/>.</param>
        /// <param name="aIsValue">The left operand is one value.</param>
        /// <param name="pb">Right operand: 2n doubles, or 2 doubles when <paramref name="bIsValue"/>.</param>
        /// <param name="bIsValue">The right operand is one value.</param>
        /// <param name="pr">Result: n bytes, each written 0 or 1.</param>
        /// <param name="n">Element count.</param>
        /// <param name="inclusive">True for <c>&lt;=</c>.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ComplexOrder(double* pa, bool aIsValue, double* pb, bool bIsValue, byte* pr, long n, bool inclusive)
        {
            // An operand that is one value is read from a buffer holding it twice, at an element step of 0.
            double* aPair = stackalloc double[4];
            double* bPair = stackalloc double[4];
            long aStep = 2, bStep = 2;
            if (aIsValue)
            {
                aPair[0] = aPair[2] = pa[0];
                aPair[1] = aPair[3] = pa[1];
                pa = aPair;
                aStep = 0;
            }
            if (bIsValue)
            {
                bPair[0] = bPair[2] = pb[0];
                bPair[1] = bPair[3] = pb[1];
                pb = bPair;
                bStep = 0;
            }

            long i = 0;
            if (Avx.IsSupported)
            {
                // 8 elements per pass (two groups of four, one 8-byte store); the loop stops while a whole pass
                // still fits, so it never reads or writes past element n. The two loops differ only in the constant
                // they pass, which the inlined body folds into its one imaginary compare.
                if (inclusive)
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        double* a = pa + i * aStep, b = pb + i * bStep;
                        uint bits = ComplexLessFour(a, aStep, b, bStep, inclusive: true)
                                    | (ComplexLessFour(a + 4 * aStep, aStep, b + 4 * bStep, bStep, inclusive: true) << 4);
                        *(ulong*)(pr + i) = SpreadBitsToBytes(bits);
                    }
                }
                else
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        double* a = pa + i * aStep, b = pb + i * bStep;
                        uint bits = ComplexLessFour(a, aStep, b, bStep, inclusive: false)
                                    | (ComplexLessFour(a + 4 * aStep, aStep, b + 4 * bStep, bStep, inclusive: false) << 4);
                        *(ulong*)(pr + i) = SpreadBitsToBytes(bits);
                    }
                }
            }

            ComparisonOp rule = inclusive ? ComparisonOp.LessEqual : ComparisonOp.Less;
            for (; i < n; i++)
            {
                double* a = pa + i * aStep, b = pb + i * bStep;
                pr[i] = ComplexCompareScalarRule(a[0], a[1], b[0], b[1], rule) ? (byte)1 : (byte)0;
            }
        }

        /// <summary>
        ///     <c>a == b</c> (or <c>a != b</c>) over <paramref name="n"/> complex128 elements, each operand a
        ///     contiguous run or one value.
        /// </summary>
        /// <param name="pa">Left operand: 2n doubles, or 2 doubles when <paramref name="aIsValue"/>.</param>
        /// <param name="aIsValue">The left operand is one value.</param>
        /// <param name="pb">Right operand: 2n doubles, or 2 doubles when <paramref name="bIsValue"/>.</param>
        /// <param name="bIsValue">The right operand is one value.</param>
        /// <param name="pr">Result: n bytes, each written 0 or 1.</param>
        /// <param name="n">Element count.</param>
        /// <param name="negate">True for <c>!=</c>.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ComplexEquality(double* pa, bool aIsValue, double* pb, bool bIsValue, byte* pr, long n, bool negate)
        {
            double* aPair = stackalloc double[4];
            double* bPair = stackalloc double[4];
            long aStep = 2, bStep = 2;
            if (aIsValue)
            {
                aPair[0] = aPair[2] = pa[0];
                aPair[1] = aPair[3] = pa[1];
                pa = aPair;
                aStep = 0;
            }
            if (bIsValue)
            {
                bPair[0] = bPair[2] = pb[0];
                bPair[1] = bPair[3] = pb[1];
                pb = bPair;
                bStep = 0;
            }

            long i = 0;
            if (Avx.IsSupported)
            {
                // != is the complement of ==: the eight answer bits are flipped before they become bytes.
                uint flip = negate ? 0xFFu : 0u;
                for (; i + 8 <= n; i += 8)
                {
                    double* a = pa + i * aStep, b = pb + i * bStep;
                    uint bits = ComplexEqualFour(a, aStep, b, bStep) | (ComplexEqualFour(a + 4 * aStep, aStep, b + 4 * bStep, bStep) << 4);
                    *(ulong*)(pr + i) = SpreadBitsToBytes(bits ^ flip);
                }
            }

            ComparisonOp rule = negate ? ComparisonOp.NotEqual : ComparisonOp.Equal;
            for (; i < n; i++)
            {
                double* a = pa + i * aStep, b = pb + i * bStep;
                pr[i] = ComplexCompareScalarRule(a[0], a[1], b[0], b[1], rule) ? (byte)1 : (byte)0;
            }
        }
    }
}
