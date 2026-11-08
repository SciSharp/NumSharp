using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Comparison.MixedSign.cs — int64 against uint64, exactly
// =============================================================================
//
// NumPy promotes int64 with uint64 to float64, but its comparison type resolver
// (PyUFunc_SimpleBinaryComparisonTypeResolver, ufunc_type_resolution.c) refuses that
// for two integer operands: it selects the dedicated (LONGLONG, ULONGLONG) loops
// `qQ` / `Qq`, which compare the two integers EXACTLY:
//
//     LONGLONG_Qq_bool_less:  in1 < 0 ? true : (npy_ulonglong)in1 < in2      (and so on)
//
// float64 cannot do it (2^63 and 2^63 - 1 round to the same double). NumSharp
// compared the pair in System.Decimal, which is exact and 30 times slower than a
// machine compare (390 us for 100,000 elements where NumPy takes 62).
//
// The exact rule needs no wider type. With s the signed value and u the unsigned,
// three facts decide all six operators:
//
//     below        =  s < 0  ||  (ulong)s <  u
//     belowOrSame  =  s < 0  ||  (ulong)s <= u
//     same         =  s >= 0 &&  (ulong)s == u
//
//     s <  u : below            s >= u : !below
//     s <= u : belowOrSame      s >  u : !belowOrSame
//     s == u : same             s != u : !same
//
// Each fact is a few 64-bit vector operations: `s < 0` is a signed compare with
// zero, an unsigned compare flips both sign bits (x ^ 2^63) and compares signed
// (AVX2 has no unsigned 64-bit compare), equality compares the bits. There is one
// loop per fact; the negation is one XOR on the eight answer bits of a pass.
//
// A key whose LEFT operand is the unsigned one (`u op s`) is the same question with
// the operands swapped and the operator mirrored (`s mirror(op) u`). An operand
// that is ONE value is read from a four-element buffer holding it, at an element
// step of 0. So the loops serve both orders and all three operand forms.
//
// Hosts without AVX2 (and the last 0-7 elements) run the scalar rule, which is also
// the reference the unit tests hold the vector loops to.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     The contiguous kernel that compares an int64 operand with a uint64 operand exactly, for one operator
        ///     and one operand form; null for a form this file does not serve (the strided forms keep the emitted
        ///     per-element loop, which compares in <see cref="decimal"/>).
        /// </summary>
        /// <param name="lhsSigned">True for the key (Int64, UInt64), false for (UInt64, Int64).</param>
        /// <param name="op">The comparison, as written: <c>lhs op rhs</c>.</param>
        /// <param name="path">
        ///     <see cref="ExecutionPath.SimdFull"/> (both operands are contiguous runs),
        ///     <see cref="ExecutionPath.SimdScalarRight"/> (the right operand is one element) or
        ///     <see cref="ExecutionPath.SimdScalarLeft"/> (the left operand is one element).
        /// </param>
        /// <returns>The kernel, or null. The kernel ignores its stride, shape and ndim arguments.</returns>
        /// <remarks>
        ///     The loops take their operands as (signed, unsigned). For an (unsigned, signed) key the lambda swaps the
        ///     operands and the operator is mirrored once, here (<c>u &lt; s</c> is <c>s &gt; u</c>).
        /// </remarks>
        internal static unsafe ComparisonKernel TryGetMixedSignComparisonKernel(bool lhsSigned, ComparisonOp op, ExecutionPath path)
        {
            bool lValue, rValue;
            switch (path)
            {
                case ExecutionPath.SimdFull: lValue = false; rValue = false; break;
                case ExecutionPath.SimdScalarRight: lValue = false; rValue = true; break;
                case ExecutionPath.SimdScalarLeft: lValue = true; rValue = false; break;
                default: return null;
            }

            if (lhsSigned)
                return (l, r, res, ls, rs, sh, nd, n) => MixedSignCompare((long*)l, lValue, (ulong*)r, rValue, (byte*)res, n, op);

            ComparisonOp mirrored = MirrorComparison(op);
            return (l, r, res, ls, rs, sh, nd, n) => MixedSignCompare((long*)r, rValue, (ulong*)l, lValue, (byte*)res, n, mirrored);
        }

        /// <summary>
        ///     The operator that gives the same answer with the operands swapped: <c>a op b</c> equals
        ///     <c>b MirrorComparison(op) a</c>.
        /// </summary>
        /// <param name="op">The comparison.</param>
        /// <returns><c>&lt;</c> and <c>&gt;</c> exchanged, <c>&lt;=</c> and <c>&gt;=</c> exchanged; <c>==</c> and
        ///     <c>!=</c> unchanged.</returns>
        /// <remarks>This is the mirror, not the negation: the negation of <c>&lt;</c> is <c>&gt;=</c>, and it is
        ///     wrong for float operands (a NaN makes both False).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static ComparisonOp MirrorComparison(ComparisonOp op)
        {
            switch (op)
            {
                case ComparisonOp.Less: return ComparisonOp.Greater;
                case ComparisonOp.LessEqual: return ComparisonOp.GreaterEqual;
                case ComparisonOp.Greater: return ComparisonOp.Less;
                case ComparisonOp.GreaterEqual: return ComparisonOp.LessEqual;
                default: return op;
            }
        }

        /// <summary>
        ///     The exact comparison of a signed 64-bit value with an unsigned one (NumPy's <c>qQ</c> loops).
        /// </summary>
        /// <param name="s">The signed value.</param>
        /// <param name="u">The unsigned value.</param>
        /// <param name="op">The comparison, as <c>s op u</c>.</param>
        /// <returns>The result.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool MixedSignScalarRule(long s, ulong u, ComparisonOp op)
        {
            switch (op)
            {
                case ComparisonOp.Equal: return s >= 0 && (ulong)s == u;
                case ComparisonOp.NotEqual: return s < 0 || (ulong)s != u;
                case ComparisonOp.Less: return s < 0 || (ulong)s < u;
                case ComparisonOp.LessEqual: return s < 0 || (ulong)s <= u;
                case ComparisonOp.Greater: return s >= 0 && (ulong)s > u;
                default: return s >= 0 && (ulong)s >= u;   // ComparisonOp.GreaterEqual
            }
        }

        /// <summary>The fact "s is below u" (see the file header).</summary>
        private const int MixedSignBelow = 0;

        /// <summary>The fact "s is below u or equals it".</summary>
        private const int MixedSignBelowOrSame = 1;

        /// <summary>The fact "s equals u".</summary>
        private const int MixedSignSame = 2;

        /// <summary>
        ///     One fact about eight signed values and eight unsigned values, as eight bits: bit k is the fact for
        ///     element k.
        /// </summary>
        /// <param name="s">The signed operand: 8 elements, or the value's buffer when the step is 0.</param>
        /// <param name="sStep">Elements per signed element: 1, or 0 for one value.</param>
        /// <param name="u">The unsigned operand, read as <c>long</c> bits.</param>
        /// <param name="uStep">Elements per unsigned element: 1, or 0 for one value.</param>
        /// <param name="fact"><see cref="MixedSignBelow"/>, <see cref="MixedSignBelowOrSame"/> or
        ///     <see cref="MixedSignSame"/>. Every call site passes a constant, so the inlined body holds one branch
        ///     of the three and no test.</param>
        /// <returns>The eight bits.</returns>
        /// <remarks>
        ///     A negative lane is decided by its sign alone: it is below every unsigned value, and it is never the
        ///     same as one whatever its bits compare to.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe uint MixedSignEight(long* s, long sStep, long* u, long uStep, int fact)
        {
            var s0 = Avx.LoadVector256(s);
            var s1 = Avx.LoadVector256(s + 4 * sStep);
            var u0 = Avx.LoadVector256(u);
            var u1 = Avx.LoadVector256(u + 4 * uStep);
            var negative0 = Avx2.CompareGreaterThan(Vector256<long>.Zero, s0);
            var negative1 = Avx2.CompareGreaterThan(Vector256<long>.Zero, s1);
            Vector256<long> m0, m1;
            if (fact == MixedSignSame)
            {
                m0 = Avx2.AndNot(negative0, Avx2.CompareEqual(s0, u0));     // !negative & bits equal
                m1 = Avx2.AndNot(negative1, Avx2.CompareEqual(s1, u1));
            }
            else
            {
                // The unsigned order of two values is the signed order of both with their top bit flipped.
                var flip = Vector256.Create(long.MinValue);
                var sb0 = Avx2.Xor(s0, flip);
                var sb1 = Avx2.Xor(s1, flip);
                var ub0 = Avx2.Xor(u0, flip);
                var ub1 = Avx2.Xor(u1, flip);
                if (fact == MixedSignBelow)
                {
                    m0 = Avx2.Or(negative0, Avx2.CompareGreaterThan(ub0, sb0));     // negative | (ulong)s < u
                    m1 = Avx2.Or(negative1, Avx2.CompareGreaterThan(ub1, sb1));
                }
                else
                {
                    // negative | (ulong)s <= u, as !(!negative & (ulong)s > u); the bits are inverted below.
                    m0 = Avx2.AndNot(negative0, Avx2.CompareGreaterThan(sb0, ub0));
                    m1 = Avx2.AndNot(negative1, Avx2.CompareGreaterThan(sb1, ub1));
                }
            }
            // vmovmskpd reads the top bit of each 64-bit lane: four bits per vector.
            uint bits = (uint)Avx.MoveMask(m0.AsDouble()) | ((uint)Avx.MoveMask(m1.AsDouble()) << 4);
            return fact == MixedSignBelowOrSame ? bits ^ 0xFFu : bits;
        }

        /// <summary>
        ///     <c>s op u</c> over <paramref name="n"/> elements: a signed operand against an unsigned one, each a
        ///     contiguous run or one value.
        /// </summary>
        /// <param name="ps">The signed operand: n elements, or one when <paramref name="sIsValue"/>.</param>
        /// <param name="sIsValue">The signed operand is one value.</param>
        /// <param name="pu">The unsigned operand: n elements, or one when <paramref name="uIsValue"/>.</param>
        /// <param name="uIsValue">The unsigned operand is one value.</param>
        /// <param name="pr">Result: n bytes, each written 0 or 1.</param>
        /// <param name="n">Element count.</param>
        /// <param name="op">The comparison, as <c>signed op unsigned</c>.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void MixedSignCompare(long* ps, bool sIsValue, ulong* pu, bool uIsValue, byte* pr, long n, ComparisonOp op)
        {
            // An operand that is one value is read from a buffer holding it four times, at an element step of 0.
            long* sFour = stackalloc long[4];
            ulong* uFour = stackalloc ulong[4];
            long sStep = 1, uStep = 1;
            if (sIsValue)
            {
                sFour[0] = sFour[1] = sFour[2] = sFour[3] = *ps;
                ps = sFour;
                sStep = 0;
            }
            if (uIsValue)
            {
                uFour[0] = uFour[1] = uFour[2] = uFour[3] = *pu;
                pu = uFour;
                uStep = 0;
            }

            long i = 0;
            if (Avx2.IsSupported)
            {
                // The operator as one fact, plain or negated (see the file header).
                bool negated = op is ComparisonOp.Greater or ComparisonOp.GreaterEqual or ComparisonOp.NotEqual;
                uint flip = negated ? 0xFFu : 0u;
                long* u = (long*)pu;

                // 8 elements per pass (two vectors, one 8-byte store). Each loop stops while a whole pass still fits,
                // so nothing is read or written past element n. The loops differ only in the constant they pass.
                if (op is ComparisonOp.Less or ComparisonOp.GreaterEqual)
                {
                    for (; i + 8 <= n; i += 8)
                        *(ulong*)(pr + i) = SpreadBitsToBytes(MixedSignEight(ps + i * sStep, sStep, u + i * uStep, uStep, MixedSignBelow) ^ flip);
                }
                else if (op is ComparisonOp.LessEqual or ComparisonOp.Greater)
                {
                    for (; i + 8 <= n; i += 8)
                        *(ulong*)(pr + i) = SpreadBitsToBytes(MixedSignEight(ps + i * sStep, sStep, u + i * uStep, uStep, MixedSignBelowOrSame) ^ flip);
                }
                else
                {
                    for (; i + 8 <= n; i += 8)
                        *(ulong*)(pr + i) = SpreadBitsToBytes(MixedSignEight(ps + i * sStep, sStep, u + i * uStep, uStep, MixedSignSame) ^ flip);
                }
            }

            for (; i < n; i++)
                pr[i] = MixedSignScalarRule(ps[i * sStep], pu[i * uStep], op) ? (byte)1 : (byte)0;
        }
    }
}
