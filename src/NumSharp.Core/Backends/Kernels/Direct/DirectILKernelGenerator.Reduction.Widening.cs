using System;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Reduction.Widening.cs — Flat WIDENING integer Sum (SIMD)
// =============================================================================
//
// WHY THIS FILE EXISTS
//   The flat (whole-array) reduction gate CanUseReductionSimd (Reduction.cs)
//   deliberately forces the SCALAR path whenever a Sum/Prod accumulates into a
//   WIDER dtype than the input (int32→int64, uint32→uint64, …): a same-type SIMD
//   vector cannot widen its lanes, so accumulating int32 lanes risks overflow.
//   That left `np.sum(int32)`/`np.sum(uint32)` — the single most common integer
//   reductions — running one element per iteration, measurably ~2.4× slower
//   in-cache than the same op on int64 (which DOES vectorize) despite moving half
//   the bytes.
//
//   This file adds the missing piece for Sum: a hand-written SIMD helper that
//   WIDENS every element to 64-bit BEFORE adding, into several independent
//   Vector256<long>/<ulong> accumulators. Two properties make it a drop-in:
//
//     * CORRECT for ALL inputs. Each element is widened to 64-bit first, so no
//       lane can overflow regardless of magnitude — exactly what the scalar path
//       guaranteed. And integer addition is ASSOCIATIVE + COMMUTATIVE, so any
//       accumulator count / lane order / tail split yields the IDENTICAL 64-bit
//       total as the sequential scalar sum — hence BIT-EXACT with NumPy (unlike
//       float sum, whose reordering changes the rounding). The differential-fuzz
//       oracle confirms this.
//
//     * NO IL EMISSION. The reduction cache stores a delegate of type
//       TypedElementReductionKernel<TResult>; that delegate can be bound to an
//       ordinary static C# method with the matching signature just as well as to
//       an emitted DynamicMethod. So this is plain, verifiable C# SIMD (the same
//       pattern as the existing FlatMinMaxF64Avx / NanSumSimdHelper* helpers),
//       not fragile hand-emitted IL.
//
//   Scope is the 32-bit→64-bit integer pair (int32→int64, uint32→uint64): a
//   single Vector256.Widen step, the run's exact target, and the highest
//   per-element payoff. Narrower widening sums (int8/int16 and their unsigned
//   siblings, which need multi-level widening) remain on the correct scalar path
//   and are a natural follow-up.
//
// STRUCTURE (per helper)
//   main loop  : 16 elements/iter — two Vector256<int> loads, each widened to a
//                (lower, upper) pair, accumulated into 4 independent 64-bit
//                vector accumulators (breaks the add-latency dependency chain).
//   vec tail   : 8 elements/iter — one load, widen, into two accumulators.
//   scalar tail: the final <8 elements.
//   combine    : horizontal Vector256.Sum of the four accumulators, once.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Gate for the flat widening-<see cref="ReductionOp.Sum"/> SIMD fast path: true only for a
        /// contiguous 32-bit→64-bit integer sum (int32→int64 or uint32→uint64) on a host with
        /// hardware 256-bit vectors. Every other reduction (same-type, prod, min/max, narrower
        /// widening, strided, non-SIMD host) is left to the existing paths.
        /// </summary>
        /// <param name="key">The reduction kernel key (op, input/accumulator dtype, contiguity).</param>
        /// <returns><c>true</c> if <see cref="TryGetWideningSumDelegate{TResult}"/> can serve this key.</returns>
        internal static bool CanUseReductionSimdWidening(ElementReductionKernelKey key)
        {
            // Sum only: prod's widening path stays scalar (a product overflows the accumulator far
            // sooner and is not the measured hot path). Contiguous only: strided views keep the
            // existing gather/strided reduction. 256-bit hardware required — else the scalar loop
            // is as good as a 1-lane "vector".
            if (key.Op != ReductionOp.Sum || !key.IsContiguous)
                return false;
            if (!Vector256.IsHardwareAccelerated)
                return false;
            return (key.InputType == NPTypeCode.Int32 && key.AccumulatorType == NPTypeCode.Int64)
                || (key.InputType == NPTypeCode.UInt32 && key.AccumulatorType == NPTypeCode.UInt64);
        }

        /// <summary>
        /// Bind the matching widening-sum C# SIMD helper to a
        /// <see cref="TypedElementReductionKernel{TResult}"/> delegate, so the reduction cache can
        /// return it in place of an emitted scalar kernel.
        /// </summary>
        /// <typeparam name="TResult">The accumulator/result CLR type the caller expects — <c>long</c>
        /// for int32→int64, <c>ulong</c> for uint32→uint64. A mismatch makes
        /// <see cref="Delegate.CreateDelegate(Type, MethodInfo)"/> throw, which is caught and
        /// reported as "no widening helper" so the caller falls back safely.</typeparam>
        /// <param name="key">The reduction kernel key.</param>
        /// <param name="del">On success, the bound delegate; otherwise <c>null</c>.</param>
        /// <returns><c>true</c> if a helper was bound.</returns>
        internal static bool TryGetWideningSumDelegate<TResult>(ElementReductionKernelKey key, out Delegate del)
            where TResult : unmanaged
        {
            del = null;
            if (!CanUseReductionSimdWidening(key))
                return false;

            // Pick the helper by input dtype; its return type (long/ulong) must equal TResult, which
            // it does because the caller derives TResult from key.AccumulatorType (Int64/UInt64).
            string name = key.InputType == NPTypeCode.Int32
                ? nameof(SumWidenInt32ToInt64)
                : nameof(SumWidenUInt32ToUInt64);

            MethodInfo mi = typeof(DirectILKernelGenerator).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic);
            if (mi == null)
                return false;

            try
            {
                // Signature-compatible bind (return type + params) — no boxing, called like the
                // emitted kernel. A TResult that does not match the helper's return type throws here.
                del = Delegate.CreateDelegate(typeof(TypedElementReductionKernel<TResult>), mi);
            }
            catch (ArgumentException)
            {
                del = null;
                return false;
            }
            return del != null;
        }

        /// <summary>
        /// Sum a contiguous <c>int</c> array into an exact <c>long</c>, widening every element to
        /// 64-bit before adding (so the result cannot overflow and is bit-exact with the scalar sum
        /// and with NumPy). Matches <see cref="TypedElementReductionKernel{TResult}"/> so it can be
        /// bound directly as a reduction kernel.
        /// </summary>
        /// <param name="input">Base pointer to logical element 0 (the caller has already applied the
        /// view's <c>Shape.offset</c>).</param>
        /// <param name="strides">Unused — this helper is bound only for the contiguous case.</param>
        /// <param name="shape">Unused (contiguous).</param>
        /// <param name="ndim">Unused (contiguous).</param>
        /// <param name="totalSize">Number of elements to sum.</param>
        /// <returns>The exact 64-bit sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe long SumWidenInt32ToInt64(void* input, long* strides, long* shape, int ndim, long totalSize)
        {
            int* p = (int*)input;
            long n = totalSize;
            long i = 0;

            if (n >= Vector256<int>.Count)
            {
                // 4 independent 64-bit accumulators keep the add pipeline busy across its latency.
                Vector256<long> a0 = Vector256<long>.Zero, a1 = Vector256<long>.Zero,
                                a2 = Vector256<long>.Zero, a3 = Vector256<long>.Zero;

                // Main loop: 16 int32 per iteration (two vectors), each widened to a (lower, upper)
                // pair of int64 vectors and folded into the four accumulators.
                long limit16 = n - 16;
                for (; i <= limit16; i += 16)
                {
                    (Vector256<long> lo0, Vector256<long> hi0) = Vector256.Widen(Vector256.Load(p + i));
                    (Vector256<long> lo1, Vector256<long> hi1) = Vector256.Widen(Vector256.Load(p + i + 8));
                    a0 += lo0; a1 += hi0; a2 += lo1; a3 += hi1;
                }

                // Vector remainder: whole 8-wide vectors that did not fill a 16-wide iteration.
                long limit8 = n - 8;
                for (; i <= limit8; i += 8)
                {
                    (Vector256<long> lo, Vector256<long> hi) = Vector256.Widen(Vector256.Load(p + i));
                    a0 += lo; a1 += hi;
                }

                long s = Vector256.Sum(a0 + a1 + a2 + a3);   // one horizontal reduction
                for (; i < n; i++) s += p[i];                // <8-element scalar tail
                return s;
            }

            long sc = 0;
            for (; i < n; i++) sc += p[i];
            return sc;
        }

        /// <summary>
        /// Unsigned twin of <see cref="SumWidenInt32ToInt64"/>: sum a contiguous <c>uint</c> array
        /// into an exact <c>ulong</c>, widening every element to 64-bit first (overflow-free,
        /// bit-exact with the scalar sum and NumPy). Same structure and contract.
        /// </summary>
        /// <param name="input">Base pointer to logical element 0 (offset already applied).</param>
        /// <param name="strides">Unused (contiguous).</param>
        /// <param name="shape">Unused (contiguous).</param>
        /// <param name="ndim">Unused (contiguous).</param>
        /// <param name="totalSize">Number of elements to sum.</param>
        /// <returns>The exact 64-bit unsigned sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe ulong SumWidenUInt32ToUInt64(void* input, long* strides, long* shape, int ndim, long totalSize)
        {
            uint* p = (uint*)input;
            long n = totalSize;
            long i = 0;

            if (n >= Vector256<uint>.Count)
            {
                Vector256<ulong> a0 = Vector256<ulong>.Zero, a1 = Vector256<ulong>.Zero,
                                 a2 = Vector256<ulong>.Zero, a3 = Vector256<ulong>.Zero;

                long limit16 = n - 16;
                for (; i <= limit16; i += 16)
                {
                    (Vector256<ulong> lo0, Vector256<ulong> hi0) = Vector256.Widen(Vector256.Load(p + i));
                    (Vector256<ulong> lo1, Vector256<ulong> hi1) = Vector256.Widen(Vector256.Load(p + i + 8));
                    a0 += lo0; a1 += hi0; a2 += lo1; a3 += hi1;
                }

                long limit8 = n - 8;
                for (; i <= limit8; i += 8)
                {
                    (Vector256<ulong> lo, Vector256<ulong> hi) = Vector256.Widen(Vector256.Load(p + i));
                    a0 += lo; a1 += hi;
                }

                ulong s = Vector256.Sum(a0 + a1 + a2 + a3);
                for (; i < n; i++) s += p[i];
                return s;
            }

            ulong sc = 0;
            for (; i < n; i++) sc += p[i];
            return sc;
        }
    }
}
