using System;
using System.Numerics;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Power.cs — NumPy's float power loops, element by element
// =============================================================================
//
// NumPy 2.4.2 has three float power loops, and each computes something different:
//   * DOUBLE_power / FLOAT_power (numpy/_core/src/umath/loops_umath_fp.dispatch.c.src): when the
//     exponent's STEP in the inner-loop call is 0, the loop reads the exponent once and, for -1, 0,
//     0.5, 1 or 2, writes 1.0/x, 1.0, sqrt(x), x or x*x instead of calling pow. Any other exponent,
//     or a nonzero step, calls the C library's pow / powf. The two answers differ: 757.7617475757002
//     ** 2 is x*x = 0x412185f5bb7003c1 but pow = ...03c2, pow(-0.0, 0.5) is +0.0 but sqrt(-0.0) is
//     -0.0, pow(-inf, 0.5) is +inf but sqrt(-inf) is NaN, and x keeps a signalling NaN signalling
//     where pow quiets it. The win-amd64 wheel has no SVML pow (it needs AVX512_SKX and SVML,
//     Linux only), so every non-fast call is the scalar CRT pow, which Math.Pow / MathF.Pow call too.
//   * HALF_power: PyUFunc_ee_e_As_ff_f with powf — the halves widen to float32 bit for bit (a
//     signalling NaN stays signalling, npy_half_to_float), powf runs in float32, and the result
//     narrows with npy_float_to_half. There is NO fast path. A double bridge is not the same
//     function: (double)Half quiets a signalling NaN (so sNaN ** 0 is 1 instead of NaN), the double
//     pow drops a negative NaN's sign for integer exponents where powf keeps it, and rounding the
//     double result to half once can differ from rounding powf's float result.
//   * float_power: PyUFunc_dd_d with pow — no fast path either (DefaultEngine.FloatPower).
//
// Which step NumPy's call sees is decided by DefaultEngine.NumPyPowerTakesStrideZeroLoop; this file
// holds the per-element and per-vector bodies the kernels call. The stride-0 bodies are generic over
// the two float loops (one method per vector width, none per dtype); the float16 kernel that runs
// HALF_power over whole arrays is HalfPower in DirectILKernelGenerator.Binary.Arith.Half.cs.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// One element of NumPy's <c>DOUBLE_power</c> / <c>FLOAT_power</c> when the exponent's step in the inner-loop
        /// call is 0: the fast path for an exponent of -1, 0, 0.5, 1 or 2, the C library <c>pow</c> / <c>powf</c>
        /// otherwise.
        /// </summary>
        /// <typeparam name="T">The loop dtype: <see cref="double"/> or <see cref="float"/> (only those loops have the
        /// stride-0 branch; the generic arithmetic resolves to Math.* / MathF.*, which call the same CRT functions as
        /// NumPy's npy_pow / npy_powf and npy_sqrt / npy_sqrtf).</typeparam>
        /// <param name="x">The base, already cast to the loop dtype.</param>
        /// <param name="e">The exponent, already cast to the loop dtype (the value NumPy's loop compares).</param>
        /// <returns><c>1 / x</c>, <c>1</c>, <c>sqrt(x)</c>, <c>x</c> (bits untouched, a signalling NaN stays
        /// signalling) or <c>x * x</c> for the five fast exponents; <c>pow(x, e)</c> otherwise.</returns>
        /// <remarks>
        /// <para>The comparisons are NumPy's: <c>-0.0 == 0.0</c> takes the <c>1</c> branch, and a NaN exponent matches
        /// nothing.</para>
        /// <para>NumPy's float32 reciprocal is <c>1.0 / in1</c> in C — a DOUBLE division rounded to float on the store.
        /// That double rounding is innocuous (53 ≥ 2·24 + 2 bits), so a float32 division gives the same bits: verified
        /// over all 2^32 float32 patterns, NaN payloads included, scalar and vector.</para>
        /// <para>Only correct where NumPy's call has a zero exponent step (every element of such a call carries the
        /// call's one exponent value, so the per-element test is NumPy's per-call test). With a nonzero step NumPy
        /// calls <c>pow</c> for every element; use the plain power kernel there.</para>
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static T PowerStrideZero<T>(T x, T e) where T : unmanaged, IFloatingPointIeee754<T>
        {
            // The chain in NumPy's order. The constants are exact in both float widths, so the comparisons are too.
            if (e == -T.One)
                return T.One / x;
            if (e == T.Zero)
                return T.One;
            if (e == T.CreateTruncating(0.5))
                return T.Sqrt(x);
            if (e == T.One)
                return x;
            if (e == T.CreateTruncating(2.0))
                return x * x;
            return T.Pow(x, e);
        }

        /// <summary>
        /// <see cref="PowerStrideZero{T}"/> for a 128-bit vector of elements. Within a NumPy call with a zero exponent
        /// step every element carries the same exponent, and the kernel's broadcast loops hand the body a
        /// <c>Vector.Create</c>'d exponent, so ONE compare decides the whole vector and the fast operation runs as one
        /// vector instruction; any other vector (lanes that differ, or an exponent off the fast set) goes element by
        /// element.
        /// </summary>
        /// <typeparam name="T">The loop dtype: <see cref="double"/> or <see cref="float"/>.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns>The powers, bit-identical to <see cref="PowerStrideZero{T}"/> on every lane (vsqrt, vmul and vdiv
        /// round like their scalar forms).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<T> PowerStrideZeroVector128<T>(Vector128<T> x, Vector128<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            // Square first: x ** 2 is the common call. EqualsAll is IEEE ==, so -0.0 matches 0 and NaN matches nothing.
            if (Vector128.EqualsAll(e, Vector128.Create(T.CreateTruncating(2.0))))
                return x * x;
            if (Vector128.EqualsAll(e, Vector128.Create(T.CreateTruncating(0.5))))
                return Vector128.Sqrt(x);
            if (Vector128.EqualsAll(e, Vector128.Create(-T.One)))
                return Vector128<T>.One / x;
            if (Vector128.EqualsAll(e, Vector128<T>.One))
                return x;
            if (Vector128.EqualsAll(e, Vector128<T>.Zero))
                return Vector128<T>.One;
            return PowerStrideZeroLanes128(x, e);
        }

        /// <summary><see cref="PowerStrideZeroVector128{T}"/> for a 256-bit vector (same contract).</summary>
        /// <typeparam name="T">The loop dtype: <see cref="double"/> or <see cref="float"/>.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns>The powers, bit-identical to <see cref="PowerStrideZero{T}"/> on every lane.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector256<T> PowerStrideZeroVector256<T>(Vector256<T> x, Vector256<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (Vector256.EqualsAll(e, Vector256.Create(T.CreateTruncating(2.0))))
                return x * x;
            if (Vector256.EqualsAll(e, Vector256.Create(T.CreateTruncating(0.5))))
                return Vector256.Sqrt(x);
            if (Vector256.EqualsAll(e, Vector256.Create(-T.One)))
                return Vector256<T>.One / x;
            if (Vector256.EqualsAll(e, Vector256<T>.One))
                return x;
            if (Vector256.EqualsAll(e, Vector256<T>.Zero))
                return Vector256<T>.One;
            return PowerStrideZeroLanes256(x, e);
        }

        /// <summary><see cref="PowerStrideZeroVector128{T}"/> for a 512-bit vector (same contract).</summary>
        /// <typeparam name="T">The loop dtype: <see cref="double"/> or <see cref="float"/>.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns>The powers, bit-identical to <see cref="PowerStrideZero{T}"/> on every lane.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector512<T> PowerStrideZeroVector512<T>(Vector512<T> x, Vector512<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (Vector512.EqualsAll(e, Vector512.Create(T.CreateTruncating(2.0))))
                return x * x;
            if (Vector512.EqualsAll(e, Vector512.Create(T.CreateTruncating(0.5))))
                return Vector512.Sqrt(x);
            if (Vector512.EqualsAll(e, Vector512.Create(-T.One)))
                return Vector512<T>.One / x;
            if (Vector512.EqualsAll(e, Vector512<T>.One))
                return x;
            if (Vector512.EqualsAll(e, Vector512<T>.Zero))
                return Vector512<T>.One;
            return PowerStrideZeroLanes512(x, e);
        }

        /// <summary>
        /// The element-by-element form of <see cref="PowerStrideZeroVector128{T}"/>: lanes whose exponents differ (the
        /// iterator walks an axis NumPy did not) or an exponent off the fast set (pow on every lane).
        /// </summary>
        /// <typeparam name="T">The loop dtype.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns><see cref="PowerStrideZero{T}"/> of each lane pair.</returns>
        /// <remarks>Not inlined: the fast vectors must not pay for this loop's spills.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector128<T> PowerStrideZeroLanes128<T>(Vector128<T> x, Vector128<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Vector128<T> r = default;
            ref T rr = ref Unsafe.As<Vector128<T>, T>(ref r);
            ref T xr = ref Unsafe.As<Vector128<T>, T>(ref x);
            ref T er = ref Unsafe.As<Vector128<T>, T>(ref e);
            for (int i = 0; i < Vector128<T>.Count; i++)
                Unsafe.Add(ref rr, i) = PowerStrideZero(Unsafe.Add(ref xr, i), Unsafe.Add(ref er, i));
            return r;
        }

        /// <summary>The element-by-element form of <see cref="PowerStrideZeroVector256{T}"/> (see
        /// <see cref="PowerStrideZeroLanes128{T}"/>).</summary>
        /// <typeparam name="T">The loop dtype.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns><see cref="PowerStrideZero{T}"/> of each lane pair.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector256<T> PowerStrideZeroLanes256<T>(Vector256<T> x, Vector256<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Vector256<T> r = default;
            ref T rr = ref Unsafe.As<Vector256<T>, T>(ref r);
            ref T xr = ref Unsafe.As<Vector256<T>, T>(ref x);
            ref T er = ref Unsafe.As<Vector256<T>, T>(ref e);
            for (int i = 0; i < Vector256<T>.Count; i++)
                Unsafe.Add(ref rr, i) = PowerStrideZero(Unsafe.Add(ref xr, i), Unsafe.Add(ref er, i));
            return r;
        }

        /// <summary>The element-by-element form of <see cref="PowerStrideZeroVector512{T}"/> (see
        /// <see cref="PowerStrideZeroLanes128{T}"/>).</summary>
        /// <typeparam name="T">The loop dtype.</typeparam>
        /// <param name="x">The bases.</param>
        /// <param name="e">The exponents.</param>
        /// <returns><see cref="PowerStrideZero{T}"/> of each lane pair.</returns>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Vector512<T> PowerStrideZeroLanes512<T>(Vector512<T> x, Vector512<T> e)
            where T : unmanaged, IFloatingPointIeee754<T>
        {
            Vector512<T> r = default;
            ref T rr = ref Unsafe.As<Vector512<T>, T>(ref r);
            ref T xr = ref Unsafe.As<Vector512<T>, T>(ref x);
            ref T er = ref Unsafe.As<Vector512<T>, T>(ref e);
            for (int i = 0; i < Vector512<T>.Count; i++)
                Unsafe.Add(ref rr, i) = PowerStrideZero(Unsafe.Add(ref xr, i), Unsafe.Add(ref er, i));
            return r;
        }

        /// <summary>
        /// One element of NumPy's <c>HALF_power</c>: both halves widened to float32 bit for bit (NumPy's
        /// <c>npy_half_to_float</c> keeps a signalling NaN signalling), <c>powf</c>, then NumPy's
        /// <c>npy_float_to_half</c> narrowing. NumPy's float16 power has no scalar-exponent fast path.
        /// </summary>
        /// <param name="a">The base.</param>
        /// <param name="b">The exponent.</param>
        /// <returns>The float16 power NumPy returns, NaN bits included.</returns>
        /// <remarks>
        /// Verified against NumPy 2.4.2 over every float16 base and 17 exponents (1,114,112 cases, no difference).
        /// The BCL route <c>(Half)Math.Pow((double)a, (double)b)</c> differed on 7,161 of them: <c>sNaN ** 0</c>
        /// returned 1 (the BCL widen quiets the NaN first), integer powers of a negative NaN lost its sign, and one
        /// result rounded once from double instead of from powf's float.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static Half PowerHalfNumPy(Half a, Half b)
            => BitConverter.UInt16BitsToHalf(SingleToHalfBits(MathF.Pow(
                HalfToFloatScalarExact(BitConverter.HalfToUInt16Bits(a)),
                HalfToFloatScalarExact(BitConverter.HalfToUInt16Bits(b)))));

        /// <summary>
        /// Emits the scalar body of NumPy's stride-0 power loop for one element: stack in <c>[lhs, rhs]</c> (their own
        /// dtypes), stack out <c>[result]</c> (<paramref name="loop"/>). Both operands are converted to the loop dtype
        /// first, as NumPy's iterator casts them before the loop runs.
        /// </summary>
        /// <param name="il">The IL generator of the kernel being built.</param>
        /// <param name="lhsType">The base's stored dtype.</param>
        /// <param name="rhsType">The exponent's stored dtype.</param>
        /// <param name="loop">The loop dtype: <see cref="NPTypeCode.Single"/> or <see cref="NPTypeCode.Double"/>.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not float32 or float64 (only those loops
        /// have the stride-0 branch).</exception>
        internal static void EmitPowerStrideZeroBody(ILGenerator il, NPTypeCode lhsType, NPTypeCode rhsType, NPTypeCode loop)
        {
            if (loop != NPTypeCode.Single && loop != NPTypeCode.Double)
                throw new NotSupportedException($"NumPy's stride-0 power loop exists for float32 and float64 only, not {loop}.");

            // [lhs, rhs] -> park the exponent so the base can be converted under it.
            var locRhs = il.DeclareLocal(GetClrType(rhsType));
            il.Emit(OpCodes.Stloc, locRhs);
            if (lhsType != loop)
                EmitConvertTo(il, lhsType, loop);
            il.Emit(OpCodes.Ldloc, locRhs);
            if (rhsType != loop)
                EmitConvertTo(il, rhsType, loop);
            il.EmitCall(OpCodes.Call, GetGenericHelper(nameof(PowerStrideZero), GetClrType(loop)), null);
        }

        /// <summary>
        /// Emits the vector body of NumPy's stride-0 power loop: stack in <c>[Vector{W}&lt;T&gt; x, Vector{W}&lt;T&gt;
        /// e]</c>, stack out <c>[Vector{W}&lt;T&gt; result]</c>, for the width the kernels were built for
        /// (<see cref="VectorBits"/>).
        /// </summary>
        /// <param name="il">The IL generator of the kernel being built.</param>
        /// <param name="loop">The loop dtype, float32 or float64 — and every operand's dtype: the Tier-3B shell runs a
        /// vector body only when all operands share one SIMD dtype, so the caller supplies it only then.</param>
        /// <remarks>
        /// The shell's broadcast loops (a 0-d or column exponent: stride 0 within the call) Vector.Create the exponent,
        /// so each vector takes one compare and one vector operation — a varying column exponent stops paying a scalar
        /// call per element (0.85× NumPy before at (2, 5000)).
        /// </remarks>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not float32 or float64, or the host has no
        /// SIMD width (the shell never asks for a vector body then).</exception>
        internal static void EmitPowerStrideZeroVectorBody(ILGenerator il, NPTypeCode loop)
        {
            if (loop != NPTypeCode.Single && loop != NPTypeCode.Double)
                throw new NotSupportedException($"NumPy's stride-0 power loop exists for float32 and float64 only, not {loop}.");
            string helper = VectorBits switch
            {
                128 => nameof(PowerStrideZeroVector128),
                256 => nameof(PowerStrideZeroVector256),
                512 => nameof(PowerStrideZeroVector512),
                _ => throw new NotSupportedException("The stride-0 power vector body needs a SIMD width."),
            };
            il.EmitCall(OpCodes.Call, GetGenericHelper(helper, GetClrType(loop)), null);
        }
    }
}
