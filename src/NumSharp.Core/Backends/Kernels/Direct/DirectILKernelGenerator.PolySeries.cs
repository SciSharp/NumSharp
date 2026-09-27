using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using NumSharp.Backends.Iteration;

// =============================================================================
// DirectILKernelGenerator.PolySeries.cs — the numpy.polynomial series primitives (U1)
// =============================================================================
//
// RESPONSIBILITY
// --------------
// numpy.polynomial's additive family (polyutils.as_series / trimseq / trimcoef / getdomain / mapparms,
// and every basis's {p}add / {p}sub / {p}trim / {p}line) is pure Python over 1-D coefficient arrays. The
// only ELEMENT loops in it are:
//   * trimseq's backward scan for the last element that is `!= 0`;
//   * as_series' `np.array(a, copy=True, dtype=common_type)` conversion copy;
//   * _add / _sub's in-place `longer[:n] += shorter` (or `-=`, or `c2 = -c2; c2[:n] += c1`);
//   * trimcoef's `np.nonzero(np.abs(c) > tol)` — of which only the LAST index is ever read.
// Everything else is NumPy SCALAR arithmetic (mapparms, {p}line), which is scalarmath, not a loop.
// This file emits each of those loops as one whole-series kernel (the DirectILKernelGenerator contract:
// the kernel walks its own strides), plus one-element kernels for NumPy's scalarmath:
//
//   PolyTrimLenKernel      (p, n, stride)                  -> trimseq's length
//   PolyCombineKernel      (a, sa, na, b, sb, nb, r)       -> _add/_sub into r, returns the trimmed length
//   PolyLastAboveKernel    (p, n, stride, tol)             -> 1 + last i with |c[i]| > tol, or 0
//   PolyCastKernel         (p, n, stride, r)               -> r[i] = (T)p[i]  (strided -> contiguous)
//   PolyScalarBinaryKernel (a, b, r)                       -> *r = *a OP *b   (both already in the loop dtype; the
//                                                             complex64 variant runs NumPy's np.complex64 scalar
//                                                             arithmetic in float32 on Complex carriers)
//   PolyScalarUnaryKernel  (a, r)                          -> *r = -*a
//   PolyScalarPredicateKernel (a)                          -> *a != 0   /   *a < 0
//
// WHY FUSED, AND WHY IT IS STILL NUMPY'S ARITHMETIC
// -------------------------------------------------
// NumPy copies both operands to the common dtype (as_series), then adds one INTO the other. The combine
// kernel reads each source element in its own dtype, converts it with the SAME house conversion the copy
// uses (EmitConvertTo), and applies the SAME house scalar op the ufunc loop uses (EmitScalarOperation), in
// NumPy's operand order — so every stored value is the one NumPy stores, with the intermediate copies (two
// allocations and two passes) never materialized. The operand ORDER is load-bearing: float16's NaN
// priority and a NaN payload follow the first operand, so `c2[:n1] += c1` is emitted as c2[i] + c1[i],
// never c1[i] + c2[i]; and `c2 = -c2; c2[:n1] += c1` negates BEFORE the add, which differs from c1 - c2 in
// which NaN survives.
//
// DTYPES
// ------
// Every kernel is keyed by the dtypes it reads and writes and built by the house emitters, so there is no
// per-dtype C#. The coefficient dtype is always np.common_type's (float16/float32/float64/complex128, and
// NumSharp's decimal); the SOURCE dtypes are anything a caller passes (every integer and char converts to
// float64 on the way in, exactly as NumPy's copy does).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     trimseq's scan (NumPy 2.4.2 <c>polyutils.trimseq</c>), with NumPy's two return shapes kept apart:
    ///     <c>seq</c> ITSELF when it is empty or its last element is nonzero, otherwise the SLICE
    ///     <c>seq[:i+1]</c> for the last <c>i</c> with <c>seq[i] != 0</c> (at least one element) — a slice even when
    ///     it keeps every element (<c>[0.]</c>), which is observable as a view.
    /// </summary>
    /// <param name="p">Address of element 0.</param>
    /// <param name="n">Element count.</param>
    /// <param name="byteStride">Signed byte stride between elements.</param>
    /// <returns><paramref name="n"/> (≥ 0) when NumPy returns <c>seq</c> itself — empty, or the last element nonzero
    ///     (NaN counts as nonzero); <c>-k</c> (≤ -1) when NumPy returns the slice <c>seq[:k]</c>.</returns>
    public unsafe delegate long PolyTrimLenKernel(byte* p, long n, long byteStride);

    /// <summary>
    ///     The body of NumPy's <c>polyutils._add</c> / <c>_sub</c> after <c>as_series</c>: writes the
    ///     in-place-updated target series <c>r[0..na)</c> (contiguous, the result dtype) from the target
    ///     <c>a</c> (the operand NumPy updates in place) and the other operand <c>b</c>, both read through their
    ///     own strides and dtypes, then runs trimseq on <c>r</c>.
    /// </summary>
    /// <param name="a">Target series element 0 (NumPy's in-place operand: the longer one, or c2 on a tie).</param>
    /// <param name="aStride">Byte stride of <paramref name="a"/>.</param>
    /// <param name="na">Length of <paramref name="a"/> (the result length before trimming).</param>
    /// <param name="b">Other series element 0.</param>
    /// <param name="bStride">Byte stride of <paramref name="b"/>.</param>
    /// <param name="nb">Length of <paramref name="b"/> (<c>nb &lt;= na</c>).</param>
    /// <param name="r">Destination, <paramref name="na"/> contiguous elements of the result dtype.</param>
    /// <returns>trimseq of the result, encoded as <see cref="PolyTrimLenKernel"/> returns it: <paramref name="na"/> for
    ///     the array itself, <c>-k</c> for the slice <c>r[:k]</c>.</returns>
    public unsafe delegate long PolyCombineKernel(byte* a, long aStride, long na, byte* b, long bStride, long nb, byte* r);

    /// <summary>
    ///     trimcoef's scan (NumPy 2.4.2 <c>polyutils.trimcoef</c>): <c>1 + ind[-1]</c> for
    ///     <c>[ind] = np.nonzero(np.abs(c) &gt; tol)</c>, or 0 when no element exceeds <paramref name="tol"/>.
    /// </summary>
    /// <param name="p">Source element 0 (the caller's array, in its own dtype).</param>
    /// <param name="n">Element count.</param>
    /// <param name="byteStride">Signed byte stride.</param>
    /// <param name="tol">One value of the comparison dtype (already converted by NumPy's rules).</param>
    /// <returns>The kept length, or 0 when nothing exceeds the tolerance.</returns>
    public unsafe delegate long PolyLastAboveKernel(byte* p, long n, long byteStride, byte* tol);

    /// <summary>
    ///     <c>np.array(src[:n], copy=True, dtype=T)</c> for a 1-D source: <c>r[i] = (T)p[i]</c> with the house
    ///     (NumPy <c>astype</c>) conversion, strided source into a contiguous destination.
    /// </summary>
    /// <param name="p">Source element 0.</param>
    /// <param name="n">Element count.</param>
    /// <param name="byteStride">Signed source byte stride.</param>
    /// <param name="r">Destination, <paramref name="n"/> contiguous elements of the target dtype.</param>
    public unsafe delegate void PolyCastKernel(byte* p, long n, long byteStride, byte* r);

    /// <summary>One NumPy scalar binary op: <c>*r = *a OP *b</c>, both operands already in the loop dtype.</summary>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    /// <param name="r">Result (may alias an operand).</param>
    public unsafe delegate void PolyScalarBinaryKernel(byte* a, byte* b, byte* r);

    /// <summary>One NumPy scalar unary op (<c>np.negative</c>): <c>*r = -*a</c>.</summary>
    /// <param name="a">Operand.</param>
    /// <param name="r">Result (may alias the operand).</param>
    public unsafe delegate void PolyScalarUnaryKernel(byte* a, byte* r);

    /// <summary>One NumPy scalar comparison against the Python int <c>0</c>.</summary>
    /// <param name="a">Operand.</param>
    /// <returns>The comparison's truth value.</returns>
    public unsafe delegate bool PolyScalarPredicateKernel(byte* a);

    /// <summary>Boxes one element as its CLR type (<c>double</c>, <see cref="Half"/>, <see cref="Complex"/>, …).</summary>
    /// <param name="a">The element.</param>
    /// <returns>The boxed value.</returns>
    public unsafe delegate object PolyScalarBoxKernel(byte* a);

    /// <summary>
    ///     <c>polyutils.mapparms</c> on four NumPy scalars of ONE dtype — the whole of NumPy's six scalarmath
    ///     operations in one call: <c>oldlen = o1 - o0; newlen = n1 - n0; off = (o1*n0 - o0*n1) / oldlen;
    ///     scl = newlen / oldlen</c>.
    /// </summary>
    /// <param name="o1"><c>old[1]</c>.</param>
    /// <param name="o0"><c>old[0]</c>.</param>
    /// <param name="n1"><c>new[1]</c>.</param>
    /// <param name="n0"><c>new[0]</c>.</param>
    /// <param name="off">Result offset (the true-division dtype: float64 for an integer dtype).</param>
    /// <param name="scl">Result scale (same dtype).</param>
    public unsafe delegate void PolyMapParmsKernel(byte* o1, byte* o0, byte* n1, byte* n0, byte* off, byte* scl);

    /// <summary>Which in-place update NumPy's <c>_add</c>/<c>_sub</c> performs (see <see cref="PolyCombineKernel"/>).</summary>
    internal enum PolyCombineOp : byte
    {
        /// <summary><c>a[:nb] += b</c>: <c>r[i] = a[i] + b[i]</c>, then <c>r[i] = a[i]</c>.</summary>
        Add,
        /// <summary><c>a[:nb] -= b</c> (<c>_sub</c> with the longer minuend): <c>r[i] = a[i] - b[i]</c>, then <c>a[i]</c>.</summary>
        Subtract,
        /// <summary><c>a = -a; a[:nb] += b</c> (<c>_sub</c> when the subtrahend is not shorter):
        ///     <c>r[i] = (-a[i]) + b[i]</c>, then <c>-a[i]</c>.</summary>
        NegateAdd,
    }

    /// <summary>The comparison a <see cref="PolyScalarPredicateKernel"/> performs against zero.</summary>
    internal enum PolyZeroTest : byte
    {
        /// <summary><c>x != 0</c> — the truth value <c>{p}line</c> branches on (NaN is nonzero).</summary>
        NotEqual,
        /// <summary><c>x &lt; 0</c> — trimcoef's <c>tol &lt; 0</c> check on a NumPy-scalar tolerance.</summary>
        Less,
    }

    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     Cache key of every polynomial-series kernel: the kernel family, up to three dtypes and one flag
        ///     byte (the op / complex-product form / zero test). Structural equality, so one key per distinct
        ///     kernel for the process lifetime.
        /// </summary>
        /// <param name="Family">Which kernel (see <see cref="PolySeriesFamily"/>).</param>
        /// <param name="T0">First dtype (source / loop dtype).</param>
        /// <param name="T1">Second dtype (other source / common dtype), or <see cref="NPTypeCode.Empty"/>.</param>
        /// <param name="T2">Third dtype (result / comparison dtype), or <see cref="NPTypeCode.Empty"/>.</param>
        /// <param name="Flag">Family-specific selector.</param>
        private readonly record struct PolySeriesKey(PolySeriesFamily Family, NPTypeCode T0, NPTypeCode T1, NPTypeCode T2, byte Flag);

        /// <summary>The kernel families of <see cref="PolySeriesKey"/>.</summary>
        private enum PolySeriesFamily : byte { TrimLen, Combine, LastAbove, Cast, ScalarBinary, ScalarNegate, ScalarPredicate, ScalarBox, MapParms, ScalarBinaryComplex64 }

        /// <summary>
        ///     Every polynomial-series kernel, emitted once per key. Values are the typed delegates of this file;
        ///     a failed emission is NOT cached (the exception propagates), so a transient failure is retried.
        /// </summary>
        private static readonly ConcurrentDictionary<PolySeriesKey, Delegate> s_polySeriesKernels = new();

        /// <summary>NumPy's complex ufunc ordering <c>CGT</c> (loops.c.src): the helper the tolerance scan calls
        ///     when the comparison dtype is complex.</summary>
        private static readonly MethodInfo s_polyComplexGreater = typeof(DirectILKernelGenerator).GetMethod(
            nameof(PolyComplexGreater), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(PolyComplexGreater));

        /// <summary>NumPy's scalarmath complex <c>&lt; 0</c>: the helper the scalar predicate calls for complex.</summary>
        private static readonly MethodInfo s_polyComplexScalarLessZero = typeof(DirectILKernelGenerator).GetMethod(
            nameof(PolyComplexScalarLessZero), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(PolyComplexScalarLessZero));

        /// <summary>
        ///     NumPy's complex comparison ufunc <c>greater</c> (<c>CGT</c> in <c>loops.c.src</c>):
        ///     <c>(a.re &gt; b.re &amp;&amp; !isnan(a.im) &amp;&amp; !isnan(b.im)) || (a.re == b.re &amp;&amp; a.im &gt; b.im)</c>.
        ///     Lexicographic, with a NaN imaginary part blocking the real-part win — NOT the sort order and NOT
        ///     scalarmath's ordering (<see cref="PolyComplexScalarLessZero"/>), which omits the NaN guards.
        /// </summary>
        /// <param name="a">Left operand.</param>
        /// <param name="b">Right operand.</param>
        /// <returns>NumPy's <c>a &gt; b</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool PolyComplexGreater(Complex a, Complex b)
            => (a.Real > b.Real && !double.IsNaN(a.Imaginary) && !double.IsNaN(b.Imaginary))
               || (a.Real == b.Real && a.Imaginary > b.Imaginary);

        /// <summary>
        ///     NumPy's scalarmath complex <c>&lt;</c> against the Python int 0 (<c>np.complex128(z) &lt; 0</c>):
        ///     <c>re &lt; 0 || (re == 0 &amp;&amp; im &lt; 0)</c>, with no NaN guard — probed 2.4.2:
        ///     <c>complex(-1, nan) &lt; 0</c> is True, <c>complex(0, nan) &lt; 0</c> is False.
        /// </summary>
        /// <param name="a">The scalar.</param>
        /// <returns>NumPy's <c>a &lt; 0</c>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool PolyComplexScalarLessZero(Complex a) => a.Real < 0 || (a.Real == 0 && a.Imaginary < 0);

        // ---------------------------------------------------------------------------------------------
        //  complex64 SCALAR arithmetic (NumPy's np.complex64, carried in a Complex)
        // ---------------------------------------------------------------------------------------------
        //
        // NumSharp has one complex dtype (complex128, #569), yet numpy.polynomial's scalar code creates complex64
        // NumPy scalars whenever a float16/float32 NumPy value meets a Python complex (NEP 50): lagline(np.float32(a),
        // 1j) computes `off + scl` in complex64 before np.array widens it, and mapdomain with a float32 domain and a
        // complex tuple derives complex64 off/scl that then scale a float64 array in complex128. A complex64 value is
        // exactly representable in complex128 (two float32 widen exactly), so the SCALAR arithmetic is emulated
        // exactly: the operands are rounded to float32 on the way in (NumPy's conversion into the complex64 loop) and
        // every elementary operation below rounds to float32 — each cast `(float)(…)` is a conv.r4, never an
        // extended-precision intermediate. The three forms are NumPy's complex128 forms in float32, probed on 2.4.2
        // (3000/3000 random full-mantissa pairs each): scalarmath's multiply is the naive product, a 0-d ufunc's is
        // simd_cmul (fused), the division is CFLOAT_divide's un-fused Smith. A complex64 ARRAY (rank >= 1) is not
        // emulated — NumSharp computes such loops in complex128 (the documented #569 dtype divergence).

        /// <summary>
        ///     NumPy's conversion of a value already in complex128 into the complex64 loop: each component rounded to
        ///     float32 (ties to even; out-of-range → ±inf, NaN kept). A float16 / float32 / small-integer operand
        ///     and an existing complex64 value pass unchanged; a Python float or complex rounds once; a Python int has
        ///     already been rounded to float64 by <c>PyLong_AsDouble</c>, so it rounds twice — as NumPy's does.
        /// </summary>
        /// <param name="v">The complex128 carrier.</param>
        /// <returns>The complex64 value, carried in a <see cref="Complex"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64Round(Complex v) => new Complex((float)v.Real, (float)v.Imaginary);

        /// <summary><c>np.complex64 + np.complex64</c>: component-wise float32 addition.</summary>
        /// <param name="a">Left operand (float32-exact components).</param><param name="b">Right operand.</param>
        /// <returns>The complex64 sum.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64Add(Complex a, Complex b)
            => new Complex((float)((float)a.Real + (float)b.Real), (float)((float)a.Imaginary + (float)b.Imaginary));

        /// <summary><c>np.complex64 - np.complex64</c>: component-wise float32 subtraction.</summary>
        /// <param name="a">Left operand (float32-exact components).</param><param name="b">Right operand.</param>
        /// <returns>The complex64 difference.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64Subtract(Complex a, Complex b)
            => new Complex((float)((float)a.Real - (float)b.Real), (float)((float)a.Imaginary - (float)b.Imaginary));

        /// <summary>
        ///     scalarmath's complex64 product (<c>np.complex64 * np.complex64</c>): <c>(ar*br - ai*bi, ar*bi + ai*br)</c>
        ///     with each product rounded to float32 before the add — NumPy's <c>@name@_ctype_multiply</c> for
        ///     <c>npy_cfloat</c>, never contracted.
        /// </summary>
        /// <param name="a">Left operand (float32-exact components).</param><param name="b">Right operand.</param>
        /// <returns>The naive complex64 product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64MultiplyNaive(Complex a, Complex b)
        {
            float ar = (float)a.Real, ai = (float)a.Imaginary, br = (float)b.Real, bi = (float)b.Imaginary;
            return new Complex((float)((float)(ar * br) - (float)(ai * bi)), (float)((float)(ar * bi) + (float)(ai * br)));
        }

        /// <summary>
        ///     The complex64 ufunc product <c>simd_cmul</c> (a 0-d ufunc operand):
        ///     <c>re = fmaf(ar, br, -(ai*bi))</c>, <c>im = fmaf(ar, bi, ai*br)</c> — the complex128 form of
        ///     <see cref="NumSharp.Utilities.NDComplexMath"/>'s multiply, in float32.
        /// </summary>
        /// <param name="a">Left operand (float32-exact components).</param><param name="b">Right operand.</param>
        /// <returns>The fused complex64 product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64MultiplySimd(Complex a, Complex b)
        {
            float ar = (float)a.Real, ai = (float)a.Imaginary, br = (float)b.Real, bi = (float)b.Imaginary;
            return new Complex(MathF.FusedMultiplyAdd(ar, br, -(float)(ai * bi)), MathF.FusedMultiplyAdd(ar, bi, (float)(ai * br)));
        }

        /// <summary>
        ///     The complex64 ufunc's contracted fallback loop (<c>loop_scalar</c>): <c>re = fmaf(ar, br, -(ai*bi))</c>,
        ///     <c>im = fmaf(ai, br, ar*bi)</c> — <see cref="ILKernelGenerator.PolyLoopScalarComplexMultiply"/> in float32.
        ///     Kept for key completeness: the U1 scalar engine never picks it (a complex64 value is never an array).
        /// </summary>
        /// <param name="a">Left operand (float32-exact components).</param><param name="b">Right operand.</param>
        /// <returns>The contracted complex64 product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Complex PolyComplex64MultiplyLoopScalar(Complex a, Complex b)
        {
            float ar = (float)a.Real, ai = (float)a.Imaginary, br = (float)b.Real, bi = (float)b.Imaginary;
            return new Complex(MathF.FusedMultiplyAdd(ar, br, -(float)(ai * bi)), MathF.FusedMultiplyAdd(ai, br, (float)(ar * bi)));
        }

        /// <summary>
        ///     <c>CFLOAT_divide</c> (NumPy <c>loops.c.src</c>, which scalarmath's complex64 division calls with one
        ///     element): <see cref="ComplexDivideNumPy"/>'s Smith algorithm with the reciprocal formed once and
        ///     multiplied, every step rounded to float32 and un-fused (the loop is baseline code: MSVC
        ///     <c>/fp:precise</c> does not contract it). Probed 2.4.2: 3000/3000 random pairs, scalar and 0-d alike.
        /// </summary>
        /// <param name="a">Dividend (float32-exact components).</param><param name="b">Divisor.</param>
        /// <returns>The complex64 quotient; a zero divisor gives the component-wise IEEE inf/nan.</returns>
        internal static Complex PolyComplex64Divide(Complex a, Complex b)
        {
            float in1r = (float)a.Real, in1i = (float)a.Imaginary, in2r = (float)b.Real, in2i = (float)b.Imaginary;
            float in2rAbs = MathF.Abs(in2r), in2iAbs = MathF.Abs(in2i);
            if (in2rAbs >= in2iAbs)
            {
                // Divide by zero: component-wise division by +0 (NumPy's complex inf / nan).
                if (in2rAbs == 0f && in2iAbs == 0f)
                    return new Complex((float)(in1r / in2rAbs), (float)(in1i / in2iAbs));
                float rat = (float)(in2i / in2r);
                float scl = (float)(1.0f / (float)(in2r + (float)(in2i * rat)));
                return new Complex((float)((float)(in1r + (float)(in1i * rat)) * scl), (float)((float)(in1i - (float)(in1r * rat)) * scl));
            }
            float rat2 = (float)(in2r / in2i);
            float scl2 = (float)(1.0f / (float)(in2i + (float)(in2r * rat2)));
            return new Complex((float)((float)((float)(in1r * rat2) + in1i) * scl2), (float)((float)((float)(in1i * rat2) - in1r) * scl2));
        }

        // ---------------------------------------------------------------------------------------------
        //  complex64 ARRAY loops (polyutils.mapdomain)
        // ---------------------------------------------------------------------------------------------
        //
        // mapdomain's `off + scl*x` runs NumPy's complex64 ufunc loops when NEP 50 makes an operation complex64 (a
        // Python complex meeting a float16/float32 x, or a complex64 off/scl meeting a narrow x). NumSharp has no
        // complex64 dtype (#569), so the result is a complex128 array — but these kernels store NumPy's complex64
        // VALUES in it, exactly: each operation in float32, with CFLOAT_multiply's own product form. They are
        // whole-array loops with an AVX2+FMA vector body (8 float32 lanes) and a scalar remainder computing the SAME
        // per-lane arithmetic, so the vector/remainder split never changes a bit.

        /// <summary>
        ///     Stores 8 complex64 values given as separate real / imaginary float32 lanes into 8 contiguous
        ///     <see cref="Complex"/> carriers (16 doubles, interleaved re/im) — each float32 widened exactly.
        /// </summary>
        /// <param name="re">The 8 real parts.</param>
        /// <param name="im">The 8 imaginary parts.</param>
        /// <param name="dst">The first of 16 doubles.</param>
        /// <remarks>Requires AVX. <c>unpacklo/hi</c> interleave within each 128-bit half, so the four 4-double stores
        ///     are ordered lo.lower, hi.lower, lo.upper, hi.upper to land elements 0..7 in order.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void PolyStoreComplex64(Vector256<float> re, Vector256<float> im, double* dst)
        {
            var lo = Avx.UnpackLow(re, im);    // r0 i0 r1 i1 | r4 i4 r5 i5
            var hi = Avx.UnpackHigh(re, im);   // r2 i2 r3 i3 | r6 i6 r7 i7
            Avx.Store(dst, Avx.ConvertToVector256Double(lo.GetLower()));
            Avx.Store(dst + 4, Avx.ConvertToVector256Double(hi.GetLower()));
            Avx.Store(dst + 8, Avx.ConvertToVector256Double(lo.GetUpper()));
            Avx.Store(dst + 12, Avx.ConvertToVector256Double(hi.GetUpper()));
        }

        /// <summary>
        ///     <c>scl * x</c> in NumPy's complex64 loop, for a complex64 scalar <c>scl = sr + si·j</c> and a REAL
        ///     float32 array x promoted to complex64 (imaginary part +0): <c>CFLOAT_multiply</c>'s scalar-broadcast
        ///     vector loop, i.e. <c>simd_cmul</c> — <c>re = fma(sr, x, -(si·0))</c>, <c>im = fma(sr, 0, si·x)</c> — at
        ///     every length: a scalar operand makes the ufunc call trivially iterable, so even a one-element x runs the
        ///     vector loop with a nonzero output stride (probed 2.4.2: <c>s * np.array([x])</c> matches simd_cmul on
        ///     300/300 draws, the stride-0 <c>loop_scalar</c> form on 188/300 for complex128 x).
        /// </summary>
        /// <param name="x">The float32 values (contiguous; exact conversions of the caller's points).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="sr">The scale's real part (float32).</param>
        /// <param name="si">The scale's imaginary part (float32).</param>
        /// <param name="r">Destination: <paramref name="n"/> complex carriers of the complex64 products.</param>
        internal static unsafe void PolyComplex64ScaleReal(float* x, long n, float sr, float si, Complex* r)
        {
            // -(a_im * b_im) with b_im = +0: one constant for the whole array (NaN when si is inf/NaN, else ±0).
            float c = -(si * 0f);
            long i = 0;
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                var vsr = Vector256.Create(sr);
                var vsi = Vector256.Create(si);
                var vc = Vector256.Create(c);
                for (; i + 8 <= n; i += 8)
                {
                    var vx = Avx.LoadVector256(x + i);
                    var re = Fma.MultiplyAdd(vsr, vx, vc);
                    var im = Fma.MultiplyAdd(vsr, Vector256<float>.Zero, Avx.Multiply(vsi, vx));
                    PolyStoreComplex64(re, im, (double*)(r + i));
                }
            }
            for (; i < n; i++)
            {
                float xv = x[i];
                r[i] = new Complex(MathF.FusedMultiplyAdd(sr, xv, c), MathF.FusedMultiplyAdd(sr, 0f, si * xv));
            }
        }

        /// <summary>
        ///     <c>off + scl*x</c> with BOTH operations in NumPy's complex64 loops, in one pass: the product of
        ///     <see cref="PolyComplex64ScaleReal"/> (the same simd_cmul form at every length) followed by <c>CFLOAT_add</c>'s
        ///     component-wise float32 sum with <c>off</c>, each component rounded to float32 at the same two points
        ///     NumPy rounds it — the intermediate product never leaves a register, which halves the memory traffic of
        ///     a product pass followed by a sum pass.
        /// </summary>
        /// <param name="x">The float32 values (contiguous; exact conversions of the caller's points).</param>
        /// <param name="n">Element count (≥ 1).</param>
        /// <param name="sr">The scale's real part (float32).</param>
        /// <param name="si">The scale's imaginary part (float32).</param>
        /// <param name="or">The offset's real part (float32).</param>
        /// <param name="oi">The offset's imaginary part (float32).</param>
        /// <param name="r">Destination: <paramref name="n"/> complex carriers.</param>
        internal static unsafe void PolyComplex64AffineReal(float* x, long n, float sr, float si, float or, float oi, Complex* r)
        {
            float c = -(si * 0f);
            long i = 0;
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                var vsr = Vector256.Create(sr);
                var vsi = Vector256.Create(si);
                var vc = Vector256.Create(c);
                var vor = Vector256.Create(or);
                var voi = Vector256.Create(oi);
                for (; i + 8 <= n; i += 8)
                {
                    var vx = Avx.LoadVector256(x + i);
                    var re = Avx.Add(vor, Fma.MultiplyAdd(vsr, vx, vc));
                    var im = Avx.Add(voi, Fma.MultiplyAdd(vsr, Vector256<float>.Zero, Avx.Multiply(vsi, vx)));
                    PolyStoreComplex64(re, im, (double*)(r + i));
                }
            }
            for (; i < n; i++)
            {
                float xv = x[i];
                float pr = MathF.FusedMultiplyAdd(sr, xv, c);
                float pi = MathF.FusedMultiplyAdd(sr, 0f, si * xv);
                r[i] = new Complex(or + pr, oi + pi);
            }
        }

        /// <summary>
        ///     <c>off + v</c> in NumPy's complex64 loop for a REAL float32 array v promoted to complex64 (imaginary
        ///     part +0) and a complex64 scalar <c>off = or + oi·j</c>: <c>(or + v, oi + 0)</c> component-wise in
        ///     float32 — the imaginary part is the same for every element, and a <c>-0</c> offset imaginary part
        ///     turns into <c>+0</c> exactly as NumPy's addition does.
        /// </summary>
        /// <param name="v">The float32 values (contiguous; an exact conversion of a float16/float32 product).</param>
        /// <param name="n">Element count.</param>
        /// <param name="or">The offset's real part (float32).</param>
        /// <param name="oi">The offset's imaginary part (float32).</param>
        /// <param name="r">Destination: <paramref name="n"/> complex carriers.</param>
        internal static unsafe void PolyComplex64AddScalarReal(float* v, long n, float or, float oi, Complex* r)
        {
            float im = oi + 0f;
            long i = 0;
            if (Avx.IsSupported)
            {
                var vor = Vector256.Create(or);
                var vim = Vector256.Create(im);
                for (; i + 8 <= n; i += 8)
                    PolyStoreComplex64(Avx.Add(vor, Avx.LoadVector256(v + i)), vim, (double*)(r + i));
            }
            for (; i < n; i++)
                r[i] = new Complex(or + v[i], im);
        }

        /// <summary>
        ///     mapdomain's <c>off + scl*x</c> for a complex128 x and complex128 <c>off</c>/<c>scl</c>, in one pass:
        ///     <c>CDOUBLE_multiply</c>'s scalar-broadcast vector loop — <c>simd_cmul</c>, <c>re = fma(sr, xr, -(si·xi))</c>,
        ///     <c>im = fma(sr, xi, si·xr)</c>, at every length (the scalar operand makes the call trivially iterable) —
        ///     then <c>CDOUBLE_add</c>'s component-wise sum with <c>off</c>. Two complex values per AVX2 vector:
        ///     one lane permute, one multiply, one <c>vfmaddsub</c>, one add.
        /// </summary>
        /// <param name="x">Element 0 of the points (complex128).</param>
        /// <param name="n">Element count.</param>
        /// <param name="xStride">Signed ELEMENT stride of the points (1 contiguous, -1 reversed, 2 every other, …).</param>
        /// <param name="scl">The scale.</param>
        /// <param name="off">The offset.</param>
        /// <param name="r">Destination: <paramref name="n"/> contiguous complex128 values (not overlapping
        ///     <paramref name="x"/>).</param>
        /// <remarks>Bit-identical to the fused np.evaluate pass it replaces for this shape (which computes the same
        ///     simd_cmul per element in scalar code) and to NumPy; only a surviving NaN's payload may differ, which is
        ///     not contractual. A complex128 value is exactly one 128-bit lane, so a strided x costs two 16-byte loads
        ///     per vector instead of one 32-byte load — the same arithmetic, which is why one kernel serves every
        ///     stride.</remarks>
        internal static unsafe void PolyComplex128Affine(Complex* x, long n, long xStride, Complex scl, Complex off, Complex* r)
        {
            double sr = scl.Real, si = scl.Imaginary, or = off.Real, oi = off.Imaginary;
            double* rp = (double*)r;
            long i = 0;
            if (Avx2.IsSupported && Fma.IsSupported)
            {
                var vsr = Vector256.Create(sr);
                var vsi = Vector256.Create(si);
                var vo = Vector256.Create(or, oi, or, oi);
                for (; i + 2 <= n; i += 2)
                {
                    // xr0 xi0 xr1 xi1 — one load when contiguous, the two 128-bit values otherwise.
                    Vector256<double> b = xStride == 1
                        ? Avx.LoadVector256((double*)(x + i))
                        : Vector256.Create(Sse2.LoadVector128((double*)(x + i * xStride)), Sse2.LoadVector128((double*)(x + (i + 1) * xStride)));
                    var bRev = Avx.Permute(b, 0b0101);                // xi0 xr0 xi1 xr1
                    // simd_cmul: even lanes sr*xr - si*xi, odd lanes sr*xi + si*xr (vfmaddsub).
                    var prod = Fma.MultiplyAddSubtract(vsr, b, Avx.Multiply(vsi, bRev));
                    Avx.Store(rp + 2 * i, Avx.Add(vo, prod));
                }
            }
            for (; i < n; i++)
            {
                double* xe = (double*)(x + i * xStride);
                double xr = xe[0], xi = xe[1];
                double re = Math.FusedMultiplyAdd(sr, xr, -(si * xi));
                double im = Math.FusedMultiplyAdd(sr, xi, si * xr);
                rp[2 * i] = or + re;
                rp[2 * i + 1] = oi + im;
            }
        }

        /// <summary>
        ///     mapdomain's <c>off + scl*x</c> in a float64 loop: NumPy's two ufunc calls — <c>DOUBLE_multiply</c> of the
        ///     scale and the points, then <c>DOUBLE_add</c> of the offset and the product — as one pass, each operation
        ///     rounded on its own (a multiply, then an add: never a fused multiply-add, which would round once and
        ///     differ in the last bit). Eight points per iteration as two AVX vectors.
        /// </summary>
        /// <param name="x">The points, contiguous float64 (already converted to the loop dtype).</param>
        /// <param name="n">Point count.</param>
        /// <param name="scl">The scale, in the loop dtype.</param>
        /// <param name="off">The offset, in the loop dtype.</param>
        /// <param name="r">Destination: <paramref name="n"/> contiguous float64 values (may be <paramref name="x"/> itself,
        ///     the pass reading each point before writing it; must not otherwise overlap it).</param>
        /// <remarks>Bit-identical to NumPy and to the fused np.evaluate pass it replaces; a surviving NaN's payload may
        ///     differ, which is not contractual. The operand order of each operation is NumPy's (<c>scl * x</c>, then
        ///     <c>off + product</c>).</remarks>
        internal static unsafe void PolyAffineDouble(double* x, long n, double scl, double off, double* r)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                var vs = Vector256.Create(scl);
                var vo = Vector256.Create(off);
                for (; i + 8 <= n; i += 8)
                {
                    // Explicit multiply and add intrinsics: nothing may contract them into an FMA.
                    Avx.Store(r + i, Avx.Add(vo, Avx.Multiply(vs, Avx.LoadVector256(x + i))));
                    Avx.Store(r + i + 4, Avx.Add(vo, Avx.Multiply(vs, Avx.LoadVector256(x + i + 4))));
                }
            }
            // RyuJIT never contracts a scalar multiply and add either.
            for (; i < n; i++)
                r[i] = off + scl * x[i];
        }

        /// <summary>
        ///     <see cref="PolyAffineDouble"/> for a float32 loop: <c>FLOAT_multiply</c> then <c>FLOAT_add</c>, each rounded
        ///     to float32 on its own, sixteen points per iteration.
        /// </summary>
        /// <param name="x">The points, contiguous float32.</param>
        /// <param name="n">Point count.</param>
        /// <param name="scl">The scale.</param>
        /// <param name="off">The offset.</param>
        /// <param name="r">Destination: <paramref name="n"/> contiguous float32 values (may be <paramref name="x"/>).</param>
        /// <remarks>As <see cref="PolyAffineDouble"/>: two float32 roundings per point, never an FMA — and never a float64
        ///     intermediate, which would round the product differently.</remarks>
        internal static unsafe void PolyAffineSingle(float* x, long n, float scl, float off, float* r)
        {
            long i = 0;
            if (Avx.IsSupported)
            {
                var vs = Vector256.Create(scl);
                var vo = Vector256.Create(off);
                for (; i + 16 <= n; i += 16)
                {
                    Avx.Store(r + i, Avx.Add(vo, Avx.Multiply(vs, Avx.LoadVector256(x + i))));
                    Avx.Store(r + i + 8, Avx.Add(vo, Avx.Multiply(vs, Avx.LoadVector256(x + i + 8))));
                }
            }
            for (; i < n; i++)
                r[i] = off + scl * x[i];
        }

        /// <summary>
        ///     The trimseq scan kernel for elements of <paramref name="t"/> (see <see cref="PolyTrimLenKernel"/>).
        /// </summary>
        /// <param name="t">Element dtype (any NumSharp dtype).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException"><paramref name="t"/> has no house comparison.</exception>
        internal static PolyTrimLenKernel GetPolyTrimLenKernel(NPTypeCode t)
            => (PolyTrimLenKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.TrimLen, t, NPTypeCode.Empty, NPTypeCode.Empty, 0),
                static k => EmitPolyTrimLen(k.T0));

        /// <summary>
        ///     The <c>_add</c>/<c>_sub</c> kernel (see <see cref="PolyCombineKernel"/>).
        /// </summary>
        /// <param name="op">Which in-place update NumPy performs.</param>
        /// <param name="ta">Dtype of the in-place target operand.</param>
        /// <param name="tb">Dtype of the other operand.</param>
        /// <param name="tr">Result dtype (np.common_type of both).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">A dtype pair the house conversion or op does not support.</exception>
        internal static PolyCombineKernel GetPolyCombineKernel(PolyCombineOp op, NPTypeCode ta, NPTypeCode tb, NPTypeCode tr)
            => (PolyCombineKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.Combine, ta, tb, tr, (byte)op),
                static k => EmitPolyCombine((PolyCombineOp)k.Flag, k.T0, k.T1, k.T2));

        /// <summary>
        ///     The trimcoef tolerance scan (see <see cref="PolyLastAboveKernel"/>).
        /// </summary>
        /// <param name="ts">Source dtype (the caller's array).</param>
        /// <param name="tc">Coefficient dtype (np.common_type of the source) the element is converted to first.</param>
        /// <param name="tk">Comparison dtype: NumPy's <c>result_type(abs(c), tol)</c>.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">A dtype the house conversion, abs or comparison does not support.</exception>
        internal static PolyLastAboveKernel GetPolyLastAboveKernel(NPTypeCode ts, NPTypeCode tc, NPTypeCode tk)
            => (PolyLastAboveKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.LastAbove, ts, tc, tk, 0),
                static k => EmitPolyLastAbove(k.T0, k.T1, k.T2));

        /// <summary>
        ///     The strided convert-copy (see <see cref="PolyCastKernel"/>).
        /// </summary>
        /// <param name="ts">Source dtype.</param>
        /// <param name="tr">Destination dtype.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">A pair the house conversion does not support.</exception>
        internal static PolyCastKernel GetPolyCastKernel(NPTypeCode ts, NPTypeCode tr)
            => (PolyCastKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.Cast, ts, tr, NPTypeCode.Empty, 0),
                static k => EmitPolyCast(k.T0, k.T1));

        /// <summary>
        ///     One NumPy scalar binary op in dtype <paramref name="t"/> (see <see cref="PolyScalarBinaryKernel"/>).
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide (true division; <paramref name="t"/> must be inexact).</param>
        /// <param name="t">The loop dtype both operands were converted to.</param>
        /// <param name="product">Which of NumPy's complex products a complex multiply reproduces: scalarmath's
        ///     naive one for NumPy scalars, the ufunc's for arrays (ignored for every other op and dtype).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">An op the dtype has no house loop for.</exception>
        /// <remarks>A bool loop maps add to logical OR and multiply to logical AND — NumPy's bool loops — because
        ///     the house integer <c>add</c> opcode would store 2 for <c>True + True</c>.</remarks>
        internal static PolyScalarBinaryKernel GetPolyScalarBinaryKernel(BinaryOp op, NPTypeCode t, PolyComplexProduct product)
        {
            // The scalar engine calls this once per NumPy-scalar operation (mapparms runs seven), so a flat slot
            // array fronts the dictionary: the four arithmetic ops x three products x every NPTypeCode value.
            int slot = (int)op <= (int)BinaryOp.Divide && (int)t < PolyDtypeSlots
                ? ((int)op * 3 + (int)product) * PolyDtypeSlots + (int)t : -1;
            if (slot >= 0 && s_polyScalarBinarySlots[slot] is { } fast)
                return fast;
            var k = (PolyScalarBinaryKernel)s_polySeriesKernels.GetOrAdd(
                new PolySeriesKey(PolySeriesFamily.ScalarBinary, t, NPTypeCode.Empty, NPTypeCode.Empty, (byte)((int)op * 4 + (int)product)),
                static key => EmitPolyScalarBinary((BinaryOp)(key.Flag / 4), key.T0, (PolyComplexProduct)(key.Flag % 4)));
            if (slot >= 0)
                Volatile.Write(ref s_polyScalarBinarySlots[slot], k);
            return k;
        }

        /// <summary>
        ///     One NumPy complex64 SCALAR binary op (see <see cref="PolyComplex64Round"/>): both operands already
        ///     rounded into the complex64 loop and carried as <see cref="Complex"/>, the result complex64 as well.
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="product">Which complex product a multiply reproduces: scalarmath's naive one for NumPy
        ///     scalars, <c>simd_cmul</c> for a 0-d ufunc operand (ignored for every other op).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">An op other than the four arithmetic ones.</exception>
        internal static PolyScalarBinaryKernel GetPolyScalarComplex64Kernel(BinaryOp op, PolyComplexProduct product)
        {
            if ((int)op > (int)BinaryOp.Divide)
                throw new NotSupportedException($"complex64 scalar {op} has no NumPy scalarmath loop here");
            int slot = (int)op * 3 + (int)product;
            if (s_polyScalarComplex64Slots[slot] is { } fast)
                return fast;
            var k = (PolyScalarBinaryKernel)s_polySeriesKernels.GetOrAdd(
                new PolySeriesKey(PolySeriesFamily.ScalarBinaryComplex64, NPTypeCode.Complex, NPTypeCode.Empty, NPTypeCode.Empty, (byte)slot),
                static key => EmitPolyScalarComplex64((BinaryOp)(key.Flag / 3), (PolyComplexProduct)(key.Flag % 3)));
            Volatile.Write(ref s_polyScalarComplex64Slots[slot], k);
            return k;
        }

        /// <summary>Fast front of <see cref="GetPolyScalarComplex64Kernel"/>: four ops × three products.</summary>
        private static readonly PolyScalarBinaryKernel[] s_polyScalarComplex64Slots = new PolyScalarBinaryKernel[4 * 3];

        /// <summary>Slots per op/product group of the fast lookup arrays: one per NPTypeCode value (Complex = 128 is the largest).</summary>
        private const int PolyDtypeSlots = 129;

        /// <summary>Fast front of <see cref="GetPolyScalarBinaryKernel"/> (null until first use of a slot).</summary>
        private static readonly PolyScalarBinaryKernel[] s_polyScalarBinarySlots = new PolyScalarBinaryKernel[4 * 3 * PolyDtypeSlots];

        /// <summary>Fast front of <see cref="GetPolyScalarNegateKernel"/>.</summary>
        private static readonly PolyScalarUnaryKernel[] s_polyScalarNegateSlots = new PolyScalarUnaryKernel[PolyDtypeSlots];

        /// <summary>Fast front of <see cref="GetPolyScalarPredicateKernel"/> (two tests per dtype).</summary>
        private static readonly PolyScalarPredicateKernel[] s_polyScalarPredicateSlots = new PolyScalarPredicateKernel[2 * PolyDtypeSlots];

        /// <summary>Fast front of <see cref="GetPolyScalarBoxKernel"/>.</summary>
        private static readonly PolyScalarBoxKernel[] s_polyScalarBoxSlots = new PolyScalarBoxKernel[PolyDtypeSlots];

        /// <summary>One NumPy scalar negation in dtype <paramref name="t"/> (unsigned integers wrap, floats flip the sign bit).</summary>
        /// <param name="t">The dtype (never bool: NumPy refuses boolean negative, and the caller raises first).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">The dtype has no house negate.</exception>
        internal static PolyScalarUnaryKernel GetPolyScalarNegateKernel(NPTypeCode t)
        {
            if ((int)t < PolyDtypeSlots && s_polyScalarNegateSlots[(int)t] is { } fast)
                return fast;
            var k = (PolyScalarUnaryKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.ScalarNegate, t, NPTypeCode.Empty, NPTypeCode.Empty, 0),
                static key => EmitPolyScalarNegate(key.T0));
            if ((int)t < PolyDtypeSlots)
                Volatile.Write(ref s_polyScalarNegateSlots[(int)t], k);
            return k;
        }

        /// <summary>One NumPy scalar comparison against zero in dtype <paramref name="t"/>.</summary>
        /// <param name="t">The scalar's dtype (comparing with the weak Python int 0 keeps it: 0 fits every dtype).</param>
        /// <param name="test">Which comparison.</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException">The dtype has no house comparison for <paramref name="test"/>.</exception>
        internal static PolyScalarPredicateKernel GetPolyScalarPredicateKernel(NPTypeCode t, PolyZeroTest test)
        {
            int slot = (int)t < PolyDtypeSlots ? (int)test * PolyDtypeSlots + (int)t : -1;
            if (slot >= 0 && s_polyScalarPredicateSlots[slot] is { } fast)
                return fast;
            var k = (PolyScalarPredicateKernel)s_polySeriesKernels.GetOrAdd(
                new PolySeriesKey(PolySeriesFamily.ScalarPredicate, t, NPTypeCode.Empty, NPTypeCode.Empty, (byte)test),
                static key => EmitPolyScalarPredicate(key.T0, (PolyZeroTest)key.Flag));
            if (slot >= 0)
                Volatile.Write(ref s_polyScalarPredicateSlots[slot], k);
            return k;
        }

        /// <summary>
        ///     Boxes one element of <paramref name="t"/> as its CLR type — how the scalar engine hands a NumPy
        ///     scalar back to C# without building a 0-d NDArray (a box is ~15 ns, an NDArray ~200 ns).
        /// </summary>
        /// <param name="t">The dtype.</param>
        /// <returns>The cached kernel.</returns>
        internal static PolyScalarBoxKernel GetPolyScalarBoxKernel(NPTypeCode t)
        {
            if ((int)t < PolyDtypeSlots && s_polyScalarBoxSlots[(int)t] is { } fast)
                return fast;
            var k = (PolyScalarBoxKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.ScalarBox, t, NPTypeCode.Empty, NPTypeCode.Empty, 0),
                static key => EmitPolyScalarBox(key.T0));
            if ((int)t < PolyDtypeSlots)
                Volatile.Write(ref s_polyScalarBoxSlots[(int)t], k);
            return k;
        }

        /// <summary>Fast front of <see cref="GetPolyMapParmsKernel"/>.</summary>
        private static readonly PolyMapParmsKernel[] s_polyMapParmsSlots = new PolyMapParmsKernel[PolyDtypeSlots];

        /// <summary>
        ///     The fused <c>polyutils.mapparms</c> of four NumPy scalars of dtype <paramref name="t"/> (see
        ///     <see cref="PolyMapParmsKernel"/>): the same scalarmath operations, in the same order, as six calls of
        ///     <see cref="GetPolyScalarBinaryKernel"/> — each intermediate rounded/wrapped in <paramref name="t"/>, the
        ///     naive complex product, and the two true divisions in float64 for an integer dtype — so it is
        ///     bit-identical to the per-operation chain while costing one call instead of six.
        /// </summary>
        /// <param name="t">The four scalars' dtype (never bool: NumPy refuses boolean subtraction, raised by the caller's
        ///     per-operation path).</param>
        /// <returns>The cached kernel.</returns>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is bool, or has no house arithmetic.</exception>
        internal static PolyMapParmsKernel GetPolyMapParmsKernel(NPTypeCode t)
        {
            if ((int)t < PolyDtypeSlots && s_polyMapParmsSlots[(int)t] is { } fast)
                return fast;
            var k = (PolyMapParmsKernel)s_polySeriesKernels.GetOrAdd(new PolySeriesKey(PolySeriesFamily.MapParms, t, NPTypeCode.Empty, NPTypeCode.Empty, 0),
                static key => EmitPolyMapParms(key.T0));
            if ((int)t < PolyDtypeSlots)
                Volatile.Write(ref s_polyMapParmsSlots[(int)t], k);
            return k;
        }

        // ---------------------------------------------------------------------------------------------
        //  Long series: routing onto the house SIMD kernels
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Series length from which the orchestration (<c>NDPolySeries</c>) runs a conversion copy or an
        ///     <c>_add</c>/<c>_sub</c> through the house vector kernels (memcpy, <see cref="TryGetCastKernel"/>, the
        ///     same-dtype <see cref="ExecutionPath.SimdFull"/> binary kernel) instead of this file's fused scalar loops.
        /// </summary>
        /// <remarks>
        ///     Below it the fused scalar kernel wins: it makes one pass and needs no kernel lookup, while the house
        ///     route pays a lookup per kernel and, for a converted or negated operand, an extra pass over the result.
        ///     Measured on AVX2 at n = 1000: the house float32 add is 0.28 µs against 0.50 µs fused, float16 1.16 µs
        ///     against 5.45 µs (the fused kernel widens float16 one element at a time), float64 0.58 µs against
        ///     0.47 µs — the float64 loss is ~0.1 µs of lookups on a call NumPy takes several µs for, so one
        ///     dtype-independent threshold is kept rather than a per-dtype table.
        /// </remarks>
        internal const long PolyHouseKernelThreshold = 64;

        /// <summary>
        ///     Resolved house contiguous cast kernels, one slot per (source, target) dtype pair indexed by
        ///     <see cref="PolyPairSlot"/>: null = not yet resolved, <see cref="s_polyNoCastKernel"/> = the pair has
        ///     no house kernel. <see cref="TryGetCastKernel"/> walks a dozen specialized resolvers before its own
        ///     cache, so the answer is kept here per pair.
        /// </summary>
        private static readonly object[] s_polyContiguousCasts = new object[32 * 32];

        /// <summary>Sentinel for "resolved: no house contiguous cast kernel" in <see cref="s_polyContiguousCasts"/>.</summary>
        private static readonly object s_polyNoCastKernel = new object();

        /// <summary>
        ///     The slot of a dtype pair in a 32×32 table. Every storage dtype's code is below 32 except
        ///     <see cref="NPTypeCode.Complex"/> (128), which masks to 0 — the slot of <see cref="NPTypeCode.Empty"/>, a
        ///     code no array carries — so the mapping is collision-free over the dtypes that reach it.
        /// </summary>
        /// <param name="ts">Source dtype.</param>
        /// <param name="tr">Target dtype.</param>
        /// <returns>The slot index.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int PolyPairSlot(NPTypeCode ts, NPTypeCode tr) => ((int)ts & 31) * 32 + ((int)tr & 31);

        /// <summary>
        ///     The house contiguous cast kernel for <paramref name="ts"/> → <paramref name="tr"/> (the SIMD widen /
        ///     convert kernels astype runs, bit-exact with NumPy's cast and gated by the astype oracle tier), or null
        ///     when the house has none for the pair — the caller then keeps <see cref="GetPolyCastKernel"/>.
        /// </summary>
        /// <param name="ts">Source dtype (never equal to <paramref name="tr"/>: a same-dtype copy is a memcpy).</param>
        /// <param name="tr">Target dtype.</param>
        /// <returns>The kernel, or null.</returns>
        /// <remarks>The first answer for a pair is kept for the life of the process, including a null seen while IL
        ///     generation was disabled — which only means the pair keeps the (equally exact) scalar poly kernel.</remarks>
        internal static CastKernel GetPolyContiguousCast(NPTypeCode ts, NPTypeCode tr)
        {
            int slot = PolyPairSlot(ts, tr);
            object resolved = Volatile.Read(ref s_polyContiguousCasts[slot]);
            if (resolved is null)
            {
                resolved = (object)TryGetCastKernel(ts, tr) ?? s_polyNoCastKernel;
                Volatile.Write(ref s_polyContiguousCasts[slot], resolved);
            }
            return resolved as CastKernel;
        }

        /// <summary>Resolved house strided cast kernels per dtype pair (same encoding as <see cref="s_polyContiguousCasts"/>).</summary>
        private static readonly object[] s_polyStridedCasts = new object[32 * 32];

        /// <summary>
        ///     The house strided cast kernel for <paramref name="ts"/> → <paramref name="tr"/> (element strides; the
        ///     sub-word SIMD deinterleave / stage-and-widen kernels for float16), or null when the house has none.
        /// </summary>
        /// <param name="ts">Source dtype (may equal <paramref name="tr"/>: a same-dtype strided copy).</param>
        /// <param name="tr">Target dtype.</param>
        /// <returns>The kernel, or null.</returns>
        /// <remarks>Only worth calling where <see cref="PolyScalarIsEmulated"/> holds for a dtype of the pair: for a
        ///     4/8-byte source the scalar poly cast kernel measured FASTER than the house's generic strided cast
        ///     (100K float32 stride-2: 20 µs against 29 µs).</remarks>
        internal static StridedCastKernel GetPolyStridedCast(NPTypeCode ts, NPTypeCode tr)
        {
            int slot = PolyPairSlot(ts, tr);
            object resolved = Volatile.Read(ref s_polyStridedCasts[slot]);
            if (resolved is null)
            {
                resolved = (object)TryGetStridedCastKernel(ts, tr) ?? s_polyNoCastKernel;
                Volatile.Write(ref s_polyStridedCasts[slot], resolved);
            }
            return resolved as StridedCastKernel;
        }

        /// <summary>
        ///     Whether this file's scalar kernels EMULATE element work in dtype <paramref name="t"/> in software — the
        ///     cost-model fact behind every "house kernel even though it adds a pass" decision of the orchestration.
        ///     float16 is the one: .NET has no scalar float16 arithmetic or conversion instruction, so each element a
        ///     fused kernel touches is widened (and, for a float16 result, narrowed) by bit-level helpers — ~1.4 ns
        ///     for a widen alone, where the house vector kernels do the same exact work 8 lanes at a time. Every
        ///     other storage dtype maps to hardware scalar instructions, for which the fused kernel is already near
        ///     the memory bound whenever a conversion or a strided read is involved.
        /// </summary>
        /// <param name="t">The dtype.</param>
        /// <returns>True for float16.</returns>
        internal static bool PolyScalarIsEmulated(NPTypeCode t) => t == NPTypeCode.Half;

        // ---------------------------------------------------------------------------------------------
        //  Emission
        // ---------------------------------------------------------------------------------------------

        /// <summary>A DynamicMethod owned by this class (skipVisibility, so the house helpers are callable).</summary>
        /// <param name="name">Kernel name (shows in JIT disassembly).</param>
        /// <param name="ret">Return type.</param>
        /// <param name="args">Parameter types.</param>
        /// <returns>The method.</returns>
        private static DynamicMethod NewPolySeriesMethod(string name, Type ret, params Type[] args)
            => new DynamicMethod(name, ret, args, typeof(DirectILKernelGenerator), skipVisibility: true);

        /// <summary>
        ///     Emits the trimseq scan over <c>[pLoc, n, stride]</c> and RETURNS its result (the emitted code ends in
        ///     <c>ret</c>), encoded as <see cref="PolyTrimLenKernel"/> documents: <c>n == 0 → 0</c>; a nonzero last
        ///     element → <c>n</c> (NumPy's early <c>return seq</c>); otherwise the first <c>i</c> from <c>n-2</c> down
        ///     to 1 with <c>p[i] != 0</c> gives <c>-(i + 1)</c>, and reaching <c>i == 0</c> gives <c>-1</c> without
        ///     reading <c>p[0]</c> — NumPy's loop breaks at the first nonzero from the end, and when none is found it
        ///     leaves <c>i = 0</c> and slices <c>seq[:1]</c> whatever <c>seq[0]</c> is.
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="p">Local holding element 0's address (overwritten: it walks backwards).</param>
        /// <param name="emitN">Pushes the element count (a long).</param>
        /// <param name="emitStride">Pushes the byte stride (a long).</param>
        /// <param name="t">Element dtype.</param>
        private static void EmitPolyTrimScanAndReturn(ILGenerator il, LocalBuilder p, Action emitN, Action emitStride, NPTypeCode t)
        {
            var i = il.DeclareLocal(typeof(long));
            var loop = il.DefineLabel();
            var one = il.DefineLabel();
            var found = il.DefineLabel();

            // if (n == 0) return 0;
            var notEmpty = il.DefineLabel();
            emitN();
            il.Emit(OpCodes.Brtrue, notEmpty);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ret);
            il.MarkLabel(notEmpty);

            // i = n - 1;  p += i * stride   (the last element)
            emitN();
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldloc, i);
            emitStride();
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, p);

            // `if seq[-1] != 0: return seq` — the whole sequence, reported as the positive length n.
            var sliced = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, p);
            EmitLoadIndirect(il, t);
            WhereNode.EmitPushZeroPublic(il, t);
            EmitComparisonOperation(il, ComparisonOp.NotEqual, t);
            il.Emit(OpCodes.Brfalse, sliced);
            emitN();
            il.Emit(OpCodes.Ret);

            // for i in n-2 .. 1: if (p[i] != 0) return -(i + 1);   return -1;   (a slice from here on)
            il.MarkLabel(sliced);
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, p);
            emitStride();
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, p);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, one);
            il.Emit(OpCodes.Ldloc, p);
            EmitLoadIndirect(il, t);
            WhereNode.EmitPushZeroPublic(il, t);
            EmitComparisonOperation(il, ComparisonOp.NotEqual, t);
            il.Emit(OpCodes.Brtrue, found);
            il.Emit(OpCodes.Br, loop);

            il.MarkLabel(found);
            il.Emit(OpCodes.Ldc_I8, -1L);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Sub);            // -1 - i = -(i + 1)
            il.Emit(OpCodes.Ret);

            il.MarkLabel(one);
            il.Emit(OpCodes.Ldc_I8, -1L);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>Emits <see cref="PolyTrimLenKernel"/> for <paramref name="t"/>.</summary>
        /// <param name="t">Element dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyTrimLenKernel EmitPolyTrimLen(NPTypeCode t)
        {
            var dm = NewPolySeriesMethod($"PolyTrimLen_{t}", typeof(long), typeof(byte*), typeof(long), typeof(long));
            var il = dm.GetILGenerator();
            var p = il.DeclareLocal(typeof(byte*));
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Stloc, p);
            EmitPolyTrimScanAndReturn(il, p, () => il.Emit(OpCodes.Ldarg_1), () => il.Emit(OpCodes.Ldarg_2), t);
            return dm.CreateDelegate<PolyTrimLenKernel>();
        }

        /// <summary>Emits <see cref="PolyCombineKernel"/>.</summary>
        /// <param name="op">The in-place update.</param>
        /// <param name="ta">Target dtype.</param>
        /// <param name="tb">Other operand's dtype.</param>
        /// <param name="tr">Result dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyCombineKernel EmitPolyCombine(PolyCombineOp op, NPTypeCode ta, NPTypeCode tb, NPTypeCode tr)
        {
            // args: 0 a, 1 aStride, 2 na, 3 b, 4 bStride, 5 nb, 6 r
            var dm = NewPolySeriesMethod($"PolyCombine_{op}_{ta}_{tb}_{tr}", typeof(long),
                typeof(byte*), typeof(long), typeof(long), typeof(byte*), typeof(long), typeof(long), typeof(byte*));
            var il = dm.GetILGenerator();
            int size = GetTypeSize(tr);
            var pa = il.DeclareLocal(typeof(byte*));
            var pb = il.DeclareLocal(typeof(byte*));
            var pr = il.DeclareLocal(typeof(byte*));
            var i = il.DeclareLocal(typeof(long));

            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, pa);
            il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Stloc, pb);
            il.Emit(OpCodes.Ldarg, 6); il.Emit(OpCodes.Stloc, pr);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);

            // Pushes the target element converted to the result dtype (as_series' copy), negated for NegateAdd
            // (NumPy's `c2 = -c2` runs on the converted copy, before the add).
            void PushA()
            {
                il.Emit(OpCodes.Ldloc, pa);
                EmitLoadIndirect(il, ta);
                EmitConvertTo(il, ta, tr);
                if (op == PolyCombineOp.NegateAdd)
                    EmitUnaryScalarOperation(il, UnaryOp.Negate, tr);
            }

            // Overlap: r[i] = A[i] OP B[i] for i < nb — NumPy's operand order, A first.
            var overlap = il.DefineLabel();
            var tail = il.DefineLabel();
            il.MarkLabel(overlap);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Bge, tail);
            il.Emit(OpCodes.Ldloc, pr);
            PushA();
            il.Emit(OpCodes.Ldloc, pb);
            EmitLoadIndirect(il, tb);
            EmitConvertTo(il, tb, tr);
            EmitScalarOperation(il, op == PolyCombineOp.Subtract ? BinaryOp.Subtract : BinaryOp.Add, tr);
            EmitStoreIndirect(il, tr);
            EmitPolyAdvance(il, pa, () => il.Emit(OpCodes.Ldarg_1));
            EmitPolyAdvance(il, pb, () => il.Emit(OpCodes.Ldarg, 4));
            EmitPolyAdvanceConst(il, pr, size);
            EmitPolyIncrement(il, i);
            il.Emit(OpCodes.Br, overlap);

            // Tail: r[i] = A[i] (or -A[i]) for nb <= i < na — the part of the target the update never touched.
            var trim = il.DefineLabel();
            il.MarkLabel(tail);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, trim);
            il.Emit(OpCodes.Ldloc, pr);
            PushA();
            EmitStoreIndirect(il, tr);
            EmitPolyAdvance(il, pa, () => il.Emit(OpCodes.Ldarg_1));
            EmitPolyAdvanceConst(il, pr, size);
            EmitPolyIncrement(il, i);
            il.Emit(OpCodes.Br, tail);

            // trimseq(ret) over the contiguous result.
            il.MarkLabel(trim);
            var pt = il.DeclareLocal(typeof(byte*));
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Stloc, pt);
            EmitPolyTrimScanAndReturn(il, pt, () => il.Emit(OpCodes.Ldarg_2), () => il.Emit(OpCodes.Ldc_I8, (long)size), tr);
            return dm.CreateDelegate<PolyCombineKernel>();
        }

        /// <summary>Emits <see cref="PolyLastAboveKernel"/>.</summary>
        /// <param name="ts">Source dtype.</param>
        /// <param name="tc">Coefficient dtype (as_series' conversion target).</param>
        /// <param name="tk">Comparison dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyLastAboveKernel EmitPolyLastAbove(NPTypeCode ts, NPTypeCode tc, NPTypeCode tk)
        {
            // args: 0 p, 1 n, 2 stride, 3 tol
            var dm = NewPolySeriesMethod($"PolyLastAbove_{ts}_{tc}_{tk}", typeof(long),
                typeof(byte*), typeof(long), typeof(long), typeof(byte*));
            var il = dm.GetILGenerator();
            // np.abs of a complex coefficient is its float64 magnitude; of every other dtype, the same dtype.
            NPTypeCode ta = tc == NPTypeCode.Complex ? NPTypeCode.Double : tc;
            var q = il.DeclareLocal(typeof(byte*));
            var i = il.DeclareLocal(typeof(long));
            var tol = il.DeclareLocal(GetClrType(tk));

            // tol is loaded once; the scan runs from the END because only NumPy's ind[-1] is ever read.
            il.Emit(OpCodes.Ldarg_3);
            EmitLoadIndirect(il, tk);
            il.Emit(OpCodes.Stloc, tol);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, q);

            var loop = il.DefineLabel();
            var none = il.DefineLabel();
            var found = il.DefineLabel();
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, none);
            // |convert_tc(p[i])| converted to the comparison dtype, then `> tol` with NumPy's loop semantics.
            il.Emit(OpCodes.Ldloc, q);
            EmitLoadIndirect(il, ts);
            EmitConvertTo(il, ts, tc);
            if (tc == NPTypeCode.Complex)
                // np.abs of complex128 is a float64 array: the magnitude itself (NumPy's npy_cabs). The house
                // unary Abs re-wraps it as a Complex for the complex ufunc loop, which is not what is compared here.
                il.EmitCall(OpCodes.Call, CachedMethods.ComplexAbs, null);
            else
                EmitUnaryScalarOperation(il, UnaryOp.Abs, tc);
            EmitConvertTo(il, ta, tk);
            il.Emit(OpCodes.Ldloc, tol);
            if (tk == NPTypeCode.Complex)
                il.EmitCall(OpCodes.Call, s_polyComplexGreater, null);
            else
                EmitComparisonOperation(il, ComparisonOp.Greater, tk);
            il.Emit(OpCodes.Brtrue, found);
            il.Emit(OpCodes.Ldloc, q);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, q);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, i);
            il.Emit(OpCodes.Br, loop);

            il.MarkLabel(found);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ret);
            il.MarkLabel(none);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyLastAboveKernel>();
        }

        /// <summary>Emits <see cref="PolyCastKernel"/>.</summary>
        /// <param name="ts">Source dtype.</param>
        /// <param name="tr">Destination dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyCastKernel EmitPolyCast(NPTypeCode ts, NPTypeCode tr)
        {
            // args: 0 p, 1 n, 2 stride, 3 r
            var dm = NewPolySeriesMethod($"PolyCast_{ts}_{tr}", typeof(void), typeof(byte*), typeof(long), typeof(long), typeof(byte*));
            var il = dm.GetILGenerator();
            int size = GetTypeSize(tr);
            var pp = il.DeclareLocal(typeof(byte*));
            var pr = il.DeclareLocal(typeof(byte*));
            var i = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldarg_0); il.Emit(OpCodes.Stloc, pp);
            il.Emit(OpCodes.Ldarg_3); il.Emit(OpCodes.Stloc, pr);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, i);
            var loop = il.DefineLabel();
            var done = il.DefineLabel();
            il.MarkLabel(loop);
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, done);
            il.Emit(OpCodes.Ldloc, pr);
            il.Emit(OpCodes.Ldloc, pp);
            EmitLoadIndirect(il, ts);
            EmitConvertTo(il, ts, tr);
            EmitStoreIndirect(il, tr);
            EmitPolyAdvance(il, pp, () => il.Emit(OpCodes.Ldarg_2));
            EmitPolyAdvanceConst(il, pr, size);
            EmitPolyIncrement(il, i);
            il.Emit(OpCodes.Br, loop);
            il.MarkLabel(done);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyCastKernel>();
        }

        /// <summary>Emits <see cref="PolyScalarBinaryKernel"/>.</summary>
        /// <param name="op">The op.</param>
        /// <param name="t">Loop dtype.</param>
        /// <param name="product">Complex product form.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="NotSupportedException">True division in an integer dtype (the caller must pick float64 first).</exception>
        private static PolyScalarBinaryKernel EmitPolyScalarBinary(BinaryOp op, NPTypeCode t, PolyComplexProduct product)
        {
            if (op == BinaryOp.Divide && (t == NPTypeCode.Boolean || t.IsInteger() || t == NPTypeCode.Char))
                throw new NotSupportedException($"true division has no {t} loop (NumPy divides integers in float64)");
            var dm = NewPolySeriesMethod($"PolyScalar_{op}_{t}_{product}", typeof(void), typeof(byte*), typeof(byte*), typeof(byte*));
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, t);
            il.Emit(OpCodes.Ldarg_1);
            EmitLoadIndirect(il, t);
            // NumPy's bool add/multiply loops are logical_or/logical_and (canonical 0/1 results).
            BinaryOp emitted = t == NPTypeCode.Boolean && op == BinaryOp.Add ? BinaryOp.BitwiseOr
                             : t == NPTypeCode.Boolean && op == BinaryOp.Multiply ? BinaryOp.BitwiseAnd
                             : op;
            if (t == NPTypeCode.Complex && op == BinaryOp.Multiply && product != PolyComplexProduct.Simd)
                il.EmitCall(OpCodes.Call, product == PolyComplexProduct.Naive
                    ? ILKernelGenerator.s_polyNaiveComplexMultiply
                    : ILKernelGenerator.s_polyLoopScalarComplexMultiply, null);
            else
                EmitScalarOperation(il, emitted, t);
            EmitStoreIndirect(il, t);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyScalarBinaryKernel>();
        }

        /// <summary>
        ///     Emits the complex64 <see cref="PolyScalarBinaryKernel"/>: load both <see cref="Complex"/> carriers, call
        ///     the float32 helper of <paramref name="op"/> / <paramref name="product"/>, store the carrier.
        /// </summary>
        /// <param name="op">Add, Subtract, Multiply or Divide.</param>
        /// <param name="product">The complex product form (Multiply only).</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="MissingMethodException">A helper was renamed (fails fast at first emission).</exception>
        private static PolyScalarBinaryKernel EmitPolyScalarComplex64(BinaryOp op, PolyComplexProduct product)
        {
            string helper = op switch
            {
                BinaryOp.Add => nameof(PolyComplex64Add),
                BinaryOp.Subtract => nameof(PolyComplex64Subtract),
                BinaryOp.Divide => nameof(PolyComplex64Divide),
                _ => product switch
                {
                    PolyComplexProduct.Naive => nameof(PolyComplex64MultiplyNaive),
                    PolyComplexProduct.LoopScalar => nameof(PolyComplex64MultiplyLoopScalar),
                    _ => nameof(PolyComplex64MultiplySimd),
                },
            };
            var method = typeof(DirectILKernelGenerator).GetMethod(helper, BindingFlags.NonPublic | BindingFlags.Static)
                         ?? throw new MissingMethodException(helper);
            var dm = NewPolySeriesMethod($"PolyScalar_{op}_Complex64_{product}", typeof(void), typeof(byte*), typeof(byte*), typeof(byte*));
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, NPTypeCode.Complex);
            il.Emit(OpCodes.Ldarg_1);
            EmitLoadIndirect(il, NPTypeCode.Complex);
            il.EmitCall(OpCodes.Call, method, null);
            EmitStoreIndirect(il, NPTypeCode.Complex);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyScalarBinaryKernel>();
        }

        /// <summary>Emits <see cref="PolyScalarUnaryKernel"/> (negation).</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyScalarUnaryKernel EmitPolyScalarNegate(NPTypeCode t)
        {
            var dm = NewPolySeriesMethod($"PolyScalar_Negate_{t}", typeof(void), typeof(byte*), typeof(byte*));
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, t);
            EmitUnaryScalarOperation(il, UnaryOp.Negate, t);
            // Sub-int32 integers negate on the int32 stack; the narrowing store keeps NumPy's wrap-around.
            EmitStoreIndirect(il, t);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyScalarUnaryKernel>();
        }

        /// <summary>Emits <see cref="PolyScalarPredicateKernel"/>.</summary>
        /// <param name="t">The dtype.</param>
        /// <param name="test">The comparison.</param>
        /// <returns>The kernel.</returns>
        private static PolyScalarPredicateKernel EmitPolyScalarPredicate(NPTypeCode t, PolyZeroTest test)
        {
            var dm = NewPolySeriesMethod($"PolyScalar_{test}Zero_{t}", typeof(bool), typeof(byte*));
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, t);
            if (t == NPTypeCode.Complex && test == PolyZeroTest.Less)
            {
                // scalarmath's complex ordering: the house comparison emitter has no complex "<".
                il.EmitCall(OpCodes.Call, s_polyComplexScalarLessZero, null);
            }
            else
            {
                WhereNode.EmitPushZeroPublic(il, t);
                EmitComparisonOperation(il, test == PolyZeroTest.NotEqual ? ComparisonOp.NotEqual : ComparisonOp.Less, t);
            }
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyScalarPredicateKernel>();
        }

        /// <summary>Emits <see cref="PolyScalarBoxKernel"/>.</summary>
        /// <param name="t">The dtype.</param>
        /// <returns>The kernel.</returns>
        private static PolyScalarBoxKernel EmitPolyScalarBox(NPTypeCode t)
        {
            var dm = NewPolySeriesMethod($"PolyScalar_Box_{t}", typeof(object), typeof(byte*));
            var il = dm.GetILGenerator();
            il.Emit(OpCodes.Ldarg_0);
            EmitLoadIndirect(il, t);
            // A sub-int32 element sits on the stack as an int32; `box` of the narrow CLR type re-narrows it.
            il.Emit(OpCodes.Box, GetClrType(t));
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyScalarBoxKernel>();
        }

        /// <summary>
        ///     Emits <see cref="PolyMapParmsKernel"/> for dtype <paramref name="t"/>: every intermediate lives in a
        ///     local of <paramref name="t"/>'s CLR type, so a sub-int32 value wraps on the store exactly as the
        ///     per-operation kernel's narrowing store wraps it, a float16 is rounded after every operation, and the
        ///     operations run in Python's statement order (a complex product is scalarmath's naive one). The two
        ///     divisions are NumPy's true division: an integer dtype's operands are converted to float64 first
        ///     (<see cref="EmitConvertTo"/> — NumPy's single-rounding int→float rules), the operation that the
        ///     per-operation path's loop-dtype conversion performs.
        /// </summary>
        /// <param name="t">The four scalars' dtype.</param>
        /// <returns>The kernel.</returns>
        /// <exception cref="NotSupportedException"><paramref name="t"/> is bool (NumPy's subtraction refuses it).</exception>
        private static PolyMapParmsKernel EmitPolyMapParms(NPTypeCode t)
        {
            if (t == NPTypeCode.Boolean)
                throw new NotSupportedException("numpy boolean subtract is not supported; the per-operation path raises NumPy's TypeError");
            NPTypeCode rt = PolyTyping.IsIntLike(t) ? NPTypeCode.Double : t;   // PolyNumber.LoopDtype's division rule
            var clr = GetClrType(t);
            var dm = NewPolySeriesMethod($"PolyMapParms_{t}", typeof(void),
                                         typeof(byte*), typeof(byte*), typeof(byte*), typeof(byte*), typeof(byte*), typeof(byte*));
            var il = dm.GetILGenerator();
            LocalBuilder o1 = il.DeclareLocal(clr), o0 = il.DeclareLocal(clr), n1 = il.DeclareLocal(clr), n0 = il.DeclareLocal(clr);
            LocalBuilder oldlen = il.DeclareLocal(clr), newlen = il.DeclareLocal(clr), p = il.DeclareLocal(clr), q = il.DeclareLocal(clr),
                         num = il.DeclareLocal(clr);

            void Load(short arg, LocalBuilder dst)
            {
                il.Emit(OpCodes.Ldarg, arg);
                EmitLoadIndirect(il, t);
                il.Emit(OpCodes.Stloc, dst);
            }

            void Op(LocalBuilder a, BinaryOp op, LocalBuilder b, LocalBuilder dst)
            {
                il.Emit(OpCodes.Ldloc, a);
                il.Emit(OpCodes.Ldloc, b);
                if (t == NPTypeCode.Complex && op == BinaryOp.Multiply)
                    il.EmitCall(OpCodes.Call, ILKernelGenerator.s_polyNaiveComplexMultiply, null);   // scalarmath's product
                else
                    EmitScalarOperation(il, op, t);
                il.Emit(OpCodes.Stloc, dst);   // a sub-int32 dtype wraps here (stloc truncates to the local's width)
            }

            void Div(LocalBuilder a, LocalBuilder b, short destArg)
            {
                il.Emit(OpCodes.Ldarg, destArg);
                il.Emit(OpCodes.Ldloc, a);
                EmitConvertTo(il, t, rt);
                il.Emit(OpCodes.Ldloc, b);
                EmitConvertTo(il, t, rt);
                EmitScalarOperation(il, BinaryOp.Divide, rt);
                EmitStoreIndirect(il, rt);
            }

            Load(0, o1);
            Load(1, o0);
            Load(2, n1);
            Load(3, n0);
            Op(o1, BinaryOp.Subtract, o0, oldlen);   // oldlen = old[1] - old[0]
            Op(n1, BinaryOp.Subtract, n0, newlen);   // newlen = new[1] - new[0]
            Op(o1, BinaryOp.Multiply, n0, p);        // old[1]*new[0]
            Op(o0, BinaryOp.Multiply, n1, q);        // old[0]*new[1]
            Op(p, BinaryOp.Subtract, q, num);
            Div(num, oldlen, 4);                     // off = (...) / oldlen
            Div(newlen, oldlen, 5);                  // scl = newlen / oldlen
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<PolyMapParmsKernel>();
        }

        /// <summary>Emits <c>p += stride</c> for a byte-pointer local and a long stride.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="p">The pointer local.</param>
        /// <param name="emitStride">Pushes the stride.</param>
        private static void EmitPolyAdvance(ILGenerator il, LocalBuilder p, Action emitStride)
        {
            il.Emit(OpCodes.Ldloc, p);
            emitStride();
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, p);
        }

        /// <summary>Emits <c>p += size</c> for a byte-pointer local and a constant element size.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="p">The pointer local.</param>
        /// <param name="size">The element size.</param>
        private static void EmitPolyAdvanceConst(ILGenerator il, LocalBuilder p, int size)
        {
            il.Emit(OpCodes.Ldloc, p);
            il.Emit(OpCodes.Ldc_I4, size);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, p);
        }

        /// <summary>Emits <c>i++</c> for a long local.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="i">The counter local.</param>
        private static void EmitPolyIncrement(ILGenerator il, LocalBuilder i)
        {
            il.Emit(OpCodes.Ldloc, i);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, i);
        }
    }
}
