using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
//   PolyScalarBinaryKernel (a, b, r)                       -> *r = *a OP *b   (both already in the loop dtype)
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
        private enum PolySeriesFamily : byte { TrimLen, Combine, LastAbove, Cast, ScalarBinary, ScalarNegate, ScalarPredicate, ScalarBox, MapParms }

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
