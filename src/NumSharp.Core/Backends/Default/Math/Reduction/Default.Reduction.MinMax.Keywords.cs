using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// Default.Reduction.MinMax.Keywords.cs — np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin with NumPy's
// out=, initial= and where= (plan docs/plans/api-parameter-parity.md, item A4)
// =============================================================================
//
// np.max(a, axis, out, keepdims, initial, where) IS np.maximum.reduce(a, axis, None, out, keepdims=, initial=, where=)
// (numpy/_core/fromnumeric.py::_wrapreduction; ndarray.max is _methods._amax, the same call), np.min np.minimum.reduce,
// and np.nanmax / np.nanmin of an ndarray np.fmax.reduce / np.fmin.reduce (numpy/lib/_nanfunctions_impl.py). So the
// keywords are ufunc.reduce's, in NumPy's order of statements:
//
//   1. the reduce prologue (ReductionArguments: a read-only out, a non-bool mask, the axis, out's rank, the iterator's
//      broadcast — NumPy's texts and order);
//   2. the LOOP dtype: the input's, or with out= the promotion of (out, input) — out's dtype acts as the reduction's first
//      input (reducelike_promote_and_resolve): an int16 input with an int8 out runs the int16 loop, a uint64 input with
//      an int64 out the float64 loop;
//   3. initial= packed into the loop dtype (PyArray_Pack — ReductionInitial: a Python int is range-checked, a float
//      truncated into an integer loop, a 0-d array cast);
//   4. no initial: where= raises "reduction operation 'maximum' does not have an identity, so to use a where mask one has
//      to specify 'initial'", and an empty reduced axis "zero-size array to reduction operation maximum which has no
//      identity";
//   5. the result is the out (any dtype: the loop's result is cast into it, unsafely) or a fresh array in the loop dtype,
//      laid out in the iteration order. It starts as the initial value (cast into out's dtype), or as each output's first
//      element (cast from the input's dtype into out's — PyArray_CopyInitialReduceValues), and the reduction folds the
//      rest in through NumPy's iterator (NumPyReduceIterator: fills, cores, masked runs, the out's round trip through its
//      own dtype at every fill).
//
// Two engines serve step 5, picked by what NumPy's answer can depend on:
//
//   * the FAST path — the existing exact engines, seeded with the initial value: NumPyMinMaxReduce (float32 / float64 under
//     maximum / minimum, and every integer dtype under any of the four ops) and NumPyHalfMinMaxReduce (float16, the seed
//     folded in after — its rule is an associative selection). It serves calls without where= whose out (if any) has the
//     input's dtype, on a non-broadcast input: there the per-output result does not depend on the out's layout or on
//     buffer fills. The exact engines keep their slab fusion and SIMD rows, so these calls stay as fast as plain np.max;
//   * the GENERAL path — NumPyReduceIterator, NumPy's own schedule: every where= call, every cast out or input, every
//     np.nanmax / np.nanmin of float32 / float64 with a keyword (fmax's vector and scalar ops disagree on a ±0 tie, so the
//     iterator's call cuts decide the bits), every broadcast input, complex / char / decimal (their loops are sequential
//     BINARY_LOOPs, exact in NumPy's visiting order), and bool (NumPy's maximum / minimum of bools ARE logical_or /
//     logical_and: an early-exit reduce and a SIMD elementwise op).
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    ///     <c>HALF_maximum</c>: keep the running value when it is NaN or <c>&gt;=</c> the next element (a ±0 tie keeps the
    ///     earlier zero), else take the next element — so the first NaN of the sequence wins, verbatim.
    /// </summary>
    internal readonly struct HalfMaxRule : ISequentialNanRule<Half>
    {
        /// <inheritdoc />
        /// <remarks>Compares through <c>float</c> — every <see cref="Half"/> converts exactly, and IEEE order is preserved.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half Fold(Half acc, Half x)
            => Half.IsNaN(acc) || (float)acc >= (float)x ? acc : x;
    }

    /// <summary><c>HALF_minimum</c>: the mirror of <see cref="HalfMaxRule"/> with <c>&lt;=</c>.</summary>
    internal readonly struct HalfMinRule : ISequentialNanRule<Half>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half Fold(Half acc, Half x)
            => Half.IsNaN(acc) || (float)acc <= (float)x ? acc : x;
    }

    /// <summary>
    ///     <c>CDOUBLE_maximum</c>: keep the running value when it has a NaN in either part, or when it is lexicographically
    ///     <c>&gt;=</c> the next element (<c>CGE</c>); otherwise take the next element. A NaN element therefore always
    ///     replaces a non-NaN running value (every <c>CGE</c> against it is false), and a tie keeps the earlier element.
    /// </summary>
    internal readonly struct ComplexMaxRule : ISequentialNanRule<Complex>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Fold(Complex acc, Complex x)
        {
            double ar = acc.Real, ai = acc.Imaginary, xr = x.Real, xi = x.Imaginary;
            if (double.IsNaN(ar) || double.IsNaN(ai))
                return acc;
            bool ge = (ar > xr && !double.IsNaN(ai) && !double.IsNaN(xi)) || (ar == xr && ai >= xi);
            return ge ? acc : x;
        }
    }

    /// <summary><c>CDOUBLE_minimum</c>: the mirror of <see cref="ComplexMaxRule"/> with <c>CLE</c>.</summary>
    internal readonly struct ComplexMinRule : ISequentialNanRule<Complex>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Fold(Complex acc, Complex x)
        {
            double ar = acc.Real, ai = acc.Imaginary, xr = x.Real, xi = x.Imaginary;
            if (double.IsNaN(ar) || double.IsNaN(ai))
                return acc;
            bool le = (ar < xr && !double.IsNaN(ai) && !double.IsNaN(xi)) || (ar == xr && ai <= xi);
            return le ? acc : x;
        }
    }

    /// <summary>The maximum of NumSharp's char (NumPy's uint16 analog): an unsigned compare, a tie keeping the running value.</summary>
    internal readonly struct CharMaxRule : ISequentialNanRule<char>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Fold(char acc, char x) => x > acc ? x : acc;
    }

    /// <summary>The minimum of NumSharp's char: the mirror of <see cref="CharMaxRule"/>.</summary>
    internal readonly struct CharMinRule : ISequentialNanRule<char>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char Fold(char acc, char x) => x < acc ? x : acc;
    }

    /// <summary>
    ///     The maximum of NumSharp's decimal (no NumPy analog): a tie keeps the running value — two equal decimals may differ
    ///     in scale (<c>1.0m</c> and <c>1.00m</c>), and the earlier one survives, as a sequential NumPy loop would keep it.
    /// </summary>
    internal readonly struct DecimalMaxRule : ISequentialNanRule<decimal>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Fold(decimal acc, decimal x) => x > acc ? x : acc;
    }

    /// <summary>The minimum of NumSharp's decimal: the mirror of <see cref="DecimalMaxRule"/>.</summary>
    internal readonly struct DecimalMinRule : ISequentialNanRule<decimal>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal Fold(decimal acc, decimal x) => x < acc ? x : acc;
    }

    /// <summary>
    ///     The two call forms of NumPy's min / max loops for <see cref="NumPyReduceIterator.Loop{T}"/>: the SIMD schedule of
    ///     <c>loops_minmax.dispatch.c.src</c> (float32 / float64 / integers, through <see cref="NumPyMinMaxReduce"/>'s lane
    ///     rules), the sequential <c>BINARY_LOOP</c>s (float16, complex, char, decimal, through a fold rule) and the bool
    ///     <c>logical_or</c> / <c>logical_and</c> loops.
    /// </summary>
    internal static unsafe class NumPyMinMaxLoops
    {
        /// <summary>
        ///     The IS_BINARY_REDUCE call of the SIMD loop: <c>simd_reduce_c</c> seeded with the out's value when the input is
        ///     contiguous, the 8-accumulator scalar unroll otherwise (<see cref="NumPyMinMaxReduce.ChainStrided{T,TLane}"/>).
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride (1 = contiguous).</param>
        /// <param name="n">The element count (&gt;= 1).</param>
        /// <returns>The out's value after the call.</returns>
        internal static T LaneReduce<T, TLane>(T acc, T* x, long xs, long n)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
            => xs == 1
                ? NumPyMinMaxReduce.ChainContiguous<T, TLane>(acc, x, n)
                : NumPyMinMaxReduce.ChainStrided<T, TLane>(acc, x, xs, n);

        /// <summary>
        ///     The elementwise call of the SIMD loop: <c>o[i] = op(o[i], x[i])</c>, with NumPy's split between the vector
        ///     section (the first <c>n - n % lanes</c> elements) and the scalar tail.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The split only shows for the P rules (fmax / fmin of float32 / float64), whose vector op keeps the second operand
        ///     on a ±0 tie while the CRT scalar op returns +0 (fmin -0). The SIMD branch is taken for contiguous operands
        ///     (<c>simd_binary_ccc</c>) and for float strides <c>npyv_loadable_stride</c> accepts; for float32 a NEGATIVE
        ///     stride fails it (the macro divides the signed stride by an unsigned size), so such a call runs scalar
        ///     throughout. Integer loops have no strided SIMD branch, and none of this changes an integer result.
        ///     </para>
        ///     <para>
        ///     The N rules' vector and scalar ops are the same rule, so their calls go through one step whatever the strides.
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        internal static void LaneBinary<T, TLane>(T* o, long os, T* x, long xs, long n)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            bool isFloat = typeof(T) == typeof(double) || typeof(T) == typeof(float);
            if (TLane.PropagatesNaN || !isFloat)
            {
                NumPyMinMaxReduce.CombineRunStep<T, NumPyMinMaxReduce.VectorOpStep<T, TLane>>(o, os, x, xs, n);
                return;
            }

            bool simd = (os == 1 && xs == 1) || !(typeof(T) == typeof(float) && (os < 0 || xs < 0));
            if (!simd)
            {
                NumPyMinMaxReduce.CombineRunStep<T, NumPyMinMaxReduce.ScalarOpStep<T, TLane>>(o, os, x, xs, n);
                return;
            }

            long v = n - n % Vector256<T>.Count;
            NumPyMinMaxReduce.CombineRunStep<T, NumPyMinMaxReduce.VectorOpStep<T, TLane>>(o, os, x, xs, v);
            NumPyMinMaxReduce.CombineRunStep<T, NumPyMinMaxReduce.ScalarOpStep<T, TLane>>(o + v * os, os, x + v * xs, xs, n - v);
        }

        /// <summary>The IS_BINARY_REDUCE call of a sequential BINARY_LOOP: the rule folded over the run in order.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold rule.</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static T SeqReduce<T, TRule>(T acc, T* x, long xs, long n)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            for (long i = 0; i < n; i++, x += xs)
                acc = TRule.Fold(acc, *x);
            return acc;
        }

        /// <summary>The elementwise call of a sequential BINARY_LOOP: <c>o[i] = rule(o[i], x[i])</c>.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold rule.</typeparam>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        internal static void SeqBinary<T, TRule>(T* o, long os, T* x, long xs, long n)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            for (long i = 0; i < n; i++, o += os, x += xs)
                *o = TRule.Fold(*o, *x);
        }

        /// <summary>
        ///     The IS_BINARY_REDUCE call of <c>BOOL_logical_or</c> (<paramref name="isOr"/>, NumPy's bool maximum / fmax) or
        ///     <c>BOOL_logical_and</c> (bool minimum / fmin): the out's value decides first — an OR already true (an AND
        ///     already false) reads nothing more — and a contiguous run is searched for the first deciding byte (NumPy's
        ///     <c>memcmp</c> / <c>memchr</c>), so the call stops there.
        /// </summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        /// <param name="isOr">True for the OR loop, false for the AND loop.</param>
        /// <returns>The out's value after the call (0 / 1, like NumPy's loop).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool BoolReduce(bool acc, bool* x, long xs, long n, bool isOr)
        {
            // A bool is any nonzero byte (NumPy reads the byte); the result is normalized to 0 / 1.
            bool decided = *(byte*)&acc != 0;
            if (decided == isOr)
                return decided;
            byte* p = (byte*)x;
            if (xs == 1)
            {
                for (long i = 0; i < n;)
                {
                    int len = (int)Math.Min(n - i, int.MaxValue);
                    var span = new ReadOnlySpan<byte>(p + i, len);
                    if (isOr ? span.IndexOfAnyExcept((byte)0) >= 0 : span.IndexOf((byte)0) >= 0)
                        return isOr;
                    i += len;
                }

                return !isOr;
            }

            for (long i = 0; i < n; i++)
                if ((p[i * xs] != 0) == isOr)
                    return isOr;
            return !isOr;
        }

        /// <summary>
        ///     The elementwise call of <c>BOOL_logical_or</c> / <c>BOOL_logical_and</c>: <c>o[i] = o[i] || x[i]</c> (or
        ///     <c>&amp;&amp;</c>), each result 0 or 1 — a contiguous run 64 bytes per step, NumPy's own two-op form
        ///     (<c>simd_logical_or_u8</c>: <c>min(a | b, 1)</c>; AND as <c>min(min(a, b), 1)</c>, nonzero exactly when both
        ///     are).
        /// </summary>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        /// <param name="isOr">True for OR, false for AND.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BoolBinary(bool* o, long os, bool* x, long xs, long n, bool isOr)
        {
            byte* po = (byte*)o, px = (byte*)x;
            long i = 0;
            if (os == 1 && xs == 1 && Vector256.IsHardwareAccelerated)
            {
                var one = Vector256.Create((byte)1);
                for (; i + 64 <= n; i += 64)
                {
                    var a0 = Vector256.Load(po + i);
                    var a1 = Vector256.Load(po + i + 32);
                    var b0 = Vector256.Load(px + i);
                    var b1 = Vector256.Load(px + i + 32);
                    Vector256.Store(Vector256.Min(isOr ? a0 | b0 : Vector256.Min(a0, b0), one), po + i);
                    Vector256.Store(Vector256.Min(isOr ? a1 | b1 : Vector256.Min(a1, b1), one), po + i + 32);
                }

                for (; i + 32 <= n; i += 32)
                {
                    var a = Vector256.Load(po + i);
                    var b = Vector256.Load(px + i);
                    Vector256.Store(Vector256.Min(isOr ? a | b : Vector256.Min(a, b), one), po + i);
                }
            }

            for (; i < n; i++)
            {
                byte a = po[i * os], b = px[i * xs];
                po[i * os] = (byte)((isOr ? (a != 0 || b != 0) : (a != 0 && b != 0)) ? 1 : 0);
            }
        }

        /// <summary>
        ///     The several-rows elementwise call of the bool loops (<see cref="NumPyReduceIterator.Loop{T}.BinaryRows"/>):
        ///     <paramref name="rows"/> input rows folded into one out row — per 32-byte column, the out loaded once, the rows
        ///     OR-ed (AND: the byte minimum) into it in order, then the 0 / 1 normalization once (an OR of bytes is nonzero
        ///     exactly when one of them is, a minimum exactly when all are, so normalizing once equals normalizing per row).
        ///     A row stride of 0 (a broadcast axis) is one row: the op is idempotent.
        /// </summary>
        /// <param name="o">The out row's first element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The first row's first element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="rs">The input's row stride (elements).</param>
        /// <param name="rows">The row count (&gt;= 1).</param>
        /// <param name="n">The elements per row.</param>
        /// <param name="isOr">True for OR, false for AND.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BoolBinaryRows(bool* o, long os, bool* x, long xs, long rs, long rows, long n, bool isOr)
        {
            if (rs == 0)
                rows = 1;
            if (os != 1 || xs != 1 || !Vector256.IsHardwareAccelerated)
            {
                for (long r = 0; r < rows; r++)
                    BoolBinary(o, os, x + r * rs, xs, n, isOr);
                return;
            }

            byte* po = (byte*)o, px = (byte*)x;
            var one = Vector256.Create((byte)1);
            long i = 0;
            for (; i + 128 <= n; i += 128)
            {
                // Four independent columns per pass, so the per-row ops of one column overlap the others'.
                var a0 = Vector256.Load(po + i);
                var a1 = Vector256.Load(po + i + 32);
                var a2 = Vector256.Load(po + i + 64);
                var a3 = Vector256.Load(po + i + 96);
                byte* p = px + i;
                for (long r = 0; r < rows; r++, p += rs)
                {
                    var b0 = Vector256.Load(p);
                    var b1 = Vector256.Load(p + 32);
                    var b2 = Vector256.Load(p + 64);
                    var b3 = Vector256.Load(p + 96);
                    a0 = isOr ? a0 | b0 : Vector256.Min(a0, b0);
                    a1 = isOr ? a1 | b1 : Vector256.Min(a1, b1);
                    a2 = isOr ? a2 | b2 : Vector256.Min(a2, b2);
                    a3 = isOr ? a3 | b3 : Vector256.Min(a3, b3);
                }

                Vector256.Store(Vector256.Min(a0, one), po + i);
                Vector256.Store(Vector256.Min(a1, one), po + i + 32);
                Vector256.Store(Vector256.Min(a2, one), po + i + 64);
                Vector256.Store(Vector256.Min(a3, one), po + i + 96);
            }

            for (; i + 32 <= n; i += 32)
            {
                var a = Vector256.Load(po + i);
                byte* p = px + i;
                for (long r = 0; r < rows; r++, p += rs)
                    a = isOr ? a | Vector256.Load(p) : Vector256.Min(a, Vector256.Load(p));
                Vector256.Store(Vector256.Min(a, one), po + i);
            }

            for (; i < n; i++)
            {
                bool acc = po[i] != 0;
                byte* p = px + i;
                for (long r = 0; r < rows; r++, p += rs)
                    acc = isOr ? acc || *p != 0 : acc && *p != 0;
                po[i] = acc ? (byte)1 : (byte)0;
            }
        }

        /// <summary>The OR several-rows form — <see cref="BoolBinaryRows"/>.</summary>
        /// <param name="o">The out row's first element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The first row's first element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="rs">The input's row stride.</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        internal static void BoolOrBinaryRows(bool* o, long os, bool* x, long xs, long rs, long rows, long n)
            => BoolBinaryRows(o, os, x, xs, rs, rows, n, isOr: true);

        /// <summary>The AND several-rows form — <see cref="BoolBinaryRows"/>.</summary>
        /// <param name="o">The out row's first element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The first row's first element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="rs">The input's row stride.</param>
        /// <param name="rows">The row count.</param>
        /// <param name="n">The elements per row.</param>
        internal static void BoolAndBinaryRows(bool* o, long os, bool* x, long xs, long rs, long rows, long n)
            => BoolBinaryRows(o, os, x, xs, rs, rows, n, isOr: false);

        /// <summary>
        ///     The several-rows elementwise call of the SIMD loops (<see cref="NumPyReduceIterator.Loop{T}.BinaryRows"/>):
        ///     each element folds the rows in order with exactly the op its per-row <see cref="LaneBinary{T,TLane}"/> calls
        ///     give it — the vector op in the first <c>n - n % lanes</c> positions, and in the rest the scalar op for the P
        ///     rules (np.nanmax / np.nanmin) and the vector rule on one element otherwise (the same rule for the N rules and
        ///     the integers) — with the running values held in registers, four columns per pass.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Rows are taken in PAIRS: <c>op(op(a, x0), x1) == op(a, op(x0, x1))</c> for each of these ops (the N rules,
        ///     <c>npyv_maxp</c> / <c>npyv_minp</c>, the CRT <c>fmax</c> / <c>fmin</c> and the integer max / min are all
        ///     associative, bits included: each selects one of its operands by a fixed rule — the first NaN, the last of the
        ///     tied maxima, or the like), so the pair's op runs off the running value's dependency chain and the chain
        ///     takes one op per two rows. Four columns of single ops left the loop bound by the op's latency.
        ///     </para>
        ///     <para>
        ///     A row stride of 0 (a broadcast axis: every row is the same row) folds one row: each of these ops is idempotent
        ///     (<c>op(op(o, x), x) == op(o, x)</c>, bits included — a NaN stays the same NaN, a ±0 tie the same zero). A
        ///     strided call, or one NumPy runs without its SIMD branch (a float32 negative stride), makes the per-row calls.
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">The out row's first element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The first row's first element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="rs">The input's row stride (elements).</param>
        /// <param name="rows">The row count (&gt;= 1).</param>
        /// <param name="n">The elements per row.</param>
        internal static void LaneBinaryRows<T, TLane>(T* o, long os, T* x, long xs, long rs, long rows, long n)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            if (rs == 0)
                rows = 1;
            if (os != 1 || xs != 1 || !Vector256.IsHardwareAccelerated)
            {
                for (long r = 0; r < rows; r++)
                    LaneBinary<T, TLane>(o, os, x + r * rs, xs, n);
                return;
            }

            // LaneBinary's split for a contiguous call: the vector op over the whole vectors, then the tail's op.
            bool scalarTail = !TLane.PropagatesNaN && (typeof(T) == typeof(double) || typeof(T) == typeof(float));
            int w = Vector256<T>.Count;
            long v = n - n % w;
            long i = 0;
            for (; i + 4 * w <= v; i += 4 * w)
            {
                var a0 = Vector256.Load(o + i);
                var a1 = Vector256.Load(o + i + w);
                var a2 = Vector256.Load(o + i + 2 * w);
                var a3 = Vector256.Load(o + i + 3 * w);
                T* p = x + i;
                long r = 0;
                for (; r + 2 <= rows; r += 2, p += 2 * rs)
                {
                    a0 = TLane.N(a0, TLane.N(Vector256.Load(p), Vector256.Load(p + rs)));
                    a1 = TLane.N(a1, TLane.N(Vector256.Load(p + w), Vector256.Load(p + rs + w)));
                    a2 = TLane.N(a2, TLane.N(Vector256.Load(p + 2 * w), Vector256.Load(p + rs + 2 * w)));
                    a3 = TLane.N(a3, TLane.N(Vector256.Load(p + 3 * w), Vector256.Load(p + rs + 3 * w)));
                }

                if (r < rows)
                {
                    a0 = TLane.N(a0, Vector256.Load(p));
                    a1 = TLane.N(a1, Vector256.Load(p + w));
                    a2 = TLane.N(a2, Vector256.Load(p + 2 * w));
                    a3 = TLane.N(a3, Vector256.Load(p + 3 * w));
                }

                Vector256.Store(a0, o + i);
                Vector256.Store(a1, o + i + w);
                Vector256.Store(a2, o + i + 2 * w);
                Vector256.Store(a3, o + i + 3 * w);
            }

            for (; i < v; i += w)
            {
                var a = Vector256.Load(o + i);
                T* p = x + i;
                for (long r = 0; r < rows; r++, p += rs)
                    a = TLane.N(a, Vector256.Load(p));
                Vector256.Store(a, o + i);
            }

            for (; i < n; i++)
            {
                T a = o[i];
                T* p = x + i;
                for (long r = 0; r < rows; r++, p += rs)
                    a = scalarTail ? TLane.S(a, *p) : TLane.V(a, *p);
                o[i] = a;
            }
        }

        /// <summary>The OR reduce form (bool maximum / fmax) — <see cref="BoolReduce"/>.</summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static bool BoolOrReduce(bool acc, bool* x, long xs, long n) => BoolReduce(acc, x, xs, n, isOr: true);

        /// <summary>The AND reduce form (bool minimum / fmin) — <see cref="BoolReduce"/>.</summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static bool BoolAndReduce(bool acc, bool* x, long xs, long n) => BoolReduce(acc, x, xs, n, isOr: false);

        /// <summary>The OR elementwise form — <see cref="BoolBinary"/>.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="n">The element count.</param>
        internal static void BoolOrBinary(bool* o, long os, bool* x, long xs, long n) => BoolBinary(o, os, x, xs, n, isOr: true);

        /// <summary>The AND elementwise form — <see cref="BoolBinary"/>.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="n">The element count.</param>
        internal static void BoolAndBinary(bool* o, long os, bool* x, long xs, long n) => BoolBinary(o, os, x, xs, n, isOr: false);

        /// <summary>The bool loop forms: NumPy's <c>BOOL_logical_or</c> (maximum / fmax) or <c>BOOL_logical_and</c>.</summary>
        /// <param name="isMax">True for maximum / fmax (OR), false for minimum / fmin (AND).</param>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<bool> Bool(bool isMax)
            => isMax
                ? new(&BoolOrReduce, &BoolOrBinary, &BoolOrMaskedReduce, &BoolOrMaskedBinary, &BoolOrBinaryRows)
                : new(&BoolAndReduce, &BoolAndBinary, &BoolAndMaskedReduce, &BoolAndMaskedBinary, &BoolAndBinaryRows);

        /// <summary>
        ///     The masked reduce call of the bool loops in one piece: an OR is true when a SELECTED byte is nonzero, an AND
        ///     false when a selected byte is zero — the same answer however NumPy cuts the call into runs, 32 bytes per
        ///     step with an early exit.
        /// </summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        /// <param name="isOr">True for OR (maximum / fmax), false for AND (minimum / fmin).</param>
        /// <returns>The out's value after the call (0 / 1).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool BoolMaskedReduce(bool acc, bool* x, long xs, byte* m, long ms, long n, bool isOr)
        {
            bool decided = *(byte*)&acc != 0;
            if (decided == isOr)
                return decided;
            byte* p = (byte*)x;
            long i = 0;
            if (xs == 1 && ms == 1 && Vector256.IsHardwareAccelerated)
            {
                for (; i + 32 <= n; i += 32)
                {
                    var sel = ~Vector256.Equals(Vector256.Load(m + i), Vector256<byte>.Zero);
                    var zx = Vector256.Equals(Vector256.Load(p + i), Vector256<byte>.Zero);
                    // OR decides on a selected nonzero byte, AND on a selected zero byte.
                    var hit = isOr ? Vector256.AndNot(sel, zx) : sel & zx;
                    if (hit != Vector256<byte>.Zero)
                        return isOr;
                }
            }

            for (; i < n; i++)
                if (m[i * ms] != 0 && (p[i * xs] != 0) == isOr)
                    return isOr;
            return !isOr;
        }

        /// <summary>
        ///     The masked elementwise call of the bool loops in one piece: <c>o[i] = o[i] || x[i]</c> (or <c>&amp;&amp;</c>)
        ///     where <c>m[i]</c>, the other outputs untouched.
        /// </summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        /// <param name="isOr">True for OR, false for AND.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BoolMaskedBinary(bool* o, long os, bool* x, long xs, byte* m, long ms, long n, bool isOr)
        {
            byte* po = (byte*)o, px = (byte*)x;
            long i = 0;
            if (os == 1 && xs == 1 && ms == 1 && Vector256.IsHardwareAccelerated)
            {
                var one = Vector256.Create((byte)1);
                for (; i + 32 <= n; i += 32)
                {
                    var ov = Vector256.Load(po + i);
                    var zo = Vector256.Equals(ov, Vector256<byte>.Zero);
                    var zx = Vector256.Equals(Vector256.Load(px + i), Vector256<byte>.Zero);
                    var r = Vector256.AndNot(one, isOr ? zo & zx : zo | zx);
                    var sel = ~Vector256.Equals(Vector256.Load(m + i), Vector256<byte>.Zero);
                    Vector256.Store(Vector256.ConditionalSelect(sel, r, ov), po + i);
                }
            }

            for (; i < n; i++)
            {
                if (m[i * ms] == 0)
                    continue;
                byte a = po[i * os], b = px[i * xs];
                po[i * os] = (byte)((isOr ? (a != 0 || b != 0) : (a != 0 && b != 0)) ? 1 : 0);
            }
        }

        /// <summary>The masked OR reduce form — <see cref="BoolMaskedReduce"/>.</summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static bool BoolOrMaskedReduce(bool acc, bool* x, long xs, byte* m, long ms, long n)
            => BoolMaskedReduce(acc, x, xs, m, ms, n, isOr: true);

        /// <summary>The masked AND reduce form — <see cref="BoolMaskedReduce"/>.</summary>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static bool BoolAndMaskedReduce(bool acc, bool* x, long xs, byte* m, long ms, long n)
            => BoolMaskedReduce(acc, x, xs, m, ms, n, isOr: false);

        /// <summary>The masked OR elementwise form — <see cref="BoolMaskedBinary"/>.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride.</param>
        /// <param name="n">The element count.</param>
        internal static void BoolOrMaskedBinary(bool* o, long os, bool* x, long xs, byte* m, long ms, long n)
            => BoolMaskedBinary(o, os, x, xs, m, ms, n, isOr: true);

        /// <summary>The masked AND elementwise form — <see cref="BoolMaskedBinary"/>.</summary>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride.</param>
        /// <param name="n">The element count.</param>
        internal static void BoolAndMaskedBinary(bool* o, long os, bool* x, long xs, byte* m, long ms, long n)
            => BoolMaskedBinary(o, os, x, xs, m, ms, n, isOr: false);

        /// <summary>
        ///     A vector of lane masks from <c>Vector256&lt;T&gt;.Count</c> mask bytes: all ones where the byte is nonzero —
        ///     each byte zero-extended to the lane width (<c>vpmovzxb*</c>), then compared with zero.
        /// </summary>
        /// <typeparam name="T">The lane type (1, 2, 4 or 8 bytes).</typeparam>
        /// <param name="m">The first mask byte (Vector256&lt;T&gt;.Count of them are read).</param>
        /// <returns>The lane mask.</returns>
        /// <remarks>Callers check <see cref="Avx2.IsSupported"/>.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<T> MaskLanes<T>(byte* m) where T : unmanaged
        {
            if (sizeof(T) == 1)
                return (~Vector256.Equals(Vector256.Load(m), Vector256<byte>.Zero)).As<byte, T>();
            if (sizeof(T) == 2)
                return (~Vector256.Equals(Avx2.ConvertToVector256Int16(m), Vector256<short>.Zero)).As<short, T>();
            if (sizeof(T) == 4)
                return (~Vector256.Equals(Avx2.ConvertToVector256Int32(m), Vector256<int>.Zero)).As<int, T>();
            return (~Vector256.Equals(Avx2.ConvertToVector256Int64(m), Vector256<long>.Zero)).As<long, T>();
        }

        /// <summary>
        ///     The masked reduce call of an INTEGER loop in one piece: the extreme of the selected elements and the out's
        ///     value. An integer maximum / minimum has no NaN and no bit-distinct tie, so the answer is the same however
        ///     NumPy cuts the call into masked runs — here two vector accumulators, the unselected lanes replaced by the
        ///     identity (the dtype's minimum for max, maximum for min).
        /// </summary>
        /// <typeparam name="T">The integer lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule (max or min).</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static T IntMaskedReduce<T, TLane>(T acc, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged, INumber<T>, IMinMaxValue<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            long i = 0;
            if (xs == 1 && ms == 1 && Avx2.IsSupported)
            {
                // The identity: the value the rule never picks over another (MaxValue beats MinValue for a max rule).
                T id = TLane.Beats(T.MaxValue, T.MinValue) ? T.MinValue : T.MaxValue;
                var vid = Vector256.Create(id);
                var v0 = vid;
                var v1 = vid;
                int w = Vector256<T>.Count;
                for (; i + 2 * w <= n; i += 2 * w)
                {
                    v0 = TLane.N(v0, Vector256.ConditionalSelect(MaskLanes<T>(m + i), Vector256.Load(x + i), vid));
                    v1 = TLane.N(v1, Vector256.ConditionalSelect(MaskLanes<T>(m + i + w), Vector256.Load(x + i + w), vid));
                }

                for (; i + w <= n; i += w)
                    v0 = TLane.N(v0, Vector256.ConditionalSelect(MaskLanes<T>(m + i), Vector256.Load(x + i), vid));
                v0 = TLane.N(v0, v1);
                for (int k = 0; k < w; k++)
                    acc = TLane.S(acc, v0.GetElement(k));
            }

            for (; i < n; i++)
                if (m[i * ms] != 0)
                    acc = TLane.S(acc, x[i * xs]);
            return acc;
        }

        /// <summary>
        ///     The masked elementwise call of an INTEGER loop in one piece: <c>o[i] = op(o[i], x[i])</c> where <c>m[i]</c> —
        ///     a blend per vector, exact for the same reason as <see cref="IntMaskedReduce{T,TLane}"/>.
        /// </summary>
        /// <typeparam name="T">The integer lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        internal static void IntMaskedBinary<T, TLane>(T* o, long os, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged, INumber<T>, IMinMaxValue<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            long i = 0;
            if (os == 1 && xs == 1 && ms == 1 && Avx2.IsSupported)
            {
                int w = Vector256<T>.Count;
                for (; i + w <= n; i += w)
                {
                    var ov = Vector256.Load(o + i);
                    var r = TLane.N(ov, Vector256.Load(x + i));
                    Vector256.Store(Vector256.ConditionalSelect(MaskLanes<T>(m + i), r, ov), o + i);
                }
            }

            for (; i < n; i++)
                if (m[i * ms] != 0)
                    o[i * os] = TLane.S(o[i * os], x[i * xs]);
        }

        /// <summary>
        ///     The next position at or after <paramref name="i"/> whose mask byte is selected (<paramref name="selected"/>)
        ///     or unselected, in a contiguous mask: 32 bytes per step through one compare, a movemask and a trailing-zero
        ///     count, so a masked call's runs are found without one span search per run edge.
        /// </summary>
        /// <param name="m">The call's first mask byte (stride 1).</param>
        /// <param name="i">The position the search starts at.</param>
        /// <param name="n">The call's element count.</param>
        /// <param name="selected">True to find a selected (nonzero) byte, false to find an unselected (zero) one.</param>
        /// <returns>The position found, or <paramref name="n"/> when none remains.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long NextMaskRunEdge(byte* m, long i, long n, bool selected)
        {
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 32 <= n; i += 32)
                {
                    uint zero = Vector256.Equals(Vector256.Load(m + i), Vector256<byte>.Zero).ExtractMostSignificantBits();
                    uint hit = selected ? ~zero : zero;
                    if (hit != 0)
                        return i + BitOperations.TrailingZeroCount(hit);
                }
            }

            for (; i < n; i++)
                if ((m[i] != 0) == selected)
                    return i;
            return n;
        }

        /// <summary>
        ///     The next run of selected elements at or after <paramref name="i"/>: <c>[s, e)</c>, every mask byte in it
        ///     nonzero, the one before it (if any after <paramref name="i"/>) zero and the one at <c>e</c> zero or past the
        ///     end — the run one <c>generic_masked_strided_loop</c> call hands to the loop.
        /// </summary>
        /// <param name="m">The call's first mask byte.</param>
        /// <param name="ms">The mask's byte stride (any nonzero value; 1 takes the movemask scan).</param>
        /// <param name="i">The position the search starts at.</param>
        /// <param name="n">The call's element count.</param>
        /// <param name="s">Receives the run's first position.</param>
        /// <param name="e">Receives the position after the run.</param>
        /// <returns>False when no selected element remains (then <paramref name="s"/> and <paramref name="e"/> are
        ///     <paramref name="n"/>).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool NextMaskRun(byte* m, long ms, long i, long n, out long s, out long e)
        {
            if (ms == 1)
            {
                s = NextMaskRunEdge(m, i, n, selected: true);
                e = s == n ? n : NextMaskRunEdge(m, s + 1, n, selected: false);
                return s < n;
            }

            // A strided mask (a mask broadcast along another axis, or a stepped view): the edges byte by byte.
            s = i;
            while (s < n && m[s * ms] == 0)
                s++;
            e = s;
            while (e < n && m[e * ms] != 0)
                e++;
            return s < n;
        }

        /// <summary>
        ///     The masked reduce call of a float SIMD loop (float32 / float64 under maximum / minimum / fmax / fmin):
        ///     NumPy's <c>generic_masked_strided_loop</c> — one loop call per run of selected elements, in order — with the
        ///     runs found by the movemask scan, and over a contiguous input each run applied through its transfer function
        ///     (<see cref="NRunContiguous{T,TLane}"/> for the N rules, <see cref="PRunContiguous{T,TLane}"/> for the P
        ///     rules), so the run's own work leaves the dependency chain through the out's value.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's cuts are part of these loops' answer: each run is its own call, whose horizontal reduce
        ///     re-canonicalizes a NaN (N rules) or fills NaN lanes (P rules) and whose vector section decides which zero
        ///     survives a ±0 tie. The result here is exactly the per-run calls'. What changes is the cost: with a typical
        ///     mask most runs are a few elements long, and one call per run is a ~40-cycle chain through the out's value
        ///     (broadcast, vector op, horizontal reduce, scalar tail) — the RUN count, not the element count, sets NumPy's
        ///     time and the plain per-run port's.
        ///     </para>
        ///     <para>
        ///     A strided input makes the call itself per run (<see cref="LaneReduce{T,TLane}"/>: NumPy's 8-accumulator
        ///     strided loop).
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The lane type (float32 / float64).</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask's byte stride (nonzero: a stride-0 mask never reaches the masked forms).</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static T LaneMaskedReduce<T, TLane>(T acc, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            if (xs != 1)
            {
                for (long i = 0; NextMaskRun(m, ms, i, n, out long s, out long e); i = e)
                    acc = LaneReduce<T, TLane>(acc, x + s * xs, xs, e - s);
                return acc;
            }

            // A contiguous input: each run through its transfer function (TLane.PropagatesNaN is a JIT-time constant).
            for (long i = 0; NextMaskRun(m, ms, i, n, out long s, out long e); i = e)
            {
                acc = TLane.PropagatesNaN
                    ? NRunContiguous<T, TLane>(acc, x + s, e - s)
                    : PRunContiguous<T, TLane>(acc, x + s, e - s);
            }

            return acc;
        }

        /// <summary>
        ///     One P-rule <c>simd_reduce_c</c> call (np.nanmax / np.nanmin of float32 / float64: <c>npyv_maxp</c> /
        ///     <c>npyv_minp</c> vectors, the CRT <c>fmax</c> / <c>fmin</c> tail) over a contiguous run of fewer than eight
        ///     vectors, computed as a function of the out's value <paramref name="acc"/> and of numbers taken from the run
        ///     alone — the twin of <see cref="NRunContiguous{T,TLane}"/>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The vector op <c>maxp(a, b) = isnan(b) ? a : (a beats b ? a : b)</c> (a NaN <c>a</c> loses to any number) and
        ///     the scalar <c>fmax</c> are associative, so lane j of <c>splat(acc)</c> after the vector section is
        ///     <c>maxp(acc, L_j)</c> with <c>L_j</c> the run's own lane fold, and the tail folds to one value first. Then
        ///     <c>npyv_reduce_maxp</c> of those lanes is:
        ///     </para>
        ///     <list type="bullet">
        ///         <item>every <c>L_j</c> NaN: every lane is <c>acc</c>, and the result is <c>acc</c> (lane 0 as it is when
        ///         <c>acc</c> is NaN too);</item>
        ///         <item><c>acc</c> NaN: the lanes are <c>L</c> itself, so the result is <c>h</c>, the reduce of <c>L</c>
        ///         (its NaN lanes filled with ∓inf, then the plain cascade);</item>
        ///         <item>otherwise, with <c>M</c> the value of <c>h</c>: every lane is a number — <c>acc beats L_j ? acc :
        ///         L_j</c> for a number <c>L_j</c>, <c>acc</c> for a NaN one — so <c>acc</c> beating <c>M</c> gives
        ///         <c>acc</c>, <c>M</c> beating <c>acc</c> gives <c>h</c> (the cascade picks among the lanes of value
        ///         <c>M</c> by position only), and a tie gives the last lane: <c>L_last</c> when it is a number of value
        ///         <c>M</c>, else <c>acc</c>.</item>
        ///     </list>
        ///     <para>
        ///     A run shorter than one vector: <c>npyv_reduce_maxp(splat(acc))</c> is <c>acc</c> itself (lane 0 when every
        ///     lane is NaN, else the cascade of equal lanes), then the elements' <c>fmax</c> fold. A run of eight vectors or
        ///     more makes the call itself (<see cref="LaneReduce{T,TLane}"/>).
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The lane type (float32 / float64).</typeparam>
        /// <typeparam name="TLane">A P rule (<see cref="NumPyMinMaxReduce.FMaxLane{T}"/> / <see cref="NumPyMinMaxReduce.FMinLane{T}"/>).</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="p">The run's first element (contiguous).</param>
        /// <param name="len">The run's element count (&gt;= 1).</param>
        /// <returns>The out's value after the call.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T PRunContiguous<T, TLane>(T acc, T* p, long len)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            int vstep = Vector256<T>.Count;
            if (len >= 8L * vstep)
                return LaneReduce<T, TLane>(acc, p, 1, len);

            long v = len - len % vstep;
            T r;
            if (v == 0)
            {
                r = acc;
            }
            else
            {
                var L = Vector256.Load(p);
                for (long k = vstep; k < v; k += vstep)
                    L = TLane.N(L, Vector256.Load(p + k));

                if (!Vector256.EqualsAny(L, L))
                {
                    r = acc;   // every L_j NaN: every lane holds acc
                }
                else
                {
                    T h = NumPyMinMaxReduce.ReduceLanes<T, TLane>(L);
                    if (T.IsNaN(acc))
                    {
                        r = h;
                    }
                    else
                    {
                        T last = L.GetElement(vstep - 1);
                        r = TLane.Beats(acc, h) ? acc : TLane.Beats(h, acc) ? h : last == h ? last : acc;
                    }
                }
            }

            if (v < len)
            {
                T t = p[v];
                for (long k = v + 1; k < len; k++)
                    t = TLane.S(t, p[k]);
                r = TLane.S(r, t);
            }

            return r;
        }

        /// <summary>
        ///     One N-rule <c>simd_reduce_c</c> call (np.max / np.min of float32 / float64) over a contiguous run of fewer than
        ///     eight vectors, computed as a function of the out's value <paramref name="acc"/> and of numbers taken from the
        ///     run alone — NumPy's bits, with only a few compares and selects depending on <paramref name="acc"/>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     NumPy's call: <c>a = splat(acc)</c>; every whole vector folded in with the vector op; <c>r =
        ///     npyv_reduce_maxn(a)</c> — the canonical NaN when any lane is NaN, else the plain lane cascade; then every
        ///     leftover element with the scalar op. For these loops the vector op and the scalar op are one rule,
        ///     <c>N(a, b) = isnan(a) ? a : (a beats b ? a : b)</c> (a tie keeps the SECOND operand), and it is associative.
        ///     So:
        ///     </para>
        ///     <list type="bullet">
        ///         <item>lane j of <c>a</c> is <c>N(acc, L_j)</c>, where <c>L_j</c> is the fold of lane j's elements of the
        ///         vector section — <c>L</c> does not depend on <paramref name="acc"/>;</item>
        ///         <item>the leftover elements fold to one value <c>t</c> first, and <c>r = N(r, t)</c>;</item>
        ///         <item>a NaN <paramref name="acc"/> or a NaN in <c>L</c> gives a NaN lane, so <c>r</c> is the canonical
        ///         NaN, and no leftover element changes it;</item>
        ///         <item>otherwise, with <c>h</c> the cascade of <c>L</c> and <c>M</c> its value, lane j is <c>acc beats L_j ?
        ///         acc : L_j</c>. If <c>acc</c> beats <c>M</c>, every lane is <c>acc</c> and <c>r = acc</c>. If <c>M</c> beats
        ///         <c>acc</c>, the lanes of value <c>M</c> are those of <c>L</c>, bits included, and the others stay below
        ///         <c>M</c>; the cascade never picks a lane below the maximum, and among the maximal lanes its choice
        ///         depends only on their positions, so <c>r = h</c>. On a tie (<c>acc == M</c>: equal values, or zeros of
        ///         either sign) every lane has the value <c>M</c>, the cascade's plain op keeps its SECOND operand at every
        ///         level, and the last lane is returned: <c>L_last</c> when its value is <c>M</c>, else <c>acc</c>.</item>
        ///     </list>
        ///     <para>
        ///     A run shorter than one vector has no vector section: <c>r = npyv_reduce_maxn(splat(acc))</c> is the canonical
        ///     NaN for a NaN <paramref name="acc"/> and <paramref name="acc"/> itself otherwise, then the elements' fold.
        ///     A run of eight vectors or more makes the call itself (<see cref="LaneReduce{T,TLane}"/>, NumPy's group tree):
        ///     its own work outweighs the chain.
        ///     </para>
        /// </remarks>
        /// <typeparam name="T">The lane type (float32 / float64).</typeparam>
        /// <typeparam name="TLane">An N rule (<see cref="NumPyMinMaxReduce.MaxLane{T}"/> / <see cref="NumPyMinMaxReduce.MinLane{T}"/>).</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="p">The run's first element (contiguous).</param>
        /// <param name="len">The run's element count (&gt;= 1).</param>
        /// <returns>The out's value after the call.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T NRunContiguous<T, TLane>(T acc, T* p, long len)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            int vstep = Vector256<T>.Count;
            if (len >= 8L * vstep)
                return LaneReduce<T, TLane>(acc, p, 1, len);

            long v = len - len % vstep;
            if (v == 0)
            {
                // No vector section: the elements' fold, then the horizontal reduce of splat(acc) in front of it.
                T t = p[0];
                for (long k = 1; k < len; k++)
                    t = TLane.S(t, p[k]);
                return T.IsNaN(acc) ? NumPyMinMaxReduce.CanonicalNaN<T>() : TLane.S(acc, t);
            }

            var L = Vector256.Load(p);
            for (long k = vstep; k < v; k += vstep)
                L = TLane.N(L, Vector256.Load(p + k));

            // A NaN lane: the canonical NaN, which no leftover element replaces (N keeps a NaN first operand).
            if (T.IsNaN(acc) || !Vector256.EqualsAll(L, L))
                return NumPyMinMaxReduce.CanonicalNaN<T>();

            T h = NumPyMinMaxReduce.ReduceLanes<T, TLane>(L);
            T last = L.GetElement(vstep - 1);
            T r = TLane.Beats(acc, h) ? acc : TLane.Beats(h, acc) ? h : last == h ? last : acc;
            if (v < len)
            {
                T t = p[v];
                for (long k = v + 1; k < len; k++)
                    t = TLane.S(t, p[k]);
                r = TLane.S(r, t);
            }

            return r;
        }

        /// <summary>
        ///     The masked elementwise call of a float SIMD loop. The N rules (np.max / np.min) apply one rule to every
        ///     element whatever the call's cuts (their vector and scalar ops agree), so the call is one masked pass — a
        ///     blend per vector. The P rules (np.nanmax / np.nanmin) give an element the vector op inside its run's vector
        ///     section and the scalar op in the run's tail, and the two differ on a ±0 tie and on two NaNs, so every run is
        ///     its own <see cref="LaneBinary{T,TLane}"/> call, as in NumPy.
        /// </summary>
        /// <typeparam name="T">The lane type (float32 / float64).</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask's byte stride (nonzero).</param>
        /// <param name="n">The element count.</param>
        internal static void LaneMaskedBinary<T, TLane>(T* o, long os, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
        {
            if (TLane.PropagatesNaN)
            {
                long i = 0;
                if (os == 1 && xs == 1 && ms == 1 && Avx2.IsSupported)
                {
                    int w = Vector256<T>.Count;
                    for (; i + w <= n; i += w)
                    {
                        var ov = Vector256.Load(o + i);
                        var r = TLane.N(ov, Vector256.Load(x + i));
                        Vector256.Store(Vector256.ConditionalSelect(MaskLanes<T>(m + i), r, ov), o + i);
                    }
                }

                for (; i < n; i++)
                    if (m[i * ms] != 0)
                        o[i * os] = TLane.S(o[i * os], x[i * xs]);
                return;
            }

            for (long i = 0; NextMaskRun(m, ms, i, n, out long s, out long e); i = e)
                LaneBinary<T, TLane>(o + s * os, os, x + s * xs, xs, e - s);
        }

        /// <summary>
        ///     The masked reduce call of a sequential loop in one piece: the rule folded over the selected elements in order —
        ///     the fold NumPy's runs make, without one call per run.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold rule.</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static T SeqMaskedReduce<T, TRule>(T acc, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            for (long i = 0; i < n; i++, x += xs, m += ms)
                if (*m != 0)
                    acc = TRule.Fold(acc, *x);
            return acc;
        }

        /// <summary>The masked elementwise call of a sequential loop in one piece: <c>o[i] = rule(o[i], x[i])</c> where
        ///     <c>m[i]</c>.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold rule.</typeparam>
        /// <param name="o">The first out element.</param>
        /// <param name="os">The out stride.</param>
        /// <param name="x">The first input element.</param>
        /// <param name="xs">The input stride.</param>
        /// <param name="m">The first mask byte.</param>
        /// <param name="ms">The mask stride (bytes).</param>
        /// <param name="n">The element count.</param>
        internal static void SeqMaskedBinary<T, TRule>(T* o, long os, T* x, long xs, byte* m, long ms, long n)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            for (long i = 0; i < n; i++, o += os, x += xs, m += ms)
                if (*m != 0)
                    *o = TRule.Fold(*o, *x);
        }

        /// <summary>
        ///     The integer SIMD-loop forms with their whole-call masked forms (an integer result does not depend on NumPy's
        ///     masked run cuts).
        /// </summary>
        /// <typeparam name="T">The integer lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> IntLane<T, TLane>()
            where T : unmanaged, INumber<T>, IMinMaxValue<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
            => new(&LaneReduce<T, TLane>, &LaneBinary<T, TLane>, &IntMaskedReduce<T, TLane>, &IntMaskedBinary<T, TLane>,
                &LaneBinaryRows<T, TLane>);

        /// <summary>
        ///     The float SIMD-loop forms (float32 / float64) with their masked forms, which make NumPy's per-run calls
        ///     themselves (<see cref="LaneMaskedReduce{T,TLane}"/>, <see cref="LaneMaskedBinary{T,TLane}"/>).
        /// </summary>
        /// <typeparam name="T">The lane type.</typeparam>
        /// <typeparam name="TLane">The lane rule.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> Lane<T, TLane>()
            where T : unmanaged, INumber<T> where TLane : struct, NumPyMinMaxReduce.ILane<T>
            => new(&LaneReduce<T, TLane>, &LaneBinary<T, TLane>, &LaneMaskedReduce<T, TLane>, &LaneMaskedBinary<T, TLane>,
                &LaneBinaryRows<T, TLane>);

        /// <summary>The sequential-loop forms for an element type and rule.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold rule.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<T> Seq<T, TRule>()
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
            => new(&SeqReduce<T, TRule>, &SeqBinary<T, TRule>, &SeqMaskedReduce<T, TRule>, &SeqMaskedBinary<T, TRule>);

        /// <summary>
        ///     The complex128 loop forms (<c>CDOUBLE_maximum</c> / <c>minimum</c> / <c>fmax</c> / <c>fmin</c>): the reduce and
        ///     elementwise calls vectorized (<see cref="ComplexReduce{TRule}"/>, <see cref="ComplexBinary{TRule}"/>), the
        ///     masked calls the sequential folds.
        /// </summary>
        /// <typeparam name="TRule">One of <see cref="ComplexMaxRule"/>, <see cref="ComplexMinRule"/>,
        ///     <see cref="ComplexFMaxRule"/>, <see cref="ComplexFMinRule"/>.</typeparam>
        /// <returns>The loop.</returns>
        internal static NumPyReduceIterator.Loop<Complex> ComplexLoop<TRule>() where TRule : struct, ISequentialNanRule<Complex>
            => new(&ComplexReduce<TRule>, &ComplexBinary<TRule>, &SeqMaskedReduce<Complex, TRule>, &SeqMaskedBinary<Complex, TRule>);

        /// <summary>How many complex elements one block of <see cref="ComplexReduce{TRule}"/> covers: 16 KB, so its second pass
        ///     re-reads the block from L1.</summary>
        private const long ComplexBlock = 1024;

        /// <summary>True for the maximum-like complex rules (<c>CGE</c>), false for the minimum-like ones (<c>CLE</c>).</summary>
        /// <typeparam name="TRule">A complex rule.</typeparam>
        /// <returns>A JIT-time constant per rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ComplexIsMax<TRule>() => typeof(TRule) == typeof(ComplexMaxRule) || typeof(TRule) == typeof(ComplexFMaxRule);

        /// <summary>True for the NaN-propagating complex rules (maximum / minimum), false for fmax / fmin.</summary>
        /// <typeparam name="TRule">A complex rule.</typeparam>
        /// <returns>A JIT-time constant per rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ComplexPropagates<TRule>() => typeof(TRule) == typeof(ComplexMaxRule) || typeof(TRule) == typeof(ComplexMinRule);

        /// <summary>
        ///     The IS_BINARY_REDUCE call of a complex128 loop: NumPy's sequential fold (<c>CDOUBLE_maximum</c> &amp; co.,
        ///     scalar in NumPy), here in blocks of <see cref="ComplexBlock"/> elements whose fold is found with two vector
        ///     passes (<see cref="ComplexBlockFold{TRule}"/>) and then folded into the running value.
        /// </summary>
        /// <remarks>
        ///     Exact because each of the four rules is an associative SELECTION — the fold of a sequence is one of its
        ///     elements, bits included — so <c>fold(acc, block) = rule(acc, fold(block))</c>. Under maximum / minimum a
        ///     running value with a NaN part never changes again, so the call stops there. A strided input (and a machine
        ///     without AVX) runs the scalar fold.
        /// </remarks>
        /// <typeparam name="TRule">The complex rule.</typeparam>
        /// <param name="acc">The out's value before the call.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        /// <returns>The out's value after the call.</returns>
        internal static Complex ComplexReduce<TRule>(Complex acc, Complex* x, long xs, long n) where TRule : struct, ISequentialNanRule<Complex>
        {
            if (xs != 1 || !Avx.IsSupported)
                return SeqReduce<Complex, TRule>(acc, x, xs, n);

            for (long b = 0; b < n; b += ComplexBlock)
            {
                if (ComplexPropagates<TRule>() && (double.IsNaN(acc.Real) || double.IsNaN(acc.Imaginary)))
                    return acc;
                acc = TRule.Fold(acc, ComplexBlockFold<TRule>(x + b, Math.Min(ComplexBlock, n - b)));
            }

            return acc;
        }

        /// <summary>
        ///     The fold of a contiguous block under a complex rule — which element it selects — in two vector passes of four
        ///     elements per step (real and imaginary parts unpacked into separate vectors), then the leftover elements one by
        ///     one.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     The selection each rule makes (NumPy's <c>CGE</c> / <c>CLE</c> compare: the real parts first, then the
        ///     imaginary parts; a tie, ±0 included, keeps the EARLIER element):
        ///     </para>
        ///     <list type="bullet">
        ///         <item>maximum / minimum: the first element with a NaN part, if any (a NaN replaces every running value and then
        ///         stays); else the first element of the extreme value;</item>
        ///         <item>fmax / fmin: the first element of the extreme value among the elements without a NaN part (a NaN element
        ///         is skipped, and a NaN running value gives way to the first number); when every element has a NaN part, the
        ///         first element (which the caller's rule then skips).</item>
        ///     </list>
        ///     <para>
        ///     Pass 1 tracks the extreme VALUE in two sets of four lanes (two sets so the compare-and-blend chains overlap) and
        ///     whether a NaN part occurs; pass 2 looks for the first element equal to that value (or the first element with a
        ///     NaN part) and stops there. The element's own bits are returned.
        ///     </para>
        /// </remarks>
        /// <typeparam name="TRule">The complex rule.</typeparam>
        /// <param name="x">The block's first element.</param>
        /// <param name="len">The block's element count (&gt;= 1).</param>
        /// <returns>The element the block's fold selects.</returns>
        private static Complex ComplexBlockFold<TRule>(Complex* x, long len) where TRule : struct, ISequentialNanRule<Complex>
        {
            bool isMax = ComplexIsMax<TRule>();
            bool propagate = ComplexPropagates<TRule>();
            double* p = (double*)x;
            long v = len & ~7L;
            Complex rep;
            long k = 0;
            if (v == 0)
            {
                rep = x[0];
                k = 1;
            }
            else
            {
                // Pass 1: the extreme value (and, under maximum / minimum, whether a NaN part occurs). Lanes hold the parts in
                // the unpacked order [c0, c2, c1, c3]; the order does not matter for the value.
                var start = Vector256.Create(isMax ? double.NegativeInfinity : double.PositiveInfinity);
                Vector256<double> r0 = start, i0 = start, r1 = start, i1 = start, nan = Vector256<double>.Zero;
                for (long j = 0; j < v; j += 8)
                {
                    double* q = p + 2 * j;
                    ComplexExtremeStep<TRule>(Avx.LoadVector256(q), Avx.LoadVector256(q + 4), ref r0, ref i0, ref nan);
                    ComplexExtremeStep<TRule>(Avx.LoadVector256(q + 8), Avx.LoadVector256(q + 12), ref r1, ref i1, ref nan);
                }

                // The two lane sets, then the four lanes, folded to one extreme value.
                var t = ComplexBeyond<TRule>(r1, i1, r0, i0);
                r0 = Avx.BlendVariable(r0, r1, t);
                i0 = Avx.BlendVariable(i0, i1, t);
                double mr = r0.GetElement(0), mi = i0.GetElement(0);
                for (int l = 1; l < 4; l++)
                {
                    double lr = r0.GetElement(l), li = i0.GetElement(l);
                    if (isMax ? lr > mr || (lr == mr && li > mi) : lr < mr || (lr == mr && li < mi))
                    {
                        mr = lr;
                        mi = li;
                    }
                }

                // Pass 2: the first element with a NaN part (maximum / minimum, when one occurred) or of the extreme value.
                bool nanPath = propagate && Avx.MoveMask(nan) != 0;
                var vr = Vector256.Create(mr);
                var vi = Vector256.Create(mi);
                long found = -1;
                for (long j = 0; j < v; j += 4)
                {
                    double* q = p + 2 * j;
                    var a = Avx.LoadVector256(q);
                    var b = Avx.LoadVector256(q + 4);
                    var xr = Avx.UnpackLow(a, b);
                    var xi = Avx.UnpackHigh(a, b);
                    var hit = nanPath
                        ? Avx.CompareUnordered(xr, xi)
                        : Avx.And(Avx.CompareEqual(xr, vr), Avx.CompareEqual(xi, vi));
                    int bits = Avx.MoveMask(hit);
                    if (bits != 0)
                    {
                        // Lane order [c0, c2, c1, c3]: bit 0 is element j, bit 2 element j + 1, bit 1 j + 2, bit 3 j + 3.
                        found = j + ((bits & 1) != 0 ? 0 : (bits & 4) != 0 ? 1 : (bits & 2) != 0 ? 2 : 3);
                        break;
                    }
                }

                // No element of the extreme value: fmax / fmin over elements that all have a NaN part — the fold of such a
                // sequence is its first element.
                rep = found >= 0 ? x[found] : x[0];
                k = v;
            }

            for (; k < len; k++)
                rep = TRule.Fold(rep, x[k]);
            return rep;
        }

        /// <summary>
        ///     One pass-1 step of <see cref="ComplexBlockFold{TRule}"/> over four elements (two interleaved vectors): a lane
        ///     takes its element when the element's value lies beyond the lane's; fmax / fmin skip an element with a NaN part,
        ///     maximum / minimum note that one occurred (their lanes' values are then never read).
        /// </summary>
        /// <typeparam name="TRule">The complex rule (a generic argument, so the rule's tests are JIT-time constants).</typeparam>
        /// <param name="a">Elements j and j + 1 (real, imaginary, real, imaginary).</param>
        /// <param name="b">Elements j + 2 and j + 3.</param>
        /// <param name="mr">The lanes' real parts.</param>
        /// <param name="mi">The lanes' imaginary parts.</param>
        /// <param name="nanSeen">All ones in a lane that met an element with a NaN part (maximum / minimum only).</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ComplexExtremeStep<TRule>(Vector256<double> a, Vector256<double> b, ref Vector256<double> mr,
            ref Vector256<double> mi, ref Vector256<double> nanSeen) where TRule : struct, ISequentialNanRule<Complex>
        {
            var xr = Avx.UnpackLow(a, b);
            var xi = Avx.UnpackHigh(a, b);
            var take = ComplexBeyond<TRule>(xr, xi, mr, mi);
            if (ComplexPropagates<TRule>())
                nanSeen = Avx.Or(nanSeen, Avx.CompareUnordered(xr, xi));
            else
                take = Avx.And(take, Avx.CompareOrdered(xr, xi));
            mr = Avx.BlendVariable(mr, xr, take);
            mi = Avx.BlendVariable(mi, xi, take);
        }

        /// <summary>Lane-wise "(ar, ai) lies strictly beyond (br, bi)" in the rule's lexicographic order (greater for the
        ///     maximum-like rules, less for the minimum-like ones; a NaN part makes it false).</summary>
        /// <typeparam name="TRule">The complex rule.</typeparam>
        /// <param name="ar">The first operand's real parts.</param>
        /// <param name="ai">The first operand's imaginary parts.</param>
        /// <param name="br">The second operand's real parts.</param>
        /// <param name="bi">The second operand's imaginary parts.</param>
        /// <returns>All ones where the first operand lies beyond the second.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> ComplexBeyond<TRule>(Vector256<double> ar, Vector256<double> ai, Vector256<double> br,
            Vector256<double> bi) where TRule : struct, ISequentialNanRule<Complex>
            => ComplexIsMax<TRule>()
                ? Avx.Or(Avx.CompareGreaterThan(ar, br), Avx.And(Avx.CompareEqual(ar, br), Avx.CompareGreaterThan(ai, bi)))
                : Avx.Or(Avx.CompareLessThan(ar, br), Avx.And(Avx.CompareEqual(ar, br), Avx.CompareLessThan(ai, bi)));

        /// <summary>
        ///     The elementwise call of a complex128 loop: <c>o[i] = rule(o[i], x[i])</c>, four elements per step for
        ///     contiguous operands — the real and imaginary parts unpacked, NumPy's keep condition evaluated lane by lane,
        ///     and each element's two parts blended together (an element is kept or replaced whole, bits included).
        /// </summary>
        /// <remarks>
        ///     NumPy's conditions (<c>loops.c.src</c>): <c>CDOUBLE_maximum</c> keeps <c>o</c> when <c>o</c> has a NaN part or
        ///     <c>CGE(o, x)</c> — <c>(or &gt; xr &amp;&amp; !isnan(oi) &amp;&amp; !isnan(xi)) || (or == xr &amp;&amp; oi &gt;= xi)</c>;
        ///     <c>CDOUBLE_fmax</c> keeps <c>o</c> when <c>x</c> has a NaN part or <c>CGE(o, x)</c>; minimum / fmin use
        ///     <c>CLE</c>. Each term is one ordered compare per lane.
        /// </remarks>
        /// <typeparam name="TRule">The complex rule.</typeparam>
        /// <param name="o">The call's first out element.</param>
        /// <param name="os">The out's element stride.</param>
        /// <param name="x">The call's first input element.</param>
        /// <param name="xs">The input's element stride.</param>
        /// <param name="n">The element count.</param>
        internal static void ComplexBinary<TRule>(Complex* o, long os, Complex* x, long xs, long n) where TRule : struct, ISequentialNanRule<Complex>
        {
            bool isMax = ComplexIsMax<TRule>();
            bool propagate = ComplexPropagates<TRule>();
            long i = 0;
            if (os == 1 && xs == 1 && Avx.IsSupported)
            {
                double* po = (double*)o, px = (double*)x;
                for (; i + 4 <= n; i += 4)
                {
                    double* qo = po + 2 * i, qx = px + 2 * i;
                    var o0 = Avx.LoadVector256(qo);
                    var o1 = Avx.LoadVector256(qo + 4);
                    var x0 = Avx.LoadVector256(qx);
                    var x1 = Avx.LoadVector256(qx + 4);
                    var ar = Avx.UnpackLow(o0, o1);
                    var ai = Avx.UnpackHigh(o0, o1);
                    var xr = Avx.UnpackLow(x0, x1);
                    var xi = Avx.UnpackHigh(x0, x1);
                    var beyond = isMax ? Avx.CompareGreaterThan(ar, xr) : Avx.CompareLessThan(ar, xr);
                    var tie = Avx.And(Avx.CompareEqual(ar, xr), isMax ? Avx.CompareGreaterThanOrEqual(ai, xi) : Avx.CompareLessThanOrEqual(ai, xi));
                    // maximum: o has a NaN part, or (o beyond x and x's imaginary part is a number), or the tie;
                    // fmax: x has a NaN part, or (o beyond x and o's imaginary part is a number), or the tie.
                    var keep = propagate
                        ? Avx.Or(Avx.Or(Avx.CompareUnordered(ar, ai), Avx.AndNot(Avx.CompareUnordered(xi, xi), beyond)), tie)
                        : Avx.Or(Avx.Or(Avx.CompareUnordered(xr, xi), Avx.AndNot(Avx.CompareUnordered(ai, ai), beyond)), tie);
                    // Back to the interleaved layout: element j's flag on both of its parts.
                    Avx.Store(qo, Avx.BlendVariable(x0, o0, Avx.UnpackLow(keep, keep)));
                    Avx.Store(qo + 4, Avx.BlendVariable(x1, o1, Avx.UnpackHigh(keep, keep)));
                }
            }

            for (; i < n; i++)
                o[i * os] = TRule.Fold(o[i * os], x[i * xs]);
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.max</c> / <c>np.min</c> / <c>np.nanmax</c> / <c>np.nanmin</c> with NumPy's keywords — <c>ufunc.reduce</c>
        ///     of <c>maximum</c> / <c>minimum</c> / <c>fmax</c> / <c>fmin</c> (see the file header for the statement order and
        ///     the two engines).
        /// </summary>
        /// <param name="a">The array to reduce, any layout.</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array to receive the result (any dtype: the loop's result is cast into it), or <c>null</c>.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from (<see cref="ReductionInitial"/>'s conversion), or
        ///     <c>null</c> for none.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: only selected elements reduce. Requires
        ///     <paramref name="initial"/>.</param>
        /// <param name="op">The reduction (<see cref="MinMaxOp.FMax"/> / <see cref="MinMaxOp.FMin"/> for np.nanmax / np.nanmin).</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array in <paramref name="a"/>'s dtype, a read-only
        ///     0-d array (NumPy's scalar) for a flat reduction without keepdims.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">A read-only or misshapen out, a mask with more axes than <paramref name="a"/>, a mask
        ///     without <paramref name="initial"/>, or an empty reduced axis without <paramref name="initial"/> (NumPy's
        ///     texts); an unconvertible <paramref name="initial"/> (a NaN into an integer loop, a sequence).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException">The mask or out does not broadcast against <paramref name="a"/>.</exception>
        /// <exception cref="OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is a complex for a real loop, or of an unconvertible type.</exception>
        internal static unsafe NDArray MaxMinNumPy(NDArray a, int? axis, NDArray @out, bool keepdims, object initial, NDArray @where,
            MinMaxOp op)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            string funcname = NumPyMinMaxReduce.UfuncName(op);
            // 1. The reduce prologue, NumPy's order and texts (read-only out, mask dtype, axis, out rank, iterator shapes).
            // Text in the input or the out: NumPy resolves the loop on (out, a) between the axis and the iterator's shape
            // checks — maximum / minimum / fmax / fmin have no text loop, so that resolution is the answer.
            ReductionArguments.ValidateParsedArguments(@out, @where);
            bool[] flags = ReductionArguments.AxisFlags(a.ndim, axis);
            if (Strings.NDStringUfuncs.IsString(a) || Strings.NDStringUfuncs.IsString(@out))
                throw Strings.NDStringUfuncs.NoLoop(funcname, (@out ?? a).dtype, a.dtype);
            ReductionArguments.ValidateOperands(a, flags, keepdims, @out, @where, funcname);

            // Python's where=True is no mask at all (the reduction skips the masked loop), while np.True_ is a mask — one
            // that needs initial=. C# spells both as a 0-d bool array (a `true` literal converts to one), so a 0-d true mask
            // reads as where=True: with initial= the two give the same result (a 0-d mask is one run per call), and only
            // NumPy's "no identity" error for np.True_ without initial= is not reproduced ([Misaligned]).
            if (@where is not null && @where.ndim == 0 && @where.GetTypeCode == NPTypeCode.Boolean && @where.GetAtIndex<bool>(0))
                @where = null;

            // 2. The loop: the input's dtype, or the promotion of (out, input) — out acts as the reduction's first input.
            NPTypeCode inT = a.GetTypeCode;
            NPTypeCode loop = @out is null ? inT : np.promote_types(@out.GetTypeCode, inT).GetTypeCode();

            // 3. initial packed into the loop dtype (PyArray_Pack), after every shape error.
            byte* seed = stackalloc byte[ReductionInitial.SlotBytes];
            bool hasInitial = ReductionInitial.TryPack(initial, loop, new Span<byte>(seed, ReductionInitial.SlotBytes));

            // 4. No identity: a mask needs initial, and an empty reduced axis has nothing to start from.
            if (!hasInitial)
            {
                if (@where is not null)
                    throw new ValueError(
                        $"reduction operation '{funcname}' does not have an identity, so to use a where mask one has to specify 'initial'");
                var dims = a.Shape.dimensions;
                for (int d = 0; d < flags.Length; d++)
                    if (flags[d] && dims[d] == 0)
                        throw new ValueError($"zero-size array to reduction operation {funcname} which has no identity");
            }

            bool flat = Array.TrueForAll(flags, f => f);
            int ax = flat ? -1 : Array.IndexOf(flags, true);

            // 4b. A broadcast input: an element met again along a stride-0 axis changes nothing the loop can see in the
            // cases MaxMinBroadcastShortcut proves, which then reduce one copy instead of every repeat.
            if (@where is null && (@out is null || @out.GetTypeCode == loop) && a.Shape.IsBroadcasted)
            {
                var shortcut = MaxMinBroadcastShortcut(a, flags, axis, keepdims, @out, initial, loop, op);
                if (shortcut is not null)
                    return shortcut;
            }

            // 4c. An integer input under a wider loop (an int32 array into an int64 out, any integer into a float out): the
            // loop dtype comes from a safe cast, which keeps order, and an integer value has one bit pattern, so the loop's
            // extreme IS the cast of the input's extreme — reduced by the fast path in the input's own dtype, then copied
            // (cast) into the out. With initial= or where= the iterator runs (the initial lives in the loop dtype).
            if (!hasInitial && @where is null && @out is not null && @out.GetTypeCode == loop && inT != loop
                && NumPyMinMaxReduce.Supports(inT) && !NumPyMinMaxReduce.IsFloat(inT) && inT != NPTypeCode.Half)
            {
                var narrow = TryMaxMinSeeded(a, flat, ax, keepdims, null, op);
                if (narrow is not null)
                {
                    try
                    {
                        np.copyto(@out, narrow, casting: "unsafe");
                    }
                    finally
                    {
                        narrow.Dispose();
                    }

                    return @out;
                }
            }

            // 5a. The fast path: the seeded exact engines (see the file header for what they may serve).
            if (@where is null && (@out is null || (@out.GetTypeCode == loop && inT == loop)))
            {
                var fast = TryMaxMinSeeded(a, flat, ax, keepdims, hasInitial ? seed : null, op);
                if (fast is not null)
                {
                    if (@out is null)
                        return fast.MarkReductionScalar();
                    try
                    {
                        np.copyto(@out, fast);
                    }
                    finally
                    {
                        fast.Dispose();
                    }

                    return @out;
                }
            }

            // 5b. The general path: NumPy's iterator over [out, input, where].
            return MaxMinThroughIterator(a, flags, keepdims, @out, @where, loop, hasInitial ? seed : null, op);
        }

        /// <summary>
        ///     The FAST path of <see cref="MaxMinNumPy"/>: the exact engines with the initial value as their seed, for the
        ///     calls whose answer cannot depend on the iterator's call cuts (see the file header).
        /// </summary>
        /// <param name="a">The array (the loop dtype is its own).</param>
        /// <param name="flat">True when every axis is reduced.</param>
        /// <param name="ax">The reduced axis when not <paramref name="flat"/>.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="seed">The packed initial value, or <c>null</c>.</param>
        /// <param name="op">The reduction.</param>
        /// <returns>A fresh result (not yet marked as a scalar), or <c>null</c> when the fast path declines.</returns>
        private static unsafe NDArray TryMaxMinSeeded(NDArray a, bool flat, int ax, bool keepdims, byte* seed, MinMaxOp op)
        {
            var t = a.GetTypeCode;
            var shape = a.Shape;
            var engine = a.TensorEngine as DefaultEngine;
            if (engine is null || shape.IsBroadcasted)
                return null;

            if (NumPyMinMaxReduce.Supports(t) && (!NumPyMinMaxReduce.IsFloat(t) || op is MinMaxOp.Max or MinMaxOp.Min))
            {
                NDArray r;
                if (flat)
                {
                    r = engine.TryExactFlatMinMaxScalar(a, op, seed);
                    if (r is not null && keepdims && shape.NDim > 0)
                        r.Storage.Reshape(Shape.Adopt(OnesDims(shape.NDim)));
                }
                else
                {
                    r = engine.TryExactAxisMinMax(a, ax, op, seed);
                    if (r is not null && keepdims)
                        r.Storage.ExpandDimension(ax);
                }

                return r;
            }

            // float16: the sequential HALF loops, walked SIMD by NumPyHalfMinMaxReduce; the initial value is the first element
            // of every output's sequence, and the rule is an associative selection, so it folds in after.
            if (t == NPTypeCode.Half && shape.size > 0 && shape.NDim > 0)
            {
                bool isMax = NumPyMinMaxReduce.IsMaxLike(op);
                bool nanIgnore = op is MinMaxOp.FMax or MinMaxOp.FMin;
                var r = engine.HalfMinMaxReduce(a, flat ? null : ax, keepdims, isMax, nanIgnore);
                if (seed != null)
                {
                    // The fresh result is dense in its own buffer (any order): fold the seed into every element.
                    Half s = *(Half*)seed;
                    Half* p = (Half*)((byte*)r.Address + r.Shape.offset * sizeof(Half));
                    long n = r.size;
                    for (long i = 0; i < n; i++)
                    {
                        p[i] = op switch
                        {
                            MinMaxOp.Max => HalfMaxRule.Fold(s, p[i]),
                            MinMaxOp.Min => HalfMinRule.Fold(s, p[i]),
                            MinMaxOp.FMax => HalfFMaxRule.Fold(s, p[i]),
                            _ => HalfFMinRule.Fold(s, p[i]),
                        };
                    }
                }

                // A 0-d result comes back already marked as NumPy's (read-only) scalar; the caller marks it again or copies it
                // into the out, so it is returned as it is.
                return r;
            }

            return null;
        }

        /// <summary>
        ///     The broadcast-input shortcut of <see cref="MaxMinNumPy"/>: the reduction computed from ONE copy of each
        ///     stride-0 axis's repeated elements, when that is provably NumPy's answer; otherwise <c>null</c>.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     Every loop but the float32 / float64 SIMD ones is a fold whose answer does not depend on how often an element
        ///     recurs: an integer or bool extreme is order-free, and the sequential loops (float16, complex, char, decimal)
        ///     keep the FIRST of tied elements, while the copies of one element are tied and identical. So a reduced stride-0
        ///     axis is cut to its one element, and a NON-reduced stride-0 axis is reduced once and its result repeated (every
        ///     output along it folds the same sequence).
        ///     </para>
        ///     <para>
        ///     float32 / float64 fold through NumPy's SIMD calls, whose horizontal reduce re-canonicalizes a NaN and whose
        ///     vector section decides a ±0 tie, so a reduce-form call over repeated elements can differ from one over a
        ///     single copy. Their reduced stride-0 axes are cut only when every call NumPy makes is ELEMENTWISE (the out moves
        ///     along the innermost iteration axis): each element then meets <c>o = op(o, x)</c> once per copy, every op of
        ///     these loops is idempotent (<c>op(op(o, x), x) == op(o, x)</c>, bits included), and the calls along the other
        ///     axes stay as they were (a stride-0 axis never votes in NpyIter's axis order).
        ///     </para>
        ///     <para>
        ///     The caller has ruled out where= (the mask can differ between the copies) and a cast out (its round trip through
        ///     the out dtype at every buffer fill is not idempotent).
        ///     </para>
        /// </remarks>
        /// <param name="a">The broadcast input.</param>
        /// <param name="flags">The reduced-axis flags.</param>
        /// <param name="axis">The axis argument, passed on unchanged.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="out">The out (its dtype is the loop's), or <c>null</c>.</param>
        /// <param name="initial">The initial value, passed on unchanged.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="op">The reduction.</param>
        /// <returns>The result (<paramref name="out"/> when given), or <c>null</c> when the shortcut does not apply.</returns>
        private static unsafe NDArray MaxMinBroadcastShortcut(NDArray a, bool[] flags, int? axis, bool keepdims, NDArray @out,
            object initial, NPTypeCode loop, MinMaxOp op)
        {
            if (loop is NPTypeCode.Single or NPTypeCode.Double)
            {
                var probe = NumPyReduceIterator.Build(a.Shape, flags, keepdims, @out?.Shape, null, false, a.GetTypeCode != loop,
                    np.getbufsize());
                if (probe.IterSize == 0 || probe.InnerStride[NumPyReduceIterator.OpOut] == 0)
                    return null;
                using var cut = CutStrideZeroAxes(a, flags, reducedOnly: true, out _);
                return cut is null ? null : MaxMinNumPy(cut, axis, @out, keepdims, initial, null, op);
            }

            using var one = CutStrideZeroAxes(a, flags, reducedOnly: false, out bool spread);
            if (one is null)
                return null;
            if (!spread)
                return MaxMinNumPy(one, axis, @out, keepdims, initial, null, op);

            // The spread route pays a fixed cost (a temporary, a second iterator setup, a broadcast copy: ~0.7 us measured);
            // below MinSpreadElements logical elements the iterator over the broadcast input is the faster exact route
            // (a (10, 100) int32 input broadcast along axis 0, reduced along axis 1: 0.81 us through the iterator, 1.5 us
            // spread; at (100, 1000) 3.4 us against 1.5 us).
            if (a.size < MinSpreadElements)
                return null;

            // A non-reduced stride-0 axis was cut too: reduce the copy into a temporary of the loop dtype (so the loop is
            // the same one), then repeat it along the cut axes into the out, or into a fresh result laid out as the
            // iterator lays out the original's (npyiter_new_temp_array).
            var partDims = new long[flags.Length];
            int pd = 0;
            for (int d = 0; d < flags.Length; d++)
                if (!flags[d] || keepdims)
                    partDims[pd++] = flags[d] ? 1 : one.Shape.dimensions[d];
            using var part = new NDArray(loop, pd == 0 ? Shape.Scalar : Shape.Adopt(partDims[..pd]), false);
            MaxMinNumPy(one, axis, part, keepdims, initial, null, op);
            if (@out is not null)
            {
                np.copyto(@out, part);
                return @out;
            }

            var it = NumPyReduceIterator.Build(a.Shape, flags, keepdims, null, null, false, a.GetTypeCode != loop, np.getbufsize());
            var result = new NDArray(loop, it.ResultDims.Length == 0 ? Shape.Scalar : Shape.Adopt(it.ResultDims, it.ResultStrides), false);
            np.copyto(result, part);
            return result;
        }

        /// <summary>The logical element count from which <see cref="MaxMinBroadcastShortcut"/>'s spread route (a NON-reduced
        ///     stride-0 axis) beats the iterator: the work it saves must outweigh its fixed cost.</summary>
        private const long MinSpreadElements = 16384;

        /// <summary>
        ///     A view of <paramref name="a"/> with its stride-0 axes of extent &gt; 1 cut to extent 1 (the one element each
        ///     repeats) — only the reduced ones when <paramref name="reducedOnly"/> — or <c>null</c> when there is none.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="flags">The reduced-axis flags.</param>
        /// <param name="reducedOnly">Cut only reduced axes.</param>
        /// <param name="spread">Receives true when a NON-reduced axis was cut (its result must then be repeated).</param>
        /// <returns>The view (the caller disposes it), or <c>null</c>.</returns>
        private static NDArray CutStrideZeroAxes(NDArray a, bool[] flags, bool reducedOnly, out bool spread)
        {
            spread = false;
            var dims = a.Shape.dimensions;
            var strides = a.Shape.strides;
            Slice[] cut = null;
            for (int d = 0; d < dims.Length; d++)
            {
                if (strides[d] != 0 || dims[d] <= 1 || (reducedOnly && !flags[d]))
                    continue;
                if (cut is null)
                {
                    cut = new Slice[dims.Length];
                    Array.Fill(cut, Slice.All);
                }

                cut[d] = new Slice(0, 1);
                spread |= !flags[d];
            }

            return cut is null ? null : a[cut];
        }

        /// <summary>
        ///     True when <paramref name="axis"/> of <paramref name="shape"/> is NpyIter's innermost axis — extent &gt; 1 and
        ///     the smallest |stride| among the extent &gt; 1 axes — so that NumPy reduces it with one reduce-form call per
        ///     output (a row). Used to pick the faster of two exact routes; an out-of-range axis answers false (the caller's
        ///     own route raises).
        /// </summary>
        /// <param name="shape">The input's shape.</param>
        /// <param name="axis">The reduced axis (negative counts from the end).</param>
        /// <returns>True for the innermost axis.</returns>
        internal static bool ReducedAxisIsInnermost(Shape shape, int axis)
        {
            int nd = shape.NDim;
            if (axis < -nd || axis >= nd)
                return false;
            if (axis < 0)
                axis += nd;
            var dims = shape.dimensions;
            var strides = shape.strides;
            if (dims[axis] <= 1)
                return false;
            long s = Math.Abs(strides[axis]);
            for (int d = 0; d < nd; d++)
                if (d != axis && dims[d] > 1 && Math.Abs(strides[d]) < s)
                    return false;
            return true;
        }

        /// <summary>
        ///     The GENERAL path of <see cref="MaxMinNumPy"/>: the result started from the initial value (or each output's first
        ///     element), then NumPy's iterator over <c>[out, input, where]</c> folding the rest in (<see cref="NumPyReduceIterator"/>).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="flags">The reduced-axis flags.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="out">The out, or <c>null</c>.</param>
        /// <param name="where">The mask, or <c>null</c>.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="seed">The packed initial value, or <c>null</c>.</param>
        /// <param name="op">The reduction.</param>
        /// <returns><paramref name="out"/>, or the fresh result.</returns>
        private static unsafe NDArray MaxMinThroughIterator(NDArray a, bool[] flags, bool keepdims, NDArray @out, NDArray @where,
            NPTypeCode loop, byte* seed, MinMaxOp op)
        {
            // COPY_IF_OVERLAP: an input or mask that may share memory with the out is read from a copy, as NumPy's iterator
            // does (the out is written while they are still read).
            NDArray input = a, mask = @where;
            NDArray inputCopy = null, maskCopy = null;
            NDArray result = null;
            try
            {
                if (@out is not null)
                {
                    if (NDMemOverlap.SolveMayShareMemory(@out, a, 0) != MemOverlap.No)
                        input = inputCopy = a.copy('K');
                    if (mask is not null && NDMemOverlap.SolveMayShareMemory(@out, mask, 0) != MemOverlap.No)
                        mask = maskCopy = mask.copy('K');
                }

                NPTypeCode inT = input.GetTypeCode;
                bool outCast = @out is not null && @out.GetTypeCode != loop;
                var it = NumPyReduceIterator.Build(input.Shape, flags, keepdims, @out?.Shape, mask?.Shape,
                    outCast, inT != loop, np.getbufsize());

                // The result: the out, or a fresh array in the loop dtype laid out in the iteration order
                // (npyiter_new_temp_array).
                result = @out ?? new NDArray(loop, it.ResultDims.Length == 0 ? Shape.Scalar : Shape.Adopt(it.ResultDims, it.ResultStrides), false);
                long skip = 0;
                if (seed != null)
                {
                    // raw_array_assign_scalar: the initial value, cast from the loop dtype into the result's.
                    using var init = new NDArray(loop, Shape.Scalar, false);
                    Buffer.MemoryCopy(seed, (byte*)init.Address, loop.SizeOf(), loop.SizeOf());
                    np.copyto(result, init, casting: "unsafe");
                }
                else if (result.size > 0)
                {
                    // PyArray_CopyInitialReduceValues: each output's first element, cast from the INPUT's dtype into the
                    // result's (the original input — NumPy copies from the operand, not from an overlap copy).
                    var first = new Slice[flags.Length];
                    for (int d = 0; d < flags.Length; d++)
                        first[d] = flags[d] ? (keepdims ? new Slice(0, 1) : Slice.Index(0)) : Slice.All;
                    using var view = flags.Length == 0 ? a.view() : a[first];
                    np.copyto(result, view, casting: "unsafe");
                    skip = result.size;
                }

                RunMaxMin(it, result, input, mask, loop, skip, op);
                if (@out is not null)
                    return @out;
                var r = result;
                result = null;   // handed to the caller: not disposed below
                return r.ndim == 0 ? r.MarkReductionScalar() : r;
            }
            finally
            {
                if (@out is null)
                    result?.Dispose();
                inputCopy?.Dispose();
                maskCopy?.Dispose();
            }
        }

        /// <summary>
        ///     Runs the iterator with the loop dtype's NumPy loop: the SIMD rules for float32 / float64 / integers, the
        ///     logical OR / AND loops for bool, the sequential rules for float16 / complex / char / decimal.
        /// </summary>
        /// <param name="it">The iterator.</param>
        /// <param name="result">The result (the out, or the fresh array).</param>
        /// <param name="input">The input (a copy when it overlapped the out).</param>
        /// <param name="mask">The mask, or <c>null</c>.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="skip">NumPy's skip_first_count.</param>
        /// <param name="op">The reduction.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is not one of the 15 dtypes.</exception>
        private static unsafe void RunMaxMin(NumPyReduceIterator it, NDArray result, NDArray input, NDArray mask, NPTypeCode loop,
            long skip, MinMaxOp op)
        {
            byte* o = (byte*)result.Address + result.Shape.offset * result.dtypesize;
            byte* x = (byte*)input.Address + input.Shape.offset * input.dtypesize;
            byte* m = mask is null ? null : (byte*)mask.Address + mask.Shape.offset;
            NPTypeCode oT = result.GetTypeCode, xT = input.GetTypeCode;
            bool isMax = NumPyMinMaxReduce.IsMaxLike(op);
            switch (loop)
            {
                case NPTypeCode.Double:
                    switch (op)
                    {
                        case MinMaxOp.Max: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<double, NumPyMinMaxReduce.MaxLane<double>>()); break;
                        case MinMaxOp.Min: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<double, NumPyMinMaxReduce.MinLane<double>>()); break;
                        case MinMaxOp.FMax: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<double, NumPyMinMaxReduce.FMaxLane<double>>()); break;
                        default: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<double, NumPyMinMaxReduce.FMinLane<double>>()); break;
                    }

                    break;
                case NPTypeCode.Single:
                    switch (op)
                    {
                        case MinMaxOp.Max: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<float, NumPyMinMaxReduce.MaxLane<float>>()); break;
                        case MinMaxOp.Min: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<float, NumPyMinMaxReduce.MinLane<float>>()); break;
                        case MinMaxOp.FMax: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<float, NumPyMinMaxReduce.FMaxLane<float>>()); break;
                        default: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Lane<float, NumPyMinMaxReduce.FMinLane<float>>()); break;
                    }

                    break;
                // Integers: the NaN-suppressing twin of max IS max (no NaN, no bit-distinct tie).
                case NPTypeCode.SByte: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<sbyte, NumPyMinMaxReduce.MaxLane<sbyte>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<sbyte, NumPyMinMaxReduce.MinLane<sbyte>>()); break;
                case NPTypeCode.Byte: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<byte, NumPyMinMaxReduce.MaxLane<byte>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<byte, NumPyMinMaxReduce.MinLane<byte>>()); break;
                case NPTypeCode.Int16: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<short, NumPyMinMaxReduce.MaxLane<short>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<short, NumPyMinMaxReduce.MinLane<short>>()); break;
                case NPTypeCode.UInt16: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<ushort, NumPyMinMaxReduce.MaxLane<ushort>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<ushort, NumPyMinMaxReduce.MinLane<ushort>>()); break;
                case NPTypeCode.Int32: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<int, NumPyMinMaxReduce.MaxLane<int>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<int, NumPyMinMaxReduce.MinLane<int>>()); break;
                case NPTypeCode.UInt32: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<uint, NumPyMinMaxReduce.MaxLane<uint>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<uint, NumPyMinMaxReduce.MinLane<uint>>()); break;
                case NPTypeCode.Int64: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<long, NumPyMinMaxReduce.MaxLane<long>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<long, NumPyMinMaxReduce.MinLane<long>>()); break;
                case NPTypeCode.UInt64: if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<ulong, NumPyMinMaxReduce.MaxLane<ulong>>()); else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.IntLane<ulong, NumPyMinMaxReduce.MinLane<ulong>>()); break;
                case NPTypeCode.Half:
                    switch (op)
                    {
                        case MinMaxOp.Max: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<Half, HalfMaxRule>()); break;
                        case MinMaxOp.Min: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<Half, HalfMinRule>()); break;
                        case MinMaxOp.FMax: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<Half, HalfFMaxRule>()); break;
                        default: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<Half, HalfFMinRule>()); break;
                    }

                    break;
                case NPTypeCode.Complex:
                    switch (op)
                    {
                        case MinMaxOp.Max: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.ComplexLoop<ComplexMaxRule>()); break;
                        case MinMaxOp.Min: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.ComplexLoop<ComplexMinRule>()); break;
                        case MinMaxOp.FMax: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.ComplexLoop<ComplexFMaxRule>()); break;
                        default: it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.ComplexLoop<ComplexFMinRule>()); break;
                    }

                    break;
                case NPTypeCode.Boolean:
                    // BOOL_maximum / BOOL_fmax ARE logical_or, minimum / fmin logical_and (NumPy aliases them): the
                    // early-exit reduce and the SIMD elementwise op.
                    it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Bool(isMax));
                    break;
                case NPTypeCode.Char:
                    if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<char, CharMaxRule>());
                    else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<char, CharMinRule>());
                    break;
                case NPTypeCode.Decimal:
                    if (isMax) it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<decimal, DecimalMaxRule>());
                    else it.Run(o, oT, x, xT, m, loop, skip, NumPyMinMaxLoops.Seq<decimal, DecimalMinRule>());
                    break;
                default:
                    throw new NotSupportedException($"np.max / np.min: dtype {loop} is not supported.");
            }
        }

        /// <summary>A rank-<paramref name="n"/> shape of ones (a flat reduction's keepdims shape).</summary>
        /// <param name="n">The rank.</param>
        /// <returns>The dims.</returns>
        private static long[] OnesDims(int n)
        {
            var d = new long[n];
            Array.Fill(d, 1L);
            return d;
        }
    }
}
