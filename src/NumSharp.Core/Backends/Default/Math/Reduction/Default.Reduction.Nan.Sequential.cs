using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;

// =============================================================================
// Default.Reduction.Nan.Sequential.cs — np.nanmax / np.nanmin for float16 and complex128
// (plan docs/plans/ndexpr-evaluate.md, "Perf review 2026-09-23", the nanmax / nanmin lever)
// =============================================================================
//
// On an ndarray np.nanmax IS np.fmax.reduce (and nanmin fmin.reduce). For float16 and complex NumPy has no SIMD loop:
// HALF_fmax / CDOUBLE_fmax (numpy/_core/src/umath/loops.c.src) are plain BINARY_LOOPs,
//
//     HALF:    out = (npy_half_ge(in1, in2) || isnan(in2)) ? in1 : in2          (fmin: npy_half_le)
//     COMPLEX: out = (isnan(in2.re) || isnan(in2.im) || CGE(in1, in2)) ? in1 : in2
//              CGE(x, y) = (x.re > y.re && !isnan(x.im) && !isnan(y.im)) || (x.re == y.re && x.im >= y.im)
//
// so a reduction is a SEQUENTIAL fold whose answer is: skip NaN elements (for complex, an element with a NaN in either
// part), take the maximum (complex: lexicographic), a TIE keeps the EARLIER element (±0 compare equal, so the first zero
// wins; for complex a (±0, ±0) pair), and an all-NaN reduction returns its FIRST NaN verbatim (payload and sign). Which
// element is "earlier" is the reduction iterator's visiting order:
//
//   * AXIS: every output folds its reduced elements in increasing logical index (NpyIter is built with
//     NPY_ITER_DONT_NEGATE_STRIDES), whatever the memory layout — so the answer is layout-independent and any walk that
//     keeps each output's k order is exact;
//   * FLAT: one fold over NpyIter's iteration order — NumPy's K order: the extent > 1 axes insertion-sorted by |stride|
//     from reversed C order, where a comparison involving a zero stride is AMBIGUOUS and skipped (so a broadcast axis
//     keeps its place), each axis walked in increasing logical index. Coalescing and buffering never reorder a fold.
//
// The engine routes here (DefaultEngine.NanMinMax): complex128 on every layout (it used to be np.amax / np.amin, which
// PROPAGATE a NaN — a value bug, not a bit one), float16 on every layout but the flat C-contiguous one (whose bit-level
// kernel already implements this rule). The float16 kernels this replaces misread non-C layouts (a byte-vs-element
// stride in the flat iterator kernel, logical-vs-physical indices in the axis fallback) and returned the canonical NaN
// for an all-NaN slice.
// =============================================================================

namespace NumSharp.Backends
{
    /// <summary>
    /// One step of a sequential <c>fmax</c> / <c>fmin</c> fold: the running value <c>acc</c> (NumPy's first operand) against
    /// the next element <c>x</c>. The rule decides which of the two survives; it never computes a new value, so the
    /// survivor keeps its exact bits (a NaN its payload, a zero its sign).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    internal interface ISequentialNanRule<T> where T : unmanaged
    {
        /// <summary>The survivor of <c>fmax(acc, x)</c> (or <c>fmin</c>) under NumPy's scalar loop rule.</summary>
        /// <param name="acc">The running value (NumPy's <c>in1</c>).</param>
        /// <param name="x">The next element (NumPy's <c>in2</c>).</param>
        /// <returns><paramref name="acc"/> or <paramref name="x"/>, unchanged.</returns>
        static abstract T Fold(T acc, T x);
    }

    /// <summary>
    /// <c>HALF_fmax</c>: keep the running value when it is <c>&gt;=</c> the next element (IEEE — <c>-0 &gt;= +0</c>, so a
    /// ±0 tie keeps the earlier zero) or when the next element is NaN; a NaN running value gives way to any number.
    /// </summary>
    internal readonly struct HalfFMaxRule : ISequentialNanRule<Half>
    {
        /// <inheritdoc />
        /// <remarks>Compares through <c>float</c> — every <see cref="Half"/> converts exactly, and IEEE order is preserved.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half Fold(Half acc, Half x)
        {
            if (Half.IsNaN(x))
                return acc;
            if (Half.IsNaN(acc))
                return x;
            return (float)acc >= (float)x ? acc : x;
        }
    }

    /// <summary><c>HALF_fmin</c>: the mirror of <see cref="HalfFMaxRule"/> with <c>&lt;=</c>.</summary>
    internal readonly struct HalfFMinRule : ISequentialNanRule<Half>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Half Fold(Half acc, Half x)
        {
            if (Half.IsNaN(x))
                return acc;
            if (Half.IsNaN(acc))
                return x;
            return (float)acc <= (float)x ? acc : x;
        }
    }

    /// <summary>
    /// <c>CDOUBLE_fmax</c>: keep the running value when the next element has a NaN in either part, or when the running
    /// value is lexicographically <c>&gt;=</c> it (<c>CGE</c>: a larger real part with no NaN imaginary part, or an equal
    /// real part and a <c>&gt;=</c> imaginary part); otherwise take the next element — which is how a NaN running value
    /// (every <c>CGE</c> comparison false) gives way to the first number.
    /// </summary>
    internal readonly struct ComplexFMaxRule : ISequentialNanRule<Complex>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Fold(Complex acc, Complex x)
        {
            double xr = x.Real, xi = x.Imaginary, ar = acc.Real, ai = acc.Imaginary;
            if (double.IsNaN(xr) || double.IsNaN(xi))
                return acc;
            bool keep = (ar > xr && !double.IsNaN(ai) && !double.IsNaN(xi)) || (ar == xr && ai >= xi);
            return keep ? acc : x;
        }
    }

    /// <summary><c>CDOUBLE_fmin</c>: the mirror of <see cref="ComplexFMaxRule"/> with <c>CLE</c>.</summary>
    internal readonly struct ComplexFMinRule : ISequentialNanRule<Complex>
    {
        /// <inheritdoc />
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Complex Fold(Complex acc, Complex x)
        {
            double xr = x.Real, xi = x.Imaginary, ar = acc.Real, ai = acc.Imaginary;
            if (double.IsNaN(xr) || double.IsNaN(xi))
                return acc;
            bool keep = (ar < xr && !double.IsNaN(ai) && !double.IsNaN(xi)) || (ar == xr && ai <= xi);
            return keep ? acc : x;
        }
    }

    /// <summary>
    /// The sequential folds themselves — NumPy's visiting order for a flat and an axis reduction of one operand (see the
    /// file header).
    /// </summary>
    internal static class NumPySequentialReduce
    {
        /// <summary>
        /// NpyIter's axis order for a FLAT reduction of one operand, innermost first: the extent &gt; 1 axes insertion-sorted
        /// from reversed C order, an axis moving inward past another only when both strides are non-zero and its |stride|
        /// is smaller (a comparison involving a zero stride is ambiguous and skipped, so a broadcast axis keeps its place;
        /// an equal |stride| keeps the later axis inner) — <c>npyiter_find_best_axis_ordering</c> with the input as the
        /// only voter (the 0-d result's strides are all zero).
        /// </summary>
        /// <param name="dims">The operand's extents.</param>
        /// <param name="strides">The operand's element strides (any sign, zero for a broadcast axis).</param>
        /// <param name="perm">Receives the axes in iteration order, innermost first (length &gt;= dims.Length).</param>
        /// <returns>How many axes were written (the extent &gt; 1 ones; 0 for a one-element operand).</returns>
        internal static int FlatVisitOrder(ReadOnlySpan<long> dims, ReadOnlySpan<long> strides, Span<int> perm)
        {
            int m = 0;
            for (int d = dims.Length - 1; d >= 0; d--)
            {
                if (dims[d] <= 1)
                    continue;   // NpyIter zeroes an extent-1 axis' stride: ambiguous everywhere, then coalesced away

                // Insert d after the reversed-C prefix perm[0..m): scan inward, skipping ambiguous comparisons; the first
                // unambiguous one decides — a strictly larger |stride| inside moves the insertion point past it,
                // anything else stops the scan.
                long s0 = strides[d];
                int ipos = m;
                for (int i1 = m - 1; i1 >= 0; i1--)
                {
                    long s1 = strides[perm[i1]];
                    if (s0 == 0 || s1 == 0)
                        continue;
                    if (Math.Abs(s1) <= Math.Abs(s0))
                        break;
                    ipos = i1;
                }

                for (int i = m; i > ipos; i--)
                    perm[i] = perm[i - 1];
                perm[ipos] = d;
                m++;
            }

            return m;
        }

        /// <summary>
        /// The FLAT fold: the operand's element at iteration position 0 (its logical origin) seeds the running value, and
        /// every further element folds in, in NumPy's iteration order (<see cref="FlatVisitOrder"/>, each axis walked in
        /// increasing logical index).
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold step.</typeparam>
        /// <param name="x">The operand's logical element 0.</param>
        /// <param name="dims">The operand's extents (a non-empty array).</param>
        /// <param name="strides">The operand's element strides.</param>
        /// <returns>The reduction, with the survivor's exact bits.</returns>
        internal static unsafe T Flat<T, TRule>(T* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> strides)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            int nd = dims.Length;
            Span<int> perm = stackalloc int[Math.Max(nd, 1)];
            int m = FlatVisitOrder(dims, strides, perm);
            T acc = x[0];
            if (m == 0)
                return acc;

            // Innermost axis as a tight loop, the rest as an odometer (outer axes perm[1..m), perm[1] fastest).
            long innerN = dims[perm[0]], innerS = strides[perm[0]];
            Span<long> coord = stackalloc long[m];
            coord.Clear();
            long off = 0;
            bool first = true;
            while (true)
            {
                T* p = x + off;
                long i = 0;
                if (first)
                {
                    i = 1;   // position 0 is the seed
                    first = false;
                }

                for (; i < innerN; i++)
                    acc = TRule.Fold(acc, p[i * innerS]);

                int j = 1;
                for (; j < m; j++)
                {
                    int d = perm[j];
                    if (++coord[j] < dims[d])
                    {
                        off += strides[d];
                        break;
                    }

                    off -= (dims[d] - 1) * strides[d];
                    coord[j] = 0;
                }

                if (j == m)
                    return acc;
            }
        }

        /// <summary>
        /// The AXIS fold: <c>o[..] = fold over k = 0 … K-1 of x[.., k, ..]</c> in increasing k for every output — NumPy's
        /// per-output sequence under <c>NPY_ITER_DONT_NEGATE_STRIDES</c>, whatever the layout. The walk only chooses the
        /// memory-friendly loop order: rows (the reduced axis is the tightest) fold one output at a time; otherwise the
        /// first reduced index is copied into every output and each further index folds across all outputs, which keeps
        /// every output's k order too.
        /// </summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <typeparam name="TRule">The fold step.</typeparam>
        /// <param name="x">The operand's logical element 0.</param>
        /// <param name="dims">The operand's extents (a non-empty array, rank &lt;= 64).</param>
        /// <param name="xs">The operand's element strides.</param>
        /// <param name="o">The result's logical element 0.</param>
        /// <param name="os">The result's element strides indexed by INPUT axis (the reduced axis' entry is ignored).</param>
        /// <param name="axis">The reduced axis (extent &gt;= 1).</param>
        internal static unsafe void Axis<T, TRule>(T* x, ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, T* o,
            ReadOnlySpan<long> os, int axis)
            where T : unmanaged where TRule : struct, ISequentialNanRule<T>
        {
            int nd = dims.Length;
            long K = dims[axis], sK = xs[axis];

            // Output walk axes: the non-reduced extent > 1 axes, the tightest input |stride| innermost (locality only).
            Span<int> walk = stackalloc int[Math.Max(nd, 1)];
            int w = 0;
            for (int d = 0; d < nd; d++)
            {
                if (d == axis || dims[d] <= 1)
                    continue;
                long a = Math.Abs(xs[d]);
                int j = w++;
                while (j > 0 && Math.Abs(xs[walk[j - 1]]) < a)
                {
                    walk[j] = walk[j - 1];
                    j--;
                }

                walk[j] = d;
            }

            // Rows: the reduced axis is at least as tight as every walked one — fold each output's k run in place.
            bool rows = w == 0 || Math.Abs(sK) <= Math.Abs(xs[walk[w - 1]]);
            Span<long> coord = stackalloc long[Math.Max(w, 1)];
            if (rows)
            {
                coord.Clear();
                long offX = 0, offO = 0;
                while (true)
                {
                    T* p = x + offX;
                    T acc = p[0];
                    for (long k = 1; k < K; k++)
                        acc = TRule.Fold(acc, p[k * sK]);
                    o[offO] = acc;
                    if (!Advance(dims, xs, os, walk, w, coord, ref offX, ref offO))
                        return;
                }
            }

            // Slabs: copy k = 0, then fold k = 1 … K-1 across every output (each output still sees k in order).
            for (long k = 0; k < K; k++)
            {
                coord.Clear();
                long offX = k * sK, offO = 0;
                while (true)
                {
                    o[offO] = k == 0 ? x[offX] : TRule.Fold(o[offO], x[offX]);
                    if (!Advance(dims, xs, os, walk, w, coord, ref offX, ref offO))
                        break;
                }
            }
        }

        /// <summary>
        /// One odometer step over the output walk axes (the last one fastest), advancing the input and output offsets
        /// together.
        /// </summary>
        /// <param name="dims">The operand's extents.</param>
        /// <param name="xs">Input element strides by input axis.</param>
        /// <param name="os">Output element strides by input axis.</param>
        /// <param name="walk">The walk axes, outermost first.</param>
        /// <param name="w">How many walk axes.</param>
        /// <param name="coord">The odometer (length &gt;= w).</param>
        /// <param name="offX">The input offset (updated).</param>
        /// <param name="offO">The output offset (updated).</param>
        /// <returns>False when the walk is complete (the odometer wrapped).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Advance(ReadOnlySpan<long> dims, ReadOnlySpan<long> xs, ReadOnlySpan<long> os, ReadOnlySpan<int> walk,
            int w, Span<long> coord, ref long offX, ref long offO)
        {
            for (int j = w - 1; j >= 0; j--)
            {
                int d = walk[j];
                if (++coord[j] < dims[d])
                {
                    offX += xs[d];
                    offO += os[d];
                    return true;
                }

                coord[j] = 0;
                offX -= (dims[d] - 1) * xs[d];
                offO -= (dims[d] - 1) * os[d];
            }

            return false;
        }
    }

    public partial class DefaultEngine
    {
        /// <summary>
        /// <c>np.nanmax</c> / <c>np.nanmin</c> for float16 and complex128 as NumPy's sequential <c>fmax</c> / <c>fmin</c>
        /// fold (see the file header): NaN elements skipped, the maximum (complex: lexicographic) kept, a tie keeping the
        /// earlier element in NumPy's visiting order, an all-NaN reduction returning its first NaN verbatim.
        /// </summary>
        /// <remarks>
        /// The caller has already handled the empty reduction (NumPy's "zero-size array" error) and the single-element
        /// input. The result is fresh: a read-only numpy-scalar 0-d for a flat reduction without keepdims (the
        /// <c>PyArray_Return</c> contract <see cref="NDArray.MarkReductionScalar"/> models), else the reduced array in
        /// NumPy's K order (<see cref="AllocateReductionResult"/>), with keepdims applied.
        /// </remarks>
        /// <param name="arr">A non-empty float16 or complex128 array (any layout, broadcast included).</param>
        /// <param name="axis">The axis to reduce (null = all axes; negative counts from the end).</param>
        /// <param name="keepdims">Keep the reduced axes as size 1.</param>
        /// <param name="isMax">fmax when true, fmin when false.</param>
        /// <returns>The reduced array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="NotSupportedException"><paramref name="arr"/> is neither float16 nor complex128, or has rank
        /// above 64 — a caller bug.</exception>
        private unsafe NDArray NanMinMaxSequential(NDArray arr, int? axis, bool keepdims, bool isMax)
        {
            var shape = arr.Shape;
            var tc = arr.GetTypeCode;
            if (tc != NPTypeCode.Half && tc != NPTypeCode.Complex)
                throw new NotSupportedException($"NanMinMaxSequential serves float16 / complex128, got {tc} — caller bug.");
            int nd = shape.NDim;
            if (nd > 64)
                throw new NotSupportedException($"NanMinMaxSequential: rank {nd} exceeds 64.");

            int es = tc.SizeOf();
            // Logical element 0: a contiguous slice re-seats Address (offset 0), a strided view keeps its base with a
            // non-zero offset — Address + offset·itemsize is right for both.
            byte* x = (byte*)arr.Address + shape.offset * es;

            if (axis == null)
            {
                var r = new NDArray(tc, Shape.Scalar, false);
                byte* slot = (byte*)r.Address + r.Shape.offset * es;
                if (tc == NPTypeCode.Half)
                    *(Half*)slot = isMax
                        ? NumPySequentialReduce.Flat<Half, HalfFMaxRule>((Half*)x, shape.dimensions, shape.strides)
                        : NumPySequentialReduce.Flat<Half, HalfFMinRule>((Half*)x, shape.dimensions, shape.strides);
                else
                    *(Complex*)slot = isMax
                        ? NumPySequentialReduce.Flat<Complex, ComplexFMaxRule>((Complex*)x, shape.dimensions, shape.strides)
                        : NumPySequentialReduce.Flat<Complex, ComplexFMinRule>((Complex*)x, shape.dimensions, shape.strides);
                if (keepdims)
                {
                    var ks = new long[nd];
                    for (int i = 0; i < nd; i++)
                        ks[i] = 1;
                    r.Storage.Reshape(new Shape(ks));
                }

                return r.MarkReductionScalar();
            }

            int ax = NormalizeAxis(axis.Value, nd);
            var dims = shape.dimensions;
            var outputDims = new long[nd - 1];
            for (int d = 0, od = 0; d < nd; d++)
                if (d != ax)
                    outputDims[od++] = dims[d];

            var result = AllocateReductionResult(tc, outputDims, shape);
            try
            {
                // Output strides indexed by INPUT axis (the reduced axis' entry is never read).
                Span<long> os = stackalloc long[nd];
                var rs = result.Shape.strides;
                for (int d = 0, od = 0; d < nd; d++)
                    os[d] = d == ax ? 0 : (outputDims.Length == 0 ? 0 : rs[od++]);
                byte* o = (byte*)result.Address + result.Shape.offset * es;
                if (tc == NPTypeCode.Half)
                {
                    if (isMax)
                        NumPySequentialReduce.Axis<Half, HalfFMaxRule>((Half*)x, dims, shape.strides, (Half*)o, os, ax);
                    else
                        NumPySequentialReduce.Axis<Half, HalfFMinRule>((Half*)x, dims, shape.strides, (Half*)o, os, ax);
                }
                else
                {
                    if (isMax)
                        NumPySequentialReduce.Axis<Complex, ComplexFMaxRule>((Complex*)x, dims, shape.strides, (Complex*)o, os, ax);
                    else
                        NumPySequentialReduce.Axis<Complex, ComplexFMinRule>((Complex*)x, dims, shape.strides, (Complex*)o, os, ax);
                }
            }
            catch
            {
                // A failing reduction must not strand the fresh buffer.
                result.Dispose();
                throw;
            }

            if (keepdims)
                result.Storage.ExpandDimension(ax);
            return result.MarkReductionScalar();
        }
    }
}
