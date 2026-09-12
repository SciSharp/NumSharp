using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    // ============================== np.trapezoid ==============================
    // Integrate along the given axis using the composite trapezoidal rule.
    //
    //   ∫ y dx ≈ Σ_i  d_i · (y[i+1] + y[i]) / 2
    //
    // where d_i is the spacing between consecutive samples: the scalar `dx`
    // (uniform spacing, the default) or the successive differences of `x`.
    //
    // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::trapezoid
    // (added in NumPy 2.0; the old name `np.trapz` was REMOVED in 2.x and now
    //  raises AttributeError, so only `trapezoid` exists here too).
    //
    // NumPy is a pure composition:
    //     d = dx                       if x is None
    //       = diff(x)  (reshaped)      if x is 1-D   (broadcast along `axis`)
    //       = diff(x, axis=axis)       if x is N-D
    //     ret = (d * (y[1:] + y[:-1]) / 2.0).sum(axis)
    //
    // NumSharp mirrors that expression exactly — same operations, same order —
    // so the NEP50 dtype promotion, broadcasting and edge cases all fall out of
    // the existing operators and are bit-identical to NumPy:
    //   * `dx` and the literal `2.0` are WEAK scalars, so e.g. float32 y stays
    //     float32, float16 stays float16, and any integer/bool result is float64.
    //   * `d` (from np.diff) is a STRONG array carrying x's dtype, promoted
    //     against `y[1:]+y[:-1]` the normal way.
    //   * a 1-D `y` reduces to a 0-D scalar (NumPy returns a Python float there).
    //
    // The elementwise half `d*(y[1:]+y[:-1])/2.0` is produced in ONE fused pass
    // via np.evaluate (three ops, no intermediate temporaries) and then reduced
    // by np.sum — whose pairwise summation is bit-identical to NumPy's `.sum(axis)`
    // for a C-contiguous reduction. Fusing avoids the three full-size temporaries
    // NumPy itself allocates, so trapezoid is 2-9x faster than NumPy at 100K+
    // elements while staying byte-for-byte equal. Should the fused kernel ever
    // reject an operand set, a bit-identical plain-operator composition is the
    // fallback. There is no hand-written per-element loop: every loop lives in
    // np.evaluate / np.sum.
    //
    // ACCEPTED 1-ULP divergence, float16 / complex128 ONLY, non-C-contiguous input
    // ONLY (float32/float64/integer are bit-exact on every layout): trapezoid ends
    // in a reduction, and summation is not associative, so the result depends on the
    // ORDER the axis is traversed. NumPy's `.sum` is itself layout-order-dependent —
    // for an F-contiguous `y`, NumPy preserves the F layout of the intermediate
    // `d*(y1+y2)/2` and sums it in F-order, whereas NumSharp's fused half is
    // C-contiguous and sums in C-order. For float16 (widen-compute-narrow pairwise)
    // and complex128 (scalar pairwise) those two orders round apart by a single ULP;
    // for float32/float64 the multi-accumulator SIMD reduction lands on the same bits
    // either way. This is the same class as the documented "order-dependent reduction
    // rounding" — the value is correct to 1 ULP and only the last bit of an f16/c128
    // result on a transposed/Fortran/strided input differs. Matching it would mean
    // reproducing NumPy's intermediate-layout preservation AND its per-layout sum
    // traversal bit-for-bit, abandoning the fused fast path for a non-contractual bit.
    public static partial class np
    {
        /// <summary>
        ///     Integrate <paramref name="y"/> along the given axis using the
        ///     composite trapezoidal rule.
        /// </summary>
        /// <param name="y">Input array to integrate (at least one dimensional).</param>
        /// <param name="x">
        ///     The sample points corresponding to the <paramref name="y"/> values.
        ///     When <c>null</c> (the default) the samples are assumed evenly spaced
        ///     <paramref name="dx"/> apart. A 1-D <paramref name="x"/> supplies the
        ///     spacing along <paramref name="axis"/> (broadcast across the other
        ///     axes); an N-D <paramref name="x"/> is differenced along
        ///     <paramref name="axis"/> directly. Integration follows
        ///     <paramref name="x"/>'s order — the points are NOT sorted, so a
        ///     decreasing <paramref name="x"/> integrates in reverse (negative area).
        /// </param>
        /// <param name="dx">
        ///     The spacing between sample points when <paramref name="x"/> is
        ///     <c>null</c>. Default 1. Ignored when <paramref name="x"/> is supplied.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to integrate; default is the last axis.
        ///     Negative axes count from the end.
        /// </param>
        /// <returns>
        ///     Definite integral approximated along <paramref name="axis"/>. If
        ///     <paramref name="y"/> is 1-D the result is a 0-D array (a scalar);
        ///     otherwise it is <paramref name="y"/> with <paramref name="axis"/>
        ///     removed. The dtype follows NEP50 promotion of the composition and is
        ///     always in the float family (float16/float32/float64/complex128, or
        ///     the NumSharp-only Decimal); integer and boolean inputs yield float64.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="y"/> is 0-D, or <paramref name="axis"/> is outside
        ///     <c>[-y.ndim, y.ndim-1]</c>.
        /// </exception>
        /// <exception cref="ValueError">
        ///     Propagated from <see cref="diff(NDArray, int, int, object, object)"/> when a 0-D
        ///     <paramref name="x"/> is supplied, or from the broadcast when a 1-D <paramref name="x"/>'s
        ///     length is incompatible with <paramref name="axis"/>.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html</remarks>
        [NDScoped]
        public static NDArray trapezoid(NDArray y, NDArray x = null, double dx = 1.0, int axis = -1)
        {
            if (y is null) throw new ArgumentNullException(nameof(y));

            int nd = y.ndim;
            // A 0-D y has no axis to integrate along. NumPy leaks a Python
            // "list assignment index out of range" IndexError here; NumSharp
            // raises the clearer axis-out-of-bounds error (both reject).
            if (nd == 0)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension 0");

            // normalize_axis_index(axis, nd) — valid range is [-nd, nd-1].
            int ax = axis;
            if (ax < 0) ax += nd;
            if (ax < 0 || ax >= nd)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {nd}");

            long len = y.shape[ax];
            long m = len > 0 ? len - 1 : 0;                 // reduced length along axis

            NDArray y1 = null, y2 = null, dd = null, d = null, half = null;
            try
            {
                // y[1:] and y[:-1] along `axis` (overlapping read-only views of y).
                y1 = SliceAlongAxis(y, ax, len - m, len);   // y[1:]
                y2 = SliceAlongAxis(y, ax, 0, m);           // y[:-1]

                // Resolve the spacing `d`. Null => uniform scalar spacing (dx).
                if (x is not null)
                {
                    if (x.ndim == 1)
                    {
                        // d = diff(x) reshaped to broadcast along `axis`:
                        // shape is all-1 except d.shape[0] on `axis`.
                        dd = np.diff(x);
                        long[] shp = new long[nd];
                        for (int i = 0; i < nd; i++) shp[i] = 1;
                        shp[ax] = dd.shape[0];
                        d = dd.reshape(new Shape(shp));     // view sharing dd's storage
                    }
                    else
                    {
                        // d = diff(x, axis=axis); a 0-D x raises inside np.diff
                        // ("diff requires input that is at least one dimensional"),
                        // exactly as NumPy does.
                        d = np.diff(x, axis: ax);
                    }
                }

                // Fused elementwise half = d * (y1 + y2) / 2.0 (one pass), then the
                // pairwise reduction. `half` is always float-family, so np.sum does
                // not widen it and the result dtype equals `half`'s dtype.
                half = TrapComputeHalf(d, y1, y2, dx);
                return np.sum(half, ax);
            }
            finally
            {
                // View-before-owner: `d` (a reshape view of `dd` in the 1-D case,
                // or a fresh diff in the N-D case) is released before `dd`.
                half?.Dispose();
                d?.Dispose();
                if (!ReferenceEquals(d, dd)) dd?.Dispose();
                y1?.Dispose();
                y2?.Dispose();
            }
        }

        /// <summary>
        ///     Computes the trapezoidal element <c>d * (y1 + y2) / 2.0</c> as a
        ///     single fused pass over the operands via <see cref="evaluate"/>
        ///     (no intermediate temporaries). <paramref name="d"/> is <c>null</c>
        ///     for uniform spacing, in which case the scalar <paramref name="dx"/>
        ///     is used (as a weak NEP50 literal, matching NumPy). Falls back to a
        ///     bit-identical plain-operator composition if the fused kernel rejects
        ///     the operand set.
        /// </summary>
        private static NDArray TrapComputeHalf(NDArray d, NDArray y1, NDArray y2, double dx)
        {
            // An INTEGER-family input must compute y[1:]+y[:-1] (and d*(…) for an integer coordinate
            // array) with MODULAR wrap at the operand's own dtype BEFORE the float promotion — exactly
            // as NumPy does (uint8 200+180 → 124, not 380). The single fused evaluate kernel promotes
            // an intermediate integer node to the final float precision, so that overflow would NOT
            // wrap; integers therefore take the plain-operator composition, where each op materialises
            // at its own dtype and wraps correctly. The FLOAT family (Half/Single/Double/Complex/
            // Decimal) cannot overflow that way, so it takes the faster single-pass fused evaluate —
            // bit-identical to the composition (same per-element add/mul/div order).
            NPTypeCode tc = y1.GetTypeCode;
            bool floatFamily = tc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double
                                   or NPTypeCode.Complex or NPTypeCode.Decimal;
            if (floatFamily)
            {
                try
                {
                    NDExpr inner = NDExpr.Arr(y1) + NDExpr.Arr(y2);
                    NDExpr expr = d is null ? inner * dx / 2.0 : NDExpr.Arr(d) * inner / 2.0;
                    return np.evaluate(expr);
                }
                catch (NotSupportedException) { /* fall through to the composition */ }
            }

            // Integer/bool/char (and the fused-kernel fallback): the same three ops in NumPy's order,
            // each materialised at its own dtype so the integer add/mul wraps.
            var innerArr = y1 + y2;
            var scaled = d is null ? innerArr * dx : d * innerArr;
            var res = scaled / 2.0;
            innerArr.Dispose();
            scaled.Dispose();
            return res;
        }
    }
}
