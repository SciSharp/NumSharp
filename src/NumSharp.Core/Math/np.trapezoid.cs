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
    // LAYOUT-ORDER PARITY (bit-exact on every layout, all dtypes). trapezoid ends in a
    // reduction, and floating-point summation is not associative, so the result depends
    // on the ORDER the axis is traversed. NumPy's `.sum` is itself layout-order-dependent:
    // when the reduced axis is the CONTIGUOUS (inner) one it sums pairwise — and for
    // float16 accumulates in float32, narrowing once — whereas a strided/outer reduced
    // axis sums sequentially (float16: narrowing at every step). Which one fires is decided
    // by the layout of the reduction input `d*(y1+y2)/2`, and NumPy allocates that in
    // KEEPORDER (its ufunc output layout, matching the operand's stride permutation), so an
    // F-contiguous / transposed / permuted `y` produces a non-C-contiguous intermediate.
    // NumSharp's np.sum already reproduces NumPy's per-layout traversal bit-for-bit, but the
    // fused `half` is C-contiguous, which formerly diverged by one ULP on float16/complex128
    // (and, for cleanly-F/transposed inputs, occasionally float32/float64) whenever the two
    // layouts disagreed on which axis is inner. RelayoutHalfToNumpyKeepOrder now re-lays the
    // float-family `half` into NumPy's exact KEEPORDER layout before the sum (a no-op copy on
    // C-contiguous input — the common fast path — and on cleanly-F operands), so every dtype
    // is bit-identical on every layout. Integer/bool sums are exact regardless of order and
    // are left untouched.
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

                // np.sum is LAYOUT-ORDER-DEPENDENT (a contiguous reduced axis sums
                // pairwise — and, for float16, accumulates in float32 and narrows
                // once — while a strided/outer reduced axis sums sequentially, for
                // float16 narrowing at every step). NumPy allocates its reduction
                // input `d*(y1+y2)/2` in KEEPORDER (the ufunc output layout, matching
                // the operand's stride permutation), so a non-C-contiguous `y` yields
                // a non-C `half` that sums in a different order than a plain C `half`.
                // NumSharp's np.sum already reproduces NumPy's per-layout traversal
                // bit-for-bit (verified across C/F/transposed/permuted), so the only
                // thing needed for exact parity is to give `half` the same KEEPORDER
                // layout NumPy's intermediate has. Only the FLOAT family's summation
                // is order-sensitive — integer/bool sums are exact regardless of
                // traversal (modular addition is associative) — and a C-contiguous
                // `y` already yields a C `half` == KEEPORDER, so both short-circuit
                // with no copy. (`half.typecode` is used rather than `y`'s dtype:
                // an integer `y` with a float coordinate `x` produces a float `half`
                // whose sum IS order-sensitive.)
                NPTypeCode htc = half.GetTypeCode;
                bool halfFloat = htc is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double
                                     or NPTypeCode.Complex or NPTypeCode.Decimal;
                if (halfFloat && !y.Shape.IsContiguous)
                {
                    NDArray relaid = RelayoutHalfToNumpyKeepOrder(half, y.Shape, x);
                    if (!ReferenceEquals(relaid, half))
                    {
                        // A fresh KEEPORDER copy replaced the mis-laid-out half; the
                        // original is now dead and must be released here (the finally
                        // disposes only the final `half`).
                        half.Dispose();
                        half = relaid;
                    }
                }
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

        /// <summary>
        ///     Relays <paramref name="half"/> (the trapezoid reduction input
        ///     <c>d·(y1+y2)/2</c>) into NumPy's <b>KEEPORDER</b> memory layout so the
        ///     layout-order-dependent <see cref="sum(NDArray, int)"/> that follows is
        ///     bit-identical to NumPy. KEEPORDER is a C-contiguous buffer laid out in the
        ///     axis permutation NumPy's ufunc output allocation picks for the final
        ///     <c>d*(y1+y2)</c> multiply — <see cref="NumpyKeepOrderPerm"/>, the port of
        ///     <c>PyArray_CreateMultiSortedStridePerm</c> over that multiply's operands
        ///     (largest |stride| axis outermost, C-order winning ties/conflicts). The
        ///     element VALUES are
        ///     layout-independent (an elementwise expression), so this is a pure strided
        ///     copy that changes only which axis is contiguous — and hence whether the
        ///     later reduction sums that axis pairwise (float16: accumulate in float32,
        ///     narrow once) or sequentially (float16: narrow every step), the sole source
        ///     of the former 1-ULP float16/complex128 divergence on non-C-contiguous input.
        /// </summary>
        /// <param name="half">
        ///     The freshly-computed, owned reduction input. Never mutated; either returned
        ///     unchanged or left for the caller to dispose after a copy is substituted.
        /// </param>
        /// <param name="yShape">
        ///     The shape of the ORIGINAL <c>y</c> operand. Its stride permutation is what the
        ///     <c>y1+y2</c> intermediate inherits (the sliced <c>y1</c>/<c>y2</c> share
        ///     <c>y</c>'s strides, and that intermediate's KEEPORDER strides preserve
        ///     <c>y</c>'s relative stride ordering — so <c>y</c>'s strides are an exact
        ///     comparator proxy for it).
        /// </param>
        /// <param name="x">
        ///     The ORIGINAL coordinate array, or <c>null</c> for uniform spacing. When it is a
        ///     full N-D array (<c>x.ndim == y.ndim</c>) the final multiply's spacing operand
        ///     <c>d = diff(x, axis)</c> inherits <b>x's</b> stride permutation (diff is a
        ///     subtract ufunc, itself KEEPORDER over x's slices), so <c>x</c>'s strides are the
        ///     exact comparator proxy for that second operand — which can shift the combined
        ///     permutation away from <c>y</c>'s. A 1-D <c>x</c>'s <c>d</c> is broadcast (all
        ///     axes but one are size 1) and never votes, so the permutation stays at <c>y</c>'s;
        ///     it is treated like uniform spacing here. (NumSharp's own <c>d.strides</c> are
        ///     deliberately NOT used — its <c>np.diff</c> output layout need not match NumPy's.)
        /// </param>
        /// <returns>
        ///     <paramref name="half"/> itself when it already has the target layout (no copy
        ///     — the C-contiguous common case and cleanly-F-contiguous operands both
        ///     short-circuit), otherwise a fresh owned C-order-in-permutation array holding
        ///     the same values (the caller owns and must dispose the original).
        /// </returns>
        private static NDArray RelayoutHalfToNumpyKeepOrder(NDArray half, Shape yShape, NDArray x)
        {
            int nd = half.ndim;
            if (nd <= 1)
                return half;                       // a single axis has only one traversal order

            long[] hdims = half.Shape.dimensions;

            // KEEPORDER permutation over the FINAL multiply's operands, exactly as NumPy's
            // PyArray_CreateMultiSortedStridePerm combines them. The y1+y2 intermediate
            // votes with half's reduced dims and y's stride ORDER; a full N-D coordinate
            // `x` adds the d = diff(x, axis) operand, voting with the same reduced dims and
            // x's stride order (diff along `axis` leaves x's cross-axis ordering intact).
            // `perm[0]` is the outermost (largest |stride|) axis; a pair is decided only by
            // operands where both axes have dim != 1, and an axis moves outward only if
            // EVERY voting operand agrees (C-order wins ties/conflicts).
            long[][] opDims;
            long[][] opStrides;
            if (x is not null && x.ndim == nd)
            {
                opDims = new[] { hdims, hdims };
                opStrides = new[] { yShape.strides, x.Shape.strides };
            }
            else
            {
                opDims = new[] { hdims };
                opStrides = new[] { yShape.strides };
            }
            int[] perm = NumpyKeepOrderPerm(nd, opDims, opStrides);

            // Target = C-contiguous element strides laid out in `perm` order over half's
            // own (reduced) dims: walk innermost (perm[nd-1]) → outermost, accumulating
            // the running product.
            long[] target = new long[nd];
            long acc = 1;
            for (int k = nd - 1; k >= 0; k--)
            {
                int ax = perm[k];
                target[ax] = acc;
                acc *= hdims[ax];
            }

            // Already in the target layout? (C input → identity perm → C strides == half's;
            // cleanly-F operand → reverse perm → F strides == half's.) Then no copy.
            long[] cur = half.Shape.strides;
            bool same = true;
            for (int i = 0; i < nd; i++)
                if (cur[i] != target[i]) { same = false; break; }
            if (same)
                return half;

            // Materialise a fresh OWNED array carrying the KEEPORDER strides over a
            // contiguous buffer (bufferSize == size), then copy the layout-independent
            // values in through np.copyto's strided assignment. fillZeros:false is safe —
            // copyto writes every element (dst and half share dims → a 1:1 full copy).
            var dims = (long[])hdims.Clone();       // don't alias half's shape array
            var dst = new NDArray(half.dtype, new Shape(dims, target), false);
            np.copyto(dst, half);
            return dst;
        }

        /// <summary>
        ///     Faithful port of NumPy's <c>PyArray_CreateMultiSortedStridePerm</c> — the axis
        ///     ordering a ufunc's KEEPORDER output allocation uses when combining several
        ///     operands. Produces the permutation with the largest-|stride| axis first
        ///     (<c>perm[0]</c> = outermost, C-order convention), by a STABLE insertion sort
        ///     whose default (on ambiguity or conflict) is the identity order — i.e. C-order
        ///     wins ties and disagreements, matching NumPy exactly. This is not a plain sort:
        ///     an axis pair is judged only by operands where BOTH axes have dim != 1 (size-1
        ///     axes, e.g. a broadcast operand's, never vote), and one axis is placed outside
        ///     another only when EVERY voting operand agrees its |stride| is strictly larger.
        /// </summary>
        /// <param name="ndim">The common rank of every operand (and of the returned permutation).</param>
        /// <param name="dims">
        ///     Per-operand dimension arrays (length <paramref name="ndim"/> each); used only
        ///     for the dim != 1 vote-eligibility test.
        /// </param>
        /// <param name="strides">
        ///     Per-operand stride arrays (length <paramref name="ndim"/> each); only the
        ///     relative order of their absolute values matters.
        /// </param>
        /// <returns>
        ///     A length-<paramref name="ndim"/> permutation of axis indices, outermost first.
        /// </returns>
        private static int[] NumpyKeepOrderPerm(int ndim, long[][] dims, long[][] strides)
        {
            int[] perm = new int[ndim];
            for (int i = 0; i < ndim; i++) perm[i] = i;

            // Stable insertion sort, biggest stride to smallest (the reverse of the
            // iterator's Fortran ordering, per numpy's comment). `ambig` stays true until
            // the first operand votes on a pair; after that a later operand can only FORCE
            // no-swap (C-order wins), never re-introduce a swap.
            for (int i0 = 1; i0 < ndim; i0++)
            {
                int ipos = i0;
                int axj0 = perm[i0];
                for (int i1 = i0 - 1; i1 >= 0; i1--)
                {
                    int axj1 = perm[i1];
                    bool ambig = true, shouldswap = false;
                    for (int k = 0; k < dims.Length; k++)
                    {
                        if (dims[k][axj0] != 1 && dims[k][axj1] != 1)
                        {
                            long s0 = strides[k][axj0] < 0 ? -strides[k][axj0] : strides[k][axj0];
                            long s1 = strides[k][axj1] < 0 ? -strides[k][axj1] : strides[k][axj1];
                            if (s0 <= s1) shouldswap = false;          // conflict/tie → C-order
                            else if (ambig) shouldswap = true;         // only while still undecided
                            ambig = false;
                        }
                    }
                    if (!ambig)
                    {
                        if (shouldswap) ipos = i1;
                        else break;
                    }
                }
                if (ipos != i0)
                {
                    for (int i1 = i0; i1 > ipos; i1--) perm[i1] = perm[i1 - 1];
                    perm[ipos] = axj0;
                }
            }
            return perm;
        }
    }
}
