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
    //   * `d` (from diff) is a STRONG array carrying x's dtype, promoted
    //     against `y[1:]+y[:-1]` the normal way.
    //   * a 1-D `y` reduces to a 0-D scalar (NumPy returns a Python float there).
    //
    // The elementwise half `d*(y[1:]+y[:-1])/2.0` is produced in ONE fused pass
    // via np.evaluate (no intermediate temporaries — an N-D x's difference is part
    // of the same pass, computed from its two slices rather than materialized by
    // np.diff) and then reduced by np.sum, whose pairwise summation is
    // bit-identical to NumPy's `.sum(axis)` for the same reduction layout. The pass
    // types every node by NEP50 the way NumPy's separate calls are typed, so integer
    // and bool nodes wrap at their own dtype exactly as NumPy's materialized
    // temporaries do. Fusing avoids the full-size temporaries NumPy allocates, so
    // trapezoid is several times faster than NumPy at 100K+ elements while staying
    // byte-for-byte equal. Should the fused kernel ever reject an operand set (or a
    // Char operand, which has no NumPy dtype), a plain-operator composition is the
    // fallback. There is no hand-written per-element loop: every loop lives in
    // np.evaluate / np.sum.
    //
    // THE AXIS goes to each statement RAW, as in NumPy: `diff(x, axis)` normalizes it against
    // x.ndim and `.sum(axis)` against the rank of the product `d * (…)`, which an N-D `x` of
    // another rank than `y` changes (y (5,) with x (3, 2, 5) integrates x's last axis and sums
    // the product's last axis, giving (3, 2)). Only the slices `y[1:]`/`y[:-1]` and the 1-D
    // spacing's reshape read it against y.ndim. NumPy resolves the spacing first, so an error in
    // `x` (a 0-d x, an axis outside x) is reported before one in y's axis.
    //
    // LAYOUT-ORDER PARITY (bit-exact on every layout, all dtypes). trapezoid ends in a reduction,
    // and floating-point summation is not associative, so the result depends on the ORDER the
    // axis is traversed: NumPy's `.sum` adds a reduced axis that is innermost in memory pairwise
    // (float16: in float32, narrowing once) and an outer one sequentially (float16: narrowing at
    // every step). Which fires is decided by the MEMORY LAYOUT of the reduction input
    // `d*(y1+y2)/2`, and NumPy builds that input through ufunc calls whose outputs NpyIter lays
    // out from their operands: `y1 + y2`, the subtract inside `diff(x, axis)` for an N-D x, then
    // `d * s`, then `/ 2.0`, which keeps its operand's layout. NumSharp computes that layout with
    // Shape.NpyIterOutputShape, NpyIter's rule ported line for line, composed over the same calls
    // (TrapezoidHalfLayout). Under that rule an operand ABSTAINS on any axis where it has extent 1
    // OR stride 0 (a broadcast axis), and a later operand can only cancel a swap, never start one
    // (C order wins conflicts). A reduction walks memory in order, so NumPy's sum over that
    // layout is bit-identical to a sum over the C-contiguous array of the same memory with the
    // axes transposed into memory order. So when the layout is not C, every operand view is built
    // transposed into that order (TrapezoidOperandView — the slice, the rank padding and the
    // transpose in one view), the fused pass writes C order at memory speed (no copy into the
    // layout, no extra pass), the C half is summed along the axis's transposed position, and the
    // result is relabelled into the caller's axis order (NDArray.AdoptLayout), which is also the
    // layout NumPy's reduction gives its result. np.sum only ever sees C input, or the flat
    // route's gapped view of it (TrapezoidIntegrateFlat, for a dense y integrated along its
    // innermost or outermost memory axis).
    //
    // History of the layout: before 4898b40a every `half` was C-ordered (a 1-ULP divergence on
    // F/transposed/permuted y). 4898b40a laid it out with concatenate's vote
    // (PyArray_CreateMultiSortedStridePerm), which abstains only on extent-1 axes, and let an
    // N-D x vote with the RESULT's dims instead of its own — so a broadcast y's stride-0 axis
    // voted as the innermost one, an x with an extent-1 axis cast a vote NumPy's x does not, and
    // a C-contiguous y skipped the vote entirely though an F-ordered x outvotes it wherever y has
    // extent 1. NpyIter's rule closes all three.
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
        ///     axes); an N-D <paramref name="x"/> is differenced along its own
        ///     <paramref name="axis"/> (counted against <c>x.ndim</c>, as NumPy's
        ///     <c>diff(x, axis=axis)</c> does) and broadcast against <paramref name="y"/>.
        ///     Integration follows <paramref name="x"/>'s order — the points are NOT
        ///     sorted, so a decreasing <paramref name="x"/> integrates in reverse
        ///     (negative area).
        /// </param>
        /// <param name="dx">
        ///     The spacing between sample points when <paramref name="x"/> is
        ///     <c>null</c>. Default 1. Ignored when <paramref name="x"/> is supplied.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to integrate; default is the last axis.
        ///     Negative axes count from the end. It is read against each array of the
        ///     composition the way NumPy reads it: <paramref name="y"/>'s slices use
        ///     <c>y.ndim</c>, the N-D <paramref name="x"/>'s difference uses <c>x.ndim</c>, and
        ///     the final sum uses the rank of the broadcast product (which an N-D
        ///     <paramref name="x"/> with more dimensions than <paramref name="y"/> raises).
        /// </param>
        /// <returns>
        ///     Definite integral approximated along <paramref name="axis"/>. If
        ///     <paramref name="y"/> is 1-D (and no higher-rank <paramref name="x"/> broadcasts
        ///     it) the result is a 0-D array (a scalar); otherwise it is the broadcast product
        ///     <c>d * (y[1:] + y[:-1])</c> with the integration axis removed, laid out as NumPy's
        ///     reduction lays it out. The dtype follows NEP50 promotion of the composition and is
        ///     always in the float family (float16/float32/float64/complex128, or the
        ///     NumSharp-only Decimal); integer and boolean inputs yield float64. The values are
        ///     bit-identical to NumPy's for every memory layout of <paramref name="y"/> and
        ///     <paramref name="x"/>, with one open exception: a <c>-0.0</c> area over a single interval
        ///     comes back <c>-0.0</c> where NumPy's sum gives <c>+0.0</c> (the length-1-axis sum,
        ///     pinned by <c>OpenBugsReductionTests.Trapezoid_OneInterval_NegativeZeroArea_IsPositiveZero</c>).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="y"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="y"/> is 0-D, or <paramref name="axis"/> is outside
        ///     <c>[-y.ndim, y.ndim-1]</c> (NumPy raises an <c>IndexError</c> from its slice list
        ///     there, an open parity gap pinned by
        ///     <c>OpenBugsReductionTests.Trapezoid_AxisOutsideY_RaisesNumPysIndexError</c>; both reject).
        ///     Also, as <see cref="diff(NDArray, int, int, object, object)"/> raises it, when an N-D
        ///     <paramref name="x"/> has no such axis — that check runs first, as in NumPy.
        /// </exception>
        /// <exception cref="ArgumentException">
        ///     A 0-D <paramref name="x"/> ("diff requires input that is at least one dimensional",
        ///     <see cref="diff(NDArray, int, int, object, object)"/>'s error).
        /// </exception>
        /// <exception cref="IncorrectShapeException">
        ///     The spacing does not broadcast against <c>y[1:] + y[:-1]</c> — NumPy's
        ///     <c>operands could not be broadcast together with shapes … </c> from the
        ///     <c>d * (…)</c> multiply, verbatim.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.trapezoid.html</remarks>
        [NDScoped]
        public static NDArray trapezoid(NDArray y, NDArray x = null, double dx = 1.0, int axis = -1)
        {
            if (y is null) throw new ArgumentNullException(nameof(y));

            int nd = y.ndim;
            var ops = new TrapezoidOperands();
            NDArray dd = null, d = null;
            try
            {
                // The spacing first, in NumPy's statement order: an error in `x` is reported before
                // one in y's axis. dLayout is the layout NumPy's `d` has (not NumSharp's np.diff
                // result's), the second voter of the final multiply; null for the scalar dx.
                Shape? dLayout = null;
                Shape x1 = default, x2 = default;
                int ax;
                if (x is not null && x.ndim == 1)
                {
                    // d = diff(x) reshaped to broadcast along `axis`: shape is all-1 except
                    // d.shape[0] on `axis`. NumPy's `shape[axis] = …` assignment is where an axis
                    // outside y fails, so y's axis is checked here, after the diff. The difference
                    // has one element per sample, so it is cheap to materialize.
                    dd = np.diff(x);
                    ax = TrapezoidAxis(axis, nd);
                    long[] shp = new long[nd];
                    for (int i = 0; i < nd; i++) shp[i] = 1;
                    shp[ax] = dd.shape[0];
                    d = dd.reshape(new Shape(shp));     // view sharing dd's storage
                    // At most one axis of extent > 1, so it never takes part in a vote: any
                    // shape with these dims lays the product out the same way.
                    dLayout = d.Shape;
                }
                else if (x is not null)
                {
                    // d = diff(x, axis=axis) for an N-D (or 0-D) x, with the RAW axis counted against
                    // x.ndim. It is NOT materialized: its two slices x[1:] and x[:-1] join the fused pass
                    // (subtract, or not_equal for a bool x, as NumPy's diff picks), which saves a full
                    // pass and an x-sized temporary. DiffAxis raises exactly what np.diff raises (a 0-D
                    // x, an axis outside x) — before y's axis is looked at, as in NumPy.
                    (x1, x2) = TrapezoidSlices(x.Shape, DiffAxis(x, axis));
                    dLayout = Shape.NpyIterOutputShape((long[])x1.dimensions.Clone(), new[] { x1, x2 });
                    ax = TrapezoidAxis(axis, nd);
                }
                else
                {
                    ax = TrapezoidAxis(axis, nd);
                }

                // y[1:] and y[:-1] along `axis` (as shapes: the views are built once the layout is known).
                var (y1, y2) = TrapezoidSlices(y.Shape, ax);

                // NumPy's layout of its reduction input (see the header's LAYOUT-ORDER PARITY).
                // `.sum(axis)` normalizes the raw axis against the PRODUCT's rank, which an N-D x of
                // higher rank than y raises; `axis` was already checked against y.ndim, which never
                // exceeds that rank, so it is in range.
                Shape layout = TrapezoidHalfLayout(y1, y2, dLayout);
                int pnd = layout.NDim;
                int sumAxis = axis < 0 ? axis + pnd : axis;
                int[] perm = layout.DenseAxisOrder();

                // Uniform spacing over a y that is dense in that layout, integrated along its innermost
                // or outermost memory axis: the slices are one contiguous run each.
                if (dLayout is null && nd >= 2 && TrapezoidFlatGate(y.Shape, ax, perm))
                    return TrapezoidIntegrateFlat(y, ax, dx, layout, perm);

                // Otherwise every operand view in the layout's memory order (or as sliced, for a C layout).
                int[] viewPerm = layout.IsContiguous || layout.size == 0 ? null : perm;
                ops.Y1 = TrapezoidOperandView(y, y1, pnd, viewPerm);
                ops.Y2 = TrapezoidOperandView(y, y2, pnd, viewPerm);
                if (d is not null)
                    ops.D = TrapezoidOperandView(d, d.Shape, pnd, viewPerm);
                else if (x is not null)
                {
                    ops.X1 = TrapezoidOperandView(x, x1, pnd, viewPerm);
                    ops.X2 = TrapezoidOperandView(x, x2, pnd, viewPerm);
                }

                return TrapezoidIntegrate(ops, dx, layout, viewPerm, sumAxis);
            }
            finally
            {
                // View-before-owner: the operand views (over y, x and d) are released before d,
                // and d (a reshape view of dd) before dd — unless the reshape handed back dd itself
                // (a 1-D y, whose shape is the difference's own), which must be released once.
                ops.Dispose();
                d?.Dispose();
                if (!ReferenceEquals(d, dd)) dd?.Dispose();
            }
        }

        /// <summary>
        ///     The array operands of trapezoid's fused pass <c>d * (y1 + y2) / 2.0</c>, owned together so
        ///     every exit path releases them in one place. The spacing is either <see cref="D"/> (a
        ///     materialized difference), or the two slices <see cref="X1"/>/<see cref="X2"/> whose difference
        ///     the pass computes itself, or neither (the scalar <c>dx</c>).
        /// </summary>
        private struct TrapezoidOperands
        {
            /// <summary><c>y[1:]</c> along the integration axis.</summary>
            public NDArray Y1;

            /// <summary><c>y[:-1]</c> along the integration axis.</summary>
            public NDArray Y2;

            /// <summary>A 1-D x's difference, reshaped to broadcast along the axis; null otherwise.</summary>
            public NDArray D;

            /// <summary>An N-D x's <c>x[1:]</c> along its own axis; null otherwise.</summary>
            public NDArray X1;

            /// <summary>An N-D x's <c>x[:-1]</c> along its own axis; null otherwise.</summary>
            public NDArray X2;

            /// <summary>
            ///     Releases every operand view this set holds. Each view is created for the set and owned by it
            ///     alone, so the holder that built the set disposes it exactly once.
            /// </summary>
            public void Dispose()
            {
                D?.Dispose();
                X1?.Dispose();
                X2?.Dispose();
                Y1?.Dispose();
                Y2?.Dispose();
            }
        }

        /// <summary>
        ///     Normalizes <paramref name="axis"/> against <paramref name="nd"/> — the check NumPy's
        ///     trapezoid makes implicitly when it indexes its per-axis slice list (and the 1-D spacing's
        ///     shape list) of length <c>y.ndim</c> with the raw axis.
        /// </summary>
        /// <param name="axis">The caller's axis; negative counts from the end.</param>
        /// <param name="nd">The rank of <c>y</c>.</param>
        /// <returns>The axis in <c>[0, nd)</c>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     <paramref name="axis"/> is outside <c>[-nd, nd-1]</c>, including every axis of a 0-D
        ///     <c>y</c>. NumPy leaks a Python "list assignment index out of range" <c>IndexError</c> here;
        ///     NumSharp raises the axis-out-of-bounds error instead (both reject). That is an open parity gap:
        ///     the house rule reproduces NumPy's leaked <c>IndexError</c>s verbatim (<c>np.tensordot</c>,
        ///     <c>np.fft</c>), and <c>OpenBugsReductionTests.Trapezoid_AxisOutsideY_RaisesNumPysIndexError</c>
        ///     pins NumPy's type and message for every route that reaches here.
        /// </exception>
        private static int TrapezoidAxis(int axis, int nd)
        {
            int ax = axis < 0 ? axis + nd : axis;
            if (ax < 0 || ax >= nd)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {nd}");
            return ax;
        }

        /// <summary>
        ///     The shapes of <c>a[1:]</c> and <c>a[:-1]</c> along <paramref name="axis"/> — the two views
        ///     <c>a[slice(1, None)]</c> / <c>a[slice(None, -1)]</c> NumPy indexes, as shapes over
        ///     <paramref name="s"/>'s buffer. Only shapes are built here: the layout vote reads shapes, and the
        ///     views themselves are created once the layout decides their axis order
        ///     (<see cref="TrapezoidOperandView"/>).
        /// </summary>
        /// <param name="s">The array's shape (any layout, broadcast and negative strides included).</param>
        /// <param name="axis">The axis to slice, normalized.</param>
        /// <returns>
        ///     <c>(a[1:], a[:-1])</c>: both have <c>max(len - 1, 0)</c> on <paramref name="axis"/> and
        ///     <paramref name="s"/>'s strides; the first starts one step along the axis.
        /// </returns>
        /// <remarks>
        ///     A length-0 axis gives two empty slices at <paramref name="s"/>'s own offset (Python's
        ///     <c>[1:]</c> of an empty axis starts at its clamped stop, 0). Each shape gets its own dims and
        ///     strides arrays: a shape is immutable only as long as the arrays it holds are.
        /// </remarks>
        private static (Shape Next, Shape Prev) TrapezoidSlices(Shape s, int axis)
        {
            long len = s.dimensions[axis];
            long m = len > 0 ? len - 1 : 0;
            var dims1 = (long[])s.dimensions.Clone();
            dims1[axis] = m;
            var dims2 = (long[])dims1.Clone();
            long start = len - m;                    // 1 for a non-empty axis, 0 for an empty one
            return (new Shape(dims1, (long[])s.strides.Clone(), s.offset + start * s.strides[axis], s.bufferSize),
                    new Shape(dims2, (long[])s.strides.Clone(), s.offset, s.bufferSize));
        }

        /// <summary>
        ///     One operand view of the fused pass, built in a single step: <paramref name="view"/> (a slice
        ///     shape over <paramref name="source"/>'s buffer) as is, or — when <paramref name="perm"/> is given —
        ///     padded to rank <paramref name="nd"/> with leading unit axes and transposed into the layout's
        ///     memory order.
        /// </summary>
        /// <param name="source">The array whose buffer the view reads (y, x or the 1-D difference).</param>
        /// <param name="view">The view's shape over that buffer (offset and strides included).</param>
        /// <param name="nd">The product's rank (≥ the view's).</param>
        /// <param name="perm">
        ///     The layout's axes in memory order, outermost first, or <c>null</c> for a C layout (the view as is;
        ///     the pass broadcasts lower ranks itself).
        /// </param>
        /// <returns>A new view the caller owns.</returns>
        /// <remarks>
        ///     <para>Padding is where broadcasting puts missing axes (leading); a padded axis has extent 1 and
        ///     stride 0, so it addresses nothing. The transpose permutes dims and strides; the offset and buffer
        ///     extent are the slice's. This is the composition <c>np.transpose(AtLeastNdView(slice), perm)</c>
        ///     without the two intermediate arrays (a view costs ~100 ns, the bulk of a 1K call's
        ///     overhead).</para>
        /// </remarks>
        private static NDArray TrapezoidOperandView(NDArray source, Shape view, int nd, int[] perm)
        {
            if (perm is null)
                return new NDArray(source.Storage, view);

            int pad = nd - view.NDim;
            var dims = new long[nd];
            var strides = new long[nd];
            for (int i = 0; i < nd; i++)
            {
                int a = perm[i];
                if (a < pad)
                {
                    dims[i] = 1;
                    strides[i] = 0;
                }
                else
                {
                    dims[i] = view.dimensions[a - pad];
                    strides[i] = view.strides[a - pad];
                }
            }

            return new NDArray(source.Storage, new Shape(dims, strides, view.offset, view.bufferSize));
        }

        /// <summary>
        ///     The memory layout NumPy gives trapezoid's reduction input <c>d * (y1 + y2) / 2.0</c> — the
        ///     layout that decides how <c>.sum(axis)</c> traverses it, and so the bits of the result.
        ///     Composes <see cref="Shape.NpyIterOutputShape"/> over the ufunc calls that build it: the
        ///     <c>add</c> of the two slices, then the <c>multiply</c> of the spacing with that sum. The
        ///     final <c>/ 2.0</c> has one array operand, which it lays its output out like, so it leaves
        ///     the layout where the multiply put it.
        /// </summary>
        /// <param name="y1"><c>y[1:]</c> along the axis.</param>
        /// <param name="y2"><c>y[:-1]</c> along the axis (same dims and strides as <paramref name="y1"/>).</param>
        /// <param name="dLayout">
        ///     NumPy's layout of the spacing array <c>d</c> — the reshaped 1-D difference, or the output layout
        ///     of the subtract an N-D x's diff runs over <c>x[1:]</c>/<c>x[:-1]</c> — or <c>null</c> for the
        ///     scalar <c>dx</c>.
        /// </param>
        /// <returns>A fresh dense shape of the product's (broadcast) dims.</returns>
        /// <exception cref="IncorrectShapeException">
        ///     The spacing does not broadcast against <c>y1 + y2</c>: NumPy's
        ///     <c>operands could not be broadcast together with shapes (d) (s) </c> from the multiply,
        ///     verbatim (both shapes in NumPy's tuple notation, trailing space included).
        /// </exception>
        /// <remarks>
        ///     <para><b>Why NpyIter's rule and not KEEPORDER or concatenate's vote.</b> An operand
        ///     abstains from ordering a pair of axes when either axis has extent 1 OR stride 0 in it, so
        ///     a broadcast <c>y</c> casts no vote along its broadcast axis and a broadcast spacing (a 1-D
        ///     <c>x</c> reshaped, or an N-D <c>x</c> with extent-1 axes) votes only on the axes it really
        ///     has. With conflicting operands C order wins. A 0-D operand (the scalar <c>dx</c>, the
        ///     literal <c>2.0</c>) never votes. NumPy's trivial-loop fast path, which lays out an output
        ///     of same-shaped C- or F-contiguous operands as plain C or F, agrees with this rule on every
        ///     axis of extent &gt; 1, the only axes a reduction's traversal reads.</para>
        ///     <para><b>Cost.</b> Two or three small shape computations per call (the vote tables live on
        ///     the stack); nothing scales with the data.</para>
        /// </remarks>
        private static Shape TrapezoidHalfLayout(Shape y1, Shape y2, Shape? dLayout)
        {
            // s = add(y[1:], y[:-1]): both slices vote (identically — same strides).
            Shape s = Shape.NpyIterOutputShape((long[])y1.dimensions.Clone(), new[] { y1, y2 });
            if (dLayout is not { } d)
                return s;                       // multiply(dx, s): the 0-D dx never votes

            // p = multiply(d, s): d broadcasts against s (NumPy's broadcast error comes from this call).
            var (pShape, _) = DefaultEngine.Broadcast(d, s);
            return Shape.NpyIterOutputShape((long[])pShape.dimensions.Clone(), new[] { d, s });
        }

        /// <summary>
        ///     Computes trapezoid's reduction input <c>d * (y1 + y2) / 2.0</c> with one fused pass and sums it
        ///     along <paramref name="axis"/> exactly as NumPy's <c>.sum(axis)</c> sums the same values laid out in
        ///     <paramref name="layout"/> (<see cref="TrapezoidHalfLayout"/>), returning the result in the layout
        ///     NumPy's reduction gives it.
        /// </summary>
        /// <param name="ops">
        ///     The pass's operand views (borrowed: the caller keeps and releases them), already transposed into
        ///     the layout's memory order when <paramref name="perm"/> is given.
        /// </param>
        /// <param name="dx">The uniform spacing, used when <paramref name="ops"/> carries no spacing array.</param>
        /// <param name="layout">NumPy's layout of the reduction input (dense, of the broadcast product's dims).</param>
        /// <param name="perm">
        ///     <paramref name="layout"/>'s axes in memory order, outermost first, when the operands were transposed
        ///     into it; <c>null</c> for a C (or empty) layout, whose operands are as sliced.
        /// </param>
        /// <param name="axis">The reduced axis, normalized against <paramref name="layout"/>'s rank.</param>
        /// <returns>The integral: a fresh array the caller owns (a read-only 0-D scalar for a 1-D product).</returns>
        /// <remarks>
        ///     <para><b>A C layout</b> is what the fused pass writes, so the half is summed as it comes.</para>
        ///     <para><b>Any other layout runs transposed.</b> NumPy's reduction over an array laid out in
        ///     <paramref name="layout"/> walks its memory in order, which is exactly how it walks the C-contiguous
        ///     array of the same memory with the axes transposed into memory order (<see cref="Shape.DenseAxisOrder"/>),
        ///     so the two sums are bit-identical. The operands are transposed into that order (they share its
        ///     memory order, so the pass streams; writing the permuted layout directly would make NDIter gather
        ///     every element in logical order, and copying into it would cost a second pass), and the C half the
        ///     pass produces is summed along the axis's transposed position. NumPy lays the sum out along the
        ///     remaining axes in that same memory order, which is what the C result of the transposed sum already
        ///     holds, so it is relabelled into the caller's axis order (<see cref="NDArray.AdoptLayout"/>,
        ///     <see cref="TrapezoidResultLayout"/>) — no data moves. The values never depend on the layout, only the
        ///     summation order and the result's strides do.</para>
        ///     <para>An empty product has no layout to honour (a reduction over it reads nothing, and the zeros
        ///     it returns are C on both sides), so it takes the C path.</para>
        /// </remarks>
        private static NDArray TrapezoidIntegrate(in TrapezoidOperands ops, double dx, Shape layout, int[] perm, int axis)
        {
            NDArray half = TrapezoidCOrder(TrapComputeHalf(ops, dx));
            try
            {
                if (perm is null)
                    return np.sum(half, axis);
                NDArray r = np.sum(half, Array.IndexOf(perm, axis));
                // A 1-D (or 0-D) sum has no axis order to restore.
                return layout.NDim <= 2 ? r : NDArray.AdoptLayout(r, TrapezoidResultLayout(layout, perm, axis));
            }
            finally
            {
                half.Dispose();
            }
        }

        /// <summary>
        ///     Gate of <see cref="TrapezoidIntegrateFlat"/>: <paramref name="ys"/> is dense in the layout's memory
        ///     order (its strides, read along <paramref name="perm"/>, are C strides from its offset — no broadcast,
        ///     no gap, no negative stride), and <paramref name="ax"/> is its innermost or its outermost axis of
        ///     extent &gt; 1, with at least two samples on it.
        /// </summary>
        /// <param name="ys">The integrand's shape (rank ≥ 2).</param>
        /// <param name="ax">The integration axis, normalized.</param>
        /// <param name="perm">The reduction input's axes in memory order, outermost first (<see cref="Shape.DenseAxisOrder"/>).</param>
        /// <returns>True when the flat route applies.</returns>
        /// <remarks>
        ///     <para>Extent-1 axes are skipped: their stride never addresses anything, and NpyIter does not order
        ///     them. An empty y fails here (the general route handles empties).</para>
        ///     <para><b>Why only the two ends.</b> The route computes one element per sample of the run, including
        ///     the elements that straddle two rows, which nobody reads. Along the innermost axis they are one per
        ///     row (a 1/len overhead) and the route saves NDIter a chunk per row — the case that costs the general
        ///     route most. Along the outermost axis there are none at all. Along a middle axis they are a whole
        ///     slab per outer index, while the general route's slices already coalesce into long chunks.</para>
        /// </remarks>
        private static bool TrapezoidFlatGate(Shape ys, int ax, int[] perm)
        {
            if (ys.size == 0 || ys.dimensions[ax] < 2)
                return false;
            long expect = 1;
            int innermost = -1, outermost = -1;
            for (int i = perm.Length - 1; i >= 0; i--)
            {
                int a = perm[i];
                long d = ys.dimensions[a];
                if (d == 1)
                    continue;
                if (ys.strides[a] != expect)
                    return false;
                expect *= d;
                if (innermost < 0)
                    innermost = a;
                outermost = a;
            }

            return ax == innermost || ax == outermost;
        }

        /// <summary>
        ///     The uniform-spacing route for a <paramref name="y"/> that is dense in NumPy's layout and integrated
        ///     along its innermost or outermost memory axis (<see cref="TrapezoidFlatGate"/>): computes
        ///     <c>(y[k+s] + y[k]) * dx / 2.0</c> for every <c>k</c> of y's memory run in ONE contiguous 1-D pass
        ///     (<c>s</c> = the axis's stride), then sums it through a view that skips the elements NumPy's half does
        ///     not have.
        /// </summary>
        /// <param name="y">The integrand (passes the gate).</param>
        /// <param name="ax">The integration axis, normalized.</param>
        /// <param name="dx">The uniform spacing.</param>
        /// <param name="layout">NumPy's layout of the reduction input.</param>
        /// <param name="perm"><paramref name="layout"/>'s axes in memory order, outermost first.</param>
        /// <returns>The integral, in the layout NumPy's reduction gives it (the caller owns it).</returns>
        /// <remarks>
        ///     <para><b>Why it is NumPy's half.</b> y occupies <c>n</c> consecutive elements from its offset, and
        ///     consecutive samples along <paramref name="ax"/> are <c>s</c> apart. So <c>y[…, j+1, …]</c> and
        ///     <c>y[…, j, …]</c> are run elements <c>k+s</c> and <c>k</c> with <c>k = Σ i·stride</c>, and the pass's
        ///     element <c>k</c> is NumPy's half at <c>[…, j, …]</c>: the same three operations on the same two
        ///     inputs. The view over the pass's result keeps y's strides (in memory order) with
        ///     <paramref name="ax"/> one shorter, which is exactly the elements NumPy's half holds; the elements
        ///     at a row's end (a <c>k</c> whose next sample starts another row — innermost axis only) are computed
        ///     and never read.</para>
        ///     <para><b>Why it is NumPy's sum.</b> The view puts every axis in memory order, so the sum walks it as
        ///     NumPy walks its dense half: an innermost reduced axis contiguous in both (the pairwise — float16:
        ///     float32-accumulated — per-row sums), an outermost one the sequential slab sum, element for element.
        ///     The result is relabelled into the caller's axis order like the general route's.</para>
        ///     <para><b>Why it exists.</b> Along the innermost axis the general route reads <c>y[1:]</c>/<c>y[:-1]</c>
        ///     as rows of <c>len-1</c> elements with a gap, which costs NDIter a chunk per row, and float16 has no
        ///     SIMD body for such rows; one contiguous run is one SIMD stream (1K float16 ~2.6×, float64 ~2×).
        ///     Along the outermost axis it saves the transposed route's operand views.</para>
        /// </remarks>
        private static NDArray TrapezoidIntegrateFlat(NDArray y, int ax, double dx, Shape layout, int[] perm)
        {
            int nd = y.ndim;
            Shape ys = y.Shape;
            long n = ys.size;
            long step = ys.strides[ax];
            var run = new TrapezoidOperands();
            NDArray hFlat = null, halfT = null;
            try
            {
                // y's memory as one run, sliced like y[1:] and y[:-1] along a 1-D axis of stride `step`.
                run.Y1 = new NDArray(y.Storage, new Shape(new[] { n - step }, new[] { 1L }, ys.offset + step, ys.bufferSize));
                run.Y2 = new NDArray(y.Storage, new Shape(new[] { n - step }, new[] { 1L }, ys.offset, ys.bufferSize));
                hFlat = TrapezoidCOrder(TrapComputeHalf(run, dx));

                // NumPy's half, read through the run in memory order: y's strides, ax one sample shorter.
                var tdims = new long[nd];
                var tstrides = new long[nd];
                for (int i = 0; i < nd; i++)
                {
                    int a = perm[i];
                    tdims[i] = a == ax ? ys.dimensions[a] - 1 : ys.dimensions[a];
                    tstrides[i] = ys.strides[a];
                }

                Shape hs = hFlat.Shape;
                halfT = new NDArray(hFlat.Storage, new Shape(tdims, tstrides, hs.offset, hs.bufferSize));
                NDArray r = np.sum(halfT, Array.IndexOf(perm, ax));
                // A 1-D sum has no axis order to restore, and a C layout's is already C.
                return nd <= 2 || layout.IsContiguous ? r : NDArray.AdoptLayout(r, TrapezoidResultLayout(layout, perm, ax));
            }
            finally
            {
                halfT?.Dispose();
                hFlat?.Dispose();
                run.Dispose();
            }
        }

        /// <summary>
        ///     Ensures the half about to be summed is C-ordered, the layout the caller computed the summation
        ///     order for. The fused pass always allocates C; the plain-operator fallback inherits the layout
        ///     rules of the operators, so this is checked rather than trusted.
        /// </summary>
        /// <param name="half">A fresh half (consumed: returned as is, or disposed after its C copy is made).</param>
        /// <returns>A C-contiguous array holding the same values (the caller owns it).</returns>
        private static NDArray TrapezoidCOrder(NDArray half)
        {
            if (half.Shape.IsContiguous)
                return half;
            try
            {
                return np.ascontiguousarray(half);
            }
            finally
            {
                half.Dispose();
            }
        }

        /// <summary>
        ///     The layout NumPy's <c>.sum(axis)</c> gives its result when its input is laid out in
        ///     <paramref name="layout"/>: dense along the input's remaining axes in the input's memory order (the
        ///     reduction iterator allocates its output along its own axis order, with the reduced axis removed).
        /// </summary>
        /// <param name="layout">The reduction input's layout (rank ≥ 3).</param>
        /// <param name="perm">That layout's axes in memory order, outermost first (<see cref="Shape.DenseAxisOrder"/>).</param>
        /// <param name="axis">The reduced axis.</param>
        /// <returns>A fresh dense shape of the result's dims (the input's, <paramref name="axis"/> removed).</returns>
        private static Shape TrapezoidResultLayout(Shape layout, int[] perm, int axis)
        {
            int nd = perm.Length;
            var dims = new long[nd - 1];
            for (int i = 0, j = 0; i < nd; i++)
                if (i != axis)
                    dims[j++] = layout.dimensions[i];

            // The remaining axes in memory order, renumbered to the result's axes (the reduced one is gone).
            var rperm = new int[nd - 1];
            for (int i = 0, j = 0; i < nd; i++)
            {
                int a = perm[i];
                if (a != axis)
                    rperm[j++] = a < axis ? a : a - 1;
            }

            return new Shape(dims, Shape.StridesForPerm(dims, rperm));
        }

        /// <summary>
        ///     Computes the trapezoidal element <c>d * (y1 + y2) / 2.0</c> as a single fused pass over the
        ///     operands via <see cref="evaluate"/> (no intermediate temporaries): <c>d</c> is the materialized
        ///     1-D difference, or an N-D x's difference computed in the same pass from its two slices
        ///     (<c>x1 - x2</c>, or <c>x1 != x2</c> for a bool x — NumPy's diff), or the scalar
        ///     <paramref name="dx"/> (a weak NEP50 literal, matching NumPy). Falls back to a bit-identical
        ///     plain-operator composition if the fused kernel rejects the operand set.
        /// </summary>
        /// <param name="ops">
        ///     The operands (borrowed, never released here). <c>Y1</c>/<c>Y2</c> are always present; at most
        ///     one of <c>D</c> and the <c>X1</c>/<c>X2</c> pair is.
        /// </param>
        /// <param name="dx">The uniform spacing, used when <paramref name="ops"/> carries no spacing array.</param>
        /// <returns>A fresh array of the broadcast operands' dims, C-contiguous on the fused path (the caller owns it).</returns>
        /// <remarks>
        ///     <para><b>Integer and bool inputs ride the fused pass too.</b> NumPy computes <c>y[1:] + y[:-1]</c>,
        ///     an integer difference and an integer product with MODULAR wrap at the operands' own dtype before
        ///     the float promotion (uint8 200 + 180 → 124, not 380), and a bool sum as a logical or. The fused
        ///     kernel types every node by NEP50 the way NumPy's separate calls are typed, so each integer node
        ///     wraps at its own dtype (probed: uint8 <c>[200,180,150,240]</c> → <c>[62, 37, 67]</c>, NumPy's
        ///     bytes). The pass is ~4× cheaper at 1K than three materialized temporaries.</para>
        ///     <para><b>Char stays on the composition.</b> It has no NumPy dtype, so its arithmetic is whatever
        ///     NumSharp's operators define; the fused kernel is not asked to reproduce that.</para>
        /// </remarks>
        private static NDArray TrapComputeHalf(in TrapezoidOperands ops, double dx)
        {
            bool boolDiff = ops.X1 is not null && ops.X1.GetTypeCode == NPTypeCode.Boolean;
            bool charOperand = ops.Y1.GetTypeCode == NPTypeCode.Char
                               || (ops.D is not null && ops.D.GetTypeCode == NPTypeCode.Char)
                               || (ops.X1 is not null && ops.X1.GetTypeCode == NPTypeCode.Char);
            if (!charOperand)
            {
                try
                {
                    NDExpr inner = NDExpr.Arr(ops.Y1) + NDExpr.Arr(ops.Y2);
                    NDExpr spacing = ops.D is not null ? NDExpr.Arr(ops.D)
                        : ops.X1 is null ? null
                        : boolDiff ? NDExpr.NotEqual(NDExpr.Arr(ops.X1), NDExpr.Arr(ops.X2))
                        : NDExpr.Arr(ops.X1) - NDExpr.Arr(ops.X2);
                    NDExpr expr = spacing is null ? inner * dx / 2.0 : spacing * inner / 2.0;
                    return np.evaluate(expr);
                }
                catch (NotSupportedException) { /* fall through to the composition */ }
            }

            // Char (and the fused-kernel fallback): the same ops in NumPy's order, each materialised at its
            // own dtype, so an integer add/subtract/multiply wraps there.
            NDArray diffArr = null, innerArr = null, scaled = null;
            try
            {
                NDArray d = ops.D;
                if (d is null && ops.X1 is not null)
                    d = diffArr = boolDiff ? (NDArray)(ops.X1 != ops.X2) : ops.X1 - ops.X2;
                innerArr = ops.Y1 + ops.Y2;
                scaled = d is null ? innerArr * dx : d * innerArr;
                return scaled / 2.0;
            }
            finally
            {
                scaled?.Dispose();
                innerArr?.Dispose();
                diffArr?.Dispose();
            }
        }
    }
}
