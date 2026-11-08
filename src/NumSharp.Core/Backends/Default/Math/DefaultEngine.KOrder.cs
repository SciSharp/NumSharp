using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     Above this rank NumPy defines no layout (its arrays stop at 64 dimensions), and the exact port of NpyIter's
        ///     axis sort (<see cref="Shape.NpyIterOutputShape"/>, an insertion sort over every axis) would be quadratic in
        ///     the rank, so <see cref="TryKOrderPlan"/> sorts only the axes longer than 1 there.
        /// </summary>
        private const int KOrderExactRankLimit = 64;

        /// <summary>
        ///     Whether an operand can only vote C in NpyIter's output-order sort: a 0-d or one-element operand never votes,
        ///     a C-contiguous one votes C, and so does any operand whose |strides| never rise from one axis of extent above 1
        ///     to the next (a stepped, reversed or broadcast view of C order: NpyIter compares |strides|, a stride-0 axis
        ///     abstains, and an equal pair keeps its order). The hot paths test this per operand before they pay for a
        ///     broadcast and <see cref="TryKOrderPlan"/>: when every operand passes, C is NumPy's answer by construction.
        /// </summary>
        /// <param name="s">The operand's shape.</param>
        /// <returns>True when the operand cannot move the K layout away from C.</returns>
        /// <remarks>O(1) through the cached flags for the common C-contiguous operand; one pass over the axes otherwise (no
        ///     allocation, linear at any rank).</remarks>
        internal static bool IsKOrderCVoter(in Shape s)
        {
            if (s.IsScalar || s.size <= 1 || s.IsContiguous)
                return true;
            var dims = s.dimensions;
            var strides = s.strides;
            long previous = long.MaxValue;
            for (int i = 0; i < dims.Length; i++)
            {
                long stride = strides[i];
                // An extent-1 axis is never stepped and a stride-0 axis abstains: neither can vote against C.
                if (dims[i] <= 1 || stride == 0)
                    continue;
                long magnitude = stride < 0 ? -stride : stride;
                if (magnitude > previous)
                    return false;
                previous = magnitude;
            }
            return true;
        }

        /// <summary>
        ///     The cheap gate of every binary K-order step: whether NumPy's layout for these two operands can differ from C at
        ///     all — a result of at least two axes with an operand that votes against C (<see cref="IsKOrderCVoter"/>). False
        ///     means the route's own C (or all-F) layout is NumPy's, without a broadcast or a plan being computed.
        /// </summary>
        /// <param name="a">The first operand's shape.</param>
        /// <param name="b">The second operand's shape.</param>
        /// <returns>True when <see cref="TryKOrderPlan"/> has to be asked.</returns>
        internal static bool MayNeedKOrder(in Shape a, in Shape b)
            => (a.NDim > 1 || b.NDim > 1) && !(IsKOrderCVoter(a) && IsKOrderCVoter(b));

        /// <summary>
        ///     Whether every operand is strictly F-contiguous or a 0-d / one-element array, with at least one F: the main
        ///     binary and comparison routes lay such a set out F themselves (<see cref="AreAllOperandsStrictFContig"/>), which
        ///     is NpyIter's K layout of it, so their K-order step has nothing to add. The common case is a transposed C
        ///     matrix (<c>a.T</c>) with a scalar or another transpose; skipping the plan keeps it on the direct F route.
        /// </summary>
        /// <param name="a">The first operand's shape.</param>
        /// <param name="b">The second operand's shape.</param>
        /// <returns>True when the route's own F layout already is NumPy's.</returns>
        /// <remarks>Only for routes that produce F for such a set; the others (the float-tier family, divmod, ldexp) take
        ///     the K-order step, whose transposed C pass and relabel cost no copy.</remarks>
        internal static bool IsAllStrictF(in Shape a, in Shape b)
        {
            bool aOne = a.IsScalar || a.size <= 1, bOne = b.IsScalar || b.size <= 1;
            bool aF = !aOne && a.IsFContiguous && !a.IsContiguous;
            bool bF = !bOne && b.IsFContiguous && !b.IsContiguous;
            return (aOne || aF) && (bOne || bF) && (aF || bF);
        }

        /// <summary>
        ///     NumPy's ufunc output layout for a FRESH elementwise result (parity QA-36): NpyIter allocates its output in
        ///     the K order of its operands (<c>npyiter_find_best_axis_ordering</c> + <c>npyiter_new_temp_array</c>), so a
        ///     ufunc over a permuted or F operand keeps that memory order. Answers whether that layout differs from C and,
        ///     when it does, how to produce it with a kernel that can only write C order: transpose every operand by
        ///     <paramref name="perm"/> (<see cref="IntoKOrder"/>), run the kernel, and relabel its C result with
        ///     <see cref="NDArray.AdoptLayout"/>.
        /// </summary>
        /// <param name="dims">The broadcast result dims (read, never kept: the layout gets its own copy).</param>
        /// <param name="inputs">The ufunc's input operands' shapes (0-d ones and NumPy scalars included: they never vote).</param>
        /// <param name="layout">The K layout when the method returns true (dense, offset 0).</param>
        /// <param name="perm">The axis permutation that puts the layout's axes in memory order (outermost first) when the
        ///     method returns true: <c>np.transpose(x, perm)</c> of a dense operand in that order is C-contiguous.</param>
        /// <returns>True when the K layout is NOT C (the caller computes through the plan); false when C is NumPy's answer
        ///     (every operand C-contiguous, a 0-d or one-element operand, at most one axis longer than 1, an empty result,
        ///     or a vote that comes out C — a reversed or stepped C view, a row broadcast).</returns>
        /// <remarks>
        ///     <para>
        ///         The cheap test runs first and decides the hot path: when every input can only vote C
        ///         (<see cref="IsKOrderCVoter"/>: C-contiguous, a 0-d / one-element operand, or a C-ordered stepped / reversed
        ///         / broadcast view) the vote's answer is C by construction, so nothing is allocated.
        ///     </para>
        ///     <para>
        ///         Up to <see cref="KOrderExactRankLimit"/> dimensions the layout is <see cref="Shape.NpyIterOutputShape"/>
        ///         itself (NumPy's sort, ported line for line). Above it the axes of extent 1 — which every operand abstains
        ///         on, and which never address an element — are left out of the sort: at most 63 axes are longer than 1 (their
        ///         product is the element count), so the sort stays tiny at any rank, and the extent-1 axes take strides that
        ///         keep the stride sequence non-increasing, so a later <see cref="Shape.DenseAxisOrder"/> stays linear. The
        ///         relative order of the longer axes is the same either way (an abstaining pair is skipped by the sort).
        ///     </para>
        /// </remarks>
        internal static bool TryKOrderPlan(long[] dims, ReadOnlySpan<Shape> inputs, out Shape layout, out int[] perm)
        {
            layout = default;
            perm = null;
            int nd = dims.Length;
            if (nd < 2)
                return false;

            // The hot path: inputs that can only vote C (C-contiguous, 0-d / one-element, or C-ordered views) give C.
            bool allC = true;
            for (int i = 0; i < inputs.Length && allC; i++)
                allC = IsKOrderCVoter(inputs[i]);
            if (allC)
                return false;

            long size = 1;
            int longAxes = 0;
            for (int d = 0; d < nd; d++)
            {
                size *= dims[d];
                if (dims[d] > 1)
                    longAxes++;
            }
            // An empty result has nothing to lay out, and with at most one long axis C and F are the same layout.
            if (size == 0 || longAxes < 2)
                return false;

            if (nd <= KOrderExactRankLimit)
            {
                layout = Shape.NpyIterOutputShape((long[])dims.Clone(), inputs);
                if (layout.IsContiguous)
                    return false;
                perm = layout.DenseAxisOrder();
                return true;
            }

            return TryKOrderPlanLongAxes(dims, inputs, longAxes, out layout, out perm);
        }

        /// <summary>
        ///     The high-rank half of <see cref="TryKOrderPlan"/>: NpyIter's sort over the axes longer than 1 only, re-expanded
        ///     to the full rank.
        /// </summary>
        /// <param name="dims">The broadcast result dims.</param>
        /// <param name="inputs">The input operands' shapes.</param>
        /// <param name="longAxes">How many of <paramref name="dims"/> are longer than 1 (at least 2).</param>
        /// <param name="layout">The full-rank K layout when the method returns true.</param>
        /// <param name="perm">The extent-1 axes first (in their order; they address nothing), then the long axes in the
        ///     layout's memory order.</param>
        /// <returns>True when the layout is not C.</returns>
        private static bool TryKOrderPlanLongAxes(long[] dims, ReadOnlySpan<Shape> inputs, int longAxes, out Shape layout,
            out int[] perm)
        {
            layout = default;
            perm = null;
            int nd = dims.Length;

            // The long axes, in logical order, and each operand's view of them (right-aligned; an absent or extent-1 axis
            // of an operand gets extent 1 and stride 0: it abstains, as npyiter_fill_axisdata makes it).
            var axisOf = new int[longAxes];
            for (int d = 0, k = 0; d < nd; d++)
                if (dims[d] > 1)
                    axisOf[k++] = d;

            var reducedDims = new long[longAxes];
            for (int k = 0; k < longAxes; k++)
                reducedDims[k] = dims[axisOf[k]];

            var reduced = new Shape[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                var od = inputs[i].dimensions;
                var os = inputs[i].strides;
                int ond = od.Length;
                var rd = new long[longAxes];
                var rs = new long[longAxes];
                for (int k = 0; k < longAxes; k++)
                {
                    int j = axisOf[k] - (nd - ond);
                    bool present = j >= 0 && od[j] > 1;
                    rd[k] = present ? od[j] : 1;
                    rs[k] = present ? os[j] : 0;
                }
                reduced[i] = new Shape(rd, rs, copy: false);
            }

            Shape reducedLayout = Shape.NpyIterOutputShape(reducedDims, reduced);
            if (reducedLayout.IsContiguous)
                return false;

            // Full strides: a long axis keeps its reduced stride; an extent-1 axis takes the stride of the next long axis
            // to its right (1 after the last), so the sequence only rises where the long axes themselves are permuted.
            var strides = new long[nd];
            long next = 1;
            for (int d = nd - 1, k = longAxes - 1; d >= 0; d--)
            {
                if (dims[d] > 1)
                {
                    strides[d] = reducedLayout.strides[k--];
                    next = strides[d];
                }
                else
                    strides[d] = next;
            }
            layout = new Shape((long[])dims.Clone(), strides, copy: false);

            // The extent-1 axes first, then the long axes outermost-first by stride (a dense layout gives every long axis a
            // distinct stride, so the order is exact).
            int[] longOrder = reducedLayout.DenseAxisOrder();
            perm = new int[nd];
            int p = 0;
            for (int d = 0; d < nd; d++)
                if (dims[d] == 1)
                    perm[p++] = d;
            for (int k = 0; k < longAxes; k++)
                perm[p++] = axisOf[longOrder[k]];
            return true;
        }

        /// <summary>
        ///     A cast copy of <paramref name="x"/> laid out as NumPy's ufunc would lay out <paramref name="x"/> itself (its
        ///     <see cref="TryKOrderPlan"/> layout, else C), for the routes that cast an operand up front where NumPy's
        ///     iterator casts it in its buffers: the buffered cast never votes, so the result must not inherit the layout
        ///     astype's K copy would give (a broadcast operand's stride ranking puts its stride-0 axes innermost, an F
        ///     layout NpyIter never picks for it).
        /// </summary>
        /// <param name="x">The operand (any layout).</param>
        /// <param name="to">The dtype to cast to.</param>
        /// <returns>A fresh, dense array of <paramref name="x"/>'s dims in <paramref name="to"/>.</returns>
        /// <remarks>Unsafe casting, as astype: the callers have already decided the cast (bool → the int64 accumulator of
        ///     the scans, a real dtype → np.angle's float64).</remarks>
        internal static NDArray CastInUfuncLayout(NDArray x, NPTypeCode to)
        {
            Shape layout = TryKOrderPlan(x.Shape.dimensions, new[] { x.Shape }, out var kLayout, out _)
                ? kLayout
                : new Shape(x.Shape.dimensions, copy: false);
            var r = new NDArray(to, layout, false);
            try
            {
                UnmanagedStorage.Copyto(r, x, NPY_CASTING.NPY_UNSAFE_CASTING);
            }
            catch
            {
                // The copy failed: the fresh array must not outlive the call.
                r.Dispose();
                throw;
            }
            return r;
        }

        /// <summary>
        ///     Runs a binary route that writes a C-ordered result so that the result comes out in NumPy's ufunc output order
        ///     (parity QA-36): when the operands' K order is not C, <paramref name="compute"/> runs over the operands
        ///     transposed into it and its C result is relabelled; otherwise it runs over the operands as they are.
        /// </summary>
        /// <typeparam name="TState">The route's own arguments (the op, the loop dtype …), passed through so the lambda can be
        ///     <c>static</c> and the hot path allocates no closure.</typeparam>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <param name="state">The route's arguments.</param>
        /// <param name="compute">The route: (state, first, second) → a fresh C-ordered (or, for an all-F pair, F-ordered)
        ///     result of the broadcast shape. It must not return one of its operands.</param>
        /// <returns>The route's result in NumPy's layout.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (raised before the route runs, with the
        ///     broadcast's text, only when the K order has to be computed).</exception>
        internal NDArray KOrderBinary<TState>(NDArray a, NDArray b, TState state, Func<TState, NDArray, NDArray, NDArray> compute)
        {
            if (MayNeedKOrder(a.Shape, b.Shape))
            {
                var (kl, _) = Broadcast(a.Shape, b.Shape);
                if (TryKOrderPlan(kl.dimensions, new[] { a.Shape, b.Shape }, out var layout, out var perm))
                {
                    int nd = kl.NDim;
                    NDArray at = IntoKOrder(a, nd, perm), bt = IntoKOrder(b, nd, perm);
                    try
                    {
                        return NDArray.AdoptLayout(compute(state, at ?? a, bt ?? b), layout);
                    }
                    finally
                    {
                        // The transposed operands are views only the route read; its result is a fresh array.
                        at?.Dispose();
                        bt?.Dispose();
                    }
                }
            }
            return compute(state, a, b);
        }

        /// <summary>
        ///     Runs a binary two-output route that writes C-ordered results (np.divmod's fused kernel) so that both fresh
        ///     outputs come out in NumPy's ufunc output order (parity QA-36): when the operands' K order is not C,
        ///     <paramref name="compute"/> runs over the operands transposed into it and both C results are relabelled;
        ///     otherwise it runs over the operands as they are.
        /// </summary>
        /// <typeparam name="TState">The route's own arguments, passed through so the lambda can be <c>static</c>.</typeparam>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand.</param>
        /// <param name="state">The route's arguments.</param>
        /// <param name="compute">The route: (state, first, second) → two fresh C-ordered results of the broadcast shape,
        ///     never one of its operands.</param>
        /// <returns>The route's two results in NumPy's layout.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (raised before the route runs, only when
        ///     the K order has to be computed).</exception>
        internal (NDArray, NDArray) KOrderBinaryPair<TState>(NDArray a, NDArray b, TState state,
            Func<TState, NDArray, NDArray, (NDArray, NDArray)> compute)
        {
            if (MayNeedKOrder(a.Shape, b.Shape))
            {
                var (kl, _) = Broadcast(a.Shape, b.Shape);
                if (TryKOrderPlan(kl.dimensions, new[] { a.Shape, b.Shape }, out var layout, out var perm))
                {
                    int nd = kl.NDim;
                    NDArray at = IntoKOrder(a, nd, perm), bt = IntoKOrder(b, nd, perm);
                    try
                    {
                        var (first, second) = compute(state, at ?? a, bt ?? b);
                        return (NDArray.AdoptLayout(first, layout), NDArray.AdoptLayout(second, layout));
                    }
                    finally
                    {
                        // The transposed operands are views only the route read; its results are fresh arrays.
                        at?.Dispose();
                        bt?.Dispose();
                    }
                }
            }
            return compute(state, a, b);
        }

        /// <summary>
        ///     Runs a unary two-output route that writes C-ordered results (np.frexp, np.modf) so that both fresh outputs
        ///     come out in NumPy's ufunc output order (parity QA-36): when the K order of the input (and of a mask, which
        ///     votes like any NpyIter operand) is not C, <paramref name="compute"/> runs over the operands transposed into
        ///     it and both C results are relabelled; otherwise it runs over the operands as they are.
        /// </summary>
        /// <typeparam name="TState">The route's own arguments, passed through so the lambda can be <c>static</c>.</typeparam>
        /// <param name="x">The input.</param>
        /// <param name="where">The <c>where=</c> mask, or null. A mask that would enlarge the result is left to the route
        ///     (it owns that shape and its errors), which then runs untransposed.</param>
        /// <param name="state">The route's arguments.</param>
        /// <param name="compute">The route: (state, input, mask) → two fresh C-ordered (or, for an F input, F-ordered)
        ///     results of the input's shape, never one of its operands.</param>
        /// <returns>The route's two results in NumPy's layout.</returns>
        /// <remarks>Only for FRESH outputs: a caller's <c>out=</c> keeps its own layout, so the routes call themselves
        ///     directly then.</remarks>
        internal (NDArray, NDArray) KOrderUnaryPair<TState>(NDArray x, NDArray where, TState state,
            Func<TState, NDArray, NDArray, (NDArray, NDArray)> compute)
        {
            bool maskVotes = where is not null && !IsKOrderCVoter(where.Shape);
            if ((!IsKOrderCVoter(x.Shape) || maskVotes) && (where is null || BroadcastsInto(where.Shape, x.Shape)))
            {
                long[] dims = x.Shape.dimensions;
                Shape[] voters = where is null ? new[] { x.Shape } : new[] { x.Shape, where.Shape };
                if (TryKOrderPlan(dims, voters, out var layout, out var perm))
                {
                    int nd = dims.Length;
                    NDArray xt = IntoKOrder(x, nd, perm), wt = where is null ? null : IntoKOrder(where, nd, perm);
                    try
                    {
                        var (first, second) = compute(state, xt ?? x, wt ?? where);
                        return (NDArray.AdoptLayout(first, layout), NDArray.AdoptLayout(second, layout));
                    }
                    finally
                    {
                        // The transposed operands are views only the route read; its results are fresh arrays.
                        xt?.Dispose();
                        wt?.Dispose();
                    }
                }
            }
            return compute(state, x, where);
        }

        /// <summary>
        ///     Whether <paramref name="s"/> broadcasts into <paramref name="target"/>'s dims without enlarging them: no more
        ///     axes, and every right-aligned extent equal or 1.
        /// </summary>
        /// <param name="s">The operand that must fit (a mask).</param>
        /// <param name="target">The shape it must fit into (the input).</param>
        /// <returns>True when the broadcast of the two is <paramref name="target"/>'s dims.</returns>
        private static bool BroadcastsInto(in Shape s, in Shape target)
        {
            var sd = s.dimensions;
            var td = target.dimensions;
            if (sd.Length > td.Length)
                return false;
            for (int i = 1; i <= sd.Length; i++)
            {
                long a = sd[sd.Length - i], b = td[td.Length - i];
                if (a != b && a != 1)
                    return false;
            }
            return true;
        }

        /// <summary>
        ///     One operand of a <see cref="TryKOrderPlan"/> computation, transposed into the plan's memory order: padded to
        ///     the result's rank with leading length-1 axes (a view), then permuted by <paramref name="perm"/> (a view).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="nd">The result's rank.</param>
        /// <param name="perm">The plan's permutation.</param>
        /// <returns>Null for a 0-d operand (it broadcasts anywhere: use it as it is); else a VIEW of
        ///     <paramref name="x"/> the caller must dispose (it shares <paramref name="x"/>'s buffer and keeps it alive).</returns>
        /// <remarks>Only views are made, never copies: a dense permuted operand comes out C-contiguous, so the kernel that
        ///     runs over it walks memory in order — the reason this route is not slower than NumPy's own K-order walk.</remarks>
        internal NDArray IntoKOrder(NDArray x, int nd, int[] perm)
        {
            if (x.ndim == 0)
                return null;
            NDArray padded = NDArray.AtLeastNdView(x, nd);
            try
            {
                return Transpose(padded, perm);
            }
            finally
            {
                // The transposed view holds its own reference on the buffer; the padding view (when one was made) is done.
                if (!ReferenceEquals(padded, x))
                    padded.Dispose();
            }
        }
    }
}
