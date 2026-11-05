using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // The layout of a fresh elementwise result
        // =============================================================================
        //
        // NumPy 2.4.2 decides a fresh ufunc output's memory order in one place (Shape.UfuncOutputShape): the trivial
        // loop's C/F allocation when the call qualifies, NpyIter's K order otherwise — so np.negative of a transposed
        // array is laid out like the transpose, an F array plus a 0-d scalar stays F, and an F array plus a C array is
        // C (C order wins NpyIter's conflicts). NumSharp's kernels write C order, so an engine that computes a non-C
        // result uses one of two equivalent mechanisms:
        //   - the linear walk: when every array operand is stored exactly as the result will be (same dims, and the
        //     strides of the result's dense layout on every axis longer than one), the whole-array kernels walk all
        //     buffers linearly and element k of each is the same logical element — NumPy's own trivial loop over F or
        //     permuted memory;
        //   - the permuted walk: otherwise every operand is viewed in the result's memory order
        //     (OperandInLayoutOrder), the engine's C routes compute a C-ordered result over those views, and
        //     NDArray.AdoptLayout relabels it into the layout without moving an element.
        // Only the strides of the result change, never a value: every kernel is elementwise.
        // =============================================================================

        /// <summary>
        ///     NumPy's layout for the fresh result of a one-input, one-output ufunc call (no <c>out=</c>, no
        ///     <c>where=</c>), when it is not plain C order.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="inputCast">True when NumPy's loop reads another dtype than <paramref name="x"/>'s (a cast input:
        ///     <c>np.sqrt</c> of an int32 array runs the float64 loop), which refuses the trivial loop for an N-D input.</param>
        /// <param name="layout">When this returns true: the dense, offset-0 layout NumPy allocates. It shares the input's
        ///     dims array, as every elementwise result does (nothing writes into a Shape's arrays).</param>
        /// <returns>False when the result is C-ordered — every C-contiguous, 0-d, 1-D or empty input, and any input whose
        ///     memory order NumPy resolves to C — so the caller allocates as it always did; true otherwise.</returns>
        /// <remarks>
        ///     Costs two flag reads on the common path; the NpyIter vote (and its strides allocation) runs only for an
        ///     input that is not C-contiguous. An empty result is left to the caller's C allocation: its flags are C and F
        ///     whatever the strides.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryUnaryNonCLayout(NDArray x, bool inputCast, out Shape layout)
        {
            var s = x.Shape;
            // A strided or reversed C-order view votes C on its own: answered without building the vote.
            if (s.IsContiguous || s.dimensions.Length < 2 || s.size == 0 || VotesCOrder(s))
            {
                layout = default;
                return false;
            }
            return TryNonCLayout(s.dimensions, new ReadOnlySpan<Shape>(in s),
                                 !inputCast && UfuncLoopOperand.IsAligned(x), out layout);
        }

        /// <summary>
        ///     NumPy's layout for the fresh result of a two-input, one-output ufunc call (no <c>out=</c>, no
        ///     <c>where=</c>), when it is not plain C order.
        /// </summary>
        /// <param name="lhs">The first input.</param>
        /// <param name="rhs">The second input.</param>
        /// <param name="lhsCast">True when the loop reads another dtype than <paramref name="lhs"/>'s (the int32 operand of
        ///     <c>int32 + float64</c>), which refuses the trivial loop when that input has two or more dimensions.</param>
        /// <param name="rhsCast">The same for <paramref name="rhs"/>.</param>
        /// <param name="layout">When this returns true: the dense, offset-0 layout NumPy allocates.</param>
        /// <returns>False when the result is C-ordered — both inputs 0-d or C-contiguous, a result of fewer than two
        ///     dimensions or no elements, or operands whose memory orders NumPy resolves to C — so the caller allocates as
        ///     it always did; true otherwise.</returns>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast (NumPy's text, from
        ///     <see cref="Broadcast(Shape, Shape)"/>) — raised only for an input that is not C-contiguous, which every route
        ///     would raise next anyway.</exception>
        internal static bool TryBinaryNonCLayout(NDArray lhs, NDArray rhs, bool lhsCast, bool rhsCast, out Shape layout)
        {
            var ls = lhs.Shape;
            var rs = rhs.Shape;
            // Two C-contiguous (or 0-d) inputs give a C result on both of NumPy's routes: the trivial loop fills C
            // strides, and every vote NpyIter's sort reads says C (a C-contiguous array's strides fall along its axes).
            if ((ls.IsContiguous || ls.dimensions.Length == 0) && (rs.IsContiguous || rs.dimensions.Length == 0))
            {
                layout = default;
                return false;
            }
            // Two operands of one shape (the common case) are their own broadcast: Broadcast would build two shapes only
            // for their dims, a measurable share of a small call's cost.
            var dims = Shape.SameDimensions(ls.dimensions, rs.dimensions)
                ? ls.dimensions
                : Broadcast(ls, rs).LeftShape.dimensions;
            // The two exact C answers that need no vote: a full-rank C operand decides C (C wins every conflict), and
            // operands that each vote C leave nothing to swap (a C array beside a permuted view, two strided C views).
            if (dims.Length < 2 || HasZeroExtent(dims) || DecidesCOrder(ls, dims.Length) || DecidesCOrder(rs, dims.Length)
                || (VotesCOrder(ls) && VotesCOrder(rs)))
            {
                layout = default;
                return false;
            }
            bool trivial = (ls.dimensions.Length < 2 || (!lhsCast && UfuncLoopOperand.IsAligned(lhs)))
                           && (rs.dimensions.Length < 2 || (!rhsCast && UfuncLoopOperand.IsAligned(rhs)));
            // The layout adopts the dims array (the operand's own, or the broadcast's): shared, as an elementwise result
            // shares its input's, since nothing writes into a Shape's arrays.
            return TryNonCLayout(dims, new[] { ls, rs }, trivial, out layout);
        }

        /// <summary>
        ///     NumPy's layout for the fresh result of an N-input, one-output elementwise call (the comparison engine's
        ///     row kernels, np.where's three operands), when it is not plain C order.
        /// </summary>
        /// <param name="inputs">The inputs' shapes.</param>
        /// <param name="dims">The broadcast result dimensions. Not adopted.</param>
        /// <param name="trivialLoopEligible">See <see cref="Shape.UfuncOutputShape"/>: false when a mask is present, the
        ///     call has two outputs, or an input of two or more dimensions is cast or unaligned.</param>
        /// <param name="layout">When this returns true: the dense, offset-0 layout NumPy allocates.</param>
        /// <returns>False when the result is C-ordered (see <see cref="TryBinaryNonCLayout"/>); true otherwise.</returns>
        internal static bool TryNonCResultLayout(ReadOnlySpan<Shape> inputs, long[] dims, bool trivialLoopEligible,
                                                 out Shape layout)
        {
            bool allC = true;
            for (int i = 0; i < inputs.Length && allC; i++)
                allC = inputs[i].IsContiguous || inputs[i].dimensions.Length == 0;
            if (allC || dims.Length < 2 || HasZeroExtent(dims))
            {
                layout = default;
                return false;
            }
            // The exact C answers that need no vote (see TryBinaryNonCLayout): one full-rank C operand, or every operand
            // voting C.
            bool allVoteC = true;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (DecidesCOrder(inputs[i], dims.Length))
                {
                    layout = default;
                    return false;
                }
                allVoteC &= VotesCOrder(inputs[i]);
            }
            if (allVoteC)
            {
                layout = default;
                return false;
            }
            return TryNonCLayout((long[])dims.Clone(), inputs, trivialLoopEligible, out layout);
        }

        /// <summary>
        ///     Whether NpyIter's vote of this operand alone is C order: along its axes, every stride that votes (an axis
        ///     longer than one with a nonzero stride; the others abstain) is no larger in magnitude than the voting stride
        ///     before it, so the insertion sort never swaps a pair it votes on (a strided C view such as <c>a[:, ::2]</c>, a
        ///     reversed one, a broadcast row).
        /// </summary>
        /// <param name="s">The operand's shape.</param>
        /// <returns>True when the operand votes C on every pair of axes it votes on.</returns>
        /// <remarks>
        ///     When every operand votes C, no pair is ever swapped, so NpyIter's result is C; the trivial loop cannot then
        ///     choose F either (an F-only operand of two or more voting axes has growing strides). The callers use it to
        ///     answer C without building the vote.
        /// </remarks>
        private static bool VotesCOrder(in Shape s)
        {
            var d = s.dimensions;
            var st = s.strides;
            long previous = long.MaxValue;
            for (int i = 0; i < d.Length; i++)
            {
                long stride = st[i];
                if (d[i] == 1 || stride == 0)
                    continue;
                long magnitude = Math.Abs(stride);
                if (magnitude > previous)
                    return false;
                previous = magnitude;
            }

            return true;
        }

        /// <summary>
        ///     Whether a C-contiguous operand votes on EVERY axis of a result of <paramref name="resultNdim"/> dimensions:
        ///     it has the result's rank and no axis of extent one (where it would abstain). It then votes C on every pair,
        ///     and NpyIter lets C order win any conflict with another operand, so the result is C whatever the others are;
        ///     the trivial loop gives C too (the operand's flag pair is C-only, and a different pair refuses the loop).
        /// </summary>
        /// <param name="s">The operand's shape.</param>
        /// <param name="resultNdim">The result's rank.</param>
        /// <returns>True when this operand alone decides a C result.</returns>
        private static bool DecidesCOrder(in Shape s, int resultNdim)
        {
            if (!s.IsContiguous || s.dimensions.Length != resultNdim)
                return false;
            var d = s.dimensions;
            for (int i = 0; i < d.Length; i++)
            {
                if (d[i] == 1)
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     Runs NumPy's layout decision and reports whether it is anything but C order.
        /// </summary>
        /// <param name="dims">The result's dims, adopted by the layout (possibly an operand's own array: shared, as an
        ///     elementwise result's dims are, since nothing writes into a Shape's arrays).</param>
        /// <param name="inputs">The inputs' shapes.</param>
        /// <param name="trivialLoopEligible">See <see cref="Shape.UfuncOutputShape"/>.</param>
        /// <param name="layout">The layout when it is not C order.</param>
        /// <returns>True when the layout is not C-contiguous.</returns>
        /// <remarks>A C-contiguous answer is always the plain C layout (NumPy's sort moves no axis when the result's
        ///     axes come out in C order), so the caller's own C allocation is the same answer.</remarks>
        private static bool TryNonCLayout(long[] dims, ReadOnlySpan<Shape> inputs, bool trivialLoopEligible, out Shape layout)
        {
            var l = Shape.UfuncOutputShape(dims, inputs, trivialLoopEligible);
            if (l.IsContiguous)
            {
                layout = default;
                return false;
            }
            layout = l;
            return true;
        }

        /// <summary>Whether a dims array has a zero extent (an empty result).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>True when any extent is zero.</returns>
        private static bool HasZeroExtent(long[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] == 0)
                    return true;
            }
            return false;
        }

        /// <summary>
        ///     Whether an operand can be walked LINEARLY into a result allocated in <paramref name="layout"/>: it has the
        ///     result's dims and is stored exactly as the layout stores them (the same stride on every axis longer than
        ///     one), so reading its buffer from its first element visits the elements in the result's memory order. Such
        ///     operands run the whole-array kernels directly — NumPy's trivial loop over F-ordered or permuted memory.
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <param name="layout">A dense layout of the result (strides all positive).</param>
        /// <returns>True when the linear walk is exact for this operand.</returns>
        /// <remarks>Equal strides to a dense positive layout imply the operand is dense with positive strides, so its
        ///     first element is at its offset and no stride-0 (broadcast) or negative axis can pass.</remarks>
        internal static bool IsLinearInLayout(NDArray x, in Shape layout)
        {
            var s = x.Shape;
            if (s.dimensions.Length != layout.dimensions.Length)
                return false;
            var dims = s.dimensions;
            var want = layout.dimensions;
            for (int k = 0; k < dims.Length; k++)
            {
                if (dims[k] != want[k])
                    return false;
            }
            return s.IsLaidOutAs(layout);
        }

        /// <summary>
        ///     An operand of an elementwise call viewed in the result layout's memory order — right-aligned against the
        ///     result's rank (a missing leading axis gets extent 1), then its axes permuted by <paramref name="perm"/>
        ///     (<see cref="Shape.DenseAxisOrder"/> of the layout, outermost first) — so a kernel that writes C order over
        ///     the permuted dims produces the layout's buffer.
        /// </summary>
        /// <param name="x">The operand (any layout; broadcast and negative strides included).</param>
        /// <param name="ndim">The result's rank (at least <paramref name="x"/>'s).</param>
        /// <param name="perm">The layout's axes in memory order, outermost first (length <paramref name="ndim"/>).</param>
        /// <returns><paramref name="x"/> itself for a 0-d operand (it broadcasts as it is); otherwise a new view over the
        ///     same buffer, which the caller must dispose (see <see cref="DisposeLayoutView"/>).</returns>
        /// <remarks>The view is a plain NDArray over the operand's storage (<see cref="UnmanagedStorage.Alias(ref Shape)"/>),
        ///     so a subclass operand's own rules (a matrix's) never apply to it, and it keeps the operand's writeability.</remarks>
        internal static NDArray OperandInLayoutOrder(NDArray x, int ndim, int[] perm)
        {
            var s = x.Shape;
            var od = s.dimensions;
            if (od.Length == 0)
                return x;
            var os = s.strides;
            int shift = ndim - od.Length;
            var dims = new long[ndim];
            var strides = new long[ndim];
            for (int i = 0; i < ndim; i++)
            {
                int k = perm[i] - shift;
                dims[i] = k < 0 ? 1 : od[k];
                strides[i] = k < 0 ? 0 : os[k];
            }
            var view = new Shape(dims, strides, s.offset, s.bufferSize > 0 ? s.bufferSize : s.size);
            return new NDArray(x.Storage.Alias(ref view)) { TensorEngine = x.TensorEngine };
        }

        /// <summary>
        ///     Computes a fresh one-input elementwise result in a non-C layout NumPy allocates, with a computation that can
        ///     only write C order (the permuted walk of the file header): the input is viewed in the layout's memory order,
        ///     <paramref name="computeC"/> writes a C-ordered result over the permuted dims, and that result is relabelled
        ///     as <paramref name="layout"/>.
        /// </summary>
        /// <typeparam name="TState">What the computation needs besides its operand (the engine, the op, the dtype), passed
        ///     through so a static lambda needs no closure.</typeparam>
        /// <param name="x">The input.</param>
        /// <param name="layout">The layout NumPy allocates (dense, offset 0, non-empty).</param>
        /// <param name="state">The computation's state.</param>
        /// <param name="computeC">The computation: given the permuted view, a fresh C-contiguous result of its dims.</param>
        /// <returns>The computed result, laid out as <paramref name="layout"/>.</returns>
        /// <exception cref="Exception">Whatever <paramref name="computeC"/> raises; the view is disposed first.</exception>
        internal static NDArray ComputeInLayout<TState>(NDArray x, in Shape layout, TState state,
                                                       Func<NDArray, TState, NDArray> computeC)
        {
            int[] perm = layout.DenseAxisOrder();
            NDArray v = null;
            try
            {
                v = OperandInLayoutOrder(x, layout.NDim, perm);
                return NDArray.AdoptLayout(computeC(v, state), layout);
            }
            finally
            {
                DisposeLayoutView(v, x);
            }
        }

        /// <summary>
        ///     <see cref="ComputeInLayout{TState}(NDArray, in Shape, TState, Func{NDArray, TState, NDArray})"/> for a
        ///     two-input computation: both inputs are viewed in the layout's memory order (a lower-rank one right-aligned
        ///     first, a 0-d one passed as it is).
        /// </summary>
        /// <typeparam name="TState">The computation's state type.</typeparam>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input.</param>
        /// <param name="layout">The layout NumPy allocates (dense, offset 0, non-empty).</param>
        /// <param name="state">The computation's state.</param>
        /// <param name="computeC">The computation: given the two permuted views, a fresh C-contiguous result of their
        ///     broadcast dims.</param>
        /// <returns>The computed result, laid out as <paramref name="layout"/>.</returns>
        /// <exception cref="Exception">Whatever <paramref name="computeC"/> raises; both views are disposed first.</exception>
        internal static NDArray ComputeInLayout<TState>(NDArray x1, NDArray x2, in Shape layout, TState state,
                                                       Func<NDArray, NDArray, TState, NDArray> computeC)
        {
            int nd = layout.NDim;
            int[] perm = layout.DenseAxisOrder();
            NDArray v1 = null, v2 = null;
            try
            {
                v1 = OperandInLayoutOrder(x1, nd, perm);
                v2 = OperandInLayoutOrder(x2, nd, perm);
                return NDArray.AdoptLayout(computeC(v1, v2, state), layout);
            }
            finally
            {
                DisposeLayoutView(v1, x1);
                DisposeLayoutView(v2, x2);
            }
        }

        /// <summary>
        ///     Disposes a view <see cref="OperandInLayoutOrder"/> made, leaving the operand itself alone (a 0-d operand is
        ///     returned as is and belongs to the caller).
        /// </summary>
        /// <param name="view">The view, or the operand itself, or null.</param>
        /// <param name="operand">The operand the view was made from.</param>
        internal static void DisposeLayoutView(NDArray view, NDArray operand)
        {
            if (view is not null && !ReferenceEquals(view, operand))
                view.Dispose();
        }
    }
}
