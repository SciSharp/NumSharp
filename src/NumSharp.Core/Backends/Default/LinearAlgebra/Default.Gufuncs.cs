using System;

namespace NumSharp.Backends
{
    /// <summary>
    ///     The product gufuncs' whole calls — <c>np.matmul</c>, <c>np.vecdot</c>, <c>np.matvec</c>, <c>np.vecmat</c> —
    ///     moved from the public functions by the np-layer dtype plan's P2, statement for statement (NumPy implements them
    ///     in C, so they are engine members: D1). Each runs its argument checks, remaps the core axes, selects the loop and
    ///     calls its product kernel (<see cref="Matmul(NDArray, NDArray)"/>, <see cref="TensorEngine.Vecdot(NDArray, NDArray)"/>,
    ///     <see cref="TensorEngine.Matvec(NDArray, NDArray)"/>, <see cref="TensorEngine.Vecmat(NDArray, NDArray)"/>), then
    ///     places the result's core axes and delivers it into <c>out=</c>. The public functions keep the numpy.matrix
    ///     boundary.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        ///     The gufunc signature <c>np.matmul</c> matches its core dimensions against. The <c>?</c>
        ///     marks OPTIONAL core dimensions: a 1-D operand drops its <c>n?</c>/<c>m?</c>, which is
        ///     how <c>matmul</c> promotes a vector operand.
        /// </summary>
        internal const string MatmulSignature = "(n?,k),(k,m?)->(n?,m?)";

        /// <summary>
        ///     The gufunc signature <c>np.vecdot</c> matches its core dimensions against.
        /// </summary>
        internal const string VecdotSignature = "(n),(n)->()";

        /// <summary>
        ///     The gufunc signature <c>np.matvec</c> matches its core dimensions against.
        /// </summary>
        internal const string MatvecSignature = "(m,n),(n)->(m)";

        /// <summary>
        ///     The gufunc signature <c>np.vecmat</c> matches its core dimensions against.
        /// </summary>
        internal const string VecmatSignature = "(n),(n,m)->(m)";

        /// <inheritdoc/>
        /// <remarks>
        ///     The checks run in this order: NumPy's argument parse (<see cref="GufuncGuard.CheckParsedArguments"/>:
        ///     <c>axis</c> with <c>axes</c>, the <c>dtype=</c> class, a read-only <c>out=</c>), the casting and order names,
        ///     the loop refusal of text (<see cref="GufuncGuard.ThrowIfText"/>), each input's and the output's cast under
        ///     <c>casting</c> (<see cref="GufuncGuard.ValidateCasting"/>), <c>keepdims</c>, <c>axis</c>, the <c>axes=</c>
        ///     list; after the product the output's core dimensions. Moved from the public <c>np.matmul</c> by the np-layer
        ///     dtype plan's P2.
        /// </remarks>
        [NDScoped] // engine boundary: the moved operands, the loop casts and the product before its delivery are reclaimed
        public override NDArray Matmul(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool? keepdims,
            DType dtype, string casting, char order)
        {
            // NumPy's argument parse (ufunc_generic_fastcall): axis with axes, the dtype= class, a read-only out, then
            // the casting and order names (convert_ufunc_arguments), then the loop resolution — all before keepdims /
            // axis semantics.
            GufuncGuard.CheckParsedArguments(axes, axis, dtype, @out);
            GufuncGuard.RequireCasting(casting);
            GufuncGuard.RequireOrder(order);
            GufuncGuard.ThrowIfText("matmul", x1, x2, dtype);
            // NumPy's validate_casting, before the gufunc checks: each input's cast to the loop, then the loop's cast to
            // out=, under `casting`.
            var loop = GufuncGuard.ResolveLoop(x1, x2, dtype);
            GufuncGuard.ValidateCasting("matmul", x1, x2, loop, @out, casting);

            // NumPy's gufunc checks (PyUFunc_GeneralizedFunctionInternal), over x1, x2 and out=: keepdims and axis
            // support, the ranks (n? / m? dropped for a 1-D operand, or for a short out=), axes=, the core sizes.
            var layout = GufuncGuard.CheckCoreDimensions(GufuncSignature.Matmul, new[] { x1, x2, @out }, axes, axis,
                keepdims.HasValue, keepdims ?? false);

            var a = x1;
            var b = x2;
            int[] outputAxes = Array.Empty<int>();
            if (axes is not null)
            {
                // The output entry as given (three entries: matmul's output has core axes): it is placed against the
                // product's rank once the loop has run.
                outputAxes = (int[])(axes[2] ?? Array.Empty<int>()).Clone();

                // Bring each operand's core axes to the trailing positions the kernel expects; the
                // result's own core axes are relocated to `outputAxes` after the product runs.
                a = BringCoreToEnd(a, layout.CoreAxes(0, x1.ndim));
                b = BringCoreToEnd(b, layout.CoreAxes(1, x2.ndim));
            }

            // NumPy's iterator lays the output's loop axes out in K order (order='K', the default; 'C' / 'F' / 'A' lay
            // out the whole output below). The operands vote as given, before a dtype= copy; with out= the answer is
            // copied into out's own layout, so no order is needed.
            int aCore = a.ndim == 1 ? 1 : 2, bCore = b.ndim == 1 ? 1 : 2;
            int[] loopPerm = null;
            Shape loopLayout = default;
            if (order == 'K' && @out is null)
            {
                long n = a.ndim == 1 ? 1 : a.Shape.dimensions[a.ndim - 2];
                long m = b.ndim == 1 ? 1 : b.Shape.dimensions[b.ndim - 1];
                // The batched product writes C order, so only a vote against C order needs work.
                GufuncLoopOrder(a, aCore, b, bCore, n * m, out loopLayout, out loopPerm);
            }

            // A dtype= request selects the loop (its casts validated above): cast so the product (and the backend route)
            // run at that dtype. astype converts a fixed-width text operand too, which only casting='unsafe' lets
            // through.
            if (dtype is not null)
            {
                a = a.astype(dtype, copy: false);
                b = b.astype(dtype, copy: false);
            }

            if (loopPerm != null)
            {
                a = PermuteLoopAxes(a, aCore, loopLayout.NDim, loopPerm);
                b = PermuteLoopAxes(b, bCore, loopLayout.NDim, loopPerm);
            }

            var result = Matmul(a, b);
            if (loopPerm != null)
                result = AdoptLoopLayout(result, loopLayout);

            // order= lays out the iterator's axes (the loop axes, then the core axes), and axes= remaps them after:
            // NumPy allocates the output over the iterator and returns it with the remapped axes.
            result = ApplyMatmulOrder(result, order, x1, x2);

            if (outputAxes.Length > 0)
                result = PlaceOutputCore(result, outputAxes);

            // The out cast and the out's core dimensions were validated before the product; without out= a 0-d answer
            // (two vectors) is handed back as NumPy's scalar stand-in.
            return GufuncGuard.Deliver(result, @out);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     The checks run in this order: NumPy's argument parse (<c>axis</c> with <c>axes</c>, the <c>dtype=</c> class, a
        ///     read-only <c>out=</c>), the loop refusal of text, each input's and the output's cast under same_kind (NumPy's
        ///     default rule; this gufunc takes no casting=), each operand's rank, the <c>axes=</c> list or the <c>axis=</c>
        ///     value, the core sizes; the delivery is an unsafe <c>np.copyto</c> (the cast validated). Moved from the
        ///     public <c>np.vecdot</c> by the np-layer dtype plan's P2.
        /// </remarks>
        [NDScoped] // engine boundary: the moved operands, the loop casts and the answer before its delivery are reclaimed
        public override NDArray Vecdot(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype)
        {
            // NumPy's argument parse (ufunc_generic_fastcall): axis with axes, the dtype= class, a read-only out, then
            // the loop resolution — all before any core-dimension, axis or keepdims check.
            GufuncGuard.CheckParsedArguments(axes, axis, dtype, @out);
            GufuncGuard.ThrowIfText("vecdot", x1, x2, dtype);
            // NumPy's validate_casting under its default rule (this gufunc takes no casting=): each input's cast to the
            // loop, then the loop's cast to out=, before the gufunc checks.
            var loop = GufuncGuard.ResolveLoop(x1, x2, dtype);
            GufuncGuard.ValidateCasting("vecdot", x1, x2, loop, @out, "same_kind");
            // NumPy's gufunc checks, over x1, x2 and out=: keepdims support (this signature allows it), the ranks,
            // axes= / axis=, the core sizes.
            var layout = GufuncGuard.CheckCoreDimensions(GufuncSignature.Vecdot, new[] { x1, x2, @out }, axes, axis,
                keepdims, keepdims);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                // Under keepdims the output gains a core dimension (the reduced axis, kept as size 1), so a
                // PROVIDED output entry names where the kept axis lands; an omitted one puts it at the end.
                outputAxes = axes.Length == 3 ? (int[])(axes[2] ?? System.Array.Empty<int>()).Clone() : System.Array.Empty<int>();

                a = MoveAxis(a, new[] {layout.CoreAxes(0, x1.ndim)[0]}, new[] {-1});
                b = MoveAxis(b, new[] {layout.CoreAxes(1, x2.ndim)[0]}, new[] {-1});
            }
            else if (axis is not null)
            {
                // NumPy allows a bare `axis` only where every core dimension is the SAME one — true
                // for vecdot's (n),(n)->() and for no other member of this family (matvec and vecmat
                // raise TypeError). It names the shared core axis in EACH operand, so both move it
                // to the end and the ordinary kernel runs. When the named axis IS already the last
                // one — the overwhelmingly common case, and what np.linalg.vecdot's axis=-1 default
                // always resolves to — the moveaxis is a pure no-op view rebuild, so skip it (the
                // validation already ran). This is what kept np.linalg.vecdot ~2x slower than a bare
                // np.vecdot on the same 1-D operands.
                int ra = layout.CoreAxes(0, a.ndim)[0];
                if (ra != a.ndim - 1)
                    a = MoveAxis(a, new[] {ra}, new[] {-1});
                int rb = layout.CoreAxes(1, b.ndim)[0];
                if (rb != b.ndim - 1)
                    b = MoveAxis(b, new[] {rb}, new[] {-1});
            }

            // NumPy's iterator lays the output's loop axes out in K order; the operands vote as given (see Matmul).
            // The product kernel takes its layout from the elementwise product, whose own vote counts the core axis
            // too, so its answer takes the voted layout even when the vote gives C order (AdoptLayout relabels a
            // product already in that order and copies any other).
            Shape loopLayout = default;
            int[] loopPerm = null;
            bool voted = @out is null && GufuncLoopOrder(a, 1, b, 1, 1, out loopLayout, out loopPerm);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);
            // The float16 loop's kernel is HALF_dot, which needs two float16 operands: NumPy casts a mixed pair (float16
            // with int8, say) to the loop, and the elementwise kernel would round each product to float16 instead.
            if (dtype is null && loop.GetTypeCode() == NPTypeCode.Half)
            {
                a = GufuncGuard.ToLoop(a, loop);
                b = GufuncGuard.ToLoop(b, loop);
            }

            if (loopPerm != null)
            {
                a = PermuteLoopAxes(a, 1, loopLayout.NDim, loopPerm);
                b = PermuteLoopAxes(b, 1, loopLayout.NDim, loopPerm);
            }

            var result = Vecdot(a, b);
            if (voted)
                result = AdoptLoopLayout(result, loopLayout);

            if (keepdims)
            {
                // The reduced axis is kept as size 1, at the position `axes` gave the output core dim
                // (validated against the OUTPUT rank), at the position `axis=` named, or — when the
                // output entry was omitted — at the END. All three collapse to one expand_dims.
                int outNdim = result.ndim + 1;
                int keptAxis = outNdim - 1;
                if (axes is not null && outputAxes.Length == 1)
                {
                    int raw = outputAxes[0];
                    keptAxis = raw < 0 ? raw + outNdim : raw;
                    if (keptAxis < 0 || keptAxis >= outNdim)
                        throw new AxisError(raw, outNdim);
                }
                else if (axis is not null)
                {
                    keptAxis = axis.Value < 0 ? axis.Value + outNdim : axis.Value;
                }

                // The answer keeps owning its buffer and its layout, as NumPy's output does: insert the length-1 axis
                // by relabelling.
                result = NDArray.AdoptInsertedAxis(result, keptAxis);
            }

            return GufuncGuard.Deliver(result, @out);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     The checks run in this order: NumPy's argument parse (<c>axis</c> with <c>axes</c>, the <c>dtype=</c> class, a
        ///     read-only <c>out=</c>), the loop refusal of text, each input's and the output's cast under same_kind
        ///     (NumPy's default rule; this gufunc takes no casting=), <c>axis</c> / <c>keepdims</c>, each operand's rank,
        ///     the <c>axes=</c> list, the core sizes; the delivery is an unsafe <c>np.copyto</c> (the cast validated).
        ///     Moved from the public <c>np.matvec</c> by the np-layer dtype plan's P2.
        /// </remarks>
        [NDScoped] // engine boundary: the moved operands, the loop casts and the answer before its delivery are reclaimed
        public override NDArray Matvec(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype)
        {
            // NumPy's argument parse (ufunc_generic_fastcall): axis with axes, the dtype= class, a read-only out, then
            // the loop resolution — all before any core-dimension, axis or keepdims check.
            GufuncGuard.CheckParsedArguments(axes, axis, dtype, @out);
            GufuncGuard.ThrowIfText("matvec", x1, x2, dtype);
            // NumPy's validate_casting under its default rule (this gufunc takes no casting=): each input's cast to the
            // loop, then the loop's cast to out=, before the gufunc checks.
            var loop = GufuncGuard.ResolveLoop(x1, x2, dtype);
            GufuncGuard.ValidateCasting("matvec", x1, x2, loop, @out, "same_kind");
            // NumPy's gufunc checks, over x1, x2 and out=: keepdims and axis support (this signature has two distinct
            // core dimensions, so both are refused), the ranks, axes=, the core sizes.
            var layout = GufuncGuard.CheckCoreDimensions(GufuncSignature.Matvec, new[] { x1, x2, @out }, axes, axis,
                keepdims, keepdims);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                // Three entries (the output has a core axis): the output entry is placed after the loop has run.
                outputAxes = (int[])(axes[2] ?? System.Array.Empty<int>()).Clone();

                // Bring (m,n) and (n) to the trailing positions the kernel expects; the result's
                // own core axis goes back to where `axes` asked for it once the loop has run.
                a = MoveAxis(a, layout.CoreAxes(0, x1.ndim), new[] {a.ndim - 2, a.ndim - 1});
                b = MoveAxis(b, new[] {layout.CoreAxes(1, x2.ndim)[0]}, new[] {-1});
            }

            // NumPy's iterator lays the output's loop axes out in K order; the operands vote as given (see Matmul).
            // The batched product writes C order, so only a vote against C order needs work.
            Shape loopLayout = default;
            int[] loopPerm = null;
            if (@out is null)
                GufuncLoopOrder(a, 2, b, 1, a.Shape.dimensions[a.ndim - 2], out loopLayout, out loopPerm);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);
            if (loopPerm != null)
            {
                a = PermuteLoopAxes(a, 2, loopLayout.NDim, loopPerm);
                b = PermuteLoopAxes(b, 1, loopLayout.NDim, loopPerm);
            }

            var result = Matvec(a, b);
            if (loopPerm != null)
                result = AdoptLoopLayout(result, loopLayout);

            if (outputAxes.Length == 1)
                result = PlaceOutputCore(result, outputAxes);

            return GufuncGuard.Deliver(result, @out);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     The checks run in this order: NumPy's argument parse (<c>axis</c> with <c>axes</c>, the <c>dtype=</c> class, a
        ///     read-only <c>out=</c>), the loop refusal of text, each input's and the output's cast under same_kind
        ///     (NumPy's default rule; this gufunc takes no casting=), <c>axis</c> / <c>keepdims</c>, each operand's rank,
        ///     the <c>axes=</c> list, the core sizes; the delivery is an unsafe <c>np.copyto</c> (the cast validated).
        ///     Moved from the public <c>np.vecmat</c> by the np-layer dtype plan's P2.
        /// </remarks>
        [NDScoped] // engine boundary: the moved operands, the loop casts and the answer before its delivery are reclaimed
        public override NDArray Vecmat(NDArray x1, NDArray x2, NDArray @out, int[][] axes, int? axis, bool keepdims,
            DType dtype)
        {
            // NumPy's argument parse (ufunc_generic_fastcall): axis with axes, the dtype= class, a read-only out, then
            // the loop resolution — all before any core-dimension, axis or keepdims check.
            GufuncGuard.CheckParsedArguments(axes, axis, dtype, @out);
            GufuncGuard.ThrowIfText("vecmat", x1, x2, dtype);
            // NumPy's validate_casting under its default rule (this gufunc takes no casting=): each input's cast to the
            // loop, then the loop's cast to out=, before the gufunc checks.
            var loop = GufuncGuard.ResolveLoop(x1, x2, dtype);
            GufuncGuard.ValidateCasting("vecmat", x1, x2, loop, @out, "same_kind");
            // NumPy's gufunc checks, over x1, x2 and out=: keepdims and axis support (this signature has two distinct
            // core dimensions, so both are refused), the ranks, axes=, the core sizes.
            var layout = GufuncGuard.CheckCoreDimensions(GufuncSignature.Vecmat, new[] { x1, x2, @out }, axes, axis,
                keepdims, keepdims);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                // Three entries (the output has a core axis): the output entry is placed after the loop has run.
                outputAxes = (int[])(axes[2] ?? System.Array.Empty<int>()).Clone();

                a = MoveAxis(a, new[] {layout.CoreAxes(0, x1.ndim)[0]}, new[] {-1});
                b = MoveAxis(b, layout.CoreAxes(1, x2.ndim), new[] {b.ndim - 2, b.ndim - 1});
            }

            // NumPy's iterator lays the output's loop axes out in K order; the operands vote as given (see Matmul).
            // The batched product writes C order, so only a vote against C order needs work.
            Shape loopLayout = default;
            int[] loopPerm = null;
            if (@out is null)
                GufuncLoopOrder(a, 1, b, 2, b.Shape.dimensions[b.ndim - 1], out loopLayout, out loopPerm);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);
            if (loopPerm != null)
            {
                a = PermuteLoopAxes(a, 1, loopLayout.NDim, loopPerm);
                b = PermuteLoopAxes(b, 2, loopLayout.NDim, loopPerm);
            }

            var result = Vecmat(a, b);
            if (loopPerm != null)
                result = AdoptLoopLayout(result, loopLayout);

            if (outputAxes.Length == 1)
                result = PlaceOutputCore(result, outputAxes);

            return GufuncGuard.Deliver(result, @out);
        }

        /// <summary>Moves an operand's core axes to its trailing positions (the kernel's canonical layout).</summary>
        /// <param name="operand">The operand.</param>
        /// <param name="coreAxes">Its core axes, normalized (non-negative), in core order.</param>
        /// <returns>A view with those axes last, in that order.</returns>
        private NDArray BringCoreToEnd(NDArray operand, int[] coreAxes)
        {
            var dest = new int[coreAxes.Length];
            for (int i = 0; i < coreAxes.Length; i++)
                dest[i] = operand.ndim - coreAxes.Length + i;
            return MoveAxis(operand, coreAxes, dest);
        }

        /// <summary>
        ///     Relocates the product's core axes (currently trailing) to the positions <c>axes[2]</c>
        ///     named, validated against the OUTPUT rank (unknown until the product ran).
        /// </summary>
        /// <param name="result">The product, its core axes last: a fresh array, consumed (it is returned relabelled).</param>
        /// <param name="outputAxes">The output entry of <c>axes=</c>, as the caller gave it (negative axes count from the
        ///     end of the output).</param>
        /// <returns>The product with the core axes where <paramref name="outputAxes"/> names them: the same buffer in its
        ///     computed memory order (C, or NumPy's K order of the loop axes), relabelled, so it still owns its data —
        ///     NumPy allocates a gufunc output in the iterator's order and returns it with the remapped axes (probed
        ///     2.4.2: <c>np.matmul(ones((2,3,4)), ones((2,4,5)), axes=[(1,2),(1,2),(0,1)])</c> has shape (3,5,2),
        ///     strides (40,8,120), OWNDATA True).</returns>
        /// <exception cref="AxisError">An output axis out of the result's range.</exception>
        private static NDArray PlaceOutputCore(NDArray result, int[] outputAxes)
        {
            var dest = new int[outputAxes.Length];
            for (int i = 0; i < outputAxes.Length; i++)
            {
                int ax = outputAxes[i] < 0 ? outputAxes[i] + result.ndim : outputAxes[i];
                if (ax < 0 || ax >= result.ndim)
                    throw new AxisError(outputAxes[i], result.ndim);
                dest[i] = ax;
            }

            // np.moveaxis's permutation, the trailing core axes moved to `dest`: each lands at its destination, and the
            // other axes fill the remaining positions in their order. The gufunc checks refused a repeated output axis,
            // so every destination is used once.
            int nd = result.ndim, ncore = outputAxes.Length;
            var order = new int[nd];
            var placed = new bool[nd];
            for (int i = 0; i < ncore; i++)
            {
                order[dest[i]] = nd - ncore + i;
                placed[dest[i]] = true;
            }

            for (int p = 0, next = 0; p < nd; p++)
            {
                if (!placed[p])
                    order[p] = next++;
            }

            // A relabel of the axes alone, valid for any dense layout of the product (C, or the K order AdoptLoopLayout
            // gave it).
            return NDArray.AdoptAxisPermutation(result, order);
        }

        /// <summary>
        ///     NumPy's memory order for a gufunc output's loop axes, when it is not C order. NumPy's gufunc iterator runs
        ///     over the loop axes plus the output's core axes. The inputs vote on the loop axes only: their own core axes
        ///     are not iterator axes, and every input's stride on an output core axis is 0, so those axes stay innermost,
        ///     in C order. <c>npyiter_find_best_axis_ordering</c> then sorts the loop axes as it sorts a ufunc's
        ///     (<see cref="Shape.NpyIterOutputShape"/>), and with order='K' (the default) the allocated output follows
        ///     that order: <c>np.matvec</c> of an F-ordered (2,3,4,5) array and a vector has strides (32,64,8), its core
        ///     axis innermost and the loop axes F-ordered (probed 2.4.2).
        /// </summary>
        /// <param name="a">The first input with its core axes last, as the caller gave it: before a <c>dtype=</c> cast,
        ///     because a cast copy would give a broadcast axis a stride that votes.</param>
        /// <param name="aCore">The number of <paramref name="a"/>'s trailing core axes.</param>
        /// <param name="b">The second input, likewise.</param>
        /// <param name="bCore">The number of <paramref name="b"/>'s trailing core axes.</param>
        /// <param name="outputCoreSize">The element count of one output core block (1 when the output has no core axes);
        ///     0 for an empty output.</param>
        /// <param name="loopLayout">Set, when the vote ran, to the loop axes' layout: the broadcast loop dims with the
        ///     dense strides NumPy allocates them with (in elements, for one element per core block).</param>
        /// <param name="perm">Set, when the vote ran and did not give C order, to the loop axes in memory order,
        ///     outermost first — the transposition under which the C-writing kernels write NumPy's order; null
        ///     otherwise.</param>
        /// <returns>Whether the vote ran: false when NumPy's output is C-ordered whatever the kernel, namely with fewer
        ///     than two loop axes (a 0-d or 1-D loop is C and F at once), two C-contiguous inputs (they vote for C order:
        ///     every pair of axes both see compares unswapped, the rest abstain), or an empty output. A kernel whose
        ///     layout follows its own operands (vecdot's elementwise product) needs the layout even when the vote
        ///     gives C order; the matrix products always write C order and need only <paramref name="perm"/>.</returns>
        private static bool GufuncLoopOrder(NDArray a, int aCore, NDArray b, int bCore, long outputCoreSize,
            out Shape loopLayout, out int[] perm)
        {
            loopLayout = default;
            perm = null;
            int aLoop = a.ndim - aCore, bLoop = b.ndim - bCore;
            int loopNd = Math.Max(aLoop, bLoop);
            if (loopNd < 2 || outputCoreSize == 0 || (a.Shape.IsContiguous && b.Shape.IsContiguous))
                return false;

            // The broadcast loop dims (CheckLoopBroadcast has validated the broadcast).
            var dims = new long[loopNd];
            for (int i = 0; i < loopNd; i++)
            {
                int ia = i - (loopNd - aLoop), ib = i - (loopNd - bLoop);
                long da = ia >= 0 ? a.Shape.dimensions[ia] : 1;
                long db = ib >= 0 ? b.Shape.dimensions[ib] : 1;
                dims[i] = da == 1 ? db : da;
                if (dims[i] == 0)
                    return false;
            }

            loopLayout = Shape.NpyIterOutputShape(dims, new[] { LoopPart(a, aLoop), LoopPart(b, bLoop) });
            var order = loopLayout.DenseAxisOrder();
            for (int i = 0; i < loopNd; i++)
            {
                if (order[i] != i)
                {
                    perm = order;
                    break;
                }
            }

            return true;
        }

        /// <summary>An input's loop axes as a shape for the vote: its leading dims and strides.</summary>
        /// <param name="x">The input, its core axes last.</param>
        /// <param name="loopNd">Its number of loop axes.</param>
        /// <returns>A shape of the first <paramref name="loopNd"/> axes (0-d when the input has none, so it abstains
        ///     everywhere).</returns>
        private static Shape LoopPart(NDArray x, int loopNd)
        {
            var dims = new long[loopNd];
            var strides = new long[loopNd];
            Array.Copy(x.Shape.dimensions, dims, loopNd);
            Array.Copy(x.Shape.strides, strides, loopNd);
            return Shape.Adopt(dims, strides);
        }

        /// <summary>
        ///     A view of an input with its loop axes in <paramref name="perm"/>'s order and its core axes still last, so
        ///     the kernels, which write C order, write the product in NumPy's memory order for the loop axes.
        /// </summary>
        /// <param name="x">The input, its core axes last (after any <c>dtype=</c> cast).</param>
        /// <param name="core">The number of its trailing core axes.</param>
        /// <param name="loopNd">The number of broadcast loop axes; the input's loop axes are right-aligned against them,
        ///     and an axis it lacks becomes a length-1 axis.</param>
        /// <param name="perm">The broadcast loop axes in memory order, outermost first (<see cref="GufuncLoopOrder"/>).</param>
        /// <returns><paramref name="x"/> itself when it has no loop axes (it broadcasts against all of them, and a 1-D
        ///     matmul operand must stay 1-D); otherwise a view (the gufunc member's scope reclaims it).</returns>
        private static NDArray PermuteLoopAxes(NDArray x, int core, int loopNd, int[] perm)
        {
            int xLoop = x.ndim - core;
            if (xLoop == 0)
                return x;

            var src = x.Shape;
            var dims = new long[loopNd + core];
            var strides = new long[loopNd + core];
            for (int p = 0; p < loopNd; p++)
            {
                int k = perm[p] - (loopNd - xLoop);
                dims[p] = k < 0 ? 1 : src.dimensions[k];
                strides[p] = k < 0 ? 0 : src.strides[k];
            }

            for (int c = 0; c < core; c++)
            {
                dims[loopNd + c] = src.dimensions[xLoop + c];
                strides[loopNd + c] = src.strides[xLoop + c];
            }

            return new NDArray(x.Storage.Alias(new Shape(dims, strides, src.offset, src.bufferSize)))
            {
                TensorEngine = x.TensorEngine
            };
        }

        /// <summary>
        ///     Relabels a product computed over <see cref="PermuteLoopAxes"/>' views (or over the inputs themselves, when
        ///     the vote gave C order) with NumPy's layout: the loop axes back in their order with the strides of
        ///     <paramref name="loopLayout"/> (scaled by the core block), the core axes innermost in C order. A product
        ///     written in that memory order is relabelled in place and keeps owning its buffer; any other is copied
        ///     into the layout (<see cref="NDArray.AdoptLayout"/>).
        /// </summary>
        /// <param name="result">The product: its loop axes in the voted memory order, then its core axes. Consumed.</param>
        /// <param name="loopLayout">The loop layout <see cref="GufuncLoopOrder"/> returned.</param>
        /// <returns>The product in NumPy's layout.</returns>
        private static NDArray AdoptLoopLayout(NDArray result, Shape loopLayout)
        {
            int loopNd = loopLayout.NDim, nd = result.ndim;
            var dims = new long[nd];
            var strides = new long[nd];
            long coreSize = 1;
            for (int c = nd - 1; c >= loopNd; c--)
            {
                dims[c] = result.Shape.dimensions[c];
                strides[c] = coreSize;
                coreSize *= dims[c];
            }

            for (int i = 0; i < loopNd; i++)
            {
                dims[i] = loopLayout.dimensions[i];
                strides[i] = loopLayout.strides[i] * coreSize;
            }

            return NDArray.AdoptLayout(result, Shape.Adopt(dims, strides));
        }

        /// <summary>
        ///     matmul's <c>order=</c>: the memory order NumPy's iterator allocates the output in, over the ITERATOR's
        ///     axes (the loop axes, then the core axes) — before an <c>axes=</c> output entry remaps them. <c>'C'</c>
        ///     is C order, <c>'F'</c> F order, <c>'A'</c> F order when both inputs are F-contiguous (a 1-D input is)
        ///     and C order otherwise, <c>'K'</c> the inputs' K order (<see cref="AdoptLoopLayout"/>, already applied).
        /// </summary>
        /// <param name="result">The product in iterator order (loop axes, then core axes), as the kernel wrote it: a fresh
        ///     C-ordered owner for every order but <c>'K'</c>. Consumed.</param>
        /// <param name="order">The validated order letter.</param>
        /// <param name="x1">The left operand as the caller passed it (its layout decides <c>'A'</c>).</param>
        /// <param name="x2">The right operand as the caller passed it.</param>
        /// <returns>The product with NumPy's exact strides for the order (an extent-1 axis included): the product itself
        ///     for <c>'K'</c>, <c>'C'</c> and a 0-d product, relabelled when it is already F-ordered, else its F copy.</returns>
        /// <remarks>
        ///     Probed 2.4.2: <c>np.matmul(a, b, axes=[(1,2),(1,2),(0,1)], order=o)</c> of two C-ordered (2,3,4) /
        ///     (2,4,5) arrays has strides (40,8,120) for <c>'C'</c> and (16,48,8) for <c>'F'</c> (neither is
        ///     C- or F-contiguous: the order applies before the remap); <c>np.matmul(np.arange(2.), C(1,2,3),
        ///     order='F')</c> has shape (1,3) and strides (8,8), where a C-ordered (1,3) array (also F-contiguous by its
        ///     flags) has (24,8).
        /// </remarks>
        private static NDArray ApplyMatmulOrder(NDArray result, char order, NDArray x1, NDArray x2)
        {
            // A 0-d product (two vectors) is C- and F-contiguous at once, and NumPy's iterator allocates the 0-d output
            // whatever the order (probed 2.4.2: np.matmul(a, a, order=o) of two vectors is a scalar for every o).
            if (result.ndim == 0)
                return result;

            bool fortran = order == 'F' || (order == 'A' && x1.Shape.IsFContiguous && x2.Shape.IsFContiguous);
            if (!fortran)
                return result;   // 'C': the kernels allocate exact C strides; 'K': the loop layout is already NumPy's

            var layout = Shape.Adopt((long[])result.Shape.dimensions.Clone(), 'F');
            // A product that is C- and F-contiguous at once has at most one axis longer than 1, so C and F order
            // address its elements alike: relabel it with exact F strides (NumPy's for its extent-1 axes) and keep the
            // buffer. Any other product is copied into the F layout.
            if (result.Shape.IsFContiguous && result.Shape.IsContiguous)
                return NDArray.AdoptLayout(result, layout);
            return result.copy('F');
        }
    }
}
