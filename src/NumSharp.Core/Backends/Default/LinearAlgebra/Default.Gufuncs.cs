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

            // A dtype= request selects the loop (its casts validated above): cast so the product (and the backend route)
            // run at that dtype. astype converts a fixed-width text operand too, which only casting='unsafe' lets
            // through.
            if (dtype is not null)
            {
                a = a.astype(dtype, copy: false);
                b = b.astype(dtype, copy: false);
            }

            var result = Matmul(a, b);

            if (outputAxes.Length > 0)
                result = PlaceOutputCore(result, outputAxes);

            result = ApplyMatmulOrder(result, order, x1, x2);

            if (@out is null)
                return result;

            // The out cast and the out's core dimensions were validated before the product.
            np.copyto(@out, result, "unsafe");
            return @out;
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

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = Vecdot(a, b);

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

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = Matvec(a, b);

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

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = Vecmat(a, b);

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
        ///     computed (C) memory order, relabelled, so it still owns its data — NumPy allocates a gufunc output in the
        ///     iterator's order and returns it with the remapped axes (probed 2.4.2:
        ///     <c>np.matmul(ones((2,3,4)), ones((2,4,5)), axes=[(1,2),(1,2),(0,1)])</c> has shape (3,5,2), strides
        ///     (40,8,120), OWNDATA True).</returns>
        /// <exception cref="AxisError">An output axis out of the result's range.</exception>
        private NDArray PlaceOutputCore(NDArray result, int[] outputAxes)
        {
            var dest = new int[outputAxes.Length];
            for (int i = 0; i < outputAxes.Length; i++)
            {
                int ax = outputAxes[i] < 0 ? outputAxes[i] + result.ndim : outputAxes[i];
                if (ax < 0 || ax >= result.ndim)
                    throw new AxisError(outputAxes[i], result.ndim);
                dest[i] = ax;
            }

            var src = new int[outputAxes.Length];
            for (int i = 0; i < outputAxes.Length; i++)
                src[i] = result.ndim - outputAxes.Length + i;
            // The moved view's shape is the product's strides permuted; relabel the product with it.
            using var moved = MoveAxis(result, src, dest);
            return NDArray.AdoptLayout(result, moved.Shape);
        }

        /// <summary>
        ///     matmul's <c>order=</c> layout: <c>'F'</c> → column-major, <c>'A'</c> → column-major only
        ///     when BOTH inputs are, else C, and <c>'C'</c>/<c>'K'</c> → the product's natural
        ///     C-contiguous output (so the default <c>'K'</c> is a no-op — the pre-keyword fast path).
        /// </summary>
        /// <param name="result">The product.</param>
        /// <param name="order">The validated order letter.</param>
        /// <param name="x1">The left operand as the caller passed it (its layout decides <c>'A'</c>).</param>
        /// <param name="x2">The right operand as the caller passed it.</param>
        /// <returns><paramref name="result"/> for <c>'K'</c>; else a C- or F-contiguous array (a copy when the layout
        ///     changes).</returns>
        private static NDArray ApplyMatmulOrder(NDArray result, char order, NDArray x1, NDArray x2)
        {
            switch (order)
            {
                case 'C':
                    return np.ascontiguousarray(result);
                case 'F':
                    return np.asfortranarray(result);
                case 'A':
                    return (x1.Shape.IsFContiguous && x2.Shape.IsFContiguous)
                        ? np.asfortranarray(result)
                        : np.ascontiguousarray(result);
                default:   // 'K'
                    return result;
            }
        }
    }
}
