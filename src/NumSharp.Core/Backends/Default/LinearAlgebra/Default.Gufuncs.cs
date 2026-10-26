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

            if (keepdims.HasValue)
                throw new TypeError(
                    $"matmul does not support keepdims: its signature {MatmulSignature} requires " +
                    "output 0 to have 2 core dimensions, but keepdims can only be used when all inputs " +
                    "have the same number of core dimensions and all outputs have no core dimensions.");

            if (axis.HasValue)
                throw new TypeError(
                    "matmul: axis can only be used with a single shared core dimension, not with the " +
                    $"3 distinct ones implied by signature {MatmulSignature}.");

            var a = x1;
            var b = x2;
            int[] outputAxes = Array.Empty<int>();

            // matmul's core RANKS depend on each operand's ndim (the n?/m? optional dims): a 1-D
            // operand carries a single core dim, a >=2-D operand two; the output carries one per
            // input that kept its optional dim (so 0 for the 1-D·1-D dot). Computed unconditionally
            // because the out= shape validator needs it to tell a core dim from a loop dim.
            int outputCore = (x1.ndim == 1 ? 0 : 1) + (x2.ndim == 1 ? 0 : 1);
            if (axes is not null)
            {
                int op0Core = x1.ndim == 1 ? 1 : 2;
                int op1Core = x2.ndim == 1 ? 1 : 2;

                // matmul's output entry is NEVER omittable — its signature output (n?,m?) is treated
                // as having core axes even when both optionals vanish (the 1-D·1-D dot), so all THREE
                // entries are required. NumPy's shared NormalizeAxes would wrongly allow a 2-entry
                // list when the effective output core rank is 0; guard that here (the output entry's
                // LENGTH is still validated against the effective rank, so `[(-1,),(-1,),()]` passes
                // for the dot while `[(-1,),(-1,)]` does not).
                if (axes.Length != 3)
                    throw new ValueError(
                        "axes should be a list with an entry for all 3 inputs and outputs; entries " +
                        "for outputs can only be omitted if none of them has core axes.");

                var resolved = GufuncGuard.NormalizeAxes("matmul", axes, new[] {op0Core, op1Core, outputCore}, x1, x2);
                RequireNoRepeatedAxes(resolved);
                outputAxes = resolved[2];

                // Bring each operand's core axes to the trailing positions the kernel expects; the
                // result's own core axes are relocated to `outputAxes` after the product runs.
                a = BringCoreToEnd(a, resolved[0]);
                b = BringCoreToEnd(b, resolved[1]);
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

            // The out cast was validated before the gufunc checks; the core dimensions, then the copy.
            ValidateMatmulOutputShape(result, @out, outputCore);
            np.copyto(@out, result, "unsafe");   // cast already validated
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
            GufuncGuard.RequireRank("vecdot", VecdotSignature, 0, x1, 1);
            GufuncGuard.RequireRank("vecdot", VecdotSignature, 1, x2, 1);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                // Under keepdims the output gains a core dimension (the reduced axis, kept as size 1),
                // so a PROVIDED output entry must have length 1 — but it may still be OMITTED. This is
                // the whole reason vecdot passes an output-core-rank override: `axes=[[1],[1]]` with
                // keepdims is legal and lands the kept axis at the end, `axes=[[1],[1],[]]` is not
                // (an explicit 0-length output entry contradicts the kept dimension).
                int outputCoreRank = keepdims ? 1 : 0;
                var resolved = GufuncGuard.NormalizeAxes("vecdot", axes, new[] {1, 1, 0}, x1, x2, outputCoreRank);
                outputAxes = resolved[2];

                a = MoveAxis(a, new[] {resolved[0][0]}, new[] {-1});
                b = MoveAxis(b, new[] {resolved[1][0]}, new[] {-1});
            }
            else if (axis is not null)
            {
                // NumPy allows a bare `axis` only where every core dimension is the SAME one — true
                // for vecdot's (n),(n)->() and for no other member of this family (matvec and vecmat
                // raise TypeError). It names the shared core axis in EACH operand, so both move it
                // to the end and the ordinary kernel runs. When the named axis IS already the last
                // one — the overwhelmingly common case, and what np.linalg.vecdot's axis=-1 default
                // always resolves to — the moveaxis is a pure no-op view rebuild, so skip it (the
                // validation still runs). This is what kept np.linalg.vecdot ~2x slower than a bare
                // np.vecdot on the same 1-D operands.
                int ra = RequireAxisInRange(axis.Value, a);
                if (ra != a.ndim - 1)
                    a = MoveAxis(a, new[] {ra}, new[] {-1});
                int rb = RequireAxisInRange(axis.Value, b);
                if (rb != b.ndim - 1)
                    b = MoveAxis(b, new[] {rb}, new[] {-1});
            }

            GufuncGuard.RequireCoreSize("vecdot", VecdotSignature, 1, 0, b.Shape.dimensions[b.ndim - 1], a.Shape.dimensions[a.ndim - 1]);

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

                result = np.expand_dims(result, keptAxis);
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
            GufuncGuard.RejectAxisAndKeepdims("matvec", MatvecSignature, axis, keepdims, 1);

            GufuncGuard.RequireRank("matvec", MatvecSignature, 0, x1, 2);
            GufuncGuard.RequireRank("matvec", MatvecSignature, 1, x2, 1);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                var resolved = GufuncGuard.NormalizeAxes("matvec", axes, new[] {2, 1, 1}, x1, x2);
                outputAxes = resolved[2];

                // Bring (m,n) and (n) to the trailing positions the kernel expects; the result's
                // own core axis goes back to where `axes` asked for it once the loop has run.
                a = MoveAxis(a, resolved[0], new[] {a.ndim - 2, a.ndim - 1});
                b = MoveAxis(b, new[] {resolved[1][0]}, new[] {-1});
            }

            GufuncGuard.RequireCoreSize("matvec", MatvecSignature, 1, 0,
                b.Shape.dimensions[b.ndim - 1], a.Shape.dimensions[a.ndim - 1]);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = Matvec(a, b);

            if (outputAxes.Length == 1)
                result = MoveAxis(result, new[] {-1}, new[] {outputAxes[0]});

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
            GufuncGuard.RejectAxisAndKeepdims("vecmat", VecmatSignature, axis, keepdims, 2);

            GufuncGuard.RequireRank("vecmat", VecmatSignature, 0, x1, 1);
            GufuncGuard.RequireRank("vecmat", VecmatSignature, 1, x2, 2);

            var a = x1;
            var b = x2;
            int[] outputAxes = System.Array.Empty<int>();

            if (axes is not null)
            {
                var resolved = GufuncGuard.NormalizeAxes("vecmat", axes, new[] {1, 2, 1}, x1, x2);
                outputAxes = resolved[2];

                a = MoveAxis(a, new[] {resolved[0][0]}, new[] {-1});
                b = MoveAxis(b, resolved[1], new[] {b.ndim - 2, b.ndim - 1});
            }

            GufuncGuard.RequireCoreSize("vecmat", VecmatSignature, 1, 0,
                b.Shape.dimensions[b.ndim - 2], a.Shape.dimensions[a.ndim - 1]);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = Vecmat(a, b);

            if (outputAxes.Length == 1)
                result = MoveAxis(result, new[] {-1}, new[] {outputAxes[0]});

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
        /// <param name="result">The product, its core axes last.</param>
        /// <param name="outputAxes">The output entry of <c>axes=</c>, as the caller gave it (negative axes count from the
        ///     end of the output).</param>
        /// <returns>A view with the core axes where <paramref name="outputAxes"/> names them.</returns>
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
            return MoveAxis(result, src, dest);
        }

        /// <summary>
        ///     NumPy's per-entry "axes item {i} has value {v} repeated" (a <c>ValueError</c> with NO
        ///     name prefix), checked on the NORMALIZED input axes and the raw output axes, in order.
        /// </summary>
        /// <param name="entries">The resolved <c>axes=</c> entries.</param>
        /// <exception cref="ValueError">An entry names one axis twice.</exception>
        private static void RequireNoRepeatedAxes(int[][] entries)
        {
            for (int i = 0; i < entries.Length; i++)
            {
                var entry = entries[i];
                for (int k = 0; k < entry.Length; k++)
                    for (int j = k + 1; j < entry.Length; j++)
                        if (entry[k] == entry[j])
                            throw new ValueError($"axes item {i} has value {entry[k]} repeated");
            }
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

        /// <summary>
        ///     Validates <paramref name="out"/>'s CORE dims (the trailing <paramref name="outputCore"/>)
        ///     against the product's — they must match exactly, reporting NumPy's verbatim
        ///     core-dimension message on mismatch. The leading LOOP dims are left to <c>copyto</c>,
        ///     which broadcasts the product into <paramref name="out"/> (so an out with extra leading
        ///     dims replicates the product, exactly as NumPy does) and raises on a real loop mismatch.
        /// </summary>
        /// <param name="result">The product.</param>
        /// <param name="out">The <c>out=</c> operand.</param>
        /// <param name="outputCore">How many trailing axes are the output's core axes.</param>
        /// <exception cref="ValueError">A core dimension of <paramref name="out"/> differs from the product's.</exception>
        private static void ValidateMatmulOutputShape(NDArray result, NDArray @out, int outputCore)
        {
            int rn = result.ndim, on = @out.ndim;
            // Only the TRAILING `outputCore` dims are the product's CORE dims — out must match them
            // EXACTLY (NumPy's verbatim core-dimension message on mismatch). The remaining leading
            // dims are LOOP dims the product BROADCASTS into, so out may carry any broadcast-compatible
            // loop shape — INCLUDING extra leading dims of any size, which NumPy fills by replicating
            // the product (probed: out=(5,2,2) for a (2,2) product is 5 copies). Those are left to
            // copyto (called next), which broadcasts the product into out and raises on a genuine
            // loop-dim mismatch. `on < outputCore` (out missing a core dim) also falls to copyto.
            if (on < outputCore)
                return;
            for (int c = 0; c < outputCore; c++)
            {
                long resDim = result.Shape.dimensions[rn - outputCore + c];
                long outDim = @out.Shape.dimensions[on - outputCore + c];
                if (outDim != resDim)
                    throw new ValueError(
                        $"matmul: Output operand 0 has a mismatch in its core dimension {c}, " +
                        $"with gufunc signature {MatmulSignature} (size {outDim} is different from {resDim})");
            }
        }

        /// <summary>NumPy's verbatim gufunc axis rejection.</summary>
        /// <param name="axis">The axis as the caller gave it (negative counts from the end).</param>
        /// <param name="operand">The operand it names an axis of.</param>
        /// <returns>The axis, normalized.</returns>
        /// <exception cref="AxisError">The axis is out of the operand's range.</exception>
        private static int RequireAxisInRange(int axis, NDArray operand)
        {
            int resolved = axis < 0 ? axis + operand.ndim : axis;
            if (resolved < 0 || resolved >= operand.ndim)
                throw new AxisError(axis, operand.ndim);
            return resolved;
        }
    }
}
