using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The gufunc signature <c>np.matmul</c> matches its core dimensions against. The <c>?</c>
        ///     marks OPTIONAL core dimensions: a 1-D operand drops its <c>n?</c>/<c>m?</c>, which is
        ///     how <c>matmul</c> promotes a vector operand.
        /// </summary>
        internal const string MatmulSignature = "(n?,k),(k,m?)->(n?,m?)";

        /// <summary>
        ///     Matrix product of two arrays — the gufunc <c>(n?,k),(k,m?)-&gt;(n?,m?)</c>, with NumPy's
        ///     full keyword surface.
        /// </summary>
        /// <param name="x1">Lhs input array, scalars not allowed.</param>
        /// <param name="x2">Rhs input array, scalars not allowed.</param>
        /// <param name="out">
        ///     Where to deposit the answer; returned as-is when given. Like a ufunc's <c>out</c> (and
        ///     unlike <see cref="dot"/>'s strict one) it may be strided and takes a cast from the
        ///     product dtype under <paramref name="casting"/>.
        /// </param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{x1, x2, out}</c>. A 2-D operand
        ///     names TWO axes, a 1-D operand ONE (its optional core dim is absent); the output entry
        ///     may be omitted only for the 1-D·1-D product, whose result has no core axes. Cannot be
        ///     combined with <paramref name="axis"/>.
        /// </param>
        /// <param name="axis">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for any value, because
        ///     <c>matmul</c>'s signature has three DISTINCT core dimensions — use <paramref name="axes"/>.
        /// </param>
        /// <param name="keepdims">
        ///     Present for signature parity only. NumPy raises <c>TypeError</c> for ANY value (True OR
        ///     False), because its output has core dimensions. Modelled with a <c>bool?</c> sentinel so
        ///     that, like NumPy's <c>np._NoValue</c> default, an explicit <c>false</c> also rejects.
        /// </param>
        /// <param name="dtype">Selects the LOOP: the product runs at this dtype, not merely the result.</param>
        /// <param name="casting">
        ///     Casting rule (default <c>"same_kind"</c>, the ufunc default) gating BOTH the input→loop
        ///     cast a <paramref name="dtype"/> forces and the product→<paramref name="out"/> cast.
        /// </param>
        /// <param name="order">Memory layout of the result — <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
        ///     <para>
        ///     The product itself is unchanged — <c>TensorEngine.Matmul</c>, which routes
        ///     float32/float64/complex128 through OpenBLAS when <c>NumSharp.Interop.OpenBLAS</c> is
        ///     referenced (byte-identical to NumPy) and through the managed GEMM otherwise. A
        ///     <paramref name="dtype"/> request is applied by casting the operands to that dtype BEFORE
        ///     the product, so the loop — and thus the backend route — follows it.
        ///     </para>
        ///     <para>
        ///     Like the five sibling product gufuncs, NumPy's remaining ufunc keywords <c>subok</c> and
        ///     <c>signature</c> are not modelled (<c>signature</c> is what <paramref name="dtype"/>
        ///     does; <c>subok</c> concerns ndarray subclasses NumSharp does not have). A
        ///     <paramref name="out"/> with a wrong CORE dim reports NumPy's verbatim core-dimension
        ///     message; extra leading LOOP dims broadcast (replicating the product, as NumPy does), and
        ///     the rarer genuine loop-dimension mismatch raises <c>copyto</c>'s broadcast
        ///     <c>ValueError</c> whose wording differs from NumPy's leaked iterator text (the same
        ///     latitude the siblings take).
        ///     </para>
        /// </remarks>
        public static NDArray matmul(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool? keepdims = null, NPTypeCode? dtype = null, string casting = "same_kind",
            char order = 'K')
        {
            RequireOrder(order);
            RequireCasting(casting);
            GufuncGuard.RejectAxisWithAxes(axes, axis);

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

            // A dtype= request selects the loop: validate each input reaches it under `casting`, then
            // cast so the product (and the backend route) run at that dtype.
            if (dtype.HasValue)
            {
                ValidateMatmulCast(a.typecode, dtype.Value, casting, "input 0");
                ValidateMatmulCast(b.typecode, dtype.Value, casting, "input 1");
                a = a.astype(dtype.Value, copy: false);
                b = b.astype(dtype.Value, copy: false);
            }

            var result = a.TensorEngine.Matmul(a, b);

            if (outputAxes.Length > 0)
                result = PlaceOutputCore(result, outputAxes);

            result = ApplyMatmulOrder(result, order, x1, x2);

            if (@out is null)
                return result;

            // out cast is validated BEFORE the shape (probed order), then the shape, then the copy.
            ValidateMatmulCast(result.typecode, @out.typecode, casting, "output");
            ValidateMatmulOutputShape(result, @out, outputCore);
            np.copyto(@out, result, "unsafe");   // cast already validated
            return @out;
        }

        /// <summary>Moves an operand's core axes to its trailing positions (the kernel's canonical layout).</summary>
        private static NDArray BringCoreToEnd(NDArray operand, int[] coreAxes)
        {
            var dest = new int[coreAxes.Length];
            for (int i = 0; i < coreAxes.Length; i++)
                dest[i] = operand.ndim - coreAxes.Length + i;
            return moveaxis(operand, coreAxes, dest);
        }

        /// <summary>
        ///     Relocates the product's core axes (currently trailing) to the positions <c>axes[2]</c>
        ///     named, validated against the OUTPUT rank (unknown until the product ran).
        /// </summary>
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

            var src = new int[outputAxes.Length];
            for (int i = 0; i < outputAxes.Length; i++)
                src[i] = result.ndim - outputAxes.Length + i;
            return moveaxis(result, src, dest);
        }

        /// <summary>
        ///     NumPy's per-entry "axes item {i} has value {v} repeated" (a <c>ValueError</c> with NO
        ///     name prefix), checked on the NORMALIZED input axes and the raw output axes, in order.
        /// </summary>
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
        private static NDArray ApplyMatmulOrder(NDArray result, char order, NDArray x1, NDArray x2)
        {
            switch (order)
            {
                case 'C':
                    return ascontiguousarray(result);
                case 'F':
                    return asfortranarray(result);
                case 'A':
                    return (x1.Shape.IsFContiguous && x2.Shape.IsFContiguous)
                        ? asfortranarray(result)
                        : ascontiguousarray(result);
                default:   // 'K'
                    return result;
            }
        }

        /// <summary>A ufunc cast gate: reports NumPy's verbatim "Cannot cast ufunc 'matmul' {role} …".</summary>
        private static void ValidateMatmulCast(NPTypeCode from, NPTypeCode to, string casting, string role)
        {
            if (from == to || np.can_cast(from, to, casting))
                return;
            throw new ArgumentException(
                $"Cannot cast ufunc 'matmul' {role} from dtype('{from.AsNumpyDtypeName()}') to " +
                $"dtype('{to.AsNumpyDtypeName()}') with casting rule '{casting}'");
        }

        /// <summary>
        ///     Validates <paramref name="out"/>'s CORE dims (the trailing <paramref name="outputCore"/>)
        ///     against the product's — they must match exactly, reporting NumPy's verbatim
        ///     core-dimension message on mismatch. The leading LOOP dims are left to <c>copyto</c>,
        ///     which broadcasts the product into <paramref name="out"/> (so an out with extra leading
        ///     dims replicates the product, exactly as NumPy does) and raises on a real loop mismatch.
        /// </summary>
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
                long resDim = result.shape[rn - outputCore + c];
                long outDim = @out.shape[on - outputCore + c];
                if (outDim != resDim)
                    throw new ValueError(
                        $"matmul: Output operand 0 has a mismatch in its core dimension {c}, " +
                        $"with gufunc signature {MatmulSignature} (size {outDim} is different from {resDim})");
            }
        }
    }
}
