namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The gufunc signature <c>np.vecdot</c> matches its core dimensions against.
        /// </summary>
        internal const string VecdotSignature = "(n),(n)->()";

        /// <summary>
        ///     Vector dot product of two arrays (NumPy 2.0) — the gufunc <c>(n),(n)->()</c>,
        ///     conjugating the first operand.
        /// </summary>
        /// <param name="x1">First operand. Conjugated when complex.</param>
        /// <param name="x2">Second operand.</param>
        /// <param name="out">Where to deposit the answer. Returned as-is when given.</param>
        /// <param name="axes">
        ///     Which axes carry the core dimensions, per operand: <c>{x1, x2, out}</c>. The output
        ///     entry may be omitted here — <c>vecdot</c>'s output has no core axes. Cannot be
        ///     combined with <paramref name="axis"/>.
        /// </param>
        /// <param name="axis">
        ///     The shared core axis, in place of the default last one. Applied to both operands —
        ///     the special case of <paramref name="axes"/> that this signature admits because both
        ///     core dimensions are the SAME one.
        /// </param>
        /// <param name="keepdims">Leave the contracted axis in the result with length 1.</param>
        /// <param name="dtype">
        ///     Selects the LOOP: computation runs at this dtype, not merely the result.
        /// </param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.vecdot.html
        ///     <para>
        ///     Unlike <see cref="sum"/>, the accumulation happens in the LOOP dtype rather than
        ///     NEP50's wider accumulator: <c>vecdot</c>'s registered loops are <c>'ii->i'</c>, so an
        ///     int32 pair reduces to int32 where <c>np.sum</c> would give int64.
        ///     </para>
        ///     <para>
        ///     Core dimensions do NOT broadcast — a length-1 core axis against a length-3 one is an
        ///     error. Only the leading (loop) axes broadcast.
        ///     </para>
        ///     <para>
        ///     NumPy's remaining ufunc keywords — <c>casting</c>, <c>order</c>, <c>subok</c> and
        ///     <c>signature</c> — are not modelled anywhere in NumSharp's ufunc surface and so are
        ///     absent here too rather than accepted and ignored. <c>signature</c> is what
        ///     <paramref name="dtype"/> already does; <c>subok</c> concerns ndarray subclasses,
        ///     which NumSharp does not have.
        ///     </para>
        /// </remarks>
        public static NDArray vecdot(NDArray x1, NDArray x2, NDArray @out = null, int[][] axes = null,
            int? axis = null, bool keepdims = false, NPTypeCode? dtype = null)
        {
            GufuncGuard.RejectAxisWithAxes(axes, axis);
            GufuncGuard.RequireRank("vecdot", VecdotSignature, 0, x1, 1);
            GufuncGuard.RequireRank("vecdot", VecdotSignature, 1, x2, 1);

            var a = x1;
            var b = x2;

            if (axes is not null)
            {
                var resolved = GufuncGuard.NormalizeAxes("vecdot", axes, new[] {1, 1, 0}, x1, x2);
                if (keepdims)
                    throw new AxisError(
                        "vecdot: operand 2 has 1 core dimensions, but 0 dimensions are specified by axes tuple.");

                a = moveaxis(a, resolved[0][0], -1);
                b = moveaxis(b, resolved[1][0], -1);
            }
            else if (axis is not null)
            {
                // NumPy allows a bare `axis` only where every core dimension is the SAME one — true
                // for vecdot's (n),(n)->() and for no other member of this family (matvec and vecmat
                // raise TypeError). It names the shared core axis in EACH operand, so both move it
                // to the end and the ordinary kernel runs.
                a = moveaxis(a, RequireAxisInRange(axis.Value, a), -1);
                b = moveaxis(b, RequireAxisInRange(axis.Value, b), -1);
            }

            GufuncGuard.RequireCoreSize("vecdot", VecdotSignature, 1, 0, b.shape[b.ndim - 1], a.shape[a.ndim - 1]);

            a = GufuncGuard.ToLoop(a, dtype);
            b = GufuncGuard.ToLoop(b, dtype);

            var result = a.TensorEngine.Vecdot(a, b);

            if (keepdims)
            {
                // The contracted axis comes back where it was taken from, not merely at the end.
                result = expand_dims(result, -1);
                if (axis is not null)
                    result = moveaxis(result, -1, axis.Value);
            }

            return GufuncGuard.Deliver(result, @out);
        }

        /// <summary>NumPy's verbatim gufunc axis rejection.</summary>
        private static int RequireAxisInRange(int axis, NDArray operand)
        {
            int resolved = axis < 0 ? axis + operand.ndim : axis;
            if (resolved < 0 || resolved >= operand.ndim)
                throw new AxisError(axis, operand.ndim);
            return resolved;
        }
    }
}
