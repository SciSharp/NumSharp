// NumPy's np.cross (numpy/_core/numeric.py) as DefaultEngine's member: NumPy's statements, a composition over
// moveaxis, multiply, negative, subtract and concatenate - each the member its np.* function dispatches to (this
// engine, the operand's engine in the static StackCross, the array object's concatenation and expand_dims, Shape's
// broadcast of the leading shapes, the dtype layer's promotion).

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the moved axes, the cast operands, the component views and their products are this call's own
        public override NDArray Cross(NDArray a, NDArray b, int axisa = -1, int axisb = -1, int axisc = -1, int? axis = null)
        {
            if (axis is not null)
                axisa = axisb = axisc = axis.Value;

            if (a.ndim < 1 || b.ndim < 1)
                throw new ValueError("At least one array has zero dimension");

            // Check axisa and axisb are within bounds, then move the working axis to the end.
            int normA = NormalizeCrossAxis(axisa, a.ndim, "axisa");
            int normB = NormalizeCrossAxis(axisb, b.ndim, "axisb");
            a = MoveAxis(a, new[] { normA }, new[] { -1 });   // np.moveaxis(a, normA, -1)'s statement
            b = MoveAxis(b, new[] { normB }, new[] { -1 });

            long aLen = a.Shape.dimensions[a.ndim - 1];
            long bLen = b.Shape.dimensions[b.ndim - 1];
            if ((aLen != 2 && aLen != 3) || (bLen != 2 && bLen != 3))
                throw new ValueError(
                    "incompatible dimensions for cross product\n(dimension must be 2 or 3)");
            // NumPy warns on 2-vectors (deprecated 2.0); NumSharp does not model warnings.

            // NumPy creates the output before it touches a value: `shape = broadcast(a[..., 0], b[..., 0]).shape`, so
            // leading axes that do not broadcast raise np.broadcast's own "shape mismatch: objects cannot be broadcast to
            // a single shape. ..." ValueError, and for a 3-vector result `axisc` is checked against that shape plus the
            // vector axis — both before the dtype promotion and the multiplies (a text operand meets them first too).
            // np.broadcast_shapes(...)'s last statement; its first two (no shapes -> (), a negative dimension -> ValueError)
            // cannot fire for two arrays' shapes.
            Shape leading = Shape.BroadcastShape(LeadingShape(a), LeadingShape(b));
            if (aLen == 3 || bLen == 3)
                NormalizeCrossAxis(axisc, leading.NDim + 1, "axisc");

            // recast both operands to the promoted dtype, exactly as NumPy does, so every
            // intermediate product/difference runs in that dtype (integer inputs stay integer, etc.).
            // NumPy casts both operands to their promoted dtype and multiplies components into the result: with text
            // the multiply has no loop, named with the promoted descriptor on both sides.
            if (Backends.Strings.NDStringUfuncs.IsText(a) || Backends.Strings.NDStringUfuncs.IsText(b))
            {
                // promote_types refuses StringDType beside a number or bytes (DTypePromotionError); beside U it is StringDType.
                var promoted = DTypePromotion.PromoteTypes(a.dtype, b.dtype);
                throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", promoted, promoted);
            }
            NPTypeCode dtype = DTypePromotion.PromoteTypes(a.typecode, b.typecode);
            a = a.astype(dtype);
            b = b.astype(dtype);

            // Component slices along the (now last) axis. A 2-vector has no [..., 2] component; its
            // contribution to the formula is zero, reproduced by NumPy's branch structure below.
            NDArray a0 = a["..., 0"], a1 = a["..., 1"];
            NDArray b0 = b["..., 0"], b1 = b["..., 1"];

            // The products below are np.multiply / np.subtract / np.negative's statements (this engine's Multiply, Subtract
            // and Negate). np.negative also copies an F-contiguous operand's result into F order when the result is not
            // F already; Negate allocates such a result in F order, so that copy never fires.
            if (aLen == 2)
            {
                if (bLen == 2)
                    // both 2-D: only the z-component, returned as a scalar (per leading position).
                    return Subtract(Multiply(a0, b1), Multiply(a1, b0));

                // a is 2-D (a2 == 0), b is 3-D.
                NDArray b2 = b["..., 2"];
                return StackCross(
                    Multiply(a1, b2),                              // a1*b2 - 0
                    Negate(Multiply(a0, b2)),                    // 0 - a0*b2
                    Subtract(Multiply(a0, b1), Multiply(a1, b0)),  // a0*b1 - a1*b0
                    axisc);
            }

            {
                // a is 3-D.
                NDArray a2 = a["..., 2"];
                if (bLen == 3)
                {
                    NDArray b2 = b["..., 2"];
                    return StackCross(
                        Subtract(Multiply(a1, b2), Multiply(a2, b1)),
                        Subtract(Multiply(a2, b0), Multiply(a0, b2)),
                        Subtract(Multiply(a0, b1), Multiply(a1, b0)),
                        axisc);
                }

                // a is 3-D, b is 2-D (b2 == 0).
                return StackCross(
                    Negate(Multiply(a2, b1)),                    // 0 - a2*b1
                    Multiply(a2, b0),                              // a2*b0 - 0
                    Subtract(Multiply(a0, b1), Multiply(a1, b0)),  // a0*b1 - a1*b0
                    axisc);
            }
        }

        /// <summary>
        ///     Assembles the three 3-vector components into an array whose component axis lands at
        ///     <paramref name="axisc"/>. Each component shares the broadcast leading shape, so the
        ///     component axis is built at the END (via <c>np.expand_dims</c> + <c>np.concatenate</c>,
        ///     NOT <c>np.stack</c>, whose <c>atleast_1d</c> would turn 0-d components into <c>(3,1)</c>)
        ///     and then moved to <paramref name="axisc"/>.
        /// </summary>
        /// <param name="cp0">The x components (the broadcast leading shape).</param>
        /// <param name="cp1">The y components.</param>
        /// <param name="cp2">The z components.</param>
        /// <param name="axisc">The result axis for the 3-vectors (negative counts from the end).</param>
        /// <returns>The stacked result, the component axis at <paramref name="axisc"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axisc"/> out of range for the result's rank.</exception>
        [NDScopedCovered] // only caller: [NDScoped] DefaultEngine.Cross (the expand_dims wrappers + pre-moveaxis stack are scope-reclaimed)
        private static NDArray StackCross(NDArray cp0, NDArray cp1, NDArray cp2, int axisc)
        {
            int outNdim = cp0.ndim + 1;                         // len(broadcast_shape + (3,))
            int normC = NormalizeCrossAxis(axisc, outNdim, "axisc");

            // np.concatenate / np.expand_dims / np.moveaxis's statements: the array object's concatenation (its default
            // casting) and axis insertion, and the stack's engine MoveAxis.
            var stacked = NDArray.concatenate(new[]
            {
                NDArray.expand_dims(cp0, -1),
                NDArray.expand_dims(cp1, -1),
                NDArray.expand_dims(cp2, -1)
            }, -1, null, null, "same_kind");
            return stacked.TensorEngine.MoveAxis(stacked, new[] { -1 }, new[] { normC });
        }

        /// <summary>
        ///     The shape of <c>x[..., 0]</c>: every axis but the (moved-to-last) vector axis — the operand NumPy's
        ///     <c>broadcast(a[..., 0], b[..., 0])</c> reads, without building the view.
        /// </summary>
        /// <param name="x">An operand whose vector axis is already last (at least 1-D).</param>
        /// <returns>The leading shape (a scalar shape for a 1-D operand).</returns>
        private static Shape LeadingShape(NDArray x)
        {
            var dims = x.Shape.dimensions;
            if (dims.Length == 1)
                return Shape.NewScalar();
            var leading = new long[dims.Length - 1];
            System.Array.Copy(dims, leading, leading.Length);
            return Shape.Adopt(leading);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim, msg_prefix)</c>: normalizes a negative axis
        ///     and, on an out-of-range value, raises <see cref="AxisError"/> reporting the ORIGINAL
        ///     axis with the operand-naming prefix (<c>axisa</c>/<c>axisb</c>/<c>axisc</c>).
        /// </summary>
        /// <param name="axis">The axis as given (negative counts from the end).</param>
        /// <param name="ndim">The rank it indexes.</param>
        /// <param name="prefix">The message prefix naming the argument.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range.</exception>
        private static int NormalizeCrossAxis(int axis, int ndim, string prefix)
        {
            int resolved = axis < 0 ? axis + ndim : axis;
            if (resolved < 0 || resolved >= ndim)
                throw new AxisError($"{prefix}: axis {axis} is out of bounds for array of dimension {ndim}");
            return resolved;
        }
    }
}
