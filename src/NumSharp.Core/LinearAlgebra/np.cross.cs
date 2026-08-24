namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the cross product of two (arrays of) vectors.
        ///     <para>
        ///     The cross product of <paramref name="a"/> and <paramref name="b"/> in R^3 is a vector
        ///     perpendicular to both. Vectors are taken along <paramref name="axisa"/>/<paramref name="axisb"/>
        ///     (the last axis by default) and may have length 2 or 3 — where a vector's length is 2 its
        ///     missing third component is treated as zero. When BOTH inputs are 2-vectors the scalar
        ///     z-component is returned; otherwise the 3-vector result is placed along <paramref name="axisc"/>.
        ///     </para>
        /// </summary>
        /// <param name="a">Components of the first vector(s).</param>
        /// <param name="b">Components of the second vector(s).</param>
        /// <param name="axisa">Axis of <paramref name="a"/> that defines the vector(s). By default, the last axis.</param>
        /// <param name="axisb">Axis of <paramref name="b"/> that defines the vector(s). By default, the last axis.</param>
        /// <param name="axisc">
        ///     Axis of the result containing the cross product vector(s). Ignored when both inputs are
        ///     2-D vectors (the return is then scalar). By default, the last axis.
        /// </param>
        /// <param name="axis">
        ///     If given, the single axis of <paramref name="a"/>, <paramref name="b"/> and the result
        ///     that defines the vector(s). Overrides <paramref name="axisa"/>/<paramref name="axisb"/>/<paramref name="axisc"/>.
        /// </param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.cross.html
        ///     <para>
        ///     A line-for-line port of NumPy's <c>numpy/_core/numeric.py</c> <c>cross</c>: a pure
        ///     composition over <see cref="multiply"/>/<see cref="negative"/>/<see cref="subtract"/>/
        ///     <see cref="moveaxis"/> (no new kernel), so it supports full broadcasting of the leading
        ///     axes and its result dtype is <c>promote_types(a.dtype, b.dtype)</c>. The 2-vector forms
        ///     are DEPRECATED in NumPy 2.0 but still compute; NumSharp does not model that warning and
        ///     simply returns the value. Distinct from <see cref="linalg.cross"/>, the Array-API form,
        ///     which accepts 3-vectors only and takes a single <c>axis</c>.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">
        ///     When either input has zero dimensions, or a vector's length is neither 2 nor 3.
        /// </exception>
        /// <exception cref="AxisError">When an axis is out of bounds for its operand.</exception>
        [NDScoped]
        public static NDArray cross(NDArray a, NDArray b, int axisa = -1, int axisb = -1, int axisc = -1,
            int? axis = null)
        {
            if (axis is not null)
                axisa = axisb = axisc = axis.Value;

            if (a.ndim < 1 || b.ndim < 1)
                throw new ValueError("At least one array has zero dimension");

            // Check axisa and axisb are within bounds, then move the working axis to the end.
            int normA = NormalizeCrossAxis(axisa, a.ndim, "axisa");
            int normB = NormalizeCrossAxis(axisb, b.ndim, "axisb");
            a = moveaxis(a, normA, -1);
            b = moveaxis(b, normB, -1);

            long aLen = a.shape[a.ndim - 1];
            long bLen = b.shape[b.ndim - 1];
            if ((aLen != 2 && aLen != 3) || (bLen != 2 && bLen != 3))
                throw new ValueError(
                    "incompatible dimensions for cross product\n(dimension must be 2 or 3)");
            // NumPy warns on 2-vectors (deprecated 2.0); NumSharp does not model warnings.

            // recast both operands to the promoted dtype, exactly as NumPy does, so every
            // intermediate product/difference runs in that dtype (integer inputs stay integer, etc.).
            NPTypeCode dtype = promote_types(a.typecode, b.typecode);
            a = a.astype(dtype);
            b = b.astype(dtype);

            // Component slices along the (now last) axis. A 2-vector has no [..., 2] component; its
            // contribution to the formula is zero, reproduced by NumPy's branch structure below.
            NDArray a0 = a["..., 0"], a1 = a["..., 1"];
            NDArray b0 = b["..., 0"], b1 = b["..., 1"];

            if (aLen == 2)
            {
                if (bLen == 2)
                    // both 2-D: only the z-component, returned as a scalar (per leading position).
                    return subtract(multiply(a0, b1), multiply(a1, b0));

                // a is 2-D (a2 == 0), b is 3-D.
                NDArray b2 = b["..., 2"];
                return StackCross(
                    multiply(a1, b2),                              // a1*b2 - 0
                    negative(multiply(a0, b2)),                    // 0 - a0*b2
                    subtract(multiply(a0, b1), multiply(a1, b0)),  // a0*b1 - a1*b0
                    axisc);
            }

            {
                // a is 3-D.
                NDArray a2 = a["..., 2"];
                if (bLen == 3)
                {
                    NDArray b2 = b["..., 2"];
                    return StackCross(
                        subtract(multiply(a1, b2), multiply(a2, b1)),
                        subtract(multiply(a2, b0), multiply(a0, b2)),
                        subtract(multiply(a0, b1), multiply(a1, b0)),
                        axisc);
                }

                // a is 3-D, b is 2-D (b2 == 0).
                return StackCross(
                    negative(multiply(a2, b1)),                    // 0 - a2*b1
                    multiply(a2, b0),                              // a2*b0 - 0
                    subtract(multiply(a0, b1), multiply(a1, b0)),  // a0*b1 - a1*b0
                    axisc);
            }
        }

        /// <summary>
        ///     Assembles the three 3-vector components into an array whose component axis lands at
        ///     <paramref name="axisc"/>. Each component shares the broadcast leading shape, so the
        ///     component axis is built at the END (via <see cref="expand_dims"/> + <see cref="concatenate"/>,
        ///     NOT <c>np.stack</c>, whose <c>atleast_1d</c> would turn 0-d components into <c>(3,1)</c>)
        ///     and then moved to <paramref name="axisc"/>.
        /// </summary>
        private static NDArray StackCross(NDArray cp0, NDArray cp1, NDArray cp2, int axisc)
        {
            int outNdim = cp0.ndim + 1;                         // len(broadcast_shape + (3,))
            int normC = NormalizeCrossAxis(axisc, outNdim, "axisc");

            var stacked = concatenate(new[]
            {
                expand_dims(cp0, -1),
                expand_dims(cp1, -1),
                expand_dims(cp2, -1)
            }, -1);
            return moveaxis(stacked, -1, normC);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim, msg_prefix)</c>: normalizes a negative axis
        ///     and, on an out-of-range value, raises <see cref="AxisError"/> reporting the ORIGINAL
        ///     axis with the operand-naming prefix (<c>axisa</c>/<c>axisb</c>/<c>axisc</c>).
        /// </summary>
        private static int NormalizeCrossAxis(int axis, int ndim, string prefix)
        {
            int resolved = axis < 0 ? axis + ndim : axis;
            if (resolved < 0 || resolved >= ndim)
                throw new AxisError($"{prefix}: axis {axis} is out of bounds for array of dimension {ndim}");
            return resolved;
        }
    }
}
