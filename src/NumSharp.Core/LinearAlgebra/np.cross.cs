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
        /// <exception cref="IncorrectShapeException">The leading axes do not broadcast: NumPy's <c>np.broadcast</c> text
        ///     (<c>shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (2,)
        ///     and arg 1 with shape (3,).</c>), raised before the dtype promotion.</exception>
        /// <exception cref="AxisError">When an axis is out of bounds for its operand (<c>axisc</c> against the result's
        ///     rank, checked right after the broadcast).</exception>
        /// <exception cref="TypeError">A text operand: NumPy multiplies the components in the promoted dtype, which has no
        ///     string loop (checked after the shape checks).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.
        /// </remarks>
        [NDScoped]
        public static NDArray cross(NDArray a, NDArray b, int axisa = -1, int axisb = -1, int axisc = -1,
            int? axis = null)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a, b))
                return NDMatrixBoundary.Plain((a, b, axisa, axisb, axisc, axis), static s => cross(s.a, s.b, s.axisa, s.axisb, s.axisc, s.axis));
            // a numpy.char.chararray operand too: NumPy's asarray drops either class (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a, b))
                return NDCharArrayBoundary.Plain((a, b, axisa, axisb, axisc, axis), static s => cross(s.a, s.b, s.axisa, s.axisb, s.axisc, s.axis));
            return np.EngineOf(a).Cross(a, b, axisa, axisb, axisc, axis);
        }
    }
}
