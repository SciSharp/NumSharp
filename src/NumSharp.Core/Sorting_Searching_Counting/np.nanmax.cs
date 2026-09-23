namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return maximum of an array or maximum along an axis, ignoring any NaNs — NumPy's <c>np.nanmax</c>, i.e.
        /// <c>np.fmax.reduce</c>, reproduced bit for bit: which zero sign survives a <c>±0</c> tie and which NaN an all-NaN
        /// slice returns (payload intact, never a canonical NaN) are the ones NumPy 2.4.2 returns for the same layout.
        /// </summary>
        /// <param name="a">Array containing numbers whose maximum is desired.</param>
        /// <param name="axis">Axis along which the maximum is computed (negative counts from the end). The default, null, computes the
        /// maximum of the flattened array.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one (for a flat reduction:
        /// shape <c>(1,)*a.ndim</c>).</param>
        /// <returns>A new array of <paramref name="a"/>'s dtype holding the maximum — a 0-d array for a flat reduction, the reduced
        /// shape for an axis one. A slice whose every element is NaN yields a NaN (NumPy also emits an "All-NaN slice
        /// encountered" RuntimeWarning, which NumSharp does not model).</returns>
        /// <remarks>
        /// <para>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html. Per dtype:
        /// <list type="bullet">
        /// <item>float32 / float64 run NumPy's own <c>fmax</c> reduction schedule — the vector op <c>maxp</c> inside each inner-loop
        /// call's vector section, the CRT <c>fmax</c> for its scalar tail — so a <c>±0</c> tie or an all-NaN slice can come back
        /// differently for two layouts of the same values, exactly as in NumPy.</item>
        /// <item>float16 and complex128 run NumPy's sequential loops: NaN elements are skipped (complex: a NaN in either part), a tie
        /// keeps the EARLIER element, an all-NaN slice returns its first NaN verbatim; complex compares lexicographically (real
        /// part, then imaginary).</item>
        /// <item>Integer, bool and char arrays cannot hold a NaN, so this is <see cref="amax(NDArray,int?,bool,DType)"/>;
        /// decimal likewise (no NumPy analog).</item>
        /// </list>
        /// </para>
        /// <para>
        /// Footgun: an EMPTY reduction raises (<c>fmax</c> has no identity) — it does not return NaN. A non-empty reduced axis of an
        /// otherwise empty array returns the empty result. A broadcast (stride-0) float input gets the right value, but its
        /// <c>±0</c> sign and NaN payload are not pinned to NumPy's.
        /// </para>
        /// </remarks>
        /// <exception cref="System.ArgumentException">The reduction is empty — NumPy's text, <c>zero-size array to reduction
        /// operation fmax which has no identity</c>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        public static NDArray nanmax(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanMax(a, axis, keepdims);
    }
}
