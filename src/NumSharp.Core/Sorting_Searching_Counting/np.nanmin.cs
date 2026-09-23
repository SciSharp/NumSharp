namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Return minimum of an array or minimum along an axis, ignoring any NaNs — NumPy's <c>np.nanmin</c>, i.e.
        /// <c>np.fmin.reduce</c>, reproduced bit for bit: which zero sign survives a <c>±0</c> tie and which NaN an all-NaN
        /// slice returns (payload intact, never a canonical NaN) are the ones NumPy 2.4.2 returns for the same layout.
        /// </summary>
        /// <param name="a">Array containing numbers whose minimum is desired.</param>
        /// <param name="axis">Axis along which the minimum is computed (negative counts from the end). The default, null, computes the
        /// minimum of the flattened array.</param>
        /// <param name="keepdims">If true, the reduced axes are left in the result as dimensions with size one (for a flat reduction:
        /// shape <c>(1,)*a.ndim</c>).</param>
        /// <returns>A new array of <paramref name="a"/>'s dtype holding the minimum — a 0-d array for a flat reduction, the reduced
        /// shape for an axis one. A slice whose every element is NaN yields a NaN (NumPy also emits an "All-NaN slice
        /// encountered" RuntimeWarning, which NumSharp does not model).</returns>
        /// <remarks>
        /// <para>
        /// https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html. Per dtype:
        /// <list type="bullet">
        /// <item>float32 / float64 run NumPy's own <c>fmin</c> reduction schedule — the vector op <c>minp</c> inside each inner-loop
        /// call's vector section, the CRT <c>fmin</c> for its scalar tail (whose tie keeps <c>-0</c> over <c>+0</c>) — so a
        /// <c>±0</c> tie or an all-NaN slice can come back differently for two layouts of the same values, exactly as in
        /// NumPy.</item>
        /// <item>float16 and complex128 run NumPy's sequential loops: NaN elements are skipped (complex: a NaN in either part), a tie
        /// keeps the EARLIER element, an all-NaN slice returns its first NaN verbatim; complex compares lexicographically (real
        /// part, then imaginary).</item>
        /// <item>Integer, bool and char arrays cannot hold a NaN, so this is <see cref="amin(NDArray,int?,bool,DType)"/>;
        /// decimal likewise (no NumPy analog).</item>
        /// </list>
        /// </para>
        /// <para>
        /// Footgun: an EMPTY reduction raises (<c>fmin</c> has no identity) — it does not return NaN. A non-empty reduced axis of an
        /// otherwise empty array returns the empty result. A broadcast (stride-0) float input gets the right value, but its
        /// <c>±0</c> sign and NaN payload are not pinned to NumPy's.
        /// </para>
        /// </remarks>
        /// <exception cref="System.ArgumentException">The reduction is empty — NumPy's text, <c>zero-size array to reduction
        /// operation fmin which has no identity</c>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        public static NDArray nanmin(NDArray a, int? axis = null, bool keepdims = false)
            => a.TensorEngine.NanMin(a, axis, keepdims);
    }
}
