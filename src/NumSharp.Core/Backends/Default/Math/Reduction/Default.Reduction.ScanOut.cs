namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     The <c>out=</c> store shared by <see cref="ReduceCumAdd"/> and <see cref="ReduceCumMul"/>: NumPy's
        ///     add.accumulate / multiply.accumulate require <paramref name="out"/> to have the natural output shape (any
        ///     layout) and store the result into it with UNSAFE casting (float to int truncates, complex to real drops the
        ///     imaginary part), returning <paramref name="out"/> itself.
        /// </summary>
        /// <param name="result">The accumulation, a fresh array of the loop dtype.</param>
        /// <param name="out">The caller's <c>out=</c>, or null.</param>
        /// <returns><paramref name="result"/> when <paramref name="out"/> is null, else <paramref name="out"/> after the
        ///     store.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (<c>output array is read-only</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have <paramref name="result"/>'s shape
        ///     (NumPy's <c>provided out is the wrong size for the accumulation.</c>; the house shape exception stands in for
        ///     NumPy's ValueError).</exception>
        /// <remarks>
        ///     Moved from <c>APIs/np.cumsum.cs</c> with the two scans (the np-layer dtype plan's P0 pilot). The store still
        ///     goes through <c>np.copyto</c>, a structural call into the facade that the plan's P6 replaces with an engine
        ///     copy; the up-call ratchet pins it with that reason.
        /// </remarks>
        internal static NDArray WriteScanToOut(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;
            // A read-only out refuses FIRST — probed 2.4.2: a wrong-sized read-only out reports
            // "output array is read-only", not the size error (accumulate validates writeability
            // ahead of the accumulation-shape check). Without the guard the copyto below raised
            // the generic assignment text instead of NumPy's ufunc-out wording.
            NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            if (!@out.Shape.Equals(result.Shape))
                // NumPy 2.4.2 verbatim (PyUFunc_Accumulate); house IncorrectShapeException stands
                // in for NumPy's ValueError per the long-standing shape-error convention.
                throw new IncorrectShapeException("provided out is the wrong size for the accumulation.");
            np.copyto(@out, result, casting: "unsafe");
            return @out;
        }
    }
}
