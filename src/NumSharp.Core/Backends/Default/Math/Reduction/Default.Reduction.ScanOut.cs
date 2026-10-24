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

        /// <summary>
        ///     The loop dtype of add.accumulate / multiply.accumulate (<see cref="ReduceCumAdd"/>,
        ///     <see cref="ReduceCumMul"/>): the <c>dtype=</c> request; else, with an <c>out=</c>,
        ///     <c>promote_types(out.dtype, arr.dtype)</c> WITHOUT NEP 50's widening of small integers
        ///     (<c>reducelike_promote_and_resolve</c>, the reductions' rule of <see cref="ReduceIntoOut"/>); else null, which
        ///     selects the input's NEP 50 accumulator.
        /// </summary>
        /// <param name="arr">The accumulated array (numeric: the scans' text routes answered before this is read).</param>
        /// <param name="dtype">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The <c>out=</c> array, or null.</param>
        /// <returns>The loop dtype, or null for the NEP 50 accumulator.</returns>
        /// <remarks>
        ///     <para>NumPy 2.4.2: int8 [100, 100, 56] accumulated into a bool out runs the int8 loop, whose running sum wraps to
        ///     0 at the third element (stored False; the int64 accumulator stored True there); float32 [16777216, 1, 1] into a
        ///     float64 out adds in float64 (16777217, 16777218); float16 [300, 300] multiplied into a float32 out gives 90000
        ///     (the float16 accumulator overflowed to inf); uint64 [2^63 + 1, 0] into an int64 out runs the float64 loop
        ///     (stored INT64_MIN twice, where the uint64 accumulator stored 2^63 + 1's bits).</para>
        ///     <para>The running value stays in the loop dtype across NumPy's buffer fills (an int8 run of 24,576 elements into
        ///     a bool out is False at every third element), so accumulating the whole run in the loop and storing it once
        ///     (<see cref="WriteScanToOut"/>) is NumPy's answer; the reductions have a residual there that accumulations do
        ///     not.</para>
        /// </remarks>
        private static DType AccumulateLoop(NDArray arr, DType dtype, NDArray @out)
            => dtype ?? (@out is null ? null : DTypePromotion.PromoteTypes(@out.dtype, arr.dtype));
    }
}
