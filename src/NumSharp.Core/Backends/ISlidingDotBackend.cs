namespace NumSharp.Backends
{
    /// <summary>
    ///     An optional capability a <see cref="IBlasBackend"/> MAY also implement to supply the
    ///     byte-parity level-1 dot product (NumPy's per-dtype <c>dotfunc</c>) behind the sliding
    ///     multiply-accumulate family — <c>np.correlate</c> and <c>np.convolve</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a second, separate interface.</b> <see cref="IBlasBackend"/> is the matrix-product
    ///     seam: every member there takes whole <see cref="NDArray"/> operands. <c>correlate</c>/
    ///     <c>convolve</c> are not matrix products — they are a sliding dot whose inner reduction runs
    ///     once per OUTPUT position (thousands of tiny dots), so the primitive they need is a raw
    ///     strided vector dot called in a hot loop, not an allocating whole-array method. Rather than
    ///     put a pointer method on the product interface, this capability is discovered by a cast:
    ///     <c>engine.Blas as ISlidingDotBackend</c>. A backend that does not implement it simply leaves
    ///     the managed sliding kernels in place — the exact optional-capability pattern
    ///     <see cref="IBlasBackend.TryMatMulBatched"/> expresses as a default method.
    ///     </para>
    ///     <para>
    ///     <b>What it is byte-for-byte.</b> This is NumPy's <c>@name@_dot</c> from
    ///     <c>numpy/_core/src/multiarray/arraytypes.c.src</c> — the very <c>dotfunc</c> that
    ///     <c>_pyarray_correlate</c> (multiarraymodule.c) calls for every ramp position and for the
    ///     middle whenever <c>small_correlate</c> declines. NumPy routes only <c>float32</c>,
    ///     <c>float64</c> and <c>complex128</c> through cblas here (its <c>#USEBLAS = 1,1,0,0,1,1,…</c>
    ///     over FLOAT/DOUBLE/…/CFLOAT/CDOUBLE), summing each dot in a <c>double</c> accumulator "for
    ///     stability" (a chunked <c>?dot</c>, <c>?dotu</c> for complex). A backend supplies exactly that
    ///     so <c>np.correlate</c>/<c>np.convolve</c> match NumPy to the last bit on the long float
    ///     kernels a portable reduction reorders — the same reason
    ///     <c>NumSharp.Interop.OpenBLAS</c> exists for the matrix products.
    ///     </para>
    ///     <para>
    ///     <b>Which positions the engine routes here.</b> The engine keeps the whole
    ///     <c>small_correlate</c>-eligible regime — real <c>float32</c>/<c>float64</c> kernels of length
    ///     ≤ 11 — on its own managed kernel, which is already byte-identical to NumPy there (a plain
    ///     sequential sum). Only real kernels longer than 11 and every complex kernel reach this
    ///     interface, and for those NumPy sends every ramp AND middle position through cblas, so the
    ///     engine does too — no mixed managed/native path within one call.
    ///     </para>
    ///     <para>
    ///     <b>Reading the operands.</b> Element strides, not bytes (NumPy's are bytes — the same logic
    ///     with <c>itemsize == 1</c>). The operands the engine hands over are always contiguous,
    ///     offset-0 buffers of the result dtype, so the stride is one element; the pointers advance to
    ///     the sub-array a given output position dots. Byte-parity still depends on the same three
    ///     levers the products do (the OpenBLAS build, its thread count, the dispatched DYNAMIC_ARCH
    ///     kernel) — level-1 <c>?dot</c> is typically single-threaded, but a host whose CBLAS differs is
    ///     free to produce different bits, exactly as with <c>dot</c>/<c>matmul</c>.
    ///     </para>
    /// </remarks>
    public interface ISlidingDotBackend
    {
        /// <summary>
        ///     Whether this backend supplies the byte-parity dot for <paramref name="dtype"/>. NumPy
        ///     routes only <c>float32</c>/<c>float64</c>/<c>complex128</c> through cblas here; complex is
        ///     served only when the loaded library also exports the complex products. Every other dtype
        ///     returns false and the engine keeps its managed kernel — which is bit-exact by
        ///     construction for the integer/bool families and matches NumPy's own scalar loops for the
        ///     rest.
        /// </summary>
        bool SupportsDot(NPTypeCode dtype);

        /// <summary>
        ///     Computes one strided vector dot, NumPy's <c>@name@_dot</c>:
        ///     <c>*result = Σ a[i·strideA] · b[i·strideB]</c> for <c>i</c> in <c>[0, count)</c>, summed
        ///     the way NumPy sums it (chunked cblas <c>?dot</c> accumulated in <c>double</c>;
        ///     <c>?dotu</c> — UNCONJUGATED — for complex, matching <c>np.correlate</c>/<c>np.convolve</c>,
        ///     which never conjugate the kernel here because <c>correlate</c> has already conjugated it).
        /// </summary>
        /// <param name="dtype">One of Single/Double/Complex — the caller has checked
        /// <see cref="SupportsDot"/> first.</param>
        /// <param name="a">Base pointer of the first operand's sub-array.</param>
        /// <param name="strideA">Element stride of <paramref name="a"/> (one, for a contiguous buffer).</param>
        /// <param name="b">Base pointer of the second operand's sub-array.</param>
        /// <param name="strideB">Element stride of <paramref name="b"/>.</param>
        /// <param name="result">Where the scalar dot is written (one element of <paramref name="dtype"/>).</param>
        /// <param name="count">Number of terms in the dot (may be 0, which writes the zero sum).</param>
        unsafe void Dot(NPTypeCode dtype, void* a, long strideA, void* b, long strideB, void* result, long count);
    }
}
