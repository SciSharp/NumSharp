using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================================================
        // THE CHAR LOOP of the add / multiply reductions. NumSharp's char has no NumPy dtype; in every arithmetic NumSharp
        // gives it, and in the oracle (gen_oracle's char tier relabels uint16 -> char), it is NumPy's uint16. A char LOOP
        // — dtype=char, or the out= rule's promote_types(out, arr) naming char — is therefore NumPy's uint16 loop: sums and
        // products modulo 2^16. The add / multiply kernels have no char loop of their own, so the reduction runs in the
        // uint16 loop and its result is handed back as char.
        // =============================================================================================================

        /// <summary>
        ///     The add / multiply reduction in a char loop (<c>dtype=char</c>, or the out= rule's
        ///     <c>promote_types(out, arr)</c> naming char): the same reduction in the uint16 loop, its result handed back as
        ///     char (the same 16 bits).
        /// </summary>
        /// <param name="arr">The reduced array (any dtype; not the uninitialized shape, whose identity answers the callers
        ///     give before this).</param>
        /// <param name="axis">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="product">True for <c>multiply.reduce</c> (<see cref="ReduceProduct"/>), false for
        ///     <c>add.reduce</c> (<see cref="ReduceAdd"/>).</param>
        /// <returns>A fresh char array holding the uint16 loop's bits in the uint16 result's layout: a read-only 0-d scalar
        ///     for a full reduction without keepdims, as the uint16 route returns it.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds (the uint16 route's check).</exception>
        /// <remarks>
        ///     <para>NumPy's uint16 loop wraps: <c>np.sum([[3, 300, 65535], [100, 2, 65000]], dtype=uint16)</c> is 65404, the
        ///     product 11008, the product along axis 0 [300, 600, 536] (65535 · 65000 wraps to 536). The routes this
        ///     replaces had no char loop: the flat sum and product raised <see cref="System.NotSupportedException"/>, the
        ///     axis product multiplied through a double and saturated (536 came out as 65535), and their callers inherited
        ///     both — np.var / np.std with <c>dtype=char</c>, the out= loop rule's char loop, and numpy.matrix's sum / prod
        ///     of a char matrix into a bool out, which raised since the class landed.</para>
        ///     <para>A char input is read as uint16 IN PLACE — a forwarding alias of its storage (the same bits, no copy) — so
        ///     the uint16 route runs its same-type kernels; any other input reaches the uint16 loop through the reduction's
        ///     own cast loop, NumPy's buffered cast into <c>dtype=uint16</c> (an int8 -1 becomes 65535).</para>
        ///     <para>The uint16 result is copied into a fresh char array of the same layout: one pass over the output, which
        ///     is never larger than the input. A relabelling alias would cost nothing more, but the result would then report
        ///     a base instead of owning its data, where a reduction's result is a fresh array.</para>
        ///     <para>The alias and the uint16 result are this method's intermediates and are released on every path; the
        ///     caller's array is never disposed.</para>
        /// </remarks>
        private NDArray ReduceInCharLoop(NDArray arr, int? axis, bool keepdims, bool product)
        {
            // The uint16 view of a char input shares its storage's block (AliasAs gives the alias a forwarding ARC slice),
            // so disposing it below only drops the alias's reference.
            NDArray proxy = arr.GetTypeCode == NPTypeCode.Char
                ? new NDArray(arr.Storage.AliasAs(NPTypeCode.UInt16)) { TensorEngine = this }
                : arr;
            NDArray reduced = null;
            try
            {
                reduced = product
                    ? ReduceProduct(proxy, axis, keepdims, np.uint16)
                    : ReduceAdd(proxy, axis, keepdims, np.uint16);
                // A fresh array of the uint16 result's dims and strides (Shape.ForFreshAllocation keeps a fresh buffer's
                // layout, and always allocates it writeable).
                var result = new NDArray(NPTypeCode.Char, reduced.Shape, false);
                // uint16 -> char is the identity on the 16 bits (both are 2-byte unsigned lanes), so the cast copy moves the
                // loop's results unchanged in whatever layout the uint16 route gave them.
                NDIter.Copy(result, reduced);
                // The uint16 route's full reduction is NumPy's scalar, read-only (MarkReductionScalar); so is its char twin.
                return reduced.Shape.IsWriteable ? result : result.MarkReductionScalar();
            }
            finally
            {
                // A reduction can hand back its input (a degenerate route); never dispose the caller's array or the alias
                // twice.
                if (reduced is not null && !ReferenceEquals(reduced, proxy) && !ReferenceEquals(reduced, arr))
                    reduced.Dispose();
                if (!ReferenceEquals(proxy, arr))
                    proxy.Dispose();
            }
        }
    }
}
