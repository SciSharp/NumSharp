using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the elements of <paramref name="arr"/> that satisfy some
        ///     <paramref name="condition"/>. Equivalent to
        ///     <c>np.take(np.ravel(arr), np.flatnonzero(np.ravel(condition)))</c> —
        ///     i.e. <c>arr.ravel()[condition.ravel()]</c> when condition is boolean.
        /// </summary>
        /// <param name="condition">
        ///     Array whose nonzero / True entries indicate the elements of
        ///     <paramref name="arr"/> to extract. May be any dtype (treated as
        ///     truthy via NumPy's "nonzero" semantics). May be any shape — it is
        ///     ravel'd before alignment with <paramref name="arr"/>.
        /// </param>
        /// <param name="arr">Input array. May be any shape; it is ravel'd.</param>
        /// <returns>
        ///     Rank-1 <see cref="NDArray"/> of values from <paramref name="arr"/>
        ///     where the corresponding ravel'd <paramref name="condition"/> entry
        ///     is truthy. Dtype matches <paramref name="arr"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.extract.html
        ///     <para>
        ///     Two execution paths:
        ///     <list type="bullet">
        ///       <item>
        ///         <b>Fast path</b>: bool condition, contig arr + condition,
        ///         <c>condition.size &lt;= arr.size</c>. Runs a fused IL kernel
        ///         (popcount → alloc → SIMD bit-scan + cpblk) avoiding the
        ///         indices NDArray allocation that the generic path needs.
        ///       </item>
        ///       <item>
        ///         <b>Generic path</b>: mirrors NumPy's literal chain
        ///         <c>take(ravel(arr), flatnonzero(ravel(condition)))</c>. Handles
        ///         non-bool conditions (any dtype interpreted as nonzero), broadcast
        ///         / strided / negative-stride sources, and the OOB-True case
        ///         (raises via take's RAISE mode).
        ///       </item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     Note that <see cref="place"/> is the inverse operation.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray extract(NDArray condition, NDArray arr)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Kept((condition, arr), static s => extract(s.condition, s.arr));
            using var plain = NDMatrixBoundary.PlainFor(condition, arr);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            // NumPy's numpy.extract: the statements are the array object's (NDArray.FunctionBase.cs)
            return NDArray.extract(condition, arr);
        }
    }
}
