using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Packs the elements of a binary-valued array into bits in a uint8 array. Each element
        ///     contributes one bit (1 iff the element is nonzero); the axis is shortened to
        ///     <c>ceil(len/8)</c> and zero-padded to a full trailing byte.
        /// </summary>
        /// <param name="a">Input array. Must be a boolean or integer dtype (float/complex/decimal raise); read in logical order for any memory layout.</param>
        /// <param name="axis">Axis to pack along. <c>null</c> (NumPy <c>None</c>) packs the C-order flattened array and returns a 1-D result.</param>
        /// <param name="bitorder">
        ///     <c>"big"</c> (default) puts the first element in the most-significant bit
        ///     (<c>[.,.,.,.,.,.,1,1] =&gt; 3</c>); <c>"little"</c> reverses it. Matched to NumPy's prefix rule:
        ///     a value beginning with <c>"little"</c> is little, one beginning with <c>"big"</c> is big;
        ///     <c>null</c> is treated as <c>"big"</c>.
        /// </param>
        /// <returns>A uint8 array with the packed bits — same ndim as <paramref name="a"/> (except <c>axis=null</c> ⇒ 1-D), F-contiguous iff <paramref name="a"/> is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> is neither a "big" nor "little" prefix.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not a boolean/integer dtype.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.packbits.html <para>The body is
        ///     <see cref="TensorEngine.Packbits"/> (the np-layer dtype plan); this facade keeps the <c>numpy.matrix</c>
        ///     rule.</para></remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray packbits(NDArray a, int? axis = null, string bitorder = "big")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, bitorder), static s => packbits(s.a, s.axis, s.bitorder));
            return np.EngineOf(a).Packbits(a, axis, bitorder);
        }

        /// <summary>
        ///     Unpacks each uint8 element of an array into 8 binary-valued (0/1) uint8 elements along an
        ///     axis — the inverse of <see cref="packbits(NDArray,int?,string)"/>.
        /// </summary>
        /// <param name="a">Input array. Must be uint8 (any other dtype raises); read in logical order for any memory layout.</param>
        /// <param name="axis">Axis to unpack along. <c>null</c> (NumPy <c>None</c>) unpacks the C-order flattened array and returns a 1-D result.</param>
        /// <param name="count">
        ///     Number of output elements along <paramref name="axis"/> — undoes packing a non-multiple-of-8
        ///     size. <c>null</c> unpacks everything (<c>8·len</c>); a non-negative value takes exactly that
        ///     many bits (truncating, or zero-padding past the available bits); a negative value trims that
        ///     many bits off the end (and must not exceed the available bits).
        /// </param>
        /// <param name="bitorder">
        ///     <c>"big"</c> (default) or <c>"little"</c>. Matched to NumPy's unpack rule: only the FIRST
        ///     character is inspected (<c>'l'</c> ⇒ little, <c>'b'</c> ⇒ big); <c>null</c> is <c>"big"</c>.
        /// </param>
        /// <returns>A uint8 array of 0/1 values — same ndim as <paramref name="a"/> (except <c>axis=null</c> ⇒ 1-D), F-contiguous iff <paramref name="a"/> is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> does not begin with 'l' or 'b', or <c>-count</c> exceeds the available bits.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not uint8.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.unpackbits.html <para>The body is
        ///     <see cref="TensorEngine.Unpackbits"/> (the np-layer dtype plan); this facade keeps the <c>numpy.matrix</c>
        ///     rule.</para></remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray unpackbits(NDArray a, int? axis = null, long? count = null, string bitorder = "big")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped((a, axis, count, bitorder), static s => unpackbits(s.a, s.axis, s.count, s.bitorder));
            return np.EngineOf(a).Unpackbits(a, axis, count, bitorder);
        }
    }
}
