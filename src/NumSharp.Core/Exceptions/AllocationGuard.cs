using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     The size guards NumPy applies before it will allocate an array.
    /// </summary>
    /// <remarks>
    ///     Port of the dimension-checking block of <c>PyArray_NewFromDescr_int</c>
    ///     (numpy/_core/src/multiarray/ctors.c). It exists because <b>a size computed by
    ///     multiplication cannot be trusted until its overflow has been checked</b>, and the
    ///     failure is silent by construction: a wrapped byte count looks like a small allocation,
    ///     not like an error.
    ///
    ///     What that looked like in NumSharp before this guard: <c>np.zeros((2^31, 2^31))</c> of
    ///     float64 has element count 2^62 — correct, and representable — but 2^62 × 8 bytes is
    ///     2^65, which wraps to <b>exactly 0</b>. The allocator was asked for zero bytes and
    ///     obliged, so the array constructed successfully, <c>[0,0]</c> read 0, and the last
    ///     element read whatever happened to be there — an out-of-bounds read from public API.
    ///     <c>np.ones</c> on the same shape went one worse and wrote out of bounds.
    ///     One dimension over (<c>(2^30, 2^33)</c>) the element count itself overflowed and
    ///     <c>size</c> reported negative.
    /// </remarks>
    internal static class AllocationGuard
    {
        internal const string TooBigMessage =
            "array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.";

        internal const string NegativeDimensionsMessage = "negative dimensions are not allowed";

        /// <summary>
        ///     Validates a shape's dimensions against <paramref name="itemSize"/> before allocating.
        /// </summary>
        /// <remarks>
        ///     Mirrors NumPy's loop exactly, including two behaviours that are easy to get wrong and
        ///     both observable (probed against 2.4.2):
        ///     <list type="bullet">
        ///     <item>The running product starts at <b>itemSize</b>, not 1 — the quantity being
        ///     overflow-checked is the BYTE count, which is why <c>np.zeros(2^60)</c> of float64 is
        ///     rejected (2^63 bytes) while the same shape of int8 is merely a
        ///     <c>MemoryError</c>.</item>
        ///     <item>A zero dimension does NOT short-circuit the scan. NumPy skips it and keeps
        ///     multiplying "as if" it were 1, so <c>np.zeros((0, 2^62))</c> of float64 still reports
        ///     "array is too big" rather than quietly succeeding at zero bytes. The byte count is
        ///     forced to 0 afterwards by the caller (an empty array allocates nothing).</item>
        ///     </list>
        ///     Dimensions are scanned left to right and each check fires in place, so the reported
        ///     error follows NumPy's ordering: <c>(-1, 2^62)</c> is "negative dimensions", while
        ///     <c>(2^62, -1)</c> overflows on the first dimension and reports "array is too big".
        /// </remarks>
        /// <param name="dims">The dimensions about to be allocated.</param>
        /// <param name="itemSize">Bytes per element.</param>
        /// <exception cref="ValueError">
        ///     If a dimension is negative, or if <c>itemSize * ∏dims</c> overflows <see cref="long"/>.
        /// </exception>
        internal static void CheckDimensions(long[] dims, int itemSize)
        {
            if (dims == null || dims.Length == 0 || itemSize <= 0)
                return;

            long nbytes = itemSize;
            for (int i = 0; i < dims.Length; i++)
            {
                long dim = dims[i];

                // Keep multiplying "as if" this were 1 so an overflow further along is still
                // reported; an empty array's byte count is zero regardless.
                if (dim == 0)
                    continue;

                if (dim < 0)
                    ThrowNegativeDimensions();

                if (dim > long.MaxValue / nbytes)
                    ThrowTooBig();

                nbytes *= dim;
            }
        }

        /// <summary>
        ///     The last line of defence, on the element count actually handed to an allocator.
        /// </summary>
        /// <remarks>
        ///     <see cref="CheckDimensions"/> is the parity guard and reports NumPy's messages from
        ///     the shape the caller asked for. This one sits at the single point where every
        ///     allocation converts a count into bytes, so a path that reaches an allocator without
        ///     having gone through a <see cref="Shape"/> at all — or a future one that forgets to —
        ///     still fails loudly instead of receiving a wrapped, undersized buffer.
        /// </remarks>
        /// <param name="count">Element count.</param>
        /// <param name="itemSize">Bytes per element.</param>
        /// <exception cref="ValueError">If <paramref name="count"/> is negative, or the byte count overflows.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void CheckElementCount(long count, int itemSize)
        {
            if (count < 0)
                ThrowNegativeDimensions();

            if (itemSize > 0 && count > long.MaxValue / itemSize)
                ThrowTooBig();
        }

        [DoesNotReturn]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowTooBig() => throw new ValueError(TooBigMessage);

        [DoesNotReturn]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ThrowNegativeDimensions() => throw new ValueError(NegativeDimensionsMessage);

        /// <summary>
        ///     Convenience overload resolving <paramref name="typeCode"/>'s item size.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void CheckDimensions(long[] dims, NPTypeCode typeCode)
        {
            if (typeCode != NPTypeCode.Empty)
                CheckDimensions(dims, typeCode.SizeOf());
        }
    }
}
