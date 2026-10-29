// The C functions of NumPy's _multiarray_umath that compute (np.dot, np.lexsort, np.unravel_index, np.packbits and
// np.unpackbits so far; Appendix D of docs/plans/np-layer-dtype-logic.md lists the rest). The C functions of the array object (np.array,
// np.copyto, np.concatenate, ...) are not engine members.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.dot(a, b, out=None)</c> — <c>PyArray_MatrixProduct2</c> — the whole call: the text operands'
        ///     answers in NumPy's order, the product (an inner product, a matrix product, a multiply with a 0-d operand, or a
        ///     sum product over the last axis of <paramref name="x"/> and the second-to-last of <paramref name="y"/>), and
        ///     <c>new_array_for_sum</c>'s strict <c>out=</c>.
        /// </summary>
        /// <param name="x">The first operand.</param>
        /// <param name="y">The second operand.</param>
        /// <param name="out">The out array, or null for a fresh result. It must have the product's exact dtype and rank, be
        ///     C-contiguous and writeable; no cast, no broadcast.</param>
        /// <returns>The product, or <paramref name="out"/> after the product was copied into it.</returns>
        /// <exception cref="ValueError">An out array the check refuses: <c>output array is not acceptable (must have the
        ///     right datatype, number of dimensions, and be a C-Array)</c>, or, of the right rank, <c>output array has wrong
        ///     dimensions</c>.</exception>
        /// <exception cref="IncorrectShapeException">Misaligned operands (NumPy's <c>shapes (…) and (…) not aligned</c>
        ///     ValueError text).</exception>
        /// <remarks>Widened in place with <paramref name="out"/> (the np-layer dtype plan, P2): a call without it is the
        ///     product alone, as before, and an engine deriving from TensorEngine implements the three-parameter member.</remarks>
        public abstract NDArray Dot(NDArray x, NDArray y, NDArray @out = null);

        /// <summary>
        ///     NumPy's <c>np.lexsort(keys, axis)</c> (<c>PyArray_LexSort</c>), the whole call: the int64 indices that sort
        ///     by several keys at once, the LAST key primary and each earlier one breaking the ties of the keys after it
        ///     (a stable sort per key).
        /// </summary>
        /// <param name="keys">The keys, every one of the same shape (any dtype; text sorts by code points). The public
        ///     <c>np.lexsort(NDArray, int)</c> splits an array along its first axis into this list.</param>
        /// <param name="axis">The axis sorted along (negative counts from the end). Keys of rank 0 accept 0 and -1 (NumPy's
        ///     backwards compatibility) and answer the 0-d <c>0</c>.</param>
        /// <returns>A fresh int64 array of the keys' shape (zeros for keys of at most one element).</returns>
        /// <exception cref="TypeError"><paramref name="keys"/> is null or empty (<c>need sequence of keys with len &gt; 0 in
        ///     lexsort</c>).</exception>
        /// <exception cref="ArgumentNullException">A key is null.</exception>
        /// <exception cref="ValueError">The keys differ in shape (<c>all keys need to be the same shape</c>).</exception>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan, phase PE, E5): the public <c>np.lexsort</c> applies its class
        ///     boundaries and calls this.</remarks>
        public abstract NDArray LexSort(NDArray[] keys, int axis = -1);

        /// <summary>
        ///     NumPy's <c>np.unravel_index(indices, shape, order='C')</c> (<c>arr_unravel_index</c>,
        ///     <c>numpy/_core/src/multiarray/compiled_base.c</c>), the whole call: the coordinates of each flat index in an
        ///     array of dimensions <paramref name="shape"/>.
        /// </summary>
        /// <param name="indices">An integer (or bool) array of flat indices, of any rank and layout. Read as int64: a uint64
        ///     index at or past 2^63 wraps negative and is out of bounds, as NumPy's same_kind cast to intp makes it.</param>
        /// <param name="shape">The dimensions to unravel into. An empty shape (NumPy's <c>()</c>) is legal: its one valid
        ///     index is 0, and only a 0-d index array may use it.</param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case.</param>
        /// <returns>One int64 array per dimension of <paramref name="shape"/>, each of <paramref name="indices"/>' shape:
        ///     the columns of one interleaved buffer of shape <c>indices.shape + (len(shape),)</c> laid out along the
        ///     indices' iteration order, so every array is a strided view of that shared buffer. A 0-d
        ///     <paramref name="indices"/> gives NumPy's scalars instead (fresh read-only 0-d arrays); an empty
        ///     <paramref name="shape"/> with a 0-d index gives an empty array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="indices"/> or <paramref name="shape"/> is null.</exception>
        /// <exception cref="ValueError">In NumPy's order: an order letter that is not C / F / A / K in either case; a
        ///     product of the dimensions that overflows int64 or is exactly -1 (<c>dimensions are too large; ...</c>); then,
        ///     after the <see cref="TypeError"/>, an <c>'A'</c> or <c>'K'</c> order (<c>only 'C' or 'F' order is
        ///     permitted</c>); the first index outside <c>[0, prod(shape))</c> in the iteration order (<c>index N is out of
        ///     bounds for array with size M</c>); and, for an empty <paramref name="shape"/>, an index array of rank 1 or more
        ///     (<c>multiple indices are not supported for 0d arrays</c>, after the indices).</exception>
        /// <exception cref="TypeError"><paramref name="indices"/> is neither integer nor bool (<c>only int indices
        ///     permitted</c>).</exception>
        /// <exception cref="NotSupportedException">Runtime code generation is unavailable (the IL kernel cannot be
        ///     emitted).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan): the public <c>np.unravel_index(NDArray, int[], char)</c> computes
        ///     a <c>numpy.matrix</c> operand as a plain array and calls this.</remarks>
        public abstract NDArray<long>[] UnravelIndex(NDArray indices, int[] shape, char order = 'C');

        /// <summary>
        ///     NumPy's <c>np.packbits(a, axis=None, bitorder='big')</c> (<c>io_pack</c> / <c>pack_bits</c>,
        ///     <c>numpy/_core/src/multiarray/compiled_base.c</c>), the whole call: the elements packed into the bits of a
        ///     uint8 array, one bit per element (1 iff it is nonzero), the axis shortened to <c>ceil(len / 8)</c> and the
        ///     last byte zero-padded.
        /// </summary>
        /// <param name="a">A bool or integer array (NumSharp's char included), read in logical order for any layout.</param>
        /// <param name="axis">The axis to pack along, or null to pack the C-order flattened array into a 1-D result. A
        ///     0-d input is promoted to <c>(1,)</c> and accepts only 0 and -1.</param>
        /// <param name="bitorder"><c>"big"</c> puts the first element in the most significant bit, <c>"little"</c> in
        ///     the least; NumPy's prefix rule: a value beginning with <c>"little"</c> is little, one beginning with
        ///     <c>"big"</c> big, null is big.</param>
        /// <returns>A new uint8 array of <paramref name="a"/>'s rank (1-D for a null axis), F-contiguous when the input
        ///     is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> begins with neither (<c>'order' must be either
        ///     'little' or 'big'</c>), checked first.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is neither bool nor integer (<c>Expected an input array of
        ///     integer or boolean data type</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan): the public <c>np.packbits</c> applies the <c>numpy.matrix</c>
        ///     rule and calls this.</remarks>
        public abstract NDArray Packbits(NDArray a, int? axis = null, string bitorder = "big");

        /// <summary>
        ///     NumPy's <c>np.unpackbits(a, axis=None, count=None, bitorder='big')</c> (<c>io_unpack</c> /
        ///     <c>unpack_bits</c>), the whole call: each uint8 element unpacked into eight 0 / 1 uint8 elements along the
        ///     axis, the inverse of <see cref="Packbits"/>.
        /// </summary>
        /// <param name="a">A uint8 array, read in logical order for any layout.</param>
        /// <param name="axis">The axis to unpack along, or null to unpack the C-order flattened array into a 1-D result.
        ///     A 0-d input is promoted to <c>(1,)</c> and accepts only 0 and -1.</param>
        /// <param name="count">The number of output elements along the axis: null unpacks every bit (<c>8 * len</c>); a
        ///     non-negative count takes that many bits (truncating, or zero-padding past the available bits); a negative
        ///     count trims that many bits off the end.</param>
        /// <param name="bitorder"><c>"big"</c> or <c>"little"</c>; NumPy's unpack rule reads only the first character
        ///     (<c>'l'</c> is little, <c>'b'</c> big), null is big.</param>
        /// <returns>A new uint8 array of 0 / 1 values of <paramref name="a"/>'s rank (1-D for a null axis), F-contiguous
        ///     when the input is.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="bitorder"/> is empty or begins with neither (<c>'order' must
        ///     begin with 'l' or 'b'</c>), checked first; or <c>-count</c> exceeds the available bits (<c>-count larger than
        ///     number of elements</c>).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not uint8 (<c>Expected an input array of unsigned byte
        ///     data type</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for <paramref name="a"/>'s rank.</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan): the public <c>np.unpackbits</c> applies the <c>numpy.matrix</c>
        ///     rule and calls this. An empty input with a count that asks for elements gives zeros, where NumPy leaks
        ///     uninitialised memory.</remarks>
        public abstract NDArray Unpackbits(NDArray a, int? axis = null, long? count = null, string bitorder = "big");
    }
}
