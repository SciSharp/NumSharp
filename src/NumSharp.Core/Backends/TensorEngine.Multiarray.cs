// The C functions of NumPy's _multiarray_umath that compute (np.dot and np.lexsort so far; Appendix D of
// docs/plans/np-layer-dtype-logic.md lists the rest). The C functions of the array object (np.array,
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
    }
}
