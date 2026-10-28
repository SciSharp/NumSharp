// The C functions of NumPy's _multiarray_umath that compute (np.dot so far; Appendix D of
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
    }
}
