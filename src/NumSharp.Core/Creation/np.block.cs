using System;
using System.Collections;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Assemble an nd-array from nested lists of blocks.
        ///     Blocks in the innermost lists are concatenated (see
        ///     <see cref="concatenate(NDArray[], int?, NDArray, DType, string)"/>)
        ///     along the last dimension (-1), then these are concatenated along
        ///     the second-last dimension (-2), and so on until the outermost
        ///     list is reached.
        ///     Blocks can be of any dimension, but will not be broadcasted using
        ///     the normal rules. Instead, leading axes of size 1 are inserted,
        ///     to make <c>block.ndim</c> the same for all blocks.
        /// </summary>
        /// <param name="arrays">
        ///     Nested "lists" of blocks. The C# mapping of NumPy's nested Python
        ///     lists:
        ///     <list type="bullet">
        ///     <item><see cref="NDArray"/> — a leaf block.</item>
        ///     <item>numeric scalar (int, double, bool, …) — a 0-d leaf.</item>
        ///     <item>any 1-D <see cref="Array"/> (<c>object[]</c>, <c>NDArray[]</c>,
        ///     <c>int[]</c>, jagged <c>int[][]</c>, …) or non-generic
        ///     <see cref="IList"/> (<c>List&lt;T&gt;</c>) — a nested LIST whose
        ///     elements are classified recursively (mirrors a Python list).</item>
        ///     <item>rank ≥ 2 rectangular arrays (<c>int[,]</c>, …) — a leaf,
        ///     converted via <see cref="asanyarray(in object, DType, char?, string, bool?)"/>.</item>
        ///     <item>tuples (<see cref="ITuple"/>) — rejected with
        ///     <see cref="TypeError"/>, matching NumPy ("np.block does not allow
        ///     implicit conversion from tuple to ndarray").</item>
        ///     </list>
        ///     If passed a single NDArray or scalar (a nested list of depth 0),
        ///     a copy is returned (matching NumPy 2.x behavior).
        /// </param>
        /// <returns>
        ///     The array assembled from the given blocks. The dimensionality of
        ///     the output is equal to the greatest of the dimensionality of all
        ///     the inputs and the depth to which the input list is nested.
        /// </returns>
        /// <exception cref="ValueError">
        ///     If list depths are mismatched — for instance <c>[[a, b], c]</c> is
        ///     illegal and should be spelt <c>[[a, b], [c]]</c> — or if lists are
        ///     empty — for instance <c>[[a, b], []]</c>.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.block.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray block(object arrays)
        {
            if (arrays is null)
                throw new ArgumentNullException(nameof(arrays));

            // NumPy's statements (_block_setup's depth checks, then repeated concatenation or the single-allocation
            // route past 2*512*512 elements) are the array object's, which also decides a numpy.matrix block's class (the
            // leaves are nested, so the computation handles them itself: NumPy's class follows the route it took).
            return NDArray.block(arrays);
        }
    }
}
