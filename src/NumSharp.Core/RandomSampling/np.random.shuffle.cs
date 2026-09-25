using System;

namespace NumSharp
{
    public partial class NumPyRandom
    {
        /// <summary>
        ///     Modify a sequence in-place by shuffling its contents.
        /// </summary>
        /// <param name="x">The writeable array to be shuffled (at least 1-D).</param>
        /// <exception cref="TypeError"><paramref name="x"/> is 0-d (<c>len() of unsized object</c> — NumPy evaluates <c>len(x)</c> first).</exception>
        /// <exception cref="ValueError"><paramref name="x"/> is read-only (<c>array is read-only</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
        ///     <br/>
        ///     This function only shuffles the array along the first axis of a multi-dimensional array.
        ///     The order of sub-arrays is changed but their contents remain the same.
        ///     <br/>
        ///     Byte-identical to NumPy's legacy <c>RandomState.shuffle</c>: a Fisher–Yates with
        ///     <c>random_interval</c> (mask rejection) — over the raw elements of a 1-D array (honouring its stride),
        ///     or over the rows of an N-D array (the same draws, applied as one gather + NDIter copy-back). Holds the
        ///     bit generator's lock for the draws. NumPy's Generator API (<c>rng.shuffle</c>) adds an axis parameter;
        ///     the legacy one does not.
        /// </remarks>
        /// <example>
        ///     <code>
        ///     // 1D array - elements are shuffled
        ///     var arr = np.arange(10);
        ///     np.random.shuffle(arr);
        ///
        ///     // 2D array - rows are shuffled, contents within rows unchanged
        ///     var arr2d = np.arange(9).reshape(3, 3);
        ///     np.random.shuffle(arr2d);
        ///     // e.g. [[6,7,8], [0,1,2], [3,4,5]] - rows reordered
        ///     </code>
        /// </example>
        [NDScoped] // reclaims the N-D path's index array and gathered rows after the copy-back
        public unsafe void shuffle(NDArray x)
        {
            // NumPy: `n = len(x)` is evaluated first, so a 0-d array is a TypeError.
            if (x.ndim == 0)
                throw new TypeError("len() of unsized object");

            // Fisher-Yates swaps write the buffer directly, bypassing the guarded setters, so a non-writeable
            // target (a broadcast view, or a read-only interop / mmap('r') array) must be rejected up front —
            // NumPy: "array is read-only".
            NumSharpException.ThrowIfNotWriteable(x.Shape, "array");

            long n = x.shape[0];
            if (x.size == 0 || n <= 1)
                return;

            if (x.ndim == 1)
            {
                // NumPy's fast path: _shuffle_raw over the buffer with the array's own stride.
                int itemsize = x.dtypesize;
                lock (randomizer.@lock)
                    BoundedIntegers.ShuffleRaw(randomizer, x.Storage.Address + x.Shape.offset * itemsize, n,
                                               x.Shape.strides[0] * itemsize, itemsize);
                return;
            }

            // NumPy's N-D path swaps rows x[i] <-> x[random_interval(i)] for i = n-1..1; running the same draws
            // over an index vector yields the identical permutation, applied as a gather + write-back.
            long[] idx;
            lock (randomizer.@lock)
                idx = BoundedIntegers.FisherYatesIndices(randomizer, n);
            var reordered = np.take(x, np.array(idx), axis: 0);
            np.copyto(x, reordered);
        }
    }
}
