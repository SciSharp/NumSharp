using System.Collections;
using System.Collections.Generic;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     Flat (1-D, C-order) element iterator — the NumSharp analog of NumPy's
    ///     <c>flatiter</c>, used by <see cref="np.Broadcast.iters"/>.
    ///
    ///     It wraps an operand already broadcast (via <see cref="np.broadcast_to(NDArray, Shape)"/>)
    ///     to the broadcast result shape, and yields each logical element — in C-order, expanding
    ///     stride-0 (broadcast) dimensions — exactly like NumPy's <c>broadcast.iters[i]</c>:
    ///
    ///     <code>
    ///     // numpy: np.broadcast([1,2,3], [[10],[20]]).iters[0] -> 1,2,3,1,2,3
    ///     //                                            .iters[1] -> 10,10,10,20,20,20
    ///     </code>
    ///
    ///     The broadcast expansion is the same Shape/stride machinery NpyIter uses; element access
    ///     resolves the (possibly stride-0) coordinates per step, so no buffer is materialized.
    /// </summary>
    public sealed class NpyFlatIterator : IEnumerable<object>, IEnumerable
    {
        private readonly NDArray _view;

        /// <param name="broadcastView">An operand already broadcast to the target (result) shape.</param>
        internal NpyFlatIterator(NDArray broadcastView)
        {
            _view = broadcastView;
        }

        /// <summary>Total number of elements yielded (the broadcast result size).</summary>
        public long size => _view.size;

        public IEnumerator<object> GetEnumerator()
        {
            long n = _view.size;
            for (long i = 0; i < n; i++)
                yield return _view.GetAtIndex(i);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
