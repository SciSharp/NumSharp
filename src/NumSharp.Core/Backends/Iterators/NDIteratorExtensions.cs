using System;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     REMOVED. The <c>AsIterator</c> factory built the legacy <see cref="NDIterator{TOut}"/>,
    ///     which has been retired in favor of <c>NpyIter</c>/<c>NpyIterRef</c> (kernels) and
    ///     <see cref="NumSharp.Backends.Iteration.NpyFlatIterator"/> (np.broadcast(...).iters).
    ///     These overloads are kept only as <see cref="ObsoleteAttribute"/> tombstones for
    ///     source/binary compatibility; each throws <see cref="NotSupportedException"/>.
    /// </summary>
    [Obsolete("AsIterator/NDIterator have been removed. Iterate the NDArray directly, or use NpyIter/NpyIterRef. Every overload throws NotSupportedException.", false)]
    public static class NDIteratorExtensions
    {
        private const string Removed =
            "AsIterator/NDIterator have been removed. Iterate the NDArray directly, or use NpyIter/NpyIterRef.";

        public static NDIterator<T> AsIterator<T>(this NDArray nd, bool autoreset = false) where T : unmanaged
            => throw new NotSupportedException(Removed);

        public static NDIterator AsIterator(this NDArray nd, bool autoreset = false)
            => throw new NotSupportedException(Removed);

        public static NDIterator AsIterator(this UnmanagedStorage us, bool autoreset = false)
            => throw new NotSupportedException(Removed);

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape)
            => throw new NotSupportedException(Removed);

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, bool autoreset)
            => throw new NotSupportedException(Removed);

        public static NDIterator AsIterator(this IArraySlice arr, Shape shape, Shape broadcastShape, bool autoReset)
            => throw new NotSupportedException(Removed);
    }
}
