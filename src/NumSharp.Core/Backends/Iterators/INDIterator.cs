using System;
using System.Collections;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public delegate ref T MoveNextReferencedDelegate<T>() where T : unmanaged;

    /// <summary>
    ///     REMOVED. The legacy per-element NDIterator has been retired. Kernel iteration now
    ///     goes through <c>NpyIter</c>/<c>NpyIterRef</c>, and <see cref="np.Broadcast.iters"/> is
    ///     backed by <see cref="NumSharp.Backends.Iteration.NpyFlatIterator"/>. This interface is
    ///     kept only as an <see cref="ObsoleteAttribute"/> tombstone for source/binary compatibility;
    ///     the sole implementation (<see cref="NDIterator{TOut}"/>) throws on every member.
    /// </summary>
    [Obsolete("NDIterator has been removed. Use NpyIter/NpyIterRef for kernels, iterate the NDArray directly, or np.broadcast(...).iters (NpyFlatIterator).", false)]
    public interface NDIterator : IEnumerable
    {
        IMemoryBlock Block { get; }
        Shape Shape { get; }
        Shape? BroadcastedShape { get; }
        bool AutoReset { get; }

        Func<T> MoveNext<T>() where T : unmanaged;
        MoveNextReferencedDelegate<T> MoveNextReference<T>() where T : unmanaged;

        Func<bool> HasNext { get; }
        Action Reset { get; }
    }
}
