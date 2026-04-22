using System;
using System.Collections;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public delegate ref T MoveNextReferencedDelegate<T>() where T : unmanaged;

    /// <summary>
    /// Non-generic NDIterator surface, preserved so that <see cref="np.Broadcast.iters"/>
    /// can expose iterators of mixed element types as a single array. Concrete
    /// implementations live in <see cref="NDIterator{TOut}"/>.
    /// </summary>
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
