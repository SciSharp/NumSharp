using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     REMOVED. NDIterator (the legacy per-element iterator) has been retired. It was already a
    ///     thin wrapper over an owned <c>NpyIterState</c>; all real iteration now goes through
    ///     <c>NpyIter</c>/<c>NpyIterRef</c> (kernels, reductions, copy) and
    ///     <see cref="NumSharp.Backends.Iteration.NpyFlatIterator"/> (np.broadcast(...).iters).
    ///
    ///     This type is preserved ONLY as an <see cref="ObsoleteAttribute"/> tombstone so existing
    ///     source/binaries that name it keep compiling; every constructor and member throws
    ///     <see cref="NotSupportedException"/>. To iterate an <see cref="NDArray"/> element-by-element,
    ///     enumerate it directly or read elements via <c>GetAtIndex</c>.
    /// </summary>
    [Obsolete("NDIterator has been removed. Use NpyIter/NpyIterRef for kernel iteration, iterate the NDArray directly, or np.broadcast(...).iters (NpyFlatIterator). Every member throws NotSupportedException.", false)]
    public unsafe partial class NDIterator<TOut> : NDIterator, IEnumerable<TOut>, IDisposable
        where TOut : unmanaged
    {
        private const string Removed =
            "NDIterator has been removed. Use NpyIter/NpyIterRef, iterate the NDArray directly, or np.broadcast(...).iters (NpyFlatIterator).";

        public NDIterator(NDArray arr, bool autoReset = false) => throw new NotSupportedException(Removed);
        public NDIterator(UnmanagedStorage storage, bool autoReset = false) => throw new NotSupportedException(Removed);
        public NDIterator(IArraySlice slice, Shape shape, Shape? broadcastedShape, bool autoReset = false) => throw new NotSupportedException(Removed);
        public NDIterator(IMemoryBlock block, Shape shape, Shape? broadcastedShape, bool autoReset = false) => throw new NotSupportedException(Removed);

        public IEnumerator<TOut> GetEnumerator() => throw new NotSupportedException(Removed);
        IEnumerator IEnumerable.GetEnumerator() => throw new NotSupportedException(Removed);

        public void Dispose() { }

        // Explicit non-generic NDIterator surface — all throwing.
        IMemoryBlock NDIterator.Block => throw new NotSupportedException(Removed);
        Shape NDIterator.Shape => throw new NotSupportedException(Removed);
        Shape? NDIterator.BroadcastedShape => throw new NotSupportedException(Removed);
        bool NDIterator.AutoReset => throw new NotSupportedException(Removed);
        Func<T1> NDIterator.MoveNext<T1>() => throw new NotSupportedException(Removed);
        MoveNextReferencedDelegate<T1> NDIterator.MoveNextReference<T1>() => throw new NotSupportedException(Removed);
        Func<bool> NDIterator.HasNext => throw new NotSupportedException(Removed);
        Action NDIterator.Reset => throw new NotSupportedException(Removed);
    }
}
