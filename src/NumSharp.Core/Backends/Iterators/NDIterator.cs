using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    /// Legacy per-element iterator surface preserved for backward compatibility.
    ///
    /// Internally this is now a thin wrapper over the modern <see cref="NpyIter"/>
    /// machinery — the iteration is pre-materialized into a flat TOut buffer via
    /// <see cref="NpyIter.Copy(UnmanagedStorage, UnmanagedStorage)"/> so that
    /// source layout (contiguous, sliced, broadcast, transposed) and source-to-
    /// TOut dtype casting are both handled once up front. The resulting buffer
    /// is then walked by the <see cref="MoveNext"/>, <see cref="HasNext"/>,
    /// and <see cref="Reset"/> delegates.
    ///
    /// Trade-off: iteration allocates O(size) memory for the materialized buffer.
    /// In exchange, per-element MoveNext is a simple pointer index with no
    /// delegate dispatch or coordinate arithmetic in the hot path, and the
    /// dtype-dispatch switch that used to live in the 12 partial
    /// <c>NDIterator.Cast.&lt;T&gt;.cs</c> files is gone entirely.
    /// </summary>
    public unsafe class NDIterator<TOut> : NDIterator, IEnumerable<TOut>, IDisposable
        where TOut : unmanaged
    {
        public readonly IMemoryBlock Block;

        /// <summary>The shape this iterator iterates.</summary>
        public Shape Shape;

        /// <summary>The broadcasted version of <see cref="Shape"/>. Null when iterating an un-broadcasted shape.</summary>
        public Shape? BroadcastedShape;

        /// <summary>When true, <see cref="HasNext"/> always returns true and <see cref="MoveNext"/> wraps around at the end.</summary>
        public bool AutoReset;

        /// <summary>Total number of elements this iterator visits before (non-auto-reset) end.</summary>
        public long size;

        /// <summary>Moves to next iteration and returns the next value. Always check <see cref="HasNext"/> first.</summary>
        public Func<TOut> MoveNext;

        /// <summary>Moves to next iteration and returns a reference to the next value.</summary>
        public MoveNextReferencedDelegate<TOut> MoveNextReference;

        /// <summary>Returns whether there are more elements to iterate.</summary>
        public Func<bool> HasNext;

        /// <summary>Resets the internal cursor to the beginning.</summary>
        public Action Reset;

        // NpyIter-materialized backing storage. Owned by this iterator and released in Dispose().
        private NDArray _materialized;
        private long _cursor;
        private bool _disposed;

        public NDIterator(IMemoryBlock block, Shape shape, Shape? broadcastedShape, bool autoReset = false)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDIterator with an empty shape.");

            Block = block ?? throw new ArgumentNullException(nameof(block));
            Shape = shape;
            BroadcastedShape = broadcastedShape;
            long effSize = broadcastedShape?.size ?? shape.size;
            size = effSize;
            AutoReset = (broadcastedShape.HasValue && shape.size != broadcastedShape.Value.size) || autoReset;

            Materialize(block, shape, broadcastedShape);
            SetDelegates();
        }

        public NDIterator(IArraySlice slice, Shape shape, Shape? broadcastedShape, bool autoReset = false)
            : this((IMemoryBlock)slice, shape, broadcastedShape, autoReset) { }

        public NDIterator(UnmanagedStorage storage, bool autoReset = false)
            : this((IMemoryBlock)storage?.InternalArray, storage?.Shape ?? default, null, autoReset) { }

        public NDIterator(NDArray arr, bool autoReset = false)
            : this(arr?.Storage.InternalArray, arr?.Shape ?? default, null, autoReset) { }

        /// <summary>
        /// Reconfigure after construction. Any non-default <paramref name="reshape"/>
        /// triggers a re-materialization of the backing buffer at the new shape.
        /// </summary>
        public void SetMode(bool autoreset, Shape reshape = default)
        {
            AutoReset = autoreset;
            if (!reshape.IsEmpty)
            {
                Shape = reshape;
                size = BroadcastedShape?.size ?? Shape.size;
                Materialize(Block, Shape, BroadcastedShape);
                SetDelegates();
            }
        }

        private void Materialize(IMemoryBlock srcBlock, Shape srcShape, Shape? broadcastedShape)
        {
            var srcSlice = srcBlock as IArraySlice
                ?? throw new ArgumentException(
                    $"NDIterator expected source block to implement IArraySlice; got {srcBlock.GetType()}.");

            // Use CreateBroadcastedUnsafe to bypass the UnmanagedStorage ctor's
            // "shape.size == slice.Count" check — our srcShape can carry stride=0
            // broadcast axes whose logical size exceeds the backing slice.
            var srcStorage = UnmanagedStorage.CreateBroadcastedUnsafe(srcSlice, srcShape);

            // Destination must be freshly C-order-contiguous and writeable, even
            // when srcShape (or broadcastedShape) carries broadcast stride=0. Drop
            // the stride metadata by constructing the target shape from dimensions
            // only — this gives a fresh, writeable, row-major shape.
            var srcDims = broadcastedShape ?? srcShape;
            var targetShape = new Shape((long[])srcDims.dimensions.Clone());
            var targetTypeCode = InfoOf<TOut>.NPTypeCode;

            // NpyIter.Copy broadcasts src -> targetShape and casts
            // src.typecode -> TOut in one pass.
            _materialized = new NDArray(targetTypeCode, targetShape, false);
            NpyIter.Copy(_materialized.Storage, srcStorage);
        }

        private void SetDelegates()
        {
            _cursor = 0;
            MoveNext = DefaultMoveNext;
            HasNext = DefaultHasNext;
            Reset = DefaultReset;
            MoveNextReference = DefaultMoveNextReference;
        }

        private TOut DefaultMoveNext()
        {
            if (_cursor >= size)
            {
                if (AutoReset) _cursor = 0;
                else throw new InvalidOperationException("NDIterator: no more elements.");
            }
            return *((TOut*)_materialized.Address + _cursor++);
        }

        private bool DefaultHasNext() => AutoReset || _cursor < size;

        private void DefaultReset() => _cursor = 0;

        private ref TOut DefaultMoveNextReference()
        {
            if (_cursor >= size)
            {
                if (AutoReset) _cursor = 0;
                else throw new InvalidOperationException("NDIterator: no more elements.");
            }
            return ref Unsafe.AsRef<TOut>((TOut*)_materialized.Address + _cursor++);
        }

        public IEnumerator<TOut> GetEnumerator()
        {
            long n = size;
            for (long i = 0; i < n; i++)
                yield return ReadAt(i);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private TOut ReadAt(long i) => *((TOut*)_materialized.Address + i);

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Dispose()
        {
            if (_disposed) return;
            MoveNext = null;
            Reset = null;
            HasNext = null;
            MoveNextReference = null;
            _materialized = null;
            _disposed = true;
        }

        #region Explicit interface implementations for non-generic NDIterator

        IMemoryBlock NDIterator.Block => Block;
        Shape NDIterator.Shape => Shape;
        Shape? NDIterator.BroadcastedShape => BroadcastedShape;
        bool NDIterator.AutoReset => AutoReset;
        Func<T1> NDIterator.MoveNext<T1>() => (Func<T1>)(object)MoveNext;
        MoveNextReferencedDelegate<T1> NDIterator.MoveNextReference<T1>() => (MoveNextReferencedDelegate<T1>)(object)MoveNextReference;
        Func<bool> NDIterator.HasNext => HasNext;
        Action NDIterator.Reset => Reset;

        #endregion
    }
}
