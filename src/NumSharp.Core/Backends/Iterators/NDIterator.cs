using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    /// Per-element iterator backed by an owned <see cref="NpyIterState"/>.
    /// Contiguous / sliced / strided / broadcast layouts are handled by the
    /// NpyIter state machine itself — MoveNext reads through
    /// <see cref="NpyIterState.DataPtrs"/> and advances via
    /// <see cref="NpyIterState.Advance"/> at one element per call.
    /// AutoReset loops forever by resetting the state when IterIndex
    /// reaches IterEnd; Reset restarts from IterStart.
    ///
    /// Same-dtype: the TOut value is read directly from the source via the
    /// data pointer (<c>*(TOut*)_state->DataPtrs[0]</c>). MoveNextReference
    /// returns a <c>ref TOut</c> into the source buffer.
    ///
    /// Cross-dtype: the source bytes are interpreted as the declared source
    /// dtype, then pushed through <see cref="Converts.FindConverter{TSrc, TOut}"/>
    /// on each step. MoveNextReference throws because a converted value has
    /// no backing ref in the source.
    ///
    /// Lifecycle: this class owns the NpyIterState pointer. Dispose (or GC
    /// finalization via the explicit IDisposable call) frees the state via
    /// <see cref="NpyIterRef.FreeState"/>.
    /// </summary>
    public unsafe partial class NDIterator<TOut> : NDIterator, IEnumerable<TOut>, IDisposable
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

        public Func<TOut> MoveNext;
        public MoveNextReferencedDelegate<TOut> MoveNextReference;
        public Func<bool> HasNext;
        public Action Reset;

        private NpyIterState* _state;  // Owned; freed in Dispose
        private readonly NDArray _srcKeepAlive;  // GC-anchor: keeps the underlying storage alive while we hold its pointers
        private bool _disposed;

        public NDIterator(NDArray arr, bool autoReset = false)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var shape = arr.Shape;
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDIterator with an empty shape.");

            _srcKeepAlive = arr;
            Block = arr.Storage.InternalArray;
            Shape = shape;
            BroadcastedShape = null;
            size = shape.size;
            AutoReset = autoReset;

            _state = InitState(arr);
            SetDelegates(arr.GetTypeCode);
        }

        public NDIterator(UnmanagedStorage storage, bool autoReset = false)
            : this(StorageToNDArray(storage), autoReset) { }

        public NDIterator(IArraySlice slice, Shape shape, Shape? broadcastedShape, bool autoReset = false)
            : this((IMemoryBlock)slice, shape, broadcastedShape, autoReset) { }

        public NDIterator(IMemoryBlock block, Shape shape, Shape? broadcastedShape, bool autoReset = false)
        {
            if (block is null) throw new ArgumentNullException(nameof(block));
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDIterator with an empty shape.");

            Block = block;
            Shape = shape;
            BroadcastedShape = broadcastedShape;
            size = broadcastedShape?.size ?? shape.size;
            AutoReset = (broadcastedShape.HasValue && shape.size != broadcastedShape.Value.size) || autoReset;

            var srcSlice = block as IArraySlice
                ?? throw new ArgumentException(
                    $"NDIterator expected source block to implement IArraySlice; got {block.GetType()}.");

            // When broadcastedShape expands beyond shape, build an NDArray on
            // the broadcasted shape so NpyIter iterates the full (cyclical)
            // extent via stride=0 broadcast axes.
            var effShape = broadcastedShape.HasValue && shape.size != broadcastedShape.Value.size
                ? broadcastedShape.Value
                : shape;
            var srcStorage = UnmanagedStorage.CreateBroadcastedUnsafe(srcSlice, effShape);
            _srcKeepAlive = new NDArray(srcStorage);

            _state = InitState(_srcKeepAlive);
            SetDelegates(block.TypeCode);
        }

        private static NDArray StorageToNDArray(UnmanagedStorage storage)
        {
            if (storage is null) throw new ArgumentNullException(nameof(storage));
            return new NDArray(storage);
        }

        private static NpyIterState* InitState(NDArray arr)
        {
            // NpyIterRef.New builds state with stride/broadcast info. Transfer
            // ownership into our field so the state outlives the ref struct.
            //
            // NPY_CORDER forces traversal in the view's logical row-major
            // order — the contract NDIterator historically provides (e.g.
            // iterating a transposed (4, 3) view yields elements in the order
            // 0,4,8, 1,5,9, ...). The default NPY_KEEPORDER would reorder to
            // the underlying memory layout, which would silently break
            // callers of AsIterator that depend on logical order.
            var iter = NpyIterRef.New(
                arr,
                NpyIterGlobalFlags.None,
                NPY_ORDER.NPY_CORDER,
                NPY_CASTING.NPY_SAFE_CASTING);
            try
            {
                return iter.ReleaseState();
            }
            catch
            {
                iter.Dispose();
                throw;
            }
        }

        /// <summary>Reconfigure the iterator after construction.</summary>
        public void SetMode(bool autoreset, Shape reshape = default)
        {
            AutoReset = autoreset;
            if (!reshape.IsEmpty)
            {
                Shape = reshape;
                size = BroadcastedShape?.size ?? Shape.size;
            }
            // Rebuild delegates — AutoReset may have changed.
            SetDelegates(Block.TypeCode);
        }

        private void SetDelegates(NPTypeCode srcType)
        {
            var dstType = InfoOf<TOut>.NPTypeCode;
            HasNext = DefaultHasNext;
            Reset = DefaultReset;

            if (srcType == dstType)
            {
                MoveNext = SameType_MoveNext;
                MoveNextReference = SameType_MoveNextReference;
                return;
            }

            MoveNextReference = () => throw new NotSupportedException(
                "Unable to return references during iteration when casting is involved.");

            switch (srcType)
            {
                case NPTypeCode.Boolean: MoveNext = BuildCastingMoveNext<bool>(); break;
                case NPTypeCode.Byte: MoveNext = BuildCastingMoveNext<byte>(); break;
                case NPTypeCode.SByte: MoveNext = BuildCastingMoveNext<sbyte>(); break;
                case NPTypeCode.Int16: MoveNext = BuildCastingMoveNext<short>(); break;
                case NPTypeCode.UInt16: MoveNext = BuildCastingMoveNext<ushort>(); break;
                case NPTypeCode.Int32: MoveNext = BuildCastingMoveNext<int>(); break;
                case NPTypeCode.UInt32: MoveNext = BuildCastingMoveNext<uint>(); break;
                case NPTypeCode.Int64: MoveNext = BuildCastingMoveNext<long>(); break;
                case NPTypeCode.UInt64: MoveNext = BuildCastingMoveNext<ulong>(); break;
                case NPTypeCode.Char: MoveNext = BuildCastingMoveNext<char>(); break;
                case NPTypeCode.Half: MoveNext = BuildCastingMoveNext<Half>(); break;
                case NPTypeCode.Single: MoveNext = BuildCastingMoveNext<float>(); break;
                case NPTypeCode.Double: MoveNext = BuildCastingMoveNext<double>(); break;
                case NPTypeCode.Decimal: MoveNext = BuildCastingMoveNext<decimal>(); break;
                case NPTypeCode.Complex: MoveNext = BuildCastingMoveNext<Complex>(); break;
                default: throw new NotSupportedException($"NDIterator: source dtype {srcType} not supported.");
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void EnsureNext()
        {
            if (_state->IterIndex >= _state->IterEnd)
            {
                if (!AutoReset)
                    throw new InvalidOperationException("NDIterator: no more elements.");
                _state->Reset();
            }
        }

        private TOut SameType_MoveNext()
        {
            EnsureNext();
            TOut v = *(TOut*)_state->DataPtrs[0];
            _state->Advance();
            return v;
        }

        private ref TOut SameType_MoveNextReference()
        {
            EnsureNext();
            ref TOut r = ref Unsafe.AsRef<TOut>((TOut*)_state->DataPtrs[0]);
            _state->Advance();
            return ref r;
        }

        private Func<TOut> BuildCastingMoveNext<TSrc>() where TSrc : unmanaged
        {
            var conv = Converts.FindConverter<TSrc, TOut>();
            return () =>
            {
                EnsureNext();
                TSrc v = *(TSrc*)_state->DataPtrs[0];
                _state->Advance();
                return conv(v);
            };
        }

        private bool DefaultHasNext()
        {
            if (AutoReset) return true;
            return _state->IterIndex < _state->IterEnd;
        }

        private void DefaultReset()
        {
            _state->Reset();
        }

        public IEnumerator<TOut> GetEnumerator()
        {
            var next = MoveNext;
            var hasNext = HasNext;
            while (hasNext())
                yield return next();
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public void Dispose()
        {
            if (_disposed) return;
            if (_state != null)
            {
                NpyIterRef.FreeState(_state);
                _state = null;
            }
            MoveNext = null;
            Reset = null;
            HasNext = null;
            MoveNextReference = null;
            _disposed = true;
        }

        ~NDIterator()
        {
            if (!_disposed && _state != null)
            {
                NpyIterRef.FreeState(_state);
                _state = null;
            }
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
