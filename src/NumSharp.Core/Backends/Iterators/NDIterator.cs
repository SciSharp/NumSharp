using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    /// Lazy per-element iterator. Supports contiguous/sliced/strided/broadcast
    /// source layouts and any source-to-TOut numeric dtype cast, without
    /// materializing a copy of the iterated data.
    ///
    /// Path selection at construction time picks the fastest MoveNext for the
    /// concrete layout + cast combination:
    ///
    /// <list type="bullet">
    ///   <item>Same-type contiguous (offset = 0, no AutoReset): direct
    ///     <c>*(TOut*)(addr + cursor++)</c> — one pointer increment per call.</item>
    ///   <item>Same-type strided or offset != 0: walks offsets via
    ///     <see cref="ValueOffsetIncrementor"/> / <see cref="ValueOffsetIncrementorAutoresetting"/>,
    ///     reads <c>*(TOut*)(addr + offset)</c>.</item>
    ///   <item>Cross-type: reads the source bytes as the actual src dtype, passes
    ///     through <see cref="Converts.FindConverter{TIn, TOut}"/>, and returns
    ///     the converted TOut. MoveNextReference throws — references into a
    ///     cast value don't exist.</item>
    /// </list>
    ///
    /// AutoReset on non-broadcast iteration is implemented via the incrementor's
    /// auto-resetting wrapper (or modulo on the contig-scalar-cursor path) so
    /// iteration cycles forever without allocating.
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

        /// <summary>Moves to next iteration and returns a reference to the next value. Throws when iteration involves a dtype cast.</summary>
        public MoveNextReferencedDelegate<TOut> MoveNextReference;

        /// <summary>Returns whether there are more elements to iterate.</summary>
        public Func<bool> HasNext;

        /// <summary>Resets the internal cursor to the beginning.</summary>
        public Action Reset;

        private bool _disposed;

        public NDIterator(IMemoryBlock block, Shape shape, Shape? broadcastedShape, bool autoReset = false)
        {
            if (shape.IsEmpty || shape.size == 0)
                throw new InvalidOperationException("Can't construct NDIterator with an empty shape.");

            Block = block ?? throw new ArgumentNullException(nameof(block));
            Shape = shape;
            BroadcastedShape = broadcastedShape;
            size = broadcastedShape?.size ?? shape.size;
            AutoReset = (broadcastedShape.HasValue && shape.size != broadcastedShape.Value.size) || autoReset;

            SetDefaults();
        }

        public NDIterator(IArraySlice slice, Shape shape, Shape? broadcastedShape, bool autoReset = false)
            : this((IMemoryBlock)slice, shape, broadcastedShape, autoReset) { }

        public NDIterator(UnmanagedStorage storage, bool autoReset = false)
            : this((IMemoryBlock)storage?.InternalArray, storage?.Shape ?? default, null, autoReset) { }

        public NDIterator(NDArray arr, bool autoReset = false)
            : this(arr?.Storage.InternalArray, arr?.Shape ?? default, null, autoReset) { }

        /// <summary>Reconfigure the iterator after construction.</summary>
        public void SetMode(bool autoreset, Shape reshape = default)
        {
            AutoReset = autoreset;
            if (!reshape.IsEmpty)
            {
                Shape = reshape;
                size = BroadcastedShape?.size ?? Shape.size;
            }
            SetDefaults();
        }

        private void SetDefaults()
        {
            var srcType = Block.TypeCode;
            var dstType = InfoOf<TOut>.NPTypeCode;

            if (srcType == dstType)
            {
                SetDefaults_NoCast();
                return;
            }

            SetDefaults_WithCast(srcType);
        }

        // ---------------------------------------------------------------------
        // Same-type (no cast) — direct pointer reads. Four sub-paths depending
        // on whether the shape is contiguous-with-zero-offset and whether
        // AutoReset is active.
        // ---------------------------------------------------------------------

        private void SetDefaults_NoCast()
        {
            var localBlock = Block;
            var localShape = Shape;

            if (localShape.IsContiguous && localShape.offset == 0)
            {
                if (AutoReset)
                {
                    long localSize = localShape.size;
                    long cursor = 0;
                    MoveNext = () =>
                    {
                        TOut ret = *((TOut*)localBlock.Address + cursor);
                        cursor++;
                        if (cursor >= localSize) cursor = 0;
                        return ret;
                    };
                    MoveNextReference = () =>
                    {
                        ref TOut r = ref Unsafe.AsRef<TOut>((TOut*)localBlock.Address + cursor);
                        cursor++;
                        if (cursor >= localSize) cursor = 0;
                        return ref r;
                    };
                    Reset = () => cursor = 0;
                    HasNext = () => true;
                }
                else
                {
                    long localSize = size;
                    long cursor = 0;
                    MoveNext = () => *((TOut*)localBlock.Address + cursor++);
                    MoveNextReference = () => ref Unsafe.AsRef<TOut>((TOut*)localBlock.Address + cursor++);
                    Reset = () => cursor = 0;
                    HasNext = () => cursor < localSize;
                }
                return;
            }

            // Strided / sliced / broadcast — walk offsets via the incrementor.
            if (AutoReset)
            {
                var incr = new ValueOffsetIncrementorAutoresetting(localShape);
                MoveNext = () => *((TOut*)localBlock.Address + incr.Next());
                MoveNextReference = () => ref Unsafe.AsRef<TOut>((TOut*)localBlock.Address + incr.Next());
                Reset = () => incr.Reset();
                HasNext = () => true;
            }
            else
            {
                var incr = new ValueOffsetIncrementor(localShape);
                MoveNext = () => *((TOut*)localBlock.Address + incr.Next());
                MoveNextReference = () => ref Unsafe.AsRef<TOut>((TOut*)localBlock.Address + incr.Next());
                Reset = () => incr.Reset();
                HasNext = () => incr.HasNext;
            }
        }

        // ---------------------------------------------------------------------
        // Cross-type — same offset-walking strategy, plus a Converts.FindConverter
        // step that turns the bytes at the source pointer into TOut. MoveNextReference
        // is not meaningful when a conversion happens, so it throws.
        // ---------------------------------------------------------------------

        private void SetDefaults_WithCast(NPTypeCode srcType)
        {
            MoveNextReference = () => throw new NotSupportedException(
                "Unable to return references during iteration when casting is involved.");

            switch (srcType)
            {
                case NPTypeCode.Boolean: BuildCastingMoveNext<bool>(); break;
                case NPTypeCode.Byte: BuildCastingMoveNext<byte>(); break;
                case NPTypeCode.Int16: BuildCastingMoveNext<short>(); break;
                case NPTypeCode.UInt16: BuildCastingMoveNext<ushort>(); break;
                case NPTypeCode.Int32: BuildCastingMoveNext<int>(); break;
                case NPTypeCode.UInt32: BuildCastingMoveNext<uint>(); break;
                case NPTypeCode.Int64: BuildCastingMoveNext<long>(); break;
                case NPTypeCode.UInt64: BuildCastingMoveNext<ulong>(); break;
                case NPTypeCode.Char: BuildCastingMoveNext<char>(); break;
                case NPTypeCode.Single: BuildCastingMoveNext<float>(); break;
                case NPTypeCode.Double: BuildCastingMoveNext<double>(); break;
                case NPTypeCode.Decimal: BuildCastingMoveNext<decimal>(); break;
                default: throw new NotSupportedException($"NDIterator: source dtype {srcType} not supported.");
            }
        }

        private void BuildCastingMoveNext<TSrc>() where TSrc : unmanaged
        {
            var conv = Converts.FindConverter<TSrc, TOut>();
            var localBlock = Block;
            var localShape = Shape;

            if (localShape.IsContiguous && localShape.offset == 0)
            {
                if (AutoReset)
                {
                    long localSize = localShape.size;
                    long cursor = 0;
                    MoveNext = () =>
                    {
                        TSrc v = *((TSrc*)localBlock.Address + cursor);
                        cursor++;
                        if (cursor >= localSize) cursor = 0;
                        return conv(v);
                    };
                    Reset = () => cursor = 0;
                    HasNext = () => true;
                }
                else
                {
                    long localSize = size;
                    long cursor = 0;
                    MoveNext = () => conv(*((TSrc*)localBlock.Address + cursor++));
                    Reset = () => cursor = 0;
                    HasNext = () => cursor < localSize;
                }
                return;
            }

            if (AutoReset)
            {
                var incr = new ValueOffsetIncrementorAutoresetting(localShape);
                MoveNext = () => conv(*((TSrc*)localBlock.Address + incr.Next()));
                Reset = () => incr.Reset();
                HasNext = () => true;
            }
            else
            {
                var incr = new ValueOffsetIncrementor(localShape);
                MoveNext = () => conv(*((TSrc*)localBlock.Address + incr.Next()));
                Reset = () => incr.Reset();
                HasNext = () => incr.HasNext;
            }
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
            MoveNext = null;
            Reset = null;
            HasNext = null;
            MoveNextReference = null;
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
