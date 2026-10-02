using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.MLNet
{
    /// <summary>
    ///     An ML.NET <see cref="IDataView"/> that reads a NumSharp <see cref="NDArray"/> — the view
    ///     <see cref="NDArrayMLNetInterop.AsDataView(NDArray, string)"/> /
    ///     <see cref="NDArrayMLNetInterop.ToDataView(NDArray, string)"/> return. It holds an ARC reference on the
    ///     array's buffer and reads rows <b>lazily</b> through the array's own strides (so any layout works — C /
    ///     Fortran / sliced / transposed / negative-stride / broadcast — with no densifying copy), producing each
    ///     value or <see cref="VBuffer{T}"/> only when an ML.NET cursor asks for it.
    ///
    ///     <para><b>Lifetime.</b> <see cref="IDataView"/> is not <see cref="IDisposable"/>, so this concrete type
    ///     adds it: <see cref="Dispose"/> releases the buffer pin (and disposes the internal Half→Single /
    ///     Decimal→Double / snapshot temporary, if any), and a finalizer is the safety net. Every cursor takes its
    ///     own pin, so a cursor mid-iteration keeps the buffer alive even if the view is disposed underneath it.
    ///     Because the pin survives an explicit <c>source.Dispose()</c>, it is safe to dispose the source array
    ///     while the view (or its cursors) still read it — unlike a raw ML.NET data source over a managed array.</para>
    /// </summary>
    public sealed class NDArrayDataView : IDataView, IDisposable
    {
        // Per output column: how (row, withinVector) maps to a physical element index of the fed array, and the
        // dtype the column carries. physical(row, j) = ElemBase + row*RowStride + j*InnerStride  (j == 0 for scalar).
        internal readonly struct ColumnPlan
        {
            public readonly string Name;
            public readonly NPTypeCode ItemCode;
            public readonly bool IsVector;
            public readonly int VectorWidth;
            public readonly long ElemBase;
            public readonly long RowStride;
            public readonly long InnerStride;

            public ColumnPlan(string name, NPTypeCode itemCode, bool isVector, int vectorWidth, long elemBase, long rowStride, long innerStride)
            {
                Name = name;
                ItemCode = itemCode;
                IsVector = isVector;
                VectorWidth = vectorWidth;
                ElemBase = elemBase;
                RowStride = rowStride;
                InnerStride = innerStride;
            }
        }

        private readonly NDArray _fed;          // the array actually read (source, or a converted / snapshot temporary)
        private readonly IArraySlice _slice;    // the view's own ARC pin on _fed's buffer
        private readonly bool _ownsFed;         // dispose _fed on Dispose (converted / snapshot temporary)
        private readonly ColumnPlan[] _plans;
        private readonly DataViewSchema _schema;
        private readonly long _rowCount;
        private readonly IntPtr _baseAddr;      // _slice.Address — storage element 0 (offsets are baked into the plans)
        private readonly long _itemsize;
        private int _disposed;

        internal unsafe NDArrayDataView(NDArray fed, IArraySlice slice, bool ownsFed, ColumnPlan[] plans, DataViewSchema schema, long rowCount)
        {
            _fed = fed;
            _slice = slice;
            _ownsFed = ownsFed;
            _plans = plans;
            _schema = schema;
            _rowCount = rowCount;
            _baseAddr = (IntPtr)slice.Address;
            _itemsize = fed.dtypesize;
            NDArrayMLNetInterop.ExportOpened();
        }

        /// <inheritdoc />
        public DataViewSchema Schema
        {
            get
            {
                ThrowIfDisposed();
                return _schema;
            }
        }

        /// <inheritdoc />
        /// <remarks>Random-access is possible, but shuffling is not implemented; ML.NET falls back to sequential.</remarks>
        public bool CanShuffle => false;

        /// <inheritdoc />
        public long? GetRowCount() => _rowCount;

        /// <inheritdoc />
        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            ThrowIfDisposed();
            return new NDArrayCursor(this, columnsNeeded);
        }

        /// <inheritdoc />
        /// <remarks>NumSharp reads are inexpensive and single-threaded here; one cursor is returned (ML.NET may split work itself).</remarks>
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };

        internal ColumnPlan Plan(int columnIndex)
        {
            if ((uint)columnIndex >= (uint)_plans.Length)
                throw new ArgumentOutOfRangeException(nameof(columnIndex), $"column index {columnIndex} is out of range for a schema of {_plans.Length} columns.");
            return _plans[columnIndex];
        }

        internal IArraySlice TakeCursorPin() => NDArrayMLNetInterop.Pin(_fed);   // one ARC ref per cursor
        internal IntPtr BaseAddr => _baseAddr;
        internal long ItemSize => _itemsize;
        internal long RowCount => _rowCount;

        /// <summary>Release the buffer pin (and the owned temporary, if any). Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            ReleaseCore();
        }

        /// <summary>Safety net for a view that was never disposed (drops the ARC pin).</summary>
        ~NDArrayDataView()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            ReleaseCore();
        }

        private void ReleaseCore()
        {
            _slice.Release();
            NDArrayMLNetInterop.ExportClosed();
            if (_ownsFed)
                _fed.Dispose();   // the Half→Single / Decimal→Double / snapshot temporary nobody else can dispose
            else
                GC.KeepAlive(_fed);
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(NDArrayDataView), "the NDArray-backed IDataView has been disposed; its NumSharp buffer is no longer pinned.");
        }

        // =====================================  the cursor  =====================================

        /// <summary>
        ///     The row cursor, following ML.NET's cursor contract exactly as ML.NET's own cursors do (probed against its
        ///     in-memory view, cache, transforms and row filters): <see cref="Position"/> is -1 before the first row, after
        ///     <see cref="MoveNext"/> returned false and after <c>Dispose</c>; a getter called anywhere but on a row throws
        ///     ML.NET's own <see cref="InvalidOperationException"/> instead of reading. That check is what keeps the cursor
        ///     inside the array: the address of a value is computed from the position, so a getter called before the first
        ///     row or after the last used to READ OUTSIDE THE ARRAY (position -1 or <c>rowCount</c>), and one called after
        ///     <c>Dispose</c> read a buffer whose pin was gone.
        /// </summary>
        private sealed class NDArrayCursor : DataViewRowCursor
        {
            /// <summary>ML.NET's own text for a getter called outside a row (its <c>RowCursorUtils.FetchValueStateError</c>).</summary>
            private const string FetchValueStateError =
                "Values cannot be fetched at this time. This method was called either before the first call to MoveNext, " +
                "or at any point after that method returned false.";

            private readonly NDArrayDataView _view;
            private readonly DataViewSchema _schema; // captured: the cursor answers Schema after its view is disposed (the view's own property throws then)
            private readonly IArraySlice _slice;     // this cursor's own ARC pin (survives the view being disposed)
            private readonly IntPtr _baseAddr;
            private readonly long _itemsize;
            private readonly long _rowCount;
            private readonly bool[] _active;
            private long _position = -1;             // -1 off a row: before the first, after the last, after Dispose (ML.NET's convention)
            private bool _done;                      // MoveNext returned false or the cursor was disposed: it never moves again
            private int _disposed;

            /// <summary>Opens a cursor over <paramref name="view"/>, taking the cursor's own pin on its buffer.</summary>
            /// <param name="view">The (undisposed) view to read.</param>
            /// <param name="columnsNeeded">The columns to activate; columns of another schema (index out of range) are ignored.</param>
            /// <exception cref="ObjectDisposedException">The view's buffer has already been released.</exception>
            internal NDArrayCursor(NDArrayDataView view, IEnumerable<DataViewSchema.Column> columnsNeeded)
            {
                _view = view;
                _schema = view.Schema;
                _slice = view.TakeCursorPin();
                _baseAddr = view.BaseAddr;
                _itemsize = view.ItemSize;
                _rowCount = view.RowCount;
                _active = new bool[_schema.Count];
                if (columnsNeeded is not null)
                {
                    foreach (DataViewSchema.Column c in columnsNeeded)
                        if ((uint)c.Index < (uint)_active.Length)
                            _active[c.Index] = true;
                }
                NDArrayMLNetInterop.ExportOpened();
            }

            /// <inheritdoc />
            public override long Position => _position;

            /// <inheritdoc />
            public override long Batch => 0;

            /// <inheritdoc />
            /// <remarks>The schema captured when the cursor opened, so it stays readable after the view is disposed.</remarks>
            public override DataViewSchema Schema => _schema;

            /// <inheritdoc />
            /// <remarks>
            ///     Once it has returned false — or the cursor was disposed — it returns false forever, with
            ///     <see cref="Position"/> back at -1, as ML.NET's own cursors do.
            /// </remarks>
            public override bool MoveNext()
            {
                if (_done)
                    return false;
                long next = _position + 1;
                if (next < _rowCount)
                {
                    _position = next;
                    return true;
                }
                _done = true;
                _position = -1;
                return false;
            }

            /// <inheritdoc />
            public override bool IsColumnActive(DataViewSchema.Column column)
                => (uint)column.Index < (uint)_active.Length && _active[column.Index];

            /// <inheritdoc />
            /// <remarks>Row ids are the row numbers; outside a row the getter throws like the value getters.</remarks>
            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId id) => id = new DataViewRowId((ulong)CurrentRow(), 0);

            /// <inheritdoc />
            /// <exception cref="ArgumentOutOfRangeException">The column is not in this cursor's schema, or not active.</exception>
            /// <exception cref="InvalidOperationException"><typeparamref name="TValue"/> does not match the column's type.</exception>
            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                NDArrayDataView.ColumnPlan p = _view.Plan(column.Index);
                if (!IsColumnActive(column))
                    throw new ArgumentOutOfRangeException(nameof(column), $"column '{p.Name}' (index {column.Index}) is not active in this cursor.");
                return p.IsVector ? BuildVectorGetter<TValue>(p) : BuildScalarGetter<TValue>(p);
            }

            /// <summary>
            ///     The row the cursor is on — the only position a getter may read. Off a row the position is -1, which as an
            ///     unsigned number is never below the row count, so one comparison refuses every off-row state: before the
            ///     first row, after the last, and after <c>Dispose</c> (which parks the position at -1 too).
            /// </summary>
            /// <returns>The current row index, in <c>[0, rowCount)</c>.</returns>
            /// <exception cref="InvalidOperationException">The cursor is not on a row; ML.NET's own message.</exception>
            private long CurrentRow()
            {
                long position = _position;
                if ((ulong)position >= (ulong)_rowCount)
                    throw new InvalidOperationException(FetchValueStateError);
                return position;
            }

            // ---- scalar getter: TValue IS the element type X ----------------------------------------
            /// <summary>Builds the getter for a scalar column: one element read per row, through the column's strides.</summary>
            /// <typeparam name="TValue">The requested value type; must be the column's CLR type.</typeparam>
            /// <param name="p">The column's plan.</param>
            /// <returns>The getter.</returns>
            /// <exception cref="InvalidOperationException"><typeparamref name="TValue"/> does not match the column's type.</exception>
            private ValueGetter<TValue> BuildScalarGetter<TValue>(NDArrayDataView.ColumnPlan p)
            {
                Type want = NDArrayMLNetInterop.ToDataViewClrType(p.ItemCode);
                if (typeof(TValue) != want)
                    throw new InvalidOperationException($"column '{p.Name}' is {p.ItemCode} (ML.NET {want.Name}); a ValueGetter<{typeof(TValue).Name}> does not match. Request GetGetter<{want.Name}>.");
                long elemBase = p.ElemBase, rowStride = p.RowStride, itemsize = _itemsize;
                IntPtr baseAddr = _baseAddr;
                ValueGetter<TValue> getter = (ref TValue value) =>
                {
                    long phys = elemBase + CurrentRow() * rowStride;
                    unsafe { value = Unsafe.Read<TValue>((void*)(baseAddr + (nint)(phys * itemsize))); }
                };
                return getter;
            }

            // ---- vector getter: TValue is VBuffer<X>; dispatch by the item dtype to build ValueGetter<VBuffer<X>> --
            /// <summary>Builds the getter for a vector column, dispatching on the column's item dtype.</summary>
            /// <typeparam name="TValue">The requested value type; must be <c>VBuffer</c> of the column's item type.</typeparam>
            /// <param name="p">The column's plan.</param>
            /// <returns>The getter.</returns>
            /// <exception cref="InvalidOperationException"><typeparamref name="TValue"/> does not match the column's type.</exception>
            /// <exception cref="NotSupportedException">The item dtype has no ML.NET type (never, for a view this bridge built).</exception>
            private ValueGetter<TValue> BuildVectorGetter<TValue>(NDArrayDataView.ColumnPlan p)
            {
                object getter = p.ItemCode switch
                {
                    NPTypeCode.Boolean => MakeVectorGetter<bool>(p),
                    NPTypeCode.Byte => MakeVectorGetter<byte>(p),
                    NPTypeCode.SByte => MakeVectorGetter<sbyte>(p),
                    NPTypeCode.Int16 => MakeVectorGetter<short>(p),
                    NPTypeCode.UInt16 => MakeVectorGetter<ushort>(p),
                    NPTypeCode.Char => MakeVectorGetter<ushort>(p),   // UTF-16 code units
                    NPTypeCode.Int32 => MakeVectorGetter<int>(p),
                    NPTypeCode.UInt32 => MakeVectorGetter<uint>(p),
                    NPTypeCode.Int64 => MakeVectorGetter<long>(p),
                    NPTypeCode.UInt64 => MakeVectorGetter<ulong>(p),
                    NPTypeCode.Single => MakeVectorGetter<float>(p),
                    NPTypeCode.Double => MakeVectorGetter<double>(p),
                    _ => throw new NotSupportedException(NDArrayMLNetInterop.UnsupportedExportMessage(p.ItemCode)),
                };
                if (getter is not ValueGetter<TValue> typed)
                    throw new InvalidOperationException($"column '{p.Name}' is a vector of {p.ItemCode}; a ValueGetter<{typeof(TValue).Name}> does not match. Request GetGetter<VBuffer<{NDArrayMLNetInterop.ToDataViewClrType(p.ItemCode).Name}>>.");
                return typed;
            }

            /// <summary>
            ///     The vector getter for item type <typeparamref name="X"/>: each call fills the caller's buffer with the
            ///     current row (reusing its array, as ML.NET's getters do) through the column's strides.
            /// </summary>
            /// <typeparam name="X">The item CLR type.</typeparam>
            /// <param name="p">The column's plan.</param>
            /// <returns>The getter.</returns>
            private ValueGetter<VBuffer<X>> MakeVectorGetter<X>(NDArrayDataView.ColumnPlan p) where X : unmanaged
            {
                long elemBase = p.ElemBase, rowStride = p.RowStride, innerStride = p.InnerStride, itemsize = _itemsize;
                int width = p.VectorWidth;
                IntPtr baseAddr = _baseAddr;
                return (ref VBuffer<X> dst) =>
                {
                    // Position checked FIRST: off a row nothing may be read (or the caller's buffer resized).
                    long rowBase = elemBase + CurrentRow() * rowStride;
                    VBufferEditor<X> editor = VBufferEditor.Create(ref dst, width);
                    Span<X> vals = editor.Values;
                    unsafe
                    {
                        if (innerStride == 1)
                        {
                            // contiguous run — one bulk read
                            var src = new ReadOnlySpan<X>((void*)(baseAddr + (nint)(rowBase * itemsize)), width);
                            src.CopyTo(vals);
                        }
                        else
                        {
                            for (int j = 0; j < width; j++)
                                vals[j] = Unsafe.Read<X>((void*)(baseAddr + (nint)((rowBase + (long)j * innerStride) * itemsize)));
                        }
                    }
                    dst = editor.Commit();
                };
            }

            /// <summary>
            ///     Safety net for a cursor nobody disposed: releases its pin when the cursor is collected. ML.NET itself
            ///     abandons cursors — its row-shuffling transform (<c>ShuffleRows</c>, used by training pipelines) never
            ///     disposes the cursor it opens on its input, even after its own cursor is disposed — and without this a
            ///     single shuffle kept the NumSharp buffer referenced forever: the pool never got it back, and
            ///     <c>resize(refcheck)</c> refused for the life of the process. The view and every export handle have the
            ///     same net; the cursor was the one object without it.
            /// </summary>
            ~NDArrayCursor()
            {
                Dispose(false);
            }

            /// <summary>
            ///     Releases the cursor's pin, once (an explicit dispose and the finalizer race through the same interlocked
            ///     guard). The cursor is parked off any row first (position -1, done), so a getter still held by the caller
            ///     throws instead of reading a buffer this pin no longer keeps alive.
            /// </summary>
            /// <param name="disposing"><c>true</c> from <c>Dispose()</c>, <c>false</c> from the finalizer; the pin is released either way.</param>
            protected override void Dispose(bool disposing)
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                {
                    _done = true;
                    _position = -1;
                    _slice.Release();
                    NDArrayMLNetInterop.ExportClosed();
                    GC.KeepAlive(_view);
                }
                base.Dispose(disposing);
            }
        }
    }
}
