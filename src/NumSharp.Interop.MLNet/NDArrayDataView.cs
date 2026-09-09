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

        private sealed class NDArrayCursor : DataViewRowCursor
        {
            private readonly NDArrayDataView _view;
            private readonly IArraySlice _slice;   // this cursor's own ARC pin (survives the view being disposed)
            private readonly IntPtr _baseAddr;
            private readonly long _itemsize;
            private readonly long _rowCount;
            private readonly bool[] _active;
            private long _position = -1;
            private int _disposed;

            internal NDArrayCursor(NDArrayDataView view, IEnumerable<DataViewSchema.Column> columnsNeeded)
            {
                _view = view;
                _slice = view.TakeCursorPin();
                _baseAddr = view.BaseAddr;
                _itemsize = view.ItemSize;
                _rowCount = view.RowCount;
                _active = new bool[view.Schema.Count];
                if (columnsNeeded is not null)
                {
                    foreach (DataViewSchema.Column c in columnsNeeded)
                        if ((uint)c.Index < (uint)_active.Length)
                            _active[c.Index] = true;
                }
                NDArrayMLNetInterop.ExportOpened();
            }

            public override long Position => _position;
            public override long Batch => 0;
            public override DataViewSchema Schema => _view.Schema;

            public override bool MoveNext()
            {
                if (Volatile.Read(ref _disposed) != 0)
                    return false;
                if (_position + 1 >= _rowCount)
                {
                    _position = _rowCount;   // parked past the end (matches ML.NET's cursor contract)
                    return false;
                }
                _position++;
                return true;
            }

            public override bool IsColumnActive(DataViewSchema.Column column)
                => (uint)column.Index < (uint)_active.Length && _active[column.Index];

            public override ValueGetter<DataViewRowId> GetIdGetter()
                => (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

            public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
            {
                NDArrayDataView.ColumnPlan p = _view.Plan(column.Index);
                if (!IsColumnActive(column))
                    throw new ArgumentOutOfRangeException(nameof(column), $"column '{p.Name}' (index {column.Index}) is not active in this cursor.");
                return p.IsVector ? BuildVectorGetter<TValue>(p) : BuildScalarGetter<TValue>(p);
            }

            // ---- scalar getter: TValue IS the element type X ----------------------------------------
            private ValueGetter<TValue> BuildScalarGetter<TValue>(NDArrayDataView.ColumnPlan p)
            {
                Type want = NDArrayMLNetInterop.ToDataViewClrType(p.ItemCode);
                if (typeof(TValue) != want)
                    throw new InvalidOperationException($"column '{p.Name}' is {p.ItemCode} (ML.NET {want.Name}); a ValueGetter<{typeof(TValue).Name}> does not match. Request GetGetter<{want.Name}>.");
                long elemBase = p.ElemBase, rowStride = p.RowStride, itemsize = _itemsize;
                IntPtr baseAddr = _baseAddr;
                ValueGetter<TValue> getter = (ref TValue value) =>
                {
                    long phys = elemBase + _position * rowStride;
                    unsafe { value = Unsafe.Read<TValue>((void*)(baseAddr + (nint)(phys * itemsize))); }
                };
                return getter;
            }

            // ---- vector getter: TValue is VBuffer<X>; dispatch by the item dtype to build ValueGetter<VBuffer<X>> --
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

            private ValueGetter<VBuffer<X>> MakeVectorGetter<X>(NDArrayDataView.ColumnPlan p) where X : unmanaged
            {
                long elemBase = p.ElemBase, rowStride = p.RowStride, innerStride = p.InnerStride, itemsize = _itemsize;
                int width = p.VectorWidth;
                IntPtr baseAddr = _baseAddr;
                return (ref VBuffer<X> dst) =>
                {
                    VBufferEditor<X> editor = VBufferEditor.Create(ref dst, width);
                    Span<X> vals = editor.Values;
                    long rowBase = elemBase + _position * rowStride;
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

            protected override void Dispose(bool disposing)
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                {
                    _slice.Release();
                    NDArrayMLNetInterop.ExportClosed();
                    GC.KeepAlive(_view);
                }
                base.Dispose(disposing);
            }
        }
    }
}
