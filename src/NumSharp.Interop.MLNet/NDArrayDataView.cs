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
    ///
    ///     <para><b>Reachable means valid.</b> Whatever can reach the view — the caller, an ML.NET transform's output, a
    ///     cursor, a getter delegate (its closure holds the cursor, the cursor the view) — keeps the pins and the buffer
    ///     alive; a getter keeps its cursor reachable until its read is complete, so a collection that lands while it
    ///     copies a row cannot release the memory under it. The finalizers of the view and of its cursors only ABANDON
    ///     their references (never the eager free at zero), so an alias of the buffer that the caller still holds stays
    ///     valid after they run, in any order.</para>
    /// </summary>
    public sealed class NDArrayDataView : IDataView, IDisposable
    {
        // Per output column: how (row, withinVector) maps to a physical element index of the array it reads, and the
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

            /// <summary>The index of the array the column reads, in the view's sources (0 for a view of one array).</summary>
            public readonly int Source;

            /// <summary>Describes one column: its name and dtype, its shape (scalar or vector), and where its elements lie.</summary>
            /// <param name="name">The column's name in the schema.</param>
            /// <param name="itemCode">The dtype the column carries (the fed dtype, after any Half / Decimal conversion).</param>
            /// <param name="isVector"><c>true</c> for a vector column (each row a <see cref="VBuffer{T}"/>), <c>false</c> for a scalar column.</param>
            /// <param name="vectorWidth">The elements per row of a vector column (0 for a scalar column).</param>
            /// <param name="elemBase">The physical element index of row 0's first element (the array's offset baked in).</param>
            /// <param name="rowStride">The elements from one row to the next.</param>
            /// <param name="innerStride">The elements from one vector element to the next (0 for a scalar column).</param>
            /// <param name="source">The index of the array the column reads in the view's sources (0 for a view of one array).</param>
            public ColumnPlan(string name, NPTypeCode itemCode, bool isVector, int vectorWidth, long elemBase, long rowStride, long innerStride, int source = 0)
            {
                Name = name;
                ItemCode = itemCode;
                IsVector = isVector;
                VectorWidth = vectorWidth;
                ElemBase = elemBase;
                RowStride = rowStride;
                InnerStride = innerStride;
                Source = source;
            }
        }

        /// <summary>
        ///     One array a view reads: the array, the view's own ARC pin on its buffer, whether the view owns the array
        ///     (a Half→Single / Decimal→Double conversion or a snapshot nobody else can dispose), and the address and
        ///     element size the getters compute with. A view of one array has one source; a view of several arrays (the
        ///     dictionary form, one column per array) has one per array.
        /// </summary>
        internal readonly unsafe struct Source
        {
            /// <summary>The array actually read (the caller's, or a converted / snapshot temporary).</summary>
            public readonly NDArray Fed;

            /// <summary>The view's own ARC pin on <see cref="Fed"/>'s buffer.</summary>
            public readonly IArraySlice Slice;

            /// <summary>Dispose <see cref="Fed"/> with the view (a converted or snapshot temporary).</summary>
            public readonly bool OwnsFed;

            /// <summary><see cref="Slice"/>'s address: storage element 0 (the offsets are baked into the plans).</summary>
            public readonly IntPtr BaseAddr;

            /// <summary>The element size of <see cref="Fed"/>'s dtype, in bytes.</summary>
            public readonly long ItemSize;

            /// <summary>Describes a pinned array the view reads.</summary>
            /// <param name="fed">The array read.</param>
            /// <param name="slice">The pin already taken on its buffer (the view releases it).</param>
            /// <param name="ownsFed"><c>true</c> when the view must dispose <paramref name="fed"/>.</param>
            public Source(NDArray fed, IArraySlice slice, bool ownsFed)
            {
                Fed = fed;
                Slice = slice;
                OwnsFed = ownsFed;
                BaseAddr = (IntPtr)slice.Address;
                ItemSize = fed.dtypesize;
            }
        }

        private readonly Source[] _sources;
        private readonly ColumnPlan[] _plans;
        private readonly DataViewSchema _schema;
        private readonly long _rowCount;
        private int _disposed;

        /// <summary>A view of ONE array (the single-array verbs): its pin and its ownership become the view's.</summary>
        /// <param name="fed">The array read.</param>
        /// <param name="slice">The pin already taken on its buffer.</param>
        /// <param name="ownsFed"><c>true</c> when the view must dispose <paramref name="fed"/>.</param>
        /// <param name="plans">The columns, every one reading source 0.</param>
        /// <param name="schema">The schema of the columns.</param>
        /// <param name="rowCount">The rows.</param>
        internal NDArrayDataView(NDArray fed, IArraySlice slice, bool ownsFed, ColumnPlan[] plans, DataViewSchema schema, long rowCount)
            : this(new[] { new Source(fed, slice, ownsFed) }, plans, schema, rowCount)
        {
        }

        /// <summary>
        ///     A view of SEVERAL arrays (the dictionary form): each plan names the source it reads; every source's pin and
        ///     ownership become the view's.
        /// </summary>
        /// <param name="sources">The pinned arrays.</param>
        /// <param name="plans">The columns.</param>
        /// <param name="schema">The schema of the columns.</param>
        /// <param name="rowCount">The rows, the same for every source.</param>
        internal NDArrayDataView(Source[] sources, ColumnPlan[] plans, DataViewSchema schema, long rowCount)
        {
            _sources = sources;
            _plans = plans;
            _schema = schema;
            _rowCount = rowCount;
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

        /// <summary>
        ///     <c>true</c>: the view reads any row at the same cost, so a cursor opened with a <see cref="Random"/> walks
        ///     the rows in a full random permutation (ML.NET's shuffling contract), the way ML.NET's own in-memory views do.
        /// </summary>
        /// <remarks>
        ///     <para>Why it matters: a trainer that shuffles (SDCA) uses the source's shuffle when the source has one and
        ///     falls back to ML.NET's POOL shuffle otherwise, which cannot mix rows stored in class order - SDCA trained on
        ///     20,000 class-sorted rows scored 0.50 accuracy from a non-shuffling view and 0.72 once the rows were permuted.
        ///     NumSharp code builds training sets in class order naturally (<c>np.vstack</c> of class blocks).</para>
        ///     <para><c>false</c> only for a view of more rows than a managed array can index (the permutation is an
        ///     <c>int[]</c> of the row count, 4 bytes a row, made per shuffled cursor; an unshuffled cursor makes none).</para>
        /// </remarks>
        public bool CanShuffle => _rowCount <= Array.MaxLength;

        /// <inheritdoc />
        public long? GetRowCount() => _rowCount;

        /// <summary>Opens a cursor over the rows - in order, or in a random permutation drawn from <paramref name="rand"/>.</summary>
        /// <param name="columnsNeeded">The columns the caller will read; the others stay inactive.</param>
        /// <param name="rand">
        ///     <c>null</c> for the rows in order; otherwise the source of a full permutation of the rows (a fixed seed gives the
        ///     same order every time). Ignored when <see cref="CanShuffle"/> is <c>false</c>, as ML.NET's contract allows.
        /// </param>
        /// <returns>A cursor holding its own pin on the buffer; dispose it.</returns>
        /// <exception cref="ObjectDisposedException">The view has been disposed, or its buffer released.</exception>
        public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
        {
            ThrowIfDisposed();
            return new NDArrayCursor(this, columnsNeeded, CanShuffle ? rand : null);
        }

        /// <inheritdoc />
        /// <remarks>NumSharp reads are inexpensive and single-threaded here; one cursor is returned (ML.NET may split work itself), shuffled when <paramref name="rand"/> is given.</remarks>
        public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            => new[] { GetRowCursor(columnsNeeded, rand) };

        internal ColumnPlan Plan(int columnIndex)
        {
            if ((uint)columnIndex >= (uint)_plans.Length)
                throw new ArgumentOutOfRangeException(nameof(columnIndex), $"column index {columnIndex} is out of range for a schema of {_plans.Length} columns.");
            return _plans[columnIndex];
        }

        /// <summary>
        ///     The pins of a new cursor: one ARC reference on every source's buffer, in source order, so a cursor survives
        ///     the view being disposed. All or nothing: a source whose buffer is gone releases the pins already taken.
        /// </summary>
        /// <returns>The cursor's pins, indexed like the sources.</returns>
        /// <exception cref="ObjectDisposedException">A source's buffer has already been released.</exception>
        internal IArraySlice[] TakeCursorPins()
        {
            var pins = new IArraySlice[_sources.Length];
            try
            {
                for (int s = 0; s < pins.Length; s++)
                    pins[s] = NDArrayMLNetInterop.Pin(_sources[s].Fed);
                return pins;
            }
            catch
            {
                foreach (IArraySlice taken in pins)
                    taken?.Release();
                throw;
            }
        }

        /// <summary>The address storage element 0 of source <paramref name="source"/> lies at.</summary>
        /// <param name="source">A source index.</param>
        /// <returns>The address the getters offset from.</returns>
        internal IntPtr BaseAddr(int source) => _sources[source].BaseAddr;

        /// <summary>The element size of source <paramref name="source"/>'s dtype, in bytes.</summary>
        /// <param name="source">A source index.</param>
        /// <returns>The size.</returns>
        internal long ItemSize(int source) => _sources[source].ItemSize;

        internal long RowCount => _rowCount;

        /// <summary>Release every buffer pin (and dispose the owned temporaries, if any). Idempotent.</summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            GC.SuppressFinalize(this);
            foreach (Source source in _sources)
                source.Slice.Release();
            NDArrayMLNetInterop.ExportClosed();
            foreach (Source source in _sources)
            {
                if (source.OwnsFed)
                    source.Fed.Dispose();   // the Half→Single / Decimal→Double / snapshot temporary nobody else can dispose
                else
                    GC.KeepAlive(source.Fed);
            }
        }

        /// <summary>
        ///     Safety net for a view that was never disposed: gives the view's reference on the buffer back — abandoning
        ///     it, never freeing eagerly — and does nothing else.
        /// </summary>
        /// <remarks>
        ///     <para><see cref="IArraySlice.Abandon"/> rather than <see cref="IArraySlice.Release"/>: this view being
        ///     unreachable proves nothing about OTHER reachable aliases of the buffer that hold no counted reference (a bare
        ///     <c>IArraySlice</c> taken from the source array), and releasing the last reference here freed the buffer under
        ///     such an alias. The memory block's own finalizer frees the buffer once nothing can reach it — NumSharp's own
        ///     <c>~NDArray</c> rule.</para>
        ///     <para>The owned temporary is not disposed here: it is reachable only through this view, so it is unreachable
        ///     too and its own finalizer gives its reference back. Any finalization order is safe: this only decrements a
        ///     count and a counter.</para>
        /// </remarks>
        ~NDArrayDataView()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            foreach (Source source in _sources)
                source.Slice.Abandon();
            NDArrayMLNetInterop.ExportClosed();
        }

        /// <summary>Refuses every use of a disposed view (its schema, a new cursor), naming the usual way to get there.</summary>
        /// <remarks>
        ///     The usual way is ML.NET's laziness, not a second <c>Dispose</c>: <c>model.Transform(view)</c> returns a view
        ///     that reads this one only when IT is read, so a <c>using</c> scope around this view that ends before the
        ///     output's <c>ToNDArray</c> / <c>Evaluate</c> lands here, far from the scope. The message says so, with the
        ///     remedy, because nothing at the throw site points back at the scope.
        /// </remarks>
        /// <exception cref="ObjectDisposedException">The view has been disposed.</exception>
        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref _disposed) != 0)
                throw new ObjectDisposedException(nameof(NDArrayDataView),
                    "the NDArray-backed IDataView has been disposed; its NumSharp buffer is no longer pinned. ML.NET reads lazily: " +
                    "a Transform's output (and every view built on this one) reads this view only when it is read itself, so keep " +
                    "this view undisposed until the last read of anything built on it (ToNDArray, Evaluate, Fit, a cursor).");
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
            private readonly IArraySlice[] _slices;  // this cursor's own ARC pins, one per source (they survive the view being disposed)
            private readonly long _rowCount;
            private readonly bool[] _active;
            private readonly int[] _order;           // the row of each position for a shuffled cursor; null = position IS the row
            private long _position = -1;             // -1 off a row: before the first, after the last, after Dispose (ML.NET's convention)
            private bool _done;                      // MoveNext returned false or the cursor was disposed: it never moves again
            private int _disposed;

            /// <summary>Opens a cursor over <paramref name="view"/>, taking the cursor's own pin on every buffer it reads.</summary>
            /// <param name="view">The (undisposed) view to read.</param>
            /// <param name="columnsNeeded">The columns to activate; columns of another schema (index out of range) are ignored.</param>
            /// <param name="rand">
            ///     <c>null</c> for the rows in order; otherwise the source of the permutation the cursor walks (the caller
            ///     passes <c>null</c> when the view cannot shuffle).
            /// </param>
            /// <exception cref="ObjectDisposedException">A buffer of the view has already been released.</exception>
            internal NDArrayCursor(NDArrayDataView view, IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand)
            {
                _view = view;
                _schema = view.Schema;
                _rowCount = view.RowCount;
                _active = new bool[_schema.Count];
                if (columnsNeeded is not null)
                {
                    foreach (DataViewSchema.Column c in columnsNeeded)
                        if ((uint)c.Index < (uint)_active.Length)
                            _active[c.Index] = true;
                }
                // Drawn BEFORE the pins are taken: the permutation is the one allocation that can fail here (a view of a
                // billion rows asks for 4 GB), and a failure after the pins would leak them - the cursor's Dispose never
                // runs for a constructor that threw.
                if (rand is not null)
                    _order = Permutation((int)_rowCount, rand);
                _slices = view.TakeCursorPins();
                NDArrayMLNetInterop.ExportOpened();
            }

            /// <summary>
            ///     A uniformly random permutation of <c>[0, count)</c> by Fisher-Yates, every draw from <paramref name="rand"/>,
            ///     so a seeded <see cref="Random"/> gives the same order every time (ML.NET seeds it from the context's seed).
            /// </summary>
            /// <param name="count">The number of rows; at most <see cref="Array.MaxLength"/> (the view's <c>CanShuffle</c>).</param>
            /// <param name="rand">The source of randomness.</param>
            /// <returns>The row to visit at each position.</returns>
            private static int[] Permutation(int count, Random rand)
            {
                var order = new int[count];
                for (int i = 0; i < count; i++)
                    order[i] = i;
                for (int i = count - 1; i > 0; i--)
                {
                    int j = rand.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
                return order;
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
            /// <remarks>
            ///     Row ids are the SOURCE row numbers - under a shuffle too, so an id names the same row in every cursor (as
            ///     in ML.NET's own shuffling views); outside a row the getter throws like the value getters.
            /// </remarks>
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
            ///     first row, after the last, and after <c>Dispose</c> (which parks the position at -1 too). A shuffled
            ///     cursor maps the position through its permutation, whose entries are rows by construction.
            /// </summary>
            /// <returns>The current row index, in <c>[0, rowCount)</c>.</returns>
            /// <exception cref="InvalidOperationException">The cursor is not on a row; ML.NET's own message.</exception>
            private long CurrentRow()
            {
                long position = _position;
                if ((ulong)position >= (ulong)_rowCount)
                    throw new InvalidOperationException(FetchValueStateError);
                return _order is null ? position : _order[position];
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
                long elemBase = p.ElemBase, rowStride = p.RowStride, itemsize = _view.ItemSize(p.Source);
                IntPtr baseAddr = _view.BaseAddr(p.Source);   // the column's own array (one of several in a dictionary view)
                ValueGetter<TValue> getter = (ref TValue value) =>
                {
                    long phys = elemBase + CurrentRow() * rowStride;
                    unsafe { value = Unsafe.Read<TValue>((void*)(baseAddr + (nint)(phys * itemsize))); }
                    // The raw address does not keep the cursor alive, and nothing else may: the caller can hold only this
                    // delegate, which is not reported live while it runs. See the vector getter.
                    GC.KeepAlive(this);
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
                long elemBase = p.ElemBase, rowStride = p.RowStride, innerStride = p.InnerStride, itemsize = _view.ItemSize(p.Source);
                int width = p.VectorWidth;
                IntPtr baseAddr = _view.BaseAddr(p.Source);   // the column's own array (one of several in a dictionary view)
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
                    // The cursor (and through it the view and the pins) must stay reachable until the copy above is done.
                    // The raw address does not root it, and a caller holding only this delegate does not either: once the
                    // getter's last field read is behind it, the JIT reports nothing live, and a collection that lands while
                    // the row is copied finalizes the cursor, whose release can free the buffer mid-copy. Probed: under GC
                    // pressure, 117 of 279 cursors reachable only through their getter were finalized during that getter's
                    // own call.
                    GC.KeepAlive(this);
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
            ///     Gives the cursor's pin back, once (an explicit dispose and the finalizer race through the same interlocked
            ///     guard). The cursor is parked off any row first (position -1, done), so a getter still held by the caller
            ///     throws instead of reading a buffer this pin no longer keeps alive.
            /// </summary>
            /// <remarks>
            ///     From <c>Dispose()</c> the pin is RELEASED (freeing the buffer if it was the last reference — the caller
            ///     asserts nothing reads it any more); from the finalizer it is ABANDONED: the cursor being unreachable proves
            ///     nothing about other reachable aliases of the buffer that hold no counted reference, so the finalizer must not
            ///     claim the eager free (the memory block's own finalizer frees the buffer once nothing can reach it).
            /// </remarks>
            /// <param name="disposing"><c>true</c> from <c>Dispose()</c>, <c>false</c> from the finalizer.</param>
            protected override void Dispose(bool disposing)
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                {
                    _done = true;
                    _position = -1;
                    foreach (IArraySlice slice in _slices)
                    {
                        if (disposing)
                            slice.Release();
                        else
                            slice.Abandon();
                    }
                    NDArrayMLNetInterop.ExportClosed();
                    GC.KeepAlive(_view);
                }
                base.Dispose(disposing);
            }
        }
    }
}
