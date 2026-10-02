using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.MLNet
{
    public static partial class NDArrayMLNetInterop
    {
        // ===========================  ML.NET  ->  NumSharp  ===========================

        /// <summary>
        ///     Materialize one column of an <see cref="IDataView"/> — across every row — into a fresh owning
        ///     <see cref="NDArray"/>. A SCALAR column of <c>R</c> rows becomes a 1-D array <c>(R,)</c>; a fixed-size
        ///     VECTOR column of width <c>C</c> becomes a 2-D array <c>(R, C)</c> (sparse rows are densified). This is
        ///     the output half of the bridge: after <c>transformer.Transform(...)</c>, read a score column straight
        ///     into NumSharp — <c>output.ToNDArray("Score")</c> — instead of a per-row POCO loop.
        ///
        ///     <para>The dtype follows <see cref="FromDataViewType"/> (a <see cref="NumberDataViewType.UInt16"/> column
        ///     reads back as <see cref="NPTypeCode.UInt16"/>, never <see cref="NPTypeCode.Char"/>). A text column, or a
        ///     variable-length vector column whose rows disagree on length, is refused with a message naming the cause.</para>
        /// </summary>
        /// <exception cref="ArgumentException">No column of that name, or a variable-length vector column.</exception>
        /// <exception cref="NotSupportedException">The column's element type has no NumSharp dtype (text, key of an unmapped raw type, ...).</exception>
        public static NDArray ToNDArray(this IDataView view, string columnName)
        {
            if (view is null)
                throw new ArgumentNullException(nameof(view));
            if (columnName is null)
                throw new ArgumentNullException(nameof(columnName));
            DataViewSchema.Column? col = view.Schema.GetColumnOrNull(columnName);
            if (col is null)
                throw new ArgumentException($"the data view has no column named '{columnName}'; its columns are [{ColumnNames(view.Schema)}].", nameof(columnName));
            return ToNDArray(view, col.Value);
        }

        /// <summary>Materialize a specific schema <paramref name="column"/> of <paramref name="view"/> into an <see cref="NDArray"/> (see <see cref="ToNDArray(IDataView, string)"/>).</summary>
        public static NDArray ToNDArray(this IDataView view, DataViewSchema.Column column)
        {
            if (view is null)
                throw new ArgumentNullException(nameof(view));
            DataViewType type = column.Type;
            bool isVector = type is VectorDataViewType;
            NPTypeCode itemCode = FromDataViewType(type);   // throws NotSupported for text / unmapped raw types

            switch (itemCode)
            {
                case NPTypeCode.Boolean: return Materialize<bool>(view, column, isVector);
                case NPTypeCode.Byte: return Materialize<byte>(view, column, isVector);
                case NPTypeCode.SByte: return Materialize<sbyte>(view, column, isVector);
                case NPTypeCode.Int16: return Materialize<short>(view, column, isVector);
                case NPTypeCode.UInt16: return Materialize<ushort>(view, column, isVector);
                case NPTypeCode.Int32: return Materialize<int>(view, column, isVector);
                case NPTypeCode.UInt32: return Materialize<uint>(view, column, isVector);
                case NPTypeCode.Int64: return Materialize<long>(view, column, isVector);
                case NPTypeCode.UInt64: return Materialize<ulong>(view, column, isVector);
                case NPTypeCode.Single: return Materialize<float>(view, column, isVector);
                case NPTypeCode.Double: return Materialize<double>(view, column, isVector);
                default: throw new NotSupportedException(UnsupportedImportMessage(type));
            }
        }

        /// <summary>
        ///     Reads one column of every row into a fresh owning <see cref="NDArray"/>, writing each row straight into
        ///     NumSharp's unmanaged memory (see <see cref="ColumnStage{T}"/>).
        /// </summary>
        /// <remarks>
        ///     The rows used to be staged in managed arrays — a <c>List&lt;T&gt;</c> for a scalar column, one <c>T[]</c> per row
        ///     plus a final <c>T[rowCount * width]</c> for a vector column — which held the data twice and capped a column
        ///     at <see cref="Array.MaxLength"/> elements in total: a larger column failed with OutOfMemoryException ("Array
        ///     dimensions exceeded supported range") although an NDArray holds it. Now a view that reports its row count
        ///     (every NumSharp view, ML.NET's in-memory and cached views, any transform that keeps rows) is read into ONE
        ///     array of exactly that many rows, which IS the result; a view that does not (a row filter) is read into
        ///     growing chunks copied together once at the end.
        /// </remarks>
        /// <typeparam name="T">The column's element CLR type.</typeparam>
        /// <param name="view">The data view to read.</param>
        /// <param name="column">The column to read.</param>
        /// <param name="isVector">Whether the column is a vector column (each row a <see cref="VBuffer{T}"/>).</param>
        /// <returns>A 1-D <c>(rows,)</c> array for a scalar column, a 2-D <c>(rows, width)</c> array for a vector column.</returns>
        /// <exception cref="ArgumentException">A variable-length vector column whose rows disagree on length.</exception>
        private static unsafe NDArray Materialize<T>(IDataView view, DataViewSchema.Column column, bool isVector) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            long? knownRows = view.GetRowCount();
            var needed = new[] { column };
            using DataViewRowCursor cursor = view.GetRowCursor(needed);
            ColumnStage<T> stage = null;
            try
            {
                if (!isVector)
                {
                    ValueGetter<T> getter = cursor.GetGetter<T>(column);
                    stage = new ColumnStage<T>(tc, width: -1, knownRows);
                    T value = default;
                    while (cursor.MoveNext())
                    {
                        getter(ref value);
                        *stage.NextRow() = value;
                    }
                    return stage.Finish();
                }

                ValueGetter<VBuffer<T>> vectorGetter = cursor.GetGetter<VBuffer<T>>(column);
                // A fixed-size column declares its width; a variable-size one (Size == 0) is learned from its first row.
                int declaredWidth = column.Type is VectorDataViewType declared ? declared.Size : 0;
                var buffer = default(VBuffer<T>);
                while (cursor.MoveNext())
                {
                    vectorGetter(ref buffer);
                    stage ??= new ColumnStage<T>(tc, buffer.Length, knownRows);
                    if (buffer.Length != stage.Width)
                        throw new ArgumentException($"column '{column.Name}' is a variable-length vector (row {stage.Rows} has length {buffer.Length}, an earlier row {stage.Width}); NumSharp arrays are rectangular. Pad the column to a fixed size first (e.g. an ML.NET vector-size transform).");
                    // VBuffer.CopyTo(Span<T>) densifies a sparse row into the full-length span.
                    buffer.CopyTo(new Span<T>(stage.NextRow(), stage.Width));
                }
                stage ??= new ColumnStage<T>(tc, declaredWidth, knownRows: 0);   // no rows: (0, declared width)
                return stage.Finish();
            }
            finally
            {
                // Finish hands the result over and empties the stage; on any failure this frees what was staged.
                stage?.Dispose();
            }
        }

        /// <summary>
        ///     The rows of a column being read, in NumSharp's unmanaged memory: chunks of whole rows (a scalar column's row is
        ///     one element, a vector column's row is <see cref="Width"/> elements), the first sized to the view's reported
        ///     row count so that a view that reports it truthfully is read into a single chunk which becomes the result as
        ///     is. More rows than reported, or no reported count, add chunks of doubling capacity; <see cref="Finish"/> then
        ///     copies them into one array of the exact size. Owns every chunk until <see cref="Finish"/> hands the result
        ///     over; <see cref="Dispose"/> frees whatever it still owns.
        /// </summary>
        /// <typeparam name="T">The element CLR type.</typeparam>
        private sealed unsafe class ColumnStage<T> : IDisposable where T : unmanaged
        {
            /// <summary>The capacity, in rows, of the first chunk when the view does not report its row count.</summary>
            private const long InitialRows = 4096;

            /// <summary>The largest chunk, in elements (64 M — 256 MB of float), so growth never doubles past what a column needs by much.</summary>
            private const long MaxChunkElements = 64L << 20;

            private readonly NPTypeCode _code;
            private readonly bool _scalar;
            private readonly List<(NDArray Chunk, long Rows)> _full = new List<(NDArray, long)>();
            private NDArray _current;
            private long _currentCapacity;
            private long _currentRows;
            private long _nextCapacity;

            /// <summary>Creates an empty stage.</summary>
            /// <param name="code">The dtype of the result.</param>
            /// <param name="width">The elements per row of a vector column, or -1 for a scalar column.</param>
            /// <param name="knownRows">The row count the view reports, or null when it reports none.</param>
            internal ColumnStage(NPTypeCode code, int width, long? knownRows)
            {
                _code = code;
                _scalar = width < 0;
                Width = _scalar ? 1 : width;
                _nextCapacity = knownRows ?? InitialRows;
            }

            /// <summary>The elements per row (1 for a scalar column).</summary>
            internal int Width { get; }

            /// <summary>The rows written so far.</summary>
            internal long Rows { get; private set; }

            /// <summary>Reserves the next row and returns where its <see cref="Width"/> elements go, growing a chunk when the current one is full.</summary>
            /// <returns>The address of the row's first element.</returns>
            internal T* NextRow()
            {
                if (_current is null || _currentRows == _currentCapacity)
                    Grow();
                T* row = (T*)_current.Storage.InternalArray.Address + _currentRows * Width;   // a fresh array: offset 0
                _currentRows++;
                Rows++;
                return row;
            }

            /// <summary>Retires the full current chunk and allocates the next, at the planned capacity (at least one row).</summary>
            private void Grow()
            {
                if (_current is not null)
                    _full.Add((_current, _currentRows));
                long capacity = Math.Max(1, _nextCapacity);
                _current = new NDArray(_code, ChunkShape(capacity), fillZeros: false);
                _currentCapacity = capacity;
                _currentRows = 0;
                // Rows beyond the reported count (or a view with none) grow geometrically, capped per chunk.
                long maxRows = Math.Max(1, MaxChunkElements / Math.Max(1, Width));
                _nextCapacity = Math.Min(Math.Max(capacity * 2, InitialRows), maxRows);
            }

            /// <summary>The shape of a chunk (or of the result) holding <paramref name="rows"/> rows.</summary>
            /// <param name="rows">The row count.</param>
            /// <returns><c>(rows,)</c> for a scalar column, <c>(rows, width)</c> for a vector column.</returns>
            private Shape ChunkShape(long rows) => _scalar ? Shape.Vector(rows) : new Shape(rows, Width);

            /// <summary>
            ///     The column as one array of exactly <see cref="Rows"/> rows, handed over to the caller (the stage keeps no
            ///     reference to it): the single chunk itself when every row landed in one exactly-full chunk, otherwise a new
            ///     array the filled rows of every chunk are copied into (the chunks stay owned by the stage and are freed by
            ///     its <see cref="Dispose"/>, which the caller runs in a <c>finally</c>).
            /// </summary>
            /// <returns>The result.</returns>
            internal NDArray Finish()
            {
                if (_current is not null && _full.Count == 0 && _currentRows == _currentCapacity)
                {
                    NDArray only = _current;
                    _current = null;
                    return only;
                }

                var result = new NDArray(_code, ChunkShape(Rows), fillZeros: false);
                try
                {
                    long written = 0;
                    long itemsize = sizeof(T);
                    byte* dst = (byte*)result.Storage.InternalArray.Address;
                    if (_current is not null)
                        _full.Add((_current, _currentRows));
                    _current = null;
                    foreach ((NDArray chunk, long rows) in _full)
                    {
                        long bytes = rows * Width * itemsize;
                        Buffer.MemoryCopy(chunk.Storage.InternalArray.Address, dst + written, bytes, bytes);
                        written += bytes;
                    }
                    return result;
                }
                catch
                {
                    result.Dispose();
                    throw;
                }
            }

            /// <summary>Frees every chunk the stage still owns (all of them after a failure; none after <see cref="Finish"/> returned the single chunk).</summary>
            public void Dispose()
            {
                foreach ((NDArray chunk, _) in _full)
                    chunk.Dispose();
                _full.Clear();
                _current?.Dispose();
                _current = null;
            }
        }

        // ===========================  ML.NET VBuffer  ->  NumSharp  ===========================

        /// <summary>
        ///     Copy an ML.NET <see cref="VBuffer{T}"/> into a fresh owning 1-D <see cref="NDArray"/> of length
        ///     <see cref="VBuffer{T}.Length"/>. A sparse buffer is densified (missing entries become zero, as ML.NET
        ///     defines a sparse vector). <see cref="ushort"/> reads back as <see cref="NPTypeCode.UInt16"/>.
        /// </summary>
        public static unsafe NDArray ToNDArray<T>(this VBuffer<T> buffer) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            int length = buffer.Length;
            var nd = new NDArray(tc, Shape.Vector(length), fillZeros: false);
            if (length == 0)
                return nd;

            byte* dst = (byte*)nd.Storage.InternalArray.Address;
            if (buffer.IsDense)
            {
                ReadOnlySpan<T> src = buffer.GetValues();
                src.CopyTo(new Span<T>(dst, length));
            }
            else
            {
                // Sparse: zero the whole vector, then scatter the stored values at their indices. The clear is sized in
                // ELEMENTS: a byte count (length * sizeof(T)) overflows int once the dense vector reaches 2 GB — 2^28
                // doubles, an ordinary size for a hashed sparse feature vector — and threw ArgumentOutOfRangeException.
                var target = new Span<T>(dst, length);
                target.Clear();
                ReadOnlySpan<T> values = buffer.GetValues();
                ReadOnlySpan<int> indices = buffer.GetIndices();
                for (int i = 0; i < values.Length; i++)
                    target[indices[i]] = values[i];
            }
            return nd;
        }

        /// <summary>
        ///     View a DENSE ML.NET <see cref="VBuffer{T}"/> as a NumSharp array over the buffer's OWN backing store —
        ///     <b>zero-copy</b>, mutations visible both ways. The 1-D array of length <see cref="VBuffer{T}.Length"/>
        ///     wraps the private <c>T[]</c> that backs the buffer (reached through a compiled-expression accessor —
        ///     <see cref="VBufferAccessor{T}"/>), pinned for the view's lifetime; the pin is released when the LAST
        ///     NumSharp view over the memory (derived slices included) is disposed or collected.
        ///
        ///     <para><b>Aliasing hazard — read this.</b> The view shares the VBuffer's array, so it is valid only
        ///     while that array is not refilled or reused. Getting a VBuffer from a cursor (<c>getter(ref buf)</c>)
        ///     REUSES one buffer across rows, so a view taken there is invalidated by the next <c>MoveNext</c> —
        ///     take <see cref="ToNDArray{T}(VBuffer{T})"/> (a copy) for cursor buffers. Zero-copy is safe for a
        ///     STANDALONE buffer you own and will not refill. The view does not own its data (like <c>np.frombuffer</c>):
        ///     a size-changing <c>ndarray.resize</c> refuses.</para>
        ///
        ///     <para>A SPARSE buffer, an empty buffer, or an ML.NET version whose <see cref="VBuffer{T}"/> is not the
        ///     expected managed-array form falls back to <see cref="ToNDArray{T}(VBuffer{T})"/> (a copy) — never a
        ///     wrong reinterpretation. <see cref="ushort"/> reads back as <see cref="NPTypeCode.UInt16"/>.</para>
        /// </summary>
        public static unsafe NDArray AsNDArray<T>(this VBuffer<T> buffer) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            int length = buffer.Length;
            if (length == 0)
                return new NDArray(tc, Shape.Vector(0), fillZeros: false);   // nothing to share

            Func<VBuffer<T>, T[]> getValues = VBufferAccessor<T>.GetValuesArray;
            T[] raw = getValues?.Invoke(buffer);
            // Zero-copy only for a dense buffer whose backing array is at least Length long; else copy (densify).
            if (getValues is null || !buffer.IsDense || raw is null || raw.Length < length)
                return buffer.ToNDArray();

            GCHandle handle = GCHandle.Alloc(raw, GCHandleType.Pinned);
            var lease = new ImportLease(handle.Free, (long)length * sizeof(T));
            try
            {
                void* p = (void*)handle.AddrOfPinnedObject();   // element 0 of the pinned T[]
                IArraySlice slice = WrapExternal(tc, p, length, lease.Release);
                // Alias() so the storage reports VIEW semantics (owndata == False): a size-changing resize refuses
                // instead of silently detaching from the VBuffer's array.
                return new NDArray(new UnmanagedStorage(slice, Shape.Vector(length)).Alias());
            }
            catch
            {
                lease.Release();
                throw;
            }
        }

        // ---- helpers ------------------------------------------------------------------------------

        private static string ColumnNames(DataViewSchema schema)
        {
            var names = new List<string>(schema.Count);
            foreach (DataViewSchema.Column c in schema)
                names.Add(c.Name);
            return string.Join(", ", names);
        }
    }
}
