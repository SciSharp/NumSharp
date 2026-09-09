using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

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

        private static unsafe NDArray Materialize<T>(IDataView view, DataViewSchema.Column column, bool isVector) where T : unmanaged
        {
            var needed = new[] { column };
            using DataViewRowCursor cursor = view.GetRowCursor(needed);

            if (!isVector)
            {
                ValueGetter<T> getter = cursor.GetGetter<T>(column);
                var values = new List<T>();
                T value = default;
                while (cursor.MoveNext())
                {
                    getter(ref value);
                    values.Add(value);
                }
                return BuildNDArray(values.ToArray(), Shape.Vector(values.Count));
            }
            else
            {
                ValueGetter<VBuffer<T>> getter = cursor.GetGetter<VBuffer<T>>(column);
                var rows = new List<T[]>();
                int width = -1;
                var buffer = default(VBuffer<T>);
                while (cursor.MoveNext())
                {
                    getter(ref buffer);
                    if (width < 0)
                        width = buffer.Length;
                    else if (buffer.Length != width)
                        throw new ArgumentException($"column '{column.Name}' is a variable-length vector (row {rows.Count} has length {buffer.Length}, an earlier row {width}); NumSharp arrays are rectangular. Pad the column to a fixed size first (e.g. an ML.NET vector-size transform).");
                    var dense = new T[width];
                    buffer.CopyTo(dense);   // densifies a sparse buffer into the full-length span
                    rows.Add(dense);
                }

                int rowCount = rows.Count;
                if (width < 0)
                    width = column.Type is VectorDataViewType v && v.Size > 0 ? v.Size : 0;   // no rows: use the declared size
                var flat = new T[(long)rowCount * width];
                for (int r = 0; r < rowCount; r++)
                    Array.Copy(rows[r], 0, flat, (long)r * width, width);
                return BuildNDArray(flat, new Shape(rowCount, width));
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
                // Sparse: zero the whole vector, then scatter the stored values at their indices.
                new Span<byte>(dst, length * sizeof(T)).Clear();
                ReadOnlySpan<T> values = buffer.GetValues();
                ReadOnlySpan<int> indices = buffer.GetIndices();
                var target = new Span<T>(dst, length);
                for (int i = 0; i < values.Length; i++)
                    target[indices[i]] = values[i];
            }
            return nd;
        }

        // ---- helpers ------------------------------------------------------------------------------

        private static unsafe NDArray BuildNDArray<T>(T[] flat, Shape shape) where T : unmanaged
        {
            NPTypeCode tc = TypeCodeOf<T>();
            var nd = new NDArray(tc, shape, fillZeros: false);
            if (flat.Length > 0)
            {
                byte* dst = (byte*)nd.Storage.InternalArray.Address;
                fixed (T* src = flat)
                {
                    long nbytes = (long)flat.Length * sizeof(T);
                    Buffer.MemoryCopy(src, dst, nbytes, nbytes);
                }
            }
            return nd;
        }

        private static string ColumnNames(DataViewSchema schema)
        {
            var names = new List<string>(schema.Count);
            foreach (DataViewSchema.Column c in schema)
                names.Add(c.Name);
            return string.Join(", ", names);
        }
    }
}
