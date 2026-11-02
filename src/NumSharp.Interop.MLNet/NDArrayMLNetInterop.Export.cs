using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.MLNet
{
    public static partial class NDArrayMLNetInterop
    {
        // ===========================  NumSharp  ->  ML.NET  ===========================

        /// <summary>
        ///     Expose a NumSharp array to an ML.NET pipeline as an <see cref="IDataView"/> with ONE vector column —
        ///     the "feature matrix" shape a trained pipeline consumes. A 2-D <c>(R, C)</c> array becomes <c>R</c> rows,
        ///     each a length-<c>C</c> <see cref="VBuffer{T}"/>; a 1-D <c>(C,)</c> array becomes 1 row of a length-<c>C</c>
        ///     vector (a single sample). The array is read <b>lazily through its own strides</b> (any layout — C /
        ///     Fortran / sliced / transposed / negative-stride / broadcast — no densifying copy) and ARC-pinned for the
        ///     view's lifetime, so it is even safe to dispose the source array while the view still reads it.
        ///
        ///     <para><b>Feeding a model</b> is then a two-verb bridge — no per-transformer runner:
        ///     <c>transformer.Transform(nd.AsDataView("Features")).ToNDArray("Score")</c>.</para>
        ///
        ///     <para><b>Dtypes:</b> the 8 integers, float, double and bool cross as themselves;
        ///     <see cref="NPTypeCode.Char"/> as <see cref="NumberDataViewType.UInt16"/>. ML.NET has no half / decimal
        ///     column type: <see cref="NPTypeCode.Half"/> is converted to Single (lossless) and
        ///     <see cref="NPTypeCode.Decimal"/> to Double in a temporary the view owns.
        ///     <see cref="NPTypeCode.Complex"/> throws.</para>
        ///
        ///     <para>The returned view is <see cref="IDisposable"/> (an <see cref="NDArrayDataView"/>): dispose it to
        ///     release the pin (a <c>using</c>, or wire it into your <c>MLContext</c>'s lifetime). Not disposing leaks
        ///     the pin until finalization.</para>
        /// </summary>
        /// <remarks>
        ///     The rank, the dtype and the vector width are checked before anything is converted or read; the buffer's
        ///     liveness is checked next, before the Half / Decimal conversion reads the source.
        /// </remarks>
        /// <param name="source">The array to expose.</param>
        /// <param name="columnName">The name of the single vector column (ML.NET's convention is <c>"Features"</c>).</param>
        /// <returns>A disposable data view reading the array lazily; it holds the buffer pin (and any conversion) until disposed.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or <paramref name="columnName"/> is null/empty.</exception>
        /// <exception cref="NotSupportedException">
        ///     The dtype has no ML.NET column type (Complex, or text: a U, S or StringDType array), or the feature axis is
        ///     wider than ML.NET's buffers hold (2,146,435,071 elements, a little below <see cref="Array.MaxLength"/>): every
        ///     row reaches ML.NET as a <see cref="VBuffer{T}"/>.
        /// </exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static NDArrayDataView AsDataView(this NDArray source, string columnName = "Features")
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (string.IsNullOrEmpty(columnName))
                throw new ArgumentException("a column name is required.", nameof(columnName));

            // Refuse what the shape and the dtype alone decide before the Half / Decimal conversion reads the source:
            // an array that cannot cross costs no conversion, and a released one is not read just to be refused.
            ValidateVectorColumn(source.Shape, FeedTypeCode(source.typecode));
            NDArray fed = ResolveFeedArray(source, out NDArray converted);
            try
            {
                NDArrayDataView view = BuildVectorView(fed, ownsFed: converted is not null, columnName);
                return view;
            }
            catch
            {
                converted?.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Expose a NumSharp array to ML.NET as an <see cref="IDataView"/> with ONE SCALAR COLUMN PER FEATURE —
        ///     the "tabular" shape for building training data or feeding pipelines that reference named scalar columns.
        ///     A 2-D <c>(R, C)</c> array becomes <c>R</c> rows and <c>C</c> scalar columns named by
        ///     <paramref name="columnNames"/> (which must have length <c>C</c>); a 1-D <c>(R,)</c> array with one name
        ///     becomes <c>R</c> rows and one scalar column. Same lazy, any-layout, pinned reading as
        ///     <see cref="AsDataView(NDArray, string)"/>.
        /// </summary>
        /// <param name="source">The array to expose.</param>
        /// <param name="columnNames">One non-empty name per column: <c>shape[1]</c> names for a 2-D array, one for a 1-D array.</param>
        /// <returns>A disposable data view reading the array lazily; it holds the buffer pin (and any conversion) until disposed.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or the name count does not match the column count, or a name is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex, or text: a U, S or StringDType array).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static NDArrayDataView AsDataView(this NDArray source, string[] columnNames)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (columnNames is null || columnNames.Length == 0)
                throw new ArgumentException("at least one column name is required.", nameof(columnNames));

            // Refuse what the shape, the dtype and the names alone decide before the conversion reads the source.
            ValidateScalarColumns(source.Shape, FeedTypeCode(source.typecode), columnNames);
            NDArray fed = ResolveFeedArray(source, out NDArray converted);
            try
            {
                return BuildScalarView(fed, ownsFed: converted is not null, columnNames);
            }
            catch
            {
                converted?.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <see cref="AsDataView(NDArray, string)"/> over an independent SNAPSHOT: the array is copied first, so the
        ///     returned view has no lifetime coupling to <paramref name="source"/> — mutate or dispose the source
        ///     freely afterwards. Use it when the source is transient; prefer <c>AsDataView</c> for the zero-copy read.
        /// </summary>
        /// <remarks>
        ///     The rank, the dtype and the vector width are checked before the snapshot is taken, so an array the view
        ///     would refuse is not copied first (a 2 GB copy just to be refused); the buffer's liveness is checked next,
        ///     before the snapshot reads the source.
        /// </remarks>
        /// <param name="source">The array to snapshot.</param>
        /// <param name="columnName">The name of the single vector column (ML.NET's convention is <c>"Features"</c>).</param>
        /// <returns>A disposable data view over a private copy of the array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or <paramref name="columnName"/> is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex, or text: a U, S or StringDType array), or the feature axis is wider than ML.NET's buffers hold (2,146,435,071 elements).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static NDArrayDataView ToDataView(this NDArray source, string columnName = "Features")
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (string.IsNullOrEmpty(columnName))
                throw new ArgumentException("a column name is required.", nameof(columnName));

            // The snapshot used to come first, so a view the shape or dtype ruled out cost a full copy of the source
            // before the refusal, and a released source was copied from freed memory before anything checked it.
            ValidateVectorColumn(source.Shape, FeedTypeCode(source.typecode));
            NDArray snapshot = Snapshot(source);
            try
            {
                return BuildVectorView(snapshot, ownsFed: true, columnName);
            }
            catch
            {
                snapshot.Dispose();
                throw;
            }
        }

        /// <summary><see cref="AsDataView(NDArray, string[])"/> over an independent snapshot (see <see cref="ToDataView(NDArray, string)"/>).</summary>
        /// <param name="source">The array to snapshot.</param>
        /// <param name="columnNames">One non-empty name per column: <c>shape[1]</c> names for a 2-D array, one for a 1-D array.</param>
        /// <returns>A disposable data view over a private copy of the array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or the name count does not match the column count, or a name is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex, or text: a U, S or StringDType array).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static NDArrayDataView ToDataView(this NDArray source, string[] columnNames)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (columnNames is null || columnNames.Length == 0)
                throw new ArgumentException("at least one column name is required.", nameof(columnNames));

            // Refuse what the shape, the dtype and the names alone decide before the snapshot copies the source.
            ValidateScalarColumns(source.Shape, FeedTypeCode(source.typecode), columnNames);
            NDArray snapshot = Snapshot(source);
            try
            {
                return BuildScalarView(snapshot, ownsFed: true, columnNames);
            }
            catch
            {
                snapshot.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Expose SEVERAL NumSharp arrays to ML.NET as ONE <see cref="IDataView"/>, one column per array - what
        ///     training needs: a <c>"Features"</c> vector and a <c>"Label"</c> of its own dtype in the same rows, as pandas'
        ///     <c>DataFrame({"Features": X, "Label": y})</c>. Each array is read lazily through its own strides (any
        ///     layout, no copy) and ARC-pinned for the view's lifetime, as <see cref="AsDataView(NDArray, string)"/> reads
        ///     one.
        /// </summary>
        /// <remarks>
        ///     <para><b>Axis 0 is always the rows.</b> A 1-D <c>(R,)</c> array is a SCALAR column of <c>R</c> rows (the
        ///     label shape) and a 2-D <c>(R, C)</c> array a VECTOR column of width <c>C</c> (the features shape); every
        ///     array must have the same <c>R</c>. (The single-array <see cref="AsDataView(NDArray, string)"/> reads a 1-D
        ///     array as ONE row: a sample, not a column.)</para>
        ///     <para><b>Dtypes</b> per array, as the single-array verbs map them: each column keeps its own (a bool label
        ///     next to float32 features needs no conversion step); Half and Decimal are converted to Single / Double into a
        ///     temporary the view owns; Complex and text arrays are refused.</para>
        ///     <para>Every refusal the shapes, dtypes and names decide comes before anything is converted or pinned. The
        ///     view holds one pin per array; disposing it releases them all, and each cursor takes its own.</para>
        /// </remarks>
        /// <param name="columns">Column name → array, in schema order; at least one entry, every name non-empty, every array 1-D or 2-D with the same number of rows.</param>
        /// <returns>A disposable data view reading the arrays lazily; it holds their pins (and any conversion) until disposed.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="columns"/> is null, or an array in it is null.</exception>
        /// <exception cref="ArgumentException">No column, an empty name, an array that is not 1-D or 2-D, or arrays with different numbers of rows.</exception>
        /// <exception cref="NotSupportedException">An array's dtype has no ML.NET column type (Complex, or text), or a vector column is wider than ML.NET's buffers hold.</exception>
        /// <exception cref="ObjectDisposedException">An array's buffer has already been released.</exception>
        public static NDArrayDataView AsDataView(this IReadOnlyDictionary<string, NDArray> columns) => BuildTableView(columns, snapshot: false);

        /// <summary>
        ///     <see cref="AsDataView(IReadOnlyDictionary{string, NDArray})"/> over independent SNAPSHOTS: each array is
        ///     copied first, so the view has no lifetime coupling to them - mutate or dispose them freely afterwards.
        /// </summary>
        /// <param name="columns">Column name → array, as for <see cref="AsDataView(IReadOnlyDictionary{string, NDArray})"/>.</param>
        /// <returns>A disposable data view over private copies of the arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="columns"/> is null, or an array in it is null.</exception>
        /// <exception cref="ArgumentException">No column, an empty name, an array that is not 1-D or 2-D, or arrays with different numbers of rows.</exception>
        /// <exception cref="NotSupportedException">An array's dtype has no ML.NET column type (Complex, or text), or a vector column is wider than ML.NET's buffers hold.</exception>
        /// <exception cref="ObjectDisposedException">An array's buffer has already been released.</exception>
        public static NDArrayDataView ToDataView(this IReadOnlyDictionary<string, NDArray> columns) => BuildTableView(columns, snapshot: true);

        /// <summary>
        ///     The dictionary form of both verbs: validate every array from its shape, dtype and name, then feed each one
        ///     (as it is, converted, or snapshotted), pin it, and describe its column.
        /// </summary>
        /// <remarks>
        ///     Two passes on purpose: the first decides every refusal without reading a buffer, so a mistake in the third
        ///     array costs no conversion or copy of the first two; the second takes the pins, and a failure there (a
        ///     released buffer) gives back every pin and every temporary already made.
        /// </remarks>
        /// <param name="columns">Column name → array.</param>
        /// <param name="snapshot"><c>true</c> to copy every array first (<c>ToDataView</c>), <c>false</c> to read them in place (<c>AsDataView</c>).</param>
        /// <returns>The view, owning the pins and the temporaries.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="columns"/> is null, or an array in it is null.</exception>
        /// <exception cref="ArgumentException">No column, an empty name, an array that is not 1-D or 2-D, or arrays with different numbers of rows.</exception>
        /// <exception cref="NotSupportedException">An array's dtype has no ML.NET column type, or a vector column is too wide.</exception>
        /// <exception cref="ObjectDisposedException">An array's buffer has already been released.</exception>
        private static NDArrayDataView BuildTableView(IReadOnlyDictionary<string, NDArray> columns, bool snapshot)
        {
            if (columns is null)
                throw new ArgumentNullException(nameof(columns));
            if (columns.Count == 0)
                throw new ArgumentException("at least one column is required.", nameof(columns));

            // ---- pass 1: every refusal the names, shapes and dtypes decide, before a buffer is read
            var names = new string[columns.Count];
            var arrays = new NDArray[columns.Count];
            var itemTypes = new PrimitiveDataViewType[columns.Count];
            long rows = -1;
            string rowsFrom = null;
            int c = 0;
            foreach (KeyValuePair<string, NDArray> column in columns)
            {
                if (string.IsNullOrEmpty(column.Key))
                    throw new ArgumentException("a column name is null or empty.", nameof(columns));
                NDArray array = column.Value ?? throw new ArgumentNullException(nameof(columns), $"the array of column '{column.Key}' is null.");
                Shape shape = array.Shape;
                if (shape.NDim != 1 && shape.NDim != 2)
                    throw new ArgumentException(
                        $"AsDataView (columns): column '{column.Key}' is {shape.NDim}-D (shape {shape}); a column is a 1-D (rows,) array (a scalar column) " +
                        "or a 2-D (rows, width) array (a vector column).", nameof(columns));
                // A 2-D array is a vector column, checked like the single-array view's (dtype, width); a 1-D one a scalar column.
                itemTypes[c] = shape.NDim == 2
                    ? ValidateVectorColumn(shape, FeedTypeCode(array.typecode))
                    : ToDataViewType(FeedTypeCode(array.typecode));
                long r = shape.Dimensions[0];
                if (rows < 0)
                {
                    rows = r;
                    rowsFrom = column.Key;
                }
                else if (r != rows)
                {
                    throw new ArgumentException(
                        $"AsDataView (columns): column '{column.Key}' has {r} rows but column '{rowsFrom}' has {rows}; every array must have the same " +
                        "length along axis 0 (the rows).", nameof(columns));
                }

                names[c] = column.Key;
                arrays[c] = array;
                c++;
            }

            // ---- pass 2: feed (in place, converted or snapshotted), pin, describe
            var sources = new NDArrayDataView.Source[arrays.Length];
            var plans = new NDArrayDataView.ColumnPlan[arrays.Length];
            var builder = new DataViewSchema.Builder();
            int made = 0;
            try
            {
                for (int s = 0; s < arrays.Length; s++)
                {
                    NDArray converted = null;
                    NDArray fed = snapshot ? Snapshot(arrays[s]) : ResolveFeedArray(arrays[s], out converted);
                    bool owns = snapshot || converted is not null;
                    IArraySlice pin;
                    try
                    {
                        pin = Pin(fed);
                    }
                    catch
                    {
                        if (owns)
                            fed.Dispose();   // the temporary of this array: nothing else will ever dispose it
                        throw;
                    }

                    sources[s] = new NDArrayDataView.Source(fed, pin, owns);
                    made = s + 1;

                    long[] strides = fed.Shape.Strides;
                    long offset = fed.Shape.Offset;
                    if (fed.ndim == 1)
                    {
                        plans[s] = new NDArrayDataView.ColumnPlan(names[s], fed.typecode, isVector: false, vectorWidth: 0, offset, strides[0], innerStride: 0, source: s);
                        builder.AddColumn(names[s], itemTypes[s]);
                    }
                    else
                    {
                        // width <= MaxBufferLength (< int.MaxValue) was checked by ValidateVectorColumn, so the cast is exact.
                        int width = (int)fed.Shape.Dimensions[1];
                        plans[s] = new NDArrayDataView.ColumnPlan(names[s], fed.typecode, isVector: true, width, offset, strides[0], strides[1], source: s);
                        builder.AddColumn(names[s], new VectorDataViewType(itemTypes[s], width));
                    }
                }

                return new NDArrayDataView(sources, plans, builder.ToSchema(), rows);
            }
            catch
            {
                // Give back what pass 2 made before the failure: every pin, and every temporary only the view would own.
                for (int s = 0; s < made; s++)
                {
                    sources[s].Slice.Release();
                    if (sources[s].OwnsFed)
                        sources[s].Fed.Dispose();
                }
                throw;
            }
        }

        /// <summary>Half→Single / Decimal→Double conversion, else a C-order copy — an owned array no caller shares.</summary>
        /// <remarks>
        ///     This used to be the only export path that never pinned the source at all: the conversion or the copy read
        ///     it straight away, so even a contiguous released source was snapshotted from freed memory, and the view
        ///     then served whatever array the pool had handed that memory to. Both reads now run under a reference on
        ///     the source's buffer (<see cref="ResolveFeedArray"/>, <see cref="CopyPinned"/>).
        /// </remarks>
        /// <param name="source">The array to snapshot.</param>
        /// <returns>A new array the caller owns and must dispose.</returns>
        /// <exception cref="ObjectDisposedException">The source's buffer has already been released.</exception>
        private static NDArray Snapshot(NDArray source)
        {
            NDArray fed = ResolveFeedArray(source, out NDArray converted);
            if (converted is not null)
                return converted;                       // astype already produced a fresh, owned array (in the source's memory order — the view reads any layout through its strides)
            return CopyPinned(fed);                     // an owned C-order copy of the source, taken under a pin of its buffer
        }

        /// <summary>
        ///     Every refusal of the single-vector-column view that the shape and the dtype alone decide: the rank (1-D or
        ///     2-D), the dtype's column type (Complex has none) and the vector width. The verbs run it on the SOURCE
        ///     before converting, copying or reading it, so an array that cannot cross costs nothing and a released one
        ///     is not read just to be refused; <see cref="BuildVectorView"/> runs it again on the array it feeds (the
        ///     conversion and the copy keep the shape, so the answer is the same).
        /// </summary>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="itemCode">The dtype the array crosses as (<see cref="FeedTypeCode"/> of the source's dtype).</param>
        /// <returns>The column's item type.</returns>
        /// <exception cref="ArgumentException">The array is not 1-D or 2-D.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type, or the feature axis is wider than ML.NET's buffers hold (<see cref="MaxBufferLength"/>).</exception>
        private static PrimitiveDataViewType ValidateVectorColumn(Shape shape, NPTypeCode itemCode)
        {
            int ndim = shape.NDim;
            if (ndim != 1 && ndim != 2)
                throw new ArgumentException($"AsDataView (single vector column) needs a 1-D or 2-D array; got {ndim}-D shape {shape}. A 2-D (rows, features) array becomes one vector column of `rows` rows; reshape higher-rank data first.");
            PrimitiveDataViewType itemType = ToDataViewType(itemCode);   // throws NotSupported for Complex

            // Every row reaches ML.NET as a dense VBuffer<T> of the column's width, which the cursor fills through
            // VBufferEditor.Create — and ML.NET's buffer sizing refuses any length past MaxBufferLength (2,146,435,071,
            // a little below a T[]'s own Array.MaxLength). The guard compared against int.MaxValue, so a width in between
            // built a view whose every row read then failed inside the cursor (ArgumentOutOfRangeException
            // "min must not exceed max" from ML.NET), long after the export had succeeded.
            long width = shape.Dimensions[ndim - 1];
            if (width > MaxBufferLength)
                throw new NotSupportedException(
                    $"AsDataView: an ML.NET vector column hands each row over as a VBuffer<T>, and ML.NET's buffers hold at most " +
                    $"{MaxBufferLength} elements (its ArrayUtils.ArrayMaxSize, a little below Array.MaxLength); the feature axis has {width}. " +
                    "Reshape so each row holds fewer features.");
            return itemType;
        }

        /// <summary>
        ///     Every refusal of the scalar-columns view that the shape, the dtype and the names alone decide: the rank,
        ///     the dtype's column type, the name count and each name. Run on the source before it is converted, copied or
        ///     read, and again by <see cref="BuildScalarView"/> on the array it feeds.
        /// </summary>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="itemCode">The dtype the array crosses as (<see cref="FeedTypeCode"/> of the source's dtype).</param>
        /// <param name="columnNames">The names, one per column.</param>
        /// <returns>The columns' item type.</returns>
        /// <exception cref="ArgumentException">The array is not 1-D or 2-D, the name count does not match the column count, or a name is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex).</exception>
        private static PrimitiveDataViewType ValidateScalarColumns(Shape shape, NPTypeCode itemCode, string[] columnNames)
        {
            int ndim = shape.NDim;
            if (ndim != 1 && ndim != 2)
                throw new ArgumentException($"AsDataView (scalar columns) needs a 1-D or 2-D array; got {ndim}-D shape {shape}.");
            PrimitiveDataViewType itemType = ToDataViewType(itemCode);   // throws NotSupported for Complex

            long colCount = ndim == 1 ? 1 : shape.Dimensions[1];
            if (columnNames.Length != colCount)
                throw new ArgumentException($"AsDataView (scalar columns): the array has {colCount} column(s) (shape {shape}) but {columnNames.Length} name(s) were given.", nameof(columnNames));
            for (int c = 0; c < colCount; c++)
                if (string.IsNullOrEmpty(columnNames[c]))
                    throw new ArgumentException($"column name {c} is null or empty.", nameof(columnNames));
            return itemType;
        }

        /// <summary>
        ///     Build the single-vector-column view over <paramref name="fed"/>, pinning its buffer for the view's
        ///     lifetime. The view reads rows lazily through the array's own strides, so any layout is fine.
        /// </summary>
        /// <param name="fed">The array the view reads (the caller's array, a conversion or a snapshot).</param>
        /// <param name="ownsFed"><c>true</c> when <paramref name="fed"/> is a temporary the view must dispose.</param>
        /// <param name="columnName">The vector column's name.</param>
        /// <returns>The view, holding the pin (and <paramref name="fed"/> when it owns it).</returns>
        /// <exception cref="ArgumentException">The array is not 1-D or 2-D.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type, or the feature axis is wider than ML.NET's buffers hold (<see cref="MaxBufferLength"/>).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        private static NDArrayDataView BuildVectorView(NDArray fed, bool ownsFed, string columnName)
        {
            PrimitiveDataViewType itemType = ValidateVectorColumn(fed.Shape, fed.typecode);
            int ndim = fed.ndim;
            NPTypeCode itemCode = fed.typecode;

            long[] dims = fed.Shape.Dimensions;
            long[] strides = fed.Shape.Strides;
            long offset = fed.Shape.Offset;

            long rowCount;
            long width;
            long rowStride;
            long innerStride;
            if (ndim == 1)
            {
                rowCount = 1;
                width = dims[0];
                rowStride = 0;
                innerStride = strides[0];
            }
            else
            {
                rowCount = dims[0];
                width = dims[1];
                rowStride = strides[0];
                innerStride = strides[1];
            }
            // width ≤ MaxBufferLength (< int.MaxValue) was checked by ValidateVectorColumn, so the (int) casts are exact.

            var plans = new[] { new NDArrayDataView.ColumnPlan(columnName, itemCode, isVector: true, (int)width, offset, rowStride, innerStride) };
            var builder = new DataViewSchema.Builder();
            builder.AddColumn(columnName, new VectorDataViewType(itemType, (int)width));
            DataViewSchema schema = builder.ToSchema();

            IArraySlice slice = Pin(fed);
            try
            {
                return new NDArrayDataView(fed, slice, ownsFed, plans, schema, rowCount);
            }
            catch
            {
                slice.Release();
                throw;
            }
        }

        /// <summary>
        ///     Build the scalar-columns view over <paramref name="fed"/>, pinning its buffer for the view's lifetime.
        ///     Each column reads its values lazily through the array's own strides, so any layout is fine.
        /// </summary>
        /// <param name="fed">The array the view reads (the caller's array, a conversion or a snapshot).</param>
        /// <param name="ownsFed"><c>true</c> when <paramref name="fed"/> is a temporary the view must dispose.</param>
        /// <param name="columnNames">One non-empty name per column.</param>
        /// <returns>The view, holding the pin (and <paramref name="fed"/> when it owns it).</returns>
        /// <exception cref="ArgumentException">The array is not 1-D or 2-D, the name count does not match the column count, or a name is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex).</exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        private static NDArrayDataView BuildScalarView(NDArray fed, bool ownsFed, string[] columnNames)
        {
            PrimitiveDataViewType itemType = ValidateScalarColumns(fed.Shape, fed.typecode, columnNames);
            int ndim = fed.ndim;
            NPTypeCode itemCode = fed.typecode;

            long[] dims = fed.Shape.Dimensions;
            long[] strides = fed.Shape.Strides;
            long offset = fed.Shape.Offset;

            // The name count and every name were checked by ValidateScalarColumns.
            long rowCount = dims[0];
            long colCount = ndim == 1 ? 1 : dims[1];

            long rowStride = strides[0];
            long colStride = ndim == 1 ? 0 : strides[1];

            var plans = new NDArrayDataView.ColumnPlan[colCount];
            var builder = new DataViewSchema.Builder();
            for (int c = 0; c < colCount; c++)
            {
                string name = columnNames[c];
                long elemBase = offset + c * colStride;
                plans[c] = new NDArrayDataView.ColumnPlan(name, itemCode, isVector: false, vectorWidth: 0, elemBase, rowStride, innerStride: 0);
                builder.AddColumn(name, itemType);
            }
            DataViewSchema schema = builder.ToSchema();

            IArraySlice slice = Pin(fed);
            try
            {
                return new NDArrayDataView(fed, slice, ownsFed, plans, schema, rowCount);
            }
            catch
            {
                slice.Release();
                throw;
            }
        }

        // ===========================  NumSharp  ->  ML.NET VBuffer  ===========================

        /// <summary>
        ///     Copy a NumSharp array into a dense ML.NET <see cref="VBuffer{T}"/> of length <c>size</c>, reading the
        ///     array in logical (C) order. Always a COPY — <see cref="VBuffer{T}"/> exposes no public constructor over
        ///     foreign memory, so a zero-copy view is not possible (this is inherent to VBuffer's dense/sparse design).
        ///
        ///     <para><typeparamref name="T"/> must be the C# type ML.NET uses for the array's dtype
        ///     (<see cref="ToDataViewClrType"/>): <see cref="ushort"/> for <see cref="NPTypeCode.Char"/>, otherwise the
        ///     dtype's own type — a mismatch throws. <see cref="NPTypeCode.Half"/> converts to Single (request
        ///     <c>ToVBuffer&lt;float&gt;</c>) and <see cref="NPTypeCode.Decimal"/> to Double
        ///     (<c>ToVBuffer&lt;double&gt;</c>).</para>
        /// </summary>
        /// <remarks>
        ///     <para>The element type and the size are checked before anything is converted, copied or allocated; the
        ///     buffer's liveness is checked next, even for an empty array.</para>
        ///     <para>The size limit is ML.NET's, not the array's: ML.NET's buffer sizing (its internal
        ///     <c>ArrayUtils.ArrayMaxSize</c>) stops at 2,146,435,071 elements. A longer <see cref="VBuffer{T}"/> can be
        ///     built, but <see cref="VBuffer{T}.CopyTo(ref VBuffer{T})"/> and every cursor getter refuse its length, so it
        ///     would fail later, inside ML.NET; it is refused here instead.</para>
        /// </remarks>
        /// <typeparam name="T">The C# element type ML.NET uses for the array's dtype (float for Half, double for Decimal); anything else is refused.</typeparam>
        /// <param name="source">The array to copy; any layout.</param>
        /// <returns>A new dense buffer of length <c>size</c> holding the values in logical (C) order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> is not the element type ML.NET uses for the array's dtype.</exception>
        /// <exception cref="NotSupportedException">
        ///     The dtype has no ML.NET type (Complex, or text: a U, S or StringDType array), or the array has more elements than ML.NET's buffers hold
        ///     (2,146,435,071 elements — a little below a <c>T[]</c>'s own <see cref="Array.MaxLength"/>, see the remarks).
        /// </exception>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        public static unsafe VBuffer<T> ToVBuffer<T>(this NDArray source) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));

            // Decide everything the dtype and the shape alone decide BEFORE the source is read: the element type is
            // that of the FED dtype (Half crosses as Single, Decimal as Double), and the conversion and the copy never
            // change the element count.
            NPTypeCode feedCode = FeedTypeCode(source.typecode);
            EnsureElementType<T>(feedCode, nameof(ToVBuffer));

            // The limit is ML.NET's MaxBufferLength (2,146,435,071), NOT int.MaxValue. Comparing against int.MaxValue let
            // every size past Array.MaxLength (2,147,483,591, a T[]'s own limit) reach `new T[n]`, which fails with
            // OutOfMemoryException ("Array dimensions exceeded supported range") instead of this documented refusal;
            // and between the two limits it built a VBuffer ML.NET cannot copy (VBuffer.CopyTo and every cursor getter
            // refuse the length), so the failure surfaced only later, inside ML.NET.
            long size = source.Shape.Size;
            if (size > MaxBufferLength)
                throw new NotSupportedException(
                    $"ToVBuffer: ML.NET's buffers hold at most {MaxBufferLength} elements (its ArrayUtils.ArrayMaxSize, a little below Array.MaxLength) — " +
                    $"a longer VBuffer<T> cannot be copied, and ML.NET copies one in every getter; the array has {size}. Expose it with AsDataView " +
                    "instead (as (rows, features), or a 1-D array as one scalar column): the view reads its rows lazily, so only one row must fit.");

            // Take the reference on the SOURCE's buffer before reading it in any way: Pin's TryAddRef is the only check
            // that the buffer has not been released. The Half / Decimal conversion and the densifying copy below used to
            // run first (and an empty array was never pinned at all), so a released buffer was read silently and the
            // buffer carried another array's values. Holding the reference across every read also keeps the buffer
            // alive while it is read; the managed array is allocated only once the source is known to be live.
            IArraySlice sourcePin = Pin(source);
            try
            {
                using NDArray converted = feedCode != source.typecode ? source.astype(feedCode) : null;
                NDArray logical = converted ?? source;

                var data = new T[size];
                if (size > 0)
                {
                    using NDArray dense = logical.Shape.IsContiguous ? null : logical.copy();
                    NDArray src = dense ?? logical;
                    IArraySlice slice = Pin(src);
                    try
                    {
                        long nbytes = size * src.dtypesize;
                        byte* p = (byte*)slice.Address + src.Shape.Offset * src.dtypesize;
                        fixed (T* dst = data)
                            Buffer.MemoryCopy(p, dst, nbytes, nbytes);
                    }
                    finally
                    {
                        slice.Release();
                        GC.KeepAlive(src);
                    }
                }

                return new VBuffer<T>((int)size, data);
            }
            finally
            {
                sourcePin.Release();
                GC.KeepAlive(source);
            }
        }
    }
}
