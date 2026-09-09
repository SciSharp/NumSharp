using System;
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
        /// <param name="source">The array to expose.</param>
        /// <param name="columnName">The name of the single vector column (ML.NET's convention is <c>"Features"</c>).</param>
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or <paramref name="columnName"/> is null/empty.</exception>
        /// <exception cref="NotSupportedException">The dtype has no ML.NET column type (Complex).</exception>
        public static NDArrayDataView AsDataView(this NDArray source, string columnName = "Features")
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (string.IsNullOrEmpty(columnName))
                throw new ArgumentException("a column name is required.", nameof(columnName));
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
        /// <exception cref="ArgumentException"><paramref name="source"/> is not 1-D or 2-D, or the name count does not match the column count.</exception>
        public static NDArrayDataView AsDataView(this NDArray source, string[] columnNames)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (columnNames is null || columnNames.Length == 0)
                throw new ArgumentException("at least one column name is required.", nameof(columnNames));
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
        public static NDArrayDataView ToDataView(this NDArray source, string columnName = "Features")
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (string.IsNullOrEmpty(columnName))
                throw new ArgumentException("a column name is required.", nameof(columnName));
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
        public static NDArrayDataView ToDataView(this NDArray source, string[] columnNames)
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            if (columnNames is null || columnNames.Length == 0)
                throw new ArgumentException("at least one column name is required.", nameof(columnNames));
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

        /// <summary>Half→Single / Decimal→Double conversion then a C-order copy — an owned array no caller shares.</summary>
        private static NDArray Snapshot(NDArray source)
        {
            NDArray fed = ResolveFeedArray(source, out NDArray converted);
            if (converted is not null)
                return converted;                       // astype already produced a fresh, owned, C-order copy
            return fed.copy();                          // an owned C-order copy of the source
        }

        private static NDArrayDataView BuildVectorView(NDArray fed, bool ownsFed, string columnName)
        {
            int ndim = fed.ndim;
            if (ndim != 1 && ndim != 2)
                throw new ArgumentException($"AsDataView (single vector column) needs a 1-D or 2-D array; got {ndim}-D shape {fed.Shape}. A 2-D (rows, features) array becomes one vector column of `rows` rows; reshape higher-rank data first.");
            NPTypeCode itemCode = fed.typecode;
            PrimitiveDataViewType itemType = ToDataViewType(itemCode);   // throws NotSupported for Complex

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
            if (width > int.MaxValue)
                throw new NotSupportedException($"AsDataView: an ML.NET vector column holds at most {int.MaxValue} elements, the feature axis has {width}.");

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

        private static NDArrayDataView BuildScalarView(NDArray fed, bool ownsFed, string[] columnNames)
        {
            int ndim = fed.ndim;
            if (ndim != 1 && ndim != 2)
                throw new ArgumentException($"AsDataView (scalar columns) needs a 1-D or 2-D array; got {ndim}-D shape {fed.Shape}.");
            NPTypeCode itemCode = fed.typecode;
            PrimitiveDataViewType itemType = ToDataViewType(itemCode);   // throws NotSupported for Complex

            long[] dims = fed.Shape.Dimensions;
            long[] strides = fed.Shape.Strides;
            long offset = fed.Shape.Offset;

            long rowCount = dims[0];
            long colCount = ndim == 1 ? 1 : dims[1];
            if (columnNames.Length != colCount)
                throw new ArgumentException($"AsDataView (scalar columns): the array has {colCount} column(s) (shape {fed.Shape}) but {columnNames.Length} name(s) were given.", nameof(columnNames));

            long rowStride = strides[0];
            long colStride = ndim == 1 ? 0 : strides[1];

            var plans = new NDArrayDataView.ColumnPlan[colCount];
            var builder = new DataViewSchema.Builder();
            for (int c = 0; c < colCount; c++)
            {
                string name = columnNames[c];
                if (string.IsNullOrEmpty(name))
                    throw new ArgumentException($"column name {c} is null or empty.", nameof(columnNames));
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
        public static unsafe VBuffer<T> ToVBuffer<T>(this NDArray source) where T : unmanaged
        {
            if (source is null)
                throw new ArgumentNullException(nameof(source));
            using NDArray converted = source.typecode == NPTypeCode.Half ? source.astype(NPTypeCode.Single)
                                    : source.typecode == NPTypeCode.Decimal ? source.astype(NPTypeCode.Double)
                                    : null;
            NDArray logical = converted ?? source;
            EnsureElementType<T>(logical.typecode, nameof(ToVBuffer));

            long size = logical.Shape.Size;
            if (size > int.MaxValue)
                throw new NotSupportedException($"ToVBuffer: a VBuffer<T> holds at most {int.MaxValue} elements, the array has {size}.");

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
    }
}
