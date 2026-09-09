using System;
using System.Numerics;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.MLNet
{
    /// <summary>
    ///     Interop between NumSharp <see cref="NDArray"/> and ML.NET data — the
    ///     <see cref="IDataView"/> pipeline currency and the <see cref="VBuffer{T}"/> vector primitive.
    ///
    ///     <para>This is a CONVERSION library in the mould of <c>NumSharp.Interop.pythonnet</c> /
    ///     <c>NumSharp.Interop.OnnxRuntime</c>: it moves data (or descriptions of the same memory) across the
    ///     boundary. It is NOT an engine backend — ML.NET does not compute NumSharp operations — so there is no
    ///     <c>TensorEngine</c> seam, no <c>[ModuleInitializer]</c>, and referencing the package changes nothing
    ///     until a verb is called. It depends only on the standalone <c>Microsoft.ML.DataView</c> contract
    ///     package (<see cref="IDataView"/>, <see cref="VBuffer{T}"/>, <see cref="DataViewSchema"/>,
    ///     <see cref="DataViewType"/>); the runtime you already reference — <c>Microsoft.ML</c> (MLContext,
    ///     trainers, transforms) — brings everything else.</para>
    ///
    ///     <para><b>The verbs</b> follow the house convention: <c>As…</c> shares memory, <c>To…</c> copies.</para>
    ///     <list type="table">
    ///         <item><term><see cref="Export.AsDataView(NDArray, string)"/></term><description>NumSharp → ML.NET, an <see cref="IDataView"/> that reads the array's buffer <b>lazily</b> through its strides (any layout, no copy; the array is ARC-pinned for the view's lifetime)</description></item>
    ///         <item><term><see cref="Export.ToDataView(NDArray, string)"/></term><description>NumSharp → ML.NET, an <see cref="IDataView"/> over an independent snapshot (no lifetime coupling)</description></item>
    ///         <item><term><see cref="Export.ToVBuffer{T}(NDArray)"/></term><description>NumSharp → ML.NET, a dense <see cref="VBuffer{T}"/> (a copy: <see cref="VBuffer{T}"/> exposes no public zero-copy constructor)</description></item>
    ///         <item><term><see cref="Import.ToNDArray(IDataView, string)"/></term><description>ML.NET → NumSharp, materialize a column across all rows into a fresh owning <see cref="NDArray"/></description></item>
    ///         <item><term><see cref="Import.ToNDArray{T}(VBuffer{T})"/></term><description>ML.NET → NumSharp, a dense 1-D <see cref="NDArray"/> from a <see cref="VBuffer{T}"/> (sparse buffers are densified)</description></item>
    ///     </list>
    ///
    ///     <para><b>Dtype map</b> — 11 NumSharp dtypes map to an ML.NET column type (bool → <see cref="BooleanDataViewType"/>;
    ///     the 8 integers, float and double → the matching <see cref="NumberDataViewType"/>). <see cref="NPTypeCode.Char"/>
    ///     maps to <see cref="NumberDataViewType.UInt16"/> (UTF-16 code units), one-way. ML.NET has no half, decimal or
    ///     complex column type: <see cref="NPTypeCode.Half"/> is converted to Single and <see cref="NPTypeCode.Decimal"/>
    ///     to Double by the view-producing verbs (the low-level maps refuse them); <see cref="NPTypeCode.Complex"/> is a
    ///     genuine gap and is refused with the workaround named. See <see cref="ToDataViewType"/> /
    ///     <see cref="FromDataViewType"/>.</para>
    /// </summary>
    public static partial class NDArrayMLNetInterop
    {
        private static int _liveExports;

        /// <summary>
        ///     Number of live <see cref="Export.AsDataView(NDArray, string)"/> views — NumSharp buffers currently
        ///     shared into an ML.NET <see cref="IDataView"/>. Each <see cref="NDArrayDataView"/> increments it on
        ///     creation and decrements it when disposed (or finalized); a steady non-zero count is a leak. (There is
        ///     no import counter: every ML.NET → NumSharp verb copies — <see cref="VBuffer{T}"/> exposes no memory to
        ///     lease — so nothing is ever pinned on the way back.)
        /// </summary>
        public static int LiveExports => Volatile.Read(ref _liveExports);

        internal static void ExportOpened() => Interlocked.Increment(ref _liveExports);
        internal static void ExportClosed() => Interlocked.Decrement(ref _liveExports);

        // ===================================  the dtype map  ===================================

        /// <summary>
        ///     The ML.NET <see cref="PrimitiveDataViewType"/> a NumSharp dtype crosses as. Direct for the 11 numeric
        ///     dtypes; <see cref="NPTypeCode.Char"/> → <see cref="NumberDataViewType.UInt16"/> (UTF-16 code units, the
        ///     rule the pythonnet / ONNX bridges also use).
        /// </summary>
        /// <exception cref="NotSupportedException">
        ///     <see cref="NPTypeCode.Half"/> / <see cref="NPTypeCode.Decimal"/> (no ML.NET column type — convert to
        ///     Single / Double first; the view-producing verbs do so automatically) or <see cref="NPTypeCode.Complex"/>
        ///     (ML.NET has no complex column type).
        /// </exception>
        public static PrimitiveDataViewType ToDataViewType(NPTypeCode code)
        {
            if (!TryToDataViewType(code, out PrimitiveDataViewType type))
                throw new NotSupportedException(UnsupportedExportMessage(code));
            return type;
        }

        /// <summary><see cref="ToDataViewType"/> without the throw; <c>false</c> for Half / Decimal / Complex.</summary>
        public static bool TryToDataViewType(NPTypeCode code, out PrimitiveDataViewType type)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: type = BooleanDataViewType.Instance; return true;
                case NPTypeCode.Byte: type = NumberDataViewType.Byte; return true;
                case NPTypeCode.SByte: type = NumberDataViewType.SByte; return true;
                case NPTypeCode.Int16: type = NumberDataViewType.Int16; return true;
                case NPTypeCode.UInt16: type = NumberDataViewType.UInt16; return true;
                case NPTypeCode.Int32: type = NumberDataViewType.Int32; return true;
                case NPTypeCode.UInt32: type = NumberDataViewType.UInt32; return true;
                case NPTypeCode.Int64: type = NumberDataViewType.Int64; return true;
                case NPTypeCode.UInt64: type = NumberDataViewType.UInt64; return true;
                case NPTypeCode.Char: type = NumberDataViewType.UInt16; return true;    // UTF-16 code units
                case NPTypeCode.Single: type = NumberDataViewType.Single; return true;
                case NPTypeCode.Double: type = NumberDataViewType.Double; return true;
                default: type = null; return false;                                     // Half, Decimal, Complex
            }
        }

        /// <summary>
        ///     The NumSharp dtype an ML.NET column type reads back as, unwrapping a <see cref="VectorDataViewType"/> to
        ///     its item type. Mapped by the type's <see cref="DataViewType.RawType"/>, so a
        ///     <see cref="NumberDataViewType.UInt16"/> comes back as <see cref="NPTypeCode.UInt16"/>, never
        ///     <see cref="NPTypeCode.Char"/> (an ML.NET column carries no "these are characters" bit); a
        ///     <see cref="KeyDataViewType"/> comes back as its unsigned storage integer.
        /// </summary>
        /// <exception cref="NotSupportedException">
        ///     A text column (NumSharp has no string dtype) or any raw type with no NumSharp dtype.
        /// </exception>
        public static NPTypeCode FromDataViewType(DataViewType type)
        {
            if (!TryFromDataViewType(type, out NPTypeCode code))
                throw new NotSupportedException(UnsupportedImportMessage(type));
            return code;
        }

        /// <summary><see cref="FromDataViewType"/> without the throw.</summary>
        public static bool TryFromDataViewType(DataViewType type, out NPTypeCode code)
        {
            if (type is null)
            {
                code = default;
                return false;
            }
            Type raw = type is VectorDataViewType vector ? vector.ItemType.RawType : type.RawType;
            return TryFromRawType(raw, out code);
        }

        /// <summary>The NumSharp dtype for a column's CLR <see cref="DataViewType.RawType"/>.</summary>
        internal static bool TryFromRawType(Type raw, out NPTypeCode code)
        {
            if (raw == typeof(bool)) { code = NPTypeCode.Boolean; return true; }
            if (raw == typeof(byte)) { code = NPTypeCode.Byte; return true; }
            if (raw == typeof(sbyte)) { code = NPTypeCode.SByte; return true; }
            if (raw == typeof(short)) { code = NPTypeCode.Int16; return true; }
            if (raw == typeof(ushort)) { code = NPTypeCode.UInt16; return true; }
            if (raw == typeof(int)) { code = NPTypeCode.Int32; return true; }
            if (raw == typeof(uint)) { code = NPTypeCode.UInt32; return true; }
            if (raw == typeof(long)) { code = NPTypeCode.Int64; return true; }
            if (raw == typeof(ulong)) { code = NPTypeCode.UInt64; return true; }
            if (raw == typeof(float)) { code = NPTypeCode.Single; return true; }
            if (raw == typeof(double)) { code = NPTypeCode.Double; return true; }
            code = default;
            return false;
        }

        /// <summary>
        ///     The C# element type ML.NET stores a NumSharp dtype's column values in — <see cref="char"/> becomes
        ///     <see cref="ushort"/>; <c>null</c> for Half / Decimal / Complex (no ML.NET column type). This is the
        ///     <typeparamref name="T"/> a <see cref="VBuffer{T}"/> / <see cref="ValueGetter{T}"/> for the column uses.
        /// </summary>
        public static Type ToDataViewClrType(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: return typeof(bool);
                case NPTypeCode.Byte: return typeof(byte);
                case NPTypeCode.SByte: return typeof(sbyte);
                case NPTypeCode.Int16: return typeof(short);
                case NPTypeCode.UInt16: return typeof(ushort);
                case NPTypeCode.Int32: return typeof(int);
                case NPTypeCode.UInt32: return typeof(uint);
                case NPTypeCode.Int64: return typeof(long);
                case NPTypeCode.UInt64: return typeof(ulong);
                case NPTypeCode.Char: return typeof(ushort);
                case NPTypeCode.Single: return typeof(float);
                case NPTypeCode.Double: return typeof(double);
                default: return null;
            }
        }

        /// <summary>The NumSharp dtype a <see cref="VBuffer{T}"/> element type reads back as (<see cref="char"/> → <see cref="NPTypeCode.Char"/>).</summary>
        /// <exception cref="NotSupportedException">A type ML.NET vectors do not carry (Half / Decimal / Complex / anything else).</exception>
        internal static NPTypeCode TypeCodeOf<T>() where T : unmanaged
        {
            if (typeof(T) == typeof(bool)) return NPTypeCode.Boolean;
            if (typeof(T) == typeof(byte)) return NPTypeCode.Byte;
            if (typeof(T) == typeof(sbyte)) return NPTypeCode.SByte;
            if (typeof(T) == typeof(short)) return NPTypeCode.Int16;
            if (typeof(T) == typeof(ushort)) return NPTypeCode.UInt16;
            if (typeof(T) == typeof(int)) return NPTypeCode.Int32;
            if (typeof(T) == typeof(uint)) return NPTypeCode.UInt32;
            if (typeof(T) == typeof(long)) return NPTypeCode.Int64;
            if (typeof(T) == typeof(ulong)) return NPTypeCode.UInt64;
            if (typeof(T) == typeof(char)) return NPTypeCode.Char;
            if (typeof(T) == typeof(float)) return NPTypeCode.Single;
            if (typeof(T) == typeof(double)) return NPTypeCode.Double;
            throw new NotSupportedException($"VBuffer<{typeof(T).Name}> has no NumSharp dtype: ML.NET vectors carry bool, the eight integer types, float and double (System.Half / decimal / complex have no ML.NET column type).");
        }

        internal static string UnsupportedExportMessage(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Half:
                    return "ML.NET has no half-precision column type (NumberDataViewType has Single and Double, not Half). The view-producing " +
                           "verbs (AsDataView / ToDataView / ToVBuffer) convert a Half array to Single for you (lossless — every Half value is an " +
                           "exact Single); the low-level dtype map refuses it — convert explicitly with nd.astype(NPTypeCode.Single).";
                case NPTypeCode.Decimal:
                    return "ML.NET has no decimal column type (NumberDataViewType has Single and Double). The view-producing verbs (AsDataView / " +
                           "ToDataView / ToVBuffer) convert a Decimal array to Double for you (lossy beyond ~16 significant digits); the low-level " +
                           "dtype map refuses it — convert explicitly with nd.astype(NPTypeCode.Double).";
                case NPTypeCode.Complex:
                    return "ML.NET has no complex column type. Split the array into real and imaginary planes (np.real / np.imag) and feed those as " +
                           "two columns / two vectors.";
                default:
                    return $"NumSharp dtype {code} has no ML.NET column type.";
            }
        }

        internal static string UnsupportedImportMessage(DataViewType type)
        {
            if (type is null)
                return "the column type is null.";
            if (type is TextDataViewType || (type is VectorDataViewType v && v.ItemType is TextDataViewType))
                return "ML.NET text columns have no NumSharp dtype (NumSharp has no string / object arrays). Read them with the ML.NET " +
                       "cursor's ValueGetter<ReadOnlyMemory<char>> instead.";
            Type raw = type is VectorDataViewType vec ? vec.ItemType.RawType : type.RawType;
            return $"ML.NET column type {type} (raw type {raw.Name}) has no NumSharp dtype: NumSharp carries bool, the eight integer types, " +
                   "float and double.";
        }

        /// <summary>
        ///     Validate that a caller-chosen <typeparamref name="T"/> is the element type ML.NET uses for the array's
        ///     dtype (<see cref="ToDataViewClrType"/>). A mismatch would silently reinterpret the values as a different
        ///     dtype, so it is refused up front.
        /// </summary>
        internal static void EnsureElementType<T>(NPTypeCode code, string verb) where T : unmanaged
        {
            Type want = ToDataViewClrType(code);
            if (want is null)
                throw new NotSupportedException(UnsupportedExportMessage(code));
            if (typeof(T) != want)
                throw new ArgumentException(
                    $"{verb}<{typeof(T).Name}>: a NumSharp {code} array crosses as VBuffer<{want.Name}> " +
                    $"({(code == NPTypeCode.Char ? "chars are UTF-16 code units" : "the dtype's own C# type")}); " +
                    $"reinterpreting it as {typeof(T).Name} would hand ML.NET the wrong dtype. Call {verb}<{want.Name}>() or convert with astype() first.");
        }

        // ===================================  shared plumbing  ===================================

        /// <summary>
        ///     The NumSharp array actually shared with ML.NET for a source dtype: Half and Decimal have no ML.NET
        ///     column type, so they are converted (Half → Single, lossless; Decimal → Double, lossy beyond ~16
        ///     digits) to a TEMPORARY the caller owns; every other dtype is fed unchanged. Returns the array to feed
        ///     and, via <paramref name="converted"/>, the temporary to dispose (null when none was made).
        /// </summary>
        internal static NDArray ResolveFeedArray(NDArray source, out NDArray converted)
        {
            converted = null;
            switch (source.typecode)
            {
                case NPTypeCode.Half: return converted = source.astype(NPTypeCode.Single);
                case NPTypeCode.Decimal: return converted = source.astype(NPTypeCode.Double);
                default: return source;
            }
        }

        /// <summary>ML.NET's row-major <c>long[]</c> shape → a NumSharp <see cref="Shape"/> (an empty array is the 0-d scalar).</summary>
        internal static Shape ShapeFrom(long[] dims)
        {
            if (dims is null || dims.Length == 0)
                return Shape.Scalar;
            return new Shape(dims);
        }

        /// <summary>
        ///     Take an ARC reference on the array's buffer so it survives an explicit dispose / collection of the
        ///     source while a view still reads it. The caller owns the reference and must <see cref="IArraySlice.Release"/> it.
        /// </summary>
        internal static IArraySlice Pin(NDArray source)
        {
            IArraySlice slice = source.Storage.InternalArray;
            if (!slice.TryAddRef())
                throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");
            return slice;
        }
    }
}
