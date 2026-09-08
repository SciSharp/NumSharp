using System;
using System.Numerics;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.OnnxRuntime
{
    /// <summary>
    ///     Zero-copy interop between NumSharp <see cref="NDArray"/> and ONNX Runtime tensors
    ///     (<see cref="OrtValue"/> — the ORT ≥ 1.16 preferred API — and the legacy
    ///     <see cref="DenseTensor{T}"/> / <see cref="NamedOnnxValue"/> surface).
    ///
    ///     <para>This is a CONVERSION library in the mould of <c>NumSharp.Interop.pythonnet</c>: it moves
    ///     data (or rather, descriptions of the same memory) across the boundary. It is NOT an engine
    ///     backend — ONNX Runtime does not compute NumSharp operations, it consumes tensors NumSharp
    ///     produces and produces tensors NumSharp reads — so there is no <c>TensorEngine</c> seam, no
    ///     <c>[ModuleInitializer]</c>, and referencing the package changes nothing until a verb is
    ///     called.</para>
    ///
    ///     <para><b>The verbs</b> follow the house convention: <c>As…</c> shares memory (zero-copy view),
    ///     <c>To…</c> copies.</para>
    ///     <list type="table">
    ///         <item><term><see cref="Export.AsOrtValue"/></term><description>NumSharp → ORT, zero-copy <see cref="OrtValue"/> over the NDArray's buffer (C-contiguous only; the buffer is ARC-rooted for the handle's lifetime)</description></item>
    ///         <item><term><see cref="Export.ToOrtValue"/></term><description>NumSharp → ORT, an ORT-allocated copy (any layout, no lifetime coupling)</description></item>
    ///         <item><term><see cref="Export.AsDenseTensor{T}"/> / <see cref="Export.ToDenseTensor{T}"/></term><description>the legacy <see cref="DenseTensor{T}"/> twins</description></item>
    ///         <item><term><see cref="Import.ToNDArray(OrtValue)"/></term><description>ORT → NumSharp, a fresh owning C-contiguous copy (the safe default)</description></item>
    ///         <item><term><see cref="Import.AsNDArray(OrtValue, bool)"/></term><description>ORT → NumSharp, a zero-copy view over ORT's buffer (CPU tensors only), optionally taking ownership of the OrtValue</description></item>
    ///     </list>
    ///
    ///     <para><b>Dtype map</b> — 12 NumSharp dtypes cross zero-copy (bool, the 8 integers, Half ↔ Float16
    ///     as a bit-identical reinterpret, float, double); <see cref="NPTypeCode.Char"/> exports as
    ///     <see cref="TensorElementType.UInt16"/> (UTF-16 code units); <see cref="NPTypeCode.Decimal"/> has
    ///     no ONNX type and is converted to Double by the array-producing verbs (the low-level maps refuse
    ///     it); <see cref="NPTypeCode.Complex"/>, BFloat16 and String are genuine mutual gaps and are
    ///     refused with a message naming the workaround. See <see cref="ToTensorElementType"/> /
    ///     <see cref="FromTensorElementType"/>.</para>
    /// </summary>
    public static partial class NDArrayOnnxInterop
    {
        private static int _liveExports;
        private static int _liveImports;

        /// <summary>
        ///     Number of live export handles (<see cref="OrtTensor"/> / <see cref="OrtTensor{T}"/>) —
        ///     NumSharp buffers currently pinned for ONNX Runtime. Every handle increments it on creation
        ///     and decrements it when disposed (or finalized); a steady non-zero count is a leak.
        /// </summary>
        public static int LiveExports => Volatile.Read(ref _liveExports);

        /// <summary>
        ///     Number of live import leases — NumSharp views (<see cref="Import.AsNDArray(OrtValue, bool)"/> /
        ///     <see cref="Import.AsNDArray{T}(DenseTensor{T})"/>) currently holding ORT (or pinned managed)
        ///     memory. Released when the LAST NumSharp view over the memory — derived slices included — is
        ///     disposed or collected.
        /// </summary>
        public static int LiveImports => Volatile.Read(ref _liveImports);

        internal static void ExportOpened() => Interlocked.Increment(ref _liveExports);
        internal static void ExportClosed() => Interlocked.Decrement(ref _liveExports);
        internal static void ImportOpened() => Interlocked.Increment(ref _liveImports);
        internal static void ImportClosed() => Interlocked.Decrement(ref _liveImports);

        // ===================================  the dtype map  ===================================

        /// <summary>
        ///     The ONNX tensor element type a NumSharp dtype crosses as. Zero-copy for the 12 numeric
        ///     dtypes incl. <see cref="NPTypeCode.Half"/> → <see cref="TensorElementType.Float16"/>
        ///     (ORT's <c>Float16</c> is a blittable 16-bit IEEE struct, binary-identical to
        ///     <see cref="System.Half"/>) and <see cref="NPTypeCode.Char"/> → <see cref="TensorElementType.UInt16"/>
        ///     (UTF-16 code units, the rule the pythonnet bridge also uses).
        /// </summary>
        /// <exception cref="NotSupportedException">
        ///     <see cref="NPTypeCode.Decimal"/> (no ONNX type — convert to Double first; the array-producing
        ///     verbs do so automatically) or <see cref="NPTypeCode.Complex"/> (ORT accepts no complex tensors).
        /// </exception>
        public static TensorElementType ToTensorElementType(NPTypeCode code)
        {
            if (!TryToTensorElementType(code, out TensorElementType type))
                throw new NotSupportedException(UnsupportedExportMessage(code));
            return type;
        }

        /// <summary><see cref="ToTensorElementType"/> without the throw; <c>false</c> for Decimal / Complex.</summary>
        public static bool TryToTensorElementType(NPTypeCode code, out TensorElementType type)
        {
            switch (code)
            {
                case NPTypeCode.Boolean: type = TensorElementType.Bool; return true;
                case NPTypeCode.Byte: type = TensorElementType.UInt8; return true;
                case NPTypeCode.SByte: type = TensorElementType.Int8; return true;
                case NPTypeCode.Int16: type = TensorElementType.Int16; return true;
                case NPTypeCode.UInt16: type = TensorElementType.UInt16; return true;
                case NPTypeCode.Int32: type = TensorElementType.Int32; return true;
                case NPTypeCode.UInt32: type = TensorElementType.UInt32; return true;
                case NPTypeCode.Int64: type = TensorElementType.Int64; return true;
                case NPTypeCode.UInt64: type = TensorElementType.UInt64; return true;
                case NPTypeCode.Char: type = TensorElementType.UInt16; return true;     // UTF-16 code units
                case NPTypeCode.Half: type = TensorElementType.Float16; return true;    // bit-identical reinterpret
                case NPTypeCode.Single: type = TensorElementType.Float; return true;
                case NPTypeCode.Double: type = TensorElementType.Double; return true;
                default: type = default; return false;                                   // Decimal, Complex
            }
        }

        /// <summary>
        ///     The NumSharp dtype an ONNX tensor element type reads back as. Directional on purpose:
        ///     <see cref="TensorElementType.UInt16"/> comes back as <see cref="NPTypeCode.UInt16"/>, never
        ///     <see cref="NPTypeCode.Char"/> (an ORT tensor carries no "these are characters" bit).
        /// </summary>
        /// <exception cref="NotSupportedException">
        ///     <see cref="TensorElementType.String"/> (read with <see cref="OrtValue.GetStringTensorAsArray"/>),
        ///     <see cref="TensorElementType.BFloat16"/> (NumSharp has no bfloat16), Complex64/Complex128.
        /// </exception>
        public static NPTypeCode FromTensorElementType(TensorElementType type)
        {
            if (!TryFromTensorElementType(type, out NPTypeCode code))
                throw new NotSupportedException(UnsupportedImportMessage(type));
            return code;
        }

        /// <summary><see cref="FromTensorElementType"/> without the throw.</summary>
        public static bool TryFromTensorElementType(TensorElementType type, out NPTypeCode code)
        {
            switch (type)
            {
                case TensorElementType.Bool: code = NPTypeCode.Boolean; return true;
                case TensorElementType.UInt8: code = NPTypeCode.Byte; return true;
                case TensorElementType.Int8: code = NPTypeCode.SByte; return true;
                case TensorElementType.Int16: code = NPTypeCode.Int16; return true;
                case TensorElementType.UInt16: code = NPTypeCode.UInt16; return true;
                case TensorElementType.Int32: code = NPTypeCode.Int32; return true;
                case TensorElementType.UInt32: code = NPTypeCode.UInt32; return true;
                case TensorElementType.Int64: code = NPTypeCode.Int64; return true;
                case TensorElementType.UInt64: code = NPTypeCode.UInt64; return true;
                case TensorElementType.Float16: code = NPTypeCode.Half; return true;
                case TensorElementType.Float: code = NPTypeCode.Single; return true;
                case TensorElementType.Double: code = NPTypeCode.Double; return true;
                default: code = default; return false;                                   // String, BFloat16, Complex64/128, DataTypeMax
            }
        }

        /// <summary>
        ///     The C# element type ORT's <see cref="DenseTensor{T}"/> uses for a NumSharp dtype —
        ///     <see cref="System.Half"/> becomes ORT's <see cref="Float16"/>, <see cref="char"/> becomes
        ///     <see cref="ushort"/>; <c>null</c> for Decimal / Complex.
        /// </summary>
        public static Type ToTensorElementClrType(NPTypeCode code)
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
                case NPTypeCode.Half: return typeof(Float16);
                case NPTypeCode.Single: return typeof(float);
                case NPTypeCode.Double: return typeof(double);
                default: return null;
            }
        }

        /// <summary>
        ///     The NumSharp dtype a <see cref="DenseTensor{T}"/> element type reads back as
        ///     (<see cref="Float16"/> → <see cref="NPTypeCode.Half"/>, <see cref="char"/> → <see cref="NPTypeCode.Char"/>).
        /// </summary>
        /// <exception cref="NotSupportedException"><see cref="BFloat16"/>, or a type ORT tensors do not carry.</exception>
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
            if (typeof(T) == typeof(Float16) || typeof(T) == typeof(Half)) return NPTypeCode.Half;
            if (typeof(T) == typeof(float)) return NPTypeCode.Single;
            if (typeof(T) == typeof(double)) return NPTypeCode.Double;
            if (typeof(T) == typeof(BFloat16))
                throw new NotSupportedException(UnsupportedImportMessage(TensorElementType.BFloat16));
            if (typeof(T) == typeof(Complex))
                throw new NotSupportedException(UnsupportedExportMessage(NPTypeCode.Complex));
            if (typeof(T) == typeof(decimal))
                throw new NotSupportedException(UnsupportedExportMessage(NPTypeCode.Decimal));
            throw new NotSupportedException($"DenseTensor<{typeof(T).Name}> has no NumSharp dtype: ORT tensors carry bool, the eight integer types, Float16, float and double.");
        }

        /// <summary>
        ///     Validate that a caller-chosen <typeparamref name="T"/> is the element type ORT uses for the
        ///     array's dtype (<see cref="ToTensorElementClrType"/>). A mismatch would silently reinterpret
        ///     the bytes as a different dtype, so it is refused up front.
        /// </summary>
        internal static void EnsureElementType<T>(NPTypeCode code, string verb) where T : unmanaged
        {
            Type want = ToTensorElementClrType(code);
            if (want is null)
                throw new NotSupportedException(UnsupportedExportMessage(code));
            if (typeof(T) != want)
                throw new ArgumentException(
                    $"{verb}<{typeof(T).Name}>: a NumSharp {code} array crosses as DenseTensor<{want.Name}> " +
                    $"({(code == NPTypeCode.Half ? "System.Half is ORT's Float16, same 16 bits" : code == NPTypeCode.Char ? "chars are UTF-16 code units" : "the dtype's own C# type")}); " +
                    $"reinterpreting it as {typeof(T).Name} would hand ORT the wrong dtype. Call {verb}<{want.Name}>() or convert with astype() first.");
        }

        internal static string UnsupportedExportMessage(NPTypeCode code)
        {
            switch (code)
            {
                case NPTypeCode.Decimal:
                    return "NumSharp Decimal (16-byte, non-IEEE) has no ONNX tensor element type. The array-producing verbs " +
                           "(AsOrtValue / ToOrtValue / ToDenseTensor) convert it to Double for you (lossy beyond ~16 significant " +
                           "digits); the low-level dtype maps refuse it — convert explicitly with nd.astype(NPTypeCode.Double).";
                case NPTypeCode.Complex:
                    return "ONNX Runtime does not support complex tensors: Complex64/Complex128 exist in TensorElementType but no " +
                           "OrtValue factory, DenseTensor or kernel accepts them. Split the array into real and imaginary planes " +
                           "(np.real / np.imag) and feed those.";
                default:
                    return $"NumSharp dtype {code} has no ONNX tensor element type.";
            }
        }

        internal static string UnsupportedImportMessage(TensorElementType type)
        {
            switch (type)
            {
                case TensorElementType.String:
                    return "ORT string tensors have no NumSharp dtype (NumSharp has no string/object arrays). Read them with " +
                           "OrtValue.GetStringTensorAsArray() / GetStringElement(i).";
                case TensorElementType.BFloat16:
                    return "ORT BFloat16 has no NumSharp dtype (NumSharp has no bfloat16). Cast the tensor to float32 or float16 " +
                           "inside the model graph, or read the raw 16-bit patterns with GetTensorDataAsSpan<BFloat16>().";
                case TensorElementType.Complex64:
                case TensorElementType.Complex128:
                    return $"ORT {type} tensors are not supported by ONNX Runtime itself (no kernel produces them); nothing to read.";
                default:
                    return $"ONNX tensor element type {type} has no NumSharp dtype.";
            }
        }

        // ===================================  shared plumbing  ===================================

        internal const string NotCContiguousMessage =
            "the array is not C-contiguous; ONNX Runtime tensors are dense row-major and carry no strides, so a sliced / " +
            "transposed / negative-stride / broadcast / Fortran-order view cannot be shared. Materialize first — " +
            "np.ascontiguousarray(nd) or nd.copy() — or call ToOrtValue(nd) for an ORT-owned copy.";

        /// <summary>The OrtValue must hold a dense tensor for any NDArray verb to apply.</summary>
        internal static void EnsureTensor(OrtValue value, string verb)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            OnnxValueType kind = value.OnnxType;
            if (kind == OnnxValueType.ONNX_TYPE_TENSOR)
                return;
            throw new NotSupportedException(
                $"{verb} needs a dense tensor OrtValue, but this value is {kind}. Sequences and maps are read element by element " +
                "with OrtValue.GetValueCount() / GetValue(i, allocator) or the ProcessSequence / ProcessMap visitors; " +
                "sparse tensors and optionals have no NumSharp analog.");
        }

        /// <summary>
        ///     A tensor is readable/viewable only when its memory is CPU-addressable: a CUDA / DirectML /
        ///     TensorRT-resident output is not (ORT's own <c>GetTensorDataAsSpan</c> would hand back a device
        ///     pointer and fault on first touch). Host-accessible device memory (<c>CudaPinned</c> — memory type
        ///     <see cref="OrtMemType.CpuOutput"/> / <see cref="OrtMemType.CpuInput"/>) is fine.
        /// </summary>
        internal static void EnsureCpuAccessible(OrtValue value, string verb)
        {
            using OrtMemoryInfo memInfo = value.GetTensorMemoryInfo();
            if (IsCpuAccessible(memInfo))
                return;
            throw new InvalidOperationException(
                $"{verb}: the tensor lives in '{memInfo.Name}' memory (device {memInfo.Id}, {memInfo.GetMemoryType()}), which the CPU " +
                "cannot address. Bind a CPU output (OrtIoBinding with OrtMemoryInfo.DefaultInstance) or copy the tensor to CPU " +
                "first, then convert; ToNDArray / AsNDArray only read CPU-accessible tensors.");
        }

        internal static bool IsCpuAccessible(OrtMemoryInfo memInfo)
        {
            OrtMemType memType = memInfo.GetMemoryType();
            if (memType == OrtMemType.CpuInput || memType == OrtMemType.CpuOutput)
                return true;   // host-accessible memory of a device EP (e.g. CudaPinned)
            if (string.Equals(memInfo.Name, "Cpu", StringComparison.OrdinalIgnoreCase))
                return true;   // the CPU EP's device OR arena allocator
            return memInfo.Equals(OrtMemoryInfo.DefaultInstance);
        }

        /// <summary>ORT's <c>long[]</c> shape → a NumSharp <see cref="Shape"/> (an empty array is the 0-d scalar).</summary>
        internal static Shape ShapeFrom(long[] dims)
        {
            if (dims is null || dims.Length == 0)
                return Shape.Scalar;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] < 0)
                    throw new ArgumentException($"tensor shape ({string.Join(", ", dims)}) has a negative dimension; a concrete tensor cannot carry symbolic dims.");
            return new Shape(dims);
        }

        /// <summary>A NumSharp shape as ORT's row-major <c>long[]</c> (0-d → an empty array).</summary>
        internal static long[] LongDims(Shape shape)
        {
            if (shape.NDim == 0)
                return Array.Empty<long>();
            long[] dims = shape.Dimensions;
            var copy = new long[dims.Length];
            Array.Copy(dims, copy, dims.Length);
            return copy;
        }

        /// <summary>A NumSharp shape as <see cref="DenseTensor{T}"/>'s <c>int[]</c> dimensions.</summary>
        internal static int[] IntDims(Shape shape, string verb)
        {
            if (shape.NDim == 0)
                return Array.Empty<int>();
            long[] dims = shape.Dimensions;
            var result = new int[dims.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] > int.MaxValue)
                    throw new NotSupportedException($"{verb}: DenseTensor dimensions are System.Int32, but axis {i} has {dims[i]} elements.");
                result[i] = (int)dims[i];
            }
            return result;
        }

        /// <summary>Column-major (Fortran) element strides for <paramref name="dims"/>.</summary>
        internal static long[] FortranStrides(long[] dims)
        {
            var strides = new long[dims.Length];
            long acc = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                strides[i] = acc;
                acc *= dims[i];
            }
            return strides;
        }

        /// <summary>
        ///     Wrap foreign (ORT-owned or pinned managed) memory as a NumSharp <see cref="IArraySlice"/>
        ///     whose memory-block Disposer invokes <paramref name="dispose"/> exactly once when the LAST
        ///     NumSharp reference (any view sharing the block) is released — deterministically via
        ///     <see cref="NDArray.Dispose"/> or by the finalizer safety net. The same primitive the pythonnet
        ///     bridge leases Python buffers with.
        /// </summary>
        internal static unsafe IArraySlice WrapExternal(NPTypeCode tc, void* p, long count, Action dispose)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)p, count, dispose));
                case NPTypeCode.Byte: return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)p, count, dispose));
                case NPTypeCode.SByte: return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)p, count, dispose));
                case NPTypeCode.Int16: return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)p, count, dispose));
                case NPTypeCode.UInt16: return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)p, count, dispose));
                case NPTypeCode.Int32: return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)p, count, dispose));
                case NPTypeCode.UInt32: return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)p, count, dispose));
                case NPTypeCode.Int64: return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)p, count, dispose));
                case NPTypeCode.UInt64: return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)p, count, dispose));
                case NPTypeCode.Char: return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)p, count, dispose));
                case NPTypeCode.Half: return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)p, count, dispose));
                case NPTypeCode.Single: return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)p, count, dispose));
                case NPTypeCode.Double: return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)p, count, dispose));
                case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>((Complex*)p, count, dispose));
                default: throw new NotSupportedException(tc.ToString());
            }
        }

        /// <summary>
        ///     Take an ARC reference on the array's buffer and resolve the pointer to its first logical
        ///     element (<c>slice.Address + Shape.Offset × itemsize</c> — right for both a contiguous slice
        ///     that re-seats the storage address and a view that keeps the base address with a non-zero
        ///     offset). The caller owns the reference and must <see cref="IArraySlice.Release"/> it.
        /// </summary>
        internal static unsafe IArraySlice Pin(NDArray source, out byte* data, out long nbytes)
        {
            IArraySlice slice = source.Storage.InternalArray;
            if (!slice.TryAddRef())
                throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");
            Shape shape = source.Shape;
            int itemsize = source.dtypesize;
            nbytes = shape.Size * itemsize;
            data = (byte*)slice.Address + shape.Offset * itemsize;
            return slice;
        }
    }
}
