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
    ///     <para><b>The verbs</b> follow the house convention: <c>As…</c> shares memory when it can (a zero-copy view),
    ///     <c>To…</c> always copies. An export <c>As…</c> verb takes an <see cref="OnnxCodecMode"/>: the default,
    ///     <see cref="OnnxCodecMode.Auto"/>, shares a C-contiguous array and copies any other layout;
    ///     <see cref="OnnxCodecMode.View"/> throws instead of copying; <see cref="OnnxCodecMode.Copy"/> always copies.</para>
    ///     <list type="table">
    ///         <item><term><see cref="AsOrtValue"/></term><description>NumSharp → ORT, an <see cref="OrtValue"/> over a NumSharp buffer behind an <see cref="OrtTensor"/> handle: the NDArray's own buffer when ORT can read it in place, a handle-owned copy otherwise (the buffer is ARC-rooted for the handle's lifetime)</description></item>
    ///         <item><term><see cref="ToOrtValue"/></term><description>NumSharp → ORT, an ORT-allocated copy (any layout, no handle, no lifetime coupling)</description></item>
    ///         <item><term><see cref="AsDenseTensor{T}"/> / <see cref="ToDenseTensor{T}"/></term><description>the legacy <see cref="DenseTensor{T}"/> twins</description></item>
    ///         <item><term><see cref="ToNDArray(OrtValue)"/></term><description>ORT → NumSharp, a fresh owning C-contiguous copy (the safe default)</description></item>
    ///         <item><term><see cref="AsNDArray(OrtValue, bool, OnnxCodecMode)"/></term><description>ORT → NumSharp, a zero-copy view over the tensor's buffer (CPU tensors only), optionally taking ownership of the OrtValue; a counted alias of the NumSharp buffer for a value this package made</description></item>
    ///     </list>
    ///
    ///     <para><b>Dtype map</b> — 12 NumSharp dtypes cross zero-copy (bool, the 8 integers, Half ↔ Float16
    ///     as a bit-identical reinterpret, float, double); <see cref="NPTypeCode.Char"/> exports as
    ///     <see cref="TensorElementType.UInt16"/> (UTF-16 code units); <see cref="NPTypeCode.Decimal"/> has
    ///     no ONNX type and is converted to Double by the array-producing verbs (the low-level maps refuse
    ///     it); text (NumSharp's fixed-width <c>U</c> / <c>S</c> arrays) crosses as an ORT string tensor, always
    ///     as a copy (<c>NDArrayOnnxInterop.Text.cs</c>); <see cref="NPTypeCode.Complex"/> and BFloat16 are
    ///     genuine mutual gaps and are refused with a message naming the workaround. See
    ///     <see cref="ToTensorElementType"/> / <see cref="FromTensorElementType"/>.</para>
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
        ///     Number of live import leases — NumSharp views (<see cref="AsNDArray(OrtValue, bool, OnnxCodecMode)"/> /
        ///     <see cref="AsNDArray{T}(DenseTensor{T}, OnnxCodecMode)"/>, the arrays of the non-tensor readers) currently holding ORT (or pinned managed)
        ///     memory. Released when the LAST NumSharp view over the memory — derived slices included — is
        ///     disposed or collected.
        /// </summary>
        public static int LiveImports => Volatile.Read(ref _liveImports);

        /// <summary>Counts one new export handle in <see cref="LiveExports"/>.</summary>
        internal static void ExportOpened() => Interlocked.Increment(ref _liveExports);

        /// <summary>Counts one export handle gone (disposed or finalized) in <see cref="LiveExports"/>.</summary>
        internal static void ExportClosed() => Interlocked.Decrement(ref _liveExports);

        /// <summary>Counts one new import lease in <see cref="LiveImports"/>.</summary>
        internal static void ImportOpened() => Interlocked.Increment(ref _liveImports);

        /// <summary>Counts one import lease ended (released or disarmed) in <see cref="LiveImports"/>.</summary>
        internal static void ImportClosed() => Interlocked.Decrement(ref _liveImports);

        private static long _importPressureBytes;

        /// <summary>
        ///     The bytes of ORT memory currently reported to the GC as memory pressure (<c>GC.AddMemoryPressure</c>) by
        ///     import views that OWN their <see cref="OrtValue"/>. A diagnostic for the tests: the GC offers no way to read
        ///     back what was reported, and the rule "only an owning view reports, once, and takes it back when it dies"
        ///     has no other observable effect. A steady non-zero value with no live owning view is a pressure leak (the GC
        ///     then collects more often for the life of the process).
        /// </summary>
        internal static long ImportPressureBytes => Interlocked.Read(ref _importPressureBytes);

        /// <summary>Records memory pressure added (positive) or removed (negative) by an import lease.</summary>
        /// <param name="delta">The byte count just passed to <c>GC.AddMemoryPressure</c>, or minus the count just passed to <c>GC.RemoveMemoryPressure</c>.</param>
        internal static void ImportPressureChanged(long delta) => Interlocked.Add(ref _importPressureBytes, delta);

        // ===================================  the dtype map  ===================================

        /// <summary>
        ///     The ONNX tensor element type a NumSharp dtype crosses as. Zero-copy for the 12 numeric
        ///     dtypes incl. <see cref="NPTypeCode.Half"/> → <see cref="TensorElementType.Float16"/>
        ///     (ORT's <c>Float16</c> is a blittable 16-bit IEEE struct, binary-identical to
        ///     <see cref="System.Half"/>) and <see cref="NPTypeCode.Char"/> → <see cref="TensorElementType.UInt16"/>
        ///     (UTF-16 code units, the rule the pythonnet bridge also uses).
        /// </summary>
        /// <remarks>
        ///     The map reads the descriptor's STORAGE class (<c>dtype.Meta.TypeCode</c>), so a non-native descriptor
        ///     (<c>&gt;i4</c>) maps like its native twin: an array of it holds native values. A descriptor whose class has no
        ///     storage in NumSharp yet (datetime64 / timedelta64) has no element type either; it is refused here with this
        ///     package's own message. It used to surface NumSharp's "has no NumSharp storage yet" error, also from the
        ///     <c>Try</c> form, which must never throw for a dtype it does not map.
        /// </remarks>
        /// <param name="dtype">The NumSharp dtype (a <see cref="DType"/>, or anything that converts to one: an <see cref="NPTypeCode"/>, a <see cref="Type"/>, a dtype string).</param>
        /// <returns>The ONNX tensor element type.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     <see cref="NPTypeCode.Decimal"/> (no ONNX type — convert to Double first; the array-producing
        ///     verbs do so automatically), <see cref="NPTypeCode.Complex"/> (ORT accepts no complex tensors), or a
        ///     datetime64 / timedelta64 descriptor (ONNX has no date or duration tensors).
        /// </exception>
        public static TensorElementType ToTensorElementType(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            if (!TryToTensorElementType(dtype, out TensorElementType type))
                throw new NotSupportedException(dtype.Meta.TypeCode == NPTypeCode.Empty ? NoStorageDtypeMessage(dtype) : UnsupportedExportMessage(dtype.Meta.TypeCode));
            return type;
        }

        /// <summary>
        ///     <see cref="ToTensorElementType"/> without the throw: <c>false</c> for every dtype that has no ONNX tensor
        ///     element type — Decimal, Complex, a datetime64 / timedelta64 descriptor — and for <c>null</c>.
        /// </summary>
        /// <param name="dtype">The NumSharp dtype, or null.</param>
        /// <param name="type">The ONNX element type when the result is <c>true</c>; <c>default</c> otherwise.</param>
        /// <returns><c>true</c> when the dtype crosses to ONNX Runtime.</returns>
        public static bool TryToTensorElementType(DType dtype, out TensorElementType type)
        {
            // Meta.TypeCode, not GetTypeCode(): the latter THROWS for a descriptor whose class has no storage yet
            // (datetime64 / timedelta64), and a Try method must answer false for those. It is NPTypeCode.Empty there,
            // which the default case refuses.
            switch (dtype?.Meta.TypeCode)
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
                case NPTypeCode.String: type = TensorElementType.String; return true;   // both fixed-width classes, U and S (a copy, never a view)
                default: type = default; return false;                                   // Decimal, Complex, a class without storage
            }
        }

        /// <summary>The refusal for a descriptor whose class has no NumSharp storage (datetime64, timedelta64).</summary>
        /// <param name="dtype">The descriptor.</param>
        /// <returns>The message: what the dtype is, why it cannot cross, and what to feed instead.</returns>
        internal static string NoStorageDtypeMessage(DType dtype) =>
            $"NumSharp dtype {dtype.name} has no ONNX tensor element type: ONNX has no date, time or duration tensors " +
            "(and NumSharp has no arrays of this dtype yet). Feed the values as an integer count, for example int64.";

        /// <summary>
        ///     The NumSharp dtype an ONNX tensor element type reads back as. Directional on purpose:
        ///     <see cref="TensorElementType.UInt16"/> comes back as <see cref="NPTypeCode.UInt16"/>, never
        ///     <see cref="NPTypeCode.Char"/> (an ORT tensor carries no "these are characters" bit).
        /// </summary>
        /// <remarks>
        ///     <see cref="TensorElementType.String"/> reads back as <see cref="NPTypeCode.String"/>, the storage lane of
        ///     NumSharp's fixed-width string arrays. That code names neither the class (<c>U</c> / <c>S</c>) nor the width,
        ///     so it cannot be converted to a <see cref="DType"/>; <see cref="TryFromTensorElementType"/> gives the
        ///     descriptor (the unsized <c>U</c>).
        /// </remarks>
        /// <param name="type">The ONNX tensor element type.</param>
        /// <returns>The NumSharp storage type code.</returns>
        /// <exception cref="NotSupportedException">
        ///     <see cref="TensorElementType.BFloat16"/> (NumSharp has no bfloat16), Complex64/Complex128.
        /// </exception>
        public static NPTypeCode FromTensorElementType(TensorElementType type)
        {
            if (!TryFromTensorElementType(type, out DType dtype))
                throw new NotSupportedException(UnsupportedImportMessage(type));
            return dtype.GetTypeCode();
        }

        /// <summary>
        ///     <see cref="FromTensorElementType"/> without the throw, and with the full descriptor.
        /// </summary>
        /// <param name="type">The ONNX tensor element type.</param>
        /// <param name="dtype">
        ///     The NumSharp dtype when the result is <c>true</c>, else <c>null</c>. For
        ///     <see cref="TensorElementType.String"/> it is the UNSIZED Unicode descriptor (<c>np.dtype("U")</c>): an ORT
        ///     string tensor has no width, and an array read from one takes the width of its longest string.
        /// </param>
        /// <returns><c>true</c> when NumSharp has a dtype for the element type.</returns>
        public static bool TryFromTensorElementType(TensorElementType type, out DType dtype)
        {
            switch (type)
            {
                case TensorElementType.String: dtype = TextDType; return true;
                case TensorElementType.Bool: dtype = NPTypeCode.Boolean; return true;
                case TensorElementType.UInt8: dtype = NPTypeCode.Byte; return true;
                case TensorElementType.Int8: dtype = NPTypeCode.SByte; return true;
                case TensorElementType.Int16: dtype = NPTypeCode.Int16; return true;
                case TensorElementType.UInt16: dtype = NPTypeCode.UInt16; return true;
                case TensorElementType.Int32: dtype = NPTypeCode.Int32; return true;
                case TensorElementType.UInt32: dtype = NPTypeCode.UInt32; return true;
                case TensorElementType.Int64: dtype = NPTypeCode.Int64; return true;
                case TensorElementType.UInt64: dtype = NPTypeCode.UInt64; return true;
                case TensorElementType.Float16: dtype = NPTypeCode.Half; return true;
                case TensorElementType.Float: dtype = NPTypeCode.Single; return true;
                case TensorElementType.Double: dtype = NPTypeCode.Double; return true;
                default: dtype = null; return false;                                     // BFloat16, Complex64/128, DataTypeMax
            }
        }

        /// <summary>
        ///     The C# element type ORT's <see cref="DenseTensor{T}"/> uses for a NumSharp dtype —
        ///     <see cref="System.Half"/> becomes ORT's <see cref="Float16"/>, <see cref="char"/> becomes
        ///     <see cref="ushort"/>; <c>null</c> for a dtype with no ONNX element type (Decimal, Complex, a
        ///     datetime64 / timedelta64 descriptor).
        /// </summary>
        /// <param name="dtype">The NumSharp dtype.</param>
        /// <returns>The element type of the tensor the dtype crosses as, or <c>null</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        public static Type ToTensorElementClrType(DType dtype)
        {
            // Meta.TypeCode: a descriptor without storage answers null like every other unmapped dtype (GetTypeCode()
            // threw NumSharp's "no storage yet" error for it).
            NPTypeCode code = (dtype ?? throw new ArgumentNullException(nameof(dtype))).Meta.TypeCode;
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
                case NPTypeCode.String: return typeof(string);   // ORT's DenseTensor<string>; the unmanaged-generic dense verbs cannot build one
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
                    return "an ORT string tensor reads back as a NumSharp text array (fixed-width U) through ToNDArray, or as a " +
                           "string[] through ortValue.ReadStringTensor() (or OrtValue.GetStringTensorAsArray() / GetStringElement(i)).";
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

        /// <summary>
        ///     Refuses the value of an export handle that has been disposed, before anything calls into ONNX Runtime with
        ///     it. ONNX Runtime does not guard a released value: a native call on one is an access violation that ends the
        ///     process, where a .NET caller expects <see cref="ObjectDisposedException"/>.
        /// </summary>
        /// <remarks>
        ///     Only values THIS package made are known to it. A value of ONNX Runtime's own that the caller disposed cannot
        ///     be recognized through the public API, and using it fails as it does anywhere in ONNX Runtime.
        /// </remarks>
        /// <param name="value">The value an import verb was given (not null).</param>
        /// <param name="verb">The public verb, named in the message.</param>
        /// <exception cref="ObjectDisposedException">The value belongs to an <see cref="OrtTensor"/> handle that has been disposed.</exception>
        internal static void EnsureNotOfADisposedHandle(OrtValue value, string verb)
        {
            if (OrtTensor.IsValueOfDisposedHandle(value))
                throw new ObjectDisposedException(nameof(OrtTensor),
                    $"{verb}: this OrtValue belongs to an OrtTensor handle that has been disposed. ONNX Runtime has released the value, and the " +
                    "NumSharp buffer behind it is no longer pinned. Read or view the value before the handle is disposed.");
        }

        /// <summary>The OrtValue must hold a dense tensor for any NDArray verb to apply.</summary>
        /// <param name="value">The value an import verb was given.</param>
        /// <param name="verb">The public verb, named in the refusal.</param>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
        /// <exception cref="NotSupportedException">The value is a sequence, a map, a sparse tensor or an optional; the message names the reader that takes it.</exception>
        internal static void EnsureTensor(OrtValue value, string verb)
        {
            if (value is null)
                throw new ArgumentNullException(nameof(value));
            OnnxValueType kind = value.OnnxType;
            if (kind == OnnxValueType.ONNX_TYPE_TENSOR)
                return;
            throw new NotSupportedException(
                $"{verb} needs a dense tensor OrtValue, but this value is {kind}. A sequence of tensors is read with ToNDArrays(); " +
                "a single map with ToMap(); a sequence of maps (the scikit-learn ZipMap shape) with ToMaps(); " +
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

        /// <summary>
        ///     Can the CPU read (and write) the memory a tensor lives in? Decided by ORT's own description of the memory,
        ///     in the order of how direct the evidence is.
        /// </summary>
        /// <remarks>
        ///     <para>1. The allocator's memory type is <see cref="OrtMemType.CpuInput"/> / <see cref="OrtMemType.CpuOutput"/>:
        ///     host memory of a device provider (how the CUDA provider registers <c>CudaPinned</c>).</para>
        ///     <para>2. ORT says the device memory is <c>HOST_ACCESSIBLE</c> (<see cref="HostAccessibleProbe"/>, ONNX Runtime
        ///     versions that have <c>OrtMemoryInfo.GetDeviceMemoryType</c>): shared or pinned memory the CPU addresses,
        ///     whatever the device — <c>CudaPinned</c>, <c>HipPinned</c>, <c>QnnHtpShared</c>.</para>
        ///     <para>3. The allocator NAME is one of ORT's CPU-device allocators (<see cref="CpuAllocatorNames"/>). The
        ///     device memory type cannot decide these: it is <c>DEFAULT</c> for the CPU's own memory and for a GPU's alike,
        ///     and the managed API of the supported versions does not expose the device type. On an ONNX Runtime too old for
        ///     rule 2 the host-accessible names (<see cref="HostAccessibleAllocatorNames"/>) are accepted here too.</para>
        ///     <para>4. The memory info equals ORT's default CPU instance.</para>
        ///     <para>Everything else (a GPU / NPU provider's default memory: <c>Cuda</c>, <c>DML</c>, <c>Hip</c>,
        ///     <c>OpenVINO_GPU</c>, …) is refused: reading it through a CPU pointer faults.</para>
        /// </remarks>
        /// <param name="memInfo">The tensor's memory info.</param>
        /// <returns><c>true</c> when a CPU pointer into the memory is valid.</returns>
        internal static bool IsCpuAccessible(OrtMemoryInfo memInfo)
        {
            OrtMemType memType = memInfo.GetMemoryType();
            if (memType == OrtMemType.CpuInput || memType == OrtMemType.CpuOutput)
                return true;   // host-accessible memory of a device EP (e.g. CudaPinned)
            if (HostAccessibleProbe.Value is { } isHostAccessible)
            {
                if (isHostAccessible(memInfo))
                    return true;   // ORT's own answer: shared / pinned memory the CPU addresses
            }
            string name = memInfo.Name;
            if (IsOneOf(name, CpuAllocatorNames))
                return true;   // a CPU device's own memory (the CPU EP's device or arena allocator, the 4 KB-aligned one)
            if (HostAccessibleProbe.Value is null && IsOneOf(name, HostAccessibleAllocatorNames))
                return true;   // an ORT without GetDeviceMemoryType: the names ORT itself maps to host-accessible memory
            return memInfo.Equals(OrtMemoryInfo.DefaultInstance);
        }

        /// <summary>
        ///     ORT's allocator names for a CPU device's own memory (<c>onnxruntime::CPU</c>, <c>onnxruntime::CPU_ALIGNED_4K</c>
        ///     in <c>core/framework/allocator.h</c>; <c>CreateMemoryInfo</c> maps both to <c>OrtDevice::CPU</c> with default
        ///     memory). <c>CpuAligned4K</c> was refused before: only the exact name <c>Cpu</c> was accepted.
        /// </summary>
        private static readonly string[] CpuAllocatorNames = { "Cpu", "CpuAligned4K" };

        /// <summary>
        ///     ORT's allocator names for memory a device shares with the host (<c>CreateMemoryInfo</c> maps each to
        ///     <c>OrtDevice::MemType::HOST_ACCESSIBLE</c>). Used only when the loaded ONNX Runtime cannot be asked
        ///     (<see cref="HostAccessibleProbe"/> is null); a newer ORT answers for itself, including for names added later.
        /// </summary>
        private static readonly string[] HostAccessibleAllocatorNames = { "CudaPinned", "HipPinned", "QnnHtpShared" };

        /// <summary>Is <paramref name="name"/> one of <paramref name="names"/>, ignoring case (ORT compares its names exactly; a caller's spelling may differ in case)?</summary>
        /// <param name="name">An allocator name.</param>
        /// <param name="names">The names to look for.</param>
        /// <returns><c>true</c> on a match.</returns>
        private static bool IsOneOf(string name, string[] names)
        {
            foreach (string candidate in names)
                if (string.Equals(name, candidate, StringComparison.OrdinalIgnoreCase))
                    return true;
            return false;
        }

        /// <summary>
        ///     "Is this memory <c>HOST_ACCESSIBLE</c>?", asked of ONNX Runtime itself through
        ///     <c>OrtMemoryInfo.GetDeviceMemoryType()</c> — or <c>null</c> when the loaded ONNX Runtime does not have that
        ///     member. The package compiles against ORT 1.16.0, which lacks both the method and its enum, so the member is
        ///     found by reflection once and compiled into a delegate (a reflection call per tensor would cost about a
        ///     microsecond on every import).
        /// </summary>
        /// <remarks>
        ///     The enum value is looked up by NAME (<c>HOST_ACCESSIBLE</c>), not assumed to be 5: if a future ORT renames or
        ///     removes it, the probe is null and the name lists decide, instead of a wrong number deciding silently.
        ///     Building the delegate cannot be allowed to fail a caller: any reflection or compilation error leaves the
        ///     probe null.
        /// </remarks>
        private static readonly Lazy<Func<OrtMemoryInfo, bool>> HostAccessibleProbe = new Lazy<Func<OrtMemoryInfo, bool>>(BuildHostAccessibleProbe);

        /// <summary>Builds <see cref="HostAccessibleProbe"/>'s delegate, or returns <c>null</c> when the loaded ONNX Runtime has no <c>GetDeviceMemoryType</c>.</summary>
        /// <returns>The compiled test, or <c>null</c>.</returns>
        private static Func<OrtMemoryInfo, bool> BuildHostAccessibleProbe()
        {
            try
            {
                System.Reflection.MethodInfo method = typeof(OrtMemoryInfo).GetMethod("GetDeviceMemoryType", Type.EmptyTypes);
                if (method is null || !method.ReturnType.IsEnum || !Enum.IsDefined(method.ReturnType, "HOST_ACCESSIBLE"))
                    return null;
                object hostAccessible = Enum.Parse(method.ReturnType, "HOST_ACCESSIBLE");
                var info = System.Linq.Expressions.Expression.Parameter(typeof(OrtMemoryInfo), "info");
                var test = System.Linq.Expressions.Expression.Equal(
                    System.Linq.Expressions.Expression.Call(info, method),
                    System.Linq.Expressions.Expression.Constant(hostAccessible, method.ReturnType));
                return System.Linq.Expressions.Expression.Lambda<Func<OrtMemoryInfo, bool>>(test, info).Compile();
            }
            catch (Exception)
            {
                return null;   // an unexpected ORT shape: fall back to the name lists rather than fail every import
            }
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
        ///     Wrap a window of a NumSharp buffer — viewed as <paramref name="tc"/> elements — as an ALIAS whose reference
        ///     counting is <paramref name="owner"/>'s own (<c>ArraySlice.WrapShared</c>): an <see cref="NDArray"/> over the
        ///     returned slice holds a counted reference on the owner's buffer and keeps it reachable, exactly as an ordinary
        ///     view does, and the owner's last release frees the buffer once. The element type may differ from the owner's
        ///     (Float16 viewed as Half, a Char buffer viewed as UInt16) — the reinterpreting-alias case the primitive exists for.
        /// </summary>
        /// <param name="tc">The dtype the alias carries.</param>
        /// <param name="p">The first aliased element, inside <paramref name="owner"/>'s buffer.</param>
        /// <param name="count">The number of <paramref name="tc"/> elements aliased.</param>
        /// <param name="owner">The buffer's slice, to which every reference-counting call of the alias is forwarded.</param>
        /// <returns>The alias; it holds no reference until an <see cref="NDArray"/> is built over it.</returns>
        /// <exception cref="NotSupportedException"><paramref name="tc"/> is not a storage dtype.</exception>
        internal static unsafe IArraySlice WrapShared(NPTypeCode tc, void* p, long count, IArraySlice owner)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return ArraySlice.WrapShared<bool>(p, count, owner);
                case NPTypeCode.Byte: return ArraySlice.WrapShared<byte>(p, count, owner);
                case NPTypeCode.SByte: return ArraySlice.WrapShared<sbyte>(p, count, owner);
                case NPTypeCode.Int16: return ArraySlice.WrapShared<short>(p, count, owner);
                case NPTypeCode.UInt16: return ArraySlice.WrapShared<ushort>(p, count, owner);
                case NPTypeCode.Int32: return ArraySlice.WrapShared<int>(p, count, owner);
                case NPTypeCode.UInt32: return ArraySlice.WrapShared<uint>(p, count, owner);
                case NPTypeCode.Int64: return ArraySlice.WrapShared<long>(p, count, owner);
                case NPTypeCode.UInt64: return ArraySlice.WrapShared<ulong>(p, count, owner);
                case NPTypeCode.Char: return ArraySlice.WrapShared<char>(p, count, owner);
                case NPTypeCode.Half: return ArraySlice.WrapShared<Half>(p, count, owner);
                case NPTypeCode.Single: return ArraySlice.WrapShared<float>(p, count, owner);
                case NPTypeCode.Double: return ArraySlice.WrapShared<double>(p, count, owner);
                case NPTypeCode.Complex: return ArraySlice.WrapShared<Complex>(p, count, owner);
                default: throw new NotSupportedException(tc.ToString());
            }
        }

        /// <summary>
        ///     Take an ARC reference on the array's buffer and resolve the pointer to its first logical
        ///     element (<c>slice.Address + Shape.Offset × itemsize</c> — right for both a contiguous slice
        ///     that re-seats the storage address and a view that keeps the base address with a non-zero
        ///     offset). The caller owns the reference and must <see cref="IArraySlice.Release"/> it.
        /// </summary>
        /// <param name="source">The array whose buffer is pinned.</param>
        /// <param name="data">The address of the array's first logical element.</param>
        /// <param name="nbytes">The byte length of the array's elements (<c>size × itemsize</c>), meaningful as one run only for a contiguous array.</param>
        /// <returns>The pinned buffer; release it exactly once.</returns>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static unsafe IArraySlice Pin(NDArray source, out byte* data, out long nbytes)
        {
            IArraySlice slice = Pin(source);
            Shape shape = source.Shape;
            int itemsize = source.dtypesize;
            nbytes = shape.Size * itemsize;
            data = (byte*)slice.Address + shape.Offset * itemsize;
            return slice;
        }

        /// <summary>
        ///     Take an ARC reference on the array's buffer, refusing a buffer that has already been released. This is
        ///     the ONLY liveness check NumSharp offers: <c>TryAddRef</c> fails once the last reference is gone, while
        ///     every read (<c>astype</c>, <c>copy()</c>, a raw pointer) goes ahead regardless and, once the pool has
        ///     handed the memory to another array, returns that array's values. So every verb takes this reference
        ///     before its first read of a caller's array, and holds it until the read is done.
        /// </summary>
        /// <param name="source">The array whose buffer is pinned.</param>
        /// <returns>The pinned buffer; release it exactly once (<see cref="IArraySlice.Release"/>).</returns>
        /// <exception cref="ObjectDisposedException">The array's buffer has already been released.</exception>
        internal static IArraySlice Pin(NDArray source)
        {
            IArraySlice slice = source.Storage.InternalArray;
            if (!slice.TryAddRef())
                throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");
            return slice;
        }

        /// <summary>
        ///     The dtype a NumSharp array is fed to ONNX Runtime as: <see cref="NPTypeCode.Decimal"/>, which has no ONNX
        ///     element type, becomes <see cref="NPTypeCode.Double"/>; every other dtype is fed as itself (Char and Half
        ///     cross bit-for-bit as UInt16 / Float16, see <see cref="ToTensorElementType"/>). Lets a verb decide its
        ///     dtype and size refusals from the source's dtype and shape alone, before it converts or reads anything.
        /// </summary>
        /// <param name="code">The source array's dtype.</param>
        /// <returns>The dtype the conversion verbs produce for it.</returns>
        internal static NPTypeCode FeedTypeCode(NPTypeCode code) => code == NPTypeCode.Decimal ? NPTypeCode.Double : code;

        /// <summary>
        ///     <c>source.astype(code)</c> while holding a reference on the source's buffer — how every verb converts a
        ///     caller's array. Use it instead of a bare <c>astype</c>, which reads the buffer without asking whether it
        ///     is still alive.
        /// </summary>
        /// <remarks>
        ///     The verbs used to convert first and pin afterwards, so the pin (<see cref="Pin(NDArray)"/>, the only
        ///     liveness check) ran on the fresh conversion and always succeeded: a source whose buffer had been released
        ///     was converted from whatever the pool had put there, and the export carried another array's values.
        ///     Holding the reference across the conversion also keeps another thread's dispose from releasing the
        ///     buffer while it is read.
        /// </remarks>
        /// <param name="source">The array to convert; any layout.</param>
        /// <param name="code">The dtype to convert to.</param>
        /// <returns>
        ///     A new array the caller owns and must dispose, laid out as <c>astype</c> lays it out (NumPy's order 'K':
        ///     a transposed source gives a Fortran-ordered result, so callers needing C order still check).
        /// </returns>
        /// <exception cref="ObjectDisposedException">The source's buffer has already been released.</exception>
        internal static NDArray ConvertPinned(NDArray source, NPTypeCode code)
        {
            IArraySlice pin = Pin(source);
            try
            {
                return source.astype(code);
            }
            finally
            {
                pin.Release();
                GC.KeepAlive(source);
            }
        }

        /// <summary>
        ///     <c>source.copy()</c> — a C-order copy, the densifying step for a view ONNX Runtime cannot share — while
        ///     holding a reference on the source's buffer, for the reason <see cref="ConvertPinned"/> gives: the copy
        ///     reads the buffer without checking that it is still alive.
        /// </summary>
        /// <param name="source">The array to copy; any layout (0-d stays 0-d, unlike <c>np.ascontiguousarray</c>).</param>
        /// <returns>A new C-contiguous array the caller owns and must dispose.</returns>
        /// <exception cref="ObjectDisposedException">The source's buffer has already been released.</exception>
        internal static NDArray CopyPinned(NDArray source)
        {
            IArraySlice pin = Pin(source);
            try
            {
                return source.copy();
            }
            finally
            {
                pin.Release();
                GC.KeepAlive(source);
            }
        }
    }
}
