using System;
using System.Numerics;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Converts between NumSharp <see cref="NDArray"/> and Python objects via Python.NET (pythonnet).
    ///
    ///     <para>Binds only to pythonnet's <see cref="PyObject"/> and Python memory protocols — NO
    ///     Numpy.NET dependency. PEP 3118 exporters (numpy, memoryview, array.array, bytes, PyArrow,
    ///     ...) enter directly; registered <see cref="IPythonArrayAdapter"/> instances expose library
    ///     objects that lack a buffer, notably <c>torch.Tensor</c> and Pandas containers, through
    ///     their official NumPy adapters. Every source then follows the same
    ///     pointer/layout/lifetime pipeline.</para>
    ///
    ///     <para><b>The four verbs</b> (everything else is packaging over these):</para>
    ///     <list type="bullet">
    ///       <item><see cref="ToNumpy(NDArray, bool?)"/> — zero-copy numpy view of NumSharp's buffer (shared
    ///         mutation; source rooted for the lifetime of ALL Python-side views; full strided fidelity
    ///         including slices, transposes, Fortran order, negative strides and read-only broadcasts).</item>
    ///       <item><see cref="ToNumpyCopy(NDArray, bool?)"/> — independent numpy array (no shared memory).</item>
    ///       <item><see cref="ToNDArray(PyObject, bool?)"/> — copy any native buffer or registered-adapter
    ///         source into a fresh C-contiguous NumSharp array (honors logical strides/order).</item>
    ///       <item><see cref="ToNDArrayView(PyObject, bool, bool?)"/> — zero-copy NumSharp view over a native
    ///         buffer or share-preserving adapter result (shared mutation; the exporter is leased for the
    ///         lifetime of ALL NumSharp-side views, including derived slices).</item>
    ///     </list>
    ///
    ///     <para><b>Lifetime safety:</b> exports take their own atomic reference on the NumSharp buffer
    ///     and hand the release to a Python <c>weakref.finalize</c> on the exported array's base object —
    ///     the buffer survives even if every C# reference (including the returned <see cref="PyObject"/>
    ///     wrapper) is disposed or collected, and is released when the last Python-side view dies.
    ///     Imports lease the Python buffer through NumSharp's memory-block reference counting — the lease
    ///     is released when the last NumSharp view over the memory (including derived slices) is disposed
    ///     or collected, never on the finalizer thread's GIL-less back. See
    ///     <see cref="LiveExports"/>/<see cref="LiveImports"/> for observability.</para>
    ///
    ///     <para><b>pythonnet 3.0.1 compatibility:</b> that version's <see cref="PyBuffer"/> is broken for
    ///     shape/strides/format flags (<c>PyBUF.ND/STRIDES</c> throw, <c>PyBUF.FORMATS</c> corrupts memory
    ///     via an <c>LPStr</c> round-trip), so buffer METADATA is read through Python's built-in
    ///     <c>memoryview</c> (correct on every version) and <see cref="PyBuffer"/> is used only with the
    ///     crash-free <c>PyBUF.SIMPLE</c>/<c>PyBUF.WRITABLE</c> flags to obtain the raw pointer.</para>
    ///
    ///     <para><b>Threading:</b> every method acquires the GIL itself (re-entrant, so nesting under an
    ///     outer <see cref="Py.GIL"/> is fine) — unless GIL management is switched off, per call via the
    ///     nullable <c>requireGIL</c> parameter each verb takes, or process-wide via
    ///     <see cref="RequireGIL"/> (the <c>null</c> fallback). With management off the calling thread
    ///     MUST already hold the GIL. The Python engine must be initialized first; conversions
    ///     made in one engine session must not be used after <see cref="PythonEngine.Shutdown"/> (import
    ///     views lose their memory with the interpreter — the shutdown handler releases their leases
    ///     crash-free; exported buffers still referenced by Python are swept right after the engine
    ///     finishes dying — pythonnet's Shutdown runs no Python atexit pass, so their
    ///     <c>weakref.finalize</c> callbacks cannot fire then).</para>
    ///
    ///     <para><b>Note on <see cref="PythonEngine.Shutdown"/> itself:</b> on .NET 8+ pythonnet 3.0.x's
    ///     shutdown crashes in its own state stashing (BinaryFormatter was removed from the runtime).
    ///     That is unrelated to this interop — its shutdown handler completes beforehand — but apps that
    ///     call Shutdown should opt out of stashing first:
    ///     <c>RuntimeData.FormatterType = typeof(NoopFormatter);</c>.</para>
    /// </summary>
    public static partial class NDArrayPythonInterop
    {
        /// <summary>
        ///     Number of NumSharp buffers currently rooted by live Python-side views
        ///     (created by <see cref="ToNumpy(NDArray, bool?)"/> / <see cref="ToMemoryView(NDArray, bool?)"/>,
        ///     released by Python garbage collection or interpreter exit).
        /// </summary>
        public static int LiveExports => PythonRuntimeInterop.LiveExports;

        /// <summary>
        ///     Number of Python buffers currently leased by live NumSharp views
        ///     (created by <see cref="ToNDArrayView(PyObject, bool, bool?)"/>, released when the last
        ///     NumSharp view over the memory is disposed or collected).
        /// </summary>
        public static int LiveImports => PythonRuntimeInterop.LiveImports;

        // ===========================  GIL policy  =============================================

        private static volatile bool _requireGil = true;

        /// <summary>
        ///     Process-wide default for GIL management (default <c>true</c>): whether conversions
        ///     acquire the GIL themselves via <see cref="Py.GIL"/>. Every conversion verb also takes
        ///     a nullable <c>requireGIL</c> parameter — a non-<c>null</c> argument overrides this
        ///     default for that call.
        ///
        ///     <para><b><c>false</c> means the caller owns the GIL.</b> Conversions then run inside a
        ///     shared no-op guard instead of <see cref="Py.GIL"/>, so the calling thread MUST already
        ///     hold the GIL — in practice an enclosing <see cref="Py.GIL"/> block. Skipping the
        ///     per-call <c>PyGILState_Ensure</c>/<c>Release</c> pair (and the <see cref="Py.GILState"/>
        ///     allocation) is a hot-loop micro-optimization and an escape hatch for embeddings where
        ///     <c>PyGILState</c> is problematic; converting GIL-less WITHOUT actually holding the GIL
        ///     is undefined behavior — probed: an immediate access violation at the first C-API call,
        ///     exactly as with any raw C-API misuse.</para>
        ///
        ///     <para><b>Python → .NET callbacks do NOT hold the GIL.</b> A .NET method or delegate body
        ///     invoked FROM Python is the one place that looks safe but is not: pythonnet's method
        ///     binder RELEASES the GIL around the managed body (probed on pythonnet 3.0.5 and 3.1.0,
        ///     both embedded and Python-hosted: <c>PyGILState_Check() == 0</c> inside the body, and a
        ///     GIL-less conversion there dies with an access violation). Keep GIL management ON inside
        ///     such callbacks — only pythonnet's argument/return-value marshaling (where codecs run)
        ///     executes under the GIL, never the body itself.</para>
        ///
        ///     <para><b>Scope:</b> the policy covers the conversion verbs only. The interop's
        ///     background machinery (deferred lease disposal, the engine-shutdown drain) always
        ///     manages the GIL itself — it runs on threads that cannot inherit the caller's GIL.</para>
        /// </summary>
        public static bool RequireGIL
        {
            get => _requireGil;
            set => _requireGil = value;
        }

        /// <summary>The shared guard handed out when GIL management is off — disposal is a no-op,
        /// so the <c>using</c> shape of every conversion body is preserved verbatim.</summary>
        private static readonly IDisposable NoGil = new NoGilGuard();

        private sealed class NoGilGuard : IDisposable
        {
            public void Dispose() { }
        }

        /// <summary>
        ///     <see cref="Py.GIL"/> per the effective policy — <paramref name="requireGIL"/> when
        ///     non-<c>null</c>, else <see cref="RequireGIL"/> — or the shared no-op guard when GIL
        ///     management is off. The policy is read exactly once, here; a concurrent
        ///     <see cref="RequireGIL"/> flip cannot split one conversion across two policies.
        /// </summary>
        internal static IDisposable AcquireGil(bool? requireGIL)
            => (requireGIL ?? _requireGil) ? Py.GIL() : NoGil;

        // ===========================  codec registration  =====================================

        /// <summary>
        ///     Registers <see cref="NumpyCodec"/> with pythonnet's conversion pipeline
        ///     (<c>PyObjectConversions</c>) with default options: <see cref="NDArray"/> arguments and
        ///     return values are auto-encoded as zero-copy numpy views, and
        ///     <c>PyObject.As&lt;NDArray&gt;()</c> decodes numpy arrays, other buffer exporters and
        ///     registered adapter sources through the view-first <see cref="NumpyCodecMode.Auto"/> policy.
        ///     Also registers <see cref="TupleCodec"/> (C# tuples &lt;-&gt; Python tuples — shapes, axes,
        ///     multi-indices; see <see cref="NumpyCodecOptions.ConvertTuples"/>).
        /// </summary>
        /// <returns><c>true</c> if the codec was registered by this call; <c>false</c> if it was already
        /// registered for the current engine session.</returns>
        /// <remarks>
        ///     Registration is per engine session — pythonnet clears all codecs during
        ///     <see cref="PythonEngine.Shutdown"/>, and this method knows to re-register after a
        ///     subsequent <see cref="PythonEngine.Initialize()"/>. Registration is process-global:
        ///     it affects every pythonnet conversion in the process (that is the point).
        /// </remarks>
        public static bool RegisterCodec() => RegisterCodec(NumpyCodecOptions.Default);

        /// <inheritdoc cref="RegisterCodec()"/>
        /// <param name="options">Encode/decode policies (view vs copy, which Python types decode).</param>
        public static bool RegisterCodec(NumpyCodecOptions options)
        {
            if (options is null) throw new ArgumentNullException(nameof(options));
            PythonRuntimeInterop.EnsureEngine();
            if (System.Threading.Interlocked.Exchange(ref PythonRuntimeInterop.CodecRegistered, 1) != 0)
                return false;
            var codec = new NumpyCodec(options);
            PyObjectConversions.RegisterEncoder(codec);
            PyObjectConversions.RegisterDecoder(codec);
            if (options.ConvertTuples)
            {
                PyObjectConversions.RegisterEncoder(TupleCodec.Instance);
                PyObjectConversions.RegisterDecoder(TupleCodec.Instance);
            }

            return true;
        }

        /// <summary>
        ///     Convenience alias for <see cref="PythonArrayAdapterRegistry.Register"/>. Registers a
        ///     library-specific adapter that feeds the existing Python memory bridge. Registration is
        ///     process-wide, thread-safe and idempotent by <see cref="IPythonArrayAdapter.Name"/>. The
        ///     built-in <see cref="TorchPythonArrayAdapter"/> and
        ///     <see cref="PandasPythonArrayAdapter"/> are already registered.
        /// </summary>
        /// <returns><c>true</c> when added; <c>false</c> when an adapter with the same name already exists.</returns>
        public static bool RegisterArrayAdapter(IPythonArrayAdapter adapter)
            => PythonArrayAdapterRegistry.Register(adapter);

        // ===========================  dtype maps  ============================================

        /// <summary>
        ///     NumSharp dtype -&gt; numpy array-interface typestr (<c>"&lt;i4"</c>, <c>"&lt;f8"</c>, <c>"|b1"</c>,
        ///     <c>"|S5"</c>, <c>"&lt;U5"</c>, ...) — the string <c>np.frombuffer</c> / <c>np.dtype</c> build the matching numpy
        ///     dtype from.
        /// </summary>
        /// <param name="dtype">The NumSharp descriptor (an <see cref="NPTypeCode"/> converts implicitly).</param>
        /// <returns>The typestr: for a fixed-width string the descriptor's own <c>dtype.str</c> (<c>"|S5"</c> is five bytes,
        ///     <c>"&lt;U5"</c> five UCS-4 code points).</returns>
        /// <remarks>
        ///     <see cref="NPTypeCode.Char"/> maps to <c>"&lt;u2"</c>: a C# char is a UTF-16 code unit and numpy has no char
        ///     dtype, so a <c>Char</c> array crosses as the code units it holds (numpy's <c>'&lt;U1'</c> is a different dtype,
        ///     one 4-byte UCS-4 code point per element; NumSharp's <c>U1</c> is its counterpart).
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException"><see cref="NPTypeCode.Decimal"/> (no numpy dtype), or a
        ///     <c>StringDType</c> (NEP 55: its text lives in NumPy's allocator, outside any buffer, so there is no typestr to
        ///     build an array from — it crosses as a copy through <see cref="ToNumpyCopy"/> / <see cref="ToNDArray"/>).</exception>
        public static string ToNumpyDtypeStr(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            // The two string classes share the NPTypeCode.String lane; only the descriptor knows the width and the layout.
            switch (dtype.Meta)
            {
                case StringDTypeMeta:
                    return dtype.str;
                case VStringDTypeMeta:
                    throw StringDTypeHasNoBufferForm(dtype.ToString());
            }

            return dtype.GetTypeCode() switch
            {
                NPTypeCode.Boolean => "|b1", NPTypeCode.Byte => "|u1",  NPTypeCode.SByte => "|i1",
                NPTypeCode.Int16 => "<i2",   NPTypeCode.UInt16 => "<u2", NPTypeCode.Int32 => "<i4",
                NPTypeCode.UInt32 => "<u4",  NPTypeCode.Int64 => "<i8",  NPTypeCode.UInt64 => "<u8",
                NPTypeCode.Half => "<f2",    NPTypeCode.Single => "<f4", NPTypeCode.Double => "<f8",
                NPTypeCode.Complex => "<c16",
                NPTypeCode.Char => "<u2",    // C# char is a 2-byte UTF-16 code unit (numpy has no native char)
                NPTypeCode.Decimal => throw new NotSupportedException(
                    "decimal has no numpy dtype (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double)."),
                NPTypeCode code => throw new NotSupportedException(code.ToString())
            };
        }

        /// <summary>
        ///     numpy dtype string / typestr (<c>"&lt;i4"</c>, <c>"|b1"</c>, <c>"&lt;f8"</c>, <c>"=f4"</c>, <c>"|S5"</c>,
        ///     <c>"&lt;U5"</c>, ...) -&gt; NumSharp descriptor. Accepts the little-endian ('&lt;'), native ('='), and
        ///     byte-order-irrelevant ('|') markers, or none.
        /// </summary>
        /// <param name="dtypeStr">The typestr, as <c>__array_interface__['typestr']</c> or <c>dtype.str</c> give it.</param>
        /// <returns>The NumSharp descriptor. It converts implicitly to <see cref="NPTypeCode"/>, so code written against the
        ///     earlier <see cref="NPTypeCode"/>-returning form keeps compiling; a fixed-width string needs the descriptor
        ///     (its width).</returns>
        /// <remarks>
        ///     A string typestr maps to the NumSharp string dtype of the same class and width (<c>"&lt;U1"</c> is NumSharp's
        ///     <c>U1</c>, not <see cref="NPTypeCode.Char"/>: one 4-byte code point per element, where a <c>Char</c> is a
        ///     2-byte UTF-16 unit — converting to <c>Char</c> is an explicit cast).
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="dtypeStr"/> is null or empty.</exception>
        /// <exception cref="NotSupportedException">The typestr has no NumSharp dtype a VIEW could use: big-endian data
        ///     ('&gt;', NumSharp buffers are native-endian — <see cref="ToNDArray"/> byte-swaps it into a copy), complex64
        ///     (widened on copy), a <c>StringDType</c> or an object array (both copy-only: see <see cref="ToNDArray"/>), or a
        ///     dtype NumSharp lacks (datetime, structured, void).</exception>
        public static DType FromNumpyDtypeStr(string dtypeStr)
        {
            if (string.IsNullOrEmpty(dtypeStr))
                throw new ArgumentNullException(nameof(dtypeStr));
            // StringDType's typestr is its str() ('StringDType()', 'StringDType(na_object=None)'): no buffer to view.
            if (dtypeStr.StartsWith("StringDType(", StringComparison.Ordinal))
                throw StringDTypeHasNoBufferForm(dtypeStr);
            char order = dtypeStr[0];
            string code = "<=|>".IndexOf(order) >= 0 ? dtypeStr.Substring(1) : dtypeStr;
            if (TryParseTextTypestr(code, out char kind, out long chars))
            {
                // A byte string has no byte order ('|S5'; numpy also accepts '<S5' / '>S5' for it). UCS-4 code points
                // do: a big-endian '>U5' cannot be a view over a native NumSharp buffer.
                if (kind == 'U' && order == '>')
                    throw new NotSupportedException(
                        $"big-endian UCS-4 text '{dtypeStr}' cannot be shared with a native-endian NumSharp buffer; ToNDArray byte-swaps it into a native U{chars} copy.");
                return TextDescr(kind, chars);
            }

            if (order == '>')
                throw new NotSupportedException($"big-endian dtype '{dtypeStr}' cannot be shared with a native-endian NumSharp buffer. Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).");
            return code switch
            {
                "b1" => DType.Boolean,
                "u1" => DType.Byte,  "i1" => DType.SByte,
                "i2" => DType.Int16, "u2" => DType.UInt16,
                "i4" => DType.Int32, "u4" => DType.UInt32,
                "i8" => DType.Int64, "u8" => DType.UInt64,
                "f2" => DType.Half,  "f4" => DType.Single, "f8" => DType.Double,
                "c16" => DType.Complex,
                "c8" => throw new NotSupportedException(
                    $"numpy dtype '{dtypeStr}' (complex64) has no exact NumSharp dtype. ToNDArray widens it to Complex (complex128) as a copy; a zero-copy view is impossible."),
                "O" => throw ObjectArrayHasNoView(dtypeStr),
                _ => throw new NotSupportedException($"numpy dtype '{dtypeStr}' has no NumSharp dtype.")
            };
        }

        /// <summary>
        ///     NumSharp dtype -&gt; PEP 3118 struct format code (<c>'?'</c>, <c>'b'</c>, <c>'B'</c>, <c>'h'</c>, ...,
        ///     <c>'Zd'</c>; <c>'5s'</c> for <c>S5</c>, <c>'5w'</c> for <c>U5</c> — the formats numpy's own buffers report).
        /// </summary>
        /// <param name="dtype">The NumSharp descriptor (an <see cref="NPTypeCode"/> converts implicitly).</param>
        /// <returns>The format code.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="NotSupportedException"><see cref="NPTypeCode.Decimal"/> (no PEP 3118 format), or a
        ///     <c>StringDType</c> (no buffer form; numpy refuses it in a buffer too).</exception>
        public static string ToBufferFormat(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            switch (dtype.Meta)
            {
                case StringDTypeMeta text:
                    // PEP 3118 counts characters: 's' is one byte, 'w' one UCS-4 code point.
                    return text.IsUnicode ? (dtype.itemsize / 4).ToString(System.Globalization.CultureInfo.InvariantCulture) + "w"
                                          : dtype.itemsize.ToString(System.Globalization.CultureInfo.InvariantCulture) + "s";
                case VStringDTypeMeta:
                    throw StringDTypeHasNoBufferForm(dtype.ToString());
            }

            return dtype.GetTypeCode() switch
            {
                NPTypeCode.Boolean => "?",  NPTypeCode.Byte => "B",   NPTypeCode.SByte => "b",
                NPTypeCode.Int16 => "h",    NPTypeCode.UInt16 => "H", NPTypeCode.Int32 => "i",
                NPTypeCode.UInt32 => "I",   NPTypeCode.Int64 => "q",  NPTypeCode.UInt64 => "Q",
                NPTypeCode.Half => "e",     NPTypeCode.Single => "f", NPTypeCode.Double => "d",
                NPTypeCode.Complex => "Zd",
                NPTypeCode.Char => "H",     // UTF-16 code unit == unsigned 2-byte
                NPTypeCode.Decimal => throw new NotSupportedException(
                    "decimal has no PEP 3118 format (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double)."),
                NPTypeCode code => throw new NotSupportedException(code.ToString())
            };
        }

        /// <summary>
        ///     PEP 3118 struct format code (+ itemsize to disambiguate 'l'/'i' across platforms) -&gt; NumSharp
        ///     descriptor. An empty format is raw bytes.
        /// </summary>
        /// <param name="format">The format (<c>memoryview.format</c>): an optional byte-order marker, then the code.</param>
        /// <param name="itemSize">The element size in bytes (<c>memoryview.itemsize</c>).</param>
        /// <returns>The NumSharp descriptor (converts implicitly to <see cref="NPTypeCode"/>; a string format needs the
        ///     descriptor's width).</returns>
        /// <remarks>
        ///     <para>Text maps the way numpy reads the same buffer: <c>'5s'</c> is <c>S5</c>, <c>'5w'</c> (and <c>'w'</c>,
        ///     <c>'1w'</c>) is <c>U5</c> / <c>U1</c> — numpy's <c>'&lt;U1'</c> is NumSharp's <c>U1</c>, not
        ///     <see cref="NPTypeCode.Char"/>. A wchar_t unit <c>'u'</c> maps by its width: 2 bytes (Windows) is a UTF-16 code
        ///     unit, exactly <see cref="NPTypeCode.Char"/> (numpy itself cannot import <c>'u'</c>); 4 bytes (linux/macOS) is a
        ///     UCS-4 code point, <c>U1</c>.</para>
        ///     <para>Big-endian data ('&gt;' / '!' markers) is refused for multi-byte types — NumSharp buffers are
        ///     native-endian and a silent reinterpretation would byte-swap every value (<see cref="ToNDArray"/> byte-swaps
        ///     into a copy instead).</para>
        /// </remarks>
        /// <exception cref="NotSupportedException">The format has no NumSharp dtype a VIEW could use: big-endian
        ///     multi-byte data, complex64 (<c>'Zf'</c>, widened on copy), an object array (<c>'O'</c>, copy-only for an
        ///     array of <c>str</c>), an extended long double, a text count that disagrees with the itemsize, or a format
        ///     NumSharp has no dtype for.</exception>
        public static DType FromBufferFormat(string format, long itemSize)
        {
            if (string.IsNullOrEmpty(format))
                return DType.Byte;
            char c0 = format[0];
            int i = "<>=@!".IndexOf(c0) >= 0 ? 1 : 0;
            string code = format.Substring(i);
            bool bigEndian = c0 == '>' || c0 == '!';

            // Counted text formats first ('5s', '5w', 's', 'w'): a byte string has no byte order, UCS-4 code points do.
            if (TryParseCountedText(code, out char textCode, out long count))
            {
                long width = textCode == 'w' ? count * 4 : count;
                if (width != itemSize)
                    throw new NotSupportedException(
                        $"buffer format '{format}' names {count} character(s) ({width} bytes) but the exporter reports itemsize {itemSize}; no NumSharp dtype matches.");
                if (textCode == 'w' && bigEndian)
                    throw new NotSupportedException(
                        $"big-endian UCS-4 buffer format '{format}' cannot be mapped onto a native-endian NumSharp buffer; ToNDArray byte-swaps it into a native U{count} copy.");
                return TextDescr(textCode == 'w' ? 'U' : 'S', count);
            }

            if (bigEndian && code != "b" && code != "B" && code != "c" && code != "?")
                throw new NotSupportedException(
                    $"big-endian buffer format '{format}' cannot be mapped onto a native-endian NumSharp buffer. Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).");
            switch (code)
            {
                case "?": return DType.Boolean;
                case "b": return DType.SByte;
                case "c": case "B": return DType.Byte;
                case "h": return DType.Int16;
                case "H": return DType.UInt16;
                case "i": case "l": return itemSize == 8 ? DType.Int64 : DType.Int32;
                case "I": case "L": return itemSize == 8 ? DType.UInt64 : DType.UInt32;
                case "n": case "q": return DType.Int64;
                case "N": case "Q": return DType.UInt64;
                case "e": return DType.Half;
                case "f": return DType.Single;
                case "d": return DType.Double;
                case "g":                                     // C long double (np.longdouble exports 'g' at EVERY width)
                    if (itemSize == 8) return DType.Double;   // MSVC long double IS IEEE double — bit-exact, viewable
                    throw new NotSupportedException(
                        $"buffer format 'g' (itemsize {itemSize}) is an extended-precision long double with no NumSharp dtype. Convert first: arr.astype(np.float64).");
                case "u":                                     // wchar_t text unit (array.array('u'), ctypes.c_wchar) — width is the platform's wchar_t
                    if (itemSize == 2) return DType.Char;     // UTF-16 code unit == System.Char, bit-exact
                    if (itemSize == 1) return DType.Byte;     // degenerate single-byte text unit: raw bytes
                    if (itemSize == 4) return TextDescr('U', 1); // 4-byte wchar_t (linux/macOS) is one UCS-4 code point
                    throw new NotSupportedException($"buffer format '{format}' (itemsize {itemSize}) has no NumSharp dtype.");
                case "O":
                    throw ObjectArrayHasNoView(format);
                case "Zd": return DType.Complex;              // complex128
                case "Zf": throw new NotSupportedException(
                    "buffer format 'Zf' (complex64) has no exact NumSharp dtype. ToNDArray widens it to Complex (complex128) as a copy; a zero-copy view is impossible.");
                default:
                    throw new NotSupportedException($"buffer format '{format}' (itemsize {itemSize}) has no NumSharp dtype.");
            }
        }

        /// <summary>
        ///     Parses a counted PEP 3118 text code — <c>'s'</c> / <c>'w'</c> preceded by an optional decimal character count
        ///     (<c>'5s'</c>, <c>'12w'</c>, <c>'w'</c> is one character).
        /// </summary>
        /// <param name="code">The format without its byte-order marker.</param>
        /// <param name="textCode">Receives <c>'s'</c> or <c>'w'</c>.</param>
        /// <param name="count">Receives the character count.</param>
        /// <returns>False for any other code (including a count NumSharp cannot hold).</returns>
        private static bool TryParseCountedText(string code, out char textCode, out long count)
        {
            textCode = '\0';
            count = 0;
            if (code.Length == 0)
                return false;
            char last = code[code.Length - 1];
            if (last != 's' && last != 'w')
                return false;
            if (code.Length == 1)
            {
                textCode = last;
                count = 1;
                return true;
            }
            // Only digits may precede the code: anything else ('Zs', '2Ws') is not a counted text code.
            if (!long.TryParse(code.AsSpan(0, code.Length - 1), System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out count))
                return false;
            textCode = last;
            return true;
        }

        /// <summary>Parses a numpy string typestr body (<c>"S5"</c>, <c>"U12"</c>; the byte-order marker already removed).</summary>
        /// <param name="code">The typestr without its byte-order marker.</param>
        /// <param name="kind">Receives <c>'S'</c> or <c>'U'</c>.</param>
        /// <param name="chars">Receives the character count.</param>
        /// <returns>False for any other typestr.</returns>
        private static bool TryParseTextTypestr(string code, out char kind, out long chars)
        {
            kind = '\0';
            chars = 0;
            if (code.Length < 2 || (code[0] != 'S' && code[0] != 'U'))
                return false;
            if (!long.TryParse(code.AsSpan(1), System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out chars))
                return false;
            kind = code[0];
            return true;
        }

        /// <summary>The native NumSharp string descriptor of a class and width (<c>S5</c>, <c>U3</c>; 0 is the unsized form).</summary>
        /// <param name="kind"><c>'S'</c> (bytes) or <c>'U'</c> (UCS-4 code points).</param>
        /// <param name="chars">The character count.</param>
        /// <returns>The descriptor.</returns>
        /// <exception cref="TypeError">The width does not fit numpy's C <c>int</c> item size (numpy's own error text).</exception>
        internal static DType TextDescr(char kind, long chars)
            => global::NumSharp.np.dtype((kind == 'S' ? "S" : "U") + chars.ToString(System.Globalization.CultureInfo.InvariantCulture));

        /// <summary>The refusal of every buffer-level map for a <c>StringDType</c>, with the copy route that works.</summary>
        /// <param name="dtype">The descriptor's text (<c>StringDType()</c>, <c>StringDType(na_object=None)</c>).</param>
        /// <returns>The exception to throw.</returns>
        internal static NotSupportedException StringDTypeHasNoBufferForm(string dtype)
            => new NotSupportedException(
                $"{dtype} has no buffer form (numpy: cannot include dtype 'numpy.dtypes.StringDType' in a buffer): NEP 55 keeps each " +
                "element's text in the array's own allocator, outside the array buffer, so a StringDType array never shares memory " +
                "across the bridge. It crosses as an exact copy: ToNDArray / ToNumpyCopy, or the codec's Auto / Copy mode.");

        /// <summary>The refusal of a view over a numpy object array, with the copy route that works for text.</summary>
        /// <param name="dtype">The typestr or buffer format that named the object dtype.</param>
        /// <returns>The exception to throw.</returns>
        internal static NotSupportedException ObjectArrayHasNoView(string dtype)
            => new NotSupportedException(
                $"numpy object dtype '{dtype}' has no NumSharp dtype: its elements are Python objects, so no NumSharp view can " +
                "share them. An object array whose every element is a str crosses as a StringDType copy (ToNDArray, or the codec's " +
                "Auto / Copy mode); any other object array is refused.");

        // ---- memoryview metadata helpers (avoid pythonnet 3.0.1's broken PyBuffer flags) --------
        // The attribute names are session-cached PyStrings (see PythonRuntimeInterop): GetAttr(PyObject)
        // skips the per-call UTF-8 marshal + unicode allocation of the string-based overload.

        internal static string GetStr(PyObject o, PyObject attr) { using var a = o.GetAttr(attr); return a.As<string>(); }
        internal static long GetLong(PyObject o, PyObject attr) { using var a = o.GetAttr(attr); return a.As<long>(); }
        internal static bool GetBool(PyObject o, PyObject attr) { using var a = o.GetAttr(attr); return a.As<bool>(); }

        internal static long[] GetLongTuple(PyObject o, PyObject attr)
        {
            using PyObject t = o.GetAttr(attr);
            if (t.IsNone())
                return null;
            using var tup = PyTuple.AsTuple(t);
            int n = (int)tup.Length();
            var values = new long[n];
            for (int i = 0; i < n; i++)
            {
                using var e = tup[i];
                values[i] = e.As<long>();
            }

            return values;
        }

        // ---- external-memory wrapping ----------------------------------------------------------

        /// <summary>
        ///     Wrap external (Python-owned) memory as a NumSharp <see cref="IArraySlice"/> whose
        ///     memory-block Disposer invokes <paramref name="dispose"/> exactly once when the LAST
        ///     NumSharp reference (any view sharing the block) is released — deterministically via
        ///     <see cref="NDArray.Dispose"/> or by the finalizer safety net.
        /// </summary>
        /// <param name="descr">The element descriptor: a numeric dtype, or a sized native <c>S</c> / <c>U</c> descriptor
        ///     (its item size decides how many bytes <paramref name="count"/> elements span).</param>
        /// <param name="p">The first byte of the external memory.</param>
        /// <param name="count">The number of ELEMENTS the memory holds.</param>
        /// <param name="dispose">The release action of the external lease, run once by the last NumSharp reference.</param>
        /// <returns>The slice (its block owns the release action).</returns>
        /// <exception cref="NotSupportedException">A descriptor no external memory can back (Decimal, <c>StringDType</c>).</exception>
        internal static unsafe IArraySlice WrapExternal(DType descr, void* p, long count, Action dispose)
        {
            // A fixed-width string element is itemsize raw bytes (S: one byte per character, U: one native-endian uint32
            // code point per character); the string slice only says how to count them, so the byte block it wraps carries
            // the lease exactly like a numeric block.
            if (descr.Meta is StringDTypeMeta)
                return new StringArraySlice(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)p, count * descr.itemsize, dispose)), descr);
            switch (descr.GetTypeCode())
            {
                case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)p, count, dispose));
                case NPTypeCode.Byte:    return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)p, count, dispose));
                case NPTypeCode.SByte:   return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)p, count, dispose));
                case NPTypeCode.Int16:   return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)p, count, dispose));
                case NPTypeCode.UInt16:  return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)p, count, dispose));
                case NPTypeCode.Int32:   return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)p, count, dispose));
                case NPTypeCode.UInt32:  return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)p, count, dispose));
                case NPTypeCode.Int64:   return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)p, count, dispose));
                case NPTypeCode.UInt64:  return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)p, count, dispose));
                case NPTypeCode.Char:    return new ArraySlice<char>(new UnmanagedMemoryBlock<char>((char*)p, count, dispose));
                case NPTypeCode.Half:    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)p, count, dispose));
                case NPTypeCode.Single:  return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)p, count, dispose));
                case NPTypeCode.Double:  return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)p, count, dispose));
                case NPTypeCode.Complex: return new ArraySlice<Complex>(new UnmanagedMemoryBlock<Complex>((Complex*)p, count, dispose));
                default: throw new NotSupportedException($"{descr.ToString(true)} cannot be backed by external memory.");
            }
        }
    }
}
