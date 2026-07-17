using System;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop
{
    /// <summary>
    ///     Converts between NumSharp <see cref="NDArray"/> and Python objects via Python.NET (pythonnet).
    ///
    ///     <para>Binds only to pythonnet's <see cref="PyObject"/> and Python's PEP 3118 buffer protocol —
    ///     NO Numpy.NET dependency, so it works with any numpy, any Python, and with ANY buffer-protocol
    ///     exporter (numpy, memoryview, array.array, bytes, PIL, torch, ...), not just numpy.</para>
    ///
    ///     <para><b>The four verbs</b> (everything else is packaging over these):</para>
    ///     <list type="bullet">
    ///       <item><see cref="ToNumpy(NDArray)"/> — zero-copy numpy view of NumSharp's buffer (shared
    ///         mutation; source rooted for the lifetime of ALL Python-side views; full strided fidelity
    ///         including slices, transposes, Fortran order, negative strides and read-only broadcasts).</item>
    ///       <item><see cref="ToNumpyCopy(NDArray)"/> — independent numpy array (no shared memory).</item>
    ///       <item><see cref="ToNDArray(PyObject)"/> — copy ANY PEP 3118 buffer object into a fresh
    ///         C-contiguous NumSharp array (honors strides / Fortran order; numpy-agnostic).</item>
    ///       <item><see cref="ToNDArrayView(PyObject, bool)"/> — zero-copy NumSharp view over Python
    ///         memory (shared mutation; the exporter is leased for the lifetime of ALL NumSharp-side
    ///         views, including derived slices).</item>
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
    ///     outer <see cref="Py.GIL"/> is fine). The Python engine must be initialized first; conversions
    ///     made in one engine session must not be used after <see cref="PythonEngine.Shutdown"/> (import
    ///     views lose their memory with the interpreter — the shutdown handler releases their leases
    ///     crash-free; exported buffers are released by Python's own finalization pass).</para>
    ///
    ///     <para><b>Note on <see cref="PythonEngine.Shutdown"/> itself:</b> on .NET 8+ pythonnet 3.0.x's
    ///     shutdown crashes in its own state stashing (BinaryFormatter was removed from the runtime).
    ///     That is unrelated to this interop — its shutdown handler completes beforehand — but apps that
    ///     call Shutdown should opt out of stashing first:
    ///     <c>RuntimeData.FormatterType = typeof(NoopFormatter);</c>.</para>
    /// </summary>
    public static partial class PythonConvert
    {
        /// <summary>
        ///     Number of NumSharp buffers currently rooted by live Python-side views
        ///     (created by <see cref="ToNumpy(NDArray)"/> / <see cref="ToMemoryView(NDArray)"/>,
        ///     released by Python garbage collection or interpreter exit).
        /// </summary>
        public static int LiveExports => InteropRuntime.LiveExports;

        /// <summary>
        ///     Number of Python buffers currently leased by live NumSharp views
        ///     (created by <see cref="ToNDArrayView(PyObject, bool)"/>, released when the last
        ///     NumSharp view over the memory is disposed or collected).
        /// </summary>
        public static int LiveImports => InteropRuntime.LiveImports;

        // ===========================  codec registration  =====================================

        /// <summary>
        ///     Registers <see cref="NumpyCodec"/> with pythonnet's conversion pipeline
        ///     (<c>PyObjectConversions</c>) with default options: <see cref="NDArray"/> arguments and
        ///     return values are auto-encoded as zero-copy numpy views, and <c>PyObject.As&lt;NDArray&gt;()</c>
        ///     decodes numpy arrays (and other buffer exporters) as copies.
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
            InteropRuntime.EnsureEngine();
            if (System.Threading.Interlocked.Exchange(ref InteropRuntime.CodecRegistered, 1) != 0)
                return false;
            var codec = new NumpyCodec(options);
            PyObjectConversions.RegisterEncoder(codec);
            PyObjectConversions.RegisterDecoder(codec);
            return true;
        }

        // ===========================  dtype maps  ============================================

        /// <summary>NumSharp dtype -&gt; numpy dtype string ("&lt;i4", "&lt;f8", "|b1", ...).</summary>
        /// <remarks><see cref="NPTypeCode.Char"/> maps to "&lt;u2" (a C# char is a UTF-16 code unit;
        /// numpy has no native char dtype). <see cref="NPTypeCode.Decimal"/> has no numpy equivalent.</remarks>
        public static string ToNumpyDtypeStr(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => "|b1", NPTypeCode.Byte => "|u1",  NPTypeCode.SByte => "|i1",
            NPTypeCode.Int16 => "<i2",   NPTypeCode.UInt16 => "<u2", NPTypeCode.Int32 => "<i4",
            NPTypeCode.UInt32 => "<u4",  NPTypeCode.Int64 => "<i8",  NPTypeCode.UInt64 => "<u8",
            NPTypeCode.Half => "<f2",    NPTypeCode.Single => "<f4", NPTypeCode.Double => "<f8",
            NPTypeCode.Complex => "<c16",
            NPTypeCode.Char => "<u2",    // C# char is a 2-byte UTF-16 code unit (numpy has no native char)
            NPTypeCode.Decimal => throw new NotSupportedException(
                "decimal has no numpy dtype (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double)."),
            _ => throw new NotSupportedException(tc.ToString())
        };

        /// <summary>
        ///     numpy dtype string / typestr ("&lt;i4", "|b1", "&lt;f8", "=f4", ...) -&gt; NumSharp dtype.
        ///     Accepts the little-endian ('&lt;'), native ('='), and byte-order-irrelevant ('|') markers,
        ///     or none. Big-endian ('&gt;') data is rejected — NumSharp buffers are native-endian.
        /// </summary>
        public static NPTypeCode FromNumpyDtypeStr(string dtypeStr)
        {
            if (string.IsNullOrEmpty(dtypeStr))
                throw new ArgumentNullException(nameof(dtypeStr));
            char order = dtypeStr[0];
            string code = "<=|>".IndexOf(order) >= 0 ? dtypeStr.Substring(1) : dtypeStr;
            if (order == '>')
                throw new NotSupportedException($"big-endian dtype '{dtypeStr}' cannot be shared with a native-endian NumSharp buffer. Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).");
            return code switch
            {
                "b1" => NPTypeCode.Boolean,
                "u1" => NPTypeCode.Byte,  "i1" => NPTypeCode.SByte,
                "i2" => NPTypeCode.Int16, "u2" => NPTypeCode.UInt16,
                "i4" => NPTypeCode.Int32, "u4" => NPTypeCode.UInt32,
                "i8" => NPTypeCode.Int64, "u8" => NPTypeCode.UInt64,
                "f2" => NPTypeCode.Half,  "f4" => NPTypeCode.Single, "f8" => NPTypeCode.Double,
                "c16" => NPTypeCode.Complex,
                _ => throw new NotSupportedException($"numpy dtype '{dtypeStr}' has no NumSharp dtype.")
            };
        }

        /// <summary>NumSharp dtype -&gt; PEP 3118 struct format code ('?', 'b', 'B', 'h', ..., 'Zd').</summary>
        public static string ToBufferFormat(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => "?",  NPTypeCode.Byte => "B",   NPTypeCode.SByte => "b",
            NPTypeCode.Int16 => "h",    NPTypeCode.UInt16 => "H", NPTypeCode.Int32 => "i",
            NPTypeCode.UInt32 => "I",   NPTypeCode.Int64 => "q",  NPTypeCode.UInt64 => "Q",
            NPTypeCode.Half => "e",     NPTypeCode.Single => "f", NPTypeCode.Double => "d",
            NPTypeCode.Complex => "Zd",
            NPTypeCode.Char => "H",     // UTF-16 code unit == unsigned 2-byte
            NPTypeCode.Decimal => throw new NotSupportedException(
                "decimal has no PEP 3118 format (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double)."),
            _ => throw new NotSupportedException(tc.ToString())
        };

        /// <summary>
        ///     PEP 3118 struct format code (+ itemsize to disambiguate 'l'/'i' across platforms) -&gt; NumSharp
        ///     dtype. An empty format is raw bytes. Big-endian data ('&gt;' / '!' markers) is rejected for
        ///     multi-byte types — NumSharp buffers are native-endian and a silent reinterpretation would
        ///     byte-swap every value.
        /// </summary>
        public static NPTypeCode FromBufferFormat(string format, long itemSize)
        {
            if (string.IsNullOrEmpty(format))
                return NPTypeCode.Byte;
            char c0 = format[0];
            int i = "<>=@!".IndexOf(c0) >= 0 ? 1 : 0;
            string code = format.Substring(i);
            if ((c0 == '>' || c0 == '!') && code != "b" && code != "B" && code != "c" && code != "?")
                throw new NotSupportedException(
                    $"big-endian buffer format '{format}' cannot be mapped onto a native-endian NumSharp buffer. Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).");
            switch (code)
            {
                case "?": return NPTypeCode.Boolean;
                case "b": return NPTypeCode.SByte;
                case "c": case "B": return NPTypeCode.Byte;
                case "h": return NPTypeCode.Int16;
                case "H": return NPTypeCode.UInt16;
                case "i": case "l": return itemSize == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32;
                case "I": case "L": return itemSize == 8 ? NPTypeCode.UInt64 : NPTypeCode.UInt32;
                case "n": case "q": return NPTypeCode.Int64;
                case "N": case "Q": return NPTypeCode.UInt64;
                case "e": return NPTypeCode.Half;
                case "f": return NPTypeCode.Single;
                case "d": return NPTypeCode.Double;
                case "Zd": return NPTypeCode.Complex;         // complex128
                case "Zf": throw new NotSupportedException(
                    "buffer format 'Zf' (complex64) has no exact NumSharp dtype. ToNDArray widens it to Complex (complex128) as a copy; a zero-copy view is impossible.");
                default:
                    throw new NotSupportedException($"buffer format '{format}' (itemsize {itemSize}) has no NumSharp dtype.");
            }
        }

        // ---- memoryview metadata helpers (avoid pythonnet 3.0.1's broken PyBuffer flags) --------

        internal static string GetStr(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<string>(); }
        internal static long GetLong(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<long>(); }
        internal static bool GetBool(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<bool>(); }

        internal static long[] GetLongTuple(PyObject o, string attr)
        {
            using PyObject t = o.GetAttr(attr);
            if (t.IsNone())
                return null;
            using var tup = PyTuple.AsTuple(t);
            int n = (int)tup.Length();
            var values = new long[n];
            for (int i = 0; i < n; i++) { using var e = tup[i]; values[i] = e.As<long>(); }
            return values;
        }

        // ---- external-memory wrapping ----------------------------------------------------------

        /// <summary>
        ///     Wrap external (Python-owned) memory as a NumSharp <see cref="IArraySlice"/> whose
        ///     memory-block Disposer invokes <paramref name="dispose"/> exactly once when the LAST
        ///     NumSharp reference (any view sharing the block) is released — deterministically via
        ///     <see cref="NDArray.Dispose"/> or by the finalizer safety net.
        /// </summary>
        internal static unsafe IArraySlice WrapExternal(NPTypeCode tc, void* p, long count, Action dispose)
        {
            switch (tc)
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
                default: throw new NotSupportedException(tc.ToString());
            }
        }
    }
}
