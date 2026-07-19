using System;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     How <see cref="NumpyCodec"/> materializes a conversion in one direction — encode
    ///     (<see cref="NDArray"/> → Python) or decode (Python → <see cref="NDArray"/>). One enum serves
    ///     both directions.
    /// </summary>
    public enum NumpyCodecMode
    {
        /// <summary>
        ///     <b>Zero-copy VIEW when the dtype and layout permit it; an independent COPY only when a
        ///     view is impossible.</b> Shared memory whenever it is achievable, a safe copy as the
        ///     fallback — never a blanket copy.
        ///
        ///     <para>On <b>decode</b> the fallback is real: complex64, big-endian, and non-contiguous
        ///     non-numpy exporters have no zero-copy NumSharp representation and transparently become
        ///     copies, while contiguous / strided-numpy sources stay zero-copy views (read-only sources
        ///     become NON-WRITEABLE views). On <b>encode</b> a view and a copy have identical dtype
        ///     coverage — both need a numpy-expressible dtype — so Auto always yields a view, and the
        ///     only unrepresentable dtype (<see cref="NPTypeCode.Decimal"/>) falls through to pythonnet's
        ///     CLR wrapping either way.</para>
        /// </summary>
        Auto = 0,

        /// <summary>
        ///     <b>Always a zero-copy VIEW</b> (shared memory, live mutation both ways). A conversion that
        ///     cannot be expressed as a view is DECLINED (the encode/decode fails) rather than silently
        ///     copying — for callers that depend on the shared-memory contract and want a loud failure
        ///     otherwise. Read-only decode sources become NON-WRITEABLE views.
        /// </summary>
        View = 1,

        /// <summary>
        ///     <b>Always an independent COPY</b> — no shared memory, no lifetime coupling, no Py_buffer
        ///     lock on the source, total dtype/layout coverage. The most predictable mode.
        /// </summary>
        Copy = 2,
    }

    /// <summary>
    ///     Policies for <see cref="NumpyCodec"/> / <see cref="NDArrayPythonInterop.RegisterCodec()"/>.
    /// </summary>
    public sealed class NumpyCodecOptions
    {
        /// <summary>The defaults: <see cref="NumpyCodecMode.Auto"/> both ways, decode any buffer exporter.</summary>
        public static readonly NumpyCodecOptions Default = new();

        /// <summary>
        ///     How <see cref="NDArray"/> values crossing INTO Python are materialized. Default
        ///     <see cref="NumpyCodecMode.Auto"/> — a zero-copy numpy view
        ///     (<see cref="NDArrayPythonInterop.ToNumpy(NDArray, bool?)"/> — shared mutation, source
        ///     rooted). On encode a view and a copy have the same dtype coverage, so
        ///     <see cref="NumpyCodecMode.Auto"/> and <see cref="NumpyCodecMode.View"/> behave identically
        ///     and only <see cref="NumpyCodecMode.Copy"/> forces
        ///     <see cref="NDArrayPythonInterop.ToNumpyCopy"/>.
        /// </summary>
        public NumpyCodecMode EncodeMode { get; init; } = NumpyCodecMode.Auto;

        /// <summary>
        ///     How Python objects crossing INTO .NET (<c>PyObject.As&lt;NDArray&gt;()</c>) are
        ///     materialized. Default <see cref="NumpyCodecMode.Auto"/> — a zero-copy view
        ///     (<see cref="NDArrayPythonInterop.ToNDArrayView"/> with <c>allowReadonly:true</c>) when the
        ///     source's dtype/layout permits, otherwise an independent copy
        ///     (<see cref="NDArrayPythonInterop.ToNDArray"/>). <see cref="NumpyCodecMode.View"/> declines
        ///     the decode when a view is impossible; <see cref="NumpyCodecMode.Copy"/> always copies.
        ///
        ///     <para><b>Auto/View share memory:</b> mutations flow both ways, read-only sources decode as
        ///     NON-WRITEABLE views (guarded writes through them throw instead of corrupting immutable
        ///     Python objects), and a live view holds a Py_buffer lock on the source (a <c>bytearray</c>
        ///     cannot be resized while the view lives). Choose <see cref="NumpyCodecMode.Copy"/> for a
        ///     detached snapshot that never touches the Python object.</para>
        /// </summary>
        public NumpyCodecMode DecodeMode { get; init; } = NumpyCodecMode.Auto;

        /// <summary>
        ///     <c>true</c> (default): besides <c>numpy.ndarray</c> (and subclasses), the built-in buffer
        ///     exporters <c>memoryview</c>, <c>bytes</c>, <c>bytearray</c> and <c>array.array</c> also
        ///     decode to <see cref="NDArray"/>. <c>false</c>: numpy arrays only.
        /// </summary>
        public bool DecodeAnyBuffer { get; init; } = true;
    }

    /// <summary>
    ///     pythonnet auto-marshaling codec: once registered (see
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/>), <see cref="NDArray"/> ⇄ numpy conversion happens
    ///     automatically at every pythonnet boundary — <c>nd.ToPython()</c>, <c>scope.Set("x", nd)</c>,
    ///     passing an <see cref="NDArray"/> to a Python callable, and <c>pyObj.As&lt;NDArray&gt;()</c> on
    ///     the way back — with no explicit conversion calls.
    ///
    ///     <para>Encoding falls back to pythonnet's default CLR-object wrapping for arrays it cannot
    ///     express as numpy (<see cref="NPTypeCode.Decimal"/>), instead of failing the conversion.</para>
    /// </summary>
    public sealed class NumpyCodec : IPyObjectEncoder, IPyObjectDecoder
    {
        private readonly NumpyCodecOptions _options;

        /// <summary>Create a codec with <see cref="NumpyCodecOptions.Default"/>.</summary>
        public NumpyCodec() : this(NumpyCodecOptions.Default) { }

        /// <summary>Create a codec with explicit <paramref name="options"/>.</summary>
        public NumpyCodec(NumpyCodecOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
        }

        // ---- CLR -> Python -----------------------------------------------------------------------

        /// <inheritdoc/>
        public bool CanEncode(Type type) => typeof(NDArray).IsAssignableFrom(type);

        /// <inheritdoc/>
        public PyObject TryEncode(object value)
        {
            var nd = (NDArray)value;
            try
            {
                switch (_options.EncodeMode)
                {
                    case NumpyCodecMode.Copy:
                        return NDArrayPythonInterop.ToNumpyCopy(nd);
                    case NumpyCodecMode.View:
                        return NDArrayPythonInterop.ToNumpy(nd);
                    default: // Auto: view-first, copy-fallback. On encode both need a numpy dtype, so the
                             // fallback is a no-op today (Decimal fails both); kept for a uniform contract
                             // and any future dtype-coverage divergence between the two paths.
                        try { return NDArrayPythonInterop.ToNumpy(nd); }
                        catch (NotSupportedException) { return NDArrayPythonInterop.ToNumpyCopy(nd); }
                }
            }
            catch (NotSupportedException)
            {
                // No numpy dtype (Decimal) in ANY mode: let pythonnet wrap the NDArray as a plain CLR
                // object, which is still fully usable from Python through the CLR binding.
                return null;
            }
        }

        // ---- Python -> CLR -----------------------------------------------------------------------

        /// <inheritdoc/>
        public bool CanDecode(PyType objectType, Type targetType)
        {
            if (targetType != typeof(NDArray))
                return false;

            try
            {
                string name = objectType.Name;   // tp_name: "numpy.ndarray", "memoryview", "array.array", ...
                if (name == "numpy.ndarray")
                    return true;
                if (_options.DecodeAnyBuffer && (name == "memoryview" || name == "bytes" || name == "bytearray" || name == "array.array"))
                    return true;
                return IsNdarraySubclass(objectType);
            }
            catch
            {
                return false;   // CanDecode must never throw into pythonnet's conversion pipeline
            }
        }

        /// <inheritdoc/>
        public bool TryDecode<T>(PyObject pyObj, out T value)
        {
            value = default;
            if (typeof(T) != typeof(NDArray))
                return false;

            NDArray nd = _options.DecodeMode switch
            {
                NumpyCodecMode.Copy => TryDecodeCopy(pyObj),
                NumpyCodecMode.View => TryDecodeView(pyObj),
                _                    => TryDecodeView(pyObj) ?? TryDecodeCopy(pyObj),   // Auto
            };

            if (nd is null)
                return false;   // no view was possible in View mode, or an unsupported dtype/layout in any mode
            value = (T)(object)nd;
            return true;
        }

        /// <summary>
        ///     A zero-copy view over the source, or <c>null</c> when the source has no zero-copy NumSharp
        ///     representation (complex64 / big-endian / non-contiguous non-numpy / stride not an element
        ///     multiple). Any failure is treated as "not viewable" — the Auto path then copies, the View
        ///     path declines the decode.
        /// </summary>
        private static NDArray TryDecodeView(PyObject pyObj)
        {
            try { return NDArrayPythonInterop.ToNDArrayView(pyObj, allowReadonly: true); }
            catch { return null; }
        }

        /// <summary>An independent copy, or <c>null</c> if the source cannot be copied either (no NumSharp dtype).</summary>
        private static NDArray TryDecodeCopy(PyObject pyObj)
        {
            try { return NDArrayPythonInterop.ToNDArray(pyObj); }
            catch { return null; }
        }

        /// <summary>Walks <c>__mro__</c> so numpy.matrix / numpy.memmap / user ndarray subclasses decode too.</summary>
        private static bool IsNdarraySubclass(PyType objectType)
        {
            using PyObject mro = objectType.GetAttr("__mro__");
            using var tup = PyTuple.AsTuple(mro);
            long n = tup.Length();
            for (int i = 0; i < n; i++)
            {
                using PyObject t = tup[i];
                using PyObject module = t.GetAttr("__module__");
                using PyObject name = t.GetAttr("__name__");
                if (module.As<string>() == "numpy" && name.As<string>() == "ndarray")
                    return true;
            }

            return false;
        }
    }
}
