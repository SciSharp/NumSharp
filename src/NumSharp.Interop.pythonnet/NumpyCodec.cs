using System;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Policies for <see cref="NumpyCodec"/> / <see cref="NDArrayInterop.RegisterCodec()"/>.
    /// </summary>
    public sealed class NumpyCodecOptions
    {
        /// <summary>The defaults: encode as zero-copy view, decode as copy, decode any buffer exporter.</summary>
        public static readonly NumpyCodecOptions Default = new();

        /// <summary>
        ///     <c>true</c> (default): <see cref="NDArray"/> values crossing into Python become zero-copy
        ///     numpy views (<see cref="NDArrayInterop.ToNumpy(NDArray, bool?)"/> — shared mutation, source rooted).
        ///     <c>false</c>: they become independent copies (<see cref="NDArrayInterop.ToNumpyCopy"/>).
        /// </summary>
        public bool EncodeAsView { get; init; } = true;

        /// <summary>
        ///     <c>false</c> (default): <c>PyObject.As&lt;NDArray&gt;()</c> copies
        ///     (<see cref="NDArrayInterop.ToNDArray"/> — safe, owns its memory).
        ///     <c>true</c>: it produces zero-copy views (<see cref="NDArrayInterop.ToNDArrayView"/> with
        ///     <c>allowReadonly:true</c> — shared mutation and shared lifetime; read-only sources decode
        ///     as NON-WRITEABLE views, so guarded writes through them throw instead of corrupting
        ///     immutable Python objects).
        /// </summary>
        public bool DecodeAsView { get; init; }

        /// <summary>
        ///     <c>true</c> (default): besides <c>numpy.ndarray</c> (and subclasses), the built-in buffer
        ///     exporters <c>memoryview</c>, <c>bytes</c>, <c>bytearray</c> and <c>array.array</c> also
        ///     decode to <see cref="NDArray"/>. <c>false</c>: numpy arrays only.
        /// </summary>
        public bool DecodeAnyBuffer { get; init; } = true;
    }

    /// <summary>
    ///     pythonnet auto-marshaling codec: once registered (see
    ///     <see cref="NDArrayInterop.RegisterCodec()"/>), <see cref="NDArray"/> ⇄ numpy conversion happens
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
                return _options.EncodeAsView ? NDArrayInterop.ToNumpy(nd) : NDArrayInterop.ToNumpyCopy(nd);
            }
            catch (NotSupportedException)
            {
                // No numpy dtype (Decimal): let pythonnet wrap the NDArray as a plain CLR object,
                // which is still fully usable from Python through the CLR binding.
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
            if (typeof(T) != typeof(NDArray))
            {
                value = default;
                return false;
            }

            try
            {
                NDArray nd = _options.DecodeAsView
                    ? NDArrayInterop.ToNDArrayView(pyObj, allowReadonly: true)
                    : NDArrayInterop.ToNDArray(pyObj);
                value = (T)(object)nd;
                return true;
            }
            catch
            {
                value = default;
                return false;   // unsupported dtype/layout: let pythonnet report its default error
            }
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
