using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Static extension members that hang NumSharp registration off pythonnet's <see cref="Py"/>
    ///     class, so it reads beside <see cref="Py.Import(string)"/> / <see cref="Py.GIL()"/>:
    ///     <code>
    ///     using NumSharp.Interop.PythonNet;   // brings the extension into scope
    ///     using Python.Runtime;
    ///
    ///     PythonEngine.Initialize();
    ///     Py.RegisterNumSharpCodecs();         // == NDArrayPythonInterop.RegisterCodec()
    ///     </code>
    ///
    ///     <para>This is a thin, discoverable alias — nothing more. Every call forwards verbatim to
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/>, so the semantics are identical: one
    ///     registration per engine session, process-global, idempotent (the second call in a session
    ///     returns <c>false</c>), and re-registering after a <see cref="PythonEngine.Shutdown"/> +
    ///     <see cref="PythonEngine.Initialize()"/>. It is an addition to
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/>, not a replacement — call whichever reads
    ///     better at the site; both hit the same sticky registration.</para>
    ///
    ///     <para>Because these are C# extension members, the <c>Py.RegisterNumSharpCodecs()</c> spelling
    ///     is visible only where <c>using NumSharp.Interop.PythonNet;</c> is in scope — which is already
    ///     the case anywhere <see cref="NDArray"/> or <see cref="NDArrayPythonInterop"/> is used.</para>
    /// </summary>
    public static class PyNumSharpExtensions
    {
        extension(Py)
        {
            /// <summary>
            ///     Alias for <see cref="NDArrayPythonInterop.RegisterCodec()"/>: registers
            ///     <see cref="NumpyCodec"/> and <see cref="TupleCodec"/> with pythonnet's conversion
            ///     pipeline so <see cref="NDArray"/> auto-encodes to a zero-copy numpy view,
            ///     <c>PyObject.As&lt;NDArray&gt;()</c> decodes numpy / buffers / registered adapters, and
            ///     C# tuples cross as Python tuples (shapes, axes, multi-indices). "Codecs" is plural
            ///     because both codecs are registered.
            /// </summary>
            /// <returns><c>true</c> if this call performed the registration; <c>false</c> if the codec
            /// was already registered for the current engine session.</returns>
            public static bool RegisterNumSharpCodecs() => NDArrayPythonInterop.RegisterCodec();

            /// <inheritdoc cref="RegisterNumSharpCodecs()"/>
            /// <param name="options">Encode/decode policies (view vs copy, which Python types decode,
            /// whether tuples convert).</param>
            public static bool RegisterNumSharpCodecs(NumpyCodecOptions options)
                => NDArrayPythonInterop.RegisterCodec(options);
        }
    }
}
