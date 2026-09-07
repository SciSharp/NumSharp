using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     Static extension members that hang NumSharp's registration surface off pythonnet's
    ///     <see cref="Py"/> class, so it reads beside <see cref="Py.Import(string)"/> /
    ///     <see cref="Py.GIL()"/>:
    ///     <code>
    ///     using NumSharp.Interop.PythonNet;   // brings the extensions into scope
    ///     using Python.Runtime;
    ///
    ///     PythonEngine.Initialize();
    ///     Py.RegisterNumSharpCodec();              // == NDArrayPythonInterop.RegisterCodec()
    ///     Py.RegisterNumSharpArrayAdapter(myAd);   // == NDArrayPythonInterop.RegisterArrayAdapter(myAd)
    ///     </code>
    ///
    ///     <para>These are thin, discoverable aliases — nothing more. Each forwards verbatim to the
    ///     matching <see cref="NDArrayPythonInterop"/> method, so semantics and return values are
    ///     identical; they are additions to that surface, not replacements. Call whichever reads better
    ///     at the site.</para>
    ///
    ///     <para>Because these are C# extension members, the <c>Py.RegisterNumSharp*</c> spellings are
    ///     visible only where <c>using NumSharp.Interop.PythonNet;</c> is in scope — which is already the
    ///     case anywhere <see cref="NDArray"/> or <see cref="NDArrayPythonInterop"/> is used.</para>
    ///
    ///     <para><b>Consumer compiler requirement:</b> extension members are a C# 14 feature, and unlike
    ///     classic (instance) extension methods they must be resolved at the <em>call site</em> by a
    ///     C# 14+ compiler (the .NET 10 SDK). A project building with C# 13 or earlier — the .NET 8/9
    ///     SDK default, even when this package's net8.0 target is used — cannot call these members (it
    ///     fails with <c>CS9202: Feature 'extensions' is not available</c>). Such consumers call the
    ///     underlying <see cref="NDArrayPythonInterop"/> methods directly, which are identical and
    ///     compile on every C# version. The extended <see cref="Py"/> is pythonnet's own <c>static</c>
    ///     class; these members are NumSharp's, added onto it — pythonnet has no knowledge of NumSharp.</para>
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
            ///     C# tuples cross as Python tuples (shapes, axes, multi-indices). Process-global and
            ///     per-engine-session (idempotent within a session; re-registers after
            ///     <see cref="PythonEngine.Shutdown"/> + <see cref="PythonEngine.Initialize()"/>).
            /// </summary>
            /// <returns><c>true</c> if this call performed the registration; <c>false</c> if the codec
            /// was already registered for the current engine session.</returns>
            public static bool RegisterNumSharpCodec() => NDArrayPythonInterop.RegisterCodec();

            /// <inheritdoc cref="RegisterNumSharpCodec()"/>
            /// <param name="options">Encode/decode policies (view vs copy, which Python types decode,
            /// whether tuples convert).</param>
            public static bool RegisterNumSharpCodec(NumpyCodecOptions options)
                => NDArrayPythonInterop.RegisterCodec(options);

            /// <summary>
            ///     Alias for <see cref="NDArrayPythonInterop.RegisterArrayAdapter(IPythonArrayAdapter)"/>:
            ///     registers a library-specific <see cref="IPythonArrayAdapter"/> so its objects feed the
            ///     existing memory bridge. Registration is process-wide, thread-safe and idempotent by
            ///     <see cref="IPythonArrayAdapter.Name"/>; the built-in Torch and Pandas adapters are
            ///     already present.
            /// </summary>
            /// <param name="adapter">The session-neutral adapter to add.</param>
            /// <returns><c>true</c> when added; <c>false</c> when an adapter with the same name already exists.</returns>
            public static bool RegisterNumSharpArrayAdapter(IPythonArrayAdapter adapter)
                => NDArrayPythonInterop.RegisterArrayAdapter(adapter);
        }
    }
}
