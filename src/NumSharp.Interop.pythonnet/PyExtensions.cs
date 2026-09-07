using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     A static extension member that hangs NumSharp's ONE registration switch off pythonnet's
    ///     <see cref="Py"/> class, so it reads beside <see cref="Py.Import(string)"/> /
    ///     <see cref="Py.GIL()"/>:
    ///     <code>
    ///     using NumSharp.Interop.PythonNet;   // brings the extension into scope
    ///     using Python.Runtime;
    ///
    ///     PythonEngine.Initialize();
    ///     Py.RegisterNumSharpCodec();                                   // everything on (defaults)
    ///     Py.RegisterNumSharpCodec(new NumpyCodecOptions { ... });      // opt a support out
    ///     </code>
    ///
    ///     <para><b>One call enables ALL of NumSharp's pythonnet support</b>, and
    ///     <see cref="NumpyCodecOptions"/> is the single control surface — every support is a flag, all
    ///     defaulting ON: <see cref="NDArray"/>⇄numpy (<see cref="NumpyCodecOptions.EncodeMode"/> /
    ///     <see cref="NumpyCodecOptions.DecodeMode"/>), PEP 3118 buffers
    ///     (<see cref="NumpyCodecOptions.DecodeAnyBuffer"/>), the built-in <c>torch</c> / <c>pandas</c>
    ///     (and any registered) adapters (<see cref="NumpyCodecOptions.DecodeArrayAdapters"/>),
    ///     <c>list</c>/<c>tuple</c>/scalar array-likes (<see cref="NumpyCodecOptions.DecodeArrayLike"/>),
    ///     and C# tuples ⇄ Python tuples (<see cref="NumpyCodecOptions.ConvertTuples"/>). So the built-in
    ///     Torch and Pandas support needs no separate registration — it is part of this call. The only
    ///     thing outside it is adding your OWN third-party adapter, a rare extension point that stays on
    ///     <see cref="NDArrayPythonInterop.RegisterArrayAdapter(IPythonArrayAdapter)"/> (it takes an
    ///     adapter instance; it is not a support toggle).</para>
    ///
    ///     <para>This is a thin, discoverable alias — it forwards verbatim to
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/> (both overloads), so semantics and return
    ///     values are identical; it is an addition to that surface, not a replacement.</para>
    ///
    ///     <para>Because this is a C# extension member, the <c>Py.RegisterNumSharpCodec()</c> spelling is
    ///     visible only where <c>using NumSharp.Interop.PythonNet;</c> is in scope — which is already the
    ///     case anywhere <see cref="NDArray"/> or <see cref="NDArrayPythonInterop"/> is used.</para>
    ///
    ///     <para><b>Consumer compiler requirement:</b> extension members are a C# 14 feature, and unlike
    ///     classic (instance) extension methods they must be resolved at the <em>call site</em> by a
    ///     C# 14+ compiler (the .NET 10 SDK). A project building with C# 13 or earlier — the .NET 8/9
    ///     SDK default, even when this package's net8.0 target is used — cannot call this member (it
    ///     fails with <c>CS9202: Feature 'extensions' is not available</c>). Such consumers call
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/> directly, which is identical and compiles
    ///     on every C# version. The extended <see cref="Py"/> is pythonnet's own <c>static</c> class;
    ///     this member is NumSharp's, added onto it — pythonnet has no knowledge of NumSharp.</para>
    /// </summary>
    public static class PyNumSharpExtensions
    {
        extension(Py)
        {
            /// <summary>
            ///     Alias for <see cref="NDArrayPythonInterop.RegisterCodec()"/> — the single switch that
            ///     turns on all NumSharp ⇄ Python support at every pythonnet boundary: registers
            ///     <see cref="NumpyCodec"/> (NDArray⇄numpy, PEP 3118 buffers, and the built-in
            ///     torch/pandas adapters, per <see cref="NumpyCodecOptions"/>) and — when
            ///     <see cref="NumpyCodecOptions.ConvertTuples"/> is set (default) — <see cref="TupleCodec"/>
            ///     (C# tuples ⇄ Python tuples). Process-global and per-engine-session: idempotent within a
            ///     session; re-registers after <see cref="PythonEngine.Shutdown"/> +
            ///     <see cref="PythonEngine.Initialize()"/>.
            /// </summary>
            /// <returns><c>true</c> if this call performed the registration; <c>false</c> if the codec
            /// was already registered for the current engine session.</returns>
            public static bool RegisterNumSharpCodec() => NDArrayPythonInterop.RegisterCodec();

            /// <inheritdoc cref="RegisterNumSharpCodec()"/>
            /// <param name="options">The control surface — every support is a flag, all default ON.
            /// Pass this to opt a support out (e.g. <c>DecodeArrayAdapters = false</c> to ignore
            /// torch/pandas, or a non-<see cref="NumpyCodecMode.Auto"/> encode/decode mode).</param>
            public static bool RegisterNumSharpCodec(NumpyCodecOptions options)
                => NDArrayPythonInterop.RegisterCodec(options);
        }
    }
}
