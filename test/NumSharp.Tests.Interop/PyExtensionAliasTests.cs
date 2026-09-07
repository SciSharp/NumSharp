using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The C#-14 static extension member on pythonnet's <see cref="Py"/> class
    ///     (<see cref="PyNumSharpExtensions"/>) is a thin alias that must forward verbatim to
    ///     <see cref="NDArrayPythonInterop.RegisterCodec()"/>. This proves it hits the SAME underlying
    ///     registration as the direct call — not a separate or dead path.
    /// </summary>
    [TestClass]
    public class PyExtensionAliasTests : InteropTestBase
    {
        [TestMethod]
        public void RegisterNumSharpCodec_HitsTheSameStickyRegistration()
        {
            // Codec registration is process-global and sticky for the engine session, so whichever test
            // registers first wins. Ensure via the direct call, then the alias must observe the SAME
            // registration and report the no-op.
            NDArrayPythonInterop.RegisterCodec();
            Py.RegisterNumSharpCodec().Should().BeFalse("the alias shares the direct call's sticky registration");

            // And the codec is genuinely active after going through the Py.* entry point.
            var nd = np.arange(3).astype(NPTypeCode.Double);
            using (Gil())
                Scope.Set("pyalias_codec", nd);
            PyStr("type(pyalias_codec).__name__").Should().Be("ndarray", "the registered codec auto-encodes NDArray -> numpy");
        }
    }
}
