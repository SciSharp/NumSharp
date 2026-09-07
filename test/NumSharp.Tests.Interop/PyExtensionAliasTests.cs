using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The C#-14 static extension members on pythonnet's <see cref="Py"/> class
    ///     (<see cref="PyNumSharpExtensions"/>) are thin aliases that must forward verbatim to
    ///     <see cref="NDArrayPythonInterop"/>. These tests prove each alias hits the SAME underlying
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

        [TestMethod]
        public void RegisterNumSharpArrayAdapter_ForwardsToTheRegistry()
        {
            // A fresh, uniquely-named adapter registers exactly once in the process; the second call is
            // the registry's idempotent-by-Name no-op, and the direct call agrees — proving the alias
            // targets the real PythonArrayAdapterRegistry, not a separate list.
            var adapter = new NoopAdapter("pytest-py-alias-" + Guid.NewGuid().ToString("N"));
            Py.RegisterNumSharpArrayAdapter(adapter).Should().BeTrue("a fresh adapter Name registers");
            Py.RegisterNumSharpArrayAdapter(adapter).Should().BeFalse("the same Name is idempotent");
            NDArrayPythonInterop.RegisterArrayAdapter(adapter).Should().BeFalse("the direct call sees the same registry");
        }

        /// <summary>Minimal adapter: only its Name matters for a registration test; it never adapts.</summary>
        private sealed class NoopAdapter : IPythonArrayAdapter
        {
            public NoopAdapter(string name) => Name = name;
            public string Name { get; }
            public bool CanAdapt(PyType objectType) => false;
            public PyObject Adapt(PyObject source, bool allowCopy) => null;
        }
    }
}
