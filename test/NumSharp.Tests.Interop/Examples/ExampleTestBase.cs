using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Tests.Interop.Examples
{
    /// <summary>
    ///     Base for the executable twins of <c>examples/NumSharp.Interop.pythonnet.Examples</c>: each test
    ///     class here is one tutorial script, each test method one of its sections, with the SAME code and
    ///     every comment that states a value turned into an assertion. The scripts are the tutorial; these
    ///     are the proof.
    ///
    ///     <para>Inherits the engine, the numpy namespace, the GIL/pump helpers and the per-test leak gate
    ///     from <see cref="InteropTestBase"/>, and adds the scripts' vocabulary: <see cref="py"/> (the
    ///     namespace as <c>dynamic</c>, exactly the scripts' <c>dynamic py = host.Namespace</c>),
    ///     <see cref="Throws"/>, <see cref="HasModule"/> and <see cref="Drain"/> (the scripts'
    ///     <c>PythonHost.Drain()</c>).</para>
    /// </summary>
    public abstract class ExampleTestBase : InteropTestBase
    {
        /// <summary>The scripts' <c>dynamic py</c>: a Python module where <c>import numpy as np</c> already ran. Use under <see cref="InteropTestBase.Gil"/>.</summary>
        protected dynamic py => Scope;

        /// <summary>The scripts' <c>PythonHost.Start()</c> registers the codec once per process; the suite does it here.</summary>
        [TestInitialize]
        public void ExampleInit() => NDArrayPythonInterop.RegisterCodec();

        /// <summary>The scripts' <c>Throws(...)</c>: the exception an action raises, or <c>null</c>.</summary>
        protected static Exception Throws(Action act)
        {
            try { act(); return null; }
            catch (Exception e) { return e; }
        }

        /// <summary>The scripts' <c>HasModule(...)</c>: imports the module into the namespace, false when it is not installed.</summary>
        protected bool HasModule(string name)
        {
            using (Gil()) return Throws(() => Scope.Exec($"import {name}")) is null;
        }

        /// <summary>The scripts' <c>PythonHost.Drain()</c>: both collectors to completion plus the interop's inline drain.</summary>
        protected static void Drain() => Pump();
    }
}
