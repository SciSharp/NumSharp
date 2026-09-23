using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     Marks a test — or a whole test class — that needs a third-party Python package beyond numpy
    ///     (torch, pandas, scipy, pyarrow, pillow, polars, opencv). Its category routes the test to the
    ///     <c>ecosystem</c> Python environment; every untagged test runs against <c>parity</c>, which
    ///     holds numpy alone.
    /// </summary>
    /// <remarks>
    ///     <para><b>Why the split exists.</b> The byte-exact live-NumPy tests need numpy pinned to the
    ///     release whose bundled OpenBLAS NumSharp ships, in a process nothing else perturbs. The
    ///     bridge tests need third-party libraries that release on their own schedule, pin numpy ranges
    ///     of their own and load their own native runtimes (torch's OpenMP/MKL). One environment
    ///     cannot promise both, so CI builds two with <c>python-envs/make_env.py</c> and runs the suite
    ///     twice, selected by this category:</para>
    ///     <code>
    ///     --filter "TestCategory!=PythonEcosystem"   # against .venvs/parity-py312
    ///     --filter "TestCategory=PythonEcosystem"    # against .venvs/ecosystem-py312
    ///     </code>
    ///     <para><b>Drift guard.</b> The tag is not optional bookkeeping: every package gate
    ///     (<see cref="InteropTestBase.SkipUnless"/>, <c>PyTorchTestGate</c>, <c>PandasTestGate</c>)
    ///     FAILS a test that calls it without carrying this attribute on the method or its class.
    ///     Untagged, such a test would run only in the numpy-only environment, skip there as
    ///     Inconclusive, and never run anywhere — green-by-skipping. Tagging a class is right when
    ///     (nearly) every test in it needs a package; tag methods of a mixed class individually.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false, Inherited = true)]
    public sealed class PythonEcosystemAttribute : TestCategoryBaseAttribute
    {
        /// <summary>The test category CI filters on (<c>TestCategory=PythonEcosystem</c>).</summary>
        public const string Category = "PythonEcosystem";

        /// <inheritdoc />
        public override IList<string> TestCategories => [Category];
    }
}
