using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Keeps <c>NumSharp.Interop.OpenBLAS</c> from installing itself for the whole test run.
    /// </summary>
    /// <remarks>
    ///     This assembly references the package so its engine tests can run (the LAPACK
    ///     factorisation gates in <c>Backends/</c>, <c>LinAlgEngineSeamTests</c>, the Level-3 BLAS
    ///     binding tests), but the vast majority of its tiers assert NumSharp's OWN managed kernels —
    ///     and <c>LinAlgEngineSeamTests</c> asserts the no-backend contract outright. The package's
    ///     design is that referencing it IS the opt-in: a module initializer installs the engine as
    ///     soon as the interop assembly loads. That would be wrong here twice over — it would flip
    ///     every managed-kernel assertion, and it would happen at whatever unpredictable moment the
    ///     first interop type is touched.
    ///     <para>
    ///     This module initializer runs when THIS (test) assembly loads — before anything can touch a
    ///     type in the interop assembly — and opts out. The engine tests then enable the backend
    ///     explicitly, per test, and disable it again.
    ///     </para>
    ///     <para>
    ///     The isolated oracle harness carries its own copy of this guard
    ///     (<c>NumSharp.Tests.Oracle</c> → <c>Fuzz/BlasEngineAutoInstallGuard.cs</c>): each
    ///     OpenBLAS-referencing test assembly is a separate module and needs its own initializer.
    ///     </para>
    /// </remarks>
    internal static class BlasEngineAutoInstallGuard
    {
        [ModuleInitializer]
        internal static void SuppressAutoInstall()
            => Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", "0");
    }
}
