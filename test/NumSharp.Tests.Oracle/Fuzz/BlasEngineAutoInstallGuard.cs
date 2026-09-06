using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Keeps <c>NumSharp.Interop.OpenBLAS</c> from installing itself for the whole test run.
    /// </summary>
    /// <remarks>
    ///     The package is referenced so its gate can run, and its whole design is that referencing
    ///     it IS the opt-in — a module initializer installs the engine as soon as the assembly
    ///     loads. In a test run that would be wrong twice over: every other tier asserts NumSharp's
    ///     OWN kernels, and the install would happen at whatever unpredictable moment the first
    ///     parity type is touched, so earlier tests would use one engine and later ones another.
    ///     <para>
    ///     This module initializer runs when the TEST assembly loads — before anything can touch a
    ///     type in the interop assembly — and opts out. The parity tests then enable the engine
    ///     explicitly, per test, and disable it again.
    ///     </para>
    /// </remarks>
    internal static class BlasEngineAutoInstallGuard
    {
        [ModuleInitializer]
        internal static void SuppressAutoInstall()
            => Environment.SetEnvironmentVariable("NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL", "0");
    }
}
