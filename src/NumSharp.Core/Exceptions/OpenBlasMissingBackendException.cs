using System;

namespace NumSharp
{
    /// <summary>
    ///     A <see cref="MissingBackendException"/> whose message says how to make the operation work:
    ///     reference the <c>NumSharp.Interop.OpenBLAS</c> NuGet package.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp.Core is 100 % managed C# and ships no matrix factorisation of its own — unlike the
    ///     matrix products, which always have a managed kernel to fall back to. The factorisations
    ///     (<c>inv</c>, <c>det</c>, <c>solve</c>, <c>svd</c>, <c>eig</c>, <c>qr</c>, …) are served by an
    ///     <see cref="NumSharp.Backends.IBlasBackend"/> assigned to
    ///     <see cref="NumSharp.Backends.TensorEngine.Blas"/>, and the package that supplies one is
    ///     <see cref="PackageId"/>. <b>There is no separate seam to install</b>: the one
    ///     <c>NumSharp.Interop.OpenBLAS</c> reference fills the whole <c>Blas</c> property — products and
    ///     factorisations alike — from a <c>[ModuleInitializer]</c>, so adding the reference is the whole
    ///     opt-in.
    ///     </para>
    ///     <para>
    ///     Distinct from the interop package's own <c>OpenBlasRequiredOverrideException</c>: that is
    ///     thrown when the package IS referenced but a hard-required build override cannot load. This one
    ///     is thrown by Core when no backend is referenced at all.
    ///     </para>
    /// </remarks>
    public class OpenBlasMissingBackendException : MissingBackendException
    {
        /// <summary>The NuGet package that supplies the <see cref="NumSharp.Backends.IBlasBackend"/>.</summary>
        public const string PackageId = "NumSharp.Interop.OpenBLAS";

        /// <summary>
        ///     The sentence that tells a caller how to make the operation work — the single source of
        ///     truth for the package name across every message that raises this exception.
        /// </summary>
        public static string HowToFix =>
            $"Reference the {PackageId} NuGet package — it assigns TensorEngine.Blas an IBlasBackend " +
            $"from a [ModuleInitializer], so adding the package reference is the whole opt-in.";

        public OpenBlasMissingBackendException(string message) : base(message) { }

        public OpenBlasMissingBackendException(string message, Exception innerException)
            : base(message, innerException) { }
    }
}
