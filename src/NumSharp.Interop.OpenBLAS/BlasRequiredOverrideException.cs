using System;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     A version override staged by the build (an <c>openblas.source.json</c> marker with
    ///     <c>mode = "version"</c>) could not be loaded.
    /// </summary>
    /// <remarks>
    ///     A version override — <c>&lt;OpenBlasVersion&gt;</c> on the <c>PackageReference</c> or
    ///     <c>NUMSHARP_OPENBLAS_VERSION</c> at build — is a HARD requirement: the consumer pinned one
    ///     specific scipy-openblas build, so discovery must not quietly substitute the bundle or a
    ///     machine-wide OpenBLAS when the staged binary is missing or unloadable. Falling back would
    ///     be the worst outcome a pin can have: every product still computes, with different bits
    ///     than the ones the caller contracted for.
    ///     <para>
    ///     Derives from <see cref="DllNotFoundException"/> so existing catch sites keep working; it
    ///     is its own type so the module-load auto-install can tell a broken CONTRACT (reported
    ///     loudly) from the ordinary no-BLAS-anywhere case (silent by design).
    ///     </para>
    /// </remarks>
    public sealed class BlasRequiredOverrideException : DllNotFoundException
    {
        public BlasRequiredOverrideException(string message) : base(message)
        {
        }
    }
}
