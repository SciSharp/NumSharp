using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Examples;

/// <summary>
/// The OpenBLAS guard shared by the example tests (the Gist ports, the Karpathy demos, the signal
/// polyfit companion). Their golden transcripts and exact-equality claims were recorded with the
/// bundled scipy-openblas at ONE thread, and <c>np.polyfit</c> needs LAPACK <c>lstsq</c>, which Core
/// ships no managed fallback for — so these tests need that backend bound.
/// </summary>
/// <remarks>
/// <para>
/// CI's <c>test</c> job deliberately stages NO native binary: the OpenBLAS runtime assets are
/// gitignored and <c>fetch_openblas.py</c> runs only in the interop / package jobs. Calling
/// <see cref="OpenBlasEngine.Enable(string, int, string)"/> unguarded therefore threw
/// <see cref="DllNotFoundException"/> there and failed 25 example tests on all three OSes. Like the
/// LAPACK factorisation and matmul-parity tests (<c>LapackEigTests.RequireLapack</c> and siblings),
/// an example that needs the backend now ends <see cref="Assert.Inconclusive(string)"/> on a host that
/// cannot load one, and still runs in full wherever the binary is staged.
/// </para>
/// <para>
/// Binding replaces the process-global <c>TensorEngine.Blas</c> seam. Callers keep their existing
/// capture-and-restore of the previous backend (a <c>finally</c> or <c>[TestCleanup]</c>); a failed
/// <c>Enable</c> is a no-op, so there is nothing extra to undo on the inconclusive path.
/// </para>
/// </remarks>
internal static class ExampleBlasBackend
{
    /// <summary>
    /// Try to bind OpenBLAS at one thread, reporting failure instead of throwing. For a class that
    /// binds in <c>[TestInitialize]</c> but has tests that do not depend on the backend.
    /// </summary>
    /// <param name="reason">
    /// Receives why binding failed (the first line of the loader's message, prefixed with the
    /// exception type), or null on success. Pass it to <see cref="Require"/> in the tests that need
    /// the backend.
    /// </param>
    /// <returns>True when OpenBLAS is now the bound backend; false when this host has no loadable CBLAS.</returns>
    /// <remarks>
    /// Catches every exception from <c>Enable</c>, exactly like the house <c>RequireLapack</c> guards:
    /// the loader reports "nothing found" as <see cref="DllNotFoundException"/> but a library that loads
    /// and lacks the CBLAS symbols fails differently, and either way the host simply has no usable
    /// backend. Only binding is guarded — a demo that throws AFTER binding still fails its test.
    /// </remarks>
    internal static bool TryEnable(out string reason)
    {
        try
        {
            OpenBlasEngine.Enable(threads: 1);
            reason = null;
            return true;
        }
        catch (Exception e)
        {
            reason = $"no CBLAS library on this host ({e.GetType().Name}): {e.Message.Split('\n')[0]}";
            return false;
        }
    }

    /// <summary>
    /// End the calling test as inconclusive when an earlier <see cref="TryEnable"/> failed.
    /// </summary>
    /// <param name="reason">The <c>reason</c> <see cref="TryEnable"/> produced; null means the backend is bound and this returns normally.</param>
    /// <exception cref="AssertInconclusiveException"><paramref name="reason"/> is non-null — the backend is not bound on this host.</exception>
    internal static void Require(string reason)
    {
        if (reason != null)
            Assert.Inconclusive(reason);
    }

    /// <summary>
    /// Bind OpenBLAS at one thread for the calling test, or end the test as inconclusive when this
    /// host cannot provide what it needs.
    /// </summary>
    /// <param name="requireLapack">
    /// Also require the LAPACK routines (<c>np.polyfit</c> rides <c>lstsq</c>/<c>gelsd</c>). A bare
    /// reference CBLAS can bind and still export none of them.
    /// </param>
    /// <exception cref="AssertInconclusiveException">
    /// No loadable CBLAS on this host, or <paramref name="requireLapack"/> is set and the loaded
    /// library exports no LAPACK.
    /// </exception>
    internal static void EnableOrInconclusive(bool requireLapack = false)
    {
        if (!TryEnable(out var reason))
            Assert.Inconclusive(reason);
        if (requireLapack && !OpenBlasEngine.LapackAvailable)
            Assert.Inconclusive("the loaded BLAS exports no LAPACK routines (a bare reference CBLAS).");
    }
}
