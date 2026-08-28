using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a method (or property accessor) as a <b>helper that always runs under an ambient
    ///     <see cref="NDScope"/> opened by its caller</b> — so its <see cref="NDArray"/> temporaries are
    ///     reclaimed by that caller's scope, and the <c>NDW012</c> leak analyzer must treat the method
    ///     as covered instead of flagging its transients.
    /// </summary>
    /// <remarks>
    ///     <para><b>What it is — and is NOT.</b> Unlike <see cref="NDScopedAttribute"/> /
    ///     <see cref="NDScopedAsyncAttribute"/>, this attribute is <b>NOT a scope boundary</b>: the IL
    ///     weaver never weaves it, no <see cref="NDScope"/> is opened here, and it is completely inert at
    ///     runtime. It exists ONLY to inform the compile-time leak analyzer
    ///     (<c>NumSharp.Weaver.Analyzer</c>): <c>NDW012</c> is a per-method dataflow pass with no
    ///     call-graph, so it cannot see that a helper's transients are reclaimed by a
    ///     <c>[NDScoped]</c>/<c>[NDScopedAsync]</c> (or hand-scoped) CALLER's ambient scope — the
    ///     documented "scope the boundary, helpers ride the ambient scope" pattern
    ///     (<c>DISPOSAL-GUIDELINES.md</c>). This attribute is the author's assertion of exactly that,
    ///     so the analyzer exempts the method.</para>
    ///
    ///     <para><b>The coverage contract the author asserts.</b> Because <see cref="NDScope"/> tracks
    ///     every array constructed while a scope is open on the current thread (the constructor funnel
    ///     <c>NDScope.Track</c>), a helper is genuinely covered iff EVERY call path that reaches it does
    ///     so <b>synchronously, on the same thread, while an ambient scope is open</b>. Marking a method
    ///     <c>[NDScopedHelper]</c> asserts that invariant holds — typically because its only callers are
    ///     <c>[NDScoped]</c> boundary methods (or other <c>[NDScopedHelper]</c>s below them). If the
    ///     method is ever invoked WITHOUT an ambient scope (a public entry point that was not scoped, a
    ///     call from a <c>lambda</c>/<c>Task.Run</c> that runs after the scope closed, another thread),
    ///     its temporaries fall back to the finalizer backstop — a real leak the analyzer will no longer
    ///     report. The assertion is the author's responsibility, the same way a wrong hand-written
    ///     <c>using</c> is.</para>
    ///
    ///     <para><b>Why not just mark it <c>[NDScoped]</c>?</b> That also silences the analyzer, but it
    ///     WEAVES a nested scope into the helper (a per-call <see cref="NDScope.Open"/> +
    ///     <see cref="NDScope.Returns{T}(T)"/>). Nested scopes compose correctly, but they are not free —
    ///     for a hot helper called under a boundary that already owns a scope, the nested scope is pure
    ///     overhead. <c>[NDScopedHelper]</c> is the zero-runtime-cost choice when the caller's scope
    ///     already does the reclamation.</para>
    ///
    ///     <para><b>No weaver, no NDW013.</b> The weaver collects targets by the exact type names
    ///     <c>NumSharp.NDScopedAttribute</c> / <c>NumSharp.NDScopedAsyncAttribute</c>, so this attribute
    ///     is never a weave target and never a target-gate (NDW002–011) subject. The
    ///     "you used <c>[NDScoped]</c> but the weaver is absent" guard (NDW013) and the weaver's own
    ///     usage pre-scan match those two names precisely (not the shared <c>NDScoped</c> prefix), so a
    ///     project using only <c>[NDScopedHelper]</c> — which needs no weaver — draws neither.</para>
    ///
    ///     <para>PUBLIC for the same reason as the scope attributes: a consumer that adopts the
    ///     <c>[NDScoped]</c> pattern can annotate its own always-ambient helpers to keep its build clean.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    public sealed class NDScopedHelperAttribute : Attribute
    {
    }
}
