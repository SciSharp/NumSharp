using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a method (or a property accessor) as an <see cref="NDScope"/> boundary: at build
    ///     time the NumSharp IL weaver (<c>tools/NumSharp.Weaver</c>, shipped to consumer projects
    ///     as the <c>NumSharp.Weaver</c> NuGet package) injects the exact code the
    ///     hand-written pattern spells —
    ///     <code>
    ///     using var scope = NDScope.Open();
    ///     ...original body, byte-for-byte...
    ///     return scope.Returns(result);        // NDArray-like returns
    ///     </code>
    ///     — so the source keeps its 100% original body and the reclamation is invisible.
    /// </summary>
    /// <remarks>
    ///     <para>What the weaver injects (see <c>DISPOSAL-GUIDELINES.md</c> → "The weaver"):
    ///     a scope local assigned from <see cref="NDScope.Open"/> before the original first
    ///     instruction; the whole original body wrapped in try/finally with
    ///     <see cref="NDScope.Dispose"/> in the finally; every <c>ret</c> routed through
    ///     <see cref="NDScope.Returns{T}(T)"/> for <see cref="NDArray"/>-like returns
    ///     (<see cref="NDScope.Returns{T}(T[])"/> for array returns, the typed <c>Returns</c> tuple
    ///     overloads for a small all-NDArray <c>ValueTuple</c> and <c>Returns(ITuple)</c> for any other
    ///     <c>ValueTuple</c>/<c>Tuple</c> — any arity up to 8, mixed components — <c>Returns(IArraySlice)</c>/
    ///     <c>Returns(UnmanagedStorage)</c> for a bare lower-layer buffer return, and
    ///     <see cref="INDArrayCarrier.YieldTo"/> for a result-struct carrier), and every
    ///     <c>out NDArray</c> parameter's final value yielded before each successful return.</para>
    ///     <para><b>Async and iterator methods are woven too</b> — through their compiler state
    ///     machines. The stub keeps only the attribute; <c>MoveNext</c> gets ONE scope for the whole
    ///     logical invocation (held in a weaver-added state-machine field), UNINSTALLED before each
    ///     await's continuation is scheduled and re-installed on whatever thread resumes — so temps
    ///     stay alive while an awaited callee still uses them, and everything is reclaimed at
    ///     SetResult/SetException (async), at the final <c>MoveNext</c>/<c>Dispose()</c> (iterators,
    ///     early <c>break</c> included), with results and <c>yield return</c>ed elements routed
    ///     through the same <c>Returns</c>/<c>YieldTo</c> egress. A NON-async method returning
    ///     <c>Task</c>/<c>ValueTask</c>[<c>&lt;T&gt;</c>] yields a completed task's result
    ///     immediately and DEFERS the scope's disposal to an incomplete task's completion (the
    ///     in-flight callee may still hold tracked temps); an incomplete <c>ValueTask</c> is
    ///     <c>Preserve()</c>d — the caller receives the multi-observable form.</para>
    ///     <para>The weaver REJECTS (build error NDW003) only shapes whose egress it cannot see:
    ///     <c>ref NDArray</c> parameters, and an UNSUPPORTED carrier — a bespoke reference type, a
    ///     collection, or a result struct that does NOT implement <see cref="INDArrayCarrier"/> —
    ///     whether returned directly, inside a <c>Task&lt;T&gt;</c>, produced by an async method, or
    ///     <c>yield return</c>ed; scope those by hand (NDW004 = an unrecognized, non-C# state-machine
    ///     shape; NDW008 = the referenced NumSharp predates the async seam).</para>
    ///     <para>A method whose body already opens an <see cref="NDScope"/> is skipped
    ///     (idempotence), so hand-scoped code may carry the attribute without double-wrapping.</para>
    ///     <para>PUBLIC because the attribute is consumer-facing: a project that installs the
    ///     <c>NumSharp.Weaver</c> package marks its own composition methods with it. Without the
    ///     package the attribute is inert metadata — the method runs unscoped, the pre-weave
    ///     finalizer-backstop behaviour.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    public sealed class NDScopedAttribute : Attribute
    {
    }
}
