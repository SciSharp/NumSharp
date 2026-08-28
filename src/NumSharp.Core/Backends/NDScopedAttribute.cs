using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a method (or a property accessor) as an <see cref="NDScope"/> boundary: at build
    ///     time the NumSharp IL weaver (<c>tools/NumSharp.Build</c>, shipped to consumer projects
    ///     as the <c>NumSharp.Build</c> NuGet package) injects the exact code the
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
    ///     <para><b>Synchronous iterators are woven too</b> — an <c>IEnumerable&lt;T&gt;</c>/
    ///     <c>IEnumerator&lt;T&gt;</c> method (<c>yield return</c> without <c>await</c>) compiles to a
    ///     state machine, so the stub keeps only the attribute and the scope is held in a weaver-added
    ///     state-machine field: ONE scope for the whole enumeration, suspended between <c>MoveNext</c>
    ///     calls and reclaimed at the final <c>MoveNext</c> or the enumerator's <c>Dispose()</c> (an
    ///     early <c>break</c> out of a <c>foreach</c> included), with every <c>yield return</c>ed
    ///     element routed through the same <c>Returns</c>/<c>YieldTo</c> egress the consumer owns.</para>
    ///     <para><b>Async methods, async iterators and non-async <c>Task</c>/<c>ValueTask</c> returns
    ///     use <see cref="NDScopedAsyncAttribute"/> instead</b> — those suspend across <c>await</c> (or
    ///     defer disposal to a task's completion) and need the deferral seam. Marking one of them
    ///     <c>[NDScoped]</c> is a build ERROR (NDW009) that names the right attribute, never a silent
    ///     unwoven ship.</para>
    ///     <para>The weaver REJECTS (build error NDW003) only shapes whose egress it cannot see:
    ///     <c>ref NDArray</c> parameters, and an UNSUPPORTED carrier — a bespoke reference type, a
    ///     collection, or a result struct that does NOT implement <see cref="INDArrayCarrier"/> —
    ///     whether returned directly or <c>yield return</c>ed; scope those by hand (NDW004 = an
    ///     unrecognized, non-C# state-machine shape; NDW008 = the referenced NumSharp predates the
    ///     iterator scope seam).</para>
    ///     <para>A method whose body already opens an <see cref="NDScope"/> is skipped
    ///     (idempotence), so hand-scoped code may carry the attribute without double-wrapping.</para>
    ///     <para>PUBLIC because the attribute is consumer-facing: a project that installs the
    ///     <c>NumSharp.Build</c> package marks its own composition methods with it. Without the
    ///     package the attribute is inert metadata — the method runs unscoped, the pre-weave
    ///     finalizer-backstop behaviour.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    public sealed class NDScopedAttribute : Attribute
    {
    }
}
