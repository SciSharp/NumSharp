using System;

namespace NumSharp
{
    /// <summary>
    ///     The ASYNC counterpart of <see cref="NDScopedAttribute"/>: marks an <b>async</b> method, an
    ///     <b>async iterator</b>, or a <b>non-async method returning <c>Task</c>/<c>ValueTask</c></b>[<c>&lt;T&gt;</c>]
    ///     as an <see cref="NDScope"/> boundary. At build time the NumSharp IL weaver
    ///     (<c>tools/NumSharp.Build</c>, shipped to consumer projects as the <c>NumSharp.Build</c>
    ///     NuGet package) weaves the method's compiler STATE MACHINE — or, for a non-async
    ///     Task-returning body, its DEFERRAL egress — so the <see cref="NDArray"/> temporaries it
    ///     drops are reclaimed at the invocation's completion instead of waiting on the finalizer,
    ///     with the source keeping its 100% original body exactly as <see cref="NDScopedAttribute"/>
    ///     does for synchronous ones.
    /// </summary>
    /// <remarks>
    ///     <para><b>What this attribute covers</b> (the shapes that suspend across <c>await</c>, or
    ///     defer disposal to a task's completion — everything that needs the async scope seam):</para>
    ///     <list type="bullet">
    ///       <item><b>Async methods</b> — <c>Task</c>/<c>Task&lt;T&gt;</c>/<c>ValueTask</c>/
    ///       <c>ValueTask&lt;T&gt;</c>/<c>void</c>, and any custom <c>[AsyncMethodBuilder]</c> task-like.</item>
    ///       <item><b>Async iterators</b> — <c>IAsyncEnumerable&lt;T&gt;</c> (<c>await</c> + <c>yield return</c>).</item>
    ///       <item><b>Non-async methods returning <c>Task</c>/<c>ValueTask</c></b>[<c>&lt;T&gt;</c>] —
    ///       e.g. <c>Task&lt;NDArray&gt; M() =&gt; ComputeAsync();</c>.</item>
    ///     </list>
    ///     <para><b>How the weave works.</b> The attributed method is a compiler-generated stub; the
    ///     real code — and every egress — lives in the state machine's <c>MoveNext</c>. The weaver
    ///     gives it ONE scope for the whole logical invocation (held in a weaver-added state-machine
    ///     field), UNINSTALLED before each await's continuation is scheduled and re-installed on
    ///     whatever thread resumes — so temps stay alive while an awaited callee still uses them, and
    ///     everything is reclaimed at SetResult/SetException (async), at the final <c>MoveNext</c>
    ///     (async iterators), with results and <c>yield return</c>ed elements routed through the same
    ///     <c>Returns</c>/<c>YieldTo</c> egress. A NON-async method returning <c>Task</c>/<c>ValueTask</c>
    ///     yields a completed task's result immediately and DEFERS the scope's disposal to an
    ///     incomplete task's completion (the in-flight callee may still hold tracked temps); an
    ///     incomplete <c>ValueTask</c> is <c>Preserve()</c>d — the caller receives the multi-observable
    ///     form.</para>
    ///     <para><b>SYNCHRONOUS iterators stay on <see cref="NDScopedAttribute"/>.</b> An
    ///     <c>IEnumerable&lt;T&gt;</c>/<c>IEnumerator&lt;T&gt;</c> (<c>yield return</c> without
    ///     <c>await</c>) also compiles to a state machine and uses the same invocation-scope seam, but
    ///     it is not asynchronous — it is woven by <c>[NDScoped]</c>. Marking a synchronous iterator
    ///     <c>[NDScopedAsync]</c> is a build ERROR (NDW010); marking an async method, async iterator,
    ///     or Task-returning method <c>[NDScoped]</c> is the mirror-image ERROR (NDW009). Each error
    ///     names the correct attribute — a method has exactly one scoping model, and choosing the
    ///     wrong attribute never silently ships an unwoven method.</para>
    ///     <para>The weaver REJECTS (build error NDW003) a shape whose egress it cannot see — an
    ///     UNSUPPORTED carrier (a bespoke reference type, a collection, or a result struct that does
    ///     NOT implement <see cref="INDArrayCarrier"/>) as the async RESULT or as the <c>T</c> inside a
    ///     <c>Task&lt;T&gt;</c>; scope those by hand (NDW004 = an unrecognized, non-C# state-machine
    ///     shape; NDW008 = the referenced NumSharp predates the async seam).</para>
    ///     <para>A method whose body already opens an <see cref="NDScope"/> is skipped (idempotence).</para>
    ///     <para><b>Inherited by overrides and implementations</b> exactly like
    ///     <see cref="NDScopedAttribute"/>: on a virtual, abstract or interface member it is the contract
    ///     every override/implementation is woven under (an <c>async</c> override of a non-async
    ///     <c>Task</c>-returning declaration, or the reverse, each weave through their own shape — the
    ///     attribute names the model, the override supplies the body). An override's own
    ///     <c>[NDScoped]</c>/<c>[NDScopedAsync]</c>/<c>[NDScopedCovered]</c> wins over the inherited one;
    ///     a body-less attributed declaration is the contract, not an NDW005.</para>
    ///     <para>PUBLIC because the attribute is consumer-facing: a project that installs the
    ///     <c>NumSharp.Build</c> package marks its own async/Task composition methods with it. Without
    ///     the package the attribute is inert metadata — the method runs unscoped, the pre-weave
    ///     finalizer-backstop behaviour.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class NDScopedAsyncAttribute : Attribute
    {
    }
}
