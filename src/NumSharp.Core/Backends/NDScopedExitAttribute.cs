using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a <b>by-value parameter</b> as one the callee RETAINS — a reference it keeps past the
    ///     call (stores in a field/property, adds to a long-lived collection, captures in a closure/task
    ///     that outlives the call). At build time the NumSharp IL weaver detaches the argument from
    ///     whatever <see cref="NDScope"/> tracks it, so the CALLER's scope will NOT reclaim an
    ///     <see cref="NDArray"/> the callee is still holding.
    /// </summary>
    /// <remarks>
    ///     <para><b>The hazard it closes.</b> A <see cref="NDScope"/> disposes every array constructed
    ///     under it that was not yielded via <see cref="NDScope.Returns{T}(T)"/> — regardless of any
    ///     external reference. So a <see cref="NDScopedAttribute"/> method that hands a freshly built
    ///     array to something that KEEPS it (rather than merely reading it) would have that array
    ///     reclaimed at scope exit while the retainer still points at it — a use-after-free. Marking the
    ///     retaining parameter <c>[NDScopedExit]</c> makes the callee remove the argument from the
    ///     ambient scope, so it survives (falling to the ordinary finalizer backstop unless the retainer
    ///     disposes it — <b>survival, not eager reclamation</b>).</para>
    ///     <para><b>How the weave works.</b> The attributed method is rewritten to call
    ///     <see cref="NDScope.Detach(NDArray)"/> (or the <see cref="NDArray"/>-array /
    ///     <see cref="System.Runtime.CompilerServices.ITuple"/> overload) on the parameter at the START
    ///     of the visible method body — for an <c>async</c>/iterator method that is the compiler STUB,
    ///     which runs synchronously on the caller's thread at the call, exactly where the ambient scope
    ///     is the caller's. Because <see cref="NDScope"/> is <c>[ThreadStatic]</c> and the callee runs
    ///     within the caller's scope, <c>Detach</c> reaches into the caller's scope with no argument
    ///     re-plumbing at the call site. It is a no-op when the argument is untracked (no ambient scope,
    ///     or the array was constructed outside one), so an <c>[NDScopedExit]</c> method is always safe
    ///     to call.</para>
    ///     <para><b>What "any kind of setter / method" covers.</b> A property setter is a method whose
    ///     <c>value</c> parameter can carry the attribute (<c>[param: NDScopedExit]</c>), so
    ///     <c>obj.Prop = a</c> is covered; likewise any method or constructor parameter you own. A RAW
    ///     public-field store (<c>obj.field = a</c>) has no parameter to annotate and is NOT covered —
    ///     route it through a property setter, or detach by hand with <see cref="NDScope.Detach(NDArray)"/>.
    ///     A parameter you cannot annotate (a BCL sink such as <c>List&lt;NDArray&gt;.Add</c>) likewise
    ///     needs a hand <c>Detach</c>.</para>
    ///     <para><b>Supported parameter shapes</b> (the same NDArray-carrying shapes
    ///     <see cref="NDScope.Returns{T}(T)"/> yields): <see cref="NDArray"/> / <c>NDArray&lt;T&gt;</c>,
    ///     <see cref="NDArray"/><c>[]</c>, and any <c>ValueTuple</c>/<c>Tuple</c> of NDArrays. Anything
    ///     else — a <c>ref</c>/<c>out</c>/<c>in</c> parameter, a scalar, a bare
    ///     <c>IArraySlice</c>/<c>UnmanagedStorage</c> (never scope-tracked), or an
    ///     <see cref="INDArrayCarrier"/> result struct — is a build ERROR (NDW014); detach those by hand.</para>
    ///     <para>Orthogonal to <see cref="NDScopedAttribute"/>/<see cref="NDScopedAsyncAttribute"/>: a
    ///     method may carry an <c>[NDScopedExit]</c> parameter with or without a method-level scope
    ///     attribute (a pure "retain this argument" method needs no scope of its own). A body that
    ///     already opens an <see cref="NDScope"/> by hand is left untouched (the hand author owns its
    ///     detaches too). PUBLIC because the attribute is consumer-facing; without the
    ///     <c>NumSharp.Build</c> package it is inert metadata (the argument is not detached — the
    ///     pre-weave behaviour, where a caller must avoid handing a scoped temp to a retainer).</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
    public sealed class NDScopedExitAttribute : Attribute
    {
    }
}
