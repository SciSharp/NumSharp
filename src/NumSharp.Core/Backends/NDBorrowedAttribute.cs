using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a field, property, class or struct as <b>borrowing</b> the <see cref="NDArray"/>(s) it
    ///     references — they are owned and disposed by someone else — so the compile-time ownership
    ///     analyzer (<c>NumSharp.Build.Analyzer</c>) must not demand that the containing type dispose them.
    /// </summary>
    /// <remarks>
    ///     <para><b>What the analyzer enforces without it.</b> A type that STORES NDArrays — an
    ///     instance field or auto-property typed <see cref="NDArray"/>, an <c>NDArray[]</c>, a tuple or
    ///     collection of NDArrays, an <see cref="INDArrayCarrier"/> result struct, or another type that
    ///     itself stores NDArrays (ownership is contagious) — owns their pooled buffers, so it must
    ///     implement <see cref="IDisposable"/> (or <see cref="IAsyncDisposable"/>) and dispose every
    ///     such member from its <c>Dispose</c> path. The analyzer reports <c>NDW016</c> when the type is
    ///     not disposable at all and <c>NDW017</c> for each storing member its <c>Dispose</c> never
    ///     reaches; and it treats an instance of such a disposable type like an NDArray in the per-method
    ///     leak pass (<c>NDW012</c>): constructing one and dropping it is a leak.</para>
    ///
    ///     <para><b>What this attribute asserts.</b> On a <b>member</b>: the value it references is an
    ///     input the containing type was handed (a view over a caller's array, a shared lookup table, an
    ///     operand an iterator walks) — rule R2 of <c>DISPOSAL-GUIDELINES.md</c>, "never dispose an input
    ///     you were given" — so the member is excluded from the type's ownership set. On a <b>type</b>:
    ///     every NDArray the type references is borrowed, so the type is exempt from <c>NDW016</c>/
    ///     <c>NDW017</c> outright and is never treated as an NDArray-owning value by <c>NDW012</c>, even
    ///     when it is disposable (a disposable that owns unmanaged state of its own but only borrows its
    ///     arrays — NumSharp's <c>np.nditer</c> is the archetype).</para>
    ///
    ///     <para><b>Runtime-inert, analyzer-only.</b> Like <see cref="NDScopedCoveredAttribute"/> this
    ///     attribute changes nothing at runtime: nothing is woven, no scope is opened, nothing is disposed
    ///     or kept alive by it. It is the author's ownership statement, and a wrong one has the same
    ///     consequence as a wrong hand-written <c>using</c>: an array nobody disposes falls back to the
    ///     finalizer, or an array two owners dispose is freed under one of them. It never contains the
    ///     <c>NDScopedAttribute</c>/<c>NDScopedAsyncAttribute</c> names, so a project that uses only this
    ///     attribute draws no weaver-missing guard (<c>NDW013</c>).</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property | AttributeTargets.Class | AttributeTargets.Struct,
        AllowMultiple = false, Inherited = false)]
    public sealed class NDBorrowedAttribute : Attribute
    {
    }
}
