using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a method (or a property accessor) as an <see cref="NDScope"/> boundary: at build
    ///     time the NumSharp IL weaver (<c>tools/NumSharp.Weaver</c>) injects the exact code the
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
    ///     (<see cref="NDScope.Returns{T}(T[])"/> for tuple-style array returns), and every
    ///     <c>out NDArray</c> parameter's final value yielded before each successful return.</para>
    ///     <para>The weaver REJECTS (build error) shapes whose egress it cannot see:
    ///     <c>ref NDArray</c> parameters, carrier-struct returns (<c>UniqueResult</c>,
    ///     ValueTuples of arrays — scope those by hand), iterators and async methods.</para>
    ///     <para>A method whose body already opens an <see cref="NDScope"/> is skipped
    ///     (idempotence), so hand-scoped code may carry the attribute without double-wrapping.</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    internal sealed class NDScopedAttribute : Attribute
    {
    }
}
