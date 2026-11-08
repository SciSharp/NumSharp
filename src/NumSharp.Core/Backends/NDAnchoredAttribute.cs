using System;

namespace NumSharp
{
    /// <summary>
    ///     Marks a method that opens NO <see cref="NDScope"/> as an <b>operand anchor</b>: at build time the NumSharp IL
    ///     weaver roots each of its parameters that can reach an <see cref="NDArray"/> (and <c>this</c>, for an instance
    ///     method of a class) from the method's entry to every return, so an operand the caller passed as a temporary
    ///     cannot be collected while a kernel inside the call still reads its memory. Use it on an unscoped entry point
    ///     that hands an operand's raw address to a kernel; a <see cref="NDScopedAttribute"/> method already gets the
    ///     same rooting and must not carry this attribute too.
    /// </summary>
    /// <remarks>
    ///     <para><b>The hazard it closes ("lost GC anchors").</b> In optimized code the JIT reports a parameter to the GC
    ///     only up to its last use. A method that reads <c>a.Address</c> (or <c>Storage.Address</c>) and then runs a
    ///     kernel has no later use of <c>a</c>, so a caller that passes a temporary (<c>np.copyto(t, np.arange(n))</c>)
    ///     leaves the operand reachable from nothing during the kernel: a collection finalizes it, its ARC reference is
    ///     abandoned, and the buffer is pooled or freed under the kernel (wrong values, spurious errors, access
    ///     violations). The lifetime oracle (<c>test/NumSharp.Tests.Oracle/Fuzz/Lifetime</c>) reproduces the class.</para>
    ///     <para><b>What the weaver injects.</b> At entry, each rooted parameter is copied into a shadow local (a reference
    ///     parameter as is; a struct that carries an NDArray, such as a tuple of arrays, boxed once); before every
    ///     <c>ret</c>, each shadow is passed to <see cref="NDScope.Anchor"/>, a no-code intrinsic that extends its
    ///     lifetime to that point. Nothing is opened, tracked, disposed or allocated (except the box of an NDArray-carrying
    ///     struct), so the attribute fits hot entry points where a scope would cost too much. A reassigned parameter keeps
    ///     its ORIGINAL value rooted: the shadow is taken before the body runs.</para>
    ///     <para><b>What it does NOT root.</b> Temporaries the method itself builds are not tracked (no scope is open):
    ///     a method that converts an operand (<c>x = x.astype(...)</c>) and then reads the converted array's address
    ///     after its last use needs <see cref="NDScopedAttribute"/>, whose scope tracks and roots everything constructed
    ///     during the call. Parameters that cannot reach an NDArray (numbers, enums, strings, <see cref="Type"/>s,
    ///     spans and other by-ref-like structs) and <c>ref</c>/<c>out</c>/<c>in</c> parameters are left alone.</para>
    ///     <para><b>Rejected shapes</b> (build error NDW019): a method without a body, an <c>async</c> method or an
    ///     iterator (its parameters are already hoisted into the state machine, which the caller holds), a body with a
    ///     tail call, a method that also carries <see cref="NDScopedAttribute"/>/<see cref="NDScopedAsyncAttribute"/>,
    ///     and a method with nothing to root.</para>
    ///     <para><b>Not inherited.</b> The attribute applies to the method that declares it; mark each override that
    ///     needs rooting. Without the <c>NumSharp.Build</c> weaver the attribute is inert metadata (the analyzer warns
    ///     with NDW013), and a body that already calls <see cref="NDScope.Anchor"/> is left untouched (idempotence).</para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor, AllowMultiple = false, Inherited = false)]
    public sealed class NDAnchoredAttribute : Attribute
    {
    }
}
