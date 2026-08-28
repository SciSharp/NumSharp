using NumSharp;

namespace NDScoping
{
    /// <summary>
    ///     A tuple-standin result struct that carries two <see cref="NDArray"/>s — the L-carrier
    ///     layer's subject. It opts into weaving by implementing <see cref="INDArrayCarrier"/>:
    ///     <see cref="INDArrayCarrier.YieldTo"/> hands each member back through the scope, so a
    ///     boundary method returning a <c>PairResult</c> reclaims its temporaries while its result
    ///     members survive — exactly like a bare <see cref="NDArray"/> return.
    ///     <para>
    ///     The interface is implemented EXPLICITLY (it stays off the public API); the weaver invokes
    ///     it through a boxing-free <c>constrained.callvirt</c> at each return. The members are yielded
    ///     from INSIDE the struct because a struct's own method can read its private fields while an
    ///     enclosing type's woven method cannot (the CLR grants nested→enclosing private access, not the
    ///     reverse) — the reason the interface exists at all. A hand-written scope reaches YieldTo the
    ///     same way, via <c>((INDArrayCarrier)result).YieldTo(scope)</c>.
    ///     </para>
    /// </summary>
    public readonly struct PairResult : INDArrayCarrier
    {
        public readonly NDArray First;
        public readonly NDArray Second;

        public PairResult(NDArray first, NDArray second)
        {
            First = first;
            Second = second;
        }

        void INDArrayCarrier.YieldTo(NDScope scope)
        {
            scope.Returns(First);
            scope.Returns(Second);
        }
    }
}
