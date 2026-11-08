using NumSharp.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     NumSharp's <c>!</c> operator: NumPy's <c>np.logical_not(a)</c> as a typed bool array (Python has no <c>!</c>;
        ///     this is its logical_not).
        /// </summary>
        /// <param name="self">The operand (any dtype, any layout).</param>
        /// <returns>The truth negation, an <see cref="NDArray{T}"/> of bool over np.logical_not's result.</returns>
        /// <exception cref="TypeError"><paramref name="self"/> holds text: logical_not has no string loop (NumPy's
        ///     "ufunc 'logical_not' not supported for the input types, ..." text).</exception>
        /// <remarks>
        ///     <para>The whole call is np.logical_not's (the np-layer dtype plan, P1c): its text refusal, the bool route
        ///     (invert) and the comparison with 0 for numbers (a NaN is true, -0.0 false). The operator held a raw scalar
        ///     loop of its own for C-contiguous operands (<c>x.Equals(default)</c> per element, behind its own text guard),
        ///     which the comparison kernels outrun: float64 at 100K elements took ~0.5 ms there.</para>
        ///     <para>Scope: MakeGeneric&lt;bool&gt;() mints a typed ALIAS over the result's storage (its own ARC ref); the
        ///     untyped result wrapper would then be a strand reclaimable only by a future GC and finalizer pass (measured:
        ///     one bucketed buffer escaped per <c>!arr</c> call). The [NDScoped] weaver tracks it, yields the alias and
        ///     disposes the wrapper at exit. <paramref name="self"/> is an input constructed before the scope opens, so it
        ///     is never tracked.</para>
        /// </remarks>
        [NDScoped]
        public static NDArray<bool> operator !(NDArray self) => np.logical_not(self).MakeGeneric<bool>();
    }
}
