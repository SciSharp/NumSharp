using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluate an expression tree over NDArrays in ONE fused pass —
        ///     no intermediate arrays, one read of each operand, one write of
        ///     the result (NumSharp extension; the NumPy-ecosystem equivalent
        ///     is numexpr.evaluate).
        /// </summary>
        /// <param name="expr">
        ///     Expression with embedded array leaves, or with positional <see cref="NDExpr.Input"/> leaves that
        ///     <paramref name="operands"/> fills. NDArrays convert implicitly, so one cast lights up the whole operator
        ///     set:
        ///     <code>
        ///     NDArray r = np.evaluate((NDExpr)a * b + 2);                              // a*b+2 fused
        ///     NDArray d = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
        ///     NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b));                     // one-pass sum(a*b)
        ///     NDArray p = np.evaluate(NDExpr.Input(0) * NDExpr.Input(1), new[] { a, b }); // positional operands
        ///     NDArray o = np.evaluate((NDExpr)a * b, @out: x);                         // into x
        ///     </code>
        ///     A repeated NDArray reference becomes ONE iterator operand.
        /// </param>
        /// <param name="operands">
        ///     The operands of a tree built over positional <see cref="NDExpr.Input"/> leaves, by position, or null (the
        ///     default) for a tree whose leaves embed their arrays. A text operand is refused.
        /// </param>
        /// <param name="out">
        ///     Optional pre-allocated result (ufunc out= rules: joins the
        ///     broadcast but is never stretched; cast from the resolved dtype
        ///     under <paramref name="casting"/>; may alias an input — overlap-safe).
        ///     Mutually exclusive with <paramref name="dtype"/>. Pass it by name: the second position is
        ///     <paramref name="operands"/>.
        /// </param>
        /// <param name="where">
        ///     Optional boolean write mask (ufunc where= convention, plan P4.5): the fused kernel writes the
        ///     result only where the mask is True, so masked-off slots of <paramref name="out"/> keep their
        ///     prior contents (and without <paramref name="out"/> the unmasked slots of the fresh result are
        ///     left uninitialised, exactly as NumPy's ufuncs leave them). It broadcasts with the inputs and
        ///     <paramref name="out"/> but never stretches a provided <paramref name="out"/>, and must be a
        ///     boolean array. NOT supported on a reduction tree (a masked reduction is a different operation).
        /// </param>
        /// <param name="dtype">
        ///     Optional result dtype — an implicit root cast (plan P4.5): the tree still COMPUTES at its
        ///     natural per-node NEP50 dtypes and the RESULT is cast to this, so
        ///     <c>np.evaluate(expr, dtype: X)</c> equals <c>np.evaluate(expr).astype(X)</c> in one fused
        ///     pass. NOT an accumulator/loop dtype — for a reduction tree it is rejected (a reduction fixes
        ///     its own accumulator dtype); cast the reduction result instead. Mutually exclusive with
        ///     <paramref name="out"/> (which already fixes the result dtype).
        /// </param>
        /// <param name="casting">
        ///     The cast rule enforced when a caller <paramref name="out"/> has a different dtype than the
        ///     resolved result ('no'/'equiv'/'safe'/'same_kind'/'unsafe'); default (null) is NumPy's ufunc
        ///     <c>same_kind</c>. With no <paramref name="out"/> it only validates the string.
        /// </param>
        /// <param name="order">
        ///     The memory layout of a FRESH result (plan P4.5): 'K' (default) keeps today's heuristic
        ///     (column-major only when every input is strictly F-contiguous, else row-major), 'C' forces
        ///     row-major, 'F' forces column-major, 'A' resolves like 'K'. A caller <paramref name="out"/>
        ///     keeps its own layout, and a reduction (a fixed-layout result) rejects a non-'K' order.
        /// </param>
        /// <returns>
        ///     The evaluated array at the tree's NumPy result_type (or <paramref name="dtype"/> if given) —
        ///     dtypes otherwise match the equivalent unfused NumPy expression node-for-node (NEP50,
        ///     including weak python-scalar literals). Root reductions
        ///     (<see cref="NDExpr.Sum(NDExpr)"/> / Prod / Min / Max / Mean)
        ///     return a 0-d scalar array.
        /// </returns>
        /// <exception cref="ArgumentException">Both <paramref name="dtype"/> and <paramref name="out"/> were given; <paramref name="out"/> is not reachable from the result dtype under <paramref name="casting"/>; <paramref name="where"/> is not a boolean array (NumPy's verbatim "Cannot cast … to dtype('bool') according to the rule 'safe'"); or <paramref name="operands"/> is empty.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; <paramref name="casting"/> is not a legal rule, or <paramref name="order"/> is not one of 'C'/'F'/'A'/'K'.</exception>
        /// <exception cref="StringComputationNotSupportedException">A text operand among <paramref name="operands"/> (a fused expression computes on numbers).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="expr"/> is null, or an entry of <paramref name="operands"/> is.</exception>
        /// <exception cref="System.NotSupportedException"><paramref name="dtype"/>, a non-'K' <paramref name="order"/>, or a <paramref name="where"/> mask was given for a reduction tree; or the operands' engine has no fused evaluation.</exception>
        /// <remarks>
        ///     One overload, its optional parameters defaulted (the np-layer dtype plan's D11, D19: a function NumPy does not
        ///     have keeps one overload). The whole call is the engine member
        ///     <see cref="TensorEngine.Evaluate(NDExpr, NDArray[], NDArray, NDArray, DType, string, char)"/>, which takes the
        ///     keywords as given and checks them in this order: a read-only <paramref name="out"/>, a text operand, then the
        ///     keyword bundle.
        /// </remarks>
        public static NDArray evaluate(NDExpr expr, NDArray[] operands = null, NDArray @out = null, NDArray where = null,
            DType dtype = null, string casting = null, char order = 'K')
            => ResolveEngine(expr, operands).Evaluate(expr, operands, @out, where, dtype, casting, order);

        /// <summary>
        ///     np.dot(a, b) dispatches on <c>a.TensorEngine</c>; np.evaluate follows the same rule
        ///     with the first array the tree references (ARCHITECTURE.md P2), so an array bound to
        ///     an alternative engine is evaluated by that engine. A constant-only tree names no
        ///     array, so no operand decides and <see cref="TensorEngine.Default"/> evaluates it (its
        ///     validation rejects it).
        /// </summary>
        /// <param name="expr">The tree, or null (the engine member refuses it).</param>
        /// <param name="operands">The positional operands, or null when the tree's own leaves hold the arrays.</param>
        /// <returns>The engine of the first operand (or the tree's first array leaf), else <see cref="TensorEngine.Default"/>.</returns>
        private static TensorEngine ResolveEngine(NDExpr expr, NDArray[] operands)
        {
            NDArray first = operands is { Length: > 0 } ? operands[0] : expr?.FirstArray();
            return first is null ? TensorEngine.Default : first.TensorEngine;
        }
    }
}
