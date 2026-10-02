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
        ///     Expression with embedded array leaves. NDArrays convert
        ///     implicitly, so one cast lights up the whole operator set:
        ///     <code>
        ///     NDArray r = np.evaluate((NDExpr)a * b + 2);            // a*b+2 fused
        ///     NDArray d = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
        ///     NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b));   // one-pass sum(a*b)
        ///     </code>
        ///     A repeated NDArray reference becomes ONE iterator operand.
        /// </param>
        /// <param name="out">
        ///     Optional pre-allocated result (ufunc out= rules: joins the
        ///     broadcast but is never stretched; cast from the resolved dtype
        ///     under <paramref name="casting"/>; may alias an input — overlap-safe).
        ///     Mutually exclusive with <paramref name="dtype"/>.
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
        /// <exception cref="ArgumentException">Both <paramref name="dtype"/> and <paramref name="out"/> were given; <paramref name="out"/> is not reachable from the result dtype under <paramref name="casting"/>; or <paramref name="where"/> is not a boolean array (NumPy's verbatim "Cannot cast … to dtype('bool') according to the rule 'safe'").</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not a legal rule, or <paramref name="order"/> is not one of 'C'/'F'/'A'/'K'.</exception>
        /// <exception cref="System.NotSupportedException"><paramref name="dtype"/>, a non-'K' <paramref name="order"/>, or a <paramref name="where"/> mask was given for a reduction tree.</exception>
        public static NDArray evaluate(NDExpr expr, NDArray @out = null, NDArray where = null,
            DType dtype = null, string casting = null, char order = 'K')
        {
            // The fused inner loop writes @out through the iterator, bypassing the guarded setters
            // (same class of gap as the ufunc out= path) — reject a non-writeable target up front,
            // as the ufuncs this fuses do.
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            var options = BuildEvaluateOptions(dtype, casting, order, @out is not null, where);
            return ResolveEngine(expr, null).Evaluate(expr, @out, options);
        }

        /// <summary>
        ///     np.dot(a, b) dispatches on <c>a.TensorEngine</c>; np.evaluate follows the same rule
        ///     with the first array the tree references (ARCHITECTURE.md P2), so an array bound to
        ///     an alternative engine is evaluated by that engine. A constant-only tree names no
        ///     engine and falls to the default, whose validation rejects it.
        /// </summary>
        private static TensorEngine ResolveEngine(NDExpr expr, NDArray[] operands)
        {
            NDArray first = operands is { Length: > 0 } ? operands[0] : expr?.FirstArray();
            return first?.TensorEngine ?? BackendFactory.GetEngine();
        }

        /// <summary>
        ///     Evaluate an expression built over positional
        ///     <see cref="NDExpr.Input"/> leaves against an explicit operand
        ///     list: <c>np.evaluate(NDExpr.Input(0) * NDExpr.Input(1), new[] { a, b })</c>.
        /// </summary>
        /// <inheritdoc cref="evaluate(NDExpr, NDArray, NDArray, DType, string, char)" path="/param[@name='out' or @name='where' or @name='dtype' or @name='casting' or @name='order']"/>
        public static NDArray evaluate(NDExpr expr, NDArray[] operands, NDArray @out = null, NDArray where = null,
            DType dtype = null, string casting = null, char order = 'K')
        {
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            var options = BuildEvaluateOptions(dtype, casting, order, @out is not null, where);
            return ResolveEngine(expr, operands).Evaluate(expr, operands, @out, options);
        }

        /// <summary>
        ///     Validate and RESOLVE the <c>np.evaluate</c> keyword bundle at the API boundary (plan P4.5),
        ///     so the engine only ever sees a legal, resolved <see cref="NDEvaluateOptions"/>. A bad casting
        ///     string or order char raises HERE (before any work, and independent of the tree's shape), and
        ///     the <c>dtype</c>/<c>out</c> conflict is caught before either reaches the engine.
        /// </summary>
        /// <param name="dtype">The requested result dtype (an implicit root cast), or null for the natural type.</param>
        /// <param name="casting">The out= cast rule string, or null for <c>same_kind</c>.</param>
        /// <param name="order">The fresh-result order char ('C'/'F'/'A'/'K', case-insensitive).</param>
        /// <param name="hasOut">Whether a caller <c>out=</c> was supplied (which fixes the result dtype, conflicting with <paramref name="dtype"/>).</param>
        /// <param name="where">The optional boolean write mask (validated for boolean-ness at the engine, alongside the reduction-tree rejection), or null for an unmasked pass.</param>
        /// <returns>The resolved keyword bundle to thread to the engine.</returns>
        /// <exception cref="ArgumentException">Both <paramref name="dtype"/> and an <c>out=</c> were given.</exception>
        /// <exception cref="ValueError"><paramref name="casting"/> is not a legal rule, or <paramref name="order"/> is not one of 'C'/'F'/'A'/'K'.</exception>
        private static NDEvaluateOptions BuildEvaluateOptions(DType dtype, string casting, char order, bool hasOut, NDArray where = null)
        {
            // out= already pins the result dtype, so a second pin via dtype= is contradictory — reject the
            // pair rather than silently double-casting (NumPy's ufunc lets both differ because dtype= is a
            // COMPUTE dtype there; evaluate's dtype= is a RESULT cast, so it collapses onto out=).
            if (dtype is not null && hasOut)
                throw new ArgumentException(
                    "np.evaluate: specify either dtype= or out=, not both — out= already fixes the result dtype.");

            // The order char is validated even for a reduction tree (a bad order= is a bad argument
            // regardless of what the tree is); the engine then applies it only to a fresh elementwise result.
            char o = char.ToUpperInvariant(order);
            if (o != 'C' && o != 'F' && o != 'A' && o != 'K')
                throw new ValueError($"order must be one of 'C', 'F', 'A', 'K' (got '{order}')");

            // ParseCasting raises NumPy's verbatim "casting must be one of …" for a bad string; null keeps
            // the same_kind default the struct supplies. The where= mask's boolean-ness (and its rejection
            // on a reduction tree) is checked at the engine, next to the identical dtype=/order= guards, so
            // the whole reduction-tree keyword contract lives in one place.
            NPY_CASTING? c = casting is null ? (NPY_CASTING?)null : DTypeCasting.ParseCasting(casting);
            return new NDEvaluateOptions(dtype?.typecode, c, order, where);
        }
    }
}
