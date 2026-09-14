using System;

// =============================================================================
// NDExpr.Combinators.cs — C6 macro / decision combinators (docs/plans/ndexpr-capabilities.md §7)
// =============================================================================
//
// A vocabulary of readable, high-level nodes — activations, selectors, boolean
// logic, predicates, directional/sign helpers, and multi-way decisions — that a
// consumer would otherwise re-spell as a composition at every call site.
//
// EVERY combinator here is a PURE COMPOSITION of the primitive nodes already in
// NDExpr.cs (Where / comparisons / Min / Max / Abs / the ufunc unaries / arithmetic).
// Three consequences follow, and they are the whole reason this file carries no IL:
//
//   • It fuses. A combinator returns an ordinary tree, so it folds into the SAME
//     single np.evaluate pass as its surrounding expression — no intermediate
//     array, each operand read once (see the fused-shell contract in
//     docs/plans/ndexpr-capabilities.md §1).
//   • It is correct by construction. Its parity contract is its composition's
//     unfused NumPy chain; there is no new kernel to get wrong. The gate is the
//     metamorphic one (fused combinator == the eager np.* chain), NumSharp-only
//     and Python-free — NDExprCombinatorTests.
//   • It inherits SIMD. Whatever vector path the underlying Where/Min/Max/arith
//     nodes have (blend, hardware min/max, lane arithmetic) is used unchanged;
//     nothing here forces the scalar path that a Call node would.
//
// DELIBERATELY ABSENT — reserved for Phase 4 as first-class (vectorizable) ufunc
// nodes, NOT composed here so this file never collides with them when P4 lands:
// fmax/fmin, copysign, nextafter, logaddexp, the shifts, and the np.select /
// np.clip nodes (docs/plans/ndexpr-evaluate.md §0.1, "P4 coverage"). The multi-way
// selector below is therefore named Switch, and one-sided clamps ride Min/Max.
//
// NEP50 note that bites here: `bool + bool` is logical OR, not a count. Anything
// that COUNTS (Bucketize) must sum integer Where(cond, 1, 0) values, never bools.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    public abstract partial class NDExpr
    {
        // ===================================================================
        // Selection & masking
        // ===================================================================

        /// <summary>
        /// Inlined ternary: <paramref name="a"/> where <paramref name="cond"/> is nonzero, else
        /// <paramref name="b"/> (== <c>cond ? a : b</c>, per element). An alias of
        /// <see cref="Where"/> spelled as an if — the readable name for the most common selector.
        /// </summary>
        /// <param name="cond">Selector; any nonzero value is "true" (NumPy nonzero-test at the condition's own dtype).</param>
        /// <param name="a">Value chosen where <paramref name="cond"/> is true.</param>
        /// <param name="b">Value chosen where <paramref name="cond"/> is false.</param>
        /// <returns>A fused select node; its dtype is <c>result_type(a, b)</c>.</returns>
        public static NDExpr If(NDExpr cond, NDExpr a, NDExpr b) => Where(cond, a, b);

        /// <summary>
        /// Negated ternary: <paramref name="a"/> where <paramref name="cond"/> is FALSE, else
        /// <paramref name="b"/>. Swaps the arms rather than negating the condition, so it costs no
        /// extra node over <see cref="If"/>.
        /// </summary>
        /// <param name="cond">Selector; nonzero is "true".</param>
        /// <param name="a">Value chosen where <paramref name="cond"/> is false.</param>
        /// <param name="b">Value chosen where <paramref name="cond"/> is true.</param>
        /// <returns>A fused select node equivalent to <c>If(!cond, a, b)</c>.</returns>
        public static NDExpr IfNot(NDExpr cond, NDExpr a, NDExpr b) => Where(cond, b, a);

        /// <summary>
        /// One-armed gate: <paramref name="then"/> where <paramref name="cond"/> is true, else 0 — the
        /// masking idiom. Preserves non-numeric dtypes (unlike <c>cond * then</c>, which would demand
        /// arithmetic); the false arm is a WEAK literal 0 that adopts <paramref name="then"/>'s dtype.
        /// </summary>
        /// <param name="cond">Gate; nonzero is "true".</param>
        /// <param name="then">Value passed through where the gate is open.</param>
        /// <returns>A fused select whose false arm is 0.</returns>
        public static NDExpr When(NDExpr cond, NDExpr then) => Where(cond, then, Const(0));

        /// <summary>One-armed negated gate: <paramref name="then"/> where <paramref name="cond"/> is FALSE, else 0.</summary>
        /// <param name="cond">Gate; nonzero is "true".</param>
        /// <param name="then">Value passed through where the gate is closed.</param>
        /// <returns>A fused select whose true arm is 0.</returns>
        public static NDExpr Unless(NDExpr cond, NDExpr then) => Where(cond, Const(0), then);

        /// <summary>
        /// Multi-way switch: the value of the FIRST case whose condition is true, else
        /// <paramref name="default"/> (np.select semantics). Folds to nested <see cref="Where"/> nodes
        /// with case 0 outermost, so earlier cases win ties. Named Switch, not Select, because Phase 4
        /// reserves an <c>NDExpr.Select</c> for the condlist/choicelist form.
        /// </summary>
        /// <param name="default">Fallback when no case matches.</param>
        /// <param name="cases">Ordered (condition, value) pairs; the first true condition wins.</param>
        /// <returns>A right-folded chain of fused selects.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="default"/> is null, or a case tuple holds a null.</exception>
        public static NDExpr Switch(NDExpr @default, params (NDExpr cond, NDExpr value)[] cases)
        {
            if (@default is null) throw new ArgumentNullException(nameof(@default));
            if (cases is null) throw new ArgumentNullException(nameof(cases));
            // Fold from the last case inward so case[0] is the OUTERMOST test (first match wins).
            var acc = @default;
            for (int i = cases.Length - 1; i >= 0; i--)
            {
                if (cases[i].cond is null || cases[i].value is null)
                    throw new ArgumentNullException(nameof(cases), $"case {i} has a null condition or value.");
                acc = Where(cases[i].cond, cases[i].value, acc);
            }

            return acc;
        }

        /// <summary>
        /// Integer-indexed multiplexer: <paramref name="values"/><c>[k]</c> where <c>index == k</c>,
        /// else 0. Nested selects with index 0 outermost; a non-integer <paramref name="index"/> is
        /// compared for equality at its own dtype (so <c>1.0 == 1</c> selects slot 1).
        /// </summary>
        /// <param name="index">Selector compared for equality to 0, 1, 2, ….</param>
        /// <param name="values">Choices, one per index; an out-of-range index yields 0.</param>
        /// <returns>A right-folded chain of fused selects.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="index"/> is null, or <paramref name="values"/> holds a null.</exception>
        public static NDExpr Mux(NDExpr index, params NDExpr[] values)
        {
            if (index is null) throw new ArgumentNullException(nameof(index));
            if (values is null) throw new ArgumentNullException(nameof(values));
            NDExpr acc = Const(0);
            // Fold from the last value inward so index == 0 is the OUTERMOST test.
            for (int k = values.Length - 1; k >= 0; k--)
            {
                if (values[k] is null) throw new ArgumentNullException(nameof(values), $"value {k} is null.");
                acc = Where(Equal(index, Const(k)), values[k], acc);
            }

            return acc;
        }

        // ===================================================================
        // Clamp & saturation
        // ===================================================================

        /// <summary>Lower clamp: <c>max(x, lo)</c>. NaN in <paramref name="x"/> propagates (np.maximum semantics).</summary>
        /// <param name="x">Value to clamp.</param><param name="lo">Inclusive floor.</param>
        /// <returns>A fused np.maximum node.</returns>
        public static NDExpr ClampMin(NDExpr x, NDExpr lo) => Max(x, lo);

        /// <summary>Upper clamp: <c>min(x, hi)</c>. NaN in <paramref name="x"/> propagates.</summary>
        /// <param name="x">Value to clamp.</param><param name="hi">Inclusive ceiling.</param>
        /// <returns>A fused np.minimum node.</returns>
        public static NDExpr ClampMax(NDExpr x, NDExpr hi) => Min(x, hi);

        /// <summary>Saturating clamp to the unit interval <c>[0, 1]</c> — the shader/ML "saturate".</summary>
        /// <param name="x">Value to clamp.</param>
        /// <returns>A fused <see cref="Clamp"/> to [0, 1].</returns>
        public static NDExpr Saturate(NDExpr x) => Clamp(x, Const(0.0), Const(1.0));

        // ===================================================================
        // Robustness (NaN / finite)
        // ===================================================================

        /// <summary>
        /// Replace NaN with <paramref name="fallback"/>, pass every other value through unchanged (the
        /// NaN case of np.nan_to_num). Non-float inputs are never NaN, so they pass through.
        /// </summary>
        /// <param name="x">Value stream to sanitise.</param><param name="fallback">Replacement for NaN lanes.</param>
        /// <returns>A fused select on <c>isnan(x)</c>.</returns>
        public static NDExpr NanTo(NDExpr x, NDExpr fallback) => Where(IsNaN(x), fallback, x);

        /// <summary>
        /// First finite: <paramref name="a"/> where it is finite (neither NaN nor ±inf), else
        /// <paramref name="b"/> — a robustness fallback stronger than a NaN-only replace.
        /// </summary>
        /// <param name="a">Preferred value.</param><param name="b">Fallback when <paramref name="a"/> is non-finite.</param>
        /// <returns>A fused select on <c>isfinite(a)</c>.</returns>
        public static NDExpr Coalesce(NDExpr a, NDExpr b) => Where(IsFinite(a), a, b);

        // ===================================================================
        // Activations (NN) — the C6 macro set
        // ===================================================================

        /// <summary>Rectified linear unit: <c>max(x, 0)</c> (NaN-propagating like np.maximum).</summary>
        /// <param name="x">Pre-activation.</param><returns>A fused np.maximum-with-zero node.</returns>
        public static NDExpr Relu(NDExpr x) => Max(x, Const(0));

        /// <summary>Leaky ReLU: <c>x</c> where <c>x &gt; 0</c>, else <c>slope*x</c>.</summary>
        /// <param name="x">Pre-activation.</param><param name="slope">Negative-side slope (e.g. 0.01).</param>
        /// <returns>A fused select.</returns>
        public static NDExpr LeakyRelu(NDExpr x, NDExpr slope) => Where(Greater(x, Const(0.0)), x, x * slope);

        /// <summary>Exponential linear unit: <c>x</c> where <c>x &gt; 0</c>, else <c>alpha*(exp(x)-1)</c> (uses expm1 for small-x accuracy).</summary>
        /// <param name="x">Pre-activation.</param><param name="alpha">Saturation scale for the negative branch (typically 1).</param>
        /// <returns>A fused select over <see cref="Expm1"/>.</returns>
        public static NDExpr Elu(NDExpr x, NDExpr alpha) => Where(Greater(x, Const(0.0)), x, alpha * Expm1(x));

        /// <summary>Logistic sigmoid: <c>1 / (1 + exp(-x))</c>.</summary>
        /// <param name="x">Input.</param><returns>A fused reciprocal-of-one-plus-exp node.</returns>
        public static NDExpr Sigmoid(NDExpr x) => Const(1.0) / (Const(1.0) + Exp(-x));

        /// <summary>SiLU / Swish: <c>x * sigmoid(x)</c>.</summary>
        /// <param name="x">Input.</param><returns>A fused product of <paramref name="x"/> and its sigmoid.</returns>
        public static NDExpr Swish(NDExpr x) => x * Sigmoid(x);

        /// <summary>
        /// Softplus <c>log(1 + exp(x))</c> in the numerically STABLE form <c>max(x,0) + log1p(exp(-|x|))</c>
        /// (algebraically identical, but never overflows for large <c>x</c> — the naive <c>log(1+exp(x))</c>
        /// saturates to +inf around x≈710 at float64).
        /// </summary>
        /// <param name="x">Input.</param><returns>A fused stable-softplus node.</returns>
        public static NDExpr Softplus(NDExpr x) => Max(x, Const(0.0)) + Log1p(Exp(-Abs(x)));

        /// <summary>
        /// GELU, the tanh approximation: <c>0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c> (the form
        /// used by GPT/BERT; the exact erf form waits for an <c>Erf</c> node — capabilities §7).
        /// </summary>
        /// <param name="x">Input.</param><returns>A fused tanh-approximation GELU node.</returns>
        public static NDExpr Gelu(NDExpr x)
            // √(2/π) = 0.7978845608028654; x³ as x*x*x keeps the SIMD arithmetic path (Power would scalarize).
            => Const(0.5) * x * (Const(1.0) + Tanh(Const(0.7978845608028654) * (x + Const(0.044715) * x * x * x)));

        /// <summary>Hard sigmoid (PyTorch/Keras piecewise-linear approx): <c>clip(x/6 + 0.5, 0, 1)</c>.</summary>
        /// <param name="x">Input.</param><returns>A fused clamp of the affine map.</returns>
        public static NDExpr HardSigmoid(NDExpr x) => Clamp(x / Const(6.0) + Const(0.5), Const(0.0), Const(1.0));

        /// <summary>Unit step / hard Heaviside at 0: 1 where <c>x &gt; 0</c>, else 0 (integer result, so it composes with arithmetic).</summary>
        /// <param name="x">Input.</param><returns>A fused select producing 0/1.</returns>
        public static NDExpr Step(NDExpr x) => Where(Greater(x, Const(0.0)), Const(1), Const(0));

        // ===================================================================
        // Boolean logic (inputs treated as bool masks)
        // ===================================================================

        /// <summary>NOT-AND: <c>!(a &amp; b)</c>.</summary>
        /// <param name="a">First mask.</param><param name="b">Second mask.</param><returns>A fused bool node.</returns>
        public static NDExpr Nand(NDExpr a, NDExpr b) => !(a & b);

        /// <summary>NOT-OR: <c>!(a | b)</c>.</summary>
        /// <param name="a">First mask.</param><param name="b">Second mask.</param><returns>A fused bool node.</returns>
        public static NDExpr Nor(NDExpr a, NDExpr b) => !(a | b);

        /// <summary>Logical equivalence (NOT-XOR): true where <paramref name="a"/> and <paramref name="b"/> agree.</summary>
        /// <param name="a">First mask.</param><param name="b">Second mask.</param><returns>A fused bool node.</returns>
        public static NDExpr Xnor(NDExpr a, NDExpr b) => !(a ^ b);

        /// <summary>Material implication <c>a ⇒ b</c> == <c>!a | b</c> (false only where <paramref name="a"/> is true and <paramref name="b"/> is false).</summary>
        /// <param name="a">Antecedent.</param><param name="b">Consequent.</param><returns>A fused bool node.</returns>
        public static NDExpr Implies(NDExpr a, NDExpr b) => !a | b;

        /// <summary>Majority vote of three masks: true where at least two are true.</summary>
        /// <param name="a">First mask.</param><param name="b">Second mask.</param><param name="c">Third mask.</param><returns>A fused bool node.</returns>
        public static NDExpr Majority3(NDExpr a, NDExpr b, NDExpr c) => (a & b) | (a & c) | (b & c);

        // ===================================================================
        // Predicates (numeric -> bool)
        // ===================================================================

        /// <summary>Strictly positive: <c>x &gt; 0</c>.</summary>
        /// <param name="x">Value tested.</param><returns>A fused bool node.</returns>
        public static NDExpr IsPositive(NDExpr x) => Greater(x, Const(0.0));

        /// <summary>Strictly negative: <c>x &lt; 0</c>.</summary>
        /// <param name="x">Value tested.</param><returns>A fused bool node.</returns>
        public static NDExpr IsNegative(NDExpr x) => Less(x, Const(0.0));

        /// <summary>Integral value: <c>floor(x) == x</c> (NaN is never equal → false; ±inf → false, since floor(inf) is inf and inf==inf, so document: ±inf reads as integral — matches <c>np.floor(x)==x</c>).</summary>
        /// <param name="x">Value tested.</param><returns>A fused bool node.</returns>
        public static NDExpr IsInteger(NDExpr x) => Equal(Floor(x), x);

        /// <summary>
        /// Approximate equality, the np.isclose relation with <c>equal_nan=False</c>:
        /// <c>|a-b| &lt;= atol + rtol·|b|</c> for finite operands, and exact equality otherwise (so
        /// <c>inf</c> is close to <c>inf</c>, NaN is close to nothing). Asymmetric in <paramref name="b"/>, like NumPy.
        /// </summary>
        /// <param name="a">First value.</param><param name="b">Reference value (the rtol scale).</param>
        /// <param name="rtol">Relative tolerance.</param><param name="atol">Absolute tolerance.</param>
        /// <returns>A fused bool node.</returns>
        public static NDExpr IsClose(NDExpr a, NDExpr b, NDExpr rtol, NDExpr atol)
            => Where(IsFinite(a) & IsFinite(b),
                     LessEqual(Abs(a - b), atol + rtol * Abs(b)),
                     Equal(a, b));

        /// <summary>
        /// Same sign: both negative, or both non-negative. Uses <c>(a&lt;0) == (b&lt;0)</c>, so +0 groups
        /// with the positives and signed-zero sign is NOT distinguished (document for -0.0-sensitive use).
        /// </summary>
        /// <param name="a">First value.</param><param name="b">Second value.</param><returns>A fused bool node.</returns>
        public static NDExpr SameSign(NDExpr a, NDExpr b) => Equal(Less(a, Const(0.0)), Less(b, Const(0.0)));

        /// <summary>Inclusive range test: true where <c>lo &lt;= x &lt;= hi</c>.</summary>
        /// <param name="x">Value tested.</param><param name="lo">Inclusive lower bound.</param><param name="hi">Inclusive upper bound.</param>
        /// <returns>A fused logical-and of the two comparisons.</returns>
        public static NDExpr Between(NDExpr x, NDExpr lo, NDExpr hi) => GreaterEqual(x, lo) & LessEqual(x, hi);

        // ===================================================================
        // Directional / sign (numeric -> numeric)
        // ===================================================================

        /// <summary>Three-way comparator: <c>-1</c> if <c>a &lt; b</c>, <c>0</c> if equal, <c>+1</c> if <c>a &gt; b</c> (the classic sort comparator).</summary>
        /// <param name="a">Left operand.</param><param name="b">Right operand.</param><returns>A fused node yielding -1/0/1.</returns>
        public static NDExpr Cmp(NDExpr a, NDExpr b) => Where(Greater(a, b), Const(1), Where(Less(a, b), Const(-1), Const(0)));

        /// <summary>
        /// Heaviside step (np.heaviside): 0 where <c>x &lt; 0</c>, 1 where <c>x &gt; 0</c>, and
        /// <paramref name="h0"/> exactly at <c>x == 0</c>.
        /// </summary>
        /// <param name="x">Input.</param><param name="h0">Value AT zero (np.heaviside's second argument).</param><returns>A fused node.</returns>
        public static NDExpr Heaviside(NDExpr x, NDExpr h0)
            => Where(Less(x, Const(0.0)), Const(0.0), Where(Greater(x, Const(0.0)), Const(1.0), h0));

        /// <summary>
        /// Move <paramref name="x"/> toward <paramref name="target"/> by at most <paramref name="delta"/>
        /// (a clamped approach — control loops, animation): <c>x + clamp(target-x, -delta, delta)</c>.
        /// </summary>
        /// <param name="x">Current value.</param><param name="target">Goal.</param><param name="delta">Max step magnitude (≥ 0).</param><returns>A fused node.</returns>
        public static NDExpr StepToward(NDExpr x, NDExpr target, NDExpr delta) => x + Clamp(target - x, -delta, delta);

        /// <summary>The operand with the larger magnitude, ties keeping <paramref name="a"/> — "keep the stronger signal".</summary>
        /// <param name="a">First operand.</param><param name="b">Second operand.</param><returns>A fused select on <c>|a| &gt;= |b|</c>.</returns>
        public static NDExpr MaxMagnitude(NDExpr a, NDExpr b) => Where(GreaterEqual(Abs(a), Abs(b)), a, b);

        // ===================================================================
        // Multi-way decision & interpolation
        // ===================================================================

        /// <summary>
        /// Bucket index (np.digitize, right-open, ascending edges): the count of <paramref name="edges"/>
        /// that <paramref name="x"/> meets or exceeds, in <c>[0, edges.Length]</c>. Sums INTEGER
        /// <c>Where(x&gt;=e, 1, 0)</c> values — a bool sum would be logical OR, not a count.
        /// </summary>
        /// <param name="x">Value to place.</param><param name="edges">Ascending bin edges.</param>
        /// <returns>A fused integer-accumulating node.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null, or <paramref name="edges"/> holds a null.</exception>
        public static NDExpr Bucketize(NDExpr x, params NDExpr[] edges)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (edges is null) throw new ArgumentNullException(nameof(edges));
            NDExpr acc = Const(0);
            foreach (var e in edges)
                acc = acc + Where(GreaterEqual(x, e ?? throw new ArgumentNullException(nameof(edges), "an edge is null.")), Const(1), Const(0));
            return acc;
        }

        /// <summary>Median of three — the branch-free <c>max(min(a,b), min(max(a,b), c))</c> (a robust 3-tap de-glitcher).</summary>
        /// <param name="a">First.</param><param name="b">Second.</param><param name="c">Third.</param><returns>A fused node yielding the middle value.</returns>
        public static NDExpr Median3(NDExpr a, NDExpr b, NDExpr c) => Max(Min(a, b), Min(Max(a, b), c));

        /// <summary>PyTorch-style threshold: pass <paramref name="x"/> where <c>x &gt; t</c>, else substitute <paramref name="value"/>.</summary>
        /// <param name="x">Input.</param><param name="t">Threshold.</param><param name="value">Replacement at/below the threshold.</param><returns>A fused select.</returns>
        public static NDExpr Threshold(NDExpr x, NDExpr t, NDExpr value) => Where(Greater(x, t), x, value);

        /// <summary>Linear interpolation <c>a + (b-a)·t</c> — pure arithmetic (fused, no branch); the natural companion to the selectors.</summary>
        /// <param name="a">Value at <c>t = 0</c>.</param><param name="b">Value at <c>t = 1</c>.</param><param name="t">Interpolation parameter (unclamped).</param><returns>A fused arithmetic node.</returns>
        public static NDExpr Lerp(NDExpr a, NDExpr b, NDExpr t) => a + (b - a) * t;
    }
}
