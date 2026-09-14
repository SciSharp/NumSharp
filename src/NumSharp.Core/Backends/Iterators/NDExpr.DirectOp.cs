using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.DirectOp.cs — Phase 6.3: single-op trees delegate to the engine's direct kernel
// (docs/plans/ndexpr-evaluate.md §6.3)
// =============================================================================
//
// A tree that is ONE elementwise op over array leaves — `abs(a)`, `maximum(a, b)`, `a + b` —
// has nothing to FUSE: np.evaluate's whole reason for being (one NDIter pass, no intermediates)
// buys nothing over the engine's own whole-array kernel (`np.abs` / `np.maximum` / `a + b`), yet
// the fused pass carries a heavier fixed cost (MultiNew + ForEach + the shape/broadcast resolve).
// Measured at 1K that made np.evaluate LOSE to the direct call it wraps — `maximum(a, b)` 0.59×,
// `abs(a)` 0.63× (§0.1 Section C, the last two sub-1.0× rows).
//
// The fix (this file + the gate in DefaultEngine.EvaluateCore) recognises that shape at PROGRAM
// BUILD time and, on a PLAIN call over C-contiguous array leaves, delegates to the engine's
// ExecuteUnaryOp / ExecuteBinaryOp — the exact kernel `np.abs` / `np.maximum` run. It is bit-exact
// BY CONSTRUCTION, not by luck: the fused single-op path and the engine op are BOTH gated bit-exact
// to NumPy by the differential-fuzz corpus (evaluate.jsonl and the unary/binary tiers), so two
// things equal to NumPy are equal to each other. Three gate conditions make that argument airtight
// (see DirectOpGate below): array leaves (a weak-literal ConstNode would promote differently under
// NEP50), C-contiguous inputs (so the result is C-contiguous on BOTH paths — the fused strict-F
// heuristic never fires for C inputs), and the plain call (out=/where=/dtype=/non-'K' order all
// change result shaping and stay on the fully-general fused path).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// Which engine kernel a <see cref="NDExprDirectOp"/> delegates to. <see cref="None"/> (the
    /// struct default) means the tree is NOT a delegatable single op — the caller keeps the fused
    /// NDIter pass. A <see cref="MinMaxNode"/> folds into <see cref="Binary"/> via
    /// <see cref="BinaryOp.Minimum"/>/<see cref="BinaryOp.Maximum"/>, so there is no separate kind for it.
    /// </summary>
    internal enum NDExprDirectKind : byte
    {
        /// <summary>Not a single-op-over-array-leaves tree; no delegation (the default).</summary>
        None = 0,
        /// <summary>A single <see cref="UnaryNode"/> over one array leaf → <c>ExecuteUnaryOp</c>.</summary>
        Unary,
        /// <summary>A single <see cref="BinaryNode"/>/<see cref="MinMaxNode"/> over two array leaves → <c>ExecuteBinaryOp</c>.</summary>
        Binary,
    }

    /// <summary>
    /// The engine-kernel delegation plan for a single-op tree: the op and the input INDICES of its
    /// leaf children (indices into the call's <c>inputs[]</c>, i.e. <see cref="InputNode.Index"/>),
    /// resolved ONCE at program build so a cached program carries it with no per-call tree walk. A
    /// <c>default</c> value (<see cref="Kind"/> == <see cref="NDExprDirectKind.None"/>) means "not
    /// delegatable" — the whole point of the sentinel is that the cheap <c>Kind != None</c> check on
    /// the hot path decides whether to even look.
    /// </summary>
    internal readonly struct NDExprDirectOp
    {
        /// <summary>The delegation kind, or <see cref="NDExprDirectKind.None"/> for no delegation.</summary>
        public readonly NDExprDirectKind Kind;
        /// <summary>The unary op (valid only when <see cref="Kind"/> is <see cref="NDExprDirectKind.Unary"/>).</summary>
        public readonly UnaryOp UnOp;
        /// <summary>The binary op (valid only when <see cref="Kind"/> is <see cref="NDExprDirectKind.Binary"/>).</summary>
        public readonly BinaryOp BinOp;
        /// <summary>Input index of the (first) leaf child — the left/only operand.</summary>
        public readonly int Input0;
        /// <summary>Input index of the second leaf child (Binary only; -1 for Unary).</summary>
        public readonly int Input1;

        private NDExprDirectOp(NDExprDirectKind kind, UnaryOp un, BinaryOp bin, int in0, int in1)
        {
            Kind = kind;
            UnOp = un;
            BinOp = bin;
            Input0 = in0;
            Input1 = in1;
        }

        /// <summary>Build a unary delegation plan for <paramref name="op"/> over input <paramref name="input"/>.</summary>
        /// <param name="op">The unary op (must satisfy <see cref="NDExpr.IsDelegatableUnary"/>).</param>
        /// <param name="input">The leaf child's input index.</param>
        /// <returns>A <see cref="NDExprDirectKind.Unary"/> plan.</returns>
        public static NDExprDirectOp Unary(UnaryOp op, int input)
            => new(NDExprDirectKind.Unary, op, default, input, -1);

        /// <summary>Build a binary delegation plan for <paramref name="op"/> over inputs <paramref name="in0"/>/<paramref name="in1"/>.</summary>
        /// <param name="op">The binary op (every <see cref="BinaryOp"/> routes through <c>ExecuteBinaryOp</c>).</param>
        /// <param name="in0">The left leaf child's input index.</param>
        /// <param name="in1">The right leaf child's input index (may equal <paramref name="in0"/> — a deduplicated repeat).</param>
        /// <returns>A <see cref="NDExprDirectKind.Binary"/> plan.</returns>
        public static NDExprDirectOp Binary(BinaryOp op, int in0, int in1)
            => new(NDExprDirectKind.Binary, default, op, in0, in1);
    }

    public abstract partial class NDExpr
    {
        /// <summary>
        /// Test / diagnostics hook: when set on the current thread, np.evaluate SKIPS the Phase-6.3
        /// single-op delegation and runs the fully-general fused NDIter pass instead. The gate exists so
        /// a test can evaluate the same tree BOTH ways and prove they are byte-identical — the direct
        /// safety claim behind delegation. Thread-static so a parallel test never perturbs another's run.
        /// </summary>
        [System.ThreadStatic] internal static bool DisableDirectOp;

        /// <summary>
        /// Test / diagnostics hook: incremented (on the current thread) each time np.evaluate actually
        /// delegates a single op to the engine kernel. A test resets it, evaluates, and asserts it moved
        /// — proving the fast path ENGAGED rather than silently falling through to the fused pass (which
        /// would still give the right answer, hiding a dead optimization). Thread-static for the same
        /// reason as <see cref="DisableDirectOp"/>.
        /// </summary>
        [System.ThreadStatic] internal static int DirectOpDelegations;

        /// <summary>
        /// Whether a single <see cref="UnaryOp"/> tree may be delegated to <c>ExecuteUnaryOp</c> — a
        /// deliberately NARROW, positive allow-list of the "real math" ufuncs that compute <c>f(x)</c> at
        /// the resolved output dtype, where forcing <c>typeCode = ResultType</c> is exactly what the
        /// engine's trivial-contiguous kernel expects. Everything else stays on the fused pass, for one of
        /// two reasons: (a) it is NOT a plain-math ufunc — the bool-result predicates / <c>LogicalNot</c> /
        /// <c>SignBit</c>, the byte-result <c>BitwiseCount</c>, <c>BitwiseNot</c>, <c>Conjugate</c>,
        /// <c>Positive</c>, <c>Spacing</c> — where <c>ExecuteUnaryOp</c>'s per-op <c>typeCode</c> expectation
        /// is not "compute in the given dtype" and forcing <c>ResultType</c> reaches a trivial-contiguous
        /// path that FATAL-CRASHES the CLR for some dtype (e.g. <c>logical_not(complex)</c> — a latent engine
        /// bug the general fused pass never hits); (b) it has NO engine kernel at all — the NDExpr-only
        /// <see cref="UnaryOp.Real"/>/<see cref="UnaryOp.Imag"/>/<see cref="UnaryOp.Angle"/> (and
        /// <c>Rint</c>, which the engine routes through <see cref="UnaryOp.Round"/>, not its own op). These
        /// exclusions are niche as single-op "fusions", so leaving them fused costs nothing measurable while
        /// keeping delegation provably crash-free. The whole list is gated by the delegated-vs-fused sweep
        /// (<c>NDEvaluateTests.P63_UnaryAllowList_*</c>); a new op stays fused until it is proven here.
        /// </summary>
        /// <param name="op">The unary op to classify.</param>
        /// <returns>True when <c>ExecuteUnaryOp</c> serves <paramref name="op"/> identically to the fused emit and cannot crash.</returns>
        internal static bool IsDelegatableUnary(UnaryOp op) => op switch
        {
            // Reciprocal is DELIBERATELY absent: the engine's integer np.reciprocal and the fused
            // Reciprocal(int) disagree on every integer dtype (bytes differ — a pre-existing engine/fused
            // inconsistency, the fused path being the oracle-gated one), so delegating it would change the
            // answer. Float/complex Reciprocal match, but a per-dtype carve-out is not worth a rare single op.
            UnaryOp.Negate or UnaryOp.Abs or UnaryOp.Fabs or UnaryOp.Sqrt or UnaryOp.Cbrt
            or UnaryOp.Square or UnaryOp.Sign
            or UnaryOp.Exp or UnaryOp.Exp2 or UnaryOp.Expm1
            or UnaryOp.Log or UnaryOp.Log2 or UnaryOp.Log10 or UnaryOp.Log1p
            or UnaryOp.Sin or UnaryOp.Cos or UnaryOp.Tan
            or UnaryOp.Sinh or UnaryOp.Cosh or UnaryOp.Tanh
            or UnaryOp.ASin or UnaryOp.ACos or UnaryOp.ATan
            or UnaryOp.Asinh or UnaryOp.Acosh or UnaryOp.Atanh
            or UnaryOp.Floor or UnaryOp.Ceil or UnaryOp.Truncate or UnaryOp.Round
            or UnaryOp.Deg2Rad or UnaryOp.Rad2Deg => true,
            _ => false,
        };

        /// <summary>
        /// Whether a single <see cref="BinaryOp"/> tree may be delegated to <c>ExecuteBinaryOp</c> (with NO
        /// dtype forcing, and only over ndim ≥ 1 operands). A NARROW, positive allow-list of the "plain"
        /// ops where the engine's <c>_FindCommonType</c> equals the fused NEP50 typing for two strong
        /// arrays — arithmetic, division, remainder, bitwise, shifts, min/max (incl. NaN-ignoring
        /// fmax/fmin), and gcd/lcm. TWO families are excluded because delegating them would change the
        /// answer: (a) the FLOAT-TIER ops <see cref="BinaryOp.ATan2"/>/<see cref="BinaryOp.CopySign"/>/
        /// <see cref="BinaryOp.NextAfter"/>/<see cref="BinaryOp.LogAddExp"/>/<see cref="BinaryOp.LogAddExp2"/>/
        /// <see cref="BinaryOp.Hypot"/>/<see cref="BinaryOp.Heaviside"/>, whose fused typing PROMOTES an
        /// integer pair to a float tier while <c>_FindCommonType</c> keeps the integer dtype — so
        /// <c>ExecuteBinaryOp</c> throws "no loop" where the fused path computes; and (b)
        /// <see cref="BinaryOp.Power"/>, whose fused path raises NumPy's "Integers to negative integer
        /// powers" guard — reproducing it in the delegation would need the dtype forcing that in turn
        /// re-broke the guard, so it stays fused. These are all rare single-op forms; leaving them fused
        /// costs nothing measurable. Gated by the delegated-vs-fused sweep
        /// (<c>NDEvaluateTests.P63_BinaryAllowList_*</c>).
        /// </summary>
        /// <param name="op">The binary op to classify.</param>
        /// <returns>True when <c>ExecuteBinaryOp</c> (no dtype force, ndim ≥ 1) serves <paramref name="op"/> identically to the fused emit.</returns>
        internal static bool IsDelegatableBinary(BinaryOp op) => op switch
        {
            BinaryOp.Add or BinaryOp.Subtract or BinaryOp.Multiply or BinaryOp.Divide
            or BinaryOp.Mod or BinaryOp.FloorDivide or BinaryOp.Fmod
            or BinaryOp.BitwiseAnd or BinaryOp.BitwiseOr or BinaryOp.BitwiseXor
            or BinaryOp.LeftShift or BinaryOp.RightShift
            or BinaryOp.Maximum or BinaryOp.Minimum or BinaryOp.FMax or BinaryOp.FMin
            or BinaryOp.Gcd or BinaryOp.Lcm => true,
            _ => false,
        };

        /// <summary>
        /// Classify this (bound) node as a delegatable single elementwise op over array leaves, or
        /// <c>default</c> (<see cref="NDExprDirectKind.None"/>) when it is not. Called once per program
        /// build on the bound tree; the base returns None so every node type that is not a delegatable
        /// single op (leaves, casts, rounds, comparisons, logicals, reductions, and any node whose child
        /// is itself an op) opts out by inheritance.
        /// </summary>
        /// <returns>The delegation plan, or <c>default</c> for no delegation.</returns>
        internal virtual NDExprDirectOp AsDirectSingleOp() => default;
    }

    public sealed partial class UnaryNode
    {
        /// <summary>
        /// A unary node delegates when its child is a bound array leaf (<see cref="InputNode"/> — NOT a
        /// weak-literal <see cref="ConstNode"/>, whose NEP50 promotion the engine op would not reproduce)
        /// and its op has an engine kernel (<see cref="NDExpr.IsDelegatableUnary"/>). A 0-d array input is
        /// an <see cref="InputNode"/> too, so <c>abs(k)</c> with a 0-d <c>k</c> also qualifies.
        /// </summary>
        internal override NDExprDirectOp AsDirectSingleOp()
            => _child is InputNode inp && NDExpr.IsDelegatableUnary(_op)
                ? NDExprDirectOp.Unary(_op, inp.Index)
                : default;
    }

    public sealed partial class BinaryNode
    {
        /// <summary>
        /// A binary node delegates when BOTH children are bound array leaves (<see cref="InputNode"/>) AND
        /// the op is in the plain-op allow-list (<see cref="NDExpr.IsDelegatableBinary"/> — the float-tier
        /// promoting ops and <c>Power</c>'s guard are excluded). A weak-literal <see cref="ConstNode"/>
        /// operand (<c>a + 2</c>) opts out here — its NEP50 weak promotion is the fused path's job — and a
        /// 0-d strong operand is filtered later, at the C-contiguity gate, since its value-based promotion
        /// in <c>_FindCommonType</c> would diverge from the fused STRONG typing.
        /// </summary>
        internal override NDExprDirectOp AsDirectSingleOp()
            => _left is InputNode l && _right is InputNode r && NDExpr.IsDelegatableBinary(_op)
                ? NDExprDirectOp.Binary(_op, l.Index, r.Index)
                : default;
    }

    public sealed partial class MinMaxNode
    {
        /// <summary>
        /// A min/max node delegates like a binary op — <see cref="MinMaxNode"/> IS the engine's
        /// <see cref="BinaryOp.Minimum"/>/<see cref="BinaryOp.Maximum"/> (its <c>EmitScalar</c> emits
        /// exactly that op), so both children being bound array leaves is the whole condition, and the
        /// NaN-propagating tie semantics come from the shared kernel.
        /// </summary>
        internal override NDExprDirectOp AsDirectSingleOp()
        {
            if (_left is not InputNode l || _right is not InputNode r)
                return default;
            var op = _isMin ? BinaryOp.Minimum : BinaryOp.Maximum;   // both are in the plain allow-list
            return NDExpr.IsDelegatableBinary(op) ? NDExprDirectOp.Binary(op, l.Index, r.Index) : default;
        }
    }
}
