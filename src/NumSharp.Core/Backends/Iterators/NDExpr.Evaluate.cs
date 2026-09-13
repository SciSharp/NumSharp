using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Evaluate.cs — np.evaluate surface of the expression DSL (Wave 6.1)
// =============================================================================
//
// Adds the pieces that turn the Tier-3C compiler into a user-facing fused
// evaluator:
//
//   • ArrayNode — an NDArray embedded directly as a leaf, so trees read
//     naturally: (NDExpr)a * b + 2. np.evaluate REBINDS array leaves to
//     positional InputNodes, deduplicating repeated references — the same
//     NDArray instance appearing twice becomes ONE iterator operand
//     ((a-b)/(a+b) iterates 3 streams, not 5).
//   • implicit conversions NDArray→NDExpr and numeric→Const, so a single
//     cast at the head of an expression lights up the whole operator set.
//   • ReduceNode — root-only fused reductions (sum/prod/min/max/mean of an
//     arbitrary elementwise tree) compiled to a one-pass accumulating
//     kernel: sum(a*b) reads a and b once and never materializes a*b.
//
// Binding is a pure rewrite: nodes are immutable, so BindArrays returns the
// same instance when no array leaf lives below a node, or a rebuilt node
// otherwise. The bound tree is what typing + emission consume.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>Reduction kinds supported by <see cref="ReduceNode"/>.</summary>
    /// <remarks>
    /// <see cref="Any"/> / <see cref="All"/> are their OWN kinds, not <see cref="Max"/> / <see cref="Min"/>
    /// over a boolean child: <c>logical_or</c> / <c>logical_and</c> carry the identities <c>False</c> /
    /// <c>True</c>, so a reduction over an EMPTY input is <c>False</c> / <c>True</c> — where
    /// <c>maximum.reduce</c> / <c>minimum.reduce</c> raise "zero-size array … which has no identity". They
    /// also return <c>bool</c> where a bool <see cref="Sum"/> would widen to int64. The value fold is a
    /// bitwise OR / AND on a bool accumulator, which is associative AND idempotent — so the 4-accumulator
    /// unroll reorders it freely and stays bit-exact (unlike float <see cref="Sum"/> / <see cref="Prod"/>,
    /// whose order matters — the M1/M2 divert).
    /// </remarks>
    public enum NDExprReduceKind : byte
    {
        Sum,
        Prod,
        Min,
        Max,
        Mean,

        /// <summary><c>logical_or.reduce</c> of the nonzero-test of the child (identity <c>False</c>; bool result).</summary>
        Any,

        /// <summary><c>logical_and.reduce</c> of the nonzero-test of the child (identity <c>True</c>; bool result).</summary>
        All,

        /// <summary>
        /// <c>np.ptp</c> — peak-to-peak (<c>max - min</c>) of the child. Result dtype is the child's
        /// (integer/unsigned overflow WRAPS exactly as NumPy's, since it is a plain subtract of two
        /// same-dtype reductions), a NaN in the child propagates to the result, and there is NO empty
        /// identity (a zero-size input raises, like <see cref="Max"/>/<see cref="Min"/>). Unlike the
        /// fold-driven kinds above it is host-delegated: <see cref="np.ptp(NDArray,int?,NDArray,bool)"/>
        /// runs over the materialized child — see the host divert in <c>DefaultEngine.EvaluateReduce</c>.
        /// </summary>
        Ptp,

        /// <summary>
        /// <c>np.nanmin</c> — minimum ignoring NaN. Result dtype is the child's; a float/complex child
        /// has its NaNs skipped (an ALL-NaN input yields NaN, matching NumPy's all-NaN-slice contract),
        /// while an integer/bool child carries no NaN and folds as a plain minimum. Host-delegated to
        /// <see cref="np.nanmin(NDArray,int?,bool)"/> over the materialized child, which is why it is
        /// TYPE-AWARE for free (the ±inf NaN-replacement sentinel that would promote an int child to
        /// float64 lives inside the engine reduction, not in an NDExpr rewrite).
        /// </summary>
        NanMin,

        /// <summary>
        /// <c>np.nanmax</c> — maximum ignoring NaN (the <see cref="NanMin"/> twin, all-NaN → NaN).
        /// Host-delegated to <see cref="np.nanmax(NDArray,int?,bool)"/> over the materialized child.
        /// </summary>
        NanMax,
    }

    /// <summary>
    /// Operand collection for binding array leaves: deduplicates by reference
    /// so a repeated NDArray maps to one iterator operand.
    /// </summary>
    [NDBorrowed] // the operands an expression is bound over are the caller's arrays
    internal sealed class NDExprBindContext
    {
        public readonly List<NDArray> Operands = new();

        public int IndexOf(NDArray array)
        {
            for (int i = 0; i < Operands.Count; i++)
            {
                if (ReferenceEquals(Operands[i], array))
                    return i;
            }

            Operands.Add(array);
            return Operands.Count - 1;
        }
    }

    public abstract partial class NDExpr
    {
        // ===================================================================
        // Array leaves + literal sugar
        // ===================================================================

        /// <summary>
        /// Embed an NDArray directly as an expression leaf. np.evaluate binds
        /// every distinct array (by reference) to one iterator operand.
        /// </summary>
        public static NDExpr Arr(NDArray array) => new ArrayNode(array);

        public static implicit operator NDExpr(NDArray array) => new ArrayNode(array);
        public static implicit operator NDExpr(double value) => Const(value);
        public static implicit operator NDExpr(float value) => Const(value);
        public static implicit operator NDExpr(int value) => Const(value);
        public static implicit operator NDExpr(long value) => Const(value);
        public static implicit operator NDExpr(uint value) => Const(value);
        public static implicit operator NDExpr(ulong value) => Const(value);
        public static implicit operator NDExpr(bool value) => Const(value);
        public static implicit operator NDExpr(System.Numerics.Complex value) => Const(value);
        public static implicit operator NDExpr(Half value) => Const(value);
        public static implicit operator NDExpr(decimal value) => Const(value);

        // Mixed NDExpr/NDArray operators. Exact-match overloads are required:
        // through implicit conversions alone, `expr * ndarray` is ambiguous
        // between NDExpr.op_*(NDExpr, NDExpr) and NDArray's own
        // object-accepting operator overloads.
        public static NDExpr operator +(NDExpr a, NDArray b) => Add(a, Arr(b));
        public static NDExpr operator +(NDArray a, NDExpr b) => Add(Arr(a), b);
        public static NDExpr operator -(NDExpr a, NDArray b) => Subtract(a, Arr(b));
        public static NDExpr operator -(NDArray a, NDExpr b) => Subtract(Arr(a), b);
        public static NDExpr operator *(NDExpr a, NDArray b) => Multiply(a, Arr(b));
        public static NDExpr operator *(NDArray a, NDExpr b) => Multiply(Arr(a), b);
        public static NDExpr operator /(NDExpr a, NDArray b) => Divide(a, Arr(b));
        public static NDExpr operator /(NDArray a, NDExpr b) => Divide(Arr(a), b);
        public static NDExpr operator %(NDExpr a, NDArray b) => Mod(a, Arr(b));
        public static NDExpr operator %(NDArray a, NDExpr b) => Mod(Arr(a), b);
        public static NDExpr operator &(NDExpr a, NDArray b) => BitwiseAnd(a, Arr(b));
        public static NDExpr operator &(NDArray a, NDExpr b) => BitwiseAnd(Arr(a), b);
        public static NDExpr operator |(NDExpr a, NDArray b) => BitwiseOr(a, Arr(b));
        public static NDExpr operator |(NDArray a, NDExpr b) => BitwiseOr(Arr(a), b);
        public static NDExpr operator ^(NDExpr a, NDArray b) => BitwiseXor(a, Arr(b));
        public static NDExpr operator ^(NDArray a, NDExpr b) => BitwiseXor(Arr(a), b);

        // Scalar operators. Also required as exact matches: without them a
        // literal binds to the (NDExpr, NDArray) overload through NDArray's
        // implicit numeric conversions (NDArray is the better conversion
        // target because NDArray→NDExpr exists), silently turning a WEAK
        // NEP50 literal into a strong scalar array — f4+2.5 would promote to
        // f8 instead of staying f4.
        public static NDExpr operator +(NDExpr a, double b) => Add(a, Const(b));
        public static NDExpr operator +(double a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator +(NDExpr a, long b) => Add(a, Const(b));
        public static NDExpr operator +(long a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator +(NDExpr a, int b) => Add(a, Const(b));
        public static NDExpr operator +(int a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator -(NDExpr a, double b) => Subtract(a, Const(b));
        public static NDExpr operator -(double a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator -(NDExpr a, long b) => Subtract(a, Const(b));
        public static NDExpr operator -(long a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator -(NDExpr a, int b) => Subtract(a, Const(b));
        public static NDExpr operator -(int a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator *(NDExpr a, double b) => Multiply(a, Const(b));
        public static NDExpr operator *(double a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator *(NDExpr a, long b) => Multiply(a, Const(b));
        public static NDExpr operator *(long a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator *(NDExpr a, int b) => Multiply(a, Const(b));
        public static NDExpr operator *(int a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator /(NDExpr a, double b) => Divide(a, Const(b));
        public static NDExpr operator /(double a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator /(NDExpr a, long b) => Divide(a, Const(b));
        public static NDExpr operator /(long a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator /(NDExpr a, int b) => Divide(a, Const(b));
        public static NDExpr operator /(int a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator %(NDExpr a, double b) => Mod(a, Const(b));
        public static NDExpr operator %(double a, NDExpr b) => Mod(Const(a), b);
        public static NDExpr operator %(NDExpr a, long b) => Mod(a, Const(b));
        public static NDExpr operator %(long a, NDExpr b) => Mod(Const(a), b);
        public static NDExpr operator %(NDExpr a, int b) => Mod(a, Const(b));
        public static NDExpr operator %(int a, NDExpr b) => Mod(Const(a), b);
        public static NDExpr operator &(NDExpr a, long b) => BitwiseAnd(a, Const(b));
        public static NDExpr operator &(long a, NDExpr b) => BitwiseAnd(Const(a), b);
        public static NDExpr operator &(NDExpr a, int b) => BitwiseAnd(a, Const(b));
        public static NDExpr operator &(int a, NDExpr b) => BitwiseAnd(Const(a), b);
        public static NDExpr operator |(NDExpr a, long b) => BitwiseOr(a, Const(b));
        public static NDExpr operator |(long a, NDExpr b) => BitwiseOr(Const(a), b);
        public static NDExpr operator |(NDExpr a, int b) => BitwiseOr(a, Const(b));
        public static NDExpr operator |(int a, NDExpr b) => BitwiseOr(Const(a), b);
        public static NDExpr operator ^(NDExpr a, long b) => BitwiseXor(a, Const(b));
        public static NDExpr operator ^(long a, NDExpr b) => BitwiseXor(Const(a), b);
        public static NDExpr operator ^(NDExpr a, int b) => BitwiseXor(a, Const(b));
        public static NDExpr operator ^(int a, NDExpr b) => BitwiseXor(Const(a), b);

        // ulong: without an exact match a ulong literal has no implicit path to int/long and
        // would bind through double — turning `u8 + 18446744073709551615UL` into a float64
        // expression where NumPy keeps uint64 (and wraps). Complex: keeps the literal WEAK
        // (a Complex→NDArray conversion would make it a strong 0-d array and cost an operand).
        public static NDExpr operator +(NDExpr a, ulong b) => Add(a, Const(b));
        public static NDExpr operator +(ulong a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator -(NDExpr a, ulong b) => Subtract(a, Const(b));
        public static NDExpr operator -(ulong a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator *(NDExpr a, ulong b) => Multiply(a, Const(b));
        public static NDExpr operator *(ulong a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator /(NDExpr a, ulong b) => Divide(a, Const(b));
        public static NDExpr operator /(ulong a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator %(NDExpr a, ulong b) => Mod(a, Const(b));
        public static NDExpr operator %(ulong a, NDExpr b) => Mod(Const(a), b);
        public static NDExpr operator &(NDExpr a, ulong b) => BitwiseAnd(a, Const(b));
        public static NDExpr operator &(ulong a, NDExpr b) => BitwiseAnd(Const(a), b);
        public static NDExpr operator |(NDExpr a, ulong b) => BitwiseOr(a, Const(b));
        public static NDExpr operator |(ulong a, NDExpr b) => BitwiseOr(Const(a), b);
        public static NDExpr operator ^(NDExpr a, ulong b) => BitwiseXor(a, Const(b));
        public static NDExpr operator ^(ulong a, NDExpr b) => BitwiseXor(Const(a), b);
        public static NDExpr operator +(NDExpr a, System.Numerics.Complex b) => Add(a, Const(b));
        public static NDExpr operator +(System.Numerics.Complex a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator -(NDExpr a, System.Numerics.Complex b) => Subtract(a, Const(b));
        public static NDExpr operator -(System.Numerics.Complex a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator *(NDExpr a, System.Numerics.Complex b) => Multiply(a, Const(b));
        public static NDExpr operator *(System.Numerics.Complex a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator /(NDExpr a, System.Numerics.Complex b) => Divide(a, Const(b));
        public static NDExpr operator /(System.Numerics.Complex a, NDExpr b) => Divide(Const(a), b);

        // Half / decimal: System.Numerics.Complex declares implicit conversions from BOTH (as from
        // every primitive), and Complex converts implicitly to NDExpr, so without an exact match the
        // Complex overloads above would capture `expr + (Half)2` and `expr + 0.1m` and type them
        // complex. The exact overloads keep them the STRONG float16 / decimal scalars they are.
        public static NDExpr operator +(NDExpr a, Half b) => Add(a, Const(b));
        public static NDExpr operator +(Half a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator -(NDExpr a, Half b) => Subtract(a, Const(b));
        public static NDExpr operator -(Half a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator *(NDExpr a, Half b) => Multiply(a, Const(b));
        public static NDExpr operator *(Half a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator /(NDExpr a, Half b) => Divide(a, Const(b));
        public static NDExpr operator /(Half a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator +(NDExpr a, decimal b) => Add(a, Const(b));
        public static NDExpr operator +(decimal a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator -(NDExpr a, decimal b) => Subtract(a, Const(b));
        public static NDExpr operator -(decimal a, NDExpr b) => Subtract(Const(a), b);
        public static NDExpr operator *(NDExpr a, decimal b) => Multiply(a, Const(b));
        public static NDExpr operator *(decimal a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator /(NDExpr a, decimal b) => Divide(a, Const(b));
        public static NDExpr operator /(decimal a, NDExpr b) => Divide(Const(a), b);
        public static NDExpr operator %(NDExpr a, decimal b) => Mod(a, Const(b));
        public static NDExpr operator %(decimal a, NDExpr b) => Mod(Const(a), b);

        // bool: a WEAK Python bool (adopts any dtype); without the exact match it would bind through
        // NDArray's bool conversion as a strong 0-d operand — same dtype, one more iterator stream.
        public static NDExpr operator +(NDExpr a, bool b) => Add(a, Const(b));
        public static NDExpr operator +(bool a, NDExpr b) => Add(Const(a), b);
        public static NDExpr operator *(NDExpr a, bool b) => Multiply(a, Const(b));
        public static NDExpr operator *(bool a, NDExpr b) => Multiply(Const(a), b);
        public static NDExpr operator &(NDExpr a, bool b) => BitwiseAnd(a, Const(b));
        public static NDExpr operator &(bool a, NDExpr b) => BitwiseAnd(Const(a), b);
        public static NDExpr operator |(NDExpr a, bool b) => BitwiseOr(a, Const(b));
        public static NDExpr operator |(bool a, NDExpr b) => BitwiseOr(Const(a), b);
        public static NDExpr operator ^(NDExpr a, bool b) => BitwiseXor(a, Const(b));
        public static NDExpr operator ^(bool a, NDExpr b) => BitwiseXor(Const(a), b);

        // ===================================================================
        // Reduction factories (root-only — see ReduceNode)
        // ===================================================================

        /// <summary>One-pass fused sum of the expression (NumPy dtype rules: int→int64, uint→uint64, floats preserved).</summary>
        public static NDExpr Sum(NDExpr x) => new ReduceNode(NDExprReduceKind.Sum, x);

        /// <summary>One-pass fused product of the expression.</summary>
        public static NDExpr Prod(NDExpr x) => new ReduceNode(NDExprReduceKind.Prod, x);

        /// <summary>One-pass fused minimum of the expression (NaN-propagating, like np.min).</summary>
        public static NDExpr Min(NDExpr x) => new ReduceNode(NDExprReduceKind.Min, x);

        /// <summary>One-pass fused maximum of the expression (NaN-propagating, like np.max).</summary>
        public static NDExpr Max(NDExpr x) => new ReduceNode(NDExprReduceKind.Max, x);

        /// <summary>One-pass fused arithmetic mean of the expression (ints→float64, floats preserved).</summary>
        public static NDExpr Mean(NDExpr x) => new ReduceNode(NDExprReduceKind.Mean, x);

        // --- axis-aware fused reductions (one pass, no intermediate; e.g. evaluate(Sum(a*b, axis:0))) ---

        /// <summary>One-pass fused sum of the expression along <paramref name="axis"/>.</summary>
        public static NDExpr Sum(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Sum, x, axis, keepdims);

        /// <summary>One-pass fused product of the expression along <paramref name="axis"/>.</summary>
        public static NDExpr Prod(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Prod, x, axis, keepdims);

        /// <summary>One-pass fused minimum of the expression along <paramref name="axis"/> (NaN-propagating).</summary>
        public static NDExpr Min(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Min, x, axis, keepdims);

        /// <summary>One-pass fused maximum of the expression along <paramref name="axis"/> (NaN-propagating).</summary>
        public static NDExpr Max(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Max, x, axis, keepdims);

        /// <summary>One-pass fused arithmetic mean of the expression along <paramref name="axis"/>.</summary>
        public static NDExpr Mean(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Mean, x, axis, keepdims);

        // --- presence / count / NaN-aware reductions (plan P2 M4) --------------------------------
        //
        // Any/All are their OWN kinds (bool result, logical OR/AND fold, False/True identity — the empty
        // input is False/True, not the "no identity" throw of max/min); the child is the nonzero-test
        // NotEqual(x, 0), so the reduce just folds a bool stream.  CountNonzero/NanSum/NanProd are pure
        // COMPOSITIONS over the (M1/M2-exact) Sum/Prod, so they inherit flat+axis+bit-exactness for free:
        // a rewrite is the whole implementation, no new kernel or host path.

        /// <summary>
        /// One-pass fused <c>np.any</c>: <c>True</c> iff any element of the expression is truthy
        /// (nonzero). Result is <c>bool</c> and the identity is <c>False</c>, so <c>Any</c> of an EMPTY
        /// input is <c>False</c> — unlike <see cref="Max(NDExpr)"/>, which raises on a zero-size input.
        /// NaN is truthy (<c>nan != 0</c>), so <c>Any([nan])</c> is <c>True</c>, matching NumPy.
        /// </summary>
        public static NDExpr Any(NDExpr x) => new ReduceNode(NDExprReduceKind.Any, NotEqual(x, Const(0)));

        /// <summary>
        /// One-pass fused <c>np.all</c>: <c>True</c> iff every element of the expression is truthy
        /// (nonzero). Result is <c>bool</c> and the identity is <c>True</c>, so <c>All</c> of an EMPTY
        /// input is <c>True</c> (vacuous truth) — unlike <see cref="Min(NDExpr)"/>, which raises on a
        /// zero-size input.
        /// </summary>
        public static NDExpr All(NDExpr x) => new ReduceNode(NDExprReduceKind.All, NotEqual(x, Const(0)));

        /// <summary>
        /// One-pass fused <c>np.count_nonzero</c>: the int64 count of truthy (nonzero) elements. Composed
        /// as <c>Sum(x != 0)</c> — a sum over the bool nonzero-test — so the result is int64 (NumPy's
        /// <c>intp</c>) at every child dtype, and an empty input counts to 0.
        /// </summary>
        public static NDExpr CountNonzero(NDExpr x) => new ReduceNode(NDExprReduceKind.Sum, NotEqual(x, Const(0)));

        /// <summary>
        /// One-pass fused <c>np.nansum</c>: the sum with NaNs treated as 0. Composed as
        /// <c>Sum(Where(IsNaN(x), 0, x))</c>, so it is BIT-EXACT wherever the plain <see cref="Sum(NDExpr)"/>
        /// is (float32/float64/complex via the M1 pairwise divert). An integer/bool child never carries a
        /// NaN — <c>IsNaN</c> is all-false there and the <c>Where</c> passes the child through unchanged —
        /// so <c>NanSum(int)</c> is exactly <c>Sum(int)</c> (int64), matching NumPy. An empty / all-NaN
        /// input sums to 0.
        /// </summary>
        public static NDExpr NanSum(NDExpr x) => new ReduceNode(NDExprReduceKind.Sum, Where(IsNaN(x), Const(0), x));

        /// <summary>
        /// One-pass fused <c>np.nanprod</c>: the product with NaNs treated as 1. Composed as
        /// <c>Prod(Where(IsNaN(x), 1, x))</c> — bit-exact wherever the plain <see cref="Prod(NDExpr)"/> is
        /// (float32/float64 via the M1 sequential divert), an identity pass-through for integer children,
        /// and 1 for an empty / all-NaN input, matching NumPy.
        /// </summary>
        public static NDExpr NanProd(NDExpr x) => new ReduceNode(NDExprReduceKind.Prod, Where(IsNaN(x), Const(1), x));

        // --- range / NaN-aware min-max reductions (plan P2 M4-tail) -------------------------------
        //
        // Unlike NanSum/NanProd these are NOT tree rewrites: they are their OWN kinds whose child is the
        // RAW expression, because their semantics need machinery an NDExpr cannot carry.  Ptp is two
        // reductions (max - min) and cannot be spelled as one root reduce; NanMin/NanMax must be
        // TYPE-AWARE (a float child's NaNs are replaced with the ±inf sentinel, an int child's are not —
        // an NDExpr rewrite spelling `Where(IsNaN(x), +inf, x)` would promote every int child to float64,
        // which NumPy does not).  All three are ORDER-INDEPENDENT (min / max / their difference are the
        // same whatever the summation order), so the host materializes the child once and reduces it
        // through the already-NumPy-exact engine reduction (np.ptp / np.nanmin / np.nanmax) — bit-exact
        // by construction, needing no pairwise divert and carrying no MisalignedRegistry excuse (unlike
        // the summation kinds).  See the host divert in DefaultEngine.EvaluateReduce.

        /// <summary>
        /// One-pass fused <c>np.ptp</c>: the peak-to-peak range <c>max - min</c> of the expression. The
        /// result dtype is the child's, so integer/unsigned overflow WRAPS exactly as NumPy's does, and a
        /// NaN anywhere in the child propagates to the result. There is NO empty identity — a zero-size
        /// input raises, like <see cref="Max(NDExpr)"/>.
        /// </summary>
        public static NDExpr Ptp(NDExpr x) => new ReduceNode(NDExprReduceKind.Ptp, x);

        /// <summary>
        /// One-pass fused <c>np.nanmin</c>: the minimum of the expression with NaNs ignored. On a
        /// float/complex child NaNs are skipped (an ALL-NaN input yields NaN); on an integer/bool child —
        /// which carries no NaN — it is exactly the plain minimum, preserving the child dtype. A zero-size
        /// input raises (no identity), like <see cref="Min(NDExpr)"/>.
        /// </summary>
        public static NDExpr NanMin(NDExpr x) => new ReduceNode(NDExprReduceKind.NanMin, x);

        /// <summary>
        /// One-pass fused <c>np.nanmax</c>: the maximum of the expression with NaNs ignored — the
        /// <see cref="NanMin(NDExpr)"/> twin.
        /// </summary>
        public static NDExpr NanMax(NDExpr x) => new ReduceNode(NDExprReduceKind.NanMax, x);

        // --- axis-aware forms of the M4 reductions (one pass along `axis`) ------------------------

        /// <summary>One-pass fused <c>np.any</c> along <paramref name="axis"/> (bool result; identity <c>False</c>).</summary>
        public static NDExpr Any(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Any, NotEqual(x, Const(0)), axis, keepdims);

        /// <summary>One-pass fused <c>np.all</c> along <paramref name="axis"/> (bool result; identity <c>True</c>).</summary>
        public static NDExpr All(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.All, NotEqual(x, Const(0)), axis, keepdims);

        /// <summary>One-pass fused <c>np.count_nonzero</c> along <paramref name="axis"/> (int64 count).</summary>
        public static NDExpr CountNonzero(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Sum, NotEqual(x, Const(0)), axis, keepdims);

        /// <summary>One-pass fused <c>np.nansum</c> along <paramref name="axis"/> (NaNs treated as 0).</summary>
        public static NDExpr NanSum(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Sum, Where(IsNaN(x), Const(0), x), axis, keepdims);

        /// <summary>One-pass fused <c>np.nanprod</c> along <paramref name="axis"/> (NaNs treated as 1).</summary>
        public static NDExpr NanProd(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Prod, Where(IsNaN(x), Const(1), x), axis, keepdims);

        /// <summary>One-pass fused <c>np.ptp</c> along <paramref name="axis"/> (max − min; dtype preserved, wrapping).</summary>
        public static NDExpr Ptp(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.Ptp, x, axis, keepdims);

        /// <summary>One-pass fused <c>np.nanmin</c> along <paramref name="axis"/> (NaNs ignored; all-NaN slice → NaN).</summary>
        public static NDExpr NanMin(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.NanMin, x, axis, keepdims);

        /// <summary>One-pass fused <c>np.nanmax</c> along <paramref name="axis"/> (NaNs ignored; all-NaN slice → NaN).</summary>
        public static NDExpr NanMax(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.NanMax, x, axis, keepdims);

        // ===================================================================
        // Binding
        // ===================================================================

        /// <summary>
        /// Rewrite array leaves into positional inputs, collecting the distinct
        /// arrays into <paramref name="ctx"/>. Returns the same instance when
        /// the subtree contains no array leaf.
        /// </summary>
        internal abstract NDExpr BindArrays(NDExprBindContext ctx);

        /// <summary>True if any node in the subtree is a <see cref="ReduceNode"/>.</summary>
        internal abstract bool ContainsReduce { get; }

        /// <summary>
        /// The first embedded <see cref="NDArray"/> leaf in evaluation order, or null for a tree
        /// built over positional inputs / constants. np.evaluate dispatches on its engine.
        /// </summary>
        internal virtual NDArray FirstArray() => null;
    }

    public sealed partial class InputNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx) => this;
        internal override bool ContainsReduce => false;
    }

    public sealed partial class ConstNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx) => this;
        internal override bool ContainsReduce => false;
    }

    public sealed partial class BinaryNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new BinaryNode(_op, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
        internal override NDArray FirstArray() => _left.FirstArray() ?? _right.FirstArray();
    }

    public sealed partial class UnaryNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new UnaryNode(_op, c);
        }

        internal override bool ContainsReduce => _child.ContainsReduce;
        internal override NDArray FirstArray() => _child.FirstArray();
    }

    public sealed partial class ComparisonNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new ComparisonNode(_op, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
        internal override NDArray FirstArray() => _left.FirstArray() ?? _right.FirstArray();
    }

    public sealed partial class MinMaxNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var l = _left.BindArrays(ctx);
            var r = _right.BindArrays(ctx);
            return ReferenceEquals(l, _left) && ReferenceEquals(r, _right)
                ? this
                : new MinMaxNode(_isMin, l, r);
        }

        internal override bool ContainsReduce => _left.ContainsReduce || _right.ContainsReduce;
        internal override NDArray FirstArray() => _left.FirstArray() ?? _right.FirstArray();
    }

    public sealed partial class WhereNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _cond.BindArrays(ctx);
            var a = _a.BindArrays(ctx);
            var b = _b.BindArrays(ctx);
            return ReferenceEquals(c, _cond) && ReferenceEquals(a, _a) && ReferenceEquals(b, _b)
                ? this
                : new WhereNode(c, a, b);
        }

        internal override bool ContainsReduce => _cond.ContainsReduce || _a.ContainsReduce || _b.ContainsReduce;
        internal override NDArray FirstArray() => _cond.FirstArray() ?? _a.FirstArray() ?? _b.FirstArray();
    }

    public sealed partial class CallNode
    {
        /// <summary>Clone with new args — reuses the registered slot/method, no re-registration.</summary>
        private CallNode(CallNode source, NDExpr[] args)
        {
            _kind = source._kind;
            _method = source._method;
            _delegateType = source._delegateType;
            _slotId = source._slotId;
            _args = args;
            _paramCodes = source._paramCodes;
            _returnCode = source._returnCode;
            _signatureId = source._signatureId;
        }

        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            NDExpr[]? rebound = null;
            for (int i = 0; i < _args.Length; i++)
            {
                var b = _args[i].BindArrays(ctx);
                if (!ReferenceEquals(b, _args[i]) && rebound is null)
                {
                    rebound = new NDExpr[_args.Length];
                    Array.Copy(_args, rebound, i);
                }

                if (rebound is not null)
                    rebound[i] = b;
            }

            return rebound is null ? this : new CallNode(this, rebound);
        }

        internal override bool ContainsReduce
        {
            get
            {
                foreach (var a in _args)
                    if (a.ContainsReduce)
                        return true;
                return false;
            }
        }

        internal override NDArray FirstArray()
        {
            foreach (var a in _args)
            {
                var arr = a.FirstArray();
                if (arr is not null)
                    return arr;
            }
            return null;
        }
    }

    // =========================================================================
    // Node: ArrayNode — an NDArray leaf, replaced by Input(i) during binding.
    // Never reaches typing or emission: np.evaluate always binds first.
    // =========================================================================

    [NDBorrowed] // an expression leaf references the caller's array; np.evaluate never owns its inputs
    public sealed partial class ArrayNode : NDExpr
    {
        private readonly NDArray _array;

        public ArrayNode(NDArray array)
            => _array = array ?? throw new ArgumentNullException(nameof(array));

        internal NDArray Array => _array;

        public override bool SupportsSimd => true;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before compilation — evaluate the tree via np.evaluate, " +
                "or rewrite array leaves to NDExpr.Input(i) and pass the arrays to the iterator.");

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before compilation — evaluate the tree via np.evaluate.");

        public override void AppendSignature(StringBuilder sb)
            => sb.Append("Arr[unbound]");

        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
            => throw new InvalidOperationException(
                "ArrayNode must be bound before typing — evaluate the tree via np.evaluate.");

        internal override NDExpr BindArrays(NDExprBindContext ctx)
            => new InputNode(ctx.IndexOf(_array));

        internal override bool ContainsReduce => false;
        internal override NDArray FirstArray() => _array;
    }

    // =========================================================================
    // Node: ReduceNode — root-only fused reduction over an elementwise tree.
    //
    // np.evaluate drives it as: iterate the INPUT operands only (no output
    // operand), run a raw accumulating inner loop that evaluates the child
    // tree per element and folds into a host-owned accumulator slot (aux).
    //
    // dtype rules (NumPy 2.4.2, probed):
    //   sum/prod: bool/int→int64, uint→uint64, floats preserved
    //   min/max:  input dtype preserved
    //   mean:     bool/int→float64, floats preserved
    //
    // Accumulation detail: float sums/products/means accumulate in float64
    // and cast back once at the end — tighter than NumPy's pairwise f32 loop,
    // so f32 results can differ from np.sum in the last ulps (documented).
    // Min/max accumulate at the exact result dtype (comparisons are exact)
    // and propagate NaN like np.min/np.max.
    // =========================================================================

    public sealed partial class ReduceNode : NDExpr
    {
        private readonly NDExprReduceKind _kind;
        private readonly NDExpr _child;
        private readonly int? _axis;       // null = flat (reduce-all); else reduce this axis
        private readonly bool _keepdims;

        public ReduceNode(NDExprReduceKind kind, NDExpr child, int? axis = null, bool keepdims = false)
        {
            _kind = kind;
            _child = child ?? throw new ArgumentNullException(nameof(child));
            _axis = axis;
            _keepdims = keepdims;
        }

        internal NDExprReduceKind Kind => _kind;
        internal NDExpr Child => _child;
        internal int? Axis => _axis;
        internal bool Keepdims => _keepdims;

        public override bool SupportsSimd => false;

        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                $"Reduction nodes are driven by np.evaluate as the tree root — " +
                $"{_kind} cannot be emitted as an elementwise value.");

        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                "Reduction nodes have no vector path — they are driven by np.evaluate.");

        public override void AppendSignature(StringBuilder sb)
        {
            sb.Append("Reduce").Append(_kind).Append('(');
            _child.AppendSignature(sb);
            sb.Append(')');
        }

        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new ReduceNode(_kind, c, _axis, _keepdims);
        }

        internal override bool ContainsReduce => true;
        internal override NDArray FirstArray() => _child.FirstArray();

        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            var ct = _child.InferType(inputTypes, nodeTypes);
            var childType = ResolveChild(_child, ct, ct.IsWeak ? ct.DefaultCode : ct.Code, nodeTypes);

            var result = ResolveReduceResultType(_kind, childType);
            nodeTypes[this] = result;
            return NDExprTypeInfo.Strong(result);
        }

        internal static NPTypeCode ResolveReduceResultType(NDExprReduceKind kind, NPTypeCode child)
        {
            switch (kind)
            {
                case NDExprReduceKind.Sum:
                case NDExprReduceKind.Prod:
                    return child switch
                    {
                        NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Int16 or
                        NPTypeCode.Int32 or NPTypeCode.Int64 => NPTypeCode.Int64,
                        NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.Char or
                        NPTypeCode.UInt32 or NPTypeCode.UInt64 => NPTypeCode.UInt64,
                        _ => child, // floats / Decimal / Complex preserved
                    };

                case NDExprReduceKind.Min:
                case NDExprReduceKind.Max:
                    // dtype preserved; complex folds lexicographically on (real, imag) with the
                    // first NaN sticking — np.min([3-4j, 0, nan]) is (nan+0j) — via the engine's
                    // ComplexMinNaN/ComplexMaxNaN clamp (see EmitFold).
                    return child;

                case NDExprReduceKind.Ptp:
                case NDExprReduceKind.NanMin:
                case NDExprReduceKind.NanMax:
                    // dtype preserved (np.ptp is amax-amin at the input dtype; np.nanmin/nanmax skip
                    // NaN without changing width) — the host delegates to those engine reductions over
                    // the materialized child, so the result dtype IS the child dtype.
                    return child;

                case NDExprReduceKind.Mean:
                    return child switch
                    {
                        NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or
                        NPTypeCode.Decimal or NPTypeCode.Complex => child,
                        _ => NPTypeCode.Double,
                    };

                case NDExprReduceKind.Any:
                case NDExprReduceKind.All:
                    // logical_or / logical_and always produce bool — the child is the bool nonzero-test
                    // (NotEqual(x, 0)), so `child` is already Boolean here, but be explicit.
                    return NPTypeCode.Boolean;

                default:
                    throw new NotSupportedException($"Unknown reduce kind {kind}.");
            }
        }

        /// <summary>
        /// Accumulator dtype: result dtype, except f16/f32 sums/products/means
        /// widen to f64 (cast back at the end — see class doc).
        /// </summary>
        internal static NPTypeCode ResolveAccType(NDExprReduceKind kind, NPTypeCode result)
        {
            // Min/Max and the host-delegated M4-tail kinds (Ptp/NanMin/NanMax) never run a widening
            // summation — Min/Max fold in place and the delegated kinds bypass the fold path entirely —
            // so the accumulator dtype is simply the result dtype (no f16/f32 → f64 widening).
            if (kind == NDExprReduceKind.Min || kind == NDExprReduceKind.Max
                || kind == NDExprReduceKind.Ptp || kind == NDExprReduceKind.NanMin || kind == NDExprReduceKind.NanMax)
                return result;
            return result == NPTypeCode.Half || result == NPTypeCode.Single
                ? NPTypeCode.Double
                : result;
        }

        /// <summary>
        /// Compile the one-pass accumulating inner loop. The kernel evaluates
        /// the child tree per element (4-way unrolled with 4 accumulators for
        /// ILP) and folds into <c>*(Tacc*)aux</c>; the host initializes aux
        /// with the reduction identity and reads it back after iteration.
        /// </summary>
        internal NDInnerLoopFunc CompileReduceKernel(
            NPTypeCode[] inputTypes,
            out NPTypeCode accType, out NPTypeCode resultType,
            string? cacheKey = null)
            => CompileReduceKernel(inputTypes, null, out accType, out resultType, cacheKey);

        /// <summary>
        /// <see cref="CompileReduceKernel(NPTypeCode[], out NPTypeCode, out NPTypeCode, string)"/> with a
        /// parameter mask (NDExpr.Params.cs): flagged inputs are loaded once from the aux block —
        /// AFTER the accumulator slot, at <see cref="NDExprParamPlan.ReduceParamOffset"/> — and only
        /// the unflagged inputs are iterator operands.
        /// </summary>
        /// <param name="inputTypes">Every input's dtype, parameters included, in input order.</param>
        /// <param name="isParam">Per input, whether it is a hoisted parameter; null for none.</param>
        /// <param name="accType">Receives the accumulator dtype the host seeds and reads back.</param>
        /// <param name="resultType">Receives the reduction's NumPy result dtype.</param>
        /// <param name="cacheKey">An explicit kernel cache key, or null to derive one from the tree.</param>
        /// <returns>The compiled (cached) accumulating inner loop.</returns>
        internal NDInnerLoopFunc CompileReduceKernel(
            NPTypeCode[] inputTypes, bool[]? isParam,
            out NPTypeCode accType, out NPTypeCode resultType,
            string? cacheKey = null)
        {
            var resolved = ResolveNumPyTypes(inputTypes, out var nodeTypes);
            resultType = resolved;
            var acc = ResolveAccType(_kind, resolved);
            accType = acc;
            var exprType = nodeTypes[_child];
            var plan = NDExprParamPlan.Create(inputTypes, isParam);
            var opTypes = plan.OperandTypes;
            int nIn = plan.OperandCount;                  // iterator operands only
            var kind = _kind;
            var child = _child;

            string key = (cacheKey ?? DeriveCacheKey(inputTypes, resolved)) + "|npreduce" + plan.KeySuffix;

            return DirectILKernelGenerator.CompileRawInnerLoop(il =>
            {
                // ---- locals -------------------------------------------------
                var ptrLocals = new LocalBuilder[nIn];
                var strideLocals = new LocalBuilder[nIn];
                var inputLocals = new LocalBuilder[nIn];
                for (int j = 0; j < nIn; j++)
                {
                    ptrLocals[j] = il.DeclareLocal(typeof(byte*));
                    strideLocals[j] = il.DeclareLocal(typeof(long));
                    inputLocals[j] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(opTypes[j]));
                }

                var accClr = DirectILKernelGenerator.GetClrType(acc);
                var accLocals = new LocalBuilder[4];
                for (int l = 0; l < 4; l++)
                    accLocals[l] = il.DeclareLocal(accClr);

                var locI = il.DeclareLocal(typeof(long));
                var locN4 = il.DeclareLocal(typeof(long));

                // Parameters: loaded once, here, from aux + 16 (slot 0 is the accumulator).
                LocalBuilder[]? paramLocals = null;
                if (plan.ParamCount > 0)
                {
                    paramLocals = new LocalBuilder[plan.ParamCount];
                    plan.EmitPrologue(il, paramLocals, null, NPTypeCode.Empty, NDExprParamPlan.ReduceParamOffset);
                }

                var ctx = new NDExprCompileContext(inputTypes, exprType, inputLocals, vectorMode: false, nodeTypes,
                    NPTypeCode.Empty, plan.Slots, plan.ParamIndex, paramLocals);

                // ---- prologue: unpack dataptrs / strides --------------------
                for (int j = 0; j < nIn; j++)
                {
                    il.Emit(OpCodes.Ldarg_0);
                    if (j > 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, j * sizeof(long));
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }

                    il.Emit(OpCodes.Ldind_I);
                    il.Emit(OpCodes.Stloc, ptrLocals[j]);

                    il.Emit(OpCodes.Ldarg_1);
                    if (j > 0)
                    {
                        il.Emit(OpCodes.Ldc_I4, j * sizeof(long));
                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                    }

                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stloc, strideLocals[j]);
                }

                // acc0 carries in from aux (running value across chunks);
                // acc1..acc3 start at the per-chunk identity: 0 for sum,
                // 1 for prod, and the CURRENT carry value for min/max
                // (idempotent under min/max, so no double counting).
                il.Emit(OpCodes.Ldarg_3);
                DirectILKernelGenerator.EmitLoadIndirect(il, acc);
                il.Emit(OpCodes.Stloc, accLocals[0]);
                for (int l = 1; l < 4; l++)
                {
                    switch (kind)
                    {
                        case NDExprReduceKind.Sum:
                        case NDExprReduceKind.Mean:
                            WhereNode.EmitPushZeroPublic(il, acc);
                            break;
                        case NDExprReduceKind.Prod:
                            il.Emit(OpCodes.Ldc_I4_1);
                            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, acc);
                            break;
                        case NDExprReduceKind.Any:
                            il.Emit(OpCodes.Ldc_I4_0); // logical_or per-chunk identity: False
                            break;
                        case NDExprReduceKind.All:
                            il.Emit(OpCodes.Ldc_I4_1); // logical_and per-chunk identity: True
                            break;
                        default: // Min / Max
                            il.Emit(OpCodes.Ldloc, accLocals[0]);
                            break;
                    }

                    il.Emit(OpCodes.Stloc, accLocals[l]);
                }

                // n4 = count & ~3
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Ldc_I8, ~3L);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stloc, locN4);

                // i = 0
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locI);

                void EmitLane(int lane)
                {
                    // load inputs for this lane into the shared input locals
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        if (lane > 0)
                        {
                            il.Emit(OpCodes.Ldloc, strideLocals[j]);
                            il.Emit(OpCodes.Ldc_I4, lane);
                            il.Emit(OpCodes.Conv_I8);
                            il.Emit(OpCodes.Mul);
                            il.Emit(OpCodes.Conv_I);
                            il.Emit(OpCodes.Add);
                        }

                        DirectILKernelGenerator.EmitLoadIndirect(il, opTypes[j]);
                        il.Emit(OpCodes.Stloc, inputLocals[j]);
                    }

                    // accLane = fold(accLane, (Tacc)expr)
                    il.Emit(OpCodes.Ldloc, accLocals[lane]);
                    child.EmitScalar(il, ctx);
                    DirectILKernelGenerator.EmitConvertTo(il, exprType, acc);
                    EmitFold(il, kind, acc);
                    il.Emit(OpCodes.Stloc, accLocals[lane]);
                }

                void EmitAdvance(int elements)
                {
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        il.Emit(OpCodes.Ldloc, strideLocals[j]);
                        if (elements != 1)
                        {
                            il.Emit(OpCodes.Ldc_I4, elements);
                            il.Emit(OpCodes.Conv_I8);
                            il.Emit(OpCodes.Mul);
                        }

                        il.Emit(OpCodes.Conv_I);
                        il.Emit(OpCodes.Add);
                        il.Emit(OpCodes.Stloc, ptrLocals[j]);
                    }
                }

                // ---- unrolled loop ------------------------------------------
                var lblLoop4 = il.DefineLabel();
                var lblLoop4End = il.DefineLabel();
                il.MarkLabel(lblLoop4);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldloc, locN4);
                il.Emit(OpCodes.Bge, lblLoop4End);

                for (int lane = 0; lane < 4; lane++)
                    EmitLane(lane);
                EmitAdvance(4);

                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 4L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblLoop4);
                il.MarkLabel(lblLoop4End);

                // ---- scalar tail --------------------------------------------
                var lblTail = il.DefineLabel();
                var lblTailEnd = il.DefineLabel();
                il.MarkLabel(lblTail);
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldarg_2);
                il.Emit(OpCodes.Bge, lblTailEnd);

                EmitLane(0);
                EmitAdvance(1);

                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Ldc_I8, 1L);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblTail);
                il.MarkLabel(lblTailEnd);

                // ---- write back: *aux = fold(acc0, acc1, acc2, acc3) --------
                il.Emit(OpCodes.Ldarg_3);
                il.Emit(OpCodes.Ldloc, accLocals[0]);
                for (int l = 1; l < 4; l++)
                {
                    il.Emit(OpCodes.Ldloc, accLocals[l]);
                    EmitFold(il, kind, acc);
                }

                DirectILKernelGenerator.EmitStoreIndirect(il, acc);
                il.Emit(OpCodes.Ret);
            }, key);
        }

        /// <summary>
        /// Compile the AXIS-aware fused reduce kernel. Operands are [inputs…, output];
        /// the output operand carries the running accumulator (the host pre-seeds it with
        /// the reduction identity). Two runtime branches keyed on the output stride:
        ///   • outStride == 0 → PINNED (reduce axis is the inner loop): fold the whole
        ///     `count`-element stripe into the single output slot, 4-way unrolled (ILP) —
        ///     the same shape as the flat <see cref="CompileReduceKernel"/> but writing the
        ///     output operand instead of a host scalar.
        ///   • outStride != 0 → SLAB (a kept axis is inner): each element folds into a
        ///     DISTINCT output slot, `out[c] = fold(out[c], expr(in[c]))`. Revisited across
        ///     outer iterations, so the pre-seed is what makes the accumulation correct.
        /// The child elementwise tree is evaluated per element exactly as in the flat path,
        /// so e.g. evaluate(Sum(a*b, axis:k)) never materializes a*b.
        /// </summary>
        internal NDInnerLoopFunc CompileAxisReduceKernel(
            NPTypeCode[] inputTypes, out NPTypeCode accType, out NPTypeCode resultType,
            string? cacheKey = null)
            => CompileAxisReduceKernel(inputTypes, null, out accType, out resultType, cacheKey);

        /// <summary>
        /// <see cref="CompileAxisReduceKernel(NPTypeCode[], out NPTypeCode, out NPTypeCode, string)"/> with
        /// a parameter mask (NDExpr.Params.cs): flagged inputs are loaded once from the aux block at
        /// <see cref="NDExprParamPlan.ReduceParamOffset"/> (the same layout as the flat kernel, so the
        /// host packs one buffer for both), and only the unflagged inputs are iterator operands.
        /// </summary>
        /// <param name="inputTypes">Every input's dtype, parameters included, in input order.</param>
        /// <param name="isParam">Per input, whether it is a hoisted parameter; null for none.</param>
        /// <param name="accType">Receives the accumulator dtype the host seeds the output with.</param>
        /// <param name="resultType">Receives the reduction's NumPy result dtype.</param>
        /// <param name="cacheKey">An explicit kernel cache key, or null to derive one from the tree.</param>
        /// <returns>The compiled (cached) axis-aware accumulating inner loop.</returns>
        internal NDInnerLoopFunc CompileAxisReduceKernel(
            NPTypeCode[] inputTypes, bool[]? isParam,
            out NPTypeCode accType, out NPTypeCode resultType,
            string? cacheKey = null)
        {
            var resolved = ResolveNumPyTypes(inputTypes, out var nodeTypes);
            resultType = resolved;
            var acc = ResolveAccType(_kind, resolved);
            accType = acc;
            var exprType = nodeTypes[_child];
            var plan = NDExprParamPlan.Create(inputTypes, isParam);
            var opTypes = plan.OperandTypes;
            int nIn = plan.OperandCount;                  // iterator operands only
            var kind = _kind;
            var child = _child;

            string key = (cacheKey ?? DeriveCacheKey(inputTypes, resolved)) + "|npaxisreduce" + plan.KeySuffix;

            return DirectILKernelGenerator.CompileRawInnerLoop(il =>
            {
                var accClr = DirectILKernelGenerator.GetClrType(acc);
                var ptrLocals = new LocalBuilder[nIn];
                var strideLocals = new LocalBuilder[nIn];
                var inputLocals = new LocalBuilder[nIn];
                for (int j = 0; j < nIn; j++)
                {
                    ptrLocals[j] = il.DeclareLocal(typeof(byte*));
                    strideLocals[j] = il.DeclareLocal(typeof(long));
                    inputLocals[j] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(opTypes[j]));
                }
                var outPtr = il.DeclareLocal(typeof(byte*));
                var outStride = il.DeclareLocal(typeof(long));
                var locI = il.DeclareLocal(typeof(long));
                var locN4 = il.DeclareLocal(typeof(long));
                var accLocals = new LocalBuilder[4];
                for (int l = 0; l < 4; l++) accLocals[l] = il.DeclareLocal(accClr);

                // Parameters: loaded once, here, from aux + 16 (the flat kernel's accumulator slot is
                // reserved in both layouts so the host packs one buffer).
                LocalBuilder[]? paramLocals = null;
                if (plan.ParamCount > 0)
                {
                    paramLocals = new LocalBuilder[plan.ParamCount];
                    plan.EmitPrologue(il, paramLocals, null, NPTypeCode.Empty, NDExprParamPlan.ReduceParamOffset);
                }

                var ctx = new NDExprCompileContext(inputTypes, exprType, inputLocals, vectorMode: false, nodeTypes,
                    NPTypeCode.Empty, plan.Slots, plan.ParamIndex, paramLocals);

                // prologue: unpack input ptrs/strides (0..nIn-1) and the output ptr/stride (nIn).
                for (int j = 0; j <= nIn; j++)
                {
                    il.Emit(OpCodes.Ldarg_0);
                    if (j > 0) { il.Emit(OpCodes.Ldc_I4, j * sizeof(long)); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); }
                    il.Emit(OpCodes.Ldind_I);
                    il.Emit(OpCodes.Stloc, j < nIn ? ptrLocals[j] : outPtr);

                    il.Emit(OpCodes.Ldarg_1);
                    if (j > 0) { il.Emit(OpCodes.Ldc_I4, j * sizeof(long)); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); }
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Stloc, j < nIn ? strideLocals[j] : outStride);
                }

                void EmitLoadInputs(int lane)
                {
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        if (lane > 0)
                        {
                            il.Emit(OpCodes.Ldloc, strideLocals[j]);
                            il.Emit(OpCodes.Ldc_I4, lane); il.Emit(OpCodes.Conv_I8); il.Emit(OpCodes.Mul);
                            il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                        }
                        DirectILKernelGenerator.EmitLoadIndirect(il, opTypes[j]);
                        il.Emit(OpCodes.Stloc, inputLocals[j]);
                    }
                }

                void EmitAdvanceInputs(int elements)
                {
                    for (int j = 0; j < nIn; j++)
                    {
                        il.Emit(OpCodes.Ldloc, ptrLocals[j]);
                        il.Emit(OpCodes.Ldloc, strideLocals[j]);
                        if (elements != 1) { il.Emit(OpCodes.Ldc_I4, elements); il.Emit(OpCodes.Conv_I8); il.Emit(OpCodes.Mul); }
                        il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add);
                        il.Emit(OpCodes.Stloc, ptrLocals[j]);
                    }
                }

                var lblSlab = il.DefineLabel();
                var lblEnd = il.DefineLabel();

                // if (outStride != 0) goto SLAB
                il.Emit(OpCodes.Ldloc, outStride);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblSlab);

                // ===================== PINNED =====================
                // acc0 = *(Tacc*)outPtr (seeded identity); acc1..3 = per-chunk identity.
                il.Emit(OpCodes.Ldloc, outPtr);
                DirectILKernelGenerator.EmitLoadIndirect(il, acc);
                il.Emit(OpCodes.Stloc, accLocals[0]);
                for (int l = 1; l < 4; l++)
                {
                    switch (kind)
                    {
                        case NDExprReduceKind.Sum:
                        case NDExprReduceKind.Mean:
                            WhereNode.EmitPushZeroPublic(il, acc);
                            break;
                        case NDExprReduceKind.Prod:
                            il.Emit(OpCodes.Ldc_I4_1);
                            DirectILKernelGenerator.EmitConvertTo(il, NPTypeCode.Int32, acc);
                            break;
                        case NDExprReduceKind.Any:
                            il.Emit(OpCodes.Ldc_I4_0); // logical_or per-chunk identity: False
                            break;
                        case NDExprReduceKind.All:
                            il.Emit(OpCodes.Ldc_I4_1); // logical_and per-chunk identity: True
                            break;
                        default:
                            il.Emit(OpCodes.Ldloc, accLocals[0]);
                            break;
                    }
                    il.Emit(OpCodes.Stloc, accLocals[l]);
                }

                il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Ldc_I8, ~3L); il.Emit(OpCodes.And); il.Emit(OpCodes.Stloc, locN4);
                il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);

                void EmitPinnedLane(int lane)
                {
                    EmitLoadInputs(lane);
                    il.Emit(OpCodes.Ldloc, accLocals[lane]);
                    child.EmitScalar(il, ctx);
                    DirectILKernelGenerator.EmitConvertTo(il, exprType, acc);
                    EmitFold(il, kind, acc);
                    il.Emit(OpCodes.Stloc, accLocals[lane]);
                }

                var lblLoop4 = il.DefineLabel();
                var lblLoop4End = il.DefineLabel();
                il.MarkLabel(lblLoop4);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldloc, locN4); il.Emit(OpCodes.Bge, lblLoop4End);
                for (int lane = 0; lane < 4; lane++) EmitPinnedLane(lane);
                EmitAdvanceInputs(4);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 4L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblLoop4);
                il.MarkLabel(lblLoop4End);

                var lblTail = il.DefineLabel();
                var lblTailEnd = il.DefineLabel();
                il.MarkLabel(lblTail);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, lblTailEnd);
                EmitPinnedLane(0);
                EmitAdvanceInputs(1);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblTail);
                il.MarkLabel(lblTailEnd);

                // *(Tacc*)outPtr = fold(acc0..3)
                il.Emit(OpCodes.Ldloc, outPtr);
                il.Emit(OpCodes.Ldloc, accLocals[0]);
                for (int l = 1; l < 4; l++) { il.Emit(OpCodes.Ldloc, accLocals[l]); EmitFold(il, kind, acc); }
                DirectILKernelGenerator.EmitStoreIndirect(il, acc);
                il.Emit(OpCodes.Br, lblEnd);

                // ===================== SLAB =====================
                il.MarkLabel(lblSlab);
                il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
                var lblSlabLoop = il.DefineLabel();
                il.MarkLabel(lblSlabLoop);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldarg_2); il.Emit(OpCodes.Bge, lblEnd);
                EmitLoadInputs(0);
                // *(Tacc*)outPtr = fold( *(Tacc*)outPtr, (Tacc)expr )
                il.Emit(OpCodes.Ldloc, outPtr);                                  // store addr
                il.Emit(OpCodes.Ldloc, outPtr);
                DirectILKernelGenerator.EmitLoadIndirect(il, acc);              // cur
                child.EmitScalar(il, ctx);
                DirectILKernelGenerator.EmitConvertTo(il, exprType, acc);       // val
                EmitFold(il, kind, acc);                                         // fold(cur,val)
                DirectILKernelGenerator.EmitStoreIndirect(il, acc);
                EmitAdvanceInputs(1);
                // outPtr += outStride
                il.Emit(OpCodes.Ldloc, outPtr); il.Emit(OpCodes.Ldloc, outStride); il.Emit(OpCodes.Conv_I); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, outPtr);
                il.Emit(OpCodes.Ldloc, locI); il.Emit(OpCodes.Ldc_I8, 1L); il.Emit(OpCodes.Add); il.Emit(OpCodes.Stloc, locI);
                il.Emit(OpCodes.Br, lblSlabLoop);

                il.MarkLabel(lblEnd);
                il.Emit(OpCodes.Ret);
            }, key);
        }

        /// <summary>
        /// Fold [acc, value] → [acc'] at the accumulator dtype. Sum/Prod reuse
        /// the binary scalar emitters (full 15-dtype coverage); min/max use
        /// Math.Min/Max (NaN-propagating), with And/Or for bool and a
        /// double-roundtrip for Half (no Math.Min(Half) overload — the
        /// roundtrip is exact and keeps NaN propagation).
        /// </summary>
        private static void EmitFold(ILGenerator il, NDExprReduceKind kind, NPTypeCode acc)
        {
            switch (kind)
            {
                case NDExprReduceKind.Sum:
                case NDExprReduceKind.Mean:
                    if (acc == NPTypeCode.Boolean)
                    {
                        il.Emit(OpCodes.Or);
                        return;
                    }

                    DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Add, acc);
                    return;

                case NDExprReduceKind.Prod:
                    if (acc == NPTypeCode.Boolean)
                    {
                        il.Emit(OpCodes.And);
                        return;
                    }

                    DirectILKernelGenerator.EmitScalarOperation(il, BinaryOp.Multiply, acc);
                    return;

                // logical_or / logical_and over the bool accumulator: a bitwise OR / AND on 0/1 lanes.
                // Associative AND idempotent, so the 4-accumulator unroll reorders it with no drift.
                case NDExprReduceKind.Any:
                    il.Emit(OpCodes.Or);
                    return;
                case NDExprReduceKind.All:
                    il.Emit(OpCodes.And);
                    return;
            }

            bool isMin = kind == NDExprReduceKind.Min;

            if (acc == NPTypeCode.Boolean)
            {
                il.Emit(isMin ? OpCodes.And : OpCodes.Or);
                return;
            }

            // [acc, val] → np.minimum/np.maximum(acc, val): the ufunc's own scalar clamp, the body
            // np.minimum.reduce / np.maximum.reduce fold with — NaN-propagating with the FIRST NaN
            // sticking (acc is the first operand), the second operand on a ±0 / equal tie, the
            // lexicographic (real, imag) order for complex, Half / char / decimal covered. One body
            // for every dtype, shared with the elementwise kernels.
            DirectILKernelGenerator.EmitScalarOperation(il, isMin ? BinaryOp.Minimum : BinaryOp.Maximum, acc);
        }
    }
}
