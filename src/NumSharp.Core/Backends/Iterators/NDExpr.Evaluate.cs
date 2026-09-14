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

        /// <summary>
        /// <c>np.argmax</c> — the index of the maximum. Unlike every value reduction above, the result
        /// dtype is <b>int64</b> (NumPy's <c>intp</c>), NOT the child dtype. It is host-delegated to
        /// <see cref="np.argmax(NDArray,int,bool)"/> (flat: <see cref="np.argmax(NDArray)"/>) over the
        /// materialized child. This is bit-exact for the same reason the M4-tail value kinds are —
        /// but the property in play is subtler than order-independence: argmax's index DOES depend on
        /// visiting order at a tie (the FIRST maximum wins, and a NaN is treated as the largest so the
        /// FIRST NaN wins), yet the materialized child is a fresh C-CONTIGUOUS buffer holding exactly
        /// NumPy's own C-order intermediate, and <c>np.argmax</c> walks it in that same C-order — so the
        /// tie/NaN index is deterministic and identical to <c>np.argmax(child)</c>. There is NO empty
        /// identity (a zero-size input raises, like <see cref="Max"/>). See <c>EvaluateDelegatingReduce</c>.
        /// </summary>
        ArgMax,

        /// <summary>
        /// <c>np.argmin</c> — the index of the minimum (the <see cref="ArgMax"/> twin: int64 result,
        /// first-tie / first-NaN wins over the C-order materialized child). Host-delegated to
        /// <see cref="np.argmin(NDArray,int,bool)"/> / <see cref="np.argmin(NDArray)"/>.
        /// </summary>
        ArgMin,

        /// <summary>
        /// <c>np.nanmean</c> — the arithmetic mean with NaNs skipped (the <see cref="Mean"/> twin over
        /// the finite elements). Result dtype follows <see cref="Mean"/> (float16→float16 via a float32
        /// intermediate, float32→float32, float64→float64, complex128→complex128, int/bool→float64), an
        /// ALL-NaN input yields NaN and an integer/bool child — carrying no NaN — is exactly <c>Mean</c>.
        /// Unlike <see cref="Sum"/>/<see cref="Mean"/> it is NOT a fold kind and NOT a factory rewrite
        /// (its per-element NaN skip changes the divisor per slab): the host materializes the child once
        /// and computes <c>nansum(child) / count_of_non_NaN</c> with the SAME NumPy-exact pairwise sum the
        /// M1/M2 diverts use — see <c>DefaultEngine.EvaluateStatReduce</c>. Bit-exact vs NumPy 2.4.2, so
        /// it carries NO MisalignedRegistry excuse.
        /// </summary>
        NanMean,

        /// <summary>
        /// <c>np.var</c> — the variance <c>Σ(x−mean)² / (N−ddof)</c> (population variance at the default
        /// <c>ddof=0</c>). Result dtype: float16→float16 (computed in float32), float32→float32,
        /// float64→float64, int/bool→float64, and — like NumPy — a complex128 child yields a <b>real</b>
        /// float64 variance (the deviations are reduced as <c>|x−mean|²</c>). A NaN anywhere PROPAGATES
        /// (var uses every element, unlike <see cref="NanMean"/>); <c>ddof ≥ N</c> divides by a clamped
        /// zero → <c>+inf</c> (or NaN when the numerator is zero), and an empty input yields NaN. The host
        /// materializes the child once, then runs NumPy's own two-pass <c>_var</c> op for op over the
        /// exact pairwise sum (mean, then <c>Σ(x−mean)²</c>) — see <c>DefaultEngine.EvaluateStatReduce</c>.
        /// The degrees-of-freedom offset is carried on the node (<see cref="ReduceNode.Ddof"/>) and is
        /// part of the program identity. Bit-exact vs NumPy 2.4.2, NO excuse.
        /// </summary>
        Var,

        /// <summary>
        /// <c>np.std</c> — the standard deviation <c>sqrt(var)</c> (the <see cref="Var"/> twin, same
        /// result dtypes / ddof / NaN-propagation / empty→NaN semantics), applied as a final elementwise
        /// square root over the variance so it is bit-exact wherever <see cref="Var"/> is.
        /// </summary>
        Std,
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

        // --- index reductions (plan P2 M4c) -------------------------------------------------------
        //
        // ArgMax/ArgMin are the M4c kinds that land NOW (unlike NanMean/Std/Var/weighted Average, which
        // are summation-bound and wait on M3's streaming pairwise): they are ORDER-DETERMINISTIC over a
        // C-order buffer rather than value-order-independent, so they ride the SAME host-delegation the
        // M4-tail min/max kinds do — materialize the child once, then reduce it through the engine's
        // already-NumPy-exact np.argmax / np.argmin.  Two things set them apart from the value kinds:
        // the result dtype is int64 (an INDEX, not a child-typed value), and the answer depends on the
        // C-order tie/NaN rule (first maximum / first NaN wins) — which is bit-exact only because the
        // materialized child is the fresh C-contiguous buffer NumPy's own argmax would build.  Because
        // that index is deterministic, they carry NO MisalignedRegistry excuse.

        /// <summary>
        /// One-pass fused <c>np.argmax</c>: the <b>int64</b> index of the maximum element of the
        /// expression (flattened). Ties break to the FIRST occurrence and a NaN is treated as the
        /// largest value (so the first NaN's index wins), matching NumPy over the C-order materialized
        /// child. A zero-size input raises (no identity), like <see cref="Max(NDExpr)"/>.
        /// </summary>
        public static NDExpr ArgMax(NDExpr x) => new ReduceNode(NDExprReduceKind.ArgMax, x);

        /// <summary>
        /// One-pass fused <c>np.argmin</c>: the <b>int64</b> index of the minimum element of the
        /// expression (flattened) — the <see cref="ArgMax(NDExpr)"/> twin (first-tie / first-NaN wins).
        /// </summary>
        public static NDExpr ArgMin(NDExpr x) => new ReduceNode(NDExprReduceKind.ArgMin, x);

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

        /// <summary>One-pass fused <c>np.argmax</c> along <paramref name="axis"/> (int64 indices; first-tie / first-NaN wins).</summary>
        public static NDExpr ArgMax(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.ArgMax, x, axis, keepdims);

        /// <summary>One-pass fused <c>np.argmin</c> along <paramref name="axis"/> (int64 indices; first-tie / first-NaN wins).</summary>
        public static NDExpr ArgMin(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.ArgMin, x, axis, keepdims);

        // --- mean / variance / std reductions (plan P2 M4c summation kinds) -----------------------
        //
        // These are the summation-bound M4c kinds.  Unlike the M4-tail min/max (order-INDEPENDENT) they
        // are two-pass and drift through the engine's own np.var/np.nanmean (whose flat sum is a
        // multi-accumulator SIMD fold, not NumPy's pairwise), so they can NOT be delegated to those.
        // Instead the host materializes the child once and reproduces NumPy's _var / nanmean op for op
        // over the SAME NumPy-exact pairwise sum the M1/M2 diverts use — bit-exact by construction,
        // needing no fold kernel (EmitFold is untouched) and carrying NO MisalignedRegistry excuse.
        // (Weighted np.average is NOT here: it reduces over TWO operands — values and weights — which the
        // single-child ReduceNode cannot carry; it awaits a two-operand reduce node.)

        /// <summary>
        /// One-pass fused <c>np.nanmean</c>: the arithmetic mean with NaNs skipped. Result dtype follows
        /// <see cref="Mean(NDExpr)"/> (int/bool→float64, float16→float16, float32/float64 preserved,
        /// complex128 preserved); an ALL-NaN input yields NaN, and an integer/bool child (no NaN) is
        /// exactly <see cref="Mean(NDExpr)"/>.
        /// </summary>
        public static NDExpr NanMean(NDExpr x) => new ReduceNode(NDExprReduceKind.NanMean, x);

        /// <summary>One-pass fused <c>np.nanmean</c> along <paramref name="axis"/> (NaNs skipped per slab).</summary>
        public static NDExpr NanMean(NDExpr x, int axis, bool keepdims = false) => new ReduceNode(NDExprReduceKind.NanMean, x, axis, keepdims);

        /// <summary>
        /// One-pass fused <c>np.var</c>: the variance with <paramref name="ddof"/> delta degrees of
        /// freedom (<c>Σ(x−mean)² / (N−ddof)</c>; <c>ddof=0</c> is the population variance). Result dtype:
        /// int/bool→float64, float16→float16, float32/float64 preserved, and a complex128 child yields a
        /// <b>real</b> float64 variance. A NaN PROPAGATES; <c>ddof ≥ N</c> → <c>+inf</c>; empty → NaN.
        /// </summary>
        /// <param name="x">The expression to reduce.</param>
        /// <param name="ddof">Delta degrees of freedom; the divisor is <c>max(N − ddof, 0)</c>.</param>
        public static NDExpr Var(NDExpr x, int ddof = 0) => new ReduceNode(NDExprReduceKind.Var, x, ddof: ddof);

        /// <summary>One-pass fused <c>np.var</c> along <paramref name="axis"/> with <paramref name="ddof"/> delta degrees of freedom.</summary>
        public static NDExpr Var(NDExpr x, int axis, bool keepdims = false, int ddof = 0) => new ReduceNode(NDExprReduceKind.Var, x, axis, keepdims, ddof);

        /// <summary>
        /// One-pass fused <c>np.std</c>: the standard deviation <c>sqrt(var)</c> with <paramref name="ddof"/>
        /// delta degrees of freedom (the <see cref="Var(NDExpr,int)"/> twin — same dtypes, NaN-propagation
        /// and empty→NaN semantics, a real float64 for a complex128 child).
        /// </summary>
        /// <param name="x">The expression to reduce.</param>
        /// <param name="ddof">Delta degrees of freedom; the divisor is <c>max(N − ddof, 0)</c>.</param>
        public static NDExpr Std(NDExpr x, int ddof = 0) => new ReduceNode(NDExprReduceKind.Std, x, ddof: ddof);

        /// <summary>One-pass fused <c>np.std</c> along <paramref name="axis"/> with <paramref name="ddof"/> delta degrees of freedom.</summary>
        public static NDExpr Std(NDExpr x, int axis, bool keepdims = false, int ddof = 0) => new ReduceNode(NDExprReduceKind.Std, x, axis, keepdims, ddof);

        // --- weighted average (plan P2 M4c-average) -----------------------------------------------
        //
        // np.average(values, weights) = Σ(values·weights) / Σ(weights) — the FIRST reduction that
        // reduces over TWO operand trees, which the single-child ReduceNode cannot carry.  It is its
        // own node (WeightedAverageNode) rather than a factory rewrite because both sums must run at
        // ONE result dtype that neither child's natural dtype equals (NumPy forces the product AND both
        // sums to `result_dtype`, e.g. int·int → a float64 product, never the wrapping int64 one), and
        // because the division of two reductions is not itself an elementwise node.  The numerics are
        // already available: both sums are the M1/M2-exact pairwise, so this is a node-shape change, not
        // a new kernel — the host materializes both children once and reproduces NumPy's own average op
        // for op (see DefaultEngine.EvaluateWeightedAverage), BIT-EXACT for the Single/Double/Complex
        // result dtypes with NO MisalignedRegistry excuse; Half/Decimal are rejected (the "no float16
        // pairwise kernel" gap M1/M2/M4c-summation share).

        /// <summary>
        /// One-pass fused <c>np.average</c>: the weighted mean <c>Σ(values·weights) / Σ(weights)</c> of
        /// two expression trees (flattened). The two trees BROADCAST together like any fused binary, and
        /// the result matches <c>np.average(V, weights=W)</c> on same-shape operands. The result dtype is
        /// <b>float/complex</b> (NumPy's rule, keyed on the VALUES dtype: an integer/bool values tree
        /// promotes to <c>result_type(values, weights, float64)</c>, else <c>result_type(values,
        /// weights)</c>) — so an integer average is always at least float64, and the product/sums run at
        /// that dtype (no integer wrap). A NaN in either tree PROPAGATES to NaN. Weights that sum to zero
        /// (an empty input included) raise <see cref="DivideByZeroException"/> "Weights sum to zero, can't
        /// be normalized", exactly as <see cref="np.average(NDArray,int?,NDArray,bool,bool)"/> does. A
        /// float16 or decimal result dtype is rejected with a directed <see cref="NotSupportedException"/>
        /// (a bit-exact reduction there needs a pairwise sum kernel that dtype lacks).
        /// </summary>
        /// <param name="values">The value expression to average.</param>
        /// <param name="weights">The weight expression; each value is weighted by the aligned weight, and the sum is normalized by the total weight.</param>
        /// <returns>The fused weighted-average node (root-only, like every reduction).</returns>
        public static NDExpr Average(NDExpr values, NDExpr weights) => new WeightedAverageNode(values, weights);

        /// <summary>
        /// One-pass fused <c>np.average</c> along <paramref name="axis"/>: <c>Σ(values·weights) /
        /// Σ(weights)</c> reduced over a single axis (the <see cref="Average(NDExpr,NDExpr)"/> twin — same
        /// result dtype, NaN propagation, and zero-weight <see cref="DivideByZeroException"/>). Both sums
        /// reduce along <paramref name="axis"/>; <paramref name="keepdims"/> re-inserts it as size 1.
        /// </summary>
        /// <param name="values">The value expression to average.</param>
        /// <param name="weights">The weight expression aligned with <paramref name="values"/>.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end).</param>
        /// <param name="keepdims">When true, the reduced axis is kept as a size-1 dimension.</param>
        /// <returns>The fused weighted-average node reducing along <paramref name="axis"/>.</returns>
        public static NDExpr Average(NDExpr values, NDExpr weights, int axis, bool keepdims = false)
            => new WeightedAverageNode(values, weights, axis, keepdims);

        // --- axis=None + keepdims flat forms (plan P2 M5) -----------------------------------------
        //
        // NumPy's reductions take keepdims with axis=None too: np.sum(a, keepdims=True) reduces the
        // whole array yet returns shape (1,)*a.ndim — the broadcast-friendly form behind idioms like
        // `a / a.sum(axis=None, keepdims=True)`. The flat factories above return a 0-d scalar
        // (keepdims=False); these are their keepdims twins. The bool is REQUIRED (no default) on
        // purpose: `Sum(x)` must stay unambiguously the 0-d form, so `Sum(x, keepdims: true)` — a
        // required argument — is the only way into the size-1-everywhere form (a defaulted overload
        // would collide with `Sum(x)`). The host reshapes the 0-d result to (1,)*childNdim, so the
        // VALUE is identical to the 0-d form and only the wrapper rank differs; a 0-d child stays 0-d
        // (np.sum(scalar, keepdims=True) is 0-d), matching NumPy. TUPLE axis is deliberately NOT
        // offered — every NumSharp reduction is single-axis (int?), so multi-axis is a library-wide
        // gap, not an evaluate-specific one (plan P2 M5, deferred).

        /// <summary>One-pass fused <c>np.sum(x, axis=None, keepdims=True)</c> — the whole-array sum kept as shape (1,…,1) for broadcasting back against the input.</summary>
        public static NDExpr Sum(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Sum, x, null, keepdims);

        /// <summary>One-pass fused <c>np.prod(x, axis=None, keepdims=True)</c> — the whole-array product kept as shape (1,…,1).</summary>
        public static NDExpr Prod(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Prod, x, null, keepdims);

        /// <summary>One-pass fused <c>np.min(x, axis=None, keepdims=True)</c> — the whole-array minimum kept as shape (1,…,1) (NaN-propagating).</summary>
        public static NDExpr Min(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Min, x, null, keepdims);

        /// <summary>One-pass fused <c>np.max(x, axis=None, keepdims=True)</c> — the whole-array maximum kept as shape (1,…,1) (NaN-propagating).</summary>
        public static NDExpr Max(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Max, x, null, keepdims);

        /// <summary>One-pass fused <c>np.mean(x, axis=None, keepdims=True)</c> — the whole-array mean kept as shape (1,…,1).</summary>
        public static NDExpr Mean(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Mean, x, null, keepdims);

        /// <summary>One-pass fused <c>np.any(x, axis=None, keepdims=True)</c> — the whole-array presence test kept as shape (1,…,1) (bool).</summary>
        public static NDExpr Any(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Any, NotEqual(x, Const(0)), null, keepdims);

        /// <summary>One-pass fused <c>np.all(x, axis=None, keepdims=True)</c> — the whole-array universal test kept as shape (1,…,1) (bool).</summary>
        public static NDExpr All(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.All, NotEqual(x, Const(0)), null, keepdims);

        /// <summary>One-pass fused <c>np.count_nonzero(x, axis=None, keepdims=True)</c> — the whole-array nonzero count kept as shape (1,…,1) (int64).</summary>
        public static NDExpr CountNonzero(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Sum, NotEqual(x, Const(0)), null, keepdims);

        /// <summary>One-pass fused <c>np.nansum(x, axis=None, keepdims=True)</c> — the whole-array NaN-skipping sum kept as shape (1,…,1).</summary>
        public static NDExpr NanSum(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Sum, Where(IsNaN(x), Const(0), x), null, keepdims);

        /// <summary>One-pass fused <c>np.nanprod(x, axis=None, keepdims=True)</c> — the whole-array NaN-skipping product kept as shape (1,…,1).</summary>
        public static NDExpr NanProd(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Prod, Where(IsNaN(x), Const(1), x), null, keepdims);

        /// <summary>One-pass fused <c>np.ptp(x, axis=None, keepdims=True)</c> — the whole-array peak-to-peak range kept as shape (1,…,1) (dtype preserved, wrapping).</summary>
        public static NDExpr Ptp(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.Ptp, x, null, keepdims);

        /// <summary>One-pass fused <c>np.nanmin(x, axis=None, keepdims=True)</c> — the whole-array NaN-skipping minimum kept as shape (1,…,1).</summary>
        public static NDExpr NanMin(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.NanMin, x, null, keepdims);

        /// <summary>One-pass fused <c>np.nanmax(x, axis=None, keepdims=True)</c> — the whole-array NaN-skipping maximum kept as shape (1,…,1).</summary>
        public static NDExpr NanMax(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.NanMax, x, null, keepdims);

        /// <summary>One-pass fused <c>np.argmax(x, axis=None, keepdims=True)</c> — the flat int64 index of the maximum kept as shape (1,…,1) (first-tie / first-NaN wins).</summary>
        public static NDExpr ArgMax(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.ArgMax, x, null, keepdims);

        /// <summary>One-pass fused <c>np.argmin(x, axis=None, keepdims=True)</c> — the flat int64 index of the minimum kept as shape (1,…,1) (first-tie / first-NaN wins).</summary>
        public static NDExpr ArgMin(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.ArgMin, x, null, keepdims);

        /// <summary>One-pass fused <c>np.nanmean(x, axis=None, keepdims=True)</c> — the whole-array NaN-skipping mean kept as shape (1,…,1).</summary>
        public static NDExpr NanMean(NDExpr x, bool keepdims) => new ReduceNode(NDExprReduceKind.NanMean, x, null, keepdims);

        /// <summary>One-pass fused <c>np.var(x, axis=None, keepdims=True, ddof=ddof)</c> — the whole-array variance kept as shape (1,…,1) (real float64 for a complex child).</summary>
        /// <param name="x">The expression to reduce.</param>
        /// <param name="keepdims">Kept for API shape; a flat variance kept as size-1 everywhere.</param>
        /// <param name="ddof">Delta degrees of freedom; the divisor is <c>max(N − ddof, 0)</c>.</param>
        public static NDExpr Var(NDExpr x, bool keepdims, int ddof = 0) => new ReduceNode(NDExprReduceKind.Var, x, null, keepdims, ddof);

        /// <summary>One-pass fused <c>np.std(x, axis=None, keepdims=True, ddof=ddof)</c> — the whole-array standard deviation kept as shape (1,…,1).</summary>
        /// <param name="x">The expression to reduce.</param>
        /// <param name="keepdims">Kept for API shape; a flat std kept as size-1 everywhere.</param>
        /// <param name="ddof">Delta degrees of freedom; the divisor is <c>max(N − ddof, 0)</c>.</param>
        public static NDExpr Std(NDExpr x, bool keepdims, int ddof = 0) => new ReduceNode(NDExprReduceKind.Std, x, null, keepdims, ddof);

        /// <summary>One-pass fused <c>np.average(values, weights=weights, axis=None, keepdims=True)</c> — the whole-array weighted mean kept as shape (1,…,1).</summary>
        /// <param name="values">The value expression to average.</param>
        /// <param name="weights">The weight expression aligned with <paramref name="values"/>.</param>
        /// <param name="keepdims">Kept for API shape; a flat weighted average kept as size-1 everywhere.</param>
        /// <returns>The fused weighted-average node reducing the whole array, result kept as shape (1,…,1).</returns>
        public static NDExpr Average(NDExpr values, NDExpr weights, bool keepdims)
            => new WeightedAverageNode(values, weights, null, keepdims);

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

    public sealed partial class CastNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new CastNode(c, _target);
        }

        internal override bool ContainsReduce => _child.ContainsReduce;
        internal override NDArray FirstArray() => _child.FirstArray();
    }

    public sealed partial class RoundNode
    {
        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new RoundNode(c, _decimals);
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
        private readonly int _ddof;        // Var/Std delta degrees of freedom (divisor N-ddof); 0 for every other kind

        public ReduceNode(NDExprReduceKind kind, NDExpr child, int? axis = null, bool keepdims = false, int ddof = 0)
        {
            _kind = kind;
            _child = child ?? throw new ArgumentNullException(nameof(child));
            _axis = axis;
            _keepdims = keepdims;
            _ddof = ddof;
        }

        internal NDExprReduceKind Kind => _kind;
        internal NDExpr Child => _child;
        internal int? Axis => _axis;
        internal bool Keepdims => _keepdims;

        /// <summary>Var/Std delta degrees of freedom — the divisor is <c>max(N − Ddof, 0)</c>. 0 for all other kinds.</summary>
        internal int Ddof => _ddof;

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
            // ddof is folded into the legacy string cache key so two Var/Std trees that differ only in
            // ddof never share a compiled program (the structural cache guards this too — see
            // NDExpr.Structure.cs — but the string key must be independent as well).
            sb.Append("Reduce").Append(_kind);
            if (_ddof != 0) sb.Append("_dd").Append(_ddof);
            sb.Append('(');
            _child.AppendSignature(sb);
            sb.Append(')');
        }

        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            var c = _child.BindArrays(ctx);
            return ReferenceEquals(c, _child) ? this : new ReduceNode(_kind, c, _axis, _keepdims, _ddof);
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

                case NDExprReduceKind.ArgMax:
                case NDExprReduceKind.ArgMin:
                    // an INDEX, not a value — always int64 (NumPy's intp), regardless of the child dtype.
                    // Host-delegated to np.argmax / np.argmin over the materialized child.
                    return NPTypeCode.Int64;

                case NDExprReduceKind.Mean:
                case NDExprReduceKind.NanMean:
                    // nanmean shares mean's dtype tier: the float widths are preserved (float16 via a
                    // float32 intermediate), complex128 stays complex (its mean is complex), and every
                    // integer/bool/char widens to float64.
                    return child switch
                    {
                        NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or
                        NPTypeCode.Decimal or NPTypeCode.Complex => child,
                        _ => NPTypeCode.Double,
                    };

                case NDExprReduceKind.Var:
                case NDExprReduceKind.Std:
                    // var/std preserve the float widths (float16 computed in float32), but — like NumPy —
                    // a complex128 child yields a REAL float64 result (the variance of complex data is the
                    // mean squared |deviation|). Integer/bool/char widen to float64.
                    return child switch
                    {
                        NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal => child,
                        _ => NPTypeCode.Double, // Complex → real float64; int/bool/char → float64
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
            // Min/Max and the host-delegated M4-tail/M4c kinds (Ptp/NanMin/NanMax/ArgMax/ArgMin) never
            // run a widening summation — Min/Max fold in place and the delegated kinds bypass the fold
            // path entirely — so the accumulator dtype is simply the result dtype (no f16/f32 → f64
            // widening; the index kinds' result is int64 regardless). The M4c summation kinds
            // (NanMean/Var/Std) are host-computed in EvaluateStatReduce, which chooses its own compute
            // dtype per input, so their fold accType is likewise unused — return the result dtype.
            if (kind == NDExprReduceKind.Min || kind == NDExprReduceKind.Max
                || kind == NDExprReduceKind.Ptp || kind == NDExprReduceKind.NanMin || kind == NDExprReduceKind.NanMax
                || kind == NDExprReduceKind.ArgMax || kind == NDExprReduceKind.ArgMin
                || kind == NDExprReduceKind.NanMean || kind == NDExprReduceKind.Var || kind == NDExprReduceKind.Std)
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

    // =========================================================================
    // Node: WeightedAverageNode — root-only weighted average over TWO trees
    // (plan P2 M4c-average). np.average(values, weights) = Σ(values·weights) / Σ(weights).
    //
    // Unlike every other reduction it reduces over TWO operand trees, which the single-child
    // ReduceNode cannot carry — so it is its own root node, host-computed. The host materializes
    // BOTH children once (the M1/M2 route), casts each to the ONE result dtype (NumPy forces the
    // product and both sums to `result_dtype`, so an int·int product is a float64 product, never a
    // wrapping int64 one), forms the product, and reduces the product and the weights with the SAME
    // NumPy-exact pairwise sum np.average runs (`ExactSumArray`, not the drifting multi-accumulator
    // engine sum np.average itself uses — which is why the fused Average is bit-exact where the
    // library np.average is only allclose at large N), then divides. See
    // DefaultEngine.EvaluateWeightedAverage.
    //
    // dtype rule (NumPy _average, probed 2.4.2 — keyed on the VALUES dtype, NOT the combined type):
    //   values integer/bool → result_type(values, weights, float64)   (int average is ≥ float64)
    //   else                → result_type(values, weights)             (f2/f2 → f2 — rejected: no f16 kernel)
    // =========================================================================

    /// <summary>
    /// A root-only fused weighted average <c>Σ(values·weights) / Σ(weights)</c> over two elementwise
    /// sub-trees (<see cref="NDExpr.Average(NDExpr,NDExpr)"/>). Like <see cref="ReduceNode"/> it is
    /// host-driven (no elementwise emit) and must be the root of the expression.
    /// </summary>
    public sealed partial class WeightedAverageNode : NDExpr
    {
        private readonly NDExpr _values;
        private readonly NDExpr _weights;
        private readonly int? _axis;       // null = flat (reduce-all); else reduce this axis
        private readonly bool _keepdims;

        /// <summary>Build a weighted-average node over <paramref name="values"/> and <paramref name="weights"/>.</summary>
        /// <param name="values">The value expression (numerator's left factor).</param>
        /// <param name="weights">The weight expression (numerator's right factor and the denominator).</param>
        /// <param name="axis">The reduction axis, or null for a flat (reduce-all) average.</param>
        /// <param name="keepdims">When true, the reduced axis (or every axis, for a flat average) is kept as size 1.</param>
        /// <exception cref="ArgumentNullException"><paramref name="values"/> or <paramref name="weights"/> is null.</exception>
        public WeightedAverageNode(NDExpr values, NDExpr weights, int? axis = null, bool keepdims = false)
        {
            _values = values ?? throw new ArgumentNullException(nameof(values));
            _weights = weights ?? throw new ArgumentNullException(nameof(weights));
            _axis = axis;
            _keepdims = keepdims;
        }

        /// <summary>The value sub-tree.</summary>
        internal NDExpr Values => _values;

        /// <summary>The weight sub-tree.</summary>
        internal NDExpr Weights => _weights;

        /// <summary>The reduction axis, or null for a flat average.</summary>
        internal int? Axis => _axis;

        /// <summary>True when the reduced axis is kept as a size-1 dimension.</summary>
        internal bool Keepdims => _keepdims;

        /// <summary>Never SIMD — a reduction has no elementwise vector body; the host drives it.</summary>
        public override bool SupportsSimd => false;

        /// <summary>A weighted average is host-driven as the tree root; it emits no elementwise scalar value.</summary>
        /// <param name="il">Unused.</param>
        /// <param name="ctx">Unused.</param>
        /// <exception cref="InvalidOperationException">Always — the node is driven by np.evaluate, never emitted.</exception>
        public override void EmitScalar(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                "np.average is driven by np.evaluate as the tree root — it cannot be emitted as an elementwise value.");

        /// <summary>A weighted average has no vector path — it is driven by np.evaluate.</summary>
        /// <param name="il">Unused.</param>
        /// <param name="ctx">Unused.</param>
        /// <exception cref="InvalidOperationException">Always.</exception>
        public override void EmitVector(ILGenerator il, NDExprCompileContext ctx)
            => throw new InvalidOperationException(
                "Weighted-average nodes have no vector path — they are driven by np.evaluate.");

        /// <summary>Fold the node's identity into a structural signature (axis / keepdims + both children).</summary>
        /// <param name="sb">The signature builder.</param>
        public override void AppendSignature(StringBuilder sb)
        {
            // Axis / keepdims are read by the host per evaluation (flat vs axis path, output shape), so
            // they are part of the identity even though the two child kernels are axis-independent.
            sb.Append("WAvg");
            if (_axis is int ax) sb.Append("_ax").Append(ax);
            if (_keepdims) sb.Append("_kd");
            sb.Append('(');
            _values.AppendSignature(sb);
            sb.Append(',');
            _weights.AppendSignature(sb);
            sb.Append(')');
        }

        internal override NDExpr BindArrays(NDExprBindContext ctx)
        {
            // Values FIRST, then weights — the order the structural hash and the two sub-programs
            // (AvgValuesProgram / AvgWeightsProgram) rely on; changing it desynchronizes operand indices.
            var v = _values.BindArrays(ctx);
            var w = _weights.BindArrays(ctx);
            return ReferenceEquals(v, _values) && ReferenceEquals(w, _weights)
                ? this
                : new WeightedAverageNode(v, w, _axis, _keepdims);
        }

        internal override bool ContainsReduce => true;

        // np.evaluate dispatches on the first embedded array's engine; either child may hold it.
        internal override NDArray FirstArray() => _values.FirstArray() ?? _weights.FirstArray();

        internal override NDExprTypeInfo InferType(
            NPTypeCode[] inputTypes, Dictionary<NDExpr, NPTypeCode> nodeTypes)
        {
            // Resolve each child to a strong dtype (a pure-constant child adopts its NEP50 default), the
            // dtypes the two sub-programs will independently resolve to — so the result dtype computed
            // here matches what the materialized children will be.
            var vt = _values.InferType(inputTypes, nodeTypes);
            var vType = ResolveChild(_values, vt, vt.IsWeak ? vt.DefaultCode : vt.Code, nodeTypes);
            var wt = _weights.InferType(inputTypes, nodeTypes);
            var wType = ResolveChild(_weights, wt, wt.IsWeak ? wt.DefaultCode : wt.Code, nodeTypes);

            var result = ResolveAverageResultType(vType, wType);
            nodeTypes[this] = result;
            return NDExprTypeInfo.Strong(result);
        }

        /// <summary>
        /// NumPy's <c>np.average</c> result dtype (probed 2.4.2), keyed on the VALUES dtype: an
        /// integer/bool values tree lifts the pair to <c>result_type(values, weights, float64)</c> (so an
        /// integer average is always at least float64 and the product never wraps), otherwise the plain
        /// <c>result_type(values, weights)</c> (which can be float16 when both are float16 — the host then
        /// rejects it). Always a float/complex/decimal dtype; the host serves Single/Double/Complex.
        /// </summary>
        /// <param name="values">The values sub-tree's resolved dtype.</param>
        /// <param name="weights">The weights sub-tree's resolved dtype.</param>
        /// <returns>The weighted average's result dtype.</returns>
        internal static NPTypeCode ResolveAverageResultType(NPTypeCode values, NPTypeCode weights)
        {
            var common = NDExprTypeRules.PromoteStrong(values, weights);
            bool valuesIntBool = values == NPTypeCode.Boolean || NDExprTypeRules.IsIntegerKind(values);
            return valuesIntBool ? NDExprTypeRules.PromoteStrong(common, NPTypeCode.Double) : common;
        }
    }
}
