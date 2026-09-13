using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Vector.cs — the "vector v2" plan and lane-mask helpers (ndexpr-evaluate.md Phase 1)
// =============================================================================
//
// The first vector contract vectorized a tree only when every operand AND every node shared
// one dtype, so any comparison, where, predicate, logical node or bool operand made the whole
// tree scalar — the largest measured loss of the plan (a>0.5 at 0.16x NumPy, maximum 0.46x).
//
// v2 keeps ONE compute lane dtype W per kernel and lets every Boolean-typed node ride as a
// LANE MASK of W (all-ones / zero per lane) instead of a byte value:
//
//   * a node typed W emits Vector<W>; a node typed Boolean emits a Vector<W> mask;
//   * a mask meets arithmetic through `mask & One` (exact 1/0 lanes — never a multiply, which
//     would turn inf*0 into NaN); a value meets a Boolean slot through `~Equals(x, 0)`;
//   * comparisons are the comparison kernel's own vector compare (NaN-false, ~Equals for !=);
//     where is ConditionalSelect; min/max is the NaN-propagating clamp; predicates are
//     Equals/Abs/LessThan against ±inf;
//   * bool OPERANDS are expanded to masks and a bool OUTPUT packed to bytes by the fused
//     shell (DirectILKernelGenerator.InnerLoop.Fused.cs);
//   * when EVERY operand is bool ("byte mode", W == Boolean) nodes carry canonical 0/1 bytes
//     through the engine's normalized logical ops instead — the pre-existing bool contract.
//
// The plan (NDExprVectorPlan.TryPlan) decides W and asks every node whether it can emit at
// W; the scalar body is unchanged, so the vector path can only differ from it by a bug — which
// the metamorphic sweep (NDEvaluateVectorTests) and the evaluate.jsonl tier catch.
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>Lane-mask / vector emit helpers shared by the node emitters (v2 contract).</summary>
    internal static class NDExprVec
    {
        private static int Bits => DirectILKernelGenerator.VectorBits;

        private static Type Clr(NPTypeCode t) => DirectILKernelGenerator.GetClrType(t);

        /// <summary>[mask(V&lt;W&gt;)] → [value(V&lt;W&gt;)]: <c>mask &amp; One</c> — exact 1/0 lanes (bool→number).</summary>
        internal static void EmitMaskToNumeric(ILGenerator il, NPTypeCode w)
        {
            var clr = Clr(w);
            il.EmitCall(OpCodes.Call, VectorMethodCache.One(Bits, clr), null);
            EmitAnd(il, clr);
        }

        /// <summary>[value(V&lt;W&gt;)] → [mask]: <c>~Equals(x, 0)</c> — NumPy truthiness (NaN is true, ±0 false).</summary>
        internal static void EmitValueToMask(ILGenerator il, NPTypeCode w)
        {
            EmitIsZeroMask(il, w);
            il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(Bits, Clr(w)), null);
        }

        /// <summary>[value(V&lt;W&gt;)] → [mask]: <c>Equals(x, 0)</c> — logical_not (NaN → false).</summary>
        internal static void EmitIsZeroMask(ILGenerator il, NPTypeCode w)
        {
            var clr = Clr(w);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(Bits, clr), null);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(Bits, clr), null);
        }

        internal static void EmitAllBitsSet(ILGenerator il, NPTypeCode w)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.AllBitsSet(Bits, Clr(w)), null);

        internal static void EmitZero(ILGenerator il, NPTypeCode w)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(Bits, Clr(w)), null);

        internal static void EmitNot(ILGenerator il, NPTypeCode w)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(Bits, Clr(w)), null);

        internal static void EmitAnd(ILGenerator il, Type clr)
            => il.EmitCall(OpCodes.Call,
                VectorMethodCache.BinaryX86(Bits, "BitwiseAnd", clr)
                ?? VectorMethodCache.Generic(Bits, "BitwiseAnd", clr, paramCount: 2), null);

        internal static void EmitOr(ILGenerator il, Type clr)
            => il.EmitCall(OpCodes.Call,
                VectorMethodCache.BinaryX86(Bits, "BitwiseOr", clr)
                ?? VectorMethodCache.Generic(Bits, "BitwiseOr", clr, paramCount: 2), null);

        internal static void EmitXor(ILGenerator il, Type clr)
            => il.EmitCall(OpCodes.Call,
                VectorMethodCache.BinaryX86(Bits, "Xor", clr)
                ?? VectorMethodCache.Generic(Bits, "Xor", clr, paramCount: 2), null);

        /// <summary>[a, b] → [a &amp; ~b].</summary>
        internal static void EmitAndNot(ILGenerator il, Type clr)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(Bits, "AndNot", clr, paramCount: 2), null);

        /// <summary>[x(V&lt;W&gt;)] → [|x|] (generic Vector.Abs; identity for unsigned lanes).</summary>
        internal static void EmitAbs(ILGenerator il, NPTypeCode w)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(Bits, "Abs", Clr(w), paramCount: 1), null);

        /// <summary>[] → [V&lt;W&gt;(+inf)] for a float W.</summary>
        internal static void EmitPositiveInfinity(ILGenerator il, NPTypeCode w)
        {
            if (w == NPTypeCode.Single) il.Emit(OpCodes.Ldc_R4, float.PositiveInfinity);
            else il.Emit(OpCodes.Ldc_R8, double.PositiveInfinity);
            DirectILKernelGenerator.EmitVectorCreate(il, w);
        }

        /// <summary>Byte mode: [V&lt;byte&gt; 0/1 (or any nonzero) bytes] → [byte lane mask] via <c>&gt; 0</c>.</summary>
        internal static void EmitBytesToMask(ILGenerator il)
        {
            il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(Bits, typeof(byte)), null);
            il.EmitCall(OpCodes.Call, VectorMethodCache.GreaterThan(Bits, typeof(byte)), null);
        }

        /// <summary>[mask, a, b] → [mask ? a : b] at W's lane type (byte in byte mode).</summary>
        internal static void EmitSelect(ILGenerator il, NPTypeCode w)
            => il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(Bits, DirectILKernelGenerator.GetSimdLaneType(w)), null);

        internal static bool IsFloatLane(NPTypeCode w) => w == NPTypeCode.Single || w == NPTypeCode.Double;
    }

    /// <summary>
    /// Decides whether a NumPy-typed tree vectorizes under the v2 contract and at which lane
    /// dtype: the unique non-bool operand dtype (SIMD-capable), or Boolean when every operand is
    /// bool; every node must type to that lane or to Boolean, and every node must have a vector
    /// emit at that lane.
    /// </summary>
    internal static class NDExprVectorPlan
    {
        internal static bool TryPlan(
            NDExpr root, NPTypeCode[] inputTypes, IReadOnlyDictionary<NDExpr, NPTypeCode> nodeTypes,
            out NPTypeCode lane)
        {
            lane = NPTypeCode.Empty;
            if (DirectILKernelGenerator.VectorBits == 0)
                return false;

            bool boolInput = false;
            foreach (var t in inputTypes)
            {
                if (t == NPTypeCode.Boolean) { boolInput = true; continue; }
                if (lane == NPTypeCode.Empty) lane = t;
                else if (lane != t) return false;
            }

            if (lane == NPTypeCode.Empty)
                lane = NPTypeCode.Boolean;                       // byte mode
            else if (!DirectILKernelGenerator.CanUseSimd(lane))
                return false;

            foreach (var kv in nodeTypes)
                if (kv.Value != lane && kv.Value != NPTypeCode.Boolean)
                    return false;

            // Bool INPUT masks / a bool OUTPUT pack go through the 128/256-bit shell helpers.
            bool boolIo = boolInput || (nodeTypes.TryGetValue(root, out var rootType) && rootType == NPTypeCode.Boolean);
            if (lane != NPTypeCode.Boolean && boolIo && !DirectILKernelGenerator.FusedBoolLanesAvailable)
                return false;

            return root.CanEmitVectorV2(lane, nodeTypes);
        }
    }

    public abstract partial class NDExpr
    {
        /// <summary>
        /// Test / diagnostics hook: when set on the current thread, <see cref="CompileNumPy"/> compiles
        /// scalar-only kernels (distinct cache key), so a vector-path result can be compared byte for
        /// byte against the scalar body it must reproduce.
        /// </summary>
        [ThreadStatic] internal static bool ForceScalar;

        /// <summary>
        /// Whether this node (and its subtree) has a v2 vector emit when the kernel's lane dtype is
        /// <paramref name="lane"/> and nodes carry the dtypes in <paramref name="types"/> (each one is
        /// <paramref name="lane"/> or Boolean — the plan checks that before asking).
        /// </summary>
        internal abstract bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types);

        /// <summary>
        /// Emit <paramref name="child"/>'s vector converted to <paramref name="target"/> (lane or
        /// Boolean): a mask meeting a numeric slot becomes exact 1/0 lanes, a value meeting a Boolean
        /// slot becomes its truthiness mask. Byte mode carries bool bytes only — no conversion.
        /// </summary>
        private protected static void EmitVectorChildAs(ILGenerator il, NDExprCompileContext ctx, NDExpr child, NPTypeCode target)
        {
            child.EmitVector(il, ctx);
            var from = ctx.TypeOf(child);
            if (from == target)
                return;
            var lane = ctx.VectorLaneType;
            if (lane == NPTypeCode.Boolean)
                return;
            if (from == NPTypeCode.Boolean)
                NDExprVec.EmitMaskToNumeric(il, lane);
            else
                NDExprVec.EmitValueToMask(il, lane);
        }
    }

    public sealed partial class InputNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;
    }

    public sealed partial class ConstNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;
    }

    public sealed partial class ArrayNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class BinaryNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            if (!_left.CanEmitVectorV2(lane, types) || !_right.CanEmitVectorV2(lane, types))
                return false;
            if (types[this] == NPTypeCode.Boolean)
                return _op == BinaryOp.Add || _op == BinaryOp.Multiply ||
                       _op == BinaryOp.BitwiseAnd || _op == BinaryOp.BitwiseOr || _op == BinaryOp.BitwiseXor;
            return IsSimdOp(_op);
        }
    }

    public sealed partial class UnaryNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            if (!_child.CanEmitVectorV2(lane, types))
                return false;
            var my = types[this];
            var ct = types[_child];
            if (my == NPTypeCode.Boolean)
            {
                if (_op == UnaryOp.LogicalNot || IsPredicateResult(_op))
                    return true;
                // invert / abs / floor / ceil / trunc on a bool are the identity (or logical not) on masks
                return ct == NPTypeCode.Boolean && (_op == UnaryOp.BitwiseNot || _op == UnaryOp.Abs || IsRoundingOp(_op));
            }

            // my == lane (W != Boolean): floor/ceil/round/trunc are the identity on integer lanes
            if (IsRoundingOp(_op) && IsIntegerKind(lane))
                return true;
            return IsSimdUnaryAt(_op, lane);
        }
    }

    public sealed partial class ComparisonNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            if (!_left.CanEmitVectorV2(lane, types) || !_right.CanEmitVectorV2(lane, types))
                return false;
            var cmp = NDExprTypeRules.ComparisonType(types[_left], types[_right]);
            return cmp == lane || cmp == NPTypeCode.Boolean;
        }
    }

    public sealed partial class MinMaxNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
            => _left.CanEmitVectorV2(lane, types) && _right.CanEmitVectorV2(lane, types);
    }

    public sealed partial class WhereNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
            => _cond.CanEmitVectorV2(lane, types) && _a.CanEmitVectorV2(lane, types) && _b.CanEmitVectorV2(lane, types);
    }

    public sealed partial class CallNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class ReduceNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }
}
