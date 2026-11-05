using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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
//
// SAME-WIDTH trees (P5.3): when the non-bool dtypes differ but share ONE element size — an int64
// comparison choosing between float64 branches, `np.where(f >= 0, price, 0)` — every node computes
// at its OWN dtype in a full-width vector, and the masks cross dtypes by reinterpretation: an
// all-ones / zero lane is the same bits as a Vector<long> or a Vector<double> mask, so the
// crossing costs nothing (EmitReinterpret). Masks travel as Vector<W> (W = the root dtype, or a
// non-bool input's when the root is bool), and only VALUE edges between two different dtypes are
// refused (int64 → float64 rounds and has no AVX2 vector form): such a tree stays scalar. Before
// this plan the tree ran the scalar body, whose `where` BRANCHES per element — 0.46x of the
// unfused np.where on a random mask. Every same-width rule reduces to the uniform rule when the
// node's dtype IS the lane, so the uniform plan's kernels are emitted exactly as before.
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

        /// <summary>
        /// [V&lt;from&gt;] → [V&lt;to&gt;]: the same bits under another lane type — how a lane MASK crosses between two
        /// dtypes of one element size in a same-width tree (an int64 comparison's mask selecting float64 lanes). An
        /// all-ones / zero lane means the same under either type, and the JIT emits no instruction for the cast.
        /// </summary>
        /// <param name="il">The IL generator; the vector to reinterpret is on the stack.</param>
        /// <param name="from">The dtype the vector is typed at (a SIMD dtype).</param>
        /// <param name="to">The dtype to type it at (a SIMD dtype of the same size).</param>
        /// <remarks>
        /// Emits nothing when the dtypes are equal — the uniform plan's every case, which is why its kernels are
        /// unchanged by the same-width plan. Only for masks: reinterpreting a VALUE is not a conversion (the bits of
        /// int64 3 are not float64 3.0); the plan refuses value edges between two dtypes.
        /// </remarks>
        /// <exception cref="InvalidOperationException">The two dtypes differ in size (a plan bug: the same-width plan admits one size).</exception>
        internal static void EmitReinterpret(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            if (from == to)
                return;
            if (DirectILKernelGenerator.GetTypeSize(from) != DirectILKernelGenerator.GetTypeSize(to))
                throw new InvalidOperationException($"cannot reinterpret a {from} lane mask as {to}: the sizes differ (a same-width plan bug).");
            il.EmitCall(OpCodes.Call, VectorMethodCache.As(Bits,
                DirectILKernelGenerator.GetSimdLaneType(from), DirectILKernelGenerator.GetSimdLaneType(to)), null);
        }

        /// <summary>
        /// Whether a child of dtype <paramref name="child"/> may feed a slot of dtype <paramref name="slot"/> in a v2
        /// vector body: the same dtype (no conversion), a bool MASK meeting a numeric slot (<c>mask &amp; One</c> at the
        /// slot's dtype), or a value meeting a Boolean slot (its truthiness mask at its own dtype). A value edge between
        /// two different non-bool dtypes is refused — the uniform plan never has one, the same-width plan must not
        /// reinterpret one.
        /// </summary>
        /// <param name="child">The child node's dtype.</param>
        /// <param name="slot">The dtype the consuming node needs the child at.</param>
        /// <returns>True when <c>EmitVectorChildAs</c> can deliver the child at the slot's dtype.</returns>
        internal static bool IsVectorEdge(NPTypeCode child, NPTypeCode slot)
            => child == slot || child == NPTypeCode.Boolean || slot == NPTypeCode.Boolean;

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

        // ---- P5.2 mixed-width exact widening edges (ndexpr-evaluate.md Phase 5.2) --------------
        //
        // In a mixed-width kernel every admitted non-lane dtype is exactly HALF the lane's size
        // (the plan's container rule), so every edge is one Vector128 → Vector256 widen through the
        // AVX/AVX2 ConvertToVector256* family. Each entry is bit-identical to the scalar body's
        // EmitConvertTo for the same (from, to): integer widens sign/zero-extend exactly; int→float
        // and float→double conversions are IEEE correctly-rounded in BOTH forms (and value-exact
        // for every admitted pair except none — i4→f8, u4→f8, f4→f8, i2/u2→f4 all represent
        // exactly); uint32→double takes the sign-bias trick because AVX2 has no vcvtudq2pd (the
        // biased int convert and the +2^31 are both exact, so the composition equals conv.r.un).
        // int64/uint64→double is deliberately ABSENT: it rounds and AVX2 has no vector form, so a
        // same-size i8/u8→f8 edge keeps its whole tree scalar.

        private static MethodInfo X86Cvt(Type owner, string name, Type arg)
            => owner.GetMethod(name, new[] { arg })
               ?? throw new MissingMethodException(owner.FullName, $"{name}({arg.Name})");

        private static readonly MethodInfo s_cvtI16FromSByte = X86Cvt(typeof(Avx2), "ConvertToVector256Int16", typeof(Vector128<sbyte>));
        private static readonly MethodInfo s_cvtI16FromByte = X86Cvt(typeof(Avx2), "ConvertToVector256Int16", typeof(Vector128<byte>));
        private static readonly MethodInfo s_cvtI32FromInt16 = X86Cvt(typeof(Avx2), "ConvertToVector256Int32", typeof(Vector128<short>));
        private static readonly MethodInfo s_cvtI32FromUInt16 = X86Cvt(typeof(Avx2), "ConvertToVector256Int32", typeof(Vector128<ushort>));
        private static readonly MethodInfo s_cvtI64FromInt32 = X86Cvt(typeof(Avx2), "ConvertToVector256Int64", typeof(Vector128<int>));
        private static readonly MethodInfo s_cvtI64FromUInt32 = X86Cvt(typeof(Avx2), "ConvertToVector256Int64", typeof(Vector128<uint>));
        private static readonly MethodInfo s_cvtF64FromInt32 = X86Cvt(typeof(Avx), "ConvertToVector256Double", typeof(Vector128<int>));
        private static readonly MethodInfo s_cvtF64FromSingle = X86Cvt(typeof(Avx), "ConvertToVector256Double", typeof(Vector128<float>));
        private static readonly MethodInfo s_cvtF32FromInt32V256 = X86Cvt(typeof(Avx), "ConvertToVector256Single", typeof(Vector256<int>));

        /// <summary>
        /// Whether the mixed-width plan admits a child of dtype <paramref name="from"/> feeding a
        /// node slot of dtype <paramref name="to"/> — i.e. an EXACT, vectorizable half→full-lane
        /// widening exists (see the map above). Same-dtype edges are the caller's no-op case, not
        /// listed here; an unlisted pair (notably i8/u8→f8, which rounds) declines the whole tree
        /// to the scalar path.
        /// </summary>
        /// <param name="from">The child's dtype.</param>
        /// <param name="to">The consuming node's dtype.</param>
        /// <returns>True when <see cref="EmitWidenEdge"/> can emit this edge.</returns>
        internal static bool HasWidenEdge(NPTypeCode from, NPTypeCode to) => (from, to) switch
        {
            (NPTypeCode.SByte, NPTypeCode.Int16) => true,
            (NPTypeCode.Byte, NPTypeCode.Int16) => true,
            (NPTypeCode.Byte, NPTypeCode.UInt16) => true,
            (NPTypeCode.Int16, NPTypeCode.Int32) => true,
            (NPTypeCode.UInt16, NPTypeCode.Int32) => true,
            (NPTypeCode.UInt16, NPTypeCode.UInt32) => true,
            (NPTypeCode.Int16, NPTypeCode.Single) => true,
            (NPTypeCode.UInt16, NPTypeCode.Single) => true,
            (NPTypeCode.Int32, NPTypeCode.Int64) => true,
            (NPTypeCode.UInt32, NPTypeCode.Int64) => true,
            (NPTypeCode.UInt32, NPTypeCode.UInt64) => true,
            (NPTypeCode.Int32, NPTypeCode.Double) => true,
            (NPTypeCode.UInt32, NPTypeCode.Double) => true,
            (NPTypeCode.Single, NPTypeCode.Double) => true,
            _ => false,
        };

        /// <summary>
        /// Mixed-width edge conversion: stack [Vector128&lt;from&gt;] → [Vector256&lt;to&gt;], the exact
        /// widening of <see cref="HasWidenEdge"/>. Bit-identical to the scalar edge's
        /// <c>EmitConvertTo(from, to)</c> for every lane — the vector==scalar contract every fused
        /// path must keep (the metamorphic sweep compares the two byte for byte).
        /// </summary>
        /// <param name="il">The IL generator.</param>
        /// <param name="from">The child's dtype (a half-lane dtype; its vector is 128-bit).</param>
        /// <param name="to">The consuming node's dtype (the pair must satisfy <see cref="HasWidenEdge"/>).</param>
        /// <exception cref="NotSupportedException">The pair has no exact vector widening (plan bug — TryPlanMixed admits only mapped edges).</exception>
        internal static void EmitWidenEdge(ILGenerator il, NPTypeCode from, NPTypeCode to)
        {
            switch (from, to)
            {
                case (NPTypeCode.SByte, NPTypeCode.Int16):
                    il.EmitCall(OpCodes.Call, s_cvtI16FromSByte, null);
                    return;
                case (NPTypeCode.Byte, NPTypeCode.Int16):
                    il.EmitCall(OpCodes.Call, s_cvtI16FromByte, null);
                    return;
                case (NPTypeCode.Byte, NPTypeCode.UInt16):
                    // Zero-extend to Vector256<short>, then reinterpret — the bits are already the
                    // unsigned values (byte ≤ 255 never sets the sign bit).
                    il.EmitCall(OpCodes.Call, s_cvtI16FromByte, null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(short), typeof(ushort)), null);
                    return;
                case (NPTypeCode.Int16, NPTypeCode.Int32):
                    il.EmitCall(OpCodes.Call, s_cvtI32FromInt16, null);
                    return;
                case (NPTypeCode.UInt16, NPTypeCode.Int32):
                    il.EmitCall(OpCodes.Call, s_cvtI32FromUInt16, null);
                    return;
                case (NPTypeCode.UInt16, NPTypeCode.UInt32):
                    il.EmitCall(OpCodes.Call, s_cvtI32FromUInt16, null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(int), typeof(uint)), null);
                    return;
                case (NPTypeCode.Int16, NPTypeCode.Single):
                    // Sign-extend to int32 lanes, then cvtdq2ps — exact (every int16 is a float32).
                    il.EmitCall(OpCodes.Call, s_cvtI32FromInt16, null);
                    il.EmitCall(OpCodes.Call, s_cvtF32FromInt32V256, null);
                    return;
                case (NPTypeCode.UInt16, NPTypeCode.Single):
                    il.EmitCall(OpCodes.Call, s_cvtI32FromUInt16, null);
                    il.EmitCall(OpCodes.Call, s_cvtF32FromInt32V256, null);
                    return;
                case (NPTypeCode.Int32, NPTypeCode.Int64):
                    il.EmitCall(OpCodes.Call, s_cvtI64FromInt32, null);
                    return;
                case (NPTypeCode.UInt32, NPTypeCode.Int64):
                    il.EmitCall(OpCodes.Call, s_cvtI64FromUInt32, null);
                    return;
                case (NPTypeCode.UInt32, NPTypeCode.UInt64):
                    il.EmitCall(OpCodes.Call, s_cvtI64FromUInt32, null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(long), typeof(ulong)), null);
                    return;
                case (NPTypeCode.Int32, NPTypeCode.Double):
                    il.EmitCall(OpCodes.Call, s_cvtF64FromInt32, null);
                    return;
                case (NPTypeCode.Single, NPTypeCode.Double):
                    il.EmitCall(OpCodes.Call, s_cvtF64FromSingle, null);
                    return;
                case (NPTypeCode.UInt32, NPTypeCode.Double):
                {
                    // AVX2 has no unsigned u32→f8 convert (vcvtudq2pd is AVX-512). Bias through the
                    // signed convert instead: (int)(u ^ 0x80000000) == u - 2^31 for every uint32,
                    // that biased value converts EXACTLY (|v| < 2^31 ≤ 2^53), and adding 2^31 back
                    // in double is exact too (result ≤ 2^32) — so the composition equals the scalar
                    // conv.r.un bit for bit on every lane.
                    il.Emit(OpCodes.Ldc_I4, unchecked((int)0x80000000));
                    il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(128, typeof(uint)), null);
                    // NDExprVec.EmitXor is pinned to the module-wide width; this xor is 128-bit.
                    il.EmitCall(OpCodes.Call,
                        VectorMethodCache.BinaryX86(128, "Xor", typeof(uint))
                        ?? VectorMethodCache.Generic(128, "Xor", typeof(uint), paramCount: 2), null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(128, typeof(uint), typeof(int)), null);
                    il.EmitCall(OpCodes.Call, s_cvtF64FromInt32, null);
                    il.Emit(OpCodes.Ldc_R8, 2147483648.0);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(256, typeof(double)), null);
                    il.EmitCall(OpCodes.Call,
                        VectorMethodCache.BinaryX86(256, "Add", typeof(double))
                        ?? VectorMethodCache.Operator(256, typeof(double), "op_Addition"), null);
                    return;
                }
                default:
                    throw new NotSupportedException($"no exact mixed-width vector widening for {from} → {to}");
            }
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
    /// emit at that lane. When the tree is NOT uniform — two distinct non-bool input dtypes, or
    /// nodes promoting past the input dtype (<c>i4 + 2.5</c>) — the P5.2 MIXED-WIDTH plan is tried
    /// instead (<see cref="TryPlanMixed"/>): lane = the root dtype, half-lane nodes ride partial
    /// vectors, edges widen exactly; then the P5.3 SAME-WIDTH plan (<see cref="TryPlanSameWidth"/>):
    /// every non-bool dtype of one size, masks crossing dtypes by reinterpretation.
    /// </summary>
    internal static class NDExprVectorPlan
    {
        /// <summary>
        /// Plan the v2 vector kernel for <paramref name="root"/>, or report that it must stay scalar.
        /// </summary>
        /// <param name="root">The typed expression tree.</param>
        /// <param name="inputTypes">Every input's dtype, hoisted parameters included, in input order.</param>
        /// <param name="nodeTypes">The per-node NumPy result dtypes from the typing pass.</param>
        /// <param name="lane">Receives the compute lane dtype W (Boolean = byte mode; the ROOT dtype for a mixed-width plan); <see cref="NPTypeCode.Empty"/> when planning fails.</param>
        /// <param name="mixedWidth">
        /// Receives true when the tree took the P5.2 mixed-width plan (<see cref="TryPlanMixed"/>):
        /// the caller must then type each operand's vector local at its own dtype's container and
        /// hand the shell <c>mixedWidth</c>. False for a uniform plan and whenever planning fails.
        /// </param>
        /// <param name="sameWidth">
        /// Receives true when the tree took the P5.3 same-width plan (<see cref="TryPlanSameWidth"/>):
        /// the caller must then type each non-bool operand's vector local (and each non-bool
        /// parameter's broadcast) at its OWN dtype — a full-width vector, since every such dtype has
        /// the lane's size — and hand the shell <c>sameWidth</c>. False for the other plans and
        /// whenever planning fails.
        /// </param>
        /// <param name="isParam">
        /// Per input, whether it is a hoisted 0-d parameter (null = none). A bool PARAMETER becomes a
        /// constant lane mask via portable <c>Zero</c>/<c>AllBitsSet</c> in the prologue, so unlike a
        /// STREAMED bool operand it never needs the x86 byte→lane expansion.
        /// </param>
        /// <returns>True when the tree vectorizes on THIS host; false sends it to the scalar shell.</returns>
        internal static bool TryPlan(
            NDExpr root, NPTypeCode[] inputTypes, IReadOnlyDictionary<NDExpr, NPTypeCode> nodeTypes,
            out NPTypeCode lane, out bool mixedWidth, out bool sameWidth, bool[]? isParam = null)
        {
            lane = NPTypeCode.Empty;
            mixedWidth = false;
            sameWidth = false;
            if (DirectILKernelGenerator.VectorBits == 0)
                return false;

            // One pass over the inputs answers three questions: is there any bool input, is any
            // bool input STREAMED (a hoisted 0-d bool parameter is a portable constant mask and
            // needs no host ISA — see isParam), and do the non-bool inputs share ONE dtype (else
            // only the mixed-width plan below can vectorize the tree).
            bool boolInput = false, boolStreamedInput = false, uniformInputs = true;
            for (int i = 0; i < inputTypes.Length; i++)
            {
                var t = inputTypes[i];
                if (t == NPTypeCode.Boolean)
                {
                    boolInput = true;
                    if (isParam is null || i >= isParam.Length || !isParam[i])
                        boolStreamedInput = true;
                    continue;
                }
                if (lane == NPTypeCode.Empty) lane = t;
                else if (lane != t) uniformInputs = false;
            }

            if (uniformInputs)
            {
                if (lane == NPTypeCode.Empty)
                    lane = NPTypeCode.Boolean;                       // byte mode

                // f16 rides a widen-compute-narrow Vector256<ushort> lane (Phase 5.1): only PURE
                // arithmetic vectorizes, on a 256-bit AVX2 host. Any comparison / where / min-max /
                // transcendental / unary Half node returns false from CanEmitVectorV2 below, which
                // (ANDed up through the root) keeps such a tree scalar — as does a non-256-bit host.
                bool laneOk = lane == NPTypeCode.Boolean
                    || (lane == NPTypeCode.Half
                        ? DirectILKernelGenerator.FusedHalfArithAvailable
                        : DirectILKernelGenerator.CanUseSimd(lane));

                if (laneOk)
                {
                    bool nodesUniform = true;
                    foreach (var kv in nodeTypes)
                        if (kv.Value != lane && kv.Value != NPTypeCode.Boolean) { nodesUniform = false; break; }

                    if (nodesUniform)
                    {
                        // Bool INPUT masks / a bool OUTPUT pack go through the 128/256-bit shell helpers.
                        bool boolIo = boolInput || (nodeTypes.TryGetValue(root, out var rootType) && rootType == NPTypeCode.Boolean);
                        if (lane != NPTypeCode.Boolean && boolIo && !DirectILKernelGenerator.FusedBoolLanesAvailable)
                            return false;

                        // A STREAMED bool operand is widened to W-lane masks by an x86 sign-extend when W is
                        // 2/4/8 bytes wide. That ISA is absent on ARM64, where the emitted kernel would throw
                        // PlatformNotSupportedException at run time, so plan scalar there. Same condition as
                        // the shell's own re-check (DirectILKernelGenerator.FusedSimdViable). No fall-through
                        // to the mixed plan: TryPlanMixed declines every bool input anyway.
                        if (lane != NPTypeCode.Boolean && boolStreamedInput && !DirectILKernelGenerator.FusedBoolInputMasksAvailable(lane))
                            return false;

                        return root.CanEmitVectorV2(lane, nodeTypes);
                    }
                    // Nodes promote past the uniform input dtype (a weak literal lifted the tree,
                    // e.g. i4 + 2.5 → f8): the mixed plan below can still vectorize it.
                }
            }

            // The two non-uniform plans exclude each other (the mixed one admits no Boolean node and
            // only half→full widenings, the same-width one only one element size and no value edge
            // between dtypes), so their order decides nothing; mixed goes first as it always did.
            if (TryPlanMixed(root, inputTypes, nodeTypes, out lane, out mixedWidth))
                return true;
            return TryPlanSameWidth(root, inputTypes, nodeTypes, isParam, out lane, out sameWidth);
        }

        /// <summary>
        /// The P5.3 SAME-WIDTH plan: every non-bool dtype of the kernel — inputs, parameters and
        /// nodes — is a SIMD dtype of ONE element size, but not one dtype (an int64 comparison over
        /// float64 branches; a float64 test of an int64 count). Each node then computes at its OWN
        /// dtype in a full-width <c>Vector&lt;T&gt;</c>, Boolean nodes ride as <c>Vector&lt;W&gt;</c>
        /// lane masks, and a mask crosses into another dtype by reinterpretation
        /// (<see cref="NDExprVec.EmitReinterpret"/>: all-ones / zero lanes are the same bits at
        /// either type). The per-node capability check (<see cref="NDExpr.CanEmitVectorV2"/>) also
        /// refuses every VALUE edge between two dtypes (<see cref="NDExprVec.IsVectorEdge"/>), so a
        /// tree that would need int64 → float64 (a rounding conversion AVX2 cannot vectorize) stays
        /// scalar, as before.
        /// </summary>
        /// <param name="root">The bound tree's root.</param>
        /// <param name="inputTypes">Every input's dtype (parameters included).</param>
        /// <param name="nodeTypes">The typing pass's node→dtype table.</param>
        /// <param name="isParam">Per input, whether it is a hoisted 0-d parameter (null = none): a bool
        /// parameter is a portable constant mask, a streamed bool operand needs the x86 expansion.</param>
        /// <param name="lane">Receives W on success: the root's dtype when it is a number (the output
        /// then stores at W, as in the uniform plan), else the first non-bool input's (or the smallest
        /// non-bool node dtype when every input is bool) — any dtype of the size carries the masks.</param>
        /// <param name="sameWidth">Receives true on success.</param>
        /// <returns>True when the tree vectorizes under the same-width contract on this host.</returns>
        /// <remarks>
        /// The bool gates are the uniform plan's: a bool input or output needs the 128/256-bit
        /// mask helpers, and a STREAMED bool input the x86 byte→lane expansion at W's size.
        /// </remarks>
        private static bool TryPlanSameWidth(
            NDExpr root, NPTypeCode[] inputTypes, IReadOnlyDictionary<NDExpr, NPTypeCode> nodeTypes, bool[]? isParam,
            out NPTypeCode lane, out bool sameWidth)
        {
            lane = NPTypeCode.Empty;
            sameWidth = false;

            int size = 0;
            NPTypeCode firstInput = NPTypeCode.Empty;
            bool boolInput = false, boolStreamedInput = false;
            for (int i = 0; i < inputTypes.Length; i++)
            {
                var t = inputTypes[i];
                if (t == NPTypeCode.Boolean)
                {
                    boolInput = true;
                    if (isParam is null || i >= isParam.Length || !isParam[i])
                        boolStreamedInput = true;
                    continue;
                }
                if (!AdmitSameWidth(t, ref size))
                    return false;
                if (firstInput == NPTypeCode.Empty)
                    firstInput = t;
            }

            // The node dtypes decide too: a weak literal can lift a node past every input dtype
            // (`(i8 > 30) * 1.5` types its multiply float64 over an int64-only input set).
            NPTypeCode smallestNode = NPTypeCode.Empty;
            foreach (var kv in nodeTypes)
            {
                var t = kv.Value;
                if (t == NPTypeCode.Boolean)
                    continue;
                if (!AdmitSameWidth(t, ref size))
                    return false;
                // The smallest code, not the first met: the dictionary's order is not a contract, and
                // the choice must be a function of the tree (it shapes the emitted reinterprets).
                if (smallestNode == NPTypeCode.Empty || t < smallestNode)
                    smallestNode = t;
            }

            // No non-bool dtype at all is byte mode's tree, already answered by the uniform plan.
            if (size == 0 || !nodeTypes.TryGetValue(root, out var rootType))
                return false;

            var w = rootType != NPTypeCode.Boolean ? rootType
                : firstInput != NPTypeCode.Empty ? firstInput
                : smallestNode;

            bool boolIo = boolInput || rootType == NPTypeCode.Boolean;
            if (boolIo && !DirectILKernelGenerator.FusedBoolLanesAvailable)
                return false;
            if (boolStreamedInput && !DirectILKernelGenerator.FusedBoolInputMasksAvailable(w))
                return false;
            if (!root.CanEmitVectorV2(w, nodeTypes))
                return false;

            lane = w;
            sameWidth = true;
            return true;
        }

        /// <summary>
        /// Admits one non-bool dtype into a same-width plan: a SIMD dtype (not Half / Decimal /
        /// Complex / Char, which have no BCL vector lanes here) of the size the plan has seen so far.
        /// </summary>
        /// <param name="t">The dtype (never Boolean — the caller skips masks).</param>
        /// <param name="size">The plan's element size so far (0 = none yet); set by the first dtype.</param>
        /// <returns>False when <paramref name="t"/> is not SIMD-capable or its size differs.</returns>
        private static bool AdmitSameWidth(NPTypeCode t, ref int size)
        {
            if (!DirectILKernelGenerator.CanUseSimd(t))
                return false;
            int s = DirectILKernelGenerator.GetTypeSize(t);
            if (size == 0)
                size = s;
            return s == size;
        }

        /// <summary>
        /// The P5.2 mixed-width plan ("lane groups"): the lane W is the ROOT dtype — the widest,
        /// since NEP50 promotion never narrows along a parent edge — and the group's element count
        /// is one full <c>Vector&lt;W&gt;</c>. Every node computes at its OWN dtype in a container of
        /// <c>elemCount·sizeof(dtype)</c> bytes, admitted only when that container is 128 or 256
        /// bits (so half-lane dtypes ride <c>Vector128</c> and a partial load is exactly 16 bytes,
        /// never an over-read), and every child→parent edge must be an exact vector widening
        /// (<see cref="NDExprVec.HasWidenEdge"/>). This increment excludes Boolean anywhere (no mask
        /// widths to reconcile), Half/Decimal/Complex/Char (not SIMD lanes here), and admits only
        /// the node kinds <see cref="NDExpr.CanEmitVectorMixed"/> opts in (leaves + binary
        /// arithmetic/bitwise — the P5.1 "arithmetic only" shape); anything else keeps the whole
        /// tree on the scalar path, the pre-P5.2 behavior.
        /// </summary>
        /// <param name="root">The bound tree's root.</param>
        /// <param name="inputTypes">Every input's dtype (parameters included).</param>
        /// <param name="nodeTypes">The typing pass's node→dtype table.</param>
        /// <param name="lane">Receives the mixed lane (the root dtype) on success.</param>
        /// <param name="mixedWidth">Receives true on success.</param>
        /// <returns>True when the tree vectorizes under the mixed-width contract.</returns>
        private static bool TryPlanMixed(
            NDExpr root, NPTypeCode[] inputTypes, IReadOnlyDictionary<NDExpr, NPTypeCode> nodeTypes,
            out NPTypeCode lane, out bool mixedWidth)
        {
            lane = NPTypeCode.Empty;
            mixedWidth = false;
            if (!DirectILKernelGenerator.FusedMixedWidthAvailable)
                return false;

            // Bool operands would need lane masks at PER-NODE widths (mask widening/narrowing at
            // every edge) — deferred; a bool input keeps the mixed tree scalar.
            foreach (var t in inputTypes)
                if (t == NPTypeCode.Boolean)
                    return false;

            if (!nodeTypes.TryGetValue(root, out var rootType) || !DirectILKernelGenerator.CanUseSimd(rootType))
                return false;

            int laneSize = DirectILKernelGenerator.GetTypeSize(rootType);
            foreach (var kv in nodeTypes)
            {
                var t = kv.Value;
                if (t == rootType)
                    continue;
                // The container rule: a node's vector is elemCount·sizeof(t) bytes, legal only at
                // 128/256 bits — i.e. sizeof(t) equal to the lane's or exactly half. (A Boolean or
                // non-SIMD dtype anywhere also declines: no bool masks, no Half/Decimal/Complex.)
                if (t == NPTypeCode.Boolean || !DirectILKernelGenerator.CanUseSimd(t))
                    return false;
                int size = DirectILKernelGenerator.GetTypeSize(t);
                if (size != laneSize && size * 2 != laneSize)
                    return false;
            }

            if (!root.CanEmitVectorMixed(nodeTypes))
                return false;

            lane = rootType;
            mixedWidth = true;
            return true;
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
        /// Whether this node (and its subtree) has a MIXED-WIDTH vector emit (P5.2): nodes compute
        /// at their own dtype, every child edge an exact half→full-lane widening. Default false —
        /// a node kind that has not opted in keeps its whole tree on the scalar path (the plan's
        /// root call ANDs this up the tree). This increment opts in the leaves (Input/Const) and
        /// binary arithmetic/bitwise only — the P5.1 "pure arithmetic" shape; comparison / where /
        /// min-max / logical / unary / cast nodes stay scalar in a mixed tree because their mask
        /// and single-lane emitters are written for ONE shared lane width.
        /// </summary>
        /// <param name="types">The typing pass's node→dtype table (every dtype already passed the plan's container rule).</param>
        /// <returns>True when the subtree can emit under the mixed-width contract.</returns>
        internal virtual bool CanEmitVectorMixed(IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;

        /// <summary>
        /// Emit <paramref name="child"/>'s vector converted to <paramref name="target"/> (lane or
        /// Boolean): a mask meeting a numeric slot becomes exact 1/0 lanes, a value meeting a Boolean
        /// slot becomes its truthiness mask. Byte mode carries bool bytes only — no conversion. In
        /// MIXED-WIDTH mode a differing child dtype is instead an exact half→full-lane widening
        /// (<see cref="NDExprVec.EmitWidenEdge"/>) — the vector twin of the scalar body's
        /// <c>EmitConvertTo</c> edge (bool never occurs there; the mixed plan excludes it). In a
        /// SAME-WIDTH tree (P5.3) the mask is a <c>Vector&lt;W&gt;</c> while the value side is at its
        /// own dtype, so each mask edge reinterprets between the two (<see cref="NDExprVec.EmitReinterpret"/>
        /// — nothing when the dtype is W, the uniform plan's every case).
        /// </summary>
        /// <param name="il">The IL generator.</param>
        /// <param name="ctx">The compile context (lane W, node dtypes, plan mode).</param>
        /// <param name="child">The child to emit.</param>
        /// <param name="target">The dtype the consuming node needs it at (a value dtype, or Boolean for a mask slot).</param>
        /// <exception cref="InvalidOperationException">A value edge between two different non-bool dtypes outside
        /// the mixed-width plan (a plan bug: <see cref="NDExprVec.IsVectorEdge"/> refuses such trees).</exception>
        private protected static void EmitVectorChildAs(ILGenerator il, NDExprCompileContext ctx, NDExpr child, NPTypeCode target)
        {
            child.EmitVector(il, ctx);
            var from = ctx.TypeOf(child);
            if (from == target)
                return;
            if (ctx.MixedWidth)
            {
                NDExprVec.EmitWidenEdge(il, from, target);
                return;
            }

            var lane = ctx.VectorLaneType;
            if (lane == NPTypeCode.Boolean)
                return;
            if (from == NPTypeCode.Boolean)
            {
                // The Vector<W> mask typed at the slot's dtype first, so `& One` makes 1 / 0 of THAT dtype
                // (1.0 / 0.0 for a float64 slot under an int64 lane).
                NDExprVec.EmitReinterpret(il, lane, target);
                NDExprVec.EmitMaskToNumeric(il, target);
            }
            else if (target == NPTypeCode.Boolean)
            {
                // Truthiness is tested at the value's OWN dtype (a float64 -0.0 is false, its int64 bits are not
                // zero), and only the resulting mask is retyped to W.
                NDExprVec.EmitValueToMask(il, from);
                NDExprVec.EmitReinterpret(il, from, lane);
            }
            else
            {
                throw new InvalidOperationException(
                    $"a {from} child cannot feed a {target} slot in a vector body: a value edge between two dtypes (a vector plan bug).");
            }
        }
    }

    public sealed partial class InputNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;

        // An input's local already holds its own-dtype (possibly partial) vector — the consuming
        // parent's edge does any widening — so a leaf is always mixed-emittable.
        internal override bool CanEmitVectorMixed(IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;
    }

    public sealed partial class ConstNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;

        // A literal broadcasts at its ADOPTED dtype's own container (EmitVectorCreateAt) — the
        // parent edge widens like any child — so it is always mixed-emittable. (A Boolean-typed
        // literal cannot occur here: the mixed plan rejects Boolean nodes tree-wide.)
        internal override bool CanEmitVectorMixed(IReadOnlyDictionary<NDExpr, NPTypeCode> types) => true;
    }

    public sealed partial class ArrayNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class BinaryNode
    {
        /// <summary>The float16 binary ops the fused vector path serves (<see cref="NDExprVec"/> /
        /// <c>HalfArithVec256</c>): the widen-compute-narrow arithmetic four. Every other op keeps a
        /// Half tree scalar.</summary>
        internal static bool IsHalfVectorArith(BinaryOp op)
            => op == BinaryOp.Add || op == BinaryOp.Subtract || op == BinaryOp.Multiply || op == BinaryOp.Divide;

        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            if (!_left.CanEmitVectorV2(lane, types) || !_right.CanEmitVectorV2(lane, types))
                return false;
            var my = types[this];
            if (my == NPTypeCode.Boolean)
                return _op == BinaryOp.Add || _op == BinaryOp.Multiply ||
                       _op == BinaryOp.BitwiseAnd || _op == BinaryOp.BitwiseOr || _op == BinaryOp.BitwiseXor;
            // A Half tree vectorizes only the arithmetic four (widen-compute-narrow); a mixed op
            // (power/mod/floor_divide/min-max/bitwise on Half) has no Vector256<ushort> body → scalar.
            if (lane == NPTypeCode.Half)
                return my == NPTypeCode.Half && IsHalfVectorArith(_op);
            // Both children arrive at this node's dtype — the same dtype or a bool mask; a value edge
            // between two dtypes (int64 + float64) has no vector form (always true in the uniform plan).
            if (!NDExprVec.IsVectorEdge(types[_left], my) || !NDExprVec.IsVectorEdge(types[_right], my))
                return false;
            return IsSimdOp(_op);
        }

        /// <summary>
        /// Mixed-width (P5.2): the arithmetic four and the bitwise three at this node's OWN dtype
        /// (a half-lane node runs at <c>Vector128</c> so e.g. the int32 multiply of <c>i4*2+f8</c>
        /// WRAPS at int32 exactly like the scalar body and NumPy's unfused sequence), each child
        /// edge an exact widening from the map. Mod/Power/FloorDivide/ATan2 have no vector body
        /// (scalar-only, same as v2); Min/Max ops live on MinMaxNode, which has not opted in.
        /// A same-size-but-different-dtype edge — notably int64→float64, which ROUNDS and has no
        /// AVX2 vector form — is absent from the map, so such a tree stays whole-tree scalar.
        /// </summary>
        internal override bool CanEmitVectorMixed(IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            if (!IsSimdOp(_op))
                return false;
            var my = types[this];
            var lt = types[_left];
            var rt = types[_right];
            if (lt != my && !NDExprVec.HasWidenEdge(lt, my))
                return false;
            if (rt != my && !NDExprVec.HasWidenEdge(rt, my))
                return false;
            return _left.CanEmitVectorMixed(types) && _right.CanEmitVectorMixed(types);
        }
    }

    public sealed partial class UnaryNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            // Phase 5 vectorizes only f16 ARITHMETIC (BinaryNode); a Half unary (abs/negate/square/
            // rounding/transcendental) has no Vector256<ushort> body, so it keeps its whole tree scalar.
            if (lane == NPTypeCode.Half)
                return false;
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

            // The op runs at this node's own dtype: W in the uniform plan, any dtype of W's size in the
            // same-width one; the child arrives at that dtype (or as a bool mask), never converted.
            if (!NDExprVec.IsVectorEdge(ct, my))
                return false;
            // floor/ceil/round/trunc are the identity on integer lanes
            if (IsRoundingOp(_op) && IsIntegerKind(my))
                return true;
            return IsSimdUnaryAt(_op, my);
        }
    }

    public sealed partial class CastNode
    {
        // Scalar-only: the SIMD-widening cast lanes are Phase 5. Returning false forces the whole
        // tree scalar (a subtree's false propagates up through the ANDed root gate).
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class RoundNode
    {
        // Scalar-only: the multi-step mul→rint→div composition (and the integer float64 round-trip /
        // complex per-lane path) is not vectorized here. Forces the whole tree scalar.
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class ComparisonNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
        {
            // A comparison in a Half tree (its lane is the Vector256<ushort> f16 arithmetic lane) has
            // no vector body here — keep the whole tree scalar (Phase 5 is f16 arithmetic only).
            if (lane == NPTypeCode.Half)
                return false;
            if (!_left.CanEmitVectorV2(lane, types) || !_right.CanEmitVectorV2(lane, types))
                return false;
            var cmp = NDExprTypeRules.ComparisonType(types[_left], types[_right]);
            if (cmp == NPTypeCode.Boolean)
                return true;
            // The compare runs at its common dtype: W in the uniform plan, or — in a same-width tree —
            // any SIMD dtype of W's size, its mask then retyped to W. Neither operand may need a value
            // conversion to get there (int64 vs float64 compares at float64: a rounding edge).
            return (cmp == lane || (DirectILKernelGenerator.CanUseSimd(cmp)
                                    && DirectILKernelGenerator.GetTypeSize(cmp) == DirectILKernelGenerator.GetTypeSize(lane)))
                   && NDExprVec.IsVectorEdge(types[_left], cmp) && NDExprVec.IsVectorEdge(types[_right], cmp);
        }
    }

    public sealed partial class MinMaxNode
    {
        // The min / max runs at this node's own dtype (W, or a same-width dtype); both operands
        // arrive at it as the same dtype or a bool mask (a bool result is the mask and / or).
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
            => lane != NPTypeCode.Half   // no f16 min/max vector body (Phase 5 = f16 arithmetic only)
               && _left.CanEmitVectorV2(lane, types) && _right.CanEmitVectorV2(lane, types)
               && NDExprVec.IsVectorEdge(types[_left], types[this]) && NDExprVec.IsVectorEdge(types[_right], types[this]);
    }

    public sealed partial class WhereNode
    {
        // The condition is a mask slot (any child); the branches arrive at this node's dtype — the
        // same dtype or a bool mask — so a same-width tree selects float64 under an int64 mask, but a
        // branch needing int64 → float64 keeps the tree scalar.
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
            => lane != NPTypeCode.Half   // no f16 select vector body (Phase 5 = f16 arithmetic only)
               && _cond.CanEmitVectorV2(lane, types) && _a.CanEmitVectorV2(lane, types) && _b.CanEmitVectorV2(lane, types)
               && NDExprVec.IsVectorEdge(types[_a], types[this]) && NDExprVec.IsVectorEdge(types[_b], types[this]);
    }

    public sealed partial class LogicalNode
    {
        // Bool result via mask AND/OR/XOR; vectorizes whenever both operands do (each becomes a
        // truthiness lane mask). The plan gates the lane to a SIMD-capable dtype, so complex/decimal/
        // Half operands never reach the vector path (their lane is not SIMD-capable → whole tree scalar).
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types)
            => lane != NPTypeCode.Half   // no f16 logical vector body (Phase 5 = f16 arithmetic only)
               && _left.CanEmitVectorV2(lane, types) && _right.CanEmitVectorV2(lane, types);
    }

    public sealed partial class CallNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class ReduceNode
    {
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }

    public sealed partial class WeightedAverageNode
    {
        // Host-driven (no elementwise vector body); the two child sub-programs vectorize on their own.
        internal override bool CanEmitVectorV2(NPTypeCode lane, IReadOnlyDictionary<NDExpr, NPTypeCode> types) => false;
    }
}
