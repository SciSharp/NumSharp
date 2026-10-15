using System;
using System.Reflection.Emit;
using System.Text;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDExpr.Params.cs — 0-d inputs as kernel PARAMETERS (ndexpr-evaluate.md 3.3, "literal-as-parameter")
// =============================================================================
//
// A scalar in a fused tree can be spelled two ways, and both used to cost something on every call:
//   • a LITERAL (`a * 2.5`) is baked into the kernel's IL and its cache key, so a loop that changes
//     the value (`a * t`, a window's `M - 1`) JITs a fresh kernel per value (~0.2–1 ms each) and
//     leaks one kernel-cache entry per value;
//   • a 0-d ARRAY (`a * NDArray.Scalar(t)`) shares one kernel across values, but rode the iterator
//     as a stride-0 operand: ~250 ns of NDIter setup per call plus 8–10 % PER ELEMENT on both the
//     SIMD path (the fused shell's per-load broadcast branch) and the scalar path (the strided
//     fallback's per-element stride multiply) — measured on `a*b+k` and on a cos-window tree.
//
// Here a 0-d input becomes a kernel PARAMETER instead: np.evaluate copies its single element into
// the kernel's aux block (Ldarg_3) once per call, and the kernel's prologue loads it into a local —
// the scalar value for the scalar body, its broadcast Vector<W> (or lane mask, for a bool) for the
// vector body — so an InputNode reading it costs exactly what a literal costs per element, the
// iterator never sees it, and one kernel serves every value. Typing is untouched: the parameter is
// still the STRONG 0-d array it always was (np.float64(k) promotion), and the value is read BEFORE
// the pass runs, so a parameter aliasing the output (a 0-d view into `out`) reads its original value
// like NumPy's COPY_IF_OVERLAP would give it.
//
// The mask "which inputs are parameters" is a shape property decided per resolution (ndim == 0),
// part of the program's identity (hash + structural verification + the kernel cache key) and
// re-validated on the per-root fast path, because a 0-d array can be resized in place. When EVERY
// input is 0-d nothing is hoisted (the result is 0-d and the iterator handles it as before).
// =============================================================================

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    /// The operand/parameter split of one input signature: which inputs stream through the iterator
    /// (and at which operand slot), which are hoisted parameters (and at which aux slot), and the
    /// prologue that loads the parameters. Immutable; built once per program compile.
    /// </summary>
    internal sealed class NDExprParamPlan
    {
        /// <summary>
        /// Bytes per parameter slot in the aux block: the widest dtype (decimal / Complex) is 16 bytes,
        /// and a fixed slot keeps every load naturally aligned without per-dtype offset arithmetic.
        /// </summary>
        internal const int SlotBytes = 16;

        /// <summary>
        /// Where the parameters start inside a REDUCTION kernel's aux block: slot 0 is the flat
        /// reduction's accumulator (the host's 16-byte scratch), so parameters follow it. Elementwise
        /// kernels have no accumulator and start at 0.
        /// </summary>
        internal const int ReduceParamOffset = SlotBytes;

        /// <summary><c>BitConverter.HalfToUInt16Bits</c> — reinterprets a f16 parameter's <see cref="Half"/>
        /// struct to its raw bits for the Vector256&lt;ushort&gt; lane a Half tree broadcasts (see <see cref="EmitPrologue"/>).</summary>
        private static readonly System.Reflection.MethodInfo s_halfToUInt16Bits =
            typeof(BitConverter).GetMethod(nameof(BitConverter.HalfToUInt16Bits), new[] { typeof(Half) })
            ?? throw new MissingMethodException(typeof(BitConverter).FullName, "HalfToUInt16Bits(Half)");

        /// <summary>Input index → iterator operand slot, or -1 for a parameter (null = identity, no parameters).</summary>
        public readonly int[]? Slots;

        /// <summary>Input index → parameter index, or -1 for an operand (null = no parameters).</summary>
        public readonly int[]? ParamIndex;

        /// <summary>The iterator operands' dtypes in operand order (== the inputs when there are no parameters).</summary>
        public readonly NPTypeCode[] OperandTypes;

        /// <summary>The parameters' dtypes in parameter order (empty when there are none).</summary>
        public readonly NPTypeCode[] ParamTypes;

        /// <summary>Appended to the kernel cache key: the parameter mask changes the emitted IL.</summary>
        public readonly string KeySuffix;

        public int OperandCount => OperandTypes.Length;
        public int ParamCount => ParamTypes.Length;

        private NDExprParamPlan(int[]? slots, int[]? paramIndex, NPTypeCode[] operandTypes, NPTypeCode[] paramTypes, string keySuffix)
        {
            Slots = slots;
            ParamIndex = paramIndex;
            OperandTypes = operandTypes;
            ParamTypes = paramTypes;
            KeySuffix = keySuffix;
        }

        /// <summary>
        /// Split <paramref name="inputTypes"/> by <paramref name="isParam"/> (null / all-false = no
        /// parameters: every input is an operand at its own index, the legacy contract).
        /// </summary>
        /// <param name="inputTypes">Every input's dtype in input order.</param>
        /// <param name="isParam">Per input, whether it is hoisted as a parameter; null for none.</param>
        /// <returns>The plan.</returns>
        /// <exception cref="ArgumentException">
        /// The mask length differs from the input count, or every input is a parameter (a kernel needs
        /// at least one iterator operand — the host keeps all-0-d calls on the iterator).
        /// </exception>
        public static NDExprParamPlan Create(NPTypeCode[] inputTypes, bool[]? isParam)
        {
            if (isParam is null || Array.IndexOf(isParam, true) < 0)
                return new NDExprParamPlan(null, null, inputTypes, Array.Empty<NPTypeCode>(), string.Empty);
            if (isParam.Length != inputTypes.Length)
                throw new ArgumentException("parameter mask length must equal the input count.", nameof(isParam));

            int n = inputTypes.Length;
            var slots = new int[n];
            var paramIndex = new int[n];
            int nOps = 0, nParams = 0;
            for (int i = 0; i < n; i++)
            {
                if (isParam[i]) { paramIndex[i] = nParams++; slots[i] = -1; }
                else { slots[i] = nOps++; paramIndex[i] = -1; }
            }

            if (nOps == 0)
                throw new ArgumentException("every input is a parameter — a kernel needs at least one iterator operand.", nameof(isParam));

            var operandTypes = new NPTypeCode[nOps];
            var paramTypes = new NPTypeCode[nParams];
            for (int i = 0; i < n; i++)
            {
                if (isParam[i]) paramTypes[paramIndex[i]] = inputTypes[i];
                else operandTypes[slots[i]] = inputTypes[i];
            }

            var sb = new StringBuilder("|p", 4 + n);
            for (int i = 0; i < n; i++)
                sb.Append(isParam[i] ? '1' : '0');
            return new NDExprParamPlan(slots, paramIndex, operandTypes, paramTypes, sb.ToString());
        }

        /// <summary>
        /// Emit the prologue that loads every parameter from the aux block (<c>Ldarg_3</c>, slot
        /// <paramref name="auxByteOffset"/> + 16·j) into a fresh scalar local, and — when
        /// <paramref name="vectorLocals"/> is given — its broadcast vector at lane
        /// <paramref name="lane"/> into a fresh vector local. Leaves the stack empty.
        /// </summary>
        /// <param name="il">The kernel's IL generator (at entry, stack empty).</param>
        /// <param name="scalarLocals">Receives one local per parameter (the scalar value).</param>
        /// <param name="vectorLocals">Receives one local per parameter (the broadcast vector / lane mask), or null when the kernel has no vector body.</param>
        /// <param name="lane">The vector body's lane dtype W (byte mode = Boolean); ignored without vector locals.</param>
        /// <param name="auxByteOffset">Byte offset of parameter slot 0 inside the aux block.</param>
        /// <param name="mixedWidth">P5.2 mixed-width mode: a non-lane parameter broadcasts at its OWN
        /// dtype's partial container (<c>Vector128</c> for a half-lane dtype) instead of the lane's —
        /// the consuming node's edge widens it, exactly as for a streamed operand.</param>
        /// <exception cref="InvalidOperationException">A parameter dtype the vector plan should have rejected reached the vector prologue.</exception>
        public void EmitPrologue(ILGenerator il, LocalBuilder[] scalarLocals, LocalBuilder[]? vectorLocals, NPTypeCode lane, int auxByteOffset, bool mixedWidth = false)
        {
            for (int j = 0; j < ParamTypes.Length; j++)
            {
                var t = ParamTypes[j];
                scalarLocals[j] = il.DeclareLocal(DirectILKernelGenerator.GetClrType(t));
                il.Emit(OpCodes.Ldarg_3);
                int off = auxByteOffset + SlotBytes * j;
                if (off > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, off);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                DirectILKernelGenerator.EmitLoadIndirect(il, t);
                il.Emit(OpCodes.Stloc, scalarLocals[j]);

                if (vectorLocals is null)
                    continue;

                // Mixed-width: the parameter's vector is its OWN dtype's container (the input-node
                // local contract), so the tree's edges widen it exactly like a streamed operand —
                // one representation for both, which is what keeps 0-d-vs-array metamorphic pairs
                // bit-identical. (No bool/Half parameters here: the mixed plan excludes both.)
                if (mixedWidth && t != lane)
                {
                    int bits = DirectILKernelGenerator.VectorBits * DirectILKernelGenerator.GetTypeSize(t)
                               / DirectILKernelGenerator.GetTypeSize(lane);
                    vectorLocals[j] = il.DeclareLocal(VectorMethodCache.V(bits, DirectILKernelGenerator.GetClrType(t)));
                    il.Emit(OpCodes.Ldloc, scalarLocals[j]);
                    DirectILKernelGenerator.EmitVectorCreateAt(il, t, bits);
                    il.Emit(OpCodes.Stloc, vectorLocals[j]);
                    continue;
                }

                // The vector body sees every input as ONE CLR vector type, Vector<lane(W)>: a W-typed
                // parameter broadcasts its value; a bool parameter in W-mode is a constant lane mask
                // (all-ones / zero), the same representation the shell gives a broadcast bool operand.
                vectorLocals[j] = il.DeclareLocal(VectorMethodCache.V(DirectILKernelGenerator.VectorBits, DirectILKernelGenerator.GetSimdLaneType(lane)));
                if (lane == NPTypeCode.Half)
                {
                    // A Half tree's lane is Vector256<ushort> of raw f16 bits; the scalar local is a
                    // Half STRUCT, so reinterpret it to its ushort pattern before broadcasting (the
                    // plan only admits Half operands into a Half tree, so t == Half here).
                    il.Emit(OpCodes.Ldloc, scalarLocals[j]);
                    il.EmitCall(OpCodes.Call, s_halfToUInt16Bits, null);
                    DirectILKernelGenerator.EmitVectorCreate(il, NPTypeCode.Half);
                }
                else if (lane == NPTypeCode.Boolean || t == lane)
                {
                    il.Emit(OpCodes.Ldloc, scalarLocals[j]);
                    DirectILKernelGenerator.EmitVectorCreate(il, lane);
                }
                else if (t == NPTypeCode.Boolean)
                {
                    var lblTrue = il.DefineLabel();
                    var lblDone = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, scalarLocals[j]);
                    il.Emit(OpCodes.Brtrue, lblTrue);
                    NDExprVec.EmitZero(il, lane);
                    il.Emit(OpCodes.Br, lblDone);
                    il.MarkLabel(lblTrue);
                    NDExprVec.EmitAllBitsSet(il, lane);
                    il.MarkLabel(lblDone);
                }
                else
                {
                    throw new InvalidOperationException(
                        $"parameter {j} is {t} but the vector lane is {lane} — the vector plan admits only lane-typed or bool inputs.");
                }

                il.Emit(OpCodes.Stloc, vectorLocals[j]);
            }
        }
    }
}
