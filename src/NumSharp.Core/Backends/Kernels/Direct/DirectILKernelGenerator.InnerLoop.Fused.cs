using System;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;

// =============================================================================
// DirectILKernelGenerator.InnerLoop.Fused.cs — the np.evaluate (NDExpr) inner-loop shell
// =============================================================================
//
// The Tier-3B shell in InnerLoop.cs vectorizes only when EVERY operand shares one SIMD
// dtype — the right contract for a single ufunc, and the reason a fused tree with a
// comparison, a where, a mask or a bool operand used to fall to the scalar path whole.
//
// This shell is the "vector v2" contract of docs/plans/ndexpr-evaluate.md Phase 1:
//
//   * ONE compute lane dtype W per kernel (the unique non-bool operand dtype; Boolean when
//     every operand is bool — "byte mode").
//   * an operand of dtype W is loaded as Vector<W>; a BOOL operand (W != Boolean) is loaded
//     as a Vector<W> LANE MASK (all-ones / zero per lane) through the np.where kernels'
//     EmitInlineMaskCreation; a bool OUTPUT is a lane mask packed to one byte per lane.
//   * the vector body sees N vectors of the SAME CLR type (Vector<lane(W)>) and leaves one.
//   * runtime dispatch, in order: all operands inner-contiguous -> the 4x unrolled SIMD
//     loop; every input contiguous OR broadcast (stride 0, hoisted once) with a contiguous
//     output -> the broadcast-aware SIMD loop; strided 32/64-bit W with no bool operand ->
//     the AVX2 gather loop of the production shell; else the scalar strided fallback.
//
// The scalar body contract is the Tier-3B one (bool as an int32 0/1 scalar), so the tail
// and the fallback are the SAME code the scalar-only kernel runs — the vector path can only
// differ from it by a kernel bug, which the metamorphic sweep (NDEvaluateVectorTests) and
// the evaluate.jsonl tier are there to catch.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// Bool operands ride the lane-mask expansion (<see cref="EmitInlineMaskCreation"/>) and the
        /// mask->byte pack, both written for 128/256-bit vectors. A 512-bit host keeps the
        /// bool-free trees vectorized and lets bool-carrying trees take the scalar path.
        /// </summary>
        internal static bool FusedBoolLanesAvailable => VectorBits == 128 || VectorBits == 256;

        /// <summary>
        /// Whether the fused shell can run a vector body for these operands at lane dtype
        /// <paramref name="laneType"/>. Mirrors the NDExpr plan gate; re-checked here so a
        /// mismatched caller gets the scalar shell rather than malformed IL.
        /// </summary>
        internal static bool FusedSimdViable(NPTypeCode[] operandTypes, NPTypeCode laneType)
        {
            if (VectorBits == 0)
                return false;
            if (laneType == NPTypeCode.Boolean)
            {
                foreach (var t in operandTypes)
                    if (t != NPTypeCode.Boolean)
                        return false;
                return true;
            }

            if (!CanUseSimd(laneType))
                return false;
            bool anyBool = false;
            foreach (var t in operandTypes)
            {
                if (t == NPTypeCode.Boolean) anyBool = true;
                else if (t != laneType) return false;
            }

            return !anyBool || FusedBoolLanesAvailable;
        }

        /// <summary>
        /// Compile a fused-expression inner loop (see the file header for the contract).
        /// <paramref name="operandTypes"/> is [inputs..., output]; <paramref name="laneType"/> is the
        /// compute lane dtype W of the vector body (ignored when <paramref name="vectorBody"/> is null).
        /// </summary>
        /// <param name="operandTypes">The ITERATOR operands' dtypes, [inputs..., output].</param>
        /// <param name="laneType">The vector body's lane dtype W (ignored without a vector body).</param>
        /// <param name="scalarBody">Stack [operand scalars…] → [result scalar].</param>
        /// <param name="vectorBody">Stack [operand vectors…] → [result vector], or null for a scalar-only kernel.</param>
        /// <param name="cacheKey">The kernel's identity — everything the emitted IL depends on.</param>
        /// <param name="prologue">
        /// Emitted ONCE at kernel entry, after the operand pointers/strides are in locals and before any
        /// loop — where np.evaluate loads its parameters (0-d inputs hoisted into the aux block,
        /// <c>Ldarg_3</c>) into locals the bodies then read. May declare locals; must leave the stack empty.
        /// </param>
        /// <returns>The compiled (cached) inner loop.</returns>
        /// <exception cref="ArgumentNullException">A required argument is null.</exception>
        /// <exception cref="ArgumentException">Fewer than one input plus the output.</exception>
        internal static NDInnerLoopFunc CompileFusedInnerLoop(
            NPTypeCode[] operandTypes,
            NPTypeCode laneType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey,
            Action<ILGenerator>? prologue = null)
        {
            if (operandTypes is null) throw new ArgumentNullException(nameof(operandTypes));
            if (operandTypes.Length < 2)
                throw new ArgumentException("Need at least 1 input + 1 output operand.", nameof(operandTypes));
            if (scalarBody is null) throw new ArgumentNullException(nameof(scalarBody));
            if (cacheKey is null) throw new ArgumentNullException(nameof(cacheKey));

            return _innerLoopCache.GetOrAdd(cacheKey, _ =>
                GenerateFusedInnerLoop(operandTypes, laneType, scalarBody, vectorBody, cacheKey, prologue));
        }

        private static NDInnerLoopFunc GenerateFusedInnerLoop(
            NPTypeCode[] operandTypes,
            NPTypeCode laneType,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey,
            Action<ILGenerator>? prologue)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];

            var dm = new DynamicMethod(
                name: $"NDInnerLoop_Fused_{Sanitize(cacheKey)}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var ptrLocals = new LocalBuilder[nOp];
            var strideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                strideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadInnerLoopArgs(il, nOp, ptrLocals, strideLocals);
            // Parameters are loaded here, once per kernel call, ahead of the runtime dispatch — every
            // path below (SIMD, gather, scalar) then reads them from locals.
            prologue?.Invoke(il);

            var lblScalarStrided = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            bool simd = vectorBody != null && FusedSimdViable(operandTypes, laneType);
            if (simd)
            {
                bool byteMode = laneType == NPTypeCode.Boolean;
                int laneSize = byteMode ? 1 : GetTypeSize(laneType);

                // The byte stride each operand must show for a "contiguous" inner axis: W-sized for
                // a W operand, 1 for a bool operand (mask in, packed mask out).
                int ElemSize(int op) => operandTypes[op] == NPTypeCode.Boolean ? 1 : laneSize;

                // ── 1. every operand contiguous → straight SIMD loop ───────────────────────────
                var lblNotAllContig = il.DefineLabel();
                for (int op = 0; op < nOp; op++)
                {
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, (long)ElemSize(op));
                    il.Emit(OpCodes.Bne_Un, lblNotAllContig);
                }
                EmitFusedSimdLoop(il, operandTypes, laneType, ptrLocals, strideLocals, vectorBody!, scalarBody, allowBroadcast: false);
                il.Emit(OpCodes.Br, lblEnd);
                il.MarkLabel(lblNotAllContig);

                // ── 2. contiguous output, every input contiguous OR broadcast (stride 0) ───────
                var lblTryGather = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
                il.Emit(OpCodes.Ldc_I8, (long)ElemSize(nIn));
                il.Emit(OpCodes.Bne_Un, lblTryGather);
                for (int op = 0; op < nIn; op++)
                {
                    var lblOk = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, (long)ElemSize(op));
                    il.Emit(OpCodes.Beq, lblOk);
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Bne_Un, lblTryGather);
                    il.MarkLabel(lblOk);
                }
                EmitFusedSimdLoop(il, operandTypes, laneType, ptrLocals, strideLocals, vectorBody!, scalarBody, allowBroadcast: true);
                il.Emit(OpCodes.Br, lblEnd);
                il.MarkLabel(lblTryGather);

                // ── 3. strided 32/64-bit lanes, no bool operand anywhere → AVX2 gather ────────
                bool allLane = !byteMode && outType == laneType;
                for (int op = 0; allLane && op < nIn; op++)
                    allLane = operandTypes[op] == laneType;
                if (allLane && TryGetGatherSupport(laneType, out var gatherSupport))
                {
                    for (int op = 0; op < nIn; op++)
                    {
                        il.Emit(OpCodes.Ldloc, strideLocals[op]);
                        il.Emit(OpCodes.Ldc_I8, GatherStrideLimit);
                        il.Emit(OpCodes.Bgt, lblScalarStrided);
                        il.Emit(OpCodes.Ldloc, strideLocals[op]);
                        il.Emit(OpCodes.Ldc_I8, -GatherStrideLimit);
                        il.Emit(OpCodes.Blt, lblScalarStrided);
                    }

                    var lblGatherScatter = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
                    il.Emit(OpCodes.Ldc_I8, (long)laneSize);
                    il.Emit(OpCodes.Bne_Un, lblGatherScatter);
                    EmitSimdGatherLoop(il, operandTypes, ptrLocals, strideLocals,
                        vectorBody!, scalarBody, gatherSupport, scatterOut: false);
                    il.Emit(OpCodes.Br, lblEnd);

                    il.MarkLabel(lblGatherScatter);
                    il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
                    il.Emit(OpCodes.Ldc_I8, GatherStrideLimit);
                    il.Emit(OpCodes.Bgt, lblScalarStrided);
                    il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
                    il.Emit(OpCodes.Ldc_I8, -GatherStrideLimit);
                    il.Emit(OpCodes.Blt, lblScalarStrided);
                    EmitSimdGatherLoop(il, operandTypes, ptrLocals, strideLocals,
                        vectorBody!, scalarBody, gatherSupport, scatterOut: true);
                    il.Emit(OpCodes.Br, lblEnd);
                }

                il.MarkLabel(lblScalarStrided);
            }
            else
            {
                // No vector body: the contiguous scalar loop when every stride is its element size
                // (constant-stride addressing the JIT folds), else the strided fallback.
                for (int op = 0; op < nOp; op++)
                {
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, (long)GetTypeSize(operandTypes[op]));
                    il.Emit(OpCodes.Bne_Un, lblScalarStrided);
                }
                EmitScalarContigLoop(il, operandTypes, ptrLocals, scalarBody);
                il.Emit(OpCodes.Br, lblEnd);
                il.MarkLabel(lblScalarStrided);
            }

            EmitScalarStridedLoop(il, operandTypes, ptrLocals, strideLocals, scalarBody);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);
            return dm.CreateDelegate<NDInnerLoopFunc>();
        }

        /// <summary>
        /// The fused SIMD loop: 4x unrolled + one-vector remainder + a stride-aware scalar tail.
        /// With <paramref name="allowBroadcast"/> every input may also be a stride-0 operand: its
        /// vector is hoisted ONCE before the loop and selected per group by a loop-invariant
        /// (perfectly predicted) branch — an N-ary generalization of the production shell's
        /// scalar-lhs / scalar-rhs paths that needs no per-pattern specialization.
        /// </summary>
        private static void EmitFusedSimdLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            NPTypeCode laneType,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody,
            bool allowBroadcast)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            bool byteMode = laneType == NPTypeCode.Boolean;
            long lanes = byteMode ? VectorBytes : GetVectorCount(laneType);
            long unrollStep = lanes * 4;
            var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(laneType));

            LocalBuilder[]? isBroadcast = null;
            LocalBuilder[]? hoisted = null;
            if (allowBroadcast)
            {
                isBroadcast = new LocalBuilder[nIn];
                hoisted = new LocalBuilder[nIn];
                for (int op = 0; op < nIn; op++)
                {
                    isBroadcast[op] = il.DeclareLocal(typeof(int));
                    hoisted[op] = il.DeclareLocal(vecType);

                    // isBroadcast = (stride == 0)
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Ceq);
                    il.Emit(OpCodes.Stloc, isBroadcast[op]);

                    // if (isBroadcast) hoisted = broadcast(*ptr)
                    var lblSkip = il.DefineLabel();
                    il.Emit(OpCodes.Ldloc, isBroadcast[op]);
                    il.Emit(OpCodes.Brfalse, lblSkip);
                    EmitFusedBroadcastScalar(il, operandTypes[op], laneType, ptrLocals[op]);
                    il.Emit(OpCodes.Stloc, hoisted[op]);
                    il.MarkLabel(lblSkip);
                }
            }

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep; vectorEnd = count - lanes; i = 0
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, lanes);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            void EmitGroup(long offset)
            {
                for (int op = 0; op < nIn; op++)
                    EmitFusedLoad(il, operandTypes[op], laneType, ptrLocals[op], locI, offset,
                        isBroadcast?[op], hoisted?[op]);
                vectorBody(il);
                EmitFusedStore(il, operandTypes[nIn], laneType, ptrLocals[nIn], locI, offset);
            }

            // A bool OUTPUT of 2/4/8-byte lanes packs the four masks of an unrolled block with
            // vector narrows into ONE 16/32-byte store (NumPy's npyv_pack_b8_b64/b32 shape) instead
            // of four PDEP + 4-byte stores — the comparison kernels are store-bound at L2 sizes.
            bool packBlock = !byteMode && operandTypes[nIn] == NPTypeCode.Boolean
                             && VectorBits == 256 && GetTypeSize(laneType) >= 2;
            LocalBuilder[]? maskLocals = null;
            if (packBlock)
            {
                maskLocals = new LocalBuilder[4];
                for (int u = 0; u < 4; u++) maskLocals[u] = il.DeclareLocal(vecType);
            }

            // === 4× UNROLLED ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            if (packBlock)
            {
                for (int u = 0; u < 4; u++)
                {
                    for (int op = 0; op < nIn; op++)
                        EmitFusedLoad(il, operandTypes[op], laneType, ptrLocals[op], locI, u * lanes,
                            isBroadcast?[op], hoisted?[op]);
                    vectorBody(il);
                    il.Emit(OpCodes.Stloc, maskLocals![u]);
                }
                EmitPackedMaskBlockStore(il, laneType, maskLocals!, ptrLocals[nIn], locI);
            }
            else
            {
                for (int u = 0; u < 4; u++)
                    EmitGroup(u * lanes);
            }
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === REMAINDER (one vector) ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);
            EmitGroup(0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, lanes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL — stride-aware, so a broadcast (stride 0) operand reads its one element ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblTailEnd);
            EmitScalarElement(il, operandTypes, ptrLocals, strideLocals, locI, contig: false, scalarBody);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>Stack: [] → [Vector&lt;lane&gt;] — the broadcast of one operand's single element.</summary>
        private static void EmitFusedBroadcastScalar(ILGenerator il, NPTypeCode opType, NPTypeCode laneType, LocalBuilder ptr)
        {
            if (laneType == NPTypeCode.Boolean || opType == laneType)
            {
                il.Emit(OpCodes.Ldloc, ptr);
                EmitLoadIndirect(il, opType);
                EmitVectorCreate(il, laneType);
                return;
            }

            // bool operand in W-mode: *ptr != 0 ? AllBitsSet : Zero (the mask of a broadcast bool)
            var clr = GetClrType(laneType);
            var lblTrue = il.DefineLabel();
            var lblDone = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, ptr);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brtrue, lblTrue);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(VectorBits, clr), null);
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblTrue);
            il.EmitCall(OpCodes.Call, VectorMethodCache.AllBitsSet(VectorBits, clr), null);
            il.MarkLabel(lblDone);
        }

        /// <summary>
        /// Stack: [] → [Vector&lt;lane&gt;] — operand <c>op</c>'s vector at element (i + offset): a
        /// contiguous load (a bool operand expands to a lane mask), or the hoisted broadcast.
        /// </summary>
        private static void EmitFusedLoad(
            ILGenerator il, NPTypeCode opType, NPTypeCode laneType,
            LocalBuilder ptr, LocalBuilder locI, long offset,
            LocalBuilder? isBroadcast, LocalBuilder? hoisted)
        {
            Label lblBroadcast = default, lblDone = default;
            if (isBroadcast != null)
            {
                lblBroadcast = il.DefineLabel();
                lblDone = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, isBroadcast);
                il.Emit(OpCodes.Brtrue, lblBroadcast);
            }

            if (laneType == NPTypeCode.Boolean || opType == laneType)
            {
                EmitAddrIPlusOffset(il, ptr, locI, offset, laneType == NPTypeCode.Boolean ? 1 : GetTypeSize(laneType));
                EmitVectorLoad(il, laneType);
            }
            else
            {
                // bool operand in W-mode: N condition bytes → Vector<W> lane mask
                EmitAddrIPlusOffset(il, ptr, locI, offset, 1);
                EmitBoolBytesToLaneMask(il, laneType);
            }

            if (isBroadcast != null)
            {
                il.Emit(OpCodes.Br, lblDone);
                il.MarkLabel(lblBroadcast);
                il.Emit(OpCodes.Ldloc, hoisted!);
                il.MarkLabel(lblDone);
            }
        }

        /// <summary>Stack: [Vector&lt;lane&gt;] → [] — store the group's result at element (i + offset).</summary>
        private static void EmitFusedStore(
            ILGenerator il, NPTypeCode outType, NPTypeCode laneType,
            LocalBuilder ptr, LocalBuilder locI, long offset)
        {
            if (laneType == NPTypeCode.Boolean || outType == laneType)
            {
                EmitAddrIPlusOffset(il, ptr, locI, offset, laneType == NPTypeCode.Boolean ? 1 : GetTypeSize(laneType));
                EmitVectorStore(il, laneType);     // [vec, ptr]
                return;
            }

            // bool output in W-mode: pack the lane mask to one 0/1 byte per lane
            EmitAddrIPlusOffset(il, ptr, locI, offset, 1);
            EmitStoreMaskAsBool(il, laneType);
        }

        /// <summary>
        /// Stack: [byte*] → [Vector&lt;W&gt; mask] — <c>lanes(W)</c> condition bytes expanded to an
        /// all-ones / zero lane mask of W's width (the np.where kernels' expansion), reinterpreted
        /// as W's CLR lane type so it composes with W-typed vectors in ConditionalSelect / And.
        /// </summary>
        internal static void EmitBoolBytesToLaneMask(ILGenerator il, NPTypeCode laneType)
        {
            int size = GetTypeSize(laneType);
            EmitInlineMaskCreation(il, VectorBits, size);
            Type produced = size switch
            {
                1 => typeof(byte),
                2 => typeof(ushort),
                4 => typeof(uint),
                8 => typeof(ulong),
                _ => throw new NotSupportedException($"lane size {size} has no bool mask expansion"),
            };
            var clr = GetClrType(laneType);
            if (produced != clr)
                il.EmitCall(OpCodes.Call, VectorMethodCache.As(VectorBits, produced, clr), null);
        }

        /// <summary>
        /// Stack: [mask(Vector&lt;W&gt;), byte* dst] → [] — one 0/1 byte per lane. 1-byte lanes store
        /// <c>mask &amp; 1</c> whole; 2-byte lanes narrow the 0/1 lanes to bytes (256-bit); 4/8-byte
        /// lanes (and the 128-bit 2-byte case) extract the MSB bits and deposit them as bytes with
        /// one BMI2 PDEP + one store, or per lane without BMI2 — the comparison kernel's pack.
        /// </summary>
        internal static void EmitStoreMaskAsBool(ILGenerator il, NPTypeCode laneType)
        {
            var clr = GetClrType(laneType);
            int size = GetTypeSize(laneType);
            int lanes = GetVectorCount(laneType);
            var locDst = il.DeclareLocal(typeof(byte*));
            il.Emit(OpCodes.Stloc, locDst);                                  // [mask]

            if (size == 1)
            {
                if (clr != typeof(byte))
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(VectorBits, clr, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.One(VectorBits, typeof(byte)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(VectorBits, "BitwiseAnd", typeof(byte), paramCount: 2), null);
                il.Emit(OpCodes.Ldloc, locDst);
                EmitVectorStore(il, NPTypeCode.Byte);                        // [v, ptr] → []
                return;
            }

            if (size == 2 && VectorBits == 256)
            {
                // 16 ushort lanes of 0/1 → Narrow(v, v) puts them in the LOWER 16 bytes → one V128 store.
                if (clr != typeof(ushort))
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(VectorBits, clr, typeof(ushort)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.One(VectorBits, typeof(ushort)), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(VectorBits, "BitwiseAnd", typeof(ushort), paramCount: 2), null);
                var locV = il.DeclareLocal(VectorMethodCache.V(VectorBits, typeof(ushort)));
                il.Emit(OpCodes.Stloc, locV);
                il.Emit(OpCodes.Ldloc, locV);
                il.Emit(OpCodes.Ldloc, locV);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Narrow(VectorBits, typeof(ushort)), null);   // V256<byte>
                il.EmitCall(OpCodes.Call, VectorMethodCache.GetLower(VectorBits, typeof(byte)), null);   // V128<byte>
                il.Emit(OpCodes.Ldloc, locDst);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Store(128, typeof(byte)), null);            // [v, ptr] → []
                return;
            }

            // MSB extraction: one bit per lane (comparison masks are all-ones / zero).
            il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(VectorBits, clr), null);
            var locBits = il.DeclareLocal(typeof(uint));
            il.Emit(OpCodes.Stloc, locBits);

            if (s_bmi2X64Pdep != null && lanes <= 8 && (lanes == 8 || lanes == 4 || lanes == 2))
            {
                il.Emit(OpCodes.Ldloc, locDst);
                il.Emit(OpCodes.Ldloc, locBits);
                il.Emit(OpCodes.Conv_U8);
                il.Emit(OpCodes.Ldc_I8, unchecked((long)0x0101010101010101UL));
                il.EmitCall(OpCodes.Call, s_bmi2X64Pdep, null);                // [ptr, packed(ulong)]
                switch (lanes)
                {
                    case 8: il.Emit(OpCodes.Stind_I8); break;
                    case 4: il.Emit(OpCodes.Conv_U4); il.Emit(OpCodes.Stind_I4); break;
                    default: il.Emit(OpCodes.Conv_U2); il.Emit(OpCodes.Stind_I2); break;
                }
                return;
            }

            for (int j = 0; j < lanes; j++)
            {
                il.Emit(OpCodes.Ldloc, locDst);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldloc, locBits);
                if (j > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, j);
                    il.Emit(OpCodes.Shr_Un);
                }
                il.Emit(OpCodes.Ldc_I4_1);
                il.Emit(OpCodes.And);
                il.Emit(OpCodes.Stind_I1);
            }
        }

        /// <summary>
        /// Store the four lane masks of one unrolled block (256-bit, lane size 2/4/8) as
        /// <c>4·lanes</c> 0/1 bytes at <c>out + i</c> with vector narrows: 8-byte lanes
        /// (4×4 = 16 bools) narrow ulong→uint→ushort→byte and store 16 bytes; 4-byte lanes
        /// (4×8 = 32) narrow uint→ushort→byte and store 32; 2-byte lanes (4×16 = 64) narrow
        /// ushort→byte twice and store 2×32. <c>Vector256.Narrow(lower, upper)</c> keeps lane order
        /// (lower's lanes first) and truncates, so an all-ones mask stays all-ones down to the byte;
        /// <c>&amp; 1</c> then makes the canonical 0/1.
        /// </summary>
        private static void EmitPackedMaskBlockStore(ILGenerator il, NPTypeCode laneType, LocalBuilder[] masks, LocalBuilder outPtr, LocalBuilder locI)
        {
            var clr = GetClrType(laneType);
            int size = GetTypeSize(laneType);
            Type u8 = typeof(byte), u16 = typeof(ushort), u32 = typeof(uint), u64 = typeof(ulong);

            void LoadAs(int k, Type unsignedElem)
            {
                il.Emit(OpCodes.Ldloc, masks[k]);
                if (clr != unsignedElem)
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(VectorBits, clr, unsignedElem), null);
            }

            void Narrow(Type fromElem) => il.EmitCall(OpCodes.Call, VectorMethodCache.Narrow(VectorBits, fromElem), null);

            void AndOneBytes()
            {
                il.EmitCall(OpCodes.Call, VectorMethodCache.One(VectorBits, u8), null);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(VectorBits, "BitwiseAnd", u8, paramCount: 2), null);
            }

            void OutAddr(long byteOffset)
            {
                il.Emit(OpCodes.Ldloc, outPtr);
                il.Emit(OpCodes.Ldloc, locI);
                if (byteOffset != 0)
                {
                    il.Emit(OpCodes.Ldc_I8, byteOffset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Add);
            }

            switch (size)
            {
                case 8:
                {
                    var locN = il.DeclareLocal(VectorMethodCache.V(VectorBits, u16));
                    LoadAs(0, u64); LoadAs(1, u64); Narrow(u64);            // V<uint>  lanes 0..7
                    LoadAs(2, u64); LoadAs(3, u64); Narrow(u64);            // V<uint>  lanes 8..15
                    Narrow(u32);                                            // V<ushort> 16 lanes
                    il.Emit(OpCodes.Stloc, locN);
                    il.Emit(OpCodes.Ldloc, locN);
                    il.Emit(OpCodes.Ldloc, locN);
                    Narrow(u16);                                            // V<byte>, lower 16 valid
                    AndOneBytes();
                    il.EmitCall(OpCodes.Call, VectorMethodCache.GetLower(VectorBits, u8), null);  // V128<byte>
                    OutAddr(0);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.Store(128, u8), null);            // [v, ptr] → []
                    return;
                }
                case 4:
                {
                    LoadAs(0, u32); LoadAs(1, u32); Narrow(u32);            // V<ushort> lanes 0..15
                    LoadAs(2, u32); LoadAs(3, u32); Narrow(u32);            // V<ushort> lanes 16..31
                    Narrow(u16);                                            // V<byte> 32 lanes
                    AndOneBytes();
                    OutAddr(0);
                    EmitVectorStore(il, NPTypeCode.Byte);
                    return;
                }
                default: // 2
                {
                    LoadAs(0, u16); LoadAs(1, u16); Narrow(u16);            // V<byte> lanes 0..31
                    AndOneBytes();
                    OutAddr(0);
                    EmitVectorStore(il, NPTypeCode.Byte);
                    LoadAs(2, u16); LoadAs(3, u16); Narrow(u16);            // V<byte> lanes 32..63
                    AndOneBytes();
                    OutAddr(32);
                    EmitVectorStore(il, NPTypeCode.Byte);
                    return;
                }
            }
        }

        /// <summary>
        /// Stack: [lhs, rhs] (Vector&lt;T&gt;) → [mask] — the comparison kernel's vector compare:
        /// all-ones / zero lanes at T (NaN lanes compare false; NotEqual is ~Equals, so NaN != NaN
        /// is true like NumPy), 0/1 bytes when T is Boolean (truth-normalized byte compare).
        /// </summary>
        internal static void EmitVectorCompareMask(ILGenerator il, ComparisonOp op, NPTypeCode type)
            => EmitVectorComparison(il, op, type);
    }
}
