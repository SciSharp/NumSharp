using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// DirectILKernelGenerator.InnerLoop.cs — NDInnerLoopFunc factory
// =============================================================================
//
// Produces kernels with the NumPy ufunc inner-loop signature
//   void(void** dataptrs, long* byteStrides, long count, void* aux)
//
// Unlike the whole-array MixedType kernels (which own the entire loop and take
// shape/ndim/totalSize parameters), these kernels own only the innermost loop
// of NDIter. The iterator drives the outer loop via ForEach / ExecuteGeneric.
//
// THREE ENTRY POINTS
// ------------------
// 1. CompileRawInnerLoop(body, key)
//      Caller emits the entire IL body. Full control. Used by Tier 3A of the
//      NDIter custom-op API.
//
// 2. CompileInnerLoop(operandTypes, scalarBody, vectorBody, key)
//      Caller supplies per-element scalar/vector bodies; the factory wraps
//      them in the standard 4× unrolled SIMD + remainder + scalar-tail shell,
//      plus a strided fallback for non-contiguous inner loops. Used by Tier 3B.
//
// 3. Indirectly via NDExpr.Compile — the expression DSL compiles to Tier 3B.
//
// STRIDE CONTRACT
// ---------------
// NDInnerLoopFunc receives BYTE strides (matching NumPy's C convention).
// The emitted code uses these strides to compute pointer offsets on the
// scalar-strided path. On the contig-inner SIMD path, offsets are computed
// as i * elementSize because the inner stride equals elementSize by definition.
//
// CONTIG-INNER DETECTION
// ----------------------
// Emitted at runtime: compare each operand's stride to its element size. If
// all match, jump to the SIMD path; otherwise run the scalar-strided loop.
// This is cheap (NOp integer compares) and matches what NumPy's inner-loop
// dispatch does.
//
// CACHE
// -----
// Keyed by user-provided string. Caller is responsible for uniqueness. The
// factory stores the compiled delegate so repeated ExecuteElementWise calls
// with the same key return the same kernel instance.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Inner-Loop Kernel Cache

        internal static readonly ConcurrentDictionary<string, NDInnerLoopFunc> _innerLoopCache = new();

        // Cache size + reset are centralized on GeneratedDelegates
        // (GeneratedDelegates.InnerLoopCount / GeneratedDelegates.ClearInnerLoop).

        #endregion

        #region Tier 3A: Raw IL

        /// <summary>
        /// Compile a custom inner-loop kernel from user-emitted IL. The body
        /// is responsible for the entire method — loop, pointer arithmetic,
        /// and return. Arguments are:
        ///   arg0: void** dataptrs    — pointer to operand pointer array
        ///   arg1: long*  byteStrides — pointer to operand byte-stride array
        ///   arg2: long   count       — number of elements in this inner loop
        ///   arg3: void*  auxdata     — opaque cookie
        /// The body MUST emit its own <c>ret</c>.
        /// </summary>
        internal static NDInnerLoopFunc CompileRawInnerLoop(Action<ILGenerator> body, string cacheKey)
        {
            if (body is null) throw new ArgumentNullException(nameof(body));
            if (cacheKey is null) throw new ArgumentNullException(nameof(cacheKey));

            return _innerLoopCache.GetOrAdd(cacheKey, _ =>
            {
                var dm = new DynamicMethod(
                    name: $"NDInnerLoop_Raw_{Sanitize(cacheKey)}",
                    returnType: typeof(void),
                    parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                    owner: typeof(DirectILKernelGenerator),
                    skipVisibility: true);

                body(dm.GetILGenerator());
                return dm.CreateDelegate<NDInnerLoopFunc>();
            });
        }

        #endregion

        #region Tier 3B: Templated inner loop (element-wise)

        /// <summary>
        /// Compile an element-wise inner-loop kernel. Operand layout:
        ///   operandTypes[0..N-1] are input operand dtypes
        ///   operandTypes[N]      is the output operand dtype
        ///
        /// <paramref name="scalarBody"/> runs once per element. On entry the
        /// evaluation stack holds the N input values (in order, already
        /// loaded via the operand's ldind); on exit it must hold exactly one
        /// value of the output dtype.
        ///
        /// <paramref name="vectorBody"/> is optional. When supplied AND all
        /// operands are SIMD-capable AND share the same element size, the
        /// factory emits a 4× unrolled SIMD loop using this body. On entry
        /// the stack holds N <c>Vector{W}&lt;T_i&gt;</c> values; on exit it
        /// must hold one <c>Vector{W}&lt;T_out&gt;</c>.
        ///
        /// The generated kernel also contains a scalar-strided fallback that
        /// runs when the iterator's inner axis is not contiguous for every
        /// operand.
        /// </summary>
        internal static NDInnerLoopFunc CompileInnerLoop(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            if (operandTypes is null) throw new ArgumentNullException(nameof(operandTypes));
            if (operandTypes.Length < 2)
                throw new ArgumentException("Need at least 1 input + 1 output operand.", nameof(operandTypes));
            if (scalarBody is null) throw new ArgumentNullException(nameof(scalarBody));
            if (cacheKey is null) throw new ArgumentNullException(nameof(cacheKey));

            return _innerLoopCache.GetOrAdd(cacheKey, _ =>
                GenerateTemplatedInnerLoop(operandTypes, scalarBody, vectorBody, cacheKey));
        }

        private static NDInnerLoopFunc GenerateTemplatedInnerLoop(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];

            var dm = new DynamicMethod(
                name: $"NDInnerLoop_{Sanitize(cacheKey)}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            // ---- Shared prologue: snapshot ptrs and strides into locals. ----
            var ptrLocals = new LocalBuilder[nOp];
            var strideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                strideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadInnerLoopArgs(il, nOp, ptrLocals, strideLocals);

            // ---- SIMD viability: all types SIMD-capable and same size. ----
            bool simdPossible = vectorBody != null && CanSimdAllOperands(operandTypes);

            // ---- Binary-broadcast SIMD viability (L3-d port). ----
            // Binary ops (one input with inner stride == 0 = broadcast,
            // other input + output contig) get a dedicated SIMD path that
            // pre-broadcasts the scalar via Vector.Create() outside the
            // loop, leaving the body as one SIMD load + op + store per
            // iteration. Same restrictions as the contig SIMD path: same
            // dtype across operands and SIMD-capable.
            bool simdBroadcastBinaryPossible = simdPossible && nIn == 2;

            // ---- Hardware-gather viability (Phase 2a). ----
            // Same-dtype SIMD kernels additionally get a strided fast path:
            // inputs are loaded with AVX2 vgather (byte-offset index vector
            // hoisted out of the loop — the hot loop is stride-agnostic), the
            // user's vectorBody runs unchanged, and the output is stored
            // contiguously (or scattered per-lane when the output is strided
            // too). This is NumPy's npyv_loadn technique, POC-proven 1.3-1.9x
            // FASTER than NumPy on strided shapes.
            GatherSupport gatherSupport = default;
            bool gatherPossible = simdPossible && TryGetGatherSupport(operandTypes[0], out gatherSupport);

            var lblScalarStrided = il.DefineLabel();
            var lblTryGather = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            // Where the contig/broadcast checks bail to: the gather dispatcher
            // when available, else straight to the scalar-strided fallback.
            var lblSimdBail = gatherPossible ? lblTryGather : lblScalarStrided;

            if (simdPossible)
            {
                int elemSize = GetTypeSize(operandTypes[0]); // all operands same dtype here

                if (simdBroadcastBinaryPossible)
                {
                    // 3-way runtime dispatch on (lhsStride, rhsStride) where
                    // outStride must always == elemSize for the SIMD-write path:
                    //
                    //   (e, e) → SIMD contig+contig (existing 4x unrolled)
                    //   (0, e) → SIMD scalar-lhs broadcast (Vector.Create(*lhs))
                    //   (e, 0) → SIMD scalar-rhs broadcast (Vector.Create(*rhs))
                    //   else   → scalar-strided fallback (always present)
                    //
                    // (0, 0) is theoretically possible but degenerate (one
                    // output element repeated) — caller normally short-circuits
                    // before reaching the kernel; we fall to scalar.
                    var lblSimdCC = il.DefineLabel();
                    var lblSimdSL = il.DefineLabel();
                    var lblSimdSR = il.DefineLabel();
                    var lblLhsIsZero = il.DefineLabel();

                    // Output stride must == elemSize for SIMD store; else try gather.
                    il.Emit(OpCodes.Ldloc, strideLocals[2]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblSimdBail);

                    // Branch on lhs stride: 0 → check rhs for SL/scalar; else
                    // check for == elemSize → check rhs for CC/SR; else scalar.
                    il.Emit(OpCodes.Ldloc, strideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Beq, lblLhsIsZero);

                    // lhs != 0; require lhs == elemSize for CC or SR.
                    il.Emit(OpCodes.Ldloc, strideLocals[0]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblSimdBail);

                    // lhs == elemSize; now check rhs.
                    il.Emit(OpCodes.Ldloc, strideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Beq, lblSimdCC);
                    il.Emit(OpCodes.Ldloc, strideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Beq, lblSimdSR);
                    il.Emit(OpCodes.Br, lblSimdBail);

                    // lhs == 0; require rhs == elemSize for SL.
                    il.MarkLabel(lblLhsIsZero);
                    il.Emit(OpCodes.Ldloc, strideLocals[1]);
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
                    il.Emit(OpCodes.Bne_Un, lblSimdBail);
                    il.Emit(OpCodes.Br, lblSimdSL);

                    // ── SIMD contig+contig (existing path) ──────────────────
                    il.MarkLabel(lblSimdCC);
                    EmitSimdContigLoop(il, operandTypes, ptrLocals, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);

                    // ── SIMD scalar-lhs (lhs broadcast, rhs contig) ─────────
                    il.MarkLabel(lblSimdSL);
                    EmitSimdBroadcastBinaryLoop(
                        il, operandTypes[0], ptrLocals, scalarSide: 0,
                        vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);

                    // ── SIMD scalar-rhs (lhs contig, rhs broadcast) ─────────
                    il.MarkLabel(lblSimdSR);
                    EmitSimdBroadcastBinaryLoop(
                        il, operandTypes[0], ptrLocals, scalarSide: 1,
                        vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);
                }
                else
                {
                    // Non-binary (unary, ternary, ...): the all-contig SIMD
                    // path. Any stride != elemSize → gather (when available)
                    // → scalar fallback.
                    for (int op = 0; op < nOp; op++)
                    {
                        int sz = GetTypeSize(operandTypes[op]);
                        il.Emit(OpCodes.Ldloc, strideLocals[op]);
                        il.Emit(OpCodes.Ldc_I8, (long)sz);
                        il.Emit(OpCodes.Bne_Un, lblSimdBail);
                    }
                    EmitSimdContigLoop(il, operandTypes, ptrLocals, vectorBody!, scalarBody);
                    il.Emit(OpCodes.Br, lblEnd);
                }

                // ── Hardware-gather dispatcher (Phase 2a) ───────────────────
                // Reached when the inner axis is genuinely strided. Guards:
                // every input's byte stride must fit the int32 lane-index
                // budget (|7·stride| ≤ int.MaxValue, covers negative and
                // zero strides); then pick contig-store (out == elemSize) or
                // per-lane scatter-store (out strided within budget).
                if (gatherPossible)
                {
                    il.MarkLabel(lblTryGather);

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
                    il.Emit(OpCodes.Ldc_I8, (long)elemSize);
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
                // No-SIMD path (mixed dtypes or non-SIMD-capable types like
                // Decimal / Half / Complex). Emit a runtime check "every
                // operand stride == its element size" → contig scalar loop;
                // else the strided fallback. The contig loop uses i*elemSize
                // addressing (elemSize is a JIT compile-time constant per
                // operand, often a power of 2 → folds to a shift) which gives
                // 30-40% over EmitScalarStridedLoop's per-operand multiply.
                //
                // This is the mixed-dtype version of the same trick the SIMD
                // branch already pulls — we just can't share a label set with
                // it because their stride checks compare against different
                // per-operand sizes.
                for (int op = 0; op < nOp; op++)
                {
                    int sz = GetTypeSize(operandTypes[op]);
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, (long)sz);
                    il.Emit(OpCodes.Bne_Un, lblScalarStrided);
                }
                EmitScalarContigLoop(il, operandTypes, ptrLocals, scalarBody);
                il.Emit(OpCodes.Br, lblEnd);
                il.MarkLabel(lblScalarStrided);
            }

            // Scalar strided fallback (always present).
            EmitScalarStridedLoop(il, operandTypes, ptrLocals, strideLocals, scalarBody);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<NDInnerLoopFunc>();
        }

        #endregion

        #region Emit helpers

        /// <summary>
        /// Emits the prologue that loads each operand's data pointer and byte
        /// stride into the supplied locals.
        /// </summary>
        private static void EmitLoadInnerLoopArgs(
            ILGenerator il, int nOp,
            LocalBuilder[] ptrLocals, LocalBuilder[] strideLocals)
        {
            // ptrLocals[op] = (byte*)dataptrs[op]
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldarg_0);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * IntPtr.Size);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }

            // strideLocals[op] = strides[op]  (bytes)
            for (int op = 0; op < nOp; op++)
            {
                il.Emit(OpCodes.Ldarg_1);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * sizeof(long));
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Stloc, strideLocals[op]);
            }
        }

        /// <summary>
        /// All operands must be SIMD-capable AND share the same dtype for the
        /// templated SIMD path — the shell loads every operand through the
        /// same Vector{W}&lt;T&gt; instantiation. Mixed-type SIMD (e.g.
        /// int32+float32) is too ambiguous for a generic shell; users needing
        /// that should either call CompileRawInnerLoop (Tier 3A) with their
        /// own mixed-type IL, or accept the scalar fallback where the body
        /// handles conversion.
        /// </summary>
        private static bool CanSimdAllOperands(NPTypeCode[] types)
        {
            if (VectorBits == 0) return false;
            NPTypeCode first = types[0];
            if (!CanUseSimd(first)) return false;
            for (int i = 1; i < types.Length; i++)
                if (types[i] != first) return false;
            return true;
        }

        /// <summary>
        /// Emit the 4× unrolled SIMD loop + 1-vector remainder + scalar tail
        /// for the contiguous inner-loop fast path. Matches the shape of
        /// <c>EmitSimdFullLoop</c> in MixedType.cs but targets the
        /// NDInnerLoopFunc signature.
        /// </summary>
        private static void EmitSimdContigLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long vectorCount = GetVectorCount(outType);
            long unrollStep = vectorCount * 4;

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // === 4× UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int u = 0; u < 4; u++)
            {
                long offset = u * vectorCount;

                // Load N input vectors at (i + offset) * elemSize.
                for (int op = 0; op < nIn; op++)
                {
                    EmitAddrIPlusOffset(il, ptrLocals[op], locI, offset, elemSize);
                    EmitVectorLoad(il, operandTypes[op]);
                }

                // User vector body: stack[in0..inN-1] -> stack[out_vec]
                vectorBody(il);

                // Store(source_vec, dest_ptr) wants [vec, ptr] on stack.
                // We already have [out_vec]; push dest_ptr on top.
                EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, offset, elemSize);
                EmitVectorStore(il, outType);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === REMAINDER SIMD LOOP (0..3 vectors) ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            for (int op = 0; op < nIn; op++)
            {
                EmitAddrIPlusOffset(il, ptrLocals[op], locI, 0, elemSize);
                EmitVectorLoad(il, operandTypes[op]);
            }
            vectorBody(il);

            // Stack: [out_vec]; push dest_ptr to make [vec, ptr] for Store.
            EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, 0, elemSize);
            EmitVectorStore(il, outType);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL (contiguous) ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblTailEnd);

            EmitScalarElement(il, operandTypes, ptrLocals, /*stridesInElems*/ null, locI, contig: true, scalarBody);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// L3-d port: SIMD inner loop where ONE binary-op input is scalar-
        /// broadcast on the inner axis (stride==0) and the other input + the
        /// output are inner-contig (stride==elemSize). Pre-broadcasts the
        /// scalar value via <c>Vector.Create(*scalarPtr)</c> once outside
        /// the loop; the per-iteration body collapses to one SIMD load +
        /// op + store against the non-scalar operand.
        ///
        /// Stack convention to <paramref name="vectorBody"/> stays
        /// <c>[v_lhs, v_rhs]</c> regardless of which side is the scalar —
        /// the helper pushes them in the right order based on
        /// <paramref name="scalarSide"/>. Same for the scalar tail body
        /// (<c>[scalar_lhs, scalar_rhs]</c>).
        ///
        /// Assumes:
        ///   - operandTypes[0] == operandTypes[1] == operandTypes[2] == dtype
        ///   - dtype is SIMD-capable for the user's vectorBody
        ///   - lhs/rhs ptrLocals point at the row's element-0 address
        ///     (NDIter has already advanced them to the outer-iter start)
        ///   - scalarSide ∈ {0, 1}: 0 = lhs broadcast, 1 = rhs broadcast
        ///   - 4× unroll mirrors EmitSimdContigLoop for consistent pipelining
        /// </summary>
        private static void EmitSimdBroadcastBinaryLoop(
            ILGenerator il,
            NPTypeCode dtype,
            LocalBuilder[] ptrLocals,
            int scalarSide,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody)
        {
            int elemSize = GetTypeSize(dtype);
            long vectorCount = GetVectorCount(dtype);
            long unrollStep = vectorCount * 4;

            var clrType = GetClrType(dtype);
            var vecType = VectorMethodCache.V(VectorBits, clrType);

            LocalBuilder scalarPtr = ptrLocals[scalarSide];
            LocalBuilder contigPtr = ptrLocals[1 - scalarSide];
            LocalBuilder outPtr = ptrLocals[2];

            var locVScalar = il.DeclareLocal(vecType);
            var locScalarVal = il.DeclareLocal(clrType);

            // ── Pre-load + pre-broadcast the scalar once. ───────────────────
            // scalarVal = *scalarPtr;
            il.Emit(OpCodes.Ldloc, scalarPtr);
            EmitLoadIndirect(il, dtype);
            il.Emit(OpCodes.Dup);
            il.Emit(OpCodes.Stloc, locScalarVal);  // save for scalar tail
            // vScalar = Vector.Create(scalarVal);
            EmitVectorCreate(il, dtype);
            il.Emit(OpCodes.Stloc, locVScalar);

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            // === 4× UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int u = 0; u < 4; u++)
            {
                long offset = u * vectorCount;
                EmitBroadcastVectorBody(
                    il, dtype, contigPtr, locI, offset, elemSize,
                    locVScalar, scalarSide, vectorBody);

                // Store output at (i + offset) * elemSize.
                EmitAddrIPlusOffset(il, outPtr, locI, offset, elemSize);
                EmitVectorStore(il, dtype);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === REMAINDER SIMD LOOP (1 vector at a time) ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            EmitBroadcastVectorBody(
                il, dtype, contigPtr, locI, 0, elemSize,
                locVScalar, scalarSide, vectorBody);
            EmitAddrIPlusOffset(il, outPtr, locI, 0, elemSize);
            EmitVectorStore(il, dtype);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblTailEnd);

            // Push [lhs_val, rhs_val] in original LHS-RHS order regardless of
            // which side is the scalar.
            if (scalarSide == 0)
            {
                il.Emit(OpCodes.Ldloc, locScalarVal);
                EmitAddrIPlusOffset(il, contigPtr, locI, 0, elemSize);
                EmitLoadIndirect(il, dtype);
            }
            else
            {
                EmitAddrIPlusOffset(il, contigPtr, locI, 0, elemSize);
                EmitLoadIndirect(il, dtype);
                il.Emit(OpCodes.Ldloc, locScalarVal);
            }

            scalarBody(il);

            // Store result. Stack has [outVal]; reorder to [outAddr, outVal].
            var locOutVal = il.DeclareLocal(clrType);
            il.Emit(OpCodes.Stloc, locOutVal);
            EmitAddrIPlusOffset(il, outPtr, locI, 0, elemSize);
            il.Emit(OpCodes.Ldloc, locOutVal);
            EmitStoreIndirect(il, dtype);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// Push [v_lhs, v_rhs] onto the stack for the user's vector body in
        /// original LHS-RHS order, regardless of which side carries the
        /// pre-broadcast scalar. Then invoke vectorBody (which consumes both
        /// and produces one result vector). Result vector is left on the stack.
        /// </summary>
        private static void EmitBroadcastVectorBody(
            ILGenerator il, NPTypeCode dtype,
            LocalBuilder contigPtr, LocalBuilder locI, long offset, int elemSize,
            LocalBuilder locVScalar, int scalarSide,
            Action<ILGenerator> vectorBody)
        {
            if (scalarSide == 0)
            {
                il.Emit(OpCodes.Ldloc, locVScalar);
                EmitAddrIPlusOffset(il, contigPtr, locI, offset, elemSize);
                EmitVectorLoad(il, dtype);
            }
            else
            {
                EmitAddrIPlusOffset(il, contigPtr, locI, offset, elemSize);
                EmitVectorLoad(il, dtype);
                il.Emit(OpCodes.Ldloc, locVScalar);
            }
            vectorBody(il);
        }

        /// <summary>
        /// Emit a pure scalar contig loop. Each operand walks at its element-
        /// size step (no per-iter stride multiply — the per-operand elemSize
        /// is baked in as a constant the JIT can fold to a shift for power-
        /// of-2 sizes). Used for the mixed-dtype contig fast path when SIMD
        /// is unavailable; matches the perf of the direct path's
        /// EmitChunkLoop scalar inner walk.
        /// </summary>
        private static void EmitScalarContigLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            Action<ILGenerator> scalarBody)
        {
            var locI = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            il.MarkLabel(lblLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblLoopEnd);

            EmitScalarElement(il, operandTypes, ptrLocals, /*stridesInElems*/ null, locI, contig: true, scalarBody);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit a pure scalar strided loop. Each operand advances by its own
        /// byte stride per iteration. Used as fallback when the contig check
        /// fails OR when no vector body was supplied / types not SIMD-able.
        /// </summary>
        private static void EmitScalarStridedLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            Action<ILGenerator> scalarBody)
        {
            var locI = il.DeclareLocal(typeof(long));
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();
            il.MarkLabel(lblLoop);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblLoopEnd);

            EmitScalarElement(il, operandTypes, ptrLocals, strideLocals, locI, contig: false, scalarBody);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);
        }

        /// <summary>
        /// Emit: load N input scalars, call scalarBody, store one output.
        /// When <paramref name="contig"/> is true, addresses are computed as
        /// ptr + i*elemSize; otherwise as ptr + i*strideBytes.
        /// </summary>
        private static void EmitScalarElement(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[]? strideLocals,
            LocalBuilder locI,
            bool contig,
            Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];

            // Load N input values onto stack.
            for (int op = 0; op < nIn; op++)
            {
                if (contig)
                    EmitAddrIPlusOffset(il, ptrLocals[op], locI, 0, GetTypeSize(operandTypes[op]));
                else
                    EmitAddrIStrided(il, ptrLocals[op], locI, strideLocals![op]);
                EmitLoadIndirect(il, operandTypes[op]);
            }

            // User scalar body: stack[val0..valN-1] -> stack[valOut]
            scalarBody(il);

            // Store result. Need [outAddr, valOut] on stack; currently [valOut].
            var locOutVal = il.DeclareLocal(GetClrType(outType));
            il.Emit(OpCodes.Stloc, locOutVal);

            if (contig)
                EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, 0, GetTypeSize(outType));
            else
                EmitAddrIStrided(il, ptrLocals[nIn], locI, strideLocals![nIn]);

            il.Emit(OpCodes.Ldloc, locOutVal);
            EmitStoreIndirect(il, outType);
        }

        /// <summary>
        /// Push: basePtr + (i + offset) * elemSize.
        /// </summary>
        private static void EmitAddrIPlusOffset(
            ILGenerator il, LocalBuilder basePtr, LocalBuilder locI, long offset, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary>
        /// Push: basePtr + i * strideBytes.
        /// </summary>
        private static void EmitAddrIStrided(
            ILGenerator il, LocalBuilder basePtr, LocalBuilder locI, LocalBuilder strideBytes)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, strideBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        // =====================================================================
        // Hardware-gather strided SIMD (Phase 2a)
        // =====================================================================

        /// <summary>
        /// Byte-stride budget for vgather int32 lane indices: the largest lane
        /// offset is 7·stride (8-lane) so |stride| must stay ≤ int.MaxValue/8.
        /// Used uniformly for 4-lane dtypes too (slightly conservative).
        /// </summary>
        internal const long GatherStrideLimit = int.MaxValue / 8;

        /// <summary>Reflection handles for one dtype's gather emission.</summary>
        internal readonly struct GatherSupport
        {
            /// <summary>SIMD lanes per vector (8 for 32-bit dtypes, 4 for 64-bit at V256).</summary>
            public readonly int Lanes;

            /// <summary>Index vector CLR type (Vector256&lt;int&gt; or Vector128&lt;int&gt;).</summary>
            public readonly Type IndexVectorType;

            /// <summary><c>Vector{128|256}.Create(int×lanes)</c>.</summary>
            public readonly MethodInfo IndexCreate;

            /// <summary><c>Avx2.GatherVector256(T*, idx, scale)</c>.</summary>
            public readonly MethodInfo Gather;

            /// <summary><c>Vector256.GetElement&lt;T&gt;(vec, i)</c> for scatter stores.</summary>
            public readonly MethodInfo GetElement;

            public GatherSupport(int lanes, Type indexVectorType, MethodInfo indexCreate, MethodInfo gather, MethodInfo getElement)
            {
                Lanes = lanes;
                IndexVectorType = indexVectorType;
                IndexCreate = indexCreate;
                Gather = gather;
                GetElement = getElement;
            }
        }

        /// <summary>
        /// Whether <paramref name="dtype"/> can use the AVX2 hardware-gather
        /// strided path on this machine. Requires the baked SIMD width to be
        /// 256 (the gather output must match the Vector{W} the vectorBody
        /// operates on) and a gather-capable element width (32/64-bit) — the
        /// same dtype set NumPy's npyv_loadn gathers. Other dtypes/widths keep
        /// the scalar-strided fallback.
        /// </summary>
        internal static bool TryGetGatherSupport(NPTypeCode dtype, out GatherSupport support)
        {
            support = default;
            if (VectorBits != 256 || !Avx2.IsSupported)
                return false;

            Type clrType;
            switch (dtype)
            {
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Single:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Double:
                    clrType = GetClrType(dtype);
                    break;
                default:
                    return false;
            }

            bool is32 = GetTypeSize(dtype) == 4;
            int lanes = is32 ? 8 : 4;
            Type idxVecType = is32 ? typeof(Vector256<int>) : typeof(Vector128<int>);

            var idxArgTypes = new Type[lanes];
            for (int i = 0; i < lanes; i++) idxArgTypes[i] = typeof(int);
            MethodInfo idxCreate = (is32 ? typeof(Vector256) : typeof(Vector128))
                .GetMethod("Create", idxArgTypes)
                ?? throw new InvalidOperationException($"Vector{(is32 ? 256 : 128)}.Create(int×{lanes}) not found");

            MethodInfo gather = typeof(Avx2).GetMethod(
                    "GatherVector256",
                    new[] { clrType.MakePointerType(), idxVecType, typeof(byte) })
                ?? throw new InvalidOperationException($"Avx2.GatherVector256({clrType.Name}*) not found");

            MethodInfo getElement = typeof(Vector256).GetMethod("GetElement")!.MakeGenericMethod(clrType);

            support = new GatherSupport(lanes, idxVecType, idxCreate, gather, getElement);
            return true;
        }

        /// <summary>
        /// Push: basePtr + (i + constOffset) * strideBytes.
        /// </summary>
        private static void EmitAddrIStridedOffset(
            ILGenerator il, LocalBuilder basePtr, LocalBuilder locI, long constOffset, LocalBuilder strideBytes)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locI);
            if (constOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, constOffset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldloc, strideBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary>
        /// Emit the strided SIMD loop: per-input AVX2 hardware gather (byte-
        /// offset index vector built ONCE from the runtime stride — the hot
        /// loop advances only the base address), the caller's vectorBody, and
        /// either a contiguous vector store or a per-lane scatter store.
        /// Structure mirrors <see cref="EmitSimdContigLoop"/>: 4× unrolled +
        /// 1-vector remainder + scalar-strided tail.
        ///
        /// Stride-0 (broadcast) inputs are valid here — the index vector is
        /// all-zero and the gather replicates the element; the dedicated
        /// SL/SR broadcast paths catch the common binary cases first.
        /// </summary>
        private static void EmitSimdGatherLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody,
            GatherSupport g,
            bool scatterOut)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode dtype = operandTypes[0]; // same dtype across operands (CanSimdAllOperands)
            int elemSize = GetTypeSize(dtype);
            long lanes = g.Lanes;
            long unrollStep = lanes * 4;

            // ── Hoist per-input index vectors: Create(0, s, 2s, …, (lanes-1)s). ──
            var idxLocals = new LocalBuilder[nIn];
            var locSInt = il.DeclareLocal(typeof(int));
            for (int op = 0; op < nIn; op++)
            {
                idxLocals[op] = il.DeclareLocal(g.IndexVectorType);
                il.Emit(OpCodes.Ldloc, strideLocals[op]);
                il.Emit(OpCodes.Conv_I4); // guarded: |stride| ≤ GatherStrideLimit
                il.Emit(OpCodes.Stloc, locSInt);
                for (int k = 0; k < g.Lanes; k++)
                {
                    if (k == 0)
                    {
                        il.Emit(OpCodes.Ldc_I4_0);
                    }
                    else if (k == 1)
                    {
                        il.Emit(OpCodes.Ldloc, locSInt);
                    }
                    else
                    {
                        il.Emit(OpCodes.Ldloc, locSInt);
                        il.Emit(OpCodes.Ldc_I4, k);
                        il.Emit(OpCodes.Mul);
                    }
                }
                il.Emit(OpCodes.Call, g.IndexCreate);
                il.Emit(OpCodes.Stloc, idxLocals[op]);
            }

            var locI = il.DeclareLocal(typeof(long));
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            LocalBuilder? locOutVec = scatterOut ? il.DeclareLocal(GetVectorType(GetClrType(dtype))) : null;

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

            // === 4× UNROLLED GATHER LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);

            for (int u = 0; u < 4; u++)
            {
                long offset = u * lanes;
                EmitGatherVectorGroup(il, operandTypes, ptrLocals, strideLocals, idxLocals,
                    locI, offset, elemSize, g, vectorBody, scatterOut, locOutVec);
            }

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === REMAINDER (single vector) ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);

            EmitGatherVectorGroup(il, operandTypes, ptrLocals, strideLocals, idxLocals,
                locI, 0, elemSize, g, vectorBody, scatterOut, locOutVec);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, lanes);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL (strided) ===
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

        /// <summary>
        /// One vector group of the gather loop: gather N input vectors at
        /// element position (i + offset), run vectorBody, store the result —
        /// contiguous vector store, or per-lane extract+store when the output
        /// is strided.
        /// </summary>
        private static void EmitGatherVectorGroup(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            LocalBuilder[] idxLocals,
            LocalBuilder locI, long offset, int elemSize,
            GatherSupport g,
            Action<ILGenerator> vectorBody,
            bool scatterOut, LocalBuilder? locOutVec)
        {
            int nIn = operandTypes.Length - 1;
            NPTypeCode outType = operandTypes[nIn];

            for (int op = 0; op < nIn; op++)
            {
                // base = ptr + (i + offset) * stride  (byte*)
                EmitAddrIStridedOffset(il, ptrLocals[op], locI, offset, strideLocals[op]);
                il.Emit(OpCodes.Ldloc, idxLocals[op]);
                il.Emit(OpCodes.Ldc_I4_1); // scale = 1: indices are byte offsets
                il.Emit(OpCodes.Call, g.Gather);
            }

            vectorBody(il);

            if (!scatterOut)
            {
                // Store(vec, ptr): stack already [vec]; push dest address.
                EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, offset, elemSize);
                EmitVectorStore(il, outType);
            }
            else
            {
                // Per-lane scatter: NumPy's npyv_storen shape (no AVX2 scatter
                // instruction exists; compute is vectorized, stores are scalar).
                il.Emit(OpCodes.Stloc, locOutVec!);
                for (int k = 0; k < g.Lanes; k++)
                {
                    EmitAddrIStridedOffset(il, ptrLocals[nIn], locI, offset + k, strideLocals[nIn]);
                    il.Emit(OpCodes.Ldloc, locOutVec!);
                    il.Emit(OpCodes.Ldc_I4, k);
                    il.Emit(OpCodes.Call, g.GetElement);
                    EmitStoreIndirect(il, outType);
                }
            }
        }

        private static string Sanitize(string key)
        {
            Span<char> buf = stackalloc char[Math.Min(key.Length, 64)];
            int n = 0;
            for (int i = 0; i < key.Length && n < buf.Length; i++)
            {
                char c = key[i];
                buf[n++] = (char.IsLetterOrDigit(c) || c == '_') ? c : '_';
            }
            return new string(buf[..n]);
        }

        #endregion
    }
}
