using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
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
// BROADCAST INPUTS (stride 0)
// ---------------------------
// A binary kernel has dedicated scalar-left / scalar-right loops. Three or more
// inputs dispatch on their C/S pattern (C = contiguous, S = stride 0): the all-C
// loop inline, every other pattern a stage of its own whose S inputs are read once
// and Vector.Create'd; the gather route does the same for strided inputs next to
// broadcast ones (two or more inputs). See
// DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs.
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

        /// <summary>
        /// Struct-keyed FRONT cache over <see cref="_innerLoopCache"/> for the production
        /// ufunc routes. Those routes used to build their string key with an interpolated
        /// <c>$"npy_binop_{op}_{lhs}_{rhs}_{res}"</c> on EVERY call — four enum formats plus a
        /// string allocation, measured 45 ns + 96 B of garbage per call against ~10 ns for a
        /// lookup on a prebuilt key — a fixed tax on every NDIter-routed op that dwarfs what
        /// the iterator itself costs at small sizes. A hit here never touches a string; a
        /// miss builds the SAME string (<see cref="InnerLoopKernelKey.ToCacheKey"/>) and
        /// goes through <see cref="_innerLoopCache"/>, so DynamicMethod names, kernel
        /// identity and <c>GeneratedDelegates.InnerLoopCount</c> are unchanged.
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, NDInnerLoopFunc> _innerLoopKeyCache = new();

        // Cache size + reset are centralized on GeneratedDelegates
        // (GeneratedDelegates.InnerLoopCount / GeneratedDelegates.ClearInnerLoop).

        /// <summary>
        /// Look up a production inner-loop kernel by its packed key without building a string.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryGetInnerLoop(in InnerLoopKernelKey key, out NDInnerLoopFunc kernel)
            => _innerLoopKeyCache.TryGetValue(key, out kernel!);

        /// <summary>
        /// Struct-keyed <see cref="CompileInnerLoop(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/>:
        /// serves the kernel from the front cache, else compiles it under the equivalent
        /// string key and registers it in both caches.
        /// </summary>
        internal static NDInnerLoopFunc CompileInnerLoop(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            in InnerLoopKernelKey key)
        {
            if (_innerLoopKeyCache.TryGetValue(key, out var cached))
                return cached;

            // A binary-ufunc key names its op, which is what tells the shell that a float add/multiply needs NumPy's
            // broadcast-NaN region rule (HasNaNOperandOrder). The "npy_binop_*" string keys are minted ONLY here, so the
            // string cache can never hold a kernel of the same name compiled without the rule.
            BinaryOp? nanOrderedOp = key.Family == InnerLoopKernelKey.KernelFamily.Binary ? (BinaryOp)key.Op : null;
            var kernel = CompileInnerLoop(operandTypes, scalarBody, vectorBody, key.ToCacheKey(), nanOrderedOp);
            _innerLoopKeyCache.TryAdd(key, kernel);
            return kernel;
        }

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
            => CompileInnerLoop(operandTypes, scalarBody, vectorBody, cacheKey, nanOrderedOp: null);

        /// <summary>
        ///     <see cref="CompileInnerLoop(NPTypeCode[], Action{ILGenerator}, Action{ILGenerator}?, string)"/> that also knows
        ///     the binary op the bodies implement, so a float add/multiply gets NumPy's broadcast-NaN region rule.
        /// </summary>
        /// <param name="operandTypes">Input dtypes followed by the output dtype.</param>
        /// <param name="scalarBody">Per-element body (see the 4-argument overload).</param>
        /// <param name="vectorBody">Optional per-vector body (see the 4-argument overload).</param>
        /// <param name="cacheKey">The kernel's cache key; a kernel compiled under it is reused as-is.</param>
        /// <param name="nanOrderedOp">
        ///     The binary op of a 2-input ufunc kernel, or <see langword="null"/>. When it is a float add/multiply
        ///     (<see cref="HasNaNOperandOrder"/>), every call whose stride-0 input holds a NaN runs
        ///     <see cref="EmitBroadcastNaNRegion"/> with the <see cref="BroadcastNaNRule"/> passed as <c>auxdata</c>, and the
        ///     broadcast SIMD loops keep the plain instruction (the value they broadcast is then never NaN).
        /// </param>
        /// <returns>The cached or newly compiled kernel.</returns>
        /// <exception cref="ArgumentNullException">A required argument is null.</exception>
        /// <exception cref="ArgumentException">Fewer than two operand types.</exception>
        internal static NDInnerLoopFunc CompileInnerLoop(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey,
            BinaryOp? nanOrderedOp)
        {
            if (operandTypes is null) throw new ArgumentNullException(nameof(operandTypes));
            if (operandTypes.Length < 2)
                throw new ArgumentException("Need at least 1 input + 1 output operand.", nameof(operandTypes));
            if (scalarBody is null) throw new ArgumentNullException(nameof(scalarBody));
            if (cacheKey is null) throw new ArgumentNullException(nameof(cacheKey));

            return _innerLoopCache.GetOrAdd(cacheKey, _ =>
                GenerateTemplatedInnerLoop(operandTypes, scalarBody, vectorBody, cacheKey, nanOrderedOp));
        }

        private static NDInnerLoopFunc GenerateTemplatedInnerLoop(
            NPTypeCode[] operandTypes,
            Action<ILGenerator> scalarBody,
            Action<ILGenerator>? vectorBody,
            string cacheKey,
            BinaryOp? nanOrderedOp)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];

            // A 2-input float add/multiply whose stride-0 input can be NaN: that case runs NumPy's region rule
            // (EmitBroadcastNaNRegion), everything else the supplied bodies.
            bool nanOrdered = nanOrderedOp.HasValue && nIn == 2 && HasNaNOperandOrder(nanOrderedOp.Value, outType);

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
                        vectorBody!, scalarBody, nanOrdered ? nanOrderedOp : null, lblEnd);
                    il.Emit(OpCodes.Br, lblEnd);

                    // ── SIMD scalar-rhs (lhs contig, rhs broadcast) ─────────
                    il.MarkLabel(lblSimdSR);
                    EmitSimdBroadcastBinaryLoop(
                        il, operandTypes[0], ptrLocals, scalarSide: 1,
                        vectorBody!, scalarBody, nanOrdered ? nanOrderedOp : null, lblEnd);
                    il.Emit(OpCodes.Br, lblEnd);
                }
                else if (nIn >= 3 && nIn <= MaxBroadcastPatternInputs)
                {
                    // Ternary and wider: every input contiguous (C) or broadcast along the loop
                    // (S, stride 0) — pattern 0 is the inline all-contig loop, each other C/S
                    // pattern a stage of its own (InnerLoop.BroadcastPatterns.cs); a genuinely
                    // strided input or output → gather (when available) → scalar fallback.
                    EmitBroadcastPatternDispatch1D(il, operandTypes, ptrLocals, strideLocals,
                        vectorBody!, scalarBody, cacheKey, lblSimdBail, lblEnd);
                }
                else
                {
                    // Unary (and inputs past MaxBroadcastPatternInputs): the all-contig SIMD
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

                    // A broadcast (stride-0) input next to the strided ones is read once and broadcast instead of
                    // gathered: its pattern's stage takes the call; with none broadcast control falls through.
                    if (nIn >= 2 && nIn <= MaxBroadcastPatternInputs)
                        EmitGatherBroadcastPatternDispatch(il, operandTypes, strideLocals, vectorBody!, scalarBody,
                            gatherSupport, cacheKey, lblEnd);

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

                // A mixed-dtype float add/multiply with a stride-0 input (a STRONG 0-d of another dtype): NumPy buffers
                // the cast operand, so its calls split per BroadcastNaNRule.BufferChunk — a NaN broadcast value takes
                // the region rule, everything else the strided loop below.
                if (nanOrdered)
                    EmitStridedBroadcastNaNDispatch(il, operandTypes, ptrLocals, strideLocals, lblEnd);
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
        internal static bool CanSimdAllOperands(NPTypeCode[] types)
        {
            if (VectorBits == 0) return false;
            NPTypeCode first = types[0];
            // Boolean is admitted here even though CanUseSimd(Boolean) is false: bool loads/stores
            // ride the byte lanes (GetSimdLaneType), and a bool vectorBody only ever EXISTS when
            // the engine's op-aware gate (CanUseSimdBinary / CanUseUnarySimd) already approved the
            // (op, bool) pairing — this shell activates a supplied body, it never invents one.
            if (!CanUseSimd(first) && first != NPTypeCode.Boolean) return false;
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
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output; one shared SIMD dtype.</param>
        /// <param name="ptrLocals">Per-operand data pointers (inputs then the output).</param>
        /// <param name="vectorBody">The op's vector body: N input vectors → one output vector.</param>
        /// <param name="scalarBody">The op's scalar body (the tail): N input values → one output value.</param>
        /// <param name="isScalar">
        /// Null (every input contiguous — the shape this loop always had), or per input whether it is
        /// BROADCAST along the loop (inner byte stride 0). A broadcast input is loaded ONCE before the
        /// loop, kept as a scalar for the tail and <c>Vector.Create</c>d for every vector group — the
        /// N-ary form of <see cref="EmitSimdBroadcastBinaryLoop"/>'s hoist, used by the broadcast-pattern
        /// stages (DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs). The output is always contiguous.
        /// </param>
        /// <remarks>
        /// A broadcast input's vector is <c>Vector.Create</c> of the very element every lane would have
        /// read through a stride-0 pointer, so the result is byte-identical to the scalar strided route.
        /// The hoisted load runs even for a zero-length call; the iterator never hands a kernel a
        /// dangling operand pointer (a broadcast operand points at its one element), so it is safe.
        /// </remarks>
        private static void EmitSimdContigLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody,
            bool[]? isScalar = null)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long vectorCount = GetVectorCount(outType);
            long unrollStep = vectorCount * 4;

            // Broadcast inputs: their one value, read once, as a scalar (the tail) and as a vector
            // (every group). Null entries are the contiguous inputs, loaded per group as before.
            LocalBuilder?[]? locVScalar = null;
            LocalBuilder?[]? locScalarVal = null;
            if (isScalar != null)
            {
                // CanSimdAllOperands guarantees one shared dtype, so the output's lane vector type is
                // every input's (Boolean rides byte lanes — GetSimdLaneType).
                var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(outType));
                locVScalar = new LocalBuilder?[nIn];
                locScalarVal = new LocalBuilder?[nIn];
                for (int op = 0; op < nIn; op++)
                {
                    if (!isScalar[op]) continue;
                    locVScalar[op] = il.DeclareLocal(vecType);
                    locScalarVal[op] = il.DeclareLocal(GetClrType(operandTypes[op]));
                    // scalarVal = *ptr; vScalar = Vector.Create(scalarVal)
                    il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                    EmitLoadIndirect(il, operandTypes[op]);
                    il.Emit(OpCodes.Dup);
                    il.Emit(OpCodes.Stloc, locScalarVal[op]!);
                    EmitVectorCreate(il, operandTypes[op]);
                    il.Emit(OpCodes.Stloc, locVScalar[op]!);
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

                // Load N input vectors at (i + offset) * elemSize (a broadcast input pushes its
                // hoisted vector instead).
                for (int op = 0; op < nIn; op++)
                {
                    if (locVScalar?[op] is { } hoisted)
                    {
                        il.Emit(OpCodes.Ldloc, hoisted);
                        continue;
                    }
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
                if (locVScalar?[op] is { } hoisted)
                {
                    il.Emit(OpCodes.Ldloc, hoisted);
                    continue;
                }
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

            if (locScalarVal == null)
                EmitScalarElement(il, operandTypes, ptrLocals, /*stridesInElems*/ null, locI, contig: true, scalarBody);
            else
                EmitScalarElementWithBroadcast(il, operandTypes, ptrLocals, locScalarVal, locI, scalarBody);

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
        ///
        /// For a float add/multiply (<paramref name="nanOrderedOp"/> set) a NaN broadcast value branches to
        /// <see cref="EmitBroadcastNaNRegion"/> (NumPy's call model from the <see cref="BroadcastNaNRule"/> in
        /// <c>auxdata</c>), and the loops run the PLAIN instruction instead of the supplied bodies: with a non-NaN
        /// broadcast value only the array element can be NaN, so the operand order cannot change any result and the
        /// order-pinning compare would be wasted work.
        /// </summary>
        /// <param name="il">The kernel's IL stream (the <see cref="NDInnerLoopFunc"/> signature).</param>
        /// <param name="dtype">The shared dtype of both inputs and the output.</param>
        /// <param name="ptrLocals">Per-operand data-pointer locals (input 0, input 1, output).</param>
        /// <param name="scalarSide">0 when input 0 is the broadcast operand, 1 when input 1 is.</param>
        /// <param name="vectorBody">The caller's vector body (used when <paramref name="nanOrderedOp"/> is null).</param>
        /// <param name="scalarBody">The caller's scalar body (used when <paramref name="nanOrderedOp"/> is null).</param>
        /// <param name="nanOrderedOp">The float add/multiply this kernel implements, or <see langword="null"/>.</param>
        /// <param name="lblEnd">The kernel's exit label (the NaN path branches there).</param>
        private static void EmitSimdBroadcastBinaryLoop(
            ILGenerator il,
            NPTypeCode dtype,
            LocalBuilder[] ptrLocals,
            int scalarSide,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody,
            BinaryOp? nanOrderedOp,
            Label lblEnd)
        {
            int elemSize = GetTypeSize(dtype);
            long vectorCount = GetVectorCount(dtype);
            long unrollStep = vectorCount * 4;

            var clrType = GetClrType(dtype);
            // Vector LOCALS take the lane type (Boolean -> byte; Vector256<bool> is not a type);
            // the scalar-tail local keeps the CLR type.
            var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(dtype));

            LocalBuilder scalarPtr = ptrLocals[scalarSide];
            LocalBuilder contigPtr = ptrLocals[1 - scalarSide];
            LocalBuilder outPtr = ptrLocals[2];

            var locVScalar = il.DeclareLocal(vecType);
            var locScalarVal = il.DeclareLocal(clrType);
            var lblNaNScalar = il.DefineLabel();

            if (nanOrderedOp is BinaryOp op)
            {
                // Non-NaN broadcast value ⇒ order-independent results ⇒ the plain instruction suffices.
                vectorBody = g => EmitVectorOperation(g, op, dtype);
                scalarBody = g => EmitScalarOperation(g, op, dtype);
            }

            // ── Pre-load + pre-broadcast the scalar once. ───────────────────
            // scalarVal = *scalarPtr;
            il.Emit(OpCodes.Ldloc, scalarPtr);
            EmitLoadIndirect(il, dtype);
            il.Emit(OpCodes.Stloc, locScalarVal);  // save for scalar tail (and the NaN test)
            if (nanOrderedOp.HasValue)
                EmitBranchIfNaN(il, locScalarVal, lblNaNScalar);
            // vScalar = Vector.Create(scalarVal);
            il.Emit(OpCodes.Ldloc, locScalarVal);
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

            if (nanOrderedOp.HasValue)
            {
                il.Emit(OpCodes.Br, lblEnd);
                il.MarkLabel(lblNaNScalar);
                // Both the array input and the output are element-contiguous here (the shell's stride dispatch).
                EmitBroadcastNaNRegion(il, dtype, locScalarVal,
                    contigPtr, dtype, arrayStride: null, outPtr, outStride: null,
                    pushCount: g => g.Emit(OpCodes.Ldarg_2),
                    scalarIsSecond: scalarSide == 1,
                    pushSplitPerCall: EmitLoadRuleSplitPerCall,
                    pushBufferChunk: EmitLoadRuleBufferChunk,
                    contiguous: true, lblEnd);
            }
        }

        /// <summary>
        ///     The cold dispatch of a MIXED-dtype float add/multiply inner loop (no SIMD) at its strided entry: when exactly
        ///     one input has stride 0 and its value, converted to the loop dtype, is NaN, runs
        ///     <see cref="EmitBroadcastNaNRegion"/> over the strided operands and branches to <paramref name="lblEnd"/>;
        ///     otherwise falls through to the strided loop the caller emits next.
        /// </summary>
        /// <remarks>
        ///     NumPy buffers the cast operand of such a call, so its own inner loop sees contiguous buffers and splits the
        ///     call into <see cref="BroadcastNaNRule.BufferChunk"/>-sized pieces; the dispatcher only sets
        ///     <see cref="BroadcastNaNRule.SplitPerCall"/> when that model holds, and a null rule keeps the second-operand rule.
        /// </remarks>
        /// <param name="il">The kernel's IL stream.</param>
        /// <param name="operandTypes">Input 0, input 1 and output dtypes.</param>
        /// <param name="ptrLocals">Per-operand data-pointer locals.</param>
        /// <param name="strideLocals">Per-operand byte-stride locals.</param>
        /// <param name="lblEnd">The kernel's exit label.</param>
        private static void EmitStridedBroadcastNaNDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] strideLocals, Label lblEnd)
        {
            NPTypeCode outType = operandTypes[2];
            var lblCheckRight = il.DefineLabel();
            var lblFallThrough = il.DefineLabel();

            for (int side = 0; side < 2; side++)
            {
                if (side == 1)
                    il.MarkLabel(lblCheckRight);

                // This side broadcast (stride 0) and the other not (two stride-0 inputs = one repeated element; NumPy
                // takes its scalar loop there, which the strided loop already reproduces).
                il.Emit(OpCodes.Ldloc, strideLocals[side]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, side == 0 ? lblCheckRight : lblFallThrough);
                il.Emit(OpCodes.Ldloc, strideLocals[1 - side]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblFallThrough);

                var locS = il.DeclareLocal(GetClrType(outType));
                il.Emit(OpCodes.Ldloc, ptrLocals[side]);
                EmitLoadIndirect(il, operandTypes[side]);
                EmitConvertTo(il, operandTypes[side], outType);
                il.Emit(OpCodes.Stloc, locS);
                il.Emit(OpCodes.Ldloc, locS);
                il.Emit(OpCodes.Ldloc, locS);
                il.Emit(OpCodes.Beq, lblFallThrough);   // ordered → not NaN (and the other input is not stride-0)

                EmitBroadcastNaNRegion(il, outType, locS,
                    ptrLocals[1 - side], operandTypes[1 - side], strideLocals[1 - side],
                    ptrLocals[2], strideLocals[2],
                    pushCount: g => g.Emit(OpCodes.Ldarg_2),
                    scalarIsSecond: side == 1,
                    pushSplitPerCall: EmitLoadRuleSplitPerCall,
                    pushBufferChunk: EmitLoadRuleBufferChunk,
                    contiguous: false, lblEnd);
            }

            il.MarkLabel(lblFallThrough);
        }

        /// <summary>
        ///     Pushes <see cref="BroadcastNaNRule.SplitPerCall"/> from the kernel's <c>auxdata</c> argument (int32), or 0 when
        ///     <c>auxdata</c> is null.
        /// </summary>
        /// <param name="il">The IL stream of an <see cref="NDInnerLoopFunc"/> kernel.</param>
        private static void EmitLoadRuleSplitPerCall(ILGenerator il)
        {
            var lblNull = il.DefineLabel();
            var lblDone = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Brfalse, lblNull);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldind_I4);                       // BroadcastNaNRule.SplitPerCall at byte offset 0
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblNull);
            il.Emit(OpCodes.Ldc_I4_0);
            il.MarkLabel(lblDone);
        }

        /// <summary>
        ///     Pushes <see cref="BroadcastNaNRule.BufferChunk"/> from the kernel's <c>auxdata</c> argument (int64), or 0 when
        ///     <c>auxdata</c> is null.
        /// </summary>
        /// <param name="il">The IL stream of an <see cref="NDInnerLoopFunc"/> kernel.</param>
        private static void EmitLoadRuleBufferChunk(ILGenerator il)
        {
            var lblNull = il.DefineLabel();
            var lblDone = il.DefineLabel();
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Brfalse, lblNull);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);                       // BroadcastNaNRule.BufferChunk at byte offset 8
            il.Emit(OpCodes.Br, lblDone);
            il.MarkLabel(lblNull);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.MarkLabel(lblDone);
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
        /// The scalar-tail element of a broadcast-pattern loop: a contiguous input loads
        /// <c>ptr + i*elemSize</c>, a broadcast input pushes the value hoisted before the loop; then
        /// the scalar body, then the contiguous store — <see cref="EmitScalarElement"/>'s contig form
        /// with the broadcast inputs read once instead of through a stride-0 address.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="ptrLocals">Per-operand data pointers.</param>
        /// <param name="locScalarVal">Per input: the hoisted value of a broadcast input, or null for a contiguous one.</param>
        /// <param name="locI">The element index.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        private static void EmitScalarElementWithBroadcast(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder?[] locScalarVal,
            LocalBuilder locI,
            Action<ILGenerator> scalarBody)
        {
            int nIn = operandTypes.Length - 1;
            NPTypeCode outType = operandTypes[nIn];

            for (int op = 0; op < nIn; op++)
            {
                if (locScalarVal[op] is { } hoisted)
                {
                    il.Emit(OpCodes.Ldloc, hoisted);
                    continue;
                }
                EmitAddrIPlusOffset(il, ptrLocals[op], locI, 0, GetTypeSize(operandTypes[op]));
                EmitLoadIndirect(il, operandTypes[op]);
            }

            scalarBody(il);

            // [valOut] → [outAddr, valOut] for the indirect store.
            var locOutVal = il.DeclareLocal(GetClrType(outType));
            il.Emit(OpCodes.Stloc, locOutVal);
            EmitAddrIPlusOffset(il, ptrLocals[nIn], locI, 0, GetTypeSize(outType));
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
        /// SL/SR broadcast paths catch the common binary cases first, and the
        /// broadcast-pattern gather stages (<paramref name="isScalar"/>) the
        /// rest, so this generic form only meets them in a kernel without stages.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output; one shared SIMD dtype.</param>
        /// <param name="ptrLocals">Per-operand data pointers.</param>
        /// <param name="strideLocals">Per-operand inner byte strides (each input within the gather budget).</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body (the strided tail).</param>
        /// <param name="g">The gather instruction handles for the dtype.</param>
        /// <param name="scatterOut">True: the output is strided and stored lane by lane; false: one contiguous vector store.</param>
        /// <param name="isScalar">
        /// Null (gather every input), or per input whether it is BROADCAST along the loop (inner byte stride 0):
        /// such an input is read once before the loop and <c>Vector.Create</c>d for every group instead of being
        /// gathered with an all-zero index vector — one hoisted broadcast replacing a gather per vector, the
        /// <c>np.clip(x[::2], lo, hi)</c> shape where the bounds are 0-d.
        /// </param>
        /// <remarks>The strided scalar tail serves a broadcast input unchanged: its byte stride is 0, so
        /// <c>ptr + i*stride</c> is its one element.</remarks>
        private static void EmitSimdGatherLoop(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody,
            Action<ILGenerator> scalarBody,
            GatherSupport g,
            bool scatterOut,
            bool[]? isScalar = null)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode dtype = operandTypes[0]; // same dtype across operands (CanSimdAllOperands)
            int elemSize = GetTypeSize(dtype);
            long lanes = g.Lanes;
            long unrollStep = lanes * 4;

            // ── Broadcast inputs: their one value as a hoisted vector (no index vector, no gather). ──
            LocalBuilder?[]? locVScalar = null;
            if (isScalar != null)
            {
                locVScalar = new LocalBuilder?[nIn];
                var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(dtype));
                for (int op = 0; op < nIn; op++)
                {
                    if (!isScalar[op]) continue;
                    locVScalar[op] = il.DeclareLocal(vecType);
                    il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                    EmitLoadIndirect(il, operandTypes[op]);
                    EmitVectorCreate(il, operandTypes[op]);
                    il.Emit(OpCodes.Stloc, locVScalar[op]!);
                }
            }

            // ── Hoist per-input index vectors: Create(0, s, 2s, …, (lanes-1)s). ──
            var idxLocals = new LocalBuilder[nIn];
            var locSInt = il.DeclareLocal(typeof(int));
            for (int op = 0; op < nIn; op++)
            {
                if (locVScalar?[op] != null)
                    continue;   // broadcast: pushed from its hoisted vector, never gathered
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
                    locI, offset, elemSize, g, vectorBody, scatterOut, locOutVec, locVScalar);
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
                locI, 0, elemSize, g, vectorBody, scatterOut, locOutVec, locVScalar);

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
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="ptrLocals">Per-operand data pointers.</param>
        /// <param name="strideLocals">Per-operand inner byte strides.</param>
        /// <param name="idxLocals">Per gathered input its hoisted byte-offset index vector (null for a broadcast input).</param>
        /// <param name="locI">The element index.</param>
        /// <param name="offset">The group's element offset from <paramref name="locI"/>.</param>
        /// <param name="elemSize">The element size in bytes.</param>
        /// <param name="g">The gather handles.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scatterOut">True: per-lane strided store; false: contiguous vector store.</param>
        /// <param name="locOutVec">Scratch for the result vector (scatter only).</param>
        /// <param name="locVScalar">Null, or per input the hoisted vector of a broadcast input, pushed instead of a gather.</param>
        private static void EmitGatherVectorGroup(
            ILGenerator il,
            NPTypeCode[] operandTypes,
            LocalBuilder[] ptrLocals,
            LocalBuilder[] strideLocals,
            LocalBuilder[] idxLocals,
            LocalBuilder locI, long offset, int elemSize,
            GatherSupport g,
            Action<ILGenerator> vectorBody,
            bool scatterOut, LocalBuilder? locOutVec,
            LocalBuilder?[]? locVScalar = null)
        {
            int nIn = operandTypes.Length - 1;
            NPTypeCode outType = operandTypes[nIn];

            for (int op = 0; op < nIn; op++)
            {
                if (locVScalar?[op] is { } hoisted)
                {
                    il.Emit(OpCodes.Ldloc, hoisted);
                    continue;
                }
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
