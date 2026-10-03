using System;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs — N-ary inputs broadcast along the loop
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// An element-wise kernel with THREE or more inputs whose inner axis mixes contiguous rows (C,
// inner byte stride = element size) with values broadcast along the loop (S, inner byte stride 0)
// had no SIMD path. The shapes are ordinary:
//   • np.clip(m, row_bound, 0.75)       — (C, C, S) over (rows, cols)
//   • np.clip(m, col_bound, hi)         — (C, S, S): the column bound is one value per row
//   • np.clip(x, 0.25, 0.75, out=f32)   — a buffered cast: the 0-d bounds stay stride-0 operands
//   • np.clip(x, lo, hi, out=x_shifted) — the overlap copy leaves the 0-d bounds stride-0 too
//   • np.clip(x, lo, hi, where=mask)    — the masked driver calls the kernel once per mask run
// The binary kernels enumerate their patterns ({CC, SC, CS}); the 1-D shell sent every other
// input count to "all inputs contiguous, else hardware gather (a zero-offset gather per vector for
// each broadcast input), else scalar", and both 2-D block kernels required all-C for nIn >= 3,
// else their scalar 2-D block. Measured: np.clip(m, row, 0.75) at 0.2-0.5x NumPy — and NumPy's own
// loop for array bounds is SCALAR (clip.cpp `_npy_clip_`: one `_NPY_CLIP` per element).
//
// THE SHAPE
// ---------
// N inputs have 2^N C/S patterns. Each is a different instruction stream — a vector LOAD per group
// for a C input, a Vector.Create hoisted out of the loop (once per call; once per ROW in the 2-D
// kernels) for an S input — so each pattern needs its own emitted loop. Emitting all of them into
// the kernel's one method would spend the JIT's per-method inline budget (a local-variable limit
// and a time budget; once spent, the method inlines nothing more and its Vector256 helpers stay
// CALLS — measured 3.3x slower on the polynomial kernels, ILKernelGenerator.Polynomial.Eval.cs).
// So:
//   • the kernel computes the pattern ONCE per call: bit `op` is set when input op's inner byte
//     stride is 0; a stride that is neither 0 nor the element size leaves the SIMD path exactly
//     where it always did (gather / scalar);
//   • pattern 0 — every input contiguous, the common shape — stays INLINE, the very loop the
//     kernel always emitted, so it pays no extra call;
//   • every other pattern is a STAGE: a DynamicMethod of its own with the kernel's own signature,
//     holding that pattern's loop; the kernel forwards its arguments with one direct call. A stage
//     is JIT-compiled on its FIRST call (probed: a DynamicMethod another DynamicMethod calls is
//     compiled when the call first executes, not when the caller is), so the patterns a program
//     never meets cost only their IL emission.
// The stage bodies reuse the kernels' own pattern emitters (EmitSimdContigLoop with its broadcast
// inputs, Emit2DSimdBlock, Emit2DMaskedSimdBlock) and the SAME scalarBody / vectorBody delegates,
// so a broadcast input produces byte-identical results to the per-element route: an S input's
// vector is Vector.Create of the very element every lane would have loaded.
//
// Kernels with more than MaxBroadcastPatternInputs inputs keep the all-C requirement (2^N stages
// would grow without bound; no NumSharp op has that many inputs today).
//
// THE GATHER ROUTE
// ----------------
// The 1-D shell reaches its hardware-gather loop when an input or the output is genuinely
// strided. A broadcast input there (np.clip(x[::2], lo, hi) with 0-d bounds, np.add(x[::2], 5.0,
// out=o)) used to be gathered with an all-zero index vector — one vgather per vector group for a
// value that never changes. The gather route now computes its own pattern (bit op = stride 0) for
// two or more inputs: pattern 0 falls through to the inline gather loops, every other pattern
// forwards to a stage that gathers only the strided inputs and broadcasts the rest once per call,
// with the shell's own contiguous / scattered / scalar output dispatch.
//
// Measured (np.clip, NPY/NS, P-core pinned, clock locked): row / column bounds 5-28x at 100K
// (were 0.2-0.5x), where= over 2-D rows with a row bound 19-41x (were 1.6-2.1x), strided float32
// with 0-d bounds 1.9x (was 0.76x); the probes in the commit message have the full table.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// The widest input count that gets per-pattern broadcast stages: 4 inputs = 15 stages per
        /// kernel at most (each compiled only if its pattern is ever met). Wider kernels keep the
        /// all-contiguous SIMD requirement.
        /// </summary>
        internal const int MaxBroadcastPatternInputs = 4;

        /// <summary>
        /// The inputs a pattern number marks as broadcast along the loop (S): bit <c>op</c> of
        /// <paramref name="pattern"/> set means input <c>op</c> has inner byte stride 0.
        /// </summary>
        /// <param name="pattern">The pattern number, <c>0 .. 2^nIn - 1</c>.</param>
        /// <param name="nIn">The kernel's input count.</param>
        /// <returns>One flag per input: true = broadcast (S), false = contiguous (C).</returns>
        private static bool[] BroadcastPatternInputs(int pattern, int nIn)
        {
            var isScalar = new bool[nIn];
            for (int op = 0; op < nIn; op++)
                isScalar[op] = ((pattern >> op) & 1) != 0;
            return isScalar;
        }

        /// <summary>
        /// Emit the per-call pattern computation and its <c>switch</c>: for every input, an inner
        /// byte stride equal to <paramref name="elemSize"/> is C (bit clear), 0 is S (bit set), and
        /// anything else jumps to <paramref name="lblMismatch"/>; the resulting pattern number then
        /// indexes <paramref name="targets"/>.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="innerStrideLocals">Per-operand inner byte strides (inputs first; only the first <paramref name="nIn"/> are read).</param>
        /// <param name="nIn">The input count; <paramref name="targets"/> has <c>2^nIn</c> entries.</param>
        /// <param name="elemSize">The (shared) element size of the inputs, in bytes.</param>
        /// <param name="lblMismatch">Where an input that is neither contiguous nor broadcast goes (the kernel's non-SIMD route).</param>
        /// <param name="targets">One label per pattern number.</param>
        /// <remarks>
        /// The pattern is built with <c>or</c>s into an int local and dispatched by one IL
        /// <c>switch</c> (a jump table), so the dispatch costs <c>2·nIn</c> compares and one
        /// indirect jump per CALL — never per element. The trailing <c>br</c> is unreachable
        /// (every pattern number is in range) but keeps the IL verifiable.
        /// </remarks>
        private static void EmitBroadcastPatternSwitch(
            ILGenerator il, LocalBuilder[] innerStrideLocals, int nIn, long elemSize, Label lblMismatch, Label[] targets)
        {
            var locPattern = il.DeclareLocal(typeof(int));
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locPattern);
            for (int op = 0; op < nIn; op++)
            {
                var lblContiguous = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, elemSize);
                il.Emit(OpCodes.Beq, lblContiguous);
                il.Emit(OpCodes.Ldloc, innerStrideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblMismatch);
                // pattern |= 1 << op
                il.Emit(OpCodes.Ldloc, locPattern);
                il.Emit(OpCodes.Ldc_I4, 1 << op);
                il.Emit(OpCodes.Or);
                il.Emit(OpCodes.Stloc, locPattern);
                il.MarkLabel(lblContiguous);
            }
            il.Emit(OpCodes.Ldloc, locPattern);
            il.Emit(OpCodes.Switch, targets);
            il.Emit(OpCodes.Br, lblMismatch);
        }

        /// <summary>
        /// Emit a call that forwards the kernel's own arguments, unchanged and in order, to a stage
        /// with the same signature. The caller emits the branch that follows (the stage returns
        /// void; the kernel continues at its end label).
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="stage">The stage (same parameter list as the kernel).</param>
        /// <param name="argCount">The kernel's parameter count (4 for the 1-D inner loop, 5 for a 2-D block).</param>
        private static void EmitForwardToStage(ILGenerator il, DynamicMethod stage, int argCount)
        {
            for (int a = 0; a < argCount; a++)
                EmitLdarg(il, a);
            il.Emit(OpCodes.Call, stage);
        }

        // ---------------------------------------------------------------------
        // 1-D shell (NDInnerLoopFunc)
        // ---------------------------------------------------------------------

        /// <summary>
        /// The 1-D shell's SIMD route for three or more inputs: require a contiguous output, then
        /// dispatch on the inputs' C/S pattern — pattern 0 runs the inline contiguous loop the
        /// shell always emitted, every other pattern forwards to its stage
        /// (<see cref="Generate1DBroadcastStage"/>).
        /// </summary>
        /// <param name="il">The kernel's generator (its prologue already loaded the locals).</param>
        /// <param name="operandTypes">Inputs then the output; one shared SIMD dtype (<see cref="CanSimdAllOperands"/>).</param>
        /// <param name="ptrLocals">Per-operand data pointers.</param>
        /// <param name="strideLocals">Per-operand inner byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body (the tail).</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages for JIT diagnostics).</param>
        /// <param name="lblBail">The kernel's non-contiguous route (gather when available, else the scalar strided loop).</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        private static void EmitBroadcastPatternDispatch1D(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string cacheKey, Label lblBail, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;
            int elemSize = GetTypeSize(operandTypes[nIn]);

            // The vector store needs a contiguous output; anything else is the gather / scalar route.
            il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Bne_Un, lblBail);

            var targets = new Label[1 << nIn];
            for (int p = 0; p < targets.Length; p++)
                targets[p] = il.DefineLabel();
            EmitBroadcastPatternSwitch(il, strideLocals, nIn, elemSize, lblBail, targets);

            // Pattern 0: every input contiguous — the inline loop, unchanged.
            il.MarkLabel(targets[0]);
            EmitSimdContigLoop(il, operandTypes, ptrLocals, vectorBody, scalarBody);
            il.Emit(OpCodes.Br, lblEnd);

            for (int p = 1; p < targets.Length; p++)
            {
                il.MarkLabel(targets[p]);
                var stage = Generate1DBroadcastStage(operandTypes, BroadcastPatternInputs(p, nIn), vectorBody, scalarBody,
                    $"NDInnerLoop_{Sanitize(cacheKey)}_bp{p}");
                EmitForwardToStage(il, stage, argCount: 4);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One 1-D broadcast-pattern stage: the <see cref="NDInnerLoopFunc"/> signature, a prologue
        /// that snapshots the pointers and strides, and <see cref="EmitSimdContigLoop"/> with the
        /// pattern's broadcast inputs hoisted.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (one shared SIMD dtype).</param>
        /// <param name="isScalar">Per input: broadcast (true) or contiguous (false). At least one is true.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage, ready to be the target of a <c>call</c> (compiled on its first call).</returns>
        private static DynamicMethod Generate1DBroadcastStage(
            NPTypeCode[] operandTypes, bool[] isScalar, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            var ptrLocals = new LocalBuilder[nOp];
            var strideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                strideLocals[op] = il.DeclareLocal(typeof(long));
            }
            // The strides are dead here (the pattern fixed them); the JIT drops their loads.
            EmitLoadInnerLoopArgs(il, nOp, ptrLocals, strideLocals);
            EmitSimdContigLoop(il, operandTypes, ptrLocals, vectorBody, scalarBody, isScalar);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        /// The 1-D shell's hardware-gather route with broadcast inputs (two or more inputs; every input stride already
        /// checked within the gather budget): when no input has inner byte stride 0 this emits nothing that runs — the
        /// pattern is 0 and control falls through to the shell's inline gather loops — otherwise it forwards to the
        /// pattern's stage (<see cref="GenerateGatherBroadcastStage"/>), which gathers only the strided inputs and
        /// broadcasts the others once per call.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output (one shared gather-capable dtype).</param>
        /// <param name="strideLocals">Per-operand inner byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="g">The gather handles for the dtype.</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages).</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        /// <remarks>
        /// The gather route is reached when an input or the output is genuinely strided (the contiguous and
        /// broadcast-contiguous patterns were taken before it), so a broadcast input here sits next to a strided one —
        /// <c>np.clip(x[::2], lo, hi)</c>, <c>np.add(x[::2], 5.0, out=o)</c> — and used to cost one all-zero-index
        /// gather per vector group.
        /// </remarks>
        private static void EmitGatherBroadcastPatternDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, GatherSupport g, string cacheKey, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;

            // pattern = Σ (stride == 0 ? 1 << op : 0) — any other stride is gathered.
            var locPattern = il.DeclareLocal(typeof(int));
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locPattern);
            for (int op = 0; op < nIn; op++)
            {
                var lblNotBroadcast = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, strideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Bne_Un, lblNotBroadcast);
                il.Emit(OpCodes.Ldloc, locPattern);
                il.Emit(OpCodes.Ldc_I4, 1 << op);
                il.Emit(OpCodes.Or);
                il.Emit(OpCodes.Stloc, locPattern);
                il.MarkLabel(lblNotBroadcast);
            }

            var lblNone = il.DefineLabel();
            var targets = new Label[1 << nIn];
            targets[0] = lblNone;
            for (int p = 1; p < targets.Length; p++)
                targets[p] = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locPattern);
            il.Emit(OpCodes.Switch, targets);
            il.Emit(OpCodes.Br, lblNone);

            for (int p = 1; p < targets.Length; p++)
            {
                il.MarkLabel(targets[p]);
                var stage = GenerateGatherBroadcastStage(operandTypes, BroadcastPatternInputs(p, nIn), vectorBody, scalarBody, g,
                    $"NDInnerLoop_{Sanitize(cacheKey)}_gbp{p}");
                EmitForwardToStage(il, stage, argCount: 4);
                il.Emit(OpCodes.Br, lblEnd);
            }

            il.MarkLabel(lblNone);
        }

        /// <summary>
        /// One gather broadcast-pattern stage: the <see cref="NDInnerLoopFunc"/> signature, the gather loop with the
        /// pattern's broadcast inputs hoisted, and the shell's own output dispatch — a contiguous output takes the
        /// vector store, a strided one within the gather budget the per-lane scatter, anything else the scalar strided
        /// loop.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (one shared gather-capable dtype).</param>
        /// <param name="isScalar">Per input: broadcast (true) or gathered (false). At least one is true.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="g">The gather handles.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage (compiled on its first call).</returns>
        private static DynamicMethod GenerateGatherBroadcastStage(
            NPTypeCode[] operandTypes, bool[] isScalar, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody,
            GatherSupport g, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(void*) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            var ptrLocals = new LocalBuilder[nOp];
            var strideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                strideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadInnerLoopArgs(il, nOp, ptrLocals, strideLocals);

            var lblScatter = il.DefineLabel();
            var lblScalar = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
            il.Emit(OpCodes.Ldc_I8, (long)GetTypeSize(operandTypes[nIn]));
            il.Emit(OpCodes.Bne_Un, lblScatter);
            EmitSimdGatherLoop(il, operandTypes, ptrLocals, strideLocals, vectorBody, scalarBody, g, scatterOut: false, isScalar);
            il.Emit(OpCodes.Ret);

            // A strided output scatters lane by lane when its stride fits the budget the shell applies.
            il.MarkLabel(lblScatter);
            il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
            il.Emit(OpCodes.Ldc_I8, GatherStrideLimit);
            il.Emit(OpCodes.Bgt, lblScalar);
            il.Emit(OpCodes.Ldloc, strideLocals[nIn]);
            il.Emit(OpCodes.Ldc_I8, -GatherStrideLimit);
            il.Emit(OpCodes.Blt, lblScalar);
            EmitSimdGatherLoop(il, operandTypes, ptrLocals, strideLocals, vectorBody, scalarBody, g, scatterOut: true, isScalar);
            il.Emit(OpCodes.Ret);

            il.MarkLabel(lblScalar);
            EmitScalarStridedLoop(il, operandTypes, ptrLocals, strideLocals, scalarBody);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        // ---------------------------------------------------------------------
        // 2-D block kernel (ND2DElementwiseKernel)
        // ---------------------------------------------------------------------

        /// <summary>
        /// The 2-D block kernel's SIMD route for three or more inputs (the output row's contiguity
        /// is already checked): dispatch on the inputs' C/S pattern — pattern 0 runs the inline
        /// all-contiguous block, every other pattern forwards to its stage
        /// (<see cref="Generate2DBroadcastStage"/>), whose S inputs are loaded and broadcast once
        /// per ROW.
        /// </summary>
        /// <param name="il">The kernel's generator (its prologue already loaded the locals).</param>
        /// <param name="operandTypes">Inputs then the output (one shared SIMD dtype).</param>
        /// <param name="ptrLocals">Per-operand data pointers.</param>
        /// <param name="innerStrideLocals">Per-operand inner byte strides.</param>
        /// <param name="outerStrideLocals">Per-operand outer byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages).</param>
        /// <param name="lblScalar">The kernel's scalar 2-D block.</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        private static void Emit2DBroadcastPatternDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody,
            string cacheKey, Label lblScalar, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;
            int elemSize = GetTypeSize(operandTypes[nIn]);

            var targets = new Label[1 << nIn];
            for (int p = 0; p < targets.Length; p++)
                targets[p] = il.DefineLabel();
            EmitBroadcastPatternSwitch(il, innerStrideLocals, nIn, elemSize, lblScalar, targets);

            // Pattern 0: every input row contiguous — the inline block, unchanged.
            il.MarkLabel(targets[0]);
            Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new bool[nIn], vectorBody, scalarBody);
            il.Emit(OpCodes.Br, lblEnd);

            for (int p = 1; p < targets.Length; p++)
            {
                il.MarkLabel(targets[p]);
                var stage = Generate2DBroadcastStage(operandTypes, BroadcastPatternInputs(p, nIn), vectorBody, scalarBody,
                    $"ND2DLoop_{Sanitize(cacheKey)}_bp{p}");
                EmitForwardToStage(il, stage, argCount: 5);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One 2-D broadcast-pattern stage: the <see cref="ND2DElementwiseKernel"/> signature, a
        /// prologue that snapshots the pointers and outer strides, and <see cref="Emit2DSimdBlock"/>
        /// for the pattern (its row-shape dispatch, masked tails and per-row broadcast hoist
        /// included).
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (one shared SIMD dtype).</param>
        /// <param name="isScalar">Per input: broadcast along the row (true) or a contiguous row (false).</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage (compiled on its first call).</returns>
        private static DynamicMethod Generate2DBroadcastStage(
            NPTypeCode[] operandTypes, bool[] isScalar, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            var ptrLocals = new LocalBuilder[nOp];
            var outerStrideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadDataPtrs2D(il, nOp, ptrLocals);
            EmitLoadStrideArray(il, Arg2DOuterStrides, nOp, outerStrideLocals);
            Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, isScalar, vectorBody, scalarBody);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>locals[op] = (byte*)dataptrs[op] for each operand of a 2-D block call.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="count">How many pointers to load (the operands, plus the mask for the where= form).</param>
        /// <param name="ptrLocals">The destination locals.</param>
        private static void EmitLoadDataPtrs2D(ILGenerator il, int count, LocalBuilder[] ptrLocals)
        {
            for (int op = 0; op < count; op++)
            {
                EmitLdarg(il, Arg2DDataPtrs);
                if (op > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, op * IntPtr.Size);
                    il.Emit(OpCodes.Conv_I);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Ldind_I);
                il.Emit(OpCodes.Stloc, ptrLocals[op]);
            }
        }

        // ---------------------------------------------------------------------
        // Masked (where=) 2-D block kernel
        // ---------------------------------------------------------------------

        /// <summary>
        /// The where= 2-D kernel's SIMD route for three or more inputs under one mask mode
        /// (inputs already checked contiguous-or-broadcast, the output row contiguous): dispatch on
        /// the inputs' C/S pattern — pattern 0 runs the inline masked block, every other pattern
        /// forwards to its stage (<see cref="Generate2DMaskedBroadcastStage"/>).
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">The DATA operands, inputs then the output (the mask is the implicit trailing pointer).</param>
        /// <param name="ptrLocals">Per-operand data pointers, the mask's last.</param>
        /// <param name="innerStrideLocals">Per-operand inner byte strides, the mask's last.</param>
        /// <param name="outerStrideLocals">Per-operand outer byte strides, the mask's last.</param>
        /// <param name="elemSize">The data element size in bytes.</param>
        /// <param name="ms">The lane-mask support for the output dtype.</param>
        /// <param name="mw">The mask-word support (mask bytes → lane mask).</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="byteMask">True: one mask byte per element; false: one per row.</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages).</param>
        /// <param name="lblScalar">The kernel's scalar masked block.</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        private static void Emit2DMaskedBroadcastPatternDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, LocalBuilder[] innerStrideLocals,
            LocalBuilder[] outerStrideLocals, int elemSize, MaskSupport ms, MaskWordSupport mw,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, bool byteMask,
            string cacheKey, Label lblScalar, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;
            var targets = new Label[1 << nIn];
            for (int p = 0; p < targets.Length; p++)
                targets[p] = il.DefineLabel();
            EmitBroadcastPatternSwitch(il, innerStrideLocals, nIn, elemSize, lblScalar, targets);

            il.MarkLabel(targets[0]);
            Emit2DMaskedSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, new bool[nIn], elemSize, ms, mw, vectorBody, scalarBody, byteMask);
            il.Emit(OpCodes.Br, lblEnd);

            for (int p = 1; p < targets.Length; p++)
            {
                il.MarkLabel(targets[p]);
                var stage = Generate2DMaskedBroadcastStage(operandTypes, BroadcastPatternInputs(p, nIn), elemSize, ms, mw,
                    vectorBody, scalarBody, byteMask, $"ND2DMaskedLoop_{Sanitize(cacheKey)}_{(byteMask ? "m" : "r")}bp{p}");
                EmitForwardToStage(il, stage, argCount: 5);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One where= 2-D broadcast-pattern stage: the block-kernel signature, a prologue that
        /// snapshots the data and mask pointers and outer strides, and
        /// <see cref="Emit2DMaskedSimdBlock"/> for the pattern under one mask mode.
        /// </summary>
        /// <param name="operandTypes">The DATA operands, inputs then the output.</param>
        /// <param name="isScalar">Per input: broadcast along the row (true) or a contiguous row (false).</param>
        /// <param name="elemSize">The data element size in bytes.</param>
        /// <param name="ms">The lane-mask support for the output dtype.</param>
        /// <param name="mw">The mask-word support.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="byteMask">True: one mask byte per element; false: one per row.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage (compiled on its first call).</returns>
        private static DynamicMethod Generate2DMaskedBroadcastStage(
            NPTypeCode[] operandTypes, bool[] isScalar, int elemSize, MaskSupport ms, MaskWordSupport mw,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, bool byteMask, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            int nAll = operandTypes.Length + 1;                   // + the mask
            var ptrLocals = new LocalBuilder[nAll];
            var outerStrideLocals = new LocalBuilder[nAll];
            for (int op = 0; op < nAll; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadDataPtrs2D(il, nAll, ptrLocals);
            EmitLoadStrideArray(il, Arg2DOuterStrides, nAll, outerStrideLocals);
            Emit2DMaskedSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, isScalar, elemSize, ms, mw, vectorBody, scalarBody, byteMask);
            il.Emit(OpCodes.Ret);
            return dm;
        }
    }
}
