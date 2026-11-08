using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Backends.Iteration;

// =============================================================================
// DirectILKernelGenerator.InnerLoop.Reversed.cs — reversed runs (inner byte stride −elemSize)
// =============================================================================
//
// WHY THIS EXISTS
// ---------------
// A reversed view (a[::-1], np.flip(a), m[:, ::-1]) walks its inner axis at byte stride −elemSize.
// The Tier-3B shell's SIMD loops took only +elemSize (contiguous, C) and 0 (broadcast, S) inputs,
// so a reversed operand fell to the hardware gather (4- and 8-byte dtypes: one vgather per vector
// for lanes that sit side by side in memory) or, for the 1- and 2-byte dtypes AVX2 cannot gather,
// to the scalar strided loop. Measured on strings 4c78d673: `a[::-1] + b` cost 2.2-2.9x the
// contiguous add at 1K, 2x at 100K float32, 7-13x at 100K int16 / uint8; the 2-D block kernel
// refused a reversed inner axis outright, so `m[:, ::-1] + n` paid one gather call per row
// (5.4x the reversed-rows form at 1000 x 8).
//
// THE LOAD
// --------
// The V elements a reversed operand gives one vector group, logical i .. i+V-1, are the V
// consecutive addresses that END at element i: one contiguous load of [ptr - (i+V-1)e, ptr - i·e]
// holds them in descending order, and one lane reversal puts element i in lane 0. The op's
// vectorBody then runs unchanged, so every lane computes the very element pair it computed on the
// gather or scalar route: the results are byte-identical, the NaN operand order included (that
// rule lives in the bodies, lane by lane). A reversed OUTPUT is the mirror image: the result's
// lanes are reversed and written with one contiguous store. A group is loaded only when all of its
// V elements exist (the tail is scalar, or one masked vector in the 2-D kernel whose mask is
// reversed too), so nothing outside the view is ever read or written.
//
// LANE REVERSAL — AVX2 at 256 bits, the host set the gather route needs as well
//   8-byte lanes   vpermpd / vpermq, immediate 0x1B                       (Avx2.Permute4x64)
//   4-byte lanes   vpermps / vpermd with the index vector 7, 6, ..., 0    (Avx2.PermuteVar8x32)
//   1-/2-byte      vpshufb reversing each 128-bit half, then vpermq 0x4E swapping the halves
// One µop (two for the narrow lanes) per reversed operand per vector, against a gather's four to
// eight loads. Other hosts (V128, V512, Arm64) keep the gather / scalar routes.
//
// THE PATTERNS — stages
// ---------------------
// Every operand of a call is C (+e), R (−e) or S (0, inputs only), fixed for the whole call. The
// 1-D shell computes the call's pattern (base 3, one digit per operand, the output's digit last)
// once, after its contiguous / broadcast checks have failed and before the gather; a pattern with
// at least one R forwards to its STAGE — a DynamicMethod of its own with the kernel's signature,
// holding that pattern's 4x-unrolled loop — and every other pattern goes on to the gather / scalar
// route exactly as before. The 2-D block kernel does the same for its inputs (its output row is
// always contiguous). Stages keep each method's JIT inline budget for one loop and compile on
// their first call, so a program pays JIT time only for the patterns it meets
// (DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs has the mechanism).
//
// THE VARIANT KERNELS
// -------------------
// The dispatch and its stage calls live in a reversed-run VARIANT of each kernel (the same
// generator, withReversedRuns), cached apart (_innerLoopReversedKeyCache, _innerLoop2DReversedCache).
// NDIterRef runs the variant only when the iteration has an operand at inner element stride -1
// (unbuffered, so that stride holds for every call); every other iteration keeps the plain kernel.
// The variant is complete — its other patterns run the plain kernel's very loops — but its stage
// calls change how the JIT allocates registers across the whole method: the binary 2-D kernel
// became a non-leaf method with callee-saved registers, a frame and a spill in its contiguous
// loops (disassembly of ND2DLoop_npy_binop_Add_Int16_*), which the plain kernel must not pay.
//
// SCOPE: kernels of one or two inputs (MaxReversedRunInputs): the binary ufuncs and the unary ops
// that reach the NDIter route. Three or more inputs would need 3^N - 2^N stages per output mode
// (19 for np.clip's three); they keep the gather / scalar route. A 1-D unary op that never
// reaches the iterator has the same load in its own kernel (GetReversedUnaryKernel,
// DirectILKernelGenerator.Unary.Strided.cs).
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        /// The widest input count whose kernels get reversed-run stages. A kernel with N inputs has up to
        /// <c>2 · 3^N</c> operand patterns (each input C / R / S, the output C / R), so the stages stop at
        /// the binary kernels (at most 18 patterns, 14 of them with a reversed operand); wider kernels keep the
        /// gather / scalar route for a reversed operand.
        /// </summary>
        internal const int MaxReversedRunInputs = 2;

        /// <summary>Operand mode of a reversed-run pattern digit: contiguous, byte stride <c>+elemSize</c>.</summary>
        private const int RunModeC = 0;

        /// <summary>Operand mode of a reversed-run pattern digit: reversed, byte stride <c>−elemSize</c>.</summary>
        private const int RunModeR = 1;

        /// <summary>Operand mode of a reversed-run pattern digit: broadcast along the loop, byte stride 0 (inputs only).</summary>
        private const int RunModeS = 2;

        /// <summary>
        /// Whether a vector of <paramref name="dtype"/> lanes can be reversed in registers on this host — the gate of
        /// every reversed-run stage. Requires the 256-bit AVX2 width the kernels were baked for (the reversal
        /// instructions are Vector256 ones) and a SIMD lane of 1, 2, 4 or 8 bytes: the ten numeric SIMD dtypes and
        /// Boolean (which rides byte lanes).
        /// </summary>
        /// <param name="dtype">The shared dtype of every operand of the kernel.</param>
        /// <returns>True when <see cref="EmitReverseLanes"/> can serve the dtype here; false keeps the old routes.</returns>
        internal static bool CanReverseLanes(NPTypeCode dtype)
        {
            if (VectorBits != 256 || !Avx2.IsSupported)
                return false;
            return dtype == NPTypeCode.Boolean || CanUseSimd(dtype);
        }

        /// <summary>
        /// The reversed-run VARIANTS of the packed-key 1-D inner-loop kernels: the same kernel plus the reversed-run
        /// stages (<see cref="GenerateTemplatedInnerLoop"/> with <c>withReversedRuns</c>), served only to iterations with
        /// a reversed operand. Kept apart from <see cref="_innerLoopKeyCache"/> so the plain kernel every other
        /// iteration runs keeps its IL, and its machine code, unchanged.
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, NDInnerLoopFunc> _innerLoopReversedKeyCache = new();

        /// <summary>
        /// The reversed-run VARIANTS of the 2-D block kernels (<see cref="Generate2DElementwiseKernel"/> with
        /// <c>withReversedRuns</c>), served only to iterations with a reversed input row; see
        /// <see cref="_innerLoopReversedKeyCache"/> for why they are separate.
        /// </summary>
        internal static readonly ConcurrentDictionary<InnerLoopKernelKey, ND2DElementwiseKernel> _innerLoop2DReversedCache = new();

        /// <summary>Look up a 1-D reversed-run variant by its packed key without building a string.</summary>
        /// <param name="key">The kernel's packed key (the plain kernel's).</param>
        /// <param name="kernel">The variant, when cached.</param>
        /// <returns>True when the variant is cached.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryGetReversedRunInnerLoop(in InnerLoopKernelKey key, out NDInnerLoopFunc kernel)
            => _innerLoopReversedKeyCache.TryGetValue(key, out kernel!);

        /// <summary>
        /// Serve the 1-D reversed-run variant for <paramref name="key"/> from its cache, else compile and register it.
        /// The bodies must be the ones the plain kernel of the key uses, so every pattern both kernels run gives the
        /// same bytes.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="scalarBody">Per-element body.</param>
        /// <param name="vectorBody">Per-vector body (the variant's stages need it; callers check <see cref="ServesReversedRows(NPTypeCode[], Action{ILGenerator}?)"/>).</param>
        /// <param name="key">The plain kernel's packed key.</param>
        /// <returns>The variant kernel.</returns>
        internal static NDInnerLoopFunc CompileReversedRunInnerLoop(
            NPTypeCode[] operandTypes, Action<ILGenerator> scalarBody, Action<ILGenerator>? vectorBody, in InnerLoopKernelKey key)
        {
            if (_innerLoopReversedKeyCache.TryGetValue(key, out var cached))
                return cached;

            // The same NaN region rule wiring as the plain kernel (CompileInnerLoop's packed form): the variant runs
            // the contiguous / broadcast loops of every non-reversed pattern too.
            BinaryOp? nanOrderedOp = key.Family == InnerLoopKernelKey.KernelFamily.Binary ? (BinaryOp)key.Op : null;
            var kernel = GenerateTemplatedInnerLoop(operandTypes, scalarBody, vectorBody, key.ToCacheKey(), nanOrderedOp,
                withReversedRuns: true);
            _innerLoopReversedKeyCache.TryAdd(key, kernel);
            return kernel;
        }

        /// <summary>Look up a 2-D reversed-run variant by its packed key without building a string.</summary>
        /// <param name="key">The kernel's packed key (the plain kernel's).</param>
        /// <param name="kernel">The variant, when cached.</param>
        /// <returns>True when the variant is cached.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool TryGet2DReversedRunKernel(in InnerLoopKernelKey key, out ND2DElementwiseKernel kernel)
            => _innerLoop2DReversedCache.TryGetValue(key, out kernel!);

        /// <summary>
        /// Serve the 2-D reversed-run variant for <paramref name="key"/> from its cache, else compile and register it.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="scalarBody">Per-element body.</param>
        /// <param name="vectorBody">Per-vector body.</param>
        /// <param name="key">The plain kernel's packed key.</param>
        /// <returns>The variant kernel.</returns>
        internal static ND2DElementwiseKernel Compile2DReversedRunKernel(
            NPTypeCode[] operandTypes, Action<ILGenerator> scalarBody, Action<ILGenerator>? vectorBody, in InnerLoopKernelKey key)
        {
            if (_innerLoop2DReversedCache.TryGetValue(key, out var cached))
                return cached;

            var kernel = Generate2DElementwiseKernel(operandTypes, scalarBody, vectorBody, key.ToCacheKey(), withReversedRuns: true);
            _innerLoop2DReversedCache.TryAdd(key, kernel);
            return kernel;
        }

        /// <summary>
        /// Whether the 2-D block kernel compiled for these operands runs a REVERSED input row (inner byte stride
        /// −elemSize) as a reversed-run stage — the condition under which <c>NDIterRef</c> may hand such an iteration
        /// to the block kernel instead of the per-row route. Mirrors the kernel's own gates: a vector body, one shared
        /// SIMD dtype the host can reverse lanes for (<see cref="CanReverseLanes"/>), at most
        /// <see cref="MaxReversedRunInputs"/> inputs.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="vectorBody">The op's vector body (null: the kernel is scalar-only).</param>
        /// <returns>True when a reversed row runs vectorized in the block kernel; false keeps the per-row route.</returns>
        /// <remarks>
        /// A false answer must keep the iteration off the block kernel: its scalar block addresses a strided row with
        /// one multiply per operand per element, where the per-row route still had the gather (4- and 8-byte dtypes)
        /// or the running-pointer scalar loop.
        /// </remarks>
        internal static bool ServesReversedRows(NPTypeCode[] operandTypes, Action<ILGenerator>? vectorBody)
            => vectorBody != null
               && operandTypes.Length - 1 <= MaxReversedRunInputs
               && CanSimdAllOperands(operandTypes)
               && CanReverseLanes(operandTypes[0]);

        /// <summary>
        /// <see cref="ServesReversedRows(NPTypeCode[], Action{ILGenerator}?)"/> for a unary or binary kernel, from its
        /// dtypes directly (the packed-key routes build no operand-type array before the gate passes).
        /// </summary>
        /// <param name="in0">The first input's dtype.</param>
        /// <param name="in1">The second input's dtype (a unary caller passes <paramref name="in0"/> again).</param>
        /// <param name="outType">The output's dtype.</param>
        /// <param name="nIn">The input count (1 or 2).</param>
        /// <param name="vectorBody">The op's vector body (null: the kernel is scalar-only).</param>
        /// <returns>True when a reversed row runs vectorized in the block kernel; false keeps the per-row route.</returns>
        internal static bool ServesReversedRows(NPTypeCode in0, NPTypeCode in1, NPTypeCode outType, int nIn, Action<ILGenerator>? vectorBody)
            => vectorBody != null
               && nIn <= MaxReversedRunInputs
               && in0 == outType && in1 == outType
               && CanReverseLanes(outType);

        /// <summary>
        /// Emit the lane-reversal control a <see cref="EmitReverseLanes"/> call reads, into a local declared here:
        /// the index vector <c>7, 6, …, 0</c> for 4-byte lanes (a <c>Vector256&lt;int&gt;</c>, or
        /// <c>Vector256&lt;uint&gt;</c> for uint lanes, the operand type <c>vpermd</c>'s uint overload wants), the
        /// per-half byte shuffle for 1- and 2-byte lanes (a <c>Vector256&lt;byte&gt;</c>). Emitted once per stage,
        /// before its loops, so the loops only read a register.
        /// </summary>
        /// <param name="il">The stage's generator.</param>
        /// <param name="dtype">The shared operand dtype (<see cref="CanReverseLanes"/> approved it).</param>
        /// <returns>The control local, or null for 8-byte lanes (their reversal is an immediate).</returns>
        private static LocalBuilder? EmitReverseControl(ILGenerator il, NPTypeCode dtype)
        {
            var lane = GetSimdLaneType(dtype);
            switch (GetTypeSize(dtype))
            {
                case 8:
                    return null;
                case 4:
                {
                    // vpermd's uint overload takes a uint index vector; float and int take an int one.
                    var indexType = lane == typeof(uint) ? typeof(uint) : typeof(int);
                    var loc = il.DeclareLocal(VectorMethodCache.V(256, indexType));
                    for (int k = 7; k >= 0; k--)
                        il.Emit(OpCodes.Ldc_I4, k);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.CreateElements(256, indexType), null);
                    il.Emit(OpCodes.Stloc, loc);
                    return loc;
                }
                default:
                {
                    // vpshufb works within each 128-bit half: byte k of the result is byte control[k] of the same
                    // half. Reversing 1-byte lanes reads 15 - k; 2-byte lanes keep each lane's two bytes in order
                    // (14, 15, 12, 13, ..., 0, 1).
                    int size = GetTypeSize(dtype);
                    var loc = il.DeclareLocal(VectorMethodCache.V(256, typeof(byte)));
                    for (int half = 0; half < 2; half++)
                    {
                        for (int k = 0; k < 16; k++)
                        {
                            int lane16 = k / size, inLane = k % size;
                            int source = (16 / size - 1 - lane16) * size + inLane;
                            il.Emit(OpCodes.Ldc_I4, source);
                        }
                    }
                    il.EmitCall(OpCodes.Call, VectorMethodCache.CreateElements(256, typeof(byte)), null);
                    il.Emit(OpCodes.Stloc, loc);
                    return loc;
                }
            }
        }

        /// <summary>
        /// Emit the reversal of the vector on top of the stack: <c>[v] → [v reversed]</c>, lane 0 becoming the last
        /// lane. 8-byte lanes take one <c>vpermpd</c>/<c>vpermq</c> (immediate <c>0x1B</c>); 4-byte lanes one
        /// <c>vpermps</c>/<c>vpermd</c> with <paramref name="locControl"/>; 1- and 2-byte lanes a <c>vpshufb</c>
        /// that reverses each 128-bit half with <paramref name="locControl"/> and a <c>vpermq 0x4E</c> that swaps
        /// the halves (AVX2 has no cross-half byte or word permute). The reinterpretations around them are free.
        /// </summary>
        /// <param name="il">The stage's generator.</param>
        /// <param name="dtype">The operand dtype (<see cref="CanReverseLanes"/> approved it).</param>
        /// <param name="locControl">The control from <see cref="EmitReverseControl"/> (null for 8-byte lanes).</param>
        /// <exception cref="InvalidOperationException">An AVX2 overload the dtype needs is missing (never on a supported runtime).</exception>
        private static void EmitReverseLanes(ILGenerator il, NPTypeCode dtype, LocalBuilder? locControl)
        {
            var lane = GetSimdLaneType(dtype);
            var vecLane = VectorMethodCache.V(256, lane);
            switch (GetTypeSize(dtype))
            {
                case 8:
                    il.Emit(OpCodes.Ldc_I4, 0x1B);
                    il.EmitCall(OpCodes.Call, Avx2Method("Permute4x64", vecLane, typeof(byte)), null);
                    return;
                case 4:
                    il.Emit(OpCodes.Ldloc, locControl!);
                    il.EmitCall(OpCodes.Call, Avx2Method("PermuteVar8x32", vecLane, locControl!.LocalType), null);
                    return;
                default:
                {
                    var vecByte = VectorMethodCache.V(256, typeof(byte));
                    var vecUlong = VectorMethodCache.V(256, typeof(ulong));
                    if (lane != typeof(byte))
                        il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, lane, typeof(byte)), null);
                    il.Emit(OpCodes.Ldloc, locControl!);
                    il.EmitCall(OpCodes.Call, Avx2Method("Shuffle", vecByte, vecByte), null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(byte), typeof(ulong)), null);
                    il.Emit(OpCodes.Ldc_I4, 0x4E);
                    il.EmitCall(OpCodes.Call, Avx2Method("Permute4x64", vecUlong, typeof(byte)), null);
                    il.EmitCall(OpCodes.Call, VectorMethodCache.As(256, typeof(ulong), lane), null);
                    return;
                }
            }
        }

        /// <summary>An <see cref="Avx2"/> overload by name and exact parameter types.</summary>
        /// <param name="name">The method name.</param>
        /// <param name="parameterTypes">The overload's parameter types.</param>
        /// <returns>The method.</returns>
        /// <exception cref="InvalidOperationException">No such overload.</exception>
        private static MethodInfo Avx2Method(string name, params Type[] parameterTypes)
            => typeof(Avx2).GetMethod(name, parameterTypes)
               ?? throw new InvalidOperationException($"Avx2.{name}({string.Join(", ", (object[])parameterTypes)}) not found");

        /// <summary>
        /// Decode a reversed-run pattern number into one mode per operand (<see cref="RunModeC"/>,
        /// <see cref="RunModeR"/>, <see cref="RunModeS"/>): digit <c>op</c> in base 3, the output's digit last.
        /// </summary>
        /// <param name="pattern">The pattern number, <c>0 .. 3^count - 1</c>.</param>
        /// <param name="count">The operand count the pattern covers.</param>
        /// <returns>One mode per operand.</returns>
        private static int[] DecodeRunPattern(int pattern, int count)
        {
            var modes = new int[count];
            for (int op = 0; op < count; op++)
            {
                modes[op] = pattern % 3;
                pattern /= 3;
            }
            return modes;
        }

        /// <summary>
        /// Emit the per-call pattern computation of a reversed-run dispatch: for each of <paramref name="count"/>
        /// operands, a byte stride of <c>+elemSize</c> is digit 0 (C), <c>−elemSize</c> digit 1 (R) and 0 digit 2 (S,
        /// only where <paramref name="allowBroadcast"/> says so); any other stride jumps to <paramref name="lblMismatch"/>.
        /// The digits are summed into an int local, the operand's weight being <c>3^op</c>.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="strideLocals">The operands' byte strides (the first <paramref name="count"/> are read).</param>
        /// <param name="count">How many operands the pattern covers.</param>
        /// <param name="elemSize">The shared element size in bytes.</param>
        /// <param name="allowBroadcast">Per operand: whether stride 0 is a mode (inputs) or a mismatch (an output).</param>
        /// <param name="lblMismatch">Where a stride outside the modes goes (the kernel's next route).</param>
        /// <returns>The pattern local.</returns>
        /// <remarks>2 to 3 compares per operand, once per CALL; the loops run inside the stage the pattern picks.</remarks>
        private static LocalBuilder EmitRunPattern(
            ILGenerator il, LocalBuilder[] strideLocals, int count, long elemSize, bool[] allowBroadcast, Label lblMismatch)
        {
            var locPattern = il.DeclareLocal(typeof(int));
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locPattern);
            int weight = 1;
            for (int op = 0; op < count; op++, weight *= 3)
            {
                var lblNext = il.DefineLabel();
                var lblReversed = il.DefineLabel();
                il.Emit(OpCodes.Ldloc, strideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, elemSize);
                il.Emit(OpCodes.Beq, lblNext);                 // C: digit 0, nothing to add
                il.Emit(OpCodes.Ldloc, strideLocals[op]);
                il.Emit(OpCodes.Ldc_I8, -elemSize);
                il.Emit(OpCodes.Beq, lblReversed);
                if (!allowBroadcast[op])
                {
                    il.Emit(OpCodes.Br, lblMismatch);
                }
                else
                {
                    il.Emit(OpCodes.Ldloc, strideLocals[op]);
                    il.Emit(OpCodes.Ldc_I8, 0L);
                    il.Emit(OpCodes.Bne_Un, lblMismatch);
                    EmitAddToLocal(il, locPattern, RunModeS * weight);
                    il.Emit(OpCodes.Br, lblNext);
                }
                il.MarkLabel(lblReversed);
                EmitAddToLocal(il, locPattern, RunModeR * weight);
                il.MarkLabel(lblNext);
            }
            return locPattern;
        }

        /// <summary>Emit <c>local += value</c> for an int local.</summary>
        /// <param name="il">The generator.</param>
        /// <param name="local">The int local.</param>
        /// <param name="value">The constant to add.</param>
        private static void EmitAddToLocal(ILGenerator il, LocalBuilder local, int value)
        {
            il.Emit(OpCodes.Ldloc, local);
            il.Emit(OpCodes.Ldc_I4, value);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, local);
        }

        // ---------------------------------------------------------------------
        // 1-D shell (NDInnerLoopFunc)
        // ---------------------------------------------------------------------

        /// <summary>
        /// The 1-D shell's reversed-run route, emitted where its contiguous / broadcast checks bail and before the
        /// gather: compute the call's pattern over every operand (inputs C / R / S, the output C / R) and forward to
        /// the pattern's stage (<see cref="GenerateReversedRunStage"/>) when it holds a reversed operand. A pattern
        /// without one, or any other stride, continues at <paramref name="lblNotReversed"/> — the gather / scalar
        /// route, unchanged.
        /// </summary>
        /// <param name="il">The kernel's generator (its prologue loaded the stride locals).</param>
        /// <param name="operandTypes">Inputs then the output: one shared dtype <see cref="CanReverseLanes"/> approved.</param>
        /// <param name="strideLocals">Per-operand byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body (the tail).</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages for JIT diagnostics).</param>
        /// <param name="lblNotReversed">The kernel's next route: the gather dispatch when it has one, else the scalar strided loop.</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        /// <remarks>
        /// Every pattern with a reversed operand gets its stage's IL emitted here, eagerly (at most 14 for a binary
        /// kernel), because a <c>call</c> needs its target to exist; the JIT compiles a stage only on its first call.
        /// Reached only after the shell's own contiguous and broadcast loops declined the call, so the patterns
        /// those loops serve (and the float add / multiply NaN region rule they carry) are never taken here.
        /// </remarks>
        private static void EmitReversedRunDispatch1D(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] strideLocals,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string cacheKey, Label lblNotReversed, Label lblEnd)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            long elemSize = GetTypeSize(operandTypes[nIn]);

            var allowBroadcast = new bool[nOp];
            for (int op = 0; op < nIn; op++)
                allowBroadcast[op] = true;              // the output is never broadcast along the loop
            var locPattern = EmitRunPattern(il, strideLocals, nOp, elemSize, allowBroadcast, lblNotReversed);

            int patterns = 1;
            for (int op = 0; op < nOp; op++)
                patterns *= 3;
            var targets = new Label[patterns];
            var stageLabels = new Label?[patterns];
            for (int p = 0; p < patterns; p++)
            {
                var modes = DecodeRunPattern(p, nOp);
                bool anyReversed = Array.IndexOf(modes, RunModeR) >= 0;
                if (anyReversed && modes[nIn] != RunModeS)
                {
                    stageLabels[p] = il.DefineLabel();
                    targets[p] = stageLabels[p]!.Value;
                }
                else
                {
                    targets[p] = lblNotReversed;
                }
            }

            il.Emit(OpCodes.Ldloc, locPattern);
            il.Emit(OpCodes.Switch, targets);
            il.Emit(OpCodes.Br, lblNotReversed);

            for (int p = 0; p < patterns; p++)
            {
                if (stageLabels[p] is not { } label)
                    continue;
                il.MarkLabel(label);
                var stage = GenerateReversedRunStage(operandTypes, DecodeRunPattern(p, nOp), vectorBody, scalarBody,
                    $"NDInnerLoop_{Sanitize(cacheKey)}_rv{p}");
                EmitForwardToStage(il, stage, argCount: 4);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One 1-D reversed-run stage: the <see cref="NumSharp.Backends.Iteration.NDInnerLoopFunc"/> signature, a prologue
        /// that snapshots the pointers (the strides are fixed by the pattern), and <see cref="EmitReversedRunLoop"/>.
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (one shared dtype).</param>
        /// <param name="modes">Per operand its mode (the output's last); at least one is <see cref="RunModeR"/>.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage, ready to be the target of a <c>call</c> (compiled on its first call).</returns>
        private static DynamicMethod GenerateReversedRunStage(
            NPTypeCode[] operandTypes, int[] modes, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string name)
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
            EmitReversedRunLoop(il, operandTypes, ptrLocals, modes, vectorBody, scalarBody);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>
        /// Emit the reversed-run loop of one pattern: <see cref="EmitSimdContigLoop"/>'s 4x-unrolled vector loop,
        /// 1-vector remainder and scalar tail, where a C operand is addressed <c>ptr + i·e</c>, an R operand loads
        /// (stores) the V elements ending at its logical element i with one contiguous access and a lane reversal,
        /// and an S input is read once before the loop.
        /// </summary>
        /// <param name="il">The stage's generator.</param>
        /// <param name="operandTypes">Inputs then the output (one shared dtype <see cref="CanReverseLanes"/> approved).</param>
        /// <param name="ptrLocals">Per-operand pointers at logical element 0.</param>
        /// <param name="modes">Per operand its mode, the output's last (C or R).</param>
        /// <param name="vectorBody">The op's vector body: N input vectors → one output vector.</param>
        /// <param name="scalarBody">The op's scalar body: N input values → one output value.</param>
        /// <remarks>
        /// An R operand is addressed through <c>j = −i</c>, kept beside <c>i</c>: its group at offset <c>o</c> starts at
        /// <c>ptr + (j − o − (V−1))·e</c>, which the JIT folds into one base + index·scale + displacement address (an
        /// address built from <c>ptr − (i + o)·e</c> would cost a negate per load). Lane k of the reversed load is
        /// logical element <c>i + o + k</c>, the element the gather or scalar route gave that lane, so the vector body
        /// computes the same element pairs.
        /// </remarks>
        private static void EmitReversedRunLoop(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, int[] modes,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody)
        {
            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long vectorCount = GetVectorCount(outType);
            long unrollStep = vectorCount * 4;
            var vecType = VectorMethodCache.V(VectorBits, GetSimdLaneType(outType));

            var locControl = EmitReverseControl(il, outType);

            // Broadcast inputs: their one value, read once, as a scalar (the tail) and as a vector (every group).
            var locVScalar = new LocalBuilder?[nIn];
            var locScalarVal = new LocalBuilder?[nIn];
            for (int op = 0; op < nIn; op++)
            {
                if (modes[op] != RunModeS) continue;
                locVScalar[op] = il.DeclareLocal(vecType);
                locScalarVal[op] = il.DeclareLocal(GetClrType(operandTypes[op]));
                il.Emit(OpCodes.Ldloc, ptrLocals[op]);
                EmitLoadIndirect(il, operandTypes[op]);
                il.Emit(OpCodes.Dup);
                il.Emit(OpCodes.Stloc, locScalarVal[op]!);
                EmitVectorCreate(il, operandTypes[op]);
                il.Emit(OpCodes.Stloc, locVScalar[op]!);
            }

            var locI = il.DeclareLocal(typeof(long));
            var locJ = il.DeclareLocal(typeof(long));          // always −i
            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));
            var locOutVal = il.DeclareLocal(GetClrType(outType));

            var lblUnroll = il.DefineLabel();
            var lblUnrollEnd = il.DefineLabel();
            var lblRem = il.DefineLabel();
            var lblRemEnd = il.DefineLabel();
            var lblTail = il.DefineLabel();
            var lblTailEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep; vectorEnd = count - vectorCount; i = j = 0
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // === 4x UNROLLED SIMD LOOP ===
            il.MarkLabel(lblUnroll);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollEnd);
            for (int u = 0; u < 4; u++)
                EmitReversedRunGroup(il, operandTypes, ptrLocals, modes, locVScalar, locControl, locI, locJ,
                    u * vectorCount, vectorBody);
            EmitStepIndices(il, locI, locJ, unrollStep);
            il.Emit(OpCodes.Br, lblUnroll);
            il.MarkLabel(lblUnrollEnd);

            // === REMAINDER SIMD LOOP (0..3 vectors) ===
            il.MarkLabel(lblRem);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemEnd);
            EmitReversedRunGroup(il, operandTypes, ptrLocals, modes, locVScalar, locControl, locI, locJ, 0, vectorBody);
            EmitStepIndices(il, locI, locJ, vectorCount);
            il.Emit(OpCodes.Br, lblRem);
            il.MarkLabel(lblRemEnd);

            // === SCALAR TAIL: C at ptr + i·e, R at ptr + j·e, S its hoisted value ===
            il.MarkLabel(lblTail);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblTailEnd);
            for (int op = 0; op < nIn; op++)
            {
                if (locScalarVal[op] is { } hoisted)
                {
                    il.Emit(OpCodes.Ldloc, hoisted);
                    continue;
                }
                EmitRunElementAddr(il, ptrLocals[op], modes[op] == RunModeR ? locJ : locI, 0, elemSize);
                EmitLoadIndirect(il, operandTypes[op]);
            }
            scalarBody(il);
            il.Emit(OpCodes.Stloc, locOutVal);
            EmitRunElementAddr(il, ptrLocals[nIn], modes[nIn] == RunModeR ? locJ : locI, 0, elemSize);
            il.Emit(OpCodes.Ldloc, locOutVal);
            EmitStoreIndirect(il, outType);
            EmitStepIndices(il, locI, locJ, 1);
            il.Emit(OpCodes.Br, lblTail);
            il.MarkLabel(lblTailEnd);
        }

        /// <summary>
        /// One vector group of <see cref="EmitReversedRunLoop"/> at logical elements <c>i + offset .. i + offset + V − 1</c>:
        /// load each input (C: one load at <c>ptr + (i+offset)·e</c>; R: one load of the V elements ending at
        /// <c>i + offset</c>, then a lane reversal; S: its hoisted vector), run the vector body, store the result
        /// (C: at <c>ptr + (i+offset)·e</c>; R: reversed, at the V elements ending at <c>i + offset</c>).
        /// </summary>
        /// <param name="il">The stage's generator.</param>
        /// <param name="operandTypes">Inputs then the output.</param>
        /// <param name="ptrLocals">Per-operand pointers at logical element 0.</param>
        /// <param name="modes">Per operand its mode.</param>
        /// <param name="locVScalar">Per input the hoisted vector of an S input (null otherwise).</param>
        /// <param name="locControl">The lane-reversal control (null for 8-byte lanes).</param>
        /// <param name="locI">The logical element index i.</param>
        /// <param name="locJ">−i.</param>
        /// <param name="offset">The group's element offset from i (a multiple of V).</param>
        /// <param name="vectorBody">The op's vector body.</param>
        private static void EmitReversedRunGroup(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] ptrLocals, int[] modes, LocalBuilder?[] locVScalar,
            LocalBuilder? locControl, LocalBuilder locI, LocalBuilder locJ, long offset, Action<ILGenerator> vectorBody)
        {
            int nIn = operandTypes.Length - 1;
            NPTypeCode outType = operandTypes[nIn];
            int elemSize = GetTypeSize(outType);
            long lastLane = GetVectorCount(outType) - 1;

            for (int op = 0; op < nIn; op++)
            {
                if (locVScalar[op] is { } hoisted)
                {
                    il.Emit(OpCodes.Ldloc, hoisted);
                    continue;
                }
                if (modes[op] == RunModeR)
                {
                    // The group's lowest address is element i + offset + V - 1: ptr + (j - offset - (V-1))·e.
                    EmitRunElementAddr(il, ptrLocals[op], locJ, -(offset + lastLane), elemSize);
                    EmitVectorLoad(il, operandTypes[op]);
                    EmitReverseLanes(il, operandTypes[op], locControl);
                }
                else
                {
                    EmitRunElementAddr(il, ptrLocals[op], locI, offset, elemSize);
                    EmitVectorLoad(il, operandTypes[op]);
                }
            }

            vectorBody(il);

            // [out_vec] → [out_vec, dest] for the store.
            if (modes[nIn] == RunModeR)
            {
                EmitReverseLanes(il, outType, locControl);
                EmitRunElementAddr(il, ptrLocals[nIn], locJ, -(offset + lastLane), elemSize);
            }
            else
            {
                EmitRunElementAddr(il, ptrLocals[nIn], locI, offset, elemSize);
            }
            EmitVectorStore(il, outType);
        }

        /// <summary>
        /// Push <c>basePtr + (index + displacement) · elemSize</c>: a C operand's element with <paramref name="locIndex"/>
        /// = i and a displacement ≥ 0, an R operand's with <paramref name="locIndex"/> = j = −i and a displacement ≤ 0
        /// (for its vector groups, the lowest address of the group).
        /// </summary>
        /// <param name="il">The generator.</param>
        /// <param name="basePtr">The operand's pointer at logical element 0.</param>
        /// <param name="locIndex">i or j, a long local.</param>
        /// <param name="displacement">A constant element displacement.</param>
        /// <param name="elemSize">The element size in bytes.</param>
        private static void EmitRunElementAddr(ILGenerator il, LocalBuilder basePtr, LocalBuilder locIndex, long displacement, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locIndex);
            if (displacement != 0)
            {
                il.Emit(OpCodes.Ldc_I8, displacement);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        // ---------------------------------------------------------------------
        // 2-D block kernel (ND2DElementwiseKernel)
        // ---------------------------------------------------------------------

        /// <summary>
        /// The 2-D block kernel's reversed-run route, reached when its C / S checks miss (the output row is already
        /// known contiguous): compute the inputs' pattern (C / R / S per input row) and forward to the pattern's stage
        /// (<see cref="Generate2DReversedRunStage"/>) when a row is reversed; every other pattern, and any other
        /// inner stride, runs the kernel's scalar block as before.
        /// </summary>
        /// <param name="il">The kernel's generator.</param>
        /// <param name="operandTypes">Inputs then the output (one shared dtype <see cref="CanReverseLanes"/> approved).</param>
        /// <param name="innerStrideLocals">Per-operand inner byte strides.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="cacheKey">The kernel's cache key (names the stages).</param>
        /// <param name="lblScalar">The kernel's scalar 2-D block.</param>
        /// <param name="lblEnd">The kernel's return label.</param>
        /// <remarks>
        /// The NDIter gate admits a reversed input row only for kernels this route serves
        /// (<c>NDIterRef.Is2DElementwiseShape</c> with reversed rows allowed), so the scalar block is reached here
        /// only by direct callers with other strides.
        /// </remarks>
        private static void Emit2DReversedRunDispatch(
            ILGenerator il, NPTypeCode[] operandTypes, LocalBuilder[] innerStrideLocals,
            Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string cacheKey, Label lblScalar, Label lblEnd)
        {
            int nIn = operandTypes.Length - 1;
            long elemSize = GetTypeSize(operandTypes[nIn]);
            var allowBroadcast = new bool[nIn];
            for (int op = 0; op < nIn; op++)
                allowBroadcast[op] = true;
            var locPattern = EmitRunPattern(il, innerStrideLocals, nIn, elemSize, allowBroadcast, lblScalar);

            int patterns = 1;
            for (int op = 0; op < nIn; op++)
                patterns *= 3;
            var targets = new Label[patterns];
            var stageLabels = new Label?[patterns];
            for (int p = 0; p < patterns; p++)
            {
                if (Array.IndexOf(DecodeRunPattern(p, nIn), RunModeR) >= 0)
                {
                    stageLabels[p] = il.DefineLabel();
                    targets[p] = stageLabels[p]!.Value;
                }
                else
                {
                    targets[p] = lblScalar;
                }
            }

            il.Emit(OpCodes.Ldloc, locPattern);
            il.Emit(OpCodes.Switch, targets);
            il.Emit(OpCodes.Br, lblScalar);

            for (int p = 0; p < patterns; p++)
            {
                if (stageLabels[p] is not { } label)
                    continue;
                il.MarkLabel(label);
                var stage = Generate2DReversedRunStage(operandTypes, DecodeRunPattern(p, nIn), vectorBody, scalarBody,
                    $"ND2DLoop_{Sanitize(cacheKey)}_rv{p}");
                EmitForwardToStage(il, stage, argCount: 5);
                il.Emit(OpCodes.Br, lblEnd);
            }
        }

        /// <summary>
        /// One 2-D reversed-run stage: the <see cref="ND2DElementwiseKernel"/> signature, a prologue that snapshots the
        /// pointers and outer strides, and <see cref="Emit2DSimdBlock"/> with the pattern's broadcast and reversed
        /// input rows (its row-shape dispatch and masked tails included).
        /// </summary>
        /// <param name="operandTypes">Inputs then the output (one shared dtype).</param>
        /// <param name="inputModes">Per input its mode (<see cref="RunModeC"/> / <see cref="RunModeR"/> / <see cref="RunModeS"/>); at least one R.</param>
        /// <param name="vectorBody">The op's vector body.</param>
        /// <param name="scalarBody">The op's scalar body.</param>
        /// <param name="name">The stage's method name.</param>
        /// <returns>The stage (compiled on its first call).</returns>
        private static DynamicMethod Generate2DReversedRunStage(
            NPTypeCode[] operandTypes, int[] inputModes, Action<ILGenerator> vectorBody, Action<ILGenerator> scalarBody, string name)
        {
            var dm = new DynamicMethod(
                name: name,
                returnType: typeof(void),
                parameterTypes: new[] { typeof(void**), typeof(long*), typeof(long), typeof(long*), typeof(long) },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);
            var il = dm.GetILGenerator();

            int nOp = operandTypes.Length;
            int nIn = nOp - 1;
            var ptrLocals = new LocalBuilder[nOp];
            var outerStrideLocals = new LocalBuilder[nOp];
            for (int op = 0; op < nOp; op++)
            {
                ptrLocals[op] = il.DeclareLocal(typeof(byte*));
                outerStrideLocals[op] = il.DeclareLocal(typeof(long));
            }
            EmitLoadDataPtrs2D(il, nOp, ptrLocals);
            EmitLoadStrideArray(il, Arg2DOuterStrides, nOp, outerStrideLocals);

            var isScalar = new bool[nIn];
            var isReversed = new bool[nIn];
            for (int op = 0; op < nIn; op++)
            {
                isScalar[op] = inputModes[op] == RunModeS;
                isReversed[op] = inputModes[op] == RunModeR;
            }
            Emit2DSimdBlock(il, operandTypes, ptrLocals, outerStrideLocals, isScalar, vectorBody, scalarBody, isReversed);
            il.Emit(OpCodes.Ret);
            return dm;
        }

        /// <summary>Emit <c>i += step; j -= step</c> (j stays −i).</summary>
        /// <param name="il">The generator.</param>
        /// <param name="locI">i.</param>
        /// <param name="locJ">j = −i.</param>
        /// <param name="step">The element step.</param>
        private static void EmitStepIndices(ILGenerator il, LocalBuilder locI, LocalBuilder locJ, long step)
        {
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, step);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, step);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locJ);
        }
    }
}
