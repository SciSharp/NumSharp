using System;
using System.Reflection.Emit;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.NaNOperandOrder.cs — which NaN an elementwise float
// add / multiply returns when BOTH operands are NaN (bit-exact with NumPy 2.4.2)
// =============================================================================
//
// x86 SSE/AVX returns the FIRST source operand (quieted) when both inputs of an
// add/mul are NaN, so the payload (and sign) that survives is decided by the
// operand order the COMPILER emits. NumPy's win-amd64 wheel is MSVC 19.44, and
// its float32/float64 loops (loops_arithm_fp.dispatch.c.src) were probed per
// path (2026-10-03, lengths 1-1000 across the scalar, SIMD-body and tail regions,
// quiet and signalling NaNs):
//
//   * add / multiply, every call whose inputs are both arrays (contiguous,
//     strided, reversed, any length) and every scalar-loop call (len <= 2 *
//     lanes, or a non-contiguous operand): the SECOND operand's NaN wins —
//     MSVC put in2 first (vaddpd/vmulpd dst, b, a), both in the SIMD body and in
//     the scalar loop.
//   * add / multiply with ONE stride-0 (broadcast) input, the other input and
//     the output contiguous, len > 2 * lanes (NumPy's SIMD scalar path): the
//     unrolled body [0, len - len % (2*lanes)) keeps the BROADCAST operand's NaN
//     (the loop-invariant register is the first source, the array load folds in
//     as the memory operand), the tail keeps the ARRAY operand's NaN.
//   * subtract / divide: the first operand's NaN everywhere (non-commutative, so
//     no compiler may swap them) — RyuJIT already agrees, nothing to do.
//   * NOT these rules: np.float64 scalar math and Python floats keep the FIRST
//     operand's NaN, and the sum/prod reductions and cumsum/cumprod scans run
//     their own compiled orders (pairwise; the new element per step). So the
//     emitters below are opt-in per call site (the ufunc kernels), never a change
//     of the 3-argument EmitScalarOperation / EmitVectorOperation the reduction
//     folds use.
//   * The linux-x86_64 GCC wheel orders the operands DIFFERENTLY on the same CPU
//     (its contiguous SIMD body keeps the first operand's NaN, its broadcast loop
//     the broadcast operand's everywhere), so this is a win-amd64 pin, like the
//     float16 and complex NaN-sign contracts.
//
// RyuJIT may swap the operands of a commutative intrinsic for register
// allocation (observed — FLOAT16_DESIGN.md trap #4), so instruction order can
// never encode the rule. NaNOperandOrder.NumPyUfunc makes it order-INDEPENDENT:
//   r = (isNaN(b) ? +0 : a) OP b
// When b is NaN the only NaN operand is b, so r = quiet(b) whichever operand the
// instruction takes first; otherwise the op is exactly a OP b (a's NaN, or the
// default NaN of inf-inf / 0*inf, both order-independent). Vector cost: one
// compare + one AND per vector; scalar cost: one ordered compare + a branch that
// is never taken on NaN-free data.
//
// The broadcast rule depends on WHERE an element sits inside NumPy's call, so the
// kernels keep the plain instruction while the broadcast value is not NaN (then
// both orders agree) and run EmitBroadcastNaNRegion — a cold path that writes
// NumPy's exact NaN per region — only when it is NaN. Whether a NumSharp kernel
// call coincides with a NumPy inner-loop call is a property of the whole
// iteration, decided by the dispatcher and passed as BroadcastNaNRule.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    ///     Which operand's NaN a float32/float64 <c>add</c> or <c>multiply</c> kernel returns when BOTH operands are NaN.
    ///     Only the NaN payload and sign are affected; every non-NaN result is identical under both orders.
    /// </summary>
    internal enum NaNOperandOrder : byte
    {
        /// <summary>
        ///     The bare instruction: whichever NaN the x86 instruction RyuJIT happened to emit keeps. Fast, but the
        ///     surviving payload can differ between the SIMD body and the scalar tail of ONE kernel. Kept for callers
        ///     that model something other than NumPy's ufunc loops (reduction folds, NumPy scalar math).
        /// </summary>
        Unpinned = 0,

        /// <summary>
        ///     NumPy 2.4.2's elementwise loop rule (win-amd64 MSVC wheel): <c>add</c>/<c>multiply</c> return the SECOND
        ///     operand's NaN, <c>subtract</c>/<c>divide</c> the first. Costs a compare and an AND per vector (a never-taken
        ///     branch per scalar element) for add/multiply; every other op/dtype emits exactly the unpinned instruction.
        /// </summary>
        NumPyUfunc = 1,
    }

    /// <summary>
    ///     The <c>auxdata</c> cookie a binary ufunc route hands its inner-loop kernel, describing how NumPy splits the same
    ///     iteration into inner-loop calls — the one fact the broadcast-NaN region rule needs and a cached kernel cannot know.
    ///     Read only on the cold path that runs when a stride-0 operand's value is NaN; a <see langword="null"/> cookie means
    ///     <c>{ SplitPerCall = 0, BufferChunk = 0 }</c> (the second-operand rule).
    /// </summary>
    /// <remarks>
    ///     The layout is ABI: the emitted kernels read <see cref="SplitPerCall"/> at byte offset 0 and <see cref="BufferChunk"/>
    ///     at byte offset 8. The struct must outlive the iteration that receives its address, so callers keep it in a local of
    ///     the method that drives the iterator.
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    internal struct BroadcastNaNRule
    {
        /// <summary>
        ///     1 when every kernel call of this iteration is exactly one NumPy inner-loop call (NumPy's trivial loop, an
        ///     unbuffered 1-D iterator, or one mask-true run of either) whose array input and output are contiguous, so
        ///     NumPy's SIMD broadcast region split applies per call; 0 when NumPy's calls differ or take the scalar loop,
        ///     where the second operand's NaN wins (the rule <see cref="NaNOperandOrder.NumPyUfunc"/> already encodes).
        /// </summary>
        public int SplitPerCall;

        /// <summary>Padding that keeps <see cref="BufferChunk"/> at byte offset 8; always 0.</summary>
        public int Reserved;

        /// <summary>
        ///     When greater than 0, NumPy runs each kernel call as consecutive calls of this many elements (its buffer
        ///     size, because one operand needs a dtype cast and is buffered), the last one shorter; 0 = no chunking.
        /// </summary>
        public long BufferChunk;

        /// <summary>
        ///     The cookie for an iteration NumPy runs as one call per NumSharp kernel call.
        /// </summary>
        /// <param name="bufferChunk">NumPy's buffer size when an operand is cast (see <see cref="BufferChunk"/>), else 0.</param>
        /// <returns>A rule with <see cref="SplitPerCall"/> = 1.</returns>
        internal static BroadcastNaNRule PerCall(long bufferChunk = 0)
            => new BroadcastNaNRule { SplitPerCall = 1, BufferChunk = bufferChunk };
    }

    public static partial class DirectILKernelGenerator
    {
        /// <summary>
        ///     NumPy's SIMD broadcast block for float32, in elements: <c>2 * npyv_nlanes_f32</c> of the
        ///     <c>loops_arithm_fp</c> target this CPU dispatches (see <see cref="NumPyBroadcastSimdBlock"/>).
        /// </summary>
        private static readonly int s_numpyBroadcastBlockF32 = ComputeNumPyBroadcastSimdBlock(4);

        /// <summary>
        ///     NumPy's SIMD broadcast block for float64, in elements: <c>2 * npyv_nlanes_f64</c>.
        /// </summary>
        private static readonly int s_numpyBroadcastBlockF64 = ComputeNumPyBroadcastSimdBlock(8);

        /// <summary>
        ///     Whether <paramref name="op"/> at loop dtype <paramref name="type"/> has an operand-order-dependent NaN result
        ///     that <see cref="NaNOperandOrder.NumPyUfunc"/> pins: float32/float64 <c>add</c> and <c>multiply</c>.
        /// </summary>
        /// <remarks>
        ///     float16 already pins the rule inside its widen-compute-narrow kernels (Binary.Arith.Half.cs); complex128 follows
        ///     a different, unroll-parity-dependent MSVC pattern that these emitters do not model; subtract/divide cannot be
        ///     swapped by any compiler; integer, bool and decimal have no NaN.
        /// </remarks>
        /// <param name="op">The binary op.</param>
        /// <param name="type">The loop (result) dtype the op runs at.</param>
        /// <returns>True when the NumPy rule needs an explicit emission for this op and dtype.</returns>
        internal static bool HasNaNOperandOrder(BinaryOp op, NPTypeCode type)
            => (op == BinaryOp.Add || op == BinaryOp.Multiply)
               && (type == NPTypeCode.Single || type == NPTypeCode.Double);

        /// <summary>
        ///     NumPy's SIMD broadcast block for <paramref name="type"/>, in elements: <c>2 * npyv_nlanes</c> of the
        ///     <c>loops_arithm_fp</c> dispatch target. A broadcast call longer than this takes NumPy's SIMD scalar path, whose
        ///     unrolled body covers <c>[0, len - len % block)</c>; a call of at most this length takes the scalar loop.
        /// </summary>
        /// <remarks>
        ///     NumPy 2.4.2 builds <c>loops_arithm_fp</c> for X86_V3 (AVX2, 32-byte vectors) and X86_V2 (SSE, 16-byte) only —
        ///     no AVX-512 target — so an AVX-512 host still runs the 32-byte loop; ARM's ASIMD/NEON vectors are 16 bytes. The
        ///     value depends on the CPU, not on NumSharp's own <see cref="VectorBits"/>.
        /// </remarks>
        /// <param name="type">float32 or float64.</param>
        /// <returns>The block length in elements (8 or 16 on an AVX2 host for float64 / float32).</returns>
        internal static int NumPyBroadcastSimdBlock(NPTypeCode type)
            => type == NPTypeCode.Single ? s_numpyBroadcastBlockF32 : s_numpyBroadcastBlockF64;

        /// <summary>
        ///     Computes <c>2 * vectorBytes / itemSize</c> for the <c>loops_arithm_fp</c> target NumPy would dispatch here.
        /// </summary>
        /// <param name="itemSize">4 (float32) or 8 (float64).</param>
        /// <returns>The block length in elements.</returns>
        private static int ComputeNumPyBroadcastSimdBlock(int itemSize)
        {
            // X86_V3 = AVX2 + FMA3 + BMI1/2 + LZCNT (+ F16C, MOVBE, which .NET does not expose; every CPU with the
            // rest has them). Anything older that still runs NumPy 2.x is X86_V2 (SSE4.2), and ARM's ASIMD/NEON
            // loop is 16-byte too — so every non-V3 host gets 16-byte vectors.
            bool x86V3 = Avx2.IsSupported && Fma.IsSupported && Bmi1.IsSupported && Bmi2.IsSupported && Lzcnt.IsSupported;
            int vectorBytes = x86V3 ? 32 : 16;
            return 2 * vectorBytes / itemSize;
        }

        /// <summary>
        ///     <see cref="EmitScalarOperation(ILGenerator, BinaryOp, NPTypeCode)"/> with an explicit NaN operand order.
        ///     Stack: <c>[a, b]</c> (both <paramref name="resultType"/>) → <c>[a op b]</c>.
        /// </summary>
        /// <remarks>
        ///     For float32/float64 add/multiply under <see cref="NaNOperandOrder.NumPyUfunc"/> the op runs, and only a NaN
        ///     result whose <c>b</c> is also NaN is replaced by <c>b + b</c> (= quiet(b), whatever operand the instruction takes
        ///     first): one ordered compare per element plus a branch never taken on NaN-free data. Every other combination
        ///     forwards to the 3-argument emitter unchanged.
        /// </remarks>
        /// <param name="il">The IL stream.</param>
        /// <param name="op">The binary op.</param>
        /// <param name="resultType">The loop dtype both operands already have.</param>
        /// <param name="order">The NaN operand order to pin.</param>
        /// <exception cref="NotSupportedException">The op is not supported at <paramref name="resultType"/> (from the 3-argument emitter).</exception>
        internal static void EmitScalarOperation(ILGenerator il, BinaryOp op, NPTypeCode resultType, NaNOperandOrder order)
        {
            if (order != NaNOperandOrder.NumPyUfunc || !HasNaNOperandOrder(op, resultType))
            {
                EmitScalarOperation(il, op, resultType);
                return;
            }

            var clr = GetClrType(resultType);
            var locB = il.DeclareLocal(clr);
            var locR = il.DeclareLocal(clr);
            var done = il.DefineLabel();

            il.Emit(OpCodes.Stloc, locB);                     // [a]
            il.Emit(OpCodes.Ldloc, locB);                     // [a, b]
            il.Emit(op == BinaryOp.Add ? OpCodes.Add : OpCodes.Mul);
            il.Emit(OpCodes.Stloc, locR);                     // r = a op b, either NaN order

            // r == r (ordered) → not NaN → done. Only a NaN result can carry the wrong operand's NaN.
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Ldloc, locR);
            il.Emit(OpCodes.Beq, done);
            // b == b → b is not NaN → r is a's NaN or the invalid-op default NaN; both are already NumPy's.
            il.Emit(OpCodes.Ldloc, locB);
            il.Emit(OpCodes.Ldloc, locB);
            il.Emit(OpCodes.Beq, done);
            // Both may be NaN and NumPy keeps b's: b + b is quiet(b) (identical operands — no order to get wrong).
            il.Emit(OpCodes.Ldloc, locB);
            il.Emit(OpCodes.Ldloc, locB);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locR);

            il.MarkLabel(done);
            il.Emit(OpCodes.Ldloc, locR);                     // [r]
        }

        /// <summary>
        ///     <see cref="EmitVectorOperation(ILGenerator, BinaryOp, NPTypeCode)"/> with an explicit NaN operand order.
        ///     Stack: <c>[va, vb]</c> (<c>Vector{VectorBits}&lt;T&gt;</c>) → <c>[va op vb]</c>.
        /// </summary>
        /// <remarks>
        ///     For float32/float64 add/multiply under <see cref="NaNOperandOrder.NumPyUfunc"/> it emits
        ///     <c>(va &amp; (vb == vb)) op vb</c>: lanes where <c>b</c> is NaN combine <c>+0</c> with <c>b</c>, so the only NaN
        ///     operand is <c>b</c> and the lane is quiet(b) in either instruction order; every other lane is bit-for-bit
        ///     <c>a op b</c>. One compare and one AND per vector. Every other combination forwards unchanged.
        /// </remarks>
        /// <param name="il">The IL stream.</param>
        /// <param name="op">The binary op.</param>
        /// <param name="type">The lane dtype.</param>
        /// <param name="order">The NaN operand order to pin.</param>
        /// <exception cref="NotSupportedException">The op has no vector body at <paramref name="type"/> (from the 3-argument emitter).</exception>
        internal static void EmitVectorOperation(ILGenerator il, BinaryOp op, NPTypeCode type, NaNOperandOrder order)
        {
            if (order != NaNOperandOrder.NumPyUfunc || !HasNaNOperandOrder(op, type))
            {
                EmitVectorOperation(il, op, type);
                return;
            }

            EmitVectorSecondNaNWins(il, op, type, VectorBits);
        }

        /// <summary>
        ///     <see cref="EmitVectorOperationAt(ILGenerator, BinaryOp, NPTypeCode, int)"/> (a width-explicit container, used by
        ///     np.evaluate's mixed-width lanes) with an explicit NaN operand order. Stack: <c>[va, vb]</c> → <c>[va op vb]</c>.
        /// </summary>
        /// <param name="il">The IL stream.</param>
        /// <param name="op">The binary op.</param>
        /// <param name="type">The lane dtype.</param>
        /// <param name="bits">The container width in bits (128, 256 or 512).</param>
        /// <param name="order">The NaN operand order to pin.</param>
        /// <exception cref="NotSupportedException">The op has no vector body at this width (from the 4-argument emitter).</exception>
        internal static void EmitVectorOperationAt(ILGenerator il, BinaryOp op, NPTypeCode type, int bits, NaNOperandOrder order)
        {
            if (order != NaNOperandOrder.NumPyUfunc || !HasNaNOperandOrder(op, type))
            {
                EmitVectorOperationAt(il, op, type, bits);
                return;
            }

            EmitVectorSecondNaNWins(il, op, type, bits);
        }

        /// <summary>
        ///     Emits <c>(va &amp; (vb == vb)) op vb</c> at width <paramref name="bits"/>: NumPy's second-operand NaN for
        ///     float add/multiply, independent of the operand order RyuJIT gives the instruction.
        /// </summary>
        /// <param name="il">The IL stream; stack <c>[va, vb]</c> → <c>[result]</c>.</param>
        /// <param name="op"><see cref="BinaryOp.Add"/> or <see cref="BinaryOp.Multiply"/>.</param>
        /// <param name="type">float32 or float64.</param>
        /// <param name="bits">The container width in bits.</param>
        private static void EmitVectorSecondNaNWins(ILGenerator il, BinaryOp op, NPTypeCode type, int bits)
        {
            var clr = GetClrType(type);
            var locB = il.DeclareLocal(VectorMethodCache.V(bits, clr));

            il.Emit(OpCodes.Stloc, locB);                                                      // [va]
            il.Emit(OpCodes.Ldloc, locB);
            il.Emit(OpCodes.Ldloc, locB);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(bits, clr), null);              // [va, ordered(b)]
            il.EmitCall(OpCodes.Call, VectorMethodCache.Generic(bits, "BitwiseAnd", clr, paramCount: 2), null); // [t]
            il.Emit(OpCodes.Ldloc, locB);                                                      // [t, vb]

            // The arithmetic itself is the unpinned instruction — its operand order no longer matters.
            if (bits == VectorBits)
                EmitVectorOperation(il, op, type);
            else
                EmitVectorOperationAt(il, op, type, bits);
        }

        /// <summary>
        ///     Emits NumPy's NaN-broadcast region rule for one kernel call whose stride-0 (broadcast) operand value is NaN:
        ///     every output element is a NaN, and which one is decided by WHERE the element sits inside NumPy's inner-loop
        ///     call. Branches to <paramref name="done"/> when finished.
        /// </summary>
        /// <remarks>
        ///     <para>
        ///     For each NumPy call of <c>len</c> elements (the whole kernel call, or consecutive <c>BufferChunk</c>-sized pieces
        ///     of it), with <c>block</c> = <see cref="NumPyBroadcastSimdBlock"/>:
        ///     <c>mainEnd = SplitPerCall != 0 &amp;&amp; len &gt; block ? len - len % block : (scalarIsSecond ? len : 0)</c>;
        ///     elements <c>[0, mainEnd)</c> get quiet(s) (the broadcast operand's NaN), the rest get quiet(x) when the array
        ///     element <c>x</c> is NaN and quiet(s) otherwise. With <c>SplitPerCall == 0</c> this is exactly the second-operand
        ///     rule, so the path is also correct for iterations NumPy runs differently.
        ///     </para>
        ///     <para>
        ///     No arithmetic ever combines two DIFFERENT NaNs here (quiet(v) is <c>v + v</c>), so the result does not depend on
        ///     instruction operand order. An output that aliases the array operand is safe: each element is read before its
        ///     own slot is written. With <paramref name="contiguous"/> the constant fill and (when <paramref name="arrayType"/>
        ///     is the loop dtype) the array pass are vectorized; otherwise both are scalar loops over the byte strides.
        ///     </para>
        /// </remarks>
        /// <param name="il">The IL stream (empty stack on entry and exit).</param>
        /// <param name="resultType">The loop dtype: float32 or float64.</param>
        /// <param name="scalar">Local holding the broadcast value, already converted to <paramref name="resultType"/>; it must be NaN.</param>
        /// <param name="arrayBase">Local (<c>byte*</c>) holding the address of the array operand's first element of this call.</param>
        /// <param name="arrayType">The array operand's stored dtype; converted to <paramref name="resultType"/> per element.</param>
        /// <param name="arrayStride">Local (<c>long</c>, bytes) with the array operand's stride, or <see langword="null"/> when <paramref name="contiguous"/>.</param>
        /// <param name="outBase">Local (<c>byte*</c>) holding the address of the output's first element of this call.</param>
        /// <param name="outStride">Local (<c>long</c>, bytes) with the output's stride, or <see langword="null"/> when <paramref name="contiguous"/>.</param>
        /// <param name="pushCount">Pushes the call's element count (<c>long</c>).</param>
        /// <param name="scalarIsSecond">True when the broadcast operand is the op's SECOND input (<c>array op scalar</c>).</param>
        /// <param name="pushSplitPerCall">Pushes <see cref="BroadcastNaNRule.SplitPerCall"/> (<c>int</c>).</param>
        /// <param name="pushBufferChunk">Pushes <see cref="BroadcastNaNRule.BufferChunk"/> (<c>long</c>).</param>
        /// <param name="contiguous">True when both the array operand and the output are element-contiguous (stride == item size).</param>
        /// <param name="done">The label to branch to afterwards.</param>
        /// <exception cref="ArgumentException"><paramref name="resultType"/> is not float32/float64, or a stride local is missing for a strided call.</exception>
        internal static void EmitBroadcastNaNRegion(
            ILGenerator il, NPTypeCode resultType, LocalBuilder scalar,
            LocalBuilder arrayBase, NPTypeCode arrayType, LocalBuilder? arrayStride,
            LocalBuilder outBase, LocalBuilder? outStride,
            Action<ILGenerator> pushCount, bool scalarIsSecond,
            Action<ILGenerator> pushSplitPerCall, Action<ILGenerator> pushBufferChunk,
            bool contiguous, Label done)
        {
            if (resultType != NPTypeCode.Single && resultType != NPTypeCode.Double)
                throw new ArgumentException($"The broadcast NaN region rule is float32/float64 only, not {resultType}.", nameof(resultType));
            if (!contiguous && (arrayStride is null || outStride is null))
                throw new ArgumentException("A strided broadcast NaN region needs both stride locals.", nameof(arrayStride));

            var clr = GetClrType(resultType);
            int resultSize = GetTypeSize(resultType);
            int arraySize = GetTypeSize(arrayType);
            long block = NumPyBroadcastSimdBlock(resultType);
            // The vector array pass reads the array as the loop dtype, so it needs no conversion.
            bool vectorFill = contiguous && VectorBits != 0;
            bool vectorArray = vectorFill && arrayType == resultType;
            long lanes = vectorFill ? GetVectorCount(resultType) : 1;

            var locQs = il.DeclareLocal(clr);
            var locSplit = il.DeclareLocal(typeof(int));
            var locChunk = il.DeclareLocal(typeof(long));
            var locCount = il.DeclareLocal(typeof(long));
            var locStart = il.DeclareLocal(typeof(long));
            var locLen = il.DeclareLocal(typeof(long));
            var locEnd1 = il.DeclareLocal(typeof(long));
            var locEnd2 = il.DeclareLocal(typeof(long));
            var locK = il.DeclareLocal(typeof(long));
            var locX = il.DeclareLocal(clr);
            LocalBuilder? locVqs = vectorFill ? il.DeclareLocal(VectorMethodCache.V(VectorBits, clr)) : null;
            LocalBuilder? locVx = vectorArray ? il.DeclareLocal(VectorMethodCache.V(VectorBits, clr)) : null;

            // qs = s + s: quiet(s) — sign and payload kept, a signalling NaN quieted, exactly what x86 returns for s.
            il.Emit(OpCodes.Ldloc, scalar);
            il.Emit(OpCodes.Ldloc, scalar);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locQs);
            if (vectorFill)
            {
                il.Emit(OpCodes.Ldloc, locQs);
                EmitVectorCreate(il, resultType);
                il.Emit(OpCodes.Stloc, locVqs!);
            }

            pushSplitPerCall(il);
            il.Emit(OpCodes.Stloc, locSplit);
            pushBufferChunk(il);
            il.Emit(OpCodes.Stloc, locChunk);
            pushCount(il);
            il.Emit(OpCodes.Stloc, locCount);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locStart);

            // ── One NumPy call per iteration: [start, start + len) ────────────────────────────────────
            var lblCall = il.DefineLabel();
            il.MarkLabel(lblCall);
            il.Emit(OpCodes.Ldloc, locStart);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Bge, done);

            // len = count - start, capped at the buffer chunk when NumPy buffers a cast operand.
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldloc, locStart);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locLen);
            var lblNoCap = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locChunk);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblNoCap);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldloc, locChunk);
            il.Emit(OpCodes.Ble, lblNoCap);
            il.Emit(OpCodes.Ldloc, locChunk);
            il.Emit(OpCodes.Stloc, locLen);
            il.MarkLabel(lblNoCap);

            // mainEnd: NumPy's SIMD scalar path covers [0, len - len % block) with the broadcast NaN; its scalar loop
            // (len <= block, or a call NumPy does not split this way) keeps the SECOND operand's NaN.
            var lblScalarLoopRule = il.DefineLabel();
            var lblHaveEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locSplit);
            il.Emit(OpCodes.Brfalse, lblScalarLoopRule);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldc_I8, block);
            il.Emit(OpCodes.Ble, lblScalarLoopRule);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Ldc_I8, block);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Br, lblHaveEnd);
            il.MarkLabel(lblScalarLoopRule);
            if (scalarIsSecond)
                il.Emit(OpCodes.Ldloc, locLen);
            else
                il.Emit(OpCodes.Ldc_I8, 0L);
            il.MarkLabel(lblHaveEnd);
            il.Emit(OpCodes.Ldloc, locStart);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locEnd1);                   // absolute end of the broadcast-NaN region
            il.Emit(OpCodes.Ldloc, locStart);
            il.Emit(OpCodes.Ldloc, locLen);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locEnd2);                   // absolute end of this NumPy call
            il.Emit(OpCodes.Ldloc, locStart);
            il.Emit(OpCodes.Stloc, locK);

            // ── Region 1: out[k] = quiet(s) ─────────────────────────────────────────────────────────
            if (vectorFill)
            {
                var lblV1 = il.DefineLabel();
                var lblV1End = il.DefineLabel();
                il.MarkLabel(lblV1);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locEnd1);
                il.Emit(OpCodes.Bgt, lblV1End);
                il.Emit(OpCodes.Ldloc, locVqs!);
                EmitRegionAddress(il, outBase, outStride, locK, resultSize);
                EmitVectorStore(il, resultType);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, lblV1);
                il.MarkLabel(lblV1End);
            }

            var lblS1 = il.DefineLabel();
            var lblS1End = il.DefineLabel();
            il.MarkLabel(lblS1);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locEnd1);
            il.Emit(OpCodes.Bge, lblS1End);
            EmitRegionAddress(il, outBase, outStride, locK, resultSize);
            il.Emit(OpCodes.Ldloc, locQs);
            EmitStoreIndirect(il, resultType);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblS1);
            il.MarkLabel(lblS1End);

            // ── Region 2: out[k] = isNaN(x) ? quiet(x) : quiet(s) — the array operand's NaN wins ───────
            if (vectorArray)
            {
                var lblV2 = il.DefineLabel();
                var lblV2End = il.DefineLabel();
                il.MarkLabel(lblV2);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldloc, locEnd2);
                il.Emit(OpCodes.Bgt, lblV2End);
                EmitRegionAddress(il, arrayBase, arrayStride, locK, arraySize);
                EmitVectorLoad(il, resultType);
                il.Emit(OpCodes.Stloc, locVx!);
                // ConditionalSelect(x == x, quiet(s), x + x): non-NaN lanes take quiet(s), NaN lanes quiet(x).
                il.Emit(OpCodes.Ldloc, locVx!);
                il.Emit(OpCodes.Ldloc, locVx!);
                il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(VectorBits, clr), null);
                il.Emit(OpCodes.Ldloc, locVqs!);
                il.Emit(OpCodes.Ldloc, locVx!);
                il.Emit(OpCodes.Ldloc, locVx!);
                EmitVectorOperation(il, BinaryOp.Add, resultType);
                il.EmitCall(OpCodes.Call, VectorMethodCache.ConditionalSelect(VectorBits, clr), null);
                EmitRegionAddress(il, outBase, outStride, locK, resultSize);
                EmitVectorStore(il, resultType);
                il.Emit(OpCodes.Ldloc, locK);
                il.Emit(OpCodes.Ldc_I8, lanes);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Stloc, locK);
                il.Emit(OpCodes.Br, lblV2);
                il.MarkLabel(lblV2End);
            }

            var lblS2 = il.DefineLabel();
            var lblS2End = il.DefineLabel();
            var lblXIsNaN = il.DefineLabel();
            var lblStore = il.DefineLabel();
            il.MarkLabel(lblS2);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldloc, locEnd2);
            il.Emit(OpCodes.Bge, lblS2End);
            EmitRegionAddress(il, arrayBase, arrayStride, locK, arraySize);
            EmitLoadIndirect(il, arrayType);
            EmitConvertTo(il, arrayType, resultType);
            il.Emit(OpCodes.Stloc, locX);
            EmitRegionAddress(il, outBase, outStride, locK, resultSize);  // [outAddr]
            il.Emit(OpCodes.Ldloc, locX);
            il.Emit(OpCodes.Ldloc, locX);
            il.Emit(OpCodes.Bne_Un, lblXIsNaN);                // unordered → x is NaN
            il.Emit(OpCodes.Ldloc, locQs);                     // [outAddr, quiet(s)]
            il.Emit(OpCodes.Br, lblStore);
            il.MarkLabel(lblXIsNaN);
            il.Emit(OpCodes.Ldloc, locX);
            il.Emit(OpCodes.Ldloc, locX);
            il.Emit(OpCodes.Add);                              // [outAddr, quiet(x)]
            il.MarkLabel(lblStore);
            EmitStoreIndirect(il, resultType);
            il.Emit(OpCodes.Ldloc, locK);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locK);
            il.Emit(OpCodes.Br, lblS2);
            il.MarkLabel(lblS2End);

            il.Emit(OpCodes.Ldloc, locEnd2);
            il.Emit(OpCodes.Stloc, locStart);
            il.Emit(OpCodes.Br, lblCall);
        }

        /// <summary>
        ///     Pushes <c>basePtr + k * stride</c>: the byte stride local when one is given, else the element size
        ///     (an element-contiguous run).
        /// </summary>
        /// <param name="il">The IL stream; pushes one native pointer.</param>
        /// <param name="basePtr">Local holding the run's base address.</param>
        /// <param name="strideBytes">Local holding the byte stride, or <see langword="null"/> for a contiguous run.</param>
        /// <param name="locK">Local holding the element index.</param>
        /// <param name="elemSize">The element size used when <paramref name="strideBytes"/> is null.</param>
        private static void EmitRegionAddress(ILGenerator il, LocalBuilder basePtr, LocalBuilder? strideBytes, LocalBuilder locK, int elemSize)
        {
            il.Emit(OpCodes.Ldloc, basePtr);
            il.Emit(OpCodes.Ldloc, locK);
            if (strideBytes is null)
                il.Emit(OpCodes.Ldc_I8, (long)elemSize);
            else
                il.Emit(OpCodes.Ldloc, strideBytes);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
        }

        /// <summary>
        ///     Emits the pre-loop test of a broadcast kernel: branches to <paramref name="nanPath"/> when the broadcast value in
        ///     <paramref name="scalar"/> is NaN. A non-NaN broadcast value makes every element's result independent of NaN
        ///     operand order, so the caller's hot loop may keep the unpinned instruction.
        /// </summary>
        /// <param name="il">The IL stream (stack unchanged).</param>
        /// <param name="scalar">Local holding the broadcast value at the loop dtype.</param>
        /// <param name="nanPath">The label of the caller's <see cref="EmitBroadcastNaNRegion"/> block.</param>
        internal static void EmitBranchIfNaN(ILGenerator il, LocalBuilder scalar, Label nanPath)
        {
            il.Emit(OpCodes.Ldloc, scalar);
            il.Emit(OpCodes.Ldloc, scalar);
            il.Emit(OpCodes.Bne_Un, nanPath);   // unordered (NaN) → nanPath
        }
    }
}
