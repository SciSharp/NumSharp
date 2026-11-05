using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;

// =============================================================================
// DirectILKernelGenerator.Trace.cs — fused IL kernel for np.trace 2-D / 3-D fast path
// =============================================================================
//
// RESPONSIBILITY:
//   np.trace on a 2-D source is sum(diagonal(a)). Composing those three steps
//   (diagonal view + ascontiguousarray + axis-null sum) pays ~3μs of
//   per-call overhead that dwarfs the actual work for small matrices. This
//   kernel collapses the chain into a single strided walk with inline
//   accumulation into the promoted dtype — no NDArray intermediates between
//   input and the result.
//
//   For the inexact dtypes this walk is ONE of the two loop forms of NumPy's
//   add.reduce over the diagonal view: the ELEMENTWISE form, which NumPy runs
//   when its reduction iterator walks a 3-D source's remaining axis innermost —
//   every diagonal step is `out[o] = out[o] + x[o, k]`, a sequential sum per
//   output from the +0 identity. The other form, one pairwise reduce call per
//   output when the diagonal is innermost (every 2-D trace), is not this
//   kernel: DefaultEngine.TryTraceFast sends it to NumPyAddReduce.AccumulateCall.
//   The integers are exact in any order and decimal has no NumPy loop, so for
//   them this walk serves every layout.
//
// KERNEL (per src dtype, cached forever):
//
//   * TraceKernel(byte* src, long startOff, long diagSize, long byteStride,
//                  long outerSize, long outerSrcStride, long outerDstStride,
//                  byte* dst) → void
//
//   For each i in [0, outerSize):
//       acc = 0;            // accum-type zero
//       p   = src + startOff + i * outerSrcStride;
//       for (j = 0; j < diagSize; j++) {
//           acc += widen(*(srcT*)p);
//           p += byteStride;
//       }
//       *(resultT*)(dst + i * outerDstStride) = convert(acc);
//
//   The (src, accum, result) triple is baked into the IL at emit time:
//
//     bool / byte / sbyte / int16 / uint16 / int32 / int64 / char
//                                  → accum=long,    result=int64
//     uint32 / uint64              → accum=ulong,   result=uint64
//     single                        → accum=single,  result=single
//     double                        → accum=double,  result=double
//                                     (single / double: NumPy's elementwise
//                                      add, so two NaNs keep the ELEMENT's NaN,
//                                      the ufunc loops' second operand —
//                                      EmitScalarOperation's NumPyUfunc order)
//     Half                          → accum=single,  result=Half
//                                     (HALF_add's elementwise loop: the float32
//                                      sum is narrowed to float16 after EVERY
//                                      step, npy_float_to_half, and widened back
//                                      exactly, npy_half_to_float; two NaNs keep
//                                      the element's, as for float32)
//     decimal                       → accum=decimal, result=decimal
//                                     (op_Addition call per element)
//     Complex                       → accum=Complex, result=Complex
//                                     (two inline double accumulators)
//
//   2-D source: outerSize=1, outerSrcStride=outerDstStride=0.
//   3-D source: outerSize=non-axis-dim, outerSrcStride=that-dim-stride*elemBytes,
//               outerDstStride=resultBytes.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Walks the diagonal of one or more 2-D sub-arrays and accumulates each
    /// into the promoted dtype. Per-src dtype kernel; accum / result dtypes
    /// are baked into the IL at emit time.
    /// </summary>
    public unsafe delegate void TraceKernel(
        byte* src, long startOff, long diagSize, long byteStride,
        long outerSize, long outerSrcStride, long outerDstStride, byte* dst);

    public static partial class DirectILKernelGenerator
    {
        internal static readonly ConcurrentDictionary<Type, TraceKernel> _trace = new();

        /// <summary>
        /// IL-emitted singleton per <paramref name="srcType"/>. Returns
        /// <c>null</c> when the dtype has no kernel (no triples are
        /// unsupported in the current implementation; the field exists for
        /// graceful future expansion).
        /// </summary>
        public static TraceKernel GetTraceKernel(Type srcType)
        {
            if (!Enabled)
                return null;

            return _trace.GetOrAdd(srcType, static t =>
            {
                try
                {
                    var info = TraceTypeInfo(t);
                    if (!info.supported)
                        return null;
                    return GenerateTraceKernelIL(t, info.accum, info.result);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTraceKernel({t.Name}): {ex.GetType().Name}: {ex.Message}");
                    return null;
                }
            });
        }

        /// <summary>
        /// Maps src dtype → (accum CLR type, result CLR type, result NPTypeCode,
        /// result-byte-size, supported). Mirrors NumPy's add.reduce default-integer rule
        /// EXACTLY: signed narrow ints and bool widen to int64, but every UNSIGNED lane
        /// (uint8/uint16/uint32 — and Char, NumSharp's uint16 twin) widens to uint64 —
        /// NumPy's trace(uint8) is uint64, and mapping those lanes to int64 silently
        /// changed both the result dtype and the wrap point (caught by the instance
        /// oracle tier's uint8 trace cells, 2026-09-18). Preserves float dtypes; the Half
        /// accumulator is float32, narrowed to float16 after every step (HALF_add's
        /// elementwise loop — a double accumulator rounded once is NOT NumPy's answer).
        /// </summary>
        /// <param name="srcType">The source element CLR type.</param>
        /// <returns>The accumulator and result types, the result dtype and size, and whether a kernel exists.</returns>
        private static (Type accum, Type result, NPTypeCode resultCode, int resultBytes, bool supported)
            TraceTypeInfo(Type srcType)
        {
            if (srcType == typeof(bool) || srcType == typeof(sbyte) ||
                srcType == typeof(short) || srcType == typeof(int) || srcType == typeof(long))
                return (typeof(long), typeof(long), NPTypeCode.Int64, 8, true);

            // Unsigned family (Char ≡ uint16): NumPy's add.reduce widens these to uint64.
            if (srcType == typeof(byte) || srcType == typeof(ushort) || srcType == typeof(char) ||
                srcType == typeof(uint) || srcType == typeof(ulong))
                return (typeof(ulong), typeof(ulong), NPTypeCode.UInt64, 8, true);

            if (srcType == typeof(float))
                return (typeof(float), typeof(float), NPTypeCode.Single, 4, true);

            if (srcType == typeof(double))
                return (typeof(double), typeof(double), NPTypeCode.Double, 8, true);

            if (srcType == typeof(Half))
                return (typeof(float), typeof(Half), NPTypeCode.Half, 2, true);

            if (srcType == typeof(decimal))
                return (typeof(decimal), typeof(decimal), NPTypeCode.Decimal, 16, true);

            if (srcType == typeof(System.Numerics.Complex))
                return (typeof(System.Numerics.Complex), typeof(System.Numerics.Complex), NPTypeCode.Complex, 16, true);

            return (null, null, default, 0, false);
        }

        private static TraceKernel GenerateTraceKernelIL(Type srcType, Type accumType, Type resultType)
        {
            // Complex specialisation: bypass the struct op_Addition call by
            // treating each Complex value as two adjacent doubles and using
            // two double accumulators. op_Addition is a method-call per element
            // (~5ns); the inline two-double walk runs at memory speed (~3.5x
            // faster on 1000-element diagonals).
            if (srcType == typeof(System.Numerics.Complex))
                return GenerateComplexTraceKernelIL();

            var dm = new DynamicMethod(
                name: $"IL_Trace_{srcType.Name}_acc_{accumType.Name}_res_{resultType.Name}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(long),   // 1 startOff
                    typeof(long),   // 2 diagSize
                    typeof(long),   // 3 byteStride
                    typeof(long),   // 4 outerSize
                    typeof(long),   // 5 outerSrcStride
                    typeof(long),   // 6 outerDstStride
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locAcc = il.DeclareLocal(accumType);
            var locP = il.DeclareLocal(typeof(byte*));
            var locJ = il.DeclareLocal(typeof(long));
            var locOuterI = il.DeclareLocal(typeof(long));
            var locOuterBase = il.DeclareLocal(typeof(byte*));
            var locDstAt = il.DeclareLocal(typeof(byte*));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();

            // outerI = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterI);

            // --------- outer loop ---------
            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // outerBase = src + startOff + outerI * outerSrcStride
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuterBase);

            // dstAt = dst + outerI * outerDstStride
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstAt);

            // Init acc — primitives via Ldc+Stloc, struct accum via ldloca+initobj.
            EmitInitAcc(il, locAcc, accumType);

            // p = outerBase
            il.Emit(OpCodes.Ldloc, locOuterBase);
            il.Emit(OpCodes.Stloc, locP);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // --------- inner diag walk ---------
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // acc = acc + widen(*(srcT*)p)
            EmitLoadAndAdd(il, srcType, accumType, locAcc, locP);

            // p += byteStride
            il.Emit(OpCodes.Ldloc, locP);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locP);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // *(resultT*)dstAt = convert(acc, accumType→resultType)
            EmitStoreAccAsResult(il, locAcc, accumType, resultType, locDstAt);

            // outerI++
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuterI);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ret);

            return (TraceKernel)dm.CreateDelegate(typeof(TraceKernel));
        }

        /// <summary>
        /// Complex-specialised kernel. Same delegate signature as the generic
        /// one but the IL body treats each <c>Complex</c> as two adjacent
        /// doubles (matches the struct's <c>{m_real; m_imaginary}</c> layout)
        /// and accumulates with two double locals instead of calling
        /// <c>Complex.op_Addition</c> per element. 3-4x faster than the
        /// op_Addition path on 1000+ element diagonals.
        /// </summary>
        private static TraceKernel GenerateComplexTraceKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_Trace_Complex_inline",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(long),   // 1 startOff
                    typeof(long),   // 2 diagSize
                    typeof(long),   // 3 byteStride
                    typeof(long),   // 4 outerSize
                    typeof(long),   // 5 outerSrcStride
                    typeof(long),   // 6 outerDstStride
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(DirectILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locAccR = il.DeclareLocal(typeof(double));
            var locAccI = il.DeclareLocal(typeof(double));
            var locP = il.DeclareLocal(typeof(byte*));
            var locJ = il.DeclareLocal(typeof(long));
            var locOuterI = il.DeclareLocal(typeof(long));
            var locOuterBase = il.DeclareLocal(typeof(byte*));
            var locDstAt = il.DeclareLocal(typeof(byte*));

            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            var lblInnerHead = il.DefineLabel();
            var lblInnerEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locOuterI);

            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // outerBase = src + startOff + outerI * outerSrcStride
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuterBase);

            // dstAt = dst + outerI * outerDstStride
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 6);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstAt);

            // accR = 0.0; accI = 0.0
            il.Emit(OpCodes.Ldc_R8, 0.0);
            il.Emit(OpCodes.Stloc, locAccR);
            il.Emit(OpCodes.Ldc_R8, 0.0);
            il.Emit(OpCodes.Stloc, locAccI);

            // p = outerBase
            il.Emit(OpCodes.Ldloc, locOuterBase);
            il.Emit(OpCodes.Stloc, locP);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // accR += *(double*)p
            il.Emit(OpCodes.Ldloc, locAccR);
            il.Emit(OpCodes.Ldloc, locP);
            il.Emit(OpCodes.Ldind_R8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locAccR);

            // accI += *(double*)(p + 8)
            il.Emit(OpCodes.Ldloc, locAccI);
            il.Emit(OpCodes.Ldloc, locP);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_R8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locAccI);

            // p += byteStride
            il.Emit(OpCodes.Ldloc, locP);
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locP);

            // j++
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locJ);
            il.Emit(OpCodes.Br, lblInnerHead);

            il.MarkLabel(lblInnerEnd);

            // *(double*)dstAt = accR
            il.Emit(OpCodes.Ldloc, locDstAt);
            il.Emit(OpCodes.Ldloc, locAccR);
            il.Emit(OpCodes.Stind_R8);

            // *(double*)(dstAt + 8) = accI
            il.Emit(OpCodes.Ldloc, locDstAt);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locAccI);
            il.Emit(OpCodes.Stind_R8);

            // outerI++
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuterI);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ret);

            return (TraceKernel)dm.CreateDelegate(typeof(TraceKernel));
        }

        /// <summary>
        /// Initialise the accum local to its dtype's zero. Primitive types get
        /// a Ldc + Stloc; struct accums (decimal, Complex) use initobj which
        /// zero-fills the local in place.
        /// </summary>
        private static void EmitInitAcc(ILGenerator il, LocalBuilder locAcc, Type accumType)
        {
            if (accumType == typeof(decimal) || accumType == typeof(System.Numerics.Complex))
            {
                il.Emit(OpCodes.Ldloca, locAcc);
                il.Emit(OpCodes.Initobj, accumType);
                return;
            }
            if (accumType == typeof(long) || accumType == typeof(ulong))
            {
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Stloc, locAcc);
                return;
            }
            if (accumType == typeof(float))
            {
                il.Emit(OpCodes.Ldc_R4, 0.0f);
                il.Emit(OpCodes.Stloc, locAcc);
                return;
            }
            if (accumType == typeof(double))
            {
                il.Emit(OpCodes.Ldc_R8, 0.0);
                il.Emit(OpCodes.Stloc, locAcc);
                return;
            }
            throw new NotSupportedException($"Trace accum dtype unsupported: {accumType.Name}");
        }

        /// <summary>
        /// Emits the accumulator update for one diagonal element:
        /// <code>acc = acc + widen(*(srcT*)p);</code>
        /// For integer accums uses <see cref="OpCodes.Add"/>; for float /
        /// double accums NumPy's elementwise add with its NaN operand order
        /// (two NaNs keep the element's); for a Half source additionally the
        /// per-step float16 rounding of HALF_add; for struct accums (decimal,
        /// Complex) calls the cached op_Addition method.
        /// </summary>
        /// <param name="il">The IL stream.</param>
        /// <param name="srcType">The source element CLR type.</param>
        /// <param name="accumType">The accumulator CLR type (<see cref="TraceTypeInfo"/>).</param>
        /// <param name="locAcc">The accumulator local, read and written.</param>
        /// <param name="locP">The local holding the current element's address.</param>
        /// <exception cref="NotSupportedException">A source dtype the trace table never produces (from the widening).</exception>
        private static void EmitLoadAndAdd(
            ILGenerator il, Type srcType, Type accumType, LocalBuilder locAcc, LocalBuilder locP)
        {
            if (accumType == typeof(decimal))
            {
                // acc = decimal.op_Addition(acc, *(decimal*)p)
                il.Emit(OpCodes.Ldloc, locAcc);
                il.Emit(OpCodes.Ldloc, locP);
                il.Emit(OpCodes.Ldobj, typeof(decimal));
                il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpAddition, null);
                il.Emit(OpCodes.Stloc, locAcc);
                return;
            }

            if (accumType == typeof(System.Numerics.Complex))
            {
                // acc = Complex.op_Addition(acc, *(Complex*)p)
                il.Emit(OpCodes.Ldloc, locAcc);
                il.Emit(OpCodes.Ldloc, locP);
                il.Emit(OpCodes.Ldobj, typeof(System.Numerics.Complex));
                il.EmitCall(OpCodes.Call, CachedMethods.ComplexOpAddition, null);
                il.Emit(OpCodes.Stloc, locAcc);
                return;
            }

            // Primitive accum (long / ulong / float / double).
            // Pattern: ldloc acc; <load+widen>; add; stloc acc.
            il.Emit(OpCodes.Ldloc, locAcc);
            il.Emit(OpCodes.Ldloc, locP);
            EmitLoadAndWidenToAccum(il, srcType, accumType);
            if (accumType == typeof(float) || accumType == typeof(double))
                // For the floats this walk IS NumPy's elementwise add (out = out + x, the accumulator first), whose
                // loops keep the SECOND operand's NaN when both are NaN; a bare `add` lets RyuJIT pick either operand.
                EmitScalarOperation(il, BinaryOp.Add, accumType == typeof(float) ? NPTypeCode.Single : NPTypeCode.Double,
                    NaNOperandOrder.NumPyUfunc);
            else
                il.Emit(OpCodes.Add);
            if (srcType == typeof(Half))
            {
                // HALF_add stores every step as float16: round the float32 sum (npy_float_to_half), then widen it back
                // exactly (npy_half_to_float) so the next step adds to the stored value.
                il.EmitCall(OpCodes.Call, TraceSingleToHalfBits, null);
                il.EmitCall(OpCodes.Call, TraceHalfToSingle, null);
            }
            il.Emit(OpCodes.Stloc, locAcc);
        }

        /// <summary>
        /// <see cref="HalfToFloatScalarExact"/> for emitted calls: NumPy's <c>npy_half_to_float</c>, exact for every
        /// float16 including a signalling NaN's payload (the BCL <c>(float)Half</c> conversion quiets it).
        /// </summary>
        private static readonly MethodInfo TraceHalfToSingle =
            typeof(DirectILKernelGenerator).GetMethod(nameof(HalfToFloatScalarExact), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(DirectILKernelGenerator), nameof(HalfToFloatScalarExact));

        /// <summary>
        /// <see cref="SingleToHalfBits"/> for emitted calls: NumPy's <c>npy_float_to_half</c> (round to nearest even,
        /// overflow to infinity, a NaN's payload kept), returning the raw float16 bits.
        /// </summary>
        private static readonly MethodInfo TraceSingleToHalfBits =
            typeof(DirectILKernelGenerator).GetMethod(nameof(SingleToHalfBits), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(DirectILKernelGenerator), nameof(SingleToHalfBits));

        /// <summary>
        /// Loads <c>*p</c> as <paramref name="srcType"/> and widens it onto the
        /// IL stack as <paramref name="accumType"/>. Only used when accumType
        /// is a primitive numeric (the struct-accum cases are handled
        /// inline in <see cref="EmitLoadAndAdd"/>).
        /// </summary>
        /// <param name="il">The IL stream; the element's address is on the stack.</param>
        /// <param name="srcType">The source element CLR type.</param>
        /// <param name="accumType">The accumulator CLR type the value is widened to.</param>
        /// <exception cref="NotSupportedException">A source dtype the trace table never produces.</exception>
        private static void EmitLoadAndWidenToAccum(ILGenerator il, Type srcType, Type accumType)
        {
            if (srcType == typeof(Half))
            {
                // ldind.u2; call HalfToFloatScalarExact — the accumulator is float32, as in HALF_add, and the widening
                // is NumPy's exact one (a signalling NaN reaches the add still signalling, which then quiets it).
                il.Emit(OpCodes.Ldind_U2);
                il.EmitCall(OpCodes.Call, TraceHalfToSingle, null);
                return;
            }
            if (srcType == typeof(bool) || srcType == typeof(byte))
            {
                il.Emit(OpCodes.Ldind_U1);
                if (accumType == typeof(long) || accumType == typeof(ulong))
                    il.Emit(OpCodes.Conv_I8);
                return;
            }
            if (srcType == typeof(sbyte))
            {
                il.Emit(OpCodes.Ldind_I1);
                if (accumType == typeof(long))
                    il.Emit(OpCodes.Conv_I8);
                return;
            }
            if (srcType == typeof(short))
            {
                il.Emit(OpCodes.Ldind_I2);
                if (accumType == typeof(long))
                    il.Emit(OpCodes.Conv_I8);
                return;
            }
            if (srcType == typeof(ushort) || srcType == typeof(char))
            {
                il.Emit(OpCodes.Ldind_U2);
                if (accumType == typeof(long) || accumType == typeof(ulong))
                    il.Emit(OpCodes.Conv_I8);
                return;
            }
            if (srcType == typeof(int))
            {
                il.Emit(OpCodes.Ldind_I4);
                if (accumType == typeof(long))
                    il.Emit(OpCodes.Conv_I8);
                return;
            }
            if (srcType == typeof(uint))
            {
                il.Emit(OpCodes.Ldind_U4);
                if (accumType == typeof(ulong))
                    il.Emit(OpCodes.Conv_U8);
                return;
            }
            if (srcType == typeof(long) || srcType == typeof(ulong))
            {
                il.Emit(OpCodes.Ldind_I8);
                return;
            }
            if (srcType == typeof(float))
            {
                il.Emit(OpCodes.Ldind_R4);
                return;
            }
            if (srcType == typeof(double))
            {
                il.Emit(OpCodes.Ldind_R8);
                return;
            }
            throw new NotSupportedException($"Trace src dtype unsupported: {srcType.Name}");
        }

        /// <summary>
        /// Converts the accum value to <paramref name="resultType"/> and stores
        /// it at <c>*dstAt</c>. Handles the Half bridge (float32 accum → Half
        /// result: the accumulator already holds a float16 value after every
        /// step, so the narrowing is exact) and the struct results (decimal /
        /// Complex use Stobj; primitives use the appropriate Stind).
        /// </summary>
        /// <param name="il">The IL stream.</param>
        /// <param name="locAcc">The accumulator local.</param>
        /// <param name="accumType">The accumulator CLR type.</param>
        /// <param name="resultType">The result CLR type.</param>
        /// <param name="locDstAt">The local holding the destination element's address.</param>
        /// <exception cref="NotSupportedException">A result type the trace table never produces.</exception>
        private static void EmitStoreAccAsResult(
            ILGenerator il, LocalBuilder locAcc, Type accumType, Type resultType, LocalBuilder locDstAt)
        {
            if (resultType == typeof(Half) && accumType == typeof(float))
            {
                il.Emit(OpCodes.Ldloc, locDstAt);
                il.Emit(OpCodes.Ldloc, locAcc);
                il.EmitCall(OpCodes.Call, TraceSingleToHalfBits, null);
                il.Emit(OpCodes.Stind_I2);
                return;
            }
            if (resultType == typeof(decimal) || resultType == typeof(System.Numerics.Complex))
            {
                il.Emit(OpCodes.Ldloc, locDstAt);
                il.Emit(OpCodes.Ldloc, locAcc);
                il.Emit(OpCodes.Stobj, resultType);
                return;
            }
            // Primitive result (long / ulong / float / double).
            il.Emit(OpCodes.Ldloc, locDstAt);
            il.Emit(OpCodes.Ldloc, locAcc);
            if (resultType == typeof(long) || resultType == typeof(ulong))
            {
                il.Emit(OpCodes.Stind_I8);
            }
            else if (resultType == typeof(float))
            {
                il.Emit(OpCodes.Stind_R4);
            }
            else if (resultType == typeof(double))
            {
                il.Emit(OpCodes.Stind_R8);
            }
            else
            {
                throw new NotSupportedException($"Trace result dtype unsupported: {resultType.Name}");
            }
        }

        /// <summary>
        /// Maps src NPTypeCode → (result NPTypeCode, supported).
        /// Convenience for callers that already have the type-code.
        /// </summary>
        public static (NPTypeCode, bool) GetTraceAccumTypeCode(NPTypeCode src) => src switch
        {
            NPTypeCode.Boolean => (NPTypeCode.Int64, true),
            // Unsigned narrow lanes follow NumPy's add.reduce rule to UNSIGNED int64 —
            // Byte/UInt16 (and Char, the uint16 twin) previously mapped to Int64, which
            // changed both trace's result dtype and its wrap point vs NumPy (uint oracle
            // cells, 2026-09-18); keep in sync with TraceTypeInfo above.
            NPTypeCode.Byte => (NPTypeCode.UInt64, true),
            NPTypeCode.SByte => (NPTypeCode.Int64, true),
            NPTypeCode.Int16 => (NPTypeCode.Int64, true),
            NPTypeCode.UInt16 => (NPTypeCode.UInt64, true),
            NPTypeCode.Int32 => (NPTypeCode.Int64, true),
            NPTypeCode.Int64 => (NPTypeCode.Int64, true),
            NPTypeCode.UInt32 => (NPTypeCode.UInt64, true),
            NPTypeCode.UInt64 => (NPTypeCode.UInt64, true),
            NPTypeCode.Char => (NPTypeCode.UInt64, true),
            NPTypeCode.Single => (NPTypeCode.Single, true),
            NPTypeCode.Double => (NPTypeCode.Double, true),
            NPTypeCode.Half => (NPTypeCode.Half, true),
            NPTypeCode.Decimal => (NPTypeCode.Decimal, true),
            NPTypeCode.Complex => (NPTypeCode.Complex, true),
            _ => (default, false),
        };
    }
}
