using System;
using System.Collections.Concurrent;
using System.Reflection.Emit;

// =============================================================================
// ILKernelGenerator.Trace.cs — fused IL kernel for np.trace 2-D fast path
// =============================================================================
//
// RESPONSIBILITY:
//   np.trace on a 2-D source is sum(diagonal(a)). Composing those three steps
//   (diagonal view + ascontiguousarray + axis-null sum) pays ~3μs of
//   per-call overhead that dwarfs the actual work for small matrices. This
//   kernel collapses the chain into a single strided walk with inline
//   accumulation into the promoted dtype — no NDArray intermediates between
//   input and the 0-d result.
//
// KERNEL (per src dtype, cached forever):
//
//   * TraceKernel(byte* src, long startOff, long diagSize, long byteStride,
//                  byte* dst) → void
//
//   IL pseudocode (per src dtype, accum dtype baked in at emit time):
//
//     acc = 0;           // accum-type zero
//     p   = src + startOff;
//     for (i = 0; i < diagSize; i++) {
//         val = *(srcT*)p;
//         acc += (accumT)val;
//         p += byteStride;
//     }
//     *(accumT*)dst = acc;
//
//   Accum dtype mapping (matches NumPy's "default platform int" rule):
//     bool / byte / sbyte / int16 / uint16 / int32 / int64 / char → int64
//     uint32 / uint64                                              → uint64
//     single / double                                              → preserved
//
//   Other dtypes (Half / Decimal / Complex) have no kernel — np.trace falls
//   back to the generic diagonal+ascontig+sum chain for them.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Walks the diagonal of one or more 2-D sub-arrays and accumulates each
    /// into the promoted dtype. Per-src dtype kernel; accum dtype is baked
    /// into the IL.
    /// <para>
    /// For each <c>i in [0, outerSize)</c>:
    /// <code>
    ///   acc = 0;
    ///   p   = src + startOff + i * outerSrcStride;
    ///   for (j = 0; j &lt; diagSize; j++) {
    ///       acc += *(srcT*)p;
    ///       p += byteStride;
    ///   }
    ///   *(accumT*)(dst + i * outerDstStride) = acc;
    /// </code>
    /// </para>
    /// <para>
    /// 2-D source: <c>outerSize=1</c>, <c>outerSrcStride=outerDstStride=0</c>.
    /// 3-D source with one outer axis: <c>outerSize=outerDim</c>,
    /// <c>outerSrcStride=outerAxisStride*elemBytes</c>,
    /// <c>outerDstStride=accumBytes</c>.
    /// </para>
    /// </summary>
    public unsafe delegate void TraceKernel(
        byte* src, long startOff, long diagSize, long byteStride,
        long outerSize, long outerSrcStride, long outerDstStride, byte* dst);

    public static partial class ILKernelGenerator
    {
        private static readonly ConcurrentDictionary<Type, TraceKernel> _trace = new();

        /// <summary>
        /// IL-emitted singleton per <paramref name="srcType"/>. Returns
        /// <c>null</c> when the dtype has no fast-path kernel (Half / Decimal
        /// / Complex), so callers should treat null as "use the generic
        /// fallback path".
        /// </summary>
        public static TraceKernel GetTraceKernel(Type srcType)
        {
            if (!Enabled)
                return null;

            return _trace.GetOrAdd(srcType, static t =>
            {
                try
                {
                    var (accumType, isSupported) = TraceAccumType(t);
                    if (!isSupported)
                        return null;
                    return GenerateTraceKernelIL(t, accumType);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[ILKernel] GetTraceKernel({t.Name}): {ex.GetType().Name}: {ex.Message}");
                    return null;
                }
            });
        }

        /// <summary>
        /// Maps src dtype → (accum dtype, supported). Mirrors NumPy's
        /// "default platform integer" rule for narrow ints / bool / char,
        /// preserves float dtypes, and reports unsupported for Half / Decimal
        /// / Complex (where the inline IL would need operator-call dispatch).
        /// </summary>
        private static (Type, bool) TraceAccumType(Type t)
        {
            if (t == typeof(bool) || t == typeof(byte) || t == typeof(sbyte) ||
                t == typeof(short) || t == typeof(ushort) ||
                t == typeof(int) || t == typeof(long) ||
                t == typeof(char))
                return (typeof(long), true);

            if (t == typeof(uint) || t == typeof(ulong))
                return (typeof(ulong), true);

            if (t == typeof(float))
                return (typeof(float), true);

            if (t == typeof(double))
                return (typeof(double), true);

            // Half / Decimal / Complex → no kernel; np.trace falls back.
            return (null, false);
        }

        private static TraceKernel GenerateTraceKernelIL(Type srcType, Type accumType)
        {
            var dm = new DynamicMethod(
                name: $"IL_Trace_{srcType.Name}_to_{accumType.Name}",
                returnType: typeof(void),
                parameterTypes: new[]
                {
                    typeof(byte*),  // 0 src
                    typeof(long),   // 1 startOff
                    typeof(long),   // 2 diagSize
                    typeof(long),   // 3 byteStride
                    typeof(long),   // 4 outerSize
                    typeof(long),   // 5 outerSrcStride (bytes between outer slabs in src)
                    typeof(long),   // 6 outerDstStride (bytes between outer slabs in dst)
                    typeof(byte*),  // 7 dst
                },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locAcc = il.DeclareLocal(accumType);
            var locP = il.DeclareLocal(typeof(byte*));      // inner walker
            var locJ = il.DeclareLocal(typeof(long));       // diag loop counter
            var locOuterI = il.DeclareLocal(typeof(long));  // outer loop counter
            var locOuterBase = il.DeclareLocal(typeof(byte*));   // src + startOff + i*outerSrcStride
            var locDstAt = il.DeclareLocal(typeof(byte*));       // dst + i*outerDstStride

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
            il.Emit(OpCodes.Ldarg, 4);                  // outerSize
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // outerBase = src + startOff + outerI * outerSrcStride
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldarg_1);                   // startOff
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 5);                  // outerSrcStride
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOuterBase);

            // dstAt = dst + outerI * outerDstStride
            il.Emit(OpCodes.Ldarg, 7);
            il.Emit(OpCodes.Ldloc, locOuterI);
            il.Emit(OpCodes.Ldarg, 6);                  // outerDstStride
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDstAt);

            // acc = 0
            EmitZero(il, accumType);
            il.Emit(OpCodes.Stloc, locAcc);

            // p = outerBase
            il.Emit(OpCodes.Ldloc, locOuterBase);
            il.Emit(OpCodes.Stloc, locP);

            // j = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locJ);

            // --------- inner diag walk ---------
            il.MarkLabel(lblInnerHead);
            il.Emit(OpCodes.Ldloc, locJ);
            il.Emit(OpCodes.Ldarg_2);                   // diagSize
            il.Emit(OpCodes.Bge, lblInnerEnd);

            // acc += (accumT)*(srcT*)p
            il.Emit(OpCodes.Ldloc, locAcc);
            il.Emit(OpCodes.Ldloc, locP);
            EmitLoadAndWiden(il, srcType, accumType);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locAcc);

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

            // *(accumT*)dstAt = acc
            il.Emit(OpCodes.Ldloc, locDstAt);
            il.Emit(OpCodes.Ldloc, locAcc);
            EmitStindAccum(il, accumType);

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

        /// <summary>Emits a zero literal of the accum dtype onto the IL stack.</summary>
        private static void EmitZero(ILGenerator il, Type accumType)
        {
            if (accumType == typeof(long))
            {
                il.Emit(OpCodes.Ldc_I8, 0L);
            }
            else if (accumType == typeof(ulong))
            {
                il.Emit(OpCodes.Ldc_I8, 0L);     // verifier treats long/ulong identically on stack
            }
            else if (accumType == typeof(float))
            {
                il.Emit(OpCodes.Ldc_R4, 0.0f);
            }
            else if (accumType == typeof(double))
            {
                il.Emit(OpCodes.Ldc_R8, 0.0);
            }
            else
            {
                throw new NotSupportedException($"Trace accum dtype unsupported: {accumType.Name}");
            }
        }

        /// <summary>
        /// Emits load of src elem (using the right Ldind for srcType) and widens
        /// to the accum dtype. Ldind result type drives the Conv needed:
        /// - int64 path: narrow ints widen via Conv_I8 / Conv_U8.
        /// - ulong path: uint32 widens via Conv_U8.
        /// - float / double: direct Ldind_R4 / Ldind_R8, no conversion.
        /// </summary>
        private static void EmitLoadAndWiden(ILGenerator il, Type srcType, Type accumType)
        {
            if (srcType == typeof(bool) || srcType == typeof(byte))
            {
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Conv_I8);
            }
            else if (srcType == typeof(sbyte))
            {
                il.Emit(OpCodes.Ldind_I1);
                il.Emit(OpCodes.Conv_I8);
            }
            else if (srcType == typeof(short))
            {
                il.Emit(OpCodes.Ldind_I2);
                il.Emit(OpCodes.Conv_I8);
            }
            else if (srcType == typeof(ushort) || srcType == typeof(char))
            {
                il.Emit(OpCodes.Ldind_U2);
                il.Emit(OpCodes.Conv_I8);
            }
            else if (srcType == typeof(int))
            {
                il.Emit(OpCodes.Ldind_I4);
                il.Emit(OpCodes.Conv_I8);
            }
            else if (srcType == typeof(uint))
            {
                il.Emit(OpCodes.Ldind_U4);
                il.Emit(OpCodes.Conv_U8);
            }
            else if (srcType == typeof(long) || srcType == typeof(ulong))
            {
                il.Emit(OpCodes.Ldind_I8);
                // no conversion needed; accum is same width
            }
            else if (srcType == typeof(float))
            {
                il.Emit(OpCodes.Ldind_R4);
                // accum is float — no conversion needed
            }
            else if (srcType == typeof(double))
            {
                il.Emit(OpCodes.Ldind_R8);
                // accum is double — no conversion needed
            }
            else
            {
                throw new NotSupportedException($"Trace src dtype unsupported: {srcType.Name}");
            }
        }

        /// <summary>Stind opcode for the accum dtype, writing acc into dst.</summary>
        private static void EmitStindAccum(ILGenerator il, Type accumType)
        {
            if (accumType == typeof(long) || accumType == typeof(ulong))
            {
                il.Emit(OpCodes.Stind_I8);
            }
            else if (accumType == typeof(float))
            {
                il.Emit(OpCodes.Stind_R4);
            }
            else if (accumType == typeof(double))
            {
                il.Emit(OpCodes.Stind_R8);
            }
            else
            {
                throw new NotSupportedException($"Trace accum dtype unsupported: {accumType.Name}");
            }
        }

        /// <summary>
        /// Maps src NPTypeCode → (accum NPTypeCode, supported). Convenience
        /// for callers that already have the type-code.
        /// </summary>
        public static (NPTypeCode, bool) GetTraceAccumTypeCode(NPTypeCode src) => src switch
        {
            NPTypeCode.Boolean => (NPTypeCode.Int64, true),
            NPTypeCode.Byte => (NPTypeCode.Int64, true),
            NPTypeCode.SByte => (NPTypeCode.Int64, true),
            NPTypeCode.Int16 => (NPTypeCode.Int64, true),
            NPTypeCode.UInt16 => (NPTypeCode.Int64, true),
            NPTypeCode.Int32 => (NPTypeCode.Int64, true),
            NPTypeCode.Int64 => (NPTypeCode.Int64, true),
            NPTypeCode.UInt32 => (NPTypeCode.UInt64, true),
            NPTypeCode.UInt64 => (NPTypeCode.UInt64, true),
            NPTypeCode.Char => (NPTypeCode.Int64, true),
            NPTypeCode.Single => (NPTypeCode.Single, true),
            NPTypeCode.Double => (NPTypeCode.Double, true),
            _ => (default, false),
        };
    }
}
