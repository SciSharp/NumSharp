using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Argwhere.cs — IL-emitted kernels for np.argwhere
// =============================================================================
//
// RESPONSIBILITY:
//   Replace the typeof(T)==typeof(bool) dispatch branch and the generic-T C#
//   helpers in argwhere with a per-dtype IL kernel cache that mirrors the
//   NonZero bool-kernel design but applies to all 15 supported dtypes.
//
// KERNELS (DynamicMethod-emitted per dtype, cached forever):
//
//   * ArgwhereCountKernel (byte* src, long size) -> long
//       SIMD popcount of non-zero T elements. For SIMD-supported dtypes the
//       IL emits the Vector{N}.Load(T*) + Equals(Zero) + ExtractMostSignificantBits
//       + ~bits & laneMask + PopCount loop. For Decimal/Half/Complex the IL
//       emits a scalar loop calling op_Inequality(v, default). One DM per
//       dtype; closed-over IL means no runtime dtype check inside the loop.
//
//   * ArgwhereFlatKernel (byte* src, long size, long* outBuf) -> long
//       SIMD bit-scan that writes flat element indices of non-zero positions
//       into outBuf (pre-sized via ArgwhereCountKernel). Returns count.
//       Same SIMD vs scalar split as Count.
//
//   * ArgwhereExpandKernel (long* flat, long count, long* dims,
//                            long* dimStrides, long ndim, long* outRows) -> void
//       Dtype-agnostic. Single DM. IL-emits the incremental coord-advance
//       loop that turns flat indices into (count, ndim) row-major coordinates.
//       First row seeds coords via divmod; subsequent rows advance the
//       innermost coord by (flat[i] - flat[i-1]) and propagate the carry
//       chain — no per-element divmod on the hot path.
//
// CACHE:
//   ConcurrentDictionary<Type, delegate> keyed on element type. GetOrAdd
//   triggers the IL emission once per (dtype, kernel-kind) and the JIT
//   compiles it to native; subsequent calls hit the cache.
//
// COVERED DTYPES:
//   SIMD (Vector{N}<T> supported by BCL):
//     Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64,
//     Char, Single, Double
//   Scalar IL (no Vector<T> in BCL — emit scalar IL via op_Inequality):
//     Half, Decimal, Complex
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// IL-emitted SIMD popcount: returns the count of non-zero elements in a
    /// contiguous T-typed buffer. <paramref name="size"/> is in <b>elements</b>
    /// (not bytes).
    /// </summary>
    public unsafe delegate long ArgwhereCountKernel(byte* src, long size);

    /// <summary>
    /// IL-emitted SIMD bit-scan: writes flat element indices of non-zero T
    /// positions into <paramref name="outBuf"/> (caller must pre-size via
    /// <see cref="ArgwhereCountKernel"/>). Returns the number written.
    /// </summary>
    public unsafe delegate long ArgwhereFlatKernel(byte* src, long size, long* outBuf);

    /// <summary>
    /// IL-emitted coord-expand: converts a flat-index buffer (monotonic
    /// ascending C-order) to the (<c>count</c>, <c>ndim</c>) row-major argwhere
    /// result via incremental coord advance — no per-element divmod.
    /// Dtype-agnostic (operates on long*).
    /// </summary>
    public unsafe delegate void ArgwhereExpandKernel(
        long* flat, long count, long* dims, long* dimStrides, long ndim, long* outRows);

    public static partial class ILKernelGenerator
    {
        #region Caches

        private static readonly ConcurrentDictionary<Type, ArgwhereCountKernel> _argwhereCount = new();
        private static readonly ConcurrentDictionary<Type, ArgwhereFlatKernel> _argwhereFlat = new();
        private static ArgwhereExpandKernel _argwhereExpand;

        /// <summary>
        /// IL-emitted count of non-zero elements. Returns <c>null</c> only when
        /// <see cref="Enabled"/> is false — every supported dtype has a kernel
        /// (SIMD where Vector{T} exists, scalar IL via op_Inequality otherwise).
        /// </summary>
        public static ArgwhereCountKernel GetArgwhereCountKernel(Type elementType)
        {
            if (!Enabled)
                return null;

            return _argwhereCount.GetOrAdd(elementType, static t =>
            {
                try { return GenerateArgwhereCountKernelIL(t); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[ILKernel] GetArgwhereCountKernel({t.Name}): {ex.GetType().Name}: {ex.Message}");
                    return null;
                }
            });
        }

        /// <summary>
        /// IL-emitted bit-scan that writes flat indices of non-zero elements
        /// into a pre-sized buffer.
        /// </summary>
        public static ArgwhereFlatKernel GetArgwhereFlatKernel(Type elementType)
        {
            if (!Enabled)
                return null;

            return _argwhereFlat.GetOrAdd(elementType, static t =>
            {
                try { return GenerateArgwhereFlatKernelIL(t); }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"[ILKernel] GetArgwhereFlatKernel({t.Name}): {ex.GetType().Name}: {ex.Message}");
                    return null;
                }
            });
        }

        /// <summary>
        /// IL-emitted coord expand (singleton — same kernel handles any ndim).
        /// </summary>
        public static ArgwhereExpandKernel GetArgwhereExpandKernel()
        {
            if (!Enabled)
                return null;

            var cached = _argwhereExpand;
            if (cached != null)
                return cached;

            try
            {
                var k = GenerateArgwhereExpandKernelIL();
                System.Threading.Interlocked.CompareExchange(ref _argwhereExpand, k, null);
                return _argwhereExpand;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetArgwhereExpandKernel: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region Dtype classification

        /// <summary>
        /// Returns the element type the SIMD IL should use for <paramref name="t"/>.
        /// For most dtypes this is just <paramref name="t"/>. <c>bool</c> reinterprets as
        /// <c>byte</c> and <c>char</c> as <c>ushort</c> — both are bit-compatible
        /// reinterpretations where "is zero" coincides with their numeric zero.
        /// Returns <c>null</c> when the dtype has no SIMD path (Half/Decimal/Complex).
        /// </summary>
        private static Type ArgwhereSimdElement(Type t)
        {
            if (t == typeof(bool)) return typeof(byte);
            if (t == typeof(char)) return typeof(ushort);
            if (t == typeof(byte) || t == typeof(sbyte) ||
                t == typeof(short) || t == typeof(ushort) ||
                t == typeof(int) || t == typeof(uint) ||
                t == typeof(long) || t == typeof(ulong) ||
                t == typeof(float) || t == typeof(double))
                return t;
            return null;
        }

        /// <summary>
        /// Element size in bytes — used by the IL to advance the byte pointer by
        /// <c>i * elementSize</c> per element.
        /// </summary>
        private static int ArgwhereElementBytes(Type t)
        {
            if (t == typeof(bool) || t == typeof(byte) || t == typeof(sbyte)) return 1;
            if (t == typeof(short) || t == typeof(ushort) || t == typeof(char) || t == typeof(Half)) return 2;
            if (t == typeof(int) || t == typeof(uint) || t == typeof(float)) return 4;
            if (t == typeof(long) || t == typeof(ulong) || t == typeof(double)) return 8;
            if (t == typeof(decimal)) return 16;
            if (t == typeof(System.Numerics.Complex)) return 16;
            throw new NotSupportedException($"Argwhere: unsupported element type {t.Name}");
        }

        /// <summary>
        /// IL opcode that loads a value of type <paramref name="t"/> from a native
        /// pointer that's already on the eval stack. Used in the scalar tail.
        /// </summary>
        private static OpCode ArgwhereScalarLoadOpcode(Type t)
        {
            if (t == typeof(bool) || t == typeof(byte)) return OpCodes.Ldind_U1;
            if (t == typeof(sbyte)) return OpCodes.Ldind_I1;
            if (t == typeof(short)) return OpCodes.Ldind_I2;
            if (t == typeof(ushort) || t == typeof(char)) return OpCodes.Ldind_U2;
            if (t == typeof(int)) return OpCodes.Ldind_I4;
            if (t == typeof(uint)) return OpCodes.Ldind_U4;
            if (t == typeof(long) || t == typeof(ulong)) return OpCodes.Ldind_I8;
            if (t == typeof(float)) return OpCodes.Ldind_R4;
            if (t == typeof(double)) return OpCodes.Ldind_R8;
            // For Half/Decimal/Complex the scalar load goes via Ldobj, handled separately.
            throw new NotSupportedException($"No primitive Ldind for {t.Name}");
        }

        #endregion

        #region Reflection caches local to this partial

        private static readonly Lazy<MethodInfo> _bitOpsPopCount = new(() =>
            ScalarMethodCache.BitOp(nameof(BitOperations.PopCount), typeof(uint)));

        private static readonly Lazy<MethodInfo> _bitOpsTrailingZeroCount = new(() =>
            ScalarMethodCache.BitOp(nameof(BitOperations.TrailingZeroCount), typeof(uint)));

        private static MethodInfo BitOpsPopCount32 => _bitOpsPopCount.Value;
        private static MethodInfo BitOpsTrailingZeroCount32 => _bitOpsTrailingZeroCount.Value;

        #endregion

        #region Count kernel IL emission

        private static ArgwhereCountKernel GenerateArgwhereCountKernelIL(Type elementType)
        {
            var dm = new DynamicMethod(
                name: $"IL_ArgwhereCount_{elementType.Name}",
                returnType: typeof(long),
                parameterTypes: new[] { typeof(byte*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var simdElem = ArgwhereSimdElement(elementType);

            if (simdElem != null && VectorBits >= 128)
                EmitArgwhereCountSimdBody(il, elementType, simdElem);
            else
                EmitArgwhereCountScalarBody(il, elementType);

            return (ArgwhereCountKernel)dm.CreateDelegate(typeof(ArgwhereCountKernel));
        }

        /// <summary>
        /// Emits the three-stage count loop: SIMD popcount body + scalar tail.
        /// Idiom mirrors <c>GenerateNonZeroCountBoolKernelIL</c> but parameterized
        /// on element type, so the lane count, element-size byte stride, and
        /// chunk mask are all baked into the IL at emission time.
        /// </summary>
        private static void EmitArgwhereCountSimdBody(ILGenerator il, Type elementType, Type simdElem)
        {
            int elementSize = ArgwhereElementBytes(elementType);
            // ExtractMostSignificantBits returns uint with up to 32 valid bits — that's the
            // limit on lane count we can handle. V256<byte> = 32 lanes; bigger lane-count
            // configurations clamp to V128 here.
            int simdBits = VectorBits >= 256 ? 256 : 128;
            int laneCount = simdBits / 8 / elementSize;
            uint chunkMask = laneCount >= 32 ? uint.MaxValue : ((1u << laneCount) - 1);

            var locI = il.DeclareLocal(typeof(long));
            var locVecEnd = il.DeclareLocal(typeof(long));
            var locCount = il.DeclareLocal(typeof(long));

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();

            // i = 0; count = 0;
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locCount);

            // vecEnd = size - laneCount;
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, (long)laneCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ---- SIMD popcount loop ----
            il.MarkLabel(lblSimdHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            // vec = Vector{N}.Load<simdElem>((simdElem*)(src + i*elementSize))
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Load(simdBits, simdElem), null);

            // zero = V<simdElem>.Zero
            il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(simdBits, simdElem), null);

            // cmp = Equals(vec, zero)
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(simdBits, simdElem), null);

            // bits = ExtractMostSignificantBits(cmp)
            il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(simdBits, simdElem), null);

            // nz = ~bits & chunkMask
            il.Emit(OpCodes.Not);
            il.Emit(OpCodes.Ldc_I4, unchecked((int)chunkMask));
            il.Emit(OpCodes.And);

            // count += PopCount(nz)
            il.EmitCall(OpCodes.Call, BitOpsPopCount32, null);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locCount);

            // i += laneCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)laneCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            // ---- Scalar tail ----
            EmitArgwhereCountScalarTail(il, elementType, locI, locCount, lblScalarHead, lblScalarEnd, lblScalarSkip);

            il.MarkLabel(lblScalarEnd);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        /// Scalar-only count body for dtypes without Vector{T} support (Half /
        /// Decimal / Complex). One IL loop, calls the dtype's op_Inequality
        /// against default(T) per element.
        /// </summary>
        private static void EmitArgwhereCountScalarBody(ILGenerator il, Type elementType)
        {
            var locI = il.DeclareLocal(typeof(long));
            var locCount = il.DeclareLocal(typeof(long));
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblSkip = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locCount);

            EmitArgwhereCountScalarTail(il, elementType, locI, locCount, lblHead, lblEnd, lblSkip);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ret);
        }

        /// <summary>
        /// Shared scalar walk used by both the SIMD tail and the scalar-only body.
        /// For primitive types emits Ldind_Xn + brfalse-on-zero. For Half/Decimal/Complex
        /// emits Ldobj + op_Inequality(v, default) call.
        /// </summary>
        private static void EmitArgwhereCountScalarTail(
            ILGenerator il, Type elementType, LocalBuilder locI, LocalBuilder locCount,
            Label lblHead, Label lblEnd, Label lblSkip)
        {
            int elementSize = ArgwhereElementBytes(elementType);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblEnd);

            // Push (src + i*elementSize) as the pointer.
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Branch on (v != 0). For primitives use Ldind + Brfalse(==0). For
            // Half/Decimal/Complex load the value, push default(T) via stackalloc
            // local + initobj, and call op_Inequality.
            EmitArgwhereLoadAndIsNonZero(il, elementType, lblSkip);

            // count++
            il.Emit(OpCodes.Ldloc, locCount);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locCount);

            il.MarkLabel(lblSkip);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);
        }

        /// <summary>
        /// Loads the element at the pointer on top of the stack and branches to
        /// <paramref name="lblSkip"/> when the value is zero (i.e. should NOT be counted).
        /// Falls through when the value is non-zero. Picks the cheapest IL form
        /// for each dtype: Ldind + brfalse for ints/floats, Ldobj + op_Inequality
        /// for Half/Decimal/Complex.
        /// </summary>
        private static void EmitArgwhereLoadAndIsNonZero(ILGenerator il, Type elementType, Label lblSkip)
        {
            if (elementType == typeof(bool) || elementType == typeof(byte))
            {
                il.Emit(OpCodes.Ldind_U1);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(sbyte))
            {
                il.Emit(OpCodes.Ldind_I1);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(short))
            {
                il.Emit(OpCodes.Ldind_I2);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(ushort) || elementType == typeof(char))
            {
                il.Emit(OpCodes.Ldind_U2);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(int))
            {
                il.Emit(OpCodes.Ldind_I4);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(uint))
            {
                il.Emit(OpCodes.Ldind_U4);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            if (elementType == typeof(long) || elementType == typeof(ulong))
            {
                il.Emit(OpCodes.Ldind_I8);
                il.Emit(OpCodes.Ldc_I8, 0L);
                il.Emit(OpCodes.Beq, lblSkip);
                return;
            }
            if (elementType == typeof(float))
            {
                // Use SIMD-numerical zero check: v != 0 (handles -0 == 0 correctly).
                il.Emit(OpCodes.Ldind_R4);
                il.Emit(OpCodes.Ldc_R4, 0.0f);
                il.Emit(OpCodes.Beq, lblSkip);
                return;
            }
            if (elementType == typeof(double))
            {
                il.Emit(OpCodes.Ldind_R8);
                il.Emit(OpCodes.Ldc_R8, 0.0);
                il.Emit(OpCodes.Beq, lblSkip);
                return;
            }
            if (elementType == typeof(Half) ||
                elementType == typeof(decimal) ||
                elementType == typeof(System.Numerics.Complex))
            {
                // Ldobj loads the value; op_Inequality returns bool — branch on the
                // bool (zero == "equal, skip", non-zero == "not equal, count").
                var defaultLocal = il.DeclareLocal(elementType);
                il.Emit(OpCodes.Ldobj, elementType);     // value
                il.Emit(OpCodes.Ldloca, defaultLocal);
                il.Emit(OpCodes.Initobj, elementType);   // zero-init the local
                il.Emit(OpCodes.Ldloc, defaultLocal);
                il.EmitCall(OpCodes.Call, ScalarMethodCache.BinaryOp(elementType, "op_Inequality"), null);
                il.Emit(OpCodes.Brfalse, lblSkip);
                return;
            }
            throw new NotSupportedException($"EmitArgwhereLoadAndIsNonZero: unsupported {elementType.Name}");
        }

        #endregion

        #region Flat kernel IL emission

        private static ArgwhereFlatKernel GenerateArgwhereFlatKernelIL(Type elementType)
        {
            var dm = new DynamicMethod(
                name: $"IL_ArgwhereFlat_{elementType.Name}",
                returnType: typeof(long),
                parameterTypes: new[] { typeof(byte*), typeof(long), typeof(long*) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var simdElem = ArgwhereSimdElement(elementType);

            if (simdElem != null && VectorBits >= 128)
                EmitArgwhereFlatSimdBody(il, elementType, simdElem);
            else
                EmitArgwhereFlatScalarBody(il, elementType);

            return (ArgwhereFlatKernel)dm.CreateDelegate(typeof(ArgwhereFlatKernel));
        }

        private static void EmitArgwhereFlatSimdBody(ILGenerator il, Type elementType, Type simdElem)
        {
            int elementSize = ArgwhereElementBytes(elementType);
            int simdBits = VectorBits >= 256 ? 256 : 128;
            int laneCount = simdBits / 8 / elementSize;
            uint chunkMask = laneCount >= 32 ? uint.MaxValue : ((1u << laneCount) - 1);

            var locI = il.DeclareLocal(typeof(long));      // element index
            var locOut = il.DeclareLocal(typeof(long));    // output write index
            var locVecEnd = il.DeclareLocal(typeof(long));
            var locNz = il.DeclareLocal(typeof(uint));
            var locPos = il.DeclareLocal(typeof(int));

            var lblSimdHead = il.DefineLabel();
            var lblSimdEnd = il.DefineLabel();
            var lblBitHead = il.DefineLabel();
            var lblBitEnd = il.DefineLabel();
            var lblScalarHead = il.DefineLabel();
            var lblScalarEnd = il.DefineLabel();
            var lblScalarSkip = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locOut);

            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldc_I8, (long)laneCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVecEnd);

            // ---- SIMD outer loop ----
            il.MarkLabel(lblSimdHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVecEnd);
            il.Emit(OpCodes.Bgt, lblSimdEnd);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Load(simdBits, simdElem), null);

            il.EmitCall(OpCodes.Call, VectorMethodCache.Zero(simdBits, simdElem), null);
            il.EmitCall(OpCodes.Call, VectorMethodCache.Equals(simdBits, simdElem), null);
            il.EmitCall(OpCodes.Call, VectorMethodCache.ExtractMostSignificantBits(simdBits, simdElem), null);

            il.Emit(OpCodes.Not);
            il.Emit(OpCodes.Ldc_I4, unchecked((int)chunkMask));
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, locNz);

            // ---- Inner bit-scan loop ----
            il.MarkLabel(lblBitHead);
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Beq, lblBitEnd);

            // pos = TrailingZeroCount(nz)
            il.Emit(OpCodes.Ldloc, locNz);
            il.EmitCall(OpCodes.Call, BitOpsTrailingZeroCount32, null);
            il.Emit(OpCodes.Stloc, locPos);

            // outBuf[out] = i + pos
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locPos);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stind_I8);

            // out++
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOut);

            // nz &= nz - 1
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldloc, locNz);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.And);
            il.Emit(OpCodes.Stloc, locNz);

            il.Emit(OpCodes.Br, lblBitHead);

            il.MarkLabel(lblBitEnd);

            // i += laneCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)laneCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblSimdHead);

            il.MarkLabel(lblSimdEnd);

            EmitArgwhereFlatScalarTail(il, elementType, locI, locOut, lblScalarHead, lblScalarEnd, lblScalarSkip);

            il.MarkLabel(lblScalarEnd);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ret);
        }

        private static void EmitArgwhereFlatScalarBody(ILGenerator il, Type elementType)
        {
            var locI = il.DeclareLocal(typeof(long));
            var locOut = il.DeclareLocal(typeof(long));
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblSkip = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Ldc_I8, 0L); il.Emit(OpCodes.Stloc, locOut);

            EmitArgwhereFlatScalarTail(il, elementType, locI, locOut, lblHead, lblEnd, lblSkip);

            il.MarkLabel(lblEnd);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ret);
        }

        private static void EmitArgwhereFlatScalarTail(
            ILGenerator il, Type elementType, LocalBuilder locI, LocalBuilder locOut,
            Label lblHead, Label lblEnd, Label lblSkip)
        {
            int elementSize = ArgwhereElementBytes(elementType);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblEnd);

            // pointer = src + i*elementSize
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, (long)elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            EmitArgwhereLoadAndIsNonZero(il, elementType, lblSkip);

            // outBuf[out] = i
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I4_8);
            il.Emit(OpCodes.Conv_I8);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Stind_I8);

            // out++
            il.Emit(OpCodes.Ldloc, locOut);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locOut);

            il.MarkLabel(lblSkip);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);
        }

        #endregion

        #region Expand kernel IL emission

        /// <summary>
        /// Emits the coord-expand kernel:
        /// <code>
        /// void Expand(long* flat, long count, long* dims, long* dimStrides, long ndim, long* outRows) {
        ///     // Seed coords[0..ndim) from flat[0] via single divmod chain.
        ///     long f = flat[0];
        ///     long[ndim] coords = stackalloc;
        ///     for (long d = 0; d &lt; ndim; d++) {
        ///         long s = dimStrides[d];
        ///         coords[d] = f / s; f %= s;
        ///     }
        ///     // Write row 0.
        ///     for (long d = 0; d &lt; ndim; d++) outRows[d] = coords[d];
        ///
        ///     long lastFlat = flat[0];
        ///     long innerSize = dims[ndim - 1];
        ///     for (long i = 1; i &lt; count; i++) {
        ///         long fi = flat[i];
        ///         long delta = fi - lastFlat; lastFlat = fi;
        ///         long newInner = coords[ndim-1] + delta;
        ///         if (newInner &lt; innerSize) coords[ndim-1] = newInner;
        ///         else {
        ///             long carry = newInner / innerSize;
        ///             coords[ndim-1] = newInner % innerSize;
        ///             for (long d = ndim - 2; d &gt;= 0 &amp;&amp; carry &gt; 0; d--) {
        ///                 long sum = coords[d] + carry;
        ///                 if (sum &lt; dims[d]) { coords[d] = sum; carry = 0; }
        ///                 else { coords[d] = sum % dims[d]; carry = sum / dims[d]; }
        ///             }
        ///         }
        ///         long* row = outRows + i * ndim;
        ///         for (long d = 0; d &lt; ndim; d++) row[d] = coords[d];
        ///     }
        /// }
        /// </code>
        /// All loops are IL. <c>coords</c> is stack-allocated via <c>Localloc</c>
        /// so we don't depend on a managed array.
        /// </summary>
        private static ArgwhereExpandKernel GenerateArgwhereExpandKernelIL()
        {
            var dm = new DynamicMethod(
                name: "IL_ArgwhereExpand",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(long*), typeof(long), typeof(long*), typeof(long*), typeof(long), typeof(long*) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();

            var locCoords = il.DeclareLocal(typeof(long*));    // stackalloc'd coord buffer
            var locF = il.DeclareLocal(typeof(long));
            var locS = il.DeclareLocal(typeof(long));
            var locD = il.DeclareLocal(typeof(long));
            var locI = il.DeclareLocal(typeof(long));
            var locInnerSize = il.DeclareLocal(typeof(long));
            var locLastFlat = il.DeclareLocal(typeof(long));
            var locFi = il.DeclareLocal(typeof(long));
            var locDelta = il.DeclareLocal(typeof(long));
            var locNewInner = il.DeclareLocal(typeof(long));
            var locCarry = il.DeclareLocal(typeof(long));
            var locSum = il.DeclareLocal(typeof(long));
            var locRow = il.DeclareLocal(typeof(long*));

            // --- coords = stackalloc long[ndim] ---
            il.Emit(OpCodes.Ldarg, 4);            // ndim
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_U);
            il.Emit(OpCodes.Localloc);
            il.Emit(OpCodes.Stloc, locCoords);

            // --- Seed: f = flat[0] ---
            il.Emit(OpCodes.Ldarg_0);             // flat
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locF);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Stloc, locLastFlat);

            // --- for (d = 0; d < ndim; d++) coords[d] = f / dimStrides[d]; f %= dimStrides[d]; ---
            var lblSeedHead = il.DefineLabel();
            var lblSeedEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locD);

            il.MarkLabel(lblSeedHead);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Bge, lblSeedEnd);

            // s = dimStrides[d]
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locS);

            // coords[d] = f / s
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stind_I8);

            // f = f % s
            il.Emit(OpCodes.Ldloc, locF);
            il.Emit(OpCodes.Ldloc, locS);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stloc, locF);

            // d++
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblSeedHead);

            il.MarkLabel(lblSeedEnd);

            // --- Write row 0: for (d = 0; d < ndim; d++) outRows[d] = coords[d] ---
            EmitWriteRow(il, /*rowPtrLocal=*/ null, /*rowIndexLocal=*/ null, locCoords);

            // --- innerSize = dims[ndim - 1] ---
            il.Emit(OpCodes.Ldarg_2);                        // dims
            il.Emit(OpCodes.Ldarg, 4);                       // ndim
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locInnerSize);

            // --- Outer loop: for (i = 1; i < count; i++) ---
            var lblOuterHead = il.DefineLabel();
            var lblOuterEnd = il.DefineLabel();
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Stloc, locI);

            il.MarkLabel(lblOuterHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Bge, lblOuterEnd);

            // fi = flat[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Stloc, locFi);

            // delta = fi - lastFlat
            il.Emit(OpCodes.Ldloc, locFi);
            il.Emit(OpCodes.Ldloc, locLastFlat);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locDelta);

            // lastFlat = fi
            il.Emit(OpCodes.Ldloc, locFi);
            il.Emit(OpCodes.Stloc, locLastFlat);

            // newInner = coords[ndim-1] + delta
            //   addr = coords + (ndim-1)*8
            var lblInnerNoOverflow = il.DefineLabel();
            var lblAdvanceEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldloc, locDelta);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locNewInner);

            // if (newInner < innerSize) goto inner_no_overflow
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Blt, lblInnerNoOverflow);

            // --- Overflow path: carry chain ---
            // carry = newInner / innerSize; coords[ndim-1] = newInner % innerSize
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locCarry);

            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Ldloc, locInnerSize);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // for (d = ndim - 2; d >= 0 && carry > 0; d--) { ... }
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 2L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);

            var lblCarryHead = il.DefineLabel();
            var lblCarryEnd = il.DefineLabel();

            il.MarkLabel(lblCarryHead);
            // if (d < 0) break
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Blt, lblCarryEnd);
            // if (carry == 0) break
            il.Emit(OpCodes.Ldloc, locCarry);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Ble, lblCarryEnd);

            // sum = coords[d] + carry
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Ldloc, locCarry);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locSum);

            // axisSize = dims[d]
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);

            // if (sum < axisSize) { coords[d] = sum; carry = 0; }
            var lblCarryOverflow = il.DefineLabel();
            var lblCarryStep = il.DefineLabel();
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ble, lblCarryOverflow);

            // sum < axisSize → assign + zero carry
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Stind_I8);
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locCarry);
            il.Emit(OpCodes.Br, lblCarryStep);

            // sum >= axisSize → recompute coords[d] = sum % dims[d] and carry = sum / dims[d].
            // Ble already consumed (axisSize, sum) from the stack.
            il.MarkLabel(lblCarryOverflow);

            // coords[d] = sum % dims[d]
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Rem);
            il.Emit(OpCodes.Stind_I8);

            // carry = sum / dims[d]
            il.Emit(OpCodes.Ldloc, locSum);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);
            il.Emit(OpCodes.Div);
            il.Emit(OpCodes.Stloc, locCarry);

            il.MarkLabel(lblCarryStep);
            il.Emit(OpCodes.Ldloc, locD);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locD);
            il.Emit(OpCodes.Br, lblCarryHead);

            il.MarkLabel(lblCarryEnd);
            il.Emit(OpCodes.Br, lblAdvanceEnd);

            // --- No overflow path: just store newInner into innermost coord ---
            il.MarkLabel(lblInnerNoOverflow);
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldloc, locNewInner);
            il.Emit(OpCodes.Stind_I8);

            il.MarkLabel(lblAdvanceEnd);

            // --- row = outRows + i * ndim * 8 ---
            il.Emit(OpCodes.Ldarg, 5);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locRow);

            EmitWriteRow(il, locRow, null, locCoords);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblOuterHead);

            il.MarkLabel(lblOuterEnd);
            il.Emit(OpCodes.Ret);

            return (ArgwhereExpandKernel)dm.CreateDelegate(typeof(ArgwhereExpandKernel));
        }

        /// <summary>
        /// Emits an inner loop that copies <c>ndim</c> coords into a row pointer.
        /// When <paramref name="rowPtrLocal"/> is null the row is <c>outRows</c>
        /// (i.e. row 0). Otherwise it's the pre-computed row pointer.
        /// </summary>
        private static void EmitWriteRow(ILGenerator il, LocalBuilder rowPtrLocal, LocalBuilder rowIndexLocal, LocalBuilder locCoords)
        {
            var locDw = il.DeclareLocal(typeof(long));
            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locDw);

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldarg, 4); // ndim
            il.Emit(OpCodes.Bge, lblEnd);

            // dest = row + dw*8 (or outRows + dw*8 for row 0)
            if (rowPtrLocal != null)
                il.Emit(OpCodes.Ldloc, rowPtrLocal);
            else
                il.Emit(OpCodes.Ldarg, 5); // outRows
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // value = coords[dw]
            il.Emit(OpCodes.Ldloc, locCoords);
            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 8L);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_I8);

            il.Emit(OpCodes.Stind_I8);

            il.Emit(OpCodes.Ldloc, locDw);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locDw);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        #endregion
    }
}
