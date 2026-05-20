using System;
using System.Collections.Concurrent;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// ILKernelGenerator.Where - IL-generated np.where(condition, x, y) kernels
// =============================================================================
//
// RESPONSIBILITY:
//   - Generate optimized kernels for conditional selection
//   - result[i] = cond[i] ? x[i] : y[i]
//
// ARCHITECTURE:
//   Uses IL emission to generate type-specific kernels at runtime.
//   The challenge is bool mask expansion: condition is bool[] (1 byte per element),
//   but x/y can be any dtype (1-8 bytes per element).
//
//   | Element Size | V256 Elements | Bools to Load |
//   |--------------|---------------|---------------|
//   | 1 byte       | 32            | 32            |
//   | 2 bytes      | 16            | 16            |
//   | 4 bytes      | 8             | 8             |
//   | 8 bytes      | 4             | 4             |
//
// KERNEL TYPES:
//   - WhereKernel<T>: Main kernel delegate (cond*, x*, y*, result*, count)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Delegate for where operation kernels.
    /// </summary>
    public unsafe delegate void WhereKernel<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged;

    public static partial class ILKernelGenerator
    {
        /// <summary>
        /// Cache of IL-generated where kernels.
        /// Key: Type
        /// </summary>
        private static readonly ConcurrentDictionary<Type, Delegate> _whereKernelCache = new();

        #region Public API

        /// <summary>
        /// Get or generate an IL-based where kernel for the specified type.
        /// Returns null if IL generation is disabled or fails.
        /// </summary>
        public static WhereKernel<T>? GetWhereKernel<T>() where T : unmanaged
        {
            if (!Enabled)
                return null;

            var type = typeof(T);

            if (_whereKernelCache.TryGetValue(type, out var cached))
                return (WhereKernel<T>)cached;

            var kernel = TryGenerateWhereKernel<T>();
            if (kernel == null)
                return null;

            if (_whereKernelCache.TryAdd(type, kernel))
                return kernel;

            return (WhereKernel<T>)_whereKernelCache[type];
        }

        /// <summary>
        /// Execute where operation using IL-generated kernel or fallback to static helper.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void WhereExecute<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged
        {
            if (count == 0)
                return;

            var kernel = GetWhereKernel<T>();
            if (kernel != null)
            {
                kernel(cond, x, y, result, count);
            }
            else
            {
                // Fallback to scalar loop
                WhereScalar(cond, x, y, result, count);
            }
        }

        #endregion

        #region Kernel Generation

        private static WhereKernel<T>? TryGenerateWhereKernel<T>() where T : unmanaged
        {
            try
            {
                return GenerateWhereKernelIL<T>();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] TryGenerateWhereKernel<{typeof(T).Name}>: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static unsafe WhereKernel<T> GenerateWhereKernelIL<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();

            // SIMD eligibility:
            //  - 1-byte types (byte) only touch portable Vector128/Vector256 APIs, so they work
            //    on any SIMD-capable platform (including ARM64/Neon).
            //  - 2/4/8-byte types need Sse41.ConvertToVector128Int* (V128 path) or
            //    Avx2.ConvertToVector256Int* (V256 path) to expand the bool-mask lanes.
            //    These x86 intrinsics throw PlatformNotSupportedException on ARM64.
            bool canSimdDtype = elementSize <= 8 && IsSimdSupported<T>();
            bool needsX86 = elementSize > 1;
            bool useV256 = VectorBits >= 256 && (!needsX86 || Avx2.IsSupported);
            bool useV128 = !useV256 && VectorBits >= 128 && (!needsX86 || Sse41.IsSupported);
            bool emitSimd = canSimdDtype && (useV256 || useV128);

            var dm = new DynamicMethod(
                name: $"IL_Where_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(T*), typeof(T*), typeof(T*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(long));  // loop counter

            // Labels
            var lblScalarLoop = il.DefineLabel();
            var lblScalarLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            if (emitSimd)
            {
                EmitWhereSIMDLoop<T>(il, locI, useV256);
            }

            // Scalar loop for remainder
            il.MarkLabel(lblScalarLoop);

            // if (i >= count) goto end
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Bge, lblScalarLoopEnd);

            // result[i] = cond[i] ? x[i] : y[i]
            EmitWhereScalarElement<T>(il, locI);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblScalarLoop);

            il.MarkLabel(lblScalarLoopEnd);
            il.Emit(OpCodes.Ret);

            return (WhereKernel<T>)dm.CreateDelegate(typeof(WhereKernel<T>));
        }

        private static void EmitWhereSIMDLoop<T>(ILGenerator il, LocalBuilder locI, bool useV256) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = useV256 ? (32 / elementSize) : (16 / elementSize);
            long unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;

            var locUnrollEnd = il.DeclareLocal(typeof(long));
            var locVectorEnd = il.DeclareLocal(typeof(long));

            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblVectorLoop = il.DefineLabel();
            var lblVectorLoopEnd = il.DefineLabel();

            // unrollEnd = count - unrollStep (for 4x unrolled loop)
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // vectorEnd = count - vectorCount (for remainder loop)
            il.Emit(OpCodes.Ldarg, 4);  // count
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // ========== 4x UNROLLED SIMD LOOP ==========
            il.MarkLabel(lblUnrollLoop);

            // if (i > unrollEnd) goto UnrollLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollLoopEnd);

            // Process 4 vectors per iteration
            for (long u = 0; u < unrollFactor; u++)
            {
                long offset = vectorCount * u;
                if (useV256)
                    EmitWhereV256BodyWithOffset<T>(il, locI, elementSize, offset);
                else
                    EmitWhereV128BodyWithOffset<T>(il, locI, elementSize, offset);
            }

            // i += unrollStep
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblUnrollLoop);

            il.MarkLabel(lblUnrollLoopEnd);

            // ========== REMAINDER SIMD LOOP (1 vector at a time) ==========
            il.MarkLabel(lblVectorLoop);

            // if (i > vectorEnd) goto VectorLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVectorLoopEnd);

            // Process 1 vector
            if (useV256)
                EmitWhereV256BodyWithOffset<T>(il, locI, elementSize, 0L);
            else
                EmitWhereV128BodyWithOffset<T>(il, locI, elementSize, 0L);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblVectorLoop);

            il.MarkLabel(lblVectorLoopEnd);
        }

        private static void EmitWhereV256BodyWithOffset<T>(ILGenerator il, LocalBuilder locI, long elementSize, long offset) where T : unmanaged
        {
            var loadMethod = VectorMethodCache.Load(256, typeof(T));
            var storeMethod = VectorMethodCache.Store(256, typeof(T));
            var selectMethod = VectorMethodCache.ConditionalSelect(256, typeof(T));

            // Load address: cond + (i + offset)
            il.Emit(OpCodes.Ldarg_0);  // cond
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Inline mask creation - emit AVX2 instructions directly instead of calling helper
            EmitInlineMaskCreationV256(il, (int)elementSize);

            // Load x vector: x + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_1);  // x
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Load y vector: y + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_2);  // y
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Stack: mask, xVec, yVec
            // ConditionalSelect(mask, x, y)
            il.Emit(OpCodes.Call, selectMethod);

            // Store result: result + (i + offset) * elementSize
            il.Emit(OpCodes.Ldarg_3);  // result
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeMethod);
        }

        private static void EmitWhereV128BodyWithOffset<T>(ILGenerator il, LocalBuilder locI, long elementSize, long offset) where T : unmanaged
        {
            var loadMethod = VectorMethodCache.Load(128, typeof(T));
            var storeMethod = VectorMethodCache.Store(128, typeof(T));
            var selectMethod = VectorMethodCache.ConditionalSelect(128, typeof(T));

            // Load address: cond + (i + offset)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Inline mask creation - emit SSE4.1 instructions directly
            EmitInlineMaskCreationV128(il, (int)elementSize);

            // Load x vector
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // Load y vector
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, loadMethod);

            // ConditionalSelect
            il.Emit(OpCodes.Call, selectMethod);

            // Store
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (offset > 0)
            {
                il.Emit(OpCodes.Ldc_I8, offset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Call, storeMethod);
        }

        private static void EmitWhereScalarElement<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            var typeCode = InfoOf<T>.NPTypeCode;

            // result[i] = cond[i] ? x[i] : y[i]
            var lblFalse = il.DefineLabel();
            var lblEnd = il.DefineLabel();

            // Load result address: result + i * elementSize
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // Load cond[i]: cond + i (bool is 1 byte)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);  // Load bool as byte

            // if (!cond[i]) goto lblFalse
            il.Emit(OpCodes.Brfalse, lblFalse);

            // True branch: load x[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);
            il.Emit(OpCodes.Br, lblEnd);

            // False branch: load y[i]
            il.MarkLabel(lblFalse);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);

            il.MarkLabel(lblEnd);
            // Stack: result_ptr, value
            EmitStoreIndirect(il, typeCode);
        }

        #endregion

        #region Inline Mask IL Emission

        // Vector-related MethodInfos for np.where are cached in the partial CachedMethods class
        // below (see "Where Kernel Methods" region at the end of this file).

        /// <summary>
        /// Emit inline V256 mask creation. Stack: byte* -> Vector256{T} (as mask).
        /// </summary>
        private static void EmitInlineMaskCreationV256(ILGenerator il, int elementSize)
            => EmitInlineMaskCreation(il, simdBits: 256, elementSize);

        /// <summary>
        /// Emit inline V128 mask creation. Stack: byte* -> Vector128{T} (as mask).
        /// </summary>
        private static void EmitInlineMaskCreationV128(ILGenerator il, int elementSize)
            => EmitInlineMaskCreation(il, simdBits: 128, elementSize);

        /// <summary>
        /// Emit mask-creation IL for the np.where contig SIMD body. Unified across V256/V128.
        ///
        /// <para>Input stack:  <c>byte*</c> pointing to <paramref name="elementSize"/>'s worth
        /// of cond bytes per output lane.</para>
        /// <para>Output stack: <c>Vector{simdBits}&lt;UTarget&gt;</c> all-ones-where-cond-true.</para>
        ///
        /// <para>Strategy by element size:</para>
        /// <list type="bullet">
        ///   <item><c>elementSize == 1</c> — direct: V&lt;byte&gt;.Load(ptr) → GreaterThan(_, Zero).</item>
        ///   <item>otherwise — load <c>simdBits/8/elementSize</c> bytes as a scalar
        ///   (Ldind_U2/U4/I8 for byteCount 2/4/8) or as <c>V128&lt;byte&gt;</c> (byteCount 16),
        ///   reinterpret as bytes, sign-extend to <c>V&lt;simdBits&gt;&lt;signedT&gt;</c> via
        ///   <c>Avx2/Sse41.ConvertToVector{...}Int{N}</c>, reinterpret as
        ///   <c>V&lt;simdBits&gt;&lt;unsignedT&gt;</c> for the GreaterThan compare.</item>
        /// </list>
        /// </summary>
        private static void EmitInlineMaskCreation(ILGenerator il, int simdBits, int elementSize)
        {
            if (elementSize == 1)
            {
                // bool/byte: load N condition bytes, compare > 0 directly.
                il.Emit(OpCodes.Call, VectorMethodCache.Load(simdBits, typeof(byte)));
                il.Emit(OpCodes.Call, VectorMethodCache.Zero(simdBits, typeof(byte)));
                il.Emit(OpCodes.Call, VectorMethodCache.GreaterThan(simdBits, typeof(byte)));
                return;
            }

            // Number of cond bytes needed = lane count of output vector.
            int byteCount = simdBits / 8 / elementSize;

            // 2..8 bytes: scalar load + CreateScalar(scalarT) + AsByte gets us to V128<byte>.
            // 16 bytes  : direct V128<byte>.Load.
            switch (byteCount)
            {
                case 2:
                    il.Emit(OpCodes.Ldind_U2);
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateScalar(128, typeof(ushort)));
                    il.Emit(OpCodes.Call, VectorMethodCache.As(128, typeof(ushort), typeof(byte)));
                    break;
                case 4:
                    il.Emit(OpCodes.Ldind_U4);
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateScalar(128, typeof(uint)));
                    il.Emit(OpCodes.Call, VectorMethodCache.As(128, typeof(uint), typeof(byte)));
                    break;
                case 8:
                    il.Emit(OpCodes.Ldind_I8);
                    il.Emit(OpCodes.Call, VectorMethodCache.CreateScalar(128, typeof(ulong)));
                    il.Emit(OpCodes.Call, VectorMethodCache.As(128, typeof(ulong), typeof(byte)));
                    break;
                case 16:
                    il.Emit(OpCodes.Call, VectorMethodCache.Load(128, typeof(byte)));
                    break;
                default:
                    throw new NotSupportedException($"SIMD={simdBits} elementSize={elementSize} -> byteCount={byteCount} not supported");
            }

            // Sign-extend V128<byte> → V<simdBits><signedT>, reinterpret as unsigned, compare > 0.
            int targetElemBits = elementSize * 8;
            Type signedT = targetElemBits switch
            {
                16 => typeof(short),
                32 => typeof(int),
                64 => typeof(long),
                _ => throw new NotSupportedException($"Target lane width {targetElemBits} not supported")
            };
            Type unsignedT = targetElemBits switch
            {
                16 => typeof(ushort),
                32 => typeof(uint),
                64 => typeof(ulong),
                _ => throw new NotSupportedException()
            };

            il.Emit(OpCodes.Call, VectorMethodCache.ByteLaneSignExtend(simdBits, targetElemBits));
            il.Emit(OpCodes.Call, VectorMethodCache.As(simdBits, signedT, unsignedT));
            il.Emit(OpCodes.Call, VectorMethodCache.Zero(simdBits, unsignedT));
            il.Emit(OpCodes.Call, VectorMethodCache.GreaterThan(simdBits, unsignedT));
        }

        #endregion

        #region Scalar Fallback

        /// <summary>
        /// Scalar fallback for where operation.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void WhereScalar<T>(bool* cond, T* x, T* y, T* result, long count) where T : unmanaged
        {
            for (long i = 0; i < count; i++)
            {
                result[i] = cond[i] ? x[i] : y[i];
            }
        }

        #endregion

        // The previous "Where Kernel Methods" region of CachedMethods (~30 fields covering
        // Load/Store/ConditionalSelect/CreateScalar/AsByte/GreaterThan/Zero variants plus the
        // x86 byte-lane sign-extend intrinsics) has been replaced by VectorMethodCache. Each
        // entry now resolves lazily and is keyed on (simdBits, name, elemType) so cross-file
        // call sites share one cached MethodInfo.
    }
}
