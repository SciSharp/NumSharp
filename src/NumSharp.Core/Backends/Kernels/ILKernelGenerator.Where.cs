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

            // Determine if we can use SIMD
            bool canSimd = elementSize <= 8 && IsSimdSupported<T>();

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

            if (canSimd && VectorBits >= 128)
            {
                // Generate SIMD path
                EmitWhereSIMDLoop<T>(il, locI);
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

        private static void EmitWhereSIMDLoop<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = VectorBits >= 256 ? (32 / elementSize) : (16 / elementSize);
            long unrollFactor = 4;
            long unrollStep = vectorCount * unrollFactor;
            bool useV256 = VectorBits >= 256;

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
            var loadMethod = CachedMethods.V256LoadGeneric.MakeGenericMethod(typeof(T));
            var storeMethod = CachedMethods.V256StoreGeneric.MakeGenericMethod(typeof(T));
            var selectMethod = CachedMethods.V256ConditionalSelectGeneric.MakeGenericMethod(typeof(T));

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
            var loadMethod = CachedMethods.V128LoadGeneric.MakeGenericMethod(typeof(T));
            var storeMethod = CachedMethods.V128StoreGeneric.MakeGenericMethod(typeof(T));
            var selectMethod = CachedMethods.V128ConditionalSelectGeneric.MakeGenericMethod(typeof(T));

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
        /// Emit inline V256 mask creation. Stack: byte* -> Vector256{T} (as mask)
        /// </summary>
        private static void EmitInlineMaskCreationV256(ILGenerator il, int elementSize)
        {
            // Stack has: byte* pointing to condition bools

            switch (elementSize)
            {
                case 8: // double/long: load 4 bytes, expand to 4 qwords
                    // *(uint*)ptr
                    il.Emit(OpCodes.Ldind_U4);
                    // Vector128.CreateScalar<uint>(value)
                    il.Emit(OpCodes.Call, CachedMethods.V128CreateScalarUInt);
                    // .AsByte()
                    il.Emit(OpCodes.Call, CachedMethods.V128UIntAsByte);
                    // Avx2.ConvertToVector256Int64(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Avx2ConvertToV256Int64);
                    // .AsUInt64()
                    il.Emit(OpCodes.Call, CachedMethods.V256LongAsULong);
                    // Vector256<ulong>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V256GetZeroULong);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V256GreaterThanULong);
                    break;

                case 4: // float/int: load 8 bytes, expand to 8 dwords
                    // *(ulong*)ptr
                    il.Emit(OpCodes.Ldind_I8);
                    // Vector128.CreateScalar<ulong>(value)
                    il.Emit(OpCodes.Call, CachedMethods.V128CreateScalarULong);
                    // .AsByte()
                    il.Emit(OpCodes.Call, CachedMethods.V128ULongAsByte);
                    // Avx2.ConvertToVector256Int32(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Avx2ConvertToV256Int32);
                    // .AsUInt32()
                    il.Emit(OpCodes.Call, CachedMethods.V256IntAsUInt);
                    // Vector256<uint>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V256GetZeroUInt);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V256GreaterThanUInt);
                    break;

                case 2: // short/char: load 16 bytes, expand to 16 words
                    // Vector128.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, CachedMethods.V128LoadByte);
                    // Avx2.ConvertToVector256Int16(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Avx2ConvertToV256Int16);
                    // .AsUInt16()
                    il.Emit(OpCodes.Call, CachedMethods.V256ShortAsUShort);
                    // Vector256<ushort>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V256GetZeroUShort);
                    // Vector256.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V256GreaterThanUShort);
                    break;

                case 1: // byte/bool: load 32 bytes, compare directly
                    // Vector256.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, CachedMethods.V256LoadByte);
                    // Vector256<byte>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V256GetZeroByte);
                    // Vector256.GreaterThan(vec, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V256GreaterThanByte);
                    break;

                default:
                    throw new NotSupportedException($"Element size {elementSize} not supported");
            }
        }

        /// <summary>
        /// Emit inline V128 mask creation. Stack: byte* -> Vector128{T} (as mask)
        /// </summary>
        private static void EmitInlineMaskCreationV128(ILGenerator il, int elementSize)
        {
            switch (elementSize)
            {
                case 8: // double/long: load 2 bytes, expand to 2 qwords
                    // *(ushort*)ptr
                    il.Emit(OpCodes.Ldind_U2);
                    // Vector128.CreateScalar<ushort>(value)
                    il.Emit(OpCodes.Call, CachedMethods.V128CreateScalarUShort);
                    // .AsByte()
                    il.Emit(OpCodes.Call, CachedMethods.V128UShortAsByte);
                    // Sse41.ConvertToVector128Int64(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Sse41ConvertToV128Int64);
                    // .AsUInt64()
                    il.Emit(OpCodes.Call, CachedMethods.V128LongAsULong);
                    // Vector128<ulong>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V128GetZeroULong);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V128GreaterThanULong);
                    break;

                case 4: // float/int: load 4 bytes, expand to 4 dwords
                    // *(uint*)ptr
                    il.Emit(OpCodes.Ldind_U4);
                    // Vector128.CreateScalar<uint>(value)
                    il.Emit(OpCodes.Call, CachedMethods.V128CreateScalarUInt);
                    // .AsByte()
                    il.Emit(OpCodes.Call, CachedMethods.V128UIntAsByte);
                    // Sse41.ConvertToVector128Int32(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Sse41ConvertToV128Int32);
                    // .AsUInt32()
                    il.Emit(OpCodes.Call, CachedMethods.V128IntAsUInt);
                    // Vector128<uint>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V128GetZeroUInt);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V128GreaterThanUInt);
                    break;

                case 2: // short/char: load 8 bytes, expand to 8 words
                    // *(ulong*)ptr
                    il.Emit(OpCodes.Ldind_I8);
                    // Vector128.CreateScalar<ulong>(value)
                    il.Emit(OpCodes.Call, CachedMethods.V128CreateScalarULong);
                    // .AsByte()
                    il.Emit(OpCodes.Call, CachedMethods.V128ULongAsByte);
                    // Sse41.ConvertToVector128Int16(bytes)
                    il.Emit(OpCodes.Call, CachedMethods.Sse41ConvertToV128Int16);
                    // .AsUInt16()
                    il.Emit(OpCodes.Call, CachedMethods.V128ShortAsUShort);
                    // Vector128<ushort>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V128GetZeroUShort);
                    // Vector128.GreaterThan(expanded, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V128GreaterThanUShort);
                    break;

                case 1: // byte/bool: load 16 bytes, compare directly
                    // Vector128.Load<byte>(ptr)
                    il.Emit(OpCodes.Call, CachedMethods.V128LoadByte);
                    // Vector128<byte>.Zero
                    il.Emit(OpCodes.Call, CachedMethods.V128GetZeroByte);
                    // Vector128.GreaterThan(vec, zero)
                    il.Emit(OpCodes.Call, CachedMethods.V128GreaterThanByte);
                    break;

                default:
                    throw new NotSupportedException($"Element size {elementSize} not supported");
            }
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

        // Per the CachedMethods pattern in ILKernelGenerator.cs, reflection lookups for np.where
        // live alongside the other cached entries. Fail-fast at type init so a renamed API shows
        // up immediately instead of NREs at first use.
        private static partial class CachedMethods
        {
            #region Where Kernel Methods

            private static MethodInfo FindGenericMethod(Type container, string name, int? paramCount = null)
            {
                foreach (var m in container.GetMethods())
                {
                    if (m.Name == name && m.IsGenericMethodDefinition &&
                        (paramCount is null || m.GetParameters().Length == paramCount.Value))
                        return m;
                }
                throw new MissingMethodException(container.FullName, name);
            }

            private static MethodInfo FindMethodExact(Type container, string name, Type[] argTypes)
                => container.GetMethod(name, argTypes)
                   ?? throw new MissingMethodException(container.FullName, name);

            private static MethodInfo GetZeroGetter(Type vectorOfT)
                => vectorOfT.GetProperty("Zero")?.GetMethod
                   ?? throw new MissingMethodException(vectorOfT.FullName, "get_Zero");

            // Generic definitions — caller must MakeGenericMethod(typeof(T)) before emitting.
            public static readonly MethodInfo V256LoadGeneric = FindGenericMethod(typeof(Vector256), "Load", 1);
            public static readonly MethodInfo V256StoreGeneric = FindGenericMethod(typeof(Vector256), "Store", 2);
            public static readonly MethodInfo V256ConditionalSelectGeneric = FindGenericMethod(typeof(Vector256), "ConditionalSelect");

            public static readonly MethodInfo V128LoadGeneric = FindGenericMethod(typeof(Vector128), "Load", 1);
            public static readonly MethodInfo V128StoreGeneric = FindGenericMethod(typeof(Vector128), "Store", 2);
            public static readonly MethodInfo V128ConditionalSelectGeneric = FindGenericMethod(typeof(Vector128), "ConditionalSelect");

            // Already-specialised generic methods used during mask creation.
            public static readonly MethodInfo V256LoadByte = FindGenericMethod(typeof(Vector256), "Load").MakeGenericMethod(typeof(byte));
            public static readonly MethodInfo V128LoadByte = FindGenericMethod(typeof(Vector128), "Load").MakeGenericMethod(typeof(byte));

            public static readonly MethodInfo V128CreateScalarUInt = FindGenericMethod(typeof(Vector128), "CreateScalar").MakeGenericMethod(typeof(uint));
            public static readonly MethodInfo V128CreateScalarULong = FindGenericMethod(typeof(Vector128), "CreateScalar").MakeGenericMethod(typeof(ulong));
            public static readonly MethodInfo V128CreateScalarUShort = FindGenericMethod(typeof(Vector128), "CreateScalar").MakeGenericMethod(typeof(ushort));

            public static readonly MethodInfo V128UIntAsByte = FindGenericMethod(typeof(Vector128), "AsByte").MakeGenericMethod(typeof(uint));
            public static readonly MethodInfo V128ULongAsByte = FindGenericMethod(typeof(Vector128), "AsByte").MakeGenericMethod(typeof(ulong));
            public static readonly MethodInfo V128UShortAsByte = FindGenericMethod(typeof(Vector128), "AsByte").MakeGenericMethod(typeof(ushort));

            public static readonly MethodInfo V256LongAsULong = FindGenericMethod(typeof(Vector256), "AsUInt64").MakeGenericMethod(typeof(long));
            public static readonly MethodInfo V256IntAsUInt = FindGenericMethod(typeof(Vector256), "AsUInt32").MakeGenericMethod(typeof(int));
            public static readonly MethodInfo V256ShortAsUShort = FindGenericMethod(typeof(Vector256), "AsUInt16").MakeGenericMethod(typeof(short));

            public static readonly MethodInfo V128LongAsULong = FindGenericMethod(typeof(Vector128), "AsUInt64").MakeGenericMethod(typeof(long));
            public static readonly MethodInfo V128IntAsUInt = FindGenericMethod(typeof(Vector128), "AsUInt32").MakeGenericMethod(typeof(int));
            public static readonly MethodInfo V128ShortAsUShort = FindGenericMethod(typeof(Vector128), "AsUInt16").MakeGenericMethod(typeof(short));

            public static readonly MethodInfo V256GreaterThanULong = FindGenericMethod(typeof(Vector256), "GreaterThan").MakeGenericMethod(typeof(ulong));
            public static readonly MethodInfo V256GreaterThanUInt = FindGenericMethod(typeof(Vector256), "GreaterThan").MakeGenericMethod(typeof(uint));
            public static readonly MethodInfo V256GreaterThanUShort = FindGenericMethod(typeof(Vector256), "GreaterThan").MakeGenericMethod(typeof(ushort));
            public static readonly MethodInfo V256GreaterThanByte = FindGenericMethod(typeof(Vector256), "GreaterThan").MakeGenericMethod(typeof(byte));

            public static readonly MethodInfo V128GreaterThanULong = FindGenericMethod(typeof(Vector128), "GreaterThan").MakeGenericMethod(typeof(ulong));
            public static readonly MethodInfo V128GreaterThanUInt = FindGenericMethod(typeof(Vector128), "GreaterThan").MakeGenericMethod(typeof(uint));
            public static readonly MethodInfo V128GreaterThanUShort = FindGenericMethod(typeof(Vector128), "GreaterThan").MakeGenericMethod(typeof(ushort));
            public static readonly MethodInfo V128GreaterThanByte = FindGenericMethod(typeof(Vector128), "GreaterThan").MakeGenericMethod(typeof(byte));

            // Non-generic exact overloads on Avx2/Sse41 for byte-lane sign-extend expansion.
            public static readonly MethodInfo Avx2ConvertToV256Int64 = FindMethodExact(typeof(Avx2), "ConvertToVector256Int64", new[] { typeof(Vector128<byte>) });
            public static readonly MethodInfo Avx2ConvertToV256Int32 = FindMethodExact(typeof(Avx2), "ConvertToVector256Int32", new[] { typeof(Vector128<byte>) });
            public static readonly MethodInfo Avx2ConvertToV256Int16 = FindMethodExact(typeof(Avx2), "ConvertToVector256Int16", new[] { typeof(Vector128<byte>) });
            public static readonly MethodInfo Sse41ConvertToV128Int64 = FindMethodExact(typeof(Sse41), "ConvertToVector128Int64", new[] { typeof(Vector128<byte>) });
            public static readonly MethodInfo Sse41ConvertToV128Int32 = FindMethodExact(typeof(Sse41), "ConvertToVector128Int32", new[] { typeof(Vector128<byte>) });
            public static readonly MethodInfo Sse41ConvertToV128Int16 = FindMethodExact(typeof(Sse41), "ConvertToVector128Int16", new[] { typeof(Vector128<byte>) });

            // Vector*<T>.Zero property getters — emitted as a call, not a field load, so we cache the getter MethodInfo.
            public static readonly MethodInfo V256GetZeroULong = GetZeroGetter(typeof(Vector256<ulong>));
            public static readonly MethodInfo V256GetZeroUInt = GetZeroGetter(typeof(Vector256<uint>));
            public static readonly MethodInfo V256GetZeroUShort = GetZeroGetter(typeof(Vector256<ushort>));
            public static readonly MethodInfo V256GetZeroByte = GetZeroGetter(typeof(Vector256<byte>));
            public static readonly MethodInfo V128GetZeroULong = GetZeroGetter(typeof(Vector128<ulong>));
            public static readonly MethodInfo V128GetZeroUInt = GetZeroGetter(typeof(Vector128<uint>));
            public static readonly MethodInfo V128GetZeroUShort = GetZeroGetter(typeof(Vector128<ushort>));
            public static readonly MethodInfo V128GetZeroByte = GetZeroGetter(typeof(Vector128<byte>));

            #endregion
        }
    }
}
