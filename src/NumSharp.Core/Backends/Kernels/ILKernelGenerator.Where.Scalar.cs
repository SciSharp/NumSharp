using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp.Utilities;

// =============================================================================
// ILKernelGenerator.Where.Scalar.cs - IL-emitted scalar-broadcast Where kernels
// =============================================================================
//
// MOTIVATION:
//   np.where(cond, scalar, arr) and np.where(cond, arr, scalar) are very common
//   patterns ("if cond then constant else value"). Today asanyarray(scalar)
//   produces a 0-d array that broadcast_arrays expands to a stride-0 view —
//   the contig-array gate in np.where_internal fails and we fall through to
//   NpyIter, which is ~6.5× slower than the IL contig WhereKernel.
//
// KERNELS (each IL-emitted via DynamicMethod, cached per T):
//
//   WhereScalarXKernel<T>(bool* cond, T scalarX, T* y, T* result, long count)
//     result[i] = cond[i] ? scalarX : y[i]
//
//   WhereScalarYKernel<T>(bool* cond, T* x, T scalarY, T* result, long count)
//     result[i] = cond[i] ? x[i] : scalarY
//
//   WhereScalarXYKernel<T>(bool* cond, T scalarX, T scalarY, T* result, long count)
//     result[i] = cond[i] ? scalarX : scalarY
//
//   The scalar value is hoisted into a Vector256/Vector128.Create<T>(scalar)
//   ONCE before the loop and stored in a local. Inside the SIMD body we Ldloc
//   that pre-built vector instead of loading from memory each iteration.
//   In the scalar tail we Ldarg the value directly.
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public unsafe delegate void WhereScalarXKernel<T>(bool* cond, T scalarX, T* y, T* result, long count) where T : unmanaged;
    public unsafe delegate void WhereScalarYKernel<T>(bool* cond, T* x, T scalarY, T* result, long count) where T : unmanaged;
    public unsafe delegate void WhereScalarXYKernel<T>(bool* cond, T scalarX, T scalarY, T* result, long count) where T : unmanaged;

    public static partial class ILKernelGenerator
    {
        #region Caches

        private static readonly ConcurrentDictionary<Type, Delegate> _whereScalarXCache = new();
        private static readonly ConcurrentDictionary<Type, Delegate> _whereScalarYCache = new();
        private static readonly ConcurrentDictionary<Type, Delegate> _whereScalarXYCache = new();

        #endregion

        #region Public API

        public static WhereScalarXKernel<T> GetWhereScalarXKernel<T>() where T : unmanaged
        {
            if (!Enabled) return null;
            var type = typeof(T);
            if (_whereScalarXCache.TryGetValue(type, out var cached))
                return (WhereScalarXKernel<T>)cached;
            try
            {
                var kernel = GenerateWhereScalarXKernelIL<T>();
                if (kernel == null) return null;
                return (WhereScalarXKernel<T>)_whereScalarXCache.GetOrAdd(type, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetWhereScalarXKernel<{type.Name}>: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        public static WhereScalarYKernel<T> GetWhereScalarYKernel<T>() where T : unmanaged
        {
            if (!Enabled) return null;
            var type = typeof(T);
            if (_whereScalarYCache.TryGetValue(type, out var cached))
                return (WhereScalarYKernel<T>)cached;
            try
            {
                var kernel = GenerateWhereScalarYKernelIL<T>();
                if (kernel == null) return null;
                return (WhereScalarYKernel<T>)_whereScalarYCache.GetOrAdd(type, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetWhereScalarYKernel<{type.Name}>: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        public static WhereScalarXYKernel<T> GetWhereScalarXYKernel<T>() where T : unmanaged
        {
            if (!Enabled) return null;
            var type = typeof(T);
            if (_whereScalarXYCache.TryGetValue(type, out var cached))
                return (WhereScalarXYKernel<T>)cached;
            try
            {
                var kernel = GenerateWhereScalarXYKernelIL<T>();
                if (kernel == null) return null;
                return (WhereScalarXYKernel<T>)_whereScalarXYCache.GetOrAdd(type, kernel);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ILKernel] GetWhereScalarXYKernel<{type.Name}>: {ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        #endregion

        #region SIMD eligibility (mirrors GenerateWhereKernelIL)

        // Same SIMD gate as the contig WhereKernel — V256 needs Avx2 for byte-lane
        // sign-extend, V128 needs Sse41. 1-byte types skip the x86 requirement.
        private static (bool emitSimd, bool useV256) WhereScalarSimdMode<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();
            bool canSimdDtype = elementSize <= 8 && IsSimdSupported<T>();
            bool needsX86 = elementSize > 1;
            bool useV256 = VectorBits >= 256 && (!needsX86 || Avx2.IsSupported);
            bool useV128 = !useV256 && VectorBits >= 128 && (!needsX86 || Sse41.IsSupported);
            return (canSimdDtype && (useV256 || useV128), useV256);
        }

        #endregion

        #region IL Emission: WhereScalarX

        /// <summary>
        /// Emit: result[i] = cond[i] ? scalarX : y[i]
        ///
        /// Layout (similar to the contig WhereKernel but x is a hoisted scalar):
        ///   Ldarg_0 = bool* cond
        ///   Ldarg_1 = T   scalarX
        ///   Ldarg_2 = T*  y
        ///   Ldarg_3 = T*  result
        ///   Ldarg_S 4 = long count
        ///
        /// Pre-loop:
        ///   scalarXVec = V&lt;T&gt;.Create(scalarX)
        ///
        /// SIMD body (per vectorCount lanes):
        ///   mask = expand(cond + i)
        ///   yVec = V.Load(y + i*elemSize)
        ///   V.Store(result + i*elemSize, V.ConditionalSelect(mask, scalarXVec, yVec))
        /// </summary>
        private static WhereScalarXKernel<T> GenerateWhereScalarXKernelIL<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();
            var (emitSimd, useV256) = WhereScalarSimdMode<T>();

            var dm = new DynamicMethod(
                name: $"IL_WhereScalarX_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(T), typeof(T*), typeof(T*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var locI = il.DeclareLocal(typeof(long));

            // i = 0;
            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            LocalBuilder locScalarXVec = null;
            if (emitSimd)
            {
                // scalarXVec = V<T>.Create(scalarX)  -- ONCE before the loop.
                int simdBits = useV256 ? 256 : 128;
                var vType = VectorMethodCache.CreateBroadcast(simdBits, typeof(T));

                locScalarXVec = il.DeclareLocal(VectorMethodCache.V(simdBits, typeof(T)));
                il.Emit(OpCodes.Ldarg_1);  // scalarX
                il.EmitCall(OpCodes.Call, vType, null);
                il.Emit(OpCodes.Stloc, locScalarXVec);

                EmitWhereScalarXSimdLoop<T>(il, locI, useV256, locScalarXVec);
            }

            // Scalar tail (also handles non-SIMD platforms / unsupported dtypes).
            EmitWhereScalarXScalarTail<T>(il, locI);

            il.Emit(OpCodes.Ret);

            return (WhereScalarXKernel<T>)dm.CreateDelegate(typeof(WhereScalarXKernel<T>));
        }

        private static void EmitWhereScalarXSimdLoop<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locScalarXVec) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = useV256 ? (32 / elementSize) : (16 / elementSize);

            var locVectorEnd = il.DeclareLocal(typeof(long));
            var lblVecHead = il.DefineLabel();
            var lblVecEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            il.MarkLabel(lblVecHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVecEnd);

            EmitWhereScalarXSimdBody<T>(il, locI, useV256, locScalarXVec, elementSize, 0);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblVecHead);

            il.MarkLabel(lblVecEnd);
        }

        /// <summary>
        /// One SIMD body iteration. <paramref name="laneOffset"/> shifts the base index by
        /// that many elements (in element units, not bytes) — used to emit multiple bodies
        /// in an unrolled outer loop without advancing <c>locI</c> between them.
        /// </summary>
        private static void EmitWhereScalarXSimdBody<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locScalarXVec, long elementSize, long laneOffset) where T : unmanaged
        {
            int simdBits = useV256 ? 256 : 128;
            var loadM = VectorMethodCache.Load(simdBits, typeof(T));
            var storeM = VectorMethodCache.Store(simdBits, typeof(T));
            var selectM = VectorMethodCache.ConditionalSelect(simdBits, typeof(T));

            // Mask: cond + (i + laneOffset)
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, laneOffset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            if (useV256) EmitInlineMaskCreationV256(il, (int)elementSize);
            else         EmitInlineMaskCreationV128(il, (int)elementSize);

            // scalarXVec
            il.Emit(OpCodes.Ldloc, locScalarXVec);

            // yVec = Load(y + (i + laneOffset)*elementSize)
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, laneOffset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, loadM, null);

            // ConditionalSelect(mask, scalarXVec, yVec)
            il.EmitCall(OpCodes.Call, selectM, null);

            // Store at result + (i + laneOffset)*elementSize
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0)
            {
                il.Emit(OpCodes.Ldc_I8, laneOffset);
                il.Emit(OpCodes.Add);
            }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, storeM, null);
        }

        private static void EmitWhereScalarXScalarTail<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            var typeCode = InfoOf<T>.NPTypeCode;

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblTakeY = il.DefineLabel();
            var lblStore = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblEnd);

            // result + i*elementSize  (kept on stack for the Store)
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            // bool c = cond[i]
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblTakeY);

            // True: load scalarX
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Br, lblStore);

            // False: load y[i]
            il.MarkLabel(lblTakeY);
            il.Emit(OpCodes.Ldarg_2);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);

            il.MarkLabel(lblStore);
            EmitStoreIndirect(il, typeCode);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        #endregion

        #region IL Emission: WhereScalarY

        /// <summary>
        /// result[i] = cond[i] ? x[i] : scalarY
        ///   Ldarg_0 = bool* cond
        ///   Ldarg_1 = T*   x
        ///   Ldarg_2 = T    scalarY
        ///   Ldarg_3 = T*   result
        ///   Ldarg_S 4 = long count
        /// </summary>
        private static WhereScalarYKernel<T> GenerateWhereScalarYKernelIL<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();
            var (emitSimd, useV256) = WhereScalarSimdMode<T>();

            var dm = new DynamicMethod(
                name: $"IL_WhereScalarY_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(T*), typeof(T), typeof(T*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var locI = il.DeclareLocal(typeof(long));

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            LocalBuilder locScalarYVec = null;
            if (emitSimd)
            {
                int simdBits = useV256 ? 256 : 128;
                var createM = VectorMethodCache.CreateBroadcast(simdBits, typeof(T));

                locScalarYVec = il.DeclareLocal(VectorMethodCache.V(simdBits, typeof(T)));
                il.Emit(OpCodes.Ldarg_2);  // scalarY
                il.EmitCall(OpCodes.Call, createM, null);
                il.Emit(OpCodes.Stloc, locScalarYVec);

                EmitWhereScalarYSimdLoop<T>(il, locI, useV256, locScalarYVec);
            }

            EmitWhereScalarYScalarTail<T>(il, locI);

            il.Emit(OpCodes.Ret);
            return (WhereScalarYKernel<T>)dm.CreateDelegate(typeof(WhereScalarYKernel<T>));
        }

        private static void EmitWhereScalarYSimdLoop<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locScalarYVec) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = useV256 ? (32 / elementSize) : (16 / elementSize);

            var locVectorEnd = il.DeclareLocal(typeof(long));
            var lblVecHead = il.DefineLabel();
            var lblVecEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            il.MarkLabel(lblVecHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVecEnd);

            EmitWhereScalarYSimdBody<T>(il, locI, useV256, locScalarYVec, elementSize, 0);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblVecHead);

            il.MarkLabel(lblVecEnd);
        }

        private static void EmitWhereScalarYSimdBody<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locScalarYVec, long elementSize, long laneOffset) where T : unmanaged
        {
            int simdBits = useV256 ? 256 : 128;
            var loadM = VectorMethodCache.Load(simdBits, typeof(T));
            var storeM = VectorMethodCache.Store(simdBits, typeof(T));
            var selectM = VectorMethodCache.ConditionalSelect(simdBits, typeof(T));

            // Mask
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            if (useV256) EmitInlineMaskCreationV256(il, (int)elementSize);
            else         EmitInlineMaskCreationV128(il, (int)elementSize);

            // xVec = Load(x + (i+laneOffset)*elementSize)
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, loadM, null);

            // scalarYVec
            il.Emit(OpCodes.Ldloc, locScalarYVec);

            // ConditionalSelect(mask, xVec, scalarYVec)
            il.EmitCall(OpCodes.Call, selectM, null);

            // Store at result + (i+laneOffset)*elementSize
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, storeM, null);
        }

        private static void EmitWhereScalarYScalarTail<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            var typeCode = InfoOf<T>.NPTypeCode;

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblTakeY = il.DefineLabel();
            var lblStore = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblEnd);

            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblTakeY);

            // True: load x[i]
            il.Emit(OpCodes.Ldarg_1);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect(il, typeCode);
            il.Emit(OpCodes.Br, lblStore);

            // False: load scalarY
            il.MarkLabel(lblTakeY);
            il.Emit(OpCodes.Ldarg_2);

            il.MarkLabel(lblStore);
            EmitStoreIndirect(il, typeCode);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        #endregion

        #region IL Emission: WhereScalarXY

        /// <summary>
        /// result[i] = cond[i] ? scalarX : scalarY
        ///   Ldarg_0 = bool* cond
        ///   Ldarg_1 = T    scalarX
        ///   Ldarg_2 = T    scalarY
        ///   Ldarg_3 = T*   result
        ///   Ldarg_S 4 = long count
        /// </summary>
        private static WhereScalarXYKernel<T> GenerateWhereScalarXYKernelIL<T>() where T : unmanaged
        {
            int elementSize = Unsafe.SizeOf<T>();
            var (emitSimd, useV256) = WhereScalarSimdMode<T>();

            var dm = new DynamicMethod(
                name: $"IL_WhereScalarXY_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(bool*), typeof(T), typeof(T), typeof(T*), typeof(long) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true);

            var il = dm.GetILGenerator();
            var locI = il.DeclareLocal(typeof(long));

            il.Emit(OpCodes.Ldc_I8, 0L);
            il.Emit(OpCodes.Stloc, locI);

            LocalBuilder locXVec = null;
            LocalBuilder locYVec = null;
            if (emitSimd)
            {
                int simdBits = useV256 ? 256 : 128;
                var createM = VectorMethodCache.CreateBroadcast(simdBits, typeof(T));

                var vecT = VectorMethodCache.V(simdBits, typeof(T));
                locXVec = il.DeclareLocal(vecT);
                locYVec = il.DeclareLocal(vecT);

                il.Emit(OpCodes.Ldarg_1);
                il.EmitCall(OpCodes.Call, createM, null);
                il.Emit(OpCodes.Stloc, locXVec);

                il.Emit(OpCodes.Ldarg_2);
                il.EmitCall(OpCodes.Call, createM, null);
                il.Emit(OpCodes.Stloc, locYVec);

                EmitWhereScalarXYSimdLoop<T>(il, locI, useV256, locXVec, locYVec);
            }

            EmitWhereScalarXYScalarTail<T>(il, locI);

            il.Emit(OpCodes.Ret);
            return (WhereScalarXYKernel<T>)dm.CreateDelegate(typeof(WhereScalarXYKernel<T>));
        }

        private static void EmitWhereScalarXYSimdLoop<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locXVec, LocalBuilder locYVec) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            long vectorCount = useV256 ? (32 / elementSize) : (16 / elementSize);

            var locVectorEnd = il.DeclareLocal(typeof(long));
            var lblVecHead = il.DefineLabel();
            var lblVecEnd = il.DefineLabel();

            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            il.MarkLabel(lblVecHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblVecEnd);

            EmitWhereScalarXYSimdBody<T>(il, locI, useV256, locXVec, locYVec, elementSize, 0);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblVecHead);

            il.MarkLabel(lblVecEnd);
        }

        private static void EmitWhereScalarXYSimdBody<T>(ILGenerator il, LocalBuilder locI, bool useV256, LocalBuilder locXVec, LocalBuilder locYVec, long elementSize, long laneOffset) where T : unmanaged
        {
            int simdBits = useV256 ? 256 : 128;
            var storeM = VectorMethodCache.Store(simdBits, typeof(T));
            var selectM = VectorMethodCache.ConditionalSelect(simdBits, typeof(T));

            // Mask
            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            if (useV256) EmitInlineMaskCreationV256(il, (int)elementSize);
            else         EmitInlineMaskCreationV128(il, (int)elementSize);

            // ConditionalSelect(mask, xVec, yVec)
            il.Emit(OpCodes.Ldloc, locXVec);
            il.Emit(OpCodes.Ldloc, locYVec);
            il.EmitCall(OpCodes.Call, selectM, null);

            // Store at result + (i+laneOffset)*elementSize
            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            if (laneOffset != 0) { il.Emit(OpCodes.Ldc_I8, laneOffset); il.Emit(OpCodes.Add); }
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.EmitCall(OpCodes.Call, storeM, null);
        }

        private static void EmitWhereScalarXYScalarTail<T>(ILGenerator il, LocalBuilder locI) where T : unmanaged
        {
            long elementSize = Unsafe.SizeOf<T>();
            var typeCode = InfoOf<T>.NPTypeCode;

            var lblHead = il.DefineLabel();
            var lblEnd = il.DefineLabel();
            var lblTakeY = il.DefineLabel();
            var lblStore = il.DefineLabel();

            il.MarkLabel(lblHead);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_S, (byte)4);
            il.Emit(OpCodes.Bge, lblEnd);

            il.Emit(OpCodes.Ldarg_3);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);

            il.Emit(OpCodes.Ldarg_0);
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Ldind_U1);
            il.Emit(OpCodes.Brfalse, lblTakeY);

            il.Emit(OpCodes.Ldarg_1);  // scalarX
            il.Emit(OpCodes.Br, lblStore);

            il.MarkLabel(lblTakeY);
            il.Emit(OpCodes.Ldarg_2);  // scalarY

            il.MarkLabel(lblStore);
            EmitStoreIndirect(il, typeCode);

            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I8, 1L);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);
            il.Emit(OpCodes.Br, lblHead);

            il.MarkLabel(lblEnd);
        }

        #endregion
    }
}
