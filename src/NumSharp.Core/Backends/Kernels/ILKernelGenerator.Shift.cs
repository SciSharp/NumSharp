using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Shift.cs - Shift operations (LeftShift, RightShift)
// =============================================================================
//
// OWNERSHIP: Bit shift operations for integer types
// RESPONSIBILITY:
//   - LeftShift (<<) and RightShift (>>) operations
//   - SIMD path for array << scalar (uniform shift amount)
//   - Scalar loop for array << array (element-wise different shifts)
//   - Arithmetic shift for signed types, logical shift for unsigned
//
// SUPPORTED TYPES: Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64
// NOT SUPPORTED: Boolean, Char, Single, Double, Decimal (non-integer)
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Shift operations - LeftShift and RightShift with SIMD optimization.
    /// </summary>
    public sealed partial class ILKernelGenerator
    {
        #region Shift Kernel Delegates

        /// <summary>
        /// Delegate for shift operation with scalar shift amount.
        /// This is the SIMD-optimized path for uniform shifts.
        /// </summary>
        /// <typeparam name="T">Element type (must be integer)</typeparam>
        /// <param name="input">Pointer to input data</param>
        /// <param name="output">Pointer to output data</param>
        /// <param name="shiftAmount">Number of bits to shift</param>
        /// <param name="count">Number of elements to process</param>
        public unsafe delegate void ShiftScalarKernel<T>(T* input, T* output, int shiftAmount, int count) where T : unmanaged;

        /// <summary>
        /// Delegate for shift operation with per-element shift amounts.
        /// This is the scalar loop path for element-wise shifts.
        /// </summary>
        /// <typeparam name="T">Element type (must be integer)</typeparam>
        /// <param name="input">Pointer to input data</param>
        /// <param name="shifts">Pointer to shift amounts (as int32)</param>
        /// <param name="output">Pointer to output data</param>
        /// <param name="count">Number of elements to process</param>
        public unsafe delegate void ShiftArrayKernel<T>(T* input, int* shifts, T* output, int count) where T : unmanaged;

        #endregion

        #region Shift Kernel Cache

        private static readonly ConcurrentDictionary<(BinaryOp, Type, bool), Delegate> _shiftKernelCache = new();

        /// <summary>
        /// Clear the shift kernel cache.
        /// </summary>
        public static void ClearShiftCache() => _shiftKernelCache.Clear();

        #endregion

        #region Public API

        /// <summary>
        /// Get or generate a SIMD-optimized shift kernel for uniform shift amount.
        /// </summary>
        /// <typeparam name="T">Integer element type</typeparam>
        /// <param name="isLeftShift">True for left shift, false for right shift</param>
        /// <returns>Kernel delegate or null if not supported</returns>
        public static ShiftScalarKernel<T>? GetShiftScalarKernel<T>(bool isLeftShift) where T : unmanaged
        {
            if (!Enabled || !IsShiftSupported<T>())
                return null;

            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            var key = (op, typeof(T), true); // true = scalar shift

            if (_shiftKernelCache.TryGetValue(key, out var cached))
                return (ShiftScalarKernel<T>)cached;

            try
            {
                var kernel = GenerateShiftScalarKernel<T>(isLeftShift);
                if (_shiftKernelCache.TryAdd(key, kernel))
                    return kernel;
                return (ShiftScalarKernel<T>)_shiftKernelCache[key];
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Get or generate a shift kernel for element-wise shift amounts.
        /// </summary>
        /// <typeparam name="T">Integer element type</typeparam>
        /// <param name="isLeftShift">True for left shift, false for right shift</param>
        /// <returns>Kernel delegate or null if not supported</returns>
        public static ShiftArrayKernel<T>? GetShiftArrayKernel<T>(bool isLeftShift) where T : unmanaged
        {
            if (!Enabled || !IsShiftSupported<T>())
                return null;

            var op = isLeftShift ? BinaryOp.LeftShift : BinaryOp.RightShift;
            var key = (op, typeof(T), false); // false = array shift

            if (_shiftKernelCache.TryGetValue(key, out var cached))
                return (ShiftArrayKernel<T>)cached;

            try
            {
                var kernel = GenerateShiftArrayKernel<T>(isLeftShift);
                if (_shiftKernelCache.TryAdd(key, kernel))
                    return kernel;
                return (ShiftArrayKernel<T>)_shiftKernelCache[key];
            }
            catch
            {
                return null;
            }
        }

        #endregion

        #region Kernel Generation - Scalar Shift (SIMD)

        /// <summary>
        /// Generate IL kernel for shift with scalar shift amount.
        /// Uses SIMD for the main loop and scalar for the tail.
        /// </summary>
        private static unsafe ShiftScalarKernel<T> GenerateShiftScalarKernel<T>(bool isLeftShift) where T : unmanaged
        {
            var dm = new DynamicMethod(
                name: $"IL_Shift{(isLeftShift ? "Left" : "Right")}_Scalar_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(T*), typeof(T*), typeof(int), typeof(int) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(int));           // loop counter
            var locVectorEnd = il.DeclareLocal(typeof(int));   // count - vectorCount
            var locUnrollEnd = il.DeclareLocal(typeof(int));   // count - vectorCount*4

            int elementSize = Unsafe.SizeOf<T>();
            int vectorCount = GetShiftVectorCount<T>();
            int unrollStep = vectorCount * 4;

            // Labels
            var lblUnrollLoop = il.DefineLabel();
            var lblUnrollLoopEnd = il.DefineLabel();
            var lblRemainderLoop = il.DefineLabel();
            var lblRemainderLoopEnd = il.DefineLabel();
            var lblTailLoop = il.DefineLabel();
            var lblTailLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // vectorEnd = count - vectorCount
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locVectorEnd);

            // unrollEnd = count - vectorCount*4
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Sub);
            il.Emit(OpCodes.Stloc, locUnrollEnd);

            // ========== 4x UNROLLED SIMD LOOP ==========
            il.MarkLabel(lblUnrollLoop);

            // if (i > unrollEnd) goto UnrollLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locUnrollEnd);
            il.Emit(OpCodes.Bgt, lblUnrollLoopEnd);

            // Process 4 vectors per iteration
            for (int u = 0; u < 4; u++)
            {
                int offset = vectorCount * u;

                // Load input vector at (i + offset)
                il.Emit(OpCodes.Ldarg_0);                      // input
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorLoad<T>(il);

                // Load shift amount
                il.Emit(OpCodes.Ldarg_2);                      // shiftAmount

                // Perform vector shift
                EmitVectorShift<T>(il, isLeftShift);

                // Store result at (i + offset)
                il.Emit(OpCodes.Ldarg_1);                      // output
                il.Emit(OpCodes.Ldloc, locI);
                if (offset > 0)
                {
                    il.Emit(OpCodes.Ldc_I4, offset);
                    il.Emit(OpCodes.Add);
                }
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4, elementSize);
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                EmitVectorStore<T>(il);
            }

            // i += vectorCount * 4
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, unrollStep);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblUnrollLoop);
            il.MarkLabel(lblUnrollLoopEnd);

            // ========== REMAINDER SIMD LOOP (0-3 vectors) ==========
            il.MarkLabel(lblRemainderLoop);

            // if (i > vectorEnd) goto RemainderLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldloc, locVectorEnd);
            il.Emit(OpCodes.Bgt, lblRemainderLoopEnd);

            // Load input vector
            il.Emit(OpCodes.Ldarg_0);                      // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorLoad<T>(il);

            // Load shift amount
            il.Emit(OpCodes.Ldarg_2);                      // shiftAmount

            // Perform vector shift
            EmitVectorShift<T>(il, isLeftShift);

            // Store result
            il.Emit(OpCodes.Ldarg_1);                      // output
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitVectorStore<T>(il);

            // i += vectorCount
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4, vectorCount);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblRemainderLoop);
            il.MarkLabel(lblRemainderLoopEnd);

            // ========== TAIL LOOP (scalar) ==========
            il.MarkLabel(lblTailLoop);

            // if (i >= count) goto TailLoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Bge, lblTailLoopEnd);

            // output[i] = input[i] << shiftAmount (or >>)
            EmitScalarShiftBody<T>(il, isLeftShift, locI, elementSize, useArrayShift: false);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblTailLoop);
            il.MarkLabel(lblTailLoopEnd);

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ShiftScalarKernel<T>>();
        }

        #endregion

        #region Kernel Generation - Array Shift (Scalar Loop)

        /// <summary>
        /// Generate IL kernel for shift with per-element shift amounts.
        /// Uses pure scalar loop since SIMD doesn't support variable shifts per element.
        /// </summary>
        private static unsafe ShiftArrayKernel<T> GenerateShiftArrayKernel<T>(bool isLeftShift) where T : unmanaged
        {
            var dm = new DynamicMethod(
                name: $"IL_Shift{(isLeftShift ? "Left" : "Right")}_Array_{typeof(T).Name}",
                returnType: typeof(void),
                parameterTypes: new[] { typeof(T*), typeof(int*), typeof(T*), typeof(int) },
                owner: typeof(ILKernelGenerator),
                skipVisibility: true
            );

            var il = dm.GetILGenerator();

            // Locals
            var locI = il.DeclareLocal(typeof(int));           // loop counter

            int elementSize = Unsafe.SizeOf<T>();

            // Labels
            var lblLoop = il.DefineLabel();
            var lblLoopEnd = il.DefineLabel();

            // i = 0
            il.Emit(OpCodes.Ldc_I4_0);
            il.Emit(OpCodes.Stloc, locI);

            // ========== SCALAR LOOP ==========
            il.MarkLabel(lblLoop);

            // if (i >= count) goto LoopEnd
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldarg_3);                      // count
            il.Emit(OpCodes.Bge, lblLoopEnd);

            // output[i] = input[i] << shifts[i]
            EmitScalarShiftBody<T>(il, isLeftShift, locI, elementSize, useArrayShift: true);

            // i++
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Ldc_I4_1);
            il.Emit(OpCodes.Add);
            il.Emit(OpCodes.Stloc, locI);

            il.Emit(OpCodes.Br, lblLoop);
            il.MarkLabel(lblLoopEnd);

            il.Emit(OpCodes.Ret);

            return dm.CreateDelegate<ShiftArrayKernel<T>>();
        }

        #endregion

        #region IL Emission Helpers

        /// <summary>
        /// Check if type is supported for shift operations.
        /// </summary>
        private static bool IsShiftSupported<T>() where T : unmanaged
        {
            return typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(short) ||
                   typeof(T) == typeof(ushort) ||
                   typeof(T) == typeof(int) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(long) ||
                   typeof(T) == typeof(ulong);
        }

        /// <summary>
        /// Check if type is unsigned for right shift selection.
        /// </summary>
        private static bool IsUnsignedType<T>() where T : unmanaged
        {
            return typeof(T) == typeof(byte) ||
                   typeof(T) == typeof(ushort) ||
                   typeof(T) == typeof(uint) ||
                   typeof(T) == typeof(ulong);
        }

        /// <summary>
        /// Get vector count for shift operations based on current SIMD width.
        /// </summary>
        private static int GetShiftVectorCount<T>() where T : unmanaged
        {
            return VectorBits switch
            {
                512 => Vector512<T>.Count,
                256 => Vector256<T>.Count,
                128 => Vector128<T>.Count,
                _ => 1
            };
        }

        /// <summary>
        /// Emit vector shift operation.
        /// Stack: [vector, shiftAmount] -> [shifted vector]
        /// </summary>
        private static void EmitVectorShift<T>(ILGenerator il, bool isLeftShift) where T : unmanaged
        {
            var containerType = GetVectorContainerType();
            var vectorType = GetVectorType(typeof(T));

            string methodName;
            if (isLeftShift)
            {
                methodName = "ShiftLeft";
            }
            else
            {
                // For right shift: arithmetic for signed, logical for unsigned
                methodName = IsUnsignedType<T>() ? "ShiftRightLogical" : "ShiftRightArithmetic";
            }

            // Find the non-generic overload that takes Vector256<T> and int
            var methods = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == methodName && !m.IsGenericMethod)
                .ToList();

            // Find method matching our vector type
            var shiftMethod = methods.FirstOrDefault(m =>
            {
                var parms = m.GetParameters();
                return parms.Length == 2 &&
                       parms[0].ParameterType == vectorType &&
                       parms[1].ParameterType == typeof(int);
            });

            if (shiftMethod == null)
                throw new InvalidOperationException($"Could not find {methodName} for {vectorType.Name}");

            il.EmitCall(OpCodes.Call, shiftMethod, null);
        }

        /// <summary>
        /// Emit scalar shift operation body.
        /// For scalar shift (useArrayShift=false): output[i] = input[i] << shiftAmount (arg2)
        /// For array shift (useArrayShift=true): output[i] = input[i] << shifts[i] (arg1)
        /// </summary>
        private static void EmitScalarShiftBody<T>(ILGenerator il, bool isLeftShift, LocalBuilder locI, int elementSize, bool useArrayShift) where T : unmanaged
        {
            // Address: output + i * elementSize
            if (useArrayShift)
            {
                il.Emit(OpCodes.Ldarg_2);                      // output (arg2 for array shift)
            }
            else
            {
                il.Emit(OpCodes.Ldarg_1);                      // output (arg1 for scalar shift)
            }
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);

            // Load input[i]
            il.Emit(OpCodes.Ldarg_0);                      // input
            il.Emit(OpCodes.Ldloc, locI);
            il.Emit(OpCodes.Conv_I);
            il.Emit(OpCodes.Ldc_I4, elementSize);
            il.Emit(OpCodes.Mul);
            il.Emit(OpCodes.Add);
            EmitLoadIndirect<T>(il);

            // Load shift amount
            if (useArrayShift)
            {
                // Load shifts[i]
                il.Emit(OpCodes.Ldarg_1);                  // shifts (arg1 for array shift)
                il.Emit(OpCodes.Ldloc, locI);
                il.Emit(OpCodes.Conv_I);
                il.Emit(OpCodes.Ldc_I4_4);                 // sizeof(int)
                il.Emit(OpCodes.Mul);
                il.Emit(OpCodes.Add);
                il.Emit(OpCodes.Ldind_I4);
            }
            else
            {
                // Load scalar shift amount
                il.Emit(OpCodes.Ldarg_2);                  // shiftAmount
            }

            // Perform scalar shift
            EmitScalarShift<T>(il, isLeftShift);

            // Store to output[i]
            EmitStoreIndirect<T>(il);
        }

        /// <summary>
        /// Emit scalar shift instruction.
        /// Stack: [value, shiftAmount] -> [result]
        /// </summary>
        private static void EmitScalarShift<T>(ILGenerator il, bool isLeftShift) where T : unmanaged
        {
            if (isLeftShift)
            {
                // Left shift is the same for all integer types
                il.Emit(OpCodes.Shl);
            }
            else
            {
                // Right shift: Shr for signed (arithmetic), Shr_Un for unsigned (logical)
                if (IsUnsignedType<T>())
                {
                    il.Emit(OpCodes.Shr_Un);
                }
                else
                {
                    il.Emit(OpCodes.Shr);
                }
            }

            // Apply appropriate truncation for smaller types
            if (typeof(T) == typeof(byte))
            {
                il.Emit(OpCodes.Conv_U1);
            }
            else if (typeof(T) == typeof(short))
            {
                il.Emit(OpCodes.Conv_I2);
            }
            else if (typeof(T) == typeof(ushort))
            {
                il.Emit(OpCodes.Conv_U2);
            }
            // int, uint, long, ulong don't need truncation
        }

        #endregion
    }
}
