using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Unary.Predicate.cs - Predicate IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitIsFiniteCall - float.IsFinite/double.IsFinite
//   - EmitIsNanCall - float.IsNaN/double.IsNaN
//   - EmitIsInfCall - float.IsInfinity/double.IsInfinity
//   - Integer types always return true/false appropriately
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Unary Predicate IL Emission
        /// <summary>
        /// Emit IsFinite check.
        /// For float/double: calls float.IsFinite/double.IsFinite
        /// For integer types: always true (integers cannot be infinite or NaN)
        /// </summary>
        private static void EmitIsFiniteCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsFinite(x)
                var method = typeof(float).GetMethod("IsFinite", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsFinite(x)
                var method = typeof(double).GetMethod("IsFinite", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always true
                // Pop the value from stack and push true (1)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_1);
            }
        }

        /// <summary>
        /// Emit IsNaN check.
        /// For float/double: calls float.IsNaN/double.IsNaN
        /// For integer types: always false (integers cannot be NaN)
        /// </summary>
        private static void EmitIsNanCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsNaN(x)
                var method = typeof(float).GetMethod("IsNaN", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsNaN(x)
                var method = typeof(double).GetMethod("IsNaN", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always false
                // Pop the value from stack and push false (0)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_0);
            }
        }

        /// <summary>
        /// Emit IsInfinity check.
        /// For float/double: calls float.IsInfinity/double.IsInfinity
        /// For integer types: always false (integers cannot be infinite)
        /// </summary>
        private static void EmitIsInfCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // float.IsInfinity(x)
                var method = typeof(float).GetMethod("IsInfinity", new[] { typeof(float) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else if (type == NPTypeCode.Double)
            {
                // double.IsInfinity(x)
                var method = typeof(double).GetMethod("IsInfinity", new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);
            }
            else
            {
                // For all integer types: always false
                // Pop the value from stack and push false (0)
                il.Emit(OpCodes.Pop);
                il.Emit(OpCodes.Ldc_I4_0);
            }
        }

        #endregion
    }
}
