using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Unary.Vector.cs - SIMD Vector IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryVectorOperation - main vector op dispatch
//   - EmitVectorSquare - x * x
//   - EmitVectorReciprocal - 1 / x
//   - EmitVectorDeg2Rad, EmitVectorRad2Deg - angle conversion
//   - EmitVectorBitwiseNot - ~x
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region Unary Vector IL Emission
        /// <summary>
        /// Emit Vector unary operation (adapts to V128/V256/V512).
        /// </summary>
        internal static void EmitUnaryVectorOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            var clrType = GetClrType(type);

            // Specialized emitters for ops that can't be expressed as a single container call.
            switch (op)
            {
                case UnaryOp.Square:     EmitVectorSquare(il, clrType);     return;
                case UnaryOp.Reciprocal: EmitVectorReciprocal(il, clrType); return;
                case UnaryOp.Deg2Rad:    EmitVectorScale(il, clrType, Math.PI / 180.0); return;
                case UnaryOp.Rad2Deg:    EmitVectorScale(il, clrType, 180.0 / Math.PI); return;
                case UnaryOp.BitwiseNot:
                    il.EmitCall(OpCodes.Call, VectorMethodCache.OnesComplement(VectorBits, clrType), null);
                    return;
            }

            string methodName = op switch
            {
                UnaryOp.Negate => "op_UnaryNegation",
                UnaryOp.Abs => "Abs",
                UnaryOp.Sqrt => "Sqrt",
                UnaryOp.Floor => "Floor",
                UnaryOp.Ceil => "Ceiling",  // Vector uses "Ceiling" not "Ceil"
                UnaryOp.Round => "Round",
                UnaryOp.Truncate => "Truncate",
                _ => throw new NotSupportedException($"SIMD operation {op} not supported")
            };

            MethodInfo method;
            if (op == UnaryOp.Negate)
            {
                // Negation is an operator on Vector<T>.
                method = VectorMethodCache.V(VectorBits, clrType).GetMethod(methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    null, new[] { VectorMethodCache.V(VectorBits, clrType) }, null)
                    ?? throw new InvalidOperationException($"Could not find {methodName} for Vector{VectorBits}<{clrType.Name}>");
            }
            else if (op == UnaryOp.Floor || op == UnaryOp.Ceil || op == UnaryOp.Round || op == UnaryOp.Truncate)
            {
                // Floor/Ceiling/Round/Truncate are NOT generic — overloaded per-type.
                var vT = VectorMethodCache.V(VectorBits, clrType);
                method = VectorMethodCache.Container(VectorBits).GetMethod(methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vT }, null)
                    ?? throw new InvalidOperationException($"Could not find {methodName} for Vector{VectorBits}<{clrType.Name}>");
            }
            else
            {
                // Abs, Sqrt are generic static methods on Vector container.
                method = VectorMethodCache.Generic(VectorBits, methodName, clrType, paramCount: 1);
            }

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector square: x * x using Vector.Multiply.
        /// </summary>
        private static void EmitVectorSquare(ILGenerator il, Type clrType)
        {
            // Stack has: vector x — duplicate then multiply.
            il.Emit(OpCodes.Dup);
            il.EmitCall(OpCodes.Call, VectorMethodCache.MultiplyVectorVector(VectorBits, clrType), null);
        }

        /// <summary>
        /// Emit Vector reciprocal: 1 / x using Vector.Divide with ones vector.
        /// </summary>
        private static void EmitVectorReciprocal(ILGenerator il, Type clrType)
        {
            var vectorType = VectorMethodCache.V(VectorBits, clrType);
            var locX = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locX);

            // Create ones vector via Vector<T>.One property.
            il.EmitCall(OpCodes.Call, VectorMethodCache.One(VectorBits, clrType), null);
            il.Emit(OpCodes.Ldloc, locX);

            il.EmitCall(OpCodes.Call, VectorMethodCache.DivideVectorVector(VectorBits, clrType), null);
        }

        /// <summary>
        /// Emit <c>x * factor</c> via <c>Vector.Multiply(Vector.Create(factor), x)</c> — used by
        /// Deg2Rad and Rad2Deg with the appropriate scalar factor.
        /// </summary>
        private static void EmitVectorScale(ILGenerator il, Type clrType, double factor)
        {
            // Stack: [x_vector] — push the scalar, broadcast, then multiply.
            if (clrType == typeof(float))
                il.Emit(OpCodes.Ldc_R4, (float)factor);
            else
                il.Emit(OpCodes.Ldc_R8, factor);

            il.EmitCall(OpCodes.Call, VectorMethodCache.CreateBroadcast(VectorBits, clrType), null);

            // Swap stack so we have [x, factor] for the multiply.
            var vectorType = VectorMethodCache.V(VectorBits, clrType);
            var locFactor = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locFactor);
            il.Emit(OpCodes.Ldloc, locFactor);

            il.EmitCall(OpCodes.Call, VectorMethodCache.MultiplyVectorVector(VectorBits, clrType), null);
        }

        #endregion
    }
}
