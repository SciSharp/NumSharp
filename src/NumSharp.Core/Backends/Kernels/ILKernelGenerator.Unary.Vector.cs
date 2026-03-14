using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Unary.Vector.cs - SIMD Vector IL Emission
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
    public sealed partial class ILKernelGenerator
    {
        #region Unary Vector IL Emission
        /// <summary>
        /// Emit Vector unary operation (adapts to V128/V256/V512).
        /// </summary>
        private static void EmitUnaryVectorOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            var containerType = GetVectorContainerType();
            var clrType = GetClrType(type);
            var vectorType = GetVectorType(clrType);

            // Handle special cases that don't map to a single method call
            if (op == UnaryOp.Square)
            {
                // Square = x * x: duplicate and multiply
                EmitVectorSquare(il, containerType, clrType, vectorType);
                return;
            }

            if (op == UnaryOp.Reciprocal)
            {
                // Reciprocal = 1 / x: create ones vector and divide
                EmitVectorReciprocal(il, containerType, clrType, vectorType);
                return;
            }

            if (op == UnaryOp.Deg2Rad)
            {
                // Deg2Rad = x * (π/180): multiply by constant vector
                EmitVectorDeg2Rad(il, containerType, clrType, vectorType);
                return;
            }

            if (op == UnaryOp.Rad2Deg)
            {
                // Rad2Deg = x * (180/π): multiply by constant vector
                EmitVectorRad2Deg(il, containerType, clrType, vectorType);
                return;
            }

            if (op == UnaryOp.BitwiseNot)
            {
                // BitwiseNot = ~x: OnesComplement
                EmitVectorBitwiseNot(il, containerType, clrType, vectorType);
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

            MethodInfo? method;

            if (op == UnaryOp.Negate)
            {
                // Negation is an operator on Vector<T>
                method = vectorType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vectorType }, null);
            }
            else if (op == UnaryOp.Floor || op == UnaryOp.Ceil || op == UnaryOp.Round || op == UnaryOp.Truncate)
            {
                // Floor/Ceiling/Round/Truncate are NOT generic - they're overloaded for specific types
                // Use the single-parameter overload (default MidpointRounding.ToEven for Round)
                method = containerType.GetMethod(methodName, BindingFlags.Public | BindingFlags.Static,
                    null, new[] { vectorType }, null);
            }
            else
            {
                // Abs, Sqrt are generic static methods on Vector container
                method = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                    .Where(m => m.Name == methodName && m.IsGenericMethod && m.GetParameters().Length == 1)
                    .Select(m => m.MakeGenericMethod(clrType))
                    .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);
            }

            if (method == null)
                throw new InvalidOperationException($"Could not find {methodName} for {vectorType.Name}");

            il.EmitCall(OpCodes.Call, method, null);
        }

        /// <summary>
        /// Emit Vector square: x * x using Vector.Multiply.
        /// </summary>
        private static void EmitVectorSquare(ILGenerator il, Type containerType, Type clrType, Type vectorType)
        {
            // Stack has: vector x
            // We need: x * x
            // Duplicate the vector and call Multiply
            il.Emit(OpCodes.Dup);

            // Vector.Multiply<T>(Vector<T>, Vector<T>) is a generic method
            var multiplyMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Multiply" && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType &&
                                     m.GetParameters()[1].ParameterType == vectorType);

            if (multiplyMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Multiply<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, multiplyMethod, null);
        }

        /// <summary>
        /// Emit Vector reciprocal: 1 / x using Vector.Divide with ones vector.
        /// </summary>
        private static void EmitVectorReciprocal(ILGenerator il, Type containerType, Type clrType, Type vectorType)
        {
            // Stack has: vector x
            // We need: ones / x
            // Store x, create ones, load x, divide

            var locX = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locX);

            // Create ones vector: Vector<T>.One
            var oneProperty = vectorType.GetProperty("One", BindingFlags.Public | BindingFlags.Static)
                ?? throw new InvalidOperationException($"Could not find Vector<{clrType.Name}>.One");
            var oneGetter = oneProperty.GetGetMethod()
                ?? throw new InvalidOperationException($"Could not find getter for Vector<{clrType.Name}>.One");
            il.EmitCall(OpCodes.Call, oneGetter, null);

            // Load x
            il.Emit(OpCodes.Ldloc, locX);

            // Vector.Divide<T>(Vector<T>, Vector<T>)
            var divideMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Divide" && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType &&
                                     m.GetParameters()[1].ParameterType == vectorType);

            if (divideMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Divide<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, divideMethod, null);
        }

        /// <summary>
        /// Emit Vector deg2rad: x * (π/180) using Vector.Multiply with constant vector.
        /// </summary>
        private static void EmitVectorDeg2Rad(ILGenerator il, Type containerType, Type clrType, Type vectorType)
        {
            // Stack has: vector x
            // Create constant vector and multiply

            // Create Deg2Rad factor vector
            if (clrType == typeof(float))
            {
                il.Emit(OpCodes.Ldc_R4, (float)(Math.PI / 180.0));
            }
            else // double
            {
                il.Emit(OpCodes.Ldc_R8, Math.PI / 180.0);
            }

            // Vector.Create<T>(T value) - creates vector with all elements = value
            var createMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Create" && m.IsGenericMethod && m.GetParameters().Length == 1)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == clrType);

            if (createMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Create<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, createMethod, null);

            // Now stack has: [x_vector, factor_vector]
            // Swap them for multiply (need x * factor, but stack has factor on top)
            var locFactor = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locFactor);
            // Stack: [x_vector]
            il.Emit(OpCodes.Ldloc, locFactor);
            // Stack: [x_vector, factor_vector]

            // Vector.Multiply<T>(Vector<T>, Vector<T>)
            var multiplyMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Multiply" && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType &&
                                     m.GetParameters()[1].ParameterType == vectorType);

            if (multiplyMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Multiply<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, multiplyMethod, null);
        }

        /// <summary>
        /// Emit Vector rad2deg: x * (180/π) using Vector.Multiply with constant vector.
        /// </summary>
        private static void EmitVectorRad2Deg(ILGenerator il, Type containerType, Type clrType, Type vectorType)
        {
            // Stack has: vector x
            // Create constant vector and multiply

            // Create Rad2Deg factor vector
            if (clrType == typeof(float))
            {
                il.Emit(OpCodes.Ldc_R4, (float)(180.0 / Math.PI));
            }
            else // double
            {
                il.Emit(OpCodes.Ldc_R8, 180.0 / Math.PI);
            }

            // Vector.Create<T>(T value) - creates vector with all elements = value
            var createMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Create" && m.IsGenericMethod && m.GetParameters().Length == 1)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == clrType);

            if (createMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Create<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, createMethod, null);

            // Swap for multiply
            var locFactor = il.DeclareLocal(vectorType);
            il.Emit(OpCodes.Stloc, locFactor);
            il.Emit(OpCodes.Ldloc, locFactor);

            // Vector.Multiply<T>(Vector<T>, Vector<T>)
            var multiplyMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "Multiply" && m.IsGenericMethod && m.GetParameters().Length == 2)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType &&
                                     m.GetParameters()[1].ParameterType == vectorType);

            if (multiplyMethod == null)
                throw new InvalidOperationException($"Could not find Vector.Multiply<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, multiplyMethod, null);
        }

        /// <summary>
        /// Emit Vector bitwise not: ~x using Vector.OnesComplement.
        /// </summary>
        private static void EmitVectorBitwiseNot(ILGenerator il, Type containerType, Type clrType, Type vectorType)
        {
            // Stack has: vector x
            // Call Vector.OnesComplement<T>(Vector<T>)

            var onesComplementMethod = containerType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.Name == "OnesComplement" && m.IsGenericMethod && m.GetParameters().Length == 1)
                .Select(m => m.MakeGenericMethod(clrType))
                .FirstOrDefault(m => m.GetParameters()[0].ParameterType == vectorType);

            if (onesComplementMethod == null)
                throw new InvalidOperationException($"Could not find Vector.OnesComplement<{clrType.Name}>");

            il.EmitCall(OpCodes.Call, onesComplementMethod, null);
        }

        #endregion
    }
}
