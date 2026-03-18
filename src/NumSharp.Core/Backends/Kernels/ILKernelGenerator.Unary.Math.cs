using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Unary.Math.cs - Math Function IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryScalarOperation - main dispatch for all unary ops
//   - EmitMathCall - Math.X/MathF.X function emission
//   - Trig functions: Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh
//   - Exp/Log: Exp, Exp2, Log, Log2, Log10, Log1p, Expm1
//   - Rounding: Floor, Ceil, Round, Truncate
//   - Sign, Reciprocal, Deg2Rad, Rad2Deg
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Unary Math IL Emission
        /// <summary>
        /// Emit unary scalar operation.
        /// </summary>
        internal static void EmitUnaryScalarOperation(ILGenerator il, UnaryOp op, NPTypeCode type)
        {
            // Special handling for decimal
            if (type == NPTypeCode.Decimal)
            {
                EmitUnaryDecimalOperation(il, op);
                return;
            }

            switch (op)
            {
                case UnaryOp.Negate:
                    // For unsigned types, use two's complement: ~val + 1
                    // For signed types, use neg opcode
                    if (IsUnsigned(type))
                    {
                        // ~val + 1 = two's complement negation
                        il.Emit(OpCodes.Not);
                        il.Emit(OpCodes.Ldc_I4_1);
                        // Need to widen to correct type before add
                        if (type == NPTypeCode.UInt64)
                        {
                            il.Emit(OpCodes.Conv_U8);
                        }
                        il.Emit(OpCodes.Add);
                    }
                    else
                    {
                        il.Emit(OpCodes.Neg);
                    }
                    break;

                case UnaryOp.Abs:
                    EmitAbsCall(il, type);
                    break;

                case UnaryOp.Sqrt:
                    EmitMathCall(il, "Sqrt", type);
                    break;

                case UnaryOp.Exp:
                    EmitMathCall(il, "Exp", type);
                    break;

                case UnaryOp.Log:
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sin:
                    EmitMathCall(il, "Sin", type);
                    break;

                case UnaryOp.Cos:
                    EmitMathCall(il, "Cos", type);
                    break;

                case UnaryOp.Tan:
                    EmitMathCall(il, "Tan", type);
                    break;

                case UnaryOp.Sinh:
                    EmitMathCall(il, "Sinh", type);
                    break;

                case UnaryOp.Cosh:
                    EmitMathCall(il, "Cosh", type);
                    break;

                case UnaryOp.Tanh:
                    EmitMathCall(il, "Tanh", type);
                    break;

                case UnaryOp.ASin:
                    EmitMathCall(il, "Asin", type);
                    break;

                case UnaryOp.ACos:
                    EmitMathCall(il, "Acos", type);
                    break;

                case UnaryOp.ATan:
                    EmitMathCall(il, "Atan", type);
                    break;

                case UnaryOp.Exp2:
                    // Use Math.Pow(2, x) since Math.Exp2 may not be available
                    EmitExp2Call(il, type);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1: call Exp then subtract 1
                    EmitMathCall(il, "Exp", type);
                    EmitSubtractOne(il, type);
                    break;

                case UnaryOp.Log2:
                    EmitMathCall(il, "Log2", type);
                    break;

                case UnaryOp.Log10:
                    EmitMathCall(il, "Log10", type);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x): add 1 then call Log
                    EmitAddOne(il, type);
                    EmitMathCall(il, "Log", type);
                    break;

                case UnaryOp.Sign:
                    EmitSignCall(il, type);
                    break;

                case UnaryOp.Ceil:
                    EmitMathCall(il, "Ceiling", type);
                    break;

                case UnaryOp.Floor:
                    EmitMathCall(il, "Floor", type);
                    break;

                case UnaryOp.Round:
                    EmitMathCall(il, "Round", type);
                    break;

                case UnaryOp.Truncate:
                    EmitMathCall(il, "Truncate", type);
                    break;

                case UnaryOp.Reciprocal:
                    // 1 / x
                    EmitReciprocalCall(il, type);
                    break;

                case UnaryOp.Square:
                    // x * x - duplicate value and multiply
                    il.Emit(OpCodes.Dup);
                    il.Emit(OpCodes.Mul);
                    break;

                case UnaryOp.Deg2Rad:
                    // x * (π/180)
                    EmitDeg2RadCall(il, type);
                    break;

                case UnaryOp.Rad2Deg:
                    // x * (180/π)
                    EmitRad2DegCall(il, type);
                    break;

                case UnaryOp.BitwiseNot:
                    // ~x (bitwise complement)
                    il.Emit(OpCodes.Not);
                    break;

                case UnaryOp.LogicalNot:
                    // Logical NOT: x == 0 (for boolean: !x)
                    // Compare to zero and return 1 if equal, 0 otherwise
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Ceq);
                    break;

                case UnaryOp.Cbrt:
                    // Cube root - Math.Cbrt (no SIMD intrinsic)
                    EmitMathCall(il, "Cbrt", type);
                    break;

                case UnaryOp.IsFinite:
                    // Test for finiteness (not infinity and not NaN)
                    // For integer types: always true
                    // For float/double: use IsFinite static method
                    EmitIsFiniteCall(il, type);
                    break;

                case UnaryOp.IsNan:
                    // Test for NaN
                    // For integer types: always false
                    // For float/double: use IsNaN static method
                    EmitIsNanCall(il, type);
                    break;

                case UnaryOp.IsInf:
                    // Test for infinity (positive or negative)
                    // For integer types: always false
                    // For float/double: use IsInfinity static method
                    EmitIsInfCall(il, type);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported");
            }
        }

        /// <summary>
        /// Emit call to Math.X method with appropriate overload.
        /// </summary>
        private static void EmitMathCall(ILGenerator il, string methodName, NPTypeCode type)
        {
            MethodInfo? method;

            if (type == NPTypeCode.Single)
            {
                // Use MathF for float
                method = typeof(MathF).GetMethod(methodName, new[] { typeof(float) });
            }
            else if (type == NPTypeCode.Double)
            {
                // Use Math for double
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
            }
            else
            {
                // For integer types, convert to double, call Math, convert back
                // Stack has: value (as output type)
                // Need to: conv to double, call Math.X, conv back

                // Convert to double first
                EmitConvertToDouble(il, type);

                // Call Math.X(double)
                method = typeof(Math).GetMethod(methodName, new[] { typeof(double) });
                il.EmitCall(OpCodes.Call, method!, null);

                // Convert back to target type
                EmitConvertFromDouble(il, type);
                return;
            }

            il.EmitCall(OpCodes.Call, method!, null);
        }

        /// <summary>
        /// Convert stack value to double.
        /// </summary>
        private static void EmitConvertToDouble(ILGenerator il, NPTypeCode from)
        {
            if (from == NPTypeCode.Double)
                return;

            if (IsUnsigned(from))
                il.Emit(OpCodes.Conv_R_Un);
            il.Emit(OpCodes.Conv_R8);
        }

        /// <summary>
        /// Emit abs operation with optimized bitwise implementation for integers.
        /// For signed integers: abs(x) = (x ^ (x >> (bits-1))) - (x >> (bits-1))
        /// For unsigned integers: identity (already non-negative)
        /// For float/double: use Math.Abs/MathF.Abs (hardware-optimized)
        /// </summary>
        private static void EmitAbsCall(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    // Use MathF.Abs for float - has hardware support
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFAbsFloat, null);
                    break;

                case NPTypeCode.Double:
                    // Use Math.Abs for double - has hardware support
                    il.EmitCall(OpCodes.Call, CachedMethods.MathAbsDouble, null);
                    break;

                case NPTypeCode.Byte:
                case NPTypeCode.UInt16:
                case NPTypeCode.UInt32:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                case NPTypeCode.Boolean:
                    // Unsigned types are already non-negative - abs is identity
                    // Value is already on stack, nothing to do
                    break;

                case NPTypeCode.Int16:
                    // abs(x) = (x ^ (x >> 15)) - (x >> 15)
                    // Stack: x
                    {
                        var locSign = il.DeclareLocal(typeof(int));
                        il.Emit(OpCodes.Dup);           // x, x
                        il.Emit(OpCodes.Ldc_I4, 15);    // x, x, 15
                        il.Emit(OpCodes.Shr);           // x, (x >> 15) = sign extension (-1 or 0)
                        il.Emit(OpCodes.Stloc, locSign);// x            ; locSign = s
                        il.Emit(OpCodes.Ldloc, locSign);// x, s
                        il.Emit(OpCodes.Xor);           // x ^ s
                        il.Emit(OpCodes.Ldloc, locSign);// (x ^ s), s
                        il.Emit(OpCodes.Sub);           // (x ^ s) - s = abs(x)
                        il.Emit(OpCodes.Conv_I2);       // Ensure result fits in short
                    }
                    break;

                case NPTypeCode.Int32:
                    // abs(x) = (x ^ (x >> 31)) - (x >> 31)
                    // Stack: x
                    {
                        var locSign = il.DeclareLocal(typeof(int));
                        il.Emit(OpCodes.Dup);           // x, x
                        il.Emit(OpCodes.Ldc_I4, 31);    // x, x, 31
                        il.Emit(OpCodes.Shr);           // x, (x >> 31) = sign extension (-1 or 0)
                        il.Emit(OpCodes.Stloc, locSign);// x            ; locSign = s
                        il.Emit(OpCodes.Ldloc, locSign);// x, s
                        il.Emit(OpCodes.Xor);           // x ^ s
                        il.Emit(OpCodes.Ldloc, locSign);// (x ^ s), s
                        il.Emit(OpCodes.Sub);           // (x ^ s) - s = abs(x)
                    }
                    break;

                case NPTypeCode.Int64:
                    // abs(x) = (x ^ (x >> 63)) - (x >> 63)
                    // Stack: x (as int64)
                    {
                        var locSign = il.DeclareLocal(typeof(long));
                        il.Emit(OpCodes.Dup);           // x, x
                        il.Emit(OpCodes.Ldc_I4, 63);    // x, x, 63
                        il.Emit(OpCodes.Shr);           // x, (x >> 63) = sign extension (-1 or 0)
                        il.Emit(OpCodes.Stloc, locSign);// x            ; locSign = s
                        il.Emit(OpCodes.Ldloc, locSign);// x, s
                        il.Emit(OpCodes.Xor);           // x ^ s
                        il.Emit(OpCodes.Ldloc, locSign);// (x ^ s), s
                        il.Emit(OpCodes.Sub);           // (x ^ s) - s = abs(x)
                    }
                    break;

                default:
                    throw new NotSupportedException($"Abs not supported for type {type}");
            }
        }

        /// <summary>
        /// Emit 2^x calculation using Math.Pow(2, x).
        /// </summary>
        private static void EmitExp2Call(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // For float: convert to double, call Pow, convert back
                il.Emit(OpCodes.Conv_R8);
                il.Emit(OpCodes.Ldc_R8, 2.0);
                // Stack: [exponent, base] - but Pow expects (base, exponent)
                // Need to swap them
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                // Now push base then exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
                il.Emit(OpCodes.Conv_R4);
            }
            else if (type == NPTypeCode.Double)
            {
                // For double: just call Pow
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
            }
            else
            {
                // For integer types: convert to double, call Pow, convert back
                EmitConvertToDouble(il, type);
                var locExp = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locExp);  // Save exponent
                il.Emit(OpCodes.Ldc_R8, 2.0);
                il.Emit(OpCodes.Ldloc, locExp);
                il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
                EmitConvertFromDouble(il, type);
            }
        }

        /// <summary>
        /// Emit subtraction of 1 from the value on stack.
        /// Used for expm1 = exp(x) - 1.
        /// </summary>
        private static void EmitSubtractOne(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Ldc_R4, 1.0f);
                    il.Emit(OpCodes.Sub);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
                default:
                    // For integer types, value is already double from math call
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    break;
            }
        }

        /// <summary>
        /// Emit addition of 1 to the value on stack.
        /// Used for log1p = log(1 + x).
        /// </summary>
        private static void EmitAddOne(ILGenerator il, NPTypeCode type)
        {
            // Convert to appropriate float type first, then add 1
            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, 1.0f);
                il.Emit(OpCodes.Add);
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
            else
            {
                // For integer types, convert to double first, then add 1
                // The conversion to double will happen in EmitMathCall
                EmitConvertToDouble(il, type);
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Add);
            }
        }

        /// <summary>
        /// Emit sign operation with optimized bitwise implementation for integers.
        /// For signed integers: sign(x) = (x >> (bits-1)) | ((-x) >> (bits-1) &amp; 1)
        ///   This produces: -1 for negative, 0 for zero, 1 for positive
        /// For unsigned integers: sign(x) = (x != 0) ? 1 : 0
        /// For float/double: use Math.Sign/MathF.Sign with NaN handling
        /// NumPy: sign(NaN) returns NaN, but .NET Math.Sign throws ArithmeticException.
        /// </summary>
        private static void EmitSignCall(ILGenerator il, NPTypeCode type)
        {
            switch (type)
            {
                case NPTypeCode.Single:
                    {
                        // NumPy: sign(NaN) = NaN. .NET MathF.Sign(NaN) throws.
                        // Check for NaN first: if (float.IsNaN(x)) return x; else return MathF.Sign(x);
                        var lblNotNaN = il.DefineLabel();
                        var lblEnd = il.DefineLabel();

                        il.Emit(OpCodes.Dup);  // duplicate for NaN check
                        il.EmitCall(OpCodes.Call, CachedMethods.FloatIsNaN, null);
                        il.Emit(OpCodes.Brfalse, lblNotNaN);

                        // Is NaN - value is already on stack, jump to end
                        il.Emit(OpCodes.Br, lblEnd);

                        il.MarkLabel(lblNotNaN);
                        // Not NaN - call MathF.Sign
                        il.EmitCall(OpCodes.Call, CachedMethods.MathFSign, null);
                        il.Emit(OpCodes.Conv_R4);

                        il.MarkLabel(lblEnd);
                    }
                    break;

                case NPTypeCode.Double:
                    {
                        // NumPy: sign(NaN) = NaN. .NET Math.Sign(NaN) throws.
                        // Check for NaN first: if (double.IsNaN(x)) return x; else return Math.Sign(x);
                        var lblNotNaN = il.DefineLabel();
                        var lblEnd = il.DefineLabel();

                        il.Emit(OpCodes.Dup);  // duplicate for NaN check
                        il.EmitCall(OpCodes.Call, CachedMethods.DoubleIsNaN, null);
                        il.Emit(OpCodes.Brfalse, lblNotNaN);

                        // Is NaN - value is already on stack, jump to end
                        il.Emit(OpCodes.Br, lblEnd);

                        il.MarkLabel(lblNotNaN);
                        // Not NaN - call Math.Sign
                        il.EmitCall(OpCodes.Call, CachedMethods.MathSignDouble, null);
                        il.Emit(OpCodes.Conv_R8);

                        il.MarkLabel(lblEnd);
                    }
                    break;

                case NPTypeCode.Decimal:
                    {
                        // Decimal has its own Sign method that returns int
                        il.EmitCall(OpCodes.Call, CachedMethods.MathSignDecimal, null);
                        // Convert int to decimal
                        il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                    }
                    break;

                case NPTypeCode.Boolean:
                    // sign(true) = 1, sign(false) = 0 - value is already correct (0 or 1)
                    break;

                case NPTypeCode.Byte:
                case NPTypeCode.UInt16:
                case NPTypeCode.UInt32:
                case NPTypeCode.UInt64:
                case NPTypeCode.Char:
                    // For unsigned: sign(x) = (x != 0) ? 1 : 0
                    // Stack: x
                    il.Emit(OpCodes.Ldc_I4_0);      // x, 0
                    il.Emit(OpCodes.Cgt_Un);        // (x > 0) as 0 or 1
                    // Convert back to original type
                    EmitConvertFromInt(il, type);
                    break;

                case NPTypeCode.Int16:
                    // sign(x) = (x >> 15) | ((int)(-x) >> 31 & 1)
                    // Simplified: (x > 0) - (x < 0)
                    // Stack: x
                    {
                        var locX = il.DeclareLocal(typeof(int));
                        il.Emit(OpCodes.Stloc, locX);   // save x

                        // (x > 0) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // x
                        il.Emit(OpCodes.Ldc_I4_0);      // x, 0
                        il.Emit(OpCodes.Cgt);           // (x > 0) as 0 or 1

                        // (x < 0) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // (x>0), x
                        il.Emit(OpCodes.Ldc_I4_0);      // (x>0), x, 0
                        il.Emit(OpCodes.Clt);           // (x>0), (x<0)

                        // result = (x > 0) - (x < 0)
                        il.Emit(OpCodes.Sub);           // (x>0) - (x<0) = -1, 0, or 1
                        il.Emit(OpCodes.Conv_I2);       // Convert to short
                    }
                    break;

                case NPTypeCode.Int32:
                    // sign(x) = (x > 0) - (x < 0)
                    // Stack: x
                    {
                        var locX = il.DeclareLocal(typeof(int));
                        il.Emit(OpCodes.Stloc, locX);   // save x

                        // (x > 0) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // x
                        il.Emit(OpCodes.Ldc_I4_0);      // x, 0
                        il.Emit(OpCodes.Cgt);           // (x > 0) as 0 or 1

                        // (x < 0) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // (x>0), x
                        il.Emit(OpCodes.Ldc_I4_0);      // (x>0), x, 0
                        il.Emit(OpCodes.Clt);           // (x>0), (x<0)

                        // result = (x > 0) - (x < 0)
                        il.Emit(OpCodes.Sub);           // (x>0) - (x<0) = -1, 0, or 1
                    }
                    break;

                case NPTypeCode.Int64:
                    // sign(x) = (x > 0) - (x < 0)
                    // Stack: x (as int64)
                    {
                        var locX = il.DeclareLocal(typeof(long));
                        il.Emit(OpCodes.Stloc, locX);   // save x

                        // (x > 0L) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // x
                        il.Emit(OpCodes.Ldc_I8, 0L);    // x, 0L
                        il.Emit(OpCodes.Cgt);           // (x > 0) as 0 or 1

                        // (x < 0L) ? 1 : 0
                        il.Emit(OpCodes.Ldloc, locX);   // (x>0), x
                        il.Emit(OpCodes.Ldc_I8, 0L);    // (x>0), x, 0L
                        il.Emit(OpCodes.Clt);           // (x>0), (x<0)

                        // result = (x > 0) - (x < 0), then convert to long
                        il.Emit(OpCodes.Sub);           // (x>0) - (x<0) = -1, 0, or 1 (as int32)
                        il.Emit(OpCodes.Conv_I8);       // Convert to long
                    }
                    break;

                default:
                    throw new NotSupportedException($"Sign not supported for type {type}");
            }
        }

        /// <summary>
        /// Convert int on stack to target type.
        /// </summary>
        private static void EmitConvertFromInt(ILGenerator il, NPTypeCode to)
        {
            switch (to)
            {
                case NPTypeCode.Boolean:
                    il.Emit(OpCodes.Ldc_I4_0);
                    il.Emit(OpCodes.Cgt_Un);
                    break;
                case NPTypeCode.Byte:
                    il.Emit(OpCodes.Conv_U1);
                    break;
                case NPTypeCode.Int16:
                    il.Emit(OpCodes.Conv_I2);
                    break;
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                    il.Emit(OpCodes.Conv_U2);
                    break;
                case NPTypeCode.Int32:
                    // Already int, no conversion needed
                    break;
                case NPTypeCode.UInt32:
                    il.Emit(OpCodes.Conv_U4);
                    break;
                case NPTypeCode.Int64:
                    il.Emit(OpCodes.Conv_I8);
                    break;
                case NPTypeCode.UInt64:
                    il.Emit(OpCodes.Conv_U8);
                    break;
                case NPTypeCode.Single:
                    il.Emit(OpCodes.Conv_R4);
                    break;
                case NPTypeCode.Double:
                    il.Emit(OpCodes.Conv_R8);
                    break;
                default:
                    throw new NotSupportedException($"Conversion from int to {to} not supported");
            }
        }

        /// <summary>
        /// Emit reciprocal (1/x) calculation.
        /// For float/double: divide 1 by value.
        /// For integer types: convert to double, compute reciprocal, convert back (result is 0 for |x| > 1).
        /// </summary>
        private static void EmitReciprocalCall(ILGenerator il, NPTypeCode type)
        {
            if (type == NPTypeCode.Single)
            {
                // 1.0f / x
                var locX = il.DeclareLocal(typeof(float));
                il.Emit(OpCodes.Stloc, locX);
                il.Emit(OpCodes.Ldc_R4, 1.0f);
                il.Emit(OpCodes.Ldloc, locX);
                il.Emit(OpCodes.Div);
            }
            else if (type == NPTypeCode.Double)
            {
                // 1.0 / x
                var locX = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locX);
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Ldloc, locX);
                il.Emit(OpCodes.Div);
            }
            else
            {
                // For integer types: convert to double, compute 1/x, convert back
                // Note: This will give 0 for any |x| > 1 (integer truncation)
                EmitConvertToDouble(il, type);
                var locX = il.DeclareLocal(typeof(double));
                il.Emit(OpCodes.Stloc, locX);
                il.Emit(OpCodes.Ldc_R8, 1.0);
                il.Emit(OpCodes.Ldloc, locX);
                il.Emit(OpCodes.Div);
                EmitConvertFromDouble(il, type);
            }
        }

        /// <summary>
        /// Emit degrees to radians conversion: x * (π/180).
        /// </summary>
        private static void EmitDeg2RadCall(ILGenerator il, NPTypeCode type)
        {
            const double Deg2RadFactor = Math.PI / 180.0;
            const float Deg2RadFactorF = (float)(Math.PI / 180.0);

            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, Deg2RadFactorF);
                il.Emit(OpCodes.Mul);
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, Deg2RadFactor);
                il.Emit(OpCodes.Mul);
            }
            else
            {
                // For integer types: convert to double, multiply, convert back
                EmitConvertToDouble(il, type);
                il.Emit(OpCodes.Ldc_R8, Deg2RadFactor);
                il.Emit(OpCodes.Mul);
                EmitConvertFromDouble(il, type);
            }
        }

        /// <summary>
        /// Emit radians to degrees conversion: x * (180/π).
        /// </summary>
        private static void EmitRad2DegCall(ILGenerator il, NPTypeCode type)
        {
            const double Rad2DegFactor = 180.0 / Math.PI;
            const float Rad2DegFactorF = (float)(180.0 / Math.PI);

            if (type == NPTypeCode.Single)
            {
                il.Emit(OpCodes.Ldc_R4, Rad2DegFactorF);
                il.Emit(OpCodes.Mul);
            }
            else if (type == NPTypeCode.Double)
            {
                il.Emit(OpCodes.Ldc_R8, Rad2DegFactor);
                il.Emit(OpCodes.Mul);
            }
            else
            {
                // For integer types: convert to double, multiply, convert back
                EmitConvertToDouble(il, type);
                il.Emit(OpCodes.Ldc_R8, Rad2DegFactor);
                il.Emit(OpCodes.Mul);
                EmitConvertFromDouble(il, type);
            }
        }

        #endregion
    }
}
