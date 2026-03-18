using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Unary.Decimal.cs - Decimal IL Emission
// =============================================================================
//
// RESPONSIBILITY:
//   - EmitUnaryDecimalOperation - all decimal unary operations
//   - Negate, Abs, Sign, Ceiling, Floor, Round, Truncate
//   - Sqrt, trig functions via double conversion
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Unary Decimal IL Emission
        /// <summary>
        /// Emit unary operation for decimal type.
        /// </summary>
        private static void EmitUnaryDecimalOperation(ILGenerator il, UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Negate:
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpUnaryNegation, null);
                    break;

                case UnaryOp.Abs:
                    il.EmitCall(OpCodes.Call, CachedMethods.MathAbsDecimal, null);
                    break;

                case UnaryOp.Sign:
                    // Math.Sign(decimal) returns int, convert back to decimal
                    il.EmitCall(OpCodes.Call, CachedMethods.MathSignDecimal, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                    break;

                case UnaryOp.Ceil:
                    // Math.Ceiling has decimal overload
                    il.EmitCall(OpCodes.Call, CachedMethods.MathCeilingDecimal, null);
                    break;

                case UnaryOp.Floor:
                    // Math.Floor has decimal overload
                    il.EmitCall(OpCodes.Call, CachedMethods.MathFloorDecimal, null);
                    break;

                case UnaryOp.Round:
                    // Math.Round has decimal overload
                    il.EmitCall(OpCodes.Call, CachedMethods.MathRoundDecimal, null);
                    break;

                case UnaryOp.Sqrt:
                case UnaryOp.Exp:
                case UnaryOp.Log:
                case UnaryOp.Sin:
                case UnaryOp.Cos:
                case UnaryOp.Tan:
                case UnaryOp.Sinh:
                case UnaryOp.Cosh:
                case UnaryOp.Tanh:
                case UnaryOp.ASin:
                case UnaryOp.ACos:
                case UnaryOp.ATan:
                case UnaryOp.Log2:
                case UnaryOp.Log10:
                    // Convert to double, perform operation, convert back
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);

                    string mathMethod = op switch
                    {
                        UnaryOp.Sqrt => "Sqrt",
                        UnaryOp.Exp => "Exp",
                        UnaryOp.Log => "Log",
                        UnaryOp.Sin => "Sin",
                        UnaryOp.Cos => "Cos",
                        UnaryOp.Tan => "Tan",
                        UnaryOp.Sinh => "Sinh",
                        UnaryOp.Cosh => "Cosh",
                        UnaryOp.Tanh => "Tanh",
                        UnaryOp.ASin => "Asin",
                        UnaryOp.ACos => "Acos",
                        UnaryOp.ATan => "Atan",
                        UnaryOp.Log2 => "Log2",
                        UnaryOp.Log10 => "Log10",
                        _ => throw new NotSupportedException()
                    };

                    il.EmitCall(OpCodes.Call,
                        typeof(Math).GetMethod(mathMethod, new[] { typeof(double) })
                            ?? throw new MissingMethodException(typeof(Math).FullName, mathMethod),
                        null);

                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.Exp2:
                    // 2^x for decimal: convert to double, use Math.Pow, convert back
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    // Stack: [exponent (double)] - need to call Pow(2, exponent)
                    var locExpDec = il.DeclareLocal(typeof(double));
                    il.Emit(OpCodes.Stloc, locExpDec);
                    il.Emit(OpCodes.Ldc_R8, 2.0);
                    il.Emit(OpCodes.Ldloc, locExpDec);
                    il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.Expm1:
                    // exp(x) - 1 for decimal
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.MathExp, null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Sub);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.Log1p:
                    // log(1 + x) for decimal
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    il.Emit(OpCodes.Ldc_R8, 1.0);
                    il.Emit(OpCodes.Add);
                    il.EmitCall(OpCodes.Call, CachedMethods.MathLog, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.Truncate:
                    // decimal.Truncate has direct support
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalTruncate, null);
                    break;

                case UnaryOp.Reciprocal:
                    // 1 / x for decimal
                    {
                        var locX = il.DeclareLocal(typeof(decimal));
                        il.Emit(OpCodes.Stloc, locX);
                        il.Emit(OpCodes.Ldc_I4_1);
                        il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                        il.Emit(OpCodes.Ldloc, locX);
                        il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpDivision, null);
                    }
                    break;

                case UnaryOp.Square:
                    // x * x for decimal
                    il.Emit(OpCodes.Dup);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpMultiply, null);
                    break;

                case UnaryOp.Deg2Rad:
                    // x * (π/180) for decimal - convert to double, multiply, convert back
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    il.Emit(OpCodes.Ldc_R8, Math.PI / 180.0);
                    il.Emit(OpCodes.Mul);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.Rad2Deg:
                    // x * (180/π) for decimal - convert to double, multiply, convert back
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    il.Emit(OpCodes.Ldc_R8, 180.0 / Math.PI);
                    il.Emit(OpCodes.Mul);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.BitwiseNot:
                    // Bitwise not doesn't make sense for decimal - throw
                    throw new NotSupportedException("BitwiseNot is not supported for decimal type");

                case UnaryOp.LogicalNot:
                    // Logical NOT for decimal: x == 0
                    // Compare to decimal.Zero and return bool
                    il.Emit(OpCodes.Ldsfld, CachedMethods.DecimalZero);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalOpEquality, null);
                    // Result is bool (int32 0 or 1), convert to decimal
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalImplicitFromInt, null);
                    break;

                case UnaryOp.Cbrt:
                    // Cube root for decimal - convert to double, call Math.Cbrt, convert back
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalToDouble, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.MathCbrt, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DecimalExplicitFromDouble, null);
                    break;

                case UnaryOp.IsFinite:
                    // Decimal is always finite - pop value, push true
                    il.Emit(OpCodes.Pop);
                    il.Emit(OpCodes.Ldc_I4_1);
                    break;

                case UnaryOp.IsNan:
                case UnaryOp.IsInf:
                    // Decimal cannot be NaN or Inf - pop value, push false
                    il.Emit(OpCodes.Pop);
                    il.Emit(OpCodes.Ldc_I4_0);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for decimal");
            }
        }

        #endregion
    }
}
