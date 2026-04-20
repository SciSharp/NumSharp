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

        #region Unary Complex IL Emission

        /// <summary>
        /// Emit unary operation for Complex type.
        /// </summary>
        private static void EmitUnaryComplexOperation(ILGenerator il, UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Negate:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexNegate, null);
                    break;

                case UnaryOp.Sqrt:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexSqrt, null);
                    break;

                case UnaryOp.Exp:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexExp, null);
                    break;

                case UnaryOp.Log:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexLog, null);
                    break;

                case UnaryOp.Sin:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexSin, null);
                    break;

                case UnaryOp.Cos:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexCos, null);
                    break;

                case UnaryOp.Tan:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexTan, null);
                    break;

                case UnaryOp.Abs:
                    // Complex.Abs returns magnitude as double
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAbs, null);
                    // Convert double back to Complex (real part only)
                    il.Emit(OpCodes.Ldc_R8, 0.0);
                    il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);
                    break;

                case UnaryOp.Square:
                    // z * z
                    il.Emit(OpCodes.Dup);
                    il.EmitCall(OpCodes.Call, typeof(System.Numerics.Complex).GetMethod("op_Multiply",
                        BindingFlags.Public | BindingFlags.Static,
                        new[] { typeof(System.Numerics.Complex), typeof(System.Numerics.Complex) })!, null);
                    break;

                case UnaryOp.Reciprocal:
                    // 1 / z
                    {
                        var locZ = il.DeclareLocal(typeof(System.Numerics.Complex));
                        il.Emit(OpCodes.Stloc, locZ);
                        il.Emit(OpCodes.Ldc_R8, 1.0);
                        il.Emit(OpCodes.Ldc_R8, 0.0);
                        il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);
                        il.Emit(OpCodes.Ldloc, locZ);
                        il.EmitCall(OpCodes.Call, typeof(System.Numerics.Complex).GetMethod("op_Division",
                            BindingFlags.Public | BindingFlags.Static,
                            new[] { typeof(System.Numerics.Complex), typeof(System.Numerics.Complex) })!, null);
                    }
                    break;

                case UnaryOp.Sign:
                    // Complex Sign: returns unit vector z / |z|, or 0 if z = 0.
                    // NumPy: sign(1+2j) = (0.447+0.894j), sign(0+0j) = (0+0j).
                    // EmitSignCall already has inline IL for Complex at Unary.Math.cs — reuse.
                    EmitSignCall(il, NPTypeCode.Complex);
                    break;

                case UnaryOp.IsNan:
                    // Complex.IsNaN = double.IsNaN(z.Real) || double.IsNaN(z.Imaginary)
                    EmitComplexComponentPredicate(il, CachedMethods.DoubleIsNaN, combineWithAnd: false);
                    break;

                case UnaryOp.IsInf:
                    // Complex.IsInfinity = double.IsInfinity(z.Real) || double.IsInfinity(z.Imaginary)
                    EmitComplexComponentPredicate(il, CachedMethods.DoubleIsInfinity, combineWithAnd: false);
                    break;

                case UnaryOp.IsFinite:
                    // Complex.IsFinite = double.IsFinite(z.Real) && double.IsFinite(z.Imaginary)
                    EmitComplexComponentPredicate(il, CachedMethods.DoubleIsFinite, combineWithAnd: true);
                    break;

                case UnaryOp.Log10:
                    // Complex.Log10(z) — NumPy: np.log10(complex) returns complex (base-10 log, principal branch).
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexLog10, null);
                    break;

                case UnaryOp.Log2:
                    // Complex.Log(z, 2.0) yields NaN imaginary for z=0+0j because its component-wise
                    // division by the base loses sign info when |z|=0. Work around by computing
                    // Complex.Log(z) and scaling both components by 1/ln(2) manually. Pseudo-C#:
                    //   var logZ = Complex.Log(z);
                    //   return new Complex(logZ.Real * (1/ln2), logZ.Imaginary * (1/ln2));
                    {
                        var locLog = il.DeclareLocal(typeof(System.Numerics.Complex));
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexLog, null);      // [Complex logZ]
                        il.Emit(OpCodes.Stloc, locLog);

                        // newobj Complex(logZ.Real * k, logZ.Imaginary * k) — k = 1/ln(2)
                        il.Emit(OpCodes.Ldloca, locLog);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null);
                        il.Emit(OpCodes.Ldsfld, CachedMethods.LogE_Inv_Ln2Field);
                        il.Emit(OpCodes.Mul);
                        il.Emit(OpCodes.Ldloca, locLog);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
                        il.Emit(OpCodes.Ldsfld, CachedMethods.LogE_Inv_Ln2Field);
                        il.Emit(OpCodes.Mul);
                        il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);
                    }
                    break;

                case UnaryOp.Exp2:
                    // B22: Complex.Pow(Complex(2,0), z) returns NaN+NaNj for z = ±inf+0j because
                    // the internal exp(z·log(2)) computes (±inf)·0 = NaN in the imaginary
                    // dimension. NumPy: exp2(-inf+0j) = 0+0j, exp2(+inf+0j) = inf+0j. Both are
                    // satisfied by Math.Pow(2, r) for pure-real inputs. Pseudo-C#:
                    //   if (z.Imaginary == 0.0)
                    //       return new Complex(Math.Pow(2.0, z.Real), 0.0);
                    //   return Complex.Pow(new Complex(2.0, 0.0), z);
                    // Bne_Un also branches on NaN, so imag=NaN correctly falls through to
                    // Complex.Pow (which propagates NaN per NumPy: exp2(r+nanj) = nan+nanj).
                    {
                        var locZ = il.DeclareLocal(typeof(System.Numerics.Complex));
                        var lblImagNonZero = il.DefineLabel();
                        var lblEnd = il.DefineLabel();

                        il.Emit(OpCodes.Stloc, locZ);

                        // if (z.Imaginary != 0.0 || double.IsNaN(z.Imaginary)) goto general;
                        il.Emit(OpCodes.Ldloca, locZ);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
                        il.Emit(OpCodes.Ldc_R8, 0.0);
                        il.Emit(OpCodes.Bne_Un, lblImagNonZero);

                        // Pure-real: new Complex(Math.Pow(2.0, z.Real), 0.0)
                        il.Emit(OpCodes.Ldc_R8, 2.0);
                        il.Emit(OpCodes.Ldloca, locZ);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null);
                        il.EmitCall(OpCodes.Call, CachedMethods.MathPow, null);
                        il.Emit(OpCodes.Ldc_R8, 0.0);
                        il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);
                        il.Emit(OpCodes.Br, lblEnd);

                        // General: Complex.Pow(new Complex(2.0, 0.0), z)
                        il.MarkLabel(lblImagNonZero);
                        il.Emit(OpCodes.Ldc_R8, 2.0);
                        il.Emit(OpCodes.Ldc_R8, 0.0);
                        il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);
                        il.Emit(OpCodes.Ldloc, locZ);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexPow, null);

                        il.MarkLabel(lblEnd);
                    }
                    break;

                case UnaryOp.Log1p:
                    // Complex.Log(1 + z). NumPy principal branch: log1p(-1+0j) = -inf+0j (matches Complex.Log).
                    {
                        il.Emit(OpCodes.Ldsfld, CachedMethods.ComplexOne);
                        // Stack: z, 1.
                        // op_Addition takes (Complex, Complex), emit in a way that the order is z+1 = 1+z.
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexOpAddition, null);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexLog, null);
                    }
                    break;

                case UnaryOp.Expm1:
                    // Complex.Exp(z) - 1.
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexExp, null);
                    il.Emit(OpCodes.Ldsfld, CachedMethods.ComplexOne);
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexOpSubtraction, null);
                    break;

                // Note: UnaryOp.Cbrt is deliberately NOT handled for Complex — NumPy's np.cbrt raises
                // TypeError for complex inputs, so falling through to the default throw keeps parity.

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for Complex");
            }
        }

        /// <summary>
        /// Emit a component-wise predicate on a Complex value: <c>predicate(z.Real) OP predicate(z.Imaginary)</c>
        /// where OP is <c>and</c> (combineWithAnd=true, used for IsFinite) or <c>or</c>
        /// (combineWithAnd=false, used for IsNaN / IsInfinity).
        ///
        /// Stack contract: expects [Complex z] on top, leaves [bool] on top.
        /// </summary>
        private static void EmitComplexComponentPredicate(ILGenerator il, MethodInfo doublePredicate, bool combineWithAnd)
        {
            var locZ = il.DeclareLocal(typeof(System.Numerics.Complex));
            il.Emit(OpCodes.Stloc, locZ);

            // predicate(z.Real)
            il.Emit(OpCodes.Ldloca, locZ);
            il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null);
            il.EmitCall(OpCodes.Call, doublePredicate, null);

            // predicate(z.Imaginary)
            il.Emit(OpCodes.Ldloca, locZ);
            il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);
            il.EmitCall(OpCodes.Call, doublePredicate, null);

            il.Emit(combineWithAnd ? OpCodes.And : OpCodes.Or);
        }

        // Log-base-2 conversion constant: 1 / ln(2) = log2(e). Loaded via Ldsfld in the
        // inline IL for UnaryOp.Log2 (Complex branch). Kept at file scope (not inside
        // CachedMethods) because it's a runtime-computed double, not a reflection lookup.
        internal static readonly double LogE_Inv_Ln2 = 1.0 / System.Math.Log(2.0);

        #endregion

        #region Unary Half IL Emission

        /// <summary>
        /// Emit unary operation for Half type.
        /// </summary>
        private static void EmitUnaryHalfOperation(ILGenerator il, UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Negate:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfNegate, null);
                    break;

                case UnaryOp.Abs:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfAbs, null);
                    break;

                case UnaryOp.Sqrt:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfSqrt, null);
                    break;

                case UnaryOp.Sin:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfSin, null);
                    break;

                case UnaryOp.Cos:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfCos, null);
                    break;

                case UnaryOp.Tan:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfTan, null);
                    break;

                case UnaryOp.Exp:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfExp, null);
                    break;

                case UnaryOp.Log:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfLog, null);
                    break;

                case UnaryOp.Log10:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfLog10, null);
                    break;

                case UnaryOp.Log2:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfLog2, null);
                    break;

                case UnaryOp.Cbrt:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfCbrt, null);
                    break;

                case UnaryOp.Exp2:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfExp2, null);
                    break;

                case UnaryOp.Log1p:
                    // B21: Half.LogP1(x) computes (1 + x) in Half precision, which rounds
                    // subnormal x to 0 because Half epsilon ≫ 2^-24. Promote to double (NumPy's
                    // own model: float32 isn't enough either — float32 epsilon near 1 is ~2^-23,
                    // already coarser than Half's smallest subnormal 2^-24).
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleLogP1, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    break;

                case UnaryOp.Expm1:
                    // B21: Half.ExpM1(x) suffers the same subnormal-precision loss as LogP1
                    // (internal exp(x)-1 step loses bits). Promote through double.
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleExpM1, null);
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    break;

                case UnaryOp.Floor:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfFloor, null);
                    break;

                case UnaryOp.Ceil:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfCeiling, null);
                    break;

                case UnaryOp.Truncate:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfTruncate, null);
                    break;

                case UnaryOp.Square:
                    // x * x
                    il.Emit(OpCodes.Dup);
                    il.EmitCall(OpCodes.Call, typeof(Half).GetMethod("op_Multiply",
                        BindingFlags.Public | BindingFlags.Static,
                        new[] { typeof(Half), typeof(Half) })!, null);
                    break;

                case UnaryOp.Reciprocal:
                    // 1 / x - convert via double
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                    {
                        var locX = il.DeclareLocal(typeof(double));
                        il.Emit(OpCodes.Stloc, locX);
                        il.Emit(OpCodes.Ldc_R8, 1.0);
                        il.Emit(OpCodes.Ldloc, locX);
                        il.Emit(OpCodes.Div);
                    }
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    break;

                case UnaryOp.Sign:
                    // Half Sign with NaN handling: if NaN, return NaN; else return sign
                    // NumPy: sign(NaN) = NaN, sign(0) = 0, sign(+x) = 1, sign(-x) = -1
                    // Use helper method to handle NaN properly
                    il.EmitCall(OpCodes.Call, typeof(ILKernelGenerator).GetMethod(nameof(HalfSignHelper),
                        BindingFlags.NonPublic | BindingFlags.Static)!, null);
                    break;

                case UnaryOp.IsNan:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfIsNaN, null);
                    break;

                case UnaryOp.IsInf:
                    il.EmitCall(OpCodes.Call, typeof(Half).GetMethod("IsInfinity",
                        BindingFlags.Public | BindingFlags.Static, new[] { typeof(Half) })!, null);
                    break;

                case UnaryOp.IsFinite:
                    il.EmitCall(OpCodes.Call, typeof(Half).GetMethod("IsFinite",
                        BindingFlags.Public | BindingFlags.Static, new[] { typeof(Half) })!, null);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for Half");
            }
        }

        /// <summary>
        /// Helper for Half sign: handles NaN properly (returns NaN).
        /// NumPy: sign(NaN) = NaN, sign(0) = 0, sign(+x) = 1, sign(-x) = -1
        /// </summary>
        internal static Half HalfSignHelper(Half value)
        {
            if (Half.IsNaN(value))
                return Half.NaN;
            if (value == Half.Zero)
                return Half.Zero;
            return value > Half.Zero ? (Half)1.0 : (Half)(-1.0);
        }

        #endregion
    }
}
