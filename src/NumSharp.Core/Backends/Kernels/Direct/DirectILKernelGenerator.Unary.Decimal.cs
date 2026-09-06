using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// DirectILKernelGenerator.Unary.Decimal.cs - Decimal IL Emission
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
    public static partial class DirectILKernelGenerator
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
                case UnaryOp.Asinh:
                case UnaryOp.Acosh:
                case UnaryOp.Atanh:
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
                        UnaryOp.Asinh => "Asinh",
                        UnaryOp.Acosh => "Acosh",
                        UnaryOp.Atanh => "Atanh",
                        UnaryOp.Log2 => "Log2",
                        UnaryOp.Log10 => "Log10",
                        _ => throw new NotSupportedException()
                    };

                    il.EmitCall(OpCodes.Call, ScalarMethodCache.MathFn1(typeof(double), mathMethod), null);
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
                case UnaryOp.IsPosInf:
                case UnaryOp.IsNegInf:
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

                case UnaryOp.Conjugate:
                    // np.conjugate(complex128): flip the sign of the imaginary part.
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexConjugate, null);
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
                    // NDComplexMath.Square = FMA-contracted z*z (matches NumPy's complex multiply
                    // overflow/cancellation, which Complex.op_Multiply does not).
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexSquare, null);
                    break;

                case UnaryOp.Reciprocal:
                    // NumPy nc_recip: conj(z)/|z|^2 (signed zeros + overflow match NumPy, unlike 1/z).
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexReciprocal, null);
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
                    // NDComplexMath.Exp2 = Exp(z*ln2). Routing through the C99-correct Exp reproduces
                    // NumPy's non-finite results (exp2(+-inf+0j), exp2(inf+inf j), ...) that
                    // Complex.Pow(2, z) turned into NaN+NaNj.
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexExp2, null);
                    break;

                case UnaryOp.Log1p:
                    // NDComplexMath.Log1p = Log((1+re, im)) — preserves a -0 imaginary part that the
                    // naive Complex.One + z would flip to +0 on the cut.
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexLog1p, null);
                    break;

                case UnaryOp.Expm1:
                    // NDComplexMath.Expm1 = nc_expm1 formula (real = expm1(x)*cos(y) - 2*sin^2(y/2),
                    // imag = exp(x)*sin(y)). The naive Complex.Exp(z)-1 loses NumPy's non-finite
                    // imaginary parts and origin signed zeros.
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexExpm1, null);
                    break;

                // Hyperbolic and inverse-trig: NDComplexMath wraps the BCL with C99 Annex G non-finite
                // tables and branch-cut/signed-zero fixups so the results match NumPy on every input.
                case UnaryOp.Sinh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexSinh, null);
                    break;

                case UnaryOp.Cosh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexCosh, null);
                    break;

                case UnaryOp.Tanh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexTanh, null);
                    break;

                case UnaryOp.ASin:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAsin, null);
                    break;

                case UnaryOp.ACos:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAcos, null);
                    break;

                case UnaryOp.ATan:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAtan, null);
                    break;

                case UnaryOp.Asinh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAsinh, null);
                    break;

                case UnaryOp.Acosh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAcosh, null);
                    break;

                case UnaryOp.Atanh:
                    il.EmitCall(OpCodes.Call, CachedMethods.ComplexAtanh, null);
                    break;

                // Note: UnaryOp.Cbrt is deliberately NOT handled for Complex — NumPy's np.cbrt raises
                // TypeError for complex inputs, so falling through to the default throw keeps parity.

                case UnaryOp.Round:
                    // NumPy rint(complex) / around(complex) rounds the real and imaginary parts
                    // SEPARATELY, half-to-even (Math.Round default). floor/ceil/trunc have no complex
                    // loop in NumPy (TypeError), so Round is the only rounding op handled here.
                    {
                        var locZ = il.DeclareLocal(typeof(System.Numerics.Complex));
                        il.Emit(OpCodes.Stloc, locZ);                                             // [] (z saved)
                        il.Emit(OpCodes.Ldloca, locZ);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetReal, null);            // [re]
                        il.EmitCall(OpCodes.Call, ScalarMethodCache.MathFn1(typeof(double), "Round"), null); // [rint(re)]
                        il.Emit(OpCodes.Ldloca, locZ);
                        il.EmitCall(OpCodes.Call, CachedMethods.ComplexGetImaginary, null);       // [rint(re), im]
                        il.EmitCall(OpCodes.Call, ScalarMethodCache.MathFn1(typeof(double), "Round"), null); // [rint(re), rint(im)]
                        il.Emit(OpCodes.Newobj, CachedMethods.ComplexCtor);                       // [Complex(rint(re), rint(im))]
                    }
                    break;

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
        /// <summary>
        /// NumPy-faithful float16 negation: flip the IEEE sign bit (<c>h ^ 0x8000</c>). The BCL
        /// <c>Half</c> unary <c>operator -</c> evaluates <c>(Half)(-(float)h)</c> — a float
        /// roundtrip measured 7.3× slower than this bit flip (and 7.3× slower than
        /// <see cref="Half.Abs"/>'s sign-bit mask, which is why f16 abs was fast and f16 negate
        /// was the worst cell in the elementwise matrix at ~0.14× NumPy). Bit-exact with NumPy's
        /// npy_half negate across normals, ±0, ±inf, and NaN (sign flips, payload preserved).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static Half NegateHalf(Half value)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(value);
            return BitConverter.UInt16BitsToHalf((ushort)(bits ^ 0x8000));
        }

        /// <summary>
        /// Emit a Half unary math op computed in <b>float32</b> — NumPy's exact npy_half model
        /// <c>(half) op_f((float)h)</c> (<c>loops_half.dispatch.c.src</c>: widen with
        /// <c>npy_half_to_float</c>, run the CRT float function, round back with
        /// <c>npy_float_to_half</c>). The Half↔float conversions are the <c>[Intrinsic]</c>
        /// <c>op_Explicit</c> pair the JIT lowers to hardware F16C (<c>vcvtph2ps</c>/<c>vcvtps2ph</c>),
        /// so the round-trip is nearly free. This replaces two slower per-element shapes the Half
        /// switch used before: the Half→<b>double</b> bridge (a 2-deep Half→float→double widen plus
        /// the slower <c>Math.*</c> double functions) and the BCL <c>Half.*</c> transcendentals (an
        /// un-inlined managed call per element). Consumes a <c>Half</c> on the stack, leaves a
        /// <c>Half</c>. Bit-parity: <c>MathF.*</c> is the same <c>ucrtbase</c> CRT NumPy's HALF loop
        /// calls (<c>npy_sinhf</c>=<c>MathF.Sinh</c>, …), so the narrowed result is byte-identical to
        /// NumPy 2.4.2 for the platform-libm ops (verified over all 65 536 f16 values).
        /// </summary>
        private static void EmitUnaryHalfViaFloat(ILGenerator il, UnaryOp op)
        {
            il.EmitCall(OpCodes.Call, CachedMethods.HalfToFloat, null);   // Half → float (vcvtph2ps)
            switch (op)
            {
                // Transcendentals: MathF.* == the CRT npy_*f that NumPy's HALF loop calls (same
                // ucrtbase). Deliberately NOT the Single SIMD kernels (SingleExp/Sin/Tanh/…): those
                // are NumPy's *float32-array* loops, which disagree with the *half* loop's scalar CRT.
                case UnaryOp.Sinh:  EmitMathCall(il, "Sinh",  NPTypeCode.Single); break;
                case UnaryOp.Cosh:  EmitMathCall(il, "Cosh",  NPTypeCode.Single); break;
                case UnaryOp.Tanh:  EmitMathCall(il, "Tanh",  NPTypeCode.Single); break;
                case UnaryOp.ASin:  EmitMathCall(il, "Asin",  NPTypeCode.Single); break;
                case UnaryOp.ACos:  EmitMathCall(il, "Acos",  NPTypeCode.Single); break;
                case UnaryOp.ATan:  EmitMathCall(il, "Atan",  NPTypeCode.Single); break;
                case UnaryOp.Asinh: EmitMathCall(il, "Asinh", NPTypeCode.Single); break;
                case UnaryOp.Acosh: EmitMathCall(il, "Acosh", NPTypeCode.Single); break;
                case UnaryOp.Atanh: EmitMathCall(il, "Atanh", NPTypeCode.Single); break;
                case UnaryOp.Tan:   EmitMathCall(il, "Tan",   NPTypeCode.Single); break;
                case UnaryOp.Cbrt:  EmitMathCall(il, "Cbrt",  NPTypeCode.Single); break;
                case UnaryOp.Log2:  EmitMathCall(il, "Log2",  NPTypeCode.Single); break;
                case UnaryOp.Log10: EmitMathCall(il, "Log10", NPTypeCode.Single); break;

                // exp2: the CRT float.Exp2 (== npy_exp2f). Byte-identical to NumPy's half loop over
                // ALL 65 536 f16 inputs, NaN payloads included. NOT NDFloatMath.Exp2 (the fast SIMD
                // float32-array kernel): its scalar entry is markedly slower here and blanks the NaN
                // payload. exp2's true win needs vectorized F16C (absent in this runtime); scalar, the
                // CRT is both correct and the fastest 2^x available.
                case UnaryOp.Exp2:  il.EmitCall(OpCodes.Call, CachedMethods.SingleExp2Crt, null); break;

                // Arithmetic — float32 IEEE, bit-identical to NumPy's half loop by construction.
                case UnaryOp.Reciprocal: EmitReciprocalCall(il, NPTypeCode.Single); break;
                case UnaryOp.Square:     il.Emit(OpCodes.Dup); il.Emit(OpCodes.Mul); break;

                default:
                    throw new NotSupportedException(
                        $"EmitUnaryHalfViaFloat: op {op} is not mapped to a float32 path");
            }
            il.EmitCall(OpCodes.Call, CachedMethods.FloatToHalf, null);   // float → Half (vcvtps2ph)
        }

        private static void EmitUnaryHalfOperation(ILGenerator il, UnaryOp op)
        {
            // float32 fast path (NumPy's npy_half model, F16C round-trip) for the transcendental /
            // arithmetic ops — see EmitUnaryHalfViaFloat. The remaining cases below stay on their
            // specialized bit-fiddle / BCL paths (Negate/Abs/Sqrt/Floor/Ceil/Truncate/Sign/predicates).
            switch (op)
            {
                case UnaryOp.Sinh:
                case UnaryOp.Cosh:
                case UnaryOp.Tanh:
                case UnaryOp.ASin:
                case UnaryOp.ACos:
                case UnaryOp.ATan:
                case UnaryOp.Asinh:
                case UnaryOp.Acosh:
                case UnaryOp.Atanh:
                case UnaryOp.Tan:
                case UnaryOp.Cbrt:
                case UnaryOp.Log2:
                case UnaryOp.Log10:
                case UnaryOp.Exp2:
                case UnaryOp.Reciprocal:
                case UnaryOp.Square:
                    EmitUnaryHalfViaFloat(il, op);
                    return;
                // Expm1 / Log1p deliberately stay on the double bridge below: .NET's float.ExpM1 /
                // float.LogP1 lose small-|x| precision (float.ExpM1(2^-24) → 0x0002 where NumPy's
                // npy_expm1f gives 0x0001; log1p is worse — 17 409 finite half diffs), while
                // double.ExpM1 / double.LogP1 are byte-identical to NumPy's half loop.
            }

            switch (op)
            {
                case UnaryOp.Negate:
                    // NumPy sign-bit flip, NOT Half.op_UnaryNegation's float roundtrip (7.3× slower).
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

                case UnaryOp.Exp:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfExp, null);
                    break;

                case UnaryOp.Log:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfLog, null);
                    break;

                case UnaryOp.Round:
                case UnaryOp.Deg2Rad:
                case UnaryOp.Rad2Deg:
                    // The three ops NOT covered by the float32 fast path above (Round is bit-preserving,
                    // Deg2Rad/Rad2Deg fold a constant). Half → double → Math.X → Half roundtrip
                    // (house Half policy; matches NumPy's promote-compute-round model for npy_half).
                    // sinh/cosh/tanh/asin/acos/atan used to share this arm but now take the float32
                    // path (EmitUnaryHalfViaFloat) — bit-identical to NumPy's half loop and faster.
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                    EmitUnaryScalarOperation(il, op, NPTypeCode.Double);
                    il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    break;

                case UnaryOp.Log1p:
                    // B21: Half.LogP1(x) computes (1 + x) in Half precision, which rounds
                    // subnormal x to 0 because Half epsilon ≫ 2^-24. Promote to double (NumPy's
                    // own model: float32 isn't enough either — float32 epsilon near 1 is ~2^-23,
                    // already coarser than Half's smallest subnormal 2^-24).
                    //
                    // Sign-of-zero: .NET's double.LogP1(-0.0) returns +0.0, dropping the sign.
                    // NumPy preserves sign through log1p. Wrap the result in CopySign(result, x)
                    // to restore the input's sign. This happens to be correct over log1p's
                    // entire domain because log1p(x) always has the same sign as x when
                    // x ∈ (-1, ∞).
                    {
                        var locIn = il.DeclareLocal(typeof(double));
                        il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                        il.Emit(OpCodes.Stloc, locIn);
                        il.Emit(OpCodes.Ldloc, locIn);
                        il.EmitCall(OpCodes.Call, CachedMethods.DoubleLogP1, null);
                        il.Emit(OpCodes.Ldloc, locIn);
                        il.EmitCall(OpCodes.Call, CachedMethods.MathCopySign, null);
                        il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    }
                    break;

                case UnaryOp.Expm1:
                    // B21: Half.ExpM1(x) suffers the same subnormal-precision loss as LogP1
                    // (internal exp(x)-1 step loses bits). Promote through double. Same
                    // CopySign sign-of-zero correction as Log1p — expm1(x) has the same sign
                    // as x over its entire domain.
                    {
                        var locIn = il.DeclareLocal(typeof(double));
                        il.EmitCall(OpCodes.Call, CachedMethods.HalfToDouble, null);
                        il.Emit(OpCodes.Stloc, locIn);
                        il.Emit(OpCodes.Ldloc, locIn);
                        il.EmitCall(OpCodes.Call, CachedMethods.DoubleExpM1, null);
                        il.Emit(OpCodes.Ldloc, locIn);
                        il.EmitCall(OpCodes.Call, CachedMethods.MathCopySign, null);
                        il.EmitCall(OpCodes.Call, CachedMethods.DoubleToHalf, null);
                    }
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

                case UnaryOp.Sign:
                    // Half Sign with NaN handling: if NaN, return NaN; else return sign
                    // NumPy: sign(NaN) = NaN, sign(0) = 0, sign(+x) = 1, sign(-x) = -1
                    il.EmitCall(OpCodes.Call, GetHelper(nameof(HalfSignHelper)), null);
                    break;

                case UnaryOp.IsNan:
                    il.EmitCall(OpCodes.Call, CachedMethods.HalfIsNaN, null);
                    break;

                case UnaryOp.IsInf:
                    il.EmitCall(OpCodes.Call,
                        ScalarMethodCache.Predicate(typeof(Half), "IsInfinity"), null);
                    break;

                case UnaryOp.IsPosInf:
                    il.EmitCall(OpCodes.Call,
                        ScalarMethodCache.Predicate(typeof(Half), "IsPositiveInfinity"), null);
                    break;

                case UnaryOp.IsNegInf:
                    il.EmitCall(OpCodes.Call,
                        ScalarMethodCache.Predicate(typeof(Half), "IsNegativeInfinity"), null);
                    break;

                case UnaryOp.IsFinite:
                    il.EmitCall(OpCodes.Call,
                        ScalarMethodCache.Predicate(typeof(Half), "IsFinite"), null);
                    break;

                default:
                    throw new NotSupportedException($"Unary operation {op} not supported for Half");
            }
        }

        /// <summary>
        /// Helper for Half sign — the scalar (strided / tail) path. Works on the raw f16 bit pattern,
        /// no float round-trip: NumPy's HALF_sign is `x>0 ? 1 : x<0 ? -1 : x`, so a <b>NaN input is
        /// passed through unchanged</b> (its exact bits, sign and payload), a ±0 collapses to +0, and
        /// every other value maps to ±1. Matches NumPy 2.4.2 byte-for-byte over all 65 536 f16 inputs
        /// (the previous form returned the canonical <c>Half.NaN</c>, which diverged on every NaN
        /// input — an excused-but-real difference now closed) and is bit-identical to the SIMD
        /// <see cref="HalfSignContiguous"/> fast path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        internal static Half HalfSignHelper(Half value)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(value);
            ushort mag = (ushort)(bits & 0x7fff);
            if (mag > 0x7c00) return value;                     // NaN → input unchanged
            if (mag == 0) return Half.Zero;                     // ±0 → +0
            return (bits & 0x8000) != 0 ? (Half)(-1.0) : (Half)1.0;
        }

        /// <summary>
        /// Whole-array SIMD Half <c>sign</c> — the contiguous fast path (<see cref="UnaryKernel"/>
        /// signature). np.sign on float16 is defined entirely on the bit pattern, so — unlike every
        /// other Half math op — it needs <b>no</b> float32 round-trip and vectorizes directly over the
        /// raw <c>ushort</c> lanes: 16 elements per <c>Vector256&lt;ushort&gt;</c> iteration, a
        /// branchless <c>NaN → input</c> / <c>±0 → +0</c> / <c>±normal → ±1</c> select chain, scalar
        /// tail via <see cref="HalfSignHelper"/>. This is the one degraded f16 op that can actually
        /// beat NumPy (whose speed elsewhere comes from vectorized F16C conversions this runtime does
        /// not expose): here there is nothing to convert. Byte-identical to NumPy 2.4.2 over all
        /// 65 536 inputs. Falls back to the scalar helper when SIMD is unavailable.
        /// </summary>
        internal static unsafe void HalfSignContiguous(
            void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
        {
            ushort* pi = (ushort*)input;
            ushort* po = (ushort*)output;
            long n = totalSize;
            long i = 0;

            // Direct AVX2 (not the generic Vector256.* API): its unsigned ushort GreaterThan /
            // ConditionalSelect did NOT lower to single instructions here — the kernel measured ~30×
            // slower than these intrinsics. `mag` and 0x7c00 are both ≤ 0x7fff, so the signed
            // CompareGreaterThan is sign-safe; BlendVariable selects per byte on all-ones/all-zeros
            // lane masks. Same primitive style as the Cast.Half kernels.
            if (Avx2.IsSupported)
            {
                var m7f  = Vector256.Create((short)0x7fff);
                var m80  = Vector256.Create(unchecked((short)0x8000));
                var vinf = Vector256.Create((short)0x7c00);              // f16 +inf; mag > it ⇒ NaN
                var vp1  = Vector256.Create(unchecked((short)0x3c00));   // f16 +1
                var vn1  = Vector256.Create(unchecked((short)0xbc00));   // f16 -1
                var vz   = Vector256<short>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    var b      = Avx.LoadVector256((short*)(pi + i));
                    var mag    = Avx2.And(b, m7f);
                    var isNan  = Avx2.CompareGreaterThan(mag, vinf);     // mag,vinf ≤ 0x7fff: sign-safe
                    var isZero = Avx2.CompareEqual(mag, vz);
                    var neg    = Avx2.CompareEqual(Avx2.And(b, m80), m80);
                    var r = Avx2.BlendVariable(vp1.AsByte(), vn1.AsByte(), neg.AsByte());   // ±1
                    r = Avx2.BlendVariable(r, vz.AsByte(), isZero.AsByte());                // ±0 → +0
                    r = Avx2.BlendVariable(r, b.AsByte(), isNan.AsByte());                  // NaN → input
                    Avx.Store((short*)(po + i), r.AsInt16());
                }
            }

            for (; i < n; i++)
            {
                ushort bits = pi[i];
                ushort mag = (ushort)(bits & 0x7fff);
                po[i] = mag > 0x7c00 ? bits
                      : mag == 0 ? (ushort)0
                      : (bits & 0x8000) != 0 ? (ushort)0xbc00 : (ushort)0x3c00;
            }
        }

        /// <summary>
        /// Whole-array SIMD Half <c>negate</c> — <c>bits ^ 0x8000</c> over raw ushort lanes
        /// (16 per <c>Vector256</c> iteration). NumPy's npy_half negate is exactly the sign-bit flip
        /// (payload preserved on NaN, ±0/±inf flip), so this is bit-identical over all 65 536 f16
        /// values — with NO float round-trip, the reason it beats NumPy. The scalar <see cref="NegateHalf"/>
        /// bit-flip (used by the strided path) already produced the right value but ran one managed
        /// call per element — the f16 negate cell's ~0.06× floor. Contiguous fast path only.
        /// </summary>
        internal static unsafe void HalfNegateContiguous(
            void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
        {
            ushort* pi = (ushort*)input;
            ushort* po = (ushort*)output;
            long n = totalSize;
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m80 = Vector256.Create(unchecked((short)0x8000));
                for (; i + 16 <= n; i += 16)
                    Avx.Store((short*)(po + i), Avx2.Xor(Avx.LoadVector256((short*)(pi + i)), m80));
            }
            for (; i < n; i++) po[i] = (ushort)(pi[i] ^ 0x8000);
        }

        /// <summary>
        /// Whole-array SIMD Half <c>abs</c> — <c>bits &amp; 0x7fff</c> over raw ushort lanes. NumPy's
        /// npy_half abs clears the sign bit unconditionally (NaN payload preserved, sign cleared), so
        /// this is bit-identical over all 65 536 f16 values. Replaces the per-element BCL
        /// <see cref="Half.Abs"/> call (the scalar/strided path keeps it — also a sign-bit mask, same
        /// result). Contiguous fast path only.
        /// </summary>
        internal static unsafe void HalfAbsContiguous(
            void* input, void* output, long* strides, long* shape, int ndim, long totalSize)
        {
            ushort* pi = (ushort*)input;
            ushort* po = (ushort*)output;
            long n = totalSize;
            long i = 0;
            if (Avx2.IsSupported)
            {
                var m7f = Vector256.Create((short)0x7fff);
                for (; i + 16 <= n; i += 16)
                    Avx.Store((short*)(po + i), Avx2.And(Avx.LoadVector256((short*)(pi + i)), m7f));
            }
            for (; i < n; i++) po[i] = (ushort)(pi[i] & 0x7fff);
        }

        #endregion
    }
}
