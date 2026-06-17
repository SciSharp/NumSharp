using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise power with array exponents: x1 ** x2
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, Type dtype)
            => Power(lhs, rhs, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise power: <c>x1 ** x2</c>, NumPy-aligned.
        ///
        /// Promotion / dispatch (NEP50, matches numpy 2.x):
        ///   - int^int        → integer result, dtype-native wrap (e.g. uint8 ** 8 = 0).
        ///                      Negative exponent rejected: NumPy raises ValueError unconditionally
        ///                      ("Integers to negative integer powers are not allowed.").
        ///   - int^float      → float64 (mirrors NumPy's int-base / np-float-exp rule).
        ///   - float^np.int   → float64 (NEP50 strict promotion: float32 ** np.int32 → float64).
        ///   - float^float    → wider float; float32 ** float32 → float32.
        ///   - complex^*      → complex128 (via Complex.Pow).
        ///   - decimal^*      → decimal (via DecimalMath.Pow).
        ///
        /// Stride/layout: routes through <c>ExecuteBinaryOp</c>, which handles
        /// contiguous + sliced + broadcast + F-contig via the IL kernel's
        /// SimdFull/SimdScalarRight/SimdScalarLeft/SimdChunk/General paths.
        /// The integer kernel calls <see cref="Utilities.NpyIntegerPower"/> for
        /// exact dtype wrapping (replaces the previous double round-trip).
        /// </summary>
        public override NDArray Power(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy rule: signed integer exponents cannot be negative when the LOOP is
            // integer**integer. The check is on the exponent, regardless of base value
            // (NumPy throws even for base=1 or base=-1 where the answer would be exact).
            // An explicit dtype= selects the loop, so power(2, -1, dtype=f64) = 0.5 is
            // legal while power(2, -1, dtype=i64) still raises (probed 2.4.2).
            bool loopIsInteger = typeCode.HasValue
                ? typeCode.Value.IsInteger()
                : lhs.GetTypeCode.IsInteger() && rhs.GetTypeCode.IsInteger();
            if (loopIsInteger && lhs.GetTypeCode.IsInteger() && rhs.GetTypeCode.IsInteger() && IsSignedInteger(rhs.GetTypeCode))
            {
                if (ContainsNegative(rhs))
                    throw new ArgumentException("Integers to negative integer powers are not allowed.");
            }

            // ufunc out=/where=: skip the scalar-exponent fast paths (they return
            // fresh arrays) and route through the iterator with the provided out.
            // dtype= composes with out — the loop COMPUTES in dtype and the value
            // is same_kind-cast into out (probed 2.4.2: power(10,8,out=f64,
            // dtype=f32) stores the float32-rounded value; the out-cast error
            // reports the dtype-overridden loop, not the promoted one).
            if (@out is not null || where is not null)
                return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power, @out, where, typeCode);

            // ufunc dtype= without out: the loop runs IN the requested dtype
            // (NumPy resolves the loop signature from dtype= — power(10,11,
            // dtype=f64) = 1e11 exactly, NOT int32-wrap-then-cast). The
            // scalar-exponent fast paths below substitute whole other ufunc
            // loops (sqrt/multiply/reciprocal), which resolve differently
            // under a dtype request — skip them and let the iterator compute
            // at the requested precision.
            if (typeCode.HasValue)
                return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power, null, null, typeCode);

            // Scalar-exponent fast paths (mirror NumPy's loops.c.src constant-time bodies):
            //   - exp = 0 → ones_like(lhs) in result dtype
            //   - exp = 1 → lhs (cast to result dtype if needed)
            //   - exp = 2 → lhs * lhs (uses SIMD Multiply kernel)
            //   - exp = 0.5 (float)  → sqrt(lhs)
            //   - exp = -1.0 (float) → reciprocal(lhs)
            // Only triggered when:
            //   - rhs has size == 1 (scalar or 1-element array), AND
            //   - the trivially-substituted op produces the same result dtype the
            //     general Power path would.
            if (rhs.size == 1)
            {
                var fast = TryScalarExponentFastPath(lhs, rhs);
                if (fast is not null)
                    return fast;
            }

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power);
        }

        /// <summary>
        /// Try the scalar-exponent fast paths. Returns null if the exponent value isn't
        /// in {0, 1, 2, 0.5, -1.0} or if the fast substitution would produce a different
        /// dtype than the regular Power path.
        /// </summary>
        private NDArray TryScalarExponentFastPath(NDArray lhs, NDArray rhs)
        {
            var rhsTc = rhs.GetTypeCode;
            var lhsTc = lhs.GetTypeCode;

            // exp = 0 — works for integer or float exponent
            if (IsScalarValueZero(rhs))
            {
                // ones_like preserves lhs dtype. The general Power path would promote
                // by _FindCommonType(lhs, rhs); if that promotion differs from lhs's
                // dtype the fast path would be wrong — bail out and use the slow path.
                var resultType = ResolvePowerResultType(lhs, rhs);
                if (resultType == lhsTc)
                    return np.ones_like(lhs);
                return null;
            }

            // exp = 1 — works for integer or float exponent. Returns lhs (in result dtype).
            if (IsScalarValueOne(rhs))
            {
                var resultType = ResolvePowerResultType(lhs, rhs);
                return resultType == lhsTc ? lhs.copy() : Cast(lhs, resultType, copy: true);
            }

            // exp = 2 — multiplication path (SIMD-optimized). Returns lhs * lhs.
            // The general Power path promotes via _FindCommonType; we route through
            // Multiply(lhs, lhs) so the result dtype is lhs's own dtype. For mixed-dtype
            // Power (e.g. f32_arr ** int32_scalar where NEP50 says result is f64),
            // bail out unless the resolved result type equals lhs's type.
            if (IsScalarValueTwo(rhs))
            {
                var resultType = ResolvePowerResultType(lhs, rhs);
                if (resultType == lhsTc)
                    return Multiply(lhs, lhs);
                return null;
            }

            // Float-only fast paths: exp = 0.5 (sqrt) and exp = -1.0 (reciprocal).
            // Only fire when rhs is a float dtype with the exact constant value.
            if (rhsTc == NPTypeCode.Single || rhsTc == NPTypeCode.Double || rhsTc == NPTypeCode.Half)
            {
                double v = ReadScalarAsDouble(rhs);
                if (v == 0.5)
                {
                    // np.sqrt promotes int -> float64 the same way as power would,
                    // and preserves float dtypes (f32 -> f32, f64 -> f64).
                    return np.sqrt(lhs);
                }
                if (v == -1.0 && lhsTc.IsFloatingPoint())
                {
                    // np.reciprocal preserves float dtype. For integer base we'd need to
                    // promote, which the general path handles — let it through.
                    return np.reciprocal(lhs);
                }
            }

            return null;
        }

        /// <summary>
        /// Read a size-1 NDArray's scalar value as double, regardless of dtype or rank.
        /// </summary>
        private static double ReadScalarAsDouble(NDArray nd)
        {
            object v = nd.GetAtIndex(0);
            // System.Half does not implement IConvertible, so Convert.ToDouble(object) throws for it
            // (the scalar-exponent fast paths support Half — see ScalarEqualsExact). Cast it directly.
            if (v is Half h)
                return (double)h;
            return Convert.ToDouble(v, System.Globalization.CultureInfo.InvariantCulture);
        }

        private static NPTypeCode ResolvePowerResultType(NDArray lhs, NDArray rhs)
        {
            // Use the NDArray overload so the weak/strict scalar rule (size-1 0-D as weak)
            // matches what ExecuteBinaryOp uses; otherwise the fast path would bail out
            // whenever NumSharp's promotion preserves the float dtype against a 0-D int.
            var resultType = np._FindCommonType(lhs, rhs);
            if (lhs.GetTypeCode.GetGroup() <= 2 && rhs.GetTypeCode.GetGroup() == 3)
                resultType = NPTypeCode.Double;
            return resultType;
        }

        private static bool IsScalarValueZero(NDArray rhs) => ScalarEqualsExact(rhs, 0.0);
        private static bool IsScalarValueOne(NDArray rhs) => ScalarEqualsExact(rhs, 1.0);
        private static bool IsScalarValueTwo(NDArray rhs) => ScalarEqualsExact(rhs, 2.0);

        /// <summary>
        /// Compare a size-1 NDArray's value against an exact double constant. Skips Complex
        /// (the fast paths don't apply) and returns false for non-numeric or unknown dtypes.
        /// </summary>
        private static bool ScalarEqualsExact(NDArray rhs, double target)
        {
            switch (rhs.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Byte:
                case NPTypeCode.SByte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Char:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                    return ReadScalarAsDouble(rhs) == target;
                default:
                    return false;
            }
        }

        private static bool IsSignedInteger(NPTypeCode tc)
            => tc == NPTypeCode.SByte
            || tc == NPTypeCode.Int16
            || tc == NPTypeCode.Int32
            || tc == NPTypeCode.Int64;

        /// <summary>
        /// Stride/broadcast-aware scan for any negative element in a signed integer array.
        /// Mirrors numpy's per-element check in <c>@TYPE@_power</c> but hoisted to a single
        /// pre-pass so the inner kernel stays branch-free.
        /// </summary>
        private static bool ContainsNegative(NDArray nd)
        {
            switch (nd.GetTypeCode)
            {
                case NPTypeCode.SByte: return AnyNegativeSByte(nd);
                case NPTypeCode.Int16: return AnyNegativeInt16(nd);
                case NPTypeCode.Int32: return AnyNegativeInt32(nd);
                case NPTypeCode.Int64: return AnyNegativeInt64(nd);
                default: return false;
            }
        }

        private static bool AnyNegativeSByte(NDArray nd)
        {
            long n = nd.size;
            for (long i = 0; i < n; i++)
                if (nd.GetSByte(i) < 0) return true;
            return false;
        }

        private static bool AnyNegativeInt16(NDArray nd)
        {
            long n = nd.size;
            for (long i = 0; i < n; i++)
                if (nd.GetInt16(i) < 0) return true;
            return false;
        }

        private static bool AnyNegativeInt32(NDArray nd)
        {
            long n = nd.size;
            for (long i = 0; i < n; i++)
                if (nd.GetInt32(i) < 0) return true;
            return false;
        }

        private static bool AnyNegativeInt64(NDArray nd)
        {
            long n = nd.size;
            for (long i = 0; i < n; i++)
                if (nd.GetInt64(i) < 0) return true;
            return false;
        }
    }
}
