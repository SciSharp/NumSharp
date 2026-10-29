using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Returns a boolean array where two arrays are element-wise equal within a
        /// tolerance.
        /// The tolerance values are positive, typically very small numbers.The
        /// relative difference (`rtol` * abs(`b`)) and the absolute difference
        /// `atol` are added together to compare against the absolute difference
        /// between `a` and `b`.
        /// Warning: The default `atol` is not appropriate for comparing numbers
        /// that are much smaller than one(see Notes).
        ///
        /// See also <seealso cref="allclose"/>
        ///
        ///Notes:
        /// For finite values, isclose uses the following equation to test whether
        /// two floating point values are equivalent.
        /// <code>absolute(`a` - `b`) less than or equal to (`atol` + `rtol` * absolute(`b`))</code>
        /// Unlike the built-in `math.isclose`, the above equation is not symmetric
        /// in `a` and `b` -- it assumes `b` is the reference value -- so that
        /// `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
        /// the default value of atol is not zero, and is used to determine what
        /// small values should be considered close to zero.The default value is
        /// appropriate for expected values of order unity: if the expected values
        /// are significantly smaller than one, it can result in false positives.
        /// `atol` should be carefully selected for the use case at hand. A zero value
        /// for `atol` will result in `False` if either `a` or `b` is zero.
        /// </summary>
        /// <param name="a">Input array to compare with b</param>
        /// <param name="b">Input array to compare with a.</param>
        /// <param name="rtol">The relative tolerance parameter(see Notes)</param>
        /// <param name="atol">The absolute tolerance parameter(see Notes)</param>
        /// <param name="equal_nan">Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        ///considered equal to NaN's in `b` in the output array.</param>
        ///<returns>
        ///  Returns a boolean array of where `a` and `b` are equal within the
        /// given tolerance.If both `a` and `b` are scalars, returns a single
        /// boolean value.
        ///</returns>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the formula builds a dozen transient arrays (and up to two
        ///     dtype-promoted copies of the operands), and the woven scope releases every one of them on return,
        ///     exceptions included, so a tight isclose/allclose loop does not pile them up for the finalizer. The
        ///     returned typed array survives: the weaver yields it, and it holds its own reference on the buffer it
        ///     shares with the last temporary.
        /// </remarks>
        [NDScoped]
        public override NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b);   // an engine method computes plain: np.abs / np.isfinite below must not apply its matrix rule
            // NumPy implementation (from numpy/_core/numeric.py):
            // result = (less_equal(abs(x - y), atol + rtol * abs(y))
            //           & isfinite(y)
            //           | (x == y))
            // if equal_nan:
            //     result |= isnan(x) & isnan(y)

            // Compute in NumPy's exact dtype, not blanket float64. NumPy's isclose forces the
            // REFERENCE operand inexact via result_type(b, 1.0) and then evaluates the whole formula
            // in result_type(x, y): so (f32,f32)->float32, (f16,f16)->float16, (complex,complex)->
            // complex128, while bool + every integer -> float64. Casting both operands through
            // InexactPromote and letting the elementwise operators do the NEP50 promotion reproduces
            // that (mixed pairs too, verified bit-exact vs NumPy 2.4.2). The prior code cast BOTH to
            // Double unconditionally, which (a) did 2x the memory work NumPy does for float32/float16
            // — the isclose/allclose float32 perf gap — and (b) was WRONG for complex, dropping the
            // imaginary part (isclose([1+0j],[1+100j]) returned True where NumPy returns False).
            // float64/int/bool/char/decimal still map to Double, so their cast is byte-identical to
            // before. astype(copy:false) returns the input itself when no conversion is needed and a
            // FRESH array otherwise — it never mutates a/b (NumPy semantics). Ownership is structural
            // under the woven scope: a converted x/y is built inside it and released on return, while
            // an unconverted x/y IS the caller's a/b, built before the scope opened, which the scope
            // never tracks and so never disposes (no ReferenceEquals guard needed). Every other local
            // is a fresh allocation (each elementwise operator/ufunc returns a new array), released
            // on return the same way.
            var x = a.astype(InexactPromote(a.typecode), copy: false);
            var y = b.astype(InexactPromote(b.typecode), copy: false);

            // Vectorized computation using existing np operations
            var xMinusY = x - y;
            var diff = np.abs(xMinusY);            // |a - b|
            var absY = np.abs(y);
            var rtolAbsY = rtol * absY;
            var tolerance = atol + rtolAbsY;       // atol + rtol * |b|

            // Core formula: |a - b| <= tolerance AND diff is finite AND y is finite, OR exact equality
            // Note: We explicitly check diffFinite because NumSharp's <= operator has a bug where
            // NaN <= value returns True instead of False (IEEE 754 requires False for all NaN comparisons)
            var diffFinite = np.isfinite(diff);      // diff must be finite for tolerance check
            var withinTolerance = diff <= tolerance; // |a - b| <= (atol + rtol * |b|)
            var yFinite = np.isfinite(y);            // Only apply tolerance to finite values
            var exactEqual = x == y;                 // Handles infinities (inf == inf is true)

            // Combine: (within tolerance & diff finite & y finite) | exact equality
            var withinAndFinite = withinTolerance & diffFinite;
            var toleranceMet = withinAndFinite & yFinite;

            // MakeGeneric<bool>() returns a typed alias that takes its own reference on the combined
            // array's buffer: the weaver yields the alias, and releasing the backing temp with the rest
            // of the scope leaves the returned buffer alive.
            if (!equal_nan)
            {
                var close = toleranceMet | exactEqual;
                return close.MakeGeneric<bool>();
            }

            // Handle NaN comparison if requested: result | (isnan(x) & isnan(y))
            var toleranceClose = toleranceMet | exactEqual;
            var nanX = np.isnan(x);
            var nanY = np.isnan(y);
            var bothNan = nanX & nanY;
            var close2 = toleranceClose | bothNan;
            return close2.MakeGeneric<bool>();
        }

        /// <summary>
        /// NumPy's <c>result_type(dtype, 1.0)</c> — the "make the reference operand inexact" step of
        /// <c>np.isclose</c>. A float or complex dtype keeps itself; bool and every integer promote to
        /// float64. NumSharp's non-NumPy dtypes (Char/Decimal) also fall through to float64, preserving
        /// the pre-change behaviour for them. Casting BOTH operands through this (and letting the
        /// elementwise operators promote) reproduces isclose's exact computation dtype
        /// <c>result_type(x, result_type(y, 1.0))</c> — so float32/float16/complex compute at native
        /// precision instead of being widened to float64.
        /// </summary>
        private static NPTypeCode InexactPromote(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex => tc,
            _ => NPTypeCode.Double,
        };
    }
}
