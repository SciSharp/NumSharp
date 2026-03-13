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
        public override NDArray<bool> IsClose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8, bool equal_nan = false)
        {
            // NumPy implementation (from numpy/_core/numeric.py):
            // result = (less_equal(abs(x - y), atol + rtol * abs(y))
            //           & isfinite(y)
            //           | (x == y))
            // if equal_nan:
            //     result |= isnan(x) & isnan(y)

            // Convert to double for comparison (NumPy casts to inexact type)
            var x = a.astype(NPTypeCode.Double, copy: false);
            var y = b.astype(NPTypeCode.Double, copy: false);

            // Vectorized computation using existing np operations
            var diff = np.abs(x - y);                    // |a - b|
            var tolerance = atol + rtol * np.abs(y);     // atol + rtol * |b|

            // Core formula: |a - b| <= tolerance AND diff is finite AND y is finite, OR exact equality
            // Note: We explicitly check diffFinite because NumSharp's <= operator has a bug where
            // NaN <= value returns True instead of False (IEEE 754 requires False for all NaN comparisons)
            var diffFinite = np.isfinite(diff);          // diff must be finite for tolerance check
            var withinTolerance = diff <= tolerance;     // |a - b| <= (atol + rtol * |b|)
            var yFinite = np.isfinite(y);                // Only apply tolerance to finite values
            var exactEqual = x == y;                     // Handles infinities (inf == inf is true)

            // Combine: (within tolerance & diff finite & y finite) | exact equality
            NDArray<bool> result = ((withinTolerance & diffFinite & yFinite) | exactEqual).MakeGeneric<bool>();

            // Handle NaN comparison if requested
            if (equal_nan)
            {
                NDArray<bool> bothNan = (np.isnan(x) & np.isnan(y)).MakeGeneric<bool>();
                result = (result | bothNan).MakeGeneric<bool>();
            }

            return result;
        }
    }
}
