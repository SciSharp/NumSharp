using System;
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
            // Broadcast arrays to common shape
            var (ba, bb) = np.broadcast_arrays(a, b);

            var result = new NDArray<bool>(ba.Shape, true);
            int size = ba.size;

            unsafe
            {
                var dst = (bool*)result.Address;

                // Convert both to double for comparison (NumPy behavior)
                for (int i = 0; i < size; i++)
                {
                    double aVal = ba.GetAtIndex<double>(i);
                    double bVal = bb.GetAtIndex<double>(i);
                    dst[i] = IsCloseValue(aVal, bVal, rtol, atol, equal_nan);
                }
            }

            return result.MakeGeneric<bool>();
        }

        private static bool IsCloseValue(double a, double b, double rtol, double atol, bool equal_nan)
        {
            // Handle NaN
            if (double.IsNaN(a) && double.IsNaN(b))
                return equal_nan;
            if (double.IsNaN(a) || double.IsNaN(b))
                return false;

            // Handle infinities - must be same sign
            if (double.IsInfinity(a) || double.IsInfinity(b))
                return a == b;

            // NumPy formula: |a - b| <= (atol + rtol * |b|)
            return Math.Abs(a - b) <= (atol + rtol * Math.Abs(b));
        }
    }
}
