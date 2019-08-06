using System;
using System.Numerics;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
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
        public static NDArray<bool> isclose(NDArray a, NDArray b, double rtol = 1.0E-5, double atol = 1.0E-8,
            bool equal_nan = false)
            => a.TensorEngine.IsClose(a, b, rtol, atol, equal_nan);

        /// <summary>
        /// Test element-wise for finiteness (not infinity or not Not a Number).
        /// </summary>
        /// <param name="a"></param>
        /// <returns>The result is returned as a boolean array.</returns>
        public static NDArray<bool> isfinite(NDArray a)
            => a.TensorEngine.IsFinite(a);

        /// <summary>
        /// Test element-wise for Not a Number.
        /// </summary>
        /// <param name="a"></param>
        /// <returns>The result is returned as a boolean array.</returns>
        public static NDArray<bool> isnan(NDArray a)
            => a.TensorEngine.IsNan(a);

        /// <summary>
        ///     Returns true incase of a number, bool or string. If null, returns false.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.isscalar.html</remarks>
        public static bool isscalar(object obj)
        {
            switch (obj)
            {
                case null:
                    return false;
                case NDArray nd:
                    return nd.ndim == 0 && nd.size == 1;
                case Type _:
                    break;
                case Complex _:
                case string _:
                case bool _:
                    return true;
            }

            var type = obj as Type ?? obj.GetType();
            if (type.IsArray)
            {
                return false;
            }

            //type.IsPrimitive checks for: Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, IntPtr, UIntPtr, Char, Double, and Single.
            return type.IsPrimitive || obj is decimal;
        }
    }
}
