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
            //    if (a.size > b.size)
            //        throw new ArgumentException("Array a must not be larger in size than array b");
            //    var result = new NDArray<bool>(a.shape);
            //    bool[] rdata = result.Array;
            //    if (a.dtype == np.uint8 || a.dtype == np.int16 || a.dtype == np.int32 || a.dtype == np.int64 && b.dtype == typeof(double) || b.dtype == typeof(float))
            //    {
            //        //  convert both to double and compare
            //        double[] a_arr = a.Data<double>();
            //        double[] b_arr = a.Data<double>();
            //        for (int i = 0; i < a_arr.Length; i++)
            //            rdata[i] = is_within_tol(a_arr[i], b_arr[i], rtol, atol, equal_nan);
            //    }
            //    var adata = a.Array;
            //    switch (adata)
            //    {
            //        case double[] a_arr:
            //            {
            //                var b_arr = b.Data<double>();
            //                for (int i = 0; i < a_arr.Length; i++)
            //                    rdata[i] = is_within_tol(a_arr[i], b_arr[i], rtol, atol, equal_nan);
            //                break;
            //            }
            //        case float[] a_arr:
            //            {
            //                var b_arr = b.Data<float>();
            //                for (int i = 0; i < a_arr.Length; i++)
            //                    rdata[i] = is_within_tol(a_arr[i], b_arr[i], rtol, atol, equal_nan);
            //                break;
            //            }
            //        case Complex[] arr:
            //            {
            //                throw new NotImplementedException("Comparing Complex arrays is not implemented yet.");
            //            }
            //        default:
            //            {
            //                throw new IncorrectTypeException();
            //            }
            //    }
            //    return result;
            return null;
        }

        private static bool is_within_tol(object a_obj, object b_obj, double rtol = 1.0E-5, double atol = 1.0E-8,
            bool equal_nan = false)
        {
            return false;
        }

        //{
        //    switch (a_obj)
        //    {
        //        case double a:
        //            {
        //                var b = (double)b_obj;
        //                if (double.IsInfinity(a) && double.IsInfinity(b))
        //                    return true;
        //                if (equal_nan && double.IsNaN(a) && double.IsNaN(b))
        //                    return true;
        //                return Math.Abs(a - b) <= atol + rtol * Math.Abs(b);
        //            }
        //        case float a:
        //            {
        //                var b = (float)b_obj;
        //                if (float.IsInfinity(a) && float.IsInfinity(b))
        //                    return true;
        //                if (equal_nan && float.IsNaN(a) && float.IsNaN(b))
        //                    return true;
        //                return Math.Abs(a - b) <= atol + rtol * Math.Abs(b);
        //            }
        //    }
        //    throw new NotImplementedException($"Comparing type {a_obj.GetType()} not implemented");
        //}
    }
}
