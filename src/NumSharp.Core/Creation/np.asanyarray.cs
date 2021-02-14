using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">Input data, in any form that can be converted to an array. This includes scalars, lists, lists of tuples, tuples, tuples of tuples, tuples of lists, and ndarrays.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input data.</param>
        /// <returns>Array interpretation of a. If a is an ndarray or a subclass of ndarray, it is returned as-is and no copy is performed.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.asanyarray.html</remarks>
        public static NDArray asanyarray(in object a, Type dtype = null) //todo support order
        {
            NDArray ret;
            switch (a) {
                case null:
                    throw new ArgumentNullException(nameof(a));
                case NDArray nd:
                    return nd;
                case Array array:
                    ret = new NDArray(array);
                    break;
                case string str:
                    ret = (NDArray)str; //implicit cast located in NDArray.Implicit.Array
                    break;
                default:
                    var type = a.GetType();
                    //is it a scalar
                    if (type.IsPrimitive || type == typeof(decimal))
                    {
                        ret = NDArray.Scalar(a);
                        break;
                    }

                    throw new NotSupportedException($"Unable resolve asanyarray for type {a.GetType().Name}");
            }

            if (dtype != null && a.GetType() != dtype)
                return ret.astype(dtype, true);

            return ret;
        }
    }
}
