using System;
using System.Collections.Generic;
using System.Text;

namespace Numpy
{

    public static partial class np
    {

        /// <summary>
        /// Create an array.
        /// 
        /// <param name="data">
        /// The array to initialize the ndarray with
        /// </param>
        /// <returns>
        /// An array object satisfying the specified requirements.
        /// </returns>
        public static NDarray<T> array<T>(params T[] data)
        {
            return NumPy.Instance.array(data);
        }

        public static NDarray<T> array<T>(T[,,] data, Dtype dtype = null, bool? copy = null, string order = null, bool? subok = null, int? ndmin = null)
        {
            return NumPy.Instance.array(data, dtype, copy, order, subok, ndmin);
        }

        public static NDarray asarray(ValueType a, Dtype dtype = null)
            => NumPy.Instance.asarray(a, dtype: dtype);

        /// <summary>
        /// Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        /// Scalar representation of a. The output data type is the same type
        /// returned by the input’s item method.
        /// </returns>
        public static T asscalar<T>(NDarray a) => NumPy.Instance.asscalar<T>(a);

    }
}
