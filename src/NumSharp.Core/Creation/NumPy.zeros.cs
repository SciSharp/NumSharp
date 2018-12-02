using System;
using System.Collections.Generic;
using System.Text;

namespace NumSharp.Core
{
    public partial class NumPy
    {
        /// <summary>
        /// Return a new float array of given shape, filled with zeros.
        /// </summary>
        /// <param name="np"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        public NDArray zeros(params int[] shape)
        {
            var nd = new NDArray(float64, new Shape(shape));
            return nd;
        }

        public NDArray zeros<T>(params int[] shape)
        {
            var nd = new NDArray(typeof(T));
            nd.Storage.Shape = new Shape(shape);

            return nd;
        }

        public NDArray zeros(Shape shape, Type dtype = null)
        {
            return new NDArray(dtype == null? float64 : dtype, shape);
        }
    }
}
